"""Host-side NRPT tuners: schedule, chain count, and local-exploration count.

These orchestration loops drive the jitted core in ``hamon.nrpt`` (they call
``nrpt`` and reuse its compiled round loop) — none of the hot path lives here.
Split out of ``hamon.nrpt`` to keep that module focused on the compiled core.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hamon._tuning_host_schedule import (
    _acceptance_rate_host as _acceptance_rate_host,
    _optimize_schedule_host as _optimize_schedule_host,
    _pchip_interp_host as _pchip_interp_host,
    _pchip_slopes_host as _pchip_slopes_host,
    _phase_diagnostics_host as _phase_diagnostics_host,
    _pooled_lambda_host as _pooled_lambda_host,
)
from hamon.block_sampling import BlockSamplingProgram, SamplingSchedule
from hamon.device import (
    DeviceLike,
    default_device_ctx,
    enable_persistent_compile_cache,
    resolve_entry_device,
)
from hamon.models.ebm import AbstractEBM
from hamon.observers import AbstractNRPTObserver
from hamon.round_trips import barrier_is_identified, conveyor_is_alive
from hamon.nrpt import (
    _ChainSource,
    nrpt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Iterative schedule tuning
# ---------------------------------------------------------------------------


# Default "settled" floor for max|Δβ| (β units): a well-tuned ladder keeps
# jittering by about this much, so a tighter value would never be reached.
_DEFAULT_TUNE_TOL = 0.02


def _require_proper_beta_start(beta0: float, ebm) -> None:
    """Fail fast when a ladder would start at β = 0 for a model with no proper
    β = 0 member (continuous/unbounded state spaces — see
    ``AbstractEBM.proper_at_beta_zero``). Only checkable on the template route
    (``ebm`` given); the factory route is caught at run time by ``nrpt``."""
    if (
        ebm is not None
        and float(beta0) == 0.0
        and not getattr(ebm, "proper_at_beta_zero", True)
    ):
        raise ValueError(
            f"{type(ebm).__name__} is not proper at beta=0 (unbounded state "
            "space), but beta_range starts at exactly 0. Pass "
            "beta_range=(beta_min > 0, beta_max)."
        )


def _tune_phase_adaptive_rounds(
    run_phase,
    key,
    betas,
    states,
    *,
    round_batch,
    min_rounds,
    max_rounds,
    lambda_rtol,
    stable_k,
):
    """Run one schedule-tuning phase for an adaptive number of rounds.

    Instead of a fixed ``rounds_per_tune``, run ``run_phase`` in fixed-size
    ``round_batch`` batches (each a separate ``nrpt`` call that reuses the same
    compiled round loop — see the BlockSpec value-equality cache), threading the
    chain ``states`` forward and pooling the integer ``accepted``/``attempted``
    swap counters across batches. Stop once the pooled barrier estimate
    ``Λ = sum(rejection_rates)`` is stable — ``|ΔΛ|/Λ < lambda_rtol`` for
    ``stable_k`` consecutive batches — past a ``min_rounds`` floor, or at the
    ``max_rounds`` ceiling. Pooling reduces the estimate's Monte-Carlo variance
    (var ~ r(1-r)/n_attempts), so the returned ``rejection_rates`` give
    ``optimize_schedule`` a clean signal rather than a noisy single-batch one.

    Round-trip continuity is intentionally not preserved across batches (each
    ``nrpt`` call resets the index process); tuning only needs rejection rates.
    The single continuous production run keeps round-trip tracking.

    **Returns** ``(states, pooled_stats, rounds_used)`` where ``pooled_stats``
    mirrors an ``nrpt`` stats dict over the pooled counters.
    """
    acc_total = None
    att_total = None
    rounds_used = 0
    lambda_prev = None
    stable_count = 0

    # Always run at least one batch; break on stability or the round ceiling.
    while True:
        key, subkey = jax.random.split(key)
        batch = min(round_batch, max_rounds - rounds_used)
        # Padded states stay padded across batches: re-slicing and re-padding
        # every batch would compile a slice/pad kernel pair per live N.
        states, stats = run_phase(
            subkey, betas, states, batch, return_stacked=True, keep_padded_states=True
        )
        acc_total = (
            stats["accepted"] if acc_total is None else acc_total + stats["accepted"]
        )
        att_total = (
            stats["attempted"] if att_total is None else att_total + stats["attempted"]
        )
        rounds_used += batch

        lambda_cur = _pooled_lambda_host(acc_total, att_total, betas.dtype)
        if rounds_used >= min_rounds and lambda_prev is not None:
            rel = abs(lambda_cur - lambda_prev) / max(lambda_cur, 1e-9)
            stable_count = stable_count + 1 if rel < lambda_rtol else 0
            if stable_count >= stable_k:
                break
        lambda_prev = lambda_cur
        if rounds_used >= max_rounds:
            break

    assert acc_total is not None and att_total is not None
    acceptance_rate = _acceptance_rate_host(acc_total, att_total, betas.dtype)
    pooled_stats = {
        "accepted": acc_total,
        "attempted": att_total,
        "acceptance_rate": acceptance_rate,
        "rejection_rates": np.asarray(1, dtype=betas.dtype) - acceptance_rate,
        "betas": np.asarray(betas),
    }
    return states, pooled_stats, rounds_used


def _finalize_schedule_stats(
    stats: dict, betas: jax.Array | np.ndarray, n_rounds: int
) -> None:
    """Attach the two health verdicts to a finished ``tune_schedule`` run.

    Post-hoc reporting only — reads the production stats and writes
    ``barrier_identified`` and ``conveyor_alive`` back into ``stats`` in place.
    Kept separate from the phase loop so "run the tuning" and "explain the
    result" are not interleaved, and kept in this module so its log records
    still carry the ``hamon.tuning`` logger name that callers capture on.

    The two verdicts are independent: ``barrier_identified`` is structural (did
    the ladder saturate?), ``conveyor_alive`` is dynamical (did the index
    process actually round-trip?), and the latter is ``None`` when the window
    was too short to tell — which must never be reported as a stall.
    """
    # Gate on the PRODUCTION run's rates (the kept ladder Lambda comes from) —
    # the loop-local `rej` belongs to a schedule that may not have been kept.
    prod_rej = stats["rejection_rates"]
    resolved = barrier_is_identified(prod_rej)
    stats["barrier_identified"] = resolved
    if not resolved:
        logger.warning(
            "tune_schedule: barrier NOT resolved — the ladder saturates "
            "(max rejection=%.3f), so Lambda=%.2f is capped by the chain count "
            "(Lambda <= N-1 = %d) rather than measuring the barrier; it is an "
            "underestimate. Add chains or equalize the ladder.",
            float(np.asarray(prod_rej).max()),
            float(np.asarray(prod_rej).sum()),
            int(betas.shape[0]) - 1,
        )

    rtd = stats.get("round_trip_diagnostics")
    if rtd is None:
        return
    n_rt = int(np.sum(np.asarray(rtd["round_trips_per_chain"])))
    alive = conveyor_is_alive(
        float(rtd["tau_observed"]), float(rtd["tau_predicted"]), n_rounds
    )
    stats["conveyor_alive"] = alive
    if alive is None:
        logger.info(
            "tune_schedule: round-trip rate not measured — n_rounds=%d "
            "affords only %.1f expected trips at the optimal rate "
            "(tau_pred=%.4f), too few to distinguish a slow conveyor from an "
            "unlucky window. Lambda=%.2f stands on its own (resolved=%s).",
            n_rounds,
            float(rtd["tau_predicted"]) * n_rounds,
            float(rtd["tau_predicted"]),
            float(rtd["Lambda"]),
            resolved,
        )
    elif not alive:
        logger.warning(
            "tune_schedule: DEO conveyor is slow — %d round trips, "
            "efficiency=%.3f of the optimal rate over %d rounds. Samples "
            "decorrelate slowly even though Lambda=%.2f is resolved=%s.",
            n_rt,
            float(rtd["efficiency"]),
            n_rounds,
            float(rtd["Lambda"]),
            resolved,
        )


def tune_schedule(
    key: jax.Array,
    ebm_factory: Callable | None = None,
    program_factory: Callable | None = None,
    init_states: Sequence[list] = (),
    clamp_state: list | None = None,
    n_rounds: int = 0,
    gibbs_steps_per_round: int = 0,
    initial_betas: jax.Array | np.ndarray | None = None,
    n_tune: int = 5,
    rounds_per_tune: int = 200,
    track_round_trips: bool = True,
    *,
    ebm: AbstractEBM | None = None,
    program: BlockSamplingProgram | None = None,
    observer: AbstractNRPTObserver | None = None,
    adaptive_tuning: bool = True,
    tune_tol: float | None = None,
    equalize_tol: float = 0.05,
    phase_patience: int = 2,
    min_tune_phases: int = 1,
    round_batch: int = 50,
    min_rounds_per_tune: int = 50,
    round_stable_k: int = 2,
    lambda_rtol: float = 0.05,
    lambda_plateau_rtol: float = 0.05,
    saturation_tol: float = 0.99,
    device: DeviceLike = "auto",
    pad_chains_to: int | None = None,
    _return_stacked: bool = False,
) -> tuple[list, dict]:
    """NRPT with iterative schedule optimization (Algorithm 4).

    ``pad_chains_to`` enables chain masking in every phase's round loop (see
    :func:`hamon.nrpt.nrpt`): phases at different chain counts padded to the
    same length share one compiled executable. Stats and states are sliced
    back to the true count before any tuning math sees them, so the schedule
    optimization is untouched. Incompatible with ``observer``.

    Adapts the β schedule over tuning phases, then runs the final ``n_rounds``
    production phase with the optimized schedule. Each phase logs one INFO line
    (Λ, mean acceptance, schedule movement) so long runs are not silent.

    Instead of providing ``ebm_factory`` and ``program_factory``, you can pass
    a template ``ebm`` and ``program`` and the factories will be built
    internally using ``ebm.with_beta()`` and ``program.with_ebm()``.

    **Convergence-driven tuning (default, ``adaptive_tuning=True``):** budgets
    are chosen automatically, so callers need not guess them. Each phase runs as
    many rounds as needed for the Λ estimate to settle (between
    ``min_rounds_per_tune`` and the ``rounds_per_tune`` ceiling, in
    ``round_batch`` increments — see ``_tune_phase_adaptive_rounds``), giving a
    low-variance rejection-rate estimate. The schedule with the **best**
    rejection rates seen across phases is kept for production — not the last,
    which can be noisier. "Best" ranks an **unsaturated** ladder (no pair at
    ``max(rej) >= saturation_tol``, i.e. none pinned at ~100% rejection) above
    any saturated one, then by lowest spread: a saturated pair means an
    unbridged gap at β_c, so Λ̂ across it is a within-basin artifact, and its
    spread can score *better* than the partially-retuned ladders of the tuning
    transient — ranking on spread alone reverts to the initial schedule and
    makes tuning a no-op on glassy targets.

    Tuning stops once the schedule is well-equalized
    (``std(rejection_rates) < equalize_tol``) OR has settled (``max|Δβ|`` below
    the effective ``tune_tol`` — its Monte-Carlo floor) OR Λ̂ has **plateaued**
    (relative change ``<= lambda_plateau_rtol`` on an unsaturated ladder), for
    ``phase_patience`` consecutive phases, after at least ``min_tune_phases``,
    capped at ``n_tune``. Neither of the first two alone is reliable: ``max|Δβ|``
    plateaus at a problem-dependent noise floor rather than going to zero, and on
    a glassy target ``std(rejection_rates)`` settles just *above* a useful
    ``equalize_tol`` — so without the Λ-plateau check the cap silently decides,
    which is the failure the plateau stop exists to prevent. ``n_tune`` is a
    backstop, not the intended stopping rule. When ``tune_tol`` is left ``None``
    it defaults to ``_DEFAULT_TUNE_TOL`` here. Counts are deterministic for a
    given seed but problem-dependent — do not assume a fixed round/phase count.

    **Legacy mode (``adaptive_tuning=False``):** runs exactly ``n_tune`` phases
    of exactly ``rounds_per_tune`` rounds and uses the last schedule.
    ``tune_tol`` then behaves as the optional early-stop it always was (``None``
    ⇒ run all ``n_tune`` phases). ``n_tune`` and ``rounds_per_tune`` act as
    safety caps in both modes.

    Returns ``(states, stats)`` where stats includes tuning history in
    ``stats["tuning_history"]`` (each entry records ``max_beta_shift``).
    States are ordered by ascending β — the **cold chain** (target
    distribution) is ``states[-1]``.

    ``device`` is resolved **once** here and passed to every tuning and
    production phase, so the device never flips mid-run (see ``hamon.device``).
    """
    if clamp_state is None:
        clamp_state = []
    if initial_betas is None:
        raise ValueError("initial_betas is required.")
    if not init_states:
        raise ValueError("init_states is required (one initial state per chain).")

    # _ChainSource hides the template-vs-factory difference from every phase.
    source = _ChainSource(ebm_factory, program_factory, ebm, program)

    # Resolve the device once for all phases — a flip between phases would
    # recompile the round loop and shuttle states across devices.
    dev = resolve_entry_device(
        device,
        n_chains=len(initial_betas),
        n_nodes=source.metadata_free_nodes(initial_betas, device),
        arrays=(init_states, initial_betas, key),
    )
    source.device_put_template(dev)

    def _run_phase(
        phase_key,
        phase_betas,
        phase_states,
        rounds,
        phase_observer=None,
        emit_diag=False,
        return_stacked=False,
        keep_padded_states=False,
    ):
        # Tuning batches skip the eager round-trip summary and per-call unstack
        # they never read; only the production run pays for the public return.
        chain_ebms, chain_programs = source.nrpt_args(phase_betas)
        return nrpt(
            phase_key,
            chain_ebms,
            chain_programs,
            phase_states,
            clamp_state,
            rounds,
            gibbs_steps_per_round,
            betas=phase_betas,
            track_round_trips=track_round_trips,
            observer=phase_observer,
            device=dev,
            pad_chains_to=pad_chains_to,
            _emit_diagnostics=emit_diag,
            _return_stacked=return_stacked,
            # Discovery probes (`_return_stacked`, from tune_chains) consume
            # even the production stats in Python, so those go host-side too;
            # the public tune_schedule production stats stay JAX-native.
            _host_stats=not emit_diag or _return_stacked,
            _keep_padded_states=keep_padded_states,
        )

    # Commit betas to the device up front: jit keys on commitment, so phase 0's
    # uncommitted initial_betas would build a second single-use executable.
    betas = jax.device_put(initial_betas, dev) if dev is not None else initial_betas
    current_states = init_states
    tuning_history = []

    # In adaptive mode an unset tune_tol means "use the default β-movement
    # floor"; in legacy mode an unset tune_tol means "never early-stop".
    effective_tol = (
        (tune_tol if tune_tol is not None else _DEFAULT_TUNE_TOL)
        if adaptive_tuning
        else tune_tol
    )

    # best_saturated starts True so the first unsaturated ladder displaces the
    # initial one, whose rej_std can look deceptively good (see the docstring).
    best_betas = betas
    best_quality = float("inf")
    best_saturated = True
    prev_lambda: float | None = None
    stop_streak = 0

    phase = 0
    while phase < n_tune:
        key, subkey = jax.random.split(key)
        if adaptive_tuning:
            current_states, stats, rounds_used = _tune_phase_adaptive_rounds(
                _run_phase,
                subkey,
                betas,
                current_states,
                round_batch=round_batch,
                min_rounds=min_rounds_per_tune,
                max_rounds=rounds_per_tune,
                lambda_rtol=lambda_rtol,
                stable_k=round_stable_k,
            )
        else:
            current_states, stats = _run_phase(
                subkey,
                betas,
                current_states,
                rounds_per_tune,
                return_stacked=True,
                keep_padded_states=True,
            )
            rounds_used = rounds_per_tune

        rej = stats["rejection_rates"]
        old_betas = betas
        new_betas = _optimize_schedule_host(rej, betas)
        # rej_std depends only on the pre-update rates, so keep-best records
        # old_betas.
        rej_std_a, shift_a, lambda_a, mean_acc_a, max_rej_a = _phase_diagnostics_host(
            rej, old_betas, new_betas, stats["acceptance_rate"]
        )
        quality = float(rej_std_a)
        max_rej = float(max_rej_a)
        # Unsaturated outranks saturated before spread is compared (docstring,
        # "Best") — otherwise keep-best reverts to the untuned input on glass.
        saturated = max_rej >= saturation_tol
        if adaptive_tuning and (
            (best_saturated and not saturated)
            or (saturated == best_saturated and quality < best_quality)
        ):
            best_quality = quality
            best_saturated = saturated
            best_betas = old_betas
        betas = new_betas

        max_beta_shift = float(shift_a)
        phase_lambda = float(lambda_a)
        tuning_history.append(
            {
                "iteration": phase,
                "betas": old_betas,
                "rejection_rates": rej,
                "acceptance_rate": stats["acceptance_rate"],
                "Lambda": phase_lambda,
                "max_beta_shift": max_beta_shift,
                "rej_std": quality,
                "rounds_used": rounds_used,
            }
        )
        phase += 1
        logger.info(
            "tune_schedule tune %d/%d: Lambda=%.3f mean_acceptance=%.3f rej_std=%.4g max|dbeta|=%.4g rounds=%d",
            phase,
            n_tune,
            phase_lambda,
            float(mean_acc_a),
            quality,
            max_beta_shift,
            rounds_used,
        )

        if adaptive_tuning:
            # Stop rules per the docstring ("Tuning stops"); the Λ̂ plateau is
            # what keeps the phase cap from silently deciding on glassy targets.
            equalized = quality < equalize_tol
            settled = effective_tol is not None and max_beta_shift < effective_tol
            plateaued = (
                prev_lambda is not None
                and not saturated
                and abs(phase_lambda - prev_lambda)
                <= lambda_plateau_rtol * max(abs(phase_lambda), 1e-9)
            )
            prev_lambda = phase_lambda
            stop_streak = stop_streak + 1 if (equalized or settled or plateaued) else 0
            if phase >= min_tune_phases and stop_streak >= phase_patience:
                logger.info(
                    "tune_schedule: schedule converged after %d phase(s) "
                    "(rej_std=%.4g, max|dbeta|=%.4g); skipping remaining tuning",
                    phase,
                    quality,
                    max_beta_shift,
                )
                break
        elif effective_tol is not None and max_beta_shift < effective_tol:
            # Legacy early-stop (unchanged semantics).
            logger.info(
                "tune_schedule: schedule converged after %d phase(s) "
                "(max|dbeta|=%.4g < tune_tol=%.4g); skipping remaining tuning",
                phase,
                max_beta_shift,
                effective_tol,
            )
            break

    if adaptive_tuning:
        betas = best_betas

    # Production run — emit the full round-trip diagnostics (tuning batches skip
    # them).
    key, subkey = jax.random.split(key)
    states, stats = _run_phase(
        subkey,
        betas,
        current_states,
        n_rounds,
        phase_observer=observer,
        emit_diag=True,
        return_stacked=_return_stacked,
        keep_padded_states=_return_stacked,
    )
    stats["tuning_history"] = tuning_history

    _finalize_schedule_stats(stats, betas, n_rounds)
    return states, stats


# ---------------------------------------------------------------------------
# Energy-variance barrier pre-estimate
# ---------------------------------------------------------------------------
# Syed et al. (2021), Theorem 2. Seeds the chain-count search below so it
# converges in one probe instead of two; also reused by the plain block-Gibbs
# calibration in the last section, which shares the adaptive-warmup machinery.


_ENERGY_GRID = 11
_ENERGY_SAMPLES = 500
_ENERGY_RESTARTS = 8  # independent chains per β (coverage + the R̂ trap detector)
_ENERGY_RHAT_MAX = 1.1  # Gelman–Rubin cutoff: above this, trust the PT pilot

# Adaptive-warmup knobs — see _adaptive_warmup for the stable/plateau/cap
# stopping rules these feed.
_ENERGY_WARMUP_BATCH = 50  # sweeps per warmup batch (one compiled kernel)
_ENERGY_WARMUP_MIN = 200  # earliest sweep count at which stopping is checked
_ENERGY_WARMUP_MAX = 2000  # hard ceiling
_ENERGY_WARMUP_WINDOW = 4  # batch-end snapshots per restart in the running R̂
_ENERGY_WARMUP_PASSES = 2  # consecutive R̂ passes required (guards noisy R̂)
# ~p95 of the null window-R̂ at (R=8, W=4, 10 informative β) — looser than
# _ENERGY_RHAT_MAX because a 4-snapshot R̂ is noisy even for converged chains;
# recalibrate if the window/restart/grid counts change.
_ENERGY_WARMUP_RHAT = 1.45
_ENERGY_PLATEAU_K = 3  # consecutive no-improvement checks ⇒ trapped, stop
_ENERGY_PLATEAU_TOL = 0.02  # relative R̂ improvement that resets the plateau
_ENERGY_SAMPLING_TAG = 0x53414D50  # "SAMP": fold_in tag for the sampling keys


@eqx.filter_jit
def _grid_sweep(keys, base_prog, sched, states, clamp_state, stacked_pbi, f_obs, carry):
    """One vmapped sweep over every (β, restart) lane of the energy grid.

    Each lane runs ``_sample_with_observation_core`` with its own β-tempered
    interaction tensors (stacked and vmapped, as in nrpt's per-chain-sequence
    mode). Serves both warmup batches (``sched = (B, 1, 1)`` — returns the
    state after B sweeps) and the recording pass (``sched = (0, ns, 1)``).
    Module-level so all batches and repeat calls share one traced kernel per
    schedule.
    """
    from hamon.block_sampling import _sample_with_observation_core
    from hamon.nrpt import _make_pbi_in_axes

    nb = len(states)
    pbi_axes = _make_pbi_in_axes(stacked_pbi)

    def _lane(k, init_free, pbi_c):
        prog_c = eqx.tree_at(lambda p: p.per_block_interactions, base_prog, pbi_c)
        _, res = _sample_with_observation_core(
            k, prog_c, sched, init_free, clamp_state, carry, f_obs
        )
        return res

    return jax.vmap(_lane, in_axes=(0, [0] * nb, pbi_axes))(keys, states, stacked_pbi)


def _window_rhat_max(e_window: np.ndarray, betas: np.ndarray) -> float:
    """Max over β>0 of the Gelman–Rubin R̂ of batch-end energy snapshots.

    ``e_window``: (W, G, R) — W batch-end snapshots for R restarts at G grid
    points. Same estimator as the final trust gate, with the recorded-sample
    window replaced by the batch-end window.
    """
    W = e_window.shape[0]
    rhs = []
    for i, b in enumerate(betas):
        if b <= 0.0:  # β=0 mixes trivially; its R̂ is uninformative
            continue
        v = e_window[:, i, :]  # (W, R)
        within = float(v.var(axis=0).mean())
        between = W * float(v.mean(axis=0).var(ddof=1))
        rhs.append(np.sqrt(((W - 1) / W * within + between / W) / max(within, 1e-12)))
    return float(max(rhs, default=1.0))


@eqx.filter_jit
def _grid_base_energies(base_ebm, beta_ref, spec, stacked, clamp_state):
    """Base potential V for energy-grid samples.

    Module-level (not a per-call closure) so repeat ``_estimate_barrier_energy``
    calls share the tracing cache; evaluated per β at the sequential path's
    (R·ns) batch shape — XLA's reduction strategy (and therefore the
    floating-point summation order of the energy) can depend on the batch
    size, so evaluating all G·R·ns rows at once would perturb V in the last
    bits and break the bit-identity of Λ̂.
    """
    from hamon._nrpt_energy import _compute_base_energies

    return _compute_base_energies(base_ebm, beta_ref, spec, stacked, clamp_state)


def _energy_mad_halved(v: np.ndarray) -> float:
    """½·E|V₁−V₂| of a 1-D energy sample, via the sorted-Gini formula (O(M log M))."""
    v = np.sort(np.asarray(v))
    M = v.size
    return float((1.0 / M**2) * np.sum((2 * np.arange(1, M + 1) - M - 1) * v))


def _adaptive_warmup(
    states, run_batch, batch_energies, *, cap, rhat_threshold, rhat_betas, log_label
):
    """Batched warmup with the windowed cross-replica R̂ stop.

    Runs ``run_batch(states, batch_index) -> states`` in
    ``_ENERGY_WARMUP_BATCH``-sweep increments, tracking the windowed R̂ of
    ``batch_energies(states)`` snapshots, and stops on the earliest of
    **stable** (R̂ < threshold for ``_ENERGY_WARMUP_PASSES`` checks),
    **plateau** (R̂ stopped improving while failing — trapped replicas), or the
    ``cap``. Shared by the energy-grid barrier seed and
    :func:`tune_sampling_schedule`.

    Returns ``(states, total_sweeps, exit_reason, last_rhat)``.
    """
    cap = max(int(cap), _ENERGY_WARMUP_BATCH)
    check_from = min(_ENERGY_WARMUP_MIN, cap)
    e_hist: list[np.ndarray] = []
    rh = float("nan")
    rh_best, rh_passes, rh_stalls = float("inf"), 0, 0
    total, exit_reason = 0, "cap"
    while total < cap:
        states = run_batch(states, total // _ENERGY_WARMUP_BATCH)
        total += _ENERGY_WARMUP_BATCH
        e_hist.append(batch_energies(states))
        if total < check_from or len(e_hist) < _ENERGY_WARMUP_WINDOW:
            continue
        rh = _window_rhat_max(np.stack(e_hist[-_ENERGY_WARMUP_WINDOW:]), rhat_betas)
        logger.debug("%s warmup: %d sweeps, window rhat=%.3f", log_label, total, rh)
        rh_passes = rh_passes + 1 if rh < rhat_threshold else 0
        if rh_passes >= _ENERGY_WARMUP_PASSES:
            exit_reason = "stable"
            break
        if rh < rh_best * (1.0 - _ENERGY_PLATEAU_TOL):
            rh_best, rh_stalls = rh, 0
        else:
            rh_stalls += 1
            if rh_stalls >= _ENERGY_PLATEAU_K:
                exit_reason = "plateau"
                break
    return states, total, exit_reason, rh


def _estimate_barrier_energy(
    key: jax.Array,
    source: _ChainSource,
    init_factory: Callable,
    clamp_state: list,
    beta_range: tuple[float, float],
    dev,
    *,
    n_grid: int = _ENERGY_GRID,
    warmup: int = _ENERGY_WARMUP_MAX,
    n_samples: int = _ENERGY_SAMPLES,
    restarts: int = _ENERGY_RESTARTS,
) -> tuple[float, float]:
    """Estimate Λ from energy samples (no PT) plus a trapping diagnostic.

    Theorem 2 of Syed et al. gives the local barrier in closed form,
    ``λ(β) = ½·E|V₁−V₂|`` where ``V`` is the base potential and ``V₁,V₂`` are
    independent draws from ``π^(β)``; ``Λ = ∫₀¹ λ(β) dβ``. We draw from ``π^(β)``
    with ``restarts`` independent block-Gibbs chains (random inits) at a β grid —
    **local exploration only, no DEO ladder** — so this compiles just the (cheap)
    Gibbs kernel (reused across the grid via the structure cache), never the round
    loop. ``V`` is recovered exactly via the same ``_compute_base_energies`` path
    the swap step uses, so the estimate is on the same scale as ``Σ rejection``.

    Theorem 2 is exact *given equilibrium samples*; the bias on a glassy target
    is a sampling artifact — local Gibbs traps in a basin and misses the
    inter-mode energy spread. The independent restarts expose exactly that: the
    returned **Gelman–Rubin R̂** (max over β) measures whether independent
    starts converge to the same distribution. R̂ ≈ 1 ⇒ local Gibbs mixes ⇒ the
    seed is trustworthy; R̂ ≫ 1 ⇒ glassy ⇒ the caller should fall back to the
    robust ``max_chains`` pilot.

    Warmup is adaptive: ``_ENERGY_WARMUP_BATCH``-sweep batches, stopping on the
    earliest of **stable** (cross-restart window R̂ passes), **plateau** (R̂
    stopped improving while failing — trapped restarts that more local sweeps
    cannot merge), or the ``warmup`` **cap** — see the ``_ENERGY_WARMUP_*``
    constants. Stability is only ever declared on *cross-restart agreement* (a
    single trapped chain looks stable), and the post-hoc R̂ gate on the recorded
    window remains the arbiter.

    Deliberately **not** merged with ``tune_sampling_schedule`` despite the
    similar shape: the two run different lane geometries (``n_grid × restarts``
    here, ``R`` replicas there), derive their keys differently, and sweep
    through separately jitted kernels (``_grid_sweep`` rebinds per-lane
    interactions, ``_replica_sweep`` shares one program), so a shared driver
    would merge two jit caches and change one path's key stream. Only the
    warmup machinery is shared, and that is already factored out into
    ``_adaptive_warmup``.

    Returns ``(Λ̂, R̂_max)``.
    """
    from hamon._nrpt_energy import _make_reference_ebm
    from hamon.device import tree_device_put
    from hamon.nrpt import _stack_pbi_across_chains
    from hamon.observers import StateObserver

    betas = np.linspace(beta_range[0], beta_range[1], int(n_grid))
    G, R, ns = len(betas), max(2, int(restarts)), int(n_samples)
    sched_batch = SamplingSchedule(_ENERGY_WARMUP_BATCH, 1, 1)
    sched_sample = SamplingSchedule(0, ns, 1)

    # The β = 1 base EBM and spec are grid-invariant; build once so the
    # base-energy kernel compiles a single time.
    one = jnp.asarray(1.0)
    base_ebm, beta_ref = _make_reference_ebm(source.ebms_for_init(one[None]), one[None])

    # One vmapped call over all (β, restart) lanes per pass; the per-lane key
    # layout — lane i·R+c owns split(fold_in(key, i))[c], batch b folds in b,
    # the recording pass folds in _ENERGY_SAMPLING_TAG — keeps streams
    # deterministic per seed.
    keys_flat = jnp.concatenate(
        [jax.random.split(jax.random.fold_in(key, i), R) for i in range(G)]
    )
    ebms_0 = source.ebms_for_init(jnp.full((R,), float(betas[0])))
    programs_0 = source._make_programs(ebms_0)
    base_prog = programs_0[0]
    spec = base_prog.gibbs_spec
    nb = len(spec.free_blocks)
    init_cols: list[list] = [[] for _ in range(nb)]
    lane_pbi: list = []  # per-lane per_block_interactions, in lane order
    for i, b in enumerate(betas):
        if i == 0:
            ebms_R, programs_R = ebms_0, programs_0
        else:
            betas_R = jnp.full((R,), float(b))
            ebms_R = source.ebms_for_init(betas_R)
            programs_R = source._make_programs(ebms_R)  # with_ebm — cheap, cached
        inits = init_factory(R, ebms_R, programs_R)  # R distinct random inits
        # Factories may return per-chain lists or the stacked (R, ...) form.
        if inits and not isinstance(inits[0], (list, tuple)):
            for blk in range(nb):
                init_cols[blk].append(inits[blk])
        else:
            for blk in range(nb):
                init_cols[blk].append(jnp.stack([inits[c][blk] for c in range(R)]))
        lane_pbi.extend([programs_R[0].per_block_interactions] * R)
    init_flat = [jnp.concatenate(cols) for cols in init_cols]
    stacked_pbi = [
        [
            _stack_pbi_across_chains([lane_pbi[lane][b][g] for lane in range(G * R)])
            for g in range(len(base_prog.per_block_interactions[b]))
        ]
        for b in range(nb)
    ]

    if dev is not None:
        keys_flat, base_prog, init_flat, stacked_pbi, clamp_dev = tree_device_put(
            (keys_flat, base_prog, init_flat, stacked_pbi, clamp_state), dev
        )
    else:
        clamp_dev = clamp_state
    device_ctx = default_device_ctx(dev)

    f_obs = StateObserver(spec.free_blocks)
    carry = f_obs.init()
    fold_all = jax.vmap(jax.random.fold_in, in_axes=(0, None))

    with device_ctx:

        def run_batch(states, batch_index):
            out = _grid_sweep(
                fold_all(keys_flat, batch_index),
                base_prog,
                sched_batch,
                states,
                clamp_dev,
                stacked_pbi,
                f_obs,
                carry,
            )
            return [o[:, 0] for o in out]

        def batch_energies(states):
            return np.asarray(
                _grid_base_energies(base_ebm, beta_ref, spec, states, clamp_state)
            ).reshape(G, R)

        states, total, exit_reason, _ = _adaptive_warmup(
            init_flat,
            run_batch,
            batch_energies,
            cap=warmup,
            rhat_threshold=_ENERGY_WARMUP_RHAT,
            rhat_betas=betas,
            log_label="energy grid",
        )
        logger.debug("energy grid warmup: %d sweeps (%s exit)", total, exit_reason)
        raw = _grid_sweep(
            fold_all(keys_flat, _ENERGY_SAMPLING_TAG),
            base_prog,
            sched_sample,
            states,
            clamp_dev,
            stacked_pbi,
            f_obs,
            carry,
        )  # list per free block, each (G*R, ns, …)

    lambdas, rhats = [], []
    for i, b in enumerate(betas):
        flat = [
            blk[i * R : (i + 1) * R].reshape((R * ns, *blk.shape[2:])) for blk in raw
        ]
        V = np.asarray(
            _grid_base_energies(base_ebm, beta_ref, spec, flat, clamp_state)
        ).reshape(R, ns)
        lambdas.append(float(np.mean([_energy_mad_halved(V[c]) for c in range(R)])))
        if b > 0.0:  # β=0 mixes trivially; its R̂ is uninformative
            within = float(V.var(axis=1).mean())
            between = ns * float(V.mean(axis=1).var(ddof=1))
            rhats.append(
                np.sqrt(((ns - 1) / ns * within + between / ns) / max(within, 1e-12))
            )
    return float(np.trapezoid(np.asarray(lambdas), betas)), float(
        max(rhats, default=1.0)
    )


# ---------------------------------------------------------------------------
# Iterative chain count discovery
# ---------------------------------------------------------------------------


def _probe_history_entry(
    iteration: int,
    n: int,
    lambda_raw: float,
    lambda_max: float,
    n_recommended: int,
    rejection_rates,
    betas,
    barrier_identified: bool | None = None,
) -> dict[str, Any]:
    """One ``tune_chains`` per-probe history record."""
    return {
        "iteration": iteration,
        "n": int(n),
        "Lambda_raw": float(lambda_raw),
        "Lambda_max": float(lambda_max),
        "n_recommended": int(n_recommended),
        "rejection_rates": rejection_rates,
        "betas": betas,
        "barrier_identified": barrier_identified,
    }


# Probe production-window floor, in units of the minimum DEO traversal
# 2*(n-1): a shorter window cannot observe the conveyor at all, and ~5-6x is
# what gives a high-N pilot a measurable round-trip rate.
_PROBE_MIN_RT_ROUNDS_FACTOR = 6


def tune_chains(
    key: jax.Array,
    ebm_factory: Callable | None = None,
    program_factory: Callable | None = None,
    init_factory: Callable | None = None,
    clamp_state: list | None = None,
    beta_range: tuple[float, float] = (0.0, 1.0),
    gibbs_steps_per_round: int = 0,
    initial_n: int | None = None,
    seed_from_energy: bool = True,
    target_acceptance: float = 0.5,
    rounds_per_probe: int = 200,
    n_tune_per_probe: int = 16,
    max_iters: int = 6,
    min_chains: int = 3,
    max_chains: int = 128,
    lambda_rtol: float = 0.05,
    *,
    ebm: AbstractEBM | None = None,
    program: BlockSamplingProgram | None = None,
    tune_tol: float | None = None,
    safety_margin: float = 0.05,
    device: DeviceLike = "auto",
    pad_probes: bool = False,
    compile_cache: bool | str = True,
) -> dict:
    """Iteratively discover the right chain count for a given target acceptance.

    Follows the N-tuning method of Syed et al. (2021): the global communication
    barrier Λ is a schedule invariant (Σ rejection_rates ≈ Λ at any chain count),
    so it is estimated at a single fixed N from a schedule-tuned run rather than
    searched for by probing many chain counts.

    1. Estimate Λ̂ = Σ rejection_rates at the current N (each probe runs
       ``tune_schedule``, which tunes the schedule toward equi-acceptance).
    2. Recommend N* = ceil(Λ̂·(1 + safety_margin) / r_target) + 1 — the
       round-trip-optimal 2Λ + 1 chains at r* = 1/2 (target_acceptance = 0.5) —
       using the running **max** of Λ̂ over probes (under-resolution can only
       bias Λ̂ low, never high, so the max is the least-biased estimate).
    3. Iterate this fixed point (re-estimate Λ̂ at N*) until N* stops moving.

    With no ``initial_n`` the first probe runs a **high** pilot of
    ``max_chains`` chains, on purpose: a low pilot's rejection rates saturate
    and bias Λ̂ low, forcing the fixed point to climb over several probes,
    while an over-resolved pilot gives an unbiased Λ̂ in one probe, so
    discovery converges in ~2 probes regardless of problem size.

    Instead of providing ``ebm_factory`` and ``program_factory``, you can pass
    a template ``ebm`` and ``program`` and the factories will be built
    internally using ``ebm.with_beta()`` and ``program.with_ebm()``.
    ``init_factory`` is still required as initialization varies by use case.

    Args:
        key: PRNG key
        ebm_factory: betas_array → list[EBM]
        program_factory: list[EBM] → list[Program]
        init_factory: (n_chains, list[EBM], list[Program]) → list[init_states].
            Receives EBMs and programs so it can extract the correct
            free_blocks for initialization (block nodes must be the same
            objects as the EBMs' nodes). May instead return the **stacked**
            form — one array per free block with a leading ``(n_chains,)``
            axis, e.g. ``hinton_init(key, ebm, blocks, (n_chains,))`` — which
            avoids a per-probe restack compile; drawing at a fixed width
            (e.g. ``max_chains``) and slicing to ``n_chains`` also keeps the
            init compile shape-stable across probes.
        clamp_state: clamped block states
        beta_range: (β_min, β_max) for the temperature range
        gibbs_steps_per_round: Gibbs sweeps between swap attempts
        initial_n: starting chain count. The default ``None`` runs a high pilot
            probe at ``max_chains`` for an unbiased Λ̂ (no initial guess needed),
            unless ``seed_from_energy`` is set; pass an int to start there instead.
        seed_from_energy: seed the search from a cheap energy-variance Λ̂
            (Theorem 2, no PT ladder; see :func:`_estimate_barrier_energy`) so it
            converges in one probe instead of the ``max_chains`` pilot's two —
            fewer compiles. Default ``True`` (applies when ``initial_n`` is
            ``None``); pass ``False`` to always run the pilot. **Self-guarding:**
            the estimate is only trustworthy when local exploration mixes, so the
            energy probe also returns a Gelman–Rubin R̂; if R̂ exceeds the cutoff
            (trapping — a glassy target where the estimate would be unreliable)
            the search falls back to the robust ``max_chains`` pilot. On a mixing
            target (R̂≈1) the seed lands on N* and, because the probe RNG is
            key-aligned with the pilot, the discovered N and schedule are
            bit-identical to the pilot path; on a glassy target it is exactly the
            pilot. So it never under-provisions — it only ever *saves* (mixing) or
            *matches* the pilot (glassy), at the cost of the energy probe.
        target_acceptance: desired per-pair swap acceptance rate. Default 0.5 —
            the round-trip-optimal rejection r* = 1/2 (N* ≈ 2Λ; Syed et al.), not
            the 0.77 from the reversible-PT literature.
        rounds_per_probe: rounds per probe. A probe's production window is topped
            up to ``6*(n-1)`` when that exceeds this (so a high-N pilot can
            actually round-trip and identify the barrier); the tuning phases and
            low-N probes stay at ``rounds_per_probe``.
        n_tune_per_probe: schedule tuning iterations for the final probe
        max_iters: maximum discovery iterations
        min_chains: floor on chain count
        max_chains: ceiling on chain count
        lambda_rtol: relative tolerance for Λ stabilization (default 5%)
        safety_margin: small fractional pad on N* (default 0.05) covering residual
            barrier bias and ELE-assumption violations; 0.0 gives the bare
            round-trip-optimal count
        pad_probes: run every probe's round loop padded to ``max_chains`` with
            chain masking (see :func:`hamon.nrpt.nrpt`), so probes at different
            chain counts share ONE compiled round loop instead of recompiling
            per count — the dominant cold cost of discovery. Padding chains do
            wasted-but-decoupled Gibbs work (~free on a dispatch-bound
            accelerator; real cost on CPU, so leave off there). Template
            (temperature-linear) mode only. Probe *statistics* are computed on
            the sliced live prefix, but the probe RNG stream differs from an
            unpadded run, so discovered N can shift within its normal
            probe-to-probe variability.
        compile_cache: ``True`` (default) enables the persistent compile cache
            at the default path, a ``str`` enables it at that path, ``False``
            leaves it untouched. Discovery is compile-dominated cold, so
            without the cache every fresh process pays the full XLA compile;
            same default as :func:`hamon.autotune`. See
            :func:`hamon.enable_persistent_compile_cache`.

    Returns:
        dict with keys:
            n_chains: final recommended chain count
            betas: optimized schedule at that chain count
            Lambda: conservative (max) barrier estimate
            Lambda_raw: last raw estimate (may be lower than Lambda)
            target_acceptance: the target used
            converged_reason: "chain_count" | "lambda_stable" | "max_iters"
            barrier_identified: whether the final count's ladder round-tripped, so
                the reported Lambda is identified rather than a stalled-conveyor
                within-basin artifact (``None`` if round trips were not tracked).
            history: list of per-probe dicts (each carries ``barrier_identified``)
    """
    source = _ChainSource(ebm_factory, program_factory, ebm, program)

    if init_factory is None:
        raise ValueError("init_factory is required.")
    if clamp_state is None:
        clamp_state = []
    _require_proper_beta_start(beta_range[0], ebm)
    if compile_cache:
        enable_persistent_compile_cache(
            compile_cache if isinstance(compile_cache, str) else None
        )

    r_target = max(1.0 - target_acceptance, 1e-3)
    min_chains = int(min_chains)
    max_chains = int(max_chains)
    max_probes = int(max_iters)

    def _clamp(n):
        return max(min_chains, min(max_chains, int(n)))

    # One device for all probes (avoids transfer thrash), scored at the pilot
    # chain count; the max_chains pilot itself is justified in the docstring.
    _pilot_n = initial_n if initial_n is not None else max_chains
    _meta_betas = np.linspace(beta_range[0], beta_range[1], 1)
    dev = resolve_entry_device(
        device,
        n_chains=_clamp(_pilot_n),
        n_nodes=source.metadata_free_nodes(_meta_betas, device),
        arrays=(key,),
    )

    history: list[dict[str, Any]] = []
    probed: dict[int, dict[str, Any]] = {}

    def probe(n: int) -> dict[str, Any]:
        """One schedule-tuned NRPT probe at ``n`` chains, cached by ``n``."""
        nonlocal key
        n = _clamp(n)
        if n in probed:
            return probed[n]
        betas0 = np.linspace(
            beta_range[0],
            beta_range[1],
            n,
            dtype=jax.dtypes.canonicalize_dtype(np.float64),
        )
        # On the template route every entry is the template program, so no
        # per-chain programs are constructed for init.
        ebms = source.ebms_for_init(betas0)
        programs = source.programs_for_init(n, ebms)
        inits = init_factory(n, ebms, programs)
        key, k_probe = jax.random.split(key)
        # Top up only the production window so a high-N probe can round-trip
        # (see _PROBE_MIN_RT_ROUNDS_FACTOR); tuning phases need no trips.
        probe_rounds = max(rounds_per_probe, _PROBE_MIN_RT_ROUNDS_FACTOR * (n - 1))
        # Passing the concrete device bypasses tune_schedule's heuristic, so
        # probes never flip devices.
        _, stats = tune_schedule(
            k_probe,
            ebm_factory,
            program_factory,
            inits,
            clamp_state,
            n_rounds=probe_rounds,
            gibbs_steps_per_round=gibbs_steps_per_round,
            initial_betas=betas0,
            n_tune=n_tune_per_probe,
            rounds_per_tune=rounds_per_probe,
            adaptive_tuning=True,
            tune_tol=tune_tol,
            lambda_rtol=lambda_rtol,
            ebm=ebm,
            program=program,
            device=dev,
            pad_chains_to=max_chains if pad_probes else None,
            _return_stacked=True,
        )
        rej = np.asarray(stats["rejection_rates"])
        out: dict[str, Any] = {
            "n": n,
            "Lambda_raw": float(np.sum(rej)),
            "rejection_rates": rej,
            "betas": np.asarray(stats["betas"]),
            # Round-trip trust gate: whether this probe's ladder round-tripped, so
            # its Λ̂ is a real barrier estimate and not a within-basin artifact.
            "barrier_identified": stats.get("barrier_identified"),
        }
        if out["barrier_identified"] is False:
            # tune_schedule already logged the authoritative gate message; add
            # only the discovery-specific reassurance so this reads as INFO,
            # not an error.
            logger.info(
                "tune_chains: probe at n=%d (%d rounds) did not round-trip; its "
                "Lambda_raw=%.2f may under-estimate the barrier. The running-max "
                "Lambda guards N* — a higher-N probe that round-trips dominates.",
                n,
                probe_rounds,
                out["Lambda_raw"],
            )
        probed[n] = out
        return out

    if max_probes <= 0:
        n_final = _clamp(initial_n if initial_n is not None else min_chains)
        return {
            "n_chains": int(n_final),
            "betas": np.linspace(
                beta_range[0],
                beta_range[1],
                n_final,
                dtype=jax.dtypes.canonicalize_dtype(np.float64),
            ),
            "Lambda": 0.0,
            "Lambda_raw": 0.0,
            "target_acceptance": target_acceptance,
            "converged_reason": "max_iters",
            "history": history,
        }

    # --- N tuning (Syed et al. 2021, Sec. "Tuning N"): N* is driven by the
    # running MAX of Λ̂ so biased-low low-N estimates on glassy targets cannot
    # collapse N — see the docstring for the fixed point.
    margin = 1.0 + max(0.0, float(safety_margin))
    if getattr(ebm, "beta_affine", False) and seed_from_energy:
        # Theorem 2 assumes the linear path E_β = β·E_base; an affine path's
        # integrand is Var(Δ), so the seed would be biased — use the pilot.
        logger.debug(
            "tune_chains: beta-affine EBM — skipping the energy-variance seed "
            "(linear-path assumption); using the max_chains pilot."
        )
        seed_from_energy = False
    if initial_n is not None:
        n = _clamp(initial_n)
    elif seed_from_energy:
        # Consuming exactly one key split (mirroring the discarded pilot probe)
        # keeps the mixing-target path bit-identical to the pilot path.
        key, k_energy = jax.random.split(key)
        lam_seed, rhat = _estimate_barrier_energy(
            k_energy, source, init_factory, clamp_state, beta_range, dev
        )
        if rhat <= _ENERGY_RHAT_MAX:
            n = _clamp(int(np.ceil(lam_seed * margin / r_target)) + 1)
        else:
            n = _clamp(max_chains)  # trapping detected → robust pilot
    else:
        n = _clamp(max_chains)
    lambda_raw = 0.0  # last current-N barrier estimate
    lambda_max = 0.0  # running max over probes; drives N* and reported as Λ
    best_betas = None
    n_star = n
    seen: set[int] = set()
    reason = "max_iters"

    for _ in range(max_probes):
        res = probe(n)
        n = res["n"]
        lambda_raw = float(res["Lambda_raw"])
        lambda_max = max(lambda_max, lambda_raw)
        best_betas = res["betas"]
        seen.add(n)
        n_star = _clamp(int(np.ceil(lambda_max * margin / r_target)) + 1)
        history.append(
            _probe_history_entry(
                len(history),
                n,
                lambda_raw,
                lambda_max,
                n_star,
                res["rejection_rates"],
                res["betas"],
                res.get("barrier_identified"),
            )
        )
        if abs(n_star - n) <= 1:
            reason = "chain_count"
            break
        if n_star in seen:  # 2-cycle: the barrier estimate has settled
            reason = "lambda_stable"
            n_star = max(n_star, n)  # conservative against undershoot
            break
        n = n_star

    # On chain_count convergence, prefer the (always cached) last probed n over
    # an unprobed n_star within tolerance — an extra probe would recompile the
    # round loop just to land on a count already in hand.
    if reason == "chain_count" and _clamp(n_star) not in probed:
        n_final = n
    else:
        n_final = _clamp(n_star)
    final_stats = probed[n_final] if n_final in probed else probe(n_final)
    best_betas = final_stats["betas"]
    lambda_max = max(lambda_max, float(final_stats["Lambda_raw"]))
    if n_final not in seen:
        history.append(
            _probe_history_entry(
                len(history),
                n_final,
                final_stats["Lambda_raw"],
                lambda_max,
                n_final,
                final_stats["rejection_rates"],
                final_stats["betas"],
                final_stats.get("barrier_identified"),
            )
        )

    return {
        "n_chains": int(n_final),
        "betas": best_betas,
        "Lambda": float(lambda_max),
        "Lambda_raw": float(lambda_raw),
        "target_acceptance": target_acceptance,
        "converged_reason": reason,
        # Whether the final chosen count's ladder round-tripped, so the reported
        # Lambda is identified (not a stalled-conveyor within-basin artifact).
        "barrier_identified": final_stats.get("barrier_identified"),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Plain block-Gibbs schedule calibration
# ---------------------------------------------------------------------------


# ~p95 of the null single-series window R̂ at (8 restarts, 4-batch window):
# simulated p50≈1.03, p95≈1.25. The grid's _ENERGY_WARMUP_RHAT is looser
# (1.45) because it maximizes over 10 grid points; a single chain family
# needs the tighter cut. Recalibrate if the window/restart counts change.
_REPLICA_RHAT = 1.25
_TAU_PROBE_TAG = 0x54415550  # "TAUP": fold_in tag for the autocorrelation probe


@eqx.filter_jit
def _replica_sweep(keys, program, sched, states, clamp_state, f_obs, carry):
    """Vmapped sweep of independent replicas sharing one program.

    The single-program sibling of ``_grid_sweep``: serves the warmup batches
    and probe passes of :func:`tune_sampling_schedule`.
    """
    from hamon.block_sampling import _sample_with_observation_core

    nb = len(states)

    def _lane(k, init_free):
        _, res = _sample_with_observation_core(
            k, program, sched, init_free, clamp_state, carry, f_obs
        )
        return res

    return jax.vmap(_lane, in_axes=(0, [0] * nb))(keys, states)


def _integrated_autocorr(series: np.ndarray, max_lag: int) -> float:
    """Pooled integrated autocorrelation time of (R, M) series.

    τ_int = 1 + 2·Σ ρ_k, truncated at the first lag whose pooled
    autocorrelation drops below 0.05 (a simple, conservative cut)."""
    x = series - series.mean(axis=1, keepdims=True)
    denom = float((x * x).sum())
    if denom <= 0.0:
        return 1.0
    tau = 1.0
    for k in range(1, max_lag):
        rho = float((x[:, :-k] * x[:, k:]).sum()) / denom
        if rho < 0.05:
            break
        tau += 2.0 * rho
    return tau


def tune_sampling_schedule(
    key: jax.Array,
    ebm: AbstractEBM,
    program: BlockSamplingProgram,
    init_states: list,
    clamp_state: list | None = None,
    *,
    target_ess: int = 64,
    warmup_cap: int = _ENERGY_WARMUP_MAX,
    probe_samples: int = 200,
    rhat_threshold: float = _REPLICA_RHAT,
    device: DeviceLike = "auto",
) -> tuple[SamplingSchedule, dict]:
    """Calibrate a :class:`~hamon.SamplingSchedule` for plain block-Gibbs runs.

    Answers "what warmup, thinning, and sample count does *this* model at
    *these* parameters need?" — the numbers users otherwise hand-pick for
    ``estimate_kl_grad`` phases or ``sample_states`` draws. Designed as a
    calibrate-then-freeze step (e.g. once per training epoch, at the current
    θ): the returned schedule is static, so a jitted epoch stays one compiled
    scan.

    Mechanics, reusing the energy-grid warmup machinery:

    - **n_warmup** — adaptive: ``_ENERGY_WARMUP_BATCH``-sweep batches over the
      independent replica chains in ``init_states``, stopped by the windowed
      cross-replica R̂ of batch-end energies (stable / plateau / ``warmup_cap``
      exits, exactly as in ``_estimate_barrier_energy``). A plateau exit means
      the replicas are trapped in separate basins: the returned warmup is then
      a floor, not a guarantee — plain Gibbs cannot equilibrate that target
      and tempered sampling (``autotune``) is the honest fix.
    - **steps_per_sample** — the pooled integrated autocorrelation time of the
      chain energy, measured over a ``probe_samples``-sweep probe after
      warmup; thinning at ~τ makes recorded samples approximately
      independent.
    - **n_samples** — ``target_ess``: after thinning at τ the effective sample
      size is roughly the recorded count (mild residual correlation makes
      this an approximation, not a bound).

    Arguments:
        key: PRNG key.
        ebm: the model (energies are evaluated through its β = 1 base, the
            same scale the NRPT swap step uses).
        program: sampling program for the phase being calibrated (clamped
            blocks included for e.g. a positive/data-clamped phase).
        init_states: per-block arrays with a leading replica axis ``(R, …)``
            — e.g. ``hinton_init(key, ebm, free_blocks, (R,))``. R ≥ 2.
        clamp_state: clamped-block values shared by all replicas.
        target_ess: recorded samples (≈ effective samples after thinning).
        warmup_cap: warmup ceiling in sweeps.
        probe_samples: sweeps recorded for the autocorrelation estimate.
        rhat_threshold: stopping threshold for the windowed replica R̂.
        device: where to run; resolved once, as elsewhere.

    Returns:
        ``(schedule, info)`` — the calibrated schedule and a dict with
        ``n_warmup``, ``warmup_exit``, ``tau``, ``rhat_final`` (the window R̂
        at the stop), and ``steps_per_sample``.

    Note:
        This reuses the warmup machinery of ``_estimate_barrier_energy`` (via
        ``_adaptive_warmup``) but is deliberately a separate driver — see that
        function's docstring for why merging the two would fuse their jit
        caches and change a key stream. Nothing here is tempered: no ladder,
        no swaps, one β.
    """
    from hamon._nrpt_energy import _make_reference_ebm
    from hamon.device import free_node_count, tree_device_put
    from hamon.observers import StateObserver

    if clamp_state is None:
        clamp_state = []
    R = int(jax.tree.leaves(init_states)[0].shape[0])
    if R < 2:
        raise ValueError("tune_sampling_schedule needs at least 2 replicas.")

    spec = program.gibbs_spec
    base_ebm, beta_ref = _make_reference_ebm([ebm], jnp.ones(1))

    dev = resolve_entry_device(
        device,
        n_chains=R,
        n_nodes=free_node_count(program),
        arrays=(init_states, clamp_state, key),
    )
    if dev is not None:
        key, program, init_states, clamp_dev = tree_device_put(
            (key, program, init_states, clamp_state), dev
        )
    else:
        clamp_dev = clamp_state
    device_ctx = default_device_ctx(dev)

    sched_batch = SamplingSchedule(_ENERGY_WARMUP_BATCH, 1, 1)
    sched_probe = SamplingSchedule(0, int(probe_samples), 1)
    f_obs = StateObserver(spec.free_blocks)
    carry = f_obs.init()
    keys_flat = jax.random.split(key, R)
    fold_all = jax.vmap(jax.random.fold_in, in_axes=(0, None))
    one_beta = np.ones(1)

    with device_ctx:

        def run_batch(states, batch_index):
            out = _replica_sweep(
                fold_all(keys_flat, batch_index),
                program,
                sched_batch,
                states,
                clamp_dev,
                f_obs,
                carry,
            )
            return [o[:, 0] for o in out]

        def batch_energies(states):
            return np.asarray(
                _grid_base_energies(base_ebm, beta_ref, spec, states, clamp_state)
            ).reshape(1, R)

        states, total, exit_reason, rh = _adaptive_warmup(
            init_states,
            run_batch,
            batch_energies,
            cap=warmup_cap,
            rhat_threshold=rhat_threshold,
            rhat_betas=one_beta,
            log_label="tune_sampling_schedule",
        )

        raw = _replica_sweep(
            fold_all(keys_flat, _TAU_PROBE_TAG),
            program,
            sched_probe,
            states,
            clamp_dev,
            f_obs,
            carry,
        )  # per free block: (R, probe_samples, …)
        M = int(probe_samples)
        flat = [blk.reshape((R * M, *blk.shape[2:])) for blk in raw]
        E = np.asarray(
            _grid_base_energies(base_ebm, beta_ref, spec, flat, clamp_state)
        ).reshape(R, M)

    tau = _integrated_autocorr(E, max_lag=M // 4)
    steps = max(1, int(np.ceil(tau)))
    if exit_reason == "plateau":
        logger.warning(
            "tune_sampling_schedule: replicas did not merge (window rhat=%.2f) — "
            "plain Gibbs is trapped on this target; the returned warmup is a "
            "floor, consider tempered sampling (autotune).",
            rh,
        )

    schedule = SamplingSchedule(total, int(target_ess), steps)
    info = {
        "n_warmup": total,
        "warmup_exit": exit_reason,
        "tau": float(tau),
        "steps_per_sample": steps,
        "rhat_final": float(rh),
    }
    logger.debug("tune_sampling_schedule: %s", info)
    return schedule, info


# ---------------------------------------------------------------------------
# Adaptive local-exploration count (n_expl = gibbs_steps_per_round)
# ---------------------------------------------------------------------------


def _select_gibbs_steps(
    probe: Callable[[int], dict],
    start_steps: int,
    max_steps: int,
    improve_tol: float,
) -> tuple[dict, list[dict]]:
    """Pick n_expl maximizing a per-compute objective by doubling until the peak.

    ``probe(n_expl)`` runs NRPT at that exploration count and returns a record
    with at least ``"objective"`` (ESS per unit compute) and
    ``"efficiency_limiter"``.

    The objective rises then falls in n_expl: more local exploration decorrelates
    the cold chain (raising ESS and the round-trip rate toward the schedule-set
    ceiling) while the per-scan cost grows as O(n_expl), so the ESS/compute ratio
    has an interior (or boundary) maximum. We double n_expl until the objective
    stops improving by more than ``improve_tol`` (we have passed the peak), hit
    ``max_steps``, or the report attributes the inefficiency to the schedule
    rather than local exploration (``efficiency_limiter == "schedule"`` — more
    sweeps will not help). The best record seen is returned, so the choice is the
    argmax over probed counts.

    Pure (no NRPT / JAX here) so the control logic is unit-testable with a
    synthetic ``probe``.
    """
    history: list[dict] = []
    best: dict | None = None
    n = start_steps
    while True:
        res = probe(n)
        history.append(res)
        if best is None or res["objective"] > best["objective"] * (1.0 + improve_tol):
            best = res
            improved = True
        else:
            improved = False
        if n >= max_steps or not improved:
            break
        if res.get("efficiency_limiter") == "schedule":
            break
        n *= 2
    assert best is not None
    return best, history


def _time_per_round(
    key: jax.Array,
    ebms,
    programs,
    init_states: Sequence[list],
    clamp_state: list,
    betas: jax.Array,
    n_expl: int,
    dev,
    time_rounds: int,
    time_reps: int,
    observer: AbstractNRPTObserver | None = None,
) -> float:
    """Measured steady-state wall time per NRPT round at ``n_expl`` (seconds).

    Times the round loop on the already-tuned schedule. A warm-up run absorbs the
    one-time XLA compile and pages in the device, so the timed reps see only
    steady-state execution; the median over ``time_reps`` runs of ``time_rounds``
    rounds divides out to the per-round cost ``c₀ + n_expl·c_s``.
    ``track_round_trips`` is left on so the in-loop index update (real per-round
    cost) is included; only the host-side summary is skipped. Pass the **same
    observer the production probe used** to reuse its compiled executable — no
    separate ``observer=None`` compile, which the cost model relies on;
    ``observer=None`` (default) times the lean loop.
    """

    def run(n_rounds: int):
        states, _ = nrpt(
            key,
            ebms,
            programs,
            init_states,
            clamp_state,
            n_rounds,
            n_expl,
            betas=betas,
            track_round_trips=True,
            observer=observer,
            device=dev,
            _emit_diagnostics=False,
        )
        return states

    jax.block_until_ready(run(time_rounds))  # warm: compile + first execution
    times = []
    for _ in range(max(1, time_reps)):
        t0 = time.perf_counter()
        jax.block_until_ready(run(time_rounds))
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) / time_rounds


def _fit_cost_line(ns: Sequence[int], ts: Sequence[float]) -> tuple[float, float]:
    """Least-squares fit ``t_round = c₀ + n_expl·c_s`` over probed (n, t) points.

    The points are the per-probe production timings (reused, not separately
    measured). One shared line — rather than each count's own noisy timing —
    makes the objective's argmax reproducible: the ranking then depends on the
    common ``c₀, c_s`` and the deterministic ESS, not on per-count timing noise
    (the objective is nearly flat near its peak, so pairwise-comparing noisy
    per-count timings picked the stopping count at random). Both terms are
    floored at 0 against noise that could yield a negative slope or intercept;
    with a single distinct point the slope is 0 (flat line at that level).
    """
    a = np.asarray(ns, dtype=float)
    b = np.asarray(ts, dtype=float)
    if a.size >= 2 and a.min() != a.max():
        cs, c0 = (float(v) for v in np.polyfit(a, b, 1))
    else:
        cs, c0 = 0.0, (float(b[0]) if b.size else 0.0)
    return max(c0, 0.0), max(cs, 0.0)


def _select_gibbs_steps_ele(
    probe: Callable[[int], dict],
    start_steps: int,
    max_steps: int,
    improve_tol: float,
    target_efficiency: float,
) -> tuple[dict, list[dict]]:
    """Pick the smallest n_expl that adequately approximates ELE (Syed et al.).

    n_expl exists to approximate Efficient Local Exploration (assumption A2):
    enough local moves between swaps that the energy decorrelates. ELE-adequacy is
    read off the round-trip **efficiency** ``τ_obs / τ̂`` where ``τ̂ = 1/(2+2Λ)``
    is the theoretical optimum (Thm 3) — both computed from rejection / round-trip
    counts, so this is **fully deterministic** (no wall-clock) and reproducible
    across runs, unlike an ESS-per-second timing search.

    Efficiency rises with n_expl and plateaus once ELE is approximated. We double
    n_expl while efficiency keeps gaining > ``improve_tol``, stopping when it
    reaches ``target_efficiency``, plateaus, hits ``max_steps``, or the report
    blames the schedule (``efficiency_limiter == "schedule"`` — more local moves
    cannot help; add chains instead). We then return the **smallest** n_expl
    within ``improve_tol`` of the best efficiency seen: the knee of the curve, the
    cheapest count giving near-optimal index-process mixing. Under the paper's
    cost model (cost ∝ n_expl) that knee maximizes round-trips per compute; on a
    dispatch-bound accelerator going higher is ~free but only marginally helps
    (the ESS/sec plateau), so the knee is a sound, reproducible choice.

    Pure control logic (no JAX), unit-testable with a synthetic ``probe``.
    """

    def _eff(rec: dict) -> float:
        e = rec.get("efficiency")
        return float(e) if e is not None else 0.0

    history: list[dict] = []
    n = start_steps
    while True:
        rec = probe(n)
        history.append(rec)
        if _eff(rec) >= target_efficiency:
            break
        if rec.get("efficiency_limiter") == "schedule":
            break
        if n >= max_steps:
            break
        if len(history) >= 2 and _eff(rec) <= _eff(history[-2]) * (1.0 + improve_tol):
            break  # efficiency plateaued: more local moves no longer help ELE
        n *= 2
    best_eff = max(_eff(r) for r in history)
    best = history[-1]
    for r in history:  # ascending n: cheapest count within tol of the plateau
        if _eff(r) >= best_eff * (1.0 - improve_tol):
            best = r
            break
    return best, history


def _select_gibbs_steps_cost(
    probe: Callable[[int], dict],
    start_steps: int,
    max_steps: int,
    improve_tol: float,
    rounds_per_probe: int,
) -> tuple[dict, list[dict]]:
    """Pick n_expl by ESS-driven doubling, scored against one fitted cost line.

    The reuse-timing strategy: because every probe reuses the same schedule, the
    doubling is driven by **ESS** alone (a deterministic probe set), and wall
    time enters only afterwards through a single least-squares cost line
    ``t_round ≈ c₀ + c_s·n_expl`` fitted across all probes. Scoring every record
    against that fitted line rather than its own measured time is what keeps the
    near-flat peak from flipping between counts on timing noise; requiring a
    ``> improve_tol`` gain to climb resolves it to the cheaper count.

    Note the ``history`` records are rewritten in place: ``t_round`` is replaced
    by its fitted value and ``objective`` is filled in. Callers surface that same
    list, so the fitted numbers are what users see.

    Sibling of :func:`_select_gibbs_steps` and :func:`_select_gibbs_steps_ele`.
    The three share a doubling skeleton but differ in their stopping metric
    (fitted objective / round-trip efficiency / raw ESS), the order their breaks
    are tested, and how the winner is chosen (running best / cheapest within tol
    of the plateau / post-hoc argmax over the rescored history). They are kept
    separate deliberately: the probe count drives ``jax.random.split`` in
    ``tune_exploration``, so merging them behind a shared predicate list would
    risk silently reordering a break and re-baselining the sample stream.

    Pure control logic (no JAX), unit-testable with a synthetic ``probe``.
    """
    history: list[dict] = []
    n = start_steps
    while True:
        rec = probe(n)
        history.append(rec)
        if rec.get("efficiency_limiter") == "schedule":
            break  # schedule-limited: more local exploration cannot help
        if n >= max_steps:
            break
        if len(history) >= 2 and rec["ess_median"] <= history[-2]["ess_median"] * (
            1.0 + improve_tol
        ):
            break  # ESS saturated: extra sweeps no longer decorrelate
        n *= 2
    c0, cs = _fit_cost_line(
        [r["n_expl"] for r in history], [r["t_round"] for r in history]
    )
    for r in history:
        tr = c0 + r["n_expl"] * cs
        r["t_round"] = tr
        r["objective"] = r["ess_median"] / (rounds_per_probe * tr) if tr > 0 else 0.0
    best = history[0]
    for r in history[1:]:
        if r["objective"] > best["objective"] * (1.0 + improve_tol):
            best = r
    return best, history


def tune_exploration(
    key: jax.Array,
    ebm_factory: Callable | None = None,
    program_factory: Callable | None = None,
    init_states: Sequence[list] = (),
    clamp_state: list | None = None,
    initial_betas: jax.Array | None = None,
    start_steps: int = 1,
    max_steps: int = 64,
    rounds_per_probe: int = 400,
    n_tune_per_probe: int = 3,
    improve_tol: float = 0.05,
    time_rounds: int = 200,
    time_reps: int = 3,
    cost_model: bool = True,
    select_by: str = "cost",
    target_efficiency: float = 0.9,
    *,
    fixed_schedule: jax.Array | None = None,
    ebm: AbstractEBM | None = None,
    program: BlockSamplingProgram | None = None,
    device: DeviceLike = "auto",
) -> dict:
    """Discover the local-exploration count ``gibbs_steps_per_round`` (n_expl).

    n_expl is the only major NRPT knob hamon does not otherwise auto-tune
    (``tune_chains`` sets N from Λ; ``tune_schedule`` sets the
    schedule). The **objective** maximized here is effective sample size per
    **measured steady-state wall-second**,

        objective(n_expl) = ESS_median(cold chain) / (rounds · t_round(n_expl)),

    where ``t_round`` is the per-round wall time *measured on the target device*
    after warm-up (so XLA compile is excluded). This is the honest endgame and,
    crucially, it self-calibrates to the hardware's real cost structure. The
    per-round cost is ``t_round = c₀ + n_expl·c_s``: a fixed overhead ``c₀``
    (swap pass, energy recompute, host/kernel dispatch, scan bookkeeping) plus
    ``n_expl`` Gibbs sweeps at ``c_s`` each. The Syed et al. compute model assumes
    ``c₀ = 0`` (cost ∝ n_expl), which makes n_expl=1 optimal; but cold-chain ESS
    grows *sub-linearly* in n_expl while real per-round cost grows *less than
    linearly* when ``c₀`` is non-trivial — so on a dispatch-bound backend where
    ``c₀ ≳ 1.4·c_s`` the optimum shifts to n_expl > 1. Measuring ``t_round``
    rather than assuming cost ∝ n_expl is what lets the search see this.

    Why not the cheaper round-trip proxy ``τ_obs / n_expl``? Empirically it
    *under-picks* n_expl: round trips count excursions but not how decorrelated
    each cold sample is, so doubling n_expl can sharply raise cold-chain ESS
    while barely moving τ_obs (the r≈0.81 ESS↔round-trip correlation of Syed
    et al. breaking down). The round-trip rate is still used as a robust **gate
    and cross-check** — ``efficiency = τ_obs/τ̄`` is the ELE-violation meter, and
    the ``efficiency_limiter`` from :func:`report_nrpt_diagnostics` stops the
    search when a probe is schedule-limited (an unequalized ladder, where more
    local exploration cannot help). ``rt_per_compute`` and ``t_round`` are
    recorded per probe alongside the ESS-per-second objective.

    See :func:`tune_chains` for the N analogue. The chain count is held
    fixed at ``len(initial_betas)`` — Λ (hence N) is a schedule invariant robust
    to n_expl, so the two searches decouple; run ``tune_chains`` first
    if N is unknown.

    Each probe (1) gets a schedule, (2) measures ESS over a cold-chain trace via
    an :class:`~hamon.NRPTStateObserver`, and (3) times the steady-state round
    loop (warm-up absorbs the one-time compile; the median of ``time_reps`` runs
    of ``time_rounds`` rounds gives ``t_round``). With ``fixed_schedule=None``
    (default) the schedule is re-tuned per probe via :func:`tune_schedule`; pass
    ``fixed_schedule`` (a pre-tuned ladder) to **reuse it** and run each probe as
    a single :func:`nrpt` production call — much cheaper, and sound because the
    equi-acceptance schedule is invariant to n_expl. ``autotune`` uses this mode.
    Instead of ``ebm_factory`` / ``program_factory`` you may pass a single
    template ``ebm`` and ``program`` (temperature-linear mode).

    Args:
        key: PRNG key.
        ebm_factory: per-chain EBM factory.
        program_factory: per-chain sampling-program factory.
        ebm: single EBM template (temperature-linear mode; alternative to
            ``ebm_factory``).
        program: single sampling-program template (alternative to
            ``program_factory``).
        init_states: one initial block-state list per chain (fixed across probes).
        clamp_state: clamped block states.
        initial_betas: the (fixed) β ladder; its length sets the chain count.
        start_steps: smallest n_expl to try (``≥ 1``).
        max_steps: largest n_expl before the search stops.
        rounds_per_probe: production rounds per probe (and the tuning ceiling) —
            must be large enough for ESS and τ_obs to be low-variance.
        n_tune_per_probe: schedule-tuning phases per probe.
        improve_tol: minimum fractional objective gain to keep doubling
            (guards against chasing Monte-Carlo noise past the peak).
        time_rounds: rounds per timed run when measuring ``t_round``.
        time_reps: number of timed runs to reduce over (noise control).
        select_by: how to choose n_expl. ``"cost"`` (default) maximizes cold-chain
            ESS per wall-second (see ``cost_model``) — the sample-quality objective;
            the pick depends on the machine's measured cost ratio, so it is best
            used as a one-time per-hardware calibration. ``"ele"`` instead picks
            the smallest count whose round-trip efficiency ``τ_obs/τ̂`` reaches the
            ELE-adequacy knee — deterministic (no wall-clock) and the criterion the
            Syed et al. analysis prescribes, but it optimizes index-process mixing
            rather than cold-sample ESS, so it under-picks n_expl on a
            dispatch-bound accelerator where extra sweeps are nearly free; use it
            for index-efficiency or severe-ELE-violation regimes.
        target_efficiency: ELE-adequacy threshold for ``select_by="ele"`` — stop
            climbing once ``τ_obs/τ̂`` reaches this (it also stops on a plateau or
            a schedule-limited verdict).
        cost_model: for ``select_by="cost"`` only — when ``True`` (and a
            ``fixed_schedule`` is given) fit ``t_round = c₀ + n_expl·c_s`` by least
            squares across the probes (each timing reuses the production
            executable, no separate ``observer=None`` compile) and take the argmax
            from that shared line; ``False`` times each probe independently (the
            flat objective then lets timing noise pick the count at random).
        fixed_schedule: a pre-tuned β ladder to reuse across all probes (each
            probe becomes one production run, no per-probe re-tuning). ``None``
            (default) re-tunes per probe; ``autotune`` passes the ladder from
            ``tune_chains``.
        device: where to run; resolved once and reused across probes. Timing is
            measured on this device, so the chosen n_expl is calibrated to it.

    Returns:
        dict with keys:
            gibbs_steps_per_round: the chosen n_expl.
            objective: ESS per measured wall-second at the chosen n_expl.
            ess_median / tau_observed / efficiency / rt_per_compute / t_round:
                at the choice (``t_round`` in seconds).
            betas: the tuned schedule at the chosen n_expl.
            history: list of per-probe records (n_expl, objective, ess_median,
                tau_obs, rt_per_compute, t_round, efficiency, efficiency_limiter,
                betas).
    """
    from hamon.diagnostics import effective_sample_size, report_nrpt_diagnostics
    from hamon.observers import NRPTStateObserver

    if clamp_state is None:
        clamp_state = []
    if initial_betas is None:
        raise ValueError("initial_betas is required (its length sets the chain count).")
    if not init_states:
        raise ValueError("init_states is required (one initial state per chain).")
    if int(start_steps) < 1:
        raise ValueError("start_steps must be >= 1 (n_expl = 0 has no exploration).")

    # Build a chain source so the timing phase can reconstruct nrpt() arguments
    # for either the template or factory route.
    source = _ChainSource(ebm_factory, program_factory, ebm, program)
    betas = jnp.asarray(initial_betas)
    dev = resolve_entry_device(
        device,
        n_chains=len(betas),
        n_nodes=source.metadata_free_nodes(betas, device),
        arrays=(init_states, betas, key),
    )
    source.device_put_template(dev)

    obs = NRPTStateObserver(chain_indices=(-1,))
    # Reuse-timing cost model: only on a fixed schedule, where each probe is a
    # single production run whose executable we can re-time directly.
    use_reuse_timing = cost_model and fixed_schedule is not None

    def probe(n_expl: int) -> dict[str, Any]:
        nonlocal key
        key, k_probe, k_time = jax.random.split(key, 3)
        # (1) Get a schedule + cold-chain trace at this n_expl. With a fixed
        # schedule each probe is a single production run (no re-tuning, since the
        # equi-acceptance schedule is n_expl-invariant); otherwise re-tune.
        if fixed_schedule is not None:
            tuned_betas = jnp.asarray(fixed_schedule)
            ebms_t, programs_t = source.nrpt_args(tuned_betas)
            _, stats = nrpt(
                k_probe,
                ebms_t,
                programs_t,
                init_states,
                clamp_state,
                rounds_per_probe,
                n_expl,
                betas=tuned_betas,
                track_round_trips=True,
                observer=obs,
                device=dev,
            )
        else:
            _, stats = tune_schedule(
                k_probe,
                ebm_factory,
                program_factory,
                init_states,
                clamp_state,
                n_rounds=rounds_per_probe,
                gibbs_steps_per_round=n_expl,
                initial_betas=betas,
                n_tune=n_tune_per_probe,
                rounds_per_tune=rounds_per_probe,
                ebm=ebm,
                program=program,
                observer=obs,
                device=dev,
            )
            tuned_betas = jnp.asarray(stats["betas"])
            ebms_t, programs_t = source.nrpt_args(tuned_betas)
        rep = report_nrpt_diagnostics(stats)
        tau_obs = float(rep.tau_observed) if rep.tau_observed is not None else 0.0

        # ESS is per-column then median-reduced (steadier than min across
        # seeds), so cold-chain column order is irrelevant.
        cold = [np.asarray(o)[:, 0] for o in stats["observations"]]
        trace = np.concatenate([c.reshape(c.shape[0], -1) for c in cold], axis=1)
        ess = effective_sample_size(trace)

        # Only the timing-based selectors need wall time; ELE is timing-free.
        if select_by == "ele":
            t_round = 0.0
            objective = rep.efficiency if rep.efficiency is not None else 0.0
        else:
            t_round = _time_per_round(
                k_time,
                ebms_t,
                programs_t,
                init_states,
                clamp_state,
                tuned_betas,
                n_expl,
                dev,
                time_rounds,
                time_reps,
                observer=obs if use_reuse_timing else None,
            )
            objective = ess.median_ess / (rounds_per_probe * t_round)
        return {
            "n_expl": int(n_expl),
            "objective": objective,  # ESS/wall-second (cost) or efficiency (ele)
            "ess_median": ess.median_ess,
            "tau_obs": tau_obs,
            "rt_per_compute": tau_obs / n_expl,
            "t_round": t_round,
            "efficiency": rep.efficiency,
            "efficiency_limiter": rep.efficiency_limiter,
            "betas": np.asarray(tuned_betas),
        }

    if select_by == "ele":
        # Timing-free: smallest n_expl whose round-trip efficiency reaches the
        # ELE-adequacy knee, so the pick reproduces across runs.
        best, history = _select_gibbs_steps_ele(
            probe,
            int(start_steps),
            int(max_steps),
            float(improve_tol),
            float(target_efficiency),
        )
    elif use_reuse_timing:
        best, history = _select_gibbs_steps_cost(
            probe,
            int(start_steps),
            int(max_steps),
            float(improve_tol),
            int(rounds_per_probe),
        )
    else:
        best, history = _select_gibbs_steps(
            probe, int(start_steps), int(max_steps), float(improve_tol)
        )

    return {
        "gibbs_steps_per_round": best["n_expl"],
        "objective": best["objective"],
        "ess_median": best["ess_median"],
        "tau_observed": best["tau_obs"],
        "rt_per_compute": best["rt_per_compute"],
        "t_round": best["t_round"],
        "efficiency": best["efficiency"],
        "betas": best["betas"],
        "history": history,
    }
