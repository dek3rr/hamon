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

import jax
import jax.numpy as jnp
import numpy as np

from hamon.block_sampling import BlockSamplingProgram
from hamon.device import DeviceLike, resolve_entry_device
from hamon.models.ebm import AbstractEBM
from hamon.observers import AbstractNRPTObserver
from hamon.nrpt import (
    _ChainSource,
    _phase_diagnostics,
    _pooled_lambda,
    _swap_rate_stats,
    nrpt,
    optimize_schedule,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Convenience: NRPT with iterative schedule tuning
# ---------------------------------------------------------------------------

# Default schedule-movement floor used by adaptive tuning when the caller leaves
# tune_tol unset. In β units: a schedule update that moves every β by less than
# this is treated as "settled" (at the Monte-Carlo noise floor — the ladder
# keeps jittering by ~this much even when well tuned, so a much tighter value is
# never reached). Pair with the equalization check, which catches already-good
# schedules whose β jitter alone never crosses this floor.
_DEFAULT_TUNE_TOL = 0.02


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
        states, stats = run_phase(subkey, betas, states, batch, return_stacked=True)
        acc_total = (
            stats["accepted"] if acc_total is None else acc_total + stats["accepted"]
        )
        att_total = (
            stats["attempted"] if att_total is None else att_total + stats["attempted"]
        )
        rounds_used += batch

        lambda_cur = float(_pooled_lambda(acc_total, att_total, betas.dtype))
        if rounds_used >= min_rounds and lambda_prev is not None:
            rel = abs(lambda_cur - lambda_prev) / max(lambda_cur, 1e-9)
            stable_count = stable_count + 1 if rel < lambda_rtol else 0
            if stable_count >= stable_k:
                break
        lambda_prev = lambda_cur
        if rounds_used >= max_rounds:
            break

    assert acc_total is not None and att_total is not None
    pooled_stats = _swap_rate_stats(acc_total, att_total, betas)
    return states, pooled_stats, rounds_used


def tune_schedule(
    key: jax.Array,
    ebm_factory: Callable | None = None,
    program_factory: Callable | None = None,
    init_states: Sequence[list] = (),
    clamp_state: list | None = None,
    n_rounds: int = 0,
    gibbs_steps_per_round: int = 0,
    initial_betas: jax.Array | None = None,
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
    device: DeviceLike = "auto",
) -> tuple[list, dict]:
    """NRPT with iterative schedule optimization (Algorithm 4).

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
    (lowest-spread) rejection rates seen across phases is kept for production —
    not the last, which can be noisier. Tuning stops once the schedule is
    well-equalized (``std(rejection_rates) < equalize_tol``) OR has settled
    (``max|Δβ|`` below the effective ``tune_tol`` — its Monte-Carlo floor) for
    ``phase_patience`` consecutive phases, after at least ``min_tune_phases``,
    capped at ``n_tune``. (``max|Δβ|`` alone is not a reliable convergence
    signal: it plateaus at a problem-dependent noise floor rather than going to
    zero, so the equalization check is what stops already-good schedules.) When
    ``tune_tol`` is left ``None`` it defaults to ``_DEFAULT_TUNE_TOL`` here.
    Counts are deterministic for a given seed but problem-dependent — do not
    assume a fixed round/phase count.

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

    # The template route hands nrpt the same β = 1 base pair every phase
    # (temperature-linear mode, jit cache reuse); the factory route builds
    # per-chain sequences per phase. _ChainSource hides the difference.
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
    ):
        # Tuning batches default to emit_diag=False (they only read swap rates,
        # so the eager round-trip summary is skipped) and return_stacked=True
        # (states are threaded in stacked form, skipping the per-call unstack).
        # The production run passes emit_diag=True and keeps the public
        # per-chain return.
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
            _emit_diagnostics=emit_diag,
            _return_stacked=return_stacked,
        )

    # Pin the working schedule to the resolved device so optimize_schedule (and
    # the other per-phase reductions) see a committed array every phase. Phase 0
    # would otherwise pass the caller's uncommitted initial_betas, and jit would
    # build a second, single-use executable keyed on the uncommitted input.
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

    # Keep the best-equalized schedule actually evaluated (adaptive mode), so a
    # noisy late phase can't hand a worse ladder to production.
    best_betas = betas
    best_quality = float("inf")
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
                subkey, betas, current_states, rounds_per_tune, return_stacked=True
            )
            rounds_used = rounds_per_tune

        rej = stats["rejection_rates"]
        old_betas = betas
        new_betas = optimize_schedule(rej, betas)
        # Equalization quality (lower spread of per-pair rejection rates = better
        # tuned; drives keep-best and the equalization stop), ladder movement,
        # Λ, and mean acceptance — all in one fused kernel instead of separate
        # eager reductions. rej_std depends only on the pre-update rates, so
        # keep-best still records old_betas.
        rej_std_a, shift_a, lambda_a, mean_acc_a = _phase_diagnostics(
            rej, old_betas, new_betas, stats["acceptance_rate"]
        )
        quality = float(rej_std_a)
        if adaptive_tuning and quality < best_quality:
            best_quality = quality
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
            "tune_schedule tune %d/%d: Lambda=%.3f mean_acceptance=%.3f "
            "rej_std=%.4g max|dbeta|=%.4g rounds=%d",
            phase,
            n_tune,
            phase_lambda,
            float(mean_acc_a),
            quality,
            max_beta_shift,
            rounds_used,
        )

        if adaptive_tuning:
            # Combined stop: well-equalized OR schedule movement at its noise
            # floor, sustained for phase_patience consecutive phases.
            equalized = quality < equalize_tol
            settled = effective_tol is not None and max_beta_shift < effective_tol
            stop_streak = stop_streak + 1 if (equalized or settled) else 0
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
        subkey, betas, current_states, n_rounds, phase_observer=observer, emit_diag=True
    )
    stats["tuning_history"] = tuning_history
    return states, stats


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
    }


def tune_chains(
    key: jax.Array,
    ebm_factory: Callable | None = None,
    program_factory: Callable | None = None,
    init_factory: Callable | None = None,
    clamp_state: list | None = None,
    beta_range: tuple[float, float] = (0.0, 1.0),
    gibbs_steps_per_round: int = 0,
    initial_n: int | None = None,
    target_acceptance: float = 0.5,
    rounds_per_probe: int = 200,
    n_tune_per_probe: int = 4,
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
) -> dict:
    """Iteratively discover the right chain count for a given target acceptance.

    Follows the N-tuning method of Syed et al. (2021): the global communication
    barrier Λ is a schedule invariant (Σ rejection_rates ≈ Λ at any chain count),
    so it is estimated at a single fixed N from a schedule-tuned run rather than
    searched for by probing many chain counts.

    1. Estimate Λ̂ = Σ rejection_rates at the current N (each probe runs
       ``tune_schedule``, which tunes the schedule toward equi-acceptance).
    2. Recommend N* = ceil(Λ̂·(1 + safety_margin) / r_target) + 1 — the
       round-trip-optimal 2Λ + 1 chains at r* = 1/2 (target_acceptance = 0.5).
    3. Iterate this fixed point (re-estimate Λ̂ at N*) until N* stops moving.

    Because Λ̂ comes from the current probe (not a running maximum), the result is
    essentially independent of the starting N — discovery from ``initial_n=None``
    and from a reasonable guess converge to the same count. With no ``initial_n``
    the first probe runs at a **high** pilot of ``max_chains`` chains: high on
    purpose, because a low pilot's rejection rates saturate and bias Λ̂ low,
    forcing the fixed point to climb over several probes. An over-resolved pilot
    gives an unbiased Λ̂ in one probe, landing n* within ±1 immediately, so
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
            objects as the EBMs' nodes).
        clamp_state: clamped block states
        beta_range: (β_min, β_max) for the temperature range
        gibbs_steps_per_round: Gibbs sweeps between swap attempts
        initial_n: starting chain count. The default ``None`` runs a high pilot
            probe at ``max_chains`` for an unbiased Λ̂ (no initial guess needed);
            pass an int to start there instead.
        target_acceptance: desired per-pair swap acceptance rate. Default 0.5 —
            the round-trip-optimal rejection r* = 1/2 (N* ≈ 2Λ; Syed et al.), not
            the 0.77 from the reversible-PT literature.
        rounds_per_probe: rounds per probe (and for the final production probe)
        n_tune_per_probe: schedule tuning iterations for the final probe
        max_iters: maximum discovery iterations
        min_chains: floor on chain count
        max_chains: ceiling on chain count
        lambda_rtol: relative tolerance for Λ stabilization (default 5%)
        safety_margin: small fractional pad on N* (default 0.05) covering residual
            barrier bias and ELE-assumption violations; 0.0 gives the bare
            round-trip-optimal count

    Returns:
        dict with keys:
            n_chains: final recommended chain count
            betas: optimized schedule at that chain count
            Lambda: conservative (max) barrier estimate
            Lambda_raw: last raw estimate (may be lower than Lambda)
            target_acceptance: the target used
            converged_reason: "chain_count" | "lambda_stable" | "max_iters"
            history: list of per-probe dicts
    """
    source = _ChainSource(ebm_factory, program_factory, ebm, program)

    if init_factory is None:
        raise ValueError("init_factory is required.")
    if clamp_state is None:
        clamp_state = []

    r_target = max(1.0 - target_acceptance, 1e-3)
    min_chains = int(min_chains)
    max_chains = int(max_chains)
    max_probes = int(max_iters)

    def _clamp(n):
        return max(min_chains, min(max_chains, int(n)))

    # Resolve the device once for all probes. Chain counts vary across probes
    # (each recompiles regardless), but a single device avoids transfer thrash;
    # the conservative score uses the (pilot) starting chain count. Borderline
    # workloads can pass an explicit device.
    # Pilot at the budget ceiling: an over-resolved first probe gives an unbiased
    # Λ̂ (low-N probes saturate rejection rates and bias Λ̂ low), so the fixed
    # point lands within ±1 of N* immediately and converges in ~2 probes instead
    # of climbing. Measured: 1600-node grid 4→2 probes, −40% stage-1 wall.
    _pilot_n = initial_n if initial_n is not None else max_chains
    _meta_betas = jnp.linspace(beta_range[0], beta_range[1], 1)
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
        betas0 = jnp.linspace(beta_range[0], beta_range[1], n)
        # init_factory receives a programs list for free-block extraction; on the
        # template route every entry is the template program (identical
        # gibbs_spec) and no per-chain programs are constructed.
        ebms = source.ebms_for_init(betas0)
        programs = source.programs_for_init(n, ebms)
        inits = init_factory(n, ebms, programs)
        key, k_probe = jax.random.split(key)
        # Forward whichever route the caller used; tune_schedule re-dispatches
        # through its own _ChainSource. The concrete device (or None) bypasses its
        # heuristic, so probes never flip devices. Tuning is adaptive, so a
        # wrong-N probe still self-limits its rounds.
        _, stats = tune_schedule(
            k_probe,
            ebm_factory,
            program_factory,
            inits,
            clamp_state,
            n_rounds=rounds_per_probe,
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
        )
        rej = np.asarray(stats["rejection_rates"])
        out: dict[str, Any] = {
            "n": n,
            "Lambda_raw": float(np.sum(rej)),
            "rejection_rates": rej,
            "betas": np.asarray(stats["betas"]),
        }
        probed[n] = out
        return out

    if max_probes <= 0:
        n_final = _clamp(initial_n if initial_n is not None else min_chains)
        return {
            "n_chains": int(n_final),
            "betas": np.asarray(jnp.linspace(beta_range[0], beta_range[1], n_final)),
            "Lambda": 0.0,
            "Lambda_raw": 0.0,
            "target_acceptance": target_acceptance,
            "converged_reason": "max_iters",
            "history": history,
        }

    # --- N tuning (Syed et al. 2021, Sec. "Tuning N") -----------------------
    # Λ is a schedule invariant: Λ̂ = Σ rejection_rates ≈ Λ at any chain count
    # (each probe runs tune_schedule, which tunes the schedule to equi-
    # acceptance). Estimate Λ̂ at the current N, set N* = ceil(Λ̂·margin/r) + 1
    # (= 2Λ + 1 at r* = 1/2), and iterate this fixed point until N* settles.
    # Using the current-N estimate (not a running maximum) makes the result
    # independent of the starting N.
    margin = 1.0 + max(0.0, float(safety_margin))
    n = _clamp(initial_n) if initial_n is not None else _clamp(max_chains)
    lambda_raw = 0.0  # last current-N barrier estimate; drives N*
    lambda_max = 0.0  # running max, reported as the conservative Λ
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
        n_star = _clamp(int(np.ceil(lambda_raw * margin / r_target)) + 1)
        history.append(
            _probe_history_entry(
                len(history),
                n,
                lambda_raw,
                lambda_max,
                n_star,
                res["rejection_rates"],
                res["betas"],
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

    # Produce the returned schedule at the final count, reusing a cached probe
    # where possible. On chain_count convergence n_star is within 1 of the last
    # probed n (always cached), so when n_star itself was never probed, return n
    # instead of running a full extra probe — that probe would recompile the
    # round loop at a brand-new chain count (~1s) just to land on a count within
    # the convergence tolerance of one already in hand. (Whether n_star equals
    # the last probed n or is off by 1 is incidental to the Λ estimate, so this
    # also makes the probe count robust across equally-good colourings.)
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
            )
        )

    return {
        "n_chains": int(n_final),
        "betas": best_betas,
        "Lambda": float(lambda_max),
        "Lambda_raw": float(lambda_raw),
        "target_acceptance": target_acceptance,
        "converged_reason": reason,
        "history": history,
    }


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
        ebm_factory / program_factory: per-chain factories, or use ``ebm`` /
            ``program`` template objects.
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

        # Flatten the per-round cold-chain block states into a (rounds, nodes)
        # trace. ESS is per-column then median-reduced, so column order / node
        # type are irrelevant — no node-reordering needed. median (not min) is
        # the steadier order statistic across seeds.
        cold = [np.asarray(o)[:, 0] for o in stats["observations"]]
        trace = np.concatenate([c.reshape(c.shape[0], -1) for c in cold], axis=1)
        ess = effective_sample_size(trace)

        # (2) Per-round wall time — only the timing-based ("cost"/empirical)
        # selectors need it. The ELE selector is timing-free (it reads the
        # deterministic round-trip efficiency from this same run), so skip it.
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
        # Deterministic, timing-free: pick the smallest n_expl whose round-trip
        # efficiency (τ_obs/τ̂) reaches the ELE-adequacy knee. Reproducible across
        # runs (no wall-clock in the objective).
        best, history = _select_gibbs_steps_ele(
            probe,
            int(start_steps),
            int(max_steps),
            float(improve_tol),
            float(target_efficiency),
        )
    elif use_reuse_timing:
        # Doubling driven by ESS growth (deterministic — same key ⇒ same ESS ⇒
        # the probed set is fixed run-to-run), collecting each probe's reused
        # production timing. Then fit ONE cost line and pick the argmax objective,
        # requiring a >improve_tol gain to climb so the near-flat peak resolves to
        # the lower (cheaper) count instead of flipping on timing noise.
        history = []
        n = int(start_steps)
        while True:
            rec = probe(n)
            history.append(rec)
            if rec.get("efficiency_limiter") == "schedule":
                break  # schedule-limited: more local exploration cannot help
            if n >= int(max_steps):
                break
            if len(history) >= 2 and rec["ess_median"] <= history[-2]["ess_median"] * (
                1.0 + float(improve_tol)
            ):
                break  # ESS saturated: extra sweeps no longer decorrelate
            n *= 2
        c0, cs = _fit_cost_line(
            [r["n_expl"] for r in history], [r["t_round"] for r in history]
        )
        for r in history:
            tr = c0 + r["n_expl"] * cs
            r["t_round"] = tr
            r["objective"] = (
                r["ess_median"] / (rounds_per_probe * tr) if tr > 0 else 0.0
            )
        best = history[0]
        for r in history[1:]:
            if r["objective"] > best["objective"] * (1.0 + float(improve_tol)):
                best = r
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
