"""Full NRPT autotuning: orchestrate chain count, exploration, and schedule.

The one-call front door (``autotune`` / ``autosample``) and its result objects,
pure host orchestration over the tuners in ``hamon.tuning`` and the jitted core
in ``hamon.nrpt``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hamon.block_sampling import BlockSamplingProgram
from hamon.device import DeviceLike, resolve_entry_device
from hamon.models.ebm import AbstractEBM
from hamon.nrpt import _ChainSource
from hamon.tuning import (
    _require_proper_beta_start,
    tune_chains,
    tune_exploration,
    tune_schedule,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Full autotuning: orchestrate N, exploration, and schedule
# ---------------------------------------------------------------------------

# Deterministic local-exploration count used when the exploration search is off.
# The ESS-per-wall-second objective is flat across n_expl 2-8 on a dispatch-bound
# accelerator (extra Gibbs sweeps per round are nearly free there), so a fixed
# mid-range value captures ~all the benefit AND is reproducible across runs —
# unlike a wall-timed search, whose argmax wanders the flat region with the GPU's
# clock/thermal state. CPU is compute-bound (cost grows ~linearly with n_expl),
# so 1 is optimal there.
_ACCELERATOR_DEFAULT_GIBBS_STEPS = 4


def _default_gibbs_steps(dev) -> int:
    """Device-calibrated n_expl when the exploration search is off."""
    platform = getattr(dev, "platform", "cpu") if dev is not None else "cpu"
    return _ACCELERATOR_DEFAULT_GIBBS_STEPS if platform != "cpu" else 1


@dataclass
class AutotuneReport:
    """Diagnostics from an :func:`autotune` run.

    Attributes:
        n_chains: discovered chain count N.
        gibbs_steps_per_round: discovered local-exploration count n_expl.
        Lambda: estimated global communication barrier Λ.
        betas: the final tuned β ladder.
        device: the resolved device (string) or ``None``.
        chain_history: per-probe records from the N search (:func:`tune_chains`).
        exploration: the :func:`tune_exploration` result dict, or ``None`` when
            the n_expl search was skipped.
        round_trip_diagnostics: round-trip summary from the final production run.
        total_round_trips: total completed round trips observed during the final
            production run (summed across chains), or ``None``.
        production_rounds: number of rounds the production run used (the window
            ``total_round_trips`` and ``tau_observed`` were measured over).
    """

    n_chains: int
    gibbs_steps_per_round: int
    Lambda: float
    betas: np.ndarray
    device: str | None
    chain_history: list
    exploration: dict | None
    round_trip_diagnostics: dict | None
    total_round_trips: int | None = None
    production_rounds: int | None = None

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [
            "AUTOTUNE: "
            f"N={self.n_chains}  n_expl={self.gibbs_steps_per_round}  "
            f"Lambda={self.Lambda:.3f}  device={self.device}",
        ]
        if self.exploration is not None:
            ess_sec = self.exploration.get("objective")
            lines.append(
                f"  exploration: chose n_expl={self.gibbs_steps_per_round} "
                f"(ESS/sec={ess_sec:.1f}, "
                f"t_round={self.exploration.get('t_round', 0.0) * 1e3:.3f} ms)"
            )
        rt = self.round_trip_diagnostics
        if rt is not None:
            trips = (
                f"{self.total_round_trips}"
                if self.total_round_trips is not None
                else "?"
            )
            window = (
                f" over {self.production_rounds} rounds"
                if self.production_rounds is not None
                else ""
            )
            lines.append(
                f"  round trips: {trips}{window}  "
                f"tau_obs={float(rt['tau_observed']):.4f}  "
                f"tau_pred={float(rt['tau_predicted']):.4f}  "
                f"efficiency={float(rt['efficiency']):.3f}"
            )
        return "\n".join(lines)


def _cold_trace_from_observations(observations: list, col_perm: jax.Array) -> jax.Array:
    """Stack a cold-chain ``NRPTStateObserver`` output into ``(T, n_nodes)``.

    ``observations`` is one array per free block, each ``(T, 1, *block_shape)``
    — the leading axis is the round, the size-1 axis is the single observed
    (cold) chain. Concatenate the blocks in free-block order, then permute the
    columns into the caller's ``sample_nodes`` order via ``col_perm``.
    """
    cols = [jnp.reshape(o[:, 0], (o.shape[0], -1)) for o in observations]
    flat = jnp.concatenate(cols, axis=1)
    return flat[:, col_perm]


@dataclass
class NRPTPlan:
    """A tuned NRPT configuration plus a warm, equilibrated ladder.

    Returned by :func:`autotune`. Holds the discovered hyperparameters (N,
    schedule, n_expl) and the warm state of *every* chain, so :meth:`sample`
    can draw repeatedly with no re-tuning, reusing the compiled round loop.

    Attributes:
        n_chains / betas / gibbs_steps_per_round / Lambda: the tuned config.
        device: the resolved device (or ``None``).
        report: the :class:`AutotuneReport`.
    """

    n_chains: int
    betas: np.ndarray
    gibbs_steps_per_round: int
    Lambda: float
    device: Any
    report: AutotuneReport
    # Tempered-draw state (the correct, multimodal-safe sampler):
    _source: _ChainSource
    _betas_dev: jax.Array
    _warm_ladder: list
    _col_perm: jax.Array
    # Single-chain fast-draw state (tempered=False):
    _cold_program: BlockSamplingProgram
    _warm_state: list
    _clamp_state: list
    _obs_block: Any
    # When set (= max_chains under chain masking), the tempered draw pads its
    # ladder to this length and records the cold chain at a traced live index, so
    # draws at different discovered N share ONE compiled observer round loop.
    _pad_draw: int | None = None

    def sample(
        self,
        key: jax.Array,
        n_samples: int,
        *,
        n_warmup: int = 0,
        steps_per_sample: int = 1,
        tempered: bool = True,
    ) -> jax.Array:
        """Draw ``n_samples`` from the target (cold chain).

        ``tempered=True`` (default) draws the **tempered cold chain**: it runs
        the tuned NRPT ladder from the stored warm ladder and records the cold
        chain each round. Because tempering stays active during the draw, the
        hot chains keep crossing energy barriers and DEO swaps carry that mixing
        down to the cold chain — so the samples represent *all* modes of a
        multimodal target. This is the correct sampler whenever the cold chain
        does not mix on its own (the case NRPT exists for). Returns a
        ``(n_samples, n_nodes)`` array in ``sample_nodes`` column order; call
        again with a fresh ``key`` for independent draws, with no re-tuning (the
        compiled round loop is reused across same-shape draws).

        ``tempered=False`` runs plain single-chain block Gibbs at the tuned cold
        β from one warm cold state — cheaper, but a single chain **cannot cross
        barriers**, so it mode-collapses on a multimodal target. Use it only
        when the cold chain is known to mix on its own (unimodal / low barrier).
        A warning is emitted if the tuning run round-tripped, which is direct
        evidence the target needs tempering to mix.

        ``steps_per_sample`` is the number of rounds between recorded samples
        (thinning); ``n_warmup`` leading rounds are discarded before recording.
        """
        if tempered:
            return self._sample_tempered(
                key, int(n_samples), int(n_warmup), int(steps_per_sample)
            )
        return self._sample_cold_gibbs(key, n_samples, n_warmup, steps_per_sample)

    def _sample_tempered(
        self, key: jax.Array, n_samples: int, n_warmup: int, steps_per_sample: int
    ) -> jax.Array:
        from hamon.nrpt import nrpt
        from hamon.observers import ColdIndexObserver, NRPTStateObserver

        steps = max(1, steps_per_sample)
        n_total = n_warmup + n_samples * steps
        dev = self.device if self.device is not None else "auto"
        ebms, programs = self._source.nrpt_args(self._betas_dev)
        # Chain-masked draw: pad the ladder to max_chains and record the live cold
        # chain (absolute index n_chains-1 of the padded ladder) via a traced
        # index, so draws at different discovered N reuse ONE compiled observer
        # round loop. Bit-identical to the unpadded draw on the live prefix
        # (threefry key/uniform streams are prefix-stable). Off (pad=None) ⇒ the
        # original unpadded static -1 observer.
        pad = self._pad_draw
        observer = (
            ColdIndexObserver(self.n_chains - 1)
            if pad is not None
            else NRPTStateObserver(chain_indices=(-1,))
        )
        _, stats = nrpt(
            key,
            ebms,
            programs,
            self._warm_ladder,
            self._clamp_state,
            n_total,
            self.gibbs_steps_per_round,
            betas=self._betas_dev,
            track_round_trips=False,
            observer=observer,
            device=dev,
            pad_chains_to=pad,
        )
        trace = _cold_trace_from_observations(stats["observations"], self._col_perm)
        # Discard warmup, thin by steps_per_sample, keep exactly n_samples rows.
        return trace[n_warmup::steps][:n_samples]

    def _sample_cold_gibbs(
        self,
        key: jax.Array,
        n_samples: int,
        n_warmup: int,
        steps_per_sample: int,
    ) -> jax.Array:
        from hamon.block_sampling import SamplingSchedule, sample_states

        # Flag the silent-collapse risk of a decoupled cold chain, keyed on the
        # tuning run's round-trip evidence. warnings.warn (not logging) so it is
        # visible by default and, under the default filter, fires once per
        # call-site rather than once per draw.
        rt = self.report.total_round_trips if self.report is not None else None
        if rt:
            warnings.warn(
                f"NRPTPlan.sample(tempered=False): the tuning run round-tripped "
                f"{rt} time(s), so the cold target is multimodal and a single "
                f"decoupled chain will mode-collapse. Use tempered=True (the "
                f"default) for correct samples.",
                stacklevel=3,
            )
        elif rt == 0:
            # PT itself stalled — tempering never crossed the barrier, so even a
            # tempered draw may miss modes; flag the ambiguity here too.
            warnings.warn(
                "NRPTPlan.sample(tempered=False): the tuning run recorded 0 round "
                "trips — the ladder never crossed the barrier, so samples may "
                "under-represent modes regardless of tempered. Raise max_chains "
                "or n_rounds; otherwise prefer tempered=True.",
                stacklevel=3,
            )
        schedule = SamplingSchedule(n_warmup, n_samples, steps_per_sample)
        dev = self.device if self.device is not None else "auto"
        raw = sample_states(
            key,
            self._cold_program,
            schedule,
            self._warm_state,
            self._clamp_state,
            [self._obs_block],
            device=dev,
        )
        return raw[0]


# The production draw's StateObserver carries this Block as a jit static, and
# Blocks compare by identity — a fresh Block over the same nodes per autotune
# call would retrace and recompile the draw kernel every call. Keyed on node
# identities; entries pin their nodes (no id reuse while live), bounded FIFO —
# the same pattern as the factor-block caches in hamon.models.
_OBS_BLOCK_CACHE: dict = {}
_OBS_BLOCK_CACHE_MAX = 32


def _obs_block(out_nodes: list):
    from hamon.block_management import Block

    key = tuple(map(id, out_nodes))
    hit = _OBS_BLOCK_CACHE.get(key)
    if hit is not None:
        return hit[1]
    blk = Block(out_nodes)
    if len(_OBS_BLOCK_CACHE) >= _OBS_BLOCK_CACHE_MAX:
        _OBS_BLOCK_CACHE.pop(next(iter(_OBS_BLOCK_CACHE)))
    _OBS_BLOCK_CACHE[key] = (tuple(out_nodes), blk)
    return blk


def autotune(
    key: jax.Array,
    *,
    ebm: AbstractEBM | None = None,
    program: BlockSamplingProgram | None = None,
    ebm_factory: Callable | None = None,
    program_factory: Callable | None = None,
    init_factory: Callable,
    clamp_state: list | None = None,
    sample_nodes: Sequence | None = None,
    beta_range: tuple[float, float] = (0.0, 1.0),
    target_acceptance: float = 0.5,
    min_chains: int = 3,
    max_chains: int = 128,
    initial_n: int | None = None,
    seed_from_energy: bool = True,
    gibbs_steps_per_round: int | None = None,
    search_exploration: bool = False,
    max_exploration_steps: int = 8,
    cost_model: bool = True,
    select_by: str = "cost",
    target_efficiency: float = 0.9,
    rounds_per_probe: int = 400,
    n_tune: int = 16,
    n_polish: int = 2,
    n_rounds: int = 1000,
    compile_cache: bool | str = True,
    pad_probes: bool | None = None,
    device: DeviceLike = "auto",
) -> NRPTPlan:
    """Autotune the full NRPT configuration: N, exploration count, and schedule.

    The one-call front door for solving a problem with hamon. Runs the
    dependency-ordered, cheap→expensive recipe and returns an :class:`NRPTPlan`
    you draw from with :meth:`NRPTPlan.sample`:

    1. **N** via :func:`tune_chains`, probed at the final n_expl when it is
       already known (pinned or the deterministic device default) so stage 3
       reuses stage 1's compiled round loop — the biggest cold-run compile.
       When the n_expl search is on, probes run at n_expl=1 instead (cheapest;
       Λ — hence N\\* — is invariant to n_expl).
    2. **n_expl** — by default a deterministic device-calibrated count
       (accelerator → a fixed mid-range value, CPU → 1): reproducible across runs
       and ~free, since the ESS-per-wall-second objective is flat in n_expl on a
       dispatch-bound accelerator. Pin it explicitly with ``gibbs_steps_per_round``
       (e.g. a value calibrated for your hardware), or pass
       ``search_exploration=True`` to tune it via :func:`tune_exploration` at the
       fixed N, reusing the schedule from step 1 (the equi-acceptance schedule is
       n_expl-invariant, so this needs no re-tuning and never re-discovers N); the
       ``"cost"`` search maximizes ESS per *measured* wall-second but its pick is
       not reproducible across runs (it depends on the machine's clock state), so
       it is best used as a one-time per-hardware calibration.
    3. **Schedule polish** via :func:`tune_schedule` at the chosen (N, n_expl),
       which also leaves an equilibrated warm cold-chain state.

    The multi-probe search recompiles per chain count and per n_expl, so by
    default the **persistent compilation cache** is enabled (``compile_cache``)
    to amortize those compiles across probes and runs.

    Pass either a single template ``ebm`` + ``program`` (temperature-linear mode)
    or per-chain ``ebm_factory`` + ``program_factory``, exactly as the individual
    tuners accept. ``init_factory(n_chains, ebms, programs) -> list`` builds one
    initial state per chain at the discovered N.

    Args:
        key: PRNG key.
        ebm: single EBM template (temperature-linear mode).
        program: single sampling-program template (temperature-linear mode).
        ebm_factory: per-chain EBM factory (alternative to ``ebm``).
        program_factory: per-chain program factory (alternative to ``program``).
        init_factory: ``(n_chains, ebms, programs) -> list`` of initial states.
        clamp_state: clamped block states.
        sample_nodes: nodes defining the column order of drawn samples (must be
            free nodes of the program). ``None`` (default) uses all free nodes in
            free-block order; pass the model's canonical node list to get samples
            in that order (single node type only).
        beta_range: ``(β_min, β_max)`` temperature range.
        target_acceptance: per-pair swap acceptance target for the N search.
            Default 0.5 — the round-trip-optimal r* = 1/2 (N* ≈ 2Λ; Syed et al.).
        min_chains: lower bound for the chain-count (N) search.
        max_chains: upper bound for the chain-count (N) search; also the pilot size.
        initial_n: starting chain count for the search; ``None`` runs a pilot at
            ``max_chains``.
        seed_from_energy: seed the chain-count search from a cheap energy-variance
            Λ̂ (no PT ladder) so it converges in one probe — but only when local
            exploration mixes; a Gelman–Rubin R̂ check falls back to the robust
            ``max_chains`` pilot on glassy targets, so it never under-provisions.
            Same discovered N, fewer compiles when it applies. Default ``True``;
            pass ``False`` to always run the pilot; see
            :func:`hamon.tuning.tune_chains`.
        gibbs_steps_per_round: pin n_expl to this value, skipping both the device
            default and the search (step 2). For hardware you have already
            calibrated. ``None`` (default) uses the device default or the search.
        search_exploration: tune n_expl by a wall-timed search (step 2). Default
            ``False`` uses a deterministic device-calibrated n_expl (reproducible
            across runs); ``True`` runs :func:`tune_exploration`. Ignored when
            ``gibbs_steps_per_round`` is set.
        max_exploration_steps: ceiling for the n_expl doubling search (when
            ``search_exploration=True``).
        select_by: for ``search_exploration=True`` — ``"cost"`` (default)
            maximizes cold-chain ESS per wall-second; ``"ele"`` picks n_expl by the
            deterministic round-trip efficiency knee (reproducible, but optimizes
            index-process mixing rather than sample ESS). See
            :func:`tune_exploration`.
        target_efficiency: ELE-adequacy threshold for ``select_by="ele"``.
        cost_model: for the ``select_by="cost"`` path, fit one n_expl cost line
            from reused production timings instead of timing each probe
            separately; see :func:`tune_exploration`.
        rounds_per_probe: rounds per tuning/exploration probe (the cheap search
            budget).
        n_tune: schedule-tuning phases per N probe.
        n_polish: schedule-tuning phases in the final polish.
        n_rounds: rounds for the final production run — equilibrates the warm
            cold state and is the window the reported round-trip rate / efficiency
            are measured over. Should be ``≫ 2·N`` for a representative rate; the
            default (1000) suits the autotuned chain counts.
        compile_cache: ``True`` enables the persistent compile cache at the
            default path, a ``str`` enables it at that path, ``False`` leaves
            placement untouched. See
            :func:`hamon.enable_persistent_compile_cache`.
        pad_probes: chain-mask the stage-1 probes — pad every probe's round
            loop to ``max_chains`` so all probes share ONE compiled loop
            instead of recompiling per chain count (see
            :func:`hamon.tuning.tune_chains`). ``None`` (default) enables it
            on an accelerator in template mode (where the padding Gibbs work
            is ~free) and disables it on CPU or the factory route. Stages 2-3
            and the production draw always run unpadded.
        device: where to run; resolved once and reused across every stage.

    Returns:
        An :class:`NRPTPlan`.
    """
    from hamon.device import enable_persistent_compile_cache

    if init_factory is None:
        raise ValueError("init_factory is required.")
    _require_proper_beta_start(beta_range[0], ebm)
    if compile_cache:
        enable_persistent_compile_cache(
            compile_cache if isinstance(compile_cache, str) else None
        )
    clamp_state = clamp_state or []
    source = _ChainSource(ebm_factory, program_factory, ebm, program)

    # Resolve the device once for every stage. Match tune_chains' pilot (the
    # max_chains ceiling) so the CPU/GPU sizing heuristic scores the same chain
    # count the first probe runs.
    _pilot_n = initial_n if initial_n is not None else max_chains
    _meta_betas = jnp.linspace(beta_range[0], beta_range[1], 1)
    dev = resolve_entry_device(
        device,
        n_chains=max(min_chains, min(max_chains, _pilot_n)),
        n_nodes=source.metadata_free_nodes(_meta_betas, device),
        arrays=(key,),
    )
    source.device_put_template(dev)

    k_chains, k_expl, k_polish = jax.random.split(key, 3)

    # --- Stage 1: chain count ---
    # Probe at the final n_expl whenever it is already known (pinned, or the
    # deterministic device default because the search is off). The scan length
    # doesn't change compile time and extra sweeps are ~free on a
    # dispatch-bound accelerator, while stage 3 then reuses stage 1's compiled
    # round loop instead of recompiling it at a new n_expl — the single
    # biggest cold-run compile. The wall-timed search still probes at
    # n_expl=1 (cheapest; Λ — hence N* — is invariant to n_expl).
    if gibbs_steps_per_round is not None:
        if int(gibbs_steps_per_round) < 1:
            raise ValueError("gibbs_steps_per_round must be >= 1.")
        probe_n_expl = int(gibbs_steps_per_round)
    elif search_exploration and max_exploration_steps > 1:
        probe_n_expl = 1
    else:
        probe_n_expl = _default_gibbs_steps(dev)

    # Chain masking default: on for accelerator + template mode (padding Gibbs
    # work is ~free there and masking needs temperature-linear β scaling); off
    # on CPU (padding is real compute) and on the factory route.
    if pad_probes is None:
        platform = getattr(dev, "platform", "cpu") if dev is not None else "cpu"
        pad_probes = bool(platform != "cpu" and ebm is not None and program is not None)
    pad_probes = bool(pad_probes)

    disc = tune_chains(
        k_chains,
        ebm_factory,
        program_factory,
        init_factory,
        clamp_state,
        beta_range=beta_range,
        gibbs_steps_per_round=probe_n_expl,
        target_acceptance=target_acceptance,
        rounds_per_probe=rounds_per_probe,
        n_tune_per_probe=n_tune,
        min_chains=min_chains,
        max_chains=max_chains,
        initial_n=initial_n,
        seed_from_energy=seed_from_energy,
        ebm=ebm,
        program=program,
        device=dev,
        pad_probes=pad_probes,
    )
    n_chains = int(disc["n_chains"])
    Lambda = float(disc["Lambda"])
    betas0 = jnp.asarray(disc["betas"])

    # Initial states at the discovered N (reused by stages 2 and 3).
    ebms_init = source.ebms_for_init(betas0)
    programs_init = source.programs_for_init(n_chains, ebms_init)
    init_states = init_factory(n_chains, ebms_init, programs_init)

    # --- Stage 2: exploration count at fixed N, reusing the schedule ---
    # Precedence: an explicit gibbs_steps_per_round pins n_expl (skip stage 2);
    # else a wall-timed search if opted in; else a deterministic device-calibrated
    # default (reproducible and ~free, since the ESS/sec objective is flat in
    # n_expl on a dispatch-bound accelerator).
    exploration: dict | None = None
    if gibbs_steps_per_round is not None or not (
        search_exploration and max_exploration_steps > 1
    ):
        n_expl = probe_n_expl
    else:
        exploration = tune_exploration(
            k_expl,
            ebm_factory,
            program_factory,
            init_states,
            clamp_state,
            initial_betas=betas0,
            start_steps=1,
            max_steps=max_exploration_steps,
            rounds_per_probe=rounds_per_probe,
            cost_model=cost_model,
            select_by=select_by,
            target_efficiency=target_efficiency,
            fixed_schedule=betas0,
            ebm=ebm,
            program=program,
            device=dev,
        )
        n_expl = int(exploration["gibbs_steps_per_round"])

    # --- Stage 3: schedule polish at (N, n_expl) + warm cold state ---
    # The production run uses n_rounds (not the short probe budget): it both
    # equilibrates the warm cold state and measures a representative round-trip
    # rate. A round trip needs >= ~2N rounds, so a short window badly
    # underestimates tau_obs / efficiency for large N.
    # When probes were masked, the polish + production run masked too: the same
    # padded round loop then serves every stage (one big compile total), instead
    # of stage 3 re-compiling an exact-shape loop the probes no longer built.
    warm_states, polish_stats = tune_schedule(
        k_polish,
        ebm_factory,
        program_factory,
        init_states,
        clamp_state,
        n_rounds=n_rounds,
        gibbs_steps_per_round=n_expl,
        initial_betas=betas0,
        n_tune=n_polish,
        rounds_per_tune=rounds_per_probe,
        ebm=ebm,
        program=program,
        device=dev,
        pad_chains_to=max_chains if pad_probes else None,
    )
    betas = jnp.asarray(polish_stats["betas"])
    warm_cold = warm_states[-1]  # cold chain = highest β

    # Build the cold-β program for the production draw (real β_cold weights, not
    # the temperature-linear template).
    cold_beta = float(np.asarray(betas)[-1])
    if ebm is not None and program is not None:
        cold_program = program.with_ebm(ebm.with_beta(jnp.asarray(cold_beta)))
    else:
        assert ebm_factory is not None and program_factory is not None
        cold_ebm = ebm_factory(jnp.asarray([cold_beta]))[0]
        cold_program = program_factory([cold_ebm])[0]

    # Output column order: caller-supplied (e.g. the model's original node
    # order) or, by default, all free nodes in free-block (colour) order.
    if sample_nodes is not None:
        out_nodes = list(sample_nodes)
    else:
        out_nodes = [n for b in cold_program.gibbs_spec.free_blocks for n in b.nodes]
    if len({type(n) for n in out_nodes}) > 1:
        raise NotImplementedError(
            "autotune sampling currently supports single-node-type models; "
            "draw manually from plan with sample_states for mixed-type models."
        )
    obs_block = _obs_block(out_nodes)

    # Column permutation mapping the tempered draw's free-block-order cold-chain
    # observations to ``out_nodes`` order (identity when out_nodes is the default
    # free-block order). Keyed by node identity — the free nodes are shared
    # across every chain's program, so this is stable across the ladder.
    flat_nodes = [n for b in cold_program.gibbs_spec.free_blocks for n in b.nodes]
    _flat_pos = {id(n): j for j, n in enumerate(flat_nodes)}
    try:
        col_perm = jnp.asarray([_flat_pos[id(n)] for n in out_nodes], dtype=jnp.int32)
    except KeyError as exc:
        raise ValueError("sample_nodes must all be free nodes of the program.") from exc

    rt_diag = polish_stats.get("round_trip_diagnostics")
    total_round_trips = (
        int(np.sum(np.asarray(rt_diag["round_trips_per_chain"])))
        if rt_diag is not None
        else None
    )
    if total_round_trips == 0:
        logger.warning(
            "autotune: the production run recorded 0 round trips — the cold "
            "chain is not crossing barriers, so samples (tempered or not) may "
            "under-represent some modes. Consider raising max_chains or n_rounds."
        )
    report = AutotuneReport(
        n_chains=n_chains,
        gibbs_steps_per_round=n_expl,
        Lambda=Lambda,
        betas=np.asarray(betas),
        device=str(dev) if dev is not None else None,
        chain_history=disc["history"],
        exploration=exploration,
        round_trip_diagnostics=rt_diag,
        total_round_trips=total_round_trips,
        production_rounds=n_rounds,
    )
    return NRPTPlan(
        n_chains=n_chains,
        betas=np.asarray(betas),
        gibbs_steps_per_round=n_expl,
        Lambda=Lambda,
        device=dev,
        report=report,
        _source=source,
        _betas_dev=betas,
        _warm_ladder=warm_states,
        _col_perm=col_perm,
        _cold_program=cold_program,
        _warm_state=warm_cold,
        _clamp_state=clamp_state,
        _obs_block=obs_block,
        # Mask the tempered draw exactly when the probes were masked, so the whole
        # pipeline (probes + polish + production + draw) shares one padded ladder
        # length and repeated/varying-N draws reuse the observer round loop.
        _pad_draw=(max_chains if pad_probes else None),
    )


def autosample(
    key: jax.Array,
    *,
    n_samples: int,
    n_warmup: int = 0,
    steps_per_sample: int = 1,
    tempered: bool = True,
    **autotune_kwargs,
) -> tuple[jax.Array, AutotuneReport]:
    """One-shot: :func:`autotune` then draw — returns ``(samples, report)``.

    The convenience entry for "give me samples." Forwards all keyword arguments
    to :func:`autotune` (``ebm``/``program`` or factories, ``init_factory``,
    ``beta_range``, ``device``, …), then draws ``n_samples`` from the tuned plan.
    For repeated draws from one tuned configuration, call :func:`autotune` once
    and reuse :meth:`NRPTPlan.sample`.

    ``tempered=True`` (default) draws the tempered cold chain, which represents
    all modes of a multimodal target; pass ``tempered=False`` only when the cold
    chain is known to mix on its own. See :meth:`NRPTPlan.sample`.
    """
    k_tune, k_draw = jax.random.split(key)
    plan = autotune(k_tune, **autotune_kwargs)
    samples = plan.sample(
        k_draw,
        n_samples,
        n_warmup=n_warmup,
        steps_per_sample=steps_per_sample,
        tempered=tempered,
    )
    return samples, plan.report
