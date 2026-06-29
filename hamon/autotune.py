"""Full NRPT autotuning: orchestrate chain count, exploration, and schedule.

The one-call front door (``autotune`` / ``autosample``) and its result objects,
pure host orchestration over the tuners in ``hamon.tuning`` and the jitted core
in ``hamon.nrpt``.
"""

from __future__ import annotations

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
from hamon.tuning import tune_chains, tune_exploration, tune_schedule


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


@dataclass
class NRPTPlan:
    """A tuned NRPT configuration plus a warm cold-chain state.

    Returned by :func:`autotune`. Holds the discovered hyperparameters (N,
    schedule, n_expl) and an equilibrated cold-chain state, so :meth:`sample`
    can draw repeatedly and cheaply — no re-tuning, reusing the compiled loop.

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
    _cold_program: BlockSamplingProgram
    _warm_state: list
    _clamp_state: list
    _obs_block: Any

    def sample(
        self,
        key: jax.Array,
        n_samples: int,
        *,
        n_warmup: int = 0,
        steps_per_sample: int = 1,
    ) -> jax.Array:
        """Draw ``n_samples`` from the target (cold chain) — cheap and repeatable.

        Runs single-chain block Gibbs at the tuned cold β from the stored warm
        state (the established post-NRPT draw). Returns a ``(n_samples,
        n_nodes)`` array; call again with a fresh ``key`` for more, with no
        re-tuning.
        """
        from hamon.block_sampling import SamplingSchedule, sample_states

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
    seed_from_energy: bool = False,
    gibbs_steps_per_round: int | None = None,
    search_exploration: bool = False,
    max_exploration_steps: int = 8,
    cost_model: bool = True,
    select_by: str = "cost",
    target_efficiency: float = 0.9,
    rounds_per_probe: int = 400,
    n_tune: int = 4,
    n_polish: int = 2,
    n_rounds: int = 1000,
    compile_cache: bool | str = True,
    device: DeviceLike = "auto",
) -> NRPTPlan:
    """Autotune the full NRPT configuration: N, exploration count, and schedule.

    The one-call front door for solving a problem with hamon. Runs the
    dependency-ordered, cheap→expensive recipe and returns an :class:`NRPTPlan`
    you draw from with :meth:`NRPTPlan.sample`:

    1. **N** via :func:`tune_chains` at n_expl=1 (cheapest probes; Λ — hence N\\* —
       is invariant to n_expl).
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
        ebm / program: single template objects (temperature-linear mode), or
        ebm_factory / program_factory: per-chain factories.
        init_factory: ``(n_chains, ebms, programs) -> list`` of initial states.
        clamp_state: clamped block states.
        sample_nodes: nodes defining the column order of drawn samples (must be
            free nodes of the program). ``None`` (default) uses all free nodes in
            free-block order; pass the model's canonical node list to get samples
            in that order (single node type only).
        beta_range: ``(β_min, β_max)`` temperature range.
        target_acceptance: per-pair swap acceptance target for the N search.
            Default 0.5 — the round-trip-optimal r* = 1/2 (N* ≈ 2Λ; Syed et al.).
        min_chains / max_chains / initial_n: N-search bounds / start.
        seed_from_energy: seed the chain-count search from a cheap energy-variance
            Λ̂ (no PT ladder) so it converges in one probe; see
            :func:`hamon.tuning.tune_chains`. Same discovered N, fewer compiles.
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
        device: where to run; resolved once and reused across every stage.

    Returns:
        An :class:`NRPTPlan`.
    """
    from hamon.block_management import Block
    from hamon.device import enable_persistent_compile_cache

    if init_factory is None:
        raise ValueError("init_factory is required.")
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

    # --- Stage 1: chain count at n_expl = 1 ---
    disc = tune_chains(
        k_chains,
        ebm_factory,
        program_factory,
        init_factory,
        clamp_state,
        beta_range=beta_range,
        gibbs_steps_per_round=1,
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
    if gibbs_steps_per_round is not None:
        if int(gibbs_steps_per_round) < 1:
            raise ValueError("gibbs_steps_per_round must be >= 1.")
        n_expl = int(gibbs_steps_per_round)
    elif search_exploration and max_exploration_steps > 1:
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
    else:
        n_expl = _default_gibbs_steps(dev)

    # --- Stage 3: schedule polish at (N, n_expl) + warm cold state ---
    # The production run uses n_rounds (not the short probe budget): it both
    # equilibrates the warm cold state and measures a representative round-trip
    # rate. A round trip needs >= ~2N rounds, so a short window badly
    # underestimates tau_obs / efficiency for large N.
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
    obs_block = Block(out_nodes)

    rt_diag = polish_stats.get("round_trip_diagnostics")
    total_round_trips = (
        int(np.sum(np.asarray(rt_diag["round_trips_per_chain"])))
        if rt_diag is not None
        else None
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
        _cold_program=cold_program,
        _warm_state=warm_cold,
        _clamp_state=clamp_state,
        _obs_block=obs_block,
    )


def autosample(
    key: jax.Array,
    *,
    n_samples: int,
    n_warmup: int = 0,
    steps_per_sample: int = 1,
    **autotune_kwargs,
) -> tuple[jax.Array, AutotuneReport]:
    """One-shot: :func:`autotune` then draw — returns ``(samples, report)``.

    The convenience entry for "give me samples." Forwards all keyword arguments
    to :func:`autotune` (``ebm``/``program`` or factories, ``init_factory``,
    ``beta_range``, ``device``, …), then draws ``n_samples`` from the tuned plan.
    For repeated draws from one tuned configuration, call :func:`autotune` once
    and reuse :meth:`NRPTPlan.sample`.
    """
    k_tune, k_draw = jax.random.split(key)
    plan = autotune(k_tune, **autotune_kwargs)
    samples = plan.sample(
        k_draw, n_samples, n_warmup=n_warmup, steps_per_sample=steps_per_sample
    )
    return samples, plan.report
