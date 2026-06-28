"""Non-Reversible Parallel Tempering with vectorized swaps.

Based on Syed et al. (2021), "Non-Reversible Parallel Tempering:
a Scalable Highly Parallel MCMC Scheme" (arXiv:1905.02939).

Exploits temperature-linearity (E_β = β·E_base) for single-eval-per-chain
swap decisions. Adaptive schedule optimization (Algorithm 4) equalizes
rejection rates. Optional energy caching with boundary-only deltas for
rectangular block partitions.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from hamon._nrpt_energy import _compute_base_energies, _make_reference_ebm
from hamon._nrpt_schedule import _pchip_interp
from hamon._nrpt_swap import _make_swap_branch
from hamon.block_sampling import _run_blocks, BlockSamplingProgram
from hamon.device import (
    DeviceLike,
    free_node_count,
    resolve_entry_device,
    tree_device_put,
)
from hamon.models.ebm import AbstractEBM
from hamon.observers import AbstractNRPTObserver
from hamon.round_trips import (
    init_index_state,
    round_trip_summary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_factories(
    ebm_factory: Callable | None,
    program_factory: Callable | None,
    ebm: AbstractEBM | None,
    program: BlockSamplingProgram | None,
) -> tuple[Callable, Callable]:
    """Resolve (ebm_factory, program_factory) or (ebm, program) into callables."""
    if ebm_factory is None and program_factory is None:
        if ebm is None or program is None:
            raise ValueError(
                "Provide either (ebm_factory, program_factory) or (ebm=, program=)."
            )
        _ebm = ebm
        _prog = program

        def _make_ebms(betas):
            # Keep each β on its current device; jnp.array(float(b)) would force a
            # blocking device→host transfer per chain when betas is a device array.
            return [_ebm.with_beta(jnp.asarray(b)) for b in betas]

        def _make_programs(ebms):
            return [_prog.with_ebm(e) for e in ebms]

        return _make_ebms, _make_programs
    elif ebm_factory is not None and program_factory is not None:
        return ebm_factory, program_factory
    else:
        raise ValueError("Provide both ebm_factory and program_factory, or neither.")


class _ChainSource:
    """Uniform producer of per-chain EBM/program arguments for the template
    and factory routes of ``tune_schedule`` and ``tune_chains``.

    Template route (single ``ebm`` + ``program``, no factories):
    ``nrpt_args`` returns the same β = 1 base pair on every call, so nrpt's
    temperature-linear mode presents identical static structure to the jit
    cache and per-chain programs are never constructed. Factory route:
    ``nrpt_args`` materializes per-chain sequences via the user factories.
    """

    def __init__(
        self,
        ebm_factory: Callable | None,
        program_factory: Callable | None,
        ebm: AbstractEBM | None,
        program: BlockSamplingProgram | None,
    ):
        self.template_mode = (
            ebm_factory is None
            and program_factory is None
            and ebm is not None
            and program is not None
        )
        if self.template_mode:
            assert ebm is not None and program is not None
            # Rebase to β = 1 once; every later nrpt_args call hands back the
            # identical pair so the jitted round loop compiles exactly once.
            self._base_ebm = ebm.with_beta(jnp.asarray(1.0))
            self._base_program = program.with_ebm(self._base_ebm)
            self._template_program = program
        # Factories exist in both routes; the template route still uses them
        # for the cheap per-chain EBM list handed to init factories.
        self._make_ebms, self._make_programs = _resolve_factories(
            ebm_factory, program_factory, ebm, program
        )

    def nrpt_args(self, betas):
        """``(ebms, programs)`` arguments for ``nrpt`` at this β schedule."""
        if self.template_mode:
            return self._base_ebm, self._base_program
        ebms = self._make_ebms(betas)
        return ebms, self._make_programs(ebms)

    def ebms_for_init(self, betas):
        """Per-chain EBM list for init factories (cheap, no program builds)."""
        return self._make_ebms(betas)

    def programs_for_init(self, n_chains: int, ebms):
        """Per-chain program list for init factories.

        The template route returns the template program repeated (identical
        ``gibbs_spec``) instead of constructing ``n_chains`` programs.
        """
        if self.template_mode:
            return [self._template_program] * n_chains
        return self._make_programs(ebms)

    def metadata_free_nodes(self, betas, device: DeviceLike) -> int:
        """Free-node count for the device-routing work score.

        The factory route must build one throwaway single-chain program to
        read the size, so it only does this when the spec is ``"auto"`` —
        explicit devices ignore the score."""
        if self.template_mode:
            return free_node_count(self._base_program)
        if isinstance(device, str) and device.lower() == "auto":
            ebms = self._make_ebms(jnp.asarray(betas)[:1])
            return free_node_count(self._make_programs(ebms)[0])
        return 0

    def device_put_template(self, dev) -> None:
        """Commit the β = 1 template pair to ``dev`` once, outside the phase
        loop: every later ``nrpt_args`` call then returns arrays already on
        the device, nrpt's device_put hits its identity fast path, and the
        jit cache sees the same objects each phase. No-op on the factory
        route, whose per-phase arrays are moved by nrpt itself."""
        if dev is not None and self.template_mode:
            self._base_ebm, self._base_program = tree_device_put(
                (self._base_ebm, self._base_program), dev
            )


def _stack_pbi_across_chains(interaction_list: list) -> object:
    return jax.tree.map(
        lambda *leaves: (
            jnp.stack(leaves) if isinstance(leaves[0], jax.Array) else leaves[0]
        ),
        *interaction_list,
    )


def _make_pbi_in_axes(stacked_pbi):
    return jax.tree.map(
        lambda x: 0 if isinstance(x, jax.Array) else None,
        stacked_pbi,
    )


def _interaction_float_dtype(pbi) -> jnp.dtype:
    """The dtype the Gibbs kernel computes in: the result type of every
    floating-point interaction array. β values are cast to this so that
    enabling x64 in the host application does not promote a float32 model
    to float64 on device."""
    dtypes = [
        x.dtype
        for x in jax.tree.leaves(pbi)
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating)
    ]
    return jnp.result_type(*dtypes) if dtypes else jnp.dtype(jnp.float32)


# ---------------------------------------------------------------------------
# Adaptive schedule (Section 5.4)
# ---------------------------------------------------------------------------


@jax.jit
def optimize_schedule(rejection_rates: jax.Array, betas: jax.Array) -> jax.Array:
    """Equalize per-pair rejection rates by redistributing β values.

    Estimates the cumulative communication barrier Λ(β_k) = Σ_{i≤k} r_i, then
    places the new ladder at a regular grid of Λ (the equi-acceptance schedule of
    Syed et al. 2021, Algorithm 2: β*_k = Λ⁻¹(k/N · Λ)). The inverse Λ⁻¹ is taken
    as a **Fritsch–Carlson monotone cubic** of β against the cumulative barrier —
    smoother than the previous piecewise-linear inverse while staying monotone
    (no overshoot), which the paper recommends over linear interpolation.

    The result keeps the dtype of ``betas`` so repeated tuning phases do not
    drift to float64 when x64 is enabled."""
    betas = jnp.asarray(betas)
    rej = jnp.asarray(rejection_rates, dtype=betas.dtype)
    cum = jnp.concatenate([jnp.zeros(1, dtype=betas.dtype), jnp.cumsum(rej)])
    target = jnp.linspace(0.0, cum[-1], len(betas), dtype=betas.dtype)
    if betas.shape[0] >= 3:
        new = _pchip_interp(target, cum, betas)
    else:  # fewer than 3 knots: monotone cubic degenerates to linear
        new = jnp.interp(target, cum, betas)
    return new.at[0].set(betas[0]).at[-1].set(betas[-1])


# ---------------------------------------------------------------------------
# Core: NRPT round loop
# ---------------------------------------------------------------------------


class NRPTCarry(NamedTuple):
    """Scan carry for the NRPT inner loop."""

    key: jax.Array
    states: Any  # list of (n_chains, ...) arrays, one per free block
    accepted: jax.Array
    attempted: jax.Array
    idx_state: Any  # round-trip tracking dict
    base_E: jax.Array
    obs_carry: Any  # observer carry (None when no observer)


# Incremented each time _nrpt_rounds is (re)traced. Test instrumentation for
# verifying that repeated calls with identical static structure reuse the
# compiled executable instead of retracing.
_nrpt_rounds_trace_count = [0]


def _build_gibbs_runner(
    run_program: BlockSamplingProgram,
    clamp_state: list,
    gibbs_steps_per_round: int,
    base_pbi,
    chain_data,
    n_free_blocks: int,
):
    """Build the vmapped per-chain Gibbs kernel for one round.

    Temperature-linear mode (``base_pbi`` set) scales the shared β = 1
    interaction arrays by each chain's β inside the kernel; the per-chain
    sequence mode maps over the stacked interaction pytree. Returns the vmapped
    ``run_chains(gibbs_keys, states_free, chain_data)``.
    """
    null_ss = [None] * n_free_blocks

    def _run_one(gibbs_key, state_free, pbi):
        new_state, _, _ = _run_blocks(
            gibbs_key,
            run_program,
            state_free,
            clamp_state,
            gibbs_steps_per_round,
            null_ss,
            per_block_interactions=pbi,
        )
        return new_state

    if base_pbi is not None:
        # Temperature-linear mode: one shared base program at β = 1; scale
        # every interaction array by the chain's β inside the vmapped kernel.
        # Scalar-times-array commutes with the slicing done at program
        # construction, so for β-linear interactions this is bit-identical to
        # building per-chain programs, without materializing n_chains copies
        # of the weight tensors.
        _base_pbi = base_pbi
        chain_in_axes: object = 0

        def _chain_step(gibbs_key, state_free, chain_input):
            pbi_c = jax.tree.map(
                lambda x: chain_input * x if isinstance(x, jax.Array) else x,
                _base_pbi,
            )
            return _run_one(gibbs_key, state_free, pbi_c)
    else:
        chain_in_axes = _make_pbi_in_axes(chain_data)

        def _chain_step(gibbs_key, state_free, chain_input):
            return _run_one(gibbs_key, state_free, chain_input)

    return jax.vmap(
        _chain_step,
        in_axes=(0, [0] * n_free_blocks, chain_in_axes),
    )


def _build_swap_passes(
    betas: jax.Array,
    n_chains: int,
    n_free_blocks: int,
    track_round_trips: bool,
):
    """Build the even/odd DEO swap branches for ``lax.cond``.

    Returns ``(do_even, do_odd)`` — the two parity branches the round loop
    alternates between (single-pass DEO preserves non-reversibility).
    """
    n_pairs = n_chains - 1
    even_pairs = jnp.arange(0, n_pairs, 2, dtype=jnp.int32)
    odd_pairs = jnp.arange(1, n_pairs, 2, dtype=jnp.int32)
    att_even = jnp.zeros(n_pairs, dtype=jnp.int32).at[even_pairs].set(1)
    att_odd = jnp.zeros(n_pairs, dtype=jnp.int32).at[odd_pairs].set(1)
    base_perm = jnp.arange(n_chains, dtype=jnp.int32)

    swap_args = (betas, n_chains, n_pairs, n_free_blocks, base_perm, track_round_trips)
    do_even = _make_swap_branch(even_pairs, len(even_pairs), att_even, *swap_args)
    do_odd = _make_swap_branch(odd_pairs, len(odd_pairs), att_odd, *swap_args)
    return do_even, do_odd


def _build_energy_compute(
    energy_delta_fn: Callable | None,
    ebm_ref: AbstractEBM,
    beta_ref: jax.Array,
    base_spec,
    clamp_state: list,
):
    """Build the per-round base-energy update.

    With ``energy_delta_fn`` set, energies are advanced by boundary-only deltas
    off the cached value; otherwise they are recomputed from scratch each round.
    Signature: ``(new_states, old_states, cached_base_E) -> base_E``.
    """
    if energy_delta_fn is not None:
        _delta_fn = energy_delta_fn

        def _energy_cached(st_states, old_states, cached_bE):
            return cached_bE + _delta_fn(old_states, st_states)

        return _energy_cached

    def _energy_fresh(st_states, old_states, cached_bE):
        return _compute_base_energies(
            ebm_ref, beta_ref, base_spec, st_states, clamp_state
        )

    return _energy_fresh


def _build_observer_hooks(observer: AbstractNRPTObserver | None):
    """Build ``(init, step)`` observer hooks; no-ops when ``observer`` is None."""
    if observer is not None:
        return observer.init, observer

    def _obs_init():
        return None

    def _obs_step(stacked_states, base_energies, round_idx, carry):
        return carry, None

    return _obs_init, _obs_step


@eqx.filter_jit
def _nrpt_rounds(
    key: jax.Array,
    run_program: BlockSamplingProgram,
    ebm_ref: AbstractEBM,
    beta_ref: jax.Array,
    base_pbi,
    chain_data,
    stacked_states: list,
    clamp_state: list,
    betas: jax.Array,
    n_rounds: int | jax.Array,
    gibbs_steps_per_round: int,
    energy_delta_fn: Callable | None,
    observer: AbstractNRPTObserver | None,
    track_round_trips: bool,
) -> tuple[NRPTCarry, Any]:
    """The jitted NRPT round loop: vmapped Gibbs sweeps + DEO swaps.

    Module-level and ``eqx.filter_jit``-decorated so the compilation cache
    persists across calls: repeated invocations with the same program/observer
    structure and array shapes (e.g. the tuning phases of ``tune_schedule``)
    reuse the compiled executable. Arrays — including ``betas`` — are traced
    data, so schedule updates between phases do not retrigger compilation.

    Without an observer (the common case) the loop is a dynamic-trip-count
    ``lax.fori_loop`` and ``n_rounds`` arrives as a **traced** scalar, so the
    compiled executable is independent of the round count: the tuning batches
    and the production run of ``tune_schedule``, and discovery probes at the
    same chain count, all share a single compile. With an observer we must
    collect a per-round output stack, which needs ``lax.scan``'s static length,
    so ``n_rounds`` arrives as a static ``int`` and each distinct value compiles
    separately.
    """
    _nrpt_rounds_trace_count[0] += 1

    n_chains = betas.shape[0]
    n_free_blocks = len(stacked_states)
    n_pairs = n_chains - 1
    base_spec = run_program.gibbs_spec

    run_chains = _build_gibbs_runner(
        run_program,
        clamp_state,
        gibbs_steps_per_round,
        base_pbi,
        chain_data,
        n_free_blocks,
    )
    do_even, do_odd = _build_swap_passes(
        betas, n_chains, n_free_blocks, track_round_trips
    )
    energy_compute = _build_energy_compute(
        energy_delta_fn, ebm_ref, beta_ref, base_spec, clamp_state
    )
    observer_init, observer_step = _build_observer_hooks(observer)

    base_E = _compute_base_energies(
        ebm_ref, beta_ref, base_spec, stacked_states, clamp_state
    )

    # --- Scan body ------------------------------------------------------------
    def one_round(carry: NRPTCarry, round_idx):
        key, k_gibbs, k_swap = jax.random.split(carry.key, 3)

        old_states = carry.states
        gibbs_keys = jax.random.split(k_gibbs, n_chains)
        new_states = run_chains(gibbs_keys, carry.states, chain_data)

        bE = energy_compute(new_states, old_states, carry.base_E)

        new_states, acc, att, idx_st, pm = lax.cond(
            (round_idx & 1) == 0,
            do_even,
            do_odd,
            (new_states, carry.accepted, carry.attempted, k_swap, bE, carry.idx_state),
        )
        # Keep energies aligned with the states the swap just permuted. The
        # cached strategy needs this for the next round's deltas; observers
        # need it in both modes, since they receive (states, energies) pairs.
        bE = bE[pm]

        obs_carry, obs_out = observer_step(new_states, bE, round_idx, carry.obs_carry)
        return NRPTCarry(key, new_states, acc, att, idx_st, bE, obs_carry), obs_out

    init_carry = NRPTCarry(
        key=key,
        states=stacked_states,
        accepted=jnp.zeros(n_pairs, dtype=jnp.int32),
        attempted=jnp.zeros(n_pairs, dtype=jnp.int32),
        idx_state=init_index_state(n_chains),
        base_E=base_E,
        obs_carry=observer_init(),
    )

    # No observer ⇒ no per-round outputs needed, so use a dynamic-trip-count
    # fori_loop (n_rounds is a traced scalar here): the compile is reused across
    # round counts. With an observer, scan's static length is required to build
    # the stacked per-round output.
    if observer is None:

        def _loop_body(_round_idx, carry):
            new_carry, _ = one_round(carry, _round_idx)
            return new_carry

        final_carry = lax.fori_loop(0, n_rounds, _loop_body, init_carry)
        return final_carry, None

    return lax.scan(one_round, init_carry, jnp.arange(n_rounds))


# ---------------------------------------------------------------------------
# Run-input resolution and per-call helpers
# ---------------------------------------------------------------------------


def _acceptance_rate(accepted: jax.Array, attempted: jax.Array, dtype) -> jax.Array:
    """Per-pair acceptance rate, 0 where a pair was never attempted.

    Computed in the model compute ``dtype`` so the int/int division does not
    promote to float64 when x64 is enabled.
    """
    rate = accepted.astype(dtype) / jnp.maximum(attempted, 1).astype(dtype)
    return jnp.where(attempted > 0, rate, 0.0)


@jax.jit
def _swap_rate_stats(
    accepted: jax.Array, attempted: jax.Array, betas: jax.Array
) -> dict[str, Any]:
    """The base NRPT stats dict shared by the production run and tuning batches.

    Jitted so the handful of reductions fuse into a single dispatch/compile
    rather than running as separate eager ops on every phase (see
    ``_phase_diagnostics`` for the same motivation)."""
    acceptance_rate = _acceptance_rate(accepted, attempted, betas.dtype)
    return {
        "accepted": accepted,
        "attempted": attempted,
        "acceptance_rate": acceptance_rate,
        "rejection_rates": 1.0 - acceptance_rate,
        "betas": betas,
    }


@jax.jit
def _phase_diagnostics(
    rej: jax.Array,
    old_betas: jax.Array,
    new_betas: jax.Array,
    acceptance_rate: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The per-phase scalar diagnostics of ``tune_schedule``'s tuning loop.

    Returns ``(rej_std, max_beta_shift, Lambda, mean_acceptance)`` in one fused
    kernel. Computing them as separate eager ``jnp.std`` / ``jnp.max`` /
    ``jnp.sum`` / ``jnp.mean`` calls makes each its own XLA dispatch (and a
    separate compile the first time a shape is seen), which dominates the
    cold-start cost of an otherwise tiny per-phase computation. ``rej_std`` is
    the equalization quality (keep-best + equalize-stop); ``max_beta_shift`` is
    the ladder movement (settle check)."""
    return (
        jnp.std(rej),
        jnp.max(jnp.abs(new_betas - old_betas)),
        jnp.sum(rej),
        jnp.mean(acceptance_rate),
    )


@partial(jax.jit, static_argnums=(2,))
def _pooled_lambda(accepted: jax.Array, attempted: jax.Array, dtype) -> jax.Array:
    """Pooled barrier estimate ``Λ = Σ(1 − acceptance_rate)`` for one tuning
    batch, fused into a single dispatch (called once per ``round_batch``)."""
    return jnp.sum(1.0 - _acceptance_rate(accepted, attempted, dtype))


class _RunInputs(NamedTuple):
    """Resolved per-call NRPT inputs, common to both input modes."""

    run_program: BlockSamplingProgram
    n_free_blocks: int
    n_chains: int
    betas: jax.Array
    chain_data: Any
    base_pbi: Any  # set only in temperature-linear mode
    ebm_ref: AbstractEBM
    beta_ref: jax.Array
    compute_dtype: Any


def _resolve_run_inputs(
    ebms: Sequence[AbstractEBM] | AbstractEBM,
    programs: Sequence[BlockSamplingProgram] | BlockSamplingProgram,
    init_states: Sequence[list],
    betas: jax.Array | None,
    stacked_init: bool,
) -> _RunInputs:
    """Resolve the two input modes into a common ``_RunInputs``.

    Temperature-linear mode (single template ``ebms``/``programs``) rebases to a
    β = 1 base pair and scales interactions by β in the kernel; the per-chain
    sequence mode stacks each chain's interaction tensors. Both yield the
    reference (EBM, β) used to recover base energies and the model compute dtype
    (kept off float64 so x64 host apps don't promote a float32 model).
    """
    base_pbi = None  # set in temperature-linear mode only
    if isinstance(ebms, AbstractEBM) and isinstance(programs, BlockSamplingProgram):
        if betas is None:
            raise ValueError(
                "betas is required when passing single template ebm/program objects (temperature-linear mode)."
            )
        betas = jnp.asarray(betas)
        n_chains = len(betas)
        if not stacked_init and len(init_states) != n_chains:
            raise ValueError(
                "len(init_states) must equal len(betas) in temperature-linear mode."
            )
        beta_attr = getattr(ebms, "beta", None)
        if beta_attr is not None and float(beta_attr) == 1.0:
            # Already a β = 1 base pair. Reuse it as-is so repeated calls
            # (e.g. tune_schedule tuning phases) present identical static
            # structure to the jit cache and skip retracing entirely.
            base_ebm = ebms
            run_program = programs
        else:
            base_ebm = ebms.with_beta(jnp.asarray(1.0))
            run_program = programs.with_ebm(base_ebm)
        base_spec = run_program.gibbs_spec
        n_free_blocks = len(base_spec.free_blocks)
        base_pbi = run_program.per_block_interactions
        compute_dtype = _interaction_float_dtype(base_pbi)
        betas = betas.astype(compute_dtype)
        chain_data: object = betas
        ebm_ref, beta_ref = base_ebm, jnp.asarray(1.0, dtype=compute_dtype)
    elif isinstance(ebms, AbstractEBM) or isinstance(programs, BlockSamplingProgram):
        raise ValueError(
            "Pass ebms and programs either both as per-chain sequences, or "
            "both as single template objects (temperature-linear mode)."
        )
    else:
        if not stacked_init and not (len(ebms) == len(programs) == len(init_states)):
            raise ValueError(
                "ebms, programs, and init_states must have the same length."
            )
        if len(ebms) != len(programs):
            raise ValueError("ebms and programs must have the same length.")

        base_spec = programs[0].gibbs_spec
        n_free_blocks = len(base_spec.free_blocks)
        base_clamped = len(base_spec.clamped_blocks)
        base_nodes = [
            set(id(n) for n in block.nodes) for block in base_spec.free_blocks
        ]
        for i, prog in enumerate(programs[1:], 1):
            if (
                len(prog.gibbs_spec.free_blocks) != n_free_blocks
                or len(prog.gibbs_spec.clamped_blocks) != base_clamped
            ):
                raise ValueError("All programs must share the same block structure.")
            for b, block in enumerate(prog.gibbs_spec.free_blocks):
                prog_nodes = set(id(n) for n in block.nodes)
                if prog_nodes != base_nodes[b]:
                    raise ValueError(
                        f"programs[{i}] free block {b} contains different node "
                        f"objects than programs[0]. All programs must share the "
                        f"same node instances. When using factories, ensure "
                        f"with_beta() / with_ebm() reuse the original nodes."
                    )

        n_chains = len(ebms)
        if betas is None:
            betas = jnp.array([float(getattr(ebm, "beta")) for ebm in ebms])
        run_program = programs[0]
        stacked_pbi = [
            [
                _stack_pbi_across_chains(
                    [programs[c].per_block_interactions[b][g] for c in range(n_chains)]
                )
                for g in range(len(programs[0].per_block_interactions[b]))
            ]
            for b in range(n_free_blocks)
        ]
        chain_data = stacked_pbi
        compute_dtype = _interaction_float_dtype(stacked_pbi)
        betas = jnp.asarray(betas).astype(compute_dtype)
        ebm_ref, beta_ref = _make_reference_ebm(ebms, betas)
        beta_ref = jnp.asarray(beta_ref, dtype=compute_dtype)

    return _RunInputs(
        run_program=run_program,
        n_free_blocks=n_free_blocks,
        n_chains=n_chains,
        betas=betas,
        chain_data=chain_data,
        base_pbi=base_pbi,
        ebm_ref=ebm_ref,
        beta_ref=beta_ref,
        compute_dtype=compute_dtype,
    )


def _validate_beta_ladder(betas: jax.Array, n_chains: int) -> None:
    """Validate the β ladder is 1-D, one entry per chain, and ascending.

    Everything downstream — adjacent-pair DEO swaps, the cold-chain convention
    (states[-1]), the round-trip diagnostics — assumes the ladder is sorted
    hottest to coldest. A shuffled or descending ladder runs without error but
    silently hands back the wrong chains.
    """
    betas_np = np.asarray(betas)
    if betas_np.ndim != 1 or betas_np.size != n_chains:
        raise ValueError(
            f"betas must be a 1D array with one entry per chain (got shape {betas_np.shape} for {n_chains} chains)."
        )
    if np.any(np.diff(betas_np) < 0):
        raise ValueError(
            "betas must be in ascending order (hottest chain first, coldest "
            "chain last). Sort the ladder — and the matching ebms/programs/"
            "init_states — before calling nrpt."
        )


def _stack_init_states(
    init_states: Sequence[list],
    stacked_init: bool,
    n_chains: int,
    n_free_blocks: int,
) -> list:
    """Stack per-chain init states into ``(n_chains, ...)`` arrays per free block.

    ``init_states`` is either already-stacked (one array per free block with a
    leading n_chains axis) or a sequence of per-chain block-state lists.
    """
    if stacked_init:
        stacked_states = list(init_states)
        if len(stacked_states) != n_free_blocks:
            raise ValueError(
                f"Stacked init_states must have one entry per free block "
                f"({n_free_blocks}), got {len(stacked_states)}."
            )
        for leaf in jax.tree.leaves(stacked_states):
            if leaf.shape[0] != n_chains:
                raise ValueError(
                    f"Stacked init_states leaves must have leading dimension n_chains={n_chains}, got {leaf.shape}."
                )
        return stacked_states
    states = [list(s) for s in init_states]
    return [
        jnp.stack([states[c][b] for c in range(n_chains)]) for b in range(n_free_blocks)
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def nrpt(
    key: jax.Array,
    ebms: Sequence[AbstractEBM] | AbstractEBM,
    programs: Sequence[BlockSamplingProgram] | BlockSamplingProgram,
    init_states: Sequence[list],
    clamp_state: list,
    n_rounds: int,
    gibbs_steps_per_round: int,
    betas: jax.Array | None = None,
    track_round_trips: bool = True,
    energy_delta_fn: Callable | None = None,
    observer: AbstractNRPTObserver | None = None,
    device: DeviceLike = "auto",
    _emit_diagnostics: bool = True,
    _return_stacked: bool = False,
) -> tuple[list, dict]:
    """Non-Reversible Parallel Tempering with vectorized swaps.

    Single-pass DEO: one swap parity per round, alternating even/odd.
    Multi-pass breaks non-reversibility (even∘odd∘odd∘even = identity).

    **Temperature-linear mode**: instead of per-chain sequences, ``ebms``
    and ``programs`` may each be a *single* template object (any β; it is
    rebased to β = 1 via ``with_beta()``/``with_ebm()``). ``betas`` is then
    required and defines the chain count. Interactions are scaled by each
    chain's β inside the sampling kernel, which assumes every interaction
    array is linear in β — true for the ``DiscreteEBMFactor`` family (and
    anything built from β-scaled factor weights), and consistent with the
    E_β = β·E_base assumption the swap math already makes. This avoids
    constructing one program per chain and storing per-chain copies of
    every interaction tensor. For models whose interactions are *not*
    linear in β, pass explicit per-chain sequences instead.

    Chains are ordered by ascending β: index 0 is the **hottest** chain
    (lowest β, closest to the reference distribution) and index −1 is the
    **coldest** chain (highest β, the target distribution you want to
    sample from).  The returned ``states`` list preserves this ordering,
    and ``betas`` must be sorted ascending (validated).

    ``init_states`` may be either a sequence of per-chain block-state
    lists, or a single block-state list whose arrays carry a leading
    ``(n_chains, ...)`` axis — e.g. straight from
    ``hinton_init(key, model, blocks, (n_chains,))`` — avoiding the
    per-chain list/restack dance.
    A hottest chain at exactly β = 0 (sampling the reference distribution)
    is supported; base energies are computed from a β = 1 copy of the EBM
    when ``with_beta()`` is available, falling back to the coldest chain.

    .. warning::
       To collect samples from the target distribution, always use
       ``states[-1]`` (the cold chain), **not** ``states[0]``.

    ``device`` selects where the computation runs: ``"auto"`` (default) routes
    small workloads to the CPU and large ones to a visible accelerator (see
    ``hamon.device``), ``"cpu"``/``"gpu"`` force a platform, a concrete
    ``jax.Device`` is used as-is, and ``None`` leaves placement untouched.
    Routing re-commits the entry arrays (program tensors, states, betas) to
    the chosen device; outputs come back committed there. Arrays closed over
    by ``energy_delta_fn`` cannot be moved by routing — keep them uncommitted
    or on the target device.

    Stats keys:
        accepted, attempted, acceptance_rate, rejection_rates, betas
        round_trip_diagnostics (if track_round_trips=True):
            Lambda, tau_predicted, tau_observed, efficiency,
            lambda_profile, round_trips_per_chain, restarts_per_chain
        observations (if observer is not None):
            Per-round observer output stacked along axis 0.
        observer_carry (if observer is not None):
            Final observer carry after all rounds.
    """
    # --- Validation and mode selection -----------------------------------------
    clamp_state = clamp_state or []

    # init_states may be a sequence of per-chain block-state lists, or a
    # single block-state list of stacked (n_chains, ...) arrays (e.g. from
    # hinton_init with batch_shape=(n_chains,)). Per-chain entries are always
    # lists, so the formats are unambiguous.
    stacked_init = bool(init_states) and not isinstance(init_states[0], (list, tuple))

    ri = _resolve_run_inputs(ebms, programs, init_states, betas, stacked_init)
    run_program = ri.run_program
    n_free_blocks = ri.n_free_blocks
    n_chains = ri.n_chains
    betas = ri.betas
    chain_data = ri.chain_data
    base_pbi = ri.base_pbi
    ebm_ref = ri.ebm_ref
    beta_ref = ri.beta_ref
    compute_dtype = ri.compute_dtype

    _validate_beta_ladder(betas, n_chains)

    # --- Device routing --------------------------------------------------------
    dev = resolve_entry_device(
        device,
        n_chains=n_chains,
        n_nodes=free_node_count(run_program),
        arrays=(init_states, betas, chain_data, clamp_state, key),
    )
    if dev is not None:
        (
            key,
            run_program,
            chain_data,
            betas,
            clamp_state,
            ebm_ref,
            beta_ref,
            observer,
            init_states,
        ) = tree_device_put(
            (
                key,
                run_program,
                chain_data,
                betas,
                clamp_state,
                ebm_ref,
                beta_ref,
                observer,
                init_states,
            ),
            dev,
        )
        if base_pbi is not None:
            # base_pbi aliases run_program.per_block_interactions; re-derive it
            # from the moved program so the kernel reads on-device tensors.
            base_pbi = run_program.per_block_interactions
    device_ctx = (
        jax.default_device(dev) if dev is not None else contextlib.nullcontext()
    )

    with device_ctx:
        stacked_states = _stack_init_states(
            init_states, stacked_init, n_chains, n_free_blocks
        )

        # --- Run --------------------------------------------------------------
        n_pairs = n_chains - 1
        if n_rounds > 0:
            # Without an observer the round loop is compile-independent of the
            # round count, so hand it a traced scalar (different n_rounds reuse
            # one compile). The observer path needs scan's static length, so
            # pass the Python int.
            n_rounds_arg: int | jax.Array = (
                jnp.asarray(n_rounds, dtype=jnp.int32) if observer is None else n_rounds
            )
            final, observations = _nrpt_rounds(
                key,
                run_program,
                ebm_ref,
                beta_ref,
                base_pbi,
                chain_data,
                stacked_states,
                clamp_state,
                betas,
                n_rounds_arg,
                gibbs_steps_per_round,
                energy_delta_fn,
                observer,
                track_round_trips,
            )
        else:
            final = NRPTCarry(
                key=key,
                states=stacked_states,
                accepted=jnp.zeros(n_pairs, dtype=jnp.int32),
                attempted=jnp.zeros(n_pairs, dtype=jnp.int32),
                idx_state=init_index_state(n_chains),
                base_E=jnp.zeros(n_chains, dtype=compute_dtype),
                obs_carry=None,
            )
            observations = None

    # --- Unstack --------------------------------------------------------------
    # The public return is per-chain [chain][block] lists. ``_return_stacked``
    # (set by tune_schedule's tuning loop) instead hands back the stacked
    # [block]-of-(n_chains, ...) carry as-is, so threading states across the many
    # tuning batches skips the n_chains × n_free_blocks eager slices this
    # unstack would otherwise dispatch every call. nrpt re-ingests the stacked
    # form directly (see ``_stack_init_states``).
    if _return_stacked:
        states_out = final.states
    else:
        states_out = [
            [final.states[b][c] for b in range(n_free_blocks)] for c in range(n_chains)
        ]
    stats: dict[str, Any] = _swap_rate_stats(final.accepted, final.attempted, betas)
    rejection_rates = stats["rejection_rates"]

    # ``_emit_diagnostics=False`` (set by tune_schedule's tuning batches) skips
    # this eager per-call summary — a handful of host-dispatched reductions that
    # tuning never reads. ``track_round_trips`` itself is left untouched so the
    # in-loop index tracking stays in the jitted body and the compiled round loop
    # is still shared with the production run.
    if track_round_trips and _emit_diagnostics:
        stats["round_trip_diagnostics"] = round_trip_summary(
            final.idx_state,
            rejection_rates,
            betas,
            n_rounds,
        )
        stats["index_state"] = final.idx_state

    if observer is not None and n_rounds > 0:
        stats["observations"] = observations
        stats["observer_carry"] = final.obs_carry

    return states_out, stats


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
    target_acceptance: float = 0.6,
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
    (a small pilot) and from a reasonable guess converge to the same count. With
    no ``initial_n`` the first estimate is taken at a cheap pilot of
    ``min_chains + 16`` chains.

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
        initial_n: starting chain count. The default ``None`` estimates a
            starting point from a cheap pilot probe (no initial guess needed);
            pass an int to start there instead.
        target_acceptance: desired per-pair swap acceptance rate
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
    _pilot_n = initial_n if initial_n is not None else min_chains + 16
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
    n = _clamp(initial_n) if initial_n is not None else _clamp(min_chains + 16)
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
) -> float:
    """Measured steady-state wall time per NRPT round at ``n_expl`` (seconds).

    Times the lean round loop (``observer=None``, no eager diagnostics) on the
    already-tuned schedule. A warm-up run absorbs the one-time XLA compile and
    pages in the device, so the timed reps see only steady-state execution; the
    median over ``time_reps`` runs of ``time_rounds`` rounds divides out to the
    per-round cost ``c₀ + n_expl·c_s``. ``track_round_trips`` is left on so the
    in-loop index update (real per-round cost) is included and the compiled
    executable matches ``tune_schedule``'s tuning loop; only the host-side
    summary is skipped.
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
            observer=None,
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
        time_reps: number of timed runs to take the median over (noise control).
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

        # (2) Measure steady-state wall time per round on this schedule.
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
        )
        wall = rounds_per_probe * t_round  # seconds to produce this ESS
        return {
            "n_expl": int(n_expl),
            "objective": ess.median_ess / wall,  # ESS per measured wall-second
            "ess_median": ess.median_ess,
            "tau_obs": tau_obs,
            "rt_per_compute": tau_obs / n_expl,
            "t_round": t_round,
            "efficiency": rep.efficiency,
            "efficiency_limiter": rep.efficiency_limiter,
            "betas": np.asarray(tuned_betas),
        }

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


# ---------------------------------------------------------------------------
# Full autotuning: orchestrate N, exploration, and schedule
# ---------------------------------------------------------------------------


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
        round_trip_diagnostics: round-trip summary from the final polish run.
    """

    n_chains: int
    gibbs_steps_per_round: int
    Lambda: float
    betas: np.ndarray
    device: str | None
    chain_history: list
    exploration: dict | None
    round_trip_diagnostics: dict | None

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
            lines.append(
                f"  round trips: tau_obs={float(rt['tau_observed']):.4f}  "
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
    target_acceptance: float = 0.6,
    min_chains: int = 3,
    max_chains: int = 128,
    initial_n: int | None = None,
    search_exploration: bool = True,
    max_exploration_steps: int = 8,
    rounds_per_probe: int = 400,
    n_tune: int = 4,
    n_polish: int = 2,
    compile_cache: bool | str = True,
    device: DeviceLike = "auto",
) -> NRPTPlan:
    """Autotune the full NRPT configuration: N, exploration count, and schedule.

    The one-call front door for solving a problem with hamon. Runs the
    dependency-ordered, cheap→expensive recipe and returns an :class:`NRPTPlan`
    you draw from with :meth:`NRPTPlan.sample`:

    1. **N** via :func:`tune_chains` at n_expl=1 (cheapest probes; Λ — hence N\\* —
       is invariant to n_expl).
    2. **n_expl** via :func:`tune_exploration` at the fixed N, **reusing** the
       schedule from step 1 (production-only probes; the equi-acceptance schedule
       is invariant to n_expl, so this needs no re-tuning and never re-discovers
       N). Maximizes ESS per *measured* wall-second, so it self-calibrates to the
       device. Skipped when ``search_exploration=False``.
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
        min_chains / max_chains / initial_n: N-search bounds / start.
        search_exploration: tune n_expl (step 2); ``False`` fixes n_expl=1.
        max_exploration_steps: ceiling for the n_expl doubling search.
        rounds_per_probe: rounds per tuning/exploration probe.
        n_tune: schedule-tuning phases per N probe.
        n_polish: schedule-tuning phases in the final polish.
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

    # Resolve the device once for every stage.
    _pilot_n = initial_n if initial_n is not None else min_chains + 16
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
    exploration: dict | None = None
    n_expl = 1
    if search_exploration and max_exploration_steps > 1:
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
            fixed_schedule=betas0,
            ebm=ebm,
            program=program,
            device=dev,
        )
        n_expl = int(exploration["gibbs_steps_per_round"])

    # --- Stage 3: schedule polish at (N, n_expl) + warm cold state ---
    warm_states, polish_stats = tune_schedule(
        k_polish,
        ebm_factory,
        program_factory,
        init_states,
        clamp_state,
        n_rounds=rounds_per_probe,
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

    report = AutotuneReport(
        n_chains=n_chains,
        gibbs_steps_per_round=n_expl,
        Lambda=Lambda,
        betas=np.asarray(betas),
        device=str(dev) if dev is not None else None,
        chain_history=disc["history"],
        exploration=exploration,
        round_trip_diagnostics=polish_stats.get("round_trip_diagnostics"),
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
