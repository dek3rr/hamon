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
from collections.abc import Sequence
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
from hamon.interaction import interaction_float_dtype as _interaction_float_dtype
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
    _round_trip_summary_host,
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


# Test instrumentation: incremented on each (re)trace of _nrpt_rounds so
# tests can assert the jit cache was reused.
_nrpt_rounds_trace_count = [0]


def _build_gibbs_runner(
    run_program: BlockSamplingProgram,
    clamp_state: list,
    gibbs_steps_per_round: int,
    base_pbi,
    chain_data,
    n_free_blocks: int,
    base_pbi_offset=None,
):
    """Build the vmapped per-chain Gibbs kernel for one round.

    Temperature-linear mode (``base_pbi`` set) scales the shared β = 1
    interaction arrays by each chain's β inside the kernel; affine
    (reference-annealing) mode (``base_pbi_offset`` also set) interpolates them
    as ``offset + β·slope``; the per-chain sequence mode maps over the stacked
    interaction pytree. Returns the vmapped
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

    if base_pbi is not None and base_pbi_offset is not None:
        # Affine (reference-annealing) mode: E_β = E₀ + β·(E₁ − E₀), so each
        # chain's interactions are offset (reference, β=0) + β·slope.
        _slope_pbi = base_pbi
        _offset_pbi = base_pbi_offset
        chain_in_axes: object = 0

        def _chain_step(gibbs_key, state_free, chain_input):
            pbi_c = jax.tree.map(
                lambda o, s: o + chain_input * s if isinstance(o, jax.Array) else o,
                _offset_pbi,
                _slope_pbi,
            )
            return _run_one(gibbs_key, state_free, pbi_c)
    elif base_pbi is not None:
        # Temperature-linear mode: scale one shared β=1 program by each
        # chain's β inside the vmapped kernel — bit-identical to per-chain
        # programs for β-linear interactions (scalar-times-array commutes with
        # program-construction slicing) without n_chains weight copies.
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
    live_chains: jax.Array | None = None,
):
    """Build the even/odd DEO swap branches for ``lax.cond``.

    Returns ``(do_even, do_odd)`` — the two parity branches the round loop
    alternates between (single-pass DEO preserves non-reversibility).
    ``live_chains`` (traced) masks the pass to the live prefix of a padded
    ladder — see ``_make_swap_branch``.
    """
    n_pairs = n_chains - 1
    even_pairs = jnp.arange(0, n_pairs, 2, dtype=jnp.int32)
    odd_pairs = jnp.arange(1, n_pairs, 2, dtype=jnp.int32)
    att_even = jnp.zeros(n_pairs, dtype=jnp.int32).at[even_pairs].set(1)
    att_odd = jnp.zeros(n_pairs, dtype=jnp.int32).at[odd_pairs].set(1)
    base_perm = jnp.arange(n_chains, dtype=jnp.int32)

    swap_args = (
        betas,
        n_chains,
        n_pairs,
        n_free_blocks,
        base_perm,
        track_round_trips,
        live_chains,
    )
    do_even = _make_swap_branch(even_pairs, len(even_pairs), att_even, *swap_args)
    do_odd = _make_swap_branch(odd_pairs, len(odd_pairs), att_odd, *swap_args)
    return do_even, do_odd


def _build_energy_compute(
    energy_delta_fn: Callable | None,
    ebm_ref: AbstractEBM,
    beta_ref: jax.Array,
    base_spec,
    clamp_state: list,
    ebm_ref0: AbstractEBM | None = None,
):
    """Build the per-round base-energy update.

    With ``energy_delta_fn`` set, energies are advanced by boundary-only deltas
    off the cached value; otherwise they are recomputed from scratch each round.
    With ``ebm_ref0`` set (affine/reference-annealing mode) the base energy is
    ``Δ = E₁ − E₀``: the swap ratio at (β_i, β_j) is
    ``(β_i − β_j)·(Δ(x_j) − Δ(x_i))`` — the β-independent E₀ cancels exactly,
    so Δ is the only quantity the DEO swaps may see.
    Signature: ``(new_states, old_states, cached_base_E) -> base_E``.
    """
    if energy_delta_fn is not None:
        _delta_fn = energy_delta_fn

        def _energy_cached(st_states, old_states, cached_bE):
            return cached_bE + _delta_fn(old_states, st_states)

        return _energy_cached

    def _energy_fresh(st_states, old_states, cached_bE):
        base = _compute_base_energies(
            ebm_ref, beta_ref, base_spec, st_states, clamp_state
        )
        if ebm_ref0 is not None:
            base = base - _compute_base_energies(
                ebm_ref0, beta_ref, base_spec, st_states, clamp_state
            )
        return base

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
    live_chains: jax.Array | None = None,
    base_pbi_offset=None,
    ebm_ref0: AbstractEBM | None = None,
) -> tuple[NRPTCarry, Any]:
    """The jitted NRPT round loop: vmapped Gibbs sweeps + DEO swaps.

    ``live_chains`` (a **traced** scalar, or ``None``) enables chain masking:
    the arrays are padded to a fixed ladder length, only the first
    ``live_chains`` chains form the live ladder (swaps at pair index ≥
    live_chains − 1 are forced-rejected, so the padding is fully decoupled),
    and because it is traced data, probes at every live count share ONE
    compiled executable. Padding chains still do (wasted, harmless) Gibbs
    work. Callers slice results back to the live prefix.

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
        base_pbi_offset,
    )
    do_even, do_odd = _build_swap_passes(
        betas, n_chains, n_free_blocks, track_round_trips, live_chains
    )
    energy_compute = _build_energy_compute(
        energy_delta_fn, ebm_ref, beta_ref, base_spec, clamp_state, ebm_ref0
    )
    observer_init, observer_step = _build_observer_hooks(observer)

    base_E = _compute_base_energies(
        ebm_ref, beta_ref, base_spec, stacked_states, clamp_state
    )
    if ebm_ref0 is not None:
        # Affine mode: swap energies are Δ = E₁ − E₀ (see _build_energy_compute).
        base_E = base_E - _compute_base_energies(
            ebm_ref0, beta_ref, base_spec, stacked_states, clamp_state
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
        # Keep energies aligned with the just-permuted states; the cached
        # strategy and the observers both consume the pair.
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

    # Without an observer, a traced-n_rounds fori_loop lets one compile serve
    # every round count; an observer needs scan's static length for its
    # stacked per-round output.
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
    rather than running as separate eager ops on every phase."""
    acceptance_rate = _acceptance_rate(accepted, attempted, betas.dtype)
    return {
        "accepted": accepted,
        "attempted": attempted,
        "acceptance_rate": acceptance_rate,
        "rejection_rates": 1.0 - acceptance_rate,
        "betas": betas,
    }


def _swap_rate_stats_host(
    accepted: jax.Array,
    attempted: jax.Array,
    betas: jax.Array,
    n_pairs: int | None = None,
) -> dict[str, Any]:
    """Host equivalent of :func:`_swap_rate_stats` for tuning orchestration.

    Schedule tuning consumes these tiny counters in Python immediately.  Running
    their reductions through XLA would compile a distinct executable for every
    live ladder length, even when the padded NRPT round loop is shared.  This is
    deliberately private: the public production path remains JAX-native.

    ``n_pairs`` slices padded swap counters to the live prefix *after* the
    device fetch — masked padding pairs never attempt, so the sliced values are
    exactly the unpadded ones, and no per-live-N slice kernel is compiled.
    """
    accepted_np, attempted_np, betas_np = (
        np.asarray(x) for x in jax.device_get((accepted, attempted, betas))
    )
    if n_pairs is not None:
        accepted_np = accepted_np[:n_pairs]
        attempted_np = attempted_np[:n_pairs]
    dtype = betas_np.dtype
    acceptance_rate = np.divide(
        accepted_np.astype(dtype),
        np.maximum(attempted_np, 1).astype(dtype),
        out=np.zeros_like(accepted_np, dtype=dtype),
        where=attempted_np > 0,
    )
    return {
        "accepted": accepted_np,
        "attempted": attempted_np,
        "acceptance_rate": acceptance_rate,
        "rejection_rates": np.asarray(1, dtype=dtype) - acceptance_rate,
        "betas": betas_np,
    }


class _RunInputs(NamedTuple):
    """Resolved per-call NRPT inputs, common to both input modes."""

    run_program: BlockSamplingProgram
    n_free_blocks: int
    n_chains: int
    betas: jax.Array
    chain_data: Any
    base_pbi: Any  # set only in temperature-linear mode (β-slope when affine)
    ebm_ref: AbstractEBM
    beta_ref: jax.Array
    compute_dtype: Any
    # Affine (reference-annealing) template mode only — None otherwise:
    base_pbi_offset: Any = None  # interactions at β = 0 (the reference)
    ebm_ref0: AbstractEBM | None = None  # β = 0 EBM for Δ = E₁ − E₀ swap energies


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
    base_pbi_offset = None  # set in affine (reference-annealing) mode only
    ebm_ref0: AbstractEBM | None = None
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
            # Reuse the β=1 base pair as-is so repeated calls present
            # identical static structure to the jit cache.
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
        if getattr(base_ebm, "beta_affine", False):
            # Affine path: build the β=0 offset program alongside the β=1
            # slope so the kernel interpolates and swaps use Δ = E₁ − E₀.
            ebm_ref0 = base_ebm.with_beta(jnp.asarray(0.0, dtype=compute_dtype))
            program0 = run_program.with_ebm(ebm_ref0)
            base_pbi_offset = program0.per_block_interactions
            base_pbi = jax.tree.map(
                lambda one, zero: (one - zero) if isinstance(one, jax.Array) else one,
                base_pbi,
                base_pbi_offset,
            )
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
        if any(getattr(e, "beta_affine", False) for e in ebms):
            # Per-chain-sequence swap math assumes E_β = β·E_base; an affine
            # path needs Δ = E₁ − E₀, which only template mode computes.
            raise ValueError(
                "beta-affine EBMs (e.g. AnnealedEBM) require temperature-linear "
                "template mode: pass a single ebm/program pair with betas=."
            )

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
        base_pbi_offset=base_pbi_offset,
        ebm_ref0=ebm_ref0,
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
                f"Stacked init_states must have one entry per free block ({n_free_blocks}), got {len(stacked_states)}."
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


@partial(jax.jit, static_argnums=1)
def _pad_stacked_states(stacked_states: list, pad: int) -> list:
    """Extend every free block's chain axis by ``pad`` copies of its last chain.

    Fused under one jit so the per-block concatenate/broadcast pair compiles
    once, not once per ragged block: the padding rows are decoupled masked
    chains, so which state they copy is irrelevant.
    """
    return [
        jnp.concatenate([x, jnp.broadcast_to(x[-1:], (pad, *x.shape[1:]))])
        for x in stacked_states
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
    pad_chains_to: int | None = None,
    _emit_diagnostics: bool = True,
    _return_stacked: bool = False,
    _host_stats: bool = False,
    _keep_padded_states: bool = False,
) -> tuple[list, dict]:
    """Non-Reversible Parallel Tempering with vectorized swaps.

    ``pad_chains_to`` (≥ n_chains) enables **chain masking**: the ladder is
    padded to that fixed length with copies of the coldest chain and the round
    loop runs with the true chain count as *traced* data — swaps beyond the
    live prefix are forced-rejected (identity permutation), so the padding is
    fully decoupled from the live ladder. All returned states/stats are sliced
    back to the true count, so callers see exactly the shapes and semantics of
    an unpadded run. The point: probes at *different* chain counts padded to
    the same length share ONE compiled round loop instead of recompiling per
    count (the dominant cold cost of ``tune_chains``), at the price of wasted
    Gibbs work on the padding chains (~free on a dispatch-bound accelerator).
    Temperature-linear mode only; incompatible with ``energy_delta_fn``, and an
    ``observer`` must be masking-safe (see
    :class:`~hamon.observers.ColdIndexObserver`).

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
    per-chain list/restack dance. With ``pad_chains_to``, a stacked init may
    instead carry the padded ``(pad_chains_to, ...)`` leading axis (rows past
    the live count are decoupled padding, e.g. the retained carry of a
    previous masked run).
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

    # init_states is either per-chain block-state lists or one stacked
    # (n_chains, ...) block-state list; per-chain entries are always lists, so
    # the formats are unambiguous.
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
    base_pbi_offset = ri.base_pbi_offset
    ebm_ref0 = ri.ebm_ref0

    _validate_beta_ladder(betas, n_chains)

    if ebm_ref0 is not None and energy_delta_fn is not None:
        raise ValueError(
            "energy_delta_fn is incompatible with a beta-affine EBM: the swap "
            "energies are Δ = E₁ − E₀, and a user delta function advances only "
            "the target energy."
        )

    # Continuous/unbounded models have no proper β=0 member (variance 1/(β·P)
    # diverges), so a β=0 rung would silently go non-finite — fail loudly.
    if float(np.asarray(betas)[0]) == 0.0 and not ebm_ref.proper_at_beta_zero:
        raise ValueError(
            f"{type(ebm_ref).__name__} is not proper at beta=0 (unbounded state "
            "space), but the ladder starts at exactly 0. Use a beta ladder with "
            "beta_min > 0."
        )

    # --- Chain-masking validation ----------------------------------------------
    pad_to = int(pad_chains_to) if pad_chains_to is not None else None
    if pad_to is not None:
        if pad_to < n_chains:
            raise ValueError(
                f"pad_chains_to={pad_to} must be >= the chain count ({n_chains})."
            )
        if base_pbi is None:
            raise ValueError(
                "pad_chains_to requires temperature-linear mode (single template "
                "ebm/program); per-chain sequences are not supported."
            )
        if energy_delta_fn is not None:
            raise ValueError(
                "pad_chains_to is incompatible with energy_delta_fn (boundary deltas would span the padded ladder)."
            )
        # Padding chains evolve independently, so an observer must read only
        # live positions (a raw -1 index records a divergent copy); a
        # masking-safe observer like ColdIndexObserver reads a traced live
        # index, letting draws at different live N share one compiled loop.
        if observer is not None and not getattr(observer, "masking_safe", False):
            raise ValueError(
                "pad_chains_to is incompatible with this observer: it would see "
                "the padded ladder. Use a masking-safe observer (e.g. "
                "ColdIndexObserver) that reads only live positions."
            )

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
            if base_pbi_offset is not None:
                # slope/offset/β=0-EBM are standalone trees (slope does not
                # alias the program), so move them to the device directly.
                base_pbi, base_pbi_offset, ebm_ref0 = tree_device_put(
                    (base_pbi, base_pbi_offset, ebm_ref0), dev
                )
            else:
                # base_pbi aliases the program's interactions; re-derive it
                # from the moved program so the kernel reads on-device tensors.
                base_pbi = run_program.per_block_interactions
    device_ctx = (
        jax.default_device(dev) if dev is not None else contextlib.nullcontext()
    )

    with device_ctx:
        # When list-form states feed a masked ladder, pad the *list* before
        # stacking it — otherwise each live N compiles a separate eager
        # ``jnp.stack``. A stacked init may also arrive already padded (e.g.
        # a ``_keep_padded_states`` carry from a previous tuning batch).
        states_pre_padded = False
        stack_rows = n_chains
        if pad_to is not None and pad_to > n_chains:
            if not stacked_init:
                init_states = [
                    *init_states,
                    *([init_states[-1]] * (pad_to - n_chains)),
                ]
                states_pre_padded = True
                stack_rows = pad_to
            elif jax.tree.leaves(list(init_states))[0].shape[0] == pad_to:
                states_pre_padded = True
                stack_rows = pad_to
        stacked_states = _stack_init_states(
            init_states,
            stacked_init,
            stack_rows,
            n_free_blocks,
        )

        # Chain masking: pad the ladder to pad_to with copies of the coldest
        # chain and pass the true count as traced data (in temperature-linear
        # mode chain_data IS the betas array).
        live_chains = None
        betas_run = betas
        chain_data_run = chain_data
        if pad_to is not None:
            live_chains = jnp.asarray(n_chains, dtype=jnp.int32)
            pad = pad_to - n_chains
            if pad > 0:
                # Pad on host (exact copies): a device pad would compile a
                # concatenate/broadcast pair per live N.
                betas_np = np.asarray(betas)
                betas_padded = np.concatenate(
                    [betas_np, np.broadcast_to(betas_np[-1:], (pad,))]
                )
                betas_run = (
                    jax.device_put(betas_padded, dev)
                    if dev is not None
                    else jnp.asarray(betas_padded)
                )
                chain_data_run = betas_run
                if not states_pre_padded:
                    stacked_states = _pad_stacked_states(stacked_states, pad)

        # --- Run --------------------------------------------------------------
        n_pairs = n_chains - 1
        if n_rounds > 0:
            # Traced n_rounds shares one compile across round counts; the
            # observer path needs scan's static length (Python int).
            n_rounds_arg: int | jax.Array = (
                jnp.asarray(n_rounds, dtype=jnp.int32) if observer is None else n_rounds
            )
            final, observations = _nrpt_rounds(
                key,
                run_program,
                ebm_ref,
                beta_ref,
                base_pbi,
                chain_data_run,
                stacked_states,
                clamp_state,
                betas_run,
                n_rounds_arg,
                gibbs_steps_per_round,
                energy_delta_fn,
                observer,
                track_round_trips,
                live_chains,
                base_pbi_offset,
                ebm_ref0,
            )
            # Slice padded carries back to the live prefix. The private
            # host-stats path instead slices counters and the index process
            # on host (a device slice here is a per-live-N XLA executable).
            if pad_to is not None and pad_to > n_chains:
                if _host_stats:
                    if not _keep_padded_states:
                        final = final._replace(
                            states=[st[:n_chains] for st in final.states]
                        )
                else:
                    final = final._replace(
                        states=(
                            final.states
                            if _keep_padded_states
                            else [st[:n_chains] for st in final.states]
                        ),
                        accepted=final.accepted[: n_chains - 1],
                        attempted=final.attempted[: n_chains - 1],
                        idx_state={k: v[:n_chains] for k, v in final.idx_state.items()},
                        base_E=final.base_E[:n_chains],
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

    # ``_return_stacked`` (tune_schedule's tuning loop) keeps the stacked
    # [block]-of-(n_chains, ...) carry, skipping the n_chains × n_free_blocks
    # eager slices per call; nrpt re-ingests it via ``_stack_init_states``.
    if _return_stacked:
        states_out = final.states
    else:
        states_out = [
            [final.states[b][c] for b in range(n_free_blocks)] for c in range(n_chains)
        ]
    stats: dict[str, Any] = (
        _swap_rate_stats_host(final.accepted, final.attempted, betas, n_chains - 1)
        if _host_stats
        else _swap_rate_stats(final.accepted, final.attempted, betas)
    )
    rejection_rates = stats["rejection_rates"]

    # ``_emit_diagnostics=False`` (tuning batches) skips these host-dispatched
    # reductions tuning never reads; in-loop round-trip tracking stays jitted
    # so the compiled round loop is still shared with production.
    if track_round_trips and _emit_diagnostics:
        if _host_stats:
            # Host avoids one more per-ladder-length executable per probe;
            # padding machines never trip, so the slice is exact.
            idx_state_host = {
                k: v[:n_chains] for k, v in jax.device_get(final.idx_state).items()
            }
            stats["round_trip_diagnostics"] = _round_trip_summary_host(
                idx_state_host,
                rejection_rates,
                stats["betas"],
                n_rounds,
            )
            stats["index_state"] = idx_state_host
        else:
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
