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
from typing import Any, Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

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
    update_index_state,
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
            return [_ebm.with_beta(jnp.array(float(b))) for b in betas]

        def _make_programs(ebms):
            return [_prog.with_ebm(e) for e in ebms]

        return _make_ebms, _make_programs
    elif ebm_factory is not None and program_factory is not None:
        return ebm_factory, program_factory
    else:
        raise ValueError("Provide both ebm_factory and program_factory, or neither.")


class _ChainSource:
    """Uniform producer of per-chain EBM/program arguments for the template
    and factory routes of ``nrpt_adaptive`` and ``discover_chain_count``.

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
# Core: energy computation
# ---------------------------------------------------------------------------


def _compute_base_energies(
    ebm_ref: AbstractEBM,
    beta_ref: jax.Array,
    spec,
    stacked_states: list,
    clamp_state: list,
) -> jax.Array:
    """Compute E_base(x) for all chains via vmap. Shape: (n_chains,).

    E_base = ebm_ref.energy(x, spec) / β_ref (temperature linearity).
    β_ref must be nonzero; callers should prefer a β=1 reference EBM so the
    division is exact (see `_make_reference_ebm`).
    """

    def _energy_one_chain(*block_slices):
        state = list(block_slices) + clamp_state
        return ebm_ref.energy(state, spec)

    return jax.vmap(_energy_one_chain)(*stacked_states) / beta_ref


def _make_reference_ebm(
    ebms: Sequence[AbstractEBM], betas: jax.Array
) -> tuple[AbstractEBM, jax.Array]:
    """Pick the (EBM, β) pair used to recover base energies E_base = E(x)/β.

    Using the hottest chain (β₀) breaks when β₀ = 0 — a standard NRPT ladder
    anchored at the reference distribution — because E(x) is then identically
    0 and the division yields NaN, which silently rejects every swap. Prefer
    an exact β=1 copy of the EBM so no division error is possible; for EBM
    classes that do not implement `with_beta()`, fall back to the coldest
    chain, whose β is the largest (best-conditioned) divisor in the ladder.
    """
    try:
        return ebms[-1].with_beta(jnp.asarray(1.0)), jnp.asarray(1.0)
    except NotImplementedError:
        if float(betas[-1]) == 0.0:
            raise ValueError(
                "Cannot compute base energies: the coldest chain has β = 0 and "
                f"{type(ebms[-1]).__name__} does not implement with_beta(). "
                "Either implement with_beta() or use a ladder whose coldest "
                "chain has β > 0."
            )
        return ebms[-1], betas[-1]


# ---------------------------------------------------------------------------
# Core: vectorized swap pass
# ---------------------------------------------------------------------------


def _vectorized_swap(
    key: jax.Array,
    stacked_states: list,
    betas: jax.Array,
    base_energies: jax.Array,
    pair_indices: jax.Array,
    n_active: int,
    n_chains: int,
    n_pairs: int,
    n_free_blocks: int,
    base_perm: jax.Array,
) -> tuple[list, jax.Array, jax.Array]:
    """Execute all swaps for one set of non-overlapping pairs.

    Returns (new_states, accept_counts, permutation).
    """
    i_idx = pair_indices
    j_idx = pair_indices + 1

    log_r = (betas[i_idx] - betas[j_idx]) * (
        base_energies[i_idx] - base_energies[j_idx]
    )
    accept_probs = jnp.exp(jnp.minimum(0.0, log_r))
    u = jax.random.uniform(key, shape=(n_active,), dtype=accept_probs.dtype)
    accepted = u < accept_probs

    perm = base_perm
    perm = perm.at[i_idx].set(jnp.where(accepted, j_idx, i_idx))
    perm = perm.at[j_idx].set(jnp.where(accepted, i_idx, j_idx))
    new_states = [stacked_states[b][perm] for b in range(n_free_blocks)]

    acc = (
        jnp.zeros(n_pairs, dtype=jnp.int32)
        .at[pair_indices]
        .set(accepted.astype(jnp.int32))
    )

    return new_states, acc, perm


def _make_swap_branch(
    pair_indices: jax.Array,
    n_active: int,
    att_mask: jax.Array,
    betas: jax.Array,
    n_chains: int,
    n_pairs: int,
    n_free_blocks: int,
    base_perm: jax.Array,
    track_round_trips: bool,
):
    """Build a lax.cond branch for even or odd swap pass.

    Returns (states, acc, att, idx_state, perm).
    """

    def _branch(args):
        ss, ac, at, sk, bE, ist = args
        ss2, ac2, pm = _vectorized_swap(
            sk,
            ss,
            betas,
            bE,
            pair_indices,
            n_active,
            n_chains,
            n_pairs,
            n_free_blocks,
            base_perm,
        )
        # Static flag: with round-trip tracking disabled, the index-process
        # update is dropped from the compiled program entirely.
        new_ist = update_index_state(ist, pm, n_chains) if track_round_trips else ist
        return (
            ss2,
            ac + ac2,
            at + att_mask,
            new_ist,
            pm,
        )

    return _branch


# ---------------------------------------------------------------------------
# Adaptive schedule (Section 5.4)
# ---------------------------------------------------------------------------


def optimize_schedule(rejection_rates: jax.Array, betas: jax.Array) -> jax.Array:
    """Equalize per-pair rejection rates by redistributing β values.

    The result keeps the dtype of ``betas`` so repeated tuning phases do not
    drift to float64 when x64 is enabled."""
    betas = jnp.asarray(betas)
    rej = jnp.asarray(rejection_rates, dtype=betas.dtype)
    cum = jnp.concatenate([jnp.zeros(1, dtype=betas.dtype), jnp.cumsum(rej)])
    target = jnp.linspace(0.0, cum[-1], len(betas), dtype=betas.dtype)
    new = jnp.interp(target, cum, betas)
    return new.at[0].set(betas[0]).at[-1].set(betas[-1])


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
    n_rounds: int,
    gibbs_steps_per_round: int,
    energy_delta_fn: Callable | None,
    observer: AbstractNRPTObserver | None,
    track_round_trips: bool,
) -> tuple[NRPTCarry, Any]:
    """The jitted NRPT round loop: vmapped Gibbs sweeps + DEO swaps via scan.

    Module-level and ``eqx.filter_jit``-decorated so the compilation cache
    persists across calls: repeated invocations with the same program/observer
    structure and array shapes (e.g. the tuning phases of ``nrpt_adaptive``)
    reuse the compiled executable. Arrays — including ``betas`` — are traced
    data, so schedule updates between phases do not retrigger compilation.
    """
    _nrpt_rounds_trace_count[0] += 1

    n_chains = betas.shape[0]
    n_free_blocks = len(stacked_states)
    base_spec = run_program.gibbs_spec

    # --- Vmapped Gibbs kernel -------------------------------------------------
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

    run_chains = jax.vmap(
        _chain_step,
        in_axes=(0, [0] * n_free_blocks, chain_in_axes),
    )

    # --- Swap setup -----------------------------------------------------------
    n_pairs = n_chains - 1
    even_pairs = jnp.arange(0, n_pairs, 2, dtype=jnp.int32)
    odd_pairs = jnp.arange(1, n_pairs, 2, dtype=jnp.int32)
    att_even = jnp.zeros(n_pairs, dtype=jnp.int32).at[even_pairs].set(1)
    att_odd = jnp.zeros(n_pairs, dtype=jnp.int32).at[odd_pairs].set(1)
    base_perm = jnp.arange(n_chains, dtype=jnp.int32)

    swap_args = (betas, n_chains, n_pairs, n_free_blocks, base_perm, track_round_trips)
    do_even = _make_swap_branch(
        even_pairs,
        len(even_pairs),
        att_even,
        *swap_args,
    )
    do_odd = _make_swap_branch(
        odd_pairs,
        len(odd_pairs),
        att_odd,
        *swap_args,
    )

    # --- Energy strategy (cached vs recomputed) -------------------------------
    if energy_delta_fn is not None:
        _delta_fn = energy_delta_fn

        def _energy_cached(st_states, old_states, cached_bE):
            return cached_bE + _delta_fn(old_states, st_states)

        energy_compute = _energy_cached
    else:

        def _energy_fresh(st_states, old_states, cached_bE):
            return _compute_base_energies(
                ebm_ref, beta_ref, base_spec, st_states, clamp_state
            )

        energy_compute = _energy_fresh

    # --- Observer strategy (present vs absent) --------------------------------
    if observer is not None:
        observer_init, observer_step = observer.init, observer
    else:

        def _obs_init():
            return None

        def _obs_step(stacked_states, base_energies, round_idx, carry):
            return carry, None

        observer_init, observer_step = _obs_init, _obs_step

    # --- Initial energy -------------------------------------------------------
    base_E = _compute_base_energies(
        ebm_ref,
        beta_ref,
        base_spec,
        stacked_states,
        clamp_state,
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

    return lax.scan(one_round, init_carry, jnp.arange(n_rounds))


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
            # (e.g. nrpt_adaptive tuning phases) present identical static
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

    # --- Beta ladder validation ------------------------------------------------
    # Everything downstream — adjacent-pair DEO swaps, the cold-chain
    # convention (states[-1]), the round-trip diagnostics — assumes the
    # ladder is sorted hottest to coldest. A shuffled or descending ladder
    # runs without error but silently hands back the wrong chains.
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
        # --- Stack states -----------------------------------------------------
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
        else:
            states = [list(s) for s in init_states]
            stacked_states = [
                jnp.stack([states[c][b] for c in range(n_chains)])
                for b in range(n_free_blocks)
            ]

        # --- Run --------------------------------------------------------------
        n_pairs = n_chains - 1
        if n_rounds > 0:
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
                n_rounds,
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
    states_out = [
        [final.states[b][c] for b in range(n_free_blocks)] for c in range(n_chains)
    ]
    # Rates are reported in the model's compute dtype; the plain int/int
    # division would yield float64 under x64.
    acceptance_rate = jnp.where(
        final.attempted > 0,
        final.accepted.astype(betas.dtype)
        / jnp.maximum(final.attempted, 1).astype(betas.dtype),
        0.0,
    )
    rejection_rates = 1.0 - acceptance_rate

    stats: dict[str, Any] = {
        "accepted": final.accepted,
        "attempted": final.attempted,
        "acceptance_rate": acceptance_rate,
        "rejection_rates": rejection_rates,
        "betas": betas,
    }

    if track_round_trips:
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
    acceptance_rate = None
    rounds_used = 0
    lambda_prev = None
    stable_count = 0

    # Always run at least one batch; break on stability or the round ceiling.
    while True:
        key, subkey = jax.random.split(key)
        batch = min(round_batch, max_rounds - rounds_used)
        states, stats = run_phase(subkey, betas, states, batch)
        acc_total = (
            stats["accepted"] if acc_total is None else acc_total + stats["accepted"]
        )
        att_total = (
            stats["attempted"] if att_total is None else att_total + stats["attempted"]
        )
        rounds_used += batch

        rate = acc_total.astype(betas.dtype) / jnp.maximum(att_total, 1).astype(
            betas.dtype
        )
        acceptance_rate = jnp.where(att_total > 0, rate, 0.0)
        lambda_cur = float(jnp.sum(1.0 - acceptance_rate))
        if rounds_used >= min_rounds and lambda_prev is not None:
            rel = abs(lambda_cur - lambda_prev) / max(lambda_cur, 1e-9)
            stable_count = stable_count + 1 if rel < lambda_rtol else 0
            if stable_count >= stable_k:
                break
        lambda_prev = lambda_cur
        if rounds_used >= max_rounds:
            break

    assert acc_total is not None and att_total is not None
    assert acceptance_rate is not None
    pooled_stats = {
        "accepted": acc_total,
        "attempted": att_total,
        "acceptance_rate": acceptance_rate,
        "rejection_rates": 1.0 - acceptance_rate,
        "betas": betas,
    }
    return states, pooled_stats, rounds_used


def nrpt_adaptive(
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

    def _run_phase(phase_key, phase_betas, phase_states, rounds, phase_observer=None):
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
        )

    betas = initial_betas
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
                subkey, betas, current_states, rounds_per_tune
            )
            rounds_used = rounds_per_tune

        rej = stats["rejection_rates"]
        # Equalization quality of the schedule just evaluated: lower spread of
        # per-pair rejection rates = better tuned. Drives keep-best and the
        # equalization stop.
        quality = float(jnp.std(rej))
        old_betas = betas
        if adaptive_tuning and quality < best_quality:
            best_quality = quality
            best_betas = old_betas
        betas = optimize_schedule(rej, betas)

        max_beta_shift = float(jnp.max(jnp.abs(betas - old_betas)))
        phase_lambda = float(jnp.sum(rej))
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
            "nrpt_adaptive tune %d/%d: Lambda=%.3f mean_acceptance=%.3f "
            "rej_std=%.4g max|dbeta|=%.4g rounds=%d",
            phase,
            n_tune,
            phase_lambda,
            float(jnp.mean(stats["acceptance_rate"])),
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
                    "nrpt_adaptive: schedule converged after %d phase(s) "
                    "(rej_std=%.4g, max|dbeta|=%.4g); skipping remaining tuning",
                    phase,
                    quality,
                    max_beta_shift,
                )
                break
        elif effective_tol is not None and max_beta_shift < effective_tol:
            # Legacy early-stop (unchanged semantics).
            logger.info(
                "nrpt_adaptive: schedule converged after %d phase(s) "
                "(max|dbeta|=%.4g < tune_tol=%.4g); skipping remaining tuning",
                phase,
                max_beta_shift,
                effective_tol,
            )
            break

    if adaptive_tuning:
        betas = best_betas

    # Production run
    key, subkey = jax.random.split(key)
    states, stats = _run_phase(
        subkey, betas, current_states, n_rounds, phase_observer=observer
    )
    stats["tuning_history"] = tuning_history
    return states, stats


# ---------------------------------------------------------------------------
# Iterative chain count discovery
# ---------------------------------------------------------------------------


def discover_chain_count(
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
    device: DeviceLike = "auto",
) -> dict:
    """Iteratively discover the right chain count for a given target acceptance.

    The bootstrapping problem: Λ estimated with too few chains is biased low
    because the schedule can't resolve the peak in λ(β). The decision policy
    below works as follows:

    1. With no ``initial_n``, take a cheap, ceiling-independent pilot probe for a
       first (biased-low) Λ; otherwise start from ``initial_n``.
    2. Each probe: track the running max-Λ, extrapolate Λ to N→∞ from the two
       latest probes, and recommend N from the extrapolated Λ with a margin sized
       to the observed rejection-rate spread and barrier growth.
    3. Jump straight to the recommendation when the estimate is confident (low
       noise, or Λ still clearly rising), else take a damped step toward it.
    4. Stop once the per-pair rejection meets target **and** Λ has stopped rising
       (growth-gated, so it never commits while still under-resolved), or once N
       has converged.

    Extrapolating to the asymptotic Λ — rather than stepping halfway toward a
    max-Λ recommendation — converges in fewer probes, and returning the
    margin-padded recommendation directly (no monotone ratchet up to the last
    probed N) keeps the result at/above target without systematic overshoot.

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
        # Forward whichever route the caller used; nrpt_adaptive re-dispatches
        # through its own _ChainSource. The concrete device (or None) bypasses its
        # heuristic, so probes never flip devices. Tuning is adaptive, so a
        # wrong-N probe still self-limits its rounds.
        _, stats = nrpt_adaptive(
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

    def _unpack(res: dict[str, Any]) -> tuple[float, float, float, float]:
        lam = max(0.0, float(res["Lambda_raw"]))
        rates = res["rejection_rates"]
        if len(rates) > 0:
            m = float(np.mean(rates))
            cv = float(np.std(rates)) / (m + 1e-12)
            sp = float(np.max(rates) - np.min(rates)) / (m + 1e-12)
        else:
            m = cv = sp = 0.0
        return lam, m, cv, sp

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

    # --- Decision policy: estimate the asymptotic barrier from a couple of
    # probes and land on N ≈ Λ/r_target + 1 without overshooting. ---
    best_betas = None
    lambda_raw = lambda_max = 0.0
    used = 0
    seen: list = []  # (n, lambda_raw, cv, sp, mean_rejection)
    short_hist: list = []
    reason = "max_iters"

    if initial_n is None:
        # Ceiling-independent pilot: a small fixed offset above min_chains yields
        # a first (biased-low) Λ without a hard-coded guess.
        res = probe(min_chains + 16)
        used += 1
        lambda_raw, m, cv, sp = _unpack(res)
        lambda_max, best_betas = lambda_raw, res["betas"]
        seen.append((res["n"], lambda_raw, cv, sp, m))
        g0 = (
            1.0
            + max(0.01, 0.45 * lambda_rtol)
            + 0.10 * min(1.0, cv)
            + 0.08 * min(1.0, sp)
        )
        n_current = _clamp(int(np.ceil(lambda_raw * g0 / r_target)) + 1)
    else:
        n_current = _clamp(initial_n)

    n_recommended = n_current
    while used < max_probes:
        res = probe(n_current)
        used += 1
        n_current = res["n"]
        lambda_raw, m, cv, sp = _unpack(res)
        seen.append((n_current, lambda_raw, cv, sp, m))
        if lambda_raw >= lambda_max:
            lambda_max, best_betas = lambda_raw, res["betas"]

        short_hist.append((n_current, lambda_raw))
        if len(short_hist) > 2:
            short_hist.pop(0)

        # Extrapolate Λ to N→∞ from the two latest distinct probes (linear in
        # 1/(N-1)), clamped to [Λ_max, 1.4·Λ_max] against noisy blow-ups.
        lambda_hat = lambda_max
        if len(short_hist) == 2 and short_hist[0][0] != short_hist[1][0]:
            (na, ya), (nb, yb) = short_hist
            x1, x2 = 1.0 / max(1, na - 1), 1.0 / max(1, nb - 1)
            if abs(x2 - x1) > 1e-12:
                slope = (yb - ya) / (x2 - x1)
                lambda_hat = ya - slope * x1
                lambda_hat = min(max(lambda_hat, lambda_max), 1.4 * lambda_max)

        growth = 0.0
        if len(seen) >= 2:
            (n1, l1, _, _, _), (n2, l2, _, _, _) = sorted(seen, key=lambda t: t[0])[-2:]
            if n2 > n1:
                growth = max(0.0, (l2 - l1) / (abs(l2) + 1e-12))

        cv_recent = max(t[2] for t in seen[-2:])
        sp_recent = max(t[3] for t in seen[-2:])
        guard = (
            1.0
            + 0.2 * lambda_rtol
            + 0.1 * min(1.0, cv_recent)
            + 0.1 * min(1.0, sp_recent)
            + 0.1 * min(1.0, growth)
        )
        n_recommended = _clamp(int(np.ceil(lambda_hat * guard / r_target)) + 1)

        history.append(
            {
                "iteration": len(history),
                "n": n_current,
                "Lambda_raw": lambda_raw,
                "Lambda_max": lambda_max,
                "n_recommended": n_recommended,
                "rejection_rates": res["rejection_rates"],
                "betas": res["betas"],
            }
        )

        # Early stop only once Λ has stopped rising (growth-gated): never commit
        # while still under-resolved, even if observed rejection already looks ok.
        if n_current > 1:
            expected_rej = lambda_hat / max(1, n_current - 1)
            if growth < 0.05 and (
                expected_rej <= r_target * (1.0 + lambda_rtol)
                or (m and m <= r_target * (1.0 + lambda_rtol))
            ):
                reason = "lambda_stable"
                break

        stable = len(seen) < 2 or seen[-1][1] <= seen[-2][1] * (1.0 + 0.5 * lambda_rtol)
        if n_current >= n_recommended and stable:
            reason = "chain_count"
            break
        if used >= max_probes:
            break

        remain = max_probes - used
        if n_current < n_recommended:
            if growth > 0.10 or cv_recent < 0.20 or sp_recent < 0.30 or remain <= 2:
                # Confident enough → jump straight to the recommendation.
                n_current = _clamp(n_recommended)
            else:
                # Otherwise a damped step (80% rec + 20% current), never backward.
                step = int(np.ceil(0.80 * n_recommended + 0.20 * n_current))
                n_current = _clamp(max(n_current + 1, step))
        else:
            n_current = _clamp(max(n_current, n_recommended))

    # The latest recommendation is the answer (no monotone ratchet up to the
    # last probed N → no systematic overshoot). Produce the returned schedule at
    # that count, reusing the probe if it was already run.
    n_final = _clamp(n_recommended)
    final_stats = probed[n_final] if n_final in probed else probe(n_final)
    best_betas = final_stats["betas"]
    lambda_max = max(lambda_max, final_stats["Lambda_raw"])
    history.append(
        {
            "iteration": len(history),
            "n": int(n_final),
            "Lambda_raw": float(final_stats["Lambda_raw"]),
            "Lambda_max": float(lambda_max),
            "n_recommended": int(n_final),
            "rejection_rates": final_stats["rejection_rates"],
            "betas": final_stats["betas"],
        }
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
