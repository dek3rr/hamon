"""Non-Reversible Parallel Tempering with vectorized swaps.

Based on Syed et al. (2021), "Non-Reversible Parallel Tempering:
a Scalable Highly Parallel MCMC Scheme" (arXiv:1905.02939).

Exploits temperature-linearity (E_β = β·E_base) for single-eval-per-chain
swap decisions. Adaptive schedule optimization (Algorithm 4) equalizes
rejection rates. Optional energy caching with boundary-only deltas for
rectangular block partitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from hamon.block_sampling import _run_blocks, BlockSamplingProgram
from hamon.models.ebm import AbstractEBM
from hamon.observers import AbstractNRPTObserver
from hamon.round_trips import (
    init_index_state,
    update_index_state,
    round_trip_summary,
)


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
    u = jax.random.uniform(key, shape=(n_active,))
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
        return (
            ss2,
            ac + ac2,
            at + att_mask,
            update_index_state(ist, pm, n_chains),
            pm,
        )

    return _branch


# ---------------------------------------------------------------------------
# Adaptive schedule (Section 5.4)
# ---------------------------------------------------------------------------


def optimize_schedule(rejection_rates: jax.Array, betas: jax.Array) -> jax.Array:
    """Equalize per-pair rejection rates by redistributing β values."""
    cum = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(rejection_rates)])
    target = jnp.linspace(0.0, cum[-1], len(betas))
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
    sample from).  The returned ``states`` list preserves this ordering.
    A hottest chain at exactly β = 0 (sampling the reference distribution)
    is supported; base energies are computed from a β = 1 copy of the EBM
    when ``with_beta()`` is available, falling back to the coldest chain.

    .. warning::
       To collect samples from the target distribution, always use
       ``states[-1]`` (the cold chain), **not** ``states[0]``.

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

    base_pbi = None  # set in temperature-linear mode only
    if isinstance(ebms, AbstractEBM) and isinstance(programs, BlockSamplingProgram):
        if betas is None:
            raise ValueError(
                "betas is required when passing single template ebm/program "
                "objects (temperature-linear mode)."
            )
        betas = jnp.asarray(betas)
        n_chains = len(betas)
        if len(init_states) != n_chains:
            raise ValueError(
                "len(init_states) must equal len(betas) in temperature-linear mode."
            )
        base_ebm = ebms.with_beta(jnp.asarray(1.0))
        run_program = programs.with_ebm(base_ebm)
        base_spec = run_program.gibbs_spec
        n_free_blocks = len(base_spec.free_blocks)
        base_pbi = run_program.per_block_interactions
        chain_data: object = betas
        chain_in_axes: object = 0
        ebm_ref, beta_ref = base_ebm, jnp.asarray(1.0)
    elif isinstance(ebms, AbstractEBM) or isinstance(programs, BlockSamplingProgram):
        raise ValueError(
            "Pass ebms and programs either both as per-chain sequences, or "
            "both as single template objects (temperature-linear mode)."
        )
    else:
        if not (len(ebms) == len(programs) == len(init_states)):
            raise ValueError(
                "ebms, programs, and init_states must have the same length."
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
        chain_in_axes = _make_pbi_in_axes(stacked_pbi)
        ebm_ref, beta_ref = _make_reference_ebm(ebms, betas)

    # --- Stack states ---------------------------------------------------------
    states = [list(s) for s in init_states]
    stacked_states = [
        jnp.stack([states[c][b] for c in range(n_chains)]) for b in range(n_free_blocks)
    ]

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

        def _chain_step(gibbs_key, state_free, chain_input):
            pbi_c = jax.tree.map(
                lambda x: chain_input * x if isinstance(x, jax.Array) else x,
                _base_pbi,
            )
            return _run_one(gibbs_key, state_free, pbi_c)
    else:

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

    accepted = jnp.zeros(n_pairs, dtype=jnp.int32)
    attempted = jnp.zeros(n_pairs, dtype=jnp.int32)
    idx_state = init_index_state(n_chains)

    use_cached = energy_delta_fn is not None

    swap_args = (betas, n_chains, n_pairs, n_free_blocks, base_perm)
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
    if use_cached:
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

    # --- Run ------------------------------------------------------------------
    init_carry = NRPTCarry(
        key=key,
        states=stacked_states,
        accepted=accepted,
        attempted=attempted,
        idx_state=idx_state,
        base_E=base_E,
        obs_carry=observer_init(),
    )

    if n_rounds > 0:
        final, observations = lax.scan(one_round, init_carry, jnp.arange(n_rounds))
    else:
        final = init_carry
        observations = None

    # --- Unstack --------------------------------------------------------------
    states_out = [
        [final.states[b][c] for b in range(n_free_blocks)] for c in range(n_chains)
    ]
    acceptance_rate = jnp.where(
        final.attempted > 0, final.accepted / final.attempted, 0.0
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
) -> tuple[list, dict]:
    """NRPT with iterative schedule optimization (Algorithm 4).

    Runs n_tune adaptation phases, each of rounds_per_tune rounds, updating
    the β schedule after each phase. Then runs the final n_rounds production
    phase with the optimized schedule.

    Instead of providing ``ebm_factory`` and ``program_factory``, you can pass
    a template ``ebm`` and ``program`` and the factories will be built
    internally using ``ebm.with_beta()`` and ``program.with_ebm()``.

    Returns ``(states, stats)`` where stats includes tuning history in
    ``stats["tuning_history"]``.  States are ordered by ascending β — the
    **cold chain** (target distribution) is ``states[-1]``.
    """
    # Template route: hand the single (ebm, program) pair straight to nrpt's
    # temperature-linear mode. No per-phase EBM/program rebuilds — only the
    # betas array changes between tuning phases.
    if clamp_state is None:
        clamp_state = []
    if initial_betas is None:
        raise ValueError("initial_betas is required.")

    if (
        ebm_factory is None
        and program_factory is None
        and ebm is not None
        and program is not None
    ):
        template_ebm, template_program = ebm, program

        def _run_phase(
            phase_key, phase_betas, phase_states, rounds, phase_observer=None
        ):
            return nrpt(
                phase_key,
                template_ebm,
                template_program,
                phase_states,
                clamp_state,
                rounds,
                gibbs_steps_per_round,
                betas=phase_betas,
                track_round_trips=track_round_trips,
                observer=phase_observer,
            )
    else:
        _make_ebms, _make_programs = _resolve_factories(
            ebm_factory, program_factory, ebm, program
        )

        def _run_phase(
            phase_key, phase_betas, phase_states, rounds, phase_observer=None
        ):
            chain_ebms = _make_ebms(phase_betas)
            chain_programs = _make_programs(chain_ebms)
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
            )

    betas = initial_betas
    current_states = init_states
    tuning_history = []

    for tune_iter in range(n_tune):
        key, subkey = jax.random.split(key)
        states, stats = _run_phase(subkey, betas, current_states, rounds_per_tune)
        old_betas = betas
        betas = optimize_schedule(stats["rejection_rates"], betas)
        current_states = states

        tuning_history.append(
            {
                "iteration": tune_iter,
                "betas": old_betas,
                "rejection_rates": stats["rejection_rates"],
                "acceptance_rate": stats["acceptance_rate"],
                "Lambda": float(jnp.sum(stats["rejection_rates"])),
            }
        )

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
    initial_n: int = 8,
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
) -> dict:
    """Iteratively discover the right chain count for a given target acceptance.

    The bootstrapping problem: Λ estimated with too few chains is biased low
    because the schedule can't resolve the peak in λ(β). Each iteration:

    1. Build N chains, run a short schedule optimization to estimate Λ.
    2. Update the running max-Λ (conservative: never underestimate).
    3. Compute N_rec from max-Λ, step halfway toward it.
    4. Stop when EITHER:
       - N_rec ≈ N (chain count converged), OR
       - Λ has stabilized (|ΔΛ/Λ| < lambda_rtol for 2 consecutive iters)

    Using max-Λ prevents the "overshoot then drop" pattern where a noisy
    high estimate at iteration k inflates the recommendation, then a lower
    estimate at k+1 can't undo the damage. Stabilization detection catches
    the case where Λ is already well-resolved but N_rec still differs from
    N by a few chains.

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
        initial_n: starting chain count
        target_acceptance: desired per-pair swap acceptance rate
        rounds_per_probe: rounds for the final production probe
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
            converged_reason: "chain_count" | "lambda_stable" | "no_progress" | "max_iters"
            history: list of per-iteration dicts
    """
    _make_ebms, _make_programs = _resolve_factories(
        ebm_factory, program_factory, ebm, program
    )

    if init_factory is None:
        raise ValueError("init_factory is required.")
    if clamp_state is None:
        clamp_state = []

    r_target = 1.0 - target_acceptance
    n_current = initial_n
    history = []
    best_betas = None
    lambda_max = 0.0
    lambda_raw = 0.0
    lambda_prev = 0.0
    stable_count = 0
    converged_reason = "max_iters"

    # Template route: probes run through nrpt's temperature-linear mode, so
    # per-chain programs are never constructed. init_factory still receives a
    # programs list for free-block extraction; in template mode every entry
    # is the template program (identical gibbs_spec).
    template_mode = (
        ebm_factory is None
        and program_factory is None
        and ebm is not None
        and program is not None
    )

    for iteration in range(max_iters):
        betas = jnp.linspace(beta_range[0], beta_range[1], n_current)

        key, k_probe = jax.random.split(key)
        ebms = _make_ebms(betas)
        programs = [program] * n_current if template_mode else _make_programs(ebms)
        inits = init_factory(n_current, ebms, programs)

        # Early iterations: cheap probes. Final: full budget.
        is_early = iteration < max_iters - 1
        probe_tune = max(2, n_tune_per_probe // 2) if is_early else n_tune_per_probe
        probe_rounds = max(50, rounds_per_probe // 3) if is_early else rounds_per_probe

        if template_mode:
            _, stats = nrpt_adaptive(
                k_probe,
                ebm=ebm,
                program=program,
                init_states=inits,
                clamp_state=clamp_state,
                n_rounds=probe_rounds,
                gibbs_steps_per_round=gibbs_steps_per_round,
                initial_betas=betas,
                n_tune=probe_tune,
                rounds_per_tune=probe_rounds,
            )
        else:
            _, stats = nrpt_adaptive(
                k_probe,
                _make_ebms,
                _make_programs,
                inits,
                clamp_state,
                n_rounds=probe_rounds,
                gibbs_steps_per_round=gibbs_steps_per_round,
                initial_betas=betas,
                n_tune=probe_tune,
                rounds_per_tune=probe_rounds,
            )

        lambda_raw = float(jnp.sum(stats["rejection_rates"]))
        lambda_max = max(lambda_max, lambda_raw)
        best_betas = stats["betas"]

        n_recommended = max(min_chains, int(np.ceil(lambda_max / max(r_target, 0.01))))
        n_recommended = min(n_recommended, max_chains)

        history.append(
            {
                "iteration": iteration,
                "n": n_current,
                "Lambda_raw": lambda_raw,
                "Lambda_max": lambda_max,
                "n_recommended": n_recommended,
                "rejection_rates": np.array(stats["rejection_rates"]),
                "betas": np.array(stats["betas"]),
            }
        )

        if abs(n_recommended - n_current) <= 1:
            converged_reason = "chain_count"
            break

        if iteration > 0 and lambda_max > 0:
            rel_change = abs(lambda_raw - lambda_prev) / lambda_max
            if rel_change < lambda_rtol:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= 2:
                n_current = n_recommended
                converged_reason = "lambda_stable"
                break

        lambda_prev = lambda_raw

        step = int(np.ceil((n_recommended - n_current) / 2))
        n_next = n_current + step
        n_next = max(min_chains, min(n_next, max_chains))

        if n_next == n_current:
            converged_reason = "no_progress"
            break

        n_current = n_next

    return {
        "n_chains": n_current,
        "betas": best_betas,
        "Lambda": lambda_max,
        "Lambda_raw": lambda_raw,
        "target_acceptance": target_acceptance,
        "converged_reason": converged_reason,
        "history": history,
    }
