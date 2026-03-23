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
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from hamon.block_sampling import _run_blocks, BlockSamplingProgram
from hamon.models.ebm import AbstractEBM
from hamon.round_trips import (
    init_index_state,
    update_index_state,
    round_trip_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_sampler_states(program: BlockSamplingProgram) -> list:
    return [s.init() for s in program.samplers]


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
    ebm0: AbstractEBM,
    beta0: jax.Array,
    spec,
    stacked_states: list,
    clamp_state: list,
) -> jax.Array:
    """Compute E_base(x) for all chains via vmap. Shape: (n_chains,).

    E_base = ebm0.energy(x, spec) / β₀ (temperature linearity).
    """

    def _energy_one_chain(*block_slices):
        state = list(block_slices) + clamp_state
        return ebm0.energy(state, spec)

    return jax.vmap(_energy_one_chain)(*stacked_states) / beta0


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
    return_perm: bool = False,
):
    """Build a lax.cond branch for even or odd swap pass.

    Returns (states, acc, att, idx_state[, perm]) — perm included when return_perm=True.
    """
    if not return_perm:

        def _branch_no_perm(args):
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
            return ss2, ac + ac2, at + att_mask, update_index_state(ist, pm, n_chains)

        return _branch_no_perm

    def _branch_with_perm(args):
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

    return _branch_with_perm


# ---------------------------------------------------------------------------
# Adaptive schedule (Section 5.4)
# ---------------------------------------------------------------------------


def optimize_schedule(rejection_rates: jax.Array, betas: jax.Array) -> jax.Array:
    """Equalize per-pair rejection rates by redistributing β values."""
    cum = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(rejection_rates)])
    target = jnp.linspace(0.0, cum[-1], len(betas))
    new = jnp.interp(target, cum, betas)
    return new.at[0].set(betas[0]).at[-1].set(betas[-1])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def nrpt(
    key: jax.Array,
    ebms: Sequence[AbstractEBM],
    programs: Sequence[BlockSamplingProgram],
    init_states: Sequence[list],
    clamp_state: list,
    n_rounds: int,
    gibbs_steps_per_round: int,
    betas: jax.Array | None = None,
    sampler_states: Sequence[list] | None = None,
    track_round_trips: bool = True,
    energy_delta_fn: Callable | None = None,
) -> tuple[list, list, dict]:
    """Non-Reversible Parallel Tempering with vectorized swaps.

    Single-pass DEO: one swap parity per round, alternating even/odd.
    Multi-pass breaks non-reversibility (even∘odd∘odd∘even = identity).

    Stats keys:
        accepted, attempted, acceptance_rate, rejection_rates, betas
        round_trip_diagnostics (if track_round_trips=True):
            Lambda, tau_predicted, tau_observed, efficiency,
            lambda_profile, round_trips_per_chain, restarts_per_chain
    """
    if not (len(ebms) == len(programs) == len(init_states)):
        raise ValueError("ebms, programs, and init_states must have the same length.")

    base_spec = programs[0].gibbs_spec
    n_free_blocks = len(base_spec.free_blocks)
    base_clamped = len(base_spec.clamped_blocks)
    for prog in programs[1:]:
        if (
            len(prog.gibbs_spec.free_blocks) != n_free_blocks
            or len(prog.gibbs_spec.clamped_blocks) != base_clamped
        ):
            raise ValueError("All programs must share the same block structure.")

    clamp_state = clamp_state or []
    n_chains = len(ebms)

    if betas is None:
        betas = jnp.array([float(getattr(ebm, "beta")) for ebm in ebms])

    states = [list(s) for s in init_states]
    sampler_states_l = (
        [list(s) for s in sampler_states]
        if sampler_states is not None
        else [_init_sampler_states(p) for p in programs]
    )
    all_ss_none = all(s is None for chain_ss in sampler_states_l for s in chain_ss)

    # Stack states
    stacked_states = [
        jnp.stack([states[c][b] for c in range(n_chains)]) for b in range(n_free_blocks)
    ]
    if all_ss_none:
        stacked_ss = [None] * n_free_blocks
    else:
        stacked_ss = [
            jnp.stack([sampler_states_l[c][b] for c in range(n_chains)])
            if sampler_states_l[0][b] is not None
            else None
            for b in range(n_free_blocks)
        ]

    # Stack per-block interactions
    stacked_pbi = [
        [
            _stack_pbi_across_chains(
                [programs[c].per_block_interactions[b][g] for c in range(n_chains)]
            )
            for g in range(len(programs[0].per_block_interactions[b]))
        ]
        for b in range(n_free_blocks)
    ]
    pbi_in_axes = _make_pbi_in_axes(stacked_pbi)

    # Build vmapped Gibbs kernel
    _run_all_chains = None
    if all_ss_none:
        null_ss = [None] * n_free_blocks

        def _run_one(gibbs_key, state_free, pbi):
            new_state, _, _ = _run_blocks(
                gibbs_key,
                programs[0],
                state_free,
                clamp_state,
                gibbs_steps_per_round,
                null_ss,
                per_block_interactions=pbi,
            )
            return new_state

        _run_all_chains = jax.vmap(
            _run_one,
            in_axes=(0, [0] * n_free_blocks, pbi_in_axes),
        )

    # Pair indices and precomputed constants
    n_pairs = n_chains - 1
    even_pairs = jnp.arange(0, n_pairs, 2, dtype=jnp.int32)
    odd_pairs = jnp.arange(1, n_pairs, 2, dtype=jnp.int32)
    n_even = len(even_pairs)
    n_odd = len(odd_pairs)
    att_even = jnp.zeros(n_pairs, dtype=jnp.int32).at[even_pairs].set(1)
    att_odd = jnp.zeros(n_pairs, dtype=jnp.int32).at[odd_pairs].set(1)
    base_perm = jnp.arange(n_chains, dtype=jnp.int32)

    # Reference EBM for vmapped energy — any chain works,
    # temperature linearity means E_base = E_β / β for all.
    ebm0 = ebms[0]

    accepted = jnp.zeros(n_pairs, dtype=jnp.int32)
    attempted = jnp.zeros(n_pairs, dtype=jnp.int32)
    idx_state = init_index_state(n_chains)

    use_cached = energy_delta_fn is not None

    # Narrow type for Pyright — guaranteed non-None when all_ss_none is True.
    assert _run_all_chains is not None
    run_chains = _run_all_chains

    swap_args = (betas, n_chains, n_pairs, n_free_blocks, base_perm)
    do_even = _make_swap_branch(
        even_pairs,
        n_even,
        att_even,
        *swap_args,
        return_perm=use_cached,
    )
    do_odd = _make_swap_branch(
        odd_pairs,
        n_odd,
        att_odd,
        *swap_args,
        return_perm=use_cached,
    )

    if not use_cached:

        def one_round(carry, round_idx):
            key, st_states, st_ss, acc, att, idx_st = carry
            key, k_gibbs, k_swap = jax.random.split(key, 3)

            gibbs_keys = jax.random.split(k_gibbs, n_chains)
            st_states = run_chains(gibbs_keys, st_states, stacked_pbi)

            base_E = _compute_base_energies(
                ebm0,
                betas[0],
                base_spec,
                st_states,
                clamp_state,
            )

            st_states, acc, att, idx_st = lax.cond(
                (round_idx & 1) == 0,
                do_even,
                do_odd,
                (st_states, acc, att, k_swap, base_E, idx_st),
            )
            return (key, st_states, st_ss, acc, att, idx_st), None

        if n_rounds > 0:
            init_carry = (
                key,
                stacked_states,
                stacked_ss,
                accepted,
                attempted,
                idx_state,
            )
            (key, stacked_states, stacked_ss, accepted, attempted, idx_state), _ = (
                lax.scan(one_round, init_carry, jnp.arange(n_rounds))
            )

    else:
        # Cached path: carry base_E, update via delta, permute after swap.
        # bE[pm] MUST be applied outside lax.cond (traced intermediates).
        base_E = _compute_base_energies(
            ebm0,
            betas[0],
            base_spec,
            stacked_states,
            clamp_state,
        )

        # Narrow type for Pyright — we're in the `use_cached` branch.
        assert energy_delta_fn is not None
        delta_fn = energy_delta_fn

        def one_round_cached(carry, round_idx):
            key, st_states, st_ss, acc, att, idx_st, cached_bE = carry
            key, k_gibbs, k_swap = jax.random.split(key, 3)

            old_states = st_states
            gibbs_keys = jax.random.split(k_gibbs, n_chains)
            st_states = run_chains(gibbs_keys, st_states, stacked_pbi)

            # delta_fn(old, new) = E_base(new) - E_base(old), no beta factor.
            cached_bE = cached_bE + delta_fn(old_states, st_states)

            st_states, acc, att, idx_st, pm = lax.cond(
                (round_idx & 1) == 0,
                do_even,
                do_odd,
                (st_states, acc, att, k_swap, cached_bE, idx_st),
            )
            # Permute cached energies to match post-swap chain ordering.
            cached_bE = cached_bE[pm]
            return (key, st_states, st_ss, acc, att, idx_st, cached_bE), None

        if n_rounds > 0:
            init_carry = (
                key,
                stacked_states,
                stacked_ss,
                accepted,
                attempted,
                idx_state,
                base_E,
            )
            (key, stacked_states, stacked_ss, accepted, attempted, idx_state, _), _ = (
                lax.scan(one_round_cached, init_carry, jnp.arange(n_rounds))
            )

    # Unstack
    states_out = [
        [stacked_states[b][c] for b in range(n_free_blocks)] for c in range(n_chains)
    ]
    if all_ss_none:
        ss_out = [[None] * n_free_blocks for _ in range(n_chains)]
    else:
        ss_arrays = [s for s in stacked_ss if s is not None]
        ss_out = [
            [ss_arrays[b][c] for b in range(n_free_blocks)] for c in range(n_chains)
        ]

    acceptance_rate = jnp.where(attempted > 0, accepted / attempted, 0.0)
    rejection_rates = 1.0 - acceptance_rate

    stats: dict[str, Any] = {
        "accepted": accepted,
        "attempted": attempted,
        "acceptance_rate": acceptance_rate,
        "rejection_rates": rejection_rates,
        "betas": betas,
    }

    if track_round_trips:
        stats["round_trip_diagnostics"] = round_trip_summary(
            idx_state,
            rejection_rates,
            betas,
            n_rounds,
        )
        stats["index_state"] = idx_state

    return states_out, ss_out, stats


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
) -> tuple[list, list, dict]:
    """NRPT with iterative schedule optimization (Algorithm 4).

    Runs n_tune adaptation phases, each of rounds_per_tune rounds, updating
    the β schedule after each phase. Then runs the final n_rounds production
    phase with the optimized schedule.

    Instead of providing ``ebm_factory`` and ``program_factory``, you can pass
    a template ``ebm`` and ``program`` and the factories will be built
    internally using ``ebm.with_beta()`` and ``program.with_ebm()``.

    Returns (states, sampler_states, stats) where stats includes tuning
    history in stats["tuning_history"].
    """
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
    elif ebm_factory is not None and program_factory is not None:
        _make_ebms = ebm_factory
        _make_programs = program_factory
    else:
        raise ValueError("Provide both ebm_factory and program_factory, or neither.")

    if clamp_state is None:
        clamp_state = []
    if initial_betas is None:
        raise ValueError("initial_betas is required.")

    betas = initial_betas
    current_states = init_states
    tuning_history = []

    for tune_iter in range(n_tune):
        key, subkey = jax.random.split(key)
        ebms = _make_ebms(betas)
        programs = _make_programs(ebms)
        states, ss, stats = nrpt(
            subkey,
            ebms,
            programs,
            current_states,
            clamp_state,
            rounds_per_tune,
            gibbs_steps_per_round,
            betas=betas,
            track_round_trips=track_round_trips,
        )
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
    ebms = _make_ebms(betas)
    programs = _make_programs(ebms)
    states, ss, stats = nrpt(
        subkey,
        ebms,
        programs,
        current_states,
        clamp_state,
        n_rounds,
        gibbs_steps_per_round,
        betas=betas,
        track_round_trips=track_round_trips,
    )
    stats["tuning_history"] = tuning_history
    return states, ss, stats


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
    elif ebm_factory is not None and program_factory is not None:
        _make_ebms = ebm_factory
        _make_programs = program_factory
    else:
        raise ValueError("Provide both ebm_factory and program_factory, or neither.")

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

    for iteration in range(max_iters):
        betas = jnp.linspace(beta_range[0], beta_range[1], n_current)

        key, k_probe = jax.random.split(key)
        ebms = _make_ebms(betas)
        programs = _make_programs(ebms)
        inits = init_factory(n_current, ebms, programs)

        # Early iterations: cheap probes. Final: full budget.
        is_early = iteration < max_iters - 1
        probe_tune = max(2, n_tune_per_probe // 2) if is_early else n_tune_per_probe
        probe_rounds = max(50, rounds_per_probe // 3) if is_early else rounds_per_probe

        _, _, stats = nrpt_adaptive(
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
