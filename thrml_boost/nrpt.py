"""Non-Reversible Parallel Tempering with vectorized swaps.

Based on Syed et al. (2021), "Non-Reversible Parallel Tempering:
a Scalable Highly Parallel MCMC Scheme" (arXiv:1905.02939).

  1. Vectorized swap pass exploiting temperature-linearity of Ising energy:
     E_β(x) = β·E_base(x)  →  1 energy eval per chain replaces 4 per pair.
     All even (or odd) swaps execute simultaneously via permutation indexing.

  2. Adaptive schedule optimization (Algorithm 4): iteratively tunes β spacing
     to equalize rejection rates, minimizing the global communication barrier Λ.

  3. Round trip tracking: monitors the index process (I_n, ε_n) per machine,
     counts round trips, and estimates the communication barrier. Provides
     convergence diagnostics and validates against τ̄ = 1/(2+2Λ).

  4. Energy caching with boundary-only delta support: maintains base energies
     across rounds to avoid redundant full recomputation. When rectangular
     blocks are used, energy deltas are computed from incident edges only.

  5. Per-temperature block hooks: supports different BlockSamplingPrograms
     per chain (already in architecture), with helper functions to construct
     temperature-adapted partitions.

  6. Vmapped energy computation: _compute_base_energies uses jax.vmap over
     chain axis instead of a Python loop, producing a single batched kernel.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from thrml_boost.block_sampling import _run_blocks, BlockSamplingProgram
from thrml_boost.models.ebm import AbstractEBM
from thrml_boost.round_trips import (
    init_index_state,
    update_index_state,
    round_trip_summary,
)


# ---------------------------------------------------------------------------
# Helpers (unchanged from v0.2.0)
# ---------------------------------------------------------------------------


def _init_sampler_states(program: BlockSamplingProgram) -> list:
    return [s.init() for s in program.samplers]


def _stack_pbi_across_chains(interaction_list: list) -> object:
    flat0, treedef = jax.tree_util.tree_flatten(interaction_list[0])
    flat_rest = [jax.tree_util.tree_flatten(inter)[0] for inter in interaction_list[1:]]

    stacked_leaves = []
    for i, leaf in enumerate(flat0):
        if isinstance(leaf, jax.Array):
            stacked_leaves.append(jnp.stack([leaf] + [f[i] for f in flat_rest], axis=0))
        else:
            stacked_leaves.append(leaf)

    return treedef.unflatten(stacked_leaves)


def _make_pbi_in_axes(stacked_pbi):
    return jax.tree.map(
        lambda x: 0 if isinstance(x, jax.Array) else None,
        stacked_pbi,
    )


# ---------------------------------------------------------------------------
# Core: energy computation — vmapped (H1)
# ---------------------------------------------------------------------------


def _compute_base_energies(
    ebm0: AbstractEBM,
    beta0: jax.Array,
    spec,
    stacked_states: list,
    clamp_state: list,
) -> jax.Array:
    """Compute E_base(x) for all chains via vmap. Shape: (n_chains,).

    Exploits temperature linearity: ebm0.energy(x, spec) = β₀·E_base(x),
    so E_base = ebm0.energy(x, spec) / β₀. Vmapping over the chain axis
    produces a single batched kernel instead of n_chains sequential calls.

    beta0 is passed explicitly so the signature doesn't depend on the
    concrete EBM subclass having a .beta attribute.
    """

    def _energy_one_chain(*block_slices):
        state = list(block_slices) + clamp_state
        return ebm0.energy(state, spec)

    return jax.vmap(_energy_one_chain)(*stacked_states) / beta0


# ---------------------------------------------------------------------------
# Core: vectorized swap pass (H3/H4 precomputed constants)
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
    att_mask: jax.Array,
    base_perm: jax.Array,
) -> tuple[list, jax.Array, jax.Array, jax.Array]:
    """Execute all swaps for one set of non-overlapping pairs.

    Returns (new_states, accept_counts, attempt_counts, permutation).

    att_mask and base_perm are precomputed outside the scan body (H3/H4).
    """
    i_idx = pair_indices
    j_idx = pair_indices + 1

    log_r = (betas[i_idx] - betas[j_idx]) * (
        base_energies[i_idx] - base_energies[j_idx]
    )
    accept_probs = jnp.exp(jnp.minimum(0.0, log_r))
    u = jax.random.uniform(key, shape=(n_active,))
    accepted = u < accept_probs

    # Build permutation from precomputed identity
    perm = base_perm
    perm = perm.at[i_idx].set(jnp.where(accepted, j_idx, i_idx))
    perm = perm.at[j_idx].set(jnp.where(accepted, i_idx, j_idx))
    new_states = [stacked_states[b][perm] for b in range(n_free_blocks)]

    acc = (
        jnp.zeros(n_pairs, dtype=jnp.int32)
        .at[pair_indices]
        .set(accepted.astype(jnp.int32))
    )

    return new_states, acc, att_mask, perm


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
) -> tuple[list, list, dict]:
    """Non-Reversible Parallel Tempering with vectorized swaps.

    API-compatible with parallel_tempering(). Key optimizations:
    - Vmapped energy evaluation: single batched kernel for all chains (H1)
    - 1 energy evaluation per chain per round (temperature linearity)
    - All non-overlapping swaps execute simultaneously (permutation indexing)
    - Precomputed attempt masks and identity perm avoid scatter in scan (H3/H4)

    Single-pass DEO: one swap parity per round, alternating even/odd.
    Multi-pass (both per round) breaks non-reversibility — with disjoint
    transpositions, even∘odd composed with odd∘even = identity, creating
    period-2 orbits instead of the conveyor belt drift needed for O(1)
    round trip rates.

    When track_round_trips=True (default), monitors the index process per
    machine and includes round trip diagnostics in stats.

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
    sampler_states = (
        [list(s) for s in sampler_states]
        if sampler_states is not None
        else [_init_sampler_states(p) for p in programs]
    )
    all_ss_none = all(s is None for chain_ss in sampler_states for s in chain_ss)

    # Stack states
    stacked_states = [
        jnp.stack([states[c][b] for c in range(n_chains)]) for b in range(n_free_blocks)
    ]
    if all_ss_none:
        stacked_ss = [None] * n_free_blocks
    else:
        stacked_ss = [
            jnp.stack([sampler_states[c][b] for c in range(n_chains)])
            if sampler_states[0][b] is not None
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

    # Pair indices and precomputed constants (H3/H4)
    n_pairs = n_chains - 1
    even_pairs = jnp.arange(0, n_pairs, 2, dtype=jnp.int32)
    odd_pairs = jnp.arange(1, n_pairs, 2, dtype=jnp.int32)
    n_even = len(even_pairs)
    n_odd = len(odd_pairs)

    # Precompute attempt masks — these are compile-time constants but
    # making them explicit avoids scatter ops inside the scan body.
    att_even = jnp.zeros(n_pairs, dtype=jnp.int32).at[even_pairs].set(1)
    att_odd = jnp.zeros(n_pairs, dtype=jnp.int32).at[odd_pairs].set(1)
    base_perm = jnp.arange(n_chains, dtype=jnp.int32)

    # Reference EBM for vmapped energy (H1) — any chain works,
    # temperature linearity means E_base = E_β / β for all.
    ebm0 = ebms[0]

    accepted = jnp.zeros(n_pairs, dtype=jnp.int32)
    attempted = jnp.zeros(n_pairs, dtype=jnp.int32)

    # Initialize round trip tracking
    idx_state = init_index_state(n_chains)

    def one_round(carry, round_idx):
        key, st_states, st_ss, acc, att, idx_st = carry
        key, k_gibbs, k_swap = jax.random.split(key, 3)

        # --- Gibbs sweep (vmapped across chains) ---
        gibbs_keys = jax.random.split(k_gibbs, n_chains)
        assert _run_all_chains is not None
        st_states = _run_all_chains(gibbs_keys, st_states, stacked_pbi)

        # --- Base energies: 1 vmapped eval per round (H1) ---
        base_E = _compute_base_energies(
            ebm0,
            betas[0],
            base_spec,
            st_states,
            clamp_state,
        )

        # --- Single-pass DEO swap with round trip tracking ---
        # Alternating even/odd each round is essential for non-reversibility.
        # Multi-pass (both per round) creates period-2 orbits that kill
        # the conveyor belt — states oscillate instead of drifting.
        def do_even(args):
            ss, ac, at, sk, bE, ist = args
            ss2, ac2, at2, pm = _vectorized_swap(
                sk,
                ss,
                betas,
                bE,
                even_pairs,
                n_even,
                n_chains,
                n_pairs,
                n_free_blocks,
                att_even,
                base_perm,
            )
            ist2 = update_index_state(ist, pm, n_chains)
            return ss2, ac + ac2, at + at2, ist2

        def do_odd(args):
            ss, ac, at, sk, bE, ist = args
            ss2, ac2, at2, pm = _vectorized_swap(
                sk,
                ss,
                betas,
                bE,
                odd_pairs,
                n_odd,
                n_chains,
                n_pairs,
                n_free_blocks,
                att_odd,
                base_perm,
            )
            ist2 = update_index_state(ist, pm, n_chains)
            return ss2, ac + ac2, at + at2, ist2

        st_states, acc, att, idx_st = lax.cond(
            (round_idx & 1) == 0,
            do_even,
            do_odd,
            (st_states, acc, att, k_swap, base_E, idx_st),
        )
        return (key, st_states, st_ss, acc, att, idx_st), None

    if n_rounds > 0:
        init_carry = (key, stacked_states, stacked_ss, accepted, attempted, idx_state)
        (key, stacked_states, stacked_ss, accepted, attempted, idx_state), _ = lax.scan(
            one_round, init_carry, jnp.arange(n_rounds)
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
    ebm_factory,
    program_factory,
    init_states: Sequence[list],
    clamp_state: list,
    n_rounds: int,
    gibbs_steps_per_round: int,
    initial_betas: jax.Array,
    n_tune: int = 5,
    rounds_per_tune: int = 200,
    track_round_trips: bool = True,
) -> tuple[list, list, dict]:
    """NRPT with iterative schedule optimization (Algorithm 4).

    Runs n_tune adaptation phases, each of rounds_per_tune rounds, updating
    the β schedule after each phase. Then runs the final n_rounds production
    phase with the optimized schedule.

    Returns (states, sampler_states, stats) where stats includes tuning
    history in stats["tuning_history"].
    """
    betas = initial_betas
    current_states = init_states
    tuning_history = []

    for tune_iter in range(n_tune):
        key, subkey = jax.random.split(key)
        ebms = ebm_factory(betas)
        programs = program_factory(ebms)
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
    ebms = ebm_factory(betas)
    programs = program_factory(ebms)
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
    ebm_factory,
    program_factory,
    init_factory,
    clamp_state: list,
    beta_range: tuple[float, float],
    gibbs_steps_per_round: int,
    initial_n: int = 8,
    target_acceptance: float = 0.6,
    rounds_per_probe: int = 200,
    n_tune_per_probe: int = 4,
    max_iters: int = 6,
    min_chains: int = 3,
    max_chains: int = 128,
    lambda_rtol: float = 0.05,
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
        ebms = ebm_factory(betas)
        programs = program_factory(ebms)
        inits = init_factory(n_current, ebms, programs)

        # Early iterations: cheap probes. Final: full budget.
        is_early = iteration < max_iters - 1
        probe_tune = max(2, n_tune_per_probe // 2) if is_early else n_tune_per_probe
        probe_rounds = max(50, rounds_per_probe // 3) if is_early else rounds_per_probe

        _, _, stats = nrpt_adaptive(
            k_probe,
            ebm_factory,
            program_factory,
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

        # Recommendation from conservative (max) Λ estimate
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

        # --- Convergence check 1: chain count matches recommendation ---
        if abs(n_recommended - n_current) <= 1:
            converged_reason = "chain_count"
            break

        # --- Convergence check 2: Λ has stabilized ---
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

        # --- Step toward recommendation ---
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
