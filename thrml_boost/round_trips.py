"""Round trip tracking for Non-Reversible Parallel Tempering.

Implements the index process monitoring from Syed et al. (2021):
- Track which chain slot each machine's state occupies via permutation
- Count round trips (machine visits chain 0, then chain N, then chain 0)
- Estimate local communication barrier λ(β) from rejection rates
- Estimate global barrier Λ = ∫λ(β)dβ
- Predict optimal round trip rate τ̄ = 1/(2+2Λ)

The index process state is carried through the lax.scan loop alongside
chain states, adding minimal overhead (a few int/bool arrays of size
n_chains).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Index process state
# ---------------------------------------------------------------------------


def init_index_state(n_chains: int) -> dict:
    """Initialize index process tracking arrays.

    ``machine_to_chain[j]`` = which chain position machine j's state
    currently occupies.  Initially machine j is at chain j.

    ``visited_top[j]`` = whether machine j has reached chain N since
    its last round trip completion.

    Returns a dict suitable for inclusion in lax.scan carry.
    """
    return {
        "machine_to_chain": jnp.arange(n_chains, dtype=jnp.int32),
        "visited_top": jnp.zeros(n_chains, dtype=jnp.bool_),
        "round_trips": jnp.zeros(n_chains, dtype=jnp.int32),
        "restarts": jnp.zeros(n_chains, dtype=jnp.int32),
    }


def update_index_state(
    index_state: dict,
    perm: jax.Array,
    n_chains: int,
) -> dict:
    """Update the index process after a swap pass.

    The swap pass applies a permutation to the stacked states:
    ``new_states[i] = old_states[perm[i]]``.  This means the state
    that *was* at chain ``perm[i]`` is *now* at chain ``i``.

    For each machine j whose state was at chain ``old_pos = m2c[j]``,
    the new position is ``inv_perm[old_pos]``.

    Args:
        index_state: current tracking dict
        perm: (n_chains,) int array — the permutation applied to states
        n_chains: total number of chains
    """
    old_m2c = index_state["machine_to_chain"]
    visited = index_state["visited_top"]
    rts = index_state["round_trips"]
    restarts = index_state["restarts"]
    N = n_chains - 1

    # Self inverses perm == inv_perm
    new_m2c = perm[old_m2c]

    # Detect visits to top (chain N)
    at_top = new_m2c == N
    new_visited = visited | at_top
    new_restarts = restarts + (at_top & ~visited).astype(jnp.int32)

    # Detect round trips: visited top previously and now at bottom (chain 0)
    at_bottom = new_m2c == 0
    completed = at_bottom & new_visited
    new_rts = rts + completed.astype(jnp.int32)

    # Reset visited_top for machines that completed a round trip
    new_visited = new_visited & ~completed

    return {
        "machine_to_chain": new_m2c,
        "visited_top": new_visited,
        "round_trips": new_rts,
        "restarts": new_restarts,
    }


# ---------------------------------------------------------------------------
# Communication barrier estimation
# ---------------------------------------------------------------------------


def estimate_local_barrier(
    rejection_rates: jax.Array,
    betas: jax.Array,
) -> jax.Array:
    """Estimate λ(β) at midpoints from per-pair rejection rates.

    λ(β) ≈ r(i,i+1) / |β_{i+1} - β_i|  (Theorem 2, Syed et al.)

    Returns array of shape (n_pairs,) with λ estimates at
    β_mid = (β_i + β_{i+1}) / 2.
    """
    dbeta = jnp.diff(betas)
    safe_dbeta = jnp.maximum(jnp.abs(dbeta), 1e-10)
    return rejection_rates / safe_dbeta


def estimate_global_barrier(
    rejection_rates: jax.Array,
) -> jax.Array:
    """Estimate Λ = Σ r(i,i+1) ≈ ∫λ(β)dβ (Corollary 2, Syed et al.)."""
    return jnp.sum(rejection_rates)


def predict_optimal_round_trip_rate(Lambda: float | jax.Array) -> jax.Array:
    """τ̄ = 1/(2+2Λ) — the asymptotic optimal for NRPT (Theorem 3)."""
    return jnp.array(1.0) / (2.0 + 2.0 * jnp.asarray(Lambda))


def empirical_round_trip_rate(
    index_state: dict,
    n_rounds: int,
) -> jax.Array:
    """Compute observed round trip rate from index process state.

    τ_obs = total_round_trips / n_rounds
    """
    total_rts = jnp.sum(index_state["round_trips"])
    return total_rts / jnp.maximum(n_rounds, 1)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def round_trip_summary(
    index_state: dict,
    rejection_rates: jax.Array,
    betas: jax.Array,
    n_rounds: int,
) -> dict:
    """Compute full diagnostic summary for NRPT run.

    Returns dict with:
        Lambda: global communication barrier estimate
        tau_predicted: theoretical optimal round trip rate
        tau_observed: empirical round trip rate
        efficiency: tau_observed / tau_predicted (closer to 1 = better)
        lambda_profile: local barrier at each pair midpoint
        round_trips_per_chain: per-machine round trip counts
        restarts_per_chain: per-machine restart counts
    """
    Lambda = estimate_global_barrier(rejection_rates)
    tau_pred = predict_optimal_round_trip_rate(Lambda)
    tau_obs = empirical_round_trip_rate(index_state, n_rounds)
    lambda_profile = estimate_local_barrier(rejection_rates, betas)

    return {
        "Lambda": Lambda,
        "tau_predicted": tau_pred,
        "tau_observed": tau_obs,
        "efficiency": tau_obs / jnp.maximum(tau_pred, 1e-10),
        "lambda_profile": lambda_profile,
        "round_trips_per_chain": index_state["round_trips"],
        "restarts_per_chain": index_state["restarts"],
    }


def recommend_n_chains(
    Lambda: float | jax.Array,
    target_acceptance: float = 0.6,
) -> int:
    """Suggest chain count for a given barrier and target acceptance rate.

    For NRPT with equalized rejection rates: Nr* ≈ Λ where r* = 1 - target_acceptance.
    Solving: N = Λ / r* = Λ / (1 - target_acceptance).

    The default target_acceptance=0.6 means 40% rejection per pair.

    **Warning**: Λ estimated from a run with too few chains is biased low,
    because the schedule can't resolve the peak in λ(β). If recommend_n_chains
    keeps increasing on successive calls, use ``discover_chain_count`` in
    ``nrpt.py`` which handles the bootstrapping problem iteratively.

    Args:
        Lambda: estimated global communication barrier
        target_acceptance: desired per-pair acceptance rate (default: 0.6 = 60%)

    Returns:
        Recommended number of chains (minimum 2).
    """
    r_star = 1.0 - target_acceptance
    n_opt = float(Lambda) / max(r_star, 0.01)
    return max(2, int(n_opt + 0.5))
