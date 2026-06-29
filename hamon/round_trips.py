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

    Args:
        index_state: current tracking dict
        perm: (n_chains,) int array — permutation applied to states
        n_chains: total number of chains
    """
    old_m2c = index_state["machine_to_chain"]
    visited = index_state["visited_top"]
    rts = index_state["round_trips"]
    restarts = index_state["restarts"]
    N = n_chains - 1

    # DEO swap permutations are products of disjoint transpositions → self-inverse.
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
    # Weak-typed scalars keep the result in Λ's dtype (a strong jnp.array(1.0)
    # would promote a float32 Λ to float64 under x64).
    return 1.0 / (2.0 + 2.0 * jnp.asarray(Lambda))


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


@jax.jit
def round_trip_summary(
    index_state: dict,
    rejection_rates: jax.Array,
    betas: jax.Array,
    n_rounds: int,
) -> dict:
    """Compute full diagnostic summary for NRPT run.

    Jitted so the handful of reductions below (Λ, τ̄, the local-barrier profile,
    the round-trip rate) fuse into a single compiled kernel instead of ~8 eager
    op-by-op dispatches, each of which otherwise pays a first-shape XLA compile
    when called once per probe at a new chain count. ``n_rounds`` is traced (not
    static), so the compile is shared across round counts.

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
    # int/int division in the empirical rate follows the x64 default dtype;
    # align it with the rejection-rate dtype the rest of the summary uses.
    tau_obs = empirical_round_trip_rate(index_state, n_rounds).astype(tau_pred.dtype)
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


# ---------------------------------------------------------------------------
# Normalizing constant (thermodynamic integration)
# ---------------------------------------------------------------------------


def thermodynamic_integration(
    betas: jax.Array,
    mean_energies: jax.Array,
    *,
    method: str = "trapezoid",
) -> jax.Array:
    r"""Log normalizing-constant ratio via thermodynamic integration.

    Estimates ``log Z(β_max) / Z(β_min) = -∫ μ(β) dβ`` (Syed et al. 2021,
    Sec. 5.5), where ``μ(β) = E_{π^(β)}[V]`` is the mean base energy and the
    integral runs over the supplied ladder ``[β_min, β_max]``. ``mean_energies``
    are the per-chain means μ(β_i) — accumulate them with
    [`hamon.NRPTEnergyObserver`][] and divide its ``(sum_E, count)`` carry, or
    use :func:`nrpt_log_normalizing_constant`.

    The reference chain (``β_min``, typically 0) has a known normalizer: for a
    discrete model with a uniform β=0 reference over ``M`` configurations,
    ``log Z(β_min) = log M`` (e.g. ``n·log 2`` for ``n`` spins), so the absolute
    ``log Z(β_max)`` is this result plus that constant.

    Args:
        betas: ascending β ladder, shape ``(n_chains,)``.
        mean_energies: per-chain mean base energy μ(β_i), shape ``(n_chains,)``.
        method: ``"trapezoid"`` (default, O(N⁻²)) or ``"riemann"`` — the
            right-Riemann sum of Syed et al. Eq. 5.5.

    Returns:
        Scalar ``log Z(β_max) / Z(β_min)``, in the dtype of ``mean_energies``.
    """
    betas = jnp.asarray(betas)
    mu = jnp.asarray(mean_energies)
    dbeta = jnp.diff(betas).astype(mu.dtype)
    if method == "trapezoid":
        integral = jnp.sum(0.5 * (mu[1:] + mu[:-1]) * dbeta)
    elif method == "riemann":
        integral = jnp.sum(mu[1:] * dbeta)
    else:
        raise ValueError(f"method must be 'trapezoid' or 'riemann', got {method!r}.")
    return -integral


def nrpt_log_normalizing_constant(
    stats: dict,
    *,
    log_z0: float = 0.0,
    method: str = "trapezoid",
) -> jax.Array:
    r"""Log normalizing constant from an NRPT run with an energy observer.

    Convenience over :func:`thermodynamic_integration`: reads the
    ``(sum_E, count)`` carry left by [`hamon.NRPTEnergyObserver`][] in
    ``stats["observer_carry"]``, forms the per-chain mean energies, and
    integrates them against ``stats["betas"]``.

    Args:
        stats: the stats dict from [`hamon.nrpt`][] / [`hamon.tune_schedule`][]
            run with ``observer=NRPTEnergyObserver(...)``.
        log_z0: ``log Z(β_min)`` of the reference chain, added to the integrated
            ratio to return the absolute ``log Z(β_max)``. Defaults to ``0``
            (returns the ratio ``log Z(β_max) / Z(β_min)``). For an ``n``-spin
            model with a β=0 uniform reference, pass ``n·log 2``.
        method: quadrature rule, see :func:`thermodynamic_integration`.

    Returns:
        Scalar ``log Z(β_max)`` (or the ratio when ``log_z0 = 0``).
    """
    if "observer_carry" not in stats:
        raise ValueError(
            "stats has no 'observer_carry'; run nrpt/tune_schedule with "
            "observer=NRPTEnergyObserver(n_chains) to accumulate mean energies."
        )
    sum_E, count = stats["observer_carry"]
    mean_energies = jnp.asarray(sum_E) / jnp.maximum(jnp.asarray(count), 1).astype(
        jnp.asarray(sum_E).dtype
    )
    logz = thermodynamic_integration(stats["betas"], mean_energies, method=method)
    return logz + jnp.asarray(log_z0, dtype=logz.dtype)


def recommend_n_chains(
    Lambda: float | jax.Array,
    target_acceptance: float = 0.5,
) -> int:
    """Suggest chain count for a given barrier and target acceptance rate.

    For NRPT with equalized rejection rates: Nr* ≈ Λ where r* = 1 - target_acceptance.
    Solving: N = Λ / r* = Λ / (1 - target_acceptance).

    The default target_acceptance=0.5 (r* = 1/2 ⇒ N* ≈ 2Λ) is the round-trip-
    optimal rejection rate from Syed et al., not the 0.77 of reversible PT.

    Note: Λ from too few chains is biased low. Use ``tune_chains``
    for iterative bootstrapping if recommendations keep increasing.

    Args:
        Lambda: estimated global communication barrier
        target_acceptance: desired per-pair acceptance rate (default: 0.5 = 50%)

    Returns:
        Recommended number of chains (minimum 2).
    """
    r_star = 1.0 - target_acceptance
    n_opt = float(Lambda) / max(r_star, 0.01)
    return max(2, int(n_opt + 0.5))
