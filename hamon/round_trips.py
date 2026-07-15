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
import numpy as np


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
    n_chains: int | jax.Array,
) -> dict:
    """Update the index process after a swap pass.

    Args:
        index_state: current tracking dict
        perm: (n_chains,) int array — permutation applied to states
        n_chains: total number of chains; may be a traced scalar (the live
            count of a padded ladder), which redefines the round-trip "top"
            without changing array shapes
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


# Saturation floor for the barrier estimate. Every rejection rate is <= 1, so
# Λ̂ = Σ rej <= n_pairs = N-1 *arithmetically*: when pairs pin at r = 1 the
# estimate reports its own cap, not the barrier.
#
# Calibrated (2026-07-15, RTX 5080, tune_schedule with the plateau stop,
# n_rounds=4000, N swept 3..96) against a converged large-N reference Λ̂ on
# three families — planted frustrated-loop 32x32 (Λ≈21.6), random ±J glass
# 24x24 (Λ≈18.5), ferromagnet 16x16 (Λ≈10.3). The max(rej) → Λ̂-error relation
# is monotone and consistent across all three:
#   1.000-0.992 → capped: −36% .. −86% (Λ̂ = N-1 exactly at N = 4, 8)
#   0.903 / 0.870 / 0.837 → −29% / −19% / −14%
#   0.782 → −11%
#   0.689 / 0.638 / 0.569 → −9% / −8% / −4%
#   <= 0.465 → <= 4%
# The two populations are separated by a measured gap: ladders the tuner itself
# converges to (design point r* = 0.5, N* ≈ 2Λ) top out at max(rej) ≈ 0.69,
# while the under-provisioned band starts at 0.78. 0.75 sits in that gap —
# >= 0.06 above every observed working ladder (also clearing per-pair estimator
# noise: σ(r̂) ≈ 0.02 at 500 attempts, max over <= 127 pairs adds ~ +0.05) and
# below the first under-provisioned reading — and corresponds to a Λ̂
# underestimate of ~10%, the point where the error stops being absorbable by
# tune_chains' 5% safety margin plus its ±1-chain convergence tolerance.
# So "resolved" ⇒ Λ̂ within ~10%.
#
# Re-validated in higher dimensions (same protocol; 3-D ferromagnet 8³ Λ≈14.4,
# 3-D ±J glass 8³ Λ≈17.4 — a different hardness class, with a finite-T
# spin-glass transition unlike any 2-D family — and 4-D ±J 4⁴ Λ≈12.8): the
# sharper barrier concentration of higher d does NOT push healthy ladders over
# the threshold. Working band tops at 0.64; under-provisioned band starts at
# 0.773-0.775 (−12.6% / −16.7% error) — every rejected row errs >= 12.6%, every
# admitted row <= 8.4%, bracketing the ~10% claim. Note the upper margin is
# thin (0.773 vs 0.75 ≈ 1σ of per-pair noise), so a ~12%-error ladder can
# occasionally read resolved; that direction is safe — tune_chains drives N*
# off the running MAX of Λ̂ over probes, which a slightly-low probe cannot drag
# down.
#
# Continuous-state limit (q=16 clock ≡ 2-D XY model, 16², J=2, Λ≈15.7 — the
# continuous limit hamon's uint8 states can express: continuous U(1) symmetry,
# KT transition, spin waves; sampler verified exact against enumeration): the
# mechanism replicates (caps exact at N-1, curve monotone, budget-invariant),
# but the smooth spin-wave spectrum COMPRESSES the max(rej) scale — an
# under-provisioned ladder shows lower max(rej) for the same fractional error —
# so the boundary row (0.691 → −11.0%) is the one admitted row past 10% across
# all six families. The honest guarantee is therefore "resolved ⇒ Λ̂ within
# ~10-12%". Tightening to 0.70 would instead false-alarm on measured healthy
# ladders (2-D working band reaches 0.705), a worse trade.
_MAX_REJ_RESOLVED = 0.75

# Efficiency floor for the *conveyor* check (a different question — see
# :func:`conveyor_is_alive`). Same calibration sweeps, efficiency = τ_obs/τ_pred
# measured at n_rounds=4000 (past the transient, see below):
#   genuinely stalled (no trips, saturated ladder):     exactly 0.000
#   saturation-driven crawl (max rej 0.84-0.99):        0.034 - 0.135
#   every ladder at its design N*, all three families:  0.206 - 0.558
#     (the ±J glass family plateaus lowest, 0.21-0.30 — a floor of 0.25 would
#      false-alarm on a healthy spin glass, which is why it is 0.15)
# 0.15 splits the crawl band from the design band, sitting ~1.3σ below the
# weakest healthy reading (Poisson noise on 22 observed trips ≈ 0.044).
# Higher-d re-validation (3-D ferro/±J 8³, 4-D ±J 4⁴): healthy plateaus
# 0.181-0.691, stalls/crawls 0.000-0.069 — the floor holds, with the thinnest
# healthy margin at the 3-D glass near design N (0.181, so ±0.04 Poisson noise
# can occasionally trip a spurious "slow" warning there; WARNING-level only).
# Continuous limit (q=16 clock/XY): healthy plateaus 0.337-0.719 — the fastest
# conveyor measured anywhere, as the physics predicts (the KT phase is critical,
# not broken into discrete basins, so nothing traps the index process).
_MIN_EFFICIENCY = 0.15

# Expected round trips (τ_pred · n_rounds) the window must afford before τ_obs
# is a verdict rather than noise. Two grounds, both measured at fixed N=47 on
# the planted 32x32 (τ_pred ≈ 0.023):
#  - transient: the index process starts from a fresh permutation, so the first
#    completed trips only arrive after a full traversal; measured efficiency vs
#    expected-trips: 11.6 → 0.000, 23 → 0.043, 46 → 0.240, 92 → 0.272,
#    276 → 0.348 — readings below ~40 expected trips are transient-suppressed.
#  - statistics: at the 0.15 floor, 40 expected trips mean a barely-alive
#    conveyor shows ~6 on average, so observing *zero* is a P ≈ e⁻⁶ ≈ 0.25%
#    event — a stall verdict is then conclusive, not an unlucky window. (The
#    numeric efficiency estimate stays noisy near the minimum window — ~40%
#    relative at 6 trips — but the None/False boundary only has to protect
#    against false stall verdicts.)
# Higher-d re-validation (3-D ±J 8³ at fixed design N=48): a HEALTHY ladder
# reads efficiency 0.144 — below the 0.15 floor — at 27.8 expected trips, and
# the 40-trip guard is what suppresses that would-be false "slow" verdict
# (None); by 55 expected trips the reading clears to 0.217 and the verdict is
# correct. The 3-D transient ends by ~55 expected trips vs ~46 in 2-D, so at
# the 40 boundary a healthy ladder reads ~0.18-0.24 in both — above the floor.
_MIN_EXPECTED_TRIPS = 40.0


def barrier_is_identified(
    rejection_rates: jax.Array | np.ndarray,
    *,
    max_rej_resolved: float = _MAX_REJ_RESOLVED,
) -> bool:
    """Whether ``Λ̂ = Σ rejection_rates`` is a trustworthy barrier estimate.

    Λ̂ is **cap-limited**, not merely noisy, when the ladder saturates: each
    rejection rate is at most 1, so ``Λ̂ <= N-1`` by construction and a ladder
    whose pairs pin at ``r = 1`` reports that cap rather than the barrier (an
    unbridged pair also blocks the DEO conveyor outright, freezing every chain —
    and every replica — in one basin, which is why replicas *agree* on the
    artifact and low cross-replica spread is a false consistency signal). So the
    question "is Λ̂ resolved?" is **structural**: it is answered by the ladder's
    own rejection rates, and needs no round-trip observation.

    This deliberately does **not** gate on the round-trip rate. That signal is
    *budget-dependent* and answers a different question (:func:`conveyor_is_alive`):
    measured at fixed N = 47 on a planted 32x32, the same well-resolved ladder
    (Λ̂ ≈ 20.7, max rej ≈ 0.6) reports 0 round trips at 500 rounds and 96 at
    12000 — the zero-trip reading is indistinguishable from a genuine stall, so
    using it here reports "add chains" for a ladder that is already correct.

    Returns ``True`` when Λ̂ is resolved. ``False`` means the ladder is
    saturating and Λ̂ is a cap-limited underestimate — add chains / equalize
    rather than believing it.
    """
    # np.max accepts both host numpy (the diagnostics path) and jax arrays (the
    # tuning path — a tiny 1-D host read, once per run).
    return bool(float(np.max(np.asarray(rejection_rates))) < max_rej_resolved)


def conveyor_is_alive(
    tau_observed: float,
    tau_predicted: float,
    n_rounds: int,
    *,
    min_efficiency: float = _MIN_EFFICIENCY,
    min_expected_trips: float = _MIN_EXPECTED_TRIPS,
) -> bool | None:
    """Whether the DEO index process is observed traversing the ladder.

    A *dynamical* question, distinct from :func:`barrier_is_identified`: a ladder
    can be perfectly resolved yet not (yet) observed round-tripping, and the two
    dissociate cleanly under budget — Λ̂ and max(rej) are flat across a 24x round
    sweep while efficiency swings 0.000 → 0.348.

    Gates on **efficiency** ``τ_obs / τ_pred``, not on ``τ_obs`` itself: with
    ``τ_pred = 1/(2+2Λ)`` an absolute floor demands an efficiency that grows with
    Λ (0.01 asks 19% of optimal at Λ = 8.5, 43% at Λ = 21, and is unsatisfiable
    at Λ >= 49 — strictest exactly where a trustworthy answer matters most).

    Returns ``None`` when the window is too short to tell (fewer than
    ``min_expected_trips`` round trips expected even at the optimal rate), so
    callers can say "not measured" instead of "stalled".
    """
    if tau_predicted * max(n_rounds, 0) < min_expected_trips:
        return None
    if tau_predicted <= 0.0:
        return None
    return bool(tau_observed / tau_predicted >= min_efficiency)


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
