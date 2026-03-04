"""Tests for round_trips.py.

Covers:
- Index state initialization
- Permutation-based position tracking
- Round trip counting (visit chain N, then return to chain 0)
- Restart counting (first visit to chain N)
- Identity permutation changes nothing
- Communication barrier estimation from rejection rates
- Optimal round trip rate prediction
- Diagnostics summary
- Chain count recommendation
"""

import jax.numpy as jnp

from thrml_boost.round_trips import (
    init_index_state,
    update_index_state,
    estimate_local_barrier,
    estimate_global_barrier,
    predict_optimal_round_trip_rate,
    empirical_round_trip_rate,
    round_trip_summary,
    recommend_n_chains,
)


class TestInitIndexState:
    def test_shapes(self):
        state = init_index_state(5)
        assert state["machine_to_chain"].shape == (5,)
        assert state["visited_top"].shape == (5,)
        assert state["round_trips"].shape == (5,)
        assert state["restarts"].shape == (5,)

    def test_initial_values(self):
        state = init_index_state(4)
        assert jnp.array_equal(state["machine_to_chain"], jnp.arange(4))
        assert jnp.all(~state["visited_top"])
        assert jnp.all(state["round_trips"] == 0)
        assert jnp.all(state["restarts"] == 0)


class TestUpdateIndexState:
    def test_identity_perm_no_change(self):
        """Identity permutation should not move any machine."""
        state = init_index_state(4)
        perm = jnp.arange(4)
        new = update_index_state(state, perm, 4)
        assert jnp.array_equal(new["machine_to_chain"], jnp.arange(4))
        assert jnp.all(new["round_trips"] == 0)

    def test_single_swap(self):
        """Swap pair (1,2) should exchange machines at those positions."""
        state = init_index_state(4)
        # perm: new[0]=old[0], new[1]=old[2], new[2]=old[1], new[3]=old[3]
        perm = jnp.array([0, 2, 1, 3])
        new = update_index_state(state, perm, 4)
        # Machine 1 was at chain 1, now at chain 2 (inv_perm[1] = 2)
        # Machine 2 was at chain 2, now at chain 1 (inv_perm[2] = 1)
        assert int(new["machine_to_chain"][1]) == 2
        assert int(new["machine_to_chain"][2]) == 1
        # Machines 0 and 3 unchanged
        assert int(new["machine_to_chain"][0]) == 0
        assert int(new["machine_to_chain"][3]) == 3

    def test_restart_on_reaching_top(self):
        """Machine reaching chain N should trigger a restart."""
        n = 4
        state = {
            "machine_to_chain": jnp.array([0, 1, 2, 3]),
            "visited_top": jnp.zeros(n, dtype=jnp.bool_),
            "round_trips": jnp.zeros(n, dtype=jnp.int32),
            "restarts": jnp.zeros(n, dtype=jnp.int32),
        }
        # Swap (2,3): machine 2 goes to chain 3 (=N)
        perm = jnp.array([0, 1, 3, 2])
        new = update_index_state(state, perm, n)
        assert bool(new["visited_top"][2])
        assert int(new["restarts"][2]) == 1
        # Machine 3 goes to chain 2, not at top
        assert not bool(new["visited_top"][3])

    def test_round_trip_completion(self):
        """Machine that visited top and returns to chain 0 completes a trip."""
        n = 4
        state = {
            "machine_to_chain": jnp.array([1, 0, 3, 2]),
            "visited_top": jnp.array([True, False, True, False]),
            "round_trips": jnp.zeros(n, dtype=jnp.int32),
            "restarts": jnp.ones(n, dtype=jnp.int32),
        }
        # Swap (0,1): machine 0 (at chain 1) goes to chain 0
        perm = jnp.array([1, 0, 2, 3])
        new = update_index_state(state, perm, n)
        # Machine 0: was at chain 1, visited_top=True, now at chain 0 → round trip!
        assert int(new["round_trips"][0]) == 1
        # visited_top should reset after completing
        assert not bool(new["visited_top"][0])

    def test_no_false_round_trip(self):
        """Machine at chain 0 that hasn't visited top should not count."""
        n = 3
        state = {
            "machine_to_chain": jnp.array([1, 0, 2]),
            "visited_top": jnp.array([False, False, False]),
            "round_trips": jnp.zeros(n, dtype=jnp.int32),
            "restarts": jnp.zeros(n, dtype=jnp.int32),
        }
        # Swap (0,1): machine 1 at chain 0 stays or swaps
        perm = jnp.array([1, 0, 2])
        new = update_index_state(state, perm, n)
        # Machine 1 goes from chain 0 to chain 1, no round trip
        # Machine 0 goes from chain 1 to chain 0, but hasn't visited top
        assert int(new["round_trips"][0]) == 0

    def test_full_conveyor_belt(self):
        """Simulate all-accepted DEO on 4 chains to get a round trip."""
        n = 4
        state = init_index_state(n)

        # Even swap: pairs (0,1) and (2,3) accepted
        perm_even = jnp.array([1, 0, 3, 2])
        state = update_index_state(state, perm_even, n)
        # M0: 0→1, M1: 1→0, M2: 2→3(=N), M3: 3→2
        assert int(state["machine_to_chain"][0]) == 1
        assert int(state["machine_to_chain"][2]) == 3  # at top
        assert int(state["restarts"][2]) == 1

        # Odd swap: pair (1,2) accepted
        perm_odd = jnp.array([0, 2, 1, 3])
        state = update_index_state(state, perm_odd, n)
        # M0 was at 1, now at 2. M2 was at 3, stays at 3 (pair 1,2 doesn't touch 3)
        # Wait: inv_perm of [0,2,1,3] is [0,2,1,3]
        # M0 at chain 1 → inv_perm[1] = 2 → M0 now at chain 2
        # M1 at chain 0 → inv_perm[0] = 0 → M1 stays at chain 0
        # M2 at chain 3 → inv_perm[3] = 3 → M2 stays at chain 3
        # M3 at chain 2 → inv_perm[2] = 1 → M3 now at chain 1
        assert int(state["machine_to_chain"][0]) == 2

        # Even swap again: pairs (0,1) and (2,3)
        state = update_index_state(state, perm_even, n)
        # M0 at 2 → inv[2]=3 → M0 at 3 (top!)
        assert int(state["machine_to_chain"][0]) == 3
        assert bool(state["visited_top"][0])

        # Odd swap: pair (1,2)
        state = update_index_state(state, perm_odd, n)
        # M0 at 3 → inv[3]=3 → stays at 3

        # Even swap
        state = update_index_state(state, perm_even, n)
        # M0 at 3 → inv[3]=2 → M0 at 2

        # Odd swap
        state = update_index_state(state, perm_odd, n)
        # M0 at 2 → inv[2]=1 → M0 at 1

        # Even swap
        state = update_index_state(state, perm_even, n)
        # M0 at 1 → inv[1]=0 → M0 at 0! visited_top=True → round trip!
        assert int(state["machine_to_chain"][0]) == 0
        assert int(state["round_trips"][0]) >= 1


class TestBarrierEstimation:
    def test_local_barrier(self):
        betas = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        rej = jnp.array([0.1, 0.2, 0.3, 0.4])
        lam = estimate_local_barrier(rej, betas)
        assert jnp.allclose(lam, jnp.array([0.4, 0.8, 1.2, 1.6]))

    def test_global_barrier(self):
        rej = jnp.array([0.1, 0.2, 0.3])
        Lambda = estimate_global_barrier(rej)
        assert jnp.allclose(Lambda, 0.6)

    def test_optimal_rate_zero_barrier(self):
        """Λ=0 → τ̄ = 1/2 (perfect conveyor belt)."""
        tau = predict_optimal_round_trip_rate(0.0)
        assert jnp.allclose(tau, 0.5)

    def test_optimal_rate_positive_barrier(self):
        """Λ=1 → τ̄ = 1/4."""
        tau = predict_optimal_round_trip_rate(1.0)
        assert jnp.allclose(tau, 0.25)


class TestDiagnostics:
    def test_summary_keys(self):
        idx_state = init_index_state(4)
        rej = jnp.array([0.1, 0.2, 0.3])
        betas = jnp.linspace(0.5, 2.0, 4)
        summary = round_trip_summary(idx_state, rej, betas, 100)

        expected_keys = {
            "Lambda",
            "tau_predicted",
            "tau_observed",
            "efficiency",
            "lambda_profile",
            "round_trips_per_chain",
            "restarts_per_chain",
        }
        assert expected_keys == set(summary.keys())

    def test_empirical_rate_zero_rounds(self):
        idx_state = init_index_state(4)
        rate = empirical_round_trip_rate(idx_state, 0)
        assert rate == 0.0

    def test_recommend_n_chains(self):
        n = recommend_n_chains(2.0, target_acceptance=0.6)
        assert n == 5

    def test_recommend_minimum(self):
        n = recommend_n_chains(0.01, target_acceptance=0.99)
        assert n >= 2
