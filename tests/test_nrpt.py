"""Integration tests for nrpt.py

Tests that the nrpt function:
- Passes all original API contracts
- Returns round trip diagnostics when track_round_trips=True
- Diagnostics have correct shapes and value ranges
- Round trips are counted over sufficient runs
- Lambda estimation is consistent with rejection rates
- track_round_trips=False omits diagnostics
- nrpt_adaptive returns tuning history
"""

import jax
import jax.numpy as jnp

from thrml_boost import Block, SpinNode, make_empty_block_state
from thrml_boost.models import IsingEBM, IsingSamplingProgram


# We import from the updated nrpt module
from thrml_boost.nrpt import nrpt, nrpt_adaptive, optimize_schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_ising(n_temps=2, betas=None):
    grid = [[SpinNode() for _ in range(2)] for _ in range(2)]
    nodes = [n for row in grid for n in row]
    edges = []
    for i in range(2):
        for j in range(2):
            edges.append((grid[i][j], grid[i][(j + 1) % 2]))
            edges.append((grid[i][j], grid[(i + 1) % 2][j]))
    even = [grid[i][j] for i in range(2) for j in range(2) if (i + j) % 2 == 0]
    odd = [grid[i][j] for i in range(2) for j in range(2) if (i + j) % 2 == 1]
    free_blocks = [Block(even), Block(odd)]
    biases = jnp.zeros(len(nodes))
    weights = jnp.zeros(len(edges))
    if betas is None:
        betas = [float(t) / n_temps for t in range(1, n_temps + 1)]
    ebms = [IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas]
    programs = [IsingSamplingProgram(e, free_blocks, clamped_blocks=[]) for e in ebms]
    init_state = make_empty_block_state(free_blocks, ebms[0].node_shape_dtypes)
    return nodes, edges, free_blocks, ebms, programs, init_state


def _ising_with_weights(n_nodes, betas, coupling=1.0):
    nodes = [SpinNode() for _ in range(n_nodes)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n_nodes - 1)]
    biases = jnp.zeros(n_nodes)
    weights = jnp.ones(n_nodes - 1) * coupling
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    ebms = [IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas]
    programs = [IsingSamplingProgram(e, free_blocks, clamped_blocks=[]) for e in ebms]
    init_state = [jnp.zeros(len(b), dtype=jnp.bool_) for b in free_blocks]
    return nodes, edges, free_blocks, ebms, programs, init_state


# ---------------------------------------------------------------------------
# API compatibility
# ---------------------------------------------------------------------------


class TestAPICompatibility:
    def test_basic_smoke(self):
        _, _, fb, ebms, progs, init = _tiny_ising(3)
        states, ss, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 3,
            [],
            n_rounds=5,
            gibbs_steps_per_round=2,
        )
        assert len(states) == 3
        assert "accepted" in stats
        assert "attempted" in stats

    def test_zero_rounds(self):
        _, _, fb, ebms, progs, init = _tiny_ising(2)
        states, _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 2,
            [],
            n_rounds=0,
            gibbs_steps_per_round=1,
        )
        assert jnp.all(stats["accepted"] == 0)


# ---------------------------------------------------------------------------
# Round trip diagnostics
# ---------------------------------------------------------------------------


class TestRoundTripDiagnostics:
    def test_diagnostics_present(self):
        """With track_round_trips=True, diagnostics should be in stats."""
        _, _, _, ebms, progs, init = _tiny_ising(4)
        _, _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 4,
            [],
            n_rounds=20,
            gibbs_steps_per_round=2,
            track_round_trips=True,
        )
        assert "round_trip_diagnostics" in stats
        diag = stats["round_trip_diagnostics"]
        assert "Lambda" in diag
        assert "tau_predicted" in diag
        assert "tau_observed" in diag
        assert "efficiency" in diag
        assert "lambda_profile" in diag
        assert "round_trips_per_chain" in diag
        assert "restarts_per_chain" in diag

    def test_diagnostics_absent_when_disabled(self):
        _, _, _, ebms, progs, init = _tiny_ising(3)
        _, _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 3,
            [],
            n_rounds=10,
            gibbs_steps_per_round=2,
            track_round_trips=False,
        )
        assert "round_trip_diagnostics" not in stats

    def test_index_state_shapes(self):
        _, _, _, ebms, progs, init = _tiny_ising(4)
        _, _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 4,
            [],
            n_rounds=10,
            gibbs_steps_per_round=2,
        )
        idx = stats["index_state"]
        assert idx["machine_to_chain"].shape == (4,)
        assert idx["visited_top"].shape == (4,)
        assert idx["round_trips"].shape == (4,)

    def test_lambda_matches_rejection(self):
        """Λ should equal sum of rejection rates."""
        _, _, _, ebms, progs, init = _ising_with_weights(8, [0.5, 1.0, 1.5, 2.0], 0.5)
        _, _, stats = nrpt(
            jax.random.key(42),
            ebms,
            progs,
            [init] * 4,
            [],
            n_rounds=50,
            gibbs_steps_per_round=3,
        )
        diag = stats["round_trip_diagnostics"]
        Lambda = diag["Lambda"]
        rej_sum = jnp.sum(stats["rejection_rates"])
        assert jnp.allclose(Lambda, rej_sum, atol=1e-5)

    def test_tau_predicted_bounded(self):
        """τ̄ should be in (0, 0.5]."""
        _, _, _, ebms, progs, init = _ising_with_weights(8, [0.5, 1.0, 2.0], 1.0)
        _, _, stats = nrpt(
            jax.random.key(7),
            ebms,
            progs,
            [init] * 3,
            [],
            n_rounds=30,
            gibbs_steps_per_round=2,
        )
        tau = stats["round_trip_diagnostics"]["tau_predicted"]
        assert 0.0 < float(tau) <= 0.5

    def test_round_trips_counted(self):
        """With many rounds and zero coupling, round trips should occur."""
        _, _, _, ebms, progs, init = _tiny_ising(4)
        # Zero weights → all swaps accepted → fast round trips
        _, _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 4,
            [],
            n_rounds=100,
            gibbs_steps_per_round=1,
        )
        total_rts = int(jnp.sum(stats["index_state"]["round_trips"]))
        assert total_rts > 0, "Expected round trips with zero-coupling model"

    def test_lambda_profile_shape(self):
        _, _, _, ebms, progs, init = _ising_with_weights(6, [0.3, 0.6, 1.0, 1.5], 0.5)
        _, _, stats = nrpt(
            jax.random.key(11),
            ebms,
            progs,
            [init] * 4,
            [],
            n_rounds=20,
            gibbs_steps_per_round=2,
        )
        lam = stats["round_trip_diagnostics"]["lambda_profile"]
        assert lam.shape == (3,)  # n_pairs = n_chains - 1


# ---------------------------------------------------------------------------
# Adaptive NRPT
# ---------------------------------------------------------------------------


class TestAdaptiveNRPT:
    def test_tuning_history(self):
        """nrpt_adaptive should return tuning history."""
        _, edges, fb, ebms, progs, init = _ising_with_weights(
            6,
            [0.5, 1.0, 1.5, 2.0],
            0.8,
        )

        def ebm_factory(betas):
            return [
                IsingEBM(
                    ebms[0].nodes,
                    ebms[0].edges,
                    ebms[0].biases,
                    ebms[0].weights,
                    jnp.array(float(b)),
                )
                for b in betas
            ]

        def prog_factory(e_list):
            return [IsingSamplingProgram(e, fb, []) for e in e_list]

        states, ss, stats = nrpt_adaptive(
            jax.random.key(0),
            ebm_factory,
            prog_factory,
            [init] * 4,
            [],
            n_rounds=20,
            gibbs_steps_per_round=2,
            initial_betas=jnp.array([0.5, 1.0, 1.5, 2.0]),
            n_tune=3,
            rounds_per_tune=10,
        )

        assert "tuning_history" in stats
        assert len(stats["tuning_history"]) == 3
        for entry in stats["tuning_history"]:
            assert "Lambda" in entry
            assert "rejection_rates" in entry


# ---------------------------------------------------------------------------
# Backward compatibility: schedule optimization
# ---------------------------------------------------------------------------


class TestScheduleOptimization:
    def test_preserves_endpoints(self):
        betas = jnp.array([0.1, 0.5, 1.0, 1.5, 2.0])
        rej = jnp.array([0.1, 0.5, 0.3, 0.8])
        new = optimize_schedule(rej, betas)
        assert new[0] == betas[0]
        assert new[-1] == betas[-1]

    def test_uniform_noop(self):
        betas = jnp.linspace(0.5, 2.0, 5)
        rej = jnp.array([0.3, 0.3, 0.3, 0.3])
        new = optimize_schedule(rej, betas)
        assert jnp.allclose(new, betas, atol=1e-5)
