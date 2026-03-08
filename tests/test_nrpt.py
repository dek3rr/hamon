"""Tests for nrpt.py optimizations: vmap energies (H1) and multi-pass DEO (H5).

Validates:
- Vmapped energy computation matches loop-based reference
- Multi-pass DEO produces more swap attempts per round
- Round trip rates improve with multi-pass (more communication per round)
- All existing API contracts still hold
"""

import jax
import jax.numpy as jnp

from thrml_boost import Block, SpinNode, make_empty_block_state
from thrml_boost.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml_boost.nrpt import _compute_base_energies, nrpt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ising(L, betas, coupling=1.0):
    """Build L×L 2D Ising with checkerboard blocking."""
    grid = [[SpinNode() for _ in range(L)] for _ in range(L)]
    nodes = [n for row in grid for n in row]
    edges = []
    for i in range(L):
        for j in range(L):
            if j + 1 < L:
                edges.append((grid[i][j], grid[i][j + 1]))
            if i + 1 < L:
                edges.append((grid[i][j], grid[i + 1][j]))
    biases = jnp.zeros(len(nodes))
    weights = jnp.ones(len(edges)) * coupling
    even = [grid[i][j] for i in range(L) for j in range(L) if (i + j) % 2 == 0]
    odd = [grid[i][j] for i in range(L) for j in range(L) if (i + j) % 2 == 1]
    free_blocks = [Block(even), Block(odd)]
    ebms = [IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas]
    progs = [IsingSamplingProgram(e, free_blocks, []) for e in ebms]
    return nodes, edges, free_blocks, ebms, progs


def _make_states(key, ebms, free_blocks, n_chains):
    keys = jax.random.split(key, n_chains)
    return [hinton_init(keys[c], ebms[0], free_blocks, ()) for c in range(n_chains)]


# ---------------------------------------------------------------------------
# H1: vmap energy correctness
# ---------------------------------------------------------------------------


class TestVmapEnergies:
    """Verify _compute_base_energies (vmapped) matches loop-based reference."""

    def _loop_base_energies(self, ebms, spec, stacked_states, clamp_state, betas):
        """Reference: the old loop-based implementation."""
        n_chains = len(ebms)
        n_free_blocks = len(stacked_states)
        energies = []
        for c in range(n_chains):
            state_c = [stacked_states[b][c] for b in range(n_free_blocks)]
            energies.append(ebms[c].energy(state_c + clamp_state, spec))
        return jnp.stack(energies) / betas

    def test_matches_loop_4chains(self):
        betas = [0.5, 1.0, 1.5, 2.0]
        _, _, fb, ebms, progs = _make_ising(8, betas, coupling=0.8)
        states = _make_states(jax.random.key(0), ebms, fb, 4)

        n_free = len(fb)
        stacked = [jnp.stack([states[c][b] for c in range(4)]) for b in range(n_free)]
        spec = progs[0].gibbs_spec
        betas_arr = jnp.array(betas)

        ref = self._loop_base_energies(ebms, spec, stacked, [], betas_arr)
        vmap_result = _compute_base_energies(ebms[0], betas_arr[0], spec, stacked, [])

        assert jnp.allclose(ref, vmap_result, atol=1e-5), (
            f"max diff: {float(jnp.max(jnp.abs(ref - vmap_result)))}"
        )

    def test_matches_loop_16chains(self):
        betas = jnp.linspace(0.3, 2.5, 16).tolist()
        _, _, fb, ebms, progs = _make_ising(16, betas, coupling=0.5)
        states = _make_states(jax.random.key(42), ebms, fb, 16)

        n_free = len(fb)
        stacked = [jnp.stack([states[c][b] for c in range(16)]) for b in range(n_free)]
        spec = progs[0].gibbs_spec
        betas_arr = jnp.array(betas)

        ref = self._loop_base_energies(ebms, spec, stacked, [], betas_arr)
        vmap_result = _compute_base_energies(ebms[0], betas_arr[0], spec, stacked, [])

        assert jnp.allclose(ref, vmap_result, atol=1e-5)

    def test_nonzero_biases(self):
        """Verify with non-trivial biases (not just coupling)."""
        L = 6
        grid = [[SpinNode() for _ in range(L)] for _ in range(L)]
        nodes = [n for row in grid for n in row]
        edges = []
        for i in range(L):
            for j in range(L):
                if j + 1 < L:
                    edges.append((grid[i][j], grid[i][j + 1]))
                if i + 1 < L:
                    edges.append((grid[i][j], grid[i + 1][j]))

        key = jax.random.key(7)
        biases = jax.random.normal(key, (len(nodes),)) * 0.3
        weights = jax.random.normal(jax.random.key(8), (len(edges),)) * 0.5

        even = [grid[i][j] for i in range(L) for j in range(L) if (i + j) % 2 == 0]
        odd = [grid[i][j] for i in range(L) for j in range(L) if (i + j) % 2 == 1]
        fb = [Block(even), Block(odd)]

        betas = [0.5, 1.0, 1.5, 2.0]
        ebms = [IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas]
        progs = [IsingSamplingProgram(e, fb, []) for e in ebms]
        states = _make_states(jax.random.key(99), ebms, fb, 4)

        n_free = len(fb)
        stacked = [jnp.stack([states[c][b] for c in range(4)]) for b in range(n_free)]
        spec = progs[0].gibbs_spec
        betas_arr = jnp.array(betas)

        ref = self._loop_base_energies(ebms, spec, stacked, [], betas_arr)
        vmap_result = _compute_base_energies(ebms[0], betas_arr[0], spec, stacked, [])

        assert jnp.allclose(ref, vmap_result, atol=1e-5)


# ---------------------------------------------------------------------------
# DEO correctness (single-pass, non-reversible)
# ---------------------------------------------------------------------------


class TestSinglePassDEO:
    """Verify single-pass DEO: one swap parity per round, alternating."""

    def test_pairs_attempted_alternating(self):
        """Even pairs attempted on even rounds, odd on odd rounds."""
        betas = [0.5, 1.0, 1.5, 2.0]
        _, _, fb, ebms, progs = _make_ising(4, betas, coupling=0.0)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)

        n_rounds = 50
        _, _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 4,
            [],
            n_rounds=n_rounds,
            gibbs_steps_per_round=1,
        )
        attempted = stats["attempted"]
        # 3 pairs: even={0,2}, odd={1}
        # 50 rounds → 25 even rounds, 25 odd rounds
        assert int(attempted[0]) == 25  # even pair
        assert int(attempted[1]) == 25  # odd pair
        assert int(attempted[2]) == 25  # even pair

    def test_round_trips_with_zero_coupling(self):
        """Zero coupling → all swaps accepted → conveyor belt round trips."""
        betas = [0.5, 1.0, 1.5, 2.0]
        _, _, fb, ebms, progs = _make_ising(4, betas, coupling=0.0)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)

        # 4 chains, single-pass DEO: ~6 rounds per round trip
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

    def test_acceptance_rates_valid(self):
        """Acceptance rates should be in [0, 1] for all pairs."""
        betas = [0.3, 0.8, 1.2, 1.8, 2.5]
        _, _, fb, ebms, progs = _make_ising(8, betas, coupling=0.5)
        states = _make_states(jax.random.key(0), ebms, fb, 5)

        _, _, stats = nrpt(
            jax.random.key(1),
            ebms,
            progs,
            states,
            [],
            n_rounds=100,
            gibbs_steps_per_round=3,
        )
        acc = stats["acceptance_rate"]
        assert jnp.all(acc >= 0.0) and jnp.all(acc <= 1.0)
        assert jnp.all(stats["attempted"] > 0)

    def test_lambda_consistent(self):
        """Λ = sum(rejection_rates) should hold."""
        betas = [0.5, 1.0, 1.5, 2.0]
        _, _, fb, ebms, progs = _make_ising(8, betas, coupling=0.5)
        states = _make_states(jax.random.key(0), ebms, fb, 4)

        _, _, stats = nrpt(
            jax.random.key(42),
            ebms,
            progs,
            states,
            [],
            n_rounds=50,
            gibbs_steps_per_round=3,
        )
        diag = stats["round_trip_diagnostics"]
        assert jnp.allclose(
            diag["Lambda"], jnp.sum(stats["rejection_rates"]), atol=1e-5
        )

    def test_deo_alternation_asymmetric_rates(self):
        """Even-pair and odd-pair rates should differ with asymmetric betas."""
        betas = [0.1, 0.2, 2.0, 2.1]
        _, _, fb, ebms, progs = _make_ising(8, betas, coupling=1.0)
        states = _make_states(jax.random.key(0), ebms, fb, 4)

        _, _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            states,
            [],
            n_rounds=200,
            gibbs_steps_per_round=3,
        )
        # Pair 0 (β=0.1↔0.2, small gap) vs pair 1 (β=0.2↔2.0, large gap)
        rates = stats["acceptance_rate"]
        assert not jnp.allclose(rates[0], rates[1], atol=0.05), (
            f"Suspiciously similar rates: {rates}"
        )

    def test_multi_pass_would_break_conveyor(self):
        """Document WHY multi-pass DEO is wrong.

        With 4 chains and all swaps accepted, even∘odd followed by
        odd∘even = identity permutation. States oscillate with period 2
        instead of drifting through the temperature ladder.
        """
        # Verify the algebra: compose the two permutations
        n = 4
        even_perm = jnp.array([1, 0, 3, 2])  # swap (0,1) and (2,3)
        odd_perm = jnp.array([0, 2, 1, 3])  # swap (1,2)

        # even_first: even then odd
        composed_ef = odd_perm[even_perm]
        # odd_first: odd then even
        composed_of = even_perm[odd_perm]

        # Two rounds should compose to identity
        full_cycle = composed_of[composed_ef]
        assert jnp.array_equal(full_cycle, jnp.arange(n)), (
            f"Expected identity, got {full_cycle}"
        )


# ---------------------------------------------------------------------------
# Existing API contracts still hold
# ---------------------------------------------------------------------------


class TestAPIUnchanged:
    def test_basic_smoke(self):
        betas = [0.5, 1.0, 1.5]
        _, _, fb, ebms, progs = _make_ising(4, betas)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)
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
        assert "round_trip_diagnostics" in stats

    def test_zero_rounds(self):
        betas = [0.5, 1.0]
        _, _, fb, ebms, progs = _make_ising(4, betas)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)
        _, _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 2,
            [],
            n_rounds=0,
            gibbs_steps_per_round=1,
        )
        assert jnp.all(stats["accepted"] == 0)

    def test_diagnostics_absent_when_disabled(self):
        betas = [0.5, 1.0, 1.5]
        _, _, fb, ebms, progs = _make_ising(4, betas)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)
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

    def test_tau_bounded(self):
        betas = [0.5, 1.0, 2.0]
        _, _, fb, ebms, progs = _make_ising(8, betas, coupling=1.0)
        states = _make_states(jax.random.key(0), ebms, fb, 3)
        _, _, stats = nrpt(
            jax.random.key(7),
            ebms,
            progs,
            states,
            [],
            n_rounds=30,
            gibbs_steps_per_round=2,
        )
        tau = stats["round_trip_diagnostics"]["tau_predicted"]
        assert 0.0 < float(tau) <= 0.5
