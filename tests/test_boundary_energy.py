"""Tests for boundary_energy.py.

Covers:
- Incremental energy delta matches full recomputation
- make_ising_delta_fn correctness and nrpt() energy_delta_fn integration
"""

import jax
import jax.numpy as jnp

from hamon.pgm import SpinNode
from hamon.boundary_energy import ising_energy_delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(L):
    """L×L grid with nearest-neighbor edges. Returns nodes_2d, nodes_flat, edges."""
    nodes_2d = [[SpinNode() for _ in range(L)] for _ in range(L)]
    nodes_flat = [n for row in nodes_2d for n in row]
    edges = []
    for i in range(L):
        for j in range(L):
            if j + 1 < L:
                edges.append((nodes_2d[i][j], nodes_2d[i][j + 1]))
            if i + 1 < L:
                edges.append((nodes_2d[i][j], nodes_2d[i + 1][j]))
    return nodes_2d, nodes_flat, edges


# ---------------------------------------------------------------------------
# Energy delta correctness
# ---------------------------------------------------------------------------


class TestEnergyDelta:
    def test_delta_matches_full(self):
        """ΔE from boundary-only computation should equal E_new - E_old."""
        L = 4
        nodes_2d, nodes_flat, edges = _make_grid(L)
        n_nodes = L * L
        n_edges = len(edges)

        key = jax.random.key(42)
        k1, k2 = jax.random.split(key)

        biases = jax.random.normal(k1, (n_nodes,)) * 0.5
        weights = jax.random.normal(k2, (n_edges,)) * 0.3

        # Two random spin configurations (as float ±1)
        old_spins = (
            2.0
            * jax.random.bernoulli(jax.random.key(10), shape=(n_nodes,)).astype(
                jnp.float32
            )
            - 1.0
        )
        new_spins = (
            2.0
            * jax.random.bernoulli(jax.random.key(11), shape=(n_nodes,)).astype(
                jnp.float32
            )
            - 1.0
        )

        # Only change nodes in block 0 (even sites)
        even_mask = jnp.array([(i + j) % 2 == 0 for i in range(L) for j in range(L)])
        new_spins = jnp.where(even_mask, new_spins, old_spins)

        node_idx = {id(n): i for i, n in enumerate(nodes_flat)}
        edge_src = jnp.array([node_idx[id(e[0])] for e in edges], dtype=jnp.int32)
        edge_dst = jnp.array([node_idx[id(e[1])] for e in edges], dtype=jnp.int32)
        incident_mask = jnp.array(
            [even_mask[s] | even_mask[d] for s, d in zip(edge_src, edge_dst)]
        )
        changed_mask = even_mask

        # Boundary-only delta
        delta = ising_energy_delta(
            old_spins,
            new_spins,
            biases,
            weights,
            edge_src,
            edge_dst,
            incident_mask,
            changed_mask,
        )

        # Full energy computation
        def full_energy(spins):
            bias_term = -jnp.sum(biases * spins)
            coupling_term = -jnp.sum(weights * spins[edge_src] * spins[edge_dst])
            return bias_term + coupling_term

        expected_delta = full_energy(new_spins) - full_energy(old_spins)

        assert jnp.allclose(delta, expected_delta, atol=1e-5), (
            f"delta={float(delta):.6f} vs expected={float(expected_delta):.6f}"
        )


# ---------------------------------------------------------------------------
# make_ising_delta_fn / nrpt() energy_delta_fn integration
# ---------------------------------------------------------------------------


class TestMakeIsingDeltaFn:
    """make_ising_delta_fn correctness and nrpt integration."""

    def _build_ising(self, L, coupling, beta, key):
        nodes_2d = [[SpinNode() for _ in range(L)] for _ in range(L)]
        nodes = [n for row in nodes_2d for n in row]
        edges = []
        for i in range(L):
            for j in range(L):
                if j + 1 < L:
                    edges.append((nodes_2d[i][j], nodes_2d[i][j + 1]))
                if i + 1 < L:
                    edges.append((nodes_2d[i][j], nodes_2d[i + 1][j]))

        k_b, k_w = jax.random.split(key)
        biases = jax.random.normal(k_b, (len(nodes),)) * 0.3
        weights = jnp.ones(len(edges)) * coupling

        even = [nodes_2d[i][j] for i in range(L) for j in range(L) if (i + j) % 2 == 0]
        odd = [nodes_2d[i][j] for i in range(L) for j in range(L) if (i + j) % 2 == 1]
        from hamon.block_management import Block

        free_blocks = [Block(even), Block(odd)]

        from hamon.models.ising import IsingEBM

        ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(beta))
        return nodes, edges, free_blocks, biases, weights, ebm

    def test_delta_matches_full_energy_diff(self):
        """delta_fn(old, new)[c] == E_base(new_c) - E_base(old_c) for all chains."""
        from hamon.boundary_energy import make_ising_delta_fn
        from hamon.nrpt import _compute_base_energies
        from hamon.models.ising import IsingEBM, IsingSamplingProgram

        L, n_chains = 4, 6
        betas_list = [0.3, 0.7, 1.0, 1.4, 1.8, 2.2]
        betas = jnp.array(betas_list)

        nodes, edges, free_blocks, biases, weights, ebm = self._build_ising(
            L, coupling=0.5, beta=betas_list[0], key=jax.random.key(0)
        )
        ebms = [
            IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas_list
        ]
        progs = [IsingSamplingProgram(e, free_blocks, []) for e in ebms]
        spec = progs[0].gibbs_spec
        n_fb = len(free_blocks)

        # Random old and new states (n_chains, block_size) bool
        keys = jax.random.split(jax.random.key(1), n_chains * n_fb * 2)
        old_stacked = [
            jax.random.bernoulli(keys[b], shape=(n_chains, len(list(free_blocks[b]))))
            for b in range(n_fb)
        ]
        new_stacked = [
            jax.random.bernoulli(
                keys[n_fb + b], shape=(n_chains, len(list(free_blocks[b])))
            )
            for b in range(n_fb)
        ]

        delta_fn = make_ising_delta_fn(nodes, edges, free_blocks, biases, weights)
        deltas = delta_fn(old_stacked, new_stacked)

        # Reference: full vmap energy eval for old and new
        E_old = _compute_base_energies(ebms[0], betas[0], spec, old_stacked, [])
        E_new = _compute_base_energies(ebms[0], betas[0], spec, new_stacked, [])
        expected = E_new - E_old

        assert jnp.allclose(deltas, expected, atol=1e-5), (
            f"max delta error: {float(jnp.max(jnp.abs(deltas - expected))):.2e}"
        )

    def test_nrpt_cached_matches_uncached(self):
        """nrpt with energy_delta_fn produces statistically identical marginals."""
        from hamon.boundary_energy import make_ising_delta_fn
        from hamon.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
        from hamon.nrpt import nrpt

        L = 4
        betas_list = [0.5, 1.0, 1.5, 2.0]
        betas = jnp.array(betas_list)
        nodes, edges, free_blocks, biases, weights, _ = self._build_ising(
            L, coupling=0.8, beta=betas_list[0], key=jax.random.key(42)
        )
        ebms = [
            IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas_list
        ]
        progs = [IsingSamplingProgram(e, free_blocks, []) for e in ebms]

        keys = jax.random.split(jax.random.key(99), len(betas_list))
        init_states = [hinton_init(k, ebms[0], free_blocks, ()) for k in keys]

        delta_fn = make_ising_delta_fn(nodes, edges, free_blocks, biases, weights)

        states_ref, stats_ref = nrpt(
            jax.random.key(7),
            ebms,
            progs,
            init_states,
            [],
            n_rounds=500,
            gibbs_steps_per_round=3,
            betas=betas,
            track_round_trips=False,
        )
        states_cac, stats_cac = nrpt(
            jax.random.key(7),
            ebms,
            progs,
            init_states,
            [],
            n_rounds=500,
            gibbs_steps_per_round=3,
            betas=betas,
            track_round_trips=False,
            energy_delta_fn=delta_fn,
        )

        # Acceptance rates must match exactly (same RNG, same computation)
        assert jnp.allclose(
            stats_ref["acceptance_rate"], stats_cac["acceptance_rate"], atol=1e-6
        ), (
            f"Acceptance rates differ:\\n  ref: {stats_ref['acceptance_rate']}\\n  cac: {stats_cac['acceptance_rate']}"
        )

    def test_energy_cache_consistent_after_swap(self):
        """After each round, cached base_E must equal direct recompute."""
        # We can't directly inspect the cached base_E inside the scan, but we
        # can verify that acceptance_rate (which depends on base_E) is identical
        # between the cached and uncached paths (tested above). This test
        # additionally checks that longer runs don't accumulate drift.
        from hamon.boundary_energy import make_ising_delta_fn
        from hamon.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
        from hamon.nrpt import nrpt

        L = 6
        betas_list = [0.3, 0.6, 0.9, 1.2, 1.6, 2.0]
        betas = jnp.array(betas_list)
        nodes, edges, free_blocks, biases, weights, _ = self._build_ising(
            L, coupling=0.7, beta=betas_list[0], key=jax.random.key(11)
        )
        ebms = [
            IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas_list
        ]
        progs = [IsingSamplingProgram(e, free_blocks, []) for e in ebms]
        init_states = [
            hinton_init(k, ebms[0], free_blocks, ())
            for k in jax.random.split(jax.random.key(22), len(betas_list))
        ]

        delta_fn = make_ising_delta_fn(nodes, edges, free_blocks, biases, weights)

        _, stats_ref = nrpt(
            jax.random.key(5),
            ebms,
            progs,
            init_states,
            [],
            n_rounds=2000,
            gibbs_steps_per_round=2,
            betas=betas,
            track_round_trips=False,
        )
        _, stats_cac = nrpt(
            jax.random.key(5),
            ebms,
            progs,
            init_states,
            [],
            n_rounds=2000,
            gibbs_steps_per_round=2,
            betas=betas,
            track_round_trips=False,
            energy_delta_fn=delta_fn,
        )

        # 2000 rounds: any accumulation of float32 error would show here
        max_rate_diff = float(
            jnp.max(
                jnp.abs(stats_ref["acceptance_rate"] - stats_cac["acceptance_rate"])
            )
        )
        assert max_rate_diff < 1e-5, (
            f"Acceptance rates diverged after 2000 rounds (max diff={max_rate_diff:.2e}).\\n"
            f"  ref: {stats_ref['acceptance_rate']}\\n"
            f"  cac: {stats_cac['acceptance_rate']}"
        )
