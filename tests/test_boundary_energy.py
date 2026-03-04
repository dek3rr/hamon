"""Tests for boundary_energy.py.

Covers:
- Edge classification (boundary vs interior vs external)
- Checkerboard blocks: all edges are boundary (ratio = 1.0)
- Rectangular blocks: interior edges exist (ratio < 1.0)
- Incremental energy delta matches full recomputation
- Rectangular block construction and 4-coloring
- Savings estimation
- Edge index pre-computation
"""

import jax
import jax.numpy as jnp

from thrml_boost.pgm import SpinNode
from thrml_boost.boundary_energy import (
    EdgePartition,
    ising_energy_delta,
    make_rectangular_blocks,
    estimate_boundary_savings,
    precompute_edge_indices,
)


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


def _checkerboard_blocks(nodes_2d, L):
    """Standard 2-coloring for L×L grid."""
    even = [nodes_2d[i][j] for i in range(L) for j in range(L) if (i + j) % 2 == 0]
    odd = [nodes_2d[i][j] for i in range(L) for j in range(L) if (i + j) % 2 == 1]
    return [even, odd]


# ---------------------------------------------------------------------------
# Edge classification
# ---------------------------------------------------------------------------


class TestEdgePartition:
    def test_checkerboard_all_boundary(self):
        """For checkerboard on a grid, every edge crosses the boundary."""
        L = 4
        nodes_2d, nodes_flat, edges = _make_grid(L)
        blocks = _checkerboard_blocks(nodes_2d, L)

        ep = EdgePartition(edges, blocks)

        # Every edge has one endpoint in each color → all boundary
        for b_idx in range(2):
            assert ep.n_interior[b_idx] == 0
            assert ep.n_boundary[b_idx] == len(edges)
            assert ep.boundary_ratio[b_idx] == 1.0

    def test_rectangular_has_interior(self):
        """2×2 blocks on 4×4 grid should have interior edges."""
        L = 4
        nodes_2d, nodes_flat, edges = _make_grid(L)
        blocks, color_classes = make_rectangular_blocks(L, 2, nodes_2d)

        ep = EdgePartition(edges, blocks)

        # 2×2 blocks have 1 horizontal + 1 vertical interior edge each
        has_interior = any(ep.n_interior[b] > 0 for b in range(len(blocks)))
        assert has_interior

    def test_savings_factor(self):
        """Rectangular blocks should have savings_factor < 1."""
        L = 8
        nodes_2d, nodes_flat, edges = _make_grid(L)
        blocks, _ = make_rectangular_blocks(L, 4, nodes_2d)

        ep = EdgePartition(edges, blocks)

        for b_idx in range(len(blocks)):
            sf = ep.savings_factor(b_idx)
            assert 0.0 < sf < 1.0, f"Block {b_idx}: savings_factor={sf}"

    def test_external_edges_exist(self):
        """Each block should have external edges (untouched by its update)."""
        L = 8
        nodes_2d, nodes_flat, edges = _make_grid(L)
        blocks, _ = make_rectangular_blocks(L, 2, nodes_2d)

        ep = EdgePartition(edges, blocks)

        for b_idx in range(len(blocks)):
            assert ep.n_external[b_idx] > 0


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

        # Pre-compute edge indices
        idx = precompute_edge_indices(
            nodes_flat, edges, _checkerboard_blocks(nodes_2d, L)
        )
        edge_src = jnp.array(idx["edge_src"])
        edge_dst = jnp.array(idx["edge_dst"])
        incident_mask = jnp.array(idx["incident_masks"][0])
        changed_mask = jnp.array(idx["changed_masks"][0])

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
# Rectangular blocks
# ---------------------------------------------------------------------------


class TestRectangularBlocks:
    def test_coverage(self):
        """All nodes should be assigned to exactly one block."""
        L = 8
        nodes_2d, nodes_flat, edges = _make_grid(L)
        blocks, color_classes = make_rectangular_blocks(L, 3, nodes_2d)

        all_nodes = set()
        for block in blocks:
            for node in block:
                assert id(node) not in all_nodes, "Node in multiple blocks"
                all_nodes.add(id(node))
        assert len(all_nodes) == L * L

    def test_four_color_classes(self):
        """Should produce exactly 4 color classes."""
        L = 8
        nodes_2d, _, _ = _make_grid(L)
        _, color_classes = make_rectangular_blocks(L, 2, nodes_2d)
        assert len(color_classes) == 4

    def test_all_blocks_assigned(self):
        """Every block index should appear in exactly one color class."""
        L = 6
        nodes_2d, _, _ = _make_grid(L)
        blocks, color_classes = make_rectangular_blocks(L, 2, nodes_2d)
        all_indices = set()
        for cc in color_classes:
            for idx in cc:
                assert idx not in all_indices
                all_indices.add(idx)
        assert all_indices == set(range(len(blocks)))

    def test_block_size_respects_boundary(self):
        """Blocks at grid edges should be smaller if L not divisible by m."""
        L = 7
        nodes_2d, _, _ = _make_grid(L)
        blocks, _ = make_rectangular_blocks(L, 3, nodes_2d)
        # Some blocks should have < 9 nodes
        sizes = [len(b) for b in blocks]
        assert min(sizes) < 9  # boundary blocks


class TestSavingsEstimate:
    def test_checkerboard_no_savings(self):
        est = estimate_boundary_savings(16, 1)
        assert est["checkerboard_savings"] == 1.0

    def test_larger_blocks_lower_boundary_ratio(self):
        """Larger blocks have lower boundary/incident ratio per block."""
        est2 = estimate_boundary_savings(32, 2)
        est8 = estimate_boundary_savings(32, 8)
        assert est8["boundary_ratio_per_block"] < est2["boundary_ratio_per_block"]


class TestEdgeIndexPrecomputation:
    def test_shapes(self):
        L = 4
        nodes_2d, nodes_flat, edges = _make_grid(L)
        blocks = _checkerboard_blocks(nodes_2d, L)
        idx = precompute_edge_indices(nodes_flat, edges, blocks)

        assert idx["edge_src"].shape == (len(edges),)
        assert idx["edge_dst"].shape == (len(edges),)
        assert len(idx["incident_masks"]) == 2
        assert len(idx["changed_masks"]) == 2
        assert idx["node_to_block"].shape == (L * L,)
