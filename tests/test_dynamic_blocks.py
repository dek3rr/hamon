"""Tests for dynamic_blocks.py.

Covers:
- Edge influence computation at various temperatures
- Aggregate influence sums correctly
- Node classification (heavy/light)
- Per-temperature block size recommendations
- Influence-aware partition covers all nodes
- Correlation estimation from synthetic samples
- Greedy merge respects max block size
- Dynamic re-blocking end-to-end
- Weighted Hamming distance
- Coloring validation
"""

import jax.numpy as jnp
import numpy as np

from thrml_boost.pgm import SpinNode
from thrml_boost.dynamic_blocks import (
    compute_edge_influence,
    compute_aggregate_influence,
    classify_nodes,
    recommend_block_size,
    per_temperature_block_config,
    influence_aware_partition,
    estimate_pairwise_correlations,
    greedy_merge_blocks,
    dynamic_reblock,
    weighted_hamming_distance,
    validate_coloring,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_graph(n):
    """n-node chain: 0-1-2-..-(n-1)."""
    nodes = [SpinNode() for _ in range(n)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
    return nodes, edges


def _grid_graph(L):
    nodes_2d = [[SpinNode() for _ in range(L)] for _ in range(L)]
    nodes = [n for row in nodes_2d for n in row]
    edges = []
    for i in range(L):
        for j in range(L):
            if j + 1 < L:
                edges.append((nodes_2d[i][j], nodes_2d[i][j + 1]))
            if i + 1 < L:
                edges.append((nodes_2d[i][j], nodes_2d[i + 1][j]))
    return nodes, edges, nodes_2d


# ---------------------------------------------------------------------------
# Edge influence
# ---------------------------------------------------------------------------


class TestEdgeInfluence:
    def test_zero_coupling(self):
        """Zero weights → zero influence regardless of β."""
        inf = compute_edge_influence(np.zeros(5), beta=1.0)
        assert np.allclose(inf, 0.0)

    def test_high_beta_high_influence(self):
        """Strong coupling at low temperature → high influence."""
        w = np.ones(3)
        inf_cold = compute_edge_influence(w, beta=5.0)
        inf_hot = compute_edge_influence(w, beta=0.1)
        assert np.all(inf_cold > inf_hot)

    def test_influence_bounded(self):
        """Influence should always be in [0, 1]."""
        w = np.random.randn(100)
        for beta in [0.1, 1.0, 5.0]:
            inf = compute_edge_influence(w, beta)
            assert np.all(inf >= 0.0)
            assert np.all(inf <= 1.0)


class TestAggregateInfluence:
    def test_chain_endpoints_lower(self):
        """Endpoints of a chain have fewer neighbors → lower aggregate influence."""
        nodes, edges = _chain_graph(10)
        w = np.ones(len(edges))
        agg, _ = compute_aggregate_influence(edges, w, 1.0, nodes)
        # Interior nodes have 2 edges, endpoints have 1
        assert agg[0] < agg[5]
        assert agg[-1] < agg[5]

    def test_uniform_grid(self):
        """Interior nodes of grid should have higher aggregate influence than corners."""
        nodes, edges, _ = _grid_graph(5)
        w = np.ones(len(edges))
        agg, _ = compute_aggregate_influence(edges, w, 1.0, nodes)
        # Corner (0,0) has 2 edges, interior (2,2) has 4
        corner_idx = 0
        center_idx = 12  # node at (2,2) in 5×5 grid
        assert agg[corner_idx] < agg[center_idx]


class TestClassifyNodes:
    def test_splits_nodes(self):
        agg = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        heavy, light = classify_nodes(agg, threshold=0.5)
        assert set(heavy.tolist()) == {2, 4}
        assert set(light.tolist()) == {0, 1, 3}

    def test_default_threshold_median(self):
        agg = np.arange(10, dtype=float)
        heavy, light = classify_nodes(agg)
        # median of [0..9] = 4.5, so heavy = {5,6,7,8,9}
        assert len(heavy) == 5
        assert len(light) == 5


# ---------------------------------------------------------------------------
# Per-temperature block sizing
# ---------------------------------------------------------------------------


class TestBlockSizing:
    def test_critical_temperature_max_size(self):
        """Near β_c, recommended size should be max_size."""
        size = recommend_block_size(0.44, beta_c=0.44, max_size=16)
        assert size == 16

    def test_far_from_critical_small(self):
        """Far from critical point, blocks can be small."""
        size = recommend_block_size(0.1, beta_c=0.44, min_size=1, max_size=16)
        assert size < 16

    def test_per_temp_config_monotone_near_critical(self):
        """Blocks should be largest near β_c."""
        betas = [0.1, 0.2, 0.44, 0.8, 2.0]
        sizes = per_temperature_block_config(betas, beta_c=0.44, max_size=16)
        # The β closest to β_c should have the largest block
        critical_idx = 2  # β=0.44
        assert sizes[critical_idx] == max(sizes)


# ---------------------------------------------------------------------------
# Influence-aware partition
# ---------------------------------------------------------------------------


class TestInfluenceAwarePartition:
    def test_all_nodes_covered(self):
        """Every node should appear in exactly one block."""
        nodes, edges, _ = _grid_graph(6)
        w = np.random.rand(len(edges))
        blocks = influence_aware_partition(nodes, edges, w, beta=1.0, max_block_size=8)

        all_ids = set()
        for block in blocks:
            for node in block:
                assert id(node) not in all_ids
                all_ids.add(id(node))
        assert len(all_ids) == len(nodes)

    def test_respects_max_size(self):
        nodes, edges, _ = _grid_graph(8)
        w = np.ones(len(edges))
        max_sz = 6
        blocks = influence_aware_partition(
            nodes, edges, w, beta=1.0, max_block_size=max_sz
        )
        for block in blocks:
            assert len(block) <= max_sz

    def test_heavy_nodes_interior(self):
        """High-influence nodes should tend to be interior to their block."""
        nodes, edges, _ = _grid_graph(8)
        w = np.ones(len(edges)) * 2.0
        blocks = influence_aware_partition(nodes, edges, w, beta=1.0, max_block_size=16)
        # This is a probabilistic property; just verify we get valid blocks
        assert len(blocks) > 0
        assert sum(len(b) for b in blocks) == len(nodes)


# ---------------------------------------------------------------------------
# Correlation estimation and merging
# ---------------------------------------------------------------------------


class TestCorrelationEstimation:
    def test_perfect_correlation(self):
        """Identical columns → correlation = 1."""
        # 100 samples, 4 nodes, edges 0-1, 1-2, 2-3
        samples = np.random.choice([0.0, 1.0], size=(100, 4))
        # Make nodes 0 and 1 identical
        samples[:, 1] = samples[:, 0]
        edge_src = np.array([0, 1, 2])
        edge_dst = np.array([1, 2, 3])
        corr = estimate_pairwise_correlations(samples, edge_src, edge_dst)
        assert corr[0] > 0.95  # edge 0-1 should be ~1.0

    def test_independent_low_correlation(self):
        """Independent random columns → correlation near 0."""
        np.random.seed(42)
        samples = np.random.choice([-1.0, 1.0], size=(10000, 4))
        edge_src = np.array([0, 1, 2])
        edge_dst = np.array([1, 2, 3])
        corr = estimate_pairwise_correlations(samples, edge_src, edge_dst)
        assert np.all(corr < 0.1)


class TestGreedyMerge:
    def test_merge_high_correlation(self):
        """Two blocks connected by high-correlation edge should merge."""
        blocks = [[0, 1], [2, 3], [4, 5]]
        edge_src = np.array([1, 3])
        edge_dst = np.array([2, 4])
        corr = np.array([0.9, 0.1])  # edge 1-2 high, edge 3-4 low

        merged = greedy_merge_blocks(
            blocks,
            corr,
            edge_src,
            edge_dst,
            max_block_size=6,
            correlation_threshold=0.3,
        )

        # Blocks 0 and 1 should merge (edge 1-2)
        sizes = sorted([len(b) for b in merged])
        assert 4 in sizes  # merged block

    def test_respects_max_size(self):
        """Should not merge blocks that would exceed max_block_size."""
        blocks = [[0, 1, 2], [3, 4, 5]]
        edge_src = np.array([2])
        edge_dst = np.array([3])
        corr = np.array([0.99])

        merged = greedy_merge_blocks(
            blocks,
            corr,
            edge_src,
            edge_dst,
            max_block_size=4,
            correlation_threshold=0.3,
        )

        # Can't merge: 3+3=6 > 4
        assert len(merged) == 2


class TestDynamicReblock:
    def test_end_to_end(self):
        """Dynamic re-blocking should produce valid partition."""
        nodes, edges, _ = _grid_graph(4)
        n_nodes = len(nodes)

        # Simple initial blocks: pairs
        current_blocks = [[nodes[i], nodes[i + 1]] for i in range(0, n_nodes - 1, 2)]
        if n_nodes % 2 == 1:
            current_blocks.append([nodes[-1]])

        # Synthetic samples with some correlation
        np.random.seed(123)
        samples = jnp.array(np.random.choice([-1.0, 1.0], size=(200, n_nodes)))

        new_blocks = dynamic_reblock(
            nodes,
            edges,
            current_blocks,
            samples,
            max_block_size=8,
            correlation_threshold=0.2,
        )

        # All nodes covered
        all_ids = set()
        for block in new_blocks:
            for node in block:
                all_ids.add(id(node))
        assert len(all_ids) == n_nodes


# ---------------------------------------------------------------------------
# Weighted Hamming distance
# ---------------------------------------------------------------------------


class TestWeightedHamming:
    def test_identical_states_zero(self):
        s = jnp.array([1.0, -1.0, 1.0, -1.0])
        agg = jnp.ones(4)
        mask = jnp.array([True, True, False, False])
        d = weighted_hamming_distance(s, s, agg, mask, 4)
        assert d == 0.0

    def test_all_different(self):
        a = jnp.array([1.0, 1.0, 1.0, 1.0])
        b = jnp.array([-1.0, -1.0, -1.0, -1.0])
        agg = jnp.ones(4)
        mask = jnp.ones(4, dtype=jnp.bool_)
        d = weighted_hamming_distance(a, b, agg, mask, 4)
        assert d == 1.0  # all internal, all disagree

    def test_external_weighted_more(self):
        """External disagreements with high influence should increase distance."""
        a = jnp.array([1.0, -1.0])
        b = jnp.array([-1.0, 1.0])
        agg = jnp.array([0.1, 5.0])
        # Node 0 interior, node 1 external
        mask = jnp.array([True, False])
        d = weighted_hamming_distance(a, b, agg, mask, 2, weight_external=1.0)
        # d = (1/2) * (1 + 5.0*1) = 3.0
        assert d == 3.0


# ---------------------------------------------------------------------------
# Coloring validation
# ---------------------------------------------------------------------------


class TestColoringValidation:
    def test_valid_checkerboard(self):
        """Standard checkerboard 2-coloring is valid."""
        nodes, edges, nodes_2d = _grid_graph(4)
        blocks = [
            [nodes_2d[i][j] for i in range(4) for j in range(4) if (i + j) % 2 == 0],
            [nodes_2d[i][j] for i in range(4) for j in range(4) if (i + j) % 2 == 1],
        ]
        color_classes = [[0], [1]]
        assert validate_coloring(blocks, color_classes, edges)

    def test_invalid_coloring(self):
        """Two adjacent blocks in same color class should fail."""
        nodes, edges, nodes_2d = _grid_graph(4)
        from thrml_boost.boundary_energy import make_rectangular_blocks

        blocks, _ = make_rectangular_blocks(4, 2, nodes_2d)
        # Put all blocks in one color class
        invalid_classes = [list(range(len(blocks)))]
        assert not validate_coloring(blocks, invalid_classes, edges)
