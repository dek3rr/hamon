"""Tests for hamon.graph_utils graph colouring."""

import numpy as np
import pytest

from hamon.graph_utils import rlf_coloring


def _reference_rlf(n_nodes, edges):
    """Original set-based RLF (pre-vectorisation) — the semantic contract.

    The vectorised implementation must reproduce this colouring exactly,
    including all tie-breaking, so downstream block structures (and therefore
    sample streams) are unchanged.
    """
    adj = [set() for _ in range(n_nodes)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    colors = [-1] * n_nodes
    uncolored = set(range(n_nodes))
    k = 0
    while uncolored:
        U = set(uncolored)
        W = set()
        seed = min(U, key=lambda x, U=U: (-len(adj[x] & U), x))
        colors[seed] = k
        nbrs = adj[seed] & U
        U -= nbrs
        U.discard(seed)
        W |= nbrs
        while U:
            chosen = min(
                U, key=lambda x, U=U, W=W: (-len(adj[x] & W), len(adj[x] & U), x)
            )
            colors[chosen] = k
            nbrs = adj[chosen] & U
            U -= nbrs
            U.discard(chosen)
            W |= nbrs
        uncolored = {v for v in uncolored if colors[v] == -1}
        k += 1
    return colors


def _assert_proper(colors, edges):
    for u, v in edges:
        if u != v:
            assert colors[u] != colors[v], f"edge ({u},{v}) monochromatic"
    assert all(c >= 0 for c in colors)


def _lattice2d(L, periodic=True):
    edges = []
    for i in range(L):
        for j in range(L):
            v = i * L + j
            if j + 1 < L:
                edges.append((v, v + 1))
            elif periodic:
                edges.append((v, i * L))
            if i + 1 < L:
                edges.append((v, v + L))
            elif periodic:
                edges.append((v, j))
    return L * L, edges


def test_empty_graph():
    assert rlf_coloring(0, []) == []


def test_no_edges():
    assert rlf_coloring(4, []) == [0, 0, 0, 0]


def test_self_loops_and_duplicates_ignored():
    edges = [(0, 1), (1, 0), (0, 1), (2, 2)]
    colors = rlf_coloring(3, edges)
    assert colors == _reference_rlf(3, edges)
    assert colors[0] != colors[1]


def test_bipartite_lattice_two_colors():
    for L in (4, 8, 16):
        n, edges = _lattice2d(L)
        colors = rlf_coloring(n, edges)
        _assert_proper(colors, edges)
        assert max(colors) + 1 == 2, "even periodic lattice is 2-chromatic"


def test_odd_cycle_three_colors():
    n = 7
    edges = [(i, (i + 1) % n) for i in range(n)]
    colors = rlf_coloring(n, edges)
    _assert_proper(colors, edges)
    assert max(colors) + 1 == 3


def test_complete_graph_n_colors():
    n = 9
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    colors = rlf_coloring(n, edges)
    _assert_proper(colors, edges)
    assert max(colors) + 1 == n


def test_array_input_matches_list_input():
    n, edges = _lattice2d(6)
    from_list = rlf_coloring(n, edges)
    from_array = rlf_coloring(n, np.asarray(edges, dtype=np.int64))
    assert from_list == from_array


def test_deterministic():
    n, edges = _lattice2d(8)
    assert rlf_coloring(n, edges) == rlf_coloring(n, edges)


@pytest.mark.parametrize("seed", range(20))
def test_matches_reference_on_random_graphs(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 80))
    m = int(rng.integers(0, n * 4))
    edges = [
        (int(rng.integers(0, n)), int(rng.integers(0, n))) for _ in range(m)
    ]
    got = rlf_coloring(n, edges)
    assert got == _reference_rlf(n, edges)
    _assert_proper(got, [(u, v) for u, v in edges if u != v])


def test_matches_reference_on_lattice():
    n, edges = _lattice2d(10)
    assert rlf_coloring(n, edges) == _reference_rlf(n, edges)
