# hamon/graph_utils.py
"""
Graph-coloring utilities for automatic construction of sampling order.

The 4-color theorem does not apply: interaction graphs are rarely planar (one
on ``n`` nodes with more than ``3n - 6`` edges cannot be).  The bounds that do
apply are ``χ ≤ Δ + 1`` (Brooks) and the usually much tighter
``χ ≤ degeneracy + 1``.
"""

from collections.abc import Iterable

import numpy as np


def rlf_coloring(n_nodes: int, edges: Iterable[tuple[int, int]]) -> list[int]:
    """Color an integer-indexed graph with Recursive Largest First (RLF).

    Nodes are ``0 .. n_nodes - 1``; ``edges`` are (u, v) index pairs (direction
    and duplicates ignored, self-loops dropped) — any iterable of pairs or an
    ``(m, 2)`` integer array. Returns ``colors`` where ``colors[i]`` is the
    color class of node ``i``.

    RLF builds each color class as a maximal independent set, repeatedly adding
    the vertex with the most neighbors already excluded from the class (ties
    broken toward fewest remaining-candidate neighbors, then smallest index).
    It minimizes the number of colors more aggressively than first-fit/greedy
    heuristics on dense graphs and matches them on sparse/bipartite ones — and
    in hamon the color count is the number of sequential block-Gibbs groups,
    which sets the NRPT round-loop XLA compile cost. Deterministic (index
    tie-breaking) so colorings are reproducible.

    Carries no worst-case bound and can exceed ``degeneracy + 1``, though
    rarely enough on real interaction graphs that a smallest-last fallback
    was measured and did not earn its keep.

    The selection rule is evaluated with incrementally maintained neighbor
    counters over a CSR adjacency and a vectorized argmax, so the cost is
    O(|V|² elementwise + |V|·χ + |E|·χ) in NumPy rather than O(|V|·|E|) in
    Python-set operations. The coloring produced is identical to the
    reference set-based implementation for any input.
    """
    if n_nodes == 0:
        return []

    # --- normalize edges to a deduplicated undirected (m, 2) index array ---
    e = np.asarray(
        edges if isinstance(edges, np.ndarray) else list(edges), dtype=np.int64
    )
    e = e.reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]  # drop self-loops
    if e.shape[0]:
        e = np.unique(np.sort(e, axis=1), axis=0)

    if e.shape[0] == 0:
        return [0] * n_nodes  # no conflicts: everything in one class

    # --- CSR adjacency (both directions) ---
    und = np.concatenate([e, e[:, ::-1]])
    order = np.argsort(und[:, 0], kind="stable")
    csr_dst = np.ascontiguousarray(und[order, 1])
    degrees = np.bincount(und[:, 0], minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(degrees, out=indptr[1:])

    NEG = np.iinfo(np.int64).min
    # Lexicographic keys packed into one int64 score (primary * BASE ±
    # secondary, BASE > any secondary); np.argmax's first-maximum is exactly
    # the reference implementation's smallest-index tie-break.
    BASE = np.int64(n_nodes + 1)

    colors = np.full(n_nodes, -1, dtype=np.int64)
    uncolored = np.ones(n_nodes, dtype=bool)
    n_uncolored = n_nodes

    # gU[x] = |adj(x) ∩ uncolored| is maintained across classes so each class
    # starts with cU = gU.copy() instead of the O(|E|) recount that made dense
    # graphs quadratic in |E|.
    gU = degrees.astype(np.int64)
    cW = np.zeros(n_nodes, dtype=np.int64)  # |adj(x) ∩ W| (excluded nbrs)
    key = np.empty(n_nodes, dtype=np.int64)

    k = 0
    while n_uncolored:
        # U: vertices that may still join this color class; W: their already-
        # excluded neighbors. cU[x] = |adj(x) ∩ U| is maintained exactly:
        # decremented once for every neighbor of x that leaves U.
        in_U = uncolored.copy()
        n_U = n_uncolored
        cU = gU.copy()
        cW.fill(0)

        def select(v: int, in_U=in_U, cU=cU, k=k) -> int:
            """Color v, expel its U-neighbors to W, update counters/keys."""
            colors[v] = k
            in_U[v] = False
            key[v] = NEG
            nb = csr_dst[indptr[v] : indptr[v + 1]]
            cU[nb] -= 1  # v left U
            gU[nb] -= 1  # ... and left the uncolored set for good
            moved = nb[in_U[nb]]
            removed = 1 + moved.size
            if moved.size:
                in_U[moved] = False
                key[moved] = NEG
                # Gather the moved vertices' CSR rows as one ragged range:
                # for row i, positions starts[i] .. starts[i]+lens[i)-1.
                starts = indptr[moved]
                lens = indptr[moved + 1] - starts
                mnb = csr_dst[
                    np.arange(int(lens.sum()))
                    + np.repeat(starts + lens - np.cumsum(lens), lens)
                ]
                np.add.at(cU, mnb, -1)  # moved left U
                np.add.at(cW, mnb, 1)  # ... and joined W
                touched = np.concatenate([nb, mnb])
            else:
                touched = nb
            # Refresh packed keys for vertices whose counters changed.
            key[touched] = np.where(
                in_U[touched], cW[touched] * BASE - cU[touched], NEG
            )
            return removed

        # Seed with the highest-degree vertex in the remaining subgraph
        # (max cU, then smallest index).
        seed = int(np.argmax(np.where(in_U, cU, NEG)))
        np.copyto(key, np.where(in_U, cW * BASE - cU, NEG))
        n_U -= select(seed)

        # Grow the class: max |adj ∩ W|, then min |adj ∩ U|, then min index.
        while n_U:
            n_U -= select(int(np.argmax(key)))

        newly = colors == k
        n_uncolored -= int(np.count_nonzero(newly))
        uncolored &= ~newly
        k += 1

    return colors.tolist()
