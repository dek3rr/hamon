# hamon/graph_utils.py
"""
Graph-coloring utilities for automatic construction of sampling order.

The key entry point is :func:`auto_color_blocks`, which inspects a set of
:class:`~hamon.InteractionGroup` objects and returns a ``free_super_blocks``
list ready to pass directly into :class:`~hamon.BlockGibbsSpec`.

Background
----------
Block Gibbs sampling only requires that blocks within the *same* sampling group
are conditionally independent — i.e. no block in the group appears as a tail
(neighbor) of any other block in the group.  Deciding which blocks can share a
group is equivalent to graph coloring: nodes are blocks, and there is an edge
between two blocks whenever one block's nodes appear in an interaction that
affects the other block.  Each color class becomes one ``SuperBlock``
(sampling group), and blocks in the same color class are updated
simultaneously from the same global-state snapshot.

Fewer color groups → fewer sequential write-back barriers per scan step →
faster wall-clock time.  The greedy algorithm used here gives the *chromatic
number* for bipartite graphs (the most common case in practice) and produces a
good coloring for general graphs, though it is not globally optimal — graph
coloring is NP-hard.

The 4-color theorem does not apply: interaction graphs are rarely planar (one
on ``n`` nodes with more than ``3n - 6`` edges cannot be).  The bounds that do
apply are ``χ ≤ Δ + 1`` (Brooks) and the usually much tighter
``χ ≤ degeneracy + 1``.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from hamon.block_management import Block
from hamon.interaction import InteractionGroup

# Re-export the SuperBlock type alias so callers only need one import.
from hamon.block_sampling import SuperBlock  # noqa: F401


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

        def select(v: int, in_U=in_U, cU=cU) -> int:
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


def auto_color_blocks(
    free_blocks: Sequence[Block],
    interaction_groups: Sequence[InteractionGroup],
) -> list[SuperBlock]:
    """Derive a minimal parallel sampling order from the interaction graph.

    Analyzes which free blocks interact with which others and returns a list of
    ``SuperBlock`` values (each either a plain :class:`~hamon.Block` or a
    tuple of :class:`~hamon.Block` objects) that can be passed directly to
    :class:`~hamon.BlockGibbsSpec` as ``free_super_blocks``.

    Blocks assigned to the same ``SuperBlock`` are conditionally independent —
    their nodes never appear in each other's ``tail_nodes`` — so they can safely
    be updated simultaneously from the same global-state snapshot.

    The algorithm runs at program-construction time (Python, no JAX tracing) and
    is O(|blocks|² + |interaction_groups| · |block_sizes|) — negligible compared
    with the sampling loop.

    **Arguments:**

    - ``free_blocks``: The free blocks whose sampling order you want to optimize.
      The order of this list is preserved within each color group, so the
      resulting ``BlockGibbsSpec`` will have the same ``free_blocks`` ordering.
    - ``interaction_groups``: The compiled interactions for your program (e.g.
      the output of ``factor.to_interaction_groups()``).  Only interactions whose
      *head* nodes belong to ``free_blocks`` contribute to the conflict graph.

    **Returns:**

    A list of ``SuperBlock`` values.  Pass this directly to
    ``BlockGibbsSpec(free_super_blocks=..., ...)``.

    **Example** — 1-D Ising chain::

        nodes  = [SpinNode() for _ in range(5)]
        edges  = [(nodes[i], nodes[i + 1]) for i in range(4)]
        model  = IsingEBM(nodes, edges, ...)

        even   = Block(nodes[::2])   # {0, 2, 4}
        odd    = Block(nodes[1::2])  # {1, 3}

        # Every chain edge links an even node to an odd one, so the two
        # blocks conflict and must be sequential sampling groups;
        # auto_color_blocks derives that order from the interaction groups:
        igs    = [f.to_interaction_groups() for f in model.factors]
        igs    = [g for sublist in igs for g in sublist]
        super_blocks = auto_color_blocks([even, odd], igs)
        # => [even, odd]  — the conflict is detected; two sequential groups
        spec   = BlockGibbsSpec(super_blocks, clamped_blocks=[])
    """
    free_blocks = list(free_blocks)
    n = len(free_blocks)

    if n == 0:
        return []

    # -------------------------------------------------------------------------
    # Step 1 — map each node to its block index for O(1) lookup.
    # -------------------------------------------------------------------------
    node_to_block: dict = {}
    for block_idx, block in enumerate(free_blocks):
        for node in block.nodes:
            node_to_block[node] = block_idx

    # -------------------------------------------------------------------------
    # Step 2 — build a conflict adjacency set.
    #
    # Two blocks conflict if one block's nodes appear as tail_nodes in an
    # interaction whose head_nodes belong to the other block, or vice versa.
    # The conflict relation is symmetric: if A influences B then B and A
    # cannot safely be co-updated (updating A changes the input B would read
    # if they were in the same group, violating the shared-snapshot contract).
    # -------------------------------------------------------------------------
    conflicts: set[tuple[int, int]] = set()

    for ig in interaction_groups:
        # Identify which free block (if any) owns the head nodes.
        head_block_indices: set[int] = set()
        for node in ig.head_nodes.nodes:
            idx = node_to_block.get(node)
            if idx is not None:
                head_block_indices.add(idx)

        # Identify which free blocks own any tail nodes.
        tail_block_indices: set[int] = set()
        for tail_block in ig.tail_nodes:
            for node in tail_block.nodes:
                idx = node_to_block.get(node)
                if idx is not None:
                    tail_block_indices.add(idx)

        # Every (head, tail) pair where head ≠ tail is a conflict.
        for h in head_block_indices:
            for t in tail_block_indices:
                if h != t:
                    conflicts.add((h, t))
                    conflicts.add((t, h))  # symmetric

    # -------------------------------------------------------------------------
    # Step 3 — color the block-conflict graph with Recursive Largest First.
    #
    # RLF minimizes the color count (= number of sequential sampling groups,
    # which sets the round-loop compile cost) more aggressively than first-fit,
    # especially on dense conflict graphs, and matches it on sparse/bipartite
    # ones. Optimal for bipartite graphs; a good heuristic otherwise.
    # -------------------------------------------------------------------------
    color = rlf_coloring(n, conflicts)

    # -------------------------------------------------------------------------
    # Step 4 — group blocks by color, preserving original order within groups.
    # -------------------------------------------------------------------------
    color_groups: dict[int, list[Block]] = defaultdict(list)
    for block_idx in range(n):  # ascending index preserves original order
        color_groups[color[block_idx]].append(free_blocks[block_idx])

    # Return color groups in ascending color order so the sampling sequence
    # is deterministic and independent of dict iteration order.
    result: list[SuperBlock] = []
    for c in sorted(color_groups):
        group = color_groups[c]
        if len(group) == 1:
            result.append(group[0])
        else:
            result.append(tuple(group))

    return result
