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
(neighbour) of any other block in the group.  Deciding which blocks can share a
group is equivalent to graph colouring: nodes are blocks, and there is an edge
between two blocks whenever one block's nodes appear in an interaction that
affects the other block.  Each colour class becomes one ``SuperBlock``
(sampling group), and blocks in the same colour class are updated
simultaneously from the same global-state snapshot.

Fewer colour groups → fewer sequential write-back barriers per scan step →
faster wall-clock time.  The greedy algorithm used here gives the *chromatic
number* for bipartite graphs (the most common case in practice) and produces a
good colouring for general graphs, though it is not globally optimal for all
graph topologies.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence

from hamon.block_management import Block
from hamon.interaction import InteractionGroup

# Re-export the SuperBlock type alias so callers only need one import.
from hamon.block_sampling import SuperBlock  # noqa: F401


def rlf_coloring(n_nodes: int, edges: Iterable[tuple[int, int]]) -> list[int]:
    """Colour an integer-indexed graph with Recursive Largest First (RLF).

    Nodes are ``0 .. n_nodes - 1``; ``edges`` are (u, v) index pairs (direction
    and duplicates ignored, self-loops dropped). Returns ``colors`` where
    ``colors[i]`` is the colour class of node ``i``.

    RLF builds each colour class as a maximal independent set, repeatedly adding
    the vertex with the most neighbours already excluded from the class (ties
    broken toward fewest remaining-candidate neighbours, then smallest index).
    It minimises the number of colours more aggressively than first-fit/greedy
    heuristics on dense graphs and matches them on sparse/bipartite ones — and
    in hamon the colour count is the number of sequential block-Gibbs groups,
    which sets the NRPT round-loop XLA compile cost. Runtime is O(|V|·|E|);
    deterministic (index tie-breaking) so colourings are reproducible.
    """
    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    colors = [-1] * n_nodes
    uncolored = set(range(n_nodes))
    k = 0
    while uncolored:
        # U: vertices that may still join this colour class; W: their already-
        # excluded neighbours (each adjacent to a class member).
        U = set(uncolored)
        W: set[int] = set()

        # Seed with the highest-degree vertex in the remaining subgraph.
        seed = min(U, key=lambda x, U=U: (-len(adj[x] & U), x))
        colors[seed] = k
        nbrs = adj[seed] & U
        U -= nbrs
        U.discard(seed)
        W |= nbrs

        while U:
            chosen = min(
                U,
                key=lambda x, U=U, W=W: (-len(adj[x] & W), len(adj[x] & U), x),
            )
            colors[chosen] = k
            nbrs = adj[chosen] & U
            U -= nbrs
            U.discard(chosen)
            W |= nbrs

        uncolored = {v for v in uncolored if colors[v] == -1}
        k += 1

    return colors


def auto_color_blocks(
    free_blocks: Sequence[Block],
    interaction_groups: Sequence[InteractionGroup],
) -> list[SuperBlock]:
    """Derive a minimal parallel sampling order from the interaction graph.

    Analyses which free blocks interact with which others and returns a list of
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

    - ``free_blocks``: The free blocks whose sampling order you want to optimise.
      The order of this list is preserved within each colour group, so the
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

        # Each of even/odd is internally an independent set, but every chain
        # edge links an even node to an odd one, so the two blocks conflict and
        # cannot share a sampling group — they must be updated sequentially. By
        # hand you would have to reason this out and write:
        #   free_super_blocks = [even, odd]   # two separate groups
        #
        # auto_color_blocks derives the same order from the interaction groups:
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
    # Step 3 — colour the block-conflict graph with Recursive Largest First.
    #
    # RLF minimises the colour count (= number of sequential sampling groups,
    # which sets the round-loop compile cost) more aggressively than first-fit,
    # especially on dense conflict graphs, and matches it on sparse/bipartite
    # ones. Optimal for bipartite graphs; a good heuristic otherwise.
    # -------------------------------------------------------------------------
    color = rlf_coloring(n, conflicts)

    # -------------------------------------------------------------------------
    # Step 4 — group blocks by colour, preserving original order within groups.
    # -------------------------------------------------------------------------
    color_groups: dict[int, list[Block]] = defaultdict(list)
    for block_idx in range(n):  # ascending index preserves original order
        color_groups[color[block_idx]].append(free_blocks[block_idx])

    # Return colour groups in ascending colour order so the sampling sequence
    # is deterministic and independent of dict iteration order.
    result: list[SuperBlock] = []
    for c in sorted(color_groups):
        group = color_groups[c]
        if len(group) == 1:
            result.append(group[0])
        else:
            result.append(tuple(group))

    return result
