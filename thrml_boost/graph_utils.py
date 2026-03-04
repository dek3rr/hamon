# thrml_boost/graph_utils.py
"""
Graph-coloring utilities for automatic construction of sampling order.

The key entry point is :func:`auto_color_blocks`, which inspects a set of
:class:`~thrml_boost.InteractionGroup` objects and returns a ``free_super_blocks``
list ready to pass directly into :class:`~thrml_boost.BlockGibbsSpec`.

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
from typing import Sequence

from thrml_boost.block_management import Block
from thrml_boost.interaction import InteractionGroup

# Re-export the SuperBlock type alias so callers only need one import.
from thrml_boost.block_sampling import SuperBlock  # noqa: F401


def auto_color_blocks(
    free_blocks: Sequence[Block],
    interaction_groups: Sequence[InteractionGroup],
) -> list[SuperBlock]:
    """Derive a minimal parallel sampling order from the interaction graph.

    Analyses which free blocks interact with which others and returns a list of
    ``SuperBlock`` values (each either a plain :class:`~thrml_boost.Block` or a
    tuple of :class:`~thrml_boost.Block` objects) that can be passed directly to
    :class:`~thrml_boost.BlockGibbsSpec` as ``free_super_blocks``.

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

    **Example** — Ising checkerboard::

        nodes  = [SpinNode() for _ in range(5)]
        edges  = [(nodes[i], nodes[i + 1]) for i in range(4)]
        model  = IsingEBM(nodes, edges, ...)

        even   = Block(nodes[::2])   # {0, 2, 4}
        odd    = Block(nodes[1::2])  # {1, 3}

        # Without auto_color_blocks the user must know that even/odd are
        # independent and manually write:
        #   free_super_blocks = [(even, odd)]
        #
        # With auto_color_blocks:
        igs    = [f.to_interaction_groups() for f in model.factors]
        igs    = [g for sublist in igs for g in sublist]
        super_blocks = auto_color_blocks([even, odd], igs)
        # => [(even, odd)]  — detected automatically
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
    # Step 3 — greedy graph colouring.
    #
    # Process blocks in their original order.  Assign each block the smallest
    # colour not used by any of its conflicting neighbours that have already
    # been coloured.  This produces the chromatic number for bipartite graphs
    # and a good (not necessarily optimal) colouring for general graphs.
    # -------------------------------------------------------------------------
    color: dict[int, int] = {}
    for i in range(n):
        neighbour_colors = {color[j] for j in range(i) if (i, j) in conflicts}
        # Smallest non-negative integer not in neighbour_colors.
        c = 0
        while c in neighbour_colors:
            c += 1
        color[i] = c

    # -------------------------------------------------------------------------
    # Step 4 — group blocks by colour, preserving original order within groups.
    # -------------------------------------------------------------------------
    color_groups: dict[int, list[Block]] = defaultdict(list)
    for block_idx, c in sorted(color.items()):  # sorted preserves original order
        color_groups[c].append(free_blocks[block_idx])

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
