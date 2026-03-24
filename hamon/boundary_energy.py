"""Boundary-only energy delta computation for Ising models.

After updating block b, ΔE depends only on edges incident to b.
For m×m rectangular blocks with 4-coloring, the boundary/total edge
ratio scales as O(1/m). Provides edge classification, incremental ΔE,
and rectangular block construction.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np

from hamon.pgm import AbstractNode


# ---------------------------------------------------------------------------
# Edge classification
# ---------------------------------------------------------------------------


class EdgePartition:
    """Pre-computed classification of edges relative to a block partition.

    For each block b, edges are classified as:
    - incident: at least one endpoint in b (needed for ΔE after updating b)
    - boundary: exactly one endpoint in b, one outside
    - interior: both endpoints in b (only possible for non-independent-set blocks)
    - external: neither endpoint in b (ΔE = 0, skip entirely)

    All arrays are numpy (used at Python level for BlockSpec construction,
    not inside JIT).
    """

    def __init__(
        self,
        edges: list[tuple[AbstractNode, AbstractNode]],
        blocks: list[list[AbstractNode]],
    ):
        n_edges = len(edges)
        n_blocks = len(blocks)

        # Map nodes to block index (-1 if not in any free block)
        node_to_block = {}
        for b_idx, block in enumerate(blocks):
            for node in block:
                node_to_block[id(node)] = b_idx

        # Classify each edge
        self.edge_block_a = np.array(
            [node_to_block.get(id(e[0]), -1) for e in edges], dtype=np.int32
        )
        self.edge_block_b = np.array(
            [node_to_block.get(id(e[1]), -1) for e in edges], dtype=np.int32
        )

        # Per-block incident edge masks
        self.incident_masks = []  # (n_blocks, n_edges) bool
        self.boundary_masks = []
        self.interior_masks = []

        for b_idx in range(n_blocks):
            a_in = self.edge_block_a == b_idx
            b_in = self.edge_block_b == b_idx
            either = a_in | b_in
            both = a_in & b_in
            boundary = either & ~both

            self.incident_masks.append(either)
            self.boundary_masks.append(boundary)
            self.interior_masks.append(both)

        # Summary statistics
        self.n_incident = [int(m.sum()) for m in self.incident_masks]
        self.n_boundary = [int(m.sum()) for m in self.boundary_masks]
        self.n_interior = [int(m.sum()) for m in self.interior_masks]
        self.n_external = [n_edges - inc for inc in self.n_incident]

    @property
    def boundary_ratio(self) -> list[float]:
        """Fraction of incident edges that are boundary (vs interior).

        For independent-set blocks (checkerboard), this is always 1.0.
        For rectangular blocks, this decreases as block size grows.
        """
        return [b / max(i, 1) for b, i in zip(self.n_boundary, self.n_incident)]

    def savings_factor(self, block_idx: int) -> float:
        """FLOPS ratio: incident_edges / total_edges for block b.

        < 1.0 means boundary-only delta is cheaper than full recomputation.
        """
        total = len(self.edge_block_a)
        return self.n_incident[block_idx] / max(total, 1)


# ---------------------------------------------------------------------------
# Incremental energy delta for Ising
# ---------------------------------------------------------------------------


def ising_energy_delta(
    old_state_flat: jax.Array,
    new_state_flat: jax.Array,
    biases: jax.Array,
    weights: jax.Array,
    edge_src_idx: jax.Array,
    edge_dst_idx: jax.Array,
    incident_mask: jax.Array,
    changed_mask: jax.Array,
) -> jax.Array:
    """Compute energy change from a block update using only incident edges.

    E = -β(Σ b_i s_i + Σ J_ij s_i s_j)

    ΔE = E_new - E_old = -β[Σ_{i∈changed} b_i(s'_i - s_i)
                           + Σ_{(i,j)∈incident} J_ij(s'_is'_j - s_is_j)]

    Args:
        old_state_flat: (n_nodes,) float, old spin values as ±1 or {0,1}
        new_state_flat: (n_nodes,) float, new spin values
        biases: (n_nodes,) bias terms
        weights: (n_edges,) coupling terms
        edge_src_idx: (n_edges,) int, source node indices
        edge_dst_idx: (n_edges,) int, destination node indices
        incident_mask: (n_edges,) bool, True for edges incident to updated block
        changed_mask: (n_nodes,) bool, True for nodes that were updated

    Returns:
        Scalar energy delta (WITHOUT the -β factor; caller multiplies).
    """
    # Bias delta: only changed nodes contribute
    ds = new_state_flat - old_state_flat
    bias_delta = jnp.sum(biases * ds * changed_mask.astype(ds.dtype))

    # Coupling delta: only incident edges contribute
    old_prod = old_state_flat[edge_src_idx] * old_state_flat[edge_dst_idx]
    new_prod = new_state_flat[edge_src_idx] * new_state_flat[edge_dst_idx]
    coupling_delta = jnp.sum(
        weights * (new_prod - old_prod) * incident_mask.astype(weights.dtype)
    )

    return -(bias_delta + coupling_delta)


# ---------------------------------------------------------------------------
# Cached energy tracker for NRPT
# ---------------------------------------------------------------------------


def init_energy_cache(
    n_chains: int,
    base_energies: jax.Array | None = None,
) -> jax.Array:
    """Initialize energy cache. If base_energies provided, use those."""
    if base_energies is not None:
        return base_energies
    return jnp.zeros(n_chains, dtype=jnp.float32)


def update_energy_cache(
    cached_energies: jax.Array,
    energy_deltas: jax.Array,
    betas: jax.Array,
) -> jax.Array:
    """Update cached base energies with deltas.

    cached_base_E[c] += delta_E[c] / betas[c]

    The delta is the raw energy change (includes β factor from the EBM),
    so we divide by β to get the base energy change.
    """
    return cached_energies + energy_deltas / betas


# ---------------------------------------------------------------------------
# NRPT integration: vmapped delta function factory
# ---------------------------------------------------------------------------


def make_ising_delta_fn(
    nodes: list[AbstractNode],
    edges: list[tuple[AbstractNode, AbstractNode]],
    free_blocks,
    biases: jax.Array,
    weights: jax.Array,
):
    """Build a vmapped base-energy delta function for use with nrpt().

    Returns delta_fn(old_stacked_states, new_stacked_states) -> (n_chains,),
    where delta_fn[c] = E_base(new_c) - E_base(old_c).

    Pass the result as the energy_delta_fn keyword argument to nrpt():

        delta_fn = make_ising_delta_fn(ebm.nodes, ebm.edges,
                                       free_blocks, ebm.biases, ebm.weights)
        nrpt(..., energy_delta_fn=delta_fn)

    FLOPS note:
        For checkerboard (2-block) partitions every edge is incident to at
        least one block, so incident_mask = all-ones — same arithmetic as a
        full recompute but without the equinox dispatch overhead.  The strict
        FLOPS savings appear with rectangular blocks (4-coloring) where the
        incident fraction is O(1/m) for m×m blocks.

    Args:
        nodes:       all nodes in global order (IsingEBM.nodes)
        edges:       all edges               (IsingEBM.edges)
        free_blocks: the free blocks used in the sampling program; any
                     iterable of node-iterables (Block objects work directly)
        biases:      (n_nodes,) bias array   (IsingEBM.biases)
        weights:     (n_edges,) weight array (IsingEBM.weights)
    """
    node_map: dict[int, int] = {id(n): i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    # Per-block arrays of global node indices — static, computed once.
    block_indices = [
        jnp.array([node_map[id(n)] for n in block], dtype=jnp.int32)
        for block in free_blocks
    ]

    edge_src = jnp.array([node_map[id(e[0])] for e in edges], dtype=jnp.int32)
    edge_dst = jnp.array([node_map[id(e[1])] for e in edges], dtype=jnp.int32)

    # Full-graph masks (all nodes updated). For single-colour-class updates, pass custom masks.
    incident_mask = jnp.ones(len(edges), dtype=jnp.float32)
    changed_mask = jnp.ones(n_nodes, dtype=jnp.float32)

    def _assemble_flat(stacked_states_list: list) -> jax.Array:
        """Scatter per-block bool states into a (n_chains, n_nodes) float±1 array."""
        n_chains = stacked_states_list[0].shape[0]
        flat = jnp.zeros((n_chains, n_nodes), dtype=jnp.float32)
        for b, indices in enumerate(block_indices):
            # bool {0,1} → float {-1, +1} to match SpinEBMFactor convention
            spins = 2.0 * stacked_states_list[b].astype(jnp.float32) - 1.0
            flat = flat.at[:, indices].set(spins)
        return flat

    def delta_fn(old_stacked: list, new_stacked: list) -> jax.Array:
        """Return (n_chains,) array of E_base deltas."""
        old_flat = _assemble_flat(old_stacked)
        new_flat = _assemble_flat(new_stacked)

        def _delta_one(old_f: jax.Array, new_f: jax.Array) -> jax.Array:
            return ising_energy_delta(
                old_f,
                new_f,
                biases,
                weights,
                edge_src,
                edge_dst,
                incident_mask,
                changed_mask,
            )

        return jax.vmap(_delta_one)(old_flat, new_flat)

    return delta_fn


# ---------------------------------------------------------------------------
# Rectangular block construction (4-coloring for 2D grids)
# ---------------------------------------------------------------------------


def make_rectangular_blocks(
    L: int,
    block_size: int,
    nodes_2d: list[list[AbstractNode]],
) -> tuple[list[list[AbstractNode]], list[list[int]]]:
    """Partition an L×L grid into rectangular blocks with 4-coloring.

    Returns (blocks, color_classes) where:
    - blocks: list of node lists, one per rectangular block
    - color_classes: 4 lists of block indices that can update simultaneously

    For m×m blocks in an L×L grid:
    - n_blocks = ceil(L/m)² per axis
    - 4 color classes (checkerboard of blocks)
    - boundary/total edge ratio ≈ 4m/(2m²) = 2/m

    Args:
        L: grid side length
        block_size: m, the side length of each rectangular block
        nodes_2d: L×L array of nodes, `nodes_2d[i][j]` is node at row `i`, col `j`
    """
    m = block_size
    n_blocks_per_axis = (L + m - 1) // m

    blocks = []
    block_grid = {}  # (bi, bj) → block_index

    for bi in range(n_blocks_per_axis):
        for bj in range(n_blocks_per_axis):
            block_nodes = []
            for i in range(bi * m, min((bi + 1) * m, L)):
                for j in range(bj * m, min((bj + 1) * m, L)):
                    block_nodes.append(nodes_2d[i][j])
            block_grid[(bi, bj)] = len(blocks)
            blocks.append(block_nodes)

    # 4-coloring: (bi % 2, bj % 2) gives 4 classes
    color_classes = [[] for _ in range(4)]
    for bi in range(n_blocks_per_axis):
        for bj in range(n_blocks_per_axis):
            color = (bi % 2) * 2 + (bj % 2)
            color_classes[color].append(block_grid[(bi, bj)])

    return blocks, color_classes


def estimate_boundary_savings(L: int, block_size: int) -> dict:
    """Estimate FLOPS savings for boundary-only energy delta on an L×L grid with m×m blocks."""
    m = block_size
    n_blocks_axis = (L + m - 1) // m
    total_edges = 2 * L * (L - 1)

    # Per block: boundary edges ~ 4*(m-1) for interior blocks
    # Interior edges ~ 2*(m-1)*m - (m-1) per block
    boundary_per_block = 4 * (m - 1) if m < L else 2 * L * (L - 1)
    interior_per_block = max(0, 2 * m * (m - 1) - 2 * (m - 1))

    incident_per_block = boundary_per_block + interior_per_block

    # For full energy after one color class update:
    blocks_per_color = (n_blocks_axis // 2 + n_blocks_axis % 2) ** 2
    incident_per_color = blocks_per_color * incident_per_block

    # Per-block ratio: boundary / incident — the metric that improves with m
    boundary_ratio = boundary_per_block / max(incident_per_block, 1)

    return {
        "total_edges": total_edges,
        "blocks_per_color_class": blocks_per_color,
        "boundary_edges_per_block": boundary_per_block,
        "interior_edges_per_block": interior_per_block,
        "incident_edges_per_color": min(incident_per_color, total_edges),
        "savings_ratio": min(incident_per_color, total_edges) / max(total_edges, 1),
        "boundary_ratio_per_block": boundary_ratio,
        "checkerboard_savings": 1.0,  # no savings for checkerboard
    }


# ---------------------------------------------------------------------------
# Edge index pre-computation (for JIT-compatible delta computation)
# ---------------------------------------------------------------------------


def precompute_edge_indices(
    nodes: list[AbstractNode],
    edges: list[tuple[AbstractNode, AbstractNode]],
    blocks: list[list[AbstractNode]],
) -> dict:
    """Pre-compute all index arrays needed for boundary energy deltas.

    Returns numpy arrays (to be converted to jnp at call site).
    These are static/compile-time constants, not part of the JIT trace.
    """
    node_map = {id(n): i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    edge_src = np.array([node_map[id(e[0])] for e in edges], dtype=np.int32)
    edge_dst = np.array([node_map[id(e[1])] for e in edges], dtype=np.int32)

    # Per-block masks
    node_to_block = np.full(n_nodes, -1, dtype=np.int32)
    for b_idx, block in enumerate(blocks):
        for node in block:
            node_to_block[node_map[id(node)]] = b_idx

    incident_masks = []
    changed_masks = []
    for b_idx in range(len(blocks)):
        a_in = node_to_block[edge_src] == b_idx
        b_in = node_to_block[edge_dst] == b_idx
        incident_masks.append(a_in | b_in)
        changed_masks.append(node_to_block == b_idx)

    return {
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "incident_masks": incident_masks,
        "changed_masks": changed_masks,
        "node_to_block": node_to_block,
    }
