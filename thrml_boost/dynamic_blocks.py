"""Dynamic block construction and re-partitioning for parallel tempering.

Implements three strategies:

1. Per-temperature blocks (Efthymiou & Zampetakis 2024):
   Different block partitions at different β values. Hot chains use small
   cheap blocks; cold chains use large blocks to overcome critical slowing.

2. Influence-aware partitioning (Efthymiou):
   Aggregate influence A(w) = Σ_{z~w} |1-exp(βJ)|/(1+exp(βJ)) identifies
   "heavy" vertices that should be interior to blocks, not on boundaries.

3. Dynamic re-blocking (Venugopal & Gogate 2013):
   Measure pairwise correlations during sampling, re-partition blocks to
   group highly correlated variables together.

These operate at the Python level (block construction is a compile-time
concern, not a JIT-traced operation). The resulting block partitions
are passed to IsingSamplingProgram which builds the padded interaction
structures for GPU execution.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from thrml_boost.pgm import AbstractNode


# ---------------------------------------------------------------------------
# Aggregate influence computation (Efthymiou)
# ---------------------------------------------------------------------------


def compute_edge_influence(
    weights: np.ndarray | jax.Array,
    beta: float,
) -> np.ndarray:
    """Compute per-edge influence Γ_e = |1-exp(βJ_e)| / (1+exp(βJ_e)).

    This measures how strongly each edge couples its endpoints at
    temperature β. High influence = strong correlation = harder to
    mix when updating one endpoint while the other is fixed.
    """
    bJ = np.asarray(beta * weights, dtype=np.float64)
    exp_bJ = np.exp(bJ)
    return np.abs(1.0 - exp_bJ) / (1.0 + exp_bJ)


def compute_aggregate_influence(
    edges: list[tuple[AbstractNode, AbstractNode]],
    weights: np.ndarray | jax.Array,
    beta: float,
    nodes: list[AbstractNode],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-node aggregate influence A(w) = Σ_{z~w} Γ_{w,z}.

    Returns (aggregate_influence, edge_influence) both as numpy arrays.
    Heavy nodes (A(w) > threshold) should be buried inside blocks.
    """
    node_map = {id(n): i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    edge_inf = compute_edge_influence(weights, beta)
    agg = np.zeros(n_nodes, dtype=np.float64)

    for (u, v), gamma in zip(edges, edge_inf):
        agg[node_map[id(u)]] += gamma
        agg[node_map[id(v)]] += gamma

    return agg, edge_inf


def classify_nodes(
    aggregate_influence: np.ndarray,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split nodes into heavy (high influence) and light sets.

    Default threshold: median aggregate influence.
    Returns (heavy_indices, light_indices).
    """
    if threshold is None:
        threshold = float(np.median(aggregate_influence))
    heavy = np.where(aggregate_influence > threshold)[0]
    light = np.where(aggregate_influence <= threshold)[0]
    return heavy, light


# ---------------------------------------------------------------------------
# Per-temperature block sizing (Swanson + Efthymiou)
# ---------------------------------------------------------------------------


def recommend_block_size(
    beta: float,
    beta_c: float = 0.4407,
    min_size: int = 1,
    max_size: int = 16,
) -> int:
    """Recommend block size based on inverse temperature.

    Near-criticality correlation length diverges, requiring larger blocks
    to overcome critical slowing down. The block size should scale with
    the correlation length ξ ~ |β - β_c|^{-ν} (ν=1 for 2D Ising).

    Below criticality: small blocks suffice (fast mixing).
    At criticality: maximize block size.
    Above criticality: large blocks needed (but energy landscape is simpler).
    """
    if abs(beta - beta_c) < 0.05:
        return max_size

    # Correlation length proxy
    distance = abs(beta - beta_c)
    xi = min(1.0 / max(distance, 0.01), max_size)

    # Block should be ~2ξ to capture correlation structure
    recommended = int(2 * xi)
    return max(min_size, min(recommended, max_size))


def per_temperature_block_config(
    betas: Sequence[float],
    beta_c: float = 0.4407,
    min_size: int = 1,
    max_size: int = 16,
) -> list[int]:
    """Generate block size recommendations for each temperature in a PT chain.

    Returns list of recommended block sizes, one per chain.
    """
    return [recommend_block_size(float(b), beta_c, min_size, max_size) for b in betas]


# ---------------------------------------------------------------------------
# Influence-aware block construction (Efthymiou)
# ---------------------------------------------------------------------------


def influence_aware_partition(
    nodes: list[AbstractNode],
    edges: list[tuple[AbstractNode, AbstractNode]],
    weights: np.ndarray,
    beta: float,
    max_block_size: int = 16,
    buffer_depth: int = 2,
) -> list[list[AbstractNode]]:
    """Build blocks where heavy nodes are interior, light nodes form boundary.

    Algorithm:
    1. Compute aggregate influence for each node
    2. Sort nodes by influence (descending)
    3. Greedily build blocks: start from heaviest unassigned node,
       BFS outward until block reaches max_size or runs out of neighbors
    4. Heavy nodes end up interior, light neighbors form buffer

    For standard grids, this reduces to larger blocks around high-coupling
    regions. For heterogeneous couplings (learned EBMs, spin glasses),
    it adapts the partition to the coupling structure.
    """
    node_map = {id(n): i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    # Build adjacency list
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        ui, vi = node_map[id(u)], node_map[id(v)]
        adj[ui].append(vi)
        adj[vi].append(ui)

    agg_inf, _ = compute_aggregate_influence(edges, weights, beta, nodes)

    # Sort by influence descending
    order = np.argsort(-agg_inf)

    assigned = np.zeros(n_nodes, dtype=bool)
    blocks = []

    for seed in order:
        if assigned[seed]:
            continue

        # BFS from seed
        block_indices = [seed]
        assigned[seed] = True
        frontier = [seed]

        while len(block_indices) < max_block_size and frontier:
            next_frontier = []
            for node_idx in frontier:
                for nbr in adj[node_idx]:
                    if not assigned[nbr] and len(block_indices) < max_block_size:
                        block_indices.append(nbr)
                        assigned[nbr] = True
                        next_frontier.append(nbr)
            frontier = next_frontier

        blocks.append([nodes[i] for i in block_indices])

    # Assign any remaining unassigned nodes
    for i in range(n_nodes):
        if not assigned[i]:
            blocks.append([nodes[i]])

    return blocks


# ---------------------------------------------------------------------------
# Correlation-based re-blocking (Venugopal & Gogate 2013)
# ---------------------------------------------------------------------------


def estimate_pairwise_correlations(
    samples: jax.Array | np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
) -> np.ndarray:
    """Estimate correlation along edges from samples.

    Uses Hellinger-like distance between joint and product marginals,
    approximated as |Cor(s_i, s_j)|.

    Args:
        samples: (n_samples, n_nodes) float array of spin values
        edge_src: (n_edges,) source node indices
        edge_dst: (n_edges,) destination node indices

    Returns:
        (n_edges,) absolute correlation values
    """
    arr = np.asarray(samples, dtype=np.float64)

    src_vals = arr[:, edge_src]  # (n_samples, n_edges)
    dst_vals = arr[:, edge_dst]

    # Pearson correlation per edge
    src_mean = src_vals.mean(axis=0)
    dst_mean = dst_vals.mean(axis=0)
    src_centered = src_vals - src_mean
    dst_centered = dst_vals - dst_mean

    numer = (src_centered * dst_centered).mean(axis=0)
    src_std = np.sqrt((src_centered**2).mean(axis=0) + 1e-10)
    dst_std = np.sqrt((dst_centered**2).mean(axis=0) + 1e-10)

    correlations = np.abs(numer / (src_std * dst_std))
    return np.asarray(np.clip(correlations, 0.0, 1.0))


def greedy_merge_blocks(
    current_blocks: list[list[int]],
    edge_correlations: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    max_block_size: int = 16,
    correlation_threshold: float = 0.3,
) -> list[list[int]]:
    """Greedily merge blocks connected by high-correlation edges.

    Algorithm 2 from Venugopal & Gogate (2013), simplified:
    1. Compute inter-block correlation as max edge correlation between blocks
    2. Merge the pair with highest inter-block correlation
    3. Repeat until no pair exceeds threshold or would exceed max_block_size

    Args:
        current_blocks: list of node index lists
        edge_correlations: per-edge correlation values
        edge_src, edge_dst: edge endpoint indices
        max_block_size: maximum allowed block size after merge
        correlation_threshold: minimum correlation to trigger merge
    """
    blocks = [list(b) for b in current_blocks]

    # Map nodes to block index
    n_nodes = max(max(b) for b in blocks) + 1
    node_to_block = np.full(n_nodes, -1, dtype=np.int32)
    for bi, block in enumerate(blocks):
        for ni in block:
            node_to_block[ni] = bi

    changed = True
    while changed:
        changed = False

        # Compute inter-block max correlation
        best_pair = None
        best_corr = correlation_threshold

        for e_idx in range(len(edge_src)):
            bi = node_to_block[edge_src[e_idx]]
            bj = node_to_block[edge_dst[e_idx]]
            if bi == bj or bi == -1 or bj == -1:
                continue
            if edge_correlations[e_idx] > best_corr:
                merged_size = len(blocks[bi]) + len(blocks[bj])
                if merged_size <= max_block_size:
                    best_corr = edge_correlations[e_idx]
                    best_pair = (min(bi, bj), max(bi, bj))

        if best_pair is not None:
            bi, bj = best_pair
            blocks[bi] = blocks[bi] + blocks[bj]
            for ni in blocks[bj]:
                node_to_block[ni] = bi
            # Remove bj by swapping with last
            blocks.pop(bj)
            # Reindex
            for bi2, block in enumerate(blocks):
                for ni in block:
                    node_to_block[ni] = bi2
            changed = True

    return blocks


def dynamic_reblock(
    nodes: list[AbstractNode],
    edges: list[tuple[AbstractNode, AbstractNode]],
    current_blocks: list[list[AbstractNode]],
    samples: jax.Array,
    max_block_size: int = 16,
    correlation_threshold: float = 0.3,
) -> list[list[AbstractNode]]:
    """Re-partition blocks based on empirical correlations from recent samples.

    This is the main entry point for dynamic re-blocking. Call periodically
    during NRPT (e.g. every 50-100 rounds) to adapt blocks to the actual
    correlation structure at each temperature.

    Args:
        nodes: all nodes in the model
        edges: all edges
        current_blocks: current block partition (list of node lists)
        samples: (n_samples, n_nodes) recent samples from this chain
        max_block_size: maximum nodes per block
        correlation_threshold: minimum correlation to trigger merge

    Returns:
        New block partition (list of node lists).
    """
    node_map = {id(n): i for i, n in enumerate(nodes)}

    edge_src = np.array([node_map[id(e[0])] for e in edges], dtype=np.int32)
    edge_dst = np.array([node_map[id(e[1])] for e in edges], dtype=np.int32)

    # Convert current blocks to index lists
    current_idx_blocks = [[node_map[id(n)] for n in block] for block in current_blocks]

    # Estimate correlations
    correlations = estimate_pairwise_correlations(samples, edge_src, edge_dst)

    # Merge highly correlated blocks
    new_idx_blocks = greedy_merge_blocks(
        current_idx_blocks,
        correlations,
        edge_src,
        edge_dst,
        max_block_size=max_block_size,
        correlation_threshold=correlation_threshold,
    )

    # Convert back to node lists
    return [[nodes[i] for i in block] for block in new_idx_blocks]


# ---------------------------------------------------------------------------
# Influence-weighted convergence diagnostic (Efthymiou Eq 25)
# ---------------------------------------------------------------------------


def weighted_hamming_distance(
    state_a: jax.Array,
    state_b: jax.Array,
    aggregate_influence: jax.Array,
    block_interior_mask: jax.Array,
    n_nodes: int,
    weight_external: float = 1.0,
) -> jax.Array:
    """Compute influence-weighted Hamming distance between two configurations.

    d(σ, τ) = Σ_{internal} 1{disagree} + C·Σ_{external} A_out(z)·1{disagree}

    where A_out(z) = aggregate influence from edges leaving the block.
    This gives sharper mixing detection than unweighted Hamming.

    Args:
        state_a, state_b: (n_nodes,) spin configurations
        aggregate_influence: (n_nodes,) per-node aggregate influence
        block_interior_mask: (n_nodes,) bool, True for interior nodes
        n_nodes: total node count (for normalization)
        weight_external: scaling factor C for external disagreements
    """
    disagree = (state_a != state_b).astype(jnp.float32)
    interior_f = block_interior_mask.astype(jnp.float32)

    internal_dist = jnp.sum(disagree * interior_f)
    external_dist = jnp.sum(
        disagree * (1.0 - interior_f) * aggregate_influence * weight_external
    )

    return (internal_dist + external_dist) / n_nodes


# ---------------------------------------------------------------------------
# Coloring validation
# ---------------------------------------------------------------------------


def validate_coloring(
    blocks: list[list[AbstractNode]],
    color_classes: list[list[int]],
    edges: list[tuple[AbstractNode, AbstractNode]],
) -> bool:
    """Verify that blocks in the same color class don't share boundary edges.

    Two blocks can update simultaneously iff no edge connects a node in one
    block to a node in the other.
    """
    node_to_block = {}
    for b_idx, block in enumerate(blocks):
        for node in block:
            node_to_block[id(node)] = b_idx

    for color_class in color_classes:
        block_set = set(color_class)
        for u, v in edges:
            bu = node_to_block.get(id(u), -1)
            bv = node_to_block.get(id(v), -1)
            if bu != bv and bu in block_set and bv in block_set:
                return False
    return True
