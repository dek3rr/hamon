# Modified from the original thrml library (https://github.com/Extropic-AI/thrml)

import copy
from typing import (
    Generic,
    TypeAlias,
    TypeVar,
)
from collections.abc import Iterator, Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int, PyTree, Shaped

from .pgm import AbstractNode

_Node = TypeVar("_Node", bound=AbstractNode)
_PyTreeStruct: TypeAlias = tuple[
    PyTree,
    tuple[jax.ShapeDtypeStruct, ...],
]
_GlobalState: TypeAlias = PyTree[Shaped[Array, "nodes_global ?*state"], "_GlobalState"]
_State = PyTree[Shaped[Array, "nodes ?*state"], "State"]
_Node_SD = Mapping[type[AbstractNode], PyTree[jax.ShapeDtypeStruct]]


class Block(Generic[_Node]):
    """
    A Block is the basic unit through which Gibbs sampling can operate.

    Each block represents a collection of nodes that can efficiently be sampled
    simultaneously in a JAX-friendly SIMD manner. In hamon, this means that the nodes must all be of the same type.

    **Attributes:**

    - `nodes`: the tuple of nodes that this block contains
    """

    nodes: tuple[_Node, ...]
    _id_cache: tuple[int, ...] | None

    def __init__(self, nodes: Sequence[_Node]) -> None:
        nodes_tuple = tuple(nodes)
        if nodes_tuple:
            first_type = type(nodes_tuple[0])
            # set(map(...)) keeps the type scan in C; blocks can hold every
            # node of a large graph, so this runs O(|V|) per construction.
            if set(map(type, nodes_tuple)) != {first_type}:
                raise ValueError("All nodes in a block must be of the same type")
        self.nodes = nodes_tuple
        self._id_cache = None

    def _ids(self) -> tuple[int, ...]:
        """Identity key of the node sequence, computed once per Block.

        Structure-cache keys are built from node identities; caching the tuple
        here turns every repeated key computation over a reused Block from
        O(|nodes|) ``id()`` calls into an attribute read. ``nodes`` is
        immutable after construction, so the cache can never go stale."""
        ids = self._id_cache
        if ids is None:
            ids = self._id_cache = tuple(map(id, self.nodes))
        return ids

    @property
    def node_type(self) -> type[_Node]:
        if not self.nodes:
            raise ValueError(
                "Block is empty and doesn't have a node type. Most methods in hamon do not support empty blocks."
            )
        return type(self.nodes[0])

    def __getitem__(self, index: int) -> _Node:
        return self.nodes[index]

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[_Node]:
        return iter(self.nodes)

    def __contains__(self, item) -> bool:
        return item in self.nodes

    def __add__(self, other):
        if isinstance(other, Block):
            if self.nodes and other.nodes:
                if type(self.nodes[0]) is not type(other.nodes[0]):
                    raise ValueError("Cannot add Blocks of different node types")
            return Block(self.nodes + other.nodes)
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nodes={self.nodes!r})"


def _hash_pytree(x: PyTree[jax.ShapeDtypeStruct]) -> _PyTreeStruct:
    return (jax.tree.structure(x), tuple(jax.tree.leaves(x)))


class BlockSpec:
    """
    This contains the necessary mappings for logging indices of states and node types.

    This helps convert between block states and global states. A block state is a list
    of pytrees, where each pytree leaf has shape[0] = number of nodes in the block.
    The length of the block state is the number of blocks. The global state is a
    flattened version of this. Each pytree type is combined (regardless of which block
    they are in), to make a list of pytrees where each leaf shape[0] is the total
    number of nodes of that pytree shape. As an example, imagine an Ising model,
    every node is the same pytree (just a scalar array), as such the block state is
    a list of arrays where each array is the state of the block and the global state
    would be a length-1 list that contains an array of shape (total_nodes,).

    **Attributes:**

    - `blocks`: the list of blocks this spec contains
    - `all_block_sds`: a SD is a single `_PyTreeStruct`. Each node/block has only
        one SD associated with it, but each node can have neighbors of many types.
        This is the SD of each block (in the same order as blocks, this internal
        ordering is quite important for bookkeeping). This list is just the list
        of SDs for each block (and thus has length = len(blocks)).
    - `global_sd_order`: the list of SDs, providing a SoT for the global ordering
    - `sd_index_map`: a dictionary mapping the SD to an integer in the
        `global_sd_order`. This is like calling `.index` on it.
    - `node_global_location_map`: a dictionary mapping a given node to a tuple.
        That tuple contains the global index (i.e. which element in the global
        list it is in) and the relative position in that pytree. That is to say,
        you can get the state of the node via
        `map(x[tuple[1]], global_repr[tuple[0]])`
    - `block_to_global_slice_spec`: a list over unique SDs (so length
        global_sd_order), where each list inside this is the list over blocks
        which contain that pytree. E.g. [[0, 1], [2]] indicates that blocks[0]
        and blocks[1] are both of pytree SD 0.
    - `node_shape_dtypes`: a dictionary mapping node types to hashable `_PyTreeStruct`
    - `node_shape_struct`: a dictionary mapping node types to pytrees of JAX-shaped
        dtype structs (just for user access, since the keys aren't hashable that
        creates issues for JAX in other areas.)

    """

    blocks: list[Block]
    all_block_sds: list[_PyTreeStruct]
    global_sd_order: list[_PyTreeStruct]
    sd_index_map: dict[_PyTreeStruct, int]
    node_global_location_map: dict[AbstractNode, tuple[int, int]]
    block_to_global_slice_spec: list[list[int]]
    node_shape_dtypes: dict[type[AbstractNode], _PyTreeStruct]
    node_shape_struct: dict[type[AbstractNode], PyTree[jax.ShapeDtypeStruct]]

    def __init__(
        self,
        blocks: list[Block],
        node_shape_dtypes: _Node_SD,
    ) -> None:
        """
        Create a BlockSpec from blocks.

        Based on the information passed in via node_shape_dtypes, determine the minimal global state that can be used
        to represent the blocks.

        **Arguments:**

        - `blocks`: the list of `Block`s that this specification operates on
        - `node_shape_dtypes`: the mapping of node types to their structures. This
                should be a pytree of `jax.ShapeDtypeStruct`s.
        """
        self.node_shape_struct = dict(node_shape_dtypes)
        self.node_shape_dtypes = {
            i: _hash_pytree(j) for i, j in node_shape_dtypes.items()
        }

        self.blocks = blocks

        # Deduplicate while preserving insertion order
        all_sds = list(dict.fromkeys(self.node_shape_dtypes.values()))
        self.global_sd_order = all_sds

        self.sd_index_map = {sd: i for i, sd in enumerate(self.global_sd_order)}

        for block in blocks:
            if len(block) == 0:
                raise ValueError("Encountered an empty block in BlockSpec.")

            if block.node_type not in node_shape_dtypes:
                raise ValueError(
                    f"Block with node type {block.node_type} not found in node_shape_dtypes."
                )

        self.all_block_sds = [
            self.node_shape_dtypes[block.node_type] for block in blocks
        ]

        block_to_global_slice_spec = [[] for _ in self.global_sd_order]

        node_global_location_map = {}
        arr_ind_tracker = [0 for _ in self.global_sd_order]
        for block_idx, (block, sds) in enumerate(zip(blocks, self.all_block_sds)):
            block_len = len(block)

            sd_ind = self.sd_index_map[sds]
            start_ind = arr_ind_tracker[sd_ind]
            arr_ind_tracker[sd_ind] += block_len
            block_to_global_slice_spec[sd_ind].append(block_idx)
            for k, node in enumerate(block.nodes):
                if node in node_global_location_map:
                    raise RuntimeError(
                        "A node should not show up twice in the blocks input to BlockSpec."
                    )
                node_global_location_map[node] = (sd_ind, start_ind + k)
        self.block_to_global_slice_spec = block_to_global_slice_spec
        self.node_global_location_map = node_global_location_map

    def _structure_key(self):
        """Everything about this spec that fixes the compiled sampling program:
        the block partition (the node objects per block, in order), the
        global-state layout (`block_to_global_slice_spec`, which also encodes the
        per-block vs. concatenated layout), the Gibbs sampling order, and the
        node-type SD map.

        Keyed on node identity: a spec rebuilt over the *same* nodes and
        structure compares equal and shares the `eqx.filter_jit` cache. Without
        this, every `program.with_ebm(...)` — called once per `nrpt` /
        `tune_schedule` invocation and per discovery probe — builds a fresh spec
        object that misses the cache and forces a full recompile of the
        otherwise-identical NRPT round loop. (Specs over *different* node objects
        stay distinct, which is what we want: the surrounding program's blocks
        carry those same nodes and are themselves identity-compared in the cache
        key, so structural-only equality would buy nothing here.)
        """
        return (
            type(self),
            tuple(block._ids() for block in self.blocks),
            tuple(tuple(s) for s in self.block_to_global_slice_spec),
            tuple(tuple(g) for g in getattr(self, "sampling_order", ())),
            frozenset(self.node_shape_dtypes.items()),
        )

    def __eq__(self, other):
        return isinstance(other, BlockSpec) and (
            self._structure_key() == other._structure_key()
        )

    def __hash__(self):
        return hash(self._structure_key())


def _stack(*args):
    if eqx.is_array(args[0]):
        if args[0].shape == ():
            return jnp.stack(args)
        # concatenate across node dim
        return jnp.concatenate(args, axis=0)
    else:
        assert all(args[0] == arg for arg in args[1:])
        return args[0]


def block_state_to_global(
    block_state: list[_State], spec: BlockSpec
) -> list[_GlobalState]:
    """
    Convert block-local state to the global stacked representation.

    The block representation is a list where ``block_state[i]`` contains the
    state of ``spec.blocks[i]`` and every node occupies index 0 of its leaf.

    The global representation is a shorter list (one entry per distinct
    PyTree structure) in which all blocks with the same structure are
    concatenated along their node axis.

    **Arguments:**

    - `block_state`: State organised per block, same length as
        ``spec.blocks``.
    - `spec`: The [`hamon.BlockSpec`][] that defines the mapping.

    **Returns:**

    A list whose length equals
    ``len(spec.global_sd_order)``—the stacked global state.
    """
    global_state = []
    for sd_indexes in spec.block_to_global_slice_spec:
        if not sd_indexes:
            global_state.append(None)
            continue

        collected = [block_state[i] for i in sd_indexes]

        if len(collected) == 1:
            global_state.append(collected[0])
        else:
            global_state.append(jax.tree.map(_stack, *collected))

    return global_state


def _block_layout(block: Block, spec: BlockSpec) -> tuple[int, int | None, np.ndarray]:
    """Locate *block* inside the global state.

    Returns ``(sd_index, contiguous_start, positions)``:

    * *sd_index* — which entry of the global state list holds the block's
      nodes;
    * *contiguous_start* — the static start index when the positions form a
      contiguous ascending range (a ``BlockSpec`` layout invariant for its own
      blocks), or ``None`` for arbitrary node sets;
    * *positions* — the per-node global positions as a numpy array.

    A contiguous start lets reads and writes lower to static-offset
    ``lax.dynamic_slice`` / ``dynamic_update_slice`` instead of
    gathers/scatters, which XLA fuses far better. This is the single source
    of truth for that check; ``scatter_block_to_global``,
    ``from_global_state``, and ``BlockSamplingProgram`` all use it.
    """
    # Read the slot from the location map (rather than sd_index_map[node_type])
    # so this is correct under any layout, including the per-block layout where a
    # block's slot is its own index rather than its shared structure group. All
    # of a block's nodes live in one slot, so the first node fixes it. Behaviour
    # is identical to the structure-group lookup under the default layout.
    sd_ind = spec.node_global_location_map[block.nodes[0]][0]
    locs = np.array([spec.node_global_location_map[node][1] for node in block])
    start = None
    if locs.size and np.array_equal(locs, np.arange(locs[0], locs[0] + locs.size)):
        start = int(locs[0])
    return sd_ind, start, locs


def scatter_block_to_global(
    global_state: list[_GlobalState],
    new_block_state: _State,
    block: Block,
    spec: BlockSpec,
) -> list[_GlobalState]:
    """
    Scatter a single block's updated state back into the global state.

    This is an incremental alternative to calling ``block_state_to_global``
    from scratch after every block update. Instead of rebuilding the full
    concatenated global tensor, it writes only the positions that changed.
    When the block occupies a contiguous range of the global state (always
    the case for blocks laid out by ``BlockSpec``), the write lowers to
    ``lax.dynamic_update_slice`` with a static offset, which XLA fuses far
    better than a scatter; non-contiguous node sets fall back to
    ``jnp.ndarray.at[...].set(...)``.

    Because the clamped blocks never change, carrying global state across
    scan iterations and calling this function after each block update avoids
    all redundant work on the clamped portion of the global state.

    **Arguments:**

    - `global_state`: The current global state list (will not be mutated;
        a new list is returned).
    - `new_block_state`: The freshly sampled state for ``block``.
    - `block`: The block that was just sampled.
    - `spec`: The [`hamon.BlockSpec`][] that defines the mapping.

    **Returns:**

    A new global state list with the positions belonging to ``block``
    replaced by ``new_block_state``.
    """
    sd_ind, start, locs = _block_layout(block, spec)

    new_global = list(global_state)  # shallow copy; only one slot changes
    if len(spec.block_to_global_slice_spec[sd_ind]) == 1:
        # The block is the sole occupant of its slot (per-block layout), so the
        # whole slot is replaced — no slice/scatter and no copy of an unchanged
        # neighbour.
        new_global[sd_ind] = new_block_state
    elif start is not None:
        new_global[sd_ind] = jax.tree.map(
            lambda g, s: jax.lax.dynamic_update_slice_in_dim(g, s, start, axis=0),
            global_state[sd_ind],
            new_block_state,
        )
    else:
        positions = jnp.array(locs)
        new_global[sd_ind] = jax.tree.map(
            lambda g, s: g.at[positions].set(s),
            global_state[sd_ind],
            new_block_state,
        )
    return new_global


def get_node_locations(
    nodes: Block, spec: BlockSpec
) -> tuple[int, Int[Array, " nodes"]]:
    """
    Locate a contiguous set of nodes inside the global state.

    **Arguments:**

    - `nodes`: A [`hamon.Block`][] whose nodes you want locations for.
    - `spec`: The [`hamon.BlockSpec`][] generated from the same graph.

    **Returns:**

    Tuple ``(sd_index, positions)`` where

    * *sd_index* is the position inside the global list returned by
      [`hamon.block_state_to_global`][], and
    * *positions* is a 1D array with the indices each node
      occupies inside that particular PyTree.
    """
    sd_ind, _, locs = _block_layout(nodes, spec)
    return sd_ind, jnp.array(locs)


def from_global_state(
    global_state: list[_GlobalState],
    spec_from: BlockSpec,
    blocks_to_extract: list[Block],
) -> list[_State]:
    """
    Extract the states for a subset of blocks from a global state.

    **Arguments:**

    - `global_state`: A state produced by
        [`hamon.block_state_to_global`][].
    - `spec_from`: The [`hamon.BlockSpec`][] associated with *global_state*.
    - `blocks_to_extract`: The blocks whose node states should be returned.

    **Returns:**

    A list with one element per *blocks_to_extract*—each element is a PyTree
    with exactly ``len(block)`` nodes in its leading dimension.

    Blocks whose nodes occupy a contiguous range of the global state (always
    the case for blocks laid out by ``BlockSpec``) are extracted with a
    static-offset slice instead of a gather, mirroring
    [`hamon.scatter_block_to_global`][].
    """
    loc_map = spec_from.node_global_location_map
    out = []
    for block in blocks_to_extract:
        slots = [loc_map[node][0] for node in block.nodes]
        if len(set(slots)) <= 1:
            # All nodes live in one slot: a static slice (contiguous range) or a
            # single gather. This is the only path under the default layout.
            sd_ind, start, locs = _block_layout(block, spec_from)
            if start is not None:
                length = int(locs.size)
                out.append(
                    jax.tree.map(
                        lambda x, _s=start, _n=length: jax.lax.dynamic_slice_in_dim(
                            x, _s, _n, axis=0
                        ),
                        global_state[sd_ind],
                    )
                )
            else:
                positions = jnp.array(locs)
                out.append(
                    jax.tree.map(
                        lambda x, _p=positions: jnp.take(x, _p, axis=0),
                        global_state[sd_ind],
                    )
                )
        else:
            # The block spans several slots (e.g. an all-nodes observation block
            # under the per-block layout). Concatenate the involved slots once and
            # gather node order from that — two ops regardless of slot count, and
            # the concatenation is the same per-type view the default layout keeps
            # resident. Runs only on the (cold) extraction path, never inside the
            # Gibbs sweep.
            positions = [loc_map[node][1] for node in block.nodes]
            involved = sorted(set(slots))
            offset = {}
            acc = 0
            for s in involved:
                offset[s] = acc
                acc += int(jax.tree.leaves(global_state[s])[0].shape[0])
            full = jax.tree.map(
                lambda *xs: jnp.concatenate(xs, axis=0),
                *[global_state[s] for s in involved],
            )
            flat_pos = jnp.array([offset[s] + p for s, p in zip(slots, positions)])
            out.append(
                jax.tree.map(lambda x, _p=flat_pos: jnp.take(x, _p, axis=0), full)
            )
    return out


def to_per_block_layout(spec):
    """Return a shallow copy of *spec* whose global state has one slot per block.

    Under the default layout, every block sharing a PyTree structure is
    concatenated into a single global array, so writing one block's update back
    is a ``dynamic_update_slice`` into that shared array that also copies the
    unchanged portion every step. The per-block layout gives each block its own
    global slot, so a block update replaces its slot outright — no slice/scatter
    and no copy of the rest, which is the dominant cost in dispatch-bound Gibbs
    sweeps.

    Only valid when every interaction reads each of its tail blocks from a single
    block (so the gather's "slot from the first tail node" assumption holds for
    every tail node). Callers must verify this before transforming; see
    ``BlockSamplingProgram.__init__``.
    """
    new_spec = copy.copy(spec)
    new_spec.block_to_global_slice_spec = [[i] for i in range(len(spec.blocks))]
    new_spec.node_global_location_map = {
        node: (block_idx, k)
        for block_idx, block in enumerate(spec.blocks)
        for k, node in enumerate(block.nodes)
    }
    return new_spec


def make_empty_block_state(
    blocks: list[Block],
    node_shape_dtypes: _Node_SD,
    batch_shape: tuple | None = None,
) -> list[_State]:
    """
    Allocate a zero-initialised block state.

    **Arguments:**

    - `blocks`: All blocks in the graph (order is preserved).
    - `node_shape_dtypes`: Maps every node class to its
        `jax.ShapeDtypeStruct` PyTree template.
    - `batch_shape`: Optional batch dimension(s) to prepend to every leaf.

    **Returns:**

    A list of PyTrees—one per *block*—whose leaves are
    ``zeros(batch_shape + (len(block),) + leaf.shape)``.
    """
    state = []
    for block in blocks:
        types = node_shape_dtypes[block.node_type]
        if batch_shape is None:
            this_state = jax.tree.map(
                lambda x: jnp.zeros(shape=(len(block), *x.shape), dtype=x.dtype),
                types,
            )
        else:
            this_state = jax.tree.map(
                lambda x: jnp.zeros(
                    shape=(*batch_shape, len(block), *x.shape), dtype=x.dtype
                ),
                types,
            )
        state.append(this_state)
    return state


def _check_pytree_compat(
    spec_tree,
    data_tree,
) -> tuple[int, ...] | None:
    """
    Verify that a PyTree of arrays matches up with a PyTree of ShapeDtypeStructs, up to a uniform batch shape.

    **Arguments:**

    - `spec_tree`: Pytree with `jax.ShapeDtypeStruct` leaves (at positions you want checked).
    - `data_tree`: Pytree with arrays at matching positions.

    **Returns:**

    The extracted batch shape if the two pytrees are compatible
    """

    if not jax.tree.structure(spec_tree) == jax.tree.structure(data_tree):
        raise RuntimeError("Tree structure mismatch between shape/dtype spec and data")

    spec_leaves, _ = jax.tree.flatten_with_path(spec_tree)
    val_leaves, _ = jax.tree.flatten_with_path(data_tree)

    batch_shape = None

    for (path, spec_leaf), (_, val_leaf) in zip(spec_leaves, val_leaves):
        if isinstance(spec_leaf, jax.ShapeDtypeStruct):
            if not eqx.is_array(val_leaf):
                raise RuntimeError("Array missing from data")

            vshape, vdtype = val_leaf.shape, val_leaf.dtype
            sshape, sdtype = spec_leaf.shape, spec_leaf.dtype

            val_shape_without_batch = (
                () if not len(sshape) else vshape[-(len(sshape)) :]
            )

            if val_shape_without_batch != sshape:
                raise RuntimeError("Shape of data mismatched with spec")

            cur_batch = vshape[: len(vshape) - len(sshape)]
            if batch_shape is None:
                batch_shape = cur_batch
            elif cur_batch != batch_shape:
                raise RuntimeError("Inconsistent batch shape in data")

            if vdtype != sdtype:
                raise RuntimeError(f"Data has incorrect type {vdtype} vs {sdtype}")

    return batch_shape


def verify_block_state(
    blocks: list[Block],
    states: list[_State],
    node_shape_dtypes: _Node_SD,
    block_axis: int | None = None,
) -> None:
    """
    Check that a state is what it should be given some blocks and node shape/dtypes.

    Passing incompatible state information into hamon functions can lead to unintended casting/other weird silent
    errors, so we should always check this.

    **Arguments:**

    - `blocks`: A list of Blocks.
    - `states`: A list of states to verify against blocks.
    - `node_shape_dtypes`: Maps every node class to its
        `jax.ShapeDtypeStruct` PyTree template.
    - `block_axis`: Index in the state batch shape at which to expect the block length.

    **Returns:**

    None. Raises RuntimeError if blocks and states are incompatible.
    """

    if not len(blocks) == len(states):
        raise RuntimeError("Number of states not equal to number of blocks")

    for block, state in zip(blocks, states):
        expected_sd = node_shape_dtypes[type(block.nodes[0])]
        batch_shape = _check_pytree_compat(expected_sd, state)
        assert batch_shape is not None
        if block_axis is not None:
            if not batch_shape[block_axis] == len(block.nodes):
                raise RuntimeError("State shape did not match detected block length")
