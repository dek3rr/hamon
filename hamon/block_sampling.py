# Modified from the original thrml library (https://github.com/Extropic-AI/thrml)

import contextlib
import dataclasses
from collections import defaultdict
from typing import TypeAlias
from collections.abc import Mapping, Sequence

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Key, PyTree, Shaped

from hamon.block_management import (
    Block,
    BlockSpec,
    _block_layout,
    block_state_to_global,
    from_global_state,
    scatter_block_to_global,
    to_per_block_layout,
    verify_block_state,
)
from hamon.device import (
    DeviceLike,
    free_node_count,
    resolve_entry_device,
    tree_device_put,
)
from hamon.interaction import InteractionGroup
from hamon.pgm import DEFAULT_NODE_SHAPE_DTYPES, AbstractNode

from .conditional_samplers import AbstractConditionalSampler, _SamplerState
from .observers import AbstractObserver, ObserveCarry, StateObserver

# A SuperBlock is a tuple of Blocks that share the same global state during sampling.
SuperBlock: TypeAlias = tuple[Block, ...] | Block
_SD: TypeAlias = Mapping[type[AbstractNode], PyTree[jax.ShapeDtypeStruct]]


class BlockGibbsSpec(BlockSpec):
    """
    A BlockGibbsSpec is a type of BlockSpec which contains additional information
    on free and clamped blocks.

    This entity also supports `SuperBlock`s, which are merely groups of blocks
    which are sampled at the same time algorithmically, but not programmatically.
    That is to say, superblock = (block1, block2) means that the states input to
    block1 and block2 are the same, but they are not executed at the same time.
    This may be because they are the same color on a graph, but require vastly
    different sampling methods such that JAX SIMD approaches are not feasible
    to parallelize them.

    A recurring theme in `hamon` is the importance of implicit indexing. One
    such example can be seen here. Because global states are created by
    concatenating lists of free and clamped blocks, providing the inputs
    in the same order as the blocks are defined is essential. This is almost
    always taken care of internally, but when writing custom functions or
    interfaces this is important to keep in mind.

    **Attributes:**

    - `free_blocks`: the list of free blocks (in order)
    - `sampling_order`: a list of `len(superblocks)` lists, where each
        `sampling_order[i]` is the index of `free_blocks` to sample.
        Sampling is done by iterating over this order and sampling each
        sublist of free blocks at the same algorithmic time.
    - `clamped_blocks`: the list of clamped blocks
    - `superblocks`: the list of superblocks
    """

    free_blocks: list[Block]
    sampling_order: list[list[int]]
    clamped_blocks: list[Block]
    superblocks: list[tuple[Block, ...]]

    def __init__(
        self,
        free_super_blocks: Sequence[SuperBlock],
        clamped_blocks: list[Block],
        node_shape_dtypes: _SD = DEFAULT_NODE_SHAPE_DTYPES,
    ):
        """Create a Gibbs specification from free and clamped blocks.

        **Arguments:**

        - `free_super_blocks`: An ordered sequence where each element is either
            a single `Block`, or a tuple of blocks that must share the same global
            state when calling their individual samplers.
        - `clamped_blocks`: Blocks whose nodes stay fixed during sampling.
        - `node_shape_dtypes`: Mapping from node class to a PyTree of
            `jax.ShapeDtypeStruct`; identical to the argument in `BlockSpec`.
        """
        free_blocks = []
        sampling_order = []
        superblocks = []
        i = 0
        for super_block in free_super_blocks:
            if isinstance(super_block, Block):
                blocks = (super_block,)
            else:
                blocks = super_block

            superblocks.append(blocks)
            sampling_group = []
            for block in blocks:
                free_blocks.append(block)
                sampling_group.append(i)
                i += 1
            sampling_order.append(sampling_group)

        super().__init__(free_blocks + clamped_blocks, node_shape_dtypes)
        self.free_blocks = free_blocks
        self.clamped_blocks = clamped_blocks
        self.sampling_order = sampling_order
        self.superblocks = superblocks


def _tree_slice(x, sl):
    if eqx.is_array(x):
        return jnp.take(x, sl, axis=0)
    return x


def _bind_one_interaction(interaction, interaction_slices, active_arr):
    """Gather an interaction group's tensor for one free block and zero its pad."""
    sliced = jax.tree.map(lambda x: _tree_slice(x, interaction_slices), interaction)

    def _premask(x):
        if eqx.is_array(x):
            mask = active_arr.astype(x.dtype)
            return x * mask.reshape(mask.shape + (1,) * (x.ndim - 2))
        return x

    return jax.tree.map(_premask, sliced)


@eqx.filter_jit
def _bind_weight_recipe(interactions, slices, masks):
    """Bind every (block, group) weight tensor in one traced pass.

    The per-block gathers and pad-masks are all different shapes, so run eagerly
    they each pay a first-shape XLA compile — dozens of them per construction,
    and again on every ``with_ebm``. Fused under one jit they become a single
    executable that a repeat ``with_ebm`` (same graph, re-scaled weights) reuses.
    ``interactions`` is the recipe-ordered nesting of interaction pytrees;
    ``slices`` and ``masks`` are the parallel cached index / active-mask arrays.
    """
    return [
        [
            _bind_one_interaction(inter, sl, mask)
            for inter, sl, mask in zip(block_inter, block_sl, block_mask)
        ]
        for block_inter, block_sl, block_mask in zip(interactions, slices, masks)
    ]


def _build_output_sd(block: Block, template_sd: PyTree) -> PyTree:
    """Resize a template ShapeDtypeStruct pytree for *block*'s node count."""

    def _resize(leaf):
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct((len(block.nodes), *leaf.shape), leaf.dtype)
        return leaf

    return jax.tree.map(_resize, template_sd)


@dataclasses.dataclass
class _BlockStructure:
    """Weight-independent result of building a :class:`BlockSamplingProgram`.

    Every field here is fixed by the *graph* (the spec partition + the
    interaction-group node structure), not by the interaction weight *values*.
    Caching it lets ``program.with_ebm(...)`` — same graph, β-scaled weights —
    skip the ``O(nodes × interactions)`` host construction loops and only re-bind
    the weight tensors. ``weight_recipe[b]`` lists, for free block ``b``, one
    ``(group_index, interaction_slices, active_arr, global_inds, global_slices)``
    tuple per interaction group it reads.
    """

    gibbs_spec: "BlockGibbsSpec"
    weight_recipe: list
    block_sd_inds: list
    block_positions: list
    block_output_sds: list
    block_slice_starts: list
    block_owns_slot: list


# Keyed on node-identity structure only (weight values deliberately
# excluded); the cached spec keeps its nodes alive, so the id()-based key
# cannot suffer reuse-after-GC false hits.
_STRUCTURE_CACHE: dict = {}


def _structure_cache_key(gibbs_spec: "BlockGibbsSpec", interaction_groups):
    return (
        gibbs_spec._structure_key(),
        tuple(
            (
                type(ig),
                ig.head_nodes._ids(),
                tuple(tb._ids() for tb in ig.tail_nodes),
            )
            for ig in interaction_groups
        ),
    )


def _build_block_structure(
    gibbs_spec: "BlockGibbsSpec", interaction_groups
) -> _BlockStructure:
    """Run the weight-independent block-structure construction (see __init__)."""
    node_to_block_idx = {
        node: b_idx
        for b_idx, block in enumerate(gibbs_spec.blocks)
        for node in block.nodes
    }
    already_one_per_slot = all(
        len(slot) <= 1 for slot in gibbs_spec.block_to_global_slice_spec
    )

    head_node_map = defaultdict(list)
    for i, interaction_group in enumerate(interaction_groups):
        for j, node in enumerate(interaction_group.head_nodes.nodes):
            head_node_map[node].append((i, j))

    interaction_inds = []
    max_n_interactions = []
    for block in gibbs_spec.free_blocks:
        this_block_interaction_info = [
            [[] for _ in range(len(block.nodes))]
            for _ in range(len(interaction_groups))
        ]
        for j, node in enumerate(block.nodes):
            for info in head_node_map[node]:
                this_block_interaction_info[info[0]][j].append(info[1])
        interaction_inds.append(this_block_interaction_info)
        max_n_interactions.append(
            [
                max([len(x) for x in this_int])
                for this_int in this_block_interaction_info
            ]
        )

    def _block_split_safe(block_interact_inds) -> bool:
        for ig, interact_inds in zip(interaction_groups, block_interact_inds):
            for tail_block in ig.tail_nodes:
                used_slots = {
                    node_to_block_idx.get(tail_block.nodes[ind], -1)
                    for inds in interact_inds
                    for ind in inds
                }
                if -1 in used_slots or len(used_slots) > 1:
                    return False
        return True

    split_safe = all(_block_split_safe(b) for b in interaction_inds)
    if split_safe and not already_one_per_slot:
        gibbs_spec = to_per_block_layout(gibbs_spec)

    weight_recipe = []
    for block, block_interact_inds, block_n_interactions in zip(
        gibbs_spec.free_blocks, interaction_inds, max_n_interactions
    ):
        block_recipe = []
        for g_idx, (interaction_group, interact_inds, n_interactions) in enumerate(
            zip(interaction_groups, block_interact_inds, block_n_interactions)
        ):
            if n_interactions > 0:
                n_nodes = len(block.nodes)
                interaction_slices = np.zeros((n_nodes, n_interactions), dtype=int)
                global_inds: list[int | None] = [
                    None for _ in interaction_group.tail_nodes
                ]
                global_slices = [
                    np.zeros((n_nodes, n_interactions), dtype=int)
                    for _ in interaction_group.tail_nodes
                ]
                active = np.zeros((n_nodes, n_interactions), dtype=bool)
                for i, inds in enumerate(interact_inds):
                    for j, ind in enumerate(inds):
                        interaction_slices[i, j] = ind
                        active[i, j] = 1
                        for k, tail_block in enumerate(interaction_group.tail_nodes):
                            loc = gibbs_spec.node_global_location_map[
                                tail_block.nodes[ind]
                            ]
                            global_slices[k][i, j] = loc[1]
                            if global_inds[k] is None:
                                global_inds[k] = loc[0]
                            elif global_inds[k] != loc[0]:
                                raise RuntimeError(
                                    "Tail neighbors of a free block span "
                                    "multiple global slots; cannot build a "
                                    "single-slot gather."
                                )
                for k, tail_block in enumerate(interaction_group.tail_nodes):
                    if global_inds[k] is None:
                        global_inds[k] = gibbs_spec.node_global_location_map[
                            tail_block.nodes[0]
                        ][0]
                # device_put, not jnp.array: these are concrete host arrays, so
                # a plain transfer avoids the first-shape XLA "stage" compile
                # that jnp.array pays once per distinct block shape.
                block_recipe.append(
                    (
                        g_idx,
                        jax.device_put(interaction_slices),
                        jax.device_put(active),
                        global_inds,
                        [jax.device_put(x) for x in global_slices],
                    )
                )
        weight_recipe.append(block_recipe)

    block_sd_inds = []
    block_positions = []
    block_output_sds = []
    block_slice_starts = []
    block_owns_slot = []
    for block in gibbs_spec.free_blocks:
        sd_ind, start, locs = _block_layout(block, gibbs_spec)
        block_sd_inds.append(sd_ind)
        block_positions.append(jax.device_put(locs))
        block_slice_starts.append(start)
        block_owns_slot.append(len(gibbs_spec.block_to_global_slice_spec[sd_ind]) == 1)
        template_sd = gibbs_spec.node_shape_struct[block.node_type]
        block_output_sds.append(_build_output_sd(block, template_sd))

    return _BlockStructure(
        gibbs_spec=gibbs_spec,
        weight_recipe=weight_recipe,
        block_sd_inds=block_sd_inds,
        block_positions=block_positions,
        block_output_sds=block_output_sds,
        block_slice_starts=block_slice_starts,
        block_owns_slot=block_owns_slot,
    )


class BlockSamplingProgram(eqx.Module):
    """A PGM block-sampling program.

    This class encapsulates everything that is needed to run a PGM block sampling program in hamon.
    `per_block_interactions` and `per_block_interaction_active` are parallel to the free blocks in `gibbs_spec`, and
    their members are passed directly to a sampler when the state of the corresponding free block is being updated
    during a sampling program. `per_block_interaction_global_inds` and `per_block_interaction_global_slices` are
    also parallel to the free blocks, and are used to slice the global state of the program to produce the
    state information required to update the state of each block alongside the static information contained in the
    interactions.

    **Attributes:**

    - `gibbs_spec`: A division of some PGM into free and clamped blocks.
    - `samplers`: A sampler to use to update every free block in `gibbs_spec`.
    - `per_block_interactions`: All the interactions that touch each free block in `gibbs_spec`.
    - `per_block_interaction_active`: indicates which interactions are real
        and which interactions are not part of the model and have been added to pad data structures so that they
        can be rectangular.
    - `per_block_interaction_global_inds`: how to find the information required to update each block within the global
        state list
    - `per_block_interaction_global_slices`: how to slice each array in the global state list to find the information
        required to update each block
    - `_block_sd_inds`: precomputed sd_index for each free block (avoids recomputing inside scan)
    - `_block_positions`: precomputed node positions in global state for each free block (avoids recomputing inside scan)
    - `_block_output_sds`: precomputed output ShapeDtypeStruct pytree for each free block
    - `_block_slice_starts`: static start index when the block occupies a contiguous
        range of the global state (always the case for blocks laid out by `BlockSpec`),
        or `None` to fall back to a gather-index scatter. A contiguous range lets the
        write-back lower to `lax.dynamic_update_slice` instead of a scatter, which XLA
        fuses far better.
    """

    def with_ebm(self, ebm) -> "BlockSamplingProgram":
        """Return a copy of this program rewired to a different EBM.

        Subclasses that want to work with `tune_schedule(ebm=..., program=...)`
        must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement with_ebm(). "
            "Either implement it or provide explicit factory callables to tune_schedule."
        )

    gibbs_spec: BlockGibbsSpec
    samplers: list[AbstractConditionalSampler]
    per_block_interactions: list[list[PyTree]]
    per_block_interaction_active: list[list[Array]]
    per_block_interaction_global_inds: list[list[list[int]]]
    per_block_interaction_global_slices: list[list[list[Array]]]
    # Precomputed scatter indices per free block, used by _run_blocks to avoid
    # calling get_node_locations inside the traced scan body.
    _block_sd_inds: list[int]
    _block_positions: list[Array]
    _block_output_sds: list[PyTree]
    _block_slice_starts: list[int | None]
    # True when a free block is the sole occupant of its global slot, so the
    # write-back replaces the whole slot rather than slicing into it.
    _block_owns_slot: list[bool]

    def __init__(
        self,
        gibbs_spec: BlockGibbsSpec,
        samplers: list[AbstractConditionalSampler],
        interaction_groups: list[InteractionGroup],
    ):
        """Construct a `BlockSamplingProgram`.

        Takes in a set of information that implicitly defines a sampling program
        and manipulates it into a shape appropriate for vectorized block-sampling.
        This involves reindexing, slicing, and often padding.

        **Arguments:**

        - `gibbs_spec`: A division of some PGM into free and clamped blocks.
        - `samplers`: The update rule to use for each free block in `gibbs_spec`.
        - `interaction_groups`: A list of `InteractionGroups` that define how the
            variables in your sampling program affect one another.
        """

        self.samplers = samplers

        n_free_blocks = len(gibbs_spec.free_blocks)
        if len(self.samplers) != n_free_blocks:
            raise ValueError(
                f"Expected {n_free_blocks} samplers, received {len(self.samplers)}"
            )

        # Block structure is fixed by the graph, not the weight values, so
        # cache it across with_ebm rebuilds — only the weight tensors below
        # are re-bound per construction.
        key = _structure_cache_key(gibbs_spec, interaction_groups)
        struct = _STRUCTURE_CACHE.get(key)
        if struct is None:
            struct = _build_block_structure(gibbs_spec, interaction_groups)
            _STRUCTURE_CACHE[key] = struct

        self.gibbs_spec = struct.gibbs_spec

        # Bind the weights: slice each group's tensor by the cached indices and
        # pre-zero padded entries with the cached active mask. The gather+mask
        # runs under one fused jit (see _bind_weight_recipe); the active masks
        # and global slice indices are weight-independent, so they are copied
        # straight from the cached recipe with no device work.
        recipe_interactions = [
            [interaction_groups[g_idx].interaction for g_idx, *_ in block_recipe]
            for block_recipe in struct.weight_recipe
        ]
        recipe_slices = [
            [interaction_slices for _g, interaction_slices, *_ in block_recipe]
            for block_recipe in struct.weight_recipe
        ]
        per_block_interaction_active = [
            [active_arr for _g, _sl, active_arr, *_ in block_recipe]
            for block_recipe in struct.weight_recipe
        ]
        per_block_interaction_global_inds = [
            [global_inds for *_, global_inds, _gs in block_recipe]
            for block_recipe in struct.weight_recipe
        ]
        per_block_interaction_global_slices = [
            [global_slices for *_, global_slices in block_recipe]
            for block_recipe in struct.weight_recipe
        ]
        per_block_interactions = _bind_weight_recipe(
            recipe_interactions, recipe_slices, per_block_interaction_active
        )

        self.per_block_interactions = per_block_interactions
        self.per_block_interaction_active = per_block_interaction_active
        self.per_block_interaction_global_inds = per_block_interaction_global_inds
        self.per_block_interaction_global_slices = per_block_interaction_global_slices
        self._block_sd_inds = struct.block_sd_inds
        self._block_positions = struct.block_positions
        self._block_output_sds = struct.block_output_sds
        self._block_slice_starts = struct.block_slice_starts
        self._block_owns_slot = struct.block_owns_slot


_State: TypeAlias = PyTree[Shaped[Array, "nodes ?*state"], "_State"]


def sample_single_block(
    key: Key[Array, ""],
    state_free: list[_State],
    clamp_state: list[_State],
    program: BlockSamplingProgram,
    block: int,
    sampler_state: _SamplerState,
    global_state: list[PyTree] | None = None,
    per_block_interactions: list[list[PyTree]] | None = None,
) -> tuple[_State, _SamplerState]:
    """Samples a single block within a Gibbs sampling program based on the current
    states and program configurations. It extracts neighboring states, processes
    required data, and applies a sampling function to generate output samples.

    **Arguments:**

    - `key`: Pseudo-random number generator key to ensure reproducibility of sampling.
    - `state_free`: Current states of free blocks, representing the values to be
        updated during sampling.
    - `clamp_state`: Clamped states that remain fixed during the sampling process.
    - `program`: The Gibbs sampling program containing specifications, samplers,
        neighborhood information, and parameters.
    - `block`: Index of the block to be sampled in the current iteration.
    - `sampler_state`: The current state of the sampler that will be used to
        perform the update.
    - `global_state`: Optionally precomputed global state for the concatenated
        free and clamped blocks; when omitted the function constructs it internally.
    - `per_block_interactions`: Optional override for the interaction weights. When
        provided (e.g. inside a vmapped multi-chain runner), this is used instead of
        `program.per_block_interactions`. The caller is responsible for ensuring the
        PyTree structure matches `program.per_block_interactions`.

    **Returns:**

    - Updated block state and sampler state for the specified block.
    """
    if global_state is None:
        global_state = block_state_to_global(
            state_free + clamp_state, program.gibbs_spec
        )
    per_interaction_global_inds = program.per_block_interaction_global_inds[block]
    per_interaction_slices = program.per_block_interaction_global_slices[block]

    all_interaction_states = []
    for interaction_global_inds, interaction_slices in zip(
        per_interaction_global_inds, per_interaction_slices
    ):
        this_interaction_states = []
        for ind, sl in zip(interaction_global_inds, interaction_slices):
            this_interaction_states.append(
                jax.tree.map(
                    lambda x: jnp.take(x, sl, axis=0),  # shape -> (n, m, …)
                    global_state[ind],
                )
            )
        all_interaction_states.append(this_interaction_states)

    sd_to_pass = program._block_output_sds[block]

    block_interactions = (
        per_block_interactions[block]
        if per_block_interactions is not None
        else program.per_block_interactions[block]
    )

    sampler = program.samplers[block]
    out_samples, out_sampler_state = sampler.sample(
        key,
        block_interactions,
        program.per_block_interaction_active[block],
        all_interaction_states,
        sampler_state,
        sd_to_pass,
    )
    return out_samples, out_sampler_state


def sample_blocks(
    key: Key[Array, ""],
    state_free: list[_State],
    clamp_state: list[_State],
    program: BlockSamplingProgram,
    sampler_state: list[_SamplerState],
) -> tuple[list[_State], list[_SamplerState]]:
    """Perform one iteration of sampling, visiting every block.

    **Arguments:**

    - `key`: The JAX PRNG key.
    - `state_free`: The state of the free blocks.
    - `clamp_state`: The state of the clamped blocks.
    - `program`: The Gibbs program.
    - `sampler_state`: The state of the sampler.

    **Returns:**

    - Updated free-block state list and sampler-state list.
    """
    if __debug__:
        sds = program.gibbs_spec.node_shape_struct
        verify_block_state(program.gibbs_spec.free_blocks, state_free, sds, -1)
        verify_block_state(program.gibbs_spec.clamped_blocks, clamp_state, sds, -1)

    # Work on copies so the caller's lists are never mutated.
    state_free = list(state_free)
    sampler_state = list(sampler_state)

    keys = jax.random.split(key, (len(program.gibbs_spec.free_blocks),))
    global_state = block_state_to_global(state_free + clamp_state, program.gibbs_spec)

    for sampling_group in program.gibbs_spec.sampling_order:
        state_updates = {}
        for i in sampling_group:
            state_updates[i], sampler_state[i] = sample_single_block(
                keys[i],
                state_free,
                clamp_state,
                program,
                i,
                sampler_state[i],
                global_state,
            )
        for i, new_state in state_updates.items():
            state_free[i] = new_state
            # Targeted scatter: update only the positions that changed rather
            # than rebuilding the full global tensor at the next group boundary.
            global_state = scatter_block_to_global(
                global_state,
                new_state,
                program.gibbs_spec.free_blocks[i],
                program.gibbs_spec,
            )

    return state_free, sampler_state


def _run_blocks(
    key: Key[Array, ""],
    program: BlockSamplingProgram,
    init_chain_state: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    n_iters: int,
    sampler_states: list[_SamplerState],
    per_block_interactions: list[list[PyTree]] | None = None,
) -> tuple[
    list[PyTree[Shaped[Array, "nodes ?*state"]]], list[_SamplerState], list[PyTree]
]:
    """Perform `n_iters` steps of block sampling.

    The scan carries only the sampler states and the concatenated global
    state. Free-block states would duplicate data already present in the
    global state (samplers read exclusively from the global state), so they
    are extracted once after the scan instead of being threaded through it.

    **Arguments:**

    - `per_block_interactions`: Optional override for interaction weights.
        When provided, replaces `program.per_block_interactions` throughout
        the scan. Used by parallel tempering to inject per-chain β-scaled
        weights into a vmapped runner.
    """

    # Build global state once before scan (clamped slice is static).
    init_global_state = block_state_to_global(
        init_chain_state + state_clamp, program.gibbs_spec
    )

    if n_iters == 0:
        return init_chain_state, sampler_states, init_global_state

    pbi = (
        per_block_interactions
        if per_block_interactions is not None
        else program.per_block_interactions
    )

    block_sd_inds = program._block_sd_inds
    block_positions = program._block_positions
    block_slice_starts = program._block_slice_starts
    block_owns_slot = program._block_owns_slot

    def body_fn(carry, _key):
        sampler_state, global_state = carry

        keys = jax.random.split(_key, len(program.gibbs_spec.free_blocks))

        for sampling_group in program.gibbs_spec.sampling_order:
            # Collect all updates for this group before writing back.
            new_states = {}
            new_sampler_states = {}
            for i in sampling_group:
                new_states[i], new_sampler_states[i] = sample_single_block(
                    keys[i],
                    [],
                    state_clamp,
                    program,
                    i,
                    sampler_state[i],
                    global_state,
                    per_block_interactions=pbi,
                )

            sampler_state = [
                new_sampler_states[i] if i in new_sampler_states else sampler_state[i]
                for i in range(len(sampler_state))
            ]
            for i in new_states:
                sd_ind = block_sd_inds[i]
                new_global = list(global_state)
                if block_owns_slot[i]:
                    # Sole occupant of its slot (per-block layout): replace
                    # the whole slot with no slice/scatter — the main lever
                    # against dispatch-bound Gibbs sweeps.
                    new_global[sd_ind] = new_states[i]
                elif block_slice_starts[i] is not None:
                    # Contiguous block: a static-offset dynamic_update_slice,
                    # which XLA fuses far better than a gather-index scatter.
                    start = block_slice_starts[i]
                    new_global[sd_ind] = jax.tree.map(
                        lambda g, s: jax.lax.dynamic_update_slice_in_dim(
                            g, s, start, axis=0
                        ),
                        global_state[sd_ind],
                        new_states[i],
                    )
                else:
                    positions = block_positions[i]
                    new_global[sd_ind] = jax.tree.map(
                        lambda g, s: g.at[positions].set(s),
                        global_state[sd_ind],
                        new_states[i],
                    )
                global_state = new_global

        return (sampler_state, global_state), None

    keys = jax.random.split(key, n_iters)
    (final_sampler_states, final_global), _ = jax.lax.scan(
        body_fn, (sampler_states, init_global_state), keys
    )
    # Free-block states are contiguous slices of the global state, so this
    # extraction lowers to static slices that XLA fuses away.
    final_state_free = from_global_state(
        final_global, program.gibbs_spec, program.gibbs_spec.free_blocks
    )
    return final_state_free, final_sampler_states, final_global


@dataclasses.dataclass(frozen=True)
class SamplingSchedule:
    """
    Represents a sampling schedule for a process.

    Frozen so it is safely hashable as a static ``jit`` argument: ``frozen=True``
    auto-generates a value-based ``__hash__`` and makes the schedule immutable,
    so one already used as a compilation cache key cannot be mutated out from
    under the cache.

    **Attributes:**

    - `n_warmup`: The number of warmup steps to run before collecting samples.
    - `n_samples`: The number of samples to collect.
    - `steps_per_sample`: The number of steps to run between each sample.
    """

    n_warmup: int
    n_samples: int
    steps_per_sample: int


def sample_with_observation(
    key: Key[Array, ""],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_chain_state: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    observation_carry_init: ObserveCarry,
    f_observe: AbstractObserver,
    *,
    device: DeviceLike = "auto",
) -> tuple[ObserveCarry, list[PyTree[Shaped[Array, "n_samples nodes ?*state"]]]]:
    """Run the full chain and call an Observer after every recorded sample.

    **Arguments:**

    - `key`: RNG key.
    - `program`: The sampling program.
    - `schedule`: Warm-up length, number of samples, number of steps between samples.
    - `init_chain_state`: Initial free-block state.
    - `state_clamp`: Clamped-block state.
    - `observation_carry_init`: Initial carry handed to `f_observe`.
    - `f_observe`: Observer instance.
    - `device`: Where to run — `"auto"` (default; small workloads on CPU, large
        on a visible accelerator), `"cpu"`/`"gpu"`, a concrete `jax.Device`, or
        `None` to leave placement untouched. See `hamon.device`. A no-op when
        called inside jit/vmap/grad.

    **Returns:**

    - Tuple `(final_observer_carry, samples)` where `samples` is a PyTree whose
        leading axis has size `schedule.n_samples`.
    """
    dev = resolve_entry_device(
        device,
        n_chains=1,
        n_nodes=free_node_count(program),
        arrays=(init_chain_state, state_clamp, observation_carry_init, key),
    )
    if dev is not None:
        (
            key,
            program,
            init_chain_state,
            state_clamp,
            observation_carry_init,
            f_observe,
        ) = tree_device_put(
            (
                key,
                program,
                init_chain_state,
                state_clamp,
                observation_carry_init,
                f_observe,
            ),
            dev,
        )
    device_ctx = (
        jax.default_device(dev) if dev is not None else contextlib.nullcontext()
    )

    # Device placement stays out here (a no-op under jit/vmap); the jitted
    # core compiles its scans once and reuses them across calls — the old
    # eager path recompiled ~0.9 s of XLA per call to run single-digit ms of
    # sampling. Static `schedule` specializes per warmup/sample/step counts.
    with device_ctx:
        return _sample_with_observation_core(
            key,
            program,
            schedule,
            init_chain_state,
            state_clamp,
            observation_carry_init,
            f_observe,
        )


@eqx.filter_jit
def _sample_with_observation_core(
    key: Key[Array, ""],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_chain_state: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    observation_carry_init: ObserveCarry,
    f_observe: AbstractObserver,
) -> tuple[ObserveCarry, list[PyTree[Shaped[Array, "n_samples nodes ?*state"]]]]:
    """Jitted compute core of :func:`sample_with_observation`.

    Split out so the warmup + sampling scans compile once and reuse the cache;
    `schedule` is a non-array (static) argument that keys the cache, and all
    placement happens in the caller before this runs.
    """
    sampler_states = [s.init() for s in program.samplers]

    key, subkey = jax.random.split(key)
    warmup_state, warmup_sampler_states, warmup_global = _run_blocks(
        subkey,
        program,
        init_chain_state,
        state_clamp,
        schedule.n_warmup,
        sampler_states,
    )
    mem, warmup_observation = f_observe(
        program,
        warmup_state,
        state_clamp,
        observation_carry_init,
        jnp.array(0),
        warmup_global,
    )

    if schedule.n_samples <= 1:
        warmup_observation = jax.tree.map(lambda x: x[None], warmup_observation)
        return mem, warmup_observation

    def body_fn(carry, input):
        (prev_state, prev_sampler_state), _mem = carry

        _key, i = input

        new_state, new_sampler_state, new_global = _run_blocks(
            _key,
            program,
            prev_state,
            state_clamp,
            schedule.steps_per_sample,
            prev_sampler_state,
        )
        _mem, observe_out = f_observe(
            program, new_state, state_clamp, _mem, i, new_global
        )
        new_carry = ((new_state, new_sampler_state), _mem)
        return new_carry, observe_out

    keys = jax.random.split(key, schedule.n_samples - 1)
    outer_iters = jnp.arange(1, schedule.n_samples)

    inputs = (keys, outer_iters)

    (_, mem_out), observed_results = jax.lax.scan(
        body_fn, ((warmup_state, warmup_sampler_states), mem), inputs
    )

    # need to prepend the first observation from the warmup
    def prepend_warmup_observation(_warmup, _rest):
        return jnp.concatenate([_warmup[None], _rest], axis=0)

    observed_results = jax.tree.map(
        prepend_warmup_observation, warmup_observation, observed_results
    )

    return mem_out, observed_results


def sample_states(
    key: Key[Array, ""],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state_free: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    nodes_to_sample: list[Block],
    *,
    device: DeviceLike = "auto",
) -> list[PyTree[Shaped[Array, "n_samples nodes ?*state"]]]:
    """Convenience wrapper to collect state information for *nodes_to_sample* only.

    Internally builds a [`hamon.StateObserver`][], runs
    [`hamon.sample_with_observation`][] (which `device` is forwarded to), and
    returns a stacked tensor of shape `(schedule.n_samples, ...)`.
    """
    f_observe = StateObserver(nodes_to_sample)
    carry_init = f_observe.init()

    mem_out, results_out = sample_with_observation(
        key,
        program,
        schedule,
        init_state_free,
        state_clamp,
        carry_init,
        f_observe,
        device=device,
    )

    return results_out


def sample_states_batched(
    key: Key[Array, ""],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_states_free: list[PyTree[Shaped[Array, "chains nodes ?*state"]]],
    state_clamp: list[_State],
    nodes_to_sample: list[Block],
    *,
    device: DeviceLike = "auto",
) -> list[PyTree[Shaped[Array, "chains n_samples nodes ?*state"]]]:
    """Run several independent single-chain draws in parallel via `jax.vmap`.

    A single-chain `sample_states` launches one tiny kernel per Gibbs sweep, so
    on an accelerator it is dispatch-bound (the GPU sits idle between launches).
    Running ``n_chains`` chains under one `vmap` keeps the launch count the same
    while each kernel does ``n_chains`` times the work, so the same wall time
    yields ``n_chains`` times the samples — or, splitting a fixed sample budget
    across chains, fewer sample-collection iterations for the same total.

    The chains are independent (separate keys) and share the program, schedule
    and clamped state. Device routing happens once here with the full
    ``n_chains × free nodes`` work score; the inner `sample_with_observation`
    calls see tracers, so their own routing is a no-op.

    **Arguments:**

    - `init_states_free`: per-free-block states with a leading ``n_chains`` axis.
    - other arguments: as in [`hamon.sample_states`][].

    **Returns:**

    - A list parallel to `nodes_to_sample`; each entry has shape
      ``(n_chains, schedule.n_samples, ...)``.
    """
    leaves = jax.tree.leaves(init_states_free)
    n_chains = int(leaves[0].shape[0]) if leaves else 1

    dev = resolve_entry_device(
        device,
        n_chains=n_chains,
        n_nodes=free_node_count(program),
        arrays=(init_states_free, state_clamp, key),
    )
    if dev is not None:
        key, program, init_states_free, state_clamp = tree_device_put(
            (key, program, init_states_free, state_clamp), dev
        )
    device_ctx = (
        jax.default_device(dev) if dev is not None else contextlib.nullcontext()
    )

    f_observe = StateObserver(nodes_to_sample)
    carry_init = f_observe.init()
    keys = jax.random.split(key, n_chains)
    n_free = len(program.gibbs_spec.free_blocks)

    def _one_chain(k, init_free):
        _, results = sample_with_observation(
            k,
            program,
            schedule,
            init_free,
            state_clamp,
            carry_init,
            f_observe,
            device=None,
        )
        return results

    with device_ctx:
        return jax.vmap(_one_chain, in_axes=(0, [0] * n_free))(keys, init_states_free)
