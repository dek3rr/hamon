# Modified from the original thrml library (https://github.com/Extropic-AI/thrml)

import dataclasses
from collections import defaultdict
from typing import Mapping, Sequence, Type, TypeAlias

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Key, PyTree, Shaped

from hamon.block_management import (
    Block,
    BlockSpec,
    block_state_to_global,
    scatter_block_to_global,
    get_node_locations,
    verify_block_state,
)
from hamon.interaction import InteractionGroup
from hamon.pgm import DEFAULT_NODE_SHAPE_DTYPES, AbstractNode

from .conditional_samplers import AbstractConditionalSampler, _SamplerState
from .observers import AbstractObserver, ObserveCarry, StateObserver

# A SuperBlock is a tuple of Blocks that share the same global state during sampling.
SuperBlock: TypeAlias = tuple[Block, ...] | Block
_SD: TypeAlias = Mapping[Type[AbstractNode], PyTree[jax.ShapeDtypeStruct]]


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


def _build_output_sd(block: Block, template_sd: PyTree) -> PyTree:
    """Resize a template ShapeDtypeStruct pytree for *block*'s node count."""

    def _resize(leaf):
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct((len(block.nodes), *leaf.shape), leaf.dtype)
        return leaf

    return jax.tree.map(_resize, template_sd)


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
    """

    def with_ebm(self, ebm) -> "BlockSamplingProgram":
        """Return a copy of this program rewired to a different EBM.

        Subclasses that want to work with `nrpt_adaptive(ebm=..., program=...)`
        must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement with_ebm(). "
            "Either implement it or provide explicit factory callables to nrpt_adaptive."
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

        self.gibbs_spec = gibbs_spec
        self.samplers = samplers

        n_free_blocks = len(self.gibbs_spec.free_blocks)
        if len(self.samplers) != n_free_blocks:
            raise ValueError(
                f"Expected {n_free_blocks} samplers, received {len(self.samplers)}"
            )

        # Map every head node to (interaction_group_index, position_within_group).
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
                this_node_interaction_info = head_node_map[node]
                for info in this_node_interaction_info:
                    this_block_interaction_info[info[0]][j].append(info[1])
            interaction_inds.append(this_block_interaction_info)
            this_max_n = [
                max([len(x) for x in this_int])
                for this_int in this_block_interaction_info
            ]
            max_n_interactions.append(this_max_n)

        # Build per-block interaction data and global-state slicers.
        per_block_interactions = []
        per_block_interaction_active = []
        per_block_interaction_global_inds = []
        per_block_interaction_global_slices = []

        for block, block_interact_inds, block_n_interactions in zip(
            gibbs_spec.free_blocks, interaction_inds, max_n_interactions
        ):
            this_block_interactions = []
            this_block_active = []
            this_block_global_inds = []
            this_block_global_slices = []
            for interaction_group, interact_inds, n_interactions in zip(
                interaction_groups, block_interact_inds, block_n_interactions
            ):
                if n_interactions > 0:
                    n_nodes = len(block.nodes)
                    interaction_slices = np.zeros((n_nodes, n_interactions), dtype=int)

                    global_inds = []
                    global_slices = []
                    for tail_block in interaction_group.tail_nodes:
                        global_inds.append(
                            gibbs_spec.node_global_location_map[tail_block.nodes[0]][0]
                        )
                        global_slices.append(
                            np.zeros((n_nodes, n_interactions), dtype=int)
                        )

                    active = np.zeros((n_nodes, n_interactions), dtype=bool)
                    for i, inds in enumerate(interact_inds):
                        for j, ind in enumerate(inds):
                            interaction_slices[i, j] = ind
                            active[i, j] = 1

                            for k, tail_block in enumerate(
                                interaction_group.tail_nodes
                            ):
                                s = gibbs_spec.node_global_location_map[
                                    tail_block.nodes[ind]
                                ][1]
                                global_slices[k][i, j] = s

                    interaction_slices = jnp.array(interaction_slices)

                    sliced_interaction = jax.tree.map(
                        lambda x: _tree_slice(
                            x, interaction_slices
                        ),  # shape -> (n, m, …)
                        interaction_group.interaction,
                    )

                    this_block_interactions.append(sliced_interaction)
                    this_block_active.append(jnp.array(active))
                    this_block_global_inds.append(global_inds)
                    this_block_global_slices.append(
                        [jnp.array(x) for x in global_slices]
                    )
            per_block_interactions.append(this_block_interactions)
            per_block_interaction_active.append(this_block_active)
            per_block_interaction_global_inds.append(this_block_global_inds)
            per_block_interaction_global_slices.append(this_block_global_slices)

        self.per_block_interactions = per_block_interactions
        self.per_block_interaction_active = per_block_interaction_active
        self.per_block_interaction_global_inds = per_block_interaction_global_inds
        self.per_block_interaction_global_slices = per_block_interaction_global_slices

        # Precompute scatter indices and output SDs per free block.
        block_sd_inds = []
        block_positions = []
        block_output_sds = []
        for block in gibbs_spec.free_blocks:
            sd_ind, positions = get_node_locations(block, gibbs_spec)
            block_sd_inds.append(sd_ind)
            block_positions.append(positions)
            template_sd = gibbs_spec.node_shape_struct[block.node_type]
            block_output_sds.append(_build_output_sd(block, template_sd))
        self._block_sd_inds = block_sd_inds
        self._block_positions = block_positions
        self._block_output_sds = block_output_sds


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

    def body_fn(carry, _key):
        state_free, sampler_state, global_state = carry

        keys = jax.random.split(_key, len(program.gibbs_spec.free_blocks))

        for sampling_group in program.gibbs_spec.sampling_order:
            # Collect all updates for this group before writing back.
            new_states = {}
            new_sampler_states = {}
            for i in sampling_group:
                new_states[i], new_sampler_states[i] = sample_single_block(
                    keys[i],
                    state_free,
                    state_clamp,
                    program,
                    i,
                    sampler_state[i],
                    global_state,
                    per_block_interactions=pbi,
                )

            # Apply updates functionally.
            state_free = [
                new_states[i] if i in new_states else state_free[i]
                for i in range(len(state_free))
            ]
            sampler_state = [
                new_sampler_states[i] if i in new_sampler_states else sampler_state[i]
                for i in range(len(sampler_state))
            ]
            for i in new_states:
                sd_ind = block_sd_inds[i]
                positions = block_positions[i]
                new_global = list(global_state)
                new_global[sd_ind] = jax.tree.map(
                    lambda g, s: g.at[positions].set(s),
                    global_state[sd_ind],
                    new_states[i],
                )
                global_state = new_global

        return (state_free, sampler_state, global_state), None

    keys = jax.random.split(key, n_iters)
    (final_state_free, final_sampler_states, final_global), _ = jax.lax.scan(
        body_fn, (init_chain_state, sampler_states, init_global_state), keys
    )
    return final_state_free, final_sampler_states, final_global


@dataclasses.dataclass
class SamplingSchedule:
    """
    Represents a sampling schedule for a process.

    **Attributes:**

    - `n_warmup`: The number of warmup steps to run before collecting samples.
    - `n_samples`: The number of samples to collect.
    - `steps_per_sample`: The number of steps to run between each sample.
    """

    n_warmup: int
    n_samples: int
    steps_per_sample: int

    def __hash__(self) -> int:
        return hash((self.n_warmup, self.n_samples, self.steps_per_sample))


def sample_with_observation(
    key: Key[Array, ""],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_chain_state: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    observation_carry_init: ObserveCarry,
    f_observe: AbstractObserver,
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

    **Returns:**

    - Tuple `(final_observer_carry, samples)` where `samples` is a PyTree whose
        leading axis has size `schedule.n_samples`.
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
) -> list[PyTree[Shaped[Array, "n_samples nodes ?*state"]]]:
    """Convenience wrapper to collect state information for *nodes_to_sample* only.

    Internally builds a [`hamon.StateObserver`][], runs
    [`hamon.sample_with_observation`][], and returns a stacked tensor of shape
    `(schedule.n_samples, ...)`.
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
    )

    return results_out
