"""Tests for block_sampling.py.

All original tests are preserved. Added:
- TestRunBlocksGlobalState: _run_blocks returns a 3-tuple; the third element
  (global_state) is consistent with block_state_to_global on the final state.
- TestPerBlockInteractionsOverride: passing per_block_interactions to
  _run_blocks / sample_single_block changes the output in the expected way.
"""

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key, PyTree

from hamon.block_management import Block, block_state_to_global
from hamon.block_sampling import (
    BlockGibbsSpec,
    BlockSamplingProgram,
    SamplingSchedule,
    _run_blocks,
    sample_blocks,
    sample_single_block,
    sample_states,
)
from hamon.conditional_samplers import (
    AbstractConditionalSampler,
    _SamplerState,
    _State,
)
from hamon.interaction import InteractionGroup
from hamon.pgm import AbstractNode


class ContinousScalarNode(AbstractNode):
    pass


class PlusInteraction(eqx.Module):
    multiplier: Array


class MinusInteraction(eqx.Module):
    multiplier: Array


class MemoryInteraction(eqx.Module):
    multiplier: Array


class PlusMinusSampler(AbstractConditionalSampler):
    def sample(
        self,
        key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: jax.ShapeDtypeStruct,
    ):
        output = jnp.zeros(output_sd.shape, dtype=output_sd.dtype)
        for interaction, active, state in zip(interactions, active_flags, states):
            active = active.astype(interaction.multiplier.dtype)
            s = state[0].astype(interaction.multiplier.dtype)
            if isinstance(interaction, (PlusInteraction, MemoryInteraction)):
                output += jnp.sum(interaction.multiplier * active * s, axis=-1)
            elif isinstance(interaction, MinusInteraction):
                output -= jnp.sum(interaction.multiplier * active * s, axis=-1)
            else:
                raise RuntimeError("Invalid interaction passed to PlusMinusSampler")
        return output, sampler_state

    def init(self) -> _SamplerState:
        return None


class TestPlusMinus(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        key = jax.random.key(424)

        free_nodes = [ContinousScalarNode() for _ in range(3)]
        minus_nodes = [ContinousScalarNode() for _ in range(2)]
        plus_nodes = [ContinousScalarNode() for _ in range(2)]

        key, subkey = jax.random.split(key, 2)
        self.minus_weights = jax.random.uniform(subkey, (3,), minval=0, maxval=1)
        key, subkey = jax.random.split(key, 2)
        self.plus_weights = jax.random.uniform(subkey, (3,), minval=0, maxval=1)

        minus_interaction_group = InteractionGroup(
            MinusInteraction(self.minus_weights),
            Block([free_nodes[0], free_nodes[0], free_nodes[1]]),
            [Block([minus_nodes[0], minus_nodes[1], minus_nodes[1]])],
        )
        plus_interaction_group = InteractionGroup(
            PlusInteraction(self.plus_weights),
            Block([free_nodes[1], free_nodes[2], free_nodes[2]]),
            [Block([plus_nodes[0], plus_nodes[0], plus_nodes[1]])],
        )
        memory_interaction_group = InteractionGroup(
            MemoryInteraction(jnp.ones(len(free_nodes))),
            Block(free_nodes),
            [Block(free_nodes)],
        )

        block_spec = BlockGibbsSpec(
            [Block([free_nodes[0]]), Block(free_nodes[1:])],
            [Block(plus_nodes + minus_nodes)],
            {ContinousScalarNode: jax.ShapeDtypeStruct((), jnp.float32)},
        )

        self.program = BlockSamplingProgram(
            block_spec,
            [PlusMinusSampler(), PlusMinusSampler()],
            [minus_interaction_group, plus_interaction_group, memory_interaction_group],
        )

        keys = jax.random.split(key, 4)

        self.state_free = [
            jax.random.uniform(keys[0], (1,), minval=1.0, maxval=5.0),
            jax.random.uniform(keys[1], (2,), minval=1.0, maxval=5.0),
        ]
        self.state_clamped = [jax.random.uniform(keys[2], (4,), minval=1.0, maxval=5.0)]
        self.key = keys[-1]

    def test_sample_block(self):
        outputs = []
        for block in [0, 1]:
            outputs.append(
                sample_single_block(
                    self.key,
                    self.state_free,
                    self.state_clamped,
                    self.program,
                    block,
                    None,
                )[0]
            )

        first_output = self.state_free[0][0] - jnp.sum(self.minus_weights[:2] * self.state_clamped[0][2:])
        second_output = (
            self.state_free[1][0]
            - self.minus_weights[-1] * self.state_clamped[0][-1]
            + self.plus_weights[0] * self.state_clamped[0][0]
        )
        third_output = self.state_free[1][1] + jnp.sum(self.plus_weights[1:] * self.state_clamped[0][:2])

        self.assertTrue(np.allclose(outputs[0], [first_output], rtol=1e-6))
        self.assertTrue(np.allclose(outputs[1], [second_output, third_output], rtol=1e-6))

    def test_sample_blocks(self):
        sample_blocks(self.key, self.state_free, self.state_clamped, self.program, [None, None])

    def test_sample_states(self):
        schedule = SamplingSchedule(5, 5, 5)
        sample_states(
            self.key,
            self.program,
            schedule,
            self.state_free,
            self.state_clamped,
            self.program.gibbs_spec.free_blocks,
        )

    def test_state_gaurdrailing(self):
        wrong_state_free = [self.state_free[0], jnp.zeros((2,), dtype=jnp.bool)]
        wrong_state_clamped = [jnp.zeros((4,), dtype=jnp.bool)]

        with self.assertRaises(RuntimeError) as error:
            _ = sample_blocks(
                self.key,
                wrong_state_free,
                self.state_clamped,
                self.program,
                [None, None],
            )
        self.assertIn("type", str(error.exception))

        with self.assertRaises(RuntimeError) as error:
            _ = sample_blocks(
                self.key,
                self.state_free,
                wrong_state_clamped,
                self.program,
                [None, None],
            )
        self.assertIn("type", str(error.exception))


class TestSamplerValidation(unittest.TestCase):
    def test_mismatched_sampler_list_raises(self):
        block_a = Block([ContinousScalarNode()])
        block_b = Block([ContinousScalarNode()])
        node_shape_dtypes = {ContinousScalarNode: jax.ShapeDtypeStruct((), jnp.float32)}
        spec = BlockGibbsSpec([block_a, block_b], [], node_shape_dtypes)

        with self.assertRaisesRegex(ValueError, "Expected 2 samplers"):
            BlockSamplingProgram(spec, [PlusMinusSampler()], [])


class MultiNode(AbstractNode):
    pass


class MultiNodeState(eqx.Module):
    float_counter: Array
    cat_counter: Array


class IncrementSampler(AbstractConditionalSampler):
    def sample(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ):
        assert isinstance(output_sd, MultiNodeState)
        for interaction, active, state in zip(interactions, active_flags, states):
            if isinstance(interaction, PlusInteraction):
                return (
                    MultiNodeState(
                        state[0].float_counter[:, 0, :] + 1,
                        state[0].cat_counter[:, 0, :] + 1,
                    ),
                    sampler_state,
                )

    def init(self) -> _SamplerState:
        return None


class TestPyTreeState(unittest.TestCase):
    def test_pytree_state(self):
        n_float = 2
        n_cat = 4

        sd_map = {
            MultiNode: MultiNodeState(
                jax.ShapeDtypeStruct((n_float,), jnp.float32),
                jax.ShapeDtypeStruct((n_cat,), jnp.int8),
            )
        }

        nodes = [MultiNode() for _ in range(10)]
        key = jax.random.key(424)

        interaction_group = InteractionGroup(PlusInteraction(jnp.ones((len(nodes),))), Block(nodes), [Block(nodes)])
        spec = BlockGibbsSpec([Block(nodes)], [], sd_map)

        key, subkey = jax.random.split(key, 2)
        init_float = jax.random.normal(subkey, (len(nodes), n_float))
        key, subkey = jax.random.split(key, 2)
        init_cat = jax.random.randint(subkey, (len(nodes), n_cat), minval=-4, maxval=4)

        init_state = [MultiNodeState(init_float, init_cat)]
        prog = BlockSamplingProgram(spec, [IncrementSampler()], [interaction_group])

        res, _ = sample_single_block(key, init_state, [], prog, 0, None)

        self.assertTrue(jnp.allclose(init_state[0].cat_counter + 1, res.cat_counter))
        self.assertTrue(jnp.allclose(init_state[0].float_counter + 1, res.float_counter))


# ---------------------------------------------------------------------------
# New tests for _run_blocks global_state return and per_block_interactions
# ---------------------------------------------------------------------------


class TestRunBlocksGlobalState(unittest.TestCase):
    """_run_blocks now returns a 3-tuple (state, sampler_states, global_state).
    Verify that the returned global_state is consistent with reconstructing it
    manually from the returned free state.
    """

    def _make_simple_program(self):
        nodes = [ContinousScalarNode() for _ in range(4)]
        key = jax.random.key(1)
        weights = jax.random.normal(key, (len(nodes),))
        interaction = InteractionGroup(PlusInteraction(weights), Block(nodes), [Block(nodes)])
        spec = BlockGibbsSpec(
            [Block(nodes[:2]), Block(nodes[2:])],
            [],
            {ContinousScalarNode: jax.ShapeDtypeStruct((), jnp.float32)},
        )
        prog = BlockSamplingProgram(spec, [PlusMinusSampler(), PlusMinusSampler()], [interaction])
        init_state = [jnp.ones((2,), jnp.float32), jnp.ones((2,), jnp.float32)]
        return prog, init_state

    def test_returns_three_tuple(self):
        prog, init_state = self._make_simple_program()
        result = _run_blocks(
            jax.random.key(0),
            prog,
            init_state,
            [],
            n_iters=2,
            sampler_states=[None, None],
        )
        self.assertEqual(len(result), 3, "Expected _run_blocks to return a 3-tuple")

    def test_global_state_consistent_with_final_state(self):
        """The returned global_state should match block_state_to_global applied to the final free state."""
        prog, init_state = self._make_simple_program()
        # _run_blocks is an internal function that gets jitted when called from
        # within a jitted context. Call it directly here; it will be compiled
        # on first call anyway via equinox's implicit tracing.
        final_state, _, returned_global = _run_blocks(
            jax.random.key(0),
            prog,
            init_state,
            [],
            n_iters=3,
            sampler_states=[None, None],
        )

        expected_global = block_state_to_global(final_state, prog.gibbs_spec)

        self.assertEqual(len(returned_global), len(expected_global))
        for a, b in zip(returned_global, expected_global):
            self.assertTrue(
                jnp.allclose(a, b),
                "Returned global_state inconsistent with final state",
            )

    def test_zero_iters_returns_init_global(self):
        """n_iters=0 early-return path must also return a valid global_state."""
        prog, init_state = self._make_simple_program()
        final_state, _, returned_global = _run_blocks(
            jax.random.key(0),
            prog,
            init_state,
            [],
            n_iters=0,
            sampler_states=[None, None],
        )

        expected_global = block_state_to_global(init_state, prog.gibbs_spec)
        for a, b in zip(returned_global, expected_global):
            self.assertTrue(jnp.allclose(a, b))

    def test_block_slice_starts_precomputed(self):
        """BlockSpec lays free blocks out contiguously, so every free block
        gets a static slice start (the scatter fallback should be unused)."""
        prog, _ = self._make_simple_program()
        # Two 2-node blocks sharing one structural group → starts at 0 and 2.
        self.assertEqual(prog._block_slice_starts, [0, 2])


class TestPerBlockInteractionsOverride(unittest.TestCase):
    """Passing per_block_interactions to sample_single_block and _run_blocks
    should override the program's own interactions, changing the output."""

    def setUp(self):
        nodes = [ContinousScalarNode() for _ in range(2)]
        key = jax.random.key(9)

        self.weights_a = jax.random.normal(key, (len(nodes),)) + 5.0  # far from zero
        key, _ = jax.random.split(key)
        self.weights_b = -self.weights_a  # opposite sign

        int_a = InteractionGroup(PlusInteraction(self.weights_a), Block(nodes), [Block(nodes)])
        int_b = InteractionGroup(PlusInteraction(self.weights_b), Block(nodes), [Block(nodes)])

        spec = BlockGibbsSpec(
            [Block(nodes)],
            [],
            {ContinousScalarNode: jax.ShapeDtypeStruct((), jnp.float32)},
        )
        self.prog_a = BlockSamplingProgram(spec, [PlusMinusSampler()], [int_a])
        self.prog_b = BlockSamplingProgram(spec, [PlusMinusSampler()], [int_b])
        self.init_state = [jnp.ones((len(nodes),), jnp.float32)]
        self.key = jax.random.key(42)

    def test_override_changes_sample_single_block(self):
        """sample_single_block with per_block_interactions=prog_b's interactions
        should give the same result as running prog_b directly."""
        result_prog_b, _ = sample_single_block(self.key, self.init_state, [], self.prog_b, block=0, sampler_state=None)
        result_override, _ = sample_single_block(
            self.key,
            self.init_state,
            [],
            self.prog_a,
            block=0,
            sampler_state=None,
            per_block_interactions=self.prog_b.per_block_interactions,
        )

        self.assertTrue(jnp.allclose(result_prog_b, result_override))

    def test_override_differs_from_original(self):
        """The overridden result should differ from running prog_a."""
        result_prog_a, _ = sample_single_block(self.key, self.init_state, [], self.prog_a, block=0, sampler_state=None)
        result_override, _ = sample_single_block(
            self.key,
            self.init_state,
            [],
            self.prog_a,
            block=0,
            sampler_state=None,
            per_block_interactions=self.prog_b.per_block_interactions,
        )

        self.assertFalse(
            jnp.allclose(result_prog_a, result_override),
            "Expected different results for opposite-sign weights",
        )

    def test_override_in_run_blocks(self):
        """_run_blocks with per_block_interactions override gives same final
        state as running prog_b directly (same key, same n_iters)."""
        n_iters = 3
        ss = [None]

        state_b, _, _ = _run_blocks(self.key, self.prog_b, self.init_state, [], n_iters, ss)
        state_override, _, _ = _run_blocks(
            self.key,
            self.prog_a,
            self.init_state,
            [],
            n_iters,
            ss,
            per_block_interactions=self.prog_b.per_block_interactions,
        )

        for a, b in zip(state_b, state_override):
            self.assertTrue(jnp.allclose(a, b))
