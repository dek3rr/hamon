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
    sample_states_batched,
    sample_with_observation,
)
from hamon.conditional_samplers import (
    AbstractConditionalSampler,
    _SamplerState,
    _State,
)
from hamon.interaction import InteractionGroup
from hamon.observers import MomentAccumulatorObserver, StateObserver
from hamon.pgm import AbstractNode, SpinNode


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

        first_output = self.state_free[0][0] - jnp.sum(
            self.minus_weights[:2] * self.state_clamped[0][2:]
        )
        second_output = (
            self.state_free[1][0]
            - self.minus_weights[-1] * self.state_clamped[0][-1]
            + self.plus_weights[0] * self.state_clamped[0][0]
        )
        third_output = self.state_free[1][1] + jnp.sum(
            self.plus_weights[1:] * self.state_clamped[0][:2]
        )

        self.assertTrue(np.allclose(outputs[0], [first_output], rtol=1e-6))
        self.assertTrue(
            np.allclose(outputs[1], [second_output, third_output], rtol=1e-6)
        )

    def test_sample_blocks(self):
        sample_blocks(
            self.key, self.state_free, self.state_clamped, self.program, [None, None]
        )

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

        interaction_group = InteractionGroup(
            PlusInteraction(jnp.ones((len(nodes),))), Block(nodes), [Block(nodes)]
        )
        spec = BlockGibbsSpec([Block(nodes)], [], sd_map)

        key, subkey = jax.random.split(key, 2)
        init_float = jax.random.normal(subkey, (len(nodes), n_float))
        key, subkey = jax.random.split(key, 2)
        init_cat = jax.random.randint(subkey, (len(nodes), n_cat), minval=-4, maxval=4)

        init_state = [MultiNodeState(init_float, init_cat)]
        prog = BlockSamplingProgram(spec, [IncrementSampler()], [interaction_group])

        res, _ = sample_single_block(key, init_state, [], prog, 0, None)

        self.assertTrue(jnp.allclose(init_state[0].cat_counter + 1, res.cat_counter))
        self.assertTrue(
            jnp.allclose(init_state[0].float_counter + 1, res.float_counter)
        )


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
        interaction = InteractionGroup(
            PlusInteraction(weights), Block(nodes), [Block(nodes)]
        )
        spec = BlockGibbsSpec(
            [Block(nodes[:2]), Block(nodes[2:])],
            [],
            {ContinousScalarNode: jax.ShapeDtypeStruct((), jnp.float32)},
        )
        prog = BlockSamplingProgram(
            spec, [PlusMinusSampler(), PlusMinusSampler()], [interaction]
        )
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
        """Every free block gets a static slice start (the scatter fallback
        should be unused)."""
        prog, _ = self._make_simple_program()
        self.assertTrue(all(s is not None for s in prog._block_slice_starts))
        # This program is split-safe, so it uses the per-block layout: each
        # block is the sole occupant of its own slot and therefore starts at 0.
        self.assertEqual(prog._block_slice_starts, [0, 0])


class TestPerBlockInteractionsOverride(unittest.TestCase):
    """Passing per_block_interactions to sample_single_block and _run_blocks
    should override the program's own interactions, changing the output."""

    def setUp(self):
        nodes = [ContinousScalarNode() for _ in range(2)]
        key = jax.random.key(9)

        self.weights_a = jax.random.normal(key, (len(nodes),)) + 5.0  # far from zero
        key, _ = jax.random.split(key)
        self.weights_b = -self.weights_a  # opposite sign

        int_a = InteractionGroup(
            PlusInteraction(self.weights_a), Block(nodes), [Block(nodes)]
        )
        int_b = InteractionGroup(
            PlusInteraction(self.weights_b), Block(nodes), [Block(nodes)]
        )

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
        result_prog_b, _ = sample_single_block(
            self.key, self.init_state, [], self.prog_b, block=0, sampler_state=None
        )
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
        result_prog_a, _ = sample_single_block(
            self.key, self.init_state, [], self.prog_a, block=0, sampler_state=None
        )
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

        state_b, _, _ = _run_blocks(
            self.key, self.prog_b, self.init_state, [], n_iters, ss
        )
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


class TestSampleStatesBatched(unittest.TestCase):
    """sample_states_batched runs N independent chains under one vmap; each
    chain must match single-chain sample_states with the same per-chain key."""

    def _make_program(self):
        nodes = [ContinousScalarNode() for _ in range(4)]
        weights = jax.random.normal(jax.random.key(2), (len(nodes),))
        interaction = InteractionGroup(
            PlusInteraction(weights), Block(nodes), [Block(nodes)]
        )
        spec = BlockGibbsSpec(
            [Block(nodes[:2]), Block(nodes[2:])],
            [],
            {ContinousScalarNode: jax.ShapeDtypeStruct((), jnp.float32)},
        )
        prog = BlockSamplingProgram(
            spec, [PlusMinusSampler(), PlusMinusSampler()], [interaction]
        )
        return prog, nodes

    def test_matches_looped_single_chains(self):
        prog, nodes = self._make_program()
        obs = Block(nodes)
        schedule = SamplingSchedule(3, 4, 1)
        n_chains = 3
        key = jax.random.key(5)
        # Distinct per-chain initial states so the chains genuinely differ.
        inits = [
            [
                jax.random.normal(jax.random.fold_in(key, c), (2,)),
                jax.random.normal(jax.random.fold_in(key, 100 + c), (2,)),
            ]
            for c in range(n_chains)
        ]
        stacked = [jnp.stack(x) for x in zip(*inits)]

        batched = sample_states_batched(
            key, prog, schedule, stacked, [], [obs], device=None
        )[0]

        # sample_states_batched splits `key` the same way internally.
        keys = jax.random.split(key, n_chains)
        looped = jnp.stack(
            [
                sample_states(
                    keys[c], prog, schedule, inits[c], [], [obs], device=None
                )[0]
                for c in range(n_chains)
            ]
        )

        self.assertEqual(batched.shape, looped.shape)
        self.assertEqual(batched.shape[0], n_chains)
        self.assertTrue(jnp.array_equal(batched, looped))


# ---------------------------------------------------------------------------
# Passthrough single-block program: shared fixture for the fast-path / edge-case
# / observer-through-sampling tests below.
# ---------------------------------------------------------------------------


class PassthroughSampler(AbstractConditionalSampler):
    """Returns zeros matching output_sd."""

    def sample(self, key, interactions, active_flags, states, sampler_state, output_sd):
        if isinstance(output_sd, jax.ShapeDtypeStruct):
            return jnp.zeros(output_sd.shape, output_sd.dtype), sampler_state
        return jax.tree.map(
            lambda sd: (
                jnp.zeros(sd.shape, sd.dtype)
                if isinstance(sd, jax.ShapeDtypeStruct)
                else sd
            ),
            output_sd,
        ), sampler_state

    def init(self):
        return None


def _make_passthrough_program():
    """Single-block SpinNode program with self-interaction and a passthrough sampler."""
    nodes = [SpinNode() for _ in range(4)]
    block = Block(nodes)
    sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
    spec = BlockGibbsSpec([block], [], sd)
    ig = InteractionGroup(jnp.ones(4), block, [block])
    prog = BlockSamplingProgram(spec, [PassthroughSampler()], [ig])
    state = [jnp.zeros(4, dtype=jnp.bool_)]
    return prog, state, block


class TestSampleSingleBlockGlobalState(unittest.TestCase):
    def test_with_vs_without_global_state(self):
        prog, state, _ = _make_passthrough_program()
        key = jax.random.key(42)
        out_no_gs, _ = sample_single_block(key, state, [], prog, 0, None)
        gs = block_state_to_global(state, prog.gibbs_spec)
        out_with_gs, _ = sample_single_block(
            key, state, [], prog, 0, None, global_state=gs
        )
        self.assertTrue(jnp.array_equal(out_no_gs, out_with_gs))


class TestSamplingScheduleEdgeCases(unittest.TestCase):
    def test_single_sample(self):
        prog, state, block = _make_passthrough_program()
        schedule = SamplingSchedule(n_warmup=1, n_samples=1, steps_per_sample=1)
        samples = sample_states(jax.random.key(0), prog, schedule, state, [], [block])
        self.assertEqual(samples[0].shape[0], 1)

    def test_zero_warmup(self):
        prog, state, block = _make_passthrough_program()
        schedule = SamplingSchedule(n_warmup=0, n_samples=3, steps_per_sample=1)
        samples = sample_states(jax.random.key(0), prog, schedule, state, [], [block])
        self.assertEqual(samples[0].shape[0], 3)


class TestEmptyClampedBlocks(unittest.TestCase):
    def test_sampling_no_clamped(self):
        prog, state, block = _make_passthrough_program()
        schedule = SamplingSchedule(n_warmup=2, n_samples=3, steps_per_sample=1)
        samples = sample_states(jax.random.key(0), prog, schedule, state, [], [block])
        self.assertEqual(samples[0].shape, (3, 4))


class TestPrecomputedOutputSDs(unittest.TestCase):
    def test_matches_runtime_computation(self):
        """_block_output_sds should match what _resize_sd would produce at call time."""
        prog, _, _ = _make_passthrough_program()
        for i, block in enumerate(prog.gibbs_spec.free_blocks):
            template = prog.gibbs_spec.node_shape_struct[block.node_type]

            def _resize(leaf):
                if isinstance(leaf, jax.ShapeDtypeStruct):
                    return jax.ShapeDtypeStruct(
                        (len(block.nodes), *leaf.shape), leaf.dtype
                    )
                return leaf

            expected = jax.tree.map(_resize, template)
            actual = prog._block_output_sds[i]

            exp_leaves = jax.tree.leaves(expected)
            act_leaves = jax.tree.leaves(actual)
            self.assertEqual(len(exp_leaves), len(act_leaves))
            for e, a in zip(exp_leaves, act_leaves):
                if isinstance(e, jax.ShapeDtypeStruct):
                    self.assertEqual(e.shape, a.shape)
                    self.assertEqual(e.dtype, a.dtype)
                else:
                    self.assertEqual(e, a)


class TestFusedWeightBind(unittest.TestCase):
    """The weight-binding gather+mask runs under one fused jit; it must produce
    exactly what a per-block eager gather+mask would, and a with_ebm rebuild for
    the same graph must reuse the structure and land bit-identical weights."""

    def _ising(self, n=24, seed=0):
        import numpy as _np

        from hamon.models.ising import IsingEBM, IsingSamplingProgram, _ising_graph

        rng = _np.random.default_rng(seed)
        # heterogeneous degree so the blocks have several distinct shapes
        edges = _np.array(
            sorted(
                {(0, j) for j in range(1, n)}
                | {
                    (int(min(a, b)), int(max(a, b)))
                    for a, b in rng.integers(0, n, (n, 2))
                    if a != b
                }
            )
        )
        nodes, node_edges, fb = _ising_graph(n, edges)
        w = jnp.asarray(rng.normal(0, 1.0, len(edges)))
        ebm = IsingEBM(nodes, node_edges, jnp.zeros(n), w, jnp.array(1.3))
        return IsingSamplingProgram(ebm, fb, []), ebm, edges

    def test_bound_weights_match_eager_gather_and_mask(self):
        from hamon.block_sampling import (
            _STRUCTURE_CACHE,
            _structure_cache_key,
            _tree_slice,
        )

        prog, ebm, _ = self._ising()
        igs = [g for f in ebm.factors for g in f.to_interaction_groups()]
        struct = _STRUCTURE_CACHE[_structure_cache_key(prog.gibbs_spec, igs)]

        for b, block_recipe in enumerate(struct.weight_recipe):
            for g_pos, (g_idx, isl, active, _gi, _gs) in enumerate(block_recipe):
                sliced = jax.tree.map(
                    lambda x, _s=isl: _tree_slice(x, _s), igs[g_idx].interaction
                )

                def _premask(x, _m=active):
                    if eqx.is_array(x):
                        m = _m.astype(x.dtype)
                        return x * m.reshape(m.shape + (1,) * (x.ndim - 2))
                    return x

                expected = jax.tree.map(_premask, sliced)
                got = prog.per_block_interactions[b][g_pos]
                for e, a in zip(jax.tree.leaves(expected), jax.tree.leaves(got)):
                    self.assertTrue(np.array_equal(np.asarray(e), np.asarray(a)))
                    self.assertEqual(np.asarray(e).dtype, np.asarray(a).dtype)

    def test_rescaled_ebm_rebuilds_identically(self):
        # A fresh program for a beta-scaled ebm must produce the same bound
        # weights as the original scaled by the same factor — the fused bind is
        # a pure function of the interaction values.
        prog, ebm, edges = self._ising()
        from hamon.models.ising import IsingSamplingProgram

        scaled = ebm.with_beta(ebm.beta * 2.0)
        prog2 = IsingSamplingProgram(scaled, prog.gibbs_spec.free_blocks, [])
        scaled_any = False
        for blk1, blk2 in zip(
            prog.per_block_interactions, prog2.per_block_interactions
        ):
            for g1, g2 in zip(blk1, blk2):
                for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g2)):
                    a, b = np.asarray(a), np.asarray(b)
                    if np.issubdtype(a.dtype, np.floating):
                        np.testing.assert_allclose(2.0 * a, b, rtol=1e-6, atol=1e-6)
                        scaled_any = scaled_any or bool(a.any())
                    else:
                        np.testing.assert_array_equal(a, b)  # metadata, unscaled
        self.assertTrue(scaled_any, "expected at least one weight leaf to scale")


class TestBlockGibbsSpecSuperblocks(unittest.TestCase):
    def test_sequential_order(self):
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        b1 = Block([SpinNode(), SpinNode()])
        b2 = Block([SpinNode(), SpinNode(), SpinNode()])
        spec = BlockGibbsSpec([b1, b2], [], sd)
        self.assertEqual(spec.sampling_order, [[0], [1]])

    def test_parallel_order(self):
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        b1 = Block([SpinNode(), SpinNode()])
        b2 = Block([SpinNode(), SpinNode(), SpinNode()])
        spec = BlockGibbsSpec([(b1, b2)], [], sd)
        self.assertEqual(spec.sampling_order, [[0, 1]])

    def test_clamped_separate(self):
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        free = Block([SpinNode(), SpinNode()])
        clamped = Block([SpinNode()])
        spec = BlockGibbsSpec([free], [clamped], sd)
        self.assertEqual(len(spec.free_blocks), 1)
        self.assertEqual(len(spec.clamped_blocks), 1)
        self.assertEqual(len(spec.blocks), 2)


class TestStateObserverThroughSampling(unittest.TestCase):
    """StateObserver driven through sample_with_observation (loop integration)."""

    def test_through_sample_with_observation(self):
        prog, state, block = _make_passthrough_program()
        observer = StateObserver([block])
        schedule = SamplingSchedule(n_warmup=2, n_samples=3, steps_per_sample=1)
        _, samples = sample_with_observation(
            jax.random.key(0), prog, schedule, state, [], observer.init(), observer
        )
        self.assertEqual(samples[0].shape, (3, 4))


class TestMomentAccumulatorThroughSampling(unittest.TestCase):
    """MomentAccumulatorObserver driven through sample_with_observation."""

    def test_accumulation(self):
        node = SpinNode()
        block = Block([node])
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        spec = BlockGibbsSpec([block], [], sd)
        ig = InteractionGroup(jnp.ones(1), block, [block])
        prog = BlockSamplingProgram(spec, [PassthroughSampler()], [ig])

        def spin_transform(state, _):
            return [2 * x.astype(jnp.float32) - 1 for x in state]

        observer = MomentAccumulatorObserver([[(node,)]], f_transform=spin_transform)
        schedule = SamplingSchedule(n_warmup=0, n_samples=5, steps_per_sample=1)
        with jax.numpy_dtype_promotion("standard"):
            moments, _ = sample_with_observation(
                jax.random.key(0),
                prog,
                schedule,
                [jnp.array([False])],
                [],
                observer.init(),
                observer,
            )
        # PassthroughSampler → 0 (False) → transform → -1; 5 × -1 = -5
        self.assertAlmostEqual(float(moments[0][0]), -5.0, places=4)
