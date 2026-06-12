"""Comprehensive backend tests for hamon.

Targets gaps identified in the architecture audit:
- Roundtrip fidelity: block → global → block for heterogeneous pytree states
- Scatter correctness: scatter_block_to_global matches full rebuild
- _hash_pytree edge cases
- get_node_locations / make_empty_block_state
- sample_single_block with/without precomputed global_state
- _run_blocks edge cases (0 iters, global_state consistency)
- SamplingSchedule edge cases (1 sample, 0 warmup)
- Observer JIT + accumulation through sampling
- IsingEBM.factors caching: identity check, energy determinism
- verify_block_state, Block edge cases, InteractionGroup validation
- BlockGibbsSpec superblock ordering, BlockSamplingProgram validation
- Duplicate node detection, energy fast path, empty clamped blocks
- MomentAccumulatorObserver dedup correctness
"""

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from hamon.block_management import (
    Block,
    BlockSpec,
    block_state_to_global,
    from_global_state,
    get_node_locations,
    make_empty_block_state,
    scatter_block_to_global,
    verify_block_state,
    _hash_pytree,
)
from hamon.block_sampling import (
    BlockGibbsSpec,
    BlockSamplingProgram,
    SamplingSchedule,
    _run_blocks,
    sample_single_block,
    sample_states,
    sample_with_observation,
)
from hamon.conditional_samplers import (
    AbstractConditionalSampler,
)
from hamon.interaction import InteractionGroup
from hamon.observers import MomentAccumulatorObserver, StateObserver
from hamon.pgm import AbstractNode, CategoricalNode, SpinNode


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class CompoundState(eqx.Module):
    data: Array
    label: int  # static leaf


class CompoundNode(AbstractNode):
    pass


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


def _make_simple_program():
    """Single-block SpinNode program with self-interaction."""
    nodes = [SpinNode() for _ in range(4)]
    block = Block(nodes)
    sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
    spec = BlockGibbsSpec([block], [], sd)
    ig = InteractionGroup(jnp.ones(4), block, [block])
    prog = BlockSamplingProgram(spec, [PassthroughSampler()], [ig])
    state = [jnp.zeros(4, dtype=jnp.bool_)]
    return prog, state, block


# ---------------------------------------------------------------------------
# 1. Roundtrip fidelity
# ---------------------------------------------------------------------------


class TestRoundtripFidelity(unittest.TestCase):
    def test_single_type_scalar(self):
        nodes_a = [SpinNode() for _ in range(5)]
        nodes_b = [SpinNode() for _ in range(3)]
        blocks = [Block(nodes_a), Block(nodes_b)]
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        spec = BlockSpec(blocks, sd)

        state = [
            jnp.array([True, False, True, False, True]),
            jnp.array([False, True, False]),
        ]
        gs = block_state_to_global(state, spec)
        recovered = from_global_state(gs, spec, blocks)
        for orig, rec in zip(state, recovered):
            self.assertTrue(jnp.array_equal(orig, rec))

    def test_multi_type(self):
        spin_nodes = [SpinNode() for _ in range(4)]
        cat_nodes = [CategoricalNode() for _ in range(3)]
        blocks = [Block(spin_nodes), Block(cat_nodes)]
        sd = {
            SpinNode: jax.ShapeDtypeStruct((), jnp.bool_),
            CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint8),
        }
        spec = BlockSpec(blocks, sd)
        state = [
            jnp.array([True, True, False, False]),
            jnp.array([0, 2, 1], dtype=jnp.uint8),
        ]
        gs = block_state_to_global(state, spec)
        recovered = from_global_state(gs, spec, blocks)
        for orig, rec in zip(state, recovered):
            self.assertTrue(jnp.array_equal(orig, rec))

    def test_subset_extraction(self):
        nodes_a = [SpinNode() for _ in range(3)]
        nodes_b = [SpinNode() for _ in range(2)]
        blocks = [Block(nodes_a), Block(nodes_b)]
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        spec = BlockSpec(blocks, sd)
        state = [jnp.array([True, False, True]), jnp.array([False, True])]
        gs = block_state_to_global(state, spec)
        recovered = from_global_state(gs, spec, [blocks[1]])
        self.assertTrue(jnp.array_equal(recovered[0], state[1]))


# ---------------------------------------------------------------------------
# 2. Scatter correctness
# ---------------------------------------------------------------------------


class TestScatterCorrectness(unittest.TestCase):
    def _setup(self):
        nodes_a = [SpinNode() for _ in range(4)]
        nodes_b = [SpinNode() for _ in range(3)]
        blocks = [Block(nodes_a), Block(nodes_b)]
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        spec = BlockSpec(blocks, sd)
        return blocks, spec

    def test_scatter_matches_full_rebuild(self):
        blocks, spec = self._setup()
        state = [jnp.array([True, False, True, False]), jnp.array([True, True, True])]
        gs = block_state_to_global(state, spec)

        new_b1 = jnp.array([False, False, False])
        gs_scattered = scatter_block_to_global(gs, new_b1, blocks[1], spec)
        gs_rebuilt = block_state_to_global([state[0], new_b1], spec)

        for s, r in zip(gs_scattered, gs_rebuilt):
            if s is not None and r is not None:
                self.assertTrue(jnp.array_equal(s, r))

    def test_scatter_preserves_unmodified(self):
        blocks, spec = self._setup()
        state = [jnp.array([True, False, True, False]), jnp.array([True, True, True])]
        gs = block_state_to_global(state, spec)
        gs_new = scatter_block_to_global(
            gs, jnp.array([False, False, False]), blocks[1], spec
        )
        recovered = from_global_state(gs_new, spec, [blocks[0]])
        self.assertTrue(jnp.array_equal(recovered[0], state[0]))

    def test_scatter_heterogeneous(self):
        spin_nodes = [SpinNode() for _ in range(3)]
        cat_nodes = [CategoricalNode() for _ in range(2)]
        blocks = [Block(spin_nodes), Block(cat_nodes)]
        sd = {
            SpinNode: jax.ShapeDtypeStruct((), jnp.bool_),
            CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint8),
        }
        spec = BlockSpec(blocks, sd)
        state = [jnp.array([True, False, True]), jnp.array([1, 2], dtype=jnp.uint8)]
        gs = block_state_to_global(state, spec)

        new_cat = jnp.array([0, 0], dtype=jnp.uint8)
        gs_new = scatter_block_to_global(gs, new_cat, blocks[1], spec)
        self.assertTrue(
            jnp.array_equal(from_global_state(gs_new, spec, [blocks[0]])[0], state[0])
        )
        self.assertTrue(
            jnp.array_equal(from_global_state(gs_new, spec, [blocks[1]])[0], new_cat)
        )


# ---------------------------------------------------------------------------
# 3. _hash_pytree
# ---------------------------------------------------------------------------


class TestHashPytree(unittest.TestCase):
    def test_identical_equal(self):
        a = jax.ShapeDtypeStruct((), jnp.float32)
        b = jax.ShapeDtypeStruct((), jnp.float32)
        self.assertEqual(_hash_pytree(a), _hash_pytree(b))

    def test_different_dtype(self):
        a = jax.ShapeDtypeStruct((), jnp.float32)
        b = jax.ShapeDtypeStruct((), jnp.float64)
        self.assertNotEqual(_hash_pytree(a), _hash_pytree(b))

    def test_different_shape(self):
        self.assertNotEqual(
            _hash_pytree(jax.ShapeDtypeStruct((3,), jnp.float32)),
            _hash_pytree(jax.ShapeDtypeStruct((4,), jnp.float32)),
        )

    def test_nested(self):
        a = [jax.ShapeDtypeStruct((), jnp.bool_), jax.ShapeDtypeStruct((2,), jnp.int8)]
        b = [jax.ShapeDtypeStruct((), jnp.bool_), jax.ShapeDtypeStruct((2,), jnp.int8)]
        self.assertEqual(_hash_pytree(a), _hash_pytree(b))

    def test_eqx_module(self):
        sd = CompoundState(jax.ShapeDtypeStruct((4,), jnp.float32), 42)
        self.assertEqual(_hash_pytree(sd), _hash_pytree(sd))


# ---------------------------------------------------------------------------
# 4. get_node_locations
# ---------------------------------------------------------------------------


class TestGetNodeLocations(unittest.TestCase):
    def test_unique_positions(self):
        nodes_a = [SpinNode() for _ in range(5)]
        nodes_b = [SpinNode() for _ in range(3)]
        blocks = [Block(nodes_a), Block(nodes_b)]
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        spec = BlockSpec(blocks, sd)
        all_pos = set()
        for block in blocks:
            sd_ind, positions = get_node_locations(block, spec)
            for p in positions.tolist():
                self.assertNotIn((sd_ind, p), all_pos)
                all_pos.add((sd_ind, p))
        self.assertEqual(len(all_pos), 8)

    def test_match_manual_lookup(self):
        nodes = [SpinNode() for _ in range(4)]
        blocks = [Block(nodes[:2]), Block(nodes[2:])]
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        spec = BlockSpec(blocks, sd)
        for block in blocks:
            sd_ind, positions = get_node_locations(block, spec)
            for j, node in enumerate(block.nodes):
                exp_sd, exp_pos = spec.node_global_location_map[node]
                self.assertEqual(sd_ind, exp_sd)
                self.assertEqual(positions[j].item(), exp_pos)


# ---------------------------------------------------------------------------
# 5. make_empty_block_state
# ---------------------------------------------------------------------------


class TestMakeEmptyBlockState(unittest.TestCase):
    def test_shapes(self):
        nodes = [SpinNode() for _ in range(5)]
        sd = {SpinNode: jax.ShapeDtypeStruct((3,), jnp.float32)}
        state = make_empty_block_state([Block(nodes)], sd)
        self.assertEqual(state[0].shape, (5, 3))

    def test_batch_shape(self):
        nodes = [SpinNode() for _ in range(4)]
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        state = make_empty_block_state([Block(nodes)], sd, batch_shape=(10, 2))
        self.assertEqual(state[0].shape, (10, 2, 4))

    def test_all_zeros(self):
        nodes = [CategoricalNode() for _ in range(3)]
        sd = {CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint8)}
        state = make_empty_block_state([Block(nodes)], sd)
        self.assertTrue(jnp.all(state[0] == 0))


# ---------------------------------------------------------------------------
# 6. sample_single_block: precomputed global_state fast path
# ---------------------------------------------------------------------------


class TestSampleSingleBlockGlobalState(unittest.TestCase):
    def test_with_vs_without_global_state(self):
        prog, state, _ = _make_simple_program()
        key = jax.random.key(42)
        out_no_gs, _ = sample_single_block(key, state, [], prog, 0, None)
        gs = block_state_to_global(state, prog.gibbs_spec)
        out_with_gs, _ = sample_single_block(
            key, state, [], prog, 0, None, global_state=gs
        )
        self.assertTrue(jnp.array_equal(out_no_gs, out_with_gs))


# ---------------------------------------------------------------------------
# 7. _run_blocks edge cases
# ---------------------------------------------------------------------------


class TestRunBlocksEdgeCases(unittest.TestCase):
    def test_zero_iters(self):
        prog, state, _ = _make_simple_program()
        out_state, _, _ = _run_blocks(jax.random.key(0), prog, state, [], 0, [None])
        self.assertTrue(jnp.array_equal(out_state[0], state[0]))

    def test_global_state_consistent(self):
        prog, state, _ = _make_simple_program()
        out_state, _, out_gs = _run_blocks(
            jax.random.key(0), prog, state, [], 5, [None]
        )
        rebuilt = block_state_to_global(out_state, prog.gibbs_spec)
        for a, b in zip(out_gs, rebuilt):
            self.assertTrue(jnp.array_equal(a, b))


# ---------------------------------------------------------------------------
# 8. SamplingSchedule edge cases
# ---------------------------------------------------------------------------


class TestSamplingScheduleEdgeCases(unittest.TestCase):
    def test_single_sample(self):
        prog, state, block = _make_simple_program()
        schedule = SamplingSchedule(n_warmup=1, n_samples=1, steps_per_sample=1)
        samples = sample_states(jax.random.key(0), prog, schedule, state, [], [block])
        self.assertEqual(samples[0].shape[0], 1)

    def test_zero_warmup(self):
        prog, state, block = _make_simple_program()
        schedule = SamplingSchedule(n_warmup=0, n_samples=3, steps_per_sample=1)
        samples = sample_states(jax.random.key(0), prog, schedule, state, [], [block])
        self.assertEqual(samples[0].shape[0], 3)


# ---------------------------------------------------------------------------
# 9. Observer JIT + accumulation
# ---------------------------------------------------------------------------


class TestStateObserverJIT(unittest.TestCase):
    def test_through_sample_with_observation(self):
        prog, state, block = _make_simple_program()
        observer = StateObserver([block])
        schedule = SamplingSchedule(n_warmup=2, n_samples=3, steps_per_sample=1)
        _, samples = sample_with_observation(
            jax.random.key(0), prog, schedule, state, [], observer.init(), observer
        )
        self.assertEqual(samples[0].shape, (3, 4))


class TestMomentAccumulatorMultiStep(unittest.TestCase):
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


# ---------------------------------------------------------------------------
# 10. MomentAccumulatorObserver dedup correctness
# ---------------------------------------------------------------------------


class TestMomentAccumulatorDedup(unittest.TestCase):
    """Verify that the refactored __init__ correctly deduplicates nodes."""

    def test_blocks_to_sample_no_duplicates(self):
        """A node appearing in multiple moments should appear only once in blocks_to_sample."""
        n1, n2 = SpinNode(), SpinNode()
        # n1 appears in both moment types
        observer = MomentAccumulatorObserver([[(n1,), (n2,)], [(n1, n2)]])
        total_nodes_in_blocks = sum(len(b) for b in observer.blocks_to_sample)
        self.assertEqual(
            total_nodes_in_blocks, 2, "Each node should appear exactly once"
        )

    def test_flat_scatter_index_is_permutation(self):
        """After dedup, _flat_scatter_index should contain each index exactly once."""
        n1, n2, n3 = SpinNode(), SpinNode(), SpinNode()
        observer = MomentAccumulatorObserver([[(n1, n2)], [(n2, n3)]])
        idx = observer._flat_scatter_index
        self.assertEqual(len(idx), 3)
        self.assertEqual(
            len(set(idx.tolist())), 3, "Scatter index must be a permutation"
        )

    def test_flat_value_order_inverts_scatter(self):
        """_flat_value_order should be argsort(_flat_scatter_index)."""
        n1, n2 = SpinNode(), CategoricalNode()
        observer = MomentAccumulatorObserver([[(n1, n2)]])
        expected = jnp.argsort(observer._flat_scatter_index)
        self.assertTrue(jnp.array_equal(observer._flat_value_order, expected))


# ---------------------------------------------------------------------------
# 11. IsingEBM: cached factors
# ---------------------------------------------------------------------------


class TestIsingEBMFactors(unittest.TestCase):
    def _make_model(self):
        from hamon.models.ising import IsingEBM

        nodes = [SpinNode() for _ in range(5)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
        return IsingEBM(nodes, edges, jnp.zeros(5), jnp.ones(4) * 0.5, jnp.array(1.0))

    def test_factors_structurally_consistent(self):
        """Repeated .factors access returns structurally equivalent objects."""
        model = self._make_model()
        f1 = model.factors
        f2 = model.factors
        self.assertEqual(len(f1), len(f2))
        for a, b in zip(f1, f2):
            self.assertEqual(type(a), type(b))

    def test_energy_deterministic(self):
        model = self._make_model()
        blocks = [Block(model.nodes)]
        state = [jnp.array([True, False, True, False, True])]
        e1 = model.energy(state, blocks)
        e2 = model.energy(state, blocks)
        self.assertTrue(jnp.allclose(e1, e2))

    def test_factors_weights_include_beta_scaling(self):
        """factors property should return beta * weights / beta * biases."""
        from hamon.models.ising import IsingEBM

        nodes = [SpinNode() for _ in range(3)]
        edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
        biases = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.array([0.5, 0.7])
        beta = jnp.array(2.0)
        model = IsingEBM(nodes, edges, biases, weights, beta)
        self.assertTrue(jnp.allclose(model.factors[0].weights, beta * biases))
        self.assertTrue(jnp.allclose(model.factors[1].weights, beta * weights))

    def test_factors_not_cached_for_ad(self):
        """factors must NOT be cached — AD needs tracer propagation through beta*weights."""
        model = self._make_model()
        # Two accesses should produce different list objects (not cached)
        f1 = model.factors
        f2 = model.factors
        self.assertIsNot(f1, f2)


# ---------------------------------------------------------------------------
# 12. verify_block_state
# ---------------------------------------------------------------------------


class TestVerifyBlockState(unittest.TestCase):
    def test_valid_passes(self):
        nodes = [SpinNode() for _ in range(3)]
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        verify_block_state(
            [Block(nodes)], [jnp.array([True, False, True])], sd, block_axis=-1
        )

    def test_wrong_dtype_raises(self):
        nodes = [SpinNode() for _ in range(3)]
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        with self.assertRaises(RuntimeError):
            verify_block_state(
                [Block(nodes)], [jnp.ones(3, jnp.float32)], sd, block_axis=-1
            )

    def test_wrong_length_raises(self):
        nodes = [SpinNode() for _ in range(3)]
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        with self.assertRaises(RuntimeError):
            verify_block_state(
                [Block(nodes)], [jnp.array([True, False])], sd, block_axis=-1
            )


# ---------------------------------------------------------------------------
# 13. Block edge cases
# ---------------------------------------------------------------------------


class TestBlockEdgeCases(unittest.TestCase):
    def test_mixed_types_raise(self):
        with self.assertRaises(ValueError):
            Block([SpinNode(), CategoricalNode()])

    def test_add_same_type(self):
        self.assertEqual(len(Block([SpinNode(), SpinNode()]) + Block([SpinNode()])), 3)

    def test_add_different_type_raises(self):
        with self.assertRaises(ValueError):
            Block([SpinNode()]) + Block([CategoricalNode()])

    def test_contains(self):
        n = SpinNode()
        self.assertIn(n, Block([n, SpinNode()]))


# ---------------------------------------------------------------------------
# 14. InteractionGroup validation
# ---------------------------------------------------------------------------


class TestInteractionGroupValidation(unittest.TestCase):
    def test_mismatched_tail_length(self):
        head = Block([SpinNode() for _ in range(3)])
        tail = Block([SpinNode() for _ in range(2)])
        with self.assertRaises(RuntimeError):
            InteractionGroup(jnp.ones(3), head, [tail])

    def test_mismatched_interaction_dim(self):
        head = Block([SpinNode() for _ in range(3)])
        tail = Block([SpinNode() for _ in range(3)])
        with self.assertRaises(RuntimeError):
            InteractionGroup(jnp.ones(5), head, [tail])


# ---------------------------------------------------------------------------
# 15. BlockGibbsSpec superblocks
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 16. BlockSamplingProgram validation
# ---------------------------------------------------------------------------


class TestBlockSamplingProgramValidation(unittest.TestCase):
    def test_wrong_sampler_count(self):
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        b1 = Block([SpinNode(), SpinNode()])
        b2 = Block([SpinNode(), SpinNode(), SpinNode()])
        spec = BlockGibbsSpec([b1, b2], [], sd)
        ig = InteractionGroup(jnp.ones(2), b1, [b1])
        with self.assertRaisesRegex(ValueError, "Expected 2 samplers"):
            BlockSamplingProgram(spec, [PassthroughSampler()], [ig])


# ---------------------------------------------------------------------------
# 17. Duplicate node detection
# ---------------------------------------------------------------------------


class TestDuplicateNodeDetection(unittest.TestCase):
    def test_same_node_in_two_blocks(self):
        shared = SpinNode()
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        with self.assertRaises(RuntimeError):
            BlockSpec([Block([shared, SpinNode()]), Block([shared, SpinNode()])], sd)


# ---------------------------------------------------------------------------
# 18. Energy fast path
# ---------------------------------------------------------------------------


class TestEnergyFastPath(unittest.TestCase):
    def test_blockspec_vs_list(self):
        from hamon.models.ising import IsingEBM

        nodes = [SpinNode() for _ in range(6)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(5)]
        model = IsingEBM(nodes, edges, jnp.ones(6), jnp.ones(5) * 0.3, jnp.array(2.0))
        blocks = [Block(nodes)]
        state = [jax.random.bernoulli(jax.random.key(99), 0.5, (6,))]
        e_list = model.energy(state, blocks)
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        e_spec = model.energy(state, BlockSpec(blocks, sd))
        self.assertTrue(jnp.allclose(e_list, e_spec))

    def test_jit_compatible(self):
        from hamon.models.ising import IsingEBM

        nodes = [SpinNode() for _ in range(4)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(3)]
        model = IsingEBM(nodes, edges, jnp.zeros(4), jnp.ones(3), jnp.array(1.0))
        sd = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        bs = BlockSpec([Block(nodes)], sd)
        state = [jnp.array([True, True, False, True])]
        e_eager = model.energy(state, bs)
        e_jit = eqx.filter_jit(model.energy)(state, bs)
        self.assertTrue(jnp.allclose(e_eager, e_jit))


# ---------------------------------------------------------------------------
# 19. Empty clamped blocks
# ---------------------------------------------------------------------------


class TestEmptyClampedBlocks(unittest.TestCase):
    def test_sampling_no_clamped(self):
        prog, state, block = _make_simple_program()
        schedule = SamplingSchedule(n_warmup=2, n_samples=3, steps_per_sample=1)
        samples = sample_states(jax.random.key(0), prog, schedule, state, [], [block])
        self.assertEqual(samples[0].shape, (3, 4))


# ---------------------------------------------------------------------------
# 20. Precomputed _block_output_sds matches on-the-fly computation
# ---------------------------------------------------------------------------


class TestPrecomputedOutputSDs(unittest.TestCase):
    def test_matches_runtime_computation(self):
        """_block_output_sds should match what _resize_sd would produce at call time."""
        prog, _, _ = _make_simple_program()
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


if __name__ == "__main__":
    unittest.main()
