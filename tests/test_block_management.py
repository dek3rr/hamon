"""Tests for block_management.py.

All original tests are preserved. Added:
- TestBlockSpecOrdering: verifies global_sd_order is deterministic and
  insertion-order-preserving (the dict.fromkeys fix vs the old set comprehension).
"""

import unittest

import equinox as eqx
import hamon.pgm
import jax
import jax.numpy as jnp
from hamon import block_management
from hamon.block_management import (
    Block,
    BlockSpec,
    _hash_pytree,
    block_state_to_global,
    from_global_state,
    get_node_locations,
    make_empty_block_state,
    scatter_block_to_global,
)
from hamon.pgm import CategoricalNode, SpinNode
from jaxtyping import Array


class Node1(hamon.pgm.AbstractNode):
    pass


class Node2(hamon.pgm.AbstractNode):
    pass


class Node3(hamon.pgm.AbstractNode):
    pass


class TestBlocks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng_key = jax.random.key(424)
        self.blocks = [
            block_management.Block([Node1() for _ in range(5)]),
            block_management.Block([Node2() for _ in range(9)]),
            block_management.Block([Node2() for _ in range(7)]),
            block_management.Block([Node3() for _ in range(3)]),
        ]

        bool_type = jax.ShapeDtypeStruct((2,), dtype=jnp.bool_)
        float_type = jax.ShapeDtypeStruct((5,), dtype=jnp.float32)
        int_type = jax.ShapeDtypeStruct((2,), dtype=jnp.int16)

        self.node_types = {
            Node1: [bool_type],
            Node2: [float_type, bool_type],
            Node3: [int_type, bool_type],
        }

        self.node_types1 = {
            Node1: [bool_type],
            Node2: [float_type, bool_type],
            Node3: [int_type, float_type],
        }

        self.node_types2 = {
            Node1: [int_type],
            Node2: [int_type],
            Node3: [int_type],
        }

        class CustomObj(eqx.Module):
            a: jax.ShapeDtypeStruct = float_type

        self.node_types3 = {
            Node1: int_type,
            Node2: {"a": bool_type, "b": int_type},
            Node3: CustomObj(),
        }

        self.node_type_dicts = {
            "variation0": self.node_types,
            "variation1": self.node_types1,
            "variation2": self.node_types2,
            "variation3": self.node_types3,
        }

        self.configs = {}
        for label, node_dict in self.node_type_dicts.items():
            spec = block_management.BlockSpec(self.blocks, node_dict)
            all_types = [type_list for type_list in node_dict.values()]
            block_state = block_management.make_empty_block_state(
                self.blocks, node_dict
            )
            self.configs[label] = (spec, block_state, all_types)

    def test_shape_transforms(self):
        for label, (spec, block_state, _) in self.configs.items():
            with self.subTest(msg=f"Testing shape_transforms with {label}"):
                global_state = block_management.block_state_to_global(block_state, spec)
                re_block = block_management.from_global_state(
                    global_state, spec, spec.blocks
                )
                self.assertTrue(eqx.tree_equal(block_state, re_block))

    def test_node_lookup(self):
        for label, (spec, block_state, _) in self.configs.items():
            with self.subTest(msg=f"Testing node_lookup with {label}"):
                global_state = block_management.block_state_to_global(block_state, spec)
                for block, state in zip(spec.blocks, block_state):
                    type_inds, arr_inds = block_management.get_node_locations(
                        block, spec
                    )
                    vals = jax.tree.map(
                        lambda x, arr_inds=arr_inds: x[arr_inds],
                        global_state[type_inds],
                    )
                    self.assertTrue(eqx.tree_equal(vals, state))

    def test_empty_state(self):
        for label, (spec, block_state, _) in self.configs.items():
            with self.subTest(msg=f"Testing empty_state with {label}"):
                batch_shape = (10, 2)
                empty_state = block_management.make_empty_block_state(
                    spec.blocks, spec.node_shape_struct, batch_shape
                )
                empty_state = jax.tree.map(
                    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), empty_state
                )
                b_state = jax.tree.map(
                    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), block_state
                )
                eqx.tree_equal(empty_state, b_state)


class Template2(eqx.Module):
    scalar: int
    data: Array


class Template1(eqx.Module):
    temp_2: Template2
    data: Array
    scalar: float


class TestBlockCompat(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_shape = (4, 2, 10)

        temp_2_sd = Template2(1, jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.float32))
        self.temp_1_sd = Template1(
            temp_2_sd, jax.ShapeDtypeStruct(shape=(), dtype=jnp.int8), 4.3
        )
        self.temp_2_good = Template2(
            3, jnp.zeros((*self.batch_shape, 4), dtype=jnp.float32)
        )

        self.block_1 = block_management.Block([Node1() for _ in range(5)])
        self.block_2 = block_management.Block([Node2() for _ in range(3)])
        self.block_3 = block_management.Block([Node3() for _ in range(9)])

        self.blocks = [self.block_1, self.block_2, self.block_3]

        self.node_sd_map = {
            Node1: (
                jax.ShapeDtypeStruct(shape=(1, 2), dtype=jnp.bool),
                jax.ShapeDtypeStruct(shape=(), dtype=jnp.uint8),
            ),
            Node2: self.temp_1_sd,
            Node3: jax.ShapeDtypeStruct(shape=(7,), dtype=jnp.float32),
        }

        self.good_state_1 = (
            jnp.zeros((*self.batch_shape, 5, 1, 2), dtype=jnp.bool),
            jnp.zeros((*self.batch_shape, 5), dtype=jnp.uint8),
        )

        t2 = Template2(5, jnp.zeros((*self.batch_shape, 3, 4), dtype=jnp.float32))
        self.good_state_2 = Template1(
            t2,
            jnp.zeros(
                (
                    *self.batch_shape,
                    3,
                ),
                dtype=jnp.int8,
            ),
            19.9,
        )

        self.good_state_3 = jnp.zeros((*self.batch_shape, 9, 7), dtype=jnp.float32)

    def test_good(self):
        temp_1_good = Template1(
            self.temp_2_good, jnp.zeros(self.batch_shape, dtype=jnp.int8), 7.1
        )
        batch_shape = block_management._check_pytree_compat(self.temp_1_sd, temp_1_good)
        self.assertEqual(batch_shape, self.batch_shape)

    def test_bad_dtype(self):
        temp_1_bad = Template1(
            self.temp_2_good, jnp.zeros(self.batch_shape, dtype=jnp.float32), 10.2
        )
        with self.assertRaises(RuntimeError) as error:
            _ = block_management._check_pytree_compat(self.temp_1_sd, temp_1_bad)
        self.assertIn("type", str(error.exception))

    def test_bad_shape(self):
        temp_1_bad = Template1(
            self.temp_2_good, jnp.zeros((*self.batch_shape, 1), dtype=jnp.int8), 11.9
        )
        with self.assertRaises(RuntimeError) as error:
            _ = block_management._check_pytree_compat(self.temp_1_sd, temp_1_bad)
        self.assertIn("shape", str(error.exception))

    def test_missing_array(self):
        temp_1_bad = Template1(self.temp_2_good, 1.0, 11.9)
        with self.assertRaises(RuntimeError) as error:
            _ = block_management._check_pytree_compat(self.temp_1_sd, temp_1_bad)
        self.assertIn("missing", str(error.exception))

    def test_bad_structure(self):
        temp_1_bad = jnp.array(1.0)
        with self.assertRaises(RuntimeError) as error:
            _ = block_management._check_pytree_compat(self.temp_1_sd, temp_1_bad)
        self.assertIn("structure", str(error.exception))

    def test_good_state(self):
        block_management.verify_block_state(
            self.blocks,
            [self.good_state_1, self.good_state_2, self.good_state_3],
            self.node_sd_map,
            block_axis=-1,
        )

    def test_wrong_state_len(self):
        with self.assertRaises(RuntimeError) as error:
            block_management.verify_block_state(
                self.blocks,
                [self.good_state_1, self.good_state_2],
                self.node_sd_map,
                block_axis=-1,
            )
        self.assertIn("of states not equal", str(error.exception))

    def test_bad_block(self):
        bad_state = self.good_state_3.astype(jnp.bool)
        with self.assertRaises(RuntimeError) as error:
            block_management.verify_block_state(
                self.blocks,
                [self.good_state_1, self.good_state_2, bad_state],
                self.node_sd_map,
                block_axis=-1,
            )
        self.assertIn("type", str(error.exception))

    def test_length_mismatch(self):
        bad_state = jnp.zeros((*self.batch_shape, 4, 7), dtype=jnp.float32)
        with self.assertRaises(RuntimeError) as error:
            block_management.verify_block_state(
                self.blocks,
                [self.good_state_1, self.good_state_2, bad_state],
                self.node_sd_map,
                block_axis=-1,
            )
        self.assertIn("block length", str(error.exception))


class TestDuplicate(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.good_blocks = [
            block_management.Block([Node1() for _ in range(3)]) for _ in range(2)
        ]
        self.node_sd = {Node1: jax.ShapeDtypeStruct}

    def test_good(self):
        _ = block_management.BlockSpec(self.good_blocks, self.node_sd)

    def test_duplicate(self):
        with self.assertRaises(RuntimeError) as error:
            _ = block_management.BlockSpec(
                self.good_blocks + self.good_blocks, self.node_sd
            )
        self.assertIn("show up twice", str(error.exception))


class TestBlockSpecOrdering(unittest.TestCase):
    """Verify that BlockSpec.global_sd_order is deterministic and follows
    insertion order, not set-iteration order.

    The fix: replaced ``list({sd for sd in ...})`` with
    ``list(dict.fromkeys(...))`` in BlockSpec.__init__.  Python sets have
    non-deterministic iteration order across interpreter runs, which would
    make the global state layout irreproducible.
    """

    def _make_spec(self, node_types):
        blocks = [
            block_management.Block([Node1() for _ in range(2)]),
            block_management.Block([Node2() for _ in range(2)]),
            block_management.Block([Node3() for _ in range(2)]),
        ]
        return block_management.BlockSpec(blocks, node_types)

    def test_same_spec_same_order(self):
        """Building the same BlockSpec twice produces identical global_sd_order."""
        node_types = {
            Node1: jax.ShapeDtypeStruct((), jnp.bool_),
            Node2: jax.ShapeDtypeStruct((), jnp.float32),
            Node3: jax.ShapeDtypeStruct((), jnp.int8),
        }
        spec_a = self._make_spec(node_types)
        spec_b = self._make_spec(node_types)
        self.assertEqual(spec_a.global_sd_order, spec_b.global_sd_order)

    def test_shared_sd_deduplicated_stably(self):
        """When multiple node types share the same ShapeDtypeStruct, it should
        appear exactly once in global_sd_order, at the position of its first
        occurrence in the iteration order of node_shape_dtypes."""
        shared = jax.ShapeDtypeStruct((), jnp.bool_)
        node_types = {
            Node1: shared,
            Node2: shared,  # duplicate — same object
            Node3: jax.ShapeDtypeStruct((), jnp.float32),
        }
        spec = self._make_spec(node_types)

        # shared appears first (Node1 is first), float32 second
        self.assertEqual(len(spec.global_sd_order), 2)
        # The unique SD from Node1/Node2 (bool_) should come before float32
        # because Node1 is encountered first.
        from hamon.block_management import _hash_pytree

        hashed_order = spec.global_sd_order
        hashed_bool = _hash_pytree(shared)
        hashed_float = _hash_pytree(jax.ShapeDtypeStruct((), jnp.float32))
        self.assertEqual(hashed_order[0], hashed_bool)
        self.assertEqual(hashed_order[1], hashed_float)

    def test_roundtrip_is_stable(self):
        """block_state_to_global → from_global_state is stable across two
        independently constructed BlockSpecs with the same inputs."""
        node_types = {
            Node1: jax.ShapeDtypeStruct((), jnp.bool_),
            Node2: jax.ShapeDtypeStruct((), jnp.float32),
            Node3: jax.ShapeDtypeStruct((), jnp.int8),
        }
        blocks = [
            block_management.Block([Node1() for _ in range(3)]),
            block_management.Block([Node2() for _ in range(2)]),
            block_management.Block([Node3() for _ in range(4)]),
        ]

        block_state = block_management.make_empty_block_state(blocks, node_types)

        spec1 = block_management.BlockSpec(blocks, node_types)
        spec2 = block_management.BlockSpec(blocks, node_types)

        gs1 = block_management.block_state_to_global(block_state, spec1)
        gs2 = block_management.block_state_to_global(block_state, spec2)

        # Both global states should have the same structure and values
        self.assertEqual(len(gs1), len(gs2))
        for a, b in zip(gs1, gs2):
            self.assertTrue(jnp.array_equal(a, b))


class TestScatterBlockToGlobal(unittest.TestCase):
    """Both write-back paths (contiguous slice update and scatter fallback)
    must agree with a reference positional scatter."""

    def setUp(self):
        self.sd = {Node1: jax.ShapeDtypeStruct((), jnp.float32)}
        self.blocks = [
            block_management.Block([Node1() for _ in range(4)]),
            block_management.Block([Node1() for _ in range(3)]),
        ]
        self.spec = block_management.BlockSpec(self.blocks, self.sd)
        self.global_state = [jax.random.normal(jax.random.key(0), (7,))]

    def _reference(self, block, new_state):
        sd_ind, positions = block_management.get_node_locations(block, self.spec)
        return sd_ind, self.global_state[sd_ind].at[positions].set(new_state)

    def test_contiguous_block(self):
        """Blocks laid out by BlockSpec are contiguous → slice-update path."""
        new_state = jnp.arange(3, dtype=jnp.float32) + 100.0
        out = block_management.scatter_block_to_global(
            self.global_state, new_state, self.blocks[1], self.spec
        )
        sd_ind, ref = self._reference(self.blocks[1], new_state)
        self.assertTrue(jnp.array_equal(out[sd_ind], ref))
        # positions outside the block are untouched
        self.assertTrue(jnp.array_equal(out[sd_ind][:4], self.global_state[sd_ind][:4]))

    def test_non_contiguous_block_falls_back(self):
        """A block interleaving nodes from two spec blocks has non-contiguous
        global positions and must take the scatter fallback."""
        mixed = block_management.Block(
            [self.blocks[0][0], self.blocks[1][0], self.blocks[0][1]]
        )
        _, positions = block_management.get_node_locations(mixed, self.spec)
        pos = list(map(int, positions))
        self.assertNotEqual(pos, list(range(pos[0], pos[0] + len(pos))))

        new_state = jnp.array([7.0, 8.0, 9.0])
        out = block_management.scatter_block_to_global(
            self.global_state, new_state, mixed, self.spec
        )
        sd_ind, ref = self._reference(mixed, new_state)
        self.assertTrue(jnp.array_equal(out[sd_ind], ref))

    def test_single_node_block(self):
        """A length-1 block is trivially contiguous."""
        single = block_management.Block([self.blocks[0][2]])
        new_state = jnp.array([42.0])
        out = block_management.scatter_block_to_global(
            self.global_state, new_state, single, self.spec
        )
        sd_ind, ref = self._reference(single, new_state)
        self.assertTrue(jnp.array_equal(out[sd_ind], ref))


class TestFromGlobalState(unittest.TestCase):
    """from_global_state must agree with a reference gather on both the
    contiguous slice path and the non-contiguous fallback."""

    def setUp(self):
        self.sd = {Node1: jax.ShapeDtypeStruct((), jnp.float32)}
        self.blocks = [
            block_management.Block([Node1() for _ in range(4)]),
            block_management.Block([Node1() for _ in range(3)]),
        ]
        self.spec = block_management.BlockSpec(self.blocks, self.sd)
        self.global_state = [jax.random.normal(jax.random.key(7), (7,))]

    def _reference(self, block):
        sd_ind, positions = block_management.get_node_locations(block, self.spec)
        return jnp.take(self.global_state[sd_ind], positions, axis=0)

    def test_contiguous_blocks(self):
        out = block_management.from_global_state(
            self.global_state, self.spec, self.blocks
        )
        for block, extracted in zip(self.blocks, out):
            self.assertTrue(jnp.array_equal(extracted, self._reference(block)))

    def test_non_contiguous_block_falls_back(self):
        mixed = block_management.Block(
            [self.blocks[1][2], self.blocks[0][1], self.blocks[1][0]]
        )
        out = block_management.from_global_state(self.global_state, self.spec, [mixed])
        self.assertTrue(jnp.array_equal(out[0], self._reference(mixed)))


class _CompoundState(eqx.Module):
    data: Array
    label: int  # static leaf


class TestRoundtripFidelity(unittest.TestCase):
    """block_state_to_global → from_global_state fidelity for hetero pytrees."""

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


class TestScatterCorrectness(unittest.TestCase):
    """scatter_block_to_global matches a full rebuild, incl. heterogeneous SDs."""

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
        sd = _CompoundState(jax.ShapeDtypeStruct((4,), jnp.float32), 42)
        self.assertEqual(_hash_pytree(sd), _hash_pytree(sd))


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
