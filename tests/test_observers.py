"""Tests for observers.py.

Covers:
- MomentAccumulatorObserver preserves mixed node value types (existing)
- StateObserver: global_state fast path gives same result as recomputing
- MomentAccumulatorObserver: global_state fast path gives same result
- MomentAccumulatorObserver: dtype parameter propagates to accumulators
- MomentAccumulatorObserver: accumulation is numerically equivalent both paths
"""

import types
import unittest

import jax
import jax.numpy as jnp

from hamon.block_management import Block, block_state_to_global
from hamon.block_sampling import BlockGibbsSpec
from hamon.observers import MomentAccumulatorObserver, StateObserver
from hamon.pgm import CategoricalNode, SpinNode


def _make_simple_program(blocks, node_shape_dtypes):
    """Build a minimal namespace that looks like a BlockSamplingProgram to observers."""
    gibbs_spec = BlockGibbsSpec(blocks, [], node_shape_dtypes)
    return types.SimpleNamespace(gibbs_spec=gibbs_spec)


class TestMomentObserver(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spin = SpinNode()
        self.cat = CategoricalNode()

        self.blocks = [Block([self.spin]), Block([self.cat])]
        self.node_shape_dtypes = {
            SpinNode: jax.ShapeDtypeStruct((), jnp.bool_),
            CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint8),
        }
        self.program = _make_simple_program(self.blocks, self.node_shape_dtypes)

    def test_preserves_mixed_node_values(self):
        """Moment observer correctly handles mixed bool/uint8 node types."""
        observer = MomentAccumulatorObserver([[(self.spin, self.cat)]])
        carry = observer.init()

        state_free = [
            jnp.array([True], dtype=jnp.bool_),
            jnp.array([2], dtype=jnp.uint8),
        ]

        with jax.numpy_dtype_promotion("standard"):
            carry_out, _ = observer(
                self.program, state_free, [], carry, jnp.array(0, dtype=jnp.int32)
            )

        # spin=True → ±1 transform gives +1; cat=2 → 2; product = 2
        self.assertEqual(carry_out[0][0], 2)

    def test_dtype_parameter_float32(self):
        """Default dtype is float32 and is used by init() and accumulate."""
        observer = MomentAccumulatorObserver([[(self.spin,)]])
        carry = observer.init()
        self.assertEqual(carry[0].dtype, jnp.float32)

    def test_dtype_parameter_float64(self):
        """dtype=float64 is canonicalized at construction time. On platforms
        without x64 enabled JAX truncates float64 → float32, so
        _accumulate_dtype will be float32. The test checks the stored dtype
        matches whatever JAX actually produces."""
        import jax.dtypes as jdt

        observer = MomentAccumulatorObserver([[(self.spin,)]], dtype=jnp.float64)

        expected = jdt.canonicalize_dtype(jnp.float64)
        self.assertEqual(observer._accumulate_dtype, expected)

        carry = observer.init()
        self.assertEqual(carry[0].dtype, expected)

    def test_global_state_fast_path_matches_recompute(self):
        """Passing global_state explicitly gives the same result as omitting it."""
        observer = MomentAccumulatorObserver([[(self.spin,)], [(self.cat,)]])
        carry = observer.init()

        state_free = [
            jnp.array([True], dtype=jnp.bool_),
            jnp.array([3], dtype=jnp.uint8),
        ]

        # Path 1: let the observer recompute global state internally
        with jax.numpy_dtype_promotion("standard"):
            carry_no_gs, _ = observer(self.program, state_free, [], carry, jnp.array(0))

        # Path 2: pass precomputed global state
        global_state = block_state_to_global(state_free, self.program.gibbs_spec)
        with jax.numpy_dtype_promotion("standard"):
            carry_with_gs, _ = observer(
                self.program,
                state_free,
                [],
                carry,
                jnp.array(0),
                global_state=global_state,
            )

        for a, b in zip(carry_no_gs, carry_with_gs):
            self.assertTrue(jnp.allclose(a, b))


class TestStateObserver(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = [SpinNode() for _ in range(4)]
        self.block = Block(self.nodes)
        self.node_shape_dtypes = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        self.program = _make_simple_program([self.block], self.node_shape_dtypes)

    def test_returns_correct_state(self):
        """StateObserver returns the state of the requested block."""
        observer = StateObserver([self.block])
        state_free = [jnp.array([True, False, True, False], dtype=jnp.bool_)]

        _, samples = observer(self.program, state_free, [], None, jnp.array(0))
        self.assertTrue(jnp.array_equal(samples[0], state_free[0]))

    def test_global_state_fast_path_matches_recompute(self):
        """Passing global_state explicitly gives the same result as omitting it."""
        observer = StateObserver([self.block])
        state_free = [jnp.array([True, True, False, True], dtype=jnp.bool_)]

        _, samples_no_gs = observer(self.program, state_free, [], None, jnp.array(0))

        global_state = block_state_to_global(state_free, self.program.gibbs_spec)
        _, samples_with_gs = observer(
            self.program,
            state_free,
            [],
            None,
            jnp.array(0),
            global_state=global_state,
        )

        self.assertTrue(jnp.array_equal(samples_no_gs[0], samples_with_gs[0]))

    def test_carry_is_always_none(self):
        """StateObserver is stateless — carry is always None."""
        observer = StateObserver([self.block])
        state_free = [jnp.array([False, False, False, False], dtype=jnp.bool_)]

        carry_out, _ = observer(self.program, state_free, [], None, jnp.array(0))
        self.assertIsNone(carry_out)


class TestMomentAccumulation(unittest.TestCase):
    """Verify that the moment accumulator sums correctly over multiple calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node = SpinNode()
        self.block = Block([self.node])
        self.node_shape_dtypes = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}
        self.program = _make_simple_program([self.block], self.node_shape_dtypes)

    def _spin_transform(self, state, _blocks):
        return [2 * state[0].astype(jnp.int8) - 1]

    def test_first_moment_accumulates(self):
        """Running sum of ±1 transforms should match manual computation."""
        observer = MomentAccumulatorObserver(
            [[(self.node,)]],
            f_transform=self._spin_transform,
        )
        carry = observer.init()

        # +1, -1, +1 → sum = +1
        for val, spin in [(True, 1), (False, -1), (True, 1)]:
            state_free = [jnp.array([val], dtype=jnp.bool_)]
            with jax.numpy_dtype_promotion("standard"):
                carry, _ = observer(self.program, state_free, [], carry, jnp.array(0))

        self.assertAlmostEqual(float(carry[0][0]), 1.0, places=5)

    def test_second_moment_accumulates(self):
        """Running sum of product s_i * s_j for i=j should equal sum of s_i^2 = n_steps."""
        observer = MomentAccumulatorObserver(
            [[(self.node, self.node)]],
            f_transform=self._spin_transform,
        )
        carry = observer.init()

        n_steps = 5
        for _ in range(n_steps):
            state_free = [jnp.array([True], dtype=jnp.bool_)]
            with jax.numpy_dtype_promotion("standard"):
                carry, _ = observer(self.program, state_free, [], carry, jnp.array(0))

        # (+1)^2 * n_steps = n_steps
        self.assertAlmostEqual(float(carry[0][0]), float(n_steps), places=5)
