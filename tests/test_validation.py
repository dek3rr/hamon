"""Tests for semantic validation checks."""

import unittest

import jax
import jax.numpy as jnp
from hamon import Block, SpinNode
from hamon.models import IsingEBM, IsingSamplingProgram, hinton_init
from hamon.models.ising import ising_sample
from hamon.nrpt import nrpt


class TestNodeIdentityValidation(unittest.TestCase):
    """nrpt() should reject programs whose node objects don't match."""

    def test_mismatched_nodes_raises(self):
        """Programs built from independent SpinNode sets should be rejected."""
        n = 4
        betas = [0.5, 1.0]

        # Build two programs from DIFFERENT node objects.
        def _make(beta):
            nodes = [SpinNode() for _ in range(n)]
            edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
            biases = jnp.zeros(n)
            weights = jnp.ones(n - 1)
            ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(beta))
            fb = [Block(nodes[::2]), Block(nodes[1::2])]
            return ebm, IsingSamplingProgram(ebm, fb, []), fb

        ebm0, prog0, fb0 = _make(betas[0])
        ebm1, prog1, fb1 = _make(betas[1])

        key = jax.random.key(0)
        init0 = hinton_init(key, ebm0, fb0, ())
        init1 = hinton_init(key, ebm1, fb1, ())

        with self.assertRaises(ValueError, msg="different node objects"):
            nrpt(
                key,
                [ebm0, ebm1],
                [prog0, prog1],
                [init0, init1],
                [],
                n_rounds=1,
                gibbs_steps_per_round=0,
                betas=jnp.array(betas),
            )

    def test_shared_nodes_accepted(self):
        """Programs built via with_beta / with_ebm (shared nodes) should pass."""
        n = 4
        nodes = [SpinNode() for _ in range(n)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
        biases = jnp.zeros(n)
        weights = jnp.ones(n - 1)
        ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(0.5))
        fb = [Block(nodes[::2]), Block(nodes[1::2])]
        prog = IsingSamplingProgram(ebm, fb, [])

        ebm2 = ebm.with_beta(jnp.array(1.0))
        prog2 = prog.with_ebm(ebm2)

        key = jax.random.key(0)
        inits = [hinton_init(key, ebm, fb, ()) for _ in range(2)]

        # Should not raise.
        nrpt(
            key,
            [ebm, ebm2],
            [prog, prog2],
            inits,
            [],
            n_rounds=1,
            gibbs_steps_per_round=0,
            betas=jnp.array([0.5, 1.0]),
        )


class TestDegenerateModelWarnings(unittest.TestCase):
    """ising_sample should warn about degenerate models."""

    def test_zero_couplings_warns(self):
        key = jax.random.key(10)
        biases = jnp.array([0.5, -0.5, 0.3])
        edges = jnp.array([[0, 1], [1, 2]])
        weights = jnp.zeros(2)

        with self.assertLogs("hamon.models.ising", level="WARNING") as cm:
            ising_sample(biases, edges, weights, key=key, n_samples=10)
        self.assertTrue(
            any("zero" in msg.lower() for msg in cm.output),
            f"Expected zero-coupling warning, got: {cm.output}",
        )

    def test_identical_biases_warns(self):
        key = jax.random.key(11)
        biases = jnp.ones(4) * 0.5
        edges = jnp.array([[0, 1], [1, 2], [2, 3]])
        weights = jnp.ones(3)

        with self.assertLogs("hamon.models.ising", level="WARNING") as cm:
            ising_sample(biases, edges, weights, key=key, n_samples=10)
        self.assertTrue(
            any("identical" in msg.lower() for msg in cm.output),
            f"Expected identical-bias warning, got: {cm.output}",
        )


class TestMeanSpinsDiagnostic(unittest.TestCase):
    """ising_sample should report mean_spins in diagnostics."""

    def test_mean_spins_present(self):
        key = jax.random.key(20)
        n = 5
        biases = jnp.zeros(n)
        edges = jnp.array([[i, i + 1] for i in range(n - 1)])
        weights = jnp.ones(n - 1) * 0.5

        samples, diagnostics = ising_sample(
            biases,
            edges,
            weights,
            key=key,
            beta=1.0,
            n_samples=50,
        )

        self.assertIn("mean_spins", diagnostics)
        # mean_spins should be between 0 and n.
        self.assertGreaterEqual(diagnostics["mean_spins"], 0.0)
        self.assertLessEqual(diagnostics["mean_spins"], n)
        # Verify it matches the actual samples.
        expected = float(jnp.mean(jnp.sum(samples, axis=1).astype(jnp.float32)))
        self.assertAlmostEqual(diagnostics["mean_spins"], expected, places=5)
