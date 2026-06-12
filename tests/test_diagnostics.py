import unittest

import jax
import jax.numpy as jnp

from hamon.diagnostics import (
    energy_balance,
    report_nrpt_diagnostics,
    marginal_entropy,
    sample_convergence,
)


class TestSampleConvergence(unittest.TestCase):
    def test_converged_stable_samples(self):
        """IID Bernoulli samples with fixed p should converge quickly."""
        key = jax.random.key(0)
        p = jnp.array([0.8, 0.2, 0.5, 0.9, 0.1])
        samples = jax.random.bernoulli(key, p, shape=(10_000, 5))
        report = sample_convergence(samples, target_k=3)
        self.assertEqual(report.status, "CONVERGED")
        self.assertEqual(len(report.drifts), 3)
        for d in report.drifts:
            self.assertLess(d, 0.02)
        self.assertGreaterEqual(report.rank_stability, 0.8)

    def test_need_more_with_few_samples(self):
        """Very few samples from a near-uniform distribution should not converge."""
        key = jax.random.key(1)
        samples = jax.random.bernoulli(key, 0.5, shape=(20, 50))
        report = sample_convergence(samples, target_k=10)
        # With only 20 samples, marginals are noisy Ã¢â‚¬â€ should not be CONVERGED.
        self.assertIn(report.status, ("BORDERLINE", "NEED_MORE"))

    def test_target_k_clamped(self):
        """target_k larger than n_variables should not error."""
        samples = jnp.ones((100, 3), dtype=jnp.bool_)
        report = sample_convergence(samples, target_k=100)
        self.assertIsNotNone(report.rank_stability)


class TestMarginalEntropy(unittest.TestCase):
    def test_uniform_high_entropy(self):
        """50/50 Bernoulli samples should have entropy near 1."""
        key = jax.random.key(2)
        samples = jax.random.bernoulli(key, 0.5, shape=(50_000, 20))
        h = marginal_entropy(samples)
        self.assertGreater(h, 0.95)

    def test_frozen_low_entropy(self):
        """All-True samples should have entropy near 0."""
        samples = jnp.ones((1000, 10), dtype=jnp.bool_)
        h = marginal_entropy(samples)
        self.assertLess(h, 0.01)

    def test_mixed(self):
        """Half frozen, half uniform should give intermediate entropy."""
        key = jax.random.key(3)
        n = 10_000
        frozen = jnp.ones((n, 5), dtype=jnp.bool_)
        uniform = jax.random.bernoulli(key, 0.5, shape=(n, 5))
        samples = jnp.concatenate([frozen, uniform], axis=1)
        h = marginal_entropy(samples)
        self.assertGreater(h, 0.3)
        self.assertLess(h, 0.7)


class TestEnergyBalance(unittest.TestCase):
    def test_balanced(self):
        """Comparable bias and coupling magnitudes should have ratio near 1."""
        biases = jnp.array([0.5, -0.5, 0.3, -0.3])
        edges = jnp.array([[0, 1], [1, 2], [2, 3]])
        weights = jnp.array([0.4, 0.4, 0.4])
        report = energy_balance(biases, edges, weights, beta=1.0)
        self.assertGreater(report.ratio, 0.05)
        self.assertLess(report.ratio, 20.0)
        self.assertGreater(report.bias_energy_spread, 0.0)
        self.assertGreater(report.coupling_energy_per_spin, 0.0)

    def test_bias_dominated(self):
        """Huge biases, tiny couplings should have low ratio."""
        biases = jnp.array([10.0, -10.0, 5.0])
        edges = jnp.array([[0, 1], [1, 2]])
        weights = jnp.array([0.001, 0.001])
        report = energy_balance(biases, edges, weights, beta=1.0, warn_low=0.05)
        self.assertLess(report.ratio, 0.05)

    def test_coupling_dominated(self):
        """Tiny biases, huge couplings should have high ratio."""
        biases = jnp.array([0.001, 0.001, 0.001, 0.001])
        edges = jnp.array([[0, 1], [1, 2], [2, 3], [0, 3]])
        weights = jnp.array([5.0, 5.0, 5.0, 5.0])
        report = energy_balance(biases, edges, weights, beta=1.0)
        self.assertGreater(report.ratio, 2.0)

    def test_beta_scaling(self):
        """Doubling beta should not change the ratio (both terms scale equally)."""
        biases = jnp.array([1.0, -1.0, 0.5])
        edges = jnp.array([[0, 1], [1, 2]])
        weights = jnp.array([0.8, 0.8])
        r1 = energy_balance(biases, edges, weights, beta=1.0)
        r2 = energy_balance(biases, edges, weights, beta=2.0)
        self.assertAlmostEqual(r1.ratio, r2.ratio, places=5)

    def test_zero_biases(self):
        """Zero biases should give infinite ratio without error."""
        biases = jnp.zeros(3)
        edges = jnp.array([[0, 1], [1, 2]])
        weights = jnp.array([1.0, 1.0])
        report = energy_balance(biases, edges, weights)
        self.assertEqual(report.ratio, float("inf"))


class TestNRPTHealthReport(unittest.TestCase):
    """Verdict logic of report_nrpt_diagnostics."""

    def _stats(
        self,
        rej=(0.4, 0.4, 0.4),
        attempted=(100, 100, 100),
        tau_obs=0.05,
        efficiency=0.6,
        with_rt=True,
        lam_profile=None,
    ):
        rej = jnp.array(rej)
        stats = {
            "acceptance_rate": 1.0 - rej,
            "rejection_rates": rej,
            "attempted": jnp.array(attempted),
            "betas": jnp.linspace(0.2, 1.0, len(rej) + 1),
        }
        if with_rt:
            n_chains = len(rej) + 1
            lam = jnp.array(lam_profile) if lam_profile is not None else rej / jnp.diff(stats["betas"])
            tau_pred = 1.0 / (2.0 + 2.0 * float(jnp.sum(rej)))
            stats["round_trip_diagnostics"] = {
                "Lambda": jnp.sum(rej),
                "tau_observed": jnp.array(tau_obs),
                "tau_predicted": jnp.array(tau_pred),
                "efficiency": jnp.array(efficiency),
                "lambda_profile": lam,
                "round_trips_per_chain": jnp.ones(n_chains, dtype=jnp.int32) * 3,
                "restarts_per_chain": jnp.zeros(n_chains, dtype=jnp.int32),
            }
        return stats

    def test_healthy_run(self):
        report = report_nrpt_diagnostics(self._stats())
        self.assertTrue(report.healthy)
        self.assertEqual(report.issues, [])
        self.assertFalse(report.insufficient_data)

    def test_zero_round_trips_fails(self):
        report = report_nrpt_diagnostics(self._stats(tau_obs=0.0))
        self.assertFalse(report.healthy)
        self.assertTrue(any("round trip" in i for i in report.issues))

    def test_low_efficiency_fails_with_recommendation(self):
        report = report_nrpt_diagnostics(self._stats(efficiency=0.1))
        self.assertFalse(report.healthy)
        self.assertIsNotNone(report.recommended_n_chains)
        self.assertGreaterEqual(report.recommended_n_chains, 2)

    def test_unequalized_schedule_fails(self):
        report = report_nrpt_diagnostics(self._stats(rej=(0.05, 0.8, 0.05)))
        self.assertFalse(report.healthy)
        self.assertTrue(any("equalized" in i for i in report.issues))

    def test_insufficient_data_withholds_verdict(self):
        report = report_nrpt_diagnostics(self._stats(rej=(0.05, 0.8, 0.05), attempted=(5, 5, 5)))
        self.assertTrue(report.insufficient_data)
        self.assertEqual(report.issues, [])  # demoted to warnings
        self.assertFalse(report.healthy)
        self.assertTrue(any("insufficient" in w for w in report.warnings))

    def test_missing_round_trips_warns(self):
        report = report_nrpt_diagnostics(self._stats(with_rt=False))
        self.assertIsNone(report.Lambda)
        self.assertTrue(any("track_round_trips" in w for w in report.warnings))

    def test_frozen_samples_fail(self):
        samples = jnp.ones((200, 8), dtype=jnp.bool_)
        report = report_nrpt_diagnostics(self._stats(), samples=samples)
        self.assertFalse(report.healthy)
        self.assertTrue(any("frozen" in i for i in report.issues))

    def test_convergence_is_informational_only(self):
        """A NEED_MORE convergence status alone must not fail the verdict."""
        key = jax.random.key(3)
        # Drifting marginals: first half mostly False, second half mostly True.
        half = jax.random.bernoulli(key, 0.15, shape=(100, 8))
        samples = jnp.concatenate([half, ~half], axis=0)
        report = report_nrpt_diagnostics(self._stats(), samples=samples)
        self.assertNotEqual(report.convergence_status, "CONVERGED")
        self.assertTrue(report.healthy)

    def test_barrier_peak_warns(self):
        report = report_nrpt_diagnostics(
            self._stats(
                rej=(0.1, 0.1, 0.1, 0.1),
                attempted=(100, 100, 100, 100),
                lam_profile=(0.1, 0.1, 5.0, 0.1),
            )
        )
        self.assertIsNotNone(report.barrier_peak_beta)
        self.assertTrue(any("barrier" in w for w in report.warnings))

    def test_summary_renders(self):
        report = report_nrpt_diagnostics(self._stats())
        text = report.summary()
        self.assertIn("VERDICT", text)
        self.assertIn("Lambda", text)
