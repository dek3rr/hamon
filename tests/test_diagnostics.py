import unittest

import jax
import jax.numpy as jnp
import numpy as np
from hamon.diagnostics import (
    ESSReport,
    effective_sample_size,
    energy_balance,
    marginal_entropy,
    report_nrpt_diagnostics,
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
        # With only 20 samples, marginals are noisy — should not be CONVERGED.
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


class TestEffectiveSampleSize(unittest.TestCase):
    @staticmethod
    def _ar1(n, rho, seed=0):
        """An AR(1) series with lag-1 autocorrelation ~rho (ESS fraction known)."""
        rng = np.random.default_rng(seed)
        eps = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = eps[0]
        s = np.sqrt(1.0 - rho**2)
        for t in range(1, n):
            x[t] = rho * x[t - 1] + s * eps[t]
        return x

    def test_iid_fraction_near_one(self):
        """IID samples should have ESS ≈ n (fraction near 1)."""
        samples = np.asarray(
            jax.random.bernoulli(jax.random.key(1), 0.5, shape=(20_000, 4))
        )
        report = effective_sample_size(samples)
        self.assertIsInstance(report, ESSReport)
        self.assertEqual(report.per_variable.shape, (4,))
        self.assertGreater(report.ess_fraction, 0.7)
        self.assertLessEqual(report.min_ess, report.n_samples)

    def test_correlated_fraction_small(self):
        """AR(1) with rho=0.9 → fraction ≈ (1-rho)/(1+rho) = 0.0526."""
        x = self._ar1(12_000, rho=0.9)
        report = effective_sample_size(x)  # 1-D input
        self.assertEqual(report.per_variable.shape, (1,))
        # Generous bounds around the 0.053 theoretical value (estimator is noisy).
        self.assertLess(report.ess_fraction, 0.15)
        self.assertGreater(report.ess_fraction, 0.01)

    def test_frozen_column_is_full_ess(self):
        """A zero-variance column carries no autocorrelation info → ESS = n."""
        samples = np.ones((1000, 3))
        report = effective_sample_size(samples)
        np.testing.assert_allclose(report.per_variable, 1000.0)
        self.assertEqual(report.min_ess, 1000.0)

    def test_mixed_columns(self):
        """Frozen + iid columns: all ESS in (0, n], shape preserved."""
        iid = np.asarray(jax.random.bernoulli(jax.random.key(5), 0.5, shape=(5000, 2)))
        frozen = np.ones((5000, 1))
        samples = np.concatenate([iid, frozen], axis=1)
        report = effective_sample_size(samples)
        self.assertEqual(report.per_variable.shape, (3,))
        self.assertTrue(np.all(report.per_variable > 0))
        self.assertTrue(np.all(report.per_variable <= 5000))


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
            lam = (
                jnp.array(lam_profile)
                if lam_profile is not None
                else rej / jnp.diff(stats["betas"])
            )
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

    def test_barrier_identified_on_healthy_run(self):
        """An unsaturated ladder ⇒ the barrier estimate is resolved."""
        report = report_nrpt_diagnostics(self._stats())
        self.assertTrue(report.barrier_identified)

    def test_barrier_not_identified_when_ladder_saturates(self):
        """Pairs pinned at r=1 ⇒ Λ̂ reports its own N-1 cap, not the barrier."""
        report = report_nrpt_diagnostics(self._stats(rej=(1.0, 1.0, 1.0)))
        self.assertFalse(report.barrier_identified)
        self.assertTrue(any("saturates" in i for i in report.issues))
        self.assertIn("BARRIER NOT IDENTIFIED", report.summary())

    def test_resolution_is_independent_of_round_trip_rate(self):
        """Zero round trips must NOT read as an unresolved barrier.

        Resolution is structural (Λ̂ <= N-1); the round-trip rate is
        budget-dependent. The same well-resolved ladder reads zero trips on a
        short window, so gating resolution on it reports "add chains" for a
        ladder that is already correct. The short window is still flagged — as a
        conveyor complaint, not a barrier one.
        """
        report = report_nrpt_diagnostics(self._stats(tau_obs=0.0))
        self.assertTrue(report.barrier_identified)
        self.assertTrue(any("round trip" in i for i in report.issues))
        self.assertFalse(any("saturates" in i for i in report.issues))

    def test_barrier_identified_none_without_round_trips(self):
        """No round-trip diagnostics ⇒ identifiability is unknown (None)."""
        report = report_nrpt_diagnostics(self._stats(with_rt=False))
        self.assertIsNone(report.barrier_identified)

    def test_barrier_is_identified_helper(self):
        import jax.numpy as jnp
        from hamon.round_trips import barrier_is_identified

        self.assertTrue(barrier_is_identified(jnp.array([0.4, 0.5, 0.6])))
        # A single pinned pair blocks the conveyor and caps Λ̂, even if the rest
        # of the ladder is well equalized.
        self.assertFalse(barrier_is_identified(jnp.array([0.4, 1.0, 0.4])))
        self.assertFalse(barrier_is_identified(jnp.array([1.0, 1.0, 1.0])))

    def test_conveyor_is_alive_helper(self):
        from hamon.round_trips import conveyor_is_alive

        # tau_pred=0.02 over 4000 rounds affords 80 expected trips: measurable.
        self.assertTrue(conveyor_is_alive(0.010, 0.02, 4000))  # eff 0.50
        self.assertFalse(conveyor_is_alive(0.000, 0.02, 4000))  # genuinely stalled
        # Same ladder, short window: not measurable ⇒ None, not "stalled".
        self.assertIsNone(conveyor_is_alive(0.0, 0.02, 100))
        # 32 expected trips is still inside the measured transient (efficiency
        # reads 0.000-0.043 there even for a healthy conveyor) ⇒ None.
        self.assertIsNone(conveyor_is_alive(0.0040, 0.008, 4000))
        # The floor is relative, so a hard target (small tau_pred) is not held to
        # a stricter efficiency than an easy one.
        self.assertTrue(conveyor_is_alive(0.0040, 0.008, 12000))  # eff 0.50, Λ≈61

    def test_low_efficiency_fails_with_recommendation(self):
        report = report_nrpt_diagnostics(self._stats(efficiency=0.1))
        self.assertFalse(report.healthy)
        self.assertIsNotNone(report.recommended_n_chains)
        self.assertGreaterEqual(report.recommended_n_chains, 2)

    def test_low_efficiency_equalized_blames_local_exploration(self):
        """Equalized schedule + low efficiency → blame the local kernel."""
        # Default rej=(0.4, 0.4, 0.4) is perfectly equalized (std=0).
        report = report_nrpt_diagnostics(self._stats(efficiency=0.1))
        self.assertEqual(report.efficiency_limiter, "local_exploration")
        self.assertIsNotNone(report.recommended_n_chains)
        self.assertTrue(any("gibbs_steps_per_round" in i for i in report.issues))

    def test_low_efficiency_unequalized_blames_schedule(self):
        """Unequalized schedule + low efficiency → blame the schedule."""
        report = report_nrpt_diagnostics(
            self._stats(rej=(0.05, 0.8, 0.05), efficiency=0.1)
        )
        self.assertEqual(report.efficiency_limiter, "schedule")
        self.assertIsNotNone(report.recommended_n_chains)
        self.assertTrue(any("tune the schedule" in i for i in report.issues))

    def test_efficiency_warn_level_sets_limiter(self):
        """The warn band (no hard fail) still attributes a limiter."""
        report = report_nrpt_diagnostics(self._stats(efficiency=0.3))
        # 0.2 (fail) < 0.3 < 0.35 (warn): a warning, not an issue.
        self.assertEqual(report.issues, [])
        self.assertEqual(report.efficiency_limiter, "local_exploration")
        self.assertTrue(any("local exploration kernel" in w for w in report.warnings))

    def test_healthy_efficiency_no_limiter(self):
        report = report_nrpt_diagnostics(self._stats(efficiency=0.6))
        self.assertIsNone(report.efficiency_limiter)

    def test_unequalized_schedule_fails(self):
        report = report_nrpt_diagnostics(self._stats(rej=(0.05, 0.8, 0.05)))
        self.assertFalse(report.healthy)
        self.assertTrue(any("equalized" in i for i in report.issues))

    def test_insufficient_data_withholds_verdict(self):
        report = report_nrpt_diagnostics(
            self._stats(rej=(0.05, 0.8, 0.05), attempted=(5, 5, 5))
        )
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

    def test_ess_reported_for_iid_samples(self):
        """ESS fields populated; iid samples do not trigger the low-ESS warning."""
        samples = jax.random.bernoulli(jax.random.key(4), 0.5, shape=(5000, 8))
        report = report_nrpt_diagnostics(self._stats(), samples=samples)
        self.assertIsNotNone(report.min_ess)
        self.assertIsNotNone(report.ess_fraction)
        self.assertGreater(report.ess_fraction, 0.5)
        self.assertFalse(any("effective sample size" in w for w in report.warnings))

    def test_low_ess_warns(self):
        """Highly autocorrelated samples trigger a low-ESS warning (not a failure)."""
        # Each value repeated 20× → ESS fraction ≈ 1/20 = 0.05 < ess_warn.
        base = jax.random.bernoulli(jax.random.key(7), 0.7, shape=(60, 8))
        samples = jnp.repeat(base, 20, axis=0)
        report = report_nrpt_diagnostics(self._stats(), samples=samples)
        self.assertIsNotNone(report.min_ess)
        self.assertLess(report.ess_fraction, 0.1)
        self.assertTrue(any("effective sample size" in w for w in report.warnings))
        # Low ESS is informational only — it must not flip the verdict.
        self.assertTrue(report.healthy)

    def test_summary_renders(self):
        report = report_nrpt_diagnostics(self._stats())
        text = report.summary()
        self.assertIn("VERDICT", text)
        self.assertIn("Lambda", text)

    def test_summary_includes_ess(self):
        samples = jax.random.bernoulli(jax.random.key(8), 0.5, shape=(2000, 6))
        report = report_nrpt_diagnostics(self._stats(), samples=samples)
        self.assertIn("ess", report.summary())
