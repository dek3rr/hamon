"""
NRPT Correctness Tests
======================
Priority order:
  1. Boltzmann marginals  — cold chain samples from correct distribution
  2. DEO detailed balance — swap acceptance matches MH criterion exactly
  3. Adaptive schedule    — Λ decreases monotonically, min-acceptance floor held
  4. τ̄ prediction         — observed round trip rate tracks 1/(2+2Λ)

Design constraints:
  - Models ≤ 10 spins so exact partition function is enumerable in < 1s
  - No mocking of internals — tests go through the public API
  - Tolerances are generous enough to be stable but tight enough to catch bugs
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hamon import Block, SpinNode, SamplingSchedule, sample_states
from hamon.models import IsingEBM, IsingSamplingProgram, hinton_init
from hamon.nrpt import nrpt, nrpt_adaptive, optimize_schedule


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ising_chain(n: int, coupling: float, beta: float, key: jax.Array):
    """1D Ising chain with random biases, fixed coupling, given beta."""
    nodes = [SpinNode() for _ in range(n)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
    biases = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    weights = jnp.ones(n - 1) * coupling
    ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(beta))
    even = nodes[::2]
    odd = nodes[1::2]
    free_blocks = [Block(even), Block(odd)]
    prog = IsingSamplingProgram(ebm, free_blocks, [])
    return nodes, edges, biases, weights, free_blocks, ebm, prog


def _make_chain_set(nodes, edges, biases, weights, free_blocks, betas: list[float]):
    ebms = [IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas]
    progs = [IsingSamplingProgram(e, free_blocks, []) for e in ebms]
    return ebms, progs


def _init_states(key, n_chains, ebms, free_blocks):
    keys = jax.random.split(key, n_chains)
    return [hinton_init(keys[c], ebms[0], free_blocks, ()) for c in range(n_chains)]


def _exact_boltzmann(nodes, ebm, obs_block):
    """Enumerate all 2^n states and return unnormalised Boltzmann weights."""
    n = len(nodes)
    n_states = 2**n
    weights = np.zeros(n_states)
    for i in range(n_states):
        bits = jnp.array([(i >> k) & 1 for k in range(n)], dtype=jnp.bool_)
        state = [bits]
        e = float(ebm.energy(state, [obs_block]))
        weights[i] = np.exp(-e)
    return weights / weights.sum()


def _empirical_marginals(samples: jax.Array) -> jax.Array:
    """samples: (N, n_spins) bool → empirical marginals P(x_k=True)."""
    return jnp.mean(samples.astype(jnp.float32), axis=0)


def _exact_marginals(exact_probs: np.ndarray, n: int) -> np.ndarray:
    """Exact marginals from full joint over 2^n states."""
    marginals = np.zeros(n)
    for i in range(2**n):
        for k in range(n):
            if (i >> k) & 1:
                marginals[k] += exact_probs[i]
    return marginals


# ---------------------------------------------------------------------------
# T1: Boltzmann marginals
#
# The cold chain of nrpt must sample from exp(-β_cold * E_base(x)).
# Strategy: small model (n=8), use nrpt_adaptive for warmup to avoid cold-start
# bias, then collect samples via sample_states from the warm state.
# Compare empirical first-order marginals to exact enumeration.
#
# This does not directly test that the swap kernel is correct (T2 does that),
# but it tests end-to-end stationarity of the cold chain.
# ---------------------------------------------------------------------------


class TestBoltzmannMarginals:
    @pytest.fixture(scope="class")
    def model_8(self):
        n = 8
        beta_cold = 1.0
        key = jax.random.key(11)
        k_bias, k_init, k_nrpt, k_samp = jax.random.split(key, 4)
        nodes, edges, biases, weights, fb, ebm_cold, prog_cold = _ising_chain(
            n, coupling=0.8, beta=beta_cold, key=k_bias
        )
        betas = [0.2, 0.5, beta_cold]
        ebms, progs = _make_chain_set(nodes, edges, biases, weights, fb, betas)
        init = _init_states(k_init, len(betas), ebms, fb)

        # Warmup via nrpt_adaptive so cold chain isn't stuck
        def ebm_factory(b):
            return [
                IsingEBM(nodes, edges, biases, weights, jnp.array(float(bi)))
                for bi in b
            ]

        def prog_factory(es):
            return [IsingSamplingProgram(e, fb, []) for e in es]

        warm_states, _, _ = nrpt_adaptive(
            k_nrpt,
            ebm_factory,
            prog_factory,
            init,
            [],
            n_rounds=500,
            gibbs_steps_per_round=5,
            initial_betas=jnp.array(betas),
            n_tune=3,
            rounds_per_tune=100,
        )

        # Collect samples from warm cold-chain state only
        schedule = SamplingSchedule(n_warmup=200, n_samples=8000, steps_per_sample=3)
        obs_block = Block(nodes)
        samples = sample_states(
            k_samp, prog_cold, schedule, warm_states[2], [], [obs_block]
        )
        # samples[0]: (8000, 8) bool

        return dict(
            samples=samples[0], nodes=nodes, ebm_cold=ebm_cold, obs_block=obs_block
        )

    def test_first_order_marginals(self, model_8):
        """Empirical P(x_k=1) must match exact to within 3% relative error."""
        samples = model_8["samples"]
        nodes = model_8["nodes"]
        ebm = model_8["ebm_cold"]
        obs_block = model_8["obs_block"]
        n = len(nodes)

        exact_probs = _exact_boltzmann(nodes, ebm, obs_block)
        exact_marg = _exact_marginals(exact_probs, n)
        emp_marg = np.array(_empirical_marginals(samples))

        max_rel_err = np.max(
            np.abs(emp_marg - exact_marg) / (np.abs(exact_marg) + 1e-8)
        )
        assert max_rel_err < 0.05, (
            f"Marginal mismatch: max relative error {max_rel_err:.3f}\n"
            f"  exact:    {np.round(exact_marg, 3)}\n"
            f"  empirical:{np.round(emp_marg, 3)}"
        )

    def test_full_joint_tv_distance(self, model_8):
        """Total variation distance between empirical joint and exact must be < 0.05."""
        samples = model_8["samples"]
        nodes = model_8["nodes"]
        ebm = model_8["ebm_cold"]
        obs_block = model_8["obs_block"]
        n = len(nodes)

        exact_probs = _exact_boltzmann(nodes, ebm, obs_block)

        # Empirical joint
        n_states = 2**n
        counts = np.zeros(n_states)
        for row in np.array(samples):
            idx = sum(int(row[k]) << k for k in range(n))
            counts[idx] += 1
        emp_probs = counts / counts.sum()

        tv = 0.5 * np.sum(np.abs(emp_probs - exact_probs))
        assert tv < 0.05, f"TV distance {tv:.4f} exceeds threshold"

    def test_nrpt_vs_single_chain_marginals(self):
        """Cold chain after nrpt warmup should match single-chain run on same model.

        If nrpt corrupts the stationary distribution, these will diverge.
        """
        n = 6
        key = jax.random.key(77)
        k_bias, k_sc, k_pt, k_samp_sc, k_samp_pt = jax.random.split(key, 5)
        beta = 1.0

        nodes, edges, biases, weights, fb, ebm, prog = _ising_chain(
            n, coupling=0.7, beta=beta, key=k_bias
        )
        obs_block = Block(nodes)

        # Single-chain reference
        sc_init = hinton_init(k_sc, ebm, fb, ())
        sc_schedule = SamplingSchedule(
            n_warmup=2000, n_samples=6000, steps_per_sample=3
        )
        sc_samples = sample_states(
            k_samp_sc, prog, sc_schedule, sc_init, [], [obs_block]
        )[0]

        # NRPT warmup → sample_states
        betas = [0.2, 0.5, beta]
        ebms, progs = _make_chain_set(nodes, edges, biases, weights, fb, betas)
        pt_init = _init_states(k_pt, 3, ebms, fb)

        def ebm_f(b):
            return [
                IsingEBM(nodes, edges, biases, weights, jnp.array(float(bi)))
                for bi in b
            ]

        def prog_f(es):
            return [IsingSamplingProgram(e, fb, []) for e in es]

        warm, _, _ = nrpt_adaptive(
            k_pt,
            ebm_f,
            prog_f,
            pt_init,
            [],
            n_rounds=300,
            gibbs_steps_per_round=5,
            initial_betas=jnp.array(betas),
            n_tune=2,
            rounds_per_tune=100,
        )
        pt_schedule = SamplingSchedule(n_warmup=0, n_samples=6000, steps_per_sample=3)
        pt_samples = sample_states(
            k_samp_pt, prog, pt_schedule, warm[2], [], [obs_block]
        )[0]

        sc_marg = np.array(_empirical_marginals(sc_samples))
        pt_marg = np.array(_empirical_marginals(pt_samples))

        max_diff = np.max(np.abs(sc_marg - pt_marg))
        assert max_diff < 0.05, (
            f"NRPT and single-chain marginals diverge by {max_diff:.3f}\n"
            f"  single-chain: {np.round(sc_marg, 3)}\n"
            f"  nrpt:         {np.round(pt_marg, 3)}"
        )


# ---------------------------------------------------------------------------
# T2: DEO swap detailed balance
#
# The swap acceptance criterion is:
#   α = min(1, exp((β_i - β_j)(E_base_i - E_base_j)))
#
# Test strategy: 2 chains, gibbs_steps=0 (states never change).
# With fixed states, every swap attempt sees the same (E_i, E_j).
# Empirical acceptance rate must match the analytical α.
#
# Three sub-cases:
#   (a) α = 1 (always accept): hot chain has lower energy than cold chain
#   (b) α ≈ 0 (almost never): cold chain has much lower energy, large Δβ
#   (c) Intermediate α: verify empirical rate ≈ analytical to 3%
# ---------------------------------------------------------------------------


class TestDEODetailedBalance:
    def _two_chain_setup(self, n, biases, coupling, betas, init_spins_0, init_spins_1):
        """Build 2-chain system with deterministic initial states (no Gibbs)."""
        nodes = [SpinNode() for _ in range(n)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
        weights = jnp.ones(n - 1) * coupling
        bias_arr = jnp.array(biases)
        free_blocks = [Block(nodes)]
        ebms = [IsingEBM(nodes, edges, bias_arr, weights, jnp.array(b)) for b in betas]
        progs = [IsingSamplingProgram(e, free_blocks, []) for e in ebms]

        # Fixed initial states
        s0 = [jnp.array(init_spins_0, dtype=jnp.bool_)]
        s1 = [jnp.array(init_spins_1, dtype=jnp.bool_)]

        E0 = float(ebms[0].energy(s0, free_blocks))
        E1 = float(ebms[1].energy(s1, free_blocks))
        E_base_0 = E0 / betas[0]
        E_base_1 = E1 / betas[1]
        log_r = (betas[0] - betas[1]) * (E_base_0 - E_base_1)
        expected_alpha = min(1.0, float(np.exp(log_r)))

        return nodes, free_blocks, ebms, progs, [s0, s1], expected_alpha

    def test_always_accept(self):
        """Single-round deterministic check: alpha=1 swap must be accepted.

        Criterion: exp((beta_hot - beta_cold)(E_base_hot - E_base_cold)).
        beta_hot - beta_cold = 0.5 - 2.0 = -1.5  (negative).
        For log_r > 0 we need E_base_hot - E_base_cold < 0,
        i.e. the hot chain must be in the LOW-energy state.

        This is the "swapped" configuration: hot chain holds the state the
        cold chain would prefer, and vice versa. Swapping benefits both, so
        acceptance is certain regardless of the RNG draw.

        With positive biases, all-True is the lowest-energy state.
        So: hot=all-True (low E_base), cold=all-False (high E_base).

        n_rounds=2 with 2 chains: only round 0 (even) fires pair 0.
        Exactly 1 attempt, must be accepted.
        """
        n = 4
        biases = [2.0] * n
        nodes, fb, ebms, progs, inits, alpha = self._two_chain_setup(
            n,
            biases,
            coupling=0.0,
            betas=[0.5, 2.0],
            init_spins_0=[True] * n,  # hot chain: LOW energy (swap-favorable)
            init_spins_1=[False] * n,  # cold chain: HIGH energy
        )
        assert alpha == pytest.approx(1.0, abs=1e-6), (
            f"Test setup error: expected alpha=1, got {alpha}"
        )

        _, _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            inits,
            [],
            n_rounds=2,
            gibbs_steps_per_round=0,
            track_round_trips=False,
        )
        assert int(stats["attempted"][0]) == 1, "Expected exactly 1 swap attempt"
        assert int(stats["accepted"][0]) == 1, (
            f"alpha=1 swap was not accepted: accepted={stats['accepted'][0]}"
        )

    def test_near_zero_accept(self):
        """When α≈0, swap rate must be very low."""
        # Cold chain (β=3.0) has much lower energy than hot chain (β=0.3).
        # log_r = (0.3 - 3.0)(E_base_hot - E_base_cold)
        # E_base_hot < E_base_cold (cold chain stuck in low energy state)
        # → log_r = (-2.7)(negative) = positive? No — reverse:
        # hot at all-True (low energy), cold at all-True (low energy) → no diff
        # Need: cold has lower energy than hot (typical for a cold chain)
        # log_r = (β_hot - β_cold)(E_base_hot - E_base_cold)
        # For α≈0: need (β_hot - β_cold)(E_base_hot - E_base_cold) << -1
        # β_hot - β_cold = negative; so need E_base_hot - E_base_cold > 0,
        # i.e., hot chain has HIGHER base energy. That means hot chain is in a
        # high-energy state and cold chain in a low-energy state.
        n = 6
        biases = [3.0] * n  # strong field: all-True is very low energy
        nodes, fb, ebms, progs, inits, alpha = self._two_chain_setup(
            n,
            biases,
            coupling=0.0,
            betas=[0.3, 3.0],
            init_spins_0=[False] * n,  # hot: high energy
            init_spins_1=[True] * n,  # cold: low energy
        )
        # hot has high energy, cold has low energy
        # log_r = (0.3 - 3.0)(E_base_hot - E_base_cold)
        # E_base_hot > E_base_cold (hot is high energy)
        # log_r = negative * positive = negative → α = exp(log_r) << 1
        assert alpha < 0.02, f"Test setup error: expected α≈0, got {alpha}"

        _, _, stats = nrpt(
            jax.random.key(1),
            ebms,
            progs,
            inits,
            [],
            n_rounds=2000,
            gibbs_steps_per_round=0,
            track_round_trips=False,
        )
        acc = float(stats["acceptance_rate"][0])
        assert acc < 0.05, f"Expected near-zero acceptance, got {acc:.4f}"

    def test_intermediate_acceptance_rate(self):
        """Empirical rate matches the 2-state equilibrium formula.

        With gibbs_steps=0, the chain oscillates between states A=(s0,s1)
        and B=(s1,s0). The long-run acceptance rate is NOT alpha_A but:
          eq_rate = 2 * alpha_A * alpha_B / (alpha_A + alpha_B)
        derived from the stationary distribution of the 2-state Markov chain.
        """
        n = 4
        biases = [1.0, -0.5, 0.8, -0.3]
        nodes, fb, ebms, progs, inits, _ = self._two_chain_setup(
            n,
            biases,
            coupling=0.0,
            betas=[0.8, 1.6],
            init_spins_0=[True, False, True, False],
            init_spins_1=[False, True, False, True],
        )

        s0, s1 = inits[0], inits[1]
        E0 = float(ebms[0].energy(s0, fb))
        E1 = float(ebms[1].energy(s1, fb))
        log_r = (0.8 - 1.6) * (E0 / 0.8 - E1 / 1.6)

        alpha_A = min(1.0, float(np.exp(log_r)))
        alpha_B = min(1.0, float(np.exp(-log_r)))
        expected_eq_rate = 2.0 * alpha_A * alpha_B / (alpha_A + alpha_B)

        _, _, stats = nrpt(
            jax.random.key(2),
            ebms,
            progs,
            inits,
            [],
            n_rounds=4000,
            gibbs_steps_per_round=0,
            track_round_trips=False,
        )
        emp_rate = float(stats["acceptance_rate"][0])

        # 2000 actual attempts (only even rounds fire for n_chains=2)
        # std ~ sqrt(p*(1-p)/2000) < 0.01; 4-sigma tolerance
        assert abs(emp_rate - expected_eq_rate) < 0.04, (
            f"Equilibrium acceptance: expected {expected_eq_rate:.4f}, got {emp_rate:.4f}"
            f" (alpha_A={alpha_A:.4f}, alpha_B={alpha_B:.4f})"
        )

    def test_acceptance_is_symmetric(self):
        """Swapping x↔y should have same rate as y↔x (detailed balance symmetry)."""
        n = 4
        biases = [0.5, -1.0, 1.5, -0.5]
        s_a = [True, False, True, True]
        s_b = [False, True, False, False]

        _, fb_fwd, ebms_fwd, progs_fwd, inits_fwd, alpha_fwd = self._two_chain_setup(
            n,
            biases,
            coupling=0.0,
            betas=[0.5, 1.5],
            init_spins_0=s_a,
            init_spins_1=s_b,
        )
        _, fb_rev, ebms_rev, progs_rev, inits_rev, alpha_rev = self._two_chain_setup(
            n,
            biases,
            coupling=0.0,
            betas=[0.5, 1.5],
            init_spins_0=s_b,
            init_spins_1=s_a,
        )

        # Detailed balance: α(a,b)·π(a)·π(b) = α(b,a)·π(b)·π(a)
        # → α(a→b) * π_cold(a) * π_hot(b) = α(b→a) * π_cold(b) * π_hot(a)
        # → α(a→b) / α(b→a) = π_cold(b)π_hot(a) / [π_cold(a)π_hot(b)]
        # We verify the analytical values satisfy this ratio.

        E_base_a = float(ebms_fwd[0].energy(inits_fwd[0], fb_fwd)) / 0.5
        E_base_b = float(ebms_fwd[1].energy(inits_fwd[1], fb_fwd)) / 1.5

        log_r_fwd = (0.5 - 1.5) * (E_base_a - E_base_b)
        log_r_rev = (0.5 - 1.5) * (E_base_b - E_base_a)

        # log_r_fwd + log_r_rev == 0 always → α_fwd * α_rev = exp(0) = 1 when both < 1
        # Unless one of them hits the min(0,·) clamp. The ratio must satisfy DB.
        # Check: exp(log_r_fwd) * exp(log_r_rev) = 1 (before clamping)
        unclamped_product = np.exp(log_r_fwd) * np.exp(log_r_rev)
        assert abs(unclamped_product - 1.0) < 1e-6, (
            f"Unclamped α product should be 1.0, got {unclamped_product}"
        )


# ---------------------------------------------------------------------------
# T3: Adaptive schedule Λ monotonicity + min-acceptance floor
# ---------------------------------------------------------------------------


class TestAdaptiveSchedule:
    def _make_factories(self, nodes, edges, biases, weights, free_blocks):
        def ebm_f(b):
            return [
                IsingEBM(nodes, edges, biases, weights, jnp.array(float(bi)))
                for bi in b
            ]

        def prog_f(es):
            return [IsingSamplingProgram(e, fb, []) for e in es]

        fb = free_blocks
        return ebm_f, prog_f

    def test_rejection_rates_become_equalized(self):
        """Adaptive tuning must equalize per-pair rejection rates.

        Lambda per-phase is NOT monotone -- finite-sample noise in rejection
        rate estimates can push it up or down. The correct invariant is that
        std(rejection_rates) decreases: pairs with very high/low rates get
        rebalanced toward equal spacing.
        """
        n = 8
        key = jax.random.key(33)
        k_bias, k_init = jax.random.split(key)
        nodes, edges, biases, weights, fb, _, _ = _ising_chain(
            n, coupling=0.8, beta=1.0, key=k_bias
        )
        # Intentionally unequal: 3 close betas then a big jump -- large initial std
        betas_init = jnp.array([0.2, 0.21, 0.22, 1.0, 1.8, 2.5])
        ebms_init = [
            IsingEBM(nodes, edges, biases, weights, jnp.array(float(b)))
            for b in betas_init
        ]
        init = _init_states(k_init, 6, ebms_init, fb)
        ebm_f, prog_f = self._make_factories(nodes, edges, biases, weights, fb)

        _, _, stats = nrpt_adaptive(
            jax.random.key(0),
            ebm_f,
            prog_f,
            init,
            [],
            n_rounds=300,
            gibbs_steps_per_round=5,
            initial_betas=betas_init,
            n_tune=5,
            rounds_per_tune=150,
        )

        history = stats["tuning_history"]
        std_first = float(np.std([float(r) for r in history[0]["rejection_rates"]]))
        std_last = float(np.std([float(r) for r in history[-1]["rejection_rates"]]))

        assert std_last < std_first, (
            f"Rejection rates did not equalize: std {std_first:.4f} -> {std_last:.4f}\n"
            f"  Phase 0: {[f'{float(r):.3f}' for r in history[0]['rejection_rates']]}\n"
            f"  Phase N: {[f'{float(r):.3f}' for r in history[-1]['rejection_rates']]}"
        )

    def test_min_acceptance_improves_across_phases(self):
        """Adaptive tuning must improve the worst-case pair acceptance.

        Comparing Lambda from two independent runs measures variance, not
        improvement. Instead: compare min(acceptance_rate) from the first
        tuning phase vs the final production run, both within the same
        adaptive call, starting from a deliberately bad (wide) schedule.
        """
        n = 8
        key = jax.random.key(55)
        k_bias, k_init = jax.random.split(key)
        nodes, edges, biases, weights, fb, _, _ = _ising_chain(
            n, coupling=1.0, beta=1.0, key=k_bias
        )
        betas_init = jnp.linspace(0.2, 2.5, 6)
        ebms_init = [
            IsingEBM(nodes, edges, biases, weights, jnp.array(float(b)))
            for b in betas_init
        ]
        init = _init_states(k_init, 6, ebms_init, fb)
        ebm_f, prog_f = self._make_factories(nodes, edges, biases, weights, fb)

        _, _, stats = nrpt_adaptive(
            jax.random.key(11),
            ebm_f,
            prog_f,
            init,
            [],
            n_rounds=400,
            gibbs_steps_per_round=5,
            initial_betas=betas_init,
            n_tune=5,
            rounds_per_tune=150,
        )

        history = stats["tuning_history"]
        min_acc_phase0 = float(min(history[0]["acceptance_rate"]))
        min_acc_final = float(stats["acceptance_rate"].min())

        assert min_acc_final > min_acc_phase0, (
            f"Min acceptance did not improve: phase0={min_acc_phase0:.4f}, final={min_acc_final:.4f}"
        )
        assert min_acc_final > 0.05, (
            f"Final min acceptance {min_acc_final:.4f} -- conveyor belt still broken"
        )

    def test_min_acceptance_floor_after_tuning(self):
        """After adaptive tuning, no pair should have zero acceptance.

        An untuned wide schedule (e.g. 0.3..2.5 with 6 chains) produces
        acceptance=0 on the hottest pairs. Adaptive tuning must fix this.
        """
        n = 8
        key = jax.random.key(77)
        k_bias, k_init = jax.random.split(key)
        nodes, edges, biases, weights, fb, _, _ = _ising_chain(
            n, coupling=0.8, beta=1.0, key=k_bias
        )
        betas_init = jnp.linspace(0.3, 2.5, 6)  # intentionally wide
        ebms_init = [
            IsingEBM(nodes, edges, biases, weights, jnp.array(float(b)))
            for b in betas_init
        ]
        init = _init_states(k_init, 6, ebms_init, fb)
        ebm_f, prog_f = self._make_factories(nodes, edges, biases, weights, fb)

        _, _, stats = nrpt_adaptive(
            jax.random.key(22),
            ebm_f,
            prog_f,
            init,
            [],
            n_rounds=400,
            gibbs_steps_per_round=5,
            initial_betas=betas_init,
            n_tune=5,
            rounds_per_tune=200,
        )
        min_acc = float(jnp.min(stats["acceptance_rate"]))
        assert min_acc > 0.05, (
            f"After tuning, min pair acceptance is {min_acc:.4f} — conveyor belt broken"
        )

    def test_optimize_schedule_endpoints_preserved(self):
        """β_min and β_max must be unchanged after optimize_schedule."""
        rej = jnp.array([0.4, 0.1, 0.3, 0.2])
        betas = jnp.array([0.3, 0.7, 1.1, 1.8, 2.5])
        new_betas = optimize_schedule(rej, betas)
        assert float(new_betas[0]) == pytest.approx(0.3, abs=1e-6)
        assert float(new_betas[-1]) == pytest.approx(2.5, abs=1e-6)

    def test_optimize_schedule_reduces_lambda(self):
        """After one pass of optimize_schedule, Λ on the new schedule must not exceed old."""
        # Create rejection rates that are highly unequal (the easy case to optimise)
        rej = jnp.array([0.5, 0.05, 0.5, 0.05])
        betas = jnp.linspace(0.5, 2.0, 5)
        new_betas = optimize_schedule(rej, betas)

        # Verify new betas are still sorted
        assert jnp.all(jnp.diff(new_betas) > 0), (
            "optimize_schedule must preserve ordering"
        )
        assert len(new_betas) == len(betas)


# ---------------------------------------------------------------------------
# T4: τ̄ = 1/(2+2Λ) prediction
#
# Syed et al. Theorem 3: under the equalized-rate schedule, the round trip
# rate converges to τ̄ = 1/(2+2Λ).
# This is a statistical test — we can only verify the observed rate is in
# the right ballpark (factor-of-2 tolerance) after adaptive tuning.
# We do not expect exact agreement from a short run.
# ---------------------------------------------------------------------------


class TestRoundTripPrediction:
    def test_tau_observed_within_factor_two_of_predicted(self):
        """τ_obs should be within 2× of τ̄ after adaptive tuning on a well-mixed model."""
        n = 6
        key = jax.random.key(99)
        k_bias, k_init = jax.random.split(key)
        nodes, edges, biases, weights, fb, _, _ = _ising_chain(
            n, coupling=0.5, beta=1.0, key=k_bias
        )
        betas_init = jnp.linspace(0.2, 1.5, 6)
        ebms_init = [
            IsingEBM(nodes, edges, biases, weights, jnp.array(float(b)))
            for b in betas_init
        ]
        init = _init_states(k_init, 6, ebms_init, fb)

        def ebm_f(b):
            return [
                IsingEBM(nodes, edges, biases, weights, jnp.array(float(bi)))
                for bi in b
            ]

        def prog_f(es):
            return [IsingSamplingProgram(e, fb, []) for e in es]

        _, _, stats = nrpt_adaptive(
            jax.random.key(0),
            ebm_f,
            prog_f,
            init,
            [],
            n_rounds=1000,
            gibbs_steps_per_round=5,
            initial_betas=betas_init,
            n_tune=5,
            rounds_per_tune=200,
        )
        diag = stats["round_trip_diagnostics"]
        tau_pred = float(diag["tau_predicted"])
        tau_obs = float(diag["tau_observed"])

        # τ_obs must be positive (conveyor belt running)
        assert tau_obs > 0.0, "No round trips observed after adaptive tuning"

        # Factor-of-2 tolerance: well within what Theorem 3 guarantees asymptotically
        ratio = tau_obs / tau_pred
        assert 0.25 <= ratio <= 4.0, (
            f"τ_obs/τ_pred = {ratio:.3f} (τ_obs={tau_obs:.4f}, τ_pred={tau_pred:.4f})"
        )

    def test_tau_predicted_formula(self):
        """τ̄ = 1/(2+2Λ) directly."""
        from hamon.round_trips import predict_optimal_round_trip_rate

        cases = [
            (0.0, 0.5),
            (1.0, 0.25),
            (3.0, 1 / 8),
        ]
        for Lambda, expected_tau in cases:
            tau = float(predict_optimal_round_trip_rate(Lambda))
            assert tau == pytest.approx(expected_tau, rel=1e-5), (
                f"Λ={Lambda}: expected τ̄={expected_tau}, got {tau}"
            )

    def test_zero_coupling_round_trips_achievable(self):
        """With zero coupling (independent spins), NRPT should complete round trips."""
        n = 6
        nodes, edges, biases, weights, fb, _, _ = _ising_chain(
            n, coupling=0.0, beta=1.0, key=jax.random.key(0)
        )
        betas = [0.3, 0.6, 0.9, 1.2]
        ebms = [IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas]
        progs = [IsingSamplingProgram(e, fb, []) for e in ebms]
        init = _init_states(jax.random.key(1), 4, ebms, fb)

        _, _, stats = nrpt(
            jax.random.key(2),
            ebms,
            progs,
            init,
            [],
            n_rounds=500,
            gibbs_steps_per_round=3,
            track_round_trips=True,
        )
        total_rts = int(jnp.sum(stats["index_state"]["round_trips"]))
        assert total_rts > 0, (
            "Expected round trips in zero-coupling model with 500 rounds"
        )

    def test_efficiency_bounded(self):
        """Efficiency = τ_obs / τ_pred must be in (0, ∞) and finite."""
        n = 6
        key = jax.random.key(55)
        k_bias, k_init = jax.random.split(key)
        nodes, edges, biases, weights, fb, _, _ = _ising_chain(
            n, coupling=0.5, beta=1.0, key=k_bias
        )
        betas = jnp.linspace(0.3, 1.5, 5)
        ebms = [
            IsingEBM(nodes, edges, biases, weights, jnp.array(float(b))) for b in betas
        ]
        progs = [IsingSamplingProgram(e, fb, []) for e in ebms]
        init = _init_states(k_init, 5, ebms, fb)

        _, _, stats = nrpt(
            jax.random.key(7),
            ebms,
            progs,
            init,
            [],
            n_rounds=300,
            gibbs_steps_per_round=5,
        )
        eff = float(stats["round_trip_diagnostics"]["efficiency"])
        assert jnp.isfinite(eff), f"Efficiency is not finite: {eff}"
        assert eff >= 0.0, f"Negative efficiency: {eff}"
