"""Log normalizing constant via thermodynamic integration.

Validates:
  1. thermodynamic_integration against the analytic independent-spins result
     log Z(β)/Z(0) = Σ log cosh(β b_i)   (exact, no sampling).
  2. nrpt_log_normalizing_constant end-to-end against brute-force enumeration
     of a small Ising model's partition function.

Design constraints mirror test_nrpt_correctness: models ≤ 10 spins so the exact
log Z is enumerable in well under a second, and everything goes through the
public API.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hamon import Block, NRPTEnergyObserver, SpinNode
from hamon.models import IsingEBM, IsingSamplingProgram, hinton_init
from hamon.nrpt import nrpt_adaptive
from hamon.round_trips import (
    nrpt_log_normalizing_constant,
    thermodynamic_integration,
)


# ---------------------------------------------------------------------------
# 1. Analytic check: independent spins
# ---------------------------------------------------------------------------


class TestThermodynamicIntegrationAnalytic:
    def test_independent_spins_trapezoid(self):
        """For independent spins with biases b_i and base energy E = -Σ b_i s_i,
        μ(β) = -Σ b_i tanh(β b_i) and -∫_0^1 μ dβ = Σ log cosh(b_i)."""
        b = jnp.array([0.5, -0.8, 1.2, 0.3])
        betas = jnp.linspace(0.0, 1.0, 256)
        # μ(β_k) on a fine grid.
        mu = -jnp.sum(b[None, :] * jnp.tanh(betas[:, None] * b[None, :]), axis=1)

        est = float(thermodynamic_integration(betas, mu, method="trapezoid"))
        expected = float(jnp.sum(jnp.log(jnp.cosh(b))))
        assert est == pytest.approx(expected, rel=1e-3)

    def test_riemann_close_to_trapezoid(self):
        b = jnp.array([0.4, -0.6, 1.0])
        betas = jnp.linspace(0.0, 1.0, 512)
        mu = -jnp.sum(b[None, :] * jnp.tanh(betas[:, None] * b[None, :]), axis=1)
        trap = float(thermodynamic_integration(betas, mu, method="trapezoid"))
        riem = float(thermodynamic_integration(betas, mu, method="riemann"))
        # On a fine grid the two quadratures agree closely.
        assert riem == pytest.approx(trap, abs=1e-2)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="trapezoid"):
            thermodynamic_integration(
                jnp.linspace(0.0, 1.0, 4), jnp.zeros(4), method="simpson"
            )


# ---------------------------------------------------------------------------
# 2. End-to-end: brute-force enumeration
# ---------------------------------------------------------------------------


def _brute_force_log_z(ebm_beta1: IsingEBM, nodes, beta_cold: float) -> float:
    """Exact log Z(beta_cold) = log Σ_x exp(-beta_cold · E_base(x)).

    ebm_beta1 is the β=1 EBM, so ebm_beta1.energy(x) == E_base(x).
    """
    n = len(nodes)
    obs_block = Block(nodes)
    energies = np.empty(2**n)
    for i in range(2**n):
        bits = jnp.array([(i >> k) & 1 for k in range(n)], dtype=jnp.bool_)
        energies[i] = float(ebm_beta1.energy([bits], [obs_block]))
    a = -beta_cold * energies
    m = a.max()
    return float(m + np.log(np.sum(np.exp(a - m))))


class TestLogZEndToEnd:
    def test_logz_matches_enumeration(self):
        n = 6
        n_chains = 10
        key = jax.random.key(2024)
        k_bias, k_init, k_run = jax.random.split(key, 3)

        nodes = [SpinNode() for _ in range(n)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
        biases = jax.random.uniform(k_bias, (n,), minval=-1.0, maxval=1.0)
        weights = jnp.ones(n - 1) * 0.3  # mild coupling → smooth μ(β)
        # β=1 template EBM; ladder spans the reference (β=0) to the target (β=1).
        ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
        free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
        program = IsingSamplingProgram(ebm, free_blocks, [])

        betas0 = jnp.linspace(0.0, 1.0, n_chains)
        keys = jax.random.split(k_init, n_chains)
        init = [hinton_init(keys[c], ebm, free_blocks, ()) for c in range(n_chains)]

        obs = NRPTEnergyObserver(n_chains)
        _, stats = nrpt_adaptive(
            k_run,
            init_states=init,
            clamp_state=[],
            n_rounds=3000,
            gibbs_steps_per_round=5,
            initial_betas=betas0,
            n_tune=4,
            rounds_per_tune=200,
            ebm=ebm,
            program=program,
            observer=obs,
        )

        # The observer carry must be threaded back out.
        assert "observer_carry" in stats
        beta_cold = float(np.asarray(stats["betas"])[-1])
        assert beta_cold == pytest.approx(1.0, abs=1e-6)  # endpoint preserved

        log_z0 = n * np.log(2.0)  # β=0 reference is uniform over 2**n states
        est = float(nrpt_log_normalizing_constant(stats, log_z0=log_z0))
        exact = _brute_force_log_z(ebm, nodes, beta_cold)

        assert est == pytest.approx(exact, abs=max(0.5, 0.1 * abs(exact))), (
            f"log Z estimate {est:.4f} vs exact {exact:.4f}"
        )

    def test_missing_observer_carry_raises(self):
        with pytest.raises(ValueError, match="observer_carry"):
            nrpt_log_normalizing_constant({"betas": jnp.linspace(0.0, 1.0, 4)})
