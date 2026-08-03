"""Tests for the Gaussian MRF model (continuous-state block Gibbs).

Guards:
- Exactness: block-Gibbs samples match the closed-form N(P⁻¹h, (βP)⁻¹) —
  mean and full covariance. The GMRF is the one continuous model with an exact
  answer, which is why it is the first continuous family (never calibrate or
  benchmark on an unverified sampler).
- Energy convention: GaussianEBM.energy equals β(½xᵀPx − hᵀx), matching the
  physical-energy convention of the discrete models (π ∝ exp(−E)).
- The improper-β=0 guard: ladders starting at exactly β = 0 are rejected for
  models with proper_at_beta_zero = False, at nrpt and at the tuner entries.
- NRPT template mode runs a Gaussian ladder (β-linear interactions) with
  finite results.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hamon import Block, GaussianNode
from hamon.block_sampling import SamplingSchedule, sample_states
from hamon.models import GaussianEBM, GaussianSamplingProgram, gaussian_init
from hamon.nrpt import nrpt
from hamon.tuning import tune_chains, tune_schedule


def _lattice_gmrf(L=4, d=4.5, c=-1.0, beta=0.8, seed=0):
    """Periodic L×L lattice GMRF: P_ii = d, P_ij = c on nearest-neighbor
    edges (each undirected edge once), random linear term h. Diagonal dominance
    (d > 4|c|) makes P positive definite."""
    n = L * L

    def idx(r, col):
        return (r % L) * L + (col % L)

    edges_ix = []
    for r in range(L):
        for col in range(L):
            edges_ix.append((idx(r, col), idx(r, col + 1)))
            edges_ix.append((idx(r, col), idx(r + 1, col)))
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, 0.5, size=n).astype(np.float32)

    P = np.zeros((n, n))
    np.fill_diagonal(P, d)
    for a, b in edges_ix:
        P[a, b] += c
        P[b, a] += c

    nodes = [GaussianNode() for _ in range(n)]
    node_edges = [(nodes[a], nodes[b]) for a, b in edges_ix]
    ebm = GaussianEBM(
        nodes,
        node_edges,
        jnp.full(n, d, dtype=jnp.float32),
        jnp.asarray(h),
        jnp.full(len(edges_ix), c, dtype=jnp.float32),
        jnp.array(beta),
    )
    even = [
        nodes[r * L + col] for r in range(L) for col in range(L) if (r + col) % 2 == 0
    ]
    odd = [
        nodes[r * L + col] for r in range(L) for col in range(L) if (r + col) % 2 == 1
    ]
    blocks = [Block(even), Block(odd)]
    program = GaussianSamplingProgram(ebm, blocks, [])
    # node index of each observed column, in Block(nodes) order used below
    return nodes, ebm, program, blocks, P, h, beta


class TestGaussianExactness:
    def test_mean_and_covariance_match_closed_form(self):
        nodes, ebm, program, blocks, P, h, beta = _lattice_gmrf()
        n = len(nodes)
        mu = np.linalg.solve(P, h)
        cov = np.linalg.inv(beta * P)

        n_samples = 20_000
        init = gaussian_init(jax.random.key(0), ebm, blocks)
        raw = sample_states(
            jax.random.key(1),
            program,
            SamplingSchedule(300, n_samples, 2),
            init,
            [],
            [Block(nodes)],
            device=None,
        )
        s = np.asarray(raw[0], dtype=np.float64)  # (n_samples, n) node order
        assert s.shape == (n_samples, n)
        assert np.all(np.isfinite(s))

        # Thinned Gibbs samples remain autocorrelated; take a conservative
        # effective sample size for the tolerance.
        n_eff = n_samples / 5.0
        mean_tol = 6.0 * np.sqrt(np.diag(cov) / n_eff)
        assert np.all(np.abs(s.mean(0) - mu) < mean_tol), (
            np.abs(s.mean(0) - mu) / mean_tol
        ).max()

        emp_cov = np.cov(s.T)
        rel_frob = np.linalg.norm(emp_cov - cov) / np.linalg.norm(cov)
        assert rel_frob < 0.15, rel_frob

    def test_energy_matches_closed_form(self):
        nodes, ebm, _program, blocks, P, h, beta = _lattice_gmrf()
        rng = np.random.default_rng(3)
        x = rng.normal(size=len(nodes)).astype(np.float32)
        # split x into the block layout
        pos = {id(nd): i for i, nd in enumerate(nodes)}
        state = [
            jnp.asarray([x[pos[id(nd)]] for nd in b.nodes], dtype=jnp.float32)
            for b in blocks
        ]
        expected = beta * (0.5 * x @ P @ x - h @ x)
        got = float(ebm.energy(state, blocks))
        assert np.isclose(got, expected, rtol=5e-5, atol=5e-4), (got, expected)

    def test_gaussian_init_shapes_and_dtype(self):
        _nodes, ebm, _program, blocks, _P, _h, _beta = _lattice_gmrf()
        init = gaussian_init(jax.random.key(2), ebm, blocks)
        assert len(init) == len(blocks)
        for st, b in zip(init, blocks):
            assert st.shape == (len(b.nodes),)
            assert st.dtype == jnp.float32
            assert bool(jnp.all(jnp.isfinite(st)))


class TestImproperBetaZeroGuard:
    def test_nrpt_rejects_beta_zero_ladder(self):
        _nodes, ebm, program, blocks, _P, _h, _beta = _lattice_gmrf()
        nc = 4
        betas = jnp.linspace(0.0, 1.0, nc)
        inits = [gaussian_init(jax.random.key(i), ebm, blocks) for i in range(nc)]
        with pytest.raises(ValueError, match="beta"):
            nrpt(jax.random.key(0), ebm, program, inits, [], 5, 1, betas=betas)

    def test_tune_chains_rejects_beta_zero_range(self):
        _nodes, ebm, program, blocks, _P, _h, _beta = _lattice_gmrf()

        def init_factory(n_chains, ebms, programs):
            return [
                gaussian_init(jax.random.key(i), ebms[0], blocks)
                for i in range(n_chains)
            ]

        with pytest.raises(ValueError, match="beta"):
            tune_chains(
                jax.random.key(0),
                ebm=ebm,
                program=program,
                init_factory=init_factory,
                beta_range=(0.0, 1.0),
                gibbs_steps_per_round=1,
                max_chains=6,
                rounds_per_probe=10,
            )

    def test_propriety_flags(self):
        """Discrete EBMs stay proper at beta=0 (default); Gaussian is not."""
        from hamon import SpinNode
        from hamon.models import IsingEBM

        s_nodes = [SpinNode() for _ in range(2)]
        ising = IsingEBM(
            s_nodes,
            [(s_nodes[0], s_nodes[1])],
            jnp.zeros(2),
            jnp.ones(1),
            jnp.array(1.0),
        )
        assert ising.proper_at_beta_zero is True
        _nodes, ebm, _program, _blocks, _P, _h, _beta = _lattice_gmrf()
        assert ebm.proper_at_beta_zero is False


class TestGaussianNRPT:
    def test_template_mode_ladder_runs_finite(self):
        _nodes, ebm, program, blocks, _P, _h, _beta = _lattice_gmrf(beta=1.0)
        nc = 5
        betas = jnp.linspace(0.2, 1.0, nc)
        inits = [
            gaussian_init(jax.random.key(10 + i), ebm.with_beta(b), blocks)
            for i, b in enumerate(betas)
        ]
        states, stats = nrpt(
            jax.random.key(1),
            ebm,
            program,
            inits,
            [],
            30,
            1,
            betas=betas,
            device=None,
        )
        for chain in states:
            for st in chain:
                assert bool(jnp.all(jnp.isfinite(st)))
        rej = np.asarray(stats["rejection_rates"])
        assert np.all(np.isfinite(rej)) and np.all((rej >= 0) & (rej <= 1))

    def test_tune_schedule_runs_finite(self):
        nodes, ebm, program, blocks, _P, _h, _beta = _lattice_gmrf(beta=1.0)
        nc = 5
        betas = jnp.linspace(0.2, 1.0, nc)
        inits = [
            gaussian_init(jax.random.key(20 + i), ebm.with_beta(b), blocks)
            for i, b in enumerate(betas)
        ]
        _, stats = tune_schedule(
            jax.random.key(2),
            None,
            None,
            inits,
            [],
            n_rounds=40,
            gibbs_steps_per_round=1,
            initial_betas=betas,
            n_tune=2,
            rounds_per_tune=40,
            ebm=ebm,
            program=program,
            device=None,
        )
        lam = float(np.asarray(stats["rejection_rates"]).sum())
        assert np.isfinite(lam) and lam >= 0
        out_betas = np.asarray(stats["betas"])
        assert out_betas[0] == pytest.approx(0.2) and np.all(np.diff(out_betas) > 0)

        # Continuous (float) samples: the binary-only diagnostics sections must
        # be skipped with a note — not computed as garbage — while ESS (numeric-
        # generic) is still reported.
        from hamon.diagnostics import report_nrpt_diagnostics

        rng = np.random.default_rng(0)
        float_samples = rng.normal(size=(400, len(nodes))).astype(np.float32)
        report = report_nrpt_diagnostics(stats, samples=float_samples)
        assert report.marginal_entropy is None
        assert report.convergence_status is None
        assert report.min_ess is not None and np.isfinite(report.min_ess)
        assert any("non-boolean" in w for w in report.warnings)
