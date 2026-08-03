"""Tests for the double-well (φ⁴) model and slice-sampling-within-Gibbs.

Guards:
- Exactness of the slice kernel against 1-D and 2-D quadrature (the φ⁴
  conditional has no closed form, so numerical integration is the ground
  truth). Never calibrate or benchmark on an unverified sampler.
- Energy convention matches the documented closed form.
- Bimodality: a tempered NRPT ladder visits both wells of a ferromagnetic φ⁴
  lattice; a single cold chain started in one well stays there — the continuous
  multimodal case NRPT exists for.
- Chain masking stays bit-identical with a slice sampler in the loop. The
  slice kernel's data-dependent while_loop is the one place masking's
  prefix-stability argument could break: draws are keyed by
  fold_in(key, iteration), so a padding lane needing more iterations must not
  perturb live lanes.
- The improper-β=0 guard applies (unbounded state space).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hamon import Block, GaussianNode
from hamon.block_sampling import SamplingSchedule, sample_states
from hamon.models import DoubleWellEBM, DoubleWellSamplingProgram, double_well_init
from hamon.nrpt import nrpt
from hamon.observers import NRPTStateObserver


def _pair_model(a=1.2, h=(0.4, 0.0), c=-1.5, beta=1.0):
    """Two coupled double-well sites (the smallest quadrature-checkable model).

    With c = 0 the sites decouple, giving the single-site check for free."""
    nodes = [GaussianNode(), GaussianNode()]
    ebm = DoubleWellEBM(
        nodes,
        [(nodes[0], nodes[1])],
        jnp.full(2, a, dtype=jnp.float32),
        jnp.asarray(h, dtype=jnp.float32),
        jnp.asarray([c], dtype=jnp.float32),
        jnp.array(beta),
    )
    blocks = [Block([nodes[0]]), Block([nodes[1]])]
    program = DoubleWellSamplingProgram(ebm, blocks, [])
    return nodes, ebm, program, blocks


def _quad_grid(a, h1, h2, c, beta, lim=3.0, m=481):
    """2-D quadrature of p(x1,x2) ∝ exp(−β[Σ a x⁴ − 2a x² − h x + c x1 x2])."""
    x = np.linspace(-lim, lim, m)
    x1, x2 = np.meshgrid(x, x, indexing="ij")

    def u(xx, hh):
        return a * xx**4 - 2.0 * a * xx**2 - hh * xx

    logw = -beta * (u(x1, h1) + u(x2, h2) + c * x1 * x2)
    p = np.exp(logw - logw.max())
    p /= p.sum()
    return x, p


def _draw(program, blocks, ebm, n_samples, thin=5, seed=1):
    init = double_well_init(jax.random.key(0), ebm, blocks)
    raw = sample_states(
        jax.random.key(seed),
        program,
        SamplingSchedule(400, n_samples, thin),
        init,
        [],
        [Block([n for b in blocks for n in b.nodes])],
        device=None,
    )
    return np.asarray(raw[0], dtype=np.float64)  # (n_samples, 2): node order


class TestSliceExactness:
    def test_single_site_matches_quadrature(self):
        """c = 0 decouples the pair: each site is an independent 1-D tilted
        double well, checked against quadrature via histogram TV distance."""
        a, h1, beta = 1.2, 0.4, 1.0
        _nodes, ebm, program, blocks = _pair_model(a=a, h=(h1, 0.0), c=0.0, beta=beta)
        n_samples = 40_000
        s = _draw(program, blocks, ebm, n_samples)

        x, p2 = _quad_grid(a, h1, 0.0, 0.0, beta)
        p1 = p2.sum(axis=1)  # exact marginal of site 1 (tilted)
        edges = np.linspace(-3.0, 3.0, 61)
        p_bins = np.add.reduceat(p1, np.searchsorted(x, edges[:-1]))
        p_bins /= p_bins.sum()
        emp, _ = np.histogram(s[:, 0], bins=edges)
        emp = emp / emp.sum()
        tv = 0.5 * np.abs(emp - p_bins).sum()
        assert tv < 0.05, tv

        # The tilt h > 0 must bias toward the +1 well (sign sanity on h).
        assert s[:, 0].mean() > 0.2

    def test_coupled_pair_matches_quadrature(self):
        """c < 0 couples the wells; check the marginal AND the cross moment
        E[x1 x2] against 2-D quadrature."""
        a, h1, c, beta = 1.2, 0.4, -1.5, 1.0
        _nodes, ebm, program, blocks = _pair_model(a=a, h=(h1, 0.0), c=c, beta=beta)
        n_samples = 60_000
        s = _draw(program, blocks, ebm, n_samples)

        x, p = _quad_grid(a, h1, 0.0, c, beta)
        p1 = p.sum(axis=1)
        edges = np.linspace(-3.0, 3.0, 41)
        p_bins = np.add.reduceat(p1, np.searchsorted(x, edges[:-1]))
        p_bins /= p_bins.sum()
        emp, _ = np.histogram(s[:, 0], bins=edges)
        emp = emp / emp.sum()
        tv = 0.5 * np.abs(emp - p_bins).sum()
        assert tv < 0.05, tv

        x1x2_exact = float((p * np.outer(x, np.ones_like(x)) * x[None, :]).sum())
        x1x2_emp = float((s[:, 0] * s[:, 1]).mean())
        # ferromagnetic coupling: strongly positive cross moment
        assert x1x2_exact > 0.5
        assert np.isclose(x1x2_emp, x1x2_exact, atol=0.05), (x1x2_emp, x1x2_exact)

    def test_energy_matches_closed_form(self):
        a, h1, c, beta = 1.2, 0.4, -1.5, 0.7
        _nodes, ebm, _program, blocks = _pair_model(a=a, h=(h1, 0.0), c=c, beta=beta)
        x = np.array([0.3, -1.4], dtype=np.float32)
        state = [jnp.asarray([x[0]]), jnp.asarray([x[1]])]
        expected = beta * (
            (a * x**4 - 2 * a * x**2).sum() - h1 * x[0] + c * x[0] * x[1]
        )
        got = float(ebm.energy(state, blocks))
        assert np.isclose(got, expected, rtol=1e-5, atol=1e-5), (got, expected)


def _lattice_model(L=5, a=1.0, c=-0.7, beta=1.0):
    n = L * L

    def idx(r, col):
        return (r % L) * L + (col % L)

    edges_ix = []
    for r in range(L):
        for col in range(L):
            edges_ix.append((idx(r, col), idx(r, col + 1)))
            edges_ix.append((idx(r, col), idx(r + 1, col)))
    nodes = [GaussianNode() for _ in range(n)]
    ebm = DoubleWellEBM(
        nodes,
        [(nodes[i], nodes[j]) for i, j in edges_ix],
        jnp.full(n, a, dtype=jnp.float32),
        jnp.zeros(n, dtype=jnp.float32),
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
    program = DoubleWellSamplingProgram(ebm, blocks, [])
    return nodes, ebm, program, blocks


class TestDoubleWellMultimodality:
    def test_tempered_ladder_visits_both_wells_single_chain_does_not(self):
        """The φ⁴ ferromagnet at cold β is bimodal (±1 ordered wells). A
        tempered NRPT cold chain must visit both signs of the magnetization; a
        single cold chain started in one well must stay there. This is the
        continuous multimodal case the tempering machinery exists for."""
        nodes, ebm, program, blocks = _lattice_model()
        nc = 6
        betas = jnp.linspace(0.1, 1.0, nc)
        inits = [
            double_well_init(jax.random.key(30 + i), ebm.with_beta(b), blocks)
            for i, b in enumerate(betas)
        ]
        _, stats = nrpt(
            jax.random.key(3),
            ebm,
            program,
            inits,
            [],
            1500,
            2,
            betas=betas,
            observer=NRPTStateObserver(chain_indices=(-1,)),
            track_round_trips=False,
            device=None,
        )
        obs = stats["observations"]  # per free block: (rounds, 1, block)
        cold = np.concatenate([np.asarray(o[:, 0]) for o in obs], axis=1)
        mag = cold.mean(axis=1)
        assert mag.min() < -0.25 and mag.max() > 0.25, (mag.min(), mag.max())

        # Single cold chain from one well: magnetization sign frozen.
        init = double_well_init(jax.random.key(9), ebm, blocks)
        # force the all-plus well
        init = [jnp.abs(st) for st in init]
        raw = sample_states(
            jax.random.key(4),
            program,
            SamplingSchedule(0, 1500, 1),
            init,
            [],
            [Block(nodes)],
            device=None,
        )
        mag_single = np.asarray(raw[0]).mean(axis=1)
        assert mag_single.min() > 0.25, mag_single.min()


class TestSliceMaskingAndGuards:
    def test_chain_masking_bit_identical_with_slice_sampler(self):
        """The slice kernel's while_loop trip counts are data-dependent — the
        one mechanism that could break masking's prefix-stability. Draws are
        keyed per (site, iteration), so the padded run must stay bitwise
        identical on the live prefix."""
        _nodes, ebm, program, blocks = _lattice_model(L=3)
        nc = 5
        betas = jnp.linspace(0.1, 1.0, nc)
        inits = [
            double_well_init(jax.random.key(40 + i), ebm.with_beta(b), blocks)
            for i, b in enumerate(betas)
        ]
        key = jax.random.key(5)
        kw = {"betas": betas, "track_round_trips": True, "device": None}
        s_a, st_a = nrpt(key, ebm, program, inits, [], 40, 2, **kw)
        s_b, st_b = nrpt(key, ebm, program, inits, [], 40, 2, pad_chains_to=9, **kw)
        for chain_a, chain_b in zip(s_a, s_b):
            for st1, st2 in zip(chain_a, chain_b):
                assert np.array_equal(np.asarray(st1), np.asarray(st2))
        for k2 in ("accepted", "attempted"):
            assert np.array_equal(np.asarray(st_a[k2]), np.asarray(st_b[k2]))

    def test_beta_zero_guard(self):
        _nodes, ebm, program, blocks = _lattice_model(L=3)
        assert ebm.proper_at_beta_zero is False
        nc = 4
        betas = jnp.linspace(0.0, 1.0, nc)
        inits = [
            double_well_init(
                jax.random.key(i), ebm.with_beta(jnp.maximum(b, 0.1)), blocks
            )
            for i, b in enumerate(betas)
        ]
        with pytest.raises(ValueError, match="beta"):
            nrpt(jax.random.key(0), ebm, program, inits, [], 5, 1, betas=betas)
