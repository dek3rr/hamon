"""Tests for the reference-annealing path (AnnealedEBM + NRPT affine mode).

The path is E_β = E_ref + β·(E_target − E_ref): β = 0 *is* the reference, so an
unbounded-state-space target can be tempered from a proper Gaussian reference
and the ladder may start at exactly β = 0.

Guards:
- Energy is the convex combination of reference and target energies.
- The decisive check: an all-Gaussian annealed ladder (diagonal Gaussian
  reference → lattice GMRF target) has a closed form at EVERY rung —
  N(P_β⁻¹·βh, P_β⁻¹) with P_β = (1−β)R + βP — so a long NRPT run's per-rung
  marginals verify the affine kernel AND the Δ = E₁−E₀ swap energies
  end-to-end, starting from β = 0. Wrong swap energies (using E₁, under which
  the reference term fails to cancel) would break the joint invariance and
  drift these marginals.
- Chain masking stays bit-identical in affine mode.
- The factory route and energy_delta_fn are rejected (their swap math assumes
  the linear path); an improper *reference* is still guarded at β = 0.
- The φ⁴ target annealed from a Gaussian reference runs under the slice
  sampler (mixed quadratic + polynomial interactions) from β = 0.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hamon import Block, GaussianNode
from hamon.models import (
    AnnealedEBM,
    DoubleWellEBM,
    DoubleWellSamplingProgram,
    GaussianEBM,
    GaussianSamplingProgram,
    gaussian_init,
)
from hamon.nrpt import nrpt
from hamon.observers import NRPTStateObserver
from hamon.tuning import tune_schedule


def _annealed_gaussian(L=4, r=2.0, d=4.5, c=-1.0, seed=0):
    """Diagonal-Gaussian reference (precision r·I) annealed into a lattice GMRF
    target (precision P, linear h) over the SAME nodes."""
    n = L * L

    def idx(row, col):
        return (row % L) * L + (col % L)

    edges_ix = []
    for row in range(L):
        for col in range(L):
            edges_ix.append((idx(row, col), idx(row, col + 1)))
            edges_ix.append((idx(row, col), idx(row + 1, col)))
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, 0.5, size=n).astype(np.float32)

    P = np.zeros((n, n))
    np.fill_diagonal(P, d)
    for a, b in edges_ix:
        P[a, b] += c
        P[b, a] += c
    R = r * np.eye(n)

    nodes = [GaussianNode() for _ in range(n)]
    reference = GaussianEBM(
        nodes,
        [],
        jnp.full(n, r, dtype=jnp.float32),
        jnp.zeros(n, dtype=jnp.float32),
        jnp.zeros(0, dtype=jnp.float32),
        jnp.array(1.0),
    )
    target = GaussianEBM(
        nodes,
        [(nodes[a], nodes[b]) for a, b in edges_ix],
        jnp.full(n, d, dtype=jnp.float32),
        jnp.asarray(h),
        jnp.full(len(edges_ix), c, dtype=jnp.float32),
        jnp.array(1.0),
    )
    annealed = AnnealedEBM(reference, target, jnp.array(1.0))
    even = [
        nodes[row * L + col]
        for row in range(L)
        for col in range(L)
        if (row + col) % 2 == 0
    ]
    odd = [
        nodes[row * L + col]
        for row in range(L)
        for col in range(L)
        if (row + col) % 2 == 1
    ]
    blocks = [Block(even), Block(odd)]
    program = GaussianSamplingProgram(annealed, blocks, [])
    # node index of each position in [even..., odd...] concatenation order
    pos = {id(nd): i for i, nd in enumerate(nodes)}
    obs_perm = np.asarray([pos[id(nd)] for b in blocks for nd in b.nodes])
    return nodes, reference, target, annealed, program, blocks, P, R, h, obs_perm


class TestAnnealedEnergy:
    def test_energy_is_convex_combination(self):
        (nodes, reference, target, annealed, program, blocks, P, R, h, _) = (
            _annealed_gaussian()
        )
        rng = np.random.default_rng(1)
        x = rng.normal(size=len(nodes)).astype(np.float32)
        pos = {id(nd): i for i, nd in enumerate(nodes)}
        state = [
            jnp.asarray([x[pos[id(nd)]] for nd in b.nodes], dtype=jnp.float32)
            for b in blocks
        ]
        for beta in (0.0, 0.3, 1.0):
            e_ref = float(reference.energy(state, blocks))
            e_tgt = float(target.energy(state, blocks))
            got = float(annealed.with_beta(beta).energy(state, blocks))
            expected = (1.0 - beta) * e_ref + beta * e_tgt
            assert np.isclose(got, expected, rtol=1e-5, atol=1e-4), (beta, got)

    def test_beta_zero_member_is_the_reference(self):
        (nodes, reference, target, annealed, *_) = _annealed_gaussian()
        # The target's OWN tempered family is improper at beta=0 (weights -> 0,
        # flat over R^n) — but the annealed family's beta=0 member is the
        # reference at FULL weight, a proper PD Gaussian. So annealing is
        # exactly what makes beta=0 legal for an unbounded-state-space target.
        assert target.proper_at_beta_zero is False
        assert annealed.proper_at_beta_zero is True
        assert annealed.beta_affine is True


class TestAnnealedLadderExactness:
    def test_every_rung_matches_closed_form_from_beta_zero(self):
        """Run a β ∈ [0, 1] ladder and check every rung's marginal mean and
        per-site variance against N(P_β⁻¹·βh, P_β⁻¹), P_β = (1−β)R + βP.
        This is the end-to-end proof of the affine kernel and the Δ-swap
        energies: wrong swaps break joint invariance and drift these marginals.
        """
        (nodes, reference, target, annealed, program, blocks, P, R, h, obs_perm) = (
            _annealed_gaussian()
        )
        n = len(nodes)
        nc = 4
        betas = jnp.linspace(0.0, 1.0, nc)
        inits = [
            gaussian_init(
                jax.random.key(10 + i), target.with_beta(jnp.maximum(b, 0.5)), blocks
            )
            for i, b in enumerate(betas)
        ]
        n_rounds = 4000
        _, stats = nrpt(
            jax.random.key(2),
            annealed,
            program,
            inits,
            [],
            n_rounds,
            2,
            betas=betas,
            observer=NRPTStateObserver(chain_indices=tuple(range(nc))),
            track_round_trips=False,
            device=None,
        )
        obs = stats["observations"]  # per block: (rounds, nc, block)
        all_states = np.concatenate([np.asarray(o) for o in obs], axis=2)
        # undo the block concatenation ordering -> node order
        inv = np.empty(n, dtype=int)
        inv[obs_perm] = np.arange(n)
        all_states = all_states[:, :, inv]
        burn = 500
        for k, beta in enumerate(np.asarray(betas)):
            P_b = (1.0 - beta) * R + beta * P
            cov = np.linalg.inv(P_b)
            mu = cov @ (beta * h)
            s = all_states[burn:, k, :].astype(np.float64)
            n_eff = s.shape[0] / 10.0
            mean_tol = 6.0 * np.sqrt(np.diag(cov) / n_eff)
            assert np.all(np.abs(s.mean(0) - mu) < mean_tol), (
                k,
                (np.abs(s.mean(0) - mu) / mean_tol).max(),
            )
            var_ratio = s.var(0) / np.diag(cov)
            assert np.all((var_ratio > 0.8) & (var_ratio < 1.25)), (
                k,
                var_ratio.min(),
                var_ratio.max(),
            )


class TestAnnealedGuardsAndMasking:
    def test_chain_masking_bit_identical_affine(self):
        (nodes, reference, target, annealed, program, blocks, P, R, h, _) = (
            _annealed_gaussian()
        )
        nc = 5
        betas = jnp.linspace(0.0, 1.0, nc)
        inits = [
            gaussian_init(
                jax.random.key(20 + i), target.with_beta(jnp.maximum(b, 0.5)), blocks
            )
            for i, b in enumerate(betas)
        ]
        key = jax.random.key(6)
        kw = dict(betas=betas, track_round_trips=True, device=None)
        s_a, st_a = nrpt(key, annealed, program, inits, [], 30, 2, **kw)
        s_b, st_b = nrpt(
            key, annealed, program, inits, [], 30, 2, pad_chains_to=9, **kw
        )
        for chain_a, chain_b in zip(s_a, s_b):
            for st1, st2 in zip(chain_a, chain_b):
                assert np.array_equal(np.asarray(st1), np.asarray(st2))
        for k2 in ("accepted", "attempted"):
            assert np.array_equal(np.asarray(st_a[k2]), np.asarray(st_b[k2]))

    def test_factory_route_rejected(self):
        (nodes, reference, target, annealed, program, blocks, P, R, h, _) = (
            _annealed_gaussian()
        )
        nc = 3
        betas = jnp.linspace(0.0, 1.0, nc)
        ebms = [annealed.with_beta(b) for b in betas]
        programs = [program.with_ebm(e) for e in ebms]
        inits = [
            gaussian_init(
                jax.random.key(i), target.with_beta(jnp.maximum(b, 0.5)), blocks
            )
            for i, b in enumerate(betas)
        ]
        with pytest.raises(ValueError, match="template"):
            nrpt(jax.random.key(0), ebms, programs, inits, [], 5, 1, betas=betas)

    def test_energy_delta_fn_rejected(self):
        (nodes, reference, target, annealed, program, blocks, P, R, h, _) = (
            _annealed_gaussian()
        )
        nc = 3
        betas = jnp.linspace(0.0, 1.0, nc)
        inits = [
            gaussian_init(
                jax.random.key(i), target.with_beta(jnp.maximum(b, 0.5)), blocks
            )
            for i, b in enumerate(betas)
        ]
        with pytest.raises(ValueError, match="delta"):
            nrpt(
                jax.random.key(0),
                annealed,
                program,
                inits,
                [],
                5,
                1,
                betas=betas,
                energy_delta_fn=lambda old, new: jnp.zeros(nc),
            )


class TestAnnealedPhi4:
    def test_phi4_from_gaussian_reference_runs_from_beta_zero(self):
        """The physics payoff: a φ⁴ target annealed from a proper Gaussian
        reference tempered from EXACTLY β = 0, under the slice sampler (mixed
        quadratic-reference + polynomial-target interactions per site)."""
        from hamon.models import double_well_init

        L = 3
        n = L * L

        def idx(row, col):
            return (row % L) * L + (col % L)

        edges_ix = []
        for row in range(L):
            for col in range(L):
                edges_ix.append((idx(row, col), idx(row, col + 1)))
                edges_ix.append((idx(row, col), idx(row + 1, col)))
        nodes = [GaussianNode() for _ in range(n)]
        reference = GaussianEBM(
            nodes,
            [],
            jnp.full(n, 2.0, dtype=jnp.float32),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros(0, dtype=jnp.float32),
            jnp.array(1.0),
        )
        tgt = DoubleWellEBM(
            nodes,
            [(nodes[i], nodes[j]) for i, j in edges_ix],
            jnp.ones(n, dtype=jnp.float32),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.full(len(edges_ix), -0.6, dtype=jnp.float32),
            jnp.array(1.0),
        )
        annealed = AnnealedEBM(reference, tgt, jnp.array(1.0))
        assert annealed.proper_at_beta_zero is True
        even = [
            nodes[r * L + c] for r in range(L) for c in range(L) if (r + c) % 2 == 0
        ]
        odd = [nodes[r * L + c] for r in range(L) for c in range(L) if (r + c) % 2 == 1]
        blocks = [Block(even), Block(odd)]
        program = DoubleWellSamplingProgram(annealed, blocks, [])
        nc = 5
        betas = jnp.linspace(0.0, 1.0, nc)
        inits = [
            double_well_init(jax.random.key(50 + i), tgt, blocks) for i in range(nc)
        ]
        warm, stats = tune_schedule(
            jax.random.key(3),
            None,
            None,
            inits,
            [],
            n_rounds=60,
            gibbs_steps_per_round=1,
            initial_betas=betas,
            n_tune=2,
            rounds_per_tune=40,
            ebm=annealed,
            program=program,
            device=None,
        )
        rej = np.asarray(stats["rejection_rates"])
        assert np.all(np.isfinite(rej)) and np.all((rej >= 0) & (rej <= 1))
        out_betas = np.asarray(stats["betas"])
        assert out_betas[0] == pytest.approx(0.0)  # the ladder reaches β = 0
        for chain in warm:
            for st in chain:
                assert bool(jnp.all(jnp.isfinite(st)))
