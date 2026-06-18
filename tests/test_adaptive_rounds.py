"""Tests for convergence-driven (adaptive) NRPT tuning in nrpt_adaptive /
discover_chain_count: adaptive per-phase rounds, keep-best schedule, combined
phase stop, and the legacy (adaptive_tuning=False) escape hatch."""

import jax
import jax.numpy as jnp
import numpy as np

from hamon.models import hinton_init
from hamon.nrpt import _nrpt_rounds_trace_count, discover_chain_count, nrpt_adaptive

from .utils import make_ising_grid


def _setup(L, coupling, n_chains, *, key=0):
    betas = jnp.linspace(0.1, 1.0, n_chains)
    nodes, edges, fb, ebms, progs = make_ising_grid(
        L, [float(b) for b in betas], coupling=coupling
    )
    inits = [
        hinton_init(k, ebms[-1], fb, ())
        for k in jax.random.split(jax.random.key(key), n_chains)
    ]
    return ebms[-1], progs[-1], fb, betas, inits


def _run(ebm, prog, betas, inits, **kw):
    _, stats = nrpt_adaptive(
        jax.random.key(3),
        ebm=ebm,
        program=prog,
        init_states=inits,
        clamp_state=[],
        n_rounds=60,
        gibbs_steps_per_round=1,
        initial_betas=betas,
        device="cpu",
        **kw,
    )
    return stats


def test_legacy_mode_unchanged():
    """adaptive_tuning=False runs exactly n_tune phases of rounds_per_tune rounds."""
    ebm, prog, fb, betas, inits = _setup(4, 0.5, 6)
    stats = _run(
        ebm, prog, betas, inits, adaptive_tuning=False, n_tune=3, rounds_per_tune=20
    )
    h = stats["tuning_history"]
    assert len(h) == 3
    assert all(e["rounds_used"] == 20 for e in h)


def test_adaptive_respects_round_caps():
    """Each adaptive phase uses a whole number of batches, never above the ceiling."""
    ebm, prog, fb, betas, inits = _setup(4, 0.5, 6)
    stats = _run(
        ebm,
        prog,
        betas,
        inits,
        n_tune=4,
        rounds_per_tune=200,
        round_batch=50,
        min_rounds_per_tune=50,
    )
    for e in stats["tuning_history"]:
        assert 50 <= e["rounds_used"] <= 200
        assert e["rounds_used"] % 50 == 0


def test_min_tune_phases_floor():
    """A trivially-easy (equalized-immediately) problem still runs >= min_tune_phases."""
    ebm, prog, fb, betas, inits = _setup(4, 0.0, 6)  # zero coupling -> rates ~0
    stats = _run(
        ebm, prog, betas, inits, n_tune=8, min_tune_phases=3, rounds_per_tune=100
    )
    assert len(stats["tuning_history"]) >= 3


def test_adaptive_stops_early_and_saves_rounds():
    """On an easy problem, adaptive uses fewer total tuning rounds than the
    legacy fixed budget (n_tune * rounds_per_tune)."""
    ebm, prog, fb, betas, inits = _setup(4, 0.05, 6)
    n_tune, rounds_per_tune = 6, 200
    stats = _run(
        ebm, prog, betas, inits, n_tune=n_tune, rounds_per_tune=rounds_per_tune
    )
    total = sum(e["rounds_used"] for e in stats["tuning_history"])
    assert total < n_tune * rounds_per_tune


def test_keeps_best_schedule():
    """Production uses the best-equalized schedule seen, not necessarily the last."""
    ebm, prog, fb, betas, inits = _setup(5, 0.7, 6)
    stats = _run(ebm, prog, betas, inits, n_tune=5, rounds_per_tune=120)
    h = stats["tuning_history"]
    best = min(h, key=lambda e: e["rej_std"])
    assert np.allclose(np.asarray(stats["betas"]), np.asarray(best["betas"]), atol=1e-5)


def test_compile_count_bounded():
    """With round_batch == n_rounds, the tuning batches and the production run
    share one compiled round loop (<= 2 traces total)."""
    ebm, prog, fb, betas, inits = _setup(4, 0.5, 6)
    before = _nrpt_rounds_trace_count[0]
    nrpt_adaptive(
        jax.random.key(3),
        ebm=ebm,
        program=prog,
        init_states=inits,
        clamp_state=[],
        n_rounds=40,
        gibbs_steps_per_round=1,
        initial_betas=betas,
        device="cpu",
        n_tune=3,
        rounds_per_tune=40,
        round_batch=40,
        min_rounds_per_tune=40,
    )
    assert _nrpt_rounds_trace_count[0] - before <= 2


def test_discover_forwards_tune_tol():
    """discover_chain_count accepts tune_tol and returns a sane chain count."""
    betas = jnp.linspace(0.0, 1.0, 8)
    nodes, edges, fb, ebms, progs = make_ising_grid(
        4, [float(b) for b in betas], coupling=0.4
    )

    def init_factory(n_chains, chain_ebms, chain_programs):
        f = chain_programs[0].gibbs_spec.free_blocks
        return [
            hinton_init(k, chain_ebms[0], f, ())
            for k in jax.random.split(jax.random.key(7), n_chains)
        ]

    out = discover_chain_count(
        jax.random.key(1),
        ebm=ebms[-1],
        program=progs[-1],
        init_factory=init_factory,
        beta_range=(0.0, 1.0),
        gibbs_steps_per_round=1,
        initial_n=8,
        rounds_per_probe=80,
        n_tune_per_probe=3,
        max_iters=3,
        min_chains=3,
        max_chains=24,
        tune_tol=0.1,
        device="cpu",
    )
    assert 3 <= out["n_chains"] <= 24
    assert out["converged_reason"] in {
        "chain_count",
        "lambda_stable",
        "no_progress",
        "max_iters",
    }
