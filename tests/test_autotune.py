"""Tests for the autotune orchestrator (autotune / autosample / NRPTPlan).

The full search picks device-dependent N and n_expl, so the end-to-end tests are
structural (valid result, correct sample shapes, repeatability, column order)
rather than asserting specific tuned values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hamon import Block, SpinNode, autosample, autotune
from hamon.models import IsingEBM, IsingSamplingProgram, hinton_init


def _model(n=8, coupling=0.6, seed=1):
    nodes = [SpinNode() for _ in range(n)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
    biases = jax.random.uniform(jax.random.key(seed), (n,), minval=-0.5, maxval=0.5)
    weights = jnp.ones(n - 1) * coupling
    ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(ebm, free_blocks, [])

    def init_factory(n_chains, ebms, programs):
        fb = programs[0].gibbs_spec.free_blocks
        keys = jax.random.split(jax.random.key(seed + 1), n_chains)
        return [hinton_init(keys[c], ebms[0], fb, ()) for c in range(n_chains)]

    return nodes, ebm, program, init_factory


_KW = {
    "clamp_state": [],
    "max_chains": 20,
    "max_exploration_steps": 4,
    "rounds_per_probe": 150,
    "n_tune": 2,
    "n_polish": 2,
    "n_rounds": 300,
    "device": "cpu",
}


def _ferro_grid(L=8, coupling=1.0):
    """Small 2D ferromagnet — strongly bimodal (all-up / all-down) at β=1."""
    n2 = [[SpinNode() for _ in range(L)] for _ in range(L)]
    nodes = [n for row in n2 for n in row]
    edges = []
    for i in range(L):
        for j in range(L):
            if j + 1 < L:
                edges.append((n2[i][j], n2[i][j + 1]))
            if i + 1 < L:
                edges.append((n2[i][j], n2[i + 1][j]))
    biases = jnp.zeros(len(nodes))
    weights = jnp.ones(len(edges)) * coupling
    even = [n2[i][j] for i in range(L) for j in range(L) if (i + j) % 2 == 0]
    odd = [n2[i][j] for i in range(L) for j in range(L) if (i + j) % 2 == 1]
    ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
    program = IsingSamplingProgram(ebm, [Block(even), Block(odd)], [])

    def init_factory(n_chains, ebms, programs):
        fb = programs[0].gibbs_spec.free_blocks
        keys = jax.random.split(jax.random.key(42), n_chains)
        return [hinton_init(keys[c], ebms[0], fb, ()) for c in range(n_chains)]

    return nodes, ebm, program, init_factory


_BIMODAL_KW = {
    "clamp_state": [],
    "beta_range": (0.2, 1.0),
    "gibbs_steps_per_round": 2,
    "max_chains": 24,
    "rounds_per_probe": 150,
    "n_tune": 2,
    "n_polish": 2,
    "n_rounds": 400,
    "device": "cpu",
}


def _mode_fractions(samples):
    """Fraction of samples in the +m and -m magnetization modes."""
    m = (2.0 * np.asarray(samples).astype(float) - 1.0).mean(axis=1)
    return float((m > 0.2).mean()), float((m < -0.2).mean())


class TestAutotune:
    def test_plan_structure_and_repeatable_sampling(self):
        nodes, ebm, program, init_factory = _model()
        plan = autotune(
            jax.random.key(0),
            ebm=ebm,
            program=program,
            init_factory=init_factory,
            sample_nodes=nodes,
            **_KW,
        )
        assert plan.n_chains >= 2
        assert plan.gibbs_steps_per_round >= 1
        assert plan.betas.shape == (plan.n_chains,)
        # endpoints span the requested beta range
        assert float(plan.betas[0]) == 0.0
        assert float(plan.betas[-1]) == 1.0
        # round-trip diagnostics are surfaced and measured over the production run
        assert plan.report.total_round_trips is not None
        assert plan.report.production_rounds == 300
        assert "round trips" in plan.report.summary()

        s1 = plan.sample(jax.random.key(1), 300)
        s2 = plan.sample(jax.random.key(2), 300)
        assert s1.shape == (300, len(nodes))
        assert s1.dtype == jnp.bool_
        # repeatable (no re-tuning) yet independent across keys
        assert not np.array_equal(np.asarray(s1), np.asarray(s2))

    def test_autosample_oneshot(self):
        nodes, ebm, program, init_factory = _model()
        samples, report = autosample(
            jax.random.key(3),
            n_samples=400,
            n_warmup=50,
            ebm=ebm,
            program=program,
            init_factory=init_factory,
            sample_nodes=nodes,
            **_KW,
        )
        assert samples.shape == (400, len(nodes))
        assert report.n_chains >= 2
        assert report.gibbs_steps_per_round >= 1
        assert "AUTOTUNE" in report.summary()

    def test_default_uses_device_calibrated_n_expl(self):
        # Default (search_exploration=False) sets n_expl deterministically by
        # device and runs no exploration search. On CPU the calibrated value is 1.
        _nodes, ebm, program, init_factory = _model()
        plan = autotune(
            jax.random.key(4),
            ebm=ebm,
            program=program,
            init_factory=init_factory,
            **_KW,
        )
        assert plan.gibbs_steps_per_round == 1  # CPU device default
        assert plan.report.exploration is None

    def test_search_exploration_runs_the_search(self):
        # Opt-in: search_exploration=True runs tune_exploration and reports it.
        _nodes, ebm, program, init_factory = _model()
        plan = autotune(
            jax.random.key(4),
            ebm=ebm,
            program=program,
            init_factory=init_factory,
            search_exploration=True,
            **_KW,
        )
        assert plan.gibbs_steps_per_round >= 1
        assert plan.report.exploration is not None
        assert "history" in plan.report.exploration

    def test_explicit_gibbs_steps_override(self):
        # An explicit gibbs_steps_per_round pins n_expl and skips the search,
        # even when search_exploration=True.
        _nodes, ebm, program, init_factory = _model()
        plan = autotune(
            jax.random.key(4),
            ebm=ebm,
            program=program,
            init_factory=init_factory,
            gibbs_steps_per_round=3,
            search_exploration=True,
            **_KW,
        )
        assert plan.gibbs_steps_per_round == 3
        assert plan.report.exploration is None

    def test_compile_cache_opt_out(self):
        # compile_cache=False must not raise and must not touch jax config.
        _nodes, ebm, program, init_factory = _model()
        plan = autotune(
            jax.random.key(5),
            ebm=ebm,
            program=program,
            init_factory=init_factory,
            compile_cache=False,
            **_KW,
        )
        assert plan.n_chains >= 2


class TestMultimodalSampling:
    """The tempered draw must recover a multimodal target that a single decoupled
    cold chain cannot (the whole point of NRPT)."""

    def test_tempered_draw_recovers_both_modes(self):
        # The tuned ladder round-trips on a bimodal ferromagnet, so the tempered
        # draw (default) must visit BOTH magnetization modes, while the decoupled
        # single-chain draw (tempered=False) collapses into one.
        nodes, ebm, program, init_factory = _ferro_grid(8)
        plan = autotune(
            jax.random.key(0),
            ebm=ebm,
            program=program,
            init_factory=init_factory,
            sample_nodes=nodes,
            **_BIMODAL_KW,
        )
        assert plan.report.total_round_trips and plan.report.total_round_trips > 0

        tempered = plan.sample(jax.random.key(1), 400)  # default == tempered
        # The decoupled draw warns (this model round-tripped), evidence of the
        # collapse risk it is about to demonstrate.
        with pytest.warns(UserWarning, match="mode-collapse"):
            cold = plan.sample(jax.random.key(1), 400, tempered=False)
        assert tempered.shape == (400, len(nodes)) and tempered.dtype == jnp.bool_
        assert cold.shape == (400, len(nodes)) and cold.dtype == jnp.bool_

        # default draw is the tempered path (same key ⇒ bit-identical)
        again = plan.sample(jax.random.key(1), 400, tempered=True)
        assert np.array_equal(np.asarray(tempered), np.asarray(again))

        t_pos, t_neg = _mode_fractions(tempered)
        c_pos, c_neg = _mode_fractions(cold)
        assert min(t_pos, t_neg) > 0.1, (
            "tempered should span both modes",
            t_pos,
            t_neg,
        )
        assert max(c_pos, c_neg) > 0.9, ("single chain should collapse", c_pos, c_neg)

    def test_tempered_thinning_and_warmup_shapes(self):
        # n_warmup discard + steps_per_sample thinning still yield exactly
        # n_samples rows (draw length is independent of the production window).
        nodes, ebm, program, init_factory = _ferro_grid(8)
        plan = autotune(
            jax.random.key(0),
            ebm=ebm,
            program=program,
            init_factory=init_factory,
            sample_nodes=nodes,
            **_BIMODAL_KW,
        )
        s = plan.sample(jax.random.key(3), 120, n_warmup=40, steps_per_sample=3)
        assert s.shape == (120, len(nodes)) and s.dtype == jnp.bool_
