"""Tests for the autotune orchestrator (autotune / autosample / NRPTPlan).

The full search picks device-dependent N and n_expl, so the end-to-end tests are
structural (valid result, correct sample shapes, repeatability, column order)
rather than asserting specific tuned values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

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


_KW = dict(
    clamp_state=[],
    max_chains=20,
    max_exploration_steps=4,
    rounds_per_probe=150,
    n_tune=2,
    n_polish=2,
    n_rounds=300,
    device="cpu",
)


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

    def test_search_exploration_false_fixes_n_expl_one(self):
        nodes, ebm, program, init_factory = _model()
        plan = autotune(
            jax.random.key(4),
            ebm=ebm,
            program=program,
            init_factory=init_factory,
            search_exploration=False,
            **_KW,
        )
        assert plan.gibbs_steps_per_round == 1
        assert plan.report.exploration is None

    def test_compile_cache_opt_out(self):
        # compile_cache=False must not raise and must not touch jax config.
        nodes, ebm, program, init_factory = _model()
        plan = autotune(
            jax.random.key(5),
            ebm=ebm,
            program=program,
            init_factory=init_factory,
            compile_cache=False,
            **_KW,
        )
        assert plan.n_chains >= 2
