"""Tests for NRPTPlan search sessions: the ColdChainObserver energy trace,
extend() warm restarts, sample_until(), and diagnostics wiring."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hamon import Block, SpinNode, autotune
from hamon.models import IsingEBM, IsingSamplingProgram, hinton_init


def _model(n=10, seed=1):
    """Integer-valued couplings/biases: energies are exact in float32, so the
    observer trace can be compared bit-level against recomputation."""
    rng = np.random.default_rng(seed)
    nodes = [SpinNode() for _ in range(n)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
    biases = jnp.asarray(rng.choice([-1.0, 0.0, 1.0], n))
    weights = jnp.asarray(rng.choice([-1.0, 1.0], n - 1))
    ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
    program = IsingSamplingProgram(ebm, [Block(nodes[::2]), Block(nodes[1::2])], [])

    def init_factory(n_chains, ebms, programs):
        fb = programs[0].gibbs_spec.free_blocks
        keys = jax.random.split(jax.random.key(seed + 1), n_chains)
        return [hinton_init(keys[c], ebms[0], fb, ()) for c in range(n_chains)]

    return nodes, ebm, program, init_factory, np.asarray(biases), np.asarray(weights)


_KW = dict(
    clamp_state=[],
    max_chains=12,
    rounds_per_probe=120,
    n_tune=2,
    n_polish=2,
    n_rounds=200,
    device="cpu",
)


def _make_plan(seed=1, **overrides):
    nodes, ebm, program, init_factory, b, w = _model(seed=seed)
    kw = dict(_KW, **overrides)
    plan = autotune(
        jax.random.key(3),
        ebm=ebm,
        program=program,
        init_factory=init_factory,
        sample_nodes=nodes,
        beta_range=(0.0, 1.5),
        **kw,
    )
    return plan, b, w


def _energies(samples, b, w):
    s = 2.0 * np.asarray(samples).astype(np.float64) - 1.0
    return -(s @ b + (w[None, :] * s[:, :-1] * s[:, 1:]).sum(1))


@pytest.fixture(scope="module")
def plan_bw():
    return _make_plan()


def test_energy_trace_matches_returned_samples_bitwise(plan_bw):
    plan, b, w = plan_bw
    samples = plan.sample(jax.random.key(5), 60, n_warmup=20, steps_per_sample=3)
    trace = np.asarray(plan._energy_chunks[0], dtype=np.float64)
    assert trace.shape == (60,)
    np.testing.assert_array_equal(trace, _energies(samples, b, w))


def test_extend_appends_aligned_energy_chunks(plan_bw):
    plan, b, w = plan_bw
    plan.sample(jax.random.key(6), 40)
    more = plan.extend(jax.random.key(7), 30)
    assert more.shape[0] == 30
    assert len(plan._energy_chunks) == 2
    np.testing.assert_array_equal(
        np.asarray(plan._energy_chunks[1], dtype=np.float64), _energies(more, b, w)
    )
    assert plan.last_advice is not None
    assert plan.report.search_advice is plan.last_advice


def test_extend_is_deterministic():
    p1, _, _ = _make_plan(seed=2)
    p2, _, _ = _make_plan(seed=2)
    s1 = p1.sample(jax.random.key(8), 40)
    s2 = p2.sample(jax.random.key(8), 40)
    np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
    e1 = p1.extend(jax.random.key(9), 40)
    e2 = p2.extend(jax.random.key(9), 40)
    np.testing.assert_array_equal(np.asarray(e1), np.asarray(e2))


def test_extend_continues_not_restarts(plan_bw):
    plan, b, w = plan_bw
    plan.sample(jax.random.key(10), 50)
    first_ladder = plan._last_ladder
    plan.extend(jax.random.key(11), 50)
    # the retained ladder must advance (a restart would reproduce it)
    same = all(
        np.array_equal(np.asarray(a), np.asarray(c))
        for a, c in zip(first_ladder, plan._last_ladder)
    )
    assert not same


def test_no_retrace_on_repeated_extend(plan_bw):
    from hamon.nrpt import _nrpt_rounds_trace_count

    plan, _, _ = plan_bw
    plan.sample(jax.random.key(12), 40)
    plan.extend(jax.random.key(13), 40)  # may compile its shape once
    before = _nrpt_rounds_trace_count[0]
    plan.extend(jax.random.key(14), 40)
    plan.extend(jax.random.key(15), 40)
    assert _nrpt_rounds_trace_count[0] == before


def test_extend_requires_prior_tempered_draw():
    plan, _, _ = _make_plan(seed=3)
    with pytest.raises(RuntimeError, match="tempered"):
        plan.extend(jax.random.key(1), 10)


def test_sample_until_respects_budget_and_returns_advice(plan_bw):
    plan, _, _ = plan_bw
    samples, advice = plan.sample_until(
        jax.random.key(16), chunk=32, max_total=96, patience_deliveries=1e9
    )
    assert samples.shape[0] == 96
    assert advice is not None
    assert advice is plan.last_advice


def test_sample_until_stops_on_target(plan_bw):
    plan, _, _ = plan_bw
    samples, _ = plan.sample_until(
        jax.random.key(17), chunk=32, max_total=320, target_energy=1e9
    )
    assert samples.shape[0] == 32  # first chunk already beats the target


def test_sample_until_delivery_patience_stops_on_plateau(plan_bw):
    # A healthy conveyor that plateaus should stop before the budget: with a
    # low delivery threshold the flat stretch trips the patience quickly.
    plan, _, _ = plan_bw
    samples, _ = plan.sample_until(
        jax.random.key(18), chunk=32, max_total=3200, patience_deliveries=0.5
    )
    # any single delivered trip on a flat chunk crosses 0.5 -> well under budget
    assert samples.shape[0] < 3200


def test_padded_and_unpadded_sessions_bit_identical(monkeypatch):
    # autotune no longer exposes pad_probes; pin the internal auto-rule to
    # exercise both paths on the CPU test device. (importlib: the package
    # attribute `hamon.autotune` is the function, shadowing the submodule.)
    import importlib

    autotune_mod = importlib.import_module("hamon.autotune")

    monkeypatch.setattr(autotune_mod, "_auto_pad_probes", lambda *a: True)
    p_pad, _, _ = _make_plan(seed=4)
    monkeypatch.setattr(autotune_mod, "_auto_pad_probes", lambda *a: False)
    p_raw, _, _ = _make_plan(seed=4)
    s_pad = p_pad.sample(jax.random.key(20), 40)
    s_raw = p_raw.sample(jax.random.key(20), 40)
    np.testing.assert_array_equal(np.asarray(s_pad), np.asarray(s_raw))
    e_pad = p_pad.extend(jax.random.key(21), 40)
    e_raw = p_raw.extend(jax.random.key(21), 40)
    np.testing.assert_array_equal(np.asarray(e_pad), np.asarray(e_raw))
    np.testing.assert_array_equal(
        np.asarray(p_pad._energy_chunks[1]), np.asarray(p_raw._energy_chunks[1])
    )


def _advice_stub(trace):
    """Minimal stand-in carrying just what _compute_advice reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        _energy_chunks=[jnp.asarray(trace)],
        _window_stats=[
            {
                "rejection_rates": np.full(3, 0.3),
                "n_rounds": 5000,
                "total_round_trips": 200,  # healthy conveyor
            }
        ],
        report=None,
        betas=np.array([0.0, 1.5]),
        search_context=None,
        last_advice=None,
    )


def test_extend_escalates_an_unchanged_beta_verdict_to_a_warning(caplog):
    """sample() files BETA at info; the first extend() must still warn.

    Regression: the emit-on-change key was (verdict, confidence) only, so a
    plan that was BETA_LIMITED from its very first draw never warned at all,
    even though extend()/sample_until() declare the search intent that makes
    the verdict worth escalating.
    """
    import logging

    from hamon.advisor import SearchVerdict
    from hamon.autotune import NRPTPlan

    rng = np.random.default_rng(21)
    trace = np.where(rng.random(2000) < 0.5, -10.0, -9.0)
    trace[1] = -10.0  # plateaued on a heavy floor -> BETA_LIMITED, high
    stub = _advice_stub(trace)

    with caplog.at_level(logging.DEBUG, logger="hamon.autotune"):
        first = NRPTPlan._compute_advice(stub, warn_beta_limited=False)
        assert first.verdict is SearchVerdict.BETA_LIMITED
        assert not first.should_warn
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

        caplog.clear()
        second = NRPTPlan._compute_advice(stub, warn_beta_limited=True)

    assert second.verdict is first.verdict
    assert second.confidence == first.confidence  # nothing about the trace moved
    assert second.should_warn
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warned, "the first escalation to warning level must be emitted"
    assert "BETA_LIMITED" in warned[0].getMessage()


def test_repeated_extend_at_the_same_verdict_stays_quiet(caplog):
    """...but only the *first* escalation: an unchanged verdict is noise."""
    import logging

    from hamon.autotune import NRPTPlan

    rng = np.random.default_rng(22)
    trace = np.where(rng.random(2000) < 0.5, -10.0, -9.0)
    trace[1] = -10.0
    stub = _advice_stub(trace)

    with caplog.at_level(logging.DEBUG, logger="hamon.autotune"):
        NRPTPlan._compute_advice(stub, warn_beta_limited=False)
        NRPTPlan._compute_advice(stub, warn_beta_limited=True)
        caplog.clear()
        for _ in range(3):
            NRPTPlan._compute_advice(stub, warn_beta_limited=True)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_ising_sample_diagnostics_backward_compatible():
    from hamon.models.ising import ising_sample

    rng = np.random.default_rng(0)
    n = 12
    edges = np.array([(i, i + 1) for i in range(n - 1)])
    w = rng.uniform(0.2, 1.0, n - 1)
    samples, diag = ising_sample(
        jnp.zeros(n),
        jnp.asarray(edges),
        jnp.asarray(w),
        key=jax.random.key(0),
        beta="auto",
        n_samples=100,
        n_warmup=50,
        device="cpu",
    )
    for key in (
        "n_chains",
        "betas",
        "Lambda",
        "gibbs_steps_per_round",
        "mean_spins",
        "device",
        "round_trip_diagnostics",
        "report",
        "beta_estimate",
        "search_advice",
    ):
        assert key in diag, key
    assert diag["beta_estimate"] is not None
    assert diag["beta_estimate"].method == "tree-exact"
    assert diag["search_advice"] is not None
    assert samples.shape == (100, n)
