"""Tests for tune_exploration (adaptive n_expl on ESS per measured wall-second).

The peak-finding control logic (``_select_gibbs_steps``) is validated
deterministically with a synthetic probe. The end-to-end search is validated
structurally — the *value* it returns is device- and timing-dependent (on a
dispatch-bound GPU it flips to n_expl>1; on a compute-bound CPU it stays at 1),
so the test asserts a valid result rather than a specific count.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hamon import Block, SpinNode
from hamon.models import IsingEBM, IsingSamplingProgram, hinton_init
from hamon.tuning import (
    _select_gibbs_steps,
    _select_gibbs_steps_cost,
    tune_exploration,
)

# ---------------------------------------------------------------------------
# Unit: peak-finding control logic (deterministic, no NRPT)
# ---------------------------------------------------------------------------


def _fake_probe(objective_table, limiter_at=None):
    """A synthetic probe over a known objective(n_expl) table."""

    def probe(n):
        return {
            "n_expl": n,
            "objective": objective_table[n],
            "efficiency_limiter": "schedule" if n == limiter_at else None,
        }

    return probe


class TestSelectGibbsSteps:
    def test_interior_peak(self):
        """Doubling stops one step past the peak and returns the argmax."""
        probe = _fake_probe({1: 1.0, 2: 1.5, 4: 2.0, 8: 1.2, 16: 0.5})
        best, hist = _select_gibbs_steps(probe, 1, 16, 0.0)
        assert best["n_expl"] == 4
        assert [h["n_expl"] for h in hist] == [1, 2, 4, 8]

    def test_monotone_decreasing_returns_start(self):
        probe = _fake_probe({1: 2.0, 2: 1.0, 4: 0.5})
        best, hist = _select_gibbs_steps(probe, 1, 4, 0.0)
        assert best["n_expl"] == 1
        assert [h["n_expl"] for h in hist] == [1, 2]

    def test_schedule_limiter_stops_early(self):
        """A schedule-limited probe halts the search even if still improving."""
        probe = _fake_probe({1: 1.0, 2: 2.0, 4: 3.0}, limiter_at=2)
        best, hist = _select_gibbs_steps(probe, 1, 4, 0.0)
        assert [h["n_expl"] for h in hist] == [1, 2]
        assert best["n_expl"] == 2

    def test_improve_tol_rejects_marginal_gain(self):
        """A sub-tolerance gain is treated as no improvement (peak passed)."""
        probe = _fake_probe({1: 1.0, 2: 1.02})  # 2% gain < 5% tol
        best, hist = _select_gibbs_steps(probe, 1, 16, 0.05)
        assert best["n_expl"] == 1
        assert [h["n_expl"] for h in hist] == [1, 2]

    def test_improve_tol_prefers_smaller_on_marginal_gain(self):
        """Hysteresis: a later, only-marginally-better n_expl is not preferred
        over a smaller one, so the chosen objective need not be the raw argmax."""
        probe = _fake_probe({1: 1.0, 2: 2.0, 4: 2.06})  # n=4 only 3% over n=2
        best, hist = _select_gibbs_steps(probe, 1, 4, 0.05)
        assert best["n_expl"] == 2
        assert best["objective"] < max(h["objective"] for h in hist)
        assert [h["n_expl"] for h in hist] == [1, 2, 4]

    def test_max_steps_ceiling(self):
        probe = _fake_probe({1: 1.0, 2: 2.0, 4: 4.0})  # always improving
        best, hist = _select_gibbs_steps(probe, 1, 4, 0.0)
        assert [h["n_expl"] for h in hist] == [1, 2, 4]
        assert best["n_expl"] == 4


# ---------------------------------------------------------------------------
# Unit: the reuse-timing (fitted cost line) strategy
# ---------------------------------------------------------------------------


def _fake_cost_probe(ess_table, t_table, limiter_at=None):
    """Synthetic probe for the ESS-driven, cost-line-scored search."""

    def probe(n):
        return {
            "n_expl": n,
            "ess_median": ess_table[n],
            "t_round": t_table[n],
            "efficiency_limiter": "schedule" if n == limiter_at else None,
        }

    return probe


class TestSelectGibbsStepsCost:
    """``_select_gibbs_steps_cost`` doubles on ESS, then scores on a fitted line.

    Doubling is driven by ESS alone so the probe set is deterministic; wall time
    enters only through the least-squares cost line fitted across all probes.
    """

    def test_doubles_until_ess_saturates(self):
        ess = {1: 10.0, 2: 25.0, 4: 60.0, 8: 61.0, 16: 62.0}
        t = {1: 1.0, 2: 2.0, 4: 4.0, 8: 8.0, 16: 16.0}
        best, hist = _select_gibbs_steps_cost(
            _fake_cost_probe(ess, t), 1, 16, 0.05, rounds_per_probe=10
        )
        # n=8 gains only 1.7% over n=4, under the 5% tolerance -> stop there.
        assert [h["n_expl"] for h in hist] == [1, 2, 4, 8]
        assert best["n_expl"] == 4

    def test_schedule_limiter_stops_immediately(self):
        ess = {1: 10.0, 2: 25.0, 4: 60.0}
        t = {1: 1.0, 2: 2.0, 4: 4.0}
        _, hist = _select_gibbs_steps_cost(
            _fake_cost_probe(ess, t, limiter_at=2), 1, 16, 0.0, rounds_per_probe=10
        )
        assert [h["n_expl"] for h in hist] == [1, 2]

    def test_max_steps_ceiling(self):
        ess = {1: 10.0, 2: 40.0, 4: 160.0}  # always improving
        t = {1: 1.0, 2: 2.0, 4: 4.0}
        best, hist = _select_gibbs_steps_cost(
            _fake_cost_probe(ess, t), 1, 4, 0.0, rounds_per_probe=10
        )
        assert [h["n_expl"] for h in hist] == [1, 2, 4]
        assert best["n_expl"] == 4

    def test_history_is_rescored_against_the_fitted_line(self):
        """t_round is replaced by its fitted value and objective is filled in.

        Callers surface this same list, so the fitted numbers are what users
        see -- not the raw per-probe timings.
        """
        ess = {1: 10.0, 2: 40.0, 4: 160.0}
        t = {1: 5.0, 2: 6.0, 4: 8.0}  # exactly linear: t = 4 + 1*n
        _, hist = _select_gibbs_steps_cost(
            _fake_cost_probe(ess, t), 1, 4, 0.0, rounds_per_probe=10
        )
        for r in hist:
            assert r["t_round"] == pytest.approx(4.0 + r["n_expl"], rel=1e-9)
            assert r["objective"] == pytest.approx(
                r["ess_median"] / (10 * r["t_round"]), rel=1e-9
            )

    def test_zero_cost_line_yields_zero_objective(self):
        """A degenerate (all-zero) timing fit must not divide by zero."""
        ess = {1: 10.0, 2: 40.0, 4: 160.0}
        best, hist = _select_gibbs_steps_cost(
            _fake_cost_probe(ess, dict.fromkeys([1, 2, 4], 0.0)),
            1,
            4,
            0.0,
            rounds_per_probe=10,
        )
        assert all(r["objective"] == 0.0 for r in hist)
        assert best["n_expl"] == 1  # no record beats the first

    def test_marginal_objective_gain_prefers_smaller(self):
        """Hysteresis: climbing needs a > improve_tol gain, so a near-flat peak
        resolves to the cheaper count instead of flipping on timing noise."""
        # ESS doubles (keeps the search going) but cost doubles too, so the
        # objective is nearly flat across counts.
        ess = {1: 100.0, 2: 201.0, 4: 403.0}
        t = {1: 1.0, 2: 2.0, 4: 4.0}
        best, _ = _select_gibbs_steps_cost(
            _fake_cost_probe(ess, t), 1, 4, 0.05, rounds_per_probe=10
        )
        assert best["n_expl"] == 1


# ---------------------------------------------------------------------------
# End-to-end: structural smoke (CPU, small budgets)
# ---------------------------------------------------------------------------


def test_discover_gibbs_steps_smoke():
    n = 6
    nodes = [SpinNode() for _ in range(n)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
    biases = jax.random.uniform(jax.random.key(1), (n,), minval=-0.5, maxval=0.5)
    weights = jnp.ones(n - 1) * 0.6
    ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(ebm, free_blocks, [])

    n_chains = 5
    betas = jnp.linspace(0.0, 1.0, n_chains)
    keys = jax.random.split(jax.random.key(2), n_chains)
    init = [hinton_init(keys[c], ebm, free_blocks, ()) for c in range(n_chains)]

    res = tune_exploration(
        jax.random.key(3),
        init_states=init,
        clamp_state=[],
        initial_betas=betas,
        start_steps=1,
        max_steps=4,
        rounds_per_probe=150,
        n_tune_per_probe=2,
        time_rounds=50,
        time_reps=2,
        select_by="cost",  # timing-based selection (t_round measured)
        ebm=ebm,
        program=program,
        device="cpu",
    )

    assert res["gibbs_steps_per_round"] in (1, 2, 4)  # within the probed grid
    assert res["t_round"] > 0.0
    assert res["objective"] > 0.0
    assert len(res["history"]) >= 1
    first = res["history"][0]
    for k in ("n_expl", "objective", "ess_median", "t_round", "rt_per_compute"):
        assert k in first
    # The returned objective is the chosen n_expl's probe objective. (Selection
    # applies improve_tol hysteresis, so it need not be the raw argmax — a later,
    # only-marginally-better n_expl is not preferred over a smaller one.)
    chosen = res["gibbs_steps_per_round"]
    match = [h for h in res["history"] if h["n_expl"] == chosen]
    assert len(match) == 1
    assert res["objective"] == match[0]["objective"]


def test_discover_gibbs_steps_ele_smoke():
    # select_by="ele": timing-free, deterministic. The chosen n_expl is within
    # the probed grid and the objective is the round-trip efficiency.
    n = 6
    nodes = [SpinNode() for _ in range(n)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
    biases = jax.random.uniform(jax.random.key(1), (n,), minval=-0.5, maxval=0.5)
    weights = jnp.ones(n - 1) * 0.6
    ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(ebm, free_blocks, [])

    n_chains = 5
    betas = jnp.linspace(0.0, 1.0, n_chains)
    keys = jax.random.split(jax.random.key(2), n_chains)
    init = [hinton_init(keys[c], ebm, free_blocks, ()) for c in range(n_chains)]

    kw = {
        "init_states": init,
        "clamp_state": [],
        "initial_betas": betas,
        "start_steps": 1,
        "max_steps": 4,
        "rounds_per_probe": 150,
        "n_tune_per_probe": 2,
        "select_by": "ele",
        "ebm": ebm,
        "program": program,
        "device": "cpu",
    }
    res = tune_exploration(jax.random.key(3), **kw)

    assert res["gibbs_steps_per_round"] in (1, 2, 4)
    assert res["efficiency"] is not None
    assert len(res["history"]) >= 1
    # Deterministic: same inputs ⇒ same chosen n_expl (no wall-clock in the rule).
    res2 = tune_exploration(jax.random.key(3), **kw)
    assert res2["gibbs_steps_per_round"] == res["gibbs_steps_per_round"]
