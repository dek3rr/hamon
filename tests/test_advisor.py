"""Tests for hamon.advisor: β estimation from excitation costs and the
post-draw search advisor (verdict decision tree, pooling, warn policy)."""

from __future__ import annotations

import itertools
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from hamon.advisor import (
    SearchVerdict,
    communication_barrier,
    diagnose_search,
    estimate_beta_max,
    excess_energy,
    gs_occupancy,
    _pool_windows,
)
from hamon.models.ising import ising_excitation_costs


# ---------------------------------------------------------------------------
# estimator: closed forms vs brute-force enumeration
# ---------------------------------------------------------------------------


def _enumerate_chain(weights, beta):
    """Exact P(E = E_GS) and <E> - E_GS of an open chain by enumeration."""
    n = len(weights) + 1
    energies = []
    for bits in itertools.product((-1.0, 1.0), repeat=n):
        s = np.array(bits)
        energies.append(-(weights * s[:-1] * s[1:]).sum())
    e = np.array(energies)
    w = np.exp(-beta * (e - e.min()))
    p_gs = w[np.isclose(e, e.min())].sum() / w.sum()
    mean_excess = ((e - e.min()) * w).sum() / w.sum()
    return p_gs, mean_excess


@pytest.mark.parametrize("beta", [0.5, 2.0, 8.0])
def test_tree_closed_forms_match_enumeration(beta):
    rng = np.random.default_rng(0)
    weights = rng.uniform(0.1, 1.0, 5)  # 6-spin chain
    costs = 2.0 * np.abs(weights)
    p_gs, mean_excess = _enumerate_chain(weights, beta)
    assert np.isclose(gs_occupancy(costs, beta), p_gs, rtol=1e-10)
    assert np.isclose(excess_energy(costs, beta), mean_excess, rtol=1e-10)


def test_estimate_beta_max_meets_tolerance_and_is_monotone():
    rng = np.random.default_rng(1)
    costs = 2.0 * rng.uniform(0.0, 1.0, 127)
    scale = costs.sum() / 2.0
    est3 = estimate_beta_max(costs, scale, gap_tol=1e-3)
    est4 = estimate_beta_max(costs, scale, gap_tol=1e-4)
    assert est3.predicted_excess <= 1e-3 * scale * (1 + 1e-9)
    assert est4.beta_max > est3.beta_max  # tighter tolerance -> colder
    assert est3.predicted_Lambda > 0
    assert est3.predicted_n_chains >= 2
    assert "beta_max" in est3.summary()


def test_zero_cost_modes_are_ignored():
    # soft modes must not push beta to infinity (the 2D lesson)
    costs = np.array([0.0, 1e-15, 0.8, 1.2])
    est = estimate_beta_max(costs, 1.0, gap_tol=1e-3)
    assert est.n_costs == 2
    assert est.beta_max < 50.0


def test_communication_barrier_saturates():
    rng = np.random.default_rng(2)
    costs = 2.0 * rng.uniform(0.0, 1.0, 127)
    lam30 = communication_barrier(costs, 30.0)
    lam300 = communication_barrier(costs, 300.0)
    assert lam300 < 1.3 * lam30  # overshooting beta is cheap in chains


# ---------------------------------------------------------------------------
# cost extraction (hamon.models.ising)
# ---------------------------------------------------------------------------


def test_costs_tree_exact_path():
    rng = np.random.default_rng(3)
    n = 10
    edges = np.array([(i, i + 1) for i in range(n - 1)])
    w = rng.normal(size=n - 1)
    costs, scale, method = ising_excitation_costs(np.zeros(n), edges, w)
    assert method == "tree-exact"
    np.testing.assert_allclose(costs, 2.0 * np.abs(w))
    assert np.isclose(scale, np.abs(w).sum())


def test_costs_probe_on_triangle_is_exact():
    # Ferromagnetic triangle: any single-spin-stable state is a uniform global
    # minimum (a lone dissenter always has positive flip gain), so the probe
    # must return exactly cost 4 per site and |E_GS| = 3.
    edges = np.array([(0, 1), (0, 2), (1, 2)])
    costs, scale, method = ising_excitation_costs(np.zeros(3), edges, np.ones(3))
    assert method == "descent-probe"  # loopy graph
    np.testing.assert_allclose(costs, [4.0, 4.0, 4.0])
    assert np.isclose(scale, 3.0)


def test_costs_probe_ferromagnet_sees_domain_walls():
    # An open ferro grid has marginally stable domain-wall minima: the per-site
    # min across replicas may legitimately report wall-scale (or zero) costs,
    # never more than the uniform-state cost 2*degree.
    L = 3
    edges, degree = [], np.zeros(L * L)
    for r in range(L):
        for c in range(L):
            a = r * L + c
            if c + 1 < L:
                edges.append((a, a + 1))
            if r + 1 < L:
                edges.append((a, a + L))
    edges = np.array(edges)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    costs, scale, method = ising_excitation_costs(
        np.zeros(L * L), edges, np.ones(len(edges))
    )
    assert method == "descent-probe"
    assert (costs <= 2.0 * degree + 1e-12).all()
    assert (costs >= 0.0).all()
    assert scale <= len(edges) + 1e-12  # best-found E can't beat the true GS


# ---------------------------------------------------------------------------
# diagnose_search: decision tree on synthetic evidence
# ---------------------------------------------------------------------------


def _healthy_stats(n_pairs=4, n_rounds=2000, trips=200):
    rej = np.full(n_pairs, 0.3)
    return {
        "rejection_rates": rej,
        "n_rounds": n_rounds,
        "total_round_trips": trips,
    }


def _noise(rng, T):
    return rng.normal(0.0, 1.0, T)


def test_draw_limited_when_records_recent():
    rng = np.random.default_rng(4)
    e = _noise(rng, 1000) + 5.0
    e[950] = e.min() - 1.0  # fresh record near the end
    adv = diagnose_search(e, stats=_healthy_stats(), log=False)
    assert adv.verdict is SearchVerdict.DRAW_LIMITED
    assert adv.confidence == "high"
    assert adv.recommended_n_more == 2000
    assert adv.should_warn


def test_beta_limited_when_plateaued():
    rng = np.random.default_rng(5)
    e = np.abs(_noise(rng, 2000)) + 3.0
    e[3] = 1.0  # single early record, long silent tail
    adv = diagnose_search(e, stats=_healthy_stats(), cold_beta=2.0, log=False)
    assert adv.verdict is SearchVerdict.BETA_LIMITED
    assert adv.confidence == "high"
    assert adv.expected_tail_records > 3.0
    assert not adv.should_warn  # beta verdict is quiet by default
    adv2 = diagnose_search(
        e, stats=_healthy_stats(), cold_beta=2.0, warn_beta_limited=True, log=False
    )
    assert adv2.should_warn


def test_mixing_limited_barrier_saturated_wins_over_everything():
    rng = np.random.default_rng(6)
    e = _noise(rng, 1000)
    stats = _healthy_stats()
    stats["rejection_rates"] = np.array([0.2, 0.95, 0.3])  # saturated pair
    adv = diagnose_search(e, stats=stats, log=False)
    assert adv.verdict is SearchVerdict.MIXING_LIMITED
    assert adv.barrier_identified is False
    assert adv.recommended_n_chains is not None
    assert adv.should_warn


def _plateaued(rng, T, floor=1.0, base=3.0):
    """Trace with a single early record and a long silent tail (unique min)."""
    e = np.abs(rng.normal(0.0, 1.0, T)) + base
    e[3] = floor
    return e


def test_mixing_limited_dead_conveyor_attributes_knob():
    rng = np.random.default_rng(7)
    e = _plateaued(rng, 2000)  # plateau + tiny floor mass -> stuck-in-basin
    # equalized ladder, ample expected trips, but none observed -> ELE knob
    stats = {
        "rejection_rates": np.full(3, 0.3),
        "n_rounds": 5000,
        "total_round_trips": 1,
    }
    report = SimpleNamespace(gibbs_steps_per_round=3)
    adv = diagnose_search(e, stats=stats, report=report, log=False)
    assert adv.verdict is SearchVerdict.MIXING_LIMITED
    assert adv.conveyor_alive is False
    assert adv.recommended_gibbs_steps == 6
    # unequalized variant -> schedule knob (chains), not exploration
    stats2 = dict(stats, rejection_rates=np.array([0.05, 0.1, 0.7]))
    adv2 = diagnose_search(e, stats=stats2, report=report, log=False)
    assert adv2.verdict is SearchVerdict.MIXING_LIMITED
    assert adv2.recommended_n_chains is not None
    assert adv2.recommended_gibbs_steps is None


def test_dead_conveyor_with_heavy_floor_mass_is_freezeout_not_mixing():
    # cold-β freeze-out: slow conveyor but half the draws sit on the floor —
    # the search converged; MIXING would be a false alarm (GPU replay case)
    rng = np.random.default_rng(70)
    e = np.where(rng.random(2000) < 0.5, -10.0, -9.0)
    e[1] = -10.0
    stats = {
        "rejection_rates": np.full(3, 0.3),
        "n_rounds": 5000,
        "total_round_trips": 1,
    }
    adv = diagnose_search(e, stats=stats, log=False)
    assert adv.verdict is SearchVerdict.BETA_LIMITED
    assert adv.conveyor_alive is False
    assert any("freeze-out" in n for n in adv.notes)


def test_floor_mass_is_measured_over_post_record_tail():
    # A session that found its floor at draw 800 of 3000 and then sat on it
    # 30% of the time: whole-trace mass (22%) would read stuck-in-a-basin
    # (< 25% -> MIXING) but the post-record tail (30%) shows freeze-out
    # convergence — the sample_until GPU-replay artifact.
    tail = np.where(np.arange(2200) % 10 < 3, -10.0, -9.5)
    e = np.concatenate([[-8.0], np.full(799, -9.0), tail])
    stats = {
        "rejection_rates": np.full(3, 0.3),
        "n_rounds": 5000,
        "total_round_trips": 1,  # dead-slow conveyor
    }
    adv = diagnose_search(e, stats=stats, log=False)
    assert adv.last_record_draw == 800
    assert np.isclose(adv.fraction_at_min, 0.3, atol=0.01)  # tail, not 660/3000
    assert adv.verdict is SearchVerdict.BETA_LIMITED  # freeze-out, not MIXING
    assert any("freeze-out" in n for n in adv.notes)


def test_inconclusive_when_tail_has_too_few_deliveries():
    # plateaued trace, conveyor window too short to judge (conveyor None),
    # and only ~1 delivery in the silent tail -> silence proves nothing
    rng = np.random.default_rng(8)
    e = _plateaued(rng, 300)
    stats = {
        "rejection_rates": np.full(3, 0.3),
        "n_rounds": 100,  # tau_pred * 100 = 26 expected trips < 40 -> conveyor None
        "total_round_trips": 1,
    }
    adv = diagnose_search(e, stats=stats, log=False)
    assert adv.verdict is SearchVerdict.INCONCLUSIVE
    assert adv.conveyor_alive is None


def test_inconclusive_ess_fallback_without_roundtrip_data():
    rng = np.random.default_rng(80)
    adv = diagnose_search(_plateaued(rng, 12), log=False)
    assert adv.verdict is SearchVerdict.INCONCLUSIVE


def test_drifting_trace_is_draw_limited_despite_tiny_ess():
    # an EA-1600-like still-improving trace: records recent, ESS ~ 0 —
    # improvement is proof enough, the ESS gate must not block it
    rng = np.random.default_rng(9)
    e = -0.01 * np.arange(2000) + rng.normal(0.0, 0.5, 2000)
    adv = diagnose_search(e, stats=_healthy_stats(), log=False)
    assert adv.verdict is SearchVerdict.DRAW_LIMITED


def test_floor_context_overrides_draw_limited():
    # records still arriving BUT the predicted floor at this beta is 9% of
    # |E| — the 128-chain beta=3 replay case: go colder, not longer
    rng = np.random.default_rng(10)
    e = _noise(rng, 1000) + 5.0
    e[950] = e.min() - 1.0
    adv = diagnose_search(
        e,
        stats=_healthy_stats(),
        cold_beta=3.0,
        predicted_floor_rel=0.09,
        estimator_beta=37.0,
        log=False,
    )
    assert adv.verdict is SearchVerdict.BETA_LIMITED
    assert adv.confidence == "high"
    assert adv.recommended_beta == 37.0
    assert any("colder beats going longer" in n for n in adv.notes)


def test_frozen_trace_is_beta_limited_with_full_floor_fraction():
    e = np.full(500, -7.0)
    adv = diagnose_search(e, stats=_healthy_stats(), cold_beta=1.0, log=False)
    assert adv.verdict is SearchVerdict.BETA_LIMITED
    assert adv.fraction_at_min == 1.0
    assert adv.recommended_beta is None  # already sitting on the floor


def test_recommended_beta_two_level_formula():
    # 40% at level 0, 60% at level 1, cold_beta=5:
    # est = 5 + ln(0.6/0.4)/1.0 = 5.405 -> clamped up to 1.5*cold = 7.5
    e = np.array([0.0] * 400 + [1.0] * 600)
    rng = np.random.default_rng(9)
    rng.shuffle(e)
    e[0] = 1.0
    e[1] = 0.0  # ensure the min appears early enough to plateau
    adv = diagnose_search(e, stats=_healthy_stats(), cold_beta=5.0, log=False)
    assert adv.verdict is SearchVerdict.BETA_LIMITED
    assert np.isclose(adv.recommended_beta, 7.5)


def test_pool_windows_exact_totals():
    w1 = {
        "rejection_rates": np.array([0.2, 0.4]),
        "n_rounds": 100,
        "total_round_trips": 5,
    }
    w2 = {
        "rejection_rates": np.array([0.4, 0.6]),
        "n_rounds": 300,
        "total_round_trips": 3,
    }
    pooled = _pool_windows([w1, w2])
    np.testing.assert_allclose(pooled["rejection_rates"], [0.3, 0.5])
    assert pooled["n_rounds"] == 400
    assert pooled["total_round_trips"] == 8
    assert np.isclose(pooled["tau_observed"], 8 / 400)


def test_logging_policy(caplog):
    rng = np.random.default_rng(10)
    e = _noise(rng, 1000) + 5.0
    e[990] = e.min() - 1.0
    with caplog.at_level(logging.INFO, logger="hamon.advisor"):
        diagnose_search(e, stats=_healthy_stats())  # DRAW -> warning
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    caplog.clear()
    e2 = np.abs(_noise(rng, 2000)) + 3.0
    e2[3] = 1.0
    with caplog.at_level(logging.INFO, logger="hamon.advisor"):
        diagnose_search(e2, stats=_healthy_stats())  # BETA -> info only
    assert not any(r.levelno == logging.WARNING for r in caplog.records)
    assert any(r.levelno == logging.INFO for r in caplog.records)
