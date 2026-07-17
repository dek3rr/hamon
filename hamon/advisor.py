"""Ground-state-search guidance: β estimation and post-draw diagnosis.

Two host-side, model-agnostic tools for users who sample to *minimize* energy
(ground-state search) rather than to integrate at a fixed temperature:

- :func:`estimate_beta_max` picks the coldest ladder endpoint **before**
  tuning, from an excitation-cost spectrum of the energy landscape, so the
  full NRPT configuration is tuned exactly once — no raise-β-and-retry loops.
- :func:`diagnose_search` classifies, **after** a draw, which knob limits the
  result — mixing, draw budget, or β — from the cold chain's energy trace and
  the round-trip diagnostics, and says what to change.

Both operate on plain numpy scalars/arrays; nothing here touches jit.

Calibration (2026-07 study on the RL4Ising benchmark families, where exact
ground states are certifiable):

- The **equilibrium excess energy** of an independent-defect spectrum,
  ``⟨E⟩ − E_GS = Σ_i c_i·p_i(β)`` with ``p_i = 1/(1+e^{β c_i})``, predicts the
  observed min-over-draws floor within a small factor (within 25% at cold β).
  Selecting the smallest β whose predicted relative excess is below ``gap_tol``
  reproduced the empirically-derived good β on every family tested (1-D chains
  32-128, toroidal 2-D glasses 16-256, open EA 100-1600).
- A **ground-state-occupancy** target is the wrong selector on loopy graphs:
  near-zero-cost soft modes push it to β → ∞ while contributing nothing to the
  energy gap. The excess is self-regularizing (``c·p ≤ c/2``).
- ``Λ(β) = ∫₀^β σ_E(b)/√π db`` with ``σ_E²= Σ c²p(1−p)`` matched the measured
  communication barrier within 1-6% on chains, and its saturation in β means
  overshooting β_max costs almost no extra chains — err cold, never warm.
- Equilibrium occupancy does NOT predict exact-hit rates at large n (conveyor
  freeze-out: delivered states quench before equilibrating at the coldest
  rungs), so this module reports occupancy as rationale but never promises
  hit probabilities; closing the residual gap is the job of more draws
  (:meth:`hamon.NRPTPlan.extend`), which :func:`diagnose_search` detects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from collections.abc import Sequence

import numpy as np

from hamon.diagnostics import _ess_1d
from hamon.round_trips import (
    barrier_is_identified,
    conveyor_is_alive,
    recommend_n_chains,
)

if TYPE_CHECKING:
    from hamon.autotune import AutotuneReport

logger = logging.getLogger(__name__)

_EXP_CLAMP = 700.0  # exp overflow guard; occupation is exactly 0 beyond


# ---------------------------------------------------------------------------
# β estimation from an excitation-cost spectrum
# ---------------------------------------------------------------------------


def _defect_occupation(costs: np.ndarray, beta: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.minimum(beta * costs, _EXP_CLAMP)))


def excess_energy(costs: np.ndarray, beta: float) -> float:
    """Equilibrium mean excess ``⟨E⟩_β − E_GS`` of an independent-defect spectrum."""
    return float((costs * _defect_occupation(costs, beta)).sum())


def gs_occupancy(costs: np.ndarray, beta: float) -> float:
    """P(one equilibrium draw sits exactly at the ground state) — exact for
    independent defects (forests), an upper-bound proxy elsewhere."""
    x = np.minimum(beta * costs, _EXP_CLAMP)
    return float(np.exp(-np.log1p(np.exp(-x)).sum()))


def energy_std(costs: np.ndarray, beta: float) -> float:
    """``σ_β(E)`` of the independent-defect spectrum."""
    p = _defect_occupation(costs, beta)
    return float(np.sqrt((costs**2 * p * (1.0 - p)).sum()))


def communication_barrier(
    costs: np.ndarray, beta_max: float, n_grid: int = 400
) -> float:
    """Predicted NRPT barrier ``Λ(0→β_max) = ∫ σ_E(β)/√π dβ`` (trapezoid).

    Small-Δβ rejection density of Predescu/Syed under a Gaussian energy
    marginal; measured within 1-6% of hamon's own Λ̂ on chain instances.
    """
    betas = np.linspace(0.0, beta_max, n_grid)
    sig = np.array([energy_std(costs, b) for b in betas])
    return float(np.trapezoid(sig, betas) / np.sqrt(np.pi))


@dataclass
class BetaEstimate:
    """Result of :func:`estimate_beta_max`, with its rationale."""

    beta_max: float
    gap_tol: float
    energy_scale: float  # |E| scale the tolerance is relative to
    predicted_excess: float  # absolute equilibrium excess at beta_max
    predicted_Lambda: float  # communication barrier of the [0, beta_max] ladder
    predicted_n_chains: int  # ~2Λ, the round-trip-optimal count
    gs_occupancy: float  # occupancy proxy at beta_max (rationale only)
    method: str  # "tree-exact" | "descent-probe"
    n_costs: int

    def summary(self) -> str:
        return (
            f"beta_max={self.beta_max:.3g} ({self.method}, {self.n_costs} costs): "
            f"predicted thermal floor {self.predicted_excess:.3g} "
            f"(<= {self.gap_tol:g} x |E|~{self.energy_scale:.3g}), "
            f"Lambda~{self.predicted_Lambda:.1f} (~{self.predicted_n_chains} chains), "
            f"GS occupancy proxy {self.gs_occupancy:.2g}"
        )


def estimate_beta_max(
    costs: np.ndarray,
    energy_scale: float,
    *,
    gap_tol: float = 1e-3,
    method: str = "unspecified",
) -> BetaEstimate:
    """Smallest β whose predicted equilibrium excess is ≤ ``gap_tol·|energy_scale|``.

    **Arguments:**

    - `costs`: elementary excitation costs of the landscape — ``2|J_i|`` per
      nonzero bond on a field-free forest (exact), or ``2|local field|``
      spectra from greedy-descent minima elsewhere (see
      :func:`hamon.models.ising.ising_excitation_costs`). Non-positive
      entries are ignored (zero-cost modes are degeneracies, not defects).
    - `energy_scale`: magnitude of the ground-state energy (or best estimate);
      sets the meaning of the relative tolerance.
    - `gap_tol`: target relative thermal floor. The default 1e-3 reproduced
      the empirically-good β on every benchmark family; lower is safe but
      pays (slightly, Λ saturates) in chains.

    Monotone bisection; returns the full rationale as a :class:`BetaEstimate`.
    """
    costs = np.asarray(costs, dtype=np.float64)
    costs = costs[costs > 1e-12]
    if costs.size == 0:
        raise ValueError("no positive excitation costs — nothing to estimate from")
    target = gap_tol * abs(energy_scale)
    lo, hi = 0.0, 1.0
    while excess_energy(costs, hi) > target:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if excess_energy(costs, mid) <= target:
            hi = mid
        else:
            lo = mid
    lam = communication_barrier(costs, hi)
    return BetaEstimate(
        beta_max=hi,
        gap_tol=gap_tol,
        energy_scale=abs(energy_scale),
        predicted_excess=excess_energy(costs, hi),
        predicted_Lambda=lam,
        predicted_n_chains=recommend_n_chains(lam),
        gs_occupancy=gs_occupancy(costs, hi),
        method=method,
        n_costs=int(costs.size),
    )


# ---------------------------------------------------------------------------
# post-draw diagnosis
# ---------------------------------------------------------------------------


class SearchVerdict(StrEnum):
    MIXING_LIMITED = "mixing_limited"  # fix mixing before believing anything
    DRAW_LIMITED = "draw_limited"  # records still arriving: draw more
    BETA_LIMITED = "beta_limited"  # equilibrated at this beta: go colder
    INCONCLUSIVE = "inconclusive"  # too little effective data


@dataclass
class SearchAdvice:
    """Verdict of :func:`diagnose_search` plus the evidence behind it."""

    verdict: SearchVerdict
    confidence: str  # "low" | "medium" | "high"
    min_energy: float
    argmin_draw: int
    n_draws: int
    effective_draws: float
    last_record_draw: int
    expected_tail_records: float  # x = ln(n_draws / (last_record_draw + 1))
    fraction_at_min: float
    energy_gap: float | None  # min -> next distinct observed level
    cold_beta: float | None
    barrier_identified: bool | None = None
    conveyor_alive: bool | None = None
    efficiency: float | None = None
    total_round_trips: int | None = None
    recommended_beta: float | None = None
    recommended_n_more: int | None = None
    recommended_n_chains: int | None = None
    recommended_gibbs_steps: int | None = None
    notes: list[str] = field(default_factory=list)
    should_warn: bool = False  # the warn-policy outcome; emission is the caller's

    def summary(self) -> str:
        head = f"search advice: {self.verdict.value.upper()} ({self.confidence} confidence)"
        lines = [head]
        lines.append(
            f"  min E = {self.min_energy:.6g} at draw {self.argmin_draw}/{self.n_draws} "
            f"(last record at {self.last_record_draw}; "
            f"~{self.expected_tail_records:.2f} records expected since, "
            f"{self.fraction_at_min:.1%} of draws at min)"
        )
        if self.verdict is SearchVerdict.DRAW_LIMITED and self.recommended_n_more:
            lines.append(
                f"  -> new minima were still arriving; draw more "
                f"(~{self.recommended_n_more} further samples via plan.extend / "
                f"sample_until)"
            )
        if self.verdict is SearchVerdict.BETA_LIMITED:
            tgt = (
                f" (try beta~{self.recommended_beta:.3g})"
                if self.recommended_beta
                else ""
            )
            if self.fraction_at_min >= 0.5:
                lines.append(
                    f"  -> converged at beta={self.cold_beta}: the trace sits on "
                    f"its floor ({self.fraction_at_min:.0%} of draws at the "
                    f"minimum); if a lower state exists, only a colder beta can "
                    f"reveal it" + tgt
                )
            else:
                lines.append(
                    f"  -> the chain has equilibrated at beta={self.cold_beta}: "
                    f"the remaining excess is the thermal floor; raise beta and "
                    f"re-tune" + tgt
                )
        if self.verdict is SearchVerdict.MIXING_LIMITED:
            knob = []
            if self.recommended_n_chains:
                knob.append(f"max_chains -> ~{self.recommended_n_chains}")
            if self.recommended_gibbs_steps:
                knob.append(f"gibbs_steps_per_round -> {self.recommended_gibbs_steps}")
            lines.append(
                "  -> the tempering conveyor is not delivering independent "
                f"states ({'; '.join(knob) if knob else 'raise chains/exploration'}); "
                "results at ALL betas are untrustworthy until this is fixed"
            )
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def _pool_windows(stats_seq: Sequence[dict]) -> dict[str, Any] | None:
    """Pool per-window swap/trip tallies exactly across draw windows."""
    trips = rounds = 0.0
    rej_sum = None
    n_windows = 0
    for st in stats_seq:
        rtd = st if "rejection_rates" in st else st.get("round_trip_diagnostics", {})
        rej = rtd.get("rejection_rates")
        if rej is None:
            continue
        n_windows += 1
        rej = np.asarray(rej, dtype=np.float64)
        rej_sum = rej if rej_sum is None else rej_sum + rej
        rounds += float(rtd.get("n_rounds", 0) or 0)
        trips += float(rtd.get("total_round_trips", 0) or 0)
    if rej_sum is None or n_windows == 0:
        return None
    rej_mean = rej_sum / n_windows
    lam = float(rej_mean.sum())
    tau_pred = 1.0 / (2.0 + 2.0 * lam)
    tau_obs = trips / rounds if rounds else 0.0
    return {
        "rejection_rates": rej_mean,
        "Lambda": lam,
        "tau_predicted": tau_pred,
        "tau_observed": tau_obs,
        "n_rounds": rounds,
        "total_round_trips": trips,
    }


def diagnose_search(
    energy_trace: np.ndarray,
    *,
    stats: Sequence[dict] | dict | None = None,
    report: AutotuneReport | None = None,
    cold_beta: float | None = None,
    predicted_floor_rel: float | None = None,
    estimator_beta: float | None = None,
    min_effective_draws: int = 30,
    min_tail_deliveries: float = 3.0,
    floor_alarm: float = 1e-2,
    draw_evidence: float = 1.0,
    plateau_evidence: float = 3.0,
    level_rtol: float = 1e-6,
    warn_beta_limited: bool = False,
    log: bool = True,
) -> SearchAdvice:
    """Classify what limits a minimum-energy search and say which knob to turn.

    **Arguments:**

    - `energy_trace`: cold-chain energies of the draws the caller received
      (post-warmup, post-thinning), in draw order.
    - `stats`: per-draw-window NRPT stats dict(s) (each containing
      ``rejection_rates``, ``n_rounds``, ``total_round_trips`` — or a
      ``round_trip_diagnostics`` sub-dict with them); pooled exactly across
      windows. Optional; `report` is the fallback evidence source.
    - `report`: the :class:`hamon.AutotuneReport` of the plan (tuning-time
      mixing evidence).
    - `predicted_floor_rel` / `estimator_beta`: optional landscape context
      (from :func:`estimate_beta_max` at the *current* β): the predicted
      relative thermal floor and the estimator's recommended β. When the
      floor exceeds ``floor_alarm``, "records still arriving" is overridden
      to BETA_LIMITED — going colder beats going longer by orders of
      magnitude when the current β's equilibrium sits far above tolerance.
    - `warn_beta_limited`: escalate a confident BETA_LIMITED verdict to
      ``logger.warning``. Off by default — sampling at the requested β is
      working-as-designed unless the caller declared a search intent
      (``extend`` / ``sample_until`` set this).

    Decision order (v2, recalibrated on the RL4Ising GPU replays):

    1. A saturated ladder (``barrier_is_identified`` False) is MIXING_LIMITED
       outright — structural, nothing downstream is trustworthy.
    2. Recent records (``x = ln(T/(r_last+1)) < draw_evidence``) mean draws
       are still paying → DRAW_LIMITED — unless the landscape context says
       the thermal floor at this β is ≫ tolerance (BETA override above).
       Record recency is checked *before* any effective-sample gate: a
       drifting trace has ~zero ESS precisely because it is improving.
    3. A silent tail only establishes a plateau if enough conveyor *deliveries*
       occurred in it: at cold β records arrive per round trip, not per
       draw, so fewer than ``min_tail_deliveries`` expected deliveries since
       the last record → INCONCLUSIVE (ESS is the fallback gate when no
       round-trip data exists).
    4. An established plateau with a dead conveyor AND little mass at the
       minimum (< 25%) means stuck-in-a-basin → MIXING_LIMITED. A dead-slow
       conveyor with heavy floor mass is cold-β freeze-out — expected during
       ground-state search — and stays a note on the BETA/converged verdict.
    """
    e = np.asarray(energy_trace, dtype=np.float64).ravel()
    T = int(e.size)
    if T == 0:
        raise ValueError("empty energy trace")

    running = np.minimum.accumulate(e)
    argmin = int(np.argmin(e))
    records = np.flatnonzero(np.diff(running, prepend=np.inf) < 0)
    r_last = int(records[-1]) if records.size else 0
    x = float(np.log(T / (r_last + 1)))
    e_min = float(e[argmin])
    scale = max(abs(e_min), 1.0)
    at_min = np.isclose(e, e_min, rtol=level_rtol, atol=level_rtol * scale)
    frac_min = float(at_min.mean())
    above = e[~at_min]
    gap = float(above.min() - e_min) if above.size else None
    ess = float(_ess_1d(e))

    # --- mixing evidence (pooled draw windows, else the tuning report) ---
    if isinstance(stats, dict):
        stats = [stats]
    pooled = _pool_windows(stats) if stats else None
    if pooled is None and report is not None:
        rtd = dict(report.round_trip_diagnostics or {})
        rej = getattr(report, "rejection_rates", None)
        if rej is None:
            rej = rtd.get("rejection_rates")
        if rej is not None:
            pooled = {
                "rejection_rates": np.asarray(rej, dtype=np.float64),
                "Lambda": float(rtd.get("Lambda", np.asarray(rej).sum())),
                "tau_predicted": float(rtd.get("tau_predicted", 0.0)),
                "tau_observed": float(rtd.get("tau_observed", 0.0)),
                "n_rounds": float(report.production_rounds or 0),
                "total_round_trips": float(report.total_round_trips or 0),
            }

    barrier = conveyor = None
    efficiency = None
    trips = None
    rec_chains = rec_gibbs = None
    if pooled is not None:
        rej = pooled["rejection_rates"]
        lam = pooled["Lambda"]
        trips = int(pooled["total_round_trips"])
        barrier = barrier_is_identified(rej)
        conveyor = conveyor_is_alive(
            pooled["tau_observed"], pooled["tau_predicted"], int(pooled["n_rounds"])
        )
        if pooled["tau_predicted"] > 0:
            efficiency = pooled["tau_observed"] / pooled["tau_predicted"]
        if not barrier:
            rec_chains = max(
                recommend_n_chains(lam), int(np.ceil(1.5 * (len(rej) + 1)))
            )
        elif conveyor is False:
            if float(np.std(rej)) <= 0.15:
                # equalized ladder but slow conveyor: ELE violation
                rec_gibbs = None if report is None else 2 * report.gibbs_steps_per_round
            else:
                rec_chains = recommend_n_chains(lam)

    def _advice(verdict, confidence, **kw):
        adv = SearchAdvice(
            verdict=verdict,
            confidence=confidence,
            min_energy=e_min,
            argmin_draw=argmin,
            n_draws=T,
            effective_draws=ess,
            last_record_draw=r_last,
            expected_tail_records=x,
            fraction_at_min=frac_min,
            energy_gap=gap,
            cold_beta=cold_beta,
            barrier_identified=barrier,
            conveyor_alive=conveyor,
            efficiency=efficiency,
            total_round_trips=trips,
            **kw,
        )
        confident = confidence in ("medium", "high")
        adv.should_warn = confident and (
            verdict is SearchVerdict.MIXING_LIMITED
            or verdict is SearchVerdict.DRAW_LIMITED
            or (verdict is SearchVerdict.BETA_LIMITED and warn_beta_limited)
        )
        if log:
            (logger.warning if adv.should_warn else logger.info)(adv.summary())
        return adv

    # 1. saturated ladder: structural failure, absolute
    if barrier is False:
        return _advice(
            SearchVerdict.MIXING_LIMITED,
            "high",
            recommended_n_chains=rec_chains,
        )

    floor_high = predicted_floor_rel is not None and predicted_floor_rel > floor_alarm

    # 2. recent records: draws are demonstrably paying (checked before any
    # effective-sample gate — a drifting trace has ~zero ESS by construction)
    if x < draw_evidence:
        if floor_high:
            adv = _advice(
                SearchVerdict.BETA_LIMITED, "high", recommended_beta=estimator_beta
            )
            adv.notes.append(
                f"records are still arriving, but the predicted thermal floor at "
                f"beta={cold_beta} is {predicted_floor_rel:.2g} of |E| — far above "
                f"tolerance; going colder beats going longer"
            )
            return adv
        return _advice(
            SearchVerdict.DRAW_LIMITED,
            "high" if x < 0.5 else "medium",
            recommended_n_more=2 * T,
        )

    # 3. plateau side. A DEAD conveyor (ample expected trips, none observed)
    # with little floor mass means stuck-in-a-basin — MIXING, and the low
    # delivery count is itself the evidence, not a reason for doubt. A dead-
    # slow conveyor with heavy floor mass is cold-β freeze-out (expected
    # during ground-state search) and falls through to BETA with a note.
    if conveyor is False:
        if frac_min < 0.25:
            return _advice(
                SearchVerdict.MIXING_LIMITED,
                "high",
                recommended_n_chains=rec_chains,
                recommended_gibbs_steps=rec_gibbs,
            )
    elif trips is not None:
        # otherwise a silent tail only establishes a plateau if it plausibly
        # contained independent visits (deliveries arrive per round trip)
        tail_deliveries = trips * (T - r_last) / max(T, 1)
        if tail_deliveries < min_tail_deliveries:
            adv = _advice(SearchVerdict.INCONCLUSIVE, "low")
            adv.notes.append(
                f"only ~{tail_deliveries:.1f} conveyor deliveries since the last "
                f"record (< {min_tail_deliveries:g}); the silence does not yet "
                "establish a plateau — draw more before concluding"
            )
            return adv
    elif ess < min_effective_draws:
        adv = _advice(SearchVerdict.INCONCLUSIVE, "low")
        adv.notes.append(
            f"only ~{ess:.0f} effective draws (< {min_effective_draws}); "
            "record statistics are not meaningful yet"
        )
        return adv

    confidence = "high" if x >= plateau_evidence else ("medium" if x >= 2.0 else "low")
    rec_beta = estimator_beta if floor_high else None
    if (
        rec_beta is None
        and cold_beta is not None
        and gap is not None
        and gap > 0
        and frac_min < 0.5
    ):
        # two-level Boltzmann estimate: β at which the floor level reaches
        # even odds against the next observed level, clamped to a sane band.
        est = cold_beta + float(np.log((1.0 - frac_min) / max(frac_min, 1e-12))) / gap
        rec_beta = float(np.clip(est, 1.5 * cold_beta, 10.0 * cold_beta))
    adv = _advice(SearchVerdict.BETA_LIMITED, confidence, recommended_beta=rec_beta)
    if conveyor is False:
        knob = (
            f"gibbs_steps_per_round -> {rec_gibbs}"
            if rec_gibbs
            else f"max_chains -> ~{rec_chains}"
            if rec_chains
            else "more exploration"
        )
        adv.notes.append(
            f"the conveyor is slow (efficiency {efficiency:.2f}) — expected "
            f"during cold-β freeze-out; if you suspect missed modes, {knob}"
        )
    elif conveyor is None:
        adv.notes.append(
            "round-trip rate not yet measurable in this window; the BETA verdict "
            "rests on the record statistics alone"
        )
    return adv
