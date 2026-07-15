"""Diagnostics for sample quality and model health."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from jaxtyping import Array, Bool, Shaped

logger = logging.getLogger(__name__)

# These diagnostics are one-shot host-side summaries over small arrays (a few
# hundred samples × a few hundred variables, plus length-n_chains stat vectors).
# Running them in numpy on the host — rather than jax.numpy on an accelerator —
# avoids the dominant cost, which is per-op XLA compilation: each first-seen
# shape triggers a separate kernel compile (~40 ms each), so the jax path spent
# ~1 s compiling ~25 ms of arithmetic. numpy has no compile step and needs a
# single device→host transfer of `samples` instead of one sync per reduction.
# Inputs may be jax arrays; ``np.asarray`` pulls them to the host once.
# (If these were ever run on very large inputs, the host transfer + host compute
# could flip this trade-off and a jitted jax implementation would win.)


# ---------------------------------------------------------------------------
# Sample convergence
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceReport:
    """Result of :func:`sample_convergence`.

    Attributes:
        status: ``"CONVERGED"``, ``"BORDERLINE"``, or ``"NEED_MORE"``.
        drifts: L1 drift in marginals between consecutive quartile checkpoints
            (three values: 25→50 %, 50→75 %, 75→100 %).
        rank_stability: Jaccard similarity of the top-*k* variables between
            the first and second half of the samples.
    """

    status: str
    drifts: list[float]
    rank_stability: float


def sample_convergence(
    samples: Bool[Array, "n_samples n_variables"],
    *,
    target_k: int = 15,
    drift_threshold: float = 0.01,
    jaccard_threshold: float = 0.8,
) -> ConvergenceReport:
    """Measure stability of marginal probability estimates.

    Splits *samples* into quartile checkpoints (25 %, 50 %, 75 %, 100 %),
    computes marginals at each checkpoint, and reports the L1 drift between
    consecutive checkpoints together with the rank stability of the top-*k*
    variables.

    Args:
        samples: boolean array of shape ``(n_samples, n_variables)``.
        target_k: number of top variables to track for rank stability.
        drift_threshold: maximum acceptable L1 drift per variable for the
            final checkpoint to be considered converged.
        jaccard_threshold: minimum Jaccard similarity of top-*k* sets
            between halves for rank stability to be considered converged.

    Returns:
        A :class:`ConvergenceReport`.
    """
    s = np.asarray(samples)
    n_samples, n_vars = s.shape
    target_k = min(target_k, n_vars)

    quartile_indices = [n_samples * q // 4 for q in range(1, 5)]
    marginals = [
        np.mean(s[:idx].astype(np.float32), axis=0) for idx in quartile_indices
    ]

    drifts = [
        float(np.mean(np.abs(marginals[i + 1] - marginals[i])))
        for i in range(len(marginals) - 1)
    ]

    # Rank stability: Jaccard of top-k between first and second half.
    half = n_samples // 2
    m_first = np.mean(s[:half].astype(np.float32), axis=0)
    m_second = np.mean(s[half:].astype(np.float32), axis=0)

    # stable sort matches jax.numpy.argsort's default tie-breaking.
    top_first = set(np.argsort(-m_first, kind="stable")[:target_k].tolist())
    top_second = set(np.argsort(-m_second, kind="stable")[:target_k].tolist())
    jaccard = len(top_first & top_second) / len(top_first | top_second)

    final_drift = drifts[-1]
    if final_drift <= drift_threshold and jaccard >= jaccard_threshold:
        status = "CONVERGED"
    elif final_drift <= drift_threshold * 3 and jaccard >= jaccard_threshold * 0.8:
        status = "BORDERLINE"
    else:
        status = "NEED_MORE"

    return ConvergenceReport(status=status, drifts=drifts, rank_stability=jaccard)


# ---------------------------------------------------------------------------
# Marginal entropy
# ---------------------------------------------------------------------------


def marginal_entropy(
    samples: Bool[Array, "n_samples n_variables"],
) -> float:
    """Normalized entropy of the empirical marginal distribution.

    Computes the mean per-variable binary entropy, normalized to [0, 1].
    A value near 0 means most variables are frozen (all True or all False);
    near 1 means each variable is near 50/50.

    Args:
        samples: boolean array of shape ``(n_samples, n_variables)``.

    Returns:
        Scalar in [0, 1].
    """
    p = np.mean(np.asarray(samples).astype(np.float32), axis=0)
    # In float32 the upper clip 1.0 - 1e-10 rounds to exactly 1.0, so a frozen
    # variable gives 1 - safe_p == 0 -> log2(0) == -inf -> 0 * -inf == NaN. The
    # np.where below masks those entries to 0; errstate just silences the
    # transient warnings (jax produced the same NaN-then-mask silently).
    safe_p = np.clip(p, 1e-10, 1.0 - 1e-10)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(safe_p * np.log2(safe_p) + (1 - safe_p) * np.log2(1 - safe_p))
    # Zero out entropy for variables that are truly frozen.
    h = np.where((p == 0.0) | (p == 1.0), 0.0, h)
    return float(np.mean(h))


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------


@dataclass
class ESSReport:
    """Result of :func:`effective_sample_size`.

    Attributes:
        per_variable: per-column effective sample size, shape ``(n_variables,)``.
        min_ess: smallest ESS across variables (the worst-mixing variable — the
            conservative number to quote).
        median_ess / mean_ess: summary ESS across variables.
        ess_fraction: ``min_ess / n_samples`` — the efficiency of the
            worst-mixing variable, in ``[0, 1]``.
        n_samples: number of samples the estimate was computed from.
    """

    per_variable: np.ndarray
    min_ess: float
    median_ess: float
    mean_ess: float
    ess_fraction: float
    n_samples: int


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation ρ(0..n-1) of a 1-D series via FFT.

    Uses zero-padding to ``2n`` so the inverse transform gives the *linear*
    (non-circular) autocovariance; ρ(0) is normalized to 1. O(n log n).
    """
    n = x.shape[0]
    x = x - x.mean()
    # Pad to >= 2n so the circular FFT correlation has no wrap-around.
    size = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, n=size)
    acov = np.fft.irfft(f * np.conjugate(f), n=size)[:n]
    return acov / acov[0]


def _ess_1d(x: np.ndarray) -> float:
    """ESS of one variable via Geyer's initial-positive-sequence estimator.

    τ_int = 1 + 2·Σ ρ(k), truncated where consecutive lag-pair sums Γ_m =
    ρ(2m+1) + ρ(2m+2) first turn non-positive (Geyer 1992) — this guards
    against the noisy tail of the empirical autocorrelation. ESS = n / τ_int,
    clipped to ``[0, n]``. A frozen (zero-variance) series carries no
    autocorrelation information and is reported as ESS = n.
    """
    n = x.shape[0]
    if n < 2 or x.var() == 0.0:
        return float(n)

    rho = _autocorrelation(x)
    # Pair the lag-≥1 autocorrelations: Γ_m = ρ(2m+1) + ρ(2m+2).
    tail = rho[1:]
    n_pairs = tail.shape[0] // 2
    if n_pairs == 0:
        return float(n)
    pair_sums = tail[: 2 * n_pairs].reshape(n_pairs, 2).sum(axis=1)

    # Initial positive sequence: keep pairs up to the first non-positive one.
    nonpos = np.nonzero(pair_sums <= 0)[0]
    cutoff = int(nonpos[0]) if nonpos.size else n_pairs

    tau = 1.0 + 2.0 * float(pair_sums[:cutoff].sum())
    tau = max(tau, 1.0)  # ESS can never exceed n
    return float(np.clip(n / tau, 0.0, n))


def effective_sample_size(
    samples: Shaped[Array, "n_samples n_variables"] | np.ndarray,
) -> ESSReport:
    """Estimate the effective sample size of an autocorrelated MCMC trace.

    MCMC draws are autocorrelated, so ``n`` correlated samples carry the
    information of fewer independent ones. ESS estimates that effective count:
    the Monte-Carlo error of any estimate computed from the trace scales as
    ``σ/√ESS``, not ``σ/√n``. For an iid trace ESS ≈ ``n``; for a slowly mixing
    one it can be far smaller. Pair ``ess_fraction`` (or ``min_ess``) with the
    run's wall-clock time to get ESS/second, the efficiency metric used to
    compare schedules or chain counts.

    Computed on the host with numpy (FFT autocorrelation + Geyer
    initial-positive-sequence; see :func:`_ess_1d`), so there is no XLA compile
    cost. Inputs may be jax arrays; they are pulled to the host once.

    For multimodal parallel tempering, a low single-marginal ESS can reflect
    the chain *correctly* jumping between modes (mode switches are long-range
    correlation), so read ESS alongside the round-trip diagnostics rather than
    instead of them.

    Args:
        samples: array of shape ``(n_samples, n_variables)`` (a 1-D
            ``(n_samples,)`` trace is treated as a single variable). Boolean or
            numeric.

    Returns:
        An :class:`ESSReport`.
    """
    s = np.asarray(samples)
    if s.ndim == 1:
        s = s[:, None]
    s = s.astype(np.float64)
    n_samples, n_vars = s.shape

    per_variable = np.array([_ess_1d(s[:, j]) for j in range(n_vars)])
    min_ess = float(np.min(per_variable))
    return ESSReport(
        per_variable=per_variable,
        min_ess=min_ess,
        median_ess=float(np.median(per_variable)),
        mean_ess=float(np.mean(per_variable)),
        ess_fraction=min_ess / n_samples if n_samples else 0.0,
        n_samples=int(n_samples),
    )


# ---------------------------------------------------------------------------
# Energy balance
# ---------------------------------------------------------------------------


@dataclass
class EnergyBalanceReport:
    """Result of :func:`energy_balance`.

    Attributes:
        bias_energy_spread: range (max − min) of per-node bias contributions
            ``β·|b_i|``.
        coupling_energy_per_spin: mean total absolute coupling energy per
            variable, ``β · mean_i(Σ_j |J_ij|)``.
        ratio: ``coupling_energy_per_spin / bias_energy_spread``.  Values
            well below 1 mean biases dominate; well above 1 mean couplings
            dominate.
    """

    bias_energy_spread: float
    coupling_energy_per_spin: float
    ratio: float


def energy_balance(
    biases: Shaped[Array, " n"],
    edges: Shaped[Array, "m 2"],
    weights: Shaped[Array, " m"],
    *,
    beta: float = 1.0,
    warn_low: float = 0.05,
    warn_high: float = 2.0,
) -> EnergyBalanceReport:
    r"""Check whether bias and coupling energy scales are balanced.

    Computes the energy contribution from biases vs couplings at the given
    temperature and reports their ratio.  Logs a warning when the ratio
    falls outside ``[warn_low, warn_high]``.

    Args:
        biases: per-node bias array of shape ``(n,)``.
        edges: integer index pairs of shape ``(m, 2)``.
        weights: per-edge coupling of shape ``(m,)``.
        beta: inverse temperature.
        warn_low: ratio below which a warning is logged.
        warn_high: ratio above which a warning is logged.

    Returns:
        An :class:`EnergyBalanceReport`.
    """
    biases_np = np.asarray(biases)
    edges_np = np.asarray(edges)
    weights_np = np.asarray(weights)
    n = biases_np.shape[0]

    bias_contributions = beta * np.abs(biases_np)
    bias_spread = float(np.max(bias_contributions) - np.min(bias_contributions))

    # Sum of absolute coupling weights incident on each node. np.add.at is the
    # unbuffered scatter-add matching jax's coupling_per_node.at[idx].add(...).
    abs_w = beta * np.abs(weights_np)
    coupling_per_node = np.zeros(n, dtype=abs_w.dtype)
    np.add.at(coupling_per_node, edges_np[:, 0], abs_w)
    np.add.at(coupling_per_node, edges_np[:, 1], abs_w)
    coupling_per_spin = float(np.mean(coupling_per_node))

    if bias_spread > 0:
        ratio = coupling_per_spin / bias_spread
    else:
        ratio = float("inf")

    if ratio < warn_low:
        logger.warning(
            "Energy balance ratio %.3f < %.3f: biases dominate, couplings may be irrelevant.",
            ratio,
            warn_low,
        )
    elif ratio > warn_high:
        logger.warning(
            "Energy balance ratio %.3f > %.1f: couplings dominate, biases may be irrelevant.",
            ratio,
            warn_high,
        )

    return EnergyBalanceReport(
        bias_energy_spread=bias_spread,
        coupling_energy_per_spin=coupling_per_spin,
        ratio=ratio,
    )


# ---------------------------------------------------------------------------
# NRPT health verdict
# ---------------------------------------------------------------------------


@dataclass
class NRPTHealthReport:
    """Result of :func:`report_nrpt_diagnostics`.

    Attributes:
        healthy: ``True`` when no issues were found and there was enough
            data to judge.
        insufficient_data: ``True`` when swap-attempt counts were too low to
            apply the pass/fail criteria; would-be issues are demoted to
            warnings and ``healthy`` reflects only what could be checked.
        issues: Hard failures — the samples should not be trusted.
        warnings: Soft findings worth investigating.
        acceptance_mean / rejection_std: Swap-rate statistics.
        Lambda / tau_observed / tau_predicted / efficiency / total_round_trips:
            Round-trip diagnostics (``None`` when the run was made with
            ``track_round_trips=False``).
        barrier_identified: ``False`` when the index process did not round-trip
            (a stalled conveyor), so ``Lambda`` is a within-basin artifact and
            must not be trusted — add chains / equalize the ladder. ``True`` when
            round trips flowed; ``None`` when round-trip diagnostics were
            unavailable. See :func:`hamon.round_trips.barrier_is_identified`.
        recommended_n_chains: Suggested chain count when efficiency is low.
        efficiency_limiter: When round-trip efficiency is low, which knob to
            turn — ``"schedule"`` (the ladder is not equalized: tune it further
            or add chains) or ``"local_exploration"`` (the ladder *is*
            equalized, so the local kernel is the bottleneck — an ELE violation;
            raise ``gibbs_steps_per_round``, or increase N as the alternative
            lever). ``None`` when efficiency is healthy or unavailable.
        barrier_peak_beta: Midpoint β of a sharp barrier peak, if detected.
        convergence_status / rank_stability / marginal_entropy: Sample-based
            metrics (``None`` when *samples* was not provided). For NRPT,
            convergence is informational only — correct multi-modal sampling
            shifts marginals between run halves, so a non-CONVERGED status is
            **not** treated as a failure.
        min_ess / median_ess / ess_fraction: Effective-sample-size summaries
            over the provided *samples* (``None`` when not provided).
            ``ess_fraction`` is ``min_ess / n_samples`` for the worst-mixing
            variable; a low value drives a warning (never a hard failure — see
            :func:`effective_sample_size` on the multimodal caveat).
    """

    healthy: bool
    insufficient_data: bool
    issues: list[str]
    warnings: list[str]
    acceptance_mean: float
    rejection_std: float
    Lambda: float | None = None
    tau_observed: float | None = None
    tau_predicted: float | None = None
    efficiency: float | None = None
    total_round_trips: int | None = None
    barrier_identified: bool | None = None
    recommended_n_chains: int | None = None
    efficiency_limiter: str | None = None
    barrier_peak_beta: float | None = None
    convergence_status: str | None = None
    rank_stability: float | None = None
    marginal_entropy: float | None = None
    min_ess: float | None = None
    median_ess: float | None = None
    ess_fraction: float | None = None

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = []
        if self.insufficient_data:
            lines.append("VERDICT: insufficient data — pass/fail withheld")
        elif self.healthy:
            lines.append("VERDICT: healthy — samples are reliable")
        else:
            lines.append("VERDICT: unhealthy — do not trust these samples")
        lines.append(
            f"  acceptance mean={self.acceptance_mean:.3f}  rejection std={self.rejection_std:.3f}"
        )
        if self.Lambda is not None:
            limiter = (
                f"  limiter={self.efficiency_limiter}"
                if self.efficiency_limiter
                else ""
            )
            unident = (
                "  BARRIER NOT IDENTIFIED (conveyor stalled)"
                if self.barrier_identified is False
                else ""
            )
            lines.append(
                f"  Lambda={self.Lambda:.3f}  tau_obs={self.tau_observed:.4f}  "
                f"tau_pred={self.tau_predicted:.4f}  "
                f"efficiency={self.efficiency:.3f}  "
                f"round_trips={self.total_round_trips}{limiter}{unident}"
            )
        if self.marginal_entropy is not None:
            note = ""
            if self.convergence_status not in (None, "CONVERGED"):
                note = " (informational only for multi-modal PT)"
            lines.append(
                f"  entropy={self.marginal_entropy:.3f}  convergence={self.convergence_status}{note}"
            )
        if self.min_ess is not None:
            lines.append(
                f"  ess(min)={self.min_ess:.1f}  ess(median)={self.median_ess:.1f}  "
                f"ess_fraction={self.ess_fraction:.3f}"
            )
        for issue in self.issues:
            lines.append(f"  ISSUE: {issue}")
        for warning in self.warnings:
            lines.append(f"  WARNING: {warning}")
        return "\n".join(lines)


def report_nrpt_diagnostics(
    stats: dict,
    samples: Bool[Array, "n_samples n_variables"] | None = None,
    *,
    tau_min: float = 0.01,
    efficiency_fail: float = 0.2,
    efficiency_warn: float = 0.35,
    rej_std_max: float = 0.15,
    entropy_frozen: float = 0.05,
    entropy_uniform: float = 0.95,
    min_attempts: int = 50,
    ess_warn: float = 0.1,
) -> NRPTHealthReport:
    """Evaluate NRPT stats (and optionally samples) into a single verdict.

    For NRPT, **round-trip diagnostics are the primary quality signal**:
    states must travel the full temperature ladder for tempering to work.
    Marginal-convergence checks are reported for information but never used
    as pass/fail criteria — when PT correctly samples multiple modes, the
    marginals shift between halves of the run and a naive convergence test
    produces false "NEED_MORE" verdicts.

    Decision criteria (each threshold is a keyword argument):

    - ISSUE: ``tau_observed < tau_min`` — no round trips, information is not
      flowing through the ladder.
    - ISSUE/WARN: ``efficiency < efficiency_fail`` / ``< efficiency_warn`` — the
      round-trip rate is below the ELE-optimal τ̄. The report sets
      ``efficiency_limiter`` to attribute the cause and point at the right knob:
      ``"schedule"`` when the ladder is not equalized
      (``std(rejection_rates) > rej_std_max`` — tune further / add chains) or
      ``"local_exploration"`` when it *is* equalized (an ELE violation — raise
      ``gibbs_steps_per_round``, or add chains as the alternative lever). A
      chain-count recommendation is included either way.
    - ISSUE: ``std(rejection_rates) > rej_std_max`` — schedule not equalized.
    - ISSUE: ``marginal_entropy < entropy_frozen`` — sampler frozen.
    - WARN:  ``marginal_entropy > entropy_uniform`` — β may be too low.
    - WARN:  a sharp peak in the λ(β) profile (barrier bottleneck).
    - WARN:  ``ess_fraction < ess_warn`` — worst-mixing variable has low
      effective sample size (informational; never a hard failure).

    All of these statistics are noisy when few swaps were attempted: when
    ``min(attempted) < min_attempts`` the would-be issues are demoted to
    warnings and ``insufficient_data`` is set instead of condemning a short
    tuning probe.

    **Arguments:**

    - `stats`: The stats dict returned by [`hamon.nrpt`][] /
      [`hamon.tune_schedule`][].
    - `samples`: Optional node-ordered boolean samples (e.g. from
      [`hamon.nrpt_node_samples`][]); enables the entropy and convergence
      sections.

    **Returns:**

    An :class:`NRPTHealthReport`. Issues are logged at WARNING level.
    """
    issues: list[str] = []
    warnings: list[str] = []

    acc = np.asarray(stats["acceptance_rate"])
    rej = np.asarray(stats["rejection_rates"])
    attempted = np.asarray(stats["attempted"])
    acceptance_mean = float(np.mean(acc))
    rejection_std = float(np.std(rej))

    insufficient = bool(np.min(attempted) < min_attempts)
    if insufficient:
        warnings.append(
            f"insufficient swap attempts (min {int(np.min(attempted))} < {min_attempts}); pass/fail verdict withheld"
        )

    def _flag(message: str) -> None:
        (warnings if insufficient else issues).append(message)

    if rejection_std > rej_std_max:
        _flag(f"rejection rates poorly equalized (std={rejection_std:.3f})")

    report = NRPTHealthReport(
        healthy=True,  # finalized below
        insufficient_data=insufficient,
        issues=issues,
        warnings=warnings,
        acceptance_mean=acceptance_mean,
        rejection_std=rejection_std,
    )

    rt = stats.get("round_trip_diagnostics")
    if rt is not None:
        from hamon.round_trips import barrier_is_identified

        report.Lambda = float(rt["Lambda"])
        report.tau_observed = float(rt["tau_observed"])
        report.tau_predicted = float(rt["tau_predicted"])
        report.efficiency = float(rt["efficiency"])
        report.total_round_trips = int(np.sum(np.asarray(rt["round_trips_per_chain"])))
        # Resolution is structural (Λ̂ <= N-1; a saturating ladder reports its cap),
        # so it reads the rejection rates, not the round-trip rate — the latter is
        # budget-dependent and would call a correct-but-under-observed ladder
        # stalled. See hamon.round_trips.barrier_is_identified.
        report.barrier_identified = barrier_is_identified(rej)

        if not report.barrier_identified:
            _flag(
                f"ladder saturates (max rejection={rej.max():.3f}) — barrier "
                f"estimate Lambda={report.Lambda:.2f} is capped by the chain "
                f"count (Lambda <= N-1 = {rej.size}) rather than measured, so it "
                f"is an underestimate; add chains / equalize the ladder"
            )
        if report.tau_observed < tau_min:
            # A separate, DYNAMICAL complaint: the conveyor was not observed
            # traversing. Deliberately does not claim Lambda is wrong — a
            # well-resolved ladder reads zero round trips on a short window (the
            # rate is budget-dependent where resolution is not).
            _flag(
                f"few round trips observed (tau_obs={report.tau_observed:.4f}) — "
                f"the conveyor is slow or the window is short, so samples "
                f"decorrelate slowly. On its own this does not invalidate "
                f"Lambda={report.Lambda:.2f} (see barrier_identified)"
            )
        elif report.efficiency < efficiency_warn:
            from hamon.round_trips import recommend_n_chains

            report.recommended_n_chains = recommend_n_chains(report.Lambda)
            # Attribute the inefficiency to the right knob. An equalized ladder
            # means the communication phase is already tuned, so the gap to the
            # ELE-optimal rate τ̄ is the local exploration kernel failing to
            # decorrelate the energy between swaps (an ELE violation) — raise
            # gibbs_steps_per_round, with more chains as the alternative lever.
            # An unequalized ladder means the schedule itself is the limiter.
            if rejection_std <= rej_std_max:
                report.efficiency_limiter = "local_exploration"
                msg = (
                    f"round-trip efficiency {report.efficiency:.3f} despite an "
                    f"equalized schedule (rejection std={rejection_std:.3f}): the "
                    f"local exploration kernel limits mixing (ELE violation). "
                    f"Raise gibbs_steps_per_round, or increase N to "
                    f"~{report.recommended_n_chains}."
                )
            else:
                report.efficiency_limiter = "schedule"
                msg = (
                    f"round-trip efficiency {report.efficiency:.3f} with an "
                    f"unequalized schedule (rejection std={rejection_std:.3f}): "
                    f"tune the schedule further or use "
                    f"~{report.recommended_n_chains} chains "
                    f"(Lambda={report.Lambda:.2f})."
                )
            # Severity follows the same fail/warn split as before.
            if report.efficiency < efficiency_fail:
                _flag(msg)
            else:
                warnings.append(msg)

        lam_profile = np.asarray(rt["lambda_profile"])
        peak_val = float(np.max(lam_profile))
        mean_val = float(np.mean(lam_profile))
        if mean_val > 0 and peak_val > 3 * mean_val:
            peak_idx = int(np.argmax(lam_profile))
            betas = np.asarray(stats["betas"])
            report.barrier_peak_beta = float(
                (betas[peak_idx] + betas[peak_idx + 1]) / 2
            )
            warnings.append(
                f"sharp barrier peak near beta={report.barrier_peak_beta:.3f} "
                f"(peak={peak_val:.3f}, mean={mean_val:.3f})"
            )
    else:
        warnings.append(
            "round trip diagnostics unavailable (track_round_trips=False); the primary NRPT quality signal is missing"
        )

    if samples is not None:
        # The marginal-entropy and convergence sections interpret samples as
        # binary 0/1 (per-variable Bernoulli entropy of a mean, top-k marginal
        # overlap) — on continuous (float) samples they would return
        # garbage/NaN, not diagnostics. Skip them for non-boolean samples; the
        # ESS section below is numeric-generic and stays.
        samples_binary = np.asarray(samples).dtype == np.bool_
        if samples_binary:
            conv = sample_convergence(samples)
            report.convergence_status = conv.status
            report.rank_stability = conv.rank_stability

            report.marginal_entropy = marginal_entropy(samples)
            if report.marginal_entropy < entropy_frozen:
                _flag(f"frozen marginals (entropy={report.marginal_entropy:.3f})")
            elif report.marginal_entropy > entropy_uniform:
                warnings.append(
                    f"near-uniform marginals (entropy={report.marginal_entropy:.3f}) — beta may be too low"
                )
        else:
            warnings.append(
                "samples are non-boolean (continuous model): entropy/convergence "
                "sections skipped; ESS still reported"
            )

        ess = effective_sample_size(samples)
        report.min_ess = ess.min_ess
        report.median_ess = ess.median_ess
        report.ess_fraction = ess.ess_fraction
        if ess.ess_fraction < ess_warn:
            warnings.append(
                f"low effective sample size (min ESS={ess.min_ess:.1f} of "
                f"{ess.n_samples}, fraction={ess.ess_fraction:.3f}) — "
                f"worst-mixing variable is highly autocorrelated"
            )

    report.healthy = not issues and not insufficient
    for issue in issues:
        logger.warning("NRPT health: %s", issue)
    for warning in warnings:
        logger.info("NRPT health: %s", warning)
    return report
