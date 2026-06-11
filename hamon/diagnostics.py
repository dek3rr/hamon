"""Diagnostics for sample quality and model health."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Bool, Shaped

logger = logging.getLogger(__name__)


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
    samples = jnp.asarray(samples)
    n_samples, n_vars = samples.shape
    target_k = min(target_k, n_vars)

    quartile_indices = [n_samples * q // 4 for q in range(1, 5)]
    marginals = [
        jnp.mean(samples[:idx].astype(jnp.float32), axis=0) for idx in quartile_indices
    ]

    drifts = [
        float(jnp.mean(jnp.abs(marginals[i + 1] - marginals[i])))
        for i in range(len(marginals) - 1)
    ]

    # Rank stability: Jaccard of top-k between first and second half.
    half = n_samples // 2
    m_first = jnp.mean(samples[:half].astype(jnp.float32), axis=0)
    m_second = jnp.mean(samples[half:].astype(jnp.float32), axis=0)

    top_first = set(jnp.argsort(-m_first)[:target_k].tolist())
    top_second = set(jnp.argsort(-m_second)[:target_k].tolist())
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
    p = jnp.mean(jnp.asarray(samples).astype(jnp.float32), axis=0)
    # Use jnp.where to handle p=0 and p=1 without NaN from 0*log(0).
    safe_p = jnp.clip(p, 1e-10, 1.0 - 1e-10)
    h = -(safe_p * jnp.log2(safe_p) + (1 - safe_p) * jnp.log2(1 - safe_p))
    # Zero out entropy for variables that are truly frozen.
    h = jnp.where((p == 0.0) | (p == 1.0), 0.0, h)
    return float(jnp.mean(h))


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
    biases = jnp.asarray(biases)
    edges = jnp.asarray(edges)
    weights = jnp.asarray(weights)
    n = biases.shape[0]

    bias_contributions = beta * jnp.abs(biases)
    bias_spread = float(jnp.max(bias_contributions) - jnp.min(bias_contributions))

    # Sum of absolute coupling weights incident on each node.
    abs_w = beta * jnp.abs(weights)
    coupling_per_node = jnp.zeros(n)
    coupling_per_node = coupling_per_node.at[edges[:, 0]].add(abs_w)
    coupling_per_node = coupling_per_node.at[edges[:, 1]].add(abs_w)
    coupling_per_spin = float(jnp.mean(coupling_per_node))

    if bias_spread > 0:
        ratio = coupling_per_spin / bias_spread
    else:
        ratio = float("inf")

    if ratio < warn_low:
        logger.warning(
            "Energy balance ratio %.3f < %.3f: biases dominate, "
            "couplings may be irrelevant.",
            ratio,
            warn_low,
        )
    elif ratio > warn_high:
        logger.warning(
            "Energy balance ratio %.3f > %.1f: couplings dominate, "
            "biases may be irrelevant.",
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
        Lambda / tau_observed / tau_predicted / efficiency /
        total_round_trips: Round-trip diagnostics (``None`` when the run was
            made with ``track_round_trips=False``).
        recommended_n_chains: Suggested chain count when efficiency is low.
        barrier_peak_beta: Midpoint β of a sharp barrier peak, if detected.
        convergence_status / rank_stability / marginal_entropy: Sample-based
            metrics (``None`` when *samples* was not provided). For NRPT,
            convergence is informational only — correct multi-modal sampling
            shifts marginals between run halves, so a non-CONVERGED status is
            **not** treated as a failure.
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
    recommended_n_chains: int | None = None
    barrier_peak_beta: float | None = None
    convergence_status: str | None = None
    rank_stability: float | None = None
    marginal_entropy: float | None = None

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
            f"  acceptance mean={self.acceptance_mean:.3f}  "
            f"rejection std={self.rejection_std:.3f}"
        )
        if self.Lambda is not None:
            lines.append(
                f"  Lambda={self.Lambda:.3f}  tau_obs={self.tau_observed:.4f}  "
                f"tau_pred={self.tau_predicted:.4f}  "
                f"efficiency={self.efficiency:.3f}  "
                f"round_trips={self.total_round_trips}"
            )
        if self.marginal_entropy is not None:
            note = ""
            if self.convergence_status not in (None, "CONVERGED"):
                note = " (informational only for multi-modal PT)"
            lines.append(
                f"  entropy={self.marginal_entropy:.3f}  "
                f"convergence={self.convergence_status}{note}"
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
    - ISSUE: ``efficiency < efficiency_fail`` — schedule badly miscalibrated
      (the report then includes a chain-count recommendation).
    - WARN:  ``efficiency < efficiency_warn``.
    - ISSUE: ``std(rejection_rates) > rej_std_max`` — schedule not equalized.
    - ISSUE: ``marginal_entropy < entropy_frozen`` — sampler frozen.
    - WARN:  ``marginal_entropy > entropy_uniform`` — β may be too low.
    - WARN:  a sharp peak in the λ(β) profile (barrier bottleneck).

    All of these statistics are noisy when few swaps were attempted: when
    ``min(attempted) < min_attempts`` the would-be issues are demoted to
    warnings and ``insufficient_data`` is set instead of condemning a short
    tuning probe.

    **Arguments:**

    - `stats`: The stats dict returned by [`hamon.nrpt`][] /
      [`hamon.nrpt_adaptive`][].
    - `samples`: Optional node-ordered boolean samples (e.g. from
      [`hamon.nrpt_node_samples`][]); enables the entropy and convergence
      sections.

    **Returns:**

    An :class:`NRPTHealthReport`. Issues are logged at WARNING level.
    """
    issues: list[str] = []
    warnings: list[str] = []

    acc = jnp.asarray(stats["acceptance_rate"])
    rej = jnp.asarray(stats["rejection_rates"])
    attempted = jnp.asarray(stats["attempted"])
    acceptance_mean = float(jnp.mean(acc))
    rejection_std = float(jnp.std(rej))

    insufficient = bool(jnp.min(attempted) < min_attempts)
    if insufficient:
        warnings.append(
            f"insufficient swap attempts (min {int(jnp.min(attempted))} < "
            f"{min_attempts}); pass/fail verdict withheld"
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
        report.Lambda = float(rt["Lambda"])
        report.tau_observed = float(rt["tau_observed"])
        report.tau_predicted = float(rt["tau_predicted"])
        report.efficiency = float(rt["efficiency"])
        report.total_round_trips = int(jnp.sum(rt["round_trips_per_chain"]))

        if report.tau_observed < tau_min:
            _flag(
                f"near-zero round trip rate (tau_obs="
                f"{report.tau_observed:.4f}) — information not flowing"
            )
        elif report.efficiency < efficiency_fail:
            from hamon.round_trips import recommend_n_chains

            report.recommended_n_chains = recommend_n_chains(report.Lambda)
            _flag(
                f"low round-trip efficiency ({report.efficiency:.3f}); "
                f"consider {report.recommended_n_chains} chains for "
                f"Lambda={report.Lambda:.2f}"
            )
        elif report.efficiency < efficiency_warn:
            warnings.append(
                f"round-trip efficiency below {efficiency_warn} "
                f"({report.efficiency:.3f})"
            )

        lam_profile = jnp.asarray(rt["lambda_profile"])
        peak_val = float(jnp.max(lam_profile))
        mean_val = float(jnp.mean(lam_profile))
        if mean_val > 0 and peak_val > 3 * mean_val:
            peak_idx = int(jnp.argmax(lam_profile))
            betas = jnp.asarray(stats["betas"])
            report.barrier_peak_beta = float(
                (betas[peak_idx] + betas[peak_idx + 1]) / 2
            )
            warnings.append(
                f"sharp barrier peak near beta={report.barrier_peak_beta:.3f} "
                f"(peak={peak_val:.3f}, mean={mean_val:.3f})"
            )
    else:
        warnings.append(
            "round trip diagnostics unavailable (track_round_trips=False); "
            "the primary NRPT quality signal is missing"
        )

    if samples is not None:
        conv = sample_convergence(samples)
        report.convergence_status = conv.status
        report.rank_stability = conv.rank_stability

        report.marginal_entropy = marginal_entropy(samples)
        if report.marginal_entropy < entropy_frozen:
            _flag(f"frozen marginals (entropy={report.marginal_entropy:.3f})")
        elif report.marginal_entropy > entropy_uniform:
            warnings.append(
                f"near-uniform marginals "
                f"(entropy={report.marginal_entropy:.3f}) — beta may be too low"
            )

    report.healthy = not issues and not insufficient
    for issue in issues:
        logger.warning("NRPT health: %s", issue)
    for warning in warnings:
        logger.info("NRPT health: %s", warning)
    return report
