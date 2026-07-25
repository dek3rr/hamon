"""NumPy twins of the NRPT schedule math, for the host-side tuning loop.

Private helpers re-imported into ``hamon.tuning``, where the iterative tuners
consume tiny swap counters in Python between NRPT batches. Running these
reductions through XLA instead would compile a distinct executable for every
live ladder length, so the duplication against the jitted originals —
``hamon._nrpt_schedule``'s PCHIP core and ``hamon.nrpt``'s ``optimize_schedule``
/ ``_acceptance_rate`` — is deliberate, not drift: the device path stays
JAX-native for production runs while tuning never round-trips to the device.

Pure NumPy leaves. Nothing here logs, traces, or touches placement, which is
what lets them live outside ``hamon.tuning`` at all (a module of its own would
otherwise log under the wrong logger name).
"""

from __future__ import annotations

import numpy as np


def _acceptance_rate_host(accepted, attempted, dtype) -> np.ndarray:
    """Acceptance rates for the small, host-driven tuning control loop."""
    accepted = np.asarray(accepted)
    attempted = np.asarray(attempted)
    return np.divide(
        accepted.astype(dtype),
        np.maximum(attempted, 1).astype(dtype),
        out=np.zeros_like(accepted, dtype=dtype),
        where=attempted > 0,
    )


def _pooled_lambda_host(accepted, attempted, dtype) -> float:
    return float(
        np.sum(
            np.asarray(1, dtype=dtype)
            - _acceptance_rate_host(accepted, attempted, dtype)
        )
    )


def _pchip_slopes_host(h: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """NumPy counterpart of the private JAX PCHIP tangent calculation."""
    h0, h1 = h[:-1], h[1:]
    d0, d1 = delta[:-1], delta[1:]
    w1 = 2.0 * h1 + h0
    w2 = h1 + 2.0 * h0
    same_sign = (d0 * d1) > 0
    safe0 = np.where(d0 == 0, 1.0, d0)
    safe1 = np.where(d1 == 0, 1.0, d1)
    denom = w1 / safe0 + w2 / safe1
    interior = np.where(same_sign, (w1 + w2) / np.where(same_sign, denom, 1.0), 0.0)

    def edge(hh0, hh1, m0, m1):
        d = ((2.0 * hh0 + hh1) * m0 - hh0 * m1) / (hh0 + hh1)
        d = np.where(np.sign(d) != np.sign(m0), 0.0, d)
        clamp = (np.sign(m0) != np.sign(m1)) & (np.abs(d) > 3.0 * np.abs(m0))
        return np.where(clamp, 3.0 * m0, d)

    left = edge(h[0], h[1], delta[0], delta[1]).astype(delta.dtype)
    right = edge(h[-1], h[-2], delta[-1], delta[-2]).astype(delta.dtype)
    return np.concatenate([left[None], interior.astype(delta.dtype), right[None]])


def _pchip_interp_host(xq: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.maximum(x[1:] - x[:-1], np.finfo(x.dtype).eps)
    delta = (y[1:] - y[:-1]) / h
    d = _pchip_slopes_host(h, delta)
    n = x.shape[0]
    idx = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, n - 2)
    hx = h[idx]
    t = (xq - x[idx]) / hx
    t2 = t * t
    t3 = t2 * t
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y[idx]
        + (t3 - 2.0 * t2 + t) * hx * d[idx]
        + (-2.0 * t3 + 3.0 * t2) * y[idx + 1]
        + (t3 - t2) * hx * d[idx + 1]
    )


def _optimize_schedule_host(rejection_rates, betas) -> np.ndarray:
    """Host-only schedule update used by tuning's Python control loop.

    This mirrors ``nrpt.optimize_schedule`` including endpoint pinning and its
    monotone-cubic inverse.  Keeping it here avoids changing the traced public
    helper while eliminating per-ladder XLA executables during chain discovery.
    """
    betas = np.asarray(betas)
    rej = np.asarray(rejection_rates, dtype=betas.dtype)
    cum = np.concatenate([np.zeros(1, dtype=betas.dtype), np.cumsum(rej)])
    target = np.linspace(0.0, cum[-1], len(betas), dtype=betas.dtype)
    if betas.shape[0] >= 3:
        new = _pchip_interp_host(target, cum, betas)
    else:
        new = np.interp(target, cum, betas).astype(betas.dtype, copy=False)
    new[0] = betas[0]
    new[-1] = betas[-1]
    return new


def _phase_diagnostics_host(rej, old_betas, new_betas, acceptance_rate):
    rej = np.asarray(rej)
    old_betas = np.asarray(old_betas)
    new_betas = np.asarray(new_betas)
    acceptance_rate = np.asarray(acceptance_rate)
    return (
        float(np.std(rej)),
        float(np.max(np.abs(new_betas - old_betas))),
        float(np.sum(rej)),
        float(np.mean(acceptance_rate)),
        float(np.max(rej)),
    )
