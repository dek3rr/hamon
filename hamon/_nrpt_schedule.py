"""Monotone-cubic (Fritsch–Carlson / PCHIP) interpolation core for NRPT.

Private numeric helpers behind ``hamon.nrpt.optimize_schedule``: they invert the
cumulative communication barrier Λ with a shape-preserving monotone cubic, which
Syed et al. (2021) recommend over piecewise-linear interpolation. The public
``optimize_schedule`` wrapper lives in ``hamon.nrpt``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _pchip_slopes(h: jax.Array, delta: jax.Array) -> jax.Array:
    """Fritsch–Carlson (PCHIP) monotone-cubic tangents at the knots.

    ``h`` are the knot spacings and ``delta`` the secant slopes (both length
    ``n - 1`` for ``n`` knots). Interior tangents use the weighted harmonic mean
    of the two adjacent secants — zeroed at local extrema — which guarantees the
    cubic Hermite interpolant is monotone; the endpoints use Fritsch's one-sided
    estimate with the standard clamping. Returns ``n`` tangents.
    """
    dtype = delta.dtype
    h0, h1 = h[:-1], h[1:]
    d0, d1 = delta[:-1], delta[1:]
    w1 = 2.0 * h1 + h0
    w2 = h1 + 2.0 * h0
    same_sign = (d0 * d1) > 0
    # Guard the unused (opposite-sign) branch against division by zero.
    safe0 = jnp.where(d0 == 0, 1.0, d0)
    safe1 = jnp.where(d1 == 0, 1.0, d1)
    denom = w1 / safe0 + w2 / safe1
    interior = jnp.where(same_sign, (w1 + w2) / jnp.where(same_sign, denom, 1.0), 0.0)

    def _edge(hh0, hh1, m0, m1):
        d = ((2.0 * hh0 + hh1) * m0 - hh0 * m1) / (hh0 + hh1)
        d = jnp.where(jnp.sign(d) != jnp.sign(m0), 0.0, d)
        clamp = (jnp.sign(m0) != jnp.sign(m1)) & (jnp.abs(d) > 3.0 * jnp.abs(m0))
        return jnp.where(clamp, 3.0 * m0, d)

    left = _edge(h[0], h[1], delta[0], delta[1]).astype(dtype)
    right = _edge(h[-1], h[-2], delta[-1], delta[-2]).astype(dtype)
    return jnp.concatenate([left[None], interior.astype(dtype), right[None]])


def _pchip_interp(xq: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    """Evaluate the Fritsch–Carlson monotone cubic through ``(x, y)`` at ``xq``."""
    h = jnp.maximum(x[1:] - x[:-1], jnp.finfo(x.dtype).eps)  # eps guards ties
    delta = (y[1:] - y[:-1]) / h
    d = _pchip_slopes(h, delta)
    n = x.shape[0]
    idx = jnp.clip(jnp.searchsorted(x, xq, side="right") - 1, 0, n - 2)
    hx = h[idx]
    t = (xq - x[idx]) / hx
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * y[idx] + h10 * hx * d[idx] + h01 * y[idx + 1] + h11 * hx * d[idx + 1]
