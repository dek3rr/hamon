"""Base-energy computation for NRPT swaps.

NRPT's vectorized swap decision needs E_base(x) = E_β(x) / β for every chain
(temperature linearity). These pure helpers are split out of ``hamon.nrpt`` and
re-imported there; tests also import them directly from ``hamon.nrpt``.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from hamon.models.ebm import AbstractEBM


def _compute_base_energies(
    ebm_ref: AbstractEBM,
    beta_ref: jax.Array,
    spec,
    stacked_states: list,
    clamp_state: list,
) -> jax.Array:
    """Compute E_base(x) for all chains via vmap. Shape: (n_chains,).

    E_base = ebm_ref.energy(x, spec) / β_ref (temperature linearity).
    β_ref must be nonzero; callers should prefer a β=1 reference EBM so the
    division is exact (see `_make_reference_ebm`).
    """

    def _energy_one_chain(*block_slices):
        state = list(block_slices) + clamp_state
        return ebm_ref.energy(state, spec)

    return jax.vmap(_energy_one_chain)(*stacked_states) / beta_ref


def _make_reference_ebm(
    ebms: Sequence[AbstractEBM], betas: jax.Array
) -> tuple[AbstractEBM, jax.Array]:
    """Pick the (EBM, β) pair used to recover base energies E_base = E(x)/β.

    Using the hottest chain (β₀) breaks when β₀ = 0 — a standard NRPT ladder
    anchored at the reference distribution — because E(x) is then identically
    0 and the division yields NaN, which silently rejects every swap. Prefer
    an exact β=1 copy of the EBM so no division error is possible; for EBM
    classes that do not implement `with_beta()`, fall back to the coldest
    chain, whose β is the largest (best-conditioned) divisor in the ladder.
    """
    try:
        return ebms[-1].with_beta(jnp.asarray(1.0)), jnp.asarray(1.0)
    except NotImplementedError:
        if float(betas[-1]) == 0.0:
            raise ValueError(
                "Cannot compute base energies: the coldest chain has β = 0 and "
                f"{type(ebms[-1]).__name__} does not implement with_beta(). "
                "Either implement with_beta() or use a ladder whose coldest "
                "chain has β > 0."
            )
        return ebms[-1], betas[-1]
