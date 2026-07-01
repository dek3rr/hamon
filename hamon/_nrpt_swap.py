"""Vectorized DEO (deterministic-even-odd) swap pass for NRPT.

One non-reversible swap parity per round, executed for all non-overlapping
chain pairs at once via a single permutation. Pure builders consumed by the
jitted round loop in ``hamon.nrpt``; ``_make_swap_branch`` is re-imported there.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hamon.round_trips import update_index_state


def _vectorized_swap(
    key: jax.Array,
    stacked_states: list,
    betas: jax.Array,
    base_energies: jax.Array,
    pair_indices: jax.Array,
    n_active: int,
    n_pairs: int,
    n_free_blocks: int,
    base_perm: jax.Array,
    live_chains: jax.Array | None = None,
) -> tuple[list, jax.Array, jax.Array]:
    """Execute all swaps for one set of non-overlapping pairs.

    With ``live_chains`` set (a traced chain count ≤ the padded ladder length),
    pairs at index ≥ live_chains − 1 are forced-rejected: the permutation stays
    identity there, so padding chains can never exchange with (or influence)
    the live ladder — DEO coupling is nearest-neighbour, so masking the
    boundary pair fully decouples the padding.

    Returns (new_states, accept_counts, permutation).
    """
    i_idx = pair_indices
    j_idx = pair_indices + 1

    log_r = (betas[i_idx] - betas[j_idx]) * (
        base_energies[i_idx] - base_energies[j_idx]
    )
    accept_probs = jnp.exp(jnp.minimum(0.0, log_r))
    u = jax.random.uniform(key, shape=(n_active,), dtype=accept_probs.dtype)
    accepted = u < accept_probs
    if live_chains is not None:
        accepted = accepted & (pair_indices < live_chains - 1)

    perm = base_perm
    perm = perm.at[i_idx].set(jnp.where(accepted, j_idx, i_idx))
    perm = perm.at[j_idx].set(jnp.where(accepted, i_idx, j_idx))
    new_states = [stacked_states[b][perm] for b in range(n_free_blocks)]

    acc = (
        jnp.zeros(n_pairs, dtype=jnp.int32)
        .at[pair_indices]
        .set(accepted.astype(jnp.int32))
    )

    return new_states, acc, perm


def _make_swap_branch(
    pair_indices: jax.Array,
    n_active: int,
    att_mask: jax.Array,
    betas: jax.Array,
    n_chains: int,
    n_pairs: int,
    n_free_blocks: int,
    base_perm: jax.Array,
    track_round_trips: bool,
    live_chains: jax.Array | None = None,
):
    """Build a lax.cond branch for even or odd swap pass.

    With ``live_chains`` set, swap attempts and the round-trip "top" are
    masked/redefined to the live prefix of a padded ladder (see
    ``_vectorized_swap``); attempt counters only advance for live pairs, so
    downstream acceptance/rejection rates over the sliced prefix are exactly
    what an unpadded ladder of ``live_chains`` chains would report.

    Returns (states, acc, att, idx_state, perm).
    """
    if live_chains is not None:
        att_mask = att_mask * (
            jnp.arange(n_pairs, dtype=jnp.int32) < live_chains - 1
        ).astype(att_mask.dtype)
    top_count = live_chains if live_chains is not None else n_chains

    def _branch(args):
        ss, ac, at, sk, bE, ist = args
        ss2, ac2, pm = _vectorized_swap(
            sk,
            ss,
            betas,
            bE,
            pair_indices,
            n_active,
            n_pairs,
            n_free_blocks,
            base_perm,
            live_chains,
        )
        # Static flag: with round-trip tracking disabled, the index-process
        # update is dropped from the compiled program entirely.
        new_ist = update_index_state(ist, pm, top_count) if track_round_trips else ist
        return (
            ss2,
            ac + ac2,
            at + att_mask,
            new_ist,
            pm,
        )

    return _branch
