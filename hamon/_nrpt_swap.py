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
) -> tuple[list, jax.Array, jax.Array]:
    """Execute all swaps for one set of non-overlapping pairs.

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
):
    """Build a lax.cond branch for even or odd swap pass.

    Returns (states, acc, att, idx_state, perm).
    """

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
        )
        # Static flag: with round-trip tracking disabled, the index-process
        # update is dropped from the compiled program entirely.
        new_ist = update_index_state(ist, pm, n_chains) if track_round_trips else ist
        return (
            ss2,
            ac + ac2,
            at + att_mask,
            new_ist,
            pm,
        )

    return _branch
