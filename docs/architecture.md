# Architecture

Hamon's internals are organized around one idea: **block-structured Gibbs
sampling on concatenated PyTree states, compiled once via JAX**.

## The core abstraction

A `BlockSamplingProgram` holds everything needed to run block Gibbs on a single
model: the `BlockGibbsSpec` (which blocks to sample, which to clamp), the
conditional samplers for each block, and the factors that define the energy.
At construction the program pre-slices every interaction tensor per block,
pre-zeroes padded interaction entries using the active mask, and precomputes
each block's contiguous slice into the global state.

The `BlockSpec` manages the mapping between block-local and global state arrays.
Per-node *interaction* data is padded to the maximum neighbor count so it is
rectangular (with an `active` mask marking real entries), which costs some
wasted FLOPs but avoids Python-level loops that would make XLA compile times
scale with node count.

## State representation

Global state is a list of arrays, one per distinct node structure (shape/dtype
template). All blocks sharing a structure are **concatenated along the node
axis**, each block occupying a contiguous range; `node_global_location_map`
records every node's position. `block_state_to_global` and `from_global_state`
convert between the per-block and concatenated representations, and because
block ranges are contiguous these lower to static slices rather than
gathers/scatters.

Node identity matters: the same `SpinNode()` object must appear in both the EBM
and the block definitions. Hamon enforces this through `init_factory`, which
always extracts `free_blocks = programs[0].gibbs_spec.free_blocks` rather than
capturing block objects from an outer scope.

## Parallel tempering layout

NRPT runs \( N \) chains, each at a different inverse temperature \( \beta_i \).
Rather than looping over chains in Python (which unrolls \( N \) copies of the
computation graph in XLA), Hamon stacks all chain states into a leading dimension
and uses `jax.vmap` for the Gibbs sweeps.

The tempering round is a `lax.scan` loop:

```
for round in range(n_rounds):
    # 1. Gibbs sweeps (vmapped across chains)
    states = vmap(gibbs_sweep)(states)

    # 2. DEO swap proposals (single parity)
    parity = round % 2
    states, accepted = deo_swap(states, energies, parity)
```

Swap decisions use the Metropolis-Hastings criterion with the
temperature-linearity trick: for Ising models, \( E(\beta, x) = \beta \cdot E(1, x) \),
so swap acceptance ratios reduce to
\( \exp\bigl((\beta_{i+1} - \beta_i)(V^{(i+1)} - V^{(i)})\bigr) \)
without recomputing energies at each temperature.

Temperature linearity is exploited end-to-end: `nrpt` accepts a single
template `(ebm, program)` pair, rebases it to \( \beta = 1 \), and scales the
interaction arrays by each chain's \( \beta \) inside the vmapped kernel —
no per-chain program construction and no per-chain copies of the weight
tensors. The whole round loop lives in a module-level jitted function, so
repeated calls with the same base pair (e.g. the tuning phases of
`tune_schedule`) compile exactly once; the \( \beta \) schedule is traced
data and can change freely between phases.

## Energy caching

When an `energy_delta_fn` is provided (e.g. from `make_ising_delta_fn`), Hamon
computes the full energy only once at round 0, then maintains a running cache
that is updated incrementally after each Gibbs sweep. After swaps, the cache is
permuted to match the new chain ordering.

This eliminates the \( O(|\mathcal{E}|) \) energy recomputation that would
otherwise dominate each round.

## Index process tracking

The index process tracks which chain position each machine's state occupies
over time. `index_state` is a dict of four `(n_chains,)` arrays:
`machine_to_chain` (current ladder position per machine), `visited_top`
(whether the machine has reached the coldest chain since its last completed
trip), and `round_trips` / `restarts` counters.

Because DEO swaps are disjoint transpositions, the swap permutation is
self-inverse (`inv_perm == perm`), so updating the index state after a swap
requires only a single gather — no `jnp.argsort`. When
`track_round_trips=False`, the update is omitted from the compiled program
entirely.

Round-trip counting and \( \Lambda \) estimation happen in `round_trip_summary`
from the accumulated index state and rejection rates.

## Schedule optimization

`optimize_schedule` implements the equi-acceptance reparameterization from
Algorithm 4 of Syed et al. (2021). Given observed rejection rates
\( r_0, \ldots, r_{N-2} \), it constructs the CDF of the rejection cost and
inverts it to find \( \beta \) values that equalize the per-level contribution
to \( \Lambda \).

`tune_schedule` wraps this in a loop: run a short burn-in, measure rejections,
reposition betas, repeat for `n_tune` phases, then run production.

## Autotuning orchestration

`autotune` composes the three searches in dependency order, cheap to expensive:
`tune_chains` (at `n_expl=1`) for the chain count, then `tune_exploration` at
that fixed count — **reusing** the schedule via the `fixed_schedule` argument, so
each exploration probe is one production run rather than a full re-tune (sound
because the schedule is invariant to `n_expl`) — then a short `tune_schedule`
polish that also leaves an equilibrated cold-chain state. The result is an
`NRPTPlan` whose `sample()` draws from that warm state, so repeat draws skip
tuning entirely and reuse the compiled loop. `tune_exploration` measures the
steady-state per-round wall time on the target device (warm-up first, then the
median of timed runs), which is what lets the chosen `n_expl` reflect real
hardware cost rather than the idealized "cost ∝ `n_expl`" model. Because each
distinct chain count and each distinct `n_expl` recompiles the round loop,
`autotune` enables JAX's persistent compilation cache by default so those
recompiles are amortized across probes and across runs.

## Class hierarchy

### Factors

```
AbstractFactor
├── WeightedFactor
│   └── DiscreteEBMFactor
│       ├── SquareDiscreteEBMFactor
│       │   ├── SpinEBMFactor
│       │   └── SquareCategoricalEBMFactor
│       └── CategoricalEBMFactor
└── EBMFactor
```

### Conditional samplers

```
AbstractConditionalSampler
└── AbstractParametricConditionalSampler
    ├── BernoulliConditional
    │   └── SpinGibbsConditional
    └── SoftmaxConditional
        └── CategoricalGibbsConditional
```

### Observers

```
AbstractObserver
├── StateObserver
└── MomentAccumulatorObserver
```

### Models

```
AbstractEBM
└── AbstractFactorizedEBM
    └── IsingEBM
```
