# Concepts

This page explains the ideas behind Hamon without requiring prior knowledge of
probabilistic graphical models or MCMC. If you already know this material, skip
to the [API reference](api/pgm.md).

## Energy-based models

An energy-based model (EBM) assigns a scalar energy \( E(x) \) to every possible
configuration \( x \) of a set of discrete variables. Lower energy means higher
probability:

\[
  p(x) = \frac{1}{Z} \exp\bigl(-E(x)\bigr), \qquad Z = \sum_x \exp\bigl(-E(x)\bigr)
\]

The partition function \( Z \) makes this a proper distribution, but computing it
requires summing over every configuration — exponential in the number of
variables. Hamon sidesteps this by *sampling* from \( p(x) \) without ever
computing \( Z \).

### Ising models

The simplest discrete EBM. Each variable (spin) takes values in \(\{-1, +1\}\).
The energy is:

\[
  E(x) = -\beta \Bigl(\sum_{(i,j) \in \mathcal{E}} w_{ij}\, x_i x_j + \sum_i h_i\, x_i\Bigr)
\]

where \( \beta \) is inverse temperature, \( w_{ij} \) are coupling weights,
and \( h_i \) are biases. Hamon represents these as `IsingEBM` objects.

## Nodes, blocks, and factors

Hamon models the dependency structure as a bipartite factor graph.

**Nodes** are the random variables. `SpinNode` for \(\pm 1\) variables,
`CategoricalNode` for multi-valued variables.

**Factors** encode interactions between groups of nodes. A `WeightedFactor`
stores a weight tensor and connects a set of nodes via an `InteractionGroup`.

**Blocks** are disjoint subsets of nodes that can be updated simultaneously. In a
two-color Ising model, one block holds even-indexed spins and the other holds
odd-indexed spins. Because spins in the same block don't interact directly, they
can be sampled in parallel — this is the core of block Gibbs sampling.

A `BlockSpec` manages the index bookkeeping: which nodes live in which block,
how to scatter block-level updates back into the global state, and how to pad
variable-size blocks so JAX can stack them into arrays.

## Block Gibbs sampling

Gibbs sampling updates one variable at a time from its conditional distribution.
Block Gibbs updates an entire block at once. Hamon's `sample_blocks` function
sweeps through each free block in sequence, sampling all its nodes in parallel
given the current state of everything else.

```
for each sweep:
    for block in free_blocks:
        sample all nodes in block | rest of state
```

A `SamplingSchedule` controls warmup (burn-in), the number of samples to
collect, and how many sweeps to run between recorded samples.

## The temperature problem

Block Gibbs at a single temperature can get trapped in local energy minima,
especially when the model has frustrated interactions or phase transitions.
The sampler mixes locally but can't cross energy barriers.

The solution: run the same model at multiple temperatures simultaneously.

## Parallel tempering

Parallel tempering (PT) maintains \( N \) copies of the system at inverse
temperatures \( 0 = \beta_0 < \beta_1 < \cdots < \beta_N = 1 \):

- **Hot chains** (\( \beta \approx 0 \)) sample almost uniformly — they explore
  freely but carry no information about the target distribution.
- **Cold chains** (\( \beta = 1 \)) sample from the target — they resolve fine
  structure but can't escape local minima.
- **Swap moves** periodically propose exchanging states between adjacent
  temperature levels. Accepted swaps transport information from hot to cold.

The round-trip rate \( \tau \) measures how quickly information flows through the
temperature ladder. A state that starts at the reference (\( \beta = 0 \)),
travels to the target (\( \beta = 1 \)), and returns has completed one round
trip.

### DEO vs. SEO

Hamon uses **Deterministic Even-Odd (DEO)** swap communication, which makes the
index process *non-reversible*. On even rounds, swap proposals go to pairs
\((0,1), (2,3), \ldots\); on odd rounds, \((1,2), (3,4), \ldots\).

The non-reversibility creates a directional drift through the temperature ladder
— a conveyor belt effect that roughly doubles the round-trip rate compared to
the reversible alternative (Stochastic Even-Odd, SEO).

!!! warning "Single-pass only"
    Hamon applies one parity per round, not both. Composing even-then-odd swaps
    in the same round produces the identity permutation, destroying the
    non-reversibility that makes DEO work.

### Adaptive schedules

The spacing of temperature levels matters. Hamon implements Algorithm 4 from
[Syed et al. (2021)](https://arxiv.org/abs/1905.02939): it monitors rejection
rates across the ladder and repositions \( \beta \) values to equalize
communication cost. `tune_schedule` runs this tuning loop automatically.

### Communication barrier

The global communication barrier \( \Lambda \) is the fundamental quantity that
governs tempering performance:

\[
  \tau_\infty = \frac{1}{2 + 2\Lambda}
\]

where \( \Lambda = \sum_{i=0}^{N-1} \frac{r_i}{1 - r_i} \) and \( r_i \) is the
rejection rate between chains \( i \) and \( i+1 \). Hamon reports \( \Lambda \)
in the round-trip diagnostics and uses it in `tune_chains` to recommend
how many chains to use.

## What Hamon optimizes

All of the above — block Gibbs, parallel tempering, schedule adaptation — is
mathematically standard. Hamon's contribution is making it fast in JAX:

- **`jax.vmap` over chains**: all \( N \) tempering chains run as a single
  vectorized kernel, not a Python loop. Compile time is constant in \( N \).
- **`lax.scan` carry threading**: global state flows through the scan loop
  without redundant reconstruction each iteration.
- **Vectorized DEO swaps**: swap decisions are computed for all pairs in one
  `jnp.where` call, exploiting temperature-linearity of Ising energies.
- **Incremental ΔE tracking**: `boundary_energy` classifies edges and computes
  energy deltas from block updates without recomputing the full energy.

See the [architecture guide](architecture.md) for implementation details.
