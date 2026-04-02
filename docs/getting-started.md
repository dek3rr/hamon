# Getting Started

## Installation

Hamon requires Python ≥ 3.10 and a working
[JAX installation](https://jax.readthedocs.io/en/latest/installation.html).

=== "pip"

    ```bash
    pip install hamon
    ```

=== "From source"

    ```bash
    git clone https://github.com/dek3rr/hamon.git
    cd hamon
    pip install -e .
    ```

=== "With extras"

    ```bash
    # Notebooks and plotting
    pip install hamon[examples]

    # Development (ruff, pyright, pytest)
    pip install -e ".[development,testing,examples]"
    ```

!!! tip "JAX GPU setup"
    Hamon itself is pure Python — the GPU acceleration comes from JAX. Make sure
    you have a JAX build that matches your CUDA version. See the
    [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html).

## Your first model

A minimal Ising chain: 8 spins, nearest-neighbor coupling, sampled with
two-color block Gibbs.

```python
import jax
import jax.numpy as jnp
from hamon import SpinNode, Block, SamplingSchedule, sample_states
from hamon.models import IsingEBM, IsingSamplingProgram, hinton_init

# 1. Define the graph
nodes = [SpinNode() for _ in range(8)]
edges = [(nodes[i], nodes[i + 1]) for i in range(7)]

# 2. Build the model
biases = jnp.zeros(8)
weights = jnp.ones(7) * 0.4
beta = jnp.array(1.0)
model = IsingEBM(nodes, edges, biases, weights, beta)

# 3. Set up block Gibbs — even/odd checkerboard
free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

# 4. Sample
key = jax.random.key(42)
k_init, k_sample = jax.random.split(key)

init_state = hinton_init(k_init, model, free_blocks, ())
schedule = SamplingSchedule(n_warmup=200, n_samples=500, steps_per_sample=2)

samples = sample_states(
    k_sample, program, schedule, init_state,
    clamp_state=[], obs_blocks=[Block(nodes)]
)
# samples shape: (500, 8) boolean array
```

## Adding parallel tempering

Single-chain Gibbs can get stuck in local minima. Non-reversible parallel
tempering (NRPT) runs multiple chains at different temperatures and shuffles
information between them.

```python
from hamon.nrpt import nrpt, nrpt_adaptive

betas = [0.2, 0.5, 0.8, 1.0]  # hot → cold
ebms = [IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas]
progs = [IsingSamplingProgram(e, free_blocks, []) for e in ebms]

keys = jax.random.split(jax.random.key(0), len(betas))
init_states = [hinton_init(keys[i], ebms[0], free_blocks, ()) for i in range(len(betas))]

states, _, stats = nrpt(
    jax.random.key(1),
    ebms, progs, init_states,
    clamp_state=[],
    n_rounds=500,
    gibbs_steps_per_round=3,
)

print(f"Acceptance rates: {stats['acceptance_rate']}")
print(f"Round-trip rate:  {stats['round_trip_diagnostics']['tau_observed']:.4f}")
```

## Adaptive schedule

Let Hamon optimize the temperature ladder automatically:

```python
states, _, stats = nrpt_adaptive(
    jax.random.key(2),
    ebm_factory=lambda b: IsingEBM(nodes, edges, biases, weights, b),
    program_factory=lambda e: IsingSamplingProgram(e, free_blocks, []),
    init_states=init_states,
    clamp_state=[],
    n_rounds=500,
    gibbs_steps_per_round=3,
    initial_betas=jnp.array(betas),
    n_tune=5,
    rounds_per_tune=100,
)

# stats["tuning_history"] has Λ and β schedules from each adaptation phase
```

## What to read next

- [**Concepts**](concepts.md) — how blocks, factors, and tempering fit together
- [**Architecture**](architecture.md) — what Hamon optimizes under the hood
- [**Examples**](examples/01_all_of_hamon.ipynb) — full worked notebooks
