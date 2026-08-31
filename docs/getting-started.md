# Getting Started

## Installation

Hamon requires Python ≥ 3.12 and a working
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
    k_sample, program, schedule, init_state, clamp_state=[], obs_blocks=[Block(nodes)]
)
# samples shape: (500, 8) boolean array
```

## Adding parallel tempering

Single-chain Gibbs can get stuck in local minima. Non-reversible parallel
tempering (NRPT) runs multiple chains at different temperatures and shuffles
information between them. The easiest way to use it is **autotuning** — hamon
discovers the chain count, the local-exploration count, and the temperature
schedule, then draws from the target:

```python
from hamon import autosample


# init_factory builds one initial state per chain at the discovered chain count.
# It must extract free_blocks from the program so node identity is preserved.
def init_factory(n_chains, ebms, programs):
    fb = programs[0].gibbs_spec.free_blocks
    keys = jax.random.split(jax.random.key(7), n_chains)
    return [hinton_init(keys[c], ebms[0], fb, ()) for c in range(n_chains)]


samples, report = autosample(
    jax.random.key(1),
    n_samples=2000,
    ebm=model,  # one template EBM; β is rebased internally
    program=program,
    init_factory=init_factory,
    clamp_state=[],
    beta_range=(0.0, 1.0),  # reference (β=0) → target (β=1)
)

print(report.summary())  # discovered N, n_expl, Λ, round-trip efficiency
# samples shape: (2000, 8)
```

To draw more samples without re-tuning, keep the plan and reuse it:

```python
from hamon import autotune

plan = autotune(
    jax.random.key(2),
    ebm=model,
    program=program,
    init_factory=init_factory,
    clamp_state=[],
)
more = plan.sample(jax.random.key(3), 5000)  # cheap, repeatable
```

For Ising models specifically, `ising_sample(biases, edges, weights, key=...)`
wraps all of this into a single call.

## Manual control

If you want to drive the pieces yourself, the building blocks are public:
`tune_chains` (chain count), `tune_exploration` (local-exploration count),
`tune_schedule` (β ladder), and `nrpt` (a single run). For example, the core
run on a fixed ladder:

```python
from hamon.nrpt import nrpt

betas = [0.2, 0.5, 0.8, 1.0]  # hot → cold
ebms = [IsingEBM(nodes, edges, biases, weights, jnp.array(b)) for b in betas]
progs = [IsingSamplingProgram(e, free_blocks, []) for e in ebms]
keys = jax.random.split(jax.random.key(0), len(betas))
init_states = [hinton_init(keys[i], ebms[0], free_blocks, ()) for i in range(4)]

states, stats = nrpt(
    jax.random.key(1),
    ebms,
    progs,
    init_states,
    clamp_state=[],
    n_rounds=500,
    gibbs_steps_per_round=3,
)
print(f"Round-trip rate: {stats['round_trip_diagnostics']['tau_observed']:.4f}")
```

## What to read next

- [**Concepts**](concepts.md) — how blocks, factors, and tempering fit together
- [**Architecture**](architecture.md) — what Hamon optimizes under the hood
- [**Examples**](examples/01_block_gibbs_sampling.ipynb) — full worked notebooks
