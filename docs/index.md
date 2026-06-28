# Hamon

**JAX-native thermal sampling for discrete energy-based models.**

---

Hamon gives you GPU-accelerated block Gibbs sampling and non-reversible parallel
tempering for Ising models, Boltzmann machines, and other discrete energy-based
models — all in pure JAX.

```python
import jax
from hamon import SpinNode, Block, SamplingSchedule, sample_states
from hamon.models import IsingEBM, IsingSamplingProgram, hinton_init
import jax.numpy as jnp

# Define a 5-spin Ising chain
nodes = [SpinNode() for _ in range(5)]
edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
model = IsingEBM(nodes, edges, jnp.zeros(5), jnp.ones(4) * 0.5, jnp.array(1.0))

# Two-color block Gibbs
free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, [])

key = jax.random.key(0)
k1, k2 = jax.random.split(key)
state = hinton_init(k1, model, free_blocks, ())
schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

samples = sample_states(k2, program, schedule, state, [], [Block(nodes)])
```

## Why "Hamon"?

In Japanese swordsmithing, the *hamon* (刃文) is the visible wave along a katana's
edge — a pattern created entirely by differential hardening. The smith coats the
blade in clay, heats it to critical temperature, and quenches it. The edge cools
fast into hard martensite; the spine cools slowly into tough pearlite. The hamon
is the boundary between phases, born from a thermal process.

This library does the same thing computationally. It runs MCMC chains at different
temperatures and exchanges information across the thermal gradient. Structure
emerges at the boundary between mixing regimes — hot chains explore, cold chains
resolve detail, and communication between them is what makes sampling work.

## What Hamon provides

**Block Gibbs sampling** on sparse, heterogeneous factor graphs with JAX-native
compilation. Define your model as nodes, factors, and blocks; Hamon handles the
index bookkeeping and padding for you.

**Non-reversible parallel tempering (NRPT)** with single-pass DEO swaps and
round-trip diagnostics, based on the theoretical framework of
[Syed et al. (2021)](https://arxiv.org/abs/1905.02939).

**One-call autotuning** via `autotune` / `autosample`: hamon discovers the
chain count, the local-exploration count, *and* the temperature schedule for
you, then draws from the target. The exploration count is calibrated to your
hardware by maximizing effective samples per measured wall-second. This is the
primary way to solve a problem with hamon; the individual tuners (`tune_chains`,
`tune_exploration`, `tune_schedule`) are available for manual control.

**Dynamic block management** with influence-aware partitioning, per-temperature
block sizing, and correlation-based re-blocking.

## Origin

Hamon began as a performance fork of
[Extropic AI's THRML](https://github.com/Extropic-AI/thrml) library. It has
since diverged into an independent project with its own algorithmic contributions.
The original work is gratefully acknowledged under the
[Apache 2.0 license](https://github.com/dek3rr/hamon/blob/main/NOTICE).

## Next steps

<div class="grid" markdown>

[**Getting started** — install and run your first model](getting-started.md){ .md-button }

[**Concepts** — understand the building blocks](concepts.md){ .md-button }

[**API reference** — full module documentation](api/pgm.md){ .md-button }

</div>
