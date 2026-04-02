<h1 align="center">Hamon</h1>

<p align="center">
JAX-native thermal sampling for discrete energy-based models.
</p>

<p align="center">
<a href="https://pypi.org/project/hamon"><img src="https://img.shields.io/pypi/v/hamon" alt="PyPI"></a>
<a href="https://pypi.org/project/hamon"><img src="https://img.shields.io/pypi/pyversions/hamon" alt="Python"></a>
<a href="https://github.com/dek3rr/hamon/blob/main/LICENSE"><img src="https://img.shields.io/github/license/dek3rr/hamon" alt="License"></a>
</p>

---

Hamon is a JAX library for sampling from discrete probabilistic graphical models.
It provides GPU-accelerated block Gibbs sampling, non-reversible parallel tempering
with adaptive schedule optimization, and tools for building and training Ising models,
RBMs, and other discrete energy-based models.

Built on [Extropic AI's thrml](https://github.com/Extropic-AI/thrml) foundation,
Hamon diverges as an independent library with original algorithmic contributions
and performance optimizations.

## Why "Hamon"?

In Japanese swordsmithing, the *hamon* (刃文, "blade pattern") is the visible
wave that appears along the edge of a katana after differential hardening. The
smith coats the blade in clay — thin along the cutting edge, thick along the
spine — then heats the steel to critical temperature and quenches it in water.
The edge cools fast into hard martensite; the spine cools slowly into tough
pearlite. The boundary between these two phases is the hamon: a pattern born
entirely from a thermal process, where controlled temperature gradients reveal
structure hidden in disordered steel.

The parallel to this library is direct. Hamon explores discrete energy
landscapes by running chains at different temperatures and exchanging
information across the thermal gradient. Structure emerges at the boundary
between mixing regimes — hot chains explore freely, cold chains resolve fine
detail, and the communication between them is what makes sampling work. The
hamon on a blade is proof that a thermal process found the right boundary.
The diagnostics in this library measure the same thing.

## Installation

```bash
pip install hamon
```

For development:

```bash
git clone https://github.com/dek3rr/hamon.git
cd hamon
pip install -e ".[development,testing,examples]"
```

Requires Python ≥ 3.10 and a JAX installation ([GPU setup guide](https://jax.readthedocs.io/en/latest/installation.html)).

## Quick example

```python
import jax
import jax.numpy as jnp
from hamon import SpinNode, Block, SamplingSchedule, sample_states
from hamon.models import IsingEBM, IsingSamplingProgram, hinton_init

nodes = [SpinNode() for _ in range(5)]
edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
model = IsingEBM(nodes, edges, jnp.zeros(5), jnp.ones(4) * 0.5, jnp.array(1.0))

free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

key = jax.random.key(0)
k_init, k_samp = jax.random.split(key, 2)
init_state = hinton_init(k_init, model, free_blocks, ())
schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
```

## Non-reversible parallel tempering

Hamon implements adaptive NRPT based on
[Syed et al. (2021)](https://arxiv.org/abs/1905.02939), with vectorized swaps
that exploit the temperature-linearity of Ising energies:

```python
from hamon.nrpt import nrpt_adaptive

ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
program = IsingSamplingProgram(ebm, free_blocks, [])

states, _, stats = nrpt_adaptive(
    jax.random.key(42),
    init_states=[init_state] * 8,
    clamp_state=[],
    n_rounds=500,
    gibbs_steps_per_round=5,
    initial_betas=jnp.linspace(0.1, 2.0, 8),
    n_tune=5,
    rounds_per_tune=200,
    ebm=ebm,
    program=program,
)

print(f"Final Λ: {stats['round_trip_diagnostics']['Lambda']:.3f}")
print(f"Round trip rate: {stats['round_trip_diagnostics']['tau_observed']:.4f}")
```

Key features of the NRPT implementation:

- **Vectorized swaps**: 1 energy evaluation per chain (not 4 per pair), all
  non-overlapping swaps execute simultaneously via permutation indexing
- **Adaptive scheduling**: iteratively tunes β spacing to equalize rejection
  rates, minimizing the global communication barrier Λ
- **Round trip tracking**: monitors the index process per machine, estimates
  Λ and predicted optimal rate τ̄ = 1/(2+2Λ)
- **Chain count discovery**: iteratively probes to find the right number of
  chains for a target acceptance rate

## What makes Hamon fast

**All chains run in one kernel.** Parallel tempering uses `jax.vmap` over chains
instead of a Python loop. Compile time is constant regardless of chain count.

**No redundant work in the sampler loop.** Global state is threaded through
`lax.scan` as a carry. Block updates use targeted scatters instead of rebuilding
the full state tensor each iteration.

**Energy evaluation skips unnecessary work.** Pre-built `BlockSpec` objects are
passed through directly — no reconstruction on every `energy()` call.

**Accumulator dtypes are explicit.** The moment accumulator pins its dtype at
construction, avoiding silent float64 promotion on GPU.

## Citing Hamon

If you use Hamon in your research, please cite:

```bibtex
@software{kerr2026hamon,
    author       = {Kerr, Douglas E. Jr.},
    title        = {Hamon: JAX-Native Thermal Sampling for Discrete Energy-Based Models},
    year         = {2026},
    url          = {https://github.com/dek3rr/hamon},
    version      = {0.2.0},
    license      = {Apache-2.0},
}
```

Hamon's block sampling and PGM infrastructure is derived from
[thrml](https://github.com/Extropic-AI/thrml) (v0.1.3) by
[Extropic AI](https://extropic.ai), licensed under Apache 2.0.
See [NOTICE](NOTICE) for full attribution. If you use the underlying
block Gibbs framework, please also cite:

```bibtex
@misc{jelincic2025efficient,
    title        = {An efficient probabilistic hardware architecture for diffusion-like models},
    author       = {Andraž Jelinčič and Owen Lockwood and Akhil Garlapati and Guillaume Verdon and Trevor McCourt},
    year         = {2025},
    eprint       = {2510.23972},
    archivePrefix= {arXiv},
    primaryClass = {cs.LG},
}
```

The non-reversible parallel tempering implementation is based on:

> Syed, S., Bouchard-Côté, A., Deligiannidis, G., & Doucet, A. (2021).
> Non-Reversible Parallel Tempering: a Scalable Highly Parallel MCMC Scheme.
> [arXiv:1905.02939](https://arxiv.org/abs/1905.02939)

## License

Apache 2.0. See [LICENSE](LICENSE).
