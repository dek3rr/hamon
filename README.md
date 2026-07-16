<h1 align="center">Hamon</h1>

<p align="center">
JAX-native thermal sampling for discrete and continuous energy-based models.
</p>

<p align="center">
<a href="https://pypi.org/project/hamon"><img src="https://img.shields.io/pypi/v/hamon" alt="PyPI"></a>
<a href="https://pypi.org/project/hamon"><img src="https://img.shields.io/pypi/pyversions/hamon" alt="Python"></a>
<a href="https://github.com/dek3rr/hamon/blob/main/LICENSE"><img src="https://img.shields.io/github/license/dek3rr/hamon" alt="License"></a>
</p>

---

Hamon is a JAX library for sampling from probabilistic graphical models —
discrete and continuous. It provides GPU-accelerated block Gibbs sampling,
non-reversible parallel tempering with adaptive schedule optimization, and
tools for building and training Ising models, RBMs, Gaussian Markov random
fields, and continuous multimodal (φ⁴) lattice models.

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

The parallel to this library is direct. Hamon explores energy
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

Requires Python ≥ 3.12 and a JAX installation ([GPU setup guide](https://jax.readthedocs.io/en/latest/installation.html)).

## Device routing

With CUDA jax installed, JAX places everything on the GPU — including the
small, dispatch-bound programs where a CPU finishes several times faster.
hamon's entry points (`nrpt`, `tune_schedule`, `tune_chains`,
`ising_sample`, `sample_states`, `sample_with_observation`, …) therefore take
a `device` argument:

- `"auto"` (default) — with no accelerator visible, placement is untouched.
  Otherwise the work score `n_chains × free nodes` decides: small workloads
  run on the CPU, large ones on the accelerator. The default threshold (4096,
  the steady-state crossover measured on an RTX 5080) can be overridden with
  `HAMON_DEVICE_THRESHOLD` (calibrate yours with
  `python benchmarks/device_crossover.py`); `HAMON_DEVICE=cpu|gpu|none`
  forces a choice without code changes. Very short one-shot flows are
  compile-dominated and can favor the CPU regardless of size — pass
  `device="cpu"` for those, or set `JAX_COMPILATION_CACHE_DIR` so repeated
  runs skip GPU compilation entirely.
- `"cpu"` / `"gpu"` — that platform, raising if it is not visible.
- a concrete `jax.Device` — used as-is.
- `None` — hamon never touches placement.

Routing re-commits the entry arrays (program tensors, states, β ladder) to
the chosen device and returns outputs committed there; pass `device=None` to
keep full manual control of placement. Orchestrators resolve the device once
and reuse it across all tuning phases, so jit caches stay warm.

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

## Continuous models

The sampling engine is dtype-generic — continuous states flow through the same
block-Gibbs and NRPT machinery as spins. Three stacks ship in `hamon.models`:

- **Gaussian MRFs, sampled exactly.** The single-site conditionals of a GMRF
  are themselves Gaussian, so block Gibbs over a graph coloring stays *exact*
  — within a color class they are independent scalar Gaussians, no linear
  solve anywhere. All interactions are β-linear, so NRPT's temperature-linear
  template mode applies bit-exactly. Verified against the closed form
  `N(P⁻¹h, (βP)⁻¹)` (mean and full covariance to Monte-Carlo precision),
  both for plain block Gibbs and for the cold chain of a tempered ladder.

- **Continuous *multimodal* targets — the case tempering exists for.**
  `DoubleWellEBM` is the lattice φ⁴ field: at cold β with ferromagnetic
  couplings the target is bimodal and a single chain mode-collapses; NRPT
  round trips carry the ± flips. The single-site conditional has no closed
  form, so `SliceGibbsConditional` performs one slice-sampling transition per
  site (Neal 2003, the exactly-reversible bounded variant), vectorized over
  each color class and verified against quadrature. Slice draws are keyed so
  that chain masking stays bit-identical despite data-dependent loop lengths.

- **Reference annealing — β can start at exactly 0.** An unbounded state
  space has no proper β = 0 member (no uniform distribution over ℝⁿ), so
  these models report `proper_at_beta_zero = False` and `nrpt` / `tune_chains`
  / `autotune` reject a ladder starting at exactly β = 0 — either use
  `beta_range=(β_min > 0, 1.0)`, or anneal from a proper reference:
  `AnnealedEBM(reference, target, β)` implements the standard PT path
  `E_β = (1−β)·E_ref + β·E_target`, whose β = 0 member is the *reference at
  full weight*. Every rung is then proper and the ladder covers the full
  entropic path. NRPT's template mode handles this with an affine
  interpolation (`offset + β·slope`) and swap energies `Δ = E₁ − E₀` — the
  shared reference term cancels in every swap ratio, verified against a
  per-rung closed form on an all-Gaussian ladder.

The Gaussian quick example mirrors the Ising one:

```python
import jax
import jax.numpy as jnp
from hamon import Block, GaussianNode, SamplingSchedule, sample_states
from hamon.models import GaussianEBM, GaussianSamplingProgram, gaussian_init

nodes = [GaussianNode() for _ in range(5)]
edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
model = GaussianEBM(
    nodes, edges,
    diag=jnp.full(5, 2.0),        # precision diagonal (diagonally dominant ⇒ PD)
    lin=jnp.zeros(5),             # linear term h
    couplings=jnp.full(4, -0.5),  # off-diagonal precision per edge
    beta=jnp.array(1.0),
)

free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = GaussianSamplingProgram(model, free_blocks, clamped_blocks=[])

key = jax.random.key(0)
k_init, k_samp = jax.random.split(key)
init_state = gaussian_init(k_init, model, free_blocks, ())
schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
```

And a φ⁴ target annealed from a Gaussian reference, tempered from β = 0:

```python
from hamon.models import AnnealedEBM, DoubleWellEBM, DoubleWellSamplingProgram

reference = GaussianEBM(nodes, [], jnp.full(5, 2.0), jnp.zeros(5),
                        jnp.zeros(0), jnp.array(1.0))
target = DoubleWellEBM(nodes, edges,
                       barrier=jnp.ones(5),           # well coefficient a
                       lin=jnp.zeros(5),
                       couplings=jnp.full(4, -0.6),   # ferromagnetic ⇒ bimodal
                       beta=jnp.array(1.0))
annealed = AnnealedEBM(reference, target, jnp.array(1.0))
program = DoubleWellSamplingProgram(annealed, free_blocks, clamped_blocks=[])
# beta_range=(0.0, 1.0) is valid here: every rung of the ladder is proper.
```

## Non-reversible parallel tempering

Hamon implements adaptive NRPT based on
[Syed et al. (2021)](https://arxiv.org/abs/1905.02939), with vectorized swaps
that exploit the temperature-linearity of Ising energies. **The primary
interface is autotuning** — `autotune` / `autosample` discover the chain count,
the local-exploration count, and the schedule for you:

```python
from hamon import autosample

# Tunes N, gibbs_steps_per_round, and the β ladder, then draws from the target.
samples, report = autosample(
    jax.random.key(0),
    n_samples=2000,
    ebm=ebm,                  # a single template EBM (any β)
    program=program,
    init_factory=init_factory,  # (n_chains, ebms, programs) -> [init per chain]
    clamp_state=[],
    beta_range=(0.0, 1.0),
)
print(report.summary())       # N, n_expl, Λ, round-trip efficiency

# Or keep the tuned plan and draw repeatedly without re-tuning:
plan = autotune(jax.random.key(1), ebm=ebm, program=program,
                init_factory=init_factory, clamp_state=[])
more = plan.sample(jax.random.key(2), 5000)
```

For Ising models, `ising_sample` wraps this in a one-liner (biases, edges,
weights → samples) and autotunes everything automatically.

Key design elements:

- **Full autotuning**: `autotune` runs chain count → exploration count →
  schedule in dependency order and returns an `NRPTPlan` for cheap repeated
  draws. Every default is chosen to be **reproducible**: identical inputs give
  identical tuning decisions and samples.
- **Robust chain-count discovery**: `tune_chains` pilots at `max_chains` (an
  over-resolved ladder gives an unbiased first Λ̂) and takes
  `N* = 2Λ + 1`, the round-trip optimum at rejection r\* = ½ (Syed et al.).
  The **running-max Λ̂** over probes keeps glassy targets — where a coarse
  ladder under-resolves the barrier and biases Λ̂ low — from collapsing to a
  chain count that cannot mix. `seed_from_energy` skips the pilot using the
  closed-form energy-variance barrier (Theorem 2), gated by a Gelman–Rubin R̂
  across independent restarts that falls back to the pilot when local
  exploration traps.
- **Deterministic exploration count**: `gibbs_steps_per_round` defaults to a
  fixed device-calibrated value (accelerator → 4, CPU → 1) at the flat top of
  the ESS-per-second curve. The wall-timed search (`search_exploration=True`)
  is opt-in because its argmax is not reproducible across runs — best used
  once per hardware, then pinned.
- **Glassy-target stability**: schedule tuning ranks any *unsaturated* ladder
  above a saturated one — a rung pinned at ~100% rejection severs the DEO
  conveyor, making Λ̂ = Σ rej a within-basin artifact rather than a barrier
  estimate — and stops once Λ̂ plateaus rather than when a phase cap runs out.
  Chain-count discovery therefore consumes a stable Λ̂ even on glassy targets,
  and its `N* = 2Λ + 1` fixed point converges in a couple of probes instead of
  chasing tuning noise.
- **Trustworthy diagnostics**: a two-part round-trip trust gate.
  `barrier_identified` asks a *structural* question — does the ladder
  saturate? (max rejection < 0.75, a threshold calibrated across nine model
  families in 2–4 dimensions, including glasses and a continuous-symmetry
  clock model) — so "identified" means Λ̂ is within ~10% regardless of the
  round budget. `conveyor_alive` separately answers the *dynamical* question
  (round-trip efficiency ≥ 0.15, reported as unmeasured rather than stalled
  when the window affords fewer than 40 expected trips).
  `efficiency_limiter` attributes low round-trip efficiency to the schedule
  vs. local exploration; per-variable ESS and opt-in log Z via thermodynamic
  integration (`NRPTEnergyObserver`).
- **Multimodal-safe draws**: `NRPTPlan.sample` / `autosample` default to a
  *tempered* draw — the tuned ladder keeps running and the cold chain is
  recorded each round, so DEO swaps keep carrying barrier crossings into the
  samples. `tempered=False` restores the cheaper single-chain cold-β draw for
  unimodal targets, and warns when the tuning run's round-trip evidence says
  the target is multimodal and the decoupled draw would mode-collapse.
- **Vectorized swaps + temperature-linear mode**: one energy evaluation per
  chain, all non-overlapping swaps as a single permutation; one β = 1 base
  program serves every chain with interactions scaled by β inside the kernel
  (no per-chain program construction or interaction copies). Reference-annealed
  EBMs (`AnnealedEBM`) use the affine variant: interactions interpolate as
  `offset + β·slope` and swap energies use `Δ = E₁ − E₀`, under the same
  shared-program machinery.

### Log Z and effective sample size

```python
import jax.numpy as jnp
from hamon import NRPTEnergyObserver, nrpt_log_normalizing_constant
from hamon.nrpt import tune_schedule

obs = NRPTEnergyObserver(n_chains=8)
states, stats = tune_schedule(
    jax.random.key(0),
    init_states=[init_state] * 8,
    clamp_state=[],
    n_rounds=500,
    gibbs_steps_per_round=5,
    initial_betas=jnp.linspace(0.0, 1.0, 8),
    ebm=ebm,
    program=program,
    observer=obs,  # opt-in: accumulates mean energy on the production run
)

# log Z(1) for an n-spin model (β=0 reference is uniform over 2**n states).
log_z = nrpt_log_normalizing_constant(stats, log_z0=len(nodes) * jnp.log(2.0))

# Effective sample size of the cold-chain trace.
from hamon import effective_sample_size, report_nrpt_diagnostics

report = report_nrpt_diagnostics(stats, samples=my_cold_chain_samples)
print(report.summary())  # includes ess(min)/ess(median)/ess_fraction
```

## What makes Hamon fast

On a GPU the wall-clock cost of tuning-heavy sampling is **XLA compilation,
not the sampling itself** (measured ~85% of a cold chain-count search; the
actual device work is well under a second). Hamon is engineered so everything
compiles once:

**One kernel, compiled once, for every configuration.** All chains run under
one `jax.vmap` (compile time is flat in chain count), the round count is
traced (any number of rounds reuses one executable), and **chain masking**
pads the ladder to a fixed width with the live count as traced data — so the
entire autotune (every probe, polish, production run, and the tempered draw,
which records the cold chain at a *traced* index via `ColdIndexObserver`)
shares a single compiled round loop, even as the discovered chain count
drifts across a parameter sweep or training run. Masking is bit-identical to
the unpadded run: JAX's key/uniform streams are prefix-stable and masked
swaps keep the identity permutation.

**Caches that actually hit.** Jit caches key on program *structure*
(value-based `BlockSpec` equality), so `with_ebm` rebuilds and repeated tuner
calls reuse executables; the persistent compile cache is on by default in
`autotune`, so a repeat run in a new process does **zero** XLA compiles.

**A lean sampler loop.** State threads through `lax.scan` as a carry with
static-offset slice writebacks; post-hoc diagnostics run in host numpy (no
per-shape kernel compiles, one device→host transfer) and hot paths avoid
per-edge host syncs and eager dispatch.

## Citing Hamon

If you use Hamon in your research, please cite:

```bibtex
@software{kerr2026hamon,
    author       = {Kerr, Douglas E. Jr.},
    title        = {Hamon: JAX-Native Thermal Sampling for Discrete and Continuous Energy-Based Models},
    year         = {2026},
    url          = {https://github.com/dek3rr/hamon},
    version      = {0.10.0},
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
