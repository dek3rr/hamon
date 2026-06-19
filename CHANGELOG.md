# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`sample_states_batched`** — runs several independent single-chain draws in
  parallel under one `jax.vmap`, and `ising_sample` gains an `n_draw_chains`
  argument (default `1`, exact previous behaviour) that splits the sample budget
  across that many chains, each seeded from the equilibrated cold state. The
  draw is dispatch-bound on an accelerator, so fewer, wider kernels collect the
  same total samples in less wall time.

### Fixed

- **NRPT round-loop recompiled on every call** — `BlockSpec` now has value-based
  `__eq__`/`__hash__` keyed on its structure (block partition, global-state
  layout, sampling order, SD map). `program.with_ebm(...)` rebuilds the spec on
  every `nrpt` / `nrpt_adaptive` call and every `discover_chain_count` probe;
  the fresh spec object previously missed the `eqx.filter_jit` cache for
  `_nrpt_rounds`, forcing a full recompile (~1 s at 484 nodes / 16 chains) even
  though the round loop itself runs in microseconds. Repeated NRPT runs at the
  same scale now compile once and reuse the executable (~40–60× faster steady
  state); a cold `ising_sample` improves ~25–30%.

### Changed

- **`discover_chain_count` rewritten with an extrapolation-based policy** —
  replaces the fixed "step halfway toward the max-Λ recommendation" loop. It now takes a
  cheap, ceiling-independent pilot probe (so `initial_n` defaults to `None` — no
  initial guess needed), extrapolates Λ to N→∞ from the two latest probes,
  recommends N with a margin sized to the observed rejection-rate spread and
  barrier growth, and returns that recommendation directly (no monotone ratchet
  up to the last probed N, which previously caused systematic overshoot). On a
  surrogate suite of varied barriers this converges in ~2 probes and roughly
  doubles the fraction of problems that land within tolerance of the target
  acceptance, without undershooting.
- **Convergence-driven NRPT tuning (default)** — `nrpt_adaptive` and
  `discover_chain_count` now auto-allocate their tuning budgets instead of
  running fixed `n_tune` × `rounds_per_tune` phases. Each tuning phase runs only
  as many rounds as the Λ estimate needs to settle (`round_batch` increments up
  to the `rounds_per_tune` ceiling); the best-equalised schedule seen is kept
  for production (not the noisy last one); and phases stop once the ladder is
  equalised or its movement is at the Monte-Carlo floor for `phase_patience`
  consecutive phases (capped at `n_tune`). `n_tune`/`rounds_per_tune` become
  safety caps. Pass `adaptive_tuning=False` for the exact previous behaviour.
  Evaluated across easy→hard Ising problems (chain/grid, ferro/frustrated):
  adaptive matches the old fixed-full schedule's correctness (cold-chain
  marginals vs exact enumeration) and round-trip efficiency while using fewer
  tuning rounds, stays healthy where an under-budgeted fixed config does not,
  and is insensitive to a bad initial β ladder. Counts are seed-deterministic
  but problem-dependent — do not assume a fixed round/phase count.

- **Per-block global-state layout** — when a block-Gibbs program is
  "split-safe" (every free block reads each of its tail blocks from a single
  block, e.g. a 2-coloured grid), each free block now occupies its own
  global-state slot instead of all same-structure blocks being concatenated
  into one array. A block update then replaces its slot outright rather than
  `dynamic_update_slice`-ing into a shared array, removing the device-to-device
  copies of the unchanged portion on every sweep (fewer, cheaper kernels).
  Sampling output is bit-identical; programs that are not split-safe keep the
  previous concatenated layout.

## [0.4.0] — 2026-06-12

### Added

- **Automatic CPU/GPU device routing** — public entry points (`nrpt`,
  `nrpt_adaptive`, `discover_chain_count`, `ising_sample`, `sample_states`,
  `sample_with_observation`, `estimate_moments`, `estimate_kl_grad`) take a
  `device` argument, default `"auto"`: with no accelerator visible placement
  is untouched; otherwise the work score `n_chains × free nodes` routes small
  workloads to the CPU and large ones to the accelerator, so installing CUDA
  jax never makes a workload slower than CPU-only jax. The default threshold
  (4096) is the steady-state crossover measured on an RTX 5080, where every
  sweep point at score ≥ 4096 ran 2–11× faster on GPU; override via
  `HAMON_DEVICE_THRESHOLD` (calibrate with
  `benchmarks/device_crossover.py`); force with `HAMON_DEVICE=cpu|gpu|none`;
  full opt-out with `device=None`. Orchestrators resolve the device once and
  reuse it across tuning phases, preserving the jit-once round loop.
  `hamon.resolve_device` is exported for pre-resolving with custom thresholds.
- **`nrpt_node_samples`** — converts NRPT observer output into node order in
  one call (``samples[:, i]`` is the state of ``nodes[i]``), replacing the
  ~20 lines of block-concatenation and permutation-inversion every observer
  user previously had to write, where a forgotten inversion produced
  plausible-looking but scrambled samples.
- **`report_nrpt_diagnostics` / `NRPTHealthReport`** — a single
  "did sampling work?" verdict built on round-trip diagnostics (the primary
  PT quality signal), schedule equalization, and optional sample entropy.
  Marginal-convergence checks are reported but never used as pass/fail —
  correct multi-modal PT shifts marginals between run halves. Thresholds are
  keyword arguments; verdicts are withheld (not failed) when swap-attempt
  counts are too low to judge. Low efficiency auto-suggests a chain count
  via `recommend_n_chains`. `ising_sample` now includes the report under
  ``diagnostics["health"]``.
- **`nrpt` accepts stacked initial states** — ``init_states`` may be a
  single block-state list with a leading ``(n_chains, ...)`` axis, e.g.
  straight from ``hinton_init(key, model, blocks, (n_chains,))``, instead of
  a per-chain list of lists.
- **`nrpt_adaptive(tune_tol=...)`** — optional early stop for schedule
  tuning: when an update moves every β by less than the tolerance, the
  remaining phases are skipped. Each phase now logs one INFO line
  (Λ, mean acceptance, schedule movement) and records ``max_beta_shift`` in
  the tuning history.

### Fixed

- **`nrpt` validates the β ladder** — a descending or shuffled ladder
  previously ran without error while silently breaking the cold-chain
  convention (``states[-1]``) and the DEO pairing; it now raises, as does a
  betas/chain-count length mismatch.
- **Float32 models stay float32 under x64** — enabling `jax_enable_x64` in the
  host application (common when hamon is mixed with double-precision
  analytics) used to promote the entire device sampling loop to float64
  through hamon's internal scalars: `IsingEBM`'s β, the `jnp.array(0.0)`
  energy seed in `AbstractFactorizedEBM.energy`, the β ladder and reference-β
  scalars in `nrpt`, and the round-trip diagnostics. β values are now cast to
  the float dtype of the interaction weights, so the model's parameters alone
  decide the compute precision; pass float64 weights to opt in to
  double-precision sampling. Verified by `tests/test_dtype_preservation.py`,
  which runs NRPT under x64 with strict dtype promotion.

### Changed

- **Test suite defaults to the CPU device** — tiny test models are
  compile/dispatch-bound on GPU (~4× slower end to end). The GPU stays
  enumerable: a new `gpu` pytest marker runs a smoke subset on real hardware
  (auto-skipped when absent), and `HAMON_TEST_DEVICE=gpu` runs the whole
  suite on the GPU.
- **Persistent XLA compilation cache in the test suite** (GPU backends only) —
  the suite compiles hundreds of small programs, and on GPU compilation
  dominates wall time; with a warm cache (`~/.cache/jax`) GPU runs are ~3×
  faster. Set `JAX_COMPILATION_CACHE_DIR` to override.
- **Single source of truth for block layout** — the contiguity check and
  node-position lookup that existed in three copies (`scatter_block_to_global`,
  `from_global_state`, `BlockSamplingProgram.__init__`) now live in one
  `_block_layout` helper; `get_node_locations` is reimplemented on top of it.
  No behavioral change.
- **Shared Ising-grid test fixture** — `tests/utils.make_ising_grid` replaces
  per-file copies of the lattice builder.
- **One dispatch point for NRPT's template/factory routes** — the
  template-vs-factory mode check existed three times (`nrpt_adaptive` twice,
  `discover_chain_count` once) with duplicated phase/probe call sites. A new
  internal `_ChainSource` owns the dispatch: `nrpt_adaptive` has a single
  `_run_phase`, `discover_chain_count` a single probe call that forwards the
  caller's route. Jit-cache behavior is preserved (the template route hands
  back the identical β = 1 pair every phase). `nrpt_adaptive` also validates
  `init_states` up front instead of failing later with a shape error.
- `EdgePartition` documented as analysis/planning tooling (it is not part of
  the sampling pipeline).

### Removed

- Dead `init_energy_cache` / `update_energy_cache` in `boundary_energy` —
  never called by the library or tests, and their β-division convention
  contradicted how `nrpt` actually consumes `energy_delta_fn`.

## [0.3.0] — 2026-06-10

### Added

- **Jit-once NRPT round loop** — the Gibbs + DEO swap scan now lives in a
  module-level `eqx.filter_jit` function, so the compilation cache persists
  across `nrpt` calls. Templates already at β = 1 are reused without
  rebasing, and `nrpt_adaptive` rebases once before the phase loop, so all
  tuning phases plus production trace and compile **exactly once** (β arrays
  and states are traced data). Verified by a trace-count regression test;
  measured 1.80s → 0.54s (~3.3×) per `nrpt_adaptive` call on an 8-chain
  48×48 Ising benchmark (CPU).

- **Temperature-linear NRPT mode** — `nrpt` now accepts a *single* template
  `(ebm, program)` pair (plus an explicit `betas` array) instead of per-chain
  sequences. One base program is built at β = 1 and every interaction array
  is scaled by the chain's β inside the vmapped Gibbs kernel — valid for any
  model whose interactions are linear in β (the `DiscreteEBMFactor` family),
  consistent with the E_β = β·E_base assumption the swap math already makes.
  This avoids constructing one program per chain and storing per-chain copies
  of every interaction tensor (n_chains× less interaction memory).
  `nrpt_adaptive` and `discover_chain_count` use this mode automatically on
  their template (`ebm=`/`program=`) routes, eliminating all per-phase
  EBM/program rebuilds during schedule tuning (~22% faster adaptive tuning
  on CPU for an 8-chain 48×48 Ising benchmark). Results are bit-identical
  to the per-chain-programs path. Explicit factory routes are unchanged.

### Breaking

- **Minimum supported Python is now 3.11** (was 3.10). The JAX (≥ 0.9) and
  jaxtyping (≥ 0.3) releases hamon is developed against both require
  Python ≥ 3.11, so the previous 3.10 floor could only resolve to stale
  dependency versions. Python 3.14 is now supported and tested in CI.
- **Padded interaction entries are pre-zeroed at program construction** —
  `BlockSamplingProgram` masks the sliced interaction tensors with the active
  flags once, and the built-in spin/categorical conditionals no longer
  multiply by the active mask on every Gibbs step. Custom samplers invoked
  through a `BlockSamplingProgram` may now rely on inactive entries being
  zero; samplers called directly with hand-built (unmasked) interactions must
  keep applying the active flags themselves.

### Changed

- **jaxtyping floor raised to 0.3.10** (was 0.2.23) — picks up the fix for
  `PyTree[A | B]` isinstance checks silently passing (0.3.9) and cloudpickle
  round-trips of variadic annotations like `Shaped[Array, "..."]` (0.3.10)
- **optax floor (testing extra) raised to 0.2.8** (was 0.2.4) — older optax
  imports the `jax_pmap_shmap_merge` config option removed in JAX 0.10.0 and
  fails at import
- **CI test and example matrices now cover Python 3.11–3.14** (was 3.10–3.13)
- **Gibbs scan carries one copy of the sampling state, not two** — the
  `_run_blocks` scan previously threaded both the per-block state list and
  the concatenated global state through the carry, although samplers only
  read the global state. The carry now holds just the sampler states and
  the global state; per-block states are extracted once after the scan.
  `from_global_state` gained the same contiguous-slice fast path as the
  write-back side, so the extraction lowers to static slices. Results are
  bit-identical; measured ~6% faster NRPT rounds on CPU (the duplicate
  carry was multiplied by `vmap` across chains) and ~3% faster plain Gibbs.
- **Block write-back uses contiguous slice updates instead of scatters** —
  free blocks always occupy contiguous ranges of the global state (a
  `BlockSpec` layout invariant), so the per-block write-back in the Gibbs
  scan and in `scatter_block_to_global` now lowers to
  `lax.dynamic_update_slice` with a precomputed static offset instead of a
  gather-index scatter, which XLA fuses far better (the isolated op is ~30×
  faster on CPU; scatters are disproportionately expensive on GPU).
  Non-contiguous node sets keep the scatter fallback. Results are
  bit-identical.
- **`track_round_trips=False` now skips the index-process update** inside the
  swap pass instead of only omitting the summary.
- **Conditional samplers accumulate in the weights' dtype** — the spin and
  categorical Gibbs conditionals previously seeded their parameter
  accumulators with float32 zeros regardless of the model dtype.
- Documentation refreshed for the new internals: `architecture.md` now
  describes the concatenated (not padded) global state layout and the actual
  index-process representation; stale `hinton_init` and `CategoricalNode`
  docstrings corrected.

### Fixed

- **`sample_blocks` no longer mutates the caller's state and sampler-state
  lists.**
- **NRPT observers received post-swap states paired with pre-swap energies** —
  in the default (non-cached) energy mode, the base-energy vector was not
  permuted after accepted swaps before being handed to the observer, so
  `base_energies[c]` described the state that *used to* occupy chain `c`.
  Energies are now permuted alongside the states in both energy modes.
- **β₀ = 0 produced NaN base energies in NRPT, silently rejecting every swap** —
  `nrpt` recovered base energies by dividing the hottest chain's energy by β₀,
  which is 0/0 when the ladder is anchored at the reference distribution
  (`beta_range=(0.0, ...)`, the `discover_chain_count` default and the range
  `ising_sample` uses). Swap acceptance became NaN and every swap was rejected
  with no error, degrading parallel tempering into independent chains and
  inflating Λ estimates. Base energies are now computed from an exact β = 1
  copy of the EBM via `with_beta()`, falling back to the coldest chain's β for
  EBM classes without `with_beta()` (raising a clear error if that β is 0).

## [0.2.0] — 2026-04-02

### Breaking

- **NRPT return signature** — refactored `nrpt` / `nrpt_adaptive` to remove unused
  return values; callers that destructure the full tuple will need updating

### Added

- **Observer support for NRPT** — collect per-round samples via pluggable observers
- **Tuning diagnostics** — utilities for inspecting adaptive schedule behaviour
- **`ising_sample` wrapper** — high-level convenience function for Ising model sampling
- **Template EBM/program objects** — `nrpt_adaptive` and `discover_chain_count` now
  accept a template object directly, reducing boilerplate
- **Boundary energy ↔ NRPT integration** — connected `boundary_energy` module to the
  NRPT pipeline
- **NRPT correctness test**

### Fixed

- Random key propagation and block-edge validation in dynamic blocks
- Flaky `TestBigGrid` stabilized
- Lazy import guard for docs build
- MkDocs warnings resolved

### Changed

- Internal NRPT refactor (cleaner scan carry, simplified state threading)
- Removed old benchmarks
- Refreshed example notebooks
- Converted leftover thrml references in docs

## [0.1.0] — 2026-03-08

First release under the **Hamon** name. Hamon is a spiritual successor to
[Extropic AI's thrml](https://github.com/Extropic-AI/thrml) (v0.1.3), diverging
as an independent library focused on GPU-accelerated thermal sampling for discrete
energy-based models.

### Added

- **Non-reversible parallel tempering** (`nrpt`, `nrpt_adaptive`)
  - Vectorized swap pass exploiting temperature-linearity: 1 energy eval per chain
    instead of 4 per adjacent pair
  - Single-pass DEO (deterministic even-odd) swap scheduling
  - Adaptive schedule optimization (Algorithm 4, Syed et al. 2021) to equalize
    rejection rates and minimize the global communication barrier Λ
  - Iterative chain count discovery (`discover_chain_count`)
- **Round trip tracking** (`round_trips` module)
  - Index process monitoring carried through `lax.scan` with minimal overhead
  - Communication barrier estimation: local λ(β) and global Λ
  - Predicted vs observed round trip rate diagnostics
  - Chain count recommendation from Λ estimates
- **Dynamic block construction** (`dynamic_blocks` module)
  - Influence-aware partitioning: aggregate influence A(w) identifies heavy vertices
  - Per-temperature block sizing based on correlation length heuristics
  - Correlation-based re-blocking from empirical samples (Venugopal & Gogate 2013)
  - Influence-weighted Hamming distance for mixing diagnostics
- **Boundary energy deltas** (`boundary_energy` module)
  - Edge classification (incident, boundary, interior, external) per block partition
  - Rectangular block construction with 4-coloring for 2D grids
- **vmap parallel tempering** — all chains run in a single kernel via `jax.vmap`,
  replacing the original Python for-loop that unrolled N copies into XLA
- **Scan carry threading** — global state carried through `lax.scan` with targeted
  scatter updates; no redundant `block_state_to_global` per iteration
- **BlockSpec fast path** — `energy()` accepts pre-built `BlockSpec` / `BlockGibbsSpec`
  directly, skipping reconstruction on every call
- **Precomputed scatter indices** on `BlockSamplingProgram` (`_block_sd_inds`,
  `_block_positions`, `_block_output_sds`)
- Comprehensive test suite for all new modules

### Fixed

- **Deterministic global state layout** — replaced `set` with `dict.fromkeys` for
  `global_sd_order` in `BlockSpec.__init__`; state ordering is now reproducible
- **MomentAccumulatorObserver dtype** — pinned at construction to avoid silent
  float64 promotion on GPU
- **Non-array pytree leaves under vmap** — `_stack_pbi_across_chains` preserves
  Python ints for slice indexing inside vmapped function bodies

### Changed

- Renamed from `thrml-boost` to `hamon`; this project no longer tracks upstream
  thrml changes
- Package directory: `thrml_boost/` → `hamon/`
- Version reset to 0.1.0 to reflect new project identity

### Attribution

Core block sampling, factor, PGM, and observer infrastructure derived from
[thrml](https://github.com/Extropic-AI/thrml) (v0.1.3) by Extropic AI,
licensed under Apache 2.0. See [NOTICE](NOTICE) for details.
