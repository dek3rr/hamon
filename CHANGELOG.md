# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Adaptive local-exploration count (`discover_gibbs_steps`).** Auto-tunes
  `gibbs_steps_per_round` (n_expl) — the last major NRPT knob hamon did not
  set for you — by maximizing effective sample size per **measured steady-state
  wall-second** (compile excluded via warm-up). Because the objective measures
  real per-round cost `t_round = c₀ + n_expl·c_s` rather than assuming the
  paper's cost ∝ n_expl, it **self-calibrates to the device**: on a compute-bound
  CPU it returns n_expl=1, but on a dispatch-bound GPU (where the fixed per-round
  overhead c₀ dominates a single sweep) it flips to n_expl=2–4, measured to give
  1.7–2.3× ESS/sec on the models tried. Round-trip efficiency and the
  `efficiency_limiter` gate it (a schedule-limited probe stops the search). The
  chain count is held fixed (Λ is robust to n_expl), so it composes after
  `discover_chain_count`.
- **Efficiency-cause attribution in the health report.** When round-trip
  efficiency is below the ELE-optimal rate, `report_nrpt_diagnostics` now sets
  `NRPTHealthReport.efficiency_limiter` to point at the right knob:
  `"schedule"` when the ladder is not equalized (tune further / add chains) or
  `"local_exploration"` when it *is* equalized — an ELE violation whose fix is a
  larger `gibbs_steps_per_round` (with more chains as the alternative lever).
  Previously low efficiency was always attributed to chain count; the report now
  distinguishes a schedule problem from a local-kernel problem using the
  rejection-rate spread it already computes.
- **Effective sample size (ESS).** `hamon.effective_sample_size` estimates the
  per-variable ESS of a sample trace (FFT autocorrelation + Geyer
  initial-positive-sequence) and `report_nrpt_diagnostics` now reports
  `min_ess`/`median_ess`/`ess_fraction` (with a low-ESS warning) whenever
  `samples` are provided. ESS is the standard answer to "how much do I trust
  these samples?" — Monte-Carlo error scales as `σ/√ESS`, not `σ/√n` — and is
  the gold-standard efficiency metric (ESS/compute) of Syed et al. (2021),
  complementing the existing round-trip proxy. Pure host numpy, no XLA compile.
- **Log normalizing constant via thermodynamic integration.** A new opt-in
  `hamon.NRPTEnergyObserver` accumulates the per-chain mean energy μ(β), and
  `hamon.thermodynamic_integration` / `hamon.nrpt_log_normalizing_constant`
  turn it into `log Z(β_max)/Z(β_min) = -∫μ dβ` (Syed et al. 2021, Sec. 5.5).
  This recovers the model evidence / free energy — the quantity ordinary MCMC
  discards but parallel tempering reconstructs almost for free — enabling
  Bayes-factor model comparison, EBM/RBM test log-likelihood evaluation, and
  Ising free-energy analysis. Opt-in: attaching the observer is the only way to
  trigger it, so the default `nrpt`/`nrpt_adaptive`/`ising_sample` fast paths
  are unchanged.

## [0.6.0] — 2026-06-23

### Changed

- **`discover_chain_count` reuses a cached probe on convergence.** When the
  chain-count search converges via `|n_star - n| <= 1` but the recommended
  `n_star` was never probed, it now returns the last probed count `n` (always
  cached) instead of running a full extra probe at `n_star`. That probe
  recompiled the NRPT round loop at a brand-new chain count just to land on a
  count within the convergence tolerance of one already in hand — costing ~2-4 s
  on roughly half of all discovery runs (those that converge one step off the
  last probe). This also makes the probe count robust across equally-good
  colourings. The returned `n_chains` may be 1 lower than before (within the
  existing tolerance), so `ising_sample` results can differ for affected models;
  the impact is small — the cold-chain draw is robust to ±1 ladder rung and the
  22x22 grid bench is unchanged.
- **Block colouring now uses Recursive Largest First (RLF).** `ising_sample`
  previously coloured the variable graph with networkx DSATUR, and the
  `auto_color_blocks` helper used a first-fit greedy. Both now use RLF
  (`hamon.graph_utils.rlf_coloring`), which minimises the colour count more
  aggressively on dense graphs (a 484-node 6-/12-regular graph drops 5→4 / 7→6
  colours) and matches DSATUR on sparse/bipartite ones. The colour count is the
  number of sequential block-Gibbs groups in the NRPT round loop, which sets its
  XLA compile cost, so fewer colours directly cuts compile on dense models. RLF
  is O(|V|·|E|) (~tens of ms, one-time) and deterministic, and the change also
  drops the networkx dependency from the `ising_sample` path. **Because the
  block partition changes, samples from `ising_sample` differ byte-for-byte from
  previous releases** — still a correct draw from the same target distribution,
  but no longer bit-identical across this version boundary.

- **`SamplingSchedule` is frozen** — it is passed as a static `jit` argument (its
  hash keys the compiled draw), so it is now
  `@dataclasses.dataclass(frozen=True)`: the value-based `__hash__` is
  auto-generated and the schedule is immutable, so one already used as a
  compilation cache key cannot be mutated out from under the cache. No behavior
  change.

- **`BlockSamplingProgram` caches its weight-independent structure** — the block
  layout, per-node gather/slice index arrays, and scatter positions are fixed by
  the graph, not the interaction weight values, but were recomputed from scratch
  on every construction — including each `program.with_ebm(...)`. They are now
  cached (keyed on the spec plus the interaction groups' node structure, the same
  node-identity scheme `BlockSpec` uses; the cached value holds the spec so the
  key's nodes stay alive and `id()` cannot be reused) so `with_ebm` only re-binds
  the weight tensors — ~16 ms → ~2 ms per rebuild, which compounds across
  repeated `nrpt` / `nrpt_adaptive` calls (training, sweeps). Bit-identical.
- **`ising_sample` no longer rebuilds the sampling program per chain** — after
  chain-count discovery it built one `IsingSamplingProgram` per chain
  (`[program.with_ebm(e) for e in init_ebms]`) only to read
  `programs[0].free_blocks`, re-running the full block-structure construction
  ~`n_chains` times for a structure that never changes. It now reuses the
  template program, removing ~`n_chains` redundant rebuilds (~200–380 ms of host
  time at 22×22). Outputs are bit-identical.
- **Round-trip diagnostics fused into one compile** — `round_trip_summary` (the
  Λ / τ̄ / efficiency / local-barrier summary emitted by each production NRPT
  phase) ran as ~8 eager `jnp` reductions, each paying a first-shape XLA compile
  the first time it was seen at a new chain count. It is now `jax.jit`-compiled,
  folding those into a single kernel (~13 compiles → 1, ~300 ms → ~37 ms per
  chain count); `n_rounds` is traced so the compile is shared across round
  counts. Outputs are bit-identical. Together with the schedule-commitment fix
  below, a cold `ising_sample` drops ~9.7 s → ~8.0 s at 22×22.
- **One fewer schedule-optimizer compile per tuning run** — `optimize_schedule`
  is `jax.jit`-compiled, and a jit cache keys on input *commitment* as well as
  shape and dtype. The first tuning phase passed the caller's uncommitted
  `initial_betas` while later phases passed the committed output of the previous
  phase, so XLA built two executables for the same computation and used the
  uncommitted one exactly once. `nrpt_adaptive` now pins the working schedule to
  the resolved device up front, so every phase shares one compile (~330 ms per
  probe, plus a couple of other betas-derived ops that were splitting the same
  way). Outputs are bit-identical.
- **Faster NRPT cold start** — the per-phase schedule-tuning math in
  `nrpt_adaptive` (the `optimize_schedule` monotone-cubic interpolation, the
  swap-rate statistics, and the per-phase Λ / rejection-spread / ladder-movement
  diagnostics) now runs inside `jax.jit` instead of as dozens of eager
  op-by-op dispatches, so each tuning phase compiles a single fused kernel
  rather than recompiling tiny primitives one shape at a time. In addition, when
  no observer is attached the round loop now runs as a dynamic-trip-count
  `lax.fori_loop` (with `n_rounds` passed as a traced value), which makes the
  `_nrpt_rounds` compilation independent of the round count, so a tuning batch
  and the production run — and discovery probes at the same chain count — share
  one compiled executable. Together these cut a cold `nrpt_adaptive` by ~40%
  (5.2 s → 3.1 s at 484 nodes / 8 chains; ~120 → ~56 XLA compilations) and a
  cold `ising_sample` by ~30% (30 s → 21 s at 22×22). Steady-state (warm) time
  is unchanged and outputs are bit-identical. With an observer the round loop
  still uses `lax.scan` to collect the per-round output stack.
- **Less host↔device traffic in `ising_sample` / `nrpt_adaptive`** — the
  remaining non-compile overhead was dominated by blocking device→host
  transfers and eager op-by-op dispatch. `ising_sample` built its graph by
  indexing the edge array on the host (`int(e[0])` per endpoint); the array is
  now pulled to host once with `np.asarray` instead of forcing ~2·n_edges
  blocking transfers. The adaptive tuning loop now threads chain states in
  their stacked form across batches (avoiding an `n_chains × n_blocks` eager
  slice unstack/restack on every batch), skips the eager round-trip summary on
  tuning batches that never read it, and keeps per-chain β values on-device.
  Together these cut host syncs ~3950 → ~210 and eager dispatches ~12.6k →
  ~0.6k, shaving a further ~40% off a cold `ising_sample` (≈19.5 s → ≈11.5 s at
  22×22). Outputs remain bit-identical.
- **Sample-quality diagnostics now run on the host in numpy** — the
  `report_nrpt_diagnostics` health report and its helpers (`sample_convergence`,
  `marginal_entropy`, `energy_balance`) computed their small one-shot reductions
  in `jax.numpy` on the accelerator, where each first-seen array shape triggered
  a separate XLA kernel compile and every `float()` / `.tolist()` forced a
  blocking device→host transfer (~31 compiles + ~16 syncs, ≈1 s spent compiling
  ≈25 ms of arithmetic). These are tiny post-hoc summaries over the returned
  samples, so they now run in plain `numpy`: no per-shape compilation, and a
  single `np.asarray(samples)` transfer instead of one sync per reduction. The
  in-pipeline health report drops from ≈1.4 s to ≈14 ms. Samples are untouched
  and every health-verdict field is unchanged; only the reported
  `acceptance_mean` / `Lambda` scalars may differ by ≤6e-7 (float32 host vs
  accelerator rounding).
- **The sample-collection draw compiles once instead of on every call** — the
  warmup and sampling `lax.scan`s in `sample_with_observation` ran on the eager
  (un-jitted) path, so XLA recompiled both scans on every `sample_states` /
  `sample_with_observation` call (~0.9 s at 484 nodes), even though the on-device
  sampling itself takes single-digit milliseconds. The compute core is now a
  module-level `eqx.filter_jit` function (device placement stays in the
  un-jitted wrapper, where it is a no-op under `jit` / `vmap`), with the
  `SamplingSchedule` as a static argument so distinct warmup/sample/step counts
  specialize and identical ones reuse the cache. Repeated draws with a fixed
  program now compile once and reuse the executable (≈1.0 s → ≈7 ms; one XLA
  compilation instead of ~20 across a handful of calls). Outputs are
  bit-identical.

## [0.5.0] — 2026-06-21

### Breaking

- **Minimum supported Python is now 3.12** (was 3.11). Current Ubuntu releases
  ship 3.12 as their default interpreter, so 3.11 was dropped from the CI test
  and example matrices (now 3.12–3.14).

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

- **`discover_chain_count` reworked to follow Syed et al. (2021) N-tuning** —
  replaces the fixed "step halfway toward the max-Λ recommendation" loop. The
  global barrier Λ is a schedule invariant (`Σ rejection_rates ≈ Λ` at any chain
  count), so it is estimated at a single fixed N from a schedule-tuned
  `nrpt_adaptive` run and the optimal count taken directly as
  `N* = ceil(Λ̂·(1 + safety_margin) / r_target) + 1` (the round-trip-optimal
  `2Λ + 1` at `r* = 1/2`), iterating that fixed point until `N*` settles. Because
  the estimate uses the current-N rejection rates rather than a running maximum,
  the result is essentially independent of the starting `N` — discovery from
  `initial_n=None` (a small pilot) and from a reasonable guess converge to the
  same count (≤ 1 chain apart on Ising chains). `initial_n` defaults to `None`
  (no initial guess needed) and a new `safety_margin` (default 0.05) pads `N*`
  against residual bias / ELE-assumption violations.
- **Monotone-cubic schedule optimization** — `optimize_schedule` now places the
  equi-acceptance ladder using a Fritsch–Carlson (PCHIP) monotone-cubic inverse
  of the cumulative barrier instead of piecewise-linear interpolation (Syed et
  al. 2021, Algorithm 2), giving a smoother schedule while staying monotone (no
  overshoot).
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
- **Refreshed example notebooks** — re-ran and updated all example notebooks so
  their output reflects the algorithm improvements in this release (NRPT chain
  discovery, monotone-cubic schedule, convergence-driven tuning) and for
  consistency across the example set.

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
