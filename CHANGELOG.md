# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Changed

- **jaxtyping floor raised to 0.3.10** (was 0.2.23) — picks up the fix for
  `PyTree[A | B]` isinstance checks silently passing (0.3.9) and cloudpickle
  round-trips of variadic annotations like `Shaped[Array, "..."]` (0.3.10)
- **optax floor (testing extra) raised to 0.2.8** (was 0.2.4) — older optax
  imports the `jax_pmap_shmap_merge` config option removed in JAX 0.10.0 and
  fails at import
- **CI test and example matrices now cover Python 3.11–3.14** (was 3.10–3.13)
- **Block write-back uses contiguous slice updates instead of scatters** —
  free blocks always occupy contiguous ranges of the global state (a
  `BlockSpec` layout invariant), so the per-block write-back in the Gibbs
  scan and in `scatter_block_to_global` now lowers to
  `lax.dynamic_update_slice` with a precomputed static offset instead of a
  gather-index scatter, which XLA fuses far better (the isolated op is ~30×
  faster on CPU; scatters are disproportionately expensive on GPU).
  Non-contiguous node sets keep the scatter fallback. Results are
  bit-identical.

### Fixed

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
