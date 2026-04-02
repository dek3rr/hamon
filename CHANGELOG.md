# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
