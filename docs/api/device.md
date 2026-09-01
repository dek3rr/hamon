# Device Routing

With CUDA JAX installed, JAX places everything on the GPU — including the small,
dispatch-bound programs where a CPU finishes several times faster. hamon's entry
points therefore take a `device` argument:

- `"auto"` (default) — with no accelerator visible, placement is untouched.
  Otherwise the [work score][hamon.work_score] (`n_chains × free nodes`) decides:
  small workloads run on the CPU, large ones on the accelerator.
- `"cpu"` / `"gpu"` — that platform, raising if it is not visible.
- a concrete `jax.Device` — used as-is.
- `None` — hamon never touches placement.

The default threshold (4096, the steady-state crossover measured on an RTX 5080)
can be overridden with `HAMON_DEVICE_THRESHOLD`; calibrate your own with
`python benchmarks/device_crossover.py`. `HAMON_DEVICE=cpu|gpu|none` forces a
choice without code changes.

Very short one-shot flows are compile-dominated and can favor the CPU regardless
of size — pass `device="cpu"` for those, or enable the persistent compile cache
so repeated runs skip GPU compilation entirely.

::: hamon.resolve_device

::: hamon.work_score

::: hamon.enable_persistent_compile_cache

::: hamon.DeviceLike
