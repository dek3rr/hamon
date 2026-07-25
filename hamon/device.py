"""Device routing: run each workload on the device where it wins.

With CUDA jax installed, JAX places everything on the GPU by default — including
the tiny, dispatch-bound programs where a CPU finishes several times faster.
hamon's entry points therefore accept a ``device`` argument:

- ``"auto"`` (default): if no accelerator is visible, do nothing (placement is
  untouched, exactly as with CPU-only jax). Otherwise compare a work score —
  ``n_chains × free nodes``, the width of the parallel front per Gibbs sweep —
  against a threshold: small workloads run on the CPU, large ones on the
  accelerator. Threshold via ``HAMON_DEVICE_THRESHOLD`` (see
  ``benchmarks/device_crossover.py`` to calibrate); routing can be forced with
  ``HAMON_DEVICE=cpu|gpu|none``.
- ``"cpu"`` / ``"gpu"`` / ``"tpu"``: that device, raising if it is not visible.
- a concrete ``jax.Device``: used as-is (how orchestrators pass their one-time
  decision down so the device never flips between tuning phases).
- ``None``: hamon never touches placement — the full opt-out.

Routing re-commits entry arrays to the chosen device and runs the computation
under ``jax.default_device``; outputs come back committed to that device.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any, TypeVar

import jax
import jax.core

if TYPE_CHECKING:
    # jax.Device is a runtime alias into the pybind11 extension; type
    # checkers cannot use it in type expressions, so signatures treat it
    # as Any.
    JaxDevice = Any
else:
    JaxDevice = jax.Device

DeviceLike = str | JaxDevice | None
_T = TypeVar("_T")

# Steady-state crossover measured on an RTX 5080 (jax 0.10.1): score <= 2048
# always ran faster on CPU, >= 4096 always faster on GPU (2-11x). Short
# one-shot flows are compile-dominated — force "cpu" or enable the persistent
# compile cache. Calibrate per machine with benchmarks/device_crossover.py.
DEFAULT_DEVICE_THRESHOLD = 4096

_THRESHOLD_ENV = "HAMON_DEVICE_THRESHOLD"
_DEVICE_ENV = "HAMON_DEVICE"


def _lower_persistence_thresholds() -> None:
    """Let sub-second compiles persist to the on-disk cache.

    JAX's ``min_compile_time_secs`` default of **1.0s** drops nearly all of the
    autotuning search's many sub-second probe compiles (tuning math,
    ``optimize_schedule``, ``round_trip_summary``, even most ``_nrpt_rounds``),
    so the cache would store almost nothing and repeat runs would recompile
    everything — the amortization this module promises never happens. Set both
    minimums to 0 so every executable persists; an explicit
    ``JAX_PERSISTENT_CACHE_MIN_*`` env var still wins.
    """
    if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    if "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES" not in os.environ:
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


def enable_persistent_compile_cache(path: str | None = None) -> str | None:
    """Turn on JAX's persistent compilation cache (idempotent).

    XLA compile dominates the cold cost of NRPT and especially the multi-probe
    autotuning search (each chain count and each n_expl recompiles the round
    loop). The persistent cache stores compiled executables on disk and reuses
    them across processes — measured ≈ −72% wall on repeat cold runs — which is
    what keeps autotuning affordable.

    Only enabled on an **accelerator** backend. On a CPU-only backend XLA's AOT
    loader logs a "machine-feature mismatch" error (warning of a theoretical
    SIGILL) for every reloaded executable, and CPU compiles are cheap, so caching
    there is net-negative; the test ``conftest`` makes the same accelerator-only
    choice. An explicit ``JAX_COMPILATION_CACHE_DIR`` overrides this on any
    backend (the user's deliberate opt-in/out wins).

    If ``JAX_COMPILATION_CACHE_DIR`` is already set in the environment its
    directory is respected (including opting out via an empty value). Otherwise,
    when an accelerator is present, the cache dir is set to ``path`` (or
    ``~/.cache/jax`` by default), matching the GPU default used in the test
    suite. Whenever caching is active the persistence thresholds are lowered (see
    :func:`_lower_persistence_thresholds`).

    Args:
        path: cache directory; defaults to ``~/.cache/jax``.

    Returns:
        The active cache directory, or ``None`` if caching is disabled (CPU-only
        backend, or an empty env var).
    """
    env = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if env is not None:
        # Explicit env var wins on any backend: empty disables, else opt in.
        if env:
            _lower_persistence_thresholds()
        return env or None

    if path is not None:
        # An explicit path is a deliberate opt-in too; honor it on any backend.
        _lower_persistence_thresholds()
        jax.config.update("jax_compilation_cache_dir", path)
        return path

    # Default (compile_cache=True): only worthwhile on an accelerator. On a
    # CPU-only backend the AOT loader logs a machine-feature-mismatch error per
    # reloaded executable and compiles are cheap, so caching is net-negative.
    if accelerator_device() is None:
        return None
    _lower_persistence_thresholds()
    target = os.path.join(os.path.expanduser("~"), ".cache", "jax")
    jax.config.update("jax_compilation_cache_dir", target)
    return target


def cpu_device() -> JaxDevice:
    """The first CPU device (always present)."""
    return jax.devices("cpu")[0]


def accelerator_device() -> JaxDevice | None:
    """The first visible GPU, else the first TPU, else None. Never raises."""
    for platform in ("gpu", "tpu"):
        try:
            return jax.devices(platform)[0]
        except RuntimeError:
            continue
    return None


def free_node_count(program) -> int:
    """Total free nodes in a sampling program — O(1) Python metadata."""
    return sum(len(block.nodes) for block in program.gibbs_spec.free_blocks)


def work_score(n_chains: int, n_nodes: int) -> int:
    """The routing heuristic's work estimate: width of the parallel front.

    Rounds are deliberately excluded — the round loop is a single jitted scan,
    so per-round dispatch does not scale with round count, and compile cost is
    amortized by the persistent compilation cache."""
    return int(n_chains) * int(n_nodes)


def device_threshold(threshold: float | None = None) -> float:
    """Resolve the auto-routing threshold: argument > env > default."""
    if threshold is not None:
        return float(threshold)
    env = os.environ.get(_THRESHOLD_ENV, "").strip()
    if env:
        return float(env)
    return float(DEFAULT_DEVICE_THRESHOLD)


def _contains_tracer(trees: Any) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(trees))


def resolve_device(
    device: DeviceLike = "auto",
    *,
    score: int | None = None,
    threshold: float | None = None,
) -> JaxDevice | None:
    """Resolve a device spec into a concrete ``jax.Device`` or ``None``.

    ``None`` means "leave placement alone" and is what ``"auto"`` resolves to
    when no accelerator is visible, so CPU-only installs behave identically
    with or without routing. With an accelerator present, ``"auto"`` routes to
    it when ``score`` meets the threshold and to the CPU otherwise (or when no
    score is supplied). Explicit ``"cpu"``/``"gpu"``/``"tpu"`` requests fail
    loudly if the device is absent.
    """
    if device is None:
        return None
    if isinstance(device, jax.Device):
        return device
    if not isinstance(device, str):
        raise TypeError(
            f"device must be a str, jax.Device, or None; got {type(device).__name__}"
        )

    spec = device.lower()
    if spec == "auto":
        forced = os.environ.get(_DEVICE_ENV, "").strip().lower()
        if forced and forced != "auto":
            if forced in ("none", "off"):
                return None
            return resolve_device(forced, score=score, threshold=threshold)
        accelerator = accelerator_device()
        if accelerator is None:
            return None
        if score is not None and score >= device_threshold(threshold):
            return accelerator
        return cpu_device()
    if spec == "cpu":
        return cpu_device()
    if spec in ("gpu", "cuda", "tpu"):
        platform = "gpu" if spec == "cuda" else spec
        try:
            return jax.devices(platform)[0]
        except RuntimeError as e:
            raise RuntimeError(
                f"device={device!r} was requested but no {platform.upper()} is "
                f"visible to JAX ({e}). If JAX_PLATFORMS is set, it may be "
                f"hiding the device."
            ) from None
    raise ValueError(
        f"Unrecognized device spec {device!r}; expected 'auto', 'cpu', 'gpu', 'tpu', a jax.Device, or None."
    )


def resolve_entry_device(
    device: DeviceLike,
    *,
    n_chains: int,
    n_nodes: int,
    arrays: Any = (),
) -> JaxDevice | None:
    """Entry-point resolution: heuristic score plus a tracer guard.

    When any entry array is a tracer the caller is already inside a
    jit/vmap/grad trace, where opening device contexts or transferring arrays
    is not meaningful — routing becomes a no-op."""
    if _contains_tracer(arrays):
        return None
    return resolve_device(device, score=work_score(n_chains, n_nodes))


def _on_device(x: jax.Array, device: JaxDevice) -> bool:
    try:
        return x.committed and x.devices() == {device}
    except (AttributeError, ValueError):
        return False


def tree_device_put(tree: _T, device: JaxDevice | None) -> _T:
    """Commit every ``jax.Array`` leaf of ``tree`` to ``device``.

    Non-array leaves (blocks, nodes, samplers, specs) pass through with object
    identity preserved, so equinox's static partition hashes identically and
    jit caches stay warm. If every array leaf is already committed to
    ``device``, the original object is returned unchanged — this keeps
    repeated calls (e.g. ``tune_schedule`` tuning phases) presenting the
    literally-same pytree to the jit cache."""
    if device is None:
        return tree
    array_leaves = [x for x in jax.tree.leaves(tree) if isinstance(x, jax.Array)]
    if all(_on_device(x, device) for x in array_leaves):
        return tree
    return jax.tree.map(
        lambda x: jax.device_put(x, device) if isinstance(x, jax.Array) else x,
        tree,
    )


def default_device_ctx(device: JaxDevice | None) -> contextlib.AbstractContextManager:
    """Pin ``jax.default_device`` to ``device``, or do nothing when it is ``None``.

    The ``None`` case is hamon's opt-out (see the module docstring), and it must
    stay a genuine no-op rather than a default placement, so every entry point
    that resolves a device pairs it with this. Returns the context manager
    itself rather than wrapping it in a generator, so entering it costs no extra
    frame on paths that run it per tuning phase."""
    return (
        jax.default_device(device) if device is not None else contextlib.nullcontext()
    )
