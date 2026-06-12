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

import os
from typing import TYPE_CHECKING, Any, TypeVar, Union

import jax
import jax.core

if TYPE_CHECKING:
    # jax.Device is a runtime alias into the pybind11 extension; type
    # checkers cannot use it in type expressions, so signatures treat it
    # as Any.
    JaxDevice = Any
else:
    JaxDevice = jax.Device

DeviceLike = Union[str, JaxDevice, None]
_T = TypeVar("_T")

# Steady-state crossover measured on an RTX 5080 (benchmarks/device_crossover.py,
# jax 0.10.1): every sweep point at score <= 2048 ran faster on CPU and every
# point at score >= 4096 ran faster on GPU (2-11x). Holds for production-length
# runs (hundreds of rounds or repeated calls); very short one-shot flows are
# compile-dominated (GPU compiles cost ~2x CPU's) — force those to "cpu", or
# enable the persistent compilation cache (JAX_COMPILATION_CACHE_DIR) to
# amortize. Calibrate per machine with benchmarks/device_crossover.py.
DEFAULT_DEVICE_THRESHOLD = 4096

_THRESHOLD_ENV = "HAMON_DEVICE_THRESHOLD"
_DEVICE_ENV = "HAMON_DEVICE"


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


def device_threshold(threshold: "float | None" = None) -> float:
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
    score: "int | None" = None,
    threshold: "float | None" = None,
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
        raise TypeError(f"device must be a str, jax.Device, or None; got {type(device).__name__}")

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
    repeated calls (e.g. ``nrpt_adaptive`` tuning phases) presenting the
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
