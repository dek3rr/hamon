import os

import jax
import pytest

jax.config.update("jax_numpy_dtype_promotion", "strict")

# Pin the default device to CPU for the suite: test models are tiny, so GPU
# runs are dominated by compilation and kernel-launch overhead (~4x slower
# end to end). jax_default_device only changes placement — the GPU stays
# enumerable, so @pytest.mark.gpu tests can still request it explicitly.
# Set HAMON_TEST_DEVICE=gpu to run the whole suite on the GPU instead.
_test_device = os.environ.get("HAMON_TEST_DEVICE", "cpu").strip().lower()
if _test_device == "cpu":
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

# Persist compiled XLA executables across test runs (env vars take precedence).
# With the CPU pin above this mainly serves the gpu-marked smoke subset and
# HAMON_TEST_DEVICE=gpu runs; GPU-only because the CPU AOT loader logs spurious
# machine-feature-mismatch errors when reloading cached executables. The
# threshold drops to 0 because most compiles here are sub-second, below the
# 1s default cutoff.
if jax.default_backend() == "gpu":
    if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
        jax.config.update(
            "jax_compilation_cache_dir",
            os.path.join(os.path.expanduser("~"), ".cache", "jax"),
        )
    if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


def _gpu_available() -> bool:
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


def pytest_collection_modifyitems(config, items):
    if _gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="no GPU visible to JAX")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
