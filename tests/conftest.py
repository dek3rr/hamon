import os

import jax

jax.config.update("jax_numpy_dtype_promotion", "strict")

# Persist compiled XLA executables across test runs (env vars take precedence).
# GPU only: that is where compilation — not compute — dominates suite wall
# time (a warm cache is ~3x faster), while the CPU AOT loader logs spurious
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
