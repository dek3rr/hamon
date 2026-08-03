"""Float32 models must stay float32 on device even when the host application
enables x64 (``jax_enable_x64``), as is common in code that mixes hamon with
double-precision analytics. On consumer GPUs float64 also runs at a fraction
of the float32 rate, so a silent promotion is a performance bug as well as a
numerics change.

Strict dtype promotion (set in conftest) escalates any float32/float64 mix
into an error, so simply running NRPT under x64 here is a leak detector for
the whole sampling path, on top of the explicit dtype assertions.
"""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
from hamon import Block
from hamon.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
from hamon.nrpt import nrpt, optimize_schedule
from hamon.pgm import SpinNode


@contextlib.contextmanager
def enable_x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _ring_model(n=8, dtype=jnp.float32):
    rng = np.random.default_rng(0)
    nodes = [SpinNode() for _ in range(n)]
    edges = [(nodes[i], nodes[(i + 1) % n]) for i in range(n)]
    biases = jnp.asarray(rng.normal(size=n), dtype=dtype)
    weights = jnp.asarray(rng.normal(size=n), dtype=dtype)
    blocks = [
        Block([nodes[i] for i in range(0, n, 2)]),
        Block([nodes[i] for i in range(1, n, 2)]),
    ]
    # jnp.array(1.0) is deliberately strong-typed: float64 when x64 is on.
    ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
    program = IsingSamplingProgram(ebm, blocks, [])
    return ebm, program, blocks


class TestX64DoesNotPromoteFloat32Models:
    def test_beta_follows_param_dtype(self):
        with enable_x64():
            ebm, _, _ = _ring_model()
            assert ebm.beta.dtype == jnp.float32
            for factor in ebm.factors:
                assert factor.weights.dtype == jnp.float32

    def test_energy_stays_float32(self):
        with enable_x64():
            ebm, _, blocks = _ring_model()
            state = [jnp.zeros(len(b), dtype=jnp.bool_) for b in blocks]
            assert ebm.energy(state, blocks).dtype == jnp.float32

    def test_nrpt_template_mode_stays_float32(self):
        with enable_x64():
            ebm, program, blocks = _ring_model()
            n_chains = 4
            betas = jnp.linspace(0.1, 1.0, n_chains)  # float64 under x64
            keys = jax.random.split(jax.random.key(0), n_chains)
            inits = [hinton_init(keys[c], ebm, blocks, ()) for c in range(n_chains)]
            _, stats = nrpt(
                jax.random.key(1), ebm, program, inits, [], 6, 1, betas=betas
            )
            assert stats["betas"].dtype == jnp.float32
            assert stats["acceptance_rate"].dtype == jnp.float32
            assert stats["round_trip_diagnostics"]["Lambda"].dtype == jnp.float32

    def test_nrpt_per_chain_mode_stays_float32(self):
        with enable_x64():
            ebm, program, blocks = _ring_model()
            beta_values = [0.2, 0.6, 1.0]
            ebms = [ebm.with_beta(jnp.array(b)) for b in beta_values]
            programs = [program.with_ebm(e) for e in ebms]
            keys = jax.random.split(jax.random.key(0), len(ebms))
            inits = [
                hinton_init(keys[c], ebms[c], blocks, ()) for c in range(len(ebms))
            ]
            _, stats = nrpt(jax.random.key(1), ebms, programs, inits, [], 4, 1)
            assert stats["betas"].dtype == jnp.float32
            assert stats["acceptance_rate"].dtype == jnp.float32

    def test_optimize_schedule_preserves_beta_dtype(self):
        with enable_x64():
            betas = jnp.linspace(0.0, 1.0, 5).astype(jnp.float32)
            rejection = jnp.linspace(0.1, 0.4, 4)  # float64 under x64
            assert optimize_schedule(rejection, betas).dtype == jnp.float32

    def test_float64_model_opt_in_is_preserved(self):
        with enable_x64():
            ebm, _, blocks = _ring_model(dtype=jnp.float64)
            assert ebm.beta.dtype == jnp.float64
            state = [jnp.zeros(len(b), dtype=jnp.bool_) for b in blocks]
            assert ebm.energy(state, blocks).dtype == jnp.float64
