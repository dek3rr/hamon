"""GPU execution smoke tests (auto-skipped when no GPU is visible).

The rest of the suite runs on the CPU (see conftest); this subset proves the
GPU code paths stay healthy: routing places arrays there, the jitted round
loop compiles and reuses its cache, and dtype preservation holds on device.
"""

import jax
import jax.numpy as jnp
import pytest

from hamon.device import accelerator_device
from hamon.models.ising import hinton_init
from hamon.nrpt import nrpt, nrpt_adaptive

from .utils import make_ising_grid

pytestmark = pytest.mark.gpu


def _gpu():
    return jax.devices("gpu")[0]


def _make_states(key, ebms, free_blocks, n_chains):
    keys = jax.random.split(key, n_chains)
    return [hinton_init(keys[c], ebms[0], free_blocks, ()) for c in range(n_chains)]


class TestGPUSmoke:
    def test_nrpt_pipeline_on_gpu(self):
        from hamon.nrpt import _nrpt_rounds_trace_count

        _, _, fb, ebms, progs = make_ising_grid(6, [1.0], coupling=0.5)
        inits = _make_states(jax.random.key(0), ebms, fb, 3)

        before = _nrpt_rounds_trace_count[0]
        states, stats = nrpt_adaptive(
            jax.random.key(1),
            ebm=ebms[0],
            program=progs[0],
            init_states=inits,
            clamp_state=[],
            n_rounds=10,
            gibbs_steps_per_round=1,
            initial_betas=jnp.array([0.5, 1.0, 1.5]),
            n_tune=2,
            rounds_per_tune=10,
            device="gpu",
        )
        traces = _nrpt_rounds_trace_count[0] - before
        assert traces == 1, f"expected 1 trace across 3 phases, got {traces}"

        gpu = _gpu()
        for chain in states:
            for block in chain:
                assert block.committed and block.devices() == {gpu}
        acc = stats["acceptance_rate"]
        assert bool(jnp.all((acc >= 0.0) & (acc <= 1.0)))

    def test_dtype_preservation_on_gpu(self):
        from .test_dtype_preservation import _ring_model, enable_x64

        with enable_x64():
            ebm, program, blocks = _ring_model()
            betas = jnp.linspace(0.1, 1.0, 3)  # float64 under x64
            keys = jax.random.split(jax.random.key(0), 3)
            inits = [hinton_init(keys[c], ebm, blocks, ()) for c in range(3)]
            _, stats = nrpt(
                jax.random.key(1),
                ebm,
                program,
                inits,
                [],
                4,
                1,
                betas=betas,
                device="gpu",
            )
            assert stats["betas"].dtype == jnp.float32
            assert stats["betas"].devices() == {_gpu()}

    def test_auto_routes_large_score_to_gpu(self, monkeypatch):
        # threshold of 1 makes even this small model "large" — proves the
        # auto → accelerator branch without building a big model
        monkeypatch.setenv("HAMON_DEVICE_THRESHOLD", "1")
        assert accelerator_device() is not None

        _, _, fb, ebms, progs = make_ising_grid(4, [1.0], coupling=0.5)
        inits = _make_states(jax.random.key(0), ebms, fb, 2)
        states, _ = nrpt(
            jax.random.key(1),
            ebms[0],
            progs[0],
            inits,
            [],
            4,
            1,
            betas=jnp.array([0.5, 1.0]),
            device="auto",
        )
        gpu = _gpu()
        for chain in states:
            for block in chain:
                assert block.committed and block.devices() == {gpu}
