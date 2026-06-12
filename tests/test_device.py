"""Unit tests for hamon.device routing.

Everything here must pass on CPU-only machines (CI): GPU-dependent branches
are exercised by monkeypatching the accelerator lookup with a sentinel device.
GPU hardware execution lives in test_gpu_smoke.py.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hamon.device as device_mod
from hamon.block_sampling import SamplingSchedule, sample_states
from hamon.device import (
    DEFAULT_DEVICE_THRESHOLD,
    accelerator_device,
    device_threshold,
    resolve_device,
    resolve_entry_device,
    tree_device_put,
    work_score,
)
from hamon.models.ising import hinton_init
from hamon.nrpt import nrpt, nrpt_adaptive

from .utils import make_ising_grid

# hamon/__init__.py re-exports the nrpt *function* under the name hamon.nrpt,
# shadowing the module attribute — go through importlib for the real module.
nrpt_mod = importlib.import_module("hamon.nrpt")

CPU = jax.devices("cpu")[0]


def _gpu_present() -> bool:
    return accelerator_device() is not None


def _make_states(key, ebms, free_blocks, n_chains):
    keys = jax.random.split(key, n_chains)
    return [hinton_init(keys[c], ebms[0], free_blocks, ()) for c in range(n_chains)]


class TestResolveDevice:
    def test_none_passthrough(self):
        assert resolve_device(None) is None

    def test_concrete_device_passthrough(self):
        assert resolve_device(CPU) is CPU

    def test_cpu_string(self):
        assert resolve_device("cpu") == CPU

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unrecognized device spec"):
            resolve_device("quantum")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            resolve_device(42)

    def test_explicit_gpu_without_gpu_raises(self):
        if _gpu_present():
            pytest.skip("a real GPU is visible")
        with pytest.raises(RuntimeError, match="no GPU is visible"):
            resolve_device("gpu")

    def test_auto_without_accelerator_is_noop(self, monkeypatch):
        monkeypatch.setattr(device_mod, "accelerator_device", lambda: None)
        assert resolve_device("auto", score=10**9) is None


class TestAutoHeuristic:
    @pytest.fixture
    def sentinel(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(device_mod, "accelerator_device", lambda: sentinel)
        return sentinel

    def test_below_threshold_routes_cpu(self, sentinel):
        assert resolve_device("auto", score=DEFAULT_DEVICE_THRESHOLD - 1) == CPU

    def test_at_threshold_routes_accelerator(self, sentinel):
        assert resolve_device("auto", score=DEFAULT_DEVICE_THRESHOLD) is sentinel

    def test_no_score_routes_cpu(self, sentinel):
        assert resolve_device("auto") == CPU

    def test_env_threshold(self, sentinel, monkeypatch):
        monkeypatch.setenv("HAMON_DEVICE_THRESHOLD", "10")
        assert resolve_device("auto", score=10) is sentinel
        assert resolve_device("auto", score=9) == CPU

    def test_arg_threshold_beats_env(self, sentinel, monkeypatch):
        monkeypatch.setenv("HAMON_DEVICE_THRESHOLD", "10")
        assert resolve_device("auto", score=10, threshold=100) == CPU

    def test_hamon_device_env_forces_cpu(self, sentinel, monkeypatch):
        monkeypatch.setenv("HAMON_DEVICE", "cpu")
        assert resolve_device("auto", score=10**9) == CPU

    def test_hamon_device_env_forces_none(self, sentinel, monkeypatch):
        monkeypatch.setenv("HAMON_DEVICE", "none")
        assert resolve_device("auto", score=10**9) is None

    def test_threshold_resolution_order(self, monkeypatch):
        assert device_threshold() == DEFAULT_DEVICE_THRESHOLD
        monkeypatch.setenv("HAMON_DEVICE_THRESHOLD", "123")
        assert device_threshold() == 123.0
        assert device_threshold(7) == 7.0

    def test_work_score(self):
        assert work_score(10, 4096) == 40960


class TestTreeDevicePut:
    def test_none_is_identity(self):
        tree = ([jnp.ones(3)], {"a": jnp.zeros(2)})
        assert tree_device_put(tree, None) is tree

    def test_moves_and_commits(self):
        tree = ([jnp.ones(3)], {"a": jnp.zeros(2)})
        moved = tree_device_put(tree, CPU)
        for leaf in jax.tree.leaves(moved):
            assert leaf.committed and leaf.devices() == {CPU}

    def test_identity_fast_path(self):
        tree = ([jnp.ones(3)], {"a": jnp.zeros(2)})
        moved = tree_device_put(tree, CPU)
        assert tree_device_put(moved, CPU) is moved

    def test_non_array_leaves_keep_identity(self):
        _, _, fb, ebms, progs = make_ising_grid(2, [1.0])
        prog = progs[0]
        moved = tree_device_put(prog, CPU)
        # leaf-level static structure (the spec holds blocks and node objects
        # whose equality is identity-based) must be the same objects so
        # equinox's jit cache keys hash identically; container Modules like
        # samplers are reconstructed by tree.map but compare equal, which the
        # cache also accepts (proven by the trace-count tests in test_nrpt.py)
        assert moved.gibbs_spec is prog.gibbs_spec
        for leaf in jax.tree.leaves(moved):
            if isinstance(leaf, jax.Array):
                assert leaf.committed and leaf.devices() == {CPU}

    def test_numpy_leaves_untouched(self):
        arr = np.ones(3)
        tree = {"np": arr, "jnp": jnp.ones(3)}
        moved = tree_device_put(tree, CPU)
        assert moved["np"] is arr


class TestEntryPoints:
    def test_nrpt_cpu_end_to_end(self):
        betas = jnp.array([0.5, 1.0])
        _, _, fb, ebms, progs = make_ising_grid(3, [1.0], coupling=0.5)
        inits = _make_states(jax.random.key(0), ebms, fb, 2)
        states, stats = nrpt(
            jax.random.key(1),
            ebms[0],
            progs[0],
            inits,
            [],
            4,
            1,
            betas=betas,
            device="cpu",
        )
        for chain in states:
            for block in chain:
                assert block.committed and block.devices() == {CPU}
        assert stats["betas"].devices() == {CPU}

    def test_nrpt_device_none_leaves_placement_alone(self):
        betas = jnp.array([0.5, 1.0])
        _, _, fb, ebms, progs = make_ising_grid(3, [1.0], coupling=0.5)
        inits = _make_states(jax.random.key(0), ebms, fb, 2)
        states, _ = nrpt(
            jax.random.key(1),
            ebms[0],
            progs[0],
            inits,
            [],
            4,
            1,
            betas=betas,
            device=None,
        )
        for chain in states:
            for block in chain:
                assert not block.committed

    def test_adaptive_resolves_auto_exactly_once(self, monkeypatch):
        auto_calls = []
        real = device_mod.resolve_entry_device

        def spy(device, **kwargs):
            if isinstance(device, str) and device == "auto":
                auto_calls.append(device)
            return real(device, **kwargs)

        # nrpt.py references the imported name, so patch it there
        monkeypatch.setattr(nrpt_mod, "resolve_entry_device", spy)

        _, _, fb, ebms, progs = make_ising_grid(3, [1.0], coupling=0.5)
        inits = _make_states(jax.random.key(0), ebms, fb, 3)
        nrpt_adaptive(
            jax.random.key(1),
            ebm=ebms[0],
            program=progs[0],
            init_states=inits,
            clamp_state=[],
            n_rounds=4,
            gibbs_steps_per_round=1,
            initial_betas=jnp.array([0.5, 1.0, 1.5]),
            n_tune=3,
            rounds_per_tune=4,
        )
        # trace counts alone cannot detect a device flip (the jaxpr cache can
        # hit while the executable recompiles), so assert resolve-once directly
        assert len(auto_calls) == 1

    def test_adaptive_cpu_outputs_on_cpu(self):
        _, _, fb, ebms, progs = make_ising_grid(3, [1.0], coupling=0.5)
        inits = _make_states(jax.random.key(0), ebms, fb, 2)
        states, stats = nrpt_adaptive(
            jax.random.key(1),
            ebm=ebms[0],
            program=progs[0],
            init_states=inits,
            clamp_state=[],
            n_rounds=4,
            gibbs_steps_per_round=1,
            initial_betas=jnp.array([0.5, 1.0]),
            n_tune=2,
            rounds_per_tune=4,
            device="cpu",
        )
        for chain in states:
            for block in chain:
                assert block.committed and block.devices() == {CPU}

    def test_sample_states_under_vmap_is_noop(self):
        # tracer guard: "auto" routing inside a vmap trace must not try to
        # open device contexts or transfer tracers
        _, _, fb, ebms, progs = make_ising_grid(2, [1.0])
        prog = progs[0]
        init = [jnp.zeros(len(b), dtype=jnp.bool_) for b in fb]
        schedule = SamplingSchedule(1, 2, 1)

        def run(key):
            return sample_states(key, prog, schedule, init, [], [fb[0]])

        out = jax.vmap(run)(jax.random.split(jax.random.key(0), 2))
        assert out[0].shape[:2] == (2, 2)


class TestResolveEntryDevice:
    def test_tracer_guard(self):
        def f(x):
            return resolve_entry_device("cpu", n_chains=1, n_nodes=1, arrays=(x,))

        # under trace the guard must return None even for explicit specs
        results = []

        def g(x):
            results.append(f(x))
            return x

        jax.vmap(g)(jnp.ones(2))
        assert results == [None]

    def test_concrete_arrays_resolve(self):
        dev = resolve_entry_device("cpu", n_chains=1, n_nodes=1, arrays=(jnp.ones(2),))
        assert dev == CPU
