"""Tests for tune_chains in nrpt.py.

Guards against:
- Node identity mismatch (SpinNode KeyError)
- Return structure correctness
- Max-Λ tracking (conservative estimate)
- Stabilization detection
- Convergence reason reporting
- Chain count bounds
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hamon import Block, SpinNode
from hamon.models import IsingEBM, IsingSamplingProgram, hinton_init
from hamon.tuning import tune_chains


# ---------------------------------------------------------------------------
# Shared graph — built ONCE, all factories close over these objects
# ---------------------------------------------------------------------------

_NODES = [SpinNode() for _ in range(16)]
_EDGES = [(n, _NODES[i + 1]) for i, n in enumerate(_NODES[:-1])]
_BIASES = jnp.zeros(16)
_WEIGHTS = jnp.ones(15) * 0.8
_FREE_BLOCKS = [Block(_NODES[::2]), Block(_NODES[1::2])]


def _ebm_factory(betas):
    return [
        IsingEBM(_NODES, _EDGES, _BIASES, _WEIGHTS, jnp.array(float(b))) for b in betas
    ]


def _program_factory(ebms):
    return [IsingSamplingProgram(e, _FREE_BLOCKS, []) for e in ebms]


def _init_factory(n_chains, ebms, programs):
    """Correct factory: extracts free_blocks from programs."""
    fb = programs[0].gibbs_spec.free_blocks
    ks = jax.random.split(jax.random.key(0), n_chains)
    return [hinton_init(ks[i], ebms[0], fb, ()) for i in range(n_chains)]


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


class TestDiscoverChainCount:
    def test_seed_from_energy_matches_pilot(self):
        # The energy-variance seed (Theorem 2, no PT ladder) converges in fewer
        # probes and, when the estimate hits N*, yields the SAME discovered N and
        # schedule as the max_chains pilot — the probe RNG is key-aligned, so the
        # result is bit-identical (only the reported Λ, a max over probes, differs).
        ebm = IsingEBM(_NODES, _EDGES, _BIASES, _WEIGHTS, jnp.array(1.0))
        program = IsingSamplingProgram(ebm, _FREE_BLOCKS, [])
        kw = dict(
            ebm=ebm,
            program=program,
            init_factory=_init_factory,
            clamp_state=[],
            beta_range=(0.0, 1.0),
            gibbs_steps_per_round=1,
            max_chains=32,
            rounds_per_probe=60,
            n_tune_per_probe=2,
        )
        pilot = tune_chains(jax.random.key(7), seed_from_energy=False, **kw)
        energy = tune_chains(jax.random.key(7), seed_from_energy=True, **kw)
        assert energy["n_chains"] == pilot["n_chains"]
        assert jnp.allclose(jnp.asarray(energy["betas"]), jnp.asarray(pilot["betas"]))
        assert len(energy["history"]) <= len(pilot["history"])

    @pytest.mark.parametrize("seed_from_energy", [False, True])
    def test_stacked_init_factory_matches_per_chain(self, seed_from_energy):
        # A factory returning the stacked (n_chains, ...) form with the same
        # values as the per-chain form must give bit-identical discovery, on
        # both the pilot route and the energy-seed route (which restacks
        # per-chain inits itself).
        def _stacked_factory(n_chains, ebms, programs):
            per_chain = _init_factory(n_chains, ebms, programs)
            nb = len(per_chain[0])
            return [
                jnp.stack([per_chain[c][b] for c in range(n_chains)]) for b in range(nb)
            ]

        ebm = IsingEBM(_NODES, _EDGES, _BIASES, _WEIGHTS, jnp.array(1.0))
        program = IsingSamplingProgram(ebm, _FREE_BLOCKS, [])
        kw = dict(
            ebm=ebm,
            program=program,
            clamp_state=[],
            beta_range=(0.0, 1.0),
            gibbs_steps_per_round=1,
            max_chains=16,
            rounds_per_probe=60,
            n_tune_per_probe=2,
            seed_from_energy=seed_from_energy,
        )
        a = tune_chains(jax.random.key(7), init_factory=_init_factory, **kw)
        b = tune_chains(jax.random.key(7), init_factory=_stacked_factory, **kw)
        assert b["n_chains"] == a["n_chains"]
        assert np.array_equal(np.asarray(b["betas"]), np.asarray(a["betas"]))
        assert b["Lambda"] == a["Lambda"]

    def test_n_star_driven_by_running_max_lambda(self):
        # On a frustrated grid the per-probe Λ̂ is biased low at low N (a coarse
        # ladder can't resolve the barrier), so N* must be driven by the running
        # MAX Λ̂, not the current probe — else the search collapses to a too-low N.
        # Assert convergence to a non-degenerate N consistent with the reported
        # (max) Λ: N = ceil((1+safety_margin)·Λ / (1-target_acceptance)) + 1.
        L = 6
        n = L * L

        def idx(r, c):
            return (r % L) * L + (c % L)

        edges = []
        for r in range(L):
            for c in range(L):
                edges.append((idx(r, c), idx(r, c + 1)))
                edges.append((idx(r, c), idx(r + 1, c)))
        rng = np.random.default_rng(0)
        nodes = [SpinNode() for _ in range(n)]
        node_edges = [(nodes[a], nodes[b]) for a, b in edges]
        weights = jnp.asarray(rng.choice([-2.0, 2.0], size=len(edges)))
        even = [
            nodes[r * L + c] for r in range(L) for c in range(L) if (r + c) % 2 == 0
        ]
        odd = [nodes[r * L + c] for r in range(L) for c in range(L) if (r + c) % 2 == 1]
        ebm = IsingEBM(nodes, node_edges, jnp.zeros(n), weights, jnp.array(1.0))
        program = IsingSamplingProgram(ebm, [Block(even), Block(odd)], [])

        def init_factory(nc, ebms, programs):
            fb = programs[0].gibbs_spec.free_blocks
            ks = jax.random.split(jax.random.key(0), nc)
            return [hinton_init(ks[i], ebms[0], fb, ()) for i in range(nc)]

        result = tune_chains(
            jax.random.key(1),
            ebm=ebm,
            program=program,
            init_factory=init_factory,
            clamp_state=[],
            beta_range=(0.0, 1.0),
            gibbs_steps_per_round=1,
            target_acceptance=0.5,
            safety_margin=0.05,
            max_chains=64,
            rounds_per_probe=40,
            n_tune_per_probe=2,
            # Pin the pilot path: this test exercises the running-max Λ̂
            # mechanism itself, not the energy seed's R̂ fallback into it.
            seed_from_energy=False,
        )
        N, Lam = int(result["n_chains"]), float(result["Lambda"])
        assert 3 <= N <= 64  # non-degenerate, within bounds
        expected = math.ceil((1.05 * Lam) / 0.5) + 1  # = ceil(2.1·Λ)+1
        assert abs(N - expected) <= 1  # N driven by the (max) Λ, ±cached-final

    def test_runs_without_error(self):
        result = tune_chains(
            jax.random.key(42),
            _ebm_factory,
            _program_factory,
            _init_factory,
            [],
            beta_range=(0.2, 2.0),
            gibbs_steps_per_round=2,
            initial_n=4,
            target_acceptance=0.5,
            rounds_per_probe=30,
            n_tune_per_probe=2,
            max_iters=3,
        )

        assert "n_chains" in result
        assert "betas" in result
        assert "Lambda" in result
        assert "Lambda_raw" in result
        assert "converged_reason" in result
        assert "history" in result

    def test_output_types(self):
        result = tune_chains(
            jax.random.key(1),
            _ebm_factory,
            _program_factory,
            _init_factory,
            [],
            beta_range=(0.5, 1.5),
            gibbs_steps_per_round=2,
            initial_n=4,
            rounds_per_probe=20,
            n_tune_per_probe=2,
            max_iters=2,
        )

        assert isinstance(result["n_chains"], int)
        assert isinstance(result["Lambda"], float)
        assert isinstance(result["Lambda_raw"], float)
        assert result["Lambda"] >= 0.0
        assert len(result["history"]) >= 1

    def test_chain_count_within_bounds(self):
        result = tune_chains(
            jax.random.key(2),
            _ebm_factory,
            _program_factory,
            _init_factory,
            [],
            beta_range=(0.5, 1.5),
            gibbs_steps_per_round=2,
            initial_n=4,
            min_chains=3,
            max_chains=20,
            rounds_per_probe=20,
            n_tune_per_probe=2,
            max_iters=3,
        )

        assert result["n_chains"] >= 3
        assert result["n_chains"] <= 20

    def test_history_records_iterations(self):
        result = tune_chains(
            jax.random.key(3),
            _ebm_factory,
            _program_factory,
            _init_factory,
            [],
            beta_range=(0.2, 2.0),
            gibbs_steps_per_round=2,
            initial_n=3,
            rounds_per_probe=20,
            n_tune_per_probe=2,
            max_iters=4,
        )

        for h in result["history"]:
            assert "iteration" in h
            assert "n" in h
            assert "Lambda_raw" in h
            assert "Lambda_max" in h
            assert "n_recommended" in h
            assert h["n"] >= 3

    def test_target_acceptance_stored(self):
        result = tune_chains(
            jax.random.key(4),
            _ebm_factory,
            _program_factory,
            _init_factory,
            [],
            beta_range=(0.5, 1.5),
            gibbs_steps_per_round=2,
            initial_n=4,
            target_acceptance=0.7,
            rounds_per_probe=20,
            n_tune_per_probe=2,
            max_iters=2,
        )

        assert result["target_acceptance"] == 0.7

    def test_betas_length_matches_n_chains(self):
        result = tune_chains(
            jax.random.key(5),
            _ebm_factory,
            _program_factory,
            _init_factory,
            [],
            beta_range=(0.2, 2.0),
            gibbs_steps_per_round=2,
            initial_n=4,
            rounds_per_probe=20,
            n_tune_per_probe=2,
            max_iters=2,
        )

        assert len(result["betas"]) == result["n_chains"]

    def test_converged_reason_is_valid(self):
        result = tune_chains(
            jax.random.key(6),
            _ebm_factory,
            _program_factory,
            _init_factory,
            [],
            beta_range=(0.5, 1.5),
            gibbs_steps_per_round=2,
            initial_n=4,
            rounds_per_probe=20,
            n_tune_per_probe=2,
            max_iters=3,
        )

        assert result["converged_reason"] in {
            "chain_count",
            "lambda_stable",
            "no_progress",
            "max_iters",
        }

    def test_zero_iters_returns_the_same_keys(self):
        """The no-probe early return must not drop keys the normal path supplies.

        max_iters<=0 short-circuits before any probe runs and builds its own
        result dict. Callers read the result by key regardless of which branch
        produced it, so the two key sets have to agree.
        """
        common = dict(
            beta_range=(0.5, 1.5),
            gibbs_steps_per_round=2,
            rounds_per_probe=20,
            n_tune_per_probe=2,
            initial_n=4,
        )
        early = tune_chains(
            jax.random.key(0),
            _ebm_factory,
            _program_factory,
            _init_factory,
            [],
            max_iters=0,
            **common,
        )
        normal = tune_chains(
            jax.random.key(0),
            _ebm_factory,
            _program_factory,
            _init_factory,
            [],
            max_iters=1,
            **common,
        )
        assert set(early) == set(normal)
        # Nothing measured the barrier, so it is unknown rather than False.
        assert early["barrier_identified"] is None


# ---------------------------------------------------------------------------
# Max-Λ tracking
# ---------------------------------------------------------------------------


class TestMaxLambdaTracking:
    def test_lambda_geq_lambda_raw(self):
        """Conservative Λ (max) should be >= the last raw estimate."""
        result = tune_chains(
            jax.random.key(10),
            _ebm_factory,
            _program_factory,
            _init_factory,
            [],
            beta_range=(0.2, 2.0),
            gibbs_steps_per_round=2,
            initial_n=3,
            rounds_per_probe=30,
            n_tune_per_probe=2,
            max_iters=4,
        )

        assert result["Lambda"] >= result["Lambda_raw"] - 1e-6

    def test_lambda_max_monotonic_in_history(self):
        """Lambda_max in history should be non-decreasing."""
        result = tune_chains(
            jax.random.key(11),
            _ebm_factory,
            _program_factory,
            _init_factory,
            [],
            beta_range=(0.2, 2.0),
            gibbs_steps_per_round=2,
            initial_n=3,
            rounds_per_probe=30,
            n_tune_per_probe=2,
            max_iters=5,
        )

        maxes = [h["Lambda_max"] for h in result["history"]]
        for i in range(1, len(maxes)):
            assert maxes[i] >= maxes[i - 1] - 1e-6


# ---------------------------------------------------------------------------
# Node identity
# ---------------------------------------------------------------------------


class TestNodeIdentity:
    def test_correct_factory_works(self):
        """init_factory using programs[0].gibbs_spec.free_blocks should work."""
        betas = jnp.linspace(0.5, 1.5, 4)
        ebms = _ebm_factory(betas)
        programs = _program_factory(ebms)
        inits = _init_factory(4, ebms, programs)
        assert len(inits) == 4
        assert len(inits[0]) == 2

    def test_stale_blocks_raise(self):
        """Using free_blocks from a DIFFERENT set of nodes should fail."""
        other_nodes = [SpinNode() for _ in range(16)]
        other_blocks = [Block(other_nodes[::2]), Block(other_nodes[1::2])]
        ebm = IsingEBM(_NODES, _EDGES, _BIASES, _WEIGHTS, jnp.array(1.0))

        with pytest.raises(KeyError):
            hinton_init(jax.random.key(0), ebm, other_blocks, ())

    def test_factory_isolation(self):
        """Repeated ebm_factory calls should share node objects."""
        ebms_a = _ebm_factory(jnp.array([0.5, 1.0]))
        ebms_b = _ebm_factory(jnp.array([0.3, 0.7, 1.2]))
        assert ebms_a[0].nodes[0] is ebms_b[0].nodes[0]


# ---------------------------------------------------------------------------
# Chain masking (pad_probes / nrpt pad_chains_to)
# ---------------------------------------------------------------------------


class TestChainMasking:
    """Masking is a pure compile-sharing optimization: the live prefix of a
    padded run is bit-identical to an unpadded run (threefry key/uniform
    streams are prefix-stable, masked pairs keep the identity permutation)."""

    def _template(self):
        ebm = IsingEBM(_NODES, _EDGES, _BIASES, _WEIGHTS, jnp.array(1.0))
        return ebm, IsingSamplingProgram(ebm, _FREE_BLOCKS, [])

    def test_nrpt_padded_bit_identical(self):
        from hamon.nrpt import nrpt

        ebm, program = self._template()
        nc = 5
        betas = jnp.linspace(0.0, 1.0, nc)
        inits = _init_factory(nc, [ebm] * nc, [program] * nc)
        kw = dict(betas=betas, track_round_trips=True, device=None)
        key = jax.random.key(3)
        s_a, st_a = nrpt(key, ebm, program, inits, [], 40, 2, **kw)
        for pad in (nc, 12):  # pad == n (structural no-op) and pad > n
            s_b, st_b = nrpt(
                key, ebm, program, inits, [], 40, 2, pad_chains_to=pad, **kw
            )
            for c in range(nc):
                for b in range(len(s_a[c])):
                    assert np.array_equal(
                        np.asarray(s_a[c][b]), np.asarray(s_b[c][b])
                    ), (pad, c, b)
            for k2 in ("accepted", "attempted"):
                assert np.array_equal(np.asarray(st_a[k2]), np.asarray(st_b[k2]))
            rt_a = st_a["round_trip_diagnostics"]
            rt_b = st_b["round_trip_diagnostics"]
            for k2 in ("round_trips_per_chain", "restarts_per_chain"):
                assert np.array_equal(np.asarray(rt_a[k2]), np.asarray(rt_b[k2]))

    def test_pad_probes_bit_identical_discovery(self):
        ebm, program = self._template()
        kw = dict(
            ebm=ebm,
            program=program,
            init_factory=_init_factory,
            clamp_state=[],
            beta_range=(0.0, 1.0),
            gibbs_steps_per_round=1,
            max_chains=16,
            rounds_per_probe=60,
            n_tune_per_probe=2,
        )
        plain = tune_chains(jax.random.key(7), **kw)
        masked = tune_chains(jax.random.key(7), pad_probes=True, **kw)
        assert masked["n_chains"] == plain["n_chains"]
        assert np.array_equal(np.asarray(masked["betas"]), np.asarray(plain["betas"]))
        assert masked["Lambda"] == plain["Lambda"]
        assert [h["n"] for h in masked["history"]] == [h["n"] for h in plain["history"]]

    def test_cold_index_observer_padded_bit_identical(self):
        """A masked draw's cold trace matches the unpadded draw's exactly.

        The padded run records the live cold chain at absolute index n-1; the
        unpadded run records it at -1. Same physical chain, so the collected
        per-round observations must be bitwise equal.
        """
        from hamon.nrpt import nrpt
        from hamon.observers import ColdIndexObserver, NRPTStateObserver

        ebm, program = self._template()
        nc = 5
        betas = jnp.linspace(0.0, 1.0, nc)
        inits = _init_factory(nc, [ebm] * nc, [program] * nc)
        key = jax.random.key(5)
        kw = dict(betas=betas, track_round_trips=False, device=None)
        _, st_a = nrpt(
            key, ebm, program, inits, [], 12, 1, observer=NRPTStateObserver((-1,)), **kw
        )
        for pad in (nc, 12):  # pad == n (structural no-op) and pad > n
            _, st_b = nrpt(
                key,
                ebm,
                program,
                inits,
                [],
                12,
                1,
                observer=ColdIndexObserver(nc - 1),
                pad_chains_to=pad,
                **kw,
            )
            obs_a, obs_b = st_a["observations"], st_b["observations"]
            assert len(obs_a) == len(obs_b)
            for b in range(len(obs_a)):
                assert np.array_equal(np.asarray(obs_a[b]), np.asarray(obs_b[b])), (
                    pad,
                    b,
                )

    def test_pad_rejects_only_non_masking_safe_observers(self):
        """Padding admits live-index observers and rejects the rest.

        A raw -1 index records a divergent padding copy and an all-chains
        aggregate is polluted by padding, so both must be refused; a
        ColdIndexObserver reads a live position and is allowed.
        """
        from hamon.nrpt import nrpt
        from hamon.observers import (
            ColdIndexObserver,
            NRPTEnergyObserver,
            NRPTStateObserver,
        )

        ebm, program = self._template()
        nc = 4
        betas = jnp.linspace(0.0, 1.0, nc)
        inits = _init_factory(nc, [ebm] * nc, [program] * nc)
        kw = dict(betas=betas, track_round_trips=False, device=None, pad_chains_to=8)
        for obs in (NRPTEnergyObserver(nc), NRPTStateObserver((-1,))):
            assert not obs.masking_safe
            with pytest.raises(ValueError, match="observer"):
                nrpt(
                    jax.random.key(0), ebm, program, inits, [], 6, 1, observer=obs, **kw
                )
        live = ColdIndexObserver(nc - 1)
        assert live.masking_safe
        _, st = nrpt(
            jax.random.key(0), ebm, program, inits, [], 6, 1, observer=live, **kw
        )
        assert st["observations"] is not None

    def test_padded_draw_shares_compile_across_chain_counts(self):
        """Padded observer draws at different live N reuse ONE compiled loop.

        This is the point of the traced cold index: an unpadded draw (or a
        static chain_indices) specializes per chain count, so a workload whose
        discovered N drifts recompiles the draw every time.
        """
        from hamon.nrpt import _nrpt_rounds_trace_count, nrpt
        from hamon.observers import ColdIndexObserver

        ebm, program = self._template()

        def draw(nc):
            betas = jnp.linspace(0.0, 1.0, nc)
            inits = _init_factory(nc, [ebm] * nc, [program] * nc)
            nrpt(
                jax.random.key(1),
                ebm,
                program,
                inits,
                [],
                8,
                1,
                betas=betas,
                observer=ColdIndexObserver(nc - 1),
                pad_chains_to=12,
                track_round_trips=False,
                device=None,
            )

        draw(6)  # compiles the padded observer loop
        before = _nrpt_rounds_trace_count[0]
        draw(9)  # different live N, same padded ladder -> no retrace
        assert _nrpt_rounds_trace_count[0] == before

    def test_pad_rejects_factory_route_and_observer(self):
        from hamon.nrpt import nrpt
        from hamon.observers import NRPTEnergyObserver

        ebm, program = self._template()
        nc = 4
        betas = jnp.linspace(0.0, 1.0, nc)
        ebms = _ebm_factory(betas)
        programs = _program_factory(ebms)
        inits = _init_factory(nc, ebms, programs)
        with pytest.raises(ValueError, match="temperature-linear"):
            nrpt(
                jax.random.key(0),
                ebms,
                programs,
                inits,
                [],
                10,
                1,
                pad_chains_to=8,
                device=None,
            )
        with pytest.raises(ValueError, match="observer"):
            nrpt(
                jax.random.key(0),
                ebm,
                program,
                inits,
                [],
                10,
                1,
                betas=betas,
                observer=NRPTEnergyObserver(nc),
                pad_chains_to=8,
                device=None,
            )


class TestAdaptiveGridWarmup:
    """The energy-grid barrier seed's adaptive warmup (batched, R̂-stopped)."""

    def _source(self, weight_scale=0.3, frustrated=False, seed=0):
        from hamon.nrpt import _ChainSource

        rng = np.random.default_rng(seed)
        n = 36
        nodes = [SpinNode() for _ in range(n)]
        edges = [(nodes[i], nodes[(i + 1) % n]) for i in range(n)]
        if frustrated:
            w = jnp.asarray(rng.choice([-1.0, 1.0], size=n) * weight_scale)
        else:
            w = jnp.ones(n) * weight_scale
        biases = jnp.asarray(rng.normal(0, 0.1, size=n))
        ebm = IsingEBM(nodes, edges, biases, w, jnp.array(1.0))
        program = IsingSamplingProgram(ebm, [Block(nodes[::2]), Block(nodes[1::2])], [])
        src = _ChainSource(None, None, ebm, program)
        k = jax.random.key(11)

        def init_factory(n_chains, ebms, programs):
            fb = programs[0].gibbs_spec.free_blocks
            ks = jax.random.split(k, n_chains)
            return [hinton_init(ks[i], ebms[0], fb, ()) for i in range(n_chains)]

        return src, init_factory

    def test_window_rhat_converged_vs_split(self):
        from hamon.tuning import _ENERGY_WARMUP_RHAT, _window_rhat_max

        rng = np.random.default_rng(0)
        betas = np.linspace(0.0, 1.0, 11)
        # Converged: iid energies across restarts — R̂ near 1, below the
        # calibrated stopping threshold in the typical case.
        conv = rng.normal(size=(4, 11, 8))
        rh_conv = np.median(
            [_window_rhat_max(rng.normal(size=(4, 11, 8)), betas) for _ in range(20)]
        )
        assert rh_conv < _ENERGY_WARMUP_RHAT
        # Split basins: two well-separated means — R̂ far above the threshold.
        split = rng.normal(size=(4, 11, 8)) * 0.1
        split[:, :, :4] += 10.0
        assert _window_rhat_max(split, betas) > 2.0
        # β = 0 lane is excluded: split ONLY at β=0 must not trigger.
        only_b0 = rng.normal(size=(4, 11, 8)) * 0.1
        only_b0[:, 0, :4] += 10.0
        assert _window_rhat_max(only_b0, betas) < _ENERGY_WARMUP_RHAT
        del conv, split

    def test_easy_target_stops_early(self, caplog):
        import logging

        from hamon.tuning import _estimate_barrier_energy

        src, init_factory = self._source(weight_scale=0.3)
        with caplog.at_level(logging.DEBUG, logger="hamon.tuning"):
            lam, rhat = _estimate_barrier_energy(
                jax.random.key(3),
                src,
                init_factory,
                [],
                (0.0, 1.0),
                None,
                n_grid=5,
                n_samples=60,
                restarts=4,
            )
        assert math.isfinite(lam) and lam >= 0.0
        assert math.isfinite(rhat)
        exits = [r for r in caplog.records if "exit)" in r.getMessage()]
        assert exits, "expected a warmup-exit log line"
        msg = exits[-1].getMessage()
        assert "stable exit" in msg or "plateau exit" in msg
        # Never runs past the cap.
        sweeps = int(msg.split()[3])
        assert sweeps <= 2000

    def test_cap_is_respected(self, caplog):
        import logging

        from hamon.tuning import _ENERGY_WARMUP_BATCH, _estimate_barrier_energy

        src, init_factory = self._source(weight_scale=0.3)
        cap = 100
        with caplog.at_level(logging.DEBUG, logger="hamon.tuning"):
            _estimate_barrier_energy(
                jax.random.key(3),
                src,
                init_factory,
                [],
                (0.0, 1.0),
                None,
                n_grid=5,
                warmup=cap,
                n_samples=40,
                restarts=4,
            )
        msg = [r.getMessage() for r in caplog.records if "exit)" in r.getMessage()][-1]
        sweeps = int(msg.split()[3])
        assert sweeps < cap + _ENERGY_WARMUP_BATCH


class TestTuneSamplingSchedule:
    """tune_sampling_schedule: calibrated warmup / thinning / sample count."""

    def _model(self, scale=0.5):
        rng = np.random.default_rng(0)
        n = 24
        nodes = [SpinNode() for _ in range(n)]
        edges = [(nodes[i], nodes[(i + 1) % n]) for i in range(n)]
        ebm = IsingEBM(
            nodes,
            edges,
            jnp.asarray(rng.normal(0, 0.2, n)),
            jnp.ones(n) * scale,
            jnp.array(1.0),
        )
        program = IsingSamplingProgram(ebm, [Block(nodes[::2]), Block(nodes[1::2])], [])
        return ebm, program

    def test_returns_sane_schedule(self):
        from hamon import tune_sampling_schedule

        ebm, program = self._model()
        init = hinton_init(jax.random.key(1), ebm, program.gibbs_spec.free_blocks, (4,))
        sched, info = tune_sampling_schedule(
            jax.random.key(2), ebm, program, init, target_ess=16, device=None
        )
        assert sched.n_samples == 16
        assert sched.steps_per_sample >= 1
        assert sched.n_warmup >= 1
        assert sched.n_warmup == info["n_warmup"]
        assert info["warmup_exit"] in ("stable", "plateau", "cap")
        assert info["tau"] >= 1.0
        assert math.isfinite(info["rhat_final"])

    def test_deterministic(self):
        from hamon import tune_sampling_schedule

        ebm, program = self._model()
        init = hinton_init(jax.random.key(1), ebm, program.gibbs_spec.free_blocks, (4,))
        a = tune_sampling_schedule(
            jax.random.key(2), ebm, program, init, target_ess=8, device=None
        )
        b = tune_sampling_schedule(
            jax.random.key(2), ebm, program, init, target_ess=8, device=None
        )
        assert a[0] == b[0] and a[1] == b[1]

    def test_cap_respected(self):
        from hamon import tune_sampling_schedule
        from hamon.tuning import _ENERGY_WARMUP_BATCH

        ebm, program = self._model()
        init = hinton_init(jax.random.key(1), ebm, program.gibbs_spec.free_blocks, (4,))
        sched, info = tune_sampling_schedule(
            jax.random.key(2),
            ebm,
            program,
            init,
            target_ess=8,
            warmup_cap=100,
            device=None,
        )
        assert info["n_warmup"] < 100 + _ENERGY_WARMUP_BATCH

    def test_rejects_single_replica(self):
        from hamon import tune_sampling_schedule

        ebm, program = self._model()
        init = hinton_init(jax.random.key(1), ebm, program.gibbs_spec.free_blocks, (1,))
        with pytest.raises(ValueError, match="2 replicas"):
            tune_sampling_schedule(jax.random.key(2), ebm, program, init, device=None)
