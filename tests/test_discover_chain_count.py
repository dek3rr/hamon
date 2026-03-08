"""Tests for discover_chain_count in nrpt.py.

Guards against:
- Node identity mismatch (SpinNode KeyError)
- Return structure correctness
- Max-Λ tracking (conservative estimate)
- Stabilization detection
- Convergence reason reporting
- Chain count bounds
"""

import jax
import jax.numpy as jnp
import pytest

from thrml_boost import Block, SpinNode
from thrml_boost.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml_boost.nrpt import discover_chain_count


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
    def test_runs_without_error(self):
        result = discover_chain_count(
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
        result = discover_chain_count(
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
        result = discover_chain_count(
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
        result = discover_chain_count(
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
        result = discover_chain_count(
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
        result = discover_chain_count(
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
        result = discover_chain_count(
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


# ---------------------------------------------------------------------------
# Max-Λ tracking
# ---------------------------------------------------------------------------


class TestMaxLambdaTracking:
    def test_lambda_geq_lambda_raw(self):
        """Conservative Λ (max) should be >= the last raw estimate."""
        result = discover_chain_count(
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
        result = discover_chain_count(
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
