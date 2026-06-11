"""Tests for nrpt.py optimizations: vmap energies (H1) and multi-pass DEO (H5).

Validates:
- Vmapped energy computation matches loop-based reference
- Multi-pass DEO produces more swap attempts per round
- Round trip rates improve with multi-pass (more communication per round)
- All existing API contracts still hold
"""

import jax
import jax.numpy as jnp
import pytest

from hamon import (
    AbstractNRPTObserver,
    nrpt_node_samples,
    make_empty_block_state,
    make_ising_delta_fn,
    NRPTStateObserver,
)
from hamon.models import AbstractEBM, IsingEBM, hinton_init
from hamon.nrpt import _compute_base_energies, _make_reference_ebm, nrpt, nrpt_adaptive

from .utils import make_ising_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_make_ising = make_ising_grid


def _make_states(key, ebms, free_blocks, n_chains):
    keys = jax.random.split(key, n_chains)
    return [hinton_init(keys[c], ebms[0], free_blocks, ()) for c in range(n_chains)]


# ---------------------------------------------------------------------------
# H1: vmap energy correctness
# ---------------------------------------------------------------------------


class TestVmapEnergies:
    """Verify _compute_base_energies (vmapped) matches loop-based reference."""

    def _loop_base_energies(self, ebms, spec, stacked_states, clamp_state, betas):
        """Reference: the old loop-based implementation."""
        n_chains = len(ebms)
        n_free_blocks = len(stacked_states)
        energies = []
        for c in range(n_chains):
            state_c = [stacked_states[b][c] for b in range(n_free_blocks)]
            energies.append(ebms[c].energy(state_c + clamp_state, spec))
        return jnp.stack(energies) / betas

    def test_matches_loop_4chains(self):
        betas = [0.5, 1.0, 1.5, 2.0]
        _, _, fb, ebms, progs = _make_ising(8, betas, coupling=0.8)
        states = _make_states(jax.random.key(0), ebms, fb, 4)

        n_free = len(fb)
        stacked = [jnp.stack([states[c][b] for c in range(4)]) for b in range(n_free)]
        spec = progs[0].gibbs_spec
        betas_arr = jnp.array(betas)

        ref = self._loop_base_energies(ebms, spec, stacked, [], betas_arr)
        vmap_result = _compute_base_energies(ebms[0], betas_arr[0], spec, stacked, [])

        assert jnp.allclose(ref, vmap_result, atol=1e-5), (
            f"max diff: {float(jnp.max(jnp.abs(ref - vmap_result)))}"
        )

    def test_matches_loop_16chains(self):
        betas = jnp.linspace(0.3, 2.5, 16).tolist()
        _, _, fb, ebms, progs = _make_ising(16, betas, coupling=0.5)
        states = _make_states(jax.random.key(42), ebms, fb, 16)

        n_free = len(fb)
        stacked = [jnp.stack([states[c][b] for c in range(16)]) for b in range(n_free)]
        spec = progs[0].gibbs_spec
        betas_arr = jnp.array(betas)

        ref = self._loop_base_energies(ebms, spec, stacked, [], betas_arr)
        vmap_result = _compute_base_energies(ebms[0], betas_arr[0], spec, stacked, [])

        assert jnp.allclose(ref, vmap_result, atol=1e-5)

    def test_nonzero_biases(self):
        """Verify with non-trivial biases (not just coupling)."""
        L = 6
        n_nodes, n_edges = L * L, 2 * L * (L - 1)
        biases = jax.random.normal(jax.random.key(7), (n_nodes,)) * 0.3
        weights = jax.random.normal(jax.random.key(8), (n_edges,)) * 0.5

        betas = [0.5, 1.0, 1.5, 2.0]
        _, _, fb, ebms, progs = _make_ising(L, betas, biases=biases, weights=weights)
        states = _make_states(jax.random.key(99), ebms, fb, 4)

        n_free = len(fb)
        stacked = [jnp.stack([states[c][b] for c in range(4)]) for b in range(n_free)]
        spec = progs[0].gibbs_spec
        betas_arr = jnp.array(betas)

        ref = self._loop_base_energies(ebms, spec, stacked, [], betas_arr)
        vmap_result = _compute_base_energies(ebms[0], betas_arr[0], spec, stacked, [])

        assert jnp.allclose(ref, vmap_result, atol=1e-5)


# ---------------------------------------------------------------------------
# DEO correctness (single-pass, non-reversible)
# ---------------------------------------------------------------------------


class TestSinglePassDEO:
    """Verify single-pass DEO: one swap parity per round, alternating."""

    def test_pairs_attempted_alternating(self):
        """Even pairs attempted on even rounds, odd on odd rounds."""
        betas = [0.5, 1.0, 1.5, 2.0]
        _, _, fb, ebms, progs = _make_ising(4, betas, coupling=0.0)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)

        n_rounds = 50
        _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 4,
            [],
            n_rounds=n_rounds,
            gibbs_steps_per_round=1,
        )
        attempted = stats["attempted"]
        # 3 pairs: even={0,2}, odd={1}
        # 50 rounds â†’ 25 even rounds, 25 odd rounds
        assert int(attempted[0]) == 25  # even pair
        assert int(attempted[1]) == 25  # odd pair
        assert int(attempted[2]) == 25  # even pair

    def test_round_trips_with_zero_coupling(self):
        """Zero coupling â†’ all swaps accepted â†’ conveyor belt round trips."""
        betas = [0.5, 1.0, 1.5, 2.0]
        _, _, fb, ebms, progs = _make_ising(4, betas, coupling=0.0)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)

        # 4 chains, single-pass DEO: ~6 rounds per round trip
        _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 4,
            [],
            n_rounds=100,
            gibbs_steps_per_round=1,
        )
        total_rts = int(jnp.sum(stats["index_state"]["round_trips"]))
        assert total_rts > 0, "Expected round trips with zero-coupling model"

    def test_acceptance_rates_valid(self):
        """Acceptance rates should be in [0, 1] for all pairs."""
        betas = [0.3, 0.8, 1.2, 1.8, 2.5]
        _, _, fb, ebms, progs = _make_ising(8, betas, coupling=0.5)
        states = _make_states(jax.random.key(0), ebms, fb, 5)

        _, stats = nrpt(
            jax.random.key(1),
            ebms,
            progs,
            states,
            [],
            n_rounds=100,
            gibbs_steps_per_round=3,
        )
        acc = stats["acceptance_rate"]
        assert jnp.all(acc >= 0.0) and jnp.all(acc <= 1.0)
        assert jnp.all(stats["attempted"] > 0)

    def test_lambda_consistent(self):
        """Î› = sum(rejection_rates) should hold."""
        betas = [0.5, 1.0, 1.5, 2.0]
        _, _, fb, ebms, progs = _make_ising(8, betas, coupling=0.5)
        states = _make_states(jax.random.key(0), ebms, fb, 4)

        _, stats = nrpt(
            jax.random.key(42),
            ebms,
            progs,
            states,
            [],
            n_rounds=50,
            gibbs_steps_per_round=3,
        )
        diag = stats["round_trip_diagnostics"]
        assert jnp.allclose(
            diag["Lambda"], jnp.sum(stats["rejection_rates"]), atol=1e-5
        )

    def test_deo_alternation_asymmetric_rates(self):
        """Even-pair and odd-pair rates should differ with asymmetric betas."""
        betas = [0.1, 0.2, 2.0, 2.1]
        _, _, fb, ebms, progs = _make_ising(8, betas, coupling=1.0)
        states = _make_states(jax.random.key(0), ebms, fb, 4)

        _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            states,
            [],
            n_rounds=200,
            gibbs_steps_per_round=3,
        )
        # Pair 0 (Î²=0.1â†”0.2, small gap) vs pair 1 (Î²=0.2â†”2.0, large gap)
        rates = stats["acceptance_rate"]
        assert not jnp.allclose(rates[0], rates[1], atol=0.05), (
            f"Suspiciously similar rates: {rates}"
        )

    def test_multi_pass_would_break_conveyor(self):
        """Document WHY multi-pass DEO is wrong.

        With 4 chains and all swaps accepted, evenâˆ˜odd followed by
        oddâˆ˜even = identity permutation. States oscillate with period 2
        instead of drifting through the temperature ladder.
        """
        # Verify the algebra: compose the two permutations
        n = 4
        even_perm = jnp.array([1, 0, 3, 2])  # swap (0,1) and (2,3)
        odd_perm = jnp.array([0, 2, 1, 3])  # swap (1,2)

        # even_first: even then odd
        composed_ef = odd_perm[even_perm]
        # odd_first: odd then even
        composed_of = even_perm[odd_perm]

        # Two rounds should compose to identity
        full_cycle = composed_of[composed_ef]
        assert jnp.array_equal(full_cycle, jnp.arange(n)), (
            f"Expected identity, got {full_cycle}"
        )


# ---------------------------------------------------------------------------
# Existing API contracts still hold
# ---------------------------------------------------------------------------


class TestAPIUnchanged:
    def test_basic_smoke(self):
        betas = [0.5, 1.0, 1.5]
        _, _, fb, ebms, progs = _make_ising(4, betas)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)
        states, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 3,
            [],
            n_rounds=5,
            gibbs_steps_per_round=2,
        )
        assert len(states) == 3
        assert "accepted" in stats
        assert "attempted" in stats
        assert "round_trip_diagnostics" in stats

    def test_zero_rounds(self):
        betas = [0.5, 1.0]
        _, _, fb, ebms, progs = _make_ising(4, betas)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)
        _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 2,
            [],
            n_rounds=0,
            gibbs_steps_per_round=1,
        )
        assert jnp.all(stats["accepted"] == 0)

    def test_diagnostics_absent_when_disabled(self):
        betas = [0.5, 1.0, 1.5]
        _, _, fb, ebms, progs = _make_ising(4, betas)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)
        _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 3,
            [],
            n_rounds=10,
            gibbs_steps_per_round=2,
            track_round_trips=False,
        )
        assert "round_trip_diagnostics" not in stats

    def test_tau_bounded(self):
        betas = [0.5, 1.0, 2.0]
        _, _, fb, ebms, progs = _make_ising(8, betas, coupling=1.0)
        states = _make_states(jax.random.key(0), ebms, fb, 3)
        _, stats = nrpt(
            jax.random.key(7),
            ebms,
            progs,
            states,
            [],
            n_rounds=30,
            gibbs_steps_per_round=2,
        )
        tau = stats["round_trip_diagnostics"]["tau_predicted"]
        assert 0.0 < float(tau) <= 0.5


# ---------------------------------------------------------------------------
# NRPT observer
# ---------------------------------------------------------------------------


class TestNRPTObserver:
    """Verify NRPTStateObserver collects per-round states correctly."""

    def test_observer_cold_chain_shape(self):
        """Observations should have shape (n_rounds, 1, ...) for cold chain."""
        betas = [0.5, 1.0, 1.5]
        n_chains = 3
        n_rounds = 10
        _, _, fb, ebms, progs = _make_ising(4, betas, coupling=0.5)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)

        obs = NRPTStateObserver(chain_indices=(-1,))
        _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * n_chains,
            [],
            n_rounds=n_rounds,
            gibbs_steps_per_round=2,
            observer=obs,
        )
        assert "observations" in stats
        # One array per free block; leading axis = n_rounds, then 1 chain
        for arr in stats["observations"]:
            assert arr.shape[0] == n_rounds
            assert arr.shape[1] == 1  # one chain index

    def test_observer_multiple_chains(self):
        """Collect states from multiple chain indices."""
        betas = [0.5, 1.0, 1.5, 2.0]
        n_chains = 4
        n_rounds = 5
        _, _, fb, ebms, progs = _make_ising(4, betas)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)

        obs = NRPTStateObserver(chain_indices=(0, -1))
        _, stats = nrpt(
            jax.random.key(1),
            ebms,
            progs,
            [init] * n_chains,
            [],
            n_rounds=n_rounds,
            gibbs_steps_per_round=1,
            observer=obs,
        )
        for arr in stats["observations"]:
            assert arr.shape[0] == n_rounds
            assert arr.shape[1] == 2  # two chain indices

    def test_observer_last_round_matches_final_state(self):
        """The last observation should match the returned cold-chain state."""
        betas = [0.5, 1.0, 1.5]
        n_chains = 3
        n_rounds = 10
        _, _, fb, ebms, progs = _make_ising(4, betas, coupling=0.5)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)

        obs = NRPTStateObserver(chain_indices=(-1,))
        states, stats = nrpt(
            jax.random.key(7),
            ebms,
            progs,
            [init] * n_chains,
            [],
            n_rounds=n_rounds,
            gibbs_steps_per_round=2,
            observer=obs,
        )
        # states[-1] is the cold chain; observations[-1] is last round
        cold_state = states[-1]  # list of arrays, one per free block
        for b, arr in enumerate(stats["observations"]):
            last_obs = arr[-1, 0]  # last round, first (only) chain index
            assert jnp.array_equal(last_obs, cold_state[b])

    def test_no_observer_backward_compat(self):
        """Without observer, stats should not contain observation keys."""
        betas = [0.5, 1.0, 1.5]
        _, _, fb, ebms, progs = _make_ising(4, betas)
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)

        _, stats = nrpt(
            jax.random.key(0),
            ebms,
            progs,
            [init] * 3,
            [],
            n_rounds=5,
            gibbs_steps_per_round=2,
        )
        assert "observations" not in stats
        assert "observer_carry" not in stats


# ---------------------------------------------------------------------------
# Î²â‚€ = 0 ladders (regression: base energies were NaN, silently rejecting
# every swap)
# ---------------------------------------------------------------------------


class _OpaqueEBM(AbstractEBM):
    """Delegates energy to an IsingEBM but does not implement with_beta().

    Exercises the fallback reference-energy path in nrpt for EBM classes
    that only satisfy the minimal AbstractEBM contract.
    """

    inner: IsingEBM

    def energy(self, state, blocks):
        return self.inner.energy(state, blocks)


class TestZeroBetaHotChain:
    """A hottest chain at exactly Î² = 0 must produce finite, working swaps."""

    def test_swaps_accepted_with_zero_beta0(self):
        """Regression: Î²â‚€ = 0 yielded NaN base energies â†’ 0% acceptance."""
        betas = [0.0, 0.4, 0.8, 1.2]
        _, _, fb, ebms, progs = _make_ising(4, betas, coupling=0.4)
        states = _make_states(jax.random.key(0), ebms, fb, 4)

        _, stats = nrpt(
            jax.random.key(1),
            ebms,
            progs,
            states,
            [],
            n_rounds=100,
            gibbs_steps_per_round=2,
        )
        acc = stats["acceptance_rate"]
        assert jnp.all(jnp.isfinite(acc)), f"non-finite acceptance: {acc}"
        # Pre-fix this was exactly 0 for every pair. With this gentle ladder
        # every pair should accept at least once over 100 rounds.
        assert jnp.all(acc > 0.0), f"swaps never accepted: {acc}"

    def test_reference_ebm_prefers_beta1_copy(self):
        """The reference EBM is an exact Î²=1 copy when with_beta() exists,
        and base energies match a directly-constructed Î²=1 model."""
        betas = [0.0, 0.5, 1.0]
        nodes, edges, fb, ebms, progs = _make_ising(4, betas, coupling=0.7)
        states = _make_states(jax.random.key(3), ebms, fb, 3)

        betas_arr = jnp.array(betas)
        ebm_ref, beta_ref = _make_reference_ebm(ebms, betas_arr)
        assert float(ebm_ref.beta) == 1.0
        assert float(beta_ref) == 1.0

        spec = progs[0].gibbs_spec
        stacked = [jnp.stack([states[c][b] for c in range(3)]) for b in range(len(fb))]
        base_E = _compute_base_energies(ebm_ref, beta_ref, spec, stacked, [])
        assert jnp.all(jnp.isfinite(base_E))

        ebm_unit = IsingEBM(
            nodes, edges, ebms[0].biases, ebms[0].weights, jnp.array(1.0)
        )
        expected = jnp.stack(
            [
                ebm_unit.energy([states[c][b] for b in range(len(fb))], spec)
                for c in range(3)
            ]
        )
        assert jnp.allclose(base_E, expected, atol=1e-5)

    def test_fallback_without_with_beta_matches_ising_run(self):
        """EBMs lacking with_beta() use the coldest chain as reference and
        must reproduce the IsingEBM run exactly (same RNG, same energies)."""
        betas = [0.5, 1.0, 1.5]
        _, _, fb, ebms, progs = _make_ising(4, betas, coupling=0.5)
        opaque_ebms = [_OpaqueEBM(inner=e) for e in ebms]
        states = _make_states(jax.random.key(5), ebms, fb, 3)
        betas_arr = jnp.array(betas)

        _, stats_ising = nrpt(
            jax.random.key(6),
            ebms,
            progs,
            states,
            [],
            n_rounds=60,
            gibbs_steps_per_round=2,
            betas=betas_arr,
        )
        _, stats_opaque = nrpt(
            jax.random.key(6),
            opaque_ebms,
            progs,
            states,
            [],
            n_rounds=60,
            gibbs_steps_per_round=2,
            betas=betas_arr,
        )
        assert jnp.array_equal(stats_ising["attempted"], stats_opaque["attempted"])
        assert jnp.array_equal(stats_ising["accepted"], stats_opaque["accepted"])

    def test_fallback_with_zero_cold_beta_raises(self):
        """No with_beta() and Î²_cold = 0 is unrecoverable â€” must raise."""
        betas = [0.0, 0.0]
        _, _, fb, ebms, progs = _make_ising(4, betas)
        opaque_ebms = [_OpaqueEBM(inner=e) for e in ebms]
        init = make_empty_block_state(fb, ebms[0].node_shape_dtypes)

        with pytest.raises(ValueError, match="with_beta"):
            nrpt(
                jax.random.key(0),
                opaque_ebms,
                progs,
                [init] * 2,
                [],
                n_rounds=5,
                gibbs_steps_per_round=1,
                betas=jnp.array(betas),
            )

    def test_adaptive_tuning_with_zero_beta0(self):
        """nrpt_adaptive (the path ising_sample uses) works from Î²â‚€ = 0."""
        betas = jnp.linspace(0.0, 1.2, 4)
        _, _, fb, ebms, progs = _make_ising(4, [float(b) for b in betas], coupling=0.6)
        ebm, prog = ebms[-1], progs[-1]
        states = _make_states(jax.random.key(8), ebms, fb, 4)

        _, stats = nrpt_adaptive(
            jax.random.key(9),
            ebm=ebm,
            program=prog,
            init_states=states,
            clamp_state=[],
            n_rounds=50,
            gibbs_steps_per_round=2,
            initial_betas=betas,
            n_tune=2,
            rounds_per_tune=50,
        )
        assert jnp.all(jnp.isfinite(stats["betas"]))
        assert jnp.all(jnp.isfinite(stats["acceptance_rate"]))
        assert jnp.all(stats["acceptance_rate"] > 0.0)
        for phase in stats["tuning_history"]:
            assert jnp.isfinite(phase["Lambda"])


# ---------------------------------------------------------------------------
# Observer energy/state alignment (regression: in fresh-energy mode the
# observer received post-swap states paired with pre-swap energies)
# ---------------------------------------------------------------------------


class _EnergyRecordingObserver(AbstractNRPTObserver):
    """Record (states, base_energies) each round so alignment can be checked."""

    def __call__(self, stacked_states, base_energies, round_idx, carry):
        return None, (list(stacked_states), base_energies)


class TestObserverEnergyAlignment:
    """base_energies handed to observers must describe the states they
    accompany, in both energy modes."""

    n_rounds = 12
    n_chains = 4
    # Close betas â†’ high swap acceptance, so misalignment cannot hide
    # behind an identity permutation.
    betas = [0.8, 1.0, 1.2, 1.4]

    def _run(self, energy_delta_fn=None):
        nodes, edges, fb, ebms, progs = _make_ising(4, self.betas, coupling=0.5)
        states = _make_states(jax.random.key(0), ebms, fb, self.n_chains)

        kwargs = {}
        if energy_delta_fn == "ising":
            kwargs["energy_delta_fn"] = make_ising_delta_fn(
                nodes, edges, fb, ebms[0].biases, ebms[0].weights
            )

        obs = _EnergyRecordingObserver()
        _, stats = nrpt(
            jax.random.key(1),
            ebms,
            progs,
            states,
            [],
            n_rounds=self.n_rounds,
            gibbs_steps_per_round=2,
            observer=obs,
            **kwargs,
        )
        # The regression only manifests when swaps actually happen.
        assert int(jnp.sum(stats["accepted"])) > 0, "test needs accepted swaps"

        ebm_unit = IsingEBM(
            nodes, edges, ebms[0].biases, ebms[0].weights, jnp.array(1.0)
        )
        spec = progs[0].gibbs_spec
        obs_states, obs_energies = stats["observations"]

        for r in range(self.n_rounds):
            for c in range(self.n_chains):
                state_rc = [obs_states[b][r][c] for b in range(len(fb))]
                expected = float(ebm_unit.energy(state_rc, spec))
                recorded = float(obs_energies[r][c])
                assert abs(recorded - expected) < 1e-3, (
                    f"round {r} chain {c}: recorded {recorded}, state energy {expected}"
                )

    def test_fresh_energy_mode(self):
        """Default mode: energies recomputed each round, permuted after swaps."""
        self._run()

    def test_cached_energy_mode(self):
        """Delta-cached mode: running cache stays aligned through swaps."""
        self._run(energy_delta_fn="ising")


# ---------------------------------------------------------------------------
# Temperature-linear mode (single base program + per-chain Î² scaling)
# ---------------------------------------------------------------------------


class TestTemperatureLinearMode:
    """nrpt with single template (ebm, program) objects must reproduce the
    per-chain-programs path bit-for-bit for Î²-linear models."""

    def _setup(self):
        betas = [0.4, 0.8, 1.2, 1.6]
        _, _, fb, ebms, progs = _make_ising(4, betas, coupling=0.5)
        inits = _make_states(jax.random.key(2), ebms, fb, 4)
        return jnp.array(betas), fb, ebms, progs, inits

    def test_matches_per_chain_programs(self):
        betas, fb, ebms, progs, inits = self._setup()
        obs = NRPTStateObserver(chain_indices=(0, -1))

        states_seq, stats_seq = nrpt(
            jax.random.key(11),
            ebms,
            progs,
            inits,
            [],
            41,
            2,
            betas=betas,
            observer=obs,
        )
        # Template objects at an arbitrary Î² â€” rebased to Î²=1 internally.
        states_lin, stats_lin = nrpt(
            jax.random.key(11),
            ebms[-1],
            progs[-1],
            inits,
            [],
            41,
            2,
            betas=betas,
            observer=obs,
        )

        # Guard against a vacuous comparison: swaps must actually happen.
        assert int(jnp.sum(stats_seq["accepted"])) > 0
        assert jnp.array_equal(stats_seq["accepted"], stats_lin["accepted"])
        assert jnp.array_equal(stats_seq["attempted"], stats_lin["attempted"])
        assert jnp.array_equal(
            stats_seq["index_state"]["round_trips"],
            stats_lin["index_state"]["round_trips"],
        )
        for c in range(4):
            for b in range(len(fb)):
                assert jnp.array_equal(states_seq[c][b], states_lin[c][b])
        for o_seq, o_lin in zip(stats_seq["observations"], stats_lin["observations"]):
            assert jnp.array_equal(o_seq, o_lin)

    def test_mixed_single_and_sequence_raises(self):
        betas, fb, ebms, progs, inits = self._setup()
        with pytest.raises(ValueError, match="both"):
            nrpt(jax.random.key(0), ebms[-1], progs, inits, [], 5, 1, betas=betas)

    def test_requires_betas(self):
        betas, fb, ebms, progs, inits = self._setup()
        with pytest.raises(ValueError, match="betas"):
            nrpt(jax.random.key(0), ebms[-1], progs[-1], inits, [], 5, 1)

    def test_init_states_length_mismatch_raises(self):
        betas, fb, ebms, progs, inits = self._setup()
        with pytest.raises(ValueError, match="init_states"):
            nrpt(
                jax.random.key(0),
                ebms[-1],
                progs[-1],
                inits[:3],
                [],
                5,
                1,
                betas=betas,
            )


class TestJitCacheReuse:
    """Repeated nrpt calls with the same Î²=1 base pair must reuse the
    compiled round loop instead of retracing."""

    def test_repeated_calls_do_not_retrace(self):
        from hamon.nrpt import _nrpt_rounds_trace_count

        betas = jnp.array([0.5, 1.0, 1.5])
        _, _, fb, ebms, progs = _make_ising(4, [0.5, 1.0, 1.5], coupling=0.5)
        inits = _make_states(jax.random.key(0), ebms, fb, 3)
        base_ebm = ebms[0].with_beta(jnp.asarray(1.0))
        base_prog = progs[0].with_ebm(base_ebm)

        before = _nrpt_rounds_trace_count[0]
        nrpt(jax.random.key(1), base_ebm, base_prog, inits, [], 6, 1, betas=betas)
        assert _nrpt_rounds_trace_count[0] == before + 1

        # Same static structure, different betas values and key â†’ cache hit.
        nrpt(
            jax.random.key(2),
            base_ebm,
            base_prog,
            inits,
            [],
            6,
            1,
            betas=jnp.array([0.4, 0.9, 1.4]),
        )
        assert _nrpt_rounds_trace_count[0] == before + 1, "round loop retraced"

    def test_adaptive_phases_compile_once(self):
        from hamon.nrpt import _nrpt_rounds_trace_count

        _, _, fb, ebms, progs = _make_ising(4, [0.5, 1.0, 1.5], coupling=0.5)
        inits = _make_states(jax.random.key(3), ebms, fb, 3)

        before = _nrpt_rounds_trace_count[0]
        nrpt_adaptive(
            jax.random.key(4),
            ebm=ebms[-1],
            program=progs[-1],
            init_states=inits,
            clamp_state=[],
            n_rounds=10,  # == rounds_per_tune so production reuses the trace
            gibbs_steps_per_round=1,
            initial_betas=jnp.array([0.5, 1.0, 1.5]),
            n_tune=3,
            rounds_per_tune=10,
        )
        traces = _nrpt_rounds_trace_count[0] - before
        assert traces == 1, f"expected 1 trace across 4 phases, got {traces}"


# ---------------------------------------------------------------------------
# Usability features: node-order extraction, beta validation, stacked inits,
# tuning early-stop
# ---------------------------------------------------------------------------


class TestNodeOrderSamples:
    """nrpt_node_samples must invert the block->node permutation exactly."""

    def _setup(self):
        betas = [0.6, 1.0]
        nodes, _, fb, ebms, progs = _make_ising(2, betas, coupling=0.3)
        return nodes, fb, ebms, progs

    def test_synthetic_permutation_inverted(self):
        nodes, fb, ebms, progs = self._setup()
        node_idx = {id(n): i for i, n in enumerate(nodes)}
        n_rounds = 3
        observations = []
        for block in fb:
            vals = jnp.array([node_idx[id(n)] for n in block.nodes], dtype=jnp.int32)
            arr = jnp.broadcast_to(vals, (n_rounds, len(block)))
            # chain slot 1 carries +100 so chain selection is testable
            observations.append(jnp.stack([arr, arr + 100], axis=1))

        out = nrpt_node_samples(observations, progs[0], nodes, chain_index=0)
        expected = jnp.broadcast_to(
            jnp.arange(len(nodes), dtype=jnp.int32), (n_rounds, len(nodes))
        )
        assert jnp.array_equal(out, expected)

        out_cold = nrpt_node_samples(observations, progs[0], nodes, chain_index=1)
        assert jnp.array_equal(out_cold, expected + 100)

    def test_end_to_end_matches_manual_indexing(self):
        nodes, fb, ebms, progs = self._setup()
        inits = _make_states(jax.random.key(0), ebms, fb, 2)
        obs = NRPTStateObserver(chain_indices=(0, -1))
        _, stats = nrpt(
            jax.random.key(1),
            ebms,
            progs,
            inits,
            [],
            8,
            1,
            betas=jnp.array([0.6, 1.0]),
            observer=obs,
        )
        out = nrpt_node_samples(stats["observations"], progs[0], nodes, chain_index=1)

        # Independent reference: locate each node by scanning the free blocks.
        for i, node in enumerate(nodes):
            for b, block in enumerate(fb):
                if node in block.nodes:
                    k = block.nodes.index(node)
                    ref = stats["observations"][b][:, 1, k]
                    assert jnp.array_equal(out[:, i], ref)
                    break

    def test_subset_and_reorder(self):
        nodes, fb, ebms, progs = self._setup()
        node_idx = {id(n): i for i, n in enumerate(nodes)}
        observations = []
        for block in fb:
            vals = jnp.array([node_idx[id(n)] for n in block.nodes], dtype=jnp.int32)
            observations.append(jnp.broadcast_to(vals, (2, 1, len(block))))
        subset = [nodes[3], nodes[0]]
        out = nrpt_node_samples(observations, progs[0], subset)
        assert jnp.array_equal(out, jnp.array([[3, 0], [3, 0]]))

    def test_foreign_node_raises(self):
        from hamon import SpinNode

        nodes, fb, ebms, progs = self._setup()
        observations = [jnp.zeros((2, 1, len(b)), dtype=jnp.int32) for b in fb]
        with pytest.raises(ValueError, match="not found"):
            nrpt_node_samples(observations, progs[0], [SpinNode()])

    def test_wrong_observation_count_raises(self):
        nodes, fb, ebms, progs = self._setup()
        with pytest.raises(ValueError, match="per free block"):
            nrpt_node_samples([jnp.zeros((2, 1, 2))], progs[0], nodes)


class TestBetaLadderValidation:
    def test_descending_betas_raise_linear_mode(self):
        nodes, _, fb, ebms, progs = _make_ising(2, [0.5, 1.0, 1.5])
        inits = _make_states(jax.random.key(0), ebms, fb, 3)
        with pytest.raises(ValueError, match="ascending"):
            nrpt(
                jax.random.key(1),
                ebms[-1],
                progs[-1],
                inits,
                [],
                4,
                1,
                betas=jnp.array([1.5, 1.0, 0.5]),
            )

    def test_shuffled_betas_raise_sequence_mode(self):
        nodes, _, fb, ebms, progs = _make_ising(2, [0.5, 1.0, 1.5])
        inits = _make_states(jax.random.key(0), ebms, fb, 3)
        with pytest.raises(ValueError, match="ascending"):
            nrpt(
                jax.random.key(1),
                ebms,
                progs,
                inits,
                [],
                4,
                1,
                betas=jnp.array([1.0, 0.5, 1.5]),
            )

    def test_betas_length_mismatch_raises(self):
        nodes, _, fb, ebms, progs = _make_ising(2, [0.5, 1.0, 1.5])
        inits = _make_states(jax.random.key(0), ebms, fb, 3)
        with pytest.raises(ValueError, match="one entry per chain"):
            nrpt(
                jax.random.key(1),
                ebms,
                progs,
                inits,
                [],
                4,
                1,
                betas=jnp.array([0.5, 1.0]),
            )


class TestStackedInitStates:
    def test_stacked_matches_per_chain_lists(self):
        betas = jnp.array([0.5, 1.0, 1.5])
        nodes, _, fb, ebms, progs = _make_ising(4, [float(b) for b in betas])
        per_chain = _make_states(jax.random.key(0), ebms, fb, 3)
        stacked = [
            jnp.stack([per_chain[c][b] for c in range(3)]) for b in range(len(fb))
        ]

        states_a, stats_a = nrpt(
            jax.random.key(2), ebms[-1], progs[-1], per_chain, [], 20, 2, betas=betas
        )
        states_b, stats_b = nrpt(
            jax.random.key(2), ebms[-1], progs[-1], stacked, [], 20, 2, betas=betas
        )
        assert jnp.array_equal(stats_a["accepted"], stats_b["accepted"])
        for c in range(3):
            for b in range(len(fb)):
                assert jnp.array_equal(states_a[c][b], states_b[c][b])

    def test_hinton_init_batch_shape_works_directly(self):
        betas = jnp.array([0.5, 1.0, 1.5])
        nodes, _, fb, ebms, progs = _make_ising(4, [float(b) for b in betas])
        stacked = hinton_init(jax.random.key(0), ebms[0], fb, (3,))
        _, stats = nrpt(
            jax.random.key(1), ebms[-1], progs[-1], stacked, [], 10, 1, betas=betas
        )
        assert jnp.all(jnp.isfinite(stats["acceptance_rate"]))

    def test_wrong_leading_dim_raises(self):
        betas = jnp.array([0.5, 1.0, 1.5])
        nodes, _, fb, ebms, progs = _make_ising(4, [float(b) for b in betas])
        stacked = hinton_init(jax.random.key(0), ebms[0], fb, (2,))  # 2 != 3
        with pytest.raises(ValueError, match="leading dimension"):
            nrpt(jax.random.key(1), ebms[-1], progs[-1], stacked, [], 4, 1, betas=betas)


class TestTuneEarlyStop:
    def _run(self, tune_tol):
        betas = jnp.array([0.5, 1.0, 1.5])
        nodes, _, fb, ebms, progs = _make_ising(4, [float(b) for b in betas])
        inits = _make_states(jax.random.key(0), ebms, fb, 3)
        _, stats = nrpt_adaptive(
            jax.random.key(1),
            ebm=ebms[-1],
            program=progs[-1],
            init_states=inits,
            clamp_state=[],
            n_rounds=10,
            gibbs_steps_per_round=1,
            initial_betas=betas,
            n_tune=4,
            rounds_per_tune=10,
            tune_tol=tune_tol,
        )
        return stats

    def test_huge_tol_stops_after_first_phase(self):
        stats = self._run(tune_tol=10.0)
        assert len(stats["tuning_history"]) == 1

    def test_default_runs_all_phases(self):
        stats = self._run(tune_tol=None)
        assert len(stats["tuning_history"]) == 4
        assert all("max_beta_shift" in h for h in stats["tuning_history"])
