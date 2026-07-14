#!/usr/bin/env python
"""MNIST training demonstration for hamon.

Trains an Ising energy-based model (a double-grid RBM-style architecture) on a
filtered 3-class MNIST subset using persistent contrastive divergence (PCD),
then measures classification accuracy by clamping the image and sampling the
label spots. The schedules for both training phases are calibrated per segment
at the current parameters (``tune_sampling_schedule``) rather than hand-picked,
which is the behaviour this end-to-end run exercises.

This used to live in the unit suite (``tests/test_train_mnist.py``) but it is a
full training pipeline, not a unit test: it takes minutes, loads ~31 MB of data,
and asserts a single downstream metric. It belongs with the other benchmarks.

Usage:
    python benchmarks/train_mnist.py                 # 1 epoch, expects acc > 0.9
    python benchmarks/train_mnist.py --epochs 3

Exit code is 0 when best accuracy exceeds the threshold (0.9), 1 otherwise.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Key

from hamon.block_management import Block
from hamon.block_sampling import SamplingSchedule, sample_states
from hamon.models.ising import (
    Edge,
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
    estimate_kl_grad,
    hinton_init,
)
from hamon.pgm import AbstractNode, SpinNode
from hamon.tuning import tune_sampling_schedule

_DATA_DIR = Path(__file__).resolve().parent / "mnist_test_data"
_ACCURACY_THRESHOLD = 0.9


def get_double_grid(
    side_len: int,
    jumps: Sequence[int],
    n_visible: int,
    node: type[AbstractNode],
    key: Key[Array, ""],
) -> tuple[Block, Block, Block, Block, list[AbstractNode], list[Edge]]:
    size = side_len**2
    assert n_visible <= size

    indices = np.arange(size, dtype=np.int32)
    rows, cols = np.divmod(indices, side_len)

    edge_groups = [np.column_stack((indices, indices))]
    for d in jumps:
        for di, dj in ((-d, 0), (d, 0), (0, -d), (0, d)):
            neighbours = ((rows + di) % side_len) * side_len + ((cols + dj) % side_len)
            edge_groups.append(np.column_stack((indices, neighbours)))

    edges_arr = np.concatenate(edge_groups, axis=0)

    deg = 4 * len(jumps) + 1
    total_edges = size * deg
    assert edges_arr.shape == (total_edges, 2)

    nodes_upper = [node() for _ in range(size)]
    nodes_lower = [node() for _ in range(size)]
    all_nodes = nodes_upper + nodes_lower
    all_edges = [(nodes_upper[i], nodes_lower[j]) for i, j in edges_arr.tolist()]

    visible_indices = np.asarray(
        jax.device_get(jax.random.permutation(key, jnp.arange(size))[:n_visible]),
        dtype=np.intp,
    )
    visible_nodes = [nodes_upper[i] for i in visible_indices]

    visible_mask = np.zeros(size, dtype=np.bool_)
    visible_mask[visible_indices] = True
    upper_without_visible = [
        upper_node for i, upper_node in enumerate(nodes_upper) if not visible_mask[i]
    ]

    return (
        Block(nodes_upper),
        Block(nodes_lower),
        Block(visible_nodes),
        Block(upper_without_visible),
        all_nodes,
        all_edges,
    )


class MnistTraining:
    def __init__(self, n_epochs: int = 1):
        self.target_classes = [0, 3, 4]
        self.num_label_spots = 10
        label_size = len(self.target_classes) * self.num_label_spots
        data_dim = 28 * 28 + label_size

        self.train_data_filtered = jnp.load(_DATA_DIR / "train_data_filtered.npy")
        self.sep_images_test = {}
        for digit in self.target_classes:
            self.sep_images_test[digit] = jnp.load(
                _DATA_DIR / f"sep_images_test_{digit}.npy"
            )

        (
            upper_grid,
            lower_grid,
            visible_nodes,
            upper_without_visible,
            all_nodes,
            all_edges,
        ) = get_double_grid(40, [1, 4, 15], data_dim, SpinNode, jax.random.key(0))

        self.init_model = IsingEBM(
            all_nodes,
            all_edges,
            jnp.zeros((len(all_nodes),), dtype=float),
            jnp.zeros((len(all_edges),), dtype=float),
            jnp.array(1.0),
        )

        self.positive_sampling_blocks = [upper_without_visible, lower_grid]
        self.negative_sampling_blocks = [upper_grid, lower_grid]
        self.training_data_blocks = [visible_nodes]

        image_block = Block(visible_nodes.nodes[: 28 * 28])
        image_nodes = set(image_block.nodes)
        upper_without_image = Block(
            [node for node in upper_grid if node not in image_nodes]
        )
        self.classification_sampling_blocks = [upper_without_image, lower_grid]
        self.classification_data_blocks = [image_block]
        self.classification_label_block = Block(visible_nodes.nodes[28 * 28 :])

        # Schedules are calibrated per epoch at the current parameters (see
        # _calibrate_schedules) instead of hand-picked constants; sample
        # budgets (n_samples) match the previous hardcoded schedules.
        self.n_samples_negative = 40
        self.n_samples_positive = 20
        self.calibration_replicas = 8

        self.optim = optax.adam(learning_rate=0.01)
        self.n_epochs = n_epochs

    @staticmethod
    def _pow2_ceil(x, cap=16):
        """Round up to the nearest power of two (≤ cap).

        Quantising ``steps_per_sample`` this way keeps the number of distinct
        schedules — and therefore segment-scan recompiles — small as the
        calibrated thinning grows monotonically through training.
        """
        s = 1
        while s < x and s < cap:
            s *= 2
        return s

    def _calibrate_schedules(self, key, model):
        """Measure warmup and thinning for both training phases at this θ.

        The positive phase is data-clamped (heavily conditioned) and the
        negative phase is free — they mix at different rates and get separate
        schedules. The negative chains are persistent (PCD): warmup is paid
        once up front, so the per-step negative schedule carries none. Thinning
        is quantised to a power of two to bound recompiles.
        """
        k_pos, k_neg, k_ip, k_in = jax.random.split(key, 4)
        R = self.calibration_replicas

        prog_neg = IsingSamplingProgram(model, self.negative_sampling_blocks, [])
        init_neg = hinton_init(k_in, model, prog_neg.gibbs_spec.free_blocks, (R,))
        sched_neg, info_neg = tune_sampling_schedule(
            k_neg,
            model,
            prog_neg,
            init_neg,
            target_ess=self.n_samples_negative,
        )

        prog_pos = IsingSamplingProgram(
            model, self.positive_sampling_blocks, self.training_data_blocks
        )
        clamp_pos = [self.train_data_filtered[0].astype(jnp.bool_)]
        init_pos = hinton_init(k_ip, model, prog_pos.gibbs_spec.free_blocks, (R,))
        sched_pos, info_pos = tune_sampling_schedule(
            k_pos,
            model,
            prog_pos,
            init_pos,
            clamp_pos,
            target_ess=self.n_samples_positive,
        )

        # PCD: the per-step negative schedule drops the warmup (chains are
        # persistent); the calibrated warmup pre-warms them once.
        schedule_negative = SamplingSchedule(
            0, self.n_samples_negative, self._pow2_ceil(sched_neg.steps_per_sample)
        )
        # Positive warmup quantised to a coarse grid for the same reason.
        pos_warmup = int(np.ceil(sched_pos.n_warmup / 64) * 64)
        schedule_positive = SamplingSchedule(
            pos_warmup,
            self.n_samples_positive,
            self._pow2_ceil(sched_pos.steps_per_sample),
        )
        return (
            schedule_negative,
            sched_neg.n_warmup,
            schedule_positive,
            (info_neg, info_pos),
        )

    def run(self) -> float:
        def do_epoch_simplified(
            key,
            model,
            bsz_positive,
            bsz_negative,
            data_positive,
            neg_state,
            opt_state,
            cal_key,
            n_segments=6,
        ):
            def batch_data(key, data, _bsz, clamped_blocks):
                clamped_nodes = [node for block in clamped_blocks for node in block]
                data_size = data.shape[0]
                assert data.shape == (data_size, len(clamped_nodes))
                key, key_shuffle = jax.random.split(key)
                idxs = jax.random.permutation(key_shuffle, jnp.arange(data_size))
                data = data[idxs]
                _n_batches = data_size // _bsz
                tot_len = _n_batches * _bsz
                batched_data = jnp.reshape(
                    data[:tot_len], (_n_batches, _bsz, len(clamped_nodes))
                ).astype(jnp.bool)
                return batched_data, _n_batches

            key, key_pos = jax.random.split(key, 2)
            batched_data_pos, n_batches = batch_data(
                key_pos, data_positive, bsz_positive, self.training_data_blocks
            )

            def make_body(schedule_positive, schedule_negative):
                def body_fun(carry, key_and_data):
                    _key, _data_pos = key_and_data

                    _opt_state, _params, _neg_state = carry
                    _model = eqx.tree_at(
                        lambda m: (m.weights, m.biases), model, _params
                    )
                    key_train, key_init_pos = jax.random.split(_key, 2)
                    vals_free_pos = hinton_init(
                        key_init_pos,
                        _model,
                        self.positive_sampling_blocks,
                        (1, bsz_positive),
                    )

                    ebm = IsingTrainingSpec(
                        _model,
                        self.training_data_blocks,
                        [],
                        self.positive_sampling_blocks,
                        self.negative_sampling_blocks,
                        schedule_positive,
                        schedule_negative,
                    )

                    # Persistent chains (PCD): the negative chains continue from
                    # the previous step's final state instead of re-warming from
                    # a fresh hinton_init — they track the slowly moving model
                    # distribution, so the negative schedule carries no warmup.
                    grad_w, grad_b, _, _, _neg_state = estimate_kl_grad(
                        key_train,
                        ebm,
                        _model.nodes,
                        model.edges,
                        [_data_pos],
                        [],
                        vals_free_pos,
                        _neg_state,
                        return_negative_state=True,
                    )

                    grads = (grad_w, grad_b)
                    with jax.numpy_dtype_promotion("standard"):
                        updates, _opt_state = self.optim.update(
                            grads, _opt_state, _params
                        )

                    # estimate_kl_grad returns d(KL)/dθ:
                    #   -beta * (positive_phase - negative_phase).
                    # Optax transforms that gradient into an additive descent
                    # update, so apply_updates performs the correct minimization.
                    _params = optax.apply_updates(_params, updates)

                    return (_opt_state, _params, _neg_state), None

                return body_fun

            # Recalibrate the schedules every segment: the mixing time τ grows
            # as θ moves away from 0, so a schedule fixed at the start of the
            # epoch (τ ≈ 1, uniform model) badly under-thins the later steps.
            # Segments are equal-length so each distinct (quantised) schedule
            # compiles its scan body at most once.
            seg_size = n_batches // n_segments
            params = model.weights, model.biases
            cal_keys = jax.random.split(cal_key, n_segments)
            for seg in range(n_segments):
                seg_data = batched_data_pos[seg * seg_size : (seg + 1) * seg_size]
                _model = eqx.tree_at(lambda m: (m.weights, m.biases), model, params)
                schedule_negative, _, schedule_positive, _ = self._calibrate_schedules(
                    cal_keys[seg], _model
                )
                keys = jax.random.split(jax.random.fold_in(key, seg), seg_size)
                init_carry = opt_state, params, neg_state
                out_carry, _ = jax.lax.scan(
                    make_body(schedule_positive, schedule_negative),
                    init_carry,
                    (keys, seg_data),
                )
                opt_state, params, neg_state = out_carry

            new_model = eqx.tree_at(lambda m: (m.weights, m.biases), model, params)
            return new_model, opt_state, neg_state

        def compute_accuracy(
            key,
            model,
            bsz_per_digit,
        ):
            """Takes images separated into classes based on which digit they are and
            for each class computes the probability that the model assigns a 1 to the label
            of each digit. Based on this it computes the accuracy (the fraction of
            digits where the argmax of the output labels gives the correct digit)
            and records the average label probabilities for each digit.
            """
            program = IsingSamplingProgram(
                model,
                self.classification_sampling_blocks,
                self.classification_data_blocks,
            )

            # Calibrate the eval schedule at the *trained* parameters — the
            # mixing rate here bears no relation to the untrained model's.
            key, k_cal, k_ci = jax.random.split(key, 3)
            some_image = self.sep_images_test[self.target_classes[0]][0].astype(
                jnp.bool_
            )
            cal_init = hinton_init(
                k_ci,
                model,
                program.gibbs_spec.free_blocks,
                (self.calibration_replicas,),
            )
            accuracy_schedule, _ = tune_sampling_schedule(
                k_cal, model, program, cal_init, [some_image], target_ess=40
            )

            accuracy = 0.0
            for i, digit in enumerate(self.target_classes):
                images = self.sep_images_test[digit][:bsz_per_digit].astype(jnp.bool_)

                key, key_sample, key_init = jax.random.split(key, 3)

                init_free_states = hinton_init(
                    key_init,
                    model,
                    self.classification_sampling_blocks,
                    (bsz_per_digit,),
                )

                keys_samp = jax.random.split(key_sample, bsz_per_digit)

                samples = jax.vmap(
                    lambda k, init_state, data: sample_states(
                        k,
                        program,
                        accuracy_schedule,
                        init_state,
                        data,
                        [self.classification_label_block],
                    )
                )(keys_samp, init_free_states, [images])[0]

                labels = samples.reshape(
                    samples.shape[0],
                    samples.shape[1],
                    self.num_label_spots,
                    len(self.target_classes),
                )
                labels = jnp.mean(labels, axis=(1, 2))
                generated_digit = jnp.argmax(labels, axis=1)

                # Labels contain class indices, not the digit values themselves.
                accuracy += jnp.mean(i == generated_digit)

            return accuracy / len(self.target_classes)

        best_accuracy = 0.0
        opt_state = self.optim.init((self.init_model.weights, self.init_model.biases))

        model = self.init_model
        bsz_negative = 25
        neg_state = None

        for i in range(self.n_epochs):
            if neg_state is None:
                # Pre-warm the persistent negative chains once (at the current
                # θ) with the calibrated warmup; afterwards they persist across
                # steps and segments, tracking the model distribution, so the
                # per-step negative schedule carries no warmup.
                _, neg_warmup, _, _ = self._calibrate_schedules(
                    jax.random.key(100 + i), model
                )
                prog_neg = IsingSamplingProgram(
                    model, self.negative_sampling_blocks, []
                )
                k_i, k_w = jax.random.split(jax.random.key(50 + i))
                fresh = hinton_init(
                    k_i, model, prog_neg.gibbs_spec.free_blocks, (bsz_negative,)
                )
                warm_sched = SamplingSchedule(neg_warmup, 1, 1)
                keys_w = jax.random.split(k_w, bsz_negative)
                warmed = jax.vmap(
                    lambda k, init: sample_states(
                        k,
                        prog_neg,
                        warm_sched,
                        init,
                        [],
                        list(prog_neg.gibbs_spec.free_blocks),
                    )
                )(keys_w, fresh)
                neg_state = [blk[:, 0] for blk in warmed]

            model, opt_state, neg_state = do_epoch_simplified(
                jax.random.key(0),
                model,
                50,
                bsz_negative,
                self.train_data_filtered,
                neg_state,
                opt_state,
                jax.random.key(200 + i),
            )

            accuracy = compute_accuracy(
                jax.random.key(2),
                model,
                500,
            )
            best_accuracy = max(best_accuracy, accuracy)

        return float(best_accuracy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of training epochs (default: 1)"
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    best_accuracy = MnistTraining(n_epochs=args.epochs).run()
    wall = time.perf_counter() - t0

    passed = best_accuracy > _ACCURACY_THRESHOLD
    print(
        f"best accuracy={best_accuracy:.4f}  threshold={_ACCURACY_THRESHOLD}  "
        f"{'PASS' if passed else 'FAIL'}  | {wall:.1f}s",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
