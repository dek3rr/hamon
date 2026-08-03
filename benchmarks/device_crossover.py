#!/usr/bin/env python
"""Measure where the CPU/GPU crossover sits for hamon's NRPT workloads.

Sweeps lattice size x chain count, timing the same `nrpt` run on the CPU and
the GPU (compile time and steady-state separated), then recommends a
HAMON_DEVICE_THRESHOLD for this machine: hamon's "auto" routing sends a
workload to the accelerator when `n_chains x n_free_nodes` meets the
threshold.

    python benchmarks/device_crossover.py
    python benchmarks/device_crossover.py --sizes 8,16,32,64 --chains 4,16
    python benchmarks/device_crossover.py --portfolio   # ~500-node calibration flow

Exits cleanly with a message when no GPU is visible.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from hamon import Block, nrpt, tune_chains, tune_schedule
from hamon.device import accelerator_device
from hamon.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
from hamon.pgm import SpinNode


def make_grid(L: int, coupling: float = 0.5, seed: int = 0):
    """L x L periodic-free 2D Ising lattice with checkerboard blocking."""
    rng = np.random.default_rng(seed)
    grid = [[SpinNode() for _ in range(L)] for _ in range(L)]
    nodes = [n for row in grid for n in row]
    edges = []
    for i in range(L):
        for j in range(L):
            if j + 1 < L:
                edges.append((grid[i][j], grid[i][j + 1]))
            if i + 1 < L:
                edges.append((grid[i][j], grid[i + 1][j]))
    biases = jnp.asarray(rng.normal(0.0, 0.1, len(nodes)).astype(np.float32))
    weights = jnp.asarray(rng.normal(0.0, coupling, len(edges)).astype(np.float32))
    even = [grid[i][j] for i in range(L) for j in range(L) if (i + j) % 2 == 0]
    odd = [grid[i][j] for i in range(L) for j in range(L) if (i + j) % 2 == 1]
    free_blocks = [Block(even), Block(odd)]
    ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
    program = IsingSamplingProgram(ebm, free_blocks, [])
    return ebm, program, free_blocks


def time_nrpt(device, ebm, program, free_blocks, n_chains, rounds, gibbs, repeats):
    """Return (compile_seconds, steady_seconds) for one configuration."""
    betas = jnp.linspace(0.1, 1.0, n_chains)
    keys = jax.random.split(jax.random.key(0), n_chains)
    inits = [hinton_init(keys[c], ebm, free_blocks, ()) for c in range(n_chains)]

    def run(k):
        states, _ = nrpt(
            k,
            ebm,
            program,
            inits,
            [],
            rounds,
            gibbs,
            betas=betas,
            device=device,
            track_round_trips=False,
        )
        jax.block_until_ready(states[0][0])

    t0 = time.perf_counter()
    run(jax.random.key(1))
    first = time.perf_counter() - t0
    times = []
    for r in range(repeats):
        t0 = time.perf_counter()
        run(jax.random.key(2 + r))
        times.append(time.perf_counter() - t0)
    steady = min(times)
    return max(first - steady, 0.0), steady


def recommend_threshold(rows) -> int | None:
    """Smallest power of two separating the CPU-wins from the GPU-wins band."""
    cpu_wins = [r["score"] for r in rows if r["gpu_steady"] >= r["cpu_steady"]]
    gpu_wins = [r["score"] for r in rows if r["gpu_steady"] < r["cpu_steady"]]
    if not gpu_wins:
        return None
    low = max(cpu_wins) if cpu_wins else min(gpu_wins) / 2
    high = (
        min(s for s in gpu_wins if s > low) if any(s > low for s in gpu_wins) else low
    )
    geo = math.sqrt(low * high) if low > 0 else high
    return 2 ** max(round(math.log2(geo)), 0)


def sweep(args) -> list[dict]:
    rows = []
    for L in args.sizes:
        ebm, program, free_blocks = make_grid(L)
        for n_chains in args.chains:
            row = {"L": L, "n_chains": n_chains, "score": n_chains * L * L}
            for device in ("cpu", "gpu"):
                compile_s, steady_s = time_nrpt(
                    device,
                    ebm,
                    program,
                    free_blocks,
                    n_chains,
                    args.rounds,
                    args.gibbs_steps,
                    args.repeats,
                )
                row[f"{device}_compile"] = compile_s
                row[f"{device}_steady"] = steady_s
            row["speedup"] = row["cpu_steady"] / row["gpu_steady"]
            rows.append(row)
            print(
                f"L={L:4d} chains={n_chains:3d} score={row['score']:8d}  "
                f"cpu={row['cpu_steady']:.4f}s gpu={row['gpu_steady']:.4f}s  "
                f"gpu speedup={row['speedup']:.2f}x  "
                f"(compile cpu={row['cpu_compile']:.2f}s gpu={row['gpu_compile']:.2f}s)",
                flush=True,
            )
    return rows


def portfolio(args) -> None:
    """Time the discover + adaptive calibration flow on a ~500-node model."""
    import networkx as nx

    n = 500
    g = nx.erdos_renyi_graph(n, 0.02, seed=0)
    nodes = [SpinNode() for _ in range(n)]
    edges = [(nodes[a], nodes[b]) for a, b in g.edges()]
    rng = np.random.default_rng(0)
    biases = jnp.asarray(rng.normal(0.0, 0.2, n).astype(np.float32))
    weights = jnp.asarray(rng.normal(0.0, 0.3, len(edges)).astype(np.float32))

    coloring = nx.coloring.greedy_color(g, strategy="DSATUR")
    n_colors = max(coloring.values()) + 1
    groups: list[list[SpinNode]] = [[] for _ in range(n_colors)]
    for idx, color in coloring.items():
        groups[color].append(nodes[idx])
    free_blocks = [Block(group) for group in groups]

    ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
    program = IsingSamplingProgram(ebm, free_blocks, [])

    def init_factory(n_chains, ebms, programs):
        keys = jax.random.split(jax.random.key(7), n_chains)
        return [hinton_init(keys[c], ebms[0], free_blocks, ()) for c in range(n_chains)]

    def flow(device, key):
        discovery = tune_chains(
            key,
            ebm=ebm,
            program=program,
            init_factory=init_factory,
            beta_range=(0.1, 1.0),
            gibbs_steps_per_round=args.gibbs_steps,
            rounds_per_probe=args.rounds,
            max_iters=3,
            device=device,
        )
        betas = discovery["betas"]
        inits = init_factory(len(betas), [ebm], [program])
        states, _ = tune_schedule(
            jax.random.key(11),
            ebm=ebm,
            program=program,
            init_states=inits,
            clamp_state=[],
            n_rounds=2 * args.rounds,
            gibbs_steps_per_round=args.gibbs_steps,
            initial_betas=betas,
            n_tune=2,
            rounds_per_tune=args.rounds,
            device=device,
        )
        jax.block_until_ready(states[-1][0])
        return discovery["n_chains"]

    print(
        f"\nportfolio flow: {n} nodes, {len(edges)} edges, {args.rounds} rounds/probe ({n_colors} colors)"
    )
    for device in ("cpu", "gpu"):
        t0 = time.perf_counter()
        n_chains = flow(device, jax.random.key(3))
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        flow(device, jax.random.key(4))
        warm = time.perf_counter() - t0
        score = n_chains * n
        print(
            f"  {device}: cold={cold:.1f}s warm={warm:.1f}s (discovered {n_chains} chains, production score={score})",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", default="8,16,32,64,128", help="comma-separated lattice side lengths"
    )
    parser.add_argument(
        "--chains", default="4,8,16,32", help="comma-separated chain counts"
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--gibbs-steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--csv", default=None, help="write results to CSV")
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="also time a ~500-node calibration flow",
    )
    args = parser.parse_args()
    args.sizes = [int(x) for x in str(args.sizes).split(",")]
    args.chains = [int(x) for x in str(args.chains).split(",")]

    if accelerator_device() is None:
        print(
            "No GPU/TPU visible to JAX — nothing to calibrate; hamon's "
            "'auto' routing is already a no-op on this machine."
        )
        return 0

    print(f"jax {jax.__version__}, accelerator: {accelerator_device()}\n")
    rows = sweep(args)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")

    threshold = recommend_threshold(rows)
    if threshold is None:
        print(
            "\nGPU never beat CPU in this sweep — leave routing at its default (or raise HAMON_DEVICE_THRESHOLD)."
        )
    else:
        print(f"\nRecommended threshold for this machine: {threshold}")
        print(f"  bash:       export HAMON_DEVICE_THRESHOLD={threshold}")
        print(f"  PowerShell: $env:HAMON_DEVICE_THRESHOLD = '{threshold}'")

    if args.portfolio:
        portfolio(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
