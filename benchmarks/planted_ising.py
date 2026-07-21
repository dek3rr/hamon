#!/usr/bin/env python
"""Planted frustrated-loops Ising benchmark for hamon.

Generates a 2D periodic-lattice ±J Ising with a **certified** planted ground
state. hamon's energy is ``E(s) = -beta * (sum_i b_i s_i + sum_ij J_ij s_i s_j)``
(h = 0 here), so the ground state *maximizes* ``sum J_ij s_i s_j``. We add
frustrated loops (rectangle perimeters on the lattice); in each loop, under the
planted state, exactly one bond is antiferromagnetic and the rest ferromagnetic.
The planted state then maximizes every loop term simultaneously, hitting the
sum-of-loop-maxima upper bound -> it is a global maximizer of ``sum J s s``, i.e.
the certified ground state, with exactly known energy. A random gauge hides the
trivial all-up state. Hardness is tunable via the loop density.

This gives an exact success metric (did the sampler reach the planted energy?)
on a genuinely glassy, multimodal target -- unlike the easy random grids we have
benched -- which is exactly the regime that stresses the energy-variance seed's
ELE assumption.

Usage:
    python benchmarks/planted_ising.py --verify                  # brute-force GS check
    python benchmarks/planted_ising.py --L 16 --loops-per-spin 2 --samples 4000
    python benchmarks/planted_ising.py --L 16 --compare-seed     # seed_from_energy A/B
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from hamon import Block, SpinNode
from hamon.models.ising import IsingEBM, IsingSamplingProgram, hinton_init, ising_sample


@dataclass
class PlantedInstance:
    L: int
    nodes: list
    node_edges: list  # (SpinNode, SpinNode) pairs
    edges: np.ndarray  # (m, 2) int node indices
    weights: np.ndarray  # (m,) gauged couplings J
    biases: np.ndarray  # (n,) zeros
    free_blocks: list  # checkerboard Blocks (2 colors)
    planted_state: np.ndarray  # (n,) in {-1, +1}
    planted_energy: float
    n_loops: int


def _lattice(L):
    """Periodic L x L nearest-neighbor edges; returns (edges, emap, idx)."""

    def idx(r, c):
        return (r % L) * L + (c % L)

    edges, emap = [], {}
    for r in range(L):
        for c in range(L):
            for a, b in (((r, c), (r, c + 1)), ((r, c), (r + 1, c))):
                ia, ib = idx(*a), idx(*b)
                key = (min(ia, ib), max(ia, ib))
                if key not in emap:
                    emap[key] = len(edges)
                    edges.append(key)
    return edges, emap, idx


def _rect_perimeter(r0, c0, h, w, idx):
    """NN bonds around a rectangle perimeter (a simple cycle on the lattice)."""
    bonds = []
    for c in range(w):
        bonds.append((idx(r0, c0 + c), idx(r0, c0 + c + 1)))
        bonds.append((idx(r0 + h, c0 + c), idx(r0 + h, c0 + c + 1)))
    for r in range(h):
        bonds.append((idx(r0 + r, c0), idx(r0 + r + 1, c0)))
        bonds.append((idx(r0 + r, c0 + w), idx(r0 + r + 1, c0 + w)))
    return bonds


def make_planted_loops(L, loops_per_spin=2.0, seed=0, max_loop=None, gauge=True):
    assert L % 2 == 0, "use even L for a 2-colorable (bipartite) periodic lattice"
    rng = np.random.default_rng(seed)
    n = L * L
    edges, emap, idx = _lattice(L)
    m = len(edges)
    J = np.zeros(m)
    max_loop = max_loop or max(1, L // 2)
    n_loops = int(round(loops_per_spin * n))
    for _ in range(n_loops):
        r0, c0 = int(rng.integers(L)), int(rng.integers(L))
        h, w = int(rng.integers(1, max_loop + 1)), int(rng.integers(1, max_loop + 1))
        bonds = _rect_perimeter(r0, c0, h, w, idx)
        af = int(rng.integers(len(bonds)))  # the single antiferromagnetic bond
        for k, (a, b) in enumerate(bonds):
            J[emap[(min(a, b), max(a, b))]] += -1.0 if k == af else 1.0

    eta = rng.choice([-1.0, 1.0], size=n) if gauge else np.ones(n)
    ea = np.array([a for a, _ in edges])
    eb = np.array([b for _, b in edges])
    Jg = J * eta[ea] * eta[eb]  # gauge: planted GS becomes eta, energy unchanged
    planted_energy = float(-(Jg * eta[ea] * eta[eb]).sum())  # E = -sum J s_i s_j

    nodes = [SpinNode() for _ in range(n)]
    node_edges = [(nodes[a], nodes[b]) for a, b in edges]
    even = [nodes[r * L + c] for r in range(L) for c in range(L) if (r + c) % 2 == 0]
    odd = [nodes[r * L + c] for r in range(L) for c in range(L) if (r + c) % 2 == 1]
    return PlantedInstance(
        L,
        nodes,
        node_edges,
        np.array(edges),
        Jg,
        np.zeros(n),
        [Block(even), Block(odd)],
        eta,
        planted_energy,
        n_loops,
    )


def sample_energies(samples_bool, edges, weights):
    """E(s) = -sum J_ij s_i s_j for each sample (h = 0), s in {-1, +1}."""
    s = 2.0 * np.asarray(samples_bool).astype(np.float64) - 1.0
    return -(weights[None, :] * s[:, edges[:, 0]] * s[:, edges[:, 1]]).sum(1)


def brute_force_min(inst: PlantedInstance) -> float:
    """Exhaustive ground-state energy (small n only) — certifies the construction."""
    n = len(inst.nodes)
    ea, eb, w = inst.edges[:, 0], inst.edges[:, 1], inst.weights
    best = np.inf
    for bits in itertools.product((-1.0, 1.0), repeat=n):
        s = np.array(bits)
        best = min(best, float(-(w * s[ea] * s[eb]).sum()))
    return best


def _init_factory(inst):
    def f(n_chains, ebms, programs):
        fb = programs[0].gibbs_spec.free_blocks
        ks = jax.random.split(jax.random.key(7), n_chains)
        return [hinton_init(ks[c], ebms[0], fb, ()) for c in range(n_chains)]

    return f


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--loops-per-spin", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--device", default="gpu")
    p.add_argument(
        "--verify", action="store_true", help="brute-force GS check (small L)"
    )
    p.add_argument("--compare-seed", action="store_true", help="seed_from_energy A/B")
    args = p.parse_args()

    if args.verify:
        # L=4 (n=16) is the largest brute-force-feasible square; check several
        # densities/seeds to confirm the planted state is the certified GS.
        for lps in (1.0, 2.0, 4.0):
            for seed in (0, 1):
                inst = make_planted_loops(4, lps, seed=seed)
                bf = brute_force_min(inst)
                ok = abs(bf - inst.planted_energy) < 1e-9
                print(
                    f"L=4 n=16 loops={inst.n_loops:3d} seed={seed}: "
                    f"planted E*={inst.planted_energy:7.1f} brute-force min={bf:7.1f}  "
                    f"{'OK' if ok else 'MISMATCH!'}"
                )
        return 0

    inst = make_planted_loops(args.L, args.loops_per_spin, seed=args.seed)
    n = len(inst.nodes)
    print(
        f"planted Ising: {inst.L}x{inst.L}={n} spins, {len(inst.edges)} bonds, "
        f"{inst.n_loops} frustrated loops, planted E*={inst.planted_energy:.1f}\n",
        flush=True,
    )

    if args.compare_seed:
        # autotune now picks the seed-vs-pilot route itself, so the A/B lives
        # at the tune_chains level (where the knob remains). Discovery is
        # key-aligned to be identical across routes; only wall should differ.
        from hamon.tuning import tune_chains

        ebm = IsingEBM(
            inst.nodes,
            inst.node_edges,
            jnp.asarray(inst.biases),
            jnp.asarray(inst.weights),
            jnp.array(1.0),
        )
        program = IsingSamplingProgram(ebm, inst.free_blocks, [])
        for label, seed_energy in (("pilot ", False), ("energy", True)):
            t0 = time.perf_counter()
            disc = tune_chains(
                jax.random.key(1),
                None,
                None,
                _init_factory(inst),
                [],
                beta_range=(0.0, 1.0),
                gibbs_steps_per_round=4,
                max_chains=128,
                seed_from_energy=seed_energy,
                pad_probes=args.device != "cpu",
                ebm=ebm,
                program=program,
                device=args.device,
            )
            wall = time.perf_counter() - t0
            probes = [h["n"] for h in disc["history"]]
            print(
                f"[{label}] N={disc['n_chains']} Lambda={disc['Lambda']:.2f} "
                f"probes={probes} | {wall:.1f}s",
                flush=True,
            )
        return 0

    t0 = time.perf_counter()
    samples, diag = ising_sample(
        jnp.asarray(inst.biases),
        jnp.asarray(inst.edges),
        jnp.asarray(inst.weights),
        key=jax.random.key(1),
        beta=1.0,
        n_samples=args.samples,
        device=args.device,
    )
    wall = time.perf_counter() - t0
    E = sample_energies(samples, inst.edges, inst.weights)
    gap = float(E.min()) - inst.planted_energy
    print(
        f"autotune: N={diag['n_chains']} n_expl={diag['gibbs_steps_per_round']} "
        f"Lambda={float(diag['Lambda']):.2f} colors={diag.get('n_colors', '?')}\n"
        f"E_min={E.min():.1f}  planted E*={inst.planted_energy:.1f}  "
        f"gap={gap:.1f} ({'GROUND STATE FOUND' if gap < 1e-6 else f'{gap / n:.3f}/spin above'})\n"
        f"GS hit-rate={np.mean(np.abs(E - inst.planted_energy) < 1e-6):.4f}  "
        f"mean residual/spin={(E.mean() - inst.planted_energy) / n:.3f}  | {wall:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
