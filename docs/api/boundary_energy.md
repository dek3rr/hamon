# Boundary Energy

Incremental energy-delta computation for Ising models. After a block update,
ΔE depends only on the edges incident to that block, so the swap energies in
[`nrpt`][hamon.nrpt] can be advanced by a delta instead of recomputed in full —
pass the factory's result as `nrpt`'s `energy_delta_fn`.

::: hamon.make_ising_delta_fn
