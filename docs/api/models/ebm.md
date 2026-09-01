# Energy-Based Models

Base classes for energy-based models. `AbstractEBM` defines the interface;
`AbstractFactorizedEBM` adds factor-sum energies. `AnnealedEBM` implements the
standard parallel-tempering path between a *reference* and a *target*.

Two properties on `AbstractEBM` drive how NRPT may temper a model.
`proper_at_beta_zero` says whether the β = 0 member is a proper distribution —
`False` for unbounded state spaces, which is why a ladder starting at exactly
β = 0 is rejected for the continuous families. `beta_affine` says whether
interactions interpolate as `offset + β·slope` rather than scaling linearly with
β, which is what `AnnealedEBM` needs.

::: hamon.models.AbstractEBM
    options:
        members:
            - energy

::: hamon.models.AbstractFactorizedEBM
    options:
        members:
            - __init__

::: hamon.models.FactorizedEBM
    options:
        members:
            - __init__

::: hamon.models.EBMFactor

## Reference annealing

`AnnealedEBM(reference, target, β)` implements `E_β = (1−β)·E_ref + β·E_target`,
whose β = 0 member is the *reference at full weight*. Every rung of the ladder
is then proper, so the ladder can cover the full entropic path even when the
target alone has no proper β = 0 member. NRPT handles this with an affine
interpolation and swap energies `Δ = E₁ − E₀` — the shared reference term
cancels in every swap ratio.

::: hamon.models.AnnealedEBM
    options:
        members:
            - __init__
