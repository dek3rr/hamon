# Gaussian MRFs

Continuous-state EBMs with **exact** block Gibbs. The energy is quadratic, so
the target is a multivariate normal with precision `β·P` and mean `P⁻¹h`, and
the single-site conditionals are themselves Gaussian — within a color class
they are independent scalar Gaussians, so no linear solve is needed anywhere.

Positive definiteness of `P` is the caller's responsibility; strict diagonal
dominance is a simple sufficient condition. All interaction arrays are linear
in β, so NRPT's temperature-linear template mode applies bit-exactly.

Because an unbounded state space has no proper β = 0 member, `GaussianEBM`
reports `proper_at_beta_zero = False` and [`nrpt`][hamon.nrpt] rejects a ladder
starting at exactly β = 0. Either use `beta_range=(β_min > 0, 1.0)`, or anneal
from a proper reference with
[`AnnealedEBM`][hamon.models.AnnealedEBM].

::: hamon.models.GaussianEBM
    options:
        members:
            - __init__

::: hamon.models.GaussianSamplingProgram
    options:
        members:
            - __init__

::: hamon.models.gaussian_init

::: hamon.models.GaussianGibbsConditional

## Factors and interactions

::: hamon.models.QuadraticSelfInteraction

::: hamon.models.QuadraticPairInteraction

::: hamon.models.QuadraticSelfEBMFactor
    options:
        members:
            - __init__

::: hamon.models.QuadraticPairEBMFactor
    options:
        members:
            - __init__
