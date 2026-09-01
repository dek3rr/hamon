# Double-Well (φ⁴) Models

The continuous *multimodal* case — the one tempering exists for. `DoubleWellEBM`
is the lattice φ⁴ field: at cold β with ferromagnetic couplings the target is
bimodal, a single chain mode-collapses into one well, and NRPT round trips are
what carry the ± flips into the samples.

The single-site conditional has no closed form, so
[`SliceGibbsConditional`][hamon.models.SliceGibbsConditional] performs one
slice-sampling transition per site (Neal 2003, the exactly-reversible bounded
variant), vectorized over each color class. Slice draws are keyed by iteration
so that chain masking stays bit-identical despite data-dependent loop lengths.

Like [`GaussianEBM`][hamon.models.GaussianEBM], an unbounded state space has no
proper β = 0 member, so `proper_at_beta_zero = False`; pair the target with
[`AnnealedEBM`][hamon.models.AnnealedEBM] to temper from exactly β = 0.

::: hamon.models.DoubleWellEBM
    options:
        members:
            - __init__

::: hamon.models.DoubleWellSamplingProgram
    options:
        members:
            - __init__

::: hamon.models.double_well_init

::: hamon.models.SliceGibbsConditional

## Factors and interactions

::: hamon.models.PolynomialSelfInteraction

::: hamon.models.PolynomialSelfEBMFactor
    options:
        members:
            - __init__
