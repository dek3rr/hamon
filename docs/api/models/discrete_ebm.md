# Discrete EBM

Concrete discrete energy-based model implementations with spin and
categorical factor types.

::: hamon.models.DiscreteEBMInteraction

::: hamon.models.DiscreteEBMFactor

::: hamon.models.SquareDiscreteEBMFactor

::: hamon.models.SpinEBMFactor

::: hamon.models.CategoricalEBMFactor

::: hamon.models.SquareCategoricalEBMFactor

## Conditional samplers

The exact single-site conditionals for discrete states, vectorized over a color
class. `SpinGibbsConditional` is Bernoulli; `CategoricalGibbsConditional` is
softmax over `n_categories`.

::: hamon.models.SpinGibbsConditional

::: hamon.models.CategoricalGibbsConditional
