# Ising Models

Ising-specific model, program, and training utilities.

For most problems `ising_sample` is the entry point: it goes from
`(biases, edges, weights)` to samples in one call, building the coloring,
autotuning the ladder, and drawing from the target. `IsingEBM` is the
lower-level construction — the factor graph from a list of nodes, edges,
biases, and coupling weights — for when you want to drive the sampler yourself.

## Front door

::: hamon.ising_sample

### Choosing β from the model

`beta="auto"` reads the coldest useful temperature off the model's own
excitation-cost spectrum instead of guessing. Both halves of that estimator are
available on their own.

::: hamon.ising_estimate_beta

::: hamon.ising_excitation_costs

## Model construction

::: hamon.models.IsingEBM
    options:
        members:
            - __init__

::: hamon.models.IsingSamplingProgram
    options:
        members:
            - __init__

::: hamon.models.hinton_init

## Training

`estimate_kl_grad` computes the contrastive-divergence gradient of the KL
objective — the positive phase clamped to data, the negative phase free — for an
`IsingTrainingSpec` that pairs the model with its two sampling programs. Pass
`return_negative_state=True` and feed the returned state into the next step for
persistent contrastive divergence.

::: hamon.models.IsingTrainingSpec
    options:
        members:
            - __init__

::: hamon.models.estimate_moments

::: hamon.models.estimate_kl_grad
