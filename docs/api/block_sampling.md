# Block Sampling

The sampling engine. `BlockSamplingProgram` wraps a model and its block
structure into a callable that runs Gibbs sweeps. `sample_states` is the
main entry point for collecting samples from a single chain.

::: hamon.BlockGibbsSpec
    options:
        members:
            - __init__

::: hamon.BlockSamplingProgram
    options:
        members:
            - __init__

::: hamon.SamplingSchedule
    options:
        members: false

::: hamon.sample_blocks

::: hamon.sample_single_block

::: hamon.sample_with_observation

::: hamon.sample_states

::: hamon.sample_states_batched
