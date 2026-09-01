# NRPT

The primary interface is **autotuning**: [`autotune`][hamon.autotune] /
[`autosample`][hamon.autosample] discover the chain count, local-exploration
count, and schedule together, then draw from the target. The building-block
tuners and the core single run are below them.

## Autotuning

::: hamon.autosample

::: hamon.autotune

::: hamon.NRPTPlan
    options:
        members:
            - sample
            - extend
            - sample_until

::: hamon.AutotuneReport
    options:
        members:
            - summary

## Building blocks

::: hamon.nrpt

::: hamon.tune_schedule

::: hamon.optimize_schedule

::: hamon.tune_chains

::: hamon.tune_exploration

::: hamon.tune_sampling_schedule
