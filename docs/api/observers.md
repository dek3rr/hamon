# Observers

Observers collect statistics during sampling. `StateObserver` records raw
states; `MomentAccumulatorObserver` computes running means and variances
without storing every sample.

::: hamon.AbstractObserver
    options:
        members:
            - init

::: hamon.StateObserver
    options:
        members:
            - __init__

::: hamon.MomentAccumulatorObserver
    options:
        members:
            - __init__

## NRPT observers

Observers called once per NRPT round (after the Gibbs sweeps and swaps).
`NRPTStateObserver` records chain states; `NRPTEnergyObserver` accumulates the
per-chain mean energy used for the log normalizing constant
([`thermodynamic_integration`][hamon.thermodynamic_integration]).

::: hamon.AbstractNRPTObserver
    options:
        members:
            - init

::: hamon.NRPTStateObserver
    options:
        members:
            - __init__

::: hamon.NRPTEnergyObserver
    options:
        members:
            - __init__

`ColdChainObserver` records only the coldest chain, which is what makes a
*tempered* draw cheap: the tuned ladder keeps running so DEO swaps keep carrying
barrier crossings into the samples, but only the target-β chain is stored.

::: hamon.ColdChainObserver
    options:
        members:
            - __init__

::: hamon.nrpt_node_samples
