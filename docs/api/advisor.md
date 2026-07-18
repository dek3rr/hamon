# Search advisor

Ground-state-search guidance: choose the coldest useful β **before** tuning
(one autotune, no raise-and-retry loops), and diagnose **after** a draw which
knob — mixing, draw budget, or β — limits the minimum found. Host-side numpy;
no XLA compiles. Ising front ends: [`hamon.ising_estimate_beta`][] and
`ising_sample(..., beta="auto")`; continuation:
[`hamon.NRPTPlan.extend`][hamon.NRPTPlan] and `NRPTPlan.sample_until`.

::: hamon.estimate_beta_max

::: hamon.BetaEstimate
    options:
        members:
            - summary

::: hamon.diagnose_search

::: hamon.SearchAdvice
    options:
        members:
            - summary

::: hamon.SearchVerdict

::: hamon.advisor.excess_energy

::: hamon.advisor.gs_occupancy

::: hamon.advisor.communication_barrier
