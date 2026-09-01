# Round-Trip Diagnostics

Functions for measuring tempering performance: round-trip rates, the
communication barrier Λ, chain count recommendations, and the log
normalizing constant via thermodynamic integration.

Before trusting a Λ̂ or a round-trip rate, check the two-part trust gate.
`barrier_is_identified` asks the *structural* question — does the ladder
saturate? — so that "identified" means Λ̂ is within ~10% regardless of the round
budget. `conveyor_is_alive` separately answers the *dynamical* question, and
reports `None` (unmeasured, not stalled) when the window affords too few
expected trips to tell.

::: hamon.round_trip_summary

::: hamon.recommend_n_chains

::: hamon.thermodynamic_integration

::: hamon.nrpt_log_normalizing_constant

## Trust gates

::: hamon.round_trips.barrier_is_identified

::: hamon.round_trips.conveyor_is_alive
