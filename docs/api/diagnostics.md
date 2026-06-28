# Diagnostics

Host-side diagnostics for sample quality and NRPT run health. These run in
numpy (no XLA compile) over the collected samples and the stats dict returned
by [`hamon.nrpt`][] / [`hamon.tune_schedule`][].

::: hamon.report_nrpt_diagnostics

::: hamon.NRPTHealthReport
    options:
        members:
            - summary

::: hamon.effective_sample_size

::: hamon.ESSReport

::: hamon.diagnostics.sample_convergence

::: hamon.diagnostics.marginal_entropy

::: hamon.diagnostics.energy_balance
