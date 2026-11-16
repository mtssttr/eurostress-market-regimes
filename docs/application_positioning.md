# Application wording and technical ownership

Use these descriptions only to the extent you can explain, modify and reproduce the work.
This is an AI-assisted independent educational project, not professional banking experience.
The 2026 extension is exploratory and is not a fresh confirmatory test.

## Operations emphasis

**EuroStress - Financial Data Controls and Risk Reporting**

- Built a Python/DuckDB workflow for European financial data with source hashes,
  chronological alignment and blocking input-integrity controls.
- Implemented an offline demonstration detecting and resolving four synthetic input
  problems: duplicate observations, missing values, stale data and incorrect unit metadata.
- Produced machine-readable investigation records, documented corrections and reran
  calculations only after input controls passed.

Interview introduction: "I wanted to understand how an unreliable input can affect a
financial calculation. I built a reproducible risk-analysis workflow and a separate
synthetic demonstration that blocks calculations when input controls fail. Each
correction is traceable to a source of truth. That taught me to distinguish a data
problem from a model problem and to document both."

## Quantitative Strats emphasis

**EuroStress - Volatility Forecasting and Market-Risk Validation**

- Evaluated volatility forecasts across 312 held-out weeks and four one-day risk models
  across 1,510 observations, using chronological information cut-offs and simple benchmarks.
- Implemented VaR coverage and exception-independence tests, with independent numerical
  checks of tail formulas, likelihood statistics and the GARCH variance recursion.
- Extended the study to Gaussian/Student t GARCH with fixed/monthly refitting. Monthly
  Student t reduced 99% VaR breaches from 37 to 25 against the matched fixed Gaussian
  comparator, but still rejected correct coverage (p=0.019).

Interview introduction: "I tested whether regime models improved volatility forecasts.
The simple EWMA benchmark remained strongest. I then investigated tail-risk calibration,
separating distribution choice from parameter updating in a four-specification GARCH
comparison. Student t with monthly refitting reduced quantile loss, but did not solve
undercoverage. I validated key calculations independently before interpreting that result."

## Prepare to demonstrate

1. Rebuild a synthetic control case without assistance and explain why the calculation stops.
2. Derive a one-step GARCH variance forecast and explain percent-squared versus decimal units.
3. Explain the unit-variance scaling of Student t errors and the condition nu > 2.
4. Explain why a non-significant independence test does not establish correct coverage.
5. Explain why fewer breaches alone do not establish a better quantile forecast.
6. Explain why refits exclude the day's return and why this extension is exploratory.

## Boundaries of the claim

- Synthetic corrections do not establish financial loss prevented, real client impact,
  or professional operations experience. Do not invent time-saving percentages.
- Declared-unit checks verify metadata consistency, not the economic correctness of
  a provider's numbers. Original local manifests lack unit metadata; six explicit
  warnings remain. They have not been silently backfilled or called verified.
- Dates are aligned to the forecast information cut-off. Historical revision vintages
  for every downloaded ECB series have not been independently reconstructed.
- Numerical reference checks validate selected calculations. They do not certify the
  entire pipeline or turn a retrospective study into a production banking system.
- The 37-versus-25 result belongs to the new matched comparison; the original Gaussian
  GARCH used a different estimation window and recorded 34 breaches.
- No causal regime, trading-profitability or formal ES-validation claim is supported.
