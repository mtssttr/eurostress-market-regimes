# EuroStress portfolio interview guide

## Thirty-second project summary

I built a European financial-data and market-risk research pipeline. Simple EWMA remained the strongest weekly volatility benchmark. I then tested daily tail-risk calibration and added an exploratory comparison separating distribution choice from parameter updating. Monthly Student t GARCH reduced breaches against a matched Gaussian comparator, but still failed correct coverage. Blocking input controls, synthetic investigations and independent numerical checks support the reliability of the analysis.

See [application positioning](application_positioning.md) for Operations and Quantitative Strats wording and [the exploratory report](../reports/tail_extension/report.md) for the new comparison. The original daily-model statistics below remain a separate result set.

## CV bullets

### Quant research or systematic-strategy version

- Built a point-in-time European cross-asset risk pipeline across 974 complete weekly observations, with train-only transformations, chronological validation and a sealed 312-week test.
- Evaluated GMM, HMM, PCA and qualified-seed ensemble specifications against historical-variance and EWMA benchmarks using QLIKE and block-bootstrap uncertainty.
- Found fixed EWMA remained the strongest variance forecast; retained the negative result and showed that exploratory regime stress correlated 0.75 with ECB New CISS.

### Market-risk or risk-analytics version

- Built and backtested historical, Gaussian, EWMA and GARCH(1,1) one-day VaR/Expected Shortfall models across 1,510 EURO STOXX 50 returns.
- Implemented Kupiec coverage, Christoffersen independence and conditional-coverage tests; all models exceeded the intended 99% exception rate, while GARCH avoided significant exception clustering.
- Connected tail-risk failures to an existing European stress-regime monitor and compared coverage during COVID, the 2022 repricing and March 2023 banking stress.

### Short one-line version

- Built a European market-risk validation pipeline spanning HMM regimes, GARCH volatility, formal VaR backtests and descriptive ES diagnostics; simple models remained difficult to beat and all original 99% VaR specifications under-covered realised losses.

## Interview talking points

### Why this project?

The project connects my physics background with financial modelling: latent states resemble hidden physical regimes, but financial data require much stricter attention to timing, non-stationarity and benchmark discipline. I wanted to show that I could formulate and falsify a hypothesis rather than optimise a chart after seeing the answer.

### What was the hypothesis?

The primary question was whether cross-asset latent regimes improved next-week realised-variance QLIKE over a fixed EWMA forecast. The regime model was not assumed to succeed. Stress alignment and tail-event metrics were secondary, and the PCA and ensemble branches remained exploratory.

### Why QLIKE?

QLIKE is appropriate for non-negative variance forecasts and penalises economically important forecast errors without requiring squared-error comparability. More importantly, it was frozen before test opening and applied to every model consistently.

### Why is EWMA a serious benchmark?

EWMA is simple, transparent and highly adaptive to recent volatility. A more complex model only adds value if it improves on that baseline out of sample after accounting for instability and estimation error. In this test, it did not.

### What failed?

The original cross-asset HMM did not pass all pre-specified stability rules. The exploratory PCA and seed ensemble improved robustness but still lost to EWMA. A strict shock score also generated four false positives and missed all seven defined next-week tail events.

### What worked?

The data lineage, leakage controls, deterministic fitting, seed diagnostics and sealed evaluation all worked as designed. The exploratory high-risk-state probability also aligned strongly with ECB New CISS, suggesting that the latent state was useful as a descriptive stress monitor.

### Why add VaR and Expected Shortfall?

The regime study established statistical research skill but did not directly answer a market-risk validation question. The extension gives the regime label an operational use: I can ask whether standard loss limits have correct coverage, whether exceptions cluster, and whether failures differ across lower- and higher-risk states. I kept it post-hoc and did not rewrite the original sealed result.

### What did the VaR tests show?

At 99%, the models produced 26 to 39 exceptions instead of the 15.1 expected. Kupiec therefore rejected correct unconditional coverage for every model. Historical simulation, rolling Gaussian and EWMA exceptions were also significantly clustered. GARCH still had too many exceptions, but its Christoffersen independence p-value was 0.794, so the remaining failures were not detectably clustered at first order.

### Why could GARCH pass independence but fail coverage?

The tests answer different questions. Kupiec asks whether the total exception frequency is correct. Christoffersen asks whether one exception makes another immediately more likely. GARCH's exception count was excessive without detectable first-order clustering. These tests alone do not identify the cause: distributional misspecification, parameter estimation and changing market dynamics can all matter. The later matched extension investigates distribution choice and parameter updating separately. Non-rejection of independence is not proof of independent exceptions.

### How did you evaluate Expected Shortfall?

I reported 97.5% ES, matching the Basel one-tailed confidence level, and compared realised exception losses with forecast ES. Every model had a severity ratio above one, meaning realised tail losses were larger on average. I call this a diagnostic rather than a formal ES backtest because rigorous ES validation should use a joint VaR/ES testing framework.

### What would you do next?

I have now completed the Student t and monthly-refitting extension, preserving the original results. The next research priorities would be a genuinely untouched prospective evaluation and a formal joint VaR/ES backtest. A broader asset sample would help assess generality. Filtered historical simulation could be another predetermined comparison; EVT would require checking threshold stability and retaining enough tail observations. I would not retune the HMM to make the old test look better.

### How did your physics training help?

Physics training helped with model abstraction, stochastic processes, numerical diagnostics and the discipline of distinguishing measurement from interpretation. The most transferable skill was treating failure rules and uncertainty as part of the model, not as presentation details.

## Questions to expect

1. How did you prevent leakage at the Friday cut-off?
2. Why did the original cross-asset model fail its admissibility rules?
3. How were states aligned across random seeds?
4. Why was PCA considered exploratory?
5. What does the block bootstrap account for?
6. Why can a model correlate with CISS yet forecast variance poorly?
7. What would make the monitor operationally useful?
8. Which conclusion would you refuse to make from these results?
9. What is the difference between coverage and independence?
10. Why does Expected Shortfall need a different validation framework from VaR?
11. Why fit GARCH only through 2019 rather than re-estimating with test outcomes?
12. Why can a contemporaneous weekly regime label miss an abrupt daily shock?

## Claims to avoid

- “The HMM predicts crises.”
- “The model beat EWMA.”
- “The New CISS correlation proves predictive power.”
- “The alerts form a profitable trading signal.”
- “The exploratory branches are confirmatory because they were frozen before test opening.”

The strongest defensible framing is that EuroStress is a reproducible market-risk and model-validation pipeline with two honest findings: HMM complexity did not beat EWMA, and conventional Gaussian tail-risk models did not achieve their advertised 99% coverage.
