# EuroStress portfolio interview guide

## Thirty-second project summary

I built a point-in-time European market-risk research pipeline using weekly equity, FX, yield-curve and sovereign-fragmentation data from 2007 to 2025. I compared HMM-based regime forecasts with historical variance and EWMA, used chronological validation and seed-stability rules, and opened a sealed six-year test once. EWMA remained the best next-week variance forecaster, while the exploratory regime probability correlated 0.75 with ECB New CISS. The useful conclusion was therefore stress monitoring rather than superior forecasting or crash prediction.

## CV bullets

### Quant research or systematic-strategy version

- Built a point-in-time European cross-asset risk pipeline across 979 weekly observations, with train-only transformations, chronological validation and a sealed 312-week test.
- Evaluated GMM, HMM, PCA and qualified-seed ensemble specifications against historical-variance and EWMA benchmarks using QLIKE and block-bootstrap uncertainty.
- Found fixed EWMA remained the strongest variance forecast; retained the negative result and showed that exploratory regime stress correlated 0.75 with ECB New CISS.

### Market-risk or risk-analytics version

- Developed an auditable European stress monitor combining equity, FX, yield-curve and sovereign-fragmentation signals with explicit information-date controls.
- Implemented model-admissibility, state-occupancy, duration and seed-stability diagnostics before a one-time 2020–2025 locked evaluation.
- Demonstrated that latent regimes described contemporaneous systemic stress more reliably than they predicted next-week variance or tail events.

### Short one-line version

- Built and locked-tested a European cross-asset HMM risk monitor; EWMA won the forecast comparison, while regime stress correlated 0.75 with ECB New CISS.

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

### What would you do next?

I would keep the locked result unchanged and start a new protocol. The next study could test a simpler dynamic-factor or supervised volatility model, add carefully lagged option-implied information, and evaluate calibration and decision value in a new future sample. I would not retune the current HMM on the test period.

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

## Claims to avoid

- “The HMM predicts crises.”
- “The model beat EWMA.”
- “The New CISS correlation proves predictive power.”
- “The alerts form a profitable trading signal.”
- “The exploratory branches are confirmatory because they were frozen before test opening.”

The strongest defensible framing is that EuroStress is a reproducible market-risk research pipeline with an honest negative forecasting result and a useful contemporaneous stress signal.
