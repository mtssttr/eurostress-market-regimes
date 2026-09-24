# Research Protocol v1.1

## Regime-Aware European Market Risk Monitor

**Status:** Frozen after source-coverage audit and before return/market-outcome exploration  
**Protocol date:** 23 September 2026  
**Confirmatory sample end:** 31 December 2025

### Research question

Can a point-in-time hidden-state model built from European equity, foreign-exchange, and interest-rate data forecast next-week European equity risk more accurately than simple volatility benchmarks?

The purpose is risk monitoring, not market-timing or return prediction. The primary comparison is between a cross-asset Hidden Markov Model and fixed EWMA volatility.

### Hypotheses and chronological split

The primary hypothesis is that the cross-asset HMM produces lower locked-test QLIKE loss for next-week EURO STOXX 50 realised variance than EWMA. Secondary comparisons test whether cross-asset features improve on a volatility-only HMM and whether filtered stress probabilities agree with the ECB New CISS index.

- Overall sample: 30 March 2007–31 December 2025.
- Training: 30 March 2007–31 December 2015.
- Validation: 1 January 2016–31 December 2019.
- Locked test: 1 January 2020–31 December 2025.
- Raw data are daily; models use Friday weekly observations.
- Randomly shuffled time-series validation is prohibited.

### Predetermined features

1. EURO STOXX 50 one-week log return.
2. Log 20-trading-day realised equity volatility.
3. EUR/USD one-week log return.
4. Weekly AAA two-year yield change in basis points.
5. Weekly change in the AAA 10-year-minus-two-year slope.
6. Weekly change in the all-ratings-minus-AAA 10-year fragmentation spread.

Levels are sampled before changes are calculated. Values stale by more than five calendar days are excluded. Scaling is estimated using training data only. New CISS is external validation, not a feature.

### Models and selection

Benchmarks are historical volatility, EWMA with daily `lambda = 0.94`, Gaussian Mixture Models, and a volatility-only HMM. Candidate cross-asset Gaussian HMMs use two to four states, diagonal covariance, and 20 deterministic initialisations. A candidate is inadmissible if a state has less than 5% validation occupancy, median duration below two weeks, non-convergence, or material seed instability. Validation predictive log likelihood selects among admissible models, with ties resolved in favour of fewer states.

Only filtered probabilities may be used for forecasting. PCA is optional and train-only; it is not part of the core specification.

### Evaluation and failure rules

The primary outcome is next-week realised variance. The primary effect is `QLIKE_EWMA - QLIKE_HMM`; positive values favour the HMM. Confirmatory outperformance requires a positive mean with a 95% moving-block-bootstrap interval excluding zero.

Secondary outputs include Brier score and precision-recall AUC for returns below the training fifth percentile, predictive likelihood, state duration, seed sensitivity, feature ablations, and false-positive/missed-shock analysis. The point-in-time shock score is `-log p(x_t | x_1,...,x_{t-1})`, with its threshold frozen at the validation 99th percentile.

If the HMM does not beat EWMA, that negative result will be reported. If the cross-asset HMM does not beat the volatility-only version, the extra inputs will be judged non-incremental. Historical event matches do not prove crisis prediction. The core study makes no causal ECB claim and reports no trading strategy, Sharpe ratio, or profitability claim.

### Amendment policy

Every amendment must be dated, justified, and classified according to whether validation or test outcomes had already been viewed. Version 1.0 remains preserved. Confirmatory claims may use only decisions frozen before the locked test is scored.
