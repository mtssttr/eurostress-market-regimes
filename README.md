# EuroStress

**A point-in-time test of whether latent European cross-asset regimes improve next-week equity-risk forecasts.**

EuroStress is an end-to-end quantitative research project built on weekly European equity, foreign-exchange, yield-curve, sovereign-fragmentation and ECB stress data from 2007 to 2025. It compares Gaussian hidden Markov models with simple volatility benchmarks under chronological validation, model-stability rules and a sealed six-year test.

The main result is deliberately not a success story: fixed-lambda EWMA produced the lowest locked-test QLIKE. The regime model was more useful as a contemporaneous stress monitor than as a next-week variance or crash-warning model.

![EuroStress portfolio cover](reports/figures/portfolio_cover.png)

**Portfolio presentation:** [view the 12-slide PDF](reports/EuroStress_Quant_Risk_Portfolio_Deck.pdf) or [download the editable PowerPoint](reports/EuroStress_Quant_Risk_Portfolio_Deck.pptx).

## Result in one minute

- **Forecasting:** EWMA achieved the lowest mean QLIKE, 0.5530, across 312 locked-test weeks.
- **Confirmatory evidence:** the volatility-only HMM did not establish an improvement over EWMA; effect -0.0912 with a frozen 95% block-bootstrap interval of [-0.1768, 0.0015].
- **Exploratory evidence:** neither the PCA HMM nor the qualified-seed ensemble beat EWMA.
- **Monitoring:** the PCA high-risk-state probability had a Spearman correlation of 0.746 with ECB New CISS.
- **Tail events:** a validation-frozen shock rule produced four false alerts and missed all seven defined next-week tail events.
- **Feature sensitivity:** equity realised volatility was the only input with clear positive marginal value in the frozen no-refit ablation.

![Locked-test QLIKE ranking](reports/figures/locked_test_qlike.png)

## Research question

> Do latent European cross-asset regimes improve next-week equity-risk forecasts over simple volatility benchmarks?

The project separates three uses of the same latent-state machinery:

1. **Variance forecasting:** does a regime-weighted forecast reduce next-week realised-variance QLIKE?
2. **Stress monitoring:** does the inferred high-risk probability align with an external systemic-stress measure?
3. **Tail-event screening:** do frozen tail probabilities or shock scores identify unusually negative next-week equity returns?

Only the first question was the primary forecasting objective. Monitoring and tail metrics are reported with their original confirmatory or exploratory status.

## Point-in-time design

| Stage | Dates | Purpose |
|---|---|---|
| Training | 30 Mar 2007–31 Dec 2015 | Fit scaling, candidate models and event thresholds |
| Validation | 1 Jan 2016–31 Dec 2019 | Apply admissibility rules and freeze candidate choices |
| Locked test | 1 Jan 2020–26 Dec 2025 | Evaluate once across 312 weekly forecasts |

Every model row uses a Friday information cut-off. Source dates are retained, feature scaling is fit on the training sample only, and next-week targets begin after the information date.

### Weekly feature set

- Euro-area equity return
- Log 20-day equity realised variance
- EUR/USD return
- AAA two-year yield change
- AAA 10y–2y slope change
- Sovereign-fragmentation spread change

ECB New CISS is kept outside the forecasting feature set and used as an external stress-monitoring reference.

## Models and governance

The benchmark set contains 20-day historical variance and EWMA with fixed `lambda = 0.94`.

The original model family tested two- to four-state diagonal-covariance GMMs and HMMs across 20 deterministic initialisations. The original cross-asset HMM did not pass every frozen admissibility rule, so its confirmatory test comparison was not available.

Two branches proceeded to the locked evaluation:

- **Confirmatory secondary:** a two-state volatility-only HMM that passed the frozen rules.
- **Exploratory:** a train-only PCA HMM and an ensemble over validation-qualified seeds. These were frozen before the test was opened, but they remain labelled exploratory.

This distinction is part of the result. Stabilising a model does not retroactively convert it into confirmatory evidence.

## Locked-test results

Lower QLIKE is better.

| Forecast | Mean QLIKE | Effect vs EWMA | Frozen 95% interval | Status |
|---|---:|---:|---:|---|
| EWMA | **0.5530** | — | — | Benchmark winner |
| Cross-asset seed ensemble | 0.6149 | -0.0620 | [-0.1508, 0.0373] | Exploratory |
| Cross-asset PCA HMM | 0.6404 | -0.0874 | [-0.1741, 0.0083] | Exploratory |
| Volatility-only HMM | 0.6442 | -0.0912 | [-0.1768, 0.0015] | Confirmatory secondary |
| Historical variance | 0.6728 | — | — | Benchmark |

Effects are `reference loss − candidate loss`, so a positive value favours the candidate. No HMM established an improvement over EWMA.

### Stress monitoring

![Regime probability and ECB New CISS](reports/figures/regime_stress_monitor.png)

The exploratory PCA model remained structurally coherent in the test period: the lower-risk state occupied 77.3% of weeks, the higher-risk state 22.7%, and their median hard-filtered durations were eight and five weeks. Its high-risk probability correlated 0.746 with New CISS.

This is contemporaneous agreement with a systemic-stress indicator. It is not evidence of prediction, causality or trading profitability.

### Tail-event and sensitivity findings

![Tail-event and feature-sensitivity analysis](reports/figures/failure_analysis.png)

Seven locked-test weeks fell below the training fifth-percentile next-week return threshold of -5.21%. Average precision was low for every model. The strict PCA shock rule fired four times, with four false positives and seven missed events.

In a frozen no-refit ablation, removing equity realised volatility increased mean PCA-model QLIKE by 0.0246. Removing the other cross-asset inputs did not show incremental next-week variance value.

## What the project demonstrates

- point-in-time financial feature engineering with explicit information dates;
- chronological train, validation and locked-test separation;
- deterministic multi-seed GMM/HMM estimation and stability diagnostics;
- PCA and seed aggregation as clearly labelled exploratory stabilisation;
- benchmark discipline, QLIKE evaluation and block-bootstrap uncertainty;
- reproducible data lineage, hashes, automated tests and failure reporting.

The appropriate claim is: **simple EWMA remained difficult to beat, while latent regimes aligned more convincingly with contemporaneous European systemic stress than with next-week risk outcomes.**

## Reproduce the pipeline

Python 3.11 or newer is required.

```bash
make setup
make audit
make test
make data
make init-db
make features
make benchmarks
make models
make exploratory
```

The final evaluation was intentionally not exposed as a routine Make target. Its specification, hashes and one-time unlock process are documented in [`docs/final_evaluation_specification.md`](docs/final_evaluation_specification.md) and [`protocol/final_evaluation_lock.json`](protocol/final_evaluation_lock.json).

The frozen lock hash used for the reported evaluation is:

```text
f9d65e273660135d3b67b81af8ab3abaeb6be6b452d4363842f134fc7e0d58e9
```

## Repository map

```text
config/                 Source catalogue and frozen evaluation configuration
docs/                   Feature, benchmark, evaluation and portfolio documentation
protocol/               Research protocol, amendments and final lock
reports/                Selection reports, final results and presentation figures
src/eurostress/         Reusable package code
tests/                  Deterministic unit and leakage tests
data/                   Local-only raw, processed, model and evaluation artifacts
```

## Data and licensing

Raw third-party observations are not redistributed. The repository contains source identifiers, transformations, timing rules, tests and hashes. Users should review the relevant Yahoo Finance and ECB terms before downloading data locally.

## Portfolio materials

- [`reports/final_project_report.md`](reports/final_project_report.md): full locked-test report
- [`docs/portfolio_interview_guide.md`](docs/portfolio_interview_guide.md): CV bullets and interview talking points
- [`reports/EuroStress_Quant_Risk_Portfolio_Deck.pdf`](reports/EuroStress_Quant_Risk_Portfolio_Deck.pdf): recruiter-friendly 12-slide presentation
- [`reports/EuroStress_Quant_Risk_Portfolio_Deck.pptx`](reports/EuroStress_Quant_Risk_Portfolio_Deck.pptx): editable PowerPoint with speaker notes

The project supports no trading-profitability, causal-policy or reliable crash-prediction claim.
