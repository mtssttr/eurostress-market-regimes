# EuroStress final results

**Evaluation date:** 24 September 2026  
**Locked-test period:** January 2020–December 2025, 312 weekly forecasts  
**Frozen evaluation lock:** `f9d65e273660135d3b67b81af8ab3abaeb6be6b452d4363842f134fc7e0d58e9`

## Executive conclusion

Simple EWMA produced the lowest mean QLIKE in the locked test. Neither the confirmatory
volatility-only HMM nor the exploratory cross-asset regime models established an improvement
over EWMA.

The cross-asset regime model was more useful as a contemporaneous stress monitor than as a
next-week variance or tail-event forecaster. Its higher-risk-state probability had a strong
Spearman correlation of 0.746 with ECB New CISS, but its strict shock alerts did not identify any
of the seven frozen next-week equity-tail events.

This is a negative forecasting result and a positive research-process result: the project kept
the test period sealed, retained failure rules, reported unstable and under-occupied models, and
did not replace the original hypothesis with a post-hoc success claim.

## Frozen QLIKE results

Lower mean QLIKE is better. Each effect is `reference loss − candidate loss`, so a positive value
favours the candidate.

| Forecast | Mean QLIKE | Comparison with EWMA | Frozen 95% interval | Interpretation |
|---|---:|---:|---:|---|
| EWMA, `lambda = 0.94` | **0.5530** | — | — | Lowest mean loss |
| Cross-asset seed ensemble | 0.6149 | -0.0620 | [-0.1508, 0.0373] | Exploratory; no improvement established |
| Cross-asset PCA HMM | 0.6404 | -0.0874 | [-0.1741, 0.0083] | Exploratory; no improvement established |
| Volatility-only HMM | 0.6442 | -0.0912 | [-0.1768, 0.0015] | Confirmatory secondary outperformance not established |
| Historical 20-day variance | 0.6728 | — | — | Highest mean loss |

The original confirmatory cross-asset comparison was not testable because no original Phase 4
cross-asset model passed every frozen admissibility rule.

The exploratory ensemble improved on the single PCA HMM by 0.0255 QLIKE, with a descriptive 95%
interval of [0.0049, 0.0481]. Four- and thirteen-week bootstrap blocks gave the same directional
result. This supports seed aggregation as a robustness device, but it does not turn the ensemble
into a confirmatory model and it still did not beat EWMA.

## Regime-monitoring result

The PCA HMM remained structurally coherent in the test period:

- lower-training-risk state occupancy: 77.3%;
- higher-training-risk state occupancy: 22.7%;
- median hard-filtered durations: eight and five weeks;
- higher-risk probability versus New CISS Spearman correlation: **0.746**.

High regime probabilities occurred during several recognizable market-stress windows, including
March–June 2020, early 2022, and March 2023. This is contemporaneous agreement with a systemic
stress indicator, not proof that the model predicted those events or identified their causes.

## Tail events and shock detection

The training fifth-percentile threshold for next-week equity return was -5.21%. Seven locked-test
weeks crossed that threshold.

| Model | Brier score | Average precision |
|---|---:|---:|
| Volatility-only HMM | 0.0225 | 0.0451 |
| PCA HMM | **0.0224** | **0.0516** |
| Seed ensemble HMM | 0.0229 | 0.0429 |

All average-precision values were low. The PCA shock score produced four alerts at its frozen
validation 99th-percentile threshold. None preceded a defined next-week tail event, and all seven
tail events were missed by that strict alert rule. The shock score therefore should not be
presented as a successful crash-warning system.

## Feature sensitivity

Setting one standardized PCA input at a time to its training mean gave the following changes in
mean test QLIKE relative to the unablated PCA model. Positive changes mean performance worsened
when the feature was removed.

| Removed feature | Mean QLIKE change |
|---|---:|
| Equity log 20-day realised volatility | **+0.0246** |
| Equity weekly return | -0.0003 |
| EUR/USD weekly return | -0.0070 |
| AAA two-year yield change | -0.0133 |
| AAA curve-slope change | -0.0137 |
| Fragmentation-spread change | -0.0007 |

Equity volatility was the only input with clear positive marginal value under this frozen
no-refit sensitivity test. The remaining cross-asset inputs did not demonstrate incremental
next-week variance-forecast value, which is consistent with the broader negative forecasting
result.

## Appropriate project claim

EuroStress demonstrates a reproducible, leakage-controlled comparison of European market-regime
models against strong volatility baselines. The correct conclusion is not that an HMM predicts
crises. It is that simple EWMA remained difficult to beat, seed instability materially affected
cross-asset HMMs, and latent regimes aligned more convincingly with contemporaneous systemic
stress than with next-week risk outcomes.

A concise CV description could be:

> Built a point-in-time European cross-asset risk monitor on 2007–2025 equity, FX, yield-curve,
> fragmentation and ECB stress data; evaluated GMM/HMM/PCA-ensemble models with chronological
> validation, seed-stability controls, a sealed six-year test and block-bootstrap QLIKE. Found
> EWMA remained the strongest variance forecast, while filtered regime stress correlated 0.75
> with ECB New CISS.

The project supports no trading-profitability or causal-policy claim.
