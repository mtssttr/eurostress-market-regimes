# Phase 4B exploratory stabilization report

**Classification:** Exploratory  
**Run ID:** `exploratory-d4b36fffcc78cc48`  
**Training rows:** 452 feature rows; 451 target-aligned rows  
**Validation rows:** 209  
**Locked-test rows read:** 0  
**Target-performance metrics computed:** No

## Outcome

Both stabilization approaches produced a structurally viable two-state cross-asset model.

| Approach | Selected states | Selected initialization/members | Validation predictive log likelihood per observation | Role |
|---|---:|---|---:|---|
| Training-only PCA HMM | 2 | Seed 12 | -3.7945 | Primary exploratory model |
| Full-feature seed ensemble HMM | 2 | 14 qualified seeds | -6.5318 | Robustness model |

The likelihood levels are not compared across approaches because PCA and the full-feature model
have different input dimensions. State counts were selected only within each approach.

The three-state and four-state PCA models failed occupancy/duration rules. The three-state and
four-state full-feature ensembles had no members that jointly occupied all states sufficiently
and maintained the required duration.

## PCA stability and representation

The PCA transformation was fitted on the 452 training rows only. The first three components
explain 71.08% of training-feature variance:

| Component | Explained variance |
|---|---:|
| PC1 | 33.02% |
| PC2 | 22.86% |
| PC3 | 15.20% |

The selected two-state PCA HMM passed the original single-model rules:

- 100% of its 20 seeds converged;
- validation seed-score IQR: 0.0008 nats per observation;
- minimum validation occupancy: 7.56%;
- minimum validation median duration: 7.5 weeks.

## Training-period regime profiles

The state means below are transformed back into standardized original-feature units. Labels are
ordered using the training-only state-to-variance mapping.

| Quantity | Lower-training-risk state | Higher-training-risk state |
|---|---:|---:|
| Training variance mapping | 0.0385 | 0.1021 |
| Equity weekly return | +0.013 | -0.020 |
| Log 20-day equity volatility | -0.545 | +0.853 |
| EUR/USD weekly return | -0.025 | +0.039 |
| AAA two-year yield change | +0.098 | -0.153 |
| AAA curve-slope change | -0.029 | +0.046 |
| Fragmentation-spread change | -0.076 | +0.119 |

The higher-training-risk regime therefore combines substantially higher equity volatility,
slightly negative equity returns, falling short yields, curve steepening, and widening
fragmentation in the fitted training sample. These are descriptive associations, not causal
claims.

The fitted weekly self-transition probabilities are 95.0% for the lower-risk state and 92.3%
for the higher-risk state, implying persistent rather than week-to-week flickering regimes.

## Seed ensemble

All 20 two-state seeds belonged to the high-training-likelihood basin. Fourteen also passed the
original validation occupancy and duration rules and were retained. Their predictive densities
and training-calibrated variance forecasts are averaged with equal weights. The ensemble is kept
as a robustness forecast rather than the primary regime model because its independently fitted
members do not share one canonical state-label system.

## Research status

This successful stabilization does not alter the Phase 4 confirmatory result: no original
cross-asset candidate was admissible. Phase 4B was designed after viewing Phase 4 validation
diagnostics, so all PCA and ensemble results must remain labelled exploratory. No realised
variance losses, QLIKE values, benchmark rankings, or locked-test outcomes were computed.
