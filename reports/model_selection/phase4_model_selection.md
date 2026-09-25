# Phase 4 model-selection report

**Run ID:** `models-4ec8603766073d42`  
**Training rows:** 452 feature rows; 451 target-aligned rows  
**Validation rows:** 209  
**Locked-test rows read:** 0  
**Target-performance metrics computed:** No

## Outcome

Only the two-state volatility-only HMM passed every frozen admissibility rule. Its selected
maximum-training-likelihood initialization was seed 7.

No cross-asset HMM or GMM was admissible. This is a model-selection result, not evidence from the
locked test, and it must not be hidden or repaired by retroactively weakening the confirmatory
rules.

| Family | States | Validation predictive log likelihood per observation | Minimum occupancy | Minimum median duration | Seed-score IQR | Admissible | Failure reason |
|---|---:|---:|---:|---:|---:|---|---|
| Cross-asset GMM | 2 | -6.5504 | 3.69% | n/a | 0.0012 | No | Occupancy |
| Cross-asset GMM | 3 | -6.5041 | 1.22% | n/a | 0.0002 | No | Occupancy |
| Cross-asset GMM | 4 | -6.3378 | 0.44% | n/a | 0.1432 | No | Seed instability; occupancy |
| Volatility-only HMM | 2 | -1.8777 | 8.61% | 9.0 weeks | 0.0009 | **Yes** | — |
| Volatility-only HMM | 3 | -1.6807 | 4.05% | 3.0 weeks | 0.0003 | No | Occupancy |
| Volatility-only HMM | 4 | -1.1016 | 4.21% | 1.5 weeks | 0.4883 | No | Seed instability; occupancy; duration |
| Cross-asset HMM | 2 | -6.5314 | 5.85% | 4.0 weeks | 0.3707 | No | Seed instability |
| Cross-asset HMM | 3 | -6.1793 | 1.75% | 2.5 weeks | 0.0002 | No | Occupancy |
| Cross-asset HMM | 4 | -6.0098 | 1.42% | 1.5 weeks | 0.0829 | No | Seed instability; occupancy; duration |

Predictive log-likelihood levels are used only within each model family because the
volatility-only and cross-asset models have different input dimensions.

## Integrity checks

- All 180 deterministic fits completed: 3 families × 3 state counts × 20 seeds.
- The causal forward filter reproduces the fitted HMM sequence likelihood to numerical precision.
- State-to-variance mappings use only training outcomes.
- Validation outcomes were not used for fitting, calibration, selection, or scoring.
- The selected validation forecast file contains no outcomes or losses.

## Confirmatory interpretation

The frozen protocol did not identify a stable, adequately occupied cross-asset regime model.
Therefore, the planned primary cross-asset-HMM-versus-EWMA locked-test claim currently has no
admissible confirmatory model. The volatility-only HMM remains available for a secondary analysis.
Any revised cross-asset specification must be explicitly labelled exploratory and must not be
presented as the original confirmatory test.
