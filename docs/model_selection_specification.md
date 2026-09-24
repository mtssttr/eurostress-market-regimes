# Phase 4 model-selection specification

**Frozen before fitting:** 24 September 2026  
**Model implementation version:** 1.0.0

## Families

| Family | Inputs | State/component counts | Temporal dependence |
|---|---|---:|---|
| Cross-asset GMM | All six standardized features | 2, 3, 4 | No |
| Volatility-only HMM | Standardized 20-day log realised volatility | 2, 3, 4 | Yes |
| Cross-asset HMM | All six standardized features | 2, 3, 4 | Yes |

Every fit uses diagonal covariance, a covariance floor of `1e-4`, at most 500 iterations, a
convergence tolerance of `1e-4`, and deterministic seeds 0–19. Within a family/state count, the
converged seed with the highest training likelihood supplies the candidate model.

## Admissibility and selection

An HMM candidate must satisfy all of the following on the validation sequence:

- each state's mean filtered probability is at least 5%;
- each state's median contiguous hard-filtered run is at least two weeks;
- at least 80% of seeds converge;
- the across-seed IQR of predictive log likelihood is no more than 0.05 nats per observation.

GMM uses the convergence, stability, and occupancy rules but not duration, because it has no
transition process. Within each family, validation predictive log likelihood selects among
admissible candidates. Scores within 0.01 nats per observation are ties and the lower state count
wins.

## Variance forecasts

Only causal filtered probabilities are used. For each selected model, state `k` receives the
training-only variance

`v_k = sum_t(p_tk × realised_variance_t) / sum_t(p_tk)`.

The Friday forecast is `sum_k(p_tk × v_k)`. Validation outcomes are not used for fitting,
calibration, or selection. Phase 4 reads only training and validation rows and does not calculate
QLIKE or any other target-performance measure.
