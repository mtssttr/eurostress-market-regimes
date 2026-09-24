# Exploratory analysis log

## Phase 4B — cross-asset HMM stabilization

- **Date frozen:** 24 September 2026
- **Classification:** Exploratory
- **Reason:** No cross-asset Phase 4 candidate passed all confirmatory admissibility rules. The two-state model failed seed stability; higher-state models produced under-occupied regimes.
- **Information already viewed:** Phase 4 training and validation likelihood, occupancy, duration, convergence, and seed-stability diagnostics.
- **Target-performance results viewed:** No.
- **Locked-test results viewed:** No.
- **Locked-test access during Phase 4B:** Prohibited.

Two stabilization approaches are fixed before Phase 4B fitting:

1. Fit a three-component PCA on the six standardized training features only, transform validation with the frozen training loadings, and apply the original HMM admissibility and selection rules.
2. For each full-feature HMM state count, retain converged seeds whose training log likelihood is within 0.05 nats per observation of the best seed and that individually pass the original 5% validation-occupancy and two-week median-duration rules. At least five qualified members are required. Average member predictive densities as an equal-weight mixture for state-count selection and average their training-calibrated variance forecasts.

These analyses may motivate an exploratory model but cannot rescue or replace the failed
confirmatory Phase 4 cross-asset model-selection result.

### Post-stabilization role decision

Both approaches produced structurally viable two-state models. Before viewing any target-loss
metric, the PCA HMM was designated the primary exploratory model because it passed the original
single-model admissibility rules and retains one coherent transition process. The seed ensemble
was designated a robustness forecast because its independently fitted members do not share a
single state-label system. This decision remains exploratory and does not alter Phase 4.
