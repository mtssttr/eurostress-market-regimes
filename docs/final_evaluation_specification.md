# Final evaluation specification

**Status:** Frozen before the locked test is opened  
**Freeze date:** 24 September 2026

## Loss and paired effect

For positive realised variance `y` and forecast `h`, QLIKE is

`QLIKE(y, h) = y/h - log(y/h) - 1`.

It equals zero for a perfect forecast and is otherwise positive. Non-finite or non-positive inputs
cause evaluation failure. Strictly positive values below `1e-12` are floored only for numerical
safety, and the number of floored values must be reported.

Every comparison uses the weekly paired loss difference

`d_t = QLIKE(reference_t) - QLIKE(candidate_t)`.

A positive mean favors the candidate.

## Uncertainty

The main 95% interval is the percentile interval from 10,000 paired circular moving-block
bootstrap samples with an eight-week block and random seed `20260924`. Every model comparison
uses the same resampled week indices. Four- and thirteen-week blocks are reported as sensitivity
checks and cannot replace the main interval.

## Comparison hierarchy

The original confirmatory cross-asset comparison is recorded as not testable because Phase 4 had
no admissible cross-asset model. It will not be silently replaced by an exploratory model.

The only remaining confirmatory-class comparison is the secondary volatility-only HMM versus
EWMA comparison. It succeeds only if the mean paired effect is positive and its main 95% interval
lies strictly above zero. No multiplicity correction is needed for one confirmatory comparison.

The PCA HMM comparisons are exploratory; the seed ensemble is a robustness analysis. Their
intervals are descriptive regardless of whether they exclude zero.

## Secondary outputs

- Brier score and average precision (the `sklearn.metrics.average_precision_score` summary of the precision-recall curve) for a next-week equity return below the training fifth percentile. State tail probabilities are calibrated with training outcomes only.
- PCA-HMM shock scores using negative one-step predictive log density. The threshold is the validation 99th percentile with the `higher` quantile rule.
- Spearman correlation between the filtered higher-training-risk PCA state probability and New CISS.
- Median contiguous hard-filtered state duration.
- One-at-a-time PCA-HMM sensitivity ablations that set a standardized feature to its training mean of zero without refitting.
- Counts of shock false positives and missed tail events under the frozen definitions.

These outputs support risk-monitoring interpretation only. They do not establish causality or
trading profitability.

## One-time unlock procedure

1. Generate `protocol/final_evaluation_lock.json` without loading any test row.
2. Verify hashes for configuration, code, manifests, and trained model artifacts.
3. Confirm the expected locked sample is exactly 312 complete rows.
4. Run the evaluator once with an explicit `--unlock-locked-test` flag.
5. Write a permanent opened marker containing the lock hash, timestamp, and result hashes.
6. Permit later reproduction only with the frozen lock and an explicit reproduction flag; no model, threshold, or reporting choice may be changed after opening.
