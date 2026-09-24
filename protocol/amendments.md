# Protocol amendment log

## Amendment 1 — common sample start

- **Date:** 23 September 2026
- **Affected section:** Sample and chronological split
- **Old decision:** Overall and training sample began 1 September 2005.
- **New decision:** Overall and training sample begin 30 March 2007; all other split dates are unchanged.
- **Reason:** The configured `^STOXX50E` feed returns its first observation on 30 March 2007. The long-lived `EXW1` ETF was considered as a replacement, but the configured provider does not supply its missing 2005–2007 history. Splicing an ETF and price index would introduce an avoidable definition break. Starting the common sample in March 2007 still includes the global financial crisis.
- **Validation results viewed:** No.
- **Locked-test results viewed:** No.
- **Classification:** Confirmatory; operational source-coverage amendment made before feature construction or outcome analysis.

The original v1.0 protocol is preserved in `protocol/archive/research_protocol_v1.0.md`.

## Amendment 2 — realised-volatility implementation clarification

- **Date:** 23 September 2026
- **Affected section:** Predetermined features
- **Old decision:** Use log 20-trading-day realised equity volatility; the exact annualisation formula was not stated.
- **New decision:** Define it as `log(sqrt(252 × mean(r_daily²)))` over exactly 20 trailing daily log returns. A weekly value uses the last estimate available on or before Friday.
- **Reason:** This removes an implementation ambiguity before feature values are calculated. Root-mean-square realised volatility is defined even when the short-window mean return is non-zero and maps directly to realised variance used later in evaluation.
- **Validation results viewed:** No.
- **Locked-test results viewed:** No.
- **Classification:** Confirmatory implementation clarification made before feature construction or outcome analysis.

## Amendment 3 — target and benchmark timing definitions

- **Date:** 24 September 2026
- **Affected section:** Evaluation and benchmark definitions
- **Old decision:** Use next-week realised variance, historical volatility, and daily EWMA with `lambda = 0.94`; exact horizon, annualisation, and initialization conventions were not stated.
- **New decision:** For a Friday `t` forecast, define the target as `252/n × sum(r_daily²)` from returns ending in `(t, t+7 days]`. Define historical variance as `252/20 × sum(r_daily²)` over the 20 returns available at `t`. Initialize EWMA at the twentieth return using the first 20 squared returns, then update `h_d = 0.94 h_(d-1) + 0.06 r_d²`; the forecast is `252 h_d` using the last update available at `t`.
- **Reason:** These conventions align every forecast and outcome to an auditable information set and remove implementation ambiguity before any benchmark losses are calculated.
- **Validation results viewed:** No.
- **Locked-test results viewed:** No.
- **Classification:** Confirmatory implementation clarification made before target construction or performance analysis.

## Amendment 4 — minimum weekly-target coverage

- **Date:** 24 September 2026
- **Affected section:** Target data-quality rule
- **Old decision:** A target window was admissible with any positive number of observed daily returns.
- **New decision:** Require at least two observed daily returns in each seven-calendar-day target window.
- **Reason:** The structural coverage audit identified one year-end window containing only one return. A single return is not a credible weekly realised-variance estimate. The threshold retains legitimate two-day holiday weeks.
- **Information viewed:** Target-window dates and observation counts only; no returns, realised-variance values, losses, rankings, or model outputs.
- **Validation results viewed:** No.
- **Locked-test results viewed:** No.
- **Classification:** Confirmatory data-quality amendment made before performance analysis.

## Amendment 5 — model selection and state-to-risk implementation

- **Date:** 24 September 2026
- **Affected section:** Models and selection
- **Old decision:** Candidate HMM state counts, covariance type, number of starts, broad admissibility rules, and validation-likelihood selection were fixed; numerical stability thresholds, GMM selection, tie tolerance, and conversion from state probabilities to variance forecasts were not fully specified.
- **New decision:** Use seeds 0–19; require at least 80% convergence; reject validation log-likelihood IQR above 0.05 nats per observation; define ties within 0.01 nats per observation; measure occupancy from filtered probabilities and duration from hard-filtered runs; apply two-to-four-state validation-likelihood selection to GMM without the HMM-only duration rule. Map states to risk through filtered-probability-weighted mean next-week variance estimated solely from training targets.
- **Reason:** These choices make model selection and risk forecasts deterministic, causal, and auditable before any models are fit or validation likelihoods are observed.
- **Validation results viewed:** No.
- **Locked-test results viewed:** No.
- **Classification:** Confirmatory implementation clarification made before model fitting.

Future entries must include:

- date;
- affected protocol section;
- old and new decision;
- reason;
- whether validation results had been viewed;
- whether locked-test results had been viewed;
- whether the resulting analysis is confirmatory or exploratory.
