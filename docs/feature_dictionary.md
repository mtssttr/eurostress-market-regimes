# Weekly feature dictionary

**Frozen before feature calculation:** 23 September 2026  
**Feature implementation version:** 1.0.0

Every row has a Friday `information_date`. Only observations dated on or before that Friday
are eligible. A sampled level is invalid if its last source observation is more than five
calendar days old.

| Column | Definition | Unit | Model input? |
|---|---|---:|---:|
| `equity_return_1w` | Log change in the last available EURO STOXX 50 level from the previous Friday to the current Friday | log return | Yes |
| `equity_log_rv_20d` | Natural log of annualised trailing realised volatility, `log(sqrt(252 * mean(r_daily^2)))`, using exactly 20 daily log returns ending on or before Friday | log annualised volatility | Yes |
| `eurusd_return_1w` | Log change in USD per EUR from the previous Friday to the current Friday | log return | Yes |
| `aaa_2y_change_bps` | Weekly change in the sampled AAA two-year yield, multiplied by 100 | basis points | Yes |
| `aaa_curve_slope_change_bps` | Weekly change in sampled `(AAA 10-year − AAA 2-year)`, multiplied by 100 | basis points | Yes |
| `fragmentation_spread_change_bps` | Weekly change in sampled `(all-ratings 10-year − AAA 10-year)`, multiplied by 100 | basis points | Yes |
| `new_ciss_external` | Last available ECB New CISS observation | index | No; external validation only |

The six model features are also written in standardized form. Means and population standard
deviations are estimated only on rows with information dates through 31 December 2015, then
applied unchanged to validation and locked-test rows. CISS is never standardized with or passed
to the model.

The processed directory also contains source dates and staleness for every sampled raw series,
a manifest with input and output hashes, and a deterministic feature-run identifier.
