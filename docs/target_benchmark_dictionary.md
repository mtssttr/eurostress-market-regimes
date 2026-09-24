# Target and benchmark dictionary

**Frozen before target construction:** 24 September 2026  
**Benchmark implementation version:** 1.0.0

For every Friday information date `t`, forecast inputs use only price observations dated on or
before `t`. The target is deliberately forward-looking and is kept separate from all feature and
forecast columns.

| Column | Definition | Availability |
|---|---|---|
| `realised_variance` | `252/n × sum(r_daily²)` for the `n` daily log returns ending in `(t, t+7 calendar days]` | Outcome; known only after the next Friday |
| `historical_variance_forecast` | `252/20 × sum(r_daily²)` over the last 20 daily returns available at `t` | Forecast known at `t` |
| `ewma_variance_forecast` | `252 h_t`, where `h_d = 0.94 h_(d-1) + 0.06 r_d²` | Forecast known at `t` |

EWMA is initialized on the date of the twentieth daily return using the mean of the first 20
squared returns. `historical_source_date` and `ewma_source_date` record the last observation each
forecast used. Target boundary dates, first and last included return dates, and the number of
included daily returns are also retained.

A target row is retained only when its seven-day window contains at least two daily returns. This
rule was frozen from date/count coverage information before any target values or losses were
examined.

No QLIKE values, forecast rankings, or other performance statistics are calculated during this
phase.
