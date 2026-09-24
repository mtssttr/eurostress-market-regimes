# Phase 3 structural build report

**Build date:** 24 September 2026  
**Benchmark version:** 1.0.0  
**Run ID:** `benchmarks-2d84d10d40ad41f3`

## Integrity result

**PASS.** The point-in-time builder produced 972 complete target/forecast rows from 4 May
2007 through 19 December 2025.

| Split | Rows |
|---|---:|
| Training | 451 |
| Validation | 209 |
| Locked test | 312 |
| **Total** | **972** |

The final table has:

- zero missing or non-finite values;
- forecast source dates no later than their Friday information dates;
- target returns strictly after the information date and no later than the next Friday;
- between two and five observed daily returns per retained target window;
- a seven-calendar-day target horizon for every row;
- one matching DuckDB record per output row;
- hashes for the configuration, raw-data manifest, feature manifest, and output;
- an idempotent build that replaces, rather than duplicates, the same run.

One training-period week ending 2 January 2015 was excluded because its source window contained
only one observed return. This decision used dates and counts only and is recorded as protocol
Amendment 4.

No QLIKE values, benchmark comparisons, forecast rankings, model outputs, or locked-test
performance statistics were computed or inspected.
