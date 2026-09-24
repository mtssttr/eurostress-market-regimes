# Weekly feature build report

**Build date:** 23 September 2026  
**Feature version:** 1.0.0  
**Run ID:** `features-041905aa49a82143`

## Integrity result

**PASS.** The point-in-time builder produced 974 complete Friday rows from 4 May 2007
through 26 December 2025.

| Split | Rows |
|---|---:|
| Training | 452 |
| Validation | 209 |
| Locked test | 313 |
| **Total** | **974** |

The table contains all six frozen model inputs and the external-validation CISS column. It has:

- zero missing or non-finite model-feature values;
- zero source dates after their associated Friday information date;
- a separate train-only standardized table;
- one provenance row for every feature row;
- 5,844 long-form feature records in DuckDB (974 weeks × six features);
- hashes for the raw manifest, configuration, and all processed outputs.

The build did not calculate model performance, inspect regimes, rank stress dates, or score the
locked test. Counts and dates are structural integrity information only.
