# Exploratory tail-model comparison

Do distribution choice and monthly parameter updating change tail calibration?

2020-2025 outcomes had already been examined. These are exploratory comparisons, not a new untouched holdout or proof of future performance.

## Fixed design

GARCH(1,1), constant mean, normal versus unit-variance Student t innovations; fixed versus first-observation-of-month refitting. Every fit uses exactly the preceding 1260 returns. Both schedules start with the same estimation window; parameters for fixed models remain unchanged while conditional variance updates daily. The new fixed Gaussian comparator uses a different estimation window/library from the original project baseline and must not be presented as its exact reproduction.

## 99% VaR comparison

| Model | Days | Breaches | Rate | Coverage p | Independence p | Quantile loss |
|---|---:|---:|---:|---:|---:|---:|
| normal_fixed | 1510 | 37 | 2.45% | 0.0000 | 0.9217 | 0.000516 |
| normal_monthly | 1510 | 36 | 2.38% | 0.0000 | 0.8790 | 0.000510 |
| t_fixed | 1510 | 30 | 1.99% | 0.0007 | 0.6259 | 0.000486 |
| t_monthly | 1510 | 25 | 1.66% | 0.0193 | 0.4312 | 0.000477 |

Quantile loss rewards calibrated quantiles rather than simply fewer breaches. Fewer breaches alone can result from excessively conservative limits. Coverage and first-order independence p-values use asymptotic reference distributions; non-rejection does not establish a correct model. No multiplicity-adjusted superiority claim is made.

ES severity is reported descriptively at both tail levels in comparison.csv. It is not a formal ES validation test. The 97.5% ES diagnostic uses 2.5% VaR exception days. Single-index results do not establish trading-book suitability.

## Reproduce

`make tail-extension` (requires existing local equity data).

References: https://bashtage.github.io/arch/univariate/univariate_volatility_modeling.html
https://doi.org/10.2307/2527341
