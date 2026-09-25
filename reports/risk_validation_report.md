# Market-risk validation extension

**Classification:** Post-hoc extension; the original sealed EuroStress result is unchanged.

## Question

> How reliable are conventional market-risk models across financial-stress regimes?

Daily one-step forecasts compare 250-day historical simulation, a 250-day Gaussian model, RiskMetrics-style EWMA volatility and a Gaussian GARCH(1,1). Every forecast uses information available by the preceding trading close. Parameters for GARCH are estimated only through 31 December 2019 and then held fixed through 2020–2025.

## Primary 99% VaR backtest

| Model | Exceptions | Expected exceptions | Exception rate | Kupiec p-value | Independence p-value | Conditional-coverage p-value |
|---|---|---|---|---|---|---|
| Historical simulation | 26 | 15.1 | 1.722% | 0.011 | <0.001 | <0.001 |
| Rolling Gaussian | 37 | 15.1 | 2.450% | <0.001 | <0.001 | <0.001 |
| EWMA Gaussian | 39 | 15.1 | 2.583% | <0.001 | 0.019 | <0.001 |
| GARCH(1,1) Gaussian | 34 | 15.1 | 2.252% | <0.001 | 0.794 | <0.001 |

## Key findings

- All four models produced more 99% VaR exceptions than the 15.1 expected under correct coverage. Historical simulation was closest with 26 exceptions, but its Kupiec p-value of 0.011 still rejects correct unconditional coverage at the 5% level.
- GARCH reduced the COVID-window exception count to three, compared with six for EWMA, eight for historical simulation and eleven for the rolling Gaussian model. Its full-sample exception rate remained too high, but its Christoffersen independence p-value of 0.794 did not indicate first-order clustering.
- Historical and rolling-Gaussian exception rates rose in the higher-risk regime. EWMA and GARCH showed the reverse descriptive pattern because their volatility forecasts increased after shocks; the weekly regime label can also arrive after a sudden daily loss. This is evidence for conditional model diagnostics, not for a causal regime effect.

Kupiec tests unconditional coverage. Christoffersen tests first-order exception independence. The conditional-coverage statistic combines both. The ES severity ratio is realised mean exception loss divided by forecast ES on exception days; it is a descriptive diagnostic, not presented as a standalone formal ES test.

## 97.5% Expected Shortfall diagnostics

| Model | Exceptions | Expected exceptions | Exception rate | Mean ES loss | Realised Tail Loss | ES severity ratio |
|---|---|---|---|---|---|---|
| Historical simulation | 45 | 37.8 | 2.980% | 3.637% | 3.675% | 1.13 |
| Rolling Gaussian | 48 | 37.8 | 3.179% | 2.840% | 3.717% | 1.43 |
| EWMA Gaussian | 55 | 37.8 | 3.642% | 2.693% | 3.122% | 1.29 |
| GARCH(1,1) Gaussian | 55 | 37.8 | 3.642% | 2.733% | 3.188% | 1.19 |

The 97.5% level matches the one-tailed confidence level used for Expected Shortfall in the Basel market-risk framework. A severity ratio above one means realised exception losses were larger on average than the model's ES forecast. This ratio is descriptive; formal ES validation requires a joint VaR/ES testing framework and is left as a clearly identified extension.

## Coverage by EuroStress regime

| Model | Regime | Observations | Exceptions | Exception rate | Coverage Ratio |
|---|---|---|---|---|---|
| Historical simulation | higher-risk regime | 338 | 8 | 2.367% | 2.37 |
| Historical simulation | lower-risk regime | 1171 | 18 | 1.537% | 1.54 |
| Rolling Gaussian | higher-risk regime | 338 | 13 | 3.846% | 3.85 |
| Rolling Gaussian | lower-risk regime | 1171 | 24 | 2.050% | 2.05 |
| EWMA Gaussian | higher-risk regime | 338 | 6 | 1.775% | 1.78 |
| EWMA Gaussian | lower-risk regime | 1171 | 33 | 2.818% | 2.82 |
| GARCH(1,1) Gaussian | higher-risk regime | 338 | 5 | 1.479% | 1.48 |
| GARCH(1,1) Gaussian | lower-risk regime | 1171 | 29 | 2.477% | 2.48 |

## Named stress-window diagnostics

| Stress Window | Model | Observations | Exceptions | Exception rate | Mean VaR loss |
|---|---|---|---|---|---|
| banking_stress | Historical simulation | 36 | 2 | 5.556% | 3.256% |
| banking_stress | Rolling Gaussian | 36 | 2 | 5.556% | 2.881% |
| banking_stress | EWMA Gaussian | 36 | 2 | 5.556% | 2.735% |
| banking_stress | GARCH(1,1) Gaussian | 36 | 2 | 5.556% | 2.764% |
| covid_market_shock | Historical simulation | 49 | 8 | 16.327% | 4.709% |
| covid_market_shock | Rolling Gaussian | 49 | 11 | 22.449% | 3.342% |
| covid_market_shock | EWMA Gaussian | 49 | 6 | 12.245% | 7.160% |
| covid_market_shock | GARCH(1,1) Gaussian | 49 | 3 | 6.122% | 7.306% |
| inflation_energy_repricing | Historical simulation | 173 | 3 | 1.734% | 4.146% |
| inflation_energy_repricing | Rolling Gaussian | 173 | 5 | 2.890% | 3.279% |
| inflation_energy_repricing | EWMA Gaussian | 173 | 3 | 1.734% | 3.640% |
| inflation_energy_repricing | GARCH(1,1) Gaussian | 173 | 3 | 1.734% | 3.538% |

## GARCH fit

- Mean daily return: 0.04812%
- Omega: 0.031347 percent-squared
- Alpha: 0.1109
- Beta: 0.8761
- Persistence: 0.9869
- Pre-2020 estimation observations: 3,190
- Optimizer converged: yes

## Interpretation limits

This is a deliberately contained validation exercise on one equity index. It is not a bank trading-book capital model, does not implement liquidity-horizon scaling, and does not establish that the exploratory regime labels cause model failure. Named stress windows and regime-conditional results are descriptive. The first evaluation day is excluded from regime splits because no prior Friday state estimate existed. Raw daily observations and row-level forecasts remain local and are not redistributed.

## Methodological references

- Bollerslev, T. (1986), *Generalized Autoregressive Conditional Heteroskedasticity*, Journal of Econometrics 31(3), 307–327. https://doi.org/10.1016/0304-4076(86)90063-1
- Kupiec, P. (1995), *Techniques for Verifying the Accuracy of Risk Measurement Models*, Federal Reserve FEDS 95-24. https://www.fedinprint.org/item/fedgfe/34596/original
- Christoffersen, P. (1998), *Evaluating Interval Forecasts*, International Economic Review 39(4), 841–862. https://doi.org/10.2307/2527341
- McNeil, A. and Frey, R. (2000), *Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series*, Journal of Empirical Finance 7, 271–300. https://doi.org/10.1016/S0927-5398(00)00012-8
- Basel Committee on Banking Supervision, *Basel Framework*, MAR33: Internal models approach. https://www.bis.org/baselframework/BaselFramework.pdf
