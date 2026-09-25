# Risk-validation technical defense

Use this document to prepare for a 20–30 minute technical interview. The objective is not to
memorise every number; it is to understand the information set, equations, test hypotheses and
limits well enough to reconstruct the argument.

## The study in one sentence

EuroStress compares four conventional one-day tail-risk models on 1,510 EURO STOXX 50 returns
from 2020–2025, formally tests their VaR exception sequences, and describes how failures differ
across the pre-existing weekly stress regimes.

## Information timing

For a return observed on trading day `t`, every forecast uses information available at the close of
the previous trading day. Historical and Gaussian windows exclude `r_t`. EWMA and GARCH update with
`r_(t-1)` before forecasting `t`. GARCH parameters are estimated only on returns ending by
31 December 2019. A daily forecast receives the most recent weekly regime probability available at
its forecast origin, not at its realised-return date.

This timing rule is the first thing to explain if asked about leakage.

## Model equations

Let `r_t` be the daily log return and `L_t = -r_t` the loss.

### Historical simulation

The one-day VaR is the negative empirical lower-tail quantile of the previous 250 returns. Expected
Shortfall is the negative mean of historical returns at or below that quantile. It makes no normality
assumption, but it reacts slowly when an old crisis observation enters or leaves the window.

### Rolling Gaussian

Estimate `mu_t` and `sigma_t` from the previous 250 returns. For lower-tail probability `alpha`:

```text
VaR_loss,t(alpha) = -(mu_t + sigma_t Phi^-1(alpha))
ES_loss,t(alpha)  = -mu_t + sigma_t phi(Phi^-1(alpha)) / alpha
```

The weakness is not the volatility estimate alone: the Gaussian tail can be too thin.

### EWMA Gaussian

The conditional variance forecast is:

```text
h_t = 0.94 h_(t-1) + 0.06 r_(t-1)^2
```

The conditional mean is set to zero and Gaussian VaR/ES formulas are applied to `sqrt(h_t)`. EWMA
responds rapidly to recent shocks but has a fixed persistence structure.

### GARCH(1,1) Gaussian

```text
h_t = omega + a (r_(t-1) - mu)^2 + b h_(t-1)
```

The fitted parameters are `a = 0.1109`, `b = 0.8761`, so persistence is `0.9869`. The model is
stationary because `a + b < 1`. Parameters are estimated by Gaussian maximum likelihood using
3,190 pre-2020 returns and then held fixed.

## Validation tests

### Exception indicator

`I_t = 1` when realised loss exceeds forecast VaR. A correct 99% VaR should have exception
probability 1%.

### Kupiec unconditional-coverage test

The null is that the exception probability equals the requested tail probability. It compares the
binomial likelihood under `p = alpha` with the likelihood under the observed exception rate. It does
not test whether exceptions cluster.

### Christoffersen independence test

The hit sequence is treated as a two-state first-order Markov chain. The null is that the probability
of an exception is the same after a non-exception and after an exception. It tests clustering but not
whether the total exception rate is correct.

### Conditional coverage

The conditional-coverage statistic is the sum of the Kupiec and independence likelihood-ratio
statistics. It asks whether both frequency and first-order independence are acceptable.

### Expected Shortfall diagnostic

At the 97.5% level, the project reports forecast ES and the ratio:

```text
mean realised loss on VaR-exception days / mean forecast ES on those days
```

A ratio above one indicates underestimated exception severity. This is deliberately called a
diagnostic, not a formal standalone ES backtest. A defensible next step is a joint VaR/ES test.

## Numbers worth knowing

| Model | 99% exceptions | Rate | Kupiec p | Independence p | 97.5% ES severity ratio |
|---|---:|---:|---:|---:|---:|
| Historical simulation | 26 | 1.72% | 0.011 | <0.001 | 1.13 |
| Rolling Gaussian | 37 | 2.45% | <0.001 | <0.001 | 1.43 |
| EWMA Gaussian | 39 | 2.58% | <0.001 | 0.019 | 1.29 |
| GARCH(1,1) Gaussian | 34 | 2.25% | <0.001 | 0.794 | 1.19 |

The target was 15.1 exceptions. Do not say GARCH “passed.” It passed only the independence test;
it failed coverage because 34 exceptions are too many.

## Interpretation of the regime result

Historical and rolling-Gaussian exception rates were higher in the higher-risk state. EWMA and
GARCH showed lower exception rates there because conditional volatility rose after shocks. This
does not mean low-risk regimes are intrinsically more dangerous. A weekly contemporaneous regime
can label a sudden shock only after it has occurred, while a volatility model may already have
raised its forecast following the first large return. The result motivates timing-aware conditional
diagnostics rather than a causal story.

## Likely interview challenges

### Why use log returns?

They aggregate additively through time and are conventional for conditional-volatility modelling.
For one-day returns the numerical difference from simple returns is small, but the definition is
kept consistent across every model.

### Why 250 observations?

It approximates one trading year and is common in historical risk estimation. The choice trades
adaptability against quantile precision: at a 1% tail, only about 2.5 observations determine the
historical quantile, which is one reason the estimate is step-like and unstable.

### Why use both 99% VaR and 97.5% ES?

The 99% VaR supports familiar exception backtests. The Basel market-risk framework uses 97.5%
one-tailed ES, so the ES diagnostic is reported at that level. They answer different questions:
VaR is a threshold; ES concerns average loss beyond a tail threshold.

### Why did Gaussian GARCH still under-cover?

GARCH addresses time-varying conditional variance, not necessarily heavy-tailed innovations,
asymmetry or jumps. Its independence result suggests that volatility clustering was handled better,
while the excessive total count points to distributional tail misspecification.

### Why not Student-t or EVT immediately?

The extension was intentionally contained. Student-t GARCH is the cleanest next benchmark. EVT
requires threshold selection, stability diagnostics and enough tail observations; adding it without
those checks would create another fragile layer rather than strengthen the project.

### Are the named stress windows pre-specified?

No. They are post-hoc descriptive windows and are labelled accordingly. The formal statistical
claims come from the complete exception sequence, not from choosing favorable episodes.

### Is this a regulatory capital model?

No. It is a one-index, one-day validation exercise. It does not implement a trading-book portfolio,
10-day horizons, liquidity-horizon scaling, stressed calibration, modellability tests or P&L
attribution. Those limitations should be stated before an interviewer has to point them out.

## References to defend the methodology

- Bollerslev (1986): GARCH model and stationarity structure.
- Kupiec (1995): unconditional VaR coverage testing.
- Christoffersen (1998): conditional interval evaluation and exception independence.
- McNeil and Frey (2000): conditional volatility, tail risk, VaR and ES.
- Basel Framework MAR33: 97.5% one-tailed ES in the internal-models market-risk framework.

## Final defensible claim

The extension does not show that one model solved market risk. It shows that volatility dynamics and
tail calibration are separate validation problems: GARCH improved the timing of exceptions, but no
Gaussian model delivered correct 99% loss coverage, and every ES diagnostic understated realised
tail severity.
