# Ng–Perron and KSS unit-root tests

`oe.ngperron` and `oe.kss` run native float64 Torch regressions and return
OpenEconometrics DataFrames with exportable tables and JSON-safe `.attrs`.
They are method-level coverage, not blanket Stata parity. Neither requires
SciPy, R or an external estimation engine at runtime.

```python
ng = oe.ngperron(df, "income", time="period", trend="constant", maxlag=8)
fixed = oe.ngperron(df, "income", trend="trend", lags=2)
nonlinear = oe.kss(df, "income", time="period", trend="constant", lags=2)
print(ng)
ng.to_latex()
```

KSS tests a unit-root null and rejects in the **lower tail**. Every KSS row
contains its statistic, the published 1%, 5% and 10% critical values and
three rejection indicators. Ng–Perron currently returns **statistics only**:
its independent published-table calibration remains unresolved in MARKET-136.
No Ng–Perron critical values or rejection decisions are returned.
`p_value` is explicitly `None`: interpolation or
a conventional Student-t p-value would fabricate a distribution not supplied
by these references. The quantiles are **asymptotic**, including for small
accepted samples; they do not guarantee finite-sample size. Serial dependence
needs adequate augmentation. Breaks, arbitrary heteroskedasticity and general
dependence-robust inference are outside these contracts.

## Ng–Perron GLS M tests

[Ng and Perron (2001), Econometrica 69(6), 1519–1554](https://www.columbia.edu/~sn2294/pub/ecta01.pdf)
define MZa, MZt, MSB and MPT in equations (3)–(6), (8)–(9), with critical
values in Table I (p. 1524). `trend="constant"` tests level stationarity with
GLS local parameter c=−7; `trend="trend"` tests trend stationarity with c=−13.5.

`inference="none"` is the only currently available inference mode. Asking
for `inference="published"` raises `critical_values_unverified`, rather than
producing an unverified decision. The independent Table I simulation retained
20,000 paths/5,000 steps and found about 8.2–8.5% lower-tail rejection at the
published trend 10% thresholds, outside its declared tolerance. This is an
unresolved replication discrepancy, not evidence that the paper is wrong.
The statistics and MAIC selection separately agree with an independent SVD
replication. The following table documents the publication, not available
calibrated engine output; MARKET-136 remains open.

The n input levels represent y₀,…,yT, **T=n−1**, rather than n increments.
GLS uses α=1+c/T and an unweighted initial observation, followed by the
quasi-differences. A constant or constant/linear trend is removed by GLS.
Positive rescaling and an origin shift stabilize the calculation and leave
these test statistics unchanged. Existing `dfgls` conventions are preserved.

Fixed `lags` specifies the number of lagged differences in the detrended
AR regression. Otherwise, MAIC compares **every order 0,…,maxlag** on one
common sample of T−maxlag increments. It minimizes
log(SSR/N)+2·(τ+k)/N, where τ=b₀²·Σu²/(SSR/N), then refits the selected
order on T−k increments. The default upper bound is
min(floor(12·(T/100)^¼), floor((T−2)/2)). `lags` and `maxlag` cannot both be
supplied. `.attrs["lag_selection_results"]` retains every candidate.

The AR spectral variance is **s²=(SSR/(T−k))/(1−Σφj)²**, using GLS-detrended
data, not an OLS-detrended alternate variant and not residual degrees of freedom.
With I=T⁻²Σu(t−1)² and E=T⁻¹uT², MZa=(E−s²)/(2I), MSB=√(I/s²),
MZt=MZa·MSB. MPT=(c²I−cE)/s² for the constant and
(c²I+(1−c)E)/s² for the trend. `.attrs` records the horizon, GLS parameter,
selected order, AR observation count and spectral variance in normalized units.

| Case | Statistic | 1% | 5% | 10% |
| --- | --- | ---: | ---: | ---: |
| Constant | MZa | −13.8 | −8.1 | −5.7 |
| Constant | MZt | −2.58 | −1.98 | −1.62 |
| Constant | MSB | .174 | .233 | .275 |
| Constant | MPT | 1.78 | 3.17 | 4.45 |
| Trend | MZa | −23.8 | −17.3 | −14.2 |
| Trend | MZt | −3.42 | −2.91 | −2.62 |
| Trend | MSB | .143 | .168 | .185 |
| Trend | MPT | 4.03 | 5.48 | 6.67 |

## KSS nonlinear test

[Shin and Snell's original Edinburgh discussion paper 69](https://www.econ.ed.ac.uk/papers/id69_esedps.pdf)
derives the Taylor auxiliary regression and tabulates the three distributions
in Table 1 (p. 5). The final reference is
[Kapetanios, Shin and Snell (2003), Journal of Econometrics 112, 359–379](https://doi.org/10.1016/S0304-4076(02)00202-6).
The alternative is a globally stationary exponential smooth-transition
autoregressive (ESTAR) process. This procedure does not test arbitrary
nonlinearity or establish an ESTAR parameter estimate.

`trend="none"` uses raw levels and requires the zero-mean/vanishing initial
condition assumed by the raw reference distribution. `"constant"` demeans
over all input levels; `"trend"` first OLS demeans/detrends over all levels.
After transformation, regress Δz(t) on z(t−1)³ and `lags` lagged differences
with **no additional intercept or time trend**. The headline statistic is the
cubic coefficient's classical OLS t ratio with N−(lags+1) residual degrees
of freedom. Its reference distribution is nonstandard. `lags=0` is the
default; augmentation is fixed explicitly, with no automatic lag search.

Before cubing, levels are divided by a positive scale to avoid overflow. The
coefficient and its standard error in `.attrs` refer to these normalized units;
the t ratio is scale invariant. Raw KSS is **not** level-shift invariant.

| Transformation | 1% | 5% | 10% |
| --- | ---: | ---: | ---: |
| Raw | −2.82 | −2.22 | −1.92 |
| Demeaned | −3.48 | −2.93 | −2.66 |
| Detrended | −3.93 | −3.40 | −3.13 |

## Samples, resources and evidence

Input is resident DataFrame-compatible data, in row order or sorted by `time`.
Named time must contain complete consecutive integer periods. Datetimes are
refused: explicitly encode the intended frequency rather than silently
compressing calendar gaps. Missing values, duplicates, gaps, nonfinite data,
constant/deterministic unresolved series, singular fits and unresolved AR
spectral denominators raise structured `AnalysisError`s. No input is modified
and no rows are dropped. At least eight levels and positive regression degrees
of freedom are required; this numerical minimum is not a calibration guarantee.

`max_memory_mb=256`, also capped by the global workspace budget, admits
estimated selected-input/tensor/QR buffers before those blocks are built.
Initial coercion to a DataFrame, caller input, result Python objects, private
BLAS memory and allocator overhead are excluded; this is not a process RSS
limit. `max_work=100_000_000` limits the complete planned row-column QR work.
Refusal returns no partial test or partially selected lag.

The [independent SVD reference](../../scripts/unitroot_reference.py) refits
each lag separately and uses the original unnormalized observations. The
[validation script](../../scripts/validate_unitroot_methods.py) reproduces the
papers' simulation counts: 20,000 Wiener paths/5,000 steps for Ng–Perron;
100,000 normal random walks/T=1,000 for KSS. A separate declared 250-draw
finite-sample grid records unconditional rejection rates, failures and Wilson
intervals for iid and AR(.4) unit roots, and linear/ESTAR alternatives at
100/250 levels. It is a bounded diagnostic, not universal size/power validation.
See [the acceptance evidence](../evidence/unitroot-methods-2026-10-07/README.md).

MARKET-137 covers calibrated KSS; MARKET-136 retains the unresolved Ng–Perron
calibration. Parent MARKET-125 retains
the other break, Fourier, cointegration and causality methods as open scope.
