# Saved OLS spatial diagnostics

`oe.spatial_diagnostics` evaluates eight distinct diagnostic targets from a
JSON-restored, resident, numeric, unweighted OLS result and its exact original
data. It reuses the saved coefficients. It does not estimate another OLS model
or treat regression residuals as exchangeable observations.

```python
fit = oe.ols(data=frame, y="y", x=["x1", "x2"], covariance="nonrobust")
saved = oe.ResultBundle.model_validate_json(fit.model_dump_json())
diagnostics = oe.spatial_diagnostics(
    saved, data=frame, key="region", spatial_weights=W,
    tests=["moran_normal", "moran_gaussian_mc", "lm_error", "lm_lag",
           "robust_lm_error", "robust_lm_lag", "lm_joint", "wx_f"],
    wx_predictors=["x1", "x2"], simulations=999, seed=27,
)
state = oe.summary_state(diagnostics)
restored = oe.restore_summary(state)
for name, index_names in restored.attrs["table_index_names"].items():
    restored[name].index.names = index_names
```

The input must match the original fitted model columns and physical row order.
Missing observations already excluded by that OLS fit remain excluded. The
caller declares the association between physical rows and unique spatial keys:
the original OLS fit never saw a spatial key or graph. The new diagnostic result
records this association and both original and induced effective graph hashes.
The original model hash excludes pandas index labels; labels and index names
are retained from this diagnostic call, not authenticated as fit-time identity.
No missing-unit graph or replacement sample is inferred.

All tests condition on a fixed design and fixed exogenous graph, under an iid
Gaussian homoskedastic error null. The following inference laws are different.

| Target | Null statistic and reference |
| --- | --- |
| `moran_normal` | Projected-residual Moran I, exact Gaussian moments and an approximate normal tail |
| `moran_gaussian_mc` | Fixed-design projected Gaussian simulation, complete null statistics and an inclusive plus-one tail |
| `lm_error` | Spatial error Rao score; asymptotic chi-square with one degree of freedom |
| `lm_lag` | Missing endogenous spatial lag Rao score; asymptotic chi-square with one degree of freedom |
| `robust_lm_error` | Error score adjusted for local competing lag dependence |
| `robust_lm_lag` | Lag score adjusted for local competing error dependence |
| `lm_joint` | Joint lag/error score and full two-score information; asymptotic chi-square with two degrees of freedom |
| `wx_f` | Added, declared spatially lagged exogenous covariates; finite Gaussian nested-model F |

“Robust” here refers to local competing spatial misspecification. It does not
provide heteroskedastic, clustered, survey or general distributional robustness.
The score targets have asymptotic references; they are not finite-sample tests.
Each p-value belongs to its declared test; the eight-target display does not
apply a simultaneous multiple-testing correction.

## Projection and score contracts

With QR design basis Q, use M=I-QQ', r=N-P and S=(W+W')/2. Moran uses
I=(N/sum(W)) e'Se/(e'e). Its Gaussian expectation is
(N/sum(W)) tr(MS)/r and variance is
(N/sum(W))^2 2[tr(MSMS)-tr(MS)^2/r]/[r(r+2)]. The simulation target projects
private seeded standard normal draws with M; its two-sided tail is centered
at that design-specific expectation. Permuting observed fitted residuals is
not this null experiment. [Cliff–Ord implementation](https://github.com/r-spatial/spdep/blob/main/R/lm.morantest.R)

The score tests use sigma_ML^2=e'e/N, a=e'We/sigma_ML^2, b=e'Wy/sigma_ML^2,
T=tr(W'W+W^2), h=||MW fitted||^2/sigma_ML^2 and D=T+h. The joint efficient
information is [[D,T],[T,T]] for [lag,error]. Computing h directly avoids
subtracting nearly equal D and T. Locally adjusted/joint targets require
identified positive h; a row-normalized intercept-only fit generally cannot
identify both lag and error. [Anselin et al. (1996)](https://doi.org/10.1016/0166-0462(95)02111-6),
[author-maintained score formulas](https://github.com/r-spatial/spdep/blob/main/R/lm.RStests.R).

The WX target adds only declared slope columns, excluding a lagged intercept.
Every requested projected direction must be independent; redundant directions
are refused rather than silently removed. The numerator degrees of freedom
equal their count and the denominator is N-P-count. This is an exogenous mean
specification test, distinct from the endogenous Wy and spatial-error scores.
[Nested linear model F comparison](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/anova.lm.html).

## Admission and persistence

The bounded route is resident CPU float64 with at most 512 original and effective observations
and 32 saved design parameters. Complete source/sample identity, rank, saved
coefficient/covariance consistency, work and workspace admission precede
evaluation. Unresolved numerical geometry, degenerate Moran variance, an
unidentified requested score target or an invalid requested WX direction
produces a structured error. Select an identified subset explicitly when a
complete eight-target request is unavailable.

Weighted, robust/cluster covariance, categorical/formula, panel/time and Dataset
results have no route in this contract. Spatial-panel/IV/GMM diagnostics and
licensed cross-vendor comparisons remain separate work. The existing raw-variable
`oe.moran`, spatial fits, impacts and saved mean targets are preserved.

Complete result tables, null simulations, source model state and diagnostic
settings are retained in `oe.summary_state`, independently of editor previews.
The shared table serializer retains index values; this result's
`table_index_names` metadata preserves names for the explicit restoration above.
The delivery evidence distinguishes source numerical checks, installed wheel,
frozen runtime and actual native Run/true Quit/relaunch persistence.
