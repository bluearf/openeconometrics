# Econometrics in OpenEconometrics

OpenEconometrics implements the estimation and testing procedures of Stata, SPSS and
EViews itself, on float64 PyTorch tensors. No third-party estimation library
(statsmodels, linearmodels, SciPy optimizers or distributions, scikit-learn)
runs when a model is fitted; those packages appear only in the test suite, as
independent oracles.

Every estimator is reached as `oe.<name>(...)` with Stata-style keyword
arguments, or through the lower-level `oe.fit(oe.ModelSpec(...), data=...)`.
Model fits return a `ResultBundle` (coefficients, covariance, metrics,
specification tests, provenance) that prints with `.summary()` and exports
with `.to_latex()`. Test procedures (t tests, ANOVA, unit-root tests,
cross-tabulations, factor analysis) return tables.

```python
import openecon as oe

fe = oe.xtreg(data=df, y="wage", x=["educ", "exper"], panel="id", time="year")
re = oe.xtreg(data=df, y="wage", x=["educ", "exper"], panel="id", time="year", model="re")
oe.hausman(fe, re)

oe.ivregress(data=df, y="lwage", x=["exper"], endog=["educ"], instruments=["fatheduc", "motheduc"])
oe.dfuller(df, "gdp", time="quarter", lags=4, trend="trend")
oe.oneway(df, "score", "group", posthoc=["tukey", "games_howell"])
```

`oe.capabilities()` lists every estimator with its options, column roles,
covariance estimators and weight types. [coverage.md](coverage.md) maps each
OpenEconometrics function to its Stata, SPSS and EViews counterpart.

## Family guides

Each guide gives the model, the estimator and its formulas, the covariance
conventions and their sources, what the result reports, the equivalent
Stata/SPSS/EViews commands, worked examples, limitations, and the conventions
that could not be confirmed.

### Regression and panel data

| Guide | Functions |
| --- | --- |
| [linear.md](linear.md) | `areg`, `reghdfe`, `cnsreg`; `regress` and `newey` (wrappers over `oe.ols`) |
| [panel.md](panel.md) | `xtreg` (fe, re, be, fd, mle, pooled; Driscoll–Kraay), `hausman`, `xtfmb` |
| [panel-prediction-extensions.md](panel-prediction-extensions.md) | PLS1, `xtreg(model='cre')`, Mundlak test, `xthtaylor`, reverse/calendar panel windows and `rolling_predict` |
| [panel_diagnostics.md](panel_diagnostics.md), [panel_dependence.md](panel_dependence.md) | `xtserial`, `xttest3`; `xtcd` (Pesaran CD, balanced BP LM/scaled LM) |
| [panel_homogeneity.md](panel_homogeneity.md) | `xthst` (Pesaran–Yamagata slope homogeneity, balanced/unbalanced static panels) |
| [iv.md](iv.md) | `ivregress` (2SLS, LIML, GMM; first-stage, overidentification, endogeneity tests), `xtivreg`, `ivreghdfe` |
| [dpanel.md](dpanel.md) | `xtdpd`, `xtabond`, `xtdpdsys` (difference and system GMM) |
| [systems.md](systems.md) | `sureg`, `mvreg`, `reg3`, `gmm`, `frontier`, `xtgls`, `xtpcse` |
| [quantile.md](quantile.md) | `qreg`, `bsqreg`, `sqreg`, `iqreg`, `rreg`, `nl` |

### Limited and categorical outcomes

| Guide | Functions |
| --- | --- |
| [glm.md](glm.md) | `glm`, `poisson`, `nbreg`, `cloglog`, `fracreg`, `betareg`, `ppmlhdfe` |
| [discrete.md](discrete.md) | `ologit`, `oprobit`, `mlogit`, `clogit`, `hetprobit`, `biprobit` |
| [nested-choice.md](nested-choice.md) | `nlogit`, `nlogit_restore`, `nlogit_predict`, `nlogit_margins` |
| [limited.md](limited.md) | `tobit`, `truncreg`, `intreg`, `heckman`, `heckprobit`, `ivprobit`, `ivtobit` |
| [count.md](count.md) | `zip`, `zinb`, `tpoisson`, `tnbreg`, `churdle`, `hurdle`, `gnbreg` |
| [mixed.md](mixed.md) | `mixed`, `melogit`, `meprobit`, `mepoisson`, `xtlogit`, `xtprobit`, `xtpoisson`, `xtgee` |
| [survival.md](survival.md) | `sts`, `stcox`, `stcurve`, `streg`, `ltable` |
| [teffects.md](teffects.md) | `teffects` (ra, ipw, ipwra, aipw, nnmatch, psmatch), `didregress`, `eventstudy`, `csdid`, `rdrobust`, `rdplot` |

### Time series

| Guide | Functions |
| --- | --- |
| [arima.md](arima.md) | `arima`, `forecast`, `corrgram`, `wntestq`, `jarque_bera`, `archlm`, `tssmooth` |
| [arch.md](arch.md) | `arch` (ARCH, GARCH, EGARCH, GJR, PARCH, IGARCH, ARCH-M), `arch_forecast` |
| [var.md](var.md) | `var`, `varsoc`, `irf`, `var_forecast`, `vecrank`, `vec`, `vec_forecast` |
| [unitroot.md](unitroot.md) | `dfuller`, `pperron`, `dfgls`, `kpss`, `zandrews`, `egranger`, `chow`, `sbsingle`, `cusum`, `xtunitroot`, `xtcointtest` |
| [cips.md](cips.md) | `xtcips` (individual CADF, CIPS and CIPS*, fixed/selected lags and published quantiles) |
| [panic.md](panic.md) | `xtpanic` (Bai–Ng fixed-factor common/idiosyncratic decomposition, component ADF and MQc/MQf tests) |
| [toda_yamamoto.md](toda_yamamoto.md) | `tycausality` (augmented levels VAR; first base-lag restrictions, asymptotic ML-covariance Wald tests) |
| [tsmodels.md](tsmodels.md) | `prais`, `ardl`, `tsfilter`, `ucm`, `mswitch`, `threshold` |
| [nardl.md](nardl.md) | `nardl`, `nardl_multipliers` (delta method and conditional recursive residual/wild bootstrap) |

### Classical statistics (SPSS)

| Guide | Functions |
| --- | --- |
| [stats.md](stats.md) | `ttest`, `sdtest`, `oneway`, `anova`, `rm_anova`, `manova`, `correlate`, `pcorr`, `describe` |
| [nonparametric.md](nonparametric.md) | rank tests, distribution tests, `crosstab`, `tabulate`, `roc`, `roccomp` |
| [multivariate.md](multivariate.md) | `pca`, `factor`, `alpha`, `cluster_kmeans`, `cluster_hierarchical`, `discrim`, `canon`, `mds`, `ca` |
| [selection.md](selection.md) | `stepwise`, `collin`, `curvefit`, `tabstat` |

### Post-estimation

| Guide | Functions |
| --- | --- |
| [postest.md](postest.md) | `lrtest`, `estat_ic`, `bootstrap`, `jackknife`, `suest`, `fcast_eval`, `dm_test` |
| [prediction.md](prediction.md) | Saved-result `predict` and `margins`: 21 adapters, including scalar means, ordered/multinomial probabilities and heteroskedastic-probit scale/effects |

`oe.test`, `oe.testparm`, `oe.lincom` and `oe.nlcom` accept saved `ResultBundle`
coefficients with valid fitted covariance and inference records. `oe.predict`
and `oe.margins` retain the native OLS methods and support the 21 saved-result
adapters listed in [prediction.md](prediction.md); other generic adapters remain
explicitly unsupported. Model-specific forecasts and predictions are separate.

## How the results were checked

Every family was built by one engineer and then checked by an independent one
who wrote separate oracles: explicit NumPy algebra, brute-force maximization of
independently written likelihoods, statsmodels and SciPy where the same model
and convention exist, invariances (frequency weights equal duplicated rows,
row order, rescaling), and adversarial inputs. Where published Stata output was
available — statsmodels' bundled Stata results, Stata manual examples, textbook
examples — it is reproduced in the tests; each guide says which.

`provenance["stata_parity_validated"]` remains `False` on every result: a
convention is only called confirmed when it was compared with real output, and
the guides list the ones that were not.

## Writing a new family

The [five capability milestones](capability-closures-2026-10-07.md) document
CRE/HT, interaction mediation/Fairlie, signed partial-sum tests, CUE/effective F
and extended window workflows, with separate numerical/runtime evidence.

The [research method stage plans](roadmaps/README.md) separate survey, specialized
quantile and advanced state-space proposals from implemented capabilities. The
[single-stage survey declaration](survey-design.md) validates design geometry
without providing survey estimates or inference.

See [development.md](development.md) for the registry contract, the shared
sample/design/result layer and the numerical rules.

The [deferred research plans](../research/README.md) define staged scope and
acceptance for latent SEM, Bayesian, supervised prediction and mixture families.
These are planning deliverables; their implementation trackers remain open.
