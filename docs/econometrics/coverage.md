# Coverage matrix: Stata, SPSS and EViews procedures in OpenEconometrics

<!-- BEGIN source-generated capability scope -->
Current source: **157 registered fit names**, **92 Dataset fit routes**, **74 common saved predict/margins adapters**. The [generated method/option inventory](../capabilities.md) states conditions, exclusions and devices.

Fit routes, common prediction adapters and family-specific helpers/forecasts have separate contracts. Source implementation does not establish independent scientific validation, installed-package verification or public shipment for a method/option. Those require their own dated, source-pinned evidence; historical measurements retain their original scope.
<!-- END source-generated capability scope -->

Implemented estimators run in OpenEconometrics itself on float64 PyTorch tensors; no
third-party estimation library runs at fit time. In this historical family
overview, **done** denotes implementation of the stated domain, **partial**
denotes explicitly excluded options, and **in progress** / **planned** denote
remaining work. These labels do not establish independent scientific, installed
or public-release coverage. Each family has a guide under `docs/econometrics/`;
any validation claim belongs to its identified method, options and dated receipt. The [source-generated current inventory](../capabilities.md) supplies versioned estimator/Dataset/prediction counts, exact option records and device contracts; the hand-written family rows below describe bounded method domains.

Parity notes: `provenance["stata_parity_validated"]` stays `False` until a
result has been compared with real Stata/SPSS/EViews output. Conventions
(small-sample factors, degrees of freedom, default covariance) follow the
documented Methods and formulas of the named command; deviations are listed in
each family guide.

## Wave 1 — linear, panel, instrumental variables, GLM

| OpenEconometrics | Stata | SPSS | EViews | Status |
| --- | --- | --- | --- | --- |
| `oe.ols` (weights, HC0–HC3, multiway cluster, HAC, bootstrap/jackknife; post-estimation `oe.test`, `oe.lincom`, `oe.nlcom`, `oe.predict`, `oe.margins`) — `openecon.linear_ols` | `regress`, `test`, `lincom`, `nlcom`, `margins` | REGRESSION, WLS | LS with White/HAC/cluster | done (separate package) |
| `oe.regress`, `oe.newey` — Stata-named wrappers over `oe.ols` | `regress`, `newey` | — | LS (HAC) | done |
| `oe.areg` | `areg` | — | — | done |
| `oe.reghdfe` (multiway fixed effects, singletons, 1–2 way cluster) | `reghdfe` | — | — | done |
| `oe.cnsreg` (linear equality constraints) | `cnsreg` | — | LS with restrictions | done |
| `oe.xtreg` fe / re / be / fd / mle / pooled, Driscoll–Kraay | `xtreg`, `xtscc` | — | Pool / panel LS, fixed and random effects | done |
| `oe.hausman`, Breusch–Pagan LM (in `xtreg` re results) | `hausman`, `xttest0` | — | Hausman test | done |
| `oe.xtserial`, `oe.xttest3` — [panel residual diagnostics](panel_diagnostics.md) | `xtserial`, `xttest3` (community commands) | — | Panel residual diagnostics | done (independent matrix oracles; no live Stata parity) |
| `oe.xtcd` — [supplied-residual Pesaran CD for balanced/unbalanced panels; BP LM and scaled LM for balanced panels](panel_dependence.md); bounded tensor workspace | Community cross-section dependence tests | — | Cross-section dependence diagnostics | done for documented raw-residual contracts (independent pairwise oracles; no live Stata parity) |
| `oe.xthst` — [Pesaran–Yamagata slope homogeneity](panel_homogeneity.md), original/adjusted delta statistics with bounded tensor workspace | Community slope homogeneity tests | — | Slope homogeneity diagnostics | done for documented static exogenous-regressor contracts (independent matrix oracles; asymptotic inference, no live Stata parity) |
| `oe.xtfmb` (Fama–MacBeth) | `xtfmb` | — | — | done |
| `oe.ivregress` 2sls / liml / gmm with first-stage, overid, endogeneity tests | `ivregress`, `estat firststage/overid/endogenous`, `ivreg2` | 2SLS | TSLS, GMM, LIML | done |
| `oe.xtivreg`, `oe.ivreghdfe` | `xtivreg`, `ivreghdfe` | — | Panel TSLS | done |
| `oe.glm` (families, links, scale, offset) | `glm` | GENLIN | GLM | done |
| `oe.poisson`, `oe.nbreg`, `oe.cloglog`, `oe.fracreg`, `oe.betareg` | same names | GENLIN (Poisson, NB) | Count models | done |
| `oe.ppmlhdfe` | `ppmlhdfe` | — | — | done |

## Wave 2 — discrete choice, limited dependent variables, quantile, robust

| OpenEconometrics | Stata | SPSS | EViews | Status |
| --- | --- | --- | --- | --- |
| `oe.ologit`, `oe.oprobit` | `ologit`, `oprobit` | PLUM (ordinal regression) | Ordered logit/probit | done |
| `oe.mlogit` | `mlogit` | NOMREG (multinomial logistic) | — | done |
| `oe.clogit` (= `xtlogit, fe`) | `clogit`, `xtlogit fe` | COXREG-style conditional logit | — | done |
| `oe.hetprobit`, `oe.biprobit` | `hetprobit`, `biprobit` | — | — | done |
| `oe.tobit`, `oe.truncreg`, `oe.intreg` | same names | — | Censored / truncated regression | done |
| `oe.heckman` (ML and two-step), `oe.heckprobit` | `heckman`, `heckprobit` | — | Heckman selection | done |
| `oe.zip`, `oe.zinb`, `oe.tpoisson`, `oe.tnbreg`, `oe.churdle`, `oe.hurdle`, `oe.gnbreg` | `zip`, `zinb`, `tpoisson`, `tnbreg`, `churdle`, `gnbreg` | — | Count models (ZIP, NB) | done |
| `oe.cpoisson`, `oe.cnbreg` (NB2/NB1) — [left/right and inclusive interval count censoring](censored_count.md), fixed/per-observation limits, offset/exposure and likelihood covariances | `cpoisson`; censored NB is an OpenEconometrics extension | — | Censored count models | done for documented numerical domain (independent likelihood/derivative oracles; no live Stata parity) |
| `oe.qreg`, `oe.bsqreg`, `oe.sqreg`, `oe.iqreg` | `qreg`, `bsqreg`, `sqreg`, `iqreg` | Quantile regression | Quantile regression (QREG) | done |
| `oe.rreg` (Huber then biweight IRLS; S/MM estimation not included) | `rreg` | — | ROBUSTLS (M) | done |
| `oe.nl` nonlinear least squares (symbolic Jacobian) | `nl` | NLR | NLS | done |
| `oe.ivprobit`, `oe.ivtobit` (ML and Newey two-step) | `ivprobit`, `ivtobit` | — | — | done |

## Wave 3 — time series (EViews core)

| OpenEconometrics | Stata | SPSS | EViews | Status |
| --- | --- | --- | --- | --- |
| `oe.arima` (ARIMA, SARIMA, ARMAX; exact ML via Kalman filter, CSS), `oe.forecast`, `oe.corrgram`, `oe.wntestq`, `oe.jarque_bera`, `oe.archlm`, `oe.tssmooth` | `arima`, `predict`, `corrgram`, `wntestq`, `tssmooth` | ARIMA (Forecasting), Exponential smoothing | ARMA / ARIMA, correlogram | done |
| `oe.arch` (ARCH, GARCH, EGARCH, GJR/TARCH, PARCH, IGARCH, ARCH-M; normal, t, GED), `oe.arch_forecast` | `arch` | — | ARCH family | done |
| `oe.var` (+ IRF orthogonalized/generalized, FEVD, Granger, stability), `oe.varsoc`, `oe.irf`, `oe.var_forecast` | `var`, `irf`, `vargranger`, `varsoc` | — | VAR | done |
| `oe.tycausality` — [Toda–Yamamoto augmented levels VAR](toda_yamamoto.md), ML residual covariance and restrictions on the base lags only | Community augmented-VAR Wald procedure | — | Toda–Yamamoto causality | supported for declared base/dmax orders and full-rank complete samples; asymptotic chi-square, no bootstrap/HAC or automatic order selection |
| `oe.bccaustest` — [Breitung–Candelon frequency restrictions](frequency_panel_causality.md) | Community frequency-domain noncausality | — | Frequency noncausality | stationary bivariate fixed p >= 3, interior pointwise frequencies, df-corrected approximate F; other integration/endpoint/HAC domains unavailable |
| `oe.dhcausality` — [Dumitrescu–Hurlin heterogeneous panel](frequency_panel_causality.md) | Community `xtgcause` (upper-tail convention differs) | — | Panel noncausality | balanced stationary common fixed K, cross-section independent iid Gaussian innovations, approximate Ztilde moments; no dependent bootstrap/automatic lag choice |
| `oe.vec`, `oe.vecrank` (Johansen), `oe.vec_forecast` | `vec`, `vecrank` | — | VEC, Johansen cointegration test | done |
| `oe.ardl` (lag selection, error-correction form, Pesaran–Shin–Smith bounds test) | `ardl` | — | ARDL, bounds test | done |
| `oe.nardl`, `oe.nardl_multipliers` — [partial sums, separately imposed long-run/lagwise short-run symmetry, constraint-aware QR/covariance/lag selection, delta-method or conditional recursive residual/wild bootstrap pointwise multipliers](nardl.md); specialized bounds calibration remains pending | Community NARDL | — | NARDL | partial (bootstrap conditions on observed predictors, initial values and selected lags; no general cointegration coverage claim) |
| `oe.dfuller`, `oe.pperron`, `oe.kpss`, `oe.dfgls`, `oe.zandrews` | `dfuller`, `pperron`, `kpss`, `dfgls`, `zandrews` | — | Unit root tests | done |
| `oe.egranger` (Phillips–Ouliaris not included: tables unavailable) | `egranger` | — | Cointegration tests | done |
| `oe.prais` (Prais–Winsten, Cochrane–Orcutt, six rho estimators) | `prais` | — | AR(1) LS | done |
| `oe.tsfilter` (HP, Baxter–King, Christiano–Fitzgerald, Hamilton) | `tsfilter` | — | Hodrick–Prescott, band-pass filters | done |
| `oe.tssmooth` (simple, Holt, Holt–Winters additive/multiplicative) | `tssmooth` | Exponential smoothing | ETS / smoothing | done |
| `oe.ucm` unobserved components (level, trend, seasonal, cycle; exact diffuse Kalman filter and smoother), `oe.ucm_components`, `oe.ucm_forecast` | `ucm`, `sspace` | — | State space | done |
| `oe.mswitch` Markov-switching (dynamic regression / AR), `oe.mswitch_probabilities`, `oe.threshold` | `mswitch`, `threshold` | — | Switching, threshold regression | done |
| Series diagnostics `oe.corrgram`, `oe.wntestq`, `oe.jarque_bera`, `oe.archlm`, `oe.chow`, `oe.cusum`, `oe.sktest`; OLS residual tests (Breusch–Godfrey, Breusch–Pagan/hettest, White, RESET, Durbin–Watson) as methods of `oe.ols` results | `wntestq`, `corrgram`, `sktest`, `estat bgodfrey/archlm/hettest/ovtest/dwatson`, `estat sbcusum` | — | Residual tests, stability tests | done |
| `oe.xtunitroot` (LLC, IPS, Fisher, Hadri, Breitung, Harris–Tzavalis), `oe.xtcointtest` (Kao; Pedroni not included), `oe.chow`, `oe.sbsingle`, `oe.cusum` | `xtunitroot`, `xtcointtest`, `estat sbknown/sbsingle/sbcusum` | — | Panel unit root / cointegration, Chow and CUSUM tests | done |
| `oe.xtcips`, `oe.xtunitroot(test="cips"/"cadf")` — [cross-sectionally augmented CADF, CIPS and CIPS*](cips.md), fixed/selected lags, original published quantiles | Community CADF/CIPS tests | — | Cross-sectionally augmented panel unit-root tests | done for complete balanced common-date panels; no exact p-values or quantile extrapolation |
| `oe.xtpanic` — [Bai–Ng PANIC](panic.md), fixed-factor dense PCA, idiosyncratic/common ADF, sequential MQc/MQf and opt-in independent pooling | Community PANIC procedure | — | PANIC | supported for complete balanced common-date panels and documented inference domains; trend-case idiosyncratic statistics only; automatic factor/lag selection, Pa/Pb and unbalanced panels remain unsupported |

## Wave 4 — dynamic panels, treatment effects, survival, systems, mixed models

| OpenEconometrics | Stata | SPSS | EViews | Status |
| --- | --- | --- | --- | --- |
| `oe.xtdpd`, `oe.xtabond`, `oe.xtdpdsys` (difference and system GMM, Windmeijer, AR tests, Sargan/Hansen) | `xtabond`, `xtdpdsys`, `xtabond2` | — | Panel GMM | done |
| `oe.ahreg` — [Anderson–Hsiao AR(1) IV](ahreg.md), level/difference instruments, panel CR1 and sparse differenced-error MA(1) covariance | Manual first-difference IV specification | — | Manual first-difference IV specification | done (independent matrix oracles; no live Stata parity) |
| `oe.xtlogit`, `oe.xtprobit`, `oe.xtpoisson` (fe / re / pa) | `xtlogit`, `xtprobit`, `xtpoisson` | — | — | done |
| `oe.mixed` (ML/REML, two nested levels, `oe.mixed_predict`), `oe.melogit`, `oe.meprobit`, `oe.mepoisson` (adaptive Gauss–Hermite) | `mixed`, `melogit`, `meprobit`, `mepoisson` | MIXED, GENLINMIXED | — | done |
| `oe.xtgee` (exchangeable, independent, AR(1), unstructured, stationary, nonstationary) | `xtgee` | GEE | — | done |
| `oe.teffects` ra / ipw / ipwra / aipw / nnmatch / psmatch | `teffects` | — | — | done |
| `oe.didregress`, `oe.eventstudy`, `oe.csdid` (Callaway–Sant'Anna), `oe.rdrobust`, `oe.rdplot` | `didregress`, `csdid`, `rdrobust`, `rdplot` | — | — | done |
| `oe.sts` (Kaplan–Meier, Nelson–Aalen, log-rank family), `oe.stcox` (+ PH tests, `oe.stcurve`), `oe.streg`, `oe.ltable` | `sts`, `stcox`, `estat phtest`, `streg`, `ltable` | KM, COXREG, SURVIVAL | — | done |
| `oe.sureg`, `oe.reg3`, `oe.mvreg` | `sureg`, `reg3`, `mvreg` | — | System estimation (SUR, 3SLS) | done |
| `oe.frontier` (half-normal, exponential, truncated normal), `oe.frontier_efficiency`; `oe.xtgls`, `oe.xtpcse` | `frontier`, `xtgls`, `xtpcse` | — | — | done |
| `oe.gmm` (nonlinear moment conditions, symbolic Jacobians, Hansen J) | `gmm` | — | GMM | done |

## Wave 5 — classical statistics (SPSS core)

| OpenEconometrics | Stata | SPSS | EViews | Status |
| --- | --- | --- | --- | --- |
| `oe.ttest` one-sample / paired / independent (pooled and Welch, Levene, Cohen's d / Hedges' g / Glass's delta), `oe.sdtest` | `ttest`, `sdtest`, `robvar` | T-TEST | Equality tests | done |
| `oe.oneway` with post hoc (Tukey, Bonferroni, Šidák, Scheffé, LSD, Games–Howell, Dunnett, Holm), Welch and Brown–Forsythe, effect sizes | `oneway`, `pwmean` | ONEWAY | — | done |
| `oe.anova` factorial / ANCOVA (Type I–III SS, marginal means), `oe.rm_anova` (Mauchly, Greenhouse–Geisser, Huynh–Feldt) | `anova` | UNIANOVA, GLM repeated measures | — | done |
| `oe.manova` (Wilks, Pillai, Hotelling–Lawley, Roy, Box's M) | `manova` | GLM multivariate | — | done |
| `oe.correlate` (Pearson, Spearman, Kendall tau-b), `oe.pcorr`, `oe.describe` | `correlate`, `pwcorr`, `pcorr`, `spearman`, `ktau`, `summarize` | CORRELATIONS, PARTIAL CORR, DESCRIPTIVES | — | done |
| `oe.ranksum`, `oe.kwallis` (Dunn post hoc), `oe.median_test`, `oe.jonckheere` | `ranksum`, `kwallis`, `median` | NPAR TESTS (M-W, K-W, median, J-T) | — | done |
| `oe.signrank`, `oe.signtest`, `oe.mcnemar`, `oe.symmetry`, `oe.friedman`, `oe.cochran_q` | `signrank`, `signtest`, `mcc`, `symmetry` | NPAR TESTS (Wilcoxon, sign, McNemar, Friedman, Cochran) | — | done |
| `oe.ksmirnov`, `oe.swilk`, `oe.sfrancia`, `oe.sktest`, `oe.runtest`, `oe.bitest`, `oe.prtest`, `oe.chi2gof` | `ksmirnov`, `swilk`, `sfrancia`, `sktest`, `runtest`, `bitest`, `prtest` | NPAR TESTS (K-S, runs, binomial, chi-square), EXAMINE | — | done |
| `oe.crosstab` (chi-square, likelihood ratio, Fisher exact, association measures, kappa, odds ratio, CMH), `oe.tabulate` | `tabulate`, `kap`, `cc` | CROSSTABS, FREQUENCIES | — | done |
| `oe.roc`, `oe.roccomp` (DeLong) | `roctab`, `roccomp` | ROC | — | done |
| `oe.alpha` (Cronbach), item statistics | `alpha` | RELIABILITY | — | done |
| `oe.factor`, `oe.pca`, rotations (varimax, promax, oblimin), scores, KMO/Bartlett | `factor`, `pca`, `rotate`, `estat kmo` | FACTOR | Factor analysis | done |
| `oe.cluster_kmeans`, `oe.cluster_hierarchical` (Ward, average, complete, centroid) | `cluster` | QUICK CLUSTER, CLUSTER | — | done |
| `oe.discrim` LDA / QDA, `oe.canon`, `oe.mds`, `oe.ca` | `discrim`, `canon`, `mds`, `ca` | DISCRIMINANT, CANCORR, PROXSCAL, CORRESPONDENCE | — | done |
| `oe.stepwise` (forward/backward/stepwise), `oe.collin` (VIF, condition indices), `oe.curvefit`, `oe.tabstat` | `stepwise`, `estat vif`, `tabstat` | REGRESSION (STEPWISE, COLLIN), CURVEFIT, MEANS | — | done |

## Wave 6 — shared post-estimation for supported result targets

| OpenEconometrics | Stata | Status |
| --- | --- | --- |
| `oe.test`, `oe.testparm`, `oe.lincom`, `oe.nlcom` (delta method) — [all fitted registry results with valid reporting covariance/inference](inference.md), including JSON-restored results; existing OLS contrast state retained | `test`, `testparm`, `lincom`, `nlcom` | done |
| `oe.margins` (AME, MEM, at values), `oe.predict` — [current saved adapters and target restrictions](../capabilities.md), including scalar means, ordered/multinomial probabilities and heteroskedastic-probit scale/effects; native OLS behavior retained | `margins`, `predict` | implemented for documented adapters; remaining common adapters listed in the generated inventory |
| `oe.lrtest`, `oe.estat_ic`, `oe.bootstrap`, `oe.jackknife`; `oe.suest` restricted to its implemented score providers (ols, logit, probit, poisson, ologit, oprobit, mlogit) | `lrtest`, `estat ic`, `bootstrap`, `jackknife`, `suest` | done for supported inputs; broader `suest` planned |
| `oe.fcast_eval` (RMSE, MAE, MAPE, sMAPE, Theil U1/U2, MSE decomposition, MASE), `oe.dm_test` (Diebold–Mariano with HLN correction); `oe.forecast` for arima, arch, var, vec, ucm | — (EViews Forecast evaluation) | done |
