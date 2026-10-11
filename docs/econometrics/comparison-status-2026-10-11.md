# Stata / SPSS / EViews comparison: 11 October 2026 reconciliation

Source reviewed: `6eb04e2b9824ac230d14a034961d9b91521e98d1`
(11 October 2026). This document is a dated review; no vendor-parity claim follows.

The [generated capability inventory](../capabilities.md) is authoritative for
current registered names, options, devices, Dataset routes and common saved
prediction adapters. The reviewed baseline contains 171 fit names, 92 Dataset
fit routes, 74 common saved predict/margins adapters and 698 helper registrations.
Aliases, helpers, vendor commands and independent estimators have different
denominators. No completion percentage follows from these counts.

The [row index](comparison-rows.md) and its
[review manifest](comparison-row-contracts.json) preserve vendor table occurrences,
source citations and product boundaries. A row marked `needs-review` is an open
comparison task under [MARKET-832](https://linear.app/bluearf/issue/MARKET-832),
even when its method family has a working API. A family link is a navigation aid;
it does not certify all options in that vendor row. Unsupported and unreviewed
are different states. The six companion Stata research notes are supporting
sources; the manifest declares exactly which files and rows it covers.

## How to read each contract

Each family below names available APIs, a bounded domain, exclusions and a
canonical remaining issue. Its linked method guide supplies the detailed
**S**ample / missing-value policy, **W**eights and category alignment,
**V**ariance / full covariance / inference, **D**iagnostics / convergence,
**P**ostestimation and saved-result **R**eplay contract. These fields must be
checked for the exact row before changing its review state. A null direct API
mapping in the manifest makes no assertion that the statistical method is absent.

Evidence is recorded separately:

| Field | Meaning and acceptance boundary |
| --- | --- |
| Source | Registered implementation at the source pin; exact options come from the generated inventory and method guide. |
| Independent science | Only the analytical, published-table, synthetic or author/package reference actually identified in the linked guide/receipt. A guide link alone is not a new successful test run. |
| Wheel / frozen | Only a dated package or frozen-runtime receipt with its own source identity. |
| Installed UI | Only the recorded application/version, Run/export/Quit and relevant user flow. A frozen backend is insufficient. |
| Windows | Only a dated hosted Windows acceptance receipt for that artifact/source. |
| Persistence | Source JSON replay and installed cold reopening have distinct evidence. |
| Public release | Only the artifact and public source recorded in the [distribution status](../validation/distribution-status-2026-10-11.md). Current source APIs are not automatically in older installers or PyPI packages. |

Unlisted evidence layers are **not established by this comparison review**.
This document runs no new estimator/reference comparisons. A printed coefficient
match is limited by printed precision and does not prove full covariance,
calibration or optimizer acceptance. A nonzero reference convergence code remains
a failure, even if parameters agree. Historical receipts retain their original
date, source and failed cases.

## Vendor source boundaries

| Product / source | Reviewed reference scope | Reconciliation |
| --- | --- | --- |
| [Stata inventory](../../research_notes/Ekonometrik%20modeller%20ve%20Stata%20katalo%C4%9Fu/stata_official_inventory.md) | Stata 19 / StataNow research snapshot, official manuals and help citations retained | The historical 252-row count means 248 **non-prefix occurrences**, then 247 after the repeated `mediate` occurrence is removed; it never meant 248 distinct names plus four prefixes. This is a normalization of the recorded snapshot, not a fresh live command census. Community packages remain separately labelled. |
| [SPSS procedures](../../research_notes/Ekonometrik%20modeller%20Stata%20EViews%20SPSS%20katalo%C4%9Fu/spss_official_procedures.md) | IBM Statistics 32 procedures; each row retains its edition/module and CSR source | Product-bundled Python/R extensions are still extensions. IBM Statistics and Amos are separate products. |
| [SPSS extensions / Amos / macros](../../research_notes/Ekonometrik%20modeller%20Stata%20EViews%20SPSS%20katalo%C4%9Fu/spss_extensions_amos_gaps.md) | Extension manifests, companion products and macros | A matching name, wrapper or recalled feature is not executed vendor evidence. The table's `UNVERIFIED` markers remain visible. |
| [EViews single-equation / time series](../../research_notes/Ekonometrik%20modeller%20Stata%20EViews%20SPSS%20katalo%C4%9Fu/eviews_single_equation_time_series.md) | EViews 14 generally available; add-ins and programming distinguished | [EViews 15 public beta](https://register1.eviews.com/beta/) exists, checked 11 October 2026. The [main product page](https://www.eviews.com/) presents EViews 14. Beta features are a separate scope. |
| [EViews multivariate / panel / add-ins](../../research_notes/Ekonometrik%20modeller%20Stata%20EViews%20SPSS%20katalo%C4%9Fu/eviews_multivariate_panel_addins.md) | EViews 14 manuals and identified add-ins | The old “no EViews 15 announcement found” conclusion is superseded by the official beta page; it does not establish EViews 15 GA or public beta command documentation. |

The beta requires an existing EViews 14 Standard/Enterprise licence and is a
64-bit Windows prerelease. Its advertised features are not additional mandatory
acceptance gates for this project. No licensed Stata/SPSS/EViews executable
comparison is required under [AGENTS.md](../../AGENTS.md).

## Current family contracts and remaining work

<a id="regression"></a>
### Linear regression and diagnostics

`oe.ols`, `regress`, `newey`, `areg`, `reghdfe` and constrained regression are
implemented. [Linear contracts](linear.md) and [inference](inference.md) define
S/W/V/D/P/R, including estimator-specific HC/cluster/HAC corrections. This is not
arbitrary selection-aware or dependence-robust inference. Remaining common
targets: [MARKET-143](https://linear.app/bluearf/issue/MARKET-143).

<a id="weak-iv"></a>
### IV and weak identification

`oe.ivregress`, `jive`, `ivlewbel`, `shift_share_iv`, `weakivtest` and bounded wild
tests exist. [IV](iv.md), [extensions](iv-extensions.md),
[wild inference](iv-wild-bootstrap.md) and [CLR](weakiv-clr.md) record S/W/V/D/P/R
and the independent reference conventions. Robust/subvector inversion and wider
weight/many-instrument domains remain [MARKET-153](https://linear.app/bluearf/issue/MARKET-153).

<a id="glm"></a>
### GLM, counts and limited responses

`oe.glm`, Poisson/NB, fractional/beta, ordered/binary, censored/truncated and
selection estimators have bounded implementations. [GLM](glm.md), [counts](count.md),
[censored counts](censored_count.md), [limited responses](limited.md) and
[selection](selection.md) specify S/W/V/D/P/R. Tweedie, wider ordered-link/weight
and nonlinear response domains remain
[MARKET-177](https://linear.app/bluearf/issue/MARKET-177) and
[MARKET-189](https://linear.app/bluearf/issue/MARKET-189); supported families do not
imply every vendor link or censoring convention.

<a id="systems"></a>
### Simultaneous and nonlinear systems

`oe.sureg`, `reg3`, `mvreg`, `gmm` and `nl` exist.
[Systems](systems.md) and [nonlinear SUR](nonlinear-sur.md) specify equation/sample
alignment, full cross-equation covariance, diagnostics and saved output.
Nonlinear full-information and wider system inference remain
[MARKET-178](https://linear.app/bluearf/issue/MARKET-178).

<a id="panel"></a>
### Panel and dynamic panel

`oe.xtreg`, `xtivreg`, `xtgls`, `xtpcse`, `xtdpd`, `xtabond`, `xtdpdsys` and bounded
panel-response routes exist. [Panel](panel.md), [dynamic panel](dpanel.md) and
[panel prediction](panel-prediction-extensions.md) define S/W/V/D/P/R; panel
identity/time, gaps, singleton handling and instrument construction are material.
Wider nonlinear panel options remain [MARKET-189](https://linear.app/bluearf/issue/MARKET-189);
fixed-effect quantile targets remain [MARKET-180](https://linear.app/bluearf/issue/MARKET-180).

<a id="quantile"></a>
### Quantile and IV quantile

`oe.qreg`, `bsqreg`, `sqreg`, `iqreg` and scalar iid IVQR exist.
[Quantile](quantile.md) and [IVQR](ivquantile.md) define sample, rank/LP diagnostics,
covariance, inversion and replay. [MARKET-213](https://linear.app/bluearf/issue/MARKET-213)
is completed for its scalar iid domain. QARDL's author-point/KKT/inference
acceptance remains [MARKET-637](https://linear.app/bluearf/issue/MARKET-637) /
[MARKET-214](https://linear.app/bluearf/issue/MARKET-214); wider quantile-process,
QVAR/QTE research remains [MARKET-215](https://linear.app/bluearf/issue/MARKET-215).

<a id="exact"></a>
### Exact, Firth and robust regression

[Conditional regression](conditional-regression.md) documents one-target
Bernoulli/Poisson exact inference with up to three nuisance sufficient statistics;
[finite regression](finite-regression.md) and [robust regression](robust.md)
document fitted Firth/LTS and M-estimation domains, covariance, refusal and replay.
Arbitrary joint-vector exact targets, broader weighted/categorical geometry and
FAST-LTS remain [MARKET-176](https://linear.app/bluearf/issue/MARKET-176). Multivariable
conditional regression is not wholly absent.

<a id="causal"></a>
### Treatment effects, DiD, RD and sensitivity

`oe.teffects`, DiD/event-study/synthetic-control, RD and bounded causal inference
helpers exist. [Treatment effects](teffects.md), [modern DiD](modern-did.md),
[causal learning](causal-learning.md) and [causal sensitivity](causal-sensitivity.md)
record S/W/V/D/P/R and each identification/sampling law. Switching/multivalue/DDD,
advanced RD and broader sensitivity targets remain
[MARKET-166](https://linear.app/bluearf/issue/MARKET-166),
[MARKET-181](https://linear.app/bluearf/issue/MARKET-181),
[MARKET-187](https://linear.app/bluearf/issue/MARKET-187).

<a id="mediation"></a>
<a id="control-function"></a>
### Mediation and control functions

[Binary mediation](binary-mediation.md) and [control functions](control-functions.md)
define implemented decompositions and continuous-OLS first stages, generated-control
covariance, sample alignment, structural targets and replay. Mixed first stages
and generalized CF scientific acceptance remain
[MARKET-731](https://linear.app/bluearf/issue/MARKET-731) /
[MARKET-732](https://linear.app/bluearf/issue/MARKET-732), under
[MARKET-168](https://linear.app/bluearf/issue/MARKET-168). Wider mediation remains
[MARKET-167](https://linear.app/bluearf/issue/MARKET-167). Failed cluster-Wald
calibration is not accepted by coefficient agreement.

<a id="choice"></a>
### Choice and scored conjoint

`oe.mprobit`, strict full/partial rank logit and two-level disjoint nested logit
are implemented. [MProbit](mprobit.md), [ranked choice](rank-ordered-choice.md),
[nested choice](nested-choice.md) define category/rank/choice-set alignment,
full covariance, diagnostics, saved targets and replay. Four-plus alternatives,
cross/deeper nesting, panel random coefficients, ties and broader weights remain
[MARKET-164](https://linear.app/bluearf/issue/MARKET-164).
[Conjoint](conjoint.md) already includes scored full profiles, importance/share
simulation **and bootstrap uncertainty**. Ranked/CBC, mixed-level orthogonal
arrays, broader dependence/survey weights, actual-market calibration and joint
BCa targets remain [MARKET-173](https://linear.app/bluearf/issue/MARKET-173).

<a id="conjoint"></a>
Conjoint's sample, respondent/profile weights, utility and importance/share
uncertainty, holdout diagnostics and portable replay are specified separately in
[the conjoint guide](conjoint.md); the choice estimators above do not establish
CBC/ranked conjoint acceptance.

<a id="survival"></a>
### Survival

KM/Cox, parametric/frailty/recurrent survival and covariate interval Weibull exist.
[Survival](survival.md), [frailty](survival-frailty.md) and
[interval Weibull](interval-weibull-covariate.md) define time/censoring/sample
alignment, OIM/full covariance, convergence, prediction and saved replay.
[MARKET-689](https://linear.app/bluearf/issue/MARKET-689) is completed;
its dated evidence (internal evidence excluded from this public snapshot)
does not establish every later binary. Fine–Gray/Gray original acceptance remains
[MARKET-684](https://linear.app/bluearf/issue/MARKET-684); interval Cox and wider
dependence/weight scope remains [MARKET-163](https://linear.app/bluearf/issue/MARKET-163).

<a id="spatial"></a>
### Spatial

Cross-sectional SAR, SEM, SDM, SAC and spatial IV plus diagnostics exist.
[Spatial](spatial.md) and [spatial diagnostics](spatial-diagnostics.md) specify
matrix/sample alignment, identification, covariance and saved targets.
Spatial panel FE/RE/IV/GMM and wider weights/categories remain
[MARKET-165](https://linear.app/bluearf/issue/MARKET-165).
`oe.sem` means **spatial error**; structural SEM uses `latent_sem`.

<a id="mixed"></a>
### Mixed and repeated residual covariance

`oe.mixed`, bounded KR/Satterthwaite contrasts, `mixedresid`, ordinal/censored
random-intercept models and repeated GLS/ML/REML exist.
[Mixed](mixed.md), [extensions](mixed-extensions.md), [repeated residual GLS](repeated-residual-gls.md)
and [extended mixed](extended-mixed.md) define grouping/missing/weight alignment,
full covariance and degrees of freedom, diagnostics and replay.
Broader covariance combinations, KR calibration, random slopes and GLMM integration
domains remain [MARKET-152](https://linear.app/bluearf/issue/MARKET-152).

<a id="sem"></a>
### CFA and structural SEM

`oe.cfa` and single-group continuous Gaussian `oe.latent_sem` exist.
[CFA](cfa.md) and [latent SEM](latent-sem.md) define identification, complete-case
sample, covariance, fit diagnostics, prediction and persistence.
Multigroup original acceptance remains [MARKET-733](https://linear.app/bluearf/issue/MARKET-733);
FIML, robust/GSEM and multilevel scope remains
[MARKET-360](https://linear.app/bluearf/issue/MARKET-360). Amos availability is
separate from Statistics availability and from OpenEconometrics acceptance.

<a id="bayes"></a>
### Bayesian

[Proper conjugate Gaussian](bayesian-conjugate.md),
[posterior hypotheses](bayesian-hypothesis.md), [BMA](bayesian-bma.md) and
[BVAR](../bayesian_var_conjugate.md) exist. Their guides specify proper priors,
sample restrictions, posterior full covariance/draw law, diagnostics, forecasts,
model probabilities and portable replay. BMA has at most eight optional regressors;
BVAR is a bounded proper MNIW model, not a TVP/SV/hierarchical VAR.
[MARKET-690](https://linear.app/bluearf/issue/MARKET-690) and
[MARKET-734](https://linear.app/bluearf/issue/MARKET-734) remain completed.
General HMC/MCMC acceptance remains [MARKET-685](https://linear.app/bluearf/issue/MARKET-685)
under [MARKET-362](https://linear.app/bluearf/issue/MARKET-362); local branch work is
not a merged public API.

<a id="irt"></a>
### IRT and reliability

[Six fitted IRT families](irt.md) cover Rasch/2PL/fixed-known-guessing 3PL/GRM/PCM/RSM;
[measurement reliability](measurement-reliability.md) and
[calibrated banks](irt-calibrated-banks.md) are separate contracts. Person/item
missing/category alignment, information/covariance, convergence, scoring and
replay follow those guides. A supplied GPCM/nominal bank's scoring is not its
parameter estimation. Free guessing, estimated common discrimination,
multidimensional/multigroup invariance/DIF and wider weights remain
[MARKET-170](https://linear.app/bluearf/issue/MARKET-170).

<a id="mixture"></a>
### Finite mixtures and latent classes

[Fixed-K Gaussian mixture](finite-mixture.md) exists with its documented sample,
identification, covariance, restart/label diagnostics and replay. Binary/Poisson
mixtures, membership covariates, polytomous latent classes and nonregular K
comparison remain [MARKET-364](https://linear.app/bluearf/issue/MARKET-364).

<a id="categorical"></a>
### Optimal scaling and categorical statistics

CATREG/CATPCA/MCA/OVERALS/loglinear and bounded frequency-weighted scaling,
rotation/spline/bootstrap are implemented. [Categorical](categorical.md),
[frequency](categorical-frequency.md), [spline/weights](categorical-spline-weighted-options.md)
and [sampling/selection](categorical-sampling-selection.md) specify category,
frequency, full covariance/uncertainty, rank diagnostics, transformations and replay.
Higher-degree/adaptive knots, analytic/survey weights and broader dependence
remain [MARKET-171](https://linear.app/bluearf/issue/MARKET-171).

<a id="multivariate"></a>
<a id="classical"></a>
### Classical and multivariate statistics

Tests/ANOVA/MANOVA/PCA/EFA/CCA/clustering/discriminant procedures exist.
[Classical statistics](stats.md), [multivariate](multivariate.md),
[selection/contrasts](multivariate-selection-contrasts.md),
[weight/matrix](multivariate-weight-matrix.md) and
[spectral uncertainty](spectral-geometry-uncertainty.md) specify S/W/V/D/P/R.
Selected-model, multifactored/adaptive and broader extraction/rotation/weight
domains remain [MARKET-185](https://linear.app/bluearf/issue/MARKET-185).
The separately owned [MARKET-824](https://linear.app/bluearf/issue/MARKET-824)–
[831](https://linear.app/bluearf/issue/MARKET-831) work is not accepted here.

<a id="clustering"></a>
### Clustering

`oe.cluster_kmeans` and `cluster_hierarchical` follow the sample, distance/linkage,
determinism, validation and saved-output contracts in [multivariate](multivariate.md).
The separate `oe.twostep` route supplies bounded mixed-data CF-tree clustering,
selection, assignment, diagnostics and replay as specified in [TwoStep](twostep.md).
[MARKET-172](https://linear.app/bluearf/issue/MARKET-172) remains completed for its
delivered domain. Individual vendor options not yet mapped remain row-review
work in [MARKET-832](https://linear.app/bluearf/issue/MARKET-832), rather than
reopening that issue or inferring universal clustering support.

<a id="frontier"></a>
### Stochastic frontier

`oe.frontier`, `frontier_efficiency` and the bounded panel `oe.xtfrontier` have distribution, sample,
covariance, convergence and saved-target contracts described in
[advanced families](advanced_families.md), [extended mixed](extended-mixed.md) and
the [generated inventory](../capabilities.md).
[MARKET-133](https://linear.app/bluearf/issue/MARKET-133) remains completed for its
delivered scope. A vendor's particular panel-frontier options or DEA row still requires its own
mapping under [MARKET-832](https://linear.app/bluearf/issue/MARKET-832); the source
name alone does not establish those domains.

<a id="ml"></a>
### Supervised ML and regularization

[Weighted numeric CART](supervised-cart.md) and
[regularized families](regularized-all-family.md) exist. Training-only pruning,
sample/weight alignment, held-out metrics and portable tree state have their own
contract; causal-forest nuisance learning does not establish a generic random
forest estimator. General RF acceptance remains
[MARKET-686](https://linear.app/bluearf/issue/MARKET-686); boosting, CHAID/QUEST,
KNN/SVM and neural scope remains [MARKET-363](https://linear.app/bluearf/issue/MARKET-363).

<a id="smoothing"></a>
### Smoothing and nonparametric regression

Gaussian GAM, splines/RCS, LOESS, FP/MFP, MARS and mixed-categorical kernel
regression exist. [Nonparametric](nonparametric.md),
[smoothing stages](smoothing-stages.md) and [saved targets](smoothing-postestimation.md)
define sample/category/bandwidth alignment, covariance/derivatives/contrasts,
rank/convergence diagnostics and replay. Non-Gaussian models, heterogeneous
penalties, robust multivariate LOESS and adaptive bandwidth/bias inference remain
[MARKET-158](https://linear.app/bluearf/issue/MARKET-158).

<a id="mi"></a>
### Multiple imputation

EM/MCAR, MVN/monotone normal, normal/PMM/binary and discrete FCS, passive transforms,
Rubin/Barnard–Rubin pooling and D1 linear contrasts exist.
[Generation](mi-generation.md), [chained](mi-chained.md), [discrete](mi-discrete.md),
[pooling](mi-pooling.md) and [sensitivity](mi-sensitivity.md) define missing patterns,
category/posterior-draw laws, pooled full covariance/df, diagnostics and replay.
Gaussian NARFCS original acceptance remains [MARKET-687](https://linear.app/bluearf/issue/MARKET-687);
within-chain passive feedback, SMC-FCS, survey weights, pooled LR and general
MNAR remain [MARKET-145](https://linear.app/bluearf/issue/MARKET-145).

<a id="survey"></a>
### Complex survey and replicate inference

Bounded one-to-four-stage SRSWOR, fully stratified lower stages and supplied
BRR/Fay/jackknife/bootstrap replication exist.
[Design](survey-design.md), [four-stage](survey-four-stage.md),
[regression](survey-regression.md), [replication](survey-regression-replication.md)
and [margins](survey-replicate-margins.md) specify PSU/strata/FPC/domain/sample
alignment, full design covariance, df, diagnostics and saved replay.
PPS, five-plus stages, calibration/poststratification and broader singleton/DEFF
domains remain [MARKET-211](https://linear.app/bluearf/issue/MARKET-211);
replicate-unit/partial-FPC inference remains [MARKET-212](https://linear.app/bluearf/issue/MARKET-212).

<a id="timeseries"></a>
### ARIMA, seasonal adjustment and forecasting

ARIMA/SARIMA, additive ETS selection, STL, native bounded X-11, calendar/intervention,
Croston, combination and backtesting exist. [ARIMA](arima.md),
[seasonal adjustment](seasonal-adjustment.md) and [forecast extensions](forecasting-extensions.md)
define calendar/gap/sample/weight policy, likelihood/covariance, convergence,
forecast targets and replay. Multiplicative ETS original reference convergence
remains [MARKET-688](https://linear.app/bluearf/issue/MARKET-688). Wider joint
selection, outlier/revision/transfer, X-13/TRAMO and other forecast engines remain
[MARKET-162](https://linear.app/bluearf/issue/MARKET-162).

<a id="volatility"></a>
### Volatility

ARCH/GARCH/component/FIGARCH/MIDAS, ADCC, diagonal BEKK/VECH and documented skewed
laws exist. [ARCH](arch.md), [GARCH extensions](garch-extensions.md) and
[multivariate GARCH](mgarch.md) specify S/W/V/D/P/R and stationarity/positivity
admission. FIEGARCH/VCC, higher orders, broader ARCH-M/variance regressors and
Dataset extensions remain [MARKET-815](https://linear.app/bluearf/issue/MARKET-815).

<a id="var"></a>
### VAR, VEC and proxy identification

`oe.var`, VEC/SVAR, IRF/FEVD and bounded single-shock point-only `proxy_svar` exist.
[VAR](var.md) and [proxy SVAR](proxy-svar.md) define period alignment, full reduced-form
covariance, identification/stability diagnostics, saved shocks/IRF and replay.
Proxy uncertainty and panel VAR/wider inference remain
[MARKET-150](https://linear.app/bluearf/issue/MARKET-150).

<a id="ardl"></a>
### ARDL and NARDL

[Time-series models](tsmodels.md) and [NARDL](nardl.md) document available lag/sample
contracts, asymptotic ARDL bounds and conditional multipliers, covariance,
diagnostics and saved forecasts. Finite-sample bounds and simultaneous multiplier
calibration remain [MARKET-151](https://linear.app/bluearf/issue/MARKET-151).
[MARKET-112](https://linear.app/bluearf/issue/MARKET-112) records the NARDL theory
boundary; borrowed PSS critical values are not silently supplied.

<a id="fractional"></a>
### Fractional memory

`oe.fracdiff`, `gph`, **conditional finite-filter `arfima`** and saved forecasts
exist. [Fractional memory](fractional-memory.md) specifies filter/burn/calendar
sample, OIM/covariance, diagnostics and replay. Exact stationary Gaussian ML/MPL
and broader uncertainty/diagnostics remain [MARKET-160](https://linear.app/bluearf/issue/MARKET-160).
ARFIMA as a whole must not be labelled absent.

<a id="unitroot"></a>
### Unit roots, cointegration and breaks

[Unit-root](unitroot.md), [advanced breaks](advanced-unitroot-breaks.md),
[cointegration breaks](cointegration-breaks.md), [panel cointegration](panel-cointegration.md)
and [causality](causality-extensions.md) describe available fixed/selected lag,
calendar, missing, statistic, critical-value, covariance and saved-summary domains.
Ng–Perron constant inference and bounded causality/break cointegration deliveries
are retained. Trend calibration [MARKET-813](https://linear.app/bluearf/issue/MARKET-813),
Enders–Siklos critical values [MARKET-814](https://linear.app/bluearf/issue/MARKET-814)
and broader [MARKET-125](https://linear.app/bluearf/issue/MARKET-125) remain;
automatic Fisher–Johansen/MHM/Westerlund ECM remain
[MARKET-149](https://linear.app/bluearf/issue/MARKET-149). A sourced critical table
does not establish arbitrary p-values or finite-sample calibration.
[CIPS](cips.md) and [PANIC](panic.md) have bounded balanced-panel implementations;
their broader residual/calibration scope remains
[MARKET-179](https://linear.app/bluearf/issue/MARKET-179).

<a id="state"></a>
### State space

Proper-prior Gaussian state-space, masked/scheduled systems and **fixed-equation
exact diffuse** are implemented. [Scheduled state space](sspace-scheduled.md) and
[exact diffuse](sspace-diffuse.md) define observed-cell/calendar alignment,
full state/disturbance covariance, likelihood diagnostics, smoother/forecast and
portable replay. Fixed equations and diffuse filtering do not establish general
parameter estimation with diffuse initialization or its uncertainty. Wider scope
remains [MARKET-216](https://linear.app/bluearf/issue/MARKET-216).

<a id="dfm"></a>
### Dynamic factors and nowcasting

[Dynamic factor](dynamic-factor.md) implements identified factor AR(1), white
diagonal measurement noise and a **known finite initial prior**, including arbitrary
missing observations, OIM/full covariance and saved state/forecast replay.
Stationary/serial-idiosyncratic original author-fixture acceptance remains
[MARKET-647](https://linear.app/bluearf/issue/MARKET-647). Wider factor VAR,
release-vintage/no-lookahead, monthly/quarterly aggregation, news/revision targets
remain [MARKET-217](https://linear.app/bluearf/issue/MARKET-217).

<a id="mixed-frequency"></a>
### Mixed frequency and temporal disaggregation

[Temporal disaggregation](temporal-disaggregation.md) and
[time-series workflows](tsworkflows.md) describe MIDAS/U-MIDAS and fixed-rho
Denton/Cholette/Chow–Lin/Fernandez/Litterman domains, release/calendar/sample
alignment, covariance, diagnostics, targets and replay. Estimated rho/full
uncertainty, broader lag laws, vintages and mixed-frequency VAR scope remain
[MARKET-159](https://linear.app/bluearf/issue/MARKET-159).

<a id="dsge"></a>
### DSGE

No accepted DSGE estimator is asserted. [MARKET-218](https://linear.app/bluearf/issue/MARKET-218)
owns implementation/research scope. S/W/V/D/P/R and package/native evidence are
unestablished for this family; a state-space API is not DSGE acceptance.

<a id="meta"></a>
### Meta-analysis

[Meta extensions](meta-extensions.md) and [dependent meta](dependent-meta.md) cover
independent/dependent/three-level ML/REML, CR2/Satterthwaite/HTZ and bounded bias
sensitivity. Effect/sample/covariance alignment, convergence, full component/target
uncertainty and replay use those contracts. The completed 816–823 frozen backend
receipts do not assert installed UI acceptance. Wider unstructured multivariate
component uncertainty/calibration remains [MARKET-174](https://linear.app/bluearf/issue/MARKET-174).

<a id="sequential"></a>
### Power and sequential design

[Power designs](power-designs.md) and [group sequential](group-sequential.md) cover
their stated models, including known-variance Gaussian canonical boundaries with
four spending laws. Inputs, correlation/law, error probability, numerical
diagnostics and saved design replay are explicit. Futility, unknown variance,
ICC/adaptation, binomial/PH/integer-N/accrual scope remains
[MARKET-175](https://linear.app/bluearf/issue/MARKET-175).

<a id="inference"></a>
### General joint and simultaneous inference

[Finite-sample joint](finite-sample-joint.md),
[distribution-free joint](distribution-free-joint.md),
[confidence sequences](confidence-sequences.md) and [multiple testing](multiple-testing.md)
have bounded null, sampling/dependence, covariance, diagnostic and saved-replay
contracts. Model-derived general nulls/adaptive broader targets remain
[MARKET-191](https://linear.app/bluearf/issue/MARKET-191) and
[MARKET-192](https://linear.app/bluearf/issue/MARKET-192).

<a id="postestimation"></a>
### Common prediction and saved targets

[Prediction domains](prediction-domains.md), [saved targets](saved-target-extensions.md)
and [postestimation](postest.md) define training-sample/category/weight alignment,
full target covariance, diagnostics and saved replay. The 74 common adapters are
a separate count from family-specific forecast/scoring helpers; absence of a
common route does not mean all postestimation is absent. Remaining targets,
observation intervals and suest/score routes remain
[MARKET-143](https://linear.app/bluearf/issue/MARKET-143).

<a id="capacity"></a>
### Capacity and devices

[Dataset stages](dataset_stages.md) and the [generated inventory](../capabilities.md)
separate streaming preparation, estimator resource guards and statistical
restrictions. Available CPU acceptance and supported MPS paths do not imply CUDA
acceptance. CUDA is explicitly deferred under [AGENTS.md](../../AGENTS.md).
Remaining declared capacity/target scope: [MARKET-74](https://linear.app/bluearf/issue/MARKET-74).

<a id="distribution"></a>
### Distribution and team acceptance

Use the [dated distribution status](../validation/distribution-status-2026-10-11.md)
for source, package, frozen, installed UI, Windows, persistence and anonymous
public readback. [MARKET-515](https://linear.app/bluearf/issue/MARKET-515) owns the
unified release; [MARKET-801](https://linear.app/bluearf/issue/MARKET-801) owns live
installed team acceptance, under [MARKET-75](https://linear.app/bluearf/issue/MARKET-75).
Developer ID/notarization is deferred under [AGENTS.md](../../AGENTS.md); ad-hoc
internal Mac testing does not become notarized acceptance.

<a id="demand"></a>
### Demand systems and production functions

[Structural micro research plan](structural-micro-research-plan.md) and
[MARKET-365](https://linear.app/bluearf/issue/MARKET-365) own demand/BLP/production
research and implementation stages. No accepted complete estimator or S/W/V/D/P/R
contract is inferred from a catalogue mention. Prior research/priority decisions
remain in that issue.

## Work order and closure

High-priority release/team work (515/801) and active multivariate work (824–831)
retain their existing owners. The comparison reconciliation first removes
incorrect absence claims and makes every review obligation visible. Open
High-priority method families 143/145/149/150/151/152/153/211/360 then need their
own coherent implementation stages and relevant scientific acceptance; their
priority is not lowered by this documentation batch. Original failed reference
or calibration cases remain recorded in their canonical child issues.

Only fully reviewed rows may become accepted mappings. A successful document
check, quick merge gate or completed child does not close a method parent.
[MARKET-832](https://linear.app/bluearf/issue/MARKET-832) remains open while the
manifest reports `needs-review` occurrences. [MARKET-833](https://linear.app/bluearf/issue/MARKET-833)
has its own distribution-document acceptance. Follow the
[current CI policy](../validation/ci-cadence.md): focused document/generator
checks and fresh strict quick gate, with no routine full local numerical matrix.
