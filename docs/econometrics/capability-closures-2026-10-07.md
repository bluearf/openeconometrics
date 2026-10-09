# Five bounded capability milestones

This delivery completes the core milestones of #34, #54, #66, #71 and #72.
Earlier native implementations remain part of these milestones. The staged
research extensions below remain open work. No global Stata, EViews or SPSS
parity flag is implied. Community procedures and add-ins are comparison routes,
not built-in vendor capabilities.

| Issue | Existing main support | Additions in this delivery |
| --- | --- | --- |
| #34 | MG, PMG, DFE, CCE pooled/mean-group, AMG, DCCE, CS-ARDL and CS-DL | CRE/Mundlak, Hausman–Taylor and saved structural prediction |
| #54 | Linear/nonlinear mediation, moderation and Oaxaca–Blinder | Exposure–mediator interaction natural effects and binary Fairlie decomposition |
| #66 | Dumitrescu–Hurlin W/Z/Zbar and Breitung–Candelon restrictions | Signed partial sums with augmented-VAR restrictions and explicitly conditional iid null resampling |
| #71 | Free k-class, Fuller, LIML and restricted Stock–Yogo tables | Continuously updated IV GMM and compatible effective-F statistic |
| #72 | Bounded ARIMA search, fixed/expanding model refits and saved forecasts | Reverse windows, incremental OLS QR, panel calendar windows and training-window selection/evaluation |

## Execution and result domain

New estimators use native float64 CPU Torch at fit time. These are resident,
unweighted numeric domains; no Dataset collection, GPU execution or categorical
encoding is added. Unsupported ModelSpec options, weights and covariance names
are rejected by the registry. Existing methods retain their own wider domains.
Missing values raise by default; `missing="drop"` defines the retained sample
before group means or counterfactual averages are constructed. Time-based
procedures require a complete regular calendar after sample selection.

`ivcue`, `cre`, `xthtaylor`, `mediation_interaction` and `fairlie` are registered
estimators producing ResultBundles with full covariance, sample positions,
specifications, inference conventions and JSON persistence. All support LaTeX
export. Auxiliary diagnostics and windows return named TableSets with JSON-safe
attributes. They do not manufacture an estimator or inference target.

Common `predict`/`margins` adapters are not extended for these five estimators.
Use the explicit `panel_structural_predict` route for saved CRE/HT means. CUE
weak-instrument inference uses the separately existing structural AR/CLR routes;
normal coefficient intervals do not protect against weak identification.
All work/width/replication limits are checked or enforced, and singular designs,
moments, failed optimizers or failed bootstrap draws remain explicit errors.

## CRE/Mundlak and Hausman–Taylor

`cre` constructs selected numeric within-unit means from **all and only the
retained estimation sample**, augments a Swamy–Arora feasible RE GLS design, and
tests the joint zero mean coefficients using the complete covariance. Default
`robust` covariance is panel CR1 with t(G−1) inference; nonrobust and one declared
cluster nested within panels are also supported. Select `means` explicitly when
some predictors are invariant. Variance components, theta, unit labels and fitted
mean vectors are persisted. Means are not reconstructed from prediction rows.

`xthtaylor` declares varying exogenous/endogenous `x1`/`x2` and invariant
exogenous/endogenous `z1`/`z2` separately. Endogeneity means correlation with the
unit effect; all varying regressors must remain orthogonal to idiosyncratic
errors. At least three independent units, two observations per unit, full rank
and `len(x1)>=len(z2)` are required. It uses within estimates for varying slopes,
between IV for invariant effects and quasidemeaned 2SLS with instruments
`[all varying within deviations, (1-theta_i)*[1,z1,mean(x1)]]`.

The declared finite-sample variance convention is
`sigma_e²=within_RSS/(N-G-K_tv)` and
`sigma_u²=max(between_IV_residual_RSS/(G-K_invariant) - sigma_e²*mean(1/T_i),0)`.
Zero truncation is disclosed. Nonrobust covariance is `sigma_e²*bread`; `robust`
uses panel CR1 transformed score sums and t(G−1) inference. This explicit
convention is independently checked; equality to every vendor HT convention is
not asserted. Units may be unbalanced; duplicate panel periods are rejected.

Saved CRE prediction requires known units and reuses recorded means. HT predicts
the population structural mean with the unit effect integrated to zero. Both
use full coefficient delta covariance and exclude new-outcome/unit noise.
Width is bounded at 32 with an explicit work budget.

References: [Hausman and Taylor](https://web.mit.edu/14.33/www/hausman.pdf),
[official SAS unbalanced HT details](https://support.sas.com/documentation/cdl/en/etsug/68148/HTML/default/etsug_panel_details31.htm).

## Exposure–mediator interaction and Fairlie

`mediation_interaction` supports one exposure, one continuous mediator and one
continuous outcome with numeric controls. Equations are `M~A+C` and
`Y~A+M+A*M+C`. Standardization holds the retained sample's control means fixed.
The full stacked HC1 or one-way cluster CR1 score covariance and native delta
map yield pure/total direct effects, pure/total indirect effects and total effect.
Both identities, `total=PNDE+TNIE=TNDE+PNIE`, are retained. Normal intervals
condition on the declared equations and fixed control distribution.

Interpretation defaults to associational. Causal labels require explicit
consistency, positivity, sequential ignorability and no exposure-induced
mediator/outcome confounding declarations; data do not verify these assumptions.
The existing `mediation` route covers its declared nonlinear outcome domains;
nonlinear exposure–mediator interaction is outside this new API.

`fairlie` fits an actual logit/probit ML reference, with explicit ordered groups
and pooled/A/B coefficients. Pooled omits a group dummy. Stable probability ranks
match observations; the larger group is randomly subsampled and feature switching
orders are randomized. The exact full-group explained difference is reported
alongside an explicit finite matching Monte Carlo remainder; components are never
silently rescaled. Stratified pairs bootstrap repeats fitting and matching with a
local saved seed. All draws, full joint covariance and pointwise percentile
intervals are saved. The coefficient table uses a normal approximation to the
bootstrap covariance. Neither target is a causal decomposition.

Fairlie requires 49..2000 bootstrap draws, 1..1000 matching repetitions and width
at most 24, with explicit work bounds. Failed draws fail the procedure rather
than being omitted. Weights and categorical expansions are unsupported.

References: [natural effects with exposure–mediator interaction](https://pmc.ncbi.nlm.nih.gov/articles/PMC3659198/),
[Fairlie 2005](https://journals.sagepub.com/doi/10.3233/JEM-2005-0259),
[author community implementation conventions](https://github.com/benjann/fairlie/blob/main/fairlie.hlp).

## Asymmetric predictive noncausality

`asymcausality` splits raw increments into positive/negative zero-initial cumulative
sums. A bivariate constant VAR(lags+dmax) tests only the first base lags of the
cause; augmented lags remain nuisance parameters. `dmax` is declared, not selected
by a pretest. ML residual covariance uses divisor T. Full system covariance,
transformation/sample identity and asymptotic chi-square results are retained.

Optional resampling imposes the null on the effect equation, centers joint
residual vectors, resamples them jointly and recursively simulates the augmented
partial-sum VAR from fixed initial values. Plus-one p-values, Monte Carlo SEs and
every statistic are saved. This **conditional iid partial-sum VAR null** can
generate nonmonotone signed paths. It is not the original Hatemi-J leverage/wild
bootstrap or a calibrated heteroskedasticity-robust test. It is labeled explicitly
in returned results. Lag order <=12, dmax<=2, zero or 49..5000 draws and a declared
work budget bound the domain. Predictive rejection is not structural causality.

The runtime receipt also records a narrow size diagnostic: independent Gaussian
random walks, N=160, positive-positive signs, lags=1/dmax=1, 100 datasets and 99
draws per dataset. It reports all p-values, failures, empirical rejection and a
Wilson interval. This diagnostic does not calibrate unrestricted null size.
The recorded run had 1 rejection in 100 datasets (1%; Wilson 95% interval
0.18%..5.45%) and no failed draws. It is conservative in this narrow run, so it
must not be presented as a generally calibrated 5% test.
Dependence, finite-T and frequency-domain tests are covered separately in
[the existing panel/frequency evidence](frequency_panel_causality.md).

References: [Hatemi-J partial-sum framework](https://doi.org/10.1007/s00181-011-0484-x).
Kónya SUR bootstrap, heterogeneous-integration E–K, calibrated wild/leverage
resampling and rolling/Fourier variants remain staged research work in
[#109](https://github.com/bluearf/openecon/issues/109).

## CUE and effective F

`ivcue` minimizes `N*g(b)'S(b)^(-1)*g(b)` with S recomputed at every candidate.
Native BFGS uses exact autodiff gradients/Hessian, a scaled QR 2SLS start and
explicit convergence/curvature checks. Full covariance is the efficient
asymptotic `inv(D'S^(-1)D)/N`, not the objective Hessian. HC0 iid or uncorrected
one-way cluster moment covariance, centered or uncentered, are supported. More
clusters than moments, `N>L>=K`, K<=24 and L<=48 are required. Singular S is an
error; no ridge silently changes the estimator. Hansen J uses L−K degrees of
freedom under strong identification; just-identified J has no p-value.

`effective_f` uses one endogenous regressor residualized on controls:
`fitted'fitted / trace(inv(Z'Z)*first_stage_moment_meat)`. It accepts iid N-divisor,
HC0, declared cluster or explicit HAC meat. This differs from KP/robust Wald F.
It returns **a statistic only**, without rejection or critical values.
Estimator/bias-specific MOP cutoff calibration remains open. Existing sourced
Stock–Yogo tables retain their restricted homoskedastic scalar 2SLS domain;
they are never attached to robust effective F or robust KP.

References: [CUE definition in EViews](https://help.eviews.com/content/gmmiv-Generalized_Method_of_Moments.html),
[effective-F definition, equation 6](https://arxiv.org/html/2309.01637v3#S2.E6).
JIVE, Lewbel-generated instruments and shock-level shift-share inference remain
separately sourced community research milestones in
[#110](https://github.com/bluearf/openecon/issues/110), alongside MOP cutoff calibration.

## Window workflows

`rolling(..., reverse=True)` traverses fixed windows backward; with expanding it
fits suffixes anchored at the final period. Reverse outputs are retrospective and
reject forward forecasts/evaluation. `recursive_ols` updates an augmented [X,y]
QR using its previous R and only newly admitted rows. Coefficients, RSS, full
`RSS/(N-K)*inv(X'X)` covariance and t(N−K) inference match independent prefix OLS.
Only numerical QR is incremental; sample/provenance/result construction still
visits each window. Other covariance/weight/category domains use ordinary refits.

`rolling_panel` counts consecutive integer calendar periods, preserves all
observed cross-sectional rows in each window, and refits within the window's own
sample/lag/rank contract. Unbalanced units are retained when the estimator permits
them. Dates are not silently ranked, gaps are errors, and original physical row
positions remain mapped. No general panel forecasting adapter is synthesized.

`rolling(..., selection={...}, forecast_steps=H, evaluate=True)` reruns bounded
ARIMA order/differencing selection inside each training window. Optional forward
evaluation for single-response ARIMA/ETS/UCM/sspace reads observed future targets
only after fitting/forecasting. It records physical target positions, errors,
squared errors and absolute errors. Future regressors are supplied explicitly per
origin. Changing future outcomes leaves earlier coefficients/forecasts unchanged
and changes only their scores. Inference conditions on the selected order.
All windows save full V/spec/IDs/sample positions and explicit failures;
`on_error="record"` retains failures. See [workflow domains](tsworkflows.md).

## Reproduce and interpret the evidence

```sh
PYTHONPATH=src python scripts/validate_capability_closures.py --size
PYTHONPATH=src python -m pytest tests/test_capability_closure_extensions.py
```

The driver above is the runnable deterministic example for all new public routes.
Native runtime receipt (internal evidence excluded from this public snapshot)
records source hashes, forbidden-import guards, JSON/LaTeX and saved-prediction
checks. Independent development-only NumPy/SciPy oracles cover full covariance,
counterfactual Jacobians, CUE likelihood/objective, first resampling draws and
window fits. Existing public reference fixtures cover earlier milestone methods.
Licensed Stata/SPSS/EViews execution and new packaged desktop execution were not
performed in this delivery. Catalog registration and numerical agreement do not
substitute for those separate evidence layers.
