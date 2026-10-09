# Regularized prediction and orthogonal PLR inference

This family supplies nine distinct `ModelSpec`/`ResultBundle` estimators:
`ridge`, `lasso`, `elasticnet`, `pls`, `kernelreg`, `localreg`, `postdouble`,
`partiallingout` and `dmlplr`. Numerical estimation uses native CPU float64
PyTorch. There is no SciPy, statsmodels or scikit-learn runtime dependency.
These methods have independent algebraic verification; this is not Stata
parity evidence, a release validation, or a claim that every causal model is
identified.

Separate `elasticnet_logit` and `elasticnet_poisson` predictive estimators add
Bernoulli/Poisson paths, training-only CV, frequency/empirical-loss weights,
categorical maps, forced controls and literal feature penalty factors. Their
complete state and explicit uncertainty/domain boundaries are documented in
[regularized-glm.md](regularized-glm.md). The numeric unweighted Gaussian/PLS
contracts below remain method-specific. Weighted/category Gaussian and scalar
partial PLS1 fixed/CV routes are documented in
[regularized-all-family.md](regularized-all-family.md); they retain the existing
numeric IID route and do not extend unvalidated score plug-in inference.

Scalar PLS1, saved component selection and the remaining GLM/weight/category
option stages are described in [panel-prediction-extensions.md](panel-prediction-extensions.md).

```python
import openecon as oe

prediction = oe.elasticnet(data=df, y="outcome", x=["age", "income"],
                           l1_ratio=0.5, selection="cv",
                           lambda_path=[1.0, 0.2, 0.04], folds=5, seed=1729)
future_y = oe.regularized_predict(prediction, new_df)

smooth = oe.localreg(data=df, y="outcome", x=["age"],
                     selection="fixed", bandwidth=3.0, query=[[25.0], [40.0]])

effect = oe.dmlplr(data=df, y="outcome", treatment="participation",
                   x=["age", "income"], nuisance="lasso",
                   selection="plugin", folds=5, seed=1729)

# An alternative honest nested-CV nuisance learner:
effect_ridge = oe.dmlplr(data=df, y="outcome", treatment="participation",
                         x=["age", "income"], nuisance="ridge", selection="cv",
                         lambda_path=[0.1, 0.01, 0.001], folds=5, seed=1729)

oe.regularized_table(prediction).to_latex("prediction.tex")
effect.to_latex("effect.tex")
```

Direct specs use `predictors` for the predictor/control columns,
`columns={"treatment": "participation"}` for inferential methods and
`options={...}` for tuning. The public convenience APIs follow the existing
`data=`, `y=`, `x=` convention. The local and inferential methods described here
retain their method-specific numeric/unweighted restrictions. Gaussian/PLS
predictive methods additionally admit explicit weighted/category fixed/CV
routes under the separate contract above; cluster covariance, panels and
time-series dependence remain unsupported for those prediction targets.
Missing inputs raise by default; `missing="drop"` records the
original positional estimation sample, dropped count and input/sample hashes.
Prediction from saved state rejects incomplete inputs and preserves row indices.

## Prediction is a separate target

The first five estimators estimate a prediction function, not an unpenalized
coefficient target with ordinary OLS confidence intervals. Their result has
**empty `coefficients` and `covariance_matrix`**, `inference.available=False`
and `extra.target="prediction"`. The actual coefficients/constants and
lambda path are in `extra.penalized_state`; query conditional means and the
saved smoothing function are in `extra.query_estimates` and
`extra.smoother_state`. JSON round trips preserve these values and allow future
prediction. `oe.regularized_table(result)` exports the actual predictive terms
or query means to LaTeX without fabricated standard errors or p-values.

The smoothing state retains the selected training columns/outcome so the
nonparametric function remains reproducible after saving. This is an explicit
in-memory model state, not a streaming or privacy-preserving synopsis.

## Ridge, Lasso and elastic net

Let `z_j=(x_j-center_j)/scale_j`, with the training-sample mean removed only
when `intercept=True`. `standardize=True` uses centered RMS when an intercept
is fitted and raw RMS otherwise. Constant centered columns use scale one and
receive coefficient zero. The outcome is centered with an intercept and is
never standardized. The minimized objective is

\[
 \frac{1}{2n}\sum_i(y_i-\bar y-z_i'\beta)^2
 +\lambda\left[\rho\sum_j q_j\ell_j|\beta_j|
       +\frac{1-\rho}{2}\sum_j q_j\beta_j^2\right].
\]

`rho=0` is ridge; `rho=1` is Lasso; elastic net exposes `l1_ratio=rho`.
The intercept is unpenalized. Literal nonnegative `penalty_factors` are `q_j` (default one; no factor
normalization). Named `forced_controls` set their factor to zero; their block
must be identified in every training fold. Nonunit factors or forced controls
refuse `selection="plugin"`; fixed/CV factor paths preserve their own complete
fold state. See [factor conventions and validation](next-eight-regularized-prediction.md).
Original-unit coefficients are `beta/scale`
and the reported predictive constant is `mean(y)-center @ (beta/scale)`.
Ridge uses a direct linear solve (SVD least squares for a zero penalty);
Lasso/elastic net use cyclic residual coordinate descent. Each path point
records its objective, sweep count and maximum KKT residual. Solver success
requires the KKT threshold, rather than an arbitrary small objective change.
Convergence failures raise; no unconverged estimate receives a CI.

The definition follows [Friedman, Hastie and Tibshirani (2010)](https://web.stanford.edu/~hastie/Papers/glmnet.pdf).
Its lambda normalization matters when comparing a package that defines ridge
penalties on a sum rather than a mean of squared residuals.

* `selection="fixed"` requires a finite nonnegative `penalty`. Optional
  `lambda_path` is sorted decreasingly and must contain that exact penalty.
* `selection="cv"` minimizes pooled out-of-fold MSE over `lambda_path` or a
  recorded geometric path. A supplied path is a common fixed absolute grid.
  An automatic path compares common dimensionless fractions of `lambda_max`:
  each fold derives its own absolute grid from its training predictors/outcomes,
  then the selected fraction is refitted using the full supplied training sample.
  Fold-specific grids, selector units, selected fraction and final path are saved.
  Validation labels cannot affect that fold's candidate grid. Each fold fits
  centering/scaling on its own training rows. A local generator records
  reproducible `seed` and fold assignments;
  the global random state is untouched. Ties select the largest lambda.
  With unit factors, the path reference is `max(abs(z.T @ yc/n))/max(rho,0.01)`, bounded below
  by `1e-8`; for ridge this is a numerical reference scale, because ridge
  has no finite penalty at which all coefficients vanish.
  With zero-factor controls, the automatic reference first residualizes y and
  penalized predictors against the identified unpenalized training block,
  and divides each penalized score by its literal factor. Validation outcomes
  still cannot enter that fold's grid.
* `selection="plugin"` estimates heteroskedastic score loadings iteratively.
  Under the half-MSE objective, `lambda = c * Phi^-1(1-gamma/(2p)) /
  (sqrt(n)*rho)` and `ell_j=sqrt(mean(z_j^2*residual^2))`, with configurable
  `plugin_c>=1`, `plugin_gamma` and `plugin_iterations`. Every loading update
  and its convergence status are recorded. Failure to stabilize raises.
  The score plug-in has an L1 target; ridge rejects this setting and supports
  fixed/CV tuning. Elastic-net plug-in uses the same L1 score calibration
  with its explicitly recorded L2 component.

The heteroskedastic calibration and residual-loading update are described in
[Chernozhukov's MIT high-dimensional econometrics notes](https://ocw.mit.edu/courses/14-387-applied-econometrics-mostly-harmless-big-data-fall-2014/d0354c71c7f7d3584107e9a1b2086174_MIT14_387F14_large_p.pdf).
A plug-in penalty is an implementable choice under distributional and sparsity
conditions, not a guarantee of inference for the predictive coefficient vector.

## General kernel and local-linear regression

These APIs estimate continuous conditional means for arbitrary numeric
predictors; no running-variable cutoff or treatment discontinuity is assumed.
They use multivariate product Gaussian, Epanechnikov, uniform or triangular
kernels. `kernelreg` computes the Nadaraya-Watson weighted mean. `localreg`
fits a weighted local hyperplane in bandwidth-scaled differences and reports
its intercept at each query. It reproduces an affine function at observed
boundaries when the local design is identified. That is the local-linear
boundary-bias correction, not a claim of zero bias for arbitrary functions.
See [Wasserman's nonparametric regression notes](https://stat.cmu.edu/~larry/%3Dsml/nonpar.pdf).

Bandwidth may be a positive scalar or one positive value per predictor.
`selection="fixed"` takes `bandwidth`; `selection="cv"` takes a nonempty
`bandwidth_path` and evaluates exact leave-one-out MSE on every estimation
row. A candidate failing local support/rank is recorded as invalid, with its
reason, and is never scored on a smaller sample. If all candidates fail, the
whole fit raises. `selection="plugin"` uses the stated multivariate
normal-reference pilot `1.06*min(sd,IQR/1.349)*n^(-1/(p+4))`. This is a pilot,
not a regression-risk-optimal bandwidth assertion.

Query matrices have shape `m by p`; absent `query`, the estimation sample is
evaluated. An explicit query evaluates those requested rows only; training
MSE and the training chart sample are omitted because they were not evaluated.
The saved state records bandwidth/kernel, observed bounding box,
effective local count `(sum w)^2/sum(w^2)`, nonzero support, local design
condition number and boundary flags. Compact kernels reject empty support;
local-linear estimates reject an unidentified/ill-conditioned slope design.
`min_effective` is enforced for every query. Queries outside the observed
bounding box raise unless `support="extrapolate"` is explicitly selected;
box inclusion alone does not establish local support. No pointwise confidence
intervals are returned, because smoothing bias and bandwidth uncertainty are
not removed by a naive weighted-regression standard error.

## Inferential treatment targets

All three inferential methods target a single scalar `theta` in
`Y=D*theta+g(X)+u`, with the identifying restriction `E[u|D,X]=0` and positive
conditional treatment variation. They report **only the treatment effect**;
nuisance/control coefficients do not get confidence intervals. Their model
state records the nuisance fits, score, Jacobian, influence function and actual
sample/fold assignment.

* `postdouble` fits Lasso for outcome and treatment on controls, takes the
  union of the two selected control sets and projects both variables on that
  union plus the optional intercept. It then solves the residual orthogonal
  treatment score. This is the double-selection construction of
  [Belloni, Chernozhukov and Hansen (2013/2014)](https://cemmap.ac.uk/publication/inference-on-treatment-effects-after-selection-amongst-high-dimensional-controls-2/).
  Covariance comes from the target influence score; selecting only the outcome
  variables and attaching their naive OLS CI is not implemented.
* `partiallingout` solves the orthogonal score using same-sample regularized
  outcome/treatment nuisance residuals. It requires appropriate approximate
  sparsity and empirical-process/Lasso rate conditions. Its state explicitly
  says `cross_fitted=False`; same-sample learning is not presented as DML.
* `dmlplr` implements the pooled DML2 partialling-out score. Each outer test
  observation receives nuisance predictions from fits trained solely on the
  complementary rows. Plug-in loading estimation, centering and scaling use
  only those training rows. CV nuisance tuning is nested inside the outer
  training sample; no outer test outcome/treatment is consulted for fitting,
  tuning or scaling. Every physical row is tested once. The same `folds`
  setting is used for outer folds and inner CV folds, and derived nuisance
  seeds are persisted. Supported learners are ridge, Lasso and elastic net.
  More elaborate nuisance features must be supplied as explicit numeric
  controls; this API does not silently create nonlinear features.

Writing `v_i=D_i-mhat(X_i)` and `r_i=Y_i-lhat(X_i)`, all three solve

\[
 \hat\theta=\frac{\sum_i v_i r_i}{\sum_i v_i^2},\quad
 \psi_i=v_i(r_i-\hat\theta v_i),\quad
 \hat J=n^{-1}\sum_i v_i^2,\quad
 IF_i=\psi_i/\hat J,\quad
 \widehat{Var}(\hat\theta)=\frac{n^{-1}\sum_i IF_i^2}{n}.
\]

The influence covariance uses **estimated nuisance predictions**, including
out-of-fold predictions for DML. Orthogonality removes first-order nuisance
perturbations under the required rate conditions; a separate naive
selected-model coefficient covariance is never substituted. `HC0` is this
asymptotic IID sandwich. `HC1` multiplies by `n/(n-1)` for partialling-out/DML
or by `n/(n-rank(selected controls including intercept)-1)` for post-double
selection. Treatment tests use normal asymptotic inference. Unidentified
treatment variation, saturated selection and zero/nonfinite score variance
raise explicitly.

[Chernozhukov et al. (2018)](https://academic.oup.com/ectj/article/21/1/C1/5056401?login=false)
establish the role of orthogonal moments and cross-fitting. The PLR DML
contract needs finite moments, nuisance L2 consistency and product error
`o(n^-1/2)` in addition to identification. The in-sample methods have stronger
sparsity/selection conditions. CV is a prediction criterion and is explicitly
warned against as a stand-alone uniform-inference guarantee in same-sample
selection methods. Numerical score means and nuisance-score sensitivity are
diagnostics, not proof of these statistical assumptions.

## Execution and validation boundaries

All eight methods use the shared `ModelFrame` and reject replayable `Dataset`
inputs with `streaming_unsupported` **before any source iteration**. They do
not collect a Dataset under a streaming label. Live buffer estimates are
checked against `OPENECON_WORKSPACE_MB` before designs, CV copies, local
factors and persisted state are constructed. `max_work` additionally bounds
planned arithmetic: ridge factors, coordinate sweeps, nested nuisance fits or
querywise local fits. A exceeded budget raises with its scope; no sample,
fold, path or query is silently truncated. The estimate excludes caller-owned
inputs, Python object overhead and private BLAS/allocator storage and is not
an RSS guarantee. Kernel queries are processed individually using `O(np)`
numeric workspace rather than constructing an `n by n` smoothing matrix.

Independent development oracles in `tests/test_regularized_oracles.py` use
NumPy sign/support enumeration to solve convex penalties without the runtime
coordinate-descent algorithm, independent train-only ridge fold algebra,
weighted local-regression algebra and the estimated-nuisance influence
sandwich. Adversarial contracts in `tests/test_regularized_contracts.py`
exercise missing alignment, honest split separation, prediction persistence,
Torch-free discovery, high dimensionality, collinearity, support holes,
invalid candidates, unavailable inference, nonconvergence, budget refusal
and Dataset nonmaterialization. Existing estimators are untouched.

`scripts/validate_regularized.py` adds a prespecified 64-replication,
320-observation IID sparse-linear heteroskedastic PLR experiment. The saved
receipt is `reports/validation/market127_regularized.json`: target and
influence-variance discrepancies against independent NumPy algebra were below
`7e-16` and `4e-18`. Observed 95% coverage was 61/64 for post-double-selection
and 60/64 for partialling-out and ridge-nuisance DML; Monte Carlo Wilson
intervals are included. This is one bounded DGP, not uniform statistical
coverage, desktop packaging or release evidence.
