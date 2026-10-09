# Observational marginal distributions and quantiles

MARKET-483–485 extend the randomized distribution targets with three distinct
observational estimators. They return complete checksummed `TableSet` artifacts
through `causal_design_save/load`. This is bounded method support under
MARKET-187/GitHub75; it does not close every observational survival,
distribution, sensitivity or instrumental-variable target.

```python
cdf = oe.treatment_cdf_ipw(
    data, "outcome", "treated", x=["pretreatment_x"],
    design="unconfounded", thresholds=[-.25, .75],
)
qte = oe.treatment_quantile_ipw(
    data, "outcome", "treated", x=["pretreatment_x"],
    design="unconfounded", quantiles=[.25, .5, .75], reps=199, seed=1729,
)
dr = oe.treatment_cdf_aipw(
    data, "outcome", "treated", x=["pretreatment_x"],
    design="unconfounded", thresholds=[-.25, .75], folds=3, seed=1729,
)
```

`design` and `x` are required. `design="unconfounded"` declares consistency,
no interference, independent identically distributed rows, pretreatment
controls, conditional unconfoundedness and population positivity. The method
does not infer those assumptions from a table. Numeric treatment is exactly
0/1; Boolean labels are refused. `x=[]` fits an intercept-only propensity.
Control columns must be finite numeric quantities, distinct from outcome and
treatment, with no duplicate or automatically omitted columns. Categorical
features require an explicit pretreatment numeric encoding.

The target population is the sampling population of the admitted rows.
`missing="raise"` is the default; explicit `missing="drop"` admits complete
cases across outcome, treatment and every control. This changes the analyzed
population and does not establish missing-at-random identification. Original
physical positions and typed row labels, including duplicate labels, persist.

## IPW marginal CDF

Write `e(X)=P(D=1|X)`, `p0=1-e`, `p1=e` and `Z(t)=1{Y<=t}`. Both arm CDFs use
normalized Hájek weights:

`Fhat_d(t) = sum[1{D=d} Z(t)/phat_d(X)] / sum[1{D=d}/phat_d(X)]`.

`treatment_cdf_ipw` estimates an unpenalized logistic propensity with a fixed
intercept and the supplied controls. Correct finite-dimensional logistic
specification, a regular finite MLE and population overlap are additional
assumptions. The inclusive thresholds must be a predeclared finite grid;
output preserves their supplied order.

Inference stacks `theta=(beta,F0(grid),F1(grid))`. Its row equations are the
logit score `X*(D-e)` and `1{D=d}/p_d*(Z-F_d)`. The analytic derivative retains
both the fitted-propensity derivative and each observed normalization mass.
If `A` is the mean derivative and `psi_i` the full row score, contributions are
`-A^-1 psi_i/N`; HC0 covariance sums their outer products. A shared propensity
fit induces joint dependence across arms and thresholds, so independent-arm
variance addition is inappropriate. The returned `nuisance_covariance` includes
the complete propensity and arm-target parameter covariance.

`effects` reports `F1-F0`; `arm_targets` reports both marginal CDFs.
`joint_covariance` retains their covariance with the effects, and `covariance`
is the effect block. Pointwise normal z/p/intervals use `level=.95` by default,
with `df=None`. Intervals are not truncated to parameter bounds or advertised
as simultaneous bands. Constant empirical arm indicators retain their exact
0/1 means. Zero empirical variance has SE=0, a point interval and undefined
z/p; it does not establish certainty about unsampled tails.
Nonzero normal statistics or confidence shifts that float64 cannot represent
are refused rather than displayed as a point interval.

## IPW marginal quantiles

`treatment_quantile_ipw` uses the same fitted propensity and computes the
weighted left inverse separately in each arm. Sort outcomes stably, accumulate
positive inverse weights, and choose the first outcome whose cumulative weight
is at least `q` times the final sorted cumulative weight. There is no quantile
interpolation. Predeclared quantiles are distinct values strictly in `(0,1)`.
The QTE is `Q1(q)-Q0(q)`, a difference between two marginal potential-outcome
quantiles. It does not identify the distribution of individual treatment
effects.

A private CPU Torch generator resamples complete IID `(X,D,Y)` rows with
replacement. Every replicate re-centers its controls and fits its own
propensity before computing weights and quantiles. Resampling arms separately
or holding the original propensity fixed would omit relevant nuisance and
sampling variation. All draw indices, duplicate original positions, nuisance
fit certificates, inverse weights, sorted cumulative weights and arm/effect
replicates persist. A bad replicate rejects the whole result; no replicate is
skipped, redrawn or silently repaired.

The joint replicate covariance uses divisor `reps-1`; SE and pointwise
percentile intervals derive from the full replicate matrix. Percentile
endpoints use linear interpolation across bootstrap target values. No normal
bootstrap z or p-value is invented. Exact constant replicate columns are
labeled `zero_empirical_bootstrap_variance`. Bootstrap inference additionally
requires continuous marginal distributions, unique quantiles with positive
finite density, a regular nuisance MLE and the usual empirical-bootstrap
regularity. Processing observed ties does not prove these population
conditions. Defaults are `reps=199`, `seed=1729` and `max_work=1_000_000_000`;
the larger declared work budget accommodates full nuisance refits.

## Cross-fitted AIPW marginal CDF

`treatment_cdf_aipw` makes balanced, deterministic row folds using a private
seed. For every fold it estimates a training-only propensity logit, then a
separate conditional indicator logit within each training arm and threshold.
Every row receives predictions from models that excluded that row. Let
`g_d(t,X)=P(Y<=t|D=d,X)`. Held-out potential CDF scores are

`phi_d(t) = ghat_d(t,X) + 1{D=d}/phat_d(X)*(Z(t)-ghat_d(t,X))`.

Arm targets average these scores. The complete joint covariance of both
arms and their contrast is the HC0 covariance of centered scores divided by
`N` before taking outer products. Fold membership, every training fit,
held-out propensity/outcome predictions, individual scores, normalized
contributions and joint covariance persist.

Point consistency requires a consistent propensity, or consistent conditional
indicator means in both arms, together with positivity and finite moments.
The displayed orthogonal-score confidence intervals require **both**
propensity and outcome nuisance consistency and sufficient product rates;
cross-fitting alone does not verify these conditions. One correct nuisance
under misspecification of the other is not advertised as sufficient for valid
confidence intervals from this covariance.

Raw AIPW arm estimates may be nonmonotone over thresholds or outside `[0,1]` in
a finite sample. They are retained without clipping or rearrangement. This API
does not invert that estimated grid into a quantile estimate. The output tables
are `effects`, `arm_targets`, `joint_covariance` and `covariance`.

## Fit certificates and admission

All numerical kernels use native CPU float64 Torch. Logistic models maximize
the unpenalized Bernoulli likelihood with a bounded analytic Newton algorithm.
Training-only centering/scaling is an invertible change of coordinates. No
objective penalty or curvature ridge is applied. Certificates include the
complete training design/response, coefficients, probabilities, per-row and
aggregate scores, Hessian, robust coefficient covariance, likelihood/iteration
history, line-search fractions, scaled gradient and undamped Newton step.
Convergence requires positive-definite curvature, a small scaled gradient and
a small **undamped** Newton step. Small score alone is insufficient to certify
a finite MLE for a separating likelihood. Training probabilities at or beyond
`1e-10` or `1-1e-10` are outside this supported numerical domain.
Information-matrix condition numbers at or above `1e12` are also refused.

`max_iterations=40` is configurable from 1 to 200; each update has at most 12
line-search trials. Separation, singular design/curvature, nonconvergence,
empty training arms/classes and failed overlap reject the complete computation.
In particular, a threshold that makes a training-arm indicator constant is
outside the AIPW threshold-logit domain. It is not replaced by an implicit
constant model. Every fitted propensity, including bootstrap and both training
and held-out cross-fit propensity values, must lie inside the declared overlap
interval `[overlap,1-overlap]`; `overlap=.01` by default. No propensity clipping,
row trimming, feature dropping, learner substitution or automatic tuning occurs.

The admitted domain is at most 10,000 original input rows, 16 controls and 16
thresholds/quantiles, at least 30 complete rows and six rows per original arm.
Each quantile bootstrap arm also needs six rows. Cross-fitting supports 2..5
folds, bootstrap supports 20..999 replicates, and seeds are integers in
`[0,2**63-1]`. Insufficient nuisance training support is refused. Dataset
collection, non-CPU devices, generic observation weights, clustered/survey
inference and adaptive grids/policies are outside this route.

CDF methods default to `max_work=100_000_000`. Before sample/design or nuisance
allocation, the complete work plan sums the worst-case derivative/Hessian and
factorization calls, the separate line-search value calls, target covariance
work and all replicate sorting. No ridge-search branch is charged because none
is executed. The workspace plan includes full source/design, fitted nuisance
states, scores/covariances, every draw and Python/JSON artifact copies. Plans
use the original input count conservatively, including rows later dropped.
Budgets refuse complete calculations rather than returning truncated state.
The workspace estimate is not a process-RSS guarantee.

Independent source fixtures compare fitted logits with statsmodels, complete
stacked derivatives/covariance with NumPy and numerical differentiation, every
recorded bootstrap refit/inverse with an independent implementation, and all
held-out AIPW nuisance fits/scores/covariance. Fixed medium-sample synthetic
diagnostics use analytically known logistic-shift marginal targets; they are
not universal coverage guarantees. Complete serialization, dtype/sample
identity, private RNG, ties, separation under rescaling, unsupported domains
and preflight failures have explicit checks. Source/wheel/frozen/native app
acceptance and licensed vendor comparisons remain separate evidence layers.

Primary sources:

- [Donald–Hsu: observational IPW marginal CDFs and quantile inversion](https://www.econ.sinica.edu.tw/~econ/pdfPaper/12-A016.pdf).
- [Firpo: marginal quantile treatment identification and regularity](https://math.pku.edu.cn/teachers/xirb/Courses/QR2013/ReadingForFinal/EfficientSemiparametricEstimationOfQuantileTreatmentEffects.pdf).
- [DoubleML IRM/APO orthogonal scores](https://docs.doubleml.org/stable/guide/scores.html).
- [Chernozhukov et al.: DML nuisance and inference conditions](https://arxiv.org/abs/1608.00060).

The fixed-dimensional logit and finite-grid implementation does not claim to
reproduce the nonparametric efficiency or entire-function inference results of
these references. No Stata/vendor parity flag is enabled.
