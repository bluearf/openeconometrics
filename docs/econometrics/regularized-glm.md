# Regularized binary and count prediction

`oe.elasticnet_logit` and `oe.elasticnet_poisson` fit bounded, resident
CPU float64 prediction models. Binary logistic outcomes must be exactly 0 or
1, with both classes present in every fitted training partition. Poisson
outcomes must be integer counts between zero and 1,000,000, with at least one
positive count in every training partition. Both expose fixed
penalties, reproducible cross-validation, categorical predictors, observation
weights, per-predictor penalty factors and unpenalized forced controls.

```python
import openecon as oe

binary = oe.elasticnet_logit(
    data=df, y="event", x=["age", "income", "sector"],
    categorical=["sector"], l1_ratio=0.5,
    selection="cv", lambda_path=[0.5, 0.2, 0.08], folds=5, seed=1729,
)
probability = oe.regularized_glm_predict(binary, future_df)
linear_predictor = oe.regularized_glm_predict(binary, future_df, kind="link")
path = oe.regularized_glm_path(binary)

counts = oe.elasticnet_poisson(
    data=df, y="count", x=["age", "income"],
    selection="fixed", penalty=0.1,
    weights="frequency", weight_type="fweight",
    forced_controls=["age"],
)
expected_count = oe.regularized_glm_predict(counts, future_df)
```

## Target and objective

Let raw positive weights be `w_i`, normalized weights
`q_i=w_i/sum(w)`, and the encoded, standardized design be `z_i`.
Each candidate minimizes

\[
\sum_i q_i\,\ell(y_i,b_0+z_i'\beta)
 +\lambda\sum_j v_j\left[\rho|\beta_j|
                    +\frac{1-\rho}{2}\beta_j^2\right].
\]

Here `rho=l1_ratio`, `v_j` is the declared nonnegative penalty factor,
and the intercept and forced controls have zero penalty. The binary loss is
`log(1+exp(eta))-y*eta`; the count loss is `exp(eta)-y*eta`, omitting the
outcome-only log-factorial constant. Changing that constant cannot change
coefficients or the selected candidate. `rho=1` gives Lasso, `rho=0` ridge,
and intermediate values mix both penalties. Weighted mean loss fixes the
penalty scale independently of a common rescaling of the observation weights.
This penalized GLM objective follows
[Friedman, Hastie and Tibshirani (2010)](https://www.jstatsoft.org/article/view/v033i01).

Scaling is fitted on the actual training rows using their normalized weights.
With an intercept, encoded columns are centered; `standardize=True` also
divides by their training weighted RMS. Without an intercept, centers are
zero and RMS uses the raw encoded columns. A centered constant contributes
no predictor variation. Its scale is safely one. Response values are never
standardized. The state saves standardized coefficients, the fitted constant
and the transform. `oe.regularized_glm_path` exports original-unit predictive
coefficients and the adjusted constant; prediction replays the saved design.

These are predictive coefficients. `ResultBundle.coefficients` and
`covariance_matrix` are empty, and `inference.available=False`. There are no
coefficient standard errors, p-values, confidence intervals, likelihood-ratio
reference tests or automatic post-selection inference. Ordinary unpenalized
GLM inference cannot be applied to the selected path point. Cross-validation
loss measures prediction within the declared sample and folds; it does not
prove calibration, causal interpretation or generalization to a new population.

## Penalty selection and honest folds

`selection="fixed"` requires a finite nonnegative `penalty`. An optional
unique `lambda_path` must contain that penalty exactly; it is sorted decreasingly.
Every path point is solved and checked; an unconverged point fails the call.

`selection="cv"` scores all retained physical rows once out of fold. A supplied
`lambda_path` is a common fixed absolute penalty grid. Without a supplied grid,
the common candidates are geometric **fractions** of a training-derived
reference penalty: each fold derives its own absolute penalty grid from its
training predictors and outcomes. The selected fraction is then refitted on
the full retained training sample. Validation outcomes do not set that fold's
reference penalty. The reference uses the weighted intercept-only null score,
divided by the positive penalty factors and `max(l1_ratio, 0.01)`, with a
finite lower bound of `1e-8`. With ridge or forced controls this is a numerical
path reference; it does not assert a threshold for an exactly zero solution.

Every fold learns its own centers, scales, categorical levels and reference
grid. Scoring pools the original validation weights over all rows, rather than
equally averaging fold means when their weight totals differ. The criterion
is weighted binary/count negative log-likelihood, without the penalty term.
An exact loss tie selects the largest penalty. There is no one-standard-error
selection rule. A local seeded generator controls the saved physical-row fold
assignments; fitting does not advance the user's global random generator.
The saved state includes each training/validation partition, learned design,
path, absolute grid and complete out-of-fold scoring record. The independent
OpenEconometrics fold convention must be matched explicitly for comparisons
with [glmnet cross-validation](https://glmnet.stanford.edu/articles/glmnet.html).

## Weights and encoding

`weights` names one data column. Raw weights must be finite and nonnegative.
Zero-weight rows are explicitly excluded from the estimation sample, with
their original physical positions and dropped count recorded. Retained weights
are strictly positive. `fweight` additionally requires exactly representable
integers no larger than `2**53`: fixed fits match literal
row replication under the normalized objective. CV holds one physical row's
implicit replicas together in its fold. This is a prediction validation
convention, not the same experiment as randomly scattering explicitly copied
rows across folds.

`ResultBundle.nobs` and saved `physical_nobs` count retained physical rows;
the separate saved `frequency_total` records the sum of admitted frequency
weights. Replication equivalence of the fixed prediction objective does not
turn the physical rows into independent CV units or inferential degrees of freedom.

`aweight` and `pweight` specify normalized empirical loss weighting only.
`pweight` supplies neither survey-design variance nor probability-sampling
inference. No sampling model, survey degrees of freedom or robust covariance
is manufactured for any weight type.

Categorical columns must be declared in `categorical`. There are at most 16
typed levels per column. Their first-appearance order in training rows sets
the reference/level map; treatment coding omits the first level with an
intercept and full one-hot coding is used without one. The map and expanded
design are saved. Penalty factors and forced
controls are specified by original predictor name and propagated to its
encoded columns. Forced controls override their factors to zero. Unknown names,
duplicate forced controls and negative factors fail explicitly. A validation category absent
from its training fold rejects the entire CV call; no observations or folds
are silently excluded to finish scoring. A new query category absent from the
saved full training design also fails. Numeric prediction is algebraic
extrapolation using the saved function; no support interval or extrapolation
confidence is supplied. Poisson response prediction requires a finite link
within `[-700, 700]`; values outside that numeric domain fail rather than
clipping the mean or silently returning an underflowed zero. `kind="link"`
can still return the finite saved linear predictor.

## Samples, persistence and bounds

The estimation sample jointly admits outcome, predictor, categorical and weight
columns. Missing inputs raise by default; `missing="drop"` records retained
original physical positions, counts and hashes. Data-frame index labels may
repeat and are not used as row identity. The caller's input remains unchanged.
Prediction needs only the saved predictor columns, preserves retained query
index labels and uses its own explicit missing policy.

Full predictive state lives in `extra.regularized_glm_state`, including design,
all path points, chosen index, normalized/raw weights, physical rows and CV
records. `ResultBundle.model_dump_json()` and
`ResultBundle.model_validate_json(...)` preserve the state. Prediction and
path readback validate its version, finite values, integrity digest and binding
to the result's specification. They do not refit, tune or infer a new model.
Editing the specification or copying a different family's state cannot turn
the saved function into an admissible result.
The digest is an integrity checksum, not authentication. Even when recomputed,
the saved design, factor/selection options, physical sample counts, supplied
grids or geometric fraction selector, CV partition and pooled loss must remain
internally consistent. A nonfinite link is rejected before applying logistic
sigmoid or exponential transforms, so a saturated response cannot conceal
arithmetic overflow.

The fit bound is 5,000 retained rows, 16 original predictors, 64 expanded
columns and 100 path points. Memory is planned before design/solver allocation;
`max_work` bounds the planned iterative work, including the outer Newton
iterations and inner coordinate/backtracking budgets. The default
`max_iterations=200` is the outer-iteration ceiling, not permission to save an
unfinished fit; its supported maximum is 10,000. `tolerance` is in
`[1e-12, 1e-3]`, with default `1e-8`. Prediction has a separate work
limit and permits at most 100,000 physical query rows. Calls refuse oversized work or workspace rather than truncating rows,
paths, folds or iterations. Supported computation is CPU float64 only;
`Dataset`, CUDA/MPS/automatic acceleration, offsets/exposure, panel/time/cluster
specifications, formula specifications and covariance/inference options are
outside this stage. Binary/count validity, convergence and nonfinite arithmetic
are checked explicitly. This delivery is bounded method-level coverage, with
separate source, frozen-runtime and installed-app acceptance evidence; it is
not blanket GLM coverage or a vendor-parity claim.
