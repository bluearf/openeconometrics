# Saved-result prediction and marginal effects

`oe.predict` and `oe.margins` cover **58 estimator names**, with explicit
conditions on fitted options and response targets. The [generated inventory](../capabilities.md)
tracks registered names and native Dataset fitting routes; fitting does not establish prediction
coverage. These interfaces reconstruct the supported target from the
saved `ModelSpec`, coefficients, covariance and fitted category metadata. They
do not refit the model or invoke an optimizer. New evaluation rows need the
original mean predictors and any recorded offset, exposure or binomial trials.
The outcome, instruments, censoring status and censoring endpoints are not
needed for the direct adapters below. Auxiliary beta-precision and ordinary
negative-binomial dispersion predictors do not affect those unconditional means.
Mixture/hurdle counts and heteroskedastic probit need predictors from both fitted
equations; truncated negative-binomial conditional means also depend on dispersion.

Existing fitted OLS objects still call their original prediction and margins
methods, including their retained design, weights, streamed contrasts and OLS
inference adjustments. Their existing default `kind='xb'` is preserved. Generic
saved bundles default to `kind='response'` and require explicit evaluation data.

```python
import openecon as oe
from openecon.models import ResultBundle

model = oe.poisson(data=df, y="count", x=["income", "region"],
                   categorical=["region"], exposure="years", covariance="robust")
saved = ResultBundle.model_validate_json(model.model_dump_json())

# new_rows contains income, region and years, without count.
oe.predict(saved, new_rows, kind="response", interval="mean")
oe.predict(saved, new_rows, kind="xb")
oe.predict(saved, new_rows, kind="stdp")
oe.predict(saved, new_rows, kind="derivative", term="income", interval="mean")
oe.margins(saved, ["income", "region"], data=new_rows, method="ame")
oe.margins(saved, "income", data=new_rows, method="mem")
oe.margins(saved, "income", data=new_rows,
           at={"income": [10, 20], "region": ["North", "South"]})
```

## Supported response adapters

The 21 direct adapters below support response means/probabilities, the linear index `xb`, its standard
error `stdp`, continuous response derivatives, and AME/MEM. They support ordinary
numeric predictors and saved treatment-coded categories. The additional 20
linear/population adapters and five mixture/truncated-count adapters follow their
own saved domains below. Unsupported formula designs remain rejected.

| Estimator | Response mean and fitted link |
| --- | --- |
| `ols`, `cnsreg`, `ivregress` | Identity linear mean. Constrained fixed coefficients are included with zero sampling uncertainty. IV evaluation requires endogenous regressors, without instruments. |
| `logit`, `probit`, `cloglog` | Binary success probability through the corresponding inverse link. |
| `hetprobit` | Positive-outcome probability `Phi(X beta exp(-Z gamma))`, using both saved equations and their full covariance. `sigma` returns latent-error standard deviation `exp(Z gamma)`. |
| `fracreg` | Fractional mean using fitted logit or probit. |
| `betareg` | Beta-response mean using fitted logit, probit, cloglog or loglog. Precision coefficients do not enter the mean. |
| `poisson`, `nbreg`, `gnbreg` | `exp(X beta + offset)` or `exposure * exp(X beta)`. NB1/NB2 and generalized dispersion coefficients do not enter the mean. |
| `cpoisson`, `cnbreg` | **Latent unconditional count mean** `exp(X beta + offset)`, including NB1/NB2. This is neither the expected recorded top-coded count nor a mean conditional on a censoring event. |
| `glm` | The saved inverse link for Gaussian, binomial, Poisson, gamma, inverse Gaussian or fixed-dispersion negative binomial. Binomial models with trials return expected success counts, `trials * probability`. |
| `tobit` | Expected **recorded censored** normal outcome, using the resolved fitted lower/upper limits. `latent` gives the underlying unconditional normal mean; `conditional` gives its mean conditional on lying between those fitted limits. |
| `truncreg` | Expected normal outcome **conditional on inclusion** between the persisted fitted truncation limits. `latent` gives the untruncated population mean; `conditional` is an explicit synonym for this model's response mean. |
| `intreg` | **Latent unconditional normal mean**. Interval endpoints describe the estimation observations; they do not define an observation or censoring mechanism for new rows. `conditional` is rejected because this model does not persist common prediction limits. |
| `ologit`, `oprobit` | Probabilities of each persisted ordered outcome, with direct fitted cutpoints. `xb`/`stdp` use the common latent index; logistic scale and probit variance are fixed normalizations. |
| `mlogit` | Probabilities of every persisted outcome, including the fitted base. `xb`/`stdp` use category-versus-base log odds; the base index and its uncertainty are exactly zero. |

### Linear, population and quantile targets

The additional names are `areg`, `reghdfe`, `ivreghdfe`, `xtreg`, `xtivreg`,
`xtgls`, `xtpcse`, `xtfmb`, `prais`, `rreg`, `qreg`, `bsqreg`, `iqreg`, `sqreg`,
`xtgee`, `ppmlhdfe`, `mixed`, `xtlogit`, `xtprobit` and `xtpoisson`.

Structural linear fits predict their recorded coefficient response. Random-effect
linear fits and `mixed` return the population linear mean, without conditioned
group BLUPs. `rreg` returns a robust location; `qreg`/`bsqreg` return a quantile;
`iqreg` returns the interquantile difference. `sqreg` needs an explicit saved
quantile equation, such as `outcome='q25'`, when several equations exist.

Absorbed fits and panel FE fits support **`xb`, `stdp` and
`margins(kind='xb')` only**. Their index excludes unpersisted group effects;
response predictions and response-scale derivatives are refused. Between fits
also use index-only targets and require explicitly supplied panel-mean covariates.
First-difference fits reject raw-level prediction. Panel logit/probit/Poisson
support the population-averaged target only with `model='pa'`; FE/RE likelihood
responses need separate integration or conditional-state adapters.
The [prediction domains](prediction-domains.md) document defines each target,
including structural versus disturbance forecasts for time/panel linear fits.

### Mixture, hurdle and truncated counts

| Estimator | `response` | `conditional` |
| --- | --- | --- |
| `zip`, `zinb` | `(1 - inflation probability) * latent count mean` | Positive-count conditional mean; inflation cancels from this conditional target |
| `hurdle` | Participation probability times positive-count conditional mean | Positive-count conditional mean |
| `tpoisson`, `tnbreg` | Count mean conditional on exceeding the persisted fitted lower cutoff | Same conditional target |

Their `xb` and `stdp` describe the count-equation index, including its offset,
and exclude the auxiliary index. Response/effect inference retains the **full
saved covariance**, including count/auxiliary/dispersion cross blocks. Numeric
effects and categorical changes reconstruct all fitted equation roles.

Row-specific lower cutoffs must be nonnegative integers: at most `1e12` for
Poisson and `1e7 - 1` for negative binomial. MEM requires the global encoded
cutoff to remain a valid integer; an average of differing cutoffs need not satisfy
this domain. Prepare a valid constant cutoff in the evaluation source for such
a MEM target. A cutoff that is not an original predictor is not an `at` grid
intervention. Native tail recurrences use bounded workspaces and report explicit
precision or convergence errors when they cannot certify a conditional mean.

GLM uses its existing family/link compatibility rules. Supported link names are
identity, log, logit, probit, cloglog, loglog, power, reciprocal, inverse squared
and the canonical negative-binomial link. Power uses the saved exponent;
canonical negative binomial uses the saved fixed dispersion. New rows must stay
inside the link domain and yield finite means in the family support. No clipping
is used to turn invalid means into predictions.

`xb` includes the recorded offset or log exposure. `stdp` measures uncertainty
of this linear index. `derivative` differentiates the response mean with respect
to one continuous original predictor. For binomial trials this is the derivative
of the success-count mean. A predictor also serving as offset, exposure or trials
needs a separate intervention contract and is rejected for marginal effects.

### Censored, truncated and interval normal responses

```python
saved = ResultBundle.model_validate_json(
    oe.tobit(data=df, y="hours", x=["income"], ll=0).model_dump_json())
oe.predict(saved, new_rows, kind="response", interval="mean")  # E[max(0, Y*)|X]
oe.predict(saved, new_rows, kind="latent")                     # E[Y*|X]
oe.predict(saved, new_rows, kind="conditional")                # E[Y*|Y*>0, X]
oe.margins(saved, "income", data=new_rows, kind="conditional")
```

For these three models, `xb` includes the fitted offset and equals the latent
normal mean. `stdp` gives its standard error. `kind='derivative'` differentiates
the model's **response** contract in the table above; `margins(kind='latent')`
and `margins(kind='conditional')` select effects on those explicit alternatives.
Tobit limits requested as `ll='min'`/`ul='max'` use the numeric limits resolved
during fitting, without reading new outcome values. Saved `/sigma` parameters
for Tobit/truncated regression and `/lnsigma` for interval regression retain
their original reporting scale; delta-method inference includes their full
covariance with the mean coefficients. Normal-response AME/MEM and categorical
contrasts use this same full parameter vector.

For finite standardized bounds `a=(lower-mu)/sigma`,
`b=(upper-mu)/sigma`, the observed censored mean is
`lower*Phi(a) + mu*(Phi(b)-Phi(a)) + sigma*(phi(a)-phi(b)) + upper*Phi(-b)`;
the corresponding terms disappear for open bounds. Its derivative with respect
to the latent mean is the probability of lying between the limits. The bounded
conditional mean is `mu + sigma*(phi(a)-phi(b))/(Phi(b)-Phi(a))`, whose derivative
with respect to `mu` equals its conditional variance divided by `sigma**2`.
These are the separate censored and truncated expectation definitions in the
[official Tobit postestimation manual](https://www.stata.com/manuals/rtobitpostestimation.pdf)
and [truncated-regression manual](https://www.stata.com/manuals/rtruncregpostestimation.pdf).
OpenEconometrics's `response` default is defined explicitly above; it differs from
Stata's `xb` default. Interval regression's latent index is consistent with the
[interval-regression postestimation manual](https://www.stata.com/manuals/rintregpostestimation.pdf).

Central conditional windows use analytic normal moments. Far tails and narrow
windows use normalized, cached 96-node native Torch quadrature in shifted/scaled
coordinates, accumulated in blocks of at most 4,096 rows. This avoids subtracting
near-equal Mills ratios or canceling a large latent index against its tail mean.
The moment calculations retain sigma derivatives and the gradients of continuous
effects. Narrow censored windows also integrate the density and derivative
moments directly. Their analytic autograd rules retain small sigma cross-gradients
that would be lost by subtracting endpoint/CDF gradients; clipped means use
anchored interval algebra. Nonfinite standardized bounds and bounds beyond `1e100` standard
deviations are refused: central third-moment underflow can otherwise spoil the
mixed derivative while that derivative still has a representable value.
Conditional windows narrower than the smallest normal float64 value after
standardization are refused because their coordinates cannot retain relative
precision; they are never replaced by an endpoint prediction.
No observation-level prediction intervals, conditional residuals or alternate
row-specific prediction limits are supplied by these adapters.

### Ordered and multinomial outcomes

```python
saved = ResultBundle.model_validate_json(
    oe.ologit(data=df, y="rating", x=["income", "region"],
              categorical=["region"]).model_dump_json())
oe.predict(saved, new_rows, interval="mean")  # all fitted-category probabilities
oe.predict(saved, new_rows, outcome="good", interval="mean")
oe.predict(saved, new_rows, kind="derivative", term="income", outcome="good")
oe.margins(saved, ["income", "region"], data=new_rows)  # one row per outcome/effect
oe.margins(saved, "income", data=new_rows, method="mem", outcome="good")
```

`outcome` selects an actual persisted category label, not a category position.
Strings, finite numbers and booleans retain their identities: `True` cannot
select a numeric category `1`, and string `"1"` cannot select numeric `1`.
Omitting it returns all category probabilities/effects in fitted order. Wide
prediction columns are named `response[<JSON label>]`, with matching
`std_error`, `ci_low` and `ci_high` columns for intervals; JSON quoting prevents
string/numeric label collisions. A selected outcome retains the ordinary scalar
column names. `attrs['outcome_labels']` and `attrs['outcome_columns']` preserve
the complete label mapping. Margins include an `outcome` column.

Ordered probabilities are `F(cut_j - eta) - F(cut_(j-1) - eta)`, with open
outer boundaries. `F` is logistic or standard normal. Thresholds are the saved
direct `/cut1`, ... parameters; there is no extra fitted scale parameter.
`xb` and `stdp` describe a single common latent index and reject `outcome`.
Multinomial probabilities use stable softmax of all category indexes, with
the fitted base index fixed zero. Multinomial `xb`/`stdp` can select an outcome
or return all category indexes. These probability/index definitions follow the
[ordered-logit](https://www.stata.com/manuals/rologitpostestimation.pdf),
[ordered-probit](https://www.stata.com/manuals/roprobitpostestimation.pdf) and
[multinomial-logit](https://www.stata.com/manuals/rmlogitpostestimation.pdf)
postestimation manuals; OpenEconometrics always requests evaluation data explicitly.

Probability and effect uncertainty uses the entire saved parameter covariance,
including threshold/slope and cross-equation covariance. Continuous effects are
native analytic location/pairwise multinomial derivatives; categorical effects are discrete
probability changes against the fitted predictor reference. AME/MEM retain the
sample, encoded-mean and weighting rules below. Both aggregate-effect gradients
and row-interval Jacobians use native analytic Torch formulas in the full saved
parameterization, with no matrix of row-output seeds. Allocation guards count
all outcomes and live gradient intermediates even when one outcome is selected.
Their allocation is bounded to 256 MiB; larger row predictions require batches,
and excessive AME gradient allocations raise `prediction_memory_limit`.

Narrow ordered intervals integrate the density with eight-node native
Gauss-Legendre quadrature, retaining physical threshold differences and stable
first/second location derivatives. Normal tails use erfc and logistic tails use
log-domain interval algebra. Complementary probabilities, signed log products
and complete pair/triple multinomial weights preserve effects after a dominant
probability rounds to one or a density underflows before slope/design scaling.
Covariance-weighted uncertain coordinates are normalized in log space, retaining
standard errors whose squared variance would underflow. Fixed parameters cannot
erase small uncertain derivatives. A nearly singular covariance projection with
extreme coordinate scales that float64 cannot certify raises
`prediction_precision`; nonfinite indexes, threshold differences, confidence
limits or marginal-effect statistics are refused.
Saved categories, links, direct cuts, category counts, base selection, equation
blocks, omissions and covariance are validated before evaluation. Classification,
scores, stochastic predictions, pairwise index-difference inference, free-scale
ordered models and proportional-odds diagnostic tests are outside these adapters.

## Design, sample and inference

The fitted category levels and reference are reused even when the new sample
contains only one level or a different category order. Unseen levels are rejected.
Reported rank omissions contribute zero; constrained fixed terms contribute their
saved values. Coefficient order follows the stored parameter records, including
auxiliary equations, rather than the order of new columns. Full parameter
covariance enters delta-method inference; mean gradients for dispersion or
precision parameters are zero for direct unconditional count and beta-response
adapters. Limited normal means can depend on sigma, while mixture/truncated
counts can depend on auxiliary or dispersion parameters.

With `missing='drop'`, predictions preserve every original index and duplicate
index position, placing NaN in excluded rows. Resident outputs list positional
indexes in `output.attrs['missing_row_positions']`; Dataset outputs preserve the
index on disk and record `metadata['analysis']['missing_prediction_rows']`
without retaining a full-sample position manifest. With `missing='raise'`, missing
prediction inputs raise an error. Missing outcomes in evaluation data have no
effect because outcomes are not used.

`interval='mean'` gives pointwise symmetric delta-method confidence intervals for
the chosen mean or derivative. The variance is `J V J'`. Intervals use the fitted
normal or Student-t inference record and saved alpha, with an optional alpha
override. A symmetric probability interval may extend outside `[0, 1]`; it is not
clipped or presented as a transformed-link interval. Observation/prediction
intervals, residuals, censor/tail probabilities and classifications are unsupported.
Fitted ordered/multinomial category probabilities are the explicit exception to
the scalar-response contract.

AME averages derivatives or discrete contrasts over the explicitly supplied
evaluation rows. It does not infer the estimation sample from a chart sample.
For an average over the complete fitted sample, explicitly supply its eligible
original rows. Resident fits can use their retained sample positions; streamed
fits do not retain that full manifest. Reapply the fitted sample selection,
missing and weight rules rather than using a stored chart preview. MEM evaluates at weighted means of the **encoded
design**, offset and binomial trials, retaining fractional category proportions
rather than inventing a modal category. Categories report each nonreference level
minus the fitted reference; continuous predictors use analytic inverse-link
derivatives. `kind='xb'` gives effects on the linear-index scale.

When the fit records weights, margins require that weight column in the evaluation
data. Nonnegative weights are normalized safely by their maximum before summing;
finite weights near the float64 limit remain scale invariant. Frequency weights
must be integers. Zero-weight rows are excluded before category encoding and link
evaluation: they neither add unseen levels nor overflow otherwise valid effects.
`attrs['evaluation_rows']` records the positive-weight complete sample;
`input_evaluation_rows`, `complete_evaluation_rows` and `zero_weight_rows_excluded`
record the sample accounting. The full saved parameter covariance is applied to the gradient
of each aggregate effect. Automatic differentiation is confined to the parameter
vector, without fitting or changing coefficients.

Margins tables contain scalar estimates, standard errors, statistics, distributions,
degrees of freedom, p-values and confidence limits. Diagnostic derivatives are
separate in `output.attrs['delta_gradients']`, in the column order recorded by
`output.attrs['parameter_terms']`. A deterministic fixed/omitted coefficient can
have zero effect variance: its interval is the exact estimate and statistic/p-value
are undefined, instead of fabricated infinite statistics.

Evaluation grids accept original predictors and at most 1,000 combinations.
Resident row-design and Jacobian allocations remain guarded; use Dataset inputs
for evaluations larger than those resident workspaces.

## Full Dataset evaluation

All 58 common saved adapters accept a replayable `Dataset`, within the same
fitted-option and response-domain conditions. Computation uses bounded native
CPU float64 blocks. `batch_rows` applies only to Dataset evaluation, from 1 to
65,536; the default adapts to the recorded model/gradient workspace.

```python
rows = oe.scan("evaluation.parquet")
predictions = oe.predict(saved, rows, interval="mean", batch_rows=16_384)
predictions.head()  # small resident preview
for block in predictions.iter_batches():
    consume(block)

effects = oe.margins(saved, ["income", "region"], data=rows, method="ame")
at_means = oe.margins(saved, "income", data=rows, method="mem")
grid = oe.margins(saved, "income", data=rows,
                  at={"income": [10, 20], "region": ["North", "South"]})
```

`predict` completes an **owned indexed temporary Parquet Dataset** before
returning. The output is disk-backed and survives later source changes; it
retains the original index values/order/duplicates and inserts NaNs for excluded
rows. Its temporary storage is owned by the result and is cleaned up on failure
or release. Output columns and any fitted outcome-label mapping remain explicit.

AME globally reduces weighted effects **and their full parameter gradients**
before applying the saved covariance once. It does not average batch standard
errors. MEM first reduces the entire weighted encoded design, offsets/trials and
other required roles, then evaluates at those means. Each `at` grid cell uses
the corresponding full evaluation population. These procedures return small
resident effect tables with their existing inference and gradient metadata.

Complete source replays validate selected-column content and index identity,
including dropped rows. Changes between passes are refused. The stored **400-row
fit/chart preview is independent of full Dataset prediction and margins**.
No shared total-row ceiling applies, but model width, category/outcome expansion,
live covariance/gradient buffers, source-reader allocations, repeated I/O and
local output/scratch capacity remain real constraints. A buffer plan is not a
bound on total process RSS, and no universal hardware speed is claimed.

[Chunked helper analyses](streaming_helpers.md) cover t/variance tests,
one-way/factorial ANOVA, Pearson/partial correlations, PCA and factor analysis.
They share native inference and publication-table exports. PCA/factor observation
scores are **lazy replayable Datasets**, unlike common prediction's completed
owned Parquet output; exhaust a score pass to complete its source-integrity check.
Exact rank correlations, repeated-measures ANOVA, MANOVA and grouped descriptive
quantiles also accept Dataset inputs through bounded external adapters.
See [streaming helpers](streaming_helpers.md) and the recorded million-row probe.

Static formula/system/selection/frontier targets are listed in
[advanced saved prediction](advanced-prediction.md).

## Remaining generic adapters

At this implementation snapshot, 58 registered estimator names have
common saved prediction adapters. The following established families remain unsupported by
this generic path. Model-specific prediction or forecasting APIs are separate and
are preserved when a fitted object already implements its own method.

| Family | Remaining names |
| --- | --- |
| Discrete | `clogit` |
| Time series | `arima`, `arch`, `var`, `vec`, `ardl`, `ucm`, `mswitch`, `nardl` |
| Survival | `stcox`, `streg` |
| Dynamic panel | `ahreg`, `xtdpd` |
| Causal | `teffects`, `didregress`, `eventstudy`, `rdrobust`, `csdid` |
| Mixed likelihood | `melogit`, `meprobit`, `mepoisson` |
| General moments | `gmm` |

The [remaining prediction domains](prediction-domains.md) describe the missing
state and target semantics for these names, and distinguish existing
`forecast`, `mixed_predict`, `stcurve` and other family-specific APIs.

Missing adapters raise `unsupported_prediction` with the estimator name; they do
not silently interpret an unknown model as a linear regression. Malformed saved
categories, links, inference degrees of freedom or covariance produce structured
errors. Plain saved OLS bundles carrying contrast-specific Hansen/DF adjustments
need the retained native OLS contrast state and are refused.

Independent tests compare forward links and delta gradients against NumPy/SciPy
finite differences, binary and Poisson results against statsmodels, and saved
censored-count means against their coefficient algebra. Normal limited-response
means, tails, narrow windows and ancillary gradients are checked against
independent adaptive quadrature and finite differences. JSON round trips,
coefficient permutations, missing/duplicate indexes, fixed coefficients, omitted
terms, large finite weights and no-refit behavior are checked. These are local
library checks; they do not establish licensed Stata parity or an installed desktop
release containing these changes.

Ordered/multinomial tests additionally compare every outcome's probabilities,
continuous/dummy AME/MEM, complete delta gradients and full-covariance confidence
limits with independent SciPy/NumPy algebra and finite differences. Narrow-window
probabilities/effects have 100-digit stdlib Decimal oracles and tail quadrature
checks. Base indexes, typed outcome labels, JSON restoration, coefficient
permutations, probability/effect/gradient sums and malformed saved state are
covered. Independent adversarial tests use closed-form tail derivatives and
Cholesky/hypot covariance oracles, including underflow followed by slope/design
rescaling, mixed predictor scales, rare third categories, tied dominant indexes,
nearly PSD projected-negative covariance and degenerate covariance. Memory
guards and latent-index slope gradients are checked separately. No runtime
estimator is delegated to an external statistics package.

## Heteroskedastic probit: two saved equations

The native `hetprobit` adapter follows the latent normalization
`Pr(y > 0 | X,Z) = Phi(a)`, where `m = X beta`, `l = Z gamma`,
`sigma = exp(l)` and `a = m exp(-l)`. Its variance equation has no constant.
This normalization and the meanings of probability, linear-index and sigma
predictions follow the [official model manual](https://www.stata.com/manuals/rhetprobit.pdf)
and [postestimation manual](https://www.stata.com/manuals/rhetprobitpostestimation.pdf).
`response` (aliases `pr` and `probability`) predicts the positive-outcome
probability; `xb` is the mean-equation index, `stdp` its standard error, and
`sigma` is the latent-error standard deviation. These scalar predictions do
not select or classify response labels. Scores, residuals, observation intervals,
row-specific offsets and alternate variance normalizations are unsupported.

For a continuous original predictor `v`, the response effect is
`phi(a) exp(-l) [beta_v - m gamma_v]`. Missing and omitted equation-specific
coefficients in this expression are zero. A shared predictor changes both
equations; a variance-only predictor can change the probability despite having
no mean slope. Sigma and linear-index effects are `sigma gamma_v` and `beta_v`.
Categories use finite changes against their stored reference, changing both
equations together. `margins(..., kind='response'/'xb'/'sigma')` supports weighted
AME, encoded weighted-mean MEM and explicit evaluation grids. With `gamma=0`,
probability/effect points reduce to ordinary probit; their standard errors still
include uncertain variance parameters unless the stored covariance fixes them.

All probability, sigma and effect Jacobians are analytic and use the complete
saved mean/variance covariance, including cross-equation blocks. Componentwise
signed log products preserve representable density-times-scale/slope/design
products; covariance-weighted scaling preserves small uncertain coordinates next
to large fixed ones. Covariance standard deviations enter the analytic log
products **before** a raw derivative can underflow; inference then projects these
scaled gradients through the saved correlation matrix. Thus a raw float64
diagnostic derivative may round to zero while its uncertainty remains nonzero.
The shared-predictor effect is factored before density multiplication, preserving
exact cancellations between its mean and variance paths even with large
covariance. The internal estimation centres have already been removed
from the reported coefficients and covariance and are **not reapplied**.
Native roles, all retained/omitted terms, equation labels, categorical coding,
positive binary counts and scale metadata must agree. JSON restoration,
permuted coefficients, outcome-free evaluation data, missing-row alignment and
zero-uncertainty effects retain the generic saved-result contracts.

Explicit precision errors reject nonfinite indexes, an overflowing standardized
index, or a nonzero standardized index that underflows and cannot certify later
derivative products. Nonfinite or underflowed nonzero factors needed to certify
response effects, or underflowed nonzero covariance-scaled gradient products,
are explicitly refused rather than reporting unproved zero uncertainty.
A requested sigma must be strictly positive and finite in
float64. Response predictions can remain valid when sigma itself is outside
that range, provided every requested intermediate and result is certifiable.
All full-parameter Jacobian workspaces are budgeted conservatively at thirty-two
float64 matrices within 256 MiB, including MEM and continuous/category effects;
larger evaluations use Dataset blocks. This adapter is CPU float64; its common
Dataset wrapper supports global AME/MEM/grids and bounded row output, within
the same numerical-domain checks. It does not claim GPU execution or unlimited
model geometry. Independent NumPy/SciPy closed formulas and
finite differences verify full Jacobians, weighted/dummy effects, degenerate
covariance, rescaled underflow products and saturated tail contrasts. These
source tests do not establish an installed desktop release or licensed parity.
CPU allocation is scoped to each hetprobit call and restores the caller's Torch
device context. `xb`/`stdp` evaluate only the mean index, while `sigma` evaluates
only the scale index; overflow in an unrelated equation does not reject those
otherwise finite outputs. Evaluation data still follow the saved complete-input
contract for both equations.
