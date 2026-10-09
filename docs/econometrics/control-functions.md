# Two-stage control functions with complete generated-residual uncertainty

The resident implementation covers MARKET-506–513 under MARKET-168: Gaussian,
binary logit/probit/complementary-log-log, Poisson, Gamma, inverse Gaussian and
fractional logit outcomes. Each outcome supports both uncorrected HC0 and whole
one-way cluster CR0 sampling laws. The first stage is OLS for one continuous
endogenous variable; excluded instruments and included exogenous regressors are
declared by the caller. Instrument validity and the control-function conditional
mean restriction are assumptions, not facts established by a successful fit.

| API | Conditional mean | Outcome support |
| --- | --- | --- |
| `cfregress` | identity | finite real values |
| `cflogit` | logistic | binary 0/1 |
| `cfprobit` | standard normal CDF | binary 0/1 |
| `cfcloglog` | `1 - exp(-exp(eta))` | binary 0/1 |
| `cfpoisson` | `exp(eta)` | nonnegative integer counts |
| `cfgamma` | `exp(eta)` | strictly positive values |
| `cfinvgauss` | `exp(eta)` | strictly positive values |
| `cffraclogit` | logistic | fractions in [0,1], including endpoints |

```python
import openecon as oe

fit = oe.cfprobit(
    data=frame, y="participates", endogenous="price", x=["income"],
    instruments=["cost", "distance"], covariance="cluster", cluster="firm",
)
saved = fit.model_dump_json()
conditional = oe.cf_predict(result=saved, data=new_rows)
```

`new_rows` supplies the endogenous variable, all exogenous regressors and every
first-stage instrument. It does not require the outcome. Predictions are
conditional fitted means using the residual implied by those supplied values.
They are not average treatment effects, structural intervention means,
marginalized latent-error probabilities or future-outcome prediction intervals.

## Equations, parameter order and working criteria

Write `z = [constant, exogenous, excluded instruments]`,
`d = z gamma + r`, `x = [constant, exogenous, endogenous]`, and
`q = [x, r]`. The second-stage index is `eta = q beta`, with the last outcome
coefficient `rho` multiplying the generated residual. With `intercept=False`,
the constant is absent in both stages. The complete reported parameter vector
orders every `first_stage:<term>` coefficient before every `outcome:<term>`,
ending in `outcome:ControlResidual`. The result includes the full joint
covariance, including first-stage/outcome cross terms.

The outcome is fitted through its declared score with working dispersion fixed
at one. Gamma and inverse Gaussian dispersion is not estimated by ML; their
working criterion is not reported as a fully estimated outcome distribution.
Fractional logit uses `y eta - log(1+exp(eta))`, with no binomial choose constant
or invented number of trials. LR/AIC or a model-distribution claim would require
a separately justified likelihood domain.

For scalar criterion score `s = d criterion / d eta` and observed derivative
`h = d s / d eta`, the score rows are `[z r, q s]`. The criteria give:

| Outcome | `s` | `h` |
| --- | --- | --- |
| Gaussian | `y - eta` | `-1` |
| Logit/fractional logit | `y - mu` | `-mu(1-mu)` |
| Poisson | `y - mu` | `-mu` |
| Gamma | `y/mu - 1` | `-y/mu` |
| Inverse Gaussian | `(y-mu)/mu²` | `(-2y+mu)/mu²` |

Probit uses the derivative of the observed standard-normal binary criterion;
complementary-log-log uses the derivative of its observed binary criterion.
Both include link curvature. Observed derivatives for noncanonical links and
positive outcomes must not be replaced by expected Fisher information.

## Stacked score covariance and conditional prediction

Let `e_r` select the residual coordinate in q. The complete negative derivative
of the summed estimating equations has block form:

```text
B = [ z'z                                         0             ]
    [ sum(e_r s z' + rho q h z')     -q' diag(h) q              ]
```

It is generally nonsymmetric. The lower-left block includes both changing the
generated design and changing its index. Symmetrizing B, freezing first-stage
residuals or retaining only the second-stage sandwich changes the uncertainty.

For HC0, `M = sum(score_i score_i')`. For CR0, first sum the complete score within
each declared whole cluster, then form `M = sum(score_g score_g')`. Neither meat
is centered or multiplied by HC1/CR1 degrees-of-freedom corrections. The joint
covariance is `B^-1 M B^-T`; singleton clusters reduce exactly to HC0. Clustering
is one-way and permits arbitrary score dependence within the supplied cluster;
no finite-cluster calibration is claimed. Inferential admission requires a
full-rank positive definite joint covariance. CR0 requires more whole clusters
than joint coefficients; singular meat/covariance is refused without PSD repair.

At a supplied prediction row, `r_new = d_new - z_new gamma`. The full index
gradient is `[-rho z_new, q_new]`; the mean gradient multiplies it by the inverse
link derivative. `cf_predict` retains both gradients and the full joint
parameter covariance. This factorization defines every cross-row covariance
as `J C J'` without allocating an unrestricted square output matrix. Columns
include mean/SE, eta/SE and residual. Normal confidence limits transform
`eta +/- z_(1-alpha/2) * SE(eta)` through the inverse link; they cover only
conditional mean parameter uncertainty under the stated asymptotics.

Gaussian outcome coefficients, excluding the residual coefficient, reproduce
2SLS point estimates. The observed stacked covariance coincides with the
corresponding uncorrected 2SLS sandwich in the just-identified scalar case.
Overidentified observed stacked covariance can differ; point equality does
not justify asserting covariance equality. Probit coefficients retain the
conditional standard-normal normalization actually fitted. No automatic
latent-error rescaling converts them to structural or marginal coefficients.

## Saved state, bounds and independent verification

The resident v1 `extra["control_function_state"]` retains declared specification,
physical common sample and original positions, ordered designs, coefficients,
residuals, scalar derivatives, complete row scores, unsymmetrized sensitivity,
score meat, full joint covariance and convergence information. Restoring or
predicting validates this state against its input/sample binding and numerical
semantics. An integrity digest detects accidental changes; it is not external
authentication. Missing-value handling uses one declared common sample across
both stages, including instruments and clusters.

At most 5,000 original rows use the self-contained resident v1 state. Larger
resident frames automatically use bounded replay, and `Dataset` inputs select
the same exact replay path at any size admitted by the work/workspace budget.
The resident source itself remains caller-owned; use `scan()` to avoid loading a
large file into a DataFrame. All paths retain the 16 exogenous predictor,
16 excluded instrument and 36 joint coefficient bounds, with more retained
rows than coefficients. Explicit work and workspace admission also apply.
It provides no weights, multiple endogenous variables,
binary/count/generalized first stages, interactions with residuals, weak-IV
robust sets, endogenous instruments, simultaneous-equation identification,
accelerator support. Saved-model query rows have a separate
bounded Dataset route described below. Rank-deficient designs, unsupported
outcomes, nonconvergence and unidentified or nonpositive outcome information
are refused instead of producing ordinary inferential results.

## Common saved prediction and margins (MARKET-553–560)

All eight fitted outcomes also support the common `oe.predict` and `oe.margins`
APIs on explicit resident covariates or CSV/Parquet `oe.scan()` sources:

```python
restored = oe.ResultBundle.model_validate_json(saved)
query = oe.scan("new_covariates.parquet")
predictions = oe.predict(restored, query, interval="mean", batch_rows=8192)
for block in predictions.iter_batches(batch_rows=8192):
    process(block)  # preserve index and missing-row placeholders
ame = oe.margins(restored, ["income", "price", "cost"], data=query, method="ame")
mem = oe.margins(restored, "price", data=query, method="mem", at={"income": [10, 20]})
```

The query requires the observed endogenous covariate, exogenous covariates and
all excluded instruments; it needs no outcome or training cluster labels.
Supported prediction kinds are `response`, `xb`, `stdp` and `derivative`
(`term=` selects a declared covariate). AME averages continuous conditional-mean
derivatives globally; MEM evaluates at global encoded covariate means.
`at=` replaces declared covariates before each global evaluation. Derivatives
include how the generated residual changes: for covariate v the index slope is
`beta_v + rho * (1[v=d] - gamma_v)`, using zero for an absent coefficient.
These are conditional covariate derivatives, not identified structural APEs.

The complete saved state is semantically validated once before query replay.
Compact Gamma/Beta parameters, covariate roles and full HC0/CR0 covariance are
then reused, without refitting or retaining the training sample per query block.
The common API uses symmetric normal delta-method mean limits, including all
cross-stage covariance blocks. These limits may leave a bounded response range;
they intentionally differ from the legacy `cf_predict` link-transformed limits.
`stdp` returns index uncertainty. Observation intervals, structural targets and
equation/outcome selectors are rejected. The legacy helper is unchanged.
The compact v2 helper also accepts resident evaluation rows with the same
link-transformed limits; use common `predict` for Dataset evaluation rows.

The existing Dataset contract provides bounded projected buffers, missing-row
alignment and typed index preservation, source-integrity checks across passes,
global value/gradient reduction and owned scratch cleanup. Query row counts are
not limited by the resident v1 state's 5,000-row bound. Query and training
measurements are separate; neither implies unlimited-data or GPU support.
Unresolved inverse-link boundaries fail explicitly.

## Exact streamed training and compact saved state

All eight commands accept a replayable `Dataset`. The first stage uses global
TSQR, the generated residual is reconstructed from the global Gamma in every
batch, and the outcome is solved on the complete common sample. The method does
not average batch coefficients. Binary and fractional separation checks,
strict link-domain admission, original-unit rank/condition checks, observed
derivatives and the complete nonsymmetric stacked sandwich are preserved.
Whole-cluster score sums combine labels across batch boundaries before CR0.

```python
source = oe.scan("training.parquet")
fit = oe.cfprobit(
    data=source, y="participates", endogenous="price", x=["income"],
    instruments=["cost", "distance"], covariance="cluster", cluster="firm",
    batch_rows=8192, max_iterations=20,
)
saved = fit.model_dump_json()
restored = oe.cf_restore(result=saved, data=source)
conditional = oe.predict(restored, data=oe.scan("evaluation.parquet"))
effects = oe.margins(restored, data=source, variables=["income", "price"])
```

The compact `openecon.control_function.stream.v2` state retains the declared
specification, global coefficients, complete bread/meat/covariance and bounded
source/sample receipts. It does not embed training designs or scores. Semantic
replay requires the unchanged estimation source: local CSV/Parquet paths may be
resolved from the saved descriptor; moved files, in-memory frames and batch
factories require explicit `cf_restore(result=..., data=...)` rebinding. The
source contents, retained positions and typed index/cluster identities are
validated again. A digest alone is not semantic validation or authentication.
The source is needed even when evaluation rows come from another file.
Nonlinear restoration checks the recorded refined Newton criterion using a
diagonally equilibrated observed-information norm and an explicit float64
score-reduction error bound. Coherently rewriting all reported statistics does
not make nonstationary coefficients valid.

For an unchanged single physical CSV/Parquet source, a transient validation
cache can reuse the complete numerical replay. Each lookup still checks the
full file bytes and all reporting JSON by SHA256, plus the source handle and
caller budgets. Fresh JSON reloads and changed files or reports require full
semantic replay. Frame/factory and directory sources never use this cache.

`batch_rows` is a bounded buffer target from 1 to 65,536, constrained further by
workspace admission. `max_work` admits a conservative complete computation,
including the declared maximum nonlinear iterations and separation certificate;
large N with a large iteration/parameter budget can be refused. Reducing
`max_iterations` is valid only if the complete solver then converges. There is
no fixed Dataset row cap and no claim that every geometry fits every budget.
CPU float64 and the statistical scope above remain unchanged.

`benchmarks/control_prediction_acceptance.py` fixes its physical inputs before
measurement: eight 100k-row Parquet queries and Gaussian/Poisson 1m-row CSV
queries, including missing covariates. A separate NumPy/pandas-only checker
compares every response/derivative/SE/CI, full Gamma/Beta finite-difference
gradients, global AME/MEM, and all 16 HC0/CR0 small cases. It imports no SciPy,
Torch or SDK. Physical query rows repeat a fixed synthetic covariate geometry;
this is not a million-row estimation or cold-cache hardware comparison.

The development oracle in `scripts/verify_control_function_oracles.py` imports
neither OpenEconometrics nor Torch. NumPy/SciPy independently refit all eight
criteria, central-difference the entire nonsymmetric stacked Jacobian, construct
HC0/whole-cluster CR0 meat and all cross-covariance blocks, and difference the
complete conditional prediction map. Tests include singleton-cluster equality,
just- versus overidentified Gaussian 2SLS, probit normalization, fractional
endpoints, saved prediction cross-row uncertainty and common physical samples.
The receipt command is:

```text
python scripts/verify_control_function_oracles.py \
  --native-receipt path/to/native-receipt.json --report path/to/oracle-report.json
```

A successful comparison establishes the supplied cases and uncertainty domains.
Source tests, frozen-runtime identity, installed native output and full restart
persistence are separate evidence. No licensed Stata execution or blanket vendor
parity is claimed. Existing `ivprobit` two-step inference uses its separate Newey
MCS contract; it is not silently replaced by this generic residual-inclusion
stacked estimator.

The conceptual scope follows Wooldridge's
[control-function lecture](https://www.nber.org/sites/default/files/2022-09/slides_6_controlfuncs.pdf)
and Papke and Wooldridge's original
[fractional-response methods paper](https://www.nber.org/papers/t0147)
(NBER Technical Working Paper 147, 1993; published in 1996).
The official [Stata cfregress manual](https://www.stata.com/manuals/rcfregress.pdf)
and [cfprobit manual](https://www.stata.com/manuals/rcfprobit.pdf) are specification
references and future comparison targets. Their broader first-stage,
interaction and covariance options are outside this bounded implementation.

Binary and fractional fits use the existing native global-replay separation certificate before optimization and saved replay. Complete or quasi separation, ambiguous precision and exhausted certificate budgets refuse the fit. Fractional interior rows impose both signed constraints, so valid finite mixed-interior models remain admissible. Certificate work is included in the complete fit/replay/prediction budget.
