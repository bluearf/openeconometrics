# Adding an estimator family

Every OpenEconometrics estimator is implemented in this repository on float64 PyTorch
tensors. No third-party estimation library (statsmodels, linearmodels, SciPy
optimizers or distributions, scikit-learn) runs at fit time. Those packages may
appear in tests only, as independent oracles.

This page is the contract between the shared layer and an estimator family.

## Layout

```
src/openecon/econometrics/
    registry.py              Torch-free catalogue; validates ModelSpec, drives oe.fit and oe.capabilities
    core.py                  ModelFrame (sample), Design, ml_covariance, wald_test, build_result
    <family>/__init__.py     ESTIMATORS = (EstimatorInfo(...), ...)   -- imports nothing heavy
    <family>/*.py            kernels (tensors only) and the analysis layer (spec -> ResultBundle)
src/openecon/engines/
    distributions.py         normal / t / chi2 / F tails and quantiles, Gauss-Hermite nodes
    linalg.py                weighted QR least squares, Stata-style collinearity screen, Wald statistic
    covariance.py            HC, cluster, multiway, HAC, Driscoll-Kraay sandwich meats
    absorb.py                multiway fixed-effect absorption, singletons, degrees of freedom
    optimize.py              Newton and BFGS maximizers, numerical derivatives, information inverse
tests/test_econ_<family>*.py
docs/econometrics/<family>.md
```

`<family>/__init__.py` must not import Torch, pandas or the family's own
estimation modules: `oe.capabilities()` and spec validation read it in the web
control process, where the tensor runtime is deliberately not loaded.

## The specification

`ModelSpec` carries the fields common to all estimators (`outcome`,
`predictors`, `categorical`, `intercept`, `covariance`, `cluster`, `missing`,
`alpha`, `weights` + `weight_type`, `panel`, `time`) and two open containers:

- `columns`: additional named column roles, e.g. `{"endogenous": ["price"],
  "instruments": ["z1", "z2"]}`.
- `options`: estimator settings, e.g. `{"model": "fe"}`.

The family declares, per estimator, exactly which roles, options, weights,
covariance estimators and panel/time structure are legal:

```python
from openecon.econometrics.registry import EstimatorInfo, Option, Role

ESTIMATORS = (
    EstimatorInfo(
        name="xtreg", title="Panel-data linear regression", family="panel",
        entry="openecon.econometrics.panel.estimators:fit_xtreg",   # (spec, data) -> ResultBundle
        function="xtreg",                                           # exported as oe.xtreg
        stata=("xtreg",),
        covariances=("nonrobust", "HC1", "cluster"), default_covariance="nonrobust",
        panel="required", time="optional", weights=("aweight", "fweight", "pweight"),
        inference="t",
        options=(Option("model", "str", "fe", choices=("fe", "re", "be", "fd")),),
        description="...",
    ),
)
```

Anything not declared is rejected when the spec is constructed. Covariance names
are canonical across OpenEconometrics (`registry.COVARIANCE_KINDS`): `nonrobust`,
`HC0`..`HC3`, `robust`, `cluster`, `opg`, `hac`, `driscoll_kraay`, `bootstrap`,
`jackknife`. Several cluster columns with `covariance="cluster"` mean multiway
clustering (raise `cluster_dimensions`).

Each estimator has a Stata-style convenience function with explicit keyword
parameters and a full docstring. It only builds a spec and calls `fit`:

```python
def xtreg(*, data, y, x, panel, time=None, model="fe", covariance=None, cluster=None,
          categorical=None, weights=None, weight_type=None, missing="raise", alpha=0.05):
    """Docstring: model, estimator, formulas, covariance conventions, Stata equivalent, example."""
    spec = make_spec("xtreg", outcome=y, predictors=column_list(x, "x"), panel=panel, time=time,
                     covariance=covariance, cluster=cluster, categorical=column_list(categorical, "categorical"),
                     weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
                     options={"model": model})
    return fit(spec, data=data)          # from openecon.analysis import fit
```

## The entry function

```python
def fit_xtreg(spec: ModelSpec, data) -> ResultBundle:
    frame = ModelFrame(spec, data)                 # sample selection, missing policy, weights
    frame.sort_panel()                             # by panel, time (duplicate periods are an error)
    design = frame.design(intercept=...)           # treatment-coded categoricals
    design = frame.drop_collinear(design, weights) # Stata-style omission, recorded in provenance
    y, w = frame.numeric(spec.outcome), frame.weights()
    ... tensor kernel via kernel_call(...) ...
    return build_result(frame, terms=design.terms, params=beta, covariance=V, ...)
```

`ModelFrame` gives `numeric(name)`, `matrix(names)`, `codes(names) -> (int64
codes, G)`, `cluster_dimensions()`, `weights()`, `option(name)`, `role(name)`,
`time_index()`, `restrict(mask, reason)`, `reorder(order, by)`, `sort_panel()`,
`design(...)`, `drop_collinear(...)`, `warn(message)`. `allow_missing=[...]`
keeps rows whose missing value is meaningful to the estimator.

Rules for entry functions:

- Raise `AnalysisError(code, message)` for every invalid input or numerical
  failure; wrap kernels with `kernel_call` so `KernelError` is translated. The
  message tells the user what to change. Never return a partially valid fit.
- Never silently change the model. Dropping collinear terms, singletons or
  unusable observations must go through `frame.warn` / `frame.restrict(...,
  reason)` so that it is recorded in the result.
- Follow the Methods and formulas of the equivalent Stata command for the
  estimator, the covariance small-sample factors and the reference
  distribution, and say which convention is used in `inference["correction"]`.
- `provenance["stata_parity_validated"]` stays `False`: we do not claim parity
  that has not been measured against Stata output.

## The result

`build_result` produces the `ResultBundle`. Conventions:

- `terms` are unique. The constant is named `Intercept`. Multi-equation models
  pass `equations=[...]` and prefix auxiliary terms (`select:age`); ancillary
  parameters use Stata's slash names (`/lnsigma`, `/cut1`, `/athrho`).
- Standard metric names: `r_squared`, `adjusted_r_squared`,
  `r_squared_within`, `r_squared_between`, `r_squared_overall`,
  `pseudo_r_squared`, `log_likelihood`, `aic`, `bic`, `rmse`, `df_model`,
  `df_resid`, `sigma_u`, `sigma_e`, `rho`, `n_groups`. Order the dict as it
  should be read; `summary()` prints it in that order.
- `tests` holds specification tests as `{name: {"statistic", "df", "df2"?,
  "p_value", "distribution", "label"}}` (use `wald_test`, `lr_test`).
- `extra` holds bounded model-specific output (for example variance
  components or first-stage summaries). Do not store per-observation arrays.
- `fitted` yields the bounded chart sample; pass `observed` when the plotted
  response is not the raw outcome column.

## Forecasts and shared public names

Public names are unique across families: `registry.public_exports()` refuses a
second `EXPORTS` entry with the same name. Time-series families do not export
their own `forecast`; they register it in the manifest,

```python
FORECAST = {"arima": "openecon.econometrics.arima.forecast:forecast"}
```

and `oe.forecast(result, steps, ...)` dispatches on `result.spec.estimator`.

Specification mistakes (an unknown option value, an undeclared column role, an
unsupported weight type) are rejected when the `ModelSpec` is built. Building a
`ModelSpec` directly raises pydantic's `ValidationError`; convenience functions
build theirs with `core.make_spec`, which reports the same message as
`AnalysisError("invalid_spec", ...)`. Data problems and numerical failures
always raise `AnalysisError`.

Helpers shared by the likelihood families (weight handling with objective
scaling, non-raising Newton wrapper, centring of regressors with exact
back-mapping, probit starting fits, Mills ratio, bivariate normal kernels)
currently live in `discrete/common.py`, `discrete/kernels.py` and
`discrete/bivariate.py`; import them from there rather than copying.

## Test procedures

Procedures that are not model fits (t tests, ANOVA tables, rank tests,
unit-root tests, cross-tabulations, factor analysis) register no estimator.
They are plain keyword functions published through the manifest's `EXPORTS`
dict and return tables:

- `core.table(data, columns=..., **attrs)` builds one result table (an
  `openecon.frame.DataFrame`, rendered as a table by the console and exportable
  to LaTeX); scalar results go in `.attrs`.
- `core.TableSet({"anova": ..., "posthoc": ...}, title=..., **attrs)` groups
  the tables of a multi-table procedure.

Column names are lowercase snake_case (`statistic`, `df`, `p_value`, `mean`,
`std_error`, `ci_low`, `ci_high`, `n`). Never store per-observation arrays in
`attrs`; provide a separate function that returns them as a table.

## Numerical rules

- float64 tensors throughout; integer codes are int64. No autograd: analytic
  scores and Hessians, checked in tests with `optimize.check_derivatives`.
- No Python loop over observations and no n-by-n matrix. Group work uses
  `index_add_` (`covariance.group_sums`, `absorb.group_means`).
- Solve least squares by QR (`linalg.least_squares`), never by inverting `X'X`.
- Likelihood covariance uses `core.ml_covariance` so that `nonrobust`, `opg`,
  `robust` and `cluster` mean the same thing in every family.

## Tests

`tests/test_econ_<family>.py` compares every estimator, covariance option and
weight type with an independent oracle: explicit linear algebra in NumPy, a
brute-force likelihood maximization, or a statsmodels model where one exists.
It also covers the failure contract (error codes), collinearity handling,
missing-data policy, JSON round trip (`result.model_dump_json()`), and the
LaTeX/summary rendering of one model. Tests are deterministic and fast.
