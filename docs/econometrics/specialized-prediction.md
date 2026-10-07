# Saved survival, history and causal targets

These APIs consume saved parameters without refitting. They keep their explicit
target contracts separate from scalar `predict/margins` dispatch. Dataset
outputs are complete, owned indexed temporary Parquet; release them when no
longer needed. Computation uses native float64 PyTorch kernels. SciPy and
statsmodels are development oracles only.

## Survival

```python
baseline = oe.cox_baseline(saved_cox, original_dataset, batch_rows=8192)
curve = oe.stcurve(saved_cox, data=original_dataset, at={"age": 60})
predicted = oe.survival_predict(
    saved_cox, evaluation_dataset, target="survival", time="evaluation_time",
    baseline=baseline, interval="mean",
)
```

`cox_baseline` requires the original Dataset fit's projected source and retained
sample identity. It replays immutable native risk records and produces every
failure time, including baseline gradients in saved coefficient order. Breslow,
Efron, delayed entry, strata, offsets and the existing exactp/Peto-Breslow curve
convention are preserved. User weights retain the fitted Breslow-only contract.
The raw source is reverified after the replay; changing sources are rejected.
`stcurve(data=Dataset)` returns the full step function. `stcurve(data=None)`
continues to use the saved display preview, which may be thinned to 400 times.

| Result | Explicit `target` | Required inputs and boundaries |
| --- | --- | --- |
| Cox | `relative_hazard` | Covariates and offset; no absolute baseline needed. |
| Cox | `survival`, `cumulative_hazard`, `hazard_jump` | Full matching baseline, time scalar/column and fitted stratum. `hazard_jump` is the event jump, not a continuous instantaneous hazard. Times from zero through the last observed failure are supported; later extrapolation and unknown strata are rejected. |
| Cox | `quantile` | Full matching baseline and `quantile` strictly between 0 and 1. Inverts the observed step CDF. Beyond observed support is undefined; a regular parameter delta interval for this discrete inverse is rejected. |
| STREG | `survival`, `hazard`, `cumulative_hazard` | Explicit strictly positive finite time scalar/column. All six saved distributions and their supported PH/AFT metrics use their fitted ancillary equations. |
| STREG | `relative_hazard` | PH metric only. |
| STREG | `quantile` | Probability strictly between 0 and 1; native generalized-gamma inversion is checked for resolution. |
| STREG | `mean` | Defined exponential, Weibull, lognormal, loglogistic and generalized-gamma means only. Loglogistic requires gamma < 1; generalized gamma requires finite moment support. Gompertz mean and unresolved near-zero generalized-gamma shape domains are explicitly unsupported. |

Times are from zero at constant supplied covariates. Entry-conditional survival
and time-varying subject paths are different targets. Cox TVC paths are rejected
until an explicit event-time path API is available. Parametric delta intervals
use the complete saved covariance, including ancillary cross terms. Cox delta
intervals include the baseline's coefficient derivative but **exclude baseline
counting-process variance**; metadata records this limitation. Legacy resident
Cox fits lacking replay identity can use the existing resident `stcurve` path.

## Dynamic history

```python
forecast = oe.dynamic_predict(
    saved_arima, history_dataset, target="forecast_levels",
    origin=120, horizon=12, future=future_dataset,
)
```

`history` and `origin` are mandatory. `origin="end"` consumes the supplied
history. An exact integer origin uses only declared integer periods at or before
that origin; it must be present. Numeric periods must be consecutive and unique.
Declared date histories currently use `origin="end"`. SQLite stores and sorts
history on disk and the source digest is checked again before returning output.
Saved categorical coding is reused. The history must reproduce the fitted
design; insufficient periods or predictor variation produce a structured state
error. Unknown levels are rejected. Missing interior periods
cannot silently reset a filter.

| Result | `target` | State and uncertainty |
| --- | --- | --- |
| ARIMA | `forecast_levels` | Replayed fixed-parameter ARMA filter and differences; integrated levels restored. Native innovation/filter forecast error. |
| ARCH | `forecast_levels`, `conditional_variance` | Replayed residual/variance state and presample convention. Native forecast-error intervals for levels; expected conditional variance recursion has no variance-state interval. |
| VAR, VEC | `forecast_levels` | Bounded lag tail, saved cointegration/VAR representation and deterministic terms. Existing VAR parameter contribution is retained when available; VEC retains its native fixed-parameter innovation convention. |
| UCM | `forecast_levels`, `components_smoothed` | Native filter/smoother and latent-state covariance at fixed saved parameters. Smoothed components depend on the complete supplied sequence. |
| Markov switching | `probabilities_filtered`, `probabilities_predicted`, `probabilities_smoothed` | Native Hamilton/Kim probabilities at fixed parameters. Smoothed probabilities depend on later observations in supplied history. No parameter probability bands are claimed. |
| ARDL, NARDL | `fitted_levels` | Saved selected lags, complete level coefficients/covariance, deterministic terms and bounded lag carry. Full parameter delta intervals conditional on supplied history. NARDL requires the original complete source and its fitted partial-sum zero origin. |
| AH, XTDPD | `fitted_transformed` | First differences or fitted forward orthogonal deviations, actual physical-row/period alignment and full saved parameter covariance. Complete panel history through `origin="end"`; no recovered unit effects or level forecast. |

Forecast targets require a horizon of 1–5000 and the family's required future
exogenous path. A Dataset future path is collected only within that explicit
bounded horizon and must contain exactly one row per step. `uncertainty="native"`
preserves the existing family convention; parameter-only or combined intervals
requiring an unvalidated filter/tangent contract are rejected explicitly.
Forecast error and fitted parameter uncertainty are not relabeled as each other.
Static marginal effects and unsupported regime-conditioned forecasts remain
unsupported. Old EC results missing complete level covariance are rejected.

## Causal, RD and GMM

```python
effect = oe.causal_evaluate(saved_teffects, target="population_effect", treatment="1")
standardized = oe.causal_evaluate(
    saved_teffects, evaluation_dataset, population="fixed_evaluation",
    target="standardized_outcome", treatment="1",
)
gmm_mean = oe.causal_evaluate(
    saved_gmm, evaluation_dataset, population="fixed_evaluation",
    target="outcome_mean", outcome_function="{b0}+{b1}*income",
)
```

| Result | `population="estimation"` targets | Explicit selectors |
| --- | --- | --- |
| Treatment effects | `population_effect`, `population_potential_outcome` | Saved treatment label. Uses the saved ATE/ATET/POM estimand and complete influence covariance. |
| DiD | `population_effect` | Saved TWFE ATET; unsupported cohort selectors are rejected. |
| Event study | `event_effect` | Exact identified relative `event`, including explicit zero reference and saved binning. No extrapolation outside identified event support. |
| CSDID | `simple`, `dynamic`, `group`, `calendar` | Identified saved aggregations with joint ATT/cohort-share influence uncertainty. `group` requires a saved `cohort`; optional `event` filters dynamic support. |
| RD | `cutoff_effect`, `cutoff_side` | `point` must equal the exact saved cutoff. Side limit requires `side="left"` or `"right"`; main/bias bandwidths are recorded. `correction="conventional"`, `"bias_corrected"` or `"robust"` selects one alternative estimate and its proper covariance. |

These original-population targets accept no new evaluation rows. A saved ATT,
cutoff jump or moment coefficient is never treated as a unit response mean.

`population="fixed_evaluation"` requires an explicit Dataset and allows
`outcome_mean` per row or global weighted `standardized_outcome`. Treatment
effects RA/IPWRA/AIPW fits now persist a bounded complete joint nuisance/POM
parameter vector and sandwich covariance in reporting coordinates. Linear,
logit, probit and Poisson outcome regressions use their explicit treatment
equation and fitted categorical design. Saved propensity overlap is checked on
evaluation covariates. IPW/matching results without an outcome regression, and
legacy results without complete nuisance state, are rejected without refitting.

RD derivative/kink fits need their own explicit derivative target and are
rejected here. Covariate-adjusted cutoff effects retain native inference;
adjusted one-sided outcome means require additional joint adjustment-nuisance
state and are explicitly rejected.

General GMM requires `outcome_function` in the safe named-parameter formula
grammar, explicit numeric inputs and identified saved parameters. Moment
residuals are not inferred to be outcomes. No arbitrary code evaluation occurs.

Global standardization reduces both values and parameter gradients across all
eligible positive-weight rows before applying the full covariance once. Its
interval conditions on supplied evaluation rows: it does **not** include their
sampling uncertainty or establish a transported ATE/ATT. All outputs record
target, population, source/model identity and uncertainty definition. Selectors
outside their family/support are rejected.

## Verification scope

`test_econ_saved_survival.py` checks independent complete risk masks, six
distribution density/quantile calculations and full ancillary covariance.
`test_econ_saved_dynamic.py` checks independent recursive forecasts, lag/partial
sum designs, transformations and native filter/component/probability parity.
`test_econ_causal_evaluation.py` checks independent stacked M-estimation state,
delta gradients, original identified contrasts, RD sides and safe GMM formulas.
JSON restore, source mutation, missing alignment, invalid domains and owned
scratch/output cleanup are separate checks. These are method-level contracts,
not blanket Stata parity or public-release/platform validation.

Method references: [STREG postestimation](https://www.stata.com/manuals/ststregpostestimation.pdf),
[Cox postestimation](https://www.stata.com/manuals/ststcoxpostestimation.pdf),
[ARIMA postestimation](https://www.stata.com/manuals/tsarimapostestimation.pdf),
[VAR postestimation](https://www.stata.com/manuals/tsvarpostestimation.pdf), and
[GMM postestimation](https://www.stata.com/manuals/rgmmpostestimation.pdf).
