# Saved prediction domains and the remaining adapters

The common saved-result interface is selected by a model's actual response
contract. A coefficient vector is not sufficient evidence that a conditional
mean, treatment effect, survival curve and dynamic forecast are interchangeable.
No adapter refits the model, collects a Dataset, silently removes a time
transformation, or invents unrecorded group effects.

## Linear and population-mean extensions

These contracts use fitted treatment levels and the full saved parameter
covariance, including zero Jacobian columns for ancillary parameters. Numeric
effects and categorical contrasts use the selected response contract. Dataset
AME aggregates parameter gradients over every eligible row before applying
the covariance once. Dataset MEM evaluates at the global weighted encoded
design; it is not an average of batch MEMs.

| Models/options | Supported target | Explicit boundary |
| --- | --- | --- |
| `rreg` | Robust fitted linear location `X beta` | This describes the robust location functional, not an independently identified conditional population mean. |
| `qreg`, `bsqreg` | Fitted conditional quantile `Q(tau|X)` | The response is a quantile. It is not relabeled as a conditional mean. |
| `iqreg` | `Q(high|X)-Q(low|X)` with its saved covariance | This is the fitted interquantile difference. It does not predict either separate quantile from a difference-only covariance. |
| `sqreg` | One explicitly selected saved quantile equation, e.g. `outcome='q25'` | Selection is mandatory when several quantiles exist; the full original covariance and parameter order remain intact. |
| `xtreg`, pooled | Pooled linear response mean | The panel/time fields describe fit structure and covariance, not an omitted prediction transformation. |
| `xtreg`, RE/ML; `xtivreg`, RE | Population linear mean `X beta`, integrating mean-zero Gaussian panel effects | Does not add empirical Bayes panel predictions. |
| `mixed` | Population linear mean `X beta`, integrating mean-zero Gaussian random effects | Existing `mixed_predict` provides different group-conditioned BLUP targets. Variance coefficients remain in the full covariance with zero population-mean Jacobian columns. |
| `xtgls`, `xtpcse` | Structural linear response `X beta` | Does not condition on future realized disturbances or add residual forecasts. |
| `prais` | Structural response `X_t beta` | AR(1) disturbance forecasts and lagged residual conditioning are separate targets. |
| `xtfmb` | Linear predictor using the mean fitted period coefficient vector | This does not claim a particular period's unpersisted coefficient vector. |
| `xtgee`; `xtlogit`, `xtprobit`, `xtpoisson` with `model='pa'` | Population-averaged response through the saved family/link and offset/exposure | FE/RE likelihood fits cannot reuse this GEE mean contract. |
| `areg`, `reghdfe`, `ivreghdfe`, `ppmlhdfe`; `xtreg`/`xtivreg` FE | Explicit `xb`, its `stdp`, and `margins(kind='xb')` | New fits persist fitted joint group sums. Conditional means use known groups; historical absent state and unknown groups are rejected. Full nuisance inference is supported for bounded nonrobust geometry, otherwise point means only. See [saved group prediction](group-prediction.md). |
| `xtreg`/`xtivreg`, between | Explicit `xb` evaluated at supplied panel-mean covariates; `stdp` and `xb` effects | Does not silently aggregate new individual covariates to panel means or relabel between predictions as individual response means. |
| `xtreg`/`xtivreg`, first differences | No generic raw-level prediction | Consecutive within-panel differences and their row alignment are essential. A difference coefficient cannot be applied silently to level inputs. |

All these branches reject missing mandatory model roles, conflicting fitted
estimator identity, unsupported formula terms, unfitted categorical levels,
invalid family/link metadata and nonfinite prediction domains. A predictor
also serving as offset/exposure requires a separate intervention contract.

## Static formula, system and ancillary targets

`nl`, `sureg`, `mvreg`, `reg3`, `threshold`, `biprobit`, `heckman`,
`heckprobit`, `churdle`, `ivprobit`, `ivtobit` and `frontier` now use
[explicit saved target adapters](advanced-prediction.md). The target names,
raw-mean MEM contract, missing-state boundaries and tail limits are documented
there.

## Estimator names without a common saved adapter

This inventory concerns `predict/margins` dispatch. It does not change native
Dataset fitting coverage or imply that a family has no existing postestimation.

| Estimator names | State and target required before adding generic dispatch |
| --- | --- |
| `clogit` | Exact complete-group conditional probabilities are implemented with saved success counts and row-multiset contracts. Conditional probabilities need the complete alternative/group set and an explicitly known conditioned success count. Independent per-row logit probabilities are not the fitted conditional likelihood. Dataset groups may cross batch boundaries. |
| `melogit`, `meprobit`, `mepoisson`, `menbreg` | Integrated population means and explicit-effect conditional means are implemented with full variance/covariance gradients. Distinguish conditional means at explicit random effects from population means integrated over their fitted distribution. Random slopes alter the row-specific integration variance; delta gradients must include all variance/covariance parameters. Group posterior predictions are another target. |
| `stcox` | Specify log relative hazard, hazard/survival at an evaluation time and stratum, and handling of time-varying covariate paths. Absolute curves need a full baseline or verified source replay, not a thinned display table. |
| `streg` | Specify survival, hazard, time quantiles or expected failure time under the saved distribution and PH/AFT metric. Validate each parameterization, ancillary equation, entry/stratum role and time domain; some distributional means do not exist. |
| `arima`, `arch`, `var`, `vec`, `ucm` | Dynamic targets require lag/filter state, forecast origin, future exogenous paths and horizon. Preserve forecast-error versus parameter-uncertainty conventions; a row-wise mean adapter cannot erase recursion or cointegration state. Existing `forecast` dispatch is a distinct API. |
| `ardl`, `nardl` | Reconstruct lag/difference designs, lag history, deterministic terms and forecast origin. NARDL additionally needs accumulated positive/negative paths. Long-run/dynamic multipliers are not the same derivative as a static row-wise response. |
| `mswitch` | Specify regime-conditioned versus filtered/smoothed mixture predictions. Persist transition/filter state, probabilities and origin; posterior probabilities may depend on a future observation sequence. |
| `ahreg`, `xtdpd` | Reconstruct panel lags and the fitted difference/system transformation. Separate transformed-equation predictions from level forecasts; fixed-effect recovery and dynamic accumulation require explicit state. |
| `gmm` | General moment parameters need an explicit outcome prediction function. A moment equation or parameter named after a regressor does not establish a response mean. |
| `didregress`, `eventstudy`, `csdid`, `teffects` | Preserve treatment-effect estimands, treatment/cohort/time support and identified evaluation populations. A reported ATE/ATT/event-time coefficient is not a unit-level conditional mean. Potential-outcome or policy targets need their own saved nuisance state and influence function. |
| `rdrobust` | Specify cutoff side, local polynomial/bandwidth support, estimand and bias-corrected target. A cutoff treatment effect cannot be reused as an arbitrary covariate response mean. |

## Existing separate APIs

`oe.survival_predict`, `oe.cox_baseline`, `oe.dynamic_predict` and
`oe.causal_evaluate` provide explicit Dataset time/stratum, history/origin and
identified-population targets. Their [target matrix and scientific limits](specialized-prediction.md)
describe the saved state, uncertainty and unsupported paths. They do not change
common scalar `predict/margins` dispatch counts.

`oe.forecast(result, steps, ...)` already dispatches to ARIMA, ARCH, VAR, VEC
and UCM forecast functions. Their saved final state can support a horizon
forecast without refitting. `dynamic_predict` additionally replays an explicit
Dataset history through its selected origin; generic marginal effects and all
conditional forecast options are not implied.

`oe.mixed_predict(..., kind='xb'|'fitted'|'reffects')` distinguishes the fixed
linear predictor, fitted values with BLUPs, and group random effects. Its
group-conditioned targets need observed outcomes/group design and are separate
from the new generic population mean. The resident `ModelFrame` path remains available; Dataset BLUPs now use bounded
complete top groups, fitted typed contracts and owned indexed output. See
[saved group prediction](group-prediction.md) for targets and inference limits.

`oe.stcurve` already returns Cox survivor/cumulative hazard curves. Without
estimation data it uses a bounded stored baseline (at most 400 failure times),
and reports whether that baseline was thinned. A full baseline is recomputed
when estimation data are supplied. A Dataset uses immutable disk risk replay
and owned full output; a resident frame retains its resident preparation path.
`survival_predict` supplies separate time/stratum targets with explicit domains.

`oe.ucm_components`, `oe.mswitch_probabilities`, and `oe.nardl_multipliers`
also have specific result targets. They must retain their state/row-order and
uncertainty contracts when connected to a shared interface.

Normal panel RE uses an integrated likelihood adapter; PA keeps its saved GEE
link. These are separate contracts. Complete-group targets and every stored
effect/variance inference domain are listed in [group prediction](group-prediction.md).
