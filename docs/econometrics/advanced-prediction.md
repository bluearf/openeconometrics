# Explicit saved static targets

These adapters evaluate persisted coefficients and full joint covariance without
refitting. A `Dataset` produces completed owned indexed Parquet output; release
the output object to remove its scratch storage. AME reduces values and complete
parameter gradients globally. MEM first reduces original numeric covariates and
category indicators globally, then evaluates the saved nonlinear function once.
It does not average formula values or regime indicators across batches.

```python
predictions = oe.predict(saved_system, oe.scan("new.parquet"), outcome="wage", interval="mean")
effects = oe.margins(saved_nl, "age", data=oe.scan("new.parquet"), method="mem")
joint = oe.predict(saved_biprobit, data=new_rows, outcome="joint11")
selected = oe.predict(saved_heckman, data=new_rows, outcome="conditional")
efficiency = oe.predict(saved_frontier, data=new_rows, outcome="te")
```

| Family | Explicit `outcome` targets | Meaning and boundary |
| --- | --- | --- |
| `nl` | `function` (default) | Persisted safe formula tape; original parameter names and raw covariate means. Formula text is parsed, never executed as Python. |
| `sureg`, `mvreg`, `reg3` | Saved equation name (required for multiple equations) | That equation's structural response, fitted categories, omitted features and fixed constraints. Complete saved covariance is retained. |
| `threshold` | `regime_mean` (default) | Fitted thresholds and common/regime coefficient blocks. Ties use `q <= threshold`. Slope uncertainty conditions on the fitted thresholds. The threshold-column derivative is undefined at a boundary; no estimated threshold derivative is invented. |
| `biprobit` | `joint00`, `joint01`, `joint10`, `joint11` (default), `marginal1`, `marginal2`, `conditional1`, `conditional2` | Joint cells, marginal probabilities, or `Pr(Y1=1|Y2=1)` / `Pr(Y2=1|Y1=1)`. Both equations and correlation derivatives enter the delta covariance. Recursive binary intervention paths remain explicitly unsupported. |
| `heckprobit` | `joint11` (default), `marginal1`, `selection`, `conditional1` | Joint outcome/selection, unconditional population success, selection probability, success conditional on selection. |
| `heckman` | `mean` (default), `conditional`, `selection` | Unconditional latent outcome mean, selected outcome mean with the Mills loading, or selection probability. Both ML and two-step complete joint covariances are supported. |
| `churdle` | `mean` (default), `conditional`, `participation` | Expected recorded amount, positive/interior mean, or crossing probability. Linear/exponential models, fitted hurdle and ancillary scale are preserved. |
| `ivprobit` ML | `structural` (default) | Structural probit probability at supplied endogenous covariates; does not condition on first-stage residuals. |
| `ivtobit` ML | `structural` (default) | Expected recorded censored outcome with the resolved fitted limits and outcome scale. `kind='conditional'` gives the latent mean within the limits. |
| IV two-step | `structural`, `kind='xb'|'stdp'|'latent'` only | Saved normalized Newey outcome index. Historical two-step results lack the scale/nuisance state needed to identify a structural probability or censored response. Those targets are refused. |
| `frontier` | `frontier` (default), `mean`, `u`, `te` | Deterministic frontier, expected recorded output, posterior inefficiency, or posterior efficiency. `u`/`te` require the observed outcome. Half-normal, truncated-normal, exponential and production/cost conventions are separate. |

`xb`/`stdp` describe the selected structural index, including the selection index
for probability targets named `selection`, `participation`, or `marginal2`.
`kind='conditional'` is available only for targets with an explicit conditional
definition. Frontier posterior targets do not substitute for unconditional means.
Conditional bivariate tails beyond 35 standard deviations give a precision error;
the adapter does not silently return an unchecked probability ratio. Normal
censoring/truncation reuse the native stable interval/tail moments. Mills ratios
use a continued fraction in the far negative tail.

Malformed equation metadata, inconsistent estimator identity, missing parameters,
unknown categories, unsupported targets and nonfinite domains produce structured
errors. No numerical runtime dependency on SciPy or statsmodels is introduced.
Independent development tests use NumPy/SciPy closed formulas and quadrature,
finite-difference parameter Jacobians, full covariance, raw-mean MEM, JSON restore,
coefficient permutation, constraints, category contrasts, typed indices, missing
rows, disk cleanup, boundaries and precision errors.

Primary target references: [Stata bivariate probit postestimation](https://www.stata.com/manuals/rbiprobitpostestimation.pdf),
[Heckman postestimation](https://www.stata.com/manuals13/rheckmanpostestimation.pdf),
and [Stata hurdle models](https://www.stata.com/features/overview/hurdle-models/).
