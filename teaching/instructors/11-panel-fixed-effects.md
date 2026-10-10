# Instructor notes · Lab 11

Use the [student handout](../labs/11-panel-fixed-effects/index.md) to separate comparison, common trend and uncertainty unit. The native scatterplot removes firm means but does not residualize the trend; students should not treat its visual slope as the complete adjusted coefficient.

## Worked answers

1. The firm intercept disappears; investment deviations, trend deviations and idiosyncratic error deviations remain. Demeaning does not remove time-varying confounding.
2. Pooled 4.508013 (SE .148080, CI [4.211705,4.804321]) includes correlated persistent advantages. FE 2.064449 (SE .094610, CI [1.875135,2.253762]) uses within-firm variation at fixed trend. Units are index points per thousand investment units per worker.
3. Firm effects remove additive persistent levels. Trend controls a specified common linear mean change. Clustering permits dependent disturbances within firms; it changes covariance, not the confounded target of pooled OLS.
4. Panel time identifies order and duplicate dates. It does not create year regressors. A one-year common demand collapse would motivate a more flexible calendar specification.
5. Any firm-constant column has zero within variation and is omitted. Its separate slope is unidentified once unrestricted firm effects are present.
6. If a positive productivity shock induces future investment, conditioning on the complete investment history can reveal the current shock. Strict exogeneity fails. A covariance correction cannot restore that moment condition.
7. Include N=360, G=60, t(59), estimate and interval, common linear trend and known simulated design. Do not equate within R²=.824644 with overall pooled R².

## Reproduction and verification

The [complete saved reference](../labs/11-panel-fixed-effects/reference.json) retains both models. `run_lab(output_dir="lab11-review")` explicitly exports it and the table. Independent within demeaning, cluster-score accumulation and the stated nested-effect finite-sample factor reproduce the FE slopes and covariance, including the reported intercept transformation. Independent pooled cluster OLS is also checked. The source checks zero within means, identical 360 positions, 59 cluster df and full JSON roundtrip. Initial coefficient and covariance differences were below $2\times10^{-14}$.

Native displays are `OLSResult`, `ResultBundle`, `PlotSpec`, `Latex`. This is a balanced one-way firm FE model with an explicit numerical trend; it is not a demonstration of unrestricted two-way fixed effects or a general dynamic-panel estimator.

**Prepared input:** [firm_panel.xlsx](../labs/11-panel-fixed-effects/firm_panel.xlsx). Use the stored observations for the published analysis. The [original generator](generators/11-panel-fixed-effects.py) supports new dataset editions; update results and answers when changing observations.
