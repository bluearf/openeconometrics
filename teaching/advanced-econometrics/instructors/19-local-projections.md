# Instructor companion · Lab 19

## Teaching sequence

Before running, have students substitute the AR(1) equation into itself to obtain the horizon-$h$ equation, read off the targets $\theta_h=0.8(0.5)^h$, and write out the projection error to see that it is a moving average of order $h$. Then derive the row counts $n_h=T-1-h$ from the sample rule.

After running, read the horizon-2 table, then the four-horizon path: estimates, growing standard errors, row counts and mechanism values side by side. Close by separating pointwise intervals from a joint band, and the exogenous synthetic shock from an observed policy variable.

## Worked exercise answers

**1.** Estimated lagged-outcome coefficients (HAC standard errors): 0.502 (0.036), 0.227 (0.064), 0.130 (0.066), 0.063 (0.075) at horizons 0 to 3. The substitution implies $\gamma_h=0.5^{h+1}$: 0.5, 0.25, 0.125 and 0.0625, all well inside one standard error of the estimates. The coefficient shrinks because the influence of the pre-shock state decays geometrically with the AR coefficient 0.5.

**2.** From the horizon-0 regression, $\hat\theta_0=0.8907$ and $\hat\gamma_0=0.5024$, so the iterated responses are 0.448, 0.225 and 0.113 at horizons 1 to 3, against projection estimates 0.480, 0.239 and 0.248 and mechanism values 0.4, 0.2 and 0.1. The iterated path is smoother and here closer to the mechanism at horizon 3, because it imposes the correct AR(1) dynamics and uses all the information in one regression. The projection does not rely on that form: if the true dynamics were richer, the iterated path would compound the misspecification across horizons, while each projection would still target its own horizon. Projections trade efficiency for robustness.

**3.** Horizon-2 shock standard errors: nonrobust 0.102; HAC with 0 lags 0.105 (identical to HC1, since only $\hat\Gamma_0$ enters with the same $n/(n-3)$ factor); 4 lags 0.110; 8 lags 0.114; 12 lags 0.121. Allowing for the overlap of the MA(2) projection error raises the standard error. Beyond the true overlap, extra Bartlett lags add mostly noise and continue to drift upward here; the change from 4 to 12 lags is modest relative to the estimate, but it should be reported.

**4.** Without the lagged outcome, the horizon-2 response is 0.260 with HAC standard error 0.113 (95% interval 0.039 to 0.482), against 0.239 and 0.110 with the control. Because $x_t$ is independent of $y_{t-1}$ by construction, omitting the control does not bias the shock coefficient; the lagged outcome becomes part of the error, raising residual variance and slightly reducing precision. In observational data, where a shock may respond to the past state, the control would matter for identification too.

**5.** With critical value $t_{0.9875}\approx2.516$ (253 to 256 degrees of freedom), the per-horizon intervals are 0.739 to 1.043, 0.227 to 0.732, −0.037 to 0.516 and −0.020 to 0.516. The band contains the mechanism path (0.8, 0.4, 0.2, 0.1). Horizons 0 and 1 still exclude zero; horizons 2 and 3 no longer do. Bonferroni is conservative: it ignores the correlation among the four estimates, which a band built from their joint covariance would use.

**6.** For example: "Projection estimates fall from 0.891 at impact to 0.480 and 0.239 at horizons 1 and 2, then stay near 0.25 at horizon 3; the rise from horizon 2 to 3 is 0.009, far smaller than either HAC standard error. Each pointwise 95% interval excludes zero, but these are not joint statements, and with a Bonferroni adjustment horizons 2 and 3 are not distinguishable from zero." The synthetic shock is exogenous by construction; "proves" overstates what pointwise intervals can show.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/19-local-projections/local_projections.xlsx) for the published analysis. The [original generator](generators/19-local-projections.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/19-local-projections/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/19-local-projections/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: one tail row lost each horizon, same hac convention, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** An observed exposure is not automatically an exogenous shock. Pointwise significance across horizons does not provide simultaneous coverage, and horizon/sample conventions must be reported.
