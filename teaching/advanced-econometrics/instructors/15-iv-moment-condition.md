# Instructor companion · Lab 15

## Teaching sequence

Before running, write the structural equation and the three moment conditions, and have students say which variables instrument themselves. Derive $(Z^\top X)^{-1}Z^\top y$, then the projection form, and show that $P_ZX$ leaves the intercept and control unchanged.

After the run, read the first stage before the 2SLS coefficients, confirm the projection formula reproduces the native estimate, and spend time on the residual that enters $\hat\sigma^2$. Close on what the data cannot show: exclusion.

## Worked exercise answers

**1.** The reduced-form `z` coefficient is 0.9142 and the first-stage `z` coefficient is 0.8471; their ratio, 1.0792, equals the 2SLS slope (indirect least squares). With more excluded instruments than endogenous regressors there would be one ratio per instrument, generally different, and 2SLS would combine them; the single ratio is the IV estimator only in the exactly identified case.

**2.** The hand-run second stage gives the same slope, 1.0792, but a standard error of 0.1334 against 0.0807. Both use the same matrix $(\hat X^\top\hat X)^{-1}$; they differ only in $\hat\sigma$. The structural residuals give $\hat\sigma=1.219$ (residual sum of squares 470.80 over 317 degrees of freedom), while the fitted-$x$ residuals give 2.013 (1,284.89 over 317), so the standard-error ratio is $1.219/2.013=0.605$. Here the naive standard error is too large; in other data it can be too small.

**3.** The first-stage $t$ statistic for `z` is 15.856, and $15.856^2=251.41$, the reported first-stage and Cragg–Donald F. With one excluded instrument and classical covariance these coincide. The number shows that Z strongly predicts X given the control; it says nothing about whether Z is correlated with the structural error.

**4.** If $y_i=\beta_0+\beta_cc_i+\beta_xx_i+\gamma z_i+u_i$, the reduced-form coefficient on `z` becomes $\beta_x\pi+\gamma$, where $\pi$ is the first-stage coefficient, so the IV slope converges to $\beta_x+\gamma/\pi$. With $\pi=0.8471$, $\gamma=0.1$ shifts the IV slope by $0.1/0.8471=0.118$. A stronger first stage (larger $\pi$) shrinks the shift but never removes it.

**5.** With `covariance="robust"` the X standard error is 0.0830 against 0.0807 classical. The difference is small here, consistent with roughly homoskedastic structural errors. The robust version should be reported whenever the error variance may depend on the regressors or instruments, which is the usual situation in applied work.

**6.** For example: "In 320 synthetic records, 2SLS with Z as the excluded instrument for X and the control instrumenting itself gives an X slope of 1.079 (classical standard error 0.081, 317 degrees of freedom), against 1.510 by OLS. The first stage is strong (F = 251). The Durbin and Wu–Hausman tests reject exogeneity of X, but only if Z is a valid instrument; the causal reading rests on the untestable assumption that Z affects Y only through X given the control."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/15-iv-moment-condition/iv_moment_condition.xlsx) for the published analysis. The [original generator](generators/15-iv-moment-condition.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/15-iv-moment-condition/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/15-iv-moment-condition/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: projection formula matches iv, same complete sample, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Exclusion is an assumption about the mechanism, not proved by first-stage strength. Exactly identified IV supplies no overidentification test, and weak instruments can make usual inference unreliable.
