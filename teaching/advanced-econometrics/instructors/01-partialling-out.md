# Instructor companion · Lab 01

## Teaching sequence

Start from the normal equations and ask what "holding `z` constant" can mean when `z` is continuous. Build $M_W$ with the intercept inside $W$, derive the identity on the board, and only then run the three regressions. Have students compute $\sum\tilde x_i\tilde y_i/\sum\tilde x_i^2$ by hand from the chart data before they look at the summary table.

Before running, ask students whether the full and residualized slopes should differ. After running, separate three claims: the coefficient identity (exact), the standard error (from the full model only) and a causal reading (not supported by the algebra).

## Worked exercise answers

**1.** With $V=[\mathbf 1,\ x]$ and $M_V=I-V(V'V)^{-1}V'$, regress `y` on `x` and `z` on `x` (each with an intercept) and keep the residuals. The residual-on-residual slope is 0.7522, equal to the full-model `z` coefficient 0.752157 to rounding error. FWL applies to any partition of the columns.

**2.** (a) Raw `y` on $\tilde x$ without an intercept gives 1.1882, the full coefficient: $\tilde x'y=x'M_Wy=x'M_WM_Wy=\tilde x'\tilde y$ by symmetry and idempotence, so the numerator is unchanged. (b) $\tilde y$ on raw `x` with an intercept gives a slope of 0.7328 (intercept 0.0667). Its numerator is again $\tilde x'\tilde y$, because $\tilde y$ is orthogonal to the fitted part of `x`, but its denominator is the total variation of `x`, 297.63, rather than 183.56. Hence the slope equals $\hat\beta_x(1-R^2_{x\mid z})=1.1882\times0.6167=0.7328$. Partialling the control out of `x` is essential; partialling it out of `y` alone is not enough.

**3.** The residual regression with HC3 reports a standard error of 0.1107 on 239 degrees of freedom, against 0.1119 on 237 in the full model. Two things differ: the residual regression's leverages $\tilde x_i^2/\sum\tilde x_j^2$ omit the intercept and `z` directions, and its degrees of freedom ignore two estimated parameters. With classical covariance the standard errors are 0.10231 (residual regression) and 0.10274 (full model). The residual sums of squares are identical (459.24), so the ratio is $\sqrt{239/237}=1.00421$ exactly.

**4.** $R^2=0.3833$ for `x` on `z`, so 61.67% of the sample variation of `x` (183.56 of 297.63) remains in $\tilde x$, and $1/(1-R^2)=1.621$. Under homoskedasticity $\operatorname{Var}(\hat\beta_x)=\sigma^2/\sum\tilde x_i^2=\sigma^2/[\mathrm{SST}_x(1-R^2)]$, so this variance inflation factor multiplies the variance that `x` would have with an uncorrelated control. As `z` approaches a copy of `x`, $R^2\to1$, $\sum\tilde x_i^2\to0$ and the variance grows without bound (Lab 04).

**5.** The short slope of `y` on `x` is 1.7059. The regression of `z` on `x` has slope 0.6882, and $1.1882+0.7522\times0.6882=1.1882+0.5176=1.7059$. The short slope adds to the controlled slope the part of `z`'s association with `y` that travels with `x`; Lab 02 develops this decomposition.

**6.** For example: "In 240 synthetic units, the coefficient on `x` in a regression of `y` on `x` and `z` is 1.188 (HC3 SE 0.112, 237 degrees of freedom). The residual-on-residual regression reproduces this coefficient but not its standard error, which is taken from the full model. The estimate adjusts linearly for the observed `z` only and is not, by itself, evidence of a causal effect."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/01-partialling-out/partialling_out.xlsx) for the published analysis. The [original generator](generators/01-partialling-out.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/01-partialling-out/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/01-partialling-out/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: partial slope identity, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Partialling out is an algebraic identity. It does not remove unobserved confounding or justify using residual-regression degrees of freedom for the original model.
