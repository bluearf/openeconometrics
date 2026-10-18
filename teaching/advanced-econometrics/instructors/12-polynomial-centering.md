# Instructor companion · Lab 12

## Teaching sequence

Start from the expansion of $(x-c)^2$ and have students match powers before running anything; ask which coefficients must stay fixed. Then show that the centered design is $XT$ for an invertible $T$, so the projection, fitted values and residuals cannot change, and point out which columns the argument needs.

After the run, compare the two coefficient tables row by row and connect the change in the linear standard error to $A\widehat V A^\top$. Stress that `x` is the raw regressor, with mean 8.05; centering is what the lab does to it, not a property of the workbook column.

## Worked exercise answers

**1.** The slope $b_1+2b_2x$ is $0.28155+2(-0.0256374)(7)=-0.0774$ at X = 7 and $-0.1799$ at X = 9. Refitting with $x-9$ and $(x-9)^2$ gives a linear coefficient of −0.1799 (HC3 standard error 0.1204; intercept 6.2997). The standard error at 9 exceeds the 0.0735 at the mean because 9 lies farther from the center of the observed X values, where the fitted curve is less tightly pinned down.

**2.** Matching constant terms gives $a_0=b_0+b_1c+b_2c^2$ (with $a_z=b_z$ the control term is unaffected). Numerically, $5.84235+0.281549\times8.054672-0.0256374\times64.8777=6.4468$, equal to the centered intercept 6.44683 up to rounding of the inputs.

**3.** Since $a_1=b_1+2cb_2$, $\operatorname{Var}(\hat a_1)=\operatorname{Var}(\hat b_1)+4c\operatorname{Cov}(\hat b_1,\hat b_2)+4c^2\operatorname{Var}(\hat b_2)$. With the raw HC3 entries 0.485786, −0.0304031 and 0.00192349, this is $0.485786-0.979547+0.499168=0.0054067$, and its square root 0.07353 equals the centered standard error. The full matrix identity $A\widehat V_{\text{raw}}A^\top=\widehat V_{\text{centered}}$ holds to about $3\times10^{-16}$.

**4.** In both parameterizations the HC3 Wald statistic is 3.2666 (F = 1.633 on 2 and 236 degrees of freedom, $p=0.197$). The null $b_1=b_2=0$ is the same set of models as $a_1=a_2=0$ because $(a_1,a_2)$ is an invertible linear function of $(b_1,b_2)$, and the HC3 covariance transforms with the same matrix, so the quadratic form is unchanged. The X terms are not jointly significant at 5% in this sample.

**5.** Without an intercept the raw fit has residual sum of squares 211.36 and the centered fit 6,588.53 (against 207.72 for both fits with an intercept); the largest fitted-value gap is 8.05. The expansion $(x-c)^2=x^2-2cx+c^2\cdot1$ needs the constant column, so without it the two designs span different spaces and the projection argument fails.

**6.** For example: "In 240 synthetic records, the fitted quadratic gives a slope of −0.131 outcome units per X unit at the sample mean X = 8.05 (HC3 standard error 0.074, $p=0.075$). The raw linear coefficient 0.282 ($t=0.40$) is the slope at X = 0, outside the observed range, and says nothing about typical values. The quadratic coefficient and the joint test of the X terms are imprecise, so the data do not pin down the shape of the relationship."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/12-polynomial-centering/polynomial_centering.xlsx) for the published analysis. The [original generator](generators/12-polynomial-centering.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/12-polynomial-centering/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/12-polynomial-centering/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: identical fitted values, centered slope interpretation, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Dropping the quadratic term after centering changes the model. Centering alone does not justify extrapolation outside the observed X support.
