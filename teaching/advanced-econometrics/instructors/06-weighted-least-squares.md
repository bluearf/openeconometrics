# Instructor companion · Lab 06

## Teaching sequence

Start with the `variance` column, not the regression: have students sort the rows by variance and notice that noise grows with the distance of `x` from zero. Derive the weighted normal equations and the $\sqrt{w_i}$ transformation on the board, with explicit attention to the transformed intercept, before anyone runs code.

Before running, ask students to predict which coefficient's classical OLS standard error is most distorted. After running, use the reversed standard-error comparison for `x` (0.1299 OLS classical, 0.1513 WLS) to separate "which estimator is more efficient" from "which standard-error formula is valid", and close with the residual funnel in the chart.

## Worked exercise answers

**1.** Every row satisfies $\text{variance}=(1+|x|)^2$ exactly (largest absolute discrepancy 0 in the stored values). Because $|x|\ge0$, every variance is at least 1 and every weight at most 1; the largest weight, 0.9924, belongs to row 107 with $x=-0.0038$, the row closest to zero. A weight below a quarter of the maximum (0.2481) requires $1+|x|$ to exceed twice its smallest value, roughly $|x|>1.008$; **74** of the 240 rows qualify.

**2.** Correct rescaled regression ($y^*$ on $\sqrt{w}$, $x^*$, $z^*$, no constant): 2.0355, 1.4455, 0.8162, identical to the WLS fit. With an ordinary column of ones added to $x^*$ and $z^*$ instead of $\sqrt{w}$: constant 1.2381, $x^*$ 1.4926, $z^*$ 0.7839. Dividing that regression back by $\sqrt{w_i}$ shows it fits $y_i=\gamma\sigma_i+\beta_1x_i+\beta_2z_i+u_i$ with $\sigma_i=1+|x_i|$ and no intercept, a different conditional mean that absorbs a V-shaped term in $x$ and shifts both slopes.

**3.** With raw weights and WLS residuals, $\sum_i w_i\hat e_i^2=233.92$ and dividing by $n-k=237$ gives **0.987**. If the variances are exactly right, the rescaled errors have unit variance and this statistic has expectation 1, so 0.987 is consistent with the variance column. The software's estimate 2.38 is the same quantity on the normalized-weight scale ($0.987/\bar w$ with $\bar w=0.4152$). Fixing $\sigma^2=1$ gives $SE(\hat\beta_1)=0.15130/\sqrt{0.98699}=$ **0.1523**, almost identical to the estimated-scale value.

**4.** OLS with HC3 gives $SE(\hat\beta_1)=$ **0.1966** (HC0 0.1916, HC1 0.1928), about 1.5 times the classical 0.1299. The fair comparison is 0.1966 against the WLS 0.1513, both valid under the stated heteroskedasticity: the variance ratio is $(0.1513/0.1966)^2=0.59$, so WLS has about 59% of the OLS sampling variance (the OLS standard error is 1.30 times the WLS one). Using HC3 for WLS as well (0.1597) gives a ratio of 0.66; the conclusion is the same.

**5.** With weight $1/\sqrt{\text{variance}}=1/(1+|x|)$ the coefficients are 2.0008, 1.4512 and 0.8220, still close to the others because every weighting is unbiased here. The nonrobust standard error of the `x` slope is 0.1389 and the HC3 standard error is 0.1723. Report the HC3 value: the nonrobust formula assumes $\operatorname{Var}(u_i)=\sigma^2/w_i$, which these weights get wrong. Its precision lies between correct WLS (0.1513 model-based, 0.1597 HC3) and OLS (0.1966 HC3), as expected from a weight that only partly matches the variance shape.

**6.** For example: "Weighting by the known inverse variances leaves the x slope essentially unchanged (1.446 against 1.447 for OLS). The classical OLS standard error of 0.130 assumes constant variance and understates OLS uncertainty here; the heteroskedasticity-robust OLS standard error is 0.197, larger than the WLS standard error of 0.151, so weighting improved precision."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/06-weighted-least-squares/weighted_least_squares.xlsx) for the published analysis. The [original generator](generators/06-weighted-least-squares.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/06-weighted-least-squares/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/06-weighted-least-squares/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: transformed design identity, strictly positive weights, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Misspecified weights can reduce efficiency or alter the appropriate covariance. Survey and frequency weights require their own contracts and must not be relabeled as precision weights.
