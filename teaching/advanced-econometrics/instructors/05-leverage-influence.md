# Instructor companion · Lab 05

## Teaching sequence

Start with the hat matrix and derive $\operatorname{tr}(H)=k$ before showing any output, so that the average leverage 0.0125 is a prediction rather than a surprise. Then separate three quantities that students often merge: leverage (from the regressors only), the residual (shrunk at high-leverage rows) and influence (the deletion formula, which needs both).

Before running, ask for the average leverage. After running, keep three responses to row 1 apart: HC3 keeps the row and widens the standard error; the sensitivity fit drops it and changes the estimate; neither shows that the row is wrong. Note that the generic dictionary labels do not mention that row 1 is a constructed stress row; the student handout now says so.

## Worked exercise answers

**1.** With $k=3$ and $n=240$, $2k/n=0.025$ and $3k/n=0.0375$. Fourteen rows exceed 0.025 and four exceed 0.0375: `row_id` 1 (0.2648), 126 (0.0419), 148 (0.0439) and 181 (0.0558). Even with well-behaved regressors, leverage has a right-skewed distribution (it is a scaled squared distance from the center), so fixed cutoffs always flag some rows; they mark rows to inspect, not rows to delete.

**2.** The fit without row 1 predicts $1.9687+1.2404\times8+0.8279\times0=11.892$ for row 1, whose recorded `y` is 21.995, so the prediction error is 10.103, exactly the summary table's $\hat e_1/(1-h_{11})$. Proof: $y_1-\mathbf x_1'\hat\beta_{(1)}=\hat e_1+\mathbf x_1'(\hat\beta-\hat\beta_{(1)})=\hat e_1+h_{11}\hat e_1/(1-h_{11})=\hat e_1/(1-h_{11})$, using the deletion formula and $\mathbf x_1'(X'X)^{-1}\mathbf x_1=h_{11}$. The leverage-scaled residual is the leave-one-out prediction error.

**3.** With `y` for row 1 set to 11.892, row 1's leverage is unchanged at 0.2648 and the all-rows coefficients equal those of the fit without row 1 (intercept 1.9687, `x` 1.2404, `z` 0.8279); row 1's residual is zero. The row keeps its full potential to move the fit, but because it agrees with the other rows it has no influence. Leverage is a property of position; influence requires disagreement.

**4.** The standard errors on `x` are 0.0734 (classical), 0.2518 (HC1) and 0.3354 (HC3). The classical formula pools one residual variance across rows, so row 1's large residual is averaged away. HC1 uses each row's own squared residual, and row 1's residual at `x` equal to 8 dominates the `x` element of the sandwich. HC3 further multiplies that contribution by $1/(1-0.2648)^2=1.85$.

**5.** With $s=1.1516$, $D_1=7.428^2\times0.2648/[3\times1.1516^2\times(1-0.2648)^2]=6.79$. The largest value among the other rows is 0.044 (`row_id` 101). Row 1's Cook's distance is about 150 times the next largest.

**6.** For example: "With all 240 synthetic rows, the slope on `x` is 1.57 (HC3 standard error 0.34). One row at `x` equal to 8 has leverage 21 times the average and a leave-one-out prediction error of 10.1; without it the slope is 1.24 (HC3 standard error 0.08). We report both fits, because nothing indicates that the row is an error, and we describe the slope as sensitive to that row."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/05-leverage-influence/leverage_influence.xlsx) for the published analysis. The [original generator](generators/05-leverage-influence.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/05-leverage-influence/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/05-leverage-influence/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: leverage trace equals rank, flagged leverage above average, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Sensitivity does not justify removing a valid row merely to change significance. Data-quality evidence, sampling scope and robustness should be discussed separately.
