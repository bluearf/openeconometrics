# Instructor companion · Lab 14

## Teaching sequence

Before running, derive the differenced equation and have students state the three construction rules (sort, group, drop) and the expected row count. Then derive $\operatorname{Cov}(\Delta u_{it},\Delta u_{i,t-1})=-\sigma_u^2$ for independent level errors and ask what that implies for treating 150 changes as independent.

After the run, read the bookkeeping check before the coefficients, then contrast the clustered and HC3 standard errors and the first-difference and within estimates. Keep the point that the two estimators coincide only with two periods.

## Worked exercise answers

**1.** A plain `.diff()` on the sorted frame leaves 179 rows, of which 29 changes straddle two units (each unit's period 0 differenced against the previous unit's period 5). The clustered X slope becomes 1.4383 (against 1.3871); the contaminated rows compare different units' levels, so the unit effects no longer cancel.

**2.** Unit fixed effects on the same workbook give an X slope of 1.2983 (cluster SE 0.0743) against 1.3871 (0.0776) for first differences. Both remove the unit effects but weight the within variation differently, so with six periods they differ. With independent level errors the within estimator is more efficient; with random-walk level errors first differences is.

**3.** Over the 120 pairs of consecutive residuals within units, the correlation is −0.486, close to the −1/2 implied by independent level errors. That pattern suggests the level errors are roughly independent over time, which favors the within estimator on efficiency grounds and confirms that the changes must not be treated as independent.

**4.** With HC3 (147 residual degrees of freedom) the X standard error is 0.0638 and the Z standard error 0.0763, against clustered values of 0.0776 and 0.0950; clustering raises the X standard error by about 22%. HC3 assumes the 150 changes are independent, which Section 3 shows is false, so the unit-clustered standard errors are the appropriate ones.

**5.** On periods 4 and 5 only, first differences without an intercept and unit fixed effects both give an X slope of 1.5833; first differences with an intercept and fixed effects with a period-5 dummy both give 1.5415. With two periods, each unit's demeaned values are plus and minus half its single change, so the two transformations carry the same information. The clustered standard errors still differ (for example 0.169 against 0.241 for the intercept pair) because the CR1 factor $(N-1)/(N-K)$ is computed on 30 rows with 3 coefficients in one case and 60 rows with 33 coefficients in the other. With six periods, demeaning and differencing weight the observations differently.

**6.** For example: "In a balanced synthetic panel of 30 units over 6 periods, the 150 within-unit changes give a first-difference X slope of 1.387 outcome units per X unit (standard error clustered by unit 0.078, 29 degrees of freedom). The changes are not independent: adjacent changes share a level error, so inference rests on the 30 units. The estimate removes time-invariant unit levels and assumes errors in adjacent periods are unrelated to the regressors."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/14-first-differences/first_differences.xlsx) for the published analysis. The [original generator](generators/14-first-differences.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/14-first-differences/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/14-first-differences/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: one first period lost per unit, positive remaining time, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Differencing can amplify measurement error and does not remove time-varying confounding. With more than two periods its efficiency and estimand weighting can differ from within estimation.
