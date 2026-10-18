# Instructor companion · Lab 07

## Teaching sequence

Begin with the sampling unit: 40 groups of six consecutive rows. Write the error-components equation and the block-diagonal covariance, then derive the cluster meat as the row-level meat plus within-group score cross-products before showing any output.

Before running, ask students to predict which coefficient's standard error will change most under clustering. After running, use the ratio column (intercept 1.81, `x` 0.96, `z` 1.13) and the within-group correlation approximation to explain why a regressor that is constant within groups is affected most, and why a cluster standard error can fall below HC3 without the dependence being absent.

## Worked exercise answers

**1.** With every row its own cluster, $S_g$ is a single row's score, so the meat is $\sum_i(X_i\hat e_i)(X_i\hat e_i)'$, the HC0 meat. The factor becomes $\frac{n}{n-1}\cdot\frac{n-1}{n-k}=\frac{n}{n-k}=240/237=1.0127$, which is the HC1 degrees-of-freedom factor. The standard errors are 0.0901 (intercept), 0.0895 (`x`) and 0.0829 (`z`), identical to `covariance="HC1"`.

**2.** One-way ANOVA intraclass correlations by `group`: `x` −0.052, `z` 0.007, OLS residuals 0.461. With $T=6$ the approximation predicts variance ratios of $1+5(1)(0.461)=3.31$ for the intercept, $1+5(-0.052)(0.461)=0.88$ for `x` and $1+5(0.007)(0.461)=1.02$ for `z`. The actual ratios of squared standard errors are 3.28, 0.93 and 1.27. The approximation captures the intercept well and the direction for `x`; it misses `z` because it describes expected inflation, while the realized cross-products of `z`'s scores across only 40 groups happen to be positive. The actual ratios also mix two different finite-sample corrections (CR1 against HC3).

**3.** After partialling out the other regressors: for `x`, squared row scores sum to 442.15 and within-group cross-products to **−28.74** (−6.5% of the own part), total 413.42; for `z`, 598.50 and **+166.78** (+27.9%), total 765.27. (For the intercept, 455.70 and +1027.30.) Without finite-sample factors this gives CR0 standard errors of 0.0860 for `x`, below HC0 0.0890, and 0.0931 for `z`, above HC0 0.0824. The negative cross-product sum is why the `x` ratio is below one; the positive one is why the `z` ratio is above one.

**4.** HC3: critical value $t_{0.975,237}=1.9700$, interval [1.0294, 1.3867], width 0.3573. Clustered: $t_{0.975,39}=2.0227$, interval [1.0311, 1.3850], width 0.3539. The clustered interval is slightly narrower, but only by 1% (width ratio 0.990), not by the 3.5% that the standard-error ratio 0.965 suggests, because the larger critical value (ratio 1.027) offsets most of the smaller standard error.

**5.** Pairs of adjacent groups (20 clusters): intercept 0.1973, `x` 0.0841, `z` 0.1015, with 19 degrees of freedom. Cross-cutting grouping `(row_id - 1) % 40` (40 clusters): intercept 0.0919, `x` 0.0858, `z` 0.0648, with 39 degrees of freedom. The pairing nests every true group inside one cluster, so it remains valid; it costs degrees of freedom and its standard errors are noisier. The cross-cutting grouping puts the six rows of each true group into six different clusters, so the shared shock becomes correlation between clusters, which the estimator assumes away; the intercept's standard error falls back to roughly the HC3 level and understates uncertainty. A grouping must follow the dependence, not the result.

**6.** For example: "The OLS slope on x is 1.208. Clustering by the 40 groups gives a standard error of 0.087 on 39 degrees of freedom, close to the row-level HC3 value of 0.091, because x varies within groups. The within-group dependence is still present (the intercept's standard error rises from 0.091 to 0.164 under clustering), so inference rests on 40 independent groups and the clustered standard error is the one to report."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/07-clustered-inference/clustered_inference.xlsx) for the published analysis. The [original generator](generators/07-clustered-inference.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/07-clustered-inference/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/07-clustered-inference/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: covariance only preserves coefficients, df is clusters minus one, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** With very few clusters, ordinary cluster asymptotics may be unreliable. Choosing groups after searching for favorable significance does not supply a valid sampling design.
