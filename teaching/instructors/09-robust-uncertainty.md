# Instructor notes · Lab 09

The student handout is [here](../labs/09-robust-uncertainty/index.md). Preserve the comparison's same rows, intercept and regressors. The deliberately modest difference between classical and robust slope SEs helps students reject the rule that visible heteroskedasticity must greatly enlarge every SE.

## Worked answers

1. An additional 1,000 income units corresponds to 44.169160 fitted spending units. An additional 500 corresponds to 22.084580. These are level differences, not percentages.
2. The OLS objective is unchanged. Sample selection, weights, regressor transformations or intercept treatment would break the comparison if changed.
3. Conditional mean zero holds; constant conditional variance fails because the error scale is $12+10X$.
4. RMS measures outcome noise; coefficient variance also depends on the design matrix and the locations of that noise. There is no contradiction.
5. Evaluate the actual median-split fits. Within each half, all covariance choices must retain the same rows and design. Across halves, income support, sample size and realized residuals change, so an SE difference cannot be attributed solely to heteroskedasticity. No one realized split establishes a universal ranking.
6. Repeated rows may have within-household dependence. Consider the sampling design and household clustering rather than treating twelve records as independent household draws.
7. A suitable paragraph reports 420 original simulated households, slope 44.169160, HC3 SE 1.375479, interval [41.465442,46.872879], units and the associational scope.

## Reproduction and verification

Run the complete paired source in a clean session. `run_lab(output_dir="lab09-review")` explicitly exports full `reference.json` and `table.tex`; default execution writes no files. The checked-in [complete reference](../labs/09-robust-uncertainty/reference.json) retains covariance matrices and sample positions. Independent matrix calculations check classical, HC1 and HC3 covariances, slope equality, all 420 positions, t(418) conventions and full JSON roundtrip. Initial coefficient and covariance maximum errors were below $2\times10^{-13}$.

Native callback outputs are `OLSResult`, `PlotSpec`, `Latex`. Treat the simulated heteroskedasticity as known by construction; do not use the residual plot to claim universal diagnostic power or causal identification.

**Prepared input:** [household_spending.xlsx](../labs/09-robust-uncertainty/household_spending.xlsx). Use the stored observations for the published analysis. The [original generator](generators/09-robust-uncertainty.py) supports new dataset editions; update results and answers when changing observations.
