# Published Stata reference validation

The pinned 74-row numeric projection of Stata's [r19 `auto.dta`](https://www.stata-press.com/data/r19/auto.dta) is compared with three official publications. This is a source-run reference check. No Stata executable was run and no installed desktop package is certified by it.

| Native specification | Published reference | Checked output |
| --- | --- | --- |
| `oe.ols(data=data, y="mpg", x=["weight", "foreign"], covariance="nonrobust", device="cpu")` | [`regress`, Example 1, printed p. 6](https://www.stata.com/manuals/rregress.pdf) | All three coefficient estimates, SEs, t statistics, p values and 95% intervals; sums of squares, R², adjusted R², root MSE and model F. Exact N = 74 and residual df = 71. |
| `oe.ttest(data, "mpg", mu=20)` | [`ttest`, Example 1, printed p. 4](https://www.stata.com/manuals/rttest.pdf) | Mean, SD, SE, mean confidence interval, t statistic and lower, two-sided and upper tail probabilities. Exact N = 74 and df = 73. |
| `oe.logit(data=data, y="foreign", x=["weight", "length"], covariance="nonrobust")` | [Official marginal-effects FAQ](https://www.stata.com/support/faqs/statistics/marginal-effects-methods/) | All three coefficients, SEs, z statistics, p values and 95% intervals; log likelihood and pseudo R². |
| `oe.margins(saved_logit, data=data, method="mem")` | Same FAQ, historical `mfx, predict(p) nose tracelvl(2)` | Continuous weight and length effects at predictor means, probability at means and logistic link slope. Point estimates only. |

The FAQ explicitly describes Stata 10 and earlier. Stata 11 replaced `mfx` with `margins`. The example's evaluation point is the mean of the predictors: it is compared with native MEM, not the average of rowwise marginal effects (AME). It publishes no effect SEs or confidence intervals because it uses `nose`. The current r19 data projection matches its printed numbers, but the historical example's original binary dataset/version is not identified.

Each printed decimal is kept as a string, including trailing zeros. Its tolerance is half the last printed decimal unit plus `32 * binary64_epsilon * max(1, abs(reference), abs(observed))` for negligible conversion noise. For example, `.0006371` permits approximately `5e-8`, whereas `0.130` permits approximately `5e-4`. A printed `0.000` is a rounded small probability, not an exact zero. Counts and degrees of freedom are compared exactly. These are checks against displayed precision; they are not full-precision Stata equivalence.

Both a resident DataFrame and a replayable Dataset with observed seven-row producer blocks are checked. JSON-restored logit results are used for MEM and outcome-free prediction at predictor means. Published t-test bounds are compared to the native **statistics** table's interval for the mean; the native **test** table instead contains an interval for the mean minus 20.

The input checksum and download source are in `tests/fixtures/stata/auto-source.json`. Printed reference cells and their locations are in `published-auto-reference.json`; `tests/test_stata_published_auto_reference.py` implements the comparisons. The public receipt is `docs/evidence/stata-published-auto-2026-10-06.json`.

`tests/fixtures/stata/reproduce-auto-reference.do` is an unexecuted reproduction script for a future operator with a licensed Stata 19 installation. It accepts the pinned CSV and an existing dedicated output directory, records Stata's version, and saves full-precision coefficient/covariance logs and fitted `.ster` files. It logs whether obsolete `mfx` is available and separately runs modern `margins, dydx(weight length) atmeans predict(pr)`. Any resulting modern MEM uncertainty would be a new validation source, not uncertainty recovered from the historical `nose` example.

This evidence covers only these complete-data, unweighted specifications and printed cells. It does not validate the unpublished off-diagonal covariance entries, MEM uncertainty, other VCE options, weighting or missing-value semantics, all estimators, or large-data performance.
