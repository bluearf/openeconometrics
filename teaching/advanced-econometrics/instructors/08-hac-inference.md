# Instructor companion · Lab 08

## Teaching sequence

Start from the double sum $\operatorname{Var}(\sum_ts_t)=\sum_t\sum_{t'}E[s_ts_{t'}']$ and group its terms by lag before introducing any kernel. Then motivate the Bartlett weights (0.8, 0.6, 0.4, 0.2 for four lags) as the price of a positive semidefinite, truncated estimate, and connect the hand-built block to the formula line by line.

Before running, ask students whether the intercept or the `x` slope should be more affected by persistent errors. After running, use the ratio column (intercept 1.60, `x` 0.87, `z` 0.93) and the factorization of the score autocovariance to show that error persistence matters only through the regressor's own persistence, as with clustering in Lab 07.

## Worked exercise answers

**1.** OLS residual autocorrelations at lags 1–4: 0.590, 0.320, 0.229, 0.169, close to $0.6^\ell=0.600, 0.360, 0.216, 0.130$. Autocorrelations of `x`: −0.032, −0.060, −0.060, −0.035, all near zero and slightly negative. With $\rho_x(\ell)\approx0$, the product $\rho_x(\ell)\rho_u(\ell)$ is near zero, so serially correlated errors should barely change the slope's standard error, and the slightly negative values are consistent with the HAC standard error coming out below the independent one.

**2.** For the `x` slope, the lag-zero part of the meat is 294.17 and the Bartlett-weighted lag contributions are −26.82 (lag 1), −35.54 (lag 2), −13.76 (lag 3) and +8.01 (lag 4), summing to −68.11; the full meat is 0.768 times the lag-zero part. (That ratio times the HC1 variance reproduces the HAC standard error: $0.07007\times\sqrt{0.768}=0.0614$.) For the intercept, the lag-zero part is 342.03 and the contributions are +322.19, +129.47, +63.37 and +23.23; the ratio is 2.574. Every intercept term is positive because its score is essentially the persistent error itself.

**3.** Intercept and `x` standard errors by lag count:

| `lags` | Intercept SE | `x` SE |
|---:|---:|---:|
| 0 | 0.0777 | 0.0701 |
| 2 | 0.1098 | 0.0649 |
| 4 | 0.1247 | 0.0614 |
| 8 | 0.1399 | 0.0600 |
| 12 | 0.1495 | 0.0588 |
| 20 | 0.1556 | 0.0596 |

`lags=0` keeps only $\hat\Gamma_0$ with the factor $n/(n-k)$, which is HC1 (0.0777, 0.0701, 0.0817). The intercept's standard error rises steadily as more of the error persistence is included; the `x` standard error drifts slightly down and then levels off, staying below the HC values throughout.

**4.** With $\rho_x=1$ for the constant, the predicted ratio is $1+2(0.8\cdot0.6+0.6\cdot0.36+0.4\cdot0.216+0.2\cdot0.1296)=1+2(0.8083)=2.617$. The observed HAC-to-HC3 variance ratio is $1.595^2=2.545$ (2.574 against HC1, which uses the same finite-sample factor). The untruncated long-run factor is $(1+0.6)/(1-0.6)=4$, a standard-error ratio of 2, so four Bartlett lags capture only about 65% of the intercept's long-run variance (2.62 of 4). With `lags=20` the intercept's ratio to HC1 reaches $(0.1556/0.0777)^2\approx4.0$.

**5.** After `sample(frac=1, random_state=8)` and resetting `time`, the four-lag HAC standard errors are 0.0775 (intercept), 0.0666 (`x`) and 0.0917 (`z`); the coefficients are unchanged because OLS does not depend on row order. The intercept's standard error falls from 0.1247 to roughly the HC3 level (0.0782) because shuffling breaks the persistence: neighboring rows are no longer neighboring periods, so the lagged cross-products average out. The slopes' standard errors move only by chance. HAC is valid only when `time` records the true order.

**6.** For example: "The OLS slope on x is 1.270. Its Bartlett HAC standard error with four lags is 0.061, close to the HC3 value of 0.071, because x itself shows little serial correlation, so the persistent errors add little to the slope's uncertainty. The errors are still serially correlated (the intercept's standard error rises from 0.078 to 0.125 under HAC), so the time order matters and the HAC result is the one to report."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/08-hac-inference/hac_inference.xlsx) for the published analysis. The [original generator](generators/08-hac-inference.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/08-hac-inference/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/08-hac-inference/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: covariance only preserves slope, regular unique time, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** HAC is not a cure for omitted dynamics, endogenous regressors or nonstationarity. Different lag choices change the estimated covariance and should be sensitivity analyses rather than significance searches.
