# Model building: stepwise selection, collinearity, curve estimation, group summaries

Four procedures that surround a regression analysis rather than fit one model:

| procedure | what it does | SPSS | Stata |
| --- | --- | --- | --- |
| `oe.stepwise` | forward, backward and stepwise selection of regressors | `REGRESSION /METHOD=STEPWISE \| FORWARD \| BACKWARD` | `stepwise, pe() pr(): regress` |
| `oe.collin` | tolerance, VIF, condition indices, variance proportions | `REGRESSION /STATISTICS=COLLIN TOL` | `estat vif`, `collin`, `coldiag2` |
| `oe.curvefit` | eleven one-predictor curve models | `CURVEFIT` | `regress` on transformed variables |
| `oe.tabstat` | weighted summary statistics by group, eta-squared ANOVA | `MEANS` | `tabstat`, `summarize, detail` |

Everything on this page is computed in OpenEconometrics on float64 PyTorch tensors. No
statistics library runs at fit time; NumPy, SciPy and statsmodels appear only
in the test suite as independent oracles.

These are procedures, not estimators: they register nothing in the model
registry and return **result tables** (`openecon.frame.DataFrame`, printable and
exportable with `.to_latex()`) or a `TableSet`, a dictionary of named tables
with `str()` and `.to_latex()` for the whole set. Scalar results and settings
are in `.attrs`.

```python
import openecon as oe

result = oe.stepwise(df, "y", ["x1", "x2", "x3", "x4"])
print(result)                       # every table
result["coefficients"]              # one table
result.attrs["selected"]            # terms of the final model
result["steps"].to_latex()
```

Plain OLS with robust covariance, tests and margins is `oe.ols`; use it to
refit the model you settle on. `oe.describe` (classical-statistics family) is
the unweighted descriptive table with confidence intervals; `oe.tabstat` adds
weights and the SPSS MEANS statistics.

## Samples, missing data and failures

Inputs are a DataFrame, a mapping of columns or a list of row records.

| procedure | missing values |
| --- | --- |
| `stepwise` | **listwise** over the outcome, *every* candidate, forced term and the weight, so that all steps use the same sample (as SPSS and Stata do). `missing="drop"` (default) excludes the rows and reports `attrs["n_missing"]`; `missing="raise"` rejects them (`missing_values`). |
| `collin` | listwise over the regressors; `missing="drop"` / `"raise"`. |
| `curvefit` | listwise over `y` and `x`; `missing="drop"` / `"raise"`. |
| `tabstat` | **variable-wise**: each variable uses its own non-missing values (Stata `tabstat`, SPSS `MEANS`); `listwise=True` keeps only rows complete in all variables (Stata's `casewise`). Rows with a missing group or weight are always excluded. |

Weights (`stepwise`, `tabstat`) are `weight_type="aweight"` (analytic, the
default when a weight column is given) or `"fweight"` (frequency: a row counts
`w` times and `n` is the sum of the weights). Observations with a zero weight
leave the sample, as in Stata, and are mentioned in `attrs["notes"]`.

Every failure is an `openecon.AnalysisError` with a `code` and a message that
says what to change: `invalid_spec`, `invalid_option`, `missing_columns`,
`non_numeric_column`, `missing_values`, `empty_data`, `empty_sample`,
`non_finite_values`, `negative_weights`, `noninteger_frequency_weights`,
`zero_variance`, `insufficient_observations`, `perfect_fit`,
`perfect_collinearity`, `design_too_large`, `duplicate_columns` (the data have
two columns with one name), `duplicate_terms` (a predictor name collides with a
generated indicator name such as `region[north]`), `invalid_groups`
(unhashable group labels) and, should a factorization fail,
`numerical_failure` / `singular_design`. Statistics that are undefined for
part of a table (the standard deviation of one observation, the F statistic of
an exact fit, a curve model whose transformation does not exist for the data)
are missing cells, never text, NaN strings or infinities; the reason is given
in `attrs["notes"]`.

---

## `oe.stepwise` — stepwise, forward and backward selection

```python
oe.stepwise(data, y, x, *, method="stepwise", criterion="pvalue",
            p_enter=0.05, p_remove=0.10, forced=None, categorical=None,
            weights=None, weight_type=None, tolerance=1e-4,
            missing="drop", alpha=0.05)
```

### Model and search

The model is the linear regression `y = b0 + sum_j b_j x_j + e` with a
constant, fitted by (weighted) least squares. A **term** is one column, or all
indicator columns of a `categorical` predictor, which enter and leave together.

For a term of `q` columns and a current model with `k` regressor columns, `n`
observations and residual sum of squares `SSE`:

```
F-to-enter  = ((SSE - SSE_with) / q)    / (SSE_with / (n - 1 - k - q))
F-to-remove = ((SSE_without - SSE) / q) / (SSE      / (n - 1 - k))
```

referred to the F distribution with `(q, df2)` degrees of freedom. For a single
column they are the squared t statistic of that regressor.

| `method` | search |
| --- | --- |
| `"forward"` | start from the constant (plus `forced` terms); enter the term with the smallest p-value while `p < p_enter`. |
| `"backward"` | start from all terms; remove the term with the largest p-value while `p > p_remove`. |
| `"stepwise"` | forward selection in which, before every entry, the term with the largest p-value is removed if `p > p_remove` (SPSS STEPWISE). Needs `p_enter <= p_remove`. |

`criterion="aic"` or `"bic"` replaces the significance tests: at each step the
single entry (forward), removal (backward) or either (stepwise) that lowers
`n ln(SSE/n) + c k` the most is taken (`c = 2` for AIC, `ln n` for BIC), until
no move lowers it. `p_enter` and `p_remove` are then ignored; the F test of
each step is still reported.

**Tolerance.** A term may enter only if its tolerance — the share of its
variance not explained by the regressors already in the model — is at least
`tolerance` (SPSS default `0.0001`), and if entering it would not push the
tolerance of a regressor already in the model below that value (SPSS's
*minimum tolerance* test). This is what keeps duplicated or nearly collinear
candidates out; they appear in the `excluded` table with their tolerance. In
backward elimination, terms that fail the test against the terms listed before
them are left out of the starting model and named in `attrs["notes"]`.

**Forced terms** (`forced=[...]`, SPSS `/METHOD=ENTER`, Stata `lockterm1`) are
entered first and never tested for removal. They do not need to be listed in
`x`.

**Categorical terms** (`categorical=[...]`) are expanded into indicator
columns, first level as reference, and tested with all their degrees of
freedom, like Stata's parenthesized terms `(i.region)`. SPSS REGRESSION has no
equivalent.

### How it is computed

For the search the data are read **once**, in row blocks, to form the
mean-centred weighted cross-product matrix of all candidate columns and the
outcome
(`C = sum_i w_i (z_i - zbar)(z_i - zbar)'`; centring is the sweep of the
constant). `C` is scaled to the correlation matrix `R`, and the whole search
runs on that `(p+1) x (p+1)` matrix with the **sweep operator**. After
sweeping the columns of the current model `S`:

- `A_yy = 1 - R^2`;
- for a column `k` in the model, `A_ky` is its standardized coefficient and
  `A_ky^2 / (-A_kk)` the R-squared lost by removing it;
- for a column outside, `A_kk` is its tolerance and `A_ky^2 / A_kk` the
  R-squared gained by entering it.

A step is one rank-one update of the small matrix, so the cost of the search
does not depend on the number of rows. The final model is solved again from the
selected block of `R` by Cholesky, so reported coefficients do not carry
rounding from the search path.

**Precision of the final model.** On the correlation matrix the residual share
is a difference, `1 - R^2 = 1 - r' R^-1 r`, which loses about
`log10(1 / (1 - R^2))` digits as R-squared approaches one (SPSS works on the
same matrix). The final model therefore takes its residual sum of squares from
a second pass over the data, `SSE = sum_i w_i ((y_i - ybar) - (x_i - xbar)' b)^2`.
That pass costs `O(n k)`, and an error `d` in `b` changes the sum only by
`d' X'X d`. Standard errors, the ANOVA table, R-squared, `rmse`, `aic`, `bic`
and the last row of `steps` (whose model *is* the final model) agree with
statsmodels' least-squares fit to better than 1e-8 relative in the test suite,
down to `1 - R^2` of about 1e-15. The statistics of
earlier steps and of the `excluded` table come from the sweeps. They are exact
to about `1e-16 / (1 - R^2)` relative, which is negligible unless the fit is
almost perfect.

**Almost perfect and perfect fits.** When a step brings `1 - R^2` below `1e-13`,
the swept matrix holds at most one or two correct digits of the residual share. The search then
stops: the F statistic and p-value of that step, and the entry statistics of the
excluded terms, are left missing (their tolerances are still shown), and
`attrs["notes"]` explains why. The final model is still reported from the data.
If the recomputed residual sum of squares is zero to rounding
(`SSE <= 1e-24 SST`), the outcome is an exact linear combination of the
selected terms. No standard error exists then, and the call fails with
`perfect_fit`, naming those terms.

### Weights

`weight_type="aweight"` (default; WLS weights, SPSS `/REGWGT`): `n` is the
number of rows and the weights are rescaled to sum to `n`, as Stata's
`regress [aw=]` does. Coefficients, standard errors, t and F tests, R-squared
and the standardized coefficients do not depend on the scale of the weights.
The sums of squares in `anova`, `rmse` and the log likelihood behind `aic` /
`bic` are on Stata's normalized scale. SPSS's `REGWGT` output uses the weights
as given, so its sums of squares and standard error of the estimate differ by
the factor `sum(w) / n` (or its square root). `weight_type="fweight"`: each row
counts `w` times and `n = sum w`. The results are identical to those of the
expanded data set (this is SPSS's `WEIGHT BY`).

### Returned tables

| table | content |
| --- | --- |
| `steps` | one row per step: `step`, `action` (`entered` / `removed`; a `start` row with step 0 when the starting model has regressors), `term`, `statistic` (F-to-enter or F-to-remove — SPSS's *F Change*; for `start` the overall F), `df1`, `df2`, `p_value`, then the model after the step: `r_squared`, `adjusted_r_squared`, `r_squared_change`, `rmse`, `aic`, `bic`. |
| `coefficients` | final model, `Intercept` first, then terms in order of entry: `b`, `std_error`, `beta` (standardized), `t`, `p_value`, `ci_low`, `ci_high`, `tolerance`, `vif`. |
| `excluded` | every term not in the final model: `beta_in`, `t`, `statistic` (F-to-enter), `df1`, `df2`, `p_value`, `partial_correlation`, `tolerance`, `min_tolerance` (SPSS *Excluded Variables*). `beta_in` and `t` exist for one-column terms; for a categorical term `partial_correlation` is the nonnegative root of the partial R-squared. |
| `anova` | `regression`, `residual`, `total`: `ss`, `df`, `ms`, `statistic`, `p_value`. |

Inference in `coefficients` is classical: `s^2 = SSE / (n - k - 1)`, Student t
with `n - k - 1` degrees of freedom, `1 - alpha` confidence intervals.
`aic` and `bic` follow Stata's `estat ic` after `regress`:
`-2 lnL + 2 (k+1)` and `-2 lnL + (k+1) ln n` with
`lnL = -n/2 (ln 2 pi + ln(SSE/n) + 1)`.

`attrs`: `selected` (terms of the final model in order of entry), `n`,
`n_missing`, `r_squared`, `adjusted_r_squared`, `rmse`, `statistic`, `df1`,
`df2`, `p_value`, `aic`, `bic`, the settings, `weights` and `notes`.

### Worked example: Hald's cement data

The classic example of Draper and Smith, with the 0.15 thresholds used there:

```python
hald = pd.DataFrame({
    "x1": [7, 1, 11, 11, 7, 11, 3, 1, 2, 21, 1, 11, 10],
    "x2": [26, 29, 56, 31, 52, 55, 71, 31, 54, 47, 40, 66, 68],
    "x3": [6, 15, 8, 8, 6, 9, 17, 22, 18, 4, 23, 9, 8],
    "x4": [60, 52, 20, 47, 33, 22, 6, 44, 22, 26, 34, 12, 12],
    "y": [78.5, 74.3, 104.3, 87.6, 95.9, 109.2, 102.7, 72.5, 93.1, 115.9,
          83.8, 113.3, 109.4]})
result = oe.stepwise(hald, "y", ["x1", "x2", "x3", "x4"],
                     p_enter=0.15, p_remove=0.15)
```

```
[steps]
   step   action term  statistic  df1   df2     p_value  r_squared  r_squared_change
      1  entered   x4    22.7985    1  11.0  5.7623e-04     0.6745            0.6745
      2  entered   x1   108.2239    1  10.0  1.1053e-06     0.9725            0.2979
      3  entered   x2     5.0259    1   9.0  5.1687e-02     0.9823            0.0099
      4  removed   x4     1.8633    1   9.0  2.0540e-01     0.9787           -0.0037

[coefficients]
                 b  std_error    beta        t     p_value   ci_low  ci_high  tolerance     vif
Intercept  52.5773     2.2862     NaN  22.9980  5.4566e-10  47.4834  57.6713        NaN     NaN
x1          1.4683     0.1213  0.5741  12.1047  2.6922e-07   1.1980   1.7386     0.9478  1.0551
x2          0.6623     0.0459  0.6850  14.4424  5.0290e-08   0.5601   0.7644     0.9478  1.0551

[excluded]
    beta_in       t  statistic  df1  df2  p_value  partial_correlation  tolerance  min_tolerance
x3   0.1064  1.3536     1.8321    1  9.0   0.2089               0.4113     0.3183         0.3076
x4  -0.2632 -1.3650     1.8633    1  9.0   0.2054              -0.4141     0.0528         0.0528
```

`x4` is the best single predictor, but once `x1` and `x2` are in the model it
adds nothing and is removed. With the SPSS defaults (`p_enter=0.05`,
`p_remove=0.10`) `x2` (p = 0.052) is not entered and the search stops at
`x4, x1`. The equivalent commands are

```
REGRESSION /CRITERIA=PIN(.15) POUT(.15) /DEPENDENT y /METHOD=STEPWISE x1 x2 x3 x4.
stepwise, pe(.15) pr(.15) forward: regress y x1 x2 x3 x4
```

Other calls:

```python
oe.stepwise(df, "wage", xs, method="backward", p_remove=0.10)
oe.stepwise(df, "wage", xs, criterion="bic")                       # lowest BIC, both directions
oe.stepwise(df, "wage", [*xs, "region"], categorical=["region"],   # region enters as a block
            forced=["age"], weights="w")                           # age always in; WLS
```

### Limitations

- Linear regression with a constant only (no `noconstant`, no logit / Poisson
  stepwise).
- p-values and intervals of the final model are those of an ordinary
  regression on the selected terms; they ignore the selection and are
  optimistic. This is how SPSS and Stata report them too.
- Classical covariance only. Refit the selected model with `oe.ols` for robust
  or clustered standard errors.
- No hierarchical terms (Stata's `hierarchical`) and no interactions; build
  the columns beforehand.

---

## `oe.collin` — collinearity diagnostics

```python
oe.collin(data, x, *, intercept=True, missing="drop")
```

For the design matrix `[1, x_1, ..., x_k]` of a linear model (the outcome
plays no role):

- **Tolerance and VIF.** With `R_k^2` the R-squared of the regression of
  `x_k` on the other regressors and the constant,
  `tolerance_k = 1 - R_k^2` and `VIF_k = 1 / tolerance_k`. The VIFs are the
  diagonal of the inverse correlation matrix of the regressors.
- **Condition indices and variance-decomposition proportions** (Belsley, Kuh
  and Welsch 1980). The columns of the design, *including the constant and
  not centred*, are scaled to unit length. With singular values
  `d_1 >= ... >= d_p` and right singular vectors `v_j` of the scaled design,
  the eigenvalues of the scaled cross-product matrix are `d_j^2`, the
  condition indices `d_1 / d_j`, and the proportion of `var(b_k)` associated
  with dimension `j` is

  ```
  pi_jk = (v_kj^2 / d_j^2) / sum_j (v_kj^2 / d_j^2).
  ```

  Each coefficient's proportions sum to one. The usual reading: a condition
  index above about 30 together with two or more proportions above 0.5 in the
  same row identifies the regressors involved in a near dependency. A VIF
  above 10 is the common rule of thumb for a problematic regressor.

`intercept=False` analyses the model without a constant: uncentred VIFs
(Stata `estat vif, uncentered`) and a condition table without the constant
column.

**Computation.** One Householder QR of `[1, X - s]` (`s` is a rough mean,
removed only for precision) is accumulated over row blocks; the trailing block
of its triangular factor is the factor of the centred regressors (VIFs from
its inverse), and multiplying the factor by a unit triangular matrix restores
the factor of the raw design (singular values for the condition table). The
cross-product matrix is never formed or inverted.

**Returned tables.** `vif`: one row per regressor with `tolerance`, `vif`,
`r_squared`. `condition`: one row per dimension with `eigenvalue`,
`condition_index` and one column of variance proportions per coefficient
(`Intercept` first). `attrs`: `n`, `n_missing`, `mean_vif`, `max_vif`,
`condition_number` (the largest condition index), `intercept`.

An exact dependency (a duplicated column, a sum of other columns) has an
infinite VIF; it is reported as `perfect_collinearity`, naming the first
column that is a linear combination of the ones before it. A column without
variation next to the constant is `zero_variance`.

### Example

```python
oe.collin(hald, ["x1", "x2", "x3", "x4"])
```

```
[vif]
    tolerance       vif  r_squared
x1     0.0260   38.4962     0.9740
x2     0.0039  254.4232     0.9961
x3     0.0213   46.8684     0.9787
x4     0.0035  282.5129     0.9965

[condition]
           eigenvalue  condition_index   Intercept      x1          x2      x3          x4
1          4.1197e+00           1.0000  5.5089e-06  0.0004  1.8329e-05  0.0002  3.6406e-05
2          5.5389e-01           2.7272  8.8123e-08  0.0100  1.2647e-05  0.0027  1.0070e-04
3          2.8870e-01           3.7775  3.0610e-07  0.0006  3.1981e-04  0.0016  1.6803e-03
4          3.7638e-02          10.4621  1.2679e-04  0.0574  2.7840e-03  0.0457  8.8373e-04
5          6.6138e-05         249.5783  9.9987e-01  0.9316  9.9687e-01  0.9498  9.9730e-01
```

The four ingredients sum to nearly 100 percent in every row of the data, so
they are almost collinear with the constant: the last dimension carries more
than 93% of the variance of every coefficient. For the Longley data the
function reproduces the well-known VIFs (135.5, 1788.5, 33.6, 3.6, 399.2,
759.0) and the condition number 43,275.

Equivalent commands: SPSS `REGRESSION /STATISTICS=COEFF COLLIN TOL /DEPENDENT y
/METHOD=ENTER x1 x2 x3 x4`; Stata `regress y x1 x2 x3 x4`, then `estat vif`
and `coldiag2` (or `collin x1 x2 x3 x4`).

Limitations: numeric columns only (create indicator columns first); no
weights; the diagnostics describe the regressors, not any particular outcome.

---

## `oe.curvefit` — curve estimation

```python
oe.curvefit(data, y, x, *, models=None, upper_bound=None, intercept=True,
            missing="drop")
```

Eleven one-predictor models, each estimated by ordinary least squares after
the transformation that makes it linear, exactly as SPSS CURVEFIT does:

| model | equation | regression |
| --- | --- | --- |
| `linear` | `y = b0 + b1 x` | `y` on `x` |
| `logarithmic` | `y = b0 + b1 ln(x)` | `y` on `ln x` |
| `inverse` | `y = b0 + b1 / x` | `y` on `1/x` |
| `quadratic` | `y = b0 + b1 x + b2 x^2` | `y` on `x, x^2` |
| `cubic` | `y = b0 + b1 x + b2 x^2 + b3 x^3` | `y` on `x, x^2, x^3` |
| `compound` | `y = b0 * b1^x` | `ln y` on `x`; `b0 = exp(a0)`, `b1 = exp(a1)` |
| `power` | `y = b0 * x^b1` | `ln y` on `ln x`; `b0 = exp(a0)` |
| `s` | `y = exp(b0 + b1 / x)` | `ln y` on `1/x` |
| `growth` | `y = exp(b0 + b1 x)` | `ln y` on `x` |
| `exponential` | `y = b0 * exp(b1 x)` | `ln y` on `x`; `b0 = exp(a0)` |
| `logistic` | `y = 1 / (1/u + b0 * b1^x)` | `ln(1/y - 1/u)` on `x`; `b0 = exp(a0)`, `b1 = exp(a1)` |

`u` is `upper_bound`; without it `1/u = 0` (SPSS's default).

`r_squared`, the F statistic `((SST - SSE) / df1) / (SSE / df2)` and its
p-value belong to the **transformed** regression. R-squared values of models
with different outcome transformations (for example `linear` and `growth`) are
therefore not on the same scale; compare them with care, or compare fitted
curves. Compound, growth and exponential are the same regression in three
parameterizations and share their R-squared.

With `intercept=False` (SPSS `/NOCONSTANT`) every model is fitted without its
constant: R-squared is measured about the origin, `df2 = n - df1`, and `b0` is
not reported (the multiplicative models then have `b0 = 1`).

**Domain problems skip a model, they do not fail the call.** Models in
`ln(y)` need a positive outcome; `logarithmic` and `power` need a positive
predictor; `inverse` and `s` a nonzero predictor; `logistic` needs
`0 < y < upper_bound`. A skipped model keeps its row in `summary` with missing
statistics, has no column in `fitted`, is listed in `attrs["skipped"]` and
explained in `attrs["notes"]`. The same happens when a polynomial has too few
distinct predictor values or too few observations.

**Computation.** The (transformed) predictor is centred and scaled before
powers are taken, the regression is solved by Householder QR, and the
coefficients of the raw powers are recovered with the binomial theorem. A
predictor such as a calendar year therefore does not make the cubic model
collinear, and the fitted curve is evaluated in the well-conditioned form.

**Returned tables.** `summary` (index = model): `equation`, `r_squared`,
`statistic`, `df1`, `df2`, `p_value`, `b0`, `b1`, `b2`, `b3`. `fitted`: 400
evenly spaced values of `x` between its minimum and maximum and one column per
estimated model with the fitted curve on the scale of `y`, ready for plotting
(printing the result shows only the first and last rows of this table).
`attrs`: `n`, `n_missing`, `fitted_models`, `skipped`, `intercept`,
`upper_bound`, `notes`.

### Example

```python
result = oe.curvefit(hald, "y", "x2",
                     models=["linear", "logarithmic", "quadratic", "power", "growth"])
result["summary"]
```

```
                           equation  r_squared  statistic  df1  df2  p_value       b0       b1      b2
linear                y = b0 + b1*x     0.6663    21.9606    1   11   0.0007  57.4237   0.7891     NaN
logarithmic       y = b0 + b1*ln(x)     0.7008    25.7635    1   11   0.0004 -42.9227  36.2037     NaN
quadratic    y = b0 + b1*x + b2*x^2     0.7215    12.9540    2   10   0.0017  19.2978   2.5574 -0.0185
power                 y = b0 * x^b1     0.7211    28.4355    1   11   0.0002  20.7205   0.3965     NaN
growth           y = exp(b0 + b1*x)     0.6839    23.8038    1   11   0.0005   4.1307   0.0086     NaN
```

```python
curve = result["fitted"]            # 400 rows; columns x2, linear, logarithmic, ...
```

Plot any of its columns against `x2` to draw the fitted curves over a scatter
of the data.

A back-transformed constant that cannot be represented in double precision
(`exp(a)` overflowing, or underflowing below the smallest normal number, which
happens only when the predictor is measured in absurdly small units) is left
missing with a note. The model's R-squared, F test and fitted curve are still
reported.

Equivalent command: `CURVEFIT /VARIABLES=y WITH x2 /CONSTANT /MODEL=LINEAR
LOGARITHMIC QUADRATIC POWER GROWTH`. In Stata each row is a `regress` of the
transformed variables (`regress lny x2` for the growth model).

Limitations: one predictor; no weights; no prediction intervals; the
nonlinear models are estimated on the transformed scale (as in SPSS), not by
nonlinear least squares, so their errors are multiplicative.

---

## `oe.tabstat` — summary statistics by group

```python
oe.tabstat(data, columns, *, by=None, stats=None, weights=None, weight_type=None,
           moments="stata", listwise=False, total=True, anova=False)
```

One row per variable (and group) and one column per statistic. The default
statistics are `n, mean, sd, min, max`.

| request | column | definition |
| --- | --- | --- |
| `n` (`count`) | `n` | observations: rows, or the sum of frequency weights |
| `mean`, `sum` | same | `sum(w x) / W`; `sum(w x)` (Stata's `r(sum)`) |
| `min`, `max`, `range` | same | |
| `sd`, `variance` (`var`) | `std_dev`, `variance` | `variance = (n / W) sum(w (x - mean)^2) / (n - 1)` |
| `semean` (`se`) | `std_error` | `sd / sqrt(n)` |
| `cv` | `cv` | `sd / mean` |
| `skewness`, `kurtosis` | same | see `moments` below |
| `se_skewness`, `se_kurtosis` | same | SPSS: `sqrt(6n(n-1) / ((n-2)(n+1)(n+3)))` and `sqrt(4(n^2-1) se_skew^2 / ((n-3)(n+5)))` |
| `median`, `p1` ... `p99`, `p2.5` | same | Stata percentile (below) |
| `iqr`, `q` | `iqr`; `p25`, `p50`, `p75` | `p75 - p25`; the three quartiles |
| `gmedian` | `grouped_median` | median of grouped data (below) |
| `harmonic` | `harmonic_mean` | `W / sum(w / x)`, when every value is positive |
| `geometric` | `geometric_mean` | `exp(sum(w ln x) / W)`, when every value is positive |

`W = sum w` (`w = 1` without weights).

**Weights** follow Stata's `summarize`. Frequency weights (`fweight`) replicate
rows: `n = W`. Analytic weights (`aweight`, the default) are rescaled to sum to
the number of rows: `n` is the row count and the variance is
`sum(v (x - mean)^2) / (n - 1)` with `v = w n / W`. Every statistic except
`sum` is independent of the scale of the weights. `sum` is the weighted total
`sum(w x)` with the weights as given (Stata's `r(sum) = r(mean) * r(sum_w)`), so
it scales with them. SPSS's `WEIGHT BY` corresponds to `fweight`.

**Percentiles** use Stata's default definition, also with weights: with
`P = W p / 100` and cumulative weights `W_(i)` of the sorted values, the
percentile is `x_(i)` for the first `i` with `W_(i) > P`, or
`(x_(i-1) + x_(i)) / 2` when `W_(i-1) = P` exactly. Unweighted this is the
usual median (mean of the two middle values when `n` is even).

**Moments.** `moments="stata"` (default): `skewness = m3 / m2^1.5`,
`kurtosis = m4 / m2^2` with `m_r = sum(w (x - mean)^r) / W` (a normal variable
has kurtosis 3). `moments="spss"`: the bias-corrected
`G1 = g1 sqrt(n(n-1)) / (n-2)` and the excess
`G2 = (n-1) ((n+1) g2 - 3(n-1)) / ((n-2)(n-3))` that SPSS reports.

**Grouped median** (SPSS `GMEDIAN`). Every distinct value is treated as the
midpoint of a class whose limits lie halfway to the neighbouring distinct
values (for codes 35, 45, 55 the classes are 30-40, 40-50, 50-60). With `L`
and `h` the lower limit and width of the class in which the cumulative weight
reaches `W / 2`, `f` its weight and `F` the cumulative weight below it, the
grouped median is `L + (W/2 - F) / f * h`.

**Groups.** `by="g"` gives one row per variable and group (ascending group
order, categorical columns in category order) and, unless `total=False`, a row
labelled `Total`.

**ANOVA and eta** (`anova=True`, SPSS `MEANS /STATISTICS ANOVA`): the result is
a `TableSet` with `statistics`, `anova` (per variable: `between`, `within`,
`total` with `ss`, `df`, `ms`, `statistic`, `p_value`) and `association`
(`eta`, `eta_squared = SS_between / SS_total`). Between-group degrees of
freedom are the number of non-empty groups minus one; weights enter as
described above.

**Returned table.** Without `by`: index = variable names. With `by`: columns
`variable`, the `by` name, then the statistics. `attrs`: `by`, `groups`,
`stats`, `weights`, `moments`, `missing`, `n`, `notes`.

### Example

```python
hald["plant"] = ["a"] * 6 + ["b"] * 7
oe.tabstat(hald, ["y", "x1"], by="plant", stats=["n", "mean", "sd", "median", "gmedian"])
```

```
  variable  plant   n     mean  std_dev  median  grouped_median
0        y      a   6  91.6333  13.9745   91.75          91.750
1        y      b   7  98.6714  16.2239  102.70         101.975
2        y  Total  13  95.4231  15.0437   95.90          96.900
3       x1      a   6   8.0000   3.9497    9.00           9.000
4       x1      b   7   7.0000   7.4610    3.00           4.500
5       x1  Total  13   7.4615   5.8824    7.00           7.625
```

```python
result = oe.tabstat(hald, ["y"], by="plant", anova=True)
result["anova"]          # between: F = 0.6888, p = 0.4242
result["association"]    # eta = 0.2428, eta squared = 0.0589
oe.tabstat(df, ["income"], by="region", weights="pop", stats=["mean", "median", "p90"])
oe.tabstat(df, ["income"], weights="count", weight_type="fweight", stats=["n", "sd"])
```

Equivalent commands: Stata `tabstat y x1, by(plant) statistics(n mean sd
median)` and `tabstat income [aw=pop], by(region) statistics(mean median p90)`;
SPSS `MEANS TABLES=y x1 BY plant /CELLS=COUNT MEAN STDDEV MEDIAN GMEDIAN
/STATISTICS ANOVA`.

Limitations: one grouping column (no layered `BY`); no test of linearity
(SPSS `/STATISTICS LINEARITY`); `first`, `last` and percent-of-total cells of
SPSS MEANS are not provided; sampling weights (`pweight`) are not accepted
because `tabstat` reports no design-based standard errors.

---

## Performance

All procedures read the rows a bounded number of times and never build an
`n x n` matrix or loop over observations. Timings on a 12-core laptop
(float64, CPU, six Torch threads):

| task | time |
| --- | --- |
| `stepwise`, 1,000,000 rows, 200 candidates (stepwise, forward or BIC) | 1.1 s |
| the same, backward elimination | 1.3 s |
| the same with a 5-level categorical term and analytic weights | 1.6 s |
| `stepwise`, 100,000 rows (1,000,000 rows), 10 candidates | 0.03 s (0.2 s) |
| `collin`, 1,000,000 rows, 10 regressors (100 regressors) | 0.1 s (1.2 s) |
| `curvefit`, 1,000,000 rows, all eleven models | 0.1 s |
| `tabstat`, 1,000,000 rows, 10 variables by 50 groups, default statistics | 0.3 s |
| the same with analytic weights, medians, percentiles, grouped median, moments and ANOVA | 2.0 s (0.2 s on 100,000 rows) |

In `stepwise` almost all of the time is the pass that forms the cross-product
matrix plus the `O(n k)` residual pass of the final model. The search itself
(about 160 steps over 200 candidates) takes roughly 0.05 s whatever the number
of rows. All timings grow linearly with the number of rows.

## Conventions that are choices

Parity with SPSS or Stata output has not been measured for this family: the
test suite checks every number against independent NumPy / SciPy / statsmodels
computations and against published textbook examples (Hald's cement data,
the Longley data). Where the packages differ or their documentation is not
explicit, OpenEconometrics does the following.

- **Entry and removal rules.** A term enters when `p < p_enter` and leaves when
  `p > p_remove`, the strict inequalities of the SPSS Statistics Algorithms
  description of REGRESSION. SPSS's output footnote prints them as
  "Probability-of-F-to-enter <= .050, Probability-of-F-to-remove >= .100", and
  Stata enters when `p < pe()` and removes when `p >= pr()`. These variants
  differ only when a p-value equals a threshold exactly.
- **Order of moves in `method="stepwise"`.** A removal is tried before every
  entry (SPSS). Stata's forward stepwise does the same except immediately after
  a removal, where it tries an entry first; the paths can differ in rare cases.
- **Equal thresholds.** `p_enter == p_remove` is accepted (as in SAS and
  Minitab); SPSS and Stata ask for a strictly smaller entry level. A step limit
  of twice the number of terms protects against cycling and is reported in the
  notes if it is ever reached.
- **Minimum tolerance.** SPSS's two tolerance tests are applied in every
  method; Stata relies on `regress` dropping exactly collinear terms. The
  reported `min_tolerance` of an excluded term is the smallest tolerance of
  *all* columns of the model it would create, the candidate included. Whether
  SPSS's "Minimum Tolerance" includes the candidate itself was not verified;
  the two readings coincide whenever the candidate's own tolerance is not the
  smallest.
- **Categorical terms** are tested as blocks (Stata). SPSS REGRESSION has no
  such terms.
- **`criterion="aic"` / `"bic"`** is the greedy search of R's `step()`; neither
  SPSS nor Stata offers it in these commands. The reported values use Stata's
  `estat ic` definition (the error variance is not counted as a parameter).
- **Analytic weights** are rescaled to sum to `n` (Stata). SPSS's `REGWGT`
  reports sums of squares and a standard error of the estimate on the scale of
  the raw weights; coefficients, standard errors and tests are unaffected.
- **`tabstat` `sum` with analytic weights** is `sum(w x)` with the weights as
  given. This follows our reading of Stata's `summarize`, where
  `r(sum) = r(mean) * r(sum_w)` and `r(sum_w)` is the raw sum of weights. It was
  not checked against Stata output. Every other `tabstat` statistic uses
  Stata's normalized weights and is independent of their scale.
- **Grouped median.** The textbook class-interval interpolation described
  above (class limits halfway between neighbouring distinct values). SPSS's
  own interpolation rule for `GMEDIAN` was not verified against SPSS output
  and may differ, in particular for unequally spaced values.
- **Harmonic and geometric means** are reported only when every value is
  positive.
- **SPSS moments with analytic weights** use the row count as `n`; SPSS itself
  has only frequency-type weights.
- **Standard errors of skewness and kurtosis** are SPSS's formulas in both
  `moments` settings; Stata reports none.
- **`curvefit` without a constant** reports no `b0` for the multiplicative
  models (the fitted curve uses `b0 = 1`) and measures R-squared about the
  origin.
- **`collin(intercept=False)`** reports uncentred VIFs, as Stata's
  `estat vif, uncentered`.
