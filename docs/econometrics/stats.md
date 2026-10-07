# Classical parametric statistics

t tests, tests of variances, one-way and factorial analysis of variance and
covariance with post hoc comparisons, repeated measures, MANOVA, correlations
and descriptive statistics: `ttest`, `sdtest`, `oneway`, `anova`, `rm_anova`,
`manova`, `correlate`, `pcorr` and `describe`. They cover SPSS's T-TEST, ONEWAY,
UNIANOVA, GLM, CORRELATIONS, PARTIAL CORR, DESCRIPTIVES and EXAMINE, and
Stata's `ttest`, `sdtest`, `robvar`, `oneway`, `pwmean`, `anova`, `manova`,
`correlate`, `pwcorr`, `spearman`, `ktau`, `pcorr`, `summarize` and `tabstat`.

Everything on this page is computed in OpenEconometrics on float64 PyTorch tensors:
group statistics are O(n) scatter sums, ranks and percentiles come from one
sort, linear models are solved by Householder QR, and the studentized range and
Dunnett distributions are evaluated by quadrature written for this package. No
statistics library runs at fit time; SciPy and statsmodels appear only in the
test suite as independent oracles.

These procedures are tests and summaries, not model fits. They register no
estimator and return **result tables** (`openecon.frame.DataFrame`, printable
and exportable with `.to_latex()`) or a `TableSet`, a dictionary of named
tables with `str()` and `.to_latex()` for the whole set. Scalar results are in
`.attrs`.

```python
import openecon as oe

result = oe.oneway(df, "wage", "region", posthoc=["tukey"])
print(result)                          # every table
result["anova"]                        # one table
result.attrs["statistic"], result.attrs["p_value"]
result["posthoc_tukey"].to_latex()
```

| SPSS / Stata | OpenEconometrics |
| --- | --- |
| `T-TEST /TESTVAL=5 /VARIABLES=y`; `ttest y == 5` | `oe.ttest(df, "y", mu=5)` |
| `T-TEST PAIRS=y WITH x (PAIRED)`; `ttest y == x` | `oe.ttest(df, "y", paired_with="x")` |
| `T-TEST GROUPS=g(0 1) /VARIABLES=y`; `ttest y, by(g)` [`unequal`] | `oe.ttest(df, "y", by="g")` (both rows) |
| `sdtest y, by(g)`; `robvar y, by(g)`; `sdtest y == 2` | `oe.sdtest(df, "y", by="g")`; `oe.sdtest(df, "y", sd=2)` |
| `ONEWAY y BY g /STATISTICS ... /POSTHOC=TUKEY GH`; `oneway y g, bonferroni` | `oe.oneway(df, "y", "g", posthoc=["tukey", "games_howell"])` |
| `UNIANOVA y BY a b WITH x`; `anova y a##b c.x` | `oe.anova(df, "y", ["a", "b"], covariates=["x"])` |
| `/METHOD=SSTYPE(1)`; `anova ..., sequential` | `ss_type=1` |
| `/EMMEANS=TABLES(a) COMPARE ADJ(SIDAK)`; `margins a, asbalanced` + `pwcompare` | `emmeans=["a"], adjust="sidak"` |
| `GLM t1 t2 t3 BY g /WSFACTOR=time 3`; `anova y g / id\|g time g#time, repeated(time)` | `oe.rm_anova(long, "y", "id", ["time"], between=["g"])` |
| `GLM y1 y2 BY a b WITH x`; `manova y1 y2 = a##b c.x` | `oe.manova(df, ["y1", "y2"], ["a", "b"], covariates=["x"])` |
| `CORRELATIONS`; `pwcorr x y z, sig obs` / `correlate` | `oe.correlate(df, ["x", "y", "z"])` / `pairwise=False` |
| `NONPAR CORR /PRINT=SPEARMAN` / `KENDALL`; `spearman`, `ktau` | `method="spearman"` / `"kendall"` |
| `PARTIAL CORR y x BY w`; `pcorr y x w` | `oe.pcorr(df, "y", ["x"], controls=["w"])`; `oe.pcorr(df, "y", ["x", "w"])` |
| `DESCRIPTIVES`, `EXAMINE`; `summarize, detail`, `tabstat ..., by(g)` | `oe.describe(df, ["x", "y"], by="g")` |

## Samples, missing data and failures

Inputs are a DataFrame, a mapping of columns or a list of row records. Columns
that must be numbers are checked before anything else (`non_numeric_column`);
infinite values are refused (`non_finite_values`). Grouping columns may hold
any scalar values; groups and factor levels are ordered by ascending value
(categorical columns by the order of their categories; unused categories are
ignored).

Sums of squares must be representable: a column with a value above `1e150` in
magnitude, or whose values are all below `1e-100` in magnitude (and not all
zero), is refused with `non_finite_values` and the advice to rescale it. Inside
that range no result depends on the unit of measurement: higher moments are
computed from standardized deviations and products of sums of squares are never
formed where they could overflow.

| procedure | missing values |
| --- | --- |
| `ttest`, `sdtest`, `oneway`, `anova`, `manova`, `pcorr` | **listwise**: rows with a missing value in any column used are excluded (`missing="drop"`, the default) or rejected (`missing="raise"`, error `missing_values`). The number excluded is `attrs["n_missing"]`. |
| `rm_anova` | rows with missing values are excluded (`missing="drop"`) or rejected (`missing="raise"`); the design must then still be complete, otherwise `unbalanced_design` |
| `correlate` | **pairwise** by default (each pair uses the rows where both are observed; ranks are recomputed per pair); `pairwise=False` is listwise |
| `describe` | **variable-wise** by default (each variable uses its own non-missing values); `listwise=True` |

Every failure is an `openecon.AnalysisError` with a `code` and a message that
says what to change, for example `invalid_groups` (wrong number of groups),
`insufficient_observations`, `zero_variance` (the statistic is undefined
because nothing varies), `empty_cells`, `collinear_design`, `no_residual_df`,
`perfect_fit`, `single_level`, `unbalanced_design`, `singular_error_matrix`,
`invalid_option`, `invalid_spec`. A statistic that is undefined for part of a
table (the standard deviation of a single observation, a confidence interval
that a method does not define) is a missing cell, never `inf` or text; when a
whole block is undefined the reason is listed in `attrs["notes"]`.

Location-invariant statistics are computed on outcomes minus a common constant,
so data such as `1e9 + noise` lose no digits; covariates of `anova` and
`manova` are centred internally for the same reason (see below).

A column name that would be ambiguous in a result is refused with
`invalid_spec`: a factor or covariate called `Intercept`, `corrected_model`,
`error`, `total` or `corrected_total` (rows of the ANOVA table), two terms with
the same name (a column literally called `a#b` next to the interaction of `a`
and `b`), or a grouping column named like a statistic of the table it labels
(`describe(..., by="mean")`).

## `ttest`

```python
oe.ttest(data, y, *, by=None, paired_with=None, mu=0.0, alpha=0.05, missing="drop")
```

* **One sample**: `t = (ybar - mu) / (s / sqrt(n))`, `n - 1` degrees of freedom.
* **Paired** (`paired_with`): the one-sample test of `d = y - paired_with`
  against `mu` on the rows where both are observed. `attrs["correlation"]` and
  `attrs["correlation_p_value"]` give SPSS's Paired Samples Correlations.
* **Independent** (`by` with exactly two values; difference = first - second in
  ascending order of the group value): the table `test` has both rows SPSS prints,
  * `equal_variances`: `se = s_p sqrt(1/n1 + 1/n2)`,
    `s_p^2 = [(n1-1) s1^2 + (n2-1) s2^2] / (n1+n2-2)`, `df = n1+n2-2` (Stata's default);
  * `unequal_variances`: Welch, `se = sqrt(s1^2/n1 + s2^2/n2)` with Satterthwaite's
    `df = (s1^2/n1 + s2^2/n2)^2 / [(s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1)]`
    (Stata `unequal`; Stata's `welch` option uses a slightly different df formula
    and is not reported),

  and `levene` is Levene's test based on the group means, as in SPSS.

Tables: `statistics` (n, mean, std_dev, std_error, ci_low, ci_high of each mean;
for independent samples the two groups and `combined`), `test` (statistic, df,
p_value two-sided, p_less = P(T < t), p_greater = P(T > t), mean_difference,
std_error, ci_low, ci_high), `levene`, `effect_sizes` (standardizer and estimate).
`mean_difference` and its interval are measured from `mu`. The variance is
computed about the sample mean and the tested difference as
`(mean - c) + (c - mu)` with `c` the rounded mean, so the test is exact both for
`mu` next to a large common level and for `mu` far from the data.

Effect sizes (SPSS 27+): Cohen's d divides the mean difference by the pooled
standard deviation (independent), the standard deviation of the differences
(paired) or the sample standard deviation (one sample). Hedges' g multiplies d
by `J(df) = Gamma(df/2) / (sqrt(df/2) Gamma((df-1)/2))`. Glass's delta uses the
standard deviation of the second group (`attrs["glass_delta_first"]` uses the
first, Stata's `esize` reports both). Confidence intervals of the effect sizes
(noncentral t) are not computed. `attrs["one_sided_p_value"]` is the smaller
tail, SPSS's "One-Sided p".

```python
r = oe.ttest(df, "score", by="treated")
r["test"].loc["unequal_variances", ["statistic", "df", "p_value"]]
r.attrs["cohens_d"], r.attrs["levene_p_value"]
```

## `sdtest`

```python
oe.sdtest(data, y, *, by=None, sd=None, center="mean", alpha=0.05, missing="drop")
```

With `by`: `variance_ratio` (two groups only) is `F = s1^2 / s2^2` with
`(n1-1, n2-1)` df, `p_value = 2 min(P(F<f), P(F>f))` (Stata `sdtest`), and
`robust` holds Levene's `W0` (deviations from the group mean; SPSS's Levene
test) and the Brown-Forsythe variants `W50` (median) and `W10` (10% trimmed
mean), each the one-way ANOVA F of `z_ij = |y_ij - center_i|` with
`(k-1, N-k)` df (Stata `robvar`). `center` picks the headline test in `attrs`.
Every observation enters each statistic; only the centre changes (SciPy's
`levene(center="trimmed")` instead drops the trimmed observations).
Without `by`, `sd` is required: `chi2 = (n-1) s^2 / sd^2` with `n-1` df and
`p_value = 2 min(P(chi2 < c), P(chi2 > c))` (Stata `sdtest y == #`).

## `oneway`

```python
oe.oneway(data, y, by, *, posthoc=None, alpha=0.05, welch=True, control=None, missing="drop")
```

| table | content |
| --- | --- |
| `descriptives` | n, mean, std_dev, std_error, ci_low, ci_high, min, max per group and `Total` |
| `anova` | `between`, `within`, `total`: ss, df, ms, statistic (F), p_value |
| `homogeneity` | `levene_mean`, `levene_median`, `levene_median_adjusted_df`, `bartlett` (chi-square; df in `df1`) |
| `robust` | `welch` and `brown_forsythe`: statistic, df1, df2, p_value |
| `effect_sizes` | `eta_squared`, `epsilon_squared`, `omega_squared` (fixed effect) |
| `posthoc_<method>` | group_i, group_j, mean_difference (i - j), std_error, statistic (t), df, p_value, ci_low, ci_high |

Formulas. `F = MS_between / MS_within` with `(k-1, N-k)` df.
Welch: `w_i = n_i/s_i^2`, `F = [sum w_i (ybar_i - ybar_w)^2/(k-1)] / [1 + 2(k-2) L/(k^2-1)]`,
`L = sum (1-w_i/W)^2/(n_i-1)`, `df2 = (k^2-1)/(3L)`. Brown-Forsythe:
`F* = SS_between / sum (1-n_i/N) s_i^2` with Satterthwaite `df2`. Bartlett:
`chi2 = [(N-k) ln s_p^2 - sum (n_i-1) ln s_i^2] / C`. Effect sizes:
`eta^2 = SS_b/SS_t`, `epsilon^2 = (SS_b - (k-1) MS_w)/SS_t`,
`omega^2 = (SS_b - (k-1) MS_w)/(SS_t + MS_w)`; negative estimates are kept.

Post hoc methods (`posthoc=[...]`), all on `t_ij = (ybar_i - ybar_j) / se_ij`:

| method | standard error and df | p-value and interval |
| --- | --- | --- |
| `lsd` | pooled MS_within, N - k | Student t, unadjusted |
| `bonferroni` | same | `min(1, m p)`, `m = k(k-1)/2`; interval at `alpha/m` |
| `sidak` | same | `1 - (1-p)^m`; interval at `1 - (1-alpha)^(1/m)` |
| `holm` | same | step-down Bonferroni; no simultaneous interval (missing) |
| `scheffe` | same | `F = t^2/(k-1)` on `(k-1, N-k)`; half-width `sqrt((k-1) F_alpha) se` |
| `tukey` | same | studentized range, `q = abs(t) sqrt(2)`, k groups, N - k df (Tukey-Kramer for unequal n) |
| `games_howell` | `sqrt(s_i^2/n_i + s_j^2/n_j)`, Welch df per pair | studentized range with k groups and the pair's df |
| `dunnett` | pooled, N - k | each group against `control` (default: first group; SPSS's default is the last), two-sided, exact many-one t distribution |

### Studentized range and Dunnett distributions

`openecon.econometrics.stats.srange` provides `ptukey(q, k, df)`,
`ptukey_sf`, `qtukey(p, k, df)`, `qtukey_upper(p, k, df)` (the quantile from the
upper tail probability), `pdunnett_sf(d, lambdas, df)`, `qdunnett(p, lambdas, df)`
and `qdunnett_upper`, vectorized over `(q, df)` pairs. Critical values of the
post hoc intervals come from the `_upper` forms, so a very small `alpha` is not
rounded away in `1 - alpha`. Both distributions
are scale mixtures `P(Q > q) = int f_nu(s) T(q s) ds` over `s = sqrt(chi2_nu/nu)`.
The infinite-df tail `T` is tabulated once per `k` (or per set of correlations)
by Gauss-Legendre quadrature in a cancellation-free form and interpolated at
degree 15; the outer integral uses 70 Gauss-Legendre panels placed where the chi
density lives. Accuracy against `scipy.stats.studentized_range` is about 1e-11
for 2 to 100 groups and df from 2 (fractional df from 1) to infinity; far tails
keep their relative accuracy. (For df of 1e5 and more SciPy returns the
infinite-df value, an error near 1e-5; an independent quadrature agrees with
OpenEconometrics there.) For Dunnett, `lambda_i = sqrt(n_i/(n_i + n_0))` so
that `corr(T_i, T_j) = lambda_i lambda_j` (Dunnett 1955); designs where a group
is several hundred times larger than the control are refused
(`dunnett_unbalanced`).

## `anova`

```python
oe.anova(data, y, factors, *, covariates=None, interactions="full", ss_type=3,
         within=None, subject=None, emmeans=None, adjust="bonferroni",
         homogeneity_of_slopes=False, alpha=0.05, missing="drop")
```

The general linear model with **sum-to-zero (effect) coding** built internally:
a factor with L levels has L - 1 columns (level l is the unit vector e_l, the
last level is -1 in every column), interaction columns are products of factor
columns and covariates enter linearly. `interactions` is `"full"` (all
interactions among the factors), `"none"`, or a list such as `[["a", "b"]]` or
`["a#b"]` (also `"a*b"`, `"a:b"`); a covariate may be part of an interaction, and
`homogeneity_of_slopes=True` adds every factor-by-covariate term.

**Marginality.** An interaction is accepted only together with the effects it
contains: `a#b#c` needs `a#b`, `a#c` and `b#c`; `a#b#x` needs `a#x` and `b#x`.
SPSS, SAS and Stata build an interaction from the indicator columns of its
cells, which span the interaction and everything contained in it, whereas
products of sum-to-zero columns span the interaction contrasts only. The two
are the same model exactly when the contained effects are present, so a list
that omits them raises `invalid_spec` (naming the missing terms) instead of
fitting a model with another meaning. For the same reason `ss_type=1` requires
an interaction to be listed after the effects it contains.

**Covariates are centred internally.** When the model is hierarchical in its
covariates (always, unless a product such as `a#x#w` is listed without `a#x`
and `a#w`) the design uses `x - mean(x)`. The raw coefficients and
`(X'X)^{-1}` follow exactly from `b_raw = T^{-1} b_c`, with `T` a sparse matrix
of products of the means, and every hypothesis is still the one on the raw
coefficients: the intercept, and a factor in a model with factor-by-covariate
slopes, are tested at covariate = 0 as in SPSS and Stata. The benefit is purely
numerical: a covariate such as `1e8 + noise` loses no digits and is not
mistaken for a constant.

One blocked QR reduces `[X, y]` to its triangular factor; every hypothesis is
then a small QR problem. The sum of squares of a term T inside a model M is the
Wald form `b_T' [(X_M'X_M)^{-1}_TT]^{-1} b_T`, equal to the reduction in the
residual sum of squares when T is dropped from M:

| `ss_type` | M | meaning |
| --- | --- | --- |
| 1 | terms up to and including T | sequential; order: covariates, main effects as listed, interactions (lowest order first, or as listed), slopes |
| 2 | every term that does not contain T | T adjusted for all effects that do not contain it |
| 3 | the full model | T adjusted for everything (SPSS and SAS default; Stata's partial SS) |

"U contains T" follows SPSS and SAS: both have the same covariates and the
factors of T are a proper subset of those of U. A factor `a` is therefore not
contained in the slope term `a#x`, and its Type II sum of squares is adjusted
for `a#x`; R's `car::Anova` and statsmodels treat `a:x` as containing `a` and
differ in that one case. With sum-to-zero coding and every cell of
each interaction observed, Type III tests equal-weighted marginal means and is
valid for unbalanced data. An unobserved cell of an interaction raises
`empty_cells` (Type IV is not implemented); a main-effects model
(`interactions="none"`) tolerates empty cells.

The `anova` table has the rows of SPSS's Tests of Between-Subjects Effects:
`corrected_model`, `Intercept`, one row per term (`a`, `a#b`, `x`, `a#x`),
`error`, `total` (uncorrected, df N) and `corrected_total`, with ss, df, ms,
statistic (F against MS_error), p_value and partial_eta_squared
= SS / (SS + SS_error). `attrs`: `r_squared`, `adjusted_r_squared`, `rmse`,
`df_model`, `df_resid`, `n`, `n_missing`, `ss_type`, `terms`, `levels`.

**Estimated marginal means** (`emmeans=["a", ["a", "b"]]`): predictions
averaged with equal weights over the levels of the other factors, covariates at
their sample means, with standard error `sqrt(MS_error L (X'X)^{-1} L')`, error
df and t intervals (`emmeans_<factor>`). Single factors also get
`pairwise_<factor>` with `adjust="bonferroni"` (default), `"sidak"` or `"lsd"`
(no adjustment, SPSS's default) applied to p-values and intervals.

```python
r = oe.anova(df, "wage", ["sector", "gender"], covariates=["tenure"], emmeans=["sector"])
r["anova"]; r["emmeans_sector"]; r["pairwise_sector"]; r.attrs["r_squared"]
```

Passing `within=` and `subject=` hands the call to `rm_anova` with `factors` as
between-subject factors. Options that the repeated-measures analysis does not
have (`covariates`, `emmeans`, `homogeneity_of_slopes`, `ss_type` other than 3,
`interactions` other than `"full"`) are rejected rather than ignored.

## `rm_anova`

```python
oe.rm_anova(data, y, subject, within, *, between=None, alpha=0.05, missing="drop")
```

Univariate repeated-measures ANOVA on **long** data (one row per subject and
within-subject cell) for one or more within-subject factors and optional
between-subject factors (split-plot). The data are pivoted to the subjects by
cells matrix Y; for each within effect W an orthonormal contrast matrix M_W
gives `T_W = Y M_W`; the between-subject design (sum-to-zero, Type III) is
fitted to all transformed variables; then `SS(W x g) = trace H_g(T_W)` with
`d df_g` df and `SS(error W) = trace E(T_W)` with `d (S - rank X)` df, where d
is the number of contrasts of W and S the number of subjects. The scaled
subject means give the `between` table. Group sizes may differ.

For an effect with `d >= 2`, with `l` the eigenvalues of `E(T_W)` and
`v = S - rank X`:

* Mauchly: `W = det(E) / (trace(E)/d)^d`, `chi2 = -(v - (2d^2+d+2)/(6d)) ln W`,
  `df = d(d+1)/2 - 1`;
* Greenhouse-Geisser `eps = (sum l)^2 / (d sum l^2)`; Huynh-Feldt
  `eps = min(1, (S d eps_GG - 2) / (d (v - d eps_GG)))`; lower bound `1/d`.

Tables: `within` (long: source, correction in `sphericity_assumed`,
`greenhouse_geisser`, `huynh_feldt`, `lower_bound`; ss, df, ms, statistic,
p_value, partial_eta_squared; error terms are the sources `error(W)`),
`sphericity`, `between`, `descriptives` (cell n, mean, std_dev).

The design must be complete: every subject exactly once in every cell, and
between-subject factors constant within a subject. Otherwise
`unbalanced_design` (use a mixed model) or `invalid_design`.

Degenerate error terms. An outcome that does not vary raises `zero_variance`.
When one error term is zero the rest of the analysis is still reported: an
effect that is identical for every subject has missing F, p-value, Mauchly and
epsilon cells, and scores with the same mean for every subject (ranks or
shares within a subject) have a missing between-subject F; each case adds a
line to `attrs["notes"]`. Sums of squares at the rounding level of the total
(below `1e-24` of it) count as zero, so rounding noise is never tested.

## `manova`

```python
oe.manova(data, y, factors, *, covariates=None, interactions="full", alpha=0.05, missing="drop")
```

The model of `anova` fitted to several outcomes. For every term (and the
intercept) the Type III hypothesis SSCP matrix H and the residual SSCP matrix E
give the eigenvalues `l_1 >= ... >= l_s` of `E^{-1}H`. With p outcomes, q
hypothesis df, v error df, `s = min(p, q)`, `m = (|p-q|-1)/2`, `n = (v-p-1)/2`:

| test | value | F and df |
| --- | --- | --- |
| `pillai` | `sum l/(1+l)` | `(2n+s+1)/(2m+s+1) V/(s-V)`; `s(2m+s+1)`, `s(2n+s+1)` |
| `wilks` | `prod 1/(1+l)` | Rao: `(1-L^(1/t))/L^(1/t) df2/df1`, `t = sqrt((p^2q^2-4)/(p^2+q^2-5))`, `df1 = pq`, `df2 = (v-(p-q+1)/2) t - (pq-2)/2` |
| `hotelling` | `sum l` | `2(sn+1) U / (s^2(2m+s+1))`; `s(2m+s+1)`, `2(sn+1)` |
| `roy` | `l_1` | `l_1 (v-r+q)/r`, `r = max(p,q)`; `r`, `v-r+q` (upper bound) |

`f_type` says whether the F is `exact` (Wilks for `s <= 2`; all four for
`s = 1`), `approximate` or an `upper_bound` (Roy). Multivariate
partial_eta_squared follows SPSS: `V/s`, `1 - L^(1/s)`, `(U/s)/(1+U/s)`,
`l_1/(1+l_1)`. The `univariate` table stacks the Type III ANOVA table of each
outcome. `box_m` is Box's test of equal covariance matrices across the cells of
the factors, `M = (N-g) ln|S| - sum (n_i-1) ln|S_i|`, with its chi-square and F
approximations (Box 1949); it is omitted (with a note) when a cell has no more
observations than outcomes. It is computed from the outcomes within the cells
of the factors; covariates do not enter it.

Note that statsmodels and SAS (with `n > 0`) use McKeon's F approximation for
the Hotelling-Lawley trace; OpenEconometrics reports the approximation printed by SPSS
and Stata, so that F and its denominator df differ from statsmodels when
`s > 1` while the trace itself is identical.

## `correlate` and `pcorr`

```python
oe.correlate(data, columns, *, method="pearson", pairwise=True, ci=False, alpha=0.05)
oe.pcorr(data, y, x, *, controls=None, missing="drop")
```

`correlate` returns the square tables `coefficients`, `p_values` and `n`.

* `pearson`: `t = r sqrt((n-2)/(1-r^2))` with `n-2` df.
* `spearman`: Pearson's r of midranks (ties share the mean rank), same t test.
* `kendall`: tau-b, `S / sqrt((n0-n1)(n0-n2))`, with the tie-corrected variance
  of S and `z = S / sqrt(var S)`, no continuity correction (SPSS; Stata's `ktau`
  applies one and its p-value differs slightly). Discordant pairs are counted
  as inversions by a merge sort made of vectorized sorts, O(n log^2 n).

`ci=True` adds `intervals` (var_i, var_j, coefficient, ci_low, ci_high, n):
Fisher-z limits `tanh(atanh(r) -+ z se)` with `se^2 = 1/(n-3)` (Pearson),
`1.06/(n-3)` (Spearman) and `0.437/(n-4)` (Kendall), the Fieller-Hartley-Pearson
variances. A coefficient involving a variable that does not vary is missing.

`pcorr` regresses y on a constant, all of `x` and all of `controls` and reports,
for each variable in `x`, `partial_corr = t / sqrt(t^2 + df)`,
`semipartial_corr = sign(t) sqrt(t^2 (1-R^2)/df)`, their squares, t, df and the
p-value of t (Stata `pcorr`). `controls` are held constant but not reported, so
`pcorr(df, "y", ["x"], controls=["w1", "w2"])` is SPSS's `PARTIAL CORR y x BY w1 w2`.

## `describe`

```python
oe.describe(data, columns=None, *, by=None, stats=None, percentile_method="stata",
            moments="spss", listwise=False, alpha=0.05)
```

One row per variable (index = variable name) or per variable and group (columns
`variable` and the `by` name). `stats` chooses the columns, in order: `n`,
`missing`, `mean`, `std_dev` (`sd`), `std_error` (`se`), `variance`, `cv`, `sum`,
`min`, `max`, `range`, `ci` (ci_low, ci_high of the mean, Student t), `median`,
`iqr`, any percentile `p1`..`p99` (also `p2.5`), `skewness`, `se_skewness`,
`kurtosis`, `se_kurtosis`. Default: n, mean, std_dev, std_error, ci, min, p25,
p50, p75, max, skewness, kurtosis.

* `percentile_method="stata"` (default; Stata `summarize, detail`): with
  `P = n p/100`, the mean of `x_(P)` and `x_(P+1)` when P is an integer,
  otherwise `x_(floor(P)+1)`. `"haverage"` is SPSS's default (and Stata's
  `altdef`): the weighted average at position `(n+1) p/100`.
* `moments="spss"` (default): `G1 = n M3 / ((n-1)(n-2) s^3)` and excess
  `G2 = n(n+1) M4 / ((n-1)(n-2)(n-3) s^4) - 3(n-1)^2/((n-2)(n-3))` with standard
  errors `sqrt(6n(n-1)/((n-2)(n+1)(n+3)))` and
  `sqrt(4(n^2-1) se_G1^2/((n-3)(n+5)))`. `"stata"`: `m3/m2^1.5` and `m4/m2^2`
  with divisor n (a normal variable has kurtosis 3), no standard errors.

`stats` is a list; the aliases `sd`, `se`, `median`, `count` and `var` are
reported under their column names `std_dev`, `std_error`, `p50`, `n` and
`variance`, and naming the same statistic twice (`["sd", "std_dev"]`) is an
error.

## Worked example

The output below was produced by this code.

```python
import openecon as oe

data = {
    "score": [5.1, 4.9, 6.2, 5.8, 6.9, 7.4, 6.6, 7.1, 8.0, 7.7, 8.4, 9.1],
    "group": ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
    "sex":   ["f", "m", "f", "m", "f", "m", "f", "m", "f", "m", "f", "m"],
    "age":   [31, 45, 28, 52, 39, 41, 33, 47, 36, 50, 29, 44],
}
r = oe.oneway(data, "score", "group", posthoc=["tukey"])
r["anova"]
#               ss  df      ms  statistic  p_value
# between  15.7067   2  7.8533    27.8268   0.0001
# within    2.5400   9  0.2822        NaN      NaN
# total    18.2467  11     NaN        NaN      NaN
r["posthoc_tukey"]
#   group_i group_j  mean_difference  std_error  statistic  df  p_value  ci_low  ci_high
# 0       a       b             -1.5     0.3756    -3.9931   9   0.0079 -2.5488  -0.4512
# 1       a       c             -2.8     0.3756    -7.4538   9   0.0001 -3.8488  -1.7512
# 2       b       c             -1.3     0.3756    -3.4607   9   0.0177 -2.3488  -0.2512
r.attrs["eta_squared"]                       # 0.8608

a = oe.anova(data, "score", ["group", "sex"], covariates=["age"], interactions="none",
             emmeans=["group"])
a["anova"]
#                        ss  df      ms  statistic  p_value  partial_eta_squared
# corrected_model   16.1817   4  4.0454    13.7133   0.0020               0.8868
# Intercept          8.4747   1  8.4747    28.7278   0.0011               0.8041
# age                0.4217   1  0.4217     1.4294   0.2708               0.1696
# group             16.0315   2  8.0158    27.1721   0.0005               0.8859
# sex                0.4637   1  0.4637     1.5718   0.2502               0.1834
# error              2.0650   7  0.2950        NaN      NaN                  NaN
# total            595.1000  12     NaN        NaN      NaN                  NaN
# corrected_total   18.2467  11     NaN        NaN      NaN                  NaN
a["emmeans_group"]
#   group    mean  std_error  df  ci_low  ci_high
# 0     a  5.4708     0.2727   7  4.8261   6.1156
# 1     b  7.0208     0.2721   7  6.3774   7.6643
# 2     c  8.3083     0.2717   7  7.6660   8.9507

long = {"id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], "time": [1, 2, 3] * 5,
        "y": [5.0, 6.1, 7.4, 4.2, 5.9, 6.0, 6.3, 6.8, 8.1, 5.1, 5.2, 7.7, 4.8, 6.4, 6.9]}
m = oe.rm_anova(long, "y", "id", ["time"])
m["within"].head(2)
#   source          correction       ss      df      ms  statistic  p_value  partial_eta_squared
# 0   time  sphericity_assumed  11.4653  2.0000  5.7327    23.6236   0.0004               0.8552
# 1   time  greenhouse_geisser  11.4653  1.1536  9.9389    23.6236   0.0052               0.8552
m["sphericity"]
#       mauchly_w    chi2  df  p_value  epsilon_gg  epsilon_hf  epsilon_lb
# time     0.2663  3.9697   2   0.1374      0.5768      0.6619         0.5
```

## Performance

Timings on a laptop CPU (Apple M-series), one million rows:

| call | seconds |
| --- | --- |
| `ttest` (one sample / paired / independent) | 0.01 / 0.02 / 0.04 |
| `sdtest` with 2 or 8 groups (two sorts) | 0.35 |
| `oneway`, 8 groups, Tukey + Games-Howell + Dunnett + Bonferroni | 0.2 |
| `anova`, 4 x 3 x 2 factorial with a covariate and marginal means (Type I, II or III) | 0.35 |
| `anova`, 10 x 8 factorial with a covariate (81 parameters) | 2 |
| `rm_anova`, 250,000 subjects x 4 occasions, 3 groups | 0.15 |
| `manova`, 3 outcomes, 4 x 3 factorial with a covariate | 0.2 |
| `correlate`, 3 variables: Pearson / Spearman / Kendall | 0.07 / 0.35 / 2.3 |
| `pcorr`, 3 regressors | 0.04 |
| `describe`, 3 variables, default statistics (ungrouped / by 100 groups) | 0.4 / 0.5 |

Everything is linear in the number of rows (sort-based statistics n log n):
100,000 rows take a tenth of these times. The linear models cost O(n k^2) for k
design columns (one blocked QR), so a factorial with hundreds of parameters on a
million rows takes seconds (a 200-level factor: 3 s); `oneway` handles a single
factor with any number of levels in O(n).

All 4,950 pairwise Tukey comparisons of 100 groups take about 0.2 s and the
same number of Games-Howell intervals (one studentized-range quantile per pair)
about 2 s; a one-way ANOVA of 100 groups with all eight post hoc methods takes
about 4 s.

## Limitations

* No weights (frequency or sampling) in this family.
* `anova` / `manova`: Type IV sums of squares, random factors, nested terms and
  custom contrasts are not implemented; covariates enter linearly; interactions
  must respect marginality (the effects they contain must be in the model) and
  designs are limited to 2,000 parameters.
* The order of entry for `ss_type=1` is fixed: covariates, main effects in the
  order of `factors`, interactions. Stata's `anova ..., sequential` with a
  factor entered before a covariate cannot be reproduced.
* `rm_anova`: complete data only, no covariates, univariate approach only (the
  multivariate tests of within-subject effects are not reported).
* `oneway`: SPSS's fourth homogeneity row (Levene based on the 5% trimmed mean)
  and homogeneous-subset tables (SNK, Duncan, REGWQ) are not produced; the
  random-effect omega squared is not reported.
* Dunnett's test is two-sided only. Confidence intervals of effect sizes
  (noncentral t / F) are not computed.
* Kendall's tau has no exact small-sample p-value; tau-c is not reported.
* Values above `1e150` in magnitude, or all below `1e-100`, are refused.
* Results have not been compared with SPSS or Stata output files; agreement is
  established against SciPy, statsmodels and explicit algebra (next section).

## How the results were verified

Besides the implementation's own tests, an independent verification pass
(`tests/test_econ_stats_oracle.py`, `tests/test_econ_stats_oracle_tests.py`,
`tests/test_econ_stats_verify.py`) re-derives every reported number without
using the package's formulas:

* factorial models through the **cell-means** parametrization (hypothesis
  matrices on cell means for Type III, residual-sum-of-squares differences of
  over-parametrized indicator designs for Type I and II, averaged predictions
  for marginal means), including heterogeneous slopes and a covariate with a
  level of 1e8 checked against 60-digit arithmetic;
* MANOVA from cell-means SSCP matrices with the F approximations written from
  the references, Hotelling's T-squared for two groups and the exact Wilks F
  for two outcomes; Box's M from determinants;
* repeated measures from the classical sums-of-squares decomposition (one and
  two within factors, balanced and unbalanced split-plot), Box's epsilon from
  the covariance matrix and Mauchly's W from eigenvalues;
* t tests, variance tests, one-way tables, robust tests and every post hoc
  method against SciPy (`ttest_*`, `levene`, `bartlett`, `f_oneway`,
  `tukey_hsd`, `studentized_range`, `dunnett`) and statsmodels
  (`anova_oneway`, `multipletests`);
* correlations against SciPy pair by pair (ties, pairwise and listwise
  deletion) and brute-force pair counting; partial correlations as correlations
  of residuals; descriptives against NumPy/SciPy and both percentile
  definitions written out for small n;
* the studentized range and Dunnett distributions against SciPy and an
  independent double quadrature (agreement 1e-11 or better);
* invariance to row order, to affine changes of units from 1e-8 to 1e8 (1e-90
  to 1e100 for moments and correlations), and the failure contract on empty,
  constant, collinear, tiny and badly typed inputs.

## Conventions that are not certain

* **W10 trimming** (`sdtest`): the 10% trimmed mean removes `floor(0.05 n_i)`
  observations from each tail of a group and the deviations of all observations
  are taken from it; Stata's `robvar` may count the trimmed observations
  differently for group sizes that are not multiples of 20.
* **Levene with adjusted df** (`oneway`): `df2 = (sum u_i)^2 / sum u_i^2/(n_i-1)`
  with `u_i` the within-group sum of squares of the absolute deviations, the
  Satterthwaite form believed to be SPSS's "Based on Median and with adjusted df".
* **Hedges' g**: the exact gamma-function factor is used; software that applies
  the approximation `1 - 3/(4 df - 1)` differs in the fifth decimal.
* **Glass's delta** uses the second group as the control, as SPSS does for
  independent samples.
* **Huynh-Feldt epsilon** for designs with between-subject groups uses the
  number of subjects S in the numerator (the formula SPSS documents), not
  Lecoutre's corrected `S - g + 1`. With fewer error degrees of freedom than
  contrasts the formula has no meaning and 1 is reported.
* **Type II sum of squares of the intercept** follows the containment rule
  literally (the intercept is contained in every pure-factor effect, so it is
  adjusted for the covariate terms only).
* **Order of entry for Type I**: covariates before factors, as SPSS lists them
  in its default design.
* **Box's M with covariates** is computed from the raw outcomes within the
  factor cells; SPSS may base it on a different residual matrix in MANCOVA.
* **Kendall's tau-b p-value** has no continuity correction (believed to be
  SPSS's convention; Stata's `ktau` applies one).
* **Fisher-z intervals** of Pearson's r use `1/(n-3)` without the bias adjustment
  SPSS offers as an option; those of the rank correlations use the
  Fieller-Hartley-Pearson variances, and SPSS offers other estimators as options.
* **Dunnett control group** defaults to the first group (Stata's base level),
  not SPSS's default last category.
* **Zero error terms in `rm_anova`** are reported as missing cells with a note
  rather than as an error (sums of squares below `1e-24` of the total count as
  zero); what SPSS and Stata print in that case was not checked.

## References

Box (1949) Biometrika 36; Brown and Forsythe (1974) JASA 69 and Technometrics
16; Dunnett (1955) JASA 50; Fieller, Hartley and Pearson (1957) Biometrika 44;
Games and Howell (1976) J. Educ. Statist. 1; Greenhouse and Geisser (1959)
Psychometrika 24; Holm (1979) Scand. J. Statist. 6; Huynh and Feldt (1976) J.
Educ. Statist. 1; Knight (1966) JASA 61; Levene (1960); Mauchly (1940) Ann.
Math. Statist. 11; Rao (1973) Linear Statistical Inference; Tukey (1949);
Welch (1947, 1951) Biometrika 34, 38; IBM SPSS Statistics Algorithms (T-TEST,
ONEWAY, GLM, CORRELATIONS, DESCRIPTIVES); Stata Base Reference Manual (ttest,
sdtest, oneway, anova, manova, correlate, pcorr, summarize).
