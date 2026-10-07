# Nonparametric and categorical-data procedures

Rank tests for independent and related samples, one-sample distribution and
normality tests, cross-tabulation with exact tests and measures of association,
and ROC analysis: the procedures of SPSS `NPAR TESTS`, `CROSSTABS`, `FREQUENCIES`
and `ROC`, and of Stata `ranksum`, `signrank`, `signtest`, `kwallis`, `median`,
`ksmirnov`, `swilk`, `sfrancia`, `sktest`, `runtest`, `bitest`, `prtest`,
`tabulate`, `symmetry`, `roctab` and `roccomp`.

Everything on this page is computed by OpenEconometrics on float64 PyTorch tensors. No
statistics library runs at test time: ranks come from one sort (midranks for
ties), group sums from `index_add_`, exact null distributions from dynamic
programming over the support of the statistic, and reference distributions from
`openecon.engines.distributions` or from the published algorithms cited below.
One million rows take a fraction of a second (see [Performance](#performance)).

These procedures are tests and tables, not model fits. They register no
estimator. A procedure with a single natural table returns that table (an
`openecon.frame.DataFrame`: it prints as a table and exports with
`.to_latex()`); a procedure with several tables returns a `TableSet`, a mapping
from table name to table that prints all of them. In both cases the scalar
results are in `.attrs`:

```python
import openecon as oe

result = oe.ranksum(df, "wage", "union")
print(result)                       # the ranks table, the tests table and the scalars
result["tests"]                     # one table
result.attrs["z"], result.attrs["p_value"]
result.attrs.get("p_exact")         # present when the exact test was computed
print(result.to_latex())
```

| key in `attrs` | meaning |
| --- | --- |
| `statistic`, `df`, `p_value` | the headline statistic and its p-value (the one the source package prints first) |
| `p_exact`, `exact` | the exact p-value and whether it was computed |
| `n`, `n_dropped` | observations used, and rows removed by the missing-data rule |
| `notes` | caveats that apply to this result (small expected counts, conservative p-values, ...) |
| `label` | what the procedure is |

## Contents

| Procedure | Function | Stata | SPSS |
| --- | --- | --- | --- |
| Wilcoxon rank-sum / Mann-Whitney | [`oe.ranksum`](#ranksum) | `ranksum y, by(g)` | `NPAR TESTS /M-W` |
| Kruskal-Wallis, Dunn post hoc | [`oe.kwallis`](#kwallis) | `kwallis y, by(g)` | `NPAR TESTS /K-W` |
| Mood's median test | [`oe.median_test`](#median_test) | `median y, by(g)` | `NPAR TESTS /MEDIAN` |
| Jonckheere-Terpstra | [`oe.jonckheere`](#jonckheere) | `jonter` (community) | `NPAR TESTS /J-T` |
| Wilcoxon signed-rank | [`oe.signrank`](#signrank) | `signrank y = x` | `NPAR TESTS /WILCOXON` |
| Sign test | [`oe.signtest`](#signtest) | `signtest y = x` | `NPAR TESTS /SIGN` |
| McNemar | [`oe.mcnemar`](#mcnemar) | `mcc a b` | `NPAR TESTS /MCNEMAR` |
| Bowker symmetry, marginal homogeneity | [`oe.symmetry`](#symmetry) | `symmetry a b` | `CROSSTABS /STATISTICS=MCNEMAR`, `NPAR TESTS /MH` |
| Friedman, Kendall's W | [`oe.friedman`](#friedman) | `friedman` (community) | `NPAR TESTS /FRIEDMAN /KENDALL` |
| Cochran's Q | [`oe.cochran_q`](#cochran_q) | `cochran` (community) | `NPAR TESTS /COCHRAN` |
| Kolmogorov-Smirnov (one and two samples), Lilliefors | [`oe.ksmirnov`](#ksmirnov) | `ksmirnov` | `NPAR TESTS /K-S`, `EXAMINE` |
| Shapiro-Wilk | [`oe.swilk`](#swilk) | `swilk y` | `EXAMINE /PLOT NPPLOT` |
| Shapiro-Francia | [`oe.sfrancia`](#sfrancia) | `sfrancia y` | |
| Skewness / kurtosis tests, Jarque-Bera | [`oe.sktest`](#sktest) | `sktest y` | |
| Runs test | [`oe.runtest`](#runtest) | `runtest y` | `NPAR TESTS /RUNS` |
| Exact binomial test | [`oe.bitest`](#bitest) | `bitest y == p` | `NPAR TESTS /BINOMIAL` |
| z tests of proportions | [`oe.prtest`](#prtest) | `prtest` | `PROPORTIONS` |
| Chi-square goodness of fit | [`oe.chi2gof`](#chi2gof) | `csgof` (community) | `NPAR TESTS /CHISQUARE` |
| Two-way tables | [`oe.crosstab`](#crosstab) | `tabulate a b, chi2 exact ...`, `cc`, `mhodds`, `kap` | `CROSSTABS` |
| One-way frequencies | [`oe.tabulate`](#tabulate) | `tabulate x` | `FREQUENCIES` |
| ROC curve and area | [`oe.roc`](#roc) | `roctab y score` | `ROC` |
| Comparing ROC areas | [`oe.roccomp`](#roccomp) | `roccomp y s1 s2` | `ROC ANALYSIS ... PAIR` |

Spearman's and Kendall's rank correlations of raw data (Stata `spearman`,
`ktau`) are `oe.correlate(data, columns, method="spearman" | "kendall")` in the
`stats` family; `crosstab` reports Spearman's rho and Kendall's tau for a
table.

## Samples, missing data and failures

* **Data**: a DataFrame, a mapping of columns, or a list of row records.
* **Missing values**: every procedure deletes rows listwise over the columns it
  uses (`missing="drop"`, the default, as SPSS and Stata do) and reports
  `n` and `n_dropped`; `missing="raise"` refuses incomplete rows
  (`missing_values`). `friedman`, `cochran_q` and `roccomp` delete listwise over
  all their columns. `tabulate` is the exception: missing values of the
  tabulated column are counted in a "missing" row.
* **Groups and categories** may be numbers, text, booleans or dates. They are
  sorted ascending; a pandas categorical column keeps its declared order.
  Outcomes that must be numeric raise `non_numeric_column` otherwise.
* **Binary columns** (`bitest`, `prtest`, `roc`, `cochran_q`): the event is the
  larger of the two values (1 for 0/1 data) unless `positive=` names it.
* **Failures** are `openecon.AnalysisError(code, message)`; the message says what
  to change. Codes: `missing_columns`, `missing_values`, `empty_data`,
  `empty_sample`, `duplicate_columns`, `invalid_data`, `invalid_spec` (wrong
  argument type, a column used twice), `invalid_option`, `non_numeric_column`,
  `non_finite_values`, `invalid_groups` (wrong number of groups), `not_binary`,
  `no_variation` (a constant column, all differences zero, an empty side),
  `sample_size`, `too_few_observations`, `invalid_values`, `invalid_weights`,
  `too_many_categories`, `too_many_groups`, `exact_unavailable` (an exact
  distribution too large to enumerate: use `exact=False`), `numerical_failure`
  (values beyond 1e150 in magnitude: rescale the column; every procedure here
  is invariant to the unit of measurement).
* Cells of a result table that do not apply are empty (NaN); `attrs` never
  contain NaN or infinity.

## Exact p-values

`exact=None` (the default) follows a documented size rule, `exact=True` forces
the exact computation and `exact=False` suppresses it. The asymptotic p-value
is always reported next to the exact one.

| Test | Exact distribution | Default rule | Method |
| --- | --- | --- | --- |
| `ranksum` | permutation distribution of the rank sum | n1 n2 <= 400 and no ties | subset-sum recursion over the scores |
| `signrank` | sign-flip distribution of T+ | at most 25 nonzero differences, no ties | coefficients of prod (1 + q^r) |
| `signtest`, `mcnemar`, `bitest` | binomial | always | incomplete beta |
| `ksmirnov` one sample | Kolmogorov's D_n | n <= 1000, continuous F with given parameters | Marsaglia-Tsang-Wang matrix; Birnbaum-Tingey tail |
| `ksmirnov` two samples | Smirnov's D_{m,n} | n1 n2 <= 10,000 and no ties | lattice-path count |
| `runtest` | number of runs given n1, n2 | N <= 50 | closed form |
| `crosstab` 2x2 | Fisher (hypergeometric) | always (integer counts) | ratios of successive probabilities, normalized |
| `crosstab` r x c | Fisher-Freeman-Halton | only with `exact=True` | enumeration, else Monte Carlo |

The hypergeometric distribution of a 2x2 table is built from the ratios
P(a + 1) / P(a), so no factorial of the table total is needed. For
frequency-weighted tables whose support exceeds 4,000,000 points only the
window of 50 standard deviations + 300 around the mean is tabulated (the
probability outside is below 1e-300); beyond totals of roughly 1e10 the Fisher
rows are omitted with a note, and `exact=True` raises `exact_unavailable`.

With ties, `ranksum(exact=True)` and `signrank(exact=True)` use the exact
*conditional* distribution given the observed midranks (doubled midranks are
integers, so the same recursion applies). SPSS instead prints an exact p "not
corrected for ties". Two-sided exact p-values of rank statistics are
P(|T - E(T)| >= |t - E(T)|); without ties the distributions are symmetric and
this equals twice the smaller tail.

## Independent samples

### ranksum

Wilcoxon rank-sum (Mann-Whitney U) test of H0: two groups have the same
distribution.

```python
oe.ranksum(data, y, by, *, exact=None, alpha=0.05, missing="drop")
```

The pooled values are ranked. With T1 the rank sum of the first group (levels
of `by` in sorted order), n1, n2 the group sizes, N = n1 + n2 and t_j the sizes
of the tie groups:

```
U1 = T1 - n1 (n1 + 1) / 2              pairs with y1 > y2, ties counted 1/2
E(T1) = n1 (N + 1) / 2
Var(T1) = n1 n2 / 12 * [ (N + 1) - sum_j (t_j^3 - t_j) / (N (N - 1)) ]
z = (T1 - E(T1)) / sqrt(Var(T1))
```

* `z`, `p_value`: no continuity correction, as SPSS and Stata print.
* `z_continuity`, `p_value_continuity`: |T1 - E| reduced by 1/2 (R, SciPy).
* `u1`, `u2`, `u` = min(U1, U2) (SPSS "Mann-Whitney U"), `w` (SPSS "Wilcoxon W":
  the rank sum of the smaller group, of the first group when sizes are equal).
* `p_exact`, `p_exact_lower`, `p_exact_upper`: exact two-sided p and the tails
  P(T1 <= t1), P(T1 >= t1).
* `effect_r` = z / sqrt(N); `porder` = U1 / (n1 n2), the probability that a
  value of the first group exceeds one of the second plus half the
  probability of a tie (Stata's `porder`).
* `hl_estimate`, `hl_ci_low`, `hl_ci_high`: the Hodges-Lehmann shift estimate,
  the median of the n1 n2 differences (first minus second group), with the
  distribution-free interval (D_(q), D_(n1 n2 + 1 - q)) on the ordered
  differences. q is the alpha/2 quantile of the exact null distribution of U
  when it was computed without ties (the rule of R's `wilcox.test`), otherwise
  floor(n1 n2 / 2 + 1/2 - z_(alpha/2) sd). Computed when n1 n2 <= 1,000,000.

Tables: `ranks` (n, rank_sum, mean_rank, expected) and `tests` (rows `z`,
`z_continuity`, `exact`).

```python
data = {"wage": [12, 15, 9, 20, 22, 18, 25, 30], "union": [0, 0, 0, 0, 1, 1, 1, 1]}
r = oe.ranksum(data, "wage", "union")
r.attrs["u"], r.attrs["z"], r.attrs["p_exact"]        # 1.0, -2.0207, 0.0571
r.attrs["hl_estimate"], r.attrs["hl_ci_low"], r.attrs["hl_ci_high"]   # -10.0, -21.0, 2.0
```

### kwallis

Kruskal-Wallis test for k independent samples, with optional Dunn pairwise
comparisons.

```python
oe.kwallis(data, y, by, *, pairwise=False, adjust="bonferroni", missing="drop")
```

```
H = 12 / (N (N + 1)) * sum_j R_j^2 / n_j - 3 (N + 1)
H_ties = H / (1 - sum_g (t_g^3 - t_g) / (N^3 - N))          ~ chi2(k - 1)
```

`tests` has both lines (`chi2`, `chi2_ties`), as Stata prints; `statistic` /
`p_value` are the tie-corrected ones (SPSS). Effect sizes:
`epsilon_squared = H_ties / (N - 1)` and
`eta_squared = (H_ties - k + 1) / (N - k)`.

`pairwise=True` adds Dunn's (1964) comparisons as in SPSS's "Pairwise
Comparisons": z = (Rbar_i - Rbar_j) / sqrt([N (N + 1) / 12 - sum (t^3 - t) /
(12 (N - 1))] (1/n_i + 1/n_j)), two-sided normal p, and `p_adjusted` by
`adjust="bonferroni"` (SPSS), `"holm"` or `"none"`.

### median_test

Mood's median test: Pearson chi-square of the 2 x k table "above the pooled
median / not above" by group, df = k - 1.

```python
oe.median_test(data, y, by, *, ties="below", missing="drop")
```

Values equal to the median count as not above (`"below"`, SPSS and Stata's
default), as above (`"above"`) or are left out (`"drop"`). For two groups the
continuity-corrected chi-square and Fisher's exact p are added. Stata's
`medianties(split)` is not implemented.

### jonckheere

Jonckheere-Terpstra test against an ordered alternative.

```python
oe.jonckheere(data, y, by, *, order=None, missing="drop")
```

J = sum over pairs of groups i < j of the Mann-Whitney counts (values of the
later group exceeding values of the earlier one, ties 1/2), with

```
E(J) = (N^2 - sum n_i^2) / 4
Var(J) = [N(N-1)(2N+5) - sum n_i(n_i-1)(2n_i+5) - sum t_g(t_g-1)(2t_g+5)] / 72
       + [sum n_i(n_i-1)(n_i-2)] [sum t_g(t_g-1)(t_g-2)] / (36 N (N-1)(N-2))
       + [sum n_i(n_i-1)] [sum t_g(t_g-1)] / (8 N (N-1))
```

and z = (J - E(J)) / sqrt(Var(J)). `p_value` is two-sided (SPSS), `p_increasing`
and `p_decreasing` are the one-sided ones. `order=[...]` gives the groups from
lowest to highest (default: sorted levels). J is never computed from the N^2
pairs: with few groups it comes from the group-by-value cross-classification
(O(k m) for m distinct values), and when that table would exceed 2e7 cells from
a sort-based count that does not depend on the number of groups. For the
latter the observations are ordered by group and, within a group, by decreasing
value; an earlier observation with a smaller value then always lies in a lower
group, so the concordant pairs are the ascending pairs of the value sequence.
These are counted one binary digit at a time (two keys first differ at a digit
where the earlier has 0 and the later 1), each digit costing a few O(N) tensor
operations: O(N log N) in total, 0.9 s for a million rows in 300,000 groups.

## Related samples

### signrank

Wilcoxon signed-rank test for paired data or one sample.

```python
oe.signrank(data, y, paired_with=None, *, mu=0.0, zero_method="drop", exact=None,
            missing="drop")
```

The differences d = y - paired_with - mu are ranked by |d|; T+ and T- are the
rank sums of the positive and negative differences. With r_j the ranks of the
nonzero differences,

```
E(T+) = sum_j r_j / 2,    Var(T+) = sum_j r_j^2 / 4,    z = (T+ - E(T+)) / sqrt(Var(T+))
```

which is n(n+1)/4 and n(n+1)(2n+1)/24 without zeros and ties and contains the
tie correction sum (t^3 - t)/48 otherwise. No continuity correction.

| `zero_method` | zero differences | package |
| --- | --- | --- |
| `"drop"` (default) | removed before ranking (Wilcoxon) | SPSS |
| `"adjust"` | ranked, then left out of both sums (Pratt) | Stata `signrank` |

`z` has the sign of T+ - E(T+); SPSS prints the z of the smaller rank sum,
which is -|z|. `statistic` is min(T+, T-). `effect_r` = z / sqrt(number of
ranked differences).

### signtest

Exact sign test: with n+ positive and n- negative differences, n+ is
Binomial(n+ + n-, 1/2) under H0.

```python
oe.signtest(data, y, paired_with=None, *, mu=0.0, missing="drop")
```

`p_value` = min(1, 2 P(X >= max(n+, n-))); `p_positive` = P(X >= n+) and
`p_negative` = P(X >= n-) are Stata's one-sided lines; `z`, `p_value_normal`
are the continuity-corrected normal approximation z = (|n+ - n-| - 1) /
sqrt(n) that SPSS prints for n > 25.

### mcnemar

McNemar's test for two paired binary columns, from the discordant counts n12
and n21:

```python
oe.mcnemar(data, a, b, *, missing="drop")
```

| row of `tests` | statistic | package |
| --- | --- | --- |
| `mcnemar` | (n12 - n21)^2 / (n12 + n21), chi2(1) | Stata `mcc` |
| `continuity` | (\|n12 - n21\| - 1)^2 / (n12 + n21), 0 when n12 = n21 | SPSS (more than 25 discordant pairs) |
| `exact` | 2 P(X <= min(n12, n21)), X ~ Binomial(n12 + n21, 1/2) | SPSS (25 or fewer), Stata |

Without discordant pairs the chi-squares are undefined (`None`) and the exact
p is 1.

### symmetry

Tests for a square K x K table of two classifications of the same subjects.

```python
oe.symmetry(data, a, b, *, missing="drop")
```

* `symmetry`: Bowker's chi2 = sum_{i<j} (n_ij - n_ji)^2 / (n_ij + n_ji); the
  degrees of freedom count the pairs with n_ij + n_ji > 0 (Stata, SPSS).
* `marginal_homogeneity`: Stuart-Maxwell chi2 = d' V^- d with d_i = n_i. -
  n_.i, V_ii = n_i. + n_.i - 2 n_ii, V_ij = -(n_ij + n_ji), df = rank(V).
* `marginal_homogeneity_ordinal` (numeric categories only): z = sum (a - b) /
  sqrt(sum (a - b)^2), the SPSS `NPAR TESTS /MH` statistic.

For K = 2 all three equal McNemar's uncorrected chi-square (z squared).

### friedman

Friedman's test for k related samples in wide layout (one row per subject,
one column per condition), with Kendall's W.

```python
oe.friedman(data, columns, *, pairwise=False, adjust="bonferroni", missing="drop")
```

Values are ranked within each row; R_j are the column rank sums and T_i =
sum (t^3 - t) over the tie groups of row i:

```
chi2 = [12 / (n k (k + 1)) * sum_j R_j^2 - 3 n (k + 1)] / [1 - sum_i T_i / (n k (k^2 - 1))]
W = chi2 / (n (k - 1))
```

chi2 ~ chi2(k - 1). `pairwise=True` adds SPSS's Dunn-type comparisons, z =
(Rbar_i - Rbar_j) / sqrt(k (k + 1) / (6 n)).

### cochran_q

Cochran's Q for k related binary columns:

```python
oe.cochran_q(data, columns, *, positive=None, missing="drop")
```

Q = (k - 1) [k sum C_j^2 - (sum C_j)^2] / [k sum L_i - sum L_i^2] ~ chi2(k - 1),
with C_j the successes of column j and L_i those of subject i. For k = 2 it is
McNemar's chi-square.

## One-sample distribution tests

### ksmirnov

Kolmogorov-Smirnov test of one sample against a distribution, or of two samples.

```python
oe.ksmirnov(data, y, *, by=None, distribution="normal", params=None, exact=None,
            missing="drop")
```

**One sample.** With F the hypothesized cdf and x_(1) <= ... <= x_(n),

```
D+ = max_i (i/n - F(x_(i))),   D- = max_i (F(x_(i)) - (i-1)/n),   D = max(D+, D-)
```

| `distribution` | `params` | estimated when `params=None` (SPSS) |
| --- | --- | --- |
| `"normal"` | (mean, sd) | sample mean, sample sd (divisor n - 1) |
| `"uniform"` | (low, high) | sample minimum and maximum |
| `"exponential"` | (mean,) | sample mean |
| `"poisson"` | (mean,) | sample mean |

For the Poisson distribution the differences are taken on both sides of every
jump: D+ = max_v (F_n(v) - F(v)), D- = max_v (F(v - 1) - F_n(v-)).

The table has rows `positive`, `negative` (reported with a minus sign, as
SPSS's "Most Extreme Differences") and `combined`, with the asymptotic
p-values: exp(-2 n D+-^2) for the one-sided rows and the Kolmogorov
distribution at z = sqrt(n) D for the combined row (SPSS's "Kolmogorov-Smirnov
Z", Stata's combined K-S). `attrs["p_value"]` is the recommended p-value:

1. `p_lilliefors` for a normal distribution with estimated parameters. The
   asymptotic p-value is then far too large; SPSS prints the "Lilliefors
   Significance Correction". OpenEconometrics uses the Dallal-Wilkinson (1986) formula
   where it applies (p <= 0.1, 5 <= n; D is rescaled by (n/100)^0.49 for
   n > 100) and, above 0.1, the polynomial in Stephens' modified statistic
   used by R's `nortest::lillie.test`. SPSS reports only ">= .200" in that
   range.
2. `p_exact` when the exact distribution was computed (continuous distribution,
   given parameters): P(D_n >= D) by the Durbin matrix algorithm of Marsaglia,
   Tsang and Wang (2003), and in the far right tail by twice the exact
   one-sided probability of Birnbaum and Tingey (1951).
3. `p_asymptotic` otherwise. With estimated parameters of a non-normal
   distribution, and for the discrete Poisson distribution, it is conservative
   (a note says so).

**Two samples** (`by=` a column with two groups): D+ = max (F_1 - F_2),
D- = max (F_2 - F_1); z = D sqrt(n1 n2 / (n1 + n2)) is referred to the
Kolmogorov distribution. The exact p counts the lattice paths that leave the
band |i/n1 - j/n2| < D (only positive terms are added, so p-values as small as
1e-17 keep their relative accuracy).

```python
oe.ksmirnov(df, "x")                                  # normal, estimated: Lilliefors p
oe.ksmirnov(df, "x", params=[0, 1])                   # standard normal: exact p
oe.ksmirnov(df, "wait", distribution="exponential", params=[2.0])
oe.ksmirnov(df, "x", by="group")                      # two samples
```

| Stata / SPSS | OpenEconometrics |
| --- | --- |
| `ksmirnov x = normal((x - 1) / 2)` | `oe.ksmirnov(df, "x", params=[1, 2])` |
| `ksmirnov x, by(group) exact` | `oe.ksmirnov(df, "x", by="group", exact=True)` |
| `NPAR TESTS /K-S(NORMAL)=x` | `oe.ksmirnov(df, "x")` |
| `NPAR TESTS /K-S(POISSON)=k` | `oe.ksmirnov(df, "k", distribution="poisson")` |

### swilk

Shapiro-Wilk W for 3 <= n <= 5000 by Royston's algorithm AS R94 (1995), the
algorithm of SPSS, R and SciPy; Stata's `swilk` uses the same normalizing
transformation. W = (sum a_i x_(i))^2 / sum (x_i - xbar)^2 with Royston's (1992)
approximation of the weights from m_i = Phi^{-1}((i - 3/8)/(n + 1/4)). The
p-value is exact for n = 3 and otherwise 1 - Phi(z), z a normalizing
transformation of ln(1 - W) with polynomial mean and standard deviation in n
(4 <= n <= 11) or ln n (n >= 12).

```python
oe.swilk(data, y, *, missing="drop")        # one-row table: n, statistic (W), z, p_value
```

Stata additionally prints V, an index derived from W; it is not reported.

### sfrancia

Shapiro-Francia W' for 5 <= n <= 5000: the squared correlation of the ordered
data with the normal scores m_i, normalized as in Royston (1993):
mu = -1.2725 + 1.0521 (ln ln n - ln n), sigma = 1.0308 - 0.26758 (ln ln n +
2 / ln n), z = (ln(1 - W') - mu) / sigma, p = 1 - Phi(z). This is the default
of Stata's `sfrancia`; its `boxcox` option is not implemented.

### sktest

Stata's `sktest`: D'Agostino's (1970) test of skewness, the Anscombe-Glynn
(1983) test of kurtosis, and their joint tests, plus Jarque-Bera. The
coefficients are the moment ratios sqrt(b1) = m3 / m2^(3/2) and b2 = m4 / m2^2
(divisor n), as Stata's `summarize, detail` reports. n >= 8.

| row | statistic | reference |
| --- | --- | --- |
| `skewness` | sqrt(b1), z = Z1 | standard normal |
| `kurtosis` | b2, z = Z2 | standard normal |
| `joint_adjusted` | K2 with Royston's (1991) adjustment (Stata's default) | chi2(2) |
| `joint` | K2 = Z1^2 + Z2^2 (D'Agostino-Pearson; `sktest, noadjust`) | chi2(2) |
| `jarque_bera` | n/6 [b1 + (b2 - 3)^2 / 4] | chi2(2) |

`attrs["statistic"]` and `attrs["p_value"]` are the adjusted joint test. (The
time-series Jarque-Bera test of regression residuals is `oe.jarque_bera` in
the `arima` family.)

### runtest

Wald-Wolfowitz runs test of the order of a series (row order).

```python
oe.runtest(data, y, *, threshold="median", ties="above", continuity=None, exact=None,
           missing="drop")
```

With n1 values on one side of the threshold, n2 on the other, N = n1 + n2 and R
runs: E(R) = 2 n1 n2 / N + 1, Var(R) = 2 n1 n2 (2 n1 n2 - N) / (N^2 (N - 1)),
z = (R - E(R)) / sqrt(Var(R)).

| option | values | SPSS | Stata |
| --- | --- | --- | --- |
| `threshold` | `"median"`, `"mean"`, a number | `/RUNS(MEDIAN)` etc. | `threshold(#)`, `mean` |
| `ties` | `"above"`: x >= cut is above | default | |
| | `"below"`: x <= cut is below | | default |
| | `"drop"` | | `drop` |
| `continuity` | `None`: \|R - E\| - 1/2 when N < 50 | default | off unless `continuity` |

`p_exact` sums the exact probabilities of all run counts at least as far from
E(R) as the observed one; `p_exact_lower` = P(R <= r) indicates clustering
(positive dependence).

### bitest

Exact binomial test with the Clopper-Pearson interval.

```python
oe.bitest(data, y, *, p=0.5, positive=None, alpha=0.05, missing="drop")
```

`p_value` is the total probability of the outcomes that are no more probable
than the observed count (Stata `bitest`, R, SciPy); for p = 1/2 this is SPSS's
doubled one-tailed p. `p_lower` = P(K <= k), `p_upper` = P(K >= k). The
interval limits are beta quantiles, computed through the F distribution.

### prtest

Large-sample z tests of one proportion against `p` (default 0.5) or of two
proportions (`by=`), as Stata's `prtest`: the test uses the null standard
error (p (1 - p) / n, or the pooled proportion for two samples) and the
intervals use the Wald (unpooled) standard errors.

```python
oe.prtest(data, y, *, p=None, by=None, positive=None, alpha=0.05, missing="drop")
```

### chi2gof

Chi-square goodness of fit of the frequencies of one column.

```python
oe.chi2gof(data, y, *, expected=None, weights=None, missing="drop")
```

`expected` is `None` (equal frequencies, SPSS's default), a list in the sorted
order of the categories, or a mapping `{category: number}`; the numbers are
relative and are rescaled to the sample size. chi2 = sum (O - E)^2 / E on k - 1
degrees of freedom. A note reports the cells with expected frequency below 5.

## Cross-tabulation

### crosstab

```python
oe.crosstab(data, row, column, *, layer=None, weights=None, exact=None, expected=False,
            percentages=None, residuals=False, exact_reps=10_000, seed=2_000_003,
            alpha=0.05, missing="drop")
```

Returns a `TableSet`:

| table | content | when |
| --- | --- | --- |
| `counts` | counts with `total` margins | always |
| `expected` | r_i c_j / N | `expected=True` |
| `row_percent`, `column_percent`, `total_percent` | percentages with margins | `percentages="all"` or a list of `"row"`, `"column"`, `"total"` |
| `residuals`, `adjusted_residuals` | (f - E)/sqrt(E) and (f - E)/sqrt(E (1 - r/N)(1 - c/N)) | `residuals=True` |
| `tests` | chi-square family and exact tests | at least a 2x2 table |
| `measures` | value, ase, t, p_value per measure | at least a 2x2 table |
| `risk` | odds ratio and relative risks | 2x2 |
| `cmh`, `common_odds_ratio` | stratified analysis | `layer=` and 2x2 |

`weights` are frequency weights (non-integer weights are allowed; exact tests
then are skipped). Empty rows and columns are removed before testing. A table
may have at most 100,000 cells, 500 layers and 2,000,000 cells over all layers
(`too_many_categories` otherwise: group continuous values into classes first).

**Tests.**

| row | statistic | df |
| --- | --- | --- |
| `pearson` | sum (f - E)^2 / E | (R-1)(C-1) |
| `likelihood_ratio` | 2 sum f ln(f / E) | (R-1)(C-1) |
| `continuity` (2x2) | N (max(0, \|ad - bc\| - N/2))^2 / (r1 r2 c1 c2) | 1 |
| `fisher_exact`, `fisher_exact_less`, `fisher_exact_greater` (2x2) | hypergeometric; two-sided = probability of all tables no more probable than the observed one; one-sided P(n11 <= a), P(n11 >= a) | |
| `fisher_exact` (r x c, `exact=True`) | Fisher-Freeman-Halton | |
| `linear_by_linear` | (N - 1) r^2, r the Pearson correlation of the category scores | 1 |

Scores are the numeric category values, or 1..K for non-numeric categories.
For r x c tables the exact test enumerates every table with the observed
margins, column by column, as long as at most 3,000,000 partial tables are
alive; otherwise it draws `exact_reps` tables with the observed margins by
permuting the column labels (seed `seed`) and reports the share that are no
more probable than the observed one, with a 99% interval, as SPSS Exact Tests'
Monte Carlo option (`exact_method`, `exact_reps`, `exact_seed`,
`p_exact_ci_low`, `p_exact_ci_high` in `attrs`). A note reports, as SPSS does,
how many cells have an expected count below 5 and the minimum expected count.

**Measures.** `ase` is the asymptotic standard error that does not assume
independence (SPSS "Asymptotic Standard Error", ASE1). Every measure is a
function of the cell proportions, so ASE1^2 = sum f_ij (d theta / d f_ij)^2;
the analytic derivatives reproduce the closed forms of the SPSS algorithms
(and are checked in the tests against numerical differentiation). `t` is the
value divided by its standard error under independence (ASE0) and `p_value`
its two-sided normal p ("Approximate T" / "Approximate Significance").

| rows | definition | p-value |
| --- | --- | --- |
| `phi`, `cramers_v`, `contingency_coefficient` | sqrt(chi2/N) (signed for 2x2), sqrt(chi2/(N (q-1))), sqrt(chi2/(chi2+N)) | Pearson chi2 |
| `lambda_symmetric`, `lambda_row`, `lambda_column` | proportional reduction in the error of predicting the modal category | t |
| `goodman_kruskal_tau_row`, `_column` | proportional reduction in the Gini variation | chi2 approximation (N-1)(K-1) tau |
| `uncertainty_symmetric`, `_row`, `_column` | (H(R) + H(C) - H(RC)) / H(dependent) | likelihood-ratio chi2 |
| `gamma` | (P - Q) / (P + Q) | t |
| `kendall_tau_b`, `kendall_tau_c` | (P - Q)/sqrt(D_r D_c); q (P - Q) / (N^2 (q - 1)) | t |
| `somers_d_symmetric`, `_row`, `_column` | (P - Q) over the pairs untied on the independent variable | t |
| `pearson_r`, `spearman_rho` | correlation of the scores / of the midrank scores | t(N - 2) |
| `kappa`, `kappa_linear`, `kappa_quadratic` | Cohen's (weighted) kappa | z = kappa / ASE0 |

"_row" / "_column" name the *dependent* variable. P and Q are twice the
numbers of concordant and discordant pairs, q = min(R, C), D_r = N^2 - sum
r_i^2. Kappa is computed when rows and columns have exactly the same
categories; the weighted versions (weights 1 - |i - j|/(K - 1) and
1 - (i - j)^2/(K - 1)^2) are added for K > 2.

**Risk (2x2).** `odds_ratio` = ad/(bc) with Woolf's interval
exp(ln OR +- z sqrt(1/a + 1/b + 1/c + 1/d)); `risk_ratio_column_1` =
[a/(a+b)] / [c/(c+d)] and `risk_ratio_column_2` = [b/(a+b)] / [d/(c+d)] with
log-scale intervals (SPSS "For cohort ..." rows). Entries that need a zero cell
are omitted.

**Layers.** With `layer=` every table is repeated for each layer level and for
the total; index labels are prefixed `layer=level | ` and `total | `. `attrs`
describe the total table. For 2x2 tables:

| row of `cmh` | statistic |
| --- | --- |
| `cmh` | (sum (a_k - E_k))^2 / sum V_k, V_k = r1 r2 c1 c2 / (n^2 (n - 1)) |
| `cmh_continuity` | (\|sum (a_k - E_k)\| - 1/2)^2 / sum V_k (SPSS "Mantel-Haenszel") |
| `cochran` | same numerator, binomial variance r1 r2 c1 c2 / n^3 (SPSS "Cochran's") |
| `breslow_day` | sum (a_k - A_k)^2 / Var(a_k \| psi_MH), df K - 1 |
| `tarone` | Breslow-Day minus (sum (a_k - A_k))^2 / sum Var |

`common_odds_ratio` is the Mantel-Haenszel estimate sum(a d / n) / sum(b c / n)
with the Robins-Breslow-Greenland standard error of its logarithm
(`std_error_log`), the log-scale interval and `z` = ln(estimate) / se with its
two-sided normal `p_value` (SPSS "Asymp. Sig." of the common odds ratio).
Layers with a zero margin are left out (noted).

```python
t = oe.crosstab(df, "treated", "cured", percentages=["row"])
t["counts"], t["row_percent"], t["tests"], t["measures"], t["risk"]
t.attrs["statistic"], t.attrs["p_value"], t.attrs["p_exact"], t.attrs["cramers_v"]

oe.crosstab(df, "exposed", "case", layer="age_group")["cmh"]
oe.crosstab(df, "rater1", "rater2")["measures"].loc[["kappa", "kappa_linear"]]
oe.crosstab(df, "region", "party", exact=True).attrs["p_exact"]     # Fisher-Freeman-Halton
```

| Stata / SPSS | OpenEconometrics |
| --- | --- |
| `tabulate a b, chi2 lrchi2 exact V gamma taub` | `oe.crosstab(df, "a", "b")` |
| `tabulate a b [fw=n], expected row column cell` | `oe.crosstab(df, "a", "b", weights="n", expected=True, percentages="all")` |
| `cc case exposed, by(stratum) bd` / `mhodds` | `oe.crosstab(df, "exposed", "case", layer="stratum")` |
| `kap rater1 rater2` / `kap ..., wgt(w)` / `wgt(w2)` | `kappa` / `kappa_linear` / `kappa_quadratic` rows |
| `CROSSTABS /TABLES=a BY b BY c /STATISTICS=ALL /CELLS=COUNT EXPECTED ROW COLUMN TOTAL SRESID ASRESID` | `oe.crosstab(df, "a", "b", layer="c", expected=True, percentages="all", residuals=True)` |
| `CROSSTABS ... /METHOD=EXACT` / `/METHOD=MC SAMPLES(10000)` | `exact=True` (enumeration when feasible, else Monte Carlo) |

### tabulate

One-way frequency table (SPSS `FREQUENCIES`): `count`, `percent` (of all rows,
missing included), `valid_percent`, `cumulative_percent`, with a final
`missing` row when the column has missing values. `weights` are frequency
weights.

```python
oe.tabulate(data, column, *, weights=None, missing="drop")
```

## ROC analysis

### roc

```python
oe.roc(data, y, score, *, positive=None, alpha=0.05, missing="drop")
```

`curve` lists, for each cutoff c ("positive when score >= c"), `sensitivity`,
`specificity` and `one_minus_specificity`. As in SPSS the cutoffs are the
smallest score minus 1, the midpoints between consecutive distinct scores, and
the largest score plus 1. More than 400 cutoffs are thinned to 400 evenly
spaced ones (`attrs["curve_points"]` is the full number); the area always uses
all of them.

`auc` is the trapezoidal area, identical to the Mann-Whitney statistic
U / (n1 n0). Two standard errors are reported, each with a normal interval
clipped to [0, 1]:

| row | standard error | package |
| --- | --- | --- |
| `delong` | sqrt(S10 / n1 + S01 / n0), the variances of the placement values (DeLong, DeLong and Clarke-Pearson 1988) | Stata `roctab` default |
| `hanley_mcneil` | sqrt([A(1-A) + (n1-1)(Q1-A^2) + (n0-1)(Q2-A^2)] / (n1 n0)) with Q1, Q2 estimated from the data as in Table II of Hanley and McNeil (1982) | SPSS "nonparametric" assumption |

Q1 is the probability that two random positives both score above one random
negative, Q2 that one positive scores above two random negatives. With p_g
positives and m_g negatives at the g-th distinct score, P_g positives above it
and M_g negatives below it,

```
Q1 = sum_g m_g (P_g^2 + P_g p_g + p_g^2 / 3) / (n0 n1^2)
Q2 = sum_g p_g (M_g^2 + M_g m_g + m_g^2 / 3) / (n1 n0^2)
```

(the 1/3 is the chance that both members of a tied pair win when ties are
broken at random). On the rating data of Hanley and McNeil (1982) this gives
their published W = 0.893, SE = 0.032; the DeLong row gives 0.8932 and 0.0307
with the interval [0.83295, 0.95339] printed for the same data in Stata's
`roctab` manual entry. Stata's `roctab, hanley` instead plugs in the
approximations Q1 = A / (2 - A), Q2 = 2 A^2 / (1 + A) and is not reported.

`z`, `p_value` test H0: area = 0.5 with the null (Mann-Whitney) standard error
`se_null` = sqrt([(N + 1) - sum (t^3 - t) / (N (N - 1))] / (12 n1 n0)), which
is tie-corrected, so they coincide with `oe.ranksum`. SPSS's "Asymptotic Sig."
uses the variance without the tie term, (N + 1) / (12 n1 n0); that version is
in `se_null_uncorrected`, `z_uncorrected`, `p_value_uncorrected` (identical
when the scores have no ties). `youden_index` and `youden_threshold` give the
cutoff maximizing sensitivity + specificity - 1.

Everything comes from the tie groups of the pooled scores (one sort): the
placement of a positive is the share of negatives below it plus half the share
tied with it. No n1-by-n0 matrix is formed.

### roccomp

```python
oe.roccomp(data, y, scores, *, positive=None, alpha=0.05, missing="drop")
```

DeLong's test for k correlated ROC areas measured on the same cases: with the
covariance S = S10/n1 + S01/n0 of the areas and L contrasting every score with
the first, chi2 = (L A)'(L S L')^-(L A) on rank(L S L') degrees of freedom
(Stata's `roccomp`). `auc` gives each area with its DeLong standard error,
`pairwise` the difference, standard error, z and unadjusted p for every pair,
`attrs["covariance"]` the k x k matrix S. A score that is a monotone transform
of another has the same curve; its contrast is dropped.

## Performance

Timings on one million rows (Apple silicon laptop, CPU, float64):

| call | seconds |
| --- | --- |
| `ranksum`, `kwallis` (6 groups, pairwise), `median_test` | 0.11, 0.11, 0.09 |
| `jonckheere` (6 groups; 300,000 groups) | 0.29, 0.9 |
| `signrank`, `signtest`, `mcnemar`, `symmetry` | 0.20, 0.02, 0.02, 0.03 |
| `friedman` (3 columns, pairwise), `cochran_q` | 0.08, 0.03 |
| `ksmirnov` normal / Poisson / two-sample | 0.09, 0.02, 0.11 |
| `runtest`, `bitest`, `prtest`, `chi2gof`, `sktest` | 0.08, 0.03, 0.01, 0.01, 0.08 |
| `crosstab` 5x6; with layer, weights and all tables; 2x2 with layer | 0.02, 0.03, 0.06 |
| `tabulate` | 0.03 |
| `roc`, `roccomp` (3 scores) | 0.11, 0.23 |
| `swilk`, `sfrancia` (n = 5000, their maximum) | 0.002, 0.001 |

Exact computations: `ranksum` with n1 = n2 = 60 takes 0.01 s, `signrank` with
n = 300 0.01 s, the one-sample Kolmogorov distribution at n = 1000 0.001 s,
the two-sample one at 100 x 100 0.002 s.

## Limitations

* No exact test for Kruskal-Wallis, Friedman, Jonckheere-Terpstra, Cochran's Q
  or the median test with more than two groups (SPSS Exact Tests offers them).
* Exact rank distributions are limited by memory: about 6 million table cells
  for `ranksum` (roughly n1 = n2 = 150 without ties) and a support-times-n
  product of 3e8 for `signrank` (roughly n = 800); larger requests raise
  `exact_unavailable`.
* Size limits (`too_many_groups` / `too_many_categories`): pairwise comparisons
  of `kwallis` and `friedman` for at most 200 groups / columns; paired tables
  (`mcnemar`, `symmetry`) with at most 1,000 categories; `tabulate` with at most
  100,000 distinct values; `crosstab` with at most 100,000 cells, 500 layers and
  2,000,000 cells over all layers; `roccomp` with at most 50 scores.
* `ksmirnov`: one-sided exact p-values and Stata's "corrected" two-sample
  p-value are not reported; no Lilliefors correction for the exponential
  distribution.
* `crosstab`: the generalized Cochran-Mantel-Haenszel tests for R x C x K
  tables and exact confidence limits of the odds ratio are not implemented;
  Monte Carlo exact tests are limited to N x reps <= 5e7.
* `roc`: only "larger score = positive"; negate the score for the opposite
  direction. The bi-negative-exponential standard error of SPSS (which is also
  what Stata's `roctab, hanley` uses) and Bamber's (1975) standard error of
  Stata's `roctab, bamber` are not implemented.
* Rank correlations of raw data are `oe.correlate(..., method="spearman")` and
  `method="kendall"` in the `stats` family; there is no separate `oe.spearman`
  / `oe.ktau`.

## Verification and conventions that remain unchecked

Nothing here has been run side by side with Stata or SPSS, and no parity is
claimed. What the test suite does establish:

* **Worked examples printed in the manuals are reproduced** to every printed
  digit: `[R] ranksum` (fuel data: rank sums 128 / 172, adjusted variance
  295.96, z = -1.279, p = 0.2010), `[R] signrank` and `[R] signtest` on the
  same data (T+ = 13.5, adjusted variance 160.62, z = -1.973, p = 0.0485;
  one-sided 0.9673 / 0.1133, two-sided 0.2266), `[R] roctab` (Hanley-McNeil
  rating data: 0.8932, 0.0307, [0.83295, 0.95339]), `[R] sktest` (auto data:
  13.13 -> 10.95, 4.05 -> 4.19) and Hanley and McNeil (1982): W = 0.893,
  SE = 0.032.
* **Permutation moments and exact distributions** are checked by complete
  enumeration (rank sum, signed-rank sum, Jonckheere-Terpstra, runs,
  two-sample Smirnov, Fisher-Freeman-Halton), the exact Kolmogorov distribution
  against 50-digit evaluations of the Durbin matrix (it is more accurate than
  SciPy's `kstwo` for n > 140, which switches to an approximation good to
  about 1e-6), and every measure of association and its ASE1 against the
  textbook definition and a numerically differentiated multinomial delta
  method.
* The remaining statistics are compared with SciPy and statsmodels.

The following choices rest on the published formulas as recalled and should be
compared with the packages before parity is claimed:

* **Lilliefors p-values above 0.1**: the polynomial of `nortest::lillie.test`
  (within about 0.01-0.03 of simulated tails for n = 20, 300, 1000). SPSS
  prints only a lower bound (.200) there.
* **Shapiro-Francia normalization** (Royston 1993 constants); calibration under
  normality is tested, but no independent implementation was available.
* **Hodges-Lehmann interval** in `ranksum`: the order-statistic rule of
  Hollander and Wolfe / R; SPSS's NPTESTS may round the index differently.
* **Exact p with ties** (`ranksum`, `signrank`): the conditional permutation
  distribution; SPSS and SciPy ignore ties in the exact distribution.
* **Two-sided exact p** for asymmetric distributions: P(|T - E| >= |t - E|).
* **Default size rules for exact p-values** follow SPSS (`ranksum`:
  n1 n2 <= 400) or the classical tables (`signrank`: n <= 25); current Stata
  versions compute exact p-values for samples up to 200.
* **`symmetry`'s ordinal marginal-homogeneity z** as SPSS's `NPAR TESTS /MH`.
* **`roc`**: SPSS's null standard error is taken to be sqrt((N + 1) / (12 n1
  n0)) without a tie term (`p_value_uncorrected`), and its "nonparametric"
  standard error to be the Table II formula of Hanley and McNeil with the
  n=^2 / 3 tie term; the headline `p_value` is the tie-corrected rank-sum test.
  Stata's `roctab, hanley` formula (Q1 = A/(2 - A), Q2 = 2A^2/(1 + A)) is as
  recalled from its manual.
* **`runtest`** defaults follow SPSS (ties above, continuity for N < 50);
  Stata's defaults need `ties="below", continuity=False`.
* **`kappa`**: `p_value` is two-sided (SPSS); Stata's `kap` prints the
  one-sided Prob > Z.
* **Cramer's V** is nonnegative; Stata prints a signed V for 2x2 tables, which
  is the `phi` row here.
* **Effect sizes** `effect_r` (z / sqrt(N)), `epsilon_squared`, `eta_squared`
  follow the common textbook definitions (Tomczak and Tomczak 2014); neither
  package prints them in these procedures.
* **Pearson's r and Spearman's rho in `crosstab`** are tested with
  t = r sqrt((N - 2) / (1 - r^2)) on N - 2 degrees of freedom (the SPSS
  algorithms); SPSS's output footnote calls this significance "based on normal
  approximation", so its printed value may come from the normal distribution.
* **Approximate T (ASE0)** of the ordinal measures, lambda and the uncertainty
  coefficient uses the formulas of the SPSS algorithms (Brown and Benedetti
  1977) as recalled; they plug the observed cells into the null variance, so
  the p-values differ from permutation-based tests of the same measures (for
  example SciPy's `kendalltau`). The ASE1 values are verified numerically, and
  lambda's against Goodman and Kruskal's closed form.
