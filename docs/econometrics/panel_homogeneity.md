# Panel slope homogeneity

`oe.xthst` tests whether every numeric coefficient in a common predictor list
is the same across panel units, while allowing each unit its own intercept.
It computes the Pesaran–Yamagata standardized slope-dispersion statistic and
its Gaussian moment adjustment directly with float64 Torch. It returns a
diagnostic table, rather than a fitted model or a `ResultBundle`.

```python
import openecon as oe

test = oe.xthst(data=df, y="growth", x=["investment", "education"],
               panel="country", time="year")
display(test)
test.to_latex(caption="Slope homogeneity", notes=test.attrs["notes"])

# Complete-case removal is explicit, and changes each unit's T_i.
test = oe.xthst(data=df, y="growth", x=["investment", "education"],
               panel="country", time="year", missing="drop",
               method="adjusted", block_size=128, memory_mb=64)
```

## Public contract

```python
oe.xthst(*, data, y, x, panel, time, method="all", missing="raise",
          block_size=128, memory_mb=64)
```

| Argument | Contract |
| --- | --- |
| `data` | In-memory OpenEconometrics/pandas frame, mapping of columns, or row records. Streaming `Dataset` inputs fail explicitly. |
| `y`, `x`, `panel`, `time` | Distinct column names; `x` is a nonempty list of finite real numeric predictors. Unit intercepts are always included and are not tested. |
| `method` | `"delta"`, `"adjusted"`, or `"all"`; default returns exactly two rows. Both balanced and unbalanced panels are supported. |
| `missing` | `"raise"` rejects missing model inputs. `"drop"` excludes incomplete y/x rows. Missing unit/date keys and duplicate keys always fail, including on rows that would otherwise be dropped. |
| `block_size` | Maximum units per QR tile, integer 1–512. Actual tile size may be smaller to meet the workspace estimate. |
| `memory_mb` | Estimated additional tensor-workspace budget in MiB, integer at least 1. This excludes input preparation, unit indices/means, and private library workspace; it is not a total process-memory limit. |

Every unit must retain at least `max(6, k+2)` complete observations, where
`k = len(x)`. The `T_i > 5` condition respects the inverse-moment requirement
in the original Gaussian setting; `T_i > k+1` leaves residual degrees of
freedom. These are domain guards, **not** evidence that asymptotic inference
is accurate in a particular short panel. No unit or predictor is silently
omitted. All within-unit predictor designs must have the same full rank.

The result is an OpenEconometrics DataFrame with `statistic`, `p_value`, `n_groups`,
`nobs`, and `k` columns and method row names. JSON-safe `attrs` retain the
sample counts, exact-date balance, minimum/maximum `T_i`, dispersion sum,
workspace estimates, assumptions, source and data hash. DataFrame JSON export
contains the table; serialize `attrs` separately when preserving the diagnostic
contract, for example:

```python
import json

payload = {"table": json.loads(test.to_json(orient="split")),
           "metadata": test.attrs}
json.dumps(payload, allow_nan=False)
```

`to_latex()` provides the common escaped booktabs table format. Supply
`notes=test.attrs["notes"]` to include the inferential assumptions in the
export; DataFrame metadata is not automatically printed as table notes.

## Statistic and variance definition

The original authors’ [CESifo Working Paper 1438](https://www.ifo.de/DocDL/cesifo1_wp1438.pdf)
gives the variance/pooled-estimator construction in equations (3.1)–(3.3),
Gaussian moments in (3.17), and the unbalanced extension in Remark 3.2,
equations (3.20)–(3.23). The journal publication is
[Pesaran and Yamagata (2008), Journal of Econometrics 142(1), 50–93](https://doi.org/10.1016/j.jeconom.2007.05.010).

For unit `i`, let \(M_i\) remove its intercept and let
\(\hat\beta_i\) be its individual within OLS slopes. First fit the
**unweighted** pooled within estimator,

\[
\hat\beta_{FE}
=\left(\sum_i X_i'M_iX_i\right)^{-1}\sum_i X_i'M_iy_i.
\]

Estimate each error variance from this null-constrained fit:

\[
\tilde\sigma_i^2
=\frac{(y_i-X_i\hat\beta_{FE})'M_i(y_i-X_i\hat\beta_{FE})}{T_i-1}.
\]

This is deliberately different from unrestricted unit OLS residual variance
divided by \(T_i-k-1\). Replacing one with the other produces a different test.
Then form the variance-weighted pooled estimator and individual dispersion,

\[
\tilde\beta_{WFE}
=\left(\sum_i\frac{X_i'M_iX_i}{\tilde\sigma_i^2}\right)^{-1}
\sum_i\frac{X_i'M_iy_i}{\tilde\sigma_i^2},\qquad
d_i=(\hat\beta_i-\tilde\beta_{WFE})'
\frac{X_i'M_iX_i}{\tilde\sigma_i^2}
(\hat\beta_i-\tilde\beta_{WFE}).
\]

The large-period scaling is

\[
\tilde\Delta=\frac{\sum_i(d_i-k)}{\sqrt{2kN}}.
\]

For Gaussian errors, the true-null quadratic-form ratio has mean `k` and
variance \(v_i^2=2k(T_i-k-1)/(T_i+1)\). The implemented adjustment is

\[
\tilde\Delta_{adj}=\frac1{\sqrt N}
\sum_i\frac{d_i-k}{\sqrt{2k(T_i-k-1)/(T_i+1)}}.
\]

On a balanced panel, all denominators coincide and this reduces to the usual
balanced expression. On an unbalanced panel, **each unit is standardized
separately**; substituting an average `T` or the sum of variances changes the
statistic. Calendar overlap is not required under the independent-error
contract; absent dates are not imputed and observations are not paired across
units.

## Inference scope

The supported inferential contract is a **static** linear model with strictly
exogenous regressors and Gaussian errors independent across units and time.
Variance is positive and constant over time within each unit, but may differ
across units. The routine cannot verify these assumptions from column values.
Large `N` and increasing unit time lengths, together with the paper's design
and moment regularity conditions, remain necessary. The reported p-values are
upper standard-normal tails: large positive dispersion rejects homogeneity.
An unusually negative statistic does not establish the heterogeneous-slope
alternative and is not turned into a small two-sided p-value.

The finite-period moment calculation concerns an ideal ratio evaluated at the
true null slope. Feasible `d_i` also estimate pooled slopes and variances.
Consequently **neither reported p-value is exact at finite `N,T`**, and
`adjusted` is not a finite-`N` correction or a bootstrap. The normal reference
is an asymptotic approximation even when every `T_i` exceeds the minimum.

The [2008 publisher abstract](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001224)
also describes a non-Gaussian large-panel result with a relative-expansion
condition \(\sqrt N/T^2\to0\). That theorem's additional assumptions must not
be inferred merely from selecting `method="delta"`; this release limits its
documented inference to the Gaussian contract above. It does not implement
non-Gaussian moment estimators, dynamic-panel/near-unit-root corrections,
HAC or cross-section-dependence corrections, weights, categorical treatment
coding, a subset of slopes with heterogeneous nuisance slopes, or a saved-fit
adapter. In particular, adding time dummies or using lagged outcomes does not
automatically make inference valid for common effects or dynamic models.

`provenance["stata_parity_validated"]` stays `False`. Matching independent
mathematical oracles is not a licensed Stata comparison or support for every
option of the community-contributed Stata command with the same name.

## Dates, rank and numerical guards

Integer period types preserve exact identity, including int64 dates above
\(2^{53}\) and uint64 dates. Floating dates must be integral within
\(2^{53}-1\); use an integer dtype for larger values. Datetime columns, also
with timezones, are accepted. Boolean/text dates, missing keys and repeated
unit/date keys fail. Gaps and different date sets are accepted because this
static procedure forms no lags. `balanced` means the units share the complete
exact date set; `equal_unit_lengths` is recorded separately.

Predictors are origin-shifted within units, globally rescaled per column, and
demeaned. Column-normalized QR factors are checked for full rank using the
panel family's relative within-variation threshold: a smallest/largest
singular-value ratio no greater than `sqrt(1e-13)` fails. Numerically unresolved
collinearity is rejected rather than silently changing `k` across units.
Pooled-FE residual norms no greater than `1e-14` of their unit outcome norm
also fail instead of reporting a variance dominated by roundoff. The
calculation checks finite factors, weighting and statistics throughout.

Errors include `invalid_spec`, `invalid_option`, `missing_columns`,
`duplicate_columns`, `empty_data`, `missing_panel_time`, `invalid_time`,
`duplicate_panel_time`, `missing_values`, `empty_sample`,
`insufficient_panels`, `insufficient_panel_observations`,
`rank_deficient_panel`, `degenerate_panel_variance`, `workspace_too_small`,
`non_finite_values`, `complex_values`, `non_numeric_column`,
`streaming_unsupported` and `numerical_failure`.

## Computational and validation scope

Three batched tall-skinny QR passes obtain the unweighted pooled fit, the
variance-weighted fit, and the dispersion. Each pass compresses bounded
unit/row tiles, uses only small QR factors and triangular solves, and never
forms normal equations or a dense time projection. Unit factors are recomputed
between passes rather than stored as an `N × k × k` cache. Numeric work is
\(O(nk^2)\), with column/input sorting and indexing costs separate. There are
no Python loops over observations, unit pairs, `N × N` matrices, or `T_i × T_i`
matrices in production.

Input preparation is in memory and retains \(O(nk+Nk)\) values/means plus
row/unit/date indices and temporary copies. `workspace` reports conservative
live tensor tile estimates and the materialized numeric input size. pandas
and Torch allocation overhead and private LAPACK workspaces are excluded.
The kernel runs on **CPU float64**; this diagnostic makes no GPU, streaming,
total-RAM cap or arbitrary-row-count performance guarantee.

`tests/test_econ_panel_homogeneity.py` supplies independent NumPy projection
and stacked least-squares oracles with SciPy normal tails. It separately
checks pooled-null versus unrestricted variance, per-unit versus average-T
adjustment, Gaussian ratio moments from a Beta distribution, common-basis
and unit-offset invariance, large/tiny units of measure, exact date identity,
missing/rank/domain guards, bounded QR dimensions, JSON/LaTeX exports and
seeded null/power simulations. Simulation checks use a fixed design and seed
with broad tolerances; they are not a universal finite-sample size guarantee.
