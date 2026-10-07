# Cross-section residual dependence

`oe.xtcd` tests a supplied long-form residual panel for contemporaneous dependence
between units. It does not fit a model, reconstruct saved coefficients, or infer
which residual definition an estimator intended.

```python
import openecon as oe

# residual_data has one row per unit/date and a numeric fitted residual e.
test = oe.xtcd(data=residual_data, residual="e", panel="country", time="year")
test.to_latex()

# The LM variants require every unit to share the same complete date set.
all_tests = oe.xtcd(data=balanced_residual_data, residual="e",
                   panel="country", time="year", method="all")

# Missing residual rows become absent observations, with exact pairwise dates.
cd = oe.xtcd(data=residual_data, residual="e", panel="country", time="year",
             method="cd", missing="drop", block_size=128, memory_mb=64)
```

## Arguments and result

```python
oe.xtcd(*, data, residual, panel, time, method="cd", missing="raise",
         min_overlap=4, block_size=128, memory_mb=64)
```

| Argument | Contract |
| --- | --- |
| `data` | In-memory OpenEconometrics/pandas DataFrame, column mapping or row records. |
| `residual`, `panel`, `time` | Three distinct column names. No weights or outcome/predictor specification is accepted. |
| `method` | `"cd"`, `"lm"`, `"scaled_lm"` or `"all"`. `"all"` produces those three rows only on a balanced common-date sample. |
| `missing` | `"raise"` rejects missing residuals; `"drop"` drops them before determining overlaps. Missing unit/date keys always fail. |
| `min_overlap` | Integer at least 4; every unit pair must meet it. No undefined pair is skipped. |
| `block_size` | Requested maximum units per calculation tile, integer 1–512. It may be reduced to meet the workspace estimate. |
| `memory_mb` | Calculation-workspace estimate budget, integer 1–1024 MiB. It excludes the input/preprocessing arrays and private library workspaces. |

The result is an OpenEconometrics DataFrame, indexed by requested method, with
`statistic`, `distribution`, `df`, `p_value`, and `alternative` columns. `df`
is populated for the chi-square LM row. CD and scaled LM use two-sided normal
p-values; BP LM uses the upper chi-square tail.

JSON-safe `attrs` record input/retained/dropped row counts, unit/date/pair counts,
whether the sample is balanced, overlap extrema, correlation summaries,
workspace settings and assumptions. `to_latex()` uses the standard publication
table output. The procedure does not return a full correlation matrix or store
one record per unit pair.

## Statistics

The methods follow [Pesaran (2004), IZA Discussion Paper 1240](https://docs.iza.org/dp1240.pdf),
equations (3), (6), (7), (31), and section 9. Let \(P=N(N-1)/2\),
\(T_{ij}\) be the number of shared dates, and \(r_{ij}\) the Pearson correlation
after centering each residual on that pair's shared dates. Then

\[
CD = P^{-1/2}\sum_{i<j}\sqrt{T_{ij}}\,r_{ij}.
\]

For a balanced common-date sample, \(T_{ij}=T\), and

\[
LM=T\sum_{i<j}r_{ij}^{2},\qquad
LM_{scaled}=\frac{LM-P}{\sqrt{2P}}.
\]

Their asymptotic reference distributions are standard normal for CD/scaled LM
and chi-square with \(P\) degrees of freedom for BP LM. These are asymptotic
approximations. BP LM is intended for fixed/small \(N\) and sufficiently large
\(T\); scaled LM takes the large-\(T\) limit before the large-\(N\) limit.
Both can be badly sized with short panels and many units. Scaled LM is **not**
the later bias-adjusted LM procedure.

The unbalanced extension of CD requires more than three shared dates per pair.
This implementation enforces at least four for every method. It does not
substitute varying \(T_{ij}\) into the LM variants; requesting LM/scaled LM/all
on an unbalanced sample raises `unbalanced_lm_unsupported`.

## Residual definition and inference assumptions

The caller supplies the residuals appropriate to the economic model and null.
Individual regressions with intercepts are the original heterogeneous-panel
setting. Panel-specific positive scales and constants do not change pairwise
correlations. There is no saved-`xtreg` adapter in this procedure, and no claim
that residuals from every panel estimator are interchangeable.

The documented CD null assumes appropriate regressions, serially independent
innovations and symmetry conditions. The procedure cannot check these conditions
from a residual column. It supplies no HAC, bootstrap, estimated time/common
effect or latent-factor correction. Testing residuals after estimating many
shared effects requires its own theory and correction; such a correction is
not implied by this function. Positive and negative correlations can cancel in
CD even when some pairs are dependent.

`provenance["stata_parity_validated"]` remains `False`: independent mathematical
oracles are not a comparison with a licensed Stata result. The numerical core
uses float64 PyTorch on CPU. No GPU or out-of-core guarantee is made.

## Alignment and numerical guards

Date identities are global, not unit-specific row positions. Irregular spacing
is allowed because the procedure matches contemporaneous observations rather
than forming lags. Integer columns retain exact dates, including signed int64
dates beyond \(2^{53}\) and unsigned integer dates. Float dates must be integer
valued within \(2^{53}-1\); use an integer dtype for larger periods. Datetime
columns, including timezone-aware dates, are accepted. Boolean/text dates,
missing keys and duplicate unit/date keys fail.

Residuals must be finite real numbers. Each unit needs sufficient observations
and nonconstant residuals; every pair needs sufficient overlap and nonconstant
residuals within that overlap. Unit centering and L2 normalization avoid
overflow when residual magnitudes differ greatly. For unbalanced overlaps the
implementation forms centered moments from tiled sums. If cancellation makes a
variance no larger than 64 float64 epsilons times its uncentered second moment,
it raises `degenerate_pair_variance` instead of reporting an unreliable ratio.
Correlations outside their domain by more than numerical roundoff also fail.

Errors include `invalid_spec`, `invalid_option`, `missing_columns`,
`duplicate_columns`, `empty_data`, `missing_panel_time`, `duplicate_panel_time`,
`invalid_time`, `missing_values`, `empty_sample`, `non_finite_values`,
`insufficient_panels`, `insufficient_panel_observations`,
`insufficient_pair_overlap`, `degenerate_residuals`, `degenerate_pair_variance`,
`unbalanced_lm_unsupported`, `workspace_too_small` and `numerical_failure`.

## Computational scope

Input preprocessing retains an in-memory residual/date representation with
\(O(n)\) storage. The additional numerical calculation uses bounded unit/date
tiles and compensated scalar accumulation. It does not allocate unbounded
\(N\times N\), padded \(N\times T\), or observation-by-observation matrices.
Workspace metadata reports a conservative estimate for live tensor tiles,
masks, pair buffers and expression temporaries; it is **not** a hard total
process-RAM limit. pandas input preparation, Torch allocation overhead and BLAS
private workspace are outside that estimate.

Balanced CD alone uses
\[
\sum_{i<j}r_{ij}=\tfrac12\left(\left\|\sum_i\tilde e_i\right\|^2-N\right),
\]
where every centered \(\tilde e_i\) has unit L2 norm. This has linear work in
the number of residual observations. LM and unbalanced CD examine all unit
pairs, with bounded pair/time tiles; their time cost remains quadratic in
unit count. A memory budget does not turn quadratic work into a performance
guarantee for arbitrarily many units.

## Independent validation

`tests/test_econ_panel_dependence.py` computes correlations and all three
statistics independently using NumPy per-pair intersections and SciPy reference
tails. Tests cover independent per-unit OLS residuals, unequal common-date
samples, permutation/positive-scale/block-size invariance, large integer and
datetime dates, missing/invalid/degenerate samples, allocated tile dimensions,
the linear balanced-CD path, JSON/LaTeX export, seeded independent-innovation
simulation and common-factor power. These validate the documented implementation;
they do not establish universal finite-sample size or Stata parity.
