# Dynamic panel-data GMM: `xtdpd`, `xtabond`, `xtdpdsys`

Linear dynamic panel models estimated by the generalized method of moments:
the Arellano-Bond (1991) *difference GMM* estimator and the Arellano-Bover
(1995) / Blundell-Bond (1998) *system GMM* estimator, one- and two-step, with
the Windmeijer (2005) finite-sample corrected two-step covariance, the
Arellano-Bond tests for serial correlation and the Sargan, Hansen and
difference-in-Sargan/Hansen tests of the instruments. Instruments are
specified with the semantics of Roodman's `xtabond2` (`gmmstyle()` and
`ivstyle()`), and the reported statistics follow the conventions of
`xtabond2`'s Mata code (its `xtabond2_mata()`, `_H`, `_xform`, `_Orthog`,
`_MakeGMMinsts` and `_ARTests` routines), which was read for this purpose.

Everything is implemented in OpenEconometrics on float64 PyTorch tensors. Lags,
differences, orthogonal deviations and instruments are built by index
arithmetic on a (panel, period) grid, the GMM problems are solved as small
least-squares problems by Householder QR, and every per-panel sum is one
`index_add_` pass. No estimation library runs at fit time. A two-step robust
system GMM model on 50,000 panels by 8 periods fits in about 0.3 s (timings at
the end).

```python
import openecon as oe

ab = oe.xtabond(data=df, y="n", x=["w", "k"], panel="id", time="year",
                twostep=True, robust=True)          # xtabond n w k, twostep vce(robust)
bb = oe.xtdpdsys(data=df, y="n", x=["w", "k"], panel="id", time="year",
                 twostep=True, robust=True)         # xtdpdsys n w k, twostep vce(robust)
full = oe.xtdpd(data=df, y="n", x=["w", "k"], panel="id", time="year",
                gmm=[{"columns": ["n"], "lags": [2, None]},
                     {"columns": ["w", "k"], "lags": [2, None]}],
                iv=[], system=True, twostep=True, robust=True, collapse=True)
print(full.tests["ar2"], full.tests["hansen"])
```

## The model

    y_it = a_1 y_i,t-1 + ... + a_p y_i,t-p + x_it' b + u_i + e_it,    i = 1..G,  t = 1..T_i

The panel effect `u_i` is correlated with the lagged dependent variable, so
pooled OLS and the within estimator are inconsistent for short panels. GMM
removes `u_i` by transforming the equation and instruments the transformed
lagged dependent variable with its own deeper lags, which are uncorrelated
with the transformed error when `e_it` is serially uncorrelated.

**Transformations.**

- *First differences* (default): `Δy_it = y_it - y_i,t-1`, defined only when
  both periods are observed (gaps break differences). `Δe_it` is MA(1), so
  `y_i,t-2` and earlier are valid instruments.
- *Forward orthogonal deviations* (`orthogonal=True`, Arellano-Bover):
  `y*_it = c_it (y_it - mean of the panel's later observations)`, `c_it =
  sqrt(T_it / (T_it + 1))` with `T_it` the number of later observations. It
  uses every later observation (so gaps cost less data) and keeps i.i.d.
  errors i.i.d. As in `xtabond2`, the deviation of period `t` is *dated*
  `t + 1`, so the same instrument lags (`[2, None]` for `y`) are valid under
  both transformations.

A *level observation* is a row whose `p` lagged dependent variables exist;
transformations are formed from consecutive (differences) or later
(deviations) level observations. Rows that serve only as lags or instrument
sources (for example the first period of each panel) are not part of the
estimation sample: `result.dropped_rows` counts them together with rows
excluded for missing values (`extra["rows_lag_only"]` has the first number).

**System GMM** (`system=True`) adds the equation in levels, `y_it = ... + u_i
+ e_it`, instrumented by lagged *differences*, which are uncorrelated with
`u_i` under the Blundell-Bond mean-stationarity assumption. The constant
(`Intercept`) exists only in the level equation; in difference GMM it
differences out and is not estimated (`xtabond2` convention).

## Instruments (`gmm` and `iv`)

GMM-style groups, `xtabond2`'s `gmmstyle()`:

```python
{"columns": ["y", "w"], "lags": [2, None], "equation": "both", "collapse": False}
```

- Transformed equation (`"diff"` and `"both"`): the levels `v_i,t-lo ..
  v_i,t-hi` of every listed variable (`hi=None`: all available lags; a larger
  `hi` is clamped). Uncollapsed, each (period, lag) pair is its own instrument
  column, zero outside its period (the Arellano-Bond block-diagonal instrument
  matrix); `collapse=True` uses one column per lag for all periods (Roodman's
  remedy for instrument proliferation). Unavailable values are zeros.
- `"both"` (default) in system GMM also instruments the level equation with
  the single lagged difference `Δv_i,t-m`, `m = lo - 1` (`m = 0` when `lo = 0`),
  one column per period or one collapsed column. Deeper lagged differences are
  redundant given the transformed-equation instruments (`xtabond2`'s "extra"
  Blundell-Bond instruments). `lags=[0, 0]` with `"both"` would need a forward
  difference and is rejected.
- `"level"` (system GMM) instruments only the level equation, with the
  differences `Δv_i,t-lo .. Δv_i,t-hi`: here the lags count lags *of the
  difference*, exactly as `xtabond2`'s `gmm(v, lag(lo hi) eq(level))` (so the
  standard level instrument of `y` is `{"columns": ["y"], "lags": [1, 1],
  "equation": "level"}`, and `lo = 0` gives the contemporaneous difference).
- `"diff"` instruments only the transformed equation. In difference GMM,
  `"both"` means `"diff"` and `"level"` is an error. Default `lags`: `[1, None]`.
  Negative (forward) lags are not supported.

IV-style groups, `ivstyle()`:

```python
{"columns": ["x"], "equation": "diff", "passthru": False}
```

One column per variable: transformed like the regressors in the transformed
equation (differenced or orthogonal deviations) and untransformed in the
level equation. With `equation="both"` (default) both values share one
column, as in `xtabond2`; listing the variable once with `"diff"` and once
with `"level"` gives two columns. `passthru=True` uses the untransformed level
dated like the transformed row (period `t` for differences, `t + 1` for
orthogonal deviations, which `xtabond2` stores one period late); as in
`xtabond2` it is rejected together with `"both"` in system GMM.

Defaults of `oe.xtdpd`: `gmm=[{"columns": [y], "lags": [2, None]}]` (when
`lags >= 1`) and `iv=[{"columns": x, "equation": "diff"}]` over the regressors
that no `gmm` group instruments (taken as strictly exogenous). Pass `gmm=[]`
or `iv=[]` to switch them off. In system GMM the constant is also an IV-style
instrument of the level equation; `time_dummies=True` adds period dummies
(base: the first period with a level observation) as regressors and as
IV-style instruments of the transformed equation (difference GMM) or of the
level equation (system GMM).

Instrument columns that are zero in every row are dropped, and exactly
collinear instrument columns are removed left to right by a QR screen
(relative length 1e-9, so that the columns of a variable with a large common
level are not mistaken for dependent ones); neither changes the estimator.
`n_instruments` is `xtabond2`'s count: the remaining columns less the rank
deficiency of `Σ Z_i'HZ_i` (with `h=3` in system GMM, `H` has rank `T` per
panel, so a redundant instrument set can make it singular although `Z` has
full rank); `extra` records the planned, zero, collinear and remaining
columns (`instrument_columns`). The stacked instrument matrix is held densely:
more than 2,000 planned columns or more than 256 MiB raise
`instrument_count_too_large` (use `collapse=True` or cap the lags). When the
instruments outnumber the panels a warning is recorded, as `xtabond2` does.

## Estimators

Stack panel `i`'s transformed rows (and level rows in system GMM) into `X_i`,
`y_i`, `Z_i`; `A = Σ Z_i'X_i`, `b = Σ Z_i'y_i`.

- **One-step**: `b1 = (A'W1A)^-1 A'W1 b`, `W1 = (Σ Z_i' H Z_i)^-1`. `H`
  (option `h`, `xtabond2`'s `h()`): with `M` the transformation, `h=3`
  (default) is the covariance of the stacked errors under i.i.d. `e` with unit
  variance (ignoring `u_i`): `M M'` for the transformed block (2 on the diagonal
  and -1 between adjacent differences; the identity for orthogonal
  deviations), `I` for levels and `M` for the transformed-level cross block;
  `h=2` sets the cross blocks to zero; `h=1` is `H = I`. With orthogonal
  deviations the cross block is, as in `xtabond2`, taken from the deviation
  matrix `F_g` of a panel spanning all periods (row of period `t`:
  `c (e_t - mean of e_t+1 .. e_T)`, `c = sqrt(n/(n+1))`, `n` the number of
  later *grid* periods), which equals the panel's own weights only for panels
  observed up to the last period without gaps; since `F_g F_g' = I`, the
  factor `B = F_g'Z_T + Z_L` is formed on every grid cell, with each run of
  cells without a level observation compressed into one row. In difference GMM
  `h=2` and `h=3` coincide; with `h=1` one-step difference GMM is 2SLS on the
  differenced data (coefficients, standard errors and Sargan statistic equal
  `oe.ivregress` on the differenced variables; tested).
- **Two-step** (`twostep=True`): `W2 = (Σ g_i g_i')^-1` with `g_i = Z_i'ê1_i`
  from the one-step residuals (uncentered), `b2 = (A'W2A)^-1 A'W2 b`.

Both are computed as the least-squares problem of `R b` on `R A` with
`R'R = W` (QR, never inverting `A'WA`). Weight matrices are generalized
inverses `S (S Ω S)^+ S`, `S = diag(Ω)^-1/2`: the ordinary inverse when `Ω`
has full rank, and the Moore-Penrose inverse of the correlation-scale matrix
when it does not, with a warning. `Ω` itself is never formed: `Σ Z_i'HZ_i =
B'B` with `B = N'Z` (`N` maps level errors to the stacked rows, `H = N N'` for
`h=3`) and `Σ g_i g_i' = G'G` with the `[G, L]` panel moments, so `R` comes
from a blocked QR of `B` (or `G`) and an SVD of the small triangular factor;
singular values below 1e-11 of the largest count as zero. When `Σ Z_i'HZ_i`
is singular, the moments `Z'u = B'r` lie in its range, so the one-step
estimator does not depend on which generalized inverse is used (it equals
`xtabond2`'s `invsym` result); a singular *two-step* matrix (more
instruments than panels) is not covered by this argument, see the
limitations.

In system GMM the regressors and the outcome are centred at their
level-equation means before estimation (the constant is zero in the
transformed rows, so those rows are unchanged) and the coefficients and
covariance are mapped back exactly (`β = Jβ̃`, `V = JṼJ'`). GMM is
equivariant to this reparametrization; it keeps full precision when `y` has a
large level: with `y` around 1e8 the coefficients agree with a 60-digit
`mpmath` evaluation to 1e-8, where an implementation with explicit float64
inverses is off by more than 100%. The collinearity screen of the regressors
therefore partials out the constant first, as `xtabond2`'s `_rmcoll` does.

## Error variance, covariances and small-sample statistics

The one-step error variance is `xtabond2`'s

    σ² = ê1'ê1 / (c · wttot)

with `ê1` the one-step residuals of the transformed equation (of the *level*
equation when `h=1` in system GMM), `c = 2` for first differences unless
`h=1` (`H = I` then takes the transformed errors as having variance `σ²`) and
`c = 1` for orthogonal deviations or `h=1`, and `wttot` the number of
transformed observations in system GMM with `h > 1` and the reported number
of observations otherwise. Two-step estimation recomputes it from `ê2` for
`sigma_e`.

| option | one-step | two-step |
| --- | --- | --- |
| `nonrobust` (default) | `σ² (A'W1A)^-1` | `(A'W2A)^-1` (known to be downward biased) |
| `robust` | `P1 (Σ g_i g_i') P1'`, `P1 = (A'W1A)^-1 A'W1` (sandwich clustered on panels, no finite-sample factor) | Windmeijer (2005) corrected |

The **Windmeijer correction** accounts for the estimated weight matrix:

    V_c = V2 + D V2 + V2 D' + D V1r D',
    D[:, j] = V2 A'W2 [Σ_i Z_i'(x_ij ê1_i' + ê1_i x_ij') Z_i] W2 Z'ê2,

with `V2 = (A'W2A)^-1`, `V1r` the one-step robust covariance, `ê1`, `ê2` the
one- and two-step residuals and `x_ij` column `j` of `X_i`. `D` is the
derivative of the two-step estimator with respect to the one-step estimate
through `W2`; the tests verify it against a numerical Jacobian and against
`xtabond2`'s expression (`V2 + D V1r D' + (D + D) V2`, symmetrized). It is
computed without per-panel matrices: `Σ_i Z_i'X_i s_i = Z'(X ⊙ s)` and
`Σ_i g_i (X_i'Z_i q)'`.

**`small=True`** (`xtabond2`): t and F reference distributions. After one-step
nonrobust estimation the covariance is multiplied by `wttot/(wttot - k)` and
the degrees of freedom are `N - k`; after robust or two-step estimation (also
two-step nonrobust) by `G/(G-1) · (N-1)/(N-k)` with `G - 1` degrees of freedom
in system GMM with a constant and `G` otherwise (`xtabond2`'s `NGroups -
consopt`). `σ²` and `sigma_e` are scaled by `wttot/(wttot - k)`. `N` is the
number of reported observations, `G` panels, `k` coefficients. Without
`small`, z and chi2 with no finite-sample factors (`xtabond`/`xtdpdsys`
report z). Robust and two-step estimation need at least two panels
(`insufficient_clusters`).

## Tests

- `ar1`, `ar2`, ... (`artests`, default 2): **Arellano-Bond** z statistics for
  zero autocorrelation of order `m` in the *differenced* residuals (always
  differences of level residuals, also after orthogonal deviations or system
  GMM). With `w` the residuals lagged `m` periods and `a_i = Σ_t w_it Δê_it`:

      z = Σ a_i / sqrt( Σ a_i² - 2 w'X* P (Σ_i g_i a_i) + w'X* V X*'w ),

  `X*` the differenced regressors, `P = (A'WA)^-1 A'W` and `V` the covariance
  of the step *before* any small-sample factor, `g_i` that step's per-panel
  moments (both equations in system GMM). After one-step nonrobust estimation
  the homoskedastic form of `xtabond2`'s `_ARTests` replaces `ê_i ê_i'` by
  `σ² H`: `Σ a_i²` becomes `σ² w'DD'w` and `Σ g_i a_i` becomes `σ² (M'Z_T)'(D'w)`
  (the covariance of the differenced errors with the *transformed* equation,
  with `F_g` in place of `M` for orthogonal deviations; `xtabond2` sets the
  covariance with the level equation of system GMM to zero, and so does
  OpenEconometrics); with `h=1`, `H = I`: `σ² w'w` and `σ² Z_T'w`
  with `w` matched to the transformed rows by date. Here `σ²` includes the
  small-sample factor when `small=True` (as in `xtabond2`). A negative
  homoskedastic variance gives a missing statistic with a note. `ar1` is
  expected to reject (differenced i.i.d. errors are MA(1)); a rejection of
  `ar2` invalidates `y_t-2` as an instrument.
- `sargan`: `ĝ1' W1 ĝ1 / σ²`, the minimized one-step criterion (one-step `σ²`,
  no small-sample factor); chi2(`n_instruments` - k). Not robust to
  heteroskedasticity, but not weakened by many instruments. Always reported.
- `hansen` (with `robust` or `twostep`): `ĝ2' W2 ĝ2`, the minimized two-step
  criterion (computed also after one-step robust estimation, as `xtabond2`
  does); chi2(`n_instruments` - k). Robust, but weakened by many instruments.
- Difference tests for the level-equation GMM instruments (`level`, system
  GMM) and for every `gmm` / `iv` group (`gmm1`, `iv1`, ..., `time`; the
  constant is not tested): the model is re-estimated without the subset with
  the matching block of the full model's moment matrix and the difference
  with the full statistic has as many degrees of freedom as the subset has
  columns (`xtabond2`'s procedure). After robust or two-step estimation the
  block is that of `Σ g_i g_i'` (one-step residuals) and the tests are
  `hansen_excl_<g>` / `diff_hansen_<g>`; after one-step nonrobust estimation
  it is that of `σ² Σ Z_i'HZ_i` and the tests are `sargan_excl_<g>` /
  `diff_sargan_<g>` (difference-in-Sargan). A difference can be negative in
  finite samples; it is then reported as 0. Infeasible subsets (the rest does
  not identify the model) report a note instead.
- `model`: Wald chi2 (F with `small`, with the degrees of freedom above) of all
  coefficients except the constant.

## Results

- Coefficients: `Intercept` (system GMM), `L1.y` ... `Lp.y`, the `x` columns,
  time dummies `year[2003]` ...
- `metrics`: `n_groups`, `n_instruments`, `obs_per_group_min`,
  `obs_per_group_avg`, `obs_per_group_max`, `df_model`, `sigma_e`
  (`sqrt(σ²)` of the reported step), and `df_resid` with `small`.
- `nobs`: transformed observations (difference GMM) or level observations
  (system GMM), as `xtabond2` reports them; `sample_positions` are those rows.
- `inference`: `correction` names the covariance and small-sample convention;
  `df_inference` / `df_resid` are the t degrees of freedom with `small`.
- `extra`: transformation, equations, steps, `h`, observation counts of both
  equations, instrument groups with labels and column counts, planned / zero
  / collinear / remaining instrument columns, ranks of both weight matrices,
  the one-step `σ²`, whether the Windmeijer correction was applied and how the
  constant was handled.
- `fitted`: the level equation (`y` against `x'b`, the residual includes `u_i`)
  for system GMM, the transformed equation for difference GMM.

## Stata and OpenEconometrics

| Stata | OpenEconometrics |
| --- | --- |
| `xtabond y x, lags(1)` | `oe.xtabond(data=df, y="y", x=["x"], panel="id", time="t")` |
| `xtabond y x, lags(2) maxldep(3) twostep vce(robust)` | `oe.xtabond(..., lags=2, maxldep=3, twostep=True, robust=True)` |
| `xtabond y x, pre(w) endogenous(v) maxlags(2)` | `oe.xtabond(..., predetermined=["w"], endogenous=["v"], maxlags=2)` |
| `xtdpdsys y x, twostep vce(robust)` | `oe.xtdpdsys(data=df, y="y", x=["x"], panel="id", time="t", twostep=True, robust=True)` |
| `xtabond2 y L.y x, gmm(L.y) iv(x) noleveleq robust twostep` | `oe.xtdpd(..., x=["x"], twostep=True, robust=True)` (defaults) |
| `xtabond2 y L.y x, gmm(L.y, collapse) iv(x, eq(diff)) robust` (system) | `oe.xtdpd(..., system=True, collapse=True, robust=True)` |
| `xtabond2 ..., gmm(L.(y w), lag(1 3)) iv(yr*, eq(level))` | `gmm=[{"columns": ["y", "w"], "lags": [2, 4]}]`, `time_dummies=True` (note: lags are counted from the level, so `L.y` with `lag(1 3)` is `y` with `[2, 4]`) |
| `xtabond2 ..., gmm(y, lag(1 1) eq(level))` | `gmm=[{"columns": ["y"], "lags": [1, 1], "equation": "level"}]` |
| `xtabond2 ..., orthogonal small h(2)` | `orthogonal=True, small=True, h=2` |
| `xtdpd L(0/1).y x, dgmmiv(y) lgmmiv(y) div(x)` | `oe.xtdpd(..., system=True, gmm=[{"columns": ["y"], "lags": [2, None]}], iv=[{"columns": ["x"], "equation": "diff"}])` |
| `estat abond, artests(3)` / `estat sargan` | `artests=3`, `result.tests["ar3"]` / `result.tests["sargan"]` |

Note that `xtabond2` estimates system GMM unless `noleveleq` is given, while
`oe.xtdpd` defaults to difference GMM (`system=False`).

`oe.xtabond` and `oe.xtdpdsys` only build the instrument groups and call
`oe.xtdpd` (the spec's estimator is `xtdpd`): the dependent variable GMM-style
from lag 2 (`maxldep` lags: `y_t-2 .. y_t-1-maxldep`), predetermined variables
from lag 1 and endogenous ones from lag 2 (`maxlags` lags each), `x` and
`instruments` as differenced IV-style instruments (Stata's `div()`);
`xtdpdsys` adds the level equation (`Δy_t-1`, `Δw_t` for predetermined,
`Δv_t-1` for endogenous variables, and the constant). No SPSS procedure
exists for these estimators; EViews' panel GMM / DPD estimator with
Arellano-Bond instruments corresponds to difference GMM here.

## Worked example

```python
import numpy as np, pandas as pd
import openecon as oe

rng = np.random.default_rng(2024)
n, periods, rho, beta = 500, 7, 0.6, 0.4
u = rng.normal(size=n)
y = (1 + 0.5 * beta) * u / (1 - rho) + rng.normal(size=n)    # mean-stationary start
rows = []
for t in range(periods):
    x = rng.normal(size=n) + 0.5 * u                          # correlated with u_i
    y = rho * y + beta * x + u + rng.normal(size=n)
    rows.append(pd.DataFrame({"firm": np.arange(n), "year": 2010 + t, "y": y, "x": x}))
df = pd.concat(rows, ignore_index=True)

ab = oe.xtabond(data=df, y="y", x=["x"], panel="firm", time="year", twostep=True, robust=True)
bb = oe.xtdpdsys(data=df, y="y", x=["x"], panel="firm", time="year", twostep=True, robust=True)
```

Output (rounded; true values `L1.y` 0.6, `x` 0.4):

```
Two-step difference GMM (dynamic panel data) — y
Term  Estimate  Std. error        z
L1.y  0.626796   0.0474299  13.2152
x     0.414245   0.0250018  16.5686
n_groups: 500  |  n_instruments: 16  |  obs 2500 (5 per firm)
Arellano-Bond AR(1): z = -12.636, p = 1.3e-36     AR(2): z = 0.140, p = 0.889
Sargan chi2(14) = 9.570, p = 0.793                Hansen chi2(14) = 10.438, p = 0.729

Two-step system GMM (dynamic panel data) — y
Intercept  -0.00543665   0.0549795  -0.0988849
L1.y          0.601743   0.0334589     17.9846
x             0.406208   0.0239801     16.9394
n_groups: 500  |  n_instruments: 22  |  obs 3000 (6 per firm)
AR(2): z = 0.066, p = 0.947    Hansen chi2(19) = 20.514, p = 0.364
Difference-in-Hansen, GMM instruments for levels: chi2(5) = 9.467, p = 0.092
```

System GMM is more precise (standard error of `L1.y` 0.033 against 0.047)
because the level instruments are informative when `rho` is large; AR(2) does
not reject, so lag-2 instruments are valid. The one-step `oe.xtabond` fit of
the same data reports `diff_sargan_iv1` (chi2(1) = 0.819) instead of the
difference-in-Hansen tests.

## Validation

- `tests/test_econ_dpanel_xtabond2.py` is an independent NumPy oracle written
  from `xtabond2`'s Mata algorithm: every panel on the full grid of T periods,
  orthogonal deviations stored one period late, `H` built as `_H` does,
  instruments and residuals of unused rows zero, the `σ²`, Sargan, Hansen,
  Windmeijer, `small`, AR (both forms) and difference-in-Sargan/Hansen
  formulas of that code. It matches coefficients and full covariance matrices
  (relative 1e-7 / 1e-6), standard errors, p-values, confidence intervals,
  `sigma_e`, instrument counts, Sargan, Hansen and AR statistics for all four
  covariance variants, difference and system GMM, `h = 1, 2, 3`, collapsed
  and uncollapsed groups, `lags=2`, predetermined variables, level-only and
  `lo = 0` groups, `passthru`, time dummies, singular `Σ Z_i'HZ_i`, balanced
  panels, late-starting panels and panels with gaps and early ends (for
  orthogonal deviations with gaps except the homoskedastic AR test and
  `passthru`, see the limitations), the `small` conventions with the F test,
  and the wrappers.
- `tests/test_econ_dpanel_oracle.py` reimplements the estimators with explicit
  per-panel transformation and `H_i` matrices and textbook inverses, including
  unbalanced panels with gaps.
- `tests/test_econ_dpanel.py` checks the Windmeijer term against a numerical
  Jacobian, one-step `h=1` difference GMM against `oe.ivregress` 2SLS on
  differenced data (coefficients, standard errors and the Sargan test) and
  two-step against its cluster-weighted IV-GMM, the orthogonal-deviation
  kernel against the explicit Helmert matrix, `y` around 1e6 against `mpmath`,
  the wrappers, the small-sample factors, consistency on large simulated
  panels and the size of the AR(2), Sargan and Hansen tests by Monte Carlo.
- `tests/test_econ_dpanel_adversarial.py` covers `y` around 1e8 against
  `mpmath`, the exact back-mapping of the centring, invariances (duplicated
  panels, rescaled variables, shifted time, relabelled panels, permuted rows),
  options at their bounds and the error codes of bad inputs;
  `tests/test_econ_dpanel_contract.py` the failure contract, missing data,
  serialization, rendering and the 50,000-panel timing.

## Performance

Apple M-series laptop, default PyTorch CPU threads (one run each):

| rows | specification | k | instruments | time |
| --- | --- | --- | --- | --- |
| 400,000 (50,000 x 8) | difference, two-step robust, collapsed | 2 | 7 | 0.15 s |
| 400,000 | difference, two-step robust, uncollapsed | 2 | 22 | 0.30 s |
| 400,000 | system, two-step robust, collapsed | 3 | 9 | 0.31 s |
| 400,000 | system, orthogonal, two-step robust, collapsed | 3 | 9 | 0.28 s |
| 400,000 | system, two-step robust, uncollapsed | 3 | 29 | 0.67 s |
| 1,000,000 (125,000 x 8) | difference, one-step, collapsed | 10 | 15 | 1.4 s |
| 1,000,000 | difference, two-step robust, collapsed | 10 | 15 | 1.4 s |
| 1,000,000 | difference, two-step robust, uncollapsed | 10 | 30 | 2.0 s |
| 1,000,000 | system, two-step robust, collapsed | 11 | 17 | 2.4 s |
| 1,000,000 | system, orthogonal, two-step robust, collapsed | 11 | 17 | 2.4-3.1 s |
| 480,000 (60,000 x 8) | system, two-step robust, uncollapsed | 11 | 37 | 2.0 s |

Uncollapsed system GMM on one million rows (37 instruments over 1.6 million
stacked rows) exceeds the 256 MiB bound of the dense instrument matrix and
raises `instrument_count_too_large` with the advice to collapse; the peak
memory of the 480,000-row uncollapsed system model is about 2 GB. A Monte
Carlo with 120 replications (150 panels, 5 periods) gives AR(2) z statistics
and Windmeijer-corrected t statistics of `L1.y` with standard deviations near
1 and rejection rates of the AR(2), Sargan and Hansen tests below 15% at the
5% level (`tests/test_econ_dpanel.py`).

## Limitations

- No weights and no clustering on a variable other than the panel (`xtabond2`
  accepts `cluster()` and weights).
- A row with a missing value in any model variable is removed before lags and
  instruments are formed, so its other values are not used as instruments
  either (`xtabond2` still uses available instrument values from such rows).
- `ivstyle`'s `mz`, `gmmstyle`'s `passthru`, `orthogonal` and `split`
  suboptions, negative (forward) lags, `arlevels`, `pca` and `nodiffsargan`
  are not available. A `passthru` instrument whose date is absent is zero
  (`xtabond2` drops that transformed row unless `mz` is given); with
  orthogonal deviations and panels with gaps this can differ from `xtabond2`.
- Orthogonal deviations in panels with gaps: after one-step nonrobust
  estimation `xtabond2`'s homoskedastic AR variance also pairs lagged
  residuals with periods that have a deviation but no difference (gap
  periods); OpenEconometrics uses the differenced rows only, so that test can differ
  slightly (relative 1e-3 in the tests). Everything else, including the
  full-span `F_g` blocks, matches `xtabond2` on unbalanced panels.
- Singular two-step moment matrices (typically more instruments than panels):
  OpenEconometrics uses the Moore-Penrose inverse on the correlation scale, `xtabond2`
  Stata's `invsym`, which drops columns in its own order; the two-step
  estimates and the Hansen statistic then differ. When `Σ Z_i'HZ_i` is
  singular, the difference tests count the degrees of freedom in columns.
- `xtabond2`'s own Wald statistic includes the constant in the quadratic form
  while using `k - 1` degrees of freedom; OpenEconometrics reports the Wald test of the
  slopes (official Stata convention).
- The instrument matrix is stored densely (bounded at 256 MiB); very large
  uncollapsed instrument sets must be collapsed or lag-limited.
- No `predict`/`margins` post-estimation in this family.

## Conventions and how certain they are

Taken from `xtabond2`'s Mata code (read for this verification, high
confidence that OpenEconometrics reproduces that code, verified by the oracle):
`σ²` (including `h=1`), the Sargan and Hansen statistics and their degrees of
freedom, the robust and Windmeijer covariances, the `small` factors and
degrees of freedom, both forms of the AR test (including the zero level
block and the use of the covariance before and `σ²` after the small-sample
factor), the difference-in-Sargan/Hansen procedure, the `gmmstyle` and
`ivstyle` semantics (including `eq(level)` lags and `passthru` dating) and the
`H` matrices (including the full-span deviation matrix for orthogonal
deviations). `provenance["stata_parity_validated"]` stays `False`: no Stata
output was compared.

Uncertain:

- Stata's official `xtabond` / `xtdpd` / `xtdpdsys` may use conventions that
  differ from `xtabond2` (for example `xtabond` reports a constant in
  difference GMM and its own `σ²`); OpenEconometrics follows `xtabond2` throughout.
- `time_dummies=True` instruments the dummies in the level equation only in
  system GMM (in the transformed equation in difference GMM); an `xtabond2`
  user writing `iv(yr*)` gets one shared column in both equations.
- The reading of `maxldep` / `maxlags` as numbers of lags in the wrappers.
- The default `iv` group of `oe.xtdpd` uses `equation="diff"` (Stata's
  `div()`), not `xtabond2`'s default `eq(both)`.
