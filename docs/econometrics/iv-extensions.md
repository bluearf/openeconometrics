# IV extensions: JIVE/UJIVE, Lewbel instruments, shift-share AKM and MOP cutoffs

Staged scope of [MARKET-183 / GitHub #110](https://github.com/bluearf/openecon/issues/110).
Everything runs in OpenEconometrics on float64 PyTorch tensors; the sources
named below were read as author code and transcribed, they do not run at fit
time. No vendor (Stata) execution was performed: the comparisons are against
explicit NumPy algebra, author-code transcriptions, Python linearmodels 7.0 /
statsmodels and R AER/sandwich. `provenance["stata_parity_validated"]` stays
`False`.

| function | kind | reference |
| --- | --- | --- |
| `oe.jive(method="jive1" \| "jive2" \| "ujive")` | registered estimator `jive` | Angrist, Imbens & Krueger (1999); Kolesár (2013); [ManyIV `R/ujive.R`](https://github.com/kolesarm/ManyIV/blob/master/R/ujive.R) |
| `oe.ivlewbel(...)` | registered estimator `ivlewbel` | Lewbel (2012, JBES 30:67–80); [ivreg2h.ado](http://fmwww.bc.edu/repec/bocode/i/ivreg2h.ado) |
| `oe.shift_share_iv(...)`, `oe.shift_share_instrument(...)` | procedures (`TableSet`) | Adão, Kolesár & Morales (2019, QJE 134:1949–2010); [ShiftShareSE `R/ols.R`, `R/iv.R`](https://github.com/kolesarm/ShiftShareSE) |
| `oe.weakivtest(...)` | procedure (`TableSet`) | Montiel Olea & Pflueger (2013, JBES 31:358–369); [weakivtest.ado](http://fmwww.bc.edu/repec/bocode/w/weakivtest.ado) |

The examples below share this data (all printed numbers were produced by this code):

```python
import numpy as np, pandas as pd, openecon as oe
rng = np.random.default_rng(7)
n = 600
z = rng.normal(size=(n, 8)); v = rng.normal(size=n)
df = pd.DataFrame(z, columns=[f"z{j}" for j in range(8)])
df["x1"] = rng.normal(size=n)
df["d"] = z @ np.full(8, 0.12) + 0.5 * df.x1 + v
df["y"] = 1 + 0.5 * df.d - df.x1 + 0.8 * v + rng.normal(size=n) * (1 + 0.5 * np.abs(z[:, 0]))
inst = [f"z{j}" for j in range(8)]
```

## `oe.jive` — jackknife IV

    oe.jive(*, data, y, endog, instruments, x=None, method="jive1", covariance=None,
            cluster=None, small=False, categorical=None, intercept=True, missing="raise",
            alpha=0.05)

`Q = [1 x instruments]`, `h_i` the diagonal of its projection (squared row norms
of one thin QR factor; no n-by-n matrix), `X = [1 x endog]`, `pi = (Q'Q)^-1 Q'X`.

- **JIVE1** (AIK 1999): every column of `X` is replaced by its leave-one-out
  prediction `xhat_i = (q_i'pi - h_i x_i)/(1 - h_i)` (exogenous columns are
  reproduced exactly); `b = (Xhat'X)^-1 Xhat'y`. The endogenous coefficient
  equals ManyIV's `jive1`.
- **JIVE2** (AIK 1999): `xhat_i = (q_i'pi - h_i x_i)/(1 - 1/N)` for every column
  (exogenous columns become `(1-h_i)x_i/(1-1/N)`, AIK's literal definition).
- **UJIVE** (Kolesár 2013; ManyIV's `old ujive`/`UJIVEold`, one endogenous
  regressor): `dhat = (I-D_Q)^-1(H_Q-D_Q)d - (I-D_1)^-1(H_1-D_1)d`, the
  leave-one-out prediction of `d` from all instruments minus that from
  `[1 x]`; `b = dhat'y/dhat'd`. Only the endogenous coefficient is reported
  (`extra["partialled_out"]`), residual `M_1(y - d b)` as in ManyIV. The
  development ManyIV's current `ujive` row (`HD - MD (dX - dW)/(1 - dX)`) is a
  different estimator and is not implemented.

Covariance: the IV sandwich with the jackknife instrument `A` and
`B = (A'X)^-1` (ManyIV's "textbook" standard error): `nonrobust`
`s^2 B A'A B'`; `robust` (default) `B (sum u_i^2 a_i a_i') B'`; `cluster` (one
column) cluster sums times `G/(G-1)`. `small=True`: `RSS/(N-K)`, `N/(N-K)`,
`(N-1)/(N-K)`, t with `N-K` (`G-1`) degrees of freedom; otherwise z.
`K` counts every coefficient, also for UJIVE.

```python
for method in ("jive1", "jive2", "ujive"):
    oe.jive(data=df, y="y", x=["x1"], endog=["d"], instruments=inst, method=method)
# d: jive1 0.399533 (0.200941), jive2 0.40045 (0.200686), ujive 0.420908 (0.194211)
# robust 2SLS for comparison: 0.505106 (0.17348)
```

Unsupported, explicitly: weights; leave-one-cluster-out jackknife (leave-out
is by observation even with a cluster covariance, so own-cluster first-stage
dependence is not removed); many-instrument (Bekker/Chao et al.) and
heterogeneous-effect (Evdokimov–Kolesár) variance corrections; UJIVE with
several endogenous regressors (`unsupported_domain`).

Errors: `unit_leverage` (JIVE1/UJIVE with an observation of leverage one —
for example the only member of an instrument category; nothing is dropped
automatically, unlike ManyIV's `dropleverage`), `underidentified`,
`insufficient_observations`, `perfect_fit`, `constant_outcome`,
`invalid_spec`, `missing_values`.

**Validation** (`tests/test_econ_iv_extensions.py`): JIVE1 against a
brute-force loop of N leave-one-out regressions (all covariances, `small`
on/off, two endogenous regressors; max abs deviation 2.4e-12, rel 1.0e-13);
JIVE2 against the dense hat-matrix formula; UJIVE against a NumPy
transcription of ManyIV `tests/testthat/test_ujive.R::jive.fit` and against
brute-force leave-one-out predictions (rel 6e-15); JIVE1 equals OLS when the
instruments reproduce the regressor.

## `oe.ivlewbel` — heteroskedasticity-based generated instruments

    oe.ivlewbel(*, data, y, x, endog, instruments=None, generators=None, method="2sls",
                first_step="corrected", covariance=None, cluster=None, small=False,
                categorical=None, missing="raise", alpha=0.05)

Lewbel (2012): `y = X1'b1 + X2'b2 + e1`, `X2_m = X1'g_m + e2_m`, identified by
`cov(Z, e1 e2) = 0` and `cov(Z, e2^2) != 0` for `Z = generators` (default every
non-constant exogenous regressor, which must be numeric columns of `x`). The
instruments `(Z_j - mean Z_j) * e2hat_m` (`e2hat` = OLS residuals on all of
`[1 x]`, estimation-sample means) named `<endog>_<z>_g` are added to the
optional external `instruments`; 2SLS or two-step efficient GMM. This is
ivreg2h's generated-only branch (`reg e inexog_t`, ado line 443); its
generated-plus-external branch regresses on the `z()` list instead (line
667), which agrees when `z()` is every included exogenous regressor.

`first_step`:

- `"ignored"` — ivreg2h's convention: the generated instruments are data and
  every number (covariance, first stage, overidentification, endogeneity,
  weak identification) is `ivregress` on the augmented instrument list
  (ivregress finite-sample conventions; `nonrobust` allowed).
- `"corrected"` (default; `robust` or `cluster`): the instrument moments
  `h_i u_i` are replaced by influence functions that include the estimated
  `g_m` and generator means,
  `r_{i,(m,j)} = G_{i,(m,j)} u_i - e2hat_{im} [P_X1((Z_j - Zbar_j) o u)]_i - (e2hat_m'u/N)(Z_ij - Zbar_j)`.
  2SLS: `V = B C'(sum r r')C B'`, `B = (X'P_H X)^-1`, `C = (H'H)^-1 H'X`. GMM: weight
  matrix = inverse corrected moment covariance at the 2SLS residuals, efficient
  `(X'H S^-1 H'X)^-1`, Hansen J with `L-K` degrees of freedom. Cluster sums
  replace White sums; `small` and `G/(G-1)` as in ivregress. Weak-identification
  statistics are reported conditional on the generated instruments (labelled);
  other ivregress diagnostics are not, because they ignore the first step.

`tests["lewbel_bp_<endog>"]`: Koenker's studentized Breusch–Pagan `N R^2` of
`e2hat^2` on `[1 Z]`, chi2(p) — a check of Lewbel's relevance condition.

```python
n = 2000
x1, x2, f = rng.normal(size=(3, n))          # continuing the rng above
e2 = np.exp(0.5 * x1) * rng.normal(size=n) + f
lw = pd.DataFrame({"x1": x1, "x2": x2, "p": 1 + x1 + e2})
lw["y"] = 1 + x1 - x2 + 0.5 * lw.p + f + rng.normal(size=n)
oe.ivlewbel(data=lw, y="y", x=["x1", "x2"], endog=["p"])
# p 0.531411, se 0.0504098 (corrected) vs 0.052231 (first_step="ignored");
# Koenker BP chi2(2) = 147.412, Cragg-Donald F(2, 1995) = 279.168
```

Unsupported: weights, no constant, HAC covariance, categorical generators
(give explicit dummies), LIML, panel/fixed effects (ivreg2h's `ivreg2hfe`).
Errors: `unsupported_covariance` (nonrobust with the correction),
`invalid_spec`, `underidentified`, `no_endogenous_regressors`,
`singular_weight_matrix`.

**Validation**: `first_step="ignored"` against linearmodels 7.0 IV2SLS
(unadjusted, robust, robust debiased, clustered debiased; IVGMM two-step
coefficients) and R AER `ivreg` + sandwich `vcovHC(HC0)`, `vcovCL(HC0/HC1)`
from `scripts/generate_iv_extension_oracles.{py,R}` (fixture
`tests/fixtures/iv_extensions/lewbel.json`; max abs 6.9e-17, rel 6.1e-15);
`first_step="corrected"` against the sandwich of the stacked exactly
identified first-step + 2SLS estimating equations with a finite-difference
Jacobian, and efficient GMM built from those stacked influence rows (robust
and cluster, with and without external instruments; rel 4.4e-11); the BP
statistic against statsmodels `het_breuschpagan(robust=True)`.

## `oe.shift_share_iv` — Bartik designs with AKM inference

    oe.shift_share_instrument(*, shares, shocks, unit="unit", sector="sector",
                              share="share", shock="shock")
    oe.shift_share_iv(*, data, y, unit, shares, shocks, endog=None, x=None,
                      sector="sector", share="share", shock="shock", weights=None,
                      cluster=None, shock_cluster=None, beta0=0.0, alpha=0.05,
                      intercept=True, missing="raise", max_cells=50_000_000)

Long-format inputs: `data` has one row per `unit`; `shares` holds
`(unit, sector, share)` rows (absent pairs are zero; negative shares,
duplicate pairs, units without share rows and sectors without shocks are
errors); `shocks` holds `(sector, shock)` and optionally a `shock_cluster`
column. `X_i = sum_s w_is g_s`. Without `endog` the shift-share variable is the
regressor (`reg_ss`); with `endog` it instruments it (`ivreg_ss`). Controls
`Z = [1 x]`; optional region `weights` (weighted least squares).

Formulas transcribe ShiftShareSE (see `shift_share.py` docstring):
`homoskedastic`, `ehw`, `region_cluster` (with `cluster`), `akm`
(`sqrt(sum_s cR_s^2)/|RX|`, `cR_s = hX_s sum_i W_is w_i e_i`, `hX` the
coefficients of the residualized shift-share variable on the share matrix,
summed within `shock_cluster` first) and `akm0` (null-imposed residuals at
`beta0` for the p-value; confidence set by test inversion: a bounded
interval, the complement of an interval — rows with `None` endpoints in
`tables["akm0_confidence_set"]` and `attrs["akm0_set"]` — or the real line).
ShiftShareSE's factors are kept: OLS EHW `n/(n-p)` and region cluster
`G/(G-1)(n-1)/(n-p)`; IV EHW, region cluster, AKM and AKM0 without factors;
standard normal reference. Collinear share columns are dropped left to right
in `shocks` order and listed in `attrs["omitted_sectors"]`.
`attrs["effective_shock_count"]` is the inverse Herfindahl index of the
exposure weights `s_s = sum_i e_i w_is / sum_i e_i sum_k w_ik` (Borusyak, Hull &
Jaravel 2022). A note is added when share sums vary and are not spanned by the
controls (incomplete shares: control for the sum of shares).

```python
regions, sectors = 300, 60
raw = rng.gamma(0.5, size=(regions, sectors)) * (rng.random((regions, sectors)) < 0.4) + 1e-3
w = raw / raw.sum(1, keepdims=True)
g = rng.normal(size=sectors)
common = w @ rng.normal(size=sectors)
units = pd.DataFrame({"region": np.arange(regions), "c1": rng.normal(size=regions)})
units["emp"] = 1 + 2.0 * (w @ g) + 0.3 * units.c1 + common + rng.normal(size=regions)
units["wage"] = 0.5 * units.emp + common + rng.normal(size=regions)
rows, cols = np.nonzero(w)
shares = pd.DataFrame({"region": rows, "sector": cols, "share": w[rows, cols]})
shocks = pd.DataFrame({"sector": np.arange(sectors), "shock": g})
out = oe.shift_share_iv(data=units, y="wage", endog="emp", x=["c1"], unit="region",
                        shares=shares, shocks=shocks)
```

```
          method  estimate  std_error  statistic  p_value  ci_low  ci_high   ci_type
0  homoskedastic    0.5958     0.1205     4.9447    0.000  0.3596   0.8319  interval
1            ehw    0.5958     0.1092     5.4555    0.000  0.3817   0.8098  interval
2            akm    0.5958     0.1277     4.6644    0.000  0.3454   0.8461  interval
3           akm0    0.5958     0.1551     2.5131    0.012  0.2348   0.8429  interval
```

(`effective_shock_count` 58.71.) The AKM0 `std_error` is half the interval
length over the normal quantile (ShiftShareSE); its p-value uses the
null-imposed standard error. Unsupported: several endogenous regressors or
shift-share instruments, BHJ shock-level regression output, panel
(stacked-period) share structures beyond what the user encodes as units.
Errors: `duplicate_units`, `duplicate_shares`, `duplicate_shocks`,
`missing_shares`, `missing_shocks`, `negative_shares`, `missing_values`,
`collinear_shift_share`, `underidentified`, `singular_shares`,
`insufficient_clusters`, `resource_limit`, `invalid_option`.

**Validation**: a NumPy transcription of `reg_ss.fit`/`ivreg_ss.fit`
(R ShiftShareSE is not installed in the oracle environment) for OLS and IV,
weighted and unweighted, with region and shock clusters, bounded and
complement AKM0 sets, and collinear sectors (max rel 1.0e-14); with one sector
per region and only a constant, AKM equals the uncorrected sector-clustered
variance.

## `oe.weakivtest` — Montiel Olea–Pflueger effective-F critical values

    oe.weakivtest(*, data, y, endog, instruments, x=None, covariance="robust",
                  cluster=None, time=None, lags=None, alpha=0.05,
                  tau=(0.05, 0.10, 0.20, 0.30), intercept=True, missing="raise")

One endogenous regressor. Transcribes the authors' Stata `weakivtest`
(Pflueger & Wang; version with the 2018 cluster/weight and 2023 reghdfe fixes):
residualize on `[1 x]`, orthonormalize the instruments (`Zs'Zs = N I`),
`Omega` and `W = S N/(N-K-k1)` from the residuals of `y` and `d` on `Zs` with
the avar-style moment covariance (`nonrobust` `(U'U/N) kron I`, HC0, cluster
sums, Bartlett HAC with `lags`), `F_eff = a F_CD K Omega_22 / tr(W2)` with
`a = N/(N-1)(G-1)/G` for clusters. The TSLS and LIML Nagar-bias bounds
`B_e` (weakivtest's `Bmaxfunction` and `BmaxLIML`) are maximized over the
whole real line (`b = tan(t)` with the scale-invariant limit at `t = +-pi/2`,
2048-point grid plus golden-section refinement; weakivtest searches a bounded
interval with Nelder–Mead). Critical values: Patnaik
`c = q_{1-alpha}(chi2_{K_eff}(x K_eff))/K_eff`,
`K_eff = 2(1+2x)/(2 sum l^2 + 4 x max l)` (`l` normalized eigenvalues of `W2`)
with `x = B_e/tau` (generalized TSLS, LIML) or `x = 1/tau` (simplified,
conservative for TSLS), from an exact noncentral chi-square quantile (Poisson
mixture of `engines.distributions.gamma_p`, safeguarded Newton). weakivtest
uses `x = 3.33 B` for tau = 30%; pass `tau=[1/3.33]` to reproduce it.

`tables["critical_values"]` (tau × tsls, liml, simplified) and
`tables["details"]` (x, K_eff, critical value, `reject_weak = F_eff > c`);
`attrs`: `effective_f`, `effective_f_unadjusted` (equal to `oe.effective_f`, the
N-divisor statistic without degrees-of-freedom factors), `cragg_donald_f`,
`b_tsls`, `b_liml`, their maximizing `b` (`None` = at infinity), `w_matrix`,
`omega`. The existing statistic-only `oe.effective_f` is unchanged and still
returns no cutoffs; homoskedastic Stock–Yogo values are never attached to the
robust effective F.

```python
r = oe.weakivtest(data=df, y="y", x=["x1"], endog=["d"], instruments=inst)
# effective_f 11.668, B_TSLS 0.8182, B_LIML 0.1756
```

```
    tau    tsls   liml  simplified
0  0.05  23.168  7.423      27.378
1  0.10  13.400  4.956      15.622
2  0.20   8.198  3.594       9.389
3  0.30   6.349  3.098       7.183
```

Unsupported: several endogenous regressors (`unsupported_target`), weights
(weakivtest's `aweight` branch), absorbed fixed effects, the Monte Carlo
(non-Patnaik) critical value and the OLS-benchmark variant of Windmeijer
(2025). Errors: `unsupported_target`, `invalid_option`,
`unsupported_covariance`, `invalid_spec` (HAC without `time`/`lags`),
`singular_first_stage`, `insufficient_observations`.

**Validation**: a NumPy/SciPy transcription of the Mata code (W with a
symmetric orthonormalization, `Bmaxfunction`/`BmaxLIML` maximized on a
sinh-spaced grid with the limits and bounded Brent refinement,
`scipy.stats.ncx2.ppf`) for iid, HC0, cluster and HAC (max rel 2.0e-14 on
F_eff, B and every critical value); `F_eff` equals ivregress's Cragg–Donald
F under `nonrobust`; invariance to instrument rescaling/rotation; with one
instrument every critical value equals the upper-alpha quantile of
`chi2_1(1/tau)` (Windmeijer 2025, [arXiv 2309.01637](https://arxiv.org/html/2309.01637v3)
Sec. 4), 23.1085 at tau = 10%. The documented weakivtest example
(Yogo 2004 `Data_USAQ.dta`) was not executed: the data file was not downloaded
and no Stata output values were available as text, so no published numerical
example is claimed.

## Reproducing the fixtures

    python scripts/generate_iv_extension_oracles.py --r --rscript Rscript

needs numpy, pandas, statsmodels, linearmodels and (with `--r`) R with AER,
sandwich and jsonlite. Regular tests read only the committed JSON.

    OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_econ_iv_extensions.py
