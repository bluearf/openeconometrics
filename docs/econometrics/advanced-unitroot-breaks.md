# Break unit-root, explosive-root and robust multiple-break tests

This page covers four method families added to the unit-root family
(`src/openecon/econometrics/unitroot/`):

| Function | Method | Stata / RATS / R counterpart |
| --- | --- | --- |
| `oe.lee_strazicich` | Lee–Strazicich minimum LM unit-root test, one or two endogenous breaks, models A (crash) and C (level and trend break) | authors' GAUSS `LS_One.g` / `LS_two.g`; RATS `@LSUNIT`; Stata `lsmlm` |
| `oe.clemente` | Perron–Vogelsang (one shift) and Clemente–Montañés–Reyes (two shifts) additive-outlier (AO) and innovational-outlier (IO) unit-root tests | Stata `clemao1`, `clemao2`, `clemio1`, `clemio2`; Perron–Ng RATS `aom0alph`/`iom0alph` |
| `oe.psy` | Phillips–Shi–Yu SADF and GSADF tests with the BSADF date-stamping sequence and an episodes table | R `psymonitor`, `exuber` |
| `oe.bai_perron(errors=...)` | Bai–Perron heteroskedasticity- and HAC-robust supF/UDmax/WDmax, sequential supF(ℓ+1\|ℓ) tests and break-date confidence intervals | Perron's GAUSS `BRCODE.SRC`; R `mbreaks`, `strucchange` |

All statistics are computed in OpenEconometrics on float64 CPU PyTorch tensors;
no estimation library runs at test time. Every procedure is a test: it
registers no estimator and returns a `core.TableSet` whose `.attrs` hold the
statistic, p-value (or `None`), critical values, an `inference_status`, the
settings and notes; every table renders with `str()` and `.to_latex()` and the
attributes are JSON-safe. Series must be complete and regularly spaced (rows in
order, or sorted by `time` with consecutive integer periods or datetimes);
missing values, repeated periods, gaps, constant or deterministic series,
singular regressions and over-budget requests raise `AnalysisError` with a
stable code. No row is dropped.

```python
import openecon as oe

ls = oe.lee_strazicich(df, "lgdp", time="year", breaks=2, model="break")
ls["test"], ls.attrs["break_periods"], ls.attrs["critical_values"]

cmr = oe.clemente(df, "rate", time="month", model="ao", breaks=2,
                  calibration="monte_carlo", replications=499, seed=7)

bubble = oe.psy(df, "price_dividend", time="month", seed=1)
bubble["tests"], bubble["episodes"]

bp = oe.bai_perron(df, "inflation", ["lag_inflation"], time="quarter",
                   max_breaks=5, trim=0.15, errors="hac")
bp["sequential"], bp["date_intervals"]
```

## Lee–Strazicich minimum LM test

The data-generating process is `y_t = δ'Z_t + X_t`, `X_t = βX_{t-1} + ε_t`.
Model A (`model="crash"`) has `Z_t = [1, t, D_jt]`, model C (`model="break"`)
adds `DT_jt`, with `D_jt = 1(t ≥ TB_j + 1)` and `DT_jt = (t − TB_j) D_jt`.
With `δ̃` from regressing `Δy_t` on `ΔZ_t` (t = 2..T), `S̃_t = y_t − ψ̃ − Z_t δ̃`
and `ψ̃ = y_1 − Z_1 δ̃`, the LM regression is

`Δy_t = δ'ΔZ_t + φ S̃_{t−1} + Σ_{j≤k} γ_j ΔS̃_{t−j} + u_t`, t = k+2..T,

and the statistic is the minimum over break configurations of the t ratio of
φ. H0 is a unit root **with** breaks; rejection indicates trend-break
stationarity. Small values reject.

The search reproduces the authors' GAUSS programs (Junsoo Lee,
`LS_One.g`/`LS_two.g`, archived at
<http://web.archive.org/web/2012/http://www.cba.ua.edu/~jlee/gauss/>):

* break dates `TB ∈ [round(0.1T), round(0.9T)]` (GAUSS rounding, halves away
  from zero), with the first candidate raised to `maxlag + 3` when
  `round(0.1T) < maxlag + 2`; `trim` changes 0.1;
* the second break at least 2 (model A) or 3 (model C) periods after the first;
* at **every** configuration the lag order is chosen general-to-specific:
  the largest `k ≤ maxlag` (default 8) whose last augmentation term has
  |t| > 1.645, each `k` fitted on its own sample t = k+2..T; 0 if none
  (`lags=` fixes `k` instead);
* ties keep the first configuration in scan order.

The impulse dummies `B_jt = ΔD_jt` fit one observation each, so they are
applied as row masks in the batched QR (exactly equivalent; the final
regression table shows them explicitly). The statistic is invariant to the
origin and unit of the series. `break_indices` are 0-based positions of `TB`
(the last observation of the earlier regime) and `break_periods` their labels.

**Critical values** (no p-value is tabulated; `p_value` is `None`):

| Test | 1% | 5% | 10% | Source |
| --- | ---: | ---: | ---: | --- |
| one break, model A | −4.239 | −3.566 | −3.211 | Lee and Strazicich (2013), Table 1 |
| two breaks, model A | −4.545 | −3.842 | −3.504 | Lee and Strazicich (2003), Table 2 |

| λ (one break, model C) | 1% | 5% | 10% |
| --- | ---: | ---: | ---: |
| 0.1 | −5.11 | −4.50 | −4.21 |
| 0.2 | −5.07 | −4.47 | −4.20 |
| 0.3 | −5.15 | −4.45 | −4.18 |
| 0.4 | −5.05 | −4.50 | −4.18 |
| 0.5 | −5.11 | −4.51 | −4.17 |

| (λ1, λ2) (two breaks, model C) | 1% | 5% | 10% |
| --- | ---: | ---: | ---: |
| (0.2, 0.4) | −6.16 | −5.59 | −5.27 |
| (0.2, 0.6) | −6.41 | −5.74 | −5.32 |
| (0.2, 0.8) | −6.33 | −5.71 | −5.33 |
| (0.4, 0.6) | −6.45 | −5.67 | −5.31 |
| (0.4, 0.8) | −6.42 | −5.65 | −5.32 |
| (0.6, 0.8) | −6.32 | −5.73 | −5.32 |

All values were simulated by the authors with T = 100. The one-break table
was typed from the journal page
(<https://www.accessecon.com/Pubs/EB/2013/Volume33/EB-13-V33-I4-P234.pdf>,
p. 2488) and matches Aptech tspdlib `src/lm_1br.src`; its note says model-C
values are symmetric in λ and 1 − λ and "can be interpolated", so the estimated
`λ = TB/T` is folded to `min(λ, 1 − λ)` and interpolated linearly between rows
(`inference_status="published_table_interpolated"`). The two-break table comes
from tspdlib `src/lm_2br.src` (<https://github.com/aptech/tspdlib>), which
cites Lee and Strazicich (2003, Table 2); the REStat text itself was not
retrievable, so no author interpolation rule could be confirmed. The **nearest
published cell** to the estimated (λ1, λ2) is used and flagged
(`inference_status="published_table_nearest_cell"`, `critical_value_rule`); this
is an OpenEconometrics convention, not the authors'. Treat these as
finite-sample T = 100 values. With the authors' per-configuration lag rule the
bounded check below rejects about 10% (model A) and 15% (model C) of iid
random walks at the nominal 5% level; with `lags=0` it is close to nominal.

## Clemente–Montañés–Reyes / Perron–Vogelsang mean-shift tests

With `DU_jt = 1(t > TB_j)` and `DTB_jt = 1(t = TB_j + 1)`:

* **IO** (`model="io"`): `y_t = μ + Σ_j (d_j DU_jt + δ_j DTB_jt) + α y_{t−1} +
  Σ_{i≤k} c_i Δy_{t−i} + e_t`, t = k+2..T.
* **AO** (`model="ao"`): first `y_t = μ + Σ_j d_j DU_jt + ỹ_t` on t = 1..T, then
  `ỹ_t = Σ_j Σ_{i=0..k} ω_ji DTB_j,t−i + α ỹ_{t−1} + Σ_{i≤k} c_i Δỹ_{t−i} + e_t`.

The statistic is the minimum over candidate dates of `(α − 1)/se`; H0 is a unit
root (shifts enter as impulses), H1 stationarity around one (`breaks=1`,
Perron and Vogelsang 1992) or two (`breaks=2`, Clemente et al. 1998) shifted
means. The regressions and the t-sig lag rule follow Perron and Ng's RATS
model-0 procedures (`iom0alph.src`, `aom0alph.src` in
<https://blogs.bu.edu/perron/files/2020/05/unit-root.zip>): at every candidate
date the largest `k ≤ maxlag` whose last lag has |t| ≥ 1.645, else 0.
Candidate dates follow Baum's Stata commands: `ntrim = int(trim·T + 0.49)`
(default `trim=0.05`), `TB ∈ [ntrim + 1, T − ntrim]`, additionally
`TB ≥ maxlag + 2` and `TB ≤ T − 2` (IO) or `T − maxlag − 2` (AO), and
`TB_2 ≥ TB_1 + separation` (default 2). Duplicate AO impulses (when two shifts
are within `k` periods) span one column.

Differences from Baum's Stata commands, which are **not** reproduced: Stata
searches the date with a fixed lag of one and then chooses `k` at that date by
F tests at the confidence level, and runs some first-stage regressions on the
trimmed sample. Stata prints only 5% critical values (−3.560 AO / −4.270 IO
for one shift, attributed to Perron–Vogelsang 1992 Tables 3–4 at T = 150, and
−5.490 for two shifts); these second-hand single values are not used.

**Inference** (`calibration`):

* `"asymptotic"` (default), one shift: the asymptotic distribution of
  Perron–Vogelsang's model-0 min t_α, common to the IO and AO statistics,
  tabulated at percentiles 1..99 in Tim Vogelsang's PVAL program (`FF1.C`,
  function f1, in the Perron–Ng archive above). Critical values 1% −4.949, 5%
  −4.444, 10% −4.194; the p-value interpolates linearly between percentiles as
  that program does and is bounded to [0.01, 0.99] with a note. The program
  accompanies procedures that search all dates, so a trimmed search is
  conservative under it.
* `"asymptotic"`, two shifts: Clemente, Montañés and Reyes (1998) critical
  values could not be obtained from a primary source (paper or author code).
  `critical_values` and `p_value` are `None` and `inference_status` explains
  why; no value is invented.
* `"monte_carlo"`: the complete procedure (date search and lag rule) is
  repeated on `replications` independent Gaussian random walks of the same
  length with a local seeded generator (draw-major, batches of 16);
  `p = (1 + #{null ≤ observed})/(B + 1)`, empirical 1/5/10% quantiles. Failed
  draws abort the call; the denominator is recorded.
* `"none"`: statistic only.

## Phillips–Shi–Yu SADF, GSADF and BSADF

For a window of levels `y_s..y_e`, `ADF(s, e)` is the t ratio of `b` in
`Δy_t = a + b y_{t−1} + Σ_{j≤lags} c_j Δy_{t−j} + u_t` (t = s+1+lags..e).
With minimum window `w` (levels; default `floor(T(0.01 + 1.8/√T))`, i.e.
`r0 = 0.01 + 1.8/√T`):

* `BSADF(e) = max_{s ≤ e−w+1} ADF(s, e)`;
* `SADF = max_e ADF(0, e)`;
* `GSADF = max_e BSADF(e)` (Phillips, Shi and Yu 2015, eqs. 4–6);
* the whole-sample `ADF` is also reported.

H0 is a unit root (random walk with asymptotically negligible drift); H1 mildly
explosive episodes; large values reject. `lags` is fixed (default 0, as in
PSY's simulations and in psymonitor/exuber).

**Calibration.** `"monte_carlo"` (default) simulates `y_t = d T^{−η} + y_{t−1} + e_t`
with `d = η = 1`, `e ~ N(0, 1)`, `y_0 = 0` (PSY 2015, section 3; psymonitor's
`cvPSYmc`), same `T`, `w` and `lags`, `replications` draws (default 999) with a
local seeded generator. SADF/GSADF critical values are empirical 90/95/99%
quantiles, p-values `(1 + #{null ≥ observed})/(B + 1)`, and the date-stamping
critical value at each end date is the quantile of the simulated `BSADF(e)` at
that date. `"wild_bootstrap"` implements Phillips and Shi (2020, section 5;
<https://cowles.yale.edu/sites/default/files/2022-08/d2152.pdf>): fit
`Δy_t = a + Σ φ_j Δy_{t−j} + e_t` under the null, generate
`Δy*_t = Σ φ̂_j Δy*_{t−j} + w_t e_l(t)` with `w_t ~ N(0, 1)` and `e_l` drawn
with replacement, initial values `y*_i = y_i` (i ≤ lags + 1), and take the
`level` quantile of `max BSADF` over the first `horizon` monitoring points as a
family-wise critical value for the whole sequence (default horizon: every
monitoring point, which makes it the GSADF critical value). The AR recursion is
solved as one lower-triangular system. psymonitor 0.0.2's `cvPSYwmboot` draws a
single normal weight for all observations (`matrix(rnorm(1), ...)`); this
implementation draws one per observation as the paper describes.

**Date stamping** follows PSY's first-crossing rules: an episode originates at
the first end date with `BSADF > cv` and terminates at the first end date at
least `ceil(min_duration)` observations later with `BSADF < cv`
(`min_duration = δ log T`, default δ = 1); the search then restarts after the
termination. `episodes` lists origination, termination (`None` and
`ongoing=True` if the sample ends first), the last exceedance, duration and peak
BSADF; `sequence` lists every BSADF, its critical values, the exceedance flag
and the start of the maximizing window.

**Numerics.** All windows are evaluated from cumulative cross products per
window start, with levels measured from `y_s` (the intercept makes the statistic
invariant to that shift, which keeps the sums well scaled); each window solves
its small normal-equation system with a Cholesky factor of the diagonally
scaled Gram matrix (closed form for `lags = 0`). Windows whose regression is
singular or exact are excluded from the maxima and counted
(`undefined_windows`). Work is bounded (`max_work`).

**Published check.** PSY (CFDP 1843, Table 1(b),
<https://cowles.yale.edu/sites/default/files/2022-08/d1843.pdf>) report 5,000-draw
finite-sample critical values with lag 0. `scripts/generate_unitroot_breaks_oracles.py psy`
reproduces them:

| Design | Statistic | Published 90/95/99% | Seed 20261010 (B = 5000) | Seed 11 (B = 5000) |
| --- | --- | --- | --- | --- |
| T = 100, r0 = 0.4 | SADF | 0.72 / 1.05 / 1.66 | 0.749 / 1.083 / 1.735 | 0.757 / 1.083 / 1.694 |
| T = 100, r0 = 0.4 | GSADF | 1.16 / 1.48 / 2.08 | 1.179 / 1.480 / 2.163 | 1.195 / 1.497 / 2.133 |
| T = 200, r0 = 0.2 | SADF | 0.97 / 1.30 / 1.86 | 1.018 / 1.337 / 1.973 | 1.000 / 1.290 / 1.914 |
| T = 200, r0 = 0.2 | GSADF | 1.64 / 1.88 / 2.46 | 1.640 / 1.947 / 2.485 | 1.619 / 1.876 / 2.436 |

The seed-to-seed spread (up to 0.07 at the 95% point) is Monte Carlo error;
the drift `d/σ` (0 or 1, σ = 1 or 6.79) moved the T = 200 quantiles by less
than 0.01.

## Bai–Perron robust tests, sequential tests and date intervals

`oe.bai_perron` keeps its pure-change global SSR search and its finite-sample
Gaussian Monte Carlo (`errors="iid_gaussian"`, the default; see
[bai-perron.md](bai-perron.md)) and adds, following Perron's GAUSS program
`BRCODE.SRC` (October 2004, <https://blogs.bu.edu/perron/files/2020/05/m-Break.zip>):

* **Robust supF(k), UDmax, WDmax.** `errors="heteroskedastic"` uses the segment
  covariance `(Z_j'Z_j)^{-1} n_j/(n_j − q) Σ z z'û² (Z_j'Z_j)^{-1}` (the zero-lag
  limit of the GAUSS HAC); `errors="hac"` uses `pvdel` with `robust = 1,
  hetdat = 1, hetvar = 1`: per segment `n_j · correct(z_j, û_j)`, the
  quadratic-spectral kernel over all lags with Andrews' AR(1) plug-in bandwidth
  (unit weights) divided by `n_j − q`, with VAR(1) prewhitening when
  `prewhiten=True` (default). The statistic is
  `F = (T − (k+1)q)/(T k q) · δ̂'R'(R V̂ R')^{-1}Rδ̂` at the **SSR-minimizing**
  dates (GAUSS `pftest`), reported in F units. HAC scores use the regressors in
  their original units, as the bandwidth is not invariant to rescaling one
  column.
* **Asymptotic tables** (`asymptotic_tests`, for every error type): supF(k),
  UDmax and WDmax at 10/5/2.5/1% from Bai and Perron (1998, Table I) and
  (2003, Econometrics Journal 6, Tables 1–2). The tables refer to `q·F` (the
  GAUSS code prints "scaled by q"), so they are divided by `q`. They cover
  `trim ∈ {0.05, 0.10, 0.15, 0.20, 0.25}`, `q ≤ 10`, `k ≤ 9/8/5/3/2` for the five
  trims, and UDmax/WDmax for `max_breaks = M = 5, 5, 5, 3, 2`. Outside that
  domain the critical values are `None` with `asymptotic_inference` explaining
  why. `WDmax = max_k [c(q,α,1)/c(q,α,k)] supF(k)` is level specific. With
  robust errors the `tests` table carries these asymptotic values and
  `p_value` is `None`; with `iid_gaussian` it keeps the Monte Carlo p-values.
* **Sequential supF(ℓ+1|ℓ)** (`sequential`, ℓ = 0..max_breaks−1; GAUSS
  `spflp1`): within every segment of the global ℓ-break optimum with at least
  `2h` observations, the SSR-minimizing split with `h` observations on each
  side is tested by the one-break F statistic computed on that segment with the
  chosen covariance; the statistic is the largest (0, as in the GAUSS code, when
  no segment can hold another break). Critical values: Bai and
  Perron's supF(ℓ+1|ℓ) table. `sequential_breaks` is the first ℓ that is not
  rejected at `level` (0 if supF(1|0) is not rejected; `max_breaks` if all are).
* **Selection.** `selection=None` uses BIC with iid errors and the sequential
  rule otherwise (BIC, with a note, when the sequential critical values are
  unavailable); `"bic"`/`"sequential"` force a rule (a forced sequential rule
  without tables raises `critical_values_unavailable`). Coefficients,
  covariance (robust segment sandwich for robust errors) and intervals describe
  the selected partition.
* **Break-date intervals** (`date_intervals`, Bai 1997; Bai and Perron 1998,
  section 4): GAUSS `interval` with `hetq = hetomega = 1` — regime-specific
  `Q_i = Z_i'Z_i/n_i` and error variances (`iid_gaussian`) or long-run
  covariances (White or QS-HAC), the asymmetric argmax distribution of a
  two-sided drifted Brownian motion, bounds `b − c_p/a` and the GAUSS rounding
  `int()` (+1 on upper ends). `break_index` is the 0-based first row of the next
  regime (the GAUSS date `b`); unrounded bounds are reported too, and labels of
  bounds beyond the sample are clipped to the first/last period. A break whose
  interval cannot be formed (no coefficient change, nonpositive variance) gets
  `status="unavailable: ..."` without failing the call.

Deliberate deviations from `BRCODE.SRC`, each also in its docstring: (1) the
VAR(1) recolouring is `(I − A)^{-1} J (I − A)^{-T}` with `A` the VAR matrix
(Andrews–Monahan 1992; also mbreaks), whereas BRCODE transposes `A` (identical
for one changing coefficient); (2) the argmax CDF evaluates every
`exp(c x)Φ(−d√|x|)` product in log space, whereas the GAUSS `funcg` switches to a
first-order Mills ratio beyond an argument of 30 and multiplies one x ≤ 0 term
by an extra `Φ(−√|x|/2)` there (errors up to 1.5e−4 in the CDF in our checks);
(3) the quantile bisection runs 100 halvings instead of stopping once
`|CDF − level| < 1e−6`. mbreaks 1.0.1 additionally divides the second regime's
HAC term by `Q_i` instead of `Q_{i+1}` and uses `round`; BRCODE's choices are kept.
Partial-change models, the one-at-a-time `sequa` dating, and the
`hetdat = 0`/`hetvar = 0` robust variants are not provided.

## Validation evidence

| Method | Reference | Max deviation |
| --- | --- | --- |
| Lee–Strazicich (A/C, 1–2 breaks) | literal NumPy transcription of `LS_One.g`/`LS_two.g`, synthetic T = 48, maxlag 3 (tests); authors' `gdp.txt` example data, maxlag 4 and 8 (not committed) | statistic rel. < 1e−9; identical dates and lags |
| Clemente (IO/AO, 1–2 shifts) | literal NumPy transcription of Perron–Ng `iom0alph`/`aom0alph` with explicit dummies | rel. < 1e−8; identical dates and lags |
| Clemente Monte Carlo | draw-by-draw replay of the null with the transcription | p-value exact, quantiles rel. 1e−7 |
| PSY windows | every window refitted with NumPy least squares (psymonitor `ADF`, IC = 0), lags 0–2 | abs. < 1e−9 |
| PSY calibrations | Monte Carlo and wild bootstrap replayed draw by draw; PSY Table 1(b) | rel. 1e−8; table above |
| Bai–Perron HAC/iid supF, supF(ℓ+1\|ℓ) | mbreaks 1.0.1 `pftest`/`spflp1` (q-scaled), fixture `tests/fixtures/unitroot_breaks/bai_perron_r_oracles.json` | rel. < 1e−10 |
| Bai–Perron heteroskedastic F | explicit NumPy sandwich | rel. < 1e−10 |
| Bai–Perron intervals | strucchange 1.6-0 `pargmaxV` (iid; HAC with BRCODE's inputs) | abs. < 1e−8 observations |
| argmax CDF | strucchange `pargmaxV` on a 12-point grid, four parameter sets | abs. < 1e−13 |
| Tables | BRCODE.SRC vs mbreaks CSV (all 60 level/trim tables); LS 2013 journal vs tspdlib | identical |

Fixtures are generated by `scripts/generate_unitroot_breaks_oracles.R` (mbreaks
R sources + strucchange); tables by
`scripts/generate_unitroot_breaks_oracles.py tables`. The tests are
`tests/test_econ_unitroot_breaks.py` and `tests/test_econ_bubble_tests.py`.

## Bounded size and power check

`scripts/generate_unitroot_breaks_oracles.py size --datasets 400 --boot 199`
(seed 20261010, T = 100, nominal 5%, 400 independent datasets per cell, no
failures; Wilson 95% intervals) is a declared bounded diagnostic, not a
universal size/power guarantee:

| Procedure | Data | Rejection rate |
| --- | --- | --- |
| Lee–Strazicich one break, model A, general-to-specific lags (maxlag 8) | iid Gaussian random walk | 0.102 (0.076–0.136) |
| Lee–Strazicich one break, model C, general-to-specific lags | iid Gaussian random walk | 0.147 (0.116–0.186) |
| Clemente one shift, IO, asymptotic | iid Gaussian random walk | 0.068 (0.047–0.096) |
| Clemente one shift, AO, asymptotic | iid Gaussian random walk | 0.080 (0.057–0.111) |
| PSY GSADF, Monte Carlo critical value (1.954, B = 1999, w = 19) | iid random walk | 0.040 (0.025–0.064) |
| PSY GSADF, composite wild bootstrap | iid random walk | 0.013 (0.005–0.029) |
| PSY GSADF, Monte Carlo | random walk, innovation s.d. tripling at T/2 | 0.280 (0.238–0.326) |
| PSY GSADF, composite wild bootstrap | same variance break | 0.140 (0.109–0.177) |
| PSY GSADF, Monte Carlo | explosive episode (ρ = 1.04, observations 50–69) | 0.410 (0.363–0.459) |
| PSY GSADF, composite wild bootstrap | same explosive episode | 0.268 (0.226–0.313) |

Diagnosis of the departures, from separate 400-dataset runs (seed 99): with
the lag order fixed at 0 the Lee–Strazicich test rejects 0.0625 (model A) and
0.0375 (model C), and the empirical 5% quantile of model A (−3.66) matches the
authors' −3.62 (2013, Table 2); the excess with the general-to-specific rule
comes from choosing the lag separately at every configuration before
minimizing, which is the authors' procedure, and was also 0.0975 in that run.
The composite bootstrap's innovations `w_t·e_l` are a product of two
independent normal-like draws (kurtosis about 9), whose finite-sample 95%
GSADF quantile at T = 100 is 2.43 against 1.93 for Gaussian innovations; that
makes it conservative under iid errors, while it halves the over-rejection of
the Gaussian Monte Carlo under the variance break (because `e_l` is resampled,
the time pattern of the variance is not reproduced).

## Limitations

* No vendor parity is claimed; `stata_parity_validated` semantics do not apply
  to these test procedures.
* Lee–Strazicich and Clemente critical values are finite-sample (T = 100) or
  asymptotic values; the two-break CMR table is unavailable (Monte Carlo only).
* PSY assumes a fixed lag; information-criterion lag selection inside every
  window is not provided. Date stamps are first-crossing estimates without
  their own confidence statement.
* Bai–Perron robust inference is asymptotic and requires the tabulated trims;
  the iid Monte Carlo remains conditional on the fixed design. Break-date
  intervals are conditional on the estimated dates and on the selected count.

## References

* Bai, J. (1997). Estimation of a change point in multiple regression models.
  *REStat* 79, 551–563. <https://doi.org/10.1162/003465397557132>
* Bai, J. and Perron, P. (1998). *Econometrica* 66, 47–78, Table I.
  <https://doi.org/10.2307/2998540>
* Bai, J. and Perron, P. (2003). Computation and analysis of multiple structural
  change models. *JAE* 18, 1–22. <https://doi.org/10.1002/jae.659>
* Bai, J. and Perron, P. (2003). Critical values for multiple structural change
  tests. *Econometrics Journal* 6, 72–78, Tables 1–2.
  <https://doi.org/10.1111/1368-423x.00102>
* Clemente, J., Montañés, A. and Reyes, M. (1998). *Economics Letters* 59,
  175–182. <https://doi.org/10.1016/S0165-1765(98)00052-4>
* Lee, J. and Strazicich, M. C. (2003). *REStat* 85, 1082–1089, Table 2.
  <https://doi.org/10.1162/003465303772815961>
* Lee, J. and Strazicich, M. C. (2013). *Economics Bulletin* 33(4),
  2483–2492, Table 1.
* Perron, P. and Vogelsang, T. J. (1992). *JBES* 10, 301–320.
  <https://doi.org/10.1080/07350015.1992.10509907>
* Phillips, P. C. B., Shi, S. and Yu, J. (2015). *IER* 56, 1043–1078.
  <https://doi.org/10.1111/iere.12132>
* Phillips, P. C. B. and Shi, S. (2020). Real time monitoring of asset markets:
  bubbles and crises. *Handbook of Statistics* 42, 61–80.
  <https://doi.org/10.1016/bs.host.2018.12.002>
