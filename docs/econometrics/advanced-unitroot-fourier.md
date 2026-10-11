# Fourier, point-optimal and seasonal unit-root tests

`oe.fourier_adf`, `oe.fourier_kpss`, `oe.fourier_lm`, `oe.ers_pt` and
`oe.hegy` are univariate test procedures of the `unitroot` family. They run
float64 Torch least squares (QR) and return OpenEconometrics tables
(`core.table` / `core.TableSet`) with JSON-safe `.attrs`; nothing from SciPy,
statsmodels or R runs at runtime. They are method-level coverage, not vendor
parity: `stata_parity_validated` claims do not apply to test procedures and
no licensed-software comparison was made.

```python
fadf = oe.fourier_adf(df, "spread", trend="constant", frequency="auto", lags="t", maxlag=8)
fkpss = oe.fourier_kpss(df, "real_rate", trend="constant", lags=8)
flm = oe.fourier_lm(df, "spread", frequency=1, lags="t", maxlag=8)
pt = oe.ers_pt(df, "gdp", trend="trend", lags="bic", minlag=3, maxlag=8)
seasonal = oe.hegy(df, "sales", seasons=4, deterministic="seasonal", lags="bic")
print(fadf); fadf.to_latex()
```

| Function | Null | Rejects | Deterministic terms | Critical values | p-value |
| --- | --- | --- | --- | --- | --- |
| `fourier_adf` | unit root | small tau, large F | constant or constant + trend, sin/cos(2πkt/T) | Enders & Lee (2012, *Economics Letters*) Tables 1a/1b | none |
| `fourier_kpss` | stationarity | large tau, large F | level or trend, sin/cos(2πkt/T) | Becker, Enders & Lee (2006) Table 1 (k = 1–3); seeded simulation (k = 4, 5; supremum F) | none |
| `fourier_lm` | unit root | small tau, large F | trend estimated in differences, sin/cos | Enders & Lee (2012, *Oxford Bulletin*) Table 1 panels a–c | none |
| `ers_pt` | unit root | small P_T | constant (c̄ = −7) or trend (c̄ = −13.5) | Elliott, Rothenberg & Stock Table 1 | none |
| `hegy` | seasonal/zero-frequency unit roots | small t, large F | none / constant / + trend / + seasonal dummies / both | seeded simulation | none |

No procedure reports a p-value: every source publishes quantiles only (or,
for HEGY, no published source replicated), and interpolating between three
quantiles would invent a distribution. `p_value` is `None` with
`p_value_reason`; decisions are in `reject` (and `f_reject`), per level.
All critical values are interpolated in the sample size, with a note when the
sample lies outside the tabulated range (the end row is then used).

## Samples and failures

Series are taken in row order or sorted by `time` (consecutive integer
periods or datetimes). Missing values (`missing_values`), gaps
(`time_gaps`), repeated periods, constant series (`constant_series`), too
few observations (`insufficient_observations`: 12 for the Fourier tests, 10
for `ers_pt`, 4·S for `hegy`), invalid options (`invalid_option`,
`invalid_lags`), rank-deficient designs (`collinear_regressors`) and exact
fits of deterministic series (`perfect_fit`) raise `AnalysisError`. No rows
are dropped. Every statistic is invariant to the scale of the series and,
whenever a constant is included, to its level; the tests check both.

## Fourier ADF (Enders and Lee 2012, *Economics Letters* 117, 196–199)

Regression (3) of the paper, on t = p+2..T (T observations, t = 1..T):

Δy_t = ρ y_{t−1} + c₁ [+ c₂ t] + c₃ sin(2πkt/T) + c₄ cos(2πkt/T) + Σ_{j≤p} b_j Δy_{t−j} + e_t.

tau is the OLS t ratio of ρ. `trend="constant"` is τ_DF_C (Table 1b),
`"trend"` is τ_DF_t (Table 1a). F tests c₃ = c₄ = 0 in the same regression:
F = ((SSR_r − SSR_u)/2)/(SSR_u/(N − K)).

`frequency="auto"` estimates k by the smallest SSR over 1..`max_frequency`
(default 5; ties to the lower k), as in step 1 of the paper; the critical
values of the chosen k are used, as the paper recommends (they ignore the
selection step). `lags` is fixed or `"aic"`, `"bic"`, `"t"`: for each
candidate k the orders 0..`maxlag` are compared on the common sample
t = maxlag+2..T with the rules of `oe.zandrews` (`"t"`: largest p whose last
lag has |t| ≥ 1.645); k is then chosen by the SSR of those common-sample
regressions and the chosen (k, p) is refitted on t = p+2..T. The
`frequencies` table lists every candidate.

**Critical values.** Enders and Lee tabulate τ for k = 1..5 and
T = 100, 200, 500, 2500, and the supremum F(k̂) = max_k F(k) for k ≤ 5. In the
paper's F rows the header reads 1%, 5%, 10% but the entries are printed
10%, 5%, 1% (the text quotes 9.14 as the T = 100 5% value); they are stored as
1%, 5%, 10%. F critical values are therefore given only for
`frequency="auto"` with `max_frequency=5`; for a fixed k or a smaller
candidate set `f_critical_values` is `None` with `f_inference_reason`.
Source: the authors' revised final version (4 March 2012) hosted at
[time-series.net](https://www.time-series.net/assets/docs/Enders_Lee_DF_Fourier_tests_Final_version_March_5_2012.40131805.pdf),
Tables 1a and 1b.

## Fourier KPSS (Becker, Enders and Lee 2006, *JTSA* 27, 381–409)

OLS of y on a constant [and t] and sin/cos(2πkt/T) (regressions 3a/3b);
tau = Σ S_t²/(T² s²(l)) with S_t the partial sums of the residuals and
s²(l) the Bartlett long-run variance of `oe.kpss` (default truncation
int(12 (T/100)^¼); `auto=True` uses the Newey–West bandwidth). F of eq. (7):
((SSR₀ − SSR₁)/2)/(SSR₁/(T − q)). Frequency selection as above (smallest SSR).

**Critical values.** τ for k = 1, 2, 3 and T = 100, 500: BEL (2006) Table 1,
as reproduced by the authors in Table 6 of Jones and Enders (2014),
[author-hosted copy](https://www.time-series.net/assets/docs/Flexible_Fourier_Form06172012.3985105.pdf).
No primary copy of the k = 4, 5 entries was located (they appear only in
TSPDLIB), so those cells, and the supremum-F linearity pretest, use the
OpenEconometrics simulation (200,000 replications per cell, iid errors,
s² = SSR/T). The published supremum-F values were not used: at T = 100 they
exceed the simulated quantiles of eq. (7) by 0.10–0.24 (5–18 Monte Carlo
standard errors); they are consistent with a denominator of SSR₁/T instead
of SSR₁/(T − q). BEL note that F over-rejects with serially correlated
errors (they bootstrap it); the simulated values assume iid errors. A
fixed-frequency F has no calibrated distribution here.

## Fourier LM (Enders and Lee 2012, *Oxford Bulletin* 74, 574–599)

Equations (8)–(10): Δy_t on [1, Δsin, Δcos] (t = 2..T) gives δ̃; the
detrended series is S̃_t = y_t − ψ̃ − δ̃₀t − δ̃₁ sin − δ̃₂ cos with S̃₁ = 0; tau_LM
is the t ratio of φ in Δy_t = φ S̃_{t−1} + d₀ + d₁ Δsin + d₂ Δcos
+ Σ b_j ΔS̃_{t−j} + e_t. The linear trend is always part of the model.

The F statistic follows the authors' published simulation program for
Table 1 ([table1critvalues.PRG](https://www.time-series.net/assets/docs/table1critvalues.148164133.PRG)):
with S^L the linearly detrended LM series, F tests Δsin = Δcos = 0 in
Δy on [S^L_{t−1}, 1, Δsin, Δcos, lags]; this is the restricted/unrestricted
pair of the paper's Lemma 2. (Restricting within the Fourier-detrended
regression instead gives a different, much smaller distribution; it is not
what Table 1 tabulates.)

**Critical values.** Table 1 of the authors' final version (16 May 2011,
[time-series.net](https://www.time-series.net/assets/docs/Enders_Lee_Final_version_May_16_2011.39101351.pdf)):
panel a (τ_LM, k = 1..5), panel b (F for a given k), panel c (supremum F),
T = 100, 200, 500, 2500. Panels b and c are printed 10%, 5%, 1% under a 1%,
5%, 10% header (the RATS program prints fractiles 90, 95, 99); they are stored
as 1%, 5%, 10%. Panel b is used for a fixed k, panel c for `"auto"` over 1..5.

## ERS point-optimal P_T (Elliott, Rothenberg and Stock 1996)

With z_t = 1 or (1, t), c̄ = −7 or −13.5 and ᾱ = 1 + c̄/T, S(a) is the SSR of
the GLS regression of (y₁, y₂ − a y₁, …) on (z₁, z₂ − a z₁, …) and
P_T = [S(ᾱ) − ᾱ S(1)]/ω̂² (eqs. 13–15). ω̂² = s²/(1 − Σâ_j)² from
Δy_t = d₀ + a₀ y_{t−1} + Σ a_j Δy_{t−j} (eqs. 20–21; constant only, also in the
trend case, as ERS and `urca::ur.ers` do — ERS note the trend-case statistic
is then not strictly invariant to the trend slope under the alternative),
s² = SSR/(N − K). `lags="bic"` (default, ERS's AR(BIC)) or `"aic"`/`"t"`
compare `minlag..maxlag` on the common sample (Schwarz = the SC of
`oe.dfgls`; its MAIC is defined for the GLS-detrended regression and is not
offered here); ERS used 3 ≤ p ≤ 8 (`minlag=3, maxlag=8`). Small P_T rejects.

**Critical values.** Table 1, panels A and B, of the NBER Technical Working
Paper 130 version ([nber.org](https://www.nber.org/system/files/working_papers/t0130/t0130.pdf)),
T = 50, 100, 200, 500, ∞ at 1%, 2.5%, 5%, 10% (simulated by ERS with
ω̂² = σ̂²), linear in T, the ∞ row above 500. urca 1.3-4 (`cvals.ptest`) holds
the same T = 50, 100, 200, ∞ entries at 1%, 5%, 10%.

R comparison: `urca::ur.ers(type = "P-test")` agrees to 2.2×10⁻¹³ relative
error once two documented differences are accounted for: urca chooses
1..lag.max lags by BIC on unequal samples (the tests pass urca's chosen order
as fixed `lags`), and in the trend case urca demeans (0, Δy) over all T entries,
so its S(1) exceeds the GLS residual sum by (y_T − y₁)²/(T(T − 1)).

## HEGY seasonal unit roots (Hylleberg, Engle, Granger and Yoo 1990)

Δ_S y_t = Σ_i π_i y_{i,t−1} + deterministic + Σ_{j≤p} φ_j Δ_S y_{t−j} + e_t on
t = S+p+1..T with y₁ = (1 + L + … + L^{S−1})y, y₂ = −(1 − L + … − L^{S−1})y and
two filters per harmonic j (frequency 2πj/S), constructed as in R's
`uroot::hegy.regressors` (for S = 4: HEGY's −(1 − L²)y_{t−2}, −(1 − L²)y_{t−1}).
`seasons` is 4 (quarterly) or 12 (monthly, the Beaulieu–Miron 1993 setting).
Statistics: `t_1` (frequency 0), `t_2` (π), one `F_a_b` per harmonic pair,
`F_2_S` (all seasonal roots) and `F_1_S` (all roots; Ghysels, Lee and Noh
1994). Deterministic: `"none"`, `"constant"`, `"trend"`, `"seasonal"`
(constant + S − 1 seasonal dummies, default) or `"seasonal_trend"`. Lags fixed
or chosen by `"aic"`/`"bic"`/`"t"` on the common sample and refitted on the
full sample (uroot's `hegy.test` convention).

**Critical values: seeded simulation.** `hegy_tables.py` is generated by
`scripts/generate_unitroot_hegy_oracles.py` (seed 19900101): seasonal random
walk null, iid N(0,1), zero initial values, no augmentation; 50,000 (quarterly,
N = 40, 80, 160, 320, 1280) and 20,000 (monthly, N = 120, 240, 480, 1200)
replications per cell; N = T − S − p is the number of regression observations;
linear interpolation in 1/N. Monthly pair-F quantiles pool the five harmonic
pairs (each pair agrees with the pool within 4.5 Monte Carlo standard errors).
With augmentation the values are asymptotic approximations. Published sources
were compared and not used:

- HEGY (1990) Table 1a/1b as stored in uroot 2.1-3 (`.HEGY.orig.cv`): up to
  0.47 more extreme than the simulation (t_π1 with seasonal dummies and trend
  at T = 48; 0.34 with seasonal dummies only). uroot's own `hegy.test`
  (an independent `lm()` implementation) simulated under the same null
  reproduces the OpenEconometrics quantiles, not the table: with seasonal
  dummies at T = 48 its 5% point of t_π1 is −2.76 (simulation −2.79, table
  −3.08), using 3,000 replications at T = 48 and at T = 100.
- Díaz-Emparanza (2014, *CSDA* 76) response surfaces as ported in uroot
  (`hegy.rs.pvalue`): at T = 100 with seasonal dummies, 7.6% (F_3:4) and 7.7%
  (F_2:4) of 3,000 null series reject at nominal 5%; 98 of 180 response-surface
  quantiles at the simulation grid lie more than 5 Monte Carlo standard errors
  from the simulation. One surface serves every S although the limiting
  distributions of the joint F tests depend on S.
- Beaulieu and Miron (1993) monthly tables (uroot `.BM.orig.cv`) agree with the
  simulation within 0.16 everywhere and are recorded as corroboration.

## Evidence and reproduction

| Check | Reference | Measured |
| --- | --- | --- |
| Fourier ADF/KPSS/LM statistics, F, frequency/lag selection | explicit NumPy regressions from the papers' equations; statsmodels OLS (one case) | rel. error ≤ 1e−9 |
| ERS P_T | NumPy (eqs. 13–15, 20–21); R urca 1.3-4, 36 cases | 2.2e−13 relative |
| HEGY statistics, chosen lags | uroot 2.1-3 `hegy.test` (R `lm`), 128 cases, quarterly and monthly; NumPy oracle with Smith–Taylor filters Δ_S/(1 − 2cos ω L + L²), all five deterministic cases | 1.5e−12 relative |
| Published Fourier ADF entries (108) | seeded simulation, 100,000 replications | max 0.033 (τ), 0.10 (sup F) |
| Published Fourier KPSS τ, k = 1–3 (36) | seeded simulation, 200,000 replications | max 0.0047 |
| Published Fourier LM entries (99) | seeded simulation, 100,000 replications | max 0.035 (τ), 0.11 (F) |
| ERS Table 1 (32) | seeded simulation, 100,000 replications | max 0.145 (trend, T = 100, 1%; within combined MC error) |
| Second transcription | TSPDLIB (`github.com/aptech/tspdlib`, `src/fourier_*.src`), urca `cvals.ptest` | identical where both exist |

Fixtures: `tests/fixtures/unitroot_fourier/simulated_quantiles.json`
(`scripts/generate_unitroot_fourier_oracles.py`, seed 20261010, ~1 minute),
`tests/fixtures/unitroot_fourier/ers_urca.json`
(`scripts/generate_unitroot_fourier_oracles.R`),
`tests/fixtures/unitroot_hegy/simulated_critical_values.json`
(`scripts/generate_unitroot_hegy_oracles.py`, ~4 minutes) and
`tests/fixtures/unitroot_hegy/uroot_reference.json`
(`scripts/generate_unitroot_hegy_oracles.R`, sources the SHA-256-verified uroot
2.1-3 tarball without installing it). Tests:
`tests/test_econ_unitroot_fourier.py`, `tests/test_econ_unitroot_ers.py`,
`tests/test_econ_unitroot_hegy.py`.

## Not supported

- p-values for any of these statistics; bootstrap inference (e.g. BEL's
  recursive bootstrap of the KPSS F test).
- Cumulative frequencies (τ(n), Enders–Lee Tables 2a/2b), fractional
  frequencies, Fourier GLS (Rodrigues–Taylor) and panel Fourier tests.
- The Fourier LM test without a trend; the Fourier ADF without a constant.
- A strictly invariant P_T^τ (t in the spectral regression) and user-chosen c̄.
- HEGY for seasonal periods other than 4 and 12, seasonal dummies interacted
  with trends, or critical values for augmented regressions beyond the
  asymptotic approximation.
- Calibrated size/power beyond the iid-error null designs above; serially
  correlated errors rely on augmentation (unit-root tests) or the long-run
  variance (KPSS) as in the papers.
