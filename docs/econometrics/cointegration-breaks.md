# Cointegration tests with breaks, Fourier terms and threshold adjustment

Single-equation tests of the null of no cointegration (Fourier KPSS: of
cointegration) that allow for structural breaks, smooth Fourier breaks or
asymmetric threshold adjustment. Every statistic runs natively on CPU float64
Torch tensors (no autograd, QR least squares, batched over candidate break
dates); no third-party estimation code runs at call time. The published
programs of the original authors define the statistics where they exist, and
the reference checks below compare against those programs or independent
packages. This is not a vendor-parity claim: no licensed Stata/EViews
execution is involved (the project's 9 October 2026 acceptance decision).

| Function | Test (estimand) | H0 | Critical values / p-values |
| --- | --- | --- | --- |
| `oe.gregory_hansen` | Gregory–Hansen ADF\*, Z_t\*, Z_α\*: one break, models C, C/T, C/S, regime+trend | no cointegration | GH (1996a) Table 1 and (1996b) Table 1, 1/2.5/5/10/97.5%, m ≤ 4, trim 0.15 |
| `oe.hatemi_j` | Hatemi-J ADF\*, Z_t\*, Z_α\*: two regime shifts | no cointegration | Hatemi-J (2008) Table 1, 1/5/10%, k ≤ 4, trim 0.15 (one cell withheld) |
| `oe.maki_coint` | Maki min-τ with up to 5 sequential breaks, models 0–3 | no cointegration | Maki (2012) Table 1, 1/5/10%, k ≤ 4, m ≤ 5, trim 0.10 |
| `oe.bayer_hanck` | Fisher combinations EG-J and EG-J-Ba-Bo | no cointegration | component p-values from the authors' null draws; Fisher critical values 1/5/10%, K ≤ 11; no combination p-value |
| `oe.ecm_cointegration` | Banerjee–Dolado–Mestre t or Boswijk Wald (the Bayer–Hanck components) | no cointegration | p-value from the authors' null draws, K ≤ 11 |
| `oe.tsong_fourier` | Tsong et al. CI_f (DOLS or OLS, one Fourier frequency) | cointegration | Tsong et al. (2016) Table 1 cells agreed by two transcriptions, k ≤ 3, m ≤ 4 |
| `oe.fourier_adl` | Banerjee–Arčabić–Lee Fourier ADL t | no cointegration | BAL (2017) Tables 1a/1b via fcoint, conservative bracket for T ≥ 100, m ≤ 3, k ≤ 5 |
| `oe.enders_siklos` | Enders–Siklos TAR / M-TAR Φ (Φ\*), t-Max, symmetry F; τ = 0 or Chan estimate | no cointegration; symmetric adjustment | **none for Φ and t-Max** (tables not obtained); symmetry F with F(1, n−k) p-value |

All functions take `data, y, x` (a list of regressor names) and an optional
`time` column. Series must be complete and regularly spaced: missing values,
repeated periods and gaps raise `AnalysisError`; there are no weights,
categorical terms or panels. The return value is a `TableSet` whose `attrs`
hold the statistics, critical values, break dates (0-based position of the last
pre-break observation plus its period label), settings, `inference_status`
and `source`; `oe.summary_state` / `oe.restore_summary` round-trip it.

## Gregory–Hansen (1996a, b)

For every candidate T_b ∈ {round(0.15 n), …, round(0.85 n)} (GAUSS rounding,
half away from zero, as the authors' program) the cointegrating regression with
φ_t = 1(t > T_b) is fitted on all n observations:

* `model="level"` (C): y = μ1 + μ2 φ + α′x + e;
* `"trend"` (C/T): adds βt;
* `"regime"` (C/S): y = μ1 + μ2 φ + α1′x + α2′x φ + e;
* `"regime_trend"` (1996b eq. 2.2): adds β1 t + β2 t φ to C/S.

ADF(T_b) is the t ratio of ρ in Δê_t = ρ ê_{t−1} + Σ_{j≤K} c_j Δê_{t−j} (no
constant, t = K+2..n). K is chosen at every T_b exactly as Hansen's published
program `shifts` (GAUSS/R, https://users.ssc.wisc.edu/~bhansen/progs/joe_96.html):
`lags="t"` tests downward from `maxlag` (default 6, the paper's K_max) until
|t| ≥ 1.96 on the last lag; `"aic"`/`"bic"` minimize ln(SSR/n1) + c (K+2)/n1
on each order's own sample; an integer fixes K. Z_α(T_b) = n(ρ\*−1) and
Z_t(T_b) = (ρ\*−1)/ŝ use the program's correction: λ = Σ_{j ≤ M} w(j/M) γ̂_j
with quadratic-spectral weights, M = 1.3221 (a2 (n−1))^{1/5}, a2 = 4b²/(1−b)⁴ and
b the AR(1) coefficient of the first-stage innovations (no prewhitening — the
paper's text mentions prewhitening, the authors' code does not; the lag sum is
truncated at n−2 where the published program would fail). The tests are the
minima over T_b.

Critical values were read from the authors' scans of the articles
(joe_96.pdf p. 109, sha256 `07e2004a…f7164`; OxBulletin_96.pdf p. 559,
sha256 `a73bc4bb…27280`); the 1/5/10% columns agree with tspdlib
(`coint_ghansen.src`, commit a3660eaa) and the SSC module `ghansen`, except
ghansen's typo −75.56 for the C/S, m = 4, 10% Z_α\* value (the scan shows
−72.56). They apply to the paper's trimming 0.15 and m ≤ 4 regressors; any
other setting reports `critical_values=None` with a reason. No p-value exists.

## Hatemi-J (2008)

Two regime shifts T1 < T2 in the C/S model,
y = a0 + a1 D1 + a2 D2 + b0′x + b1′D1 x + b2′D2 x + u, searched over
T1 ∈ [round(0.15n), round(0.70n)], T2 ∈ [T1 + round(0.15n), round(0.85n)]
(Hatemi-J's GAUSS program `CItest2b`, SSC g00006, which calls Hansen's ADF and
PHILLIPS procedures). Only this model, the one of the article's Table 1, is
implemented. Critical values for k = 1..4 at 1/5/10% are the values on which
three transcriptions agree — tspdlib `coint_hatemiJ.src`, CRAN hatemicoint
1.1.0 and the SSC module hatemij — because the article could not be retrieved.
The 1% Z_α\* value for k = 1 is **withheld** (−90.704 in tspdlib, −90.794 in
the other two). The 10% Z_α\* value for k = 1 (−52.232) is out of line with
the 5% value in all transcriptions; it is reported as printed.

## Maki (2012)

Up to m ≤ 5 breaks located sequentially as in Maki's own GAUSS code (received
from the author by Aptech, tspdlib `coint_maki.src`): at each step one more
break is tried at every date of every segment between the already chosen
breaks that keeps round(0.10 n) observations on either side; the segment with
the smallest ADF t ratio is kept and the break is the date with the smallest
SSR of its **ADF regression** (makicoint's `engine="gauss"`; makicoint reports
that the article's text uses the cointegrating-regression SSR instead — that
variant is not implemented). The statistic is the smallest t ratio of all
trials; lags follow the author's rule (downward from 12, |t| > 1.654, each
order on its own sample) or are fixed. Models `level`, `trend`, `regime`,
`regime_trend` are Maki's models 0–3. Critical values (Table 1) come from the
author's code and equal CRAN makicoint 2.0.0, whose NEWS states they match the
article; they apply to the author's trimming 0.10. The first row (level model,
k = 1, m = 1: −5.709 / −4.602 / −4.354) is reproduced as published.

## Bayer–Hanck (2013) and the ECM tests

`oe.bayer_hanck` reproduces the authors' Stata implementation `bayerhanck.ado`
0.9 (SSC): Engle–Granger τ (`dfuller, lags(L) nocon` on the static residuals),
Johansen **maximum-eigenvalue** statistic for rank 0 (`vecrank, lags(L+1)`
with trend none / unrestricted constant / unrestricted trend; the authors'
code uses λ-max, not trace), and the Banerjee et al. t and Boswijk Wald
statistics of the partialled ECM regression (W_t = Δx_t and L lags of Δ(y, x);
Boswijk = b′V⁻¹b over all coefficients, the deterministic ones being zero by
construction). Component p-values are the share of the authors' 10,000
simulated null draws beyond the statistic plus 1e-12, exactly as the ado. The
draws (`NullDistr.dta`, identical to CRAN combcoint 0.2.0 `null_dist`) are
shipped as `bayer_hanck_null.xz`, rounded to 1e-4: a p-value can differ from
the float32 original by one draw (1e-4) when the statistic is within 5e-5 of a
draw (3 of 87 reference p-values). The draws are redistributed under
combcoint's MIT license (`docs/reference-licenses/combcoint-MIT.txt`). Fisher
critical values at 1/5/10% for K = 1..11 come from the ado; the authors publish
no p-value for the combination, so `p_value` is None. `oe.ecm_cointegration`
returns either ECM component alone with its null-draw p-value.

## Fourier tests

**Tsong, Lee, Tsai and Hu (2016)**: CI_f = n⁻² Σ S_t² / ω̂² from the DOLS
(default; q leads/lags, default int(4(T/100)^{2/9})) or OLS regression with a
constant (and trend) and one Fourier pair, the frequency k ≤ `max_frequency`
minimizing the SSR; ω̂² is a Bartlett long-run variance (default bandwidth
round(4(n/100)^{2/9})) or the residual variance (`kernel="iid"`, the tspdlib
default). Large values reject cointegration. The Fourier F statistic is
reported without a p-value. **Banerjee, Arčabić and Lee (2017)**: the t ratio
of y_{t−1} in the ADL regression with Fourier terms, x_{t−1}, Δx_t…Δx_{t−q} and
Δy lags; following the authors' RATS code relayed by the SSC module fcoint, p,
q = 1..`max_lag` (6) are chosen by AIC/BIC on the common sample of the largest
order (one q per regressor for m ≤ 2), then the frequency (1..3 by default)
with the smallest SSR.

Neither article could be retrieved. Tsong critical values are the cells on
which tspdlib and fcoint agree (constant model k = 3, m = 4 and the 10% value
of k = 3, m = 1 are withheld). FADL critical values are fcoint's transcription
of Tables 1a/1b (T = 100, 500 and a larger T). A committed seeded simulation
(`fourier_simulation.json`; 10,000 replications at T = 1000 for Tsong, 5,000 /
5,000 / 2,000 at T = 100 / 500 / 2500 for FADL, fixed k and no lag selection)
reproduces the tables within 9.9% (Tsong, the tabulated values being on
average the larger) and 4.1% (FADL). FADL values are never interpolated: for
100 ≤ T < 500 the more negative of the T = 100 and T = 500 values is reported,
for T ≥ 500 the more negative of the T = 500 and larger-T values, below 100
none.

## Enders–Siklos (2001)

Engle–Granger residuals μ_t of y on (1, x); Δμ_t = I_t ρ1 μ_{t−1} +
(1−I_t) ρ2 μ_{t−1} + Σ γ_i Δμ_{t−i} with I_t = 1(μ_{t−1} ≥ τ) (TAR) or
1(Δμ_{t−1} ≥ τ) (M-TAR). τ = 0 or Chan's (1993) estimate: the SSR-minimizing
distinct value of the threshold variable at sorted positions
fix(0.15N)..N−fix(0.15N)+1 (Tom Doan's ENDERSSIKLOS procedure distributed on
Enders's web site). Lags are fixed or chosen by AIC/BIC on the common sample.
Φ (Φ\*) is the F statistic of ρ1 = ρ2 = 0, t-Max the larger t ratio, and the
symmetry F tests ρ1 = ρ2 with the F(1, n−k) p-value Enders and Siklos use under
cointegration. **Enders and Siklos's critical-value tables could not be
obtained from an authoritative source; Φ and t-Max are therefore reported
without critical values or p-values** (`inference_status`). For orientation
only, the seeded simulation below gives null quantiles of Φ (TAR, τ = 0, one
lag, one regressor, n = 100): 90% 5.13, 95% 6.14, 99% 9.04 — these are not used
for inference.

## Reference checks

`tests/test_econ_cointegration_breaks.py` (CI runs only committed fixtures and
NumPy/SciPy):

| Reference | Cases | Max deviation |
| --- | --- | --- |
| Hansen's published R procedures `estimate/adf/phillips` (joe_96r.zip), four models × four lag rules, three datasets | 25 GH | 2.4e-14 relative; identical break dates and lags |
| Same procedures with Hatemi-J's two-break grid; CRAN hatemicoint ADF\* (fixed lags) | 4 HJ | 1.2e-14 relative; identical break pairs |
| CRAN makicoint 2.0.0 (`engine="gauss"`), all models, m = 2, 3, 5 | 9 | 1.7e-14 relative; identical break dates and critical values |
| urca `ur.df` (EG), `ca.jo` (λ-max, constant case), R ECM regressions, combcoint null draws | 27 BH | 3.6e-14 relative (statistics); p-values ≤ 1e-4 (one draw) |
| CRAN cointests 1.1.0 `fcoint` (FADL with AIC/BIC grid, Tsong DOLS/OLS) | 14 | 8.4e-15 relative |
| NumPy re-implementations: every BH case (incl. Johansen none/trend), plus dedicated GH, HJ, Maki, FADL (fixed lags), Tsong (DOLS/Bartlett) and Enders–Siklos (TAR/M-TAR, τ = 0 and Chan search, BIC lags) cases | 27 + 11 | ≤ 1e-9 relative |

Hansen's own program on his distributed data (sample 1959:1–1990:4) gives
C/S ADF\* −4.59, Z_t\* −5.86, Z_α\* −54.6 against −4.71, −5.82, −53.57 in the
paper's Table 5 (sample 1960:1–1990:4, different rate series): the published
empirical results cannot be reproduced exactly from the distributed data, so
they are not used as a fixture.

## Size and power (seeded documentation simulation)

`size_power.json`: 500 replications, seed 20261014, n = 100, one regressor,
rejection at the published 5% point; size under independent random walks (for
Tsong under cointegration), power under cointegration with AR(0.4) errors and a
level shift of 4 at n/2 (GH level), a slope shift at n/2 (GH regime), two
slope shifts (Hatemi-J), two level shifts (Maki), no break (Bayer–Hanck) or a
smooth sin break (FADL); Tsong power under no cointegration.

| Test | Size | Power |
| --- | --- | --- |
| GH level ADF\* / Z_t\* / Z_α\* | 0.098 / 0.050 / 0.020 | 0.996 / 1.000 / 1.000 |
| GH regime ADF\* / Z_t\* / Z_α\* | 0.118 / 0.074 / 0.008 | 1.000 / 1.000 / 1.000 |
| Hatemi-J ADF\* / Z_t\* | 0.152 / 0.088 | 1.000 / 1.000 |
| Maki level, m = 2 | 0.154 | 0.818 |
| Bayer–Hanck EG-J / EG-J-Ba-Bo | 0.058 / 0.064 | 1.000 / 1.000 |
| Fourier ADL (AIC lags, k ≤ 3) | 0.192 | 0.942 |
| Tsong CI_f (DOLS, Bartlett) | 0.018 | 0.342 |

The asymptotic critical values are oversized at n = 100 for ADF\*-type
statistics with data-dependent lags (as Gregory and Hansen report for ADF\*)
and undersized for Z_α\*. The Fourier ADL distortion comes from the
pre-test selection of the frequency and lags (fixed k = 1 with fixed lags:
0.06; frequency selection only: 0.11; lag selection only: 0.13 in a 300-draw
check): the tables are conditional on k.

## Unsupported domains

Weights, missing-value handling other than refusal, panels and devices other
than CPU; GH/HJ/Maki critical values for other trimmings or more than four
regressors; Hatemi-J models other than the regime shift; Maki's
cointegrating-SSR break rule and more than five breaks; p-values of the
Gregory–Hansen, Hatemi-J, Maki, Fourier and Bayer–Hanck combination statistics
(none are published); Bayer–Hanck with more than 11 regressors; Fourier
frequencies above 3 (Tsong) or 5 (FADL) and FADL below T = 100; Enders–Siklos
critical values; bootstrap or dependence-robust calibrations.

## Regeneration

```sh
python scripts/generate_cointegration_breaks_oracles.py --sources DIR --rlib LIBS \
    [--fourier-check]            # R oracles, null asset, Fourier simulation
python scripts/generate_cointegration_breaks_oracles.py --sources . --rlib . --size-power
```

DIR holds `joe_96r.zip`, `NullDistr.dta` and `combcoint_0.2.0.tar.gz` (checked
against the sha256 values in the script); LIBS is a colon-separated R library
path with urca, jsonlite, makicoint 2.0.0, hatemicoint 1.1.0 and cointests
1.1.0. Fixture hashes: `critical_values.json`
`07fe58a57c42ccb4374907ccdda91a00d15f34025593661da0a0ed6cdd95f71e` (also
`TRANSCRIPTION_SHA256` in `coint_breaks_tables.py`); the null-draw payload
`0041f59794b6a67d963260412d46dc541e01e5b9dc476a84770a314ab87c1103`.
