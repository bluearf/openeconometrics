# Bootstrap causality extensions

Five predictive-noncausality procedures extend the
[Toda–Yamamoto](toda_yamamoto.md), [Dumitrescu–Hurlin/Breitung–Candelon](frequency_panel_causality.md)
and partial-sum (`asymcausality`) routes with bootstrap calibration. All run on
CPU float64 Torch, read complete in-memory tables, return `TableSet` results
with console text and LaTeX, never call a third-party estimator at run time,
draw from a private seeded `torch.Generator` (the global RNG is untouched),
vectorise the replications in bounded chunks and refuse a request whose
estimated floating-point work exceeds `max_work`. A replication with a
numerically singular design or a nonfinite statistic is a failed draw: the
call raises `bootstrap_failure` with the attempted and failed counts; no draw
is dropped or replaced. Rejections describe **predictive** (Granger)
noncausality, not structural causal effects. None of this is a Stata, EViews
or GAUSS parity claim; licensed vendor runs are not part of the evidence.

| Function | Method | Inference |
| --- | --- | --- |
| `oe.konyacausality` | Kónya (2006) SUR panel Wald, unit-specific bootstrap critical values | null-imposed period bootstrap, per unit |
| `oe.ekcausality` | Emirmahmutoğlu–Köse (2011) heterogeneous mixed-panel LA-VAR Fisher test | period bootstrap of the Fisher statistic |
| `oe.fouriertycausality` | Fourier Toda–Yamamoto, single (Nazlioglu, Gormus and Soytas 2016) or cumulative (Nazlioglu, Soytas and Gormus 2019) frequencies | fixed-design residual bootstrap |
| `oe.rollingcausality` | Rolling-window bootstrap Granger causality (Balcilar, Ozdemir and Arslanturk 2010) | residual bootstrap per window |
| `oe.asymcausality(..., resampling="leverage")` | Hatemi-J (2012) asymmetric test with the Hacker–Hatemi-J (2006) leveraged bootstrap | authors' leveraged recursive bootstrap |

Shared conventions: the bootstrap p-value is `(1 + #{T* >= T}) / (B + 1)` with
Monte Carlo error `sqrt(p (1 - p) / B)`, and critical values at 1%, 5% and
10% are the order statistics `T*_(ceil((B + 1)(1 - a)))`, so that `p <= a`
exactly when `T` exceeds the level-`a` critical value (no ties). The
leveraged asymmetric route instead reports the authors' averaged order
statistic (below). Each result records the seed, the documented
`torch.randint` draw order, the first index vector and (panel routes) a
SHA-256 digest of every index drawn, so the same draws can be supplied to an
independent implementation.

## Kónya SUR panel Wald

```python
res = oe.konyacausality(data=df, y="growth", x="exports", panel="country",
                        time="year", lags=(2, 1), bootstrap=999, seed=1)
res["tests"]        # one row per direction and unit: W_i, cv_1/5/10, bootstrap p
res["dependence"]   # Breusch-Pagan LM and Pesaran CD of the residual correlations
```

For each direction the N unit equations
`y_it = a_i + sum_l b_il y_i,t-l + sum_l g_il x_i,t-l + e_it` (own lags
`ly`, cross lags `lx`, common to all units) form one seemingly unrelated
system. Estimation is two-step feasible GLS: `Sigma = U U'/T` from the unit
OLS residuals and `b = [X'(Sigma^-1 ⊗ I_T)X]^-1 X'(Sigma^-1 ⊗ I_T)y`. Because
every equation has its own intercept, slopes are computed from within-unit
demeaned data (the GLS projection on the intercept space is demeaning for any
Sigma), and the reported full covariance in unit blocks `[Intercept, own
lags, cross lags]` is assembled exactly as `Cov(a) = Sigma/T + W̄ V W̄'`,
`Cov(a, b) = -W̄ V`. Unit i's Wald statistic tests `g_i1 = ... = g_i,lx = 0`
with the SUR covariance.

Bootstrap (Kónya 2006): (1) estimate the system under the null
(`g_i = 0` for every unit) with the same SUR estimator and keep its N × T
residual matrix; (2) draw whole periods (columns) with replacement, which
preserves the contemporaneous cross-unit correlation; (3) rebuild `y*`
recursively from the restricted equations with the observed initial values
and the observed `x`; (4) re-estimate the unrestricted SUR (Sigma included)
and recompute every unit Wald; (5) repeat B times. Each unit receives its own
critical values; there is no panel-wide statistic. Draws: per direction (x→y
first), `torch.randint(T', (chunk, T' + burn_in))` per chunk of replications
(one chunk when the 64 MiB budget allows; the chunk size is recorded).
Kónya's own description of the method (La Trobe working paper 2004.02, the
same SUR procedure later published as Kónya 2006) restates these five steps:
no residual centring is mentioned, `y*` is recursive and `x` does not enter
the bootstrap equation. Its implementation remark draws `T* > T` columns,
starts the recursion at zero and uses only the last T values for the Wald
tests; `burn_in=k` reproduces that scheme (zero start, `k` discarded
periods), while the default `burn_in=0` starts from the observed initial
values. The working paper does not say whether the restricted system in step
1 is fitted by SUR or OLS; its estimation is by SUR throughout, which this
implementation follows (tspdlib's `Panel_SURwald` mixes SUR coefficients with
OLS residuals).

`lags` is an integer, an `(own, cross)` pair, or `None` with `maxlag` and
`criterion` (`"aic"`/`"sic"`): every pair in `1..maxlag` is fitted on the
common sample of the longest lag and the minimum of Kónya's system criteria
`AIC = ln|W| + 2N²q/T`, `SC = ln|W| + N²q ln(T)/T` (W the SUR residual
covariance, N equations, q = 1 + own + cross coefficients per equation; La
Trobe WP 2004.02, equations (3)–(4)) is re-estimated on its own longest
sample. The asymptotic chi-square p-value is a
diagnostic only. The dependence table reports the Breusch–Pagan (1980) LM
statistic `T Σ_{i<j} ρ_ij²` (chi-square, N(N−1)/2 df; fixed N, large T) and
Pesaran's CD `sqrt(2T/(N(N−1))) Σ_{i<j} ρ_ij` (two-sided normal) on the
unrestricted OLS residual correlations that SUR exploits.

Domain: complete balanced panels on one regular time grid; 2–60 units; lags
≤ 8; usable periods greater than N + 1 (Sigma must be nonsingular) and than
the lags per equation plus two; no weights, trend or missing values; series
used as supplied (stationary, or correctly specified, dynamics as in the
paper; no unit-root pretest).

## Emirmahmutoğlu–Köse mixed-panel Fisher test

```python
res = oe.ekcausality(data=df, y="growth", x="energy", panel="country", time="year",
                     lags=None, maxlag=4, criterion="aic",
                     dmax={"TR": 1, "DE": 0, "US": 1}, bootstrap=999, seed=2)
res["tests"]        # Fisher lambda, chi2(2N) p-value, bootstrap cv/p
res["individual"]   # unit k_i, dmax_i, W_i, p_i
```

Each unit has its own levels VAR with lag order `k_i` and declared maximal
integration order `d_i` (Toda–Yamamoto augmentation, so I(0), I(1) and
cointegrated units can be mixed). The unit Wald `W_i = T'(RSS_r − RSS_u)/RSS_u`
(Toda–Yamamoto ML residual variance) tests the first `k_i` cause lags; the
`d_i` augmentation lags stay unrestricted. Fisher's
`λ = −2 Σ ln p_i` with `p_i = P(χ²(k_i) > W_i)` is computed from stable
closed-form log tails (no underflow for very large `W_i`); its `χ²(2N)`
p-value assumes cross-section independence. The bootstrap distribution of `λ`
handles dependence: unit OLS fits under the null, centred residuals, whole
periods of the `T' × N` residual matrix resampled with replacement, `y*`
regenerated recursively (observed initial values, observed `x`, augmentation
terms kept), and every `W_i*`, `p_i*`, `λ*` recomputed. All units use the
common sample `t = P+1..T`, `P = max_i (k_i + d_i)`, so whole periods can be
resampled jointly. Lag orders are caller fixed (an integer or a
`{unit: k_i}` mapping) or selected per unit by the varsoc AIC/SBIC of the
unit's levels VAR with a constant over `1..maxlag` on the common sample of
`maxlag` (the same `_lag_order` routine as `oe.varsoc`). `dmax` is an integer
0–2 or a mapping; no unit-root pretest is run. Rejection supports causality in
at least one unit.

Source notes: the bivariate restatement of the EK bootstrap fits the
restricted unit equations by OLS, centres the residuals by
`ũ_t = û_t − (T − k − l − 2)⁻¹ Σ_{t=k+l+2}^{T} û_t` (Stine), with
`k = max k_i`, `l = max dmax_i`, i.e. by their mean over the common sample, and
regenerates `y*` recursively with the observed `x` held fixed — the choices
implemented here. A later trivariate application co-authored by
Emirmahmutoğlu (University of Pretoria working paper on US state asset
prices) regenerates every variable recursively under the null; that system
variant is not implemented. The initial values of `y*` are not stated in the
sources found; the observed initial values are used.

## Fourier Toda–Yamamoto

```python
res = oe.fouriertycausality(data=df, y=["reit", "oil"], maxlag=8, kmax=3,
                            dmax=1, criterion="aic", bootstrap=999, seed=3)
cum = oe.fouriertycausality(data=df, y=["reit", "oil"], cumulative=True,
                            maxlag=8, kmax=3, bootstrap=999)
```

The levels VAR(p + dmax) carries a constant and `sin(2πkt/T)`, `cos(2πkt/T)`,
`t = 1..T` (single frequency k) or the same pair for every frequency `1..k`
(cumulative). The modified Wald for each directed pair restricts the first p
cause lags in the effect equation, `W = (RSS_r − RSS_u)/(RSS_u/(T' − m))`
(p times the ordinary F; m regressors per equation including Fourier terms),
asymptotically `χ²(p)` for a valid `dmax`. With `frequency=None` and/or
`lags=None` the pair (k, p) minimises the varsoc AIC/SBIC of the VAR(p)
**without** augmentation but with the candidate Fourier terms, every
regressor counted in the penalty, on the common sample of the longest lag;
ties go to the smaller frequency, then the smaller lag.

The bootstrap follows the method author's GAUSS code
([tspdlib `GCtests.src`](https://github.com/aptech/tspdlib/blob/master/src/GCtests.src),
procedures `_get_Wald`, `_get_Wald_bootstrap`): the centred unrestricted
effect-equation residuals are resampled with replacement, re-centred, added to
the restricted fitted values, and the Wald is recomputed with the **observed
regressors held fixed**. Because the restricted fit lies in both column spaces
`W*` depends only on the resampled residuals, which is how it is evaluated.
Draws: one `torch.randint(T', (B, T'))` per pair in table order.

Deliberate differences from tspdlib, declared here: (i) the lag/frequency
selection uses a common sample and the same `t = 1..T` Fourier index as the
estimated model (tspdlib's selection stage re-indexes the trigonometric terms
on each candidate's shortened sample and does not penalise Fourier terms);
(ii) restrictions cover exactly lags `1..p` for any `dmax` (the GAUSS
restricted design keeps only the last augmentation lag, which equals this
rule for its default `dmax=1`); (iii) p-values use the plus-one convention
(tspdlib reports `#{W* >= W}/B`). Fixed-design resampling conditions on the
observed lags and assumes iid innovations. The authors' written description
(Nazlioglu, Gupta, Gormus and Soytas, University of Pretoria WP 2019-54, the
same Fourier TY test) calls the bootstrap Efron's residual sampling and refers
to Hatemi-J (2002) and Balcilar et al. (2010) for details, reports both the
asymptotic and the bootstrap p-value, and chooses (k, p) by paring the grid
down from `kmax`, `pmax` and minimising the criterion — consistent with the
global argmin used here.

## Rolling-window bootstrap sequence

```python
seq = oe.rollingcausality(data=df, y="growth", x="energy", time="year",
                          window=24, lags=2, step=1, bootstrap=499, seed=4)
seq["windows"]      # one row per window: LR, Wald, bootstrap/asymptotic p, cause-lag sums
```

Every window of `window` consecutive observations (starting rows `0, step,
2·step, …`) fits a bivariate VAR(p) with intercepts and, for both directions,
the modified LR `(T' − m) ln(RSS_r/RSS_u)` (Sims' df correction; `T' =
window − p`, `m = 2p + 1`) and the Wald `(T' − m)(RSS_r − RSS_u)/RSS_u` of the
effect equation. Within a window both are increasing functions of
`RSS_r/RSS_u`, so their bootstrap p-values coincide — the p-value does not
depend on which of the two forms is meant by "modified LR". The residual-based
bootstrap per window and direction estimates the VAR under the null (effect
equation without cause lags, cause equation unrestricted), centres its
residual **vectors**, resamples them jointly with replacement, regenerates
both series recursively from the window's first p observations and
recomputes the LR. Draws: one `torch.randint(T', (2, B, T'))` per window in
window order (slot 0 x→y, slot 1 y→x). Output is bounded to one row per window
(`max_windows` ≤ 5000) and `max_work`. Supply the series in the form the VAR
should use (for example growth rates); no differencing, lag selection,
parameter-constancy test or bootstrap interval for the coefficient sums (the
paper's figures) is computed — point sums are reported. Overlapping windows
are not independent tests and receive no multiplicity correction.
Restatements of the method by its first author's later work describe the
bootstrap samples as generated from the restricted VAR under the null, as
here; the exact small-sample "modified LR" formula of the 2010 paper was not
located in an open source, which is why the invariance of the bootstrap
p-value to the LR/Wald form matters.

## Asymmetric causality with the leveraged bootstrap

```python
res = oe.asymcausality(data=df, y="stock", x="oil", lags=2, dmax=1,
                       signs=("+", "-"), bootstrap=999, seed=5, resampling="leverage")
```

The default `resampling="iid"` route is unchanged (a declared conditional iid
partial-sum null, documented in
[capability-closures-2026-10-07.md](capability-closures-2026-10-07.md)). The
new `resampling="leverage"` route reproduces the Hacker and Hatemi-J (2006)
leveraged bootstrap that Hatemi-J (2012) uses, following the authors' GAUSS
routines (Scott Hacker's `estvar_params`, `W_test` and `Bootstrap_Toda`, as
distributed in [tspdlib `actest.src`](https://github.com/aptech/tspdlib/blob/master/src/actest.src)):

- Wald: unrestricted VAR(p + dmax) with constant on the partial sums,
  df-corrected system covariance `RES'RES/(T' − m)`; equal to
  `(RSS_r − RSS_u)/(RSS_u/(T' − m))` for the equation-1 restrictions.
- Null DGP: the restricted VAR(p) **without** augmentation lags, fitted by OLS
  equation by equation on its own sample, rows `p..n−1` with p = `lags`
  (effect equation on its own lags; cause equation unrestricted);
  augmentation coefficients are zero.
- Leverage: residual `δ_t / sqrt(1 − h_t)` with `h_t` the observation's
  leverage in its own equation's hat matrix (the paper's definition). The
  GAUSS code divides every row by the **first** observation's leverage
  (`leverage[1,1]`); this implementation uses observation-specific leverages
  and declares that difference.
- Resampling: residuals of rows `p + dmax..n−1`, an **independent** index for
  each equation (`torch.randint(T', (B, T', 2))`, as the GAUSS code draws one
  uniform per variable), each drawn column mean-adjusted, recursive generation
  from the observed initial values (`leverage_design="recursive"`, default).
- The authors' written descriptions instead generate `Y* = D̂Z + δ*` with the
  **observed** lagged regressors `Z` (Hacker and Hatemi-J, CESIS WP 223,
  [static.sys.kth.se/itm/wp/cesis/cesiswp223.pdf](https://static.sys.kth.se/itm/wp/cesis/cesiswp223.pdf);
  Hatemi-J 2021, [arXiv 2106.07612](https://arxiv.org/abs/2106.07612)), which
  conflicts with their distributed GAUSS routine. `leverage_design="fixed"`
  implements that written form (the restricted fitted values then cancel from
  the equation-1 Wald). The same written sources define the critical value as
  "the lower boundary of the top α quantile"; the GAUSS averaging rule below is
  what is reported.
- Critical values: the authors' rule, the mean of the sorted `W*` at position
  `B − floor(B a)` and the next one (1-based), for a = 1%, 5%, 10%; the 5%
  decision is `W > cv_5`. The plus-one p-value is also reported.

## Evidence

`tests/test_econ_causality_extensions.py` recomputes every statistic with
explicit NumPy (direct Kronecker-product SUR GLS with intercept columns,
uncentred regressions with explicit constants, the authors' `W_test`
vectorisation and restriction matrix) and every bootstrap distribution from
the **same** indices regenerated from the documented draw order, with
replication loops in NumPy. Measured agreement: SUR Wald, coefficients and
full covariance ≤ 1e-9 relative; EK unit Wald and Fisher ≤ 1e-9; Fourier TY
Wald (quadratic form and tspdlib RSS form) ≤ 1e-9; rolling LR/Wald ≤ 1e-9;
leveraged bootstrap draws ≤ 1e-7 relative (recursive amplification); all
bootstrap p-values and critical values identical up to these tolerances. The
tests also cover lag/frequency selection grids against NumPy criteria,
failure codes, seeds, global-RNG isolation, JSON and LaTeX output.

`scripts/generate_causality_extensions_oracles.py` runs seeded, bounded null-
size and power cells for every route and writes
`tests/fixtures/causality_extensions/size_power.json` (planned/completed
datasets, failed calls, rejections, Wilson 95% intervals, script SHA-256).
Run of 2026-10-10 (torch 2.14.0, one thread, B = 199, α = 0.05, about two
minutes; every planned dataset completed, zero failed calls):

| Cell | Tests | Rejection rate | Wilson 95% |
| --- | --- | --- | --- |
| Kónya null, N=5, T=50, cross-unit ρ=0.5 (all unit tests pooled) | 1500 | 0.045 | 0.036–0.057 |
| Kónya power, causal units (g=0.5) | 600 | 0.988 | 0.976–0.994 |
| EK bootstrap Fisher null, N=5, T=50, ρ=0.5, I(1) cause, dmax=1 | 400 | 0.063 | 0.043–0.091 |
| EK asymptotic χ²(2N) Fisher, same datasets | 400 | 0.128 | 0.098–0.164 |
| EK bootstrap power (g=0.3 in 2 of 5 units) | 400 | 0.435 | 0.387–0.484 |
| Fourier TY null, n=100, smooth intercept shift, AIC-selected k and p | 400 | 0.055 | 0.037–0.082 |
| Fourier TY power (0.3) | 400 | 0.835 | 0.795–0.868 |
| Rolling route, single full window, n=80, p=1, null | 400 | 0.060 | 0.041–0.088 |
| Rolling route power (0.3) | 400 | 0.828 | 0.787–0.861 |
| Leveraged asymmetric test, independent random walks, n=100, `W > cv_5` | 400 | 0.048 | 0.031–0.073 |

The pooled Kónya unit tests within one dataset are dependent, so that
interval is optimistic. The EK rows show the size distortion of the
independence-based χ²(2N) reference under cross-unit correlation and its
removal by the period bootstrap in this design. These are descriptive
results for the declared designs, not a proof of finite-sample validity for
every data-generating process.

Source handling: the algorithm steps above follow the cited papers. The
GAUSS library tspdlib (Saban Nazlioglu) was reviewed as a development
reference: `GCtests.src` is the Fourier Toda–Yamamoto author code that the
Fourier route follows, and `actest.src` carries Hacker and Hatemi-J's own
routines that the leveraged route follows (differences listed above). Its
`PDcaus_SURwald.src` and `PDcaus_Fisher.src` panel procedures were also
read; they deviate from the papers' recursive null-imposed bootstraps
(`Panel_SURwald` resamples OLS residuals around SUR coefficients and
regenerates only one own lag; `Panel_Fisher` adds resampled residuals to
fixed restricted fitted columns), so they are not reproduced or used as
numerical references.

References: [Kónya (2006)](https://doi.org/10.1016/j.econmod.2006.04.008)
([La Trobe WP 2004.02](https://ideas.repec.org/p/ltr/wpaper/2004.02.html));
[Emirmahmutoğlu and Köse (2011)](https://doi.org/10.1016/j.econmod.2010.10.018);
[Nazlioglu, Gormus and Soytas (2016)](https://doi.org/10.1016/j.eneco.2016.09.009);
[Nazlioglu, Soytas and Gormus (2019)](https://doi.org/10.1080/1540496X.2018.1434072);
[Balcilar, Ozdemir and Arslanturk (2010)](https://doi.org/10.1016/j.eneco.2010.05.015);
[Hatemi-J (2012)](https://doi.org/10.1007/s00181-011-0484-x);
[Hacker and Hatemi-J (2006)](https://doi.org/10.1080/00036840500405763);
[Toda and Yamamoto (1995)](https://doi.org/10.1016/0304-4076(94)01616-8).
