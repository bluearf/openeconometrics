# Unit-root, stationarity, break and cointegration tests

Tests of the time-series properties of one series, of a regression over time,
and of panels: `dfuller`, `dfgls`, `pperron`, `kpss`, `zandrews`, `egranger`,
`chow`, `sbsingle`, `cusum`, `xtunitroot` and `xtcointtest`. Everything on this
page is implemented in OpenEconometrics on float64 PyTorch tensors. Every auxiliary
regression is a Householder QR; lag-order comparisons come from one nested
triangular factor; break-date scans and recursive residuals are computed from
running cross products instead of refitting; panel tests solve all panels in
one batched factorization. No estimation library runs at test time. p-values
and critical values that are not available in closed form come from published
tables typed into `openecon/econometrics/unitroot/tables.py` (listed
[below](#tables-and-their-sources)); nothing is simulated at run time.

These procedures are tests, not model fits. They register no estimator and
return **result tables** (`openecon.frame.DataFrame`, printable and exportable
with `.to_latex()`), or a `TableSet` of named tables when a procedure also
reports its regressions. The scalar results are always in `.attrs`:

| key | meaning |
| --- | --- |
| `statistic`, `p_value` | the headline statistic and its p-value (`None` when no p-value is tabulated) |
| `distribution` | reference distribution (`"Dickey-Fuller"`, `"t"`, `"normal"`, `"F"`, ...) |
| `critical_values` | `{"1%": .., "5%": .., "10%": ..}` (KPSS adds `"2.5%"`); absent when the reference distribution is standard |
| `lags`, `trend`, `nobs` | lag order used, deterministic terms, observations in the test regression |
| `label`, `notes` | what the statistic is; the sources and caveats that apply to this result |

```python
import openecon as oe

adf = oe.dfuller(df, "lgdp", time="quarter", lags=4, trend="trend")
print(adf)                                    # one-row table
adf.attrs["statistic"], adf.attrs["p_value"], adf.attrs["critical_values"]

eg = oe.egranger(df, "consumption", ["income"], time="quarter", lags=1, ecm=True)
eg["cointegrating_regression"], eg["ecm"], eg.attrs["p_value"]
```

| Stata / EViews | OpenEconometrics |
| --- | --- |
| `dfuller y, lags(4) trend regress` | `oe.dfuller(df, "y", lags=4, trend="trend", regress=True)` |
| `dfuller y, noconstant` / `drift` | `trend="none"` / `trend="drift"` |
| `dfgls y, maxlag(8) ers` / `notrend` | `oe.dfgls(df, "y", maxlag=8)` / `trend=False` |
| `pperron y, lags(4) trend` | `oe.pperron(df, "y", lags=4, trend="trend")` |
| `kpss y, maxlag(8) notrend` / `auto` (Baum) | `oe.kpss(df, "y", lags=8)` / `auto=True`; `trend=True` for trend stationarity |
| `zandrews y, break(both) lagmethod(AIC)` (Baum) | `oe.zandrews(df, "y", break_="both", lags="aic")` |
| `egranger y x1 x2, lags(2) trend ecm` (Schaffer) | `oe.egranger(df, "y", ["x1","x2"], lags=2, trend="trend", ecm=True)` |
| `estat sbknown, break(tq(1984q1))`; EViews Chow tests | `oe.chow(df, "y", ["x"], break_at, time="quarter")` |
| `estat sbsingle, trim(15) swald`; EViews Quandt-Andrews | `oe.sbsingle(df, "y", ["x"], time="quarter")` |
| `estat sbcusum`; EViews CUSUM / CUSUM of squares | `oe.cusum(df, "y", ["x"], time="quarter")` |
| `xtunitroot llc y, lags(1) trend` | `oe.xtunitroot(df, "y", "id", "year", test="llc", lags=1, trend="trend")` |
| `xtunitroot ips y, lags(aic 4)` | `..., test="ips", lags="aic", maxlag=4` |
| `xtunitroot fisher y, dfuller lags(2)` / `pperron lags(2)` | `..., test="fisher", lags=2` / `method="pperron"` |
| `xtunitroot hadri y, robust` / `kernel(bartlett 4)` | `..., test="hadri", robust=True` / `kernel_lags=4` |
| `xtunitroot breitung y, lags(1)` / `xtunitroot ht y, altt` | `..., test="breitung", lags=1` / `test="ht", altt=True` |
| Cross-sectionally augmented CADF / CIPS / CIPS* (Pesaran 2007) | `oe.xtcips(df, "y", "id", "year", lags=1)`; `oe.xtunitroot(..., test="cips")` / `test="cadf"` |
| `xtcointtest kao y x1 x2, lags(1) kernel(bartlett 2)` | `oe.xtcointtest(df, "y", ["x1","x2"], "id", "year", kernel_lags=2)` |

For cross-sectionally augmented tests, see [cips.md](cips.md). They require
complete balanced common-date panels and use blocked native QR. Their published
quantiles cover 10–200 units and 10–200 potential differences; `inference="none"`
computes statistics outside this range without inventing critical values or
p-values. Individual CADF output is available with `individual=True`.

## Samples, missing data and failures

Ng–Perron GLS M tests and the KSS nonlinear test are documented separately in
[advanced_unitroot.md](advanced_unitroot.md), including their strict integer
calendar contracts and published asymptotic critical values.

A series is taken in row order, or sorted by `time` when a time column is
named. Integer periods must be consecutive (`time_gaps` otherwise); datetimes
are taken as consecutive in sorted order; repeated periods are an error
(`repeated_time_values`); text, boolean or fractional time values are refused
(`invalid_time`). **Missing values are refused** (`missing_values`): dropping
an interior observation would silently change every lag, so there is no
`missing="drop"` here. Restrict the sample or fill the gap first. Panels are
sorted by unit and period and must be gap free inside each unit. The input
table is never modified.

A test regression that cannot identify all its terms (a constant series, an
exact trend, too many lags) raises `collinear_regressors`, `constant_series`
or `perfect_fit` rather than omitting a term: the test would no longer be the
one that was asked for. Collinear *user* regressors (in `egranger`, `chow`,
`sbsingle`, `cusum`, `xtcointtest`) are omitted left to right as Stata does
and are listed in `attrs["omitted"]` and in the notes.

Every statistic is invariant to the unit of the series (and, with a constant,
to its level); series are measured from their first observation internally so
that a level of 1e9 does not hide the variation. Columns whose squares would
overflow or underflow float64 (magnitudes above 1e140, or a spread below
1e-140) raise `numerical_failure` with the advice to rescale, instead of being
reported as constant.

Other error codes: `empty_data`, `missing_columns`, `duplicate_columns`,
`non_numeric_column`, `non_finite_values`, `insufficient_observations`,
`invalid_lags`, `invalid_option`, `invalid_spec`, `invalid_break`,
`singular_subsample`, `too_many_regressors`, `insufficient_panels`,
`unbalanced_panel`, `ips_moments_unavailable`, `unsupported_test`.

## `dfuller`: augmented Dickey-Fuller test

OLS on `t = lags+2, ..., T` of

    Delta y_t = a + d t + b y_{t-1} + sum_{j=1..lags} c_j Delta y_{t-j} + e_t

`Z(t)` is the t ratio of `b` with the classical standard error (`N - K`
degrees of freedom). H0: unit root. `trend` selects the deterministic terms:
`"none"`, `"constant"` (default), `"trend"`, or `"drift"` (a constant, with a
random walk *with drift* as the null).

- p-value: MacKinnon (1994) approximate asymptotic p-value, the same that
  Stata prints. `drift`: lower tail of Student t with `N - K` degrees of
  freedom, with t critical values, as Stata.
- `critical_values`: MacKinnon (2010) response surfaces
  `b_inf + b1/N + b2/N^2 + b3/N^3` at the regression's N.
- `critical_values_fuller`: the "interpolated Dickey-Fuller" critical values
  that **Stata prints**, interpolated linearly in N in Fuller's table (rows
  N = 25, 50, 100, 250, 500, infinity; the first row below 25 and the last
  above 500). The two sets agree to about two decimals.
- `regress=True` returns a `TableSet` with `test` and `regression` (terms
  `L1.y`, `LD.y`, `L2D.y`, ..., `_trend`, `Intercept`; `_trend` is t - 1, as
  Stata's, so its first value in the regression is `lags + 1`).

Checked against the worked example of Stata's [TS] dfuller (airline data,
`lags(3) trend regress`: statistic -6.936, the whole regression table and the
printed critical values -4.027 / -3.445 / -3.145 are reproduced) and against
`statsmodels.tsa.stattools.adfuller`.

## `pperron`: Phillips-Perron test

OLS of `y_t = a + d t + rho y_{t-1} + u_t` on the `n = T - 1` observations,
then (Stata's Methods and formulas)

    gamma_j  = (1/n) sum_{t>j} u_t u_{t-j},   lambda^2 = gamma_0 + 2 sum_{j<=q} (1 - j/(q+1)) gamma_j
    Z(rho) = n (rho - 1) - (1/2) (n^2 sigma^2 / s^2) (lambda^2 - gamma_0)
    Z(t)   = sqrt(gamma_0/lambda^2) (rho - 1)/sigma - (1/2) (lambda^2 - gamma_0) n sigma / (lambda s)

with `sigma` the OLS standard error of `rho` and `s^2 = SSR/(n - K)`. The
default truncation is `q = int(4 (n/100)^(2/9))`. Z(t) uses the Dickey-Fuller
distribution (MacKinnon p-value; `critical_values` and
`critical_values_fuller` as in `dfuller`). The critical values of Z(rho)
(`critical_values_rho`) are interpolated linearly in `n` in Fuller's table of
`n(rho-hat - 1)`, as Stata prints them; no p-value is published for Z(rho).
The table has two rows, `Z(rho)` and `Z(t)`; `attrs` adds `z_rho`, `z_t`,
`rho`, `long_run_variance`.

Checked against the worked example of Stata's [TS] pperron (`lags(4) trend
regress`: Z(rho) = -46.405, Z(t) = -5.049, p = 0.0002, critical values
-27.687 / -20.872 / -17.643 and -4.026 / -3.444 / -3.144, regression table).

## `dfgls`: DF-GLS test (Elliott, Rothenberg and Stock)

1. GLS detrending with `a* = 1 + c/T`, `c = -13.5` (constant and trend,
   default) or `c = -7` (`trend=False`): the quasi-differenced series is
   regressed on the quasi-differenced deterministic terms and the fitted
   deterministic part is removed from the levels.
2. For `k = 1..maxlag` the Dickey-Fuller regression without deterministic
   terms is run on the detrended series, on the common sample
   `t = maxlag+2..T` (N observations); the statistic is the t ratio of
   `y*_{t-1}`. Default `maxlag = floor(12 (T/100)^0.25)` with T the number of
   observations of the series (Schwert; Stata writes `(T+1)/100` because its
   sample is `y_0..y_T`). All lag orders come from one nested QR factor, so
   the default of 120 lag orders on a million rows takes a few seconds.
3. Lag choices, with `s2_k = SSR_k/N`: Ng-Perron sequential t (largest k whose
   last lag has |t| > 1.645, else 0), SC `ln s2_k + (k+1) ln(N)/N`, and MAIC
   `ln s2_k + 2 (tau_k + k)/N` with `tau_k = b_k^2 sum y*_{t-1}^2 / s2_k`.

The table has one row per lag (`lags`, `statistic`, critical values, `rmse`,
`sc`, `maic`); `attrs["statistic"]` is the statistic at the MAIC lag and
`lags_seq_t`, `lags_sc`, `lags_maic` hold the three choices.

Critical values: with a trend, Elliott-Rothenberg-Stock (1996, Table I)
interpolated linearly at the number of observations of the series between
T = 50, 100 and 200; the T = 50 row below 50 and the asymptotic row above 200
(Stata's `ers` option, and the source of Stata's 1% value in every case; no
p-value). Without a trend the statistic has the no-constant Dickey-Fuller
distribution (MacKinnon critical values and p-value). **Stata's default** 5%
and 10% critical values come from the Cheung-Lai (1995) lag-dependent response
surface, which is not reproduced.

Checked against the worked example of Stata's [TS] dfgls (West German
investment): all eleven statistics, the three lag choices (7, 4, 1), the
printed RMSE, SIC and MAIC values, N = 80, maxlag = 11 and the 1% critical
value -3.610.

## `kpss`: KPSS stationarity test

H0 is stationarity (around a level, or around a trend with `trend=True`).
With `e_t` the residuals of `y` on a constant (and trend) and `S_t` their
partial sums, `statistic = (sum S_t^2 / T^2) / s^2(l)` where `s^2(l)` is the
Bartlett long-run variance with truncation `l`. Like Baum's `kpss` for Stata
the table reports every truncation `0..lags`; the default is Schwert's
`int(12 (T/100)^(1/4))`. `auto=True` uses the Newey-West (1994) automatic
bandwidth as described by Hobijn, Franses and Ooms (1998):
`n = int(4 (T/100)^(2/9))`, `s0 = g0 + 2 sum_{j<=n} g_j`, `s1 = 2 sum j g_j`,
bandwidth `= min(T-1, int(1.1447 ((s1/s0)^2)^(1/3) T^(1/3)))`. (statsmodels
uses `n = int(T^(2/9))` in the same formula, so its automatic bandwidth can
differ by a few lags; statistics agree for equal bandwidths.)

Critical values are those of Kwiatkowski et al. (1992, Table 1). The p-value
is **interpolated** between the four tabulated points and is therefore
bounded to `[0.01, 0.10]`; a note says when the true p-value lies outside.
Note the argument default: `trend=False` (level stationarity), whereas the
Stata command includes a trend unless `notrend` is given.

## `zandrews`: Zivot-Andrews test with one endogenous break

For each candidate break date `TB` the regression

    Delta y_t = mu + beta t + theta DU_t + gamma DT_t + alpha y_{t-1} + sum_j c_j Delta y_{t-j} + e_t
    DU_t = 1(t > TB),   DT_t = (t - TB) 1(t > TB)

is fitted with `DU` (`break_="intercept"`, model A), `DT` (`"trend"`, model B)
or both (`"both"`, model C); the statistic is the minimum t ratio of `alpha`.
Candidates are `TB = int(trim T)+1 .. T - int(trim T)`. `lags` is an integer,
or `"aic"` / `"bic"` / `"t"` to choose it once from the regression without a
break (constant and trend) among `0..maxlag` on a common sample, as Baum's
`zandrews` and statsmodels do. The scan costs O(T K): the break dummies'
cross products are reverse cumulative sums.

`attrs["break_period"]` is the label of observation `TB` (the last one before
the break) and `break_index` its 0-based position. Critical values are the
asymptotic ones of Zivot and Andrews (1992); no p-value is tabulated.
Checked against one regression per break date in NumPy and statsmodels'
`zivot_andrews` (which uses its own simulated critical values and, for the
trend-only model, dates the break one observation later).

## `egranger`: Engle-Granger cointegration test

Step 1: OLS of `y_t` on `x_t` and the deterministic terms (`trend="none"`,
`"constant"`, `"trend"`, `"quadratic"`). Step 2: Dickey-Fuller regression of
the residuals without deterministic terms; the statistic is the t ratio of
`e_{t-1}`. H0: no cointegration. The p-value is MacKinnon's (1994) for
`N = 1 + number of regressors` series (at most 6) and the critical values are
MacKinnon's (2010) response surfaces at the number of observations of step 2.
With `trend="none"` and regressors the 2010 tables have no entry; the
asymptotic critical values implied by the 1994 distribution are reported and
a note says so. `lags` is an integer or `"aic"` / `"bic"` / `"t"` (chosen
among `0..maxlag`). The sample must hold at least six observations more than
the parameters of the cointegrating regression.

The result is a `TableSet`: `test`, `cointegrating_regression` (its standard
errors are shown for reference; they are not valid for inference), and on
request `adf_regression` (`regress=True`) and `ecm` (`ecm=True`):

    Delta y_t = c + a e_{t-1} + sum_{j=1..lags} (g_j Delta y_{t-j} + h_j' Delta x_{t-j}) + u_t

`attrs["cointegrating_vector"]` maps terms to coefficients and
`attrs["adjustment"]` is `a`. Checked against `statsmodels.tsa.stattools.coint`
and explicit OLS regressions.

## `chow`: break at a known date

`break_at` is the first observation of the second regime: a value of the time
column (a period number for numeric time, a date or date string for
datetimes; rows with `time >= break_at` form the second regime), or a 0-based
row position without a time column. Rows of the result:

| row | statistic | distribution |
| --- | --- | --- |
| `chow_f` | `[(SSR_p - SSR_1 - SSR_2)/K] / [(SSR_1 + SSR_2)/(T - 2K)]` | F(K, T-2K) |
| `wald` | `K * chow_f` (Stata's `estat sbknown` chi2 with the classical VCE) | chi2(K) |
| `lr` | `T ln(SSR_p / (SSR_1 + SSR_2))` | chi2(K) |
| `forecast_f` | `[(SSR_p - SSR_1)/n2] / [SSR_1/(n1 - K)]` (Chow's predictive test) | F(n2, n1-K) |

When the second regime is too short or collinear the first three rows are
missing (with a note) and the forecast test is still reported.

## `sbsingle`: break at an unknown date

`W(n1) = (SSR_p - SSR_1 - SSR_2) / ((SSR_1 + SSR_2)/(T - 2K))` is computed for
every candidate first-regime size `n1 = m .. T - m`, `m = max(K+1,
ceil(trim T))`; `test="supwald"` reports the maximum, `"avewald"` the average
and `"expwald"` `ln(mean exp(W/2))`. All three are always in `attrs`
(`sup_wald`, `ave_wald`, `exp_wald`) with the date of the maximum. With
T = 222 and 15% trimming the candidate dates are observations 35 to 189, the
"trimmed sample" of the example in Stata's [TS] estat sbsingle.

- sup-Wald critical values: Andrews (1993) as corrected by Andrews (2003), for
  15% trimming and up to 20 parameters, in the form tabulated by Stock and
  Watson. This is the one table of the family that could **not** be compared
  with a second published transcription; it was checked by simulating the
  limiting process (10% and 5% points within 4.5%) and every result says so
  in its notes.
- sup-Wald p-value: the upper-tail approximation of DeLong (1981) quoted by
  Andrews (1993),
  `P(sup > c) ~ c^(q/2) e^(-c/2) / (2^(q/2) Gamma(q/2)) [(1 - q/c) ln(lambda) + 2/c]`
  with `lambda = pi2 (1 - pi1) / (pi1 (1 - pi2))`. Against simulated quantiles
  (K = 1, 2, 4, 8) it is within 0.02 at p = 0.05 and 0.10, within 0.03 at 0.2
  and within 0.08 at 0.5; above 0.5 it understates the p-value (a note says
  so). At the tabulated critical values it returns 0.9 to 1.25 times the
  nominal level.
  **Stata reports Hansen's (1997) approximation**, whose coefficient tables
  are not reproduced: for the manual's example (sup-Wald 14.1966, K = 3) Stata
  prints 0.0440 and this approximation gives 0.051.
- With one candidate date the statistic is an ordinary Wald statistic and the
  p-value is the chi-squared(K) tail.
- average and exponential Wald: statistic only.

## `cusum`: recursive residuals, CUSUM and CUSUM of squares

Returns one row per recursive residual
`w_t = (y_t - x_t'b_{t-1}) / sqrt(1 + x_t'(X_{t-1}'X_{t-1})^{-1} x_t)`:
`period` (the time value, or the row position), `recursive_residual`,
`cusum`, `cusum_lower`, `cusum_upper`, `cusumsq`, `cusumsq_lower`,
`cusumsq_upper`.

- CUSUM: `W_t = sum_{j<=t} w_j / s` with
  `s^2 = sum (w_j - mean(w))^2 / (T - K)`, as in Stata's `estat sbcusum`
  (Brown, Durbin and Evans, statsmodels and EViews scale by the uncentred
  `sqrt(SSR/(T-K))`, which is reported as `attrs["sigma_ols"]`). The bounds
  are `+-a sqrt(T-K) (1 + 2 (t-K)/(T-K))` where `a` solves
  `Q(3a) + exp(-4a^2)(1 - Q(a)) = alpha/2`: 1.1430, 0.9479, 0.8499 at 1%, 5%,
  10% (the values Stata prints). `attrs["statistic"]` is
  `max |W_t| / (sqrt(T-K)(1 + 2(t-K)/(T-K)))`, to be compared with `a`.
- CUSUM of squares: `S_t = sum_{j<=t} w_j^2 / sum w_j^2` with bounds
  `(t-K)/(T-K) +- c0`; `c0` is the Edgerton-Wells (1994) approximation with
  `m = (T-K)/2 - 1` (the coefficients statsmodels uses).

`cusum_crosses` and `cusumsq_crosses` say whether a path leaves its bounds at
`level`. If the first K observations do not identify the coefficients (a
dummy that switches on later), the recursion starts at the first sample that
does and a note says so. Recursive residuals agree with refitting the
regression at every date and with
`statsmodels.stats.diagnostic.recursive_olsresiduals`. Stata's OLS-residual
CUSUM has a separate `oe.ols_cusum` route, with fixed-design iid Gaussian
conditional simulation rather than a vendor asymptotic critical-value claim;
see [the inference/stability guide](next-eight-inference-stability.md).

## `xtunitroot`: panel unit-root tests

`oe.xtunitroot(data, y, panel, time, test=..., lags=0, trend="constant", demean=False, ...)`.
`lags` is an integer, or `"aic"` / `"bic"` / `"hqic"` for a panel-by-panel
choice among `1..maxlag` on the sample that `maxlag` leaves (Stata's
`lags(aic #)`; the criterion is the Gaussian `-2 ln L + K c` with `c = 2`,
`ln M` or `2 ln ln M`, M the observations of that sample). `demean=True`
subtracts the cross-sectional mean of every period first. **The default is
`lags=0` for every test; Stata's `llc` defaults to one lag.**

| `test` | H0 | panels | statistic and reference |
| --- | --- | --- | --- |
| `llc` | all panels have a unit root | balanced | adjusted `t*` ~ N(0,1), lower tail; LLC (2002) Table 2 |
| `ips` | all panels have a unit root | unbalanced ok | `W-t-bar` ~ N(0,1), lower tail; IPS (2003) Table 3 (lags <= 8) |
| `fisher` | all panels have a unit root | unbalanced ok | `P` ~ chi2(2N), `Z` ~ N(0,1), `L*` ~ t(5N+4), `Pm` ~ N(0,1) from MacKinnon p-values of ADF or PP tests |
| `hadri` | all panels are stationary | balanced | LM `z` ~ N(0,1), upper tail |
| `breitung` | all panels have a unit root | balanced | `lambda` ~ N(0,1), lower tail |
| `ht` | all panels have a unit root | balanced | Harris-Tzavalis `z` ~ N(0,1), lower tail (exact moments for fixed T) |

The formulas are in the function's docstring; they follow the Methods and
formulas of Stata's [XT] xtunitroot, which were read for this
implementation. Conventions worth knowing:

- **LLC** scales each panel by the standard deviation of its Dickey-Fuller
  residual with divisor `T - p_i - 1`, estimates the long-run variance of
  `Delta y` with divisor `T - 1` and `int(3.21 T^(1/3))` Bartlett lags (T
  periods; Stata truncates, R's `plm` rounds, LLC's own Table 2 lists rounded
  values), and interpolates Table 2 linearly at `T~ = T - mean(p_i) - 1`.
  `lrv="paper"` (default) follows the text of LLC (2002) and Stata's manual:
  `Delta y` is used as it is without deterministic terms and **minus its panel
  mean whenever panel means are in the model** (with or without trends).
  The adjustment terms `mu*` of Table 2 are, however, centred for a long-run
  variance of the undemeaned differences when there are panel means only: on
  demeaned differences the kernel estimate is too small by about
  `(lags + 1)/T`, the ratio `S_N` is too small and `t*` is shifted to the
  left. With 250 independent random walks (200 replications, `lags=0`):

  | T | `trend="none"` | `"constant"`, `lrv="paper"` | `"constant"`, `lrv="null"` | `"trend"` |
  | --- | --- | --- | --- | --- |
  | 26 | mean 0.08, 5% size 0.05 | mean -4.58, size 1.00 | mean -0.32, size 0.11 | mean -0.53, size 0.12 |
  | 51 | 0.14, 0.05 | -3.01, 0.95 | -0.13, 0.06 | 0.05, 0.03 |
  | 101 | 0.00, 0.04 | -2.00, 0.62 | -0.11, 0.07 | -0.02, 0.04 |
  | 251 | 0.07, 0.03 | -1.24, 0.35 | -0.13, 0.06 | -0.08, 0.04 |

  The shift grows with `sqrt(N)` and shrinks slowly with T. `lrv="null"`
  removes only what the null hypothesis implies (nothing with panel means,
  the panel mean with trends) and is correctly centred; it is an OpenEconometrics
  option, not a Stata one. Every `llc` result with panel means and the
  default says this in its notes. Use `lrv="null"`, or a small
  `kernel_lags`, unless T is much larger than N.
- **IPS** enters Table 3 with the number of observations of each panel's
  regression (`T_i - p_i - 1`), interpolating linearly in T and using the
  T = 100 column beyond; cells that IPS leave empty raise
  `ips_moments_unavailable`. This is Stata's output when `lags()` is given
  (including `lags(0)`). Stata's output without `lags()` (`t-bar`,
  `t-tilde-bar`, `Z-t-tilde-bar` with exact critical values) is not
  reproduced.
- **Fisher** combines MacKinnon (1994) p-values of each panel's `dfuller`
  (`lags` lagged differences) or `pperron` (`lags` Newey-West lags) statistic.
- **Hadri** uses `sigma^2 = SSR / (N (T - d))`, d = 1 or 2 deterministic
  terms; `robust=True` standardizes each panel by its own variance;
  `kernel_lags` replaces the variance by the average of the panels' long-run
  variances (divisor T).
- **Breitung** follows Stata's formulas (prewhitening on the lagged
  differences, with a constant only when there are trends; `s_i^2` with
  divisor `T - p - 2`; forward orthogonal deviations with trends). With
  trends the transformed level is `u_s - u_1 - (s-1) mean(Du)` as in Breitung
  (2000); Stata's manual prints `(T-p-1) mean(Du)` there, which would not be
  invariant to the panels' trends. The Breitung-Das robust statistic is not
  implemented.
- **HT** uses T = the number of periods in the moment formulas, as Stata's
  default (this reproduces the z of the manual's example); `altt=True` uses
  T - 1, the definition of Harris and Tzavalis. With panel means or trends
  the default over-rejects in short panels (400 random walks of 8 periods:
  mean z -2.6 with panel means and -4.2 with trends, against -0.1 and -0.2
  with `altt=True`), as Stata's manual warns; a note in the result says so.

## `xtcointtest`: Kao test of panel cointegration

`oe.xtcointtest(data, y, x, panel, time, lags=1, kernel="bartlett", kernel_lags=None, demean=False)`
fits `y_it = a_i + x_it'b + e_it` by the within estimator on a balanced panel
and reports Kao's (1999) five statistics as in Stata's Methods and formulas,
each N(0,1) under H0 of no cointegration with rejection in the lower tail:
`modified_df` (DF*_rho), `df` (DF*_t), `adf`, `unadjusted_modified_df`
(DF_rho), `unadjusted_df` (DF_t). The conditional variances `s2_v` and `s2_0v`
come from the covariance and the long-run covariance (within panels) of
`w_it = (Delta y_it, Delta x_it')`. The default bandwidth is the fixed
`int(4 (T/100)^(2/9))`; **Stata's default is the Newey-West automatic
bandwidth**, so pass `kernel_lags` to compare. `bandwidth="auto"` explicitly
adds per-unit automatic Bartlett selection. Pedroni and Westerlund (2005)
have separate [heterogeneous method contracts](panel-cointegration.md).

## Worked example

```python
import numpy as np, pandas as pd, openecon as oe

rng = np.random.default_rng(3)
T = 200
income = np.cumsum(rng.normal(0.2, 1, T))                 # random walk with drift
consumption = 5 + 0.8 * income + rng.normal(0, 1, T)      # cointegrated with income
df = pd.DataFrame({"quarter": np.arange(T), "income": income, "consumption": consumption})

adf = oe.dfuller(df, "income", time="quarter", lags=2, trend="trend")
adf.attrs["statistic"], adf.attrs["p_value"]              # -1.618, 0.785: unit root not rejected
adf.attrs["critical_values"]["5%"], adf.attrs["critical_values_fuller"]["5%"]   # -3.433, -3.437
pp = oe.pperron(df, "income", time="quarter", trend="trend")
pp.attrs["z_rho"], pp.attrs["z_t"], pp.attrs["lags"]      # -6.67, -1.80, 4 Newey-West lags
kpss = oe.kpss(df, "income", time="quarter", trend=True, lags=4)
kpss.attrs["statistic"], kpss.attrs["p_value"]            # 0.708 > 0.216, p <= 0.01: not trend stationary
gls = oe.dfgls(df, "income", time="quarter")
gls.attrs["lags_maic"], gls.attrs["statistic"]            # 1 lag, -1.06 (5% critical value -2.93)
za = oe.zandrews(df, "income", time="quarter", break_="both", lags="aic")
za.attrs["statistic"], za.attrs["break_period"]           # -3.86 (5% critical value -5.08), period 56
eg = oe.egranger(df, "consumption", ["income"], time="quarter", ecm=True)
eg.attrs["statistic"], eg.attrs["p_value"]                # -14.9, 1e-26: cointegrated
eg.attrs["cointegrating_vector"], eg.attrs["adjustment"]  # slope 0.80, intercept 4.95; adjustment -1.13
oe.chow(df, "consumption", ["income"], 100, time="quarter").loc["chow_f", "p_value"]   # 0.13
sb = oe.sbsingle(df, "consumption", ["income"], time="quarter")
sb.attrs["statistic"], sb.attrs["p_value"]                # 7.72 (5% critical value 11.72), p 0.23
path = oe.cusum(df, "consumption", ["income"], time="quarter")
path.attrs["statistic"], path.attrs["cusum_crosses"], path.attrs["cusumsq_crosses"]   # 0.38, False, False

rng = np.random.default_rng(4)
N, P = 20, 40
walk = rng.normal(size=(N, P)).cumsum(axis=1) + rng.normal(size=(N, 1)) * 5
x = rng.normal(size=(N, P)).cumsum(axis=1)
y = 2 + 0.6 * x + rng.normal(size=(N, P))
panel = pd.DataFrame({"id": np.repeat(np.arange(N), P), "year": np.tile(np.arange(1980, 1980 + P), N),
                      "walk": walk.ravel(), "x": x.ravel(), "y": y.ravel()})

oe.xtunitroot(panel, "walk", "id", "year", test="ips", lags=1).attrs["p_value"]       # 0.23
oe.xtunitroot(panel, "walk", "id", "year", test="fisher", lags=1)                     # P, Z, L*, Pm
oe.xtunitroot(panel, "walk", "id", "year", test="hadri").attrs["p_value"]             # 0.0: not stationary
llc = oe.xtunitroot(panel, "walk", "id", "year", test="llc", lags=1)
llc.attrs["statistic"], llc.attrs["p_value"]              # -1.85, 0.032: the published step rejects ...
llc.attrs["notes"][1]                                     # ... and the note explains why
oe.xtunitroot(panel, "walk", "id", "year", test="llc", lags=1, lrv="null").attrs["p_value"]   # 0.18
kao = oe.xtcointtest(panel, "y", ["x"], "id", "year")
kao.attrs["statistic"], kao.attrs["cointegrating_vector"]   # ADF -7.41; slope 0.597
```

## Tables and their sources

| table | source | second source it was compared with |
| --- | --- | --- |
| Dickey-Fuller / Engle-Granger p-values | MacKinnon (1994), Tables 3-4, N = 1..6 | identical to statsmodels' `adfvalues` |
| Dickey-Fuller / Engle-Granger critical values | MacKinnon (2010), Tables 1-4, N = 1..6 | identical to statsmodels' `adfvalues` |
| `n(rho-1)` critical values for Z(rho) | Fuller (1976) Table 8.5.1 = Hamilton (1994) Table B.5 | identical to R `fUnitRoots::adfTable` (all cases) and R `tseries::pp.test` (trend); Stata's printed values in [TS] pperron; MacKinnon's z distribution; Monte Carlo |
| Dickey-Fuller t (`critical_values_fuller`) | Fuller (1976) Table 8.5.2 = Hamilton (1994) Table B.6 | identical to R `fUnitRoots::adfTable` and `tseries::pp.test`; Stata's printed values in [TS] dfuller and [TS] pperron; within 0.04 of MacKinnon (2010) |
| DF-GLS with trend | Elliott, Rothenberg and Stock (1996), Table I | identical to R `urca::ur.ers`; Stata's printed 1% value in [TS] dfgls; Monte Carlo |
| KPSS | Kwiatkowski et al. (1992), Table 1 | identical to statsmodels; Monte Carlo (within 4%) |
| Zivot-Andrews | Zivot and Andrews (1992), Tables 2-4 | identical to R `urca::ur.za`; Monte Carlo (T = 160, within 0.22) |
| LLC `mu*`, `sigma*` | Levin, Lin and Chu (2002), Table 2 | identical to R `plm` (`adj.levinlin`); Monte Carlo of the centring |
| IPS `E[t]`, `Var[t]` | Im, Pesaran and Shin (2003), Table 3 | identical to R `plm` (`adj.ips.wtbar`); Monte Carlo of selected cells |
| sup-F / sup-Wald, 15% trimming | Andrews (2003) as tabulated by Stock and Watson | **none**: simulation of the limiting process and DeLong's tail formula only |
| CUSUM `a` | Brown, Durbin and Evans (1975), eq. (2.6), solved numerically | Stata's printed 1.1430 / 0.9479 / 0.8499 |
| CUSUM of squares | Edgerton and Wells (1994) | identical to statsmodels' coefficients |
| Hadri and Harris-Tzavalis moments | closed forms in Hadri (2000), Harris and Tzavalis (1999) | Stata's Methods and formulas; Monte Carlo under the null |

The R sources (CRAN mirrors of `plm`, `urca`, `fUnitRoots`, `tseries`) and the
Stata 19 manuals were read during verification; the tests hold independent
transcriptions of those sources and compare them with `tables.py` entry by
entry. The Andrews table is the exception stated above.

The MacKinnon (2010) rows for more than six series are not stored (the
p-values of 1994 stop at six), so `egranger` accepts at most five regressors.

## Limitations

- No `missing="drop"`: series and panels must be complete and gap free.
- `dfgls`: ERS critical values only (Stata's default Cheung-Lai surface is not
  reproduced; without a trend Stata interpolates a table where MacKinnon's
  surface is used here); no p-value with a trend.
- `kpss`: Bartlett kernel only; p-value bounded to [0.01, 0.10].
- `zandrews`: no p-value; one lag order for all break dates.
- `egranger`: at most five regressors. Separate `po_za` / `po_zt` procedures
  supply Phillips–Ouliaris inference for the calibrated 2–6-series domain;
  see [inference extensions](inference-extensions.md) for cases and limits.
- `sbsingle`: classical Wald only (no robust VCE, no LR variants, no tests on
  a subset of coefficients); approximate sup-Wald p-value instead of Hansen's
  (1997); no p-values or critical values for average / exponential Wald.
- `cusum`: recursive residuals only; separate `ols_cusum` uses declared
  conditional fixed-design iid Gaussian OLS-residual CUSUM/CUSUMSQ calibration.
- `xtunitroot`: IPS `t-tilde-bar` statistics and exact critical values and the
  Breitung-Das robust statistic are not implemented; LLC, Hadri, Breitung and
  HT need balanced panels; no Fisher `drift` variant.
- `xtcointtest`: balanced complete panels; Kao preserves its historical fixed
  bandwidth default and admits explicit automatic selection. Pedroni and
  Westerlund use separate published numerical large-T moment approximations;
  current Stata finite-sample calibration is not reproduced.

## What was checked against Stata, and what could not be

`stata_parity_validated` is false: no Stata run was available. Three worked
examples of the Stata manuals use series that are reproduced in the tests, so
`dfuller`, `pperron` and `dfgls` are checked digit for digit against printed
Stata output. For the other procedures the Methods and formulas of the
manuals were followed and the following choices remain uncertain:

1. `pperron`: the default truncation uses `n = T - 1` (the manual writes
   `T`); Z(rho) critical values use the first row below n = 25.
2. `dfgls` without a trend: Stata interpolates a table between 50 and 500
   observations; MacKinnon's no-constant surface is used here.
3. `kpss`: default truncation and automatic bandwidth as documented for
   Baum's `kpss` (the preliminary `n = int(4 (T/100)^(2/9))`).
4. `zandrews`: trimming range `int(trim T)+1 .. T - int(trim T)`, break dating
   (label of `TB`) and the 1.645 threshold of the sequential t rule.
5. `egranger`: critical values at the number of observations of the residual
   regression; ECM with lagged differences of y and x and a constant only.
6. `sbsingle`: the candidate range reproduces the "trimmed sample" printed in
   Stata's example, but the progress dots there suggest 154 tests where this
   range has 155; Wald with `s^2 = SSR_u/(T - 2K)`.
7. `cusum`: the `sqrt(T-K)` scaling of the statistic and bounds (the manual's
   formula omits it; the printed critical values require it).
8. `xtunitroot`: the information criteria use the observations of the common
   sample (the manual writes `M = T - p_max - 2`, one fewer); `ips` enters
   Table 3 with the observations of each regression; `breitung` sums over
   `s = 1..T-p-2` as in Breitung (2000) (the manual's sums run to `T-p-1`)
   and uses `(s-1) mean(Du)` (see above); combining Hadri's `robust` with
   `kernel_lags` divides each panel by its own long-run variance.
9. `xtcointtest kao`: `w = (Delta y, Delta x)` without demeaning, the divisor
   of `s_v^2` (observations used), `T` (periods) in `sqrt(N) T (rho - 1)`,
   the degrees of freedom of the pooled ADF regression's t ratio (pooled
   observations minus parameters), and the fixed default bandwidth.

## Performance

Measured on an Apple-silicon laptop CPU (while other jobs were running), one
million rows:

| call | time |
| --- | --- |
| `dfuller` (8 lags, trend) | 0.1-0.3 s |
| `pperron` / `kpss` (default lags) | 0.02-0.1 s |
| `kpss` (`auto=True`) | 0.1-0.4 s |
| `dfgls` (8 lag orders) / default (120 lag orders) | 0.1-0.2 s / 1.6-5 s |
| `zandrews` (both breaks, 4 lags; 700,000 candidate dates) | 0.4-0.7 s |
| `zandrews` / `egranger`, `lags="aic"` with the default maxlag (120) | about 2 s |
| `egranger` (2 regressors, 2 lags, ECM) | 0.2-0.3 s |
| `chow` / `sbsingle` / `cusum` (9 regressors) | 0.2-0.3 s / 0.8-0.9 s / 0.8-0.9 s |
| `xtunitroot` on 10,000 panels x 100 periods: `llc`, `ips`, `fisher`, `hadri`, `breitung`, `ht` | 0.06-0.2 s each |
| `xtunitroot ips`, `lags="aic"`, `maxlag=4` | 0.15-0.3 s |
| `xtunitroot` on 100,000 panels x 12 periods | 0.1-0.4 s each |
| `xtcointtest` (Kao, 2 regressors), 10,000 panels x 100 periods | 0.2-0.35 s |

Lag-order comparisons (`dfgls`, `lags="aic"` in `zandrews` and `egranger`)
are nested regressions computed from a single block-wise QR factor: O(T
maxlag^2) instead of O(T maxlag^3).

## References

Andrews (1993, 2003); Andrews and Ploberger (1994); Breitung (2000); Brown,
Durbin and Evans (1975); Cheung and Lai (1995); Choi (2001); Chow (1960);
DeLong (1981); Dickey and Fuller (1979); Edgerton and Wells (1994); Elliott,
Rothenberg and Stock (1996); Engle and Granger (1987); Fuller (1976, 1996);
Hadri (2000); Hamilton (1994); Hansen (1997); Harris and Tzavalis (1999);
Hobijn, Franses and Ooms (1998); Im, Pesaran and Shin (2003); Kao (1999);
Kwiatkowski, Phillips, Schmidt and Shin (1992); Levin, Lin and Chu (2002);
MacKinnon (1994, 2010); Maddala and Wu (1999); Newey and West (1994); Ng and
Perron (1995, 2001); Phillips and Perron (1988); Schwert (1989); Zivot and
Andrews (1992). Stata 19 manuals: [TS] dfuller, dfgls, pperron, estat
sbsingle, estat sbcusum; [XT] xtunitroot, xtcointtest. Full citations of the
tables are in `openecon/econometrics/unitroot/tables.py`.
