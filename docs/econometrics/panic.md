# PANIC: common and idiosyncratic panel unit roots

`oe.xtpanic` implements a bounded version of Bai and Ng's 2004
PANIC procedure using float64 CPU tensors. It tests estimated components rather
than fitting a forecasting model. Its reference distributions are asymptotic;
neither a short panel nor selecting a factor count establishes those assumptions.

```python
tests = oe.xtpanic(
    data=df, y="output", panel="country", time="year",
    factors=2, lags=1, trend="constant", method="mqc",
)
display(tests["common"])
display(tests["idiosyncratic"])
tex = tests.to_latex(notes=tests.attrs["notes"])
```

The return value is a `TableSet` of OpenEconometrics data frames. `common` contains the
single-factor ADF test or the sequential common stochastic-trend tests.
`idiosyncratic` contains each unit's component ADF statistic. With
`pooling="independent"`, `pooled` contains the standardized Fisher statistic.
Tables export to LaTeX and JSON; result settings, assumptions, source and tensor
workspace estimates are in `tests.attrs`. Pass its notes explicitly when
exporting; `TableSet` does not automatically copy metadata into LaTeX captions
or notes.

## Data and factor extraction

The sample must have distinct `y`, `panel`, `time` columns, unique panel/time
keys, no missing or infinite values, and the same complete consecutive calendar
in every unit. Exact integer periods, including int64/uint64 periods beyond
2^53, and regular datetime calendars are accepted. Floating periods must be
exact integers within 2^53−1. No gap is compressed, missing row dropped, or
unbalanced observation imputed. Input order does not determine lag order.

The factor count is a positive user-supplied integer, smaller than both N and
T−1, or `icp1`/`icp2`/`icp3` for automatic Bai–Ng information-criterion selection
over 0..`max_factors`. Zero selected factors return idiosyncratic tests only.
Unbalanced factor estimation and recursive/new-data prediction are not provided.
Correct factor specification is an inference condition, not something this
procedure can verify. Numerically unidentified factors, an unresolved repeated
singular value at the selected PCA boundary, or a unit with no resolvable
idiosyncratic component cause an explicit error.

For `trend="constant"`, PCA is extracted from **raw** first differences.
Time demeaning those differences would remove information in this case. For
`trend="trend"`, each unit's mean first difference is removed before PCA. Neither
case uses cross-sectional centering or standardizes each unit's variance. One
global measurement scale protects the tensor operations without changing the
relative PCA weights.

Let the differenced T−1 by N matrix be x. The score normalization is
f̂′f̂/(T−1)=I, the loadings are x′f̂/(T−1), and the residual differences are
ẑ=x−f̂Λ̂′. Factors and residual differences are cumulated from period 2 through
period T. Sign changes or an orthogonal rotation within the retained factor
space leave the component projections and multivariate tests unchanged.

## Supported inference

| Component | Constant case | Trend case |
|---|---|---|
| Individual idiosyncratic component | ADF without deterministic regressors; asymptotic no-constant Dickey–Fuller p-value | Statistic only by default; explicit `inference="bridge"` enables recorded Brownian-bridge Monte Carlo calibration |
| One common factor | ADF with a constant; asymptotic constant-case Dickey–Fuller reference | ADF with constant and linear trend; asymptotic trend-case Dickey–Fuller reference |
| Multiple common factors | Sequential MQ_c or MQ_f; original Table I constant-case quantiles | Sequential MQ_c or MQ_f; original Table I trend-case quantiles |
| Pooled idiosyncratic components | Opt-in standardized Fisher, requiring cross-unit independence | Available with `inference="bridge"` and the same independence opt-in |

In the trend case the individual limit is
−1/(2√∫₀¹B(s)²ds), where B is a standard Brownian bridge. It is **not** the
no-constant Dickey–Fuller distribution. This implementation does not put standard
ADF p-values or critical values beside these statistics. The optional calibration,
automatic lag/factor choices and their error/assumption records are described in
[the inference-extension contract](inference-extensions.md#panic-selection-and-trend-calibration).

The constant-case individual and single-factor common p-values use the existing
native MacKinnon approximate asymptotic response surface. Their critical values
are the infinite-sample coefficients, not finite-T observed-series corrections
applied to estimated factors. No finite-sample calibration is implied.

With `pooling="independent"`, P=(−2Σlog p_i−2N)/√(4N) uses an upper-tail standard
normal limit under the joint large-N/T conditions. The null is that **all**
idiosyncratic components have unit roots; rejection does not establish that every
unit is stationary. The paper's weak idiosyncratic cross-sectional dependence
assumption for individual tests is insufficient for this pooling: the opt-in
explicitly assumes independent unit innovations. The assumption is not estimated
or checked by the procedure. Log probabilities are accumulated using a stable
native log-CDF, without clipping tiny probabilities to arbitrary positive values.
A statistic below the response surface's published lower boundary prevents
finite pooled inference and raises an error; `pooling="none"` remains available.

`inference="none"` disables all p-values, quantiles, rejections and estimated
stochastic-trend counts. It also evaluates every requested MQ dimension rather
than stopping at a quantile. This is useful for inspecting component statistics
without claiming their calibration applies.

## Multivariate common tests and finite sample convention

For r>1, cumulated factors are demeaned (constant case) or regressed on a
constant and linear trend (trend case). Start at m=r. For each dimension m,
project on the m leading eigenvectors of the detrended-factor second moment and
test the null that the common space has m stochastic trends. A rejection at
the chosen `level` lowers m by one; the first non-rejection stops the sequence.
If every dimension rejects, the estimate is zero. This is the paper's
sequential rule, not a claim of finite-sample exact error control.

`method="mqc"` fits a VAR(1) in these projected levels and uses Bartlett weighted
one-sided residual autocovariances. If J is `bandwidth`, its weights are
1−j/(J+1), j=1..J, and the covariance divisor is the original T. The symmetric
coefficient matrix uses half the contemporaneous/lagged cross-product sum minus
the symmetric Bartlett correction. Its eigenvalues are computed as a symmetric
generalized eigenvalue problem using Cholesky whitening; no explicit inverse or
potentially complex nonsymmetric eigensolver is used. MQ_c=T(λ_min−1).
The default finite J is min(T−2, floor(4(T/100)^(2/9))); J must be smaller than
the VAR residual count. Valid asymptotics require J→∞ with
J/min(√N,√T)→0, not merely passing this finite sample guard.

`method="mqf"` fits a VAR(p) in projected factor differences with
`p=var_lags`, then filters the projected **levels** using I−Π₁L−…−Π_pL^p.
The same symmetric lagged cross-product eigenvalue statistic, without the
Bartlett correction, is used on the filtered levels. A finite-order VAR
representation is required and the supplied p must be at least the true order.
p=0 corresponds to identity filtering. Unavailable presample lag rows are
discarded, never zero padded. MQ_c and MQ_f share the published asymptotic
reference, but their short-run assumptions differ.

ADF operates on the explicitly estimated levels at t=2..T: at lag order p its
regression has T−2−p observations. It does not add a zero first observation to
the ADF residual variance. MQ's sums need the preceding level at t=1. The
cumulative anchor F̂₁=0 supplies that lag; the deterministic projection is fitted
over t=2..T and the same fitted deterministic curve is extrapolated to t=1.
MQ uses the original T multiplier and Bartlett divisor, with valid lag pairs
only. These finite sample conventions are explicit and independently tested;
exact agreement with the authors' unavailable Matlab archive or Stata is not
claimed.

Table I covers m=1..6 and the 1%, 5%, 10% left-tail quantiles. Only those values
are provided; there is no interpolation, extrapolation or MQ p-value. `level`
must be .01, .05 or .10. The printed trend/m=5/10% cell is −55.286: this value
has been checked visually against the original PDF and is preserved rather
than silently amended. Statistics for r>6 require `inference="none"`.
One-factor common testing uses ADF instead of the MQ sequence; unused nondefault
MQ options are rejected. Fixed `lags` applies to individual and single-factor
ADF; `lags="aic"`, `"bic"` or `"t"` selects component ADF lags on a common
comparison sample up to `maxlag` and records every chosen lag.

## Components and resource limits

`components=True` additionally returns `factors`, `loadings`,
`idiosyncratic_levels`, and the scaled singular-value `spectrum`. The first
three reconstruct increments measured from a zero period-1 anchor, not the
original absolute levels, unit intercepts, or original linear trends. Factor
and idiosyncratic rows use original **sorted period positions** 2..T; the sample
calendar endpoints are in `time_range`. The idiosyncratic table uses typed panel
labels in long form so identifiers such as integer 1 and string "1" remain
distinct. Factors/loadings are estimated on the whole supplied sample; there
is no saved-fit residual extraction or `newdata` projection interface.

The implementation uses an exact dense thin SVD. It avoids creating an
unbounded covariance matrix, but its thin U and V arrays still grow with
min(N,T−1). `memory_mb` (1..1024, default64) guards a conservative estimate of
live tensor decomposition, VAR, and blocked ADF workspace. `block_size`
(1..512, default128) is reduced automatically to fit that budget. Input frames,
panel sorting/index arrays, result frames and private BLAS workspace are not
included; the setting is **not** a process-memory cap. Optional long component
tables can be large and should be omitted when only tests are needed.

An additional work guard requires N(T−1)min(N,T−1)≤250,000,000. Factor count is
capped at32, ADF lag order at64, and filtered VAR order at16. These are
implementation resource limits, not statistical validity thresholds. The
procedure is not a GPU, streaming, or arbitrary-large-file PANIC solver. Its
float64 CPU context is local and restores a caller's prior default device;
it does not change the default dtype, random generator state or thread count.

## References and verification

The primary method is [Bai and Ng (2004), *A PANIC Attack on Unit Roots*,
Econometrica72(4),1127–1177](https://doi.org/10.1111/j.1468-0262.2004.00528.x),
especially equations4–8, Theorems1,3,4, and Table I. The
[published full paper](https://users.ssc.wisc.edu/~behansen/718/BaiNg2004.pdf)
provides the table and the distinct trend-case distribution.
This is the original 2004 component/standardized-Fisher procedure, not the
different pooled P_a/P_b tests in the authors' 2010 extension.

The dedicated tests compare Torch results with independent NumPy covariance
eigendecomposition, explicit OLS/covariance calculations, and SciPy symmetric
generalized eigenvalues/log-CDF. They cover projection/cumulative reconstruction,
both deterministic cases, both MQ corrections, all sequential dimensions,
factor signs/rotations, global units/offsets, input permutations, block sizes,
typed panel keys, large integer/date identity, JSON/LaTeX exports and refusal of
unsupported inference/rank/calendar/workspace domains. These checks establish
the implemented finite sample algebra, not universal finite sample size/power
or complete Stata parity.
