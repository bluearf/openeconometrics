# Anderson–Hsiao dynamic panel IV

`oe.ahreg` estimates an AR(1) panel equation with individual effects and optional
strictly exogenous numerical predictors. It implements the two Anderson–Hsiao
instrument choices directly; it is a separate estimator from `xtdpd`/`xtabond`
difference GMM and `xtdpdsys` system GMM.

```python
import openecon as oe

model = oe.ahreg(
    data=df, y="income", x=["investment"], panel="firm", time="year",
    instrument="levels",             # default: y(t-2)
)
other = oe.ahreg(
    data=df, y="income", x=["investment"], panel="firm", time="year",
    instrument="differences",        # y(t-2) - y(t-3)
)
display(model)
model.to_latex()
```

The default covariance clusters on the declared panel. `covariance="robust"`
is an alias for the same panel-clustered covariance, not a row-wise White
covariance. `covariance="nonrobust"` selects the homoskedastic-level-error
formula described below. There is no intercept after removing the individual
effect. The function accepts `x=None` for the pure AR(1) model, `missing="raise"`
(default) or `missing="drop"`, and `alpha` for confidence intervals. An explicit
`cluster` must be the same column as `panel`.

## Model and instrument choices

The level equation is

\[
y_{it}=\rho y_{i,t-1}+x_{it}'\beta+a_i+\varepsilon_{it}.
\]

First differencing gives

\[
\Delta y_{it}=\rho\Delta y_{i,t-1}+\Delta x_{it}'\beta+\Delta\varepsilon_{it}.
\]

`D.L.y` is endogenous because it contains the previous level error. The one
excluded instrument is either \(y_{i,t-2}\) or
\(\Delta y_{i,t-2}=y_{i,t-2}-y_{i,t-3}\). The included instruments are
\(\Delta x_{it}\). The resulting model is exactly identified, with one
instrument per coefficient. The assumptions are independent panels, absence
of serial correlation in the level errors, lagged outcomes uncorrelated with
future errors, and strict exogeneity of every supplied `x` with respect to all
level errors. Fixed effects may correlate with the predictors. Clustered
standard errors allow heteroskedasticity and within-panel score dependence;
they do not make invalid lag instruments valid.

The two AH estimators use a single pooled instrument column rather than the
period-specific instrument matrix of Arellano–Bond GMM. They are generally
less efficient than GMM that exploits additional valid lag moments. No
predetermined or endogenous `x`, extra excluded instruments, higher dynamic
orders, system equations or orthogonal deviations are silently inferred; use
the dynamic GMM API for those specifications.

## Samples, gaps and periods

Input rows are sorted by panel and time. Time must contain integer period
codes with unit spacing; dates must first be converted to an explicit regular
period code, such as year or year-quarter. Dates are not ranked automatically,
because ranking distinct dates could incorrectly close a missing period.
Signed int64 periods remain exact even beyond the float64 exact-integer range.
Repeated times within a panel are rejected.

For the level instrument, an equation needs observations at `t`, `t-1` and
`t-2`; the difference instrument additionally needs `t-3`. The first two or
three periods are therefore excluded. Unbalanced panels and gaps are allowed;
all required periods must actually exist in the same panel. `missing="drop"`
first excludes rows with missing model inputs and then forms windows, so a
dropped input row creates a real gap. This is a conservative listwise policy:
even missing `x` at a row needed only for lagged `y` breaks that window.
Non-numeric or infinite inputs fail explicitly, as do time-invariant/collinear
differenced predictors, irrelevant instruments, perfect fits, insufficient
degrees of freedom and fewer than two panels with usable equations.

`sample_positions` refers to original input row positions for the current
period of each retained equation. Predictions, residuals and R-squared use
the first-difference equation, not fitted levels or estimated individual
effects. R-squared is uncentered and can be negative. Lag exclusions, missing
rows, contributing panel counts and transformed observations are recorded.

## Estimation and covariance conventions

Let \(Y=\Delta y\), \(X=[\Delta y_{-1},\Delta x]\),
\(Z=[\Delta x,z_{AH}]\), \(\widehat X=P_ZX\), and
\(B=(X'P_ZX)^{-1}\). The coefficient estimator is

\[
\widehat b=B X'P_ZY.
\]

Production reuses the shared Torch float64 two-stage Householder QR kernels,
with no explicit projection matrix or normal-equation inverse. Lag assembly
is linear in input rows and does not allocate a panel-by-time grid. Covariance
uses coefficient-sized cross-products; no observation-by-observation matrix
is formed. This estimator currently accepts in-memory tables, not streaming
`Dataset` sources, and does not promise GPU execution or arbitrary row counts.

For default `cluster` and the `robust` alias, let
\(s_i=\widehat X_i'\widehat u_i\), \(N\) be retained equations,
\(K\) coefficients and \(G\) contributing panels. Then

\[
\widehat V_{CR1}=
\frac{G}{G-1}\frac{N-1}{N-K}
B\left(\sum_i s_i s_i'\right)B'.
\]

Coefficient tests use Student t with `G-1` degrees of freedom; the joint model
test uses F. The finite-sample factor is recorded in inference metadata.

For `nonrobust`, conditionally homoskedastic serially uncorrelated level errors induce
\(\operatorname{Var}(\Delta\varepsilon)=\sigma^2 H\), where `H` has diagonal 2
and off-diagonal −1 between consecutive equations within the same panel.
Here the constant conditional error variance is with respect to the relevant
instrument histories and the strictly exogenous predictors. There are no
links across gaps or between panels. We use the explicit
finite-sample variance convention

\[
\widehat\sigma^2=\frac{\widehat u'\widehat u}{2(N-K)},\qquad
\widehat V_H=\widehat\sigma^2 B\widehat X'H\widehat XB'.
\]

This variance estimate is an asymptotic plug-in with a declared finite-sample
divisor; it is not asserted to be an unbiased finite-sample AH estimator.
The sparse implementation computes
`2 * Xhat.T @ Xhat` minus both cross-products of adjacent retained equations.
Inference uses t with `N-K` degrees of freedom. This is deliberately different
from treating the differenced residuals as independent rows.

## Relevance and validity diagnostics

`extra["first_stage"]` records the excluded-instrument coefficient, partial
R-squared and a panel-clustered excluded-instrument F statistic with `G-1`
denominator degrees of freedom, even if the structural covariance is
`nonrobust`. This reports relevance; it does not provide weak-instrument-robust
AR/CLR confidence sets or a universal strength threshold. Near-unit-root
series can produce weak lag instruments and imprecise IV estimates.

`tests["overidentification"]` reports zero degrees of freedom and no statistic:
an exactly identified AH equation cannot test instrument validity with a
Sargan/Hansen overidentification test. Serial-correlation tests are not
implemented by `ahreg`; the existing dynamic GMM result includes
Arellano–Bond tests for its own fitted model and sample.

## Generic specifications and persistence

The registry, analysis API, JSON result persistence, summaries and LaTeX tables
use the same estimator. Generic specifications must explicitly set the panel
cluster and disable the intercept:

```python
from openecon.analysis import fit
from openecon.models import ModelSpec

spec = ModelSpec(
    estimator="ahreg", outcome="income", predictors=["investment"],
    panel="firm", time="year", intercept=False,
    covariance="cluster", cluster="firm", options={"instrument": "levels"},
)
result = fit(spec, data=df)
```

The authenticated `/api/analyses` endpoint accepts this spec. No estimator
implementation is imported merely to enumerate the manifest. No numerical
production dependency on NumPy, statsmodels or linearmodels is introduced.
Independent NumPy tests assemble windows by panel-period dictionaries and
check coefficients, full covariance matrices, both instruments, gaps,
listwise missingness and first-stage diagnostics against literal IV formulas.

## References and software comparison

- Anderson, T. W. and C. Hsiao (1981), *Estimation of Dynamic Models with Error
  Components*, JASA 76(375), 598–606.
  [Publisher DOI](https://doi.org/10.1080/01621459.1981.10477691).
- Anderson, T. W. and C. Hsiao (1982), *Formulation and Estimation of Dynamic
  Models Using Panel Data*, Journal of Econometrics 18(1), 47–82.
  [Publisher DOI](https://doi.org/10.1016/0304-4076(82)90095-1).
- Arellano, M. and S. Bond (1991), *Some Tests of Specification for Panel Data:
  Monte Carlo Evidence and an Application to Employment Equations*, Review
  of Economic Studies 58(2), 277–297, §2: AH instrument comparison and the
  first-difference error matrix `H`.
  [Paper](https://pages.stern.nyu.edu/~wgreene/Econometrics/Arellano-Bond.pdf),
  [Publisher DOI](https://doi.org/10.2307/2297968).
- [Stata `xtivreg` manual](https://www.stata.com/manuals/xtxtivreg.pdf): FD2SLS
  automatically differences supplied instruments. A level-instrument AH
  comparison therefore needs direct IV on the manually differenced equation;
  mechanically specifying `L2.y` to `xtivreg, fd` instead selects a differenced
  lag instrument and loses one additional initial period. Direct 2SLS on
  `D.y`, with included `D.x` and endogenous `L.D.y` instrumented by `L2.y` or
  `L2.D.y`, is the matching coefficient specification.

No live Stata run is claimed. In particular, the declared finite-sample
covariance and inference conventions are independently tested; they are not
marketed as proven bit-for-bit Stata output parity. Provenance retains
`stata_parity_validated=False`.
