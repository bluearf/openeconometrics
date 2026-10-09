# Multivariate GARCH

`oe.mgarch_ccc`, `oe.mgarch_dcc`, and `oe.mgarch_bekk` estimate Gaussian
GARCH(1,1) systems with joint conditional maximum likelihood. Every constant
mean, variance, and correlation parameter is estimated together. The reported
covariance is the inverse full observed information in physical reporting units;
it includes the nuisance cross blocks. Runtime numerical work uses native
float64 PyTorch matrix recursions and derivatives.

```python
result = oe.mgarch_dcc(frame, ["return_a", "return_b"], time="period",
                       intercept=True, tolerance=1e-7)
forecast = oe.forecast(result, 5)
result.to_latex()
forecast.to_latex(index=False)
```

| Model | Dimension | Correlation/covariance parameters | Full covariance forecast |
| --- | --- | --- | --- |
| CCC | 2 or 3 | All entries of an estimated SPD fixed correlation target | Exact one step; later off-diagonal entries use plug-in scaling |
| DCC | 2 or 3 | Estimated SPD target and scalar DCC coefficients | Exact one step; later Q-forward and variance plug-in approximation |
| Full BEKK | 2 | Lower triangular C and every entry of A and B | Exact conditional covariance expectation at every horizon |

This is bounded method coverage. CCC/DCC support no exogenous mean equation,
asymmetry, higher orders, Student-t innovations, panel pooling, weights, or robust
sandwich covariance. VECH, VCC, and component MGARCH are outside this domain.
The full BEKK model includes both off-diagonal entries in each of A and B; it is
not a diagonal BEKK alias. No Stata coefficient or uncertainty parity is claimed.

## Likelihood and constraints

Write `e_t = y_t - mu` with either a constant vector `mu` or a zero vector when
`intercept=False`. Every observation, including the first, contributes

`l_t = -0.5 * [d log(2 pi) + log|H_t| + e_t' H_t^-1 e_t]`.

Cholesky factors supply the determinant and quadratic form. No jitter, eigenvalue
clipping, PSD projection, or singular pseudo-inverse is used.

CCC/DCC marginal variances satisfy

`h_it = omega_i + alpha_i e_i,t-1^2 + beta_i h_i,t-1`.

CCC uses `H_t = D_t R D_t`, where `D_t = diag(sqrt(h_t))`. DCC instead uses

`Q_t = (1-a-b) Qbar + a z_t-1 z_t-1' + b Q_t-1`,

`R_t = diag(Q_t)^-1/2 Q_t diag(Q_t)^-1/2`, `H_t = D_t R_t D_t`,

where `z_t = D_t^-1 e_t`. Omega, alpha, beta, a, and b are strictly positive;
`alpha_i + beta_i < 1-1e-6` and `a+b < 1-1e-6`. Exponential and simplex maps
enforce these interior constraints. A normalized lower triangular factor with
positive diagonal enforces an SPD unit-diagonal target. All pairwise correlations
are reported, including the three distinct entries when d=3.

The DCC target is estimated jointly as a free SPD unit-diagonal parameter matrix.
It is **not** fixed to an empirical standardized-residual correlation matrix.
This therefore differs from the usual two-stage empirical-target estimator in
[Engle and Sheppard (2001)](https://pages.stern.nyu.edu/~rengle/Dcc-Sheppard.pdf).
Neither two-stage standard errors nor uncertainty that omits estimated marginal
variance parameters is reported. The target parametrizes this conditional model;
its normalized diagonal does not assert that it equals the unconditional
correlation of its standardized innovations.

Full BEKK uses the orientation

`H_t = C C' + A' e_t-1 e_t-1' A + B' H_t-1 B`.

C is lower triangular with positive diagonal. A and B are full matrices, with
signed off-diagonal entries. The stationarity guard is the spectral radius
`rho(A tensor A + B tensor B) < 1-1e-6`, equivalent to that of the transposed
covariance operator. A homogeneous radial map enforces this full domain; the
stronger sufficient spectral-norm contraction is recorded only as a diagnostic.
This is the BEKK recursion and orientation described in
[Fülle, Lange, Hafner, and Herwartz (2024)](https://www.jstatsoft.org/article/view/v111i04).
Sign-equivalent A/B representations and local maxima remain possible.

## Initialization, sample, and computation

The initializer is fixed and conditioned upon throughout estimation and
information calculation. By default it is the equal-weight covariance of the
first `min(75,N)` residual vectors, formed using the initial sample mean or zero.
An explicitly supplied `initial_covariance` must have the correct dimension and
be finite, symmetric, and strictly positive definite. BEKK uses it as the first
H. CCC/DCC use its diagonal as the first marginal variances and the estimated
target as the first correlation/Q state. The exact initializer and convention
are persisted; uncertainty of a data-derived backcast is not added to the
conditional-information covariance.

The sample must have at least `max(30,3*p)` complete periods, where p is the full
parameter count. An explicit time column must contain consecutive integer
periods. Dates must first be converted to a meaningful period index; the API
rejects datetime ranking that could compress actual gaps. Without a time column,
input row order defines consecutive periods. `missing="drop"` may trim a
contiguous sample but cannot remove an interior period. Sorting, original sample
positions, missing-row counts, and sample hashes use the common ModelFrame
contract.

The execution domain is eager CPU float64, with `max_n=512` by default and an
explicit upper bound of 2048. Workspace plans cover matrix paths, the sequential
derivative graph, and full information before numerical buffers are built.
Constant or nearly constant series and scale ratios over 10000 are rejected;
explicit rescaling is required. There is no Dataset or GPU route for this family,
and no unmeasured large-data scaling promise. The existing Dataset routes remain
separate.

Native BFGS operates in constrained parameter coordinates. The score and exact
observed Hessian differentiate the complete matrix recursion with PyTorch
autodiff. Invalid line-search trials are rejected. Convergence requires an
interior stationary solution and nonsingular positive information, including a
reporting-unit score check. Iteration, gradient, scaled-gradient, information,
and constraint diagnostics are recorded. A failed convergence or information
check raises an error; it does not return a successful fit. A global maximum is
not guaranteed. Reported tests and confidence intervals are Gaussian-asymptotic
normal inference from the full joint information, not finite-sample guarantees.

## Forecast meaning and persistence

For all models the next full covariance is known exactly conditional on the
fitted parameters, last residual, and last covariance state. BEKK also has an
exact multi-step full-matrix expectation recursion, because replacing future
shock outer products by their conditional covariance closes its linear
recursion.

For CCC and DCC, marginal multi-step variance expectations are exact under the
recursion. CCC later full covariance outputs use
`R_ij * sqrt(E[h_i] * E[h_j])`. Their off-diagonal entries are a plug-in
approximation: generally `E[sqrt(h_i*h_j)] != sqrt(E[h_i]*E[h_j])`. The test suite
demonstrates this gap with independent Gaussian quadrature. DCC later Q values
additionally use the explicitly approximate section 7 Q-forward recursion in
[Engle and Sheppard (2001)](https://pages.stern.nyu.edu/~rengle/Dcc-Sheppard.pdf),
`Qnext = (1-a-b) Qbar + (a+b) Q`. No exact multi-step nonlinear DCC expectation
is claimed.

Forecast attributes list exact full-covariance horizons, approximate horizons,
exact marginal-variance horizons, and the forecast method. Output correlations
normalize the returned covariance matrices; they do not estimate the expectation
of the future random conditional correlation matrix. These forecasts condition
on fitted parameters and include no parameter-uncertainty intervals. Forecast
steps are bounded to 1..1000 and have a workspace guard.
Publication notes preserve the forecast method and parameter-conditioning
disclosure in direct LaTeX and serialized console exports.

ResultBundle stores the full coefficient covariance, every conditional H/R,
model and initializer identity, physical parameter matrices, and the terminal
state. Strict JSON round trips retain enough state to reproduce forecasts;
parameter-matrix discrepancies and invalid or inconsistent forecast state are
rejected. Common publication output supplies coefficient, uncertainty, fit,
provenance, and forecast tables in LaTeX. The chart residuals represent the first
outcome series; full covariance/correlation paths are in the persisted extras.

## Independent scientific evidence

Run `scripts/validate_mgarch_market131.py` with the repository source on
PYTHONPATH. The receipt (internal evidence excluded from this public snapshot) records local
test output, complete Gaussian density checks, finite-difference scores and full
observed-information covariance checks, real joint fits, constraint diagnostics,
JSON replay, timing on the actual host, and LaTeX exports. The NumPy oracle uses
physical-unit matrix equations, separate from the production parameter map and
optimizer. Tests include 2D CCC/DCC/full BEKK, all off-diagonal BEKK entries, and
3D CCC/DCC with jointly estimated constant means.

The published reference fixture (internal evidence excluded from this public snapshot)
was generated in actual R 4.6.0 through development-only WebR 0.6.0. The
[reproduction script](../../scripts/validate_mgarch_webr.mjs) downloads and verifies
the exact BEKKs CRAN-mirror commit
[01b20c676192a99b16785ed5a02b1fd55038436c](https://github.com/cran/BEKKs/tree/01b20c676192a99b16785ed5a02b1fd55038436c).
It executes the literal source-extracted published `predict.bekk` initial
forecast loop, including its symmetric eigenvalue matrix square root. The
paper/C++ covariance recursion is independently evaluated in R with the recorded
C-orientation conversion. CCC/DCC paper formulas are independently evaluated in
R. Every full covariance and correlation path, likelihood, and five-step
forecast agrees with native Torch at the tolerances in the tests. Source hashes,
the executed literal block, R setup hash, numeric matrices, and exact versus
approximate forecast conventions are retained.

These are supplied deterministic stationary parameter/return experiments, not
published empirical parameter estimates. The complete BEKKs package or its
compiled Rcpp estimator was not executed, and external coefficient/Hessian/OPG
parity is not established. There is no installed-app, saved-project UI, frozen
desktop, deployment, or public-release validation in this receipt. WebR and
NumPy are reference-development tools, never estimator runtime dependencies.
