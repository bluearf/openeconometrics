# Fractional memory: finite-filter CSS and GPH

`fracdiff`, `gph`, `arfima` and `forecast` are four separate native CPU float64
deliveries (MARKET-203/204/205/206 under MARKET-160). None changes integer ARIMA.
All arithmetic uses Torch; NumPy/SciPy appear only in independent development
oracles. These domains are implemented, not licensed Stata/EViews comparisons.

## Operators and initial values

Hosking's generalized binomial operator has `w[0]=1` and
`w[j]=w[j-1]*(j-1-d)/j`. `oe.fracdiff(series, d, terms=256)` applies its
**finite** prefix using zero prehistory and a zero-padded *linear* FFT
convolution. `initial='drop'` discards the first `terms-1` partially initialized
outputs; each retained original position is explicit. The transformation accepts
`-1<=d<=1`, so zero is the identity and one is first differencing.

An optional positive `tolerance` chooses the first omitted coefficient smaller
than that tolerance, within the supplied terms budget. It is **not** a bound on
the infinite tail, transformed-series error or estimation bias. An unmet
tolerance raises; no truncated sample or changed approximation is hidden.

## Diagnostic

`oe.gph(series, bandwidth=m)` regresses the log periodogram at frequencies
`2*pi*j/N, j=1..m` on a constant and `-log(4*sin(frequency/2)^2)`; the slope is d.
The main `estimate` table uses the full covariance
`(pi^2/6)*(X'X)^(-1)` and asymptotic Gaussian z/p/CI, under the GPH low-frequency
conditions `m -> infinity`, `m/N -> 0`. Its intercept is a spectral nuisance
parameter. The separately named `regression` table uses the empirical OLS
residual variance and `m-2` Student-t degrees of freedom as a diagnostic, not
as the main long-memory inference. `periodogram` retains the actual frequencies
and ordinates. An estimated d outside the stationary interval is flagged;
GPH neither imposes that interval nor proves absence of short-memory bias.

## Conditional fit

```python
result = oe.arfima(data, 'y', ar=1, ma=1, terms=128, burn=8, time='t')
restored = oe.ResultBundle.model_validate_json(result.model_dump_json())
future = oe.forecast(restored, 12)
```

The declared model is
`A(L) W_terms(L) (y_t - mean) = B(L) e_t`, with
`A=1-sum(phi_j L^j)`, `B=1+sum(theta_j L^j)` and iid Gaussian innovations.
The supported objective is the approximate **conditional Gaussian CSS**
likelihood of this finite filter. Prehistory of centered observations and
innovations is zero. The explicit `burn` observations initialize recursions but
are excluded from likelihood and reported sample. Filter length and likelihood
sample stay fixed while optimizing; `terms` cannot exceed N.

AR and MA orders independently range from zero to three. Reflection-coefficient
maps enforce stationary AR and invertible MA polynomials. Estimated d is
strictly within `(-.49,.49)`; supplying `d` fixes it in `[-.49,.49]`, removes it
from the inference vector and persists that constraint. The output mean is in
the original series units, not a drift. With `constant=False` it is fixed to
zero. Gaussian scale sigma is estimated jointly. Two explicit starts for d
(zero and .25) are screened; all failed convergence, inadmissible curvature and
near-boundary solutions raise rather than returning apparently valid inference.

Torch differentiates the FFT filter, formal causal polynomial reciprocal and
full likelihood. The full inverse observed information is transformed from
optimization coordinates to reported mean/d/AR/MA/sigma coordinates, including
all cross covariances. Coefficients retain SE, Gaussian z, p and CI. Residuals
are conditional one-step innovations. Model-specific fixed constraints,
initialization, approximation, convergence, original/fitted sample positions
and source/resource provenance remain in the structured result.

This does not implement Stata's exact stationary ML/MPL or EViews' exact
Toeplitz likelihood. Finite-filter CSS and the documented zero initialization
are distinct statistical choices; no vendor numerical equivalence is claimed.

## Restored forecasts

The persisted state contains the mean, d, AR/MA, sigma, complete finite filter,
its required centered history, final innovations, sample/initialization and
parameter mapping. A canonical SHA256 plus consistency checks reject damaged
or incompatible states. Forecasts apply the *same fitted finite model*, not a
new infinite fractional integration approximation. Future innovation means are
zero. The impulse response of `B/(A W_terms)` gives a triangular map from future
innovations to future outcomes; `sigma^2 * map * map'` is the full cross-horizon
covariance. SE/normal intervals use its diagonal. This conditions on parameters
and saved initial state; parameter uncertainty and unconditional prehistory
uncertainty are explicitly excluded. JSON restoration needs no refit or original
data. Generic `predict/margins` adapters are not claimed.

## Input and resource contract

All four APIs require a complete finite real resident series; missing rows,
booleans, complex data and implicit device copies are rejected. Table helpers
accept one named numeric column. Optional time is a distinct, consecutive
integer calendar in supplied order; gaps/duplicates/reversed order and datetime
calendars are rejected. Without time the caller declares row order regular.
Weights, panels, categories and Dataset collection are unsupported.

| Operation | Bound / declared workspace |
|---|---|
| fracdiff/GPH | at most 1,000,000 rows; filter length 2..16,384 |
| GPH | N>=64; bandwidth 8..min(N/4,100,000), default floor(sqrt(N)) |
| ARFIMA | at most 32,768 rows; max(32,4K) likelihood rows; N*max_iterations*K<=60,000,000 |
| Forecast | 1..256 steps; explicitly bounded H by H covariance |
| FFT workspaces | padded FFT length *96 bytes for procedures; *768 bytes for fitting, checked against OPENECON_WORKSPACE_MB before tensor work |

FFT convolution and Newton-doubling formal reciprocals never allocate an
observation by observation matrix. Caller input, Python result objects and
allocator/RSS overhead are separate from these live tensor estimates. CPU is
explicit; CUDA/MPS, regressors, seasons, exact ML/MPL and robust/cluster
covariance raise or have no declared route. Alpha is explicit throughout.

## Validation and references

`tests/test_fractional_memory.py` checks generalized-binomial/integer limits,
direct convolution/recursive reciprocal, independent NumPy periodogram and
complete covariance/p/CI, analytic iid Gaussian estimates/information,
independent conditional likelihood/gradients and SciPy joint optimization with
full finite-difference Hessian. d=0 reproduces the existing conditional ARMA
likelihood, coefficients and forecasts. A 120,001-row fixture audits selected
direct convolution values without dense N by N storage. Missing/calendar,
budget, state-integrity and unsupported-option cases fail explicitly.
`docs/examples/fractional_memory.py` is the runnable synthetic example.
Frozen/installed native and stop/restart receipts are recorded separately under
`docs/evidence/fractional-memory-2026-10-07`; source tests alone do not certify
an installed app or public release. CUDA/Windows/full vendor parity stay false.

Primary references: [Hosking (1981), Fractional differencing](https://doi.org/10.1093/biomet/68.1.165)
defines the generalized operator and fractional ARMA family;
[Geweke and Porter-Hudak (1983)](https://doi.org/10.1111/j.1467-9892.1983.tb00371.x)
defines the low-frequency log-periodogram estimator.
[EViews' estimation method details](https://eviews.com/help/content/timeser-Estimation_Method_Details.html)
distinguishes exact Gaussian and conditional initialization/objectives;
[Stata's ARFIMA manual](https://www.stata.com/manuals/tsarfima.pdf)
is a comparison target for exact ML/MPL, not an executable oracle used here.
