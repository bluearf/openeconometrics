# Time-series selection, refits, MIDAS, general state space and ETS

These public Python routes use native CPU float64 Torch computations. They accept
resident tables within explicit model/workspace/operation budgets. Registration
does not imply a Dataset fitting route, GPU execution, automatic collection, or
unrestricted Stata parity. Existing ARIMA/UCM/smoothing forecasts remain separate.

## ARIMA selection and refits

`oe.auto_arima(data=df, y="y", time="t", d=1)` returns the selected ordinary ARIMA
ResultBundle, preserving the existing `oe.forecast` contract. It exhaustively
searches the explicitly bounded p/q/P/Q domain using exact Gaussian likelihood.
AIC, AICc or BIC includes every estimated parameter (including sigma). All
candidates use the same differencing and response positions. AICc is undefined
when N<=K+1 and that candidate fails explicitly. Failures, nobs, parameter counts,
criteria and selected order are retained. A completely failed search raises an
error with its complete `candidates` record. This is a bounded exhaustive search,
not a claim to the optimum over arbitrary orders or a stepwise approximation.

If d is omitted, repeated level-KPSS at 5% determines it before likelihood
comparisons (at most max_d=2); automatic d with regressors is rejected. Seasonal
differencing is explicit. No information criterion is compared across different
differences. Inference and intervals condition on the selected order; selection
uncertainty is excluded. Input/sample hashes and original physical positions
retain the source sample, including explicitly excluded endpoint observations.

`oe.rolling(spec, data=df, window=60, step=12)` and
`oe.recursive(spec, data=df, minimum=60, step=12)` refit an inferential ModelSpec.
Each recorded origin includes the exact half-open training window, physical
sample positions, coefficients, full covariance, inference, specification and
fit failures. Future outcome rows never enter estimation. Optional forecasts
reuse family handlers; future regressors must be supplied separately in an
origin-keyed `exog` mapping. MIDAS windows slice/rebind the original dated alignment
metadata. Prediction-only fits require a separate predictive validation workflow;
no artificial coefficients/SEs are generated. Interior missing periods and
irregular calendars are rejected. Datetime strings are parsed before sorting.

## Release-aware MIDAS

```python
aligned = oe.midas_align(low=quarterly, high=monthly,
    low_time="quarter", high_time="month", value="indicator", lags=12,
    frequency="MS", origin="forecast_origin", release_time="published_at")
model = oe.midas(data=aligned, y="growth", x=["control"], covariance="HC1")
means = oe.midas_predict(model, data=aligned)
```

The low table provides explicit forecast origins, distinct from target periods
when needed. Observation AND publication timestamps must be no later than the
origin. Positive regular calendar offsets define exact lag positions: gaps are
raised or explicitly dropped, never compressed. Release times must be monotone
in observation time and at/after observation timestamps; align multiple vintages
separately. Alignment uses binary searches and O(N_low*lags) output work.
The bound hash plus per-row dates prevents silently reusing stale/tampered
alignment metadata. Three to 256 unique numeric lags are supported.

The aggregate uses normalized exponential-Almon weights
`softmax(theta1*j + theta2*j**2)` with j in [0,1]. Amplitude, both shape parameters,
intercept and controls are estimated jointly by native nonlinear least squares.
Rank/nonpositive information/perfect fits/nonconvergence are errors. Nonrobust
covariance uses Gaussian expected information and SSE/(N-K); HC0/HC1 use observed
nonlinear curvature and row-score sandwich, with the recorded N/(N-K) correction
for HC1. Coefficient inference uses t(N-K). Predictions provide conditional means
and asymptotic normal delta-method parameter intervals, excluding new-outcome
noise. `kind="outcome"` additionally includes independent Gaussian residual noise
using the fitted residual scale. These differ from state-space process prediction
intervals and do not model serially correlated or non-Gaussian forecast errors.

## General linear Gaussian state space

```python
result = oe.sspace(data=df, y="y", system={
    "Z": [[1.0]], "T": [[0.6]], "Q": [[0.4]], "H": [[0.2]],
    "initialization": "stationary",
    "parameters": [
        {"name":"phi", "matrix":"T", "row":0, "col":0, "transform":"unit"},
        {"name":"q", "matrix":"Q", "row":0, "col":0, "transform":"positive"}]})
forecast = oe.forecast(result, 12)
```

The system is `y_t=Z*a_t+d+e_t`, `a_(t+1)=T*a_t+c+u_t`, with independent Gaussian
shocks H and Q. Additional measurements use `responses=[...]`. Arbitrary fixed
time-invariant matrices, state/measurement intercept vectors and selected free
matrix/vector cells are supported. Free cells use identity, exp-positive or
tanh(-1,1) transforms; symmetric covariance cells update both triangles and
duplicate aliases are rejected. H is strictly PD; Q and known P0 may be PSD.
Nothing is projected onto the PSD cone. At most16 states/8 measurements/20 free
parameters and explicit derivative-work budgets are admitted.

`known` initialization supplies a finite a0/P0 prior BEFORE observing y1.
`stationary` solves the exact discrete Lyapunov/mean equations and requires
spectral radius(T)<1. Known priors allow radius<=1. Diffuse initialization,
time-varying matrices, endogenous correlated measurement/state shocks and partial
missing measurements are outside this domain. Observable state rank and positive
definite full observed information are required. Gaussian ML uses analytic Torch
autodiff score/Hessian; reporting-unit covariance includes transformation cross
derivatives. Fixed systems contain no fictional coefficient inference.

Native Kalman prediction-error likelihood and Joseph covariance updates retain
all multivariate fitted/filter moments. Forecast attributes contain complete
measurement/state means and covariance matrices. The main table reports the
first measurement with Gaussian process/measurement/state intervals. Estimated
parameter uncertainty is excluded and disclosed. `sspace_filter` evaluates a
fully specified system without fitting parameters.

## Statistical ETS, separate from smoothing

`oe.ets(data=df,y="y",model="AdA",period=4)` fits additive-error ANN/AAN/AdN/ANA/AAA/AdA
innovations models by joint Gaussian ML. Free smoothing parameters, initial states
and sigma enter the full observed-information covariance. `fixed` can fix any
relevant smoothing parameter; `initial` can supply known initial states. The
coupled bounds beta<=alpha and gamma<=1-alpha, error-correction admissibility,
seasonal zero-sum initial identification and parameter/sample budgets are checked.
Multiplicative components are rejected. Interior/nonidentified/boundary failures
are reported rather than manufacturing SEs. Conditional forecasts propagate
future innovations and explicitly exclude estimated-parameter uncertainty.

## Independent evidence and references

`tests/test_tsworkflows_oracle.py`, `test_tsworkflows_review.py` and
`test_tsworkflows_ets_joint.py` compare complete multivariate Gaussian densities,
independent NumPy Kalman equations, closed-form ML information, reporting-unit
finite-difference Hessians, independent nonlinear least squares, full free ETS
parameters, future impulse-response variances, exact calendar/release alignment,
ARIMA likelihood/criteria and adversarial lookahead/window/metadata behavior.
SciPy/statsmodels/NumPy references are development-only, not estimation adapters.

- Hyndman and Khandakar (2008), [automatic forecasting](https://www.jstatsoft.org/article/view/v027i03).
- Hyndman et al., [innovations ETS equations](https://otexts.com/fpp3/ets.html).
- Ghysels, Sinko and Valkanov (2007), [MIDAS](https://doi.org/10.1080/07474930600972467).
- [Stan Gaussian dynamic linear model reference](https://mc-stan.org/docs/functions-reference/gaussian-dynamic-linear-models.html).
- [RBA Kalman likelihood equations](https://www.rba.gov.au/publications/rdp/2012/2012-08/appendix-a.html).

This is method/domain evidence, not an actual Stata execution, a public release,
an installed-desktop test or hardware-scale performance validation.
