# Censored count regression

`oe.cpoisson` estimates a Poisson model when some recorded counts are censored.
`oe.cnbreg` applies the same observed-event likelihood to negative binomial
counts: NB2 by default, or NB1 with `dispersion="constant"`.
Both return persistable `ResultBundle` objects with coefficient intervals,
model tests, likelihood, AIC/BIC, censoring counts and latent-mean predictions.

These models retain the censored observations and their covariates. They do
not divide the likelihood by a selection probability, as a truncated-count
model does. Use `oe.tpoisson` / `oe.tnbreg` when subjects outside the observed
range were absent from the sample entirely.

## Top and bottom coding

```python
import pandas as pd
import torch
import openecon as oe

torch.manual_seed(134)
x = torch.randn(800, dtype=torch.float64)
latent = torch.poisson(torch.exp(0.8 + 0.45 * x))
df = pd.DataFrame({"x": x.tolist(), "visits": latent.clamp(1, 5).tolist()})

model = oe.cpoisson(data=df, y="visits", x=["x"], ll=1, ul=5)
print(model.summary())
```

Without `censoring=`, a recorded `y <= ll` means `Y <= ll`, a recorded
`y >= ul` means `Y >= ul`, and the intervening counts are exact. Equality
belongs to the censored event. Either limit may be omitted. `ll` and `ul`
can each be an integer or a column name containing observation-specific
integer limits. In a limit column, a missing value means that this side
does not censor that row; an otherwise complete observation is retained.
Where both limits exist, auto mode requires `ll < ul`.

This is the classification in the primary
[Stata cpoisson manual](https://www.stata.com/manuals/rcpoisson.pdf).
Without limits or a status column, `cpoisson` reduces to ordinary Poisson ML,
and `cnbreg` reduces to the corresponding ordinary negative binomial ML.

## Explicit observed events and closed intervals

When a count was recorded only as a range, give `censoring="status"` and
the endpoint columns. Status values are exactly these strings:

| Status | Required limit | Observed event |
| --- | --- | --- |
| `exact` | None | `Y = y` |
| `left` | `ll` | `0 <= Y <= ll` |
| `right` | `ul` | `Y >= ul` |
| `interval` | `ll`, `ul` | `ll <= Y <= ul` |

```python
df["lower"] = ((latent // 3) * 3).tolist()
df["upper"] = (latent // 3 * 3 + 2).tolist()
df["visits"] = df["lower"]
df["status"] = "interval"
interval_model = oe.cpoisson(
    data=df, y="visits", x=["x"], ll="lower", ul="upper", censoring="status"
)
```

An explicit interval is a genuinely coarsened observation between two known
endpoints. It differs from ordinary two-sided top/bottom coding, where the
recorded endpoint identifies which tail occurred. Closed interval endpoints
may coincide, reducing its probability to a point mass. The recorded `y`
remains a nonnegative integer proxy and must lie in its declared event;
for interval rows, using its lower endpoint is convenient. It does not replace
the event probability in the likelihood. Missing an applicable endpoint
raises `missing_censoring_limit`; unused missing endpoints are allowed.

## Negative binomial counts

```python
torch.manual_seed(135)
mu = torch.exp(0.8 + 0.45 * x)
alpha = 0.9
nb_latent = torch.distributions.NegativeBinomial(
    total_count=1 / alpha, logits=torch.log(alpha * mu)
).sample()
nb_df = pd.DataFrame({"x": x.tolist(), "visits": nb_latent.clamp(1, 5).tolist()})
nb_model = oe.cnbreg(data=nb_df, y="visits", x=["x"], ll=1, ul=5)
```

NB2 has `mu = exp(x'b + offset)` and `Var(Y|X) = mu + alpha * mu^2`.
The ancillary coefficient is `/lnalpha`. NB1 has
`Var(Y|X) = mu * (1 + delta)` and ancillary coefficient `/lndelta`.
`metrics` records `alpha` or `delta`; `extra["alpha"]` records its estimate,
delta-method standard error and interval obtained by exponentiating the log
parameter's endpoints. `cnbreg` is an OpenEconometrics extension; it is not advertised
as a built-in Stata command named `cnbreg`.

## Shared options and reported results

Both functions accept the standard count-family arguments:
`categorical`, `intercept`, `missing`, `alpha`, `offset` or `exposure`,
`weights` with `weight_type`, and `covariance` with optional `cluster`.
An offset enters with coefficient one; positive exposure enters as its log.
Offset and exposure are mutually exclusive. Missing model inputs follow
`missing="raise"` (default) or an explicit `missing="drop"`.

Supported covariances are `nonrobust` (inverse observed information),
`opg`, `robust` and `cluster`, including two cluster columns through the
shared covariance engine. Frequency weights reproduce repeated observations;
analytic weights normalize to the row count; importance and probability
weights enter the likelihood as given. Probability weights require robust
or cluster inference and default to robust. These family conventions include
extensions beyond Stata cpoisson's weight and cluster interface.

Results use normal z inference. Under `nonrobust`, the model test is an LR
comparison with a constant-only model when that model converges; otherwise
it is a Wald test of the count-equation slopes. The NB fit additionally reports
the dispersion-zero LR comparison with censored Poisson under likelihood-based
covariances. Its reference is `chibar2(01)`; the chi-square-one tail is halved.

`extra["censoring_counts"]` contains `exact`, `left`, `right` and `interval`
counts, using frequencies when applicable. Chart predictions are the latent
unconditional mean `E[Y|X]`; they are not predicted top-coded records, and
their recorded-proxy residuals are not latent residual estimates. Results
support JSON save/reload and publication LaTeX export. Generic `oe.predict`
and `oe.margins` support the **latent unconditional mean**, including results
loaded from saved JSON; see [prediction and margins](prediction.md).
Censor-event probabilities, conditional means and predicted recorded
top/bottom-coded counts are not supported by that post-estimation interface.

## Likelihood and numerical work

For an exact event the likelihood is `f(y)`. Left and right events use
`F(ll)` and `1 - F(ul-1)`. A closed interval uses
`F(ul) - F(ll-1)`. These probabilities and their first and second index
derivatives are evaluated on float64 Torch. The shared block engine assembles
the coefficient score/Hessian and fits Newton steps on centered designs.
NB fits compare three interior starts; covariance and inference are then
mapped back to the original coefficient coordinates.

Poisson point masses use the deviance and a separate Stirling remainder,
avoiding cancellation between large count and log-factorial terms. Poisson
tails use regularized incomplete gamma functions with log-domain
series/continued-fraction fallbacks where ordinary probabilities underflow.
NB tails use the regularized incomplete beta function, evaluated by a
safeguarded continued fraction with analytic second-order forward derivatives.
Short finite intervals use exact log-sum-exp combinations of their point
masses. Large intervals select a stable tail difference. No count tail is
silently cut at an arbitrary maximum count and no rare tail is replaced by zero.
Special-function iterations have a convergence budget; exhaustion raises
`tail_nonconvergence` rather than returning an unconverged approximation.

Limits and observed counts must be integers from zero through `2^53`.
This is an **integer representation limit**, not a guarantee that every
special function is accurate up to that count. Exact Poisson point masses
and finite intervals of width at most 64 use the stable mass evaluator.
Events requiring cumulative Poisson cutoffs `k` use a separately validated
region `k <= 10^12` (right events require `k=ul-1`; left events require `k=ll`).
Larger cumulative cutoffs raise `precision_unsupported` before fitting.
Near-mean development checks at the upper cutoff cover roughly ten standard
deviations in either direction and agree with independent gamma evaluations
within `4e-11` absolute probability / `1e-9` absolute log probability.
These are measured validation tolerances, not universal error guarantees.

NB point masses and tails currently require **shape + event endpoint + 1
<= 10^7**; the shape is `1/alpha` for NB2 and `mu/delta` for NB1. This is a
conservative count/shape precision region for the gamma algebra, **not a
limit on the number of observations**. For example, NB1 `mu=10^5, delta=.001`
has shape `10^8` and now raises `precision_unsupported`; the previously
checked rare-tail value alone did not certify all masses or derivatives at
that scale. Outside-domain starts and final parameters retain that specific
error. Optimizer trials beyond the region are rejected without evaluating a
rounded likelihood; if precision prevents a certified maximum, fitting
reports the same error rather than disguising it as nonconvergence.
NB trial dispersions below `1e-9` also lie outside the evaluator's domain.
An independently detected zero-dispersion boundary is reported as
`boundary_solution` with advice to fit `cpoisson`.

All-zero or one-sided events with a feasible common intercept shift can make
the likelihood monotone; separated design directions raise
`separation_detected`. Removing the intercept changes the recession geometry:
balanced positive/negative regressors can have a finite solution even when
all events are right-censored or all exact counts are zero. Events covering
the entire count support provide no information and are rejected if they
comprise the whole sample. Invalid statuses, inconsistent proxies, reversed
limits, failed iterations and singular information are explicit errors.

## Verification and limits

Development tests compare fits with independent SciPy optimization,
point-mass/tail likelihoods, numerical gradients/Hessians, weighted covariance
calculations and replicated frequency data. High-precision mpmath checks cover
log probabilities below `-4900`, where ordinary floating probabilities are zero.
Large Poisson point checks extend through `2^53`; near-mean checks use the
actual float64 exponential, and differences from an exact exponential of the
saved index are bounded by its natural native-mean rounding.
Other checks cover NB1/NB2, uncensored reductions, mixed interval events,
per-row limits, offsets/exposure, persistence, LaTeX, rank/identification,
separation, dispersion boundaries and actual stopped Newton iterations.
SciPy, NumPy and mpmath are oracle tools in these tests, not dependencies of
the new likelihood kernels.

This batch uses in-memory CPU model tables. It does not add streaming/GPU
count estimation, parameter constraints, survey design inference or an actual
Stata execution comparison. `extra["stata_parity_validated"]` remains false.
