# Interval survival and competing risks

These eight stages implement bounded, iid, single-event procedures in native
Torch CPU float64. They preserve the existing `stcox`, `streg`, `sts`, `ltable`
and saved-survival workflows. They are not a claim of complete survival-command
or licensed vendor parity.

| Procedure | Estimand and supported model |
| --- | --- |
| `stinterval_exponential` | intercept-only exponential survival distribution, rate parameter |
| `stinterval_weibull` | intercept-only Weibull survival distribution, scale and shape |
| `stinterval_lognormal` | intercept-only lognormal survival distribution, log-time location and standard deviation |
| `stinterval_loglogistic` | intercept-only loglogistic survival distribution, scale and shape |
| `turnbull` | interval-censored nonparametric likelihood on maximal-intersection support cells |
| `cumulative_incidence` | Aalen–Johansen probability of each mutually exclusive cause by a requested time |
| `cause_specific_hazard` | cause-specific Nelson–Aalen cumulative hazard, rather than an instantaneous hazard curve |
| `cif_compare` | independent two-group CIF differences at prespecified times for one requested cause |

## Endpoints and sample

`lower` and `upper` are complete positional vectors. Finite interval records
mean `(lower,upper]`; `lower=0` means left censoring. Equal positive endpoints
mean an exact observation: its density enters the continuous parametric
likelihood, while its atom membership enters Turnbull's likelihood. `upper=None`
or positive infinity means right censoring at `lower`; saved JSON canonicalizes
it to `None`. `(0,infinity)` is refused because it contributes no information.
Missing, inverted, negative and zero exact times are refused; there is no silent
row deletion, replacement with a midpoint, or conversion to right censoring.

Competing-risk vectors `time,event` use positive follow-up times and integer
codes `0=censor`, positive integers for mutually exclusive causes. At a tied
time, failures use the common risk set including subjects censored at that time.
Results retain the original input order and the complete risk-set table.
Requested curve times are unique, increasing and explicit; inference is confined
to observed follow-up (to common follow-up for two groups).

All methods assume independent iid subjects and independent noninformative
censoring. Interval ML additionally assumes the selected parametric distribution
and noninformative observation intervals. There are at most 4,096 subjects,
256 complete curve/covariance components, and 256 Turnbull support cells.
Positive endpoints lie in `[1e-12,1e12]`; zero is allowed for a left endpoint or
prediction time. The configured workspace limit is checked before dense work.
There is no output thinning. CUDA/MPS, Dataset collection, covariates, delayed
entry, recurrent events, frailty, clusters and weights are outside this contract.

## Likelihood and uncertainty

Parametric likelihood contributions are `f(t)` for exact observations,
`F(upper)` for left censoring, `F(upper)-F(lower)` for interval censoring, and
`S(lower)` for right censoring. Exact contributions include the log-time
Jacobian, so the reported likelihood is on the original time scale. Stable
log-CDF/log-survival differences retain information in tails. For very narrow
normal/logistic standardized intervals, differentiable four/eight-node
transformed-density integration must agree within the declared scaled
`2e-12` log-probability gate; failed checks produce a numerical refusal.
An interior
finite optimum, score convergence and positive definite observed information
are required; no ridge or arbitrary boundary estimate supplies missing
identification. Explicit parameter and iteration limits bound computation.
Log rate/scale or log-time location lie strictly inside `(-40,40)`. Log shape
and log-time standard deviation lie inside `(-log(1000),log(1000))`; `maxiter`
is 1–2,000 per start. At least two observations are required for exponential
ML and three for the other distributions. Multiple starts select the best accepted stationary
likelihood; they do not certify a global optimum for every censoring design.
The full inverse observed information and its complete natural-parameter
Jacobian transformation are saved. Asymptotic normal parameter inference and
delta-method curve uncertainty are conditional on the chosen distribution;
pointwise intervals are not simultaneous bands. No finite reference degrees
of freedom or parameter p-value is invented when no parameter null is supplied.
Survival limits differentiate stable log cumulative hazard directly, preserving
representable uncertainty even when displayed probabilities round to zero or
one. `interval_survival_predict`
uses validated complete saved state and covariance without refitting.

Turnbull maximizes `sum(log(A @ mass))` on the probability simplex. Support
cells retain interval locations, exact atoms and an open right tail; a tail
interval is not a cure fraction at infinity. Likelihood monotonicity, simplex
constraints and a KKT/duality-gap gate are checked. Safeguarded monotone
simplex-tangent Newton acceleration is available for at most 64 support cells.
`maxiter` is 1–10,000 and the worst declared iteration work is at most
500 million estimated units, including the initial rank SVD and repeated
information crossproduct; budget
refusals occur before dense allocation. A conservative full-column
rank certificate is required for unique support masses; ambiguous mass fits
are refused. Within-cell event locations remain unidentified, and requested
CDF/survival lower and upper bounds report that ambiguity. These are
identification bounds, not sampling confidence intervals. Sampling covariance,
standard errors, p-values and confidence bands are unavailable.

At each event time, Aalen–Johansen adds `S(previous)*d(cause)/Y` to each CIF and
updates survival by `1-sum(d(cause))/Y`. Cause-specific Nelson–Aalen adds
`d(cause)/Y`. Recursive derivatives with respect to subject case weights give
the complete infinitesimal-jackknife covariance `D' D`, including cross-time
and cross-cause terms, with no finite-sample correction. No-censoring limits
recover multinomial covariance; the one-cause limit recovers Greenwood.
Competing events leave subsequent risk sets. Reported CIF intervals project
pointwise normal intervals onto `[0,1]`; zero-variance inference is explicitly
unavailable. Hazard inference refers to cumulative hazards.

Two-group contrasts report group 2 minus group 1 in deterministic label order
and add the independent groups' full covariance matrices.
Normal tests address a zero difference at each prespecified time; a joint Wald
test is reported only when the selected contrast covariance has full rank.
Singularity is reported rather than hidden behind a pseudo-inverse. These are
fixed-time contrasts; Gray's global equality test and Fine–Gray subdistribution
hazard regression remain open under MARKET-163.

## Persistence and evidence

Each `TableSet` retains complete input, censor/cause/group coding, model or
support state, covariance, inference assumptions and all result tables in its
JSON settings and attributes. Its LaTeX export includes all tables. The
[synthetic eight-method example](../examples/survival_extensions.py) saves
complete JSON files, replays inputs and checks saved prediction state. Source
tests, independent numerical oracles, a frozen runtime, and a dedicated native
QA application are distinct evidence layers. The QA app does not constitute a
public release, signing/notarization or macOS 15 compatibility validation.

## Primary sources

Endpoint and distribution conventions follow the survival package authors'
[Surv documentation](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html)
and [survreg distributions](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.distributions.html).
The nonparametric likelihood follows
[Turnbull (1976)](https://doi.org/10.1111/j.2517-6161.1976.tb01597.x)
and the optimality/uniqueness requirements follow
[Gentleman and Geyer (1994)](https://doi.org/10.1093/biomet/81.3.618).
The product-limit estimator follows
[Aalen and Johansen (1978)](https://www.math.ku.dk/bibliotek/arkivet/preprints-fra-ims/1977/preprint_1977_-_no_6_aalen__odd__johansen__s_ren_-_an_empirical_transition_matrix_for_non-homogeneous---.pdf).
Case-weight covariance follows the survival authors'
[competing-risk vignette](https://stat.ethz.ch/R-manual/R-devel/RHOME/library/survival/doc/compete.pdf)
and [Parner, Andersen and Overgaard (2023)](https://doi.org/10.1007/s10985-023-09597-5).
Runtime estimation does not import R, SciPy, statsmodels or other third-party
estimation code; development oracles are separate.
