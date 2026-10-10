# NARDL: asymmetric distributed lags

`oe.nardl` estimates a nonlinear ARDL with positive and
negative partial sums, using OpenEconometrics's float64 Torch arithmetic and the shared
ARDL QR estimator. It is unrestricted by default; selected predictors can
instead have long-run or lagwise short-run symmetry imposed separately.
There is no statsmodels estimation dependency; statsmodels
and NumPy appear only in the independent development tests.

```python
import openecon as oe

fit = oe.nardl(
    data=df, y="consumption", x=["income", "rate"],
    asymmetric=["income"], time="period",
    lags=[2, 1, 0], covariance="HC3", ec=True,
)
display(fit)
display(oe.nardl_multipliers(fit, steps=40))
fit.to_latex()
fit.tests["symmetry_long_run"]
fit.tests["symmetry_short_run"]
```

Omitting `asymmetric` splits every predictor; giving a nonempty subset leaves
the other predictors symmetric. All supplied series must be regularly spaced.
The data are sorted by `time`, or the input row order is used.
Datetime time additionally requires an explicit positive fixed interval,
e.g. `time_delta="1D"`; actual gaps are rejected instead of ranking away dates.
For calendar months/quarters, use consecutive integer period codes. Original row
positions, input-column fingerprints, the original specification and missing
row counts are retained. Missing data at an edge may be explicitly dropped;
interior gaps cannot silently become adjacent periods.

## Model, sample and lag orders

For each asymmetric x, starting at the first complete observation:

    x_positive[t] = sum_(s=1..t) max(x[s] - x[s-1], 0)
    x_negative[t] = sum_(s=1..t) min(x[s] - x[s-1], 0)

Both partial sums start at zero, and `x[t] = x[0] + x_positive[t] +
x_negative[t]`. The initial level is absorbed by the constant. `trend` is
`"constant"` or `"trend"`; a model with no deterministic term is rejected.
Both positive and negative changes are needed to identify separate effects.
The names `<x>_positive` and `<x>_negative` must not collide with model inputs.

    y[t] = c [+ d*t] + sum_i phi[i]*y[t-i]
           + sum_j sum_l b_positive[j,l]*x_positive[j,t-l]
           + sum_j sum_l b_negative[j,l]*x_negative[j,t-l]
           + symmetric distributed lags + contemporaneous exog + e[t]

The model needs at least one outcome lag. `lags=[p, q1, ..., qk]` assigns a
lag order to each **original** predictor, using the same q for its positive
and negative sums. Alternatively, give an outcome lag followed by one lag
per **expanded** predictor, in positive/negative order. For example:

```python
# Positive income gets two lags; negative income gets one. Rate remains symmetric.
fit = oe.nardl(data=df, y="consumption", x=["income", "rate"],
               asymmetric=["income"], lags=[2, 2, 1, 0])
# Exhaustive BIC selection, with at most two outcome lags and one x lag.
fit = oe.nardl(data=df, y="consumption", x=["income"], maxlags=[2, 1], ic="bic")
```

`maxlags` accepts the same original/expanded order conventions, or one shared
maximum. The search uses the existing QR reduction and a common hold-back
sample. It does not rebuild a full observation design for every candidate.
`exog=[...]` adds contemporaneous controls outside the long-run relation.

`ec=True` (default) reports the same x_t error-correction parameterization as
`oe.ardl`: adjustment, positive/negative long-run coefficients, short-run terms
and their full delta-method covariance. `ec=False` reports levels coefficients.
`extra["levels_coefficients"]` and `extra["levels_covariance"]` retain the
underlying levels representation for symmetry tests and multipliers.

## Covariance and symmetry

Coefficient covariance is conventional OLS, HC1 (`"robust"` is an alias), HC2
or HC3. Student t and Wald F use the estimation sample's N − K_free degrees of
freedom. The bounds statistics use classical covariance, as in ARDL.

- Long-run symmetry is `sum_l b_positive[j,l] = sum_l b_negative[j,l]`.
  The two long-run coefficients share the adjustment denominator, so this
  is an exact linear restriction on the levels coefficients.
- Cumulative short-run symmetry compares the sum of coefficients on
  differences in the **conditional EC form with x_(t-1)**: gamma_0 = b_0;
  gamma_l = −sum_(m>l) b_m for l=1..q−1. This representation is algebraically
  equivalent to the reported x_t EC model. The convention is recorded in
  `extra["short_run_symmetry_definition"]`.
- No short-run test is reported when both distributed lag orders are zero.
- `symmetry_long_run:<x>` and `symmetry_short_run:<x>` give each predictor's
  restriction; `symmetry_long_run`, `symmetry_short_run`, `symmetry_joint`
  test the applicable restrictions jointly across asymmetric predictors.
- `symmetry_short_run_lagwise:<x>` / `symmetry_short_run_lagwise` test equality
  at **every** conditional-EC gamma lag, padding absent lags with zero.
  `symmetry_joint_lagwise` adds long-run symmetry. These restrictions are
  stronger than equality of the two cumulative sums.

## Imposing symmetry

```python
# Different horizon restrictions for different asymmetric predictors.
fit = oe.nardl(
    data=df, y="consumption", x=["income", "rate"],
    asymmetric=["income", "rate"], lags=[2, 2, 1],
    long_run_symmetric=["income"], short_run_symmetric=["rate"],
    covariance="HC3",
)
```

`long_run_symmetric` and `short_run_symmetric` each accept a unique subset of
the **asymmetric** original predictor names. Missing lists or empty lists
leave that horizon unrestricted. Variables outside `asymmetric` remain
ordinary symmetric ARDL regressors. Unknown, duplicated or incorrectly typed
requests fail explicitly. The `restricted` option retains its previous,
independent meaning: restricting the deterministic constant/trend to the
long-run relation.

Long-run symmetry imposes equality of the sums of the positive and negative
levels coefficients. Short-run symmetry imposes `gamma_positive[l] =
gamma_negative[l]` for each applicable lag in the conditional EC form with
`x_(t-1)`, **not merely equality of cumulative sums**. The same convention as
the symmetry tests applies: gamma_0 = b_0 and gamma_l = −sum_(m>l) b_m for l ≥ 1.
Absent lag coefficients are zero. For unequal orders, gamma_0 = b_0 also
applies to the zero-order side; its contemporaneous change effect is tied to
its level coefficient. If both orders are zero, there is no independently
adjustable short-run component: a requested short-run restriction is recorded
in `extra["imposed_symmetry"]["vacuous_short_run"]`, with no added restriction
or invented test. This follows the EViews zero-lag convention.

Only long-run symmetry permits transient asymmetry. Only short-run symmetry
permits different long-run effects. Because the reported `ec=True` table uses
the equivalent **x_t** EC parameterization, its displayed SR coefficients are
not these conditional x_(t-1) gamma coefficients; equality is defined in the
explicit conditional representation above. Imposing both horizons for a
variable collapses its positive/negative lag effects to the ordinary symmetric
ARDL. With unequal orders, trailing coefficients above the common order become
exactly zero.

These are actual constrained fits. Given homogeneous levels restrictions
`R beta = 0`, a complete QR of `R.T` supplies a null-space basis `Q2`. The
estimator solves `y on X @ Q2` by Householder QR, reports `beta = Q2 @ gamma`,
and maps the free covariance back by `V_beta = Q2 @ V_gamma @ Q2.T`. Dependent
restrictions are reduced to independent rows. Residual degrees of freedom,
HC1 factors, HC2/HC3 leverage and information-criterion penalties use
`K_free = K − rank(R)`. EC coefficients and long-run effects use the constrained
levels covariance and the corresponding delta-method Jacobian.
The unrestricted lag design need not itself have full rank: restrictions can
identify `X @ Q2` even when the generated positive/negative sums and trend are
dependent. Dynamic columns are retained until this constrained identification
check. A deficient free design still fails explicitly; unrelated redundant
deterministic/exogenous columns are screened separately.

Automatic AIC/BIC selection also fits the **constrained** candidate models on
the common maximum-lag hold-back sample. One observation-level QR reduces the
full candidate design; each candidate then solves a coefficient-sized problem.
`extra["lag_selection"]` records the constraint rank and free parameter count
for the leading candidates. Equivalent restricted zero-padded models can tie
in the criterion; selected lag orders still retain the original common sample.

`extra["constraints"]` records the levels restriction matrix, its rank,
free parameter count and covariance mapping. Coefficients fixed exactly at
zero by the restrictions have no estimated standard error and are omitted
from the coefficient table; they are retained in `extra["constrained_terms"]`
and the complete saved levels representation. The same rule applies to fixed
zero SR terms in an EC table.

A symmetry already imposed or implied by the fitted constraints is marked
`imposed=True`, with no statistic or p-value. Remaining estimable restrictions
are tested using the constrained model and its degrees of freedom. For actual
restrictions, `tests["imposed_symmetry"]` additionally gives a joint Wald test
on the **same-sample, same-order unrestricted reference** with that reference's
N − K denominator degrees of freedom; it is not a test on the constrained
zero-variance contrast. If that reference is perfectly fitted, no reference
Wald statistic is fabricated. If only the constrained model is identified,
the reference is marked unavailable, with an explanatory note and no statistic
or p-value. Lag selection and any decisions about imposing
symmetry are conditioned on in this inference.

## Dynamic multipliers

`oe.nardl_multipliers(fit, steps=40, alpha=None)` returns an OpenEconometrics DataFrame
for horizons 0..steps, with positive/negative multipliers, their difference,
standard errors and pointwise confidence intervals. It exports to LaTeX and
can be passed to `oe.plot` after filtering the desired variable.

The cumulative response to a permanent +1 change in a partial-sum series is:

    m[h] = sum_(l<=h) b[l] + sum_(i=1..p) phi[i]*m[h-i], m[h<0] = 0

The derivative of this recursion with respect to every levels coefficient is
computed analytically; the interval variance is g[h]' V g[h]. These are
**pointwise delta-method** intervals conditional on the selected lag orders,
not simultaneous or bootstrap bands. `negative` is the derivative per +1 unit
of the negative partial sum; the actual response to a −1 shock in the original
series is **−negative**. Under stable AR dynamics, the paths tend to the
corresponding long-run coefficients. Stability is reported in table attrs;
unstable finite-horizon paths are not interpreted as a long-run limit.
For constrained fits, derivatives and covariance are evaluated in the free
parameter coordinates. An exactly symmetric multiplier contrast has a zero
estimate, standard error and interval width; covariance cancellation cannot
create a spurious tiny confidence band. Saved constraint mappings are validated
when multipliers are reconstructed after JSON reload.

## Recursive bootstrap multiplier intervals

```python
# Default method="delta" retains the existing analytic intervals.
bands = oe.nardl_multipliers(
    fit, data=df, method="residual", steps=40,
    replications=999, seed=2026, batch_size=64,
)
display(bands)
# The wild variant applies independent Rademacher {-1,+1} innovation weights.
wild_bands = oe.nardl_multipliers(
    fit, data=df, method="wild", replications=1999, seed=2026,
)
# Include the recorded assumptions when exporting a publication table.
bands.to_latex(notes=bands.attrs["notes"], index=False)
bands.attrs["bootstrap"]
```

These are **pointwise equal-tailed percentile** intervals, not simultaneous
bands. The original point estimates and the interpretation of the negative
partial-sum derivative remain unchanged. Bootstrap standard errors are the
sample standard deviation of the replicate multipliers with a B−1 denominator;
the endpoints are the alpha/2 and 1−alpha/2 empirical quantiles, using linear
interpolation. Coefficient covariance choice does not turn iid residual
resampling into a heteroskedastic bootstrap; choose the innovation scheme
explicitly.

A zero analytic delta standard error does not remove bootstrap uncertainty.
Nonlinear step responses can have zero first derivatives at a fitted parameter
value while their recursive refits vary. Exact zero contrasts are enforced only
when the homogeneous restriction null space proves the positive/negative
distributed coefficients identical through that horizon; the common AR filter
then proves the response difference zero for every allowed parameter draw.
Other empirical percentile endpoints and standard errors are retained.

For each draw, the original estimation residuals are centered and multiplied
by sqrt(N/(N−K_free)). `method="residual"` samples those innovations with
replacement. `method="wild"` multiplies each innovation at its original time
by an independent Rademacher weight. The outcome is **generated recursively**
from the original fitted levels equation and the observed pre-sample outcome
values; its lag regressors are then rebuilt from that generated outcome.
The refit is an actual QR least-squares fit in the original constraint null
space. It is neither a Gaussian coefficient draw nor a fixed-design refit
with the observed lagged outcomes left in place. LR/SR restrictions, selected
lags, contemporaneous controls and the deterministic case are preserved.

Lag selection is **conditioned on**, not repeated. An AIC/BIC-selected fit
retains its original maximum-lag hold-back even if the selected orders are
smaller. The predictor path, including generated positive/negative partial
sums, is held fixed, as in the single-equation algorithm of Brun-Aguerre,
Fuertes and Greenwood-Nimmo (2017), Appendix A. Residual resampling assumes
iid homoskedastic innovations independent of that conditioned path. The
Rademacher extension assumes serially uncorrelated martingale-difference
innovations, possibly conditional heteroskedasticity, and exogenous conditioned
predictors. Neither scheme reproduces contemporaneous outcome/regressor
innovation correlation or regressor feedback. The general coverage of these
conditional intervals for endogenous regressors or unit-root/local-to-unity
asymptotics has **not** been established here. Assess those assumptions and
serial correlation separately; this procedure is not a system or block
bootstrap.

The fitted conditional autoregression must be stable. An unstable original
fit fails explicitly rather than being presented as a calibrated unit-root
bootstrap. Finite-horizon draws whose refitted AR dynamics are unstable are
retained and counted, with a warning; no stable-root rejection filter changes
the empirical distribution. A singular free design, nonfinite simulated
outcome or nonfinite multiplier fails the entire call with an error. Replicates
are never silently discarded or replaced, and no partial intervals are
returned.

`data=` is required because a saved result retains only a bounded prediction
sample, not every residual. Model-input hashes check original rows, dtypes and
column values; estimation sample positions/hash, partial-sum origin, explicit
or selected orders, constraints, coefficients and the public levels/EC report
must agree with the saved result. Unused extra columns are allowed. Bootstrap
does not mutate the source data or the saved fit. JSON-reloaded fits can be
used with their original data. DataFrame rows and attrs are JSON serializable;
pass `notes=bands.attrs["notes"]` to retain the method/assumptions in LaTeX.

Numerical work uses float64 Torch on CPU, with a local seeded generator that
does not change the process-wide RNG. Changing `batch_size` preserves integer
draw order and results up to floating-point roundoff. Higher-order AR dynamics
use a direct O(Np) recurrence in the bundled native TorchScript interpreter,
processing every replicate in a batch together. Compilation uses a fixed,
trusted source string; no external compiler or runtime Python-source inspection
is needed. This avoids squaring nonnormal companion matrices, which can become
numerically inaccurate even when every root is strictly inside the unit circle.
Scalar AR(1) uses prefix doubling, with O(log N) Python blocks and O(N log N)
scalar work; an equation-residual check falls back to the direct native filter
if the prefix path fails its float64 certificate. There is no Python loop over
individual time rows. The actual algorithm, complexity and fallback batch count
are recorded in `bands.attrs["bootstrap"]["outcome_generation"]`; QR refits are
batched. Direct recurrence does not remove intrinsic conditioning problems:
high-order repeated roots near unity can amplify ordinary float64 roundoff.
Stable roots alone are not a guarantee of a well-conditioned multiplier path.
The 256 MiB inference workspace budgets generated/free designs, outcome states
and retained paths/quantile sorting, reducing the effective batch size when
necessary. If even one batch or the requested path/quantile storage exceeds
that workspace, an explicit capacity error asks for fewer replications or
horizons. This is an in-memory NARDL procedure, not a streaming estimator.
At least 20 replications are accepted for diagnostics; fewer than 999 carry a
Monte Carlo precision warning. The default is 999, and publication inference
should assess sensitivity to larger replication counts and seeds.

## Validation and remaining coverage

The tests compare levels coefficients and HC covariance to independent
statsmodels ARDL/OLS on NumPy partial sums; EC covariance to explicit Jacobian
algebra; lag selection to independent exhaustive OLS; symmetry to independent
Wald restrictions; and multiplier uncertainty to finite-difference derivatives
of an independent step-response simulation. They also exercise original-row
alignment, missing/gap rules, unsupported options and JSON/LaTeX round trips.
The constrained tests use a separate NumPy equality-constrained OLS/Lagrange
oracle, its observation influence matrix for every HC covariance, a numerical
EC Jacobian and exhaustive constrained lag searches. They cover different
predictor subsets, unequal lag orders, fixed zero terms, vacuous zero-lag SR
requests and the reduction to ordinary ARDL when both horizons are symmetric.
An independent explicit equality substitution also verifies a case where
long-run symmetry identifies an otherwise deficient partial-sum/trend design,
including HC3 covariance and constrained automatic lag selection.

Bootstrap tests compare actual recursive outcome generation, lag rebuilding,
constrained refits, step responses, percentile endpoints and empirical standard
errors against a separate NumPy/SVD oracle. They cover residual and wild draws,
deterministic cases II–V, unequal lag orders, symmetric controls, multiple
asymmetric predictors, original maximum-lag hold-back, sorted/missing-edge
samples, levels and EC reporting, vacuous zero-lag SR restrictions and omitted
static controls. A separate sequential AR oracle verifies AR(1) near unity,
oscillating AR(1)/AR(2), AR(3) through AR(6) with repeated stable roots at 0.9 and
0.99, and long/non-power-of-two sequences. Higher-order tests keep the oracle's
float64 addition order explicit and also check local equation backward error;
they do not claim exact-arithmetic accuracy for ill-conditioned dynamics.
Tests exercise cold native compilation, scalar-prefix certification/fallback,
actual algorithm/cost provenance and singleton lengths. They also check RNG isolation/batch invariance,
data/saved-state corruption, memory guards, whole-call failure without replacing
draws, unstable-refit reporting, and JSON/LaTeX exports. These numerical tests
establish implementation equivalence, not general asymptotic coverage.
An independently constructed stationary AR(2) fit additionally checks a
nonlinear cumulative response with zero first derivatives: residual and wild
refits still produce nonzero empirical confidence intervals. An unrelated
zero-lag symmetry retains only its proved response-difference identity.

The estimator records `stata_parity_validated=False`. Dependent generated partial sums
are not silently treated as independent ordinary ARDL regressors for critical
values: bounds F/t statistics are reported with `critical_values=None`,
`decision=None` and `p_value=None` until their NARDL calibration is independently
validated. The full plan's NARDL item therefore remains **partial**.

### Why linear PSS/SYG bounds are not attached

Shin, Yu and Greenwood-Nimmo (2014) read NARDL levels statistics against the
linear Pesaran–Shin–Smith bounds tables. OpenEconometrics does not attach those
bounds, even as an informational field labelled "not size-validated", because a
reported bound pair invites the cointegration decision that the evidence does
not support:

- The signed levels are generated from one raw series. With
  `s_t = sum_j |Delta x_j|`, `x+ = (x + s)/2` and `x- = (x - s)/2` up to the
  partial-sum origin. Even when `x` has no drift, `s` is I(1) with positive
  drift `E|Delta x|`, so `x+` and `x-` share one innovation and carry opposing
  deterministic drifts. Whether the linear tables remain valid for this
  generated pair has not been established. Counting `k` as the raw or the
  expanded number of regressors would be a convention choice, not a derivation.
- The frozen calibration (internal evidence excluded from this public snapshot)
  imposed each null in a recursive bootstrap and still failed its declared size
  gates under all four protocols (two seeds × two conventions). Each protocol
  had 14 or 15 violations of the nominal-5% gate, which requires a Wilson 95%
  upper bound of at most 0.08. At T=80, the adjustment and explanatory upper
  bounds ranged from 0.096 to 0.123.
- The [maintained-null analysis](nardl_bounds_next_protocol.md) shows that
  `rho = 0` with nonzero `theta` makes the outcome I(2), outside the I(0)/I(1)
  class that bounds tables assume.

MARKET-112 tracks the remaining research:

1. Write out the null limit analytically, without simulation.
2. If that limit is pivotal, tabulate it independently and run a new
   preregistered calibration.
3. Otherwise, close the item as a negative result.

The size gate, the failed receipts and the disabled decision stay unchanged in
the meantime. Linear tables are not substituted for NARDL-specific critical
values.

## Sources

- [Shin, Yu and Greenwood-Nimmo (2014), author publication page](https://www.greenwoodeconomics.com/publications.html),
  *Modelling Asymmetric Cointegration and Dynamic Multipliers in a Nonlinear
  ARDL Framework*, pp. 281–314, doi:10.1007/978-1-4899-8008-3_9.
- [EViews NARDL documentation](https://help.eviews.com/content/ardl-Views_and_Procs_of_ARDL.html),
  symmetry restrictions and dynamic multiplier conventions.
- [EViews worked NARDL example](https://blog.eviews.com/2022/09/nardl-in-eviews-13-study-of-bosnias.html),
  partial sums, long-run/cumulative short-run symmetry and the zero-lag caveat.
- [Brun-Aguerre, Fuertes and Greenwood-Nimmo (2017), accepted manuscript](https://openaccess.city.ac.uk/id/eprint/14781/1/JRSS_A_Fuertes%20et%20al_2016.pdf),
  *Heads I Win, Tails You Lose: Asymmetry in Exchange Rate Pass-Through into
  Import Prices*, JRSS A 180(2), 587–612, doi:10.1111/rssa.12213; Appendix A's
  single-equation residual recursion and multiplier percentile algorithm.
- [Gonçalves and Kilian (2002/2004), ECB Working Paper 196](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp196.pdf),
  *Bootstrapping Autoregressions with Conditional Heteroskedasticity of Unknown
  Form*, recursive wild-bootstrap methods and their stationary-autoregression
  assumptions; cited as the methodological background for the conditional
  extension, not as a general NARDL coverage theorem.
