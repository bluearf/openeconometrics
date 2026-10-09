# Known-nuisance causal survival targets

These three procedures add observational survival adjustment to the randomized
`treatment_rmst` route. They require `design="unconfounded"` and
`nuisance="known"`: the propensity column gives the **true known** conditional
treatment probability, and censoring inputs give the **true known** conditional
censoring law. Supplying fitted values does not turn them into known nuisances.
In-sample fitted probabilities, estimated censoring models, nuisance-parameter
uncertainty, cross-fitting, clusters and survey weights are unsupported.

Identification requires consistency/no interference, independent population
sampling, conditional exchangeability of potential event times given pretreatment
covariates, and the declared positivity bounds. Censor-adjusted targets additionally
require event and censoring times to be conditionally independent given treatment
and those covariates. These are caller declarations; the program cannot verify
unconfoundedness or model provenance from the input columns.

`time` is observed `U=min(T,C)`; `event` is numeric0 for right censoring and1 for
failure. Both treatment arms are numeric0/1 and require at least three subjects.
The event indicator is validated and saved. For the IPCW score formulas below,
the observed survival indicator `U>t` already includes both event and censoring
times, so the event indicator is not separately multiplied into the score.

## Fixed-grid survival with known censor-survival probabilities

```python
result = oe.treatment_survival_ipcw(
    data, "observed_time", "event", "treated", "true_propensity",
    ["G_0", "G_1", "G_2"],
    design="unconfounded", nuisance="known", thresholds=[0.0, 1.0, 2.0],
)
```

The distinct censor-survival columns correspond exactly to the strictly
increasing, prespecified grid of1..32 nonnegative times. For every subject they
contain `G_a(t|X)=P(C>t|A=a,X)` for the received treatment, are nonincreasing over
the grid, and lie in `[positivity,1]`. The propensity lies in
`[positivity,1-positivity]`, strictly within `(0,1)`; `positivity` itself must be
strictly between0 and0.5. The program refuses violations and does not trim or
clip probabilities.

For each arm and threshold, the subject score is

`Z_i,a(t) = I(A_i=a) I(U_i>t) / (p_a(X_i) G_a(t|X_i))`.

Its cohort mean estimates `S_a(t)=P(T(a)>t)`, and the effect is `S_1(t)-S_0(t)`.
Strict `>` is used for both observed time and the supplied censor-survival
probability: an event or censoring at the threshold is excluded from its observed
survival indicator. Under the stated independence and true-known probability
assumptions, the identifying expectation follows by cancelling treatment and
censor-selection probabilities. This direct identity is the implemented
Horvitz–Thompson route, distinct from a weighted hazard or Kaplan–Meier estimator.

HT curve estimates can exceed1 or increase between thresholds in a finite sample.
Those diagnostics are retained; no monotonicity projection, probability clipping
or automatic change of estimand is applied. The fixed grid is not interpolated
into an RMST integral. Identification at a threshold uses the declared true G
at that time rather than an extrapolated fitted event-time distribution.

## RMST with a known conditional exponential censoring law

```python
result = oe.treatment_rmst_ipcw(
    data, "observed_time", "event", "treated", "true_propensity", "true_censor_rate",
    design="unconfounded", nuisance="known", tau=3.0,
)
```

This bounded route requires the true conditional censor-survival law
`G_a(t|X)=exp(-lambda_i*t)` for `0<=t<tau`, where `tau` is positive, finite and
prespecified. Rates are nonnegative and the left limit
`G_a(tau-|X)=exp(-lambda_i*tau)>=positivity`; the exponential law before the
horizon is a substantive assumption. Administrative censoring may have an atom
at `tau`, which does not change this integral. Zero rate means no censoring
before `tau`, so a zero-rate subject censored earlier is structurally inconsistent
and refused. Administrative censoring at/after the horizon remains compatible.

Integrating `I(U_i>t)/G_a(t|X_i)` from0 to `tau` gives

`H_i = expm1(lambda_i*min(U_i,tau))/lambda_i`,

with the exact zero-rate limit `H_i=min(U_i,tau)`. The two arm scores are
`I(A_i=a) H_i/p_a(X_i)`. Their means and difference target
`E[min(T(a),tau)]` and the marginal RMST contrast in the sampled superpopulation.
This analytic identity handles early censoring through the known exponential
law; it does not fit an event distribution or claim a general IPCW-RMST engine.
The finite-sample HT RMST estimate can exceed `tau`; it is retained without
clipping. Positivity is assessed at the left limit at `tau`, including beyond observed times, using
the explicitly declared known law.

## Complete-horizon RMST with known propensity and fixed augmentation

```python
result = oe.treatment_rmst_aipw(
    data, "observed_time", "event", "treated", "true_propensity", "external_m0", "external_m1",
    design="unconfounded", nuisance="known", tau=3.0,
)
```

Every restricted event time must be observed: `event=1` or `U>=tau`. Early
censoring is refused. Then `z_i=min(U_i,tau)` is the fully observed capped event
time. `m0` and `m1` are externally prespecified functions in `[0,tau]`, fixed
independently of these analysis data. With true known propensity they may be
incorrect conditional-mean predictions; their correctness is not needed for
the mean identity.

The arm score is

`Z_i,a = m_a(X_i) + I(A_i=a) (z_i-m_a(X_i))/p_a(X_i)`.

Inference treats the external augmentation functions as fixed and uses the full
population-sampling covariance of these scores. This is not an estimated-nuisance
or doubly robust inference route. The same-data fitted-prediction uncertainty
cannot be recovered by labelling its column external or known.

## Inference, persistence and numerical admission

All three return `TableSet` tables named `effects`, `covariance`, `subjects` and
`scores`; the fixed-grid survival procedure also returns `curve`. The complete
joint covariance covers every control mean, treated mean and contrast (and every
grid time), using the unbiased sample covariance of full-cohort subject scores
divided by cohort size. This HC1 convention includes the `n/(n-1)` factor.
Cross-arm covariance is retained: the mutually exclusive HT arm scores generally
produce negative off-diagonals after cohort centering. Inference does not condition
on observed arm counts and does not substitute independent-arm Greenwood variance.

Intervals are pointwise asymptotic normal, with `df=None`. No simultaneous or
finite-sample coverage is claimed. Normal intervals and score estimates are
untruncated. Zero empirical variance gives a point interval, a degeneracy flag
and undefined z/p; it is not proof of population certainty.

Native CPU float64 Torch performs scoring, bounded floating-expansion means, complete covariance
and inference. No SciPy/NumPy estimation or random generator is used. Only resident
tables and `missing="raise"` are supported, preserving original row alignment;
generic weights, Dataset replay and CUDA/MPS are refused. The shared input limits
are100,000 rows and64 selected columns. Complete work and conservative workspace
plans are checked from the original row count before the selected sample is copied
or numerical scores are allocated. Budgets refuse the full computation rather
than truncating a grid or silently deleting rows; buffer estimates are not RSS
limits. Score sums retain nested cancellation remainders with a native FastTwoSum
expansion and final float64 rounding. Its hard64-part capacity and worst-case
arithmetic/buffers are included in preflight; exceeding capacity is an explicit
numerical refusal, never truncation of residuals.

Underflow of a nonzero contribution/covariance, overflow, absorbed augmentation
correction, or an unrepresentable nonzero confidence endpoint shift fails
explicitly. Positive epsilon does not repair numerical precision. All selected
source identities, original positions/labels, probabilities, rates, integrals,
augmentation functions, every subject score, covariance, settings and assumptions
survive checksummed `causal_design_save/load` without refitting.

The independent tests enumerate a complete discrete treatment/event/censoring
DGP, verify its population survival identities including ties, enumerate treatment
assignments with imperfect fixed augmentation, compare closed-form RMST scores
with dense quadrature and the exact continuous censoring expectation, and verify
all covariance cells and normal intervals independently. Additional tests cover
private inputs, full JSON/file restoration, global device state, positivity,
missing/censoring boundaries, degeneracy, numerical refusals and full preflight.
These are bounded method checks; vendor or whole-product parity is not claimed.

Primary references:

- [Cao et al. (2025), causal RMST weighting and censoring assumptions](https://academic.oup.com/aje/article/194/8/2402/7841807) and [author manuscript](https://arxiv.org/pdf/2304.00231). The paper's estimated propensity/censor-model variance has additional influence contributions; its hazard estimator is distinct from these known-nuisance HT score means.
- [Robins, Rotnitzky and Zhao (1994), inverse-probability estimating equations and augmentation](https://doi.org/10.1080/01621459.1994.10476818).
- [Tsiatis et al. (2008), covariate augmentation with known assignment probabilities](https://doi.org/10.1002/sim.3113).
