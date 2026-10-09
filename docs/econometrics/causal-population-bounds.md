# Population Manski regions and prespecified discrete Lee bounds

MARKET-662 and MARKET-663 extend the earlier empirical identification bounds
with two different targets. `manski_ate_inference` estimates population
consistency bounds and an asymptotic region covering their entire identified
set. `stratified_lee_bounds` tightens the always-selected Lee target using a
caller-declared discrete pretreatment partition; it reports identification
bounds without sampling inference. These are bounded additions under
MARKET-187/GitHub #157, not general partial-identification or vendor parity.

```python
population = oe.manski_ate_inference(
    data, "outcome", "treated", lower=0, upper=1, sampling="iid", level=.95,
)
conditional = oe.stratified_lee_bounds(
    data, "outcome", "treated", "selected", strata="baseline_group",
    levels=["A", "B"], design="randomized_within_strata",
    monotonicity="increasing",
)
```

## IID population consistency bounds

Both potential outcomes must lie in the fixed, prespecified common support
`[a,b]`. Consistency/SUTVA relates the observed outcome to the received
treatment. No random treatment, ignorability, exclusion restriction or outcome
monotonicity is imposed. The observed rows must be an IID complete sample;
`sampling="iid"` is required and is a declaration, not a diagnostic result.
The sharp population ATE identification interval has endpoint expectations

`L = E[D*(Y-b) + (1-D)*(a-Y)]`,

`U = E[D*(Y-a) + (1-D)*(b-Y)]`.

Every row satisfies `U_i-L_i=b-a`. Consequently the population interval has
fixed width `b-a`, and both estimated endpoints have the same sampling error.
The implementation uses the common center score

`C_i=(2D-1)*(Y-(a+b)/2)`

and constructs the endpoints as `mean(C) +/- (b-a)/2`. The midpoint is held as
its high float64 component and exact low remainder; subtraction also retains
its remainder. All components are summed before the final center-score
rounding. This preserves small observed outcomes under wide symmetric support
and removes large common locations without discarding midpoint residuals.
The full 2-by-2 endpoint
covariance has every entry equal to

`vhat = sum[(C_i-mean(C))**2] / [N*(N-1)]`.

This HC1 IID mean covariance is rank one when empirical variance is positive;
it is not inverted, regularized or treated as independent-arm uncertainty.
All endpoint scores, centered scores, normalized contributions, covariance
and the four missing-potential-outcome extremizers are retained.

For `alpha=1-level`, the reported confidence region is

`[Lhat-z_(1-alpha/2)*SE_L, Uhat+z_(1-alpha/2)*SE_U]`.

The two one-sided endpoint limits use Bonferroni allocation. Under the stated
IID nondegenerate-score CLT, this region covers the **entire population
identified interval** asymptotically. Here the perfect endpoint dependence
also reduces coverage to the usual two-sided center-score event. This is not
a finite-sample coverage guarantee or the optimized Imbens–Manski interval
for one unknown point in an identified set. Small admitted samples do not
establish the asymptotic approximation.

A singleton common support identifies ATE=0 algebraically. Otherwise an
empirically constant center score is labeled `zero_empirical_endpoint_variance`:
its covariance is zero and its region equals the plug-in identified set.
That empirical degeneracy does not prove zero uncertainty about unsampled
outcomes. Positive variance or confidence shifts lost to float64 underflow or
rounding are refused; no epsilon variance is substituted.

The four tables are `bounds`, `endpoint_covariance`, `subject_scores` and
`extremizers`. The confidence-region columns differ explicitly from the
identification-bound columns.

## Discrete conditional-randomization Lee bounds

`strata` names one fixed pretreatment discrete column. `levels` is a required
list of 1..32 typed, unique labels; there is no automatic binning, feature
selection or partition search. Numeric, Boolean and string labels are retained
with their types, so the integer `1`, string `"1"`, Boolean `True` and float
`1.0` are distinct labels. Integer labels must fit signed int64 and string
labels have at most 1024 UTF-8 bytes. The labels must match the actual resident
column types; every original row must belong to one declared level.

`design="randomized_within_strata"` declares that treatment is conditionally
independent of potential outcomes and selection within each fixed stratum.
Treatment allocation fractions may vary between cells. Consistency/SUTVA and
a common unit-level selection direction are also required. Explicit
`monotonicity="increasing"` means `S(1)>=S(0)`; `"decreasing"` means the reverse.
The target is the ATE among subjects selected under **both** assignments.
The implementation does not allow a data-chosen direction or cell-specific
direction switching.

For each cell `x`, all original rows determine its arm counts and selection
rates `s_d(x)`. The arm with higher selection is fractionally trimmed to
`s_low(x)/s_high(x)` of its selected empirical outcome distribution. Retaining
its lowest or highest outcome mass gives the two cell bounds, with contrast
orientation reversed for decreasing monotonicity. A boundary tie shares its
fractional retention equally among equal outcomes. Integer cross-products
determine the rate ordering and exact rational retained mass.

The always-selected distribution of covariates determines the aggregation:

`w_x = P_hat(X=x)*min(s_hat_0(x),s_hat_1(x))`

`pi_x = w_x / sum_x w_x`.

Overall endpoints average the cell endpoints using `pi_x`. Raw cohort cell
shares and raw selected low-arm cell counts are different weights. The latter
would additionally require constant treatment allocation across cells. The
full-cohort proportions above support the declared within-cell randomization
with unequal allocation fractions.

Every declared cell needs both treatment arms and at least one selected
outcome in each. A rate ordering inconsistent with the declared direction in
any cell rejects the entire result. No cell is discarded, direction reversed
or rate clipped. Such incompatibility can arise from sampling noise and is
not a test of population monotonicity. With one cell, this route reduces to
the earlier unstratified Lee functional. Population covariate information can
tighten bounds, but arbitrary finite-sample partitions are not guaranteed to
yield narrower estimates than a pooled calculation.

Both arm mean weights sum to one algebraically within each cell. Signed
outcome contributions subtract a common observed low-arm outcome anchor
before summation, so different rounded inverse cell sizes cannot turn a large
common outcome location into an invented effect. Anchors, full source
outcomes, all local/population mean weights and resulting contributions are
saved. Native bounded summation expansions retain nested cancellation
remainders. The three tables are `bounds`, `strata` and `trimming_weights`.

These are plug-in sharp conditional-law identification bounds. Covariance,
standard errors and sampling confidence intervals are explicitly absent.
Generalized varying-sign/continuous-covariate Lee bounds and their nuisance
estimation or nonregular sampling inference are outside this API.

## Admission, persistence and independent checks

Native Torch CPU float64 executes both kernels. Generic observation weights,
GPU and Dataset replay are refused. Inputs are limited to 10000 original rows;
Manski needs at least three rows and both numeric 0/1 arms. Outcomes and
support endpoints must be finite with absolute magnitude at most `1e150`.
Larger locations, including `1e300`, require rescaling and are explicitly
refused. Treatment and selection Boolean labels are not recoded to numeric
assignments. Both procedures require `missing="raise"`.

For Lee, every original treatment/selection/stratum row must be complete.
Unselected outcomes may be missing and are ignored; selected outcomes must be
finite numeric values. Unselected outcome values are never imputed and cannot
change the scientific identity. A separate full scientific sample hash includes
the complete topology, typed levels and every selected outcome at its original
physical position, together with its declared column dtype.

Before sample or tensor allocation, the original row count admits the full
work and workspace plan: every mean, sort, fractional tie, 64-part bounded
summation expansion, covariance, extremizer, source and complete portable
state. Default `max_work=100_000_000` can refuse the largest allowed Lee inputs;
raising it explicitly admits the declared complete plan. No output is
truncated to fit a budget. Workspace accounting is a named-buffer estimate,
not a total process-RSS guarantee.

The complete escaped JSON bytes of the declared typed labels are charged for
repeated row identities, hashing/state and portable copies. Every observed
scalar is checked against those declarations before repeated row-label
allocation. A 1024-byte string of control characters therefore consumes its
larger escaped JSON budget; UTF-8 length alone is not used as a saved-state
memory bound.

Original index identities are also charged by their actual escaped JSON size
before sample construction. Each saved index label is limited to 8192 JSON
bytes; within that bound, large signed or unsigned 64-bit integer labels remain
integer identities. All repeated metadata/state and portable index-label
copies enter the plan.

`causal_design_save/load` retains every table and its dtype, full source/sample
identity, assumptions, scores, extremizers, cell rates, rational trim masses,
tie retention and aggregation weights with checksums. It does not authenticate
an artifact or establish the caller's identifying assumptions.

Independent fixtures enumerate all missing-potential-outcome completions,
derive endpoint covariance from a fully enumerated bounded IID law, and solve
separate constrained fractional-retention linear programs. Tests cover unequal
cell allocation, exhaustive retained subsets, tie symmetry, exact no-trimming
and one-cell reductions, a population-law tightening example, typed labels,
ignored unselected outcomes, full typed JSON/LaTeX, private RNG/default-device
behavior, scale/translation, underflow and resource refusals. A fixed medium
IID DGP is a bounded calibration diagnostic, not universal coverage evidence.

Primary references:

- [Imbens and Manski: confidence intervals for partially identified parameters](https://www.econstor.eu/bitstream/10419/79355/1/37481080X.pdf), especially the distinction between coverage of a parameter and coverage of its identified region.
- [Lee: Training, Wages, and Sample Selection](https://www.princeton.edu/~davidlee/wp/Selection5all.pdf), covariate trimming and aggregation over always-selected subjects.
- [Semenova: Generalized Lee Bounds](https://arxiv.org/abs/2008.12720), conditional formulations and the distinction from generalized high-dimensional inference.
