# Calibration weighting and PPS without-replacement estimation

`oe.svycal` calibrates survey design weights to known population totals, and
`oe.survey_calibrated_total`, `oe.survey_calibrated_mean` and
`oe.survey_calibrated_ratio` estimate with the calibrated weights and the
calibration-adjusted (GREG-residual) Taylor variance under a declared
[single-stage design](survey-design.md). `oe.survey_pps_total`,
`oe.survey_pps_mean` and `oe.survey_pps_ratio` give Horvitz–Thompson estimates
under unequal-probability (PPS) sampling without replacement, with the
Horvitz–Thompson or Sen–Yates–Grundy variance from exact joint inclusion
probabilities, or with the Overton, Hartley–Rao, Brewer, Hájek and
with-replacement approximations that need only first-order probabilities. PPS
estimates can also be calibrated.

All numerical work is float64 PyTorch; no survey or estimation library runs at
run time. The procedures return tables (no `ModelSpec` estimator is
registered). Results are compared with R `survey` 4.5, R `sampling` 2.11 and
`samplics` 0.6 in the tests, never with a licensed Stata run, so
`stata_parity_validated` is `False`.

```python
import openecon as oe

design = oe.survey_design(df, weights="w", psu="psu", strata="region", fpc="n_psu_pop")
cal = oe.svycal(df, design, method="raking", population=10_400,
                categorical={"sex": {"F": 5_300, "M": 5_100},
                             "age": {"18-34": 3_000, "35-64": 5_200, "65+": 2_200}},
                totals={"income": 4.1e8})
cal["weights"]                     # design_weight, g_weight, calibrated_weight per row
oe.survey_calibrated_mean(df, design, ["spend"], calibration=cal, deff=True)

oe.survey_pps_total(sample, ["revenue"], prob="pi", strata="sector",
                    variance="syg", joint="hartley_rao")
```

| Stata / R | OpenEconometrics |
| --- | --- |
| `svycal regress x i.sex, generate(w2) totals(...)`; R `calibrate(d, ~sex + x, pop)` | `oe.svycal(df, design, method="linear", categorical=..., totals=..., population=N)` |
| `svycal rake i.sex i.age, ...`; R `calibrate(..., calfun="raking")`, `rake()` | `method="raking"` |
| R `calibrate(..., calfun="logit", bounds=c(L, U))`; `sampling::calib(method="logit")` | `method="logit", bounds=(L, U)` |
| R `calibrate(..., calfun="linear", bounds=c(L, U))`; `sampling::calib(method="truncated")` (different algorithm, see below) | `method="truncated", bounds=(L, U)` |
| R `calibrate` with group-interacted constraints; `samplics` `calibrate(domain=)` | `by="group"` with per-group constraints |
| `svy: total y` after `svyset ..., rake()/regress()`; R `svytotal(~y, cal)` | `oe.survey_calibrated_total(df, design, "y", calibration=cal)` |
| R `svymean`, `svyratio` on a calibrated design | `oe.survey_calibrated_mean`, `oe.survey_calibrated_ratio` |
| R `svydesign(fpc=~pi, pps=ppsmat(P), variance="HT"/"YG")`; `sampling::varHT(y, P, 1/2)` | `oe.survey_pps_total(df, "y", prob="pi", variance="ht"/"syg", joint=P)` |
| R `pps="overton"`; `pps=HR(psum)`, `HR()` | `joint="overton"`; `joint="hartley_rao", pi_squared_sum=...` |
| R `pps="brewer"` | `variance="brewer"` (default) |
| R `svydesign(probs=~pi)` without `fpc` | `variance="hansen_hurwitz"` |

SPSS Complex Samples and EViews have no calibration or PPS-without-replacement
variance procedure.

## `oe.svycal`

### The estimator

Calibrated weights `w_k = d_k g_k` satisfy the calibration equations
`sum_k w_k x_k = t_x` and minimise the distance `sum_k d_k G(g_k)` from the
design weights `d_k` (Deville and Särndal 1992; Deville, Särndal and Sautory
1993). With `F` the inverse of `G'`, `g_k = F(x_k' lambda)`, and Newton's method
on the Lagrange multipliers solves

```
phi(lambda) = sum_k d_k F(x_k' lambda) x_k - t_x = 0,     J(lambda) = sum_k d_k F'(x_k' lambda) x_k x_k'.
```

| `method` | `F(u)` | g-weights |
| --- | --- | --- |
| `linear` | `1 + u` (GREG, chi-square distance) | unbounded, may be negative; one Newton step |
| `raking` | `exp(u)` (multiplicative) | positive; equals iterative proportional fitting for categorical margins |
| `logit` | `L + (U - L) / (1 + exp(-(A u + c)))`, `A = (U-L)/((1-L)(U-1))`, `c = log((1-L)/(U-1))` | `L < g < U` |
| `truncated` | `min(max(1 + u, L), U)` | `L <= g <= U`; one side may be infinite |

These are R survey's `cal.linear`, `cal.raking` and `cal.logit` (R's
`cal.linear` with finite bounds is the truncated method). The logit form above
is algebraically identical to Deville and Särndal's
`(L(U-1) + U(1-L)e^{Au}) / (U-1 + (1-L)e^{Au})` and is evaluated with the
logistic function for stability. Each Newton step is halved until the scaled
misfit decreases (Armijo condition); when units sit at a truncation bound the
Jacobian is singular and the minimum-norm (pseudo-inverse) step is taken, as R
survey's `grake` does with `ginv`. When no halving lowers the misfit (a unit
sitting at a truncation kink, where the misfit falls only for vanishing steps),
the same Newton direction is searched on the convex dual objective
`sum_k d_k Psi(x_k' lambda) - lambda' t_x` (`Psi' = F`, gradient `phi`) instead.
Iteration stops when
`max_j |sum_k w_k x_kj - t_j| / max(|t_j|, sum_k d_k |x_kj|) <= tol`
(default `1e-10`, at most `max_iter=100` iterations).

### Constraints

- `population=N` constrains the sum of the weights.
- `categorical={"sex": {"F": 5300, "M": 5100}}` adds one indicator per declared
  level (a mapping or `[[level, count], ...]` pairs). Every sampled value needs a
  declared level; levels are matched by value (`1` matches `1.0`, booleans only
  match booleans). A declared level absent from the sample must have a zero
  count (its constraint is reported as `empty`).
- `totals={"income": 4.1e8}` constrains numeric columns. Auxiliary columns must
  be complete.
- `by="region"` calibrates each group separately (stratum-wise calibration).
  `population`, `categorical` and `totals` then map each group label to that
  group's constraints, with the same groups in every argument.

Constraints are screened for redundancy with the left-to-right QR rule used for
collinear regressors, applied to `sqrt(d) X`. Several categorical margins
together with `population` are redundant by construction; a redundant
constraint is kept only if its target equals the value implied by the sample
dependency (for example equal margin sums), otherwise the request fails with
`calibration_infeasible` naming the constraint and the implied target. Before
iterating, raking (`g > 0`) and the bounded methods check every constraint
against the totals attainable with each g-weight at its bounds; a target outside
that range fails immediately. Jointly infeasible bounds that pass this check end
with `calibration_not_converged`. The tests compare the outcome with a linear
programme (are there weights within the bounds satisfying the constraints?) at
several bounds, including upper bounds `1e-4` above and below the smallest
feasible one, where `logit` and `truncated` reproduce R survey's weights; a
feasible but extremely ill-conditioned problem can still need more than
`max_iter` iterations.

### What the result reports

A `TableSet` with

- `weights`: one row per input row, indexed like the input: `design_weight`,
  `g_weight`, `calibrated_weight`;
- `totals`: one row per declared constraint (prefixed `by=group:` with `by`):
  `target`, `design_total` (Horvitz–Thompson), `calibrated_total`,
  `difference`, `multiplier` (Lagrange multiplier of kept constraints) and
  `status` (`active`, `redundant`, `empty`).

`attrs` hold the method, bounds, iteration count, maximal relative misfit,
g-weight range, the number of negative calibrated weights (with a note), the
number of truncated units at a bound, the weight source, and
`attrs["calibration"]`: the canonical JSON specification. That specification
(possibly after `json.dumps`/`json.loads`), the `svycal` result itself, or a
plain mapping of the `svycal` arguments is accepted by the `calibration=`
argument of the estimation functions. An `svycal` result must reproduce on the
data it is used with: its design weights and g-weights are checked, and a
mismatch fails with `calibration_changed`.

Poststratification is calibration on one categorical margin: every method then
returns `g = N_c / N^_c` within cell `c`.

## Calibrated totals, means and ratios

For a target `theta` with row influence `u_k` — `y_k` for a total,
`(y_k - theta)/N^` for the Hájek mean `sum w y / sum w`, `(y_k - theta x_k)/X^`
for a ratio, each times the domain indicator — evaluated at the calibrated
weights, the linearized variable is

```
z_k = w_k e_k,    e_k = u_k - x_k' B,    B = (sum_k d_k x_k x_k')^{-1} sum_k d_k x_k u_k,
```

the residual of the design-weighted regression of the influence on the kept
calibration variables (within `by` groups), multiplied by the calibrated
weight. This is the g-weighted residual estimator of Deville and Särndal (1992)
and exactly R survey's convention for `calibrate()` designs
(`svyrecvar`: `qr.resid(qr(X sqrt(d)), x / (g sqrt(d))) * g sqrt(d)`). The
covariance is the first-stage with-replacement ultimate-cluster estimator over
PSU totals of `z`,

```
V = sum_h (1 - f_h) m_h/(m_h - 1) sum_j (z_hj - zbar_h)(z_hj - zbar_h)',    f_h = m_h / N_h,
```

with the `SurveyDesign` FPC (integer population PSU counts); census strata
contribute zero. Inference is Student t with all declared PSUs minus strata as
degrees of freedom, as in `oe.survey_total`. Calibration variables are
reproduced exactly with zero variance, which the tests check.

`domain=` names a complete Boolean/0-1 column; out-of-domain rows keep their
PSUs and enter with zero influence (never subset the data before estimation).
`missing="drop"` removes in-domain rows with any missing target value from the
estimand only; calibration always uses every row. `deff=True` adds the design
effect against R survey's `deff=TRUE` comparator: simple random sampling
without replacement of `n` units from `N^ = sum w`,
`V_srs = s_w^2(u) N^2 (N^ - n)/(N^ n)`, with `s_w^2` the weighted variance (divisor
`n-1` scaling) of the target's influence values over the eligible rows; it is
undefined (`NaN`) when `N^ <= n`.

## PPS sampling without replacement

`prob=` names the first-stage inclusion probabilities `pi_k` in `(0, 1]`;
`strata=` declares independent selections; `psu=` sums rows of a sampled
cluster (probabilities constant within PSU; PSU labels nested in strata). The
design weight is `1/pi_k`; totals are Horvitz–Thompson, means are Hájek ratios
`sum(y/pi)/sum(1/pi)` and ratios are ratios of Horvitz–Thompson totals. With
`z_k` the PSU sum of the linearized values (`y_k/pi_k` for a total) and, within
a stratum of `n` sampled PSUs, `D_kl = (pi_kl - pi_k pi_l)/pi_kl`,
`D_kk = 1 - pi_k`:

| `variance` | estimator | reference |
| --- | --- | --- |
| `ht` | `sum_k sum_l D_kl z_k z_l'` | Horvitz and Thompson (1952) |
| `syg` | `-1/2 sum_k sum_l D_kl (z_k - z_l)(z_k - z_l)'` | Sen (1953), Yates and Grundy (1953) |
| `brewer` (default) | `n/(n-1) sum_k (1 - pi_k)(z_k - zbar)(z_k - zbar)'` | Brewer (2002), Brewer and Donadio (2003), `c_k = (n-1)/(n-pi_k)`; R `pps="brewer"` |
| `hajek` | `n/(n-1) sum_k (1 - pi_k)(z_k - A)(z_k - A)'`, `A = sum (1-pi_k) z_k / sum (1-pi_k)` | Hájek (1964), Berger (2004) |
| `hansen_hurwitz` | `n/(n-1) sum_k (z_k - zbar)(z_k - zbar)'` | with-replacement approximation; R `probs=` without `fpc` |

`ht` and `syg` take `joint=`:

- a square matrix of `pi_kl` over all PSUs in order of first appearance
  (cross-stratum entries must equal `pi_k pi_l`), as a DataFrame, nested list or
  array; or a mapping `{stratum: matrix}` over each stratum's PSUs in order of
  first appearance. The diagonal must equal `pi`, the matrix must be symmetric,
  and every pair needs `max(0, pi_k + pi_l - 1) <= pi_kl <= min(pi_k, pi_l)` with
  `pi_kl > 0`. At most 4,000,000 matrix cells are admitted;
- `"overton"`: `pi_kl = 2(n-1) pi_k pi_l / (2n - pi_k - pi_l)` (Overton 1985);
- `"hartley_rao"`: `pi_kl = (n-1) pi_k pi_l / (n - pi_k - pi_l + sum_U pi_j^2 / n)`,
  the joint probabilities implied by the Hartley and Rao (1962) variance
  estimator (R survey's `HRDcheck`). `pi_squared_sum` gives the population `sum_U pi_j^2`
  (a number, or a mapping per stratum); by default it is estimated by the sample
  sum of `pi`, as in R survey's `HR()`. (R's `HR(psum)` takes `sum_U pi^2 / n`.)

The approximations have off-diagonal `D_kl = alpha + beta_k + beta_l` and are
evaluated in `O(n)` without forming the matrix. `variance="syg"` with
`joint="hartley_rao"` is the classical Hartley–Rao estimator
`1/(n-1) sum_{k<l} (1 - pi_k - pi_l + sum_U pi^2/n)(z_k - z_l)^2`; with the
sample-estimated sum it coincides with `brewer`. SYG and the approximations
assume fixed-size selections within strata. A stratum with one sampled PSU
fails (`singleton_pps_stratum`) unless that unit is a certainty unit
(`pi = 1`), which contributes zero variance; certainty units are best placed in
their own stratum, because the approximations do not set their `D_kl` to zero.

Horvitz–Thompson-form estimators can be negative for some samples; the full
covariance is retained in `attrs["covariance"]` and the table reports a `NaN`
standard error with a note. Degrees of freedom are all sampled PSUs minus
strata (R survey's `degf` on the full sample; for domains R counts only PSUs
with domain members, while here, as in `oe.survey_total` and Stata, the full
design df is kept). `deff=True` uses the comparator above with `N^ = sum 1/pi`
(R's `svytotal.pps` branch, whose comparator `s^2 N^2` omits the factor
`(N^ - n)/(N^ n)`, is not followed). Rows of one PSU are summed before the
Horvitz–Thompson/SYG forms are applied; R's `ppsmat` takes an element-level
matrix, whose YG form differences elements of the same PSU, so a clustered R
YG reference needs the PSU-aggregated sample (the tests do this).
`calibration=` calibrates the `1/pi` weights and uses the same g-weighted
residuals inside the chosen PPS variance, as R survey's `ppsvar` does for a
calibrated PPS design.

## What the estimation tables report

One row per target (`y`, or `y/x` for ratios) with `estimate`, `std_error`,
`statistic` (estimate over SE, null 0), two-sided design-t `p_value`, `ci_low`,
`ci_high` and optionally `deff`. `attrs` hold the full covariance matrix, df,
alpha, sample counts (`n_design`, `n_domain`, `n_used`, `n_missing_excluded`),
the design summary, the variance method and joint-probability source, and the
calibration record (diagnostics, canonical specification and linearization
convention). Everything in `attrs` is JSON serializable; no per-observation
arrays are stored.

## Worked example

```python
import pandas as pd
import openecon as oe

df = pd.DataFrame({
    "stratum": [1] * 6 + [2] * 6, "psu": [1, 1, 2, 2, 3, 3] * 2,
    "w": [10, 12, 9, 11, 10, 13, 8, 9, 7, 10, 9, 8],
    "sex": ["F", "M", "M", "F", "F", "M", "M", "F", "F", "M", "F", "M"],
    "x": [3.1, 4.0, 2.2, 5.1, 3.3, 4.4, 2.9, 3.8, 4.6, 2.5, 3.0, 4.1],
    "y": [20, 25, 14, 31, 22, 27, 19, 23, 28, 16, 18, 26],
    "N": [8] * 6 + [9] * 6,
})
design = oe.survey_design(df, weights="w", psu="psu", strata="stratum", fpc="N")
cal = oe.svycal(df, design, method="logit", bounds=(0.5, 2.0), population=125,
                categorical={"sex": {"F": 64, "M": 61}}, totals={"x": 450.0})
print(cal["totals"])
print(oe.survey_calibrated_total(df, design, "y", calibration=cal))
```

## Validation

`tests/test_survey_calib_calibration.py`, `tests/test_survey_calib_estimation.py`
and `tests/test_survey_calib_pps.py` compare against:

- R survey 4.5 (`calibrate` with `epsilon=1e-14` for the four distances and
  stratum-wise raking; `svytotal`, `svymean`, `svyratio` with `deff=TRUE` and
  domains on a stratified, clustered design with FPC; `ppsmat` HT/YG,
  `overton` HT/YG, `HR(psum)` and `HR()` HT/YG, `brewer`, with-replacement and
  calibrated PPS designs) and R sampling 2.11 (`calib`, `UPsampfordpi2`,
  `varHT`), generated by `tests/reference/survey_calib_calibrate.R` and
  `tests/reference/survey_calib_pps.R`; agreement is 1e-9 relative or better;
- an independent review, `tests/test_survey_calib_review.py` with
  `tests/reference/survey_calib_review.R`: unequal PSU counts and sizes with a
  census and a singleton certainty stratum, domains combined with
  `missing="drop"` for calibrated totals, means and ratios (with `deff`),
  logit/truncated bounds `1e-4` from infeasibility, by-group calibration across
  PSUs, and a clustered PPS design with a certainty stratum under every
  variance option, with and without calibration (R survey 4.5; `1e-9` relative
  for g-weights at the feasibility edge, `1e-10` elsewhere), plus the Hájek
  centre against `sampling::varest`;
- samplics 0.6 (linear calibration, raking ratio, within-domain calibration),
  `tests/reference/survey_calib_samplics.py`;
- independent NumPy/SciPy oracles: the linear closed form
  `g = 1 + x'(X'DX)^{-1}(t - X'd)`, raking against iterative proportional
  fitting, logit by SciPy root finding of Deville and Särndal's original
  formula, truncated by SciPy exact-Hessian trust-region minimisation of the
  convex dual (successful convergence and a scaled gradient at most `1e-13`
  are required), bounded
  infeasibility by a linear programme, the GREG residual variance written
  out over PSUs and strata, and the Overton/Hartley–Rao closed forms against
  explicit `n`-by-`n` Delta-check matrices;
- exact checks: Horvitz–Thompson and SYG unbiasedness by enumerating all 120
  samples of a Sampford design (whose joint probabilities are also checked
  against `UPsampfordpi2`), exact unbiasedness of every approximation under
  simple random sampling without replacement by enumeration, poststratification,
  zero variance of calibration variables, invariance to row order, weight scale
  and PSU splitting, analytic distance derivatives against finite differences,
  JSON round trips and the failure contract.

## Limitations

- Single-stage designs only: calibrated estimation uses the declared first-stage
  PSUs with replacement plus FPC; PPS designs are single-stage (or first-stage
  clusters fully observed). Multistage PPS, calibration at a cluster level (R
  `aggregate.stage`), calibration within clusters (`stage=`), heteroscedasticity
  factors `q_k` (R `variance=`, sampling `q`), bounded raking, trimming,
  replicate-weight calibration and regression models with calibrated weights are
  not provided.
- Horvitz–Thompson/SYG need joint probabilities supplied by the user; no
  sampling-design generator is included.
- Budgets: at most 1,000,000 rows, 32 targets, 200 constraints and 1,000 levels
  per group, 1,000 calibration groups and 4,000,000 joint-probability cells.

## Conventions that could not be confirmed

- Stata 18 `svycal` and `svyset ..., rake()/regress()` were not run. Their
  variance convention for calibrated weights and the definition of their bounded
  (`ll()`/`ul()`) options were not compared; the equivalences in the table are
  by method name only.
- R survey details that are deliberately not mirrored: `grake` rescales the
  weights when all population/sample total ratios exceed 20 (bounds are then
  reinterpreted); `HRDcheck` sets `D_kl = 0` when `pi_k + pi_l = 1` exactly;
  `svytotal.pps` computes its design effect without the `(N^ - n)/(N^ n)`
  factor; `[.pps` (subsetting a PPS design to a domain) zeroes the Delta-check
  entries of out-of-domain pairs, which after calibration drops the nonzero
  g-weighted residuals of out-of-domain units from the variance (here the full
  form is used: the tests reproduce R's number by imitating that zeroing); bounded
  `cal.raking` uses a derivative that ignores its own truncation;
  `sampling::calib` `truncated`/`logit` fix units that once reach a bound,
  which can differ from the Deville–Särndal truncated solution returned here
  (identical to R survey's).

References: Deville, J.-C. and Särndal, C.-E. (1992), Calibration estimators in
survey sampling, *JASA* 87, 376–382; Deville, Särndal and Sautory (1993),
Generalized raking procedures in survey sampling, *JASA* 88, 1013–1020;
Särndal, Swensson and Wretman (1992), *Model Assisted Survey Sampling*;
Horvitz and Thompson (1952); Sen (1953); Yates and Grundy (1953); Hartley and
Rao (1962); Hájek (1964); Overton (1985); Brewer (2002), *Combined Survey
Sampling Inference*; Brewer and Donadio (2003), *Survey Methodology* 29;
Berger (2004); Sampford (1967); Lumley (2004), R package survey.
