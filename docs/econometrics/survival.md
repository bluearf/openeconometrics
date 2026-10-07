# Survival analysis: `sts`, `stcox`, `stcurve`, `streg`, `ltable`

Survival (duration, time-to-event) analysis in the tradition of Stata's `st`
suite and SPSS's `KM`, `COXREG` and `SURVIVAL` procedures. Everything on this
page is implemented in OpenEconometrics on float64 PyTorch tensors: risk sets are built
from one sort of the data and cumulative sums, the Cox partial likelihood and
the parametric likelihoods have analytic scores and Hessians, and no
estimation library runs at fit time. A Cox model on one million records with
ten covariates fits in about one second (timings at the end).

| Stata | SPSS | OpenEconometrics |
| --- | --- | --- |
| `sts list`, `sts list, cumhaz`, `stsum`, `stci` | `KM t /STATUS=d(1) /PRINT=TABLE MEAN` | `oe.sts(df, "t", failure="d")` |
| `sts test g` / `, wilcoxon tware peto fh(p q)` / `, strata(s)` / `, trend` | `KM t BY g /TEST=LOGRANK BRESLOW TARONE /STRATA=s /TREND` | `oe.sts(df, "t", failure="d", by="g", test=..., strata="s", trend=True)` |
| `stcox x1 x2` (`, efron` / `exactp`) | `COXREG t /STATUS=d(1) /METHOD=ENTER x1 x2` | `oe.stcox(data=df, time="t", failure="d", x=["x1","x2"], ties=...)` |
| `stcox x, strata(s) vce(robust)` | `COXREG ... /STRATA=s` | `oe.stcox(..., strata="s", covariance="robust")` |
| `stcox x, tvc(z) texp(ln(_t))` | `COXREG ... T_COV_` | `oe.stcox(..., tvc=["z"], texp="log")` |
| `estat phtest, detail` / `estat concordance` | — | `result.tests["ph_*"]` / `result.metrics["concordance"]` |
| `stcurve, survival at(x=1)` | `COXREG ... /PLOT SURVIVAL` | `oe.stcurve(result, data=df, at={"x": 1})` |
| `streg x, distribution(weibull)` (`time`, `ancillary()`, `strata()`) | — | `oe.streg(data=df, time="t", failure="d", x=["x"], distribution="weibull")` |
| `ltable t d, by(g) intervals(5) hazard test` | `SURVIVAL TABLE=t BY g /INTERVAL=THRU 40 BY 5 /STATUS=d(1)` | `oe.ltable(df, "t", failure="d", by="g", intervals=5)` |

## Worked example (Stata's `kva` data)

The generator experiment used in [ST] stcox and [ST] streg. Every number below
was produced by the code shown; the `stcox` and `streg` numbers match the output
printed in Stata's manuals (the `sts` lines are OpenEconometrics's own output).

```python
import pandas as pd, openecon as oe
kva = pd.DataFrame({"failtime": [100, 140, 97, 122, 84, 100, 54, 52, 40, 55, 22, 30],
                    "load": [15, 15, 20, 20, 25, 25, 30, 30, 35, 35, 40, 40],
                    "bearings": [0, 1] * 6})

cox = oe.stcox(data=kva, time="failtime", x=["load", "bearings"])
cox.metrics["log_likelihood"]            # -8.577853   (Stata: -8.577853)
cox.tests["model"]["statistic"]          # 23.39       (LR chi2(2) = 23.39)
cox.extra["hazard_ratios"]["load"]       # hazard_ratio 1.52647 (Stata: 1.52647)

weib = oe.streg(data=kva, time="failtime", x=["load", "bearings"])   # Weibull PH
# Intercept -45.131913, load .469575, bearings -1.667069, /ln_p 2.051552
weib.metrics["log_likelihood"]           # 5.6934189   (Stata: 5.6934189)
weib.extra["p"]["estimate"]              # 7.779969

km = oe.sts(kva.assign(heavy=(kva.load >= 30) * 1), "failtime", by="heavy", test="logrank")
km.attrs["statistic"], km.attrs["p_value"]   # 12.094 on 1 df, p = 0.000506
km["summary"][["group", "median", "median_ci_low", "median_ci_high"]]
#    group  median  median_ci_low  median_ci_high
# 0      0   100.0           84.0           140.0
# 1      1    40.0           22.0            55.0
```

## Survival data

Every function declares the survival data in the call, the way `stset` does:

- `time`: the exit time `t_i` of each record (failure or censoring), `>= 0`.
- `failure`: the 0/1 event indicator (every record fails when omitted). Codes
  other than 0 and 1 are refused (`invalid_failure_indicator`); Stata would
  treat every nonzero value as a failure, OpenEconometrics asks for an explicit coding.
- `entry`: the entry time `t0_i` (delayed entry / left truncation, or the start
  of a `(start, stop]` record of counting-process data), 0 when omitted.
- `id` (`stcox`, `streg`): the subject of multiple records. Records of one id
  must describe disjoint intervals (`overlapping_records` otherwise).

A record is at risk at time `s` when `t0_i < s <= t_i`. Records with
`t_i <= t0_i` (for example a time of 0 without entry) are never at risk and are
excluded with a recorded note, as `stset` excludes them ("obs. end on or before
enter()"). Negative times raise `invalid_survival_time`. `missing="drop"`
excludes incomplete rows (the default of `sts` and `ltable`, as in Stata and
SPSS); the estimators default to `missing="raise"`, like every OpenEconometrics
estimator.

**How risk sets are computed.** All time values (exits and entries) are
replaced by their dense ranks and combined with the stratum code into one
integer key `stratum * (R + 1) + rank`. One sort of the exit keys and one of
the entry keys then serve every stratum:

    sum_{i at risk at tau} v_i = sum_{t_i >= tau, same stratum} v_i - sum_{t0_i >= tau, same stratum} v_i

where both terms are differences of suffix sums at positions found by binary
search. The transpose, `A_i = sum_{tau at which i is at risk} a(tau)`, is a
difference of prefix sums. Every risk-set quantity of this page (Kaplan-Meier,
the log-rank family, the Cox score and Hessian, residuals) is built from these
two operations: O(n log n) once, O(n k) per evaluation, no loop over
observations and no n-by-n object.

## `oe.sts`: Kaplan-Meier, Nelson-Aalen and the log-rank family

```python
out = oe.sts(df, "t", failure="died", by="drug", test="logrank")
out["survival"]     # per group and distinct time
out["summary"]      # median, quartiles, restricted mean
out["tests"]        # log-rank family
out.attrs["p_value"]
```

**Estimates** (Stata [ST] sts, Methods and formulas). With `n_j` at risk just
before the failure time `t_j` and `d_j` failures at `t_j`:

    S(t)       = prod_{t_j <= t} (n_j - d_j) / n_j                       Kaplan-Meier
    Var S(t)   = S(t)^2 sum_{t_j <= t} d_j / (n_j (n_j - d_j))           Greenwood
    H(t)       = sum_{t_j <= t} d_j / n_j,   Var H(t) = sum d_j / n_j^2   Nelson-Aalen

Pointwise intervals for `S` (`conftype`): `"loglog"` (Stata's default) uses the
asymptotic variance of `ln(-ln S)`, `sigma^2 = sum d/(n(n-d)) / (sum ln((n-d)/n))^2`,
giving `S^{exp(+-z sigma)}`; `"log"` gives `S exp(+-z sqrt(sum d/(n(n-d))))`
(capped at 1); `"plain"` gives `S +- z se` clipped to [0, 1] (SPSS). Nelson-Aalen
intervals are `H exp(+-z se_H / H)`. Where everyone at risk fails, `S` drops to
0 and its standard error and interval are reported as missing (Stata prints a
dot).

The `survival` table has one row per group and distinct exit time (failure or
censoring): `time, n_risk, n_event, n_censored, survivor, std_error, ci_low,
ci_high, cumulative_hazard, cumulative_hazard_std_error` and the hazard
interval. A group with more than 2000 distinct times is thinned to 2000 evenly
spaced rows (always including the first and last), and the thinning is noted in
`attrs["notes"]`.

**Summary** (`stsum`, `stci`). Percentile `p` is the smallest failure time with
`S(t) <= 1 - p` (Stata's definition; R averages over a flat stretch instead).
The confidence limits of the median are the first times at which the lower and
the upper pointwise limits of `S` fall to 0.5 or below (Brookmeyer-Crowley
inversion, as `stci` does it, with the chosen `conftype`). The restricted mean
is the area under `S` from 0 to the largest observed time, with
`SE^2 = sum_j A_j^2 d_j / (n_j (n_j - d_j))`, `A_j` the area from `t_j` to the
largest time (`stci, rmean`); when the largest time is censored the restricted
mean underestimates the mean and a note says so. The table also reports
`n`, `events` and `time_at_risk = sum (t - t0)`.

**Tests** (`sts test`, Methods and formulas). At each pooled failure time,
group g contributes `d_gj - n_gj d_j / n_j`; with a weight `W_j`

    u_g  = sum_j W_j (d_gj - n_gj d_j / n_j)
    V_gl = sum_j W_j^2 n_gj d_j (n_j - d_j) / (n_j (n_j - 1)) (1{g = l} - n_lj / n_j)
    chi2 = u' V^- u   on (number of groups - 1) degrees of freedom

| `test` | weight `W_j` |
| --- | --- |
| `"logrank"` (Mantel-Haenszel; the default reported in `attrs`) | 1 |
| `"wilcoxon"` (Wilcoxon-Breslow-Gehan) | `n_j` |
| `"tware"` (Tarone-Ware) | `sqrt(n_j)` |
| `"peto"` (Peto-Peto-Prentice) | `prod_{l <= j} (1 - d_l / (n_l + 1))` |
| `"fh"` (Fleming-Harrington, `fh_p`, `fh_q`) | `S(t_{j-1})^p (1 - S(t_{j-1}))^q`, pooled KM |

`test=None` computes all five. `strata=` sums `u` and `V` over strata (the
stratified tests; the Peto and Fleming-Harrington weights use the stratum's
pooled KM). `trend=True` adds `(a'u)^2 / a'Va` on 1 df with the group values as
scores (one numeric `by` column with at least three groups). The `expected`
table lists observed and expected failures per group. Delayed entry and
frequency weights (`weights=`, Stata's fweights) are handled throughout.

## `oe.stcox`: Cox proportional hazards regression

```python
cox = oe.stcox(data=df, time="t", failure="died", x=["age", "drug"], ties="efron",
               covariance="robust")
print(cox.summary())
cox.extra["hazard_ratios"]["drug"]          # exp(b), delta-method SE, exp-transformed CI
cox.tests["ph_global"], cox.metrics["concordance"]
```

**Model.** `h(t | x) = h0_s(t) exp(x'b + offset)` with an unspecified baseline
hazard per stratum. Coefficients are log hazard ratios; there is no constant.
Covariates that are constant within every stratum are omitted with a warning
(the screen demeans by stratum, then applies the Stata-style left-to-right
collinearity screen).

**Estimator.** Newton-Raphson from `b = 0` on the partial likelihood
(Stata [ST] stcox, Methods and formulas). With `S0_k = sum_{R_k} w exp(eta)`,
`S1_k = sum_{R_k} w exp(eta) x` over the risk set of failure time `tau_k`:

- `ties="breslow"` (Peto-Breslow, Stata's default):
  `ln L = sum_k [sum_{D_k} w_i eta_i - d_k ln S0_k]`.
- `ties="efron"`: the `c_k` tied failures see the denominators
  `S0_k - (r/c_k) sum_{D_k} exp(eta)`, `r = 0..c_k-1`.
- `ties="exactp"`: the exact partial likelihood. A failure time with one
  failure contributes its Breslow term; a tied failure time contributes the
  conditional-logit probability of its failure set within its risk set,
  evaluated with the discrete family's elementary-symmetric-function recursion
  (no enumeration). The expansion has `sum_{tied k} |R_k|` rows; more than
  5,000,000 rows or `2e7` recursion steps raise `exact_too_large` (use `efron`;
  see Performance).

Score and Hessian are analytic: `U = sum w delta x - sum_k d_k S1_k / S0_k` and
the `S2` term `sum_k a_k S2_k = X' diag(w exp(eta) A) X` with `A` the
accumulated `a_k` of each record, so no per-time k-by-k matrix is ever formed.
The linear predictor is shifted by its maximum before exponentiation and the
covariates are centred (the partial likelihood is invariant to both), which
keeps `exp(eta)` finite for covariates with large levels. Derivatives are
checked against numerical ones in the tests. A covariate that separates
failures from survivors makes the partial likelihood monotone; the fit then
stops with `separation_detected` instead of reporting a huge coefficient.

**Data features.** `strata` (separate risk sets per stratum), `entry` and
`id` (delayed entry and multiple records per subject: time-varying covariates
by episode splitting), `offset`. `tvc=["z"]` adds `z * g(t)` evaluated at each
failure time (`texp="identity"`: `g(t) = t`, Stata's default `texp(_t)`;
`texp="log"`: `ln t`), reported as terms `tvc:z` in equation `tvc` (the others
in `main`). It is implemented exactly as Stata describes the option: every
record is split at the failure times inside its interval, one row per record
and risk set (at most 5,000,000 rows, `tvc_too_large` otherwise).

**Weights** (Stata's rules: "Weights are not supported with efron and
exactp", [ST] stcox). `breslow` only: `fweight` (replication, `N = sum f`;
identical to duplicating rows), `iweight` (as given, in the weighted Breslow
likelihood of [ST] stcox Methods and formulas) and `pweight` (a robust
covariance is required; the log pseudolikelihood uses the weights normalized to
`w N / sum w` - see Uncertain conventions; coefficients and the robust
covariance do not depend on the scale of the weights). `efron` and `exactp`
take no weights (`unsupported_weights`).

**Covariance.**

| `covariance` | definition |
| --- | --- |
| `"nonrobust"` (default) | inverse of the negative Hessian of the partial likelihood |
| `"robust"` | Lin and Wei (1989): `N/(N-1) V W'W V` with the efficient score residuals `W` as scores; with an `id`, clustered on it (G/(G-1)) |
| `"cluster"` | residuals summed within `cluster`, `G/(G-1) V (sum_g W_g W_g') V` |

The efficient score residuals are
`W_i = delta_i (x_i - xbar_{k(i)}) - exp(eta_i) sum_{k: i in R_k} (d_k / S0_k) (x_i - xbar_k)`
(with Efron's down-weighted membership of the tied failures for `efron`); they
sum to the score. Stata applies `N/(N-1)` (respectively `G/(G-1)`) unless
`noadjust` is given ([ST] stcox, Options), and "stcox knew to specify
vce(cluster id) for us when we specified vce(robust)" when an id is stset;
OpenEconometrics does the same. `exactp` allows neither a robust/cluster covariance nor
`tvc` (Stata's restriction). Coefficient tests are z tests.

**Result.** `metrics`: `log_likelihood` (partial; log pseudolikelihood with
pweights), `log_likelihood_null` (`b = 0`), `aic`, `bic` (N = number of
observations, i.e. records; the number of failures is the alternative some
authors use), `df_model`, `n_subjects`, `n_failures`, `time_at_risk`,
`concordance`. `tests`:

- `model`: LR chi2(k) `2 (ll - ll_0)` under `nonrobust`, the Wald chi2(k) of all
  coefficients otherwise (Stata's `LR chi2` / `Wald chi2`);
- `score`: the score test of `b = 0`, `U(0)' I(0)^-1 U(0)` (for one binary
  covariate and no ties, the log-rank test);
- `ph_<term>` and `ph_global`: the Grambsch-Therneau tests of `estat phtest,
  detail`. With Schoenfeld residuals `r_i` of the failures, the time function
  `g` (`phtest="identity"`, Stata's default "analysis time itself"; `"log"`;
  `"km"`: 1 minus the Kaplan-Meier estimate; `"rank"`), `d` failures and the
  estimated covariance `V`:

      covariate p: chi2(1) = d (V u)_p^2 / (V_pp sum (g_i - gbar)^2),   u = sum (g_i - gbar) r_i
      global:      chi2(k) = u' V u d / sum (g_i - gbar)^2

  These are Stata's formulas with the scaled Schoenfeld residuals
  `r*_i = b + d V r_i`; each `ph_<term>` entry also carries `rho`, the
  correlation of `r*` with `g`. They are not computed after `tvc` (Stata's
  `estat phtest` is not allowed after `tvc()`). After `exactp` the residuals,
  PH tests and baseline functions use the Peto-Breslow formulas at the exact
  estimates, as Stata does ("If you specified breslow ..., exactm, or exactp,
  all predictions are carried out using the Peto-Breslow method").

`extra`: `hazard_ratios` (`exp(b)`, standard error `exp(b) se` and the interval
`exp(b +- z se)`, as Stata reports them), `baseline` (at covariates and offset
zero, per stratum, thinned evenly to at most 400 times; [ST] stcox
postestimation, Methods and formulas):

- `cumulative_hazard`: `H0(t) = sum_{tau_k <= t} d_k / S0_k` (Breslow; Stata's
  `predict basechazard`); after `ties="efron"` the increment is Efron's
  `sum_{r < c_k} 1 / (S0_k - (r/c_k) D0_k)` (Stata: after `efron` "all
  predictions are carried out using the Efron method");
- `survivor`: `exp(-H0(t))`;
- `survivor_kp`: Stata's `predict basesurv`, the Kalbfleisch-Prentice
  product-limit estimator `prod_{tau_k <= t} alpha_k`, where `alpha_k` solves
  `sum_{i in D_k} w_i u_i / (1 - alpha^{u_i}) = sum_{R_k} w u`
  (`u = exp(x'b + offset)`; Kalbfleisch and Prentice 2002, eq. 4.34). It is
  solved for all failure times at once by Newton's method in
  `psi = S0_k ln alpha`, which is scale free and, from Stata's starting value
  `psi = -d_k`, converges monotonically (the equation is increasing and convex
  in `psi`). At `b = 0` it is the Kaplan-Meier estimate and `H0` the
  Nelson-Aalen estimate, as Stata's manual states.

Further: `covariate_means`, `concordance` details (pairs, concordant, tied
predictions), `ties`, `notes`.

**Harrell's C** (`estat concordance`). A pair is usable when the shorter time
ends in failure, or when both times are equal and exactly one fails; it is
concordant when the earlier failure has the larger linear predictor;
`C = (E + T/2) / D` with tied predictions `T`, summed within strata. All pairs
are counted without enumerating them: for every failure, the later records
with a smaller predictor are counted bit by bit of the predictor's rank (one
sort and two binary searches per bit, O(n log^2 n)). As in Stata it is not
computed with weights, delayed entry, multiple records per subject or `tvc`.

## `oe.stcurve`: survivor and cumulative hazard after `stcox`

`oe.stcurve(result, data=None, at=None)` returns `H(t | x) = H0_s(t) exp(x'b)`,
`S(t | x) = exp(-H(t | x))` and the Kalbfleisch-Prentice
`S_KP(t | x) = S0_KP(t)^exp(x'b)` (column `survivor_kp`) as a table. Covariates not named in `at` are set
to their estimation-sample means (Stata's `stcurve` default); `at` names terms
(for a categorical predictor its indicator terms such as `"group[2]"`), so
`at={term: 0 for all terms}` is the baseline. Without `data` the curve uses the
bounded baseline table stored in the result; with the estimation `data` the
full step function is recomputed from the stored coefficients (a different
sample raises `data_mismatch`).

## `oe.streg`: parametric survival regression

```python
weib = oe.streg(data=df, time="t", failure="died", x=["age", "drug"])          # Weibull PH
aft = oe.streg(data=df, time="t", failure="died", x=["age", "drug"], metric="aft")
ln = oe.streg(data=df, time="t", failure="died", x=["age"], distribution="lognormal",
              ancillary=["drug"])
```

**Models** (Stata [ST] streg; `mu = x'b + offset`, ancillary `a = z'g`):

| `distribution` | metric | survivor function | ancillary term |
| --- | --- | --- | --- |
| `exponential` | PH (default) / AFT | `exp(-e^mu t)` / `exp(-e^-mu t)` | — |
| `weibull` | PH (default) / AFT | `exp(-e^mu t^p)` / `exp(-(e^-mu t)^p)`, `p = e^a` | `/ln_p` (extra: `p`, `1/p`) |
| `gompertz` | PH | `exp(-e^mu (e^(gamma t) - 1) / gamma)`, `gamma = a` | `/gamma` |
| `lognormal` | AFT | `1 - Phi((ln t - mu) / sigma)`, `sigma = e^a` | `/lnsigma` (extra: `sigma`) |
| `loglogistic` | AFT | `1 / (1 + exp((ln t - mu) / gamma))`, `gamma = e^a` | `/lngamma` (extra: `gamma`) |
| `ggamma` | AFT | `1 - I(g, u)` (kappa > 0), `I(g, u)` (kappa < 0), lognormal at kappa = 0; `g = kappa^-2`, `z = sign(kappa)(ln t - mu)/sigma`, `u = g exp(abs(kappa) z)` | `/lnsigma` (extra: `sigma`), `/kappa` |

`metric="aft"` is Stata's `time` option; asking for a metric a distribution
does not have raises `invalid_spec`. PH coefficients are log hazard ratios
(`extra["hazard_ratios"]`), AFT coefficients log time ratios
(`extra["time_ratios"]`); for the Weibull with a constant ancillary parameter
`b_AFT = -b_PH / p`.

**Estimator.** Maximum likelihood of
`sum_i w_i [delta_i ln h(t_i) + ln S(t_i) - ln S(t0_i)]` (the time-scale log
likelihood) with right censoring and delayed entry in the likelihood.

**Reported log likelihood (Stata's scale).** Stata reports the log likelihood
of the log survival time, whose density is `t f(t)`:
`ln L_Stata = ln L_time + sum_i w_i delta_i ln t_i`. The added term does not
depend on the parameters, so estimates, standard errors and likelihood-ratio
statistics are the same on either scale, but `log_likelihood`, `aic`, `bic` and
`null_log_likelihood` are Stata's numbers. This was established on the `kva`
example of [ST] streg, which OpenEconometrics reproduces to every printed digit
(Weibull PH: `ll = 5.6934189`, constant-only `-9.4408286`, `LR chi2(2) = 30.27`,
`load .4695753 (.1177884)`, `/ln_p 2.051552 (.2317074)`; the time-scale value
would be `-44.39679`). The first and second derivatives of each observation's
contribution with respect to its two indices `(mu_i, a_i)` are written out in
closed form (`parametric.py`; the Gompertz `(e^(gamma t) - 1)/gamma` and its
derivatives switch to their power series for `|gamma t| < 0.5`, the lognormal
uses `log Phi` and the inverse Mills ratio, the loglogistic the softplus), and
assembled into the score `X'(w l_mu)` and the Hessian blocks
`X' diag(w l_mu,mu) X`, ... Newton-Raphson starts from the constant-only
model; regressors of both equations are centred at their means for the
iteration and mapped back exactly.

**Generalized gamma.** `I(g, u)` is the regularized incomplete gamma
function. OpenEconometrics evaluates it on tensors (`special.py`): the power series of
`P` below `u = g + 1` and the continued fraction of `Q` above it, and for
`g > 1e4` (`abs(kappa) < 0.01`) Temme's uniform asymptotic expansion with two
terms (DLMF 8.12), each tail on the log scale where it is small. The density is
written as `ln f = -ln sigma - ln t - ln(2 pi)/2 - delta(g) - g (exp(s) - 1 - s)`,
`s = abs(kappa) z`, `delta` the Stirling remainder of `ln Gamma`, which is free of
cancellation and tends to the lognormal density as kappa -> 0; at kappa = 1 the
model is the Weibull AFT model and the likelihoods agree to rounding. The
per-observation derivatives with respect to `(mu_i, ln sigma_i)` are analytic:
with `s = kappa (ln t - mu) / sigma`, `d ln f / ds = -g (e^s - 1)`,
`d2 ln f / ds2 = -g e^s`, and for the tail `T` that forms `S` (Q for kappa > 0,
P for kappa < 0), with `m = u^g e^-u / (Gamma(g) T)` and `c = -1` (kappa > 0) or
`+1`, `d ln S / ds = c m` and `d2 ln S / ds2 = c m (g - u - c m)`, `ln m` being
evaluated free of cancellation so that the lognormal limit stays finite. Only
the derivatives in kappa - the shape of the incomplete gamma function, whose
derivative has no closed form - are five-point central differences (step 1e-3,
fourth-order accurate) of the value and of the analytic `(mu, ln sigma)`
derivatives: four extra likelihood evaluations per Newton iteration. This is the
documented numerical-derivative path for this model and the result records it
in `provenance["derivatives"]`. A generalized gamma likelihood without an
interior maximum (kappa running off) stops with `boundary_solution`. `ancillary` columns model
`ln sigma`; kappa is a single parameter (Stata's `anc2()` is not offered).

**Ancillary equation and strata.** `ancillary=[...]` models the ancillary
parameter as `z'g` (with a constant), reported as terms `ln_p:<name>` in
equation `ln_p` (`gamma`, `lnsigma`, `lngamma`); without it the single term is
`/ln_p` etc. `strata="s"` (one column) adds indicators of `s` to the main and
the ancillary equation, which is how Stata defines `strata()` for streg.

**Covariance.** `nonrobust` (observed information), `opg`, `robust` (N/(N-1)
sandwich; with an `id`, clustered on it, as "streg knows to specify
vce(cluster clustvar) if you specify vce(robust)"), `cluster` (G/(G-1)), all
through `core.ml_covariance`. Weights: `fweight`, `iweight`, `pweight` (robust
by default).

**Result.** `metrics`: `log_likelihood` (Stata's scale, above), `aic`, `bic`
(N = observations), `df_model`, `n_subjects`, `n_failures`, `time_at_risk`.
`tests["model"]`: Stata's `ml` model test - every main-equation coefficient
except the constant, including the stratum indicators of `strata=`, against the
model with a constant-only main equation and the ancillary equation as
specified (its `ancillary` and stratum terms stay in the null model). LR chi2
under `nonrobust`, Wald chi2 under robust/cluster. Verified on [ST] streg
example 8 (`streg age, strata(drug)` with three drug levels reports
`LR chi2(3)`) and example 9 (`ancillary(i.drug)` reports `LR chi2(1)`).
`extra`: hazard or time ratios (every main-equation term but the constant),
transformed ancillary parameters with delta-method standard errors and
transformed intervals (as Stata prints `p`, `1/p`, `sigma`, `gamma`),
`null_log_likelihood`. A ratio that overflows is reported as missing.

`x` may be omitted: `oe.streg(data=df, time="t", failure="d")` fits the
constant-only model (Stata's `streg, distribution(...)`), with a model test of
0 df. A covariate, category or stratum without failures makes the likelihood
monotone and stops with `separation_detected`; an ancillary parameter running
to the boundary with `boundary_solution`.

## `oe.ltable`: actuarial life tables

`oe.ltable(df, "t", failure="d", by="g", intervals=5)`. Intervals: `None`
(unit width, Stata's default), a width, or a list of cutpoints (0 is
prepended; times beyond the last cutpoint form an open-ended interval whose
hazard and density are undefined). Records with time 0 fall into the first
interval. Formulas (Stata [ST] ltable, Methods and formulas; SPSS SURVIVAL):

    n_j = N_j - m_j / 2            (noadjust=True: n_j = N_j)
    q_j = d_j / n_j,  S_j = prod_{k <= j} (1 - q_k),  Greenwood SE, ln(-ln S) interval
    hazard  = q_j / ((1 - q_j/2) w_j),  SE = hazard sqrt((1 - (w_j hazard / 2)^2) / d_j)
    density = S_{j-1} q_j / w_j,        SE = density sqrt(sum_{k<j} q_k/(n_k p_k) + p_j/(n_j q_j))

With `noadjust=True` the hazard is `q_j / w_j` with `SE = hazard / sqrt(d_j)`
and the chi-square interval `hazard chi2_{2d, alpha/2 ; 1-alpha/2} / (2d)`.
Columns: `interval_start, interval_end, entering, deaths, lost, at_risk,
death_probability, survival, std_error, ci_low, ci_high, hazard` (+ SE and
interval), `density` (+ SE). Intervals without deaths or losses are omitted.
With `by`, `tests` holds the likelihood-ratio test of homogeneity (Lawless 2003,
p. 155: `2 {(sum d_g) ln(sum T_g / sum d_g) - sum d_g ln(T_g / d_g)}` with `T_g`
the total time of group g, G - 1 df) and the log-rank test of `sts test` on
the individual times.

## Performance

Synthetic data, one million records, ten covariates, Weibull times with about
55% failures and times rounded to 1e-3 (many ties), Apple silicon CPU
(measured in the verification pass):

| call | time |
| --- | --- |
| `oe.sts(df, "t", failure="d")` | 0.8 s |
| `oe.sts(..., by=g)` with all five tests | 0.3 s |
| `oe.ltable(..., intervals=0.1, by=g)` | 0.2 s |
| `oe.stcox(..., ties="breslow", concordance=False)` | 1.0 s |
| `oe.stcox(..., ties="efron", concordance=False)` | 1.0 s |
| `oe.stcox(...)` with Harrell's C and the PH tests | 1.8 s |
| `oe.stcox(..., ties="efron", covariance="robust")` | 1.1 s |
| `oe.stcox(..., strata=g, covariance="cluster")` (1,000 clusters) | 0.8 s |
| `oe.streg(...)` exponential / Weibull / Gompertz / lognormal / loglogistic | 0.4 / 0.7 / 1.1 / 0.7 / 0.7 s |
| `oe.streg(..., covariance="robust")` (Weibull) | 1.0 s |
| `oe.streg(..., distribution="ggamma")` | 6.5 s (incomplete gamma; numerical kappa derivatives) |
| `oe.stcox(..., tvc=[z])`, 20,000 records split into 2.2 million rows | 0.9 s |

`ties="exactp"` is expensive by nature: its recursion runs over the positions
of every tied risk set (10.7 s for 3,000 records whose times are rounded to two
decimals). Problems needing more than `2e7` recursion steps
(`sum_{tied k} |R_k| min(c_k, |R_k| - c_k)`) or more than 5,000,000 expanded rows are
refused with `exact_too_large`; `ties="efron"` is the practical alternative.

## Verification

`tests/test_econ_survival_oracle.py` checks every estimator against independent
derivations, `tests/test_econ_survival_adversarial.py` the failure contract:

- **Published Stata output.** The `kva` generator data of [ST] stcox and
  [ST] streg (12 observations): `stcox load bearings` reproduces
  `ll = -8.577853`, `LR chi2(2) = 23.39`, `b = .4229578 (.1433485)`,
  `-2.754461 (1.173115)` and the hazard ratios; `streg load bearings,
  distribution(weibull)` with and without `time` reproduces every printed
  coefficient, standard error, `p`, `1/p` (with intervals), the log likelihood
  and the LR test.
- **Cox.** Explicit loops over the failure times for the Breslow, Efron and
  exact partial likelihoods (subset enumeration), maximized by SciPy with
  numerically differentiated Hessians, with strata, delayed entry, offsets,
  `tvc` (likelihood evaluated at each failure time) and weights; statsmodels
  `PHReg`; efficient score residuals written from Stata's formula (Efron:
  Therneau-Grambsch averaging) for the robust and cluster sandwiches; Schoenfeld
  residuals and the PH-test formulas of [ST] stcox PH-assumption tests; Harrell's
  C over all pairs (including tied predictions); the Breslow and Efron baseline
  hazards, the Kalbfleisch-Prentice survivor (root found by `brentq` per failure
  time; Kaplan-Meier and Nelson-Aalen at `b = 0`) and `stcurve`; the Breslow
  predictions after `exactp`; fweights equal duplicated rows; invariance to row
  order, covariate rescaling and time units.
- **streg.** Every distribution and metric against a brute-force maximization
  of the likelihood written from Stata's parameterizations (SciPy's regularized
  incomplete gamma for the generalized gamma), on Stata's log-time scale, with
  delayed entry; OPG, robust and cluster covariances from numerically
  differentiated per-observation scores; iweights, pweights, offsets; the
  stratified model test; fweights equal duplicated rows.
- **sts / ltable.** Explicit loops for the Kaplan-Meier, Greenwood, all three
  interval types, Nelson-Aalen, percentiles and their limits, the restricted mean
  and its standard error, the Peto-Peto-Prentice and Fleming-Harrington tests
  with delayed entry and the trend test; statsmodels `survdiff` for the
  log-rank, Gehan-Breslow, Tarone-Ware and Fleming-Harrington tests (with strata,
  and with delayed entry except Fleming-Harrington, whose statsmodels weights are
  undefined there); actuarial loops for the life table (adjusted and
  `noadjust`, width and cutpoint intervals) and the homogeneity tests.

Undefined table entries (the interval of `S` before the first failure or after
`S` reaches 0, a median that is never reached, the hazard of an open-ended last
interval) are missing values (NaN) in the `sts` and `ltable` tables, as Stata
prints a dot. Model results never contain NaN or infinity: a hazard or time
ratio that overflows is reported as missing (`None`).

## Limitations

- Not implemented: Stata's `exactm` ties, shared frailty (`shared()`),
  parametric frailty, `stcrreg`, `stintreg`, Gönen-Heller concordance and a
  standard error of Harrell's C, `stcurve` after `streg`, `anc2()` (covariates
  for the generalized gamma's kappa), the SPSS `COXREG` stepwise methods.
- `stcox` baseline functions and PH tests are not computed after `tvc` (they
  need the split data); Stata also restricts predictions after `tvc()`.
- Harrell's C and the PH tests are not computed in the situations listed above.
- `sts` and `ltable` take frequency weights only.

## Uncertain conventions

These follow the textbook definition because Stata's documentation does not
settle them; `provenance["stata_parity_validated"]` is `False` for every result.

- `phtest="km"` uses 1 minus the right-continuous Kaplan-Meier estimate at each
  failure time (Stata: "1 minus the Kaplan-Meier product-limit estimate"; R's
  `cox.zph` uses the left-continuous value). `phtest="rank"` uses average ranks
  of the failure times. With weights, the sums of the PH tests are weighted by
  the estimation weights and `d` is the weighted number of failures. The PH
  tests use the reported covariance (robust when requested), as Stata's formula
  is written with `Var(b)`.
- `stcox` with `pweight`s: the log pseudolikelihood (and AIC/BIC) uses the
  weights normalized to `w N / sum w`; coefficients and the robust covariance
  are invariant to that scale. Stata's documentation does not state whether
  stcox normalizes pweights in the reported log pseudolikelihood.
- `stcox` baseline hazard after `ties="efron"`: Efron's averaged risk-set
  increments, the analogue that Stata's sentence on Efron predictions implies
  (and R's `survfit.coxph` uses); Stata's manual writes only the Breslow
  formula. The Kalbfleisch-Prentice `survivor_kp` is computed from the same
  equation for every ties method. With weights, both use the weighted risk
  sums (Stata's formulas are written unweighted).
- `stcurve` without `at` evaluates at the covariate means of the estimation
  sample; whether Stata's `stcurve, survival` after `stcox` raises
  `basesurv` (our `survivor_kp`) or uses `exp(-H)` (our `survivor`) is not
  documented, so both are reported.
- `ltable` hazard confidence limits under the actuarial adjustment are not
  given by a formula in Stata's manual; OpenEconometrics uses `hazard +- z SE`, with
  the lower limit truncated at 0.
- The stcox separation check: a non-converged Newton run whose linear predictor
  moves by more than 10 across one standard deviation of a covariate
  (a hazard ratio beyond `exp(10)`) is reported as `separation_detected`; the
  same rule (and `|kappa| > 5` for the generalized gamma) applies to streg.
- `n_failures` and `time_at_risk` are frequency-weighted for fweights and
  unweighted for iweights and pweights.
- Concordance within strata (Stata's formula sums `E_k`, `T_k`, `D_k` over strata).
