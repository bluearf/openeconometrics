# Mixed-model extensions (MARKET-152)

This page documents four bounded additions to the mixed family. Each runs on
CPU float64 Torch; R, SciPy and NumPy appear only in the reference generator
and the tests. None of them claims licensed Stata/SAS/SPSS parity, and
`provenance["stata_parity_validated"]` stays `False`.

| Stage | Public API | Kind | Independent reference |
| --- | --- | --- | --- |
| 1 | `oe.mixed_contrast` | saved-result procedure (`EXPORTS`) | lme4 2.0.6 + lmerTest 3.2.1 + pbkrtest 0.5.5; dense NumPy KR formulas |
| 2 | `oe.mixedresid` | registered estimator `mixedresid` | nlme 3.1-169 `lme`; dense NumPy likelihood and finite-difference Hessian |
| 3 | `oe.meologit`, `oe.meoprobit` | registered estimators | ordinal 2026.7.26 `clmm` (nAGQ 3 and 10); NumPy/SciPy quadrature |
| 4 | `oe.metobit` | registered estimator | explicit NumPy/SciPy quadrature likelihood maximized by SciPy; exact Gaussian case |

Fixtures are produced by
[`scripts/generate_mixed_extension_oracles.py`](../../scripts/generate_mixed_extension_oracles.py)
(with its R companion `generate_mixed_extension_oracles.R`) and stored under
`tests/fixtures/mixed_extensions/`. Tests in
`tests/test_econ_mixed_extensions.py` read the JSON and need no R.

## 1. Kenward–Roger and Satterthwaite fixed-effect inference

```python
fit = oe.mixed(data=df, y="Reaction", x=["Days"], group="Subject",
               random=["Days"], covstructure="unstructured", method="reml")
kr = oe.mixed_contrast(fit, data=df, contrast={"Days": 1})                 # t test
joint = oe.mixed_contrast(fit, data=df, contrast=[{"Intercept": 1}, {"Days": 1}],
                          null=[250, 9], ddf="kenward-roger")             # F test
sat = oe.mixed_contrast(fit, data=df, contrast={"Days": 1}, ddf="satterthwaite")
```

`contrast` is a dict of fixed-term weights (one row), a list of such dicts or
a DataFrame whose columns are fixed terms (rows are hypotheses). Rows must be
full row rank; `null` is a scalar or one value per row. The fit is replayed at
its saved estimates on the matched original data (row order and values are
hash-checked); nothing is refitted.

### Covariance coordinates

The saved `mixed` marginal covariance is linear in the reported variance
parameters, `V = sum_r s_r G_r`: the top-level intercept variance, the
entries of the lowest-level random-effect covariance as reported
(variances, covariances; `identity` is one common variance and
`exchangeable` one variance plus one covariance) and the residual variance.
All second derivatives of V vanish, so the Kenward–Roger `R_ij` term is zero
and the adjustment does not depend on a nonlinear choice of coordinates. This
is pbkrtest's (and SAS's) convention.

### Kenward–Roger (REML only)

With `Phi = (X'V^-1X)^-1`, `P_r = -X'V^-1G_rV^-1X`,
`Q_rs = X'V^-1G_rV^-1G_sV^-1X` and the inverse expected REML information
`W = [tr(P G_r P G_s)/2]^-1`, the adjusted covariance is
`Phi_A = Phi + 2 Phi [sum_rs W_rs (Q_rs - P_r Phi P_s)] Phi`
(Kenward and Roger 1997, eq. 2–3). For a q-row contrast, the moments `A1`,
`A2`, `B`, `g`, `c1..c3`, `E*`, `V*`, `rho` give the denominator df
`m = 4 + (q+2)/(q rho - 1)` and scale `lambda = m(1 - A2/q)/(m - 2)`; the
joint test is `lambda F ~ F(q, m)` with `F = (Lb - l0)'(L Phi_A L')^-1(Lb - l0)/q`
(section 4 of the paper). Each row is additionally reported as a t test with
`sqrt(l'Phi_A l)` and its own KR df (the q = 1 moments, no scaling) — the
lmerTest `contest1D(ddf = "Kenward-Roger")` convention. Formulas follow
pbkrtest 0.5.5 `vcovAdj_internal`, `.KR_adjust` and `Lb_ddf`
([source](https://github.com/cran/pbkrtest)),
including `lambda = 1` when `|m - 2| < 0.01`. A non-positive or non-finite df
is refused (`invalid_degrees_of_freedom`). An ML fit is refused with
`unsupported_spec`: unlike pbkrtest, OpenEconometrics never refits silently.

### Satterthwaite (ML or REML)

`df = 2 (l'Phi l)^2 / (g' W_O g)` with `g_r = -l'Phi P_r Phi l` and `W_O` the
inverse *observed* information of the (restricted) likelihood with b
profiled out, `a_r'P a_s - tr(P G_r P G_s)/2` (REML) or
`a_r'P a_s - tr(V^-1G_rV^-1G_s)/2` (ML), `a_r = G_r V^-1 r`. At the fitted
optimum this equals lmerTest's numerical deviance Hessian in its own
coordinates. Joint tests decompose `L Phi L'` by its eigenvectors and combine
the component dfs as lmerTest `contestMD` / `get_Fstat_ddf`
(Fai and Cornelius 1996; df 2 if any component df is at most 2).
`oe.mixed_satterthwaite` (scalar, log-scale numerical derivatives) is
retained; the two agree to the tolerance of its numerical Hessian.

### Saved-state checks and output

Before any inference the procedure verifies that GLS at the saved variance
parameters reproduces the saved fixed estimates and fixed covariance, that
the saved parameters are a stationary point (scaled score below `1e-6`) and
that the saved delta-method variance covariance agrees with the analytic
observed information. Tables: `contrasts` (estimate, null, adjusted and
unadjusted standard errors, t, df, p, interval), `joint_test` (F, unscaled F,
scale, numerator and denominator df, p), `fixed_covariance` (`Phi_A` for KR,
`Phi` for Satterthwaite), `variance_parameters` (linear coordinates and their
REML/ML score) and `variance_parameter_covariance` (`W` or `W_O`), plus the
JSON `settings` row set. `oe.summary_state` / `oe.restore_summary` round-trip
every table.

Domain: unweighted model-based (`nonrobust`) `mixed` fits with one or two
nested levels and any `covstructure`, numeric or treatment-coded categorical
fixed effects, at most 8,192 rows and top-level groups of at most 512 rows
(dense per-group derivatives on padded batches with a planned workspace and
`max_work`). Robust/cluster covariance, frequency weights, `mixedflex`,
GLMMs and nonlinear covariance coordinates (which would need `R_ij`) are
refused.

### Evidence

| Case | Model | Reference | Max relative deviation |
| --- | --- | --- | --- |
| `sleepstudy_reml` | lme4 published data, unstructured slope | lme4/lmerTest/pbkrtest | pooled below |
| `unbalanced_reml` | 18 unbalanced groups, independent slope, 3-level factor | lme4/lmerTest/pbkrtest | pooled below |
| `unbalanced_ml` | the same data by ML (Satterthwaite only) | lme4/lmerTest | pooled below |
| `nested_reml` | two nested random intercepts | lme4/lmerTest/pbkrtest | pooled below |
| exchangeable, identity, unstructured (2 slopes) | synthetic | dense N×N NumPy KR (1997) and finite-difference Satterthwaite | `1e-9` (KR), `2e-5` (finite differences) |

Across the R cases the largest relative deviations were: fixed effects
`2.4e-8`, standard errors `2.4e-8`, KR df `1.1e-8`, KR adjusted covariance
`4.7e-7`, KR F `4.3e-8`, Satterthwaite df `8.9e-8`, Satterthwaite F `4.2e-8`
(the residual differences come from lme4's optimizer tolerance and
lmerTest's numerical Hessian). Test tolerances are `2e-6` (df, F),
`1e-6` (SE) and `3e-6` (`Phi_A`).

References: Kenward, M. G. and Roger, J. H. (1997), *Biometrics* 53, 983–997;
Halekoh, U. and Højsgaard, S. (2014), *J. Stat. Software* 59(9);
Kuznetsova, A., Brockhoff, P. B. and Christensen, R. H. B. (2017),
*J. Stat. Software* 82(13); Fai, A. H.-T. and Cornelius, P. L. (1996),
*J. Statist. Comput. Simul.* 54, 363–378.

## 2. Random effects with a structured residual covariance: `mixedresid`

```python
fit = oe.mixedresid(data=df, y="y", x=["x", "t"], group="id",
                    residual="ar1", occasion="t", stratum="arm", method="reml")
# nlme: lme(y ~ x + t, random = ~ 1 | id, correlation = corAR1(form = ~ t | id),
#           weights = varIdent(form = ~ 1 | arm), method = "REML")
# Stata: mixed y x t || id:, residuals(ar 1, t(t) by(arm)) reml   (by() differs: see below)
```

Model: `y_i = X_i b + Z_i u_i + e_i`, `u_i ~ N(0, G)` with a random intercept
and/or random slopes (`covstructure` `independent` or `unstructured`), and
`Var(e_i) = s2 D_i C_i D_i` where

* `residual="ar1"`: `C_ab = rho^|t_a - t_b|` on the integer `occasion` column,
  the actual gap as exponent (nlme `corAR1(form = ~ t | g)`); observed
  within-block lags must have greatest common divisor one, occasions must be
  unique within a block and gaps at most 512;
* `residual="exchangeable"`: `C_ab = rho` (nlme `corCompSymm`), negative values
  allowed down to `-1/(m - 1)` with m the largest block;
* `residual="independent"`: `C = I`;
* `within`: optional label column nested in `group` whose cells are the
  correlation blocks (nlme `form = ~ 1 | g/cell`); otherwise the groups are;
* `stratum`: optional label column with one residual standard-deviation
  ratio per level, `D = diag(delta)`, the first level in sorted label order
  being the reference (nlme `varIdent(form = ~ 1 | stratum)`). Stata's
  `residuals(..., by())` estimates a separate correlation per level as well;
  only the variance is stratified here.

A random intercept together with an exchangeable correlation on the same
blocks is not identified (both add a constant within-group covariance) and
is refused with `unidentified_covariance`; use nested `within` cells, drop
the random intercept, or use AR(1).

Estimation profiles b by GLS and maximizes the ML or REML (Harville, without
the `log|X'X|` constant, as nlme and `mixed`) log likelihood in the
log-Cholesky coordinates of G, `ln s`, `ln delta` (non-reference strata) and
`tanh`/shifted-logistic coordinates of rho, with exact Torch derivatives
(BFGS with a Newton polish), on dense padded per-group covariance matrices
(groups of at most 256 rows; planned workspace; `max_work`).

Inference (asymptotic z): **ML** reports the inverse of the full joint
observed information of b and the covariance parameters, cross terms
retained; **REML** reports the GLS covariance `(X'V^-1X)^-1` for b and the
inverse observed restricted information for the covariance parameters
(block diagonal). Reported parameters are the fixed effects, `/var(·[g])`
and `/cov(·,·[g])` of the random effects, `/var(Residual)` or one
`/var(Residual[stratum=level])` per stratum (`s2 delta^2`), and
`/rho(Residual)`, with delta-method standard errors and log-scale intervals
for variances. `extra` keeps G, rho, the strata, the conditional GLS fixed
covariance (what nlme's `vcov` reports for both criteria), and the
likelihood coordinates with their covariance. A variance at zero, a
perfectly correlated G, rho at the edge of its range or a vanishing
stratum ratio raise `boundary_solution`. Small-sample (KR/Satterthwaite)
df are not provided for this estimator; prediction is the population mean
`X b` only.

### Evidence

Five nlme 3.1-169 `lme` fits (REML/ML; random intercept, independent and
unstructured random slopes; AR(1) with gaps, nested exchangeable, varIdent)
on seeded unbalanced data. Largest deviations: fixed effects `1.4e-7`
(relative), log likelihood `1.3e-11` (absolute), G `2.5e-6`, conditional
fixed covariance `2.9e-6`, rho `5.5e-7` (absolute); these reflect nlme's
optimizer tolerance. The full covariance matches the inverse
finite-difference Hessian of an independent dense NumPy likelihood in the
reported coordinates within `2e-4` (relative) for every case, including the
ML Schur-complement identity for the variance block. nlme's `apVar` is an
`fdHess` finite-difference approximation; mapped to its coordinates (log sd,
`log((1+rho)/(1-rho))`, `log delta`, `log sigma`) our covariance agrees
within 0.52% on the correlation scale.

## Shared random-intercept quadrature (stages 3 and 4)

`meologit`, `meoprobit` and `metobit` integrate one normal random intercept
`u_g = s v_g`, `v_g ~ N(0, 1)`, by Gauss–Hermite quadrature
(`mixed/agq.py`):

* `intmethod="mcaghermite"` (default): mode-curvature adaptive quadrature
  (Liu and Pierce 1994). For every group the posterior mode `v_g` of
  `h_g(v) = sum_i ln f(y_i | eta_i + s v) + ln phi(v)` and its curvature
  `c_g = -h_g''(v_g)` define nodes `v_g + sqrt(2/c_g) x_k`. The modes are
  recomputed at every parameter value, so the approximation is one smooth
  function of the parameters — the convention of `ordinal::clmm` and
  `lme4::glmer` with `nAGQ > 1`; Stata instead holds adaptive parameters
  fixed between Newton runs, which differs at the order of the quadrature
  error. The modes come from damped Newton steps without a derivative tape,
  then two Newton steps on the tape; because the Newton map has zero
  derivative at its fixed point, the taped mode carries exact first and second
  parameter derivatives, and the gradient and Hessian are exact Torch
  derivatives of the quadrature likelihood.
* `intmethod="ghermite"`: the non-adaptive rule, nodes `sqrt(2) x_k`
  (Stata's `ghermite`). R `clmm`'s `nAGQ < 0` uses nodes `x_k` instead, a
  different rule, so it is not used as a reference.

`intpoints` (1–64, default 12) sets the points; `extra["quadrature_check"]`
re-evaluates the likelihood at twice the points (at most 128) at the
estimate and reports the difference — a sensitivity measurement, not a
refit. The densities must be log-concave in the linear predictor (true for
ordered logit/probit and the censored normal), which makes every group's
mode unique. Coordinates are optimized by BFGS with a Newton polish; the
covariance is the delta-method transform of the inverse full observed
information (all fixed, threshold and variance cross terms). A random
intercept variance at zero raises `boundary_solution`. Only the
`nonrobust` covariance is offered; weights, random slopes and robust or
cluster covariance are not part of these stages.

## 3. Ordinal GLMM: `meologit`, `meoprobit`

```python
fit = oe.meologit(data=df, y="rating", x=["x", "z"], group="judge", intpoints=10)
# R: clmm(factor(rating, ordered = TRUE) ~ x + z + (1 | judge), nAGQ = 10)
# Stata: meologit rating x z || judge:, intmethod(mcaghermite) intpoints(10)
```

`Pr(y <= k | u) = F(cut_k - x'b - u)`, logistic or standard normal F; the
`J - 1` cutpoints replace the constant (numeric outcomes ordered by value or
an ordered `Categorical`, as `oe.ologit`). Likelihood coordinates: b,
`cut_1`, `ln(cut_k - cut_{k-1})` (ordering enforced) and `ln s`. Reported:
the slopes, `/cut1`..`/cut{J-1}` and `/var(_cons[group])`; `extra` has the
categories, the quadrature check, a posterior-mean summary and the
likelihood coordinates with their covariance.

Evidence: three `ordinal::clmm` fits (ucminf, gradient tolerance `1e-10`)
on seeded unbalanced data, 40 groups, four categories: logit and probit
with nAGQ 10 and logit with nAGQ 3 (where the adaptive placement itself
matters). Largest deviations with 10 points: parameters `3.0e-8`
(absolute), log likelihood `6e-14`; covariance within `2e-5` on the
correlation scale and `5e-6` (relative) on the diagonal, against clmm's
numerical Hessian. With 3 points clmm stops at max |gradient| `3e-5`;
parameters agree within `3.1e-6` and our likelihood is `1.7e-10` higher.
An independent SciPy implementation (Brent mode, Richardson curvature)
reproduces the adaptive log likelihood within `1e-8` with a zero
finite-difference gradient; for `ghermite` it reproduces the likelihood and
the full covariance (finite-difference Hessian, `1e-4`).

## 4. Censored GLMM (random-intercept Tobit): `metobit`

```python
fit = oe.metobit(data=df, y="hours", x=["x"], group="person", ll=0, ul=2.5)
# Stata: metobit hours x || person:, ll(0) ul(2.5)    (xttobit hours x, re ll(0) ul(2.5))
```

`y* = x'b + u + e` with normal u and e; rows with `y <= ll` are
left-censored, rows with `y >= ul` right-censored (constant limits, Stata's
rule). Row contributions are `phi((y - eta)/s_e)/s_e`, `Phi((ll - eta)/s_e)`
and `Phi((eta - ul)/s_e)`. Coordinates b, `ln s_e`, `ln s_u`; reported: the
fixed effects, `/var(_cons[group])` and `/var(Residual)`; metrics include the
censoring counts. `fitted` is the latent linear prediction `x'b`.

Evidence: no R Tobit GLMM oracle is available here (glmmTMB is not
installed), so the reference is an explicit NumPy/SciPy quadrature
likelihood: a 60-point non-adaptive likelihood maximized by SciPy BFGS agrees
with the 12-point adaptive fit within `1.8e-6` (relative) and its
finite-difference Hessian with our covariance within `2.3e-4`; at 40
non-adaptive points the likelihood and full covariance coincide with the
NumPy rule (`1e-9`, `1e-4`). Without censored rows the integrand is Gaussian
in u, adaptive quadrature with one point is exact, and `metobit` reproduces
`mixed` (ML) estimates and log likelihood and the inverse finite-difference
Hessian of the closed-form Gaussian likelihood; its fixed-effect standard
errors differ from `mixed`'s only because `mixed` reports Stata's
block-diagonal covariance while `metobit` keeps the full joint information.

References: Liu, Q. and Pierce, D. A. (1994), *Biometrika* 81, 624–629;
Christensen, R. H. B., *ordinal* (CRAN); Stata *ME* manual, `meologit`,
`meoprobit`, `metobit`.
