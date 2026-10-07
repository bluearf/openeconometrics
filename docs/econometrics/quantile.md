# Quantile, robust and nonlinear regression

This family covers regressions that do not minimize a sum of squared residuals
of a linear model:

| OpenEconometrics | Stata | SPSS / EViews | What it estimates |
|---|---|---|---|
| `oe.qreg` | `qreg` | SPSS Quantile Regression, EViews `qreg` | one conditional quantile, analytic standard errors |
| `oe.bsqreg` | `bsqreg` | | one conditional quantile, bootstrap standard errors |
| `oe.sqreg` | `sqreg` | | several quantiles with their joint bootstrap covariance |
| `oe.iqreg` | `iqreg` | | the difference between two quantile regressions |
| `oe.rreg` | `rreg` | | Huber/biweight robust regression |
| `oe.nl` | `nl` | SPSS `NLR`, EViews nonlinear LS | nonlinear least squares for a formula |

Everything is implemented in OpenEconometrics on float64 tensors
(`src/openecon/econometrics/quantile/`); no estimation library runs at fit
time. Results are ordinary `ResultBundle` objects: `summary()`, `to_latex()`,
`coefficients`, `covariance_matrix`, `metrics`, `tests`, `extra`, `warnings`.
All coefficient tests in this family use Student t with N − K degrees of
freedom, as Stata prints (G − 1 for the cluster covariance of `nl`).

`provenance["stata_parity_validated"]` stays `False` because no Stata run is
part of the build. The test suite nevertheless reproduces Stata output where
it was available (see [Validation](#validation)): the `qreg` and `rreg`
examples of the Stata manual on auto.dta and Stata's kernel-density results
for Engel's data. Conventions that could not be checked are listed at the end.

## qreg: quantile regression

### Model and estimator

The τ-th conditional quantile of `y` is linear in the regressors,
Q_τ(y | x) = x'β(τ). The estimator minimizes the check loss

```
sum_i w_i ρ_τ(y_i − x_i'b),      ρ_τ(u) = u (τ − 1[u < 0]),
```

which for τ = 0.5 is least absolute deviations (median regression). The
problem is a linear program. OpenEconometrics solves it with the Frisch–Newton
interior-point method of Portnoy and Koenker (1997) applied to the dual
(max y'a subject to X'a = (1 − τ)X'1, 0 ≤ a ≤ 1) with Mehrotra
predictor–corrector steps. Each iteration needs one K×K matrix X'DX and its
Cholesky factor; nothing of size N×N is ever formed, so a million rows with
ten regressors take one to two seconds.

An interior-point iterate only approaches the solution. Once the duality gap
is small the K observations with the smallest residuals are taken as a trial
basis, the hyperplane through them is computed exactly and its optimality
certificate (the dual feasibility conditions of the linear program) is
checked. When it holds, the reported coefficients are *the* vertex solution: they
interpolate K observations exactly, as the simplex solvers of Stata, SPSS and
EViews do. If the certificate fails after the interior point has converged, a
few Barrodale–Roberts-style simplex pivots finish the job (continuous data
need none). The linear program is solved for the least-squares residuals of
the outcome, an exact reparametrization that keeps the solver accurate when
the outcome has a large level and little variation.

Discrete outcomes, dummy-only designs and duplicated rows produce *ties*: more
than K observations on the fitted hyperplane, and often several coefficient
vectors with the same minimal loss. A residual below 10⁻¹⁰ of the typical
least-squares residual counts as a tie. Such degenerate vertices are certified
with the interior-point dual solution itself: on the tied rows it lies inside
its bounds, and after a least-norm correction it is an exact (non-basic)
optimality certificate, so even thousands of ties on hundreds of indicator
columns need no simplex step. When pivots are required nevertheless, ties are
resolved by the classical perturbation device of the simplex method and the
vertex is kept exactly across degenerate pivots, so the reported vertex is
always an exact minimizer. When the minimizer is not unique one optimal vertex
is reported, `extra["unique_solution"]` is `False` and a warning is recorded;
Stata, SPSS and EViews may report a different, equally optimal vertex in that
case. The median regression of the Stata manual is such a case (Stata prints
"note: alternate solutions exist"): OpenEconometrics reports 3421.878 for the
coefficient of `foreign`, Stata 3377.771, and every value in between attains
the same minimum; all other coefficients and all standard errors coincide.
`unique_solution` is `True` only when a certificate proves uniqueness (the
rows whose dual value is strictly inside its bounds have full rank).

Collinear regressors are omitted Stata-style and listed in `warnings` and
`provenance["omitted_terms"]`.

### Header statistics

| metric | meaning | Stata |
|---|---|---|
| `sum_adev` | minimized Σ w ρ_τ(residual) | "Min sum of deviations" |
| `sum_rdev` | Σ w ρ_τ(y − q) about the raw quantile q | "Raw sum of deviations" |
| `raw_quantile` | q, the order statistic number int(τ(N + 1)) of y | the "(about …)" value |
| `pseudo_r_squared` | 1 − sum_adev / sum_rdev | "Pseudo R2" |
| `sparsity`, `density` | ŝ = 1/f̂(0) and f̂(0) | `e(sparsity)`, `e(f_r)` |
| `bandwidth` | h on the probability scale | `e(bwidth)` |
| `kernel_bandwidth` | c of the kernel density (kernel method) | `e(kbwidth)` |

The raw quantile follows Stata's output rather than the minimizer of the raw
check loss: for the 74 prices of auto.dta and τ = 0.25 Stata prints "Raw sum
of deviations 41912.75 (about 4187)", the 18th order statistic, although the
19th (4195) gives the smaller sum 41908.75. The rule int(τ(N + 1)) (at least
1) reproduces every Stata value available to us; with weights it is the first
sorted value whose cumulative weight exceeds τ(N + 1) − 1, which is the same
order statistic of the frequency-expanded data. Sums of deviations are sums of
ρ_τ, as current Stata prints them (older releases printed twice that).

Stata's `qreg` prints no model test, so `tests` is empty.

### Covariance options

All three covariances use a bandwidth h (`bandwidth`) and a density method
(`density`, Stata's `denmethod`).

`covariance="nonrobust"` (default) is Stata's `vce(iid)`:

```
V = τ(1 − τ) ŝ² (X'WX)⁻¹ ,        ŝ = 1 / f̂(0)  (the sparsity).
```

- `density="fitted"` (default; EViews' "Siddiqui (mean fitted)"): two
  additional quantile regressions at τ ± h,
  ŝ = x̄'(β̂(τ + h) − β̂(τ − h)) / (2h).
- `density="residual"`: ŝ = (F̂⁻¹(τ + h) − F̂⁻¹(τ − h)) / (2h) with F̂⁻¹ the
  empirical quantile function of the residuals (the definition of Stata's
  `summarize, detail`).
- `density="kernel"`: f̂(0) = Σ w_i K(r_i / c) / (N c) with
  c = min(sd(r), IQR(r)/1.34) · (Φ⁻¹(τ + h) − Φ⁻¹(τ − h)). Passing `kernel=`
  selects this method, like Stata's `vce(iid, kernel(parzen))`.

`covariance="robust"` is Stata's `vce(robust)`, valid when the error density
depends on x:

```
V = D⁻¹ [τ(1 − τ) Σ m_i x_i x_i'] D⁻¹ ,     D = Σ w_i f̂_i x_i x_i' ,
```

with m_i = 1 (no weights), the weight (frequency weights) or w_i² (analytic
and sampling weights) and observation-level densities

- `density="fitted"` (default): f̂_i = 2h / (x_i'β̂(τ + h) − x_i'β̂(τ − h)),
  set to zero where the two fitted quantiles cross (Hendricks and Koenker
  1992; R's `se = "nid"`). `extra["zero_density_observations"]` counts those
  observations.
- `density="kernel"`: f̂_i = K(r_i / c) / c (Powell's sandwich).

`density="residual"` gives a single sparsity and is therefore rejected for the
robust and cluster covariances.

`covariance="cluster"` (or just `cluster="id"`) is the cluster-robust
covariance of Parente and Santos Silva (2016), the one of the community
command `qreg2`: the D above — kernel by default, as in `qreg2`;
`density="fitted"` is accepted — with the meat Σ_g s_g s_g',
s_g = Σ_{i∈g} w_i (τ − 1[r_i ≤ 0]) x_i.

Bandwidth rules (`bandwidth`), with z = Φ⁻¹(τ), φ the normal density and
z_α = Φ⁻¹(0.975):

| rule | h |
|---|---|
| `hsheather` (default) | N^(−1/3) z_α^(2/3) [1.5 φ(z)² / (2z² + 1)]^(1/3) |
| `bofinger` | N^(−1/5) [4.5 φ(z)⁴ / (2z² + 1)²]^(1/5) |
| `chamberlain` | z_α √(τ(1 − τ)/N) |

Kernels (`kernel`) are those of Stata's `kdensity`: `epanechnikov` (default,
support √5), `epan2`, `biweight`, `cosine`, `gaussian`, `parzen`, `rectangle`,
`triangle`.

No finite-sample factor is applied to any of the three covariances. If τ ± h
leaves (0, 1) (extreme quantiles in small samples) the analytic covariances
are undefined and `bandwidth_out_of_range` is raised; use `oe.bsqreg`.

### Weights

`aweight` and `pweight` are rescaled to sum to N; `fweight` replicates
observations (N is the sum of the weights, and every result equals that of the
expanded data set). `pweight` requires `robust` or `cluster` and selects
`robust` by default.

### Example

The examples of [R] qreg (auto.dta, 74 cars):

```python
import openecon as oe

median = oe.qreg(data=auto, y="price", x=["weight", "length", "foreign"])
print(median.summary())
```

```
Median regression — price
Observations: 74  |  Covariance: nonrobust  |  Confidence: 95%

Term       Estimate  Std. error          t     P>|stat|  CI lower  CI upper
---------  --------  ----------  ---------  -----------  --------  --------
Intercept   344.649     5182.39  0.0665038     0.947166  -9991.31   10680.6
weight      3.93359     1.32872    2.96044    0.0041908   1.28354   6.58363
length     -41.2519     45.4647  -0.907339      0.36734  -131.928   49.4246
foreign     3421.88      885.42     3.8647  0.000245884   1655.96   5187.79

quantile: 0.5  |  pseudo_r_squared: 0.234749  |  sum_adev: 54411.3  |  sum_rdev: 71102.5
raw_quantile: 4934  |  sparsity: 5603.62  |  density: 0.000178456  |  bandwidth: 0.231415
iterations: 7  |  df_model: 3  |  df_resid: 70
```

Stata prints the same standard errors (1.328718, 45.46469, 885.4198,
5182.394), "Raw sum of deviations 71102.5 (about 4934)", "Min sum of
deviations 54411.29" and "Pseudo R2 = 0.2347"; `foreign` is the non-unique
coefficient discussed above.

```python
robust = oe.qreg(data=auto, y="price", x=["weight", "length", "foreign"], covariance="robust")
# standard errors 5096.528 (Intercept), 1.694477, 51.73571, 728.5115: Stata's vce(robust)
q25 = oe.qreg(data=auto, y="price", x=["weight", "length", "foreign"], quantile=0.25,
              covariance="robust", kernel="gaussian", bandwidth="bofinger")
clustered = oe.qreg(data=df, y="y", x=["x1", "x2"], cluster="firm")
```

```stata
qreg price weight length foreign
qreg price weight length foreign, vce(robust)
qreg price weight length foreign, quantile(.25) vce(robust, kernel(gaussian) bofinger)
qreg2 y x1 x2, cluster(firm)
```

## bsqreg, sqreg, iqreg: bootstrap quantile regressions

The point estimates are those of `qreg`. The covariance is the pairs
bootstrap: `reps` samples of N observations are drawn with replacement, every
requested quantile is re-estimated on each sample, and with the R replicates
b*_r

```
V = Σ_r (b*_r − b̄*)(b*_r − b̄*)' / (R − 1).
```

- `oe.bsqreg(..., quantile=0.5, reps=20, seed=None)`: one quantile.
- `oe.sqreg(..., quantiles=[0.25, 0.5, 0.75], reps=20)`: all quantiles are
  estimated on the *same* resamples, so `covariance_matrix` holds the
  between-quantile blocks. Terms are named `q25:Intercept`, `q25:x1`,
  `q50:Intercept`, … with equations `q25`, `q50`, `q75` (0.025 → `q2_5`).
  `metrics["pseudo_r_squared_q25"]` etc. give the pseudo R² per quantile.
- `oe.iqreg(..., quantiles=[0.25, 0.75], reps=20)`: the coefficients are
  β̂(τ_high) − β̂(τ_low); a nonzero slope means the regressor changes the
  spread of the conditional distribution. `extra` keeps the two underlying
  coefficient vectors.

Resampling is deterministic given `seed` (a `torch.Generator`; integers from
0 to 2⁶³ − 1). When `seed` is omitted one is drawn and recorded in
`extra["bootstrap"]["seed"]`, so every result can be reproduced. A resample is
represented by its multiplicity counts, i.e. as a frequency-weighted problem
on the distinct rows drawn, which has the same solution as the replicated
data. If a resample leaves the regressors collinear it is skipped and reported
in `warnings` (`metrics["reps"]` is the number of replicates used); fewer than
two usable replicates raise `bootstrap_failed`.

`cluster="id"` resamples whole clusters instead of observations (Stata:
`bootstrap, cluster(id): qreg …`). Weights are not allowed, as in Stata. With
the default 20 replications the standard errors are themselves noisy; use a
few hundred for reported results. `quantiles` may be a list, tuple or array.

```python
fit = oe.sqreg(data=auto, y="price", x=["weight", "length", "foreign"],
               quantiles=[0.25, 0.5, 0.75], reps=200, seed=1)
print(fit.summary())          # q25:weight 1.83179, q50:weight 3.93359, q75:weight 9.22291
spread = oe.iqreg(data=auto, y="price", x=["weight", "length", "foreign"], reps=200, seed=1)
# weight 7.3911, length -223.6288, foreign 1385.2078, Intercept 22122.6788 (as Stata's iqreg)
```

```stata
sqreg price weight length foreign, quantiles(.25 .5 .75) reps(200)
iqreg price weight length foreign, quantiles(.25 .75) reps(200)
bsqreg price weight length foreign, quantile(.5) reps(200)
```

The bootstrap standard errors depend on the random-number stream, so they
differ from Stata's for any seed; the point estimates and pseudo R² values are
those Stata prints (0.1697, 0.2347, 0.3840).

## rreg: robust regression

`oe.rreg(data=df, y=..., x=[...], tune=7, tolerance=0.01)` follows Stata's
`rreg` step by step:

1. **Screening.** OLS on all observations; observations with Cook's distance
   D_i = e_i² h_i / (K s² (1 − h_i)²) > 1 are excluded from everything that
   follows (`metrics["n_dropped_cooks"]`, recorded in `warnings`; N counts the
   remaining observations).
2. **Huber iterations.** With the residuals e_i of the current fit and
   M = med|e_i − med(e)|, the weights are w_i = 1 if |e_i| ≤ 2M and
   2M/|e_i| otherwise. Weighted least squares with these weights, repeated
   until the largest change in a weight is below max(0.05, `tolerance`).
3. **Biweight iterations.** With s = M/0.6745 and u_i = e_i/s,
   w_i = (1 − (u_i/c)²)² for |u_i| < c and 0 beyond, c = 4.685·`tune`/7,
   repeated until the largest change in a weight is below `tolerance`. With
   the default `tune=7`, residuals beyond about 7 MAD get weight zero. The
   coefficients of the last weighted least-squares fit are reported.
4. **Standard errors** from the pseudovalues of Street, Carroll and Ruppert
   (1988): with the final weights w_i, the scale s that produced them, the
   final residuals e_i, ψ'(u) = (1 − (u/c)²)(1 − 5(u/c)²) inside |u| < c,
   m = mean ψ'(e_i/s) and λ = 1 + (K/(N − K))(1 − m)/m,

   ```
   ỹ_i = ŷ_i + (λ / m) w_i e_i .
   ```

   OLS of ỹ on X returns the same coefficients (X'We = 0); its classical
   covariance V = (λ/m)² Σ(w_i e_i)²/(N − K) · (X'X)⁻¹ is reported, together
   with its F test in `tests["model"]`.

The manual describes the Huber stage with c_h = 1.345 on the scale s, i.e.
1.994 M ("about 2M"), and λ with K/N. The constants above (exactly 2M, and
K/(N − K)) are the ones that reproduce the manual's own example digit for
digit — the eight "maximum difference in weights" of the iteration log, the
coefficients, standard errors, confidence intervals and F(2, 71) = 168.32:

```python
fit = oe.rreg(data=auto, y="mpg", x=["weight", "foreign"])
print(fit.summary())
print(fit.extra["weights"])
```

```
Robust regression — mpg
Observations: 74  |  Covariance: nonrobust  |  Confidence: 95%

Term         Estimate   Std. error         t     P>|stat|   CI lower    CI upper
---------  ----------  -----------  --------  -----------  ---------  ----------
Intercept     40.6402      1.26384   32.1561  4.66535e-44    38.1202     43.1603
weight     -0.0063976  0.000371827  -17.2058  4.92453e-27  -0.007139  -0.0056562
foreign      -3.18264     0.627964  -5.06819  3.06649e-06   -4.43476    -1.93051

rmse: 1.9884  |  scale: 1.97418  |  iterations: 8  |  huber_iterations: 4
biweight_iterations: 4  |  n_dropped_cooks: 0  |  df_model: 2  |  df_resid: 71
F test of the slopes (pseudovalue regression): F(2, 71) = 168.325, p = 1.134e-27
{'min': 0.0, 'mean': 0.8509966123048456, 'max': 0.9998584720603984, 'n_zero': 3, 'n_below_half': 8}
```

```stata
rreg mpg weight foreign
```

`metrics`: `rmse` (of the pseudovalue regression), `scale` (the MAD scale s
behind the final weights), `iterations`, `huber_iterations`,
`biweight_iterations`, `n_dropped_cooks`, `df_model`, `df_resid`. `extra`: a
summary of the final weights (min, mean, max, number of zeros), the tuning
constants, m and λ and the iteration log (largest weight change per
iteration, like Stata's log). There are no weights, no R² and no other
covariance estimators, as in Stata.

Limitations: rreg protects against outliers in `y`; apart from the Cook's
distance screen it does not protect against bad leverage points. If at least
half of the observations are fitted exactly the MAD scale is zero and
`zero_scale` is raised. If the screening leaves no more observations than
coefficients, `insufficient_observations` says so.

## nl: nonlinear least squares

### Model and formula

y_i = f(x_i; b) + e_i, where f is written as a formula in the style of
Stata's substitutable expressions:

```python
fit = oe.nl(data=df, y="y", formula="{b0} + {b1} * exp(-{b2} * x)",
            start={"b0": 1, "b1": 2, "b2": 0.1})
```

```stata
nl (y = {b0} + {b1}*exp(-{b2}*x)), initial(b0 1 b1 2 b2 0.1)
```

Grammar:

- parameters in braces, `{b0}`, optionally with a starting value, `{b0=1.5}`;
- data columns by name (the name must be a valid identifier);
- numbers, `+ - * / ^` (`**` is also a power), parentheses, unary minus;
- functions `exp`, `ln`/`log` (natural logarithm, as in Stata), `sqrt`, `abs`,
  `sin`, `cos`, `tan`, `expit`/`invlogit`, `normal` (Φ) and `normalden` (φ).

`-x^2` means `-(x^2)`, as in Stata. Stata evaluates `a^b^c` from left to
right whereas most languages read it right to left, so an unparenthesized
chain of powers is rejected instead of being guessed (`2^-(x^2)` and
`(x^2)^3` are fine). Only the right-hand side is given; `y` is a separate
argument. Stata's linear-combination shorthand `{xb: x1 x2}` and
function-evaluator programs are not supported.

The formula is **never executed as code**. Parameters are replaced by
placeholders, Python's `ast` module builds a syntax tree, and the tree is
walked once; any construct outside the grammar (attribute access, indexing,
comparisons, lambdas, strings, keyword arguments, unknown functions) raises
`invalid_formula`. The accepted tree becomes a flat list of elementary
operations, which is all that is evaluated.

### Estimator

The estimator minimizes S(b) = Σ w_i (y_i − f(x_i; b))² by Gauss–Newton with
step halving and Levenberg–Marquardt damping (Stata uses a modified
Gauss–Newton). With the Jacobian J = ∂f/∂b', residuals r, G = J'WJ and
g = J'Wr, a step solves (G + μ·diag(G)) d = g; μ starts at 0, a step that does
not reduce S is halved up to four times, then μ is raised tenfold until S
falls and lowered again after each success. Convergence requires, after an
accepted step, |d_j| ≤ tol·(|b_j| + 10⁻³) for all parameters and
S_old − S_new ≤ tol·S_new (`tolerance`, default 10⁻⁸; Stata's `eps()` is
10⁻⁵), or a Gauss–Newton step whose predicted reduction is below tol²·S.

The Jacobian is analytic. The operation list is evaluated in forward mode,
each operation propagating exact partial derivatives by the chain rule
((ab)' = a'b + ab', (a^b)' = a^b(b' ln a + b a'/a), exp' = exp, …), so no
numerical differencing and no autograd are involved. Tests compare it with
numerical derivatives for every operator and function.

Starting values come from `start={...}`, then from `{b=value}` in the formula;
parameters with neither start at 0 with a warning, as in Stata. If the
function is not finite at the starting values (`ln` of 0, division by 0)
`invalid_start` is raised.

### Covariance and reported statistics

With K parameters, N observations and J evaluated at the solution:

| `covariance` | formula | Stata |
|---|---|---|
| `nonrobust` (default) | s²(J'WJ)⁻¹, s² = RSS/(N − K) | `vce(gnr)` |
| `robust` | N/(N − K) · (J'J)⁻¹ [Σ r_i² J_i J_i'] (J'J)⁻¹ | `vce(robust)` |
| `HC2`, `HC3` | residuals divided by √(1 − h_i) or (1 − h_i), h the leverage of J | `vce(hc2)`, `vce(hc3)` |
| `cluster` | G/(G − 1) · (N − 1)/(N − K) cluster sandwich, t with G − 1 df | `vce(cluster id)` |

`metrics`: `r_squared` = 1 − RSS/TSS and `adjusted_r_squared` =
1 − (1 − R²)(N − c)/(N − K). If a parameter enters as an additive constant
(its Jacobian column is a nonzero constant) the total sum of squares is taken
about the mean of y and c = 1, as Stata does when it reports "Parameter …
taken as constant term in model"; otherwise TSS is uncentered and c = 0.
`extra["constant_term"]` names that parameter. Also `rmse`, `rss`, `tss`,
`log_likelihood` (Gaussian; unweighted and frequency-weighted fits only),
`iterations`, `df_model`, `df_resid`. `provenance["solver_diagnostics"]`
reports convergence (iterations, function evaluations, final damping, the
last residual sums of squares, the condition number of the Jacobian). Stata's
`nl` prints no model test, so `tests` is empty. Terms carry the parameter
names (`b0`; Stata prints `/b0`).

Weights: `aweight`, `fweight` (replicated observations) and `pweight` (needs
`robust` or `cluster`; selects `robust` by default).

Failure codes: `invalid_formula`, `invalid_start`, `nonconvergence`,
`not_identified` (the Jacobian is rank deficient at the solution, e.g.
`{a} + {b} + {c}*x`), `perfect_fit`, `insufficient_observations`.

### Worked example

```python
import numpy as np, pandas as pd, openecon as oe

rng = np.random.default_rng(0)
df = pd.DataFrame({"x": rng.uniform(0, 5, 500)})
df["y"] = 1.5 + 2.5 * np.exp(-0.8 * df.x) + rng.normal(0, 0.2, 500)

fit = oe.nl(data=df, y="y", formula="{b0} + {b1} * exp(-{b2=0.5} * x)",
            start={"b0": 1, "b1": 1}, covariance="robust")
print(fit.summary())
print(fit.extra["constant_term"], fit.provenance["solver_diagnostics"]["iterations"])
```

```
Nonlinear least squares — y
Observations: 500  |  Covariance: robust  |  Confidence: 95%

Term  Estimate  Std. error        t      P>|stat|  CI lower  CI upper
----  --------  ----------  -------  ------------  --------  --------
b0     1.50925   0.0195899  77.0424  1.70888e-278   1.47076   1.54774
b1     2.52564   0.0376016  67.1685  1.74185e-251   2.45176   2.59952
b2    0.843484   0.0291772   28.909  1.58665e-108  0.786158   0.90081

r_squared: 0.916144  |  adjusted_r_squared: 0.915806  |  rmse: 0.187387  |  rss: 17.4515
tss: 208.113  |  log_likelihood: 129.326  |  iterations: 7  |  df_model: 2
df_resid: 497
b0 7
```

## Validation

Stata output reproduced by the test suite (`tests/test_econ_quantile_oracle.py`):

- **[R] qreg, auto.dta.** `qreg price weight length foreign` at the 0.5, 0.25
  and 0.75 quantiles: coefficients, default `vce(iid)` standard errors, t,
  p-values, confidence interval, raw sum of deviations with its "about" value
  and the pseudo R², to the 7 digits Stata prints; `vce(robust)` standard
  errors; the `iqreg` coefficients. (The reference numbers are those of the
  manual's examples; the median's coefficient of `foreign` is the documented
  non-unique one.)
- **Engel's data.** Stata's `qreg foodexp income, vce(iid, kernel(k) bw)` for
  all eight kernels and three bandwidth rules, shipped with statsmodels' test
  suite: `e(bwidth)` to 12 digits, `e(kbwidth)`, `e(sparsity)`, `e(f_r)`, the
  raw quantile and the standard errors to the precision of Stata's
  single-precision data (1e-6).
- **[R] rreg, auto.dta.** `rreg mpg weight foreign`: all eight values of the
  iteration log, coefficients, standard errors, t statistics, confidence
  intervals and F(2, 71) = 168.32.

R output: `summary(rq(foodexp ~ income, tau = .5), se = "nid")` on Engel's data
(81.48225, 0.56018; standard errors 19.25066, 0.02828) equals the default
robust covariance; the `se = "ker"` values are matched by the Powell formula.

Independent derivations:

- `qreg` coefficients against the exact linear program solved by SciPy's
  HiGHS on random, weighted, badly scaled and heavily tied problems; the
  uniqueness flag against the width of the LP solution set; R's `quantreg`
  values for Engel's data.
- Every `qreg` covariance (three density methods, three bandwidth rules, eight
  kernels, i.i.d./robust/cluster, aweights/fweights/pweights) against the
  formulas above written out in NumPy on top of the LP solution; frequency
  weights against duplicated rows; permutation, rescaling and equivariance
  (a + c·y, and −y ↔ 1 − τ) checks; statsmodels' `QuantReg` as a loose check.
- Bootstrap commands by redrawing the resamples from the recorded seed and
  solving each expanded replicate as an explicit linear program; `iqreg`
  against the contrast of `sqreg`'s joint covariance.
- `rreg` against a NumPy implementation of the procedure; weight and ψ
  functions against statsmodels' robust norms; Cook's distance against
  statsmodels' influence measures; the pseudovalue covariance against the
  M-estimator sandwich; the OLS limit for a huge tuning constant.
- `nl` against `scipy.optimize.least_squares` with complex-step Jacobians and
  NumPy covariance formulas for four models (exponential decay,
  Michaelis–Menten, CES, logistic growth) and every covariance and weight
  type, statsmodels OLS on the Gauss–Newton regression, the NIST StRD
  certified values of the Misra1a problem, and the analytic Jacobian against
  numerical derivatives for every operator and function of the grammar.
- `tests/test_econ_quantile_adversarial.py` checks the failure contract: empty
  and tiny samples, missing and non-numeric columns, constant outcomes,
  collinearity, bad weights, single clusters, extreme magnitudes (1e-8, 1e8),
  hostile formulas and option bounds each give a correct result or an
  `AnalysisError` that says what to change.

## Performance

Timings on one million rows with ten regressors (Apple M-series laptop CPU,
six threads), end to end through `oe.<name>`:

| call | 100 000 rows | 1 000 000 rows |
|---|---|---|
| `qreg`, default covariance (three quantile fits) | 0.3 s | 1.6 s |
| `qreg`, robust (fitted) | 0.2 s | 1.6 s |
| `qreg`, robust or i.i.d. with the kernel density | 0.1 s | 1.0 s |
| `qreg`, cluster | 0.1 s | 1.2 s |
| `qreg`, τ = 0.9 | 0.3 s | 2.3 s |
| `rreg` | 0.1 s | 0.9 s |
| `nl`, four parameters | 0.03 s | 0.24 s |
| `bsqreg`, 20 replications | 1.1 s | 8.9 s |
| `sqreg`, three quantiles, 20 replications | 3.5 s | |
| `iqreg`, 20 replications | 2.7 s | |

The quantile solver needs 10–25 interior-point iterations, each costing one
K×K cross product; time grows linearly in N. Tied data cost the same: a
rounded outcome on a million rows takes 1.5 s, a dummy-only design with a
six-valued outcome 0.3 s, and 50 000 rows with 400 indicator columns and a
four-valued outcome (20 000 tied residuals) about 10 s, all without simplex
steps.

## Limitations

- `qreg`, `bsqreg`, `sqreg`, `iqreg` and `rreg` need an in-memory table.
- The analytic `qreg` covariances require τ ± h inside (0, 1); very extreme
  quantiles in small samples need the bootstrap.
- `sqreg`/`iqreg`/`bsqreg` take no weights (as in Stata).
- `nl` has no `hac` covariance, no linear-combination shorthand and no
  post-estimation `predict` for new data.
- Post-estimation tests across quantiles (`test [q25]x = [q75]x`) are not a
  separate command yet; the joint covariance needed for them is in
  `covariance_matrix`.

## Uncertain conventions

Checked against Stata output: the default `vce(iid)` and `vce(robust)` of
`qreg`, its kernel density estimates, bandwidth rules at the default level,
the raw quantile, and the whole `rreg` procedure. Not checked — these follow
the textbook definition or our reading of the manuals:

1. **z_α in the Hall–Sheather and Chamberlain bandwidths** is fixed at
   α = 0.05 (Koenker 2005; R's `bandwidth.rq`), so standard errors do not
   change with `alpha`. Stata may tie it to `level()`; at the default level
   the two agree.
2. **`density="residual"`** uses the `summarize, detail` quantile definition
   for F̂⁻¹; Stata's exact empirical-quantile rule for `denmethod(residual)`
   could not be checked. Whether Stata accepts `residual` with `vce(robust)`
   is also unverified (OpenEconometrics rejects it).
3. **sd(r) in the kernel bandwidth** uses the divisor N − 1; in every Stata
   reference case the IQR/1.34 branch was the smaller one, so the sd branch is
   untested against Stata.
4. **qreg weights.** OpenEconometrics accepts aweights, fweights and pweights; current
   Stata documents fweights, iweights and pweights, so analytic weights are an
   OpenEconometrics extension by analogy with `regress`: the weighted i.i.d.
   covariance is τ(1 − τ)ŝ²(X'WX)⁻¹ with weights normalized to sum to N. The
   robust meat uses w² (w for frequency weights), and the weighted raw
   quantile is the frequency-weight extension described above. Only the
   frequency-weight case is pinned down (by the duplicated-rows identity); no
   weighted Stata output was available.
5. **The raw quantile rule int(τ(N + 1))** was inferred from five Stata
   outputs (auto: τ = 0.25, 0.5, 0.75; Engel: τ = 0.5, 0.75); other rules
   considered (the minimizer, `summarize`'s percentile, round(τN)) contradict
   at least one of them.
6. **Cluster-robust qreg** is not a Stata `qreg` option. OpenEconometrics uses the
   Parente–Santos Silva meat with the density matrix D of `vce(robust)`
   (kernel by default), no finite-sample factor and t(N − K) inference.
   `qreg2` uses a uniform kernel with a MAD-based scale; choose
   `kernel="rectangle"` to come closest.
7. **Bootstrap variance** is taken about the mean of the replicates with
   divisor R − 1; the random-number stream is Torch's, so replicate samples
   differ from Stata's for any seed. Cluster resampling is an extension
   (`bootstrap, cluster():`), as is skipping collinear resamples.
8. **Equation names of sqreg** for non-integer percentages (`q2_5`) are an
   OpenEconometrics convention.
9. **rreg thresholds.** The Huber stage stops at max(0.05, `tolerance`): 0.05
   is consistent with the manual's log (it continues after 0.089 and stops
   after 0.027 with the default tolerance), but how Stata combines it with a
   user-supplied `tolerance()` is not documented. The Cook's-distance
   screening and N after screening follow the manual's description and were
   not exercised by a Stata example with dropped observations.
10. **nl.** Constant detection looks for a parameter whose Jacobian column is
    a nonzero constant; Stata's own rule is not documented in detail.
    Stata documents aweights, fweights and iweights; OpenEconometrics offers aweights,
    fweights and pweights (pweights = aweights with the robust covariance).
    The log likelihood is reported only where its definition is unambiguous
    (no weights or frequency weights). No Stata output for `nl` was available;
    the covariances are those of `regress` applied to the Jacobian.
11. The Stata manual values used as references were entered from the manual's
    printed examples, not produced by a Stata run in this repository.
