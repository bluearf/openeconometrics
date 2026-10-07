# Discrete-choice models: `ologit`, `oprobit`, `mlogit`, `clogit`, `hetprobit`, `biprobit`

Maximum-likelihood models for ordered, unordered, grouped-binary and paired
binary outcomes. Everything on this page is implemented in OpenEconometrics on float64
PyTorch tensors with analytic scores and Hessians (no autograd, no estimation
library at fit time): the ordered and multinomial likelihoods, the recursive
elementary-symmetric-function algorithm of the conditional logit, the
heteroskedastic probit and a bivariate normal distribution function written
for the bivariate probit. Binary `logit` and `probit` are core estimators and
are documented elsewhere.

```python
import openecon as oe

oe.ologit(data=df, y="rating", x=["age", "income"])                  # ologit rating age income
oe.oprobit(data=df, y="rating", x=["age", "income"], cluster="firm")  # oprobit ..., vce(cluster firm)
oe.mlogit(data=df, y="mode", x=["age", "income"], base="bus")         # mlogit mode age income, baseoutcome(bus)
oe.clogit(data=df, y="chosen", x=["price", "time"], group="trip")     # clogit chosen price time, group(trip)
oe.hetprobit(data=df, y="work", x=["age", "kids"], het=["income"])    # hetprobit work age kids, het(income)
oe.biprobit(data=df, y1="work", y2="insured", x=["age", "kids"])      # biprobit work insured age kids
```

| Stata / SPSS / EViews | OpenEconometrics |
| --- | --- |
| `ologit y x1 x2` · SPSS `PLUM y WITH x1 x2 /LINK=LOGIT` · EViews `ordered(d=l)` | `oe.ologit(data=df, y="y", x=["x1","x2"])` |
| `oprobit y x1 x2` · SPSS `PLUM ... /LINK=PROBIT` · EViews `ordered(d=n)` | `oe.oprobit(data=df, y="y", x=["x1","x2"])` |
| `ologit y x1, offset(e)` | `..., offset="e"` |
| `mlogit y x1 x2` · SPSS `NOMREG y WITH x1 x2` | `oe.mlogit(data=df, y="y", x=["x1","x2"])` |
| `mlogit y x1 x2, baseoutcome(3)` · SPSS `NOMREG y (BASE=3)` | `..., base=3` |
| `clogit y x1 x2, group(id)` | `oe.clogit(data=df, y="y", x=["x1","x2"], group="id")` |
| `xtset id` + `xtlogit y x1 x2, fe` | the same call with `group="id"` |
| `hetprobit y x1 x2, het(z1 z2)` | `oe.hetprobit(data=df, y="y", x=["x1","x2"], het=["z1","z2"])` |
| `biprobit y1 y2 x1 x2` | `oe.biprobit(data=df, y1="y1", y2="y2", x=["x1","x2"])` |
| `biprobit (y1 = x1 x2) (y2 = x1 z)` | `..., x=["x1","x2"], x2=["x1","z"]` |
| `..., vce(robust)` / `vce(opg)` / `vce(cluster g)` | `covariance="robust"` / `"opg"` / `cluster="g"` |
| `... [fweight=n]` | `weights="n", weight_type="fweight"` |

Every function is keyword-only, returns a `ResultBundle` (`summary()`,
`to_latex()`, `model_dump_json()`), and can equally be run as
`oe.fit(ModelSpec(estimator="ologit", ...), data=df)`.

## Conventions shared by the six estimators

**Estimation.** Each likelihood is maximized by Newton-Raphson
(`engines.optimize.maximize_newton`) with the analytic gradient and Hessian of
the model; the tests compare both with numerical derivatives. A step is halved
until the log likelihood rises; in a non-concave region the step is a Marquardt
step. Convergence requires a negative definite Hessian, `g'(-H)^-1 g <= 1e-10`
(Stata's `nrtolerance`, default `1e-5` there), a gradient below `1e-8` and a
relative step below `1e-10`. The record is in `provenance["optimizer"]`.

**Centring (numerical, not a change of model).** In an equation with a
constant — explicit, or the cutpoints of an ordered model — the index is
`a + x'b = (a + m'b) + (x - m)'b`, so the level of a regressor only moves the
constant. The likelihood is maximized on regressors centred at their
(weighted) means `m`, and estimates and covariance are mapped back exactly:
`a = a_c - m'b` (`cut = cut_c + m'b`), `V = J V_c J'` with `J` the Jacobian of
that linear map, which is exact for every covariance estimator. The result is
the model as specified, but the Hessian no longer loses `(mean/spread)^2`
digits: with a regressor such as `1e6 + noise` the standard errors are accurate
to the precision of the data (uncentred, they were off by up to 0.4%, and the
iteration failed at a level of `1e8`). `clogit` centres within groups (its
likelihood depends on within-group differences only) and `hetprobit` also
centres its variance regressors (see below). Equations without a constant
(`intercept=False`) are not centred. Scale needs no such device: the Newton
step and the information inverse are computed on a diagonally equilibrated
Hessian, so `x * 1e8` only rescales the coefficient.

**Covariance** (`core.ml_covariance`, Stata's `ml` conventions). `H` is the
Hessian of the weighted log likelihood and `s_i` the weighted score of
observation i:

| `covariance` | formula | Stata |
| --- | --- | --- |
| `nonrobust` (default) | `(-H)^-1` | `vce(oim)` |
| `opg` | `(sum_i s_i s_i')^-1` | `vce(opg)` |
| `robust` | `N/(N-1) (-H)^-1 [sum_i s_i s_i'] (-H)^-1` | `vce(robust)` |
| `cluster` | `G/(G-1) (-H)^-1 [sum_g s_g s_g'] (-H)^-1`, `s_g` the cluster sums | `vce(cluster g)` |

Giving `cluster=` selects `cluster`. Two cluster columns give two-way
clustering (Cameron-Gelbach-Miller inclusion-exclusion with the smaller
`G/(G-1)`), an OpenEconometrics extension. Coefficient tests are z tests; confidence
intervals use the normal quantile. The convention used is recorded in
`result.inference["correction"]`. `clogit` differs in one respect: its unit of
observation is the group (see below).

**Weights.** Every likelihood, score and Hessian sum is `sum_i w_i (.)`:

| `weight_type` | `w_i` | `N` | notes |
| --- | --- | --- | --- |
| `fweight` | integer frequency | `sum w_i` | identical to the data set with rows duplicated, under every covariance |
| `aweight` | rescaled to sum to the number of rows | rows | |
| `iweight` | used as given (must be positive) | rows | covariance scales with `1 / w` |
| `pweight` | used as given | rows | needs `robust` or `cluster` (`robust` is the default of the convenience functions); the covariance is invariant to the scale of the weights |

Rows with zero weight are excluded. `clogit` accepts `fweight`, `iweight` and
`pweight` only, applied to whole groups. The unit of `iweight`s and `pweight`s
does not affect the estimates: the iteration runs on the likelihood divided by
the mean weight, so that the convergence rules mean the same thing for weights
of `1e-12` and of `1e12`, and the Hessian is scaled back afterwards. In `opg`,
`robust` and `cluster` the score of a row is `w_i s_i` for `aweight`,
`iweight` and `pweight` (the meat is `sum w_i^2 s_i s_i'`) and enters `f_i`
times for `fweight` (`sum f_i s_i s_i'`), the convention of
`core.ml_covariance`.

**Model test.** `tests["model"]` is the likelihood-ratio chi2 against the
estimator's null model when the covariance is `nonrobust`, and the Wald chi2
of the same coefficients otherwise (Stata prints `LR chi2` or `Wald chi2` in
the same way). `hetprobit` and `biprobit` always report the Wald chi2, as
Stata does. `metrics["pseudo_r_squared"]` is McFadden's `1 - ll/ll_0`.
`aic = -2 ll + 2 p` and `bic = -2 ll + p ln N` count every estimated parameter
(cutpoints, `/athrho` and variance coefficients included), as `estat ic`.

**Separation.** When a regressor predicts some outcomes perfectly the
likelihood has no finite maximum. The iteration is stopped as soon as the
fitted probabilities of correctly predicted observations are numerically one
and the estimator raises `separation_detected` instead of reporting huge
coefficients. (Stata drops the offending observations and variables for binary
models and reports "completely determined" observations for the others.) The
same code is raised for the forms of quasi-complete separation in which no
observation's own outcome is predicted perfectly: in an ordered model a
regressor that separates the two sides of one cutpoint (`y <= j` from
`y > j`), and in `mlogit` a regressor that rules a category out for part of
the sample. A regressor that bounds a coefficient from one side only, with
overlap on the other, has a finite maximum and is estimated.

**Collinearity, categoricals, missing values.** Collinear regressors are
omitted left to right with a warning and listed in
`provenance["omitted_terms"]`. In the equations that have no constant of their
own (ordered models, the variance equation of `hetprobit`) a constant column is
collinear with the implicit one and is omitted too. `categorical=[...]`
expands a column into treatment-coded indicators named `name[level]` with the
first level as reference. `missing="raise"` (default) refuses incomplete rows;
`missing="drop"` excludes and counts them.

**Errors.** Invalid input raises `AnalysisError(code, message)`; common codes:
`invalid_spec`, `empty_data`, `empty_sample`, `missing_columns`,
`missing_values`, `non_numeric_column`, `non_finite_values`,
`constant_outcome`, `invalid_binary_outcome`, `invalid_ordered_outcome`,
`too_many_categories`, `invalid_base_category`, `ambiguous_categories`,
`separation_detected`, `boundary_solution` (biprobit), `nonconvergence`,
`singular_information`, `insufficient_observations`, `insufficient_clusters`,
`unsupported_covariance` (pweights with `nonrobust`/`opg`), `negative_weights`,
`noninteger_frequency_weights`, `weights_not_constant_within_group`,
`no_outcome_variation` and `no_within_group_variation` (clogit),
`no_variance_regressors` and `variance_scale_overflow` (hetprobit). A result
never contains NaN or infinite values.

## `ologit`, `oprobit`: ordered outcomes

**Model.** For ordered categories `1 < 2 < ... < J`,

    Pr(y = j | x) = F(k_j - x'b) - F(k_{j-1} - x'b),      k_0 = -inf,  k_J = +inf,

with `F` the logistic distribution function (`ologit`, the proportional-odds
model: `ln[Pr(y <= j)/Pr(y > j)] = k_j - x'b`) or the standard normal one
(`oprobit`: a latent `y* = x'b + e`, `e ~ N(0,1)`, observed in intervals).
There is no constant; the `J - 1` cutpoints take its place and are estimated
directly, as in Stata. They are reported after the slopes as `/cut1` ..
`/cut{J-1}` with standard errors (a z test of a cutpoint against zero is
rarely of interest). SPSS PLUM ("threshold" minus "location") and EViews
("limit points") use the same sign convention. An `offset` column enters `x'b`
with coefficient one.

**Categories.** A numeric outcome is ordered by value (the values need not be
consecutive integers); an ordered pandas `Categorical` by its category order;
an unordered `Categorical` or text column is refused
(`invalid_ordered_outcome`). The categories observed in the estimation sample
define `J`, and are returned in `extra["categories"]`. More than 500 distinct
values raise `too_many_categories`.

**Estimator.** With `u_i = k_{y_i} - x_i'b`, `l_i = k_{y_i - 1} - x_i'b`,
`p_i = F(u_i) - F(l_i)`, density `f`, `A = f(u)/p`, `B = f(l)/p`:

    d ln p / du = A,        d ln p / dl = -B,
    d2 ln p / du2 = (f'/f)(u) A - A^2,   d2 ln p / dl2 = -(f'/f)(l) B - B^2,   d2 ln p / du dl = A B,

chained to `(b, k)`. The interval probability is formed on the side where both
tails are small, so `ln p` is accurate for extreme indices. Starting values:
`b = 0` and the cutpoints that reproduce the cumulative category shares (the
maximum of the cutpoints-only model). A trial step that would reorder the
cutpoints has no likelihood and is rejected by the line search.

**Reported.**

- `metrics`: `log_likelihood`, `pseudo_r_squared`, `aic`, `bic`
  (`k + J - 1` parameters), `n_categories`.
- `tests["model"]`: LR chi2(k) against the cutpoints-only model (with the same
  offset), or Wald chi2(k).
- `extra`: `categories`, `cutpoints`, `category_counts`,
  `null_log_likelihood`, `link`, `category_order`.

```python
import numpy as np, pandas as pd, openecon as oe
rng = np.random.default_rng(0)
df = pd.DataFrame({"x": rng.normal(size=800)})
df["rating"] = np.digitize(0.8 * df.x + rng.logistic(size=800), [-1.0, 0.5, 2.0])
result = oe.ologit(data=df, y="rating", x=["x"])
print(result.summary())               # x, /cut1, /cut2, /cut3
np.exp(result.coefficients[0].estimate)   # proportional odds ratio
```

## `mlogit`: multinomial logit

**Model.** For `J` unordered categories with base category `B`,

    Pr(y = j | x) = exp(x'b_j) / sum_l exp(x'b_l),        b_B = 0.

Each non-base category has its own equation; `exp(b_j)` is the relative-risk
ratio against the base. Terms are named `<category>:<term>` and grouped in
equations named after the category.

**Categories and base.** Categories are the distinct outcome values of the
sample (numbers, strings, booleans or a `Categorical`), listed in sorted order
(`Categorical` order). The base is the category with the largest (weighted)
frequency, as Stata's default; on ties the first in category order. `base=`
selects another one (`invalid_base_category` if it is not observed). SPSS
NOMREG uses the last category by default: pass `base=` to reproduce it.
Changing the base only reparameterizes: `b_j(new) = b_j - b_new`.

**Estimator.** The log likelihood `sum_i w_i [x_i'b_{y_i} - ln sum_l exp(x_i'b_l)]`
is globally concave; Newton-Raphson starts from zero. With `d_ij = 1[y_i = j]`,

    gradient_j = X'(w (d_j - p_j)),      Hessian_jl = -X' diag(w p_j (1[j = l] - p_l)) X.

The softmax uses a log-sum-exp; the `k(J-1)`-square Hessian is accumulated
over row blocks as `-(blockdiag_j X' diag(w p_j) X - M'M)`, `M = sqrt(w) (p_i kron x_i)`,
so the cost is `O(N k^2 (J-1)^2)` flops in BLAS and no `N`-by-`k(J-1)` matrix
is kept (it is only formed for `opg`, `robust` and `cluster`).

**Reported.**

- `metrics`: `log_likelihood`, `pseudo_r_squared`, `aic`, `bic`
  (`k (J - 1)` parameters), `n_categories`.
- `tests["model"]`: LR chi2 with `(k - 1)(J - 1)` degrees of freedom against
  the constant-only model (fitted shares), or the Wald chi2 of all slopes.
  With `intercept=False` the null model is `b = 0` (equal probabilities).
- `extra`: `categories`, `base`, `equations`, `category_counts`,
  `null_log_likelihood`, `base_rule`.

```python
rng = np.random.default_rng(0)
df = pd.DataFrame({"x": rng.normal(size=900)})
utility = np.column_stack([np.zeros(900), 0.5 + df.x, -0.3 - 0.7 * df.x])
df["choice"] = (utility + rng.gumbel(size=(900, 3))).argmax(axis=1)
result = oe.mlogit(data=df, y="choice", x=["x"], base=0)
[c.term for c in result.coefficients]     # ['1:Intercept', '1:x', '2:Intercept', '2:x']
```

## `clogit`: conditional (fixed-effects) logit — also `xtlogit, fe`

**Model.** A binary outcome with a group effect,
`Pr(y_i = 1) = L(a_g + x_i'b)` for row i of group g (a matched case-control
set, a choice set, a person observed over time). Conditioning on the number of
positives `m_g` of each group removes `a_g`:

    ln L_g = sum_i y_i x_i'b - ln f_g(T_g, m_g),
    f_g(T, m) = sum over all d in {0,1}^T with sum d = m of exp(sum_i d_i x_i'b).

There is no constant, and a regressor that does not vary within any group is
not identified (omitted with a warning). `xtlogit, fe` is the same estimator
with `group` = the panel variable; McFadden's choice model is the special case
of one positive per group. Every admissible outcome vector has the same number
of positives, so `L_g` depends on the regressors only through `x_i - xbar_g`:
the likelihood is evaluated on regressors centred within groups, which is the
same function of `b` and keeps its Hessian accurate whatever the level of a
regressor (uncentred, the standard errors were off by 6% at a level of `1e7`).

**Sample.** Groups whose outcomes are all 0 or all 1 contribute `L_g = 1` and
are dropped, as in Stata ("note: N groups (M obs) dropped because of all
positive or all negative outcomes"); the count is in
`metrics["n_groups_dropped"]`, `extra["observations_dropped"]` and a warning.
`nobs` counts the rows of the remaining groups. With `fweight`s a group counts
as many times as its weight in all of these numbers, exactly as if it had been
replicated.

**Estimator.** `f_g` is an elementary symmetric function of `exp(x_i'b)` and
obeys `f(t, j) = f(t-1, j) + exp(x_t'b) f(t-1, j-1)` (Howard 1972; Gail, Lubin
and Rubinstein 1981; the algorithm Stata's clogit documents). OpenEconometrics carries
the recursion in a normalized form that cannot overflow: with
`b(t, j) = Pr(y_t = 1 | j positives among the first t rows)`, the conditional
mean `D` and covariance `C` of `sum_s y_s x_s` are updated as two-component
mixtures,

    D(t, j) = D(t-1, j) + b d,       d = D(t-1, j-1) + x_t - D(t-1, j),
    C(t, j) = (1 - b) C(t-1, j) + b C(t-1, j-1) + b (1 - b) d d',

and at `(T_g, m_g)` they are the gradient and Hessian of `ln f_g`. All groups
of a block advance together (groups sorted by size; a Python loop runs over
positions within a group only). Groups with more positives than negatives are
reflected, so the recursion width is `min(m, T - m)`; groups with a single
positive (or a single negative) use the closed-form multinomial-logit
expressions. Cost: `O(N k^2)` for one positive per group and
`O(N min(m, T - m) k^2)` in general. Newton-Raphson starts from `b = 0`.

**Covariance.** The independent unit is the group, so:

- `nonrobust`: inverse observed information.
- `opg`: outer product of the group scores.
- `robust`: `G/(G-1) (-H)^-1 [sum_g s_g s_g'] (-H)^-1` with the group scores —
  Stata's `vce(robust)` for clogit is `vce(cluster groupvar)`.
- `cluster`: per-observation scores
  `(y_i - Pr(y_i = 1 | m_g)) (x_i - xbar_g)` summed within the cluster
  column(s), `G_c/(G_c - 1)`. When every group lies inside one cluster (what
  Stata requires) the cluster sums are sums of group scores and the centring
  is immaterial. Groups that are split across clusters are accepted with a
  warning (Stata needs `nonest` for them). How a group's score is divided
  among its rows is then a convention: `y_i - Pr(y_i = 1 | m_g)` sums to zero
  within a group, so any group-constant shift of `x` leaves the group score
  unchanged; OpenEconometrics uses the within-group centred regressors because that
  split does not depend on the origin of the regressors.

**Weights** apply to groups as a whole and must be constant within a group
(`weights_not_constant_within_group` otherwise), Stata's rule. `fweight`
replicates groups: `n_groups = sum of group weights`.

**Reported.**

- `metrics`: `log_likelihood`, `pseudo_r_squared` (`ll_0` = the conditional
  likelihood at `b = 0`), `aic`, `bic`, `n_groups`, `n_groups_dropped`.
- `tests["model"]`: LR chi2(k) against `b = 0`, or Wald chi2(k).
- `extra`: `group`, `group_sizes` (min, mean, max), `multiple_positive_outcomes`,
  `observations_dropped`, `null_log_likelihood`.

Error codes specific to clogit: `no_outcome_variation` (every group is all 0
or all 1), `no_within_group_variation` (no regressor varies within groups).

```python
rng = np.random.default_rng(0)
df = pd.DataFrame({"id": np.repeat(np.arange(300), 5), "x": rng.normal(size=1500)})
effect = rng.normal(size=300)[df.id]
df["y"] = (rng.random(1500) < 1 / (1 + np.exp(-(effect + 0.8 * df.x)))) * 1.0
result = oe.clogit(data=df, y="y", x=["x"], group="id")    # xtlogit y x, fe
result.metrics["n_groups"], result.metrics["n_groups_dropped"]
```

## `hetprobit`: heteroskedastic probit

**Model.** A probit whose latent error has a standard deviation that depends
on covariates (Harvey's multiplicative form):

    Pr(y = 1 | x, z) = Phi(x'b / exp(z'g)).

The variance equation has no constant (it would only rescale `b`); `g = 0` is
the ordinary probit. Its coefficients are the terms `lnsigma:<column>` in the
equation `lnsigma` (Stata 16+ labels the equation `lnsigma`, earlier releases
`lnsigma2`; the parameter is the same). A variable may appear in both `x` and
`het`; its effect on the probability then depends on both coefficients.

**Estimator.** With `s = exp(z'g)`, `t = x'b/s`, `q = 2y - 1`,
`m = q phi(q t)/Phi(q t)` and `v = m (m + t)`:

    d ll/db = m x / s,                 d ll/dg = -m t z,
    d2 ll/db db' = -v x x' / s^2,      d2 ll/db dg' = (v t - m) x z' / s,     d2 ll/dg dg' = (m t - v t^2) z z'.

`m` is computed with the scaled complementary error function in the lower
tail, so it is accurate where `phi/Phi` would be `0/0`. Newton-Raphson starts
at the probit estimates and `g = 0`. The likelihood is not globally concave
and can be flat when `z` almost separates the outcome; a failed iteration is
reported as `nonconvergence` or `separation_detected` with advice.

The iteration runs on mean-centred variance regressors. Without a constant in
the variance equation, `z'g = (z - m)'g + m'g` only rescales the mean
equation: `b = c exp(m'g)` where `c` are the coefficients of the centred
problem. Estimates and covariance are mapped back exactly
(`V(b, g) = J V(c, g) J'`, `J = [[exp(m'g) I, b m'], [0, I]]`), so the result
is the model as specified, but a variance regressor with a large mean (an
age, a calendar year) no longer wrecks the conditioning. When `|m'g| > 20` a
warning explains the scale of the mean-equation coefficients, and when
`|m'g| > 300` (coefficients and their variances would leave the float64 range)
the estimator raises `variance_scale_overflow` and asks for centred `het`
columns. The regressors of the mean equation are centred as described under
"Centring" when it has a constant.

**Reported.**

- `metrics`: `log_likelihood`, `aic`, `bic` (`k + q` parameters).
- `tests["model"]`: Wald chi2 of the slopes of the mean equation (Stata's
  header statistic for hetprobit under every VCE).
- `tests["lnsigma"]`: test of homoskedasticity `g = 0` — the LR chi2(q)
  against the probit model under `nonrobust`/`opg` (Stata's "LR test of
  lnsigma=0"), the Wald chi2(q) under `robust`/`cluster`.
- `extra`: `zero_outcomes`, `nonzero_outcomes`, `probit_log_likelihood`,
  `variance_terms`, `variance_regressor_means`. The chart sample
  (`predictions`) holds the fitted `Pr(y = 1)`.

`no_variance_regressors` is raised when no column of `het` varies.

```python
rng = np.random.default_rng(0)
df = pd.DataFrame({"x": rng.normal(size=2000), "z": rng.normal(size=2000)})
df["y"] = (0.3 + df.x + np.exp(0.6 * df.z) * rng.normal(size=2000) > 0) * 1.0
result = oe.hetprobit(data=df, y="y", x=["x"], het=["z"])
result.tests["lnsigma"]            # LR test of homoskedasticity
```

## `biprobit`: bivariate probit

**Model.** Two probit equations with correlated errors,

    y1 = 1[x1'b1 + e1 > 0],   y2 = 1[x2'b2 + e2 > 0],   corr(e1, e2) = rho,
    Pr(y1, y2 | x) = Phi2(q1 x1'b1, q2 x2'b2; q1 q2 rho),     q_j = 2 y_j - 1.

With `x2=None` both equations use `x` (`biprobit y1 y2 x`, titled "Bivariate
probit regression"); with `x2` given the model is Stata's "Seemingly unrelated
bivariate probit". Terms are `<y1>:<term>` and `<y2>:<term>` in equations
named after the outcomes, followed by the ancillary `/athrho = atanh(rho)`.
`intercept=False` removes the constant from both equations. One outcome may be
a regressor of the other equation (the recursive bivariate probit with an
endogenous dummy, whose likelihood is the same expression); listing each
outcome in the other equation is refused as incoherent.

**The bivariate normal distribution function** is implemented in
`discrete/bivariate.py` with Genz's (2004) algorithm, a refinement of Drezner
and Wesolowsky (1990):

- `|rho| < 0.925`: Plackett's identity `dPhi2/drho = phi2`, integrated in
  `t = asin(rho)` with a 20-point Gauss-Legendre rule:
  `Phi2(a, b; r) = Phi(a) Phi(b) + (1/2pi) int_0^{asin r} exp(-(a^2 + b^2 - 2ab sin t)/(2 cos^2 t)) dt`.
- `|rho| >= 0.925`: the integral runs from `|rho|` to 1, where `Phi2` is a
  univariate probability; the leading terms of the integrand at `rho = +-1`
  are integrated in closed form and the smooth remainder by the same rule.

The absolute error is about `1e-15` (tests: 30-digit quadrature and SciPy on
both branches). `1 - rho^2` is carried as `1/cosh(athrho)^2`, so the function
stays accurate when `rho` rounds to one.

A likelihood needs `ln Phi2` with *relative* accuracy, which an absolute error
of `1e-15` does not provide for the small joint probability of an outlying
observation. Probabilities below `1e-5` are therefore recomputed from the
representation with a positive integrand,

    Phi2(a, b; r) = int_{-inf}^{a} phi(x) Phi((b - r x)/sqrt(1 - r^2)) dx,      a <= b,

whose logarithm is concave: its mode and the points where it has fallen by 40
are bracketed by bisection and the integral is a 96-point Gauss-Legendre sum
accumulated in logarithms. `ln Phi2` is then accurate to about `1e-12` in
relative terms however small the probability (it does not underflow, like
`log_ndtr` in one dimension), so gross outliers do not disturb the iteration.
Only the affected observations pay for it.

**Estimator.** Newton-Raphson on `(b1, b2, athrho)` with the analytic score
and the analytic Hessian. With `w_j = q_j x_j'b_j`, `r = q1 q2 rho`,
`d = (1 - r^2)^(-1/2)`, `G_1 = phi(w_1) Phi(d (w_2 - r w_1))/Phi2`,
`G_2` symmetric and `D = phi2(w_1, w_2; r)/Phi2`:

    d ll/dw_j = G_j,     d ll/dr = D,
    d2 ll/dw_1^2 = -w_1 G_1 - r D - G_1^2,      d2 ll/dw_1 dw_2 = D - G_1 G_2,
    d2 ll/dw_1 dr = -D (d^2 (w_1 - r w_2) + G_1),
    d2 ll/dr^2 = D [d^2 r (1 - Q) + d^2 w_1 w_2 - D],    Q = d^2 (w_1^2 + w_2^2 - 2 r w_1 w_2),

chained through `drho/dathrho = 1 - rho^2`. Starting values are the two
univariate probits and `rho = 0` (Stata's comparison models).

**Reported.**

- `metrics`: `log_likelihood`, `aic`, `bic` (`k1 + k2 + 1` parameters), `rho`.
- `extra["rho"]`: `estimate`, `std_error` = `(1 - rho^2) se(athrho)` (delta
  method), `ci_low`/`ci_high` = `tanh(athrho -+ z se(athrho))` — the way
  Stata displays `rho`.
- `tests["model"]`: Wald chi2 of all slopes of both equations.
- `tests["rho"]`: the LR chi2(1) of `rho = 0` against the two separate probits
  under `nonrobust`/`opg` ("LR test of rho=0"); the Wald chi2(1) of
  `athrho = 0` under `robust`/`cluster` ("Wald test of rho=0").
- `extra`: `comparison_log_likelihood`, `probit_log_likelihoods`,
  `outcome_counts` (the weighted 2x2 table, key `"10"` = `y1 = 1, y2 = 0`),
  `seemingly_unrelated`, `recursive`. The chart sample holds `Pr(y1 = 1)`.

When the correlation runs to +-1 there is no interior maximum and
`boundary_solution` is raised: `y2` equals `y1` or its complement, or one cell
of the 2x2 table of outcomes is empty. In the last case the likelihood rises
in `|athrho|` for ever but ever more slowly, so the iteration is stopped when
it fails, or goes flat to rounding, with `|rho| > 0.995` (a fit that converges
at such a correlation is reported normally). Separation in either equation is
detected in the starting probit and reported with the equation's name.

```python
rng = np.random.default_rng(0)
df = pd.DataFrame({"x": rng.normal(size=2000), "z": rng.normal(size=2000)})
e = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=2000)
df["work"] = (0.3 + 0.8 * df.x + e[:, 0] > 0) * 1.0
df["insured"] = (-0.2 + 0.5 * df.x - 0.6 * df.z + e[:, 1] > 0) * 1.0
result = oe.biprobit(data=df, y1="work", y2="insured", x=["x"], x2=["x", "z"])
result.extra["rho"], result.tests["rho"]
```

## Limitations

- No post-estimation commands yet (`predict` for category probabilities,
  `margins`, the Brant test of proportional odds, `rrr`/`or` display options).
  Odds and relative-risk ratios are `exp(estimate)`.
- No linear constraints; `mlogit` has no `offset`; `hetprobit` and `biprobit`
  have no offsets; `biprobit` has no `partial` (partial observability) option
  and one `intercept` switch for both equations.
- `clogit` evaluates the exact conditional likelihood only; the cost grows
  with `min(m, T - m)` per group, so groups with hundreds of positives and
  negatives each are slow.
- Out-of-core datasets are not supported (`streaming_unsupported`).

## Uncertain conventions

`provenance["stata_parity_validated"]` is `False` for every estimator: the
formulas follow the Stata manuals, but results have been validated against
statsmodels, brute-force maximization and explicit algebra, not against Stata
output. Specific points where the convention is a judgment:

- **aweights.** Stata's ologit, oprobit, mlogit, hetprobit and biprobit accept
  fweights, iweights and pweights only. OpenEconometrics also accepts aweights and
  treats them as `ml` does (rescaled to sum to N).
- **iweights.** `N` is the number of rows (Stata's built-in commands report the
  sum of the weights for some estimators), and `opg` / `robust` / `cluster`
  use `w_i s_i` as the score of a row, so integer iweights do not reproduce
  fweights under those covariances. Use `fweight` for replication.
- **Negative iweights** are rejected.
- **Model test under `opg`.** `tests["model"]` is the LR chi2 only under
  `nonrobust`; under `opg` it is the Wald chi2 (Stata may print the LR
  statistic there).
- **pweights default.** The convenience functions select `robust` for
  pweights, as Stata does; `nonrobust`/`opg` with pweights is an error.
- **mlogit base on ties.** Among equally frequent categories the first in
  sorted order is the base.
- **mlogit without a constant.** The LR test and pseudo R-squared use the
  equal-probability model `b = 0` as the null.
- **Ordered models with an offset.** The null model of the LR test keeps the
  offset (it is refitted by Newton-Raphson).
- **clogit.** `opg` uses group scores; `robust` clusters on the group; with
  fweights `G` is the sum of the group weights; `aic`/`bic` use the number of
  rows in the informative groups as N. With groups split across clusters
  (Stata's `nonest`) the per-row scores use within-group centred regressors;
  Stata's `nonest` result is believed to use the uncentred ones and then
  depends on the origin of the regressors.
- **biprobit boundary.** A failed or flat iteration with `|rho| > 0.995` is
  reported as `boundary_solution`; Stata would keep iterating or report a
  correlation of +-1 with missing standard errors.
- **LR tests under `opg`.** `hetprobit` and `biprobit` keep the LR form of
  `tests["lnsigma"]` / `tests["rho"]` under `opg` and switch to Wald only for
  `robust` and `cluster`.
- **bic with weights.** `N` is the sum of fweights, and the number of rows for
  the other weight types.
- **Convergence tolerance** is stricter than Stata's default, so estimates can
  differ from Stata's in the 6th-7th significant digit.
- **Separation.** OpenEconometrics raises `separation_detected` where Stata reports
  observations "completely determined" and continues.

## Performance

Wall-clock time of the complete call (sample construction, estimation,
covariance, result) on an Apple-silicon laptop CPU with one million rows and
ten regressors, float64:

| model | Newton iterations | 100,000 rows | 1,000,000 rows |
| --- | --- | --- | --- |
| `ologit`, 5 categories | 4 | 0.3 s | 1.8 s |
| `ologit`, clustered on 5,000 clusters | 4 | 0.2 s | 2.0 s |
| `oprobit`, 5 categories | 4 | 0.2 s | 1.4 s |
| `mlogit`, 5 categories (44 parameters) | 5 | 0.3 s | 2.3 s |
| `mlogit`, `robust` | 5 | 0.3 s | 2.3 s |
| `clogit`, groups of 5, several positives per group | 5 | 0.3 s | 2.9 s |
| `clogit`, the same, clustered on 5,000 clusters | 5 | 0.3 s | 2.9 s |
| `clogit`, choice sets of 5 (one positive) | 5 | 0.1 s | 1.1 s |
| `clogit`, choice sets, `robust` | 5 | 0.1 s | 1.2 s |
| `hetprobit`, 2 variance regressors | 5 | 0.3 s | 2.0 s |
| `biprobit`, 10 regressors per equation (23 parameters) | 4 | 0.5 s | 3.9 s |
| `biprobit`, `robust` | 4 | 0.4 s | 4.3 s |

Time grows linearly in the number of rows. A separated or boundary problem is
stopped after 20 to 30 iterations instead of running to the iteration limit.

No step loops over observations: likelihood sums are tensor reductions,
category and group sums use `index_add_`, the multinomial Hessian is a blocked
BLAS product and the conditional-logit recursion loops over positions within a
group only. The bivariate normal function evaluates its 20-node rule for all
observations at once (about 0.2 s per million evaluations for `|rho| < 0.925`).
