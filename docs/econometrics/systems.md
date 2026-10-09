# Systems of equations and further estimators: `sureg`, `mvreg`, `reg3`, `gmm`, `frontier`, `xtgls`, `xtpcse`

This page covers the estimators of Stata's `sureg`, `mvreg`, `reg3`, `gmm`,
`frontier`, `xtgls` and `xtpcse` (and the corresponding EViews system and SPSS
procedures). Everything is implemented in OpenEconometrics on float64 PyTorch tensors:
no statsmodels, linearmodels, SciPy optimizer or other estimation library runs
at fit time. Likelihoods have analytic scores and Hessians; GMM moment
Jacobians are differentiated symbolically from the moment expressions.

| Stata / EViews | OpenEconometrics |
| --- | --- |
| `sureg (y1 x1 x2) (y2 x1 x3)` | `oe.sureg(data=df, equations=[{"y": "y1", "x": ["x1", "x2"]}, {"y": "y2", "x": ["x1", "x3"]}])` |
| `sureg ..., isure` / `dfk` / `small` / `corr` | `iterate=True` / `dfk=True` / `small=True` / `result.tests["breusch_pagan"]` |
| `constraint 1 [y1]x1 = [y2]x1` + `sureg ..., constraints(1)` | `constraints=[{"terms": {"y1:x1": 1, "y2:x1": -1}, "value": 0}]` |
| `mvreg y1 y2 = x1 x2, corr` | `oe.mvreg(data=df, y=["y1", "y2"], x=["x1", "x2"], corr=True)` |
| `reg3 (y1 y2 z1) (y2 y1 z2), exog(z3)` | `oe.reg3(data=df, equations=[...], exogenous=["z3"])` |
| `reg3 ..., ireg3` / `2sls` / `ols` / `sure` | `ireg3=True` / `method="2sls"` / `method="ols"` / `method="sure"` |
| EViews `system.sur` / `system.3sls` | `oe.sureg(...)` / `oe.reg3(...)` |

The examples below use the variable names of Stata's example datasets
(`auto`, `klein`, `grunfeld`); every code block on this page was executed on
synthetic data with those column names.

## Specifying a system

Multi-equation estimators take a list of equations,

```python
equations = [
    {"y": "price", "x": ["foreign", "weight", "length"]},
    {"y": "mpg", "x": ["foreign", "weight"], "name": "fuel"},   # label "fuel"
    {"y": "displacement", "x": ["foreign", "weight"], "constant": False},
]
```

`name` labels the equation (default: its outcome) and `constant` switches the
equation's constant off (default: on). Coefficients are reported as
`label:term` (`price:weight`, `fuel:Intercept`) and grouped by equation in
`summary()`. Categorical regressors (`categorical=[...]`) are treatment coded
(`price:rep78[3]`). The estimation sample is casewise over every column of
every equation, as in Stata: with `missing="drop"` a row missing any variable
of the system is dropped from all equations.

**ModelSpec mapping.** A system is stored as an ordinary `ModelSpec`:
`outcome` is the first equation's dependent variable, the equations are the
JSON option `equations`, and every referenced column (outcomes, regressors,
instruments) is listed in the column role `system`, so that sample selection
and the data hashes see them. `oe.fit(result.spec, data=df)` refits the same
system.

## Numerical approach of the linear systems

The linear system estimators need the data only through cross products of
outcomes and regressors. OpenEconometrics gathers every distinct column once in
`W = [Z, other columns]` (instruments first, the constant in column 0) and
reduces it by **one Householder QR**, `sqrt(w) W = Q R`. Because
`W'diag(w)W = R'R`, every residual `e = W a` satisfies `e'diag(w)e = ||R a||^2`:
equation-by-equation OLS, the residual covariance, the stacked GLS of SUR and
the projections of 3SLS then run on the small `p x p` factor `R` instead of the
`N` rows. The accuracy is that of a QR least-squares fit on the data, and the
`NM`-row Kronecker system is never formed. Because the leading columns of `W`
span the instruments, the projection on them is simply the leading block of
rows of `R`: `||P_Z W a||^2 = ||R[:L] a||^2`.

The stacked GLS problem `min tr(S^-1 E'E)` is solved as one least-squares
problem: with `S = C C'` and `T = C^-1`, column `k` of the whitened residual
matrix `E T'` is `sum_{i<=k} T_ki e_i`, so block row `k` of the system holds
`T_ki X_i` in the columns of equation `i`. Its QR gives the estimate and the
conventional covariance `(X'(S^-1 kron A)X)^-1` (`A = I` for SUR, `P_Z` for
3SLS) together.

**Centring.** With the constant in column 0, `R[0, j] / R[0, 0]` is the weighted
mean of column `j`, and zeroing `R[0, j]` centres the column. Equations with a
constant are therefore fitted on centred regressors (well conditioned even when
a regressor is a calendar year or a price level of `1e7`) and the constants
are mapped back exactly, `b_0 = b_0c - m'b`, with `V = J V_c J'`.

**Collinearity.** Each equation is screened left to right on its compressed,
centred columns (Stata's rule); omitted terms are recorded as `label:term` in
`warnings` and `provenance["omitted_terms"]`. Collinear instruments of `reg3`
are dropped the same way. A column that is constant in the estimation sample
is detected exactly on the data (`max == min`) and centres to exact zeros, so
in an equation with a constant it is omitted as collinear with the constant
(in compressed form it would otherwise be the QR rounding residue of
`R[1:, j]`, which a screen that judges every column against its own length
cannot recognise); in an equation without a constant it is kept and plays the
constant's role. A constant outcome raises `constant_outcome`. A singular
residual covariance (two identical equations, an outcome fitted exactly)
raises `singular_sigma` or `perfect_fit`.

**Sample size.** Each equation needs `N > k_i`; the system may have fewer rows
than distinct columns (the compressed factor `R` is then `N x p`, upper
trapezoidal, and still reproduces every inner product).

**Timing.** A three-equation system on 1,000,000 rows with eight regressors
per equation takes about 0.3 s for two-step `sureg` (one QR pass over the data,
everything else on a 13 x 13 factor); iterating to the ML estimate adds almost
nothing because iterations never touch the rows again.

## `sureg`: seemingly unrelated regression

**Model.** `y_i = X_i b_i + e_i`, `i = 1..M`, on the same `N` units, with
`E[e_i e_j'] = s_ij I_N`: errors of one unit are correlated across equations,
errors of different units are not. Stacked, `Var(e) = Sigma kron I_N`.

**Estimator** (Methods and formulas of `[R] reg3`, of which `sureg` is the
SUR case):

1. OLS equation by equation, residuals `e_i`.
2. `s_ij = e_i'e_j / N`; with `dfk=True`, `s_ij = e_i'e_j / sqrt((N-k_i)(N-k_j))`.
3. Feasible GLS `b = (X'(S^-1 kron I)X)^-1 X'(S^-1 kron I)y`, conventional
   covariance `V = (X'(S^-1 kron I)X)^-1`.
4. `iterate=True` (Stata's `isure`) repeats 2-3 on the GLS residuals until
   `max_j |b_j - b_j,old| / (|b_j,old| + 1) <= tolerance` (default `1e-9`); the
   limit is the Gaussian maximum-likelihood estimator. When an outcome is also
   a regressor of another equation (a simultaneous system treated as
   exogenous) that likelihood is unbounded and the iteration heads for a
   singular `Sigma`; this is refused with `singular_sigma` and an explanation
   (use the two-step estimator or `reg3`).

**Inference.** z statistics and Wald chi2 tests by default; `small=True` gives
Student t with the first equation's `N - k_1` degrees of freedom (Stata: "the
degrees of freedom for the t statistics are all taken to be those for the
first equation") and F tests (`df2 = N - k_i` for equation `i`). `small` does
not change the covariance divisor; combine it with `dfk` for the
small-sample divisor (see Uncertain conventions).

**Reported.**

- `metrics`: `n_equations`, `<label>:rmse = sqrt(e_i'e_i / d_ii)` (divisor `N`,
  or `N - k_i` with `dfk`) and `<label>:r_squared = 1 - RSS_i/TSS_i` from the
  final residuals (centred TSS with a constant, uncentred without),
  `log_likelihood = -N/2 (M (1 + ln 2 pi) + ln|E'E/N|)` at the final residuals
  (the maximized log likelihood for `iterate=True`), `df_model` (all slopes),
  `df_resid` (`N - k_1`).
- `tests`: `model` (Wald test of every slope of the system), `<label>:model`
  (each equation's slopes; Stata's per-equation chi2/F column) and
  `breusch_pagan`: `N sum_{i>j} r_ij^2 ~ chi2(M(M-1)/2)`, with `r_ij` the
  correlations of the residual covariance used by the final GLS step (the OLS
  residuals of the two-step estimator, the converged ones when iterated).
- `extra`: `equations` (per-equation records: terms, parameters, slopes, RMSE,
  R-squared, RSS, test), `sigma` (`e(Sigma)`), `correlation`, `iterations`.

**Constraints.** `constraints=[{"terms": {"y1:x1": 1, "y2:x1": -1}, "value": 0}]`
imposes `R b = r` over the reported terms (validated and reduced by the same
helpers as `oe.cnsreg`: redundant constraints are dropped with a warning,
inconsistent ones are an error). The first OLS step is unconstrained; the GLS
step is the exact reparameterization `b = b_p + Q_2 g`. Coefficients fixed
completely by the constraints are listed in `extra["constrained_terms"]`.

**Weights.** `aweight` (rescaled to sum to `N`) and `fweight` (rows replicated:
results equal the expanded data set, `N = sum f`). `covariance` is
`"nonrobust"` only; Stata's `sureg` offers conventional, bootstrap and
jackknife variances.

```python
import openecon as oe

eqs = [{"y": "price", "x": ["foreign", "weight", "length"]},
       {"y": "mpg", "x": ["foreign", "weight"]},
       {"y": "displacement", "x": ["foreign", "weight"]}]
fit = oe.sureg(data=auto, equations=eqs)
print(fit.summary())
print(fit.tests["breusch_pagan"])                     # sureg ..., corr
ml = oe.sureg(data=auto, equations=eqs, iterate=True)  # sureg ..., isure
same = oe.sureg(data=auto, equations=eqs, constraints=[
    {"terms": {"mpg:weight": 1, "displacement:weight": 1}, "value": 0}])
```

## `mvreg`: multivariate regression

**Model.** `y_j = X b_j + e_j` for every outcome `j`, with the **same**
regressors `X` (`k` columns including the constant) and errors correlated
across equations. GLS equals OLS equation by equation in that case, so the
coefficients are the OLS ones; `mvreg` adds their joint covariance

    Cov(b_i, b_j) = s_ij (X'X)^-1,     s_ij = e_i'e_j / (N - k)

(computed in the general form `s_ij (X_i'X_i)^-1 X_i'X_j (X_j'X_j)^-1`), which
makes tests across equations possible. Student t with `N - k` degrees of
freedom; `<y>:model` is each equation's classical regression F test,
`<y>:rmse = sqrt(RSS/(N-k))`. `corr=True` adds the Breusch-Pagan test of
independence (`extra["correlation"]` always holds the residual correlations).

```python
fit = oe.mvreg(data=auto, y=["headroom", "trunk", "turn"],
               x=["price", "mpg", "displacement"], corr=True)   # mvreg ..., corr
```

## `reg3`: three-stage least squares

**Model.** Structural equations whose right-hand sides may contain endogenous
variables, errors correlated across equations, and exogenous instruments `Z`
uncorrelated with every error.

**Which variables are endogenous** (Stata's rules). Every equation's outcome is
endogenous; `endogenous=[...]` adds right-hand-side variables that are no
equation's outcome; every other right-hand-side variable is exogenous, and
`exogenous=[...]` adds instruments that appear in no equation.
`instruments=[...]` instead gives the complete exogenous list (Stata's
`inst()`): every right-hand-side variable not in it is endogenous. The constant
is an instrument whenever an equation has a constant.

**Estimators** (`method`, Methods and formulas of `[R] reg3`):

| `method` | estimator | covariance |
| --- | --- | --- |
| `"3sls"` (default) | 2SLS `b_i = (X_i'P_Z X_i)^-1 X_i'P_Z y_i`, `S` from the structural 2SLS residuals (`N` or `dfk` divisor), then `b = [X'(S^-1 kron P_Z)X]^-1 X'(S^-1 kron P_Z)y` | `[X'(S^-1 kron P_Z)X]^-1` |
| `"3sls"`, `ireg3=True` | `S` recomputed from the 3SLS residuals until the coefficients converge | same, at the final `S` |
| `"2sls"` | equation-by-equation 2SLS; implies `dfk`, `small` and independent equations | block diagonal `s_ii (X_i'P_Z X_i)^-1`, `s_ii = e_i'e_i/(N-k_i)` |
| `"ols"` | equation-by-equation OLS, all variables exogenous; implies `dfk`, `small`, independent equations | block diagonal `s_ii (X_i'X_i)^-1` |
| `"sure"` | SUR, all right-hand-side variables exogenous (`ireg3` iterates) | as `sureg` |

The residuals are always the structural ones, `e_i = y_i - X_i b_i` with the
observed regressors, so `R^2 = 1 - RSS/TSS` may be negative. Every equation
must satisfy the rank condition (`P_Z X_i` of full column rank, which needs at
least as many excluded instruments as endogenous regressors); otherwise
`underidentified` names the equation. `method="2sls"` reproduces
`oe.ivregress(..., small=True)` equation by equation.

```python
fit = oe.reg3(data=klein, equations=[
    {"y": "consump", "x": ["wagepriv", "wagegovt"]},
    {"y": "wagepriv", "x": ["consump", "govt", "capital1"]}])
print(fit.extra["endogenous"], fit.extra["instruments"])
```

## `gmm`: generalized method of moments

**Model.** Moment conditions `E[z_ij u_j(x_i; b)] = 0` for one or more residual
expressions `u_j` and their instruments `z_ij`. `L = sum_j L_j` moments identify
`P <= L` parameters.

**Moment expressions** use the grammar of `oe.nl`: parameters in braces
(`{b0}`, or `{b0=1.5}` with a starting value), column names, numbers,
`+ - * / ^`, parentheses and the functions `exp`, `ln`/`log`, `sqrt`, `abs`,
`sin`, `cos`, `tan`, `expit`/`invlogit`, `normal`, `normalden`. A parameter that
appears in several equations is one parameter. The text is parsed into a
syntax tree and checked node by node (it is never executed as code), and the
Jacobian `du/db'` is propagated analytically in forward mode, so no numerical
derivatives are taken. Equations are labelled `1, 2, ...`, or by the keys of a
dict `{"demand": "...", "supply": "..."}`. Instruments are one list common to
every equation or one list per equation; the constant is added to each unless
`instrument_constant=False` (Stata's `instruments(..., noconstant)`).
Collinear instruments are dropped and recorded (`label:term`).

**Estimator** (Methods and formulas of `[R] gmm`). `b` minimizes
`Q(b) = g(b)'W g(b)` with `g = sum_i w_i m_i`, `m_i = (z_i1 u_i1, ..., z_iM u_iM)`,
by Gauss-Newton with Levenberg-Marquardt damping on the analytic moment
Jacobian `G = dg/db'`. With `W = S^-1`, `S = C C'`, the criterion is the
nonlinear least-squares problem `||C^-1 g||^2`, and each step solves
`min_d ||C^-1 (g + G d)||^2 + mu ||D d||^2` (`D` the column norms) by
Householder QR, which works with the condition number of `C^-1 G` rather than
its square (the normal equations `G'WG`); `mu` starts at 0, so a linear model
is solved in one step. Iterations stop when the Gauss-Newton predicted
decrease is below `tolerance^2 Q` or when the step and the criterion have
settled; changes of `Q` smaller than its rounding level
(`2 sqrt(Q) nu + nu^2`, `nu^2 = eps^2 tr(W S_a)`, `S_a` the moment covariance
built from the magnitudes `|u_i| + |du_i/db'||b|` of the terms that cancel
inside each residual) are treated as noise.

**Instrument bases.** Each equation's instruments are replaced internally by
the orthonormal basis `Q_j = Z_j R_j^-1` (`sqrt(w) Z_j = Q R`). The two-step,
iterated and every sandwich formula, and `J`, are invariant to such a change
of basis, so results are unchanged in exact arithmetic, but the moment
covariance `S` stays well conditioned when an instrument has a large offset
(a calendar year, `x + 1e6`): the linear IV model with an offset of `1e6`
reproduces the unshifted slopes, standard errors and `J` to about `1e-8`. The
identity weight refers to the original instruments (`g_Z'g_Z = ||R'g_Q||^2`)
and is applied exactly in that form.

| setting | weight matrix |
| --- | --- |
| first step | `winitial="identity"` (Stata's default) or `"unadjusted"`: `blockdiag (Z_j'Z_j)^-1`, i.e. 2SLS for linear moments |
| `twostep=True` (default) | `W = S(b_1)^-1` |
| `twostep=False` | the first step is the estimate (one-step GMM) |
| `igmm=True` | `W` recomputed from the latest estimates until `max_j |b_j - b_j,old| / (|b_j,old| + 1) <= igmm_tolerance` |

The moment covariance `S` (sum form, no small-sample factors) of type
`wmatrix`: `"robust"` `sum_i w_i^2 m_i m_i'` (default); `"cluster"` the outer
products of the cluster sums (needs `cluster=`); `"hac"` the kernel-weighted
autocovariances with `lags` and `kernel` (Bartlett default), in `time` order
within `panel` when given; `"unadjusted"` `S_jk = s_jk Z_j'Z_k` with
`s_jk = u_j'u_k / N`. `center=True` removes the mean moment first.

**Covariance.** With `B = (G'WG)^-1` at the final estimates and weight matrix:
when the estimator reweights (two-step or iterated) and the covariance type
equals the weight-matrix type (`nonrobust` pairs with `unadjusted`), the
efficient form `V = B` is reported, as `ivregress gmm` does; otherwise the
sandwich `V = B G'W S_hat W G B` with `S_hat` of the covariance type at the final
estimates (always for the one-step estimator). Cluster covariances carry
`G/(G-1)`. The covariance defaults to the weight-matrix type (`robust`).
Coefficient tests are z tests. `pweight`s need a robust or cluster covariance.

**Reported.** Terms are the parameter names. `metrics`: `j` (Hansen's J),
`criterion` (Stata's `e(Q) = J / N`), `n_moments`, `n_parameters`,
`n_equations`, `gmm_steps`, `iterations`. `tests["hansen_j"]`:
`J = g'Wg ~ chi2(L - P)` for the two-step and iterated estimators of an
overidentified model (not after one step, whose weight matrix is not
efficient). `extra`: the moments, the instrument terms of every equation,
starting values and settings.

**Linear IV special case.** A linear moment with `winitial="unadjusted"`
reproduces `oe.ivregress(..., method="gmm")` to rounding (estimates, robust,
cluster, HAC and iterated covariances, J):

```python
fit = oe.gmm(data=df, moments=["y - {b0} - {b1}*x1 - {b2}*x2"],
             instruments=["x1", "z1", "z2"], winitial="unadjusted")
# Stata: gmm (y - {b0} - {b1}*x1 - {b2}*x2), instruments(x1 z1 z2) winitial(unadjusted)

pois = oe.gmm(data=df, moments=["docvis - exp({b0} + {b1}*private + {b2}*income)"],
              instruments=["private", "income", "female"], start={"b0": 1})
two = oe.gmm(data=df, moments={"demand": "q - {a0} - {a1}*p", "supply": "q - {c0} - {c1}*p - {c2}*w"},
             instruments=[["w", "z"], ["w", "z"]])
```

Errors: `invalid_formula`, `invalid_start` (unknown parameters, or residuals not
finite at the starting values), `underidentified` (fewer moments than
parameters), `not_identified` (rank-deficient moment Jacobian, named
parameters), `singular_weight_matrix` (for example fewer clusters than
moments), `perfect_fit` (the residuals of a moment equation vanish at the
estimates, `sum w u_j^2 <= 1e-24 min_c sum w c^2` over the equation's data
columns `c`: the moment covariance and every standard error would be zero,
for example a constant outcome), `nonconvergence`.

## `frontier`: stochastic production and cost frontiers

**Model.** `y_i = x_i'b + v_i - s u_i` with `s = 1` for a production frontier
(output falls short of the frontier) and `s = -1` for a cost frontier
(`cost=True`: cost exceeds the frontier), noise `v ~ N(0, sigma_v^2)` and
one-sided inefficiency `u >= 0` independent of `v`. With `e_i = y_i - x_i'b`,
`sigma^2 = sigma_u^2 + sigma_v^2` and `lambda = sigma_u / sigma_v`, the
per-observation log likelihoods are (Stata `[R] frontier`, Methods and
formulas):

| `distribution` | inefficiency | log density `l_i` | estimated ancillaries |
| --- | --- | --- | --- |
| `"hnormal"` (default) | `N+(0, sigma_u^2)` | `1/2 ln(2/pi) - ln sigma + ln Phi(-s e_i lambda/sigma) - e_i^2/(2 sigma^2)` | `/lnsig2v`, `/lnsig2u` |
| `"exponential"` | exponential with mean `sigma_u` | `-ln sigma_u + sigma_v^2/(2 sigma_u^2) + s e_i/sigma_u + ln Phi(-s e_i/sigma_v - sigma_v/sigma_u)` | `/lnsig2v`, `/lnsig2u` |
| `"tnormal"` | `N+(mu, sigma_u^2)` | `-1/2 ln(2 pi) - ln sigma - (s e_i + mu)^2/(2 sigma^2) + ln Phi(mu*_i/sigma*) - ln Phi(mu/sigma_u)` | `/mu`, `/lnsigma2`, `/ilgtgamma` |

with `mu*_i = (mu sigma_v^2 - s e_i sigma_u^2)/sigma^2`,
`sigma*^2 = sigma_u^2 sigma_v^2/sigma^2` and, for the truncated normal, Stata's
parameterization `ln sigma^2` and `logit(gamma)`, `gamma = sigma_u^2/sigma^2`.

**Estimation.** Newton-Raphson on the analytic score and Hessian. Every log
density depends on `b` only through `e_i`, so with the per-observation
derivatives `l_e, l_ee, l_a, l_ea, l_aa` (`a` the ancillaries) the score is
`(-X'(w l_e), sum w l_a)` and the Hessian blocks are `X'diag(w l_ee)X`,
`-X'(w l_ea)` and `sum w l_aa` (O(n k^2); `ln Phi` and the Mills ratio are
evaluated stably in both tails). The derivatives are checked against
numerical ones in the tests. Regressors are centred at their means during the
iteration (the constant is mapped back exactly) and the iteration starts at
OLS with method-of-moments variances from the second and third moments of the
OLS residuals (the constant shifted by `s E[u]`), the corrected-OLS start.

**Boundary solutions are refused.** If the OLS residuals have the wrong
skewness (positive for a production frontier, negative for a cost frontier),
the half-normal and exponential likelihoods are maximized at `sigma_u = 0`
(Waldman 1982): `boundary_solution` is raised with that explanation instead of
reporting a degenerate fit. The same code is raised when the fit converges to
`sigma_u^2/sigma_v^2` or `sigma_v^2/sigma_u^2` below `1e-8`, or (tnormal) to `gamma`
within `1e-10` of 0 or 1, and when a run that did not converge is drifting to
the edge of the parameter space: a variance ratio (or `gamma`) beyond `1e-6`,
or, for the truncated normal, `mu/sigma_u < -10` (the truncated normal then
degenerates to the exponential distribution and the likelihood has no
interior maximum; use `distribution="exponential"`). A constant outcome raises
`constant_outcome` and an exact OLS fit `perfect_fit`; a truncated-normal run
that fails to converge on wrong-skewed residuals says so in its
`nonconvergence` message.

**Covariance.** `nonrobust` (observed information, default), `opg`, `robust`
(`N/(N-1)` sandwich), `cluster` (`G/(G-1)`), all through `core.ml_covariance`.
Weights: `fweight` (results equal the expanded data), `pweight` (robust by
default), `iweight`. z inference.

**Reported.**

- `metrics`: `log_likelihood`, `aic`, `bic`, `sigma_v`, `sigma_u`, `sigma2`,
  `lambda` (hnormal, exponential) or `sigma2`, `gamma`, `sigma_u2`, `sigma_v2`
  (tnormal), `log_likelihood_ols`, `df_model`.
- `extra["ancillary"]`: the same derived quantities with delta-method
  standard errors and confidence intervals (Stata's table footer).
- `tests["model"]`: Wald chi2 of the slopes. `tests["sigma_u"]`: the
  likelihood-ratio test of `sigma_u = 0` against OLS (whose maximized log
  likelihood is `-W/2 (ln(2 pi s^2) + 1)`, `s^2 = sum w e^2 / W`), distributed as
  the mixture `chibar2(01)`: `p = Pr(chi2(1) > LR)/2`. Reported for the
  `nonrobust` and `opg` covariances without pweights, as Stata does.

**Efficiency.** `oe.frontier_efficiency(result, data)` rebuilds the estimation
sample (it must match the stored sample positions) and returns a table with,
per observation, `row`, `residual`, `u = E[u|e] = sigma* (phi(z)/Phi(z) + z)`
(`z = mu*/sigma*`, Jondrow, Lovell, Materov and Schmidt 1982), `u_mode = max(mu*, 0)`
and `te = E[exp(-s u)|e]` (Battese and Coelli 1988), computed in logs as
`exp(-s mu* + sigma*^2/2) Phi(z - s sigma*) / Phi(z)`. The conditional moments
use `u | e ~ N+(mu*, sigma*^2)` with, for the half-normal,
`mu* = -s e sigma_u^2/sigma^2`; exponential `mu* = -s e - sigma_v^2/sigma_u`,
`sigma* = sigma_v`; truncated normal as above.

```python
fit = oe.frontier(data=df, y="lnoutput", x=["lncapital", "lnlabor"])   # frontier lnoutput lncapital lnlabor
print(fit.summary(), fit.tests["sigma_u"])
cost = oe.frontier(data=df, y="lncost", x=["lnq", "lnpl", "lnpk"], cost=True,
                   distribution="exponential")                         # ..., cost distribution(exponential)
eff = oe.frontier_efficiency(fit, df)                                   # predict u, u / predict te, te
print(eff["te"].mean())
```

## `xtgls`: panel feasible GLS

**Model.** `y_it = x_it'b + e_it` for panels `i = 1..m` observed over periods
`t` (`N = sum_i T_i`), with a non-spherical error covariance `Omega`:

| `panels` | assumption | estimate (from the residuals `e`) |
| --- | --- | --- |
| `"iid"` (default) | `Omega = sigma^2 I` | `sigma^2 = e'e/N` |
| `"heteroskedastic"` | `Var(e_it) = sigma_i^2`, panels independent | `sigma_i^2 = e_i'e_i/T_i` |
| `"correlated"` | `E[e_it e_jt] = sigma_ij` (`Omega = Sigma kron I_T`) | `sigma_ij = e_i'e_j/T`; balanced panels, `T >= m` |

and, with `corr="ar1"` (common) or `"psar1"` (panel specific),
`e_it = rho_i e_i,t-1 + u_it` within panels.

**Estimator** (Methods and formulas of `[XT] xtgls`):

1. OLS. With AR(1): `rho_i` from the OLS residuals of each panel by `rhotype`
   (table below; the common `rho` of `ar1` is the `T_i - 1`-weighted mean of
   the panel coefficients, the plain mean for balanced panels), then the
   Prais-Winsten transformation `z*_i1 = sqrt(1 - rho_i^2) z_i1`,
   `z*_it = z_it - rho_i z_i,t-1` of `y` and of every column of `X` (constant
   included), and OLS on the transformed data.
2. The variance structure from those residuals (table above).
3. GLS on the transformed data, `b = (X*'Omega^-1 X*)^-1 X*'Omega^-1 y*` with
   `V = (X*'Omega^-1 X*)^-1`, solved by Householder QR after whitening: weights
   `1/sigma_i^2` for heteroskedastic panels, `C^-1` (with `Sigma = C C'`)
   applied across the panels in every period for correlated panels; `Omega`
   is never formed.
4. `igls=True` repeats 1-3 from the GLS estimates (with `rho` re-estimated
   from the GLS residuals) until `max_j |b_j - b_j,old| / (|b_j,old| + 1) <=
   tolerance` (`1e-7`); without autocorrelation the limit is the Gaussian ML
   estimator.

| `rhotype` | estimator of `rho_i` (sums over the consecutive pairs of panel `i`, `k` regressors) |
| --- | --- |
| `"regress"` (default) | `sum e_t e_{t-1} / sum e_{t-1}^2` |
| `"freg"` | `sum e_t e_{t-1} / sum e_t^2` |
| `"tscorr"` | `sum e_t e_{t-1} / sum_all e_t^2` |
| `"dw"` | `1 - DW/2` |
| `"theil"` | `tscorr (T_i - k)/T_i` |
| `"nagar"` | `(dw T_i^2 + k^2)/(T_i^2 - k^2)` |

AR(1) requires consecutive periods inside every panel (`time_gaps`
otherwise) and an estimate inside `(-1, 1)` (`invalid_rho`: use the bounded
`tscorr` or `dw`). A panel observed only once has no autocorrelation; it is
left out of the common `rho` (also by `xtpcse` with `np1=True`, where its
weight `T_i = 1` would otherwise bring an undefined coefficient into the
average) and its single row is transformed by `sqrt(1 - rho^2)`. **Reported:** z inference, `tests["model"]` (Wald chi2 of
the slopes); `metrics`: `n_groups`, `n_periods`, `obs_per_group_min/avg/max`,
`estimated_covariances` (`1`, `m` or `m(m+1)/2`), `estimated_autocorrelations`
(`0`, `1` or `m`), `estimated_coefficients`, `log_likelihood` (models without
autocorrelation: the Gaussian log likelihood at the final estimates with the
variance structure of their residuals, e.g.
`-1/2 [N (ln 2 pi + 1) + T ln|Sigma|]` for correlated panels), `rho` (`ar1`);
`extra`: `rho` per panel, `sigma2` / the panel variances / `Sigma`.
`covariance` is `"nonrobust"` (the GLS covariance) only; weights are not
supported.

```python
fit = oe.xtgls(data=grunfeld, y="invest", x=["mvalue", "kstock"], panel="company",
               time="year", panels="correlated", corr="ar1")
# xtset company year ; xtgls invest mvalue kstock, panels(correlated) corr(ar1)
```

## `xtpcse`: panel-corrected standard errors

**Estimator.** OLS (`correlation="independent"`) or, with `"ar1"` / `"psar1"`,
Prais-Winsten regression with `rho` from the OLS residuals (`rhotype` as for
`xtgls`; the common `rho` weights the panels by `T_i - 1`, by `T_i` with
`np1=True`, as Stata documents).

**Covariance** (Beck and Katz 1995; `[XT] xtpcse`):
`V = (X'X)^-1 X'Omega X (X'X)^-1`, `Omega = Sigma kron I_T`, from the
(transformed) OLS residuals:

| option | `Sigma` |
| --- | --- |
| default (casewise) | `sigma_ij = sum_{t in C} e_it e_jt / |C|` over the periods `C` observed in every panel |
| `pairwise=True` | `sigma_ij = sum_{t in C_ij} e_it e_jt / |C_ij|` over the periods panels `i` and `j` share |
| `hetonly=True` | diagonal: `sigma_ii = e_i'e_i / T_i` (casewise: over the common periods) |
| `independent=True` | `sigma^2 I`, `sigma^2 = e'e/N` (the OLS covariance with divisor `N`) |

For unbalanced panels the meat is `X'Omega X = sum_t X_t' Sigma_t X_t` over the
panels observed in period `t`; the coefficients always use every observation.
The meat is accumulated without `Omega`: casewise as
`(1/|C|) sum_{s in C, t} h_st h_st'` with `h_st = sum_i e_is x_it` (when there are
fewer common periods than panels), otherwise as `sum_t X_t' Sigma X_t` on the
dense period x panel grid; `hetonly` and `independent` are weighted cross
products. A pairwise `Sigma` need not be positive semidefinite; an indefinite
result raises `invalid_covariance`.

**Reported.** z inference, `tests["model"]` (Wald chi2 of the slopes),
`metrics`: `r_squared` (`1 - RSS/TSS` of the estimated, possibly transformed,
regression; TSS centred), the panel structure and estimated-parameter counts,
`rho` (`ar1`); `extra`: `rho` per panel, `sigma_matrix` / `sigma_variances`,
`common_periods`. `covariance` is `"robust"` (the panel-corrected family) only.

```python
fit = oe.xtpcse(data=grunfeld, y="invest", x=["mvalue", "kstock"], panel="company",
                time="year", correlation="ar1")         # xtpcse invest mvalue kstock, correlation(ar1)
unb = oe.xtpcse(data=df, y="y", x=["x"], panel="id", time="t", pairwise=True)
```

## Failure contract

Every invalid input or numerical failure raises `AnalysisError(code, message)`:
`invalid_spec` (malformed equations, unknown options, conflicting lists),
`missing_columns`, `missing_values` (with `missing="raise"`), `empty_data`,
`empty_sample`, `non_numeric_column`, `non_finite_values`,
`negative_weights`, `noninteger_frequency_weights`,
`insufficient_observations`, `empty_equation`, `constant_outcome`,
`perfect_fit` (linear systems, `gmm`, `frontier`, `xtgls`, `xtpcse`),
`singular_sigma` (redundant equations, or an iterated estimator whose
criterion is unbounded), `underidentified` (3SLS rank condition, too few GMM
moments), `not_identified` (rank-deficient GMM Jacobian), `invalid_formula`,
`invalid_start`, `singular_weight_matrix`, `insufficient_clusters`,
`unsupported_covariance` (pweights with a non-robust covariance),
`nonconvergence`, `boundary_solution` (frontier), `unbalanced_panel`,
`insufficient_periods`, `time_gaps`, `invalid_time`, `invalid_rho`,
`repeated_time_values`, `no_common_periods`, `invalid_covariance`,
`design_too_large` (xtpcse period x panel grid), `invalid_constraint`,
`inconsistent_constraints`, `invalid_result`, `sample_mismatch`. Dropped
collinear terms and instruments, redundant constraints and coefficients fixed
by constraints are recorded in `warnings` (and `provenance["omitted_terms"]`).

## Verification

`provenance["stata_parity_validated"]` is `False` for every estimator on this
page: no comparison with Stata output has been made. The tests compare every
estimator, option, covariance and weight type with independent oracles:

- `tests/test_econ_systems_oracle.py`: SUR and 3SLS from the explicit stacked
  system `blockdiag(X_i)` with the full Kronecker weight `Sigma^-1 kron D`
  (`D = diag(w)`, or the weighted projection on the instruments), unweighted,
  aweighted and fweighted, with `dfk`/`small`; iterated SUR against a SciPy
  maximization of the full Gaussian likelihood and the NumPy fixed point;
  restricted GLS for constraints; mvreg and the `2sls`/`ols` methods against
  statsmodels WLS/OLS/IV2SLS; invariances (fweights equal duplicated rows,
  row order, rescaled regressors, aweight scale, identical regressors give
  OLS, exactly identified 3SLS equals 2SLS, `method="sure"` equals `sureg`),
  missing data, categorical, collinear, tiny and badly scaled designs. The
  `statsmodels.sandbox.sysreg.SUR` class does not run under NumPy 2 and is not
  used.
- `tests/test_econ_systems_oracle_ml.py`: linear GMM (two-step, one-step
  identity sandwich, unadjusted, cluster, centred, HAC in time order,
  iterated) from explicit algebra under every weight type; nonlinear and
  two-equation GMM by brute-force SciPy minimization with finite-difference
  Jacobians; frontier for all three distributions, production and cost, by
  SciPy maximization of independently written likelihoods (numerical
  Hessians and scores for the nonrobust, robust, OPG and cluster covariances,
  delta-method ancillaries, the LR test), fweights equal duplicated rows,
  pweights, and the efficiency scores by numerical integration of the
  conditional density of `u` given `e`.
- `tests/test_econ_systems_oracle_panel.py`: xtgls with the explicit `N x N`
  `Omega` and block-diagonal Prais-Winsten matrix for every
  `panels` x `corr` combination and every `rhotype`, unbalanced panels, the
  iterated estimator against a brute-force Gaussian likelihood; the
  Beck-Katz covariance with `Omega` assembled element by element (casewise,
  pairwise, hetonly, independent; balanced and unbalanced; Prais-Winsten with
  and without `np1`).
- `tests/test_econ_systems_adversarial.py`: the failure contract above,
  extreme magnitudes (`1e-8`, `1e8`, offsets of `1e6`-`1e7`) as exact
  rescalings, single panels, fewer rows than system columns, options at
  their bounds, JSON round trips and refits of every estimator.
- `tests/test_econ_systems_{sur,gmm,frontier,xtgls}.py`: the implementation's
  own tests, including the analytic frontier derivatives and the GMM moment
  Jacobian against numerical ones, and linear GMM against `oe.ivregress`.

## Performance

Measured on one CPU core of the development machine (in-memory synthetic
data, one million rows unless noted; second runs, so the first-call import
cost is excluded):

| model | time |
| --- | --- |
| `sureg`, 3 equations x 8 regressors (two-step / iterated) | 0.3 s / 0.3 s |
| `mvreg`, 3 outcomes x 10 regressors | 0.3 s |
| `reg3` (3SLS), 2 equations, 10 instruments | 0.2 s |
| `gmm`, linear IV with 11 parameters, two-step | 0.8 s |
| `gmm`, exponential mean with 3 parameters, two-step | 0.3 s |
| `frontier` hnormal / exponential / tnormal, 10 regressors | 0.5 / 0.6 / 0.7 s |
| `xtgls`, heteroskedastic panels with panel-specific AR(1), 1000 panels x 1000 periods | 0.2 s |
| `xtgls`, heteroskedastic panels, iterated, 1000 x 1000 | 0.2 s |
| `xtgls`, correlated panels with AR(1), 100 panels x 2000 periods | 0.05 s |
| `xtpcse`, full Sigma with AR(1) / hetonly / pairwise, 1000 x 1000 | 0.2 / 0.2 / 0.2 s |

The linear systems pass over the rows once (one QR of the distinct columns);
GMM costs `O(N L P + N L^2)` per Gauss-Newton evaluation; the frontier Hessian
is `O(N k^2)`; `xtgls` and `xtpcse` work on the period x panel grid without
ever forming `Omega`. No `N x N` object is built anywhere.

## Limitations

- `sureg`, `reg3` and `mvreg` report the conventional covariance only
  (Stata's bootstrap and jackknife variances are not implemented); `reg3`
  has no `allexog`, `corr()`, `dfk2` or `first` options; `sureg` has no
  `dfk2`.
- `gmm` implements residual-expression moments with instruments; Stata's
  panel-style instruments (`xtinstruments`), user-supplied derivatives,
  `winitial(xt ...)`, `wmatrix(..., independent)`, `nocommonesample` and
  moment-evaluator programs are not implemented. Starting values default to
  zero; nonlinear models usually need informed `start` values.
- `frontier` has no `uhet()`/`vhet()` heteroskedasticity equations and no
  `cm()` conditional-mean model for the truncated normal.
- `xtgls` and `xtpcse` take no weights; AR(1) models need gap-free panels;
  `xtgls` has no `force` option and `panels="correlated"` needs balanced
  panels (as in Stata).

## Uncertain conventions

These choices follow the textbook-standard reading where the vendor manual
could not be checked line by line; they are flagged so that a parity
comparison can confirm or correct them (confidence in brackets):

- `sureg`/`reg3` `small`: only the reference distributions change (t with
  the first equation's `N - k_1` degrees of freedom, per-equation F with
  `df2 = N - k_i`); the residual covariance divisor changes only with `dfk`
  (medium: Stata's text "the standard errors from each equation are computed
  using the degrees of freedom for the equation" could also mean a rescaling
  under `small`). Per-equation RMSE uses the divisor of `Sigma` (`N`, or
  `N - k_i` with `dfk`) (high for `N`: the `[R] sureg` auto example is
  consistent with `sqrt(RSS/N)`, its RMSE being the OLS Root MSE of the same
  equation times about `sqrt((N-k)/N)`; medium for `dfk`).
- `sureg` Breusch-Pagan test and `extra["sigma"]` use the residual
  covariance of the final GLS step (the OLS residuals for two-step, the
  converged residuals when iterated) (medium); `log_likelihood` is evaluated
  at the final residuals with divisor `N` for both (medium for two-step,
  high for `iterate=True`).
- Constrained `sureg`/`reg3`: the first step is unconstrained and `dfk` uses
  the unconstrained parameter counts `k_i` (low).
- `gmm`: the default first-step weight matrix is the identity (medium: the
  `[R] gmm` default may instead be `winitial(unadjusted)`; pass
  `winitial="unadjusted"` to reproduce `ivregress gmm`). When the covariance
  type equals the weight-matrix type after a two-step or iterated estimator,
  the efficient form `(G'WG)^-1` with the weight matrix of the last step is
  reported (this reproduces the iv family's `ivregress gmm`); Stata's `gmm`
  may instead use the sandwich with `S` recomputed at the final estimates
  (asymptotically equivalent) (medium). Cluster covariances carry `G/(G-1)`
  (low); the cluster weight matrix and `J` do not. `winitial="unadjusted"`
  with several equations uses a unit residual covariance
  (`blockdiag (Z_j'Z_j)^-1`) (medium). `center=True` centres the moments in
  the covariance as well as in the weight matrix (low).
- `frontier`: the LR test of `sigma_u = 0` is also reported for the truncated
  normal (where `mu` is unidentified under the null and the `chibar2(01)`
  reference is conservative) (medium); the confidence intervals of the
  derived quantities in `extra["ancillary"]` are symmetric delta-method
  intervals, whereas Stata may transform the interval of the estimated
  parameter for `sigma_v`/`sigma_u` (low); for a cost frontier
  `te = E[exp(u)|e] >= 1`, following Stata's definition `E[exp(-s u)|e]`
  (high).
- `xtgls`: the common AR(1) coefficient is the `T_i - 1`-weighted mean of
  the panel coefficients (the plain mean for balanced panels; Stata documents
  this weighting for `xtpcse`, for `xtgls` it is assumed) (medium); `igls`
  re-estimates `rho` from the GLS residuals in every iteration (medium);
  `log_likelihood` is reported only without autocorrelation, as the Gaussian
  log likelihood at the final estimates with the variance structure of their
  residuals (medium).
- `xtpcse`: casewise `Sigma` uses the periods common to all panels for every
  element (also the diagonal with `hetonly`), while the coefficients use all
  observations (medium); `R^2` of a Prais-Winsten fit is that of the
  transformed regression with a centred total sum of squares (low);
  `independent` uses `sigma^2 = e'e/N` (medium).

Shared smooth mean parameters with full joint Gaussian ML uncertainty are available through [nonlinear SUR](nonlinear-sur.md).
