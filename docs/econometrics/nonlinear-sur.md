# Bounded Gaussian nonlinear SUR

`nlsur` fits exogenous nonlinear mean equations with globally shared named
parameters and an unrestricted positive-definite residual covariance. It uses
native CPU float64 tensors. The five public helpers are `nlsur`,
`nlsur_restore`, `nlsur_predict`, `nlsur_margins` and `nlsur_contrast`.

The eight delivery scopes are MARKET-672–679: joint ML, full physical OIM,
original-observation HC0, cluster CR0, sealed numerical restoration, saved mean
and conditional prediction, continuous effects and elasticities, and joint
smooth contrasts/Wald inference. The broader MARKET-178 stays open.

## Model and specification

For each original observation, the outcome vector follows
`y_i = f(x_i, beta) + epsilon_i`. The working Gaussian model uses one common
residual covariance `Sigma` across observations. Different equations may use
different exogenous columns and share any named mean parameter. One residual
vector, rather than each equation row, is the unit of the joint likelihood.

```python
fit = oe.nlsur(data, [
    {"y": "y1", "name": "first",
     "formula": "{b0=2}+{b1=.3}*x+{amp=1.2}*exp({rate=.3})*x^2"},
    {"y": "y2", "name": "second",
     "formula": "{b0}+{b1}*z+{amp}*exp(-{rate})*z^2"},
], covariance="cr0", cluster="group")
```

The specification uses a validated expression grammar, never executable user
code. A braced name denotes a shared mean parameter. Numeric constants can
encode fixed restrictions; reusing a name imposes exact cross-equation sharing.
There is no implicit intercept or category expansion. Every outcome column is
excluded from every right-hand side, including another equation's outcome.
Endogenous structural systems require a separate identification and likelihood
contract.

Expressions are reparsed from source. The supported smooth subset uses numeric
constants, named columns, braced parameters, `+`, `-`, `*`, `/`, constant integer
powers in `[-12,12]`, and `exp`, `log`/`ln`, `sqrt`, `sin` and `cos`. Logarithms, square roots,
division and powers must have valid finite values and derivatives throughout
the accepted evaluation. Nonsmooth operations and unsupported functions fail
explicitly. A supplied `start` mapping overrides embedded starting values;
conflicting embedded declarations are rejected. Required numeric input values and
starting values have a declared finite magnitude bound of `1e12`.

The resident fit uses one strict complete aligned sample: required numeric
inputs must be finite, and every equation retains the same original rows and
index. There is no implicit row deletion, fitting weight, Dataset route, or
CUDA/MPS execution. Initial support bounds are 512 observations, two to four
equations, 12 mean parameters, 16 exogenous columns and 128 tape operations per
equation. Work and workspace options can refuse a fit or query before allocating
its complete uncertainty output. These are implementation/resource boundaries,
not statistical requirements or permission to truncate input.

## Joint likelihood and uncertainty

At a mean-parameter candidate, `Sigma = R'R/N` uses the original observation
count. Profiling this unrestricted covariance gives the Gaussian log
likelihood including its normalizing constants. An exactly fitted or singular
residual covariance fails explicitly; no ridge makes an unidentified model
appear identified.

The optimizer result is advisory. Acceptance requires a finite positive-definite
residual covariance, identified full joint observed information, and physical
score/Newton-step stationarity. Accepted estimates certify a finite local
maximum; they do not establish a global maximum or uniqueness.

The physical parameter vector contains all mean parameters followed by
`vech(Sigma)`, including each off-diagonal entry once. Covariance coordinate
aliases are `cov__equation_i__equation_j` in lower-triangle equation order.
The full observed Hessian includes nonlinear residual curvature, symmetric
off-diagonal multiplicity, and every mean/covariance cross block. A
Gauss–Newton mean-only matrix is a different uncertainty contract.

`covariance="oim"` uses the inverse full observed information.
`"hc0"` uses the outer products of complete original-observation joint scores.
`"cr0"` first sums those scores within each declared independent cluster.
Their complete bread, meat, scores and covariance are saved. No finite-sample
degrees-of-freedom multiplier is applied. Robust covariance may be singular;
target uncertainty reports that limitation instead of changing estimators.

Mean parameters and regular off-diagonal covariance coordinates use declared
asymptotic normal inference. Positive variance intervals use a positive-scale
transformation where supported. A zero-variance boundary is not an ordinary
interior Gaussian null. HC0 assumes independent observation vectors; CR0 assumes
independent clusters and an asymptotic cluster regime. They do not establish
universal small-sample or few-cluster coverage.

## Portable restoration

The `nonlinear_sur_state` attribute stores versioned expression source,
canonical parsed structure, original aligned inputs/index, all settings,
physical parameters and complete likelihood/score/information/covariance
moments. A restored model reparses the permitted source and checks numerical
replay and physical stationarity without running an optimizer. Full TableSet
restoration also checks its tables. A checksum detects accidental modification;
it is not authentication. Foreign code or a serialized executable tape is never
run.

## Predictions, effects and contrasts

```python
means = oe.nlsur_predict(fit, new_data)
conditional = oe.nlsur_predict(fit, new_data, given=["first"])
effects = oe.nlsur_margins(fit, new_data, x=["x", "z"],
                          scale="effect", weights=[1, 2, 3])
elasticities = oe.nlsur_margins(fit, new_data, x=["x", "z"],
                               scale="elasticity", weights=[1, 2, 3])
test = oe.nlsur_contrast(fit,
    {"slope": "{b1}", "asymmetry": "2*{rate}"},
    null={"slope": 0, "asymmetry": 0})
```

Unconditional queries evaluate the saved mean system. Given an observed subset
of equations in each query row, Gaussian conditioning produces the remaining
means and residual covariance by the Schur complement. The observed values are
fixed inputs to the query. Mean and conditional covariance targets use the full
joint mean/Sigma Jacobian and target covariance, including their cross terms.
Parameter uncertainty and conditional residual variation are separate outputs;
there is no fitted model of residual dependence between new query rows.

Effects differentiate the saved smooth mean/conditional predictor with respect
to a continuous exogenous column. Conditional effects can cross equations through
`Sigma`. Their parameter Jacobians include the necessary mixed derivatives.
Elasticities require positive declared covariate and predicted-mean domains.
Fixed postestimation weights are complete positional nonnegative values with
positive total mass. Elasticity domains apply even to zero-weight rows. These
weights standardize the requested averages; they do not
weight the likelihood or estimate a weight model. Every requested row and target
is accounted for, including structural zero effects and unavailable first-order
uncertainty.

Smooth contrasts reference named physical parameters through the same safe
expression grammar. Their estimates, full Jacobian and joint delta covariance
are saved. Joint Wald tests require a regular declared null and an estimable,
nonsingular restriction covariance. They are first-order asymptotic inference,
not a general certification of nonlinear-null feasibility or boundary tests.

## Evidence and remaining scope

The synthetic [example](../examples/nonlinear_sur.py) exercises all eight scopes
and writes complete result tables, attributes and LaTeX. Source tests,
independent numerical fixtures/calibration, a frozen runtime, observed native
Run/save/Quit/reopen, and a public release are separate acceptance layers.
Only dated source-pinned receipts establish their respective results.

The model definition and relationship of iterative nonlinear SUR to Gaussian
ML are described in the [Stata nonlinear SUR manual](https://www.stata.com/manuals/rnlsur.pdf).
Its default Gauss–Newton inference does not establish our full physical OIM or
any licensed-vendor comparison. Structural FIML, arbitrary executable likelihood
callbacks, broader weights/missing-data algorithms and public-release delivery
remain outside this bounded stage.

The dated acceptance evidence (internal evidence excluded from this public snapshot)
retains both the failed N320/G80 and separate predeclared N512/G128 calibration.
The seven-dimensional CR0 joint ellipse covered 79/100 at 80 clusters; this is
an explicit finite-cluster joint-inference limitation. The separate 128-cluster
run passed its predeclared diagnostic gates (96/88/91 joint coverage for
OIM/HC0/CR0); even these counts do not establish exact nominal 95% joint coverage.
All 600 attempted fits and every seed were retained. Users needing small-sample
or few-cluster joint tests require a separate inference method.
