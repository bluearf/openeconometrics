# Chained multiple imputation

`mi_chained` implements three bounded fully conditional specification (FCS)
updates: Gaussian regression (`normal`, MARKET-356), predictive mean matching
(`pmm`, MARKET-357), and binary logistic regression (`logit`, MARKET-358).
Each uses native CPU float64 Torch numerical kernels. The completed panels and
all settings, seeds, observed cells, missing geometry, and chain diagnostics are
retained in an immutable `MIResult` that supports JSON replay, `.dataset(1)`,
`.table`, and `.latex`.

```python
import pandas as pd
import openecon as oe

frame = pd.DataFrame({
    "x": [-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 2.5, 3., 3.5],
    "income": [1., 3., 2., None, 4., 7., 5., 9., None, 10., 8., 12.],
    "cost": [3., 2., None, 5., 4., 7., 6., None, 7., 10., 9., 11.],
    "employed": [0., 0., 1., 0., None, 1., 0., 1., 1., None, 1., 0.],
})
result = oe.mi_chained(
    frame, ["income", "cost", "employed", "x"],
    methods={"income": "normal", "cost": "pmm", "employed": "logit"},
    m=5, seed=20261007, burn=20, iterations=10, donors=5,
    logit_prior_scale=2.5, logit_burn=100, logit_steps=100,
)
completed = result.dataset(imputation=1)
saved_state = result.model_dump_json()
```

The `columns` argument selects the complete imputation panel. Complete selected
columns condition the updates and remain unchanged. `methods` must specify
exactly every incomplete selected column. By default each target uses all other
selected columns. To choose a predictor matrix explicitly, supply a mapping
such as `predictors={"income": ["x"], "cost": ["x", "income"],
"employed": ["x", "income"]}`. Every incomplete column needs a mapping entry;
an empty list gives an intercept-only model. Targets are visited in selected
column order and use the latest imputations of their predictors.

Each of the `m` independent chains starts by drawing an observed donor for each
missing cell, runs `burn + iterations` sweeps, and retains its final panel.
The chain seeds are `(seed + chain_index) modulo 2**63`, with zero-based chain
indices. The same seed and inputs reproduce the complete state without changing
Torch's global random generator. Observed cells and the original row index,
including repeated labels, are preserved exactly; missing rows are never dropped.

## Gaussian updates

For observed outcomes, let `X` include the intercept, `b` be the full-rank OLS
coefficient vector, `SSE` the residual sum of squares, and `nu = n_observed - q`.
Under `p(beta, sigma2) proportional to 1/sigma2`, the update draws
`sigma2 = SSE / chi_square(nu)`, then
`beta | sigma2 ~ Normal(b, sigma2 * inverse(X'X))`, including the full coefficient
covariance. Missing outcomes receive independent `Normal(X_missing beta,
sigma2)` predictive residuals. A QR factor supplies the covariance draw without
forming a regression inverse. This follows the conjugate regression update
described in the [MICE normal-model documentation](https://amices.org/mice/reference/mice.impute.norm.html),
with no ridge modification. Residual degrees of freedom, variance, and coefficient
draws are retained per update.

## Predictive mean matching

PMM draws the same Gaussian regression posterior, then matches observed fitted
means `X_observed b` to missing posterior means `X_missing beta`. This is type-1
matching as described in the [MICE PMM documentation](https://amices.org/mice/reference/mice.impute.pmm.html).
For each missing cell, the method selects the requested nearest `donors`
observations and samples one uniformly. The imputed outcome equals that donor's
original observed value; no outcome noise or synthetic donor values are added.
`pmm_ties="stable"` resolves equal distances by original observed row position.
`pmm_ties="random"` applies a seeded random ordering before stable sorting, so
only exact-distance ties change order. Donor row positions are retained for
each update. `donors` must not exceed the observed outcome count; it is never
silently reduced. Matching uses one observed-distance vector at a time.

## Binary logistic updates

Observed binary outcomes must contain exactly `0` and `1`, with both classes
represented. All coefficients, including the intercept, have independent proper
`Normal(0, logit_prior_scale**2)` priors in raw predictor units. The log posterior
is `sum(y eta - softplus(eta)) - sum(beta**2)/(2 * prior_scale**2)`.

Each conditional update performs `logit_burn` discarded and `logit_steps`
additional random-walk Metropolis transitions. Its fixed Gaussian proposal has
covariance `logit_proposal_scale**2 * inverse(0.25 X'X + I/prior_scale**2)`.
Because this proposal is symmetric and fixed within an update, acceptance uses
the exact posterior density ratio, following [Hastings (1970)](https://doi.org/10.1093/biomet/57.1.97).
The last coefficient vector supplies Bernoulli probabilities for missing cells.
Subsequent updates start from the previous coefficient vector while recomputing
the current conditional target and proposal factor.

The MH transition targets the stated posterior; a finite run is not an exact
independent posterior draw. No Laplace approximation supplies imputations.
Per-update burn/sampling acceptance counts, final coefficients, log density,
and mean predicted probability are recorded, together with aggregate acceptance
rates per variable and chain. Acceptance is not a convergence test and the result
sets `convergence_claim=False` and `stationarity_claim=False`.

## Scope and admission

Only resident real numeric DataFrames, column mappings, or row records are
admitted, with at most 10,000 rows, 16 selected columns, and 100 imputations.
There is no Dataset/replay, weights, categorical encoding, passive transformation,
ordinal/multinomial, multilevel, or device-fallback route. All selected missing
cells must be imputed. Each observed conditional design must have full column
rank. Gaussian/PMM models require more observed outcomes than coefficients and
positive, numerically resolved residual variance. All-missing targets, rank loss,
non-finite values, and unsupported requests fail with actionable errors; no
predictor is dropped and no implicit ridge is applied.

There are at most 1,000 sweeps per chain and 100,000 traced variable updates.
`max_work=100_000_000` bounds a declared conservative operation estimate that
includes PMM donor sorting and every logistic burn/sampling transition, checked
before a conditional fit. A separate shared admission plan includes retained
panels and serialized state. A further pre-fit plan includes every retained
conditional trace and its serialized checksum representation. These are
allocation plans, not measured peak RSS.

FCS is a collection of conditional models, not a guarantee that the conditionals
define a compatible joint distribution. Choosing a predictor matrix, a logistic
prior scale, and a sufficient run length remains part of the analysis. These
three methods do not claim complete MICE or Stata parity. The original
[MICE paper](https://www.jstatsoft.org/article/view/v045i03) discusses predictor
selection and the broader chained-equations framework.

## Verification

`tests/test_mi_chained.py` checks Gaussian posterior coefficient/variance and joint
predictive moments against hand matrix calculations; PMM type-1 donor decisions,
tie behavior and exact observed support; and an intercept-only logistic posterior
against independent numerical quadrature. It also checks a separated binary
sample under the proper prior, a mixed three-method chain, replay and global RNG
isolation, row/index and observed-cell preservation, JSON round-trip, tamper
rejection, singular and all-missing failures, exact donor-count refusal, and
resource rejection before fitting. These checks cover the declared methods and
finite-run contract, not convergence of every future user dataset.
