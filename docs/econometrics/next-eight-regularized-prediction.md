# Regularized and supervised prediction audit, 7 October 2026

This is the historical 7 October scope decision. The later PR159 delivery added
bounded binomial/Poisson prediction, and the current
[weighted/category scalar Gaussian/PLS contract](regularized-all-family.md)
reconciles those remaining options. Historical missing-route statements below
describe that earlier snapshot, not the current source. Supervised extensions
also have their own later source/evidence records; this document is not their
current inventory.

The native squared-error ridge, lasso, elastic-net and scalar-response PLS1
implementations now cover the continuous-outcome foundation of GitHub #33,
including explicit feature-specific penalties and forced controls. Binomial
and Poisson loss extensions remain distinct implementation stages. GitHub #42
remains an implementation task: general supervised CART,
CHAID, QUEST, forest/boosting, MLP/RBF, kNN and SVM public analytical families
are absent. Its planning document is insufficient to close that issue.

This decision was checked against the original requested sources, rather than
only the issue's later checklist. The causal/ML research catalog distinguishes
Stata's H2O integration and community scikit-learn wrappers from native
estimators. The SPSS catalog records `PLS` as an extension and records `TREE`
with four distinct growth methods, plus `MLP`, `RBF` and predictive `KNN`.
The historical complete implementation plan marks supervised prediction
families `Hayır`; that historical decision does not implement them or remove
their current requested scope. Native forest nuisance learning for `dmlplr`
and honest doubly robust CATE forests have different targets and do not supply
the general supervised workflows requested in #42.

## Implemented #33 foundation

* Numeric, unweighted, independent resident CPU float64 rows. `Dataset`, GPU,
  categories and penalized weights have no admitted route.
* Ridge uses an independent direct factorization; lasso and elastic net use
  coordinate descent with recorded KKT residuals and convergence refusal.
* Fixed paths, training-fold scaling, absolute supplied grids and automatic
  training-fold lambda fractions are persisted. Cross-validation uses pooled
  out-of-fold MSE, with deterministic local fold seeds.
* Scalar PLS1 persists complete loadings, scaling, component selection and
  original-unit coefficients. Component tuning fits preprocessing inside each
  training fold.
* Saved `ResultBundle` JSON supports `regularized_predict` without refitting;
  prediction-only coefficients receive no selected-model OLS standard errors.
* Declared arithmetic and workspace limits refuse requests before silently
  dropping rows, path points, folds or components.

Binomial and Poisson links, paths and probability/deviance validation currently
do not exist. The distinct weight and category extensions are additional
stages already documented in
[panel-prediction-extensions.md](panel-prediction-extensions.md).

## Feature-specific penalties and forced controls

`penalty_factors` supplies one finite nonnegative number per ordered `x` column.
`forced_controls` supplies distinct column names already present in `x` and
overrides their effective factors to zero. With standardized predictors
`z_j`, the explicit minimized objective is

\[
\frac{1}{2n}\sum_i(y_i-\bar y-z_i'\beta)^2+
\lambda\sum_j q_j\left(\rho|\beta_j|+
\frac{1-\rho}{2}\beta_j^2\right).
\]

The unpenalized intercept retains its existing convention. `q_j=0` removes
both penalties; the entire zero-factor block must have full rank in the full
retained sample and in every CV training fold. A constant forced column with
an intercept, duplicated forced columns, an unknown named control or a
singular training fold raises rather than silently selecting an arbitrary
control coefficient. Zero lambda uses an SVD least-squares solve; a fully
unpenalized fixed model also uses that exact limiting route. An automatic path
requires at least one positive factor.

Factors multiply both penalties, following the
[glmnet penalty definition](https://glmnet.stanford.edu/articles/glmnet.html).
OpenEcon uses the literal supplied numbers. The linked glmnet implementation
normalizes factors to sum to its predictor count, so a cross-package absolute
lambda comparison must first reconcile that convention. No glmnet, Stata or
SPSS runtime comparison is claimed here.

Automatic grids first project the training outcome onto the zero-factor
block. Their reference top penalty is then
`max(abs(z_penalized.T @ residual / n) / q_positive) / max(rho, .01)`.
For lasso this is the zero-penalized-slope KKT threshold; for ridge it remains
the documented numerical reference scale. Each CV fold repeats projection,
centering, scaling and top-penalty construction using its training rows only.
The final path uses the complete retained training sample after fraction
selection. No validation label supplies another fold's transform or candidate
path. Fixed/CV selection supports custom factors; heteroskedastic score-plugin
selection explicitly requires unit factors because a separate weighted-score
calibration has not been validated.

Full factors, zero-factor indices, named controls, selected path, per-path KKT
residuals, objective values and all fold diagnostics are retained in complete
model JSON. Named controls remain ordinary saved predictive terms; their
presence does not confer valid selected-model confidence intervals.

```python
model = oe.elasticnet(
    data=df, y="outcome", x=["age", "income", "price"],
    penalty_factors=[1, .5, 2], forced_controls=["age"],
    selection="cv", folds=5, seed=1729, l1_ratio=.5,
)
saved = oe.ResultBundle.model_validate_json(model.model_dump_json())
prediction = oe.regularized_predict(saved, new_rows)
```

Independent new tests enumerate all small penalized sign/support sets, solve
the forced and active blocks together and check complete paths, objective and
KKT. NumPy fold projections independently reproduce candidate grids and held-out
MSE; perturbing one fold's validation labels leaves that fold's grid and fit
diagnostics unchanged. Additional tests cover the zero-slope lasso maximum,
fully unpenalized OLS limit, file JSON restoration with fitting prohibited,
complete query row order, malformed factors, full/fold rank failure and work/
workspace refusal. Squared-error fitting and saved regularized prediction
locally enforce their CPU domain even when the caller selected a foreign
default tensor device; that caller preference is restored on return.

## Numerical and runtime evidence boundaries

The existing independent active-set/sign enumeration in
`tests/test_regularized_oracles.py` checks the convex squared-error optima,
paths, KKT, intercept/scaling conventions and restored predictions. Independent
NumPy fold algebra checks training-only transformations. The PLS tests include
a separate one-component direction and full-component OLS limit; optional
scikit-learn comparisons are development oracles, never runtime engines.
The pre-change audit of the targeted existing suite recorded 96 tests, 87 passed
and 9 skipped because the optional scikit-learn oracle is unavailable; the
NumPy references were executed. After the factor implementation, 111 tests
passed with zero skips across the 26 new tests and the 85 existing regularized
oracle/contract tests. These overlapping checkpoints are not additive. This
checkpoint is not a new packaged or
licensed-vendor execution.

Retained evidence is pinned to its recorded revisions:

| Evidence | Recorded scope | Current core-byte relationship |
| --- | --- | --- |
| `reports/validation/market127_regularized.json` | 64 IID sparse-linear heteroskedastic PLR replications; source influence algebra | Treatment inference, not universal selected-coefficient coverage or supervised models |
| `docs/evidence/market-124-127-129-131-132/runtime-dependencies.json` | Source ridge/DML/local runs with external estimator imports blocked | Prior source checkpoint, not new installed-app validation |
| `docs/evidence/clean-mac-install-rollback-2026-10-07/source-frozen-parity.json` | Frozen source snapshot `97ee1a3c33c077832ffd5a6714c51a74567817a9` | Pre-change `regularized/kernels.py` SHA256 `bbcc10aabe95e7fff59467a4d4b4de12cbf0fbed4f41f93772d391bf5370d88c` matched the audited baseline; new factors change this file and need fresh package evidence |
| `docs/evidence/panel-prediction-2026-10-07/receipt.json` | Actual installed PLS execution and restart at `496a9ba16fcb2b9a57d9527f5ef77fc6481932ea` | `pls.py` SHA256 `07a65ed8d1638ae5a9ddeb27ade66d4266e403593592b74b4def95a1e61d23f0` remains unchanged; pre-change shared `prediction.py` SHA256 `667c78b4e805ed0ca8eea44f15c67a7fe1663bb5ee6bf2ac754639b9c1bbb63d` matched baseline but new factor orchestration changes it |

Later registry and orthogonal-forest additions changed other modules. Matching
individual numerical files does not turn those older installed receipts into
whole-current-main acceptance. The supervised scope document remains the
future training/split/state protocol, not execution proof:
[supervised.md](../research/supervised.md).

No licensed Stata/SPSS/EViews comparison, full vendor parity or new desktop
release proof is claimed by this audit.
