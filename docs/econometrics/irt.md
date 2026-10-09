# Unidimensional IRT: six fitted families and portable postestimation

The six functions fit independent people and locally independent item responses
conditional on one latent trait, **theta ~ N(0,1)**. This fixes both the latent
location and scale; there is no estimated latent distribution. All inference
uses native CPU float64 Torch. NumPy/SciPy appear only in development oracles.

| API | Item response and identification |
| --- | --- |
| `irt_rasch(data=..., items=[...])` | Binary logistic, discrimination fixed **1**, item difficulties estimated |
| `irt_2pl(...)` | Binary logistic, positive item discriminations and difficulties estimated |
| `irt_3pl(..., guessing=[...])` | Binary `c + (1-c) logistic(a(theta-b))`; each supplied `0 <= c < .4` is **fixed** |
| `irt_grm(...)` | Samejima graded response; positive item slopes and ordered cumulative thresholds |
| `irt_pcm(...)` | Masters adjacent-category partial credit, discrimination fixed **1**, item-specific steps may be disordered |
| `irt_rsm(...)` | Andrich rating scale, discrimination fixed **1**, item locations and shared **zero-sum** steps |

`irt_rasch` is the fixed-slope Rasch model, not a free-common-slope 1PL. The
PCM and RSM functions also fix discrimination at one; software that estimates
a common discrimination with a unit latent variance fits a different model.
3PL does **not** estimate guessing. These distinctions prevent reusing vendor
example coefficients from a different identification convention as proof.

```python
import openecon as oe
model = oe.irt_2pl(data=responses, items=["item1", "item2", "item3", "item4"])
display(model["parameters"])
display(model["covariance"])
portable = model.to_json()               # complete state; no table truncation
restored = oe.irt_restore(portable)
scores = oe.irt_score(restored, data=new_responses)
curves = oe.irt_information(restored, theta=[-2., -1., 0., 1., 2.])
latex = restored.to_latex()               # all seven complete tables
```

The functions are lazy public registry helpers, available in `oe.capabilities()`
and offline editor help. They do not accept the single-outcome `ModelSpec` /
`oe.fit` contract. `IRTResult` is a named-table result with `.to_json()` and
`.to_latex()`; render individual tables with `display(model["parameters"])`.
The console table display may preview large tables; complete portable JSON and
LaTeX do not reuse that preview.

## Sample and categories

Resident DataFrame, column mapping or row records only. Fitting explicitly
uses complete listwise item responses. Returned sample tables retain every
original position/index, missing count and inclusion flag; state retains every
fitted response and typed source hash. Duplicate indices remain associated
with distinct row positions. Input is never silently recoded or weighted.

Items must be distinct numeric columns with integer codes `0..K-1`, `K=2..6`.
Every fitted item must observe all its contiguous levels; a constant/gapped
item is refused. Binary families require `0,1`; RSM requires the same `K>=3`
for every item. GRM and PCM allow different category counts per item.
Boolean/string/fractional/negative/infinite observed responses are refused.

Scoring supplied rows uses each observed item. A partially missing row retains
its original identity; a fully missing row returns prior mean **0**, SD **1**,
zero observed items and zero negative log marginal likelihood. Extreme
all-low/all-high responses have finite posterior scores. The fitted-sample
default exactly reproduces the `people` table.

## Estimation and uncertainty

Nonadaptive Gauss-Hermite integration uses the standard-normal Jacobi matrix
and native eigendecomposition. Log-domain likelihoods avoid multiplying small
probabilities; GRM middle probabilities use stable cumulative-logit differences.
L-BFGS uses bounded iterations/evaluations and positive log-slope/ordered-gap
coordinates. There is no EM or adaptive quadrature equivalence claim.

Every accepted fit requires finite parameters, max absolute gradient <=
`max(10*tolerance, 2e-5)`, positive definite observed information, and an
information eigenvalue ratio above `1e-9` (absolute floor `1e-7`). Supported
interior geometry has `.1 < discrimination < 5` and `|threshold| <= 8`.
Nonconverged, boundary and weakly identified fits are refused, not relabelled
as success. A denser quadrature of `min(121, 2*points+1)` nodes must change
the log likelihood by <=`1e-5` per person. This audits integration **at the
accepted parameters**; it does not bound parameter change after a denser refit.

Complete observed-information covariance in optimization coordinates is
transformed jointly to the natural discrimination/difficulty/step coordinates.
Tables report standard errors, asymptotic z, two-sided p and normal Wald
intervals (no finite-sample degrees of freedom). RSM exposes all shared steps;
their covariance is deliberately singular under the exact zero-sum constraint.
Optimization information remains positive definite in free coordinates.

`irt_score` reports the EAP posterior mean and **posterior SD**, conditional on
fitted item parameters. This is not a frequentist confidence interval and does
not include item-parameter uncertainty. `irt_information` retains each category
probability/derivative, expected item/test score, and item/test Fisher information
`sum_k P'_k(theta)^2/P_k(theta)`. Test information sums items under local
independence. Curves have no parameter confidence bands.

## Persistence and admission limits

`openecon.irt.v1` preserves free/natural parameters, full natural/raw covariance,
observed information, quadrature, fitted responses/categories/counts, complete
sample alignment, log likelihood/convergence/audit, confidence level and buffer
plan. The exact envelope carries SHA-256. Restore checks shape/category/sample
consistency, latent identification, positive information, its inverse,
transformed covariance, canonical quadrature and fitted likelihood. Checksum
detects accidental changes; it is not an authenticated signature.

| Limit | Value |
| --- | --- |
| Fitted people | 50..3000; also `n > 3p` |
| Items / observed categories | 3..16 / 2..6 |
| Free parameters | <=64 |
| Main quadrature / default | 21..61 / 41 |
| Iterations / evaluations | <=300 / <=600 (defaults 200 / 400) |
| Planned likelihood/Hessian work | `n*points*items*(max_eval+4p) <= 300,000,000` |
| Named workspace buffers | <=min(128 MiB, current global/task budget) |
| Scoring people / curve grid | <=3000 / <=1001 theta values in [-8,8] |
| Portable JSON input | <=8 MiB |

Plans bound named tensor/model buffers before allocation; they do not bound
caller DataFrames, Python objects, third-party allocator workspace or process RSS.
Evaluation budget exhaustion returns an explicit error. CPU selection remains
explicit even under a caller's different Torch default device. Dataset, weights,
robust/cluster covariance, free guessing/common slope, multiple groups,
multidimensional traits and latent multigroup invariance remain unsupported.
Saved descriptive EAP residual/Q3/item-fit summaries and observed-score binary
CMH/MH DIF now have [separate diagnostics contracts](next-eight-survey-measurement.md).
These provide neither calibrated model-fit p-values nor polytomous/nonuniform
or latent multigroup DIF.

## Method sources and validation boundaries

Model probability/latent-score definitions were checked against the primary
[Stata IRT manual](https://www.stata.com/manuals/irt.pdf), with fixed versus
estimated slope conventions distinguished. See its
[GRM](https://www.stata.com/manuals/irtirtgrm.pdf),
[PCM](https://www.stata.com/manuals/irtirtpcm.pdf) and
[RSM](https://www.stata.com/manuals/irtirtrsm.pdf) formulas.
[Chalmers (2012), original mirt paper](https://www.jstatsoft.org/v48/i06)
provides a primary marginal-likelihood/quadrature reference. No software
output was copied into the fit kernel.

`tests/test_irt.py` uses separate NumPy probability formulas, SciPy quadrature /
BFGS, finite-difference **full** Hessians and covariance transformations, normal
inference, EAP integration and information derivatives. Seeded six-family
recovery is a bounded example, not universal recovery or interval coverage.
Binary model reductions, RSM identification, missing alignment, tampered
state, unsupported input, global workspace/evaluation/default-device guards
and complete JSON/LaTeX roundtrips are exercised. Source, frozen runtime,
installed native Run/export/restart proofs are separately recorded in the
delivery evidence (internal evidence excluded from this public snapshot).
Licensed Stata execution, universal parity, physical GPU validation, signed
public release and deployment are not part of this acceptance.

## Supplied calibrated banks and conditional scoring

The fixed-calibration extension accepts supplied binary 2/3/4PL and mixed GRM/GPCM/nominal-response banks. It does not fit these supplied parameters. [Bank and finite-prior specification](irt-calibrated-banks.md), [conditional MLE and normal-prior MAP](irt-calibrated-scores.md), and [posterior draws and replicate prediction](irt-calibrated-predictive.md) document the bounded APIs and uncertainty targets.

The finite latent prior is the actual caller-declared distribution. Exact finite-posterior moments and predictive score distributions do not claim continuous-prior quadrature equivalence, population-model plausible-value inference, or item-calibration uncertainty.
