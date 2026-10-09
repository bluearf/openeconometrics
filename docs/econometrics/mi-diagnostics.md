# Observed Gaussian EM and Little MCAR diagnostics

`mvnorm_em(data, columns, *, max_iterations=500, tolerance=1e-8)` fits the
observed-data multivariate Gaussian likelihood. `little_mcar` has the same
arguments and computes Little's common-covariance pattern-mean homogeneity
test using that fit. Both return a frozen Pydantic `MIDiagnosticResult`.

```python
from pathlib import Path

fit = oe.mvnorm_em(data, ["income", "age", "score"])
diagnostic = oe.little_mcar(data, ["income", "age", "score"])
print(diagnostic.to_frame())
diagnostic.to_json("little-mcar.json")
restored = oe.MIDiagnosticResult.from_json(Path("little-mcar.json"))
latex = restored.to_latex(index=False)
```

The APIs use resident continuous real numeric data and Torch CPU float64
kernels. Missing values are retained as missing, rather than removed listwise
or filled for likelihood evaluation. The fit assumes an iid multivariate
Gaussian model and an ignorable missingness mechanism for likelihood
inference. EM does not establish ignorability.

## Gaussian observed-data maximum likelihood

For each missingness pattern, let O be its observed columns and M its missing
columns. Each row contributes the marginal normal density on O; its missing
coordinates are integrated out. The conditional-moment E step uses

\[
E(Y_M\mid Y_O)=\mu_M+\Sigma_{MO}\Sigma_{OO}^{-1}(Y_O-\mu_O),
\qquad
V(Y_M\mid Y_O)=\Sigma_{MM}-\Sigma_{MO}\Sigma_{OO}^{-1}\Sigma_{OM}.
\]

The M step updates the mean and population covariance from these expected
first and second moments using the informative-row count as the ML divisor.
Conditional covariance is included; filling missing cells with only their
conditional means would underestimate covariance. Computation is centered
and scaled by each column's observed moments, then transformed back to the
declared original units. All observed likelihoods include the original-unit
Jacobian and normal-density constants. Initialization is deterministic.

The returned `estimates` (also `means`) and `covariance` contain the complete
Gaussian mean vector and fitted population covariance. This covariance is
**not the sampling covariance of the estimated mean parameters**. No mean
standard errors, confidence intervals, completed data or multiple imputations
are claimed by this stage.

`loglikelihood_history` includes initialization and every EM update. The
algorithm rejects a likelihood decrease beyond float64 roundoff. It returns
a result only when both the maximum standardized parameter change and
relative observed-likelihood change are at most the declared tolerance.
Exhausting the iteration limit raises `AnalysisError("mi_nonconvergence")`
with the iteration history and final change attached to the error. No ridge,
pseudoinverse or successful-result label replaces a singular fit.

## Little pattern-mean homogeneity test

For each nonempty pattern j, the test compares its observed-column mean
against the corresponding coordinates of the EM-fitted common mean:

\[
Q=\sum_j n_j(\bar y_{O_j}-\hat\mu_{O_j})^\top
       \tilde\Sigma_{O_j O_j}^{-1}
       (\bar y_{O_j}-\hat\mu_{O_j}),\qquad
\tilde\Sigma=\frac{n}{n-1}\hat\Sigma_{ML}.
\]

The implementation explicitly uses the degrees-of-freedom covariance
correction specified by Little (1988, Section 3.1) and Li (2013, equation
7). `statistic_covariance_scale` records `n/(n-1)` separately while
`covariance` retains the uncorrected ML estimate. Here n is the number of
informative rows, excluding entirely missing rows. The reported asymptotic
chi-square degrees of freedom are `sum_j len(O_j) - p`, where each distinct
pattern contributes once, regardless of its number of rows. A zero-df design
raises `mi_degenerate_mcar`.

The result retains Q, df, upper-tail p-value and each pattern's contribution.
Rejection is evidence against MCAR within the mean-homogeneity framework.
**Failure to reject does not establish MCAR**: missingness may depend on
features this mean-based test does not detect. The implemented variant assumes
a common covariance across patterns. No categorical, covariate-dependent
missingness, unequal-pattern-covariance, finite-sample exact, survey-weighted
or advanced robust variant is claimed.

## Sample identity, admission and persistence

Every original physical row position and index identity is retained. Duplicate
index labels remain distinguishable by their physical positions. An entirely
missing row contributes no observed likelihood or mean comparison; its
original identity, empty pattern and omitted-information status remain in the
result. Adding such rows does not change the fit, statistic or correction.

The conservative supported geometry is:

- One through 16 distinct continuous real numeric columns; booleans, complex
  values and categorical/string columns are rejected. Infinity is rejected,
  rather than treated as missing.
- More informative rows than columns. Each column pair must have at least
  three jointly observed rows and a nonsingular pair covariance. This rejects
  unidentifiable covariance parameters and also conservatively rejects some
  sparse geometries that a broader estimator might support.
- At most 1,024 distinct patterns, at most 10,000 explicitly declared
  iterations, and one billion nominal iteration/dimension/pattern-solve work
  units. The configured workspace plan is checked before input tensors are
  allocated. It estimates named buffers rather than total process RSS.
- Streaming `Dataset` inputs are rejected before materialization. Unknown
  options, including weights and device overrides, are rejected. Input data
  are not modified.

The state consists of immutable tuple parameters, a frozen sample model and
frozen pattern models. JSON roundtrip retains all parameters, full covariance,
observed likelihood/convergence history, sample/pattern identity, admission
record, inference labels and original-source links. SHA256 detects state
changes and is not authentication. Restore also checks geometry, positive
definiteness, convergence and Little's inference; recomputing the digest does
not bypass those statistical consistency checks. `to_frame` exposes means,
full population covariance, likelihood history, patterns and the MCAR test
as exportable tables; `to_latex` exports the same saved state without refitting.

## Validation and original sources

The scoped tests compare Gaussian EM against an independent development-only
SciPy optimizer of the observed marginal normal densities, rather than
another EM implementation. A closed analytical pattern fixture has mean
`(3/11, 0)`, ML covariance `diag(134/121, 1)`, corrected Little statistic
`1320/737` and one degree of freedom. Additional checks cover complete-data ML
moments, original row identity, all-missing invariance, affine units and column
order, immutable JSON restore/tamper rejection, nonconvergence and admission
before allocation. SciPy and NumPy are test oracles only, not runtime kernels.

- [Little (1988), A Test of Missing Completely at Random for Multivariate Data
  with Missing Values](https://doi.org/10.1080/01621459.1988.10478722), JASA
  83(404), 1198–1202. Original common-covariance pattern test, correction and
  asymptotic degrees of freedom.
- [Dempster, Laird and Rubin (1977), Maximum Likelihood from Incomplete Data via
  the EM Algorithm](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x), JRSS B
  39(1), 1–38. Original EM framework and observed-likelihood ascent.
- [Li (2013), Little's Test of Missing Completely at
  Random](https://doi.org/10.1177/1536867X1301300407), Stata Journal 13(4),
  795–809. Primary implementation paper, equations 4 and 7 and the explicit
  distinction between the common-covariance and augmented variants.

This evidence is method-level validation. It does not establish complete
Stata/SPSS missing-data parity or licensed-vendor execution equivalence.
