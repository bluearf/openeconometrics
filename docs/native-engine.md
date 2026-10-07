# Statistical core

OpenEconometrics's current statistical API uses its own PyTorch float64 implementation.
The model, covariance, inference and likelihood logic is maintained in OpenEconometrics.
There is no NumPy estimator or alternate statistical backend. PyTorch is a
required dependency, and a fit does not call a statsmodels estimator.

Large OLS uses [bounded three-pass TSQR](streaming.md), including weighted,
categorical and supported cluster/HAC/resampling inference. The broader model
catalogue has dedicated native replay algorithms for likelihoods, panel groups,
joint systems, ordered state recursions and bootstrap. `oe.read()` automatically
opens large local CSV/Parquet inputs as Dataset; `oe.scan()` always selects that
path. Option conditions and algorithms are recorded by `oe.capabilities()`.

```python
import openecon as oe

result = oe.ols(
    data=oe.example(), y="wage", x=["education", "experience"],
    covariance="HC3",
)
print(result.summary())
```

PyTorch supplies tensor operations, QR, triangular/Cholesky solves and elementary
special functions. OpenEconometrics implements the statistical inference routines.
The separation safeguard uses OpenEconometrics's float64 Torch primal-dual LP solver
with bounded constraint generation, checked dual upper bounds and globally
replayed witness candidates. SciPy is not a runtime dependency. Pandas supplies
table and file interoperability; its transitive NumPy dependency remains.
Statsmodels and SciPy reference distributions are used in
development tests; this is different from delegating estimation to those
libraries at runtime.

## OLS

Predictors are scaled by their magnitude and RMS; with an explicit intercept,
non-intercept columns are also centered. Coefficients and covariance are
transformed back into the original units. The reported condition number concerns
the transformed design, not the raw matrix.

For the transformed design `Z = QR`, reduced QR permits a triangular solve of
`R theta = Q.T @ y`. The factors are reused for rank/condition diagnostics,
coefficient estimation and covariance:

- Classical covariance uses `SSE / (N-K)` and the triangular factor.
- HC1 uses squared residuals with the `N / (N-K)` adjustment.
- HC3 uses squared residuals divided by `(1-h_i)^2`; leverage is computed from
  row norms of `Q`. Unit leverage is rejected.
- One-way cluster covariance aggregates group scores and applies CR1:
  `G/(G-1) * (N-1)/(N-K)`.

The sandwich is evaluated in the orthogonal QR basis. No N-by-N hat matrix,
observation covariance matrix or diagonal residual-weight matrix is constructed.
Temporary tensor allocations still count toward actual memory use; the dense
design guard is not a process-wide memory limit. Singleton clusters have a
shortcut that avoids an unnecessary group-score copy.

OLS uses Student-t inference: `N-K` degrees of freedom for ordinary and HC
covariance, and `G-1` for cluster covariance. Small cluster counts remain a
statistical limitation even when the correction is computed correctly.

## Logit and probit

Both estimators maximize the Bernoulli likelihood using observed-information
Newton steps and an Armijo backtracking line search. An intercept model starts
at the empirical outcome probability. Convergence requires a small normalized
score and a small proposed parameter step; exhausted iterations or failed line
searches produce explicit errors.

Logit uses stable signed-logistic expressions. Probit uses log normal CDF and
inverse-Mills-ratio expressions, including an extreme-tail approximation to
reduce cancellation in observed information. The classical covariance is the
inverse observed negative Hessian through a Cholesky solve. Cluster covariance
uses per-observation likelihood scores aggregated by group and the same CR1
correction. Binary models use normal-z inference. Their HC1/HC3 variants are not
implemented and are rejected.

Before fitting, a bounded linear program checks complete and quasi separation.
The predictor basis is centered/scaled according to the intercept option, so
shifting a predictor does not silently change the check. Optimizer convergence
alone is never accepted: a dual certificate proves a bound for all observations,
or a candidate separating direction is checked against every replayed row.
Inconclusive precision, storage or convergence fails explicitly.

## Input and result contract

Convenience functions accept `data`, `y` and `x`; data may be a DataFrame, a
mapping of column names to values or a list of record mappings. The lower-level
`fit(ModelSpec(...), data=...)` exposes the same statistical choices. Unsupported
fields and ambiguous/invalid samples fail explicitly.

Category references are chosen before missing-row filtering and recorded.
Complete-case dropping requires an explicit choice. Saved provenance records
the actual PyTorch implementation, execution devices, float64 precision, solver diagnostics
and package versions. Execution details are result metadata, not configurable
model choices. Old files retain their original provenance and need their
historical environment for exact replay.

## Validation and limits

Development tests compare closed-form OLS fixtures, independent sandwiches,
likelihood derivatives, distribution tails/quantiles and end-to-end results with
independent reference implementations. Reference dependencies belong to the test
environment; they are not a replacement for the implementation under test.

Supported QR factors can use CUDA float64 or Mac Metal float32 preconditioning.
Metal results are refined and certified from the original CPU float64 rows;
unsafe factors fall back to native CPU QR. Global likelihood/state and inference
algorithms remain CPU float64. On the verification Mac, actual Metal factors
were compared with CPU results; CUDA hardware was unavailable. These checks do
not establish universal numerical correctness, full Stata parity or OS
peak-memory quotas.
[Recorded 0.2 benchmarks](performance.md) concern the former dual-engine
architecture and must not be used to characterize this release's speed.

Primitive references: [PyTorch reduced QR](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html),
[triangular solves](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve_triangular.html)
and [numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
