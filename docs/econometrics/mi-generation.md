# Gaussian multiple-imputation generation

`mi_mvn` and `mi_monotone` generate complete numeric datasets while preserving
every observed cell and the original row index, including duplicate labels. They
return an immutable `MIResult`, separate from pooled analysis. Fit the intended
analysis on each `result.dataset(imputation=1)` through `result.dataset(m)` and
pool the resulting estimates with the appropriate MI pooling procedure.

## Scope and admission

These methods require resident DataFrames, mappings of sized columns, or lists
of row records. Select between 1 and 16 real numeric columns, at most 10,000
rows and 100 imputations. All selected rows remain in the sample. Numeric pandas
nullable columns are accepted; strings, complex values, infinity, Dataset/replay
sources, weights and GPU inputs are not supported. No implicit listwise deletion
or streaming collection occurs. Numerical kernels and random draws use native
Torch on CPU in float64. pandas supplies table I/O and index reconstruction.

Admission checks shape, estimated work and the configured workspace budget
before materializing selected numeric buffers. `max_work` defaults to 100,000,000
and limits a conservative arithmetic estimate; `OPENECON_WORKSPACE_MB` and
`openecon.resources.use_workspace_budget(...)` bound estimated live matrix, result and JSON
buffers. These estimates are not process-RSS guarantees. Reduce the selected
sample, columns or sampling schedule when an admission limit is exceeded.

## Multivariate normal data augmentation

```python
from openecon.econometrics.mi.generation import mi_mvn

result = mi_mvn(
    data, ["income", "spending"], m=5, seed=42, burn=100, thin=20,
    prior={
        "mean": [0.0, 0.0],
        "kappa": 0.1,
        "df": 5.0,
        "scale": [[100.0, 0.0], [0.0, 100.0]],
    },
)
complete = result.dataset(imputation=1)
```

The example prior is illustrative. Choose its mean and covariance scale to
match the original data units and substantive knowledge; the function neither
standardizes the prior nor secretly estimates it from observed data. An explicit
prior is mandatory. Its exact parameterization is

\[
\Sigma\sim IW_p(\nu_0,\Lambda_0),\qquad
\mu\mid\Sigma\sim N_p(\mu_0,\Sigma/\kappa_0).
\]

Require finite `kappa > 0`, `df > p-1`, a finite mean and a symmetric positive
definite scale. When the mean exists, `E[Sigma] = scale / (df-p-1)`. The initial
missing cells and parameter mean use the prior mean; initial covariance uses
the inverse-Wishart mode `scale/(df+p+1)`.

Each iteration first draws missing row components from their conditional MVN
distribution, then draws parameters conditional on the entire augmented sample.
For the complete-data parameter update,

\[
\kappa_n=\kappa_0+n,\quad \nu_n=\nu_0+n,\quad
\mu_n=(\kappa_0\mu_0+n\bar x)/\kappa_n,
\]

\[
\Lambda_n=\Lambda_0+\sum_i(x_i-\bar x)(x_i-\bar x)' +
\frac{\kappa_0n}{\kappa_n}(\bar x-\mu_0)(\bar x-\mu_0)'.
\]

The covariance draw uses a Bartlett Wishart factor with independent Gaussian
lower-triangle entries and chi-square diagonals. The mean draw then includes
its posterior covariance uncertainty. The conditional imputation also includes
predictive noise. Fully missing rows and columns are permitted under the proper
prior, but their imputations can be strongly prior-sensitive.

Retained iterations are `burn+thin, ..., burn+m*thin` from one chain. Metadata
records the prior, seed, requested and actual iteration counts, retained steps,
parameter mean and covariance-diagonal traces, missing geometry and resource
plan. A finite burn-in and thinning schedule do **not** establish convergence or
independence. The result declares `converged=False`, convergence unassessed and
`stata_parity_validated=False`; inspect traces,
increase the schedule and compare separately seeded runs as appropriate.

This is continuous Gaussian working-model imputation under ignorable
missingness. It does not impose binary, count, nonnegative or bounded support.

## Verified monotone Gaussian regression

```python
from openecon.econometrics.mi.generation import mi_monotone

result = mi_monotone(data, ["age", "income", "spending"], m=5, seed=42)
# Or declare the nesting order explicitly:
result = mi_monotone(data, ["age", "income", "spending"],
                     order=["age", "income", "spending"], m=5, seed=42)
```

The default order sorts by increasing missing count, retaining selected-column
order for ties. The function then verifies nesting: observing a later variable
must imply observing every preceding variable. A declared order must be a
permutation of all selected columns. Non-monotone patterns are refused.

Each incomplete outcome is regressed on an intercept and all preceding
variables using its observed rows. Nesting guarantees those observed predictors
are genuinely observed, so the regression posterior can be cached independently
of previous imputations. Missing-row predictors may contain earlier imputations.
Predictors are internally centered and scaled for stable factorization. For the
standard prior `p(beta,sigma_squared) proportional to 1/sigma_squared`, draw
`sigma_squared = SSE/chi_square(n_observed-q)`, then Gaussian posterior beta and
independent predictive Gaussian noise. This incorporates both parameter and
residual uncertainty. Designs must have full rank, positive observed residual
degrees of freedom and positive residual variation. Rank deficiency and exact
fits are refused rather than repaired with a silent ridge or deterministic fill.

One ordered pass produces an imputation; no MCMC burn-in is needed. Imputation
`i` uses seed `(seed+i-1) modulo 2**63`, recorded with its 1-based ID. Metadata
records the verified order, residual degrees of freedom, regression draws and
the Gaussian and missingness assumptions. Continuous Gaussian regressions with
constant conditional residual variance are the supported model.

## Immutable results and checked persistence

`result.original` stores observed values and `None` for missing cells;
`result.completed_matrices` stores every complete draw. `result.metadata`
recursively freezes sample positions, the declared index codec, missing mask,
counts, IDs, seeds and sampler settings. `result.table` and `result.latex`
summarize observed and missing counts without replacing the full state.

```python
from openecon.econometrics.mi.common import MIResult

saved_json = result.model_dump_json()
restored = MIResult.model_validate_json(saved_json)
complete = restored.dataset(imputation=2)
```

Loading validates matrix dimensions, finite completions, unchanged observed
cells, sample and index dimensions, missing geometry, ordered IDs and seeds.
A SHA-256 checksum covers canonical full-state JSON, including missing-cell
draws and metadata. It detects accidental corruption or alterations when the
recorded digest is retained; it is not an authentication signature. Loaded
state and index reconstruction obey bounded envelope and workspace checks.
Ordinary, Range, Datetime, Timedelta, Categorical and MultiIndex row indexes are
encoded explicitly; unsupported custom indexes are refused. `.dataset(...)`
returns a fresh editable DataFrame. Fully observed inputs legitimately produce
identical complete datasets, with zero missing cells and zero imputation steps.

## Sources and validation

The implementation follows the I-step/P-step distinction in Joseph Schafer's
[NORM package documentation](https://cran.r-project.org/web/packages/norm/norm.pdf).
The conjugate NIW parameter update is consistent with Kevin Murphy's
[Conjugate Bayesian analysis of the Gaussian distribution](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf).
Posterior regression and predictive variance draws follow the
[MICE Gaussian imputation documentation](https://amices.org/mice/reference/mice.impute.norm.html).
MICE's [monotone visit-order documentation](https://amices.org/mice/reference/mice.html)
also distinguishes an actual monotone pattern from merely sorting variables;
this implementation explicitly checks nesting.

`tests/test_mi_generation.py` checks a hand-computed complete-data conjugate
posterior; NIW mean/covariance and regression posterior sampling moments;
public univariate MVN and monotone predictive moments; seed replay; varying
draws; unchanged observations; arbitrary and monotone missing patterns; full
JSON replay and index identity; immutable state and checksum corruption; and
shape, work, workspace, rank, posterior and unsupported-source refusals.
NumPy is used only for independent test oracles, never for runtime kernels.
