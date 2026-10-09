# Saved multiple-imputation pooling and D1 tests

`oe.mi_pool` adopts and audits the full-covariance Rubin/Barnard–Rubin candidate
from draft PR #91 (`830b9c387fe980ebae159a0babad45b8c0370948`). It returns an
immutable `MIPoolResult` with all per-imputation estimates/covariances, terms,
IDs, complete-data degrees of freedom, alpha and declared imputation procedure.
Fitted `ResultBundle` inputs require the same model specification, ordered
terms/equations and physical sample. Nonconverged or unavailable inference is
rejected. Different outcome values across imputations are expected; incompatible
rows are refused. Supplied tensors require the caller to justify a common
approximately normal estimand. Averaged p-values never enter this procedure.

For m imputations, coefficients Q and full within-imputation covariance U, the
result retains Qbar, Ubar, B (sample covariance of Q), and
T = Ubar + (1 + 1/m) B. Marginal SE use the diagonal of T, with Rubin's finite-m
or Barnard–Rubin finite-complete-df t calibration. Infinite denominator df is
represented by JSON null in displayed tables, with the normal limit declared.
All matrices and per-imputation input arrays remain in the saved typed state.
The integrity hash detects accidental state changes; it is not authentication.

`oe.mi_test(pool, restrictions=R, values=q0)` tests R Q = q0. Identity R and zero
q0 are defaults. R must have full row rank; R Ubar R' must be positive definite
and numerically resolved. With k restrictions, r = (1 + 1/m)
trace(R B R' (R Ubar R')^-1)/k, Ttilde = (1+r) R Ubar R', and D1 is the squared
Mahalanobis distance divided by k using Ttilde. D1 assumes approximately normal
common estimates and the proportional between/within covariance approximation.
It differs from a Wald statistic using the unrestricted Rubin covariance T.

`df_method='li1991'` declares large-complete-sample inference. Both t=k(m-1)>4
and t<=4 finite-imputation branches are implemented; r=0 yields chi-square/k.
Finite `pool.complete_df` defaults to `reiter2007`, the author's equations 1–2.
This requires t>4 and vstar>4(1+a), where a=r*t/(t-2) and
vstar=complete_df*(complete_df+1)/(complete_df+3). Outside the positive-moment
domain the procedure refuses to substitute a df. Users may explicitly request
Li1991 if its large-complete-sample assumption is scientifically justified.
At r=0 the Reiter df equals vstar, consistent with that formula. D2, D3,
nonlinear restrictions, and arbitrary pooled predictions remain unsupported.

Admission is resident numeric CPU float64 arrays or restored compatible fits,
2..100 imputations and 1..32 parameters. Bool, complex, object and accelerator
arrays are refused. Estimated tensor/result/serialization workspaces are checked
before creating buffers against `OPENECON_WORKSPACE_MB` (default 512MiB).
Every within covariance is finite symmetric PSD with positive marginals; no
repair or ridge occurs. Joint covariance must additionally be positive definite.
Roundoff of a negative PSD trace below 1e-12 is normalized to zero. No
convergence claim is assigned to supplied estimates or finite imputation chains.

```python
pool = oe.mi_pool(fitted_results, imputation_description="MVN MI under the declared prior")
display(pool.table)
joint = oe.mi_test(pool, restrictions=[[0., 1., 0.], [0., 0., 1.]])
display(joint.table)
saved = pool.model_dump_json()
restored = oe.MIPoolResult.model_validate_json(saved)
assert oe.mi_test(restored).model_dump_json() == oe.mi_test(pool).model_dump_json()
```

Primary references: [Li, Raghunathan and Rubin D1 equations, author text](https://stefvanbuuren.name/fimd/sec-multiparameter.html),
[Reiter original manuscript](https://www2.stat.duke.edu/~jerry/Papers/Bmtka07.pdf),
and [Barnard–Rubin implementation in the author's mitml package](https://raw.githubusercontent.com/simongrund1/mitml/master/R/internal-pool.R).
The independent numerical tests use NumPy matrix equations and SciPy F/t/chi-square
references in the development environment. Those dependencies do not execute in
the native frozen worker. Formula/reference checks are not licensed Stata/SPSS
execution; whole-family and vendor parity remain false.
