# Scalar instrumental quantile regression

`ivqreg` adds native CPU float64 inverse quantile regression for one declared
endogenous scalar and a fixed interior quantile. A complete resident numeric
sample and finite alpha search are explicit. This method is separate from
2SLS/AR/CLR and ordinary `qreg`; those procedures do not supply IVQR uncertainty.
The stage does not close the wider specialized quantile programme.

```python
fit = oe.ivqreg(data=df, y="outcome", endogenous="treatment",
                x=["control"], instruments=["z1", "z2"],
                grid=[-2, -1, 0, 1, 2, 3], quantile=0.5)
saved = oe.summary_state(fit)
restored = oe.ivqreg_restore(saved)
query = oe.ivqreg_predict(result=restored, data=new_data)
```

## Estimand and identifying assumptions

The declared structural conditional quantile is `D*alpha(tau)+X*beta(tau)`.
Rank similarity, excluded-instrument restrictions, monotonicity and the
conditional quantile restriction are substantive assumptions. They are saved
explicitly and are not established by sample-rank checks. This target need
not equal the observed conditional quantile of Y given an endogenous D and X.
Querying a fitted model does not establish a causal interpretation by itself.

## Inverse QR and weak-identification inversion

For every supplied alpha, ordinary quantile regression minimizes check loss
of `Y-D*alpha` on `W=[X,Z]`. The complete exogenous beta and instrument gamma
are estimated anew at each null. The point profile minimizes
`gamma' A gamma`, with fixed `A=Z' M_X Z / N`; instrument rescaling and
nonsingular changes of instrument basis preserve this criterion.

Powell's residual-space rectangle uses `f_i=I(|r_i|<=h)/(2h)`,
`H=sum(f_i W_i W_i')` and complete QR covariance
`C=tau*(1-tau)*H^-1*(W'W)*H^-T`. This estimates a conditional density
weighted information matrix; it does not replace it with a common density.
At the true alpha, the excluded-coefficient statistic
`gamma' C_gamma,gamma^-1 gamma` has an asymptotic chi-square(q) null law,
even with weak instruments, under the stated iid smooth-density QR
regularity. This is the Wald inversion in
[Chernozhukov, Hansen and Jansson (2007), pp.273–274](https://eml.berkeley.edu/~mjansson/Papers/ChernozhukovHansenJansson07.pdf).
It is not an exact finite-sample test or their distinct simulated sign-pivotal
test. All nuisance covariance blocks are retained.

Each declared grid point is tested. Accepted runs describe consecutive
accepted **tested points**. Intervening alpha values remain untested; the
outer tails remain unknown, including when an accepted run touches a grid
boundary. Empty/disconnected tested sets are preserved. No interpolation,
convexification, certified continuous boundary or unbounded-tail claim is made.

Default `inference="weak"` returns no structural Wald covariance, standard
errors or confidence intervals. Nuisance QR covariance and intervals refer to
the adjusted-outcome regression at the selected grid point and are labelled
separately. They are not structural alpha inference.

## Declared identified local inference

`inference="identified"` explicitly declares local strong identification and
consistent profile search. A finite full-rank sample cannot certify them.
There must be one strict interior coarse minimum. One-instrument fits refine
one scanned sign-changing gamma root by bisection. Overidentified fits use
golden-section refinement conditional on local unimodality. The recorded
local bracket does not certify unscanned roots or a global overidentified
minimum. Its width must be at most one percent of the structural alpha SE.

The full covariance follows the inverse-QR linearization. Let
`t=-H^-1 sum(f_i W_i D_i)` and
`a=-(t_gamma' A t_gamma)^-1 t_gamma' A`.
Map nuisance QR noise into `delta_alpha=a*delta_gamma` and
`delta_beta=delta_beta_QR+t_beta*delta_alpha`. The complete resulting influence
map L gives `C_theta=L*C*L'`, including every alpha/beta cross term. The
just-identified result equals `J^-1*S*J^-T`; positive covariance and local
Jacobian gates are required. This local normal law is based on
[Chernozhukov–Hansen, Theorem 3 and Remark 4](https://web.mit.edu/vchern/www/papers/ch_iqr_inference.pdf)
and the stated profiling algebra. It is not weak-identification robust.

Explicit positive residual-space `bandwidth` is preferable for replication.
Otherwise, each profile uses
`1.06*min(sample_sd(residual),IQR(residual)/1.349)*N^(-1/5)`.
Both the bandwidth and complete density support/bread are saved. A limiting
bandwidth sequence must shrink while sufficient kernel observations remain;
a single finite bandwidth is not a proof of asymptotic calibration. Singular
density bread, ambiguous/tied LP vertices and exhausted work cause explicit
refusals rather than a covariance repair or skipped grid point.

## Inputs and resources

Inputs are a pandas DataFrame with distinct named numeric roles. Booleans,
complex/categorical/string coercion, collinear term omission and nonfinite
cells are refused. `missing="drop"` explicitly removes incomplete physical
rows while retaining their original source/index and positional metadata.
No-intercept fits, empty included controls, nullable numeric dtypes and
duplicate/multilevel row indexes retain their declared identities.

The original sample has at most 4096 rows and at least 32 complete rows, with
at most 12 included regressors, eight excluded instruments and 16 QR columns.
Absolute input values are bounded by 1e12. Alpha has 3–257 strictly increasing
finite points in [-1e8,1e8]; tau is in [0.05,0.95]. All mandatory grid fits
must succeed. At most 1024 evaluations are admitted. Native QR has 200
interior iterations and `200+20*width` pivots per profile.
`max_work` preadmits these worst declared costs; named live source/QR/profile
buffers respect the common workspace budget. This is not a process-RSS limit.
No global Torch dtype/device or random state is changed.

Weights, clusters, panels, multiple endogenous regressors, Dataset collection,
GPU, process bands and QARDL are unsupported. There is no runtime dependency
on SciPy, statsmodels, Stata or another estimator library.

## Complete persistence and queries

Complete state includes all raw selected source columns and original dtypes,
typed index, physical sample, specification, profiles, QR basis/dual witness,
criterion, bandwidth/bread, full nuisance covariance, test p-values,
grid topology, search trace and joint identified covariance. Nothing is thinned.
`ivqreg_restore` replays the deterministic search, unique LP primal/dual
optimality certificates and every numerical matrix without QR optimization.
Covariance and bread comparisons are normalized by coordinate scales;
dimensionful scalar comparisons use relative precision without an absolute
floor. Dimensionless LP duals and test statistics retain explicit normalized
roundoff tolerances. Thus small outcome units do not hide altered uncertainty.
Generic summary tables are reconstructed from validated state. Saved iteration
counts are bounded historical diagnostics, not re-executed iterations.
SHA-256 detects corruption and is not authentication. State and JSON are
bounded to 32 MiB with preallocation resource checks.

Saved queries use `D*alpha+X*beta` on all supplied complete physical rows.
They preserve exact query index and return full parameter Jacobians. In
identified mode, the complete fixed-design cross-query delta covariance and
normal conditional-quantile intervals are retained, for at most 512 rows.
Weak mode permits up to 4096 rows with no structural Wald uncertainty.
Neither route supplies future-observation intervals or cross-tau interpolation.

## Evidence boundaries

`tests/test_ivquantile_oracles.py` compares every coefficient/profile objective,
full QR covariance and weak-ID statistic to independent SciPy linear programmes
and NumPy matrix algebra, including just-/overidentification and full joint
prediction covariance. Seeded weak/strong-instrument null experiments are
bounded sampling checks rather than universal finite-sample guarantees.
`scripts/verify_ivquantile_oracles.py` independently audits complete saved
states without importing OpenEconometrics or Torch. Licensed Stata execution,
frozen-runtime/installed application acceptance and release evidence remain
separate; no vendor parity flag is enabled.
