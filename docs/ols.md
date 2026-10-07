# OLS and weighted least squares

OpenEconometrics implements the OLS family directly in float64 PyTorch. It does not call
statsmodels, SciPy, Patsy, linearmodels or NumPy statistical routines. Pandas
provides the data interface; its own transitive dependencies remain installed.
The same Python API works in a script, the workbench and the desktop application.

## Estimation

```python
import openecon as oe

model = oe.ols(
    data=df,
    formula="wage ~ education * C(region) + experience + I(experience**2)",
    covariance="cluster",
    cluster=["firm", "year"],
)
display(model)
```

Alternatively use `y="wage", x=["education", "experience"]` and specify
categorical columns with `categorical=["region"]`. Intercept-only and
no-intercept models are supported. Collinear terms are omitted deterministically
and recorded in `model.provenance["omitted_terms"]`.

Formula operations are allowlisted AST expressions, never Python `eval`.
Supported operations are `+`, interactions `:`, factorial expansion `*`, `I()`,
`C()`, `log()`, `exp()`, `sqrt()`, `L(column, lag)` and `D(column, lag)`.
Time operators require `time="period"`; they use actual integer periods, so a
missing period produces a missing lag. Predictor history is evaluated before
dropping missing outcomes or zero-weight observations. Streamed time operators
require ascending, unique periods.

Missing model observations are dropped by default; `missing="raise"` makes the
selection explicit. Formula-generated nonfinite values follow the same rule.
Unknown category levels in new prediction data are rejected. Fitted category
levels, treatment references and model terms are retained for reproducibility.

## Weights and covariance

Use a column name plus its statistical interpretation:

```python
model = oe.ols(data=df, y="y", x=["x1", "x2"],
               weights="sampling_weight", weight_type="pweight")
```

Supported weight types are `aweight`, `fweight`, `pweight` and `iweight`.
Frequency weights are integer multiplicities, handled without physically
expanding rows. Zero weights are excluded; negative weights are rejected.
Sampling weights require a robust covariance estimate and default to HC1.

| `covariance` | Implementation and inference |
| --- | --- |
| `nonrobust` | Classical OLS/WLS; Student t, residual degrees of freedom |
| `HC0`, `HC1`, `HC2`, `HC3` | Heteroskedasticity-robust sandwich estimates; HC1 includes finite-sample scaling |
| `cluster` | One to four cluster dimensions; CGM inclusion–exclusion, each intersection's own CR1 correction; minimum marginal cluster df |
| `cluster_hc2`, `cluster_hc3` | One-way cluster leverage correction through small parameter-space Gram matrices |
| `hac` | Bartlett, Parzen, quadratic-spectral or truncated kernel; integer time gaps are respected |
| `bootstrap` | Seeded pairs or one-way cluster bootstrap; frequency resampling preserves multiplicity; normal inference |
| `jackknife` | Analytic observation or one-way cluster deletions; no row-expanded data |

The ordinary default is classical covariance. `robust` is an alias for HC1.
Supplying `cluster` without an explicit covariance selects cluster covariance.

If a multiway cluster estimate has negative covariance eigenvalues, they are
clipped in raw coefficient coordinates, and this correction is recorded in the
result. Such a projection is coordinate dependent; changing predictor units can
change the corrected estimate. Prediction and contrast calculations preserve
that reported projection rather than silently substituting a different one.

HAC accepts `time="period", lags=4, kernel="bartlett"`. `lags="auto"` uses
the Newey–West (1994) plug-in rule for Bartlett, Parzen and quadratic-spectral
kernels. The truncated kernel requires an explicit bandwidth.

`dfadjust=True` enables Bell–McCaffrey contrast-specific degrees of freedom for
HC2/HC3 and their cluster counterparts. `hansen=True` enables the HC3 adjustment
to t statistics and confidence intervals. Single-coefficient, single-contrast
and one-slope model tests use these adjustments. Joint Wald tests report their
conventional fitted degrees of freedom explicitly; they do not pretend to be a
joint contrast-specific approximation.

```python
robust = oe.ols(data=df, y="y", x=["x1", "x2"],
                covariance="cluster_hc3", cluster="firm", hansen=True)
bootstrap = oe.ols(data=df, y="y", x=["x1", "x2"],
                   covariance="bootstrap", reps=999, seed=123)
```

Seeded resampling is reproducible for the recorded draw algorithm and device.
Streamed pairs/cluster resampling uses exact conditional-binomial multinomial
counts; dense resampling uses integer draws with count reduction. These have
the same sampling law but need not generate identical realized replications
for the same seed. The draw algorithm is recorded in covariance provenance.

## Inference and postestimation

Results include coefficients, standard errors, p values, confidence intervals,
the covariance matrix, the model test, R-squared, adjusted R-squared, residual
degrees of freedom, Root MSE, sums of squares and Gaussian likelihood criteria.
Existing publication-quality LaTeX exports remain available through
`model.to_latex()`, `oe.esttab(...)` and dataframe `to_latex()`.
Table notes identify the weight interpretation, effective observation count,
physical estimation rows, covariance method, resampling settings and any
contrast-specific degrees-of-freedom or Hansen corrections.

```python
model.test(["education", "experience"])
model.testparm("region*")
model.lincom({"education": 1, "experience": -1})
model.nlcom(lambda b: b["education"] / b["experience"])
model.predict(data=new_df, interval="mean")
model.predict(data=new_df, interval="observation")
model.margins(variables=["education"], method="ame")
model.margins(variables=["education"], method="mem")
model.vif()
model.hettest()
model.white_test()
model.reset_test(powers=(2, 3, 4))
model.breusch_godfrey(lags=2)
model.diagnostics(lags=2)
```

Predictions include fitted values, residuals, scores, prediction standard errors,
leverage, standardized/studentized residuals, Cook's distance, DFBETAs,
covariance ratios, DFITS and Welsch distance where the fitted covariance and
weight interpretation support the statistic. Classical prediction/influence
statistics enforce their assumptions. Heteroskedasticity diagnostics are
available for unweighted and frequency-weighted models; serial-correlation tests
are unweighted. The diagnostic bundle reports unavailable combinations with a
reason, rather than silently changing the model.

Marginal effects support numeric derivatives, categorical discrete changes,
interactions, average marginal effects, effects at means and `at` grids. Delta
method covariance is computed from the averaged gradient, not averaged standard
errors. `nlcom` uses PyTorch automatic differentiation. Margins on lag/difference
formulas require an explicitly defined intervention over time and are rejected;
use `lincom` on their fitted coefficients instead of an ambiguous derivative.

## Large local data and devices

```python
source = oe.scan("observations.parquet")
model = oe.ols(data=source, y="y", x=["x1", "x2"], covariance="HC3")
for predictions in model.iter_predict(batch_rows=65536):
    # Save or consume each batch without collecting the whole result.
    ...
```

All listed covariance families have bounded source implementations. Streaming
centers/scales the design, uses balanced TSQR, and replays projected data for
covariance and postestimation. Cluster scores and contrast adjustments spill to
local SQLite files. Resampling generates counts during each replay and does not
retain an N-sized vector. Source content, order and physical retained positions
are checked between passes. Only a 400-row prediction preview is stored in a
serialized result. Full predictions and influence rows use iterators; streamed
diagnostics retain compact summaries.

The numerical work budget is 128 MiB with up to 384 expanded parameters.
Category dictionaries, lag histories and reader batches have separate bounded
budgets. Python, PyTorch, pandas, caller-provided data, file-parser overhead and
external batch factories contribute additional process memory. There is no
total row ceiling; practical capacity depends on disk space and model structure.
Exact quadratic-spectral HAC has quadratic pair work;
bootstrap cost grows with replications. Bounded memory does not make these
algorithms linear-time.

`device="auto"` selects CUDA for dense models when available; explicit
`device="cpu"` or `device="cuda:0"` can be used. Streamed sources use CPU TSQR
with `auto`; explicit CUDA streaming is rejected. Apple's MPS device does not
provide this engine's float64 requirements. The verification host has no CUDA
hardware, so GPU tests are skipped rather than reported as hardware validation.

## Validation and boundaries

The coefficient and classical-inference results match the published Stata
`auto` example (`mpg` on `weight` and `foreign`): N=74, R-squared=0.6627029,
Root MSE=3.4070593 and F(2,71)=69.74846. Advanced covariance formulas are tested
against independently constructed matrix/deletion/replication oracles; SciPy
and statsmodels are test references only. Dense and streamed implementations
are compared across weights, category expansions, time gaps and batch sizes.

The 0.3.6 validation ran 3,140 Python tests successfully, with five real-CUDA
tests skipped on an Apple M3 Pro. Eleven additional desktop-to-cloud contract
regressions also passed against the pinned release source. Prediction and contrast
inference also have independent centered-reference checks for offsets of `1e8`, arbitrary intercept
components, no-intercept designs and extreme predictor units. These checks do
not assert unit invariance for a raw-coordinate CGM covariance projection.

A 10-million-row, three-predictor analytic-weighted HC3 fit took 5.82 seconds
with about 308 MiB peak process RSS and three source passes on that host using
one Torch thread. The source generated synthetic observations in 65,536-row
batches. The timing includes batch generation and model passes; it excludes
imports, disk reads and UI work. Reproduce the benchmark with
`python benchmarks/ols_full.py --output benchmark.json`. It is a bounded-source
benchmark, not evidence for every estimator's speed.
Desktop archives were verified live on Cloud Run revision `openecon-00013-q82`
with fresh disposable owner/viewer identities. Ordinary, weighted and HAC
results retained their model tests, metadata, D3 charts, LaTeX and event order;
malformed models returned 422 and viewer writes returned 403. Configuration,
IAM and the remote compute image remained unchanged. Cleanup readback found no
remaining fixture resources. Release evidence is recorded in
[OLS and Mac 0.3.6 verification](evidence/ols-0.3.6.json).

This is not a claim of complete Stata numerical parity. Complex survey designs,
Bayesian/MI/prefix machinery, absorbed high-dimensional fixed effects and
restricted least squares are distinct estimator families. Multiway resampling
requires a separate defined resampling scheme and is rejected here; one-way
resampling is supported. Singular deleted cluster designs can make Hansen
inference unidentified and are rejected explicitly. Perfect-fit/degenerate
inference and models beyond the declared width/history budgets produce
structured errors.

Primary method references: [Stata regress](https://www.stata.com/manuals/rregress.pdf),
[Stata regress postestimation](https://www.stata.com/manuals/rregresspostestimation.pdf),
and [Hansen (2025), Appendix A](https://users.ssc.wisc.edu/~bhansen/papers/jae_25.pdf).
