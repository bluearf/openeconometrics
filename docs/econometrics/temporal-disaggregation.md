# Eight bounded temporal-disaggregation methods

The APIs reconstruct a complete high-frequency series from known low-frequency
benchmarks and supplied indicators. Every result reaggregates to the benchmarks
under the declared rule. The reconstructed values are estimates of unobserved
values; they do not become observed data or receive a nonnegativity guarantee.

```python
import openecon as oe

result = oe.denton_cholette(
    [410.0, 430.0], [85.0, 95.0, 105.0, 110.0, 90.0, 100.0, 115.0, 120.0],
    low_periods=["2024", "2025"],
    high_periods=[f"{year}Q{quarter}" for year in (2024, 2025) for quarter in range(1, 5)],
    criterion="proportional", aggregation="sum",
)
result["series"]  # high_period, low_period, value
result["aggregation"]  # observed, reconstructed, error for each low period
result.to_latex()  # every result table, including complete JSON settings
```

## Methods and boundary conditions

Write C for the explicitly constructed aggregation matrix, b for the benchmarks,
x for one indicator, and Δ for the square first-difference operator whose first
row is [1,0,...]. Each original Denton method minimizes the stated squared
adjustment subject to Cy=b. The first row penalizes the initial adjustment;
second differences also retain their two initial rows.

| API | Criterion | Initial condition |
| --- | --- | --- |
| `denton_additive` | ‖Δ(y−x)‖² | zero presample additive adjustment |
| `denton_additive_second` | ‖Δ²(y−x)‖² | zero presample additive adjustment and its initial difference |
| `denton_proportional` | ‖Δ(y/x−1)‖² | zero presample relative adjustment |
| `denton_proportional_second` | ‖Δ²(y/x−1)‖² | zero presample relative adjustment and its initial difference |
| `denton_cholette` | first differences of y−x or y/x−1, selected by `criterion` | initial adjustment is free; the constant null direction is identified by benchmarks |
| `chow_lin` | GLS regression plus residual distribution, required fixed `rho` | stationary AR(1) residual process |
| `fernandez` | GLS regression plus residual distribution | zero-start random-walk residual process |
| `litterman` | GLS regression plus residual distribution, required fixed `rho` | zero-start integrated AR(1) increments |

Denton accepts exactly one indicator. Proportional criteria require it to be
strictly positive; additive criteria allow signed inputs. Negative reconstructed
values are retained and their count is reported. No clipping or ridge is used.
The Cholette API implements the first-difference initial-transient correction.
It does not implement the broader Cholette–Dagum regression model with arbitrary
covariance, bias correction or extrapolation.

The GLS methods accept up to eight supplied indicator columns and optionally
add an intercept (`intercept=True`). The aggregated design must have full rank
and positive residual degrees of freedom m−p. `rho` is prespecified, not fitted;
its supported range is [−.95,.95]. `alpha` is [1e−8,.5]. Inference excludes
parameter-selection uncertainty. Chow–Lin assumes that the
relationship represented by the aggregated indicators also holds at the high
frequency. Fernandez and Litterman impose their declared finite, zero-start
integrated residual processes; replacing those initial conditions changes the
model.

## GLS estimation and uncertainty

The covariance shape Q has innovation variance factored out. Chow–Lin uses
Q[i,j]=rho^|i−j|/(1−rho²). Fernandez uses Q[i,j]=min(i+1,j+1).
For Litterman, generate innovations v[t]=rho*v[t−1]+epsilon[t] from v[−1]=0,
then u[t]=u[t−1]+v[t] from u[−1]=0; Q is the covariance of u for unit-variance
epsilon. With W=CQC′, Xl=CX and D=QC′W⁻¹:

```
beta = (Xl′ W⁻¹ Xl)⁻¹ Xl′ W⁻¹ b
y_hat = X beta + D(b − Xl beta)
s² = (b − Xl beta)′ W⁻¹ (b − Xl beta)/(m − p)
Cov(beta) = s² (Xl′ W⁻¹ Xl)⁻¹
```

High-frequency uncertainty concerns the error in predicting the latent high
series, not an independently observed outcome or measurement noise. Defining
H=X−D Xl and V=(Xl′W⁻¹Xl)⁻¹, its covariance is:

```
s² [Q − D C Q + H V H′]
```

Both the conditioned residual process and coefficient-estimation uncertainty
are included. Aggregating that covariance gives zero because the benchmark is
known exactly. In `first`/`last` aggregation, the known endpoint has zero
uncertainty. Gaussian, fixed-Q pointwise Student-t intervals use m−p degrees of
freedom and the declared `alpha`; they are not simultaneous confidence bands.
Coefficient tests use that same conditional Gaussian model. The reported
low-frequency Gaussian log likelihood uses the ML scale SSE/m; coefficient
and latent-series inference use the unbiased scale SSE/(m−p). There is no
optimization or convergence claim for fixed-rho GLS.

GLS results add `coefficients`, full `coefficient_covariance`, `low_fit`, `fit`,
full `high_covariance` and `latent_intervals` tables to the common tables.
Denton methods provide deterministic adjustments; sampling SE, covariance,
df, p-values, confidence intervals and likelihood are not applicable.

## Calendar, release and resource contract

Inputs are finite positional vectors or rectangular indicator matrices with
explicit canonical period labels: years `YYYY`, quarters `YYYYQq`, months
`YYYY-MM`. Supported routes are December-end Y→Q, Y→M and Q→M, with complete,
strictly ordered, unique calendars and exactly matching spans. Aggregation is
`sum`, `mean`, `first` or `last`; mean uses the fixed number of high periods in
each low period, not day-duration weights. Unsupported fiscal calendars,
ragged periods, partial endpoint blocks, missing values, daily data and
forecast/backcast spans fail explicitly. Rows are never silently deleted.

Release metadata is optional. If used, `as_of`, `low_releases` and
`high_releases` must all be supplied with aligned lengths and consistent
timezone awareness. Every supplied value must have been released on or before
`as_of`, and the complete high-frequency span must end on or before the date
of `as_of`. A caller-supplied early release timestamp cannot authorize a future
period. A future release fails rather than selecting a different vintage or
truncating the calendar. Without release metadata the result is retrospective
complete-data benchmarking, not a claim of historical information availability.

The route is resident CPU float64, with 2 or more low periods, at most 240 high
periods and at most 8 supplied indicators. Nonzero numeric magnitudes must lie
in [1e−50,1e50]. Dataset/table collection, sampling or regression weights and
other devices are refused. Dense work is bounded by the configured workspace
budget. Linear systems must meet the condition-number limit 1e12 and backward
error gates; aggregation and stationarity gates are also checked. Failure has
no approximate-method or regularization fallback. Complete inputs, releases,
settings, numerical checks and result tables are JSON-persistable.

Each nonzero benchmark must be reconstructed within 2e−9 times its own
absolute magnitude; a large neighboring period cannot hide a failed small
benchmark. Zero benchmarks use the local aggregation backward-error scale.
Known `first`/`last` values are assigned exactly before checking this gate.

GLS also refuses constant supplied predictors in an intercept model, zero
predictors, an aggregated rank failure and numerically degenerate innovation
variance (weighted SSE no greater than 1e−24 times
whitened response energy). These are inference failures rather than permission
to report zero standard errors or silently remove a column. Centering/scaling
is reversed in the reported coefficients and full covariance.

## References and validation scope

[Sax and Steiner (2013), *Temporal Disaggregation of Time Series*](https://journal.r-project.org/articles/RJ-2013-028/)
provides the regression/benchmark-distribution framework and residual-process
definitions. The [author-maintained tempdisagg documentation](https://cynkra.github.io/tempdisagg/reference/td.html)
distinguishes original Denton initial rows, the Cholette correction and fixed
versus fitted rho methods. Its [source repository](https://github.com/cynkra/tempdisagg)
was inspected as a mathematical reference (version 1.2.0); no GPL runtime code
is included. R was not executed, so this is not a claim of runtime agreement
with that package. The [IMF Quarterly National Accounts Manual, chapter 6](https://www.elibrary.imf.org/display/book/9781475589870/ch006.xml)
describes practical movement-preservation benchmarking and distinguishes the
broader Cholette–Dagum family.

`scripts/validate_temporal_disaggregation.py` independently builds NumPy KKT
objectives and whitened GLS reference fits for all three frequency routes and
all four aggregation rules. It compares reconstructed values, complete
coefficient and high-series covariance, conditional t intervals, df, likelihood
and exact benchmark restrictions. Its predeclared Gaussian simulation uses
two fixed seeds and 48 cells of 1,000 replications. Public API basis calls
recover the implemented linear coefficient/prediction maps and covariance
shapes; the 48,000 replications apply those maps in batches, rather than
claiming 48,000 full API calls. Coverage gates account for Monte Carlo error.

These checks establish bounded method-level agreement and conditional-model
coverage. They do not establish licensed Stata/EViews execution, estimated-rho
inference, non-Gaussian robustness, national-accounts accuracy, Windows/CUDA
support or release-package readiness. Source, frozen runtime and native
Run/restart persistence evidence are separate receipts.
