# Proper-prior masked and scheduled linear Gaussian state space

Eight bounded acceptance gates under MARKET-216 (MARKET-401–408). These extend
`sspace`; previous complete-observation fixed-system acceptance is not counted again.

`y[t] = Z[t] a[t] + d[t] + e[t]`, `a[t+1] = T[t] a[t] + c[t] + u[t]`.
Independent Gaussian shocks have full covariances H[t], Q[t]. `a0,P0` describe
the proper prior before the first observation. Q/P0 may be PSD; H is PD.

Use `missing="mask"` to retain all dates and update on the observed principal
measurement submatrix. A completely missing date has likelihood contribution
zero but retains its transition. `missing="raise"` remains the default. Dates
and exogenous columns must be complete. `ModelSpec.options.observation_missing`
records the masked observation policy separately from ordinary sample exclusions.
NaN cells become JSON null in persisted observations/innovations; masks/counts
remain explicit. N is the number of retained calendar periods; observed cell
count is also reported. AIC/BIC retain the period-count convention.

A system's `schedules` dictionary replaces any T/Z/Q/H/c/d static matrix with
one explicitly supplied array per **original physical input row**. Reordering
by `time` also reorders schedules. T[t],c[t],Q[t] are outgoing: after y[t] they
propagate the state to t+1, including the final transition into the forecast
origin. No inferred calendar hole or missing row is compressed.

`exogenous={"state": {"columns": [...], "coefficients": [...]},
"measurement": {"columns": [...], "coefficients": [...]}}` adds fixed known
linear mappings to c[t] and d[t]. Coefficients are state-by-column or
measurement-by-column; column order is saved. Free parameters cannot occupy
an explicitly scheduled matrix. Stationary initialization permits observation
schedules, but rejects changing T/Q/c or state exogenous paths. The fitted
system, static physical base, templates, observations, periods, masks,
parameter covariance and sample positions are retained. A digest detects
accidental saved-state changes; it is not authentication.

`sspace_smooth(result)` supplies full RTS state covariance.
`sspace_autocov(result)` supplies Cov(a[t+1],a[t] | all observed Y), with n−1
entries and explicit orientation. `sspace_disturbances(result)` supplies full
conditional process and measurement covariance, their cross covariance and
joint covariance. Missing correlated measurement noise is conditioned on the
observed noise subvector; it is not silently replaced by zero residuals. The
last outgoing process shock is unobserved: its mean remains zero and covariance
Q[last]. PSD-compatible conditional solves use a documented numerical rank
threshold, check nullspace compatibility and add no ridge/eigenvalue clipping.

These helpers accept a ResultBundle, its JSON object or JSON string. Forecasts
use a restored ResultBundle and `oe.forecast(result,steps,future={"schedules":
{...},"exogenous": {...}})`. Every fitted scheduled key and exogenous column
requires an exact future path. Full state/measurement moments and first-response
normal prediction intervals include conditional state and future shock
uncertainty. They exclude parameter uncertainty. Actual last retained calendar
period and physical row are recorded as origin.

ML maximizes the exact observed-cell Gaussian likelihood using Torch analytic
score/Hessian. The inverse full observed information is transformed to physical
parameter units, including off-diagonal covariance. Failed convergence or
nonpositive information does not justify a fitted inferential model.

Domain: CPU float64 resident data, ≤20,000 dates, ≤16 states, ≤8 measurements,
≤20 free parameters; explicit 50-million work admission and workspace plans.
Known exogenous coefficients only. No automatic fitting of scheduled matrices,
exact diffuse initialization, nonlinear model, Dataset, CUDA/MPS, weights,
parameter-uncertain forecasts or blanket vendor parity claim.

## References and proof boundaries

The general time-varying equation/timing follows the official
[statsmodels state-space representation](https://www.statsmodels.org/stable/statespace.html).
Helske's [KFAS author paper](https://arxiv.org/pdf/1612.01907) supplies theory
for missing observations and smoothing. Source tests instead build and condition
small dense joint Gaussian distributions directly; they do not delegate
production calculation to statsmodels/KFAS. Numerical original-author KFAS
replication, licensed Stata/EViews outputs and exact diffuse likelihood
normalization remain separate open gates. Source oracles, frozen-runtime
identity, installed native Run and full restart persistence are recorded
separately in `docs/evidence/sspace-scheduled-2026-10-07/`.
