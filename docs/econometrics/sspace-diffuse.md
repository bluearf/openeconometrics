# Exact diffuse Gaussian state-space procedure

`sspace_diffuse` evaluates **fixed caller-declared** linear Gaussian equations.
It adds exact diffuse initialization to the existing proper-prior baseline;
the known/stationary `sspace` estimator is not changed or rerouted.
This stage does not complete MARKET-216, advanced time series or vendor parity.

```python
result = oe.sspace_diffuse(data, y="measurement", time="month", system={
    "a0": [0.0], "P_inf": [[1.0]], "P_star": [[0.0]],
    "Z": [[1.0]], "T": [[1.0]], "Q": [[0.2]], "H": [[0.5]],
})
result["smoothed"]
saved = result.to_json()
restored = oe.diffuse_restore(saved)
forecast = oe.diffuse_forecast(restored, future_system=future_paths,
                              future_time=future_months)
```

The result is a `DiffuseResult`, a real `TableSet` containing filtered states,
smoothed states and period likelihood/rank diagnostics. `to_latex()` exports
the complete named tables. `to_json()` retains full state, including all
cross-state/cross-date posterior and disturbance covariance. There is no
fictional coefficient estimate, coefficient SE, df, p-value or optimizer
convergence claim. State SD and pointwise normal limits are conditional Gaussian
posterior moments at the requested `alpha`; fixed-system forecasts include
state/process/measurement uncertainty and exclude parameter uncertainty.

## Equations and exact diffuse measure

At date t, the state `a[t]` precedes the measurement:

`y[t] = Z[t] a[t] + d[t] + e[t]`, `e[t] ~ N(0,H[t])`.

The outgoing transition is
`a[t+1] = T[t] a[t] + c[t] + u[t]`, `u[t] ~ N(0,Q[t])`.
Independent process and measurement shocks are assumed. The initial prior
has covariance expansion `P(kappa)=kappa P_inf+P_star` as kappa tends to
infinity. `P_inf` is an actual finite PSD matrix, not a large finite `P0`.
`P_star` is the proper initial noise covariance. Both are supplied and saved.

For an active scalar innovation, put `F_inf=Z P_inf Z'`,
`F_star=Z P_star Z'+H`, `C_inf=P_inf Z'`, `C_star=P_star Z'`.
The exact update is

```
a       <- a + C_inf v / F_inf
P_inf   <- P_inf - C_inf C_inf' / F_inf
P_star  <- P_star - (C_inf C_star' + C_star C_inf') / F_inf
                     + F_star C_inf C_inf' / F_inf**2
```

The diffuse likelihood contribution is `-log(F_inf)/2`. It contains no
innovation quadratic or `log(2*pi)` term for that diffuse observation. Ordinary
identified observations have the complete usual Gaussian contribution.
This is the [KFAS exact-diffuse convention](https://github.com/cran/KFAS/blob/66aab472a9d29c08c52a8ebcd3da531c9daadba3/src/filter1step.f90).
It is not a proper marginal density for comparing models with different
diffuse ranks or initial normalization. Rescaling `P_inf` changes that
normalization while leaving the identified conditional moments unchanged.

For simultaneous correlated measurements, a rank-revealing principal block and
determinant-one innovation transform isolate the diffuse subspace. The remaining
proper innovations are conditioned jointly, including all cross blocks.
There is no assumption that H is diagonal, and no observation order shortcut
that discards correlated measurement noise. Diffuse rank reductions, proper and
diffuse observation counts and the finite/infinite covariance paths are saved.

## Exact smoother and singular laws

The smoother builds the proper joint covariance and loading of flat initial
conditions. With observed covariance `Omega` and diffuse design `D`, it solves
the Gaussian saddle-point system `[[Omega,D],[D',0]]`. The resulting universal
Gaussian predictor gives full state means and cross-date covariance without
a finite large prior. Lag-one covariance orders `Cov(a[t+1],a[t]|Y)`.
Process and measurement disturbance means, covariance, state cross blocks and
joint process/measurement covariance are retained. The final outgoing state
and unobserved final process disturbance are included.

P_inf, initial P_star, Q and H may be PSD. Deterministic state directions and
a noise-free measurement that consumes a diffuse direction can be admitted.
Every remaining **proper observed innovation** must have a nondegenerate
Gaussian density. Singular proper residual laws are refused explicitly;
compatible rank-zero point masses are not assigned a made-up Gaussian density.
There is no ridge, jitter, PSD projection or covariance repair. Unobserved
diffuse directions cannot vanish through a singular outgoing transition and
be treated as learned. All initial diffuse directions must be observed before
successful smoothing, saved results or forecasting. A lower-level diagnostic
filter can retain an explicitly improper terminal state when requested.

## Sample, calendar, portable state and budgets

CPU float64 numerical work is explicit; ambient Torch dtype/device/RNG settings
are preserved. Resident response tables, 1..8 real measurement columns and
1..16 states are supported. Numeric or datetime calendar columns must be
complete, distinct, regular and already chronological. Dates are never sorted,
deleted or compressed. `missing="mask"` keeps partial and entirely missing
measurement dates; all-missing periods still perform their outgoing transition.
`missing="raise"` refuses them. Typed ordinary/Range/Multi/Categorical/timezone
row indexes, original physical positions and complete calendar metadata survive
portable restoration. Schedules have exactly one array per original date.
Datetime calendars need at least three dates so their cadence can be inferred;
numeric calendars need at least two dates. Integer calendar differences are
checked as exact integers. An undeclared short calendar is refused at fitting,
before it can produce an ambiguous future continuation.

Low-level filtering admits at most 20,000 periods subject to geometry limits.
Exact batch smoothing is separately bounded to 256 periods, 1,024 stacked
state coordinates and 512 conditioning equations, with a 50-million work
ceiling. These are declared implementation/resource guards, not statistical
limits. Named tensor/workspace buffers, input conversion, derivative graphs,
rank factors and joint conditioning are admitted before factorization. The
global `OPENECON_WORKSPACE_MB` contract also applies. Portable numeric state and
typed identity serialization have named buffer estimates; private
allocator/BLAS overhead remains outside those estimates. Raw resident row/column
geometry and work are checked before DataFrame construction or copying. Mapping
inputs project only declared responses/calendar columns; Dataset collection,
row-record iterators and future-label generators are refused. Nonexact integer
measurements cannot be silently rounded to float64.

Typed row/calendar metadata are separately bounded to 2 MiB, strings/bytes to
1,024 source characters and 4,096 encoded characters, and tuple nesting to 16
levels. Saved JSON is bounded to 64 MiB and four million items before digest or
copying, with text size admission preceding parsing. Replay checks declared
real numeric response dtypes by exact reconstruction of every measurement.
Canonical cached-output comparison distinguishes Booleans, integers and floats.

Diffuse rank decisions use covariance coordinate equilibration, so changing a
state's units does not erase a small independent diffuse direction. Initial
near-collinear ranks that cannot be resolved in float64 are explicitly refused.
Derived covariance rank is checked against the known algebraic rank bound;
matrix moments are never rescaled, clipped or projected. Negative input
covariance diagonals and covariance with zero-variance components are refused.
Public filtered tables omit means/variances/intervals for still-improper state
components; their finite expansion coefficients remain in the complete state.

The versioned core `openecon.sspace.exact-diffuse.v1` saves full measurements,
masks, equations, P_inf/P_star, ranks, normalization, likelihood contributions,
posterior/noise moments and a digest. The public
`openecon.sspace.diffuse-result.v1` adds response/state labels, source dtypes,
typed indexes, original calendar and display alpha. Checksums detect accidental
mutation; they do not authenticate a dataset or its scientific assumptions.
Restoration checks source geometry before tensor allocation and recomputes
fixed-system moments without an optimizer. Sufficient ambient workspace budget
changes do not alter the statistical state.

Future forecasts require all six explicit equation paths. A saved calendar
also requires explicit future dates that continue it without gaps. No future
observation is used to refit anything. Restored full forecast means/covariance
and LaTeX disclosures preserve the conditional target. Weights, Dataset
collection, CUDA/MPS, nonlinear systems, endogenous shock correlation, fitted
schedule/exogenous parameters, parameter-uncertain forecasts and automatic
dynamic-factor/DSGE construction remain outside this stage.

## Independent acceptance

`tests/test_sspace_diffuse.py` compares complete likelihood, filtered prefixes,
full posterior covariance and disturbances against an independently constructed
NumPy primitive-noise/flat-prior GLS reference. Cases include correlated
measurements, schedules, partial/full diffuse rank, leading/interior/all missing
dates, unit roots, measurement permutation, rotated/scaled diffuse priors,
proper-prior reduction, deterministic learned states, independent score/Hessian
differences, portable replay, typed calendar/indexes and budget/tamper refusals.

`scripts/verify_sspace_diffuse_oracles.py` executes the actual **native R KFAS
1.6.0 package** as a development-only author reference for supplied synthetic
fixed systems, recording versions, GPL license, compiled-library, input and
executed-code hashes. The [KFAS author paper](https://www.jstatsoft.org/article/view/v078i10)
provides the reference package/replication materials. This is an independent
algorithm comparison, not reproduction of published empirical parameter fits.
`tests/test_sspace_diffuse_kfas.py` retains the actual native author outputs for
six fixed systems, including scheduled correlated measurements and missing
cells. All state/filter/likelihood and original disturbance moments pass at
absolute/relative tolerance `2e-10` (observed maximum absolute moment difference
`2.51e-13`). KFAS's documented overlap requires diagonal 0/1 `P1inf` and zero
finite-prior rows/columns for diffuse states; constructed model readback checks
every input matrix before execution. Its `transform="augment"` convention
exposes original correlated measurement shocks as appended states, including
missing-cell conditional means, full covariance and state/noise cross covariance.
Rotated/scaled diffuse priors outside this author package's admission domain
continue to use independent closed-form/GLS validation.
The earlier WebR package attempt produced incorrect filter results and remains
diagnostic only; it supplies no passing author evidence.

Source/oracle, frozen source identity, installed native Run/export/restart,
licensed vendor output and a public signed release are separate proof layers.
No blanket Stata/EViews parity flag is enabled by this procedure.
