# Advanced state-space and structural macro stages

Planning: MARKET-188. Implementation tracking remains [GitHub #76](https://github.com/bluearf/openecon/issues/76).
Baseline: `40add89`; candidate PR91 head `830b9c3` is open and unmerged.
Existing `sspace`/`sspace_filter` support fixed linear Gaussian systems with
proper known or exact stationary priors, bounded states/measurements/free
parameters and conditional forecasts. UCM, ARIMA and ETS are additional
foundations. Current general sspace rejects diffuse/time-varying systems and
partial measurements. The PR91 eight-line difference only adds a docstring that
advertises missing measurements; the actual kernel contains no observation-mask
logic. Reuse and extend the shared filter in place; do not rewrite a competing kernel.

## T1 — observation/initialization stages ([MARKET-216](https://linear.app/bluearf/issue/MARKET-216))

First implement observation-mask handling with sample preparation and full
filtering/smoothing tests; the PR91 description is not an implemented candidate.
A partially observed date uses the matching
rows/submatrix of Z/H; an all-missing date performs transition-only propagation.
Time does not compress across missing periods. Validate proper-prior likelihood,
conditional state covariance and a full Rauch–Tung–Striebel or equivalent
smoother against joint Gaussian conditioning; retaining filtered states alone
does not implement smoothing.

Then add explicitly declared T/Z/Q/H/c/d schedules and exogenous mappings,
recording timing, dimensions, known vs estimated cells and future exogenous
requirements. Persist the complete equations/matrices or immutable referenced
schedules, observation masks, parameter transforms/order and forecast origin.
An invalid covariance/rank fails; do not sanitize a system by PSD projection.

Exact diffuse initialization is a separate gate: retain P_inf/P_star and
diffuse-rank transitions, initialization likelihood convention and smoothing
normalization. An arbitrary huge finite P0 does not implement it. Handle unit
roots, unobserved diffuse states and singular information explicitly. Recover
each supported UCM specification through the general interface and compare
means, likelihood, full covariance and forecasts. Separate process/state and
parameter-estimation uncertainty; forecasts without the latter disclose it.

Original analytic covariance cases and the KFAS author's exact-diffuse reference
are independent gates; actual numerical replication is pending. Carry forward
CPU float64 resident state/measurement/parameter/derivative budgets, then publish
measured limits for schedules and smoother storage. Dataset/GPU and nonlinear
systems remain unsupported until separately validated.

## T2 — dynamic-factor stages ([MARKET-217](https://linear.app/bluearf/issue/MARKET-217))

Declare factor count, loading normalization/rotation/sign convention, factor and
idiosyncratic dynamics, temporal aggregation, vintage and publication timestamps.
Stage static Gaussian factor embedding first, serial idiosyncratic terms next,
then mixed-frequency ragged data and model-based news. Factor labels are not
identified without normalization. News is not the same as revising a forecast
with future information that was unavailable at the original origin.

Use the Bańbura–Modugno arbitrary-missing EM methodology as a future author
reference. Record EM objective history, starts, identification/Hessian,
convergence and failure; preserve full covariance and state uncertainty.
Require author fixture/license/hash acquisition and independent likelihood,
smoother and reporting-unit information checks. Budget EM work and factor
geometry; never infer unlimited Dataset support from a small matrix example.
Current PCA/EFA does not establish a dynamic-factor contract. The historical
Sonra decision is preserved.

## T3 — DSGE feasibility ([MARKET-218](https://linear.app/bluearf/issue/MARKET-218))

The historical Hayır/out-of-scope decision remains. A research milestone can
establish feasibility without promising a product estimator. First specify
linear rational-expectations equations, timing, shock/expectational-error maps
and solution residuals. Validate generalized-eigenvalue existence/uniqueness
conditions against Sims's original equations/code; indeterminate/no-equilibrium
systems must not produce a pretend unique model. Audit threshold sensitivity,
unit roots, near-singularity and known deterministic solutions.

A native decomposition strategy must be demonstrated before a runtime API;
SciPy QZ is a development oracle, not an implicit estimation adapter. Only after
solution replication consider structural identification, observation equations,
likelihood and uncertainty. Nonlinear/local perturbation, alternative solution
domains and Bayesian posterior inference are separate research/scope gates.
Persist steady state, equations, normalizations, eigenvalue policy, solution
status and shock mappings. None of these methods is implemented by this plan.

## Primary reference gates

- [Helske, KFAS paper](https://www.jstatsoft.org/article/view/v078i10): author reference for state-space methods including exact diffuse initialization.
- [Bańbura–Modugno ECB author working paper](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1189.pdf): arbitrary-missing dynamic-factor estimation.
- [Sims's original rational-expectations work/code index](https://www.princeton.edu/~sims/): source for solution and existence/uniqueness gates.
- [EViews official state-space specification manual](https://help.eviews.com/content/sspace-Specifying_a_State_Space_Model_in_EViews.html): separate licensed vendor equation/timing comparison target.

The [shared promotion gates](README.md) apply to each option stage. Vendor runs,
author numerical reproduction and new frozen/native estimator acceptance are
pending; completion of MARKET-188 records planning only.
