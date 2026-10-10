# Continuous latent SEM: SEM-2 and SEM-3

This stage implements bounded single-group acyclic Gaussian models with
continuous observed and latent variables. It extends the CFA foundation with
structural paths, constrained means, selected correlated disturbances, joint
observed information, standardized solutions, fit measures, path effects and
saved-model regression factor scores. The existing `oe.sem` remains the spatial
error estimator. Proposed public functions use `latent_sem` explicitly.

The full MARKET-360 parent also requires SEM-4 multigroup invariance, MAR FIML
and robust/scaled inference, and SEM-5 generalized and multilevel models.
Those stages and their external/packaged acceptance gates remain outstanding.

## Statistical model and specification

Each named observed or latent node satisfies
`z = intercept + B z + disturbance`. The declared directed graph is acyclic.
For `A = (I-B)^-1`, the joint mean is `A intercept` and covariance is
`A Psi A'`. The observed rows/columns give the fitted likelihood moments.
All observed predictors have a joint estimated random-variable distribution;
there is no fixed-exogenous conditional likelihood option.

`columns` sets the observed order. `factors` maps each latent name to its
indicators and creates directed measurement edges. `paths` maps each target
to source/coefficient entries: `None` means free, and a finite number means
fixed. A measurement edge cannot also be declared as a structural path.
Names may contain spaces but exclude the reserved `:`, `,` and `<-` delimiters.

`identification="marker"` fixes one loading per factor to one in original
observed units. `markers` explicitly chooses those indicators. The separate
`unit_disturbance` option fixes each latent disturbance variance to one and
requires positive anchor orientation. For an endogenous factor this fixes its
innovation variance; its total variance generally exceeds one. This option
must not be described as total unit-factor-variance identification.

Disturbance variances are otherwise free. Undeclared off-diagonal disturbances
are zero. `residual_covariances={("x2", "x3"): None}` estimates that selected
covariance; a finite number fixes it. Arbitrary sparse selected covariance
patterns are allowed if the fitted full disturbance covariance is positive
definite and the complete model passes identification/information gates.

`meanstructure=True` estimates observed intercepts and fixes latent intercepts
to zero by default. `intercepts` can fix an observed marker intercept and free
the associated latent intercept. Freeing latent intercepts without sufficient
observed constraints is unidentified and refuses ordinary inference.
Setting `meanstructure=False` profiles unrestricted observed means and fits
covariance structure only; it does not report latent-mean estimates.

The ordered full parameter map uses names such as `loading:x2<-factor`,
`path:second<-first`, `variance:x2`, `covariance:x2,x3` and `intercept:x2`.
`fixed` can fix an otherwise declared parameter; identification anchors cannot
be overridden. `equalities` maps free same-family parameter names to one label.
Equality constraints apply in original units, including when the two variables
have different numerical scales. The original-unit parameter table/covariance
has one entry per free equality group. Standardized tables/covariance include
every declared parameter, including fixed original-unit markers and all
members of equality groups.

## Likelihood, uncertainty and fit

Raw-data ML uses centered observed covariance with divisor **n**. With a mean
structure the objective also includes the fitted mean discrepancy. Constants
are preserved in the full Gaussian log likelihood. Summary input must declare
`divisor="n"` or `"n-1"`; the latter is converted by `(n-1)/n`, and its supplied
matrix is saved exactly. Mean-structure summary fitting requires an observed
mean vector. Omitted summary means cannot recreate individual training scores.

Ordinary inference requires an interior positive definite disturbance matrix,
an identified full mean/covariance Jacobian and positive definite **observed**
information. The exact joint observed Hessian includes mean/path/loading/
variance/covariance cross blocks. The natural-unit covariance transforms that
whole inverse information; it does not concatenate separate coefficient SEs.
Reported z tests and intervals use an asymptotic normal reference.
Heywood, weak/deficient identification, singular information and failure to
meet the declared score tolerance refuse ordinary inference.

Every standardized loading/path, residual variance/covariance and intercept
uses fitted **total** node variances. Their full joint covariance differentiates
that transformation. A fixed original-unit marker has nonzero standardized
uncertainty when fitted variances vary. A root's standardized total variance
is identically one and has zero derivative and no fabricated Wald p-value.

The normal-ML model statistic is `n FML`; df is the number of observed covariance
moments plus observed means, when modeled, minus the free group count. The
independence baseline estimates observed variances and unrestricted observed
means; its covariance df is `p(p-1)/2`. Saved baseline matrices, means, statistic
and conventions accompany CFI/TLI/RMSEA/SRMR. CFI uses the noncentrality formula
with maximum of model/baseline excess chi-square and equals one when both
nonnegative excesses are zero. TLI retains a negative baseline excess whenever
its denominator is nonzero; zero model df or a zero denominator gives one,
matching lavaan's ordinary boundary rule. TLI is not clipped to [0,1].
RMSEA is `sqrt(max(chi2-df,0)/(n*df))` for positive df and zero for a saturated
model, following the normal-ML reference boundary. SRMR combines the lower-triangular
sample-SD standardized covariance residuals, including diagonals, with the
`p` sample-SD standardized mean residuals when means are modeled. Its denominator
is `p(p+1)/2+p` for a mean model and `p(p+1)/2` otherwise, matching lavaan's
Bentler convention. `SRMR_covariance` and mean residual RMS are also reported
separately. Zero model df makes the model
p-value unavailable. These are ordinary Gaussian indices, without robust/scaled
corrections or RMSEA noncentrality intervals.

`sem_effects` sums all directed acyclic paths between one source/target pair.
It returns direct, indirect and total effects with their **joint** delta
covariance. Standardized effects additionally differentiate fitted total
variances. Zero/fixed effects retain zero variance and no Wald p-value.

`latent_scores` returns Gaussian regression conditional latent means from saved
joint moments. It retains typed row indexes and marks explicitly omitted rows
missing. Its attrs report the full latent conditional covariance. Parameters
are plugged in; this conditional covariance does not include parameter-estimation
uncertainty and is not an observed-outcome forecast interval.

## Input, resource and state contract

Native CPU float64 Torch performs numerical work. There is no SciPy, NumPy,
lavaan or optimization-library runtime dependency. Inputs are resident pandas
DataFrames or mappings of bounded real numeric columns. Series mappings must
have identical typed indexes. Boolean/complex/infinite values, unsafe integer
rounding and generators refuse. `missing="raise"` is the default;
`missing="drop"` explicitly chooses complete cases and preserves original
source/index/dtype/position identity. This is not MAR FIML. Weights, ordinal/link
models, cluster/HAC/survey/robust covariance and cyclic graphs are unsupported
in these functions.

Resident bounds are n ≤ 10,000, observed columns ≤ 16, latent factors ≤ 4 and
free groups ≤ 120. Work and workspace estimates account for raw/centered rows,
state matrices, full standardized covariance, exact Hessian/Jacobians, two
deterministic starts, at most 30 starting-variance rescalings per start, bounded
backtracking and at most 20 observed-Hessian Newton steps per start. All fitting
and restore admission precedes data copying and numerical replay. Scores admit
their query buffers/index before copying. The work ceiling is 10 billion scalar
units, with bounded 1–500 iterations; the task-local workspace budget is enforced
independently. These are implementation/resource boundaries, not sample-size
theorems. Workspace estimates are named live buffers rather than process RSS.

`SEMState` uses schema `openecon.latent_sem.v1`. Its immutable payload saves the
ordered specification/constraints, source, options, solver parameters, both
starts, complete OIM, transformations, diagnostics and resource plan. Restore
reconstructs source moments and replays the statistical equations and all cached
numeric fields without rerunning the optimizer. A rehashed forged covariance,
likelihood, mean, sample mask or diagnostics still refuses. Dimensionful cache
comparisons use each covariance's own units. JSON admits at most 32 MiB; typed
index metadata admits at most 2 MiB. Ambient Torch dtype/device and RNG state
remain untouched.

## Evidence and reference conventions

Independent source tests use primitive NumPy matrices, an independently derived
natural-parameter score and finite-difference observed Hessian, SciPy fitting,
and separate effects/standardization/score formulas. They cover marker and
unit-disturbance scaling, equality constraints, mean/covariance cross blocks,
raw/summary equivalence, unavailable fit cases and dimensional/state/resource
adversaries. Primary reference descriptions are the
[lavaan SEM tutorial](https://lavaan.ugent.be/tutorial/sem.html),
[mean structures](https://lavaan.ugent.be/tutorial/means.html),
[summary covariance conventions](https://lavaan.ugent.be/tutorial/cov.html), and
[normal versus Wishart ML](https://lavaan.ugent.be/tutorial/est.html).
Reference output must explicitly select observed information; the usual default
expected information is a different uncertainty comparison.

See `examples/latent_sem.py` for a deterministic runnable model and portable
receipt. Source tests, independent R/lavaan reference, licensed Stata/Amos output,
frozen runtime and actual installed desktop Run/Quit/reopen remain distinct
acceptance records; none is inferred from the others.
