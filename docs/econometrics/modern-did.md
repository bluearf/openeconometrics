# Heterogeneous DiD and synthetic controls

These native code-first procedures require complete balanced consecutive
integer/datetime calendars, binary absorbing adoption, no covariates or weights,
and no always-treated units. Missing input raises by default; dropping rows
must still leave an admissible balanced panel. Resident ModelFrame admission
and declared work budgets precede designs; Dataset collection is unsupported.

`oe.bacon(data, y, treatment, panel, time)` reports Goodman–Bacon's covariate-free
2x2 TWFE decomposition. It records timing comparisons, exact sample windows,
weights and contributions, checks reconstruction against full TWFE, and saves
cohort/time causal-effect weights. Late-vs-early comparisons use already-treated
controls; nonnegative comparison weights do not guarantee nonnegative causal
effect weights. This diagnostic does not repair heterogeneous-effects bias.

`oe.heterodid(..., model="bjs")` estimates untreated unit/time effects and
imputes counterfactuals only from untreated observations. Relative-period targets
average observed eligible treated units. The full influence map includes
first-stage uncertainty; unit-cluster variance uses cohort/time-centered treated
errors and G/(G-1). At least two units per treated cohort are required for that
centering. This is the conservative BJS variance domain with unrestricted
cohort/time effects. Only post-treatment targets are supported; negative event
periods require a separately specified pretrend design and are rejected.

`model="sunab"` fits saturated cohort/event interactions with unit/time effects,
using never-treated controls and the -1 reference for every cohort. All observed
nonreference interaction cells enter the fit, including events outside the
requested reporting range; there is no implicit tail binning. Aggregation uses
the observed eligible cohort shares, conditioning inference on their counts.
Full cohort/event covariance and the aggregation map are saved. Unit-cluster
CR1 uses G/(G-1)*(N-1)/(N-K) for the full dummy design. This conditional-design
target does not include uncertainty in a separately estimated population cohort
distribution. Both models assume parallel untreated trends and no anticipation.

`oe.synthcontrol(..., model="sc")` matches treated mean pre-outcome history
with nonnegative donor weights summing to one, without an intercept. No outer
predictor-importance optimization is used. `ridge=0` is default; declared positive
ridge changes that objective explicitly. A singular active QP raises instead of
adding hidden regularization. `model="sdid"` profiles intercepts in separate
unit/time weight problems. Default zeta_omega=(N1*T1)^(1/4)*sd(control first
differences), zeta_lambda=1e-6*that sd; penalties are T0*zeta_omega^2 and
N0*zeta_lambda^2. ATT is the unit/time doubly weighted outcome contrast.

Synthetic models require one shared adoption date, at least two pre-periods
and more donors than treated units. A feasible active-set QP reports objective,
simplex feasibility and dual gap; a clipped unconstrained fit is not used.
`max_work` bounds actual Gram/KKT operations across the fit and all placebos.

Uncertainty is explicitly **original-control placebo inference**. For one
treated unit, every original donor is treated in turn; for multiple treated
units, `placebo_reps` seeded same-size assignments are drawn with a local RNG.
Original treated units never enter placebo donor pools. Every weight problem
is refitted; SDID tuning parameters remain fixed at the original values, as
in the paper's Algorithm 4. V is centered placebo dispersion with divisor B,
and a plus-one absolute-effect tail probability is reported. Normal coefficient
intervals are an approximation conditional on placebo exchangeability and
comparable noise across treated/donor units, not unconditional causal guarantees.
Placebos are not filtered using observed fit quality.

Every fitted result saves sample keys/calendars, targets, full covariance,
weight/interactions state and placebo population/draws. No assumption is verified
merely by observing good pre-fit or an insignificant pretrend diagnostic.

Validation: independent full dummy SVD and statsmodels cluster matrices check
heterogeneous event estimates and off-diagonal V; analytical 2x2 mean contrasts
and weights reconstruct TWFE. Published California Proposition 99 panel data
are checked against independent SciPy constrained-QP fits, all weights and the
full refitted placebo distribution. These establish stated domains, not blanket
Stata parity or statistical identification in arbitrary applications.

Sources: [Goodman–Bacon (2021)](https://www.nber.org/papers/w25018),
[Borusyak, Jaravel and Spiess (2024)](https://doi.org/10.1093/restud/rdae007),
[Sun and Abraham (2021)](https://github.com/lsun20/EventStudyInteract),
[Arkhangelsky et al. (2021)](https://arxiv.org/abs/1812.09970),
and the authors' [synthdid reference implementation/data](https://github.com/synth-inference/synthdid).
