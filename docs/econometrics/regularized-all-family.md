# Weighted and categorical scalar regularized prediction

`ridge`, `lasso`, `elasticnet` and scalar-response `pls` now accept declared
`categorical`, `weights` and `weight_type` through their existing public APIs.
These are resident CPU float64 prediction targets. The binomial/Poisson
implementations described in [regularized-glm.md](regularized-glm.md) remain
unchanged. The previous numeric, unweighted Gaussian/PLS execution routes,
including validated IID score plug-in tuning, remain available.

## Weighted Gaussian objective and categorical blocks

For positive retained raw weights, `a_i=w_i/sum(w)` is evaluated by first
dividing by the maximum weight. The explicit Gaussian objective is

\[
 \tfrac12\sum_i a_i(y_i-\bar y_a-z_i'\beta)^2
 +\lambda\sum_j q_j\{\rho|\beta_j|+(1-\rho)\beta_j^2/2\}.
\]

The intercept is unpenalized; with no intercept neither X nor y is centered.
Predictor centering and population RMS use only the current training partition.
The response is never standardized. Native Gaussian coordinate descent/linear
factorization is applied to `sqrt(n*a_i)*z_i` and `sqrt(n*a_i)*(y_i-ybar_a)`;
this is exactly the weighted objective above. Every path point retains the
objective, coefficient vector in standardized and original units, constant,
iterations, KKT residual and its convergence threshold. Failure raises.
The elastic-net normalization follows the objective convention in
[Friedman, Hastie and Tibshirani (2010)](https://www.jstatsoft.org/article/view/v033i01);
the explicit weight/category/scaling contracts and tests here determine this
implementation's precise units. No package-wide numerical equivalence is claimed.

Typed category identities, first-seen training levels and baseline treatment
coding are reused from the GLM design contract. `intercept=False` uses all
one-hot levels. Every fold learns its own categories and moments. An unseen
held-out or future category raises; it cannot silently become the reference
category or shrink the scored sample. Original predictor factors may be an
ordered list or a name mapping; the same literal factor multiplies both L1/L2
for every column in its encoded block. There is no factor normalization.
Named forced controls have factor zero and must be jointly identified in every
training partition. Automatic Gaussian lambda references residualize against
that block before computing the weighted score/factor reference.

## Weighted scalar PLS and forced controls

PLS1 uses the same weighted train-only schema. In whitened coordinates, an
identified forced block C is projected from the response and remaining slopes:
`r_y=y_w-C*gamma_y`, `R=X_w-C*Gamma_x`. NIPALS deflates these residuals,
using direction `R.T*r_y`, normalized predictor weights, score `t=R*w`,
loadings `p=R.T*t/(t.T*t)` and response loading `q=r_y.T*t/(t.T*t)`.
For component matrices W, P and q, the remaining slope is
`b=W*solve(P.T*W,q)` and the forced slope is `gamma_y-Gamma_x*b`.
All component loadings, control projections, original units and constants are
saved. Degenerate response directions, score rank or component rotations raise.
PLS has no feature-penalty objective; `penalty_factors` are explicitly refused.
This route is scalar weighted partial PLS1, with no multivariate-response or
GLM-PLS claim. Without forced controls it is ordinary weighted PLS1.

## Weights, tuning, state and inference

Frequency counts are nonnegative exactly representable integers no larger than
2^53; zero-weight rows are excluded by the common sample contract. For fixed
fits, weights reproduce literal independent row replication without expanding
the engine's rows. Analytic/sampling weights define positive normalized empirical
prediction loss and remain invariant to multiplying all weights by a positive
constant. They do not imply a survey sampling model or design covariance.
Cross-validation assigns each original physical unit to one fold, keeping its
frequency replicas together. It pools held-out weighted squared errors on the
complete retained sample. Thus CV is not equivalent to assigning replicated
copies independently to different folds.

Fixed penalties/components and train-fold-only CV are supported. Explicit
Gaussian grids are absolute; automatic grids compare common geometric fractions
of each training fold's weighted lambda reference. The selected fraction is
refitted on the complete supplied training sample. Gaussian ties prefer larger
lambda; PLS ties prefer fewer components. PLS candidates invalid in a fold are
recorded and cannot win; no failed fold is omitted. If all candidates fail, the
fit raises. The existing local fold generator leaves global RNG state untouched.
Weighted/categorical Gaussian score plug-in calibration is not validated and
raises, preserving the separate existing numeric IID plug-in route.

`extra.regularized_extended_state` stores the complete spec/sample binding,
raw/normalized weights, typed design, path points, fold positions and designs,
candidate losses/failures, chosen index, and work plan. JSON restoration does not
refit. `regularized_predict` and `regularized_table` validate integrity plus
semantic spec/factor/scaling/physical-fold/pooled-loss bindings. A checksum is
not authentication. PLS candidates must match the requested explicit path or
the complete feasible range bounded by `max_components`. Each automatic Gaussian
fold grid must preserve its training reference multiplied by the common fractions.
No-intercept states require zero response center and constant in every path and
fold. Table validation remains on CPU even when the caller uses another default
tensor device. Empirical losses preserve finite residual subtraction and apply
weights before squaring; unrepresentable losses raise explicitly.
Query indices, including duplicated labels, are retained.
Missing estimation inputs raise by default or are explicitly dropped; prediction
inputs must be complete. Classical selected coefficient SE/df/p/CI, predictive
intervals and survey inference remain unavailable, with empty coefficient and
covariance arrays.

The extended fit domain is 4–5,000 retained rows, 1–16 original predictors,
16 levels per category, and at most 64 encoded slopes; Gaussian paths contain
at most 200 points and PLS paths at most 100 unique feasible component counts.
Queries admit at most 100,000 rows. Work and workspace admission cover full
fitting/partialling/CV and saved-state validation/query allocation. Design and
automatic-reference work are admitted before control factorization or grid
calibration. Saved JSON is limited to 32 MiB, nesting to 16 levels, and each
metadata string to 64 KiB of JSON-encoded content. Exact structural fields and
rectangular shapes, then actual canonical UTF-8 bytes, are checked before digest
construction or numeric tensor allocation. Serialization/digest buffers reserve
eight times the encoded byte count; work includes the traversal and hashing passes.
No path,
fold, sample or component is truncated. Dataset/GPU, panel/time/cluster,
formulas and unvalidated inference options are refused by the common registry.
The new weighted/category paths also compare training preprocessing with saved
replay before fitting. Large nearly constant predictors whose float64 centered
coordinates cannot be replayed consistently raise `numerical_failure` and require
rescaling; the existing GLM design helper is unchanged.

`tests/test_regularized_extended.py` uses independent support/sign enumeration,
weighted covariance/deflation, physical replication, held-out perturbation,
negative domains and complete state restoration. The runnable
[example](../examples/regularized_all_family.py) exercises all four APIs.
These establish source-level method/option evidence. Frozen, installed-app,
release and licensed vendor evidence require separate dated receipts; no such
acceptance is inferred from source tests.
