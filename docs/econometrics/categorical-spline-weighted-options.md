# Declared splines and weighted categorical options

MARKET-691–698 extend the accepted categorical and integer-frequency methods.
They are separate resident CPU float64 procedures, with no SciPy or vendor
delegation in the runtime. The earlier categorical/frequency children remain
complete independently. MARKET-171 remains open for broader spline, weight,
sampling, inference and licensed-vendor domains.

| Child | New fitting domain |
| --- | --- |
| MARKET-691 | Orthogonal varimax of count-weighted CATPCA geometry |
| MARKET-692 | Oblique promax of count-weighted CATPCA geometry |
| MARKET-693 | Nominal-predictor numeric-response replicated-population CATREG bootstrap |
| MARKET-694 | Ordered-predictor numeric-response replicated-population CATREG bootstrap |
| MARKET-695 | Nonmonotone declared degree-one predictor splines in CATREG |
| MARKET-696 | Monotone declared degree-one predictor splines with signed effects in CATREG |
| MARKET-697 | Nonmonotone declared degree-one single-vector CATPCA |
| MARKET-698 | Monotone declared degree-one single-vector CATPCA |

## Fixed spline support

`catreg_spline` and `catreg_mspline` accept a numeric response, 1–6 numeric
predictors and `knots={name: [lower, ..., upper]}`. Each variable has 2–8
strictly ordered finite knots including both support endpoints. The total
design has at most 32 segment columns; required work and buffer budgets can
refuse smaller inputs. Complete rows must exceed its dimension. Each ramp is
`clamp((x-k_j)/(k_{j+1}-k_j),0,1)`, producing a continuous piecewise-linear
degree-one transformation. Declared knots stay fixed throughout fitting.

The nonmonotone procedure solves the joint full-rank additive least-squares
design. The monotone procedure solves every one of the `2**p` signed cones
with bounded native Lawson–Hanson nonnegative least squares, accepts only KKT
solutions, and selects the minimum fixed-knot loss. Its maps are increasing;
regression coefficients retain positive or negative effect directions.
Columns and response are normalized for scale-independent KKT checks. Full-row
QR is computed once per cone; at most `40*D` active-set solves use the compact
`D`-row design, where `D` is the total segment dimension. Exhausting that bound
refuses the fit rather than returning an unverified constrained solution.
Fitted and saved-state acceptance checks residuals in orthonormal coordinates
and scales both active residual and inactive cone dual tolerances by the
smallest normalized singular value. Small normal-equation gradients alone
cannot certify a nearly collinear
fit. Numerically unresolved solutions fail explicitly.
Both publish centered unit-variance predictor maps, signed effects, original
response fitted values, residuals, SSE and R². Zero-variance effects cannot
identify a normalized transformation and fail explicitly. Rank-deficient or
ill-conditioned segment designs fail; no ridge fallback is inserted.

`catreg_spline_predict` validates saved raw training inputs, basis geometry,
all cone solutions and losses, selection, maps and complete display metadata.
It evaluates the saved continuous functions on new inputs, without fitting
them again. All queries must lie within the original closed knot spans.
Missing rows can be refused or dropped with original physical row positions.
Duplicate caller indexes never replace those positions.

Spline CATPCA uses the same declared spaces for one scalar transformation per
variable. It preserves centered unit variance and leading score/loadings
geometry, with bounded local ALS starts. Monotone maps constrain the entire
continuous interval, rather than checking the observed points alone. Local
convergence and stationary block geometry do not certify a global CATPCA
optimum. Saved supplementary projection uses the retained knot maps.

Spline procedures are unweighted. They admit at most 3000 physical training
rows, 128 MiB declared workspace and 300 million declared work units. No
automatic discretization, knot placement, extrapolation, higher polynomial
degree or adaptive coefficient Wald inference is supplied. Resource plans
cover admitted owned buffers and bounded state; they do not measure total
process RSS or caller-owned inputs.

## Count-weighted geometry and resampling

Weighted varimax/promax consume a complete `catpca_fweight` result. Rotation
acts on its loading space. Person covariance and category centroids use
literal count weights, with `X'WX/F=I`. Scalar category maps and actual
original memberships remain retained, including categories with pooled map
values. Varimax is orthogonal. Promax retains a nonsingular transformation,
pattern, structure and unit-diagonal component correlation matrix; coherent
inverse-transpose score transport preserves reconstruction. Their limits and
finite powered-target interpretation follow the earlier rotation contract.

`catreg_nominal_fweight_bootstrap` and `catreg_ordinal_fweight_bootstrap`
target numeric-response fitted means at explicitly fixed queries. Their
sampling units are the literal replicated observations: for retained counts
`f` with total `F`, each draw has `C* ~ Multinomial(F, f/F)`. Native private
seeded conditional binomial draws produce the entire physical-row count
vector without allocating `F` observations. Compressed physical rows are
never treated as equiprobable sampling units.

Every draw fully refits the maps and declared starts. A lost category,
insufficient positive physical support, rank failure or nonconvergence is a
failed draw. No replacement or surviving-only uncertainty is computed.
`failure='raise'` stops with the failure record; `failure='record'` retains
all declared draws and withholds covariance and intervals if any failed.
When all draws succeed, the procedure reports the full fixed-query empirical
covariance and percentile intervals. These are finite Monte Carlo estimates,
not coverage certification, ordinary adaptive coefficient Wald inference or
class probabilities.

The complete result retains baseline and every successful fit, count draws,
failure records, queries and predictions. Specialized restoration checks
their numerical consistency without running the optimizer. Counts, selected
source sample, missingness and all explicit work/serialization bounds remain
part of the result. Bootstrap is restricted to numeric responses and literal
integer frequencies, with at most 64 queries and 199 draws.

## Method sources and validation boundaries

The optimal-scaling formulation and spline predictor blocks follow
[Meulman and van der Kooij's ROS Regression paper](https://arxiv.org/abs/1611.05433).
IBM explicitly documents [degree-one spline CATREG examples](https://www.ibm.com/docs/en/spss-statistics/30.0.0?topic=catreg-examples-command)
and [single-vector spline CATPCA scaling](https://www.ibm.com/docs/en/spss-statistics/30.0.0?topic=command-level-keyword-catpca).
Caller-declared endpoints and knots are this API's bounded input convention;
they do not reproduce SPSS automatic discretization or knot selection.

Independent tests compare full additive fits with NumPy hat-basis least
squares, signed monotone cones with SciPy constrained solutions, weighted
rotation with independently reconstructed geometry and bootstrap with the
actual count law and separate full refits. Complete summary JSON/LaTeX,
saved numerical projections and resealed inconsistent state are checked.
Source tests, frozen runtime and installed native execution/export/restart
are distinct acceptance layers. No public release, notarization, CUDA,
streaming, licensed-vendor execution or blanket parity is implied.
