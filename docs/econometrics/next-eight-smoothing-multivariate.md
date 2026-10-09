# Smoothing and multivariate option audit — 2026-10-07

This option matrix compares the original GitHub #44 and #73 requests with
the initial audit at `45d6b64`, reconciled with merged EFA PR #139 and smoothing
postestimation PR #143 through `13bda35`, plus this delivery's separate saved
Gaussian local-regression derivative and RM contrast helpers. It records
implemented numerical domains,
remaining stages and retained evidence separately. A method-family heading is
not evidence that every Stata/SPSS option exists.

## Smoothing (#44)

| Requested domain | Public implementation and saved state | Inference and remaining domain |
| --- | --- | --- |
| Continuous local constant / local linear | `kernelreg`, `localreg`: multivariate product kernels; fixed, normal-reference pilot or explicit training leave-one-out bandwidth grid; training observations, selected bandwidth, candidate losses, effective support and bounds persist. `regularized_predict` replays that state. | Conditional mean prediction only; no pointwise or derivative SE/CI. The pilot is not an optimal regression-bandwidth claim. Compact support/rank failures refuse. |
| Continuous derivative / empirical average | `local_derivatives` and `local_average_derivatives`: saved Gaussian local-constant/local-linear estimator derivatives in original predictor units. Bandwidth and training state remain fixed. Original query row indices are preserved. | No derivative SE/CI, categorical effect, smoothing-bias correction, causal interpretation or population sampling inference. Queries must be inside observed bounds even if mean extrapolation was allowed. |
| Categorical kernel covariates | `npreg_mixed`: numeric Gaussian, unordered Aitchison–Aitken and ordered Wang–van Ryzin product kernels; declared typed levels/order; fixed or exact training LOO grid. `kernel_derivative` supplies first/second continuous normalized-kernel partials; `smoothing_contrast` supplies explicit categorical profile changes without refit. | Numeric ranges, saved category universes and effective sample gates apply. No numeric categorical derivative or intervals. Mixed local-linear geometry, bias/selection uncertainty and broader inference remain stages. Unseen levels refuse. |
| B-spline and restricted cubic spline | `bspline_regress`, `rcs_regress`: fixed knots or training predictor quantiles; complete basis and coefficients/covariance persist. `spline_derivative` supplies first/second pure partials in original units with full saved covariance. | Fixed-basis iid Gaussian conditional t SE/CI for means, derivatives and supported fixed-profile functionals; knot/model selection uncertainty excluded. B-splines reject extrapolation and discontinuous derivative knots; saved endpoints use interior one-sided limits. RCS has linear tails and zero tail second derivatives. |
| Penalized additive model | `gam_gaussian`: centered additive cubic B-splines; fixed penalty or explicit training GCV path; full penalty, centering, coefficients and frequentist covariance persist. `gam_derivative` supplies first/second saved additive-basis partials. | Approximate normal mean/derivative/profile intervals target the penalized-estimator expectation and exclude smoothing bias and penalty-selection uncertainty; no p-values or Bayesian claim. Non-Gaussian families, heterogeneous penalties and broader simultaneous/unconditional laws remain stages. |
| LOESS | `loess`: direct one-dimensional degree 1/2 tricube local polynomial; fixed span or exact training LOO path. `loess_derivative` supplies the saved-span local Taylor slope/curvature `r! beta_r / radius^r`, within the saved range and local rank gate. | The Taylor target does not differentiate the moving-neighborhood prediction algorithm. Second order requires degree two. No intervals. Robust reweighting, multivariate LOESS and adaptive/bias/selection uncertainty remain stages. |
| Fractional polynomials / MFP | `fp_regress`, `mfp_regress`: one positive nonlinear predictor, fixed linear adjustments; eight FP1 and 36 FP2 candidates; powers/scale/selection persist. `fp_derivative` supplies first/second partials with scale chain rule, repeated logarithms and retained/excluded adjustments. | Positive FP query domain; full saved iid Gaussian conditional t covariance for fixed/selected basis targets. MFP closed F selection is approximate; subsequent inference excludes selection uncertainty. General multivariable cycling and GLM MFP remain stages. |
| MARS | `mars`: deterministic bounded paired-hinge forward SSE search, degree 1–3 interactions, backward GCV pruning; knots, complete paths and selected hinges persist. `mars_derivative` supplies first retained-hinge interaction partials. | Active exact knot ties refuse; a zero other-factor multiplier gives zero partial; piecewise-linear extrapolation persists. No intervals or adaptive coefficient inference. Search heuristics are explicit; non-Gaussian families and broader adaptive uncertainty remain stages. |
| Fixed-query margins / paired profile contrasts | `smoothing_margins` averages response or first partial; `smoothing_contrast` forms response differences by original row position, optionally averaged. Both replay all eight smoothing fit families with synchronized missing exclusion and optional nonnegative original-query averaging weights. | Spline/FP/MFP and Gaussian GAM can request their respective conditional intervals. Aggregate the coefficient-order functional before `L V L'`, retaining cross-profile covariance; stored `L` and `V` factorize joint row covariance without quadratic row allocation. LOESS/MARS/mixed-kernel targets remain descriptive. Weights define a fixed query target, not a weighted training fit or population sampling law. |

The eight smoothing stages accept resident tables with explicit missing
raise/drop policy, preserve original query positions and refuse weights,
Dataset collection, generic categorical model coding, device switches and unsupported
covariances. `kernelreg`/`localreg` similarly expose bounded resident numeric
domains; Dataset rejection is explicit. All numerical fitting is native CPU
float64 Torch. [Smoothing method contracts](smoothing-stages.md) and
[saved postestimation contracts](smoothing-postestimation.md), together with
[regularized prediction contracts](regularized.md), give the formulas and resource
plans. Workspace plans cover named live tensors, not total process RSS.

The historical reports mark splines/MARS as future work and GAM as absent;
these statuses predate merged PR #113. Their current presence is verified
against source, rather than inferred from those old statuses. The research
catalog also identifies [IBM STATS GAM](https://github.com/IBMPredictiveAnalytics/STATS_GAM)
and [STATS EARTH](https://github.com/IBMPredictiveAnalytics/STATS_EARTH) as extension
routes. OpenEcon's Gaussian GAM/MARS implementations do not execute those R
extensions. [Stata's nonparametric feature list](https://www.stata.com/features/nonparametric-methods/)
includes additional bandwidth/series choices and derivative/margins uncertainty
domains beyond these conditional targets; those are comparison targets, not
measured numerical parity. The six derivative helpers plus margins and contrasts
in PR #143 use `smoothing_state`, separately from the `smoother_state` route of
`local_derivatives` / `local_average_derivatives`; the latter remain fixed-Gaussian
continuous kernel/local-linear estimator derivatives without SE/CI.

GitHub #134 remains substantive for adaptive/local/kernel sampling uncertainty,
joint/resampling laws beyond the saved covariance factorization, smoothing-bias
and tuning/model-selection uncertainty, random query-population inference,
non-Gaussian GAM, broader penalties, multivariate/robust LOESS, general
multivariable/GLM MFP and non-Gaussian MARS. Delivered conditional fixed-basis pointwise
mean/derivative/profile intervals, explicit categorical profile changes and
fixed-query averaging must not be relisted as missing work.

## Multivariate / repeated measures (#73)

| Option-level target | Implemented input/sample/weight contract | Remaining domain |
| --- | --- | --- |
| PCA raw observations | Resident or replayable Dataset; listwise drop/raise; covariance or correlation; unweighted or nonnegative integer frequency weights. Zero weights excluded; physical rows and frequency total are distinct; divisor is `sum(w)-1`. | Analytic/probability weights and pairwise covariance are unsupported. |
| PCA summary matrix | `pca_matrix` requires explicit covariance/correlation convention, sample count, symmetric PSD full matrix and positive diagonal. Singular PSD PCA is allowed. Actual means/scales are required when corresponding saved scores need them. | No inferred training means, synthetic pseudo-observations or silent PSD repair. |
| EFA raw/summary matrix | `factor` / `factor_matrix`: PF, IPF, PCF, ML, minres, alpha and explicit SAS/Guttman `image_covariance`; declared factors, convergence, moments, loading/score state. Resident and supported Dataset reductions are explicit. | GLS and SPSS Kaiser generalized image extraction remain unsupported. Minres discrepancy is not an ML test; image EFA scores are not exact principal-component scores of image predictions. |
| EFA frequency weights | `factor(..., weights="w", weight_type="fweight")`: integer nonnegative literal replication counts; anchored weighted moments feed all seven extractions, rotations and saved score coefficients. `n=sum(w)`, divisor `n-1`; physical/missing/zero rows remain separate. | Counts and total must be exactly representable (`<=2**53`). Analytic/probability weights are unsupported; Gaussian tests assume literal independent row replication, not a survey design. |
| Alpha / image covariance extraction | Alpha iterates communalities with explicit `m<p`, positive communalities/retained roots and convergence; SMCs, spectrum and iteration residual persist. Image covariance uses variable-wise linear prediction covariance with PD correlation, explicit `m<p` and positive retained image roots; full prediction/covariance state persists. | Alpha is not Cronbach alpha. Heywood cases and nonpositive generalizability notes are exposed. SAS/Guttman image covariance is distinct from the SPSS Kaiser generalized image target. |
| Score methods | Regression, Bartlett and Anderson–Rubin; AR score covariance is identity under the sample correlation matrix, with orthogonal factors/positive uniqueness/full score rank. Full weights persist. | Oblique AR scores refuse; summary scoring without necessary real moments refuses. |
| Rotation | Existing varimax/quartimax/equamax, promax, oblimin; complete orthogonal target; orthogonal/oblique geomin; orthogonal/oblique Crawford–Ferguson; identified partial orthogonal/oblique targets. Transformations and factor correlations persist. | CF/partial-target admit 2–16 full-rank factors with bounded iterations/tolerance. Partial masks need a nonzero specified anchor per axis and full masked tangent rank. Near-singular transformations refuse; iterative routines certify a local stationary solution, not a minimum/global/multistart optimum. |
| Loading uncertainty | `factor_bootstrap`: fixed unrotated one-factor principal-factor estimator functional; IID complete raw-row resampling, refitted moments/SMCs/eigenvectors and fixed sign anchor. Every planned loading/uniqueness vector, joint covariance, SE, bias and marginal percentile CI persist. | Resident CPU float64, 3–16 variables, <=10000 rows, 19–1999 replicates and explicit work/tail bounds. Every replicate must pass. No selected/multifactor/rotated loading inference, weighted/dependent/Dataset/summary bootstrap, p/df or familywise/exact coverage. |
| Discriminant | Direct LDA/QDA, saved scoring/prediction, supported covariance/rank and prior conventions in the method document. | Stepwise Wilks/Mahalanobis/MAXMINF/MINRESID/RAO selection and extra weighting domains remain unsupported. |
| Correspondence | Simple active two-variable correspondence analysis, optional count weights; `ca_project` reorders labelled nonnegative supplementary profiles and uses saved active standard coordinates without refit. | Multiple/joint correspondence analysis and supplementary refitting remain unsupported. Zero-total/unseen profiles refuse; unidentified near-zero axes have undefined coordinates. |
| Repeated measures | `rm_anova`: complete subject×within-cell long design; multiple within factors and their interactions; optional between factors/all interactions, unequal between-group sizes; Type III SSCP; Mauchly, GG, HF and lower-bound correction conventions explicit. Resident and replayable Dataset routes. | No multivariate repeated-measures approach, incomplete-cell mixed-model substitution, covariance weights or within-cell replication. Dropping a missing row does not make an incomplete design valid. |
| Saved scalar repeated-measures contrast | `rm_contrast`: one original-cell weight vector × one between-design weight vector. Original-cell coefficients, full between-design bread, residual cell covariance, typed cell order and exact coding persist from resident or Dataset fit; no new estimation. | Exact scalar t inference assumes independent Gaussian subjects and one common unrestricted cell covariance. No sphericity assumption. Joint L/M hypotheses, heterogeneous cell covariances and weighted/incomplete designs remain unsupported. |
| MANOVA | `manova`: native Type III SSCP with Pillai/Wilks/Hotelling/Roy and declared F approximations; resident and replayable Dataset reductions. | User-supplied general L/M hypothesis contrasts and additional weighting domains remain stages. |

The main factor option path is descriptive. The separate fixed one-factor PF
bootstrap supplies joint loading/uniqueness covariance, SE and marginal percentile
CI under IID complete-case sampling, finite fourth moments, fixed dimension,
nonsingular interior correlation, a separated leading reduced-matrix root,
nonzero fixed sign anchor and interior positive uniqueness. It targets that
estimator functional, not latent ML loadings or a selected/rotated factor model.
It supplies no p values, inferential df, familywise or exact coverage. Existing Bartlett and ML model tests
remain conditional on their stated sample/model assumptions. Summary covariance
standardization and weighted sample semantics must not be replaced by the
unweighted physical-row count. Resource caps, missing behavior, prediction and
Dataset support are specified in [multivariate.md](multivariate.md) and
[stats.md](stats.md).

Before claiming completion of the broader option surface, focused work must
remain visible for (1) SPSS Kaiser generalized image/exact image-component scores
and GLS extraction, (2) multifactor/rotated/selected loading uncertainty and
weighted/dependent/Dataset bootstrap beyond fixed one-factor IID PF,
(3) broader valid weighted multivariate and stepwise discriminant procedures,
and (4) general joint MANOVA/repeated-measures contrasts/designs. Alpha, integer
frequency-weight EFA, CF and identified partial orthogonal/oblique targets are
implemented and must not be relisted as missing work. These are substantive implementation
stages, not merely documentation tasks. GitHub #28 retains licensed reference
validation as separate work.

## Evidence identity and reproduction

Before PR #143, all seven `econometrics/smoothing` files were checked
byte-for-byte against `1f1848ab035ea90194424e18838e918390b4290b` and matched.
The retained original smoothing receipts (internal evidence excluded from this public snapshot)
remain evidence for that historical pin. PR #143 changes the family manifest,
local polynomial helper and replay path, and adds `postestimation.py`; the old
whole-family receipt cannot validate those changed files. All eight current
smoothing modules match the accepted frozen/installed scientific pin
`d6e183dd4d8c5d41b0537e265eb88f619c24ad0e` byte-for-byte. The
postestimation receipts (internal evidence excluded from this public snapshot)
retain source, frozen and separate installed native Run/restart evidence for
the eight new helpers at that pin, with scoped unchanged-source integrations
recorded separately. Every retained verification-file SHA was checked. That
acceptance does not cover the separate regularized-family derivative/RM helpers,
every current main API or a new public release.

At the initial audit, all sixteen multivariate files matched
`a7d45503d0e1d277a91b46faff828b8e71d1b61a`. PR #139 subsequently changed the
factor/weighting path and added extraction/rotation/uncertainty modules; the
older multivariate receipts (internal evidence excluded from this public snapshot)
retain their original pin and cannot validate that changed source. All nineteen
multivariate files in reconciled main were separately checked against the newer
accepted package pin `100db5c85e1aa0239411ff9d13a037d117b642c4` and match. The
EFA extension receipts (internal evidence excluded from this public snapshot)
record source, frozen, actual native Run/reopen and complete payload equality
for that pin. These pinned app/runtime receipts do not cover the separate
regularized-family derivative or RM contrast helpers, every current main-branch
API or a new public desktop
release. The shared capability catalog has also subsequently expanded.

Fresh source checks cover `tests/test_smoothing_eight.py`,
`tests/test_multivariate_options.py`, `tests/test_regularized_oracles.py`,
`tests/test_regularized_contracts.py` and `tests/test_econ_stats_anova.py`.
Independent development references include SciPy spline spaces, dense NumPy
full covariance/rotation/score/projection formulas, statistical tails,
factor gradients and nonparametric weighted least squares. They are test-only.
PR #143's separate postestimation checks additionally cover SciPy B-spline
derivatives and an independently fitted natural cubic space; all 36 FP2 power
pairs/orders/scales; centered GAM full frequentist covariance; unscaled LOESS
Taylor WLS; mixed-kernel/MARS off-knot derivatives; fixed weighted margins and
paired cross-profile covariance, missing alignment and saved-state replay.
Its runnable example is
[smoothing_postestimation.py](../examples/smoothing_postestimation.py).
The merged EFA extension has separately retained independent literal count
replication, published SAS alpha calculations, variable-wise NumPy image
regressions, manifold gradients/tangent identification and full bootstrap-vector
references. Its executable example is
[multivariate_extension_acceptance.py](../examples/multivariate_extension_acceptance.py).
This delivery's separate saved Gaussian local derivative checks use dense NumPy
complex-step derivatives
of independently written Gaussian estimators, linear reproduction at boundaries,
moving-weight local-linear corrections, support/state/resource refusals and CPU
default-device isolation. New RM contrast checks use dense NumPy original-cell
OLS, full bread/covariance, independent SciPy Student tails/intervals, crossed
within factors, unequal between-group sample sizes, resident/Dataset parity,
JSON restoration and degenerate/malformed-state refusal.
Executable saved-state examples are
[smoothing_stages.py](../examples/smoothing_stages.py) and
[multivariate_options.py](../examples/multivariate_options.py). New target examples
with complete file readback are in
[smoothing_multivariate_targets.py](../examples/smoothing_multivariate_targets.py).

Source correctness, numerical references, JSON replay, historical frozen/native
receipts, a new wheel run and licensed vendor comparison are separate evidence
layers. No licensed Stata/SPSS/EViews run or blanket parity flag is asserted.

## Added target formulas and saved-state limits

For the separate regularized-family Gaussian local constant means, with
normalized saved weights `w_i(q)`,
`s_ij=(x_ij-q_j)/h_j²` and fitted mean `m(q)`,
`∂m/∂q_j = Σ_i w_i(y_i-m)(s_ij-Σ_l w_l s_lj)`.
For local linear fits, the exact derivative of the moving estimator adds the
residual × changing-weight correction to the fitted local slope. That correction
is evaluated through the local SVD influence weights, without squaring the
condition number by forming a normal-equation inverse. This differentiates the
saved conditional-mean estimator; it does not differentiate bandwidth selection
or estimate an unconditional population derivative. The derivative route admits
1–16 predictors, at most 100,000 saved training rows and 8,192 query rows, with
explicit pre-allocation work and workspace plans. Each query is processed alone.

PR #143's eight smoothing postestimation helpers instead replay `smoothing_state`.
For coefficient-linear targets, derivative or paired/averaged basis rows form
`L`, with estimate `L beta` and covariance `L V L'`; the full off-diagonal saved
coefficient covariance is retained. Conditional spline/FP/MFP t inference and
Gaussian GAM estimator-expectation normal intervals follow the distinct laws
above. LOESS remains a local Taylor derivative, and MARS/mixed-kernel derivatives
have no intervals. Queries are resident, 1–100,000 original rows, with at most
1,000 saved coefficient coordinates and 100,000 saved training rows; positive
work budgets and workspace plans cover paired selection, bases, training buffers
and the cubic covariance PSD audit before numerical allocations. Their accepted
default work budget is 2,000,000,000 scalar work units, not a guarantee that every
combination at the dimensional caps is admissible.

For a saved repeated-measures subject model `Y = X B + U`, the full original-cell
coefficient matrix `B`, between-design bread `(X'X)^-1` and residual covariance
`Σhat=E/(N-rank(X))` are saved. A scalar target with ordered vectors `l` and `c`
has estimate `l' B c` and variance
`[l'(X'X)^-1 l] [c'Σhat c]`. Its t statistic uses the common subject residual df.
This allows arbitrary scalar cell contrasts without assuming sphericity. Default
`l` selects the intercept: a sum-to-zero full-factorial design's equal-group
marginal mean, which differs from an observation-weighted pooled mean when group
sizes differ. Cell contrasts are not silently rescaled or forced to sum to zero.
Saved contrast geometry is limited to 128 original within cells and 128
between-design parameters; larger existing `rm_anova` fits retain their tables
and explicitly report that this additional saved contrast path is unavailable.
Old RM summaries lacking sufficient state must be refitted; summary tables alone
cannot reconstruct missing covariance. The state digest detects accidental
payload changes and no numerical fit is run on restoration.
