# Saved smoothing models and conditional-mean prediction

MARKET-240–247 implement eight resident CPU float64 Torch stages under MARKET-158.
Every public fit accepts `data`, `y`, a list `x`, and explicit `missing='raise'/'drop'`.
For FP/MFP, the first predictor is the positive nonlinear variable; remaining
predictors are fixed linear adjustments. No implicit positive shift is performed.
Use `oe.smoothing_predict(result, data=query)` after JSON roundtrip. New-data
outcomes are ignored and no transformations are refitted. `row` is the original
zero-based query position; explicitly dropped missing queries remain identifiable.

| API | Implemented domain | Inference and selection | New-data support |
| --- | --- | --- | --- |
| `bspline_regress` | Additive degree 1–3 Cox–de Boor B-splines, intercept plus first-basis omission | IID Gaussian OLS full covariance/t/df/p/CI, conditional on basis; fixed knots or training quantiles | Reject outside saved boundaries |
| `rcs_regress` | Natural cubic truncated-power basis, at least three strictly ordered knots including boundaries | IID Gaussian OLS full covariance/t/df/p/CI, conditional on basis | Exact linear tails outside outer knots |
| `gam_gaussian` | Centered additive cubic B-splines with second-difference penalties, one shared nonnegative penalty | Fixed penalty or explicit bounded training GCV grid; full conditional frequentist coefficient covariance in state, no coefficient tests; approximate estimator-mean intervals | Reject outside each saved boundary |
| `loess` | Direct one-dimensional degree 1/2 Gaussian tricube local polynomial | Fixed span or exact training leave-one-out grid; prediction only | Within training range; rank/support failures are explicit |
| `fp_regress` | One/two powers from −2, −1, −0.5, 0, 0.5, 1, 2, 3; repeated powers multiply by log(x/scale) | Explicit powers/scale; IID Gaussian OLS full covariance/t/CI | Positive x/scale, including extrapolation with that assumption |
| `mfp_regress` | One nonlinear FP candidate variable plus always-retained linear adjustments | All 8 FP1/36 FP2 fits plus null; approximate 4/3/2 numerator-df F inclusion/form/simplification tests, residual denominator df from best FP2; selected-model conditional t inference | Saved selected powers and scale, positive domain |
| `mars` | Numeric paired-hinge forward SSE search, interaction degree 1–3, backward GCV pruning | Bounded deterministic knot grid/min-span; predictive coefficients, no p/CI after adaptive selection | Saved piecewise-linear hinge products, explicit extrapolation |
| `npreg_mixed` | Gaussian numeric, Aitchison–Aitken unordered, Wang–van Ryzin ordered product kernel | Fixed bandwidth or exact training LOO candidate grid; mean only, no CI | Numeric ranges and exact declared typed category universes/order |

Exact fixed-basis OLS inference assumes independent homoskedastic Gaussian errors.
Training-quantile knots condition on the observed predictors. MFP coefficient
intervals condition on the selected model and **exclude selection uncertainty**;
the adaptive closed tests are approximate rather than ordinary exact nested F
tests. This stage does not implement general multivariable MFP cycling, GLM MFP,
or prove unconditional model-selection error control.

For GAM, write `G=X'X`, `A=G+lambda*P`, `S=X A^-1 X'`.
The saved covariance is `sigma2 * A^-1 G A^-T`, with
`sigma2=RSS/(n-2*trace(S)+trace(S'S))`. Training GCV is
`n*RSS/(n-trace(S))^2`. Mean intervals use a normal reference and this covariance;
they concern the penalized estimator expectation, excluding smoothing bias and
penalty selection uncertainty. They are not simultaneous bands or unconditional
coverage guarantees for the unknown regression function. Smooth centering and
the first-basis constraint are stored along with the entire penalty matrix.
Candidate solves are processed one at a time, not retained as an unbounded cache.

MARS GCV complexity is `K + gcv_penalty*(K-1)/2`, including the intercept. Candidate
knots have at least `min_span` active rows on both sides and are restricted to at
most `max_candidates` evenly spaced ordered distinct training values per predictor.
Repeated use of a predictor in one hinge product is excluded. Predictor/parent/knot
order resolves forward ties; backward SSE ties remove the lower original basis
index. Complete forward and backward paths, knot grids and selected factors are
saved. This is a bounded grid implementation, not a claim to reproduce every
vendor's search heuristics or inferential output.

Mixed kernels require `variable_types={'age':'c','group':'u','grade':'o'}` and a
`categories` dictionary for exactly the u/o columns. Unordered lambda is in
`[0,(K-1)/K]`: equality weight `1-lambda`, unequal weight `lambda/(K-1)`.
Ordered lambda is in `[0,1)`: equality `1-lambda`, unequal
`(1-lambda)*lambda^abs(rank difference)/2`. Category order is the supplied list;
typed JSON levels distinguish e.g. boolean `true`, integer `1` and string `"1"`.
All-numeric queries should use the existing `kernelreg`/`localreg` APIs.
Weights are normalized in log space; no local support/effective-sample failure is
silently replaced. LOO candidate failures and successful losses are recorded.
The required bandwidth in LOO mode is the declared reference configuration,
explicitly preserved separately from the selected candidate.

Each stage refuses frequency/analytic/probability weights, cluster/robust
covariance, device switches, Dataset execution, categorical model coding,
unknown options and underidentified geometry. Input rows/predictors, knot/category
counts, candidate counts, live tensor workspace and planned scalar work are gated
before model buffers are allocated. Workspace plans exclude caller memory,
Python objects, BLAS private allocations and allocator overhead; they are not an
RSS guarantee. Replay also checks workspace/work and SHA-256 state/spec matching.

Methods are based on [R splines](https://search.r-project.org/R/refmans/splines/html/ns.html),
[R direct LOESS](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/loess.html),
[mgcv frequentist penalized covariance](https://www.stat.ethz.ch/R-manual/R-devel/library/mgcv/html/magic.post.proc.html),
[the original FP authors](https://mfp.imbi.uni-freiburg.de/fp/),
[mfp2's closed-test contract](https://search.r-project.org/CRAN/refmans/mfp2/html/select_ra2.html),
[Friedman's MARS paper](https://www.stat.yale.edu/~lc436/08Spring665/Mars_Friedman_91.pdf),
and [statsmodels' documented mixed kernel family](https://www.statsmodels.org/v0.12.2/generated/statsmodels.nonparametric.kernel_regression.KernelReg.html).
Development-only independent SciPy bases and dense NumPy/statistical references
check coefficients, complete covariance, pointwise means, selection paths and
inference where available. Licensed Stata/SPSS/EViews comparison has not run;
global `stata_parity_validated` remains false. Non-Gaussian GAM, derivatives,
uncertainty bands and broader/vendor options remain in MARKET-158 and GitHub #44.
