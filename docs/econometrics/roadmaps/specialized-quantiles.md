# Instrumental and time-series quantile stages

Planning: MARKET-186. Implementation tracking remains [GitHub #74](https://github.com/bluearf/openecon/issues/74).
Baseline: `40add89`. Reuse current qreg interior-point/check-loss and density
kernels, joint sqreg bootstrap, MM-QR, ARDL lag metadata and VAR date conventions.
None provides an IVQR identification or QARDL cointegration contract by itself.

## Q1 — IVQR ([MARKET-213](https://linear.app/bluearf/issue/MARKET-213))

Proposed first domain: one endogenous scalar, declared included exogenous
regressors/excluded instruments, fixed interior quantile and i.i.d. resident
CPU float64 sample. Structural quantile restrictions are
`E[Z * (tau - 1{Y <= D*alpha + X*beta})] = 0` under the chosen identifying
assumptions. Rank similarity, exclusion and monotonicity are substantive
conditions, not data-validation tests. Persist them as declared assumptions.

Stage point estimation through inverse quantile regression and a bounded search,
with full moment/objective/profile records. Separately validate identified
asymptotic covariance, conditional density/bandwidth and weak-ID test inversion.
An ordinary Wald CI requires justified identification; 2SLS AR/CLR tests cannot
be reused as quantile uncertainty. Confidence-set inversion must retain empty,
disconnected or boundary/unbounded sets plus grid coverage and numerical errors.
A finite grid does not establish the shape outside its scanned range.

Independent gate: CH author equations/objective and separately implemented
quantile solver on fixed fixtures; instrument rescaling and rank/sparsity/tie
failures; known exogenous reduction; weak/no-identification examples with correct
set behavior. Record sample/parameter order, complete covariance, p/CI and
quantile/moment metadata. Cluster/panel/weight variants and multiple endogenous
regressors remain separate gates. Licensed Stata IVQR matching output remains
pending; no vendor execution has occurred here.

## Q2 — QARDL ([MARKET-214](https://linear.app/bluearf/issue/MARKET-214))

Inspect the Cho/Kim/Shin author-linked GAUSS distribution, source/demo output and
license first. The repository readme identifies paper, example data and
estimation/test scripts; hashes and executable fixture reproduction are pending.
Define levels/conditional-EC/error-correction forms at every tau, deterministic
terms, lag selection rule and physical effective rows. Compare candidate orders
on a common declared sample under a bounded search; retain all failures.

Short-run effects and transformed long-run coefficients require joint parameter
covariance and denominator/nonidentification guards. Derive the paper's
cointegration and cross-quantile test scaling instead of importing mean-ARDL
bounds values or naive qreg SEs. Separate fixed finite-quantile-grid inference,
quantile-process tests, lag-selection conditioning and any proposed bootstrap.
EViews 14 supplies an official QARDL/QNARDL interface; it is a future comparison
target with exact sample/lags/terms/covariance settings, not validation evidence.

Acceptance stages: author demo reproduction; independent transformed covariance
and uncertainty/null calibration; time-gap/missing/lag lookahead failures;
licensed vendor match; public registry + persisted tau/lag/equation metadata;
frozen and installed-native restart/output. A regression on precomputed lags
does not meet these gates. QNARDL, panel, weights and continuous-process bands
remain explicitly later work.

## Q3 — research proposals ([MARKET-215](https://linear.app/bluearf/issue/MARKET-215))

Separate protocol gates are required for: (a) quantile VAR/cointegration with
identified dynamic innovations and quantile-specific responses, (b) local
quantile-on-quantile fits with bandwidth/support and joint uncertainty, and
(c) QTE targets with conditional/unconditional and rank-preservation assumptions.
Each needs an original-author source, reproducible licensed fixture, formal
uncertainty domain and a method-level priority decision before coding. Do not
combine heterogeneous estimands under a generic quantile label. Distinguish
official vendor commands from community extensions. Historical deferrals are
preserved; no completion schedule or implementation commitment is made.

## Primary reference gates

- [Chernozhukov–Hansen author inference paper](https://web.mit.edu/vchern/www/papers/ch_iqr_inference.pdf): IVQR estimation/inference foundation.
- [Chernozhukov–Hansen–Jansson author paper](https://eml.berkeley.edu/~mjansson/Papers/ChernozhukovHansenJansson07.pdf): different inference regions under weak instruments.
- [Stata official IVQR overview](https://www.stata.com/features/overview/instrumental-variable-quantile-regression/): vendor target; exact version/fixture still required.
- [Author-linked GAUSS QARDL distribution](https://github.com/aptech/gauss-qardl/blob/master/doc/read.me): source/demo acquisition target, not a production dependency.
- [EViews 14 official QARDL description](https://www.eviews.com/EViews14/ev14ecest_n.html): official built-in comparison target.

Source paper/manual reading, numerical replication and installed runtime evidence
are distinct. This plan promotes no new quantile estimator to supported status.
