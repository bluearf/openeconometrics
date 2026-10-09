# Bounded finite regression

MARKET-385..392 deliver eight explicit resident CPU float64 procedures under
MARKET-176 / GitHub #63. Existing Huber, biweight, S and MM methods are unchanged.
These are method-level results, not blanket Stata parity or the completion of
the broader parent issue.

| API | Declared behavior | Inference |
| --- | --- | --- |
| `firth_logit` | Numeric Bernoulli full-model Jeffreys-penalized fit | Full inverse expected-Fisher covariance; asymptotic normal Wald |
| `firth_predict` | Saved link and probability, original new-row identities | Full-covariance delta SE; transformed link-normal conditional mean CI |
| `firth_profile` | Constrain a raw coefficient and refit every nuisance parameter | Invert asymptotic one-df chi-square penalized LR |
| `firth_test` | Full-row-rank raw linear constraints in the same full model | Penalized LR with restriction-rank asymptotic chi-square; full contrast covariance |
| `exact_logistic` | One common odds ratio over independent fixed-margin 2×2 strata | Complete conditional distribution, CMLE, inclusive equal-tail CI and doubled smaller-tail p |
| `exact_poisson_rate` | Integer events with positive known fixed exposure | Garwood central exact CI and inclusive doubled smaller-tail Poisson p |
| `lts` | Exhaustively enumerate every declared h-subset QR fit | Descriptive coefficients and computed global trimmed-SSE minimum in the bounded full-rank domain |
| `lts_predict` | Saved LTS mean and complete new-row identities | Descriptive; no fabricated SE, p or CI |

The [editable synthetic example](../examples/finite_eight.py) emits eight complete
results. `finite_save` and `finite_load` preserve every ordered table, original
sample position/label, transform, full covariance, solver trace, exact support
and candidate subset, with inner and outer SHA-256 checks. Reloading never refits.
Artifacts are bounded to 32 MiB. Numeric designs accept `missing="raise"` or an
explicit jointly complete-row `missing="drop"`; every exclusion is recorded.
No observation weights, categorical expansion, Dataset collection or GPU fallback
is provided. Boolean numeric columns must be explicitly recoded.

Firth maximizes

`sum(y*eta - softplus(eta)) + 0.5*logdet(X' diag(sigmoid(eta)*sigmoid(-eta)) X)`.

Predictors are centred and scaled when an intercept is present, and only scaled
otherwise; the entire parameter vector and covariance are transformed to the raw
basis. Supported inputs have at most 2000 rows and eight coefficients, including
any intercept; predictors lie within ±1e6. Fit convergence requires a bounded
free score and positive negative-Hessian curvature. Safeguarded Newton ascent is
a local numerical optimum, not a general global-optimum assertion. The
expected-Fisher covariance is explicitly **not** the inverse penalized Hessian.
Profile and test restrictions are transformed from the raw basis and the same
full p-dimensional Jeffreys determinant is retained during nuisance refits.
No smaller-model penalty is substituted. Failed fits, brackets or roots return
an error without partial intervals. Firth mean predictions are not FLIC/FLAC
corrections and are not asserted to be unbiased.

The original authors' [logistf documentation](https://search.r-project.org/CRAN/refmans/logistf/html/logistf.html)
distinguishes Wald and penalized-profile inference. Tests independently check
saturated two-group analytic estimates `(events+0.5)/(n+1)`, every raw covariance
entry and Wald quantity, and non-saturated NumPy/SciPy objective, nuisance-profile
roots and constrained tests. SciPy/statsmodels are development oracles only and
are excluded from the frozen runtime.

For exact odds, pass `tables=[[[a,b],[c,d]], ...]`, where rows are exposed and
unexposed and columns are events and non-events. All strata are declared
independent with fixed row and outcome margins. Stratum support is
`max(0, events-unexposed)..min(exposed, events)`; the full sufficient statistic
is the sum of exposed events. Log-space convolution retains every support point.
At most 16 strata, 4096 total counts and 513 complete support points are accepted;
log-odds roots must bracket within ±40. Completely uninformative margins are
refused, but uninformative individual strata stay in a jointly informative
result. Null/fit probabilities and all margins are saved. Infinite CMLE or
upper CI values use `None` plus an explicit boundary flag; zero endpoints remain
zero. There is no median-unbiased substitution, probability-order Fisher test,
mid-p, Monte Carlo or multivariable exact-regression claim.

[Stata's exact logistic manual](https://www.stata.com/manuals15/rexlogistic.pdf)
provides the conditional model context. Stata's boundary MUE substitution differs
from the explicit CMLE boundary convention here. Single-stratum exact intervals
are checked against SciPy's conditional odds routine; multi-stratum probabilities
and inclusive tails against an independent integer-binomial polynomial
convolution. These are not licensed Stata executions.

For exact Poisson rate, the lower bound is
`0.5*chi2_ppf(alpha/2, 2*K)/exposure` for `K>0`, otherwise zero; the upper bound is
`0.5*chi2_isf(alpha/2, 2*(K+1))/exposure`. Central zero-event upper coverage is
one-sided `(1+level)/2`. Events and null mean are at most 10000; exposure/null
rate lie in `[1e-12, 1e12]`. `K/exposure²` is labelled a descriptive MLE plugin
variance and is not used to fabricate a Wald interval. Known exposure and a
Poisson sampling model are necessary; covariate Poisson regression and estimated
exposure are outside scope. [Stata's confidence-interval manual](https://www.stata.com/manuals13/rci.pdf)
includes the checked rounded examples K=84, exposure=36; K=27, exposure=1; and
K=0, exposure=36. Independent SciPy gamma/Poisson tests also cover tail and
zero-count behavior.

LTS minimizes the sum of the h smallest squared residuals. The
[original robustbase method documentation](https://search.r-project.org/CRAN/refmans/robustbase/html/ltsReg.html)
records this criterion and the default `floor((n+p+1)/2)`. This implementation
uses a different explicitly exhaustive small domain: n≤24, p≤3, p+1≤h≤n,
no more than 20000 h-subsets. The finite minima commute:
`min_beta min_subset SSE = min_subset min_beta SSE`.
Every subset has a full-rank normalized QR fit; any rank/condition failure
rejects the entire fit. Every computed subset/objective/condition is retained,
with deterministic enumeration and residual tie order. The claim concerns the
minimum of the complete **computed float64** domain, not exact-real arithmetic.
No random FAST-LTS, reweighting, consistency-corrected scale or selected-model
OLS uncertainty is asserted. Independent NumPy QR enumerates every candidate,
checks the minimum/coefficients, and compares contaminated and h=n cases.

All methods guard structural work and named buffer estimates before constructing
the design/autograd/covariance/conditional-support/candidate blocks. Increase
`max_work` explicitly for complete profiles within supported dimensions. The
resource plan is not a bound on total process RSS, caller-owned input, Python
allocator/BLAS internals or integrity serialization. No support truncation or
partial candidate scan is used to satisfy budgets. Full-rank design/subsets
require condition≤1e10; expected Fisher condition≤1e12. Firth logits beyond ±100
are refused.

The parent retains multivariable exact models, arbitrary weighted/categorical/
streaming geometry, large-n FAST-LTS and licensed vendor validation. Installed
native QA uses a dedicated application identity and synthetic local project;
source tests, frozen execution, native Run, actual quit/restart, installed payload
and merge/tracker readback are separate proof layers. Public releases, Windows,
notarization and CUDA/MPS execution are not implied.
