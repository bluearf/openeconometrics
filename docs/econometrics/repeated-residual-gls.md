# Repeated-observation Gaussian residual GLS

`oe.repeated_gls` estimates the covariance of repeated residuals within
independent subjects. The mean is Xβ and each subject uses the submatrix
selected by its declared occasions. All observations and original row,
subject and occasion labels remain in the result. Unequal subject sizes and
missing occasion patterns are allowed as observed designs; missing values
in supplied model columns are errors.

## Eight scoped stages

| Issue | Structure | Criterion |
| --- | --- | --- |
| MARKET-648 | Negative-capable compound symmetry (`cs`) | ML |
| MARKET-649 | Negative-capable compound symmetry (`cs`) | REML |
| MARKET-650 | Signed integer-gap AR1 (`ar1`) | ML |
| MARKET-651 | Signed integer-gap AR1 (`ar1`) | REML |
| MARKET-652 | Occasion-specific diagonal (`diagonal`) | ML |
| MARKET-653 | Occasion-specific diagonal (`diagonal`) | REML |
| MARKET-654 | Full occasion covariance (`unstructured`) | ML |
| MARKET-655 | Full occasion covariance (`unstructured`) | REML |

These are new covariance/criterion combinations under MARKET-152. That
parent retains its Kenward–Roger, wider Satterthwaite, correlated random
effects and remaining GLMM scope.

Existing `mixed` and `mixedflex` estimate random-effect covariance with iid
residuals. Their `covstructure` option describes random effects.
`mixedflex` uses ML only. `xtgee` estimates a working correlation through
moment equations, and `xtgls` uses panel residual/Prais–Winsten estimates
with different variance and time domains. `meta_dependent` receives known
sampling covariance S. None of those procedures estimates the four
occasion-level residual covariance models delivered here.

## Public API

```python
result = oe.repeated_gls(
    data=data, y="y", x=["x", "z"], subject="subject", occasion="occasion",
    structure="ar1", method="REML", level=0.95, max_work=2_000_000_000,
)
```

The fixed intercept term is named `Intercept`. `x=None` fits an
intercept-only mean. The method
is `ML` or `REML`, and the four structure names are `cs`, `ar1`, `diagonal`
and `unstructured`. The [editable eight-stage example](../examples/repeated_residual_eight.py)
uses 32 subjects, unequal observed occasion patterns and actual integer
occasions 0, 1, 3 and 4.

## Covariance geometry

For a subject with m observed occasions, compound symmetry is
R=σ²[(1−ρ)I+ρJ]. It admits negative correlation while requiring
−1/(q−1)<ρ<1 for the declared q-occasion covariance. This ensures the
complete global occasion matrix is positive definite even if no subject
observes all q occasions. A negative covariance
cannot be represented by a nonnegative random-intercept variance.
The independent fixtures include full-occasion subjects, so this bound
also agrees with the maximum observed subject size used in the original
[nlme compound-symmetry source](https://svn.r-project.org/R-packages/trunk/nlme/R/corStruct.R).

AR1 uses R_ab=σ²ρ^|t_a−t_b| with −1<ρ<1 and integer occasion values.
The exponent is the actual time gap: occasions 1 and 3 have lag two.
Negative ρ alternates the covariance sign at odd/even lags. Observed
within-subject integer lags must have greatest common divisor one. This
is a conservative implementation admission boundary; an all-even grid
loses the sign, while some coarser odd grids can identify a signed AR1.
This matches
the signed integer-time domain of [nlme `corAR1`](https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/corAR1.html).
Continuous-time positive-correlation `corCAR1` is a distinct model and is
outside these stages; occasions are never silently replaced by row ranks.

The diagonal model estimates one positive residual variance per global
occasion. The unstructured model estimates every variance and covariance
of one shared positive-definite occasion matrix. Subjects missing an
occasion select the corresponding labelled submatrix. They do not receive
an upper-left matrix based only on their row count. The original author's
[`varIdent`](https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/varIdent.html)
and [`corSymm`](https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/corSymm.html)
provide the independent external reference geometry.

## Likelihood and inference

ML maximizes the complete Gaussian likelihood, including its constants,
after profiling the fixed mean. REML uses the declared normalized
error-contrast likelihood, including the design normalization
+0.5 log|X′X|. Stable scaled/centered design calculations retain the full
covariance transformation when results return to original units.
The covariance scale remains estimated rather than supplied as known.
[nlme `gls`](https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/gls.html)
documents separate ML/REML estimation for correlated or unequal residual
variance; the [Stata mixed manual](https://www.stata.com/manuals/memixed.pdf)
separates residual structures from random-effect covariance.

For ML, the fixed-effect covariance is the fixed block of the full joint
observed-information inverse. The fixed/covariance-parameter cross terms
are retained. For REML, fixed effects use the model-based plug-in GLS
covariance, and covariance parameters have a separate restricted
observed-information covariance. These are explicit asymptotic model-based
conventions. Fixed-effect confidence intervals and tests use normal
inference. Covariance-parameter intervals are asymptotic Wald intervals,
may cross parameter boundaries and have no calibrated boundary test. They do
not claim Kenward–Roger or finite-sample Satterthwaite calibration.

The external R/nlme fixture records both its reported covariance and the
unadjusted conditional GLS matrix. nlme's ML reported fixed covariance has
an N/(N−p) factor and is not the joint ML observed-information target used
here. Its REML likelihood also omits the selected +0.5 log|X′X|
normalization. Independent comparisons account for these conventions
explicitly; changing a reported constant never changes the fitted model.

## Complete result and saved operations

Each fit retains seven common tables. ML also retains the full joint
fixed/residual-coordinate covariance as an eighth table:

| Table | Content |
| --- | --- |
| `coefficients` | Every fixed term, estimate, SE, statistic, p-value and interval. |
| `covariance` | Every fixed-effect covariance entry. |
| `covariance_parameters` | Residual covariance coordinates and model-based uncertainty. |
| `covariance_parameter_covariance` | Complete covariance of reported residual parameters. |
| `residual_covariance` | Complete observation covariance in original positional order, including every within-subject entry and between-subject zero. |
| `fitted` | Every original observation with fitted mean and residual. |
| `fit` | Criterion, likelihood, Mahalanobis residual Q, dimensions, inference and convergence/work information. |
| `joint_covariance` (ML) | Every fixed, natural residual-parameter and cross covariance entry of the joint observed-information inverse. |

`fitted` links observation positions to every original row, subject and
occasion label. The complete q×q occasion covariance is also retained in
`result.attrs["state"]["fit"]["occasion_covariance"]`. ML state retains
the raw-parameter joint covariance in addition to the natural-parameter
joint covariance shown in the table.

Portable state retains the complete sample, design, outcome, roles and
labels, residual geometry, likelihood parameters, fitted covariance,
inference convention and resource/solver receipts. Restoration verifies
scientific consistency as well as the checksum; resealing malformed
scientific state does not make it a valid fit. The state is stored in
`result.attrs["state"]` with schema `openecon.repeated.gls.v1`.

`oe.restore_repeated_gls(state)` restores the canonical complete fit.
`oe.repeated_gls_predict(result, data=future, level=0.95)` returns mean
profiles in `means` and full joint mean uncertainty in `covariance`.
`oe.repeated_gls_contrast(result,
contrast=targets, null=null, level=0.95)` retains complete scalar/joint
linear-target uncertainty in `contrasts`, `covariance` and `joint_test`,
including finite nonzero nulls. These operations
use saved fits and do not optimize a new residual covariance.

Prediction concerns X_newβ. It does not condition on existing-subject
residuals, add future residual variance or supply random-effect BLUPs.
Covariance-parameter estimation uncertainty is not integrated into REML
plug-in mean intervals. The complete JSON example checks table order,
indices, dtypes, attributes, saved state and LaTeX, then restores and repeats
the saved prediction/contrast operations.

## Explicit admission limits

This first delivery uses resident CPU float64 with 2–6 global occasions,
4–64 independent subjects, at most 256 observations and 1–6 fixed terms
including the intercept. Every occasion requires at least four subjects;
every occasion pair requires at least four jointly observed subjects for
the unstructured model. Occasions are integral numeric values with
absolute value at most 1,000,000 for every structure. Selected outcomes
and predictors are finite numeric values with absolute value at most
1,000,000. Saved mean profiles are bounded to 256 rows.

The default and maximum structural work budget is 2,000,000,000 and the
resident workspace budget is 256 MiB. The complete state limit is 8 MiB.
Covariance identification, full-rank design,
positive-definite conditioning and observed-information checks can refuse
data within those upper limits. Work/workspace admission is a resource
boundary and does not establish statistical adequacy at a sample size.

Subject/occasion pairs must be unique; original identities and observed
patterns must be unambiguous. Unsupported or missing data, weights,
Dataset input, non-CPU execution, random effects, singular covariance,
nonconvergence or nonidentifiable information are refused. No automatic
ridge, pseudoinverse, subject deletion, row truncation or fallback to GEE
is used.

## Independent and packaged evidence

The independent fixture generator executes actual R/nlme for all eight
fits, retaining unbalanced data, complete fitted covariance and likelihood
conventions. Scientific tests consume the checked-in reference JSON and
need no R installation. Development-only NumPy/SciPy oracles are separate
from the float64 Torch production likelihood.

Source numerical checks, immutable source revision, wheel module identity,
frozen execution, installed native Run, real Quit/relaunch and complete
saved readback are separate acceptance layers. Evidence for these stages
does not establish licensed Stata execution, CUDA, public binaries,
general multilevel/GLMM support or the remaining MARKET-152 scope.

The [evidence helper](../../scripts/verify_repeated_residual_eight.py)
has separate `source`, `installed` and `frozen` modes. It preserves eight
complete fit JSON artifacts plus sixteen full saved mean/contrast
artifacts. The application displays eight compact coefficient tables;
displaying summaries does not discard the complete results. Whole module
source hashes and compiled identities are checked rather than inferring
packaging coverage from public function names.
