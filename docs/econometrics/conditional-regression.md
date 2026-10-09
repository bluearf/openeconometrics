# Bounded conditional exact regression

`exact_logit_fit` and `exact_poisson_fit` estimate one target coefficient after
conditioning on the intercept sufficient statistic (total response) and up to
three declared nuisance sufficient statistics. Nuisance coefficients are not
estimated. Numeric target/nuisance columns must be integers in [-8,8], without
implicit rescaling or category expansion. Independent Bernoulli responses or
independent Poisson counts with known exposures define the underlying model.

The complete feasible allocation space holds `sum(y)` and each `z' y` fixed.
Bernoulli allocations have unit base measure; Poisson allocations have
`prod(exposure_i ** y_i / factorial(y_i))`. Aggregating these base measures by
`S = x' y` gives `P_beta(S=s) proportional to base_s * exp(beta*s)`.
The CMLE solves `E_beta(S)=S_observed`; its conditional information is `Var_beta(S)`.
Extreme observed statistics give an infinite CMLE, recorded as a null numeric
coefficient with an explicit lower/upper boundary. No median-unbiased substitute
is silently inserted. Fit reports no Wald SE, coefficient covariance, p or CI.

`exact_logit_ci` / `exact_poisson_ci` invert inclusive tails at `(1-level)/2`
for central exact coefficient intervals. Infinite endpoints remain explicit;
finite roots include brackets, iteration traces and residuals. Finite numerical
roots must lie in [-40,40], otherwise the procedure fails rather than clipping.
`exact_logit_test` / `exact_poisson_test` sum grouped statistic probabilities no
greater than the observed statistic's probability at the supplied null beta.
Ties use a declared 1e-12 log-probability tolerance. This probability-ordered
p-value need not equal a doubled smaller-tail p-value or invert the central CI.
No mid-p, score, Monte Carlo or asymptotic fallback is provided.

The conditioning and inference definitions follow the official mathematical
formulas in [Stata exlogistic](https://www.stata.com/manuals/rexlogistic.pdf) and
[Stata expoisson](https://www.stata.com/manuals/rexpoisson.pdf). These are formula
references; no licensed Stata executable was used and no vendor parity is claimed.

`exact_logit_moments` / `exact_poisson_moments` weight individual saved allocations
to report training response means and the complete response covariance. A supplied
coefficient defines a known conditional distribution; `coefficient=None` selects
an explicitly labelled CMLE plugin. At an infinite CMLE the limiting support face
retains its actual base-measure weights. Response SD is distributional spread,
not a standard error of an estimated mean. No parameter uncertainty, unconditional
new-row prediction or prediction interval is supplied. Original row positions,
finite numeric/string labels and excluded rows persist; covariance axes identify
used original positions, including duplicate labels.

CPU float64 is pinned locally even under a different caller default device.
Bernoulli input permits at most 16 original rows. Poisson permits at most 8
original rows and 24 total counts; known exposures lie in [1e-12,1e12]. The
entire unfiltered `choose(n,M)` or `choose(M+n-1,n-1)` allocation count must be
at most `max_states` (hard ceiling 20000). Named workspace and structural work
plans run before enumeration and support/covariance allocations. `max_work`
defaults to 100000000. Input design must be full rank and conditioned target
support must vary. Missing handling is explicitly `raise` or joint `drop`.
Booleans, weights, Dataset/streaming, categorical expansion, GPU and estimated
exposures are unsupported. This is one-target conditional inference, not joint
multivariable coefficient inference or large-sample exact computation.

`conditional_save` / `conditional_load` preserve all ordered tables, full feasible
allocations, aggregated weights, response covariances, sample identities,
constraints, root diagnostics and settings under a checked 32 MiB schema.
Postestimation validates the unmodified fit and never refits its support.
The [editable eight-method example](../examples/conditional_eight.py) and
`tests/test_conditional_exact.py` demonstrate independent stars-and-bars/product
space enumeration, binomial/hypergeometric reductions, exact size/coverage,
nonuniform limiting faces, joint missing alignment and roundtrip/tamper checks.

MARKET-489..496 close only these eight bounded procedures. MARKET-176 / GitHub
#63 remain open for broader joint exact inference/options, arbitrary weighted or
categorical inputs, large-n FAST-LTS and licensed vendor comparisons. Prior Firth,
fixed-margin odds, fixed-exposure rate, LTS and S/MM computation is preserved.
