# Additional mixed and panel likelihoods

These are bounded native likelihoods, using float64 Torch chain-rule gradients
and the full joint observed-information inverse. Neither SciPy nor statsmodels
is required at runtime. Boundary or singular information is an explicit error;
there is no zero-variance fallback or automatic ridge.

| API | Supported domain | Explicit limits |
| --- | --- | --- |
| `oe.mixedflex` | Gaussian joint ML; crossed or nested intercepts at 1–8 levels; one lowest-level random slope; independent or unstructured lowest-level covariance | Resident dense observation covariance, planned before allocation; ML only; arbitrary reused child labels become composite nested identities |
| `oe.mixedlogit` | One or two normal random coefficients in alternative-specific utility; independent or unstructured covariance | Subject, case and alternative keys; exactly one chosen and at least two available alternatives; binary availability; explicit alternative constants; no implicit common intercept |
| `oe.menbreg` | NB2 log-mean GLMM; normal intercept and at most one slope; independent or unstructured covariance | Nonnegative integer counts; positive exposure; finite offset; full dispersion/random/fixed cross covariance |
| `oe.xtnbreg` | Hausman–Hall–Griliches conditional FE or beta-dispersion RE panel NB likelihood | FE conditions on panel totals and drops all-zero panels; its dispersion effect is not an additive log-mean intercept |
| `oe.xtfrontier` | Common firm inefficiency, truncated normal or half normal, production or cost; time invariant or exponential time decay | Calendar determines decay relative to each firm's final observation; conditional efficiency and full-parameter delta uncertainty are saved |

Existing `mixed`, normal GLMM and cross-sectional `frontier` methods retain
their APIs. New fits do not route through GEE or a cross-sectional frontier.
These five additional estimators currently require resident frames; the
capability inventory distinguishes them from the 80 streaming estimators.

Normal integrals use deterministic product Gauss–Hermite quadrature, so no
simulation RNG is consumed. At least two doubled orders are fitted. Acceptance
requires likelihood change below `1e-5 * subjects`, standardized joint parameter
shift squared below `1e-6`, and every normalized covariance change below `0.001`.
`intpoints=16`, `max_intpoints=64` are defaults; the maximum is 128 per axis and
4,096 product nodes. A rule that fails to settle is rejected. `max_work` counts
starting values, gradient evaluations and full Hessians; a workspace plan guards
the largest integration or dense covariance geometry before work begins.

Saved `mixedlogit` output includes population choice probabilities over the
available alternative set; they sum to one per case. `menbreg` population means
are `exp(Xβ + offset + z'Gz/2)`. In both cases full parameter gradients include
all variance/covariance coordinates. Frontier efficiency is conditional on the
complete firm's residual vector and calendar, with all likelihood parameters in
its delta covariance. Generic saved prediction for the other new likelihoods is
a separate domain; unsupported targets are rejected.

Validation uses independent NumPy/SciPy joint likelihoods, high-order quadrature,
numerical Hessians and derivatives, dense Gaussian optimization, and public
fixtures. Published `airacc`, `xtfrontier1`, `drvisits` and `transport` examples
cover NB likelihoods, frontier efficiency, variance/dispersion and choice
probabilities. The transport reference uses simulated Hammersley integration;
its different likelihood is not claimed as an identical objective. Penicillin
is fitted by ML; the lme4 published example uses REML, so its variance estimates
are not used as ML targets. These checks establish selected methods and domains,
not full Stata parity.

Primary references: [Stata panel NB manual](https://www.stata.com/manuals/xtxtnbreg.pdf),
[panel frontier manual](https://www.stata.com/manuals/xtxtfrontier.pdf),
[mixed NB manual](https://www.stata.com/manuals/memenbreg.pdf),
[panel mixed choice logit manual](https://www.stata.com/manuals/cmcmxtmixlogit.pdf),
and [lme4 Penicillin documentation](https://lme4.github.io/lme4/reference/Penicillin.html).
