# Continuous person scores for a fixed calibrated IRT bank

`irt_score_mle` and `irt_score_map` estimate one continuous latent trait per
person, conditional on **known, supplied item parameters**. They do not estimate
item parameters, a latent population distribution or calibration uncertainty.
These are separate from the fitted-family `irt_score` EAP function and from the
finite-support posterior API: no quadrature approximation supplies these modes.

```python
import openecon as oe

bank = oe.irt_bank_binary(
    ["a", "b", "c"], [1.0, 1.2, 0.8], [-0.7, 0.0, 0.8]
)
mle = oe.irt_score_mle(bank, data=responses, level=0.95)
map_scores = oe.irt_score_map(
    bank, data=responses, prior_mean=0.4, prior_sd=1.2
)
display(mle["people"])
display(map_scores["people"])
restored = oe.restore_summary(oe.summary_state(map_scores))
```

Resident DataFrames, named column mappings and explicit row records are
accepted. Each bank item must be present once; unneeded columns are projected
out. Numeric integer responses use category codes `0..K-1`, with missing items
skipped in that person's conditional likelihood. Boolean, fractional, string,
infinite and unknown responses are refused. Original row positions and primitive
index labels are preserved, including duplicated indices. No listwise deletion,
weights or silent recoding occurs.

## Strict concavity and the admitted root

The complete bank must contain only binary **2PL** items (`guessing=0`,
`upper=1`), positive-slope GRM/GPCM items, or baseline-zero NRM items with
nonconstant category slopes. Nominal scoring values do not determine the NRM
likelihood, category order, mode or curvature. General 3PL/4PL banks are refused
before selecting responses because their observed conditional likelihoods need
not be concave.

For a known bank, a GRM middle-category log probability is a sum of two
log-logistic terms and a threshold-gap constant; its second derivative is
negative. GPCM curvature is minus the variance of its linear category slopes,
and NRM curvature is minus the variance of its declared category slopes. This
establishes strict concavity for an informative observed response. A normal
prior adds strictly negative curvature to the MAP objective.

CPU float64 Torch evaluates the likelihood, gradient and Hessian. Safeguarded
bisection requires a positive lower-endpoint gradient and negative upper-endpoint
gradient, both beyond the requested tolerance. Every accepted mode has an
absolute gradient no larger than `tolerance`, information greater than `1e-12`,
and distance greater than `max(1e-9, 10*tolerance)` from the search boundary.
There is no clipping or boundary estimate presented as an interior root.
Evaluation under a caller's disabled gradient/inference context is scoped
locally, and the caller's context is restored.

MLE refuses fully missing people, all-correct/all-incorrect unbounded binary
patterns, modes outside the admitted bracket, weak curvature and failures to
reach the gradient tolerance. If any person fails, the entire call raises an
explicit error. MAP uses a caller-declared **normal** prior with mean in
`[-4,4]` and SD in `[0.25,3]`. A fully missing row returns that prior mean and
SD exactly, with zero observed likelihood information, provided its mode has
an admitted interior bracket. MAP also refuses an out-of-bracket mode.

## What the uncertainty columns mean

MLE reports `standard_error = 1/sqrt(observed_information)` and two-sided normal
Wald intervals at the supplied `level`. These are **conditional, asymptotic**
fixed-item-parameter intervals; short tests need not give accurate coverage.
There are no item-parameter uncertainty, robust/cluster corrections or
finite-sample guarantees. Before the bank or responses are materialized, `level`
must give distinct interior float64 equal-tail probabilities
`0 < (1-level)/2 < .5 < 1-(1-level)/2 < 1`. Values such as `1e-300` or the
representable float immediately below one are refused when the tails round
together or onto an endpoint. The normal critical value must also be finite
and strictly positive. For every person, the computed endpoints must be finite
and strictly straddle the mode in float64; a rounded zero-width interval is
refused even when the probability and critical value remain distinguishable.

MAP reports `laplace_sd = 1/sqrt(mode_information)`, where mode information
includes `1/prior_sd**2`. This is a **local inverse-curvature Laplace
approximation**, not the exact posterior SD or a frequentist confidence
interval. MAP therefore has no confidence-interval or p-value columns.
The all-missing prior is the exact normal exception to that approximation.

## Complete saved results and resource admission

Both methods return a `TableSet` with complete `people`, `responses`, `sample`,
`trace`, `certificates` and `settings` tables. The trace records every endpoint
and bisection evaluation, objective, log likelihood, gradient, information and
current bracket. The fixed bank, controls, original sample positions and
admission plans are retained in full metadata. `oe.summary_state`/
`oe.restore_summary` preserve all rows and metadata, with identical complete
LaTeX; console previews and settings-cell pointers are not complete exports.
Restoring the bank's complete summary also reproduces subsequent scoring.

| Admission | Bound |
| --- | --- |
| People / fixed bank items | 1..1000 / 1..16 |
| Categories per item | 2..6 |
| `theta_limit` / actual bracket | 0.25..12 / `[-theta_limit,theta_limit]` |
| `tolerance` | `1e-12..1e-5`; default `1e-9` |
| `max_iter` | 1..200; default 100 |
| `max_work` | at most 300,000,000 charged work units |
| `max_bytes` | at most 128 MiB and the current global workspace budget |

Before response copying, admission uses the worst-case evaluation count
`n*(max_iter+3)` times `64*sum(K_j)` charged work units. Every actual objective,
gradient and curvature evaluation is charged again before its tensor is built.
The named workspace plan includes selected input/masks, response codes, person
results, a conservative per-person autograd graph allowance, the entire trace
and bounded bank encoding. Combinations at the individual dimension maxima
may exceed work or workspace admission. The plan bounds named live buffers,
not caller tables, Python result objects, allocator workspace or process RSS.

## Sources and independent checks

The conditional ML/MAP scoring and observed-information distinction are
documented by [mirt's author-maintained scoring reference](https://philchalmers.github.io/mirt/reference/fscores.html)
and [Stata's posterior-mode reference](https://www.stata.com/manuals/irtirt2plpostestimation.pdf).
The concavity statements above are algebraic consequences of the published
[GRM](https://www.stata.com/manuals/irtirtgrm.pdf),
[GPCM](https://www.stata.com/manuals/irtirtpcm.pdf) and
[NRM](https://www.stata.com/manuals/irtirtnrm.pdf) probability definitions;
they concern person theta with fixed parameters, not joint calibration
concavity or software parity.

`tests/test_irt_calibrated_scores.py` checks the identical-item Rasch closed-form
MLE and intervals, independent NumPy analytic scores/curvature and SciPy
bracketed roots/optimization for each family and mixed polytomous banks,
finite-difference Hessians, subnormal GRM gaps, arbitrary nominal score order,
partial/empty/extreme responses, original identity, restored/tampered bank
state, early resource/input refusal, CPU/default-device/gradient contexts and
complete JSON/LaTeX roundtrips. These bounded numerical checks do not establish
universal parameter recovery, interval coverage, licensed vendor parity,
native-app delivery or public release availability.
