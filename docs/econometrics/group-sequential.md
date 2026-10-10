# Gaussian group-sequential spending designs

`oe.sequential_design` calibrates prespecified efficacy boundaries for a
canonical known-information Gaussian model. `oe.sequential_power` evaluates
the saved boundaries at a specified effect and maximum information;
`oe.sequential_information` solves for continuous maximum information at a
target rejection probability. `oe.restore_sequential_design` reconstructs
the complete saved design without searching for new boundaries.

Four spending laws are available, each with a one-sided upper rejection rule
or a symmetric two-sided rejection rule. These are eight bounded design
configurations under MARKET-175. The existing fixed-look power, precision
and accrual methods remain separate.

```python
import openecon as oe

design = oe.sequential_design(
    fractions=[0.25, 0.5, 0.75, 1.0],
    spending="hsd",
    param=-4.0,
    sides=2,
    alpha=0.05,
    effect=0.3,
    information=100.0,
)
power = oe.sequential_power(design, effect=0.2, information=150.0)
required = oe.sequential_information(design, effect=0.3, power=0.8)
design.to_latex()
```

## Probability model and stopping rule

The planned analyses occur at fixed fractions
`0 < t_1 < ... < t_K = 1` of maximum information `I`. The input `effect`
is a signed mean parameter relative to its null value, in the units whose
known information is supplied. Under the model,

`E[Z_i] = effect * sqrt(I * t_i)`,

`Var(Z_i) = 1`, and

`Cov(Z_i, Z_j) = sqrt(min(t_i, t_j) / max(t_i, t_j))`.

Equivalently, the underlying Gaussian score process has independent
increments. The chosen information clock must justify that model. The
calculation does not fit an observed sample or estimate its variance.
The canonical joint-normal model is described in the
[gsDesign theory manual](https://keaven.github.io/gsd-tech-manual/testing.html).

With `sides=1`, the trial rejects and stops at its first `Z_i >= b_i`.
There is no lower stopping boundary. With `sides=2`, it rejects and stops
at its first `Z_i >= b_i` or `Z_i <= -b_i`. The two directions are efficacy
rejections; the negative boundary is not a futility boundary. Every later
crossing probability excludes paths that already stopped in either
direction. Trials that do not reject still consume the final information.

The reported total rejection probability is the sum of the mutually
exclusive first-crossing probabilities. For a two-sided design it includes
both directions; the upper and lower contributions are retained separately.
A negative effect is meaningful for a two-sided design. For a one-sided
upper test, negative effects point away from the rejection direction.

Expected information is

`E[I_stop] = I * (sum_i(t_i * P[first rejection at i])
                  + P[no rejection through K])`.

Thus final non-rejecting paths are included. Dividing by `I` gives an
expected information fraction. Neither quantity is an integer sample-size
or patient-count result. Mapping information to sample size requires a
separate justified endpoint and variance model. The
[official gsProbability reference](https://keaven.github.io/gsDesign/reference/gsProbability.html)
also separates per-analysis crossing probabilities and expected information
or sample-size units for a supplied design.

## Total alpha and spending laws

`alpha` is the **total null rejection probability across all looks and
directions**. It is not the nominal fixed-look alpha at every analysis.
Set `a = alpha / sides`. At each fraction `t`, OpenEcon uses the official
spending law with the per-direction argument `a`; the total cumulative
target is `sides * S(a, t)`. At `t=1` this is `alpha`.
For example, a two-sided design with `alpha=.05` uses `.025` in each call
to the spending function. Its calibrated per-direction first-crossing
probability is `.025`, and its total is `.05`.

| `spending` | Per-direction cumulative target `S(a,t)` | Parameter |
| --- | --- | --- |
| `ldof` | `2 * Phi_bar(Phi_inv(1-a/2) / sqrt(t))` | Classical Lan–DeMets exponent fixed to one |
| `ldpocock` | `a * log(1 + (e-1)*t)` | No shape parameter |
| `hsd` | `a * (1-exp(-gamma*t)) / (1-exp(-gamma))` | `param=gamma`; the continuous `gamma=0` limit is `a*t` |
| `power` | `a * t**rho` | `param=rho` |

When `param` is omitted, HSD uses `gamma=-4` and power uses `rho=2`.
The `ldof` and `ldpocock` configurations reject a supplied shape parameter.

`Phi_bar` is the standard normal upper-tail probability. The formulas for
`ldof` and `ldpocock` follow the
[official Lan–DeMets reference](https://keaven.github.io/gsDesign/reference/sfLDOF.html).
They approximate the error-spending shapes of O'Brien–Fleming and Pocock
designs; they do not promise the classical constant Pocock boundary or a
classical O'Brien–Fleming boundary sequence. Generalized Lan–DeMets
exponents are outside this `ldof` configuration.

The Hwang–Shih–DeCani formula follows
[sfHSD](https://keaven.github.io/gsDesign/reference/sfHSD.html); stable
`expm1` arithmetic and its exact linear limit avoid a zero-over-zero
calculation near `gamma=0`. The Kim–DeMets power formula follows
[sfPower](https://keaven.github.io/gsDesign/reference/sfPower.html).
OpenEcon's admitted shape ranges are narrower than the reference package's
ranges and are stated below.

Calibration uses the cumulative probability of first rejection under
`effect=0`, including the previously calibrated stopping rules. It does
not assign an isolated tail quantile to every spending increment or reuse
a fixed-look significance threshold. For the symmetric two-sided rule,
both boundaries bind when computing that null crossing probability.

## Fixed-boundary power and information

`sequential_power` preserves the calibrated null boundaries, spending law,
alpha and planned fractions. A different `effect` changes the Gaussian
drift. A different `information` scales that drift and the absolute
information at the same relative analyses. It does not recalibrate alpha
or change the stopping rule.

`sequential_information` solves along this same fixed-fraction,
fixed-boundary information path. It brackets an attainable target, solves
continuously and checks the achieved probability and bracket. Its answer
is maximum information, with the design's expected stopping information
reported separately. It is not a minimum integer `N` calculation, an MDE
solver or observed power. A target that cannot be reached inside the
admitted information and drift domain raises an explicit error.

The target must lie strictly between total alpha and one. One-sided upper
inversion requires a positive effect; two-sided inversion accepts either
nonzero direction. The returned information corresponds to the upper
endpoint of a verified drift bracket: its rejection probability reaches
the target and exceeds it by at most `2e-6`, while the lower endpoint
remains below it. The receipt retains both endpoints and the probability
refinement checks. This is a continuous numerical crossing result.

Changing the fractions after observing a result, adding unscheduled looks,
effect-dependent redesign and sample-size re-estimation are outside this
contract. A newly specified design requires fresh calibration.

## Numerical admission and computation

The implementation uses deterministic native PyTorch CPU float64
Markov-Gaussian quadrature. Composite panels of width at most six use
128- and 256-point Gauss–Legendre rules per panel. Probability and boundary
refinement must pass an absolute `1e-8` comparison; null-spending calibration
is checked separately at `2e-11`, and probability conservation at `2e-10`.
The implementation returns a result only when these checks pass. No Monte
Carlo or independent fixed-look approximation substitutes for a refused
calculation.

Where the one-sided continuation integral truncates its left tail, the
cutoff is centered relative to the relevant Gaussian mean and a declared
bound of at most `K * Phi_bar(12)` covers omitted probability. This tail
bound is separate from the quadrature refinement check. Passing two
resolutions is a numerical admission check, not a universal proof of the
quadrature error or a statistical coverage simulation.

The resident implementation admits 2–6 analyses, first fraction at least
`.2`, gaps at least `.05`, and a final fraction exactly `1`. Fractions
must already be ordered and valid; they are not sorted, clipped or
normalized. Total alpha is in `[.01, .1]`; HSD gamma is in `[-8,8]` and
power rho in `[.25,4]`. For every design or query,
`abs(effect * sqrt(information)) <= 8`. Information must be in
`[1e-12, 1e6]`, and `abs(effect)` cannot exceed `1e6`. Fractions are supplied
as a list, tuple or one-dimensional CPU float64 Torch tensor. Each
operation has a cumulative integer `max_work` budget, bounded above
by two billion work units, covering its calibration, probability
evaluations, information inversion and replay checks as applicable.
Cold quadrature preparation is charged once within that aggregate;
integration, tail evaluations and boundary-search steps are charged
separately. The receipt distinguishes preparation and integration work.
Exceeding the budget is an error.

These limits are implementation and numerical-resource boundaries, not
mathematical requirements of the spending functions. A sample or
configuration within the upper caps can still fail numerical refinement
or an inversion-attainability check. DataFrames, Dataset, observation
weights, device selection, missing values and boolean numeric settings
are unsupported. There is no GPU or streaming route.

## Complete portable results

Each result retains its primitive settings and derived boundaries,
spending targets, full canonical correlation, per-look first crossings,
continuation probabilities and complete summary. Native table and LaTeX
output describe the same stopping policy and information units.
Portable saved state retains sufficient design identity to reconstruct
the entire result and to query the fixed boundaries later.

| Table | Retained content |
| --- | --- |
| `summary` | Total and per-direction alpha, effect, maximum and expected information, null and alternative rejection probability |
| `boundaries` | Each fraction and information, upper/lower boundary, lower-presence flag and cumulative/incremental total alpha |
| `null_stages`, `alternative_stages` | Probability of reaching each look, first upper/lower rejection and continuation |
| `canonical_covariance`, `canonical_means` | Full labelled Gaussian covariance and null/alternative means |
| `numerical_accuracy` | Calibration, refinement, conservation and omitted-tail checks with their distinct tolerances |
| `information_inversion` | Verified target and drift/probability bracket; present only after information inversion |

A one-sided absent lower boundary is stored as `None` with a false
`lower_present` flag. It does not become a zero boundary or an undefined
numeric cell. A subsequent power query removes a prior inversion receipt
because it evaluates a new requested alternative.

`result.attrs["state"]` has schema `openecon.sequential.design.v1` and
retains `settings`, `calibration`, `alternative`, optional `inverse` and a
checksum. Primitive settings are fractions, spending law, sides, total
alpha, shape parameter, effect and maximum information. Derived state
retains both quadrature resolutions and their work/workspace receipts.

```python
import json

state_json = json.dumps(design.attrs["state"], sort_keys=True, allow_nan=False)
restored = oe.restore_sequential_design(state_json)

# The generic complete-summary codec can also carry the saved design.
summary_json = oe.summary_state(required)
portable_tables = oe.restore_summary(summary_json)
restored_required = oe.restore_sequential_design(portable_tables)
```

Restoration checks schema, checksum, types, dimensions, support, spending
identity, stopping geometry, numerical consistency and derived results.
It re-evaluates the saved boundaries without boundary calibration or a
new information search. Recomputing a checksum does not make an
inconsistent altered design valid. Sorted JSON key order does not change
the canonical table order or complete restored result. Complete TableSet
replay also checks every table's values, ordered rows and columns, labels,
axis names, table attributes, result title and attributes. Table mapping
key order is immaterial; replay reconstructs the canonical order and
natural numeric dtypes. Boolean values and labels retain their distinct
types.

Saved workspace budgets describe the original admitted computation;
restoring under a different adequate workspace keeps that provenance and
the original complete result bytes. Replay still checks the current
workspace and cumulative work allowance. An insufficient current budget
is refused even when the original saved computation had a larger budget.
`restore_sequential_design` accepts the raw state, its JSON string or a
complete TableSet and also supports `max_work`.

## Scientific and delivery boundaries

Independent tests compare the spending formulas, calibrated boundaries,
null and alternative first crossings, expected information and continuous
information inversion. Actual R/gsDesign execution is a separate external
reference layer. It uses the same per-direction alpha, fractions,
binding efficacy boundaries and information convention. For two-sided
information inversion the reference independently solves the total
`gsProbability` upper-plus-lower rejection probability. The default
`gsDesign` planning convention can target upper-direction power alone;
its planned sample size is not copied as an oracle for this total-rejection
target. Source tests,
wheel/sdist contents, frozen-runtime execution, actual installed native
Run and Quit/relaunch, fresh hosted head/base tests, merge and tracker
readback have separate receipts. An earlier layer does not establish a
later delivery layer.

This family does not claim unknown-SD Student-t sequential inference,
exact binomial or PH-survival endpoint planning, futility or harm rules,
estimated ICC, adaptive or data-dependent looks, conditional error
redesign, overshoot/enrolment overrun, calibrated integer sample size,
generalized Lan–DeMets exponents, public binary release, licensed vendor
execution or blanket vendor parity. MARKET-175 remains open for its
remaining prospective-design scope after these eight configurations.
