# Causal sensitivity and identification bounds

`evalue`, `manski_ate`, and `lee_bounds` return complete `TableSet` artifacts.
They compute sensitivity summaries or plug-in identification intervals. Their
reported bounds are not sampling confidence intervals; covariance, standard
errors and newly estimated confidence intervals are explicitly unavailable.

The supported domain is a resident numeric table and native CPU float64.
`Dataset`, non-CPU devices and generic observation weights are refused. The
common input guard admits at most 100,000 rows and selected raw numeric values
with magnitude at most `1e150`. Work and tensor workspace are checked before
the complete computation. An overflowing or underflowing scientific target is
refused rather than clipped or replaced with artificial precision.

## Direct risk-ratio E-values

```python
import openecon as oe

summary = {"rr": [2.0, 0.5], "lo": [1.2, 0.3], "hi": [3.0, 0.9]}
ev = oe.evalue(summary, "rr", lower="lo", upper="hi")
ev["evalues"]
```

One summary row is sufficient. All supplied risk ratios and limits must be
positive. Confidence-limit columns are optional, but must be supplied together
and satisfy `lower <= rr <= upper`. The RR=1 null is fixed. For an estimate
below one, the procedure uses its reciprocal; otherwise it uses the estimate.
For this ratio `r >= 1`, the E-value is `r + sqrt(r)*sqrt(r-1)`.

The confidence-limit E-value is one when the supplied interval contains the
null. Otherwise the endpoint closest to one is transformed: the lower limit
for increased risk and the upper limit for a protective association. Without
supplied limits, the confidence-limit result is undefined.

This summarizes the confounder association strength, on the risk-ratio scale,
needed to explain away the supplied association conditional on measured
covariates. It does not estimate actual confounding, prove a causal effect or
address other bias mechanisms. Odds ratios, hazard ratios and logged estimates
are not converted. See [VanderWeele and Ding's author manuscript](https://content.sph.harvard.edu/wwwhsph/sites/603/2017/08/EValue_Preprint.pdf)
and the [authors' EValue reference implementation](https://github.com/mayamathur/evalue_package).

## Consistency-only Manski ATE bounds

```python
bounded = {"y": [0.1, 0.3, 0.8, 0.9], "treated": [0, 0, 1, 1]}
mb = oe.manski_ate(bounded, "y", "treated", lower=0.0, upper=1.0)
mb["bounds"]
```

Declare finite common support `[L,U]` for both potential outcomes, with `L<=U`.
Both numeric treatment arms 0 and 1 must occur, and every observed outcome must
lie inside the declared support. Let `p` be the analyzed treated proportion,
`J1=mean(D*Y)` and `J0=mean((1-D)*Y)`. The sharp empirical ATE endpoints are

```
lower = J1 - J0 + (1-p)*L - p*U
upper = J1 - J0 + (1-p)*U - p*L
```

Joint moments preserve unequal arm proportions; a raw difference of arm means
does not replace these expressions. The saved `extremizers` table contains
every observed outcome and both endpoint-compatible potential-outcome
completions. They attain the interval endpoints. A singleton support identifies
zero treatment effect algebraically.

The assumptions are consistency/SUTVA and the declared potential-outcome
support, without ignorability or randomized assignment. The target is the
empirical distribution of analyzed rows. Explicit `missing="drop"` changes
that population and records each retained physical position. These plug-in
bounds do not cover sampling uncertainty about population endpoints. See
[Manski's original paper](https://ipcid.org/evaluation/outros_temas/Manski%20-%20Nonparametric%20Bounds.pdf).

## Lee bounds with monotone selection

```python
selected = {
    "y": [0.0, 2.0, None, None, 1.0, 2.0, 3.0, None],
    "treated": [0, 0, 0, 0, 1, 1, 1, 1],
    "observed": [1, 1, 0, 0, 1, 1, 1, 0],
}
lb = oe.lee_bounds(
    selected, "y", "treated", "observed",
    design="randomized", monotonicity="increasing",
)
lb["bounds"]
```

Both declarations are required. `increasing` means `S(1)>=S(0)` for every
unit; `decreasing` means `S(1)<=S(0)`. The target is the average treatment
effect among units always selected under either assignment, not among all
randomized subjects. Identification also requires random assignment independent
of potential outcomes/selection, consistency and a nonempty target population.

All original treatment and selection records determine each arm's selection
rate. Both are numeric 0/1 roles. Selected outcomes must be finite; unselected
outcomes may be missing and are ignored. `missing="drop"` is refused because
deletion could alter the selection rates. Both arms need at least one selected
outcome. Empirical rates that contradict the declared monotonicity direction
are refused; this is not a statistical test of population monotonicity.

The selected outcomes of the higher-selection arm retain a fraction
`rate_low/rate_high`. Lower and upper tail means trim exactly the required
fractional mass. Boundary ties share retention equally. Increasing selection
trims treatment; decreasing selection trims control and reverses the relevant
tail when forming treatment-minus-control bounds. Equal rates need no trimming.

Saved tables include the original arm counts, rates and every row's lower/upper
retention and normalized mean weights, including zero weights for unselected
rows. Observed outcomes, their original positions, schema and full design
sample enter the scientific identity; ignored unselected values do not.
See [Lee's author paper](https://www.princeton.edu/~davidlee/wp/resrevision8.pdf).

## Complete artifacts

```python
artifact = oe.causal_design_save(lb, "lee-bounds.json")
restored = oe.causal_design_load("lee-bounds.json")
assert oe.causal_design_save(restored) == artifact
```

Artifacts retain every table, column dtype, table order, typed row label,
physical sample position, input identity, declared assumption and scientific
state field. Checksums guard state and tables. Methods use no global random
draws. The implementation and independent reference tests are bounded method
checks; `stata_parity_validated` remains false.
