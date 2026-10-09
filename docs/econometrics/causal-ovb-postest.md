# Control benchmarks and robustness thresholds

`oe.ovb_benchmark` and `oe.ovb_robustness` take an **intact original**
`oe.ovb_sensitivity` TableSet. They verify its complete artifact, state and
table checksums and coefficient/sample contract. A restored original artifact
is accepted; another procedure's result, a table or a mutated result is refused.
Both methods retain the original complete sample, row positions and labels,
selected input state, coefficients, complete original covariance and typed
tables. They do not refit the original outcome model or select a new sample.

```python
base = oe.ovb_sensitivity(data, "y", "d", ["age", "baseline", "income"])
benchmarks = oe.ovb_benchmark(
    base, groups={"age": ["age"], "baseline_income": ["baseline", "income"]},
    kd=[0.5, 1.0], ky=[0.5, 1.0], assumption="residualized_benchmark",
)
thresholds = oe.ovb_robustness(base, q=[0.5, 1.0], alpha=[0.05, 1.0])
oe.causal_design_save(thresholds, "robustness.json")
restored = oe.causal_design_load("robustness.json")
```

## Observed-control benchmarks

`groups` is a required dictionary with 1 to 32 nonempty named lists of original
controls. A group can contain one or several controls. Its names must be unique;
different groups may overlap because each supplies a separate comparison.
Treatment, outcome, intercept and newly supplied columns cannot be benchmarks.
`kd` and `ky` are strictly increasing grids of nonnegative multipliers, with at
most 64 entries each. Every group × kd × ky combination is evaluated.

Let G be a control group and X−G the remaining controls. The observed strengths
are A=R²(D ~ G | X−G) and B=R²(Y ~ G | D,X−G). They describe the reduction in
residual variation when adding G to the respective regressions. The procedure
computes their exact classical group-Wald/FWL equivalents on the original
sample; grouping jointly accounts for correlations between controls.

Explicitly declare `assumption="residualized_benchmark"`: Z denotes the
hypothetical omitted scalar confounder's component orthogonal to every observed
control. The substantive assumptions are

```
R²(D ~ Z | X−G) <= kd * A
R²(Y ~ Z | D,X−G) <= ky * B
```

These assumptions are not established by measuring A and B. Under them, the
formal bounds in [Cinelli and Hazlett (2020), section4.4 and supplementB.2](https://doi.org/10.1111/rssb.12348)
and the [authors' benchmark implementation](https://github.com/carloscinelli/sensemakr/blob/master/R/ovb_bounds.R)
are

```
d_bound = kd*A/(1-A)
u = kd*A²/((1-kd*A)*(1-A))
y_bound = ((sqrt(ky)+sqrt(u))/sqrt(1-u))² * B/(1-B)
```

Here d_bound bounds R²(D ~ Z | X), and y_bound bounds R²(Y ~ Z | D,X).
The auxiliary u expresses the conditional association induced by treatment
adjustment. Simply multiplying the observed A/B by kd/ky does not give these
formal bounds. Multipliers implying d_bound, u or y_bound at or above one
are refused with an explicit outside-support message. In particular an outcome
bound at one is uninformative for this implementation; it is not silently capped
and passed to a degenerate standard-error calculation.

For original treatment estimate b, SE s and df ν, the implied plug-in absolute
bias bound is s*sqrt(ν*d_bound*y_bound/(1-d_bound)). The `bounds` table reports
this quantity and b±the bias bound. These endpoints do not include uncertainty
in the observed benchmark strengths and are not sampling confidence intervals.
No worst-case SE or CI is claimed by evaluating a strength-box corner.
The `benchmarks` table reports observed group strengths; the original
`coefficients` and `covariance` tables are included unchanged.

## Maximum-strength robustness

`q` is a strictly increasing grid of nonnegative proportional effect changes;
`alpha` is a strictly increasing grid in (0,1]. Both allow at most 64 values.
The reference estimate for a query is (1−q)*b, preserving the sign of b. q=1
asks about zero; q>1 allows a reference beyond zero. This is a sensitivity query
relative to the original estimate, not a data-adaptive confidence guarantee
for an independently fixed population parameter.

At alpha<1, the returned robustness value is the smallest ρ for which an
admissible omitted scalar confounder, with **both** partial-R² strengths ≤ρ,
can make the adjusted two-sided Student t test not reject that reference.
At alpha=1, it is the point-estimate tipping value: no zero-level confidence
interval is asserted. If the reference is already nonrejected after the
hypothetical additional regressor consumes one degree of freedom, ρ=0.

Let f=q*|b/s|/sqrt(ν), c=|t(alpha/2,ν−1)|/sqrt(ν−1), and h=f−c. For h>0 the
equal-strength diagonal solution is h/(0.5*hypot(h,2)+0.5*h). For c>0 and f>1/c,
the minimum over the strength box instead occurs at an interior outcome
coordinate:

```
ρ = (f²-c²)/(1+f²)
r2_treatment = ρ
r2_outcome = ρ/((1-ρ)*f²)
```

Otherwise both witness strengths equal the diagonal solution. The interior
branch matters because reducing residual outcome variance can increase
statistical significance; a diagonal crossing alone is not generally the
minimum maximum-strength threshold. See the [authors' robustness implementation](https://github.com/carloscinelli/sensemakr/blob/master/R/sensitivity_stats.R)
and [paper supplementA.4, equations43–48](https://carloscinelli.com/files/Cinelli%20and%20Hazlett%20-%20Making%20Sense%20of%20Sensitivity.pdf).

The `robustness` table preserves the query, point/significance kind, solution
regime, maximum strength, diagonal comparator, witness strengths and
reconstructed treatment estimate, SE, df, t statistic and p value relative to
the reference. The signed boundary reconstruction error is retained rather
than forcing an exact floating-point equality. Zero thresholds may already
be strictly inside the nonrejection region and need not have zero boundary
error. No covariance across hypothetical threshold queries is fabricated.

## Numerical and evidence limits

Classical homoskedastic OLS and one omitted scalar regressor remain the model
contract. The reconstructed t test consumes one additional df. Student t
coverage is exact under the usual independent homoskedastic normal-error
linear model and otherwise approximate; sensitivity parameters themselves
are not estimated confounding guarantees.

Numerical computation uses Torch float64 CPU and native QR, SPD solves and
Student t routines. Auxiliary treatment regressions and group matrices are
saved in standardized units with their coefficient vectors, complete covariance
or inverse matrices, df and scale information recoverable from the embedded
original artifact. Invalid support, underflow/overflow, positive bias or
endpoint shifts lost at the coefficient scale, unrepresentable query changes,
and thresholds rounded to one are refused. No silent clipping or partial grid
is performed. Generic weights and other devices are unsupported; a Dataset
cannot substitute for the required original result. No new missing-data option
is offered because the original listwise sample is preserved.

Work and workspace admission cover complete base integrity/state retention and
the complete requested calculation. The saved base artifact includes its
original sample hash and every table's dtype, with the shared resource
contract's exclusions for caller input, Python objects and private allocator/
BLAS overhead. Save/load restores the complete postestimation artifact without
fitting or optimizing again, including original provenance and all groups,
multipliers, assumptions, queries and witnesses.

Independent tests compare group strengths with NumPy regressions dropping one
or several controls; construct actual residualized Z satisfying the formal
benchmark ratios; solve the strength-box optimization independently; and
construct actual witness Z vectors whose augmented OLS estimates, SE and t
statistics meet the reported thresholds. Sample/missing alignment, full
original covariance, resource refusals, typed artifacts, LaTeX output and
private RNG/default-device behavior are also checked. This is bounded
method-level validation, not licensed-vendor or whole-product parity.
