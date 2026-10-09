# Linear omitted-variable sensitivity

`oe.ovb_sensitivity(data, y, treatment, controls, ...)` fits classical
homoskedastic OLS with an intercept and the explicitly listed controls.
The treatment column can be a continuous regressor or numeric 0/1. Every role
must be numeric; there is no implicit category encoding. Use `controls=[]`
for an intercept and treatment alone. Outcome and regressor names must be
distinct, and `_cons` is reserved for the intercept.

The function evaluates a hypothetical **single scalar omitted confounder** Z.
`r2_treatment` specifies its partial R² with treatment conditional on controls,
R²(D ~ Z | X). `r2_outcome` specifies its partial R² with the outcome conditional
on treatment and controls, R²(Y ~ Z | D, X). Supply strictly increasing grids
within [0,1), with at most 64 values each. Every Cartesian grid cell is retained.
These strengths are sensitivity assumptions; the procedure does not estimate
an unobserved confounder or establish causal identification.

For original treatment estimate b, classical standard error s and residual
degrees of freedom ν, the omitted-variable bias magnitude and augmented
standard error are

```
B = s * sqrt(ν * r2_outcome * r2_treatment / (1 - r2_treatment))
s_adjusted = s * sqrt((1 - r2_outcome) / (1 - r2_treatment) * ν / (ν - 1))
```

These are the exact regression identities in
[Cinelli and Hazlett (2020), equations 12 and 14](https://doi.org/10.1111/rssb.12348),
also available in the [authors' manuscript](https://carloscinelli.com/files/Cinelli%20and%20Hazlett%20-%20Making%20Sense%20of%20Sensitivity.pdf).
They apply to classical OLS inference with one additional regressor. At least
two original residual degrees of freedom and positive residual variance are
required. Adjusted t tests and two-sided confidence intervals use ν−1 degrees
of freedom. Their finite-sample coverage is exact under the usual linear model
with independent homoskedastic normal errors; otherwise Student t inference is
an approximation. Robust, clustered, survey or weighted standard errors are outside
this implementation's scope.

`direction="reduce"` returns b−sign(b)B; `"increase"` returns b+sign(b)B;
`"both"` returns both. Reduction can cross zero. An exactly zero original
coefficient uses the positive orientation, producing −B and +B. Zero strength
can eliminate bias while the hypothetical extra regressor still consumes a
degree of freedom; at both strengths zero the adjusted standard error is
s*sqrt(ν/(ν−1)).

The `robustness` table reports the equal-strength **point-estimate** tipping
value ρ that satisfies B(ρ,ρ)=|b|. With f=|b/s|/sqrt(ν),

```
ρ = f / (0.5 * hypot(f, 2) + 0.5 * f)
```

This is the stable form of the authors' robustness value for reducing the
point estimate to zero. It is not a significance tipping value, and does not
claim that confounding below ρ is absent. The original coefficient's partial
R² with the outcome is separately saved. At a zero point estimate the
point-estimate robustness value is zero. The reconstructed tipping estimate
is retained with its actual floating-point residual rather than forced to zero.

The result contains four tables: `coefficients`, `covariance`, `sensitivity`
and `robustness`. The first two preserve all original coefficients, their
classical inference and the complete covariance matrix in original units.
The sensitivity table preserves each adjusted treatment estimate, variance,
standard error, t statistic, degrees of freedom, p value and confidence limits.
The partial-R² inputs identify the adjusted treatment variance but do not
identify covariance with all other augmented coefficients or a joint covariance
across hypothetical scenarios; those are not fabricated.

An example with an explicit pair of strength grids is:

```python
result = oe.ovb_sensitivity(
    data, "outcome", "treatment", ["age", "baseline"],
    r2_treatment=[0.0, 0.1, 0.3],
    r2_outcome=[0.0, 0.1, 0.3],
    direction="both", level=0.95,
)
result["sensitivity"]
oe.causal_design_save(result, "ovb.json")
restored = oe.causal_design_load("ovb.json")
```

`missing="raise"` is the default. Explicit `"drop"` removes rows missing any
selected outcome or regressor and retains original positions and labels;
missing values in unrelated columns have no effect. Full selected inputs,
centering/scaling transforms, original and scaled model state, complete
covariance, both-direction grid results, assumptions and checksums survive
`causal_design_save/load` without refitting. Duplicate row labels are supported.

The model uses Torch float64 CPU and native Householder QR and Student t
functions. Regressors are centered and scaled before factorization, and results
are transformed back to their original units. Rank deficiency, zero residual
variance, variance indistinguishable from QR rounding, nonfinite inputs and
unrepresentable numerical results are refused. A nonzero bias adjustment or
positive confidence radius that disappears when added to the coefficient,
and a robustness threshold that rounds to the excluded endpoint one, are also
refused rather than reported as no change or a zero-width interval.
Input magnitudes above 1e150
must be rescaled under the shared numeric contract. There is no SciPy runtime
route or global random-generator mutation. Dataset collection, generic
observation weights and devices other than CPU are unsupported.

Work and workspace limits admit the complete model and grid before QR or grid
allocation; no partial grid is returned. The workspace plan accounts for named
input, tensor and result buffers, with the common resource contract's exclusions
for caller input, Python objects, allocator overhead and private BLAS workspace.
Independent validation constructs an actual omitted scalar confounder and
compares its full augmented OLS fit and FWL partial R² values, including the
adjusted estimate, standard error and Student t interval. It also verifies the
original full covariance, numerical scaling, minimum degrees of freedom,
resource refusals and exact typed artifact restoration. This is bounded
method-level coverage, not whole-product or vendor parity.
