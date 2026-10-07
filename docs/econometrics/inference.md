# Coefficient tests: `test`, `testparm`, `lincom`, `nlcom`

These public functions work with an ordinary persisted `ResultBundle`, including
registry models and `suest` systems. They use the reported coefficients and full
covariance matrix without refitting or retaining the estimation dataset. Fitted
OLS results continue to call their existing methods, retaining their anchored
design, streamed contrasts, and contrast-specific degrees-of-freedom or Hansen
adjustments.

```python
import openecon as oe

# df contains a nonnegative integer count y and numeric regressors x and z.
model = oe.poisson(data=df, y="y", x=["x", "z"], covariance="robust")
oe.test(model, "x")                            # x = 0
oe.test(model, {"x": 1, "z": -1}, value=.2)   # x - z = .2
oe.test(model, [{"x": 1}, {"z": 1}], [0, .1]) # two simultaneous restrictions
oe.testparm(model, ["x", "z"])                # joint zero test
oe.lincom(model, {"x": 2, "z": -1}, constant=.1)
oe.nlcom(model, lambda b: b["x"] / b["z"], null=1)
```

`test` accepts a coefficient name, a sequence of names, a mapping of names to
weights, a sequence of mappings, or a numeric restriction vector/matrix. Numeric
columns follow the order of `model.coefficients`. The null value can be a finite
scalar shared by all rows or a vector with one value per row. The hypothesis is
`R b = r` and the Wald quadratic form is
`(R b - r)' (R V R')^-1 (R b - r)`.

`testparm` accepts exact names, case-sensitive wildcard patterns, and factor
names such as `group` for `group[B]` and `group[C]`. Overlapping patterns test
each coefficient once, in reporting order.

`lincom` reports the estimate, standard error, statistic, p-value, confidence
interval and gradient for a named linear combination plus a constant. Its null
is zero. `nlcom` reports the same quantities for a scalar nonlinear function
and a supplied null. It takes a Python callable that receives named Torch
scalars, with automatic differentiation confined to the small parameter vector.
For example, `lambda b: torch.exp(b["x"])` works after `import torch`. It does
not evaluate expression strings. Undefined expressions, detached derivatives,
complex outputs and nonfinite derivatives are rejected. The delta-method
variance is `gradient' V gradient`.

## Inference and equations

A result recording `use_t=True` uses its positive finite `df_inference` (or
`df_resid` when no inference-DF field exists). Linear/nonlinear combinations use
Student t; a joint Wald test with `q` independent rows uses `F(q, df)` and the
Wald quadratic form divided by `q`. Normal coefficient inference uses z and
chi-squared with `q` degrees of freedom. The family name never substitutes for
the fitted inference record. Confidence intervals use the recorded alpha,
falling back to the specification alpha when it is absent.

Use the exact term names shown in `model.coefficients`. For equations, names
such as `first:x` and `second:x` allow cross-equation tests:

```python
system = oe.sureg(data=df, equations=[
    {"y": "y1", "x": ["x"], "name": "first"},
    {"y": "y2", "x": ["x", "z"], "name": "second"},
])
oe.test(system, {"first:x": 1, "second:x": -1})
oe.lincom(system, {"first:x": 1, "second:x": -1})
oe.testparm(system, "second:*")
```

Equation-qualified aliases based on `Coefficient.equation`, including
`[first]x`, also resolve. A bare local name is accepted when it identifies one
coefficient; otherwise an individual contrast raises `ambiguous_term` and needs
an equation. `testparm("x")` can select that local term across equations for a
joint zero test. The full covariance, including cross-equation entries, always
drives inference.

The four operations also work after JSON round trips through
`ResultBundle.model_validate_json(model.model_dump_json())`. They neither modify
the result nor require the original observations. A plain saved bundle carrying
OLS-specific contrast adjustments is refused: those adjustments need the
existing OLS result methods and retained contrast state.

## Validation and boundaries

Malformed coefficient/covariance shapes, duplicate names, nonfinite inputs,
asymmetric or indefinite covariance matrices and inconsistent inference records
produce structured `AnalysisError` codes. Restriction rows must be independent;
dependent rows are rejected rather than silently reducing the reported test
degrees of freedom. A singular overall covariance is allowed when the tested
restriction has positive definite sampling covariance. Zero-variance contrasts
are not estimable and are rejected.

Independent tests compare restriction algebra and delta-method gradients with
NumPy, distribution tails with SciPy, and Poisson Wald/linear tests with
statsmodels. Actual SUR results are tested after serialization, and fitted OLS
dispatch and contrast adjustments have regression checks. These checks do not
constitute a comparison with a licensed Stata installation; they do not change
`stata_parity_validated=False`.

This is coefficient inference only. Generic prediction, marginal effects,
model-specific diagnostic tests, nonlinear joint systems of restrictions,
survey corrections and reconstruction of unavailable fitted OLS contrast state
are separate capabilities.
