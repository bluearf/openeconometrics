# Saved smoothing derivatives, margins and contrasts

Eight helpers extend the existing CPU float64 smoothing fits. They use the
saved training transformations; they never estimate knots, spans, bandwidths,
centers or penalties from query data. Fits and `smoothing_predict` retain their
existing contracts. See [the executable example](../examples/smoothing_postestimation.py).

| Helper | Supported target and limits |
| --- | --- |
| `spline_derivative` | B-spline/RCS first or second pure partial in original predictor units. Order cannot exceed B-spline degree. Discontinuous derivative knots return an error; saved endpoints use the interior one-sided limit. RCS has linear tails, hence zero second derivative outside its outer knots. |
| `fp_derivative` | FP1/FP2 and selected single-variable MFP; explicit scale chain rule, log and repeated-power terms, retained linear adjustments and zero derivatives for excluded terms. Positive FP domain required. |
| `gam_derivative` | First/second partial of the saved centered cubic additive basis; differentiation removes the fixed center while all coefficient covariance is retained. |
| `mars_derivative` | First partial of retained hinge products, including interactions. Active exact knot ties are rejected; no subgradient is selected. A zero multiplier from another fixed factor makes the partial zero. Piecewise linear extrapolation is retained. |
| `loess_derivative` | Local Taylor polynomial derivative, `r! beta_r / radius^r`, using the saved tricube nearest-neighbor span and degree. This estimates a local regression slope/curvature; it does **not** differentiate the moving-neighborhood prediction algorithm. Second derivatives require degree two; queries remain within training range and pass the local rank gate. |
| `kernel_derivative` | First/second continuous query partial of the normalized mixed product-kernel mean, using analytic Gaussian score/quotient derivatives. Categories retain saved Aitchison–Aitken/Wang–van Ryzin factors. Categorical derivatives are rejected; use explicit profile contrasts. Numeric ranges, category universes and effective sample gates remain enforced. |
| `smoothing_margins` | Average response (`variable=None`) or first partial over an explicit fixed query population. Optional finite nonnegative `averaging_weights` match original query rows; retained weights are normalized after missing exclusion. LOESS averages use the same local Taylor target. |
| `smoothing_contrast` | Response at `data` minus `reference`, paired by original row position, with equal row counts and joint missing exclusion. Returns row changes or `average=True` with optional averaging weights. Mixed-kernel categorical counterfactual changes are supported. |

Tables record positional rows, retained sample, source result ID, state digest,
target and inference notes. Six derivative helpers require `variable`; order is
one or two except MARS, which supports one. `missing='raise'` is the default;
`missing='drop'` records original positions. Duplicate pandas indices do not
change pairing. Dataset collection, GPU, fit weights, robust VCE and unknown
options are unsupported. Averaging weights define a fixed postestimation
target; they do not change the training fit or its sampling design.

## Conditional uncertainty

Spline/FP/GAM derivative helpers default to `interval=True`. Other helpers
default to `interval=False`. Spline/FP/MFP intervals use full saved iid Gaussian
covariance for conditional t SE, residual df, zero-null statistic, p-value and
CI. They condition on the selected model and fixed query covariates; model,
knot and power selection uncertainty is excluded. Deterministic zero contrasts
have SE/CI zero and undefined statistic/p-value rather than invented significance.

GAM uses saved frequentist covariance `sigma² A^-1 X'X A^-T`, with
`A=X'X+lambda P`. Normal intervals approximate fluctuations around the
**penalized estimator expectation**, excluding smoothing bias and selected
penalty uncertainty. No p-values or Bayesian interpretation are supplied.
LOESS, MARS and mixed kernels reject `interval=True`; sampling intervals are
unavailable for these stages.

Average and paired targets first form the coefficient-order vector `L`, then
compute `L V L'`. Off-diagonal coefficient covariance and cross-profile
covariance remain included; averaging SEs or adding independent profile
variances is incorrect. Inferential table attributes retain
`linear_functionals` and `coefficient_covariance`: an exact factorized joint
row covariance representation avoiding quadratic query-row allocation. The
covariance passes symmetry/PSD checks. Cubic covariance audit work, query
bases, paired selections and saved training buffers are included in explicit
work/workspace planning before numerical allocations.

## Verification and sources

Development-only SciPy `BSpline.derivative` and an independently fitted natural
`CubicSpline` space verify both derivative orders and boundaries. NumPy
complex-step/power/log formulas verify all 36 FP2 power pairs, scales and
repeated logs. Unscaled local Taylor weighted least squares and polynomial
reproduction verify LOESS in original units. Kernel and retained MARS
interaction derivatives are checked against finite differences away from
knots. Full covariance, t/p/CI, average targets, paired contrasts, JSON replay,
missing alignment, resources, unsupported options and input preservation are
tested. SciPy is not a production/runtime dependency.

Primary descriptions and development references:

- [SciPy B-spline derivatives](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.derivative.html).
- [NIST LOESS and tricube/local polynomial construction](https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd144.htm).
- [R stats local polynomial regression](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/loess.html).
- [Existing smoothing definitions and sources](smoothing-stages.md) for FP, MARS, categorical kernels and Gaussian covariance. Local Taylor differentiation and normalized-kernel quotient rules here are explicit mathematical constructions, not vendor postestimation equivalence claims.

Source tests, frozen equivalence and installed native Run/restart are separate
proof layers. NonGaussian GAM, general multivariable MFP and broader
nonparametric/selection uncertainty remain open. No licensed vendor comparison
or blanket Stata parity is claimed.
