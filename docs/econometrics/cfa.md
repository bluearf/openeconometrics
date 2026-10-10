# Continuous confirmatory factor analysis

`cfa(data, factors={"ability": ["x1", "x2", "x3"]})` fits continuous
Gaussian covariance ML. `cfa_covariance` accepts a labeled covariance DataFrame
or a square numeric sequence, explicit sample size and explicit divisor `n` or
`n-1`. Indicator order is first occurrence in factor declarations. This is a
multi-outcome helper, with a separately typed `CFAState`; the existing `sem`
command retains its spatial-error meaning.

The model is `Sigma = Lambda Phi Lambda' + diag(theta)`. Undeclared loadings are
zero. Marker identification fixes one declared loading per factor at **one in
the original indicator units**. `markers` selects a different marker per factor.
`identification="unit_variance"` instead fixes latent variances to one and admits
all declared loadings, with positive first-indicator orientation. A parameter
count check is followed by a covariance-moment Jacobian rank check. A nominal
positive degree of freedom alone does not prove identification.

Observed means are unrestricted and profiled at sample means. The Gaussian
log likelihood retains `n*p*log(2*pi)/2`, the original-unit determinant and the
normal-ML sample covariance with divisor **n**. A supplied unbiased covariance
is explicitly converted by `(n-1)/n`. Summary input without means establishes
covariance/likelihood inference only; it does not invent individual scores or a
latent mean structure. These conventions match the normal-ML description in
the [lavaan estimation documentation](https://lavaan.ugent.be/tutorial/est.html).

Optimization uses CPU float64 Torch, standardized indicators, two deterministic
declared starts and bounded LBFGS/Newton polish. Every start and failure is
retained. Both score convergence and stable positive observed curvature are
required. Joint covariance is the inverse **full observed information**, mapped
back to all free original-unit loadings, latent covariances and residual
variances through a complete Jacobian. Cross terms are retained. Standard errors,
normal Wald tests and marginal intervals apply to a regular interior iid
Gaussian model. The [lavaan CFA tutorial](https://lavaan.ugent.be/tutorial/cfa.html)
reports expected-information uncertainty by default, so its rounded standard
errors are not substituted for this observed-information contract.

The source tests compare likelihood, every free parameter and full covariance
against an independent NumPy/SciPy natural-parameter optimizer and finite
differences of its observed score. Tests also check exact population moments,
raw/summary divisor equivalence, original-unit changes, marker-to-unit-variance
joint covariance transformation, underidentified models, Heywood boundaries,
nonconvergence, missing/index identity, forged cache/digest rejection and ambient
Torch state. This is independent numerical evidence, not licensed Stata parity.

`CFAState(payload=result.attrs["cfa_state"])` provides immutable saved state.
`cfa_restore(result_or_state_or_json)` replays the source/sample moments, model,
score, observed information, full covariance and likelihood **without optimizing**.
Recomputing a checksum does not bypass semantic checks. State records raw rows,
physical retained positions, original numeric dtypes, typed indexes, exact
parameter order, identification, full solver/natural state and budget plan.

Resident limits are 10,000 original rows, 16 indicators, four factors and 80 free
parameters. The workspace plan and declared scalar-work limit are checked before
copying resident data or allocating numerical buffers; replay checks them again.
These are implementation/resource boundaries. No global dtype, device, RNG or
thread setting is changed. Missing values either fail or undergo explicitly
declared complete-case deletion. Infinities, nonnumeric data, a singular sample
covariance, weak/rank-deficient identification, nonconvergence and boundary
variance fits fail explicitly; there is no PSD repair or expected-information
fallback.

This first stage does not complete MARKET-360. Latent paths/means, residual
correlations, fit indices and scores, multigroup/MAR FIML/robust uncertainty and
generalized/multilevel outcomes remain in the authorized program.
