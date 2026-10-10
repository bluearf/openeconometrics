# Scalar iid CLR confidence sets over the real line

MARKET-681 is a bounded scientific child of MARKET-153. `iv_clr_confidence_set`
implements Moreira's conditional likelihood-ratio test inversion for one
continuous endogenous regressor and a fixed full-rank instrument set. This
child does not close the robust, dependent or subvector inference parent.

```python
from openecon.econometrics.weakiv import (
    iv_clr_confidence_set, iv_clr_restore, iv_clr_test,
    CLRConfidenceSet,
)

result = iv_clr_confidence_set(
    frame, "outcome", "endogenous", ["control"],
    instruments=["instrument1", "instrument2"],
    confidence=.95, missing="raise",
)
result["intervals"]
saved = CLRConfidenceSet(payload=result.attrs["state"])
restored = iv_clr_restore(saved.model_dump_json())
null_test = iv_clr_test(restored, null=0.)
```

## Statistical contract

The structural coefficient is scalar. Outcome, endogenous variable, included
controls and excluded instruments are distinct numeric columns. The caller
declares instrument validity/exclusion and iid homoskedastic reduced-form
innovations with positive-definite joint covariance. Fixed excluded dimension
and the regularity/moment conditions of the cited source are required for
the feasible covariance branch. Observed rank checks do not establish these
statistical assumptions. There are no weights, robust/HAC/cluster covariance,
multiple endogenous regressors, subvector nuisance inversions or dependent
sampling extensions in this child.

The default joint reduced-form covariance Ω uses residuals from both responses
on all included controls and excluded instruments, divided by `n - rank(C) -
rank(Z)`. The off-diagonal covariance is retained. This feasible test has the
source's iid uniform weak-instrument asymptotic law; it is not claimed to have
finite-sample exact size with estimated Ω. `omega=[[...], [...]]` supplies known
Ω in original outcome/endogenous reporting units. Conditional CLR has an exact
finite Gaussian law under the caller-declared Gaussian reduced-form model
with that genuinely known covariance. A user-supplied estimate does not become
known covariance merely because it is passed to this argument.

After partialling included controls, let G be the projected 2 by 2 response
Gram matrix, A = Ω⁻¹ᐟ² G Ω⁻¹ᐟ², M its maximum eigenvalue, and Q_T(β) the
conditioning statistic. LR(β) = M - Q_T(β). The conditional probability is
evaluated by the Moreira/Andrews–Moreira–Stock angular integral. Mikusheva's
monotonicity result reduces whole-line inversion to a single threshold C,
where `P(LR > M-C | Q_T=C)=alpha`. We search the bounded dimensionless LR
threshold `ell=M-C` between chi-square-one and chi-square-k upper-tail critical
values. When M is at most the chi-square-k value, every scalar null is accepted
and the result is all real numbers. For one instrument the probability and
critical value reduce analytically to chi-square-one.

The remaining inequality is homogeneous quadratic in `(beta, 1)`. Its complete
solution is a bounded closed interval, two closed unbounded rays, or all real
numbers. No grid or coefficient search window is used. `None` stores an
infinite endpoint in portable state; the displayed `lower_unbounded` and
`upper_unbounded` fields make it explicit. Stable eigen-gap calculations avoid
subtracting nearly equal large concentration values. A numerically unresolved
tangent/linear/topology boundary raises `unresolved_topology`; an arbitrary
tail shape is never substituted. Exact exceptional boundary points are outside
the regular numerical admission domain.

Gauss-Legendre integration uses increasing orders 32 through `max_order`.
Small LR probabilities use bounded geometric panels in the complementary
angle to resolve the narrow integration layer near the likelihood optimum.
Successive differences are relative-tail convergence estimates. Monotone
root brackets and the corresponding endpoint brackets describe this numerical
calculation. They are **not rigorous interval-arithmetic integration bounds**.
Returned interval endpoints are approximate; a null arbitrarily close to an
endpoint can require tighter user-declared tolerances. The source theorem
establishes the global quadratic geometry; numerical diagnostics establish
the stated finite precision, not an exact arithmetic certificate.

## Source, precision and budgets

Only resident pandas-compatible DataFrames and native Torch float64 CPU
execution are admitted. Boolean, complex, categorical and object source
columns are rejected. Floating and integer dtypes, including pandas nullable
dtypes, are retained exactly. Dtypes wider than float64/int64 are refused.
Finite source entries must lie within ±1e12;
integer entries must be exactly representable in float64. Covariance and
endpoints must remain representable in original float64 units. These are
implementation/numerical boundaries, not statistical requirements.

There are at most 10,000 original rows, 12 included predictors and 12 excluded
instruments, with at least eight complete rows and two reduced-form residual
degrees of freedom. Columns have distinct nonempty names of at most 128 UTF-8
bytes; `_cons` is reserved. The full combined normalized design must have
smallest singular value greater than `1e-10*sqrt(n)`; the joint covariance
correlation minimum eigenvalue must exceed `1e-10`. These numerical rank
requirements explicitly refuse near-singular designs.

Original typed row labels are saved independently of physical positions:
Range, ordinary, datetime/timezone, timedelta, categorical, Period and nested
MultiIndex descriptors round-trip. Descriptor metadata is bounded to 2 MiB.
`missing="raise"` is the default; `missing="drop"` records an explicit common
complete-case sample from the selected columns and retains the original rows
and dtype/missing state. Infinity is invalid rather than treated as missing.

Conservative arithmetic work and named live-buffer plans are admitted before
source copying/tensor allocation. `max_work` defaults to two billion planned
operations; an early configurable workspace plan applies the normal task-local
`OPENECON_WORKSPACE_MB`/`use_workspace_budget` limit. These are buffer/work
estimates, not process-RSS guarantees. `max_order` is a power of two from 64 to
512, `max_iterations` is 8..128, probability tolerance is `1e-13..1e-7`, and
root tolerance is `1e-12..1e-5`, with probability tolerance at most one tenth
of root tolerance. Confidence is .5 through `1-1e-8`. Complete JSON is bounded
to 32 MiB and iterator-based metadata traversal is bounded before thaw/hash
or source reconstruction. Ambient Torch RNG, default dtype and device are
preserved; all numerical allocations specify CPU float64.

## Complete state and acceptance gates

`CLRConfidenceSet` stores complete source/specification/options/typed index,
physical sample, full original-unit Ω and Gram, normalized whitening/eigen
geometry, probability/root history, quadratic/interval topology and resource
accounting. `iv_clr_restore` reconstructs lossless source admission and every
mathematical result. Numeric replay uses relative/coordinate-normalized
precision without a dimensionful absolute floor, including tiny-unit
covariances. Hashes detect corruption; rehashing fabricated results does not
pass semantic replay. Typed live state produced with `model_copy` or
`model_construct` is revalidated by readers and serialization.

`iv_clr_test` saves the full required target and null/statistic/conditioning
value/probability/quadrature/decision. `CLRTestState` and
`iv_clr_test_restore` replay the entire target plus query calculation. Neither
confidence nor test state is a fitted IV model or a synthetic Wald result.

`scripts/verify_weakiv_clr_oracles.py` has no Torch/OpenEcon dependency. It uses
independent NumPy least squares and generalized eigenvalues plus SciPy
adaptive integration and Brent root inversion of the original equations.
Tests compare all whole-line topologies, one-instrument reduction, included
controls, nonsingular instrument basis, response reporting units, known Ω,
null probabilities and distant tails. Rehashed cache/target forgeries, source
dtype coercion, malformed dimensions, low budgets, missing alignment and typed
live bypasses are rejected. A deterministic Gaussian sampling check is a
sanity check, not additional proof of the source theorem.

The original-author Stata package is `st0033_2`/`condivreg`, directed by
[Moreira's official software page](https://sites.google.com/site/moreiramarceloj/statistical-packages).
The original [Mikusheva–Poi 2006 manuscript](https://economics.mit.edu/sites/default/files/publications/944.pdf)
prints housing-data CLR interval `[.0020, .0037]` and rounded test results. The
official Stata Journal package/code and Stata Press `hsng2.dta` endpoints
returned HTTP 403 when accessed in this environment. No original-author code
execution or full precision fixture is therefore claimed. Rounded paper tables
are a distinct open gate and cannot validate the complete numerical contract.
No original code/data archive is vendored or assumed licensed. A licensed
native Stata/vendor comparison remains a separate gate. Independent equation
acceptance does not change these unavailable/not-run receipts.

Primary derivation: [Mikusheva 2010, section 3.1](https://economics.mit.edu/sites/default/files/publications/thirdsubmission.pdf),
*Robust confidence sets in the presence of weak instruments*, Journal of
Econometrics 157, 236–247, DOI 10.1016/j.jeconom.2009.12.003.
