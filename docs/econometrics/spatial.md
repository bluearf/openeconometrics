# Cross-sectional spatial econometrics

`oe.spatial_weights`, `oe.moran`, `oe.sar`, `oe.sem`, `oe.sac`, `oe.sdm`, and
`oe.spatial_impacts` are lazy public APIs. The four estimators also dispatch from
`oe.fit(ModelSpec(...), data=table)` and persist as ordinary `ResultBundle`s.

```python
import openecon as oe

W = oe.spatial_weights(
    ["north", "south", "east", "west"],
    [("north", "south", 1), ("south", "east", 1),
     ("east", "west", 1), ("west", "north", 1)],
    normalization="row", diagonal="raise", isolates="raise",
)
# With more complete observations than regression + spatial + variance parameters:
fit = oe.sar(table, "income", ["education"], key="region_id", spatial_weights=W)
fit.to_latex("sar.tex")
oe.spatial_impacts(fit).to_latex(index=False)
test = oe.moran(table, "income", key="region_id", spatial_weights=W,
                null="randomization", permutations=999, seed=73)
```

Use a sufficiently sized weight domain for the estimator example; the four-unit
constructor illustrates the edge format. Source data is never modified.

## Weight and sample contract

W is immutable, keyed, nonnegative, float64 COO. Ordered edge `(i,j,v)` means
unit i receives v times unit j's value. Strings and integers preserve identity;
floats, booleans, duplicate/missing keys, unknown edge keys, duplicate ordered
edges, negative and nonfinite weights fail. No geographic neighbors or graph
symmetrization are inferred.

`normalization="row"` gives each nonzero row sum one; `"none"` preserves supplied
weights. The diagonal must be zero; `diagonal="drop"` explicitly removes supplied
self weights. `isolates="zero"` explicitly retains zero rows and their zero
spatial lags; the default rejects them. Versioned `W.to_payload()` records the
effective matrix and policies; restored payloads cannot silently change weights.
Key/edge indexing, sorting, COO copies, payload restoration/export, and keyed
alignment have workspace preflights before owned buffers are built. Sized inputs
are checked before traversal; unsized iterables use guarded growing capacities
rather than materializing an unbounded edge stream.

Full table keys must equal the W key domain before missing filtering. Row order
is irrelevant. `missing="drop"` induces the retained-key subgraph and re-normalizes
row-normalized weights; newly created isolates still obey the declared policy.
Missing identity is invalid even under drop. Results retain source row positions,
sample-aligned W, its SHA256, residual definitions and actual solver diagnostics.
Observation weights, categorical predictors and panel/time structures are rejected.
Dataset inputs fail explicitly without collecting or replacing existing routes.

## Likelihood, information and domain

Let A=I−ρW and B=I−λW₂. SAR sets B=I; SEM sets A=I and uses W₂=W;
SAC estimates both parameters, optionally with `error_weights=W2`; SDM sets B=I
and augments X with W times every numeric predictor. The constant is never
spatially lagged, including when W is unnormalized or has isolates.

With e=B(Ay−Xβ), the full Gaussian log likelihood is

\[
\ell=\log|A|+\log|B|-\frac{N}{2}[\log(2\pi)+\log\sigma^2]
             -\frac{e'e}{2\sigma^2}.
\]

The spatial search profiles β by QR GLS and σ²=e′e/N. A five-point grid per
spatial parameter initializes two scalar starts or five SAC starts. Native Newton
steps and PyTorch first/second derivatives find an interior stationary maximum.
Nonconvergence, a better evaluated grid point, near-boundary solutions, deficient
designs and non-positive-definite observed information fail explicitly. Multi-start
search records every attempt; it does not guarantee a unique global SAC maximum.

`nonrobust` covariance is the inverse **full joint observed information** in
[β,ρ/λ,ln(σ²)] reporting units. It includes nuisance variance and all cross blocks;
it is not the conditional GLS coefficient covariance. Inference is asymptotic
normal. Reported fitted values are the unconditional reduced-form mean A⁻¹Xβ;
innovations and structural residuals are separately persisted.

The supported interval is `abs(parameter)<(1−1e−6)/max_abs_row_sum(W)` for each
filter. This sufficient Neumann stability domain applies to asymmetric matrices
and may be narrower than a reciprocal eigenvalue domain. Exact spectral radii are
recorded. Determinants use dense LU `slogdet`; negative/singular filter determinants
fail. `max_n=512` is the default explicit dense observation ceiling, adjustable
up to 2048 subject to the workspace budget. Plans reserve dense filters,
factorizations, derivatives, information and multipliers before allocation. These
are buffer estimates, not RSS measurements or large-data/GPU promises.

## Impacts

For numeric predictor r, Sᵣ=A⁻¹(βᵣI+θᵣW), with θᵣ=0 for SAR/SAC. Direct impact
is trace(Sᵣ)/N, total is sum(Sᵣ)/N, and indirect is total−direct. Exact matrix
multipliers include feedback; a raw coefficient is not the SAR direct effect.
The derivative of A⁻¹ with respect to ρ is A⁻¹WA⁻¹. Full-covariance delta-method
SEs, normal confidence intervals, gradients and direct/indirect/total covariance
blocks are persisted. W is conditioned on as fixed/exogenous. Intercept impacts
and factor-variable slopes are outside this contract.

## Moran's I

I=(N/S₀)(z′Wz)/(z′z), z=y−mean(y). Under both nulls E[I]=−1/(N−1).
S₁=½Σᵢⱼ(wᵢⱼ+wⱼᵢ)²; S₂=Σᵢ(row_sumᵢ+column_sumᵢ)²; b₂=NΣzᵢ⁴/(Σzᵢ²)².
The normality variance is

\[
V_N=\frac{N^2S_1-NS_2+3S_0^2}{(N^2-1)S_0^2}-E[I]^2.
\]

The randomization variance is

\[
V_R=\frac{N[(N^2-3N+3)S_1-NS_2+3S_0^2]
 -b_2[(N^2-N)S_1-2NS_2+6S_0^2]}{(N-1)(N-2)(N-3)S_0^2}-E[I]^2.
\]

At least four observations, positive nondegenerate null variance and nonconstant
values are required. Explicit zero-row isolates count in N and remain exchangeable;
there is no `adjust.n` reduction. This is a global raw-variable statistic, not a
regression-residual Moran test or a network statistic. `p_value` is the analytic
normal approximation for the selected null. Optional `permutation.p_value` comes
from independently seeded uniform reassignments of all observed values to fixed
keys. Greater/less tails include ties; two-sided extremeness is |I−E[I]|. The
Monte Carlo probability is (extreme+1)/(B+1), never zero. Permutations are capped
at 100000 and `B*(N+nnz)` must fit `max_permutation_work` (default 50 million).
Global RNG state is preserved. Outcome scaling bounds fourth moments; total
weight outside [1e−150,1e150] fails with an explicit rescaling error.

## Independent evidence and limits

`tests/test_spatial_models.py` compares complete likelihood to an independently
constructed reduced-form multivariate Gaussian covariance density in NumPy,
finite-difference gradients/Hessians and inverse information. Development-only
Brent/differential-evolution optimizers check spatial parameters. SAR/SDM impacts
are checked by perturbing each unit's actual regressor and differentiating the
resulting reduced-form mean; parameter perturbations check delta uncertainty.
The directed-cycle determinant also matches the independent polynomial 1−(cρ)ᴺ.
`tests/test_spatial_weights_moran.py` enumerates all 5!=120 assignments, including
asymmetric W and isolates, to validate randomization expectation/variance; NumPy
tail calculations check seed, ties, plus-one probabilities and RNG preservation.

Run `scripts/validate_spatial_market124.py` with this checkout on PYTHONPATH to
regenerate [the receipt](../evidence/market-124-spatial/oracle.json) and coefficient
and impact LaTeX exports. This is source-level CPU evidence, not Stata parity,
frozen-desktop, deployment, saved-project UI or public-release verification.

## Primary implementation references

* [Bivand & Piras (2015), Comparing Implementations of Estimation Methods for Spatial Econometrics](https://www.jstatsoft.org/article/view/v063i18)
  and the authors' [spatialreg ML documentation](https://r-spatial.github.io/spatialreg/reference/ML_models.html)
  specify SAR, SEM and two-weight SAC and discuss SAC optimization.
* [The authors' spatialreg impacts documentation](https://r-spatial.github.io/spatialreg/reference/impacts.html)
  identifies the exact LeSage–Pace matrix averages and coefficient uncertainty.
* [Bivand's spdep Moran documentation](https://r-spatial.github.io/spdep/reference/moran.test.html)
  and [author-maintained moment implementation](https://github.com/r-spatial/spdep/blob/main/R/moran.R)
  give normality/randomization moment assumptions. Our isolate contract corresponds
  to retaining full N rather than the package's optional `adjust.n` reduction.
