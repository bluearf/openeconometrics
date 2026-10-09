# Sharp empirical bounds for individual treatment effects

`treatment_effect_cdf_bounds` and `treatment_effect_quantile_bounds` find the smallest and largest possible probabilities or quantiles of the **individual** effect `Y(1) - Y(0)` consistent with two supplied empirical arm distributions. Each endpoint comes with a complete attaining joint distribution and an exact maximum-flow/minimum-cut certificate.

These are identification calculations conditional on the empirical marginal PMFs. They provide no population confidence interval, standard error, covariance, or hypothesis test. Declared binary randomization, consistency and no interference motivate using the observed arm distributions as the marginal input laws; the functions do not claim that a finite trial reveals the unknown population marginals exactly. They impose no rank invariance, independence, or other dependence restriction on the two potential outcomes.

The individual-effect quantile differs from the difference of the two marginal outcome quantiles. For example, if both arms have the same empirical distribution with half their mass at 0 and half at 1, every marginal quantile difference is 0, but the sharp median individual-effect bounds are `[-1, 0]`.

## Calls and supported inputs

```python
import pandas as pd
import openecon as oe

trial = pd.DataFrame({"y": [0, 1, 2, 0, 1, 2], "d": [0, 0, 0, 1, 1, 1]})
cdf = oe.treatment_effect_cdf_bounds(
    trial, "y", "d", support=[0, 1, 2], thresholds=[-1, 0, 1],
    design="randomized",
)
quantiles = oe.treatment_effect_quantile_bounds(
    trial, "y", "d", support=[0, 1, 2], quantiles=[0.25, 0.5, 0.75],
    design="randomized",
)
print(cdf["bounds"])
print(quantiles["bounds"])
```

Both signatures require keyword-only `support` and `design`. The CDF call additionally requires `thresholds`; the quantile call requires `quantiles`. The remaining options are `missing="raise"`, `device="cpu"`, `weights=None`, and `max_work=100_000_000`.

The finite support must be prespecified independently of the realized outcomes: 1–8 distinct finite numeric values, all with absolute value at most `1e150`. Every original observed outcome must equal one of those represented values. Unobserved support cells are retained with zero marginal mass. Treatment must contain both numeric 0 and 1; Boolean labels are refused. A single observation in an arm is sufficient for this empirical identification calculation. Outcome and treatment roles must be distinct.

All pairwise treated-minus-control support differences must be **exactly representable** in binary64. A native Torch TwoSum calculation certifies zero rounding residual before copying the sample or constructing any flow. For example, support `[1, 10**16]` is refused because the true difference between these represented values is not exactly representable. Grid scalar inputs, including integers or wider-precision scalars, that would change during float64 conversion are also refused. Floating outcome dtypes wider than 8 bytes are unsupported, and integer outcomes that would change during conversion are refused. Dyadic supports are convenient; decimal-looking supports can fail this exactness requirement. CPU binary64 round-to-nearest arithmetic with gradual underflow is required, and a subnormal arithmetic probe refuses flush-to-zero execution. The magnitude guard prevents intermediate overflow.

`thresholds` is a fixed list of 1–64 distinct finite represented values; thresholds need not lie on the effect support. CDF targets use the inclusive convention `P(Y(1)-Y(0) <= threshold)`. `quantiles` is a fixed list of 1–16 distinct represented values strictly inside `(0,1)`. Targets and outcome support keep the supplied order. Quantile calculations also retain the complete, sorted difference grid, including points with zero mass in an attaining witness.

Only complete resident numeric tables on CPU are supported. `Dataset`, generic weights, other designs, GPU devices, missing selected inputs and `missing="drop"` are refused. Original row positions and typed index labels are retained, including duplicates and distinct integer/string/Boolean labels. Each original index or selected column label is limited to 256 bytes of its escaped typed JSON encoding, rather than 256 characters. There are at most 100,000 original rows, and `n_control * n_treated` must not exceed `10**10`.

## Exact transportation calculation

Let the observed arm counts at support point `i` be `c0[i]` and `c1[i]`, and let arm sizes be `n0` and `n1`. Set

```text
M = n0 * n1
row_mass[i] = c0[i] * n1
column_mass[j] = c1[j] * n0
Delta[i,j] = support[j] - support[i]
```

A coupling is a nonnegative matrix `C` with these row and column sums; probability mass is `C / M`. For a threshold `z`, the objective is the mass on edges for which `Delta[i,j] <= z`.

For the upper CDF endpoint, a source connects to control support cells with `row_mass` capacity, permitted control-to-treated edges have capacity `M`, and treated cells connect to the sink with `column_mass` capacity. The maximum flow gives the largest permitted mass. The lower CDF endpoint is `M` minus the maximum flow on the complementary edges `Delta > z`. After each flow, residual marginal mass is completed into a full coupling. No permitted mass can be added to an optimal upper flow during this completion; the saved coupling objective is independently checked against the flow value.

All capacities, flows, residuals, couplings, objectives and cut capacities use native Torch CPU `int64`. Integral capacities and the integral transportation polytope mean that these witnesses also attain the extrema over **all real-valued** couplings of the two empirical PMFs; restricting the solver to integer counts does not weaken the bounds. Every saved flow includes its exact residual-reachable source cut. Feasible flow conservation and equality between flow value and cut capacity certify optimality without trusting or rerunning the optimizer.

The implementation uses deterministic Edmonds–Karp augmenting paths. It refuses the entire calculation if the complete admitted iteration/work plan is exceeded; it never clips capacities or returns a truncated set of targets.

Each CDF endpoint is pointwise sharp. Different thresholds generally require different couplings. The lower or upper envelope is not asserted to be one simultaneously attainable whole CDF.

## Quantile witnesses and exact probability crossings

For the represented float64 probability `q`, the left quantile is the smallest difference-grid value where a coupling's CDF is at least `q`. The required integer count is computed as

```text
q = numerator / denominator       # exact float.as_integer_ratio()
k = ceil(M * numerator / denominator)
```

This uses scalar integer arithmetic, avoiding an off-by-one error from rounding `M*q`. The complete rational numerator and denominator are stored as decimal strings, which also preserves subnormal positive probabilities whose rational denominator is larger than the finite float64 range.

Write `L(z)` and `U(z)` for the exact lower and upper CDF mass counts. The lower quantile endpoint is the first grid value with `U(z) >= k`; its attaining witness is the CDF maximizer **at that value**. The upper endpoint is the first grid value with `L(z) >= k`; its attaining witness is the CDF minimizer **at the preceding grid value**, where `L < k`. Minimizing the CDF at the upper endpoint itself need not attain the upper quantile. At the first grid value, an empirical product coupling suffices because every coupling has zero probability before the minimum effect.

The saved witnesses include the complete effect PMF and CDF counts. Their predecessor and current crossings are checked directly against `k`. Quantile endpoints are separately sharp for each supplied probability; one coupling is not asserted to attain an entire quantile-bound vector.

## Results, persistence and resource admission

Both results contain `bounds`, `marginals`, `couplings`, and `subjects`. Quantile results additionally contain `witness_distributions`. The tables retain exact count columns and a common denominator alongside float64 probabilities. State retains all original selected values, assignments, support ordering, typed sample provenance, marginal counts, every optimized threshold, full network capacities/flows/residuals, source-cut memberships, complete coupling matrices, objectives and iteration counts. Quantile state adds exact rational probabilities, complete difference-grid extrema, endpoint indices, predecessor indices, witness PMFs and exact crossings.

`causal_design_save` and `causal_design_load` preserve the full tables, dtypes, indices, order and scientific state with checksums. Their semantic dispatch additionally validates all flow/coupling/min-cut certificates, regenerated typed tables and quantile witnesses. Loading verifies certificates; it does not solve a new transport problem. A checksum that was recomputed after a contradictory scientific edit does not bypass these checks.

The complete sample, all flows, certificate checks, witness tables and portable copies are admitted before their main allocations. With `K` support points, `V=2K+2` and `E=K*K+2K`, each flow is charged for the conservative `V*E` augmenting-path bound and dense residual searches. The CDF work plan includes both extrema for every supplied threshold. The quantile plan conservatively includes both extrema for all `K*K` pairwise difference points, even if repeated differences later reduce the grid. Both plans also charge original escaped labels and full witness output. Workspace estimates include named source/provenance, subject copies, graph/certificate copies, typed tables and buffers; they are not a process-RSS guarantee. The default work budget can refuse an 8-point full quantile grid even though the scientific domain permits it; callers can explicitly increase `max_work` and the workspace budget. No partial artifact is returned on refusal.

## Independent validation and references

Tests enumerate every integer transportation table for small marginals and compare every CDF boundary with independent SciPy linear programming over continuous coupling probabilities. Quantile tests enumerate all feasible couplings, include `nextafter` probabilities on both sides of ties and at 0/1 boundaries, and verify every claimed endpoint is attained. Additional tests compare TwoSum admission with exact `Fraction` differences, exercise zero support cells and unequal arm sizes, extreme translations/scales, complete integer cut certificates, resource preflights, typed-label retention, unchanged global RNG/device settings, exact full save/load and rejection of rehashed semantic tampering. SciPy linear programming and `Fraction` provide independent test oracles. Numerical optimization and count kernels use native Torch and do not delegate to NumPy or SciPy. Pandas sample and table plumbing still uses NumPy. No external optimizer is called at runtime.

The distinction between marginal quantile effects and quantiles of individual effects, pointwise sharp CDF bounds and inversion into quantile bounds follows [Fan and Park (2010), *Sharp bounds on the distribution of treatment effects and their statistical inference*](https://www.treatment-effects.com/Fan-Park-2010.pdf), particularly Lemmas 2.1 and 2.3. This implementation deliberately conditions on finite empirical marginal laws and does not implement that paper's population sampling inference. Integral-capacity flow and the primal/dual maximum-flow–minimum-cut proof are covered in [MIT 6.854 maximum-flow notes](https://courses.csail.mit.edu/6.854/19/Scribe/s6-maxflow/s6-maxflow.html). Validation is limited to the two declared finite-support APIs; no blanket Stata parity or general partial-identification inference is claimed.
