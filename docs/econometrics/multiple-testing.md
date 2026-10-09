# Multiple testing and simultaneous normal inference

[Finite-sample distribution-free families](distribution-free-joint.md) add
declared sign/assignment randomization and conservative CDF, quantile,
proportion, multinomial and bounded-mean regions with separate assumptions.

Separate [Gaussian finite-sample procedures](finite-sample-joint.md) support
declared classical OLS/WLS joint-null generation, known-shape/common-scale
joint-t intervals and an iid normal-mean Hotelling region. The three original
summary APIs documented below retain their contracts.

The public procedures consume a caller-declared, compatible hypothesis family.
They do not fit models, choose a family from unrelated tables, or reconstruct
observations, weights, categories, degrees of freedom or a null-generating design.
The numerical engine uses resident CPU float64 Torch; no SciPy, statsmodels or
NumPy numerical routine is called by the runtime implementation. Discovery of
the three auxiliary exports remains Torch-free and leaves estimator/Dataset
counts unchanged.

`oe.multipletests(p_values, method="holm", alpha=.05, labels=None,
missing="raise")` preserves the supplied hypothesis order. The complete input
sequence declares the family, and unique nonmissing scalar labels can name its
members. Booleans, numeric strings, complex numbers, infinities, unbounded
iterators and multidimensional vectors are rejected. Missing tests raise by
default. Explicit `missing="drop"` excludes them from the tested family while
retaining their original rows with missing adjusted values and rejections.
Dropping changes the inferential family and requires scientific justification;
it is never an automatic remedy for outcome-dependent missingness.

| Method | Adjusted-value construction in ascending p order | Error-control assumptions |
| --- | --- | --- |
| `bonferroni` | `min(1, m*p)` | FWER, arbitrary dependence |
| `sidak` | `1-(1-p)^m` | FWER, independence or an applicable Sidak inequality |
| `holm` | Running maximum of `(m-i+1)*p_(i)` | FWER, arbitrary dependence |
| `holm_sidak` | Running maximum of `1-(1-p_(i))^(m-i+1)` | FWER, independence or an applicable Sidak inequality |
| `hochberg` | Reverse running minimum of `(m-i+1)*p_(i)` | FWER, independence or the applicable positive dependence/Simes condition |
| `hommel` | Closed Simes testing via Hommel's finite shortcut | FWER, valid Simes intersection tests |
| `bh` | Reverse running minimum of `m*p_(i)/i` | FDR, independence or PRDS on true nulls |
| `by` | BH multiplied by `sum(1/i)` | FDR, arbitrary dependence |

Final values are bounded by one. Sorted ties have equal adjusted values, and
stable restoration preserves the caller's order. Rejection is explicitly
`adjusted_p_value <= alpha`. Output metadata saves the complete family labels,
tested/excluded original positions, ordering positions, tie groups, method,
level, missing policy and dependence assumption. FDR concerns the expected
false discovery proportion; it is not FWER under a mixture of true and false
nulls. See the original [BH paper](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x),
[BY paper](https://doi.org/10.1214/aos/1013699998),
[R's documented adjustments and author references](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html),
and [SAS's procedure formulas](https://support.sas.com/documentation/cdl/en/statug/66859/HTML/default/statug_multtest_details11.htm).

`oe.stepdown(observed, null_draws, method="romano_wolf", tail="two-sided",
calibration="monte_carlo", alpha=.05, labels=None, null_description=...)`
requires a finite joint draws-by-hypotheses matrix and a description of its
scientifically valid null/design. Romano-Wolf uses suffix maxT on comparable
studentized statistics. The `greater` and `less` tails transform the statistics
accordingly. Westfall-Young (`method="westfall_young"`) uses minP on supplied
observed and joint-null marginal p-values in [0,1]; its `tail` must remain
`two-sided` because those p-values already embody the marginal tail definition.
Tied observed tests are removed together, and adjusted values use a running
maximum to retain monotonicity and label invariance.

Complete equiprobable null enumeration (`calibration="enumerated"`) uses
`count/B` with inclusive extremes. Monte Carlo uses `(count+1)/(B+1)` and
requires scientifically appropriate independent or exchangeable null sampling;
a plus-one correction cannot make an invalid bootstrap valid. For minP the
`p_value` column preserves the **supplied raw marginal p-value**. A separate
`calibrated_marginal_p_value` column reports its empirical marginal tail rank
under the supplied draws; finite draws can make that rank different from the
supplied value. For maxT, raw marginal p-values are the null tail proportions.
The original supplied statistic and the transformed ordering statistic are
also retained.

A correct global-null matrix alone establishes neither general strong control
nor validity of arbitrary row permutations. For Romano-Wolf, the supplied
subset-null approximation must be valid for the true-null subsets and its
critical values monotone; for Westfall-Young, valid marginal null distributions
and subset pivotality are required. A documented null is a caller attestation,
and `null_design_verified=False` is saved. The implementation follows the
stepdown construction and distinctions in the
[original Romano-Wolf paper](https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffd823d949/jasa.pdf)
and [SAS's Westfall-Young procedure documentation](https://support.sas.com/documentation/cdl/en/statug/66859/HTML/default/statug_multtest_details11.htm).
Model-specific bootstrap generation, nuisance fitting and nonnormal targets
remain separate method contracts.

`oe.simultaneous_ci(estimates, covariance, family_description=..., labels=None,
alpha=.05, draws=50000, seed=1729)` uses the full joint covariance of compatible
estimates to simulate the joint normal maximum absolute standardized error.
Covariance must be finite, symmetric in marginal-SE units, positive
semidefinite within the reported correlation-eigenvalue tolerance `1e-12`, and
have strictly positive marginal variances. Symmetry checks use marginal-SE
units so tiny or highly disparate scales do not excuse material asymmetry.
Roundoff eigenvalues in `[-1e-12,0)` are clipped only for the square root; the
saved smallest eigenvalue, tolerance and clipped count disclose that operation.
The full supplied covariance and the symmetrized correlation are saved.

The table reports estimates, marginal standard errors and simultaneous endpoints
`estimate +/- critical_value*std_error`, with no invented t degrees of freedom
or marginal p-values. The empirical inverse CDF uses higher interpolation, and
the Monte Carlo CDF standard error is `sqrt(alpha*(1-alpha)/draws)`. A local CPU
seed leaves the global RNG untouched. Explicit unique labels canonicalize
simulation order, giving identical critical values under a common labeled
family's permutation; the canonical positions and labels are saved. Without
explicit identities, permuted finite seeded simulations are only Monte Carlo
equivalent. These are normal-limit intervals; normality/normal-limit validity
and correctness of the joint covariance remain the caller's responsibility.
Small-sample t, fitted-model bootstrap, weights and distribution-selection
options are not accepted.

Budgets are checked before copies or payload scans where shape/length is
available: at most 100,000 **raw** p-value positions, including dropped rows;
Hommel permits at most 5,000 raw positions. Joint stepdown permits 8 million
draw cells and `B*m*m <= 100 million` scalar-work proxy. CI permits at most 384
estimates, 1,000–200,000 draws and 8 million simulated cells. These bound resident
inputs, tensor dimensions and operation proxies; they are not a measured process
RSS cap. Caller input construction is outside these guards. CI's dense matrix
multiplication can require up to roughly 3.1 billion scalar products at the
maximum domain. Dataset/streaming/GPU routes are unsupported. Strict shape/type
and finite-arithmetic failures raise `AnalysisError`; unsupported options raise
normal Python `TypeError`, without truncation or fallback.

Results are standard OpenEcon DataFrames with JSON-safe `attrs` and escaped
LaTeX export. Pandas table JSON by itself omits `attrs`: persist the table and
metadata together, for example:

```python
from io import StringIO
import json
import pandas as pd
import openecon as oe

result = oe.multipletests([.01, .2, .03], labels=["growth", "income", "employment"])
state = {"schema": 1, "table": json.loads(result.to_json(orient="table", double_precision=15)),
         "attrs": result.attrs}
saved_json = json.dumps(state, allow_nan=False)
saved = json.loads(saved_json)
restored = oe.DataFrame(pd.read_json(StringIO(json.dumps(saved["table"])), orient="table"))
restored.attrs = saved["attrs"]
latex = restored.to_latex()
```

When excluded rows exist, Pandas can represent their JSON-null rejection as
NaN on readback; both represent an unavailable rejection, never `False`.
Covariance and inference metadata retain their full JSON float precision in
`attrs`. Tests check restored numbers, missing positions, metadata and LaTeX.
There is no ResultBundle fit or saved prediction adapter for these summary
procedures, and no estimator convergence criterion is fabricated.

The independent validation runner is `scripts/validate_multiple_testing.py`.
Run it first with `--protocol-only`, then normally with the same source and
protocol. Frozen seeds/designs/gates are saved under
`docs/evidence/market-182/protocol.json` before outcomes. It tests 5,000 planned
families per cell, at .01/.05/.10, with independent/positive-PRDS/mixed-sign
normal dependence, full and partial nulls, applicable FWER/FDR targets,
scientifically valid known-covariance maxT/minP draws and complete signed
orbits. Separate observation/null/CI streams prevent sharing null draws with
observations. Family-level counts, all failures and input/output hashes are
retained; failure never removes or replaces a planned family. Coverage uses
5,000 independent compatible normal estimate families and a native critical
value's exact shift equivariance. Analytic independent/perfect-correlation
normal CDFs and recorded Monte Carlo uncertainty provide separate references.
The frozen diagnostic tolerances are finite-simulation screens, not proofs for
all distributions. Original definitions, exhaustive closed testing, exact sign
orbits and documented community references supply separate numerical checks.

Each joint-normal design uses one independently seeded, pinned 4,999-row null
fixture across its 5,000 observation families. Its Wilson intervals describe
**conditional observation Monte Carlo uncertainty given that finite null
fixture**. They do not integrate the additional variation of drawing a fresh
null calibration matrix for every family; that experiment was not run. CI
coverage is likewise conditional on its recorded native critical calibration,
whose separate CDF Monte Carlo standard error and available analytic CDF checks
are reported. These distinctions do not change the frozen gates or turn a
conditional size screen into an unconditional bootstrap-validity claim.

The initial full run reached every planned screen but failed while writing its
final JSON because two NumPy boolean gate flags were not serializable. The
original protocol, script and failure log are preserved. Only those metadata
flags were cast to Python booleans; the native engine, numerical algorithms,
seeds, sample sizes, designs, gates and failure denominator were unchanged. A
recorded two-line patch and `serialization-repair.json` explain the identical
deterministic rerun used to recover the complete outcome artifact.

The completed artifact contains all 34 adjustment cells, 16 joint-stepdown
cells and 12 CI cells: 5,000 family evaluations per cell, zero failures and zero
replacements. All frozen screens passed. The run made 250,012 native summary
calls; CI's 60,000 coverage evaluations translate 12 native critical
calibrations. The 56 focused tests include independent exhaustive closed-Simes
Hommel checks, complete sign-orbit full/partial-null controls, strict guards,
persistence and a fresh process with external numerical imports blocked.
`docs/evidence/market-182/audit_receipt.py` independently recalculates all saved
counts, rates, uncertainty intervals and gates using pure Python, and checks
source/protocol hashes and the serialization-only repair. It writes the final
`review.json`; the full arrays remain in `scientific-validation.json.gz`.

A later strict-input review found that integers too large for float64 conversion
could escape as `OverflowError`. Exactly two conversion catch clauses now map
that failure to the existing `AnalysisError("invalid_values")`. Vector, joint
null-matrix and covariance regression tests exercise it. The prior full protocol,
scientific artifact, receipt and native source are preserved in
`prior-pre-overflow-guard/`; `overflow-guard-repair.json` records the source-pin-only
protocol change. The same 62-cell protocol is repeated with unchanged seeds/gates,
and the final auditor checks every scientific outcome against that prior source.

Source CPU evidence, independent scientific/community checks, licensed Stata/
SPSS execution and installed desktop evidence are different proof layers.
The scientific runner does not perform installed or licensed-vendor tests.
Separate frozen and installed Mac acceptance is joined in
the integration evidence (internal evidence excluded from this public snapshot), including
source bytecode identity, native Run, LaTeX and restart readback. That local QA
bundle declares macOS 26.0 for its actual bundled native libraries; it does not
establish older-macOS portability, a public release or notarization.
No licensed vendor or CUDA validation is claimed; `stata_parity_validated=False`
remains unchanged. PR91's other analytical families and broader claims were not
imported by this scoped integration.

Model-generated justified joint nulls are tracked separately in
[MARKET-191](https://linear.app/bluearf/issue/MARKET-191); nonnormal and small-sample
joint regions are tracked in [MARKET-192](https://linear.app/bluearf/issue/MARKET-192).
These stages retain the current summary APIs and require their own method-level
scientific and resource contracts.
