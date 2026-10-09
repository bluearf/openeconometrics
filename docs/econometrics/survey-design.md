# Single-stage survey declarations

`oe.survey_design` declares and validates design geometry only. It produces no
survey mean, regression, covariance, p-value, confidence interval or design
effect. [Further implementation stages](roadmaps/survey.md) remain open.

```python
import openecon as oe

frame = oe.DataFrame({
    "w": [2., 3., 4., 2., 2., 3.],
    "psu": [1, 1, 2, 1, 2, 2],
    "stratum": ["a", "a", "a", "b", "b", "b"],
    "population_psu": [8, 8, 8, 4, 4, 4],
})
design = oe.survey_design(frame, weights="w", psu="psu", strata="stratum",
                          fpc="population_psu")
assert design.validation.n_psu == 4
assert design.validation.design_df == 2
saved = design.model_dump_json()
restored = oe.SurveyDesign.model_validate_json(saved)
restored.revalidate(frame)
```

PSUs are nested within strata: PSU 1 in stratum a differs from PSU 1 in b.
Scalar integer, float, string and Boolean identifiers remain distinct after
the caller's table conversion. If pandas already coerced values to one dtype,
the original types cannot be recovered. Empty/missing/nonfinite/non-scalar IDs
fail. Labels are bounded at 256 characters; column names at 200.

Weights are positive finite real sampling weights and remain unnormalized.
Booleans, numeric strings, zero, negative and missing weights fail. No design row
is dropped. Strata omitted means one unstratified design. FPC is an optional
positive integer count of population PSUs, at most 2**53, constant within each
stratum and at least its sampled PSU count. Sampling fractions are deliberately
unsupported. A census stratum is marked certainty. Single-PSU strata require
`singleton="certainty"` and an explicit FPC of one; otherwise they fail. Counts
and `PSUs-strata` df describe geometry, not an available inferential test.

The declaration is immutable and JSON serializable. Its digest covers physical
row order, design-role values/types, column dtypes and category dictionaries/
ordering. It excludes index labels and unrelated outcome columns. No original
PSU labels, individual weights or user rows are embedded in the saved summary.
After restoring, `revalidate` recomputes the full snapshot and rejects changed
inputs or altered summary fields. The digest is neither an authenticated sample
signature nor a guarantee that real-world sampling assumptions hold.

CSV does not retain pandas dtypes/category metadata. A parser may return string
extension dtypes instead of object columns, and exact revalidation will reject
that drift. Retain the source schema and restore it explicitly before revalidation
when the underlying design values are unchanged; otherwise declare a new design.

Input is a resident DataFrame, column mapping or sequence of row mappings.
Dataset/device/multistage/replicate-weight routes are unsupported. Explicit
admission budgets are `max_rows` (1..1,000,000) and `max_memory_mb` (1..512, default
64): projected design-column storage plus conservative group/hash admission.
This bounds accepted resident geometry, not process RSS or caller-owned input.
The entire declaration is admitted or rejected; it never truncates a sample.
Declaration/JSON inspection loads no Torch or estimation runtime.

`tests/test_survey_design.py` checks independently specified geometry, nested
typed IDs, FPC/singleton failure domains, JSON file roundtrip/tampering, exact
input drift/category changes, budgets and lazy imports. This is source-level
contract evidence; it does not certify the future survey estimators or vendor
parity. [Stata's declaration manual](https://www.stata.com/manuals/svysvyset.pdf)
is a comparison reference with a broader option domain.
