# Saved marginal linear contrasts after multiple imputation

`oe.mi_lincom(pool, R, values=None, names=None, alpha=None, max_work=10_000_000)`
implements MARKET-440. It accepts a validated or explicitly restored
`MIPoolResult` and a declared numeric k-by-p contrast matrix. It retains the
entire source pool, every contrast weight, null, label and projected
per-imputation coefficient/covariance array in an immutable `MILincomResult`.
The source imputation IDs, declared procedure, common estimand/sample contract
and complete-data degrees of freedom remain available in the saved state.

For every imputation i, it first computes `Q_i* = R Q_i` and
`U_i* = R U_i R'`. This uses all off-diagonal covariance terms. It then applies
the existing `mi_pool` Rubin rules to the projected estimates: the mean Qbar,
mean within variance Ubar, sample between variance B, and
`T = Ubar + (1 + 1/m) B`. Each contrast uses its own diagonal of Ubar/B/T,
relative variance increase, missing-information fraction and reference df.
Finite `pool.complete_df` selects the existing Barnard–Rubin marginal df;
`None` selects Rubin's finite-imputation, large-complete-sample rule. Infinite
df is displayed as JSON null and uses the normal reference limit.

The estimate and marginal confidence interval describe `R Q`. The statistic is
`(Qbar* - values) / sqrt(diag(T))`; its two-sided t/normal p value tests the
declared null. Nulls default to zero. A nonzero null never shifts the stored
per-imputation estimates before calculating B: such a shift could erase small
between-imputation differences when the null is much larger than the estimates.
`alpha=None` inherits the source level; an explicit alpha changes interval
coverage. The source pool itself is preserved unchanged.

These are separate, unadjusted marginal tests and intervals. They do not provide
a simultaneous confidence region, familywise/FDR correction, D1 joint p value,
nonlinear transformation, imputation generation or pooled predictive
distribution. Scientific use requires a justified common, approximately normal
estimand and the source imputation/complete-data inference assumptions. No
convergence or licensed Stata/SPSS parity is inferred from successful execution.

Admission is resident CPU float64, m=2..100, p=1..32, and 1<=k<=p. R must have
full row rank under an explicit double-precision numerical tolerance. Projected
within covariances must be finite symmetric PSD with strictly positive marginal
variances; no repair, ridge, clipping, or diagonal-only substitute is used.
Boolean, complex, object, nonfinite, accelerator, ragged and incompatible inputs
are refused before their tensor conversion. Source/contrast labels are unique,
nonempty strings of at most 256 characters. Both source and contrast metadata
have 131072-character envelopes. The source description retains its existing
8192-character bound.

An estimated work limit (default 10 million, explicit maximum 1 billion) and a
named workspace plan cover source validation, matrix projection, retained
results, covariance checks, labels and serialized state before buffer creation.
The current `OPENECON_WORKSPACE_MB` or task override applies to construction,
JSON restoration, copied state and tabular replay. The recorded original budget
is retained when a saved result is loaded with a different sufficient budget.
This is a buffer estimate, not a process-RSS or elapsed-time guarantee.

JSON loading validates both nested pool checksums and recomputes the projection
from the retained source. Changing projected estimates/covariances, IDs, labels,
complete df, work estimates or scientific metadata is refused even when a
changed checksum has been supplied. The outer checksum detects accidental
state changes; it is not authentication. `model_copy` follows the same validation
path instead of Pydantic's unchecked copy route.

```python
pool = oe.mi_pool(fits, imputation_description="Declared MI procedure and common model")
contrast = oe.mi_lincom(pool, [[0.0, 1.0, -1.0]],
                        values=[0.5], names=["coefficient difference"])
display(contrast.table)
saved = contrast.model_dump_json()
restored = oe.MILincomResult.model_validate_json(saved)
assert restored.table.equals(contrast.table)
```

Primary references: [MICE author's scalar Rubin/Barnard–Rubin pooling reference](https://amices.org/mice/reference/pool.scalar.html)
and [van Buuren's reference-df equations](https://stefvanbuuren.name/fimd/sec-whyandwhen.html).
The independent development oracle projects each full covariance with NumPy,
then computes Rubin moments and contrast-specific df/FMI with SciPy t/normal
tails and intervals. Those libraries are development references; numerical
execution in the installed worker uses native float64 Torch. Tests cover
zero/nonzero nulls, zero-between limits, differing confidence levels, full
covariance, saved replay, raw-type refusals and resource-before-buffer guards.
