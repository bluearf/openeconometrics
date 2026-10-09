# Saved CATPCA varimax and promax

`catpca_varimax(fit)` and `catpca_promax(fit, power=4)` rotate the component
space of an already fitted, unweighted single-vector `catpca` result. The
saved source may come from `restore_summary(summary_state(fit))`. They do not
estimate new optimal category maps or change the standardized reconstruction
objective. Both support resident CPU float64, 3–12 variables, 2–min(6,p−1)
components, and at most 3000 source rows. No multiple-nominal transformation,
adaptive uncertainty, bootstrap, global optimum or vendor equivalence is claimed.

```python
base = oe.catpca(data, variables, components=2, scales=scales, orders=orders)
rotated = oe.catpca_promax(base, power=4, normalize=True)
portable = oe.summary_state(rotated)
restored = oe.restore_summary(portable)
new_scores = oe.catpca_rotated_predict(restored, new_data, missing="raise")
```

For standardized quantified input Z, the retained eigenvectors A and positive
eigenvalues λ give normalized original person scores X=ZA diag(λ⁻¹ᐟ²) and
loadings L=A diag(λ¹ᐟ²), with XᵀX/n=I. Every reported transformation T obeys

- pattern P=LT;
- rotated person scores Xr=XT⁻ᵀ;
- component correlations Φ=T⁻¹T⁻ᵀ=(TᵀT)⁻¹;
- structure S=PΦ;
- XrPᵀ=XLᵀ and PΦPᵀ=LLᵀ.

Varimax uses an orthogonal T, so Φ=I and pattern equals structure. It maximizes
the usual sum over columns of `sum(l**4) - sum(l**2)**2/p`. Optional Kaiser
normalization divides each variable's loading row by its retained communality
square root during rotation. Cyclic two-column rotations maximize each pair
exactly; a full sweep precedes convergence acceptance. The trace saves the
criterion, improvement and relative tangent-gradient stationarity. Acceptance
requires the requested tolerance; exhausted iterations raise `nonconvergence`.
This is a local solution for more than two components, not a global guarantee.

Promax first finds the same varimax solution V=LR. It constructs the target
H=sign(B)|B|ᵏ, where B is V with optional Kaiser row normalization. It solves
`min_U ||VU-H||²`, rejects singular/ill-conditioned U, and rescales columns of
RU to make diag(Φ)=1. The default power is 4; the declared bounded range is
[1,10]. Promax is a finite powered-target least-squares transformation, not an
iterative oblique criterion optimum. Pattern and structure differ whenever
the rotated components correlate. Both methods order components by decreasing
pattern squared-column norm and orient the largest absolute pattern entry
positive; tied solutions remain unidentified up to the usual transformations.

Original scalar category quantifications, numeric means/standard deviations,
typed levels, category counts and sample positions are retained unchanged.
The source does not save original per-row category identities, and different
categories can have tied quantifications. Exact original-category person
centroids cannot be recovered from that source; no category-centroid table is
invented. Rotation acts on the component space, not these scalar maps.

`catpca_rotated_predict` applies those original maps and `T⁻ᵀ` to new rows.
Unknown typed categories fail. `missing="drop"` preserves original new-input
row positions. New-person reconstruction is identical to the original saved
CATPCA projection, including after full restoration. The result supplies no
standard errors, p-values, confidence intervals or adaptive uncertainty.

Complete persistence uses `summary_state`, not truncated console previews.
The rotation state embeds the complete original summary, its checksum, T,
pattern/structure/Φ, training scores, reconstruction, trace and all settings.
After restoration, prediction checks checksums and numerical consistency:
original maps and counts, sample alignment, leading eigensystem, unit-variance
scores, reconstruction loss, orthogonality or nonsingularity, unit Φ diagonal,
structure, method-specific varimax stationarity/pairwise optimality and promax
powered-target least squares. A recomputed checksum cannot make an
inconsistent geometry valid. Table and metadata checks also prevent a saved
display from disagreeing with the computational state. Table order and
numeric representations are canonical for exact JSON and LaTeX restoration.

`max_iter` is bounded to 1–1000 and `tol` to [1e-12,1e-3]. Declared `max_work`
and `max_bytes` are checked before numerical copies. The global workspace
budget also applies. Complete embedded source is bounded to 8 MiB; individual
source text cells/metadata strings are bounded to 4096 characters. Resource
plans cover declared numerical buffers, bounded table/serialization copies and
validation. They do not measure total process RSS, caller-owned source objects
or private BLAS workspace.

The method definitions follow IBM's [CATPCA rotation
subcommand](https://www.ibm.com/docs/en/spss-statistics/30.0.0?topic=catpca-rotation-subcommand-command)
and the primary [Algorithms Guide](https://www.ibm.com/docs/en/SS3RA7_18.4.0/pdf/AlgorithmsGuide.pdf).
These sources define varimax, Kaiser normalization, promax targets and the
pattern/structure/correlation outputs; the tested API and bounds above are
OpenEconometrics' declared scope. Independent tests optimize the complete
two-dimensional periodic varimax criterion with SciPy and reconstruct promax
with NumPy least squares. They also test three-dimensional stationarity,
orthogonal/oblique reconstruction, supplementary projection, missing/category
admission, resource refusal before allocation, CPU residency, tampered state
and exact full-state/LaTeX restoration.
