# Categorical geometry and ordinal scaling

`oe.mca`, `oe.overals` and `oe.mds_nonmetric` return descriptive `TableSet`
geometries. Their kernels use resident native Torch CPU float64. Their scientific
tests use separate NumPy linear algebra and SciPy isotonic regression as oracles;
those oracle packages are not runtime dependencies. No standard errors,
inferential test, global-optimum, adjusted-inertia or vendor-equivalence claim is
made. Iterative methods select the best **converged** start; an unconverged lower
objective is retained in the start table but is not admitted as the fitted model.

## Multiple correspondence analysis

```python
result = oe.mca(data=df, variables=["sector", "region", "size"], n_components=2)
state = oe.summary_state(result)
restored = oe.restore_summary(state)
new_scores = oe.mca_project(restored, data=new_rows)
```

For `q` active variables and complete disjunctive matrix `Z`, every row has
exactly `q` ones. Calibration uses equal-person masses `r_i=1/n`, category
masses `c=mean(Z)/q` and SVD of
`S = diag(sqrt(r)) (Z/q - 1 c') diag(1/sqrt(c))`.
Raw eigenvalues are squared singular values. The raw total is `(K-q)/q`;
no Benzécri/Greenacre adjustment is applied. Row principal coordinates are
`(Z/q-c) diag(1/sqrt(c)) V`, and category principal coordinates are
`diag(1/sqrt(c)) V diag(singular_values)`. Row/category axis contributions sum
to one. Requested zero-inertia axes are refused. Sign identification makes the
largest absolute row coordinate on each axis positive; repeated singular values
remain identified only up to rotations of the tied eigenspace.

The complete state retains typed category levels, masses, standard axes,
raw inertia, input category rows, physical sample positions, row labels and
all coordinate/contribution tables. `mca_project` checks bounded calibration
dimensions, checksum, category masses, weighted orthonormality and the saved
standard-coordinate table before projection. Its transformation exactly matches
fitted rows after restoration. Unknown categories are refused; projection does
not recalibrate on supplementary observations. MCA fitting admits no weights in
this stage; existing weighted correspondence methods have a separate scope.

MCA is the complete-disjunctive special case of the categorical/homogeneity
framework described by the original authors in
[de Leeuw and Mair (2009)](https://www.jstatsoft.org/article/view/v031i04).

## Multiset nonlinear canonical/homogeneity analysis

```python
result = oe.overals(
    data=df, sets=[["sector", "size"], ["region", "rating"]],
    scales={"sector": "multiple_nominal", "size": "numeric",
            "region": "nominal", "rating": "ordinal"},
    orders={"rating": ["low", "medium", "high"]},
    n_components=2, n_starts=4, seed=0,
)
```

Each variable belongs to exactly one of at least two nonempty sets. The scale
map is mandatory. Ordinal orders must list every observed level once, with no
implicit sorting of strings or integer codes. `multiple_nominal` learns one
unrestricted category-coordinate vector per category; `nominal` learns one
scalar quantification and a dimension loading (rank-one variable contribution);
`ordinal` adds a nondecreasing constraint to that scalar; `numeric` retains the
observed affine numeric scale. Numeric variables may have up to `n` distinct
finite values; the 32-level bound applies to categorical scaling variables.

The fitted compromise `X` is centered with `X'X/n=I`. The implemented loss is
`sum_s ||X - sum_(j in set_s) Q_j[codes_j] A_j||² / (n * number_of_sets)`.
Variable blocks update against a residual including all other variables in
their own set. Unrestricted category means, weighted category rank-one SVD,
numeric least squares, and weighted monotone scalar/loading steps solve the
respective blocks. A centered polar update optimizes the common compromise.
Scalar transformations have zero mean and unit mean square on fitted people.
The objective must decrease numerically; convergence uses its decrease with the
declared tolerance. Starts have separate seeds and full objective traces.

The final common axes diagonalize the symmetric compromise-fit operator, then
receive a deterministic sign. Complete quantifications, loadings, category maps,
input/sample identities, set scores and all traces are saved. Those maps
reconstruct every set fit exactly after `restore_summary`. One variable per
multiple-nominal set reduces to MCA/homogeneity axes; all-numeric multiset fits
reduce to the common-score projection objective of linear canonical analysis.
This is an actual multiset nonlinear objective, not PCA on assigned category
codes. Spline bases, partial-row missing algorithms, weights, passive variables
and global optimality are outside this stage.

The set/scaling interpretation follows
[IBM OVERALS](https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=categories-nonlinear-canonical-correlation-analysis-overals).
The loss and constraints follow the original authors'
[Gifi theory](https://www.stat.ethz.ch/CRAN/web/packages/Gifi/vignettes/gifi_theory.html).

## Nonmetric multidimensional scaling

```python
result = oe.mds_nonmetric(
    dissimilarities=distance_table, n_components=2,
    ties="secondary", zero="include", n_starts=4, seed=0,
)
```

Input is a complete, finite, nonnegative symmetric square matrix with zero
diagonal. Labeled matrices require identical row and column labels. At least
two off-diagonal dissimilarity levels are required. The `secondary` tie rule
keeps equal dissimilarities at equal disparities. `primary` ties may split;
within each tie block current distance order gives the least-squares monotone
fit. Zero pairs are ordinary lowest-level observations with `zero="include"`;
they do **not** force zero fitted distances. `zero="exclude"` removes them from
the objective and requires the remaining positive-edge graph to be connected.
NaN/missing/asymmetric pairs and tertiary ties are refused.

At every step, weighted pooled-adjacent-violators regression estimates monotone
disparities, normalized so their squared sum equals the observed pair count.
SMACOF updates the centered coordinates using the graph Laplacian pseudoinverse
and the majorization `B` matrix. `normalized_stress` is raw residual sum of
squares divided by that pair count; `stress_1` divides by squared fitted
distances before taking its square root. Full observed pairs, disparities,
distances, residuals and start/iteration traces are saved. Axes receive a
principal-axis rotation and sign identification; rotation/reflection equivalence
remains when axes tie. Requested and effective ranks are recorded. Optional
`init` supplies only the first start; remaining starts are seeded random.

The method/tie/normalization definitions follow the original authors'
[SMACOF version 2 paper](https://www.jstatsoft.org/article/view/v102i10).
This is nonmetric ordinal stress fitting, distinct from classical/metric MDS.

## Admission and persistence boundaries

MCA/OVERALS admit 4–3000 physical rows, 2–12 named variables and 2–32 observed
levels per categorical variable. Ordinal orders are exact observed levels;
numeric OVERALS variables retain continuous numeric values. Variable names are
1–128 characters; category/row-label strings are at most 256 characters. Labels
and categories are finite JSON scalars; integer values must be exactly
representable in float64. Training missing values use explicit `missing="drop"`
(listwise) or `"raise"`. Supplementary MCA admits 1–3000 rows and defaults to
`missing="raise"`. Physical positions and typed row labels survive exclusions.

NMDS admits 4–150 objects and 1–6 dimensions below `n`; OVERALS admits 1–6
dimensions within its explicit set/scaling rank domain; MCA admits 1–12
nonzero-inertia dimensions. Iterative methods admit 1–12 starts and 1–1000
iterations; defaults are four starts and 500 iterations. Tolerance is finite
between `1e-12` and `1e-3`. No converged start means an `AnalysisError`.

`max_bytes` defaults to 128 MiB and is bounded by that cap and the global
workspace budget. `max_work` defaults to 300 million planned operations and
cannot exceed it. Geometry/workspace checks run before selected-frame or
large numerical allocations; each input shape is checked before conversion.
The geometry envelope alone does not imply every combination of its maxima
fits the budgets. Estimates cover named working buffers and retained numerical
state; caller-owned DataFrames/objects, interpreter memory and total process RSS
are not capped. These are resident methods, not streaming estimators.

`oe.summary_state(result)` retains every complete table and metadata value;
`oe.restore_summary(state)` does not re-fit. New methods canonicalize table
order and finite row representation so exact JSON and concatenated LaTeX are
preserved. A console settings preview points to full state when metadata is
large; a truncated preview is not complete calibration/state evidence.
