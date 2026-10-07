# Multivariate analysis

Principal components, factor analysis with rotations and scores, scale
reliability, k-means and hierarchical clustering, discriminant analysis,
canonical correlation, multidimensional scaling and correspondence analysis:
`pca`, `pca_scores`, `factor`, `factortest`, `factor_scores`, `alpha`,
`cluster_kmeans`, `cluster_assign`, `cluster_hierarchical`, `cluster_cut`,
`discrim`, `discrim_predict`, `canon`, `mds` and `ca`. They cover SPSS's FACTOR,
RELIABILITY, QUICK CLUSTER, CLUSTER, DISCRIMINANT, CORRESPONDENCE and PROXSCAL,
and Stata's `pca`, `factor`, `rotate`, `alpha`, `cluster`, `discrim`, `candisc`,
`canon`, `mds`, `mdsmat` and `ca`.

Everything on this page is computed in OpenEconometrics on float64 PyTorch tensors.
Moment matrices come from one pass over the data (centre, then one `X'X`
product), decompositions are `torch.linalg` eigen / QR / SVD / Cholesky
routines, the maximum-likelihood factor model is maximized with the package's
own BFGS optimizer and an analytic gradient, and rotations are iterated here.
No statistics library runs at fit time; NumPy, SciPy and statsmodels appear
only in the test suite as independent oracles.

These procedures are descriptive analyses, not model fits. They register no
estimator and return a `TableSet`: a dictionary of named result tables
(`openecon.frame.DataFrame`) with `str()` and `.to_latex()` for the whole set
and scalar results in `.attrs`. Functions that produce one value per
observation (`pca_scores`, `factor_scores`, `cluster_assign`, `cluster_cut`,
`discrim_predict`) return a table indexed like the data they are given, so the
columns can be assigned straight back to it.

```python
import openecon as oe

result = oe.factor(df, ["read", "write", "vocab", "algebra", "geometry", "stats"],
                   method="ml", factors=2, rotate="promax")
print(result)                               # every table
result["rotated_loadings"]                  # one table
result.attrs["factors"], result.attrs["heywood"]
df[["verbal", "maths"]] = oe.factor_scores(result, df)
```

| SPSS / Stata | OpenEconometrics |
| --- | --- |
| `FACTOR /EXTRACTION PC`; `pca x1-x6` | `oe.pca(df, cols)` (`mineigen=1` for SPSS's rule) |
| `pca ..., covariance`; `/METHOD=COVARIANCE` | `oe.pca(df, cols, matrix="covariance")` |
| `predict pc1 pc2`; `/SAVE REG(ALL)` | `oe.pca_scores(result, df)` (`normalize=True` for SPSS) |
| `factor x1-x6, pf` / `ipf` / `pcf` / `ml factors(2)`; `FACTOR /EXTRACTION PAF` / `ML` / `PC` | `oe.factor(df, cols, method="pf" / "ipf" / "pcf" / "ml", factors=2)` |
| `rotate, varimax normalize`; `/ROTATION VARIMAX` | `rotate="varimax"` (`kaiser=False` for Stata without `normalize`) |
| `rotate, promax(3)`; `/ROTATION PROMAX(4)` | `rotate="promax", power=3` / `power=4` |
| `rotate, oblimin(0) oblique`; `/ROTATION OBLIMIN(0)` | `rotate="oblimin", gamma=0` |
| `predict f1 f2` (`bartlett`); `/SAVE REG(ALL)` / `BART(ALL)` | `oe.factor_scores(result, df)` after `scores="regression"` / `"bartlett"` |
| `estat kmo`; `/PRINT KMO` | `oe.factortest(df, cols)` |
| `alpha q1-q5, item casewise asis`; `RELIABILITY /MODEL=ALPHA /SUMMARY=TOTAL` | `oe.alpha(df, items)` |
| `/MODEL=SPLIT`, `/MODEL=GUTTMAN` | `oe.alpha(df, items, model="split" / "guttman")` |
| `cluster kmeans x1 x2, k(3) start(firstk)`; `QUICK CLUSTER ... /CRITERIA=CLUSTER(3)` | `oe.cluster_kmeans(df, cols, 3)` (`init="spss"` for SPSS's start) |
| `cluster wardslinkage x1 x2`; `CLUSTER ... /METHOD WARD` | `oe.cluster_hierarchical(df, cols, linkage="ward", metric="sqeuclidean")` |
| `cluster generate g = groups(3)`; `/SAVE CLUSTER(3)` | `oe.cluster_cut(result, k=3, data=df)` |
| `cluster stop` (`rule(duda)`) | `result["stopping"]` |
| `discrim lda x1 x2, group(g)`; `candisc`; `DISCRIMINANT /GROUPS=g` | `oe.discrim(df, "g", cols)` |
| `discrim qda`; `estat classtable, loo`; `/STATISTICS=CROSSVALID` | `method="qda"`; `loo=True` |
| `predict, classification` / `pr`; `/SAVE CLASS PROBS` | `oe.discrim_predict(result, df)` |
| `canon (x1 x2 x3) (y1 y2)`; `MANOVA ... /DISCRIM`, CANCORR macro | `oe.canon(df, x=[...], y=[...])` |
| `mds x1-x5, id(id)`; `mdsmat D`; `PROXSCAL`, `ALSCAL` | `oe.mds(df, cols)`; `oe.mds(distances=D)` (`method="smacof"`) |
| `ca row col`; `CORRESPONDENCE TABLE=row BY col` | `oe.ca(df, "row", "col", weights="count")` |

## Samples, missing data and failures

Inputs are a DataFrame, a mapping of columns or a list of row records. Analysed
variables must be numeric (`non_numeric_column`) and finite
(`non_finite_values`); grouping and category columns may hold any scalar
labels, ordered ascending (categorical columns keep their category order).

**Missing values are deleted listwise** over the columns a procedure uses
(`missing="drop"`, the default of SPSS FACTOR, RELIABILITY and DISCRIMINANT and
of Stata's multivariate commands; note that Stata's `alpha` is pairwise unless
`casewise` is given). `attrs["n"]` and `attrs["n_missing"]` report the rows used
and dropped; `missing="raise"` refuses incomplete rows (`missing_values`).

Every invalid input or numerical failure raises
`openecon.analysis_contracts.AnalysisError(code, message)` with a message that
says what to change. The main codes: `invalid_spec`, `invalid_option`,
`missing_columns`, `non_numeric_column`, `empty_data`, `empty_sample`,
`constant_column` (a variable without variation where correlations are needed),
`singular_matrix` (redundant variables where an inverse is needed),
`insufficient_observations`, `no_convergence` (iterated principal factors, ML
factor extraction, rotations), `no_components` / `no_factors`,
`zero_communality` (Kaiser normalization of a variable without common
variance), `heywood_case` (Bartlett scores with a zero uniqueness),
`too_many_clusters`, `duplicate_centers`, `empty_cluster`,
`too_many_observations` (the n-by-n guard), `invalid_groups`,
`invalid_distances`, `too_many_dimensions`, `invalid_weights`,
`degenerate_table`, `degenerate_scale` and `invalid_result` (a result of the
wrong procedure passed to a post-estimation function).

Redundant (collinear) variables are never dropped silently: procedures that
need an inverse stop with `singular_matrix` and name the remedy, while those
that do not (PCA, `pcf` factors, classical MDS, clustering) run unchanged.

**Sign conventions.** Eigenvectors and singular vectors are defined only up to
sign, and SPSS and Stata leave the choice to their eigen routine. OpenEconometrics
fixes it: each eigenvector, factor, discriminant function, canonical pair and
scaling dimension is signed so that its entry of largest absolute value is
positive (the standardized coefficient for discriminant functions and canonical
pairs). A whole column may therefore differ in sign from another package's
output; nothing else changes. The rule used is stored in
`attrs["sign_convention"]`.

## `pca`: principal components

`oe.pca(data, columns, *, matrix="correlation", components=None, mineigen=None,
missing="drop")`

With `C` the correlation matrix `R` (default) or the covariance matrix `S`
(divisor n - 1), `C = V L V'` with eigenvalues `l_1 >= ... >= l_p`. Component k
has weights `v_k` and variance `l_k`.

Tables: `eigenvalues` (eigenvalue, difference, proportion, cumulative, for all
p components: Stata's table, SPSS "Total Variance Explained"); `eigenvectors`
(the retained unit-length eigenvectors, Stata's "Principal components", plus
`unexplained`, the variance of each variable not reproduced); `loadings`
(eigenvector times `sqrt(l_k)`: SPSS's "Component Matrix", the correlations
between variables and components for a correlation-matrix PCA);
`communalities` (`initial`, `extraction`); `descriptives` (mean, std_dev).
`attrs`: `n`, `n_missing`, `components`, `trace`, `rho` (share of the total
variance carried by the retained components).

Retention follows Stata: components with an eigenvalue above `mineigen`
(default 1e-5), at most `components`. SPSS's default rule is `mineigen=1`.

`oe.pca_scores(result, data, *, center=True, normalize=False)` returns the
scores `z'v_k` of every row, where `z` is standardized with the
estimation-sample means and standard deviations (centred only for a covariance
PCA). Their sample variance is `l_k`. `normalize=True` divides by `sqrt(l_k)`,
which gives the unit-variance component scores SPSS saves.

## `factor`: factor analysis

`oe.factor(data, columns, *, method="pf", factors=None, mineigen=None,
rotate=None, kaiser=True, power=4, gamma=0, scores=None, max_iterations=None,
tolerance=None, missing="drop")`

The model is `R = L Phi L' + Psi` with loadings `L` (p by m), factor
correlations `Phi` (the identity before rotation and after an orthogonal
rotation) and diagonal uniquenesses `Psi`.

### Extraction

| `method` | Estimator |
| --- | --- |
| `"pf"` | Principal factors (Stata's default). The diagonal of `R` is replaced by the squared multiple correlations `SMC_j = 1 - 1/r^jj`; `L = V_m sqrt(D_m)` from that reduced matrix. |
| `"ipf"` | Iterated principal factors (SPSS PAF, Stata `ipf`). Communalities are re-estimated as the row sums of squares of `L` until the largest change is below `tolerance` (0.001) within `max_iterations` (25), SPSS's defaults. Not converging is the error `no_convergence`, never a silently unconverged solution; SPSS stops in the same way. Solutions with communalities near one can need a few hundred iterations: pass `max_iterations=500`. |
| `"pcf"` | Principal-component factors: communalities of one (SPSS PC extraction). |
| `"ml"` | Maximum likelihood by Jöreskog's (1967) method, see below. |

**Maximum likelihood.** For given uniquenesses let `g_1 >= ... >= g_p` and
`w_j` be the eigenvalues and eigenvectors of `Psi^{-1/2} R Psi^{-1/2}`. The
loadings minimizing the discrepancy
`F = ln|Sigma| + tr(R Sigma^{-1}) - ln|R| - p` are
`L = Psi^{1/2} W_m (G_m - I)^{1/2}`, and the concentrated function and its
gradient are

```
F(psi) = sum_{j > m} (g_j - ln g_j - 1),      dF/dpsi_i = -(1/psi_i) sum_{j > m} (g_j - 1) w_ij^2.
```

`F` is minimized over `theta_i = ln(psi_i - 0.005)` by BFGS
(`engines.optimize.maximize_bfgs` on `-F`) with this analytic gradient, which
the tests compare with numerical differentiation. The lower bound 0.005 on a
uniqueness is the one of Jöreskog's program and of R's `factanal`. A uniqueness
that ends at the bound is a **Heywood case**: it is listed in
`attrs["heywood"]` with a note naming the bound, as SPSS and Stata warn, and the
chi-square test should then not be trusted. For `pf`, `ipf` and `pcf` a
communality that reaches or exceeds one is reported the same way; `ipf` keeps
iterating where SPSS stops with a warning. The unrotated ML loadings are in canonical form
(`L' Psi^{-1} L` diagonal).

**Number of factors.** `factors` is the maximum kept; factors whose eigenvalue
does not exceed `mineigen` are dropped. Defaults follow Stata: 5e-6 for pf, ipf
and ml (eigenvalues of the SMC-reduced matrix), 1 for pcf (eigenvalues of `R`).
ML additionally caps the number at the largest model with non-negative degrees
of freedom. When fewer factors are kept than requested, a note says so.

### Rotation

Every rotation returns `rotated = unrotated x M` (`rotation_matrix`) and
`Phi = (M'M)^{-1}`, so `L Phi L'` is unchanged.

| `rotate` | Criterion and algorithm |
| --- | --- |
| `"varimax"`, `"quartimax"`, `"equamax"` | Orthomax, maximize `sum_j [sum_i l_ij^4 - (gamma/p)(sum_i l_ij^2)^2]` with gamma = 1, 0, m/2. SVD iteration `M <- U V'` of `A'G`; Kaiser's pairwise planar rotations for gamma > 1 (equamax with three or more factors), where the SVD iteration can cycle. |
| `"oblimin"` | Direct oblimin, minimize `(1/4) sum_{s != t} [sum_i l_is^2 l_it^2 - (gamma/p) sum_i l_is^2 sum_i l_it^2]` over oblique transformations by Jennrich's (2002) gradient projection. `gamma` is SPSS's delta; 0 is direct quartimin. |
| `"promax"` | Varimax, then the least-squares fit of the target `sign(v)|v|^power` (Hendrickson and White 1964), columns rescaled so that `Phi` has a unit diagonal. `power=4` is SPSS's default kappa; Stata's default is `promax(3)`. |

`kaiser=True` (SPSS's default) rotates the rows scaled to unit length and
scales them back; Stata rotates without it unless `normalize` is given, so use
`kaiser=False` to compare. Rotated factors are ordered by decreasing sum of
squared pattern loadings. Rotations iterate until the gradient of the criterion
(projected on the manifold of admissible transformations) is below 1e-10, far
tighter than the 1e-5 of SPSS, so the rotated loadings are converged to all
printed digits; a rotation that only reaches 1e-5 is accepted with a note and
anything looser is a `no_convergence` error.

### Tables

`eigenvalues` (eigenvalue, difference, proportion and cumulative proportion of
the trace of the factored matrix, as Stata prints them, plus
`initial_eigenvalue`, the eigenvalues of `R` shown by SPSS; for `ml` the sums
of squared loadings of the retained factors); `variance` (sums of squared
loadings as shares of the total variance p, before and after rotation, as in
SPSS's "Total Variance Explained"; for oblique rotations from the structure
matrix and without a cumulative share); `loadings`; with a rotation
`rotated_loadings` (pattern matrix), `rotation_matrix` and, when oblique,
`structure` (`L Phi`) and `factor_correlations`; `communalities` (`initial` =
SMC, 1 for pcf; `extraction`; `uniqueness`); `uniqueness` (the same
uniquenesses as their own table, Stata's "Uniqueness" column); `fit`;
`score_coefficients`;
`descriptives`.

`fit` holds Bartlett-corrected likelihood-ratio tests against the saturated
model: `independence`, `chi2 = -(n - 1 - (2p+5)/6) ln|R|` with `p(p-1)/2`
degrees of freedom (Bartlett's test of sphericity), and for `ml` `model`,
`chi2 = (n - 1 - (2p+5)/6 - 2m/3) F_min` with `((p-m)^2 - (p+m))/2` degrees of
freedom.

### Scores and adequacy

`scores="regression"` (the default used by `oe.factor_scores`): coefficients
`B = R^{-1} L Phi` (Thomson). `scores="bartlett"`:
`B = Psi^{-1} L (L'Psi^{-1}L)^{-1}`, which satisfies `B'L = I`. Scores are
`z'B` for variables standardized with the estimation-sample moments;
`oe.factor_scores(result, data)` applies the stored coefficients to any data.

`oe.factortest(data, columns)` returns the Kaiser-Meyer-Olkin measure of
sampling adequacy, overall and per variable,
`KMO = sum r_ij^2 / (sum r_ij^2 + sum a_ij^2)` over `i != j` with
`a_ij = -r^ij / sqrt(r^ii r^jj)` the anti-image correlations, and Bartlett's
test of sphericity in `attrs` (`statistic`, `df`, `p_value`, `determinant`).

## `alpha`: reliability

`oe.alpha(data, columns, *, standardized=False, reverse=None, model="alpha",
missing="drop")`

With item covariance matrix `C` and scale variance `s_T^2 = 1'C1`,
`alpha = k/(k-1) (1 - sum c_jj / s_T^2)`; the standardized alpha is
`k rbar / (1 + (k-1) rbar)`. `scale` reports both with the average inter-item
covariance and correlation and the mean, variance and standard deviation of
the sum of the items. All of them describe the analysed items: with
`standardized=True` the items are z-scores, so the average inter-item
covariance equals the average correlation, the item means are 0 and the scale
variance is `1'R1`. `items` reports per item the mean, standard deviation,
item-test correlation, corrected item-total (item-rest) correlation, scale mean
and variance if the item is deleted, squared multiple correlation and alpha if
the item is deleted (SPSS "Item-Total Statistics"; Stata `alpha, item`).

`standardized=True` analyses the correlation matrix (Stata `std`).
`reverse=[...]` flips the sign of the listed items; no item is reversed
automatically (SPSS's behaviour, Stata's `asis`). `model="split"` adds SPSS's
split-half table (first `ceil(k/2)` items against the rest: correlation between
forms, Spearman-Brown for equal and unequal lengths, Guttman split-half, alpha
of each half); `model="guttman"` adds Guttman's lambda-1 to lambda-6. The
formulas are in the function's docstring.

## `cluster_kmeans`: k-means

`oe.cluster_kmeans(data, columns, k, *, init="first", seed=None,
max_iterations=10000, tolerance=0.0, standardize=False, missing="drop")`

Lloyd's algorithm minimizes the within-cluster sum of squares
`W = sum_i ||x_i - m_c(i)||^2`: assign every observation to its nearest centre,
replace the centres by the cluster means, repeat until no assignment changes
(or the largest centre shift is at most `tolerance` times the smallest distance
between initial centres, SPSS's CONVERGE). Distances use the expansion
`||x||^2 - 2x'c + ||c||^2` on centred data in blocks; no n-by-n matrix exists.

| `init` | Initial centres |
| --- | --- |
| `"first"` (default) | The first k complete cases (Stata `start(firstk)`, SPSS `NOINITIAL`). Deterministic, but poor if the data are sorted. |
| `"spss"` | SPSS QUICK CLUSTER's default: a pass that keeps k well-separated cases. Deterministic. |
| `"random"` | k distinct observations at random (Stata's default `krandom`), with `seed`. |
| `"kmeans++"` | D^2-weighted random seeding (Arthur and Vassilvitskii 2007), with `seed`. |
| k-by-p list | Explicit centres in the units of the variables (standardized like the data when `standardize=True`; `initial_centers` then shows them as z-scores). |

Tables: `initial_centers`, `centers`, `sizes` (n, percent, within_ss),
`anova` (between- and within-cluster mean squares and F per variable; as SPSS
notes these F tests are descriptive only, because the clusters were chosen to
maximize them), `iterations` (largest centre shift and W per iteration),
`descriptives`. `attrs`: `within_ss`, `between_ss`, `total_ss`,
`calinski_harabasz = [B/(k-1)] / [W/(n-k)]`, `iterations`, `converged`, `init`,
`seed` (drawn and recorded when not given). Reaching `max_iterations` is
reported through `converged=False` and a note, as SPSS does (its own default is
only 10 iterations).

SPSS's selection pass is sequential by definition (each case may replace a
centre). Runs of cases that replace nothing are screened in vectorized blocks;
runs of replacements, which occur at almost every case when the data are sorted
along a clustering variable, are stepped one case at a time on Python floats
while k p <= 1024 (about 4 microseconds per case for k = 4, p = 3), and with
block tensor operations above that (about 80 microseconds per replacement).
Both paths apply the identical rule; ties go to the centre listed first.

`oe.cluster_assign(result, data)` returns `cluster` (1..k) and `distance` to
the centre for every row of any data.

## `cluster_hierarchical`: agglomerative clustering

`oe.cluster_hierarchical(data, columns, *, linkage="ward", metric="euclidean",
standardize=False, max_n=5000, missing="drop")`

Linkages: `ward`, `average` (UPGMA; SPSS between-groups), `complete`, `single`,
`centroid`, `median`, `weighted` (WPGMA; Stata `waveragelinkage`). Metrics:
`euclidean`, `sqeuclidean`, `manhattan`, `correlation` (1 minus the correlation
between two observations' profiles). Clusters merge by the Lance-Williams
recurrences (listed in the module documentation); Ward's, centroid and median
linkage are defined on squared Euclidean distances and accept only the two
Euclidean metrics. With `metric="euclidean"` their heights are reported as
square roots (the convention of SciPy and of R's `ward.D2`); with
`metric="sqeuclidean"` on the squared scale, which is Stata's default for these
linkages and the measure SPSS recommends.

Single, complete, average, weighted and Ward's linkage are found with the
nearest-neighbour chain algorithm (O(n^2)); centroid and median linkage, which
can produce inversions, with the generic closest-pair algorithm.

**This is the one procedure that needs the n-by-n distance matrix**, so n is
guarded by `max_n` (default 5000: 200 MB). Larger samples raise
`too_many_observations`; use `cluster_kmeans`, a sample, or raise `max_n`.

Tables: `agglomeration` (SPSS's schedule: stage, the two clusters named by
their lowest case number, coefficient, the stage at which each was formed, the
next stage, size; with Ward's linkage also `within_ss`, the cumulative
within-cluster sum of squares SPSS prints as its coefficient); `dendrogram`
(the linkage matrix `left`, `right`, `height`, `size` with 0-based ids, the
format of SciPy and MATLAB); `stopping` (for 1 to 15 clusters the
Calinski-Harabasz pseudo-F and the Duda-Hart `Je(2)/Je(1)` with its pseudo-T^2,
as Stata's `cluster stop`); `cases` (case number to row position).

`oe.cluster_cut(result, k=3)` or `height=...` returns the membership; with
`data=df` it is indexed like `df`, ready to be assigned as a column.

## `discrim`: discriminant analysis

`oe.discrim(data, group, columns, *, method="lda", priors="equal", loo=False,
missing="drop")`

With `W` the pooled within-groups SSCP matrix, `S_w = W/(n - G)` and
`B = T - W`:

- `tests_of_equality`: per variable Wilks' lambda `w_jj / t_jj` and the one-way
  ANOVA F.
- `canonical_functions` (LDA): eigenvalues `l_k` of `W^{-1}B`, percent of
  variance, canonical correlations `sqrt(l_k/(1 + l_k))` and Bartlett's
  chi-square test of functions k..q,
  `-(n - 1 - (p + G)/2) ln prod_{j>=k} 1/(1 + l_j)` with `(p-k+1)(G-k)` degrees
  of freedom.
- `unstandardized_coefficients` (scaled so that the pooled within-group
  variance of each function is one, with the `Intercept` that centres the
  scores at the grand mean), `standardized_coefficients`
  (`v_jk sqrt(S_w,jj)`), `structure_matrix` (pooled within-groups correlations
  between variables and functions) and `centroids`.
- `classification_functions` (LDA): Fisher's linear functions
  `x'S_w^{-1}m_g - m_g'S_w^{-1}m_g/2 + ln prior_g`.
- `classification_table`, and with `loo=True` `classification_table_loo`:
  actual by predicted counts and percent correct.
- `box_m` and `log_determinants`: Box's M test of equal covariance matrices
  with its F approximation (the statistic SPSS prints) and chi-square.
- `groups`, `group_means`, `group_statistics`.

`method="qda"` classifies with the group covariance matrices,
`-ln|S_g|/2 - (x - m_g)'S_g^{-1}(x - m_g)/2 + ln prior_g`, as Stata's
`discrim qda`, and stores them in `group_covariances`. `priors` are `"equal"`
(the default of both SPSS and Stata), `"proportional"` or a list.

Leave-one-out classification removes each observation from its own group mean
and from the covariance matrix. It is computed in closed form from rank-one
updates of the whitened distances in O(n G p), not by refitting n times; the
tests compare it with the brute-force refits.

`oe.discrim_predict(result, data)` returns the predicted group and the
posterior probabilities `posterior_<label>` for any data.

## `canon`: canonical correlation

`oe.canon(data, x=[...], y=[...], *, missing="drop")`

Following Björck and Golub (1973), the standardized, centred sets are factored
`X = Q_x R_x`, `Y = Q_y R_y` and `Q_x'Q_y = U diag(rho) V'`; the coefficients
are triangular solves `R_x^{-1} U` and `R_y^{-1} V`. No covariance matrix is
inverted.

Tables: `correlations` (correlation, squared correlation, eigenvalue
`rho^2/(1 - rho^2)`, and for the hypothesis that correlation k and all smaller
ones are zero Wilks' lambda with Rao's F and Bartlett's chi-square
`-(n - 1 - (p+q+1)/2) ln(lambda)` on `(p-k+1)(q-k+1)` degrees of freedom);
`tests` (Wilks, Pillai, Lawley-Hotelling and Roy for all correlations, with the
F approximations of Stata's `manova`/`canon`); `raw_coefficients` (unit-variance
variates) and `standardized_coefficients`; `loadings` (canonical loadings
`Canon*` and cross loadings `Cross*`); `redundancy` (share of each set's
variance reproduced by its own and by the opposite variates).

## `mds`: multidimensional scaling

`oe.mds(data=None, columns=None, *, distances=None, dimensions=2,
method="classical", standardize=False, max_iterations=1000, tolerance=1e-9,
max_n=5000, missing="drop")`

Give either observations (`data`, `columns`: Euclidean distances between rows)
or a square dissimilarity matrix `distances` (Stata `mdsmat`).

`method="classical"` is Torgerson scaling: `B = -J D2 J / 2` is decomposed and
the coordinates are `V_k L_k^{1/2}`. For observations `B = XX'`, so the
solution comes from the p-by-p cross-product matrix and no n-by-n matrix is
formed. `eigenvalues` is Stata's table (eigenvalue with percent and cumulative
percent of the absolute and of the squared eigenvalues, ten rows as
`neigen(10)`), and `fit` gives Mardia's two measures.

`method="smacof"` minimizes raw stress `sum_{i<j} (d_ij - delta_ij)^2` by
iterated Guttman transforms from the classical start (metric scaling with the
identity transformation), reports Kruskal's stress-1 and rotates the
configuration to its principal axes. The stress never increases, so reaching
`max_iterations` gives a usable configuration and is reported by
`converged=False` and a note.

## `ca`: correspondence analysis

`oe.ca(data, row, column, *, weights=None, dimensions=2, missing="drop")`

The contingency table (built from the two columns, optionally weighted by a
count column) gives `P = N/n` with masses `r`, `c`; the SVD of
`D_r^{-1/2}(P - rc')D_c^{-1/2}` gives the singular values `s_k`. Tables:
`inertia` (singular value, principal inertia `s_k^2`, chi-square share,
percent, cumulative); `rows` and `columns` (mass, quality, share of inertia and
per dimension the principal coordinate `dim{k}`, the standard coordinate
`dim{k}_standard`, the squared correlation and the contribution); `table`.
`attrs`: `chi2` (Pearson's, = n times the total inertia), `df`, `p_value`,
`total_inertia`. Stata's default symmetric coordinates are the standard
coordinates times `sqrt(s_k)`; the principal coordinates here correspond to
`normalize(principal)`.

## Worked example

Six simulated test scores of 300 students and their chosen track (the code
runs as shown; the printed values are from that run):

```python
import numpy as np
import pandas as pd
import openecon as oe

rng = np.random.default_rng(2024)
n = 300
track = rng.choice(["arts", "mixed", "science"], size=n, p=[0.35, 0.3, 0.35])
shift = pd.Series(track).map({"arts": -0.8, "mixed": 0.0, "science": 0.8}).to_numpy()
verbal = rng.normal(size=n) - 0.3 * shift
maths = 0.3 * verbal + rng.normal(size=n) + shift
df = pd.DataFrame({
    "read": 50 + 8 * (0.85 * verbal + 0.5 * rng.normal(size=n)),
    "write": 50 + 8 * (0.80 * verbal + 0.6 * rng.normal(size=n)),
    "vocab": 50 + 8 * (0.85 * verbal + 0.5 * rng.normal(size=n)),
    "algebra": 50 + 8 * (0.85 * maths + 0.5 * rng.normal(size=n)),
    "geometry": 50 + 8 * (0.75 * maths + 0.6 * rng.normal(size=n)),
    "stats": 50 + 8 * (0.70 * maths + 0.2 * verbal + 0.6 * rng.normal(size=n)),
    "track": track,
})
tests = ["read", "write", "vocab", "algebra", "geometry", "stats"]

adequacy = oe.factortest(df, tests)
adequacy.attrs["kmo"], adequacy.attrs["statistic"], adequacy.attrs["df"]
# (0.750, 1110.2, 15.0): factorable

fa = oe.factor(df, tests, method="ml", factors=2, rotate="promax")
fa["rotated_loadings"].round(3)
#           Factor1  Factor2
# read        0.025    0.852
# write      -0.039    0.794
# vocab       0.009    0.875
# algebra     0.927   -0.058
# geometry    0.865   -0.041
# stats       0.841    0.105
fa["factor_correlations"].iloc[0, 1]          # 0.239
fa["fit"].round(3)
#               statistic    df  p_value
# model            10.929   4.0    0.027      (stats also loads on verbal ability)
# independence   1110.221  15.0    0.000
df[["maths", "verbal"]] = oe.factor_scores(fa, df)

oe.alpha(df, ["read", "write", "vocab"]).attrs["alpha"]        # 0.877

da = oe.discrim(df, "track", tests, priors="proportional", loo=True)
da["canonical_functions"][["eigenvalue", "canonical_correlation", "chi2", "df", "p_value"]]
#            eigenvalue  canonical_correlation     chi2  df  p_value
# Function1       0.593                  0.610  145.365  12    0.000
# Function2       0.029                  0.166    8.278   5    0.142
da["classification_table_loo"]
#          arts  mixed  science    n  percent_correct
# actual
# arts       96     12       14  122        78.688525
# mixed      33     28       27   88        31.818182
# science    18     23       49   90        54.444444
df["predicted"] = oe.discrim_predict(da, df)["predicted"]

km = oe.cluster_kmeans(df, tests, 3, init="spss", standardize=True)
km["sizes"]["n"].tolist()                     # [118, 96, 86]
df["cluster"] = oe.cluster_assign(km, df)["cluster"]
tree = oe.cluster_hierarchical(df, tests, linkage="ward", standardize=True)
tree["stopping"].head(3).round(3)
#    clusters  calinski_harabasz  je2_je1  pseudo_t2
# 1         1                NaN    0.760     94.035
# 2         2             94.035    0.740     85.164
# 3         3            101.835    0.648     65.211
df["ward3"] = oe.cluster_cut(tree, k=3, data=df)["cluster"]
```

## Performance

Measured by the verification pass on an Apple-silicon machine (12 cores),
one million rows and ten variables, float64, after a warm-up call:

| Call | Time |
| --- | --- |
| `pca` | 0.1 s |
| `pca_scores` (1e6 rows) | 0.12 s |
| `factor` pf / ipf + varimax / ml + promax / ml + oblimin | 0.09-0.11 s |
| `factor_scores` (1e6 rows) | 0.11 s |
| `factortest`, `alpha` (with Guttman's bounds) | 0.09-0.11 s |
| `cluster_kmeans`, k = 5, `init="spss"`, 20 iterations | 0.47 s (about 17 ms per iteration) |
| SPSS initial centres alone: random order / sorted along a variable | 0.09 s / 3.5 s |
| `cluster_assign` (1e6 rows) | 0.15 s |
| `discrim` LDA or QDA with leave-one-out, 4 groups | 0.33 s |
| `discrim_predict` (1e6 rows) | 0.21 s |
| `canon`, 5 and 5 variables | 0.13 s |
| `ca`, 1e6 records | 0.02 s |
| `cluster_hierarchical`, n = 5000: ward / average / centroid / single | 0.63 / 0.52 / 0.88 / 0.42 s |
| `mds` classical, 5000 observations | 0.005 s |
| `mds` classical, 1500 by 1500 distance matrix | 0.22 s |
| `mds` SMACOF, n = 1000 | 2.2 ms per iteration |

Everything except hierarchical clustering and distance-matrix scaling is linear
in n: the data are read once into the p-by-p moment matrix, and all
decompositions are of small matrices. k-means run time is iterations times the
per-iteration cost; on well-separated data it converges in a handful of
iterations (0.4-0.5 s in total at one million rows), while on data without
cluster structure exact convergence (`tolerance=0`) can take several hundred
iterations. Set `tolerance` (SPSS's CONVERGE) or `max_iterations` to stop
earlier.

## Limitations

- No weights (frequency or sampling) except the count column of `ca`.
- Factor analysis works on the correlation matrix of raw data; there is no
  matrix input, no covariance-matrix factoring, no minimum-residual, alpha or
  image extraction, no Anderson-Rubin scores and no standard errors of
  loadings.
- Rotations: varimax, quartimax, equamax, direct oblimin, promax. Stata's
  further criteria (geomin, Crawford-Ferguson family, target rotation) are not
  implemented.
- `alpha` has no pairwise deletion and no automatic sign reversal.
- k-means minimizes Euclidean within-cluster sums of squares only (no k-medians,
  no other dissimilarities, no running means).
- Hierarchical clustering is limited by the n-by-n matrix (`max_n`), offers four
  dissimilarities for continuous data and no measures for binary data.
- `discrim` has no stepwise variable selection (SPSS `/METHOD=WILKS`), no
  k-nearest-neighbour or logistic discriminant analysis, and classifies QDA
  from the original variables (Stata) rather than from the discriminant
  functions (SPSS `/CLASSIFY=SEPARATE`).
- `canon` reports no standard errors of the canonical coefficients.
- `mds`: metric scaling only (classical, or SMACOF with the identity
  transformation); no ordinal (nonmetric) transformations, no individual
  differences models.
- `ca`: simple correspondence analysis of two variables; no supplementary
  points, no multiple or joint correspondence analysis.
- Parity with Stata or SPSS output has not been measured against those
  programs; the oracles are independent implementations (see below).

## How the results were verified

`tests/test_econ_multivariate*.py` (about 280 tests, under 25 s in total)
compare every procedure with an independent computation. The first four files
come from the implementation; `test_econ_multivariate_oracle.py`,
`test_econ_multivariate_oracle_groups.py` and
`test_econ_multivariate_adversarial.py` were written by a separate verification
pass from its own derivations (none of the implementation's formulas or test
code reused):

- PCA and principal-factor methods with `numpy.linalg.eigh` of the correlation,
  covariance and reduced matrices, and with statsmodels' `PCA`; iterated
  principal factors with an independently written loop and with statsmodels'
  `Factor(method="pa")`.
- ML factor analysis with statsmodels' `Factor(method="ml")` and with a
  brute-force SciPy minimization of the full discrepancy over loadings and
  uniquenesses; its analytic gradient with `optimize.check_derivatives`; the
  chi-square tests by hand.
- Varimax, quartimax and oblimin with statsmodels' gradient-projection
  rotations; equamax with a SciPy maximization over rotation angles; promax
  with the Hendrickson-White formulas written in NumPy; invariants
  (`L Phi L'` unchanged, `Phi = (M'M)^{-1}`, stationarity of the criterion).
- Factor scores, KMO, Bartlett's test and every reliability coefficient with
  hand formulas (item statistics from explicit sums and regressions).
- k-means with `scipy.cluster.vq.kmeans2` and a NumPy Lloyd loop from identical
  initial centres; SPSS's initial-centre selection with a case-by-case loop;
  the ANOVA table with `scipy.stats.f_oneway`.
- Hierarchical clustering for every linkage and metric with
  `scipy.cluster.hierarchy.linkage` (identical linkage matrices) and `fcluster`;
  the agglomeration schedule by replaying it; stopping rules from explicit
  within-cluster sums of squares.
- Discriminant analysis with SciPy's generalized eigenproblem, hand formulas
  for Box's M, and brute-force leave-one-out refits for LDA and QDA.
- Canonical correlation with the covariance eigenproblem, statsmodels'
  `CanCorr` and `MANOVA` (multivariate tests), and hand formulas for the
  sequential tests.
- Classical MDS with NumPy double centring; SMACOF with an independent Guttman
  loop and a SciPy optimizer that cannot improve the stress; correspondence
  analysis with NumPy's SVD and `scipy.stats.chi2_contingency`.

Verification-pass oracles in particular: every rotation (varimax, quartimax,
equamax, with and without Kaiser normalization) against a brute-force SciPy
maximization over orthogonal matrices parametrized by matrix exponentials;
oblimin with three values of gamma against a brute-force minimization over
oblique transformations; ML factor analysis against an unconstrained SciPy
minimization of the full discrepancy over loadings and log-uniquenesses (and,
for a Heywood case, a bounded L-BFGS-B minimization); Cronbach's alpha, the
item table, split-half and Guttman's six bounds from raw sums and regressions
of the item data; SPSS's initial centres with a case-by-case loop on random,
integer, sorted and alternating data; Duda-Hart and Calinski-Harabasz indices
from partitions cut by SciPy; height cuts against SciPy's cophenetic rule on a
tree with inversions; centroid and median merges under exact ties replayed
against the closest-pair definition; leave-one-out classification by n
explicit refits with SciPy normal densities; canonical loadings and cross
loadings as correlations computed from the variates; correspondence analysis
by the transition formula and by expanding the weighted table into records.
Invariances (row permutations, rescaling by 1e-8 / 1e8, translation by 1e6,
listwise deletion against pre-dropped data, frequency weights against
duplicated records) and about 140 adversarial inputs (each with its
`AnalysisError` code, or a finite JSON-safe result) complete the suite.

The docstring examples run as part of the tests.

## Conventions that are not certain

These choices could not be checked against Stata or SPSS output and are stated
so that a discrepancy can be traced:

1. **Signs** of eigenvectors, factors, functions, pairs and dimensions follow
   OpenEconometrics's rule (largest absolute entry positive); the other packages do not
   document theirs.
2. **Default number of ML factors.** Without `factors`, ML keeps the number of
   SMC-reduced eigenvalues above `mineigen`, capped by the degrees of freedom.
   Stata's own default for `factor, ml` may differ; pass `factors` explicitly.
3. **Eigenvalue table for `ml`.** It lists the sums of squared loadings of the
   retained factors with proportions that sum to one.
4. **Variance of obliquely rotated factors** is the sum of squared structure
   loadings (the variance explained by a factor ignoring the others), which is
   how SPSS's "Rotation Sums of Squared Loadings" are usually explained (for
   example in UCLA's annotated SPSS output); Stata's rotated "Variance" column
   may be defined differently.
5. **Promax with Kaiser normalization** builds the target from the
   row-normalized varimax loadings and regresses it on the varimax loadings in
   their original metric. This is the SPSS variant as reproduced by the R
   package EFAtools (`PROMAX(type = "SPSS")`, validated by its authors against
   SPSS output; Grieder and Steiner 2022 describe the row normalization of the
   target), so confidence here is moderate to high. With `kaiser=False` the
   target is built from the loadings themselves (R's `promax`). Stata's exact
   variant is not documented in a way we could verify. SPSS's own varimax stops
   at a loose criterion, so its promax loadings can differ from ours in the
   third decimal.
6. **Bartlett correction of the factor tests.** Both the independence test and
   the ML model test use the Bartlett multipliers given above. No log
   likelihood, AIC or BIC is reported for ML factor analysis, because Stata's
   normalization of that log likelihood could not be confirmed.
7. **Lower bound 0.005 on ML uniquenesses** (Jöreskog, R). Stata lets a
   uniqueness reach zero; solutions with a Heywood case can therefore differ
   slightly.
8. **Iterated principal factors** use SPSS's defaults (25 iterations, 0.001) and
   keep the number of factors fixed at the first step; Stata's `ipf` tolerance
   may differ. Pass `tolerance` and `max_iterations` for a tighter solution.
9. **`pca_scores` of a covariance-matrix PCA** are centred by default
   (`center=True`, the textbook definition). To our knowledge Stata's `predict`
   uses the uncentred variables unless its `center` option is given; use
   `center=False` for that.
10. **SPSS's initial k-means centres** (`init="spss"`) are implemented from the
    description in the SPSS algorithms (two replacement tests per case).
11. **Split-half** uses the first `ceil(k/2)` items as the first half (SPSS).
    For a negative correlation between halves the unequal-length Spearman-Brown
    formula is applied as written (it depends on `r^2`).
12. **Sequential canonical tests.** Rao's F for correlations k..s uses
    dimensions `p-k+1`, `q-k+1` and the constant multiplier
    `n - 1 - (p+q+1)/2`; the first row therefore equals the overall Wilks test.
    statsmodels' `CanCorr.corr_test` uses a different, asymmetric multiplier.
    The Lawley-Hotelling F is the classical approximation with
    `2(sh + 1)` denominator degrees of freedom (as Stata's `manova`), not the
    McKeon approximation used by SAS and statsmodels.
13. **Heights of Ward's, centroid and median linkage** with
    `metric="euclidean"` are square roots of the Lance-Williams values (SciPy);
    Stata and SPSS report the squared scale (`metric="sqeuclidean"`), and SPSS's
    Ward coefficient is the cumulative within-cluster sum of squares
    (`within_ss`).
14. **Box's M** uses the F approximation of Box (1949), the statistic SPSS
    prints; the chi-square approximation is reported next to it.
15. **Default initial centres of k-means** are the first k cases (`init="first"`).
    SPSS's default is its selection pass (`init="spss"`) and Stata's is a random
    draw (`init="random"`); choose the one you want to reproduce.
16. **Tied dissimilarities.** Every merge joins a closest pair (verified by
    replaying the definition on tied data for every linkage), but when several
    pairs are equally close the choice can differ from SciPy's, Stata's or
    SPSS's, and with complete, average, weighted, Ward's, centroid or median
    linkage that choice can change later merges and heights: the hierarchy of
    tied data is not unique. Single linkage heights never depend on it.
    Centroid and median linkage take the first closest pair in row order; the
    nearest-neighbour chain of the other linkages takes the pair its chain
    reaches first.
17. **Cutting a tree with inversions into k groups** undoes the last k - 1
    merges (always exactly k groups, like SPSS's `/SAVE CLUSTER(k)`); SciPy's
    `fcluster(..., "maxclust")` may return fewer groups there. Height cuts
    follow the cophenetic rule (a merge is below the cut only if every merge
    inside it is), identical to SciPy's `criterion="distance"`.
18. **Iterated principal factors with a communality above one** keep iterating
    and flag a Heywood case; SPSS stops the extraction with a warning. SPSS
    reportedly takes absolute values of negative eigenvalues (Grieder and
    Steiner 2022), whereas OpenEconometrics gives a factor whose eigenvalue turns
    negative zero loadings (only possible when more factors are requested than
    the reduced matrix supports).

## References

- Arthur, D. and Vassilvitskii, S. (2007). k-means++: the advantages of careful
  seeding. *SODA*.
- Bartlett, M. S. (1950). Tests of significance in factor analysis. *British
  Journal of Psychology* 3.
- Björck, Å. and Golub, G. H. (1973). Numerical methods for computing angles
  between linear subspaces. *Mathematics of Computation* 27.
- Box, G. E. P. (1949). A general distribution theory for a class of likelihood
  criteria. *Biometrika* 36.
- Caliński, T. and Harabasz, J. (1974). A dendrite method for cluster analysis.
  *Communications in Statistics* 3.
- de Leeuw, J. (1977). Applications of convex analysis to multidimensional
  scaling. In *Recent Developments in Statistics*.
- Duda, R. O. and Hart, P. E. (1973). *Pattern Classification and Scene
  Analysis*. Wiley.
- Grieder, S. and Steiner, M. D. (2022). Algorithmic jingle jungle: a
  comparison of implementations of principal axis factoring and promax
  rotation in R and SPSS. *Behavior Research Methods* 54.
- Greenacre, M. (2007). *Correspondence Analysis in Practice*, 2nd ed.
- Guttman, L. (1945). A basis for analyzing test-retest reliability.
  *Psychometrika* 10.
- Hendrickson, A. E. and White, P. O. (1964). Promax: a quick method for
  rotation to oblique simple structure. *British Journal of Statistical
  Psychology* 17.
- Jennrich, R. I. (2002). A simple general method for oblique rotation.
  *Psychometrika* 67.
- Jöreskog, K. G. (1967). Some contributions to maximum likelihood factor
  analysis. *Psychometrika* 32.
- Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor
  analysis. *Psychometrika* 23.
- Lance, G. N. and Williams, W. T. (1967). A general theory of classificatory
  sorting strategies. *Computer Journal* 9.
- Mardia, K. V., Kent, J. T. and Bibby, J. M. (1979). *Multivariate Analysis*.
- Murtagh, F. (1983). A survey of recent advances in hierarchical clustering
  algorithms. *Computer Journal* 26.
- Rao, C. R. (1973). *Linear Statistical Inference and Its Applications*, 2nd ed.
- Torgerson, W. S. (1952). Multidimensional scaling: I. Theory and method.
  *Psychometrika* 17.
