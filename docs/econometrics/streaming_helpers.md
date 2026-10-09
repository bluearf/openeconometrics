# Chunked helper analyses

Pass an `oe.scan(...)` / `Dataset` directly to these procedures. They use the
same native float64 PyTorch kernels, table layouts, confidence intervals and
LaTeX export as the resident-data path. They never collect all source rows.

| Procedure | Dataset support |
| --- | --- |
| `ttest` | One sample, paired differences, independent samples; pooled/Welch, Levene and effect sizes |
| `sdtest` | One-sample chi-square; variance-ratio F; exact mean/median/trimmed-mean robust tests |
| `oneway` | Descriptives, ANOVA, exact median Levene, Bartlett, Welch/Brown–Forsythe, native post hoc comparisons |
| `anova` | Between-subject factorial ANOVA/ANCOVA; Type I/II/III; interactions/slopes; estimated marginal means and pairwise comparisons |
| `correlate` | Pearson, exact Spearman midranks and Kendall tau-b; pairwise/listwise missingness, exact pair counts, p-values and Fisher-z intervals |
| `describe` | Grouped moments, fractional percentiles, Stata/HAVERAGE definitions, missing counts, skewness/kurtosis and mean confidence intervals |
| `rm_anova` | Complete repeated subjects, between/within effects, multivariate contrasts, Mauchly and Greenhouse–Geisser/Huynh–Feldt corrections |
| `manova` | Type III multivariate tests, univariate tables, Box M; blocked multi-outcome QR |
| `pcorr` | Partial/semipartial correlations with controls through blocked TSQR |
| `pca` | Correlation/covariance matrix PCA, native component selection and loadings |
| `factor` | Native `pf`, `ipf`, `pcf`, `ml`, rotations and score coefficients |
| `factortest` | KMO, variable MSA/SMC and Bartlett sphericity |
| `pca_scores`, `factor_scores` | Replayable score Dataset, preserving row order and missing-row positions |

```python
import openecon as oe

source = oe.scan("study.parquet")
test = oe.ttest(source, "wage", by="treated")
variance = oe.oneway(source, "wage", "region", posthoc=["tukey"])
ancova = oe.anova(source, "wage", ["region", "treated"], covariates=["age"])
components = oe.pca(source, ["income", "hours", "assets"], components=2)

# The small result tables use the existing paper-table exporter.
latex = ancova.to_latex()

# Observation-level output stays in bounded blocks.
scores = oe.pca_scores(components, source)
for block in scores.iter_batches():
    consume(block)
```

Statistical moments use coordinates anchored at the first observed value and
stable Chan reductions. Factorial analyses and partial correlations use blocked
QR of the design/outcome and perform hypotheses on the resulting small factor;
they do not solve normal equations. Pearson pairwise reductions use compensated
masked cross products after anchored centring.

Exact group medians and trimmed means use an owned SQLite file. Its cache is
bounded and its storage is removed on success or failure.
`OPENECON_SCRATCH_DIRECTORY` selects that local scratch directory. The analyses
keep factor/group metadata within explicit structural budgets; they refuse an
oversized model before allocating its large design geometry. Estimated marginal
means also reserve their Cartesian cell/covariance geometry, and post hoc tests
reserve all pair buffers and retained result tables before allocation. The resource plan
is an estimate of live statistical buffers, not a process-RSS ceiling or a bound
on a third-party source reader.

Every complete replay pass checks selected-column content, including dropped
rows, against the first pass. Score sources capture the source digest on creation
and check every subsequent complete replay; callers must exhaust a score pass
for its final integrity check. A preview alone does not perform that final check.

Spearman ranks each complete pair on disk; Kendall counts concordance using an
owned disk Fenwick tree, with exact tie corrections and asymptotic inference.
Percentiles use external order statistics with the existing Stata and HAVERAGE
interpolation rules. No approximate ranks or quantiles are substituted.

Repeated measures enforce typed subject/cell uniqueness and complete within-cell
sets in owned SQLite storage. They replay subject contrasts into a small TSQR
factor instead of retaining the subject matrix. MANOVA reduces multiple outcomes
jointly through QR, with bounded cell moments for Box M. Factor/cell/SSCP/result
geometry is checked before allocation. The original resident layouts and formulas
remain shared with these adapters.
Grouped descriptive plans also charge the requested statistic/percentile column
count and retained result cells before creating the corresponding group state.

The reproducible million-row probe (internal evidence excluded from this public snapshot)
records actual Parquet size, two/three source passes, kernel digest, wall time,
fresh-process peak RSS and owned scratch removal. It measures CPU source kernels;
its workspace budget excludes imports/readers and is not a total-RSS promise.
Other multivariate and resampling procedures retain their separate contracts. Existing
scientific restrictions apply: unsupported Type IV empty-cell hypotheses,
confounded factorial designs, zero variances and inadmissible factor extraction
are reported as errors rather than approximated.
