# Independent-study meta-analysis

The six public helpers in [the runnable example](examples/meta_analysis.py)
implement four staged workflows (MARKET-194–197). They consume complete
independent study summaries, on CPU float64. They do not collect Dataset
sources, infer dependence, drop missing studies or automatically encode
moderators. Each selected column must exist once; optional study IDs must be
unique. Input tables are preserved. This is bounded method coverage;
MARKET-174 remains open for dependent/multilevel/multivariate models and
additional bias-adjustment workflows.

## Effect preparation

`meta_effectsize(data=..., measure=..., columns=..., study=...)` maps these
explicit roles to numeric columns:

| Measure | Roles | Effect and sampling variance |
| --- | --- | --- |
| MD | m1, sd1, n1, m2, sd2, n2 | mean difference; sd1²/n1 + sd2²/n2 |
| SMD | same | Hedges g = exact gamma-ratio J × pooled-SD standardized difference; LS variance 1/n1 + 1/n2 + g²/[2(n1+n2)] |
| OR | a,b,c,d | log(ad/bc); 1/a+1/b+1/c+1/d |
| RR | a,b,c,d | log[a/(a+b)] − log[c/(c+d)]; 1/a−1/(a+b)+1/c−1/(c+d) |
| ZCOR | r,n | atanh(r); 1/(n−3) |

Binary roles are events/non-events in groups 1 and 2. Zero cells reject by
default. Explicit `zero='add_half'` adds .5 to every cell **only in studies
containing a zero**. Empty groups, double-zero events and double-zero
non-events reject. Continuous summaries require positive SD and integer n>=2;
correlations require |r|<1 and integer n>3. SMD's gamma-ratio calculation is
bounded to combined df<=1e7. Preparation returns yi/vi/se, normal analysis-scale
CIs, original-scale effects/CI endpoints, correction flags and a sample hash.
It does not label a back-transformed SE as if it were an analysis-scale SE.
Formulas/conventions follow the author's [effect-size reference](https://wviechtb.github.io/metafor/reference/escalc.html).

## Pooling, regression and uncertainty

`meta_pool` fits an intercept; `meta_regress` accepts explicit numeric
moderators and an optional intercept. Known vi>0 and independence are caller
assumptions. GLS weights are 1/(vi+tau²). `common` fixes tau²=0;
`DL` uses the residual Q/projection-trace moment estimate;
`ML` and `REML` profile the normal likelihood (REML adds the design log
determinant). Moderators are centered/scaled for fitting and full covariance
is transformed back, including all intercept cross-covariances.
The scaled weighted Gram matrix must have a condition ratio below 1e12;
ill-conditioned models reject before coefficient covariance is computed.

The variance solver normalizes by median vi, compares zero with every
positive-to-negative score root bracketed by an 81-point bounded logarithmic
grid, and requires a resolved upper tail and KKT score. Each root has 100
iterations. An unresolved fit raises `variance_not_converged`; the finite
grid is not a global-optimization proof for arbitrary pathological data.
The reported restricted likelihood uses the design-normalized constant
log|X'X|; covariance and likelihood come from sampling variances, rather than
an unrelated residual-sigma estimate.

`inference='z'` uses GLS covariance and normal limits. `hksj` multiplies it
by final weighted RSS/(k-p), uses t(k-p), and uses F(m,k-p) for joint moderator
tests. `modified_hksj` floors that multiplier at one. Degenerate HK
uncertainty rejects. Q uses vi alone, with k-p df. Common-model I² comes from
Q; random-model I² is 100 tau²/(tau²+typical vi), with typical vi=(k-p)/trace(P0).
Full coefficient/covariance/study/heterogeneity/test tables and the complete
prediction state are retained. These conventions follow
[the author's model reference](https://wviechtb.github.io/metafor/reference/rma.uni.html).

`meta_predict` reads a fitted result or JSON-restored versioned state with
an integrity checksum and positive-definite full covariance. It returns
conditional mean CIs and approximate **latent true-study effect** PIs with
variance x'Cov(beta)x+tau², using the fitted normal or t(k-p) convention.
Future sampling error and tau² estimation uncertainty are excluded; this
does not claim exact prediction coverage. See
[the author's prediction reference](https://wviechtb.github.io/metafor/reference/predict.rma.html).
The checksum detects accidental state drift, not malicious rewriting.

## Study diagnostics and displays

`meta_diagnostics` refits the chosen pooling model for every deletion,
every cumulative prefix and optional subgroup. Input row order is explicit
by default; an `order` column must be complete, numeric and distinct. All
planned fits remain in output, including failure codes/null results, and
planned/completed/failed denominators. The singleton cumulative prefix
cannot estimate heterogeneity and is retained as a failed fit. Group/order
roles are included in the diagnostic sample hash.
See the author's [leave-one-out](https://wviechtb.github.io/metafor/reference/leave1out.html)
and [cumulative](https://wviechtb.github.io/metafor/reference/cumul.html) definitions.

Classical Egger uses weighted yi~1+se with multiplicative dispersion, testing
the se slope against t(k-2); it is equivalent to the standardized-effect
regression intercept. It requires >=10 studies and identifiable SE variation.
Unavailable/degenerate inference is recorded. Asymmetry does not establish
publication bias, and the intercept is not a causal bias correction. See
[the author's asymmetry reference](https://wviechtb.github.io/metafor/reference/regtest.html).
`meta_plot` returns editable/saved PlotSpecs. Forest plots show all sampling
CIs and the pooled mean CI. Funnel plots show effects, SE and pooled-center
sampling-normal reference bands; SE increases upward, explicitly labeled.
The shared renderer and publication LaTeX support finite data-coordinate
line segments for these bands.

## Limits and evidence

At most 2000 studies, 30 moderators (32 numeric roles), 200 diagnostic
studies and 128 MiB estimated numerical workspace. Variances are restricted
to [1e-150,1e150] with a maximum ratio of 1e12; effects to magnitude 1e75.
These are allocation/numerical guards, not process-RSS or GPU benchmarks.
The native runtime supplies its separate overall resource controls.

`tests/test_meta_analysis.py` checks author BCG displays and independent
NumPy QR/SciPy likelihood and tail oracles, every study weight, full
covariance, inference df, PI, variance boundaries, scaling, missing/zero/rank
guards, state restore and every diagnostic refit. SciPy is a development
oracle, never an estimation dependency. `scripts/validate_meta_coverage.py`
predeclares two seeds × four equal-vi independent-normal cells × 400 runs:
REML/HKSJ mean-CI coverage gate .91–.99, zero failed fits, failures counted as
misses. Its saved receipt covers 3200 runs only. Unequal-vi/weak-moderator/
selection/dependence simulation coverage and external vendor execution are
not established. Source, frozen runtime, installed UI and restart receipts
are separate from public release delivery.
