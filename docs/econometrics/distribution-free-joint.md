# Conservative finite-sample simultaneous inference

Eight resident CPU float64 procedures add model-generated randomization tests
and distribution-free confidence families. They return ordinary `TableSet`
objects. All rows, family identities, assumptions, endpoints, original sample
positions, seed/orbit provenance and settings are retained by
`oe.summary_state(result)` / `oe.restore_summary(state)`. The console's table
preview is bounded; a large preview is not a complete export. Settings cells
point to full summary export for metadata exceeding the preview's cell limit.
Every output table supports publication LaTeX.

| Procedure | Required declared sampling model | Finite-sample contract |
| --- | --- | --- |
| `mean_sign_stepdown` | Joint central symmetry of every true-null subvector about the supplied null, independent sampled rows | Common row sign transformations; exact full sign orbit or independent uniform signs with plus-one Monte Carlo calibration |
| `mean_permutation_stepdown` | Exchangeable pooled true-null subvectors at fixed two-group sizes | Full assignment orbit or independently sampled uniform assignments; equal marginal means alone do not suffice |
| `simultaneous_dkw_band` | `sampling_model="iid_marginals"` | DKW-Massart CDF bound over the entire real line, with Bonferroni across marginal distributions |
| `simultaneous_quantile_ci` | `sampling_model="iid_marginals"` | Equal-tail binomial order-statistic inversion for all prespecified column-by-probability lower quantiles; arbitrary ties are conservative |
| `simultaneous_proportion_ci` | `sampling_model="binomial_marginals"` | Clopper-Pearson intervals at per-member level `alpha/m`; binomial marginals may be dependent |
| `multinomial_region` | `sampling_model="iid_multinomial"` | CP box intersected with `sum(p)=1`, including every prespecified zero-count category; exact separate coordinate projections of that region |
| `hoeffding_mean_ci` | `sampling_model="iid_bounded"` and fixed support | Two-sided Hoeffding union bound over bounded means |
| `empirical_bernstein_mean_ci` | `sampling_model="iid_bounded"` and fixed support | Maurer-Pontil empirical variance bound and mirrored sides, Bonferroni across sides and means |

“Exact” describes a sampling law or exhaustive orbit. Coverage/rejection is
conservative, generally unequal to its nominal level. A declaration is a
scientific assumption; the software cannot verify a population's symmetry,
exchangeability, iid sampling or complete category universe from observations.
Dependent sampled rows, survey/other weights, adaptively selected families,
Dataset/out-of-core and GPU routes are outside these contracts. Unsupported
options raise an error; there is no automatic conversion or fallback.

Interpreting the randomization procedures as tests of population means also
requires finite first moments. Without them, a symmetry center or an equal
joint distribution can still define the transformation null, but a population
mean need not exist. The procedures do not turn arbitrary mean-only nulls into
valid sign-flip or permutation designs.

Data-frame procedures select a fixed family, preserve original row positions,
reject numeric strings/booleans/infinity, and require complete
aligned observations. `missing="drop"` explicitly changes the sample to complete
rows. Inference then concerns the prespecified complete-case population; it
does not solve outcome-dependent missingness. Within-row dependence between
coordinates is allowed. Count procedures preserve supplied counts and unique
family labels. A category/count is not silently omitted or merged.
CDF/quantile sample tables also retain bounded scalar index labels; bounded
means retain typed encoded labels. Randomization identifies rows by original
position and retains group identity, independently of arbitrary index labels.
CDF/quantile outcome names `original_position` and `original_label` are reserved
for the saved sample; rename those columns explicitly. Integer observations
outside the exact float64 safe range `[-2**53,2**53]` and wider floating dtypes
are refused before conversion. Bounded support scalars must also convert
exactly when supplied as integers. Count family labels are finite scalars or
nonempty strings up to 256 characters.

For DKW bands, `epsilon=min(1,sqrt(log(2*m/alpha)/(2*n)))`. Complete unique
observed knots record left and right empirical CDFs and bounds. At a knot the
right CDF includes ties and the left excludes them; between knots use the
preceding right CDF, below the first use zero and above the last use one.
These conventions and outer bounds are saved, so the output defines a band
over all real numbers, rather than confidence limits only at observed points.
The bound applies to discrete distributions too.

For quantile probability `q`, the population target is `inf{x:F(x)>=q}`.
Choose the greatest lower rank `L` whose binomial lower tail
`P(Bin(n,q)<L)` is at most `alpha/(2*m)` and the least upper rank `U` whose
upper tail `P(Bin(n,q)>=U)` is at most that level. Here `m` counts **all**
column-by-probability targets. A conservative `64*(n+1)*float64_eps` allowance
only widens borderline ranks. Rank zero denotes negative infinity and rank
`n+1` positive infinity. JSON-null endpoints plus explicit unbounded flags
preserve this distinction from a failed or missing estimate. The empirical
lower quantile is order statistic `ceil(n*q)`; no interpolation is invented.
Tail probabilities, ranks and attained binomial marginal coverage are saved.
Ties/atoms widen population coverage, so continuity is not required.

Proportion intervals reuse native scalar CP/F-quantile kernels, including zero
and all-success boundaries. Lower tails use the reciprocal swapped-df upper
F quantile, avoiding cancellation in `1-small_tail`; endpoints receive one
float64 ULP outward rounding. Multinomial output keeps the raw
CP bounds and the simplex equation. The lower coordinate projection is
`max(rawL_j,1-sum(other rawU))`; the upper is
`min(rawU_j,1-sum(other rawL))`. Combining independently chosen projected
coordinates may violate `sum(p)=1`; the region is the saved box intersected
with that constraint. This is not a Goodman/Sison-Glaz procedure.

Known support for bounded means is prespecified, never estimated from sample
minima/maxima. Hoeffding radius is
`(b-a)*sqrt(log(2*m/alpha)/(2*n))`. Empirical Bernstein uses unbiased sample
variance `s²` and radius
`sqrt(2*s²*log(4*m/alpha)/n) + 7*(b-a)*log(4*m/alpha)/(3*(n-1))`.
Intersect both intervals with the known support. They require no normal/t
pivot and report no fabricated coefficient p-values/degrees of freedom.

The four CDF/quantile/count APIs admit `alpha` in `[1e-8,.5)`, at most 32
selected columns, 10,000 raw rows and 100,000 selected cells. Fixed quantile
and count families have at most 128 members. Quantile probabilities are in
`[1e-6,1-1e-6]`. Integer counts/trials have a 1,000,000-trial cap; multinomial
total is positive and within that cap. Data labels/column names are bounded
scalar identities. Workspace plans reserve copies, masks, sorting, rank and
numeric output buffers before allocation. Other two modules have their own
explicit orbit/work and row/cell limits, saved with each result. These plans
bound named live buffers and operation proxies, not total process RSS or
caller-owned inputs. Summary JSON has the existing 32 MiB export cap and
rejects an oversized export explicitly.

The scientific basis is the original
[Romano-Wolf randomization stepdown construction, section 3.2](https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffd823d949/jasa.pdf),
[Massart DKW theorem](https://doi.org/10.1214/aop/1176990746),
[binomial order-statistic quantile derivation](https://www.stat.berkeley.edu/~stark/Teach/S240/Notes/ch5.htm),
[Clopper-Pearson limits](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm),
and [Maurer-Pontil theorem 4](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf).
Simplex coordinate projections follow directly from the saved single sum
constraint; no vendor algorithm is claimed. Development references and the
frozen protocol test numerical formulas, exhaustive orbits/count coverage,
finite-sample sensitivity and persistence. Source tests, wheel/frozen identity,
native Run/restart and licensed-vendor comparison remain distinct evidence
layers. General model/cluster bootstrap and weak-identification regions remain
under MARKET-191/192; the older normal/joint-t/Hotelling contracts are retained.
