# Discriminant selection and saved multivariate hypotheses

These contracts extend the existing multivariate methods with four selection
options and four MANOVA/repeated-measures input or hypothesis domains. They
are the acceptance scopes of MARKET-521–528. MARKET-185 remains open for other
options; completed earlier scopes are not counted again.

## Discriminant selection

`discrim_stepwise(data, group, columns, method=..., include=..., weights=...)`
uses one complete-case sample over the group, **every** candidate and the
optional frequency count. Adding/removing a predictor never changes that
sample. Included predictors stay protected. All candidates must have a
full-rank pooled within-group covariance before screening begins.

| Direction | Decision |
| --- | --- |
| `forward` | Start with the included predictors and enter the strongest eligible singleton when its reference p is below `p_enter`. |
| `backward` | Start with all candidates and remove the weakest unprotected singleton when its reference p exceeds `p_remove`. |
| `stepwise` | Try removals before the next entry, allowing earlier choices to be revised. Repeated states or exhausted iteration/work budgets do not count as successful convergence. |
| Frequency input | Apply the same three paths to exact integer count moments without expanding rows. Retain separate physical, missing, zero-count and total-frequency accounting. |

For retained set S of k predictors and an additional singleton j, let W and
T be the within-group and total SSCP matrices on the **fixed** sample. The
conditional residual sums are

```
E = W[j,j] - W[j,S] solve(W[S,S], W[S,j])
R = T[j,j] - T[j,S] solve(T[S,S], T[S,j])
rho = E / R
F = (1/rho - 1) (N - G - k) / (G - 1)
```

For removal, S contains every currently selected predictor except j. Thus the
denominator degrees depend on the conditioning set, not just N-G. Candidate
ties use the declared input order. Complete candidate diagnostics and the
accepted path are retained.

The reference F distribution describes a fixed conditional comparison under
its Gaussian assumptions. Its tail value is used as a **screening rule**;
searching among predictors does not preserve an ordinary post-selection test.
Selected outputs therefore do not report the naive canonical/equality/Box M
inferential tables of an unselected fit. Training classification is
resubstitution; it is not full-pipeline cross-validation. An empty selected
model is valid and predicts its saved class priors.

`discrim_stepwise_predict(result, data)` validates the saved selected state
and returns every input row with its original index. Missing required
predictors remain missing. It does not refit or silently apply an unselected
model. Counts must be finite nonnegative integers, with exact total at most
2^53; zero counts carry no statistical weight. Analytic/probability/survey
weights, predictor blocks, summary-input selection, nested LOO and calibrated
post-selection inference are outside this wave.

## One-way MANOVA input

`manova_oneway(data, y, group, weights=...)` accepts resident one-factor data
and optional integer frequencies. `manova_summary(group_means,
group_covariances, counts)` accepts explicitly ordered group means, unbiased
sample covariance matrices and counts. The covariance convention is n_g-1.
Summary covariance matrices must be symmetric PSD and compatible with their
counts; the pooled error matrix must have adequate rank. No synthetic rows
are constructed from summaries.

For N=sum(n_g), the within SSCP is E=sum((n_g-1) S_g), and the between SSCP is
H=sum(n_g (mean_g-grand_mean)(mean_g-grand_mean)'). The procedures retain full
group moments, pooled covariance, H/E, univariate tests and four multivariate
tests. Their sufficient coefficient/error geometry supports saved general
hypotheses. Frequency counts give the literal independent-observation
calculation; arbitrary duplicates are not a justification for Gaussian
sampling inference. Multifactor weighted models and survey designs are not
covered by this one-way adapter.

## General saved MANOVA and RM hypotheses

`manova_contrast(result, L, M=..., null=...)` tests predeclared separable
matrix hypotheses L B M = C. L has full row rank over the saved design
coefficient order; M has full column rank over the saved outcome order. C is
the complete null matrix, with zero as its default. Resident and replayable
Dataset MANOVA fits retain bounded coefficient, bread and error geometry.
MANCOVA coefficients and bread use the original covariate coordinates,
including the intercept transformation from internal centering.

For coefficient bread V=(X'X)^-1 and residual degrees nu,

```
D = L B M - C
H = D' solve(L V L', D)
E_projected = M' E M
Cov(flatten(L B M)) = (L V L') ⊗ (M' (E/nu) M)  [row-major]
```

Every estimate and covariance entry is retained in row-major target order:
each L contrast, then every M response transform. For a column-major vector
the equivalent covariance reverses the two Kronecker factors. The reported
table labels make this ordering explicit. Marginal t statistics and intervals use nu degrees of
freedom. They do not claim simultaneous familywise coverage. Full target
covariance is admitted against dimension/workspace budgets before allocation.

`rm_mtest(result, L, M=..., null=...)` applies the same predeclared hypothesis
to existing `rm_anova` saved original-cell geometry. L follows the recorded
between-design coefficient order; M follows the saved within-cell order.
The residual degrees are the number of complete subjects minus the between
design rank. Under independent Gaussian subjects and one common unrestricted
cell covariance, the multivariate test does not need sphericity. A scalar
target agrees with `rm_contrast`; a rank-one between contrast and multiple
responses yields the corresponding Hotelling test.

Both APIs reject invalid saved dimensions, sample/degrees relationships,
checksums, ordering, rank, nonfinite geometry or unresolved covariance.
Legacy RM states contain rounded original-cell coefficients. If a large common
response level makes a small transformed target unresolved, `rm_mtest` also
refuses with `unresolved_precision`; rescale/refit a stable response basis.
They do not repair covariance matrices or replace a requested hypothesis
with a pseudoinverse-based one. Hypotheses are assumed to have been specified
independently of observed outcomes; the software cannot verify this intent.

The existing multivariate conventions remain explicit:

| Test | Reported F convention |
| --- | --- |
| Pillai | Exact when min(response rank, hypothesis rank)=1; otherwise approximation. |
| Wilks | Rao F, exact for the supported rank-one/rank-two reductions, otherwise approximation. |
| Hotelling–Lawley | Existing classical F approximation, exact when the smaller rank is one; it differs from the McKeon approximation used by Statsmodels for higher ranks. |
| Roy | Exact at rank one; otherwise the reported F is an upper bound. |

No arbitrary nonseparable vec(B) hypothesis, selected contrast inference,
dependent/weighted RM, incomplete-subject extension or vendor parity is
asserted. Source numerical tests, packaged runtime, actual native Run,
complete saved-result reconstruction and application history after a full
Quit/reopen are recorded separately in the delivery evidence.

Primary definitions: [IBM discriminant inclusion rules](https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=command-inclusion-levels-discriminant),
[SAS STEPDISC](https://support.sas.com/documentation/cdl/en/statug/63347/HTML/default/statug_stepdisc_sect002.htm),
[Stata MANOVA postestimation](https://www.stata.com/manuals/mvmanovapostestimation.pdf)
and [Statsmodels multivariate linear-model implementation](https://www.statsmodels.org/stable/_modules/statsmodels/multivariate/multivariate_ols.html).


Resource admission is explicit: selection permits at most 100,000 resident
rows, 32 candidates, 32 groups and 128 steps, with a 2-billion work bound.
MANOVA options permit at most 100,000 resident rows, 64 groups, 128 design
columns/outcomes and 256 joint target cells, with a 250-million work bound.
Saved Dataset geometry does not require reloading observations. Integer
counts and total frequency cannot exceed 2^53; declared memory availability
can impose tighter bounds before allocation.
