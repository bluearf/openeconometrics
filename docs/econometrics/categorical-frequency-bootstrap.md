# Full-refit frequency CATREG prediction bootstrap

`catreg_nominal_fweight_bootstrap` and `catreg_ordinal_fweight_bootstrap`
quantify the finite-replicate variability of fixed-query predictions after
refitting the entire adaptive optimal-scaling estimator. The outcome must be
numeric. Nominal predictors may be mixed with explicitly numeric predictors;
the ordinal method also accepts declared ordinal predictors and requires an
explicit category order for each of them.

```python
result = oe.catreg_ordinal_fweight_bootstrap(
    data, "outcome", ["rating", "income"], frequency="count",
    queries=queries,
    scales={"rating": "ordinal", "income": "numeric"},
    orders={"rating": ["low", "medium", "high"]},
    reps=19, seed=83, n_starts=2, maxiter=100,
)
full_json = oe.summary_state(result)
replayed = oe.catreg_fweight_bootstrap_restore(full_json)
```

## Sampling target and compressed representation

Let the retained complete physical rows have positive integer counts
`f[0], ..., f[n-1]`, and let `F=sum(f)`. A replicate samples the literal `F`
repeated cases with replacement. Its compressed count vector follows

`C ~ Multinomial(F, f/F)`.

The implementation draws conditional binomials using a private seeded CPU
generator. At physical row `i`, the trial count is the remaining sampled
total and the probability is `f[i]` divided by the remaining **original**
integer count sum. The last physical row receives the sampled remainder.
The numerical and sampling buffers depend on physical rows and replicates;
no allocation contains `F` repeated observations. The sampling law is neither
equal-probability resampling of physical rows nor a survey-weight bootstrap.
Native generator rejection loops do not provide a strict wall-clock limit.

All supplied counts are validated before feature-missing rows are dropped.
Zero counts are excluded before inspecting their features. The law uses the
retained complete-case count total; the complete baseline state preserves
original source positions, zero rows and missing rows. Query indexes may repeat;
query positions are their integer offsets and every query must be complete.

## Full refits and empirical uncertainty

Every declared replicate refits numeric normalization, category quantifications,
signed ordinal monotone effects, regression coefficients, all declared starts
and the convergence ledger. A zero count in a replicate excludes that physical
row from its fit. A replicate that omits an original category, lacks the required
physical sample, loses rank, degenerates or fails to converge is a final failed
draw. It is never replaced or redrawn.

Predictions are aligned by the same raw queries and the original numeric
outcome units. Quantification sign and normalization changes cancel through
the fitted coefficient-times-transformation. The methods do not pool adaptive
coefficients or treat learned quantifications as fixed regression regressors.

If every replicate succeeds, the result retains the complete joint empirical
prediction covariance with denominator `reps-1`, including off-diagonal terms.
Each query also has the bootstrap mean, empirical standard deviation and
percentile endpoints. Endpoints use linear interpolation at
`(1-confidence)/2` and `(1+confidence)/2` of the finite stored draws.
These are empirical finite-replicate intervals. They do not establish exact
coverage, particularly near ordinal pooling boundaries, zero effects or
competing local solutions. There are no ordinary coefficient standard errors,
p-values, simultaneous bands, BCa/studentized intervals or vendor-parity claim.

`failure="raise"` stops at the first failed draw and exposes its count ledger
and bounded failure record on the exception. `failure="record"` completes
exactly the declared number of draws. If any failure is recorded, **all**
covariance and percentile-interval tables are absent. Successful predictions
and complete fits remain available for inspection, without surviving-draw
inference.

## Persistence and semantic replay

`oe.summary_state` saves every table, the complete baseline calibration, every
sampled count vector, every successful full calibration/trace, fixed raw queries,
original-unit predictions, fitting controls and failures. Table previews are
not complete persisted results. Generic `oe.restore_summary` preserves complete
JSON and LaTeX; the specialized restore additionally verifies the estimator.

`catreg_fweight_bootstrap_restore` reconstructs the private-seeded count law,
checks every successful saved calibration with the weighted CATREG numerical
validator, confirms its exact raw rows/counts/order/controls, rederives its query
predictions and rebuilds all empirical covariance, percentile and ledger tables.
The optimizer is never rerun. Recorded optimizer failures are declarations of
the original run; replay checks their geometry and withholding consistency but
does not rerun failed optimization. Checksums detect accidental corruption;
numerical replay independently rejects resealed inconsistent predictions or
calibrations. Checksums are not authenticated signatures.

## Input and resource boundaries

Both training and query inputs are resident pandas DataFrames. A required
distinct `frequency` column contains finite nonnegative exact integers, excluding
booleans. Counts and their supplied total are at most `1,000,000,000`; zeros are
admissible and excluded. Training input has at most 3,000 physical rows, with
at least `predictors+3` retained positive complete physical rows. At most 11
predictors, 32 retained categories per categorical predictor, 1–64 complete
queries and 2–199 replicates are supported. Defaults are 19 replicates, two
starts and 100 iterations. Confidence lies in `[0.5, 0.999]`.

CPU float64 is explicit under a different caller Torch default device. Every
fit preserves the weighted CATREG domain and rank/convergence refusals. Cumulative
work includes all planned full starts/iterations, sequential sampling, query
projections and complete output serialization. Saved replay admits the combined
work of every successful numerical calibration before replaying the loop.
Named memory plans include all retained full calibration strings, count ledgers,
one live fit/replay workspace and joint prediction outputs. Full escaped-state
bounds are admitted before fitting/sampling and limited to 32 MiB. `max_work`
is at most 300 million; `max_bytes` is at most 128 MiB and also respects the
current global/task workspace budget. These implementation boundaries do not
bound caller DataFrames, unrelated input columns, allocator overhead or process
RSS. Valid option maxima cannot always be combined within the cumulative budget.

Dataset/streaming, analytic/survey weights, residual/cluster resampling, learned
categorical outcomes, spline predictors, supplied draws and GPU devices are
outside this bootstrap contract.

## Independent checks and sources

`tests/test_categorical_frequency_bootstrap.py` checks every nominal draw against
fresh NumPy weighted dummy regression and literal-case expansion. Ordinal draws
are compared with separate increasing/decreasing SciPy constrained regressions,
including pooled categories and negative effects. Count moments are compared
with the multinomial formula; seeded sampling, private RNG, zero/missing alignment,
failure withholding, full JSON/LaTeX replay, resealed tampering and preallocation
budget guards are checked independently.

The sampling/percentile definitions follow IBM's
[simple case resampling specification](https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=bootstrap-sampling-subcommand-command)
and [percentile interval description](https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=bootstrapping-).
The private native binomial primitive is documented by
[PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.binomial.html).
The API estimates full-refit fixed-query prediction uncertainty; IBM CATREG's
separate prediction-error resampling option is not claimed equivalent.
