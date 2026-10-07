# Saved group prediction

New fits capture bounded typed group state independently of the 400-row chart
preview. Small records are inline (at most 2 MiB); larger maps are immutable
owned SQLite files (at most 256 MiB) with SHA-256 and size checks. Move a saved
result with `oe.export_group_state(result, new_path)`, which copies a disk state
file and returns an updated result. A missing or altered file is rejected.

`oe.predict(clogit_result, dataset)` evaluates the exact conditional inclusion
probability given each retained group's success count and complete alternative
row multiset. `xtlogit(model='fe')` uses the same conditional target. Group rows
may span arbitrary batches or be reordered; values and duplicate multiplicities
must match the retained fitted group. Removing or changing an alternative,
changing successes or supplying an unknown group is an error. Outcomes are
required for complete-group conditioning. This is not an independent row logit.

For normal `melogit`, `meprobit`, `mepoisson`, and `menbreg` fits,
`target='population'` integrates over the saved random covariance.
`target='conditional', random_effects={'_cons': value, ...}` instead holds
explicit supplied effects fixed; values may be finite scalars or numeric column
names. Normal panel RE logit/probit/Poisson uses a separate integrated adapter,
not the PA/GEE inverse link. Full parameter delta gradients include every saved
variance/covariance parameter. AME reduces gradients globally before applying
covariance. Group-response MEM needs a separately declared representative state
and is rejected; fixed `xb` MEM remains supported.

`target='posterior'` for the three existing normal GLMMs uses complete retained
groups and integrates the likelihood-weighted normal posterior. Doubled adaptive
quadrature must settle for both means and all delta gradients. NB posterior and
other panel FE posterior targets are not registered. Population, explicit
conditional and posterior are separate targets with separate definitions.

Absorbed models and linear FE panels use the saved fitted **joint group effect
sum** for conditional means. PPML uses the actual saved log-scale sum and its
offset/exposure. Unknown combinations and historical results without maps are
rejected. Nonrobust fits store orthogonal full nuisance information when there
are at most 1,024 effect levels and their geometry fits the 128 MiB plan; the
response gradient `X_new - P_FE X` includes slope/effect covariance. Robust or
larger fits carry point means only and reject mean intervals and response AME
uncertainty. `xb` and its inference continue to exclude fitted group effects.

`oe.mixed_predict(result, dataset, kind='fitted'/'reffects')` spools input on owned
disk, evaluates one complete top group at a time and returns owned indexed row
Parquet or bounded long group output. Nested child identities include their
parent. BLUPs condition on saved hyperparameters; this output has no full
hyperparameter interval. The existing resident API retains its behavior.

Complete evaluation defaults to 100,000 rows per top group and a 1 GiB input/
output SQLite spool; explicit maxima are 1,000,000 rows and 16 GiB. Projected
batches are at most 65,536 rows. Exact conditional recursion allows at most 128
minority outcomes and 100 million derivative-work units per group, with its
autograd tape planned before allocation. Integration and nuisance geometry have
separate workspace guards. All source columns and indexes are verified by a
second complete replay before success. Scratch files are removed on success,
source change, budget rejection or estimator failure; missing dropped rows
retain their original index and NaNs. Disk throughput and the largest complete
group still bound practicality; no total-process-RSS claim is made.

Tests independently enumerate conditional subsets, numerically integrate normal
population and posterior responses, differentiate every saved parameter, invert
full joint OLS/Poisson information, and solve dense Gaussian BLUPs. They also
exercise replayed fits, split groups, duplicate/MultiIndex rows, nested reused
labels, export/checksums, mutable sources, budgets and cleanup.
