**Delivery status · 10 October 2026.** The bounded proper-MNIW public surface is now implemented in the current integration branch: all 29 lazy root exports, the dedicated `bayesian_var_public` manifest, typed restore routes, full TableSet/JSON adapters, capabilities and 114 complete BVAR editor objects. The current catalog preserves all 1,251 prior whole objects and contains 1,365 objects, 85 full templates and 68 signature suffixes; its lossless encoded size is 478,361 bytes under the unchanged 500,000-byte cap. These are source integration and targeted validation results, pending final merge and delivery gates.

The original fixed 1,024-experiment prior-predictive study completed all 2,616 preregistered gates. Its original transport failure, adopted saved fit, fixed same-worker pause/continue history and original unobserved OS exit status remain preserved. Current targeted protocol-2 tests pass on Python 3.11 and 3.13: 122 collected parent cases, 366 outer phase records, 25 nested reports, 391 total phase records and 147 native JUnit reported execution units per minor. Separate catalog/capability validation passed all 629 actually selected cases; the original mistaken five-case registration text remains qualified in its receipt. These calibration and targeted receipts are separate from the original-author result below and do not establish final full-SDK or installed BVAR acceptance.

The three fixed original-author comparisons passed under GNU Octave 8.4.0 on Ubuntu 24.04 in [hosted run 38018613824](https://github.com/bluearf/openecon/actions/runs/38018613824) (attempt 1). The complete unchanged Chan2020 `prior_NC.m` and `forecast_BVAR_NCP.m` ran with `nsims=burnin=0` for 2, 3 and 4 series, four lags and an intercept. All seven matched full arrays (design, response, design crossproduct, posterior precision, location, innovation scale and row scale), degrees of freedom, full priors and saved own states passed the preregistered elementwise `atol=rtol=2e-12` checks. The row scale is explicitly wrapper-derived `inv(KA)`. The [complete original artifact](https://github.com/bluearf/openecon/actions/runs/38018613824/artifacts/11656559565) retains native MAT arrays, complete own JSON states, source/runtime identities, ambient RNG snapshots and original logs. This is original-source execution under Octave; no MATLAB or licensed-vendor execution or broader endpoint parity is claimed.

Installed BVAR Run/save/full Quit/cold replay, fresh exact-head/current-base full hosted SDK gates on both Python 3.11 and 3.13, merge and child-only tracker readback remain pending. The prior local full-SDK attempt intentionally stopped at the proved report-ledger capacity defect remains failed/incomplete; the historical 84b SDK failures also remain failures. The lossless capacity repair requires fresh full hosted validation and does not change scientific tolerances or resource caps.

The source proposal below is retained verbatim as historical design and admission evidence; its earlier source-only/pending wording and local-gate plan describe that frozen stage, not the current delivery status.

---

# Dedicated proper MNIW BVAR public integration — source proposal

This additive source proposal exposes the existing six Bayesian VAR APIs and
their fourteen immutable typed declarations. It adds explicit exact-schema
restore routes, complete named tables and bounded complete JSON output. The
original eighteen implementation/test/document files, 6,301 registered physical
source/runtime pins, earlier 53-test evidence and completed original 1,024-study
packet are preserved. No proposed module has been imported, no new test has been
collected, and no new fit, random draw or installed application Run has occurred.

The family is `bayesian_var_public`, with literal `EXPORTS` and empty
`ESTIMATORS`. Current main 769f5b5618b6d03f88cebfd999286995373678b6 and frozen
PR213 2a3228bec1ccc80a907cf3b46f5a9cf75e5e05a4 provide separate family/lazy export
and TableSet patterns. Root must apply any shared registration, root exports,
capability/catalog or CI edits against the actual integration base. PR213 is not
assumed merged. No generic ModelSpec, ResultBundle, common predict or common
forecast registration is invented for these distinct posterior states.

## Methods and complete state

`bayes_var`, `bayes_var_predict`, `bayes_var_contrast`, `bayes_var_draws`,
`bayes_var_forecast` and `bayes_var_irf` point directly to their unchanged original
implementations. Data are resident finite CPU float64, 2–4 ordered series,
1–4 lags, optional intercept, explicit consecutive integer time and exact typed
original index/permutation. The proper full MNIW prior is required. Its mean,
row scale, innovation scale and degrees of freedom must have exact original
geometry. `bayes_var_prior_restore` validates them with the original strict SPD
factorization; there is no jitter, prior repair or empirical default.

`bayes_var_restore` selects exactly one of six schemas. Explicit
`bayes_var_predict_restore`, `bayes_var_contrast_restore`,
`bayes_var_draws_restore`, `bayes_var_forecast_restore` and
`bayes_var_irf_restore` additionally require the named schema. Full source,
prior, posterior algebra, every cached primitive/draw/path, whole covariance,
typed labels and digests are validated. Resealing a digest does not permit
contradictory algebra or scientific arrays.

Cold draw/forecast replay reconstructs the original transforms from saved
Bartlett factors, coefficient standard normals and innovation standard normals.
It performs no fit, optimization or random generation. Recorded seeds/runtime
are descriptive; this original portable replay contract does not authenticate a
fresh generator stream against the seed. The CF private-generator replay
protocol is a separate method and is not substituted here.

`bayes_var_tables(state)` returns a dedicated TableSet with every nested numeric
array, every covariance cell and all scalar/unavailable fields. Tensor axes use
zero-based coordinates; the complete typed terms, declared series order,
calendar/index and all coordinate descriptors persist in `complete_state` attrs.
Object cells preserve signed64 periods and integer index values exactly even
beside floating series. The current console converts integers above2**53 to
exact decimal strings before JavaScript rendering.
Higher-dimensional arrays flatten their leading axes into tuple row coordinates
while retaining every last-axis column. Exact matrix-t coefficients, scalar
Student-t contrasts, one-step vector-t scale/covariance and finite-draw MC
summaries retain their owning field names and original laws/labels. No
frequentist SE/t/p is added. Unavailable means/second/fourth moments stay None;
empirical covariance and MC error never replace infinite moments. Unstable draws
remain retained.

`bayes_var_state_json(state, indent=None)` is the complete-state UTF8 route.
Indent is None or 0–4, and the state has a conservative 32 MiB escaped-output
bound. It does not export a preview or diagonal covariance subset. Original
typed model serialization/copy methods retain their original checked contracts;
arbitrary original Pydantic formatting kwargs do not acquire the new bounded
UTF8 promise merely by being public.

## Additional resource admission

The original numerical budgets/formulae remain unchanged. New restore/table/JSON
operations admit the original full replay numerical/source plan, twelve original
operation work plans, bounded traversal/labels/formatting work, complete typed
primitive/attrs copies, every table coordinate/value/frame and escaped
JSON/UTF8/encoder lifetimes before replay or output allocation. The state must
fit its original inherited max_work/max_bytes and the task workspace limit.
Original depth32/2,000,000 metadata-node bounds also apply. The conservative
escaped output bound is 64 times the original resident metadata estimate, with
three output/encoder lifetimes in the combined plan. This is a declared live
buffer estimate, not total process RSS or private allocator/BLAS memory.

A near-budget state can remain valid for its original numerical API while
refusing this extra table/export operation. The adapter does not enlarge a
budget, change a numerical admission rule or omit fields to make it fit.

## Installed Run and complete cold readback plan — pending

Use a bounded explicit 24-row/two-series/p1/intercept prior with draws8/H2,
declared string index, typed period column and finite deterministic input values.
Save all six states, all table fields and complete JSON. Each actual Run displays
explicit frames. The console's current native preview is 50 rows by 30 columns,
at most20 outputs. Iterate every full frame in complete 50x30 row/column batches,
log each field/shape/batch coordinate and finish with exact saved table inventory.
Split into Run groups of at most20 outputs. The full9x9 joint covariance,6x6
coefficient covariance,8-draw primitives, recursive paths and IRFs fit small
individual tables; scalar/source/index fields may still require row batches.
Native screenshots/readback must cover every batch, not infer full display from
one preview. TableSet itself is not assumed to become a complete native grid.
The current console also bounds individual string displays to500 characters.
The declared native case uses shorter labels; long supported saved strings need
explicit complete character segments/readback before claiming full native cells.
Whole Python tables and complete typed JSON retain those strings unchanged.

After saving, perform full Quit and verify every application/backend PID exits.
Reopen the installed app, restore all six complete states, and re-display the
entire inventory with fitting, RNG and generators forbidden. Verify source index,
prior, full cross-covariance, all primitive draws/stability, paths/IRFs, None and
MC labels, complete typed JSON and every displayed batch. A restarted process,
saved revision readback and complete cell evidence are separate acceptance
receipts. No native execution is claimed by this document.

## Existing evidence and remaining delivery gates

The existing ROOT complete original1024 readback reports statistical acceptance
and preserves all6301 source/runtime pins, all1024 phase hashes, all23572 original
inventory entries, all2616 gates and all8 negative controls. That receipt
explicitly says the original OS exit status was not observed independently;
the original reported protocol exit0 remains qualified. This proposal copies
those exact existing evidence anchors. It neither reruns the study nor freshly
audits every original raw scientific packet. The earlier53 tests are existing
evidence, not new tests of these additions.

Before MARKET-734 delivery: apply actual-current-main shared integration;
measure/preserve the complete old and114 new logical catalog objects losslessly;
run the original appropriate suite and new lifecycle/resource/cold-replay tests
under an authorized numerical window; preserve exact author/oracle source
distinctions; obtain full installed Run/Quit/reopen evidence; complete exact
current-head/base local and hosted gates; merge; perform fresh child-only Done
readback. No executed MATLAB or licensed vendor reference is inferred from
available author code. Parent MARKET-362 and separate MARKET-162 scopes are not
closed by this child or its calibration study.
