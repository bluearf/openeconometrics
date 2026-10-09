# PLS, Mundlak CRE, Hausman–Taylor and window targets

The original APIs below implement bounded numeric, unweighted, in-memory CPU float64 domains.
They make no blanket Stata/SPSS/EViews parity claim. See the runnable
`docs/examples/panel_prediction_extensions.py` and the independent tests.

`oe.pls(data=..., y=..., x=..., selection='fixed', components=...)` implements
scalar-response PLS1 deflation. Default `selection='cv'` chooses the lowest
held-out MSE, breaking ties toward fewer components; `component_path` is an
explicit candidate list. Every training fold centers and optionally scales X
using its own sample standard deviation when an intercept is fitted; with
`intercept=False`, neither X nor y is centered and X uses training RMS scaling.
y is never scaled. Final
state is refit on the retained training sample. Persisted loadings, coefficients,
fold assignments, candidate failures and scaling support `oe.regularized_predict`
and `oe.regularized_table` after restoration. Prediction is the target; there
are no coefficient standard errors or CI. Missing inputs raise or drop according
to ModelSpec; prediction queries must be complete and preserve their index.
Unidentifiable requested components, excessive work and workspace are refused.
Independent scikit-learn PLSRegression comparisons are development tests only,
alongside independent one-component and full-rank OLS limits.
[Reference algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html).

Current regularized option reconciliation for [#33](https://github.com/bluearf/openecon/issues/33):

| Option area | Current source support | Evidence and limits |
| --- | --- | --- |
| Binomial and Poisson losses | Bounded elasticnet_logit/elasticnet_poisson | [GLM contract](regularized-glm.md); PR159 source/frozen/installed evidence remains separate |
| Weights | Scalar Gaussian and PLS f/a/p fixed/CV empirical-loss routes | [Weighted objective, literal replication and sample/fold normalization](regularized-all-family.md); no survey inference |
| Categories | Typed train-only treatment/one-hot maps and unseen-level refusal | [Saved schema and complete query alignment](regularized-all-family.md) |
| Penalty factors/forced controls | Gaussian and GLM literal encoded-block factors; PLS identified control partialling | [Existing numeric factor proof](next-eight-regularized-prediction.md), [weighted/category contract](regularized-all-family.md); PLS penalties and weighted plugin calibration are explicitly inapplicable/unvalidated |

`oe.xtreg(..., model='cre')` augments numeric predictors with means computed
**after** missing/weight screening from the actual retained panel sample. It
reuses the RE variance-component engine (harmonic mean by default, `sa=True`
for the existing Baltagi–Chang convention). Time-invariant predictors stay in
the original equation; their duplicate mean columns are not added. Means must
be jointly identifiable. `oe.mundlak_test` reports the joint zero-mean-coefficient
Wald chi-square test using the fitted complete covariance. Covariances are
nonrobust or panel/nested-cluster robust; inference is normal/chi-square. CRE
currently rejects categories, weights and Dataset inputs. Saved `cre_predict`
holds fitted means fixed, rejects unknown panels and reports population means;
common predict/margins and panel BLUP are explicitly unavailable for this design.
[Primary CRE/Mundlak reference](https://www.stata.com/manuals/xtxtreg.pdf).

`oe.htaylor_moment` is a separate HT instrument-set estimator with four disjoint
roles: varying_exogenous X1, varying_endogenous X2, invariant_exogenous Z1 and
invariant_endogenous Z2. Endogenous means correlation with the panel effect;
every regressor must remain orthogonal to the idiosyncratic error. The X1 count
must cover Z2, and all pilot/final projections must have full rank. The pilot
within OLS estimates sigma_e² with RSS/(N-G); equal-panel between IV estimates
the invariant parameters. Its residual second moment minus sigma_e²/T_harmonic
estimates sigma_u², constrained at zero with a warning if negative. Final 2SLS
uses quasi-demeaned regressors/outcome and instruments [constant, within X1/X2,
means X1, Z1]. This declared finite-sample moment convention is independently
tested; equivalence to every vendor variance convention is not claimed.
Nonrobust covariance uses transformed RSS/N, or RSS/(N-K) with `small=True`;
robust means panel clustering, with G/(G-1), and additionally (N-1)/(N-K) when
small. Normal/chi-square by default; small uses t/F with N-K or cluster G-1 df.
Balanced/unbalanced panels and calendar gaps are permitted; roles are verified
on the retained sample. No weights/categories/Amemiya–MaCurdy/streaming route.
Persisted `htaylor_predict` reports X beta with no BLUP or common margins.
[HT assumptions and instrument-set reference](https://www.stata.com/manuals/xtxthtaylor.pdf).

Existing `cre`/`xthtaylor` APIs keep their selected-mean and degrees-of-freedom
variance conventions. `htaylor_moment` names the distinct equal-panel moment
convention above; it does not replace `xthtaylor` or advertise vendor equivalence.

Existing `rolling` physical-row forward and backward fixed-window behavior is
preserved. `reverse_recursive=True` explicitly selects retrospective suffix
analysis; panel/calendar modes also use this contract with `reverse=True`:
last cutoff fixed, first cutoff advances, and
the window shrinks to the specified minimum. This is retrospective analysis,
not causal prediction. `calendar=True` uses spans of integer periods, preserving
gaps; dates must be explicitly converted to a meaningful period scale. Panel
ModelSpec windows use pooled entities over those calendar spans (default for a
panel spec); they do not concatenate entity histories or fit one panel as if it
were a multi-panel model. Each estimator screens missing rows independently in
each window. Complete result state, row mappings, covariance and failures are
recorded. `forecast_steps`/future exog are not accepted in these new modes.
[Reverse/calendar terminology](https://www.stata.com/manuals/tsrolling.pdf).

`rolling_predict(spec, data=..., time=..., window=..., horizon=...)` separately
fits/tunes regularized/PLS/local predictive targets through each origin and
scores only the next disjoint span. Each model's training-fold CV is confined
to that training window; this is not a claim of serially independent CV folds.
Physical mode orders rows by complete unique time; optional panel requires
integer calendar spans. Missing evaluation labels do not alter prior training
boundaries; missing='drop' screens evaluation inputs only for scoring. Failed
training/evaluation windows can be recorded. Each window saves its result/state,
training/evaluation physical positions and held-out MSE. Overlapping evaluation
errors are not claimed independent, and no coefficient/pointwise CI is invented.
Both workflows have up-front max_fits and live-buffer plans plus each estimator's
own work budget. Dataset inputs are refused before reads or collection.
