# Dependent effects with a declared sampling covariance

`meta_dependent` fits complete, resident CPU float64 effect estimates whose known sampling covariance is supplied as a labelled positive-definite DataFrame. Gaussian sampling errors and independent study blocks are assumed for common-model normal and residual-Q references. Random-effect inference plugs in the fitted variance component and is approximate. Effects may depend within a study; different studies must be independent. The effect identifiers must match both covariance axes exactly. The retained study blocks, original row positions and labels, numeric design and outcome are recorded in the portable saved state. This is a focused extension of independent-study meta-analysis.

MARKET-629–636 cover three fits and five saved-fit procedures: common GLS, effect-level heterogeneity, study-level heterogeneity, study-robust inference, coefficient contrasts, joint mean prediction, joint future latent-effect prediction and whole-study deletion diagnostics. The broader MARKET-174 remains open.

## Three covariance models

Let `S` be the declared sampling covariance, `X` the numeric moderator design and `Z` the study indicator matrix. The marginal covariance is `S` for `model="common"`, `S + tau2 I` for `model="effect"`, and `S + tau2 Z Z'` for `model="study"`. The latter introduces one shared random intercept per independent study. Effect and study random components are separate models; this API does not fit both simultaneously.

GLS estimates `(X' V^-1 X)^-1 X' V^-1 y`; model-based coefficient covariance is `(X' V^-1 X)^-1`. ML or REML profiles one nonnegative variance component. A valid zero boundary is retained; optimizer failures, singular designs and unsupported covariance structures are rejected. Moderator associations are not causal effects. The models correspond to the declared covariance and random-intercept constructions in the [metafor author documentation](https://wviechtb.github.io/metafor/reference/rma.mv.html).

The input domain is at most 256 effects, 64 independent studies and eight coefficient terms. Numeric moderators are explicit; missing values, weights, categorical expansion, streaming datasets, GPU execution and unknown or semidefinite sampling covariance are outside this route. The core records structural workspace/work admission and scalar-solver evaluation limits before fitting. Stored covariance is caller-provided evidence, not authenticated sampling-design proof.

## Study-robust inference and contrasts

`meta_dependent_robust(result, correction="CR0"|"CR1")` uses complete study scores `u_g = X_g' V_g^-1 (y_g-X_g beta)`. The sandwich is `B (sum_g u_g u_g') B`, where `B` is the fitted GLS covariance. CR1 multiplies it by `G/(G-p)`; CR0 leaves it unchanged. Both use individual `t(G-p)` and joint `F(m,G-p)` references, requiring `G>p` and nonsingular score covariance. This is the residual-degrees-of-freedom convention described by the [metafor author robust-inference documentation](https://wviechtb.github.io/metafor/reference/robust.html). No CR2, Satterthwaite approximation or exact small-sample validity is claimed.

The fitted coefficients and variance component stay unchanged. The saved state retains both original fit covariance and current sandwich covariance, correction and inferential degrees of freedom. The complete study scores appear in their own result table and are recomputed from retained input state during validation. Restore checks recompute the fit at saved tau2 and the declared sandwich before exposing postestimation.

The wide `cluster_scores` table preserves every original coefficient term. Its group-label/count columns use collision-free names declared in `attrs["cluster_score_metadata_columns"]`; a moderator named `study`, `n_effects` or an alias cannot overwrite that metadata. Identifiers whose JSON spellings differ but numeric values compare equal, such as `1` and `1.0`, are rejected before alignment.

`meta_dependent_contrast` accepts a DataFrame whose columns name every fitted coefficient and whose rows name independent contrast vectors. It retains full `L C L'` covariance, supplied scalar or ordered vector nulls, scalar intervals/tests and the joint chi-square or F test. It does not refit. Singular or unidentified contrast sets are explicit errors; a pseudoinverse is not substituted.

## Joint prediction

`meta_dependent_predict(result, data=...)` returns `X_new beta`, full `X_new C X_new'` mean covariance and marginal mean intervals under saved normal or cluster-t inference. New data require complete numeric moderators and at most 256 explicit rows. Original prediction labels and positional covariance axes are retained even when row labels repeat.

`meta_dependent_predict_effect(result, data=..., study="future_study")` additionally returns the model-matched future latent covariance: zero for common, `tau2 I` for effect, or shared `tau2` blocks for equal future-study labels under the study model. Joint predictive covariance adds that matrix to mean covariance. Study-model labels must refer to new studies; existing-study BLUPs require a different conditional covariance contract and are rejected here. These are plug-in latent true-effect intervals, consistent with the distinction between mean and future true effects in the [metafor author prediction documentation](https://wviechtb.github.io/metafor/reference/predict.rma.html). They exclude future sampling error and variance-component uncertainty.

## Complete-denominator diagnostics and saved artifacts

`meta_dependent_diagnostics(result)` deletes every complete study, removes its full sampling-covariance block and refits the same model/method. For a robust input it also recomputes the same study-robust correction in every deletion. Successful refits retain full coefficients, covariance, variance-component fit receipt and saved state; failures retain the study label, retained effect IDs/positions and explicit error code. All planned studies remain in the attempted denominator. A conservative cumulative 100-million structural work gate precedes all refits; it is an admission estimate, not measured runtime or memory usage. Deletion sensitivity does not identify publication bias or justify discarding a study.

The [editable eight-stage example](../examples/dependent_meta_eight.py) uses synthetic known covariance. It displays one summary table per stage and saves every full result table, table attribute, result attribute, saved state and LaTeX representation. Canonical JSON bytes are compared after restoration and across source, wheel and frozen execution; installed native Run and actual quit/relaunch evidence are distinct acceptance steps. Licensed vendor execution, public-release/platform acceptance, multivariate random slopes, multiple variance components, CR2 and tau2-profile intervals remain outside these eight issues.
