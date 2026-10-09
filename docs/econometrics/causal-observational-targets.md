# Residual causal distribution, survival and OVB options

The following eight substantive child scopes were declared before coding under
the remaining MARKET-187 treatment-target and sensitivity work, originally audited in GitHub #75 and now continued in open GitHub #157.
Previously completed balancing, randomized targets and sensitivity methods
remain separate stages.

| Issue | Public API | Supported method |
|---|---|---|
| MARKET-481 | `ovb_benchmark` | Formal observed-control/group partial-R² confounder-strength benchmark on an intact original OVB result. |
| MARKET-482 | `ovb_robustness` | Point and significance robustness thresholds over prespecified q/alpha grids, with max-strength minimax witnesses. |
| MARKET-483 | `treatment_cdf_ipw` | Estimated unpenalized logit propensity, normalized Hájek marginal CDFs and complete nuisance-estimation sandwich covariance. |
| MARKET-484 | `treatment_quantile_ipw` | Weighted inverse marginal CDFs, private-seeded whole-row IID bootstrap with propensity refit in every draw. |
| MARKET-485 | `treatment_cdf_aipw` | Held-out propensity and threshold-logit outcome fits, full cross-fitted CDF scores and joint orthogonal inference. |
| MARKET-486 | `treatment_survival_ipcw` | Externally known propensity/censor survival, strict-survival HT scores and complete joint covariance at fixed times. |
| MARKET-487 | `treatment_rmst_ipcw` | Known propensity and exponential censoring, exact IPCW restricted-time integrals and full subject-score covariance. |
| MARKET-488 | `treatment_rmst_aipw` | Complete horizon observation, known propensity and externally fixed conditional RMST functions. |

All observational methods require `design="unconfounded"`: consistency/SUTVA,
conditional treatment exchangeability, independent sampled subjects and
positivity are caller declarations, not findings established by the program.
Controls must precede treatment. Survival methods additionally require
`nuisance="known"`; their subject covariance does not account for fitting
treatment or censoring functions from these same observations. Conditional
independent censoring and support through the declared horizon are required.
The complete-horizon AIPW RMST route rejects observations censored before tau.

IPW distribution methods include propensity-estimation uncertainty. Every
quantile bootstrap draw refits its nuisance model; failed refits invalidate
the complete run. Marginal quantile differences are not quantiles of
individual treatment effects. Quantile inference additionally requires
continuous distributions with positive density around the selected quantiles.

AIPW CDF point consistency is doubly robust, whereas its stated orthogonal
score inference requires both consistent nuisance functions and sufficient
product rates. One correct model alone does not justify these confidence
intervals. Raw HT/AIPW survival or CDF estimates can leave [0,1] and can be
nonmonotone in finite samples; raw RMST estimates can leave [0,tau]. Outputs
are retained without clipping or rearrangement. IPCW scores are not a
Kaplan–Meier estimator.

The original OVB result remains integrity checked, including its full sample,
design and covariance. Formal benchmark restrictions are explicit assumptions
about hidden confounding. Plug-in estimates at a bound's corner are not a
worst-case confidence guarantee over a box. Robustness thresholds retain the
admitted minimax branch and confounder-strength witness.

Detailed mathematics, primary sources and admission limits:

- [OVB benchmark and robustness](causal-ovb-postest.md)
- [Observational distribution and quantile targets](causal-observational-distribution.md)
- [Known-nuisance survival and RMST targets](causal-observational-survival.md)
- [Executable synthetic example](../examples/causal_targets_eight.py)
- Source, package and native acceptance evidence (internal evidence excluded from this public snapshot)

Native Torch CPU float64 executes the admitted calculations; generic weights,
GPU and streaming Dataset routes remain unsupported. Complete artifact
save/load retains every typed table, original physical sample identity,
nuisance fit/score/draw/fold state, covariance, restrictions and resource
admission. No blanket licensed-vendor or whole-product parity is asserted.
General data-adaptive pretrend/HonestDiD, unidentified causal targets and
unsupported nuisance/censor/design cases remain open in MARKET-187.
