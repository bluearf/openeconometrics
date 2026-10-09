# Causal sensitivity and assignment inference

Eight separate method stages were defined before implementation under
MARKET-187/GitHub75. They extend the previously delivered causal design family;
each result uses its complete `causal_design_save/load` artifact contract.

| Stage | Public API | Bounded target |
|---|---|---|
| MARKET-425 | `ovb_sensitivity` | One scalar omitted confounder, homoskedastic full-rank OLS partial-R2 adjustment and equal-strength robustness value. |
| MARKET-426 | `evalue` | Declared RR equal-strength hidden-confounding bound, optional closest-to-null confidence limit. |
| MARKET-427 | `manski_ate` | Empirical consistency-only bounded-outcome ATE identification interval. |
| MARKET-428 | `lee_bounds` | Declared randomized monotone selection, fractional empirical trimming and always-selected ATE identification interval. |
| MARKET-429 | `randomization_test` | Complete fixed-treated-count assignment under a constant additive sharp null. |
| MARKET-430 | `stratified_randomization` | Independent fixed treatment counts in declared strata, sample-size weighted statistic and sharp null. |
| MARKET-431 | `rosenbaum_rank_bounds` | Exact one-sided paired signed-rank p bounds at prespecified Gamma odds ratios. |
| MARKET-432 | `bias_sensitivity` | Externally fixed bias-box support function and complete-joint-covariance Gaussian pointwise interval union. |

Detailed contracts and primary references:

- [OLS omitted-variable sensitivity](causal-ovb.md)
- [Identification and RR bounds](causal-identification-bounds.md)
- [Assignment and paired hidden-bias tests](causal-randomization.md)
- [Externally fixed bias boxes](causal-fixed-bias.md)

Identification intervals in Manski/Lee and E-value sensitivity summaries do not
provide sampling covariance, SE or confidence intervals. Sharp-null p bounds
do not estimate an ATE confidence interval. OLS adjusted t intervals condition
on the declared one-confounder model. Fixed box intervals condition on a bias
restriction supplied independently of estimation noise. Complete assignment
universes are design declarations; these APIs do not infer a valid experiment
from the treatment labels or arbitrarily shuffle rows.

Float64 Torch CPU executes every numerical kernel. Native runtime estimation
does not use SciPy/statsmodels or development reference estimators. Selected
role/missing/sample identity, all assignments/PMFs, trim weights, full original
OLS or supplied covariance, sensitivity settings and unsupported assumptions
are retained with exact table order/dtypes and checksums. Work and workspace
limits apply to complete computation, with no truncation or sampling fallback.

General observational distribution/survival adjustment, covariate-specific Lee
bounds, multiconfounder/heteroskedastic OLS sensitivity, data-adaptive pretrend
restrictions and HonestDiD remain separate open scopes in MARKET-187. Method
validation, frozen/installed app behavior, public merge and tracker closure
are distinct evidence layers; licensed vendor and whole-product parity remain
unvalidated. See the [synthetic executable example](../examples/causal_sensitivity_eight.py)
and delivery evidence (internal evidence excluded from this public snapshot).
