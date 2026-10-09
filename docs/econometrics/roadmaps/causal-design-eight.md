# Eight causal-design implementation stages

The method domains below were recorded in Linear before implementation on
7 October 2026 under MARKET-187 / GitHub #75. Existing cross-fitted LATE/IRM,
matching and RD code is preserved. Native float64 Torch resident numerical work,
complete checksummed state and separate source/wheel/frozen/native/restart proof
are shared acceptance conditions; merge and tracker readback precede closure.

| Stage | API | Acceptance |
|---|---|---|
| MARKET-304 | `ebalance` | Exact normalized moments, positive KL weights, full-rank/support/convergence/ESS gates and independent dual optimization. |
| MARKET-305 | `cem` | Prespecified numeric cut boundary and category strata; exact ATT donor weights; complete retained/excluded target/sample state. |
| MARKET-306 | `balance` | Original-scale SMD, full fixed-weight mean covariance, weighted ECDF ties and explicit zero-weight target exclusions. |
| MARKET-307 | `rosenbaum_bounds` | Exact sign/binomial tail odds bounds, declared Gamma/direction, ties and pairing preserved. |
| MARKET-308 | `paired_randomization` | Declared paired assignment design, exact complete enumeration or private-RNG plus-one MC, every assignment/statistic saved. |
| MARKET-309 | `treatment_cdf` | Fixed randomized-arm grid and complete joint indicator covariance, pointwise uncertainty and degenerate-tail flags. |
| MARKET-310 | `treatment_quantile` | Defined empirical inverse CDF, independent-arm bootstrap with complete draws/targets/covariance and percentile intervals. |
| MARKET-312 | `treatment_rmst` | Prespecified common tau, complete KM risk tables, exact integrated Greenwood covariance, censoring/support/terminal-event guards. |

These eight child closures will leave MARKET-187/GitHub #75 open for general
pretrend sensitivity, observational survival/distribution adjustment and other
advanced targets. Author-method fixtures and independent numerical oracles are
distinct from licensed vendor executable validation and public release delivery.
