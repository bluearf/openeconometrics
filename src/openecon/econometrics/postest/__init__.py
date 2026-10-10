"""Post-estimation procedures shared by every registry estimator (Torch-free manifest).

The family registers no estimator. It publishes procedures that take fitted
``ResultBundle`` objects from any family:

- ``lrtest`` (Stata ``lrtest``) and ``estat_ic`` (``estat ic`` / ``estimates
  stats``): likelihood-ratio tests and AIC/BIC tables from stored results;
- ``bootstrap`` and ``jackknife`` (Stata's prefixes / ``vce(bootstrap)`` and
  ``vce(jackknife)``): resampling covariances by refitting ``result.spec``;
- ``suest`` (Stata ``suest``): seemingly unrelated estimation that stacks the
  score contributions of several fits into one robust covariance;
- ``fcast_eval`` (EViews forecast evaluation) and ``dm_test``
  (Diebold-Mariano with the Harvey-Leybourne-Newbold correction).

See ``docs/econometrics/postest.md``.
"""

from openecon.econometrics.registry import EstimatorInfo

ESTIMATORS: tuple[EstimatorInfo, ...] = ()

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "trajectory_bands": "openecon.econometrics.postest.trajectories:trajectory_bands",
    "summary_state": "openecon.econometrics.summary_state:summary_state",
    "restore_summary": "openecon.econometrics.summary_state:restore_summary",
    "wild_cluster_test": "openecon.econometrics.postest.wild_restricted:wild_cluster_test",
    "wild_cluster_confidence_set": "openecon.econometrics.postest.wild_restricted:wild_cluster_confidence_set",
    "export_group_state": "openecon.econometrics.postest.group_state:export_group_state",
    "lrtest": "openecon.econometrics.postest.likelihood:lrtest",
    "estat_ic": "openecon.econometrics.postest.likelihood:estat_ic",
    "bootstrap": "openecon.econometrics.postest.resampling:bootstrap",
    "jackknife": "openecon.econometrics.postest.resampling:jackknife",
    "suest": "openecon.econometrics.postest.suest:suest",
    "fcast_eval": "openecon.econometrics.postest.forecast_eval:fcast_eval",
    "dm_test": "openecon.econometrics.postest.forecast_eval:dm_test",
    "multipletests": "openecon.econometrics.postest.multiple:multipletests",
    "stepdown": "openecon.econometrics.postest.multiple:stepdown",
    "simultaneous_ci": "openecon.econometrics.postest.multiple:simultaneous_ci",
    "simultaneous_t_ci": "openecon.econometrics.postest.finite_sample:simultaneous_t_ci",
    "ols_stepdown": "openecon.econometrics.postest.finite_sample:ols_stepdown",
    "hotelling_region": "openecon.econometrics.postest.finite_sample:hotelling_region",
    "mean_sign_stepdown": "openecon.econometrics.postest.randomization_joint:mean_sign_stepdown",
    "mean_permutation_stepdown": "openecon.econometrics.postest.randomization_joint:mean_permutation_stepdown",
    "simultaneous_dkw_band": "openecon.econometrics.postest.distribution_free_joint:simultaneous_dkw_band",
    "simultaneous_quantile_ci": "openecon.econometrics.postest.distribution_free_joint:simultaneous_quantile_ci",
    "simultaneous_proportion_ci": "openecon.econometrics.postest.distribution_free_joint:simultaneous_proportion_ci",
    "multinomial_region": "openecon.econometrics.postest.distribution_free_joint:multinomial_region",
    "hoeffding_mean_ci": "openecon.econometrics.postest.bounded_joint:hoeffding_mean_ci",
    "empirical_bernstein_mean_ci": "openecon.econometrics.postest.bounded_joint:empirical_bernstein_mean_ci",
    "bernoulli_confidence_sequence": "openecon.econometrics.postest.confidence_sequences:bernoulli_confidence_sequence",
    "poisson_confidence_sequence": "openecon.econometrics.postest.confidence_sequences:poisson_confidence_sequence",
    "normal_mean_confidence_sequence": "openecon.econometrics.postest.confidence_sequences:normal_mean_confidence_sequence",
    "student_mean_confidence_sequence": "openecon.econometrics.postest.confidence_sequences:student_mean_confidence_sequence",
    "normal_variance_confidence_sequence": "openecon.econometrics.postest.confidence_sequences:normal_variance_confidence_sequence",
    "exponential_mean_confidence_sequence": "openecon.econometrics.postest.confidence_sequences:exponential_mean_confidence_sequence",
    "uniform_endpoint_confidence_sequence": "openecon.econometrics.postest.confidence_sequences:uniform_endpoint_confidence_sequence",
    "hoeffding_confidence_sequence": "openecon.econometrics.postest.confidence_sequences:hoeffding_confidence_sequence",
}
