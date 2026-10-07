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
    "export_group_state": "openecon.econometrics.postest.group_state:export_group_state",
    "lrtest": "openecon.econometrics.postest.likelihood:lrtest",
    "estat_ic": "openecon.econometrics.postest.likelihood:estat_ic",
    "bootstrap": "openecon.econometrics.postest.resampling:bootstrap",
    "jackknife": "openecon.econometrics.postest.resampling:jackknife",
    "suest": "openecon.econometrics.postest.suest:suest",
    "fcast_eval": "openecon.econometrics.postest.forecast_eval:fcast_eval",
    "dm_test": "openecon.econometrics.postest.forecast_eval:dm_test",
}
