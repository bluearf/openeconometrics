"""Classical parametric statistics: manifest of the ``stats`` family (Torch-free).

These are test procedures, not model fits: they register no estimator and are
published as ``oe.<name>`` through ``EXPORTS``. Each returns a result table or a
``TableSet`` of tables (see ``docs/econometrics/stats.md``).
"""

from openecon.econometrics.registry import EstimatorInfo

ESTIMATORS: tuple[EstimatorInfo, ...] = ()

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "ttest": "openecon.econometrics.stats.ttest:ttest",
    "sdtest": "openecon.econometrics.stats.ttest:sdtest",
    "oneway": "openecon.econometrics.stats.oneway:oneway",
    "anova": "openecon.econometrics.stats.anova:anova",
    "rm_anova": "openecon.econometrics.stats.rm_anova:rm_anova",
    "manova": "openecon.econometrics.stats.manova:manova",
    "correlate": "openecon.econometrics.stats.correlation:correlate",
    "pcorr": "openecon.econometrics.stats.correlation:pcorr",
    "describe": "openecon.econometrics.stats.descriptives:describe",
}
