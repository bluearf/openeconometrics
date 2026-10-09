"""unitroot family manifest (Torch-free).

Unit-root, stationarity, structural-break and cointegration tests for single
series and panels: ``dfuller``, ``dfgls``, ``pperron``, ``kpss``, ``zandrews``
(Zivot-Andrews), ``egranger`` (Engle-Granger), ``chow``, ``sbsingle``
(sup/average/exponential Wald at an unknown break date), ``cusum``,
``xtunitroot`` (Levin-Lin-Chu, Im-Pesaran-Shin, Fisher, Hadri, Breitung,
Harris-Tzavalis, CADF and CIPS), ``xtcips`` (cross-sectionally augmented
individual/averaged statistics), ``xtpanic`` (Bai-Ng fixed-factor common and
idiosyncratic tests) and ``xtcointtest`` (Kao, Pedroni, Westerlund 2005). The family registers no estimator:
every procedure is a test that returns a result table (``core.table`` /
``core.TableSet``) whose ``attrs`` hold the statistic, p-value, critical values
and settings.
Everything is implemented in OpenEconometrics on float64 tensors; p-values and
critical values come from the published tables typed into
``unitroot/tables.py`` and ``unitroot/cips_tables.py``. See
``docs/econometrics/unitroot.md``, ``docs/econometrics/cips.md`` and
``docs/econometrics/panic.md``.

``bai_perron`` supplies pure multiple-change OLS with bounded Gaussian iid
fixed-design Monte Carlo calibration; see ``docs/econometrics/bai-perron.md``.
"""

from openecon.econometrics.registry import EstimatorInfo

ESTIMATORS: tuple[EstimatorInfo, ...] = ()

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "ols_cusum": "openecon.econometrics.unitroot.ols_stability:ols_cusum",
    "fisher_johansen": "openecon.econometrics.unitroot.fisher_johansen:fisher_johansen",
    "ngperron": "openecon.econometrics.unitroot.advanced_series:ngperron",
    "kss": "openecon.econometrics.unitroot.advanced_series:kss",
    "bai_perron": "openecon.econometrics.unitroot.bai_perron:bai_perron",
    "phillips_ouliaris": "openecon.econometrics.unitroot.phillips_ouliaris:phillips_ouliaris",
    "po_za": "openecon.econometrics.unitroot.phillips_ouliaris:po_za",
    "po_zt": "openecon.econometrics.unitroot.phillips_ouliaris:po_zt",
    "dfuller": "openecon.econometrics.unitroot.series:dfuller",
    "dfgls": "openecon.econometrics.unitroot.series:dfgls",
    "pperron": "openecon.econometrics.unitroot.series:pperron",
    "kpss": "openecon.econometrics.unitroot.series:kpss",
    "zandrews": "openecon.econometrics.unitroot.zandrews:zandrews",
    "egranger": "openecon.econometrics.unitroot.cointegration:egranger",
    "xtunitroot": "openecon.econometrics.unitroot.xtunitroot:xtunitroot",
    "xtcips": "openecon.econometrics.unitroot.cips:xtcips",
    "xtpanic": "openecon.econometrics.unitroot.panic:xtpanic",
    "xtcointtest": "openecon.econometrics.unitroot.xtcointtest:xtcointtest",
    "chow": "openecon.econometrics.unitroot.stability:chow",
    "sbsingle": "openecon.econometrics.unitroot.stability:sbsingle",
    "cusum": "openecon.econometrics.unitroot.stability:cusum",
}
