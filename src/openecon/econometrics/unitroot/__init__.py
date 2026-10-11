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

``fourier_adf``, ``fourier_kpss`` and ``fourier_lm`` (one-frequency Fourier
unit-root/stationarity tests), ``ers_pt`` (Elliott-Rothenberg-Stock
point-optimal P_T) and ``hegy`` (quarterly/monthly seasonal unit roots) return
critical values only; their sources (author-hosted tables and seeded
simulations) are listed in ``docs/econometrics/advanced-unitroot-fourier.md``.
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
    "fourier_adf": "openecon.econometrics.unitroot.fourier:fourier_adf",
    "fourier_kpss": "openecon.econometrics.unitroot.fourier:fourier_kpss",
    "fourier_lm": "openecon.econometrics.unitroot.fourier:fourier_lm",
    "ers_pt": "openecon.econometrics.unitroot.ers:ers_pt",
    "hegy": "openecon.econometrics.unitroot.hegy:hegy",
    "lee_strazicich": "openecon.econometrics.unitroot.lee_strazicich:lee_strazicich",
    "clemente": "openecon.econometrics.unitroot.clemente:clemente",
    "psy": "openecon.econometrics.unitroot.bubbles:psy",
    "gregory_hansen": "openecon.econometrics.unitroot.coint_breaks:gregory_hansen",
    "hatemi_j": "openecon.econometrics.unitroot.coint_breaks:hatemi_j",
    "maki_coint": "openecon.econometrics.unitroot.coint_breaks:maki_coint",
    "bayer_hanck": "openecon.econometrics.unitroot.bayer_hanck:bayer_hanck",
    "ecm_cointegration": "openecon.econometrics.unitroot.bayer_hanck:ecm_cointegration",
    "tsong_fourier": "openecon.econometrics.unitroot.fourier_coint:tsong_fourier",
    "fourier_adl": "openecon.econometrics.unitroot.fourier_coint:fourier_adl",
    "enders_siklos": "openecon.econometrics.unitroot.threshold_coint:enders_siklos",
}
