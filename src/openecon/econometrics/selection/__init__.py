"""selection family manifest (Torch-free).

Regression model-building procedures in the tradition of SPSS ``REGRESSION
/METHOD=STEPWISE``, ``/STATISTICS=COLLIN TOL``, ``CURVEFIT`` and ``MEANS``, and
Stata ``stepwise``, ``estat vif``, ``coldiag2`` and ``tabstat``:

- ``stepwise``: forward, backward and stepwise selection of regressors by the
  F test of each term or by AIC / BIC, on a cross-product matrix that is
  computed once and updated with the sweep operator;
- ``collin``: tolerances, variance inflation factors, condition indices and
  variance-decomposition proportions;
- ``curvefit``: the eleven curve-estimation models of SPSS CURVEFIT;
- ``tabstat``: weighted or unweighted summary statistics by group, with the
  SPSS MEANS extras and its eta-squared ANOVA table.

The family registers no estimator: every procedure returns ``core.table`` /
``core.TableSet`` results whose ``attrs`` hold the scalar results and settings.
Everything is computed in OpenEconometrics on float64 tensors. See
``docs/econometrics/selection.md``.
"""

from openecon.econometrics.registry import EstimatorInfo

ESTIMATORS: tuple[EstimatorInfo, ...] = ()

_PACKAGE = "openecon.econometrics.selection"

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "stepwise": f"{_PACKAGE}.stepwise:stepwise",
    "collin": f"{_PACKAGE}.collin:collin",
    "curvefit": f"{_PACKAGE}.curvefit:curvefit",
    "tabstat": f"{_PACKAGE}.tabstat:tabstat",
}
