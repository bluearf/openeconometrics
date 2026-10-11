"""Regression-discontinuity local randomization inference; lazy, tensor-free public catalogue.

Community ``rdlocrand`` (Cattaneo, Titiunik and Vazquez-Bare): randomization
inference in a window around the cutoff (``rdrandinf``), window selection from
covariate balance tests (``rdwinselect``), sensitivity of p-values to the
window and the null (``rdsensitivity``) and Rosenbaum bounds (``rdrbounds``).
The family registers no estimator: every procedure returns a ``TableSet``.
See ``docs/econometrics/rd-local-randomization.md``.
"""

ESTIMATORS = ()

_PACKAGE = "openecon.econometrics.rdlocrand"

EXPORTS = {
    "rdrandinf": f"{_PACKAGE}.randinf:rdrandinf",
    "rdwinselect": f"{_PACKAGE}.windows:rdwinselect",
    "rdsensitivity": f"{_PACKAGE}.sensitivity:rdsensitivity",
    "rdrbounds": f"{_PACKAGE}.bounds:rdrbounds",
}
