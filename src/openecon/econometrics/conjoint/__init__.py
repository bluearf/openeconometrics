"""Bounded full-profile conjoint procedures; lightweight public manifest."""

ESTIMATORS = ()
_PACKAGE = "openecon.econometrics.conjoint"
EXPORTS = {
    "conjoint_plan": f"{_PACKAGE}.design:conjoint_plan",
    "conjoint_orthogonal": f"{_PACKAGE}.design:conjoint_orthogonal",
    "conjoint_diagnostics": f"{_PACKAGE}.design:conjoint_diagnostics",
    "conjoint_fit": f"{_PACKAGE}.model:conjoint_fit",
    "conjoint_predict": f"{_PACKAGE}.model:conjoint_predict",
    "conjoint_holdout": f"{_PACKAGE}.post:conjoint_holdout",
    "conjoint_importance": f"{_PACKAGE}.post:conjoint_importance",
    "conjoint_simulate": f"{_PACKAGE}.post:conjoint_simulate",
    "conjoint_save": f"{_PACKAGE}.common:conjoint_save",
    "conjoint_load": f"{_PACKAGE}.common:conjoint_load",
}
