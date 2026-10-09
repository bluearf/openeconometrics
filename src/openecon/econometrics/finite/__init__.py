"""Bounded Firth, exact common odds/rate and exhaustive LTS procedures."""

ESTIMATORS = ()
EXPORTS = {
    "firth_logit": "openecon.econometrics.finite.firth:firth_logit",
    "firth_predict": "openecon.econometrics.finite.firth:firth_predict",
    "firth_profile": "openecon.econometrics.finite.firth:firth_profile",
    "firth_test": "openecon.econometrics.finite.firth:firth_test",
    "exact_logistic": "openecon.econometrics.finite.exact:exact_logistic",
    "exact_poisson_rate": "openecon.econometrics.finite.exact:exact_poisson_rate",
    "lts": "openecon.econometrics.finite.trimmed:lts",
    "lts_predict": "openecon.econometrics.finite.trimmed:lts_predict",
    "finite_save": "openecon.econometrics.finite.common:finite_save",
    "finite_load": "openecon.econometrics.finite.common:finite_load",
}
