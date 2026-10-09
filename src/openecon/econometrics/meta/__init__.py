"""Independent-study meta-analysis; lazy, tensor-free public catalogue."""

ESTIMATORS = ()
EXPORTS = {
    "meta_dependent": "openecon.econometrics.meta.dependent:meta_dependent",
    "meta_dependent_robust": "openecon.econometrics.meta.dependent_post:meta_dependent_robust",
    "meta_dependent_contrast": "openecon.econometrics.meta.dependent_post:meta_dependent_contrast",
    "meta_dependent_predict": "openecon.econometrics.meta.dependent_post:meta_dependent_predict",
    "meta_dependent_predict_effect": "openecon.econometrics.meta.dependent_post:meta_dependent_predict_effect",
    "meta_dependent_diagnostics": "openecon.econometrics.meta.dependent_post:meta_dependent_diagnostics",
    "meta_effectsize": "openecon.econometrics.meta.effects:meta_effectsize",
    "meta_pool": "openecon.econometrics.meta.models:meta_pool",
    "meta_regress": "openecon.econometrics.meta.models:meta_regress",
    "meta_predict": "openecon.econometrics.meta.models:meta_predict",
    "meta_diagnostics": "openecon.econometrics.meta.diagnostics:meta_diagnostics",
    "meta_plot": "openecon.econometrics.meta.diagnostics:meta_plot",
}
