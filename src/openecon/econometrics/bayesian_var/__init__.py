"""Explicit proper conjugate Bayesian VAR; tensor-free integration manifest.

The public registry/export integration is deliberately a separate owner scope.
"""

EXPORTS = {
    "bayes_var": "openecon.econometrics.bayesian_var.api:bayes_var",
    "bayes_var_predict": "openecon.econometrics.bayesian_var.api:bayes_var_predict",
    "bayes_var_contrast": "openecon.econometrics.bayesian_var.api:bayes_var_contrast",
    "bayes_var_draws": "openecon.econometrics.bayesian_var.draws:bayes_var_draws",
    "bayes_var_forecast": "openecon.econometrics.bayesian_var.forecast:bayes_var_forecast",
    "bayes_var_irf": "openecon.econometrics.bayesian_var.impulse:bayes_var_irf",
}
