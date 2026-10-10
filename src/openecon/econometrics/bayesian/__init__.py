"""Analytic proper-prior posterior helpers; no frequentist estimator registration.

The manifest stays tensor-free. Finite proper-prior BMA retains every model
component; sampling-chain and BVAR work is separate.
"""

ESTIMATORS = ()

EXPORTS = {
    "bayes_bma": "openecon.econometrics.bayesian.bma:bayes_bma",
    "bayes_bma_restore": "openecon.econometrics.bayesian.bma:bayes_bma_restore",
    "bayes_bma_predict": "openecon.econometrics.bayesian.bma_query:bayes_bma_predict",
    "bayes_bma_prediction_restore": "openecon.econometrics.bayesian.bma_query:bayes_bma_prediction_restore",
    "bayes_bma_draws": "openecon.econometrics.bayesian.bma_draws:bayes_bma_draws",
    "bayes_bma_draws_restore": "openecon.econometrics.bayesian.bma_draws:bayes_bma_draws_restore",
    "bayes_linear": "openecon.econometrics.bayesian.commands:bayes_linear",
    "bayes_contrast": "openecon.econometrics.bayesian.commands:bayes_contrast",
    "bayes_predict": "openecon.econometrics.bayesian.commands:bayes_predict",
    "bayes_draws": "openecon.econometrics.bayesian.commands:bayes_draws",
    "bayes_hypothesis": "openecon.econometrics.bayesian.hypothesis:bayes_hypothesis",
    "bayes_hypothesis_restore": "openecon.econometrics.bayesian.hypothesis:bayes_hypothesis_restore",
    "bayes_hypothesis_predict": "openecon.econometrics.bayesian.hypothesis:bayes_hypothesis_predict",
    "bayes_hypothesis_draws": "openecon.econometrics.bayesian.hypothesis:bayes_hypothesis_draws",
    "bayes_hypothesis_draws_restore": "openecon.econometrics.bayesian.hypothesis:bayes_hypothesis_draws_restore",
}
