"""Tensor-free public integration of distinct proper-MNIW posterior objects.

No ResultBundle estimator or generic predict/forecast route is registered.
Original study-pinned Bayesian VAR implementation files remain untouched.
"""

ESTIMATORS = ()

EXPORTS = {
    "bayes_var": "openecon.econometrics.bayesian_var.api:bayes_var",
    "bayes_var_predict": "openecon.econometrics.bayesian_var.api:bayes_var_predict",
    "bayes_var_contrast": "openecon.econometrics.bayesian_var.api:bayes_var_contrast",
    "bayes_var_draws": "openecon.econometrics.bayesian_var.draws:bayes_var_draws",
    "bayes_var_forecast": "openecon.econometrics.bayesian_var.forecast:bayes_var_forecast",
    "bayes_var_irf": "openecon.econometrics.bayesian_var.impulse:bayes_var_irf",
    "MatrixNormalInverseWishartPrior": "openecon.econometrics.bayesian_var.posterior:MatrixNormalInverseWishartPrior",
    "BayesianVARPosterior": "openecon.econometrics.bayesian_var.posterior:BayesianVARPosterior",
    "BayesianVARSource": "openecon.econometrics.bayesian_var.posterior:BayesianVARSource",
    "PosteriorAlgebra": "openecon.econometrics.bayesian_var.posterior:PosteriorAlgebra",
    "RangeIndexState": "openecon.econometrics.bayesian_var.posterior:RangeIndexState",
    "PlainIndexState": "openecon.econometrics.bayesian_var.posterior:PlainIndexState",
    "DatetimeIndexState": "openecon.econometrics.bayesian_var.posterior:DatetimeIndexState",
    "BayesianVARPrediction": "openecon.econometrics.bayesian_var.api:BayesianVARPrediction",
    "BayesianVARContrast": "openecon.econometrics.bayesian_var.api:BayesianVARContrast",
    "BayesianVARDraws": "openecon.econometrics.bayesian_var.draws:BayesianVARDraws",
    "BayesianVARForecast": "openecon.econometrics.bayesian_var.forecast:BayesianVARForecast",
    "BayesianVARImpulse": "openecon.econometrics.bayesian_var.impulse:BayesianVARImpulse",
    "MomentAvailability": "openecon.econometrics.bayesian_var.forecast:MomentAvailability",
    "MonteCarloSummary": "openecon.econometrics.bayesian_var.forecast:MonteCarloSummary",
    "bayes_var_restore": "openecon.econometrics.bayesian_var_public.transport:bayes_var_restore",
    "bayes_var_predict_restore": "openecon.econometrics.bayesian_var_public.transport:bayes_var_predict_restore",
    "bayes_var_contrast_restore": "openecon.econometrics.bayesian_var_public.transport:bayes_var_contrast_restore",
    "bayes_var_draws_restore": "openecon.econometrics.bayesian_var_public.transport:bayes_var_draws_restore",
    "bayes_var_forecast_restore": "openecon.econometrics.bayesian_var_public.transport:bayes_var_forecast_restore",
    "bayes_var_irf_restore": "openecon.econometrics.bayesian_var_public.transport:bayes_var_irf_restore",
    "bayes_var_prior_restore": "openecon.econometrics.bayesian_var_public.transport:bayes_var_prior_restore",
    "bayes_var_tables": "openecon.econometrics.bayesian_var_public.tables:bayes_var_tables",
    "bayes_var_state_json": "openecon.econometrics.bayesian_var_public.tables:bayes_var_state_json",
}
