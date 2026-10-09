"""Unidimensional IRT: lazy, Torch-free method catalogue."""

ESTIMATORS = ()
EXPORTS = {**{name: f"openecon.econometrics.irt.models:{name}" for name in (
    "irt_rasch", "irt_2pl", "irt_3pl", "irt_grm", "irt_pcm", "irt_rsm",
    "irt_information", "irt_score", "irt_restore",
)}, **{name: f"openecon.econometrics.irt.diagnostics:{name}" for name in (
    "irt_fit_diagnostics", "irt_mh_dif", "irt_diagnostics_restore",
)}, **{name: f"openecon.econometrics.irt.calibrated:{name}" for name in (
    "irt_bank_binary", "irt_bank_polytomous", "irt_bank_restore",
    "irt_posterior", "irt_posterior_restore",
)}, **{name: f"openecon.econometrics.irt.calibrated_scores:{name}" for name in (
    "irt_score_mle", "irt_score_map",
)}, **{name: f"openecon.econometrics.irt.calibrated_predictive:{name}" for name in (
    "irt_plausible_values", "irt_predictive", "irt_test_score_distribution",
)}}
