"""Torch-free manifest for separately typed latent-model workflows.

The existing ``sem`` command remains the spatial error model.
"""

ESTIMATORS = ()
EXPORTS = {
    "cfa": "openecon.econometrics.latent.cfa:cfa",
    "cfa_covariance": "openecon.econometrics.latent.cfa:cfa_covariance",
    "cfa_restore": "openecon.econometrics.latent.cfa:cfa_restore",
    "latent_sem": "openecon.econometrics.latent.sem:latent_sem",
    "latent_sem_covariance": "openecon.econometrics.latent.sem:latent_sem_covariance",
    "latent_sem_restore": "openecon.econometrics.latent.sem:latent_sem_restore",
    "sem_effects": "openecon.econometrics.latent.sem:sem_effects",
    "latent_scores": "openecon.econometrics.latent.sem:latent_scores",
}
