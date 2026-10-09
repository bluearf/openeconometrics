"""Torch-free cross-sectional spatial econometrics catalogue."""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_OPTIONS = (
    Option(
        "spatial_weights",
        "json",
        required=True,
        doc="Keyed sparse SpatialWeights.to_payload() representation; no positional alignment.",
    ),
    Option(
        "max_n",
        "int",
        512,
        minimum=4,
        maximum=2048,
        doc="Explicit observation ceiling for exact dense likelihood/logdet/information.",
    ),
    Option("max_iterations", "int", 100, minimum=1, maximum=1000),
    Option("tolerance", "float", 1e-9, minimum=1e-12, maximum=1e-3),
)


def _estimator(name, title):
    return EstimatorInfo(
        name=name,
        title=title,
        family="spatial",
        entry="openecon.econometrics.spatial.estimators:fit_spatial",
        function=name,
        covariances=("nonrobust",),
        default_covariance="nonrobust",
        categorical=False,
        roles=(Role("key", required=True, kind="label", doc="Unique spatial unit identity."),),
        options=_OPTIONS
        + (
            (
                Option(
                    "error_weights",
                    "json",
                    None,
                    doc="Optional second keyed W for SAC errors; defaults to W.",
                ),
            )
            if name == "sac"
            else ()
        ),
        stata=("spregress, ml",),
        description="Cross-sectional Gaussian spatial ML with exact dense logdet, full observed "
        "information, key-aligned sparse weights and an explicit stable domain/budget. "
        "No observation weights, panel, categorical or Dataset route.",
    )


ESTIMATORS = (
    _estimator("sar", "Spatial autoregressive model"),
    _estimator("sem", "Spatial error model"),
    _estimator("sac", "Spatial autoregressive combined model"),
    _estimator("sdm", "Spatial Durbin model"),
    EstimatorInfo(
        name="sar_iv",
        title="SAR with explicit instrumental variables",
        family="spatial",
        entry="openecon.econometrics.spatial.iv:fit_sar_iv",
        function="sar_iv",
        covariances=("nonrobust", "HC0"),
        default_covariance="nonrobust",
        categorical=False,
        roles=(
            Role("key", required=True, kind="label"),
            Role("instruments", required=True, many=True),
        ),
        options=_OPTIONS[:2],
        description="Cross-sectional SAR QR 2SLS with supplied excluded instruments; iid/HC0 full covariance, saved reduced means and delta impacts. Fixed exogenous keyed W; no panel or weak-ID guarantee.",
    ),
)
EXPORTS = {
    "spatial_weights": "openecon.econometrics.spatial.weights:spatial_weights",
    "moran": "openecon.econometrics.spatial.moran:moran",
    "spatial_impacts": "openecon.econometrics.spatial.estimators:spatial_impacts",
    "sar_iv_predict": "openecon.econometrics.spatial.iv:sar_iv_predict",
    "spatial_predict": "openecon.econometrics.spatial.prediction:spatial_predict",
    "spatial_diagnostics": "openecon.econometrics.spatial.diagnostics:spatial_diagnostics",
}
