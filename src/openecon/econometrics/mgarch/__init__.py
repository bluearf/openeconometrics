"""Torch-free bounded Gaussian multivariate GARCH(1,1) catalogue."""

from openecon.econometrics.registry import EstimatorInfo, Option, Role


def _estimator(name, title, stata):
    return EstimatorInfo(
        name=name,
        title=title,
        family="mgarch",
        entry="openecon.econometrics.mgarch.estimators:fit_mgarch",
        function=name,
        covariances=("nonrobust",),
        default_covariance="nonrobust",
        categorical=False,
        predictors="none",
        time="optional",
        stata=stata,
        roles=(
            Role(
                "series",
                many=True,
                required=True,
                doc="Additional outcome series; outcome is the first series.",
            ),
        ),
        options=(
            Option("max_n", "int", 512, minimum=30, maximum=2048),
            Option("max_iterations", "int", 300, minimum=1, maximum=2000),
            Option("tolerance", "float", 1e-7, minimum=1e-10, maximum=1e-3),
            Option(
                "initial_covariance",
                "json",
                None,
                doc="Optional symmetric strictly positive definite presample covariance.",
            ),
        ),
        description="Joint Gaussian ML, full observed information, fixed recorded backcast, "
        "float64 CPU matrix recursions. CCC/DCC dimension 2..3, full BEKK dimension 2; "
        "CCC/DCC full multistep covariances use explicitly labeled plug-in approximations.",
    )


ESTIMATORS = (
    _estimator("mgarch_ccc", "Constant conditional correlation GARCH", ("mgarch ccc",)),
    _estimator("mgarch_dcc", "Dynamic conditional correlation GARCH", ("mgarch dcc",)),
    _estimator("mgarch_bekk", "Full BEKK GARCH", ()),
)
FORECAST = {
    "mgarch_" + kind: "openecon.econometrics.mgarch.estimators:mgarch_forecast"
    for kind in ("ccc", "dcc", "bekk")
}
