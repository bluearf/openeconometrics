"""Torch-free bounded Gaussian multivariate GARCH(1,1) catalogue.

CCC, DCC and full BEKK (MARKET-131); diagonal VECH, diagonal BEKK and asymmetric DCC
(MARKET-161, docs/econometrics/garch-extensions.md).
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role


def _estimator(
    name,
    title,
    stata,
    description="Joint Gaussian ML, full observed information, fixed recorded backcast, "
    "float64 CPU matrix recursions. CCC/DCC dimension 2..3, full BEKK dimension 2; "
    "CCC/DCC full multistep covariances use explicitly labeled plug-in approximations.",
):
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
        description=description,
    )


ESTIMATORS = (
    _estimator("mgarch_ccc", "Constant conditional correlation GARCH", ("mgarch ccc",)),
    _estimator("mgarch_dcc", "Dynamic conditional correlation GARCH", ("mgarch dcc",)),
    _estimator("mgarch_bekk", "Full BEKK GARCH", ()),
    _estimator(
        "mgarch_dvech", "Diagonal VECH GARCH", ("mgarch dvech",),
        "Joint Gaussian ML of the diagonal VECH(1,1) H_t = C + A o e e' + B o H_(t-1) in the "
        "Ding-Engle domain (C positive definite, A and B positive semidefinite, a_ii + b_ii < "
        "1), dimension 2..3, full observed information, exact multistep covariance forecasts.",
    ),
    _estimator(
        "mgarch_dbekk", "Diagonal BEKK GARCH", (),
        "Joint Gaussian ML of the diagonal BEKK(1,1) H_t = CC' + A'ee'A + B'H_(t-1)B with "
        "diagonal A, B (a_1, b_1 > 0; a_i^2 + b_i^2 < 1), dimension 2..3, full observed "
        "information, exact multistep covariance forecasts.",
    ),
    _estimator(
        "mgarch_adcc", "Asymmetric DCC GARCH", (),
        "Joint Gaussian ML of the Cappiello-Engle-Sheppard asymmetric DCC(1,1) with the "
        "estimated SPD target of mgarch_dcc and the sample second moment of the negative "
        "standardized residuals (a + b + delta g < 1), dimension 2..3, full observed "
        "information; multistep forecasts are labeled plug-in approximations.",
    ),
)
FORECAST = {
    "mgarch_" + kind: "openecon.econometrics.mgarch.estimators:mgarch_forecast"
    for kind in ("ccc", "dcc", "bekk", "dvech", "dbekk", "adcc")
}
