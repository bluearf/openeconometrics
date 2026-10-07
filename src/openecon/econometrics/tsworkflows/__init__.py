"""Time-series workflows: static manifest; no tensor or dataframe imports."""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_ITER = (
    Option("max_iterations", "int", 300, minimum=1, maximum=5000),
    Option("tolerance", "float", 1e-7, minimum=1e-12, maximum=0.01),
)
ESTIMATORS = (
    EstimatorInfo(
        name="sspace",
        title="Linear Gaussian state space",
        family="tsworkflows",
        entry="openecon.econometrics.tsworkflows.sspace:fit_sspace",
        covariances=("nonrobust",),
        default_covariance="nonrobust",
        function="sspace",
        predictors="none",
        categorical=False,
        intercept="never",
        time="optional",
        roles=(Role("responses", many=True),),
        options=(Option("system", "json", required=True), *_ITER),
        description="Time-invariant linear Gaussian state-space estimation or fixed-system filtering with full state and measurement covariance; known or stationary initialization, float64 CPU.",
    ),
    EstimatorInfo(
        name="ets",
        title="Additive innovations ETS",
        family="tsworkflows",
        entry="openecon.econometrics.tsworkflows.ets:fit_ets",
        covariances=("nonrobust",),
        default_covariance="nonrobust",
        function="ets",
        predictors="none",
        categorical=False,
        intercept="never",
        time="optional",
        options=(
            Option("model", "str", "ANN", choices=("ANN", "AAN", "AdN", "ANA", "AAA", "AdA")),
            Option("period", "int", 2, minimum=2, maximum=48),
            Option("fixed", "json", {}),
            Option("initial", "json", None),
            *_ITER,
        ),
        description="Additive Gaussian innovations ETS with optional trend, damping and seasonality, constrained joint likelihood and conditional innovation forecast intervals; multiplicative variants unsupported.",
    ),
    EstimatorInfo(
        name="midas",
        title="Exponential-Almon MIDAS regression",
        family="tsworkflows",
        entry="openecon.econometrics.tsworkflows.midas:fit_midas",
        covariances=("nonrobust", "HC0", "HC1"),
        default_covariance="nonrobust",
        function="midas",
        predictors="optional",
        categorical=False,
        inference="t",
        roles=(Role("lags", many=True, required=True),),
        time="optional",
        options=(Option("alignment", "json", required=True), *_ITER),
        description="Exponential-Almon MIDAS with release-date-aware lag alignment, joint nonlinear estimates and full covariance; independent look-ahead validation is required.",
    ),
)
EXPORTS = {
    "auto_arima": "openecon.econometrics.tsworkflows.workflows:auto_arima",
    "rolling": "openecon.econometrics.tsworkflows.workflows:rolling",
    "recursive": "openecon.econometrics.tsworkflows.workflows:recursive",
    "midas_align": "openecon.econometrics.tsworkflows.midas:midas_align",
    "midas_predict": "openecon.econometrics.tsworkflows.midas:midas_predict",
    "sspace_filter": "openecon.econometrics.tsworkflows.sspace:sspace_filter",
}
FORECAST = {
    "sspace": "openecon.econometrics.tsworkflows.sspace:forecast",
    "ets": "openecon.econometrics.tsworkflows.ets:forecast",
}
