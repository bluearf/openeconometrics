"""PMG/MG/DFE manifests; no tensor imports at catalogue time."""

from openecon.econometrics.registry import EstimatorInfo, Option

ESTIMATORS = (
    EstimatorInfo(
        name="pmg",
        title="Pooled mean-group panel ARDL",
        family="panel_ardl",
        function="pmg",
        entry="openecon.econometrics.panel_ardl.models:fit_panel_ardl",
        covariances=("nonrobust",),
        default_covariance="nonrobust",
        categorical=False,
        intercept="never",
        panel="required",
        time="required",
        options=(
            Option("p", "int", 1, minimum=1, maximum=12),
            Option("q", "int", 1, minimum=1, maximum=12),
            Option("trend", "str", "c", choices=("n", "c", "ct")),
            Option("max_iterations", "int", 500, minimum=1, maximum=5000),
        ),
        description="Pooled mean-group panel ARDL: explicit ECM constraints, unit calendars and Gaussian covariance; no imputation.",
    ),
    EstimatorInfo(
        name="mg",
        title="Mean-group panel ARDL",
        family="panel_ardl",
        function="mg",
        entry="openecon.econometrics.panel_ardl.models:fit_panel_ardl",
        covariances=("nonrobust",),
        default_covariance="nonrobust",
        categorical=False,
        intercept="never",
        panel="required",
        time="required",
        options=(
            Option("p", "int", 1, minimum=1, maximum=12),
            Option("q", "int", 1, minimum=1, maximum=12),
            Option("trend", "str", "c", choices=("n", "c", "ct")),
            Option("max_iterations", "int", 500, minimum=1, maximum=5000),
        ),
        description="Mean-group panel ARDL: explicit ECM constraints, unit calendars and Gaussian covariance; no imputation.",
    ),
    EstimatorInfo(
        name="dfe",
        title="Dynamic fixed-effects panel ARDL",
        family="panel_ardl",
        function="dfe",
        entry="openecon.econometrics.panel_ardl.models:fit_panel_ardl",
        covariances=("nonrobust",),
        default_covariance="nonrobust",
        categorical=False,
        intercept="never",
        panel="required",
        time="required",
        options=(
            Option("p", "int", 1, minimum=1, maximum=12),
            Option("q", "int", 1, minimum=1, maximum=12),
            Option("trend", "str", "c", choices=("n", "c", "ct")),
            Option("max_iterations", "int", 500, minimum=1, maximum=5000),
        ),
        description="Dynamic fixed-effects panel ARDL: explicit ECM constraints, unit calendars and Gaussian covariance; no imputation.",
    ),
)
EXPORTS = {"panel_ardl_hausman": "openecon.econometrics.panel_ardl.models:panel_ardl_hausman"}
