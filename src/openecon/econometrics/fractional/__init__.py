"""Fractional-memory manifest. No tensor runtime is imported by the catalogue."""

from openecon.econometrics.registry import EstimatorInfo, Option

ESTIMATORS = (
    EstimatorInfo(
        name="arfima", title="Conditional Gaussian ARFIMA", family="fractional",
        entry="openecon.econometrics.fractional.models:fit_arfima", function="arfima",
        predictors="none", categorical=False, intercept="optional", time="optional",
        covariances=("nonrobust",), default_covariance="nonrobust", inference="z",
        options=(
            Option("ar", "int", 0, minimum=0, maximum=3),
            Option("ma", "int", 0, minimum=0, maximum=3),
            Option("d", "float", None, minimum=-0.49, maximum=0.49,
                   doc="Fix d; absent means estimate strictly inside (-.49,.49)."),
            Option("terms", "int", 256, minimum=2, maximum=16384,
                   doc="Fixed finite fractional-filter length, including lag zero."),
            Option("burn", "int", 0, minimum=0,
                   doc="Initial innovations excluded from likelihood; all initial values are zero."),
            Option("constant", "bool", True),
            Option("method", "str", "css", choices=("css",),
                   doc="Approximate conditional Gaussian likelihood, finite filter/zero prehistory."),
            Option("max_iterations", "int", 200, minimum=1, maximum=1000),
            Option("tolerance", "float", 1e-8, minimum=1e-12, maximum=1e-4),
        ),
        description="Resident CPU float64 ARFIMA(p,d,q), p/q<=3, stationary/invertible AR/MA "
                    "and |d|<.49. Approximate finite-filter conditional likelihood; no exact "
                    "stationary ML, covariates, weights or Dataset route. See fractional-memory.md.",
    ),
)
EXPORTS = {
    "fracdiff": "openecon.econometrics.fractional.procedures:fracdiff",
    "gph": "openecon.econometrics.fractional.procedures:gph",
}
FORECAST = {"arfima": "openecon.econometrics.fractional.models:forecast"}
