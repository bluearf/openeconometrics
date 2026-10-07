"""arch estimator family manifest (Torch-free).

``arch`` fits the conditional-heteroskedasticity family (Stata's ``arch``, EViews'
ARCH estimation): ARCH, GARCH, GJR/threshold GARCH, EGARCH, power ARCH and
integrated GARCH variance equations, with an optional ARCH-in-mean term, ARMA
disturbances, variance regressors, and normal, Student-t or generalized-error
innovations. Estimation is full maximum likelihood with analytic gradients;
``oe.forecast`` / ``oe.arch_forecast`` produce multi-step mean and variance
forecasts. Everything is implemented in OpenEconometrics on float64 tensors; see
``docs/econometrics/arch.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

MODELS = ("arch", "garch", "gjr", "tarch", "egarch", "parch", "igarch")
DISTRIBUTIONS = ("normal", "t", "ged")
ARCH_IN_MEAN = ("variance", "sd", "log")

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="arch", title="ARCH family regression", family="arch",
        entry="openecon.econometrics.arch.estimators:fit_arch", function="arch",
        stata=("arch",),
        covariances=("opg", "nonrobust", "robust"), default_covariance="opg",
        predictors="optional", categorical=True, intercept="optional", time="optional",
        inference="z",
        roles=(
            Role("variance_x", many=True,
                 doc="Regressors of the conditional-variance equation (Stata's het()): "
                     "multiplicative heteroskedasticity exp(l0 + z'l) in place of the variance "
                     "constant, or additive terms of ln h for EGARCH."),
        ),
        options=(
            Option("model", "str", "garch", choices=MODELS,
                   doc="Variance equation: arch, garch (Bollerslev), gjr / tarch "
                       "(Glosten-Jagannathan-Runkle threshold terms), egarch (Nelson), parch "
                       "(power ARCH with an estimated power) or igarch (persistence fixed at 1)."),
            Option("dist", "str", "normal", choices=DISTRIBUTIONS,
                   doc="Innovation distribution: normal, t (Student t, estimated degrees of "
                       "freedom, /lndfm2 = ln(df - 2)) or ged (generalized error distribution, "
                       "/lnshape = ln(shape))."),
            Option("arch", "list[int]", None,
                   doc="Lags of the ARCH terms (default [1]); lags of the earch / earch_a pairs "
                       "for egarch and of the arch and tarch terms for gjr."),
            Option("garch", "list[int]", None,
                   doc="Lags of the GARCH terms (default [1]; none for model='arch')."),
            Option("archm", "str", None, choices=ARCH_IN_MEAN,
                   doc="ARCH-in-mean term: the conditional variance, its square root (sd) or "
                       "its logarithm enters the mean equation."),
            Option("ar", "list[int]", None,
                   doc="Lags of the autoregressive terms of the disturbance."),
            Option("ma", "list[int]", None,
                   doc="Lags of the moving-average terms of the disturbance."),
            Option("test_lags", "int", None, minimum=1,
                   doc="Lags of the ARCH-LM and Ljung-Box diagnostics of the standardized "
                       "residuals (defaults: 5 and min(floor(N/2) - 2, 40))."),
            Option("max_iterations", "int", 500, minimum=1,
                   doc="Maximum number of BFGS iterations."),
            Option("tolerance", "float", 1e-8,
                   doc="Gradient tolerance of the optimizer."),
        ),
        description="Regression with conditionally heteroskedastic errors, Stata's arch: "
                    "y_t = x_t b + psi g(h_t) + ARMA terms + e_t, e_t = sqrt(h_t) z_t, with an "
                    "ARCH, GARCH, GJR/TARCH, EGARCH, power-ARCH or IGARCH equation for h_t and "
                    "normal, Student-t or GED z_t. Full maximum likelihood by BFGS with analytic "
                    "gradients; OPG (the default, as in Stata), observed-information or robust "
                    "(quasi-ML sandwich) covariance. The series must be regularly spaced without "
                    "gaps; without a time column the row order is the time order.",
    ),
)

# oe.forecast(result, steps, ...) dispatches here for arch results.
FORECAST: dict[str, str] = {"arch": "openecon.econometrics.arch.forecast:arch_forecast"}

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
# The ARCH-LM test itself (oe.archlm) is published by the arima family.
EXPORTS: dict[str, str] = {
    "arch_forecast": "openecon.econometrics.arch.forecast:arch_forecast",
}
