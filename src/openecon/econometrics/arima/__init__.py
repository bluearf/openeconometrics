"""arima estimator family manifest (Torch-free).

``arima`` fits ARIMA, seasonal ARIMA and ARMAX models (regression with ARMA
disturbances, Stata's ``arima``) by exact Gaussian maximum likelihood or by
conditional sum of squares. The family also publishes ``forecast`` (dynamic
forecasts from a fitted model; ``arima_forecast`` is the same function under a
name no other family can claim), the series diagnostics ``corrgram``,
``wntestq``, ``jarque_bera`` and ``archlm``, and exponential smoothing
(``tssmooth``); these return result tables. Everything is implemented in
OpenEconometrics on float64 tensors; see ``docs/econometrics/arima.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="arima", title="ARIMA regression", family="arima",
        entry="openecon.econometrics.arima.estimators:fit_arima", function="arima",
        stata=("arima",),
        covariances=("opg", "nonrobust", "robust"), default_covariance="opg",
        predictors="optional", categorical=True, intercept="optional", time="optional",
        inference="z",
        options=(
            Option("order", "list[int]", None, required=True,
                   doc="[p, d, q]: autoregressive order, differences, moving-average order."),
            Option("seasonal", "list[int]", None,
                   doc="[P, D, Q]: multiplicative seasonal AR order, seasonal differences and "
                       "seasonal MA order (needs period)."),
            Option("period", "int", None, minimum=2,
                   doc="Seasonal period s (12 for monthly data, 4 for quarterly)."),
            Option("method", "str", "ml", choices=("ml", "css"),
                   doc="ml: exact Gaussian likelihood with the stationary initial state "
                       "(Stata's default); css: conditional likelihood with presample "
                       "disturbances at zero (Stata's condition option)."),
            Option("constant", "bool", True,
                   doc="Include a constant in the regression for the (differenced) outcome."),
            Option("ljung_lags", "int", None, minimum=1,
                   doc="Lags of the Ljung-Box test on the residuals; default "
                       "min(floor(N/2) - 2, 40)."),
            Option("max_iterations", "int", 200, minimum=1,
                   doc="Maximum number of BFGS iterations."),
            Option("tolerance", "float", 1e-8,
                   doc="Gradient tolerance of the optimizer."),
        ),
        description="Regression with ARMA disturbances, Stata's arima: y_t = x_t b + u_t with "
                    "phi(L) Phi(L^s) (1-L)^d (1-L^s)^D u_t = theta(L) Theta(L^s) e_t. Exact "
                    "maximum likelihood (the Kalman-filter likelihood, evaluated in closed form "
                    "without a time loop) or conditional sum of squares; OPG (default, as in "
                    "Stata), observed-information or robust covariance. The series must be "
                    "regularly spaced without gaps; without a time column the row order is the "
                    "time order.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
# oe.forecast(result, steps, ...) dispatches here for arima results.
FORECAST: dict[str, str] = {"arima": "openecon.econometrics.arima.forecast:forecast"}

EXPORTS: dict[str, str] = {
    "arima_forecast": "openecon.econometrics.arima.forecast:forecast",
    "tssmooth": "openecon.econometrics.arima.smoothing:tssmooth",
    "corrgram": "openecon.econometrics.arima.diagnostics:corrgram",
    "wntestq": "openecon.econometrics.arima.diagnostics:wntestq",
    "jarque_bera": "openecon.econometrics.arima.diagnostics:jarque_bera",
    "archlm": "openecon.econometrics.arima.diagnostics:archlm",
}
