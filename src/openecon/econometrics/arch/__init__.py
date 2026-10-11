"""arch estimator family manifest (Torch-free).

``arch`` fits the conditional-heteroskedasticity family (Stata's ``arch``, EViews'
ARCH estimation): ARCH, GARCH, GJR/threshold GARCH, EGARCH, power ARCH and
integrated GARCH variance equations, with an optional ARCH-in-mean term, ARMA
disturbances, variance regressors, and normal, Student-t, generalized-error,
Hansen skewed-t or Fernandez-Steel skewed-GED innovations. Estimation is full
maximum likelihood with analytic gradients; ``oe.forecast`` / ``oe.arch_forecast``
produce multi-step mean and variance forecasts. Everything is implemented in
OpenEconometrics on float64 tensors; see ``docs/econometrics/arch.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

MODELS = ("arch", "garch", "gjr", "tarch", "egarch", "parch", "igarch", "cgarch", "figarch")
DISTRIBUTIONS = ("normal", "t", "ged", "skewt", "sged")
ARCH_IN_MEAN = ("variance", "sd", "log")
# The bounded Dataset (streaming) route implements these; the other models and
# distributions run on resident tables only and are refused there with an explicit error.
STREAMING_MODELS = ("arch", "garch", "gjr", "tarch", "egarch", "parch", "igarch")
STREAMING_DISTRIBUTIONS = ("normal", "t", "ged")

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
                       "(power ARCH with an estimated power), igarch (persistence fixed at 1) "
                       "cgarch (Engle-Lee component GARCH(1,1): permanent and transitory "
                       "variance components) or figarch (Baillie-Bollerslev-Mikkelsen "
                       "FIGARCH(p,d,q), p, q in {0, 1}, truncated ARCH(infinity) form)."),
            Option("dist", "str", "normal", choices=DISTRIBUTIONS,
                   doc="Innovation distribution: normal, t (Student t, estimated degrees of "
                       "freedom, /lndfm2 = ln(df - 2)), ged (generalized error distribution, "
                       "/lnshape = ln(shape)), skewt (Hansen's skewed t: /lndfm2 and "
                       "/atlambda = atanh(lambda)) or sged (Fernandez-Steel skewed GED: "
                       "/lnshape and /lnxi = ln(xi))."),
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
            Option("truncation", "int", None, minimum=1, maximum=5000,
                   doc="FIGARCH only: lags of the truncated ARCH(infinity) representation "
                       "(default 1000, as Python arch)."),
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

ESTIMATORS += (
    EstimatorInfo(
        name="garch_midas", title="GARCH-MIDAS regression", family="arch",
        entry="openecon.econometrics.arch.midas:fit_garch_midas", function="garch_midas",
        covariances=("robust", "opg", "nonrobust"), default_covariance="robust",
        predictors="none", categorical=False, intercept="optional", time="optional",
        inference="z",
        roles=(
            Role("period", required=True,
                 doc="Integer low-frequency period of each observation (consecutive integers "
                     "that never decrease in time order, e.g. 12 * year + month)."),
            Role("covariate",
                 doc="Low-frequency covariate of the long-run component, constant within "
                     "each period; default the realized variance of the period."),
        ),
        options=(
            Option("lags", "int", 12, minimum=1, maximum=400,
                   doc="K, the number of low-frequency lags of the long-run component."),
            Option("weighting", "str", "beta_restricted",
                   choices=("beta_restricted", "beta_unrestricted"),
                   doc="Beta lag weights with w1 = 1 (restricted, estimates w2) or with both "
                       "w1 and w2 estimated (unrestricted, needs lags >= 3)."),
            Option("asymmetric", "bool", True,
                   doc="GJR short-run component (gamma) rather than GARCH(1,1)."),
            Option("max_iterations", "int", 500, minimum=1,
                   doc="Maximum number of BFGS iterations."),
            Option("tolerance", "float", 1e-8, doc="Gradient tolerance of the optimizer."),
        ),
        description="Engle-Ghysels-Sohn GARCH-MIDAS with the conventions of mfGARCH (Conrad "
                    "and Kleen): y_t = mu + sqrt(tau_s g_t) Z_t with a unit-mean GARCH or GJR "
                    "short-run component g and a long-run component ln tau_s = m + theta "
                    "sum_k phi_k(w1, w2) X_(s-k) driven by K lags of a low-frequency covariate "
                    "(or the realized variance) with beta weights. Gaussian quasi-maximum "
                    "likelihood with analytic gradients; robust (default), OPG or "
                    "observed-information covariance.",
    ),
)

# oe.forecast(result, steps, ...) dispatches here for arch results.
FORECAST: dict[str, str] = {"arch": "openecon.econometrics.arch.forecast:arch_forecast",
                            "garch_midas": "openecon.econometrics.arch.midas:"
                                           "garch_midas_forecast"}

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
# The ARCH-LM test itself (oe.archlm) is published by the arima family.
EXPORTS: dict[str, str] = {
    "arch_forecast": "openecon.econometrics.arch.forecast:arch_forecast",
    "garch_midas_forecast": "openecon.econometrics.arch.midas:garch_midas_forecast",
}
