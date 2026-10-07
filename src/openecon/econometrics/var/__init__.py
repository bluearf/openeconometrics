"""var estimator family manifest (Torch-free).

``var`` fits vector autoregressions by equation-by-equation OLS with Stata's
conventions and stores the lag-order statistics, Granger causality, lag
exclusion, stability, normality and LM autocorrelation tests, impulse
responses and variance decompositions in the result. ``vec`` fits a vector
error-correction model by Johansen's maximum likelihood. The family also
publishes ``varsoc`` (lag-order selection), ``vecrank`` (Johansen trace and
maximum-eigenvalue tests), ``irf`` (impulse responses and variance
decompositions as a table), ``tycausality`` (Toda-Yamamoto modified Wald tests
of first-base-lag restrictions in an augmented levels VAR), and the forecast functions ``var_forecast`` and
``vec_forecast`` (``oe.forecast`` dispatches to them). Everything is
implemented in OpenEconometrics on float64 tensors; see ``docs/econometrics/var.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_SYSTEM = Role("system", many=True, required=True,
               doc="All endogenous variables in model order; the first one is also the outcome. "
                   "The order is the Cholesky order of orthogonalized impulse responses.")

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="var", title="Vector autoregression", family="var",
        entry="openecon.econometrics.var.estimators:fit_var", function="var",
        stata=("var", "varsoc", "varstable", "vargranger", "varwle", "varnorm", "varlmar",
               "irf create", "fcast compute"),
        covariances=("nonrobust", "robust"), default_covariance="nonrobust",
        predictors="optional", categorical=True, intercept="optional", time="optional",
        inference="z", roles=(_SYSTEM,),
        options=(
            Option("lags", "int", 2, minimum=1,
                   doc="p: lags 1..p of every endogenous variable enter each equation."),
            Option("constant", "bool", True, doc="Include a constant in every equation."),
            Option("trend", "bool", False, doc="Include a linear trend t = 1, 2, ..."),
            Option("small", "bool", False,
                   doc="t and F statistics with T - m degrees of freedom; standard errors use "
                       "the equation's degrees of freedom (Stata's small)."),
            Option("dfk", "bool", False,
                   doc="Estimate Sigma with the divisor T - m instead of T (Stata's dfk)."),
            Option("maxlag", "int", None, minimum=0,
                   doc="Highest lag of the lag-order selection table (default: lags)."),
            Option("irf_steps", "int", 8, minimum=1, maximum=200,
                   doc="Horizon of the impulse responses stored in the result."),
            Option("irf_kinds", "list[str]", None,
                   doc="Stored responses: simple, orthogonalized, generalized (default all)."),
            Option("lm_lags", "int", 2, minimum=0, maximum=24,
                   doc="Lags of the residual autocorrelation LM test."),
        ),
        description="VAR(p) with optional exogenous regressors, constant and trend, estimated by "
                    "OLS equation by equation (one QR for the system). Stata conventions: "
                    "maximum-likelihood Sigma (divisor T) and z statistics by default, dfk and "
                    "small for degrees-of-freedom adjustments. The result carries lag-order "
                    "selection, Granger causality, lag-exclusion, normality and LM "
                    "autocorrelation tests, the stability check, impulse responses "
                    "(simple, orthogonalized, generalized) with delta-method standard errors "
                    "and the forecast-error variance decomposition.",
    ),
    EstimatorInfo(
        name="vec", title="Vector error-correction model", family="var",
        entry="openecon.econometrics.var.vec:fit_vec", function="vec",
        stata=("vec", "vecrank", "vecstable", "vecnorm", "veclmar", "fcast compute"),
        covariances=("nonrobust",), default_covariance="nonrobust",
        predictors="none", categorical=False, intercept="optional", time="optional",
        inference="z", roles=(_SYSTEM,),
        options=(
            Option("lags", "int", 2, minimum=1,
                   doc="Lags of the underlying VAR in levels; the VEC model has lags - 1 lagged "
                       "differences."),
            Option("rank", "int", 1, minimum=1,
                   doc="Number of cointegrating equations, 1 <= rank < K."),
            Option("trend", "str", "constant",
                   choices=("none", "rconstant", "constant", "rtrend", "trend"),
                   doc="Deterministic terms (Stata's trend()): none; rconstant = constant "
                       "restricted to the cointegrating equations; constant = unrestricted "
                       "constant; rtrend = unrestricted constant and a trend restricted to the "
                       "cointegrating equations; trend = unrestricted constant and trend."),
            Option("lm_lags", "int", 2, minimum=0, maximum=24,
                   doc="Lags of the residual autocorrelation LM test."),
        ),
        description="Cointegrated VAR in error-correction form estimated by Johansen's maximum "
                    "likelihood (reduced-rank regression) with Johansen's normalization of the "
                    "cointegrating vectors, Stata's five trend specifications and its T - d "
                    "small-sample covariance. The deterministic terms are chosen by the trend "
                    "option (the intercept field of the spec is not used). The result carries "
                    "the cointegrating equations with standard errors, adjustment coefficients, "
                    "the rank tests, stability, normality and LM autocorrelation tests.",
    ),
)

# oe.forecast(result, steps, ...) dispatches on result.spec.estimator.
FORECAST: dict[str, str] = {
    "var": "openecon.econometrics.var.postestimation:var_forecast",
    "vec": "openecon.econometrics.var.postestimation:vec_forecast",
}

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "varsoc": "openecon.econometrics.var.estimators:varsoc",
    "irf": "openecon.econometrics.var.postestimation:irf",
    "var_forecast": "openecon.econometrics.var.postestimation:var_forecast",
    "vec_forecast": "openecon.econometrics.var.postestimation:vec_forecast",
    "vecrank": "openecon.econometrics.var.johansen:vecrank",
    "tycausality": "openecon.econometrics.var.toda_yamamoto:tycausality",
}
