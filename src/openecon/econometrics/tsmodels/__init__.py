"""tsmodels estimator family manifest (Torch-free).

Further time-series models of Stata and EViews, implemented in OpenEconometrics on
float64 tensors (see ``docs/econometrics/tsmodels.md``):

- ``prais``: Prais-Winsten / Cochrane-Orcutt regression with AR(1) errors;
- ``ardl``: autoregressive distributed lag models with automatic lag selection,
  the error-correction form and the Pesaran-Shin-Smith bounds test;
- ``ucm``: unobserved-components models (local level, local linear trend,
  seasonal, cycle) by exact diffuse Kalman-filter maximum likelihood;
- ``mswitch``: Markov-switching dynamic regression and autoregression;
- ``threshold``: threshold regression (Hansen 2000) with sequential thresholds;
- ``tsfilter``: Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald and
  Hamilton filters (a table-returning procedure).
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_COMMON_LS = (
    Option("max_iterations", "int", 100, minimum=1, maximum=1000,
           doc="Maximum number of rho updates."),
    Option("tolerance", "float", 1e-6, minimum=0.0,
           doc="Convergence tolerance on the change in rho."),
)

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="prais", title="Prais-Winsten AR(1) regression", family="tsmodels",
        entry="openecon.econometrics.tsmodels.prais:fit_prais", function="prais",
        stata=("prais",),
        covariances=("nonrobust", "robust", "HC1", "HC2", "HC3"), default_covariance="nonrobust",
        predictors="required", categorical=True, intercept="optional", time="optional",
        inference="t",
        options=(
            Option("method", "str", "prais", choices=("prais", "corc"),
                   doc="prais: Prais-Winsten (first observation kept, transformed by "
                       "sqrt(1-rho^2)); corc: Cochrane-Orcutt (first observation dropped)."),
            Option("rhotype", "str", "regress",
                   choices=("regress", "freg", "tscorr", "dw", "theil", "nagar"),
                   doc="Formula for rho from the residuals (Stata's rhotype())."),
            Option("twostep", "bool", False,
                   doc="Stop after the first transformed regression (two-step estimator)."),
            *_COMMON_LS,
        ),
        description="Linear regression with AR(1) errors by iterated feasible GLS on "
                    "quasi-differenced data (Stata's prais; EViews LS with AR(1)). "
                    "Prais-Winsten keeps the first observation, Cochrane-Orcutt drops it; rho by "
                    "any of Stata's rhotype formulas; t inference on the transformed regression.",
    ),
    EstimatorInfo(
        name="ardl", title="ARDL regression", family="tsmodels",
        entry="openecon.econometrics.tsmodels.ardl:fit_ardl", function="ardl",
        stata=("ardl", "ardl, ec", "estat ectest"),
        covariances=("nonrobust", "robust", "HC1", "HC2", "HC3"), default_covariance="nonrobust",
        predictors="required", categorical=False, intercept="always", time="optional",
        inference="t",
        roles=(Role("exog", many=True,
                    doc="Exogenous regressors entering only contemporaneously (Stata's exog())."),),
        options=(
            Option("lags", "list[int]", None,
                   doc="[p, q_1, ..., q_k]: lag order of the outcome and of each regressor; "
                       "omit for automatic selection."),
            Option("maxlags", "list[int]", [4],
                   doc="Largest lags searched: one value for all variables or k + 1 values "
                       "(outcome lags 1..max, regressor lags 0..max)."),
            Option("ic", "str", "aic", choices=("aic", "bic"),
                   doc="Information criterion of the lag search."),
            Option("trend", "str", "constant", choices=("constant", "trend", "none"),
                   doc="Deterministic terms: constant, constant and linear trend, or none."),
            Option("restricted", "bool", False,
                   doc="Restrict the constant (case II) or the trend (case IV) to the long run."),
            Option("ec", "bool", False, doc="Report the error-correction parameterization."),
        ),
        description="Autoregressive distributed lag model by OLS with exhaustive AIC/BIC lag "
                    "selection, the error-correction form (speed of adjustment, long-run "
                    "coefficients by the delta method) and the Pesaran-Shin-Smith (2001) bounds "
                    "F and t tests with asymptotic I(0)/I(1) critical values.",
    ),
    EstimatorInfo(
        name="ucm", title="Unobserved-components model", family="tsmodels",
        entry="openecon.econometrics.tsmodels.ucm:fit_ucm", function="ucm",
        stata=("ucm",),
        covariances=("nonrobust", "robust", "opg"), default_covariance="nonrobust",
        predictors="optional", categorical=True, intercept="never", time="optional",
        inference="z",
        options=(
            Option("model", "str", "rwalk",
                   choices=("rwalk", "llevel", "lltrend", "rwdrift", "strend", "smooth_trend",
                            "rtrend", "lldtrend", "dtrend", "dconstant", "ntrend", "none"),
                   doc="Trend specification (Stata's ucm model() names; smooth_trend = strend)."),
            Option("seasonal", "int", None, minimum=2,
                   doc="Period of a stochastic dummy seasonal component."),
            Option("cycle", "bool", False, doc="Add a stochastic damped cycle."),
            Option("cycle_frequency", "float", None,
                   doc="Starting value of the cycle frequency in radians (0, pi)."),
            Option("max_iterations", "int", 200, minimum=1, doc="BFGS iteration limit."),
        ),
        description="Unobserved-components (structural time-series) model: level, slope, "
                    "seasonal and cycle components plus regressors, by exact maximum likelihood "
                    "with the exact diffuse Kalman filter; smoothed components and forecasts.",
    ),
    EstimatorInfo(
        name="mswitch", title="Markov-switching regression", family="tsmodels",
        entry="openecon.econometrics.tsmodels.mswitch:fit_mswitch", function="mswitch",
        stata=("mswitch dr", "mswitch ar"),
        covariances=("nonrobust", "robust", "opg"), default_covariance="nonrobust",
        predictors="optional", categorical=True, intercept="optional", time="optional",
        inference="z",
        options=(
            Option("states", "int", 2, minimum=2, maximum=6, doc="Number of regimes."),
            Option("switch", "list[str]", None,
                   doc="Terms with state-specific coefficients (default: the constant)."),
            Option("varswitch", "bool", False, doc="State-specific error variance."),
            Option("ar", "list[int]", None,
                   doc="Autoregressive lags (Stata's mswitch ar); the density then depends on "
                       "the states of the lagged periods."),
            Option("arswitch", "bool", False, doc="State-specific AR coefficients."),
            Option("max_iterations", "int", 500, minimum=1, doc="BFGS iterations per start."),
        ),
        description="Markov-switching dynamic regression and autoregression with constant "
                    "transition probabilities by exact maximum likelihood (blocked Hamilton "
                    "filter, analytic Fisher-identity gradient, Kim smoother).",
    ),
    EstimatorInfo(
        name="threshold", title="Threshold regression", family="tsmodels",
        entry="openecon.econometrics.tsmodels.threshold:fit_threshold", function="threshold",
        stata=("threshold",),
        covariances=("nonrobust", "robust", "HC1", "HC2", "HC3"), default_covariance="nonrobust",
        predictors="required", categorical=True, intercept="optional", time="optional",
        inference="t",
        roles=(Role("threshold_var", required=True, doc="The threshold variable q."),),
        options=(
            Option("nthresholds", "int", 1, minimum=1, maximum=5, doc="Number of thresholds."),
            Option("trim", "float", 0.1, doc="Minimum share of observations in every region."),
            Option("regions", "list[str]", None,
                   doc="Terms with region-specific coefficients (default: all terms)."),
            Option("bootstrap", "int", 0, minimum=0, maximum=100_000,
                   doc="Replications of Hansen's fixed-regressor bootstrap test of no "
                       "threshold effect (0: not computed)."),
            Option("seed", "int", 12345, doc="Random seed of the bootstrap."),
        ),
        description="Threshold regression (Hansen 2000): region-specific coefficients split "
                    "at thresholds of a threshold variable chosen by exhaustive SSR grid "
                    "search, sequential estimation of several thresholds with refinement, LR "
                    "confidence set and Hansen's bootstrap test of a threshold effect.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
_ARDL_INFO = next(info for info in ESTIMATORS if info.name == "ardl")
ESTIMATORS += (
    EstimatorInfo(
        name="nardl", title="Nonlinear asymmetric ARDL", family="tsmodels",
        entry="openecon.econometrics.tsmodels.nardl:fit_nardl", function="nardl",
        covariances=_ARDL_INFO.covariances, default_covariance="nonrobust",
        predictors="required", categorical=False, intercept="always", time="optional",
        inference="t", roles=_ARDL_INFO.roles,
        options=tuple(Option("ec", "bool", True, doc="Report the error-correction form.")
                      if option.name == "ec" else option for option in _ARDL_INFO.options) + (
            Option("asymmetric", "list[str]", None,
                   doc="Predictors split into positive/negative partial sums (default: all x)."),
            Option("long_run_symmetric", "list[str]", None,
                   doc="Asymmetric predictors whose positive/negative long-run effects are "
                       "constrained to equality; other predictors remain unrestricted."),
            Option("short_run_symmetric", "list[str]", None,
                   doc="Asymmetric predictors with lagwise conditional-EC short-run symmetry; "
                       "absent lags are zero. This is stronger than cumulative symmetry."),
            Option("time_delta", "str", None,
                   doc="Explicit positive fixed interval for datetime time, e.g. '1D'. "
                       "Use integer period codes for calendar months/quarters."),
        ),
        description="Shin-Yu-Greenwood-Nimmo NARDL: float64 partial sums, "
                    "shared ARDL QR and lag search, error correction, Wald symmetry tests "
                    "and dynamic multipliers; independently selected long-run/lagwise "
                    "short-run symmetry can be imposed by constrained QR. "
                    "Specialized bounds critical values are not validated.",
    ),
)

EXPORTS: dict[str, str] = {
    "dynamic_predict": "openecon.econometrics.postest.dynamic_prediction:dynamic_predict",
    "nardl_multipliers": "openecon.econometrics.tsmodels.nardl:nardl_multipliers",
    "tsfilter": "openecon.econometrics.tsmodels.filters:tsfilter",
    "ucm_components": "openecon.econometrics.tsmodels.ucm:ucm_components",
    "ucm_forecast": "openecon.econometrics.tsmodels.ucm:ucm_forecast",
    "mswitch_probabilities": "openecon.econometrics.tsmodels.mswitch:mswitch_probabilities",
}

# oe.forecast(result, steps, ...) dispatches here.
FORECAST: dict[str, str] = {"ucm": "openecon.econometrics.tsmodels.ucm:ucm_forecast"}
