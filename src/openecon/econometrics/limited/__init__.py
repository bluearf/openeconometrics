"""limited estimator family manifest (Torch-free).

Limited-dependent-variable models estimated by OpenEconometrics's own float64 kernels:
``tobit`` (censored normal regression), ``truncreg`` (truncated normal
regression), ``intreg`` (interval regression), ``heckman`` (sample selection by
full maximum likelihood or Heckman's two-step estimator), ``heckprobit``
(probit with sample selection), ``ivprobit`` and ``ivtobit`` (probit / tobit
with continuous endogenous regressors by full maximum likelihood or Newey's
minimum chi-squared two-step estimator). Every likelihood has an analytic score
and Hessian. This module imports only the registry so that spec validation and
``oe.capabilities()`` never load the tensor runtime; see
``docs/econometrics/limited.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_ML = ("nonrobust", "opg", "robust", "cluster")
_ALL_WEIGHTS = ("fweight", "aweight", "pweight", "iweight")
_OFFSET = Role("offset", doc="Column added to the linear predictor with coefficient 1.")
_LL = Option("ll", "number", None,
             doc="Lower limit: outcomes at or below it are left-censored (truncreg: excluded).")
_UL = Option("ul", "number", None,
             doc="Upper limit: outcomes at or above it are right-censored (truncreg: excluded).")
_LL_MIN = Option("ll_at_min", "bool", False,
                 doc="Use the smallest outcome as the lower limit (Stata's ll without a value).")
_UL_MAX = Option("ul_at_max", "bool", False,
                 doc="Use the largest outcome as the upper limit (Stata's ul without a value).")
_SELECT = Role("select", required=True,
               doc="Selection indicator coded 0/1; the outcome is used only where it is 1.")
_SELECT_X = Role("select_x", many=True, required=True,
                 doc="Regressors of the selection equation (a constant is always included).")
_ENDOGENOUS = Role("endogenous", many=True, required=True,
                   doc="Continuous endogenous regressors (instrumented).")
_INSTRUMENTS = Role("instruments", many=True, required=True,
                    doc="Excluded instruments. The exogenous regressors (predictors) are "
                        "instruments for themselves and must not be repeated here.")
_METHOD = Option("method", "str", "ml", choices=("ml", "twostep"),
                 doc="ml: full maximum likelihood (default); twostep: the two-step estimator.")

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="tobit", title="Tobit regression", family="limited",
        entry="openecon.econometrics.limited.censored:fit_tobit", function="tobit",
        stata=("tobit",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="t", outcome="censored",
        roles=(_OFFSET,), options=(_LL, _UL, _LL_MIN, _UL_MAX),
        description="Censored normal regression y* = x'b + e, e ~ N(0, sigma^2), observed as "
                    "max(ll, min(y*, ul)). Newton-Raphson on (b, ln sigma) with analytic "
                    "derivatives from OLS starting values; /sigma by the delta method; LR "
                    "(or Wald F) model test, McFadden pseudo R-squared, t statistics with "
                    "N - df_model degrees of freedom as Stata's tobit. EViews: censored "
                    "regression (normal errors).",
    ),
    EstimatorInfo(
        name="truncreg", title="Truncated regression", family="limited",
        entry="openecon.econometrics.limited.truncated:fit_truncreg", function="truncreg",
        stata=("truncreg",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="truncated",
        roles=(_OFFSET,), options=(_LL, _UL),
        description="Normal regression for a sample truncated to ll < y < ul: the density is "
                    "divided by Pr(ll < y* < ul | x). Observations outside the limits are "
                    "excluded and counted. Newton-Raphson on (b, ln sigma) with analytic "
                    "derivatives; /sigma by the delta method; Wald model test, z statistics.",
    ),
    EstimatorInfo(
        name="intreg", title="Interval regression", family="limited",
        entry="openecon.econometrics.limited.censored:fit_intreg", function="intreg",
        stata=("intreg",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="interval",
        roles=(
            Role("upper", required=True,
                 doc="Upper bound of the outcome interval; the model outcome is the lower "
                     "bound. A missing lower bound is left-censored data, a missing upper "
                     "bound right-censored data, equal bounds are point data."),
            _OFFSET,
        ),
        description="Normal regression when each outcome is known as a point, an interval "
                    "[y_low, y_high], or a left- or right-censored value. Newton-Raphson on "
                    "(b, ln sigma) with analytic derivatives; reports /lnsigma and sigma, an "
                    "LR (or Wald) model test and z statistics.",
    ),
    EstimatorInfo(
        name="heckman", title="Heckman selection model", family="limited",
        entry="openecon.econometrics.limited.heckman:fit_heckman", function="heckman",
        stata=("heckman", "heckman, twostep"), covariances=_ML,
        default_covariance="nonrobust", weights=_ALL_WEIGHTS, cluster_dimensions=2,
        inference="z", roles=(_SELECT, _SELECT_X), options=(_METHOD,),
        description="Regression with sample selection: y = x'b + u1 is observed when "
                    "z'g + u2 > 0, corr(u1, u2) = rho. method='ml': full maximum likelihood "
                    "on (b, g, athrho, lnsigma) with analytic score and Hessian, started at "
                    "the two-step estimates. method='twostep': Heckman's (1979) probit + "
                    "inverse-Mills-ratio regression with its consistent covariance. Reports "
                    "rho, sigma, lambda and the test of rho = 0. EViews: Heckman selection.",
    ),
    EstimatorInfo(
        name="heckprobit", title="Probit model with sample selection", family="limited",
        entry="openecon.econometrics.limited.heckprobit:fit_heckprobit", function="heckprobit",
        stata=("heckprobit",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="binary",
        roles=(_SELECT, _SELECT_X),
        description="Binary outcome y = 1[x'b + u1 > 0] observed when z'g + u2 > 0 with "
                    "correlated standard normal errors. Full maximum likelihood on "
                    "(b, g, athrho) with the bivariate normal distribution function, analytic "
                    "score and Hessian; reports rho and the LR (or Wald) test of rho = 0.",
    ),
    EstimatorInfo(
        name="ivprobit", title="Probit model with continuous endogenous regressors",
        family="limited", entry="openecon.econometrics.limited.ivmodels:fit_ivprobit",
        function="ivprobit", stata=("ivprobit", "ivprobit, twostep"), covariances=_ML,
        default_covariance="nonrobust", predictors="optional", weights=_ALL_WEIGHTS,
        cluster_dimensions=2, inference="z", outcome="binary",
        roles=(_ENDOGENOUS, _INSTRUMENTS), options=(_METHOD,),
        description="Probit with endogenous continuous regressors y2 = Z Pi + v, (u, v) "
                    "jointly normal. method='ml': full maximum likelihood of the recursive "
                    "system with analytic score and Hessian; method='twostep': Newey's (1987) "
                    "minimum chi-squared estimator. Wald test of exogeneity and first-stage "
                    "summaries.",
    ),
    EstimatorInfo(
        name="ivtobit", title="Tobit model with continuous endogenous regressors",
        family="limited", entry="openecon.econometrics.limited.ivmodels:fit_ivtobit",
        function="ivtobit", stata=("ivtobit", "ivtobit, twostep"), covariances=_ML,
        default_covariance="nonrobust", predictors="optional", weights=_ALL_WEIGHTS,
        cluster_dimensions=2, inference="z", outcome="censored",
        roles=(_ENDOGENOUS, _INSTRUMENTS), options=(_METHOD, _LL, _UL, _LL_MIN, _UL_MAX),
        description="Tobit with endogenous continuous regressors: the censored outcome "
                    "equation and the reduced forms of the endogenous regressors are "
                    "estimated jointly by maximum likelihood (analytic score and Hessian), or "
                    "by Newey's (1987) minimum chi-squared two-step estimator. Wald test of "
                    "exogeneity and first-stage summaries.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {}
