"""count estimator family manifest (Torch-free).

Extensions of the Poisson and negative binomial count models of the glm
family, all by full maximum likelihood with analytic scores and Hessians:

* ``zip`` / ``zinb``: zero-inflated Poisson and negative binomial (NB2)
  regression with a logit or probit inflation equation, the Vuong test against
  the model without inflation and (``zinb``) the LR test of ``alpha = 0``.
* ``tpoisson`` / ``tnbreg``: left-truncated Poisson and negative binomial
  regression (zero-truncated by default; a fixed or observation-specific
  truncation point).
* ``churdle``: Cragg's hurdle model for a continuous bounded outcome (probit
  selection plus a truncated normal or lognormal outcome equation).
* ``hurdle``: the count hurdle model of Mullahy (binary participation plus a
  zero-truncated Poisson or negative binomial).
* ``gnbreg``: generalized negative binomial regression, ``ln(alpha_i) = z_i'd``.

This module imports only the registry so that spec validation and
``oe.capabilities()`` never load the tensor runtime; see
``docs/econometrics/count.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_ML = ("nonrobust", "opg", "robust", "cluster")
_ALL_WEIGHTS = ("fweight", "aweight", "pweight", "iweight")
_OFFSET = Role("offset", doc="Column added to the count equation's linear predictor with "
                             "coefficient 1.")
_EXPOSURE = Role("exposure", doc="Positive exposure; ln(exposure) is added to the count "
                                 "equation's linear predictor with coefficient 1.")
_INFLATE = Role("inflate", many=True,
                doc="Regressors of the inflation equation (probability of an excess zero); "
                    "its constant is always included. Empty: constant-only inflation.")
_INFLATE_LINK = Option("inflate_link", "str", "logit", choices=("logit", "probit"),
                       doc="Link of the inflation equation (probit is Stata's probit option).")
_TRUNCATION = Role("truncation",
                   doc="Column of observation-specific truncation points (nonnegative "
                       "integers); the outcome must exceed it. Replaces the option ll.")
_LL = Option("ll", "int", 0, minimum=0,
             doc="Truncation point: the sample contains only outcomes greater than ll "
                 "(0 = zero-truncated).")
_CENSOR_ROLES = (
    Role("left_limit", doc="Observation-specific nonnegative integer left-censoring points."),
    Role("right_limit", doc="Observation-specific nonnegative integer right-censoring points."),
    Role("censoring", doc="Status column: exact, left, right or interval; omitted: infer from ll/ul."),
    _OFFSET, _EXPOSURE,
)
_CENSOR_OPTIONS = (
    Option("ll", "int", None, minimum=0, doc="Fixed left-censoring point; replaces left_limit."),
    Option("ul", "int", None, minimum=0, doc="Fixed right-censoring point; replaces right_limit."),
)

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="cpoisson", title="Censored Poisson regression", family="count",
        entry="openecon.econometrics.count.censored:fit_cpoisson", function="cpoisson",
        stata=("cpoisson",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="count",
        predictors="optional", roles=_CENSOR_ROLES, options=_CENSOR_OPTIONS,
        description="Poisson likelihood for exact, left-, right- and genuinely interval-"
                    "censored counts, with fixed or observation-specific limits and analytic "
                    "tail derivatives. Without limits this is ordinary Poisson regression.",
    ),
    EstimatorInfo(
        name="cnbreg", title="Censored negative binomial regression", family="count",
        entry="openecon.econometrics.count.censored:fit_cnbreg", function="cnbreg",
        covariances=_ML, default_covariance="nonrobust", weights=_ALL_WEIGHTS,
        cluster_dimensions=2, inference="z", outcome="count", predictors="optional",
        roles=_CENSOR_ROLES,
        options=(*_CENSOR_OPTIONS,
                 Option("dispersion", "str", "mean", choices=("mean", "constant"),
                        doc="mean: NB2, Var=mu+alpha*mu^2; constant: NB1, Var=mu*(1+delta).")),
        description="Censored NB2 or NB1 regression with an estimated log dispersion and "
                    "analytic incomplete-beta likelihood derivatives. This OpenEconometrics API is "
                    "not presented as a built-in Stata cnbreg command.",
    ),
    EstimatorInfo(
        name="zip", title="Zero-inflated Poisson regression", family="count",
        entry="openecon.econometrics.count.zeroinflated:fit_zip", function="zip",
        stata=("zip",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="count",
        predictors="optional", roles=(_INFLATE, _OFFSET, _EXPOSURE),
        options=(_INFLATE_LINK,),
        description="Zero-inflated Poisson: with probability F(z'g) the outcome is a "
                    "structural zero, otherwise Poisson(exp(x'b)). Joint Newton-Raphson with "
                    "analytic derivatives; LR (or Wald) model test, Vuong test against "
                    "Poisson with AIC and BIC corrections.",
    ),
    EstimatorInfo(
        name="zinb", title="Zero-inflated negative binomial regression", family="count",
        entry="openecon.econometrics.count.zeroinflated:fit_zinb", function="zinb",
        stata=("zinb",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="count",
        predictors="optional", roles=(_INFLATE, _OFFSET, _EXPOSURE),
        options=(_INFLATE_LINK,),
        description="Zero-inflated negative binomial (NB2) with /lnalpha: LR test of "
                    "alpha = 0 against the zero-inflated Poisson (chibar2(01)) and Vuong test "
                    "against the negative binomial model.",
    ),
    EstimatorInfo(
        name="tpoisson", title="Truncated Poisson regression", family="count",
        entry="openecon.econometrics.count.truncated:fit_tpoisson", function="tpoisson",
        stata=("tpoisson",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="count",
        predictors="optional", roles=(_TRUNCATION, _OFFSET, _EXPOSURE), options=(_LL,),
        description="Poisson regression for a sample truncated from below (y > ll, "
                    "zero-truncated by default): the likelihood conditions on y > ll with "
                    "the Poisson tail computed from the regularized incomplete gamma "
                    "function.",
    ),
    EstimatorInfo(
        name="tnbreg", title="Truncated negative binomial regression", family="count",
        entry="openecon.econometrics.count.truncated:fit_tnbreg", function="tnbreg",
        stata=("tnbreg",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="count",
        predictors="optional", roles=(_TRUNCATION, _OFFSET, _EXPOSURE),
        options=(
            _LL,
            Option("dispersion", "str", "mean", choices=("mean", "constant"),
                   doc="mean: NB2, Var = mu (1 + alpha mu); constant: NB1, "
                       "Var = mu (1 + delta)."),
        ),
        description="Negative binomial regression for a sample truncated from below "
                    "(y > ll): coefficients and ln(alpha) (or ln(delta)) by Newton-Raphson, "
                    "with the LR test of alpha = 0 against the truncated Poisson model.",
    ),
    EstimatorInfo(
        name="churdle", title="Cragg hurdle regression", family="count",
        entry="openecon.econometrics.count.hurdle:fit_churdle", function="churdle",
        stata=("churdle linear", "churdle exponential"), covariances=_ML,
        default_covariance="nonrobust", weights=_ALL_WEIGHTS, cluster_dimensions=2,
        inference="z", outcome="bounded", predictors="optional",
        roles=(Role("select_x", many=True, required=True,
                    doc="Regressors of the selection equation Pr(y > ll); its constant is "
                        "always included."),),
        options=(
            Option("model", "str", "exponential", choices=("linear", "exponential"),
                   doc="Outcome model above the limit: linear (truncated normal y) or "
                       "exponential (truncated normal ln y)."),
            Option("ll", "number", 0.0,
                   doc="Lower limit: outcomes at or below it are the bounded observations."),
            Option("select_link", "str", "probit", choices=("probit", "logit"),
                   doc="Link of the selection equation (Stata's churdle is probit)."),
        ),
        description="Cragg's hurdle model: a probit selection equation for y > ll and, above "
                    "the limit, a truncated normal regression of y (linear) or of ln y "
                    "(exponential) with /lnsigma; the two parts are estimated jointly.",
    ),
    EstimatorInfo(
        name="hurdle", title="Hurdle count regression", family="count",
        entry="openecon.econometrics.count.hurdle:fit_hurdle", function="hurdle",
        stata=("hplogit", "hnblogit"), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="count",
        predictors="optional",
        roles=(Role("select_x", many=True,
                    doc="Regressors of the participation equation Pr(y > 0); default: the "
                        "regressors of the count equation. Its constant is always included."),
               _OFFSET, _EXPOSURE),
        options=(
            Option("dist", "str", "poisson", choices=("poisson", "nbinomial"),
                   doc="Zero-truncated distribution of the positive counts."),
            Option("zero_link", "str", "logit", choices=("logit", "probit", "cloglog"),
                   doc="Link of the participation equation Pr(y > 0)."),
        ),
        description="Mullahy's hurdle model for counts: a binary model for y > 0 and a "
                    "zero-truncated Poisson or negative binomial (NB2) for the positive "
                    "counts, as the community commands hplogit / hnblogit.",
    ),
    EstimatorInfo(
        name="gnbreg", title="Generalized negative binomial regression", family="count",
        entry="openecon.econometrics.count.gnbreg:fit_gnbreg", function="gnbreg",
        stata=("gnbreg",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="count",
        predictors="optional",
        roles=(Role("lnalpha", many=True,
                    doc="Regressors of ln(alpha); its constant is always included."),
               _OFFSET, _EXPOSURE),
        description="Negative binomial (NB2) regression whose overdispersion varies with "
                    "covariates, ln(alpha_i) = z_i'd, by Newton-Raphson with analytic "
                    "derivatives.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {}
