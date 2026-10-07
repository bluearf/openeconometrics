"""glm estimator family manifest (Torch-free).

Generalized linear models and their count / fractional relatives: ``glm``
(six families, ten links, Newton-Raphson or IRLS), ``poisson``, ``nbreg``
(NB2 and NB1 by full maximum likelihood), ``cloglog``, ``fracreg``
(fractional logit / probit quasi-likelihood), ``betareg`` (beta regression
with a precision equation) and ``ppmlhdfe`` (Poisson pseudo-maximum likelihood
with absorbed fixed effects). The legacy ``logit`` and ``probit`` estimators
stay in the core; richer binary models are reached through
``glm(family='binomial')``. This module imports only the registry so that spec
validation and ``oe.capabilities()`` never load the tensor runtime; see
``docs/econometrics/glm.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

FAMILIES = ("gaussian", "binomial", "poisson", "gamma", "inverse_gaussian", "nbinomial")
LINKS = ("identity", "log", "logit", "probit", "cloglog", "loglog", "power", "reciprocal",
         "inverse_squared", "nbinomial")
_ML = ("nonrobust", "opg", "robust", "cluster")
_ALL_WEIGHTS = ("fweight", "aweight", "pweight", "iweight")
_OFFSET = Role("offset", doc="Column added to the linear predictor with coefficient 1.")
_EXPOSURE = Role("exposure", doc="Positive exposure; ln(exposure) is added to the linear "
                                 "predictor with coefficient 1.")

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="glm", title="Generalized linear model", family="glm",
        entry="openecon.econometrics.glm.glm:fit_glm", function="glm", stata=("glm",),
        covariances=_ML, default_covariance="nonrobust", weights=_ALL_WEIGHTS,
        cluster_dimensions=2, inference="z", predictors="optional",
        roles=(_OFFSET, _EXPOSURE,
               Role("trials", doc="Binomial denominator n_i; the outcome is then the count of "
                                  "successes 0..n_i.")),
        options=(
            Option("family", "str", "gaussian", choices=FAMILIES,
                   doc="Distribution family (variance function)."),
            Option("link", "str", None, choices=LINKS,
                   doc="Link function; default is the family's Stata default: identity, logit, "
                       "log, reciprocal, inverse_squared (power -2), log."),
            Option("power", "float", None, doc="Exponent a of link='power': eta = mu^a."),
            Option("dispersion", "float", 1.0,
                   doc="Fixed overdispersion k of family='nbinomial' (Var = mu + k mu^2)."),
            Option("scale", "json", None,
                   doc="Dispersion multiplying the conventional covariance: 'x2' (Pearson "
                       "chi2/df), 'dev' (deviance/df) or a positive number. Default: 1 for "
                       "binomial, poisson, nbinomial and 'x2' for the other families."),
            Option("optimizer", "str", "ml", choices=("ml", "irls"),
                   doc="ml: Newton-Raphson with the observed Hessian (OIM); irls: Fisher "
                       "scoring with the expected information (EIM)."),
            Option("max_iterations", "int", 100, minimum=1, doc="Iteration limit."),
            Option("tolerance", "float", 1e-10,
                   doc="Convergence tolerance: scaled gradient and relative step (ml) or "
                       "relative deviance change (irls)."),
        ),
        description="Stata's glm: gaussian, binomial (with trials), poisson, gamma, inverse "
                    "Gaussian and fixed-dispersion negative binomial families with identity, "
                    "log, logit, probit, cloglog, loglog, power and nbinomial links; analytic "
                    "Newton-Raphson (OIM) or IRLS (EIM); dispersion-scaled covariance.",
    ),
    EstimatorInfo(
        name="poisson", title="Poisson regression", family="glm",
        entry="openecon.econometrics.glm.commands:fit_poisson", function="poisson",
        stata=("poisson", "estat gof"), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="count",
        predictors="optional", roles=(_OFFSET, _EXPOSURE),
        description="Poisson maximum likelihood (log link) by Newton-Raphson with LR or Wald "
                    "model test, McFadden pseudo R-squared and the deviance and Pearson "
                    "goodness-of-fit tests of estat gof.",
    ),
    EstimatorInfo(
        name="nbreg", title="Negative binomial regression", family="glm",
        entry="openecon.econometrics.glm.nbreg:fit_nbreg", function="nbreg", stata=("nbreg",),
        covariances=_ML, default_covariance="nonrobust", weights=_ALL_WEIGHTS,
        cluster_dimensions=2, inference="z", outcome="count", predictors="optional",
        roles=(_OFFSET, _EXPOSURE),
        options=(
            Option("dispersion", "str", "mean", choices=("mean", "constant"),
                   doc="mean: NB2, Var = mu + alpha mu^2; constant: NB1, Var = mu (1 + delta)."),
        ),
        description="Negative binomial regression by full maximum likelihood over the "
                    "coefficients and ln(alpha) (NB2) or ln(delta) (NB1), with the LR test of "
                    "alpha = 0 against Poisson.",
    ),
    EstimatorInfo(
        name="cloglog", title="Complementary log-log regression", family="glm",
        entry="openecon.econometrics.glm.commands:fit_cloglog", function="cloglog",
        stata=("cloglog",), covariances=_ML, default_covariance="nonrobust",
        weights=("fweight", "pweight", "iweight"), cluster_dimensions=2, inference="z",
        outcome="binary", predictors="optional", roles=(_OFFSET,),
        description="Binary complementary log-log maximum likelihood, "
                    "Pr(y = 1) = 1 - exp(-exp(x'b)), with separation detection.",
    ),
    EstimatorInfo(
        name="fracreg", title="Fractional response regression", family="glm",
        entry="openecon.econometrics.glm.commands:fit_fracreg", function="fracreg",
        stata=("fracreg logit", "fracreg probit"), covariances=_ML, default_covariance="robust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="fractional",
        predictors="optional",
        options=(
            Option("link", "str", "logit", choices=("logit", "probit"),
                   doc="Conditional-mean function: logistic or standard normal cdf."),
        ),
        description="Bernoulli quasi-maximum likelihood for an outcome in [0, 1] (Papke and "
                    "Wooldridge) with a logit or probit mean; robust covariance by default.",
    ),
    EstimatorInfo(
        name="betareg", title="Beta regression", family="glm",
        entry="openecon.econometrics.glm.betareg:fit_betareg", function="betareg",
        stata=("betareg",), covariances=_ML, default_covariance="nonrobust",
        weights=("fweight", "pweight", "iweight"), cluster_dimensions=2, inference="z",
        outcome="fractional", predictors="optional",
        roles=(Role("scale", many=True,
                    doc="Regressors of the precision (scale) equation; its constant is always "
                        "included."),),
        options=(
            Option("link", "str", "logit", choices=("logit", "probit", "cloglog", "loglog"),
                   doc="Link of the mean equation."),
            Option("scale_link", "str", "log", choices=("log", "identity", "sqrt"),
                   doc="Link of the precision equation."),
        ),
        description="Beta regression for an outcome strictly inside (0, 1): mean equation and "
                    "precision equation estimated jointly by maximum likelihood with analytic "
                    "derivatives (Ferrari and Cribari-Neto).",
    ),
    EstimatorInfo(
        name="ppmlhdfe", title="Poisson pseudo-likelihood regression with fixed effects",
        family="glm", entry="openecon.econometrics.glm.ppmlhdfe:fit_ppmlhdfe",
        function="ppmlhdfe", stata=("ppmlhdfe",),
        covariances=("robust", "cluster", "nonrobust"), default_covariance="robust",
        weights=("fweight", "aweight", "pweight"), intercept="never", cluster_dimensions=2,
        inference="z", outcome="count",
        roles=(Role("absorb", many=True, required=True, kind="label",
                    doc="One or more categorical variables absorbed as fixed effects."),
               _OFFSET, _EXPOSURE),
        options=(
            Option("drop_singletons", "bool", True,
                   doc="Iteratively drop observations that are alone in a level."),
            Option("tolerance", "float", 1e-8,
                   doc="Relative deviance change at which the IRLS iteration stops."),
            Option("max_iterations", "int", 100, minimum=1, doc="IRLS iteration limit."),
        ),
        description="Correia, Guimaraes and Zylkin's ppmlhdfe: Poisson pseudo-maximum "
                    "likelihood with any number of absorbed fixed-effect dimensions by IRLS "
                    "whose weighted least-squares steps partial the effects out.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {}
