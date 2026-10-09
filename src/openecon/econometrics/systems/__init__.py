"""systems estimator family manifest (Torch-free).

Systems of equations and further estimators: ``sureg`` (Zellner's seemingly
unrelated regressions, two-step or iterated), ``mvreg`` (multivariate
regression), ``reg3`` (three-stage least squares and its 2SLS / SUR / OLS
variants), ``gmm`` (general nonlinear GMM over moment expressions),
``frontier`` (stochastic production and cost frontiers), ``xtgls`` (panel
feasible GLS) and ``xtpcse`` (Beck-Katz panel-corrected standard errors).
Every estimator is implemented in OpenEconometrics on float64 tensors; see
``docs/econometrics/systems.md``. This module imports only the registry.

Multi-equation specifications (sureg, reg3) carry the system in the option
``equations`` (a JSON list of ``{"y": ..., "x": [...], "name"?, "constant"?}``)
and list every referenced column in the role ``system`` so that sample
selection and hashing see them; ``ModelSpec.outcome`` is the first equation's
dependent variable.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_SYSTEM = Role("system", many=True, required=True, kind="numeric",
               doc="Every column the system refers to (outcomes, regressors, instruments); the "
                   "convenience functions fill it from the equations.")
_EQUATIONS = Option("equations", "json", None, required=True,
                    doc="List of equations {'y': outcome, 'x': [regressors], 'name': label "
                        "(default: the outcome), 'constant': bool (default: intercept)}.")
_COMMON = (
    Option("small", "bool", False,
           doc="Small-sample statistics: t (degrees of freedom of the first equation) and F "
               "instead of z and chi2."),
    Option("dfk", "bool", False,
           doc="Residual covariance divisor sqrt((N-k_i)(N-k_j)) instead of N."),
    Option("constraints", "json", None,
           doc="Linear constraints [{'terms': {'eq:term': coefficient}, 'value': number}]."),
    Option("tolerance", "float", 1e-9, minimum=1e-15, maximum=1e-2,
           doc="Relative convergence tolerance of the iterated estimators."),
    Option("max_iterations", "int", 1000, minimum=1,
           doc="Iteration limit of the iterated estimators."),
)
_GMM_KERNELS = ("bartlett", "parzen", "quadratic_spectral", "truncated")
_RHOTYPES = ("regress", "freg", "tscorr", "dw", "theil", "nagar")

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="sureg", title="Seemingly unrelated regression", family="systems",
        entry="openecon.econometrics.systems.sureg:fit_sureg", function="sureg",
        stata=("sureg",), covariances=("nonrobust",), default_covariance="nonrobust",
        predictors="none", weights=("aweight", "fweight"), inference="z",
        roles=(_SYSTEM,),
        options=(_EQUATIONS,
                 Option("iterate", "bool", False,
                        doc="Iterate the FGLS estimator to convergence (Gaussian ML; "
                            "Stata's isure)."),
                 *_COMMON),
        description="Zellner's seemingly unrelated regressions by feasible GLS on the stacked "
                    "system: OLS residual covariance, then GLS from one QR factor of the data "
                    "(the Kronecker structure is never expanded); iterated to the ML estimate "
                    "on request. Breusch-Pagan test of independent equations.",
    ),
    EstimatorInfo(
        name="mvreg", title="Multivariate regression", family="systems",
        entry="openecon.econometrics.systems.sureg:fit_mvreg", function="mvreg",
        stata=("mvreg",), covariances=("nonrobust",), default_covariance="nonrobust",
        weights=("aweight", "fweight"), inference="t",
        roles=(Role("outcomes", many=True, required=True, kind="numeric",
                    doc="All dependent variables (the first is ModelSpec.outcome)."),),
        options=(Option("corr", "bool", False,
                        doc="Report the residual correlation matrix and the Breusch-Pagan "
                            "test of independence."),),
        description="OLS of several outcomes on the same regressors with the cross-equation "
                    "residual covariance (N-k divisor), Student t and per-equation F tests.",
    ),
    EstimatorInfo(
        name="reg3", title="Three-stage least squares", family="systems",
        entry="openecon.econometrics.systems.reg3:fit_reg3", function="reg3",
        stata=("reg3",), covariances=("nonrobust",), default_covariance="nonrobust",
        predictors="none", weights=("aweight", "fweight"), inference="z",
        roles=(_SYSTEM,
               Role("endogenous", many=True,
                    doc="Right-hand-side variables that are endogenous although they are no "
                        "equation's outcome (Stata's endog())."),
               Role("exogenous", many=True,
                    doc="Additional exogenous variables used as instruments (exog())."),
               Role("instruments", many=True,
                    doc="The complete list of exogenous variables (inst()); every other "
                        "right-hand-side variable is then endogenous.")),
        options=(_EQUATIONS,
                 Option("method", "str", "3sls", choices=("3sls", "2sls", "sure", "ols"),
                        doc="3sls; 2sls and ols (equation by equation, imply dfk and small); "
                            "sure (all right-hand-side variables exogenous)."),
                 Option("ireg3", "bool", False,
                        doc="Iterate the residual covariance to convergence (3sls, sure)."),
                 *_COMMON),
        description="Three-stage least squares: 2SLS on the projections of every endogenous "
                    "variable on all exogenous variables, residual covariance, system GLS; "
                    "iterated 3SLS, equation-by-equation 2SLS/OLS and SUR variants.",
    ),
    EstimatorInfo(
        name="gmm", title="Generalized method of moments", family="systems",
        entry="openecon.econometrics.systems.gmm:fit_gmm", function="gmm", stata=("gmm",),
        covariances=("robust", "nonrobust", "cluster", "hac"), default_covariance="robust",
        predictors="none", weights=("aweight", "fweight", "pweight"), panel="optional",
        time="optional", inference="z",
        roles=(Role("system", many=True, required=True, kind="numeric",
                    doc="Every column used by the moment expressions and instruments; the "
                        "convenience function fills it. ModelSpec.outcome is the first column "
                        "of the first moment."),),
        options=(
            Option("moments", "json", None, required=True,
                   doc="Residual expressions (nl grammar) as a list, or a dict label -> "
                       "expression."),
            Option("instruments", "json", None,
                   doc="Instrument columns common to all equations, or one list per equation."),
            Option("instrument_constant", "bool", True,
                   doc="Include the constant among every equation's instruments."),
            Option("start", "json", None, doc="Starting values {parameter: value}."),
            Option("wmatrix", "str", None, choices=("robust", "unadjusted", "cluster", "hac"),
                   doc="Weight-matrix type of the two-step/iterated estimator (default: "
                       "robust, or the covariance type when cluster or hac)."),
            Option("winitial", "str", "identity", choices=("identity", "unadjusted"),
                   doc="First-step weight matrix: identity (Stata's default) or "
                       "blockdiag (Z_j'Z_j)^-1 (2SLS for linear moments)."),
            Option("twostep", "bool", True, doc="Two-step estimator; False gives one step."),
            Option("igmm", "bool", False, doc="Iterate the weight matrix to convergence."),
            Option("center", "bool", False, doc="Center the moments in the weight matrix."),
            Option("lags", "int", None, minimum=0, doc="HAC lag length."),
            Option("kernel", "str", "bartlett", choices=_GMM_KERNELS, doc="HAC kernel."),
            Option("tolerance", "float", 1e-8, minimum=1e-15, maximum=1e-2,
                   doc="Gauss-Newton convergence tolerance of each step."),
            Option("max_iterations", "int", 500, minimum=1,
                   doc="Gauss-Newton iteration limit of each step."),
            Option("igmm_tolerance", "float", 1e-10, minimum=1e-15, maximum=1e-2,
                   doc="Relative coefficient change that ends the iterated estimator."),
            Option("igmm_max_iterations", "int", 1000, minimum=1,
                   doc="Reweighting limit of the iterated estimator."),
        ),
        description="General nonlinear GMM over residual expressions with analytic moment "
                    "Jacobians: one-step, two-step and iterated estimators, robust, cluster, "
                    "HAC and unadjusted weight matrices, sandwich covariances and Hansen's J.",
    ),
    EstimatorInfo(
        name="frontier", title="Stochastic frontier model", family="systems",
        entry="openecon.econometrics.systems.frontier:fit_frontier", function="frontier",
        stata=("frontier",), covariances=("nonrobust", "opg", "robust", "cluster"),
        default_covariance="nonrobust", weights=("fweight", "pweight", "iweight"),
        inference="z",
        options=(
            Option("distribution", "str", "hnormal", choices=("hnormal", "exponential", "tnormal"),
                   doc="Inefficiency distribution: half-normal, exponential or truncated "
                       "normal."),
            Option("cost", "bool", False, doc="Cost frontier (y = xb + v + u)."),
        ),
        description="Stochastic production or cost frontier by maximum likelihood with "
                    "analytic derivatives (half-normal, exponential, truncated-normal "
                    "inefficiency), LR test of sigma_u = 0 and JLMS / Battese-Coelli "
                    "efficiency scores (oe.frontier_efficiency).",
    ),
    EstimatorInfo(
        name="xtgls", title="Panel-data feasible GLS", family="systems",
        entry="openecon.econometrics.systems.xtgls:fit_xtgls", function="xtgls",
        stata=("xtgls",), covariances=("nonrobust",), default_covariance="nonrobust",
        panel="required", time="required", inference="z",
        options=(
            Option("panels", "str", "iid", choices=("iid", "heteroskedastic", "correlated"),
                   doc="Error structure across panels."),
            Option("corr", "str", "independent", choices=("independent", "ar1", "psar1"),
                   doc="Within-panel autocorrelation: none, common AR(1), panel-specific "
                       "AR(1)."),
            Option("rhotype", "str", "regress", choices=_RHOTYPES,
                   doc="Estimator of the AR(1) coefficient."),
            Option("igls", "bool", False, doc="Iterate GLS to convergence."),
            Option("tolerance", "float", 1e-7, minimum=1e-15, maximum=1e-2,
                   doc="Relative coefficient change that ends igls."),
            Option("max_iterations", "int", 100, minimum=1, doc="igls iteration limit."),
        ),
        description="Feasible GLS for panels with heteroskedastic or contemporaneously "
                    "correlated errors and common or panel-specific AR(1) disturbances "
                    "(Prais-Winsten), two/three-step or iterated.",
    ),
    EstimatorInfo(
        name="xtpcse", title="Panel-corrected standard errors", family="systems",
        entry="openecon.econometrics.systems.xtpcse:fit_xtpcse", function="xtpcse",
        stata=("xtpcse",), covariances=("robust",), default_covariance="robust",
        panel="required", time="required", inference="z",
        options=(
            Option("correlation", "str", "independent", choices=("independent", "ar1", "psar1"),
                   doc="Within-panel autocorrelation (Prais-Winsten estimates with AR(1))."),
            Option("rhotype", "str", "regress", choices=_RHOTYPES,
                   doc="Estimator of the AR(1) coefficient."),
            Option("hetonly", "bool", False, doc="Heteroskedastic, uncorrelated panels."),
            Option("independent", "bool", False, doc="Independent, homoskedastic panels."),
            Option("pairwise", "bool", False,
                   doc="Sigma from the periods each pair of panels shares (default: casewise)."),
            Option("np1", "bool", False,
                   doc="Weight panel autocorrelations by T_i instead of T_i - 1."),
        ),
        description="OLS or Prais-Winsten estimates with Beck-Katz panel-corrected standard "
                    "errors (contemporaneously correlated, heteroskedastic panels; casewise or "
                    "pairwise Sigma for unbalanced panels).",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "nlsur": "openecon.econometrics.systems.nonlinear_sur:nlsur",
    "nlsur_restore": "openecon.econometrics.systems.nonlinear_sur:nlsur_restore",
    "nlsur_predict": "openecon.econometrics.systems.nonlinear_sur_postestimation:nlsur_predict",
    "nlsur_margins": "openecon.econometrics.systems.nonlinear_sur_postestimation:nlsur_margins",
    "nlsur_contrast": "openecon.econometrics.systems.nonlinear_sur_postestimation:nlsur_contrast",

    "frontier_efficiency": "openecon.econometrics.systems.frontier:frontier_efficiency",
}
