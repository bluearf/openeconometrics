"""iv estimator family manifest (Torch-free).

Instrumental-variables estimators of a linear model with endogenous
regressors: ``ivregress`` (2SLS, LIML and GMM with first-stage,
overidentification, endogeneity and weak-identification diagnostics),
``xtivreg`` (fixed-effects, first-difference, random-effects G2SLS/EC2SLS and
between panel IV) and ``ivreghdfe`` (IV with absorbed high-dimensional fixed
effects). This module imports only the registry so that spec validation and
``oe.capabilities()`` never load the tensor runtime; the estimators live in the
sibling modules and load when a model is fitted. See ``docs/econometrics/iv.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

HAC_KERNELS = ("bartlett", "parzen", "quadratic_spectral", "truncated")

_ENDOGENOUS = Role("endogenous", many=True, required=True, kind="numeric",
                   doc="Endogenous regressors (instrumented).")
_INSTRUMENTS = Role("instruments", many=True, required=True, kind="numeric",
                    doc="Excluded instruments. The exogenous regressors (predictors) are "
                        "instruments for themselves and must not be repeated here.")

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="ivregress", title="Instrumental-variables regression", family="iv",
        entry="openecon.econometrics.iv.ivregress:fit_ivregress", function="ivregress",
        stata=("ivregress 2sls", "ivregress liml", "ivregress gmm", "estat firststage",
               "estat overid", "estat endogenous", "ivreg2"),
        covariances=("nonrobust", "robust", "cluster", "hac"), default_covariance="nonrobust",
        predictors="optional", weights=("aweight", "fweight", "pweight"),
        panel="optional", time="optional", cluster_dimensions=2, inference="z",
        roles=(_ENDOGENOUS, _INSTRUMENTS),
        options=(
            Option("method", "str", "2sls", choices=("2sls", "liml", "gmm", "fuller", "kclass"),
                   doc="2sls: two-stage least squares; liml: limited-information maximum "
                       "likelihood; gmm: two-step (or iterated) efficient GMM; fuller: "
                       "adjusted LIML; kclass: explicit kappa in [0,1]."),
            Option("kappa", "float", None, doc="Explicit k-class parameter in [0,1]; method='kclass' only."),
            Option("fuller_alpha", "float", None, doc="Fuller adjustment, positive, default 1; method='fuller' only."),
            Option("small", "bool", False,
                   doc="Small-sample statistics: t and F with N-K degrees of freedom and the "
                       "regress-style factors; default z and chi2 with RSS/N."),
            Option("wmatrix", "str", None, choices=("robust", "cluster", "hac", "unadjusted"),
                   doc="GMM weight matrix. Default: robust, or the covariance type when that "
                       "is cluster or hac."),
            Option("igmm", "bool", False, doc="Iterate the GMM estimator to convergence."),
            Option("center", "bool", False,
                   doc="Center the moments when forming the GMM weight matrix."),
            Option("lags", "int", None, minimum=0,
                   doc="HAC lag length (bandwidth - 1); required with a hac covariance or "
                       "weight matrix."),
            Option("kernel", "str", "bartlett", choices=HAC_KERNELS, doc="HAC kernel."),
        ),
        description="Stata's ivregress: 2SLS, LIML and efficient GMM by QR projections, with "
                    "first-stage F and partial R-squared, Sargan/Basmann/Hansen/Anderson-Rubin "
                    "overidentification tests, Durbin-Wu-Hausman endogeneity tests and the "
                    "Cragg-Donald and Kleibergen-Paap weak-identification statistics.",
    ),
    EstimatorInfo(
        name="xtivreg", title="Panel-data instrumental-variables regression", family="iv",
        entry="openecon.econometrics.iv.xtivreg:fit_xtivreg", function="xtivreg",
        stata=("xtivreg",),
        covariances=("nonrobust", "robust", "cluster"), default_covariance="nonrobust",
        predictors="optional", intercept="always", panel="required", time="optional",
        cluster_dimensions=1, inference="z",
        roles=(_ENDOGENOUS, _INSTRUMENTS),
        options=(
            Option("model", "str", "fe", choices=("fe", "fd", "re", "be"),
                   doc="fe: within 2SLS; fd: first-difference 2SLS; re: G2SLS random effects "
                       "(EC2SLS with ec2sls); be: 2SLS on panel means."),
            Option("ec2sls", "bool", False,
                   doc="model='re': Baltagi's EC2SLS instead of G2SLS."),
            Option("small", "bool", False,
                   doc="Report t and F statistics instead of z and chi2 (Stata's small). The "
                       "covariance itself always carries the model's degrees of freedom."),
        ),
        description="Stata's xtivreg: fixed-effects, first-difference, random-effects "
                    "(Balestra-Varadharajan-Krishnakumar G2SLS or Baltagi EC2SLS) and between "
                    "two-stage least squares. robust means clustering on the panel variable.",
    ),
    EstimatorInfo(
        name="ivreghdfe", title="IV regression with high-dimensional fixed effects",
        family="iv", entry="openecon.econometrics.iv.ivreghdfe:fit_ivreghdfe",
        function="ivreghdfe", stata=("ivreghdfe", "ivreg2", "reghdfe"),
        covariances=("nonrobust", "robust", "cluster"), default_covariance="nonrobust",
        predictors="optional", intercept="never", weights=("aweight", "fweight", "pweight"),
        cluster_dimensions=2, inference="t",
        roles=(_ENDOGENOUS, _INSTRUMENTS,
               Role("absorb", many=True, required=True, kind="label",
                    doc="One or more categorical variables absorbed as fixed effects.")),
        options=(
            Option("method", "str", "2sls", choices=("2sls", "liml", "gmm"),
                   doc="2sls, liml, or two-step efficient GMM with the weight matrix of the "
                       "covariance type."),
            Option("small", "bool", True,
                   doc="t and F statistics with N-K degrees of freedom, K counting the absorbed "
                       "effects (ivreghdfe's default); False gives z and chi2."),
            Option("drop_singletons", "bool", True,
                   doc="Iteratively drop observations that are alone in a level (reghdfe)."),
            Option("tolerance", "float", 1e-8,
                   doc="Relative convergence tolerance of the alternating-projection demeaning."),
            Option("max_iterations", "int", 10000, minimum=1,
                   doc="Maximum number of demeaning sweeps."),
        ),
        description="ivreghdfe (ivreg2 + reghdfe): the absorbed fixed effects are partialled "
                    "out of the outcome, regressors and instruments; 2SLS, LIML or GMM then run "
                    "on the residualized data with reghdfe's degrees-of-freedom accounting.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "stock_yogo": "openecon.econometrics.iv.weak_inference:stock_yogo",
    "iv_weak_test": "openecon.econometrics.iv.weak_inference:iv_weak_test",
    "iv_ar_confidence_set": "openecon.econometrics.iv.weak_inference:iv_ar_confidence_set",
}
