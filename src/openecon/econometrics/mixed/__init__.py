"""mixed estimator family manifest (Torch-free).

Mixed-effects and population-averaged models for grouped data: ``mixed``
(linear mixed models by ML/REML, Stata's mixed / SPSS MIXED), random-intercept
GLMMs by adaptive Gauss-Hermite quadrature (``melogit``, ``meprobit``,
``mepoisson``), Stata's ``xtlogit`` / ``xtprobit`` / ``xtpoisson`` with
random, fixed and population-averaged effects, and generalized estimating
equations (``xtgee``; SPSS GEE). Everything is implemented in OpenEconometrics on
float64 tensors; see ``docs/econometrics/mixed.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_GROUP = Role("group", many=True, required=True, kind="label",
              doc="One grouping column, or two nested ones with the top level first.")

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="mixed", title="Linear mixed-effects regression", family="mixed",
        entry="openecon.econometrics.mixed.lmm:fit_mixed", function="mixed",
        stata=("mixed",), covariances=("nonrobust", "robust", "cluster"),
        default_covariance="nonrobust", predictors="optional", weights=("fweight",),
        inference="z",
        roles=(
            _GROUP,
            Role("random", many=True, doc="Columns with random slopes at the lowest level."),
        ),
        options=(
            Option("random_intercept", "bool", True,
                   doc="Random intercept at the lowest level."),
            Option("covstructure", "str", "independent",
                   choices=("independent", "unstructured", "exchangeable", "identity"),
                   doc="Covariance structure of the lowest-level random effects."),
            Option("method", "str", "ml", choices=("ml", "reml"),
                   doc="Maximum likelihood (Stata's default) or restricted ML."),
        ),
        description="Linear mixed model y = Xb + Zu + e with one or two nested grouping levels "
                    "(random intercept at the top level; random intercept and slopes with an "
                    "independent, unstructured, exchangeable or identity covariance at the "
                    "lowest), by ML or REML on group sufficient statistics. robust clusters on "
                    "the top-level groups (ML only).",
    ),
)

_ME_OPTIONS = (
    Option("intpoints", "int", 7, minimum=1, maximum=64,
           doc="Gauss-Hermite quadrature points per random-effect dimension."),
    Option("intmethod", "str", "mvaghermite", choices=("mvaghermite", "mcaghermite", "ghermite"),
           doc="Mean-variance adaptive (Stata's default), mode-curvature adaptive or "
               "non-adaptive Gauss-Hermite quadrature."),
)
_ME_ROLES = (
    Role("group", required=True, kind="label", doc="Grouping column of the random effects."),
    Role("random", many=True, doc="At most one column with an independent random slope."),
)
_OFFSET = (
    Role("offset", doc="Column added to the linear predictor with coefficient 1."),
    Role("exposure", doc="Positive column whose log is added to the linear predictor."),
)
_ME = ("nonrobust", "robust", "cluster")


def _me(name: str, title: str, outcome: str, link: str, roles: tuple[Role, ...]) -> EstimatorInfo:
    return EstimatorInfo(
        name=name, title=title, family="mixed",
        entry=f"openecon.econometrics.mixed.glmm:fit_{name}", function=name, stata=(name,),
        covariances=_ME, default_covariance="nonrobust", outcome=outcome, inference="z",
        roles=roles, options=_ME_OPTIONS,
        description=f"Random-intercept (optionally one independent random slope) {link} model "
                    "by maximum likelihood with mean-variance adaptive Gauss-Hermite quadrature "
                    "and analytic derivatives; robust clusters on the groups.",
    )


ESTIMATORS = ESTIMATORS + (
    _me("melogit", "Mixed-effects logistic regression", "binary", "logit", _ME_ROLES),
    _me("meprobit", "Mixed-effects probit regression", "binary", "probit", _ME_ROLES),
    _me("mepoisson", "Mixed-effects Poisson regression", "count", "Poisson",
        _ME_ROLES + _OFFSET),
)

ESTIMATORS = ESTIMATORS + (
    EstimatorInfo(
        name="xtgee", title="GEE population-averaged panel model", family="mixed",
        entry="openecon.econometrics.mixed.gee:fit_xtgee", function="xtgee", stata=("xtgee",),
        covariances=("nonrobust", "robust"), default_covariance="nonrobust", panel="required",
        time="optional", inference="z", roles=_OFFSET,
        options=(
            Option("family", "str", "gaussian",
                   choices=("gaussian", "binomial", "poisson", "gamma", "nbinomial",
                            "igaussian"), doc="Distribution family (variance function)."),
            Option("link", "str", None,
                   choices=("identity", "log", "logit", "probit", "cloglog", "loglog",
                            "reciprocal", "inverse_squared"),
                   doc="Link function; the family's canonical link by default."),
            Option("corr", "str", "exchangeable",
                   choices=("exchangeable", "independent", "ar1", "stationary",
                            "nonstationary", "unstructured"),
                   doc="Working correlation structure."),
            Option("corr_order", "int", 1, minimum=1,
                   doc="Order m of corr='stationary' / 'nonstationary'."),
            Option("scale", "json", None,
                   doc="None (family default), 'x2', 'dev' or a positive number."),
            Option("nmp", "bool", False, doc="Divide by N - p instead of N in phi."),
            Option("force", "bool", False,
                   doc="Treat unequally spaced observations as consecutive."),
            Option("nbk", "float", 1.0, minimum=1e-8,
                   doc="k of family nbinomial: Var = mu + k mu^2."),
        ),
        description="Liang-Zeger generalized estimating equations with exchangeable, AR(1), "
                    "stationary, nonstationary, unstructured or independent working "
                    "correlation; conventional or semi-robust (panel-clustered) covariance.",
    ),
)

_XT_OPTIONS = (
    Option("model", "str", "re", choices=("re", "fe", "pa"),
           doc="re: random effects; fe: conditional fixed effects; pa: population-averaged "
               "GEE."),
    Option("intpoints", "int", 12, minimum=1, maximum=64,
           doc="Quadrature points of the normal random effect (model='re')."),
    Option("intmethod", "str", "mvaghermite", choices=("mvaghermite", "mcaghermite", "ghermite"),
           doc="Quadrature rule of the normal random effect (model='re')."),
    Option("corr", "str", "exchangeable",
           choices=("exchangeable", "independent", "ar1", "stationary", "nonstationary",
                    "unstructured"),
           doc="Working correlation of model='pa'."),
    Option("corr_order", "int", 1, minimum=1, doc="Order of a stationary/nonstationary corr."),
    Option("force", "bool", False, doc="model='pa': treat gaps in time as consecutive."),
)


def _xt(name: str, title: str, outcome: str, roles: tuple[Role, ...],
        options: tuple[Option, ...] = ()) -> EstimatorInfo:
    return EstimatorInfo(
        name=name, title=title, family="mixed",
        entry=f"openecon.econometrics.mixed.xt:fit_{name}", function=name,
        stata=(name, f"{name}, re", f"{name}, fe", f"{name}, pa"),
        covariances=("nonrobust", "robust", "cluster"), default_covariance="nonrobust",
        outcome=outcome, panel="required", time="optional", inference="z", roles=roles,
        options=_XT_OPTIONS + options,
        description=f"Stata's {name}: random effects (model='re'), conditional fixed effects "
                    "(model='fe') or population-averaged GEE (model='pa'); robust clusters on "
                    "the panels.",
    )


ESTIMATORS = ESTIMATORS + (
    _xt("xtlogit", "Panel-data logistic regression", "binary", _OFFSET[:1]),
    _xt("xtprobit", "Panel-data probit regression", "binary", _OFFSET[:1]),
    _xt("xtpoisson", "Panel-data Poisson regression", "count", _OFFSET,
        (Option("normal", "bool", False,
                doc="model='re': normal instead of gamma random effect."),)),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "repeated_gls": "openecon.econometrics.mixed.repeated:repeated_gls",
    "restore_repeated_gls": "openecon.econometrics.mixed.repeated:restore_repeated_gls",
    "repeated_gls_predict": "openecon.econometrics.mixed.repeated:repeated_gls_predict",
    "repeated_gls_contrast": "openecon.econometrics.mixed.repeated:repeated_gls_contrast",
    "mixed_satterthwaite": "openecon.econometrics.mixed.contrasts:mixed_satterthwaite",
    "mixed_predict": "openecon.econometrics.mixed.predict:mixed_predict",
}

_EXT_WORK = Option('max_work', 'int', 2_000_000_000, minimum=1,
                   doc='Explicit likelihood and derivative work budget; no silent truncation.')
ESTIMATORS = ESTIMATORS + (
    EstimatorInfo(name='mixedflex', title='Crossed and multilevel Gaussian mixed model', family='mixed',
        entry='openecon.econometrics.mixed.flexible_lmm:fit_mixedflex', function='mixedflex',
        categorical=False, predictors='optional', covariances=('nonrobust',), default_covariance='nonrobust',
        roles=(Role('group', many=True, required=True, kind='label'), Role('random', many=True)),
        options=(Option('grouping', 'str', 'crossed', choices=('crossed', 'nested')),
                 Option('covstructure', 'str', 'independent', choices=('independent', 'unstructured')), _EXT_WORK),
        description='Bounded resident dense Gaussian joint ML with up to eight crossed or nested intercept factors and a lowest-level random slope; full fixed/variance information.'),
    EstimatorInfo(name='xtnbreg', title='HHG panel negative binomial regression', family='mixed',
        entry='openecon.econometrics.mixed.panel_likelihoods:fit_xtnbreg', function='xtnbreg',
        stata=('xtnbreg, re', 'xtnbreg, fe'), outcome='count', predictors='optional',
        categorical=False, panel='required', time='optional', covariances=('nonrobust',),
        default_covariance='nonrobust', roles=_OFFSET,
        options=(Option('model', 'str', 're', choices=('re', 'fe')), _EXT_WORK),
        description='HHG random beta-dispersion or conditional fixed-dispersion likelihood; full observed information.'),
    EstimatorInfo(name='xtfrontier', title='Panel stochastic frontier', family='mixed',
        entry='openecon.econometrics.mixed.panel_likelihoods:fit_xtfrontier', function='xtfrontier',
        stata=('xtfrontier, ti', 'xtfrontier, tvd'), categorical=False,
        panel='required', time='optional', covariances=('nonrobust',), default_covariance='nonrobust',
        options=(Option('distribution', 'str', 'truncated_normal', choices=('truncated_normal', 'half_normal')),
                 Option('time_varying', 'bool', False), Option('cost', 'bool', False), _EXT_WORK),
        description='Joint panel likelihood with shared truncated/half-normal inefficiency, optionally exponential calendar decay; full information and posterior efficiency delta uncertainty.'),
)

_EXT_INT = (Option('covstructure', 'str', 'independent', choices=('independent', 'unstructured')),
            Option('intpoints', 'int', 16, minimum=4, maximum=64),
            Option('max_intpoints', 'int', 64, minimum=8, maximum=128), _EXT_WORK)
ESTIMATORS = ESTIMATORS + (
    EstimatorInfo(name='mixedlogit', title='Random-parameter panel choice logit', family='mixed',
        entry='openecon.econometrics.mixed.integrated:fit_mixedlogit', function='mixedlogit',
        categorical=False, intercept='never', outcome='binary', covariances=('nonrobust',), default_covariance='nonrobust',
        roles=(Role('group', required=True, kind='label'), Role('case', required=True, kind='label'),
               Role('alternative', required=True, kind='label'), Role('random', many=True, required=True),
               Role('availability')), options=_EXT_INT,
        description='Repeated-choice normal random coefficients; complete explicit availability sets, deterministic checked product quadrature and full information.'),
    EstimatorInfo(name='menbreg', title='Mixed-effects NB2 regression', family='mixed',
        entry='openecon.econometrics.mixed.integrated:fit_menbreg', function='menbreg',
        categorical=False, predictors='optional', outcome='count', covariances=('nonrobust',), default_covariance='nonrobust',
        roles=_ME_ROLES + _OFFSET, options=_EXT_INT, stata=('menbreg, dispersion(mean)',),
        description='Normal random intercept and optional slope with conditional NB2 dispersion; checked deterministic quadrature, joint information and population mean uncertainty.'),
)
