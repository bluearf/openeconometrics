"""panel estimator family manifest (Torch-free).

Linear panel-data estimators: ``xtreg`` (fixed effects, random effects,
between, first differences, random-effects maximum likelihood, pooled OLS with
panel-aware covariances including Driscoll-Kraay), ``xtfmb`` (Fama-MacBeth)
and the ``hausman`` specification test. Every estimator is implemented in
OpenEconometrics on float64 tensors; see ``docs/econometrics/panel.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_KERNELS = ("bartlett", "truncated", "parzen", "quadratic_spectral")

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="xtreg", title="Panel-data linear regression", family="panel",
        entry="openecon.econometrics.panel.xtreg:fit_xtreg", function="xtreg",
        stata=("xtreg", "xtscc", "xttest0"),
        covariances=("nonrobust", "robust", "cluster", "driscoll_kraay"),
        default_covariance="nonrobust",
        panel="required", time="optional", weights=("aweight", "fweight", "pweight"),
        cluster_dimensions=2, inference="t", intercept="always",
        options=(
            Option("model", "str", "fe", choices=("fe", "re", "be", "fd", "mle", "pooled", "cre"),
                   doc="fe: within estimator; re: Swamy-Arora GLS; be: between; fd: first "
                       "differences; mle: Gaussian random-effects ML; pooled: pooled OLS; cre: numeric Mundlak means and joint test."),
            Option("lags", "int", None, minimum=0,
                   doc="Driscoll-Kraay bandwidth (lags); default floor(4 (T/100)^(2/9))."),
            Option("kernel", "str", "bartlett", choices=_KERNELS,
                   doc="Driscoll-Kraay kernel."),
            Option("sa", "bool", False,
                   doc="model='re': Baltagi-Chang (Swamy-Arora) unbalanced small-sample "
                       "variance components instead of the harmonic-mean approximation."),
            Option("wls", "bool", False,
                   doc="model='be': weight the between regression by the panel lengths T_i."),
        ),
        description="Stata's xtreg: fixed, random (GLS and ML), between, first-difference and "
                    "pooled panel regressions. robust means clustering on the panel variable; "
                    "cluster columns must nest the panels for fe/re; driscoll_kraay (xtscc) is "
                    "available for fe and pooled. Weights apply to fe (constant within panel), "
                    "fd and pooled only; re, be and mle take none, as in Stata.",
    ),
    EstimatorInfo(
        name="xtfmb", title="Fama-MacBeth regression", family="panel",
        entry="openecon.econometrics.panel.fmb:fit_xtfmb", function="xtfmb",
        stata=("xtfmb",), covariances=("nonrobust", "hac"), default_covariance="nonrobust",
        panel="required", time="required", inference="t", intercept="always",
        options=(
            Option("lags", "int", None, minimum=0,
                   doc="Newey-West lags over the period estimates (covariance='hac')."),
        ),
        description="Period-by-period cross-sectional OLS; coefficients are the means of the "
                    "period estimates and standard errors their standard deviation over "
                    "sqrt(T), optionally Newey-West adjusted.",
    ),
)

# Each model's distinct augmentation and inference is recorded in the result.
ESTIMATORS = ESTIMATORS + (
    EstimatorInfo(
        name="htaylor_moment", title="Hausman–Taylor panel IV regression", family="panel",
        entry="openecon.econometrics.panel.htaylor:fit_htaylor", function="htaylor_moment",
        stata=(), covariances=("nonrobust", "robust", "cluster"),
        default_covariance="nonrobust", predictors="optional", categorical=False,
        panel="required", time="optional", intercept="always", cluster_dimensions=1,
        inference="z", roles=(
            Role("varying_exogenous", many=True, kind="numeric"),
            Role("varying_endogenous", many=True, kind="numeric"),
            Role("invariant_exogenous", many=True, kind="numeric"),
            Role("invariant_endogenous", many=True, kind="numeric")),
        options=(Option("small", "bool", False),
                 Option("max_work", "int", 1000000000, minimum=1)),
        description="Declared X1/X2/Z1/Z2 HT instrument sets; endogenous means correlation with panel effects, not idiosyncratic errors. Balanced/unbalanced numeric unweighted CPU float64. Rank/role validation, full covariance and saved population prediction; no Amemiya–MaCurdy variant.",
    ),
    EstimatorInfo(
        name="xtcce", title="Common-factor panel regression", family="panel",
        entry="openecon.econometrics.panel.cce:fit_xtcce", function="xtcce",
        covariances=("nonrobust",), default_covariance="nonrobust",
        panel="required", time="required", categorical=False, intercept="always",
        options=(
            Option("model", "str", "ccemg", choices=("ccemg", "ccep", "amg", "dcce", "csardl", "csdl")),
            Option("y_lags", "int", None, minimum=0, maximum=32),
            Option("x_lags", "int", None, minimum=0, maximum=32),
            Option("cs_lags", "int", None, minimum=0, maximum=32),
            Option("trend", "bool", False),
            Option("max_work", "int", 100_000_000, minimum=1),
        ),
        description="Static observed-row CCEMG/CCEP/AMG and balanced dynamic CCE/CS-ARDL/CS-DL; native QR with explicit lag, calendar, rank and inference contracts.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
ESTIMATORS = ESTIMATORS + (
    EstimatorInfo(
        name="cre", title="Mundlak correlated random effects", family="panel",
        entry="openecon.econometrics.panel.cre:fit_cre", function="cre",
        panel="required", time="optional", categorical=False, intercept="always",
        covariances=("nonrobust", "robust", "cluster"), default_covariance="robust", cluster_dimensions=1,
        options=(Option("means", "list[str]", None), Option("max_work", "int", 100_000_000, minimum=1)),
        description="Mundlak random-effects GLS with means calculated from retained estimation rows and full joint mean-coefficient covariance/test; numeric unweighted resident panels.",
    ),
    EstimatorInfo(
        name="xthtaylor", title="Hausman--Taylor panel IV", family="panel",
        entry="openecon.econometrics.panel.cre:fit_xthtaylor", function="xthtaylor",
        panel="required", time="optional", categorical=False, predictors="none", intercept="always",
        covariances=("nonrobust", "robust"), default_covariance="nonrobust",
        roles=(Role("x1", many=True, required=True), Role("x2", many=True), Role("z1", many=True), Role("z2", many=True, required=True)),
        options=(Option("max_work", "int", 100_000_000, minimum=1),),
        description="Internally instrumented quasi-demeaned panel IV with explicit varying/invariant exogeneity partitions and variance-component/CR1 domains; numeric unweighted resident panels.",
    ),
)


EXPORTS: dict[str, str] = {
    "panel_structural_predict": "openecon.econometrics.panel.cre:panel_structural_predict",
    "mundlak_test": "openecon.econometrics.panel.mundlak:mundlak_test",
    "cre_predict": "openecon.econometrics.panel.mundlak:cre_predict",
    "htaylor_predict": "openecon.econometrics.panel.htaylor:htaylor_predict",
    "hausman": "openecon.econometrics.panel.hausman:hausman",
    "xtserial": "openecon.econometrics.panel.diagnostics:xtserial",
    "xttest3": "openecon.econometrics.panel.diagnostics:xttest3",
    "xtcd": "openecon.econometrics.panel.dependence:xtcd",
    "xthst": "openecon.econometrics.panel.homogeneity:xthst",
}
