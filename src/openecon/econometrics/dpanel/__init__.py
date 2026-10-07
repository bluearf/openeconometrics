"""dpanel estimator family manifest (Torch-free).

Dynamic panel-data GMM: ``xtdpd`` implements the Arellano-Bond difference GMM
and the Blundell-Bond system GMM estimators with xtabond2's ``gmmstyle()`` /
``ivstyle()`` instrument semantics, first differences or forward orthogonal
deviations, one- and two-step estimation, the Windmeijer (2005) corrected
two-step covariance, Arellano-Bond serial-correlation tests and the Sargan,
Hansen and difference-in-Sargan/Hansen tests. ``xtabond`` and ``xtdpdsys`` are
Stata-style wrappers that build the instrument specification. This module
imports only the registry; the estimators load when a model is fitted. See
``docs/econometrics/dpanel.md``.

``ahreg`` is the separately identified Anderson–Hsiao AR(1) IV estimator,
using one second-lagged level or difference instrument after first differencing.
See ``docs/econometrics/ahreg.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="ahreg", title="Anderson–Hsiao dynamic panel IV", family="dpanel",
        entry="openecon.econometrics.dpanel.ahreg:fit_ahreg", function="ahreg",
        stata=("ivregress 2sls on first differences with an AH lag instrument",),
        covariances=("cluster", "robust", "nonrobust"), default_covariance="cluster",
        predictors="optional", categorical=False, intercept="never",
        panel="required", time="required", inference="t",
        options=(Option("instrument", "str", "levels", choices=("levels", "differences"),
                        doc="One excluded instrument for the differenced lagged outcome: "
                            "y(t-2) or y(t-2)-y(t-3). Predictors are strictly exogenous."),),
        description="Anderson–Hsiao AR(1) dynamic panel 2SLS. First differences remove "
                    "panel fixed effects; second-lagged levels or differences instrument "
                    "the differenced lagged outcome. Default cluster and robust alias "
                    "use the model panel CR1 sandwich. Nonrobust accounts for the MA(1) "
                    "differenced-error covariance under homoskedastic level errors. "
                    "Consecutive integer periods only; unbalanced panels and gaps allowed.",
    ),
    EstimatorInfo(
        name="xtdpd", title="Dynamic panel-data GMM", family="dpanel",
        entry="openecon.econometrics.dpanel.xtdpd:fit_xtdpd", function="xtdpd",
        stata=("xtabond2", "xtdpd", "xtabond", "xtdpdsys", "estat abond", "estat sargan"),
        covariances=("nonrobust", "robust"), default_covariance="nonrobust",
        predictors="optional", categorical=False, panel="required", time="required",
        inference="z",
        options=(
            Option("lags", "int", 1, minimum=0, maximum=50,
                   doc="Number of lags of the dependent variable included as regressors "
                       "(terms L1.y ... Lp.y)."),
            Option("gmm", "json", None,
                   doc="GMM-style instrument groups (xtabond2 gmmstyle): a list of "
                       "{'columns': [...], 'lags': [lo, hi or None], 'equation': "
                       "'diff'|'level'|'both', 'collapse': bool}. 'both' adds the difference "
                       "at lag lo-1 to the level equation; 'level' uses the differences at "
                       "lags lo..hi. Default: the dependent variable with lags [2, None] when "
                       "lags >= 1."),
            Option("iv", "json", None,
                   doc="IV-style instrument groups (xtabond2 ivstyle): a list of "
                       "{'columns': [...], 'equation': 'diff'|'level'|'both', 'passthru': "
                       "bool}. Default: the regressors x that no gmm group instruments, as "
                       "strictly exogenous instruments of the transformed equation."),
            Option("system", "bool", False,
                   doc="Blundell-Bond system GMM: add the level equation instrumented by "
                       "lagged differences."),
            Option("twostep", "bool", False,
                   doc="Two-step efficient GMM (weight matrix from one-step residuals)."),
            Option("orthogonal", "bool", False,
                   doc="Forward orthogonal deviations (Arellano-Bover) instead of first "
                       "differences."),
            Option("collapse", "bool", False,
                   doc="Collapse GMM-style instruments to one column per lag (default for "
                       "groups that do not set 'collapse' themselves)."),
            Option("small", "bool", False,
                   doc="Small-sample statistics: t and F with finite-sample factors "
                       "instead of z and chi2."),
            Option("time_dummies", "bool", False,
                   doc="Add period dummies as regressors and IV-style instruments."),
            Option("h", "int", 3, choices=(1, 2, 3),
                   doc="Form of H in the one-step weight matrix (Z'HZ)^-1 (xtabond2 h()): 1 "
                       "identity; 2 transformed block from the transformation, identity for "
                       "levels; 3 (default) the exact covariance of the stacked errors."),
            Option("artests", "int", 2, minimum=1, maximum=20,
                   doc="Highest order of the Arellano-Bond serial-correlation tests."),
        ),
        description="Arellano-Bond difference and Blundell-Bond system GMM for dynamic panels "
                    "(xtabond2 semantics): GMM-style and IV-style instruments, first "
                    "differences or forward orthogonal deviations, one-step and two-step "
                    "estimation with the Windmeijer correction, Arellano-Bond AR tests, "
                    "Sargan, Hansen and difference-in-Sargan/Hansen tests, with xtabond2's "
                    "conventions for sigma2, small-sample statistics and the tests. robust is "
                    "the sandwich clustered on the panel variable.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "xtabond": "openecon.econometrics.dpanel.xtdpd:xtabond",
    "xtdpdsys": "openecon.econometrics.dpanel.xtdpd:xtdpdsys",
}
