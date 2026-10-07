"""survival estimator family manifest (Torch-free).

Survival-time analysis in the tradition of Stata's ``st`` suite and SPSS's
``KM`` / ``COXREG`` / ``SURVIVAL`` procedures: Kaplan-Meier and Nelson-Aalen
estimates with the log-rank family of tests (``sts``), Cox proportional
hazards regression with ties, strata, delayed entry, time-varying covariates
and proportional-hazards diagnostics (``stcox``, ``stcurve``), parametric
survival regression (``streg``) and actuarial life tables (``ltable``).
Everything is implemented in OpenEconometrics on float64 tensors; see
``docs/econometrics/survival.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_PACKAGE = "openecon.econometrics.survival"
_DISTRIBUTIONS = ("exponential", "weibull", "gompertz", "lognormal", "loglogistic", "ggamma")

_SURVIVAL_ROLES = (
    Role("failure", doc="0/1 failure indicator (default: every record fails)."),
    Role("entry", doc="Entry time (delayed entry; start of (start, stop] records)."),
    Role("id", kind="label", doc="Subject identifier of multiple records per subject; "
                                 "covariance='robust' then clusters on it."),
    Role("strata", many=True, kind="label", doc="Stratification columns."),
    Role("offset", doc="Offset with coefficient 1 in the linear predictor."),
)

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="stcox", title="Cox proportional hazards regression", family="survival",
        entry=f"{_PACKAGE}.cox:fit_stcox", function="stcox", stata=("stcox", "estat phtest",
                                                                   "estat concordance"),
        covariances=("nonrobust", "robust", "cluster"), default_covariance="nonrobust",
        roles=(*_SURVIVAL_ROLES,
               Role("tvc", many=True, doc="Covariates interacted with g(t) (Stata's tvc()).")),
        options=(
            Option("ties", "str", "breslow", choices=("breslow", "efron", "exactp"),
                   doc="Tied failures: Breslow (default), Efron, exact partial likelihood."),
            Option("texp", "str", "identity", choices=("identity", "log"),
                   doc="g(t) of the tvc covariates: t (Stata's texp(_t)) or ln t."),
            Option("phtest", "str", "identity", choices=("identity", "log", "km", "rank"),
                   doc="Time function of the Schoenfeld-residual PH tests."),
            Option("concordance", "bool", True, doc="Compute Harrell's C."),
        ),
        outcome="survival time", intercept="never", weights=("fweight", "iweight", "pweight"),
        inference="z",
        description="Cox regression by Newton-Raphson on the partial likelihood (Breslow, "
                    "Efron or exact ties), strata, delayed entry, multiple records per subject, "
                    "tvc, offset; Lin-Wei robust and cluster covariances; LR, score and "
                    "Grambsch-Therneau PH tests, Harrell's C, Breslow baseline functions.",
    ),
    EstimatorInfo(
        name="streg", title="Parametric survival regression", family="survival",
        entry=f"{_PACKAGE}.streg:fit_streg", function="streg", stata=("streg",),
        covariances=("nonrobust", "opg", "robust", "cluster"), default_covariance="nonrobust",
        roles=(*(role for role in _SURVIVAL_ROLES if role.name != "strata"),
               Role("strata", kind="label", doc="Stratum indicators in the main and the "
                                                "ancillary equation (Stata's strata())."),
               Role("ancillary", many=True, doc="Covariates of the ancillary parameter.")),
        options=(
            Option("distribution", "str", "weibull", choices=_DISTRIBUTIONS,
                   doc="Survival distribution."),
            Option("metric", "str", None, choices=("ph", "aft"),
                   doc="ph (default for exponential, Weibull, Gompertz) or aft (Stata's "
                       "time option; the only metric of lognormal, loglogistic, ggamma)."),
        ),
        outcome="survival time", intercept="always", weights=("fweight", "iweight", "pweight"),
        inference="z", predictors="optional",
        description="Exponential, Weibull, Gompertz, lognormal, loglogistic and generalized "
                    "gamma survival models by maximum likelihood with censoring and delayed "
                    "entry, ancillary equations, strata; OIM, OPG, robust, cluster covariances.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "sts": f"{_PACKAGE}.km:sts",
    "stcurve": f"{_PACKAGE}.curves:stcurve",
    "cox_baseline": f"{_PACKAGE}.saved_prediction:cox_baseline",
    "survival_predict": f"{_PACKAGE}.saved_prediction:survival_predict",
    "ltable": f"{_PACKAGE}.ltable:ltable",
}
