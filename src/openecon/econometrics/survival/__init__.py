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
    EstimatorInfo(
        name="stcox_frailty", title="Cox regression with shared frailty", family="survival",
        entry=f"{_PACKAGE}.frailty:fit_stcox_frailty", function="stcox_frailty",
        stata=("stcox, shared()",),
        covariances=("nonrobust",), default_covariance="nonrobust",
        roles=(
            Role("failure", doc="0/1 failure indicator (default: every record fails)."),
            Role("entry", doc="Start of (start, stop] records; the frailty multiplies the "
                              "whole history of its group."),
            Role("shared", required=True, kind="label",
                 doc="Group column of the shared frailty (Stata's shared())."),
            Role("strata", many=True, kind="label", doc="Stratification columns."),
            Role("offset", doc="Offset with coefficient 1 in the linear predictor."),
        ),
        options=(
            Option("frailty", "str", "gamma", choices=("gamma", "lognormal"),
                   doc="Frailty distribution: gamma (closed-form integrated likelihood) or "
                       "log-normal (Laplace approximation)."),
            Option("ties", "str", "breslow", choices=("breslow", "efron"),
                   doc="Tied failures: Breslow (default) or Efron."),
            Option("theta", "float", None, minimum=0.0,
                   doc="Fixed frailty variance (default: estimated)."),
        ),
        outcome="survival time", intercept="never", weights=(), inference="z",
        description="Shared gamma or log-normal frailty Cox model: penalized partial "
                    "likelihood for fixed variance, variance by the integrated partial "
                    "likelihood; LR chibar2 test of theta = 0; conditional penalized-Hessian "
                    "covariance.",
    ),
    EstimatorInfo(
        name="streg_frailty", title="Parametric survival regression with shared frailty",
        family="survival", entry=f"{_PACKAGE}.parametric_frailty:fit_streg_frailty",
        function="streg_frailty", stata=("streg, shared() frailty()",),
        covariances=("nonrobust",), default_covariance="nonrobust",
        roles=(
            Role("failure", doc="0/1 failure indicator (default: every record fails)."),
            Role("shared", required=True, kind="label",
                 doc="Group column of the shared frailty (Stata's shared())."),
            Role("offset", doc="Offset with coefficient 1 in the linear predictor."),
        ),
        options=(
            Option("distribution", "str", "weibull", choices=("exponential", "weibull"),
                   doc="Survival distribution."),
            Option("metric", "str", None, choices=("ph", "aft"),
                   doc="ph (default) or aft (Stata's time option)."),
            Option("frailty", "str", "gamma", choices=("gamma", "invgaussian"),
                   doc="Frailty distribution (mean 1, variance theta)."),
        ),
        outcome="survival time", intercept="always", weights=(), inference="z",
        predictors="optional",
        description="Exponential or Weibull PH/AFT regression with a shared gamma or "
                    "inverse-Gaussian frailty: closed-form marginal likelihood, analytic "
                    "Newton-Raphson, OIM covariance, LR chibar2 test of theta = 0.",
    ),
    EstimatorInfo(
        name="stcox_recurrent", title="Recurrent-event Cox regression", family="survival",
        entry=f"{_PACKAGE}.recurrent:fit_stcox_recurrent", function="stcox_recurrent",
        stata=("stcox, vce(cluster)", "stcox, strata() vce(cluster)"),
        covariances=("robust", "nonrobust"), default_covariance="robust",
        roles=(
            Role("failure", required=True, doc="0/1 event indicator at the record's exit."),
            Role("entry", doc="Start of the (start, stop] record (0 when omitted)."),
            Role("id", required=True, kind="label",
                 doc="Subject identifier; robust covariance clusters on it."),
        ),
        options=(
            Option("model", "str", "ag", choices=("ag", "pwp_total", "pwp_gap", "wlw"),
                   doc="Andersen-Gill, Prentice-Williams-Peterson total or gap time, or "
                       "Wei-Lin-Weissfeld marginal model."),
            Option("max_events", "int", None, minimum=1,
                   doc="Cap on event-number strata (PWP pools above it; WLW models the "
                       "first max_events events)."),
            Option("ties", "str", "breslow", choices=("breslow", "efron"),
                   doc="Tied failures: Breslow (default) or Efron."),
        ),
        outcome="survival time", intercept="never", weights=(), inference="z",
        description="Recurrent-event Cox models on counting-process records: Andersen-Gill, "
                    "PWP total/gap time stratified by event number, WLW marginal model; "
                    "Lin-Wei robust covariance clustered on the subject.",
    ),
    EstimatorInfo(
        name="stteffects", title="Survival treatment effects", family="survival",
        entry=f"{_PACKAGE}.stteffects:fit_stteffects", function="stteffects",
        stata=("stteffects ra", "stteffects ipw", "stteffects ipwra"),
        covariances=("robust", "cluster"), default_covariance="robust",
        roles=(
            Role("failure", required=True, doc="0/1 failure indicator."),
            Role("treatment", required=True, kind="label", doc="Binary treatment variable."),
            Role("tx", many=True, doc="Treatment-model covariates (default: the outcome "
                                      "covariates)."),
            Role("cx", many=True, doc="ipw: censoring-model covariates (default: the outcome "
                                      "covariates)."),
        ),
        options=(
            Option("method", "str", "ra", choices=("ra", "ipw", "ipwra"),
                   doc="Weibull regression adjustment, censoring-adjusted IPW, or IPW "
                       "regression adjustment."),
            Option("estimand", "str", "ate", choices=("ate", "atet", "pomeans"),
                   doc="Average treatment effect, ATE on the treated, or potential-outcome "
                       "means of mean survival time."),
            Option("control", "json", None, doc="Control level (default: the lowest level)."),
            Option("pstolerance", "float", 1e-5, minimum=0.0, maximum=0.5,
                   doc="Overlap check: smallest admissible estimated treatment probability."),
        ),
        outcome="survival time", predictors="optional", weights=(), inference="z",
        description="Treatment effects on mean survival time with Weibull outcome or "
                    "censoring models and a logit treatment model; stacked estimating "
                    "equations with analytic Jacobian and M-estimation sandwich covariance.",
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
