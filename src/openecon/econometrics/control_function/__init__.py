"""Bounded continuous-first-stage control-function mean estimators (Torch-free)."""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

KINDS = {
    "cfregress": ("gaussian", "continuous", "Gaussian identity"),
    "cflogit": ("logit", "binary", "Binary logit"),
    "cfprobit": ("probit", "binary", "Binary probit"),
    "cfcloglog": ("cloglog", "binary", "Binary complementary log-log"),
    "cfpoisson": ("poisson", "count", "Poisson log mean"),
    "cfgamma": ("gamma", "positive", "Gamma log mean"),
    "cfinvgauss": ("inverse_gaussian", "positive", "Inverse-Gaussian log mean"),
    "cffraclogit": ("fractional_logit", "fractional", "Fractional logit mean"),
}

_ROLES = (
    Role("endogenous", required=True, doc="One caller-declared continuous endogenous column; linear first-stage projection."),
    Role("instruments", many=True, required=True, doc="Distinct excluded numeric instruments; exogenous predictors are included in both stages."),
)
_OPTIONS = (
    Option("max_iterations", "int", 100, minimum=1, maximum=200),
    Option("tolerance", "float", 1e-9, minimum=1e-12, maximum=1e-4),
    Option("max_work", "int", 10000000000, minimum=1, maximum=10000000000),
    Option("device", "str", "cpu", choices=("cpu",)),
    Option("batch_rows", "int", None, minimum=1, maximum=65536,
           doc="Bounded replay row target; explicit setting selects compact source-bound state."),
)

def _info(name, title, outcome):
    return EstimatorInfo(
        name=name, title=f"{title} control-function regression", family="control_function",
        entry="openecon.econometrics.control_function.commands:fit_control_function",
        function=name, covariances=("robust", "cluster"), default_covariance="robust",
        roles=_ROLES, options=_OPTIONS, outcome=outcome, predictors="optional",
        categorical=False, cluster_dimensions=1, inference="z",
        description="One continuous endogenous regressor and OLS first stage. Full nonsymmetric stacked HC0/whole-cluster CR0 includes generated-control uncertainty. Saved conditional means use joint Gamma/Beta covariance. Resident or bounded Dataset replay on CPU float64; compact Dataset states require unchanged estimation-source semantic replay. No weights, nonlinear first stage, weak-IV/finite-cluster guarantee or structural treatment effects. Gamma/IG working dispersion one; fractional quasi-Bernoulli. Wald only, no estimated-phi ML, LR/AIC or outcome-distribution claim.",
    )


ESTIMATORS = (
    _info("cfregress", "Gaussian identity", "continuous"),
    _info("cflogit", "Binary logit", "binary"),
    _info("cfprobit", "Binary conditional-scale probit", "binary"),
    _info("cfcloglog", "Binary complementary log-log", "binary"),
    _info("cfpoisson", "Poisson log mean", "count"),
    _info("cfgamma", "Gamma log mean", "positive"),
    _info("cfinvgauss", "Inverse-Gaussian log mean", "positive"),
    _info("cffraclogit", "Fractional logit mean", "fractional"),
)

EXPORTS = {"cf_predict": "openecon.econometrics.control_function.postest:cf_predict",
           "cf_restore": "openecon.econometrics.control_function.stream_state:cf_restore"}
