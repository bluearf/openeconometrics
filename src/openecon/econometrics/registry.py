"""Static, Torch-free catalogue of every OpenEconometrics estimator.

The registry is the single statement of what a ``ModelSpec`` may contain for
each estimator: column roles, options, covariance estimators, weights and the
panel/time structure. It validates specifications, drives ``oe.capabilities()``
and dispatches ``oe.fit``. It deliberately imports neither Torch nor pandas:
the web control process and MCP clients read it without loading the tensor
runtime. Estimator code is imported only when a model is actually fitted.

Each family package under ``openecon.econometrics`` exposes ``ESTIMATORS``, a
tuple of :class:`EstimatorInfo`, from a dependency-free ``__init__`` module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

# Family packages, in the order their estimators are listed to users.
FAMILIES: tuple[str, ...] = (
    'linear', 'panel', 'iv', 'glm', 'discrete', 'limited', 'count',
    'quantile', 'arima', 'arch', 'var', 'unitroot', 'tsmodels', 'stats',
    'nonparametric', 'multivariate', 'selection', 'survival', 'dpanel', 'teffects', 'mixed',
    'systems', 'postest', 'spatial', 'regularized', 'tsworkflows', 'mgarch', 'robust',
    'longrun', 'panel_ardl', 'structural', 'decomposition', 'meta', 'fractional', 'smoothing',
    'irt', 'categorical', 'causal', 'survey', 'measurement', 'temporal', 'conjoint', 'mi', 'causal_design', 'survival_ext', 'finite', 'conditional', 'twostep', 'control_function',
    'bayesian', 'latent', 'ivquantile', 'weakiv', 'mixtures', 'supervised',
    'bayesian_var_public',
    'interval_weibull_covariate_public',
)

WEIGHT_TYPES = ("aweight", "fweight", "pweight", "iweight")

# Canonical covariance names and what they mean. Estimators list the subset
# they implement; anything else is rejected before estimation starts.
COVARIANCE_KINDS = {
    "nonrobust": "conventional: classical OLS/GLS variance, or the observed information for likelihood models",
    "HC0": "heteroskedasticity-robust (White), no small-sample factor",
    "HC1": "heteroskedasticity-robust with the N/(N-K) factor (Stata vce(robust) for linear models)",
    "HC2": "heteroskedasticity-robust, residuals scaled by 1/sqrt(1-h)",
    "HC3": "heteroskedasticity-robust, residuals scaled by 1/(1-h)",
    "robust": "Huber/White sandwich for likelihood and GMM estimators (Stata vce(robust))",
    "cluster": "cluster-robust sandwich; several cluster columns select multiway clustering",
    "opg": "outer product of the gradient (BHHH)",
    "hac": "heteroskedasticity- and autocorrelation-consistent (Newey-West) with a lag option",
    "driscoll_kraay": "Driscoll-Kraay cross-section and autocorrelation robust panel covariance",
    "bootstrap": "nonparametric (cluster/panel aware) bootstrap with a recorded seed",
    "jackknife": "delete-one (or delete-one-cluster) jackknife",
}


@dataclass(frozen=True)
class Role:
    """An additional named column role, stored in ``ModelSpec.columns``."""

    name: str
    many: bool = False          # a list of columns rather than one column
    required: bool = False
    kind: str = "numeric"       # "numeric" or "label" (any scalar grouping value)
    doc: str = ""


@dataclass(frozen=True)
class Option:
    """An estimator-specific setting, stored in ``ModelSpec.options``.

    ``type`` is one of: int, float, number, str, bool, list[int], list[float],
    list[str], json. ``default`` documents what the estimator uses when the
    option is absent; the spec itself only carries options that were given.
    """

    name: str
    type: str
    default: Any = None
    choices: tuple[Any, ...] = ()
    doc: str = ""
    minimum: float | None = None
    maximum: float | None = None
    required: bool = False


@dataclass(frozen=True)
class EstimatorInfo:
    name: str
    title: str
    family: str
    entry: str                              # "package.module:function" taking (spec, data)
    covariances: tuple[str, ...]
    default_covariance: str
    description: str = ""
    stata: tuple[str, ...] = ()             # equivalent Stata commands
    function: str | None = None             # public convenience function, exported as oe.<function>
    roles: tuple[Role, ...] = ()
    options: tuple[Option, ...] = ()
    outcome: str = "continuous"             # documentation: continuous, binary, count, ordered, ...
    predictors: str = "required"            # "required", "optional" or "none"
    categorical: bool = True
    intercept: str = "optional"             # "optional", "always" or "never"
    weights: tuple[str, ...] = ()
    panel: str = "none"                     # "none", "optional" or "required"
    time: str = "none"
    cluster_dimensions: int = 1
    inference: str = "z"                    # coefficient reference: "t", "z", or "none" for predictive targets
    legacy: bool = False

    def role(self, name: str) -> Role | None:
        return next((role for role in self.roles if role.name == name), None)

    def option(self, name: str) -> Option | None:
        return next((option for option in self.options if option.name == name), None)


def _legacy() -> tuple[EstimatorInfo, ...]:
    """The core estimators (ols, logit, probit); their covariance lists stay in one place."""
    from openecon.analysis_contracts import _SUPPORTED_COVARIANCES

    described = {
        "ols": ("Ordinary least squares", ("regress",), "continuous",
                "Comprehensive OLS/WLS with robust, cluster, HAC and resampling covariance "
                "(openecon.linear_ols)."),
        "logit": ("Logistic regression", ("logit",), "binary",
                  "Binary logit by Newton steps with a separation certificate."),
        "probit": ("Probit regression", ("probit",), "binary",
                   "Binary probit by Newton steps with a separation certificate."),
    }
    return tuple(
        EstimatorInfo(
            name=name, title=title, family="core", entry="openecon.analysis:fit",
            covariances=tuple(_SUPPORTED_COVARIANCES[name]) if name == "ols" else
                ("nonrobust", "cluster", "opg", "robust"), default_covariance="nonrobust",
            stata=stata, function=name, outcome=outcome, inference="t" if name == "ols" else "z",
            legacy=True, description=description,
            predictors="optional" if name == "ols" else "required",
            weights=WEIGHT_TYPES,
            cluster_dimensions=4 if name == "ols" else 1,
        )
        for name, (title, stata, outcome, description) in described.items()
    )


_catalogue: dict[str, EstimatorInfo] | None = None


def _load() -> dict[str, EstimatorInfo]:
    global _catalogue
    if _catalogue is None:
        legacy = _legacy()
        catalogue = {info.name: info for info in legacy}
        functions = {info.function for info in legacy}
        for family in FAMILIES:
            for info in import_module(f"openecon.econometrics.{family}").ESTIMATORS:
                if info.name in catalogue:
                    raise RuntimeError(f"Estimator '{info.name}' is registered twice.")
                if info.function is not None:
                    if info.function in functions:
                        raise RuntimeError(f"Public function '{info.function}' is registered twice.")
                    functions.add(info.function)
                unknown = set(info.covariances) - set(COVARIANCE_KINDS)
                if unknown or info.default_covariance not in info.covariances:
                    raise RuntimeError(f"Estimator '{info.name}' declares invalid covariance names.")
                catalogue[info.name] = info
        _catalogue = catalogue
    return _catalogue


def names() -> list[str]:
    return list(_load())


def get(name: str) -> EstimatorInfo:
    try:
        return _load()[name]
    except KeyError:
        raise ValueError(f"Unknown estimator '{name}'. Available: {', '.join(_load())}.") from None


def all_estimators() -> list[EstimatorInfo]:
    return list(_load().values())


def load_entry(info: EstimatorInfo) -> Callable[..., Any]:
    module, function = info.entry.split(":")
    return getattr(import_module(module), function)


def public_exports() -> dict[str, tuple[str, str]]:
    """Map ``oe.<name>`` convenience functions to their defining module."""
    exports = {}
    for info in _load().values():
        if info.function is not None and not info.legacy:
            exports[info.function] = (info.entry.split(":")[0], info.function)
    # Helpers shared by every family.
    exports["forecast"] = ("openecon.econometrics.core", "forecast")
    # A family may publish further helpers (tests, post-estimation) through EXPORTS.
    for family in FAMILIES:
        for name, target in getattr(import_module(f"openecon.econometrics.{family}"), "EXPORTS", {}).items():
            if name in exports:
                raise RuntimeError(f"Public function '{name}' is exported twice (family '{family}').")
            exports[name] = tuple(target.split(":"))
    return exports


def forecasters() -> dict[str, tuple[str, str]]:
    """Map estimator names to their forecast function (a family manifest's ``FORECAST`` dict).

    ``oe.forecast(result, steps, ...)`` dispatches on ``result.spec.estimator``,
    so every time-series family shares one public name.
    """
    table: dict[str, tuple[str, str]] = {}
    for family in FAMILIES:
        for name, target in getattr(import_module(f"openecon.econometrics.{family}"), "FORECAST", {}).items():
            table[name] = tuple(target.split(":"))
    return table


def auxiliary_exports() -> dict[str, dict[str, str]]:
    """Manifest helper registrations, separate from fit/prediction coverage.

    This loads only the same Torch-free manifests as public_exports. A registered
    helper has its own method/option and evidence contract; no estimation,
    numerical validation or installed/public shipment is inferred from presence.
    """
    result = {}
    for family in FAMILIES:
        manifest = import_module(f"openecon.econometrics.{family}")
        for name, target in getattr(manifest, "EXPORTS", {}).items():
            if name in result:
                raise RuntimeError(f"Auxiliary function '{name}' is exported twice.")
            result[name] = {"family": family, "entry": target}
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_option(estimator: str, option: Option, value: Any) -> None:
    kind = option.type
    checks = {
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": _is_number, "number": _is_number,
        "str": lambda v: isinstance(v, str), "bool": lambda v: isinstance(v, bool),
        "list[int]": lambda v: isinstance(v, list) and all(isinstance(i, int) and not isinstance(i, bool) for i in v),
        "list[float]": lambda v: isinstance(v, list) and all(_is_number(i) for i in v),
        "list[str]": lambda v: isinstance(v, list) and all(isinstance(i, str) for i in v),
        "json": lambda v: True,
    }
    if kind not in checks:
        raise RuntimeError(f"Option '{option.name}' of '{estimator}' declares unknown type '{kind}'.")
    if not checks[kind](value):
        raise ValueError(f"Option '{option.name}' of {estimator} must be of type {kind}.")
    if option.choices and value not in option.choices:
        raise ValueError(f"Option '{option.name}' of {estimator} must be one of: "
                         f"{', '.join(map(str, option.choices))}.")
    if kind in {"int", "float", "number"}:
        if option.minimum is not None and value < option.minimum:
            raise ValueError(f"Option '{option.name}' of {estimator} must be at least {option.minimum:g}.")
        if option.maximum is not None and value > option.maximum:
            raise ValueError(f"Option '{option.name}' of {estimator} must be at most {option.maximum:g}.")


def cluster_columns(spec: Any) -> list[str]:
    cluster = spec.cluster
    return [] if cluster is None else [cluster] if isinstance(cluster, str) else list(cluster)


def role_columns(spec: Any, name: str) -> list[str]:
    """Columns given for one extra role, always as a list."""
    value = spec.columns.get(name)
    return [] if value is None else [value] if isinstance(value, str) else list(value)


def spec_columns(spec: Any) -> list[str]:
    """Every data column a specification refers to, in a stable order."""
    names_ = [spec.outcome, *spec.predictors, *cluster_columns(spec)]
    names_.extend(name for name in (spec.weights, spec.panel, spec.time) if name is not None)
    for role in spec.columns:
        names_.extend(role_columns(spec, role))
    return list(dict.fromkeys(names_))


def validate_spec(spec: Any) -> str:
    """Check a ModelSpec against its estimator's contract; return its covariance.

    Raises ``ValueError`` (surfaced by pydantic as a ``ValidationError``) with a
    message naming the offending field. Legacy estimators keep their original,
    stricter contract: no weights, panel structure, extra roles or options.
    """
    info = get(spec.estimator)
    clusters = cluster_columns(spec)
    if len(clusters) != len(set(clusters)):
        raise ValueError("Cluster columns must be distinct.")
    if spec.outcome in clusters:
        raise ValueError("The outcome must not also be the cluster column.")
    covariance = spec.covariance
    if info.legacy:
        # Keep the unweighted core unchanged. Explicit weights have a separate
        # resident ML route, with its own recorded finite-sample conventions.
        if (spec.weights is None) != (spec.weight_type is None):
            raise ValueError("weights and weight_type must be given together.")
        weighted = spec.weights is not None
        covariance = covariance or ("cluster" if clusters else
                                   "robust" if spec.weight_type == "pweight" else "nonrobust")
        allowed = {"nonrobust", "opg", "robust", "cluster"} if weighted else {"nonrobust", "cluster"}
        if covariance not in allowed:
            raise ValueError("Binary models support nonrobust or cluster covariance without "
                             "weights; weighted fits also support opg and robust.")
        if spec.weight_type == "pweight" and covariance not in {"robust", "cluster"}:
            raise ValueError("Binary pweights require robust or cluster covariance.")
        if (isinstance(spec.cluster, list) or spec.time
                or spec.panel or spec.options or spec.columns):
            raise ValueError("Binary models do not accept OLS-specific options.")
        if covariance == "cluster" and spec.cluster is None:
            raise ValueError("Cluster covariance requires a cluster column.")
        if covariance != "cluster" and spec.cluster is not None:
            raise ValueError("A cluster column is only valid with cluster covariance.")
        return covariance
    if covariance is None:
        covariance = "cluster" if clusters else info.default_covariance
    if covariance not in info.covariances:
        raise ValueError(f"{spec.estimator} supports only these covariance estimators: "
                         f"{', '.join(info.covariances)}.")
    if covariance == "cluster" and not clusters:
        raise ValueError("Cluster covariance requires a cluster column.")
    if clusters and covariance not in {"cluster", "bootstrap", "jackknife"}:
        raise ValueError("A cluster column is only valid with cluster, bootstrap or jackknife covariance.")
    if len(clusters) > info.cluster_dimensions:
        raise ValueError(f"{spec.estimator} supports at most {info.cluster_dimensions} cluster dimension(s).")
    if info.predictors == "required" and not spec.predictors:
        raise ValueError("At least one predictor is required.")
    if info.predictors == "none" and spec.predictors:
        raise ValueError(f"{spec.estimator} does not take predictors.")
    if spec.categorical and not info.categorical:
        raise ValueError(f"{spec.estimator} does not expand categorical predictors.")
    if info.intercept == "always" and not spec.intercept:
        raise ValueError(f"{spec.estimator} always includes its constant term(s).")
    if info.intercept == "never" and spec.intercept:
        raise ValueError(f"{spec.estimator} has no constant term; pass intercept=False.")
    if (spec.weights is None) != (spec.weight_type is None):
        raise ValueError("weights and weight_type must be given together.")
    if spec.weight_type is not None and spec.weight_type not in info.weights:
        allowed = ", ".join(info.weights) if info.weights else "none"
        raise ValueError(f"{spec.estimator} does not support {spec.weight_type}s (allowed: {allowed}).")
    for field in ("panel", "time"):
        need, given = getattr(info, field), getattr(spec, field)
        if need == "required" and given is None:
            raise ValueError(f"{spec.estimator} requires a {field} column.")
        if need == "none" and given is not None:
            raise ValueError(f"{spec.estimator} does not use a {field} column.")
    for name, value in spec.columns.items():
        role = info.role(name)
        if role is None:
            allowed = ", ".join(role.name for role in info.roles) or "none"
            raise ValueError(f"{spec.estimator} has no column role '{name}' (allowed: {allowed}).")
        if isinstance(value, list) and not role.many:
            raise ValueError(f"Column role '{name}' of {spec.estimator} takes a single column.")
        columns = role_columns(spec, name)
        if len(columns) != len(set(columns)):
            raise ValueError(f"Column role '{name}' contains duplicate names.")
    for role in info.roles:
        if role.required and not role_columns(spec, role.name):
            raise ValueError(f"{spec.estimator} requires the column role '{role.name}'.")
    for name, value in spec.options.items():
        option = info.option(name)
        if option is None:
            allowed = ", ".join(option.name for option in info.options) or "none"
            raise ValueError(f"{spec.estimator} has no option '{name}' (allowed: {allowed}).")
        _check_option(spec.estimator, option, value)
    for option in info.options:
        if option.required and option.name not in spec.options:
            raise ValueError(f"{spec.estimator} requires the option '{option.name}'.")
    return covariance


def describe(info: EstimatorInfo) -> dict[str, Any]:
    """JSON-safe capability record for one estimator."""
    record: dict[str, Any] = {
        "covariances": list(info.covariances),
        "binary_outcome": info.outcome == "binary",
        "categorical_predictors": info.categorical,
        "intercept": info.intercept != "never",
        "missing": ["raise", "drop"],
        "inference": "Student t" if info.inference == "t" else "unavailable (prediction target)" if info.inference == "none" else "Normal z",
    }
    if info.legacy:
        if info.name in {"logit", "probit"}:
            record.update({
                "weights": list(info.weights),
                "unweighted_covariances": ["nonrobust", "cluster"],
                "weighted_binary": {
                    "route": "resident CPU float64; Dataset unsupported",
                    "covariances_by_weight": {
                        key: ["nonrobust", "opg", "robust", "cluster"]
                        if key != "pweight" else ["robust", "cluster"] for key in info.weights},
                    "pweight_default": "robust unless cluster is supplied",
                    "semantics": "fweight replicates rows; aweight normalizes to retained rows; "
                                 "iweight/pweight use positive supplied weights",
                    "corrections": "weighted ML robust N/(N-1), cluster G/(G-1); "
                                   "unweighted legacy CR1 is preserved",
                    "limits": {"original_rows": 100000, "parameters": 64,
                               "iterations": 100, "dense_work": 5000000000},
                    "aweight_note": "OpenEconometrics extension; not a Stata logit/probit option",
                },
            })
        return record
    record.update({
        "title": info.title, "family": info.family, "description": info.description,
        "stata": list(info.stata), "function": info.function, "outcome": info.outcome,
        "default_covariance": info.default_covariance, "predictors": info.predictors,
        "intercept_rule": info.intercept, "weights": list(info.weights),
        "panel": info.panel, "time": info.time, "cluster_dimensions": info.cluster_dimensions,
        "columns": {role.name: {"many": role.many, "required": role.required, "kind": role.kind,
                                "doc": role.doc} for role in info.roles},
        "options": {option.name: {"type": option.type, "default": option.default,
                                  "choices": list(option.choices), "required": option.required,
                                  "doc": option.doc} for option in info.options},
    })
    return record
