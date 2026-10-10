"""Covariate interval-censored Weibull AFT/PH ML and complete saved queries.

The declared PH chart is a fixed transformation of one Weibull AFT estimate.
Restoration evaluates only the saved endpoint; it never invokes an optimizer.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator

from openecon.econometrics.core import TableSet, table
from openecon.resources import plan_workspace
from . import weibull_regression_common as c
from . import weibull_regression_kernels as k

DOMAIN = dict(
    iid=True,
    family="Weibull",
    censoring="independent noninformative coarsening conditional on fixed covariates",
    endpoint_likelihood="exact density, left CDF, interval survival difference, right survival",
    engine="native torch CPU float64",
    inference="full inverse observed information; asymptotic normal/delta",
    ph_transform="gamma=-exp(-log_sigma)*beta_aft; log_shape=-log_sigma",
    global_optimum_certified=False,
    vendor_validated=False,
    unsupported=[
        "delayed entry",
        "frailty",
        "clusters",
        "weights",
        "survey",
        "recurrent events",
        "time-varying covariates",
        "interval Cox",
        "Dataset",
        "CUDA",
        "MPS",
    ],
)


def specification(*, lower, upper, x, parameterization, level, maxiter, max_work):
    lower, upper = c.name(lower), c.name(upper)
    x = c.names(x, empty=True)
    if len(x) > 8 or lower == upper or lower in x or upper in x:
        c.fail(
            "Declare distinct lower/upper columns and at most eight other numeric covariates.",
            "invalid_argument",
        )
    if parameterization not in ("aft", "ph"):
        c.fail("parameterization must be aft or ph.", "invalid_argument")
    level = c.real(level, "level")
    if not 0 < level < 1 or not 0.5 < (1 + level) / 2 < 1:
        c.fail("level must lie strictly between zero and one.", "invalid_argument")
    return dict(
        lower=lower,
        upper=upper,
        x=x,
        parameterization=parameterization,
        level=level,
        maxiter=c.integer(maxiter, "maxiter", 1, 2000),
        max_work=c.integer(max_work, "max_work", 1, 10**12),
    )


def spec_header(value):
    c.keys(
        value,
        ("lower", "upper", "x", "parameterization", "level", "maxiter", "max_work"),
        "specification",
    )
    expected = specification(**value)
    c.equal(value, expected, "specification", rtol=0)
    return expected


def resource(n, spec, *, components=0):
    q = len(spec["x"]) + 2
    units = 3 * spec["maxiter"] * n * q**2
    if units > spec["max_work"]:
        c.fail(
            f"Declared complete optimizer work {units} exceeds max_work={spec['max_work']}.",
            "work_limit",
        )
    if components > 256:
        c.fail(
            "At most256 complete survival/hazard query components; no thinning.", "resource_limit"
        )
    return plan_workspace(
        "complete interval Weibull ML/query",
        {
            "source chart, derivatives and retained metadata": 2048 * n * q**2,
            "complete joint result covariance and Jacobians": 512
            * (q**2 + components**2 + q * components),
        },
    )


def vector(value, n, label):
    return c.vector(value, n, label)


def endpoint_header(value, n, q):
    c.keys(
        value,
        (
            "log_likelihood",
            "contributions",
            "score",
            "information",
            "covariance",
            "score_quadratic",
            "equilibrated_information_eigenvalues",
        ),
        "endpoint",
    )
    c.real(value["log_likelihood"], "log likelihood")
    c.real(value["score_quadratic"], "score quadratic")
    vector(value["contributions"], n, "contributions")
    vector(value["score"], q, "score")
    vector(value["equilibrated_information_eigenvalues"], q, "OIM eigenvalues")
    for name in ("information", "covariance"):
        c.matrix(value[name], q, q, name)


def parameter_header(value, q):
    c.keys(
        value,
        (
            "aft_parameters",
            "ph_parameters",
            "aft_covariance",
            "ph_covariance",
            "ph_jacobian",
            "aft_information",
            "aft_score",
            "sigma",
            "shape",
            "aft_table",
            "ph_table",
        ),
        "result",
    )
    for name in ("aft_parameters", "ph_parameters", "aft_score"):
        vector(value[name], q, name)
    for name in ("aft_covariance", "ph_covariance", "ph_jacobian", "aft_information"):
        c.matrix(value[name], q, q, name)
    for name in ("sigma", "shape"):
        c.real(value[name], name, positive=True)
    for name in ("aft_table", "ph_table"):
        rows = value[name]
        if not isinstance(rows, (tuple, list)) or len(rows) != q:
            c.fail("Parameter table dimensions disagree.")
        for row in rows:
            c.keys(row, ("parameter", "estimate", "se", "lower", "upper", "level"), "parameter row")
            if not isinstance(row["parameter"], str) or len(row["parameter"]) > 64:
                c.fail("Parameter names exceed the admitted schema.")
            for key in ("estimate", "se", "lower", "upper", "level"):
                c.real(row[key], key)


def fit_header(value):
    c.keys(
        value,
        (
            "schema_version",
            "issue",
            "source",
            "spec",
            "domain",
            "chart",
            "theta",
            "endpoint",
            "result",
            "optimizer",
            "sha256",
        ),
        "saved fit",
    )
    if (
        value["schema_version"] != "interval_weibull_regression_v1"
        or value["issue"] != "MARKET-689"
    ):
        c.fail("Supply complete interval Weibull regression v1 state.")
    spec = spec_header(value["spec"])
    c.equal(value["domain"], DOMAIN, "scientific domain", rtol=0)
    cols, n = c.source_header(value["source"], upper=spec["upper"])
    if cols != [spec["lower"], spec["upper"], *spec["x"]]:
        c.fail("Saved source ordering disagrees with all declared endpoints/covariates.")
    k.endpoints(value["source"], spec)  # complete primitive endpoint refusal before tensors
    q = len(spec["x"]) + 2
    resource(n, spec)
    c.keys(
        value["chart"],
        (
            "location",
            "scale",
            "time_log_center",
            "normalized_design_singular_ratio",
            "original_jacobian",
        ),
        "normalization chart",
    )
    vector(value["chart"]["location"], q - 2, "location")
    vector(value["chart"]["scale"], q - 2, "scale")
    c.real(value["chart"]["time_log_center"], "time center")
    c.real(value["chart"]["normalized_design_singular_ratio"], "design ratio")
    c.matrix(value["chart"]["original_jacobian"], q, q, "chart Jacobian")
    vector(value["theta"], q, "saved normalized coefficients")
    endpoint_header(value["endpoint"], n, q)
    parameter_header(value["result"], q)
    optimizer = value["optimizer"]
    c.keys(
        optimizer,
        (
            "starts",
            "accepted_starts",
            "iterations",
            "method",
            "failures",
            "global_optimum_certified",
        ),
        "bounded optimizer provenance",
    )
    if type(optimizer["starts"]) is not int or optimizer["starts"] != 3:
        c.fail("Exactly three declared starting points are required.")
    accepted = c.integer(optimizer["accepted_starts"], "accepted starts", 1, 3)
    c.integer(optimizer["iterations"], "iterations", 0, spec["maxiter"] + 10)
    if (
        optimizer["method"] != "bfgs_with_observed_hessian"
        or optimizer["global_optimum_certified"] is not False
    ):
        c.fail("Optimizer provenance may not assert a global certificate.")
    if (
        not isinstance(optimizer["failures"], (list, tuple))
        or len(optimizer["failures"]) != 3 - accepted
    ):
        c.fail("Complete bounded start-failure provenance disagrees.")
    for failure in optimizer["failures"]:
        c.keys(failure, ("code", "message"), "start failure")
        if any(not isinstance(failure[key], str) or len(failure[key]) > 512 for key in failure):
            c.fail("Start failure provenance exceeds its primitive envelope.")
    if not isinstance(value["sha256"], str) or len(value["sha256"]) != 64:
        c.fail("Supply the complete saved state checksum.")
    return spec, n


def fit_replay(value):
    spec, _ = fit_header(value)
    if c.digest({key: val for key, val in value.items() if key != "sha256"}) != value["sha256"]:
        c.fail("Complete Weibull state checksum mismatch.")
    data = k.prepare(value["source"], spec)
    c.equal(value["chart"], data["chart"], "complete chart")
    endpoint = k.evaluate(k.tensor(value["theta"]), data)
    c.equal(value["endpoint"], endpoint, "full saved ML endpoint")
    result = k.parameter_result(value["theta"], endpoint, data, spec)
    c.equal(value["result"], result, "full AFT/PH inference")
    return value


def times_header(times):
    if not isinstance(times, (tuple, list)) or not 1 <= len(times) <= 128:
        c.fail("Declare1..128 finite prediction times.", "invalid_argument")
    values = [c.real(v, "query time") for v in times]
    if any(v < 0 or v > 1e12 or 0 < v < 1e-12 for v in values) or any(
        b <= a for a, b in zip(values, values[1:])
    ):
        c.fail(
            "Prediction times require unique increasing zero or [1e-12,1e12] endpoints.",
            "invalid_argument",
        )
    return values


def query_cache_header(value, n, m, q):
    c.keys(
        value,
        (
            "rows",
            "component_names",
            "estimates",
            "jacobian",
            "normalized_jacobian",
            "covariance",
            "parameter_query_covariance",
            "log_hazard_jacobian",
        ),
        "complete query result",
    )
    count = 2 * n * m
    vector(value["estimates"], count, "query estimates")
    for key in ("jacobian", "normalized_jacobian"):
        c.matrix(value[key], count, q, key)
    c.matrix(value["covariance"], count, count, "full query covariance")
    c.matrix(value["parameter_query_covariance"], q, count, "parameter/query cross covariance")
    c.matrix(value["log_hazard_jacobian"], n * m, q, "log hazard Jacobian")
    if (
        not isinstance(value["component_names"], (list, tuple))
        or len(value["component_names"]) != count
        or any(not isinstance(v, str) or len(v) > 64 for v in value["component_names"])
    ):
        c.fail("Complete query component labels disagree.")
    if not isinstance(value["rows"], (list, tuple)) or len(value["rows"]) != n * m:
        c.fail("Complete query row dimensions disagree.")
    fields = (
        "position",
        "time",
        "survival",
        "survival_se",
        "survival_lower",
        "survival_upper",
        "cumulative_hazard",
        "hazard_se",
        "hazard_lower",
        "hazard_upper",
        "level",
    )
    for row in value["rows"]:
        c.keys(row, fields, "query row")
        c.integer(row["position"], "query position", 0, n - 1)
        for key in fields[1:]:
            c.real(row[key], key)


def query_header(value):
    c.keys(
        value,
        ("schema_version", "issue", "target", "source", "times", "level", "result", "sha256"),
        "saved query",
    )
    if (
        value["schema_version"] != "interval_weibull_prediction_v1"
        or value["issue"] != "MARKET-689"
    ):
        c.fail("Supply complete saved Weibull prediction v1 state.")
    spec, _ = fit_header(value["target"])
    times = times_header(value["times"])
    level = c.real(value["level"], "query level")
    if not 0 < level < 1 or not 0.5 < (1 + level) / 2 < 1:
        c.fail("Query level requires a strict probability.")
    cols, n = c.source_header(value["source"])
    if cols != spec["x"]:
        c.fail("Query source must carry all fitted covariates in their exact order.")
    resource(n, spec, components=2 * n * len(times))
    query_cache_header(value["result"], n, len(times), len(spec["x"]) + 2)
    if not isinstance(value["sha256"], str) or len(value["sha256"]) != 64:
        c.fail("Query requires the complete checksum.")
    return times, level


def query_replay(value):
    times, level = query_header(value)
    if c.digest({key: val for key, val in value.items() if key != "sha256"}) != value["sha256"]:
        c.fail("Saved Weibull query checksum mismatch.")
    target = fit_replay(value["target"])
    expected = k.prediction(
        target["theta"],
        target["endpoint"]["covariance"],
        target["chart"],
        value["source"],
        times,
        level,
    )
    c.equal(value["result"], expected, "complete saved query inference")
    return value


def admitted(value, *, query=False):
    if isinstance(value, _Transport):
        value = value.payload
    value = c.load(value)
    if set(value) == {"payload"}:
        value = value["payload"]
    return query_replay(value) if query else fit_replay(value)


class _Transport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    payload: dict

    @field_validator("payload", mode="before")
    @classmethod
    def primitive_before_copy(cls, value):
        c.measure(value)
        return value

    @model_validator(mode="after")
    def complete_semantics(self):
        validated = (
            query_replay(self.payload)
            if isinstance(self, WeibullIntervalPrediction)
            else fit_replay(self.payload)
        )
        object.__setattr__(self, "payload", c._freeze(validated))
        return self

    def _full(self):
        return admitted(self, query=isinstance(self, WeibullIntervalPrediction))

    @field_serializer("payload")
    def complete_payload(self, value):
        self._full()
        return c._thaw(value)

    @wraps(BaseModel.model_dump)
    def model_dump(self, *args, **kwargs):
        self._full()
        return super().model_dump(*args, **kwargs)

    @wraps(BaseModel.model_dump_json)
    def model_dump_json(self, *args, **kwargs):
        indent = kwargs.get("indent")
        if indent is not None:
            c.integer(indent, "JSON indent", 0, 16)
        payload = self._full()
        c.measure({"payload": payload}, indent=indent or 0)
        return super().model_dump_json(*args, **kwargs)

    @classmethod
    @wraps(BaseModel.model_validate_json.__func__)
    def model_validate_json(cls, json_data, *args, **kwargs):
        value = c.load(json_data)
        return cls.model_validate(value, *args, **kwargs)

    @wraps(BaseModel.model_copy)
    def model_copy(self, *, update=None, deep=False):
        if type(deep) is not bool:
            c.fail("deep requires a boolean copy policy.")
        if update is not None:
            c.keys(update, ("payload",), "model copy update")
            prospective = c.load(update["payload"])
            if set(prospective) == {"payload"}:
                prospective = prospective["payload"]
            if isinstance(self, WeibullIntervalPrediction):
                query_header(prospective)
            else:
                fit_header(prospective)
        value = self._full()
        if update is not None:
            value = admitted(update["payload"], query=isinstance(self, WeibullIntervalPrediction))
        return type(self).model_validate({"payload": value})

    def __deepcopy__(self, memo=None):
        return self.model_copy(deep=True)

    def table(self, name=None):
        tables = self.to_tables()
        return tables[name or next(iter(tables))]

    def latex(self, name=None):
        return self.table(name).to_latex(index=False)

    def to_latex(self, name=None):
        return self.latex(name)


class WeibullIntervalFit(_Transport):
    """Complete regular iid interval-Weibull AFT/PH inference, without refitting."""

    @c.checked
    def to_tables(self):
        value = self._full()
        result, spec = value["result"], value["spec"]
        parameterization = spec["parameterization"]
        parameters = result[parameterization + "_table"]
        names = [row["parameter"] for row in parameters]
        covariance = result[parameterization + "_covariance"]
        covrows = [
            dict(row=a, column=b, covariance=covariance[i][j])
            for i, a in enumerate(names)
            for j, b in enumerate(names)
        ]
        aft_names = ["Intercept", *spec["x"], "log_sigma"]
        inforows = [
            dict(row=a, column=b, information=result["aft_information"][i][j])
            for i, a in enumerate(aft_names)
            for j, b in enumerate(aft_names)
        ]
        return TableSet(
            {
                "parameters": table(parameters),
                "parameter_covariance": table(covrows),
                "aft_observed_information": table(inforows),
                "model": table(
                    [
                        dict(
                            n=len(value["source"]["values"]),
                            log_likelihood=value["endpoint"]["log_likelihood"],
                            shape=result["shape"],
                            sigma=result["sigma"],
                            score_quadratic=value["endpoint"]["score_quadratic"],
                        )
                    ]
                ),
            },
            title="Covariate interval Weibull " + parameterization.upper(),
            state=value,
            method="stinterval_weibull_regression",
            domain=DOMAIN,
        )

    @c.checked
    def dataset(self):
        value = self._full()
        return c.frame(value["source"], upper=value["spec"]["upper"])


class WeibullIntervalPrediction(_Transport):
    """Complete cross-profile/time survival/hazard and parameter/query covariance."""

    @c.checked
    def to_tables(self):
        value = self._full()
        result = value["result"]
        names = result["component_names"]
        covariance = result["covariance"]
        covrows = [
            dict(row=a, column=b, covariance=covariance[i][j])
            for i, a in enumerate(names)
            for j, b in enumerate(names)
        ]
        return TableSet(
            {"curves": table(result["rows"]), "curve_covariance": table(covrows)},
            title="Interval Weibull survival and cumulative hazard",
            state=value,
            method="interval_weibull_regression_predict",
            domain=DOMAIN,
        )


@c.checked
def stinterval_weibull_regression(
    data,
    *,
    lower,
    upper,
    x=(),
    parameterization="aft",
    level=0.95,
    maxiter=1000,
    max_work=500000000,
    device="cpu",
    weights=None,
    cluster=None,
    entry=None,
    frailty=None,
    missing="raise",
):
    """Fit bounded iid exact/left/interval/right Weibull regression in AFT or PH units."""
    if (
        device != "cpu"
        or missing != "raise"
        or any(v is not None for v in (weights, cluster, entry, frailty))
    ):
        c.fail(
            "Native CPU iid missing=raise only; weights/clusters/entry/frailty are unsupported.",
            "unsupported_domain",
        )
    spec = specification(
        lower=lower,
        upper=upper,
        x=x,
        parameterization=parameterization,
        level=level,
        maxiter=maxiter,
        max_work=max_work,
    )
    # Count/plan before source/index copying, including mapping-Series construction.
    if isinstance(data, Mapping) and not isinstance(data, c.pd.DataFrame):
        if lower not in data or not hasattr(data[lower], "__len__"):
            c.fail("Lower source column is absent/nonresident.")
        n = len(data[lower])
    elif isinstance(data, c.pd.DataFrame):
        n = len(data)
    else:
        c.fail("Supply a resident DataFrame or aligned mapping.", "unsupported_input")
    c.integer(n, "source rows", 1, c.MAX_ROWS)
    resource(n, spec)
    source = c.source(data, [lower, upper, *spec["x"]], upper=upper)
    k.endpoints(source, spec)
    data_native = k.prepare(source, spec)
    theta, endpoint, optimizer = k.fit(data_native, spec)
    result = k.parameter_result(theta, endpoint, data_native, spec)
    payload = c.seal(
        dict(
            schema_version="interval_weibull_regression_v1",
            issue="MARKET-689",
            source=source,
            spec=spec,
            domain=DOMAIN,
            chart=data_native["chart"],
            theta=theta,
            endpoint=endpoint,
            result=result,
            optimizer=optimizer,
        )
    )
    return WeibullIntervalFit(payload=payload)


@c.checked
def restore_interval_weibull_regression(state):
    return WeibullIntervalFit(payload=admitted(state))


@c.checked
def interval_weibull_regression_predict(state, data, *, times, level=0.95, device="cpu"):
    """Evaluate declared profiles/times with full covariance; no optimizer invocation."""
    if device != "cpu":
        c.fail("Prediction requires native CPU float64.", "unsupported_device")
    times, level = times_header(times), c.real(level, "level")
    if not 0 < level < 1 or not 0.5 < (1 + level) / 2 < 1:
        c.fail("level requires a strict probability.", "invalid_argument")
    raw = c.load(state.payload if isinstance(state, _Transport) else state)
    if set(raw) == {"payload"}:
        raw = raw["payload"]
    spec, _ = fit_header(raw)  # only primitive/source preflight; no fitted-target information work
    if isinstance(data, c.pd.DataFrame):
        n = len(data)
    elif isinstance(data, Mapping) and spec["x"] and spec["x"][0] in data:
        n = len(data[spec["x"][0]])
    else:
        c.fail("Query requires a resident complete DataFrame/aligned covariate mapping.")
    c.integer(n, "query rows", 1, 128)
    resource(n, spec, components=2 * n * len(times))
    source = c.source(data, spec["x"])
    target = fit_replay(raw)
    result = k.prediction(
        target["theta"], target["endpoint"]["covariance"], target["chart"], source, times, level
    )
    payload = c.seal(
        dict(
            schema_version="interval_weibull_prediction_v1",
            issue="MARKET-689",
            target=target,
            source=source,
            times=times,
            level=level,
            result=result,
        )
    )
    return WeibullIntervalPrediction(payload=payload)


@c.checked
def restore_interval_weibull_prediction(state):
    return WeibullIntervalPrediction(payload=admitted(state, query=True))
