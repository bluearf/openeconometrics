"""Validated finite-prior systems and explicitly supplied equation paths.

Schedules replace a static matrix at each original input row. Exogenous paths
add fixed linear terms to c_t and d_t. T_t and c_t take a_t to a_{t+1}; in
particular, the final fitted transition determines the forecast-origin prior.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.tsworkflows.common import finite_tensor, horizon, pd_check


PATH_KEYS = ("T", "Z", "Q", "H", "c", "d")
_SYSTEM_KEYS = {*PATH_KEYS, "a0", "P0", "initialization", "parameters", "schedules", "exogenous"}
_MAX_EXOG_COLUMNS = 32


def _exogenous(record):
    if not isinstance(record, dict):
        raise AnalysisError("invalid_system", "A state-space system must be a dictionary.")
    definitions = record.get("exogenous", {})
    if not isinstance(definitions, dict) or set(definitions) - {"state", "measurement"}:
        raise AnalysisError(
            "invalid_exogenous", "Exogenous definitions support only state and measurement equations."
        )
    for equation, definition in definitions.items():
        if not isinstance(definition, dict) or set(definition) != {"columns", "coefficients"}:
            raise AnalysisError(
                "invalid_exogenous", f"{equation} exogenous input needs columns and coefficients."
            )
        columns = definition["columns"]
        if (
            not isinstance(columns, list)
            or not 1 <= len(columns) <= _MAX_EXOG_COLUMNS
            or any(not isinstance(name, str) or not name for name in columns)
            or len(set(columns)) != len(columns)
        ):
            raise AnalysisError(
                "invalid_exogenous",
                f"{equation} columns must be 1..{_MAX_EXOG_COLUMNS} unique nonempty names.",
            )
    return definitions


def exogenous_columns(record):
    """Model-input columns required before constructing the sample frame."""
    return list(dict.fromkeys(
        column for definition in _exogenous(record).values() for column in definition["columns"]
    ))


def _shape(value, shape, name):
    """Check geometry before tensor conversion can allocate an oversized block."""
    if (
        isinstance(value, complex)
        or getattr(getattr(value, "dtype", None), "kind", None) == "c"
        or (isinstance(value, torch.Tensor) and value.is_complex())
    ):
        raise AnalysisError("invalid_system", f"{name} must contain real numbers.")
    actual = getattr(value, "shape", None)
    if actual is not None:
        if tuple(actual) != tuple(shape):
            raise AnalysisError("invalid_system", f"{name} has invalid dimensions.")
        return
    if not shape:
        if isinstance(value, (list, tuple, dict)):
            raise AnalysisError("invalid_system", f"{name} has invalid dimensions.")
        return
    if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
        raise AnalysisError("invalid_system", f"{name} has invalid dimensions.")
    for item in value:
        _shape(item, shape[1:], name)


def _array(value, shape, name):
    _shape(value, shape, name)
    return finite_tensor(value, shape, name)


class System:
    """Static Gaussian system with optional known schedules and regressors.

    A supplied frame maps original-row schedules into its chronological sample
    order. A frame-less instance can expand future paths from persisted fitted
    static values and equation templates; training exogenous values require a
    frame. Free parameters may occupy only unscheduled matrices.
    """

    def __init__(self, record: dict, p: int, frame=None):
        if not isinstance(record, dict) or set(record) - _SYSTEM_KEYS:
            raise AnalysisError("invalid_system", "Unknown or missing state-space system fields.")
        raw = record.get("T")
        if (
            not isinstance(raw, list)
            or not 1 <= len(raw) <= 16
            or isinstance(p, bool)
            or not isinstance(p, int)
            or not 1 <= p <= 8
        ):
            raise AnalysisError("system_budget", "Supported domain: 1..16 states and 1..8 measurements.")
        self.m, self.p, self.frame = len(raw), p, frame
        m = self.m
        parameters = record.get("parameters", [])
        if not isinstance(parameters, list) or len(parameters) > 20:
            raise AnalysisError("system_budget", "At most20 individually identified free parameters are supported.")
        self.shapes = {"T": (m, m), "Z": (p, m), "Q": (m, m), "H": (p, p),
                       "c": (m,), "d": (p,), "a0": (m,), "P0": (m, m)}
        self.initialization = record.get("initialization", "known")
        if not isinstance(self.initialization, str) or self.initialization not in {"known", "stationary"}:
            raise AnalysisError(
                "invalid_initialization",
                "Use known finite prior or exact stationary initialization; diffuse priors are unsupported.",
            )
        schedules = record.get("schedules", {})
        if not isinstance(schedules, dict) or set(schedules) - set(PATH_KEYS):
            raise AnalysisError("invalid_schedule", "Only T/Z/Q/H/c/d may have known schedules.")
        self.scheduled_keys = tuple(schedules)
        definitions = _exogenous(record)
        self.exogenous_columns = exogenous_columns(record)
        self.exogenous_column_order = {
            equation: list(definition["columns"]) for equation, definition in definitions.items()
        }
        self.has_paths = bool(schedules or definitions)
        if self.initialization == "stationary" and (
            set(schedules) & {"T", "Q", "c"} or "state" in definitions
        ):
            raise AnalysisError(
                "invalid_initialization",
                "Stationary initialization requires fixed T/Q/c and no state exogenous path.",
            )
        if frame is not None:
            self.n, self.source_n = frame.n, len(frame.original)
        elif schedules:
            first = next(iter(schedules.values()))
            if not isinstance(first, list):
                raise AnalysisError("invalid_schedule", "Schedules must be original-row lists.")
            self.n = self.source_n = len(first)
        else:
            self.n = self.source_n = None
        if self.has_paths and self.n is not None and not 1 <= self.n <= 20000:
            raise AnalysisError("system_budget", "Known equation paths support 1..20000 retained periods.")
        if self.has_paths and self.source_n is not None and self.source_n > 20000:
            raise AnalysisError("system_budget", "Known schedules support at most 20000 original periods.")
        for key, values in schedules.items():
            if not isinstance(values, list) or self.source_n is None or len(values) != self.source_n:
                raise AnalysisError(
                    "invalid_schedule", f"Schedule {key} needs exactly one array per original input row."
                )
            _shape(values, (self.source_n, *self.shapes[key]), f"schedule {key}")
        static_cells = sum(math.prod(shape) for shape in self.shapes.values())
        schedule_cells = sum(math.prod(self.shapes[key]) for key in schedules)
        self.path_work = (self.n or 0) * (
            sum(self.shapes["c" if equation == "state" else "d"][0] * len(definition["columns"])
                for equation, definition in definitions.items())
            + len(schedules) * (m**3 + p**3)
        )
        if self.path_work > 50_000_000:
            raise AnalysisError("system_budget", "Known equation paths exceed the supported operation budget.")
        self._plan("state-space equation paths", {
            "static_values_and_validation": static_cells * 64,
            "original_and_sorted_schedules": ((self.source_n or 0) + (self.n or 0)) * schedule_cells * 8,
            "exogenous_columns_offsets_and_copies": (self.n or 0) * (
                3 * len(self.exogenous_columns) + 4 * (m + p)
            ) * 8 if definitions else 0,
        })
        self.base = {
            key: _array(record.get(key), self.shapes[key], key) for key in ("Z", "T", "Q", "H")
        }
        self.base.update(
            a0=_array(record.get("a0", [0.0] * m), (m,), "a0"),
            P0=_array(record.get("P0", torch.eye(m).tolist()), (m, m), "P0"),
            c=_array(record.get("c", [0.0] * m), (m,), "c"),
            d=_array(record.get("d", [0.0] * p), (p,), "d"),
        )
        self.parameters = deepcopy(parameters)
        used, names, start = set(), set(), []
        for item in self.parameters:
            if not isinstance(item, dict) or set(item) - {"name", "matrix", "row", "col", "transform"}:
                raise AnalysisError("invalid_parameter", "Free parameter needs name/matrix/row/col/transform fields.")
            key, row, col = item.get("matrix"), item.get("row"), item.get("col")
            name, transform = item.get("name"), item.get("transform", "identity")
            if not isinstance(key, str) or key not in self.base or not isinstance(name, str) or not name or name in names:
                raise AnalysisError("invalid_parameter", "Parameter names must be unique and matrices recognized.")
            if key in schedules:
                raise AnalysisError(
                    "invalid_parameter", "Free parameters cannot occupy an explicitly scheduled matrix."
                )
            shape = self.base[key].shape
            if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < shape[0]:
                raise AnalysisError("invalid_parameter", "Parameter row is outside matrix dimensions.")
            if len(shape) == 2:
                if isinstance(col, bool) or not isinstance(col, int) or not 0 <= col < shape[1]:
                    raise AnalysisError("invalid_parameter", "Matrix parameter needs a valid col.")
            elif col is not None:
                raise AnalysisError("invalid_parameter", "Vector parameter must omit col.")
            target = ((key, min(row, col), max(row, col)) if key in {"Q", "H", "P0"} else (key, row, col))
            if target in used or (self.initialization == "stationary" and key in {"a0", "P0"}):
                raise AnalysisError("invalid_parameter", "Duplicate parameter cells or free stationary prior are not identified.")
            value = float(self.base[key][row] if col is None else self.base[key][row, col])
            if transform == "positive" and value > 0:
                value = math.log(value)
            elif transform == "unit" and -1 < value < 1:
                value = math.atanh(value)
            elif transform != "identity":
                raise AnalysisError("invalid_parameter", "Transform identity/positive/unit must match its starting value.")
            start.append(value)
            names.add(name)
            used.add(target)
        self.start = torch.tensor(start, dtype=torch.float64)
        self.names = [item["name"] for item in self.parameters]
        self.lyapunov_bytes = m**4 * 8 * (12 + 4 * len(start)) if self.initialization == "stationary" else 0
        self.lyapunov_work = m**6 * max(1, len(start)) ** 2 if self.initialization == "stationary" else 0
        if self.lyapunov_work > 50_000_000:
            raise AnalysisError("system_budget", "Stationary Lyapunov solve/information geometry exceeds the supported operation budget.")
        if self.lyapunov_bytes:
            self._plan("stationary system initialization", {
                "Lyapunov_operator_factors_and_derivatives": self.lyapunov_bytes,
            })
        self.schedules = {}
        original_schedules = {}
        positions = torch.tensor(frame.positions, dtype=torch.int64) if frame is not None else None
        for key, raw_path in schedules.items():
            path = _array(raw_path, (self.source_n, *self.shapes[key]), f"schedule {key}")
            original_schedules[key] = path.tolist()
            self.schedules[key] = path.index_select(0, positions) if positions is not None else path
        self.exogenous = {
            equation: {
                "columns": list(definition["columns"]),
                "coefficients": _array(
                    definition["coefficients"],
                    (m if equation == "state" else p, len(definition["columns"])),
                    f"{equation} coefficients",
                ),
            }
            for equation, definition in definitions.items()
        }
        self.offsets = {}
        if frame is not None:
            for equation, definition in self.exogenous.items():
                if any(column not in frame.original for column in definition["columns"]):
                    raise AnalysisError("missing_columns", "The sample frame must include all declared exogenous columns.")
                x = torch.stack([frame.numeric(column) for column in definition["columns"]], dim=1)
                offset = x @ definition["coefficients"].T
                if not bool(torch.isfinite(offset).all()):
                    raise AnalysisError("non_finite_exogenous", "A fixed exogenous equation offset overflowed.")
                self.offsets["c" if equation == "state" else "d"] = offset
        self.original_record = {
            **{key: self.base[key].tolist() for key in record if key in self.base},
            **({"initialization": self.initialization} if "initialization" in record else {}),
            **({"parameters": deepcopy(self.parameters)} if "parameters" in record else {}),
            **({"schedules": original_schedules} if "schedules" in record else {}),
            **({"exogenous": {
                equation: {"columns": definition["columns"], "coefficients": definition["coefficients"].tolist()}
                for equation, definition in self.exogenous.items()
            }} if "exogenous" in record else {}),
        }
        self.record = self.original_record
        self._validate_paths(self.schedules)
        if self.exogenous and frame is None:
            self.fixed_values(self.start)
        else:
            self.values(self.start)

    def _plan(self, operation, buffers):
        if self.frame is not None:
            self.frame.workspace_plan(operation, buffers)
        else:
            from openecon.resources import plan_workspace
            plan_workspace(operation, buffers)

    def physical(self, theta):
        return torch.stack([
            torch.exp(theta[i]) if item.get("transform") == "positive"
            else torch.tanh(theta[i]) if item.get("transform") == "unit" else theta[i]
            for i, item in enumerate(self.parameters)
        ]) if len(theta) else theta

    def fixed_values(self, theta):
        """Fitted static templates, before replacing schedules or adding exog."""
        if theta.ndim != 1 or len(theta) != len(self.start) or not bool(torch.isfinite(theta).all()):
            raise AnalysisError("invalid_parameter", "System parameters must be finite and match the declared order.")
        values = {key: value.clone() for key, value in self.base.items()}
        physical = self.physical(theta)
        for i, item in enumerate(self.parameters):
            key, row, col = item["matrix"], item["row"], item.get("col")
            if col is None:
                values[key][row] = physical[i]
            else:
                values[key][row, col] = physical[i]
                if key in {"Q", "H", "P0"}:
                    values[key][col, row] = physical[i]
        if any(not bool(torch.isfinite(value).all()) for value in values.values()):
            raise AnalysisError("invalid_parameter", "Transformed system parameters overflowed.")
        for key in ("Q", "H", "P0"):
            pd_check(values[key], key, semidefinite=key != "H")
        T = values["T"]
        if "T" not in self.scheduled_keys:
            radius = float(torch.linalg.eigvals(T.detach()).abs().max())
            if not math.isfinite(radius) or radius > 1 + 1e-12 or (
                self.initialization == "stationary" and radius >= 1 - 1e-10
            ):
                raise AnalysisError("unstable_system", "T must have spectral radius <=1 (strictly<1 for stationary initialization).")
        if not set(self.scheduled_keys) & {"T", "Z"}:
            observation = torch.cat([values["Z"] @ torch.linalg.matrix_power(T, i) for i in range(self.m)])
            if int(torch.linalg.matrix_rank(observation.detach())) < self.m:
                raise AnalysisError("unidentified_system", "The state system is not observable; fix or remove unidentified states.")
        if self.initialization == "stationary":
            identity = torch.eye(self.m, dtype=torch.float64)
            values["a0"] = torch.linalg.solve(identity - T, values["c"])
            operator = torch.eye(self.m**2, dtype=torch.float64) - torch.kron(T.contiguous(), T.contiguous())
            P = torch.linalg.solve(operator, values["Q"].reshape(-1)).reshape(self.m, self.m)
            values["P0"] = (P + P.T) / 2
            pd_check(values["P0"], "stationary P0", semidefinite=True)
        return values

    def _validate_paths(self, paths):
        for key in ("Q", "H"):
            if key in paths:
                value = paths[key]
                if not torch.allclose(value, value.transpose(-1, -2), atol=1e-12, rtol=1e-12):
                    raise AnalysisError("invalid_system", f"Every scheduled {key} must be symmetric.")
                eigen = torch.linalg.eigvalsh(value.detach())
                if float(eigen.min()) < -1e-12 if key == "Q" else float(eigen.min()) <= 0:
                    raise AnalysisError(
                        "invalid_covariance",
                        f"Every scheduled {key} must be {'positive semidefinite' if key == 'Q' else 'positive definite'}; no projection is applied.",
                    )

    def values(self, theta):
        values = self.fixed_values(theta)
        if not self.has_paths:
            return values
        if self.n is None or (self.exogenous and len(self.offsets) != len(self.exogenous)):
            raise AnalysisError("missing_exog", "Training equation paths require the original exogenous sample frame.")
        for key in PATH_KEYS:
            values[key] = self.schedules.get(key, values[key].expand(self.n, *self.shapes[key]))
            if key in self.offsets:
                values[key] = values[key] + self.offsets[key]
        return values

    def future(self, steps, *, schedules=None, exog=None, theta=None):
        """Expand exact future inputs; never reuse the last fitted path value."""
        horizon(steps)
        schedules = {} if schedules is None else schedules
        if not isinstance(schedules, dict) or set(schedules) != set(self.scheduled_keys):
            raise AnalysisError(
                "invalid_schedule", "Future schedules must supply exactly every fitted scheduled key."
            )
        for key, values in schedules.items():
            _shape(values, (steps, *self.shapes[key]), f"future {key}")
        path_work = steps * (
            sum(self.shapes[key][0] ** 3 for key in schedules if key in {"Q", "H"})
            + sum((self.m if equation == "state" else self.p) * len(definition["columns"])
                  for equation, definition in self.exogenous.items())
        )
        if path_work > 50_000_000:
            raise AnalysisError("system_budget", "Future equation paths exceed the supported operation budget.")
        self._plan("future state-space equation paths", {
            "future_matrix_paths_and_validation": steps * sum(math.prod(self.shapes[key]) for key in PATH_KEYS) * 24,
            "future_exogenous_columns_and_offsets": steps * (3 * len(self.exogenous_columns) + 4 * (self.m + self.p)) * 8,
            "retained_training_paths": (self.n or 0) * sum(math.prod(self.shapes[key]) for key in self.scheduled_keys) * 8,
        })
        values = self.fixed_values(self.start if theta is None else theta)
        for key in PATH_KEYS:
            values[key] = (_array(schedules[key], (steps, *self.shapes[key]), f"future {key}")
                           if key in schedules else values[key].expand(steps, *self.shapes[key]))
        if self.exogenous:
            if exog is None:
                raise AnalysisError("missing_exog", "Forecasting requires every declared future exogenous path.")
            from openecon.analysis import _coerce_frame, _numeric
            if isinstance(exog, Mapping):
                if set(exog) != set(self.exogenous_columns):
                    raise AnalysisError("invalid_exogenous", "Future exog must have exactly the fitted columns.")
                for column in self.exogenous_columns:
                    try:
                        count = len(exog[column])
                    except TypeError as exc:
                        raise AnalysisError("invalid_exogenous", "Future exog columns must be complete forecast-length sequences.") from exc
                    if count != steps:
                        raise AnalysisError("invalid_exogenous", "Future exog must contain one row per forecast period.")
            elif isinstance(exog, (list, tuple)):
                if len(exog) != steps or any(
                    not isinstance(row, Mapping) or set(row) != set(self.exogenous_columns) for row in exog
                ):
                    raise AnalysisError("invalid_exogenous", "Future exog rows must have exactly the fitted columns and horizon.")
            source = _coerce_frame(exog)
            if source.columns.has_duplicates or set(source.columns) != set(self.exogenous_columns) or len(source) != steps:
                raise AnalysisError("invalid_exogenous", "Future exog needs exactly the fitted columns and one row per forecast period.")
            numeric = {column: _numeric(source[column], column) for column in self.exogenous_columns}
            for equation, definition in self.exogenous.items():
                x = torch.stack([numeric[column] for column in definition["columns"]], dim=1)
                key = "c" if equation == "state" else "d"
                values[key] = values[key] + x @ definition["coefficients"].T
                if not bool(torch.isfinite(values[key]).all()):
                    raise AnalysisError("non_finite_exogenous", "A future exogenous equation offset overflowed.")
        elif exog is not None:
            if not isinstance(exog, Mapping) or exog:
                raise AnalysisError("invalid_exogenous", "This system has no declared exogenous equations.")
        self._validate_paths({key: values[key] for key in schedules if key in {"Q", "H"}})
        return values


def future_values(record, physical_base, steps, *, schedules=None, exog=None):
    """Restore future equations from fitted static values and saved contracts.

    Historical schedule values and training regressors are unnecessary. Dummy
    one-period paths retain only the names of scheduled matrices so the same
    strict future expansion validates every required key and covariance.
    """
    if not isinstance(record, dict) or set(record) - _SYSTEM_KEYS:
        raise AnalysisError("invalid_system", "Unknown or missing saved state-space template fields.")
    if not isinstance(physical_base, dict) or set(physical_base) != {*PATH_KEYS, "a0", "P0"}:
        raise AnalysisError("invalid_system", "Saved fitted static values must contain T/Z/Q/H/c/d/a0/P0.")
    scheduled = record.get("schedules", {})
    if not isinstance(scheduled, dict) or set(scheduled) - set(PATH_KEYS):
        raise AnalysisError("invalid_schedule", "Only T/Z/Q/H/c/d may have saved schedules.")
    raw_Z = physical_base.get("Z")
    if not isinstance(raw_Z, list) or not 1 <= len(raw_Z) <= 8:
        raise AnalysisError("invalid_system", "Saved Z must identify the measurement dimensions.")
    restored = {
        **physical_base,
        "initialization": record.get("initialization", "known"),
        "parameters": [],
        "schedules": {key: [physical_base[key]] for key in scheduled},
        "exogenous": _exogenous(record),
    }
    return System(restored, len(raw_Z)).future(steps, schedules=schedules, exog=exog)
