"""Eight proper-prior state-space stages on synthetic, fully declared calendars."""

from copy import deepcopy
import json
import math

import pandas as pd
import openecon as oe


def fixture(*, schedules=False, exogenous=False, shuffle=False, missing=True):
    n = 7
    data = {
        "period": list(range(n)),
        "y": [0.4, None, None, 0.2, -0.5, None, 0.3],
        "other": [-0.2, 0.6, None, None, 0.8, -0.1, None],
        "x": [-0.8, 0.3, 0.7, -0.2, 1.1, 0.5, -0.4],
        "z": [0.2, -0.3, 0.8, 0.1, -0.4, 0.6, 0.9],
    }
    if not missing:
        data["y"] = [v if v is not None else 0.15 - 0.02*t for t, v in enumerate(data["y"])]
        data["other"] = [v if v is not None else -0.1 + 0.03*t for t, v in enumerate(data["other"])]
    system = {
        "Z": [[1.0, 0.2], [-0.3, 0.85]],
        "T": [[0.65, 0.13], [-0.08, 0.42]],
        "Q": [[0.15, 0.04], [0.04, 0.10]],
        "H": [[0.40, 0.13], [0.13, 0.30]],
        "a0": [0.3, -0.2], "P0": [[0.50, 0.10], [0.10, 0.40]],
        "c": [0.02, -0.01], "d": [0.1, -0.05],
    }
    if schedules:
        system["schedules"] = {
            "Z": [[[1.0+0.02*t, 0.2-0.01*t], [-0.3+0.015*t, 0.85+0.02*t]] for t in range(n)],
            "T": [[[0.65-0.02*t, 0.13+0.005*t], [-0.08+0.01*t, 0.42+0.005*t]] for t in range(n)],
            "Q": [[[0.15+0.01*t, 0.04], [0.04, 0.10+0.015*t]] for t in range(n)],
            "H": [[[0.40+0.02*t, 0.13-0.005*t], [0.13-0.005*t, 0.30+0.01*t]] for t in range(n)],
            "c": [[0.02+0.01*t, -0.01-0.005*t] for t in range(n)],
            "d": [[0.1-0.01*t, -0.05+0.02*t] for t in range(n)],
        }
    if exogenous:
        system["exogenous"] = {
            "state": {"columns": ["x", "z"], "coefficients": [[0.2, -0.1], [0.05, 0.15]]},
            "measurement": {"columns": ["z"], "coefficients": [[0.3], [-0.2]]},
        }
    if shuffle:
        permutation = [4, 1, 6, 0, 3, 5, 2]
        data = {key: [values[i] for i in permutation] for key, values in data.items()}
        if schedules:
            system["schedules"] = {
                key: [values[i] for i in permutation] for key, values in system["schedules"].items()
            }
    return data, system


def future_fixture():
    _, system = fixture(schedules=True, exogenous=True)
    return {
        "schedules": {key: deepcopy(value[-3:]) for key, value in system["schedules"].items()},
        "exogenous": {"x": [0.4, -0.7, 0.9], "z": [-0.2, 0.5, 0.3]},
    }


def ml_fixture():
    data = {
        "period": list(range(12)),
        "y": [1.3, 2.1, None, -0.4, 1.5, 3.2, None, 0.8, 1.9, 0.2, 2.7, 1.2],
    }
    system = {
        "T": [[0.0]], "Q": [[0.0]], "P0": [[0.0]], "a0": [0.0],
        "Z": [[1.0]], "c": [0.2], "d": [0.5], "H": [[1.0]],
        "schedules": {"Z": [[[z]] for z in [1.0, 1.2, 0.7, 1.5, 0.9, 1.1, 1.8, 0.6, 1.3, 1.7, 0.8, 1.4]]},
        "parameters": [
            {"name": "mean", "matrix": "d", "row": 0},
            {"name": "state_intercept", "matrix": "c", "row": 0},
            {"name": "variance", "matrix": "H", "row": 0, "col": 0, "transform": "positive"},
        ],
    }
    return data, system


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_value(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def table_state(table):
    return json_value({"index": table.index.tolist(), "columns": table.columns.tolist(),
                       "data": table.values.tolist(), "attrs": table.attrs})


def run():
    inputs = {}
    models = {}
    for name, options in [("masked", {}), ("scheduled", {"schedules": True, "shuffle": True}),
                          ("exogenous", {"schedules": True, "exogenous": True, "shuffle": True})]:
        data, system = fixture(**options)
        inputs[name] = {"data": data, "system": system, "responses": ["y", "other"], "time": "period"}
        models[name] = oe.sspace(data=pd.DataFrame(data), y="y", responses=["other"],
                                 time="period", system=system, missing="mask")
    data, system = ml_fixture()
    inputs["ml"] = {"data": data, "system": system, "responses": ["y"], "time": "period"}
    models["ml"] = oe.sspace(data=pd.DataFrame(data), y="y", time="period", system=system,
                             missing="mask", tolerance=1e-9)
    saved = {name: model.model_dump(mode="json") for name, model in models.items()}
    restored = {name: oe.ResultBundle.model_validate_json(json.dumps(value)) for name, value in saved.items()}
    tables = {}
    for name in ("masked", "scheduled", "exogenous"):
        case = inputs[name]
        tables[name] = oe.sspace_filter(data=pd.DataFrame(case["data"]), y="y", responses=["other"],
                                       time="period", system=case["system"], missing="mask")
    tables["smooth"] = oe.sspace_smooth(restored["masked"])
    tables["autocov"] = oe.sspace_autocov(restored["masked"].model_dump_json())
    tables["disturbances"] = oe.sspace_disturbances(restored["masked"].model_dump(mode="json"))
    future = future_fixture()
    tables["forecast"] = oe.forecast(restored["exogenous"], 3, future=future)
    tables["ml"] = pd.DataFrame([coefficient.model_dump() for coefficient in restored["ml"].coefficients])
    tables["ml"].attrs.update(source_result_json=restored["ml"].model_dump_json(),
                             covariance_matrix=restored["ml"].covariance_matrix)
    inputs["forecast"] = {"model": "exogenous", "future": future, "steps": 3}
    outputs = {name: table_state(table) for name, table in tables.items()}
    json.dumps({"states": saved, "poststates": outputs, "oracle_inputs": inputs}, allow_nan=False)
    show = globals().get("display", print)
    for name in ("masked", "smooth", "autocov", "disturbances", "scheduled", "exogenous", "forecast", "ml"):
        assert oe.to_latex(tables[name])
        show(tables[name])
    print("SSPACE_SCHEDULED_RECEIPT:" + json.dumps({
        "stages": 8, "proper_prior": True, "measurement_missing": "mask", "saved_results": 4,
        "conditional_forecast": True, "stata_parity_validated": False,
    }))
    return saved, outputs, inputs


if __name__ == "__main__":
    states, poststates, oracle_inputs = run()
    frame = pd.DataFrame(oracle_inputs["masked"]["data"])
    mlframe = pd.DataFrame(oracle_inputs["ml"]["data"])
