"""Development-only dense Gaussian conditioning oracle; no production imports.

Construct the joint prior of the initial state, every independent process shock
and every independent measurement shock. Conditioning observed measurement cells
gives all state/noise moments directly, independently of Kalman/RTS recursions.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import stats


FILTER_KEYS = (
    "prior_mean", "prior_covariance", "predicted", "innovation_covariance", "filtered",
    "filtered_covariance", "next_mean", "next_covariance", "log_likelihood_contributions",
)
SMOOTH_KEYS = ("smoothed", "smoothed_covariance", "smoothed_next_mean", "smoothed_next_covariance")
DISTURBANCE_KEYS = (
    "process_disturbance", "process_disturbance_covariance", "measurement_disturbance",
    "measurement_disturbance_covariance", "measurement_state_covariance", "process_state_covariance",
    "process_measurement_covariance", "joint_disturbance_covariance",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordered_case(case, fitted=None):
    data, system = case["data"], deepcopy(case["system"])
    n = len(data[case["responses"][0]])
    order = np.argsort(np.asarray(data[case["time"]]), kind="stable") if case.get("time") else np.arange(n)
    y = np.column_stack([np.asarray(data[name], dtype=float)[order] for name in case["responses"]])
    if fitted is not None:
        estimates = {coefficient["term"]: coefficient["estimate"] for coefficient in fitted["coefficients"]}
        for parameter in system.get("parameters", []):
            key, row, col = parameter["matrix"], parameter["row"], parameter.get("col")
            if col is None:
                system[key][row] = estimates[parameter["name"]]
            else:
                system[key][row][col] = estimates[parameter["name"]]
                if key in {"P0", "Q", "H"}:
                    system[key][col][row] = estimates[parameter["name"]]
    path = {}
    for key in ("Z", "T", "Q", "H", "c", "d"):
        if key in system.get("schedules", {}):
            path[key] = np.asarray(system["schedules"][key], dtype=float)[order]
        else:
            path[key] = np.repeat(np.asarray(system[key], dtype=float)[None], n, axis=0)
    for role, key in (("state", "c"), ("measurement", "d")):
        exog = system.get("exogenous", {}).get(role)
        if exog:
            x = np.column_stack([np.asarray(data[name], dtype=float)[order] for name in exog["columns"]])
            path[key] = path[key] + x @ np.asarray(exog["coefficients"], dtype=float).T
    path.update(a0=np.asarray(system["a0"], dtype=float), P0=np.asarray(system["P0"], dtype=float))
    return y, path, order


def condition(prior, measurement_map, residual, observed):
    dimension = prior.shape[0]
    if not len(observed):
        return np.zeros(dimension), prior.copy(), 0.0
    selected = measurement_map[observed]
    covariance = selected @ prior @ selected.T
    difference = residual[observed]
    sign, determinant = np.linalg.slogdet(covariance)
    require(sign > 0, "Oracle observed joint covariance must be positive definite.")
    cross = prior @ selected.T
    mean = cross @ np.linalg.solve(covariance, difference)
    posterior = prior - cross @ np.linalg.solve(covariance, cross.T)
    likelihood = -0.5 * (len(observed)*np.log(2*np.pi) + determinant + difference @ np.linalg.solve(covariance, difference))
    return mean, (posterior+posterior.T)/2, float(likelihood)


def joint_reference(y, path):
    y = np.asarray(y, dtype=float)
    n, p = y.shape
    m = len(path["a0"])
    dimension = m+n*m+n*p
    prior = np.zeros((dimension, dimension))
    prior[:m, :m] = path["P0"]
    state_maps, state_means = [np.eye(dimension)[:m]], [path["a0"]]
    for t in range(n):
        u = slice(m+t*m, m+(t+1)*m)
        e = slice(m+n*m+t*p, m+n*m+(t+1)*p)
        prior[u, u], prior[e, e] = path["Q"][t], path["H"][t]
        next_map = path["T"][t] @ state_maps[-1]
        next_map[:, u] += np.eye(m)
        state_maps.append(next_map)
        state_means.append(path["T"][t] @ state_means[-1] + path["c"][t])
    state_maps, state_means = np.asarray(state_maps), np.asarray(state_means)
    observation_maps, observation_means = [], []
    for t in range(n):
        mapping = path["Z"][t] @ state_maps[t]
        mapping[:, m+n*m+t*p:m+n*m+(t+1)*p] += np.eye(p)
        observation_maps.append(mapping)
        observation_means.append(path["Z"][t] @ state_means[t] + path["d"][t])
    observation_maps = np.vstack(observation_maps)
    observation_means = np.asarray(observation_means)
    mask = np.isfinite(y)
    residual = (y-observation_means).reshape(-1)
    observed = np.flatnonzero(mask.reshape(-1))
    mean, posterior, likelihood = condition(prior, observation_maps, residual, observed)
    state_post_mean = state_means + np.einsum("tij,j->ti", state_maps, mean)
    state_post_cov = np.asarray([mapping @ posterior @ mapping.T for mapping in state_maps])
    output = {
        "smoothed": state_post_mean[:-1], "smoothed_covariance": state_post_cov[:-1],
        "smoothed_next_mean": state_post_mean[-1], "smoothed_next_covariance": state_post_cov[-1],
        "lag_one_covariance": np.asarray([state_maps[t+1] @ posterior @ state_maps[t].T for t in range(n-1)]),
        "log_likelihood": likelihood, "observed_mask": mask, "observed_count": mask.sum(1),
    }
    accum = {key: [] for key in FILTER_KEYS if key not in {"next_mean", "next_covariance"}}
    previous_likelihood = 0.0
    for t in range(n):
        before = observed[observed < t*p]
        after = observed[observed < (t+1)*p]
        prior_mean, prior_cov, _ = condition(prior, observation_maps, residual, before)
        filtered_mean, filtered_cov, prefix_likelihood = condition(prior, observation_maps, residual, after)
        accum["prior_mean"].append(state_means[t] + state_maps[t] @ prior_mean)
        accum["prior_covariance"].append(state_maps[t] @ prior_cov @ state_maps[t].T)
        mapping = observation_maps[t*p:(t+1)*p]
        accum["predicted"].append(observation_means[t]+mapping @ prior_mean)
        accum["innovation_covariance"].append(mapping @ prior_cov @ mapping.T)
        accum["filtered"].append(state_means[t]+state_maps[t] @ filtered_mean)
        accum["filtered_covariance"].append(state_maps[t] @ filtered_cov @ state_maps[t].T)
        accum["log_likelihood_contributions"].append(prefix_likelihood-previous_likelihood)
        previous_likelihood = prefix_likelihood
    output.update({key: np.asarray(value) for key, value in accum.items()})
    output["next_mean"], output["next_covariance"] = state_post_mean[-1], state_post_cov[-1]
    for key in DISTURBANCE_KEYS:
        output[key] = []
    for t in range(n):
        u = np.arange(m+t*m, m+(t+1)*m)
        e = np.arange(m+n*m+t*p, m+n*m+(t+1)*p)
        output["process_disturbance"].append(mean[u])
        output["process_disturbance_covariance"].append(posterior[np.ix_(u, u)])
        output["measurement_disturbance"].append(mean[e])
        output["measurement_disturbance_covariance"].append(posterior[np.ix_(e, e)])
        output["measurement_state_covariance"].append(posterior[e] @ state_maps[t].T)
        output["process_state_covariance"].append(posterior[u] @ state_maps[t].T)
        output["process_measurement_covariance"].append(posterior[np.ix_(u, e)])
        joint = np.r_[u, e]
        output["joint_disturbance_covariance"].append(posterior[np.ix_(joint, joint)])
    output.update({key: np.asarray(output[key]) for key in DISTURBANCE_KEYS})
    return output


def future_path(system, future, steps):
    values = {}
    for key in ("Z", "T", "Q", "H", "c", "d"):
        if key in future.get("schedules", {}):
            values[key] = np.asarray(future["schedules"][key], dtype=float).copy()
        else:
            values[key] = np.repeat(np.asarray(system[key], dtype=float)[None], steps, axis=0)
    for role, key in (("state", "c"), ("measurement", "d")):
        exog = system.get("exogenous", {}).get(role)
        if exog:
            x = np.column_stack([np.asarray(future["exogenous"][name], dtype=float) for name in exog["columns"]])
            values[key] += x @ np.asarray(exog["coefficients"], dtype=float).T
    return values


def forecast_reference(state, system, future, steps):
    a, covariance = np.asarray(state["next_mean"]), np.asarray(state["next_covariance"])
    result = {key: [] for key in ("state_mean", "state_covariance", "measurement_mean", "measurement_covariance")}
    path = future_path(system, future, steps)
    for t in range(steps):
        values = {key: value[t] for key, value in path.items()}
        result["state_mean"].append(a)
        result["state_covariance"].append(covariance)
        result["measurement_mean"].append(values["Z"] @ a + values["d"])
        result["measurement_covariance"].append(values["Z"] @ covariance @ values["Z"].T + values["H"])
        a = values["T"] @ a + values["c"]
        covariance = values["T"] @ covariance @ values["T"].T + values["Q"]
    return {key: np.asarray(value) for key, value in result.items()}


def ml_reference(case):
    y, path, _ = ordered_case(case)
    design = np.column_stack([np.ones(len(y)), np.r_[0, path["Z"][1:, 0, 0]]])
    selected = np.isfinite(y[:, 0])
    x, response = design[selected], y[selected, 0]
    estimate = np.linalg.solve(x.T @ x, x.T @ response)
    error = response-x @ estimate
    variance = error @ error/len(response)
    covariance = np.zeros((3, 3))
    covariance[:2, :2] = variance*np.linalg.inv(x.T @ x)
    covariance[2, 2] = 2*variance**2/len(response)
    return np.r_[estimate, variance], covariance


def compare(actual, expected, name, *, atol=2e-9, rtol=2e-8):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, f"{name}: complete dimensions differ.")
    require(np.isfinite(actual).all(), f"{name}: native output is not complete finite numeric state.")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
    return float(np.max(np.abs(actual-expected), initial=0.0))


def column(table, name):
    position = table["columns"].index(name)
    return [row[position] for row in table["data"]]


def metadata(table, case, reference, order, *, observed_count=False):
    attrs = table["attrs"]
    periods = [case["data"][case["time"]][int(i)] for i in order]
    require(attrs["periods"] == periods, "Native output calendar periods differ.")
    require(attrs["sample_positions"] == order.tolist(), "Native output physical positions differ.")
    require(attrs["responses"] == case["responses"], "Native output response identity/order differs.")
    require(attrs["observed_mask"] == reference["observed_mask"].tolist(), "Native output cell masks differ.")
    require(all(type(v) is bool for row in attrs["observed_mask"] for v in row), "Native output cell masks must be Boolean.")
    if observed_count:
        require(attrs["observed_count"] == reference["observed_count"].tolist(), "Native observed-cell counts differ.")
    require(attrs["missing_policy"] == "mask", "Native missing policy is not measurement masking.")
    require(attrs["stata_parity_validated"] is False, "Native output must retain the unvalidated vendor-comparison flag.")
    uncertainty = attrs["uncertainty"].lower()
    require("conditional" in uncertainty and "parameter uncertainty excluded" in uncertainty,
            "Native output must disclose conditional, fixed-parameter uncertainty.")


def verify(payload):
    states, tables, inputs = payload["states"], payload["poststates"], payload["oracle_inputs"]
    require(set(states) == {"masked", "scheduled", "exogenous", "ml"}, "Four saved native fits are required.")
    require(len(tables) == 8, "Eight complete native output payloads are required.")
    differences = {}
    references = {}
    for name, result in states.items():
        y, path, order = ordered_case(inputs[name], result)
        reference = joint_reference(y, path)
        references[name] = reference
        require(result["sample_positions"] == order.tolist(), f"{name}: physical row order differs.")
        require(result["nobs"] == len(y), f"{name}: a missing period was deleted.")
        require(result["extra"]["periods"] == [inputs[name]["data"][inputs[name]["time"]][int(i)] for i in order], f"{name}: saved calendar periods differ.")
        require(result["extra"]["observed_mask"] == reference["observed_mask"].tolist(), f"{name}: saved masks differ.")
        require(result["extra"]["observed_count"] == reference["observed_count"].tolist(), f"{name}: saved counts differ.")
        require(result["inference"]["forecast_parameter_uncertainty"] is False, f"{name}: parameter uncertainty scope differs.")
        for key in FILTER_KEYS:
            differences[name+"/"+key] = compare(result["extra"][key], reference[key], name+"/"+key)
        differences[name+"/log_likelihood"] = compare(result["metrics"]["log_likelihood"], reference["log_likelihood"], name+" likelihood")
        if name != "ml":
            native = tables[name]
            metadata(native, inputs[name], reference, order)
            require(column(native, "row") == order.tolist(), name+": displayed physical rows differ.")
            for i in range(reference["filtered"].shape[1]):
                differences[f"{name}/display_state_{i+1}"] = compare(
                    column(native, f"state_{i+1}"), reference["filtered"][:, i], name+" displayed state")
            differences[name+"/display_covariance"] = compare(native["attrs"]["covariance"], reference["filtered_covariance"], name+" displayed full covariance")
    for name, keys in (("smooth", SMOOTH_KEYS), ("autocov", ("lag_one_covariance",)), ("disturbances", DISTURBANCE_KEYS)):
        _, _, masked_order = ordered_case(inputs["masked"])
        metadata(tables[name], inputs["masked"], references["masked"], masked_order, observed_count=True)
        row_order = masked_order[:-1] if name == "autocov" else masked_order
        require(column(tables[name], "row") == row_order.tolist(), name+": displayed physical rows differ.")
        if name == "autocov":
            require(column(tables[name], "next_row") == masked_order[1:].tolist(), "Lag next-row physical alignment differs.")
            require(tables[name]["attrs"]["orientation"] == "Cov(a[t+1],a[t]|all observed Y)", "Lag covariance orientation label differs.")
        for key in keys:
            differences[name+"/"+key] = compare(tables[name]["attrs"][key], references["masked"][key], name+"/"+key)
    m = references["masked"]["smoothed"].shape[1]
    p = references["masked"]["measurement_disturbance"].shape[1]
    for i in range(m):
        differences[f"smooth/display_state_{i+1}"] = compare(column(tables["smooth"], f"state_{i+1}"), references["masked"]["smoothed"][:, i], "displayed smoothed state")
        differences[f"disturbances/display_process_{i+1}"] = compare(column(tables["disturbances"], f"process_{i+1}"), references["masked"]["process_disturbance"][:, i], "displayed process disturbance")
        for j in range(m):
            differences[f"autocov/display_cov_{i+1}_{j+1}"] = compare(column(tables["autocov"], f"cov_{i+1}_{j+1}"), references["masked"]["lag_one_covariance"][:, i, j], "displayed lag covariance orientation")
    for i in range(p):
        differences[f"disturbances/display_measurement_{i+1}"] = compare(column(tables["disturbances"], f"measurement_{i+1}"), references["masked"]["measurement_disturbance"][:, i], "displayed measurement disturbance")
    forecast_input = inputs["forecast"]
    forecast = forecast_reference(references["exogenous"], inputs["exogenous"]["system"], forecast_input["future"], forecast_input["steps"])
    _, _, origin_order = ordered_case(inputs["exogenous"])
    attrs = tables["forecast"]["attrs"]
    require(attrs["origin"] == len(origin_order), "Forecast origin observation count differs.")
    require(attrs["origin_period"] == inputs["exogenous"]["data"][inputs["exogenous"]["time"]][int(origin_order[-1])], "Forecast origin calendar period differs.")
    require(attrs["origin_physical_row"] == int(origin_order[-1]), "Forecast origin physical row differs.")
    require(attrs["responses"] == inputs["exogenous"]["responses"], "Forecast response labels differ.")
    require("conditional" in attrs["uncertainty"].lower() and "parameter uncertainty excluded" in attrs["uncertainty"].lower(), "Forecast conditional uncertainty scope differs.")
    expanded = future_path(inputs["exogenous"]["system"], forecast_input["future"], forecast_input["steps"])
    require(set(attrs["future_paths"]) == set(expanded), "Expanded native future path keys differ.")
    for key, value in expanded.items():
        differences["forecast/expanded_"+key] = compare(attrs["future_paths"][key], value, "expanded future "+key)
    for key, value in forecast.items():
        differences["forecast/"+key] = compare(tables["forecast"]["attrs"][key], value, "forecast/"+key)
    forecast_table = tables["forecast"]
    columns = {name: i for i, name in enumerate(forecast_table["columns"])}
    data = np.asarray(forecast_table["data"], dtype=float)
    point = forecast["measurement_mean"][:, 0]
    se = np.sqrt(forecast["measurement_covariance"][:, 0, 0])
    critical = stats.norm.ppf(0.975)
    for key, value in (("forecast", point), ("std_error", se),
                       ("ci_low", point-critical*se), ("ci_high", point+critical*se)):
        differences["forecast/display_"+key] = compare(data[:, columns[key]], value, "native forecast "+key)
    params, covariance = ml_reference(inputs["ml"])
    terms = [parameter["name"] for parameter in inputs["ml"]["system"]["parameters"]]
    require([coefficient["term"] for coefficient in states["ml"]["coefficients"]] == terms, "Saved ML parameter identity/order differs.")
    require(column(tables["ml"], "term") == terms, "Displayed ML parameter identity/order differs.")
    differences["ml/coefficients"] = compare([c["estimate"] for c in states["ml"]["coefficients"]], params, "ML physical coefficients", atol=2e-6)
    differences["ml/full_covariance"] = compare(states["ml"]["covariance_matrix"], covariance, "ML full physical covariance", atol=2e-6, rtol=2e-5)
    for i, coefficient in enumerate(states["ml"]["coefficients"]):
        se = np.sqrt(covariance[i, i])
        for key, value in (("std_error", se), ("statistic", params[i]/se),
                           ("p_value", 2*stats.norm.sf(abs(params[i]/se))),
                           ("ci_low", params[i]-critical*se), ("ci_high", params[i]+critical*se)):
            differences[f"ml/{i}/{key}"] = compare(coefficient[key], value, f"ML {i} {key}", atol=3e-6, rtol=3e-5)
    for key in ("estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"):
        differences["ml/display_"+key] = compare(column(tables["ml"], key),
            [coefficient[key] for coefficient in states["ml"]["coefficients"]], "displayed ML "+key)
    return {"oracle": "dense joint primitive Gaussian conditioning; NumPy/SciPy only",
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "eight_outputs_verified": True, "four_saved_fits_verified": True,
            "maximum_absolute_differences": differences, "licensed_vendor_comparison": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.states.read_bytes()
    receipt = verify(json.loads(raw))
    receipt["native_state_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"eight_outputs_verified": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
