"""Independent NumPy/SciPy oracle for the actual installed synthetic receipt.

Development only. Reads saved envelopes; never imports OpenEconometrics or
executes anything in the installed application. No SciPy enters the runtime.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import norm


def weights(d, size):
    value = [1.]
    for lag in range(1, size):
        value.append(value[-1] * (lag - 1 - d) / lag)
    return np.array(value)


def verify(path):
    state = json.loads(path.read_text())
    difference = state["difference"]
    w = weights(difference["attrs"]["d"], difference["attrs"]["terms"])
    np.testing.assert_allclose(w, difference["attrs"]["filter_weights"], atol=1e-15)
    filtered = np.array([row["difference"] for row in difference["table"]["data"]])
    y = lfilter([1.], w, filtered)
    np.testing.assert_allclose(np.convolve(y, w)[:len(y)], filtered, atol=2e-13)
    diagnostic = state["gph"]
    m = diagnostic["attrs"]["bandwidth"]
    frequencies = 2 * math.pi * np.arange(1, m + 1) / len(y)
    periodogram = np.abs(np.fft.rfft(y - y.mean())[1:m + 1]) ** 2 / (2 * math.pi * len(y))
    x = np.column_stack((np.ones(m), -np.log(4 * np.sin(frequencies / 2) ** 2)))
    beta = np.linalg.lstsq(x, np.log(periodogram), rcond=None)[0]
    gph_covariance = math.pi ** 2 / 6 * np.linalg.inv(x.T @ x)
    estimate = diagnostic["tables"]["estimate"]
    np.testing.assert_allclose([row["estimate"] for row in estimate["table"]["data"]], beta, atol=1e-11)
    np.testing.assert_allclose(estimate["attrs"]["covariance_matrix"], gph_covariance, atol=1e-12)
    for i, row in enumerate(estimate["table"]["data"]):
        se = np.sqrt(gph_covariance[i, i])
        assert abs(row["p_value"] - 2 * norm.sf(abs(beta[i] / se))) < 1e-11
        assert abs(row["ci_low"] - (beta[i] - norm.isf(.025) * se)) < 1e-11
    model = state["model"]
    coefficients = model["coefficients"]
    params = np.array([c["estimate"] for c in coefficients])
    saved = model["extra"]["fractional_state"]
    burn, terms = saved["burn"], saved["terms"]
    assert len(params) == 3 and not saved["ar"] and not saved["ma"]
    def objective(theta):
        mean, d, sigma = theta
        if sigma <= 0 or not -.49 < d < .49:
            return math.inf
        error = np.convolve(y - mean, weights(d, terms))[:len(y)][burn:]
        return .5 * (len(error) * np.log(2 * math.pi * sigma ** 2) + np.dot(error, error) / sigma ** 2)
    oracle = minimize(objective, [y.mean(), .2, y.std()], method="L-BFGS-B",
                      bounds=[(None, None), (-.489, .489), (1e-5, None)],
                      options={"gtol": 1e-8, "ftol": 1e-14})
    assert abs(oracle.fun - objective(params)) < 1e-7
    np.testing.assert_allclose(oracle.x, params, atol=1e-5)
    assert abs(objective(params) + model["metrics"]["log_likelihood"]) < 1e-9
    h = 2e-4
    information = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h, np.eye(3)[j] * h
            information[i, j] = (objective(params + ei + ej) - objective(params + ei - ej)
                                 - objective(params - ei + ej) + objective(params - ei - ej)) / (4 * h * h)
    covariance = np.linalg.inv(information)
    np.testing.assert_allclose(model["covariance_matrix"], covariance, rtol=1e-4, atol=1e-7)
    for i, coefficient in enumerate(coefficients):
        se = math.sqrt(model["covariance_matrix"][i][i])
        assert abs(coefficient["std_error"] - se) < 1e-12
        assert abs(coefficient["p_value"] - 2 * norm.sf(abs(params[i] / se))) < 1e-12
        assert abs(coefficient["ci_low"] - (params[i] - norm.isf(.025) * se)) < 1e-12
    forecast = state["forecast"]
    horizon = len(forecast["table"]["data"])
    impulse = np.zeros(horizon)
    impulse[0] = 1
    polynomial = weights(saved["d"], terms)
    psi = lfilter([1.], polynomial, impulse)
    mapping = np.zeros((horizon, horizon))
    for row in range(horizon):
        mapping[row, :row + 1] = psi[:row + 1][::-1]
    forecast_covariance = saved["sigma"] ** 2 * mapping @ mapping.T
    np.testing.assert_allclose(forecast["attrs"]["covariance_matrix"], forecast_covariance, atol=1e-12)
    history = list(saved["history"])
    means = []
    for _ in range(horizon):
        center = -np.dot(polynomial[1:], history[-(terms - 1):][::-1])
        history.append(center)
        means.append(saved["mean"] + center)
    rows = forecast["table"]["data"]
    np.testing.assert_allclose([r["forecast"] for r in rows], means, atol=1e-12)
    for i, row in enumerate(rows):
        se = math.sqrt(forecast_covariance[i, i])
        assert abs(row["std_error"] - se) < 1e-12
        assert abs(row["ci_low"] - means[i] + norm.isf(.025) * se) < 1e-12
    return {"status": "passed", "scope": "actual installed four-stage synthetic run; independent development oracle",
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(y),
            "gph_estimate": beta.tolist(), "gph_covariance": gph_covariance.tolist(),
            "arfima_parameters": params.tolist(), "arfima_covariance": covariance.tolist(),
            "likelihood": -objective(params), "forecast_covariance": forecast_covariance.tolist(),
            "observed_information_max_abs_difference": float(np.max(np.abs(covariance - model["covariance_matrix"]))),
            "source_oracle_only": True, "vendor_parity": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.path)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "rows": result["rows"], "stages": 4}))
