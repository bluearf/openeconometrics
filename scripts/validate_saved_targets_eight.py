"""Independent development-only algebra and delta checks for eight saved targets.

The protocol is frozen before any model is fitted. NumPy/SciPy are references
only and are never imported by the runtime implementations being accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import traceback
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/saved-targets-eight-2026-10-07"
PROTOCOL = EVIDENCE / "protocol.json"
METHODS = ["sreg", "mmreg", "ivcue", "mixedflex", "sar", "sem", "sac", "sdm"]
COMMON = METHODS[:4]
SPATIAL = METHODS[4:]
SOURCE_FILES = [
    "src/openecon/econometrics/postest/linear_prediction.py",
    "src/openecon/econometrics/postest/prediction.py",
    "src/openecon/econometrics/postest/streaming_prediction.py",
    "src/openecon/econometrics/spatial/prediction.py",
    "src/openecon/econometrics/spatial/weights.py",
    "src/openecon/econometrics/robust/smm.py",
    "src/openecon/econometrics/iv/cue.py",
    "src/openecon/econometrics/mixed/flexible_lmm.py",
    "src/openecon/econometrics/spatial/estimators.py",
    "src/openecon/econometrics/spatial/kernels.py",
    "src/openecon/econometrics/postest/inference.py",
    "src/openecon/econometrics/summary_state.py",
    "src/openecon/econometrics/core.py",
    "src/openecon/models.py",
    "src/openecon/econometrics/postest/index_codec.py",
    "src/openecon/analysis.py",
    "src/openecon/engines/inference.py",
    "src/openecon/engines/distributions.py",
    "src/openecon/dataset.py",
]
SOURCES = {
    "robust_location": {
        "url": "https://cran.r-project.org/web/packages/robustbase/robustbase.pdf",
        "section": "predict.lmrob; confidence estimates describe the saved robust linear target",
        "scope": "Original-author target documentation only; no robustbase execution or estimator equality claim",
    },
    "structural_iv": {
        "url": "https://www.stata.com/manuals/rivregresspostestimation.pdf",
        "section": "predict, xb; structural fitted index from endogenous and exogenous values",
        "scope": "Target definition only; CUE prediction inference remains strong-identification asymptotic",
    },
    "gaussian_population": {
        "url": "https://www.stata.com/manuals/memixedpostestimation.pdf",
        "section": "predict, xb and stdp; fixed portion at the theoretical mean of Gaussian random effects",
        "scope": "Population target, excluding realized group effects, residuals and BLUPs",
    },
    "spatial_reduced_form": {
        "url": "https://www.stata.com/manuals/spspregresspostestimation.pdf",
        "section": "Methods and formulas, Reduced-form mean, equation (3); spatial error filters do not affect this mean",
        "scope": "Complete fixed/exogenous keyed W and X; parameter delta covariance, no innovation interval",
    },
}


def definition():
    return {
        "schema": "openecon.saved-targets-eight.protocol.v1",
        "methods": METHODS,
        "cell_count": 40,
        "cells": {
            "common": ["numeric_mean_and_gradient", "full_parameter_covariance_and_order",
                       "derivatives_and_margins", "missing_index_dataset_persistence", "unsupported_and_state_refusals"],
            "spatial": ["numeric_mean_gradient_joint_covariance", "shuffled_complete_graph_and_changed_X",
                        "full_parameter_covariance_and_order", "summary_persistence_and_latex", "unsupported_and_state_refusals"],
        },
        "fixture": {
            "robust_seed": 22, "robust_n": 95, "robust_search_seed": 41,
            "robust_starts": 250, "robust_max_iterations": 400, "robust_tolerance": 1e-9,
            "iv_seed": 6419, "iv_n": 300, "iv_center": False,
            "mixed_seed": 18, "mixed_structure": "three nested Gaussian intercept factors; six by two by two groups, four rows per lowest group",
            "spatial_seed": 281, "spatial_n": 24,
            "spatial_edges": "directed circular offsets (-1,1,5), weights (1,2,.3), row normalization",
            "spatial_dgp": "solve(I-.35W, 1+.8x-.4z+solve(I-.2W, independent standard-normal innovation))",
            "sac_error_graph": "rotate W columns by two, zero diagonal and renormalize; independently constructed W2",
            "query": "fixed nine rows for common targets; all effective graph units for spatial targets",
            "covariance_stress": "PSD dense deterministic L L' with nonzero coefficient/spatial cross blocks, jointly reversed parameter and covariance order",
        },
        "alphas": [.05, .10],
        "finite_difference": {"formula": "five-point central derivative of independently evaluated NumPy response",
                              "step": "2e-4*(1+abs(parameter))", "absolute_tolerance": 2e-7, "relative_tolerance": 3e-6},
        "gates": {"response_atol": 2e-10, "response_rtol": 2e-10,
                  "covariance_atol": 2e-9, "covariance_rtol": 2e-8,
                  "inference_atol": 2e-9, "inference_rtol": 2e-8,
                  "state": "all complete numerical tables and metadata survive JSON restore; no refitting; LaTeX nonempty",
                  "failure_policy": "Every planned cell retained, including import, fit, API and oracle failures; no replaced fixtures or unreported retries",
                  "source_policy": "Pin complete implementation hashes at start and end; drift fails acceptance; repairs create new receipts with identical protocol"},
        "independence": {"parameters": "Actual native fitted state serialized to JSON and restored before reference evaluation",
                         "design": "NumPy reconstructs fitted predictor/category features in saved parameter order; no runtime encoding helper",
                         "spatial": "NumPy reconstructs dense effective W from saved COO, solves complete fixed graph; no runtime spatial kernels",
                         "covariance": "Independent finite differences and analytic gradients; complete J V J' checked including off-diagonal prediction covariance",
                         "critical_values": "SciPy normal or Student t using the saved native inference convention"},
        "scientific_scope": {
            "sreg_mmreg": "Saved robust location X beta, not unconditional population mean; finite seeded S search remains approximate",
            "ivcue": "Structural X beta at supplied endogenous values; no instrument replay or weak-IV robustness claim",
            "mixedflex": "Population X beta integrating mean-zero Gaussian effects; all nuisance Jacobian columns exactly zero; no BLUP",
            "spatial": "SAR/SAC/SDM A^-1 X_aug beta, SEM X beta; lambda and ln_sigma2 Jacobian columns exactly zero",
            "interval": "Saved-coefficient/delta parameter uncertainty at fixed evaluation rows and W; no observation interval or empirical coverage proof",
            "weights": "These fitting families reject observation weights; extraneous evaluation weight columns do not change unweighted targets",
            "missing": "Common missing='drop' preserves rows/index with NaN; spatial prediction requires complete saved effective graph and complete predictors",
            "not_claimed": ["licensed vendor execution", "blanket parity", "frozen executable", "installed native desktop", "graph sampling uncertainty", "new graph extrapolation", "spatial Dataset streaming", "finite sample nominal delta coverage"],
        },
        "primary_sources": SOURCES,
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(definition(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if PROTOCOL.exists():
        if PROTOCOL.read_text() != encoded:
            raise RuntimeError("Existing frozen protocol differs; gates and cases cannot be revised after outcomes")
    else:
        with PROTOCOL.open("x") as stream:
            stream.write(encoded)
    return digest(PROTOCOL)


def pin_sources():
    return {name: digest(ROOT / name) if (ROOT / name).is_file() else None for name in SOURCE_FILES}


def finite_gradient(function, point):
    columns = []
    for index, value in enumerate(point):
        step = 2e-4 * (1 + abs(value))
        direction = np.eye(len(point))[index] * step
        columns.append((function(point - 2 * direction) - 8 * function(point - direction)
                        + 8 * function(point + direction) - function(point + 2 * direction)) / (12 * step))
    return np.column_stack(columns)


def check(actual, expected, *, gradient=False, covariance=False):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    atol, rtol = (2e-7, 3e-6) if gradient else (2e-9, 2e-8) if covariance else (2e-10, 2e-10)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
    difference = abs(actual - expected)
    finite = difference[np.isfinite(difference)]
    return float(finite.max()) if len(finite) else 0.


def infer(values, covariance, alpha, result, *, spatial=False):
    se = np.sqrt(np.diag(covariance))
    df = result.inference.get("df_inference", result.inference.get("df_resid")) if result.inference.get("use_t") and not spatial else None
    critical = scipy_stats.t.isf(alpha / 2, df) if df is not None else scipy_stats.norm.isf(alpha / 2)
    statistic = values / se
    probability = 2 * (scipy_stats.t.sf(abs(statistic), df) if df is not None else scipy_stats.norm.sf(abs(statistic)))
    return {"std_error": se, "ci_low": values - critical * se, "ci_high": values + critical * se,
            "statistic": statistic, "p_value": probability, "df": df, "critical": float(critical)}


def stored(result):
    encoded = result.model_dump_json()
    return ResultBundle.model_validate_json(encoded)


def covariance_stress(result):
    size = len(result.coefficients)
    lower = np.eye(size) + np.tril(np.full((size, size), .28), -1)
    scales = np.linspace(.025, .10, size)
    covariance = lower @ lower.T * scales[:, None] * scales[None, :]
    order = np.arange(size - 1, -1, -1)
    state = result.model_dump()
    state["coefficients"] = [state["coefficients"][index] for index in order]
    state["covariance_matrix"] = covariance[np.ix_(order, order)].tolist()
    for index, coefficient in enumerate(state["coefficients"]):
        coefficient["std_error"] = float(np.sqrt(state["covariance_matrix"][index][index]))
    return stored(ResultBundle.model_validate(state))


def make_fixture(name):
    if name in {"sreg", "mmreg"}:
        rng = np.random.default_rng(22)
        x = rng.normal(size=95)
        source = pd.DataFrame({"x": x, "y": 1 + 2 * x + .5 * rng.normal(size=95),
                               "g": np.resize(["A", "B", "C"], 95), "cluster": np.arange(95) % 9})
        result = getattr(oe, name)(data=source, y="y", x=["x", "g"], categorical=["g"],
                                  covariance="cluster", cluster="cluster", missing="drop",
                                  starts=250, seed=41, max_iterations=400, tolerance=1e-9)
    elif name == "ivcue":
        rng, n = np.random.default_rng(6419), 300
        z, c, v = rng.normal(size=(n, 3)), rng.normal(size=n), rng.normal(size=n)
        x = z @ [1, .7, .2] + .2 * c + v
        y = 1 + 2 * x - .5 * c + .7 * v + (1 + .6 * abs(z[:, 0])) * rng.normal(size=n)
        source = pd.DataFrame(dict(y=y, x=x, c=c, z1=z[:, 0], z2=z[:, 1], z3=z[:, 2], cluster=np.repeat(np.arange(60), 5)))
        result = oe.ivcue(data=source, y="y", x=["c"], endog=["x"], instruments=["z1", "z2", "z3"],
                          center=False, cluster="cluster", missing="drop")
    elif name == "mixedflex":
        rng = np.random.default_rng(18)
        top = np.repeat(np.arange(6), 16)
        mid = np.tile(np.repeat(np.arange(2), 8), 6)
        low = np.tile(np.repeat(np.arange(2), 4), 12)
        x = rng.normal(size=len(top))
        y = (2 + .4 * x + rng.normal(size=6)[top] + rng.normal(size=12)[top * 2 + mid]
             + rng.normal(size=24)[top * 4 + mid * 2 + low] + rng.normal(size=len(top)) * .4)
        source = pd.DataFrame(dict(y=y, x=x, top=top, mid=mid, low=low))
        result = oe.mixedflex(data=source, y="y", x=["x"], group=["top", "mid", "low"], grouping="nested", missing="drop")
    else:
        n, rng = 24, np.random.default_rng(281)
        keys = [f"unit-{i}" for i in range(n)]
        edges = [(keys[i], keys[j], value) for i in range(n)
                 for j, value in (((i - 1) % n, 1.), ((i + 1) % n, 2.), ((i + 5) % n, .3))]
        w = np.zeros((n, n))
        for origin, destination, value in edges:
            w[keys.index(origin), keys.index(destination)] = value
        w /= w.sum(axis=1)[:, None]
        x, z = rng.normal(size=n), rng.normal(size=n)
        y = np.linalg.solve(np.eye(n) - .35 * w,
                            1 + .8 * x - .4 * z + np.linalg.solve(np.eye(n) - .2 * w, rng.normal(size=n)))
        source = pd.DataFrame({"id": keys, "y": y, "x": x, "z": z})
        weights = oe.spatial_weights(keys, edges)
        options = {}
        if name == "sac":
            we = np.roll(w, 2, axis=1)
            np.fill_diagonal(we, 0)
            we /= we.sum(axis=1)[:, None]
            options["error_weights"] = oe.spatial_weights(keys, [(keys[i], keys[j], float(we[i, j]))
                                                    for i, j in zip(*np.nonzero(we), strict=True)])
        result = getattr(oe, name)(source, "y", ["x", "z"], key="id", spatial_weights=weights, **options)
    return source, stored(result)


def common_query(name):
    return pd.DataFrame({"x": [-1.5, -.9, -.4, 0., .2, .6, 1., 1.4, 1.8],
                         "c": [.1, -.2, -.4, .5, 1., .3, .7, -.5, .2],
                         "g": ["C", "B", "A", "A", "B", "C", "C", "B", "A"],
                         "w": [0., .5, 3., 4., 8., .2, 1., 10., .8]},
                        index=pd.Index(["repeat", "repeat", "other", "third", "same", "same", "end", "end", "last"], name="observation"))


def common_design(result, query):
    columns = []
    categories = result.provenance.get("categorical_encoding", {})
    for coefficient in result.coefficients:
        term = coefficient.term
        if term == "Intercept":
            column = np.ones(len(query))
        elif term in result.spec.predictors or term in result.spec.columns.get("endogenous", []):
            column = query[term].to_numpy(float)
        elif term.startswith("/"):
            column = np.zeros(len(query))
        else:
            matches = [(name, level) for name, encoding in categories.items() for level in encoding["levels"][1:]
                       if term == f"{name}[{level}]"]
            if len(matches) != 1:
                raise AssertionError(f"Independent design cannot resolve saved term {term}")
            name, level = matches[0]
            column = (query[name] == level).to_numpy(float)
        columns.append(column)
    return np.column_stack(columns)


def common_numeric(result, query, *, alpha=.05):
    beta = np.array([coefficient.estimate for coefficient in result.coefficients])
    design = common_design(result, query)
    finite = finite_gradient(lambda point: design @ point, beta)
    gradient_error = check(design, finite, gradient=True)
    covariance = design @ np.array(result.covariance_matrix) @ design.T
    expected = design @ beta
    output = oe.predict(result, query, interval="mean", alpha=alpha)
    errors = {"mean": check(output["response"], expected), "gradient": gradient_error}
    inference = infer(expected, covariance, alpha, result)
    for column in ("std_error", "ci_low", "ci_high"):
        errors[column] = check(output[column], inference[column], covariance=True)
    assert output.index.equals(query.index)
    parameter_order = [coefficient.term for coefficient in result.coefficients]
    assert output.attrs["response_definition"]
    check(oe.predict(result, query, kind="xb")["xb"], expected)
    check(oe.predict(result, query, kind="stdp")["stdp"], inference["std_error"], covariance=True)
    assert result.model_dump() == stored(result).model_dump()
    return {"maximum_errors": errors, "mean": expected.tolist(), "gradient": design.tolist(),
            "prediction_covariance": covariance.tolist(), "se": inference["std_error"].tolist(),
            "critical": inference["critical"], "df": inference["df"], "parameter_order": parameter_order,
            "parameter_order_source": "Complete JSON-restored ResultBundle; native margins also reports this order"}, output


def spatial_reference(result, query):
    payload = result.extra["spatial_weights"]
    keys = payload["keys"]
    key_column = result.spec.columns["key"]
    key_column = key_column[0] if isinstance(key_column, list) else key_column
    query_keys = query[key_column].tolist()
    assert len(query_keys) == len(keys) and set(query_keys) == set(keys)
    saved_w = np.zeros((len(keys), len(keys)))
    for i, j, value in zip(payload["rows"], payload["cols"], payload["values"], strict=True):
        saved_w[i, j] = value
    order = [keys.index(key) for key in query_keys]
    w = saved_w[np.ix_(order, order)]
    beta = np.array([coefficient.estimate for coefficient in result.coefficients])
    terms = [coefficient.term for coefficient in result.coefficients]
    augmented = []
    for term in terms:
        if term == "Intercept":
            augmented.append(np.ones(len(query)))
        elif term in result.spec.predictors:
            augmented.append(query[term].to_numpy(float))
        elif term.startswith("W:"):
            augmented.append(w @ query[term[2:]].to_numpy(float))
        else:
            augmented.append(np.zeros(len(query)))
    x = np.column_stack(augmented)

    def evaluate(point):
        rho = point[terms.index("rho")] if "rho" in terms else 0.
        return np.linalg.solve(np.eye(len(query)) - rho * w, x @ point)

    mean = evaluate(beta)
    rho = beta[terms.index("rho")] if "rho" in terms else 0.
    filter_inverse = np.linalg.solve(np.eye(len(query)) - rho * w, np.eye(len(query)))
    jacobian = filter_inverse @ x
    if "rho" in terms:
        jacobian[:, terms.index("rho")] = filter_inverse @ w @ mean
    finite = finite_gradient(evaluate, beta)
    check(jacobian, finite, gradient=True)
    covariance = jacobian @ np.array(result.covariance_matrix) @ jacobian.T
    return mean, jacobian, covariance, finite


def spatial_numeric(result, query, *, alpha=.05):
    mean, gradient, covariance, finite = spatial_reference(result, query)
    output = oe.spatial_predict(result, data=query, alpha=alpha)
    errors = {"mean": check(output["means"]["mean"], mean), "gradient": check(output["jacobian"], gradient, gradient=True),
              "finite_gradient": check(output["jacobian"], finite, gradient=True),
              "mean_covariance": check(output["mean_covariance"], covariance, covariance=True),
              "parameter_covariance": check(output["parameter_covariance"], result.covariance_matrix, covariance=True)}
    expected = infer(mean, covariance, alpha, result, spatial=True)
    for column in ("std_error", "ci_low", "ci_high"):
        errors[column] = check(output["means"][column], expected[column], covariance=True)
    assert output.attrs["parameter_order"] == [coefficient.term for coefficient in result.coefficients]
    assert output.attrs["refitted"] is False
    return {"maximum_errors": errors, "mean": mean.tolist(), "gradient": gradient.tolist(),
            "prediction_covariance": covariance.tolist(), "se": expected["std_error"].tolist(),
            "critical": expected["critical"], "parameter_order": output.attrs["parameter_order"]}, output


def expected_refusal(function):
    try:
        function()
    except AnalysisError as error:
        return {"code": error.code, "message": str(error)}
    raise AssertionError("Unsupported/state-invalid route did not raise AnalysisError")


def restored_tables(output):
    names = {name: list(frame.index.names) for name, frame in output.items()}
    output.attrs["table_index_names"] = names
    encoded = oe.summary_state(output)
    restored = oe.restore_summary(encoded)
    assert json.loads(oe.summary_state(restored)) == json.loads(encoded)
    # Shared summary JSON serializes row labels, not pandas Index.names.
    # The explicit caller metadata survives and restores names separately.
    for name, frame in restored.items():
        frame.index.names = restored.attrs["table_index_names"][name]
        assert frame.index.names == output[name].index.names
        assert frame.index.tolist() == output[name].index.tolist()
    latex = restored.to_latex()
    assert "\\begin{tabular}" in latex and len(latex) > 100
    return {"summary_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "summary_bytes": len(encoded.encode()), "tables": {name: len(frame) for name, frame in output.items()},
            "latex_characters": len(latex), "attrs": restored.attrs,
            "index_name_scope": "Explicit table_index_names caller metadata; restore_summary alone does not restore pandas index names"}


def common_cell(name, label, source, result):
    query = common_query(name)
    if label == "numeric_mean_and_gradient":
        result_before = result.model_dump()
        evidence, _ = common_numeric(result, query)
        assert result.model_dump() == result_before
        return evidence
    if label == "full_parameter_covariance_and_order":
        stressed = covariance_stress(result)
        evidence, _ = common_numeric(stressed, query, alpha=.10)
        design = common_design(stressed, query)
        complete = design @ np.array(stressed.covariance_matrix) @ design.T
        diagonal = design @ np.diag(np.diag(stressed.covariance_matrix)) @ design.T
        assert np.max(abs(complete - diagonal)) > 1e-5
        evidence["covariance_scope"] = "Synthetic PSD conditional delta stress on actual fitted coefficients; no re-estimation claim"
        return evidence
    if label == "derivatives_and_margins":
        terms = [coefficient.term for coefficient in result.coefficients]
        beta = np.array([coefficient.estimate for coefficient in result.coefficients])
        variables = ["x", "g"] if name in {"sreg", "mmreg"} else ["x", "c"] if name == "ivcue" else ["x"]
        effect_terms = ["x", "g[B]", "g[C]"] if name in {"sreg", "mmreg"} else variables
        gradient = np.eye(len(terms))[[terms.index(term) for term in effect_terms]]
        expected = gradient @ beta
        covariance = gradient @ np.array(result.covariance_matrix) @ gradient.T
        record = {"gradient": gradient.tolist(), "covariance": covariance.tolist(), "methods": {}}
        for method in ["ame", "mem"]:
            output = oe.margins(result, variables, data=query, method=method, at={"x": [-.5, .5]})
            repeated_gradient, values = np.vstack([gradient, gradient]), np.tile(expected, 2)
            check(output["estimate"], values)
            check(output.attrs["delta_gradients"], repeated_gradient)
            ref = infer(values, repeated_gradient @ np.array(result.covariance_matrix) @ repeated_gradient.T, .05, result)
            for column in ["std_error", "statistic", "p_value", "ci_low", "ci_high"]:
                check(output[column], ref[column], covariance=True)
            record["methods"][method] = {"values": values.tolist(), "p_values": ref["p_value"].tolist(),
                                       "delta_gradients": output.attrs["delta_gradients"]}
        derivative = oe.predict(result, query, kind="derivative", term="x", interval="mean")
        row = np.eye(len(terms))[terms.index("x")]
        derivative_gradient = np.repeat(row[None], len(query), axis=0)
        check(derivative["dydx[x]"], np.repeat(beta[terms.index("x")], len(query)))
        check(derivative["std_error"], np.repeat(np.sqrt(row @ np.array(result.covariance_matrix) @ row), len(query)), covariance=True)
        check(finite_gradient(lambda point: derivative_gradient @ point, beta), derivative_gradient, gradient=True)
        without_weights = oe.margins(result, variables, data=query.drop(columns="w"))
        with_weights = oe.margins(result, variables, data=query)
        check(without_weights["estimate"], with_weights["estimate"])
        record["extraneous_evaluation_weight_column_ignored"] = True
        return record
    if label == "missing_index_dataset_persistence":
        query.iloc[0, query.columns.get_loc("x")] = np.nan
        predicted = oe.predict(result, query, interval="mean")
        assert predicted.index.equals(query.index) and predicted.iloc[0].isna().all()
        complete = query.iloc[1:]
        reference, _ = common_numeric(result, complete)
        check(predicted["response"].iloc[1:], reference["mean"])
        dataset = Dataset.from_batches(lambda: (query.iloc[i:i + 2] for i in range(0, len(query), 2)), list(query.columns), row_count=len(query))
        output = oe.predict(result, dataset, interval="mean", batch_rows=2)
        collected = pd.concat(list(output.iter_batches(batch_rows=2)))
        assert collected.index.equals(query.index)
        assert collected.columns.tolist() == predicted.columns.tolist()
        check(collected, predicted, covariance=True)
        assert output.metadata["analysis"]["missing_prediction_rows"] == 1
        dataset_margins = oe.margins(result, ["x"], data=dataset, batch_rows=2)
        resident_margins = oe.margins(result, ["x"], data=query)
        check(dataset_margins["estimate"], resident_margins["estimate"])
        check(dataset_margins.attrs["delta_gradients"], resident_margins.attrs["delta_gradients"])
        wrapper = TableSet({"predictions": predicted, "margins": resident_margins},
                           prediction_attrs=predicted.attrs, margins_attrs=resident_margins.attrs)
        record = restored_tables(wrapper)
        record.update({"dataset_rows": len(collected), "missing_rows": 1, "input_index": query.index.tolist(),
                       "fitted_state_sha256": hashlib.sha256(result.model_dump_json().encode()).hexdigest(),
                       "model_latex_characters": len(result.to_latex())})
        assert record["model_latex_characters"] > 100
        return record
    refusals = {
        "observation_interval": expected_refusal(lambda: oe.predict(result, query, interval="observation")),
        "missing_predictor": expected_refusal(lambda: oe.predict(result, query.drop(columns="x"))),
        "unrecorded_group_target": expected_refusal(lambda: oe.predict(result, query, target="posterior")),
    }
    tampered = result.model_copy(deep=True)
    tampered.provenance["estimator"] = "ols"
    refusals["mismatched_estimator"] = expected_refusal(lambda: oe.predict(tampered, query))
    covariance = result.model_copy(deep=True)
    covariance.covariance_matrix[0][0] = -1.
    refusals["negative_variance"] = expected_refusal(lambda: oe.predict(covariance, query))
    weighted = result.model_copy(deep=True)
    weighted.spec = weighted.spec.model_copy(update={"weights": "w", "weight_type": "aweight"})
    refusals["unsupported_fit_weights"] = expected_refusal(lambda: oe.predict(weighted, query))
    if name in {"sreg", "mmreg"}:
        unknown = query.copy()
        unknown.iloc[0, unknown.columns.get_loc("g")] = "unfitted"
        refusals["unknown_fitted_category"] = expected_refusal(lambda: oe.predict(result, unknown))
    return {"refusals": refusals}


def spatial_cell(name, label, source, result):
    query = source.drop(columns="y").copy()
    query.index = pd.Index(np.resize(["same", "repeat", "same", "other"], len(query)), name="observation")
    if label == "numeric_mean_gradient_joint_covariance":
        before = result.model_dump()
        evidence, _ = spatial_numeric(result, query)
        assert result.model_dump() == before
        return evidence
    if label == "shuffled_complete_graph_and_changed_X":
        query["x"] = query.x + .2 * query.z + np.linspace(-.3, .3, len(query))
        query["z"] = -.7 * query.z + .1 * query.x
        query = query.iloc[np.arange(len(query) - 1, -1, -1)]
        evidence, output = spatial_numeric(result, query, alpha=.10)
        assert output["means"]["key"].tolist() == query.id.tolist()
        assert output["sample"]["position"].tolist() == list(range(len(query)))
        codes = output["sample"]["index_json"].map(json.loads).tolist()
        assert all(code[0] == "str" for code in codes)
        assert [code[1] for code in codes] == query.index.tolist()
        return evidence
    if label == "full_parameter_covariance_and_order":
        stressed = covariance_stress(result)
        evidence, _ = spatial_numeric(stressed, query, alpha=.10)
        _, gradient, complete, _ = spatial_reference(stressed, query)
        diagonal = gradient @ np.diag(np.diag(stressed.covariance_matrix)) @ gradient.T
        assert np.max(abs(complete - diagonal)) > 1e-5
        terms = [coefficient.term for coefficient in stressed.coefficients]
        for nuisance in ["lambda", "ln_sigma2"]:
            if nuisance in terms:
                assert np.all(gradient[:, terms.index(nuisance)] == 0.)
        evidence["covariance_scope"] = "Synthetic PSD conditional delta stress on actual fitted coefficients; no re-estimation claim"
        return evidence
    if label == "summary_persistence_and_latex":
        _, output = spatial_numeric(result, query)
        assert set(output) == {"means", "sample", "jacobian", "parameter_covariance", "mean_covariance", "settings"}
        record = restored_tables(output)
        record["fitted_state_sha256"] = hashlib.sha256(result.model_dump_json().encode()).hexdigest()
        record["original_index"] = query.index.tolist()
        return record
    dataset = Dataset.from_batches(lambda: iter([query]), list(query.columns), row_count=len(query))
    missing = query.copy()
    missing.iloc[0, missing.columns.get_loc("x")] = np.nan
    changed = query.copy()
    changed.iloc[0, changed.columns.get_loc("id")] = "new-unit"
    refusals = {
        "partial_network": expected_refusal(lambda: oe.spatial_predict(result, data=query.iloc[:-1])),
        "new_network_identity": expected_refusal(lambda: oe.spatial_predict(result, data=changed)),
        "missing_graph_predictor": expected_refusal(lambda: oe.spatial_predict(result, data=missing)),
        "dataset": expected_refusal(lambda: oe.spatial_predict(result, data=dataset)),
        "dense_budget": expected_refusal(lambda: oe.spatial_predict(result, data=query, max_n=2)),
        "work_budget": expected_refusal(lambda: oe.spatial_predict(result, data=query, max_work=1)),
    }
    bad = result.model_copy(deep=True)
    bad.provenance["spatial_weight_hash"] = "0" * 64
    refusals["stale_weight_hash"] = expected_refusal(lambda: oe.spatial_predict(bad, data=query))
    bad = result.model_copy(deep=True)
    bad.covariance_matrix[0][0] = -1.
    refusals["negative_parameter_variance"] = expected_refusal(lambda: oe.spatial_predict(bad, data=query))
    return {"refusals": refusals}


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def refuse_refit(*args, **kwargs):
    raise AssertionError("A saved-target evaluation attempted to refit a model")


def prohibit_refitting(method):
    paths = {"sreg": ("robust.smm", "fit_sreg"), "mmreg": ("robust.smm", "fit_mmreg"),
             "ivcue": ("iv.cue", "fit_ivcue"), "mixedflex": ("mixed.flexible_lmm", "fit_mixedflex")}
    module_name, name = paths.get(method, ("spatial.estimators", "fit_spatial"))
    module = importlib.import_module("openecon.econometrics." + module_name)
    analysis = importlib.import_module("openecon.analysis")
    previous = [(module, name, getattr(module, name)), (analysis, "fit", analysis.fit)]
    for target, attribute, _ in previous:
        setattr(target, attribute, refuse_refit)
    return previous


def run(options, protocol_sha):
    global np, pd, scipy_stats, oe, ResultBundle, Dataset, TableSet, AnalysisError
    import numpy as np
    import pandas as pd
    import scipy
    from scipy import stats as scipy_stats
    import openecon as oe
    from openecon.models import ResultBundle
    from openecon.dataset import Dataset
    from openecon.econometrics.core import TableSet
    from openecon.analysis_contracts import AnalysisError
    import torch
    started, sources = time.time(), pin_sources()
    record = {"schema": "openecon.saved-targets-eight.validation.v1", "protocol_sha256": protocol_sha,
              "source_hashes_start": sources, "validator_sha256": digest(__file__),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                              "scipy": scipy.__version__, "torch": torch.__version__, "platform": platform.platform()},
              "statistical_claim": "Conditional arithmetic and parameter delta-method correctness; no coverage, vendor, frozen-runtime or installed-desktop claim",
              "cells": []}
    with torch.device("cpu"):
        for method in METHODS:
            labels = definition()["cells"]["common" if method in COMMON else "spatial"]
            try:
                source, fitted = make_fixture(method)
                fit_error = None
            except Exception as error:
                source, fitted = None, None
                fit_error = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            guarded = prohibit_refitting(method) if fitted is not None else []
            for label in labels:
                cell = {"method": method, "case": label, "accepted": False}
                if fit_error is not None:
                    cell["failure"] = {"stage": "native_fit", **fit_error}
                else:
                    try:
                        cell["evidence"] = (common_cell if method in COMMON else spatial_cell)(method, label, source, fitted)
                        cell["accepted"] = True
                    except Exception as error:
                        cell["failure"] = {"stage": "prediction_or_independent_reference", "type": type(error).__name__,
                                           "message": str(error), "traceback": traceback.format_exc()}
                record["cells"].append(cell)
                print(json.dumps({"method": method, "case": label, "accepted": cell["accepted"],
                                  "failure": cell.get("failure", {}).get("message")}), flush=True)
            for module, attribute, function in guarded:
                setattr(module, attribute, function)
    record["source_hashes_end"] = pin_sources()
    record["source_hashes_stable"] = record["source_hashes_start"] == record["source_hashes_end"]
    record["sources_present"] = all(value is not None for value in sources.values())
    record["elapsed_seconds"] = time.time() - started
    record["accepted_cells"] = sum(cell["accepted"] for cell in record["cells"])
    record["failed_cells"] = len(record["cells"]) - record["accepted_cells"]
    record["accepted"] = record["accepted_cells"] == 40 and record["source_hashes_stable"] and record["sources_present"]
    review = {"protocol_sha256": protocol_sha, "planned_cells": 40, "observed_cells": len(record["cells"]),
              "accepted_cells": record["accepted_cells"], "failed_cells": record["failed_cells"],
              "source_hashes_stable": record["source_hashes_stable"], "sources_present": record["sources_present"],
              "accepted": record["accepted"], "per_method": {method: {
                  "accepted": sum(cell["accepted"] for cell in record["cells"] if cell["method"] == method),
                  "planned": 5} for method in METHODS},
              "not_claimed": definition()["scientific_scope"]["not_claimed"]}
    for path, value in [(EVIDENCE / options.output, record), (EVIDENCE / options.review_output, review)]:
        with path.open("x") as stream:
            json.dump(json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps(review), flush=True)
    if not record["accepted"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--output", default="scientific-validation.json")
    parser.add_argument("--review-output", default="review.json")
    options = parser.parse_args()
    protocol_sha = freeze()
    if options.protocol_only:
        print(json.dumps({"protocol_sha256": protocol_sha, "planned_cells": 40}))
        return
    run(options, protocol_sha)


if __name__ == "__main__":
    main()
