"""Independent NumPy/SciPy oracle for eight actual-native MI extension outputs.

No openecon or Torch import is used. Native execution/source/restart provenance
is supplied by the separate native receipt. Final model equations, saved inputs,
full covariance inference and the actual displayed tables are compared here;
finite-chain convergence and sampling-realization parity are not claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln, log_expit, log_softmax
from scipy.stats import norm, t

METHODS = ("poisson", "ordinal", "multinomial", "normal_delta", "logit_delta",
           "poisson_delta", "passive", "lincom")
SOURCES = (
    "https://amices.org/mice/reference/mice.impute.mnar.html",
    "https://mc-stan.org/docs/2_29/stan-users-guide/posterior-prediction-for-regressions.html",
    "https://mc-stan.org/docs/stan-users-guide/regression.html",
    "https://doi.org/10.1093/biomet/86.4.948",
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Checks:
    def __init__(self):
        self.items = []

    def truth(self, name, passed, actual=None, reference=None):
        self.items.append({"name": name, "pass": bool(passed), "actual": actual, "reference": reference})

    def equal(self, name, actual, reference):
        self.truth(name, actual == reference, actual, reference)

    def numeric(self, name, actual, reference, *, atol=1e-10, rtol=1e-8):
        a, b = np.asarray(actual, float), np.asarray(reference, float)
        same = a.shape == b.shape
        passed = same and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
        finite = np.isfinite(a) & np.isfinite(b) if same else np.array([], bool)
        error = float(np.max(abs(a[finite] - b[finite]))) if finite.any() else (0.0 if same else None)
        self.items.append({"name": name, "pass": bool(passed), "actual": a.tolist(),
                           "reference": b.tolist(), "atol": atol, "rtol": rtol, "max_abs_error": error})

    def finish(self, **values):
        return {"pass": all(item["pass"] for item in self.items), "check_count": len(self.items),
                "max_abs_error": max((item.get("max_abs_error") or 0 for item in self.items), default=0),
                "checks": self.items, **values}


def checksum(checks, state, name):
    checks.equal(name, state["integrity_sha256"], canonical_hash({k: v for k, v in state.items()
                                                              if k != "integrity_sha256"}))


def index_values(state):
    def label(item):
        if item["type"] == "tuple":
            return [label(value) for value in item["value"]]
        if item["type"] in ("nan", "nat", "missing"):
            return None
        return item["value"]
    if state["kind"] == "range":
        return list(range(state["start"], state["stop"], state["step"]))
    if state["kind"] == "categorical":
        levels = index_values(state["categories"])
        return [None if code < 0 else levels[code] for code in state["codes"]]
    if state["kind"] == "multi":
        levels = [index_values(level) for level in state["levels"]]
        return [[None if code < 0 else values[code] for values, code in zip(levels, codes)]
                for codes in zip(*state["codes"])]
    return [label(value) for value in state["values"]]


def table_frame(output):
    if output.get("type") != "table" or not isinstance(output.get("latex"), str) or not output["latex"]:
        raise ValueError("Require an actual recorded rendered/exportable table.")
    data = output["data"]
    columns, rows = data["columns"], data["rows"]
    if len(set(columns)) != len(columns) or any(len(row) != len(columns) for row in rows):
        raise ValueError("Malformed actual native table.")
    if data.get("total_rows") != len(rows) or data.get("total_columns") != len(columns):
        raise ValueError("A truncated display payload cannot establish full table acceptance.")
    return pd.DataFrame(rows, columns=columns)


def base_state(checks, state, inputs, *, displayed=None, count_columns=None, allow_original_roundoff=False):
    checksum(checks, state, "full MI state checksum")
    columns, original = state["columns"], state["original"]
    source = np.asarray([[inputs[name][i] for name in columns] for i in range(len(original))], float)
    data = np.asarray(original, float)
    missing = np.isnan(data)
    source_equal = (np.allclose(source, data, equal_nan=True, rtol=1e-10, atol=1e-12)
                    if allow_original_roundoff else np.array_equal(source, data, equal_nan=True))
    checks.truth("original state matches actual native source including null geometry",
                 source_equal, original,
                 [[inputs[name][i] for name in columns] for i in range(len(original))])
    completed = np.asarray(state["completed_matrices"], float)
    m, n, p = completed.shape
    checks.equal("completed dimensions", [n, p], list(data.shape))
    checks.truth("all completed cells finite", np.isfinite(completed).all())
    checks.numeric("every observed cell remains unchanged", completed[:, ~missing], np.broadcast_to(data[~missing], (m, (~missing).sum())), atol=0, rtol=0)
    meta = state["metadata"]
    checks.equal("native row index values including duplicates", index_values(meta["index"]), inputs["__index__"])
    checks.equal("metadata dimensions", [meta[k] for k in ("n", "p", "m")], [n, p, m])
    checks.equal("metadata original missing mask", meta["missing_mask"], missing.tolist())
    checks.equal("metadata missing counts", meta["missing_counts"], missing.sum(0).tolist())
    checks.equal("metadata missing total", meta["missing_total"], int(missing.sum()))
    checks.equal("original physical sample", meta["sample"], {"nobs_original": n, "nobs": n, "dropped_rows": 0, "positions": list(range(n))})
    checks.equal("ordered imputation IDs", meta["imputation_ids"], list(range(1, m + 1)))
    checks.truth("no convergence or blanket vendor parity claim", meta["converged"] is False and meta["stata_parity_validated"] is False)
    if displayed is not None:
        selected = columns if count_columns is None else count_columns
        checks.equal("actual displayed count-column order", displayed["column"].tolist(), list(selected))
        indices = [columns.index(name) for name in selected]
        checks.numeric("actual displayed observed counts", displayed["observed"], (n - missing.sum(0))[indices], atol=0, rtol=0)
        checks.numeric("actual displayed missing counts", displayed["missing"], missing.sum(0)[indices], atol=0, rtol=0)
        checks.numeric("actual displayed imputation counts", displayed["imputations"], [m] * len(selected), atol=0, rtol=0)
    return data, missing, completed


def require(condition, name):
    if not condition:
        raise ValueError(name)


def ordered_diagnostics(state, key):
    meta, m = state["metadata"], len(state["completed_matrices"])
    diagnostics = meta.get(key)
    seeds = meta.get("imputation_seeds")
    require(isinstance(diagnostics, list) and len(diagnostics) == m,
            f"Exactly {m} ordered {key} records are required.")
    require(type(state.get("seed")) is int and isinstance(seeds, list) and len(seeds) == m
            and all(type(seed) is int for seed in seeds)
            and seeds == [(state["seed"] + i) % (2**63) for i in range(m)],
            "Independent ordered imputation seeds must match the saved root seed.")
    for index, item in enumerate(diagnostics):
        require(isinstance(item, dict) and type(item.get("imputation")) is int
                and item["imputation"] == index + 1 and type(item.get("seed")) is int
                and item["seed"] == seeds[index], "Diagnostic imputation IDs/seeds are incomplete or unordered.")
    return diagnostics


def mh_diagnostic(checks, diagnostic, burn, steps, label, *, nonfinite):
    fields = {"burn_proposals", "sampling_proposals", "accepted_burn", "accepted_sampling",
              "acceptance_rate", "final_log_posterior"}
    if nonfinite:
        fields.add("rejected_nonfinite")
    require(isinstance(diagnostic, dict) and fields.issubset(diagnostic), label + " missing MH diagnostics.")
    require(type(diagnostic["burn_proposals"]) is int and diagnostic["burn_proposals"] == burn
            and type(diagnostic["sampling_proposals"]) is int and diagnostic["sampling_proposals"] == steps,
            label + " MH proposal counts disagree.")
    accepted_burn, accepted_sampling = diagnostic["accepted_burn"], diagnostic["accepted_sampling"]
    require(type(accepted_burn) is int and type(accepted_sampling) is int
            and 0 <= accepted_burn <= burn and 0 <= accepted_sampling <= steps,
            label + " MH acceptance counts are invalid.")
    if nonfinite:
        rejected = diagnostic["rejected_nonfinite"]
        require(type(rejected) is int and 0 <= rejected <= burn + steps - accepted_burn - accepted_sampling,
                label + " nonfinite rejection count is invalid.")
    require(not isinstance(diagnostic["final_log_posterior"], bool)
            and isinstance(diagnostic["final_log_posterior"], (int, float))
            and math.isfinite(diagnostic["final_log_posterior"]), label + " posterior target must be finite.")
    checks.numeric(label + " MH acceptance rate", diagnostic["acceptance_rate"],
                   (accepted_burn + accepted_sampling) / (burn + steps))


def discrete_coverage(checks, state, missing):
    meta, columns = state["metadata"], state["columns"]
    targets = [name for j, name in enumerate(columns) if missing[:, j].any()]
    require(targets and isinstance(meta.get("methods"), dict) and set(meta["methods"]) == set(targets)
            and meta.get("visit_order") == targets, "Discrete methods/visit order must cover every incomplete target.")
    require(isinstance(meta.get("predictors"), dict) and set(meta["predictors"]) == set(targets),
            "Discrete predictors must cover every target.")
    burn, iterations = meta.get("burn"), meta.get("iterations")
    require(type(burn) is int and 0 <= burn <= 1000 and type(iterations) is int
            and 1 <= iterations and burn + iterations <= 1000,
            "Meaningful discrete chain sweeps are required.")
    cycles = burn + iterations
    require(meta.get("cycles_per_chain") == cycles, "Discrete cycle declaration disagrees.")
    sampler = meta["sampler"]
    mh_burn, mh_steps = sampler.get("burn_per_update"), sampler.get("steps_per_update")
    require(type(mh_burn) is int and 0 <= mh_burn <= 100000 and type(mh_steps) is int
            and 1 <= mh_steps <= 100000, "Meaningful finite MH proposal counts are required.")
    chains = ordered_diagnostics(state, "chain_diagnostics")
    for index, chain in enumerate(chains):
        require(isinstance(chain.get("last_models"), dict) and set(chain["last_models"]) == set(targets),
                "Final conditional models must exactly cover all targets.")
        trace = chain.get("trace")
        require(isinstance(trace, list) and len(trace) == cycles and trace,
                "Every declared discrete sweep including the final sweep must be saved.")
        for cycle, entry in enumerate(trace, 1):
            require(isinstance(entry, dict) and type(entry.get("cycle")) is int and entry["cycle"] == cycle
                    and entry.get("phase") == ("burn" if cycle <= burn else "sampling")
                    and isinstance(entry.get("variables"), dict) and set(entry["variables"]) == set(targets),
                    "Discrete trace IDs/phases/target coverage are incomplete or unordered.")
            for target, diagnostic in entry["variables"].items():
                require(isinstance(diagnostic, dict) and diagnostic.get("method") == meta["methods"][target], "Discrete trace method disagrees.")
                label = f"chain {index + 1} cycle {cycle} {target}"
                mh_diagnostic(checks, diagnostic, mh_burn, mh_steps, label, nonfinite=True)
                require(all(key in diagnostic and not isinstance(diagnostic[key], bool)
                            and isinstance(diagnostic[key], (int, float)) and math.isfinite(diagnostic[key])
                            for key in ("imputed_mean", "imputed_variance"))
                        and diagnostic["imputed_variance"] >= 0, "Every discrete sweep needs finite predictive summaries.")
        for target, model in chain["last_models"].items():
            require(isinstance(model, dict) and model.get("method") == meta["methods"][target]
                    and model.get("predictors") == meta["predictors"][target]
                    and model.get("intercept") is (model["method"] != "ordinal")
                    and model.get("prediction_at") == "target update in final sweep, before subsequent target updates",
                    "Saved final conditional model identity is invalid.")
    checks.equal("complete discrete chain/model/trace coverage", len(chains), len(state["completed_matrices"]))
    return chains


def ordinal_distribution(design, theta, categories):
    d = design.shape[1]
    cuts = np.r_[theta[d], theta[d] + np.cumsum(np.exp(theta[d + 1:]))]
    if len(cuts) != categories - 1 or np.any(np.diff(cuts) <= 0):
        raise ValueError("Oracle ordinal thresholds must be strictly increasing.")
    shifted = cuts[None, :] - (design @ theta[:d])[:, None]
    # Ordered logistic CDF differences, evaluated with SciPy log-CDFs and the
    # exact logistic interval-mass identity to retain tiny probabilities.
    interior = (log_expit(shifted[:, 1:]) + log_expit(-shifted[:, :-1])
                + np.log(-np.expm1(-np.exp(theta[d + 1:])))[None, :])
    logs = np.column_stack([log_expit(shifted[:, 0]), interior, log_expit(-shifted[:, -1])])
    return logs, cuts


def discrete(method, state, inputs, shown):
    checks = Checks()
    data, missing, completed = base_state(checks, state, inputs, displayed=shown)
    meta, columns = state["metadata"], state["columns"]
    checks.equal("declared native stage method", sorted(set(meta["methods"].values())), [method])
    checks.truth("finite MH convergence and compatibility unassessed", meta["sampler"]["stationarity_claim"] is False and meta["convergence_claim"] is False and meta["conditional_compatibility_assessed"] is False)
    models = []
    for chain_index, chain in enumerate(discrete_coverage(checks, state, missing)):
        for target, model in chain["last_models"].items():
            j = columns.index(target)
            absent, present = np.flatnonzero(missing[:, j]), np.flatnonzero(~missing[:, j])
            checks.equal(f"chain {chain_index + 1} {target} missing positions", model["missing_row_positions"], absent.tolist())
            xo, xm = np.asarray(model["observed_design"], float), np.asarray(model["missing_design"], float)
            d = len(model["predictors"]) + int(method != "ordinal")
            xo, xm = xo.reshape(len(present), d), xm.reshape(len(absent), d)
            offset = int(method != "ordinal")
            for label, positions, design in (("observed", present, xo), ("missing", absent, xm)):
                if offset:
                    checks.numeric(f"chain {chain_index + 1} {target} {label} intercept", design[:, 0], np.ones(len(positions)), atol=0, rtol=0)
                for index, predictor in enumerate(model["predictors"]):
                    c = columns.index(predictor)
                    known = ~missing[positions, c]
                    checks.numeric(f"chain {chain_index + 1} {target} {label} known predictor {predictor}", design[known, index + offset], data[positions[known], c], atol=0, rtol=0)
                    if predictor in meta["visit_order"] and meta["visit_order"].index(predictor) < meta["visit_order"].index(target):
                        unknown = ~known
                        checks.numeric(f"chain {chain_index + 1} {target} earlier final-sweep predictor {predictor}", design[unknown, index + offset], completed[chain_index, positions[unknown], c], atol=0, rtol=0)
            scale = meta["prior"]["scale"]
            y = data[present, j]
            if method == "poisson":
                theta = np.asarray(model["coefficients"], float)
                eta = xo @ theta
                observed_rates = np.exp(eta)
                checks.truth(f"chain {chain_index + 1} {target} observed-design rate bound",
                             np.isfinite(observed_rates).all() and np.all(observed_rates <= 1_000_000))
                full_likelihood = float((y * eta - observed_rates - gammaln(y + 1)).sum())
                log_target = full_likelihood + gammaln(y + 1).sum() - theta @ theta / (2 * scale**2)
                prediction = np.exp(xm @ theta)
                checks.numeric(f"chain {chain_index + 1} {target} Poisson predictive means", model["missing_means"], prediction)
                draws = completed[:, :, j]
                checks.truth(f"{target} nonnegative integer predictive support", np.all((draws >= 0) & (draws == np.floor(draws))))
                checks.truth(f"{target} observed integer count bound", np.all((y >= 0) & (y <= 1_000_000) & (y == np.floor(y))))
                checks.truth(f"chain {chain_index + 1} {target} predicted mean bound", np.isfinite(prediction).all() and np.all(prediction <= 1_000_000))
            else:
                categories = model["categories"]
                theta = np.asarray(model["coordinates"], float)
                if method == "ordinal":
                    observed_logs, cuts = ordinal_distribution(xo, theta, len(categories))
                    missing_logs, _ = ordinal_distribution(xm, theta, len(categories))
                    checks.numeric(f"chain {chain_index + 1} {target} ordinal thresholds", model["cutpoints"], cuts)
                    checks.numeric(f"chain {chain_index + 1} {target} ordinal coefficients", model["coefficients"], theta[:d])
                    checks.numeric(f"chain {chain_index + 1} {target} ordinal log gaps", model["log_gaps"], theta[d + 1:])
                else:
                    beta = theta.reshape(d, len(categories) - 1)
                    observed_logs = log_softmax(np.column_stack([np.zeros(len(xo)), xo @ beta]), axis=1)
                    missing_logs = log_softmax(np.column_stack([np.zeros(len(xm)), xm @ beta]), axis=1)
                    checks.numeric(f"chain {chain_index + 1} {target} multinomial full coefficients", model["coefficients"], beta)
                    checks.equal(f"chain {chain_index + 1} {target} declared zero-logit baseline", model["baseline_category"], categories[0])
                codes = np.asarray([categories.index(value) for value in y])
                full_likelihood = float(observed_logs[np.arange(len(y)), codes].sum())
                log_target = full_likelihood - theta @ theta / (2 * scale**2)
                prediction = np.exp(missing_logs)
                checks.numeric(f"chain {chain_index + 1} {target} category probabilities", model["missing_probabilities"], prediction)
                checks.numeric(f"chain {chain_index + 1} {target} category probability normalization", prediction.sum(1), np.ones(len(prediction)))
                checks.truth(f"{target} declared category support", np.isin(completed[:, :, j], categories).all())
            trace = chain["trace"][-1]["variables"][target]
            checks.numeric(f"chain {chain_index + 1} {target} actual posterior target", trace["final_log_posterior"], log_target)
            fills = completed[chain_index, absent, j]
            checks.numeric(f"chain {chain_index + 1} {target} final drawn mean", trace["imputed_mean"], fills.mean())
            checks.numeric(f"chain {chain_index + 1} {target} final drawn variance", trace["imputed_variance"], fills.var())
            models.append({"imputation": chain_index + 1, "target": target, "coordinates": theta,
                           "full_observed_log_likelihood": full_likelihood, "log_posterior_without_fixed_constants": log_target,
                           "predictive_distribution": prediction})
    return checks.finish(source_values={"rows": len(data), "columns": columns, "imputations": len(completed), "models": models}, scope="Final conditional posterior equations and predictive distributions, support, physical identity and actual native count tables; no finite-chain stationarity or sampling-realization equivalence.")


def delta(method, state, inputs, shown):
    checks = Checks()
    data, missing, completed = base_state(checks, state, inputs, displayed=shown)
    meta, columns = state["metadata"], state["columns"]
    kind = method.removesuffix("_delta")
    checks.equal("declared delta stage kind", meta["kind"], kind)
    checks.equal("fixed single-target operation", meta["operation"], "fixed_single_target_pattern_mixture")
    checks.truth("unidentifiable fixed sensitivity and no stationarity claim", meta["delta_identified_from_observed_data"] is False and meta["sampler"]["stationarity_claim"] is False and meta["convergence_claim"] is False)
    target, adjustment = meta["target"], meta["delta"]
    j = columns.index(target)
    observed, absent = np.flatnonzero(~missing[:, j]), np.flatnonzero(missing[:, j])
    checks.truth("exactly one incomplete selected target", np.array_equal(missing.any(0), np.arange(len(columns)) == j))
    predictors = [columns.index(name) for name in meta["predictors"][target]]
    design = np.column_stack([np.ones(len(data)), data[:, predictors]])
    xo, xm, y = design[observed], design[absent], data[observed, j]
    if kind == "normal":
        hat = np.linalg.solve(xo.T @ xo, xo.T @ y)
        sse = np.square(y - xo @ hat).sum()
        checks.numeric("independent observed Gaussian coefficients", meta["observed_model"]["beta_hat"], hat)
        checks.numeric("independent observed Gaussian SSE", meta["observed_model"]["sse"], sse)
        checks.equal("independent Gaussian residual df", meta["observed_model"]["residual_df"], len(y) - xo.shape[1])
    models = []
    diagnostics = ordered_diagnostics(state, "imputation_diagnostics")
    checks.equal("complete delta model coverage", len(diagnostics), len(completed))
    for index, diagnostic in enumerate(diagnostics):
        required_fields = {"coefficient_draw", "base_missing_linear_predictor", "missing_linear_predictor",
                           "missing_positions", "observed_log_likelihood"}
        required_fields |= ({"sigma2_draw", "residual_df", "predictive_mean"} if kind == "normal"
                            else {"sampler", "predictive_probability" if kind == "logit" else "predictive_mean"})
        require(required_fields.issubset(diagnostic), "Delta model/likelihood/predictive fields are incomplete.")
        beta = np.asarray(diagnostic["coefficient_draw"], float)
        require(beta.shape == (design.shape[1],) and np.isfinite(beta).all(), "Delta coefficient vector is incomplete.")
        if kind != "normal":
            mh_diagnostic(checks, diagnostic["sampler"], meta["sampler"]["burn"], meta["sampler"]["steps"],
                          f"delta imputation {index + 1}", nonfinite=kind == "poisson")
        base, shifted = xm @ beta, xm @ beta + adjustment
        observed_eta = xo @ beta
        checks.equal(f"imputation {index + 1} exact missing row positions", diagnostic["missing_positions"], absent.tolist())
        checks.numeric(f"imputation {index + 1} unshifted missing predictor", diagnostic["base_missing_linear_predictor"], base)
        checks.numeric(f"imputation {index + 1} delta-shifted missing predictor", diagnostic["missing_linear_predictor"], shifted)
        if kind == "normal":
            variance = diagnostic["sigma2_draw"]
            checks.truth(f"imputation {index + 1} positive Gaussian variance", math.isfinite(variance) and variance > 0)
            likelihood = norm.logpdf(y, loc=observed_eta, scale=np.sqrt(variance)).sum()
            prediction = shifted
            checks.numeric(f"imputation {index + 1} Gaussian mean", diagnostic["predictive_mean"], prediction)
        elif kind == "logit":
            likelihood = (y * observed_eta - np.logaddexp(0, observed_eta)).sum()
            prediction = expit(shifted)
            checks.numeric(f"imputation {index + 1} Bernoulli probability", diagnostic["predictive_probability"], prediction)
            checks.numeric(f"imputation {index + 1} odds ratio delta effect", prediction / (1 - prediction), np.exp(adjustment) * expit(base) / (1 - expit(base)))
            checks.truth("binary completed support", np.isin(completed[:, :, j], [0, 1]).all())
        else:
            likelihood = (y * observed_eta - np.exp(observed_eta) - gammaln(y + 1)).sum()
            prediction = np.exp(shifted)
            checks.numeric(f"imputation {index + 1} Poisson mean", diagnostic["predictive_mean"], prediction)
            checks.numeric(f"imputation {index + 1} log-link mean ratio delta effect", prediction, np.exp(adjustment) * np.exp(base))
            checks.truth("bounded integer completed counts", np.all((completed[:, :, j] >= 0) & (completed[:, :, j] <= 1_000_000) & (completed[:, :, j] == np.floor(completed[:, :, j]))))
            checks.truth(f"imputation {index + 1} bounded positive missing means", np.all((prediction > 0) & (prediction <= 1_000_000)))
        checks.numeric(f"imputation {index + 1} full observed likelihood unaffected by delta", diagnostic["observed_log_likelihood"], likelihood)
        if kind != "normal":
            prior_penalty = beta @ beta / (2 * meta["prior"]["scale"]**2)
            posterior = likelihood - prior_penalty + (gammaln(y + 1).sum() if kind == "poisson" else 0)
            checks.numeric(f"imputation {index + 1} unchanged observed posterior target", diagnostic["sampler"]["final_log_posterior"], posterior)
        models.append({"imputation": index + 1, "coefficients": beta, "delta": adjustment,
                       "base_missing_predictor": base, "missing_predictive_distribution": prediction,
                       "full_observed_log_likelihood": float(likelihood)})
    return checks.finish(source_values={"rows": len(data), "columns": columns, "delta": adjustment, "models": models}, scope="Fixed single-target delta equations and full observed likelihood, saved predictive distribution, support, geometry and actual displayed table; no inference of the unidentifiable delta or finite-MH convergence.")


def passive_expand(values, columns, recipes):
    result, names = np.array(values, float, copy=True), list(columns)
    for name, recipe in recipes.items():
        selected = result[..., [names.index(item) for item in recipe["inputs"]]]
        if recipe["operation"] == "affine":
            derived = np.sum(selected * np.asarray(recipe["coefficients"]), axis=-1) + recipe["intercept"]
        elif recipe["operation"] == "product":
            derived = np.prod(selected, axis=-1)
        elif recipe["operation"] == "power":
            derived = selected[..., 0] ** recipe["exponent"]
        else:
            raise ValueError("Unknown passive operation.")
        derived = np.where(np.isnan(selected).any(-1), np.nan, derived)
        result = np.concatenate([result, derived[..., None]], axis=-1)
        names.append(name)
    return result, names


def passive(state, states, inputs, shown):
    checks = Checks()
    checksum(checks, state, "passive outer checksum")
    source, result = state["source"], state["result"]
    checks.equal("passive source is actual saved normal-delta result", source, states["normal_delta"])
    checksum(checks, source, "passive source checksum")
    recipes = json.loads(state["recipes_json"])
    expected_original, columns = passive_expand(source["original"], source["columns"], recipes)
    expected_complete, _ = passive_expand(source["completed_matrices"], source["columns"], recipes)
    actual_original = np.asarray(result["original"], float)
    checks.truth("all original passive DAG cells and null propagation", np.allclose(expected_original, actual_original, equal_nan=True, atol=1e-12, rtol=1e-10), result["original"], finite_json(expected_original))
    checks.numeric("all completed passive DAG cells", result["completed_matrices"], expected_complete)
    checks.equal("passive extended column order", result["columns"], columns)
    expected_inputs = {name: [None if np.isnan(value) else float(value) for value in expected_original[:, j]] for j, name in enumerate(columns)}
    expected_inputs["__index__"] = inputs["normal_delta"]["__index__"]
    base_state(checks, result, expected_inputs, displayed=shown, count_columns=list(recipes), allow_original_roundoff=True)
    checks.equal("actual displayed passive operation order", shown["operation"].tolist(), [recipe["operation"] for recipe in recipes.values()])
    meta = json.loads(state["metadata_json"])
    checks.truth("no new stochastic draw/FCS feedback/congeniality claim", meta["new_stochastic_draws"] == 0 and meta["fcs_feedback"] is False and meta["substantive_model_compatibility_claim"] is False)
    checks.equal("passive index descriptor preserved", result["metadata"]["index"], source["metadata"]["index"])
    checks.equal("passive imputation seeds preserved", result["metadata"]["imputation_seeds"], source["metadata"]["imputation_seeds"])
    return checks.finish(source_values={"rows": len(expected_original), "columns": columns, "recipes": recipes, "completed_cells": expected_complete.size}, scope="Every original/completed NumPy DAG cell with declared null propagation and source identity, plus actual displayed passive metadata; no FCS feedback or congeniality claim.")


def pool_values(pool, *, null=None):
    q, u = np.asarray(pool["estimates"], float), np.asarray(pool["covariances"], float)
    m = len(q)
    mean, within = q.mean(0), u.mean(0)
    centered = q - mean
    between = centered.T @ centered / (m - 1)
    added = (1 + 1 / m) * between
    total = within + added
    lam = np.diag(added) / np.diag(total)
    riv = np.diag(added) / np.diag(within)
    with np.errstate(divide="ignore"):
        old_df = (m - 1) / lam**2
    complete = pool["complete_df"]
    df = old_df if complete is None else 1 / (1 / old_df + 1 / (complete * (complete + 1) / (complete + 3) * (1 - lam)))
    se, alpha = np.sqrt(np.diag(total)), pool["alpha"]
    null = np.zeros(len(mean)) if null is None else np.asarray(null, float)
    statistic = (mean - null) / se
    probability = np.where(np.isinf(df), 2 * norm.sf(abs(statistic)), 2 * t.sf(abs(statistic), df))
    critical = np.where(np.isinf(df), norm.isf(alpha / 2), t.isf(alpha / 2, df))
    return {"estimate": mean, "std_error": se, "statistic": statistic, "df": df,
            "p_value": probability, "ci_low": mean - critical * se, "ci_high": mean + critical * se,
            "lambda": lam, "relative_increase_variance": riv,
            "fraction_missing_information": (riv + 2 / (df + 3)) / (1 + riv),
            "within_covariance": within, "between_covariance": between, "total_covariance": total}


def compare_pool_tables(checks, pool, tables, name):
    reference = pool_values(pool)
    coefficients = pd.DataFrame(tables["coefficients"]["rows"], columns=tables["coefficients"]["columns"])
    checks.equal(name + " term order", coefficients["term"].tolist(), pool["terms"])
    for field, values in reference.items():
        if field.endswith("_covariance"):
            checks.equal(name + " " + field + " labels", tables[field]["columns"], pool["terms"])
            checks.equal(name + " " + field + " row labels", tables[field]["index"], pool["terms"])
            checks.numeric(name + " " + field, tables[field]["rows"], values)
        else:
            actual = [np.inf if pd.isna(value) else value for value in coefficients[field]] if field == "df" else coefficients[field]
            checks.numeric(name + " " + field, actual, values)


def lincom(state, states, inputs, shown):
    checks = Checks()
    checksum(checks, state, "linear contrast outer checksum")
    source, target = state["source_pool"], state["transformed_pool"]
    checksum(checks, source, "linear contrast source pool checksum")
    checksum(checks, target, "linear contrast transformed pool checksum")
    original = states["normal_delta"]
    data, columns = np.asarray(original["completed_matrices"], float), original["columns"]
    y_name = original["metadata"]["target"]
    qs, us = [], []
    for imputation in data:
        design = np.column_stack([np.ones(len(imputation)) if term in ("_cons", "const", "Intercept", "intercept") else imputation[:, columns.index(term)] for term in source["terms"]])
        y = imputation[:, columns.index(y_name)]
        gram = design.T @ design
        beta = np.linalg.solve(gram, design.T @ y)
        df = len(y) - np.linalg.matrix_rank(design)
        sigma2 = np.square(y - design @ beta).sum() / df
        qs.append(beta)
        us.append(sigma2 * np.linalg.inv(gram))
    qs, us = np.asarray(qs), np.asarray(us)
    checks.numeric("independent per-imputation OLS coefficients", source["estimates"], qs)
    checks.numeric("independent per-imputation full OLS covariances", source["covariances"], us)
    checks.numeric("actual finite complete-data residual df", source["complete_df"], df, atol=0, rtol=0)
    compare_pool_tables(checks, source, inputs["pool_tables"], "actual native source pool")
    r, null = np.asarray(state["weights"], float), np.asarray(state["values"], float)
    transformed_q = qs @ r.T
    transformed_u = np.asarray([r @ u @ r.T for u in us])
    checks.numeric("every projected per-imputation q_i R'", target["estimates"], transformed_q)
    checks.numeric("every full projected R U_i R'", target["covariances"], transformed_u)
    checks.equal("projection label order", target["terms"], state["names"])
    checks.equal("projection imputation IDs", target["imputation_ids"], source["imputation_ids"])
    checks.equal("projection complete-data df", target["complete_df"], source["complete_df"])
    checks.equal("projection imputation description", target["imputation_description"], source["imputation_description"])
    reference_pool = {**target, "estimates": transformed_q, "covariances": transformed_u}
    values = pool_values(reference_pool, null=null)
    checks.equal("actual displayed contrast name order", shown["term"].tolist(), state["names"])
    checks.numeric("actual displayed nonzero null values", shown["null_value"], null, atol=0, rtol=0)
    for name, expected in values.items():
        if name.endswith("_covariance"):
            continue
        actual = [np.inf if pd.isna(value) else value for value in shown[name]] if name == "df" else shown[name]
        checks.numeric("actual displayed contrast " + name, actual, expected)
    checks.truth("marginal inference makes no joint/multiplicity/parity claim", json.loads(state["metadata_json"])["joint_test"] is False and json.loads(state["metadata_json"])["simultaneous_intervals"] is False and json.loads(state["metadata_json"])["stata_parity_validated"] is False)
    # Full within/between/total covariance is retained as an independent source
    # value even though the native visible table is the marginal contrast table.
    return checks.finish(source_values={"weights": r, "null": null, "independent_source_q": qs,
                         "independent_source_u": us, "projected_q": transformed_q, "projected_u": transformed_u,
                         "full_pooled_projection": values}, scope="Every original OLS q/U and full R U R' before Rubin/Barnard-Rubin pooling, contrast-specific df/t/p/CI and actual nonzero-null display; intervals target R Q, not R Q minus the null.")


_NATIVE_IDENTIFIER = "org.openecon.qa.mi-extensions-eight-20261007"
_NATIVE_MARKER = "MI_EXTENSIONS_INSTALLED:"
_NATIVE_MODULES = {
    "openecon", "openecon.models", "openecon.analysis", "openecon.linear_ols",
    "openecon.linear_ols.spec", "openecon.linear_ols.design", "openecon.linear_ols.estimation",
    "openecon.engines.torch_engine", "openecon.engines.inference", "openecon.console_worker",
    "openecon.econometrics.mi", "openecon.econometrics.mi.common", "openecon.econometrics.mi.diagnostics",
    "openecon.econometrics.mi.generation", "openecon.econometrics.mi.chained", "openecon.econometrics.mi.pooling",
    "openecon.econometrics.mi.joint", "openecon.econometrics.mi.discrete", "openecon.econometrics.mi.sensitivity",
    "openecon.econometrics.mi.passive", "openecon.econometrics.mi.lincom", "openecon.econometrics.registry",
    "openecon.analysis_contracts", "openecon.econometrics.core", "openecon.resources", "openecon.engines.distributions",
}


def native_coherence(receipt):
    """Check internal links only; a JSON document cannot authenticate its author."""
    if receipt.get("status") not in ("native-verified", "native-restarted"):
        return None
    checks = Checks()
    try:
        identity, proof, execution = (receipt.get(key) for key in ("identity", "proof", "execution_record"))
        require(all(isinstance(value, dict) for value in (identity, proof, execution)),
                "Native evidence needs identity, full execution record and proof.")
        checks.equal("dedicated native QA identifier", identity.get("identifier"), _NATIVE_IDENTIFIER)
        app, root, workspace = (Path(value) for value in (identity["app"], proof["root"], proof["workspace"]))
        checks.truth("absolute dedicated app and frozen root", app.is_absolute()
                     and app.name == "OpenEconometrics MI Extensions QA.app" and root.is_absolute()
                     and root.is_relative_to(app) and not any(".." in path.parts for path in (app, root, workspace)))
        checks.truth("native version declared", isinstance(identity.get("version"), str) and bool(identity["version"].strip()))
        checks.truth("explicit full source pin", isinstance(identity.get("source_ref"), str)
                     and re.fullmatch(r"[0-9a-f]{40}", identity["source_ref"]) is not None)
        for key in ("native_executable_sha256", "runtime_manifest_sha256", "runtime_sha256", "frontend_sha256"):
            checks.truth("declared native identity " + key, isinstance(identity.get(key), str)
                         and re.fullmatch(r"[0-9a-f]{64}", identity[key]) is not None, identity.get(key))
        checks.equal("QA signature scope", identity.get("signature"), "ad-hoc QA; no Developer ID/notarization/public release claim")
        parity, modules = identity.get("source_parity"), proof.get("modules")
        require(isinstance(parity, dict) and isinstance(modules, dict), "Native module identities are required.")
        checks.equal("complete pinned scientific module inventory", sorted(parity), sorted(_NATIVE_MODULES))
        checks.equal("frozen loaded module inventory matches source audit", sorted(modules), sorted(parity))
        checks.truth("all source module SHA values declared", all(isinstance(value, str)
                     and re.fullmatch(r"[0-9a-f]{64}", value) is not None for value in parity.values()))
        paths_valid = True
        for name, value in modules.items():
            if not isinstance(value, str):
                paths_valid = False
                continue
            path = Path(value)
            if not path.is_absolute() or not path.is_relative_to(root) or ".." in path.parts:
                paths_valid = False
                continue
            relative = path.relative_to(root).as_posix()
            module = name.replace(".", "/")
            paths_valid &= relative in (module + ".py", module + ".pyc", module + "/__init__.py", module + "/__init__.pyc")
        checks.truth("loaded module paths match frozen module names", paths_valid)
        for key in ("project_id", "script_id", "execution_id"):
            checks.truth("declared native " + key, isinstance(receipt.get(key), str) and bool(receipt[key].strip()))
        checks.truth("workspace belongs to declared dedicated project", workspace.is_absolute()
                     and workspace.parts[-5:] == ("Library", "Application Support", _NATIVE_IDENTIFIER, "projects", receipt.get("project_id")))
        checks.truth("complete actual native proof flags", type(proof.get("stages")) is int and proof["stages"] == 8
                     and all(proof.get(key) is True for key in ("frozen", "restored_equal", "resource_guard_passed"))
                     and proof.get("source_oracles_available") is False)
        states = receipt["complete_states"]
        state_bytes = json.dumps(states, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        checks.equal("proof state hash matches receipt", proof.get("states_sha256"), receipt.get("states_sha256"))
        checks.truth("proof saved state byte count", type(proof.get("states_bytes")) is int and proof["states_bytes"] == len(state_bytes), proof.get("states_bytes"), len(state_bytes))
        inputs_bytes = json.dumps(receipt["inputs"], sort_keys=True, allow_nan=False).encode()
        checks.equal("native input file digest links actual input object", receipt.get("inputs_sha256"), hashlib.sha256(inputs_bytes).hexdigest())
        checks.equal("actual native execution status", execution.get("status"), "ok")
        checks.equal("actual native execution ID", execution.get("id"), receipt.get("execution_id"))
        checks.equal("raw displayed outputs link complete execution", execution.get("outputs"), receipt.get("outputs"))
        checks.equal("complete execution digest", receipt.get("execution_sha256"), canonical_hash(execution))
        code, stdout = execution.get("code"), execution.get("stdout")
        require(isinstance(code, str) and bool(code) and isinstance(stdout, str), "Native source/stdout are required.")
        checks.equal("executed source digest", receipt.get("source_sha256"), hashlib.sha256(code.encode()).hexdigest())
        markers = [line[len(_NATIVE_MARKER):] for line in stdout.splitlines() if line.startswith(_NATIVE_MARKER)]
        require(len(markers) == 1, "Exactly one actual native proof marker is required.")
        checks.equal("stdout proof links complete execution", json.loads(markers[0]), proof)
        old, current = receipt.get("runtime_pids"), receipt.get("current_runtime_pids")
        valid_pids = (isinstance(old, list) and isinstance(current, list) and old and current
                      and all(type(pid) is int and pid > 0 for pid in [*old, *current])
                      and len(set(old)) == len(old) and len(set(current)) == len(current))
        checks.truth("old and current runtime inventories valid", valid_pids, {"old": old, "current": current})
        require(valid_pids, "Nonempty unique positive runtime PID inventories are required.")
        checks.truth("historical execution worker belongs to original runtime", type(proof.get("worker_pid")) is int and proof["worker_pid"] in old)
        checks.truth("runtime IDs obey native readback or restart relation",
                     not (set(old) & set(current)) if receipt["status"] == "native-restarted" else set(old) == set(current))
    except (KeyError, TypeError, ValueError) as error:
        checks.truth("complete coherent native receipt", False, str(error))
    return checks.finish(scope="Internal consistency of the separately produced native tool receipt; this does not authenticate arbitrary JSON or replace native Run/source/restart evidence.")


def verify(receipt):
    states, outputs, inputs = receipt["complete_states"], receipt["outputs"], receipt["inputs"]
    if set(states) != set(METHODS) or len(outputs) != len(METHODS):
        raise ValueError("Exactly eight complete saved states and actual native outputs are required.")
    if not set(METHODS[:6]).issubset(inputs) or "pool_tables" not in inputs:
        raise ValueError("Actual native source inputs and full source-pool tables are required.")
    results = {}
    for method, output in zip(METHODS, outputs):
        try:
            shown = table_frame(output)
            if method in METHODS[:3]:
                result = discrete(method, states[method], inputs[method], shown)
            elif method.endswith("_delta"):
                result = delta(method, states[method], inputs[method], shown)
            elif method == "passive":
                result = passive(states[method], states, inputs, shown)
            else:
                result = lincom(states[method], states, inputs, shown)
            result["actual_output_sha256"] = canonical_hash(output)
            result["actual_output"] = output
            result["state_sha256"] = canonical_hash(states[method])
            result["input_sha256"] = canonical_hash(inputs.get(method, inputs["normal_delta"]))
            results[method] = result
        except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError, ZeroDivisionError, np.linalg.LinAlgError) as error:
            results[method] = {"pass": False, "error": str(error), "error_type": type(error).__name__}
    state_link = {"name": "native receipt states-file hash matches its full saved states",
                  "pass": receipt.get("states_sha256") == canonical_hash(states),
                  "actual": receipt.get("states_sha256"), "reference": canonical_hash(states)}
    coherence = native_coherence(receipt)
    native = coherence is not None and coherence["pass"]
    coherent = coherence is None or coherence["pass"]
    return {"status": "pass" if coherent and state_link["pass"] and all(result["pass"] for result in results.values()) else "fail",
            "receipt_checks": [state_link], "native_receipt_coherence": coherence,
            "evidence_kind": "native-receipt-numerical-acceptance" if native else "rejected-native-receipt" if coherence is not None else "source-fixture-only",
            "method_count": len(results), "methods": results, "source_urls": SOURCES,
            "states_sha256": canonical_hash(states), "inputs_sha256": canonical_hash(inputs),
            "outputs_sha256": canonical_hash(outputs), "execution_id": receipt.get("execution_id"),
            "native_receipt_status": receipt.get("status"),
            "scope": ("Independent numerical acceptance from actual-native saved inputs/states/raw tables; native Run, pinned frozen modules and quit/reopen provenance remain separate receipt evidence." if native else "Source-fixture numerical smoke or rejected native receipt; no actual native execution is established by this report.") + " Separate source posterior-law tests do not establish arbitrary finite-chain convergence or vendor execution parity.",
            "infinite_df_representation": "null denotes the complete-data normal limit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = json.loads(args.native_receipt.read_text())
        report = verify(receipt)
        report["native_receipt"] = {"path": str(args.native_receipt.resolve()),
                                    "file_sha256": hashlib.sha256(args.native_receipt.read_bytes()).hexdigest(),
                                    "canonical_json_sha256": canonical_hash(receipt)}
    except (OSError, KeyError, ValueError, TypeError) as error:
        report = {"status": "fail", "error": str(error), "error_type": type(error).__name__}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False))
    print(json.dumps({"status": report["status"], "method_count": report.get("method_count", 0),
                      "checks": sum(value.get("check_count", 0) for value in report.get("methods", {}).values())}))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
