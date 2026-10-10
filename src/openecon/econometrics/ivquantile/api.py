"""Public scalar IVQR profiles, complete replay and structural quantile queries."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.econometrics.state_lifecycle import _encoded_json_admission, _state_copy_admission
from openecon.engines.distributions import normal_ppf
from openecon.resources import plan_workspace
from .common import (
    MAX_JSON, MAX_N, SCHEMA, controls, count, data_state, digest, fail,
    geometry, names, real, restore_source, seal,
)
from .kernels import (
    accepted_runs, close, coefficient_rows, identified_covariance,
    profile_record, replay_record,
)


def _search(config, kx, evaluate):
    """Deterministic search trace; replay invokes certificates, never QR optimization."""
    coarse = [evaluate(alpha) for alpha in config["grid"]]
    chosen = min(range(len(coarse)), key=lambda i: (coarse[i]["criterion"], i))
    bracket = None
    if config["inference"] == "identified":
        minima = [i for i in range(1, len(coarse) - 1)
                  if coarse[i]["criterion"] < coarse[i - 1]["criterion"]
                  and coarse[i]["criterion"] < coarse[i + 1]["criterion"]]
        if chosen in (0, len(coarse) - 1) or minima != [chosen]:
            fail("Conventional inference needs one strict interior coarse-profile minimum; widen/refine the declared search or use weak mode.",
                 "ambiguous_identification")
        low, high = config["grid"][chosen - 1], config["grid"][chosen + 1]
        tolerance = config["refinement_tolerance"]
        if len(coarse[0]["coefficients"]) - kx == 1:
            left, right = coarse[chosen - 1], coarse[chosen + 1]
            roots = [(i, i + 1) for i in range(len(coarse) - 1)
                     if coarse[i]["coefficients"][kx] * coarse[i + 1]["coefficients"][kx] <= 0]
            # A single scanned sign-changing root supplies a nonnegative criterion minimum.
            if len(roots) != 1:
                fail("Identified scalar-instrument inference needs one scanned gamma-root bracket.",
                     "ambiguous_identification")
            li, ri = roots[0]
            left, right = coarse[li], coarse[ri]
            low, high = left["alpha"], right["alpha"]
            while high - low > tolerance:
                mid = evaluate((low + high) / 2)
                if left["coefficients"][kx] * mid["coefficients"][kx] <= 0:
                    high, right = mid["alpha"], mid
                else:
                    low, left = mid["alpha"], mid
            selected = min([left, right], key=lambda r: r["criterion"])
            bracket = dict(lower=low, upper=high, width=high - low,
                           interpretation="scanned sign-changing QR-gamma root bracket; unscanned roots unknown")
        else:
            # The declared local unimodality assumption is not a global optimizer certificate.
            ratio = (math.sqrt(5) - 1) / 2
            aa, bb = high - ratio * (high - low), low + ratio * (high - low)
            ra, rb = evaluate(aa), evaluate(bb)
            while high - low > tolerance:
                if ra["criterion"] < rb["criterion"]:
                    high, bb, rb = bb, aa, ra
                    aa = high - ratio * (high - low)
                    ra = evaluate(aa)
                else:
                    low, aa, ra = aa, bb, rb
                    bb = low + ratio * (high - low)
                    rb = evaluate(bb)
            selected = min([ra, rb], key=lambda r: r["criterion"])
            bracket = dict(lower=low, upper=high, width=high - low,
                           interpretation="local golden-section bracket conditional on declared unimodality; no global certificate")
    else:
        selected = coarse[chosen]
    return coarse, selected, dict(coarse_selected=chosen, refinement=bracket,
                                 optimization_domain="declared bounded alpha grid with optional local refinement",
                                 unscanned_parameters="unknown")


def _joint(y, d, x, w, a, selected, config, search):
    if config["inference"] == "weak":
        return None
    result = identified_covariance(y, d, x, w, a, selected, config)
    se = math.sqrt(result["covariance"][0][0])
    if search["refinement"]["width"] > 0.01 * se:
        fail("Profile refinement error is not negligible relative to conventional SE.", "numerical_domain")
    return result


def _result(state):
    cfg, spec = state["config"], state["spec"]
    selected = state["evaluations"][state["selected_evaluation"]]
    kx = len(spec["x"]) + int(cfg["intercept"])
    values = [selected["alpha"], *selected["coefficients"][:kx]]
    cov = None if state["joint"] is None else state["joint"]["covariance"]
    profile_rows = []
    for i, record in enumerate(state["evaluations"][:len(cfg["grid"])]):
        profile_rows.append(dict(alpha=record["alpha"], criterion=record["criterion"],
                                 qr_objective=record["qr_objective"], weak_id_statistic=record["statistic"],
                                 p_value=record["p_value"], accepted=state["accepted"][i],
                                 bandwidth=record["bandwidth"], density_support=record["density_support"]))
    tables = {
        "coefficients": table(coefficient_rows(state["terms"], values, cov, cfg["confidence"])),
        "weak_id_grid": table(profile_rows),
        "accepted_grid_runs": table(state["runs"], columns=["first_index", "last_index", "first_alpha", "last_alpha",
                                                            "lower_grid_boundary", "upper_grid_boundary", "between_points", "outer_tails"]),
        "selected_nuisance_coefficients": table(
            coefficient_rows([*state["terms"][1:], *spec["instruments"]], selected["coefficients"],
                             selected["covariance"], cfg["confidence"])),
        "selected_nuisance_covariance": table(selected["covariance"],
                                              columns=[*state["terms"][1:], *spec["instruments"]],
                                              index=[*state["terms"][1:], *spec["instruments"]]),
    }
    if cov is not None:
        tables["joint_covariance"] = table(cov, columns=state["terms"], index=state["terms"])
    return saved_summary(TableSet(
        tables, title="Scalar instrumental quantile regression",
        state=state, quantile=cfg["quantile"], nobs=len(state["sample_positions"]),
        confidence=cfg["confidence"], input_sha256=digest(state["source"]),
        inference=cfg["inference"], weak_id_distribution="asymptotic chi-square(excluded instrument count), with fitted exogenous nuisance",
        point_estimate_status=("minimum among declared grid candidates; uniqueness and identification unestablished"
                               if cfg["inference"] == "weak" else
                               "locally refined profile point under declared identification; unscanned minima unknown"),
        weak_id_df=len(spec["instruments"]), accepted_grid_points=sum(state["accepted"]),
        empty_tested_grid=not any(state["accepted"]), disconnected_tested_grid=len(state["runs"]) > 1,
        outer_tails="unknown", between_grid_points="untested; runs do not certify intervals",
        sample="iid independent observations; rank similarity, exclusion, monotone structural quantile and smooth positive density are declared assumptions",
        identified_assumptions=("local strong identification, consistent profile search and local unimodality for overidentification"
                                if cfg["inference"] == "identified" else None),
        device="cpu", dtype="float64", vendor_parity_validated=False,
        unsupported=["multiple endogenous variables", "weights", "clusters", "panel", "Dataset", "GPU",
                     "continuous confidence-set boundaries", "certified unbounded tails", "cross-quantile process inference"],
        notes=["Weak-ID inversion tests alpha candidates while refitting beta and gamma at every null.",
               "QR nuisance Wald intervals refer to the adjusted-outcome regression, not structural alpha inference.",
               "A checksum detects corruption and is not authentication; restore replays LP optimality and every numerical covariance."]
    ))


@resident_cpu
def ivqreg(*, data, y: str, endogenous: str, instruments, grid,
           x=None, quantile: float = 0.5, confidence: float = 0.95,
           bandwidth: float | None = None, inference: str = "weak",
           intercept: bool = True, missing: str = "raise", device: str = "cpu",
           weights=None, cluster=None, max_work: int = 10**12,
           max_evaluations: int = 512, refinement_tolerance: float = 1e-6) -> TableSet:
    """Fit scalar iid inverse QR and invert genuine weak-ID tests at declared alpha points.

    At every alpha run QR(Y-D*alpha on X,Z), including fitted exogenous nuisance.
    Full Powell rectangle bread and iid quantile score covariance give the
    excluded-coefficient Wald statistic with asymptotic chi-square(q) law at
    the true alpha, even under weak instruments (CHJ2007). Finite accepted
    points/runs do not certify between-point boundaries or unknown outer tails.
    Default weak mode never attaches structural Wald SE/CI. Identified mode
    declares local strong identification/consistent search and requires one
    interior refined profile minimum, complete joint influence covariance and
    negligible refinement error. A finite data rank is not proof of strong ID.
    Numeric resident sample only; all inputs, dtype/index/physical positions,
    every profile and portable LP/covariance state are saved without thinning.
    """
    cfg = controls(grid=grid, quantile=quantile, confidence=confidence,
                   bandwidth=bandwidth, inference=inference, intercept=intercept,
                   missing=missing, device=device, weights=weights, cluster=cluster,
                   max_work=max_work, max_evaluations=max_evaluations,
                   refinement_tolerance=refinement_tolerance)
    source, spec, work = data_state(data, y, x, endogenous, instruments, cfg)
    yy, dd, xx, zz, ww, aa, positions = geometry(data, spec, cfg)
    records = []

    def evaluate(alpha):
        if len(records) == cfg["max_evaluations"]:
            fail("The declared IVQR profile evaluation budget is exhausted.", "work_budget_exceeded")
        record = profile_record(yy, dd, ww, aa, xx.shape[1], alpha, cfg)
        records.append(record)
        return record

    coarse, selected, search = _search(cfg, xx.shape[1], evaluate)
    accepted, runs = accepted_runs(coarse, cfg["confidence"])
    joint = _joint(yy, dd, xx, ww, aa, selected, cfg, search)
    terms = [spec["endogenous"], *(["_cons"] if cfg["intercept"] else []), *spec["x"]]
    selected_index = next(i for i, r in enumerate(records) if r is selected)
    state = seal(dict(schema=SCHEMA, source=source, spec=spec, config=cfg,
                      sample_positions=positions, worst_declared_work=work,
                      evaluations=records, selected_evaluation=selected_index,
                      terms=terms, search=search, accepted=accepted, runs=runs, joint=joint))
    if len(json.dumps(state, allow_nan=False).encode()) > MAX_JSON:
        fail("Complete IVQR state exceeds the 32 MiB envelope.", "resource_limit")
    return _result(state)


def _extract(state):
    if isinstance(state, TableSet):
        state = state.attrs.get("state")
    if isinstance(state, str):
        _encoded_json_admission(state, limit=MAX_JSON, operation="IVQR public JSON decoding")
        try:
            state = json.loads(state, parse_constant=lambda _: fail("Nonfinite saved JSON.", "invalid_state"))
        except (ValueError, RecursionError):
            fail("IVQR requires finite JSON.", "invalid_state")
    if isinstance(state, Mapping) and state.get("schema") == "openecon.summary.v1":
        state = state.get("attrs", {}).get("state")
    if not isinstance(state, Mapping):
        fail("Supply complete IVQR state, summary JSON or fitted TableSet.", "invalid_state")
    _state_copy_admission(state, operation="IVQR complete saved-state canonical copy")
    try:
        pending = [(state, 0)]
        size = 0
        while pending:
            item, depth = pending.pop()
            if depth > 18:
                fail("IVQR saved state nesting exceeds its envelope.", "invalid_state")
            size += 64
            if isinstance(item, dict):
                if len(item) > 32:
                    fail("IVQR saved object exceeds its envelope.", "invalid_state")
                pending.extend((v, depth + 1) for v in item.values())
            elif isinstance(item, list):
                if len(item) > MAX_N:
                    fail("IVQR saved array exceeds its envelope.", "invalid_state")
                pending.extend((v, depth + 1) for v in item)
            elif isinstance(item, str):
                size += len(item.encode())
            if size > MAX_JSON * 8:
                fail("IVQR saved state exceeds its object envelope.", "resource_limit")
        plan_workspace("IVQR canonical saved-state copy", {"complete JSON, typed state and copy": size * 3})
        encoded = json.dumps(state, allow_nan=False)
        if len(encoded.encode()) > MAX_JSON:
            fail("IVQR state exceeds 32 MiB.", "resource_limit")
        # Canonical copy removes aliases between caller state and restored output.
        state = json.loads(encoded)
    except (ValueError, TypeError, RecursionError):
        fail("IVQR saved state must be bounded finite JSON.", "invalid_state")
    fields = {"schema", "source", "spec", "config", "sample_positions", "worst_declared_work",
              "evaluations", "selected_evaluation", "terms", "search", "accepted", "runs", "joint", "sha256"}
    if set(state) != fields or state["schema"] != SCHEMA:
        fail("Invalid IVQR complete state schema.", "invalid_state")
    raw = dict(state)
    sha = raw.pop("sha256")
    if digest(raw) != sha:
        fail("IVQR state checksum disagrees.", "invalid_state")
    return state


@resident_cpu
def ivqreg_restore(state) -> TableSet:
    """Semantically replay every saved QR LP and full covariance without optimization.

    Validate complete source dtype/index, sample, deterministic search,
    primal/dual LP optimality, all density information/covariance blocks,
    weak-ID tests/topology and identified influence covariance. Generic summary
    tables are rebuilt from replayed state; saved optimizer counts are historical
    bounded diagnostics and are not claimed to be reproduced iteration by iteration.
    """
    value = _extract(state)
    try:
        cfg = controls(**value["config"])
        if digest(cfg) != digest(value["config"]):
            fail("Saved control normalization disagrees.", "invalid_state")
        spec = value["spec"]
        if not isinstance(spec, dict) or set(spec) != {"y", "x", "endogenous", "instruments"}:
            fail("Invalid saved IVQR specification.", "invalid_state")
        n = count(value["source"]["n_original"], "saved n_original", 1, MAX_N)
        width = len(spec["x"]) + len(spec["instruments"]) + int(cfg["intercept"])
        plan_workspace("IVQR state replay admission", {
            "bounded source, index and all profiles": n * 2048 + cfg["max_evaluations"] * (width + 1)**2 * 512,
        })
        data = restore_source(value["source"])
        source, normal_spec, work = data_state(data, spec["y"], spec["x"], spec["endogenous"], spec["instruments"], cfg)
        if digest(source) != digest(value["source"]) or normal_spec != spec or work != value["worst_declared_work"]:
            fail("Saved source dtype/input identity or work admission disagrees.", "invalid_state")
        yy, dd, xx, zz, ww, aa, positions = geometry(data, spec, cfg)
        if positions != value["sample_positions"]:
            fail("Saved physical sample positions disagree.", "invalid_state")
        records = value["evaluations"]
        if not isinstance(records, list) or not len(cfg["grid"]) <= len(records) <= cfg["max_evaluations"]:
            fail("Invalid saved profile count.", "invalid_state")
        if (type(value["selected_evaluation"]) is not int
                or not 0 <= value["selected_evaluation"] < len(records)
                or not isinstance(value["sample_positions"], list)
                or any(type(v) is not int for v in value["sample_positions"])
                or not isinstance(value["accepted"], list)
                or any(type(v) is not bool for v in value["accepted"])):
            fail("Invalid saved selection/sample types.", "invalid_state")
        cursor = 0

        def evaluate(alpha):
            nonlocal cursor
            if cursor == len(records):
                fail("Saved search trace is incomplete.", "invalid_state")
            record = records[cursor]
            close(alpha, record["alpha"], "search alpha")
            replay_record(yy, dd, ww, aa, xx.shape[1], record, cfg)
            cursor += 1
            return record

        coarse, selected, search = _search(cfg, xx.shape[1], evaluate)
        accepted, runs = accepted_runs(coarse, cfg["confidence"])
        joint = _joint(yy, dd, xx, ww, aa, selected, cfg, search)
        selected_index = next(i for i, r in enumerate(records) if r is selected)
        terms = [spec["endogenous"], *(["_cons"] if cfg["intercept"] else []), *spec["x"]]
        if (cursor != len(records) or selected_index != value["selected_evaluation"] or search != value["search"]
                or accepted != value["accepted"] or runs != value["runs"] or terms != value["terms"]):
            fail("Saved search selection/grid topology disagrees.", "invalid_state")
        if joint is None:
            if value["joint"] is not None:
                fail("Weak mode must not contain invented structural covariance.", "invalid_state")
        else:
            if not isinstance(value["joint"], dict) or set(joint) != set(value["joint"]):
                fail("Invalid saved joint structural inference.", "invalid_state")
            for key in joint:
                if isinstance(joint[key], str):
                    if joint[key] != value["joint"][key]:
                        fail("Saved inference convention disagrees.", "invalid_state")
                elif key == "profile_derivative":
                    close(joint[key], value["joint"][key], key,
                          units=torch.linalg.vector_norm(dd) / torch.linalg.vector_norm(ww, dim=0))
                elif key == "influence_map":
                    parameter_sd = torch.tensor(joint["covariance"], dtype=torch.float64, device="cpu").diagonal().sqrt()
                    qr_sd = torch.tensor(selected["covariance"], dtype=torch.float64, device="cpu").diagonal().sqrt()
                    close(joint[key], value["joint"][key], key, units=parameter_sd[:, None] / qr_sd[None, :])
                else:
                    close(joint[key], value["joint"][key], key)
        return _result(value)
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
        fail(f"Saved IVQR semantic replay failed: {error}", "invalid_state")


@resident_cpu
def ivqreg_predict(*, result, data, missing: str = "raise", confidence: float | None = None) -> TableSet:
    """No-refit structural conditional quantile and complete identified delta covariance.

    Returns D*alpha+X*beta under declared IVQR identifying assumptions, not an
    assertion about observed Q(Y|D,X). Weak-mode results have no parameter-Wald
    uncertainty. Preserve physical query rows and exact row index; no prediction
    intervals for future observations or interpolation across quantile indices.
    """
    fitted = ivqreg_restore(result)
    state = fitted.attrs["state"]
    cfg, spec = state["config"], state["spec"]
    if not isinstance(data, pd.DataFrame) or data.columns.has_duplicates:
        fail("Prediction needs a resident DataFrame with unique columns.", "unsupported_input")
    if not 1 <= len(data) <= MAX_N or missing not in ("raise", "drop"):
        fail("Prediction requires 1..4096 physical rows and explicit missing policy.")
    confidence = cfg["confidence"] if confidence is None else real(confidence, "confidence", 0, 1, strict=True)
    needed = names([spec["endogenous"], *spec["x"]], "query")
    absent = [v for v in needed if v not in data]
    if absent:
        fail(f"Missing structural query columns: {absent}.", "missing_columns")
    for name in needed:
        series = data[name]
        if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype) or is_complex_dtype(series.dtype):
            fail("Query requires real numeric dtypes.", "invalid_data")
    mask = data[needed].notna().all(axis=1)
    if not bool(mask.all()) and missing == "raise":
        fail("Missing query cells require explicit missing='drop'.", "missing_values")
    positions = mask.to_numpy().nonzero()[0].tolist()
    if not positions:
        fail("No complete query rows remain.", "empty_sample")
    cov = None if state["joint"] is None else state["joint"]["covariance"]
    if cov is not None and len(positions) > 512:
        fail("Complete joint delta query covariance supports at most 512 rows.", "resource_limit")
    plan_workspace("IVQR complete conditional-quantile query", {
        "input, Jacobian and outputs": len(data) * (len(needed) + 6) * 64,
        "full cross-query covariance": 0 if cov is None else len(positions)**2 * 24,
    })
    query = torch.tensor(data.iloc[positions][needed].to_numpy(dtype="float64", na_value=float("nan")),
                         dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(query).all()) or bool((query.abs() > 1e12).any()):
        fail("Query cells must be finite with absolute values <=1e12.", "numerical_domain")
    design = torch.cat([query[:, :1], torch.ones((len(query), 1), dtype=torch.float64, device="cpu"), query[:, 1:]], 1) \
        if cfg["intercept"] else query
    selected = state["evaluations"][state["selected_evaluation"]]
    beta = torch.tensor([selected["alpha"], *selected["coefficients"][:len(state["terms"]) - 1]],
                        dtype=torch.float64, device="cpu")
    estimate = design @ beta
    target_covariance = None if cov is None else design @ torch.tensor(cov, dtype=torch.float64, device="cpu") @ design.T
    if not bool(torch.isfinite(estimate).all()) or (target_covariance is not None
                                                   and not bool(torch.isfinite(target_covariance).all())):
        fail("Structural quantile query overflowed its numerical domain.", "numerical_failure")
    critical = normal_ppf((1 + confidence) / 2)
    rows = []
    for i, position in enumerate(positions):
        if target_covariance is None:
            se = None
        else:
            variance = float(target_covariance[i, i])
            if variance < 0 or (variance == 0 and bool((design[i] != 0).any())):
                fail("Structural query variance is nonpositive; no covariance repair is supplied.", "invalid_covariance")
            se = math.sqrt(variance)
        rows.append(dict(physical_row=position, quantile=float(estimate[i]), std_error=se,
                         ci_lower=None if se is None else float(estimate[i]) - critical * se,
                         ci_upper=None if se is None else float(estimate[i]) + critical * se))
    output = table(rows)
    output.index = data.index[positions].copy()
    return saved_summary(TableSet(
        {"predictions": output}, title="IVQR structural conditional quantile",
        quantile_index=cfg["quantile"], sample_positions=positions,
        confidence=confidence, jacobian=design.tolist(), target_covariance=None if cov is None else target_covariance.tolist(),
        source_state_sha256=state["sha256"], inference=cfg["inference"],
        estimand="structural conditional quantile under declared rank similarity, exclusion and monotonicity",
        conditional="fixed supplied query design", future_observation_interval=False,
    ))
