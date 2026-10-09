"""Complete repeated-residual Gaussian models and optimizer-free saved inference."""

from __future__ import annotations

import copy
import json
import math
from numbers import Integral, Real

import pandas as pd
import torch

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.meta.common import checksum, critical, level_check, probability
from openecon.econometrics.meta.dependent import _labels
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.engines.distributions import chi2_sf
from openecon.resources import plan_workspace

from . import repeated_kernels as kernels

SCHEMA = "openecon.repeated.gls.v1"
SOURCE = "https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/gls.html"
MAX_STATE_BYTES = 8 * 1024**2
STATE_KEYS = {"schema", "roles", "terms", "x", "y", "subject_labels", "subjects",
              "groups", "occasion_labels", "occasions", "levels", "row_labels",
              "structure", "method", "level", "max_work", "fit", "checksum"}


def _seal(state):
    saved = copy.deepcopy(state)
    saved.pop("checksum", None)
    try:
        encoded = json.dumps(saved, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode()) > MAX_STATE_BYTES:
            raise AnalysisError("resource_limit", "Complete repeated GLS state exceeds 8 MiB.")
        saved["checksum"] = checksum(saved)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError("invalid_result_state", "Save complete finite JSON values.") from exc
    return saved


def _json(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {name: _json(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool) or isinstance(left, str) or isinstance(right, str):
        return type(left) is type(right) and left == right
    if isinstance(left, Real) and isinstance(right, Real):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=2e-7, abs_tol=2e-9)
    return left == right


def _budget(max_work):
    if isinstance(max_work, bool) or not isinstance(max_work, Integral) or not 1 <= max_work <= 2_000_000_000:
        raise AnalysisError("invalid_budget", "max_work must be an integer in 1..2,000,000,000.")
    return int(max_work)


def _roles(y, x, subject, occasion):
    if x is None:
        x = []
    if not isinstance(x, (list, tuple)):
        raise AnalysisError("invalid_spec", "x must be an ordered list of numeric column names.")
    names = [y, *x, subject, occasion]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise AnalysisError("invalid_spec", "Selected roles must be distinct nonempty column names.")
    if "Intercept" in x or len(x) > 5:
        raise AnalysisError("invalid_spec", "Use at most five numeric predictors; Intercept is reserved.")
    return {"y": y, "x": list(x), "subject": subject, "occasion": occasion}


def _geometry(x, y, groups, occasions, levels, structure):
    n, p = x.shape
    q, g = len(levels), int(groups.max()) + 1
    if not 2 <= q <= 6 or not 4 <= g <= 64 or not 1 <= p <= 6 or not p < n <= 256:
        raise AnalysisError("resource_limit", "Require 2..6 occasions, 4..64 subjects, n<=256 and n>p.")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise AnalysisError("nonfinite_values", "Selected outcomes/design values must be finite.")
    if max(float(x.abs().max()), float(y.abs().max())) > 1e6:
        raise AnalysisError("numeric_domain", "Rescale outcomes/design to lie within +/-1e6.")
    if not torch.equal(x[:, 0], torch.ones(n, dtype=torch.float64)):
        raise AnalysisError("invalid_spec", "The first design column must be the fixed intercept.")
    scaled = x.clone()
    for j in range(1, p):
        sd = (x[:, j] - x[:, j].mean()).square().mean().sqrt()
        if float(sd) <= 1e-8:
            raise AnalysisError("singular_design", "Numeric predictors must vary; rescale tiny units explicitly.")
        scaled[:, j] = (x[:, j] - x[:, j].mean()) / sd
    singular = torch.linalg.svdvals(scaled)
    if float(singular[-1]) <= 1e-8 * float(singular[0]):
        raise AnalysisError("singular_design", "The scaled fixed design is singular or ill conditioned.")
    pairs = list(zip(groups.tolist(), occasions.tolist()))
    if len(set(pairs)) != n:
        raise AnalysisError("duplicate_observation", "Each subject/occasion pair must appear exactly once.")
    observed = [set(occasions[groups == j].tolist()) for j in range(g)]
    if any(sum(k in rows for rows in observed) < 4 for k in range(q)):
        raise AnalysisError("insufficient_occasion_support", "Every occasion needs at least four independent subjects.")
    if structure == "unstructured" and any(sum(a in rows and b in rows for rows in observed) < 4 for a in range(q) for b in range(a)):
        raise AnalysisError("insufficient_pair_support", "Every unstructured occasion pair needs four independent subjects.")
    distances = [abs(levels[a] - levels[b]) for rows in observed for a in rows for b in rows if a != b]
    if structure in ("ar1", "cs") and not distances:
        raise AnalysisError("unidentified_covariance", "Correlated structures require repeated observations within subjects.")
    if structure == "ar1" and math.gcd(*distances) != 1:
        raise AnalysisError("unsupported_occasion_grid", "This implementation admits signed AR1 calendars with observed integer-lag gcd one; coarser grids require a separate parameterization.")
    if float(y.std()) <= 1e-8:
        raise AnalysisError("boundary_solution", "A varying outcome is required.")
    plan_workspace("repeated Gaussian residual likelihood", {"dense_covariance_derivatives": 128 * n*n * (p + q*(q+1)//2 + 4), "sample": 64*n*(p+4)}, budget_bytes=256*1024**2)


def _sample(data, roles, structure):
    if isinstance(data, Dataset):
        raise AnalysisError("unsupported_dataset", "Repeated GLS requires a complete resident frame.")
    raw = _coerce_frame(data)
    n = len(raw)
    if not 2 <= n <= 256:
        raise AnalysisError("resource_limit", "Repeated GLS requires at most 256 resident observations.")
    names = [roles["y"], *roles["x"], roles["subject"], roles["occasion"]]
    if any(list(raw.columns).count(name) != 1 for name in names):
        raise AnalysisError("invalid_columns", "Each selected column must exist exactly once.")
    frame = raw.loc[:, names].copy()
    if bool(frame.isna().any().any()):
        raise AnalysisError("missing_values", "Selected observations must be complete; no rows are dropped.")
    numeric = [roles["y"], *roles["x"], roles["occasion"]]
    if any(pd.api.types.is_bool_dtype(frame[name].dtype) or not pd.api.types.is_numeric_dtype(frame[name].dtype) or pd.api.types.is_complex_dtype(frame[name].dtype) for name in numeric):
        raise AnalysisError("invalid_numeric", "Outcome, predictors and occasions must be real numeric columns.")
    subjects = _labels(frame[roles["subject"]], name="subject")
    unique = list(dict.fromkeys(json.dumps(value, allow_nan=False) for value in subjects))
    mapping = {value: j for j, value in enumerate(unique)}
    groups = torch.tensor([mapping[json.dumps(value, allow_nan=False)] for value in subjects], dtype=torch.int64)
    occasions = _numeric(frame[roles["occasion"]], roles["occasion"])
    if bool((occasions != occasions.round()).any()) or float(occasions.abs().max()) > 1_000_000:
        raise AnalysisError("invalid_occasion", "Occasions must be finite integer times within +/-1,000,000.")
    times = [int(value) for value in occasions.tolist()]
    levels = sorted(set(times))
    occasion_index = torch.tensor([levels.index(value) for value in times], dtype=torch.int64)
    x = torch.stack([torch.ones(n, dtype=torch.float64), *[_numeric(frame[name], name) for name in roles["x"]]], 1)
    y = _numeric(frame[roles["y"]], roles["y"])
    _geometry(x, y, groups, occasion_index, levels, structure)
    return {"x": x.tolist(), "y": y.tolist(), "subject_labels": subjects,
            "subjects": [json.loads(value) for value in unique], "groups": groups.tolist(),
            "occasion_labels": times, "occasions": occasion_index.tolist(), "levels": levels,
            "row_labels": _labels(frame.index, name="original row")}


def _arrays(state):
    return (torch.tensor(state["x"], dtype=torch.float64), torch.tensor(state["y"], dtype=torch.float64),
            torch.tensor(state["groups"], dtype=torch.int64), torch.tensor(state["occasions"], dtype=torch.int64))


def _load(result):
    state = result.attrs.get("state") if isinstance(result, TableSet) else result
    try:
        if not isinstance(state, dict) or set(state) != STATE_KEYS or state["schema"] != SCHEMA:
            raise ValueError("schema")
        state = copy.deepcopy(state)
        saved_hash = state.pop("checksum")
        if checksum(state) != saved_hash:
            raise ValueError("checksum")
        state["checksum"] = saved_hash
        if len(json.dumps(state, allow_nan=False).encode()) > MAX_STATE_BYTES:
            raise ValueError("size")
        roles = _roles(**state["roles"])
        if roles != state["roles"] or state["terms"] != ["Intercept", *roles["x"]]:
            raise ValueError("roles")
        if state["structure"] not in ("cs", "ar1", "diagonal", "unstructured") or state["method"] not in ("ML", "REML"):
            raise ValueError("model")
        level_check(state["level"])
        _budget(state["max_work"])
        n = len(state["y"])
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
               for value in state["y"]):
            raise ValueError("response type")
        if any(not isinstance(row, list) or any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                                               for value in row) for row in state["x"]):
            raise ValueError("design type")
        if any(len(state[key]) != n for key in ("x", "subject_labels", "groups", "occasion_labels", "occasions", "row_labels")):
            raise ValueError("alignment")
        subjects = _labels(state["subject_labels"], name="subject")
        encoded = list(dict.fromkeys(json.dumps(value, allow_nan=False) for value in subjects))
        if state["subjects"] != [json.loads(value) for value in encoded]:
            raise ValueError("subject universe")
        expected_groups = [encoded.index(json.dumps(value, allow_nan=False)) for value in subjects]
        if any(type(value) is not int for key in ("groups", "occasions", "occasion_labels", "levels") for value in state[key]):
            raise ValueError("index types")
        if state["groups"] != expected_groups or state["levels"] != sorted(set(state["occasion_labels"])):
            raise ValueError("geometry")
        if state["occasions"] != [state["levels"].index(value) for value in state["occasion_labels"]] or max(abs(value) for value in state["levels"]) > 1_000_000:
            raise ValueError("occasions")
        _labels(state["row_labels"], name="original row")
        x, y, groups, occasions = _arrays(state)
        if x.ndim != 2 or x.shape[1] != len(state["terms"]):
            raise ValueError("design")
        _geometry(x, y, groups, occasions, state["levels"], state["structure"])
        fit = state["fit"]
        replay = _json(kernels.replay(x, y, groups, occasions, state["levels"], state["structure"], state["method"], fit["theta"], state["max_work"]))
        if set(fit) != set(replay) | {"optimizer"} or not all(_same(fit[key], value) for key, value in replay.items()):
            raise ValueError("scientific receipts")
        optimizer = fit["optimizer"]
        if not isinstance(optimizer, dict) or optimizer.get("converged") is not True or optimizer.get("max_work") != state["max_work"] or not isinstance(optimizer.get("work_used"), int) or not 0 <= optimizer["work_used"] <= state["max_work"]:
            raise ValueError("optimizer receipt")
        if isinstance(result, TableSet):
            expected = _result(state)
            if list(result) != list(expected) or result.title != expected.title or not _same(result.attrs, expected.attrs):
                raise ValueError("displayed result metadata")
            for name, wanted in expected.items():
                actual = result[name]
                if (list(actual.columns) != list(wanted.columns) or list(actual.index) != list(wanted.index)
                        or list(map(str, actual.dtypes)) != list(map(str, wanted.dtypes))
                        or not _same(actual.values.tolist(), wanted.values.tolist())
                        or not _same(actual.attrs, wanted.attrs)):
                    raise ValueError("displayed table disagrees with state")
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, RuntimeError, AnalysisError, OverflowError) as exc:
        raise AnalysisError("invalid_result_state", "Use complete coherent repeated GLS state with original sample and covariance geometry.") from exc
    return state


def _estimates(estimate, covariance, labels, level, null=None):
    estimate = torch.tensor(estimate, dtype=torch.float64)
    covariance = torch.tensor(covariance, dtype=torch.float64)
    se = covariance.diagonal().sqrt()
    null = torch.zeros_like(estimate) if null is None else null
    z = (estimate - null) / se
    c = critical(level, "z", None)
    return table({"term": labels, "estimate": estimate.tolist(), "std_error": se.tolist(),
                  "statistic": z.tolist(), "p_value": [probability(float(value), "z", None) for value in z],
                  "ci_low": (estimate-c*se).tolist(), "ci_high": (estimate+c*se).tolist()})


def _result(state):
    fit, terms = state["fit"], state["terms"]
    beta = torch.tensor(fit["beta"], dtype=torch.float64)
    x, y, _, _ = _arrays(state)
    covariance_terms = fit["covariance_terms"]
    fitted = x @ beta
    covariance_parameters = _estimates(fit["covariance_parameters"], fit["covariance_parameter_covariance"], covariance_terms, state["level"])
    covariance_parameters = covariance_parameters.drop(columns=["statistic", "p_value"])
    tables = {"coefficients": _estimates(fit["beta"], fit["covariance"], terms, state["level"]),
              "covariance": table(fit["covariance"], columns=terms, index=terms),
              "covariance_parameters": covariance_parameters,
              "covariance_parameter_covariance": table(fit["covariance_parameter_covariance"], columns=covariance_terms, index=covariance_terms),
              "residual_covariance": table(fit["residual_covariance"], columns=list(range(len(y))), index=list(range(len(y)))),
              "fitted": table({"position": list(range(len(y))), "row_label": state["row_labels"], "subject": state["subject_labels"],
                                "occasion": state["occasion_labels"], "observed": y.tolist(), "mean": fitted.tolist(), "residual": (y-fitted).tolist()}),
              "fit": table([{"loglik_ml": fit["loglik_ml"], "loglik_reml": fit["loglik_reml"], "residual_Q": fit["rss"],
                             "structure": state["structure"], "method": state["method"], "n_observations": len(y),
                             "n_subjects": len(state["subjects"]), "n_occasions": len(state["levels"]),
                             "inference": fit["inference"], "converged": fit["optimizer"]["converged"],
                             "work_used": fit["optimizer"]["work_used"], "max_work": state["max_work"]}])}
    if state["method"] == "ML" and "joint_covariance" in fit:
        labels = [*["fixed:" + term for term in terms], *["residual:" + term for term in covariance_terms]]
        tables["joint_covariance"] = table(fit["joint_covariance"], columns=labels, index=labels)
    return TableSet(tables, title="Repeated residual Gaussian GLS", procedure="repeated_gls", state=state,
                    structure=state["structure"], method=state["method"], level=state["level"], n_observations=len(y),
                    n_subjects=len(state["subjects"]), inference="asymptotic normal; ML joint OIM / REML fixed GLS plug-in",
                    source=SOURCE, device="cpu", dtype="float64", residual_df=None, optimizer=fit["optimizer"],
                    notes=["Covariance parameter intervals are asymptotic Wald intervals and may cross parameter boundaries.",
                           "REML fixed-effect covariance plugs in estimated residual covariance; KR/Satterthwaite is unavailable.",
                           "Subjects are independent. Missing occasions select the declared global covariance submatrix; no outcomes are imputed."])


@resident_cpu
@torch.no_grad()
def repeated_gls(*, data, y, x=None, subject, occasion, structure="ar1", method="REML", level=.95, max_work=2_000_000_000):
    """Fit four repeated Gaussian residual structures by ML or REML.

    Numeric fixed effects include an intercept. Selected observations must be
    complete; subjects may have absent occasions. No random-effect, weight,
    categorical, Dataset or GPU route is silently substituted. Covariance
    inference is asymptotic; no KR or Satterthwaite correction is claimed.
    """
    if structure not in ("cs", "ar1", "diagonal", "unstructured") or method not in ("ML", "REML"):
        raise AnalysisError("invalid_option", "Use cs/ar1/diagonal/unstructured and ML/REML.")
    roles, level, max_work = _roles(y, x, subject, occasion), level_check(level), _budget(max_work)
    sample = _sample(data, roles, structure)
    arrays = _arrays(sample)
    fit = _json(kernels.fit(*arrays, sample["levels"], structure, method, max_work))
    state = _seal({"schema": SCHEMA, "roles": roles, "terms": ["Intercept", *roles["x"]], **sample,
                   "structure": structure, "method": method, "level": level, "max_work": max_work, "fit": fit})
    return _result(state)


@resident_cpu
@torch.no_grad()
def restore_repeated_gls(state):
    """Validate complete saved sample/likelihood/inference and replay without fitting."""
    return _result(_load(state))


@resident_cpu
@torch.no_grad()
def repeated_gls_predict(result, data, *, level=.95):
    """Saved joint population means with full covariance and asymptotic marginal CI."""
    state, level = _load(result), level_check(level)
    if isinstance(data, Dataset):
        raise AnalysisError("unsupported_dataset", "Saved prediction requires resident numeric profiles.")
    frame = _coerce_frame(data)
    names = state["roles"]["x"]
    if not 1 <= len(frame) <= 256 or any(list(frame.columns).count(name) != 1 for name in names):
        raise AnalysisError("invalid_profiles", "Use 1..256 complete named predictor profiles.")
    labels = _labels(frame.index, unique=True, name="profile")
    if any(not pd.api.types.is_numeric_dtype(frame[name].dtype) or pd.api.types.is_bool_dtype(frame[name].dtype) or pd.api.types.is_complex_dtype(frame[name].dtype) for name in names):
        raise AnalysisError("invalid_profiles", "Prediction columns must be complete real numbers.")
    x = torch.stack([torch.ones(len(frame), dtype=torch.float64), *[_numeric(frame[name], name) for name in names]], 1)
    if float(x.abs().max()) > 1e6:
        raise AnalysisError("numeric_domain", "Rescale prediction profiles to +/-1e6.")
    beta = torch.tensor(state["fit"]["beta"], dtype=torch.float64)
    covariance = x @ torch.tensor(state["fit"]["covariance"], dtype=torch.float64) @ x.T
    return TableSet({"means": _estimates((x@beta).tolist(), covariance.tolist(), labels, level),
                     "covariance": table(covariance.tolist(), columns=labels, index=labels)},
                    title="Saved repeated GLS population means", procedure="repeated_gls_predict", state=state,
                    level=level, refitted=False, prediction_target="population mean; no future residual variance", inference="asymptotic normal")


@resident_cpu
@torch.no_grad()
def repeated_gls_contrast(result, contrast, *, null=None, level=.95):
    """Saved scalar/joint fixed-effect contrasts with complete covariance and nonzero nulls."""
    from openecon.econometrics.meta.dependent_post import _contrasts, _null

    state, level = _load(result), level_check(level)
    if isinstance(contrast, dict):
        if any(name not in state["terms"] for name in contrast):
            raise AnalysisError("invalid_contrasts", "Contrast keys must name fitted coefficients.")
        contrast = pd.DataFrame([{name: contrast.get(name, 0.) for name in state["terms"]}], index=["contrast"])
    labels, matrix = _contrasts(contrast, state["terms"])
    hypothesis = _null(null, labels)
    beta = torch.tensor(state["fit"]["beta"], dtype=torch.float64)
    covariance = matrix @ torch.tensor(state["fit"]["covariance"], dtype=torch.float64) @ matrix.T
    estimate = matrix @ beta
    delta = estimate - hypothesis
    statistic = float(delta @ torch.linalg.solve(covariance, delta))
    estimates = _estimates(estimate.tolist(), covariance.tolist(), labels, level, hypothesis)
    estimates["null"] = hypothesis.tolist()
    return TableSet({"contrasts": estimates, "covariance": table(covariance.tolist(), columns=labels, index=labels),
                     "joint_test": table([{"statistic": statistic, "distribution": "chi2", "df": len(labels), "p_value": chi2_sf(statistic, len(labels))}])},
                    title="Saved repeated GLS fixed contrasts", procedure="repeated_gls_contrast", state=state, level=level,
                    refitted=False, inference="asymptotic normal/chi2")
