"""Three-level (multilevel) random-effects meta-analysis and meta-regression.

Effects ``i`` are nested in independent studies ``j``:
``y_ij = x_ij' beta + u_j + w_ij + e_ij`` with ``u_j ~ N(0, sigma2_study)``,
``w_ij ~ N(0, sigma2_effect)`` and known sampling errors ``e_j ~ N(0, S_j)``
(Konstantopoulos 2011, Research Synthesis Methods 2:61-76). The marginal
covariance and the restricted/full likelihoods follow the author documentation
of ``rma.mv(yi, V, random = ~ 1 | study/effect)``:
https://wviechtb.github.io/metafor/reference/rma.mv.html. Profile-likelihood
intervals follow https://wviechtb.github.io/metafor/reference/confint.rma.html;
multilevel I² follows the author note
https://www.metafor-project.org/doku.php/tips:i2_multilevel_multivariate.
"""

from __future__ import annotations

import copy
import json
import math

import pandas as pd
import torch

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import chi2_isf, chi2_sf, f_sf
from openecon.engines.optimize import information_inverse
from openecon.resources import plan_workspace

from . import multilevel_kernels as kernels
from .common import checksum, critical, level_check, probability
from .dependent import _key, _labels

SCHEMA = "openecon.meta.multilevel.v1"
SOURCE = "https://wviechtb.github.io/metafor/reference/rma.mv.html"
PROFILE_SOURCE = "https://wviechtb.github.io/metafor/reference/confint.rma.html"
I2_SOURCE = "https://www.metafor-project.org/doku.php/tips:i2_multilevel_multivariate"
PREDICT_SOURCE = "https://wviechtb.github.io/metafor/reference/predict.rma.html"
MAX_STATE_BYTES = 16*1024**2
MAX_EFFECTS, MAX_STUDIES, MAX_BLOCK, MAX_TERMS, MAX_PADDED = 2000, 1000, 100, 16, 200_000
COMPONENTS = ("study", "effect")
COMPONENT_COLUMNS = ["component", "sigma2", "sigma", "n_levels", "boundary", "std_error", "lrt_statistic", "lrt_df",
                     "lrt_p_value", "lrt_p_value_boundary_mixture", "profile_ci_low", "profile_ci_high"]
STATE_KEYS = {"schema", "device", "dtype", "roles", "sampling", "effect_ids", "study_ids", "studies",
              "sample_positions", "sample_labels", "sample_hash", "terms", "moderators", "intercept",
              "method", "sigma2", "x", "y", "s_blocks", "beta", "covariance", "inference", "residual_df",
              "n_effects", "n_studies", "level", "fit", "solver", "components", "checksum"}


def _seal(state):
    result = copy.deepcopy(state)
    result.pop("checksum", None)
    try:
        encoded = json.dumps(result, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError, OverflowError) as exc:
        raise AnalysisError("invalid_result_state", "State must contain complete finite JSON values.") from exc
    if len(encoded.encode()) > MAX_STATE_BYTES:
        raise AnalysisError("resource_limit", "The complete saved three-level state exceeds 16 MiB.")
    result["checksum"] = checksum(result)
    return result


def _digest(state):
    return checksum({name: state[name] for name in ("roles", "sampling", "effect_ids", "study_ids", "sample_positions",
                                                     "sample_labels", "terms", "x", "y", "s_blocks")})


def _check_blocks(blocks, packing):
    """Admit exactly symmetric SPD within-study blocks with bounded scaling."""
    real = packing.weight[:, :, None]*packing.weight[:, None, :]
    values = blocks*real
    if not bool(torch.isfinite(values).all()) or not torch.equal(values, values.transpose(-1, -2)):
        raise AnalysisError("invalid_covariance", "Supply finite exactly symmetric within-study sampling covariance; no symmetrization is applied.")
    diagonal = blocks.diagonal(dim1=-2, dim2=-1)[packing.mask]
    if bool(((diagonal < 1e-12) | (diagonal > 1e12)).any()) or float(diagonal.max()/diagonal.min()) > 1e12:
        raise AnalysisError("invalid_covariance", "Sampling variances must lie in [1e-12,1e12] and span at most 1e12; rescale explicitly.")
    spectrum = torch.linalg.eigvalsh(blocks/float(diagonal.median()))
    if float(spectrum.min()) <= 0 or float((spectrum.max(-1).values/spectrum.min(-1).values).max()) > 1e10:
        raise AnalysisError("invalid_covariance", "Every within-study sampling covariance block must be SPD with condition ratio at most 1e10.")


def _workspace(n, groups, size, p):
    padded = groups*size*size
    if n > MAX_EFFECTS or groups > MAX_STUDIES or size > MAX_BLOCK or padded > MAX_PADDED:
        raise AnalysisError("resource_limit", f"The three-level route admits <= {MAX_EFFECTS} effects, <= {MAX_STUDIES} studies, <= {MAX_BLOCK} effects per study and <= {MAX_PADDED} padded block entries.")
    chunk = max(1, kernels.CHUNK_ENTRIES//padded)*padded
    return plan_workspace("three-level meta likelihood", {
        "padded_sampling_structure_and_data_blocks": 8*(4*padded+2*groups*size*(p+2)),
        "batched_covariance_factor_inverse_and_products": 8*16*chunk,
        "batched_design_products": 8*8*(chunk//max(1, size))*(p+2),
        "complete_state_tables": 512*(n*(p+4)+padded+1024),
    }).record()


def _cluster(values, name):
    labels = _labels(values, name=name)
    tokens = list(dict.fromkeys(_key(value) for value in labels))
    mapping = {token: j for j, token in enumerate(tokens)}
    return labels, tokens, torch.tensor([mapping[_key(value)] for value in labels], dtype=torch.int64)


def _component_inference(lik, method, theta, loglik, fitted, level, profile, upper):
    """Observed-information SEs, LRTs against each zero component and profile limits."""
    hessian = torch.tensor(fitted["hessian_"+method], dtype=torch.float64)
    free = [a for a in range(2) if theta[a] > 0]
    covariance = [[None, None], [None, None]]
    status = "boundary component(s) fixed; no Wald SE" if len(free) < 2 else "observed information"
    if free:
        try:
            inverse = information_inverse(-hessian[free][:, free])
            for i, a in enumerate(free):
                for j, b in enumerate(free):
                    covariance[a][b] = float(inverse[i, j])
        except KernelError:
            status = "observed information not positive definite; no Wald SE"
    rows, crit = [], chi2_isf(1-level, 1)/2
    for a, name in enumerate(COMPONENTS):
        reduced = lik.profile(method, a, 0.0, upper)
        lrt = max(0.0, 2*(loglik-reduced))
        row = {"component": name, "sigma2": theta[a]*lik.scale, "sigma": math.sqrt(theta[a]*lik.scale),
               "n_levels": lik.packing.groups if a == 0 else lik.n, "boundary": theta[a] == 0,
               "std_error": None if covariance[a][a] is None else math.sqrt(covariance[a][a]),
               "lrt_statistic": lrt, "lrt_df": 1, "lrt_p_value": chi2_sf(lrt, 1) if lrt > 0 else 1.0,
               "lrt_p_value_boundary_mixture": .5*chi2_sf(lrt, 1) if lrt > 0 else 1.0,
               "profile_ci_low": None, "profile_ci_high": None}
        if profile:
            low, high = lik.profile_interval(method, a, theta, loglik, crit, upper)
            row.update(profile_ci_low=low*lik.scale, profile_ci_high=None if high is None else high*lik.scale)
        rows.append(row)
    return rows, covariance, status


@resident_cpu
@torch.no_grad()
def meta_multilevel(*, data, study, yi="yi", vi=None, covariance=None, effect=None, moderators=None,
                    intercept=True, method="REML", inference="z", level=.95, profile=True,
                    device="cpu", weights=None):
    """Fit a three-level random-effects meta-analysis/regression (effects nested in studies).

    V_j = S_j + sigma2_study 11' + sigma2_effect I with known sampling variances
    (vi) or a labelled within-study covariance DataFrame. ML/REML jointly over
    both nonnegative components with all boundary faces compared and a KKT
    check; GLS coefficients with full covariance and z or t(k-p) inference;
    observed-information component SEs, per-component LRTs (naive chi2(1) and
    50:50 boundary mixture) and profile-likelihood intervals; QE, QM, multilevel
    I², AIC/BIC/AICc and checksum-protected state for meta_multilevel_predict
    and meta_cr2. <=2000 effects, <=1000 studies, <=100 effects per study,
    <=16 terms. No weights, missing deletion, categorical expansion, Dataset or
    accelerator route; variance-component uncertainty is not propagated.
    """
    if device != "cpu" or type(device) is not str:
        raise AnalysisError("unsupported_device", "Three-level meta-analysis uses explicit CPU float64 only.")
    if weights is not None:
        raise AnalysisError("unsupported_weights", "Observation weights are unsupported; supply sampling variances or covariance.")
    if isinstance(data, Dataset):
        raise AnalysisError("unsupported_dataset", "Supply an explicitly resident complete table.")
    if method not in ("ML", "REML") or inference not in ("z", "t"):
        raise AnalysisError("invalid_spec", "Use method ML/REML and inference z/t.")
    if type(intercept) is not bool or type(profile) is not bool:
        raise AnalysisError("invalid_spec", "intercept and profile must be booleans.")
    if (vi is None) == (covariance is None):
        raise AnalysisError("invalid_spec", "Declare exactly one of vi (independent sampling variances) or covariance (labelled within-study blocks).")
    if covariance is not None and effect is None:
        raise AnalysisError("invalid_spec", "A labelled covariance requires the effect identifier column.")
    moderators = [] if moderators is None else moderators
    if not isinstance(moderators, list) or any(not isinstance(name, str) or not name for name in moderators):
        raise AnalysisError("invalid_spec", "moderators must be an explicit list of numeric column names.")
    roles = {"study": study, "yi": yi}
    if effect is not None:
        roles["effect"] = effect
    if vi is not None:
        roles["vi"] = vi
    if any(not isinstance(name, str) or not name for name in roles.values()) or len(set([*roles.values(), *moderators])) != len(roles)+len(moderators):
        raise AnalysisError("invalid_spec", "Study, effect, response, variance and moderator roles must be distinct nonempty names.")
    terms = (["Intercept"] if intercept else [])+moderators
    p = len(terms)
    if not 1 <= p <= MAX_TERMS or len(set(terms)) != p:
        raise AnalysisError("resource_limit", f"Declare 1..{MAX_TERMS} distinct numeric coefficients including any intercept.")
    level = level_check(level)
    raw = _coerce_frame(data)
    n = len(raw)
    if not 3 <= n <= MAX_EFFECTS or n <= p+2:
        raise AnalysisError("resource_limit", f"The three-level route needs 3..{MAX_EFFECTS} effects and more effects than coefficients plus two variance components.")
    selected = [*roles.values(), *moderators]
    if any(list(raw.columns).count(name) != 1 for name in selected):
        raise AnalysisError("invalid_columns", "Each selected role must exist exactly once.")
    frame = raw.loc[:, selected].copy()
    if bool(frame.isna().any().any()):
        raise AnalysisError("missing_values", "All selected effects/studies must be complete; no rows are dropped.")
    numeric = [yi, *moderators, *([vi] if vi is not None else [])]
    for name in numeric:
        if pd.api.types.is_bool_dtype(frame[name].dtype):
            raise AnalysisError("non_numeric_column", "Boolean responses, variances or moderators are unsupported.")
    values = {name: _numeric(frame[name], name).clone() for name in numeric}
    if any(float(values[name].abs().max()) > 1e8 for name in [yi, *moderators]):
        raise AnalysisError("numeric_domain", "Effects/moderators must lie in +/-1e8; rescale explicitly.")
    ids = _labels(frame[effect], unique=True, name="effect") if effect is not None else list(range(1, n+1))
    study_ids, tokens, index = _cluster(frame[study], "study")
    groups = len(tokens)
    if not 2 <= groups <= MAX_STUDIES:
        raise AnalysisError("resource_limit", f"Declare 2..{MAX_STUDIES} independent studies.")
    if groups == n:
        raise AnalysisError("unidentified_components", "Every study has one effect; the study and effect components are not separately identified.")
    packing = kernels.Packing(index, groups)
    resource = _workspace(n, groups, packing.size, p)
    if vi is not None:
        blocks = torch.diag_embed(packing.vector(values[vi]))+torch.diag_embed(1-packing.weight)
        sampling = "vi"
    else:
        if not isinstance(covariance, pd.DataFrame) or covariance.shape != (n, n):
            raise AnalysisError("invalid_covariance", "Supply a labelled n-by-n covariance DataFrame; bare matrices are ambiguous.")
        if covariance.index.has_duplicates or covariance.columns.has_duplicates:
            raise AnalysisError("covariance_alignment", "Both covariance axes must contain unique labels.")
        row_labels = _labels(covariance.index, unique=True, name="covariance row")
        col_labels = _labels(covariance.columns, unique=True, name="covariance column")
        if set(map(_key, row_labels)) != set(map(_key, ids)) or set(map(_key, col_labels)) != set(map(_key, ids)):
            raise AnalysisError("covariance_alignment", "Covariance rows and columns must contain exactly the effect labels.")
        aligned = covariance.loc[ids, ids]
        if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) for dtype in aligned.dtypes):
            raise AnalysisError("invalid_covariance", "Covariance entries must be finite real numbers.")
        dense = torch.tensor(aligned.to_numpy(dtype="float64"), dtype=torch.float64)
        if not bool(torch.isfinite(dense).all()) or not torch.equal(dense, dense.T):
            raise AnalysisError("invalid_covariance", "Supply a finite exactly symmetric sampling covariance matrix; no symmetrization is applied.")
        if bool((dense[index[:, None] != index[None, :]] != 0).any()):
            raise AnalysisError("unsupported_dependence", "Cross-study sampling covariance is unsupported; independent study blocks are required.")
        blocks = packing.matrix(dense)
        sampling = "covariance"
    _check_blocks(blocks, packing)
    parts = [torch.ones(n, dtype=torch.float64)] if intercept else []
    x = torch.stack([*parts, *[values[name] for name in moderators]], 1)
    y = values[yi]
    state = {"schema": SCHEMA, "device": "cpu", "dtype": "float64", "roles": roles, "sampling": sampling,
             "effect_ids": ids, "study_ids": study_ids, "studies": [json.loads(token) for token in tokens],
             "sample_positions": list(range(n)), "sample_labels": _labels(frame.index, name="original row"),
             "sample_hash": None, "terms": terms, "moderators": moderators.copy(), "intercept": intercept,
             "method": method, "sigma2": None, "x": x.tolist(), "y": y.tolist(),
             "s_blocks": packing.blocks(blocks), "beta": None, "covariance": None, "inference": inference,
             "residual_df": n-p if inference == "t" else None, "n_effects": n, "n_studies": groups,
             "level": level, "fit": None, "solver": None, "components": None}
    lik = kernels.Likelihood(x, y, blocks, packing, intercept)
    solution = lik.maximize(method)
    theta = solution["theta"]
    fitted = lik.finish(theta)
    upper = lik.upper()
    components, component_covariance, status = _component_inference(
        lik, method, list(theta), solution["loglik"], fitted, level, profile, upper)
    null = lik.evaluate(torch.zeros((1, 2), dtype=torch.float64))
    q0 = float(null["rss"][0])
    typical = (n-p)/(float(null["trace_p"][0])/lik.scale)
    state["sigma2"] = fitted["sigma2"]
    state["beta"] = fitted["beta"].tolist()
    state["covariance"] = fitted["covariance"].tolist()
    state["fit"] = {"loglik_ML": fitted["loglik_ML"], "loglik_REML": fitted["loglik_REML"], "rss": fitted["rss"],
                    "q0": q0, "typical_sampling_variance": typical, "variance_scale": lik.scale}
    state["solver"] = {"status": "converged", "kind": solution["kind"], "evaluations": lik.evaluations,
                       "max_evaluations": kernels.MAX_EVALUATIONS, "score_residual": solution["score_residual"],
                       "candidates": solution["candidates"], "upper_bound": solution["upper_bound"],
                       "boundary": [value == 0 for value in theta],
                       "algorithm": "batched padded-block GLS; zero, both one-component faces and log-scale Newton interior starts compared; KKT score check"}
    state["components"] = {"rows": components, "covariance": component_covariance, "status": status,
                           "profile": profile, "level": level}
    state["sample_hash"] = _digest(state)
    output = make_result(_seal(state))
    output.attrs["resource_plan"] = resource
    return output


def _replay(state):
    """Rebuild padded tensors and replay GLS at the saved components (no optimizer)."""
    n, groups = state["n_effects"], state["n_studies"]
    x = torch.tensor(state["x"], dtype=torch.float64)
    y = torch.tensor(state["y"], dtype=torch.float64)
    _, _, index = _cluster(state["study_ids"], "study")
    if int(index.max())+1 != groups or x.shape != (n, len(state["terms"])) or y.shape != (n,):
        raise ValueError("geometry")
    packing = kernels.Packing(index, groups)
    sizes = packing.counts.tolist()
    blocks = torch.diag_embed(1-packing.weight)
    if len(state["s_blocks"]) != groups:
        raise ValueError("blocks")
    for j, block in enumerate(state["s_blocks"]):
        tensor = torch.tensor(block, dtype=torch.float64)
        if tensor.shape != (sizes[j], sizes[j]):
            raise ValueError("block geometry")
        if state["sampling"] == "vi" and not torch.equal(tensor, torch.diag(tensor.diagonal())):
            raise ValueError("independent sampling blocks")
        blocks[j, :sizes[j], :sizes[j]] = tensor
    _check_blocks(blocks, packing)
    lik = kernels.Likelihood(x, y, blocks, packing, state["intercept"])
    theta = [value/lik.scale for value in state["sigma2"]]
    fitted = lik.finish(theta)
    return lik, packing, blocks, x, y, fitted


def _close(a, b):
    return bool(torch.allclose(a, b, rtol=2e-8, atol=2e-11*max(1e-12, float(b.abs().max()))))


@resident_cpu
@torch.no_grad()
def load_state(result):
    """Validate complete three-level state and replay GLS at saved components; no optimizer."""
    original = result.attrs.get("prediction_state") if isinstance(result, TableSet) else result
    try:
        encoded = json.dumps(original, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode()) > MAX_STATE_BYTES:
            raise ValueError("oversized")
        state = json.loads(encoded)
        if not isinstance(state, dict) or set(state) != STATE_KEYS:
            raise ValueError("schema fields")
        digest = state.pop("checksum")
        if checksum(state) != digest or state["schema"] != SCHEMA or state["device"] != "cpu" or state["dtype"] != "float64":
            raise ValueError("checksum or domain")
        state["checksum"] = digest
        n, groups = state["n_effects"], state["n_studies"]
        names, intercept = state["moderators"], state["intercept"]
        if type(n) is not int or type(groups) is not int or not 3 <= n <= MAX_EFFECTS or not 2 <= groups <= min(MAX_STUDIES, n):
            raise ValueError("dimensions")
        if type(intercept) is not bool or not isinstance(names, list) or any(not isinstance(v, str) or not v for v in names):
            raise ValueError("moderators")
        terms = (["Intercept"] if intercept else [])+names
        p = len(terms)
        if not 1 <= p <= MAX_TERMS or n <= p+2 or state["terms"] != terms or len(set(terms)) != p:
            raise ValueError("terms")
        if state["method"] not in ("ML", "REML") or state["sampling"] not in ("vi", "covariance"):
            raise ValueError("method")
        if state["inference"] == "z":
            if state["residual_df"] is not None:
                raise ValueError("df")
        elif state["inference"] != "t" or state["residual_df"] != n-p:
            raise ValueError("inference")
        level_check(state["level"])
        sigma2 = state["sigma2"]
        if not isinstance(sigma2, list) or len(sigma2) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in sigma2):
            raise ValueError("components")
        roles = state["roles"]
        expected = {"study", "yi"} | ({"vi"} if state["sampling"] == "vi" else {"effect"})
        if not isinstance(roles, dict) or not expected <= set(roles) <= expected | {"effect"}:
            raise ValueError("roles")
        ids = _labels(state["effect_ids"], unique=True, name="effect")
        study_ids = _labels(state["study_ids"], name="study")
        tokens = list(dict.fromkeys(_key(v) for v in study_ids))
        if len(ids) != n or len(study_ids) != n or len(tokens) != groups or [_key(v) for v in state["studies"]] != tokens:
            raise ValueError("labels")
        if state["sample_positions"] != list(range(n)) or len(state["sample_labels"]) != n:
            raise ValueError("positions")
        if _digest(state) != state["sample_hash"]:
            raise ValueError("sample digest")

        def numeric(value):
            return all(numeric(v) for v in value) if isinstance(value, list) else type(value) in (int, float)
        for name in ("x", "y", "s_blocks", "beta", "covariance"):
            if not numeric(state[name]):
                raise ValueError("non-numeric scientific values")
        if intercept and any(row[0] != 1 for row in state["x"]):
            raise ValueError("intercept")
        lik, packing, blocks, x, y, fitted = _replay(state)
        beta = torch.tensor(state["beta"], dtype=torch.float64)
        covariance = torch.tensor(state["covariance"], dtype=torch.float64)
        if beta.shape != (p,) or covariance.shape != (p, p) or not _close(beta, fitted["beta"]) or not _close(covariance, fitted["covariance"]):
            raise ValueError("saved GLS")
        fit = state["fit"]
        if not isinstance(fit, dict) or set(fit) != {"loglik_ML", "loglik_REML", "rss", "q0", "typical_sampling_variance", "variance_scale"}:
            raise ValueError("fit receipt")
        for name in ("loglik_ML", "loglik_REML", "rss", "variance_scale"):
            if type(fit[name]) not in (int, float) or not math.isclose(fit[name], fitted[name], rel_tol=2e-8, abs_tol=2e-9):
                raise ValueError("likelihood receipt")
        score = fitted["score_"+state["method"]]
        for a in range(2):
            residual = score[a]*(1+sigma2[a]/lik.scale)*lik.scale/n
            if (sigma2[a] == 0 and residual > kernels.SCORE_TOLERANCE) or (sigma2[a] > 0 and abs(residual) > kernels.SCORE_TOLERANCE):
                raise ValueError("score stationarity")
        if int(torch.linalg.cholesky_ex(covariance)[1]) != 0 or not torch.allclose(covariance, covariance.T, rtol=1e-12, atol=0):
            raise ValueError("covariance")
        if not isinstance(state["solver"], dict) or state["solver"].get("boundary") != [v == 0 for v in sigma2]:
            raise ValueError("solver receipt")
        components = state["components"]
        if not isinstance(components, dict) or set(components) != {"rows", "covariance", "status", "profile", "level"}:
            raise ValueError("component receipt")
        rows = components["rows"]
        if not isinstance(rows, list) or [row.get("component") for row in rows] != list(COMPONENTS):
            raise ValueError("component rows")
        if any(not math.isclose(row["sigma2"], value, rel_tol=1e-12, abs_tol=0) for row, value in zip(rows, sigma2)):
            raise ValueError("component values")
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OverflowError, AnalysisError) as exc:
        raise AnalysisError("invalid_result_state", "Use complete, coherent three-level saved state with exact sample/covariance alignment.") from exc
    tensors = {"x": x, "y": y, "blocks": blocks, "beta": beta, "covariance": covariance, "packing": packing,
               "likelihood": lik, "fitted": fitted}
    return state, tensors


def _nullable(records, columns=None, index=None):
    frame = table(records, columns=columns, index=index)
    return frame.astype(object).where(frame.notna(), None)


def make_result(state):
    """Build canonical tables from admitted state, without variance refitting."""
    state, tensors = load_state(state)
    beta, covariance = tensors["beta"], tensors["covariance"]
    level, inference, df = state["level"], state["inference"], state["residual_df"]
    terms = state["terms"]
    n, p = state["n_effects"], len(terms)
    se = covariance.diagonal().sqrt()
    statistics = beta/se
    c = critical(level, inference, df)
    coefficients = table({"term": terms, "estimate": beta.tolist(), "std_error": se.tolist(),
                          "statistic": statistics.tolist(),
                          "p_value": [probability(float(v), inference, df) for v in statistics],
                          "df": [df]*p, "ci_low": (beta-c*se).tolist(), "ci_high": (beta+c*se).tolist()})
    x = tensors["x"]
    fitted_values = x@beta
    sampling = tensors["packing"].unvector(tensors["blocks"].diagonal(dim1=-2, dim2=-1))
    effects = table({"effect": state["effect_ids"], "study": state["study_ids"], "position": state["sample_positions"],
                     "yi": state["y"], "sampling_variance": sampling.tolist(), "fitted": fitted_values.tolist(),
                     "residual": (tensors["y"]-fitted_values).tolist()})
    sigma2 = state["sigma2"]
    total = sum(sigma2)
    typical = state["fit"]["typical_sampling_variance"]
    q0 = state["fit"]["q0"]
    heterogeneity = table([{"I2_total_percent": 100*total/(total+typical),
                            "I2_study_percent": 100*sigma2[0]/(total+typical),
                            "I2_effect_percent": 100*sigma2[1]/(total+typical),
                            "typical_sampling_variance": typical}])
    method = state["method"]
    loglik = state["fit"]["loglik_"+method]
    parameters = p+2
    effective = n-p if method == "REML" else n
    aicc_n = max(effective, parameters+2)
    fit = table([{"method": method, "loglik": loglik, "minus_two_loglik": -2*loglik, "aic": -2*loglik+2*parameters,
                  "bic": -2*loglik+parameters*math.log(effective),
                  "aicc": -2*loglik+2*parameters*aicc_n/(aicc_n-parameters-1),
                  "loglik_ML": state["fit"]["loglik_ML"], "loglik_REML": state["fit"]["loglik_REML"],
                  "n_effects": n, "n_studies": state["n_studies"], "n_parameters": parameters}])
    tests = [{"test": "QE residual heterogeneity (sampling covariance only)", "statistic": q0, "distribution": "chi2",
              "df1": n-p, "df2": None, "p_value": chi2_sf(q0, n-p)}]
    indices = list(range(int(state["intercept"]), p))
    if indices:
        b, cv = beta[indices], covariance[indices][:, indices]
        wald = float(b@torch.linalg.solve(cv, b))
        m = len(indices)
        tests.append({"test": "QM joint moderators", "statistic": wald if inference == "z" else wald/m,
                      "distribution": "chi2" if inference == "z" else "F", "df1": m,
                      "df2": None if inference == "z" else df,
                      "p_value": chi2_sf(wald, m) if inference == "z" else f_sf(wald/m, m, df)})
    components = state["components"]
    tables = {"coefficients": coefficients, "covariance": table(covariance.tolist(), columns=terms, index=terms),
              "variance_components": _nullable(components["rows"], COMPONENT_COLUMNS),
              "component_covariance": _nullable(components["covariance"], list(COMPONENTS), list(COMPONENTS)),
              "heterogeneity": heterogeneity, "fit": fit, "tests": _nullable(tests), "effects": effects}
    return TableSet(tables, title="Three-level meta-regression" if state["moderators"] else "Three-level meta-analysis",
                    procedure="meta_multilevel", prediction_state=state, method=method, inference=inference,
                    level=level, n_effects=n, n_studies=state["n_studies"], residual_df=df, sigma2=sigma2,
                    sampling=state["sampling"], source=SOURCE, profile_source=PROFILE_SOURCE, i2_source=I2_SOURCE,
                    variance_solver=state["solver"], component_inference=components["status"],
                    notes=["Known sampling covariance, independent studies and normal random effects are supplied assumptions.",
                           "Coefficient uncertainty plugs in the estimated variance components.",
                           "Component LRTs report the naive chi2(1) p-value and the 50:50 boundary-mixture p-value.",
                           "The finite candidate search is not a global-optimization proof for arbitrary pathological data."])


@resident_cpu
@torch.no_grad()
def meta_multilevel_predict(result, *, data=None, level=None):
    """Predict three-level conditional means and plug-in latent-effect intervals.

    Uses the saved full coefficient covariance; intervals add both variance
    components (new study, new effect) and use the fitted z or t(k-p) rule.
    Excludes future sampling error and component uncertainty. <=2000 rows.
    """
    state, tensors = load_state(result)
    level = level_check(state["level"] if level is None else level)
    names = state["moderators"]
    if names:
        if isinstance(data, Dataset):
            raise AnalysisError("unsupported_dataset", "Prediction requires an in-memory table.")
        frame = _coerce_frame(data)
        if not 1 <= len(frame) <= MAX_EFFECTS or any(list(frame.columns).count(name) != 1 for name in names):
            raise AnalysisError("invalid_columns", "Supply 1..2000 rows containing every fitted moderator exactly once.")
        if bool(frame.loc[:, names].isna().any().any()):
            raise AnalysisError("missing_values", "Prediction moderators must be complete.")
        columns = []
        for name in names:
            if pd.api.types.is_bool_dtype(frame[name].dtype):
                raise AnalysisError("non_numeric_column", "Prediction moderators cannot be boolean.")
            columns.append(_numeric(frame[name], name))
        parts = [torch.ones(len(frame), dtype=torch.float64)] if state["intercept"] else []
        x = torch.stack([*parts, *columns], 1)
    else:
        if data is not None:
            raise AnalysisError("invalid_spec", "Intercept-only prediction uses data=None and returns one mean.")
        x = torch.ones((1, 1), dtype=torch.float64)
    mean = x@tensors["beta"]
    variance = torch.einsum("ip,pq,iq->i", x, tensors["covariance"], x)
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
        raise AnalysisError("numerical_failure", "Prediction mean or variance is invalid; rescale moderators.")
    total = sum(state["sigma2"])
    c = critical(level, state["inference"], state["residual_df"])
    se, pi_se = variance.sqrt(), (variance+total).sqrt()
    return table({"estimate": mean.tolist(), "std_error": se.tolist(), "ci_low": (mean-c*se).tolist(),
                  "ci_high": (mean+c*se).tolist(), "pi_low": (mean-c*pi_se).tolist(), "pi_high": (mean+c*pi_se).tolist()},
                 procedure="meta_multilevel_predict", sample_hash=state["sample_hash"], state_checksum=state["checksum"],
                 level=level, inference=state["inference"], residual_df=state["residual_df"], source=PREDICT_SOURCE,
                 interval_target="latent effect in a new study; approximate; excludes future sampling variance and component uncertainty")
