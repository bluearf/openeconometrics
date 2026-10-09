"""Six bounded unidimensional MML families and checked portable postestimation."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real

import torch

from openecon.analysis import _coerce_frame, _frame_hasher, _json_scalar, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.engines.distributions import normal_isf, normal_sf
from openecon.resources import plan_workspace, workspace_budget_bytes
from . import kernels

FAMILIES = ("rasch", "2pl", "3pl", "grm", "pcm", "rsm")
SOURCES = ["https://www.stata.com/manuals/irt.pdf", "https://www.jstatsoft.org/v48/i06"]
NOTES = ["Independent unidimensional local independence; latent theta ~ N(0,1).",
         "Nonadaptive Gauss-Hermite marginal ML; observed-information asymptotic z intervals.",
         "Rasch/PCM/RSM discrimination fixed at 1; 3PL guessing is supplied and fixed.",
         "EAP posterior SD and information curves condition on fitted item parameters.",
         "No multidimensional/DIF/item-fit/vendor-parity or licensed vendor-run claim."]


def _checksum(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise AnalysisError("invalid_spec", f"{name} must be an integer in {minimum}..{maximum}.")
    return value


def _real(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not lower < value < upper:
        raise AnalysisError("invalid_spec", f"{name} must be finite and strictly between {lower} and {upper}.")
    return float(value)


def _responses(data, items, *, fit=False, categories=None):
    if not isinstance(items, list) or not 3 <= len(items) <= 16 or any(not isinstance(s, str) or not s for s in items) or len(set(items)) != len(items):
        raise AnalysisError("invalid_items", "items must be 3..16 distinct nonempty column names.")
    raw = _coerce_frame(data)
    if not 1 <= len(raw) <= 3000:
        raise AnalysisError("resource_limit", "IRT resident input requires 1..3000 person rows; Dataset streaming is unsupported.")
    if any(list(raw.columns).count(s) != 1 for s in items):
        raise AnalysisError("invalid_columns", "Each selected item column must exist exactly once.")
    plan_workspace("IRT response admission", {"typed responses and sample": len(raw)*len(items)*8*12})
    frame = raw.loc[:, items]
    missing = frame.isna()
    if fit:
        keep = ~missing.any(axis=1)
        positions = [i for i, good in enumerate(keep.tolist()) if good]
        selected = frame.loc[keep]
        if len(selected) < 50:
            raise AnalysisError("insufficient_sample", "IRT fitting needs at least 50 complete people; fitting uses explicit listwise deletion.")
    else:
        positions, selected = list(range(len(raw))), frame
    values = []
    for name in items:
        if selected[name].dtype.kind == "b":
            raise AnalysisError("invalid_categories", "Items must be numeric integer category codes, not booleans.")
        series = selected[name].fillna(-1)
        with torch.device("cpu"):
            vector = _numeric(series, name).clone().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(vector).all()) or bool(((vector < -1) | (vector > 5) | (vector != vector.round())).any()):
            raise AnalysisError("invalid_categories", "Item categories must be integers 0..5; only actual missing cells may be excluded.")
        if bool((vector == -1).any()) and not bool(selected[name].isna().any()):
            raise AnalysisError("invalid_categories", "-1 is not a supported observed category code.")
        # Preserve the distinction between a supplied -1 and a missing cell.
        observed_values = vector[~torch.tensor(selected[name].isna().tolist(), device="cpu")]
        if bool((observed_values < 0).any()):
            raise AnalysisError("invalid_categories", "Observed categories must be nonnegative.")
        values.append(vector.long())
    responses = torch.stack(values, 1)
    if fit:
        categories = [int(v.max())+1 for v in values]
        for v, k in zip(values, categories):
            if k < 2 or set(v.tolist()) != set(range(k)):
                raise AnalysisError("invalid_categories", "Every fitted item must contain every contiguous category 0..K-1, including both binary levels.")
    elif any(bool((responses[:, j] >= k).any()) for j, k in enumerate(categories)):
        raise AnalysisError("invalid_categories", "Scoring responses contain categories absent from the fitted item universe.")
    return frame, responses, categories, positions, [_json_scalar(raw.index[i]) for i in positions]


def _terms(family, items, categories):
    if family == "rsm":
        return [f"{s}:difficulty" for s in items] + [f"shared:step_{k}" for k in range(1, categories[0])]
    terms = []
    for s, k in zip(items, categories):
        if family in ("2pl", "3pl", "grm"):
            terms.append(f"{s}:discrimination")
        terms.extend(f"{s}:difficulty_{v}" for v in range(1, k))
    return terms


def _posterior(state, responses):
    nodes, weights = kernels.tensor(state["nodes"]), kernels.tensor(state["weights"])
    loglik = kernels.pattern_loglik(kernels.tensor(state["raw_parameters"]), state["family"],
                                     state["categories"], state["guessing"], nodes, responses)
    posterior = torch.softmax(loglik+weights.log(), 1)
    mean = posterior@nodes
    variance = (posterior * (nodes[None, :] - mean[:, None]).square()).sum(1)
    return mean, variance.sqrt(), -torch.logsumexp(loglik+weights.log(), 1)


class IRTResult(TableSet):
    """Complete IRT tables plus versioned item/sample/inference state."""

    def to_json(self) -> str:
        """Return checksum-protected complete model state; never truncate tables."""
        return json.dumps({"state": self.attrs["state"], "checksum": self.attrs["checksum"]},
                          sort_keys=True, allow_nan=False)


def _assemble(state):
    terms = _terms(state["family"], state["items"], state["categories"])
    estimates = kernels.tensor(state["parameters"])
    covariance = kernels.tensor(state["covariance"])
    se = covariance.diagonal().sqrt()
    z = estimates / se
    crit = normal_isf((1-state["level"])/2)
    coefficient_rows = [dict(term=name, estimate=float(estimates[i]), std_error=float(se[i]), z=float(z[i]),
                             p_value=2*normal_sf(abs(float(z[i]))), ci_low=float(estimates[i]-crit*se[i]),
                             ci_high=float(estimates[i]+crit*se[i]), reference="asymptotic normal")
                        for i, name in enumerate(terms)]
    mean, sd, nll = _posterior(state, torch.tensor(state["responses"], dtype=torch.long, device="cpu"))
    people = table(dict(position=state["sample_positions"], source_index=state["sample_indices"],
                        eap=mean.tolist(), posterior_sd=sd.tolist(), negative_log_marginal=nll.tolist()))
    sample = table(dict(position=list(range(state["nobs_original"])), source_index=state["all_indices"],
                       included=[i in set(state["sample_positions"]) for i in range(state["nobs_original"])],
                       missing_count=state["missing_counts"]))
    items = table([dict(item=s, categories=k, observed_counts=counts, fixed_guessing=c,
                        fixed_discrimination=1. if state["family"] in ("rasch", "pcm", "rsm") else None)
                   for s, k, counts, c in zip(state["items"], state["categories"], state["category_counts"], state["guessing"])])
    diagnostics = table([dict(log_likelihood=state["log_likelihood"], nobs=state["nobs"],
                             estimated_parameters=len(state["raw_parameters"]),
                             aic=-2*state["log_likelihood"]+2*len(state["raw_parameters"]),
                             bic=-2*state["log_likelihood"]+math.log(state["nobs"])*len(state["raw_parameters"]),
                             max_gradient=state["max_gradient"], converged=True,
                             evaluations=state["evaluations"], iterations=state["iterations"],
                             quadrature_points=len(state["nodes"]), audit_points=state["audit_points"],
                             quadrature_error_per_person=state["quadrature_error_per_person"],
                             min_information_eigenvalue=state["min_information_eigenvalue"])])
    result = IRTResult({"parameters": table(coefficient_rows),
                        "covariance": table(state["covariance"], columns=terms, index=terms),
                        "item_categories": items, "diagnostics": diagnostics,
                        "people": people, "sample": sample,
                        "quadrature": table(dict(theta=state["nodes"], weight=state["weights"]))},
                       title=f"IRT {state['family']} marginal maximum likelihood", state=copy.deepcopy(state),
                       checksum=_checksum(state), family=state["family"], nobs=state["nobs"],
                       nobs_original=state["nobs_original"], dropped_rows=state["nobs_original"]-state["nobs"],
                       sample_hash=state["sample_hash"], missing="listwise fitting; observed-item scoring",
                       device="cpu", dtype="float64", notes=NOTES.copy(), sources=SOURCES.copy(),
                       stata_parity_validated=False)
    for frame in result.values():
        frame.attrs["publication_notes"] = NOTES.copy()
    return result


def _fit(data, items, family, guessing, points, max_iter, max_eval, tolerance, level):
    points = _integer(points, "points", 21, 61)
    max_iter = _integer(max_iter, "max_iter", 1, 300)
    max_eval = _integer(max_eval, "max_eval", 1, 600)
    tolerance = _real(tolerance, "tolerance", 1e-10, 1e-4)
    level = _real(level, "level", .5, 1.)
    frame, responses, categories, positions, indices = _responses(data, items, fit=True)
    if family in ("rasch", "2pl", "3pl") and any(k != 2 for k in categories):
        raise AnalysisError("invalid_categories", "Binary IRT families require only categories 0 and 1.")
    if family == "rsm" and (min(categories) < 3 or len(set(categories)) != 1):
        raise AnalysisError("invalid_categories", "Rating scale requires the same >=3 observed categories for every item.")
    if guessing is None:
        guessing = [0.]*len(items)
    if not isinstance(guessing, list) or len(guessing) != len(items) or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or not 0 <= v < .4 for v in guessing):
        raise AnalysisError("invalid_guessing", "guessing must contain one finite fixed lower asymptote in [0,.4) per item.")
    guessing = [float(v) for v in guessing]
    n, j = responses.shape
    p = kernels.parameter_count(family, categories)
    if p > 64 or n <= 3*p or n*points*j*(max_eval+4*p) > 300_000_000:
        raise AnalysisError("work_limit", "IRT requires <=64 free parameters, n>3p and <=300 million planned likelihood/Hessian work units.")
    plan = plan_workspace("IRT MML and observed information", {
        "response/autograd likelihood buffers": n*points*j*8*24,
        "Hessian optimizer and transformed covariance": p*p*8*64,
        "sample and saved tables": len(frame)*j*8*12,
    }, budget_bytes=min(128*1024**2, workspace_budget_bytes()))
    with torch.device("cpu"), torch.enable_grad():
        fitted = kernels.fit(responses, family, categories, guessing, points, max_iter, max_eval, tolerance)
    state = {"schema": "openecon.irt.v1", "family": family, "items": items.copy(), "categories": categories,
             "guessing": guessing, "latent_mean": 0., "latent_variance": 1., "level": level,
             "raw_parameters": fitted["raw"].tolist(), "parameters": fitted["estimates"].tolist(),
             "covariance": fitted["covariance"].tolist(), "raw_covariance": fitted["raw_covariance"].tolist(),
             "observed_information": fitted["hessian"].tolist(), "nodes": fitted["nodes"].tolist(),
             "weights": fitted["weights"].tolist(), "responses": responses.tolist(),
             "nobs": n, "nobs_original": len(frame), "sample_positions": positions, "sample_indices": indices,
             "all_indices": [_json_scalar(v) for v in frame.index],
             "missing_counts": frame.isna().sum(axis=1).tolist(),
             "category_counts": [[int((responses[:, v] == c).sum()) for c in range(k)] for v, k in enumerate(categories)],
             "sample_hash": _frame_hasher(frame).hexdigest(), "workspace_plan": plan.record(),
             "max_iter": max_iter, "max_eval": max_eval, "tolerance": tolerance,
             **{s: fitted[s] for s in ("log_likelihood", "max_gradient", "evaluations", "iterations",
                                       "min_information_eigenvalue", "quadrature_error_per_person", "audit_points")}}
    return _assemble(state)


def irt_rasch(*, data, items: list[str], points: int = 41, max_iter: int = 200,
              max_eval: int = 400, tolerance: float = 1e-6, level: float = .95) -> IRTResult:
    """Fit binary Rasch item difficulties; discrimination=1 and theta~N(0,1).

    Listwise fitting sample; native CPU float64 nonadaptive MML, full observed
    information z inference, versioned state and EAP posterior SD. Resident
    50..3000 people, 3..16 items, <=64 parameters, n>3p; no weights/Dataset.
    Max 300m planned work and 128 MiB/global workspace; convergence and denser
    quadrature audit required. This is not the free-common-slope 1PL model.
    """
    return _fit(data, items, "rasch", None, points, max_iter, max_eval, tolerance, level)


def irt_2pl(*, data, items: list[str], points: int = 41, max_iter: int = 200,
            max_eval: int = 400, tolerance: float = 1e-6, level: float = .95) -> IRTResult:
    """Fit binary positive-discrimination 2PL under standard-normal theta.

    See irt_rasch for resident/sample/work limits. The full covariance is
    transformed from log slopes to item discrimination/difficulty units.
    Interior slopes .1..5 and |threshold|<=8; weak information is rejected.
    """
    return _fit(data, items, "2pl", None, points, max_iter, max_eval, tolerance, level)


def irt_3pl(*, data, items: list[str], guessing: list[float], points: int = 41,
            max_iter: int = 200, max_eval: int = 400, tolerance: float = 1e-6,
            level: float = .95) -> IRTResult:
    """Fit 3PL slopes/difficulties with supplied fixed guessing in [0,.4).

    No estimated-guessing uncertainty or free-guessing fit is claimed. Other
    assumptions/limits match irt_2pl; each item lower asymptote is recorded.
    """
    return _fit(data, items, "3pl", guessing, points, max_iter, max_eval, tolerance, level)


def irt_grm(*, data, items: list[str], points: int = 41, max_iter: int = 200,
            max_eval: int = 400, tolerance: float = 1e-6, level: float = .95) -> IRTResult:
    """Fit Samejima graded responses with positive slopes/ordered thresholds.

    Each item has its own contiguous observed categories 0..K-1 (K=2..6).
    Ordered thresholds use positive gap geometry and full delta covariance.
    Standard-normal MML; other assumptions/limits match irt_2pl.
    """
    return _fit(data, items, "grm", None, points, max_iter, max_eval, tolerance, level)


def irt_pcm(*, data, items: list[str], points: int = 41, max_iter: int = 200,
            max_eval: int = 400, tolerance: float = 1e-6, level: float = .95) -> IRTResult:
    """Fit Masters partial-credit adjacent steps with discrimination fixed one.

    Contiguous item-specific categories 0..K-1 (K=2..6); adjacent thresholds
    may be disordered. This is not a free-common-slope PCM/GPCM. Identified
    theta~N(0,1); other assumptions/limits and inference match irt_rasch.
    """
    return _fit(data, items, "pcm", None, points, max_iter, max_eval, tolerance, level)


def irt_rsm(*, data, items: list[str], points: int = 41, max_iter: int = 200,
            max_eval: int = 400, tolerance: float = 1e-6, level: float = .95) -> IRTResult:
    """Fit Andrich rating scale with item locations/common zero-sum steps.

    Same contiguous 3..6 categories for every item, discrimination=1 and
    theta~N(0,1). Joint output includes the dependent last common step: its
    covariance is intentionally singular under the zero-sum constraint.
    Other assumptions/limits and inference match irt_rasch.
    """
    return _fit(data, items, "rsm", None, points, max_iter, max_eval, tolerance, level)


def _state(result):
    if isinstance(result, IRTResult):
        envelope = {"state": result.attrs.get("state"), "checksum": result.attrs.get("checksum")}
    elif isinstance(result, str):
        if len(result) > 8*1024**2:
            raise AnalysisError("resource_limit", "IRT saved JSON exceeds 8 MiB.")
        try:
            envelope = json.loads(result)
        except (ValueError, TypeError) as exc:
            raise AnalysisError("invalid_state", "Invalid IRT JSON.") from exc
    elif isinstance(result, dict):
        envelope = result
    else:
        raise AnalysisError("invalid_state", "Use an IRTResult or complete to_json() envelope.")
    try:
        state = envelope["state"]
        if _checksum(state) != envelope["checksum"] or state["schema"] != "openecon.irt.v1" or state["family"] not in FAMILIES:
            raise ValueError("schema/checksum")
        categories, items = state["categories"], state["items"]
        if not isinstance(items, list) or not 3 <= len(items) <= 16 or len(set(items)) != len(items) or any(not isinstance(s, str) or not s for s in items):
            raise ValueError("items")
        if len(categories) != len(items) or any(type(v) is not int or not 2 <= v <= 6 for v in categories):
            raise ValueError("categories")
        if state["family"] in ("rasch", "2pl", "3pl") and any(v != 2 for v in categories):
            raise ValueError("binary")
        if state["family"] == "rsm" and (categories[0] < 3 or len(set(categories)) != 1):
            raise ValueError("rating scale")
        p = kernels.parameter_count(state["family"], categories)
        m = len(_terms(state["family"], items, categories))
        n, original = state["nobs"], state["nobs_original"]
        if type(n) is not int or type(original) is not int or not 50 <= n <= original <= 3000 or p > 64 or n <= 3*p:
            raise ValueError("sample")
        positions = state["sample_positions"]
        if len(positions) != n or positions != sorted(set(positions)) or any(type(v) is not int or not 0 <= v < original for v in positions):
            raise ValueError("positions")
        if len(state["all_indices"]) != original or state["sample_indices"] != [state["all_indices"][i] for i in positions]:
            raise ValueError("indices")
        if len(state["missing_counts"]) != original or any(type(v) is not int or not 0 <= v <= len(items) for v in state["missing_counts"]):
            raise ValueError("missing")
        if positions != [i for i, v in enumerate(state["missing_counts"]) if v == 0]:
            raise ValueError("listwise sample")
        if len(state["guessing"]) != len(items) or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or not 0 <= v < .4 for v in state["guessing"]):
            raise ValueError("guessing")
        if state["family"] != "3pl" and any(state["guessing"]):
            raise ValueError("unexpected guessing")
        if state["latent_mean"] != 0 or state["latent_variance"] != 1:
            raise ValueError("latent identification")
        _real(state["level"], "level", .5, 1.)
        q = len(state["nodes"])
        if not 21 <= q <= 61 or len(state["weights"]) != q:
            raise ValueError("quadrature")
        plan_workspace("IRT state restore", {"validated state tensors": n*q*len(items)*8*24 + p*p*8*64})
        responses = torch.tensor(state["responses"], dtype=torch.long, device="cpu")
        if responses.shape != (n, len(items)) or any(type(v) is not int for row in state["responses"] for v in row):
            raise ValueError("responses")
        counts = []
        for j, k in enumerate(categories):
            if set(responses[:, j].tolist()) != set(range(k)):
                raise ValueError("response universe")
            counts.append([int((responses[:, j] == c).sum()) for c in range(k)])
        if counts != state["category_counts"]:
            raise ValueError("category counts")
        raw, natural = kernels.tensor(state["raw_parameters"]), kernels.tensor(state["parameters"])
        covariance, hessian = kernels.tensor(state["covariance"]), kernels.tensor(state["observed_information"])
        raw_covariance = kernels.tensor(state["raw_covariance"])
        if raw.shape != (p,) or natural.shape != (m,) or covariance.shape != (m, m) or hessian.shape != (p, p) or raw_covariance.shape != (p, p):
            raise ValueError("parameter shape")
        if any(not bool(torch.isfinite(v).all()) for v in (raw, natural, covariance, hessian, raw_covariance)):
            raise ValueError("finite geometry")
        a, steps, decoded = kernels.decode(raw, state["family"], categories)
        if not torch.allclose(decoded, natural, atol=1e-12, rtol=1e-12) or any(not .1 < float(v) < 5. for v in a) or any(bool((v.abs() > 8.).any()) for v in steps):
            raise ValueError("natural geometry")
        if not torch.allclose(hessian, hessian.T, atol=1e-10, rtol=1e-10) or float(torch.linalg.eigvalsh(hessian)[0]) <= 0:
            raise ValueError("information")
        if not torch.allclose(hessian @ raw_covariance, torch.eye(p, dtype=torch.float64, device="cpu"), atol=1e-7, rtol=1e-7):
            raise ValueError("information inverse")
        with torch.enable_grad():
            jacobian = torch.autograd.functional.jacobian(lambda value: kernels.decode(value, state["family"], categories)[2], raw)
        if not torch.allclose(covariance, jacobian@raw_covariance@jacobian.T, atol=1e-10, rtol=1e-10) or bool((covariance.diagonal() <= 0).any()):
            raise ValueError("joint covariance")
        nodes, weights = kernels.quadrature(q)
        if not torch.allclose(nodes, kernels.tensor(state["nodes"]), atol=1e-12, rtol=1e-12) or not torch.allclose(weights, kernels.tensor(state["weights"]), atol=1e-14, rtol=1e-12):
            raise ValueError("quadrature nodes")
        if abs(float(kernels.objective(raw, state["family"], categories, state["guessing"], nodes, weights, responses))+state["log_likelihood"]) > 1e-7:
            raise ValueError("likelihood")
        for key in ("log_likelihood", "max_gradient", "min_information_eigenvalue", "quadrature_error_per_person"):
            if isinstance(state[key], bool) or not isinstance(state[key], Real) or not math.isfinite(state[key]):
                raise ValueError("diagnostics")
        if state["quadrature_error_per_person"] > 1e-5 or state["max_gradient"] > max(2e-5, 10*state["tolerance"]):
            raise ValueError("accepted diagnostics")
        return copy.deepcopy(state)
    except (KeyError, ValueError, TypeError, IndexError, RuntimeError, OverflowError) as exc:
        raise AnalysisError("invalid_state", "IRT state failed checksum, dimension, sample, identification or inference checks.") from exc


def irt_restore(result) -> IRTResult:
    """Restore all IRT tables/state from an exact checked to_json() envelope.

    Checks schema/checksum, categories/sample, quadrature, identified geometry,
    inverse information, transformed covariance and fitted likelihood. The
    checksum detects accidental change; it is not a trusted digital signature.
    """
    return _assemble(_state(result))


def irt_score(result, *, data=None) -> TableSet:
    """Compute EAP theta/SD with fitted parameters fixed and normal prior.

    data=None reproduces fitted complete people and source positions. Supplied
    resident data scores observed items per person; a fully missing row gets
    prior mean=0, SD=1. Extreme responses remain finite. No item-parameter
    uncertainty or frequentist person confidence interval is claimed.
    """
    state = _state(result)
    if data is None:
        responses = torch.tensor(state["responses"], dtype=torch.long, device="cpu")
        positions, indices = state["sample_positions"], state["sample_indices"]
        digest = state["sample_hash"]
    else:
        frame, responses, _, positions, indices = _responses(data, state["items"], categories=state["categories"])
        digest = _frame_hasher(frame).hexdigest()
    plan = plan_workspace("IRT EAP scoring", {"posterior work": len(responses)*len(state["nodes"])*len(state["items"])*8*12})
    mean, sd, nll = _posterior(state, responses)
    observed = (responses >= 0).sum(1)
    empty = observed == 0
    mean[empty], sd[empty], nll[empty] = 0., 1., 0.
    scores = table(dict(position=positions, source_index=indices, observed_items=observed.tolist(),
                        eap=mean.tolist(), posterior_sd=sd.tolist(), negative_log_marginal=nll.tolist()))
    scores.attrs["publication_notes"] = NOTES.copy()
    return TableSet({"scores": scores}, title="IRT EAP posterior scores", sample_hash=digest,
                    conditional_item_parameters=True, fully_missing="standard normal prior", state_checksum=_checksum(state),
                    workspace_plan=plan.record(), notes=NOTES.copy())


def irt_information(result, *, theta: list[float]) -> TableSet:
    """Return category curves, expected scores and item/test Fisher information.

    Declared finite theta grid of 1..1001 points in [-8,8]; uses the exact
    fitted logistic/adjacent geometry and dP/dtheta. Information=sum(P'^2/P)
    per item; local independence sums item information into test information.
    Curves condition on saved item parameters; no curve confidence bands.
    """
    state = _state(result)
    if not isinstance(theta, list) or not 1 <= len(theta) <= 1001 or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or not -8 <= v <= 8 for v in theta):
        raise AnalysisError("invalid_theta", "theta must be a finite numeric list of 1..1001 points in [-8,8].")
    plan = plan_workspace("IRT item/test information", {"curves and gradients": len(theta)*sum(state["categories"])*8*32})
    rows, item_rows = [], []
    total_information = torch.zeros(len(theta), dtype=torch.float64, device="cpu")
    total_score = torch.zeros_like(total_information)
    with torch.enable_grad():
        x = kernels.tensor(theta).requires_grad_()
        lps = kernels.log_probabilities(kernels.tensor(state["raw_parameters"]), state["family"], state["categories"], state["guessing"], x)
        for s, lp in zip(state["items"], lps):
            probabilities = lp.exp()
            derivatives = torch.stack([torch.autograd.grad(lp[:, c].sum(), x, retain_graph=True)[0] for c in range(lp.shape[1])], 1)
            information = (probabilities*derivatives.square()).sum(1)
            expected = probabilities @ torch.arange(lp.shape[1], dtype=torch.float64, device="cpu")
            total_information += information.detach()
            total_score += expected.detach()
            for t in range(len(theta)):
                item_rows.append(dict(theta=float(x[t].detach()), item=s, expected_score=float(expected[t].detach()), information=float(information[t].detach())))
                for c in range(lp.shape[1]):
                    rows.append(dict(theta=float(x[t].detach()), item=s, category=c,
                                     probability=float(probabilities[t, c].detach()),
                                     derivative=float((probabilities[t, c]*derivatives[t, c]).detach())))
    for frame in (curves := table(rows), info := table(item_rows)):
        frame.attrs["publication_notes"] = NOTES.copy()
    return TableSet({"category_curves": curves, "item_information": info,
                     "test_information": table(dict(theta=theta, expected_score=total_score.tolist(), information=total_information.tolist()))},
                    title="IRT item/test curves and information", conditional_item_parameters=True,
                    workspace_plan=plan.record(), state_checksum=_checksum(state), notes=NOTES.copy())
