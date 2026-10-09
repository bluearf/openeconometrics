"""Descriptive EAP residual fit and binary observed-score Mantel-Haenszel DIF.

The DIF helper tests observed-score-stratified response association. It does
not fit a multiple-group latent model or establish measurement invariance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real

import torch

from openecon.analysis import _coerce_frame, _frame_hasher, _json_scalar
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, _json_safe, table
from openecon.econometrics.nonparametric.measures import mantel_haenszel
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.resources import plan_workspace
from . import kernels, models

FIT_SOURCE = "https://doi.org/10.1177/0146621616677520"
DIF_SOURCE = "https://www.ets.org/research/policy_research_reports/publications/report/1986/hwnk.html"
SCHEMA = "openecon.irt.diagnostics.v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _tables(result):
    return _json_safe({name: {"columns": list(frame.columns), "index": list(frame.index),
                              "data": frame.to_numpy().tolist()} for name, frame in result.items()})


class IRTDiagnostics(TableSet):
    """Complete portable diagnostic inputs/tables; restoration never refits."""

    def to_json(self) -> str:
        """Return checksum-protected complete diagnostic state, rejecting drift."""
        state = self.attrs.get("diagnostic_state")
        try:
            intact = (isinstance(state, dict) and _digest(state) == self.attrs.get("diagnostic_checksum")
                      and _digest(_tables(self)) == self.attrs.get("tables_checksum"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError("invalid_state", "IRT diagnostic state or tables changed.") from exc
        if not intact:
            raise AnalysisError("invalid_state", "IRT diagnostic state or tables changed.")
        text = json.dumps({"state": state, "checksum": _digest(state)}, sort_keys=True, allow_nan=False)
        if len(text.encode()) > 16 * 1024**2:
            raise AnalysisError("resource_limit", "IRT diagnostic JSON exceeds 16 MiB.")
        return text


def _result(tables, state, title, notes):
    output = IRTDiagnostics(tables, title=title, diagnostic_state=copy.deepcopy(state),
                            diagnostic_checksum=_digest(state), device="cpu", dtype="float64",
                            stata_parity_validated=False, notes=notes)
    for frame in output.values():
        frame.attrs["publication_notes"] = notes.copy()
    output.attrs["tables_checksum"] = _digest(_tables(output))
    return output


def _fit_tables(state):
    model = models._state(state["model"])
    y = kernels.tensor(model["responses"])
    eap, _, _ = models._posterior(model, y.long())
    plan_workspace("IRT saved EAP residual diagnostics", {
        "person item probability/residual buffers": len(y) * sum(model["categories"]) * 8 * 16,
        "complete residual and pair tables": len(y) * len(model["items"]) * 512,
    })
    probability = [v.exp() for v in kernels.log_probabilities(
        kernels.tensor(model["raw_parameters"]), model["family"], model["categories"], model["guessing"], eap
    )]
    means, variances = [], []
    for p in probability:
        codes = torch.arange(p.shape[1], dtype=torch.float64, device="cpu")
        mu = p @ codes
        means.append(mu)
        variances.append(p @ codes.square() - mu.square())
    expected, variance = torch.stack(means, 1), torch.stack(variances, 1)
    if not torch.isfinite(variance).all() or bool((variance <= 1e-12).any()):
        raise AnalysisError("degenerate_fit_diagnostic", "Conditional item variance is zero or numerically unstable.")
    residual = y - expected
    centered = residual - residual.mean(0)
    sums = centered.square().sum(0)
    if bool((sums <= 1e-12).any()):
        raise AnalysisError("degenerate_fit_diagnostic", "A residual item has zero or unstable variance; Q3 is undefined.")
    correlation = ((centered.T @ centered) / (sums.sqrt()[:, None] * sums.sqrt()[None, :])).clamp(-1, 1)
    pairs = [(a, b) for a in range(len(model["items"])) for b in range(a + 1, len(model["items"]))]
    average = sum(float(correlation[a, b]) for a, b in pairs) / len(pairs)
    pair_table = table([dict(item1=model["items"][a], item2=model["items"][b],
                             q3=float(correlation[a, b]), adjusted_q3=float(correlation[a, b]) - average,
                             n=model["nobs"]) for a, b in pairs])
    item_table = table([dict(item=name, n=model["nobs"], observed_mean=float(y[:, j].mean()),
                             expected_mean=float(expected[:, j].mean()), mean_residual=float(residual[:, j].mean()),
                             rmse=float(residual[:, j].square().mean().sqrt()),
                             infit_mean_square=float(residual[:, j].square().sum() / variance[:, j].sum()),
                             outfit_mean_square=float((residual[:, j].square() / variance[:, j]).mean()))
                        for j, name in enumerate(model["items"])])
    row_table = table([dict(position=model["sample_positions"][i], source_index=model["sample_indices"][i],
                            item=name, eap=float(eap[i]), observed=float(y[i, j]), expected=float(expected[i, j]),
                            conditional_variance=float(variance[i, j]), residual=float(residual[i, j]),
                            standardized_residual=float(residual[i, j] / variance[i, j].sqrt()))
                       for i in range(len(y)) for j, name in enumerate(model["items"])])
    total_residual = residual.sum(1)
    total_variance = variance.sum(1)
    test_table = table([dict(n=model["nobs"], items=len(model["items"]), average_q3=average,
                             observed_mean_score=float(y.sum(1).mean()), expected_mean_score=float(expected.sum(1).mean()),
                             score_rmse=float(total_residual.square().mean().sqrt()),
                             score_outfit_mean_square=float((total_residual.square() / total_variance).mean()))])
    sample_table = table(dict(position=list(range(model["nobs_original"])), source_index=model["all_indices"],
                             included=[i in set(model["sample_positions"]) for i in range(model["nobs_original"])],
                             missing_count=model["missing_counts"]))
    return _result({"items": item_table, "pairs": pair_table, "test": test_table,
                    "residuals": row_table, "sample": sample_table}, state, "IRT descriptive EAP residual fit", [
        "Expected category score and variance evaluated at each fitted person's EAP; all item parameters fixed.",
        "Q3 is the Pearson correlation of observed-minus-expected item residuals; adjusted Q3 subtracts the off-diagonal average.",
        "Infit/outfit are descriptive mean squares; no calibrated chi-square p, reference df, universal cutoff or invariance claim.",
        "Person/item estimates use the same responses: residual correlations have estimation bias; calibration needs a separate refit bootstrap.",
        FIT_SOURCE,
    ])


@resident_cpu
def irt_fit_diagnostics(result, *, data=None) -> IRTDiagnostics:
    """Descriptive item/test mean-square fit and EAP residual Q3 for saved IRT.

    All six unidimensional families; original listwise fitted sample, fixed
    item parameters, CPU float64. data=None uses full saved responses; supplied
    data must match the exact original selected-item hash, positions and indices.
    No refit, chi-square p, universal Q3 cutoff or item-parameter uncertainty.
    Complete portable to_json()/irt_diagnostics_restore; <=3000 people/16 items.
    """
    model = models._state(result)
    if data is not None:
        frame, responses, _, positions, indices = models._responses(data, model["items"], fit=True)
        if (_frame_hasher(frame).hexdigest() != model["sample_hash"] or positions != model["sample_positions"]
                or indices != model["sample_indices"] or responses.tolist() != model["responses"]):
            raise AnalysisError("sample_mismatch", "IRT fit diagnostics require the exact original selected-item source and sample.")
    state = {"schema": SCHEMA, "method": "eap_fit", "model": {"state": model, "checksum": models._checksum(model)}}
    return _fit_tables(state)


def _scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, (str, int, float, bool)) or isinstance(value, float) and not math.isfinite(value):
        raise AnalysisError("invalid_groups", "DIF groups must be finite scalar labels.")
    return value


def _key(value):
    value = _scalar(value)
    return type(value).__name__, str(value)


def _mh_tables(state):
    items, labels = state["items"], state["match_levels"]
    if (not isinstance(items, list) or not 1 <= len(items) <= 16 or len(set(items)) != len(items)
            or any(not isinstance(name, str) or not name for name in items)
            or not isinstance(labels, list) or not 1 <= len(labels) <= 257
            or any(type(v) is not int or not 0 <= v <= 256 for v in labels) or labels != sorted(set(labels))):
        raise ValueError("saved item/matching universe")
    if (not isinstance(state["group"], str) or not state["group"] or not isinstance(state["match"], str)
            or not state["match"] or state["group"] == state["match"]
            or state["group"] in items or state["match"] in items):
        raise ValueError("saved matching/group roles")
    alpha = state["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1 or type(state["continuity"]) is not bool:
        raise ValueError("saved inference options")
    if _key(state["reference"]) == _key(state["focal"]):
        raise ValueError("saved groups")
    positions, indices, n = state["positions"], state["source_indices"], state["n_input"]
    if (type(n) is not int or not 3 <= n <= 100000 or len(positions) != len(indices)
            or len(positions) < 3 or positions != sorted(set(positions))
            or any(type(v) is not int or not 0 <= v < n for v in positions)
            or state["missing"] not in ("raise", "drop")):
        raise ValueError("saved sample geometry")
    excluded = state["excluded_positions"]
    if (not isinstance(excluded, list) or excluded != sorted(set(excluded))
            or any(type(v) is not int or not 0 <= v < n for v in excluded)
            or set(positions) & set(excluded) or sorted(positions + excluded) != list(range(n))
            or state["missing"] == "raise" and excluded
            or not isinstance(state["source_hash"], str) or len(state["source_hash"]) != 64
            or any(c not in "0123456789abcdef" for c in state["source_hash"])):
        raise ValueError("saved source/missing alignment")
    counts_list = state["counts"]
    if (not isinstance(counts_list, list) or len(counts_list) != len(items)
            or any(len(strata) != len(labels) or any(len(row) != 4 or any(type(v) is not int or v < 0 for v in row)
                   for row in strata) for strata in counts_list)):
        raise ValueError("saved count dimensions")
    plan_workspace("IRT observed-score stratified DIF", {"counts and complete inference": len(items) * len(labels) * 1024,
                                                        "saved sample identities": n * 256})
    counts = kernels.tensor(counts_list)
    if any(float(matrix.sum()) != len(positions) for matrix in counts):
        raise ValueError("saved count totals")
    margins = torch.stack((counts[:, :, 0] + counts[:, :, 1], counts[:, :, 2] + counts[:, :, 3]), 2)
    if not torch.equal(margins, margins[0].expand_as(margins)):
        raise ValueError("saved person/group strata differ across items")
    if bool((margins[0].sum(1) <= 0).any()):
        raise ValueError("saved matching levels must be observed")
    rows, summary = [], []
    for name, matrix in zip(items, counts):
        a, b, c, d = matrix.T
        total, r1, r2, c1, c2 = matrix.sum(1), a + b, c + d, a + c, b + d
        informative = (r1 > 0) & (r2 > 0) & (c1 > 0) & (c2 > 0) & (total > 1)
        result = mantel_haenszel(matrix.reshape(-1, 2, 2), alpha)
        if "odds_ratio" not in result or "tests" not in result:
            raise AnalysisError("undefined_dif", f"Item {name}: no finite pooled odds ratio or informative conditional variance; no pseudo-counts added.")
        odds, se, lower, upper, _, _ = result["odds_ratio"]
        stat, df, pvalue = result["tests"]["cmh_continuity" if state["continuity"] else "cmh"]
        log_odds = math.log(odds)
        summary.append(dict(item=name, odds_ratio_reference_over_focal=odds, log_odds_ratio=log_odds,
                            std_error_log=se, ci_low_odds_ratio=lower, ci_high_odds_ratio=upper,
                            delta_mh=-2.35 * log_odds, chi2=stat, df=df, p_value=pvalue,
                            informative_strata=int(informative.sum()), excluded_strata=int((~informative).sum()),
                            continuity_correction=state["continuity"]))
        for i, label in enumerate(labels):
            good, ni = bool(informative[i]), float(total[i])
            expected = float(r1[i] * c1[i] / ni) if ni else None
            variance = float(r1[i] * r2[i] * c1[i] * c2[i] / (ni * ni * (ni - 1))) if good else 0.
            rows.append(dict(item=name, match_score=label, reference_correct=int(a[i]), reference_incorrect=int(b[i]),
                             focal_correct=int(c[i]), focal_incorrect=int(d[i]), informative=good,
                             expected_reference_correct=expected, conditional_variance=variance,
                             exclusion_reason="" if good else "zero group/response margin or singleton stratum"))
    return _result({"dif": table(summary), "strata": table(rows),
                    "sample": table(dict(position=positions, source_index=indices))}, state, "Binary observed-score Mantel-Haenszel DIF", [
        "Independent unweighted people; prespecified observed-score strata and two explicitly named groups.",
        "Common correct-response odds ratio: reference/focal; delta_MH=-2.35*log(odds). No item purification or multiplicity adjustment.",
        "CMH uses fixed-margin hypergeometric variance and chi-square(1) approximation; explicit half-unit continuity correction.",
        "Log-odds uncertainty uses Robins-Breslow-Greenland asymptotic variance. Sparse/zero-margin strata are retained and identified; no pseudo-counts.",
        "This tests conditional observed response association; no latent-group invariance, uniform/nonuniform DIF equivalence, causal bias or vendor-run claim.",
        DIF_SOURCE,
    ])


@resident_cpu
def irt_mh_dif(data, *, items: list[str], group: str, reference, focal, match: str,
               alpha: float = .05, continuity: bool = True, missing: str = "raise") -> IRTDiagnostics:
    """Binary observed-score stratified Mantel-Haenszel DIF, no latent-model fit.

    Supply a prespecified integer matching-score column in 0..256 and exactly
    two finite group labels. All selected items are numeric 0/1; 3..100000
    independent unweighted resident rows, 1..16 items; explicit joint missing
    raise/drop. Zero-margin strata remain visible with zero information;
    undefined pooled odds fail. Hypergeometric CMH chi2(1), optional half-unit
    continuity correction and RBG log-odds Wald CI. No weights/Dataset, item
    purification, multiplicity or latent-invariance/nonuniform-DIF claim.
    """
    if (not isinstance(items, list) or not 1 <= len(items) <= 16 or len(set(items)) != len(items)
            or any(not isinstance(name, str) or not name for name in items)
            or not isinstance(group, str) or not isinstance(match, str) or group == match
            or group in items or match in items):
        raise AnalysisError("invalid_spec", "Use distinct named items, group and matching-score columns.")
    if missing not in ("raise", "drop") or type(continuity) is not bool:
        raise AnalysisError("invalid_spec", "missing is raise/drop and continuity must be Boolean.")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise AnalysisError("invalid_spec", "alpha must be finite and strictly between zero and one.")
    reference, focal = _scalar(reference), _scalar(focal)
    if _key(reference) == _key(focal):
        raise AnalysisError("invalid_groups", "Reference and focal groups must be distinct.")
    raw = _coerce_frame(data)
    if not 3 <= len(raw) <= 100000:
        raise AnalysisError("resource_limit", "DIF requires 3..100000 resident independent rows.")
    names = items + [group, match]
    if any(list(raw.columns).count(name) != 1 for name in names):
        raise AnalysisError("invalid_columns", "Each selected DIF column must exist exactly once.")
    plan_workspace("IRT DIF resident input admission", {"selected and encoded rows": len(raw) * (len(names) + 4) * 128})
    frame = raw.loc[:, names]
    keep = ~frame.isna().any(axis=1)
    if missing == "raise" and not keep.all():
        raise AnalysisError("missing_values", "DIF selected columns have missing values; request explicit joint drop if intended.")
    selected = frame.loc[keep]
    if len(selected) < 3:
        raise AnalysisError("insufficient_sample", "DIF needs at least three joint-complete people.")
    for name in items + [match]:
        if selected[name].dtype.kind not in "ifu":
            raise AnalysisError("invalid_categories", "DIF items/matching scores require numeric integer values, not booleans or strings.")
    response = kernels.tensor(selected[items].to_numpy().tolist())
    score = kernels.tensor(selected[match].tolist())
    if not torch.isfinite(response).all() or bool(((response != 0) & (response != 1)).any()):
        raise AnalysisError("invalid_categories", "DIF selected item responses must be numeric zero or one.")
    if not torch.isfinite(score).all() or bool(((score < 0) | (score > 256) | (score != score.round())).any()):
        raise AnalysisError("invalid_matching_score", "Prespecified matching scores must be integers in 0..256.")
    group_keys = [_key(value) for value in selected[group].tolist()]
    if any(value not in (_key(reference), _key(focal)) for value in group_keys):
        raise AnalysisError("invalid_groups", "Data contain a group outside the explicitly named reference/focal universe.")
    if len(set(group_keys)) != 2:
        raise AnalysisError("invalid_groups", "Both named DIF groups must be observed.")
    levels = sorted(set(score.long().tolist()))
    group_code = torch.tensor([int(value == _key(focal)) for value in group_keys], device="cpu")
    level_codes = {value: i for i, value in enumerate(levels)}
    score_code = torch.tensor([level_codes[int(value)] for value in score.tolist()], device="cpu")
    counts = []
    for j in range(len(items)):
        # Physical order is [reference correct, reference incorrect, focal
        # correct, focal incorrect], giving the reference/focal response odds.
        code = 4 * score_code + 2 * group_code + 1 - response[:, j].long()
        counts.append(torch.bincount(code, minlength=4 * len(levels)).reshape(-1, 4).tolist())
    positions = [i for i, value in enumerate(keep.tolist()) if value]
    state = dict(schema=SCHEMA, method="mh_dif", items=items.copy(), group=group, match=match,
                 reference=reference, focal=focal, match_levels=levels, counts=counts,
                 alpha=float(alpha), continuity=continuity, missing=missing, n_input=len(raw),
                 positions=positions, source_indices=[_json_scalar(raw.index[i]) for i in positions],
                 excluded_positions=[i for i, value in enumerate(keep.tolist()) if not value],
                 source_hash=_frame_hasher(frame).hexdigest())
    return _mh_tables(state)


@resident_cpu
def irt_diagnostics_restore(result) -> IRTDiagnostics:
    """Validate and regenerate complete IRT fit/DIF tables without estimation.

    Accepts intact IRTDiagnostics, JSON text or its state/checksum envelope.
    Fit states reuse full IRT identification/covariance/sample validation;
    DIF states validate count dimensions, sample/group/stratum margins and
    replay every conditional test and pooled log-odds variance. <=16 MiB JSON.
    """
    try:
        if isinstance(result, IRTDiagnostics):
            envelope = json.loads(result.to_json())
        elif isinstance(result, str):
            if len(result.encode()) > 16 * 1024**2:
                raise ValueError("JSON admission")
            envelope = json.loads(result)
        elif isinstance(result, dict):
            if len(json.dumps(result, allow_nan=False).encode()) > 16 * 1024**2:
                raise ValueError("JSON admission")
            envelope = result
        else:
            raise ValueError("unsupported envelope")
        state = envelope["state"]
        if _digest(state) != envelope["checksum"] or state["schema"] != SCHEMA:
            raise ValueError("schema/checksum")
        if state["method"] == "eap_fit":
            return _fit_tables(state)
        if state["method"] == "mh_dif":
            return _mh_tables(state)
        raise ValueError("method")
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError, OverflowError) as exc:
        raise AnalysisError("invalid_state", "IRT diagnostic schema, checksum or saved geometry failed validation.") from exc
