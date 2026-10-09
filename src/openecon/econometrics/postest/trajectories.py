"""Saved, explicit event/horizon estimate families with joint normal-limit bands."""

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.postest.common import require_result
from openecon.econometrics.postest.multiple import simultaneous_ci
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.resources import plan_workspace


@resident_cpu
def trajectory_bands(result, *, terms=None, contrasts=None, labels=None,
                     assumptions, alpha=0.05, draws=50000, seed=1729):
    """Joint normal-limit event/horizon bands using saved full cross-target V.

    Supported saved LP/IV-LP/panel-LP and heterodid/eventstudy/csdid results.
    Declare the CLT, valid identification and sampling assumptions. Cluster
    marginal t reporting does not make a known joint-t pivot. These are not
    finite/few-cluster, weak-IV or selective/pretrend-sensitivity intervals.
    A prespecified linear reporting map uses A beta and A V A'. No refit.
    """
    result = require_result(result, "event/horizon result")
    if result.spec.estimator not in {"lp", "lpiv", "panel_lp", "heterodid", "eventstudy", "csdid"}:
        raise AnalysisError("unsupported_model", "Choose a supported saved event/horizon family.")
    if not isinstance(assumptions, str) or not assumptions.strip() or len(assumptions) > 4000:
        raise AnalysisError("missing_assumptions", "Declare a bounded nonempty CLT/identification description.")
    names = [c.term for c in result.coefficients]
    n = len(names)
    if not 1 <= n <= 384 or len(set(names)) != n:
        raise AnalysisError("work_budget", "Use a unique saved family of at most 384 targets.")
    plan_workspace("saved event/horizon covariance admission", {"full source geometry": 128*n*n})
    try:
        beta = torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)
        covariance = torch.tensor(result.covariance_matrix, dtype=torch.float64)
        se = torch.tensor([c.std_error for c in result.coefficients], dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_result_state", "Saved estimates/SEs and covariance must be numeric.") from exc
    if (covariance.shape != (n, n) or not torch.isfinite(beta).all()
            or not torch.isfinite(se).all() or (se <= 0).any() or not torch.isfinite(covariance).all()
            or not torch.allclose(covariance.diag(), se.square(), rtol=1e-8, atol=1e-12)):
        raise AnalysisError("invalid_result_state", "Saved estimates/SEs and complete covariance must agree.")
    correlation = covariance / se[:, None] / se[None, :]
    if (not torch.isfinite(correlation).all()
            or not torch.allclose(correlation, correlation.T, rtol=1e-10, atol=1e-12)
            or torch.linalg.eigvalsh((correlation+correlation.T)/2).min() < -1e-12):
        raise AnalysisError("invalid_result_state", "The complete saved covariance must be symmetric PSD.")
    if contrasts is None:
        selected = names if terms is None else terms
        if (not isinstance(selected, (list, tuple)) or not selected
                or any(not isinstance(k, str) or k not in names for k in selected)
                or len(set(selected)) != len(selected) or labels is not None):
            raise AnalysisError("invalid_family", "Choose unique saved terms; labels are for an explicit contrast map.")
        indices = [names.index(k) for k in selected]
        a = torch.eye(n, dtype=torch.float64)[indices]
        family_labels = list(selected)
    else:
        if terms is not None or not isinstance(contrasts, (list, tuple)) or not 1 <= len(contrasts) <= 384:
            raise AnalysisError("invalid_family", "Give either named terms or a bounded prespecified contrast map.")
        if (not isinstance(labels, (list, tuple)) or len(labels) != len(contrasts)
                or any(not isinstance(k, str) or not k.strip() for k in labels)
                or len(set(labels)) != len(labels)
                or any(not isinstance(row, (list, tuple)) or len(row) != n
                       or any(isinstance(v, bool) or not isinstance(v, (int, float))
                              or not math.isfinite(v) for v in row) for row in contrasts)):
            raise AnalysisError("invalid_family", "Map rows and unique labels must match stored target order.")
        a = torch.tensor(contrasts, dtype=torch.float64)
        family_labels = list(labels)
    m = len(family_labels)
    if isinstance(draws, bool) or not isinstance(draws, int) or not 1000 <= draws <= 200000:
        raise AnalysisError("invalid_draws", "draws must be an integer between 1000 and 200000.")
    if m*draws > 8000000:
        raise AnalysisError("work_budget", "The joint interval simulation exceeds the declared budget.")
    plan = plan_workspace("saved event/horizon normal-limit bands", {
        "full source and reporting geometry": 128*(n*n+m*n+m*m),
        "normal noise, responses, absolute maxima and quantile workspace": 32*m*draws+32*draws,
    }).record()
    targets, joint = a @ beta, a @ covariance @ a.T
    bands = simultaneous_ci(targets, joint, labels=family_labels, alpha=alpha,
                            draws=draws, seed=seed,
                            family_description=f"{result.spec.estimator} saved prespecified event/horizon family; {assumptions}")
    return saved_summary(TableSet(
        {"bands": bands, "reporting_map": table(a.tolist(), columns=names),
         "joint_covariance": table(joint.tolist(), columns=family_labels)},
        **{**bands.attrs, "device": "cpu"}, source_result_id=result.id, source_spec=result.spec.model_dump(mode="json"),
        source_sample_hash=result.provenance.get("sample_hash"), source_data_hash=result.provenance.get("data_hash"),
        stored_terms=names, reported_labels=family_labels, assumptions=assumptions.strip(),
        source_inference=result.inference, refitted=False, resource_plan=plan,
        excludes=["finite/few-cluster coverage", "weak-IV confidence sets", "selection after viewing effects", "pretrend sensitivity"],
    ))
