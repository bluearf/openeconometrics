"""Complete-denominator sensitivity and descriptive study display protocols."""
from __future__ import annotations

import torch

from openecon.analysis import _coerce_frame, _frame_hasher, _json_scalar, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.engines.distributions import t_isf, t_sf
from .common import SOURCES, critical, level_check, sample
from .models import meta_pool, meta_predict


def _sensitivity(frame, settings, label):
    try:
        fit = meta_pool(data=frame, **settings)
        coef = fit["coefficients"].iloc[0]
        return {"study_or_group": label, "n_studies": len(frame), "status": "ok", "error_code": "",
                "estimate": float(coef.estimate), "std_error": float(coef.std_error),
                "ci_low": float(coef.ci_low), "ci_high": float(coef.ci_high),
                "tau2": float(fit["heterogeneity"].iloc[0].tau2)}
    except AnalysisError as error:
        return {"study_or_group": label, "n_studies": len(frame), "status": "failed", "error_code": error.code,
                "estimate": None, "std_error": None, "ci_low": None, "ci_high": None, "tau2": None}


def _nullable(records):
    frame = table(records)
    # Missing results from explicitly failed fits stay null in strict JSON.
    return frame.astype(object).where(frame.notna(), None)


@torch.no_grad()
def meta_diagnostics(*, data, yi: str = "yi", vi: str = "vi", study: str | None = None,
                     group: str | None = None, order: str | None = None, method: str = "REML",
                     inference: str = "z", level: float = .95, dependence: str = "independent") -> TableSet:
    """Refit pooling for every leave-one-out, cumulative prefix and subgroup.

    Complete independent effects only, 3..200 studies, bounded CPU float64.
    Cumulative order is input order or a complete distinct numeric order column.
    All prefixes, subgroups and deletions remain in output, including failed or
    insufficient-study fits; attrs retain planned/completed/failed denominators.
    Classical Egger is weighted yi~1+se with multiplicative dispersion and a
    t(k-2) test of the se slope, requiring >=10 studies and variable se.
    Asymmetry can reflect heterogeneity/chance and does not establish publication
    bias. No trim-and-fill, selection model, dependent effects or causal claim.
    Forest intervals and funnel reference bands use the analysis scale.
    """
    level = level_check(level)
    _, values, ids, digest = sample(data, [yi, vi], study, dependence=dependence, minimum=3, maximum=200)
    raw = _coerce_frame(data)
    roles = [yi, vi, *([study] if study else [])]
    for role in (group, order):
        if role is not None:
            if not isinstance(role, str) or not role or role in roles or list(raw.columns).count(role) != 1 or bool(raw[role].isna().any()):
                raise AnalysisError("invalid_spec", "group/order must be distinct existing complete column roles.")
            roles.append(role)
    chosen = raw.loc[:, roles].copy().reset_index(drop=True)
    digest = _frame_hasher(chosen).hexdigest()
    settings = dict(yi=yi, vi=vi, study=study, method=method, inference=inference, level=level, dependence=dependence)
    overall = meta_pool(data=chosen, **settings)
    prediction = meta_predict(overall)
    k = len(chosen)
    ordering = list(range(k))
    if order:
        keys = _numeric(chosen[order], order)
        if len(set(keys.tolist())) != k:
            raise AnalysisError("ambiguous_order", "Cumulative order values must be distinct; ties need an explicit ordering.")
        ordering = keys.argsort().tolist()
    if study is None:
        # Stable original IDs must survive refits and cumulative sorting.
        label_name = "__meta_study__"
        while label_name in chosen.columns:
            label_name += "_"
        chosen[label_name] = ids
        settings["study"] = label_name
    loo = [_sensitivity(chosen.drop(index=i), settings, ids[i]) for i in range(k)]
    cumulative = [_sensitivity(chosen.iloc[ordering[:j]], settings, ids[ordering[j-1]]) for j in range(1, k+1)]
    subgroups = []
    if group:
        labels = [_json_scalar(v) for v in chosen[group]]
        if len({str(v) for v in labels}) != len(set(labels)):
            raise AnalysisError("ambiguous_group", "Group labels must have distinct displayed representations.")
        for label in dict.fromkeys(labels):
            indices = [i for i, value in enumerate(labels) if value == label]
            subgroups.append(_sensitivity(chosen.iloc[indices], settings, label))
    y, se = values[yi], values[vi].sqrt()
    egger = {"status": "ineligible", "reason": "requires >=10 studies", "n_studies": k,
             "statistic": None, "df": k-2, "p_value": None, "slope_se": None,
             "limit_estimate": None, "limit_ci_low": None, "limit_ci_high": None}
    if k >= 10:
        normalized_se = (se-se.mean())/se.square().mean().sqrt()
        x = torch.stack([torch.ones(k, dtype=torch.float64, device="cpu"), normalized_se], 1)
        z = x/se[:, None]
        singular = torch.linalg.svdvals(z)
        if float(singular[-1]) <= float(singular[0])*1e-10:
            egger["reason"] = "standard errors do not identify the asymmetry regression"
        else:
            q, r = torch.linalg.qr(z)
            b = torch.linalg.solve_triangular(r, (q.T@(y/se))[:, None], upper=True).squeeze(1)
            residual = (y-x@b)/se
            scale = float(residual.square().sum()/(k-2))
            ri = torch.linalg.solve_triangular(r, torch.eye(2, dtype=torch.float64, device="cpu"), upper=True)
            transform = torch.tensor([[1., -float(se.mean()/se.square().mean().sqrt())], [0., float(1/se.square().mean().sqrt())]], dtype=torch.float64, device="cpu")
            beta = transform@b
            cv = transform@(ri@ri.T*scale)@transform.T
            if not bool(torch.isfinite(cv).all()) or scale <= 1e-20:
                egger["reason"] = "degenerate asymmetry uncertainty"
            else:
                statistic = float(beta[1]/cv[1, 1].sqrt())
                c = t_isf((1-level)/2, k-2)
                egger.update(status="ok", reason="", statistic=statistic, p_value=2*t_sf(abs(statistic), k-2),
                             slope_se=float(beta[1]), limit_estimate=float(beta[0]),
                             limit_ci_low=float(beta[0]-c*cv[0, 0].sqrt()), limit_ci_high=float(beta[0]+c*cv[0, 0].sqrt()))
    pooled = float(prediction.iloc[0].estimate)
    c = critical(level, "z", None)
    forest = overall["studies"].loc[:, ["study", "yi", "ci_low", "ci_high", "weight_percent"]].copy()
    summary = table({"estimate": prediction.estimate.tolist(), "ci_low": prediction.ci_low.tolist(), "ci_high": prediction.ci_high.tolist(),
                     "pi_low": prediction.pi_low.tolist(), "pi_high": prediction.pi_high.tolist()})
    funnel = table({"study": ids, "effect": y.tolist(), "standard_error": se.tolist(),
                    "reference_low": (pooled-c*se).tolist(), "reference_high": (pooled+c*se).tolist()})
    counts = {name: {"planned": len(rows), "completed": sum(r["status"] == "ok" for r in rows),
                     "failed": sum(r["status"] != "ok" for r in rows)}
              for name, rows in (("leave_one_out", loo), ("cumulative", cumulative), ("subgroups", subgroups))}
    tables = {"leave_one_out": _nullable(loo), "cumulative": _nullable(cumulative),
              "egger": _nullable([egger]), "forest": forest, "summary": summary, "funnel": funnel}
    if subgroups:
        tables["subgroups"] = _nullable(subgroups)
    return TableSet(tables, title="Study sensitivity and descriptive diagnostics", procedure="meta_diagnostics",
                    sample_hash=digest, method=method, inference=inference, level=level, n_studies=k,
                    order=order or "input row order", ordered_studies=[ids[i] for i in ordering], group=group,
                    counts=counts, source=SOURCES["asymmetry"],
                    notes=["All planned refits remain in the denominator; cumulative singleton prefixes cannot estimate heterogeneity.",
                           "Egger is classical weighted regression with multiplicative dispersion. Asymmetry does not establish publication bias.",
                           "Forest uses sampling-normal study intervals; pooled CI and approximate true-effect PI use fitted inference."])


def meta_plot(result: TableSet, *, kind: str = "forest"):
    """Return a saved, editable PlotSpec for forest or funnel diagnostics.

    Uses all study rows without chart subsampling. Forest carries study-normal
    CIs and pooled fitted CI. Funnel includes pooled-center pseudo-confidence
    reference lines, with standard error on the vertical axis (zero at bottom).
    These are descriptive plots on the analysis scale, not bias corrections.
    """
    from openecon_charts.charts import PlotSpec
    if not isinstance(result, TableSet) or result.attrs.get("procedure") != "meta_diagnostics" or kind not in ("forest", "funnel"):
        raise AnalysisError("invalid_spec", "Pass meta_diagnostics output and kind='forest' or 'funnel'.")
    k = result.attrs["n_studies"]
    summary = result["summary"].iloc[0]
    if kind == "forest":
        points = [{"term": f"Study {r.study}", "estimate": float(r.yi), "ci_low": float(r.ci_low),
                   "ci_high": float(r.ci_high), "weight_percent": float(r.weight_percent)} for r in result["forest"].itertuples()]
        points.append({"term": "Pooled mean", "estimate": float(summary.estimate), "ci_low": float(summary.ci_low), "ci_high": float(summary.ci_high)})
        return PlotSpec("coefficients", "Study effects and pooled mean", "Effect (analysis scale)", "Study", points, k, k,
                        config={"meta": {"pooled_mean_row": True, "study_count": k, "sample_hash": result.attrs["sample_hash"]}})
    points = [{"x": float(r.effect), "y": float(r.standard_error)} for r in result["funnel"].itertuples()]
    maximum = max(p["y"] for p in points)*1.05
    c = critical(result.attrs["level"], "z", None)
    center = float(summary.estimate)
    # Lines are data-coordinate annotations understood by the shared renderer.
    lines = [{"type": "segment", "x0": center, "y0": 0., "x1": center+s*c*maximum, "y1": maximum} for s in (-1, 1)]
    return PlotSpec("scatter", "Funnel: descriptive asymmetry", "Effect (analysis scale)", "Standard error", points, k, k,
                    config={"options": {"annotations": lines, "ylim": [0., maximum],
                            "xlim": [min(center-c*maximum, min(p["x"] for p in points)), max(center+c*maximum, max(p["x"] for p in points))]}, "meta": {"center": center, "level": result.attrs["level"],
                            "reference": "sampling-normal pseudo-confidence bands; not publication-bias evidence", "sample_hash": result.attrs["sample_hash"]}})
