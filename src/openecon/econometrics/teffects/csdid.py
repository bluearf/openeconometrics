"""Callaway and Sant'Anna (2021) group-time average treatment effects (``csdid``).

Balanced panel of n units observed in periods t = 1..T; G_i is the first
treated period of unit i (missing = never treated). For every cohort g and
period t, with base period b (g - 1 for t >= g; t - 1 before treatment under
``base='varying'``, always g - 1 under ``'universal'``) and dY = Y_t - Y_b,

    ATT(g, t) = E[dY | G = g] - E[dY | control],

the control group being the never-treated units (``control='never'``) or the
units not yet treated at max(t, b) (``'notyet'``). With covariates x (taken
at the base period) the second term is estimated by

    reg   outcome regression: mean over cohort g of dY - x'beta, beta from OLS
          of dY on x among controls (Heckman, Ichimura and Todd 1997);
    ipw   normalized inverse-probability weighting with weights p/(1-p) from a
          logit of cohort membership on x (Abadie 2005);
    dr    doubly robust: both combined (Sant'Anna and Zhao 2020, drdid_panel).

Each ATT(g, t) solves stacked estimating equations on the cohort and its
controls; its per-unit influence contributions phi_i = -e'A^-1 psi_i (A the
analytic derivative of the summed equations) give V = Phi'Phi for all
ATT(g, t) jointly (analytic standard errors clustered by unit; ``did`` uses a
multiplier bootstrap by default). Aggregations (``did::aggte``) weight the
ATT(g, t) by estimated cohort shares P(G = g) and add the influence of those
estimated weights:

    simple    post-treatment ATT(g, t), weights P(G = g)
    dynamic   theta(e) = sum_g w_g ATT(g, g + e), overall = mean of theta(e), e >= 0
    group     theta(g) = mean over t >= g of ATT(g, t), overall weights P(G = g)
    calendar  theta(t) = sum_{g <= t} w_g ATT(g, t), overall = mean of theta(t)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, kernel_call, make_spec, wald_test,
)
from openecon.econometrics.teffects.common import require_periods
from openecon.econometrics.teffects.index import fit_index
from openecon.engines.distributions import normal_isf
from openecon.engines.linalg import collinear_columns, least_squares
from openecon.models import ModelSpec, ResultBundle

# Largest number of influence-function entries (units x ATT(g, t)) kept in memory.
_MAX_ENTRIES = 50_000_000


def _first_period(units: Tensor, n: int, start: Tensor, timing: str) -> Tensor:
    """First treated period of every unit (NaN: never treated); constant within units."""
    first = torch.full((n,), math.nan, dtype=torch.float64)
    first[units] = start
    check = torch.full((n,), math.nan, dtype=torch.float64)
    check[units.flip(0)] = start.flip(0)
    if not bool(((first == check) | (torch.isnan(first) & torch.isnan(check))).all()):
        raise AnalysisError("invalid_treatment", f"'{timing}' must be constant within each unit.")
    return first


def _cross(z1: Tensor, c: Tensor, z2: Tensor) -> Tensor:
    return (z1 * c[:, None]).T @ z2


def _att_gt(method: str, dy: Tensor, x: Tensor | None, treated: Tensor, control: Tensor,
            n: int) -> tuple[float, Tensor, list[int]]:
    """ATT(g, t), its influence contributions phi [n] (zero outside the 2x2 sample) and the
    covariate columns omitted as collinear within this comparison."""
    phi = torch.zeros(n, dtype=torch.float64)
    n1, n0 = int(treated.sum()), int(control.sum())
    if x is None:
        m1, m0 = dy[treated].mean(), dy[control].mean()
        phi[treated] = (dy[treated] - m1) / n1
        phi[control] = -(dy[control] - m0) / n0
        return float(m1 - m0), phi, []
    rows = treated | control
    xs = x[rows]
    xs = torch.cat([torch.ones((xs.shape[0], 1), dtype=torch.float64), xs - xs.mean(dim=0)],
                   dim=1)
    d = treated[rows].to(torch.float64)
    # Stata/R drop covariates that are constant or collinear in the comparison sample; the
    # outcome regression runs on the controls, so screen there too.
    kept, _ = kernel_call(collinear_columns, xs, None)
    local, _ = kernel_call(collinear_columns, xs[d < 1][:, kept], None)
    if len(local) < len(kept):
        if method != "reg":
            # constant among the controls but not in the comparison: the cohort is
            # (quasi-)separated and the propensity model has no finite maximum
            raise AnalysisError("overlap_violation", "A covariate is constant among the "
                                "control units of a group-time comparison but not among the "
                                "cohort, so the propensity-score model separates them; "
                                "remove or coarsen that covariate, or use method='reg'.")
        kept = [kept[i] for i in local]
    if kept[0] != 0:
        raise AnalysisError("invalid_spec", "The covariates of csdid are collinear with the "
                            "constant in a group-time comparison.")
    omitted = [i - 1 for i in range(1, x.shape[1] + 1) if i not in set(kept)]
    xs = xs[:, kept]
    if n0 <= xs.shape[1] and method in {"reg", "dr"}:
        raise AnalysisError("insufficient_observations", f"A group-time comparison has {n0} "
                            f"control unit(s) for {xs.shape[1]} outcome-regression parameters.")
    y = dy[rows]
    k = xs.shape[1]
    blocks: list[Tensor] = []
    names: list[str] = []
    if method in {"ipw", "dr"}:
        try:
            gamma = fit_index("logit", xs, d, torch.ones_like(d), what="propensity-score model")
        except AnalysisError as exc:
            if exc.code != "separation_detected":
                raise
            raise AnalysisError("overlap_violation", "The propensity-score model of a "
                                "group-time comparison separates the cohort from its controls; "
                                "remove or coarsen covariates.") from exc
        index = xs @ gamma
        p = torch.sigmoid(index)
        if bool((p[d < 1] > 1 - 1e-5).any()):
            raise AnalysisError("overlap_violation", "Some control units have estimated "
                                "propensity scores within 1e-5 of one (no overlap).")
        odds = torch.exp(index)
    beta = None
    if method in {"reg", "dr"}:
        fit = kernel_call(least_squares, xs[d < 1], y[d < 1], None, drop_collinear=False,
                          tol=0.0)
        beta = fit.beta
    resid = y - xs @ beta if beta is not None else y
    eta1 = float((d * resid).sum() / d.sum())
    size = (k if method in {"ipw", "dr"} else 0) + (k if beta is not None else 0) + 1 \
        + (1 if method != "reg" else 0)
    a = torch.zeros((size, size), dtype=torch.float64)
    offset = 0
    g_slice = b_slice = None
    if method in {"ipw", "dr"}:
        g_slice = slice(0, k)
        blocks.append(xs * (d - p)[:, None])
        a[g_slice, g_slice] = -_cross(xs, p * (1 - p), xs)
        offset = k
        names.append("gamma")
    if beta is not None:
        b_slice = slice(offset, offset + k)
        blocks.append(xs * ((1 - d) * (y - xs @ beta))[:, None])
        a[b_slice, b_slice] = -_cross(xs, 1 - d, xs)
        offset += k
    i1 = offset
    blocks.append((d * (resid - eta1))[:, None])
    a[i1, i1] = -d.sum()
    if b_slice is not None:
        a[i1, b_slice] = -(d @ xs)
    i0 = None
    if method != "reg":
        i0 = i1 + 1
        weight = (1 - d) * odds
        eta0 = float((weight * resid).sum() / weight.sum())
        blocks.append((weight * (resid - eta0))[:, None])
        a[i0, i0] = -weight.sum()
        a[i0, g_slice] = (weight * (resid - eta0)) @ xs
        if b_slice is not None:
            a[i0, b_slice] = -(weight @ xs)
        att = eta1 - eta0
    else:
        att = eta1
    psi = torch.cat(blocks, dim=1)
    contrast = torch.zeros(size, dtype=torch.float64)
    contrast[i1] = 1.0
    if i0 is not None:
        contrast[i0] = -1.0
    try:
        direction = torch.linalg.solve(a.T, contrast)
    except RuntimeError as exc:
        raise AnalysisError("singular_jacobian", "A group-time comparison has a singular "
                            "estimating-equation derivative; check the covariates.") from exc
    phi[rows] = -(psi @ direction)
    return att, phi, omitted


def _aggregate(att: Tensor, phi: Tensor, weights_of: Tensor, shares: Tensor,
               share_phi: Tensor) -> tuple[float, Tensor]:
    """theta = sum_k w_k att_k with w_k = c_k p_{g(k)} / sum_j c_j p_{g(j)}, and its phi.

    ``weights_of`` [K, G] maps each ATT(g, t) to its cohort (c_k times a one-hot row);
    ``shares`` [G] are the cohort shares and ``share_phi`` [n, G] their influence.
    """
    raw = weights_of @ shares                       # c_k p_g(k)
    total = raw.sum()
    weights = raw / total
    raw_phi = share_phi @ weights_of.T              # [n, K]
    weights_phi = (raw_phi * total - raw[None, :] * raw_phi.sum(dim=1, keepdim=True)) / total ** 2
    return float(weights @ att), phi @ weights + weights_phi @ att


def fit_csdid(spec: ModelSpec, data: Any) -> ResultBundle:
    from openecon.econometrics.registry import role_columns

    timing = role_columns(spec, "treatment_time")[0]
    frame = ModelFrame(spec, data, allow_missing=[timing])
    method, control, base = (frame.option(name) for name in ("method", "control", "base"))
    sample, anticipation = frame.option("sample"), frame.option("anticipation")
    rcs = sample == "repeated_cross_section"
    if rcs:
        if spec.panel is not None or spec.predictors:
            raise AnalysisError("unsupported_sample", "Repeated cross-sections currently validate independent rows without group= or covariates.")
        units, n = torch.arange(frame.n), frame.n
    else:
        if spec.panel is None:
            raise AnalysisError("invalid_spec", "Panel csdid requires group=; use sample='repeated_cross_section' for independent rows.")
        units, n = frame.codes(spec.panel)
        _first_period(units, n, frame.numeric(timing, allow_missing=True), timing)
        require_periods(frame)
    time = frame.time_index()
    start = frame.numeric(timing, allow_missing=True)
    initial_periods = torch.unique(time)
    early = ~torch.isnan(start) & (start <= initial_periods[min(anticipation, len(initial_periods) - 1)])
    if bool(early.any()):
        # did drops units that are already treated in the first period: no pre-period
        if bool(early.all()):
            raise AnalysisError("invalid_treatment", "Every unit is treated in the first period, "
                                "so no group-time effect is identified.")
        dropped = int(torch.unique(units[early]).numel())
        frame.restrict(~early, f"Excluded {dropped} unit(s) treated in or before the first "
                       "period (they have no pre-treatment period), as did does.")
        units, n = (torch.arange(frame.n), frame.n) if rcs else frame.codes(spec.panel)
        time = frame.time_index()
    periods, pcode = torch.unique(time, return_inverse=True)
    t_count = periods.numel()
    if n * t_count > _MAX_ENTRIES:
        raise AnalysisError("design_too_large", "Unit-by-period reconstruction exceeds the explicit influence design budget.")
    if sample == "balanced" and frame.n != n * t_count:
        raise AnalysisError("unbalanced_panel", "csdid needs a balanced panel: every unit "
                            "observed once in every period.")
    seen = torch.zeros((n, t_count), dtype=torch.int64)
    seen.index_put_((units, pcode), torch.ones_like(units), accumulate=True)
    if sample == "balanced" and not bool((seen == 1).all()):
        raise AnalysisError("unbalanced_panel", "csdid needs a balanced panel: every unit "
                            "observed once in every period.")
    y = torch.zeros((n, t_count), dtype=torch.float64)
    y[units, pcode] = frame.numeric(spec.outcome)
    first = _first_period(units, n, frame.numeric(timing, allow_missing=True), timing)
    covariates = None
    if spec.predictors:
        design = frame.design(intercept=False)
        covariates = torch.zeros((n, t_count, design.x.shape[1]), dtype=torch.float64)
        covariates[units, pcode] = design.x
        terms = design.terms
    never = torch.isnan(first)
    cohort_values = torch.unique(first[~never])
    late = cohort_values > periods[-1]
    if anticipation and bool(late.any()):
        raise AnalysisError("unsupported_sample", "Anticipation with cohorts beyond the observed time domain is not validated; encode genuine never-treated units explicitly.")
    if bool(late.any()):
        frame.warn("Units first treated after the last period are treated as never treated.")
        never = never | (first > periods[-1])
    cohorts = [float(g) for g in cohort_values[~late].tolist()]
    if not cohorts:
        raise AnalysisError("invalid_treatment", "No cohort is treated after the first period.")
    if bool((~torch.isin(cohort_values[~late], periods.to(torch.float64))).any()):
        raise AnalysisError("invalid_treatment", f"'{timing}' must be one of the observed "
                            "periods (or missing for never-treated units).")
    group_index = torch.full((n,), t_count + 1, dtype=torch.int64)   # never treated: beyond T
    treated_units = ~never
    group_index[treated_units] = torch.searchsorted(periods.to(torch.float64),
                                                    first[treated_units])
    if control == "never" and not bool(never.any()):
        raise AnalysisError("invalid_treatment", "control='never' needs never-treated units; "
                            "use control='notyet'.")
    pairs, atts, phis, decisions = [], [], [], []
    for g in cohorts:
        gi = int(torch.searchsorted(periods.to(torch.float64), torch.tensor(g)))
        cohort = group_index == gi
        for ti in range(t_count):
            baseline = gi - anticipation - 1
            if base == "universal" and ti == baseline:
                continue
            b = baseline if (ti >= gi - anticipation or base == "universal") else ti - 1
            if b < 0:
                continue
            controls = never.clone() if control == "never" else (never | (group_index > max(ti, b) + anticipation))
            controls &= ~cohort
            selected_cohort = cohort.clone()
            if not rcs:
                complete = (seen[:, ti] > 0) & (seen[:, b] > 0)
                controls &= complete
                selected_cohort &= complete
            decisions.append({"cohort": g, "period": float(periods[ti]), "base": float(periods[b]),
                              "treated_units": int(selected_cohort.sum()), "control_units": int(controls.sum()),
                              "selection": "period-specific independent cells" if rcs else "observed at both comparison periods"})
            if int(controls.sum()) < 2 or int(selected_cohort.sum()) < 2:
                frame.warn(f"ATT({g:g},{float(periods[ti]):g}) skipped: fewer than two control "
                           "units.")
                continue
            x = None if covariates is None else covariates[:, b]
            if rcs:
                att, phi = _att_rcs(frame.numeric(spec.outcome), pcode, ti, b, cohort, controls)
                omitted = []
            else:
                att, phi, omitted = _att_gt(method, y[:, ti] - y[:, b], x, selected_cohort, controls, n)
            for index in omitted:
                frame.warn(f"Omitted because of collinearity in some group-time comparisons: "
                           f"{terms[index]}.")
            pairs.append((g, float(periods[ti]), gi, ti))
            atts.append(att)
            phis.append(phi)
            if n * len(phis) > _MAX_ENTRIES:
                raise AnalysisError("design_too_large", "Too many units times group-time "
                                    "effects for the influence-function covariance.")
    if not atts:
        raise AnalysisError("insufficient_observations", "No group-time comparison has adequate observed sample support.")
    att = torch.tensor(atts, dtype=torch.float64)
    phi = torch.stack(phis, dim=1)
    covariance = phi.T @ phi
    cohort_ids = sorted({pair[2] for pair in pairs})
    onehot = torch.tensor([[1.0 if pair[2] == c else 0.0 for c in cohort_ids] for pair in pairs],
                          dtype=torch.float64)
    membership = torch.stack([(group_index == c).to(torch.float64) for c in cohort_ids], 1)
    shares = membership.mean(dim=0)
    share_phi = (membership - shares) / n
    extra = _aggregations(pairs, att, phi, onehot, shares, share_phi, frame.spec.alpha,
                          [float(periods[c]) for c in cohort_ids], store_influence=True)
    terms_out = [f"ATT({g:g},{t:g})" for g, t, _, _ in pairs]
    pre = [i for i, pair in enumerate(pairs) if pair[3] < pair[2] - anticipation]
    tests = {}
    if pre:
        tests["pretrends"] = wald_test(att, covariance, pre, label="Pre-treatment ATT(g,t) = 0 "
                                       "(parallel trends before treatment)")
    extra.update({"method": method, "control_group": control, "base_period": base,
                  "cohorts": cohorts, "never_treated_units": int(never.sum()),
                  "covariates": terms if covariates is not None else [],
                  "sample_contract": sample, "anticipation": anticipation,
                  "sample_decisions": decisions, "weight_population": "cohort shares among all retained units" if not rcs else "cohort shares among all retained independent rows"})
    bootstrap_info = _multiplier_bands(extra, att, phi, terms_out, frame)
    info = {"correction": "analytic influence-function covariance of the group-time ATTs; "
                          + ("independent rows" if rcs else "clustered by unit"),
            "df_inference": None, "small_sample_correction": None, **bootstrap_info}
    return build_result(
        frame, terms=terms_out, params=att, covariance=covariance,
        equations=[f"Cohort {pair[0]:g}" for pair in pairs],
        title="Callaway-Sant'Anna group-time ATT", use_t=False,
        metrics={"n_groups": n, "n_periods": t_count, "n_cohorts": len(cohort_ids),
                 "simple_att": extra["simple"]["estimate"],
                 "simple_att_se": extra["simple"]["std_error"]},
        solver="group_time_estimating_equations", inference=info, tests=tests, extra=extra,
        nobs=frame.n, categories=None, provenance={"units": n},
    )


def _row(label: Any, estimate: float, phi: Tensor, critical: float) -> dict[str, Any]:
    se = float(phi.square().sum().sqrt())
    return {"label": label, "estimate": estimate, "std_error": se,
            "ci_low": estimate - critical * se, "ci_high": estimate + critical * se,
            "_influence": phi}


def _att_rcs(y, pcode, current, baseline, treated, controls):
    """Unconditional repeated-cross-section DiD and four normalized-mean IFs."""
    phi = torch.zeros_like(y)
    att = 0.
    for group, period, sign in [(treated, current, 1), (treated, baseline, -1),
                                (controls, current, -1), (controls, baseline, 1)]:
        rows = group & (pcode == period)
        count = int(rows.sum())
        if count < 2:
            raise AnalysisError("insufficient_observations", "Repeated-cross-section ATT requires at least two observations in every treated/control by pre/post cell.")
        mean = y[rows].mean()
        att += sign * float(mean)
        phi[rows] += sign * (y[rows] - mean) / count
    return att, phi


def _multiplier_bands(extra, att, phi, terms, frame):
    from openecon.econometrics.postest.dependent_resampling import multipliers
    from openecon.econometrics.postest.resampling import _seed

    reps, uniform = frame.option("bootstrap_reps"), frame.option("uniform")
    if (reps == 1 or uniform) and reps < 2:
        raise AnalysisError("invalid_spec", "Multiplier inference and uniform bands require at least two bootstrap draws.")
    if reps * len(terms) > 2_000_000:
        raise AnalysisError("bootstrap_too_large", "Multiplier output exceeds the explicit 2,000,000-entry budget.")
    seed_value = _seed(frame.option("seed")) if reps else None
    draws = multipliers(len(phi), reps, seed_value, "normal") if reps else None
    record = {"bootstrap_reps": reps, "uniform": uniform, "multiplier": "normal" if reps else None,
              "se_method": "analytic shared influence functions", "families": {},
              "coefficient_intervals": "pointwise normal; simultaneous bands stored separately"}
    if reps:
        record["seed"] = seed_value
    families = {name: extra[name] for name in ["dynamic", "group", "calendar"]}
    families["att_gt"] = [dict(label=term, estimate=float(value), std_error=float(column.square().sum().sqrt()), _influence=column)
                          for term, value, column in zip(terms, att, phi.T, strict=True)]
    for name, rows in families.items():
        influence = torch.stack([row.pop("_influence") for row in rows], dim=1)
        if draws is None:
            continue
        se = influence.square().sum(dim=0).sqrt()
        noise = draws @ influence
        supported = se > 0
        maximum = (noise[:, supported] / se[supported]).abs().max(dim=1).values if bool(supported.any()) else torch.zeros(reps)
        critical = float(torch.quantile(maximum, 1 - frame.spec.alpha))
        record["families"][name] = {"terms": [row["label"] for row in rows], "critical_value": critical,
                                    "simultaneous": uniform}
        for j, row in enumerate(rows):
            tails = torch.quantile(noise[:, j], torch.tensor([frame.spec.alpha / 2, 1 - frame.spec.alpha / 2], dtype=torch.float64))
            row.update(pointwise_ci_low=row["estimate"] - float(tails[1]), pointwise_ci_high=row["estimate"] - float(tails[0]))
            if uniform:
                row.update(uniform_ci_low=row["estimate"] - critical * float(se[j]),
                           uniform_ci_high=row["estimate"] + critical * float(se[j]))
    if reps:
        extra["att_gt_bands"] = families["att_gt"]
    for value in extra.values():
        if isinstance(value, dict):
            value.pop("_influence", None)
    return record


def _aggregations(pairs, att: Tensor, phi: Tensor, onehot: Tensor, shares: Tensor,
                  share_phi: Tensor, alpha: float, cohort_periods: list[float], *,
                  store_influence: bool = False) -> dict[str, Any]:
    critical = normal_isf(alpha / 2)
    post = torch.tensor([pair[3] >= pair[2] for pair in pairs])
    out: dict[str, Any] = {}
    if not bool(post.any()):
        raise AnalysisError("invalid_treatment", "No post-treatment group-time effect is "
                            "identified.")
    estimate, influence = _aggregate(att, phi, onehot * post[:, None], shares, share_phi)
    out["simple"] = _row("simple", estimate, influence, critical)
    event = torch.tensor([pair[3] - pair[2] for pair in pairs])
    dynamic, dyn_phi = [], []
    for e in sorted(set(event.tolist())):
        chosen = (event == e).to(torch.float64)
        estimate, influence = _aggregate(att, phi, onehot * chosen[:, None], shares, share_phi)
        dynamic.append(_row(e, estimate, influence, critical))
        if e >= 0:
            dyn_phi.append((estimate, influence))
    out["dynamic"] = dynamic
    out["dynamic_overall"] = _row("dynamic", sum(v for v, _ in dyn_phi) / len(dyn_phi),
                                  sum(f for _, f in dyn_phi) / len(dyn_phi), critical)
    groups, group_values = [], []
    for column, period in enumerate(cohort_periods):
        chosen = post & (onehot[:, column] > 0)
        if not bool(chosen.any()):
            continue
        weights = chosen.to(torch.float64) / chosen.sum()
        estimate, influence = float(weights @ att), phi @ weights
        groups.append(_row(period, estimate, influence, critical))
        group_values.append((column, estimate, influence))
    out["group"] = groups
    columns = [c for c, _, _ in group_values]
    selected = shares[columns]
    total = selected.sum()
    values = torch.tensor([v for _, v, _ in group_values], dtype=torch.float64)
    influences = torch.stack([f for _, _, f in group_values], 1)
    weights_phi = (share_phi[:, columns] * total - selected[None, :]
                   * share_phi[:, columns].sum(dim=1, keepdim=True)) / total ** 2
    overall = float((selected / total) @ values)
    out["group_overall"] = _row("group", overall, influences @ (selected / total)
                                + weights_phi @ values, critical)
    calendar, cal_phi = [], []
    times = sorted({pair[1] for pair, flag in zip(pairs, post.tolist(), strict=True) if flag})
    for t in times:
        chosen = torch.tensor([pair[1] == t for pair in pairs]) & post
        estimate, influence = _aggregate(att, phi, onehot * chosen[:, None].to(torch.float64),
                                         shares, share_phi)
        calendar.append(_row(t, estimate, influence, critical))
        cal_phi.append((estimate, influence))
    out["calendar"] = calendar
    out["calendar_overall"] = _row("calendar", sum(v for v, _ in cal_phi) / len(cal_phi),
                                   sum(f for _, f in cal_phi) / len(cal_phi), critical)
    if not store_influence:
        for value in out.values():
            for row in value if isinstance(value, list) else [value]:
                row.pop("_influence", None)
    return out


def csdid(*, data: Any, y: str, group: str | None = None, time: str, treatment_time: str,
          x: Sequence[str] | None = None, method: str = "dr", control: str = "never",
          base: str = "varying", categorical: Sequence[str] | None = None,
          missing: str = "raise", alpha: float = 0.05, sample: str = "balanced",
          anticipation: int = 0, bootstrap_reps: int = 0, seed: int | None = None,
          uniform: bool = False) -> ResultBundle:
    """Callaway-Sant'Anna (2021) group-time ATTs and their aggregations (``csdid`` / R ``did``).

    For a balanced panel of units (``group``) over integer periods (``time``)
    with staggered adoption (``treatment_time``: the first treated period,
    missing for never-treated units), estimates ATT(g, t), the average effect
    in period t for the cohort first treated in g, by comparing the cohort's
    change Y_t - Y_b with that of a control group: never-treated units
    (``control='never'``, default) or units not yet treated
    (``'notyet'``). The base period b is g - 1 after treatment; before
    treatment ``base='varying'`` (default) uses t - 1 (short differences, as
    ``did``) and ``'universal'`` uses g - 1. Covariates ``x`` (time-invariant
    in principle; their base-period values are used) enter through
    ``method``: ``'reg'`` outcome regression, ``'ipw'`` normalized inverse
    probability weighting or ``'dr'`` (default) the doubly robust estimator of
    Sant'Anna and Zhao (2020). Heterogeneous effects across cohorts and time
    do not bias these estimates, unlike two-way fixed effects.

    Standard errors come from influence functions of the estimating equations
    (clustered by unit, z inference); bootstrap_reps optionally adds shared
    normal multiplier pointwise/uniform bands in the extra tables. Propensity
    scores are not trimmed (an estimated
    control propensity within 1e-5 of one raises ``overlap_violation``).

    The result's terms are ``ATT(g,t)`` (equations ``Cohort g``) with their
    joint covariance; ``tests['pretrends']`` is the Wald test that every
    pre-treatment ATT(g, t) is zero; ``extra`` holds the aggregations of
    ``did::aggte`` with standard errors that include the estimated cohort
    weights: ``simple`` (post-treatment average weighted by cohort size),
    ``dynamic`` (event-time effects and their post-treatment mean),
    ``group`` (cohort averages and their size-weighted mean) and
    ``calendar`` (period effects and their mean). ``metrics['simple_att']`` is
    the simple aggregate.

    ``sample='unbalanced'`` uses pair-complete units per comparison (no attrition
    adjustment); ``'repeated_cross_section'`` validates independent rows without
    group= or covariates. ``anticipation`` shifts the clean baseline and control
    eligibility. ``extra['sample_decisions']`` records support and selection.
    Simultaneous families are recorded separately in inference, with the actual
    local seed and analytic-SE normalization. See docs/econometrics/teffects.md
    for the explicit differences from R did's unbalanced and scale conventions.

    Stata: ``csdid y x, ivar(id) time(year) gvar(first) method(dripw)``
    (``notyet`` option); ``hdidregress aipw``. R: ``did::att_gt`` +
    ``aggte``.

    Example::

        import openecon as oe
        cs = oe.csdid(data=df, y="lemp", group="county", time="year",
                      treatment_time="first_treat", control="notyet")
        cs.extra["dynamic"], cs.extra["simple"]
    """
    from openecon.analysis import fit

    spec = make_spec(
        "csdid", outcome=y, predictors=column_list(x, "x"), panel=group, time=time,
        categorical=column_list(categorical, "categorical"), missing=missing, alpha=alpha,
        columns={"treatment_time": treatment_time},
        options={"method": method, "control": control, "base": base, "sample": sample,
                 "anticipation": anticipation, "bootstrap_reps": bootstrap_reps, "seed": seed, "uniform": uniform},
    )
    return fit(spec, data=data)
