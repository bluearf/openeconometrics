"""Two-way fixed-effects difference in differences and event studies.

didregress (Stata's ``didregress`` / ``xtdidregress``) fits

    y_it = a_g + c_t + delta D_it + x_it'b + e_it

with group effects a_g and time effects c_t absorbed (``engines.absorb``:
alternating projections accelerated by conjugate gradients, no dummy matrix);
``D`` is the 0/1 treated-and-post indicator and delta is the ATET. The slopes
are QR least squares on the demeaned columns (Frisch-Waugh-Lovell).

eventstudy replaces ``delta D`` by relative-time indicators
``1{t - T0_g = e}`` for treated groups (``T0_g`` the first treated period, never-
treated groups have none), e = -leads..lags without the reference period;
with ``leads``/``lags`` the end points are binned (e <= -leads, e >= lags).

Covariance: ``robust`` (default) is the cluster-robust sandwich with clusters
at the group level, Stata's default ``vce(cluster groupvar)``; ``cluster``
clusters on another column; ``HC1`` and ``nonrobust`` are the usual
regression covariances. Cluster small-sample factor G/(G-1) (N-1)/(N-K) with
t(G-1) inference; K counts the slopes and the absorbed effects that are not
nested in the clusters (reghdfe's rule: group effects nested in group clusters
are free, time effects count), see ``docs/econometrics/teffects.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis import _json_scalar
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, kernel_call, linear_covariance, make_spec,
    wald_test,
)
from openecon.econometrics.linear.common import estimation_weights, nested_in_clusters
from openecon.econometrics.teffects.common import require_periods
from openecon.engines import absorb, linalg
from openecon.engines.inference import critical_value
from openecon.models import ModelSpec, ResultBundle

_TWFE_NOTE = ("Two-way fixed-effects estimates average heterogeneous effects with weights that "
              "can be negative under staggered adoption (Goodman-Bacon 2021; de Chaisemartin and "
              "D'Haultfoeuille 2020; Sun and Abraham 2021); interpret them as a weighted average "
              "only when effects are homogeneous across cohorts and periods, or use oe.csdid.")


@dataclass
class _Panel:
    groups: Tensor             # int64 group codes
    n_groups: int
    periods: Tensor            # int64 period codes in sorted time order
    n_periods: int
    weights: Tensor | None
    nobs: int
    labels: list[Any]              # time value of each period code (JSON scalars)
    numeric_time: Tensor | None    # float64 time values per row when the column is numeric


@dataclass
class _Fit:
    design: Design
    beta: Tensor
    covariance: Tensor
    info: dict[str, Any]
    df_inference: float
    df_resid: float
    ssr: float
    demeaned_y: Tensor
    resid: Tensor
    absorbed: int


def _panel(frame: ModelFrame) -> _Panel:
    spec = frame.spec
    groups, n_groups = frame.codes(spec.panel)
    try:
        values, unique = pd.factorize(frame.series(spec.time), sort=True)
    except TypeError as exc:
        raise AnalysisError("invalid_time", f"Time column '{spec.time}' must hold sortable "
                            "values (numbers or dates).") from exc
    weights, nobs = estimation_weights(frame)
    if spec.weight_type == "pweight" and spec.covariance == "nonrobust":
        raise AnalysisError("unsupported_covariance", "pweights need a robust or cluster "
                            "covariance.")
    if n_groups < 2 or len(unique) < 2:
        raise AnalysisError("insufficient_observations", "Difference in differences needs at "
                            "least two groups and two periods.")
    series = frame.series(spec.time)
    numeric = None
    if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
        numeric = frame.numeric(spec.time)
    labels = [value.isoformat() if isinstance(value, pd.Timestamp) else _json_scalar(value)
              for value in unique]
    return _Panel(groups, n_groups, torch.from_numpy(values.astype("int64")), len(unique),
                  weights, nobs, labels, numeric)


def _twfe(frame: ModelFrame, panel: _Panel, y: Tensor, design: Design) -> _Fit:
    """Absorb group and time effects, solve by QR, covariance by ``spec.covariance``."""
    spec = frame.spec
    dims = [(panel.groups, panel.n_groups), (panel.periods, panel.n_periods)]
    block = torch.cat([y[:, None], design.x], dim=1)
    demeaned = kernel_call(absorb.demean, block, dims, panel.weights).values
    yd = demeaned[:, 0]
    design, xd = frame.drop_absorbed(design, demeaned[:, 1:], panel.weights)
    if xd.shape[1] == 0:
        raise AnalysisError("no_within_variation", "Every regressor is absorbed by the group and "
                            "time effects.")
    fit = kernel_call(linalg.least_squares, xd, yd, panel.weights, drop_collinear=False, tol=0.0)
    kind = spec.covariance
    clusters = names = None
    cluster_dimensions = []
    if kind == "robust":
        kind, clusters, names = "cluster", [(panel.groups, panel.n_groups)], [spec.panel]
        cluster_dimensions = [(panel.groups, panel.n_groups)]
    elif kind == "cluster":
        dimensions = frame.cluster_dimensions()
        cluster_dimensions = dimensions
    if spec.covariance == "robust" and panel.n_groups < 30:
        frame.warn(f"Only {panel.n_groups} groups (clusters); cluster-robust inference may be "
                   "unreliable.")
    nested = nested_in_clusters(dims, cluster_dimensions) if cluster_dimensions else [False]*len(dims)
    counted = [dimension for dimension, inside in zip(dims, nested, strict=True) if not inside]
    absorbed = absorb.absorbed_degrees_of_freedom(counted).total + int(all(nested))
    k = xd.shape[1] + absorbed
    df_resid = panel.nobs - k
    if df_resid <= 0:
        raise AnalysisError("insufficient_observations", "The two-way fixed-effects regression "
                            "leaves no residual degrees of freedom.")
    covariance, info = linear_covariance(
        frame, x=xd, resid=fit.resid, bread=fit.xtx_inv, n=panel.nobs, k=k, df_resid=df_resid,
        weights=panel.weights, kind=kind, clusters=clusters, cluster_names=names)
    if spec.covariance == "robust":
        info["covariance"] = "robust"
        info["correction"] = "cluster-robust at the group level (" + info["correction"] + ")"
    info["k_small_sample"] = k
    return _Fit(design, fit.beta, covariance, info, info.get("df_inference", df_resid), df_resid,
                float(fit.ssr), yd, fit.resid, absorbed)


def _sum_squares(values: Tensor, weights: Tensor | None) -> float:
    return float((values.square() * weights).sum() if weights is not None
                 else values.square().sum())


def _metrics(frame: ModelFrame, panel: _Panel, y: Tensor, fit: _Fit) -> dict[str, Any]:
    w = panel.weights
    mean = (y * w).sum() / w.sum() if w is not None else y.mean()
    tss = _sum_squares(y - mean, w)
    within = _sum_squares(fit.demeaned_y, w)
    metrics = {
        "r_squared": 1 - fit.ssr / tss if tss > 0 else None,
        "r_squared_within": 1 - fit.ssr / within if within > 0 else None,
        "rmse": math.sqrt(fit.ssr / fit.df_resid), "df_resid": fit.df_resid,
        "n_groups": panel.n_groups, "n_periods": panel.n_periods,
    }
    if "cluster_count" in fit.info:
        metrics["n_clusters"] = fit.info["cluster_count"]
    return metrics


def _covariates(frame: ModelFrame) -> Design:
    return frame.design(intercept=False)


# ---- didregress ---------------------------------------------------------------------


def _adoption(panel: _Panel, d: Tensor) -> tuple[Tensor, Tensor] | None:
    """First treated period of each group and whether treatment is absorbing.

    Returns (first period per group, -1 for never treated; treated-group flag) when
    D is constant within group-period cells and, once on, stays on; None otherwise.
    """
    big = panel.n_periods + 1
    first = torch.full((panel.n_groups,), big, dtype=torch.int64)
    on = d > 0.5
    first.scatter_reduce_(0, panel.groups[on], panel.periods[on], "amin")
    implied = (panel.periods >= first[panel.groups]).to(d.dtype)
    if not bool((implied == d).all()):
        return None
    treated = first < big
    return torch.where(treated, first, -1), treated


def _augmented(frame: ModelFrame, panel: _Panel, y: Tensor, base: Design, columns: Tensor,
               names: list[str], label: str) -> dict[str, Any]:
    design = Design(torch.cat([base.x, columns], dim=1), [*base.terms, *names], {}, False)
    fit = _twfe(frame, panel, y, design)
    tested = [i for i, term in enumerate(fit.design.terms) if term in set(names)]
    if not tested:
        return {"statistic": None, "df": 0, "p_value": None, "distribution": "F", "label": label,
                "note": "the test regressors are collinear with the fixed effects"}
    return wald_test(fit.beta, fit.covariance, tested, df_resid=fit.df_inference, label=label)


def _did_tests(frame: ModelFrame, panel: _Panel, y: Tensor, base: Design, d: Tensor
               ) -> tuple[dict[str, Any], dict[str, Any]]:
    timing = _adoption(panel, d)
    if timing is None:
        return {}, {"adoption": "not absorbing", "tests_note": "the treatment indicator is not "
                    "constant within group-period cells or switches off; the parallel-trends "
                    "and anticipation tests need an absorbing group-level treatment"}
    first, treated = timing
    starts = torch.unique(first[treated])
    info: dict[str, Any] = {"treated_groups": int(treated.sum()),
                            "control_groups": int((~treated).sum()),
                            "adoption": "common" if len(starts) == 1 else "staggered",
                            "first_treated_periods": [panel.labels[i] for i in starts.tolist()]}
    if len(starts) != 1 or int((~treated).sum()) == 0:
        info["tests_note"] = ("the parallel-trends and anticipation tests need one common "
                              "treatment period and never-treated controls; use eventstudy")
        return {}, info
    start = int(starts[0])
    if start < 2:
        info["tests_note"] = "at least two pre-treatment periods are needed for the tests"
        return {}, info
    group_treated = treated[panel.groups].to(torch.float64)
    # The linear trend is in the time variable itself (unequal spacing matters); a
    # non-numeric time column (dates, labels) uses the period index 0, 1, 2, ...
    period = panel.periods.to(torch.float64) if panel.numeric_time is None \
        else panel.numeric_time - panel.numeric_time.min()
    pre = (panel.periods < start).to(torch.float64)
    trend = (group_treated * period * pre)[:, None]
    tests = {"parallel_trends": _augmented(
        frame, panel, y, base, trend, ["_pretrend"],
        "Parallel-trends test: treated-group linear trend before treatment = 0 (estat ptrends)")}
    leads = torch.stack([group_treated * (panel.periods == start - lead).to(torch.float64)
                         for lead in range(1, start)], dim=1)
    tests["granger"] = _augmented(
        frame, panel, y, base, leads, [f"_lead{lead}" for lead in range(1, start)],
        "Anticipation (Granger) test: treatment leads = 0 (estat granger)")
    return tests, info


def fit_didregress(spec: ModelSpec, data: Any) -> ResultBundle:
    frame = ModelFrame(spec, data)
    panel = _panel(frame)
    treatment = frame.role("treatment")[0]
    if treatment in spec.predictors:
        raise AnalysisError("invalid_spec", "The treatment indicator must not also be a covariate.")
    d = frame.numeric(treatment)
    if bool(((d != 0) & (d != 1)).any()):
        raise AnalysisError("invalid_treatment", f"'{treatment}' must be a 0/1 indicator of "
                            "treated observations in treated periods.")
    if not bool((d == 1).any()) or not bool((d == 0).any()):
        raise AnalysisError("invalid_treatment", f"'{treatment}' must contain treated and "
                            "untreated observations.")
    y = frame.numeric(spec.outcome)
    covariates = _covariates(frame)
    term = f"ATET:r1vs0.{treatment}"
    design = Design(torch.cat([d[:, None], covariates.x], dim=1), [term, *covariates.terms],
                    covariates.categories, False)
    fit = _twfe(frame, panel, y, design)
    if fit.design.terms[0] != term:
        raise AnalysisError("no_within_variation", f"'{treatment}' is collinear with the group "
                            "and time effects, so the ATET is not identified (every group is "
                            "treated, or treatment starts at the same time in all groups and "
                            "there are no controls).")
    tests, timing = _did_tests(frame, panel, y, design, d)
    if timing.get("adoption") == "staggered":
        frame.warn(_TWFE_NOTE)
    extra = {"treatment": treatment, "group": spec.panel, "time": spec.time,
             "absorbed_degrees_of_freedom": fit.absorbed, **timing}
    equations = ["ATET"] + ["Controls"] * (len(fit.design.terms) - 1)
    return build_result(
        frame, terms=fit.design.terms, params=fit.beta, covariance=fit.covariance,
        equations=equations, title="Difference-in-differences regression", use_t=True,
        df_inference=fit.df_inference, df_resid=fit.df_resid,
        metrics=_metrics(frame, panel, y, fit), solver="absorb_twfe_qr", inference=fit.info,
        tests=tests, extra=extra, categories=covariates.categories, nobs=panel.nobs,
        provenance={"absorbed": [spec.panel, spec.time]},
    )


def didregress(*, data: Any, y: str, treatment: str, group: str, time: str,
               x: Sequence[str] | None = None, categorical: Sequence[str] | None = None,
               covariance: str | None = None, cluster: str | None = None,
               weights: str | None = None, weight_type: str | None = None,
               missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Difference in differences by two-way fixed effects (Stata ``didregress``/``xtdidregress``).

    Model: y_it = a_g + c_t + delta D_it + x_it'b + e_it for observations i of
    group g in period t (repeated cross-sections or a panel), with group
    effects a_g, time effects c_t and the treatment indicator D (1 for treated
    groups in treated periods). delta is the average treatment effect on the
    treated (ATET) under parallel trends. Both sets of effects are absorbed
    exactly (alternating projections, no dummies) and the slopes come from QR
    least squares on the demeaned data.

    Parameters: ``data``; ``y`` the outcome; ``treatment`` the 0/1 treated-and-
    post indicator; ``group`` and ``time`` the group and period columns;
    ``x`` covariates (``categorical`` treatment-coded); ``weights`` with
    ``weight_type`` ``'aweight'``, ``'fweight'`` or ``'pweight'``; ``missing``;
    ``alpha``.

    Covariance: ``'robust'`` (default) is the cluster-robust sandwich at the
    group level (Stata's default ``vce(cluster groupvar)``) with
    G/(G-1) (N-1)/(N-K) and t(G-1) inference; ``'cluster'`` with ``cluster``
    clusters on another column; ``'HC1'`` and ``'nonrobust'`` give the usual
    regression covariances with N - K degrees of freedom. K counts the
    covariates and the absorbed effects not nested in the clusters.

    Tests (``tests``), when treatment starts in one common period for all
    treated groups, treatment is absorbing and never-treated controls exist:
    ``parallel_trends`` (Stata's ``estat ptrends``: F test that a linear
    trend specific to the treated groups before treatment is zero) and
    ``granger`` (``estat granger``: joint F test of treatment leads, i.e. of
    anticipation effects). Both are computed from augmented regressions with
    the same covariance. ``extra`` reports the adoption pattern, treated and
    control group counts and first treated periods; ``metrics`` R-squared,
    within R-squared, rmse, n_groups, n_periods and n_clusters. Staggered
    adoption adds a warning about the known bias of two-way fixed effects
    under heterogeneous effects (see ``oe.eventstudy`` and the docs).

    Stata: ``didregress (y x1) (treated), group(state) time(year)``;
    ``xtdidregress (y x1) (treated), group(state) time(year)``.

    Example::

        import openecon as oe
        did = oe.didregress(data=df, y="outcome", treatment="treated_post",
                            group="county", time="year", x=["income"])
        print(did.summary())
        did.tests["parallel_trends"]
    """
    from openecon.analysis import fit

    spec = make_spec(
        "didregress", outcome=y, predictors=column_list(x, "x"), panel=group, time=time,
        categorical=column_list(categorical, "categorical"), covariance=covariance,
        cluster=cluster, weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"treatment": treatment},
    )
    return fit(spec, data=data)


# ---- event study --------------------------------------------------------------------


def _event_label(e: int) -> str:
    return f"lead{-e}" if e < 0 else f"lag{e}"


def fit_eventstudy(spec: ModelSpec, data: Any) -> ResultBundle:
    timing_column = registry_role(spec, "treatment_time")
    frame = ModelFrame(spec, data, allow_missing=[timing_column])
    panel = _panel(frame)
    require_periods(frame)
    time = frame.time_index().to(torch.float64)
    start = frame.numeric(timing_column, allow_missing=True)
    never = torch.isnan(start)
    if bool((start[~never] != start[~never].round()).any()):
        raise AnalysisError("invalid_time", f"'{timing_column}' must hold integer periods in the "
                            f"units of '{spec.time}' (missing for never-treated groups).")
    filled = torch.where(never, torch.full_like(start, -1e18), start)
    low = torch.full((panel.n_groups,), math.inf, dtype=torch.float64)
    high = torch.full((panel.n_groups,), -math.inf, dtype=torch.float64)
    low.scatter_reduce_(0, panel.groups, filled, "amin")
    high.scatter_reduce_(0, panel.groups, filled, "amax")
    if not bool((low == high).all()):
        raise AnalysisError("invalid_treatment", f"'{timing_column}' must be constant within each "
                            "group (the first treated period of the group, missing if never "
                            "treated).")
    relative = (time - filled).to(torch.int64)
    treated = ~never
    if not bool(treated.any()):
        raise AnalysisError("invalid_treatment", "No group is ever treated.")
    observed = relative[treated]
    leads, lags = frame.option("leads"), frame.option("lags")
    reference = int(frame.option("reference"))
    first = -int(leads) if leads is not None else int(observed.min())
    last = int(lags) if lags is not None else int(observed.max())
    if not first <= reference <= last:
        raise AnalysisError("invalid_spec", f"The reference period {reference} lies outside the "
                            f"event window [{first}, {last}].")
    binned = relative.clamp(first, last)
    events = [e for e in range(first, last + 1) if e != reference]
    columns = torch.stack([(treated & (binned == e)).to(torch.float64) for e in events], dim=1)
    counts = columns.sum(dim=0)
    keep = [i for i in range(len(events)) if counts[i] > 0]
    if len(keep) < len(events):
        frame.warn("No treated observation at relative time(s) "
                   f"{', '.join(str(events[i]) for i in range(len(events)) if i not in keep)}; "
                   "their indicators are omitted.")
    events = [events[i] for i in keep]
    columns = columns[:, keep]
    covariates = _covariates(frame)
    names = [_event_label(e) for e in events]
    design = Design(torch.cat([columns, covariates.x], dim=1), [*names, *covariates.terms],
                    covariates.categories, False)
    y = frame.numeric(spec.outcome)
    fit = _twfe(frame, panel, y, design)
    cohorts = torch.unique(filled[treated])
    if len(cohorts) > 1:
        frame.warn(_TWFE_NOTE)
    kept = {term: i for i, term in enumerate(fit.design.terms)}
    lead_index = [kept[_event_label(e)] for e in events if e < 0 and _event_label(e) in kept]
    tests = {}
    if lead_index:
        tests["pretrends"] = wald_test(fit.beta, fit.covariance, lead_index,
                                       df_resid=fit.df_inference,
                                       label="Joint pre-trend test: all leads = 0")
    critical = critical_value(spec.alpha, fit.df_inference)
    table = []
    for e in range(first, last + 1):
        row: dict[str, Any] = {"relative_time": e, "term": _event_label(e),
                               "binned": (e == first and leads is not None
                                          and bool((observed < first).any()))
                               or (e == last and lags is not None
                                   and bool((observed > last).any())),
                               "reference": e == reference,
                               "observations": int((treated & (binned == e)).sum())}
        if e == reference:
            row.update({"estimate": 0.0, "std_error": 0.0, "ci_low": 0.0, "ci_high": 0.0})
        elif _event_label(e) in kept:
            i = kept[_event_label(e)]
            se = float(fit.covariance[i, i].sqrt())
            estimate = float(fit.beta[i])
            row.update({"estimate": estimate, "std_error": se, "ci_low": estimate - critical * se,
                        "ci_high": estimate + critical * se})
        else:
            row.update({"estimate": None, "std_error": None, "ci_low": None, "ci_high": None})
        table.append(row)
    post = [kept[_event_label(e)] for e in events if e >= 0 and _event_label(e) in kept]
    extra: dict[str, Any] = {
        "event_table": table, "reference": reference, "window": [first, last],
        "cohorts": [int(c) for c in cohorts.tolist()],
        "never_treated_groups": int(torch.unique(panel.groups[never]).numel()),
        "treated_groups": int(torch.unique(panel.groups[treated]).numel()),
        "absorbed_degrees_of_freedom": fit.absorbed,
    }
    if post:
        weight = torch.zeros(len(fit.beta), dtype=torch.float64)
        weight[post] = 1.0 / len(post)
        estimate = float(weight @ fit.beta)
        se = float((weight @ fit.covariance @ weight).sqrt())
        extra["average_post_effect"] = {"estimate": estimate, "std_error": se,
                                        "ci_low": estimate - critical * se,
                                        "ci_high": estimate + critical * se,
                                        "periods": [e for e in events if e >= 0]}
    equations = ["Event" if term in set(names) else "Controls" for term in fit.design.terms]
    return build_result(
        frame, terms=fit.design.terms, params=fit.beta, covariance=fit.covariance,
        equations=equations, title="Event-study regression", use_t=True,
        df_inference=fit.df_inference, df_resid=fit.df_resid,
        metrics=_metrics(frame, panel, y, fit), solver="absorb_twfe_qr", inference=fit.info,
        tests=tests, extra=extra, categories=covariates.categories, nobs=panel.nobs,
        provenance={"absorbed": [spec.panel, spec.time]},
    )


def registry_role(spec: ModelSpec, name: str) -> str:
    from openecon.econometrics.registry import role_columns

    return role_columns(spec, name)[0]


def eventstudy(*, data: Any, y: str, group: str, time: str, treatment_time: str,
               x: Sequence[str] | None = None, leads: int | None = None,
               lags: int | None = None, reference: int = -1,
               categorical: Sequence[str] | None = None, covariance: str | None = None,
               cluster: str | None = None, weights: str | None = None,
               weight_type: str | None = None, missing: str = "raise",
               alpha: float = 0.05) -> ResultBundle:
    """Dynamic two-way fixed-effects event study.

    Model: y_it = a_g + c_t + sum_{e != ref} beta_e 1{t - T0_g = e} + x_it'b + e_it,
    where T0_g (``treatment_time``) is the first treated period of group g and
    is missing for never-treated groups, which (with the not-yet-treated
    observations of later cohorts) serve as controls through the time
    effects. beta_e traces the effect e periods after (lags, e >= 0) or before
    (leads, e < 0) treatment relative to the ``reference`` period (default -1,
    whose coefficient is normalized to zero).

    ``leads``/``lags`` bound the event window: relative times beyond it are
    binned into the end points (e <= -leads into lead ``leads``, e >= lags into
    lag ``lags``), the usual practice (Schmidheiny and Siegloch 2023). Without
    them every observed relative time gets its own indicator; then, if there
    are no never-treated groups, two indicators are not identified and the
    collinearity screen omits the last one (recorded in the warnings).
    ``time`` and ``treatment_time`` must be integer periods (or ``time`` a
    datetime when ``treatment_time`` holds the matching period index).

    Covariance as in ``oe.didregress`` (default: cluster-robust at the group
    level, t(G-1) inference). ``tests['pretrends']`` is the joint F test that
    all lead coefficients are zero. ``extra['event_table']`` is the table for
    an event-study plot (relative time, estimate, std_error, ci_low, ci_high,
    reference and binned flags, treated observations), with the reference
    period at zero; ``extra['average_post_effect']`` the mean of the lag
    coefficients with its standard error.

    Limitation: with staggered adoption and effects that vary across cohorts
    or over time, TWFE event-study coefficients are contaminated by effects
    from other relative periods (Sun and Abraham 2021) and the leads need not
    be zero even under parallel trends; a warning is added when more than one
    cohort is present. Use ``oe.csdid`` (Callaway-Sant'Anna group-time ATTs and
    their event-time aggregation) in that case; Sun-Abraham weights are not
    implemented.

    Stata (community): ``eventdd``, ``reghdfe y lead* lag*, absorb(id year)
    vce(cluster id)``.

    Example::

        import openecon as oe
        es = oe.eventstudy(data=df, y="outcome", group="county", time="year",
                           treatment_time="first_year", leads=4, lags=5)
        es.tests["pretrends"], es.extra["event_table"]
    """
    from openecon.analysis import fit

    spec = make_spec(
        "eventstudy", outcome=y, predictors=column_list(x, "x"), panel=group, time=time,
        categorical=column_list(categorical, "categorical"), covariance=covariance,
        cluster=cluster, weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"treatment_time": treatment_time},
        options={"leads": leads, "lags": lags, "reference": reference},
    )
    return fit(spec, data=data)
