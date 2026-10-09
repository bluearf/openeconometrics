"""Hausman–Taylor IV with explicit panel-effect exogeneity and moment components.

Pilot within OLS; equal-panel between IV; sigma_e²=RSS_within/(N-G),
sigma_u²=max(0, mean(panel pilot residual²)-sigma_e²/T_harmonic).
Final instruments: within X1/X2, panel means X1, invariant Z1 and constant.
No equivalence to alternative finite-sample variance-component conventions is claimed.
"""

from __future__ import annotations

import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame,
    build_result,
    column_list,
    kernel_call,
    make_spec,
    wald_test,
)
from openecon.econometrics.iv.common import Setting, covariance
from openecon.econometrics.regularized.kernels import work_guard
from openecon.engines import linalg
from openecon.engines.absorb import group_means
from openecon.models import ResultBundle
from openecon.resources import tensor_bytes

_ROLES = ("varying_exogenous", "varying_endogenous", "invariant_exogenous", "invariant_endogenous")


def htaylor_moment(
    *,
    data,
    y,
    panel,
    time=None,
    varying_exogenous=(),
    varying_endogenous=(),
    invariant_exogenous=(),
    invariant_endogenous=(),
    covariance="nonrobust",
    cluster=None,
    small=False,
    missing="raise",
    alpha=0.05,
    max_work=1000000000,
):
    from openecon.analysis import fit

    columns = dict(
        zip(
            _ROLES,
            (
                column_list(varying_exogenous, "varying_exogenous"),
                column_list(varying_endogenous, "varying_endogenous"),
                column_list(invariant_exogenous, "invariant_exogenous"),
                column_list(invariant_endogenous, "invariant_endogenous"),
            ),
            strict=True,
        )
    )
    return fit(
        make_spec(
            "htaylor_moment",
            outcome=y,
            predictors=[],
            panel=panel,
            time=time,
            covariance=covariance,
            cluster=cluster,
            missing=missing,
            alpha=alpha,
            columns=columns,
            options={"small": small, "max_work": max_work},
        ),
        data=data,
    )


def _iv(x, y, instruments):
    # Two compact QR projections, never an N by N projection matrix.
    first = kernel_call(linalg.least_squares, instruments, x, drop_collinear=False)
    second = kernel_call(linalg.least_squares, first.fitted, y, drop_collinear=False)
    return second.beta, second.xtx_inv, first.fitted


def fit_htaylor(spec, data):
    frame = ModelFrame(spec, data)
    if spec.predictors:
        raise AnalysisError(
            "invalid_spec",
            "Hausman–Taylor regressors belong in the four declared roles, not predictors.",
        )
    blocks = [frame.role(role) for role in _ROLES]
    names = sum(blocks, [])
    if (
        len(names) != len(set(names))
        or spec.outcome in names
        or spec.panel in names
        or spec.time in names
    ):
        raise AnalysisError(
            "invalid_roles",
            "HT regressor roles must be disjoint from each other, outcome and panel/time.",
        )
    x1, x2, z1, z2 = blocks
    if not x1 and not x2:
        raise AnalysisError("invalid_roles", "HT needs at least one time-varying regressor.")
    if len(x1) < len(z2):
        raise AnalysisError(
            "underidentified",
            "HT order condition: varying exogenous count must cover invariant endogenous count.",
        )
    n, k = frame.n, len(names) + 1
    work_guard(n * (k + len(x1) + 1) ** 2 * 12, frame.option("max_work"), "Hausman–Taylor IV")
    frame.workspace_plan(
        "Hausman–Taylor transforms and projections",
        {
            "design_instruments_qr_and_scores": tensor_bytes((n, k + len(x1) + 1), itemsize=144),
            "covariance": tensor_bytes((k, k), itemsize=80),
        },
    )
    frame.sort_panel()
    codes, groups = frame.codes(spec.panel)
    sizes = torch.bincount(codes, minlength=groups).to(torch.float64)
    if groups < 2 or n <= groups + len(x1) + len(x2) or n <= k:
        raise AnalysisError(
            "insufficient_observations",
            "HT requires multiple panels and positive within/overall degrees of freedom.",
        )
    allx = frame.matrix(names)
    means = group_means(allx, codes, groups)
    deviations = allx - means[codes]
    for j, name in enumerate(names):
        within = float(deviations[:, j].square().sum())
        scale = float((allx[:, j] - allx[:, j].mean()).square().sum())
        is_varying = within > 1e-13 * max(scale, 1e-28)
        if is_varying != (name in x1 + x2):
            raise AnalysisError(
                "invalid_roles",
                f"'{name}' disagrees with its varying/invariant role in the retained estimation sample.",
            )
    varying = len(x1) + len(x2)
    y = frame.numeric(spec.outcome)
    ybar = group_means(y, codes, groups)
    within = kernel_call(
        linalg.least_squares, deviations[:, :varying], y - ybar[codes], drop_collinear=False
    )
    sigma_e2 = float(within.ssr) / (n - groups)
    if sigma_e2 <= 1e-28 * float(y.square().sum()) / n:
        raise AnalysisError("perfect_fit", "HT within residual variance is zero up to rounding.")
    ones = torch.ones((groups, 1), dtype=torch.float64)
    zb = torch.cat([ones, means[:, varying:]], 1)
    hb = torch.cat([ones, means[:, : len(x1)], means[:, varying : varying + len(z1)]], 1)
    pilot, _, _ = _iv(zb, ybar - means[:, :varying] @ within.beta, hb)
    between_resid = ybar - means[:, :varying] @ within.beta - zb @ pilot
    harmonic = groups / float((1 / sizes).sum())
    raw_u2 = float(between_resid.square().mean()) - sigma_e2 / harmonic
    sigma_u2 = max(0.0, raw_u2)
    if raw_u2 < 0:
        frame.warn(
            "Negative pilot panel variance is constrained to zero; recorded unconstrained moment estimate retained."
        )
    theta = 1 - (sigma_e2 / (sigma_e2 + sizes * sigma_u2)).sqrt()
    x = torch.cat([torch.ones((n, 1), dtype=torch.float64), allx], 1)
    xbar = torch.cat([ones, means], 1)
    xs, ys = x - theta[codes, None] * xbar[codes], y - theta[codes] * ybar[codes]
    instruments = torch.cat(
        [
            torch.ones((n, 1), dtype=torch.float64),
            deviations[:, :varying],
            means[codes, : len(x1)],
            allx[:, varying : varying + len(z1)],
        ],
        1,
    )
    beta, bread, score = _iv(xs, ys, instruments)
    residual = ys - xs @ beta
    clusters, cluster_names = None, None
    if spec.covariance in {"robust", "cluster"}:
        name = spec.panel if spec.covariance == "robust" else spec.cluster
        cluster_codes, count = frame.codes(name)
        if count < 2:
            raise AnalysisError(
                "insufficient_clusters", "HT cluster inference needs at least two groups."
            )
        if any(len(torch.unique(cluster_codes[codes == i])) != 1 for i in range(groups)):
            raise AnalysisError("cluster_not_nested", "HT cluster columns must nest panels.")
        clusters, cluster_names = [(cluster_codes, count)], [name]
    small = frame.option("small")
    setting = Setting(
        n,
        None,
        "cluster" if clusters else spec.covariance,
        small,
        clusters=clusters,
        cluster_names=cluster_names,
    )
    v, info = covariance(
        frame,
        setting,
        score_x=score,
        resid=residual,
        bread=bread,
        k=k,
        rss=float(residual @ residual),
    )
    df = min(n - k, clusters[0][1] - 1) if clusters else n - k
    info.update(
        covariance=spec.covariance,
        variance_components="equal-panel between IV moment pilot; RSS within/(N-G)",
        df_inference=df if small else None,
    )
    terms = ["Intercept", *names]
    return build_result(
        frame,
        terms=terms,
        params=beta,
        covariance=v,
        title="Hausman–Taylor panel IV regression",
        use_t=small,
        df_inference=df if small else None,
        df_resid=n - k,
        fitted=x @ beta,
        solver="qr_ht_iv",
        inference=info,
        metrics={
            "sigma_u": math.sqrt(sigma_u2),
            "sigma_e": math.sqrt(sigma_e2),
            "rho": sigma_u2 / (sigma_u2 + sigma_e2),
            "n_groups": groups,
            "t_min": float(sizes.min()),
            "t_max": float(sizes.max()),
            "t_bar_harmonic": harmonic,
        },
        tests={
            "model": wald_test(
                beta,
                v,
                range(1, k),
                df_resid=df if small else None,
                label="Joint test of Hausman–Taylor slopes",
            )
        },
        extra={
            "ht_state": {
                "terms": terms,
                "roles": dict(zip(_ROLES, blocks, strict=True)),
                "sigma_u_squared": sigma_u2,
                "unconstrained_sigma_u_squared": raw_u2,
                "sigma_e_squared": sigma_e2,
                "theta": theta.tolist(),
                "within_parameters": within.beta.tolist(),
                "between_pilot_parameters": pilot.tolist(),
                "instrument_count": instruments.shape[1],
                "instrument_condition": float(torch.linalg.cond(instruments)),
                "first_stage_condition": float(torch.linalg.cond(score)),
                "prediction": "population mean X beta; no panel BLUP or idiosyncratic endogeneity",
                "component_convention": "equal-panel between IV residual second moment minus sigma_e²/T_harmonic",
            }
        },
    )


def htaylor_predict(result, data):
    """Predict population means using a saved Hausman–Taylor coefficient vector."""
    from openecon.analysis import _coerce_frame, _numeric
    from openecon.dataset import Dataset
    from openecon.resources import plan_workspace

    if (
        not isinstance(result, ResultBundle)
        or result.spec.estimator != "htaylor_moment"
        or "ht_state" not in result.extra
    ):
        raise AnalysisError(
            "invalid_result", "htaylor_predict requires saved Hausman–Taylor state."
        )
    if isinstance(data, Dataset):
        raise AnalysisError(
            "streaming_unsupported", "HT prediction needs in-memory data; Dataset is not collected."
        )
    data = _coerce_frame(data)
    terms = result.extra["ht_state"]["terms"]
    if not len(data) or data.columns.has_duplicates or any(name not in data for name in terms[1:]):
        raise AnalysisError(
            "invalid_predictors", "HT prediction needs unique complete regressor columns."
        )
    plan_workspace(
        "saved HT prediction", {"design": tensor_bytes((len(data), len(terms)), itemsize=40)}
    )
    columns = {name: _numeric(data[name], name) for name in terms[1:]}
    columns["Intercept"] = torch.ones(len(data), dtype=torch.float64)
    predicted = sum(columns[c.term] * c.estimate for c in result.coefficients)
    if not bool(torch.isfinite(predicted).all()):
        raise AnalysisError(
            "non_finite_prediction", "HT prediction exceeds finite float64 arithmetic."
        )
    return pd.Series(predicted.tolist(), index=data.index, name="predicted")
