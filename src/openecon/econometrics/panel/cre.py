"""Sample-defined Mundlak CRE and internally instrumented Hausman--Taylor."""

from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design,
    ModelFrame,
    TableSet,
    build_result,
    column_list,
    kernel_call,
    linear_covariance,
    make_spec,
    table,
    wald_test,
)
from openecon.econometrics.panel.common import panel_sample, random_effects_gls, covariance
from openecon.engines.absorb import group_means
from openecon.engines.linalg import least_squares


def _sample(spec, data):
    frame = ModelFrame(spec, data)
    frame.sort_panel()
    sample = panel_sample(frame, constant_weights=False)
    if sample.n < 3 or bool((sample.sizes < 2).any()):
        raise AnalysisError(
            "insufficient_groups", "Every panel needs >=2 observations and at least three panels."
        )
    width = len(spec.predictors) + sum(len(frame.role(r)) for r in ("x1", "x2", "z1", "z2"))
    if width > 32 or frame.n * (width + 1) ** 2 > frame.option("max_work"):
        raise AnalysisError(
            "work_budget_exceeded", "Panel CRE/HT exceeds its 32-regressor/max_work domain."
        )
    frame.workspace_plan("CRE/HT designs and QR", {"designs": frame.n * (4 * width + 8) * 24})
    return frame, sample


def _labels(frame, sample):
    return frame.sample[frame.spec.panel].iloc[sample.first_rows().tolist()].tolist()


def fit_cre(spec, data):
    frame, sample = _sample(spec, data)
    design = frame.design()
    selected = frame.option("means") or spec.predictors
    if (
        not selected
        or not set(selected) <= set(spec.predictors)
        or len(set(selected)) != len(selected)
    ):
        raise AnalysisError("invalid_spec", "means must select distinct numeric predictors.")
    means = group_means(frame.matrix(selected), sample.codes, sample.n)
    varying = frame.matrix(selected) - means[sample.codes]
    if bool((varying.square().sum(0) <= 1e-13 * frame.matrix(selected).square().sum(0)).any()):
        raise AnalysisError(
            "time_invariant_mean",
            "Mundlak means are only identified for varying predictors; select means explicitly.",
        )
    augmented = Design(
        torch.cat((design.x, means[sample.codes]), 1),
        [*design.terms, *[f"mean({name})" for name in selected]],
        {},
        True,
    )
    y = frame.numeric(spec.outcome)
    gls = random_effects_gls(sample, augmented, y, sa=False)
    k = len(augmented.terms)
    v, info, df = covariance(
        sample,
        x=gls.xs,
        resid=gls.resid,
        bread=gls.xtx_inv,
        nobs=frame.n,
        k=k,
        df_resid=frame.n - k,
        weights=None,
        means=torch.zeros(k - 1, dtype=torch.float64),
        nested=True,
    )
    indices = range(len(design.terms), k)
    test = wald_test(
        gls.beta,
        v,
        indices,
        df_resid=df if spec.covariance != "nonrobust" else None,
        label="Mundlak joint zero-mean-coefficient test",
    )
    state = {
        "units": _labels(frame, sample),
        "means_columns": selected,
        "unit_means": means.tolist(),
        "sample_positions": frame.positions,
        "prediction": "conditional mean given recorded Mundlak means; random intercept integrated to zero",
    }
    return build_result(
        frame,
        terms=augmented.terms,
        params=gls.beta,
        covariance=v,
        title="Mundlak correlated random effects",
        fitted=augmented.x @ gls.beta,
        use_t=spec.covariance != "nonrobust",
        df_inference=df,
        df_resid=frame.n - k,
        inference=info,
        tests={"mundlak": test},
        metrics={**sample.structure(), "sigma_e": gls.sigma_e2**0.5, "sigma_u": gls.sigma_u2**0.5},
        solver="QR feasible GLS with estimation-sample means",
        extra={
            "model": "cre",
            "panel_prediction": state,
            "theta": gls.theta.tolist(),
            "variance_component_convention": gls.method,
        },
    )


def fit_xthtaylor(spec, data):
    frame, sample = _sample(spec, data)
    names = {r: frame.role(r) for r in ("x1", "x2", "z1", "z2")}
    combined = [name for role in names.values() for name in role]
    if len(set(combined)) != len(combined) or spec.outcome in combined:
        raise AnalysisError(
            "overlapping_roles", "HT partitions must be disjoint and exclude the outcome."
        )
    if len(names["x1"]) < len(names["z2"]):
        raise AnalysisError(
            "underidentified",
            "HT needs at least as many exogenous varying regressors as endogenous invariant regressors.",
        )
    codes, g, n = sample.codes, sample.n, frame.n
    tv = frame.matrix([*names["x1"], *names["x2"]])
    invariant = torch.cat(
        (torch.ones((n, 1), dtype=torch.float64), frame.matrix([*names["z1"], *names["z2"]])), 1
    )
    tvbar, zbar = group_means(tv, codes, g), group_means(invariant, codes, g)
    if not torch.allclose(invariant, zbar[codes], atol=1e-12, rtol=1e-12):
        raise AnalysisError(
            "not_time_invariant", "z1/z2 must be constant within every retained panel."
        )
    y = frame.numeric(spec.outcome)
    ybar = group_means(y, codes, g)
    wx, wy = tv - tvbar[codes], y - ybar[codes]
    within = kernel_call(least_squares, wx, wy, drop_collinear=False)
    df_within = n - g - tv.shape[1]
    if df_within <= 0 or float(within.ssr) <= 0:
        raise AnalysisError(
            "insufficient_observations",
            "HT needs positive within residual degrees of freedom and variance.",
        )
    sigma_e2 = float(within.ssr) / df_within
    group_y = ybar - tvbar @ within.beta
    group_iv = torch.cat((zbar[:, : 1 + len(names["z1"])], tvbar[:, : len(names["x1"])]), 1)
    kernel_call(least_squares, group_iv, group_y, drop_collinear=False)
    qg, _ = torch.linalg.qr(group_iv, mode="reduced")
    between = kernel_call(least_squares, qg.T @ zbar, qg.T @ group_y, drop_collinear=False)
    if g <= zbar.shape[1]:
        raise AnalysisError(
            "insufficient_groups", "HT needs more panels than invariant coefficients."
        )
    group_resid = group_y - zbar @ between.beta
    raw_u2 = float(group_resid.square().sum()) / (g - zbar.shape[1]) - sigma_e2 * float(
        (1 / sample.sizes).mean()
    )
    sigma_u2 = max(0.0, raw_u2)
    if raw_u2 <= 0:
        frame.warn(
            "HT's estimated random-intercept variance is on the zero boundary; the recorded transform uses theta=0."
        )
    theta = 1 - torch.sqrt(sigma_e2 / (sample.sizes * sigma_u2 + sigma_e2))
    original = torch.cat((tv, invariant), 1)
    average = torch.cat((tvbar, zbar), 1)
    xs = original - theta[codes, None] * average[codes]
    ys = y - theta[codes] * ybar[codes]
    instruments = torch.cat((wx, (1 - theta[codes, None]) * group_iv[codes]), 1)
    kernel_call(least_squares, instruments, xs, drop_collinear=False)
    q, _ = torch.linalg.qr(instruments, mode="reduced")
    projected = q.T @ xs
    estimate = kernel_call(least_squares, projected, q.T @ ys, drop_collinear=False)
    beta, bread = estimate.beta, estimate.xtx_inv
    residual = ys - xs @ beta
    k = xs.shape[1]
    if spec.covariance == "nonrobust":
        v = sigma_e2 * bread
        info = {
            "covariance": "nonrobust",
            "correction": "within RSS/(N-G-K_varying)",
            "df_inference": None,
        }
        df = None
    else:
        v, info = linear_covariance(
            frame,
            x=q @ projected,
            resid=residual,
            bread=bread,
            n=n,
            k=k,
            df_resid=n - k,
            kind="cluster",
            clusters=[(codes, g)],
            cluster_names=[spec.panel],
        )
        df = g - 1
    terms = [*names["x1"], *names["x2"], "Intercept", *names["z1"], *names["z2"]]
    return build_result(
        frame,
        terms=terms,
        params=beta,
        covariance=v,
        title="Hausman--Taylor panel IV",
        fitted=original @ beta,
        use_t=df is not None,
        df_inference=df,
        df_resid=n - k,
        inference=info,
        solver="within/between QR variance components and quasi-demeaned QR 2SLS",
        metrics={
            **sample.structure(),
            "sigma_e": sigma_e2**0.5,
            "sigma_u": sigma_u2**0.5,
            "n_instruments": instruments.shape[1],
        },
        extra={
            "model": "hausman_taylor",
            "partitions": names,
            "theta": theta.tolist(),
            "sigma_u2_untruncated": raw_u2,
            "instrument_rank": int(torch.linalg.matrix_rank(instruments)),
            "panel_prediction": {
                "units": _labels(frame, sample),
                "prediction": "structural population mean; random intercept integrated to zero",
            },
            "variance_component_convention": "within df=N-G-K; mean-residual RSS/(G-K_invariant) minus sigma_e2*mean(1/T_i)",
        },
        warnings=[
            "The HT role declarations are identifying assumptions, not tested exogeneity. Covariance conditions on consistent estimated variance components; no finite-sample nuisance correction."
        ],
    )


def cre(
    *,
    data,
    y,
    x,
    panel,
    time=None,
    means=None,
    covariance="robust",
    cluster=None,
    missing="raise",
    alpha=0.05,
    max_work=100_000_000,
):
    """Mundlak feasible GLS with means from the retained estimation sample.

    Numeric unweighted resident panels; default panel CR1 covariance and a joint
    Mundlak test. Specify means when some predictors are time invariant.
    """
    from openecon.analysis import fit

    return fit(
        make_spec(
            "cre",
            outcome=y,
            predictors=column_list(x, "x"),
            panel=panel,
            time=time,
            covariance=covariance,
            cluster=cluster,
            missing=missing,
            alpha=alpha,
            options={"means": column_list(means, "means"), "max_work": max_work},
        ),
        data=data,
    )


def xthtaylor(
    *,
    data,
    y,
    panel,
    x1,
    z2,
    x2=None,
    z1=None,
    time=None,
    covariance="nonrobust",
    missing="raise",
    alpha=0.05,
    max_work=100_000_000,
):
    """HT panel IV: x1/x2 varying exogenous/endogenous, z1/z2 invariant partitions.

    Endogeneity is correlation with the unit effect ONLY; all varying regressors
    remain orthogonal to idiosyncratic errors. Independent units, unweighted
    numeric resident panels. nonrobust or panel CR1 ('robust') covariance.
    """
    from openecon.analysis import fit

    return fit(
        make_spec(
            "xthtaylor",
            outcome=y,
            panel=panel,
            time=time,
            covariance=covariance,
            missing=missing,
            alpha=alpha,
            columns={
                r: column_list(v, r) for r, v in {"x1": x1, "x2": x2, "z1": z1, "z2": z2}.items()
            },
            options={"max_work": max_work},
        ),
        data=data,
    )


def panel_structural_predict(result, *, data, alpha=None):
    """Saved CRE/HT population mean with full coefficient delta covariance.

    CRE requires known unit identities and reuses fitted-sample Mundlak means;
    no means, parameters or variance components are estimated from target rows.
    HT uses structural regressors with the unit effect integrated to zero.
    """
    from openecon.analysis import _coerce_frame, _numeric
    from openecon.models import ResultBundle
    from openecon.dataset import Dataset
    from openecon.engines.inference import critical_value

    if isinstance(data, Dataset):
        raise AnalysisError(
            "unsupported_dataset",
            "This resident saved prediction route does not collect a Dataset.",
        )
    if not isinstance(result, ResultBundle) or result.spec.estimator not in {"cre", "xthtaylor"}:
        raise AnalysisError(
            "invalid_result", "Supply a restored CRE or Hausman--Taylor ResultBundle."
        )
    target = _coerce_frame(data)
    if target.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Saved prediction inputs need unique columns.")
    state = result.extra["panel_prediction"]
    unit_means = dict(zip(state["units"], state.get("unit_means", [])))
    values = []
    for coef in result.coefficients:
        if coef.term == "Intercept":
            values.append(torch.ones(len(target), dtype=torch.float64))
        elif coef.term.startswith("mean(") and result.spec.estimator == "cre":
            j = state["means_columns"].index(coef.term[5:-1])
            if (
                result.spec.panel not in target
                or not target[result.spec.panel].isin(state["units"]).all()
            ):
                raise AnalysisError(
                    "unknown_panel",
                    "CRE predictions need a recorded estimation unit; unknown-unit means must not be invented.",
                )
            values.append(
                torch.tensor(
                    [unit_means[u][j] for u in target[result.spec.panel]], dtype=torch.float64
                )
            )
        else:
            if coef.term not in target:
                raise AnalysisError("missing_columns", f"Saved prediction needs {coef.term}.")
            values.append(_numeric(target[coef.term], coef.term))
    x = torch.stack(values, 1)
    if not bool(torch.isfinite(x).all()):
        raise AnalysisError(
            "missing_values", "Saved prediction requires finite complete regressors."
        )
    if len(target) * x.shape[1] ** 2 > 100_000_000:
        raise AnalysisError(
            "work_budget_exceeded", "Saved CRE/HT prediction exceeds its covariance work bound."
        )
    beta = torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)
    v = torch.tensor(result.covariance_matrix, dtype=torch.float64)
    mean = x @ beta
    variance = ((x @ v) * x).sum(1)
    if bool((variance < -1e-12).any()):
        raise AnalysisError(
            "invalid_covariance", "Saved joint covariance has negative prediction variance."
        )
    se = variance.clamp_min(0).sqrt()
    alpha = result.spec.alpha if alpha is None else alpha
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("invalid_option", "alpha must be in (0,1).")
    critical = critical_value(
        alpha, result.inference.get("df_inference") if result.inference.get("use_t") else None
    )
    return TableSet(
        {
            "predictions": table(
                {
                    "mean": mean.tolist(),
                    "std_error": se.tolist(),
                    "ci_low": (mean - critical * se).tolist(),
                    "ci_high": (mean + critical * se).tolist(),
                }
            )
        },
        title="Saved panel structural mean",
        source_result_id=result.id,
        interpretation=state["prediction"],
        alpha=alpha,
        uncertainty="coefficient uncertainty; new observation/unit-effect noise excluded",
    )
