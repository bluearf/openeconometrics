"""Continuously updated linear IV GMM; no development references at runtime."""

from __future__ import annotations

import math
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame,
    build_result,
    column_list,
    kernel_call,
    make_spec,
)
from openecon.engines.distributions import chi2_sf
from openecon.engines.linalg import least_squares
from openecon.engines.optimize import maximize_bfgs


def fit_ivcue(spec, data):
    frame = ModelFrame(spec, data)
    n = frame.n
    endog, instruments = frame.role("endogenous"), frame.role("instruments")
    if (
        set(endog) & set(spec.predictors)
        or set(instruments) & (set(spec.predictors) | set(endog))
        or spec.outcome in [*spec.predictors, *endog, *instruments]
    ):
        raise AnalysisError(
            "overlapping_roles",
            "CUE needs distinct outcome/exogenous/endogenous/excluded-instrument roles.",
        )
    design = frame.design()
    x = torch.cat((design.x, frame.matrix(frame.role("endogenous"))), dim=1)
    z = torch.cat((design.x, frame.matrix(frame.role("instruments"))), dim=1)
    y = frame.numeric(spec.outcome)
    k, ell = x.shape[1], z.shape[1]
    if ell < k or n <= ell or k > 24 or ell > 48:
        raise AnalysisError("invalid_domain", "CUE needs N>L>=K, K<=24 and L<=48.")
    if n * (k + ell) * ell**2 > frame.option("max_work"):
        raise AnalysisError(
            "work_budget_exceeded", "CUE exceeds max_work; no automatic collection."
        )
    frame.workspace_plan(
        "CUE differentiated moment covariance",
        {
            "designs": n * (k + ell) * 24,
            "derivatives": (k + 2) ** 2 * ell**2 * 24,
        },
    )
    sx, sz = x.square().mean(0).sqrt(), z.square().mean(0).sqrt()
    sy = y.std()
    if bool((sx <= 0).any()) or bool((sz <= 0).any()) or not float(sy) > 0:
        raise AnalysisError(
            "singular_design", "CUE requires nonzero columns and a varying outcome."
        )
    xx, zz, yy = x / sx, z / sz, y / sy
    kernel_call(least_squares, zz, xx, drop_collinear=False)
    qz, _ = torch.linalg.qr(zz, mode="reduced")
    start = kernel_call(least_squares, qz.T @ xx, qz.T @ yy, drop_collinear=False).beta
    codes = None
    groups = None
    if spec.covariance == "cluster":
        codes, groups = frame.cluster_dimensions()[0]
        if groups <= ell:
            raise AnalysisError(
                "insufficient_clusters", "CUE needs more independent clusters than moments."
            )

    def moment_state(theta):
        residual = yy - xx @ theta
        rows = zz * residual[:, None]
        g = rows.mean(0)
        if frame.option("center"):
            rows = rows - g
        if codes is not None:
            sums = rows.new_zeros((groups, ell)).index_add(0, codes, rows)
            s = sums.T @ sums / n
        else:
            s = rows.T @ rows / n
        return g, s

    calls = 0

    def value(theta):
        nonlocal calls
        calls += 1
        if calls * n * ell**2 * (k + 1) ** 2 > frame.option("max_work"):
            raise AnalysisError(
                "work_budget_exceeded",
                "CUE exhausted its declared objective/derivative work budget.",
            )
        g, s = moment_state(theta)
        chol = torch.linalg.cholesky(s)
        return -0.5 * n * (g @ torch.cholesky_solve(g[:, None], chol).flatten())

    def vg(theta):
        with torch.enable_grad():
            point = theta.detach().clone().requires_grad_(True)
            objective = value(point)
            gradient = torch.autograd.grad(objective, point)[0]
        return objective.detach(), gradient.detach()

    def hessian(theta):
        with torch.enable_grad():
            return torch.autograd.functional.hessian(value, theta)

    try:
        optimum = kernel_call(
            maximize_bfgs,
            vg,
            start,
            hessian_fn=hessian,
            max_iter=frame.option("max_iterations"),
            gradient_tol=frame.option("tolerance"),
        )
        theta = optimum.theta
        g, s = moment_state(theta)
        d = -(zz.T @ xx) / n
        information = d.T @ torch.linalg.solve(s, d)
        chol = torch.linalg.cholesky(information)
        covariance = torch.cholesky_inverse(chol) / n
    except RuntimeError as exc:
        raise AnalysisError(
            "singular_moments",
            "CUE moment covariance or information is singular; no ridge is applied.",
        ) from exc
    change = sy / sx
    beta = theta * change
    covariance = covariance * change[:, None] * change[None, :]
    j = float(-2 * optimum.value)
    degrees = ell - k
    terms = [*design.terms, *frame.role("endogenous")]
    residual = y - x @ beta
    return build_result(
        frame,
        terms=terms,
        params=beta,
        covariance=covariance,
        title="Continuously updated IV GMM",
        fitted=x @ beta,
        use_t=False,
        df_resid=n - k,
        solver="native continuously updated GMM",
        optimizer={
            "method": "BFGS with exact autodiff Hessian",
            "converged": optimum.converged,
            "iterations": optimum.iterations,
        },
        metrics={
            "j": j,
            "n_instruments": ell,
            "n_endogenous": len(frame.role("endogenous")),
            "rmse": float(residual.square().mean().sqrt()),
            "df_resid": n - k,
        },
        tests={
            "hansen_j": {
                "statistic": j,
                "df": degrees,
                "p_value": chi2_sf(j, degrees) if degrees else None,
                "distribution": "chi2",
                "note": "strong-identification asymptotics",
            }
        },
        inference={
            "covariance": spec.covariance,
            "correction": "HC0 or uncorrected cluster moment sums",
            "df_inference": None,
            "assumptions": "valid moments, independent rows/clusters and strong identification",
        },
        extra={
            "method": "cue",
            "moment_mean": g.tolist(),
            "moment_covariance_scaled": s.tolist(),
            "moment_jacobian_scaled": d.tolist(),
            "jacobian_rank": int(torch.linalg.matrix_rank(d)),
            "objective_hessian_scaled": optimum.hessian.tolist(),
            "center": frame.option("center"),
            "parameter_scales": change.tolist(),
            "instrument_scales": sz.tolist(),
            "cluster_count": groups,
            "objective_evaluations": calls,
            "declared_objective_work": calls * n * ell**2 * (k + 1) ** 2,
            "exogenous": design.terms,
            "endogenous": frame.role("endogenous"),
            "instruments": frame.role("instruments"),
        },
        warnings=[
            "CUE inference assumes strong identification; use structural AR/CLR tests for weak instruments."
        ],
    )


def ivcue(
    *,
    data,
    y,
    endog,
    instruments,
    x=None,
    cluster=None,
    center=False,
    max_iterations=300,
    tolerance=1e-8,
    max_work=100_000_000,
    missing="raise",
    alpha=0.05,
):
    """Fit unweighted numeric linear IV CUE with HC0 or one-way cluster moments.

    Minimizes N*g(b)'S(b)^-1*g(b), recomputing S at every candidate. Returns
    full efficient asymptotic covariance, moment/Jacobian and convergence state.
    CPU resident inputs only; no weights, categorical terms or HAC fitting route.
    """
    from openecon.analysis import fit

    return fit(
        make_spec(
            "ivcue",
            outcome=y,
            predictors=column_list(x, "x"),
            columns={
                "endogenous": column_list(endog, "endog"),
                "instruments": column_list(instruments, "instruments"),
            },
            covariance="cluster" if cluster is not None else "HC0",
            cluster=cluster,
            missing=missing,
            alpha=alpha,
            options={
                "center": center,
                "max_iterations": max_iterations,
                "tolerance": tolerance,
                "max_work": max_work,
            },
        ),
        data=data,
    )


def effective_f(
    *,
    data,
    y,
    endog,
    instruments,
    x=None,
    covariance="robust",
    cluster=None,
    time=None,
    lags=None,
    kernel="bartlett",
    intercept=True,
    missing="raise",
):
    """Montiel Olea--Pflueger effective first-stage F for ONE endogenous variable.

    Returns signal divided by trace of the compatible first-stage moment meat
    times inverse residualized-instrument crossproduct (HC0, cluster sums or
    explicit HAC). No Stock--Yogo or arbitrary universal cutoff is attached.
    This statistic is separate from the robust first-stage/KP Wald statistic.
    """
    from openecon.econometrics.iv.weak_inference import _geometry
    from openecon.econometrics.iv.common import Setting, moment_meat
    from openecon.econometrics.core import TableSet, table

    frame, blocks, _, d, z, q, df = _geometry(
        data, y, x, endog, instruments, covariance, cluster, time, lags, kernel, intercept, missing
    )
    if d.shape[1] != 1:
        raise AnalysisError("unsupported_target", "effective_f requires one endogenous regressor.")
    fitted = q @ (q.T @ d[:, 0])
    residual = d[:, 0] - fitted
    setting = Setting(
        frame.n,
        None,
        covariance,
        False,
        clusters=frame.cluster_dimensions() if cluster is not None else None,
        cluster_names=[cluster] if isinstance(cluster, str) else cluster,
    )
    if covariance == "nonrobust":
        meat = residual.square().mean() * (z.T @ z)
    else:
        meat = moment_meat(frame, setting, z * residual[:, None])
    cross = z.T @ z
    denominator = float(torch.linalg.solve(cross, meat).trace())
    signal = float(fitted.square().sum())
    if denominator <= 0 or not math.isfinite(denominator):
        raise AnalysisError(
            "singular_first_stage", "The effective-F normalization must be positive."
        )
    return TableSet(
        {
            "diagnostic": table(
                [
                    {
                        "endogenous": blocks.endog.terms[0],
                        "effective_f": signal / denominator,
                        "signal": signal,
                        "normalization": denominator,
                        "instruments": z.shape[1],
                    }
                ]
            )
        },
        title="Montiel Olea--Pflueger effective F",
        covariance=covariance,
        correction="none; first-stage residual covariance uses N divisor",
        sample_positions=frame.positions,
        nobs=frame.n,
        first_stage_residual_df=df,
        critical_values=None,
        rejection=None,
        source="https://arxiv.org/html/2309.01637v3#S2.E6",
        calibration="Requires MOP estimator/bias-specific cutoffs; Stock--Yogo cutoffs do not apply to robust effective F.",
        assumptions="One endogenous regressor, valid instruments and compatible independent/HAC/cluster moment covariance.",
    )
