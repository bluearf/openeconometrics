"""Structural-null Anderson--Rubin and scalar homoskedastic Moreira CLR."""

from __future__ import annotations

import math
from typing import Any, Sequence
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame,
    TableSet,
    column_list,
    kernel_call,
    make_spec,
    table,
)
from openecon.econometrics.iv.common import Setting, moment_meat, role_designs, screen
from openecon.econometrics.iv.kernels import solve
from openecon.engines.distributions import chi2_sf, f_isf, f_sf, gauss_legendre


def _geometry(
    data, y, x, endog, instruments, covariance, cluster, time, lags, kernel, intercept, missing
):
    if covariance not in ("nonrobust", "robust", "cluster", "hac"):
        raise AnalysisError(
            "unsupported_covariance", "AR accepts nonrobust, robust, cluster or hac."
        )
    if covariance == "cluster" and cluster is None:
        raise AnalysisError("invalid_spec", "cluster covariance needs a cluster column.")
    if covariance != "cluster" and cluster is not None:
        raise AnalysisError("invalid_spec", "cluster requires covariance='cluster'.")
    if covariance == "hac" and (time is None or lags is None):
        raise AnalysisError("invalid_spec", "HAC structural AR needs time and lags.")
    if covariance != "hac" and (time is not None or lags is not None):
        raise AnalysisError("invalid_spec", "time/lags are only used for HAC AR.")
    spec = make_spec(
        "ivregress",
        outcome=y,
        predictors=column_list(x, "x"),
        columns={
            "endogenous": column_list(endog, "endog"),
            "instruments": column_list(instruments, "instruments"),
        },
        covariance=covariance,
        cluster=cluster,
        time=time,
        intercept=intercept,
        missing=missing,
        options={"lags": lags, "kernel": kernel if covariance == "hac" else None},
    )
    frame = ModelFrame(spec, data)
    blocks = screen(frame, *role_designs(frame, intercept=intercept), None)
    target = frame.numeric(y)
    c, d, z = blocks.x1, blocks.x2, blocks.z2
    if c.shape[1]:
        partial = kernel_call(
            solve, c, torch.cat([target[:, None], d, z], 1), None, "exogenous regressors"
        )
        target, d, z = (
            partial.resid[:, 0],
            partial.resid[:, 1 : 1 + d.shape[1]],
            partial.resid[:, 1 + d.shape[1] :],
        )
    q, triangle = torch.linalg.qr(z, mode="reduced")
    if bool((triangle.diagonal().abs() <= 1e-10 * torch.linalg.vector_norm(z, dim=0)).any()):
        raise AnalysisError(
            "singular_design", "Excluded instruments must be full rank after exogenous projection."
        )
    df = len(target) - c.shape[1] - z.shape[1]
    if df < 2:
        raise AnalysisError(
            "insufficient_observations",
            "Structural inference needs at least two reduced-form residual degrees of freedom.",
        )
    return frame, blocks, target, d, z, q, df


def _clr_p(statistic, concentration, k):
    if statistic <= 0:
        return 1.0
    if k == 1:
        return chi2_sf(statistic, 1)
    # Beta((k-1)/2,1/2) expectation, x=sin(theta)^2 removes endpoint singularities.
    a = (k - 1) / 2
    normalization = math.exp(math.lgamma(a + 0.5) - math.lgamma(a) - math.lgamma(0.5))
    fraction = concentration / (statistic + concentration)
    previous = None
    for order in (64, 128, 256, 512):
        nodes, weights = gauss_legendre(order)
        theta = (nodes + 1) * math.pi / 4
        sine = theta.sin()
        density = 2 * normalization * sine.pow(k - 2)
        arguments = statistic / (1 - fraction * sine.square())
        values = torch.tensor([chi2_sf(float(v), k) for v in arguments], dtype=torch.float64)
        result = float(weights @ (density * values)) * math.pi / 4
        if previous is not None and abs(result - previous) < 1e-9:
            return max(0.0, min(1.0, result))
        previous = result
    raise AnalysisError(
        "numerical_failure", "CLR conditional probability quadrature did not converge."
    )


def iv_weak_test(
    *,
    data: Any,
    y: str,
    endog: Sequence[str],
    instruments: Sequence[str],
    null: float | Sequence[float],
    x: Sequence[str] | None = None,
    method: str = "ar",
    covariance: str = "nonrobust",
    cluster: str | None = None,
    time: str | None = None,
    lags: int | None = None,
    kernel: str = "bartlett",
    intercept: bool = True,
    missing: str = "raise",
) -> TableSet:
    """Test a complete structural endogenous coefficient vector without strong-IV assumptions.

    AR: exact F under iid Gaussian homoskedastic errors, or asymptotic HC0,
    one-way cluster (G/(G-1)) / HAC Wald chi2. Exogenous nuisance coefficients
    are partialled out; untested endogenous nuisance parameters are unsupported.
    CLR: Moreira (2003), one endogenous coefficient, iid homoskedastic covariance.
    It uses full reduced-form covariance and conditional numerical integration.
    Weighted/survey/subvector CLR domains are not silently approximated.
    """
    if method not in ("ar", "clr"):
        raise AnalysisError("invalid_option", "method must be 'ar' or 'clr'.")
    frame, blocks, target, d, z, q, df = _geometry(
        data, y, x, endog, instruments, covariance, cluster, time, lags, kernel, intercept, missing
    )
    try:
        beta = torch.tensor(
            [null] if isinstance(null, (int, float)) and not isinstance(null, bool) else list(null),
            dtype=torch.float64,
        )
    except (TypeError, ValueError) as exc:
        raise AnalysisError(
            "invalid_null", "null must contain one finite coefficient per endogenous regressor."
        ) from exc
    if beta.shape != (d.shape[1],) or not bool(torch.isfinite(beta).all()):
        raise AnalysisError(
            "invalid_null",
            "null must contain one finite coefficient per retained endogenous regressor.",
        )
    residual = target - d @ beta
    projected = q.T @ residual
    remainder = residual - q @ projected
    k = z.shape[1]
    extra = {
        "method": method,
        "null": beta.tolist(),
        "endogenous": blocks.endog.terms,
        "excluded_instruments": blocks.instr.terms,
        "nuisance": "included exogenous only",
        "covariance": covariance,
        "nobs": len(target),
        "df_reduced_form": df,
        "missing": missing,
        "sample_positions": list(frame.positions),
        "dtype": "float64",
        "source": "Anderson--Rubin (1949); Moreira (2003)",
    }
    if method == "clr":
        if d.shape[1] != 1 or covariance != "nonrobust":
            raise AnalysisError(
                "unsupported_inference",
                "CLR is validated for one endogenous regressor and iid homoskedastic covariance only; use structural AR for robust/cluster/HAC.",
            )
        responses = torch.cat([target[:, None], d], 1)
        innovation = responses - q @ (q.T @ responses)
        omega = innovation.T @ innovation / df
        try:
            inverse = torch.linalg.inv(omega)
            b = torch.tensor([1.0, -float(beta[0])], dtype=torch.float64)
            a = torch.tensor([float(beta[0]), 1.0], dtype=torch.float64)
            s = q.T @ responses @ b / torch.sqrt(b @ omega @ b)
            t = q.T @ responses @ inverse @ a / torch.sqrt(a @ inverse @ a)
            ss, tt, st = float(s @ s), float(t @ t), float(s @ t)
            statistic = max(0.0, 0.5 * (ss - tt + math.sqrt(max(0.0, (ss - tt) ** 2 + 4 * st**2))))
            probability = _clr_p(statistic, tt, k)
        except torch.linalg.LinAlgError as exc:
            raise AnalysisError(
                "invalid_covariance", "The full reduced-form covariance is singular."
            ) from exc
        extra.update(
            reduced_form_covariance=omega.tolist(),
            conditioning_statistic=tt,
            quadrature_tolerance=1e-9,
            distribution="conditional Moreira CLR",
        )
    elif covariance == "nonrobust":
        noise = float(remainder @ remainder)
        if noise <= 0:
            raise AnalysisError("perfect_fit", "Reduced-form null residual variance is zero.")
        statistic = float(projected @ projected) / noise * df / k
        probability = f_sf(statistic, k, df)
        extra.update(distribution="F", df=k, df_resid=df)
    else:
        design = torch.cat([blocks.x1, blocks.z2], 1)
        fit = kernel_call(
            solve, design, frame.numeric(y) - blocks.x2 @ beta, None, "null reduced form"
        )
        clusters = frame.cluster_dimensions() if covariance == "cluster" else None
        setting = Setting(
            len(target),
            None,
            covariance,
            False,
            clusters=clusters,
            cluster_names=[cluster] if cluster else None,
        )
        scores = design * fit.resid[:, None]
        meat = moment_meat(frame, setting, scores)
        if covariance == "cluster":
            groups = int(torch.unique(clusters[0][0]).numel())
            if groups < 2:
                raise AnalysisError("insufficient_clusters", "AR needs at least two clusters.")
            meat *= groups / (groups - 1)
            extra["n_clusters"] = groups
        covariance_matrix = fit.xtx_inv @ meat @ fit.xtx_inv
        estimated = fit.beta[-k:]
        block = covariance_matrix[-k:, -k:]
        try:
            torch.linalg.cholesky(block)
            statistic = float(estimated @ torch.linalg.solve(block, estimated))
        except torch.linalg.LinAlgError as exc:
            raise AnalysisError(
                "invalid_covariance", "The excluded-instrument covariance is singular."
            ) from exc
        probability = chi2_sf(statistic, k)
        extra.update(
            distribution="chi2",
            df=k,
            null_reduced_form_covariance=covariance_matrix.tolist(),
            lags=lags if covariance == "hac" else None,
            kernel=kernel if covariance == "hac" else None,
        )
    if not math.isfinite(statistic) or not math.isfinite(probability):
        raise AnalysisError(
            "numerical_failure", "Structural inference produced a nonfinite statistic."
        )
    return TableSet(
        {"test": table([{"statistic": statistic, "p_value": probability}])},
        statistic=statistic,
        p_value=probability,
        **extra,
    )


def _quadratic_set(a, b, c):
    """Exact real solution of a*t^2+b*t+c <= 0; None is an infinite endpoint."""
    scale = max(abs(a), abs(b), abs(c), 1.0)
    a, b, c = a / scale, b / scale, c / scale
    if abs(a) < 1e-14:
        if abs(b) < 1e-14:
            return [[None, None]] if c <= 0 else []
        root = -c / b
        return [[None, root]] if b > 0 else [[root, None]]
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return [[None, None]] if a < 0 else []
    root = math.sqrt(max(0.0, discriminant))
    stable = -0.5 * (b + math.copysign(root, b))
    roots = sorted([stable / a, c / stable]) if stable else [-b / (2 * a)] * 2
    return [[roots[0], roots[1]]] if a > 0 else [[None, roots[0]], [roots[1], None]]


def iv_ar_confidence_set(
    *,
    data: Any,
    y: str,
    endog: str,
    instruments: Sequence[str],
    x: Sequence[str] | None = None,
    alpha: float = 0.05,
    intercept: bool = True,
    missing: str = "raise",
) -> TableSet:
    """Exact inversion of scalar iid Gaussian AR; retains empty/disjoint/unbounded sets.

    Endpoints of None represent +/- infinity, so persistence never converts
    infinity into missing numeric data. Robust/cluster/HAC and CLR inversion
    are outside this analytical quadratic contract.
    """
    if isinstance(alpha, bool) or not 0 < alpha < 1:
        raise AnalysisError("invalid_option", "alpha must lie strictly between 0 and 1.")
    _, _, target, d, _, q, df = _geometry(
        data,
        y,
        x,
        [endog],
        instruments,
        "nonrobust",
        None,
        None,
        None,
        "bartlett",
        intercept,
        missing,
    )
    projected = q.T @ torch.stack([target, d[:, 0]], 1)
    remainder = torch.stack([target, d[:, 0]], 1) - q @ projected
    critical = f_isf(alpha, q.shape[1], df)
    geometry = projected.T @ projected - critical * q.shape[1] / df * (remainder.T @ remainder)
    intervals = _quadratic_set(
        float(geometry[1, 1]), -2 * float(geometry[0, 1]), float(geometry[0, 0])
    )
    return TableSet(
        {
            "confidence_set": table(
                [
                    {
                        "lower": a,
                        "upper": b,
                        "lower_unbounded": a is None,
                        "upper_unbounded": b is None,
                    }
                    for a, b in intervals
                ]
            )
        },
        intervals=intervals,
        empty=not intervals,
        disjoint=len(intervals) > 1,
        alpha=alpha,
        method="scalar structural AR inversion",
        null_endogenous=endog,
        boundary_polynomial=[
            float(geometry[1, 1]),
            -2 * float(geometry[0, 1]),
            float(geometry[0, 0]),
        ],
        df=q.shape[1],
        df_resid=df,
        critical_value=critical,
    )


def stock_yogo(result) -> TableSet:
    """Stock--Yogo 10% maximum size of a nominal 5% 2SLS Wald test.

    Only iid homoskedastic, unweighted, one-endogenous ivregress 2SLS and
    1..30 excluded instruments. No KP, robust, HAC, panel, weighted or LIML
    threshold is assigned. Source: Stock--Yogo (2005), Table 5.2,
    independently reproduced in US WWC Standards Handbook 4.0, Table II.7.
    """
    from openecon.models import ResultBundle

    if not isinstance(result, ResultBundle):
        raise AnalysisError("invalid_result", "stock_yogo needs a saved ResultBundle.")
    spec = result.spec
    if (
        spec.estimator != "ivregress"
        or spec.weights is not None
        or spec.covariance != "nonrobust"
        or spec.options.get("method", "2sls") != "2sls"
        or result.metrics.get("n_endogenous") != 1
    ):
        raise AnalysisError(
            "unsupported_inference",
            "Stock--Yogo size reference is available only for unweighted iid one-endogenous 2SLS; it is not a KP threshold.",
        )
    count = len(result.extra.get("instruments", []))
    values = (
        16.38,
        19.93,
        22.30,
        24.58,
        26.87,
        29.18,
        31.50,
        33.84,
        36.19,
        38.54,
        40.90,
        43.27,
        45.64,
        48.01,
        50.39,
        52.77,
        55.15,
        57.53,
        59.92,
        62.30,
        64.69,
        67.07,
        69.46,
        71.85,
        74.24,
        76.62,
        79.01,
        81.40,
        83.79,
        86.17,
    )
    if not 1 <= count <= len(values):
        raise AnalysisError(
            "critical_values_unavailable",
            "Stock--Yogo reference retains 1..30 excluded instruments only.",
        )
    diagnostic = result.tests.get("cragg_donald", {})
    statistic = diagnostic.get("statistic")
    if not isinstance(statistic, (int, float)) or not math.isfinite(statistic):
        raise AnalysisError("invalid_result", "Saved Cragg--Donald statistic is unavailable.")
    critical = values[count - 1]
    return TableSet(
        {
            "reference": table(
                [
                    {
                        "cragg_donald": statistic,
                        "critical_value": critical,
                        "reject_weak_identification": statistic > critical,
                    }
                ]
            )
        },
        critical_value=critical,
        n_instruments=count,
        maximum_size=0.10,
        nominal_size=0.05,
        source="Stock and Yogo (2005) Table 5.2; WWC 4.0 Table II.7",
        source_url="https://ies.ed.gov/ncee/wwc/Docs/referenceresources/wwc_standards_handbook_v4.pdf",
    )
