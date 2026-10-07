"""Panel ARDL(p,q) in the ECM parameterization, entirely float64 Torch.

PMG has common long-run slopes, heterogeneous adjustment, short-run terms,
deterministics and variances. MG averages independent unit ECMs. DFE has common
ECM slopes/variance and unit deterministics. Conditional initial observations
are trimmed within each unit. Unbalanced consecutive spans are supported.
"""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.advanced_common import deterministic, ls, regular_frame
from openecon.econometrics.core import build_result, column_list, kernel_call, make_spec
from openecon.engines.distributions import chi2_sf
from openecon.engines.linalg import cholesky_inverse
from openecon.engines.optimize import maximize_bfgs
from openecon.models import ResultBundle


def _blocks(frame, units):
    y, x = frame.numeric(frame.spec.outcome), frame.matrix(frame.spec.predictors)
    p, q, trend = (frame.option(key) for key in ("p", "q", "trend"))
    result = []
    start = max(p, q)
    for unit in units:
        yu, xu = y[unit], x[unit]
        t = torch.arange(start, len(unit))
        if len(t) < 2:
            raise AnalysisError(
                "insufficient_observations",
                "Every unit needs observations beyond all lag boundaries.",
            )
        dy, dx = yu[1:] - yu[:-1], xu[1:] - xu[:-1]
        d, dnames = deterministic(len(unit), trend)
        sr, names = [], []
        for lag in range(1, p):
            sr.append(dy[t - lag - 1, None])
            names.append(f"D({frame.spec.outcome},lag={lag})")
        for lag in range(q):
            sr.append(dx[t - lag - 1])
            names.extend(f"D({name},lag={lag})" for name in frame.spec.predictors)
        sr.append(d[t])
        names.extend(dnames)
        w = torch.cat(sr, dim=1)
        block = {
            "rows": unit[t],
            "dy": dy[t - 1],
            "yl": yu[t - 1],
            "xl": xu[t - 1],
            "w": w,
            "names": names,
        }
        z = torch.cat((block["yl"][:, None], block["xl"], w), dim=1)
        block["linear"] = ls(z, block["dy"])
        block["z"] = z
        if abs(float(block["linear"].beta[0])) < 1e-8:
            raise AnalysisError(
                "unidentified_long_run",
                "An adjustment coefficient near zero does not identify a long-run ratio.",
            )
        result.append(block)
    return result


def _ecm_map(beta, k):
    return torch.cat((-beta[1 : k + 1] / beta[0], beta[:1], beta[k + 1 :]))


def _mg(blocks, k):
    estimates, covariances = [], []
    for block in blocks:
        fit = block["linear"]
        v = fit.xtx_inv * (fit.ssr / len(block["dy"]))
        jac = torch.autograd.functional.jacobian(lambda b: _ecm_map(b, k), fit.beta)
        estimates.append(_ecm_map(fit.beta, k))
        covariances.append(jac @ v @ jac.T)
    estimates = torch.stack(estimates)
    mean = estimates.mean(0)
    if len(blocks) < 2:
        raise AnalysisError("insufficient_groups", "Panel ARDL needs at least two units.")
    deviations = estimates - mean
    covariance = deviations.T @ deviations / (len(blocks) * (len(blocks) - 1))
    return mean, covariance, estimates, covariances


def _profile(theta, blocks):
    value = theta.new_zeros(())
    for b in blocks:
        ect = b["yl"] - b["xl"] @ theta
        design = torch.cat((ect[:, None], b["w"]), dim=1)
        beta = torch.linalg.lstsq(design, b["dy"]).solution
        variance = (b["dy"] - design @ beta).square().mean()
        value -= 0.5 * len(ect) * (math.log(2 * math.pi) + 1 + variance.log())
    return value


def _full_loglike(parameters, blocks, k):
    theta = parameters[:k]
    offset = k
    total = parameters.new_zeros(())
    for b in blocks:
        width = b["w"].shape[1] + 1
        sr = parameters[offset : offset + width]
        logvar = parameters[offset + width]
        mean = sr[0] * (b["yl"] - b["xl"] @ theta) + b["w"] @ sr[1:]
        resid = b["dy"] - mean
        total -= 0.5 * (
            len(resid) * (math.log(2 * math.pi) + logvar)
            + resid.square().sum() * torch.exp(-logvar)
        )
        offset += width + 1
    return total


def _pmg(blocks, k, max_iterations, initial):
    @torch.enable_grad()
    def objective(theta):
        theta = theta.detach().requires_grad_(True)
        value = _profile(theta, blocks)
        gradient = torch.autograd.grad(value, theta)[0]
        return value.detach(), gradient.detach()

    def hessian(theta):
        return torch.autograd.functional.hessian(lambda t: _profile(t, blocks), theta)

    solution = kernel_call(
        maximize_bfgs,
        objective,
        initial,
        hessian_fn=hessian,
        max_iter=max_iterations,
        gradient_tol=1e-10,
        scaled_gradient_tol=1e-10,
    )
    parts, indices, offset = [solution.theta], list(range(k)), k
    for b in blocks:
        ect = b["yl"] - b["xl"] @ solution.theta
        fit = ls(torch.cat((ect[:, None], b["w"]), dim=1), b["dy"])
        if abs(float(fit.beta[0])) < 1e-8:
            raise AnalysisError(
                "unidentified_long_run",
                "PMG adjustment is near zero; no identified long-run inference.",
            )
        parts.extend((fit.beta, (fit.ssr / len(ect)).log().reshape(1)))
        indices.extend(range(offset, offset + len(fit.beta)))
        offset += len(fit.beta) + 1
    full = torch.cat(parts)
    info = -torch.autograd.functional.hessian(lambda a: _full_loglike(a, blocks, k), full)
    covariance = kernel_call(cholesky_inverse, info, what="PMG full observed information")
    return full[indices], covariance[indices][:, indices], full, covariance, solution


def _dfe(frame, blocks, k):
    # Unit deterministics, homogeneous adjustment/level/short-run coefficients.
    nd = {"n": 0, "c": 1, "ct": 2}[frame.option("trend")]
    common = blocks[0]["z"].shape[1] - nd
    width = common + nd * len(blocks)
    n = sum(len(b["dy"]) for b in blocks)
    frame.workspace_plan(
        "DFE block design", {"design_and_factors": 40 * n * width + 64 * width * width}
    )
    design = torch.zeros((n, width), dtype=torch.float64)
    offset = 0
    for i, b in enumerate(blocks):
        stop = offset + len(b["dy"])
        design[offset:stop, :common] = b["z"][:, :common]
        if nd:
            design[offset:stop, common + i * nd : common + (i + 1) * nd] = b["z"][:, -nd:]
        offset = stop
    dy = torch.cat([b["dy"] for b in blocks])
    fit = ls(design, dy)
    if abs(float(fit.beta[0])) < 1e-8:
        raise AnalysisError("unidentified_long_run", "DFE adjustment is near zero.")
    beta = _ecm_map(fit.beta, k)
    jac = torch.autograd.functional.jacobian(lambda b: _ecm_map(b, k), fit.beta)
    sigma = fit.ssr / n
    v = jac @ (fit.xtx_inv * sigma) @ jac.T
    return beta, v, fit, float(sigma)


def fit_panel_ardl(spec, data):
    frame, units = regular_frame(spec, data)
    if len(units) < 2:
        raise AnalysisError("insufficient_groups", "Panel ARDL needs at least two units.")
    k, labels = len(spec.predictors), frame.levels(spec.panel)
    blocks = _blocks(frame, units)
    width = k + sum(b["w"].shape[1] + 2 for b in blocks)
    frame.workspace_plan(
        "panel ARDL full information", {"hessian_covariance_and_autograd": 128 * width * width}
    )
    mg_beta, mg_cov, unit_estimates, unit_covariances = _mg(blocks, k)
    terms = [f"LR:{x}" for x in spec.predictors]
    state = {
        "p": frame.option("p"),
        "q": frame.option("q"),
        "trend": frame.option("trend"),
        "calendar": "consecutive integer periods per unit; unbalanced spans allowed",
        "equation": "D(y)=phi_i*(L(y)-theta_i*L(x))+short_run+deterministics",
        "units": [],
        "conditional_initial_observations": max(frame.option("p"), frame.option("q")),
    }
    for label, b, estimates, v in zip(
        labels, blocks, unit_estimates, unit_covariances, strict=True
    ):
        state["units"].append(
            {
                "unit": label,
                "unrestricted_ecm": b["linear"].beta.tolist(),
                "long_run_adjustment_short_run": estimates.tolist(),
                "covariance": v.tolist(),
                "sample_positions": [frame.positions[int(r)] for r in b["rows"]],
            }
        )
    optimizer, diagnostics, tests = None, {"converged": True}, {}
    if spec.estimator == "mg":
        beta, v = mg_beta, mg_cov
        terms += ["adjustment", *blocks[0]["names"]]
        state["constraints"] = "none; equal-unit mean of heterogeneous ECM parameters"
        state["likelihood"] = "independent Gaussian unit OLS; unit variances use SSR/T_i"
        ll = sum(
            -0.5
            * len(b["dy"])
            * (math.log(2 * math.pi) + 1 + math.log(float(b["linear"].ssr) / len(b["dy"])))
            for b in blocks
        )
    elif spec.estimator == "pmg":
        beta, v, full, full_cov, solution = _pmg(
            blocks, k, frame.option("max_iterations"), mg_beta[:k]
        )
        for label, b in zip(labels, blocks, strict=True):
            terms.extend(
                [f"unit[{label}]:adjustment", *[f"unit[{label}]:{name}" for name in b["names"]]]
            )
        state.update(
            constraints="common long-run slopes; all adjustment/SR/deterministic terms and variances heterogeneous",
            likelihood="conditional independent Gaussian; heterogeneous unit variance; profiled short run",
            full_parameter_vector=full.tolist(),
            full_covariance=full_cov.tolist(),
            full_parameter_layout="long_run, then per unit: adjustment, short_run, deterministics, log_variance",
        )
        optimizer = {
            "method": solution.method,
            "iterations": solution.iterations,
            "converged": solution.converged,
        }
        diagnostics = solution.diagnostics
        ll = solution.value
        tests["long_run_homogeneity"] = _hausman(mg_beta[:k], mg_cov[:k, :k], beta[:k], v[:k, :k])
    else:
        beta, v, fit, sigma = _dfe(frame, blocks, k)
        nd = {"n": 0, "c": 1, "ct": 2}[frame.option("trend")]
        terms += ["adjustment", *(blocks[0]["names"][:-nd] if nd else blocks[0]["names"])]
        for label in labels:
            terms.extend(f"unit[{label}]:{name}" for name in ["Intercept", "trend"][:nd])
        state.update(
            constraints="homogeneous adjustment/long-run/SR/variance; unit-specific deterministics",
            likelihood="conditional independent Gaussian; common variance SSR/N",
            variance=sigma,
            linear_parameters=fit.beta.tolist(),
            linear_covariance=(fit.xtx_inv * sigma).tolist(),
        )
        ll = -0.5 * len(fit.resid) * (math.log(2 * math.pi) + 1 + math.log(sigma))
        tests["long_run_homogeneity"] = _hausman(mg_beta[:k], mg_cov[:k, :k], beta[:k], v[:k, :k])
    rows = torch.cat([b["rows"] for b in blocks])
    keep = torch.zeros(frame.n, dtype=torch.bool)
    keep[rows] = True
    frame.restrict(keep, "Excluded initial lag observations within each unit.")
    return build_result(
        frame,
        terms=terms,
        params=beta,
        covariance=v,
        use_t=spec.estimator == "mg",
        df_inference=len(units) - 1 if spec.estimator == "mg" else None,
        df_resid=frame.n - len(beta),
        metrics={"log_likelihood": ll, "n_groups": len(units)},
        solver="float64_ecm_qr_profile_gaussian",
        solver_diagnostics=diagnostics,
        optimizer=optimizer,
        tests=tests,
        extra=state,
        inference={
            "correction": "between-unit dispersion / G"
            if spec.estimator == "mg"
            else "full Gaussian observed information, including nuisance variances"
        },
    )


def _hausman(unrestricted, vu, restricted, vr):
    difference, covariance = unrestricted - restricted, vu - vr
    eigen = torch.linalg.eigvalsh((covariance + covariance.T) / 2)
    if float(eigen.min()) <= 1e-10 * max(float(eigen.abs().max()), 1e-30):
        return {
            "status": "undefined",
            "reason": "MG minus restricted covariance is not positive definite; no PSD clipping or negative statistic.",
            "distribution": "chi2",
            "df": len(difference),
            "statistic": None,
            "p_value": None,
        }
    statistic = float(
        difference
        @ kernel_call(cholesky_inverse, covariance, what="Hausman covariance difference")
        @ difference
    )
    return {
        "status": "ok",
        "statistic": statistic,
        "p_value": chi2_sf(statistic, len(difference)),
        "df": len(difference),
        "distribution": "chi2",
    }


def panel_ardl_hausman(unrestricted, restricted):
    """MG vs PMG/DFE long-run homogeneity test; same input/sample/lag contract required."""
    if (
        not isinstance(unrestricted, ResultBundle)
        or not isinstance(restricted, ResultBundle)
        or unrestricted.spec.estimator != "mg"
        or restricted.spec.estimator not in ("pmg", "dfe")
    ):
        raise AnalysisError("invalid_result", "Compare an MG result with PMG or DFE.")
    if unrestricted.provenance["sample_hash"] != restricted.provenance["sample_hash"] or any(
        unrestricted.extra[a] != restricted.extra[a] for a in ("p", "q", "trend")
    ):
        raise AnalysisError(
            "sample_mismatch", "Homogeneity comparisons need the same data, rows and ECM design."
        )
    k = len(unrestricted.spec.predictors)
    return _hausman(
        torch.tensor([c.estimate for c in unrestricted.coefficients[:k]], dtype=torch.float64),
        torch.tensor(unrestricted.covariance_matrix, dtype=torch.float64)[:k, :k],
        torch.tensor([c.estimate for c in restricted.coefficients[:k]], dtype=torch.float64),
        torch.tensor(restricted.covariance_matrix, dtype=torch.float64)[:k, :k],
    )


def _fit(
    name,
    *,
    data,
    y,
    x,
    panel,
    time,
    p=1,
    q=1,
    trend="c",
    max_iterations=500,
    missing="raise",
    alpha=0.05,
):
    from openecon.analysis import fit

    return fit(
        make_spec(
            name,
            outcome=y,
            predictors=column_list(x, "x"),
            panel=panel,
            time=time,
            intercept=False,
            missing=missing,
            alpha=alpha,
            options={"p": p, "q": q, "trend": trend, "max_iterations": max_iterations},
        ),
        data=data,
    )


def pmg(
    *, data, y, x, panel, time, p=1, q=1, trend="c", max_iterations=500, missing="raise", alpha=0.05
):
    """Common long-run and heterogeneous short-run Gaussian panel ARDL."""
    return _fit(
        "pmg",
        data=data,
        y=y,
        x=x,
        panel=panel,
        time=time,
        p=p,
        q=q,
        trend=trend,
        max_iterations=max_iterations,
        missing=missing,
        alpha=alpha,
    )


def mg(
    *, data, y, x, panel, time, p=1, q=1, trend="c", max_iterations=500, missing="raise", alpha=0.05
):
    """Equal-unit mean of unrestricted ECMs with between-unit covariance."""
    return _fit(
        "mg",
        data=data,
        y=y,
        x=x,
        panel=panel,
        time=time,
        p=p,
        q=q,
        trend=trend,
        max_iterations=max_iterations,
        missing=missing,
        alpha=alpha,
    )


def dfe(
    *, data, y, x, panel, time, p=1, q=1, trend="c", max_iterations=500, missing="raise", alpha=0.05
):
    """Common ECM slopes and variance with unit deterministic effects."""
    return _fit(
        "dfe",
        data=data,
        y=y,
        x=x,
        panel=panel,
        time=time,
        p=p,
        q=q,
        trend=trend,
        max_iterations=max_iterations,
        missing=missing,
        alpha=alpha,
    )
