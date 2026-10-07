"""Prais-Winsten from a full-source joint current/lagged QR geometry.

Every rho update solves a small factor, with exact global lag cross-products.
Only the final HC covariance and ordered residual differences replay rows.
"""
from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import f_sf
from openecon.engines.execution import qr_factor
from openecon.engines.linalg import least_squares
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from .core import kernel_call, wald_test
from .ordered_replay import OrderedReplay
from .replay_sample import ReplaySample
from .streaming_linear import _Notes, _result
from .streaming_var import _Facade


def _fit(ordered):
    spec, notes = ordered.spec, _Notes(ordered.spec)
    prais = notes.option("method") == "prais"
    rhotype = notes.option("rhotype")
    twostep = bool(notes.option("twostep"))
    tolerance, maximum = float(notes.option("tolerance")), int(notes.option("max_iterations"))
    if tolerance <= 0:
        raise AnalysisError("invalid_option", "tolerance must be positive.")
    sample = ReplaySample(spec, ordered.source)
    design = sample.add_design("mean")
    sample.prepare()
    k, n = len(design.terms), sample.nrows
    nused = n-int(not prais)
    if k == 0:
        raise AnalysisError("empty_design", "No Prais-Winsten regressors remain.")
    if n < 3 or nused <= k:
        raise AnalysisError("insufficient_observations", "Prais-Winsten needs more usable periods than coefficients.")
    width = 2*(k+1)+1
    resource = sample.plan_rows("joint current/lagged Prais-Winsten geometry", {
        "joint_TSQR_tree_and_covariance": 1024*width**2,
        "sorted_source_reader_cache": 8*1024**2,
    }, 160*width)
    outcome = _WeightedMoments(2, intercept=True)
    for batch in sample.batches():
        y = batch.numeric(spec.outcome)
        outcome.add(torch.stack((torch.ones_like(y), y), dim=1), torch.ones_like(y), False)
    anchor, magnitude, mean = outcome.anchor[1], outcome.magnitude[1], outcome.mean[1]
    ymean = float(anchor+magnitude*mean) if spec.intercept else 0.
    yscale = float(magnitude*torch.sqrt(outcome.m2.value[1]/outcome.mass))
    if not math.isfinite(yscale) or yscale <= 0:
        raise AnalysisError("constant_outcome", "Prais-Winsten outcome has no finite variation.")

    def blocks():
        for batch in sample.batches():
            raw = batch.numeric(spec.outcome)
            y = ((raw-anchor)/magnitude-mean)*(magnitude/yscale) if spec.intercept else raw/yscale
            yield batch, batch.designs["mean"], y

    tree, previous, first = _TSQRTree(), None, None
    for _, x, y in blocks():
        values = torch.cat((x, y[:, None]), dim=1)
        if first is None:
            first = values[:1].clone()
        joined = values if previous is None else torch.cat((previous, values))
        start = 1 if previous is None else 0
        current = values[start:]
        older = joined[:len(joined)-1]
        if len(current):
            tree.add(qr_factor(torch.cat((torch.ones((len(current), 1), dtype=torch.float64), current, older), dim=1)))
        previous = values[-1:].clone()
    joint = tree.finish()
    ones = joint[:, 0]
    current, older = joint[:, 1:k+2], joint[:, k+2:]
    xc, yc, xp, yp = current[:, :k], current[:, k], older[:, :k], older[:, k]
    x0, y0 = first[0, :k], first[0, k]
    xfull, yfull = torch.cat((xc, x0[None, :])), torch.cat((yc, y0[None]))
    ols = kernel_call(least_squares, xfull, yfull, drop_collinear=False)
    if float(ols.ssr) <= 1e-20*float(yfull@yfull):
        raise AnalysisError("perfect_fit", "The regression fits the outcome exactly; rho is undefined.")

    def residual_geometry(beta):
        uc, up, u0 = yc-xc@beta, yp-xp@beta, y0-x0@beta
        cross, currentss, previousss = float(uc@up), float(uc@uc), float(up@up)
        total = currentss+float(u0*u0)
        differences = float((uc-up).square().sum())
        if min(currentss, previousss, total) <= 0:
            raise AnalysisError("perfect_fit", "Residual variation is insufficient for an AR(1) error estimate.")
        dw = differences/total
        if rhotype == "regress":
            rho = cross/previousss
        elif rhotype == "freg":
            rho = cross/currentss
        elif rhotype == "tscorr":
            rho = cross/total
        elif rhotype == "theil":
            rho = cross/total*(n-k)/n
        elif rhotype == "dw":
            rho = 1-dw/2
        else:
            rho = ((1-dw/2)*n*n+k*k)/(n*n-k*k)
        if not math.isfinite(rho):
            raise AnalysisError("numerical_failure", "rho cannot be represented as finite float64.")
        if prais and abs(rho) >= 1:
            raise AnalysisError("rho_out_of_range", "Estimated rho is outside (-1,1); use Cochrane-Orcutt or difference the series.")
        return rho, dw

    def transformed_fit(rho):
        xs, ys = xc-rho*xp, yc-rho*yp
        if prais:
            root = math.sqrt(1-rho*rho)
            xs, ys = torch.cat((xs, root*x0[None, :])), torch.cat((ys, root*y0[None]))
        fitted = kernel_call(least_squares, xs, ys, drop_collinear=False)
        return fitted, xs, ys

    rho, dw_original = residual_geometry(ols.beta)
    path, iterations, converged = [0., rho], 1, twostep
    fit, xs, ys = transformed_fit(rho)
    while not twostep and iterations < maximum:
        new, _ = residual_geometry(fit.beta)
        change, rho = abs(new-rho), new
        path.append(rho)
        iterations += 1
        fit, xs, ys = transformed_fit(rho)
        if change < tolerance:
            converged = True
            break
    if not converged:
        raise AnalysisError("nonconvergence", f"rho did not converge in {maximum} iterations; use twostep or change tolerance.")
    if abs(rho) >= 1:
        notes.warn("Estimated rho is outside (-1,1); the errors look nonstationary.")
    beta, bread, ssr = fit.beta, fit.xtx_inv, float(fit.ssr)
    df = nused-k
    kind = "HC1" if spec.covariance == "robust" else spec.covariance
    meat, dw_numerator, dw_denominator = _CompensatedSum((k, k)), _CompensatedSum(()), _CompensatedSum(())
    previous, residual_tail, seen = None, None, 0
    predictions = []
    preview_positions = [row[0] for row in ordered.db.execute(
        "SELECT pos FROM rows WHERE valid=1 ORDER BY t,pos LIMIT 400 OFFSET ?", (int(not prais),))]
    for batch, x, y in blocks():
        values = torch.cat((x, y[:, None]), dim=1)
        joined = values if previous is None else torch.cat((previous, values))
        start = 1 if previous is None else 0
        transformed = values[start:]-rho*joined[:len(joined)-1]
        if previous is None and prais:
            transformed = torch.cat((math.sqrt(1-rho*rho)*values[:1], transformed))
        tx, ty = transformed[:, :k], transformed[:, k]
        residual = ty-tx@beta
        if len(residual):
            chain = residual if residual_tail is None else torch.cat((residual_tail, residual))
            dw_numerator.add(chain.diff().square().sum())
            dw_denominator.add(residual.square().sum())
            residual_tail = residual[-1:].clone()
        if kind != "nonrobust":
            if kind in {"HC2", "HC3"}:
                leverage = ((tx@bread)*tx).sum(1)
                if bool((leverage >= 1).any()):
                    raise AnalysisError("invalid_leverage", "HC2/HC3 requires leverage below one.")
                residual = residual/(1-leverage).sqrt() if kind == "HC2" else residual/(1-leverage)
            score = tx*residual[:, None]
            meat.add(score.T@score)
        skip = int(seen == 0 and not prais)
        if len(predictions) < 400:
            raw = batch.numeric(spec.outcome)
            structural = yscale*(x@beta)+ymean
            for i in range(skip, min(len(y), skip+400-len(predictions))):
                predictions.append({"row": preview_positions[len(predictions)], "observed": float(raw[i]), "fitted": float(structural[i]), "residual": float(raw[i]-structural[i])})
        previous, seen = values[-1:].clone(), seen+len(values)
    covariance = bread*(ssr/df) if kind == "nonrobust" else bread@meat.value@bread*(nused/df if kind == "HC1" else 1.)
    target_ones = torch.cat((ones, torch.ones(1, dtype=torch.float64))) if prais else ones
    # Prais transforms the intercept into a nonconstant first-row vector.
    # Restore the original outcome level before the Stata hascons TSS, even
    # though centering the outcome is harmless for coefficient estimation.
    actual_target = ys+(ymean/yscale)*xs[:, 0] if spec.intercept else ys
    centered = actual_target-target_ones*float(target_ones@actual_target/nused) if spec.intercept else actual_target
    total = float(centered@centered)
    r2 = 1-ssr/total if total > 0 else math.nan
    mapping = yscale*design.transform
    raw_beta, raw_cov = mapping@beta, mapping@covariance@mapping.T
    if spec.intercept:
        raw_beta[0] += ymean
    terms = design.terms
    slopes = [i for i, term in enumerate(terms) if term != "Intercept"]
    tests = {}
    df_model = k-int(spec.intercept)
    if slopes and spec.intercept and kind == "nonrobust":
        value = ((total-ssr)/df_model)/(ssr/df)
        tests["model"] = {"statistic": value, "df": df_model, "df2": df,
                          "p_value": f_sf(max(value, 0.), df_model, df), "distribution": "F",
                          "label": "ANOVA F test of the transformed regression (Stata prais)"}
    elif slopes:
        tests["model"] = wald_test(raw_beta, raw_cov, slopes, df_resid=df, label="F test of slopes")
    offset = int(not prais)
    reporting = sample.sample.iloc[offset:].copy()
    positions = [record["row"] for record in predictions]
    facade = _Facade(spec, ordered, reporting, positions[:len(reporting)], nused, sample.categories, sample.provenance())
    facade.notes = sample.notes
    info = {"covariance": spec.covariance, "df_resid": df, "df_inference": df,
            "small_sample_correction": nused/df if kind == "HC1" else 1. if kind in {"HC2", "HC3"} else None,
            "correction": kind+" on transformed data; rho treated as known"}
    name = "Prais-Winsten" if prais else "Cochrane-Orcutt"
    return _result(facade, terms=terms, beta=raw_beta, covariance=raw_cov, info=info,
                   metrics={"r_squared": r2, "adjusted_r_squared": 1-(1-r2)*(nused-int(spec.intercept))/df,
                            "rmse": yscale*math.sqrt(ssr/df), "rho": rho,
                            "durbin_watson_original": dw_original, "durbin_watson_transformed": float(dw_numerator.value/dw_denominator.value),
                            "iterations": iterations, "df_model": df_model, "df_resid": df, "ssr": ssr*yscale*yscale},
                   notes=notes, predictions=predictions, tests=tests,
                   solver="global_joint_lag_TSQR_feasible_gls", diagnostics={"iterations": iterations, "converged": converged,
                        "condition_number": fit.condition_number, "rho_iterations_replay_source": False},
                   extra={"method": notes.option("method"), "rhotype": rhotype, "twostep": twostep, "rho": rho,
                          "rho_path": path, "converged": converged, "tolerance": tolerance,
                          "transformed_r_squared_definition": "1-SSR*/TSS*; centered about the actual-period mean of y* when an intercept is present"},
                   resource=resource.record(), title=f"{name} AR(1) regression")


def fit_streaming_prais(spec, source):
    with OrderedReplay(spec, source) as ordered:
        result = kernel_call(_fit, ordered)
        ordered.verify_original()
        return result
