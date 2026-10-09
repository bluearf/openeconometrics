"""Replay 2SLS: small orthogonal geometry, actual-row sandwich/diagnostics.

Residual products and heteroskedastic moments are NEVER inferred from compressed
QR rows.  Small factors preserve only linear cross products; every nonlinear
score product is reconstructed on real retained source rows and accumulated.
"""
from __future__ import annotations

import math
from copy import copy
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import cholesky_inverse, wald_statistic
from openecon.engines.streaming_ols import _CompensatedSum
from openecon.linear_ols.streaming import _WeightedMoments

from .core import Design, kernel_call, wald_test
from .iv.common import Blocks, Setting, check_fit
from .iv.diagnostics import Diagnostics, chi2_entry, unavailable
from .iv.ivregress import check_hac, gmm_options
from .replay_sample import ReplaySample
from .streaming_linear import _Notes, _covariance, _factor, _finite, _resource, _result, _solve, _weights
from .streaming_iv_methods import c_test, estimate_replay, method_diagnostics


def _roles(spec):
    endogenous, instruments = spec.columns["endogenous"], spec.columns["instruments"]
    endogenous = [endogenous] if isinstance(endogenous, str) else list(endogenous)
    instruments = [instruments] if isinstance(instruments, str) else list(instruments)
    for names, other, message in ((endogenous, spec.predictors, "Exogenous and endogenous roles must be disjoint."),
                                  (instruments, spec.predictors, "List only excluded instruments, not the exogenous regressors."),
                                  (instruments, endogenous, "An endogenous regressor cannot instrument itself.")):
        if set(names)&set(other):
            raise AnalysisError("invalid_spec", message)
    if spec.outcome in endogenous or spec.outcome in instruments or set(spec.categorical)-set(spec.predictors):
        raise AnalysisError("invalid_spec", "Only exogenous predictors may be categorical; the outcome cannot also be an instrument/regressor.")
    return endogenous, instruments


def _geometry_rows(factor, intercept, n):
    """Rotate QR rows to preserve the literal constant AND linear cross products.

    Artificial weights n/m compensate the m/n row scaling.  These rows are
    valid for cross-product geometry only, never heteroskedastic moment meats.
    """
    if not intercept:
        return factor, None
    m = len(factor)
    target = torch.ones(m, dtype=torch.float64)*math.sqrt(n/m)
    direction = factor[:, 0]-target
    norm = float(direction@direction)
    rotated = factor if norm == 0 else factor-2*direction[:, None]*(direction@factor)[None, :]/norm
    rotated = rotated*math.sqrt(m/n)
    rotated[:, 0] = 1.
    return rotated, torch.full((m,), n/m, dtype=torch.float64)


def _fit_matrix(x, y):
    y = y[:, None] if y.ndim == 1 else y
    q, r = torch.linalg.qr(x, mode="reduced")
    beta = torch.linalg.solve_triangular(r, q.T@y, upper=True)
    return beta, y-x@beta


def fit_iv_replay(spec, source, *, batch_rows=None):
    notes = _Notes(spec)
    method = notes.option("method")
    if spec.weight_type == "pweight" and spec.covariance == "nonrobust":
        raise AnalysisError("unsupported_covariance", "pweights require robust or cluster covariance.")
    wmatrix, igmm, center = gmm_options(notes, method)
    check_hac(notes, wmatrix)
    endogenous, instruments = _roles(spec)
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    regressors = sample.add_design("regressors", [*spec.predictors, *endogenous])
    instrument = sample.add_design("instruments", [*spec.predictors, *instruments])
    sample.prepare()
    exog_terms = [term for term in regressors.terms if term not in endogenous]
    k1, k = len(exog_terms), len(regressors.terms)
    q, n_instruments = k-k1, len(instrument.terms)
    l2 = n_instruments-k1
    if not q:
        raise AnalysisError("no_endogenous_regressors", "All endogenous regressors were omitted; use ordinary regression.")
    if l2 < q:
        raise AnalysisError("underidentified", "There are fewer usable excluded instruments than endogenous regressors.")
    if regressors.terms[:k1] != instrument.terms[:k1]:
        raise AnalysisError("singular_design", "Exogenous rank screens differ between the regressor/instrument blocks.")
    if sample.nrows <= len(regressors.anchor):
        raise AnalysisError("insufficient_observations", "2SLS needs more distinct rows than regressor columns.")
    resource = _resource(sample, k+n_instruments)
    omitted_instruments = [name for name in instruments if name not in instrument.terms]
    sample.notes["omitted_terms"] = list(dict.fromkeys(term for term in sample.notes["omitted_terms"] if term not in omitted_instruments))
    if omitted_instruments:
        notes.warn("Instruments omitted because of collinearity: "+", ".join(omitted_instruments)+".")
    moments = _WeightedMoments(2, intercept=True)
    for batch in sample.batches():
        y = batch.numeric(spec.outcome)
        moments.add(torch.stack((torch.ones_like(y), y), 1), _weights(sample, batch), False)
    y_mean = float(moments.anchor[1]+moments.magnitude[1]*moments.mean[1]) if spec.intercept else 0.
    variance = float(moments.m2.value[1]/moments.mass)
    y_scale = float(moments.magnitude[1])*math.sqrt(variance)
    if not spec.intercept:
        y_scale = max(y_scale, abs(float(moments.anchor[1]+moments.magnitude[1]*moments.mean[1])))
    if not math.isfinite(y_scale) or y_scale <= 0:
        raise AnalysisError("constant_outcome", "The IV outcome has no finite variation.")

    def y_work(batch):
        y = batch.numeric(spec.outcome)
        if spec.intercept:
            return ((y-moments.anchor[1])/moments.magnitude[1]-moments.mean[1])*(float(moments.magnitude[1])/y_scale)
        return y/y_scale

    def blocks():
        for batch in sample.batches():
            x, z = batch.designs["regressors"], batch.designs["instruments"]
            yield torch.cat((z, x[:, k1:]), 1), y_work(batch), _weights(sample, batch)

    factor, diagnostics = _factor(blocks, n_instruments+q, require_more=False)
    compressed, virtual_weights = _geometry_rows(factor, spec.intercept, sample.nobs)
    z, x2, y = compressed[:, :n_instruments], compressed[:, n_instruments:n_instruments+q], compressed[:, -1]
    x1 = z[:, :k1]
    designs = Blocks(Design(x1, exog_terms, sample.categories, spec.intercept),
                     Design(x2, regressors.terms[k1:], {}, False),
                     Design(z[:, k1:], instrument.terms[k1:], {}, False),
                     x1, x2, z[:, k1:], omitted_instruments)
    setting = Setting(sample.nobs, virtual_weights, "nonrobust", bool(notes.option("small")))
    instrument_scales = instrument.scales[instrument.kept]
    instrument_basis = torch.diag(instrument_scales)
    convergence_shift = torch.zeros(k, dtype=torch.float64)
    if spec.intercept:
        convergence_shift[0] = y_mean
    est, bread, loading = estimate_replay(sample, notes, setting, y, designs, method=method,
                wmatrix=wmatrix, igmm=igmm, center=center, response=y_work,
                instrument_basis=instrument_basis,
                convergence_scale=y_scale/regressors.scales[regressors.kept],
                convergence_shift=convergence_shift)
    diagnostic_est = copy(est)
    if method == "gmm":
        diagnostic_est.method = "2sls"
    geometry = Diagnostics(notes, setting, y, designs, diagnostic_est).run()
    if method == "gmm":
        label = ("Hansen's J test of overidentifying restrictions" if wmatrix != "unadjusted" else
                 "Sargan test of overidentifying restrictions (unadjusted weight matrix)")
        geometry.tests["hansen_j"] = (chi2_entry(est.gmm["j"], l2-q, label, wmatrix=wmatrix)
                 if l2>q else unavailable("Test of overidentifying restrictions", "exactly identified: there are no overidentifying restrictions"))
    beta = est.beta
    first_coef = est.proj.first.beta
    transform = regressors.transform
    score_basis = torch.linalg.inv(transform)
    small = bool(notes.option("small"))
    df = sample.nobs-k

    def residual_blocks():
        for batch in sample.batches():
            x, z = batch.designs["regressors"], batch.designs["instruments"]
            # First projection includes y and X2; X1 instruments itself.
            fitted_endog = z@first_coef[:, :q]
            if loading is not None:
                projected = z@loading
            else:
                kappa = est.kappa if est.kappa is not None else 1.
                projected = torch.cat((x[:, :k1], fitted_endog-(kappa-1)*(x[:, k1:]-fitted_endog)), 1)
            response = y_work(batch)
            batch.residual_override = response-x@beta
            yield batch, projected, response, batch.numeric(spec.outcome), y_scale

    covariance, info, rss_scaled, predictions = _covariance(sample, residual_blocks, beta, bread,
                                     df=df, k=k, notes=notes, score_basis=score_basis,
                                     kind="HC1" if spec.covariance == "robust" else None, small=small)
    if loading is not None:
        if spec.covariance == wmatrix:
            covariance = bread*info.get("small_sample_correction", 1.)
            info["correction"] = "efficient GMM (X'Z S^-1 Z'X)^-1; "+info["correction"]
        elif spec.covariance == "nonrobust":
            from openecon.engines.linalg import weighted_crossprod
            middle = loading.T@weighted_crossprod(z, virtual_weights)@loading
            covariance = rss_scaled/(df if small else sample.nobs)*(bread@middle@bread)
            info["correction"] = "GMM sandwich with unadjusted moment covariance"
    rss = rss_scaled*y_scale**2
    centered_tss = float(moments.m2.value[1])*float(moments.magnitude[1])**2*moments.weight_max
    raw_mean = float(moments.anchor[1]+moments.magnitude[1]*moments.mean[1])
    uncentered = centered_tss+sample.nobs*raw_mean**2
    tss = centered_tss if spec.intercept else uncentered
    check_fit(rss, tss, df, uncentered)
    raw_beta = y_scale*(transform@beta)
    if spec.intercept:
        raw_beta[0] += y_mean
    raw_covariance = y_scale**2*(transform@covariance@transform.T)
    _finite(raw_beta, raw_covariance)
    tests = dict(geometry.tests)
    first_stage = geometry.first_stage
    if spec.covariance != "nonrobust" and geometry.df_first >= q:
        tests, first_stage = _robust_diagnostics(spec, sample, notes, designs, est, geometry,
                                                x1, x2, z, y, virtual_weights, first_coef,
                                                y_work, k1, q, n_instruments, l2, regressors, instrument)
    c_statistic = None
    if method == "gmm":
        wide_basis = torch.diag(torch.cat((instrument_scales, regressors.scales[regressors.kept][k1:])))
        c_statistic = (unavailable("Test that the endogenous regressors are exogenous",
                                  "an endogenous regressor is fitted exactly by the instruments")
                       if geometry.exact_first_stage else
                       c_test(sample, notes, est, designs, setting, y_work, wide_basis))
    tests = method_diagnostics(tests, geometry, method, c_statistic=c_statistic)
    reference_df = info["df_inference"] if small else None
    tests["model"] = wald_test(raw_beta, raw_covariance,
                         [index for index, term in enumerate(regressors.terms) if term != "Intercept"],
                         df_resid=reference_df,
                         label="Model F test (slopes)" if small else "Wald chi2 test of the slopes")
    info.update({"covariance": spec.covariance, "df_resid": df, "df_inference": reference_df,
                 "nobs": sample.nobs, "method": method, "small": small,
                 "error_variance": "RSS/(N-K)" if small else "RSS/N",
                 "r_squared_definition": "centered" if spec.intercept else "uncentered",
                 "residual_definition": "outcome minus X b with the observed (not fitted) endogenous regressors"})
    r2 = 1-rss/tss
    metrics = {"r_squared": r2, "adjusted_r_squared": 1-(1-r2)*(sample.nobs-int(spec.intercept))/df,
               "rmse": math.sqrt(rss/(df if small else sample.nobs)), "df_model": k-int(spec.intercept),
               "df_resid": df, "n_instruments": l2, "n_endogenous": q}
    if est.kappa is not None:
        metrics["kappa"] = est.kappa
    if est.gmm is not None:
        metrics.update({"j": est.gmm["j"], "gmm_iterations": est.gmm["iterations"]})
    return _result(sample, terms=regressors.terms, beta=raw_beta, covariance=raw_covariance,
                  info=info, metrics=metrics, notes=notes, predictions=predictions, tests=tests,
                  solver="native_joint_tsqr_two_stage_and_score_replay", diagnostics=diagnostics,
                  extra={"method": method, "exogenous": exog_terms,
                         "endogenous": regressors.terms[k1:], "instruments": instrument.terms[k1:],
                         "omitted_instruments": omitted_instruments, "first_stage": first_stage,
                         **({"kappa": est.kappa} if est.kappa is not None else {}),
                         **({"gmm": est.gmm} if est.gmm is not None else {})},
                  resource=resource, use_t=small, title=f"Instrumental-variables ({method.upper()}) regression")


def _robust_diagnostics(spec, sample, notes, designs, est, geometry,
                        x1, x2, z, y, virtual_weights, first_coef,
                        y_work, k1, q, n_instruments, l2, regressors, instrument, *, absorbed=0):
    k, n = k1+q, sample.nobs
    kind = "HC1" if spec.covariance == "robust" else spec.covariance
    partial = est.proj.partial.beta
    stage_coef, _ = _fit_matrix(est.proj.z2p, est.proj.x2p)
    sqrt_weight = (torch.ones(len(z), dtype=torch.float64) if virtual_weights is None else virtual_weights.sqrt())
    stage_factor = torch.cat((est.proj.z2p, est.proj.x2p[:, :1]), 1)*sqrt_weight[:, None]
    _, stage_bread, _ = _solve(stage_factor, l2)
    excluded_scales = instrument.scales[instrument.kept][k1:]
    endog_scales = regressors.scales[regressors.kept][k1:]

    def paths(batch):
        x, all_z = batch.designs["regressors"], batch.designs["instruments"]
        exog, endog, excluded = x[:, :k1], x[:, k1:], all_z[:, k1:]
        z_partial = excluded-exog@partial[:, q+1:]
        x_partial = endog-exog@partial[:, :q]
        error = endog-all_z@first_coef[:, :q]
        return x, z_partial, x_partial, error, z_partial@stage_coef

    first_stage = [dict(row) for row in geometry.first_stage]
    for j, entry in enumerate(first_stage):
        def first_blocks():
            for batch in sample.batches():
                _, zp, xp, error, _ = paths(batch)
                batch.residual_override = error[:, j]
                yield batch, zp, xp[:, j], xp[:, j], 1.
        covariance, info, _, _ = _covariance(sample, first_blocks, stage_coef[:, j], stage_bread,
                    df=n-n_instruments-absorbed, k=n_instruments+absorbed, notes=notes,
                    score_basis=torch.diag(excluded_scales), kind=kind, collect_predictions=False)
        test = wald_test(stage_coef[:, j], covariance, range(l2), df_resid=info["df_inference"])
        entry.update({"f_statistic": test["statistic"], "df": test["df"], "df2": test.get("df2"),
                      "p_value": test["p_value"], "covariance": spec.covariance})

    def moment_meat(builder, width, *, score_basis=None):
        total = _CompensatedSum((width,))

        def moment_blocks():
            for batch in sample.batches():
                rows = builder(batch)
                total.add((rows*_weights(sample, batch)[:, None]).sum(0))
                ones = torch.ones(len(batch.frame), dtype=torch.float64)
                batch.residual_override = ones
                yield batch, rows, ones, ones, 1.

        matrix, info, _, _ = _covariance(sample, moment_blocks, torch.zeros(width, dtype=torch.float64),
                    torch.eye(width, dtype=torch.float64), df=n, k=0, notes=notes,
                    kind=kind, small=False, group_factor=False,
                    score_basis=score_basis, collect_predictions=False)
        return total.value, matrix, info

    tests = {"cragg_donald": geometry.tests["cragg_donald"]}
    if est.method == "2sls":
        if l2 == q:
            tests["overid_score"] = unavailable("Test of overidentifying restrictions",
                                                 "exactly identified: there are no overidentifying restrictions")
        else:
            projected_coef, _ = _fit_matrix(est.tsls.g, est.proj.z2p)

            def overid_rows(batch):
                x, zp, _, _, projected = paths(batch)
                return (zp-projected@projected_coef)*(y_work(batch)-x@est.beta)[:, None]

            total, matrix, _ = moment_meat(overid_rows, l2, score_basis=torch.diag(excluded_scales))
            statistic, rank = kernel_call(wald_statistic, total, matrix)
            tests["overid_score"] = chi2_entry(statistic, l2-q,
                                               "Wooldridge robust score test of overidentifying restrictions")
            if rank != l2-q:
                tests["overid_score"]["note"] = f"the moment covariance has rank {rank}, not {l2-q}"

        label = "Test that the endogenous regressors are exogenous"
        if geometry.exact_first_stage:
            for name in ("endog_robust_score", "endog_robust_regression"):
                tests[name] = unavailable(label, "an endogenous regressor is fitted exactly by the instruments")
            return tests, first_stage
        actual_x = torch.cat((x1, x2), 1)
        ols_coef, _ = _fit_matrix(actual_x, torch.cat((y[:, None], est.proj.e), 1))
        residualized_e = est.proj.e-actual_x@ols_coef[:, 1:]
        ols_resid = y-actual_x@ols_coef[:, 0]
        df2 = n-k-q-absorbed

        def endogenous_paths(batch):
            x, _, _, error, _ = paths(batch)
            return y_work(batch)-x@ols_coef[:, 0], error-x@ols_coef[:, 1:]

        try:
            augmented_factor = torch.cat((residualized_e, ols_resid[:, None]), 1)*sqrt_weight[:, None]
            augmented_beta, augmented_bread, _ = _solve(augmented_factor, q)
            if df2 <= 0:
                raise AnalysisError("insufficient_observations", "No augmented-regression residual degrees of freedom remain.")
        except AnalysisError:
            for name in ("endog_robust_score", "endog_robust_regression"):
                tests[name] = unavailable(label, "the first-stage residuals are collinear with the regressors or no degrees of freedom remain")
        else:
            def endog_score_rows(batch):
                residual, instrument_resid = endogenous_paths(batch)
                return instrument_resid*residual[:, None]

            total, matrix, _ = moment_meat(endog_score_rows, q, score_basis=torch.diag(endog_scales))
            statistic, rank = kernel_call(wald_statistic, total, matrix)
            tests["endog_robust_score"] = chi2_entry(statistic, rank, "Wooldridge robust score chi2 test of exogeneity")

            def endog_regression_blocks():
                for batch in sample.batches():
                    residual, instruments_resid = endogenous_paths(batch)
                    yield batch, instruments_resid, residual, residual, 1.

            covariance, info, _, _ = _covariance(sample, endog_regression_blocks, augmented_beta, augmented_bread,
                           df=df2, k=k+q+absorbed, notes=notes, kind=kind,
                           score_basis=torch.diag(endog_scales), collect_predictions=False)
            tests["endog_robust_regression"] = wald_test(augmented_beta, covariance, range(q),
                             df_resid=info["df_inference"], label="Robust regression-based F test of exogeneity")

    if geometry.exact_first_stage:
        return tests, first_stage

    label = "Kleibergen-Paap rk Wald F"
    try:
        weighted_xp, weighted_zp = est.proj.x2p*sqrt_weight[:, None], est.proj.z2p*sqrt_weight[:, None]
        lower_y = torch.linalg.cholesky(weighted_xp.T@weighted_xp)
        lower_z = torch.linalg.cholesky(weighted_zp.T@weighted_zp)
        cross = weighted_xp.T@weighted_zp
        theta = torch.linalg.solve_triangular(lower_z,
                    torch.linalg.solve_triangular(lower_y, cross, upper=False).T, upper=False).T
        u, singular, vh = torch.linalg.svd(theta, full_matrices=True)
        direction = torch.linalg.solve_triangular(lower_y.T, u[:, q-1:q], upper=True)
        basis = torch.linalg.solve_triangular(lower_z.T, vh[q-1:].T, upper=True)

        def rank_rows(batch):
            _, zp, _, error, _ = paths(batch)
            return (zp@basis)*(error@direction)

        _, matrix, info = moment_meat(rank_rows, l2-q+1)
        inverse = kernel_call(cholesky_inverse, matrix)
        rk = float(singular[q-1].square()*inverse[0, 0])
        statistic = rk/n*(n-n_instruments-absorbed)/l2
        if spec.covariance == "cluster":
            groups = info["cluster_count"]
            statistic = rk/(n-1)*(n-n_instruments-absorbed)*(groups-1)/groups/l2
        tests["kleibergen_paap_rk_f"] = {"statistic": statistic, "df": l2, "df2": n-n_instruments-absorbed,
                 "p_value": None, "distribution": "F", "label": label, "rk_wald_chi2": rk,
                 "rk_df": l2-q+1,
                 "note": "weak-identification statistic: compare with Stock-Yogo critical values"}
    except (AnalysisError, KernelError, torch.linalg.LinAlgError):
        tests["kleibergen_paap_rk_f"] = unavailable(label, "the covariance of the rank statistic is singular", "F")
    return tests, first_stage
