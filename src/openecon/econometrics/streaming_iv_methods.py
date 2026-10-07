"""Exact IV method algebra on QR geometry and replayed observation moments.

The factor is used only for linear geometry. GMM's residual-dependent weight
matrix and difference-in-J test always use complete, physical row replays.
"""
from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.streaming_ols import _CompensatedSum

from .core import kernel_call
from .iv import kernels
from .iv.common import IGMM_MAX_ITERATIONS, IGMM_TOLERANCE, estimate
from .iv.diagnostics import chi2_entry, unavailable
from .streaming_linear import _covariance, _weights


def moment_matrix(sample, notes, builder, width, *, kind, center=False, score_basis=None):
    """Sum-form moment covariance in the original estimator's coordinates."""
    mean = torch.zeros(width, dtype=torch.float64)
    if center:
        total = _CompensatedSum((width,))
        for batch in sample.batches():
            total.add((builder(batch)*_weights(sample, batch)[:, None]).sum(0))
        mean = total.value/sample.nobs
    def factory():
        for batch in sample.batches():
            rows = builder(batch)-mean
            one = torch.ones(len(rows), dtype=torch.float64)
            batch.residual_override = one
            yield batch, rows, one, one, 1.
    matrix, _, _, _ = _covariance(sample, factory, torch.zeros(width, dtype=torch.float64),
                        torch.eye(width, dtype=torch.float64), df=sample.nobs, k=0,
                        notes=notes, kind="HC1" if kind == "robust" else kind,
                        small=False, group_factor=False, collect_predictions=False,
                        score_basis=score_basis)
    return matrix


def estimate_replay(sample, notes, setting, y, designs, *, method, wmatrix,
                    igmm, center, response, instrument_basis=None,
                    convergence_scale=None, convergence_shift=None):
    """Return native Estimate, bread, and optional GMM score loading."""
    # Unadjusted GMM and LIML depend only on preserved linear cross products.
    initial_method = method if method != "gmm" or wmatrix == "unadjusted" else "2sls"
    est = estimate(notes, setting, y, designs, method=initial_method,
                   wmatrix=wmatrix, igmm=igmm, center=center)
    if method != "gmm" or wmatrix == "unadjusted":
        bread = (est.tsls.bread if est.kappa in {None, 1.} else
                 kernel_call(kernels.k_class, est.proj, setting.weights, est.kappa).bread)
        return est, bread, None
    x = torch.cat((designs.x1, designs.x2), 1)
    z = torch.cat((designs.x1, designs.z2), 1)
    weights = setting.weights
    from openecon.engines.linalg import weighted_crossprod
    zx, zy = weighted_crossprod(z, weights, x), weighted_crossprod(z, weights, y)
    beta, previous, iterations, converged = est.beta, None, 0, not igmm
    while True:
        def rows(batch):
            return batch.designs["instruments"]*(response(batch)-batch.designs["regressors"]@beta)[:, None]
        matrix = moment_matrix(sample, notes, rows, len(z[0]), kind=wmatrix,
                               center=center, score_basis=instrument_basis)
        step = kernel_call(kernels.gmm_step, zx, zy, matrix)
        iterations += 1
        if previous is not None:
            current_check, previous_check = step.beta, previous
            if convergence_scale is not None:
                current_check = current_check*convergence_scale
                previous_check = previous_check*convergence_scale
            if convergence_shift is not None:
                current_check = current_check+convergence_shift
                previous_check = previous_check+convergence_shift
            if float(((current_check-previous_check).abs()/(previous_check.abs()+1)).max()) < IGMM_TOLERANCE:
                converged = True
        weighting_beta = beta
        beta = step.beta
        if not igmm or converged:
            break
        if iterations >= IGMM_MAX_ITERATIONS:
            raise AnalysisError("nonconvergence", f"Iterated GMM did not converge in {IGMM_MAX_ITERATIONS} iterations.")
        previous = beta
    est.method, est.beta, est.condition_number = "gmm", beta, None
    est.resid = y-x@beta
    est.rss = float(weighted_crossprod(est.resid[:, None], weights)[0, 0])
    est.gmm = {"wmatrix": wmatrix, "iterations": iterations, "converged": converged,
               "igmm": igmm, "centered": center, "j": step.j}
    est._replay_weighting_beta = weighting_beta
    return est, step.bread, step.loading


def c_test(sample, notes, est, designs, setting, response, instrument_basis=None):
    """Native GMM difference-in-J, using the full OLS-residual moment covariance."""
    x = torch.cat((designs.x1, designs.x2), 1)
    z = torch.cat((designs.x1, designs.z2), 1)
    wide = torch.cat((z, x[:, designs.x1.shape[1]:]), 1)
    weights, q, nz = setting.weights, designs.x2.shape[1], z.shape[1]
    from openecon.engines.linalg import weighted_crossprod
    # Fit the actual response on X; this coefficient reconstructs original-row residuals.
    y = est.proj.y_p + designs.x1@est.proj.partial.beta[:, q]
    ols = kernel_call(kernels.solve, x, y, weights, "regressors")
    def rows(batch):
        actual_x, actual_z = batch.designs["regressors"], batch.designs["instruments"]
        instruments = torch.cat((actual_z, actual_x[:, designs.x1.shape[1]:]), 1)
        return instruments*(response(batch)-actual_x@ols.beta)[:, None]
    label = "C (difference-in-J) chi2 test of exogeneity"
    try:
        if est.gmm["wmatrix"] == "unadjusted":
            sigma = float(ols.ssr)/sample.nobs
            matrix = weighted_crossprod(wide, weights)*sigma
        else:
            matrix = moment_matrix(sample, notes, rows, wide.shape[1], kind=est.gmm["wmatrix"],
                                   center=est.gmm["centered"], score_basis=instrument_basis)
        exogenous = kernel_call(kernels.gmm_step, weighted_crossprod(wide, weights, x),
                                weighted_crossprod(wide, weights, y), matrix)
        model = kernel_call(kernels.gmm_step, weighted_crossprod(z, weights, x),
                            weighted_crossprod(z, weights, y), matrix[:nz, :nz])
    except AnalysisError:
        return unavailable(label, "the moment covariance is singular")
    return chi2_entry(exogenous.j-model.j, q, label, wmatrix=est.gmm["wmatrix"])


def method_diagnostics(tests, geometry, method, *, c_statistic=None):
    """Keep each method's canonical tests; common robust weak-ID is replayed."""
    if method == "2sls":
        return tests
    common = {name: row for name, row in tests.items()
              if name in {"cragg_donald", "kleibergen_paap_rk_f"}}
    if method in {"fuller", "kclass"}:
        common["overid_unavailable"] = geometry.tests["overid_unavailable"]
        return common
    if method == "liml":
        common.update({name: row for name, row in geometry.tests.items()
                       if name in {"anderson_rubin", "basmann_f"}})
    else:
        common["hansen_j"] = geometry.tests["hansen_j"]
        if c_statistic is not None:
            common["endog_c"] = c_statistic
    return common
