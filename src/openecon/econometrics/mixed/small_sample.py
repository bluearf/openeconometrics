"""Kenward-Roger and Satterthwaite fixed-effect inference for saved Gaussian ``mixed`` fits.

Covariance geometry
-------------------
A saved ``mixed`` fit has the marginal covariance ``V = sum_r s_r G_r`` that is
*linear* in its reported variance parameters ``s`` (top-level intercept
variance, the entries of the lowest-level covariance G as reported, and the
residual variance): every ``G_r`` is a fixed 0/1-pattern or ``Z_a Z_b'``
matrix, block diagonal over the independent top-level groups. In this linear
parameterization the second derivatives of V vanish, so the Kenward-Roger
(1997) adjustment has no ``R_ij`` term and is invariant to the choice of
variance coordinates (pbkrtest's and SAS's convention).

With ``Phi = (X'V^-1 X)^-1``, ``P = V^-1 - V^-1 X Phi X'V^-1``,
``P_r = -X'V^-1 G_r V^-1 X`` and ``Q_rs = X'V^-1 G_r V^-1 G_s V^-1 X``:

* expected REML information ``I_rs = tr(P G_r P G_s)/2``, ``W = I^-1``;
* ``Phi_A = Phi + 2 Phi [sum_rs W_rs (Q_rs - P_r Phi P_s)] Phi``
  (Kenward and Roger 1997, eq. 2-3 with R_ij = 0);
* for a full-row-rank q x p contrast L, ``Theta = L'(L Phi L')^-1 L``,
  ``A1 = sum W_rs tr(Theta Phi P_r Phi) tr(Theta Phi P_s Phi)``,
  ``A2 = sum W_rs tr(Theta Phi P_r Phi Theta Phi P_s Phi)`` and the
  moment-matched F(q, m) reference with scale lambda (KR 1997, section 4).

Formulas follow pbkrtest 0.5.5 (``vcovAdj_internal``, ``.KR_adjust``,
``Lb_ddf``; https://github.com/cran/pbkrtest) and the scalar t of lmerTest
3.2.1 ``contest1D`` (adjusted variance with the ``Lb_ddf`` df, no F scaling).

Satterthwaite uses the gradient of ``L Phi L'`` in s, ``dPhi/ds_r = -Phi P_r
Phi``, and the inverse *observed* information of the (restricted) likelihood
with b profiled out, ``a_r'P a_s - tr(P G_r P G_s)/2`` (REML) or
``a_r'P a_s - tr(V^-1 G_r V^-1 G_s)/2`` (ML) with ``a_r = G_r V^-1 r``. At a
stationary fit this equals lmerTest's numerical deviance Hessian after the
change of coordinates. A joint contrast is decomposed by the eigenvectors of
``L Phi L'`` and the component dfs combined as in lmerTest ``contestMD`` /
``get_Fstat_ddf`` (Fai and Cornelius 1996).

Everything is dense per top-level group on padded [groups, m, m] batches; the
work and workspace are planned before allocation.
"""

from __future__ import annotations

import math
from numbers import Real

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, TableSet, table
from openecon.econometrics.mixed.lmm import _Setup
from openecon.econometrics.postest.common import matched_frame, require_result
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.engines import distributions
from openecon.resources import plan_workspace

DDF_METHODS = ("kenward-roger", "satterthwaite")
MAX_ROWS = 8192
MAX_GROUP = 512
KR_SOURCE = "https://github.com/cran/pbkrtest"


def padded_positions(codes: Tensor, count: int) -> tuple[Tensor, int]:
    """Position of every row inside its group and the largest group size."""
    order = torch.argsort(codes, stable=True)
    sizes = torch.bincount(codes, minlength=count)
    starts = torch.cumsum(sizes, 0) - sizes
    position = torch.empty_like(codes)
    position[order] = torch.arange(len(codes)) - starts[codes[order]]
    return position, int(sizes.max())


def _pad(values: Tensor, group: Tensor, position: Tensor, count: int, m: int) -> Tensor:
    out = torch.zeros((count, m, *values.shape[1:]), dtype=torch.float64)
    out[group, position] = values
    return out


class LinearCovariance:
    """Dense per-group quantities of a Gaussian model with V linear in s."""

    def __init__(self, x: Tensor, y: Tensor, mask: Tensor, derivatives: Tensor, s: Tensor):
        # x [S,m,p], y [S,m], mask [S,m], derivatives [k,S,m,m], s [k]
        self.x, self.y, self.mask, self.g, self.s = x, y, mask, derivatives, s
        v = torch.einsum("k,ksab->sab", s, derivatives) + torch.diag_embed(1 - mask)
        lower, info = torch.linalg.cholesky_ex(v)
        if bool((info != 0).any()):
            raise AnalysisError("invalid_covariance", "The saved variance parameters do not give "
                                "a positive definite marginal covariance.")
        self.lower = lower
        self.vx = torch.cholesky_solve(x, lower)
        information = torch.einsum("sap,saq->pq", x, self.vx)
        chol, info = torch.linalg.cholesky_ex((information + information.T) / 2)
        if int(info) != 0:
            raise AnalysisError("singular_design", "X'V^-1 X is not positive definite.")
        self.phi = torch.cholesky_inverse(chol)
        vy = torch.cholesky_solve(y[..., None], lower)[..., 0]
        self.beta = self.phi @ torch.einsum("sap,sa->p", x, vy)
        resid = (y - torch.einsum("sap,p->sa", x, self.beta)) * mask
        self.vr = torch.cholesky_solve(resid[..., None], lower)[..., 0]
        # H_r = V^-1 G_r, O_r = G_r V^-1 X, P_r = -X'V^-1 G_r V^-1 X
        self.h = torch.cholesky_solve(derivatives, lower)
        self.o = derivatives @ self.vx
        self.p_r = -torch.einsum("sap,ksaq->kpq", self.vx, self.o)
        self.p_r = (self.p_r + self.p_r.transpose(1, 2)) / 2
        vo = torch.cholesky_solve(self.o, lower)
        self.q_rs = torch.einsum("isap,jsaq->ijpq", self.o, vo)
        self.ktrace = torch.einsum("isab,jsba->ij", self.h, self.h)
        phi_p = torch.einsum("pq,kqr->kpr", self.phi, self.p_r)
        self.ie2 = (self.ktrace - 2 * torch.einsum("pq,ijqp->ij", self.phi, self.q_rs)
                    + torch.einsum("ipq,jqp->ij", phi_p, phi_p))
        self.ie2 = (self.ie2 + self.ie2.T) / 2
        # a_r = G_r V^-1 r and a_r'P a_s (observed information pieces)
        a = torch.einsum("ksab,sb->ksa", derivatives, self.vr)
        va = torch.cholesky_solve(a[..., None], lower)[..., 0]
        b = torch.einsum("sap,ksa->kp", self.vx, a)
        self.apa = torch.einsum("isa,jsa->ij", a, va) - b @ self.phi @ b.T
        self.apa = (self.apa + self.apa.T) / 2
        self.rgr = torch.einsum("sa,ksa->k", self.vr, a)          # r'V^-1 G_r V^-1 r
        self.trace_vg = torch.einsum("ksaa->k", self.h)            # tr(V^-1 G_r)
        self.trace_pg = self.trace_vg + torch.einsum("pq,kqp->k", self.phi, self.p_r)

    def score(self, reml: bool) -> Tensor:
        trace = self.trace_pg if reml else self.trace_vg
        return 0.5 * (self.rgr - trace)

    def observed_information(self, reml: bool) -> Tensor:
        return self.apa - (self.ie2 if reml else self.ktrace) / 2

    def expected_information(self) -> Tensor:
        return self.ie2 / 2

    def kenward_roger_covariance(self, w: Tensor) -> Tensor:
        phi = self.phi
        middle = torch.einsum("rs,rspq->pq", w, self.q_rs) - torch.einsum(
            "rs,rpa,ab,sbq->pq", w, self.p_r, phi, self.p_r)
        adjusted = phi + 2 * phi @ middle @ phi
        return (adjusted + adjusted.T) / 2


def _inverse(information: Tensor, what: str) -> Tensor:
    matrix = (information + information.T) / 2
    if not bool(torch.isfinite(matrix).all()):
        raise AnalysisError("singular_information", f"The {what} is not finite.")
    values = torch.linalg.eigvalsh(matrix)
    if float(values[0]) <= 1e-10 * max(float(values[-1]), 1e-300):
        raise AnalysisError("singular_information", f"The {what} of the variance parameters is "
                            "not positive definite (a variance at or near its boundary).")
    return torch.linalg.inv(matrix)


def kr_statistics(lmat: Tensor, phi: Tensor, p_r: Tensor, w: Tensor) -> dict[str, float]:
    """KR (1997) moments of a full-row-rank contrast L: A1, A2, df m and F scale lambda."""
    q = lmat.shape[0]
    theta = lmat.T @ torch.linalg.solve(lmat @ phi @ lmat.T, lmat)
    u = torch.einsum("ab,bc,rcd,de->rae", theta, phi, p_r, phi)
    traces = torch.einsum("raa->r", u)
    a1 = float(torch.einsum("rs,r,s->", w, traces, traces))
    a2 = float(torch.einsum("rs,rab,sba->", w, u, u))
    if not (math.isfinite(a1) and math.isfinite(a2)) or a2 <= 0:
        raise AnalysisError("invalid_degrees_of_freedom", "The Kenward-Roger moments are not "
                            "positive; the contrast carries no variance-parameter uncertainty.")
    b = (a1 + 6 * a2) / (2 * q)
    g = ((q + 1) * a1 - (q + 4) * a2) / ((q + 2) * a2)
    denominator = 3 * q + 2 * (1 - g)
    c1, c2, c3 = g / denominator, (q - g) / denominator, (q + 2 - g) / denominator
    v0, v1, v2 = 1 + c1 * b, 1 - c2 * b, 1 - c3 * b
    if abs(v0) < 1e-10:
        v0 = 0.0
    if v1 == 0 or v2 == 0:
        raise AnalysisError("invalid_degrees_of_freedom", "The Kenward-Roger variance "
                            "approximation is undefined for this contrast.")
    rho = (1 / q) * ((1 - a2 / q) / v1) ** 2 * v0 / v2
    if q * rho - 1 == 0:
        raise AnalysisError("invalid_degrees_of_freedom", "The Kenward-Roger df is infinite.")
    df = 4 + (q + 2) / (q * rho - 1)
    if not math.isfinite(df) or df <= 0:
        raise AnalysisError("invalid_degrees_of_freedom", "The Kenward-Roger denominator df is "
                            "not positive; the small-sample approximation is not usable here.")
    scale = 1.0 if abs(df - 2) < 1e-2 else df * (1 - a2 / q) / (df - 2)
    return {"A1": a1, "A2": a2, "B": b, "g": g, "rho": rho, "df": df, "scale": scale}


def satterthwaite_df(rows: Tensor, phi: Tensor, p_r: Tensor, w: Tensor) -> list[float]:
    """2 v^2 / (grad' W grad) for every row l, v = l'Phi l, grad_r = -l'Phi P_r Phi l."""
    a = rows @ phi                                                    # [q, p]
    variance = (a * rows).sum(1)
    gradient = -torch.einsum("qa,rab,qb->qr", a, p_r, a)
    denominator = torch.einsum("qr,rs,qs->q", gradient, w, gradient)
    if not bool((denominator > 0).all()) or not bool((variance > 0).all()):
        raise AnalysisError("invalid_degrees_of_freedom", "A contrast has no variance-parameter "
                            "uncertainty; its Satterthwaite df is undefined.")
    return (2 * variance.square() / denominator).tolist()


def combined_ddf(nu: list[float], tol: float = 1e-8) -> float:
    """lmerTest/pbkrtest get_Fstat_ddf: F denominator df from component t dfs."""
    if len(nu) == 1:
        return nu[0]
    if all(abs(b - a) < tol for a, b in zip(nu, nu[1:])):
        return sum(nu) / len(nu)
    if any(value <= 2 for value in nu):
        return 2.0
    expectation = sum(value / (value - 2) for value in nu)
    return 2 * expectation / (expectation - len(nu))


def _contrast_rows(contrast, terms: list[str]) -> tuple[list[str], Tensor]:
    if isinstance(contrast, dict):
        frame = pd.DataFrame([contrast], index=["contrast"])
    elif isinstance(contrast, (list, tuple)) and contrast and all(isinstance(row, dict)
                                                                 for row in contrast):
        frame = pd.DataFrame(list(contrast), index=[f"c{i + 1}" for i in range(len(contrast))])
    elif isinstance(contrast, pd.DataFrame):
        frame = contrast
    else:
        raise AnalysisError("invalid_contrast", "Give a dict of fixed-term weights, a list of "
                            "such dicts or a DataFrame with fixed-term columns.")
    if frame.columns.has_duplicates or any(not isinstance(c, str) or c not in terms
                                           for c in frame.columns):
        raise AnalysisError("invalid_contrast", "Contrast columns must be unique stored "
                            f"fixed-effect terms ({', '.join(terms)}).")
    if not 1 <= len(frame) <= len(terms) or frame.index.has_duplicates:
        raise AnalysisError("invalid_contrast", "Use 1..p uniquely labelled contrast rows.")
    rows = []
    for _, row in frame.iterrows():
        values = []
        for name in terms:
            value = row.get(name, 0.0)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = 0.0
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise AnalysisError("invalid_contrast", "Contrast weights must be finite numbers.")
            values.append(float(value))
        rows.append(values)
    matrix = torch.tensor(rows, dtype=torch.float64)
    singular = torch.linalg.svdvals(matrix)
    if float(singular[0]) <= 0 or float(singular[-1]) <= 1e-10 * float(singular[0]):
        raise AnalysisError("invalid_contrast", "Contrast rows must be nonzero and linearly "
                            "independent (full row rank).")
    return [str(label) for label in frame.index], matrix


def _nulls(null, count: int) -> Tensor:
    if null is None:
        return torch.zeros(count, dtype=torch.float64)
    if isinstance(null, Real) and not isinstance(null, bool) and math.isfinite(null):
        return torch.full((count,), float(null), dtype=torch.float64)
    if isinstance(null, (list, tuple)) and len(null) == count and all(
            isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v) for v in null):
        return torch.tensor([float(v) for v in null], dtype=torch.float64)
    raise AnalysisError("invalid_null", "null must be a finite number or one finite number per "
                        "contrast row.")


def _mixed_geometry(setup: _Setup) -> tuple[Tensor, list[str]]:
    """Derivative matrices G_r [k, S, m, m] in the saved reported-variance order."""
    top, count = setup.top_codes, setup.n_top
    position, m = padded_positions(top, count)
    if m > MAX_GROUP:
        raise AnalysisError("work_budget", f"Top-level groups are bounded to {MAX_GROUP} rows for "
                            "the dense small-sample covariance derivatives.")
    mask = _pad(torch.ones(len(top), dtype=torch.float64), top, position, count, m)
    low = _pad(setup.low_codes.to(torch.float64) + 1, top, position, count, m)
    same = ((low[:, :, None] == low[:, None, :]) & (low[:, :, None] > 0)).to(torch.float64)
    z = _pad(setup.z, top, position, count, m)                                  # [S, m, q]
    structure, q = setup.structure, setup.z.shape[1]

    def outer(a: int, b: int) -> Tensor:
        product = z[:, :, a, None] * z[:, None, :, b]
        if a != b:
            product = product + product.transpose(1, 2)
        return same * product

    blocks = []
    if setup.two_level:
        blocks.append(mask[:, :, None] * mask[:, None, :])
    if structure.kind == "identity":
        blocks.append(sum(outer(a, a) for a in range(q)))
    elif structure.kind == "exchangeable":
        blocks.append(sum(outer(a, a) for a in range(q)))
        blocks.append(sum(outer(a, b) for a in range(q) for b in range(a)))
    else:
        blocks.extend(outer(a, b) for a, b in structure.entries)
    blocks.append(torch.diag_embed(mask))
    return torch.stack(blocks), (top, position, count, m, mask)


def _check_budget(n_rows: int, groups: int, m: int, k: int, p: int, max_work: int) -> dict:
    plan = plan_workspace("saved mixed small-sample covariance derivatives", {
        "dense group covariance derivatives and solves": 8 * groups * m * m * (3 * k + 4),
        "group design solves": 8 * groups * m * (p + 2) * (2 * k + 2),
        "fixed-effect derivative blocks": 8 * k * k * p * p * 3,
        "row geometry": 64 * n_rows,
    }).record()
    work = groups * (m ** 3 * (2 * k + 2) + k * k * m * m + k * k * m * p * p)
    if work > max_work:
        raise AnalysisError("work_budget", f"The dense derivative plan needs {work:,} operations, "
                            f"more than max_work={max_work:,}.")
    return {"resource_plan": plan, "planned_work": work}


@resident_cpu
@torch.no_grad()
def mixed_contrast(result, *, data, contrast, null=None, ddf="kenward-roger", alpha=0.05,
                   max_work=2_000_000_000):
    """Kenward-Roger or Satterthwaite t/F tests of fixed-effect contrasts of a saved mixed fit.

    Kenward-Roger requires a REML fit; Satterthwaite accepts ML or REML. The saved fit is
    replayed at its estimates against the matched original data; nothing is refitted.
    """
    result = require_result(result, "mixed result")
    spec = result.spec
    if ddf not in DDF_METHODS:
        raise AnalysisError("invalid_option", "ddf must be 'kenward-roger' or 'satterthwaite'.")
    if spec.estimator != "mixed" or spec.covariance != "nonrobust" or spec.weights:
        raise AnalysisError("unsupported_spec", "Use an unweighted model-based (nonrobust) "
                            "mixed ML/REML fit; robust, cluster and weighted fits have no "
                            "small-sample df here.")
    method = result.extra.get("method")
    if method not in ("ml", "reml"):
        raise AnalysisError("invalid_result_state", "The saved likelihood criterion is missing.")
    if ddf == "kenward-roger" and method != "reml":
        raise AnalysisError("unsupported_spec", "Kenward-Roger is defined for REML fits; refit "
                            "with method='reml' or use ddf='satterthwaite'. No silent refit.")
    if (isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1
            or isinstance(max_work, bool) or not isinstance(max_work, int) or max_work < 1):
        raise AnalysisError("invalid_option", "alpha must lie in (0, 1) and max_work must be a "
                            "positive integer.")
    if result.nobs_original > MAX_ROWS or result.nobs > MAX_ROWS:
        raise AnalysisError("work_budget", f"Saved small-sample inference is bounded to "
                            f"{MAX_ROWS} rows.")
    original, positions = matched_frame(result, data)
    setup = _Setup(ModelFrame(spec, original))
    terms, p = setup.design.terms, setup.p
    saved = [c.term for c in result.coefficients]
    variance_terms = saved[p:]
    if saved[:p] != terms or len(variance_terms) != setup.n_variance + 1:
        raise AnalysisError("invalid_result_state", "Saved terms disagree with the replayed "
                            "fixed and variance geometry.")
    if (setup.reml != (method == "reml")):
        raise AnalysisError("invalid_result_state", "Saved criterion disagrees with the spec.")
    labels, rows = _contrast_rows(contrast, terms)
    nulls = _nulls(null, len(labels))
    derivatives, (top, position, count, m, mask) = _mixed_geometry(setup)
    k = derivatives.shape[0]
    budget = _check_budget(setup.frame.n, count, m, k, p, max_work)
    try:
        s = torch.tensor([c.estimate for c in result.coefficients[p:]], dtype=torch.float64)
        beta_saved = torch.tensor([c.estimate for c in result.coefficients[:p]],
                                  dtype=torch.float64)
        saved_cov = torch.tensor(result.covariance_matrix, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_result_state", "Saved estimates must be numeric.") from exc
    if not bool(torch.isfinite(s).all()) or saved_cov.shape != (len(saved), len(saved)):
        raise AnalysisError("invalid_result_state", "Saved variance parameters are incomplete.")
    x = _pad(setup.design.x, top, position, count, m)
    y = _pad(setup.y, top, position, count, m)
    model = LinearCovariance(x, y, mask, derivatives, s)
    if (not torch.allclose(model.beta, beta_saved, rtol=1e-6, atol=1e-8)
            or not torch.allclose(model.phi, saved_cov[:p, :p], rtol=1e-6, atol=1e-12)):
        raise AnalysisError("invalid_result_state", "Saved fixed estimates or covariance disagree "
                            "with the model replayed at the saved variance parameters.")
    reml = method == "reml"
    observed = model.observed_information(reml)
    observed_cov = _inverse(observed, "observed information")
    score = model.score(reml)
    stationarity = float(score @ observed_cov @ score)
    if not math.isfinite(stationarity) or stationarity > 1e-6:
        raise AnalysisError("invalid_result_state", "The saved variance parameters are not a "
                            "stationary point of the saved likelihood.")
    # The saved delta-method variance block (numerical Hessian in log-sd coordinates) must
    # agree with the analytic observed information in the linear coordinates.
    scale = observed_cov.diagonal().sqrt()
    if not torch.allclose(saved_cov[p:, p:] / scale[:, None] / scale, observed_cov / scale[:, None]
                          / scale, rtol=0, atol=1e-4):
        raise AnalysisError("invalid_result_state", "Saved variance-parameter covariance "
                            "disagrees with the observed information of the saved model.")
    phi = model.phi
    estimate = rows @ model.beta
    q = rows.shape[0]
    if ddf == "kenward-roger":
        w = _inverse(model.expected_information(), "expected REML information")
        adjusted = model.kenward_roger_covariance(w)
        if not bool(torch.isfinite(adjusted).all()) or float(torch.linalg.eigvalsh(adjusted)[0]) <= 0:
            raise AnalysisError("invalid_covariance", "The Kenward-Roger adjusted covariance is "
                                "not positive definite.")
        row_df = [kr_statistics(rows[i:i + 1], phi, model.p_r, w)["df"] for i in range(q)]
        joint = kr_statistics(rows, phi, model.p_r, w)
        used, information_kind = adjusted, "inverse expected REML information"
        variance_cov = w
    else:
        w = observed_cov
        row_df = satterthwaite_df(rows, phi, model.p_r, w)
        used, information_kind = phi, "inverse observed information"
        variance_cov = w
    contrast_cov = rows @ used @ rows.T
    se = contrast_cov.diagonal().sqrt()
    unadjusted_se = (rows @ phi @ rows.T).diagonal().sqrt()
    records = []
    for i, label in enumerate(labels):
        df_i = row_df[i]
        if not math.isfinite(df_i) or df_i <= 0:
            raise AnalysisError("invalid_degrees_of_freedom", "A contrast df is not positive.")
        statistic = float((estimate[i] - nulls[i]) / se[i])
        critical = distributions.t_isf(alpha / 2, df_i)
        records.append({"contrast": label, "estimate": float(estimate[i]),
                        "null": float(nulls[i]), "std_error": float(se[i]),
                        "unadjusted_std_error": float(unadjusted_se[i]), "t": statistic,
                        "df": df_i, "p_value": 2 * distributions.t_sf(abs(statistic), df_i),
                        "ci_low": float(estimate[i]) - critical * float(se[i]),
                        "ci_high": float(estimate[i]) + critical * float(se[i])})
    delta = estimate - nulls
    if ddf == "kenward-roger":
        unscaled = float(delta @ torch.linalg.solve(contrast_cov, delta)) / q
        statistic, ndf, denominator_df = joint["scale"] * unscaled, q, joint["df"]
        scale = joint["scale"]
        details = {key: joint[key] for key in ("A1", "A2", "B", "g", "rho")}
    else:
        values, vectors = torch.linalg.eigh(rows @ phi @ rows.T)
        order = torch.argsort(values, descending=True)
        values, vectors = values[order], vectors[:, order]
        rotated = vectors.T @ rows
        nu = satterthwaite_df(rotated, phi, model.p_r, w)
        statistic = float(((vectors.T @ delta).square() / values).sum()) / q
        unscaled, ndf, denominator_df, scale = statistic, q, combined_ddf(nu), 1.0
        details = {"component_df": nu}
    joint_row = {"statistic": statistic, "unscaled_statistic": unscaled, "scale": scale,
                 "numerator_df": ndf, "denominator_df": denominator_df,
                 "p_value": distributions.f_sf(statistic, ndf, denominator_df),
                 "distribution": "F"}
    title = ("Kenward-Roger" if ddf == "kenward-roger" else "Satterthwaite") + \
        " fixed-effect contrasts (saved mixed fit)"
    return saved_summary(TableSet(
        {"contrasts": table(records),
         "joint_test": table([joint_row]),
         "fixed_covariance": table(used.tolist(), columns=terms, index=terms),
         "variance_parameters": table({"term": variance_terms, "estimate": s.tolist(),
                                       "score": score.tolist()}),
         "variance_parameter_covariance": table(variance_cov.tolist(), columns=variance_terms,
                                                index=variance_terms)},
        title=title, procedure="mixed_contrast", ddf_method=ddf, criterion=method,
        source_result_id=result.id, source_data_hash=result.provenance["data_hash"],
        source_sample_positions=positions, fixed_terms=terms, contrast_labels=labels,
        contrast_matrix=rows.tolist(), alpha=float(alpha), refitted=False,
        variance_coordinates="linear: reported variances/covariances, V = sum_r s_r G_r",
        variance_information=information_kind, stationarity=stationarity,
        kr_details=details if ddf == "kenward-roger" else None,
        satterthwaite_details=details if ddf == "satterthwaite" else None,
        source=KR_SOURCE, **budget,
        assumptions="Gaussian independent top-level groups, correctly specified covariance, "
                    "interior variance estimates",
        exclusions=["robust/cluster covariance", "weights", "mixedflex", "mixedresid",
                    "nonlinear covariance coordinates (R_ij)"],
        precision="float64", device="cpu", stata_parity_validated=False,
    ))
