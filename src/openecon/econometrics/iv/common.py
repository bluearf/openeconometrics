"""Sample, design and estimation steps shared by the instrumental-variables estimators.

Conventions (Stata's ivregress; ivreg2 for the absorbed-effects estimator):

* **Blocks.** ``X1`` are the included exogenous regressors (``predictors``, with
  the constant first), ``X2`` the ``endogenous`` role and ``Z2`` the excluded
  ``instruments`` role. Collinear columns are omitted left to right, Stata
  style: first within X1, then endogenous regressors collinear with X1 or with
  earlier endogenous ones, then instruments collinear with X1 or with earlier
  instruments. Omitted regressors are recorded in
  ``provenance['omitted_terms']``, omitted instruments in
  ``extra['omitted_instruments']``. The order condition (``L2 >= q``) is
  checked after the screens.
* **Weights.** aweights and pweights are rescaled to sum to the number of rows
  (N = rows); fweights replicate observations (N = sum of weights). pweights
  need a robust, cluster or HAC covariance.
* **Covariance.** Every reported covariance goes through
  ``core.linear_covariance`` with the estimator's score regressors
  (``Xk = (I - kappa M_Z) X`` for 2SLS/LIML, ``Xg = Z S^-1 Z'WX`` for GMM), so
  factors are uniform: ``small`` selects ``RSS/(N-K)``, ``N/(N-K)`` and
  ``(N-1)/(N-K)``; ``G/(G-1)`` is applied to cluster covariances when
  ``group_factor`` is set. ``K`` counts absorbed degrees of freedom.
* **GMM.** The first step is 2SLS; the moment covariance ``S`` is formed from
  the previous step's residuals (``unadjusted``: ``s^2 Z'WZ``; ``robust``,
  ``cluster``, ``hac``: the matching sandwich meat of ``z_i w_i u_i``, optionally
  centered). When the reported covariance type equals the weight-matrix type
  the efficient form ``(X'Z S^-1 Z'X)^-1`` (times the small-sample factor) is
  reported; otherwise the sandwich evaluated at the final residuals. With a
  constant, regressors and instruments are centered for the GMM algebra (GMM is
  invariant to this reparameterization) so that ``S`` is well conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import Design, ModelFrame, kernel_call, linear_covariance
from openecon.econometrics.iv import kernels
from openecon.engines.linalg import weighted_crossprod

IGMM_TOLERANCE = 1e-10
IGMM_MAX_ITERATIONS = 1000
# A residual sum of squares below this fraction of sum w y^2 is rounding noise.
_EXACT_FIT = 1e-24


# ---- weights and sums of squares -----------------------------------------------------


def check_pweights(frame: ModelFrame) -> None:
    if frame.spec.weight_type == "pweight" and frame.spec.covariance == "nonrobust":
        raise AnalysisError(
            "unsupported_covariance",
            "pweights are sampling weights and need a robust covariance: choose "
            "covariance='robust' (Stata's vce(robust)) or give a cluster column.")


def estimation_weights(frame: ModelFrame) -> tuple[Tensor | None, int]:
    """Weights as the kernels receive them and N as Stata counts it."""
    weights = frame.weights()
    if weights is None:
        return None, frame.n
    if frame.spec.weight_type == "fweight":
        return weights, int(round(float(weights.sum())))
    return weights * (frame.n / float(weights.sum())), frame.n


def weighted_mean(values: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        return values.mean(dim=0)
    if values.ndim == 1:
        return (weights * values).sum() / weights.sum()
    return (weights[:, None] * values).sum(dim=0) / weights.sum()


def sum_of_squares(values: Tensor, weights: Tensor | None, *, centered: bool) -> float:
    if centered:
        values = values - weighted_mean(values, weights)
    squares = values.square()
    return float((squares * weights).sum() if weights is not None else squares.sum())


def check_fit(rss: float, tss: float | None, df_resid: float, scale: float, *,
              resolution: float = 0.0) -> None:
    """Reject the degenerate fits whose standard errors would be undefined.

    ``scale`` is the uncentered ``sum w y^2`` against which an exact fit is judged;
    ``resolution`` an additional residual sum of squares below which the fit cannot be
    told from an exact one (the error iterative demeaning leaves, see ``ivreghdfe``).
    """
    if df_resid <= 0:
        raise AnalysisError("insufficient_observations", "The model has no residual degrees of "
                            "freedom: it needs more observations than parameters (including "
                            "absorbed fixed effects).")
    if tss is not None and tss <= 0:
        raise AnalysisError("constant_outcome", "The outcome has no variation in the estimation "
                            "sample; nothing can be explained.")
    if rss <= max(_EXACT_FIT * scale, resolution):
        raise AnalysisError("perfect_fit", "The regressors fit the outcome exactly, so the error "
                            "variance and every standard error are zero. Remove the regressor "
                            "that reproduces the outcome.")


# ---- designs and collinearity screens ---------------------------------------------------


@dataclass
class Blocks:
    """Kept designs (raw columns) and the tensors the estimation runs on."""

    exog: Design
    endog: Design
    instr: Design
    x1: Tensor                 # raw, or partialled-out by the caller's transformation
    x2: Tensor
    z2: Tensor
    omitted_instruments: list[str] = field(default_factory=list)

    @property
    def terms(self) -> list[str]:
        return [*self.exog.terms, *self.endog.terms]


def role_designs(frame: ModelFrame, *, intercept: bool) -> tuple[Design, Design, Design]:
    """Designs of the exogenous, endogenous and excluded-instrument columns."""
    spec = frame.spec
    endogenous, instruments = frame.role("endogenous"), frame.role("instruments")
    exogenous = set(spec.predictors)
    if spec.outcome in endogenous or spec.outcome in instruments:
        raise AnalysisError("invalid_spec", "The outcome cannot also be an endogenous regressor "
                            "or an instrument.")
    both = [name for name in endogenous if name in exogenous]
    if both:
        raise AnalysisError("invalid_spec", f"{', '.join(both)} is listed both as exogenous (x) "
                            "and as endogenous; keep it in one list.")
    both = [name for name in instruments if name in exogenous]
    if both:
        raise AnalysisError("invalid_spec", f"{', '.join(both)} is already an exogenous "
                            "regressor, which is an instrument for itself; list only the "
                            "EXCLUDED instruments in instruments.")
    both = [name for name in instruments if name in set(endogenous)]
    if both:
        raise AnalysisError("invalid_spec", f"{', '.join(both)} cannot be both endogenous and "
                            "its own instrument; a regressor you trust as exogenous belongs in x.")
    stray = [name for name in spec.categorical if name not in exogenous]
    if stray:
        raise AnalysisError("invalid_spec", f"categorical applies to exogenous regressors (x) "
                            f"only; {', '.join(stray)} must be numeric.")
    exog = frame.design(intercept=intercept)
    endog = frame.design(endogenous, intercept=False, categorical=[])
    instr = frame.design(instruments, intercept=False, categorical=[])
    return exog, endog, instr


def _join(first: Design, second: Design, first_block: Tensor | None,
          second_block: Tensor | None) -> tuple[Design, Tensor | None]:
    terms = [*first.terms, *second.terms]
    if len(terms) != len(set(terms)):
        raise AnalysisError("duplicate_terms", "Column names collide with generated design "
                            "terms. Rename the affected columns.")
    design = Design(torch.cat([first.x, second.x], dim=1), terms, first.categories,
                    first.intercept)
    block = None if first_block is None else torch.cat([first_block, second_block], dim=1)
    return design, block


def screen(frame: ModelFrame, exog: Design, endog: Design, instr: Design,
           weights: Tensor | None, *,
           transformed: tuple[Tensor, Tensor, Tensor] | None = None) -> Blocks:
    """Stata-style collinearity screens of the three blocks and the order condition.

    ``transformed`` holds the blocks after partialling out absorbed effects;
    columns absorbed by those effects are then omitted as well
    (``ModelFrame.drop_absorbed``) and the returned ``x1, x2, z2`` are the
    transformed ones.
    """
    def reduce(design: Design, block: Tensor | None) -> tuple[Design, Tensor]:
        if transformed is None:
            design = frame.drop_collinear(design, weights)
            return design, design.x
        return frame.drop_absorbed(design, block, weights)

    rows, columns = exog.x.shape[0], len(exog.terms) + len(endog.terms)

    def too_few_rows() -> None:
        # With no more rows than coefficients every column is "collinear": say what is wrong.
        if rows <= columns:
            raise AnalysisError(
                "insufficient_observations", f"Only {rows} observation(s) are available for "
                f"{columns} coefficient(s): the model needs more observations than regressors "
                "and at least as many instruments as regressors.")

    t1, t2, t3 = transformed if transformed is not None else (None, None, None)
    exog, x1 = reduce(exog, t1)
    k1 = len(exog.terms)
    joint, block = reduce(*_join(exog, endog, x1 if transformed is not None else None, t2))
    kept = set(joint.terms[k1:])
    endog_kept = [i for i, term in enumerate(endog.terms) if term in kept]
    if not endog_kept:
        too_few_rows()
        raise AnalysisError("no_endogenous_regressors", "Every endogenous regressor was omitted "
                            "because of collinearity" + (
                                " with the fixed effects or the other regressors"
                                if transformed is not None else "") + "; nothing is left to "
                            "instrument. Use an ordinary regression, or revise the endogenous "
                            "list.")
    x2 = block[:, k1:]
    marks = len(frame.warnings), len(frame.notes.get("omitted_terms", []))
    joint, block = reduce(*_join(exog, instr, x1 if transformed is not None else None, t3))
    omitted = frame.notes.get("omitted_terms", [])[marks[1]:]
    if omitted:
        # Instruments are not model terms: keep them out of provenance['omitted_terms'].
        del frame.notes["omitted_terms"][marks[1]:]
        del frame.warnings[marks[0]:]
        frame.warn(f"Instrument(s) omitted because of collinearity with the exogenous "
                   f"regressors or earlier instruments: {', '.join(omitted)}.")
    kept = set(joint.terms[k1:])
    instr_kept = [i for i, term in enumerate(instr.terms) if term in kept]
    endog, instr = endog.select(endog_kept), instr.select(instr_kept)
    if len(instr.terms) < len(endog.terms):
        too_few_rows()
        raise AnalysisError(
            "underidentified",
            f"The order condition fails: {len(endog.terms)} endogenous regressor(s) but only "
            f"{len(instr.terms)} usable excluded instrument(s) after the collinearity screen. "
            "Add instruments or treat fewer regressors as endogenous.")
    return Blocks(exog, endog, instr, x1, x2, block[:, k1:], list(omitted))


# ---- covariance ----------------------------------------------------------------------------


@dataclass
class Setting:
    """How one fit counts observations and parameters and which covariance it reports."""

    nobs: int                               # N as the estimator counts it
    weights: Tensor | None
    kind: str                               # nonrobust, robust, cluster or hac
    small: bool                             # t/F statistics and the regress-style factors
    group_factor: bool = True               # G/(G-1) on cluster covariances
    absorbed: int = 0                       # degrees of freedom of partialled-out effects
    clusters: list[tuple[Tensor, int]] | None = None
    cluster_names: list[str] | None = None

    def df_resid(self, k: int) -> int:
        return self.nobs - k - self.absorbed


def covariance(frame: ModelFrame, setting: Setting, *, score_x: Tensor, resid: Tensor,
               bread: Tensor, k: int, rss: float | None = None, kind: str | None = None,
               regress: bool = False) -> tuple[Tensor, dict[str, Any]]:
    """``core.linear_covariance`` under the fit's conventions (``k`` slopes + constant).

    ``regress=True`` forces the factors of Stata's regress (``N/(N-K)``,
    ``G/(G-1) (N-1)/(N-K)``) whatever ``small`` says: auxiliary first-stage and
    endogeneity regressions are reported the way ``regress`` reports them.
    """
    return linear_covariance(
        frame, x=score_x, resid=resid, bread=bread, n=setting.nobs, k=k + setting.absorbed,
        df_resid=setting.df_resid(k), weights=setting.weights, score_x=score_x,
        small=regress or setting.small, group_factor=regress or setting.group_factor, ssr=rss,
        kind=kind or setting.kind, clusters=setting.clusters, cluster_names=setting.cluster_names)


def moment_meat(frame: ModelFrame, setting: Setting, rows: Tensor,
                kind: str | None = None) -> Tensor:
    """Sum-form covariance of the moment rows ``m_i`` (weights applied here), no factors.

    ``robust``: ``sum_i w_i^2 m_i m_i'`` (``sum_i f_i m_i m_i'`` under frequency
    weights); ``cluster``: outer products of the cluster sums (inclusion-
    exclusion for two columns); ``hac``: the kernel-weighted autocovariances.
    """
    width = rows.shape[1]
    ones = torch.ones(rows.shape[0], dtype=torch.float64)
    meat, _ = linear_covariance(
        frame, x=rows, resid=ones, bread=torch.eye(width, dtype=torch.float64), n=setting.nobs,
        k=0, df_resid=setting.nobs, weights=setting.weights, score_x=rows, small=False,
        group_factor=False, kind=kind or setting.kind, clusters=setting.clusters,
        cluster_names=setting.cluster_names)
    return meat


# ---- estimation ----------------------------------------------------------------------------


@dataclass
class Estimate:
    method: str
    beta: Tensor               # [K], order [X1, X2]
    covariance: Tensor
    info: dict[str, Any]
    resid: Tensor              # y - X beta
    rss: float
    proj: kernels.Projections
    tsls: kernels.KClass       # the 2SLS fit (first GMM step; base of the diagnostics)
    kappa: float | None = None
    gmm: dict[str, Any] | None = None
    condition_number: float | None = None


def _residual_ss(resid: Tensor, weights: Tensor | None) -> float:
    return float(resid.square().sum() if weights is None else (weights * resid.square()).sum())


def _report(frame: ModelFrame, setting: Setting, x1: Tensor, proj: kernels.Projections,
            fit: kernels.KClass) -> tuple[Tensor, dict[str, Any]]:
    k = fit.beta.numel()
    if setting.kind == "nonrobust":
        empty = torch.empty((fit.resid.numel(), 0), dtype=torch.float64)
        return covariance(frame, setting, score_x=empty, resid=fit.resid, bread=fit.bread, k=k,
                          rss=fit.rss)
    score_x = kernels.score_regressors(x1, proj, fit.kappa)
    return covariance(frame, setting, score_x=score_x, resid=fit.resid, bread=fit.bread, k=k)


def estimate(frame: ModelFrame, setting: Setting, y: Tensor, blocks: Blocks, *, method: str,
             wmatrix: str = "robust", igmm: bool = False, center: bool = False) -> Estimate:
    """2SLS, LIML or GMM on the (possibly transformed) blocks, with the reported covariance."""
    w = setting.weights
    proj = kernel_call(kernels.project, y, blocks.x1, blocks.x2, blocks.z2, w)
    tsls = kernel_call(kernels.k_class, proj, w, 1.0)
    k = tsls.beta.numel()
    if setting.df_resid(k) <= 0:
        raise AnalysisError("insufficient_observations", "The model has no residual degrees of "
                            "freedom: it needs more observations than parameters (including "
                            "absorbed fixed effects).")
    if method == "gmm" and wmatrix != "unadjusted":
        return _gmm(frame, setting, y, blocks, proj, tsls, wmatrix=wmatrix, igmm=igmm,
                    center=center)
    fit, kappa = tsls, None
    if method in ("liml", "fuller") and proj.l2 > proj.q \
            and setting.nobs - setting.absorbed - proj.k1 - proj.l2 < proj.q + 1:
        raise AnalysisError(
            "insufficient_observations", "LIML needs more than L + q observations (L "
            "instruments including the exogenous regressors, q endogenous regressors) to form "
            "its eigenvalue problem; use method='2sls' or fewer instruments.")
    if method in ("liml", "fuller"):
        # Exact identification: P_Z spans the same q directions beyond X1 as the reduced
        # form of [y X2], so kappa = 1 identically and LIML is 2SLS. The eigenproblem is
        # skipped (it is undefined when the instruments reproduce a regressor exactly).
        kappa = 1.0 if proj.l2 == proj.q else kernel_call(kernels.liml_kappa, proj, w)
        kappa = 1.0 if abs(kappa - 1.0) < 1e-12 else kappa
        if method == "fuller":
            correction = frame.spec.options.get("fuller_alpha", 1.0)
            if not correction > 0:
                raise AnalysisError("invalid_option", "fuller_alpha must be positive.")
            kappa -= correction / (setting.nobs - setting.absorbed - proj.k1 - proj.l2)
        fit = tsls if kappa == 1.0 else kernel_call(kernels.k_class, proj, w, kappa)
    elif method == "kclass":
        kappa = frame.spec.options.get("kappa")
        if kappa is None or not 0 <= kappa <= 1:
            raise AnalysisError("invalid_option", "kclass requires explicit kappa in [0,1].")
        fit = tsls if kappa == 1 else kernel_call(kernels.k_class, proj, w, kappa)
    if "kappa" in frame.spec.options and method != "kclass":
        raise AnalysisError("invalid_option", "kappa applies only to method='kclass'.")
    if "fuller_alpha" in frame.spec.options and method != "fuller":
        raise AnalysisError("invalid_option", "fuller_alpha applies only to method='fuller'.")
    v, info = _report(frame, setting, blocks.x1, proj, fit)
    gmm = None
    if method == "gmm":
        # s^2 Z'WZ weighting reproduces 2SLS; J is then the Sargan form N u'P_Z u / u'u.
        gmm = {"wmatrix": "unadjusted", "iterations": 1, "converged": True, "igmm": igmm,
               "centered": False, "j": _projected_ss(blocks, proj, fit.resid, w)
               / (fit.rss / setting.nobs)}
    return Estimate(method, fit.beta, v, info, fit.resid, fit.rss, proj, tsls, kappa, gmm,
                    fit.condition_number)


def _projected_ss(blocks: Blocks, proj: kernels.Projections, resid: Tensor,
                  weights: Tensor | None) -> float:
    """``u'W P_Z u = g'(Z'WZ)^-1 g`` with ``g = Z'Wu``."""
    z = torch.cat([blocks.x1, blocks.z2], dim=1)
    moment = weighted_crossprod(z, weights, resid)
    return float(moment @ (proj.first.xtx_inv @ moment))


def gmm_system(blocks: Blocks, weights: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
    """``X = [X1 X2]`` and ``Z = [X1 Z2]`` for the GMM algebra, centered under a constant.

    Returns ``(x, z, shift)``; ``shift`` holds the regressor means removed (zero
    for the constant) or ``None`` without a constant. Centering changes only
    the constant's coefficient and leaves the residuals and J untouched.
    """
    x = torch.cat([blocks.x1, blocks.x2], dim=1)
    z = torch.cat([blocks.x1, blocks.z2], dim=1)
    if not blocks.exog.intercept:
        return x, z, None
    shift = weighted_mean(x, weights)
    shift[0] = 0.0
    offset = weighted_mean(z, weights)
    offset[0] = 0.0
    return x - shift, z - offset, shift


def moment_covariance(frame: ModelFrame, setting: Setting, z: Tensor, resid: Tensor, *,
                      wmatrix: str, center: bool = False) -> Tensor:
    """Sum-form covariance ``S`` of the moments ``z_i w_i u_i`` for a GMM weight matrix."""
    w = setting.weights
    if wmatrix == "unadjusted":
        return weighted_crossprod(z, w) * (_residual_ss(resid, w) / setting.nobs)
    rows = z * resid[:, None]
    if center:
        rows = rows - weighted_mean(rows, w)
    return moment_meat(frame, setting, rows, kind=wmatrix)


def _gmm(frame: ModelFrame, setting: Setting, y: Tensor, blocks: Blocks,
         proj: kernels.Projections, tsls: kernels.KClass, *, wmatrix: str, igmm: bool,
         center: bool) -> Estimate:
    w, n = setting.weights, setting.nobs
    x, z, shift = gmm_system(blocks, w)
    k = x.shape[1]
    zx, zy = weighted_crossprod(z, w, x), weighted_crossprod(z, w, y)

    def moment_cov(resid: Tensor) -> Tensor:
        return moment_covariance(frame, setting, z, resid, wmatrix=wmatrix, center=center)

    weighting_resid, beta, iterations, converged = tsls.resid, None, 0, not igmm
    while True:
        try:
            step = kernel_call(kernels.gmm_step, zx, zy, moment_cov(weighting_resid))
        except AnalysisError as exc:
            if exc.code != "singular_weight_matrix":
                raise
            raise AnalysisError(
                "singular_weight_matrix", "The GMM moment covariance is singular (for a cluster "
                "weight matrix: fewer clusters than instruments). Use fewer instruments, a "
                "robust weight matrix, or method='2sls'.") from exc
        iterations += 1
        resid = y - x @ step.beta
        if beta is not None and \
                float(((step.beta - beta).abs() / (beta.abs() + 1)).max()) < IGMM_TOLERANCE:
            converged = True
        beta = step.beta
        if not igmm or converged:
            break
        if iterations >= IGMM_MAX_ITERATIONS:
            raise AnalysisError("nonconvergence", f"Iterated GMM did not converge in "
                                f"{IGMM_MAX_ITERATIONS} iterations; use the two-step estimator "
                                "(igmm=False) or review the instruments.")
        weighting_resid = resid
    rss = _residual_ss(resid, w)
    score_x = z @ step.loading
    if setting.kind == wmatrix:
        # Efficient GMM: (A'S^-1 A)^-1 times the small-sample factor of this covariance type.
        _, info = covariance(frame, setting, score_x=score_x, resid=weighting_resid,
                             bread=step.bread, k=k)
        v = step.bread * info.get("small_sample_correction", 1.0)
        info["correction"] = "efficient GMM (X'Z S^-1 Z'X)^-1; " + str(info["correction"])
    elif setting.kind == "nonrobust":
        sigma2 = rss / (setting.df_resid(k) if setting.small else n)
        middle = step.loading.T @ weighted_crossprod(z, w) @ step.loading
        v = sigma2 * (step.bread @ middle @ step.bread)
        info = {"covariance": "nonrobust", "df_inference": setting.df_resid(k),
                "correction": "GMM sandwich with unadjusted moment covariance "
                              f"SSR/{'(N-K)' if setting.small else 'N'} Z'WZ"}
    else:
        v, info = covariance(frame, setting, score_x=score_x, resid=resid, bread=step.bread, k=k)
        info["correction"] = "GMM sandwich; " + str(info["correction"])
    if shift is not None:
        # b0 = b0~ - mean' b~ and V = T V~ T' with T the identity except its first row.
        transform = torch.eye(k, dtype=torch.float64)
        transform[0, 1:] = -shift[1:]
        beta, v = transform @ beta, transform @ v @ transform.T
    gmm = {"wmatrix": wmatrix, "iterations": iterations, "converged": converged, "igmm": igmm,
           "centered": center, "j": step.j}
    return Estimate("gmm", beta, v, info, resid, rss, proj, tsls, None, gmm, None)
