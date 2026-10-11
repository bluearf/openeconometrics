"""Jackknife instrumental-variables estimators: JIVE1, JIVE2 and UJIVE.

Notation as in ``kernels``: ``X1`` [n, K1] included exogenous regressors (the
constant first), ``X2`` [n, q] endogenous regressors, ``Z2`` [n, L2] excluded
instruments, ``Q = [X1 Z2]`` all instruments, ``X = [X1 X2]``. ``H_A`` is the
projection on the columns of ``A`` and ``h_i = [H_Q]_ii`` its diagonal (the
squared row norms of the thin QR factor of ``Q``; no n-by-n matrix is formed).

JIVE1 and JIVE2 (Angrist, Imbens and Krueger 1999, J. Applied Econometrics
14:57-67, equations for JIVE1/JIVE2) replace the first-stage fit of EVERY
column of ``X`` (exogenous columns included) by

    JIVE1   xhat_i = (q_i'pi - h_i x_i) / (1 - h_i)      leave-one-out prediction
    JIVE2   xhat_i = (q_i'pi - h_i x_i) / (1 - 1/N)

with ``pi = (Q'Q)^-1 Q'X``. For an exogenous column JIVE1 returns the column
itself (it is in ``Q``); JIVE2 returns ``(1 - h_i) x_i / (1 - 1/N)``. The
estimate is ``b = (Xhat'X)^-1 Xhat'y`` (exactly identified IV with instrument
matrix ``Xhat``). JIVE1 is algebraically the same coefficient on ``X2`` as the
``jive1`` entry of Kolesar's ManyIV ``ujive.fit`` (partialling ``X1`` out of the
leave-one-out instrument), https://github.com/kolesarm/ManyIV/blob/master/R/ujive.R.

UJIVE (Kolesar 2013, "Estimation in an instrumental variables model with
treatment effect heterogeneity"; the ``old ujive``/``UJIVEold`` entry of ManyIV
``R/ujive.R`` and ``tests/testthat/test_ujive.R``), for ONE endogenous regressor
``d``:

    dhat = (I - D_Q)^-1 (H_Q - D_Q) d - (I - D_1)^-1 (H_1 - D_1) d
         = leave-one-out prediction of d from Q minus that from X1,
    b    = dhat'y / dhat'd.

Only the endogenous coefficient is reported; ``X1`` is partialled out and the
residual is ``e = M_1 (y - d b)`` (ManyIV's convention).

Covariance (ManyIV's "textbook" IV sandwich with the jackknife instrument as
the instrument; many-instrument and heterogeneous-effect corrections are not
implemented): with ``A = Xhat`` (or ``dhat``) and ``B = (A'X)^-1``

    nonrobust   s^2 B (A'A) B'                s^2 = RSS/N   (RSS/(N-K) with small)
    robust      B (sum_i u_i^2 a_i a_i') B'   times N/(N-K) with small
    cluster     B (sum_g t_g t_g') B' G/(G-1) times (N-1)/(N-K) with small

where K counts every coefficient of the model (K1 + q, also for UJIVE).
Leave-one-out is by observation, also with a cluster covariance: under
within-cluster dependence of the first-stage errors the jackknife does not
remove the own-cluster bias (leave-one-cluster-out JIVE is not implemented).
JIVE1 and UJIVE need every leverage strictly below one (``unit_leverage``);
observations are never dropped silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, linear_covariance, make_spec, wald_test,
)
from openecon.econometrics.iv.common import role_designs, screen, sum_of_squares
from openecon.models import ModelSpec, ResultBundle

TITLES = {"jive1": "Jackknife IV (JIVE1) regression", "jive2": "Jackknife IV (JIVE2) regression",
          "ujive": "Unbiased jackknife IV (UJIVE) regression"}
# A leverage within this distance of one makes the leave-one-out prediction undefined.
LEVERAGE_TOLERANCE = 1e-8


def _thin_q(block: Tensor, what: str) -> Tensor:
    """Orthonormal basis of a full-rank block (already screened for collinearity)."""
    scale = torch.linalg.vector_norm(block, dim=0)
    if bool((scale <= 0).any()):
        raise AnalysisError("singular_design", f"The {what} contain a zero column.")
    basis, triangle = torch.linalg.qr(block / scale, mode="reduced")
    if bool((triangle.diagonal().abs() <= 1e-10).any()):
        raise AnalysisError("singular_design", f"The {what} are numerically rank deficient.")
    return basis


def _leverage(basis: Tensor, method: str, what: str) -> Tensor:
    lev = basis.square().sum(dim=1)
    if method != "jive2" and float(lev.max()) >= 1 - LEVERAGE_TOLERANCE:
        count = int((lev >= 1 - LEVERAGE_TOLERANCE).sum())
        raise AnalysisError(
            "unit_leverage", f"{count} observation(s) have leverage one in the {what} (for "
            "example the only observation of an instrument category), so their leave-one-out "
            "prediction is undefined. Remove or merge those instrument levels; observations "
            "are not dropped automatically.")
    return lev


def _loo(basis: Tensor, lev: Tensor, target: Tensor) -> Tensor:
    """Leave-one-out fitted values (I - D)^-1 (H - D) target, column-wise."""
    fitted = basis @ (basis.T @ target)
    return (fitted - lev[:, None] * target) / (1 - lev)[:, None]


def fit_jive(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``jive``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    method, small = frame.option("method"), bool(frame.option("small"))
    blocks = screen(frame, *role_designs(frame, intercept=spec.intercept), None)
    x1, x2, z2 = blocks.x1, blocks.x2, blocks.z2
    n, k1, q, l2 = x1.shape[0], x1.shape[1], x2.shape[1], z2.shape[1]
    if method == "ujive" and q != 1:
        raise AnalysisError("unsupported_domain", "UJIVE is defined for one endogenous "
                            "regressor (Kolesar 2013); use method='jive1' or 'jive2'.")
    if method == "ujive" and k1 == 0:
        raise AnalysisError("invalid_spec", "UJIVE subtracts the leave-one-out prediction from "
                            "the exogenous regressors; keep the constant or give x.")
    k = k1 + q
    if n - k <= 0 or n <= k1 + l2:
        raise AnalysisError("insufficient_observations", "JIVE needs more observations than "
                            "instruments and coefficients.")
    frame.workspace_plan("jackknife IV leverage and leave-one-out fits", {
        "instrument_basis": n * (k1 + l2) * 8 * 3, "fitted_blocks": n * (k + 2) * 8 * 4})
    y = frame.numeric(spec.outcome)
    instruments = torch.cat([x1, z2], dim=1)
    basis = _thin_q(instruments, "instruments [X1 Z2]")
    lev = _leverage(basis, method, "instruments")
    x = torch.cat([x1, x2], dim=1)
    if method == "ujive":
        basis1 = _thin_q(x1, "exogenous regressors")
        lev1 = _leverage(basis1, method, "exogenous regressors")
        a = _loo(basis, lev, x2) - _loo(basis1, lev1, x2)          # [n, 1]
        cross = float(a[:, 0] @ x2[:, 0])
        if not abs(cross) > 1e-12 * float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(x2)):
            raise AnalysisError("underidentified", "The UJIVE instrument is orthogonal to the "
                                "endogenous regressor; the instruments carry no information.")
        beta = (a[:, 0] @ y / cross).reshape(1)
        structural = y - x2[:, 0] * beta[0]
        resid = structural - basis1 @ (basis1.T @ structural)
        terms = list(blocks.endog.terms)
        bread = torch.tensor([[1.0 / cross]], dtype=torch.float64)
    else:
        fitted = basis @ (basis.T @ x)
        if method == "jive1":
            a = torch.cat([x1, (fitted[:, k1:] - lev[:, None] * x2) / (1 - lev)[:, None]], 1)
        else:
            a = (fitted - lev[:, None] * x) / (1 - 1 / n)
        cross = a.T @ x
        try:
            bread = torch.linalg.inv(cross)
        except RuntimeError as exc:
            raise AnalysisError("underidentified", "The jackknife instrument matrix Xhat'X is "
                                "singular; the instruments do not identify the model.") from exc
        spread = torch.outer(torch.linalg.vector_norm(a, dim=0), torch.linalg.vector_norm(x, dim=0))
        condition = float(torch.linalg.cond(cross / spread))
        if not condition < 1e14:
            raise AnalysisError("underidentified", "The jackknife instrument matrix Xhat'X is "
                                "numerically singular; the instruments do not identify the model.")
        beta = bread @ (a.T @ y)
        resid = y - x @ beta
        terms = list(blocks.terms)
    rss = float(resid.square().sum())
    df_resid = n - k
    constant = int(blocks.exog.intercept)
    tss = sum_of_squares(y, None, centered=bool(constant))
    if tss <= 0:
        raise AnalysisError("constant_outcome", "The outcome has no variation in the estimation "
                            "sample.")
    if rss <= 1e-24 * sum_of_squares(y, None, centered=False):
        raise AnalysisError("perfect_fit", "The regressors fit the outcome exactly; standard "
                            "errors are undefined.")
    kind = spec.covariance
    if kind == "nonrobust":
        s2 = rss / (df_resid if small else n)
        covariance = s2 * (bread @ (a.T @ a) @ bread.T)
        covariance = (covariance + covariance.T) / 2
        info: dict[str, Any] = {"covariance": "nonrobust", "df_inference": df_resid,
                                "correction": f"IV with jackknife instrument: SSR/{'(N-K)' if small else 'N'}"}
    else:
        clusters = frame.cluster_dimensions() if kind == "cluster" else None
        covariance, info = linear_covariance(
            frame, x=a, resid=resid, bread=bread, n=n, k=k, df_resid=df_resid, score_x=a,
            small=small, group_factor=True, kind=kind, clusters=clusters,
            cluster_names=registry.cluster_columns(spec) or None)
    df_inference = info.get("df_inference") if small else None
    r_squared = 1 - rss / tss
    leverage = {"min": float(lev.min()), "max": float(lev.max()), "mean": float(lev.mean())}
    metrics: dict[str, Any] = {
        "r_squared": r_squared, "rmse": (rss / (df_resid if small else n)) ** 0.5,
        "df_model": k - constant, "df_resid": df_resid, "n_instruments": l2,
        "n_endogenous": q, "max_leverage": leverage["max"],
    }
    slopes = [i for i, term in enumerate(terms) if term != "Intercept"]
    tests = {"model": wald_test(beta, covariance, slopes, df_resid=df_inference,
                                label="Model F test (slopes)" if small else "Wald chi2 test of the slopes")}
    info.update({
        "df_inference": df_inference, "nobs": n, "method": method, "small": small,
        "error_variance": "RSS/(N-K)" if small else "RSS/N",
        "residual_definition": ("M_X1 (y - d b): exogenous regressors partialled out (ManyIV)"
                                if method == "ujive" else "y - X b with the observed regressors"),
        "assumptions": "fixed number of instruments, independent observations (clusters); "
                       "no many-instrument or heterogeneous-effect variance correction",
    })
    return build_result(
        frame, terms=terms, params=beta, covariance=covariance, title=TITLES[method],
        use_t=small, df_inference=df_inference, df_resid=df_resid, metrics=metrics,
        fitted=y - resid if method != "ujive" else None, nobs=n, inference=info, tests=tests,
        extra={"method": method, "exogenous": blocks.exog.terms,
               "endogenous": blocks.endog.terms, "instruments": blocks.instr.terms,
               "omitted_instruments": blocks.omitted_instruments, "leverage": leverage,
               "partialled_out": blocks.exog.terms if method == "ujive" else [],
               "leave_out_unit": "observation",
               "source": {"jive1": "Angrist, Imbens and Krueger (1999) JIVE1",
                          "jive2": "Angrist, Imbens and Krueger (1999) JIVE2",
                          "ujive": "Kolesar (2013) UJIVE; ManyIV R/ujive.R 'old ujive'"}[method]},
        categories=blocks.exog.categories, solver="householder_qr_leverage_jackknife",
        solver_diagnostics={"instrument_leverage": leverage})


def jive(*, data: Any, y: str, endog: Sequence[str], instruments: Sequence[str],
         x: Sequence[str] | None = None, method: str = "jive1", covariance: str | None = None,
         cluster: str | None = None, small: bool = False,
         categorical: Sequence[str] | None = None, intercept: bool = True,
         missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Jackknife IV regression: JIVE1, JIVE2 (Angrist-Imbens-Krueger) or UJIVE (Kolesar).

    Model
        ``y = x'b1 + endog'b2 + u`` with excluded ``instruments``. The first-stage
        fit is replaced by a leave-one-out prediction so that an observation's own
        first-stage error does not enter its instrument:

        - ``method='jive1'``: ``xhat_i = (q_i'pi - h_i x_i)/(1 - h_i)`` for every
          column of ``[x endog]`` (exogenous columns are reproduced), ``h_i`` the
          leverage of the instrument matrix ``[1 x instruments]``;
          ``b = (Xhat'X)^-1 Xhat'y``.
        - ``method='jive2'``: the same with ``1 - 1/N`` in the denominator.
        - ``method='ujive'`` (one endogenous regressor): instrument
          ``dhat`` = leave-one-out prediction of ``endog`` from all instruments minus
          that from ``[1 x]``; ``b = dhat'y / dhat'endog``. Only the endogenous
          coefficient is reported (exogenous regressors partialled out).

    Parameters
        covariance: ``'robust'`` (default; ``'cluster'`` when ``cluster`` is given)
            or ``'nonrobust'``: the IV sandwich with the jackknife instrument.
        small: t and F with N-K degrees of freedom, RSS/(N-K), N/(N-K) on robust
            and (N-1)/(N-K) G/(G-1) on cluster covariances.
        categorical: exogenous regressors expanded as dummies.

    Errors
        ``unit_leverage`` (an observation with leverage one: JIVE1/UJIVE are
        undefined; nothing is dropped automatically), ``underidentified``,
        ``unsupported_domain`` (UJIVE with several endogenous regressors),
        ``insufficient_observations``, ``perfect_fit``, ``invalid_spec``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0); n = 400
        >>> z = rng.normal(size=(n, 6)); v = rng.normal(size=n)
        >>> df = pd.DataFrame(z, columns=[f"z{j}" for j in range(6)])
        >>> df["d"] = z @ np.full(6, 0.2) + v
        >>> df["y"] = 1 + 0.5 * df.d + 0.8 * v + rng.normal(size=n)
        >>> oe.jive(data=df, y="y", endog=["d"], instruments=[f"z{j}" for j in range(6)])
    """
    from openecon.analysis import fit

    if covariance is None:
        covariance = "cluster" if cluster is not None else "robust"
    spec = make_spec(
        "jive", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=covariance, cluster=cluster, missing=missing, alpha=alpha,
        columns={"endogenous": column_list(endog, "endog"),
                 "instruments": column_list(instruments, "instruments")},
        options={"method": method, "small": True if small else None})
    return fit(spec, data=data)
