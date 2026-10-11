"""Lewbel (2012) heteroskedasticity-based generated instruments: Stata's ``ivreg2h``.

Model (Lewbel 2012, J. Business & Economic Statistics 30:67-80):

    y  = X1'b1 + X2'b2 + e1,        X2_m = X1'g_m + e2_m   (m = 1..q)
    E[X1 e1] = E[X1 e2] = 0,  cov(Z, e1 e2_m) = 0,  cov(Z, e2_m^2) != 0,

with ``Z`` a subset of the non-constant exogenous regressors (``generators``,
default: all of them). The generated instruments are

    G_{m,j} = (Z_j - mean(Z_j)) * e2hat_m,   e2hat_m = M_X1 X2_m  (OLS on all of X1),

the means and first-stage regressions taken over the estimation sample. They are
added to any external excluded ``instruments`` and the model is estimated by 2SLS
or two-step efficient GMM. Generated-instrument names follow ivreg2h:
``<endog>_<generator>_g``. With ``generators`` given, the first stage still uses all
of X1 (ivreg2h's generated-only branch; its generated-plus-external branch regresses
on the z() list instead, which agrees with this one when z() is all of X1:
http://fmwww.bc.edu/repec/bocode/i/ivreg2h.ado lines 443 and 667).

Covariance (``first_step``):

* ``'ignored'`` (ivreg2h/ivreg2): the generated instruments are treated as data and
  every number, including the diagnostics, is that of ``ivregress`` on the augmented
  instrument list (ivregress's finite-sample conventions; see ``docs/econometrics/iv.md``).
* ``'corrected'`` (default; robust or cluster covariance): the moment rows
  ``h_i u_i`` of the instrument vector ``h_i = [X1 Z2 G]_i`` are replaced by their
  influence functions including the first step (``g_m`` and the means of ``Z``):

      r_{i,(m,j)} = G_{i,(m,j)} u_i - e2hat_{i,m} * [P_X1 ((Z_j - Zbar_j) o u)]_i
                    - (e2hat_m'u / N) (Z_ij - Zbar_j),

  (exogenous and external rows unchanged). 2SLS: ``V = B C' (sum r r') C B'`` with
  ``B = (X'P_H X)^-1`` and ``C = (H'H)^-1 H'X``. GMM: the weight matrix is the inverse
  of the corrected moment covariance at the 2SLS residuals and the efficient
  ``(X'H S^-1 H'X)^-1`` is reported, with Hansen's J. Cluster sums replace the
  White sums with a cluster covariance. ``small`` applies ``N/(N-K)`` (robust) or
  ``G/(G-1) (N-1)/(N-K)`` (cluster; ``G/(G-1)`` always).

``tests['lewbel_bp_<endog>']`` is Koenker's studentized Breusch-Pagan statistic
``N R^2`` of ``e2hat_m^2`` on ``[1 Z]``, chi2(p): Lewbel's relevance condition needs
``cov(Z, e2^2) != 0``. Unweighted data, an intercept and numeric generators only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, kernel_call, linear_covariance, make_spec,
)
from openecon.econometrics.iv import kernels
from openecon.econometrics.iv.common import (
    Setting, check_fit, covariance, estimate, moment_meat, role_designs, screen, sum_of_squares,
)
from openecon.econometrics.iv.diagnostics import Diagnostics, chi2_entry
from openecon.econometrics.iv.ivregress import model_test
from openecon.engines.linalg import least_squares
from openecon.models import ModelSpec, ResultBundle

TITLES = {"2sls": "IV (2SLS) with Lewbel generated instruments",
          "gmm": "IV (GMM) with Lewbel generated instruments"}
CONDITIONAL_TESTS = ("cragg_donald", "kleibergen_paap_rk_f")


def _generators(frame: ModelFrame, exog: Design) -> tuple[list[str], Tensor]:
    chosen = frame.role("generators")
    candidates = [term for term in exog.terms if term != "Intercept"]
    if not chosen:
        chosen = candidates
    stray = [name for name in chosen if name not in candidates]
    if stray:
        raise AnalysisError("invalid_spec", f"Generators {', '.join(stray)} must be numeric "
                            "exogenous regressors listed in x (categorical columns are not "
                            "accepted; create the dummies explicitly).")
    if not chosen:
        raise AnalysisError("invalid_spec", "Lewbel instruments need at least one non-constant "
                            "exogenous regressor to generate from.")
    index = [exog.terms.index(name) for name in chosen]
    return list(chosen), exog.x[:, index]


def _koenker(e2: Tensor, z: Tensor) -> tuple[float, int]:
    """Studentized Breusch-Pagan N R^2 of e2^2 on [1 Z]."""
    target = e2.square()
    design = torch.cat([torch.ones((len(target), 1), dtype=torch.float64), z], 1)
    fit = least_squares(design, target, None, drop_collinear=True)
    tss = float((target - target.mean()).square().sum())
    if not tss > 0:
        return 0.0, len(fit.kept) - 1
    return len(target) * (1 - float(fit.ssr) / tss), len(fit.kept) - 1


def fit_ivlewbel(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``ivlewbel``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    method, small = frame.option("method"), bool(frame.option("small"))
    first_step = frame.option("first_step")
    if not spec.intercept:
        raise AnalysisError("invalid_spec", "Lewbel generated instruments are centered and need "
                            "the constant; keep intercept=True.")
    if first_step == "corrected" and spec.covariance == "nonrobust":
        raise AnalysisError("unsupported_covariance", "The first-step correction is a robust or "
                            "cluster influence-function covariance; use covariance='robust' or "
                            "first_step='ignored' (ivreg2h).")
    endogenous = frame.role("endogenous")
    overlap = set(frame.role("generators")) & (set(endogenous) | set(frame.role("instruments")))
    if overlap or spec.outcome in frame.role("generators"):
        raise AnalysisError("invalid_spec", "Generators must be exogenous regressors, distinct "
                            "from the outcome, endogenous regressors and external instruments.")
    exog, endog, external = role_designs(frame, intercept=True)
    names, z = _generators(frame, exog)
    n = frame.n
    frame.workspace_plan("Lewbel generated instruments", {
        "generated": n * len(names) * len(endogenous) * 8 * 4, "first_stage": n * 8 * 4})
    first = kernel_call(least_squares, exog.x, endog.x, None, drop_collinear=True)
    e2 = first.resid if first.resid.ndim == 2 else first.resid[:, None]
    zbar = z.mean(dim=0)
    zc = z - zbar
    pieces, terms = [], []
    for m, target in enumerate(endog.terms):
        variation = float((endog.x[:, m] - endog.x[:, m].mean()).square().sum())
        residual = e2[:, m]
        if not float(residual.square().sum()) > 1e-20 * max(variation, 1e-300):
            residual = torch.zeros_like(residual)   # exact zero columns are screened out
        pieces.append(zc * residual[:, None])
        terms.extend(f"{target}_{name}_g" for name in names)
    generated = Design(torch.cat(pieces, 1), terms, {}, False)
    instruments = Design(torch.cat([external.x, generated.x], 1), [*external.terms, *terms],
                         {}, False)
    blocks = screen(frame, exog, endog, instruments, None)
    y = frame.numeric(spec.outcome)
    clusters = frame.cluster_dimensions() if spec.covariance == "cluster" else None
    setting = Setting(n, None, spec.covariance, small, clusters=clusters,
                      cluster_names=registry.cluster_columns(spec) or None)
    wmatrix = "cluster" if spec.covariance == "cluster" else "robust"
    run_method = method if first_step == "ignored" else "2sls"
    est = estimate(frame, setting, y, blocks, method=run_method, wmatrix=wmatrix)
    kept_terms = blocks.instr.terms
    k = len(blocks.terms)
    check_fit(est.rss, sum_of_squares(y, None, centered=True), n - k,
              sum_of_squares(y, None, centered=False))
    diagnostics = Diagnostics(frame, setting, y, blocks, est).run()
    tests: dict[str, Any] = {}
    gmm_record = est.gmm
    beta, v, info, resid = est.beta, est.covariance, dict(est.info), est.resid
    if first_step == "corrected":
        h = torch.cat([blocks.x1, blocks.z2], 1)
        x = torch.cat([blocks.x1, blocks.x2], 1)
        rows = _corrected_rows(h, est.tsls.resid, blocks, kept_terms, names, zc, e2,
                               endog.terms, n)
        if method == "2sls":
            loading = kernel_call(least_squares, h, x, None, drop_collinear=False).beta  # C
            total = est.tsls.bread @ loading.T
            v, info = linear_covariance(
                frame, x=rows, resid=torch.ones(n, dtype=torch.float64), bread=total, n=n, k=k,
                df_resid=n - k, score_x=rows, small=small, group_factor=True,
                kind=spec.covariance, clusters=clusters,
                cluster_names=registry.cluster_columns(spec) or None)
            info["correction"] = "Lewbel first-step corrected 2SLS sandwich; " + str(info["correction"])
        else:
            beta, v, info, resid, gmm_record = _corrected_gmm(frame, setting, h, x, y, rows, k)
            tests["hansen_j"] = chi2_entry(
                gmm_record["j"], len(kept_terms) - len(blocks.endog.terms),
                "Hansen's J with the first-step corrected moment covariance",
                wmatrix=wmatrix) if len(kept_terms) > len(blocks.endog.terms) else {
                    "statistic": None, "df": 0, "p_value": None, "distribution": "chi2",
                    "note": "exactly identified"}
        kept = {name: value for name, value in diagnostics.tests.items()
                if name in CONDITIONAL_TESTS}
        for value in kept.values():
            value["note"] = "conditional on the generated instruments (first step not propagated)"
        tests.update(kept)
    else:
        tests.update(diagnostics.tests)
    for m, target in enumerate(endog.terms):
        statistic, df = _koenker(e2[:, m], z)
        tests[f"lewbel_bp_{target}"] = chi2_entry(
            statistic, df, f"Koenker Breusch-Pagan test of first-stage heteroskedasticity "
            f"in the generators ({target})") if df > 0 else {"statistic": None}
    rss = float(resid.square().sum())
    tss = sum_of_squares(y, None, centered=True)
    df_resid = n - k
    estimate_view = type(est)(est.method if first_step == "ignored" else method, beta, v, info,
                              resid, rss, est.proj, est.tsls)
    tests = {"model": model_test(estimate_view, blocks.terms, small), **tests}
    r_squared = 1 - rss / tss
    generated_kept = [term for term in kept_terms if term in set(terms)]
    metrics: dict[str, Any] = {
        "r_squared": r_squared, "adjusted_r_squared": 1 - (1 - r_squared) * (n - 1) / df_resid,
        "rmse": (rss / (df_resid if small else n)) ** 0.5, "df_model": k - 1,
        "df_resid": df_resid, "n_instruments": len(kept_terms),
        "n_generated_instruments": len(generated_kept),
        "n_external_instruments": len(kept_terms) - len(generated_kept),
        "n_endogenous": len(blocks.endog.terms),
    }
    if gmm_record is not None:
        metrics.update({"j": gmm_record["j"], "gmm_iterations": gmm_record["iterations"]})
    info.update({
        "df_inference": info.get("df_inference") if small else None, "nobs": n,
        "method": method, "small": small, "first_step": first_step,
        "error_variance": "RSS/(N-K)" if small else "RSS/N",
        "assumptions": "Lewbel (2012): E[X e1] = E[X e2] = 0, cov(Z, e1 e2) = 0, "
                       "cov(Z, e2^2) != 0; independent observations (clusters)",
    })
    extra = {
        "method": method, "first_step": first_step, "exogenous": blocks.exog.terms,
        "endogenous": blocks.endog.terms, "instruments": kept_terms,
        "external_instruments": [t for t in kept_terms if t not in set(terms)],
        "generated_instruments": generated_kept, "generators": names,
        "generator_means": zbar.tolist(),
        "omitted_instruments": blocks.omitted_instruments,
        "first_stage": diagnostics.first_stage,
        "first_stage_regressors": exog.terms,
        "source": "Lewbel (2012) JBES 30:67-80; ivreg2h (Baum and Schaffer) conventions",
    }
    if gmm_record is not None:
        extra["gmm"] = dict(gmm_record)
    return build_result(
        frame, terms=blocks.terms, params=beta, covariance=v, title=TITLES[method],
        use_t=small, df_inference=info.get("df_inference") if small else None,
        df_resid=df_resid, metrics=metrics, fitted=y - resid, nobs=n, inference=info,
        tests=tests, extra=extra, categories=blocks.exog.categories,
        solver="householder_qr_generated_instruments")


def _corrected_rows(h: Tensor, u: Tensor, blocks, kept_terms: list[str], names: list[str],
                    zc: Tensor, e2: Tensor, endog_terms: list[str], n: int) -> Tensor:
    """Influence rows of the instrument moments including Lewbel's first step."""
    rows = h * u[:, None]
    projected = kernel_call(least_squares, blocks.x1, zc * u[:, None], None,
                            drop_collinear=False).fitted
    if projected.ndim == 1:
        projected = projected[:, None]
    k1 = blocks.x1.shape[1]
    lookup = {f"{target}_{name}_g": (m, j) for m, target in enumerate(endog_terms)
              for j, name in enumerate(names)}
    for column, term in enumerate(kept_terms):
        if term not in lookup:
            continue
        m, j = lookup[term]
        rows[:, k1 + column] -= e2[:, m] * projected[:, j] + (e2[:, m] @ u / n) * zc[:, j]
    return rows


def _corrected_gmm(frame: ModelFrame, setting: Setting, h: Tensor, x: Tensor, y: Tensor,
                   rows: Tensor, k: int):
    scale = torch.linalg.vector_norm(h, dim=0)
    hs = h / scale
    meat = moment_meat(frame, setting, rows / scale)
    step = kernel_call(kernels.gmm_step, hs.T @ x, hs.T @ y, meat)
    resid = y - x @ step.beta
    _, info = covariance(frame, setting, score_x=hs @ step.loading, resid=resid,
                         bread=step.bread, k=k)
    v = step.bread * info.get("small_sample_correction", 1.0)
    info["correction"] = ("efficient GMM (X'H S^-1 H'X)^-1 with the first-step corrected "
                          "moment covariance; " + str(info["correction"]))
    record = {"wmatrix": setting.kind, "iterations": 1, "converged": True, "igmm": False,
              "centered": False, "j": step.j, "first_step": "corrected"}
    return step.beta, v, info, resid, record


def ivlewbel(*, data: Any, y: str, x: Sequence[str], endog: Sequence[str],
             instruments: Sequence[str] | None = None, generators: Sequence[str] | None = None,
             method: str = "2sls", first_step: str = "corrected", covariance: str | None = None,
             cluster: str | None = None, small: bool = False,
             categorical: Sequence[str] | None = None, missing: str = "raise",
             alpha: float = 0.05) -> ResultBundle:
    """IV with Lewbel (2012) heteroskedasticity-based generated instruments (Stata ivreg2h).

    Model
        ``y = x'b1 + endog'b2 + e1`` and ``endog_m = x'g_m + e2_m``. Identification
        uses ``cov(Z, e1 e2) = 0`` and ``cov(Z, e2^2) != 0`` for ``Z = generators``
        (default: every non-constant exogenous regressor). The instruments
        ``(Z_j - mean Z_j) * e2hat_m`` (``e2hat`` from OLS of ``endog_m`` on ``[1 x]``) are
        added to the optional external ``instruments`` and the model is fitted by
        ``method='2sls'`` or two-step efficient ``'gmm'``.

    Parameters
        first_step: ``'corrected'`` (default) propagates the estimated first-stage
            coefficients and generator means into a robust/cluster covariance (and
            the GMM weight matrix); ``'ignored'`` treats the generated instruments as
            data, as ivreg2h does, and also allows ``covariance='nonrobust'``.
        covariance: ``'robust'`` (default) or ``'cluster'`` (implied by ``cluster``).
        small: t and F with N-K degrees of freedom and regress-style factors.

    Result
        Coefficients of ``[Intercept x endog]``; ``tests['lewbel_bp_<endog>']`` (Koenker
        Breusch-Pagan relevance check), weak-identification statistics conditional on
        the generated instruments, ``hansen_j`` (gmm); with ``first_step='ignored'`` every
        ivregress diagnostic. ``extra['generated_instruments']`` names them
        ``<endog>_<z>_g``.

    Errors
        ``underidentified``, ``unsupported_covariance`` (nonrobust with the correction),
        ``invalid_spec`` (generators outside x, no intercept), ``singular_weight_matrix``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(2); n = 2000
        >>> x1, x2, f = rng.normal(size=(3, n))
        >>> e2 = np.exp(0.5 * x1) * rng.normal(size=n) + f
        >>> df = pd.DataFrame({"x1": x1, "x2": x2, "p": 1 + x1 + e2})
        >>> df["y"] = 1 + x1 - x2 + 0.5 * df.p + f + rng.normal(size=n)
        >>> oe.ivlewbel(data=df, y="y", x=["x1", "x2"], endog=["p"])
    """
    from openecon.analysis import fit

    if covariance is None:
        covariance = "cluster" if cluster is not None else "robust"
    columns = {"endogenous": column_list(endog, "endog")}
    if instruments:
        columns["instruments"] = column_list(instruments, "instruments")
    if generators:
        columns["generators"] = column_list(generators, "generators")
    spec = make_spec(
        "ivlewbel", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), covariance=covariance,
        cluster=cluster, missing=missing, alpha=alpha, columns=columns,
        options={"method": method, "first_step": first_step, "small": True if small else None})
    return fit(spec, data=data)
