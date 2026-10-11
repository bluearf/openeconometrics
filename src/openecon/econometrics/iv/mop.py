"""Montiel Olea--Pflueger (2013) robust weak-instrument test with effective-F critical values.

Transcribes the authors' Stata ``weakivtest`` (Pflueger and Wang; source
http://fmwww.bc.edu/repec/bocode/w/weakivtest.ado, version of 10/28/2020 with the
2018/2023 weight, cluster and reghdfe fixes) for ONE endogenous regressor:

1. ``y``, the endogenous regressor ``d`` and the excluded instruments are
   residualized on the included exogenous regressors (constant included); the
   instruments are orthonormalized to ``Zs'Zs = N I``.
2. ``v1``/``v2`` are the residuals of ``d``/``y`` on ``Zs``;
   ``Omega = [v2 v1]'[v2 v1] / (N - K - k1)`` (k1 exogenous columns, K instruments).
3. ``W = S N / (N - K - k1)`` with ``S`` the (1/N) moment covariance of
   ``[Zs v2, Zs v1]`` (avar): iid ``(U'U/N) kron I``, HC0, cluster sums or Bartlett HAC
   with ``lags`` (bandwidth lags + 1). ``W1``, ``W12``, ``W2`` are its K-by-K blocks
   (reduced form of y first, first stage second).
4. ``F_eff = a F_CD K Omega_22 / tr(W2) = a ||P_Z d||^2 / tr(W2)`` with
   ``a = N/(N-1) (G-1)/G`` for a cluster covariance and 1 otherwise.
5. Nagar-bias bounds (MOP 2013 eq. for B_e, weakivtest ``Bmaxfunction``/``BmaxLIML``):
   with ``S1 = W1 - 2 b W12 + b^2 W2``, ``S12 = W12 - b W2``, ``S2 = W2``,

       TSLS  B(b) = max |tr S12 - 2 eig(sym S12)| / sqrt(tr S2 tr S1)
       LIML  B(b) = max |tr S12 - r tr S1 - eig(sym(2 S12 - r S1))| / sqrt(tr S2 tr S1),
             r = (w12 - b w22) / (w11 - 2 b w12 + b^2 w22)   (w: Omega),

   and ``B_e = sup_b B(b)``. Here the supremum is taken over the whole real line
   (``b = tan(t)``, the bounded scale-invariant form at ``t = +-pi/2``) by a dense grid and
   golden-section refinement; weakivtest uses a grid on a bounded interval and
   Nelder-Mead.
6. Patnaik critical value (``cpatnaikgen``): with ``l`` the eigenvalues of ``W2``
   normalized to sum one and ``x = B_e / tau``,
   ``K_eff = 2 (1 + 2 x) / (2 sum l^2 + 4 x max l)`` and ``c = q_{1-alpha}(chi2_{K_eff}(x K_eff)) / K_eff``
   (exact noncentral chi-square quantile, as weakivtest under Stata >= 13.1).
   The simplified (conservative for TSLS) value uses ``x = 1/tau``.

weakivtest hard-codes ``x = 3.33 B`` for tau = 30%; here ``x = B / tau`` exactly (pass
``tau=[1/3.33]`` to reproduce that column). With one instrument every value reduces to
the upper-alpha quantile of chi2_1(1/tau) (Windmeijer 2025, arXiv 2309.01637 Sec. 4).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.engines.distributions import gamma_p

SOURCE = "http://fmwww.bc.edu/repec/bocode/w/weakivtest.ado"
GRID = 2048


def _poisson_window(mean: float) -> tuple[int, int, list[float]]:
    """Poisson(mean) log-weights covering all mass above 1e-18 of the mode."""
    if mean <= 0:
        return 0, 0, [0.0]
    mode = int(math.floor(mean))

    def log_weight(j: int) -> float:
        return -mean + j * math.log(mean) - math.lgamma(j + 1)

    top = log_weight(mode)
    low = mode
    while low > 0 and log_weight(low - 1) - top > -42:
        low -= 1
    high = mode
    while log_weight(high + 1) - top > -42:
        high += 1
    return low, high, [log_weight(j) for j in range(low, high + 1)]


def _ncx2_cdf_pdf(x: float, df: float, noncentrality: float) -> tuple[float, float]:
    if x <= 0:
        return 0.0, 0.0
    low, _, logs = _poisson_window(noncentrality / 2)
    cdf = pdf = 0.0
    for offset, log_w in enumerate(logs):
        half = df / 2 + low + offset
        weight = math.exp(log_w)
        cdf += weight * gamma_p(half, x / 2)
        pdf += math.exp(log_w + (half - 1) * math.log(x / 2) - x / 2 - math.lgamma(half)) / 2
    return min(1.0, cdf), pdf


def ncx2_isf(alpha: float, df: float, noncentrality: float) -> float:
    """Upper-alpha quantile of the noncentral chi-square (safeguarded Newton)."""
    if not (df > 0 and noncentrality >= 0 and 0 < alpha < 1):
        raise AnalysisError("invalid_option", "Noncentral chi-square needs df > 0, "
                            "noncentrality >= 0 and alpha in (0, 1).")
    target = 1 - alpha
    mean, spread = df + noncentrality, math.sqrt(2 * (df + 2 * noncentrality))
    low, high = 0.0, mean + 8 * spread
    while _ncx2_cdf_pdf(high, df, noncentrality)[0] < target:
        low, high = high, 2 * high
    point = min(max(mean + 1.645 * spread, low), high)
    for _ in range(200):
        cdf, pdf = _ncx2_cdf_pdf(point, df, noncentrality)
        if cdf < target:
            low = point
        else:
            high = point
        step = (cdf - target) / pdf if pdf > 0 else math.inf
        candidate = point - step
        if not low < candidate < high:
            candidate = (low + high) / 2
        if abs(candidate - point) <= 1e-14 * max(1.0, abs(point)) or high - low <= 1e-14 * high:
            return candidate
        point = candidate
    raise AnalysisError("numerical_failure", "Noncentral chi-square quantile did not converge.")


def patnaik(eigenvalues: Tensor, x: float, alpha: float) -> tuple[float, float]:
    """weakivtest ``cpatnaikgen``: (critical value, effective degrees of freedom)."""
    share = eigenvalues / eigenvalues.sum()
    variance = 2 * float(share.square().sum()) + 4 * x * float(share.max())
    k_eff = 2 * (1 + 2 * x) / variance
    return ncx2_isf(alpha, k_eff, k_eff * x) / k_eff, k_eff


def _bias(theta: Tensor, blocks: tuple[Tensor, Tensor, Tensor], omega: Tensor,
          estimator: str) -> Tensor:
    """Scale-invariant Nagar-bias ratio at b = tan(theta) (bounded at +-pi/2)."""
    w1, w12, w2 = blocks
    c, s = theta.cos()[:, None, None], theta.sin()[:, None, None]
    s12 = c * w12 - s * w2
    s1 = c.square() * w1 - 2 * c * s * w12 + s.square() * w2
    tr12 = s12.diagonal(dim1=1, dim2=2).sum(-1)
    tr1 = s1.diagonal(dim1=1, dim2=2).sum(-1)
    scale = (w2.trace() * tr1).clamp_min(1e-300).sqrt()
    if estimator == "tsls":
        eig = torch.linalg.eigvalsh((s12 + s12.transpose(1, 2)) / 2)
        lead, factor = tr12, 2.0
    else:
        cc, ss = c[:, 0, 0], s[:, 0, 0]
        sig12 = cc * omega[0, 1] - ss * omega[1, 1]
        sig1 = cc.square() * omega[0, 0] - 2 * cc * ss * omega[0, 1] + ss.square() * omega[1, 1]
        ratio = sig12 / sig1
        matrix = 2 * s12 - ratio[:, None, None] * s1
        eig = torch.linalg.eigvalsh((matrix + matrix.transpose(1, 2)) / 2)
        lead, factor = tr12 - ratio * tr1, 1.0
    low = (lead - factor * eig[:, 0]).abs()
    high = (lead - factor * eig[:, -1]).abs()
    return torch.maximum(low, high) / scale


def bias_bound(blocks: tuple[Tensor, Tensor, Tensor], omega: Tensor,
               estimator: str) -> tuple[float, float | None]:
    """sup over the real line of the Nagar-bias ratio; returns (B, argmax b or None=inf)."""
    k = blocks[0].shape[0]
    grid = -math.pi / 2 + math.pi * torch.arange(GRID, dtype=torch.float64) / GRID
    chunk = max(1, 4_000_000 // max(1, k * k))
    values = torch.cat([_bias(grid[i:i + chunk], blocks, omega, estimator)
                        for i in range(0, GRID, chunk)])
    if not bool(torch.isfinite(values).all()):
        raise AnalysisError("numerical_failure", "The Nagar-bias ratio is not finite; the "
                            "moment covariance is degenerate.")
    left, right = values.roll(1), values.roll(-1)
    peaks = torch.nonzero((values >= left) & (values >= right)).flatten()
    order = peaks[torch.argsort(values[peaks], descending=True)][:6]
    best, best_theta = float(values.max()), float(grid[int(values.argmax())])
    step = math.pi / GRID
    golden = (math.sqrt(5) - 1) / 2

    def value(t: float) -> float:
        return float(_bias(torch.tensor([t], dtype=torch.float64), blocks, omega, estimator)[0])

    for index in order.tolist():
        a, b = float(grid[index]) - step, float(grid[index]) + step
        c, d = b - golden * (b - a), a + golden * (b - a)
        fc, fd = value(c), value(d)
        for _ in range(80):
            if fc >= fd:
                b, d, fd = d, c, fc
                c = b - golden * (b - a)
                fc = value(c)
            else:
                a, c, fc = c, d, fd
                d = a + golden * (b - a)
                fd = value(d)
            if b - a < 1e-13:
                break
        for t, f in ((c, fc), (d, fd)):
            if f > best:
                best, best_theta = f, t
    cosine = math.cos(best_theta)
    return best, (math.tan(best_theta) if abs(cosine) > 1e-12 else None)


def weakivtest(*, data: Any, y: str, endog: Sequence[str], instruments: Sequence[str],
               x: Sequence[str] | None = None, covariance: str = "robust",
               cluster: str | None = None, time: str | None = None, lags: int | None = None,
               alpha: float = 0.05, tau: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
               intercept: bool = True, missing: str = "raise") -> TableSet:
    """Montiel Olea--Pflueger robust weak-IV test: effective F with TSLS/LIML critical values.

    One endogenous regressor. Rejects weak instruments when F_eff exceeds the
    generalized (TSLS, LIML) or simplified critical value for Nagar bias tau.
    """
    from openecon.econometrics.iv.common import Setting, moment_meat
    from openecon.econometrics.iv.weak_inference import _geometry

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("invalid_option", "alpha must lie strictly between 0 and 1.")
    try:
        taus = [float(t) for t in tau]
    except (TypeError, ValueError) as exc:
        raise AnalysisError("invalid_option", "tau must be a list of fractions in (0, 1).") from exc
    if not taus or any(not 0 < t < 1 for t in taus):
        raise AnalysisError("invalid_option", "tau must be a list of fractions in (0, 1).")
    frame, blocks, target, d, z, q, df = _geometry(
        data, y, x, endog, instruments, covariance, cluster, time, lags, "bartlett", intercept,
        missing)
    if d.shape[1] != 1:
        raise AnalysisError("unsupported_target", "weakivtest requires one endogenous regressor.")
    n, k = len(target), z.shape[1]
    zs = q * math.sqrt(n)
    first = q @ (q.T @ d[:, 0])
    v1 = d[:, 0] - first
    v2 = target - q @ (q.T @ target)
    resid = torch.stack([v2, v1], 1)
    omega = resid.T @ resid / df
    clusters = frame.cluster_dimensions() if covariance == "cluster" else None
    if covariance == "nonrobust":
        s = torch.kron(resid.T @ resid / n, zs.T @ zs / n)
    else:
        setting = Setting(n, None, covariance, False, clusters=clusters,
                          cluster_names=[cluster] if isinstance(cluster, str) else None)
        rows = torch.cat([zs * v2[:, None], zs * v1[:, None]], 1)
        s = moment_meat(frame, setting, rows) / n
    w = (s + s.T) / 2 * n / df
    w1, w12, w2 = w[:k, :k], w[:k, k:], w[k:, k:]
    adjust = 1.0
    groups = None
    if covariance == "cluster":
        groups = int(clusters[0][1])
        adjust = n / (n - 1) * (groups - 1) / groups
    signal = float(first.square().sum())
    trace2 = float(w2.trace())
    noise = float(omega[1, 1])
    if not (trace2 > 0 and noise > 0):
        raise AnalysisError("singular_first_stage", "The first-stage moment covariance is "
                            "degenerate; the effective F is undefined.")
    f_cd = signal / k / noise
    f_eff = adjust * f_cd * k * noise / trace2
    eigen = torch.linalg.eigvalsh(w2)
    if float(eigen.min()) <= 0:
        raise AnalysisError("singular_first_stage", "The first-stage moment covariance W2 is "
                            "not positive definite.")
    b_tsls, beta_tsls = bias_bound((w1, w12, w2), omega, "tsls")
    b_liml, beta_liml = bias_bound((w1, w12, w2), omega, "liml")
    details, summary = [], []
    for t in taus:
        row = {"tau": t}
        for name, bound in (("tsls", b_tsls), ("liml", b_liml), ("simplified", 1.0)):
            point = bound / t
            critical, k_eff = patnaik(eigen, point, alpha)
            details.append({"estimator": name, "tau": t, "x": point, "k_eff": k_eff,
                            "critical_value": critical, "reject_weak": f_eff > critical})
            row[name] = critical
        summary.append(row)
    return TableSet(
        {"critical_values": table(summary, columns=["tau", "tsls", "liml", "simplified"]),
         "details": table(details, columns=["estimator", "tau", "x", "k_eff", "critical_value",
                                            "reject_weak"])},
        title="Montiel Olea--Pflueger robust weak-instrument test",
        effective_f=f_eff, effective_f_unadjusted=signal * n / (trace2 * df),
        cragg_donald_f=f_cd, b_tsls=b_tsls, b_liml=b_liml, b_tsls_beta=beta_tsls,
        b_liml_beta=beta_liml, alpha=alpha, tau=taus, n_instruments=k, nobs=n,
        df_first_stage=df, covariance=covariance, lags=lags if covariance == "hac" else None,
        n_clusters=groups, cluster_adjustment=adjust, w_matrix=w.tolist(),
        omega=omega.tolist(), endogenous=blocks.endog.terms[0],
        instruments=blocks.instr.terms, sample_positions=frame.positions, source=SOURCE,
        null="weak instruments: Nagar bias exceeds tau times the worst-case benchmark",
        conventions="weakivtest: W and Omega with N-K-k1 degrees of freedom, cluster F_eff "
                    "times N/(N-1)(G-1)/G, exact noncentral chi-square Patnaik quantile, "
                    "x = B/tau (weakivtest uses 3.33 B at tau=0.30).")
