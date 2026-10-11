"""Vectorised float64 kernels shared by the bootstrap causality procedures.

Everything here works on CPU float64 tensors without autograd, without a
Python loop over observations or replications (the only loops run over time
steps of a recursive simulation or over a bounded number of replication
chunks) and without forming an observation-by-observation matrix.

Conventions shared by every caller:

* Resampling indices come from a private ``torch.Generator`` seeded by the
  caller; the global RNG state is never touched. Callers document the exact
  order of their ``torch.randint`` calls so that the same draws can be
  supplied to an independent implementation.
* A replication whose design is numerically rank deficient or whose statistic
  is not finite is a *failed draw*. Callers never drop such draws silently;
  they raise ``bootstrap_failure`` with the attempted and failed counts.
* Plus-one bootstrap p-values (1 + #{T* >= T}) / (B + 1) and order-statistic
  critical values T*_(ceil((B + 1)(1 - a))) agree: p <= a exactly when T
  exceeds that critical value (no ties).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError

_EPS = torch.finfo(torch.float64).eps
# Relative pivot of a column-equilibrated R factor below which a design is
# treated as numerically rank deficient (condition number above ~1e12).
_RANK_TOL = 1e-12
LEVELS = (0.01, 0.05, 0.10)


def generator(seed: int) -> torch.Generator:
    """A private CPU generator; the global torch RNG state is not changed."""
    return torch.Generator(device="cpu").manual_seed(int(seed))


def plus_one_p_value(statistic: float, draws: Tensor) -> float:
    """(1 + #{draws >= statistic}) / (B + 1)."""
    exceed = int((draws >= statistic).sum())
    return (exceed + 1) / (draws.numel() + 1)


def mc_std_error(probability: float, draws: int) -> float:
    """Binomial Monte Carlo standard error sqrt(p (1 - p) / B) of a bootstrap p-value."""
    return math.sqrt(probability * (1.0 - probability) / draws)


def critical_values(draws: Tensor, levels: tuple[float, ...] = LEVELS) -> list[float]:
    """Upper critical values T*_(ceil((B + 1)(1 - a))) for each level a (1-based order statistic).

    ``draws`` is [B] or [..., B]; the result is a list of floats for [B] and a
    list of per-level tensors otherwise.
    """
    count = draws.shape[-1]
    ordered = torch.sort(draws, dim=-1).values
    out = []
    for level in levels:
        rank = min(count, max(1, math.ceil((count + 1) * (1.0 - level) - 1e-9)))
        value = ordered[..., rank - 1]
        out.append(float(value) if value.ndim == 0 else value)
    return out


def chi2_logsf_integer(x: Tensor, df: int) -> Tensor:
    """log P(chi2_df > x) for an integer df >= 1, elementwise and stable in both tails.

    Closed forms (Abramowitz and Stegun 26.4.4-26.4.5): for df = 2k,
    sf = exp(-x/2) sum_{j<k} (x/2)^j / j!; for df = 2k + 1,
    sf = erfc(sqrt(x/2)) + exp(-x/2) sum_{j<k} (x/2)^(j+1/2) / Gamma(j + 3/2).
    Terms are combined by log-sum-exp; log erfc(z) = log erfcx(z) - z^2.
    """
    if isinstance(df, bool) or not isinstance(df, int) or df < 1:
        raise ValueError("df must be a positive integer")
    x = x.to(torch.float64).clamp_min(0.0)
    half = x / 2.0
    log_half = torch.log(half)                       # -inf at x = 0
    if df % 2 == 0:
        k = df // 2
        j = torch.arange(k, dtype=torch.float64)
        terms = j * log_half[..., None] - torch.lgamma(j + 1.0)
        terms[..., 0] = 0.0                          # (x/2)^0 / 0! = 1, also at x = 0
        return -half + torch.logsumexp(terms, dim=-1)
    k = (df - 1) // 2
    z = torch.sqrt(half)
    log_erfc = torch.log(torch.special.erfcx(z)) - half
    if k == 0:
        return log_erfc
    j = torch.arange(k, dtype=torch.float64)
    series = -half[..., None] + (j + 0.5) * log_half[..., None] - torch.lgamma(j + 1.5)
    return torch.logsumexp(torch.cat([log_erfc[..., None], series], dim=-1), dim=-1)


def lagged(values: Tensor, lags: int, start: int) -> Tensor:
    """[..., T - start, lags] matrix of lags 1..lags of ``values`` [..., T] for rows start..T-1."""
    total = values.shape[-1]
    if lags == 0:
        return values.new_zeros((*values.shape[:-1], total - start, 0))
    return torch.stack([values[..., start - j:total - j] for j in range(1, lags + 1)], dim=-1)


def residualize(design: Tensor, outcome: Tensor, *, intercept: bool = True) -> tuple[Tensor, Tensor]:
    """Batched least-squares residuals of ``outcome`` [..., T] on ``design`` [..., T, m].

    With ``intercept`` the regression also contains a constant, which is
    swept out exactly by demeaning the outcome and every design column (the
    design itself must not contain a constant column). Columns are then
    equilibrated to unit norm before the Householder QR (residuals are
    invariant to column scaling). Returns (residuals [..., T], ok [...]),
    where ``ok`` is False for a numerically rank-deficient design (relative
    pivot below 1e-12) or a nonfinite result.
    """
    if intercept:
        design = design - design.mean(dim=-2, keepdim=True)
        outcome = outcome - outcome.mean(dim=-1, keepdim=True)
    if design.shape[-1] == 0:
        return outcome, torch.isfinite(outcome).all(-1)
    norms = torch.linalg.vector_norm(design, dim=-2, keepdim=True)
    safe = torch.where(norms > 0, norms, torch.ones_like(norms))
    q, r = torch.linalg.qr(design / safe, mode="reduced")
    pivots = r.diagonal(dim1=-2, dim2=-1).abs()
    ok = (norms[..., 0, :] > 0).all(-1) & (pivots > _RANK_TOL).all(-1)
    resid = outcome - (q @ (q.transpose(-1, -2) @ outcome[..., None]))[..., 0]
    ok = ok & torch.isfinite(resid).all(-1)
    return resid, ok


def _spd(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Batched lower Cholesky factor with the rounding-level pivot rule of engines.linalg."""
    k = matrix.shape[-1]
    sym = (matrix + matrix.transpose(-1, -2)) / 2
    factor, info = torch.linalg.cholesky_ex(sym)
    diag = sym.diagonal(dim1=-2, dim2=-1)
    lost = factor.diagonal(dim1=-2, dim2=-1).square() <= 8 * k * _EPS * diag.abs()
    ok = (info == 0) & ~lost.any(-1) & torch.isfinite(factor).all(-1).all(-1)
    return factor, ok


def sur_fit(outcome: Tensor, regressors: Tensor) -> dict[str, Tensor]:
    """Two-step feasible GLS (Zellner SUR) of N unit equations, each with its own intercept.

    ``outcome`` [..., N, T] and ``regressors`` [..., N, T, q] (no constant
    column; every equation has the same number q of slope regressors). With
    E the [N, T] OLS residuals of the unit regressions, Sigma = E E' / T and

        b = [X'(Sigma^-1 (x) I_T) X]^-1 X'(Sigma^-1 (x) I_T) y,
        V = [X'(Sigma^-1 (x) I_T) X]^-1.

    Because every equation has an intercept, the GLS projection on the
    intercept space is within-unit demeaning whatever Sigma is, so slopes are
    computed from demeaned data (Frisch-Waugh under GLS) and

        a_i = ybar_i - wbar_i' b_i,
        Cov(a) = Sigma / T + Wbar V Wbar',   Cov(a, b) = -Wbar V.

    The [Nq, Nq] normal matrix is formed blockwise as s^ij W_i'W_j on unit-norm
    columns and factored by Cholesky. Returns slopes [..., N, q], intercepts
    [..., N], ``cov_slopes`` [..., Nq, Nq], ``sigma`` (OLS residual covariance
    used as the GLS weight), SUR residuals [..., N, T], column means, the
    scaled slope vector and its scaled covariance (for scale-free Wald forms)
    and ``ok`` [...] (False for any singular step).
    """
    count, t, q = regressors.shape[-3], regressors.shape[-2], regressors.shape[-1]
    batch = regressors.shape[:-3]
    ybar = outcome.mean(-1)
    wbar = regressors.mean(-2)
    yc = outcome - ybar[..., None]
    wc = regressors - wbar[..., None, :]
    norms = torch.linalg.vector_norm(wc, dim=-2)                       # [..., N, q]
    ok = (norms > 0).all(-1).all(-1)
    ws = wc / torch.where(norms > 0, norms, torch.ones_like(norms))[..., None, :]
    qm, rm = torch.linalg.qr(ws, mode="reduced")
    ok = ok & (rm.diagonal(dim1=-2, dim2=-1).abs() > _RANK_TOL).all(-1).all(-1)
    ols_resid = yc - (qm @ (qm.transpose(-1, -2) @ yc[..., None]))[..., 0]
    sigma = ols_resid @ ols_resid.transpose(-1, -2) / t
    sigma_factor, good = _spd(sigma)
    ok = ok & good
    weight = torch.cholesky_inverse(sigma_factor)                      # Sigma^-1
    gram = torch.einsum("...itk,...jtl->...ikjl", ws, ws)
    normal = (weight[..., :, None, :, None] * gram).reshape(*batch, count * q, count * q)
    cross = torch.einsum("...itk,...jt->...ikj", ws, yc)
    rhs = (cross * weight[..., :, None, :]).sum(-1).reshape(*batch, count * q)
    normal_factor, good = _spd(normal)
    ok = ok & good
    scaled = torch.cholesky_solve(rhs[..., None], normal_factor)[..., 0]
    scaled_cov = torch.cholesky_inverse(normal_factor)
    scale = norms.reshape(*batch, count * q)
    slopes = (scaled / scale).reshape(*batch, count, q)
    cov = scaled_cov / (scale[..., :, None] * scale[..., None, :])
    resid = yc - torch.einsum("...itk,...ik->...it", wc, slopes)
    intercepts = ybar - (wbar * slopes).sum(-1)
    finite = torch.isfinite(slopes).all(-1).all(-1) & torch.isfinite(cov).all(-1).all(-1)
    return {"slopes": slopes, "intercepts": intercepts, "cov_slopes": cov, "sigma": sigma,
            "resid": resid, "means": wbar, "scaled": scaled, "scaled_cov": scaled_cov,
            "ok": ok & finite}


def unit_wald(scaled: Tensor, scaled_cov: Tensor, count: int, q: int, tested: list[int]) -> tuple[Tensor, Tensor]:
    """Unit Wald statistics b_i,R' [V_i,RR]^-1 b_i,R from a stacked SUR slope vector.

    ``tested`` are slope positions within each unit block. Column scaling
    cancels in the quadratic form, so the scaled vector and covariance are used.
    Returns (statistics [..., N], ok [...]).
    """
    index = torch.tensor([[unit * q + j for j in tested] for unit in range(count)], dtype=torch.int64)
    coef = scaled[..., index]                                           # [..., N, r]
    block = scaled_cov[..., index[:, :, None], index[:, None, :]]       # [..., N, r, r]
    factor, ok = _spd(block)
    stats = (coef[..., None, :] @ torch.cholesky_solve(coef[..., None], factor))[..., 0, 0]
    ok = ok.all(-1) & torch.isfinite(stats).all(-1) & (stats >= 0).all(-1)
    return stats, ok


def require_success(ok: Tensor, what: str) -> None:
    """Raise ``bootstrap_failure`` when any replication failed; nothing is dropped."""
    failed = int((~ok).sum())
    if failed:
        raise AnalysisError("bootstrap_failure", f"{failed} of {ok.numel()} {what} bootstrap "
                            "replications had a numerically singular design or a nonfinite "
                            "statistic. No draw is silently discarded or replaced; change the "
                            "seed, lags or sample, or reduce the number of replications.")


def simulate_ar(presample: Tensor, intercept: Tensor, coef: Tensor, fixed: Tensor,
                shocks: Tensor) -> Tensor:
    """Recursive univariate AR paths with fixed exogenous contributions, batched.

    ``presample`` [..., P] holds the initial values (oldest first), ``coef``
    [..., P] the coefficients of lags 1..P (zero padded), ``intercept`` [...],
    ``fixed`` [..., S] the exogenous part of each simulated period and
    ``shocks`` [..., S]. Returns [..., P + S]:

        y_t = intercept + sum_j coef_j y_(t-j) + fixed_t + shock_t.
    """
    order, steps = presample.shape[-1], shocks.shape[-1]
    shape = torch.broadcast_shapes(presample.shape[:-1], shocks.shape[:-1], coef.shape[:-1])
    out = presample.new_zeros((*shape, order + steps))
    out[..., :order] = presample
    reversed_coef = coef.flip(-1)                         # pairs with y_(t-P) .. y_(t-1)
    for t in range(steps):
        history = out[..., t:t + order]
        out[..., order + t] = (intercept + (history * reversed_coef).sum(-1)
                               + fixed[..., t] + shocks[..., t])
    return out


def simulate_var(presample: Tensor, intercept: Tensor, matrices: Tensor, shocks: Tensor) -> Tensor:
    """Recursive VAR paths y_t = c + sum_j A_j y_(t-j) + u_t, batched.

    ``presample`` [..., P, K] (oldest first), ``intercept`` [..., K],
    ``matrices`` [..., P, K, K] with A_j at position j - 1, ``shocks`` [..., S, K].
    Returns [..., P + S, K].
    """
    order, steps = presample.shape[-2], shocks.shape[-2]
    k = presample.shape[-1]
    shape = torch.broadcast_shapes(presample.shape[:-2], shocks.shape[:-2],
                                   matrices.shape[:-3], intercept.shape[:-1])
    out = presample.new_zeros((*shape, order + steps, k))
    out[..., :order, :] = presample
    for t in range(steps):
        value = intercept + shocks[..., t, :]
        for j in range(1, order + 1):
            value = value + (matrices[..., j - 1, :, :] @ out[..., order + t - j, :, None])[..., 0]
        out[..., order + t, :] = value
    return out


def check_bootstrap(draws: int, *, allow_zero: bool = False, maximum: int = 9999) -> None:
    """Replication count domain: 0 (when allowed) or 99..maximum."""
    if isinstance(draws, bool) or not isinstance(draws, int):
        raise AnalysisError("invalid_option", "bootstrap must be an integer.")
    if draws == 0 and allow_zero:
        return
    if not 99 <= draws <= maximum:
        raise AnalysisError("invalid_option", f"bootstrap must be {'0 or ' if allow_zero else ''}"
                            f"between 99 and {maximum} replications.")


def check_work(work: float, max_work: int, what: str) -> None:
    """Refuse a request whose declared floating-point work exceeds ``max_work``."""
    if isinstance(max_work, bool) or not isinstance(max_work, int) or max_work < 1:
        raise AnalysisError("invalid_option", "max_work must be a positive integer.")
    if work > max_work:
        raise AnalysisError("work_budget_exceeded", f"{what} needs about {work:.3g} floating-point "
                            f"operations, above max_work={max_work}. Reduce bootstrap "
                            "replications, lags, units or windows, or raise max_work explicitly.")
