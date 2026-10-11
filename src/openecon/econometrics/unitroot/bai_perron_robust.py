"""Bai--Perron robust tests, sequential supF(l+1|l) and break-date intervals.

Bai, J. and Perron, P. (1998). Estimating and testing linear models with
multiple structural changes. Econometrica 66, 47-78. Bai, J. and Perron, P.
(2003). Computation and analysis of multiple structural change models.
Journal of Applied Econometrics 18, 1-22. Bai, J. (1997). Estimation of a
change point in multiple regression models. REStat 79, 551-563.

Every formula follows Pierre Perron's GAUSS program BRCODE.SRC (October 2004
version with the Zeileis--Kleiber correction of the date intervals;
https://blogs.bu.edu/perron/files/2020/05/m-Break.zip): procedures pftest,
pvdel (pure change, hetdat = 1, hetvar = 1), correct/jhatpr/bandw/kern
(quadratic-spectral HAC with Andrews' AR(1) bandwidth and optional VAR(1)
prewhitening), spflp1 (supF(l+1|l) with the l breaks of the global
optimum), interval/cvg/funcg (hetq = 1, hetomega = 1). One deliberate
difference: the VAR(1) recolouring uses (I - A)^{-1} J (I - A)^{-T} with A
the VAR matrix (Andrews and Monahan 1992; also the CRAN port mbreaks), whereas
BRCODE.SRC transposes A; the two coincide when one coefficient changes.

Statistics are reported in F units (the Wald statistic divided by k q);
Bai and Perron's tables refer to q times this statistic, so their critical
values are divided by q here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.unitroot import break_tables
from openecon.engines.covariance import _kernel
from openecon.engines.distributions import normal_cdf, normal_sf

ERRORS = ("iid_gaussian", "heteroskedastic", "hac")


def _ols(z: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    q, r = torch.linalg.qr(z)
    beta = torch.linalg.solve_triangular(r, q.T @ y[:, None], upper=True)[:, 0]
    return beta, y - z @ beta


def bandwidth(v: Tensor) -> float:
    """Andrews (1991) AR(1) plug-in bandwidth for the QS kernel, unit weights (GAUSS bandw)."""
    n = v.shape[0]
    lead, lag = v[1:], v[:-1]
    denominator = lag.square().sum(dim=0)
    if not bool((denominator > 0).all()):
        raise AnalysisError("invalid_covariance", "A HAC score column is identically zero; the "
                            "automatic bandwidth is undefined.")
    b = (lag * lead).sum(dim=0) / denominator
    sig = (lead - lag * b).square().sum(dim=0) / (n - 1)
    a2n = (4 * b.square() * sig.square() / (1 - b) ** 8).sum()
    a2d = (sig.square() / (1 - b) ** 4).sum()
    a2 = float(a2n / a2d)
    if not math.isfinite(a2) or a2 <= 0:
        raise AnalysisError("invalid_covariance", "The AR(1) bandwidth approximation is "
                            "undefined (a unit-root or degenerate score column).")
    return 1.3221 * (a2 * n) ** 0.2


def _qs_long_run(v: Tensor) -> Tensor:
    """GAUSS jhatpr: V'V + sum_{j=1..n-1} k(j/st) (G_j + G_j'), divided by n - d."""
    n, d = v.shape
    st = bandwidth(v)
    size = 1 << (2 * n - 1).bit_length()
    taps = torch.zeros(size, dtype=torch.float64)
    lags = torch.arange(1, n, dtype=torch.float64)
    taps[1:n] = _kernel(lags / st, "quadratic_spectral")
    partner = torch.fft.irfft(torch.fft.rfft(v, n=size, dim=0)
                              * torch.fft.rfft(taps)[:, None], n=size, dim=0)[:n]
    cross = v.T @ partner                     # sum_j k_j sum_t v_t v_{t-j}'
    jhat = v.T @ v + cross + cross.T
    return 0.5 * (jhat + jhat.T) / (n - d)


def long_run(z: Tensor, u: Tensor, prewhiten: bool) -> Tensor:
    """GAUSS ``correct``: long-run covariance of z_t u_t (per observation, n - d divisor)."""
    v = z * u[:, None]
    d = v.shape[1]
    if not prewhiten:
        return _qs_long_run(v)
    previous, current = v[:-1], v[1:]
    # VAR(1): v_t = A v_{t-1} + v*_t, fitted column by column (B = A').
    b, _ = _ols_multi(previous, current)
    star = current - previous @ b
    jh = _qs_long_run(star)
    recolour, info = torch.linalg.inv_ex(torch.eye(d, dtype=torch.float64) - b.T)
    result = recolour @ jh @ recolour.T
    if int(info) != 0 or not bool(torch.isfinite(result).all()):
        raise AnalysisError("invalid_covariance", "The VAR(1) prewhitening filter is singular or "
                            "has a unit root; use prewhiten=False.")
    return result


def white(z: Tensor, u: Tensor) -> Tensor:
    """sum z z' u^2 / (n - d): the zero-bandwidth limit of ``long_run`` (no prewhitening)."""
    v = z * u[:, None]
    return v.T @ v / (v.shape[0] - v.shape[1])


def _ols_multi(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    q, r = torch.linalg.qr(x)
    beta = torch.linalg.solve_triangular(r, q.T @ y, upper=True)
    return beta, y - x @ beta


def segment_fits(z: Tensor, y: Tensor, bounds: list[int]) -> tuple[list[Tensor], Tensor]:
    """Segment-wise OLS coefficients and the stacked residuals for a partition."""
    betas, resid = [], torch.empty_like(y)
    for start, end in zip(bounds[:-1], bounds[1:]):
        beta, res = _ols(z[start:end], y[start:end])
        betas.append(beta)
        resid[start:end] = res
    return betas, resid


def segment_covariances(z: Tensor, resid: Tensor, bounds: list[int], errors: str,
                        prewhiten: bool, sigma2: float | None = None) -> list[Tensor]:
    """Per-segment covariance of the segment coefficients (pure change, block diagonal).

    iid_gaussian: sigma2 (Z_j'Z_j)^{-1}; heteroskedastic: (Z_j'Z_j)^{-1}
    n_j/(n_j - q) sum z z' u^2 (Z_j'Z_j)^{-1} (the zero-lag limit of the GAUSS
    HAC, i.e. a segment-wise HC1); hac: GAUSS pvdel robust = 1,
    hetvar = 1, (Z_j'Z_j)^{-1} n_j correct(z_j, u_j) (Z_j'Z_j)^{-1}.
    """
    out = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        zj, uj = z[start:end], resid[start:end]
        bread = torch.linalg.inv(zj.T @ zj)
        if errors == "iid_gaussian":
            out.append(sigma2 * bread)
            continue
        if errors == "heteroskedastic":
            meat = (end - start) * white(zj, uj)
        else:
            meat = (end - start) * long_run(zj, uj, prewhiten)
        out.append(bread @ meat @ bread)
    return out


def wald_f(z: Tensor, y: Tensor, bounds: list[int], errors: str, prewhiten: bool) -> float:
    """F statistic for no change against the partition ``bounds`` (GAUSS pftest / (k q)).

    (T - (k+1) q)/(T k q) delta' R' (R V R')^{-1} R delta, with V the covariance
    of the segment coefficients (iid: sigma^2 = SSR/T, a common variance).
    """
    total, width = z.shape
    breaks = len(bounds) - 2
    betas, resid = segment_fits(z, y, bounds)
    sigma2 = float(resid.square().sum()) / total
    covariances = segment_covariances(z, resid, bounds, errors, prewhiten, sigma2)
    delta = torch.cat(betas)
    v = torch.block_diag(*covariances)
    difference = torch.zeros((breaks * width, (breaks + 1) * width), dtype=torch.float64)
    eye = torch.eye(width, dtype=torch.float64)
    for j in range(breaks):
        difference[j * width:(j + 1) * width, j * width:(j + 1) * width] = -eye
        difference[j * width:(j + 1) * width, (j + 1) * width:(j + 2) * width] = eye
    contrast = difference @ delta
    middle = difference @ v @ difference.T
    try:
        factor = torch.linalg.cholesky(0.5 * (middle + middle.T))
    except RuntimeError as exc:
        raise AnalysisError("invalid_covariance", "The covariance of the coefficient changes is "
                            "not positive definite; use errors='iid_gaussian' or a longer "
                            "minimum segment.") from exc
    solved = torch.cholesky_solve(contrast[:, None], factor)[:, 0]
    wald = float(contrast @ solved)
    return (total - (breaks + 1) * width) * wald / (total * breaks * width)


def sequential_tests(z: Tensor, y: Tensor, costs: Tensor, partitions: list[list[int]],
                     h: int, errors: str, prewhiten: bool, first: float) -> list[dict]:
    """supF(l+1|l) for l = 0..m-1 with the l breaks of the global optimum (GAUSS spflp1).

    In every segment of at least 2h observations the SSR-minimizing split with h
    observations on both sides is tested by the one-break F statistic computed
    on that segment alone; the statistic is the largest of these.
    """
    total = z.shape[0]
    rows = [{"l": 0, "statistic": first, "new_break": partitions[1][0] if len(partitions) > 1
             else None, "segment": 1}]
    for ell in range(1, len(partitions) - 1):
        bounds = [0, *partitions[ell], total]
        best = (0.0, None, None)
        for segment, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
            if end - start < 2 * h:
                continue
            splits = torch.arange(start + h, end - h + 1)
            ssr = costs[start, splits] + costs[splits, end]
            if not bool(torch.isfinite(ssr).any()):
                continue
            split = int(splits[int(torch.argmin(ssr))])
            value = wald_f(z[start:end], y[start:end], [0, split - start, end - start],
                           errors, prewhiten)
            if value > best[0]:
                best = (value, split, segment)
        rows.append({"l": ell, "statistic": best[0], "new_break": best[1], "segment": best[2]})
    return rows


def _log_normal_sf(z: float) -> float:
    """log P(N(0,1) > z), accurate in the far upper tail (asymptotic series beyond z = 30)."""
    if z < 30.0:
        return math.log(normal_sf(z))
    inverse = 1.0 / (z * z)
    series = 1.0 - inverse * (1.0 - 3.0 * inverse * (1.0 - 5.0 * inverse * (1.0 - 7.0 * inverse)))
    return -0.5 * z * z - math.log(z) - 0.5 * math.log(2.0 * math.pi) + math.log(series)


def _funcg(x: float, bet: float, alph: float, b: float, deld: float, gam: float) -> float:
    """CDF of the argmax of a two-sided drifted Brownian motion (Bai 1997; GAUSS funcg).

    The GAUSS code switches to a first-order Mills-ratio approximation once the
    normal argument exceeds 30 (and then also multiplies the x <= 0 term by an
    extra Phi(-sqrt|x|/2)); here every exp(c x) Phi(-d sqrt|x|) product is
    evaluated exactly in log space instead, which agrees with strucchange's
    independent pargmaxV to rounding.
    """
    root = math.sqrt(abs(x))
    if x <= 0:
        product = math.exp(math.log(bet / alph) - alph * x + _log_normal_sf(bet * root))
        return (-math.sqrt(-x / (2 * math.pi)) * math.exp(x / 8) - product
                + ((2 * bet * bet / alph) - 2 - x / 2) * normal_cdf(-root / 2))
    product = math.exp(math.log(b * deld / gam) + gam * x + _log_normal_sf(deld * root))
    return (1 + (b / math.sqrt(2 * math.pi)) * root * math.exp(-b * b * x / 8) + product
            + (2 - b * b * x / 2 - 2 * deld * deld / gam) * normal_cdf(-b * root / 2))


def cvg(eta: float, phi1: float, phi2: float) -> list[float]:
    """2.5, 5, 95 and 97.5% points of the argmax distribution (GAUSS cvg parametrization).

    Bisection on [-2000, 2000] as the GAUSS code, but continued to 100 halvings
    (an interval of 3e-27) instead of stopping once |CDF - level| < 1e-6, so the
    quantiles do not inherit that stopping tolerance.
    """
    a = phi1 / phi2
    gam = ((phi2 / phi1) + 1) * eta / 2
    b = math.sqrt(phi1 * eta / phi2)
    deld = math.sqrt(phi2 * eta / phi1) + b / 2
    alph = a * (1 + a) / 2
    bet = (1 + 2 * a) / 2
    if _funcg(-2000.0, bet, alph, b, deld, gam) > 0.025 or _funcg(2000.0, bet, alph, b, deld, gam) < 0.975:
        raise AnalysisError("numerical_failure", "The break-date interval quantiles lie outside "
                            "[-2000, 2000]; the coefficient change is too small to date.")
    out = []
    for target in (0.025, 0.05, 0.95, 0.975):
        upper, lower = 2000.0, -2000.0
        for _ in range(100):
            middle = lower + (upper - lower) / 2
            if _funcg(middle, bet, alph, b, deld, gam) - target <= 0:
                lower = middle
            else:
                upper = middle
        out.append(lower + (upper - lower) / 2)
    return out


def date_intervals(z: Tensor, y: Tensor, bounds: list[int], errors: str,
                   prewhiten: bool) -> list[dict]:
    """Bai (1997) / Bai--Perron (1998) break-date intervals allowing heterogeneous regimes.

    GAUSS ``interval`` with hetq = hetomega = 1: Q_i = Z_i'Z_i/n_i in each regime;
    robust = 0 (iid_gaussian) uses phi_i = SSR_i/n_i; otherwise Omega_i is the
    regime's White (heteroskedastic) or QS-HAC (hac) long-run covariance and
    phi_1 = d'Omega_i d / d'Q_i d, phi_2 = d'Omega_{i+1} d / d'Q_{i+1} d,
    a = (d'Q_i d)^2 / d'Omega_i d. Bounds b - c/a with the cvg quantiles, then
    int() and +1 on the upper ends exactly as the GAUSS code.
    """
    betas, resid = segment_fits(z, y, bounds)
    out = []
    keys = ("lower_95", "upper_95", "lower_90", "upper_90")
    for i in range(1, len(bounds) - 1):
        left = slice(bounds[i - 1], bounds[i])
        right = slice(bounds[i], bounds[i + 1])
        n1, n2 = bounds[i] - bounds[i - 1], bounds[i + 1] - bounds[i]
        delta = betas[i] - betas[i - 1]
        q1 = z[left].T @ z[left] / n1
        q2 = z[right].T @ z[right] / n2
        dq1 = float(delta @ q1 @ delta)
        dq2 = float(delta @ q2 @ delta)
        try:
            eta = dq2 / dq1
            if errors == "iid_gaussian":
                phi1 = float(resid[left].square().sum()) / n1
                phi2 = float(resid[right].square().sum()) / n2
                scale = dq1 / phi1
            else:
                if errors == "heteroskedastic":
                    omega1, omega2 = white(z[left], resid[left]), white(z[right], resid[right])
                else:
                    omega1 = long_run(z[left], resid[left], prewhiten)
                    omega2 = long_run(z[right], resid[right], prewhiten)
                do1 = float(delta @ omega1 @ delta)
                do2 = float(delta @ omega2 @ delta)
                phi1, phi2 = do1 / dq1, do2 / dq2
                scale = dq1 ** 2 / do1
            if not all(math.isfinite(v) and v > 0 for v in (phi1, phi2, eta, scale)):
                raise AnalysisError("invalid_covariance", "A break-date interval needs positive "
                                    "regime variances and a nonzero coefficient change.")
            c025, c05, c95, c975 = cvg(eta, phi1, phi2)
        except (AnalysisError, ZeroDivisionError) as error:
            # One undatable break does not invalidate the tests or the other intervals.
            out.append({"break": i, "index": bounds[i], "status": f"unavailable: {error}",
                        **{key: None for key in keys}, **{f"{key}_raw": None for key in keys},
                        "eta": None, "phi1": None, "phi2": None, "scale": None})
            continue
        b = bounds[i]
        raw = {"lower_95": b - c975 / scale, "upper_95": b - c025 / scale,
               "lower_90": b - c95 / scale, "upper_90": b - c05 / scale}
        out.append({"break": i, "index": b, "status": "ok",
                    **{f"{key}_raw": value for key, value in raw.items()},
                    "lower_95": math.trunc(raw["lower_95"]),
                    "upper_95": math.trunc(raw["upper_95"]) + 1,
                    "lower_90": math.trunc(raw["lower_90"]),
                    "upper_90": math.trunc(raw["upper_90"]) + 1,
                    "eta": eta, "phi1": phi1, "phi2": phi2, "scale": scale})
    return out


def critical_value(kind: str, epsilon: float, q: int, column: int, level: str) -> float | None:
    """Bai--Perron asymptotic critical value in F units (table value / q), or None."""
    key = next((value for value in break_tables.BP_EPSILONS if abs(value - epsilon) < 1e-12), None)
    if key is None or not 1 <= q <= 10:
        return None
    tables = {"supf": break_tables.SUPF, "sequential": break_tables.SEQUENTIAL,
              "dmax": break_tables.DMAX}[kind]
    row = tables[key][level][q - 1]
    if column >= len(row):
        return None
    return row[column] / q
