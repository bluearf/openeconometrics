"""MSE- and CER-optimal bandwidths of rdbwselect (Calonico, Cattaneo and Titiunik 2014).

Each step evaluates, per side, the constants of ``rdrobust_bw`` for a local
polynomial of order ``o`` estimating derivative ``nu``, a variance bandwidth
``h_V`` (the pilot ``c_bw``) and a bias bandwidth ``h_B``:

    V = (2 nu + 1) h_V^(2 nu + 1) [(R'WR)^-1 M (R'WR)^-1]_{nu, nu}
    BConst = h_V^nu [(R'WR)^-1 R'W ((x - c)/h_V)^(o+1)]_nu
    B = sqrt(2 (o + 1 - nu)) BConst beta_{o+1}      (order-o_B fit with h_B)
    R = scale 2 (o + 1 - nu) 3 BConst^2 V(beta_{o+1})   (regularization)

with W the kernel weights at h_V, M the meat of the chosen residuals. The
pilot is ``c_bw = C_K min(sd(x), IQR(x)/1.349) N^(-1/5)`` (C_K = 2.576,
2.34, 1.843 for the triangular, Epanechnikov and uniform kernels), capped at
the largest distance to the cutoff (``bwrestrict``). Then (mserd):

    d = ((V_l + V_r) / (B_r - B_l)^2)^(1/(2o+3))         o = q + 1, nu = q + 1, o_B = q + 2,
                                                         h_B = support of each side
    b = ((V_l + V_r) / ((B_r - B_l)^2 + R_l + R_r))^(1/(2q+3))   o = q, nu = p + 1, h_B = d
    h = ((V_l + V_r) / ((B_r - B_l)^2 + R_l + R_r))^(1/(2p+3))   o = p, nu = 0, h_B = b

``msetwo`` uses one side's V and B at a time, ``msesum`` the sum of the
biases, ``msecomb1`` the minimum of mserd and msesum and ``msecomb2`` the
median of mserd, msesum and msetwo; the ``cer*`` variants multiply h by
N^(-p/((3+p)(3+2p))) (coverage-error rate; b is kept; N counts the clusters
of both sides when the variance is clustered). Every bandwidth is capped at
the distance from the cutoff to the farthest observation (``bwrestrict``).

Fuzzy designs (rdrobust's default, not ``sharpbw``) evaluate every constant on
the outcome and treatment columns combined by s = (1/tau_T, -tau_Y/tau_T^2),
tau_Y and tau_T being the order-nu coefficients of the side's pilot fit, as
rdrobust_bw does; the caller falls back to the outcome equation when the
treatment is constant on one side (rdbwselect's perfect-compliance rule).
The first step's bias bandwidth is the side's range plus 1e-8 in the units of
the running variable, exactly as in the R/Stata package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.econometrics.teffects import rdkernels as rk
from openecon.engines.contracts import KernelError

PILOT_CONSTANTS = {"triangular": 2.576, "epanechnikov": 2.34, "uniform": 1.843}
SELECTORS = ("mserd", "msetwo", "msesum", "msecomb1", "msecomb2",
             "cerrd", "certwo", "cersum", "cercomb1", "cercomb2")


@dataclass
class Constants:
    V: float
    B: float
    R: float
    rate: float


@dataclass
class Bandwidths:
    h: tuple[float, float]     # (left, right)
    b: tuple[float, float]
    pilot: float
    method: str
    restricted: bool


def _quantile_type2(x: Tensor, prob: float) -> float:
    """R's quantile(type = 2): averaged inverse of the empirical distribution."""
    s = torch.sort(x).values
    n = s.numel()
    g = n * prob
    j = math.floor(g)
    if g - j > 1e-12:
        return float(s[min(j, n - 1)])
    return float((s[max(j - 1, 0)] + s[min(j, n - 1)]) / 2)


def pilot_bandwidth(x: Tensor, kernel: str, n: int | None = None) -> float:
    n = x.numel() if n is None else n
    sd = float(x.std())
    iqr = _quantile_type2(x, 0.75) - _quantile_type2(x, 0.25)
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    return PILOT_CONSTANTS[kernel] * spread * n ** -0.2


def constants(side: rk.Side, o: int, nu: int, o_b: int, h_v: float, h_b: float, scale: float,
              kernel: str, vce: str, matches: int, fuzzy: bool = False) -> Constants:
    """``rdrobust_bw`` for one side.

    Sharp design (or ``fuzzy=False``): the outcome, the first column of ``side.y``.
    Fuzzy design: the outcome and treatment columns are combined by the linearization
    s = (1/tau_T, -tau_Y/tau_T^2) of the ratio, with tau_Y, tau_T the order-``nu``
    coefficients of this side's pilot fit (rdrobust_bw's ``s``; the common factor nu!
    cancels from every bandwidth ratio and is omitted).
    """
    part, weights = rk.weighted_part(side, h_v, kernel)
    fit = rk.local_fit(side, part, weights, o, "pilot local polynomial")
    residual = side.y[part] - fit.design @ fit.beta
    gram = residual.T @ (weights[:, None] * residual)
    adjustment = rk.covariate_adjustment(gram, side.responses)
    adjusted_beta = fit.beta @ adjustment
    response_combine = torch.zeros(side.responses, dtype=torch.float64)
    if fuzzy:
        tau_y, tau_t = float(adjusted_beta[nu, 0]), float(adjusted_beta[nu, 1])
        if tau_t == 0 or not math.isfinite(tau_y / tau_t ** 2):
            raise KernelError("bandwidth_selection_failed",
                              "The fuzzy bandwidth selector needs a nonzero treatment "
                              "coefficient in each side's pilot fit; give h explicitly or use "
                              "sharpbw=True.")
        response_combine[0], response_combine[1] = 1 / tau_t, -tau_y / tau_t ** 2
    else:
        response_combine[0] = 1.0
    combine = adjustment @ response_combine
    resid = rk.residuals(side, fit, vce, matches, None, o + 1)
    rows = fit.design * weights[:, None]
    cluster = None if side.cluster is None else side.cluster[part]
    middle = rk.meat(rows, resid, combine, cluster)
    v_v = float((fit.inverse @ middle @ fit.inverse)[nu, nu])
    moment = rows.T @ (side.dx[part] / h_v) ** (o + 1)
    b_const = h_v ** nu * float((fit.inverse @ moment)[nu])
    part_b, weights_b = rk.weighted_part(side, h_b, kernel)
    fit_b = rk.local_fit(side, part_b, weights_b, o_b, "bias local polynomial")
    regularization = 0.0
    if scale > 0:
        resid_b = rk.residuals(side, fit_b, vce, matches, None, o_b + 1)
        rows_b = fit_b.design * weights_b[:, None]
        cluster_b = None if side.cluster is None else side.cluster[part_b]
        middle_b = rk.meat(rows_b, resid_b, combine, cluster_b)
        v_b = float((fit_b.inverse @ middle_b @ fit_b.inverse)[o + 1, o + 1])
        regularization = 3 * b_const ** 2 * v_b
    bias = math.sqrt(2 * (o + 1 - nu)) * b_const * float(fit_b.beta[o + 1] @ combine)
    variance = (2 * nu + 1) * h_v ** (2 * nu + 1) * v_v
    return Constants(variance, bias, scale * 2 * (o + 1 - nu) * regularization, 1 / (2 * o + 3))


def _ratio(variance: float, bias2: float, regularization: float, rate: float, what: str) -> float:
    denominator = bias2 + regularization
    if not denominator > 0 or not variance > 0:
        raise KernelError("bandwidth_selection_failed",
                          f"The {what} bandwidth is undefined (estimated bias and regularization "
                          "are zero); give the bandwidths h and b explicitly.")
    return (variance / denominator) ** rate


def select(left: rk.Side, right: rk.Side, x: Tensor, *, p: int, q: int, kernel: str, vce: str,
           matches: int, scale: float, method: str, restrict: bool,
           fuzzy: bool = False, deriv: int = 0, masspoints: str = "adjust", bwcheck=None) -> Bandwidths:
    """rdbwselect for deriv = 0 (``fuzzy``: the fuzzy-design constants of rdrobust_bw)."""
    max_l, max_r = abs(float(left.dx[0])), abs(float(right.dx[-1]))
    bw_max = max(max_l, max_r)
    unique = left.values.numel() + right.values.numel()
    if masspoints == "adjust" and bwcheck is None and (
            1-left.values.numel()/left.n >= .2 or 1-right.values.numel()/right.n >= .2):
        bwcheck = 10
    floors = tuple(0. if bwcheck is None else float(torch.sort(side.values.abs()).values[min(bwcheck,side.values.numel())-1])
                   for side in (left,right))
    pilot = pilot_bandwidth(x, kernel, unique if masspoints == "adjust" else None)
    if restrict:
        pilot = min(pilot, bw_max)
    pilot = max(pilot, *floors)
    range_l, range_r = max_l + 1e-8, abs(float(right.dx[-1])) + 1e-8

    def cap(value: float, limit: float) -> float:
        return min(value, limit) if restrict else value

    def step(o: int, nu: int, o_b: int, h_b: tuple[float, float], scaled: float):
        return (constants(left, o, nu, o_b, pilot, h_b[0], scaled, kernel, vce, matches, fuzzy),
                constants(right, o, nu, o_b, pilot, h_b[1], scaled, kernel, vce, matches, fuzzy))

    base = method.replace("cer", "mse")
    d_l, d_r = step(q + 1, q + 1, q + 2, (range_l, range_r), 0.0)
    results: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}

    def joint(sign: float, name: str) -> None:
        d = cap(_ratio(d_l.V + d_r.V, (d_r.B + sign * d_l.B) ** 2, 0.0, d_l.rate, name), bw_max)
        d = max(d, *floors)
        b_l, b_r = step(q, p + 1, q + 1, (d, d), scale)
        b = cap(_ratio(b_l.V + b_r.V, (b_r.B + sign * b_l.B) ** 2, scale * (b_l.R + b_r.R),
                       b_l.rate, name), bw_max)
        h_l, h_r = step(p, deriv, q, (b, b), scale)
        h = cap(_ratio(h_l.V + h_r.V, (h_r.B + sign * h_l.B) ** 2, scale * (h_l.R + h_r.R),
                       h_l.rate, name), bw_max)
        results[name] = ((h, h), (b, b))

    def two() -> None:
        d = (cap(_ratio(d_l.V, d_l.B ** 2, 0.0, d_l.rate, "msetwo"), max_l),
             cap(_ratio(d_r.V, d_r.B ** 2, 0.0, d_r.rate, "msetwo"), max_r))
        d = tuple(max(v,floors[i]) for i,v in enumerate(d))
        b_l, b_r = step(q, p + 1, q + 1, d, scale)
        b = (cap(_ratio(b_l.V, b_l.B ** 2, scale * b_l.R, b_l.rate, "msetwo"), max_l),
             cap(_ratio(b_r.V, b_r.B ** 2, scale * b_r.R, b_r.rate, "msetwo"), max_r))
        h_l, h_r = step(p, deriv, q, b, scale)
        h = (cap(_ratio(h_l.V, h_l.B ** 2, scale * h_l.R, h_l.rate, "msetwo"), max_l),
             cap(_ratio(h_r.V, h_r.B ** 2, scale * h_r.R, h_r.rate, "msetwo"), max_r))
        results["msetwo"] = (h, b)

    if base in {"mserd", "msecomb1", "msecomb2"}:
        joint(-1.0, "mserd")
    if base in {"msesum", "msecomb1", "msecomb2"}:
        joint(1.0, "msesum")
    if base in {"msetwo", "msecomb2"}:
        two()
    if base == "msecomb1":
        h = min(results["mserd"][0][0], results["msesum"][0][0])
        b = min(results["mserd"][1][0], results["msesum"][1][0])
        results[base] = ((h, h), (b, b))
    elif base == "msecomb2":
        def median(side: int, which: int) -> float:
            values = [results["mserd"][which][side], results["msesum"][which][side],
                      results["msetwo"][which][side]]
            return sorted(values)[1]
        results[base] = ((median(0, 0), median(1, 0)), (median(0, 1), median(1, 1)))
    h, b = results[base]
    if method.startswith("cer"):
        # rdbwselect: N^(-p/((3+p)(3+2p))), with the clusters of each side counted instead
        # of the observations when the variance is clustered
        n = left.n + right.n
        if left.cluster is not None:
            n = int(torch.unique(left.cluster).numel() + torch.unique(right.cluster).numel())
        factor = n ** (-(p / ((3 + p) * (3 + 2 * p))))
        h = (h[0] * factor, h[1] * factor)
    return Bandwidths(h, b, pilot, method, restrict)


def side_sample(dx: Tensor, y: Tensor, cluster: Tensor | None, mask: Tensor, weights=None, responses=1) -> rk.Side:
    return rk.make_side(dx[mask], y[mask], None if cluster is None else cluster[mask],
                        None if weights is None else weights[mask], responses)
