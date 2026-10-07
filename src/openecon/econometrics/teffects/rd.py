"""Sharp and fuzzy regression discontinuity: community rdrobust (CCT 2014, CCF 2018).

On each side of the cutoff c a local polynomial of order p with kernel weights
K((x - c)/h)/h estimates the limit of E[y | x] at c (and of E[t | x] in a fuzzy
design). With R_p the polynomial design, W_h, W_b the weights at the main and
bias bandwidths, G_p = R_p'W_h R_p, G_q = R_q'W_b R_q (order q = p + 1 by
default) and L = R_p'W_h ((x - c)/h)^(p+1):

    beta_p  = G_p^-1 R_p'W_h y                         conventional
    Q       = R_p W_h - h^(p+1) (W_b R_q G_q^-1 e_{p+1}) L'
    beta_bc = G_p^-1 Q'y                               bias corrected

tau = beta_+[0] - beta_-[0] (fuzzy: the ratio of the outcome and treatment
jumps, bias corrected through the linearization s = (1/tau_T, -tau_Y/tau_T^2)).
Variances use the sandwich G_p^-1 M G_p^-1 with M built from R_p W_h
(conventional) or Q (robust) and the nearest-neighbour or HC residuals of
``rdkernels``. The three reported rows are rdrobust's:

    Conventional    tau_cl  with se_cl
    Bias-corrected  tau_bc  with se_cl
    Robust          tau_bc  with se_rb    (robust bias-corrected inference)

z inference; the confidence intervals of the Conventional and Robust rows are
rdrobust's conventional and robust intervals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, kernel_call, make_spec
from openecon.econometrics.teffects import rdbandwidth as bw
from openecon.econometrics.teffects import rdkernels as rk
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import least_squares
from openecon.models import ModelSpec, ResultBundle


@dataclass
class _SideFit:
    beta_p: Tensor             # [p+1, m]
    beta_bc: Tensor            # [p+1, m]
    inverse: Tensor            # G_p^-1
    rows_h: Tensor             # R_p W_h
    q_rows: Tensor             # Q
    res_h: Tensor
    res_b: Tensor
    cluster: Tensor | None
    n_h: int
    n_b: int
    residual_gram: Tensor


def _distinct_positive(side: rk.Side, part: slice, weights: Tensor) -> int:
    blocks = side.block[part][weights > 0]
    return int(torch.unique(blocks).numel())


def _fit_side(side: rk.Side, h: float, b: float, p: int, q: int, kernel: str, vce: str,
              matches: int, name: str) -> _SideFit:
    w_h = rk.kernel_weights(side.dx, h, kernel)
    w_b = rk.kernel_weights(side.dx, b, kernel)
    if side.weights is not None:
        w_h, w_b = w_h * side.weights, w_b * side.weights
    positive = torch.nonzero((w_h > 0) | (w_b > 0)).flatten()
    if positive.numel() == 0:
        raise KernelError("insufficient_observations", f"No observation on the {name} side lies "
                          "within the bandwidth.")
    part = slice(int(positive[0]), int(positive[-1]) + 1)
    weights_h, weights_b = w_h[part], w_b[part]
    for order, weights, label in ((p, weights_h, "h"), (q, weights_b, "b")):
        if _distinct_positive(side, part, weights) <= order:
            raise KernelError("insufficient_observations",
                              f"The {name} side has too few distinct running-variable values "
                              f"within bandwidth {label} for a polynomial of order {order}; "
                              "choose a larger bandwidth or a lower order.")
    dx, y = side.dx[part], side.y[part]
    design_q = rk.powers(dx, q)
    design_p = design_q[:, :p + 1]
    fit_p = least_squares(design_p, y, weights_h, drop_collinear=False, tol=0.0)
    fit_q = least_squares(design_q, y, weights_b, drop_collinear=False, tol=0.0)
    rows_h = design_p * weights_h[:, None]
    moment = rows_h.T @ (dx / h) ** (p + 1)
    projection = weights_b * (design_q @ fit_q.xtx_inv[:, p + 1])
    q_rows = rows_h - h ** (p + 1) * projection[:, None] * moment[None, :]
    beta_bc = fit_p.xtx_inv @ (q_rows.T @ y)
    local_p = rk.LocalFit(part, design_p, weights_h, fit_p.xtx_inv, fit_p.beta)
    local_q = rk.LocalFit(part, design_q, weights_b, fit_q.xtx_inv, fit_q.beta)
    res_h = rk.residuals(side, local_p, vce, matches, None, p + 1)
    res_b = res_h if vce == "nn" else rk.residuals(side, local_q, vce, matches, local_p, q + 1)
    cluster = None if side.cluster is None else side.cluster[part]
    return _SideFit(fit_p.beta, beta_bc, fit_p.xtx_inv, rows_h, q_rows, res_h, res_b, cluster,
                    int((weights_h > 0).sum()), int((weights_b > 0).sum()),
                    fit_p.resid.T @ (weights_h[:, None] * fit_p.resid))


def _variance(fit: _SideFit, combine: Tensor, deriv=0) -> tuple[float, float]:
    conventional = fit.inverse @ rk.meat(fit.rows_h, fit.res_h, combine, fit.cluster) @ fit.inverse
    robust = fit.inverse @ rk.meat(fit.q_rows, fit.res_b, combine, fit.cluster) @ fit.inverse
    scale = math.factorial(deriv) ** 2
    return scale * float(conventional[deriv, deriv]), scale * float(robust[deriv, deriv])


def _pair(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        pair = (float(value), float(value))
    elif isinstance(value, list) and len(value) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        pair = (float(value[0]), float(value[1]))
    else:
        raise AnalysisError("invalid_spec", f"{name} must be a positive number or a list "
                            "[left, right] of two positive numbers.")
    if not all(math.isfinite(v) and v > 0 for v in pair):
        raise AnalysisError("invalid_spec", f"{name} must be a positive, finite bandwidth in "
                            "the units of the running variable.")
    return pair


def fit_rdrobust(spec: ModelSpec, data: Any) -> ResultBundle:
    frame = ModelFrame(spec, data)
    running = frame.role("running")[0]
    fuzzy = frame.role("fuzzy")
    y = frame.numeric(spec.outcome)
    x = frame.numeric(running)
    c = float(frame.option("cutoff"))
    p = int(frame.option("p"))
    q = frame.option("q")
    q = p + 1 if q is None else int(q)
    deriv = int(frame.option("deriv"))
    if deriv > p:
        raise AnalysisError("invalid_spec", "deriv must not exceed the local polynomial order p.")
    if q <= p:
        raise AnalysisError("invalid_spec", "q (the order of the bias-correction polynomial) must "
                            "exceed p.")
    kernel, vce = frame.option("kernel"), frame.option("vce")
    matches = int(frame.option("nnmatch"))
    left_mask, right_mask = x < c, x >= c
    if int(left_mask.sum()) < 2 or int(right_mask.sum()) < 2:
        raise AnalysisError("invalid_cutoff", f"The cutoff {c:g} must leave observations on both "
                            "sides of it (x < cutoff: control; x >= cutoff: treated).")
    columns = [y]
    if fuzzy:
        columns.append(frame.numeric(fuzzy[0]))
    responses = len(columns)
    covariates = frame.role("covariates")
    columns.extend(frame.numeric(name) for name in covariates)
    outcome = torch.stack(columns, dim=1)
    cluster = None
    if spec.covariance == "cluster":
        dimensions = frame.cluster_dimensions()
        if len(dimensions) != 1:
            raise AnalysisError("cluster_dimensions", "rdrobust clusters on one column; give a "
                                "single cluster variable.")
        cluster = dimensions[0][0]
    dx = x - c
    weights = frame.weights()
    left = bw.side_sample(dx, outcome, cluster, left_mask, weights, responses)
    right = bw.side_sample(dx, outcome, cluster, right_mask, weights, responses)
    h = _pair(frame.option("h"), "h")
    b = _pair(frame.option("b"), "b")
    rho = frame.option("rho")
    selection = None
    fuzzy_bw = bool(fuzzy) and not bool(frame.option("sharpbw"))
    perfect = False
    if fuzzy_bw and any(bool((side.y[:, 1] == side.y[0, 1]).all()) for side in (left, right)):
        # rdbwselect's perf_comp: perfect compliance on one side selects the sharp bandwidth
        fuzzy_bw, perfect = False, True
    if h is None:
        if b is not None or rho is not None:
            raise AnalysisError("invalid_spec", "b and rho need an explicit main bandwidth h.")
        selection = kernel_call(
            bw.select, left, right, x, p=p, q=q, kernel=kernel, vce=vce, matches=matches,
            scale=float(frame.option("scaleregul")), method=frame.option("bwselect"),
            restrict=bool(frame.option("bwrestrict")), fuzzy=fuzzy_bw, deriv=deriv,
            masspoints=frame.option("masspoints"), bwcheck=frame.option("bwcheck"))
        h, b = selection.h, selection.b
    elif b is None:
        factor = 1.0 if rho is None else float(rho)
        b = (h[0] / factor, h[1] / factor)
    elif rho is not None:
        raise AnalysisError("invalid_spec", "Give b or rho, not both.")
    fits = [kernel_call(_fit_side, side, h[i], b[i], p, q, kernel, vce, matches, name)
            for i, (side, name) in enumerate(((left, "left"), (right, "right")))]
    gram = fits[0].residual_gram + fits[1].residual_gram
    adjustment = kernel_call(rk.covariate_adjustment, gram, responses)
    jump_p = math.factorial(deriv) * (fits[1].beta_p[deriv] - fits[0].beta_p[deriv])
    jump_bc = math.factorial(deriv) * (fits[1].beta_bc[deriv] - fits[0].beta_bc[deriv])
    adjusted_p, adjusted_bc = jump_p @ adjustment, jump_bc @ adjustment
    extra: dict[str, Any] = {"covariates": covariates, "covariate_coefficients": (-adjustment[responses:]).tolist(),
                            "derivative": deriv, "masspoints": frame.option("masspoints"),
                            "n_unique_left": left.values.numel(), "n_unique_right": right.values.numel(),
                            "weighted": weights is not None, "adjustment_matrix": adjustment.tolist()}
    if fuzzy:
        tau_y, tau_t = float(adjusted_p[0]), float(adjusted_p[1])
        if abs(tau_t) < 1e-12:
            raise AnalysisError("weak_first_stage", "The treatment probability does not jump at "
                                "the cutoff (first-stage jump is zero); the fuzzy RD estimand is "
                                "not identified.")
        combine = adjustment @ torch.tensor([1 / tau_t, -tau_y / tau_t ** 2], dtype=torch.float64)
        tau_cl = tau_y / tau_t
        tau_bc = tau_cl - float(combine @ (jump_p - jump_bc))
        first = adjustment[:, 1]
        first_cl = sum(_variance(fit, first, deriv)[0] for fit in fits)
        first_rb = sum(_variance(fit, first, deriv)[1] for fit in fits)
        extra["first_stage"] = {"conventional": tau_t, "bias_corrected": float(adjusted_bc[1]),
                                "std_error": math.sqrt(first_cl),
                                "robust_std_error": math.sqrt(first_rb)}
        extra["reduced_form"] = {"conventional": tau_y, "bias_corrected": float(adjusted_bc[0])}
    else:
        combine = adjustment[:, 0]
        tau_cl, tau_bc = float(adjusted_p[0]), float(adjusted_bc[0])
    # Fuzzy derivative estimands are ratios: the factorial in the numerator and
    # denominator cancels. The delta-method coefficients above already include it.
    variance_deriv = deriv
    var_cl = sum(_variance(fit, combine, variance_deriv)[0] for fit in fits)
    var_rb = sum(_variance(fit, combine, variance_deriv)[1] for fit in fits)
    if not (var_cl > 0 and var_rb > 0):
        raise AnalysisError("invalid_covariance", "The local-polynomial variance is not positive.")
    if selection is not None and fuzzy:
        extra["bandwidth_equation"] = "fuzzy" if fuzzy_bw else "outcome (sharp)"
        if perfect:
            extra["bandwidth_equation"] += ": perfect compliance on one side"
    return _result(frame, fits, (left, right), h, b, p, q, c, tau_cl, tau_bc, var_cl, var_rb,
                   selection, extra, kernel, vce, matches, bool(fuzzy))


def _result(frame: ModelFrame, fits: list[_SideFit], sides: tuple[rk.Side, rk.Side],
            h: tuple[float, float], b: tuple[float, float], p: int, q: int, c: float,
            tau_cl: float, tau_bc: float, var_cl: float, var_rb: float, selection, extra: dict,
            kernel: str, vce: str, matches: int, fuzzy: bool) -> ResultBundle:
    params = torch.tensor([tau_cl, tau_bc, tau_bc], dtype=torch.float64)
    covariance = torch.diag(torch.tensor([var_cl, var_cl, var_rb], dtype=torch.float64))
    metrics = {
        "h_left": h[0], "h_right": h[1], "b_left": b[0], "b_right": b[1],
        "n_left": sides[0].n, "n_right": sides[1].n,
        "n_h_left": fits[0].n_h, "n_h_right": fits[1].n_h,
        "n_b_left": fits[0].n_b, "n_b_right": fits[1].n_b,
        "p": p, "q": q, "cutoff": c,
    }
    level = 1 - frame.spec.alpha
    extra.update({
        "cutoff_evaluation": {"version":1,"point":"cutoff, saved derivative","derivative":extra["derivative"],
            "adjusted_covariates":bool(extra["covariates"]),
            "sides": {name:{"conventional_variance":_variance(fit,torch.tensor(extra["adjustment_matrix"],dtype=torch.float64)[:,0],extra["derivative"])[0],
                             "robust_variance":_variance(fit,torch.tensor(extra["adjustment_matrix"],dtype=torch.float64)[:,0],extra["derivative"])[1]}
                      for name,fit in zip(("left","right"),fits,strict=True)}},
        "kernel": kernel, "vce": vce, "nnmatch": matches if vce == "nn" else None,
        "bandwidth_selection": None if selection is None else {
            "method": selection.method, "pilot": selection.pilot,
            "bwrestrict": selection.restricted},
        "rows": ["Conventional", "Bias-corrected", "Robust"],
        "side_estimates": {name: {"conventional": float(math.factorial(extra["derivative"]) * (fit.beta_p[extra["derivative"]] @ torch.tensor(extra["adjustment_matrix"], dtype=torch.float64))[0]),
                                  "bias_corrected": float(math.factorial(extra["derivative"]) * (fit.beta_bc[extra["derivative"]] @ torch.tensor(extra["adjustment_matrix"], dtype=torch.float64))[0])}
                           for name, fit in zip(("left", "right"), fits, strict=True)},
        "design": "fuzzy" if fuzzy else "sharp", "confidence_level": level,
    })
    correction = (f"{'nearest-neighbour (nnmatch=' + str(matches) + ')' if vce == 'nn' else vce} "
                  "residual sandwich of the local polynomials; conventional (rows 1-2) and robust "
                  "bias-corrected (row 3) variances")
    if frame.spec.covariance == "cluster":
        correction += ", cluster sums with ((n-1)/(n-k)) (G/(G-1)) per side"
    info = {"correction": correction, "small_sample_correction": None, "df_inference": None,
            "variance_method": "rdrobust local-polynomial sandwich"}
    title = "Regression discontinuity: " + ("fuzzy" if fuzzy else "sharp") + " local polynomial"
    return build_result(
        frame, terms=["Conventional", "Bias-corrected", "Robust"], params=params,
        covariance=covariance, title=title, use_t=False, metrics=metrics,
        solver="local_polynomial_qr", inference=info, extra=extra, nobs=frame.n,
        provenance={"covariance_matrix": "diagonal: the three rows are alternative estimates "
                                         "of one parameter; their covariances are not reported",
                    "running_variable": frame.role("running")[0]},
    )


def rdrobust(*, data: Any, y: str, running: str, cutoff: float = 0.0, fuzzy: str | None = None,
             covariates: Sequence[str] | None = None, p: int = 1, q: int | None = None,
             weights: str | None = None, deriv: int = 0, masspoints: str = "adjust", bwcheck: int | None = None,
             kernel: str = "triangular", bwselect: str = "mserd",
             h: float | Sequence[float] | None = None, b: float | Sequence[float] | None = None,
             rho: float | None = None, vce: str = "nn", nnmatch: int = 3,
             scaleregul: float = 1.0, bwrestrict: bool = True, sharpbw: bool = False,
             covariance: str | None = None,
             cluster: str | None = None, missing: str = "raise",
             alpha: float = 0.05) -> ResultBundle:
    """Regression discontinuity with robust bias-corrected inference (community ``rdrobust``).

    Observations with running variable x >= ``cutoff`` are treated (sharp
    design) or have a jump in the treatment probability (``fuzzy``: the
    treatment column). The estimand is the jump of E[y | x] at the cutoff
    (sharp) or the ratio of the jumps of E[y | x] and E[t | x] (fuzzy, a local
    average treatment effect). On each side a local polynomial of order ``p``
    (default 1, local linear) is fitted by weighted least squares with kernel
    weights K((x - c)/h)/h; the bias of the order-p fit is estimated with a
    local polynomial of order ``q`` (default p + 1) and bandwidth b.

    Bandwidths: ``h`` and ``b`` (a number or ``[left, right]``); ``rho`` sets
    b = h/rho (default b = h when only h is given). Without ``h`` they are
    selected by ``bwselect`` (Calonico, Cattaneo and Titiunik 2014 plug-in with
    regularization ``scaleregul``): ``'mserd'`` (default; one MSE-optimal
    bandwidth for both sides), ``'msetwo'`` (one per side), ``'msesum'``,
    ``'msecomb1'``, ``'msecomb2'`` and their coverage-error-rate versions
    ``'cerrd'``, ``'certwo'``, ``'cersum'``, ``'cercomb1'``, ``'cercomb2'``.
    ``bwrestrict`` caps bandwidths at the range of the data. Fuzzy designs
    select the bandwidth from the linearized ratio of the outcome and
    treatment jumps (rdrobust's default); ``sharpbw=True`` uses the outcome
    (reduced-form) equation alone, as does perfect compliance on one side.

    Kernels: ``'triangular'`` (default), ``'epanechnikov'``, ``'uniform'``.
    Variance: ``vce='nn'`` (default; nearest-neighbour residuals with
    ``nnmatch`` = 3 neighbours) or ``'hc0'``, ``'hc1'``, ``'hc2'``, ``'hc3'``;
    ``covariance='cluster'`` with ``cluster`` sums the scores by cluster
    (rdrobust's ``nncluster``/``cluster``). ``covariates`` uses the common
    polynomial-residualized covariate slope; collinear covariates are rejected.
    ``weights`` supplies nonnegative observation weights multiplying the kernel.
    ``deriv`` estimates the derivative jump (1: regression kink), with deriv<=p.
    ``masspoints='adjust'`` uses unique-value pilot counts and a preliminary
    bandwidth floor; ``'check'`` records unique counts; ``'off'`` disables it.
    ``bwcheck`` supplies the minimum pilot unique-value count explicitly.
    Covariates, weights and derivatives currently require a resident DataFrame;
    Dataset replay retains the unadjusted estimand and mass-point options.
    The separate ``rddensity`` procedure tests running-variable manipulation.

    The result has three terms, rdrobust's rows: ``Conventional`` (estimate and
    standard error of the order-p fit), ``Bias-corrected`` (bias-corrected
    estimate, conventional standard error) and ``Robust`` (bias-corrected
    estimate with the robust standard error: robust bias-corrected inference,
    the recommended confidence interval). Inference is z. ``metrics`` hold
    the bandwidths (h_left, h_right, b_left, b_right), the sample sizes on each
    side and within each bandwidth, p, q and the cutoff; ``extra`` the side
    limits, the bandwidth-selection record and, for fuzzy designs, the first
    stage and reduced form.

    Stata/R: ``rdrobust y x, c(0) p(1) kernel(triangular) bwselect(mserd)
    vce(nn 3)``; fuzzy: ``rdrobust y x, fuzzy(t)``.

    Example::

        import openecon as oe
        rd = oe.rdrobust(data=df, y="vote", running="margin", cutoff=0.0)
        print(rd.summary())          # Conventional / Bias-corrected / Robust rows
        rd.metrics["h_left"], rd.extra["side_estimates"]
    """
    from openecon.analysis import fit

    def number_or_pair(value):
        return list(value) if isinstance(value, (list, tuple)) else value

    spec = make_spec(
        "rdrobust", outcome=y, covariance=covariance, cluster=cluster, missing=missing,
        alpha=alpha, columns={"running": running, "fuzzy": fuzzy, "covariates": list(covariates or [])},
        weights=weights, weight_type="aweight" if weights is not None else None,
        options={"cutoff": float(cutoff), "p": p, "q": q, "kernel": kernel,
                 "bwselect": bwselect, "h": number_or_pair(h), "b": number_or_pair(b),
                 "rho": rho, "vce": vce, "nnmatch": nnmatch, "scaleregul": scaleregul,
                 "bwrestrict": bwrestrict, "sharpbw": sharpbw, "deriv": deriv,
                 "masspoints": masspoints, "bwcheck": bwcheck},
    )
    return fit(spec, data=data)
