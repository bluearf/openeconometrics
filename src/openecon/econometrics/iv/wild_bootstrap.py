"""Wild restricted efficient (WRE) bootstrap for linear IV with one endogenous regressor.

Davidson and MacKinnon (2010, JBES 28) and its cluster extension as described
by Roodman, MacKinnon, Nielsen and Webb (2019, Stata Journal 19, section 6.1):

    y1 = beta y2 + X1 gamma + u1,      y2 = X1 pi1 + X2 pi2 + u2.

Under H0: beta = beta0 the structural equation is fitted by OLS of
y1 - beta0 y2 on X1 (gamma~, u1~); the reduced form is fitted efficiently by
OLS of y2 on [X1, X2, u1~] (pi~ are the X coefficients) and
U2~ = y2 - X pi~. Bootstrap samples use ONE multiplier per cluster (or per
observation) for both equations,

    y2* = X pi~ + U2~ v*,      y1* = beta0 y2* + X1 gamma~ + u1~ v*,

and the 2SLS t statistic with the same HC1/CR1 studentization is recomputed.
The Anderson--Rubin route regresses y1 - beta0 y2 on [X1, X2] and tests the X2
coefficients with a robust Wald statistic; its WRE bootstrap only perturbs the
left-hand side, (y1 - beta0 y2)* = X1 gamma~ + u1~ v*, i.e. a restricted wild
(cluster) bootstrap of that regression.

All quantities are computed on CPU float64 tensors from QR projections, with
the replications vectorised in bounded blocks; nothing per observation is
stored in the result.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, TableSet, column_list, make_spec, table
from openecon.econometrics.iv.common import role_designs, screen
from openecon.econometrics.postest import wild_multipliers as wm
from openecon.engines.distributions import chi2_sf, normal_sf

SOURCE = "https://doi.org/10.1198/jbes.2009.07221"
# Roodman, MacKinnon, Nielsen and Webb, Stata Journal 19(1), 4-60 (CREATES RP 2018-34 version).
CLUSTER_SOURCE = "https://repec.econ.au.dk/repec/creates/rp/18/rp18_34.pdf"
DEFAULT_MAX_WORK = 20_000_000_000
_BLOCK_BYTES = 64 * 1024**2
_EPS = torch.finfo(torch.float64).eps


# ---- geometry -------------------------------------------------------------------------


def _basis(matrix: Tensor, what: str) -> Tensor:
    """Orthonormal basis of the column space (full column rank required)."""
    if matrix.shape[1] == 0:
        return matrix.new_zeros((matrix.shape[0], 0))
    norms = torch.linalg.vector_norm(matrix, dim=0)
    if bool((norms == 0).any()):
        raise AnalysisError("singular_design", f"A column of the {what} is identically zero.")
    q, r = torch.linalg.qr(matrix / norms, mode="reduced")
    if bool((r.diagonal().abs() <= 1e-10).any()):
        raise AnalysisError("singular_design", f"The {what} are numerically rank deficient.")
    return q


def iv_geometry(*, data: Any, y: str, endog: Any, instruments: Any, x: Any, cluster: str | None,
                intercept: bool, missing: str) -> dict[str, Any]:
    """Sample, designs and projection bases of the single-endogenous IV model."""
    endogenous = [endog] if isinstance(endog, str) else column_list(endog, "endog")
    if len(endogenous) != 1:
        raise AnalysisError("unsupported_spec", "The WRE route supports exactly one endogenous regressor.")
    if cluster is not None and not isinstance(cluster, str):
        raise AnalysisError("invalid_cluster", "cluster must name one column (one-way clustering).")
    spec = make_spec("ivregress", outcome=y, predictors=column_list(x, "x"),
                     columns={"endogenous": endogenous, "instruments": column_list(instruments, "instruments")},
                     covariance="cluster" if cluster else "robust", cluster=cluster,
                     intercept=intercept, missing=missing)
    frame = ModelFrame(spec, data)
    blocks = screen(frame, *role_designs(frame, intercept=intercept), None)
    if blocks.x2.shape[1] != 1:
        raise AnalysisError("no_endogenous_regressors", "The endogenous regressor was omitted.")
    y1 = frame.numeric(y)
    y2 = blocks.x2[:, 0].contiguous()
    x1, x2 = blocks.x1, blocks.z2
    n, k1, l2 = len(y1), x1.shape[1], x2.shape[1]
    if cluster is not None:
        codes, groups = frame.codes([cluster])
    else:
        codes, groups = torch.arange(n), n
    if groups < 2:
        raise AnalysisError("insufficient_clusters", "At least two clusters are required.")
    if n <= k1 + l2 + 1:
        raise AnalysisError("insufficient_observations", "The model needs more observations than "
                            "instruments plus one.")
    q1 = _basis(x1, "exogenous regressors")
    q = _basis(torch.cat([x1, x2], dim=1), "instruments")
    return {"frame": frame, "blocks": blocks, "y1": y1, "y2": y2, "x1": x1, "x2": x2, "q1": q1,
            "q": q, "codes": codes.to(torch.int64), "groups": int(groups), "n": n, "k1": k1,
            "l2": l2, "cluster": cluster, "terms_exog": list(blocks.exog.terms),
            "endog": list(blocks.endog.terms)[0], "instruments": list(blocks.instr.terms)}


def _m1(g: dict[str, Any], values: Tensor) -> Tensor:
    """M_{X1} applied to the last axis of ``values`` [..., n]."""
    q1 = g["q1"]
    return values - (values @ q1) @ q1.T


def _px(g: dict[str, Any], values: Tensor) -> Tensor:
    q = g["q"]
    return (values @ q) @ q.T


def _factor(g: dict[str, Any], k: int) -> float:
    n = g["n"]
    if g["cluster"] is None:
        return n / (n - k)
    groups = g["groups"]
    return groups / (groups - 1) * (n - 1) / (n - k)


def _cluster_sums(g: dict[str, Any], values: Tensor) -> Tensor:
    """Sum the last axis of ``values`` [..., n] within clusters -> [..., G]."""
    shape = values.shape[:-1]
    flat = values.reshape(-1, g["n"])
    out = flat.new_zeros((flat.shape[0], g["groups"])).index_add_(1, g["codes"], flat)
    return out.reshape(*shape, g["groups"])


# ---- 2SLS t statistic and its WRE bootstrap ---------------------------------------------------


def tsls_t(g: dict[str, Any], y1: Tensor, y2: Tensor, beta0: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """2SLS beta, robust (HC1/CR1) standard error and t for beta0, batched over [..., n].

    By Frisch--Waugh with z = M1 P_X y2: beta = z'y1 / z'y2, u = M1(y1 - beta y2)
    and Var(beta) = c sum_g (sum_{i in g} z_i u_i)^2 / (z'z)^2, the beta element of
    the 2SLS sandwich with c = n/(n-k) (HC1) or G/(G-1)(n-1)/(n-k) (CR1),
    k = k1 + 1. Returns (beta, se, t, ok).
    """
    z = _m1(g, _px(g, y2))
    zz = (z * z).sum(-1)
    zy2 = (z * y2).sum(-1)
    beta = (z * y1).sum(-1) / zy2
    resid = _m1(g, y1 - beta[..., None] * y2)
    sums = _cluster_sums(g, z * resid)
    variance = _factor(g, g["k1"] + 1) * sums.square().sum(-1) / zz.square()
    se = variance.sqrt()
    t = (beta - beta0) / se
    ok = (zz > 1e-14 * (y2 * y2).sum(-1)) & (variance > 0) & torch.isfinite(t)
    return beta, se, t, ok


def wre_null(g: dict[str, Any], beta0: float) -> dict[str, Tensor]:
    """Restricted structural fit and efficient reduced form under beta = beta0."""
    restricted = g["y1"] - beta0 * g["y2"]
    u1 = _m1(g, restricted)                                  # y1 - beta0 y2 - X1 gamma~
    fitted1 = restricted - u1                                # X1 gamma~
    design = torch.cat([g["x1"], g["x2"], u1[:, None]], dim=1)
    norms = torch.linalg.vector_norm(design, dim=0)
    if bool((norms[-1] == 0)):
        # u1~ = 0 only when y1 - beta0 y2 lies in span(X1); the efficient term is then absent.
        design = design[:, :-1]
        norms = norms[:-1]
    q, r = torch.linalg.qr(design / norms, mode="reduced")
    if bool((r.diagonal().abs() <= 1e-12).any()):
        raise AnalysisError("singular_design", "The efficient reduced form [X1, X2, u1~] is "
                            "rank deficient; u1~ is (nearly) a linear combination of the instruments.")
    coef = torch.linalg.solve_triangular(r, (q.T @ g["y2"])[:, None], upper=True)[:, 0] / norms
    width = g["k1"] + g["l2"]
    fitted2 = torch.cat([g["x1"], g["x2"]], dim=1) @ coef[:width]
    u2 = g["y2"] - fitted2
    return {"u1": u1, "u2": u2, "fitted1": fitted1, "fitted2": fitted2,
            "delta": float(coef[width]) if coef.numel() > width else 0.0}


def wre_draws(g: dict[str, Any], beta0: float, multipliers: Tensor,
              null: dict[str, Tensor] | None = None) -> tuple[Tensor, Tensor]:
    """Bootstrap 2SLS t statistics for multipliers [B, G] (one per cluster/observation)."""
    null = wre_null(g, beta0) if null is None else null
    v = multipliers[:, g["codes"]]                           # [B, n]
    y2 = null["fitted2"] + null["u2"] * v
    y1 = beta0 * y2 + null["fitted1"] + null["u1"] * v
    _, _, t, ok = tsls_t(g, y1, y2, beta0)
    return t, ok


# ---- Anderson--Rubin and its restricted wild bootstrap ------------------------------------------


def ar_statistic(g: dict[str, Any], outcome: Tensor) -> tuple[Tensor, Tensor]:
    """Robust Wald of the X2 coefficients in OLS of ``outcome`` [..., n] on [X1, X2].

    With X2~ = M1 X2, b = (X2~'X2~)^-1 X2~' y, e = M_X y and
    V = c A [sum_g (X2~_g' e_g)(X2~_g' e_g)'] A, A = (X2~'X2~)^-1,
    c = n/(n-k) or G/(G-1)(n-1)/(n-k), k = k1 + l2. Returns (Wald, ok).
    """
    x2t = g["x2t"]
    a = g["x2t_inverse"]
    b = (outcome @ x2t) @ a                                   # [..., l2]
    resid = outcome - _px(g, outcome)
    scores = resid[..., None] * x2t                           # [..., n, l2]
    shape = scores.shape[:-2]
    flat = scores.reshape(-1, g["n"], g["l2"])
    sums = flat.new_zeros((flat.shape[0], g["groups"], g["l2"])).index_add_(1, g["codes"], flat)
    meat = sums.transpose(1, 2) @ sums
    cov = _factor(g, g["k1"] + g["l2"]) * a @ meat @ a
    cov = cov.reshape(*shape, g["l2"], g["l2"])
    factor, info = torch.linalg.cholesky_ex((cov + cov.transpose(-1, -2)) / 2)
    stat = (b[..., None, :] @ torch.cholesky_solve(b[..., :, None], factor))[..., 0, 0]
    ok = (info == 0) & torch.isfinite(stat) & (stat >= 0)
    return stat, ok


def _prepare_ar(g: dict[str, Any]) -> None:
    if "x2t" not in g:
        x2t = _m1(g, g["x2"].T).T.contiguous()
        g["x2t"] = x2t
        g["x2t_inverse"] = torch.linalg.inv(x2t.T @ x2t)


def ar_draws(g: dict[str, Any], outcome: Tensor, multipliers: Tensor) -> tuple[Tensor, Tensor]:
    """Restricted wild bootstrap AR statistics: outcome* = X1 gamma~ + u1~ v*."""
    u1 = _m1(g, outcome)
    v = multipliers[:, g["codes"]]
    # X1 gamma~ is annihilated by M1 and M_X, so only u1~ v* matters.
    return ar_statistic(g, u1 * v)


# ---- shared drivers ------------------------------------------------------------------------


def _block_rows(g: dict[str, Any]) -> int:
    per_draw = 8 * g["n"] * (8 + g["l2"]) + 8 * g["groups"] * (2 + g["l2"])
    return max(1, _BLOCK_BYTES // per_draw)


def _work(g: dict[str, Any], reps: int, nulls: int) -> float:
    width = g["k1"] + g["l2"] + 1
    return nulls * (reps + 1) * g["n"] * (4 * width + 3 * g["l2"] ** 2 + 12)


def _evaluate(g: dict[str, Any], method: str, beta0: float, kind: str, reps: int, seed: int,
              tail: str, multipliers: Tensor | None = None) -> dict[str, Any]:
    """Observed statistic and plus-one bootstrap p-value for one null value."""
    if method == "wald":
        beta, se, t, ok = tsls_t(g, g["y1"], g["y2"], beta0)
        if not bool(ok):
            raise AnalysisError("singular_covariance", "The 2SLS standard error is zero or undefined.")
        observed = float(t)
        null = wre_null(g, beta0)
    else:
        _prepare_ar(g)
        outcome = g["y1"] - beta0 * g["y2"] if beta0 is not None else g["y2"]
        stat, ok = ar_statistic(g, outcome)
        if not bool(ok):
            raise AnalysisError("singular_covariance", "The AR robust covariance is singular.")
        observed = float(stat)
    draws = []
    rows = _block_rows(g)
    source = (multipliers[i:i + rows] for i in range(0, len(multipliers), rows)) \
        if multipliers is not None else wm.blocks(kind, reps, g["groups"], seed, rows)
    for block in source:
        if method == "wald":
            values, good = wre_draws(g, beta0, block, null)
        else:
            values, good = ar_draws(g, outcome, block)
        failed = int((~good).sum())
        if failed:
            raise AnalysisError("bootstrap_failure", f"{failed} WRE bootstrap replications had a "
                                "zero or singular studentizing variance; no draw is dropped.")
        draws.append(values)
    draws = torch.cat(draws)
    probability = wm.tail_p_values(observed, draws, tail if method == "wald" else "upper")
    out = {"statistic": observed, "p_value": probability, "draws": draws}
    if method == "wald":
        out.update(beta=float(beta), std_error=float(se),
                   wald=observed ** 2, restricted=null)
    return out


def _settings(method, tail, kind, reps, seed):
    if method not in ("wald", "ar"):
        raise AnalysisError("invalid_option", "method must be 'wald' or 'ar'.")
    if tail not in ("symmetric", "equal_tail", "upper", "lower"):
        raise AnalysisError("invalid_option", "tail must be symmetric, equal_tail, upper or lower.")
    wm.check_kind(kind)
    reps = wm.check_count(reps, "reps", 99, 99_999)
    seed = wm.check_count(seed, "seed", 0, 2**63 - 1)
    return reps, seed


def _null_value(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AnalysisError("invalid_restrictions", "null must be one finite real value for beta.")
    return float(value)


def iv_wild_test(*, data: Any, y: str, endog: Any, instruments: Sequence[str], null: float,
                 x: Sequence[str] | None = None, method: str = "wald", cluster: str | None = None,
                 wild: str = "rademacher", reps: int = 999, seed: int = 0,
                 tail: str = "symmetric", intercept: bool = True, missing: str = "raise",
                 max_work: int = DEFAULT_MAX_WORK) -> TableSet:
    """WRE wild bootstrap test of H0: beta = ``null`` in a linear IV model (one endogenous regressor).

    ``method="wald"``: 2SLS t statistic studentized by the heteroskedasticity-
    robust (HC1, ``cluster=None``) or one-way cluster-robust (CR1) standard
    error; bootstrap samples follow Davidson and MacKinnon's wild restricted
    efficient DGP (restricted structural OLS, reduced form with the restricted
    residual as an extra regressor, one multiplier per observation or cluster
    shared by both equations). ``tail`` chooses the symmetric |t| (default),
    equal-tail, upper or lower p-value. ``method="ar"``: Anderson--Rubin robust
    Wald of the excluded instruments in the regression of y - null*endog on all
    instruments, bootstrapped by perturbing only that regression's restricted
    residuals (upper tail). The AR test jointly tests beta = null and the
    instrument exclusion restrictions and is robust to weak instruments.

    Multipliers: Rademacher, Webb six-point or Mammen, from a private seeded
    generator in replication-row blocks. P-values are plus-one Monte Carlo
    (1 + #exceedances)/(B + 1). The small-sample factor of the robust variance
    cancels between the observed and bootstrap statistics. Weights,
    multiway clustering, several endogenous regressors and LIML/GMM bootstrap
    estimators are outside this route. A draw with a zero or singular
    studentizing variance raises ``bootstrap_failure``.
    """
    reps, seed = _settings(method, tail, wild, reps, seed)
    beta0 = _null_value(null)
    g = iv_geometry(data=data, y=y, endog=endog, instruments=instruments, x=x, cluster=cluster,
                    intercept=intercept, missing=missing)
    work = _work(g, reps, 1)
    _check_work(work, max_work)
    out = _evaluate(g, method, beta0, wild, reps, seed, tail)
    row = {"method": method, "null": beta0, "statistic": out["statistic"],
           "bootstrap_p_value": out["p_value"], "mc_std_error": wm.mc_error(out["p_value"], reps)}
    if method == "wald":
        row.update(estimate=out["beta"], std_error=out["std_error"],
                   asymptotic_p_value=2 * normal_sf(abs(out["statistic"])), tail=tail)
        restricted = out["restricted"]
        extra = {"efficient_residual_coefficient": restricted["delta"]}
    else:
        row.update(df=g["l2"], asymptotic_p_value=chi2_sf(out["statistic"], g["l2"]), tail="upper")
        extra = {}
    return TableSet(
        {"test": table([row], title="Wild restricted efficient bootstrap test")},
        title="IV wild restricted efficient bootstrap", method=method,
        source=SOURCE, cluster_source=CLUSTER_SOURCE, outcome=y, endogenous=g["endog"],
        instruments=g["instruments"], exogenous=g["terms_exog"], observations=g["n"],
        cluster=cluster, clusters=g["groups"], multiplier=wild, draws=reps, failed_draws=0,
        seed=seed, rng="torch.Generator(cpu).manual_seed(seed); B x G multipliers in row blocks",
        first_multipliers=wm.draw(wild, 1, min(g["groups"], 50),
                                  torch.Generator().manual_seed(seed))[0].tolist(),
        studentization="CR1 G/(G-1)(n-1)/(n-k)" if cluster else "HC1 n/(n-k)",
        p_value_rule="plus-one (1 + #exceedances)/(B + 1)",
        bootstrap_statistics_summary={"mean": float(out["draws"].mean()),
                                      "std": float(out["draws"].std()),
                                      "min": float(out["draws"].min()),
                                      "max": float(out["draws"].max())},
        work=work, max_work=max_work, device="cpu", dtype="float64",
        sample_rows=len(g["frame"].positions),
        assumptions="independent clusters (or observations); valid instruments for the Wald route; "
                    "one endogenous regressor; no weights",
        **extra)


def _check_work(work: float, max_work: int) -> None:
    max_work = wm.check_count(max_work, "max_work", 1, 10**14)
    if work > max_work:
        raise AnalysisError("work_budget_exceeded", f"The bootstrap needs about {work:.3g} "
                            f"floating-point operations, above max_work={max_work}.")


def iv_wild_confidence_set(*, data: Any, y: str, endog: Any, instruments: Sequence[str],
                           x: Sequence[str] | None = None, method: str = "ar",
                           grid: Sequence[float] | None = None, points: int = 41,
                           width: float = 6.0, alpha: float = 0.05, refine: int = 12,
                           cluster: str | None = None, wild: str = "rademacher", reps: int = 999,
                           seed: int = 0, tail: str = "symmetric", intercept: bool = True,
                           missing: str = "raise", max_work: int = DEFAULT_MAX_WORK) -> TableSet:
    """Confidence set for beta by inverting WRE bootstrap tests, with bisection-refined boundaries.

    The same multipliers (one seed) are used for every null value, so p(beta0)
    is a deterministic function. It is evaluated on ``grid`` (strictly
    increasing finite values) or, by default, on ``points`` equally spaced
    values over 2SLS estimate +- ``width`` robust standard errors. Wherever two
    adjacent grid points differ in acceptance (p > alpha), the boundary is
    located by ``refine`` bisection steps; the result reports a bracket
    [rejected side, accepted side] whose width is the grid gap / 2**refine, so
    each reported crossing is certified to that accuracy. Additional crossings
    between two grid points with the same decision cannot be excluded; the
    grid gap is reported.

    Accepted components are listed in order (the set may be empty, an
    interval, or disjoint). A component reaching an end of the grid is open on
    that side. For the AR route the behaviour beyond the grid is decided by the
    limit test beta0 -> +-infinity (AR is scale invariant, so the limit is the
    wild bootstrap Wald test of the instruments in the first-stage regression
    of the endogenous regressor): if the limit is accepted the set is
    unbounded; otherwise the tail status remains 'not determined beyond grid'.
    For the Wald route tails beyond the grid are always 'not determined'.
    """
    reps, seed = _settings(method, tail, wild, reps, seed)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("invalid_option", "alpha must lie strictly between 0 and 1.")
    refine = wm.check_count(refine, "refine", 0, 40)
    g = iv_geometry(data=data, y=y, endog=endog, instruments=instruments, x=x, cluster=cluster,
                    intercept=intercept, missing=missing)
    if grid is None:
        points = wm.check_count(points, "points", 3, 401)
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not 0 < width < 1e6:
            raise AnalysisError("invalid_grid", "width must be a positive number of standard errors.")
        beta, se, _, ok = tsls_t(g, g["y1"], g["y2"], 0.0)
        if not bool(ok):
            raise AnalysisError("singular_covariance", "The 2SLS standard error is undefined.")
        centre, scale = float(beta), float(se)
        values = [centre + scale * width * (2 * i / (points - 1) - 1) for i in range(points)]
        grid_rule = f"2SLS estimate +- {width} robust standard errors, {points} points"
    else:
        if not isinstance(grid, (list, tuple)) or not 2 <= len(grid) <= 401:
            raise AnalysisError("invalid_grid", "Supply 2..401 strictly increasing finite grid values.")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in grid) or any(a >= b for a, b in zip(grid, grid[1:])):
            raise AnalysisError("invalid_grid", "The grid must be finite and strictly increasing.")
        values = [float(v) for v in grid]
        grid_rule = "caller grid"
    alpha = float(alpha)
    crossings_bound = len(values) - 1
    work = _work(g, reps, len(values) + refine * crossings_bound + 1)
    _check_work(work, max_work)
    if reps * g["groups"] > 20_000_000:
        raise AnalysisError("work_budget_exceeded", "Shared multipliers for test inversion are "
                            "limited to 20,000,000 entries (reps x clusters or observations).")
    multipliers = torch.cat(list(wm.blocks(wild, reps, g["groups"], seed, reps)))

    def p_value(beta0: float | None) -> float:
        return _evaluate(g, method, beta0, wild, reps, seed, tail, multipliers)["p_value"]

    rows = [{"null": value, "p_value": p_value(value)} for value in values]
    for row in rows:
        row["accepted"] = row["p_value"] > alpha
    boundaries = []
    for left, right in zip(rows, rows[1:]):
        if left["accepted"] == right["accepted"]:
            continue
        lo, hi = left["null"], right["null"]
        lo_accepted = left["accepted"]
        for _ in range(refine):
            mid = (lo + hi) / 2
            if (p_value(mid) > alpha) == lo_accepted:
                lo = mid
            else:
                hi = mid
        accepted_side, rejected_side = (lo, hi) if lo_accepted else (hi, lo)
        boundaries.append({"kind": "upper" if lo_accepted else "lower",
                           "accepted_side": accepted_side, "rejected_side": rejected_side,
                           "bracket_width": abs(hi - lo)})
    limit = None
    if method == "ar":
        limit = _evaluate(g, "ar", None, wild, reps, seed, tail, multipliers)
    components, start = [], None
    for i, row in enumerate(rows + [{"accepted": False}]):
        if row["accepted"] and start is None:
            start = i
        if not row["accepted"] and start is not None:
            stop = i - 1
            lower = next((b for b in boundaries if b["kind"] == "lower"
                          and rows[start - 1]["null"] <= b["accepted_side"] <= rows[start]["null"]),
                         None) if start > 0 else None
            upper = next((b for b in boundaries if b["kind"] == "upper"
                          and rows[stop]["null"] <= b["accepted_side"] <= rows[stop + 1]["null"]),
                         None) if stop < len(rows) - 1 else None

            def tail_status(edge):
                if method == "ar" and limit is not None and limit["p_value"] > alpha:
                    return "unbounded (AR limit test at infinity accepts)"
                return f"open at grid {edge}; not determined beyond grid"

            components.append({
                "lower": lower["accepted_side"] if lower else None,
                "lower_bracket": [lower["rejected_side"], lower["accepted_side"]] if lower else None,
                "lower_status": "certified bracket" if lower else tail_status("start"),
                "upper": upper["accepted_side"] if upper else None,
                "upper_bracket": [upper["accepted_side"], upper["rejected_side"]] if upper else None,
                "upper_status": "certified bracket" if upper else tail_status("end"),
            })
            start = None
    if not components:
        empty_status = "empty on grid" + (
            "; AR limit at infinity also rejects" if limit is not None and limit["p_value"] <= alpha
            else "; AR limit at infinity accepts, so the set is unbounded outside the grid"
            if limit is not None else "")
    else:
        empty_status = None
    return TableSet(
        {"grid": table(rows, title="Bootstrap p-values on the grid"),
         "components": table(components if components else [{"lower": None, "upper": None}],
                             title="Accepted components")},
        title="IV wild bootstrap confidence set", method=method, alpha=alpha,
        grid_rule=grid_rule, maximum_grid_gap=max(b - a for a, b in zip(values, values[1:])),
        refine_steps=refine, boundaries=boundaries,
        ar_limit_p_value=limit["p_value"] if limit is not None else None,
        ar_limit_statistic=limit["statistic"] if limit is not None else None,
        empty=empty_status, multiplier=wild, draws=reps, seed=seed, shared_multipliers=True,
        cluster=cluster, clusters=g["groups"], observations=g["n"], work=work, max_work=max_work,
        source=SOURCE, cluster_source=CLUSTER_SOURCE,
        note="Crossings between same-decision grid points narrower than the grid gap are not excluded.")
