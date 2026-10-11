"""Shift-share (Bartik) designs with Adao-Kolesar-Morales (2019) AKM and AKM0 inference.

Inputs are long-format tables: region-level ``data`` (one row per ``unit``),
``shares`` (``unit``, ``sector``, ``share``) and ``shocks`` (``sector``, ``shock``).
The shift-share variable is ``X_i = sum_s w_is g_s``; absent (unit, sector) pairs
are zero shares. With controls ``Z`` (the constant and ``x``) and optional region
weights, the estimators and standard errors transcribe the authors' R package
ShiftShareSE (``reg_ss.fit`` in R/ols.R and ``ivreg_ss.fit`` in R/iv.R,
https://github.com/kolesarm/ShiftShareSE; Adao, Kolesar and Morales 2019, QJE
134(4):1949-2010):

    OLS (no ``endog``): b = coefficient of X in y on [X Z].
    IV  (``endog``):    b = (coef of X in y on [X Z]) / (coef of X in endog on [X Z]).

    ddX, ddY: residuals on Z;  hX = coefficients of ddX on the share matrix W;
    RX = sum w ddX^2 (OLS) or sum w ddY2 ddX (IV); e = residuals of the fit.

    homoskedastic  OLS sqrt(sum w e^2/(n-p) / RX);  IV sqrt(sum w e^2/n)/(sqrt(sum w ddX^2) |pi|)
    ehw            OLS sqrt(n/(n-p) sum u^2)/RX;    IV sqrt(sum u^2)/|RX|,   u = w e ddX
    region_cluster OLS times G/(G-1)(n-1)/(n-p) on cluster sums;  IV cluster sums, no factor
    akm            sqrt(sum_s cR_s^2)/|RX|,  cR_s = hX_s * sum_i W_is w_i e_i
                   (summed within shock clusters first when ``shock_cluster`` is given)
    akm0           null-imposed: cR0 with residuals ddY - ddX b0 (IV: ddY1 - ddY2 b0);
                   p-value from sqrt(sum cR0^2)/|RX|; confidence set by inverting the
                   test (Q = RX^2/z^2 - sum cW^2, cW_s = hX_s sum_i W_is w_i ddX_i
                   [IV: ddY2_i]): a bounded interval when Q > 0, otherwise the
                   complement of an interval or the real line.

p-values and intervals use the standard normal. Collinear share columns are
dropped left to right in the order of the ``shocks`` table (ShiftShareSE drops
them by pivoted QR) and reported. ``effective_shock_count`` is the inverse
Herfindahl index ``1/sum_s s_s^2`` of the exposure weights
``s_s = sum_i e_i w_is / sum_i e_i sum_k w_ik`` (e: region weights), the effective
sample size of Borusyak, Hull and Jaravel (2022). When share sums vary across
regions and are not spanned by the controls a note recommends controlling for
them (incomplete-shares case).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, column_list, kernel_call, table
from openecon.engines.distributions import normal_isf, normal_sf
from openecon.engines.linalg import least_squares
from openecon.resources import plan_workspace

SOURCE = "https://github.com/kolesarm/ShiftShareSE (R/ols.R reg_ss.fit, R/iv.R ivreg_ss.fit)"


def _frame(value: Any, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    try:
        return pd.DataFrame(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("invalid_data", f"{name} must be a table.") from exc


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"{name} lacks column(s): {', '.join(absent)}.")


def _numeric(series: pd.Series, name: str) -> Tensor:
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise AnalysisError("non_numeric_column", f"Column '{name}' must be numeric.")
    values = torch.as_tensor(series.to_numpy(dtype="float64", na_value=float("nan")).copy(),
                             dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise AnalysisError("non_finite_values", f"Column '{name}' contains missing or "
                            "non-finite values.")
    return values


def _shares_matrix(units: pd.Series, shares: pd.DataFrame, shocks: pd.DataFrame, *, unit: str,
                   sector: str, share: str, shock: str, max_cells: int):
    _require(shares, [unit, sector, share], "shares")
    _require(shocks, [sector, shock], "shocks")
    if shares[[unit, sector]].isna().any().any() or shocks[sector].isna().any():
        raise AnalysisError("missing_values", "Unit and sector identifiers must not be missing.")
    if shares.duplicated([unit, sector]).any():
        raise AnalysisError("duplicate_shares", "shares has repeated (unit, sector) pairs.")
    if shocks[sector].duplicated().any():
        raise AnalysisError("duplicate_shocks", "shocks has repeated sectors.")
    weight = _numeric(shares[share], share)
    if bool((weight < 0).any()):
        raise AnalysisError("negative_shares", "Exposure shares must be non-negative.")
    used = shares[shares[unit].isin(set(units))]
    unmatched = int(len(shares) - len(used))
    missing_units = sorted(set(units) - set(used[unit]), key=str)
    if missing_units:
        raise AnalysisError("missing_shares", f"{len(missing_units)} unit(s) of the estimation "
                            "sample have no share rows (for example "
                            f"{missing_units[0]!r}); give explicit zero shares if intended.")
    order = [s for s in shocks[sector].tolist() if s in set(used[sector])]
    absent = set(used[sector]) - set(order)
    if absent:
        raise AnalysisError("missing_shocks", f"{len(absent)} sector(s) with shares have no shock "
                            f"(for example {sorted(absent, key=str)[0]!r}).")
    n, s = len(units), len(order)
    if n * s > max_cells:
        raise AnalysisError("resource_limit", f"The dense share matrix has {n * s:,} cells, above "
                            f"max_cells={max_cells:,}.")
    plan_workspace("shift-share dense share matrix", {"share_matrix": n * s * 8 * 4})
    row = pd.Index(units).get_indexer(used[unit])
    column = pd.Index(order).get_indexer(used[sector])
    matrix = torch.zeros((n, s), dtype=torch.float64)
    values = _numeric(used[share], share)
    matrix[torch.as_tensor(row), torch.as_tensor(column)] = values
    keyed = shocks.set_index(sector)
    g = _numeric(keyed.loc[order, shock], shock)
    extra_shocks = int(len(shocks) - s)
    return matrix, g, order, unmatched, extra_shocks, keyed


def shift_share_instrument(*, shares: Any, shocks: Any, unit: str = "unit",
                           sector: str = "sector", share: str = "share",
                           shock: str = "shock"):
    """Shift-share (Bartik) variable sum_s share_is * shock_s per unit from long tables.

    Returns a table with ``unit``, ``shift_share`` and ``share_sum`` (one row per unit
    in first-appearance order of ``shares``). Missing (unit, sector) pairs are zero.
    """
    shares, shocks = _frame(shares, "shares"), _frame(shocks, "shocks")
    _require(shares, [unit], "shares")
    units = pd.Series(pd.unique(shares[unit]))
    matrix, g, order, _, extra, _ = _shares_matrix(
        units, shares, shocks, unit=unit, sector=sector, share=share, shock=shock,
        max_cells=50_000_000)
    return table({unit: units.tolist(), "shift_share": (matrix @ g).tolist(),
                  "share_sum": matrix.sum(1).tolist()},
                  n_units=len(units), n_sectors=len(order), unused_shocks=extra,
                  definition="X_i = sum_s share_is * shock_s")


def _fit(design: Tensor, target: Tensor, w: Tensor | None):
    return kernel_call(least_squares, design, target, w, drop_collinear=True)


def shift_share_iv(*, data: Any, y: str, unit: str, shares: Any, shocks: Any,
                   endog: str | None = None, x: Sequence[str] | None = None,
                   sector: str = "sector", share: str = "share", shock: str = "shock",
                   weights: str | None = None, cluster: str | None = None,
                   shock_cluster: str | None = None, beta0: float = 0.0, alpha: float = 0.05,
                   intercept: bool = True, missing: str = "raise",
                   max_cells: int = 50_000_000) -> TableSet:
    """Shift-share regression or IV with AKM/AKM0 shock-level inference (ShiftShareSE).

    ``endog=None``: OLS of ``y`` on the shift-share variable; otherwise 2SLS of ``y`` on
    ``endog`` instrumented by it. Reports homoskedastic, EHW, region-cluster (with
    ``cluster``), AKM and null-imposed AKM0 standard errors, p-values and intervals.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("invalid_option", "alpha must lie strictly between 0 and 1.")
    if isinstance(beta0, bool) or not isinstance(beta0, (int, float)) or not math.isfinite(beta0):
        raise AnalysisError("invalid_option", "beta0 must be a finite number.")
    if missing not in ("raise", "drop"):
        raise AnalysisError("invalid_option", "missing must be 'raise' or 'drop'.")
    if isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells < 1:
        raise AnalysisError("invalid_option", "max_cells must be a positive integer.")
    frame = _frame(data, "data")
    shares, shocks = _frame(shares, "shares"), _frame(shocks, "shocks")
    controls = column_list(x, "x")
    columns = [unit, y, *([endog] if endog else []), *controls,
               *([weights] if weights else []), *([cluster] if cluster else [])]
    if len(set(columns)) != len(columns):
        raise AnalysisError("invalid_spec", "unit, y, endog, x, weights and cluster must be "
                            "distinct columns.")
    _require(frame, columns, "data")
    selected = frame.loc[:, columns].reset_index(drop=True)
    incomplete = selected.isna().any(axis=1)
    if bool(incomplete.any()) and missing == "raise":
        raise AnalysisError("missing_values", f"{int(incomplete.sum())} row(s) contain missing "
                            "model inputs. Choose missing='drop' explicitly to exclude them.")
    positions = [int(i) for i in (~incomplete).to_numpy().nonzero()[0]]
    sample = selected.iloc[positions].reset_index(drop=True)
    if sample[unit].duplicated().any():
        raise AnalysisError("duplicate_units", "data must have one row per unit.")
    n = len(sample)
    if shock_cluster is not None:
        _require(shocks, [shock_cluster], "shocks")
        if shocks[shock_cluster].isna().any():
            raise AnalysisError("missing_values", "Shock clusters must not be missing.")
    w_matrix, g, order, unmatched, extra, keyed = _shares_matrix(
        sample[unit], shares, shocks, unit=unit, sector=sector, share=share, shock=shock,
        max_cells=max_cells)
    target = _numeric(sample[y], y)
    weight = _numeric(sample[weights], weights) if weights else None
    if weight is not None and bool((weight <= 0).any()):
        raise AnalysisError("negative_weights", "Region weights must be positive.")
    pieces = ([torch.ones((n, 1), dtype=torch.float64)] if intercept else []) + \
        [_numeric(sample[c], c)[:, None] for c in controls]
    z = torch.cat(pieces, 1) if pieces else torch.empty((n, 0), dtype=torch.float64)
    xs = w_matrix @ g
    mm = torch.cat([xs[:, None], z], 1)
    if n <= mm.shape[1]:
        raise AnalysisError("insufficient_observations", "Need more units than regressors.")

    def residual(v: Tensor) -> Tensor:
        return v if z.shape[1] == 0 else _fit(z, v, weight).resid

    def coefficient(v: Tensor):
        fit = _fit(mm, v, weight)
        if 0 not in fit.kept:
            raise AnalysisError("collinear_shift_share", "The shift-share variable is collinear "
                                "with the controls.")
        return float(fit.beta[fit.kept.index(0)]), fit.resid, len(fit.kept)

    wgt = weight if weight is not None else torch.ones(n, dtype=torch.float64)
    ddx = residual(xs)
    if not float((wgt * ddx.square()).sum()) > 1e-20 * float((wgt * xs.square()).sum()):
        raise AnalysisError("collinear_shift_share", "The shift-share variable is collinear with "
                            "the controls (no residual variation).")
    projection = kernel_call(least_squares, w_matrix, ddx, weight, drop_collinear=True)
    kept = projection.kept
    omitted = [order[j] for j in projection.omitted]
    if not kept:
        raise AnalysisError("singular_shares", "The share matrix has no usable column.")
    hx = projection.beta
    wk = w_matrix[:, kept]
    b1, e1, rank = coefficient(target)
    if endog is None:
        model, beta, resid = "ols", b1, e1
        ddy = residual(target)
        exposure = ddx
        rx = float((wgt * ddx.square()).sum())
        first_stage = None
    else:
        regressor = _numeric(sample[endog], endog)
        pi, e2, _ = coefficient(regressor)
        if not abs(pi) > 0:
            raise AnalysisError("underidentified", "The shift-share instrument has no first stage.")
        model, beta = "iv", b1 / pi
        resid = e1 - e2 * beta
        ddy = residual(target)
        exposure = residual(regressor)
        rx = float((wgt * exposure * ddx).sum())
        first_stage = pi
    if not rx != 0:
        raise AnalysisError("underidentified", "The shift-share variable has no residual "
                            "variation after the controls.")
    u = wgt * resid * ddx
    rows: list[dict[str, Any]] = []
    variance: dict[str, float] = {}
    if model == "ols":
        resvar = float((wgt * resid.square()).sum()) / (n - rank)
        variance["homoskedastic"] = resvar / rx
        variance["ehw"] = n / (n - rank) * float(u.square().sum()) / rx**2
    else:
        rss = float((wgt * resid.square()).sum())
        den = float((wgt * ddx.square()).sum())
        variance["homoskedastic"] = rss / n / (den * first_stage**2)
        variance["ehw"] = float(u.square().sum()) / rx**2
    cluster_count = None
    if cluster is not None:
        codes, labels = pd.factorize(sample[cluster])
        cluster_count = len(labels)
        if cluster_count < 2:
            raise AnalysisError("insufficient_clusters", "Region clustering needs two clusters.")
        sums = torch.zeros(cluster_count, dtype=torch.float64).index_add(
            0, torch.as_tensor(codes), u)
        factor = cluster_count / (cluster_count - 1) * (n - 1) / (n - rank) if model == "ols" else 1.0
        variance["region_cluster"] = factor * float(sums.square().sum()) / rx**2
    restricted = ddy - (ddx if model == "ols" else exposure) * beta0
    c_r = hx * (wk.T @ (wgt * resid))
    c_r0 = hx * (wk.T @ (wgt * restricted))
    c_w = hx * (wk.T @ (wgt * exposure))
    shock_clusters = None
    if shock_cluster is not None:
        labels = keyed.loc[[order[j] for j in kept], shock_cluster]
        codes, names = pd.factorize(labels)
        shock_clusters = len(names)
        index = torch.as_tensor(codes)

        def collapse(v: Tensor) -> Tensor:
            return torch.zeros(len(names), dtype=torch.float64).index_add(0, index, v)

        c_r, c_r0, c_w = collapse(c_r), collapse(c_r0), collapse(c_w)
    variance["akm"] = float(c_r.square().sum()) / rx**2
    critical = normal_isf(alpha / 2)
    for method in ("homoskedastic", "ehw", "region_cluster", "akm"):
        if method not in variance:
            continue
        se = math.sqrt(variance[method])
        rows.append({"method": method, "estimate": beta, "std_error": se,
                     "statistic": (beta - beta0) / se, "p_value": 2 * normal_sf(abs(beta - beta0) / se),
                     "ci_low": beta - critical * se, "ci_high": beta + critical * se,
                     "ci_type": "interval"})
    se0 = math.sqrt(float(c_r0.square().sum()) / rx**2)
    q = rx**2 / critical**2 - float(c_w.square().sum())
    q2 = float((c_r * c_w).sum()) / q if q != 0 else math.inf
    middle = beta - q2
    discriminant = q2**2 + float(c_r.square().sum()) / q if q != 0 else math.inf
    if q > 0:
        half = math.sqrt(discriminant)
        pieces_ci = [[middle - half, middle + half]]
        akm0 = {"std_error": half / critical, "ci_low": middle - half, "ci_high": middle + half,
                "ci_type": "interval"}
    elif discriminant > 0 and math.isfinite(discriminant):
        half = math.sqrt(discriminant)
        pieces_ci = [[None, middle - half], [middle + half, None]]
        akm0 = {"std_error": None, "ci_low": None, "ci_high": None,
                "ci_type": "complement_of_interval"}
    else:
        pieces_ci = [[None, None]]
        akm0 = {"std_error": None, "ci_low": None, "ci_high": None, "ci_type": "real_line"}
    rows.append({"method": "akm0", "estimate": beta, "statistic": (beta - beta0) / se0 if se0 > 0 else None,
                 "p_value": 2 * normal_sf(abs(beta - beta0) / se0) if se0 > 0 else None, **akm0})
    exposure_weight = (wgt[:, None] * w_matrix).sum(0)
    total = float(exposure_weight.sum())
    effective = 1 / float((exposure_weight / total).square().sum()) if total > 0 else None
    sums_share = w_matrix.sum(1)
    notes = []
    spread = float(sums_share.max() - sums_share.min())
    if spread > 1e-10 * max(1.0, float(sums_share.abs().max())):
        leftover = residual(sums_share) if z.shape[1] else sums_share
        if float(leftover.square().sum()) > 1e-16 * float(sums_share.square().sum()):
            notes.append("Share sums vary across units and are not spanned by the controls; "
                         "with incomplete shares control for the sum of shares (AKM 2019, "
                         "Borusyak-Hull-Jaravel 2022).")
    if omitted:
        notes.append(f"{len(omitted)} collinear share column(s) dropped: AKM uses the kept "
                     "sectors only.")
    if unmatched:
        notes.append(f"{unmatched} share row(s) belong to units outside the estimation sample.")
    if len(kept) < 30:
        notes.append(f"Only {len(kept)} sectors carry AKM inference; its asymptotics need many "
                     "shocks.")
    return TableSet(
        {"inference": table(rows, columns=["method", "estimate", "std_error", "statistic",
                                           "p_value", "ci_low", "ci_high", "ci_type"]),
         "akm0_confidence_set": table([{"lower": a, "upper": b, "lower_unbounded": a is None,
                                        "upper_unbounded": b is None} for a, b in pieces_ci])},
        title="Shift-share " + ("IV" if model == "iv" else "regression") + " with AKM inference",
        model=model, estimate=beta, beta0=beta0, alpha=alpha, critical_value=critical,
        variances=variance, akm0_null_std_error=se0, akm0_set=pieces_ci,
        akm0_quadratic={"q": q, "center": middle, "discriminant": discriminant},
        first_stage_coefficient=first_stage, nobs=n, n_dropped_missing=int(incomplete.sum()),
        sample_positions=positions, n_sectors=len(order), n_sectors_used=len(kept),
        omitted_sectors=[str(s) for s in omitted], unused_shocks=extra,
        unmatched_share_rows=unmatched, shock_clusters=shock_clusters,
        region_clusters=cluster_count, effective_shock_count=effective,
        share_sum_min=float(sums_share.min()), share_sum_max=float(sums_share.max()),
        controls=(["Intercept"] if intercept else []) + controls, outcome=y, endogenous=endog,
        weights=weights, notes=notes, source=SOURCE,
        conventions="ShiftShareSE: OLS EHW n/(n-p) and region cluster G/(G-1)(n-1)/(n-p); "
                    "IV EHW and region cluster without factors; AKM/AKM0 without factors; "
                    "standard normal reference.")
