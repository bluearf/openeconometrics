"""Two-way clustered restricted wild (cluster) bootstrap for OLS coefficients.

MacKinnon, Nielsen and Webb (2021, JBES 39; CREATES Research Paper 2020-06,
"Multiway Wild (Cluster) Bootstrap Algorithms"): the test statistic uses the
three-term Cameron--Gelbach--Miller CRVE

    V3 = (X'X)^-1 [c_G M_G + c_H M_H - c_I M_I] (X'X)^-1,
    M_D = sum_{d in D} (X_d' u_d)(X_d' u_d)',  c_D = |D|(N-1)/((|D|-1)(N-k)),

with I the nonempty G x H intersections; when V3 is not positive semidefinite
it is replaced by U diag(max(lambda, 0)) U' (CGM's eigenvalue truncation), in
the sample and in every bootstrap draw. The WCR bootstrap imposes the null
(restricted OLS), multiplies the restricted residuals by wild weights that are
clustered by ONE dimension (G, H), by the intersection, or by observation
(WR), regenerates y* = X b~ + u*, re-estimates OLS and recomputes V3*.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, TableSet, kernel_call, table
from openecon.econometrics.postest import wild_multipliers as wm
from openecon.econometrics.postest.common import matched_frame, require_result
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.engines.distributions import chi2_sf, t_sf
from openecon.engines.linalg import least_squares

SOURCE = "https://doi.org/10.1080/07350015.2019.1677473"
WORKING_PAPER = "https://repec.econ.au.dk/repec/creates/rp/20/rp20_06.pdf"
DEFAULT_MAX_WORK = 20_000_000_000
_BLOCK_BYTES = 64 * 1024**2
_EPS = torch.finfo(torch.float64).eps


def multiway_geometry(result: Any, data: Any, clusters: Any) -> dict[str, Any]:
    result = require_result(result, "multiway wild result")
    spec = result.spec
    if spec.estimator != "ols" or spec.weights:
        raise AnalysisError("unsupported_spec", "The multiway wild route supports unweighted OLS results.")
    if clusters is None:
        clusters = list(spec.cluster) if isinstance(spec.cluster, (list, tuple)) else None
    if not isinstance(clusters, (list, tuple)) or len(clusters) != 2 \
            or not all(isinstance(c, str) and c for c in clusters) or clusters[0] == clusters[1]:
        raise AnalysisError("invalid_cluster", "Declare exactly two distinct cluster columns.")
    original, positions = matched_frame(result, data)
    sample = original.iloc[positions].reset_index(drop=True)
    for name in clusters:
        if name not in sample or bool(sample[name].isna().any()):
            raise AnalysisError("invalid_cluster", f"Cluster column '{name}' must be present and complete.")
    frame = ModelFrame(spec, sample)
    if frame.n != len(sample):
        raise AnalysisError("data_mismatch", "The matched estimation sample changed when rebuilt.")
    design = frame.design()
    x, terms = design.x, list(design.terms)
    n, k = x.shape
    if n <= k + 1 or n > 1_000_000 or k > 128:
        raise AnalysisError("work_budget_exceeded", "Multiway wild inference needs k < N - 1, "
                            "N <= 1,000,000 and k <= 128.")
    y = frame.numeric(spec.outcome)
    fit = kernel_call(least_squares, x, y, drop_collinear=False)
    dims = []
    for name in clusters:
        codes, labels = pd.factorize(sample[name], sort=False)
        dims.append((torch.as_tensor(codes, dtype=torch.int64), len(labels)))
    pair = dims[0][0] * dims[1][1] + dims[1][0]
    cells, inter = torch.unique(pair, return_inverse=True)
    dims.append((inter, int(cells.numel())))
    for (codes, count), name in zip(dims, [*clusters, "intersection"], strict=True):
        if count < 2:
            raise AnalysisError("insufficient_clusters", f"Dimension {name} needs at least two clusters.")
    return {"x": x, "y": y, "beta": fit.beta, "bread": fit.xtx_inv, "terms": terms, "n": n,
            "k": k, "dims": dims, "clusters": list(clusters), "result_id": result.id,
            "positions": positions}


def _restrictions(g: dict[str, Any], null: Any) -> tuple[Tensor, Tensor]:
    if not isinstance(null, dict) or not 1 <= len(null) <= 24:
        raise AnalysisError("invalid_restrictions", "Supply 1..24 term-to-null-value restrictions.")
    r = torch.zeros((len(null), g["k"]), dtype=torch.float64)
    values = []
    for i, (term, value) in enumerate(null.items()):
        if term not in g["terms"]:
            raise AnalysisError("invalid_restrictions", f"Unknown term {term!r}.")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AnalysisError("invalid_restrictions", "Null values must be finite real numbers.")
        r[i, g["terms"].index(term)] = 1.0
        values.append(float(value))
    return r, torch.tensor(values, dtype=torch.float64)


def _factor(count: int, n: int, k: int) -> float:
    return count * (n - 1) / ((count - 1) * (n - k))


def multiway_variance(g: dict[str, Any], resid: Tensor) -> tuple[Tensor, Tensor]:
    """Three-term CRVE for residuals [..., n], PSD-truncated. Returns (V [..., k, k], adjusted [...])."""
    shape = resid.shape[:-1]
    flat = resid.reshape(-1, g["n"])
    scores = flat[:, :, None] * g["x"]                            # [B, n, k]
    meat = scores.new_zeros((flat.shape[0], g["k"], g["k"]))
    for sign, (codes, count) in zip((1.0, 1.0, -1.0), g["dims"], strict=True):
        sums = scores.new_zeros((flat.shape[0], count, g["k"])).index_add_(1, codes, scores)
        meat += sign * _factor(count, g["n"], g["k"]) * (sums.transpose(1, 2) @ sums)
    bread = g["bread"]
    v = bread @ meat @ bread
    v = (v + v.transpose(1, 2)) / 2
    values, vectors = torch.linalg.eigh(v)
    tolerance = 64 * g["k"] * _EPS * values.abs().amax(-1)
    adjusted = values[:, 0] < -tolerance
    if bool(adjusted.any()):
        fixed = (vectors * values.clamp_min(0)[:, None, :]) @ vectors.transpose(1, 2)
        v = torch.where(adjusted[:, None, None], fixed, v)
    return v.reshape(*shape, g["k"], g["k"]), adjusted.reshape(shape)


def _statistic(delta: Tensor, cov: Tensor) -> tuple[Tensor, Tensor]:
    """Signed t (q = 1) or Wald (q > 1) with validity flags, batched."""
    q = delta.shape[-1]
    if q == 1:
        variance = cov[..., 0, 0]
        stat = delta[..., 0] / variance.clamp_min(0).sqrt()
        ok = (variance > 0) & torch.isfinite(stat)
        return stat, ok
    factor, info = torch.linalg.cholesky_ex(cov)
    stat = (delta[..., None, :] @ torch.cholesky_solve(delta[..., :, None], factor))[..., 0, 0]
    return stat, (info == 0) & torch.isfinite(stat) & (stat >= 0)


def wcr_draws(g: dict[str, Any], r: Tensor, target: Tensor, multipliers: Tensor,
              level: int | None) -> tuple[Tensor, Tensor, Tensor]:
    """Bootstrap statistics for multipliers [B, G_level] (``level`` None = observation level)."""
    bread, x = g["bread"], g["x"]
    r, target = r.to(torch.float64), target.to(torch.float64)
    multipliers = multipliers.to(torch.float64)
    rb = r @ bread @ r.T
    restricted = g["beta"] - bread @ r.T @ torch.linalg.solve(rb, r @ g["beta"] - target)
    resid = g["y"] - x @ restricted
    weights = multipliers if level is None else multipliers[:, g["dims"][level][0]]
    errors = resid * weights                                      # [B, n]
    change = errors @ x @ bread                                   # (X'X)^-1 X'u*, [B, k]
    fitted_resid = errors - change @ x.T
    cov, adjusted = multiway_variance(g, fitted_resid)
    delta = change @ r.T                                          # R(b* - b~) = R b* - r
    stat, ok = _statistic(delta, r @ cov @ r.T)
    return stat, ok, adjusted


def _level(g: dict[str, Any], choice: Any) -> tuple[int | None, str]:
    clusters = g["clusters"]
    if choice == "fewest":
        counts = [g["dims"][0][1], g["dims"][1][1]]
        index = 0 if counts[0] <= counts[1] else 1
        return index, clusters[index]
    if choice in clusters:
        return clusters.index(choice), choice
    if choice == "intersection":
        return 2, "intersection"
    if choice == "observation":
        return None, "observation"
    raise AnalysisError("invalid_option", "bootstrap_cluster must be 'fewest', one of the cluster "
                        "columns, 'intersection' or 'observation'.")


@resident_cpu
def wild_multiway_test(result, *, data, null, clusters=None, bootstrap_cluster="fewest",
                       reps=999, seed=0, wild="rademacher", tail="symmetric",
                       max_work=DEFAULT_MAX_WORK):
    """Two-way cluster-robust restricted wild bootstrap (WCR) test of linear coefficient restrictions.

    For an unweighted OLS ``result`` and its data, ``null`` maps 1..24 terms to
    hypothesised values. The statistic is the t (one restriction) or Wald
    (several) statistic studentized by the three-term two-way CRVE with the
    per-term factors |D|(N-1)/((|D|-1)(N-k)) and CGM eigenvalue truncation when
    it is not positive semidefinite. Bootstrap (MacKinnon, Nielsen and Webb):
    restricted OLS, wild multipliers clustered by ``bootstrap_cluster`` --
    'fewest' (default: the dimension with fewer clusters, for which the
    authors report modest simulation evidence), a cluster column name,
    'intersection', or 'observation' (the WR bootstrap) -- then OLS and V3* in
    every draw. ``tail``: symmetric |t| (default), equal_tail, upper or lower
    for one restriction; Wald tests are upper tail. P-values are plus-one
    (1 + #exceedances)/(B + 1). The asymptotic reference is t(min(G, H) - 1)
    (one restriction) or chi2(q). Draws with zero studentizing variance raise
    ``bootstrap_failure``; truncation counts are reported. Two dimensions only;
    weights, absorbed fixed effects and IV are outside this route.
    """
    wm.check_kind(wild)
    reps = wm.check_count(reps, "reps", 99, 99_999)
    seed = wm.check_count(seed, "seed", 0, 2**63 - 1)
    max_work = wm.check_count(max_work, "max_work", 1, 10**14)
    g = multiway_geometry(result, data, clusters)
    r, target = _restrictions(g, null)
    q = r.shape[0]
    if q > 1 and tail != "symmetric":
        raise AnalysisError("invalid_option", "Joint Wald tests use the upper tail only.")
    if tail not in ("symmetric", "equal_tail", "upper", "lower"):
        raise AnalysisError("invalid_option", "tail must be symmetric, equal_tail, upper or lower.")
    level, level_name = _level(g, bootstrap_cluster)
    groups = g["n"] if level is None else g["dims"][level][1]
    n, k = g["n"], g["k"]
    cells = sum(count for _, count in g["dims"])
    work = (reps + 1) * (n * k * 8 + cells * k * k * 2 + 4 * k ** 3)
    if work > max_work:
        raise AnalysisError("work_budget_exceeded", f"The bootstrap needs about {work:.3g} "
                            f"floating-point operations, above max_work={max_work}.")
    resid = g["y"] - g["x"] @ g["beta"]
    cov, adjusted = multiway_variance(g, resid[None])
    delta = (r @ g["beta"] - target)[None]
    observed, ok = _statistic(delta, r @ cov[0] @ r.T)
    if not bool(ok.all()):
        raise AnalysisError("singular_covariance", "The two-way CRVE gives zero variance for the "
                            "tested combination even after eigenvalue truncation.")
    observed = float(observed[0])
    per_draw = 8 * (3 * n * k + cells * k + 4 * k * k + n)
    block = max(1, _BLOCK_BYTES // per_draw)
    draws, truncated = [], 0
    for multipliers in wm.blocks(wild, reps, groups, seed, block):
        stat, good, adj = wcr_draws(g, r, target, multipliers, level)
        failed = int((~good).sum())
        if failed:
            raise AnalysisError("bootstrap_failure", f"{failed} multiway bootstrap draws had zero "
                                "studentizing variance after truncation; no draw is dropped.")
        truncated += int(adj.sum())
        draws.append(stat)
    draws = torch.cat(draws)
    rule = tail if q == 1 else "upper"
    probability = wm.tail_p_values(observed, draws, rule)
    smallest = min(g["dims"][0][1], g["dims"][1][1])
    if q == 1:
        asymptotic = 2 * t_sf(abs(observed), smallest - 1)
        reference = f"t({smallest - 1}), two-sided"
    else:
        asymptotic = chi2_sf(observed, q)
        reference = f"chi2({q})"
    row = {"statistic": observed, "df": q, "bootstrap_p_value": probability,
           "mc_std_error": wm.mc_error(probability, reps), "asymptotic_p_value": asymptotic,
           "reference": reference, "tail": rule}
    return TableSet(
        {"test": table([row], title="Two-way wild cluster restricted bootstrap")},
        title="Multiway wild cluster bootstrap (WCR)", source=SOURCE, working_paper=WORKING_PAPER,
        null=null, clusters=g["clusters"], cluster_counts=[d[1] for d in g["dims"][:2]],
        intersections=g["dims"][2][1], bootstrap_cluster=level_name, bootstrap_groups=groups,
        variance="three-term CGM CRVE, per-term factors |D|(N-1)/((|D|-1)(N-k)), eigenvalue truncation",
        observed_truncated=bool(adjusted[0]), truncated_draws=truncated, multiplier=wild,
        draws=reps, failed_draws=0, seed=seed, p_value_rule="plus-one (1 + #exceedances)/(B + 1)",
        covariance_matrix=cov[0].tolist() if k <= 40 else None, terms=g["terms"],
        estimates=g["beta"].tolist(), source_result_id=g["result_id"], observations=n,
        work=work, max_work=max_work, device="cpu", dtype="float64",
        assumptions="two-way clustered errors; the bootstrap clusters by one dimension only "
                    "(MNW give conditions under which each choice is valid)")
