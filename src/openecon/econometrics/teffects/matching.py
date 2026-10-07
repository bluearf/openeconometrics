"""Nearest-neighbour matching estimators (Stata's teffects nnmatch and psmatch).

Every unit i is matched with replacement to the ``M`` nearest units of the
other treatment group; ties with the M-th distance are all kept, so the
matched set J(i) has #J(i) >= M members. With imputed potential outcomes
Yhat_i(D_i) = Y_i and Yhat_i(1 - D_i) = mean of Y over J(i),

    ATE  = (1/N) sum_i (Yhat_i(1) - Yhat_i(0)),
    ATET = (1/N_1) sum_{D_i = 1} (Y_i - Yhat_i(0)).

nnmatch matches on the covariates with the distance
||x_i - x_j||_S = ((x_i - x_j)' S^-1 (x_i - x_j))^(1/2): S the sample covariance
(``mahalanobis``), its diagonal (``ivariance``) or the identity (``euclidean``).
psmatch matches on the estimated propensity score p(x) of a logit or probit
treatment model, |p_i - p_j|.

Bias adjustment (Abadie and Imbens 2011, ``biasadj``): mu_w(x) = x_B'b_w is
fitted by least squares on the units of group w weighted by their usage
K_j (the matched sample), and Yhat_i(1 - D_i) becomes
mean_{j in J(i)} (Y_j + mu(x_i) - mu(x_j)).

Variance (Abadie and Imbens 2006, Stata's vce(robust, nn(#))): with
K_j = sum_i 1{j in J(i)}/#J(i), K'_j = sum_i 1{j in J(i)}/#J(i)^2 and the
conditional variances sigma2_i = #H/(#H+1) (Y_i - mean_{H(i)} Y)^2 estimated
from the ``vce_neighbors`` nearest units H(i) of i's own group,

    V(ATE)  = (1/N^2) sum_i [(Yhat_i(1) - Yhat_i(0) - ATE)^2 + (K_i^2 + 2 K_i - K'_i) sigma2_i]
    V(ATET) = (1/N_1^2) [sum_{D=1} (Y_i - Yhat_i(0) - ATET)^2 + sum_{D=0} (K_i^2 - K'_i) sigma2_i]

(without ties K'_i = K_i/M, giving the familiar K^2 + (2M-1)/M K). psmatch
subtracts the Abadie-Imbens (2016) correction for the estimated score: with
f = dF/d(x'g), V_g the inverse information of the treatment model and
cov_w(x, y | p) estimated by matching on p within group w,

    ATE:  V - c'V_g c,              c   = (1/N) sum f_i (cov_1i / p_i + cov_0i / (1 - p_i))
    ATET: V - c_t'V_g c_t + d'V_g d, c_t = (1/N_1) sum f_i (cov_1i + p_i/(1 - p_i) cov_0i),
                                     d   = (1/N_1) sum f_i x_i (Yhat_i(1) - Yhat_i(0) - ATET).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import Design, ModelFrame, build_result, kernel_call
from openecon.econometrics.discrete.common import constant_shift
from openecon.econometrics.teffects import neighbors as nn
from openecon.econometrics.teffects.common import (
    check_overlap, coefficient_table, propensity_summary, treatment_coding,
)
from openecon.econometrics.teffects.index import fit_treatment
from openecon.engines.optimize import information_inverse
from openecon.models import ResultBundle


def _covariate_design(frame: ModelFrame, names: list[str], what: str) -> Design:
    """Covariates screened for collinearity together with a constant, which is then removed."""
    design = frame.drop_collinear(frame.design(names, intercept=True))
    design = design.select(range(1, len(design.terms)))
    if not design.terms:
        raise AnalysisError("invalid_spec", f"{what} needs at least one non-constant covariate.")
    return Design(design.x, design.terms, design.categories, False)


def _whiten(x: Tensor, metric: str) -> Tensor:
    """Coordinates in which the chosen metric is Euclidean (sample moments, n - 1 divisor)."""
    if metric == "euclidean":
        return x
    centred = x - x.mean(dim=0)
    covariance = centred.T @ centred / (x.shape[0] - 1)
    if metric == "ivariance":
        return x / covariance.diagonal().sqrt()
    try:
        chol = torch.linalg.cholesky(covariance)
    except RuntimeError as exc:
        raise AnalysisError("singular_covariance", "The covariance matrix of the matching "
                            "covariates is singular; drop redundant covariates.") from exc
    return torch.linalg.solve_triangular(chol, x.T, upper=False).T


class _Searcher:
    """Matching of one set of units to another on fixed coordinates."""

    def __init__(self, coords: Tensor, exact: Tensor | None = None):
        self.coords = coords
        self.one_d = coords.shape[1] == 1
        self.exact = exact

    def match(self, query: Tensor, pool: Tensor, values: Tensor, m: int, *,
              own: bool = False) -> nn.Matches:
        if self.exact is None:
            return self._match(query, pool, values, m, own=own)
        count, sums = torch.zeros(len(query), dtype=torch.float64), torch.zeros((len(query), values.shape[1]), dtype=torch.float64)
        distance = torch.zeros_like(count)
        usage, usage2 = torch.zeros(len(pool), dtype=torch.float64), torch.zeros(len(pool), dtype=torch.float64)
        for cell in torch.unique(self.exact[query]):
            qi = torch.nonzero(self.exact[query] == cell).flatten()
            pi = torch.nonzero(self.exact[pool] == cell).flatten()
            if len(pi) < m + int(own):
                raise AnalysisError("exact_match_support", f"Exact-match cell {int(cell)} has {len(pi)} pool units for {m + int(own)} required units; {len(qi)} queries lack support.")
            found = self._match(query[qi], pool[pi], values, m, own=own)
            count[qi], sums[qi], distance[qi] = found.count, found.sums, found.distance
            if found.usage is not None:
                usage[pi], usage2[pi] = found.usage, found.usage2
        return nn.Matches(count, sums, usage, usage2, distance)

    def _match(self, query: Tensor, pool: Tensor, values: Tensor, m: int, *, own: bool = False) -> nn.Matches:
        """Match the rows ``query`` (index tensor) to the rows ``pool``; values are per row."""
        if self.one_d:
            key = self.coords[pool, 0]
            order = torch.argsort(key, stable=True)
            sorted_pool = key[order]
            position = None
            if own:
                position = torch.empty_like(order)
                position[order] = torch.arange(len(order))
            found = kernel_call(nn.match_sorted, self.coords[query, 0], sorted_pool,
                                values[pool][order], m, own=position)
            if found.usage is not None:
                usage, usage2 = torch.empty_like(found.usage), torch.empty_like(found.usage2)
                usage[order], usage2[order] = found.usage, found.usage2
                found.usage, found.usage2 = usage, usage2
            return found
        local = None
        if own:
            local = torch.arange(len(pool))
        return kernel_call(nn.match_blocks, self.coords[query], self.coords[pool], values[pool],
                           m, own=local)


def _weighted_fit(x: Tensor, y: Tensor, weights: Tensor, what: str) -> Tensor:
    from openecon.engines.linalg import least_squares

    if int((weights > 0).sum()) <= x.shape[1]:
        raise AnalysisError("insufficient_observations", f"The bias-adjustment regression of the "
                            f"{what} has too few matched units for its covariates.")
    fit = kernel_call(least_squares, x, y, weights, drop_collinear=True)
    beta = torch.zeros(x.shape[1], dtype=torch.float64)
    beta[fit.kept] = fit.beta
    return beta


def fit_matching(frame: ModelFrame, method: str) -> ResultBundle:
    spec = frame.spec
    estimand = frame.option("estimand")
    if estimand == "pomeans":
        raise AnalysisError("unsupported_estimand", "Matching estimators estimate ate or atet.")
    if spec.covariance != "robust":
        raise AnalysisError("unsupported_covariance", "Matching estimators use the Abadie-Imbens "
                            "robust covariance (covariance='robust').")
    if spec.weights is not None:
        if spec.weight_type != "fweight":
            raise AnalysisError("unsupported_weights", "Matching supports bounded frequency duplication only; sampling weights are not validated.")
        counts = frame.weights().to(torch.int64)
        n = int(counts.sum())
        if n > 100000 or n * (len(frame.sample.columns) + len(spec.predictors)) > 2_000_000:
            raise AnalysisError("matching_too_large", "Frequency duplication exceeds 100,000 rows or 2,000,000 data entries; no partial expansion is performed.")
        expanded = frame.sample.iloc[torch.repeat_interleave(torch.arange(frame.n), counts).numpy()].reset_index(drop=True)
        plain_spec = spec.model_copy(update={"weights": None, "weight_type": None})
        result = fit_matching(ModelFrame(plain_spec, expanded), method)
        return build_result(frame, terms=[c.term for c in result.coefficients],
            params=torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64),
            covariance=torch.tensor(result.covariance_matrix, dtype=torch.float64),
            equations=[c.equation for c in result.coefficients], title=result.title, use_t=False,
            nobs=n, metrics=result.metrics, inference=result.inference, solver="nearest_neighbor_matching",
            extra={**result.extra, "frequency_duplication": {"physical_rows": frame.n, "expanded_rows": n, "max_rows": 100000}},
            provenance={"method": method, "estimand": estimand, "weight_semantics": "literal frequency duplication"})
    treatment = treatment_coding(frame, frame.role("treatment")[0], frame.option("control"))
    if treatment.levels != 2:
        raise AnalysisError("invalid_treatment", f"teffects {method} needs a binary treatment.")
    if frame.option("tlevel") is not None and frame.option("tlevel") != treatment.labels[1]:
        raise AnalysisError("invalid_treatment", "Binary matching tlevel must name the non-control level.")
    iid = frame.option("matching_vce") == "iid"
    if iid and frame.role("biasadj"):
        raise AnalysisError("unsupported_covariance", "iid matching variance currently validates unadjusted nnmatch only.")
    m, neighbours = int(frame.option("neighbors")), int(frame.option("vce_neighbors"))
    caliper = frame.option("caliper")
    y = frame.numeric(spec.outcome)
    codes = treatment.codes
    rows = [torch.nonzero(codes == level).flatten() for level in (0, 1)]
    n, n1 = frame.n, len(rows[1])
    fit = None
    if method == "nnmatch":
        covariates = _covariate_design(frame, list(spec.predictors), "nnmatch")
        coords = _whiten(covariates.x, frame.option("metric"))
        match_terms = covariates.terms
    else:
        if method == "psmatch" and neighbours < 2:
            raise AnalysisError("invalid_spec", "psmatch needs vce_neighbors >= 2 to estimate the "
                                "conditional covariances of the propensity-score correction.")
        tx = frame.role("tx") or list(spec.predictors)
        design = frame.drop_collinear(frame.design(tx, intercept=True))
        if len(design.terms) < 2:
            raise AnalysisError("invalid_spec", "psmatch needs at least one non-constant "
                                "treatment-model covariate (tx or x).")
        means = torch.zeros(design.x.shape[1], dtype=torch.float64)
        means[1:] = design.x[:, 1:].mean(dim=0)
        xt = design.x - means
        fit = fit_treatment(frame.option("tmodel"), xt, codes, 2,
                            torch.ones(n, dtype=torch.float64))
        check_overlap(fit.probabilities, float(frame.option("pstolerance")), treatment)
        coords = fit.probabilities[:, 1:2].contiguous()
        match_terms = ["propensity score"]
    exact_names = frame.role("ematch")
    exact = None
    if exact_names:
        exact = torch.as_tensor(pd.factorize(pd.MultiIndex.from_frame(frame.sample[exact_names]), sort=False)[0], dtype=torch.int64)
    searcher = _Searcher(coords, exact)
    ybar = y.mean()
    centred = y - ybar
    bias_names = frame.role("biasadj")
    xb = None
    if bias_names:
        if method == "psmatch":
            frame.warn("Stata's teffects psmatch has no bias adjustment, and the Abadie-Imbens "
                       "(2016) estimated-propensity-score correction is derived for the "
                       "unadjusted estimator; the reported standard error applies it to the "
                       "bias-adjusted estimate as an approximation.")
        bias = frame.drop_collinear(frame.design(bias_names, intercept=True))
        xb = bias.x - torch.cat([torch.zeros(1, dtype=torch.float64), bias.x[:, 1:].mean(dim=0)])
    values = centred[:, None] if xb is None else torch.cat([centred[:, None], xb], dim=1)

    directions = [1] if estimand == "atet" and method == "nnmatch" else [1, 0]
    found: dict[int, nn.Matches] = {}
    for level in directions:          # match the units of `level` to the other group
        found[level] = searcher.match(rows[level], rows[1 - level], values, m)
        if caliper is not None and bool((found[level].distance > caliper).any()):
            count = int((found[level].distance > caliper).sum())
            raise AnalysisError("caliper_violation", f"{count} observation(s) have fewer than "
                                f"{m} match(es) within caliper={caliper:g}. Enlarge the caliper, "
                                "or restrict the sample to the region of common support.")
    imputed = torch.stack([y, y], dim=1)          # columns: Yhat(0), Yhat(1)
    usage = torch.zeros(n, dtype=torch.float64)
    usage2 = torch.zeros(n, dtype=torch.float64)
    bias_tables: dict[str, Any] = {}
    betas: dict[int, Tensor] = {}
    for level, matches in found.items():
        pool = rows[1 - level]
        usage[pool], usage2[pool] = matches.usage, matches.usage2
        if xb is not None:
            betas[1 - level] = _weighted_fit(xb[pool], y[pool], matches.usage,
                                             f"{'treated' if level == 0 else 'control'} group")
    for level, matches in found.items():
        mean_y = matches.sums[:, 0] / matches.count + ybar
        if xb is not None:
            mean_x = matches.sums[:, 1:] / matches.count[:, None]
            mean_y = mean_y + (xb[rows[level]][:, 1:] - mean_x[:, 1:]) @ betas[1 - level][1:]
        imputed[rows[level], 1 - level] = mean_y
    if xb is not None:
        bias_tables = {f"group {treatment.text(level)}": {
            term: float(value) for term, value in zip(bias.terms, betas[level].tolist(),
                                                      strict=True)} for level in betas}

    # Conditional variances use own-group matches unless iid requests a common variance.
    sigma2 = torch.zeros(n, dtype=torch.float64)
    for level in (() if iid else (0, 1)):
        group = rows[level]
        within = searcher.match(group, group, centred[:, None], neighbours, own=True)
        size = within.count
        sigma2[group] = size / (size + 1) * (centred[group] - within.sums[:, 0] / size).square()
    effect = imputed[:, 1] - imputed[:, 0]
    if iid:
        # Common conditional variance from opposite-group squared pair contrasts.
        target_rows = rows[1] if estimand == "atet" else torch.arange(n)
        tau0 = effect[target_rows].mean()
        squared = torch.zeros(n, dtype=torch.float64)
        for level in directions:
            q, pool = rows[level], rows[1 - level]
            moments = searcher.match(q, pool, torch.stack([centred, centred.square()], dim=1), m)
            mu = moments.sums[:, 0] / moments.count
            second = moments.sums[:, 1] / moments.count
            sign = 2 * level - 1
            squared[q] = centred[q].square() - 2 * centred[q] * mu + second - 2 * tau0 * sign * (centred[q] - mu) + tau0.square()
        sigma2[:] = squared[target_rows].mean() / 2
    if estimand == "ate":
        tau = effect.mean()
        variance = ((effect - tau).square().sum()
                    + ((usage.square() + 2 * usage - usage2) * sigma2).sum()) / n ** 2
    else:
        treated = rows[1]
        tau = effect[treated].mean()
        control = rows[0]
        k, k2 = usage[control], usage2[control]
        variance = ((effect[treated] - tau).square().sum()
                    + ((k.square() - k2) * sigma2[control]).sum()) / n1 ** 2
    ai_variance = float(variance)
    adjustment = 0.0
    if fit is not None:
        adjustment = _propensity_adjustment(searcher, fit, xt, centred, rows, effect, float(tau),
                                            estimand, neighbours)
        variance = variance - adjustment
    if not float(variance) > 0:
        raise AnalysisError("invalid_covariance", "The estimated variance of the matching "
                            "estimator is not positive (the propensity-score correction exceeds "
                            "the Abadie-Imbens variance); increase vce_neighbors or the sample.")
    model = None if fit is None else (fit, xt, design.terms, means)
    result = _result(frame, method, treatment, estimand, tau, variance, found, usage, m, neighbours,
                     match_terms, bias_tables, model, ai_variance, adjustment)
    result.extra.update({"ematch": exact_names, "exact_cells": int(torch.unique(exact).numel()) if exact is not None else 0,
                                     "unmatched_policy": "error; no silent sample reduction", "tie_policy": "all equal-distance matches",
                                     "matching_vce": "iid" if iid else "robust"})
    result.inference["variance_method"] = "Abadie-Imbens homoskedastic matching variance" if iid else result.inference["variance_method"]
    if iid:
        result.inference["correction"] = "Abadie-Imbens iid variance: common conditional variance from opposite-treatment squared pair contrasts"
    return result


def _propensity_adjustment(searcher: _Searcher, fit, xt: Tensor, centred: Tensor,
                           rows: list[Tensor], effect: Tensor, tau: float, estimand: str,
                           neighbours: int) -> float:
    """c'V_g c (ATE) or c_t'V_g c_t - d'V_g d (ATET), Abadie and Imbens (2016)."""
    n, kt = xt.shape
    p = fit.probabilities[:, 1]
    f = fit.slopes[:, 1, 0]
    information = (xt * (-fit.curvature[:, 0, 0])[:, None]).T @ xt
    v_gamma = kernel_call(information_inverse, information)
    values = torch.cat([xt, centred[:, None], xt * centred[:, None]], dim=1)
    covariances = torch.zeros((2, n, kt), dtype=torch.float64)
    for w in (0, 1):
        group = rows[w]
        for level in (0, 1):
            query = rows[level]
            found = searcher.match(query, group, values, neighbours, own=level == w)
            size = found.count[:, None]
            sx, sy, sxy = found.sums[:, :kt], found.sums[:, kt:kt + 1], found.sums[:, kt + 1:]
            covariances[w, query] = (sxy - sx * sy / size) / (size - 1)
    if estimand == "ate":
        c = (f[:, None] * (covariances[1] / p[:, None] + covariances[0] / (1 - p)[:, None])).sum(
            dim=0) / n
        return float(c @ v_gamma @ c)
    n1 = len(rows[1])
    c = (f[:, None] * (covariances[1] + (p / (1 - p))[:, None] * covariances[0])).sum(dim=0) / n1
    d = (f * (effect - tau)) @ xt / n1
    return float(c @ v_gamma @ c - d @ v_gamma @ d)


def _result(frame: ModelFrame, method: str, treatment, estimand: str, tau: Tensor,
            variance: Tensor, found: dict[int, nn.Matches], usage: Tensor, m: int,
            neighbours: int, match_terms: list[str], bias_tables: dict[str, Any], model,
            ai_variance: float, adjustment: float) -> ResultBundle:
    name = "ATET" if estimand == "atet" else "ATE"
    term = treatment.effect_term(name, 1)
    sizes = torch.cat([matches.count for matches in found.values()])
    extra: dict[str, Any] = {
        "method": method, "estimand": estimand, "treatment": treatment.column,
        "control": treatment.text(0), "levels": [treatment.text(0), treatment.text(1)],
        "neighbors": m, "vce_neighbors": neighbours,
        "metric": frame.option("metric") if method == "nnmatch" else "propensity score",
        "matching_variables": match_terms,
        "matches": {"min": int(sizes.min()), "max": int(sizes.max()),
                    "mean": float(sizes.mean()), "units_with_ties": int((sizes > m).sum())},
        "usage": {"max": float(usage.max()), "mean_used": float(usage[usage > 0].mean()),
                  "units_used": int((usage > 0).sum())},
        "observations_by_level": {treatment.text(level): treatment.counts[level]
                                  for level in (0, 1)},
        "variance": {"abadie_imbens": ai_variance, "propensity_adjustment": adjustment},
        "bias_adjustment": bias_tables or None,
        "caliper": frame.option("caliper"),
    }
    metrics: dict[str, Any] = {"n_control": treatment.counts[0], "n_treated": treatment.counts[1],
                               "matches_min": int(sizes.min()), "matches_max": int(sizes.max())}
    correction = ("Abadie-Imbens (2006) heteroskedasticity-robust variance with "
                  f"nn({neighbours}) conditional variances")
    if model is not None:
        fit, xt, terms, means = model
        v_gamma = information_inverse((xt * (-fit.curvature[:, 0, 0])[:, None]).T @ xt)
        shift = constant_shift(means)
        extra["auxiliary_equations"] = {f"TME{treatment.text(1)}": coefficient_table(
            terms, shift @ fit.gamma[0], shift @ v_gamma @ shift.T)}
        extra["overlap"] = {"pstolerance": float(frame.option("pstolerance")),
                            "propensity": propensity_summary(fit.probabilities, treatment)}
        correction += ", minus the Abadie-Imbens (2016) estimated-propensity-score adjustment"
    info = {"correction": correction, "small_sample_correction": 1.0, "df_inference": None,
            "variance_method": "Abadie-Imbens matching variance"}
    kind = "nearest-neighbor" if method == "nnmatch" else "propensity-score"
    return build_result(
        frame, terms=[term], params=tau.reshape(1), covariance=variance.reshape(1, 1),
        equations=[name], title=f"Treatment-effects estimation: {kind} matching",
        use_t=False, metrics=metrics, solver="nearest_neighbor_matching", inference=info,
        extra=extra, nobs=frame.n,
        provenance={"method": method, "estimand": estimand, "neighbor_search":
                    "sorted one-dimensional" if method == "psmatch" or len(match_terms) == 1
                    else "blockwise exact distances"},
    )
