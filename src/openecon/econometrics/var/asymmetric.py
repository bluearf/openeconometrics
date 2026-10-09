"""Partial-sum augmented-VAR predictive tests with an explicit resampling null."""

from __future__ import annotations

import math
import torch

from openecon.analysis import _numeric, _frame_hasher
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.var.common import check_alpha
from openecon.econometrics.var.toda_yamamoto import _budget, _ordered_sample
from openecon.engines.distributions import chi2_sf
from openecon.engines.linalg import least_squares


def _fit(levels, p):
    n = len(levels) - p
    design = torch.cat(
        (
            torch.ones((n, 1), dtype=torch.float64),
            *[levels[p - lag : len(levels) - lag] for lag in range(1, p + 1)],
        ),
        1,
    )
    outcome = levels[p:]
    fit = kernel_call(least_squares, design, outcome, drop_collinear=False)
    return design, outcome, fit


def _statistic(levels, p, base):
    design, outcome, fit = _fit(levels, p)
    indices = torch.tensor([2 + 2 * lag for lag in range(base)], dtype=torch.int64)
    covariance = fit.xtx_inv * (fit.resid[:, 0].square().sum() / len(outcome))
    beta = fit.beta[indices, 0]
    restricted = covariance[indices][:, indices]
    try:
        stat = float(beta @ torch.linalg.solve(restricted, beta))
    except RuntimeError as exc:
        raise AnalysisError(
            "singular_test_covariance", "Asymmetric restriction covariance is singular."
        ) from exc
    if not math.isfinite(stat) or stat < 0:
        raise AnalysisError(
            "singular_test_covariance", "Asymmetric Wald statistic is not finite/nonnegative."
        )
    return stat, design, outcome, fit, indices


def asymcausality(
    *,
    data,
    y,
    x,
    lags=1,
    dmax=1,
    time=None,
    signs=("+", "+"),
    bootstrap=199,
    seed=0,
    alpha=0.05,
    max_work=100_000_000,
):
    """Hatemi-J-style partial-sum predictive noncausality x(sign)->y(sign).

    Raw increments split into positive/negative cumulative sums, initialized at
    zero. Fits a bivariate levels VAR(lags+dmax) with constant, testing ONLY the
    first lags of the cause. dmax is a declared upper integration order, not
    chosen by a pretest. For bootstrap inference, restrictions are imposed on
    the effect equation; centered joint innovations are drawn with replacement
    and the augmented null VAR is recursively generated from fixed initial
    partial-sum levels. This explicit iid innovation null can generate signed
    cumulative paths that are not monotone; it is NOT a claim to reproduce the
    paper's leverage-adjusted/wild algorithms or unrestricted null size.

    CPU float64, complete regular resident bivariate series, fixed lag orders.
    Results report asymptotic and conditional-null bootstrap inference apart,
    all draws and MC error, sample/transform identity, full coefficient V.
    Predictive tests do not establish structural causal effects.
    """
    from openecon.econometrics.unitroot.common import check_count

    if not isinstance(y, str) or not isinstance(x, str) or not y or not x or y == x:
        raise AnalysisError("invalid_spec", "Distinct outcome and cause columns are required.")
    if (
        not isinstance(signs, (tuple, list))
        or len(signs) != 2
        or any(s not in {"+", "-"} for s in signs)
    ):
        raise AnalysisError(
            "invalid_option", "signs selects (outcome sign, cause sign), '+' or '-'."
        )
    base = check_count(lags, "lags", minimum=1)
    augmentation = check_count(dmax, "dmax")
    draws = check_count(bootstrap, "bootstrap")
    check_count(seed, "seed")
    if base > 12 or augmentation > 2 or draws > 5000 or 0 < draws < 49:
        raise AnalysisError("invalid_domain", "Use lags<=12, dmax<=2 and bootstrap=0 or 49..5000.")
    alpha = check_alpha(alpha)
    raw = _ordered_sample(data, [y, x], time)
    p = base + augmentation
    _budget(len(raw), 2, p, 1)
    if isinstance(max_work, bool) or not isinstance(max_work, int) or max_work < 1:
        raise AnalysisError("invalid_option", "max_work must be a positive integer.")
    work = (draws + 1) * len(raw) * (2 * p + 1) ** 2
    if work > max_work:
        raise AnalysisError(
            "work_budget_exceeded",
            "Asymmetric test exceeds max_work before allocating bootstrap arrays.",
        )
    raw_levels = torch.stack((_numeric(raw[y], y), _numeric(raw[x], x)), 1)
    differences = torch.diff(raw_levels, dim=0)
    split = torch.stack(
        [
            differences[:, j].clamp_min(0) if sign == "+" else differences[:, j].clamp_max(0)
            for j, sign in enumerate(signs)
        ],
        1,
    )
    levels = torch.cat((torch.zeros((1, 2), dtype=torch.float64), split.cumsum(0)), 0)
    stat, design, outcome, fit, indices = _statistic(levels, p, base)
    sigma = fit.resid.T @ fit.resid / len(outcome)
    full_v = torch.kron(sigma.contiguous(), fit.xtx_inv.contiguous())
    null_statistics = []
    first_indices = None
    null_coef = fit.beta.clone()
    if draws:
        kept = [j for j in range(design.shape[1]) if j not in indices.tolist()]
        restricted = kernel_call(
            least_squares, design[:, kept], outcome[:, 0], drop_collinear=False
        )
        null_coef[:, 0] = 0
        null_coef[kept, 0] = restricted.beta
        innovations = outcome - design @ null_coef
        innovations -= innovations.mean(0)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for replication in range(draws):
            selected = torch.randint(len(innovations), (len(innovations),), generator=generator)
            if replication == 0:
                first_indices = selected.tolist()
            noise = innovations[selected]
            simulated = levels.clone()
            for t in range(p, len(levels)):
                row = torch.cat(
                    (
                        torch.ones(1, dtype=torch.float64),
                        *[simulated[t - lag] for lag in range(1, p + 1)],
                    )
                )
                simulated[t] = row @ null_coef + noise[t - p]
            try:
                null_statistics.append(_statistic(simulated, p, base)[0])
            except AnalysisError as exc:
                raise AnalysisError(
                    "bootstrap_failure",
                    f"Null replicate {replication} failed; no draws are silently discarded.",
                ) from exc
    exceed = sum(s >= stat for s in null_statistics)
    probability = (exceed + 1) / (draws + 1) if draws else None
    coef_terms = [
        "Intercept",
        *[
            f"L{lag}.{name}({sign})"
            for lag in range(1, p + 1)
            for name, sign in zip((y, x), signs, strict=True)
        ],
    ]
    coefficients = [
        {
            "equation": f"{name}({sign})",
            "term": term,
            "estimate": float(fit.beta[j, i]),
            "std_error": float(full_v[i * len(coef_terms) + j, i * len(coef_terms) + j].sqrt()),
            "role": "augmentation" if j > 2 * base else "base/deterministic",
        }
        for i, (name, sign) in enumerate(zip((y, x), signs, strict=True))
        for j, term in enumerate(coef_terms)
    ]
    return TableSet(
        {
            "tests": table(
                [
                    {
                        "outcome": y,
                        "cause": x,
                        "outcome_sign": signs[0],
                        "cause_sign": signs[1],
                        "statistic": stat,
                        "df": base,
                        "asymptotic_p_value": chi2_sf(stat, base),
                        "bootstrap_p_value": probability,
                        "mc_std_error": (probability * (1 - probability) / draws) ** 0.5
                        if draws
                        else None,
                    }
                ]
            ),
            "coefficients": table(coefficients),
        },
        title="Partial-sum predictive noncausality",
        source="https://doi.org/10.1007/s00181-011-0484-x",
        base_lags=base,
        dmax=augmentation,
        signs=list(signs),
        draws=draws,
        seed=seed,
        alpha=alpha,
        sample_hash=_frame_hasher(raw).hexdigest(),
        input_rows=len(raw),
        nobs=len(outcome),
        transformation="zero initial value plus cumsum(max/min(raw differences,0)); initial raw level excluded",
        covariance_matrix=full_v.tolist(),
        null_coefficients=null_coef.tolist(),
        bootstrap_statistics=null_statistics,
        work=work,
        bootstrap_first_indices=first_indices,
        sample_time=raw[time].tolist() if time is not None else None,
        null_resampling="null-imposed recursive augmented partial-sum VAR; centered joint iid residual vectors; plus-one p convention",
        limitations="Conditional iid innovation bootstrap, not calibrated unrestricted heteroskedastic/Hatemi-J wild/leverage bootstrap; bootstrap paths need not be monotone; fixed lag order and sufficient dmax assumed",
    )
