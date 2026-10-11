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
from openecon.econometrics.var import bootstrap_kernels as bk

# Hacker and Hatemi-J's GAUSS routines as distributed in tspdlib (actest.src).
_AUTHOR_CODE = "https://github.com/aptech/tspdlib/blob/master/src/actest.src"


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
    resampling="iid",
    leverage_design="recursive",
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

    ``resampling="leverage"`` instead runs the Hacker--Hatemi-J (2006)
    leveraged bootstrap used by Hatemi-J (2012), following the authors' GAUSS
    routines (``Bootstrap_Toda``/``estvar_params``/``W_test``): the Wald uses
    the df-corrected system covariance RES'RES/(T' - m); the null DGP is the
    restricted VAR(lags) WITHOUT augmentation lags (effect equation on its own
    lags, cause equation unrestricted); its residuals are divided by
    sqrt(1 - h_t) with h_t the observation's leverage in its own equation
    (observation-specific, as in the paper; the GAUSS code applies the first
    observation's leverage to all rows); each equation's modified residual is
    drawn with its own independent index (``torch.randint(T', (B, T', 2))``),
    each drawn column is mean-adjusted, and paths are generated recursively
    from the observed initial values (``leverage_design="recursive"``, the
    authors' distributed GAUSS routine). ``leverage_design="fixed"`` instead
    keeps the observed lagged regressors, y* = X D~' + delta*, the form written
    in Hacker--Hatemi-J's papers. Critical values use the authors' rule
    (average of the order statistics at B - floor(B a) and the next one);
    the plus-one bootstrap p-value is also reported. Vectorised over draws.

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
    if resampling not in ("iid", "leverage"):
        raise AnalysisError("invalid_option", "resampling must be 'iid' or 'leverage'.")
    if leverage_design not in ("recursive", "fixed"):
        raise AnalysisError("invalid_option", "leverage_design must be 'recursive' or 'fixed'.")
    if resampling == "leverage" and 0 < draws < 99:
        raise AnalysisError("invalid_domain", "The leveraged bootstrap needs bootstrap=0 or 99..5000.")
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
    if resampling == "leverage":
        return _leverage_route(levels, raw, y, x, time, signs, base, augmentation, draws, seed,
                               alpha, work, leverage_design == "recursive")
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


def _hh_design(levels, p, start):
    """Lagged regressors y_(t-1), x_(t-1), ..., y_(t-p), x_(t-p) (lag-major) and outcomes, rows start.."""
    n = levels.shape[-2]
    lagged = torch.cat([levels[..., start - lag:n - lag, :] for lag in range(1, p + 1)], dim=-1)
    return lagged, levels[..., start:, :]


def hh_wald(levels, base, augmentation):
    """Hacker--Hatemi-J ``W_test``: df-corrected Wald of the first ``base`` cause lags in equation 1.

    ``levels`` [..., n, 2]. With one regressor set shared by both equations, the
    system Wald of equation-1 restrictions with covariance kron(RES'RES/(T'-m),
    (X'X)^-1) equals (RSS_r - RSS_u) / (RSS_u / (T' - m)). Returns
    (statistics [...], ok [...]).
    """
    p = base + augmentation
    lagged, outcome = _hh_design(levels, p, p)
    restricted = [j for j in range(2 * p) if not (j % 2 == 1 and j // 2 < base)]
    resid_u, ok_u = bk.residualize(lagged, outcome[..., 0])
    resid_r, ok_r = bk.residualize(lagged[..., restricted], outcome[..., 0])
    rss_u = resid_u.square().sum(-1)
    rss_r = resid_r.square().sum(-1)
    df = outcome.shape[-2] - (2 * p + 1)
    stats = (rss_r - rss_u) / (rss_u / df)
    return stats.clamp_min(0.0), ok_u & ok_r & torch.isfinite(stats) & (rss_u > 0)


def hh_null(levels, base, augmentation):
    """Restricted VAR(base) by OLS on rows base..n-1 and its leverage-modified residuals.

    Equation 1 (effect) uses a constant and its own lags; equation 2 (cause) is
    unrestricted. Returns intercepts [2], lag matrices [p, 2, 2] (zero beyond
    ``base``; A_j[e, v] multiplies lag j of variable v in equation e) and the
    modified residuals delta_t / sqrt(1 - h_t) of rows p..n-1 [T', 2], where
    h_t is the observation's leverage in its own equation.
    """
    p = base + augmentation
    lagged, outcome = _hh_design(levels, base, base)
    ones = torch.ones((outcome.shape[0], 1), dtype=torch.float64)
    matrices = torch.zeros((p, 2, 2), dtype=torch.float64)
    intercept = torch.zeros(2, dtype=torch.float64)
    modified = torch.zeros(outcome.shape, dtype=torch.float64)
    for equation, columns in ((0, [2 * j for j in range(base)]), (1, list(range(2 * base)))):
        design = torch.cat([ones, lagged[:, columns]], dim=1)
        fit = kernel_call(least_squares, design, outcome[:, equation], drop_collinear=False,
                          need_leverage=True)
        intercept[equation] = fit.beta[0]
        for position, column in enumerate(columns):
            matrices[column // 2, equation, column % 2] = fit.beta[1 + position]
        if bool((fit.leverage >= 1 - 1e-12).any()):
            raise AnalysisError("singular_test_covariance", "A null-model leverage equals one; the "
                                "leverage-adjusted residual is undefined.")
        modified[:, equation] = fit.resid / (1 - fit.leverage).sqrt()
    return intercept, matrices, modified[augmentation:]


def hh_draws(levels, base, augmentation, index, recursive=True):
    """Leveraged-bootstrap Wald statistics for independent per-equation indices ``index`` [B, T', 2].

    ``recursive`` regenerates both series from the observed initial values (the
    authors' GAUSS routine). Otherwise y* = X D~' + delta* with the OBSERVED
    lagged regressors X (the written form Y* = D^Z + delta*); the restricted
    fitted values then cancel from the equation-1 Wald, which depends only on
    the effect-equation shocks.
    """
    p = base + augmentation
    intercept, matrices, modified = hh_null(levels, base, augmentation)
    shocks = torch.stack([modified[index[..., 0], 0], modified[index[..., 1], 1]], dim=-1)
    shocks = shocks - shocks.mean(dim=-2, keepdim=True)
    if recursive:
        simulated = bk.simulate_var(levels[:p], intercept, matrices, shocks)
        return hh_wald(simulated, base, augmentation)
    lagged, outcome = _hh_design(levels, p, p)
    restricted = [j for j in range(2 * p) if not (j % 2 == 1 and j // 2 < base)]
    resid_u, ok_u = bk.residualize(lagged, shocks[..., 0])
    resid_r, ok_r = bk.residualize(lagged[:, restricted], shocks[..., 0])
    rss_u = resid_u.square().sum(-1)
    rss_r = resid_r.square().sum(-1)
    stats = (rss_r - rss_u) / (rss_u / (outcome.shape[0] - (2 * p + 1)))
    return stats.clamp_min(0.0), ok_u & ok_r & torch.isfinite(stats) & (rss_u > 0)


def hh_critical_values(draws):
    """Authors' rule: mean of sorted W* at B - floor(B a) and the next order statistic (1-based)."""
    count = draws.numel()
    ordered = torch.sort(draws).values
    out = []
    for divisor in (100, 20, 10):
        index = count - count // divisor
        step = min(1, count // divisor)
        out.append(float((ordered[index - 1] + ordered[index - 1 + step]) / 2))
    return out


def _leverage_route(levels, raw, y, x, time, signs, base, augmentation, draws, seed, alpha, work,
                    recursive=True):
    p = base + augmentation
    with torch.device("cpu"), torch.no_grad():
        stat, ok = hh_wald(levels, base, augmentation)
        if not bool(ok):
            raise AnalysisError("singular_test_covariance", "The partial-sum VAR design is "
                                "numerically singular or fits the effect exactly.")
        statistic = float(stat)
        lagged, outcome = _hh_design(levels, p, p)
        design = torch.cat([torch.ones((len(outcome), 1), dtype=torch.float64), lagged], dim=1)
        fit = kernel_call(least_squares, design, outcome, drop_collinear=False)
        m = design.shape[1]
        sigma = fit.resid.T @ fit.resid / (len(outcome) - m)
        full_v = torch.kron(sigma.contiguous(), fit.xtx_inv.contiguous())
        intercept, matrices, modified = hh_null(levels, base, augmentation)
        boot, first, critical, probability = None, None, [None, None, None], None
        if draws:
            generator = bk.generator(seed)
            index = torch.randint(len(modified), (draws, len(modified), 2), generator=generator)
            first = index[0].tolist()
            boot, boot_ok = hh_draws(levels, base, augmentation, index, recursive)
            bk.require_success(boot_ok, "leveraged")
            critical = hh_critical_values(boot)
            probability = bk.plus_one_p_value(statistic, boot)
        terms = ["Intercept", *[f"L{lag}.{name}({sign})" for lag in range(1, p + 1)
                                for name, sign in zip((y, x), signs, strict=True)]]
        coefficients = [{"equation": f"{name}({sign})", "term": term,
                         "estimate": float(fit.beta[j, i]),
                         "std_error": float(full_v[i * m + j, i * m + j].sqrt()),
                         "role": "augmentation" if j > 2 * base else "base/deterministic"}
                        for i, (name, sign) in enumerate(zip((y, x), signs, strict=True))
                        for j, term in enumerate(terms)]
        row = {"outcome": y, "cause": x, "outcome_sign": signs[0], "cause_sign": signs[1],
               "statistic": statistic, "df": base, "asymptotic_p_value": chi2_sf(statistic, base),
               "cv_1": critical[0], "cv_5": critical[1], "cv_10": critical[2],
               "bootstrap_p_value": probability,
               "mc_std_error": bk.mc_std_error(probability, draws) if draws else None,
               "reject_5": (statistic > critical[1]) if draws else None}
        return TableSet(
            {"tests": table([row]), "coefficients": table(coefficients)},
            title="Partial-sum predictive noncausality (leveraged bootstrap)",
            source="https://doi.org/10.1007/s00181-011-0484-x",
            bootstrap_source="https://doi.org/10.1080/00036840500405763",
            author_code=_AUTHOR_CODE, resampling="leverage",
            leverage_design="recursive" if recursive else "fixed", base_lags=base,
            dmax=augmentation, signs=list(signs), draws=draws, failed_draws=0, seed=seed,
            alpha=alpha, sample_hash=_frame_hasher(raw).hexdigest(), input_rows=len(raw),
            nobs=len(outcome),
            transformation="zero initial value plus cumsum(max/min(raw differences,0)); initial raw level excluded",
            wald="df-corrected (RSS_r - RSS_u) / (RSS_u / (T' - m)), authors' W_test",
            covariance_matrix=full_v.tolist(), covariance_divisor=len(outcome) - m,
            null_intercepts=intercept.tolist(), null_lag_matrices=matrices.tolist(),
            null_model="restricted VAR(lags) without augmentation lags, OLS by equation on rows lags..n-1",
            leverage="observation-specific h_t of each null equation; residual / sqrt(1 - h_t)",
            null_resampling="independent per-equation draws of the modified residuals of rows "
                            "p..n-1, column mean adjustment after drawing, " + (
                                "recursive generation from the observed initial values (authors' "
                                "GAUSS routine)" if recursive else
                                "fixed observed lagged regressors, y* = X D~' + delta* (written "
                                "form of the papers)"),
            draw_order="torch.randint(T', (B, T', 2)) from torch.Generator(cpu).manual_seed(seed)",
            bootstrap_first_indices=first,
            bootstrap_statistics=boot.tolist() if boot is not None else [],
            critical_value_rule="(W*_(B - floor(B a)) + W*_(B - floor(B a) + 1)) / 2, a = .01, .05, .10",
            decision_rule="reject at 5% when W > cv_5 (authors' rule)",
            p_value_rule="(1 + #{W* >= W}) / (B + 1)", work=work,
            sample_time=raw[time].tolist() if time is not None else None,
            limitations="Fixed lag orders and dmax; leverage-adjusted iid resampling is not a wild "
                        "bootstrap; bootstrap partial sums need not stay monotone",
        )
