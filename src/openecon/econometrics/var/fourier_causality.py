"""Fourier Toda--Yamamoto predictive noncausality with gradual (smooth) structural shifts.

Nazlioglu, Gormus and Soytas (2016) replace the constant of the lag-augmented
levels VAR of Toda and Yamamoto (1995) by a Fourier approximation of a
smoothly shifting intercept,

    y_t = a_0 + g_1 sin(2 pi k t / T) + g_2 cos(2 pi k t / T)
          + B_1 y_(t-1) + ... + B_(p+d) y_(t-p-d) + e_t,

(single frequency k), or by the cumulative sum of frequencies 1..k
(Nazlioglu, Soytas and Gormus 2019). The modified Wald statistic restricts only
the first p lags of the cause in the effect equation; the d augmentation lags
and the Fourier terms stay unrestricted.

Conventions follow the method author's GAUSS code (tspdlib ``GCtests.src``,
Saban Nazlioglu; https://github.com/aptech/tspdlib, procedures ``_get_Wald``,
``_get_Wald_bootstrap``, ``_get_VARdata``, ``_get_kp``): W = (RSS_r - RSS_u) /
(RSS_u / (T' - m)) on the effect equation, and a fixed-design residual
bootstrap: centred unrestricted effect-equation residuals are resampled with
replacement and added to the restricted fitted values, with the observed
regressors held fixed. Because the restricted fit lies in both column spaces,
W* depends only on the resampled residuals. Differences from that code are
declared in ``docs/econometrics/causality-extensions.md``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import _coerce_frame, _frame_hasher, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.unitroot.common import check_choice, check_count, check_flag, check_magnitude
from openecon.econometrics.var import bootstrap_kernels as bk
from openecon.econometrics.var.common import check_alpha, variable_list
from openecon.econometrics.var.estimators import _lag_order
from openecon.econometrics.var.kernels import lag_block
from openecon.econometrics.var.toda_yamamoto import _budget, _ordered_sample
from openecon.engines.distributions import chi2_sf
from openecon.engines.linalg import least_squares

SOURCE = "https://doi.org/10.1016/j.eneco.2016.09.009"
CUMULATIVE_SOURCE = "https://doi.org/10.1080/1540496X.2018.1434072"
AUTHOR_CODE = "https://github.com/aptech/tspdlib/blob/master/src/GCtests.src"
DEFAULT_MAX_WORK = 20_000_000_000
_MAX_FREQUENCY = 5
_MAX_LAG = 12


def fourier_terms(total: int, frequency: int, cumulative: bool) -> tuple[Tensor, list[str]]:
    """[T, 2 * n] sin/cos columns at t = 1..T for frequency k (or 1..k when cumulative)."""
    t = torch.arange(1, total + 1, dtype=torch.float64)
    columns, names = [], []
    for j in (range(1, frequency + 1) if cumulative else (frequency,)):
        angle = 2 * math.pi * j * t / total
        columns.extend([angle.sin(), angle.cos()])
        names.extend([f"sin({j})", f"cos({j})"])
    return torch.stack(columns, dim=1), names


def _wald_statistics(design_u: Tensor, design_r: Tensor, outcome: Tensor) -> tuple[Tensor, Tensor]:
    """W = (RSS_r - RSS_u) / (RSS_u / (T' - m)) with an intercept swept out, batched in outcome."""
    resid_u, ok_u = bk.residualize(design_u, outcome)
    resid_r, ok_r = bk.residualize(design_r, outcome)
    rss_u = resid_u.square().sum(-1)
    rss_r = resid_r.square().sum(-1)
    df = outcome.shape[-1] - design_u.shape[-1] - 1
    stats = (rss_r - rss_u) / (rss_u / df)
    return stats.clamp_min(0.0), ok_u & ok_r & torch.isfinite(stats) & (rss_u > 0)


def fixed_design_draws(design_u: Tensor, design_r: Tensor, residual: Tensor, index: Tensor) -> tuple[Tensor, Tensor]:
    """Bootstrap Wald statistics for resampling indices ``index`` [B, T'].

    e* = (u - mean u)[index], re-centred, is added to restricted fitted values;
    W* is invariant to those fitted values, so it is evaluated on e* directly.
    """
    centred = residual - residual.mean()
    draws = centred[index]
    draws = draws - draws.mean(-1, keepdim=True)
    return _wald_statistics(design_u, design_r, draws)


def fouriertycausality(*, data: Any, y: Any, lags: int | None = None, maxlag: int = 8,
                       dmax: int = 1, frequency: int | None = None, kmax: int = 3,
                       cumulative: bool = False, criterion: str = "aic", time: str | None = None,
                       causes: Any = None, effects: Any = None, bootstrap: int = 999,
                       seed: int = 0, alpha: float = 0.05,
                       max_work: int = DEFAULT_MAX_WORK) -> TableSet:
    """Fourier Toda--Yamamoto tests with single or cumulative frequencies and bootstrap p-values.

    Fits one levels VAR(p + dmax) with a constant and Fourier terms
    sin(2 pi k t/T), cos(2 pi k t/T), t = 1..T (T level observations), or the
    cumulative set for frequencies 1..k, and tests each directed pair
    cause -> effect (other variables are controls) by restricting the first p
    cause lags in the effect equation. The dmax extra lags and the Fourier
    terms are unrestricted.

    Selection: with ``frequency=None`` the frequency k is chosen from
    1..``kmax`` and with ``lags=None`` the lag p from 1..``maxlag`` by the
    minimum of the VAR(p) (no augmentation) information criterion with the
    candidate Fourier terms, AIC or SIC in Stata varsoc form (-2 lnL/T +
    penalty * K * m/T, m counting every regressor per equation including the
    Fourier terms), on the common sample of the longest lag. Ties go to the
    smaller frequency, then the smaller lag. The selection is not repeated in
    the bootstrap.

    Statistic: W = (RSS_r - RSS_u)/(RSS_u/(T' - m)) (= p times the ordinary
    F), asymptotic chi2(p) for a valid dmax upper bound; T' = T - p - dmax and
    m regressors per equation.

    Bootstrap (method author's code): the centred unrestricted residuals of
    the effect equation are resampled with replacement (one
    ``torch.randint(T', (B, T'))`` call per pair, in table order, from a
    private generator), re-centred, and added to the restricted fitted values
    with the observed regressors fixed; W* is recomputed B times. The
    bootstrap p-value is (1 + #{W* >= W})/(B + 1) with its MC error.

    Domain: complete regular in-memory series, 2..10 variables, p <= 12,
    dmax 0..2, frequencies 1..5, no weights or trend. Fixed-design resampling
    conditions on the observed lags and assumes iid innovations; it is not a
    recursive, wild or block bootstrap. Predictive, not structural, causality.

    Returns
    -------
    TableSet
        ``tests`` (one row per directed pair), ``coefficients`` (full
        augmented VAR with df-corrected standard errors) and ``selection``
        (criterion grid) when anything is selected.
    """
    names = variable_list(y)
    if not 2 <= len(names) <= 10:
        raise AnalysisError("invalid_spec", "y must contain between 2 and 10 variables.")
    top = check_count(maxlag, "maxlag", minimum=1)
    if top > _MAX_LAG:
        raise AnalysisError("invalid_lags", f"maxlag is limited to {_MAX_LAG}.")
    if lags is not None:
        lags = check_count(lags, "lags", minimum=1)
        if lags > _MAX_LAG:
            raise AnalysisError("invalid_lags", f"lags is limited to {_MAX_LAG}.")
    augmentation = check_count(dmax, "dmax")
    if augmentation > 2:
        raise AnalysisError("unsupported_integration_order", "dmax must be 0, 1 or 2.")
    largest = check_count(kmax, "kmax", minimum=1)
    if largest > _MAX_FREQUENCY:
        raise AnalysisError("invalid_option", f"kmax is limited to {_MAX_FREQUENCY}.")
    if frequency is not None:
        frequency = check_count(frequency, "frequency", minimum=1)
        if frequency > _MAX_FREQUENCY:
            raise AnalysisError("invalid_option", f"frequency is limited to {_MAX_FREQUENCY}.")
    check_flag(cumulative, "cumulative")
    check_choice(criterion, "criterion", ("aic", "sic"))
    draws = check_count(bootstrap, "bootstrap")
    bk.check_bootstrap(draws, allow_zero=True)
    check_count(seed, "seed")
    alpha = check_alpha(alpha)
    if time is not None and (not isinstance(time, str) or not time or time in names):
        raise AnalysisError("invalid_spec", "time must name a distinct nonempty column.")
    selected_causes = variable_list(causes, "causes") if causes is not None else names
    selected_effects = variable_list(effects, "effects") if effects is not None else names
    if set(selected_causes + selected_effects) - set(names):
        raise AnalysisError("invalid_spec", "causes and effects must belong to y.")
    pairs = [(effect, cause) for effect in selected_effects for cause in selected_causes
             if cause != effect]
    if not pairs:
        raise AnalysisError("invalid_spec", "The selected groups define no directed test.")
    k = len(names)
    raw = _coerce_frame(data)
    max_terms = 2 * ((largest if frequency is None else frequency) if cumulative else 1)
    max_order = (lags if lags is not None else top) + augmentation
    _budget(len(raw), k, max_order, 1 + max_terms)
    chosen = _ordered_sample(raw, names, time)
    with torch.device("cpu"), torch.no_grad():
        levels = torch.stack([_numeric(chosen[name], name) for name in names], dim=1)
        check_magnitude(levels, names)
        total = len(levels)
        frequencies = [frequency] if frequency is not None else list(range(1, largest + 1))
        column = "aic" if criterion == "aic" else "sbic"
        selection_rows: list[dict[str, Any]] = []
        if lags is None or frequency is None:
            search_top = top if lags is None else lags
            best = None
            for kk in frequencies:
                terms, _ = fourier_terms(total, kk, cumulative)
                record = _lag_order(levels, terms, torch.ones((total, 1), dtype=torch.float64),
                                    search_top, 0.05, True)
                for row in record["rows"]:
                    if row["lag"] < 1 or (lags is not None and row["lag"] != lags):
                        continue
                    value = row[column]
                    selection_rows.append({"frequency": kk, "lags": row["lag"], criterion: value,
                                           "observations": record["n_obs"]})
                    key = (value, kk, row["lag"])
                    if best is None or key < best:
                        best = key
            if best is None:
                raise AnalysisError("insufficient_observations", "No lag/frequency candidate could "
                                    "be evaluated; reduce maxlag or kmax.")
            _, chosen_k, chosen_p = best
        else:
            chosen_k, chosen_p = frequency, lags
        p = chosen_p
        order = p + augmentation
        terms, term_names = fourier_terms(total, chosen_k, cumulative)
        usable = total - order
        m = k * order + 1 + terms.shape[1]
        if usable - m < 3:
            raise AnalysisError("insufficient_observations", "The augmented Fourier VAR needs at "
                                "least three residual degrees of freedom.")
        work = len(pairs) * (draws + 1) * usable * m * m * 4
        bk.check_work(work, max_work, "Fourier Toda-Yamamoto bootstrap")
        lagged_levels = lag_block(levels, order)                    # variable-major
        fourier = terms[order:]
        slopes = torch.cat([lagged_levels, fourier], dim=1)          # intercept swept by demeaning
        regressors = [f"L{j}.{name}" for name in names for j in range(1, order + 1)]
        outcome_all = levels[order:]
        gen = bk.generator(seed)
        rows, first_indices = [], []
        for effect, cause in pairs:
            e, c = names.index(effect), names.index(cause)
            tested = [c * order + j for j in range(p)]
            keep = [j for j in range(slopes.shape[1]) if j not in tested]
            design_u, design_r = slopes, slopes[:, keep]
            outcome = outcome_all[:, e]
            stat, ok = _wald_statistics(design_u, design_r, outcome)
            if not bool(ok):
                raise AnalysisError("singular_design", "The augmented Fourier VAR design is "
                                    "numerically singular or fits the effect exactly.")
            statistic = float(stat)
            row = {"effect": effect, "cause": cause, "lags": p, "dmax": augmentation,
                   "frequency": chosen_k, "cumulative": cumulative, "statistic": statistic,
                   "df": p, "asymptotic_p_value": kernel_call(chi2_sf, statistic, p)}
            if draws:
                residual, _ = bk.residualize(design_u, outcome)
                index = torch.randint(usable, (draws, usable), generator=gen)
                first_indices.append(index[0].tolist())
                boot, boot_ok = fixed_design_draws(design_u, design_r, residual, index)
                bk.require_success(boot_ok, "Fourier Toda-Yamamoto")
                probability = bk.plus_one_p_value(statistic, boot)
                critical = bk.critical_values(boot)
                row.update({"cv_1": critical[0], "cv_5": critical[1], "cv_10": critical[2],
                            "bootstrap_p_value": probability,
                            "mc_std_error": bk.mc_std_error(probability, draws),
                            "reject": probability <= alpha})
            else:
                row.update({"cv_1": None, "cv_5": None, "cv_10": None, "bootstrap_p_value": None,
                            "mc_std_error": None, "reject": row["asymptotic_p_value"] <= alpha})
            rows.append(row)
        design = torch.cat([lagged_levels, torch.ones((usable, 1), dtype=torch.float64), fourier], dim=1)
        fit = kernel_call(least_squares, design, outcome_all, drop_collinear=False)
        sigma = fit.resid.T @ fit.resid / (usable - m)
        labels = regressors + ["Intercept"] + term_names
        coefficients = [{"equation": name, "term": term, "estimate": float(fit.beta[j, i]),
                         "std_error": math.sqrt(float(sigma[i, i] * fit.xtx_inv[j, j])),
                         "role": ("tested_lag" if j < k * order and j % order < p else
                                  "augmentation" if j < k * order else
                                  "deterministic" if term == "Intercept" else "fourier")}
                        for i, name in enumerate(names) for j, term in enumerate(labels)]
        tables = {"tests": table(rows, title="Fourier Toda-Yamamoto modified Wald tests"),
                  "coefficients": table(coefficients, title="Augmented Fourier levels VAR")}
        if selection_rows:
            tables["selection"] = table(selection_rows, title=f"Frequency and lag selection ({criterion})")
        return TableSet(
            tables, title=("Cumulative" if cumulative else "Single") + " Fourier Toda-Yamamoto causality",
            source=CUMULATIVE_SOURCE if cumulative else SOURCE, author_code=AUTHOR_CODE,
            variables=names, lags=p, dmax=augmentation, fitted_lags=order, frequency=chosen_k,
            cumulative=cumulative, fourier_terms=term_names, criterion=criterion,
            selection_rule=("caller-fixed" if lags is not None and frequency is not None else
                            f"minimum varsoc {column} of VAR(p) with Fourier terms on the common "
                            "sample; ties to smaller frequency then lag"),
            observations=usable, levels=total, regressors_per_equation=m, residual_df=usable - m,
            wald="(RSS_r - RSS_u) / (RSS_u / (T' - m)) on the effect equation",
            residual_covariance=sigma, draws=draws, failed_draws=0, seed=seed,
            rng="torch.Generator(cpu).manual_seed(seed)",
            draw_order="one torch.randint(T', (B, T')) call per tested pair, in table order",
            bootstrap="fixed-design residual resampling of the effect equation (author code)",
            bootstrap_first_indices=first_indices,
            p_value_rule="(1 + #{W* >= W}) / (B + 1)",
            critical_value_rule="W*_(ceil((B + 1)(1 - a))), a = 0.01, 0.05, 0.10",
            alpha=alpha, estimated_work=work, max_work=max_work,
            sample_sha256=_frame_hasher(chosen).hexdigest(),
            time_order=f"sorted by {time}" if time else "input row order",
            device="cpu", dtype="float64",
            null="First p lags of the cause are jointly zero in the effect equation",
            inference_scope="valid dmax upper bound, smooth intercept shifts captured by the Fourier "
                            "terms, iid innovations for the fixed-design bootstrap",
            notes=["Fourier terms and dmax lags are unrestricted nuisance regressors.",
                   "The bootstrap conditions on the observed regressors (fixed design).",
                   "Predictive noncausality is not a structural causal claim."])
