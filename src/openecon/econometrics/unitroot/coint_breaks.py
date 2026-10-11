"""Residual-based cointegration tests with unknown structural breaks.

``gregory_hansen`` (one break; Gregory and Hansen 1996a, b), ``hatemi_j``
(two breaks; Hatemi-J 2008) and ``maki_coint`` (up to five sequentially
located breaks; Maki 2012). H0 is no cointegration (the residuals of the
cointegrating regression with breaks have a unit root); small, very negative
statistics reject. The statistics follow the authors' published programs
(Hansen's GAUSS/R ``shifts`` program, Hatemi-J's GAUSS ``CItest2b`` and Maki's
GAUSS code as distributed in tspdlib); critical values come from the tables in
``coint_breaks_tables.py``. See ``docs/econometrics/cointegration-breaks.md``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.unitroot import coint_breaks_tables as tables
from openecon.econometrics.unitroot.coint_kernels import (
    adf_scan, batched_ols, gauss_round, phillips_scan,
)
from openecon.econometrics.unitroot.common import (
    check_choice, check_count, check_flag, column_names, load_series, period_label,
    regress, regression_table,
)
from openecon.resources import plan_workspace

MODELS = ("level", "trend", "regime", "regime_trend")
_MODEL_LABELS = {"level": "C (level shift)", "trend": "C/T (level shift with trend)",
                 "regime": "C/S (regime shift)", "regime_trend": "regime and trend shift"}
_MAX_OBS = 20_000
_WORK_LIMIT = 4e10          # multiply-adds admitted for one break search
_STATS = ("ADF*", "Zt*", "Za*")


def _load(data: Any, y: str, x: Any, time: str | None, what: str) -> tuple[Tensor, Any, list[str]]:
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one numeric column.")
    names = column_names(x, "x")
    if not names:
        raise AnalysisError("invalid_spec", f"{what} needs at least one regressor in x.")
    values, labels = load_series(data, [y, *names], time)
    if values.shape[0] > _MAX_OBS:
        raise AnalysisError("workspace_limit", f"{what} admits at most {_MAX_OBS:,} "
                            "observations.")
    # Every model has a constant: deviations from the means leave all residuals unchanged
    # (the break interactions differ by multiples of the break dummies) and keep the
    # least-squares problems well conditioned.
    values = values - values.mean(dim=0)
    for index, name in enumerate([y, *names]):
        if not bool((values[:, index] != values[0, index]).any()):
            raise AnalysisError("constant_series", f"Column '{name}' is constant.")
    return values, labels, names


def _lag_rule(lags: Any, maxlag: Any, default_max: int) -> tuple[str, int]:
    if isinstance(lags, str):
        check_choice(lags, "lags", ("t", "aic", "bic"))
        limit = default_max if maxlag is None else check_count(maxlag, "maxlag")
        return lags, limit
    if maxlag is not None:
        raise AnalysisError("invalid_option", "maxlag applies only with lags='t', 'aic' or 'bic'.")
    return "fixed", check_count(lags, "lags")


def _designs(values: Tensor, breaks: Tensor, model: str) -> Tensor:
    """Cointegrating-regression designs [B, n, p] for break positions ``breaks`` [B, r].

    A break at position T_b (the number of observations before it) enters as
    DU_t = 1(t > T_b); ``trend`` and ``regime_trend`` add a linear trend,
    ``regime_trend`` also DU_t t, and the regime models DU_t x_t.
    """
    n = values.shape[0]
    count = breaks.shape[0]
    x = values[:, 1:]
    position = torch.arange(1, n + 1, dtype=torch.float64)
    trend = (position - position.mean()) / n
    dummies = (position[None, None, :] > breaks[:, :, None].to(torch.float64)).to(torch.float64)
    pieces = [torch.ones((count, n, 1), dtype=torch.float64), dummies.transpose(1, 2)]
    if model in {"trend", "regime_trend"}:
        pieces.append(trend.expand(count, n)[..., None])
    if model == "regime_trend":
        pieces.append((dummies * trend).transpose(1, 2))
    pieces.append(x.expand(count, *x.shape))
    if model in {"regime", "regime_trend"}:
        shifted = dummies[:, :, :, None] * x[None, None, :, :]          # [B, r, n, m]
        pieces.append(shifted.permute(0, 2, 1, 3).reshape(count, n, -1))
    return torch.cat(pieces, dim=2)


def _admit(what: str, candidates: int, n: int, p: int, kmax: int) -> int:
    """Chunk size for a break search, after the work and workspace checks."""
    per_candidate = n * (p * p + sum((k + 1) ** 2 for k in range(kmax + 1)) + 40)
    if candidates * per_candidate > _WORK_LIMIT:
        raise AnalysisError("work_limit", f"{what} would need about "
                            f"{candidates * per_candidate:.2e} operations over {candidates:,} "
                            "candidate break configurations; shorten the sample, raise trim or "
                            "lower maxlag.")
    row_bytes = 8 * n * (3 * p + 3 * (kmax + 2) + 12)
    chunk = max(1, min(candidates, (64 * 1024 ** 2) // row_bytes))
    plan_workspace(what, {"batched designs and factors": chunk * row_bytes,
                          "candidate statistics": 64 * candidates})
    return chunk


def _scan(values: Tensor, breaks: Tensor, model: str, rule: str, kmax: int, what: str,
          critical: float = 1.96, strict: bool = False, phillips: bool = True,
          ) -> dict[str, Tensor]:
    """ADF (and Phillips) statistics of every candidate configuration, chunked."""
    n = values.shape[0]
    p = _designs(values, breaks[:1], model).shape[2]
    if n - p < 8 + kmax:
        raise AnalysisError("insufficient_observations", f"{what} has {n} observations for "
                            f"{p} cointegrating-regression parameters and {kmax} lag(s).")
    if n - 1 - kmax <= kmax + 4:
        raise AnalysisError("invalid_lags", f"{kmax} lag(s) are too many for the residual "
                            f"Dickey-Fuller regressions of {n} observations.")
    chunk = _admit(what, breaks.shape[0], n, p, kmax)
    keys = ("adf", "lags", "adf_ssr", "adf_ok", "za", "zt", "pp_ok", "coint_ok", "coint_ssr")
    out: dict[str, list[Tensor]] = {key: [] for key in keys}
    target = values[:, 0]
    for start in range(0, breaks.shape[0], chunk):
        block = breaks[start:start + chunk]
        fit = batched_ols(_designs(values, block, model), target)
        resid = fit.resid
        adf = adf_scan(resid, rule=rule, kmax=kmax, critical=critical, strict=strict)
        out["adf"].append(adf.statistic)
        out["lags"].append(adf.lags)
        out["adf_ssr"].append(adf.ssr)
        out["adf_ok"].append(adf.ok & fit.ok)
        out["coint_ok"].append(fit.ok)
        out["coint_ssr"].append(fit.ssr)
        if phillips:
            pp = phillips_scan(resid)
            out["za"].append(pp.za)
            out["zt"].append(pp.zt)
            out["pp_ok"].append(pp.ok & fit.ok)
    return {key: torch.cat(parts) for key, parts in out.items() if parts}


def _argmin(statistic: Tensor, ok: Tensor, what: str) -> int:
    masked = torch.where(ok & torch.isfinite(statistic), statistic,
                         torch.full_like(statistic, float("inf")))
    index = int(torch.argmin(masked))
    if not math.isfinite(float(masked[index])):
        raise AnalysisError("collinear_regressors", f"{what}: no candidate break date gives a "
                            "full-rank cointegrating regression with a usable residual test.")
    return index


def _critical_columns(values: dict[str, float | None] | None, levels: tuple[str, ...]) -> list:
    return [None if values is None else values.get(level) for level in levels]


def _column_name(level: str) -> str:
    return "critical_" + level.rstrip("%").replace(".", "_") + "pct"


def _cointegrating_table(values: Tensor, breaks: list[int], model: str, names: list[str],
                         y: str) -> Any:
    design = _designs(values, torch.tensor([breaks], dtype=torch.int64), model)[0]
    terms = ["Intercept", *[f"DU{j + 1}" for j in range(len(breaks))]]
    if model in {"trend", "regime_trend"}:
        terms.append("_trend")
    if model == "regime_trend":
        terms += [f"DU{j + 1}#_trend" for j in range(len(breaks))]
    terms += names
    if model in {"regime", "regime_trend"}:
        terms += [f"DU{j + 1}#{name}" for j in range(len(breaks)) for name in names]
    fit = regress(design, values[:, 0], "The cointegrating regression")
    return regression_table(terms, fit, note="Variables are deviations from their sample "
                            "means and the trend is (t - mean t)/n; OLS standard errors are "
                            "not valid for inference under the null.")


def gregory_hansen(data: Any, y: str, x: list[str], *, time: str | None = None,
                   model: str = "level", lags: int | str = "t", maxlag: int | None = None,
                   trim: float = 0.15, regress: bool = False) -> TableSet:
    """Gregory-Hansen tests for cointegration with one regime shift at an unknown date.

    Model and estimator
    -------------------
    For each candidate break T_b (the number of observations before the shift)
    the cointegrating regression with phi_t = 1(t > T_b),

        level         y_t = mu1 + mu2 phi_t + a'x_t + e_t                      (C)
        trend         y_t = mu1 + mu2 phi_t + b t + a'x_t + e_t                (C/T)
        regime        y_t = mu1 + mu2 phi_t + a1'x_t + a2'x_t phi_t + e_t      (C/S)
        regime_trend  y_t = mu1 + mu2 phi_t + b1 t + b2 t phi_t
                            + a1'x_t + a2'x_t phi_t + e_t,

    is fitted by OLS on all n observations (Gregory and Hansen 1996a, models
    2-4; 1996b, eq. 2.2). Its residuals give ADF(T_b), the t ratio of rho in
    De_t = rho e_(t-1) + sum_(j<=K) c_j De_(t-j) + v_t (no constant), and the
    Phillips statistics Z_alpha(T_b) and Z_t(T_b). The tests are

        ADF* = min ADF(T_b),   Z_t* = min Z_t(T_b),   Z_alpha* = min Z_alpha(T_b)

    over T_b = round(trim n), ..., round((1 - trim) n) (the authors' program,
    GAUSS rounding; the paper's T = (0.15, 0.85)). H0: no cointegration.

    Statistic conventions (Hansen's published program ``shifts``): the lag K
    of ADF(T_b) is chosen at every T_b; ``lags="t"`` (default) tests downward
    from ``maxlag`` (default 6, as in the paper) until the last lagged
    difference has |t| >= 1.96; ``"aic"``/``"bic"`` minimize
    ln(SSR/n1) + c (K+2)/n1 on each order's own sample n1 = n-1-K; an integer
    fixes K. The Phillips correction uses quadratic-spectral weights for lags
    1..floor(M), M = 1.3221 (a2 (n-1))^(1/5) with a2 = 4 b^2/(1-b)^4 from an
    AR(1) fit b to the first-stage innovations (no prewhitening, as the
    program; the paper's text mentions prewhitening).

    Critical values
    ---------------
    Approximate asymptotic values (response-surface intercepts) for
    m = 1..4 regressors: models level/trend/regime from Gregory and Hansen
    (1996a, Table 1), regime_trend from (1996b, Table 1), at 1, 2.5, 5, 10 and
    97.5%. They apply to the paper's trimming 0.15 only; with another
    ``trim`` or m > 4 no critical value is reported. No p-value is tabulated.

    Returns
    -------
    TableSet with ``test`` (rows ADF*, Zt*, Za*: statistic, critical values,
    break_index (0-based position of the last pre-break observation),
    break_period, break_fraction T_b/n, lags) and, with ``regress=True``,
    ``cointegrating_regression`` at the ADF* break. ``attrs`` hold the same
    values plus ``candidates``, ``nobs``, ``inference_status`` and ``notes``.

    Stata: ``ghansen y x, break(level|trend|regime|regimetrend)``
    (community command). EViews: none built in.

    Example
    -------
    >>> oe.gregory_hansen(df, "m", ["y", "r"], model="regime").attrs["statistics"]
    """
    check_choice(model, "model", MODELS)
    check_flag(regress, "regress")
    if isinstance(trim, bool) or not isinstance(trim, (int, float)) or not 0.0 < trim < 0.5:
        raise AnalysisError("invalid_option", "trim must be a fraction in (0, 0.5).")
    rule, kmax = _lag_rule(lags, maxlag, 6)
    what = "The Gregory-Hansen test"
    values, labels, names = _load(data, y, x, time, what)
    n = values.shape[0]
    begin, final = gauss_round(trim * n), gauss_round((1.0 - trim) * n)
    begin, final = max(begin, 1), min(final, n - 1)
    if final < begin:
        raise AnalysisError("insufficient_observations", f"{what} has no candidate break date "
                            "after trimming.")
    breaks = torch.arange(begin, final + 1, dtype=torch.int64)[:, None]
    scan = _scan(values, breaks, model, rule, kmax, what)
    m = len(names)
    critical = tables.gregory_hansen_critical(model, m) if abs(trim - 0.15) < 1e-12 else None
    if critical is None:
        status = ("no critical values: tabulated for the paper's trimming 0.15 and m <= 4 "
                  f"regressors (trim={trim:g}, m={m})")
    else:
        status = "asymptotic critical values (Gregory and Hansen 1996" + \
            ("b, Table 1)" if model == "regime_trend" else "a, Table 1)")
    rows, stats, chosen = [], {}, {}
    for name, key, ok in (("ADF*", "adf", "adf_ok"), ("Zt*", "zt", "pp_ok"),
                          ("Za*", "za", "pp_ok")):
        index = _argmin(scan[key], scan[ok], what)
        tb = int(breaks[index, 0])
        lag = int(scan["lags"][index]) if name == "ADF*" else None
        statistic = float(scan[key][index])
        stats[name], chosen[name] = statistic, tb
        rows.append([statistic, None,
                     *_critical_columns(None if critical is None else critical[name],
                                        tables.LEVELS_GH),
                     tb - 1, period_label(labels, tb - 1), tb / n, lag])
    columns = ["statistic", "p_value", *map(_column_name, tables.LEVELS_GH), "break_index",
               "break_period", "break_fraction", "lags"]
    notes = ["H0: no cointegration; reject for statistics below the critical value.",
             "The ADF lag is chosen separately at every candidate date (Gregory-Hansen program).",
             "No p-value is tabulated by the authors."]
    result = {"test": table(rows, columns=columns, index=list(_STATS))}
    if regress:
        result["cointegrating_regression"] = _cointegrating_table(
            values, [chosen["ADF*"]], model, names, y)
    return TableSet(
        result, title=f"Gregory-Hansen test for cointegration with a regime shift: {y} on "
        f"{', '.join(names)}", test="gregory_hansen", model=model,
        model_label=_MODEL_LABELS[model], statistics=stats, p_value=None,
        critical_values=critical, break_index={k: v - 1 for k, v in chosen.items()},
        break_period={k: period_label(labels, v - 1) for k, v in chosen.items()},
        break_fraction={k: v / n for k, v in chosen.items()},
        lags=int(scan["lags"][_argmin(scan["adf"], scan["adf_ok"], what)]), lag_method=rule,
        maxlag=kmax, trim=float(trim), nobs=n, regressors=m, candidates=int(breaks.shape[0]),
        inference_status=status, null="No cointegration", dtype="float64", device="cpu",
        source="Gregory and Hansen (1996a) J. Econometrics 70; (1996b) Oxford Bull. Econ. "
        "Stat. 58; statistic as in Hansen's published program", notes=notes)


def hatemi_j(data: Any, y: str, x: list[str], *, time: str | None = None,
             lags: int | str = "t", maxlag: int | None = None, trim: float = 0.15) -> TableSet:
    """Hatemi-J tests for cointegration with two regime shifts at unknown dates.

    Model and estimator
    -------------------
    The regime-shift model with two breaks T1 < T2 (Hatemi-J 2008),

        y_t = a0 + a1 D1_t + a2 D2_t + b0'x_t + b1'D1_t x_t + b2'D2_t x_t + u_t,
        Dj_t = 1(t > Tj),

    is fitted by OLS for every pair with T1 = round(trim n), ...,
    round((1 - 2 trim) n) and T2 = T1 + round(trim n), ..., round((1 - trim) n)
    (Hatemi-J's GAUSS program CItest2b: 0.15 / 0.70 / 0.85). ADF*, Z_t* and
    Z_alpha* are the minima over all pairs of the residual statistics of
    ``gregory_hansen`` (same lag rules and Phillips correction).
    H0: no cointegration.

    Critical values
    ---------------
    Hatemi-J (2008, Table 1) for k = 1..4 regressors at 1/5/10% and the
    trimming 0.15, as agreed by three independent transcriptions (the article
    could not be retrieved). The 1% Z_alpha* value for k = 1 is withheld
    because the transcriptions disagree; the published 10% Z_alpha* value
    for k = 1 (-52.232) is out of line with the 5% value but is reproduced as
    printed. No p-value is tabulated.

    Returns
    -------
    TableSet with ``test`` (ADF*, Zt*, Za*: statistic, critical values, both
    break indexes/periods/fractions, ADF lags); ``attrs`` as in
    ``gregory_hansen``.

    Stata: ``hatemij y x`` (community command).
    """
    if isinstance(trim, bool) or not isinstance(trim, (int, float)) or not 0.0 < trim < 1 / 3:
        raise AnalysisError("invalid_option", "trim must be a fraction in (0, 1/3).")
    rule, kmax = _lag_rule(lags, maxlag, 6)
    what = "The Hatemi-J test"
    values, labels, names = _load(data, y, x, time, what)
    n = values.shape[0]
    begin = max(gauss_round(trim * n), 1)
    final1 = gauss_round((1.0 - 2.0 * trim) * n)
    final2 = min(gauss_round((1.0 - trim) * n), n - 1)
    first = torch.arange(begin, final1 + 1, dtype=torch.int64)
    second = torch.arange(2 * begin, final2 + 1, dtype=torch.int64)
    grid1, grid2 = torch.meshgrid(first, second, indexing="ij")
    admissible = grid2 >= grid1 + begin
    breaks = torch.stack([grid1[admissible], grid2[admissible]], dim=1)
    if not breaks.shape[0]:
        raise AnalysisError("insufficient_observations", f"{what} has no admissible pair of "
                            "break dates after trimming.")
    scan = _scan(values, breaks, "regime", rule, kmax, what)
    m = len(names)
    critical = tables.hatemi_j_critical(m) if abs(trim - 0.15) < 1e-12 else None
    status = ("critical values of Hatemi-J (2008, Table 1); the 1% Z_alpha* value for k = 1 is "
              "withheld (transcriptions disagree)" if critical is not None and m == 1 else
              "critical values of Hatemi-J (2008, Table 1)" if critical is not None else
              f"no critical values: tabulated for trim 0.15 and k <= 4 (trim={trim:g}, k={m})")
    rows, stats, chosen = [], {}, {}
    for name, key, ok in (("ADF*", "adf", "adf_ok"), ("Zt*", "zt", "pp_ok"),
                          ("Za*", "za", "pp_ok")):
        index = _argmin(scan[key], scan[ok], what)
        t1, t2 = (int(v) for v in breaks[index])
        statistic = float(scan[key][index])
        stats[name], chosen[name] = statistic, (t1, t2)
        rows.append([statistic, None,
                     *_critical_columns(None if critical is None else critical[name],
                                        tables.LEVELS),
                     t1 - 1, t2 - 1, period_label(labels, t1 - 1), period_label(labels, t2 - 1),
                     t1 / n, t2 / n,
                     int(scan["lags"][index]) if name == "ADF*" else None])
    columns = ["statistic", "p_value", *map(_column_name, tables.LEVELS), "break1_index",
               "break2_index", "break1_period", "break2_period", "break1_fraction",
               "break2_fraction", "lags"]
    return TableSet(
        {"test": table(rows, columns=columns, index=list(_STATS))},
        title=f"Hatemi-J test for cointegration with two regime shifts: {y} on {', '.join(names)}",
        test="hatemi_j", model="regime", statistics=stats, p_value=None, critical_values=critical,
        break_index={k: [v[0] - 1, v[1] - 1] for k, v in chosen.items()},
        break_period={k: [period_label(labels, v[0] - 1), period_label(labels, v[1] - 1)]
                      for k, v in chosen.items()},
        lag_method=rule, maxlag=kmax, trim=float(trim), nobs=n, regressors=m,
        candidates=int(breaks.shape[0]), inference_status=status, null="No cointegration",
        dtype="float64", device="cpu",
        source="Hatemi-J (2008) Empirical Economics 35; statistic as in the author's GAUSS "
        "program CItest2b (Gregory-Hansen residual procedures)",
        notes=["H0: no cointegration; reject for statistics below the critical value.",
               "No p-value is tabulated."])


def maki_coint(data: Any, y: str, x: list[str], *, time: str | None = None,
               breaks: int = 2, model: str = "regime", lags: int | str = "t",
               maxlag: int | None = None, trim: float = 0.10) -> TableSet:
    """Maki's test for cointegration with an unknown number of breaks (at most ``breaks``).

    Model and estimator
    -------------------
    The cointegrating regression with i = 1..m breaks DU_it = 1(t > T_i) is

        level         y_t = mu + sum mu_i DU_it + b'x_t + u_t                      (model 0)
        trend         ... + g t                                                    (model 1)
        regime        y_t = mu + sum mu_i DU_it + b'x_t + sum b_i'x_t DU_it + u_t  (model 2)
        regime_trend  model 2 + g t + sum g_i t DU_it                              (model 3)

    Breaks are located sequentially (Maki's GAUSS program, after Kapetanios
    2005): at step j the previously chosen breaks are kept and one more break
    is tried at every date of every segment between them that leaves
    round(trim n) observations on both sides (segment (a, b]: dates
    a + tb + 1 .. b - tb). Each trial gives the residual ADF t ratio and the
    sum of squared residuals of that ADF regression; the step keeps the
    segment with the smallest t ratio and, within it, the date with the
    smallest ADF-regression SSR. The statistic is the smallest t ratio over all
    trials of all m steps. The lag of every trial's ADF regression is chosen
    by Maki's rule (``lags="t"``: downward from ``maxlag`` = 12 until the last
    lagged difference has |t| > 1.654, each order on its own sample) or fixed
    by an integer. H0: no cointegration.

    Critical values
    ---------------
    Maki (2012, Table 1) for m = 1..5 breaks, k = 1..4 regressors and the four
    models at 1/5/10%, as distributed with the author's code; they apply to the
    author's trimming 0.10. Other settings report no critical value. No
    p-value is tabulated.

    Returns
    -------
    TableSet with ``test`` (one row: statistic, critical values, lags) and
    ``breaks`` (the located dates in time order with the step that added
    them); ``attrs`` hold the statistic, the break indexes/periods and the
    t ratio of every step.
    """
    model = check_choice(model, "model", MODELS)
    breaks = check_count(breaks, "breaks", minimum=1)
    if breaks > 5:
        raise AnalysisError("invalid_option", "Maki's procedure is implemented for at most five "
                            "breaks (the range of the published critical values).")
    if isinstance(trim, bool) or not isinstance(trim, (int, float)) or not 0.0 < trim < 0.5:
        raise AnalysisError("invalid_option", "trim must be a fraction in (0, 0.5).")
    if isinstance(lags, str):
        check_choice(lags, "lags", ("t",))
        rule, kmax = "t", 12 if maxlag is None else check_count(maxlag, "maxlag")
    else:
        if maxlag is not None:
            raise AnalysisError("invalid_option", "maxlag applies only with lags='t'.")
        rule, kmax = "fixed", check_count(lags, "lags")
    what = "The Maki test"
    values, labels, names = _load(data, y, x, time, what)
    n = values.shape[0]
    tb = max(gauss_round(trim * n), 1)
    located: list[int] = []
    steps = []
    for step in range(1, breaks + 1):
        bounds = [0, *sorted(located), n]
        segments = [(bounds[i] + tb + 1, bounds[i + 1] - tb) for i in range(len(bounds) - 1)]
        segments = [(lo, hi) for lo, hi in segments if lo <= hi]
        if not segments:
            raise AnalysisError("insufficient_observations", f"{what}: no admissible date for "
                                f"break {step} with trim={trim:g} and n={n}; lower breaks.")
        trials = torch.cat([torch.arange(lo, hi + 1, dtype=torch.int64) for lo, hi in segments])
        segment_of = torch.cat([torch.full((hi - lo + 1,), i, dtype=torch.int64)
                                for i, (lo, hi) in enumerate(segments)])
        fixed = torch.tensor(located, dtype=torch.int64).reshape(1, -1).expand(
            trials.shape[0], len(located))
        candidates = torch.cat([fixed, trials[:, None]], dim=1)
        trial = trials.tolist()
        scan = _scan(values, candidates, model, rule, kmax, what, critical=1.654, strict=True,
                     phillips=False)
        ok = scan["adf_ok"]
        tau = torch.where(ok, scan["adf"], torch.full_like(scan["adf"], float("inf")))
        ssr = torch.where(ok, scan["adf_ssr"], torch.full_like(scan["adf_ssr"], float("inf")))
        segment_tau = torch.full((len(segments),), float("inf"), dtype=torch.float64)
        segment_tau = segment_tau.scatter_reduce(0, segment_of, tau, reduce="amin")
        best_segment = int(torch.argmin(segment_tau))
        if not math.isfinite(float(segment_tau[best_segment])):
            raise AnalysisError("collinear_regressors", f"{what}: no usable trial date at step "
                                f"{step}.")
        inside = segment_of == best_segment
        local = torch.where(inside, ssr, torch.full_like(ssr, float("inf")))
        chosen = int(torch.argmin(local))
        best_tau = int(torch.argmin(torch.where(inside, tau, torch.full_like(tau, float("inf")))))
        located.append(trial[chosen])
        steps.append({"step": step, "statistic": float(segment_tau[best_segment]),
                      "lags": int(scan["lags"][best_tau]), "break": trial[chosen],
                      "trials": len(trial)})
    best = min(steps, key=lambda item: item["statistic"])
    statistic = best["statistic"]
    m = len(names)
    critical = tables.maki_critical(model, m, breaks) if abs(trim - 0.10) < 1e-12 else None
    status = ("critical values of Maki (2012, Table 1) as distributed with the author's code"
              if critical is not None else
              f"no critical values: tabulated for trim 0.10, k <= 4 and m <= 5 (trim={trim:g}, "
              f"k={m})")
    order = sorted(located)
    break_rows = [[t - 1, period_label(labels, t - 1), t / n, 1 + located.index(t)] for t in order]
    return TableSet(
        {"test": table([[statistic, None, *_critical_columns(critical, tables.LEVELS),
                         best["lags"], breaks]],
                       columns=["statistic", "p_value", *map(_column_name, tables.LEVELS),
                                "lags", "max_breaks"], index=["Maki tau"]),
         "breaks": table(break_rows, columns=["break_index", "break_period", "break_fraction",
                                              "step"])},
        title=f"Maki test for cointegration with up to {breaks} break(s): {y} on "
        f"{', '.join(names)}", test="maki_coint", model=model, statistic=statistic,
        p_value=None, critical_values=critical, break_index=[t - 1 for t in order],
        break_period=[period_label(labels, t - 1) for t in order], steps=steps,
        lag_method=rule, maxlag=kmax, trim=float(trim), nobs=n, regressors=m, max_breaks=breaks,
        inference_status=status, null="No cointegration", dtype="float64", device="cpu",
        source="Maki (2012) Economic Modelling 29; statistic as in the author's GAUSS program",
        notes=["H0: no cointegration; reject for statistics below the critical value.",
               "Breaks are located by the ADF-regression SSR rule of the author's program.",
               "No p-value is tabulated."])
