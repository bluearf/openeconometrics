"""Bootstrap panel predictive noncausality with cross-sectional dependence.

``konyacausality`` -- Kónya (2006): one SUR system of unit equations per
direction, unit-specific Wald statistics and unit-specific bootstrap critical
values from a null-imposed residual bootstrap that resamples whole time
periods (columns of the N x T residual matrix), preserving contemporaneous
cross-unit correlation.

``ekcausality`` -- Emirmahmutoğlu and Köse (2011): unit-specific lag-augmented
(LA-)VAR Wald tests with heterogeneous lag orders and integration orders,
combined by Fisher's statistic -2 sum ln p_i, with a bootstrap distribution of
the Fisher statistic that resamples whole time periods of the null-imposed
unit residuals.

Both run on complete balanced in-memory panels, CPU float64 Torch, with the
replications vectorised in bounded chunks. See
``docs/econometrics/causality-extensions.md`` for formulas, conventions and the
declared domain.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import _frame_hasher
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.unitroot.common import check_choice, check_count
from openecon.econometrics.var import bootstrap_kernels as bk
from openecon.econometrics.var.common import check_alpha
from openecon.econometrics.var.estimators import _lag_order
from openecon.econometrics.var.panel_causality import _panel_sample
from openecon.engines.distributions import chi2_sf, normal_sf
from openecon.engines.linalg import least_squares

KONYA_SOURCE = "https://doi.org/10.1016/j.econmod.2006.04.008"
EK_SOURCE = "https://doi.org/10.1016/j.econmod.2010.10.018"
DEFAULT_MAX_WORK = 20_000_000_000
_CHUNK_BYTES = 64 * 1024**2
_MAX_UNITS = 60
_MAX_LAG = 8
_MAX_STORED_COVARIANCE = 120


def _directions(direction: str, y: str, x: str) -> list[tuple[int, int, str, str]]:
    check_choice(direction, "direction", ("both", "x_to_y", "y_to_x"))
    pairs = []
    if direction in ("both", "x_to_y"):
        pairs.append((0, 1, y, x))
    if direction in ("both", "y_to_x"):
        pairs.append((1, 0, x, y))
    return pairs


def _index_digest(chunks: list[Tensor]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.to(torch.int64).contiguous().numpy().tobytes())
    return digest.hexdigest()


def _chunk(per_draw_doubles: int, draws: int) -> int:
    return max(1, min(draws, _CHUNK_BYTES // max(1, 8 * per_draw_doubles)))


# ---- Kónya (2006) -----------------------------------------------------------------------


def _konya_design(effect: Tensor, cause: Tensor, own: int, cross: int, start: int) -> Tensor:
    return torch.cat([bk.lagged(effect, own, start), bk.lagged(cause, cross, start)], dim=-1)


def _konya_criterion(effect: Tensor, cause: Tensor, own: int, cross: int, start: int,
                     criterion: str) -> float:
    """System information criterion of one SUR specification on a common sample."""
    fit = bk.sur_fit(effect[..., start:], _konya_design(effect, cause, own, cross, start))
    if not bool(fit["ok"]):
        raise AnalysisError("singular_system", "A SUR system in the lag-selection grid is "
                            "numerically singular; reduce maxlag.")
    count, usable = effect.shape[0], effect.shape[1] - start
    resid = fit["resid"]
    factor, ok = bk._spd(resid @ resid.T / usable)
    if not bool(ok):
        raise AnalysisError("singular_residual_covariance", "The SUR residual covariance of a "
                            "lag-selection candidate is singular; use fewer units or lags.")
    logdet = float(2 * factor.diagonal().log().sum())
    # Kónya's criteria (3)-(4): ln|W| + 2 N^2 q / T and ln|W| + N^2 q ln(T) / T,
    # q coefficients per equation, N equations.
    parameters = count * count * (1 + own + cross)
    penalty = 2.0 if criterion == "aic" else math.log(usable)
    return logdet + penalty * parameters / usable


def _konya_direction(levels: Tensor, labels: list[Any], e: int, c: int, effect_name: str,
                     cause_name: str, own: int, cross: int, draws: int, gen: torch.Generator,
                     levels_cv: tuple[float, ...], burn_in: int = 0) -> dict[str, Any]:
    effect, cause = levels[:, :, e], levels[:, :, c]
    count, total = effect.shape
    start = max(own, cross)
    usable = total - start
    q = own + cross
    tested = list(range(own, own + cross))
    outcome = effect[:, start:]
    fit = bk.sur_fit(outcome, _konya_design(effect, cause, own, cross, start))
    if not bool(fit["ok"]):
        raise AnalysisError("singular_system", "The unrestricted SUR system is numerically "
                            "singular (collinear lags, a perfect fit, or a singular residual "
                            "covariance with too few periods for the number of units).")
    stats, ok = bk.unit_wald(fit["scaled"], fit["scaled_cov"], count, q, tested)
    if not bool(ok):
        raise AnalysisError("singular_test_covariance", "A unit's SUR restriction covariance "
                            "is singular; no reduced-rank Wald decision is reported.")
    null = bk.sur_fit(outcome, bk.lagged(effect, own, start))
    if not bool(null["ok"]):
        raise AnalysisError("singular_system", "The null-restricted SUR system is singular.")
    coef = torch.zeros((count, start), dtype=torch.float64)
    coef[:, :own] = null["slopes"]
    # burn_in = 0: observed initial values, T' simulated periods. burn_in > 0: Kónya's
    # implementation remark -- start the recursion at zero, simulate burn_in + T' periods
    # and keep the last T levels (presample lags included) as the bootstrap series.
    presample = effect[:, :start] if burn_in == 0 else torch.zeros((count, start), dtype=torch.float64)
    steps = usable + burn_in
    fixed = torch.zeros((count, steps), dtype=torch.float64)
    residual = null["resid"]                                       # [N, T'] SUR residuals
    boot = torch.empty((draws, count), dtype=torch.float64)
    per_draw = count * count * q * q + 6 * count * steps * (q + 1) + (count * q) ** 2
    size = _chunk(per_draw, draws)
    index_chunks, attempted, first = [], 0, None
    while attempted < draws:
        width = min(size, draws - attempted)
        # Whole periods (columns of the N x T' residual matrix) are drawn jointly.
        index = torch.randint(usable, (width, steps), generator=gen)
        index_chunks.append(index)
        if first is None:
            first = index[0].tolist()
        shocks = residual[:, index].permute(1, 0, 2)              # [width, N, steps]
        simulated = bk.simulate_ar(presample, null["intercepts"], coef, fixed, shocks)[..., -total:]
        sim_fit = bk.sur_fit(simulated[..., start:],
                             _konya_design(simulated, cause.expand(width, -1, -1), own, cross, start))
        sim_stats, sim_ok = bk.unit_wald(sim_fit["scaled"], sim_fit["scaled_cov"], count, q, tested)
        bk.require_success(sim_fit["ok"] & sim_ok, "Kónya SUR")
        boot[attempted:attempted + width] = sim_stats
        attempted += width
    critical = bk.critical_values(boot.T.contiguous(), levels_cv)
    rows = []
    for i, label in enumerate(labels):
        statistic = float(stats[i])
        probability = bk.plus_one_p_value(statistic, boot[:, i])
        rows.append({"direction": f"{cause_name} -> {effect_name}", "effect": effect_name,
                     "cause": cause_name, "unit": label, "statistic": statistic, "df": cross,
                     "asymptotic_p_value": kernel_call(chi2_sf, statistic, cross),
                     **{f"cv_{round(level * 100)}": float(critical[j][i])
                        for j, level in enumerate(levels_cv)},
                     "bootstrap_p_value": probability,
                     "mc_std_error": bk.mc_std_error(probability, draws)})
    terms = ["Intercept", *[f"L{j}.{effect_name}" for j in range(1, own + 1)],
             *[f"L{j}.{cause_name}" for j in range(1, cross + 1)]]
    m = q + 1
    # Full SUR covariance in reporting order (unit blocks: intercept, slopes).
    slope_cov = fit["cov_slopes"].reshape(count, q, count, q)
    means = fit["means"]                                            # [N, q]
    full = torch.zeros((count, m, count, m), dtype=torch.float64)
    full[:, 1:, :, 1:] = slope_cov
    cross_cov = -torch.einsum("ik,ikjl->ijl", means, slope_cov)     # Cov(a_i, b_j)
    full[:, 0, :, 1:] = cross_cov
    full[:, 1:, :, 0] = cross_cov.permute(1, 2, 0)
    full[:, 0, :, 0] = fit["sigma"] / usable + torch.einsum("ik,ikjl,jl->ij", means, slope_cov, means)
    full = full.reshape(count * m, count * m)
    estimates = torch.cat([fit["intercepts"][:, None], fit["slopes"]], dim=1)
    coefficients = [{"direction": f"{cause_name} -> {effect_name}", "unit": label, "term": term,
                     "estimate": float(estimates[i, j]),
                     "std_error": math.sqrt(float(full[i * m + j, i * m + j]))}
                    for i, label in enumerate(labels) for j, term in enumerate(terms)]
    sigma = fit["sigma"]
    scale = sigma.diagonal().sqrt()
    corr = sigma / (scale[:, None] * scale[None, :])
    upper = torch.triu_indices(count, count, offset=1)
    rho = corr[upper[0], upper[1]]
    pairs = count * (count - 1) // 2
    lm = usable * float(rho.square().sum())
    cd = math.sqrt(2 * usable / (count * (count - 1))) * float(rho.sum())
    dependence = [{"direction": f"{cause_name} -> {effect_name}", "test": "Breusch-Pagan LM",
                   "statistic": lm, "df": pairs, "p_value": kernel_call(chi2_sf, lm, pairs),
                   "reference": "chi2, fixed N and large T"},
                  {"direction": f"{cause_name} -> {effect_name}", "test": "Pesaran CD",
                   "statistic": cd, "df": None, "p_value": 2 * kernel_call(normal_sf, abs(cd)),
                   "reference": "two-sided standard normal"}]
    return {"rows": rows, "coefficients": coefficients, "dependence": dependence,
            "covariance": full if count * m <= _MAX_STORED_COVARIANCE else None,
            "sigma": sigma, "own": own, "cross": cross, "start": start, "usable": usable,
            "index_sha256": _index_digest(index_chunks), "first_indices": first,
            "null_intercepts": null["intercepts"], "null_slopes": null["slopes"],
            "chunk": size}


def _lag_pair(value: Any, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise AnalysisError("invalid_lags", f"{name} must be an integer or an (own, cross) pair.")
        own, cross = (check_count(item, name, minimum=1) for item in value)
    else:
        own = cross = check_count(value, name, minimum=1)
    if max(own, cross) > _MAX_LAG:
        raise AnalysisError("invalid_lags", f"{name} is limited to {_MAX_LAG}.")
    return own, cross


def konyacausality(*, data: Any, y: str, x: str, panel: str, time: str, lags: Any = 1,
                   maxlag: int | None = None, criterion: str = "aic", direction: str = "both",
                   bootstrap: int = 999, seed: int = 0, alpha: float = 0.05,
                   burn_in: int = 0, max_work: int = DEFAULT_MAX_WORK) -> TableSet:
    """Kónya (2006) bootstrap panel Granger noncausality with SUR and unit critical values.

    For each direction (x -> y and/or y -> x) the N unit equations

        y_it = a_i + sum_{l=1}^{own} b_il y_i,t-l + sum_{l=1}^{cross} g_il x_i,t-l + e_it

    form one seemingly unrelated regression system estimated by two-step
    feasible GLS (Zellner): Sigma is the N x N covariance U U'/T of the unit
    OLS residuals, and Var(b) = [X'(Sigma^-1 (x) I_T) X]^-1. Unit i's Wald
    statistic tests g_i1 = ... = g_i,cross = 0 with that SUR covariance, so
    contemporaneous cross-unit error correlation enters the estimates and
    the tests. Slopes and intercepts differ by unit; the lag orders are common
    to all units (the paper's system lag structure).

    Unit-specific bootstrap critical values follow the paper's null-imposed
    residual bootstrap: (1) estimate the system under g_i = 0 for all i (same
    SUR estimator) and keep its N x T residual matrix; (2) draw whole periods
    (columns) with replacement, so the cross-unit correlation of each period
    is preserved; (3) rebuild y* recursively from the restricted system with
    the observed initial values and the observed x held fixed; (4) re-estimate
    the unrestricted SUR (Sigma re-estimated) and recompute every unit Wald;
    (5) repeat ``bootstrap`` times. Critical values at 1%, 5% and 10% are the
    order statistics W*_(ceil((B+1)(1-a))) of each unit's own bootstrap
    distribution; the plus-one bootstrap p-value (1 + #{W* >= W})/(B + 1) is
    reported with its Monte Carlo error. The asymptotic chi2(cross) p-value is
    printed only as a diagnostic; the paper's decision uses the bootstrap.
    ``burn_in=k > 0`` instead follows Kónya's implementation remark (2004
    working-paper version): the recursion starts at zero, burn_in + T'
    periods are simulated and only the last T levels are kept.

    ``lags`` is an integer (own = cross) or an (own, cross) pair. With
    ``lags=None`` each direction's (own, cross) is chosen from 1..``maxlag``
    on the common sample of the longest lag by Kónya's system criteria
    ln|W| + 2 N^2 q / T (AIC) or ln|W| + N^2 q ln(T) / T (SIC), W the SUR
    residual covariance and q = 1 + own + cross coefficients per equation,
    and then re-estimated on its own longest sample. The selection is not
    repeated inside the bootstrap.

    Domain: complete balanced panel on one regular time grid, 2..60 units,
    lags <= 8, usable periods T > N + 1 so that Sigma is nonsingular, no
    weights, no missing values, no trend. Levels are used as supplied; the
    procedure assumes stationary (or correctly specified) dynamics as in the
    paper and does not pretest unit roots. Failed bootstrap draws raise
    ``bootstrap_failure``; none are dropped. Seeded private RNG; replications
    vectorised in chunks of at most 64 MiB; ``max_work`` bounds the estimated
    floating-point work. Predictive, not structural, causality.

    Parameters
    ----------
    data : table
        Complete balanced panel.
    y, x : str
        The two variables; ``direction`` chooses x -> y, y -> x or both.
    panel, time : str
        Unit identifier and shared regular time key.
    lags : int, (int, int) or None
        Common own and cross lag orders, or None for criterion selection.
    maxlag : int or None
        Largest lag of the selection grid (required when ``lags`` is None).
    criterion : str
        'aic' or 'sic' system criterion for ``lags=None``.
    direction : str
        'both', 'x_to_y' or 'y_to_x'.
    bootstrap : int
        Replications B, 99..9999.
    seed : int
        Seed of the private generator; draws are made with one
        ``torch.randint(T, (chunk, T))`` call per chunk, direction by direction.
    alpha : float
        Level of the reported bootstrap decision (p <= alpha).
    max_work : int
        Upper bound on the estimated floating-point operations.

    Returns
    -------
    TableSet
        ``tests`` (one row per direction and unit), ``coefficients`` (SUR
        estimates with SUR standard errors), ``dependence`` (Breusch-Pagan LM
        and Pesaran CD on the OLS residual correlations) and, with selection,
        ``lag_selection``.
    """
    if not all(isinstance(v, str) and v for v in (y, x)) or y == x:
        raise AnalysisError("invalid_spec", "y and x must name two distinct nonempty columns.")
    pair = _lag_pair(lags, "lags")
    if pair is None:
        if maxlag is None:
            raise AnalysisError("invalid_lags", "lags=None needs maxlag for criterion selection.")
        top = check_count(maxlag, "maxlag", minimum=1)
        if top > _MAX_LAG:
            raise AnalysisError("invalid_lags", f"maxlag is limited to {_MAX_LAG}.")
        check_choice(criterion, "criterion", ("aic", "sic"))
    else:
        top = max(pair)
    directions = _directions(direction, y, x)
    draws = check_count(bootstrap, "bootstrap")
    bk.check_bootstrap(draws)
    check_count(seed, "seed")
    alpha = check_alpha(alpha)
    burn = check_count(burn_in, "burn_in")
    if burn > 1000:
        raise AnalysisError("invalid_option", "burn_in is limited to 1000 periods.")
    with torch.device("cpu"), torch.no_grad():
        chosen, levels, labels, work_bytes = _panel_sample(data, y, x, panel, time, top,
                                                           minimum=2 * top + 4)
        count, total, _ = levels.shape
        if count > _MAX_UNITS:
            raise AnalysisError("invalid_domain", f"Kónya SUR is limited to {_MAX_UNITS} units.")
        usable = total - top
        q = 2 * top
        if usable <= count + 1 or usable <= q + 2:
            raise AnalysisError("insufficient_observations", "The SUR residual covariance needs "
                                "more usable periods than units plus one, and more than the "
                                "lags per equation plus two.")
        grid = top * top if pair is None else 0
        work = len(directions) * ((draws + 1 + grid) * (count * count * usable * q * q
                                                        + (count * q) ** 3 + 4 * count * usable * q * q)
                                  + draws * count * (usable + burn) * top)
        bk.check_work(work, max_work, "Kónya SUR bootstrap")
        gen = bk.generator(seed)
        results, selection = [], []
        for e, c, effect_name, cause_name in directions:
            if pair is None:
                values = {}
                for own in range(1, top + 1):
                    for cross in range(1, top + 1):
                        values[(own, cross)] = _konya_criterion(levels[:, :, e], levels[:, :, c],
                                                                own, cross, top, criterion)
                        selection.append({"direction": f"{cause_name} -> {effect_name}",
                                          "own_lags": own, "cross_lags": cross,
                                          criterion: values[(own, cross)]})
                own, cross = min(values, key=lambda key: (values[key], key))
            else:
                own, cross = pair
            results.append(_konya_direction(levels, labels, e, c, effect_name, cause_name, own,
                                            cross, draws, gen, bk.LEVELS, burn))
        rows = [row for result in results for row in result["rows"]]
        for row in rows:
            row["reject"] = row["bootstrap_p_value"] <= alpha
        tables = {"tests": table(rows, title="Kónya SUR Wald tests with unit bootstrap critical values"),
                  "coefficients": table([r for result in results for r in result["coefficients"]],
                                        title="SUR coefficients (two-step FGLS)"),
                  "dependence": table([r for result in results for r in result["dependence"]],
                                      title="Cross-sectional dependence of unrestricted OLS residuals")}
        if selection:
            tables["lag_selection"] = table(selection, title=f"System {criterion.upper()} lag selection")
        return TableSet(
            tables, title="Kónya bootstrap panel Granger noncausality", source=KONYA_SOURCE,
            method="SUR Wald with unit-specific null-imposed bootstrap critical values",
            effect_cause=[[r[2], r[3]] for r in directions], panel=panel, time=time,
            units=count, periods=total, unit_labels=labels,
            lags={f"{c} -> {e}": {"own": r["own"], "cross": r["cross"], "start": r["start"],
                                   "usable_periods": r["usable"]}
                  for (_, _, e, c), r in zip(directions, results, strict=True)},
            lag_rule="caller-fixed" if pair is not None else
            f"Kónya system {criterion}: ln|W| + penalty N^2 q / T on the common sample",
            residual_covariance={f"{c} -> {e}": r["sigma"]
                                 for (_, _, e, c), r in zip(directions, results, strict=True)},
            covariance_matrix={f"{c} -> {e}": r["covariance"]
                               for (_, _, e, c), r in zip(directions, results, strict=True)},
            covariance_order="unit blocks [Intercept, own lags, cross lags]; stored when N*m <= 120",
            estimator="two-step FGLS SUR, Sigma = U_ols U_ols'/T",
            null_model="restricted SUR (cross lags zero in every unit)",
            null_resampling="whole periods of the restricted SUR residual matrix, with replacement; "
                            "recursive y* from " + ("observed initial values" if burn == 0 else
                                                    f"zero initial values after a {burn}-period burn-in")
                            + "; observed cause held fixed",
            burn_in=burn,
            null_coefficients={f"{c} -> {e}": {"intercepts": r["null_intercepts"],
                                                "own_lags": r["null_slopes"]}
                               for (_, _, e, c), r in zip(directions, results, strict=True)},
            draws=draws, failed_draws=0, seed=seed, rng="torch.Generator(cpu).manual_seed(seed)",
            draw_order="per direction (x -> y first), chunks of torch.randint(T', (chunk, T' + burn_in))",
            chunk_size={f"{c} -> {e}": r["chunk"] for (_, _, e, c), r in zip(directions, results, strict=True)},
            bootstrap_index_sha256={f"{c} -> {e}": r["index_sha256"]
                                    for (_, _, e, c), r in zip(directions, results, strict=True)},
            bootstrap_first_indices={f"{c} -> {e}": r["first_indices"]
                                     for (_, _, e, c), r in zip(directions, results, strict=True)},
            p_value_rule="(1 + #{W* >= W}) / (B + 1)",
            critical_value_rule="W*_(ceil((B + 1)(1 - a))), a = 0.01, 0.05, 0.10",
            alpha=alpha, estimated_work=work, max_work=max_work,
            estimated_workspace_bytes=work_bytes,
            sample_sha256=_frame_hasher(chosen).hexdigest(), sample_order=f"sorted by {panel}, {time}",
            device="cpu", dtype="float64",
            null="No predictive causality from the cause to the effect in unit i",
            inference_scope="balanced panel, common fixed lags, stationary or correctly specified "
                            "dynamics, homoskedastic serially uncorrelated innovations with "
                            "arbitrary contemporaneous cross-unit correlation",
            notes=["Each unit has its own bootstrap critical values; there is no panel-wide statistic.",
                   "Dependence diagnostics describe the OLS residual correlation that SUR exploits.",
                   "Asymptotic chi2 p-values are diagnostics only; decisions use the bootstrap.",
                   "Predictive noncausality is not a structural causal claim."])


# ---- Emirmahmutoğlu and Köse (2011) ------------------------------------------------------


def _per_unit(value: Any, labels: list[Any], name: str, *, minimum: int, maximum: int) -> list[int]:
    if isinstance(value, dict):
        keys = {}
        for key, item in value.items():
            keys[str(key)] = item
        missing = [label for label in labels if str(label) not in keys]
        if missing or len(keys) != len(labels):
            raise AnalysisError("invalid_option", f"{name} must give exactly one value for every "
                                "unit when passed as a mapping.")
        values = [keys[str(label)] for label in labels]
    else:
        values = [value] * len(labels)
    out = []
    for item in values:
        number = check_count(item, name, minimum=minimum)
        if number > maximum:
            raise AnalysisError("invalid_option", f"{name} must not exceed {maximum}.")
        out.append(number)
    return out


def _ek_wald(effect: Tensor, cause: Tensor, k: int, d: int, start: int) -> tuple[Tensor, Tensor]:
    """Toda-Yamamoto unit Wald T'(RSS_r - RSS_u)/RSS_u, batched over leading dims."""
    p = k + d
    own = bk.lagged(effect, p, start)
    other = bk.lagged(cause, p, start)
    outcome = effect[..., start:]
    unrestricted, ok_u = bk.residualize(torch.cat([own, other], dim=-1), outcome)
    restricted, ok_r = bk.residualize(torch.cat([own, other[..., k:]], dim=-1), outcome)
    rss_u = unrestricted.square().sum(-1)
    rss_r = restricted.square().sum(-1)
    usable = outcome.shape[-1]
    stats = usable * (rss_r - rss_u) / rss_u
    ok = ok_u & ok_r & torch.isfinite(stats) & (rss_u > 0)
    return stats.clamp_min(0.0), ok


def _ek_direction(levels: Tensor, labels: list[Any], e: int, c: int, effect_name: str,
                  cause_name: str, ks: list[int], ds: list[int], draws: int,
                  gen: torch.Generator) -> dict[str, Any]:
    effect, cause = levels[:, :, e], levels[:, :, c]
    count, total = effect.shape
    start = max(k + d for k, d in zip(ks, ds, strict=True))
    usable = total - start
    groups: dict[tuple[int, int], list[int]] = {}
    for unit, key in enumerate(zip(ks, ds, strict=True)):
        groups.setdefault(key, []).append(unit)
    stats = torch.empty(count, dtype=torch.float64)
    for (k, d), members in groups.items():
        value, ok = _ek_wald(effect[members], cause[members], k, d, start)
        if not bool(ok.all()):
            raise AnalysisError("singular_design", "A unit's LA-VAR design is numerically singular "
                                "or fitted exactly; no unit is skipped.")
        stats[members] = value
    log_p = torch.stack([bk.chi2_logsf_integer(stats[i:i + 1], ks[i])[0] for i in range(count)])
    fisher = float(-2 * log_p.sum())
    # Null-imposed unit regressions (first k_i cause lags excluded), with intercept.
    order = torch.zeros((count, start), dtype=torch.float64)
    intercept = torch.zeros(count, dtype=torch.float64)
    fixed = torch.zeros((count, usable), dtype=torch.float64)
    residual = torch.zeros((count, usable), dtype=torch.float64)
    coefficients = []
    for i, label in enumerate(labels):
        k, d = ks[i], ds[i]
        p = k + d
        own = bk.lagged(effect[i], p, start)
        other = bk.lagged(cause[i], p, start)
        outcome = effect[i, start:]
        design = torch.cat([own, other[:, k:], torch.ones((usable, 1), dtype=torch.float64)], dim=1)
        fit = kernel_call(least_squares, design, outcome, drop_collinear=False)
        order[i, :p] = fit.beta[:p]
        fixed[i] = other[:, k:] @ fit.beta[p:p + d] if d else 0.0
        intercept[i] = fit.beta[-1]
        residual[i] = fit.resid - fit.resid.mean()
        full = torch.cat([own, other, torch.ones((usable, 1), dtype=torch.float64)], dim=1)
        unrestricted = kernel_call(least_squares, full, outcome, drop_collinear=False)
        sigma2 = float(unrestricted.ssr) / usable
        terms = [*[f"L{j}.{effect_name}" for j in range(1, p + 1)],
                 *[f"L{j}.{cause_name}" for j in range(1, p + 1)], "Intercept"]
        roles = [*(["augmentation" if j > k else "own_lag"] for j in range(1, p + 1)),
                 *(["augmentation" if j > k else "tested"] for j in range(1, p + 1)), ["deterministic"]]
        coefficients.extend({"direction": f"{cause_name} -> {effect_name}", "unit": label,
                             "term": term, "role": role[0],
                             "estimate": float(unrestricted.beta[j]),
                             "std_error": math.sqrt(sigma2 * float(unrestricted.xtx_inv[j, j]))}
                            for j, (term, role) in enumerate(zip(terms, roles, strict=True)))
    boot = torch.empty(draws, dtype=torch.float64)
    per_draw = 8 * count * usable * (2 * start + 2)
    size = _chunk(per_draw, draws)
    attempted, index_chunks, first = 0, [], None
    while attempted < draws:
        width = min(size, draws - attempted)
        index = torch.randint(usable, (width, usable), generator=gen)
        index_chunks.append(index)
        if first is None:
            first = index[0].tolist()
        shocks = residual[:, index].permute(1, 0, 2)                # [width, N, T']
        simulated = bk.simulate_ar(effect[:, :start], intercept, order, fixed, shocks)
        total_log = torch.zeros(width, dtype=torch.float64)
        ok_all = torch.ones(width, dtype=torch.bool)
        for (k, d), members in groups.items():
            value, ok = _ek_wald(simulated[:, members], cause[members].expand(width, -1, -1), k, d, start)
            ok_all &= ok.all(-1)
            total_log += bk.chi2_logsf_integer(value, k).sum(-1)
        bk.require_success(ok_all, "Emirmahmutoglu-Kose")
        boot[attempted:attempted + width] = -2 * total_log
        attempted += width
    probability = bk.plus_one_p_value(fisher, boot)
    critical = bk.critical_values(boot)
    individual = [{"direction": f"{cause_name} -> {effect_name}", "unit": label, "lags": ks[i],
                   "dmax": ds[i], "statistic": float(stats[i]), "df": ks[i],
                   "p_value": math.exp(float(log_p[i])), "log_p_value": float(log_p[i])}
                  for i, label in enumerate(labels)]
    test = {"direction": f"{cause_name} -> {effect_name}", "effect": effect_name, "cause": cause_name,
            "statistic": fisher, "df": 2 * count,
            "asymptotic_p_value": kernel_call(chi2_sf, fisher, 2 * count),
            "cv_1": critical[0], "cv_5": critical[1], "cv_10": critical[2],
            "bootstrap_p_value": probability, "mc_std_error": bk.mc_std_error(probability, draws)}
    return {"test": test, "individual": individual, "coefficients": coefficients, "start": start,
            "usable": usable, "index_sha256": _index_digest(index_chunks), "first_indices": first,
            "chunk": size}


def _select_unit_lags(levels: Tensor, maxlag: int, criterion: str) -> tuple[list[int], list[dict[str, Any]]]:
    column = "aic" if criterion == "aic" else "sbic"
    chosen, rows = [], []
    for i in range(levels.shape[0]):
        record = _lag_order(levels[i], levels.new_zeros((levels.shape[1], 0)),
                            torch.ones((levels.shape[1], 1), dtype=torch.float64), maxlag, 0.05, True)
        candidates = [(row[column], row["lag"]) for row in record["rows"] if row["lag"] >= 1]
        chosen.append(min(candidates)[1])
        rows.extend({"unit_index": i, "lag": row["lag"], column: row[column]}
                    for row in record["rows"] if row["lag"] >= 1)
    return chosen, rows


def ekcausality(*, data: Any, y: str, x: str, panel: str, time: str, lags: Any = None,
                maxlag: int = 4, criterion: str = "aic", dmax: Any = 1, direction: str = "both",
                bootstrap: int = 999, seed: int = 0, alpha: float = 0.05,
                max_work: int = DEFAULT_MAX_WORK) -> TableSet:
    """Emirmahmutoğlu--Köse (2011) Fisher test of panel Granger noncausality in mixed panels.

    Each unit i has its own levels VAR with lag order k_i and declared maximal
    integration order dmax_i (Toda--Yamamoto augmentation). For x -> y,

        y_it = mu_i + sum_{j=1}^{k_i+d_i} a_ij y_i,t-j + sum_{j=1}^{k_i+d_i} b_ij x_i,t-j + u_it,

    and the unit Wald W_i tests b_i1 = ... = b_ik_i = 0 (the d_i augmentation
    lags stay unrestricted) with the Toda--Yamamoto ML residual variance
    RSS_u/T': W_i = T'(RSS_r - RSS_u)/RSS_u, asymptotically chi2(k_i) whether
    the unit's series are I(0), I(1) or cointegrated, provided d_i is a valid
    upper bound. Individual p-values p_i = P(chi2(k_i) > W_i) are combined by
    Fisher's lambda = -2 sum_i ln p_i (computed from stable log tails), which is
    chi2(2N) under cross-section independence.

    Under cross-sectional dependence the paper's bootstrap distribution of
    lambda is used: (1) estimate each unit's regression under its null by OLS;
    (2) centre the residuals; (3) draw whole periods (rows of the T' x N
    residual matrix) with replacement, preserving cross-unit correlation;
    (4) rebuild y* recursively from the restricted unit equations with the
    observed initial values and the observed x held fixed; (5) recompute every
    W_i*, p_i* and lambda*; (6) repeat ``bootstrap`` times. Bootstrap critical
    values are the order statistics lambda*_(ceil((B+1)(1-a))) at 1%, 5% and
    10%, and the plus-one bootstrap p-value is reported with its MC error.

    All units use the common estimation sample t = P+1..T with P = max_i
    (k_i + d_i), so that whole periods can be resampled jointly.

    ``lags`` is None (per-unit selection over 1..``maxlag`` by the varsoc AIC
    or SBIC of the unit's levels VAR with a constant, on the common sample of
    maxlag), an integer, or a mapping {unit label: k_i}. ``dmax`` is an integer
    0..2 or a mapping {unit label: d_i}; no unit-root pretest is run.

    Domain: complete balanced panel on one regular grid, 2..60 units, k_i <= 8,
    no weights/trend. Failed draws raise ``bootstrap_failure``. Seeded private
    RNG, chunked vectorised replications, ``max_work`` floating-point budget.
    Predictive, not structural, causality.

    Returns
    -------
    TableSet
        ``tests`` (Fisher statistic per direction with asymptotic and bootstrap
        inference), ``individual`` (unit Wald statistics, lags, dmax, p-values),
        ``coefficients`` (unit LA-VAR effect equations) and, with selection,
        ``lag_selection``.
    """
    if not all(isinstance(v, str) and v for v in (y, x)) or y == x:
        raise AnalysisError("invalid_spec", "y and x must name two distinct nonempty columns.")
    top = check_count(maxlag, "maxlag", minimum=1)
    if top > _MAX_LAG:
        raise AnalysisError("invalid_lags", f"maxlag is limited to {_MAX_LAG}.")
    check_choice(criterion, "criterion", ("aic", "sic"))
    directions = _directions(direction, y, x)
    draws = check_count(bootstrap, "bootstrap")
    bk.check_bootstrap(draws)
    check_count(seed, "seed")
    alpha = check_alpha(alpha)
    if lags is not None and not isinstance(lags, dict):
        check_count(lags, "lags", minimum=1)
    if not isinstance(dmax, dict):
        check_count(dmax, "dmax")
    with torch.device("cpu"), torch.no_grad():
        # p=1 only sizes the input-selection workspace; lag-specific checks follow below.
        chosen, levels, labels, work_bytes = _panel_sample(data, y, x, panel, time, 1, minimum=4)
        count, total, _ = levels.shape
        if count > _MAX_UNITS:
            raise AnalysisError("invalid_domain", f"This route is limited to {_MAX_UNITS} units.")
        ds = _per_unit(dmax, labels, "dmax", minimum=0, maximum=2)
        selection: list[dict[str, Any]] = []
        if lags is None:
            if total - top < 2 * top + 4:
                raise AnalysisError("insufficient_observations", "maxlag leaves too few periods for "
                                    "the unit lag-order statistics.")
            ks, selection = _select_unit_lags(levels, top, criterion)
            for row in selection:
                row["unit"] = labels[row.pop("unit_index")]
        else:
            ks = _per_unit(lags, labels, "lags", minimum=1, maximum=_MAX_LAG)
        start = max(k + d for k, d in zip(ks, ds, strict=True))
        usable = total - start
        widest = 2 * start + 1
        if usable < widest + 3:
            raise AnalysisError("insufficient_observations", "The common LA-VAR sample needs at "
                                "least 2*(k_i+d_i)+4 usable periods after the longest lag.")
        work = len(directions) * (draws + 1) * count * (usable * widest * widest * 4 + usable * start)
        bk.check_work(work, max_work, "Emirmahmutoglu-Kose bootstrap")
        gen = bk.generator(seed)
        results = [_ek_direction(levels, labels, e, c, effect_name, cause_name, ks, ds, draws, gen)
                   for e, c, effect_name, cause_name in directions]
        tests = [result["test"] for result in results]
        for row in tests:
            row["reject"] = row["bootstrap_p_value"] <= alpha
        tables = {"tests": table(tests, title="Emirmahmutoglu-Kose Fisher panel tests"),
                  "individual": table([r for result in results for r in result["individual"]],
                                      title="Unit LA-VAR Wald tests"),
                  "coefficients": table([r for result in results for r in result["coefficients"]],
                                        title="Unit LA-VAR effect equations")}
        if selection:
            tables["lag_selection"] = table(selection, title=f"Unit VAR {criterion.upper()} lag selection")
        keys = [f"{c} -> {e}" for _, _, e, c in directions]
        return TableSet(
            tables, title="Emirmahmutoglu-Kose bootstrap Fisher panel noncausality", source=EK_SOURCE,
            method="unit Toda-Yamamoto Wald, Fisher combination, cross-dependent period bootstrap",
            panel=panel, time=time, units=count, periods=total, unit_labels=labels,
            unit_lags=dict(zip(map(str, labels), ks, strict=True)),
            unit_dmax=dict(zip(map(str, labels), ds, strict=True)),
            lag_rule="caller-fixed" if lags is not None else f"unit varsoc {criterion} over 1..{top}",
            common_start=start, usable_periods=usable,
            wald_divisor="T' (Toda-Yamamoto ML residual variance)",
            null_model="unit OLS with the first k_i cause lags excluded; augmentation lags kept",
            null_resampling="whole periods of the centred T' x N null residual matrix with "
                            "replacement; recursive y* from observed initial values; observed "
                            "cause held fixed",
            draws=draws, failed_draws=0, seed=seed, rng="torch.Generator(cpu).manual_seed(seed)",
            draw_order="per direction (x -> y first), chunks of torch.randint(T', (chunk, T'))",
            chunk_size=dict(zip(keys, [r["chunk"] for r in results], strict=True)),
            bootstrap_index_sha256=dict(zip(keys, [r["index_sha256"] for r in results], strict=True)),
            bootstrap_first_indices=dict(zip(keys, [r["first_indices"] for r in results], strict=True)),
            p_value_rule="(1 + #{lambda* >= lambda}) / (B + 1)",
            critical_value_rule="lambda*_(ceil((B + 1)(1 - a))), a = 0.01, 0.05, 0.10",
            alpha=alpha, estimated_work=work, max_work=max_work,
            estimated_workspace_bytes=work_bytes,
            sample_sha256=_frame_hasher(chosen).hexdigest(), sample_order=f"sorted by {panel}, {time}",
            device="cpu", dtype="float64",
            null="No predictive causality from the cause to the effect in any unit",
            alternative="Predictive causality in at least one unit",
            inference_scope="balanced panel, unit-specific fixed lags and valid dmax upper bounds, "
                            "serially uncorrelated homoskedastic innovations with arbitrary "
                            "contemporaneous cross-unit correlation (bootstrap)",
            notes=["The chi2(2N) Fisher p-value assumes cross-section independence; the bootstrap does not.",
                   "Rejection supports causality in at least one unit, not in every unit.",
                   "Predictive noncausality is not a structural causal claim."])
