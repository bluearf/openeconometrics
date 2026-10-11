"""Fixed-size rolling-window residual-bootstrap Granger causality sequence.

Balcilar, Ozdemir and Arslanturk (2010) re-estimate a bivariate VAR(p) in
every fixed-size window and plot the sequence of residual-based bootstrap
p-values of the (modified) likelihood-ratio Granger noncausality statistic
in both directions, to show where predictive causality appears or vanishes.
This module returns that sequence as one bounded row per window.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from openecon.analysis import _coerce_frame, _frame_hasher, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.unitroot.common import check_count, check_magnitude, json_label
from openecon.econometrics.var import bootstrap_kernels as bk
from openecon.econometrics.var.common import check_alpha
from openecon.econometrics.var.toda_yamamoto import _ordered_sample
from openecon.engines.distributions import chi2_sf

SOURCE = "https://doi.org/10.1016/j.eneco.2010.05.015"
DEFAULT_MAX_WORK = 20_000_000_000
_MAX_WINDOWS = 5000
_CHUNK_BYTES = 64 * 1024**2


def _window_design(levels: Tensor, p: int) -> tuple[Tensor, Tensor]:
    """Lag design [..., T', 2p] (effect lags 1..p, then cause lags 1..p) and outcomes [..., T', 2]."""
    effect = bk.lagged(levels[..., 0], p, p)
    cause = bk.lagged(levels[..., 1], p, p)
    return torch.cat([effect, cause], dim=-1), levels[..., p:, :]


def window_statistics(levels: Tensor, p: int) -> tuple[Tensor, Tensor, Tensor]:
    """Modified LR (T' - m) ln(RSS_r / RSS_u) of cause -> effect, batched over windows.

    ``levels`` [..., w, 2] holds (effect, cause). Returns (LR, Wald
    (T' - m)(RSS_r - RSS_u)/RSS_u, ok); m = 2p + 1 regressors with intercept.
    """
    design, outcome = _window_design(levels, p)
    usable = outcome.shape[-2]
    df = usable - 2 * p - 1
    resid_u, ok_u = bk.residualize(design, outcome[..., 0])
    resid_r, ok_r = bk.residualize(design[..., :p], outcome[..., 0])
    rss_u = resid_u.square().sum(-1)
    rss_r = resid_r.square().sum(-1)
    lr = df * torch.log(rss_r / rss_u)
    wald = df * (rss_r - rss_u) / rss_u
    ok = ok_u & ok_r & torch.isfinite(lr) & (rss_u > 0)
    return lr.clamp_min(0.0), wald.clamp_min(0.0), ok


def null_system(levels: Tensor, p: int) -> dict[str, Tensor]:
    """OLS of the bivariate VAR(p) with intercepts under 'cause does not predict effect'.

    ``levels`` [W, w, 2]. The effect equation uses its own lags only; the
    cause equation is unrestricted. Returns intercepts [W, 2], lag matrices
    [W, p, 2, 2] (A_j[e, v] = coefficient of lag j of variable v in equation e)
    and centred residuals [W, T', 2].
    """
    design, outcome = _window_design(levels, p)
    windows, usable = outcome.shape[0], outcome.shape[1]
    ones = torch.ones((windows, usable, 1), dtype=torch.float64)
    matrices = levels.new_zeros((windows, p, 2, 2))
    intercept = levels.new_zeros((windows, 2))
    residual = levels.new_zeros((windows, usable, 2))
    for equation, columns in ((0, design[..., :p]), (1, design)):
        x = torch.cat([columns, ones], dim=-1)
        solution = torch.linalg.lstsq(x, outcome[..., equation:equation + 1], driver="gels").solution[..., 0]
        intercept[:, equation] = solution[:, -1]
        matrices[:, :, equation, 0] = solution[:, :p]
        if equation == 1:
            matrices[:, :, equation, 1] = solution[:, p:2 * p]
        fitted = (x @ solution[..., None])[..., 0]
        residual[..., equation] = outcome[..., equation] - fitted
    residual = residual - residual.mean(dim=1, keepdim=True)
    return {"intercept": intercept, "matrices": matrices, "residual": residual}


def bootstrap_window_draws(levels: Tensor, p: int, index: Tensor) -> tuple[Tensor, Tensor]:
    """Bootstrap LR statistics for windows ``levels`` [W, w, 2] and indices [W, B, T'].

    Joint residual vectors (rows) of the null system are resampled, the null
    VAR is simulated recursively from the window's first p observations, and
    the modified LR is recomputed.
    """
    null = null_system(levels, p)
    windows, draws, usable = index.shape
    shocks = torch.gather(null["residual"][:, None].expand(-1, draws, -1, -1), 2,
                          index[..., None].expand(-1, -1, -1, 2))       # [W, B, T', 2]
    presample = levels[:, None, :p, :].expand(-1, draws, -1, -1)
    simulated = bk.simulate_var(presample, null["intercept"][:, None], null["matrices"][:, None], shocks)
    lr, _, ok = window_statistics(simulated, p)
    return lr, ok


def rollingcausality(*, data: Any, y: str, x: str, window: int, lags: int = 1, step: int = 1,
                     time: str | None = None, bootstrap: int = 499, seed: int = 0,
                     alpha: float = 0.05, max_windows: int = 1000,
                     max_work: int = DEFAULT_MAX_WORK) -> TableSet:
    """Rolling-window residual-bootstrap Granger noncausality tests (one row per window).

    In every window of ``window`` consecutive observations (starting at rows
    0, step, 2 step, ...), a bivariate VAR(p) with intercepts is estimated by
    OLS and, for each direction, the modified likelihood-ratio statistic of
    the effect equation

        LR = (T' - m) ln(RSS_r / RSS_u),   T' = window - p,  m = 2p + 1,

    tests that the p cause lags are zero (Sims' degrees-of-freedom correction;
    the Wald (T' - m)(RSS_r - RSS_u)/RSS_u is also reported). Within a window
    LR and Wald are strictly increasing functions of RSS_r/RSS_u, so their
    bootstrap p-values are identical.

    Residual-based bootstrap per window and direction (Balcilar et al. 2010,
    following Mantalos and Shukur): estimate the VAR under the null (effect
    equation without cause lags, cause equation unrestricted), centre its
    residual vectors, resample whole residual vectors with replacement,
    regenerate both series recursively from the window's first p
    observations, and recompute LR*. The p-value is (1 + #{LR* >= LR})/(B + 1).
    Draws come from a private generator, one ``torch.randint(T', (2, B, T'))``
    call per window in window order (index 0: x -> y, index 1: y -> x).

    Supply series in the form the VAR should use (for example growth rates);
    no differencing, lag selection or parameter-constancy test is performed
    and the per-window bootstrap sum-of-coefficient intervals of the paper are
    not computed (point sums are reported). Windows are overlapping, so the
    p-value sequence is not a set of independent tests and carries no
    multiple-testing correction.

    Domain: complete regular bivariate series, 1 <= lags <= 8, window - lags
    >= 2 lags + 6, at most ``max_windows`` (<= 5000) windows and ``max_work``
    floating-point operations; failed draws raise ``bootstrap_failure``.

    Returns
    -------
    TableSet
        ``windows`` with one row per window: positions and time labels,
        statistics, bootstrap and asymptotic p-values and cause-lag sums for
        both directions.
    """
    if not all(isinstance(v, str) and v for v in (y, x)) or y == x:
        raise AnalysisError("invalid_spec", "y and x must name two distinct nonempty columns.")
    p = check_count(lags, "lags", minimum=1)
    if p > 8:
        raise AnalysisError("invalid_lags", "lags is limited to 8.")
    size = check_count(window, "window", minimum=1)
    stride = check_count(step, "step", minimum=1)
    draws = check_count(bootstrap, "bootstrap")
    bk.check_bootstrap(draws)
    check_count(seed, "seed")
    alpha = check_alpha(alpha)
    limit = check_count(max_windows, "max_windows", minimum=1)
    if limit > _MAX_WINDOWS:
        raise AnalysisError("invalid_option", f"max_windows is limited to {_MAX_WINDOWS}.")
    if size - p < 4 * p + 7:
        raise AnalysisError("insufficient_observations", "window must exceed 5*lags + 6 "
                            "observations so every window regression has residual df >= 6.")
    if time is not None and (not isinstance(time, str) or not time or time in (y, x)):
        raise AnalysisError("invalid_spec", "time must name a distinct nonempty column.")
    raw = _coerce_frame(data)
    if len(raw) < size:
        raise AnalysisError("insufficient_observations", "The series is shorter than one window.")
    count = (len(raw) - size) // stride + 1
    if count > limit:
        raise AnalysisError("too_many_windows", f"{count} windows exceed max_windows={limit}; "
                            "increase step or window, or raise max_windows.")
    usable = size - p
    m = 2 * p + 1
    work = count * 2 * (draws + 1) * (usable * m * m * 6 + usable * p * 8)
    bk.check_work(work, max_work, "Rolling-window bootstrap")
    chosen = _ordered_sample(raw, [y, x], time)
    with torch.device("cpu"), torch.no_grad():
        levels = torch.stack([_numeric(chosen[y], y), _numeric(chosen[x], x)], dim=1)
        check_magnitude(levels, [y, x])
        starts = torch.arange(count) * stride
        windows = torch.stack([levels[s:s + size] for s in starts.tolist()])     # [W, w, 2]
        oriented = (windows, windows.flip(-1))                                   # x -> y, y -> x
        observed = [window_statistics(series, p) for series in oriented]
        for _, _, ok in observed:
            if not bool(ok.all()):
                raise AnalysisError("singular_design", "A window regression is numerically singular "
                                    "or fits exactly; no window is skipped.")
        sums = []
        for series in oriented:
            design, outcome = _window_design(series, p)
            ones = torch.ones((*design.shape[:-1], 1), dtype=torch.float64)
            beta = torch.linalg.lstsq(torch.cat([design, ones], -1), outcome[..., :1],
                                      driver="gels").solution[..., 0]
            sums.append(beta[:, p:2 * p].sum(-1))
        gen = bk.generator(seed)
        p_values = torch.empty((2, count), dtype=torch.float64)
        per_window = 2 * draws * usable * (2 * m + 8)
        chunk = max(1, min(count, _CHUNK_BYTES // (8 * per_window)))
        first = None
        position = 0
        while position < count:
            width = min(chunk, count - position)
            block = torch.stack([torch.randint(usable, (2, draws, usable), generator=gen)
                                 for _ in range(width)])                          # [width, 2, B, T']
            if first is None:
                first = block[0, :, 0].tolist()
            for direction, series in enumerate(oriented):
                lr, ok = bootstrap_window_draws(series[position:position + width], p, block[:, direction])
                bk.require_success(ok, "rolling-window")
                stat = observed[direction][0][position:position + width]
                p_values[direction, position:position + width] = \
                    ((lr >= stat[:, None]).sum(-1) + 1).to(torch.float64) / (draws + 1)
            position += width
        labels = chosen[time].tolist() if time is not None else list(range(len(chosen)))
        rows = []
        for w, start in enumerate(starts.tolist()):
            row: dict[str, Any] = {"window": w, "start_row": start, "end_row": start + size - 1,
                                   "start": json_label(labels[start]),
                                   "end": json_label(labels[start + size - 1])}
            for direction, key in enumerate(("x_to_y", "y_to_x")):
                lr = float(observed[direction][0][w])
                row[f"lr_{key}"] = lr
                row[f"wald_{key}"] = float(observed[direction][1][w])
                row[f"asymptotic_p_{key}"] = kernel_call(chi2_sf, lr, p)
                row[f"bootstrap_p_{key}"] = float(p_values[direction, w])
                row[f"reject_{key}"] = float(p_values[direction, w]) <= alpha
                row[f"cause_sum_{key}"] = float(sums[direction][w])
            rows.append(row)
        share = [float((p_values[d] <= alpha).to(torch.float64).mean()) for d in range(2)]
        return TableSet(
            {"windows": table(rows, title="Rolling-window bootstrap Granger causality")},
            title="Rolling-window residual-bootstrap Granger causality", source=SOURCE,
            effect_cause={"x_to_y": f"{x} -> {y}", "y_to_x": f"{y} -> {x}"},
            window=size, step=stride, lags=p, windows=count, usable_per_window=usable,
            regressors_per_equation=m, statistic="modified LR (T' - m) ln(RSS_r / RSS_u)",
            draws=draws, failed_draws=0, seed=seed, rng="torch.Generator(cpu).manual_seed(seed)",
            draw_order="one torch.randint(T', (2, B, T')) per window in window order",
            bootstrap_first_indices=first,
            null_resampling="null VAR (effect equation without cause lags), centred joint residual "
                            "vectors with replacement, recursive regeneration from window start",
            p_value_rule="(1 + #{LR* >= LR}) / (B + 1)",
            rejection_share={"x_to_y": share[0], "y_to_x": share[1]}, alpha=alpha,
            estimated_work=work, max_work=max_work, max_windows=limit,
            sample_sha256=_frame_hasher(chosen).hexdigest(),
            time_order=f"sorted by {time}" if time else "input row order",
            device="cpu", dtype="float64",
            inference_scope="stationary bivariate VAR(p) within each window, iid innovations",
            notes=["Overlapping windows: the sequence is descriptive, without multiplicity control.",
                   "LR and Wald bootstrap p-values coincide within a window (monotone transform).",
                   "Predictive noncausality is not a structural causal claim."])


__all__ = ["rollingcausality"]
