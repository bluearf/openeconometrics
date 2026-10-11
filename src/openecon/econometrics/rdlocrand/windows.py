"""Window selection from covariate balance tests (community ``rdwinselect``).

Nested windows. The running variable is centred, ``Rc = R - cutoff``, and
sorted; n0 units lie below the cutoff. Without ``wmin`` the smallest window
holds ``obsmin`` (default 10) units on each side; each following window adds
``wobs`` (default 5) units per side, or widens by ``wstep``:

* symmetric windows [-w, w] (default): from the current positions the window
  grows by whole tied blocks until ``wobs`` new units (or a mass point holding
  ``wobs`` units) are added on each side, then the shorter side is extended so
  that the window is symmetric, w = max(-Rc_left, Rc_right) (the package's
  ``findwobs_sym``);
* ``wasymmetric=True``: each side grows separately and the window is
  [Rc_left, Rc_right] (``findwobs``); ``wmasspoints=True`` adds one mass point
  per side and step;
* ``wstep``: symmetric windows w_min + k wstep, k = 0, 1, ..., capped at
  max(Rc) (R's ``seq(w_min, w_max, by = wstep)``).

``wmin`` and ``wstep`` are distances from the cutoff (``wmin`` a positive
number or a pair [left, right] with left < 0 < right), as in the package.

Tests in each window. An exact binomial test of P(treated) = 1/2 for the unit
counts on the two sides, and, for every covariate, the balance test chosen by
``statistic`` (``diffmeans``, ``ksmirnov``, ``ranksum``; ``hotelling`` tests
all covariates jointly) by randomization inference over the fixed-margins law
or, with ``approx=True``, by its large-sample approximation. The recommended
window is the largest one such that the minimum balance p-value is at least
``level`` in it and in every smaller window; a window without a defined
p-value counts as failing. Units with a missing covariate are excluded from
every window's tests (``dropmissing=False``, the package default) or removed
before the windows are built (``dropmissing=True``).
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table

from . import common as c
from . import kernels as k

STATISTICS = ("diffmeans", "ksmirnov", "ranksum", "hotelling")


@dataclass(frozen=True)
class Window:
    left: float      # reported limits, centred at the cutoff
    right: float
    low: float       # membership limits, centred: low <= Rc <= high
    high: float


class _Sorted:
    """1-based positions on the sorted centred running variable, as in the package."""

    def __init__(self, rc: Tensor, n0: int):
        values = rc.tolist()
        _, inverse, counts = torch.unique_consecutive(rc, return_inverse=True, return_counts=True)
        self.r = [math.nan, *values]
        self.dups = [0, *counts[inverse].tolist()]
        self.n, self.nc = len(values), n0
        self.nt = self.n - n0
        self.left, self.right = values[:n0], values[n0:]
        self.points = (int(torch.unique(rc[:n0]).numel()), int(torch.unique(rc[n0:]).numel()))

    def controls_at_least(self, value: float) -> int:
        return self.nc - bisect_left(self.left, value)

    def treated_at_most(self, value: float) -> int:
        return bisect_right(self.right, value)


def _symmetric(s: _Sorted, wobs: int, windows: int, posl: int, posr: int) -> list[float]:
    """The package's ``findwobs_sym``: half-widths of the next symmetric windows."""
    result: list[float] = []
    while len(result) < windows and wobs < min(posl, s.nt - (posr - s.nc - 1)):
        left0, right0 = posl, posr
        while s.dups[posl] < wobs and left0 - posl + 1 < wobs:
            posl = max(posl - s.dups[posl], 1)
        while s.dups[posr] < wobs and posr - right0 + 1 < wobs:
            posr = min(posr + s.dups[posr], s.n)
        if abs(s.r[posl]) < s.r[posr]:
            posl = s.nc + 1 - s.controls_at_least(-s.r[posr])
        if abs(s.r[posl]) > s.r[posr]:
            posr = s.treated_at_most(abs(s.r[posl])) + s.nc
        result.append(max(-s.r[posl], s.r[posr]))
        posl = max(posl - s.dups[posl], 1)
        posr = min(posr + s.dups[posr], s.n)
    return result


def _asymmetric(s: _Sorted, wobs: int, windows: int, posl: int, posr: int) -> list[tuple[int, int]]:
    """The package's ``findwobs``: (left, right) positions of the next asymmetric windows."""
    result: list[tuple[int, int]] = []
    windows = min(windows, max(s.points))
    while len(result) < windows and wobs < max(posl, s.nt - (posr - s.nc - 1)):
        left0, right0 = posl, posr
        while s.dups[posl] < wobs and left0 - posl + 1 < wobs and posl > 1:
            posl = max(posl - s.dups[posl], 1)
        while s.dups[posr] < wobs and posr - right0 + 1 < wobs and posr < s.n:
            posr = min(posr + s.dups[posr], s.n)
        result.append((posl, posr))
        posl = max(posl - s.dups[posl], 1)
        posr = min(posr + s.dups[posr], s.n)
    return result


def _r_seq(start: float, stop: float, by: float) -> list[float]:
    """R's ``seq(start, stop, by = by)`` for by > 0."""
    delta = stop - start
    if delta == 0 and stop == 0:
        return [stop]
    steps = delta / by
    if steps < 0:
        raise AnalysisError("invalid_window", "The smallest window already exceeds the range of the "
                            "running variable; choose a smaller wmin.")
    if abs(delta) / max(abs(stop), abs(start)) < 100 * c.EPS:
        return [start]
    return [min(start + i * by, stop) for i in range(int(steps + 1e-10) + 1)]


def _too_few(what: str) -> AnalysisError:
    return AnalysisError("insufficient_observations", f"There are not enough observations on each side "
                         f"of the cutoff for {what}; lower obsmin/wobs or supply wmin.")


def sequence(rc: Tensor, n0: int, *, obsmin: int | None, wmin: Any, wobs: int | None,
             wstep: float | None, wasymmetric: bool, wmasspoints: bool, nwindows: int) -> list[Window]:
    """The package's nested windows on the sorted centred running variable ``rc``."""
    s = _Sorted(rc, n0)
    n = s.n
    if wmin is None:
        obsmin = 10 if obsmin is None else obsmin
        if wmasspoints:
            obsmin, wasymmetric = 1, True
        if wasymmetric:
            first = _asymmetric(s, obsmin, 1, n0, n0 + 1)
            if not first:
                raise _too_few(f"a first window with obsmin={obsmin} units per side")
            pos_left, pos_right = first[0]
            wmin_left, wmin_right = s.r[pos_left], s.r[pos_right]
        else:
            first = _symmetric(s, obsmin, 1, n0, n0 + 1)
            if not first:
                raise _too_few(f"a first window with obsmin={obsmin} units per side")
            wmin_left, wmin_right = -first[0], first[0]
    else:
        wmin_left, wmin_right = wmin
        pos_right = n0 + s.treated_at_most(wmin_right)
        pos_left = n0 - s.controls_at_least(wmin_left) + 1
    if wstep is not None:
        top = min(wmin_right + wstep * (nwindows - 1), s.r[n])
        return [Window(-w, w, -w, w) for w in _r_seq(wmin_right, top, wstep)][:nwindows]
    wobs = 1 if wmasspoints else (5 if wobs is None else wobs)
    posl = max(n0 - s.controls_at_least(wmin_left), 1)
    posr = min(n0 + 1 + s.treated_at_most(wmin_right), n)
    if wasymmetric:
        positions = [(pos_left, pos_right), *_asymmetric(s, wobs, nwindows - 1, posl, posr)]
        windows = [Window(wmin_left, wmin_right, s.r[pos_left] if pos_left <= n else math.inf,
                          s.r[pos_right])]
        windows += [Window(s.r[a], s.r[b], s.r[a], s.r[b]) for a, b in positions[1:]]
        return windows[:nwindows]
    widths = [wmin_right, *_symmetric(s, wobs, nwindows - 1, posl, posr)]
    return [Window(-w, w, -w, w) for w in widths][:nwindows]


def _type1_quantile(values: list[float], probability: float) -> float:
    """R's quantile(type = 1) of sorted ``values``."""
    size = len(values)
    position = size * probability
    j = math.floor(position * (1 + 4 * c.EPS))
    index = j + 1 if position > j else j
    return values[min(max(index, 1), size) - 1]


def sample_summary(rc: Tensor, d: Tensor) -> list[dict[str, Any]]:
    """Units per side within the 1st, 5th, 10th and 20th percentiles of the distance to the cutoff."""
    controls = torch.sort(rc[~d].abs()).values.tolist()
    treated = torch.sort(rc[d]).values.tolist()
    rows = []
    for side, values, inside in (("left", controls, lambda q: ((rc >= -q) & (rc < 0)).sum()),
                                 ("right", treated, lambda q: ((rc <= q) & (rc >= 0)).sum())):
        row: dict[str, Any] = {"side": side, "n": len(values)}
        for label, probability in (("n_1pct", 0.01), ("n_5pct", 0.05), ("n_10pct", 0.1), ("n_20pct", 0.2)):
            row[label] = int(inside(round(_type1_quantile(values, probability), 5))) if values else 0
        rows.append(row)
    return rows


@dataclass
class Selection:
    windows: list[Window]
    rows: list[dict[str, Any]]
    balance: list[dict[str, Any]]
    recommended: int | None      # 0-based index of the recommended window
    notes: list[str]


def _balance(x: Tensor, d: Tensor, rc: Tensor, names: list[str], *, statistic: str, p: int,
             kernel: str, evalat: str, w_right: float, approx: bool, reps: int, seed: int,
             method: str, vce: str) -> tuple[list[float], list[float], str, int]:
    """(statistics, p-values, method, draws) of the covariate balance tests in one window."""
    if statistic == "hotelling":
        test = k.Hotelling(x, d)
        if approx:
            return [test.observed], [k.hotelling_f(test.observed, x.shape[0], x.shape[1])], "approximate", 0
        outcome = k.randomize([test], k.Law(d), reps=reps, seed=seed, method=method)
        return [test.observed], [float(outcome.p_values[0][0])], outcome.kind, outcome.draws
    weights = c.window_weights(rc, w_right, kernel, p)
    if p > 0:
        if evalat == "cutoff":
            adjusted = rc
        else:
            adjusted = rc - torch.where(d, rc[d].mean(), rc[~d].mean())
        fit = k.hc_fit(x, d, c.powers(adjusted, p), weights, vce)
        values = [k.z_test(float(e), float(se)) for e, se in zip(fit.estimate, fit.se, strict=True)]
        return fit.estimate.tolist(), [value[1] for value in values], "approximate", 0
    if statistic == "diffmeans":
        test = k.DiffMeans(x, d, weights)
    elif statistic == "ranksum":
        test = k.RankSum(x, d, True)
    else:
        test = k.KolmogorovSmirnov(x, d, True)
    observed = test.observed.tolist()
    if not approx:
        outcome = k.randomize([test], k.Law(d), reps=reps, seed=seed, method=method)
        return observed, outcome.p_values[0].tolist(), outcome.kind, outcome.draws
    if statistic == "diffmeans" and min(int(d.sum()), int((~d).sum())) < 2:
        # Neyman and HC2 standard errors need two units per side; the package reports NaN.
        p_values = [None] * len(names)
    elif statistic == "diffmeans" and weights is None:
        diff, se = k.neyman(x, d)
        p_values = [k.z_test(float(a), float(b))[1] for a, b in zip(diff, se, strict=True)]
    elif statistic == "diffmeans":
        try:
            fit = k.hc_fit(x, d, None, weights, "HC2")
        except AnalysisError as exc:
            if exc.code != "unit_leverage":
                raise
            return observed, [None] * len(names), "approximate", 0
        p_values = [k.z_test(float(a), float(b))[1] for a, b in zip(fit.estimate, fit.se, strict=True)]
    elif statistic == "ranksum":
        p_values = [k.two_sided(z) if math.isfinite(z) else 1.0 for z in observed]
    else:
        p_values = [k.ks_large_sample(test, j) for j in range(len(names))]
    return observed, p_values, "approximate", 0


def _check_work(rc: Tensor, n0: int, complete: Tensor, windows: list[Window], columns: int, *,
                reps: int, method: str) -> None:
    """Refuse before the first draw when all windows' randomization tests exceed the budget."""
    kept = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(complete.to(torch.int64), 0)])
    low = torch.searchsorted(rc, torch.tensor([w.low for w in windows], dtype=c.FLOAT))
    high = torch.searchsorted(rc, torch.tensor([w.high for w in windows], dtype=c.FLOAT), right=True)
    high = torch.maximum(high, low)
    start = torch.minimum(low.clamp(min=n0), high)  # first treated position in the window
    units, treated = (kept[high] - kept[low]).tolist(), (kept[high] - kept[start]).tolist()
    limit = max(reps, k.EXACT_LIMIT)
    work = sum(k.plan(k.fixed_support(n, n1, limit), reps, method)[1] * n
               for n, n1 in zip(units, treated, strict=True) if 0 < n1 < n)
    k.check_work(work * columns, "Window selection")


def select(rc: Tensor, x: Tensor | None, names: list[str], *, obsmin: int | None, wmin: Any,
           wobs: int | None, wstep: float | None, wasymmetric: bool, wmasspoints: bool,
           nwindows: int, statistic: str, p: int, evalat: str, kernel: str, approx: bool,
           level: float, reps: int, seed: int, method: str, vce: str) -> Selection:
    """Windows, per-window tests and the recommended window (centred units, ``rc`` sorted)."""
    d = rc >= 0
    n0 = int((~d).sum())
    notes: list[str] = []
    if bool((rc[1:] == rc[:-1]).any()):
        notes.append("Mass points detected in the running variable; wmasspoints=True builds one "
                     "window per mass point.")
    windows = sequence(rc, n0, obsmin=obsmin, wmin=wmin, wobs=wobs, wstep=wstep,
                       wasymmetric=wasymmetric, wmasspoints=wmasspoints, nwindows=nwindows)
    if len(windows) < nwindows:
        notes.append(f"Only {len(windows)} of the {nwindows} requested windows could be formed; "
                     "consider changing wmin, wobs or wstep.")
    complete = torch.ones_like(d) if x is None else ~torch.isnan(x).any(1)
    if x is not None and not approx:
        _check_work(rc, n0, complete, windows, 1 if statistic == "hotelling" else x.shape[1],
                    reps=reps, method=method)
    rows, balance = [], []
    for index, window in enumerate(windows):
        inside = (rc >= window.low) & (rc <= window.high) & complete
        dw, rw = d[inside], rc[inside]
        n_left, n_right = int((~dw).sum()), int(dw.sum())
        row: dict[str, Any] = {"window": index + 1, "w_left": window.left, "w_right": window.right,
                               "p_value": None, "variable": None, "binomial_p_value": None,
                               "n_left": n_left, "n_right": n_right, "method": None, "draws": None}
        if n_left and n_right:
            row["binomial_p_value"] = k.binomial_half(n_right, n_left + n_right)
            if x is not None:
                observed, p_values, kind, draws = _balance(
                    x[inside], dw, rw, names, statistic=statistic, p=p, kernel=kernel,
                    evalat=evalat, w_right=window.right, approx=approx, reps=reps, seed=seed,
                    method=method, vce=vce)
                labels = ["hotelling"] if statistic == "hotelling" else names
                if None not in p_values:  # an undefined p-value leaves the minimum undefined
                    best = min(range(len(p_values)), key=lambda j: p_values[j])
                    row.update(p_value=p_values[best], variable=labels[best])
                row.update(method=kind, draws=draws or None)
                balance += [{"window": index + 1, "variable": label, "statistic": value, "p_value": pv}
                            for label, value, pv in zip(labels, observed, p_values, strict=True)]
        rows.append(row)
    recommended = None
    if x is not None:
        passing = [row["p_value"] is not None and row["p_value"] >= level for row in rows]
        if passing and passing[0]:
            recommended = passing.index(False) - 1 if False in passing else len(passing) - 1
        else:
            notes.append("The smallest window does not pass the covariate balance test; decrease the "
                         "smallest window or reduce level.")
    else:
        notes.append("No covariates were given: a recommended window needs at least one covariate.")
    return Selection(windows, rows, balance, recommended, notes)


def selection_options(*, obsmin: Any, wmin: Any, wobs: Any, wstep: Any, wasymmetric: Any,
                      wmasspoints: Any, nwindows: Any, statistic: Any, p: Any, evalat: Any,
                      kernel: Any, approx: Any, level: Any, reps: Any, seed: Any, method: Any,
                      vce: Any) -> dict[str, Any]:
    """Validated window-selection options (the package's rules)."""
    statistic = c.statistic_name(statistic, STATISTICS)
    p = c.order_p(p)
    evalat = c.choice(evalat, "evalat", ("cutoff", "means"))
    kernel = c.kernel_name(kernel)
    approx = c.flag(approx, "approx")
    wasymmetric = c.flag(wasymmetric, "wasymmetric")
    wmasspoints = c.flag(wmasspoints, "wmasspoints")
    obsmin = c.count(obsmin, "obsmin", optional=True)
    wobs = c.count(wobs, "wobs", optional=True)
    wstep = c.positive(wstep, "wstep", optional=True)
    nwindows = c.count(nwindows, "nwindows", maximum=10_000)
    reps = c.count(reps, "reps", maximum=k.MAX_REPS)
    seed = c.count(seed, "seed", minimum=0, maximum=2 ** 63 - 1)
    method = c.method_name(method)
    vce = c.vce_name(vce)
    level_value = c.number(level, "level")
    if not 0 <= level_value < 1:
        raise AnalysisError("invalid_option", "level must lie in [0, 1).")
    if p > 0 and statistic != "diffmeans":
        raise AnalysisError("unsupported_option", "Polynomial adjustment (p > 0) is available only "
                            "for statistic='diffmeans' (rdlocrand 3.0 falls back to p = 0 for the "
                            "other balance statistics); use p=0.")
    if kernel != "uniform" and statistic != "diffmeans":
        raise AnalysisError("unsupported_option", "Kernel weights are only allowed with "
                            "statistic='diffmeans'.")
    if kernel != "uniform" and evalat != "cutoff":
        raise AnalysisError("unsupported_option", "Kernel weights can only be combined with "
                            "evalat='cutoff'.")
    if obsmin is not None and wmin is not None:
        raise AnalysisError("invalid_option", "Give obsmin or wmin, not both.")
    if wobs is not None and wstep is not None:
        raise AnalysisError("invalid_option", "Give wobs or wstep, not both.")
    if wmasspoints and any(value is not None for value in (obsmin, wmin, wobs, wstep)):
        raise AnalysisError("invalid_option", "wmasspoints builds windows at the mass points and "
                            "cannot be combined with obsmin, wmin, wobs or wstep.")
    if wmin is not None:
        if isinstance(wmin, Sequence) and not isinstance(wmin, (str, bytes)):
            if len(wmin) != 2:
                raise AnalysisError("invalid_option", "wmin must be a positive number or a pair "
                                    "[left, right] of offsets from the cutoff.")
            left, right = c.number(wmin[0], "wmin"), c.number(wmin[1], "wmin")
            if not left < 0 < right:
                raise AnalysisError("invalid_option", "wmin=[left, right] needs left < 0 < right "
                                    "(offsets from the cutoff).")
            wmin = (left, right)
        else:
            width = c.positive(wmin, "wmin")
            wmin = (-width, width)
    return dict(obsmin=obsmin, wmin=wmin, wobs=wobs, wstep=wstep, wasymmetric=wasymmetric,
                wmasspoints=wmasspoints, nwindows=nwindows, statistic=statistic,
                p=p, evalat=evalat, kernel=kernel, approx=approx or p > 0, level=level_value,
                reps=reps, seed=seed, method=method, vce=vce)


def windows_table(selection: Selection, cutoff: float):
    rows = [{**row, "w_left": row["w_left"] + cutoff, "w_right": row["w_right"] + cutoff}
            for row in selection.rows]
    return table(rows, columns=["window", "w_left", "w_right", "p_value", "variable",
                                "binomial_p_value", "n_left", "n_right", "method", "draws"])


def rdwinselect(*, data: Any, running: str, covariates: Sequence[str] | None = None,
                cutoff: float = 0.0, obsmin: int | None = None, wmin: float | Sequence[float] | None = None,
                wobs: int | None = None, wstep: float | None = None, wasymmetric: bool = False,
                wmasspoints: bool = False, dropmissing: bool = False, nwindows: int = 10,
                statistic: str = "diffmeans", p: int = 0, evalat: str = "cutoff",
                kernel: str = "uniform", approx: bool = False, level: float = 0.15,
                reps: int = 1000, seed: int = 666, method: str = "auto", vce: str = "HC3",
                missing: str = "raise") -> TableSet:
    """Window selection for RD designs under local randomization (community ``rdwinselect``).

    Builds nested windows around the cutoff, tests in each the balance of the
    predetermined ``covariates`` between the units below and above the cutoff
    and the share of units above it (exact binomial test of 1/2), and
    recommends the largest window in which, and in every smaller window, the
    minimum balance p-value is at least ``level`` (Cattaneo, Frandsen and
    Titiunik 2015; Cattaneo, Titiunik and Vazquez-Bare 2016, 2017).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    running : running variable; units with ``running >= cutoff`` are treated.
    covariates : covariate columns for the balance tests (needed for a recommendation).
    cutoff : the RD cutoff.
    obsmin, wmin : units per side in the smallest window (default 10), or its
        half-width (a number, or offsets [left, right] from the cutoff).
    wobs, wstep : units added per side at each step (default 5), or the
        increment of the half-width.
    wasymmetric, wmasspoints : asymmetric windows growing by ``wobs`` units per
        side; one window per mass point of a discrete running variable.
    dropmissing : drop units with missing covariates before building windows
        (default: keep them for the windows, exclude them from the tests).
    nwindows : number of windows (default 10).
    statistic : 'diffmeans' (alias 'ttest'), 'ksmirnov', 'ranksum' or 'hotelling'.
    p, evalat, vce : order of the polynomial adjustment of the covariates
        (p > 0: large-sample HC tests of the interacted polynomial jump at the
        cutoff or, ``evalat='means'``, at the side means), with ``vce`` 'HC1',
        'HC2' or 'HC3'.
    kernel : 'uniform' (default), 'triangular' or 'epanechnikov' weights
        (diffmeans only) on |Rc|/w_right.
    approx : large-sample tests instead of randomization inference.
    level : minimum accepted p-value (default 0.15).
    reps, seed, method : randomization draws, seed of the CPU generator and
        'auto' (enumerate every assignment when there are at most ``reps``),
        'exact' or 'monte_carlo'.
    missing : 'raise' (default) or 'drop' rows with a missing running variable.

    Returns
    -------
    TableSet
        ``windows`` (one row per window: limits in the units of the running
        variable, minimum balance p-value and its covariate, binomial p-value,
        units below/above the cutoff, randomization method and draws),
        ``balance`` (every covariate's statistic and p-value per window) and
        ``sample`` (units per side and within the 1st-20th percentiles of the
        distance to the cutoff). ``attrs`` hold the recommended window
        (``recommended_left``/``recommended_right``, None when no window
        qualifies), the settings and notes.

    Stata/R: ``rdwinselect R X, cutoff(0) wobs(5) nwindows(10) level(.15)``.

    Example::

        sel = oe.rdwinselect(data=df, running="margin", covariates=["pop", "income"],
                             wmin=0.5, wstep=0.125, reps=1000)
        sel.attrs["recommended_left"], sel["windows"]
    """
    options = selection_options(obsmin=obsmin, wmin=wmin, wobs=wobs, wstep=wstep,
                                wasymmetric=wasymmetric, wmasspoints=wmasspoints,
                                nwindows=nwindows, statistic=statistic, p=p, evalat=evalat,
                                kernel=kernel, approx=approx, level=level, reps=reps, seed=seed,
                                method=method, vce=vce)
    running = c.column(running, "running")
    names = c.columns(covariates, "covariates")
    dropmissing = c.flag(dropmissing, "dropmissing")
    cutoff = c.number(cutoff, "cutoff")
    if options["statistic"] == "hotelling" and not names:
        raise AnalysisError("invalid_option", "statistic='hotelling' needs covariates.")
    frame, dropped = c.sample(data, [running], names, missing=missing,
                              optional_missing=dropmissing or missing == "drop")
    if dropmissing and names:
        keep = ~frame[names].isna().any(axis=1)
        dropped += int((~keep).sum())
        frame = frame.loc[keep].reset_index(drop=True)
        if not len(frame):
            raise AnalysisError("empty_sample", "No observations with complete covariates remain.")
    r = c.numeric(frame, running)
    c.check_cutoff(r, cutoff, strict=True)
    rc = r - cutoff
    x = torch.stack([c.numeric(frame, name, allow_missing=True) for name in names], 1) if names else None
    order = c.lexsort([rc, *([] if x is None else list(x.T))])
    rc = rc[order]
    x = None if x is None else x[order]
    result = select(rc, x, names, **options)
    rec = result.recommended
    recommended = None if rec is None else result.windows[rec]
    tables = {"windows": windows_table(result, cutoff),
              "sample": table(sample_summary(rc, rc >= 0),
                              columns=["side", "n", "n_1pct", "n_5pct", "n_10pct", "n_20pct"])}
    if names:
        tables["balance"] = table(result.balance, columns=["window", "variable", "statistic", "p_value"])
    missing_covariates = 0 if x is None else int(torch.isnan(x).any(1).sum())
    return TableSet(
        tables, title="Window selection for RD under local randomization",
        cutoff=cutoff, running=running, covariates=names, n=int(rc.numel()),
        n_left=int((rc < 0).sum()), n_right=int((rc >= 0).sum()), n_dropped=dropped,
        n_missing_covariates=missing_covariates, dropmissing=dropmissing,
        recommended_window=None if rec is None else rec + 1,
        recommended_left=None if recommended is None else recommended.left + cutoff,
        recommended_right=None if recommended is None else recommended.right + cutoff,
        recommended_n_left=None if rec is None else result.rows[rec]["n_left"],
        recommended_n_right=None if rec is None else result.rows[rec]["n_right"],
        level=options["level"], statistic=options["statistic"], p=options["p"],
        evalat=options["evalat"], kernel=options["kernel"], vce=options["vce"] if options["p"] else None,
        inference="large-sample" if options["approx"] else "randomization",
        randomization="fixed margins", reps=options["reps"], seed=options["seed"],
        method=options["method"], nwindows=options["nwindows"],
        stata_parity_validated=False, notes=result.notes, references=c.REFERENCES,
    )
