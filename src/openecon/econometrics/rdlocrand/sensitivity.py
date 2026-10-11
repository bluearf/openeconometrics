"""Sensitivity of randomization inference to the window and the null (community ``rdsensitivity``).

For every window in ``wlist`` (right limits; left limits ``wlist_left`` or the
mirror images around the cutoff) and every effect tau0 in ``tlist``, the
p-value of the sharp null of a constant effect tau0 is computed as in
``rdrandinf``: randomization inference (p = 0) under fixed margins, the same
assignments serving every tau0 of a window, or the large-sample HC test of the
polynomial jump (p > 0). A confidence set for one window inverts its column.

Defaults follow the package: the windows of ``rdwinselect`` without covariates
(10 symmetric windows, 10 units per side in the first, 5 more per step), and a
tau grid of ten steps over the rounded classical 95% interval of the first
window (p = 0 only; p > 0 needs ``tlist``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table

from . import common as c
from . import kernels as k
from . import randinf as ri
from . import windows as win


def _windows(wlist: Any, wlist_left: Any, cutoff: float) -> tuple[list[float], list[float]]:
    rights = [c.number(value, "wlist") for value in _sequence(wlist, "wlist")]
    if wlist_left is None:
        lefts = [2 * cutoff - value for value in rights]
    else:
        lefts = [c.number(value, "wlist_left") for value in _sequence(wlist_left, "wlist_left")]
        if len(lefts) != len(rights):
            raise AnalysisError("invalid_option", "wlist and wlist_left must have the same length.")
    for left, right in zip(lefts, rights, strict=True):
        if not left <= cutoff <= right or not left < right:
            raise AnalysisError("invalid_window", f"Every window must contain the cutoff; [{left:g}, "
                                f"{right:g}] does not.")
    return lefts, rights


def _check_work(r: torch.Tensor, cutoff: float, lefts: list[float], rights: list[float], columns: int,
                *, reps: int, method: str) -> None:
    """Refuse before the first draw when the randomization runs of all windows exceed the budget."""
    rounded, treated = c.round8(r - cutoff), r - cutoff >= 0
    low = torch.tensor([c.round8_scalar(value - cutoff) for value in lefts], dtype=c.FLOAT)
    high = torch.tensor([c.round8_scalar(value - cutoff) for value in rights], dtype=c.FLOAT)
    counts = []
    for side in (rounded[~treated], rounded[treated]):
        side = torch.sort(side).values
        counts.append((torch.searchsorted(side, high, right=True) - torch.searchsorted(side, low)).tolist())
    limit = max(reps, k.EXACT_LIMIT)
    work = sum(k.plan(k.fixed_support(n0 + n1, n1, limit), reps, method)[1] * (n0 + n1)
               for n0, n1 in zip(*counts, strict=True) if n0 and n1)
    k.check_work(work * columns, "The sensitivity grid")


def _sequence(value: Any, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not len(value):
        raise AnalysisError("invalid_option", f"{name} must be a nonempty list of numbers.")
    if len(value) > 10_000:
        raise AnalysisError("invalid_option", f"{name} may hold at most 10,000 values.")
    return list(value)


def rdsensitivity(*, data: Any, y: str, running: str, cutoff: float = 0.0,
                  wlist: Sequence[float] | None = None, wlist_left: Sequence[float] | None = None,
                  tlist: Sequence[float] | None = None, statistic: str = "diffmeans", p: int = 0,
                  evalat: str = "cutoff", kernel: str = "uniform", fuzzy: str | None = None,
                  ci: Sequence[float] | None = None, ci_alpha: float = 0.05, reps: int = 1000,
                  seed: int = 666, method: str = "auto", vce: str = "HC3",
                  missing: str = "raise") -> TableSet:
    """Sensitivity of RD randomization p-values to the window and the null (community ``rdsensitivity``).

    For each window [wlist_left[j], wlist[j]] and each null effect in ``tlist``
    the p-value of the sharp null of a constant effect is computed as by
    ``rdrandinf`` (Cattaneo, Titiunik and Vazquez-Bare 2016): randomization
    inference over the fixed-margins law with the same assignments for every
    effect of a window (p = 0), or the large-sample HC test of the interacted
    polynomial jump (p > 0).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y, running, cutoff : outcome, running variable and cutoff (treated: running >= cutoff).
    wlist, wlist_left : right and left window limits in the units of the
        running variable (default: rdwinselect's ten symmetric windows; the
        left limits default to the mirror images of the right ones).
    tlist : null effects (default for p = 0: ten steps over the rounded
        classical 95% interval in the first window; required for p > 0).
    statistic : 'diffmeans' (alias 'ttest'), 'ksmirnov' or 'ranksum'.
    p, evalat, vce : polynomial order (p > 0: large-sample HC inference),
        evaluation point 'cutoff' or 'means' (side means of the running
        variable in each window), HC variant 'HC1', 'HC2' or 'HC3'.
    kernel : 'uniform', 'triangular' or 'epanechnikov' (diffmeans).
    fuzzy : treatment-received column (Anderson-Rubin statistic).
    ci, ci_alpha : window [left, right] (one of the listed windows) whose
        confidence set for a constant effect is reported, and its alpha.
    reps, seed, method : randomization draws, generator seed and 'auto',
        'exact' or 'monte_carlo'.
    missing : 'raise' (default) or 'drop' incomplete rows.

    Returns
    -------
    TableSet
        ``pvalues`` (tau, window, limits, p-value), ``windows`` (limits, units
        per side, randomization method and draws) and, with ``ci``, ``ci``
        (accepted runs of the grid).

    Stata/R: ``rdsensitivity Y R, wlist(.75(.25)2) tlist(0(1)5)``.

    Example::

        sens = oe.rdsensitivity(data=df, y="vote", running="margin",
                                wlist=[0.75, 1.0, 1.25, 1.5], tlist=[0, 1, 2, 3, 4, 5], reps=2000)
        sens["pvalues"]
    """
    y = c.column(y, "y")
    running = c.column(running, "running")
    fuzzy = c.column(fuzzy, "fuzzy", optional=True)
    statistic = c.statistic_name(statistic, ("diffmeans", "ksmirnov", "ranksum"))
    p = c.order_p(p)
    evalat = c.choice(evalat, "evalat", ("cutoff", "means"))
    kernel = c.kernel_name(kernel)
    vce = c.vce_name(vce)
    cutoff = c.number(cutoff, "cutoff")
    alpha = c.level(ci_alpha, "ci_alpha")
    reps = c.count(reps, "reps", maximum=k.MAX_REPS)
    seed = c.count(seed, "seed", minimum=0, maximum=2 ** 63 - 1)
    method = c.method_name(method)
    taus = None if tlist is None else c.grid(tlist, "tlist")
    if fuzzy is not None and statistic != "diffmeans":
        raise AnalysisError("invalid_option", "A fuzzy design uses the Anderson-Rubin statistic; leave "
                            "statistic at 'diffmeans'.")
    if p > 0 and statistic != "diffmeans":
        raise AnalysisError("unsupported_option", "Polynomial adjustment (p > 0) is available only for "
                            "the difference in means (rdlocrand 3.0 falls back to p = 0 otherwise).")
    if p > 0 and taus is None:
        raise AnalysisError("invalid_option", "p > 0 needs tlist.")
    if kernel != "uniform" and (statistic != "diffmeans" or evalat != "cutoff"):
        raise AnalysisError("unsupported_option", "Kernel weights need statistic='diffmeans' and "
                            "evalat='cutoff'.")
    if wlist is None and wlist_left is not None:
        raise AnalysisError("invalid_option", "wlist_left needs wlist.")
    required = [y, running] + ([fuzzy] if fuzzy is not None else [])
    frame, dropped = c.sample(data, required, missing=missing)
    r = c.numeric(frame, running)
    yv = c.numeric(frame, y)
    t = None if fuzzy is None else c.numeric(frame, fuzzy)
    c.check_cutoff(r, cutoff, strict=False)
    if wlist is None:
        rc = torch.sort(r - cutoff).values
        c.check_cutoff(rc, 0.0, strict=True)
        default = win.sequence(rc, int((rc < 0).sum()), obsmin=None, wmin=None, wobs=5, wstep=None,
                               wasymmetric=False, wmasspoints=False, nwindows=10)
        lefts = [cutoff + window.left for window in default]
        rights = [cutoff + window.right for window in default]
    else:
        lefts, rights = _windows(wlist, wlist_left, cutoff)
    notes: list[str] = []
    if taus is None:
        taus = ri.default_grid(r, yv, t, cutoff, max(rights[0] - cutoff, abs(lefts[0] - cutoff)))
        notes.append("tlist uses the package's default grid (ten steps over the rounded classical 95% "
                     "interval in the first window).")
    if p == 0:
        _check_work(r, cutoff, lefts, rights, len(taus), reps=reps, method=method)
    rows, window_rows, columns = [], [], []
    for index, (left, right) in enumerate(zip(lefts, rights, strict=True)):
        sample = ri.window_sample(r, yv, t, None, cutoff=cutoff, wl=left, wr=right, kernel=kernel,
                                  minimum=1, centred=True)
        if evalat == "means":
            evall = cutoff + float(sample.rc[~sample.d].mean())
            evalr = cutoff + float(sample.rc[sample.d].mean())
        else:
            evall = evalr = cutoff
        p_values, kind, draws = ri.grid_pvalues(sample, taus, statistic, p=p, evall=evall, evalr=evalr,
                                                reps=reps, seed=seed, method=method, vce=vce)
        columns.append(p_values)
        window_rows.append({"window": index + 1, "w_left": left, "w_right": right,
                            "n_left": sample.n_left, "n_right": sample.n_right,
                            "method": kind, "draws": draws or None})
        rows += [{"tau": tau, "window": index + 1, "w_left": left, "w_right": right, "p_value": value}
                 for tau, value in zip(taus, p_values, strict=True)]
    tables = {"pvalues": table(rows, columns=["tau", "window", "w_left", "w_right", "p_value"]),
              "windows": table(window_rows, columns=["window", "w_left", "w_right", "n_left",
                                                     "n_right", "method", "draws"])}
    ci_window = None
    if ci is not None:
        bounds = _sequence(ci, "ci")
        if len(bounds) != 2:
            raise AnalysisError("invalid_option", "ci must be the window [left, right] whose confidence "
                                "set is reported.")
        target = (c.number(bounds[0], "ci"), c.number(bounds[1], "ci"))
        scale = max(1.0, abs(target[0]), abs(target[1]))
        matches = [j for j, (left, right) in enumerate(zip(lefts, rights, strict=True))
                   if abs(left - target[0]) <= 1e-12 * scale and abs(right - target[1]) <= 1e-12 * scale]
        if not matches:
            raise AnalysisError("invalid_option", "The ci window is not one of the listed windows.")
        ci_window = matches[0] + 1
        tables["ci"] = table(ri.confidence_rows(columns[matches[0]], taus, alpha),
                             columns=["ci_low", "ci_high"])
        if len(tables["ci"]) > 1:
            notes.append("The confidence set is disconnected; each ci row is one accepted run of the grid.")
    return TableSet(
        tables, title="Sensitivity analysis for RD under local randomization",
        cutoff=cutoff, outcome=y, running=running, fuzzy=fuzzy, statistic=statistic, p=p,
        evalat=evalat if p > 0 else None, kernel=kernel, vce=vce if p > 0 else None,
        inference="large-sample" if p > 0 else "randomization",
        randomization=None if p > 0 else "fixed margins", tlist=list(taus), n_windows=len(rights),
        n_dropped=dropped, reps=reps, seed=seed, method=method, ci_window=ci_window,
        confidence_level=None if ci is None else 1 - alpha,
        p_value_definition="share of assignments with |T*| >= |T| (no plus-one correction)",
        stata_parity_validated=False, notes=notes, references=c.REFERENCES,
    )
