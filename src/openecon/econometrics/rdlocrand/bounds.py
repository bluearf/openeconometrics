"""Rosenbaum sensitivity bounds under local randomization (community ``rdrbounds``).

Within each symmetric window |R - c| <= w the baseline p-value treats the
assignment as Bernoulli trials with the window's treated share (or the unit
probabilities ``prob``); ``fmpval`` adds the fixed-margins p-value. Hidden bias
of strength Gamma = exp(gamma) lets every unit's treatment probability lie in
[1/(1 + Gamma), Gamma/(1 + Gamma)]. As in the package (after Rosenbaum 2002),
for u = 1, ..., n the u units with the largest outcomes receive the high
probability and the others the low one; the upper (lower) bound is the
maximum (minimum) of these n p-values. Each configuration's p-value is a
Bernoulli randomization p-value (``rdrandinf``'s statistic on Y - tau0 D);
Monte Carlo configurations share one stream of uniforms (common random
numbers), so the search is not inflated by independent simulation noise.

Searching the smallest-outcome configurations for the lower bound (Rosenbaum's
one-sided construction) would visit the same n p-values: every statistic here
satisfies |T(1 - Z)| = |T(Z)| and Bernoulli trials with probabilities 1 - pi
are the law of 1 - Z under pi, so the configuration giving the high
probability to the u smallest outcomes has the p-value of the one giving it
to the n - u largest (u = n: all units high, equivalent to all low). For these
two-sided tests the n configurations need not contain the extremes over all
high/low assignments; exhaustive search on small windows usually finds p-values
below the reported lower bound and occasionally above the upper one (see the
guide).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table

from . import common as c
from . import kernels as k
from . import randinf as ri
from . import windows as win


def _list(value: Any, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not len(value):
        raise AnalysisError("invalid_option", f"{name} must be a nonempty list of numbers.")
    if len(value) > 1000:
        raise AnalysisError("invalid_option", f"{name} may hold at most 1,000 values.")
    return [c.number(item, name) for item in value]


def rdrbounds(*, data: Any, y: str, running: str, cutoff: float = 0.0,
              wlist: Sequence[float] | None = None, gamma: Sequence[float] | None = None,
              expgamma: Sequence[float] | None = None, bound: str = "both",
              statistic: str = "ranksum", kernel: str = "uniform", fuzzy: str | None = None,
              nulltau: float = 0.0, prob: str | None = None, fmpval: bool = False,
              reps: int = 1000, seed: int = 666, method: str = "auto",
              missing: str = "raise") -> TableSet:
    """Rosenbaum bounds for RD designs under local randomization (community ``rdrbounds``).

    For each symmetric window |running - cutoff| <= w and each hidden-bias
    level Gamma = exp(gamma), reports the largest and smallest Bernoulli
    randomization p-value of the sharp null over the package's configurations:
    the u units with the largest outcomes are treated with probability
    Gamma/(1+Gamma), the others with 1/(1+Gamma), u = 1..n (Rosenbaum 2002;
    Cattaneo, Titiunik and Vazquez-Bare 2016). Other assignments of the two
    probabilities can give smaller two-sided p-values than ``lower_bound`` (and,
    rarely, larger ones than ``upper_bound``).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y, running, cutoff : outcome, running variable and cutoff (treated: running >= cutoff).
    wlist : window half-lengths w around the cutoff (default: the first five
        rdwinselect windows, rounded to two decimals).
    gamma, expgamma : sensitivity parameters as log-odds or odds (default
        expgamma = [1.5, 2, 2.5, 3]); give one of them.
    bound : 'both' (default), 'upper' or 'lower'.
    statistic : 'ranksum' (default), 'diffmeans' (alias 'ttest') or 'ksmirnov'.
    kernel : 'uniform', 'triangular' or 'epanechnikov' (diffmeans).
    fuzzy : treatment-received column (Anderson-Rubin statistic).
    nulltau : effect under the sharp null.
    prob : column of unit treatment probabilities for the baseline p-value
        (default: the window's treated share).
    fmpval : also report the fixed-margins p-value of each window.
    reps, seed, method : draws, generator seed and 'auto', 'exact' or 'monte_carlo'.
    missing : 'raise' (default) or 'drop' incomplete rows.

    Returns
    -------
    TableSet
        ``pvalues`` (baseline Bernoulli and, with ``fmpval``, fixed-margins
        p-value per window) and ``bounds`` (gamma, exp(gamma), window, lower
        and upper p-value bounds).

    Stata/R: ``rdrbounds Y R, expgamma(1.5 2 3) wlist(.5 .75 1) reps(1000)``.

    Example::

        rb = oe.rdrbounds(data=df, y="vote", running="margin", wlist=[0.5, 1.0],
                          expgamma=[1.5, 2.0], reps=500)
        rb["bounds"]
    """
    y = c.column(y, "y")
    running = c.column(running, "running")
    fuzzy = c.column(fuzzy, "fuzzy", optional=True)
    prob = c.column(prob, "prob", optional=True)
    statistic = c.statistic_name(statistic, ("diffmeans", "ksmirnov", "ranksum"))
    bound = c.choice(bound, "bound", ("both", "upper", "lower"))
    kernel = c.kernel_name(kernel)
    cutoff = c.number(cutoff, "cutoff")
    nulltau = c.number(nulltau, "nulltau")
    fmpval = c.flag(fmpval, "fmpval")
    reps = c.count(reps, "reps", maximum=k.MAX_REPS)
    seed = c.count(seed, "seed", minimum=0, maximum=2 ** 63 - 1)
    method = c.method_name(method)
    if gamma is not None and expgamma is not None:
        raise AnalysisError("invalid_option", "Give gamma or expgamma, not both.")
    if gamma is not None:
        odds = [math.exp(value) for value in _list(gamma, "gamma")]
    else:
        odds = [1.5, 2.0, 2.5, 3.0] if expgamma is None else _list(expgamma, "expgamma")
    if any(not value >= 1 for value in odds):
        raise AnalysisError("invalid_option", "Sensitivity odds exp(gamma) must be at least 1.")
    if fuzzy is not None and statistic != "diffmeans":
        raise AnalysisError("invalid_option", "A fuzzy design uses the Anderson-Rubin statistic; leave "
                            "statistic at 'diffmeans'.")
    if kernel != "uniform" and statistic != "diffmeans":
        raise AnalysisError("unsupported_option", "Kernel weights need statistic='diffmeans'.")
    required = [y, running] + [name for name in (fuzzy, prob) if name is not None]
    frame, dropped = c.sample(data, required, missing=missing)
    r = c.numeric(frame, running)
    yv = c.numeric(frame, y)
    t = None if fuzzy is None else c.numeric(frame, fuzzy)
    probabilities = None
    if prob is not None:
        probabilities = c.numeric(frame, prob)
        if bool(((probabilities < 0) | (probabilities > 1)).any()):
            raise AnalysisError("invalid_probabilities", "prob must lie in [0, 1].")
    c.check_cutoff(r, cutoff, strict=True)
    if wlist is None:
        rc = torch.sort(r - cutoff).values
        default = win.sequence(rc, int((rc < 0).sum()), obsmin=None, wmin=None, wobs=5, wstep=None,
                               wasymmetric=False, wmasspoints=False, nwindows=5)
        widths = [round(window.right, 2) for window in default]
    else:
        widths = _list(wlist, "wlist")
    if any(not width > 0 for width in widths):
        raise AnalysisError("invalid_window", "Window half-lengths in wlist must be positive.")
    samples = [ri.window_sample(r, yv, t, probabilities, cutoff=cutoff, wl=cutoff - w, wr=cutoff + w,
                                kernel=kernel, minimum=1, centred=True) for w in widths]
    # n configurations per gamma plus the baseline (and fixed-margins) run, each about
    # min(reps, 2^n) assignments of n units.
    work = sum((len(odds) * s.y.numel() + 1 + fmpval) * min(reps, 2 ** min(s.y.numel(), 63))
               * s.y.numel() for s in samples)
    if work > k.MAX_WORK:
        raise AnalysisError("work_budget_exceeded", f"The bounds need about {work:,} unit evaluations, "
                            f"above the budget of {k.MAX_WORK:,}; reduce reps, gammas or windows.")
    p_rows, bound_rows = [], []
    for w, sample in zip(widths, samples, strict=True):
        outcomes = ri.null_outcomes(sample, [nulltau])
        base_prob = sample.prob if sample.prob is not None else \
            torch.full_like(sample.y, sample.n_right / sample.y.numel())
        test = ri.make_statistic(statistic, outcomes, ri.WindowSample(
            sample.r, sample.rc, sample.y, sample.d, sample.t, base_prob, sample.weights))
        base = k.randomize([test], k.Law(sample.d, base_prob), reps=reps, seed=seed, method=method)
        row = {"w": w, "n_left": sample.n_left, "n_right": sample.n_right,
               "p_value": float(base.p_values[0][0]), "p_value_fixed_margins": None}
        if fmpval:
            fixed = ri.make_statistic(statistic, outcomes, ri.WindowSample(
                sample.r, sample.rc, sample.y, sample.d, sample.t, None, sample.weights))
            row["p_value_fixed_margins"] = float(
                k.randomize([fixed], k.Law(sample.d), reps=reps, seed=seed, method=method).p_values[0][0])
        p_rows.append(row)
        order = torch.argsort(sample.y, descending=True, stable=True)
        for odds_value in odds:
            low, high = 1 / (1 + odds_value), odds_value / (1 + odds_value)
            values = []
            for u in range(1, sample.y.numel() + 1):
                probs = torch.full_like(sample.y, low)
                probs[order[:u]] = high
                outcome = k.randomize([test], k.Law(sample.d, probs), reps=reps, seed=seed,
                                      method=method)
                values.append(float(outcome.p_values[0][0]))
            bound_rows.append({"gamma": math.log(odds_value), "expgamma": odds_value, "w": w,
                               "lower_bound": None if bound == "upper" else min(values),
                               "upper_bound": None if bound == "lower" else max(values)})
    return TableSet(
        {"pvalues": table(p_rows, columns=["w", "n_left", "n_right", "p_value", "p_value_fixed_margins"]),
         "bounds": table(bound_rows, columns=["gamma", "expgamma", "w", "lower_bound", "upper_bound"])},
        title="Rosenbaum bounds for RD under local randomization",
        cutoff=cutoff, outcome=y, running=running, fuzzy=fuzzy, statistic=statistic, bound=bound,
        kernel=kernel, nulltau=nulltau, wlist=widths, expgamma=odds, n_dropped=dropped,
        reps=reps, seed=seed, method=method,
        bound_search="for u = 1..n the u units with the largest outcomes receive the high "
                     "probability; the bounds are the maximum and minimum p-value over u",
        randomization="Bernoulli trials", stata_parity_validated=False,
        notes=["Monte Carlo configurations share one stream of uniform draws (common random numbers)."],
        references=[*c.REFERENCES, "Rosenbaum (2002), Observational Studies, 2nd ed., Springer."],
    )
