"""Randomization inference in a window around the cutoff (community ``rdrandinf``).

Local randomization: inside the window W = [wl, wr] the assignment D = 1(R >=
c) behaves as if randomized, by fixed margins (every set of n1 treated units
equally likely) or Bernoulli trials with probabilities pi_i. The sharp null of
a constant effect tau0 imputes the untreated outcome Y - tau0 D (sharp design)
or Y - tau0 T for the treatment received T (fuzzy design: the Anderson-Rubin /
intention-to-treat statistic is the difference in means of Y - tau0 T by D).
Randomization p-values compare |T(Y - tau0 D)| over the assignment law with
its observed value (kernels.py). Large-sample companions follow the package:

* diffmeans: z = (difference in means)/sqrt(s1^2/n1 + s0^2/n0) (Neyman); with
  kernel weights the weighted-least-squares coefficient of D with an HC2
  standard error;
* ranksum: the standardized rank sum itself; ksmirnov: R's ks.test p-value
  (exact tie-aware Smirnov law when n0 n1 < 10,000, the Kolmogorov limit
  otherwise);
* TSLS (fuzzy='tsls'): the IV coefficient of T instrumented by D with an HC1
  (p = 0) or ``vce`` (p > 0) sandwich, z test of tau0; no randomization test.

Polynomial adjustment (p > 0), as in rdlocrand 3.0: the coefficient of D in the
weighted regression of Y - tau0 D (or Y - tau0 T) on [1, D, P, D P], P the
powers 1..p of the running variable re-centred at ``evall``/``evalr``, with an
HC1/HC2/HC3 sandwich and normal inference; no randomization p-value. The
earlier (2016) randomization test on polynomial-adjusted outcomes was replaced
by this large-sample inference in rdlocrand 3.0 and is not offered.

Confidence sets for a constant effect invert the test over a grid of tau0
values, reusing the same assignments for every grid point; the accepted grid
points (p >= alpha) are reported as maximal runs [first, last] (the package's
``find_CI``; a set can be disconnected). ``interfci`` reports Rosenbaum's
interval under arbitrary interference, T - (q_{1-a/2}, q_{a/2}) with q the
quantiles of the randomization distribution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.engines.distributions import normal_isf
from openecon.engines.linalg import least_squares

from . import common as c
from . import kernels as k
from . import windows as win

LABELS = {"diffmeans": "Diff. in means", "ksmirnov": "Kolmogorov-Smirnov",
          "ranksum": "Rank sum z-stat", "ar": "ITT (Anderson-Rubin)", "tsls": "TSLS"}


@dataclass
class WindowSample:
    """Units of one window in canonical order (centred running variable first)."""

    r: Tensor
    rc: Tensor
    y: Tensor
    d: Tensor
    t: Tensor | None
    prob: Tensor | None
    weights: Tensor | None

    @property
    def n_left(self) -> int:
        return int((~self.d).sum())

    @property
    def n_right(self) -> int:
        return int(self.d.sum())


def window_sample(r: Tensor, y: Tensor, t: Tensor | None, prob: Tensor | None, *, cutoff: float,
                  wl: float, wr: float, kernel: str, minimum: int, centred: bool = False) -> WindowSample:
    """Units with round(R, 8) in [round(wl, 8), round(wr, 8)] and their kernel weights.

    ``centred`` compares round(R - c, 8) with round(w - c, 8) instead, as the
    package does in ``rdsensitivity`` and ``rdrbounds`` (which call rdrandinf on
    the centred running variable).
    """
    if centred:
        rounded = c.round8(r - cutoff)
        inside = (rounded >= c.round8_scalar(wl - cutoff)) & (rounded <= c.round8_scalar(wr - cutoff))
    else:
        rounded = c.round8(r)
        inside = (rounded >= c.round8_scalar(wl)) & (rounded <= c.round8_scalar(wr))
    keys = [r[inside] - cutoff, y[inside]] + [v[inside] for v in (t, prob) if v is not None]
    order = c.lexsort(keys)
    rw = r[inside][order]
    rc = rw - cutoff
    d = rc >= 0
    sample = WindowSample(rw, rc, y[inside][order], d, None if t is None else t[inside][order],
                          None if prob is None else prob[inside][order], None)
    if sample.n_left < minimum or sample.n_right < minimum:
        raise AnalysisError("insufficient_observations", f"The window [{wl:g}, {wr:g}] holds "
                            f"{sample.n_left} unit(s) below and {sample.n_right} at or above the "
                            f"cutoff; at least {minimum} per side are needed. Widen the window.")
    sample.weights = c.side_weights(rc, d, wl - cutoff, wr - cutoff, kernel)
    return sample


def null_outcomes(sample: WindowSample, taus: Sequence[float]) -> Tensor:
    """[n, G] imputed outcomes Y - tau D (or Y - tau T in a fuzzy design)."""
    treatment = sample.d.to(c.FLOAT) if sample.t is None else sample.t
    return sample.y[:, None] - torch.tensor(list(taus), dtype=c.FLOAT)[None, :] * treatment[:, None]


def make_statistic(name: str, outcomes: Tensor, sample: WindowSample):
    fixed = sample.prob is None
    if name == "diffmeans":
        return k.DiffMeans(outcomes, sample.d, sample.weights)
    if name == "ranksum":
        return k.RankSum(outcomes, sample.d, fixed)
    return k.KolmogorovSmirnov(outcomes, sample.d, fixed)


def terms(sample: WindowSample, p: int, evall: float, evalr: float) -> Tensor:
    """Powers 1..p of the running variable re-centred at evalr (treated) and evall (control)."""
    centre = torch.where(sample.d, torch.full_like(sample.r, evalr), torch.full_like(sample.r, evall))
    return c.powers(sample.r - centre, p)


def grid_pvalues(sample: WindowSample, taus: Sequence[float], name: str, *, p: int, evall: float,
                 evalr: float, reps: int, seed: int, method: str, vce: str
                 ) -> tuple[list[float], str, int]:
    """p-values of H0: tau = tau0 for every grid value (shared assignments), method, draws."""
    outcomes = null_outcomes(sample, taus)
    if p > 0:
        fit = k.hc_fit(outcomes, sample.d, terms(sample, p, evall, evalr), sample.weights, vce)
        return ([k.z_test(float(e), float(s))[1] for e, s in zip(fit.estimate, fit.se, strict=True)],
                "large-sample", 0)
    test = make_statistic(name, outcomes, sample)
    outcome = k.randomize([test], k.Law(sample.d, sample.prob), reps=reps, seed=seed, method=method)
    return outcome.p_values[0].tolist(), outcome.kind, outcome.draws


def default_grid(r: Tensor, y: Tensor, t: Tensor | None, cutoff: float, half: float) -> list[float]:
    """The package's default tau grid: the classical OLS (or IV) 95% interval in |Rc| <= half,
    rounded to two decimals, in ten steps of round((ub - lb)/10, 2)."""
    rc = r - cutoff
    inside = rc.abs() <= half
    yw, dw = y[inside], (rc[inside] >= 0).to(c.FLOAT)
    if int(dw.sum()) < 1 or int((1 - dw).sum()) < 1 or yw.numel() < 3:
        raise AnalysisError("insufficient_observations", "The default grid needs units on both sides "
                            "of the cutoff in the first window; supply the grid explicitly.")
    design = torch.stack([torch.ones_like(dw), dw], 1)
    try:
        if t is None:
            fit = least_squares(design, yw, drop_collinear=False)
            beta, bread, resid = fit.beta[1], fit.xtx_inv, fit.resid
        else:
            first = least_squares(design, t[inside], drop_collinear=False)
            projected = torch.stack([torch.ones_like(dw), first.fitted], 1)
            fit = least_squares(projected, yw, drop_collinear=False)
            beta, bread = fit.beta[1], fit.xtx_inv
            resid = yw - torch.stack([torch.ones_like(dw), t[inside]], 1) @ fit.beta
    except Exception as exc:
        raise AnalysisError("insufficient_observations", "The default grid's first-window regression "
                            "is not identified; supply the grid explicitly.") from exc
    se = math.sqrt(float(resid.square().sum()) / (yw.numel() - 2) * float(bread[1, 1]))
    lower, upper = float(beta) - 1.96 * se, float(beta) + 1.96 * se
    lb, ub, step = round(lower, 2), round(upper, 2), round((round(upper, 2) - round(lower, 2)) / 10, 2)
    if not step > 0:
        lb, ub, step = lower, upper, (upper - lower) / 10
    if not step > 0:
        raise AnalysisError("invalid_option", "The default grid is degenerate; supply the grid explicitly.")
    return win._r_seq(lb, ub, step)


def confidence_rows(p_values: Sequence[float], taus: Sequence[float], alpha: float) -> list[dict[str, Any]]:
    rows = c.interval_rows([value >= alpha for value in p_values], list(taus))
    return [{"ci_low": low, "ci_high": high} for low, high in rows] or [{"ci_low": None, "ci_high": None}]


def _bernoulli(value: Any) -> tuple[str, Any]:
    """('fixed' | 'estimated' | 'common' | 'column', payload) for the ``bernoulli`` option."""
    if value is None or value is False:
        return "fixed", None
    if value is True:
        return "estimated", None
    if isinstance(value, str):
        return "column", c.column(value, "bernoulli")
    return "common", c.level(value, "bernoulli")


def _probabilities(kind: str, payload: Any, frame, n: int) -> Tensor | None:
    if kind in ("fixed", "estimated"):
        return None
    if kind == "common":
        return torch.full((n,), payload, dtype=c.FLOAT)
    values = c.numeric(frame, payload)
    if bool(((values < 0) | (values > 1)).any()):
        raise AnalysisError("invalid_probabilities", f"Bernoulli probabilities in '{payload}' must lie "
                            "in [0, 1].")
    return values


def rdrandinf(*, data: Any, y: str, running: str, cutoff: float = 0.0, wl: float | None = None,
              wr: float | None = None, statistic: str = "diffmeans", p: int = 0,
              evall: float | None = None, evalr: float | None = None, kernel: str = "uniform",
              fuzzy: str | None = None, fuzzy_statistic: str = "ar", nulltau: float = 0.0,
              d: float | None = None, dscale: float | None = None, ci: float | None = None,
              ci_grid: Sequence[float] | None = None, interfci: float | None = None,
              bernoulli: bool | float | str | None = None, reps: int = 1000, seed: int = 666,
              method: str = "auto", vce: str = "HC3", covariates: Sequence[str] | None = None,
              obsmin: int | None = None, wmin: float | Sequence[float] | None = None,
              wobs: int | None = None, wstep: float | None = None, wasymmetric: bool = False,
              wmasspoints: bool = False, nwindows: int = 10, dropmissing: bool = False,
              rdwstat: str = "diffmeans", approx: bool = False, rdwreps: int = 1000,
              level: float = 0.15, missing: str = "raise") -> TableSet:
    """Randomization inference for RD designs under local randomization (community ``rdrandinf``).

    Inside the window [wl, wr] treatment D = 1(running >= cutoff) is treated as
    randomly assigned. The sharp null of a constant effect ``nulltau`` is
    tested with the randomization law of the window's assignments
    (Cattaneo, Frandsen and Titiunik 2015; Cattaneo, Titiunik and Vazquez-Bare
    2016, 2017). The statistic is computed on Y - nulltau D (or Y - nulltau T
    in a fuzzy design) and the p-value is the share of assignments with an
    absolute statistic at least as large as the observed one.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y, running : outcome and running variable; units with ``running >= cutoff`` are treated.
    cutoff : the RD cutoff.
    wl, wr : window limits in the units of the running variable. Without them
        the window is selected by ``rdwinselect`` from ``covariates`` (options
        ``obsmin``, ``wmin``, ``wobs``, ``wstep``, ``wasymmetric``,
        ``wmasspoints``, ``nwindows``, ``dropmissing``, ``rdwstat``,
        ``approx``, ``rdwreps``, ``level``, with this call's ``p``, ``vce``,
        ``seed`` and ``method``) or, without covariates, spans the running
        variable's range.
    statistic : 'diffmeans' (alias 'ttest'), 'ksmirnov', 'ranksum' or 'all'.
    p, evall, evalr, vce : polynomial order of the outcome adjustment
        (p > 0: large-sample HC inference on the interacted polynomial jump
        between ``evalr`` and ``evall``, default the cutoff; ``vce`` 'HC1',
        'HC2' or 'HC3', default 'HC3').
    kernel : 'uniform' (default), 'triangular' or 'epanechnikov' weights
        (diffmeans and fuzzy designs), with half-widths wr - cutoff and
        cutoff - wl.
    fuzzy, fuzzy_statistic : treatment-received column of a fuzzy design and
        its statistic, 'ar' (alias 'itt'; Anderson-Rubin randomization test) or
        'tsls' (large-sample IV inference).
    nulltau : effect under the sharp null (default 0).
    d, dscale : effect size of the reported asymptotic power, or its multiple
        of the control-side outcome standard deviation (default 0.5 sd).
    ci, ci_grid : alpha of a confidence set for a constant effect, and the grid
        of effects inverted (default: the package's ten-step grid).
    interfci : alpha of Rosenbaum's interval under arbitrary interference.
    bernoulli : None (fixed margins), True (Bernoulli trials with the window's
        treated share), a common probability, or a column of unit probabilities.
    reps, seed, method : randomization draws, seed of the CPU generator, and
        'auto' (enumerate every assignment when there are at most ``reps``),
        'exact' or 'monte_carlo'.
    missing : 'raise' (default) or 'drop' incomplete rows.

    Returns
    -------
    TableSet
        ``tests`` (statistic, standard error where defined, randomization
        p-value, large-sample p-value, power against ``d``), ``sample`` (units,
        units in the window, outcome mean and standard deviation, window limit
        per side), ``ci`` and ``ci_grid`` when a confidence set is requested,
        and ``windows`` when the window was selected. ``attrs`` hold the
        window, design, randomization method and draws, interference interval
        and notes.

    Stata/R: ``rdrandinf Y R, cutoff(0) wl(-1) wr(1) statistic(diffmeans) reps(1000) seed(666)``.

    Example::

        res = oe.rdrandinf(data=df, y="vote", running="margin", wl=-2.5, wr=2.5,
                           statistic="all", reps=5000, seed=1, ci=0.05,
                           ci_grid=[x / 4 for x in range(0, 61)])
        res["tests"], res["ci"]
    """
    y = c.column(y, "y")
    running = c.column(running, "running")
    fuzzy = c.column(fuzzy, "fuzzy", optional=True)
    statistic = c.statistic_name(statistic, ("diffmeans", "ksmirnov", "ranksum", "all"))
    fuzzy_statistic = c.choice(fuzzy_statistic, "fuzzy_statistic", ("ar", "itt", "tsls"))
    fuzzy_statistic = "ar" if fuzzy_statistic == "itt" else fuzzy_statistic
    p = c.order_p(p)
    kernel = c.kernel_name(kernel)
    vce = c.vce_name(vce)
    cutoff = c.number(cutoff, "cutoff")
    nulltau = c.number(nulltau, "nulltau")
    wl, wr = c.number(wl, "wl", optional=True), c.number(wr, "wr", optional=True)
    evall, evalr = c.number(evall, "evall", optional=True), c.number(evalr, "evalr", optional=True)
    effect = c.number(d, "d", optional=True)
    dscale = c.positive(dscale, "dscale", optional=True)
    alpha = None if ci is None else c.level(ci, "ci")
    interference = None if interfci is None else c.level(interfci, "interfci")
    reps = c.count(reps, "reps", maximum=k.MAX_REPS)
    seed = c.count(seed, "seed", minimum=0, maximum=2 ** 63 - 1)
    method = c.method_name(method)
    names = c.columns(covariates, "covariates")
    taus = None if ci_grid is None else c.grid(ci_grid, "ci_grid")
    mechanism, payload = _bernoulli(bernoulli)
    dropmissing = c.flag(dropmissing, "dropmissing")
    tsls = fuzzy is not None and fuzzy_statistic == "tsls"
    if tsls and taus is not None:
        raise AnalysisError("invalid_option", "The TSLS confidence interval is the normal interval; "
                            "ci_grid applies to inverted tests only.")
    if fuzzy is not None and statistic != "diffmeans":
        raise AnalysisError("invalid_option", "A fuzzy design uses fuzzy_statistic ('ar' or 'tsls'); "
                            "leave statistic at 'diffmeans'.")
    if (wl is None) != (wr is None):
        raise AnalysisError("invalid_option", "Give both wl and wr, or neither.")
    if (evall is None) != (evalr is None):
        raise AnalysisError("invalid_option", "Give both evall and evalr, or neither.")
    if effect is not None and dscale is not None:
        raise AnalysisError("invalid_option", "Give d or dscale, not both.")
    if taus is not None and alpha is None:
        raise AnalysisError("invalid_option", "ci_grid needs ci (the alpha of the confidence set).")
    if kernel != "uniform" and statistic != "diffmeans":
        raise AnalysisError("unsupported_option", "Kernel weights are only allowed with "
                            "statistic='diffmeans' or a fuzzy design.")
    if kernel != "uniform" and evall is not None and (evall != cutoff or evalr != cutoff):
        raise AnalysisError("unsupported_option", "Kernel weights are only allowed with "
                            "evall = evalr = cutoff.")
    if p > 0 and statistic != "diffmeans":
        raise AnalysisError("unsupported_option", "Polynomial adjustment (p > 0) is available only for "
                            "the difference in means and fuzzy designs (rdlocrand 3.0 falls back to "
                            "p = 0 for ksmirnov, ranksum and all); use p=0.")
    if interference is not None and (p > 0 or fuzzy is not None or statistic == "all"
                                     or mechanism != "fixed"):
        raise AnalysisError("unsupported_option", "interfci needs p=0, a sharp design, a single "
                            "statistic and fixed-margins randomization.")
    if mechanism != "fixed" and (p > 0 or tsls):
        raise AnalysisError("unsupported_option", "Bernoulli assignment applies to randomization "
                            "inference (p=0, no TSLS); it has no role in large-sample inference.")
    if alpha is not None and statistic == "all":
        raise AnalysisError("unsupported_option", "A confidence set inverts one statistic; choose "
                            "statistic='diffmeans', 'ksmirnov' or 'ranksum'.")
    if wl is not None and names:
        raise AnalysisError("invalid_option", "covariates select the window; omit them when wl and wr "
                            "are given.")
    required = [y, running] + [name for name in (fuzzy,) if name is not None]
    if mechanism == "column":
        required.append(payload)
    frame, dropped = c.sample(data, required, names, missing=missing,
                              optional_missing=dropmissing or missing == "drop")
    r = c.numeric(frame, running)
    yv = c.numeric(frame, y)
    t = None if fuzzy is None else c.numeric(frame, fuzzy)
    prob = _probabilities(mechanism, payload, frame, len(frame))
    c.check_cutoff(r, cutoff, strict=False)
    notes: list[str] = []
    selection = None
    if wl is not None:
        if not wl < wr:
            raise AnalysisError("invalid_window", "wl must be smaller than wr.")
        if not wl <= cutoff <= wr:
            raise AnalysisError("invalid_window", "The window [wl, wr] must contain the cutoff.")
        how = "set by user"
    elif names:
        options = win.selection_options(
            obsmin=obsmin, wmin=wmin, wobs=wobs, wstep=wstep, wasymmetric=wasymmetric,
            wmasspoints=wmasspoints, nwindows=nwindows, statistic=rdwstat, p=p, evalat="cutoff",
            kernel="uniform", approx=approx, level=level, reps=rdwreps, seed=seed, method=method,
            vce=vce)
        keep = torch.ones(len(frame), dtype=torch.bool)
        if dropmissing:
            keep = torch.as_tensor(~frame[names].isna().any(axis=1).to_numpy())
        rc = r[keep] - cutoff
        x = torch.stack([c.numeric(frame, name, allow_missing=True) for name in names], 1)[keep]
        order = c.lexsort([rc, *list(x.T)])
        selection = win.select(rc[order], x[order], names, **options)
        if selection.recommended is None:
            raise AnalysisError("no_recommended_window", "rdwinselect found no window whose balance "
                                "tests pass at the given level; give wl and wr, lower level or change "
                                "the window options.")
        chosen = selection.windows[selection.recommended]
        wl, wr = cutoff + chosen.left, cutoff + chosen.right
        how = "rdwinselect"
        notes += selection.notes
    else:
        wl, wr = float(r.min()), float(r.max())
        how = "running variable range"
    if evall is None:
        evall = evalr = cutoff
    if evall < wl or evalr > wr:
        raise AnalysisError("invalid_option", "evall and evalr must lie inside the window.")
    sample = window_sample(r, yv, t, prob, cutoff=cutoff, wl=wl, wr=wr, kernel=kernel, minimum=2)
    if mechanism == "estimated":
        sample.prob = torch.full_like(sample.y, sample.n_right / sample.y.numel())
    n_left, n_right = sample.n_left, sample.n_right
    left_y, right_y = sample.y[~sample.d], sample.y[sample.d]
    sd0 = float(left_y.std(unbiased=True))
    delta = effect if effect is not None else (dscale if dscale is not None else 0.5) * sd0
    null = null_outcomes(sample, [nulltau])
    rows: list[dict[str, Any]] = []
    kind, draws, mc_se = "large-sample", 0, None
    interference_ci = None
    estimate = se = None
    if tsls or p > 0:
        poly = terms(sample, p, evall, evalr) if p > 0 else None
        if tsls:
            fit = k.hc_fit(sample.y, sample.d, poly, sample.weights, vce if p > 0 else "HC1", sample.t)
            estimate, se = float(fit.estimate[0]), float(fit.se[0])
            z, asy = k.z_test(estimate - nulltau, se)
            value = estimate
        else:
            fit = k.hc_fit(null, sample.d, poly, sample.weights, vce)
            value, se = float(fit.estimate[0]), float(fit.se[0])
            estimate = value + nulltau
            z, asy = k.z_test(value, se)
        name = "tsls" if tsls else ("ar" if fuzzy is not None else "diffmeans")
        rows.append({"test": name, "statistic": value, "std_error": se, "z": z, "p_value": None,
                     "asymptotic_p_value": asy, "power": k.power_mean(delta, se)})
    else:
        names_used = ["diffmeans", "ksmirnov", "ranksum"] if statistic == "all" else [statistic]
        tests = [make_statistic(name, null, sample) for name in names_used]
        law = k.Law(sample.d, sample.prob)
        outcome = k.randomize(tests, law, reps=reps, seed=seed, method=method,
                              keep=(0, 0) if interference is not None else None)
        kind, draws = outcome.kind, outcome.draws
        mc_se = None if kind == "exact" else [
            math.sqrt(float(pv[0]) * (1 - float(pv[0])) / float(ok[0]))
            for pv, ok in zip(outcome.p_values, outcome.valid, strict=True)]
        sigma = float(null[:, 0].std(unbiased=True))
        for name, test, p_value in zip(names_used, tests, outcome.p_values, strict=True):
            row = {"test": "ar" if fuzzy is not None else name, "statistic": float(test.observed[0]),
                   "std_error": None, "z": None, "p_value": float(p_value[0])}
            if name == "diffmeans":
                s = None
                if sample.weights is None:
                    diff, s = k.neyman(null, sample.d)
                else:
                    try:
                        fitted = k.hc_fit(null, sample.d, None, sample.weights, "HC2")
                        diff, s = fitted.estimate, fitted.se
                    except AnalysisError as exc:
                        if exc.code not in ("insufficient_observations", "unit_leverage"):
                            raise
                        notes.append("The HC2 standard error of the kernel-weighted difference in means "
                                     "is undefined (a side has fewer than two units with positive "
                                     "weight); only the randomization p-value is reported.")
                if s is None:
                    row["asymptotic_p_value"] = row["power"] = None
                else:
                    row["std_error"] = float(s[0])
                    row["z"], row["asymptotic_p_value"] = k.z_test(float(diff[0]), float(s[0]))
                    row["power"] = k.power_mean(delta, float(s[0]))
            elif name == "ranksum":
                z = float(test.observed[0])
                row["z"] = z
                row["asymptotic_p_value"] = k.two_sided(z) if math.isfinite(z) else 1.0
                row["power"] = k.power_rank(delta, sigma, n_left, n_right)
            else:
                row["asymptotic_p_value"] = k.ks_large_sample(test, 0)
                row["power"] = None
            rows.append(row)
        if interference is not None:
            quantiles = torch.quantile(outcome.distribution,
                                       torch.tensor([interference / 2, 1 - interference / 2], dtype=c.FLOAT))
            observed = rows[0]["statistic"]
            interference_ci = [observed - float(quantiles[1]), observed - float(quantiles[0])]
    tables = {
        "tests": table(rows, columns=["test", "statistic", "std_error", "z", "p_value",
                                      "asymptotic_p_value", "power"]),
        "sample": table([
            {"side": "left", "n": int((r < cutoff).sum()), "n_window": n_left,
             "mean": float(left_y.mean()), "std_dev": sd0, "window_limit": wl},
            {"side": "right", "n": int((r >= cutoff).sum()), "n_window": n_right,
             "mean": float(right_y.mean()), "std_dev": float(right_y.std(unbiased=True)),
             "window_limit": wr},
        ], columns=["side", "n", "n_window", "mean", "std_dev", "window_limit"]),
    }
    ci_method = None
    if alpha is not None:
        if tsls or (p > 0 and taus is None and fuzzy is None):
            critical = normal_isf(alpha / 2)
            tables["ci"] = table([{"ci_low": estimate - critical * se, "ci_high": estimate + critical * se}],
                                 columns=["ci_low", "ci_high"])
            ci_method = "normal (estimate +/- z se)"
        else:
            if p > 0 and taus is None:
                raise AnalysisError("invalid_option", "A fuzzy Anderson-Rubin confidence set with p > 0 "
                                    "needs ci_grid.")
            if taus is None:
                half = max(wr - cutoff, abs(wl - cutoff))
                taus = default_grid(r, yv, t, cutoff, half)
                notes.append("The confidence set uses the package's default grid (ten steps over the "
                             "rounded classical 95% interval); give ci_grid for a finer or wider grid.")
            p_values, grid_kind, _ = grid_pvalues(
                sample, taus, "diffmeans" if fuzzy is not None else statistic, p=p, evall=evall,
                evalr=evalr, reps=reps, seed=seed, method=method, vce=vce)
            tables["ci"] = table(confidence_rows(p_values, taus, alpha), columns=["ci_low", "ci_high"])
            tables["ci_grid"] = table([{"tau": tau, "p_value": value} for tau, value in zip(taus, p_values, strict=True)],
                                      columns=["tau", "p_value"])
            ci_method = ("inversion of large-sample tests over the grid" if p > 0 else
                         "inversion of randomization tests over the grid (same assignments for every value)")
            if all(value < alpha for value in p_values):
                notes.append("No grid value was accepted; consider a wider ci_grid.")
            elif p_values[0] >= alpha or p_values[-1] >= alpha:
                notes.append("The confidence set reaches the end of the grid; widen ci_grid to avoid "
                             "truncating it.")
            if len(tables["ci"]) > 1:
                notes.append("The confidence set is disconnected; each ci row is one accepted run of the grid.")
    if selection is not None:
        tables["windows"] = win.windows_table(selection, cutoff)
    randomization = None
    if not (tsls or p > 0):
        randomization = "fixed margins" if sample.prob is None else "Bernoulli"
    return TableSet(
        tables, title="Randomization inference for RD under local randomization",
        cutoff=cutoff, outcome=y, running=running, window_left=wl, window_right=wr,
        window_selection=how, n=int(r.numel()), n_window=n_left + n_right, n_left=n_left,
        n_right=n_right, n_dropped=dropped, statistic=statistic if fuzzy is None else fuzzy_statistic,
        design="fuzzy" if fuzzy is not None else "sharp", fuzzy=fuzzy, p=p, kernel=kernel,
        evall=evall if p > 0 else None, evalr=evalr if p > 0 else None,
        vce=(vce if p > 0 else "HC1") if (tsls or p > 0) else None, nulltau=nulltau,
        estimate=estimate, delta=delta,
        inference="large-sample" if (tsls or p > 0) else "randomization",
        randomization=randomization,
        bernoulli=None if mechanism == "fixed" else (payload if mechanism == "column" else
                                                     float(sample.prob[0])),
        method=kind if not (tsls or p > 0) else None, draws=draws or None, reps=reps, seed=seed,
        monte_carlo_standard_errors=mc_se, confidence_level=None if alpha is None else 1 - alpha,
        ci_method=ci_method, interference_level=None if interference is None else 1 - interference,
        interference_ci_low=None if interference_ci is None else interference_ci[0],
        interference_ci_high=None if interference_ci is None else interference_ci[1],
        p_value_definition="share of admissible assignments with |T*| >= |T| (no plus-one correction)",
        stata_parity_validated=False, notes=notes, references=c.REFERENCES,
    )
