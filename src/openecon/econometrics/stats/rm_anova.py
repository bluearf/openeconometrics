"""Univariate repeated-measures (split-plot) analysis of variance: ``oe.rm_anova``.

Method (SPSS GLM Repeated Measures; Stata ``anova ..., repeated()``). The long
table is pivoted to the S-by-p matrix Y of subjects by within-subject cells. For
each within-subject effect W an orthonormal contrast matrix M_W (p-by-d, columns
orthogonal to the constant over the factors in W and constant over the others)
gives the transformed variables T_W = Y M_W. The between-subject design X (sum-to-
zero coded intercept, between factors and their interactions) is then fitted to
all transformed variables at once, and for every between-subject term g

    SS(W x g)   = trace H_g(T_W),   df = d * df_g
    SS(error W) = trace E(T_W),     df = d * (S - rank X),

where H and E are the Type III hypothesis and residual SSCP matrices. The
intercept row of the between design is the within effect W itself. With
M = 1/sqrt(p) (the scaled subject mean) the same fit gives the between-subject
tests. Unequal group sizes are allowed; every subject must have exactly one
observation in every within-subject cell.

Sphericity of an effect with d >= 2 contrasts is assessed on E = E(T_W) with
eigenvalues l_1..l_d and error degrees of freedom v = S - rank X:

    Mauchly  W = det(E) / (trace(E) / d)^d,
             chi2 = -(v - (2 d^2 + d + 2) / (6 d)) ln W,  df = d (d + 1) / 2 - 1
    Greenhouse-Geisser  eps = (sum l)^2 / (d sum l^2)
    Huynh-Feldt         eps = min(1, (S d eps_GG - 2) / (d (v - d eps_GG)))
    lower bound         eps = 1 / d

and the corrected tests multiply both degrees of freedom of F by eps.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.stats import common as c
from openecon.econometrics.stats.glm import (
    LinearModel, build_terms, effect_coding, kron, term_name,
)

_CORRECTIONS = ("sphericity_assumed", "greenhouse_geisser", "huynh_feldt", "lower_bound")
_MAX_CELLS = 500


def sphericity(error: Tensor, df_error: int, subjects: int) -> dict[str, Any]:
    """Mauchly's test and the epsilon corrections from a d-by-d error SSCP matrix."""
    d = error.shape[0]
    if d == 1:
        return {"mauchly_w": 1.0, "chi2": None, "df": 0, "p_value": None, "epsilon_gg": 1.0,
                "epsilon_hf": 1.0, "epsilon_lb": 1.0}
    roots = torch.linalg.eigvalsh(error).clamp_min(0.0)
    total = float(roots.sum())
    gg = total ** 2 / (d * float(roots.square().sum()))
    gg = min(1.0, max(gg, 1.0 / d))
    denominator = d * (df_error - d * gg)
    hf = min(1.0, (subjects * d * gg - 2.0) / denominator) if denominator > 0 else 1.0
    w = chi2 = p_value = None
    df = d * (d + 1) // 2 - 1
    smallest = float(roots.min())
    if df_error >= d and smallest > 1e-12 * total / d:
        log_w = float(roots.log().sum()) - d * math.log(total / d)
        w = math.exp(log_w)
        scale = df_error - (2.0 * d * d + d + 2.0) / (6.0 * d)
        if scale > 0:
            chi2 = max(-scale * log_w, 0.0)
            p_value = c.chi2_upper(chi2, df)
    return {"mauchly_w": w, "chi2": chi2, "df": df, "p_value": p_value, "epsilon_gg": gg,
            "epsilon_hf": hf, "epsilon_lb": 1.0 / d}


def _contrasts(sizes: list[int]) -> tuple[Tensor, dict[tuple[int, ...], list[int]]]:
    """Orthonormal p-by-p transformation and the columns of every within effect."""
    bases = []
    for size in sizes:
        basis = torch.linalg.qr(effect_coding(size)).Q
        bases.append((torch.full((size, 1), 1.0 / math.sqrt(size), dtype=torch.float64), basis))
    blocks, columns, position = [], {}, 0
    subsets = [()] + [combo for order in range(1, len(sizes) + 1)
                      for combo in itertools.combinations(range(len(sizes)), order)]
    for subset in subsets:
        block = torch.ones((1, 1), dtype=torch.float64)
        for index in range(len(sizes)):
            block = kron(block, bases[index][1 if index in subset else 0])
        blocks.append(block)
        columns[subset] = list(range(position, position + block.shape[1]))
        position += block.shape[1]
    return torch.cat(blocks, dim=1), columns


def rm_anova(data: Any, y: str, subject: str, within: list[str], *,
             between: list[str] | None = None, alpha: float = 0.05,
             missing: str = "drop") -> TableSet:
    """Repeated-measures ANOVA (univariate approach) with sphericity corrections.

    ``data`` is in long format: one row per subject and within-subject cell. Each
    within-subject effect (one factor, or the main effects and interactions of
    several) is tested against its own subject-by-effect error term; optional
    between-subject factors give a split-plot (mixed) design with the effects
    ``within``, ``within#between`` and the between-subject tests.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records (long format).
    y : numeric outcome.
    subject : column identifying the subject.
    within : one or more within-subject factors (their level combinations are the
        repeated measurements).
    between : optional between-subject factors (constant within a subject); all
        their interactions are included. Group sizes may differ.
    alpha : kept in ``attrs`` for reporting.
    missing : ``"drop"`` (default) excludes rows with a missing value in a column
        used (counted in ``attrs["n_missing"]``); the remaining data must still be
        complete. ``"raise"`` rejects missing values.

    Returns
    -------
    TableSet with

    * ``within``: one row per source and correction (``sphericity_assumed``,
      ``greenhouse_geisser``, ``huynh_feldt``, ``lower_bound``): source, correction, ss,
      df, ms, statistic (F), p_value, partial_eta_squared; the error term of effect
      ``w`` is the source ``error(w)``;
    * ``sphericity``: Mauchly's W, its chi-square approximation, df and p_value, and
      the Greenhouse-Geisser, Huynh-Feldt and lower-bound epsilons per within effect;
    * ``between``: Intercept, between-subject terms and error (ss, df, ms, statistic,
      p_value, partial_eta_squared) on the subject means;
    * ``descriptives``: n, mean and std_dev of every cell.

    The data must be complete: every subject observed exactly once in every
    within-subject cell. Otherwise ``AnalysisError("unbalanced_design")`` is raised;
    incomplete or unbalanced repeated measures call for a mixed model. An effect
    whose error term is zero (it is identical for every subject) has missing F,
    p-value and sphericity statistics and a note in ``attrs["notes"]``; an outcome
    that does not vary at all raises ``zero_variance``.

    Formulas: see the module docstring. The Huynh-Feldt epsilon is the one SPSS
    reports, (S d eps - 2) / (d (v - d eps)) with S subjects, without Lecoutre's
    correction for between-subject groups. Mauchly's test is not reported for
    effects with a single contrast (two levels), where sphericity always holds.

    Equivalent commands: SPSS ``GLM t1 t2 t3 BY g /WSFACTOR=time 3`` (wide format);
    Stata ``anova y g / id|g time g#time, repeated(time)``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    ...         "time": [1, 2, 3] * 4,
    ...         "y": [5.0, 6.1, 7.4, 4.2, 5.9, 6.0, 6.3, 6.8, 8.1, 5.1, 5.2, 7.7]}
    >>> result = oe.rm_anova(data, "y", "id", ["time"])
    >>> sorted(result)
    ['between', 'descriptives', 'sphericity', 'within']
    """
    c.check_name(y, "y")
    c.check_name(subject, "subject")
    within = c.name_list(within, "within")
    between = c.name_list(between, "between", minimum=0)
    alpha = c.check_alpha(alpha)
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from .streaming_rm import rm_anova as replay_rm
        return replay_rm(data, y, subject, within, between, alpha, missing)
    frame, dropped = c.select(data, [y, subject, *within, *between], numeric=[y],
                              missing=missing)
    values = c.values(frame, y)
    subject_codes, subject_labels = c.group_codes(frame, subject)
    subjects = len(subject_labels)
    codes, levels = {}, {}
    for name in [*within, *between]:
        codes[name], levels[name] = c.group_codes(frame, name)
        if len(levels[name]) < 2:
            raise AnalysisError("single_level", f"Factor '{name}' has only one level.")
    sizes = [len(levels[name]) for name in within]
    cells = math.prod(sizes)
    if cells > _MAX_CELLS:
        raise AnalysisError("design_too_large", f"The within-subject design has {cells} cells; "
                            f"at most {_MAX_CELLS} are supported.")
    cell = torch.zeros_like(subject_codes)
    for name, size in zip(within, sizes, strict=True):
        cell = cell * size + codes[name]
    slot = subject_codes * cells + cell
    counts = torch.bincount(slot, minlength=subjects * cells)
    if values.shape[0] != subjects * cells or bool((counts != 1).any()):
        empty, repeated = int((counts == 0).sum()), int((counts > 1).sum())
        raise AnalysisError(
            "unbalanced_design", "Repeated-measures ANOVA needs every subject observed exactly "
            f"once in each of the {cells} within-subject cells: {empty} subject-cell "
            f"combination(s) are missing and {repeated} are repeated"
            + (f" ({dropped} row(s) with missing values were excluded)" if dropped else "")
            + ". Complete the data, drop incomplete subjects, or use a mixed model.")
    centred, shift = c.centre(values)
    total_ss = float(centred.square().sum())
    if c.negligible(total_ss, float(values.square().sum()), 1e-30):
        raise AnalysisError("zero_variance", f"'{y}' does not vary, so no F statistic is "
                            "defined.")
    wide = torch.empty((subjects, cells), dtype=torch.float64)
    wide.view(-1)[slot] = centred
    group_codes = {}
    for name in between:
        low = torch.full((subjects,), len(levels[name]), dtype=torch.int64)
        low = low.scatter_reduce(0, subject_codes, codes[name], reduce="amin")
        high = torch.full((subjects,), -1, dtype=torch.int64)
        high = high.scatter_reduce(0, subject_codes, codes[name], reduce="amax")
        if bool((low != high).any()):
            raise AnalysisError("invalid_design", f"Between-subject factor '{name}' changes "
                                "within subjects; list it in within instead.")
        group_codes[name] = low
    transform, columns = _contrasts(sizes)
    terms = build_terms(between, [], "full")
    level = torch.zeros(cells, dtype=torch.float64)
    level[0] = shift * math.sqrt(cells)          # only the subject-mean column has a level
    model = LinearModel(terms, group_codes, {name: levels[name] for name in between}, {},
                        wide @ transform, level)
    group = torch.zeros_like(subject_codes)
    for name in between:
        group = group * len(levels[name]) + codes[name]
    groups = math.prod(len(levels[name]) for name in between)
    moments = c.group_moments(values, group * cells + cell, groups * cells)
    described = []
    names = [*between, *within]
    for combo, n, mean, var in zip(
            itertools.product(*(levels[name] for name in names)), moments.n.tolist(),
            moments.mean.tolist(), moments.var.tolist(), strict=True):
        if n > 0:
            described.append([*combo, int(n), mean, math.sqrt(var) if n > 1 else None])
    return _finish(model, y, within, between, cells, columns, subjects, total_ss, described,\
                   len(values), dropped, alpha)


def _finish(model, y, within, between, cells, columns, subjects, total_ss, described, n, dropped, alpha):
    terms = model.terms
    names = [*between, *within]
    df_error = model.df_error
    hypotheses = {term.name: model.hypothesis(term, 3) for term in terms}

    notes: list[str] = []

    def clean(ss: float) -> float:
        """A sum of squares, with values at the rounding level of the total set to zero."""
        return 0.0 if c.negligible(ss, total_ss) else ss

    def effect_rows(index: list[int], label: str, sse: float,
                    epsilons: list[float | None]) -> list[list[Any]]:
        d = len(index)
        rows = []
        for term in terms:
            ss = clean(float(hypotheses[term.name][index][:, index].trace()))
            df = d * model.df(term)
            f = c.ratio(ss / df, sse / (d * df_error))
            source = label if term.name == "Intercept" else f"{label}#{term.name}"
            eta = c.ratio(ss, ss + sse)
            for name, eps in zip(_CORRECTIONS, epsilons, strict=True):
                if eps is None:
                    rows.append([source, name, ss, None, None, None, None, eta])
                else:
                    rows.append([source, name, ss, df * eps, ss / (df * eps), f,
                                 c.f_upper(f, df * eps, d * df_error * eps), eta])
        for name, eps in zip(_CORRECTIONS, epsilons, strict=True):
            df = None if eps is None else d * df_error * eps
            rows.append([f"error({label})", name, sse, df, None if df is None else sse / df,
                         None, None, None])
        return rows

    within_rows, sphericity_rows, labels = [], [], []
    for subset, index in columns.items():
        if not subset:
            continue
        label = term_name([within[i] for i in subset])
        labels.append(label)
        block = model.error[index][:, index]
        sse = clean(float(block.trace()))
        if sse == 0.0:
            # No subject-by-effect variation: F and the sphericity statistics are undefined.
            d = len(index)
            notes.append(f"The error term of '{label}' is zero (the effect is identical for "
                         "every subject), so its F tests and sphericity statistics are "
                         "undefined.")
            sphericity_rows.append([None, None, d * (d + 1) // 2 - 1, None, None, None, 1.0 / d])
            within_rows += effect_rows(index, label, sse, [1.0, None, None, 1.0 / d])
            continue
        test = sphericity(block, df_error, subjects)
        sphericity_rows.append([test[key] for key in ("mauchly_w", "chi2", "df", "p_value",
                                                      "epsilon_gg", "epsilon_hf", "epsilon_lb")])
        within_rows += effect_rows(index, label, sse, [1.0, test["epsilon_gg"],
                                                       test["epsilon_hf"], test["epsilon_lb"]])
    sse = clean(float(model.error[0, 0]))
    if sse == 0.0:
        notes.append("The subject means do not vary within the between-subject groups, so the "
                     "between-subject F tests are undefined.")
    between_rows, between_labels = [], []
    for term in terms:
        ss = clean(float(hypotheses[term.name][0, 0])) if term.name != "Intercept" \
            else max(float(hypotheses[term.name][0, 0]), 0.0)
        df = model.df(term)
        f = c.ratio(ss / df, sse / df_error)
        between_rows.append([ss, df, ss / df, f, c.f_upper(f, df, df_error),
                             c.ratio(ss, ss + sse)])
        between_labels.append(term.name)
    between_rows.append([sse, df_error, sse / df_error, None, None, None])
    tables = {
        "within": c.frame(within_rows, columns=["source", "correction", "ss", "df", "ms",
                                                "statistic", "p_value", "partial_eta_squared"]),
        "sphericity": c.frame(sphericity_rows, columns=["mauchly_w", "chi2", "df", "p_value",
                                                        "epsilon_gg", "epsilon_hf", "epsilon_lb"],
                              index=labels),
        "between": c.frame(between_rows, columns=["ss", "df", "ms", "statistic", "p_value",
                                                  "partial_eta_squared"],
                           index=[*between_labels, "error"]),
        "descriptives": c.frame(described, columns=[*names, "n", "mean", "std_dev"]),
    }
    return TableSet(tables, title=f"Repeated-measures ANOVA of {y}", n_subjects=subjects,
                    n=n, n_missing=dropped, within=within, between=between,
                    within_cells=cells, df_error_between=df_error, alpha=alpha,
                    ss_type=3, notes=notes, missing="listwise")
