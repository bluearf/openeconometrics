"""Stepwise, forward and backward selection of regressors: ``oe.stepwise``.

SPSS ``REGRESSION /METHOD=STEPWISE | FORWARD | BACKWARD`` and Stata
``stepwise, pe() pr(): regress``. The data are read once to form the centred
cross-product matrix; the whole search then runs on that small matrix with the
sweep operator (see ``selection/sweep.py``). One more O(n k) pass recomputes the
residual sum of squares of the final model from the data (``residual_sum``).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call
from openecon.econometrics.selection import common as c
from openecon.econometrics.selection import sweep as sw
from openecon.engines import distributions as dist

_MAX_COLUMNS = 4000


def design_blocks(frame: pd.DataFrame, numeric: list[str], keep: Tensor,
                  codes: list[tuple[Tensor, int]], weights: Tensor | None):
    """Yield ``(numeric block, z, w)`` over row blocks of the kept rows.

    ``z`` holds the numeric columns followed by the indicator columns of every
    categorical term: ``codes`` has, per term, full-length level codes and the
    number of levels L; levels 1..L-1 become indicators (the first level is the
    reference). ``w`` is the block of weights (None when unweighted).
    """
    width = len(numeric) + sum(levels - 1 for _, levels in codes)
    complete = bool(keep.all())
    for start, stop, block in c.blocks(frame, numeric, width):
        used = None if complete else keep[start:stop]
        if used is not None:
            if not bool(used.any()):
                continue
            block = block[used]
        pieces = [block]
        for values, levels in codes:
            level = values[start:stop] if used is None else values[start:stop][used]
            dummies = torch.zeros((len(level), levels), dtype=torch.float64)
            dummies[torch.arange(len(level)), level] = 1.0
            pieces.append(dummies[:, 1:])
        z = torch.cat(pieces, dim=1) if len(pieces) > 1 else block
        w = None
        if weights is not None:
            w = weights[start:stop] if used is None else weights[start:stop][used]
        yield block, z, w


def cross_products(frame: pd.DataFrame, numeric: list[str], keep: Tensor,
                   codes: list[tuple[Tensor, int]], weights: Tensor | None
                   ) -> tuple[Tensor, Tensor, float, int]:
    """Means and centred cross products of [numeric columns, dummy blocks] in one pass.

    Returns (means, cross products, sum of weights, rows); see ``design_blocks``
    for the column layout.
    """
    width = len(numeric) + sum(levels - 1 for _, levels in codes)
    shift = torch.zeros(width, dtype=torch.float64)
    shift[:len(numeric)] = c.rough_means(frame, numeric)
    moments = sw.CrossProducts(shift)
    peak = torch.zeros(len(numeric), dtype=torch.float64)
    for block, z, w in design_blocks(frame, numeric, keep, codes, weights):
        peak = torch.maximum(peak, c.block_peak(block, "A model column"))
        moments.add(z, w)
    c.check_scale(peak, numeric)
    means, cross = kernel_call(moments.finish)
    return means, cross, moments.weight, moments.rows


@torch.no_grad()
def residual_sum(frame: pd.DataFrame, numeric: list[str], keep: Tensor,
                 codes: list[tuple[Tensor, int]], weights: Tensor | None, columns: list[int],
                 outcome: int, slopes: Tensor, means: Tensor) -> float:
    """Weighted residual sum of squares of the final model, from the data (one O(n k) pass).

    ``columns`` and ``outcome`` index the ``design_blocks`` layout; the residual
    is ``e = (y - ybar) - (X - xbar) b``. Unlike ``1 - r' R^{-1} r`` this does not
    cancel when R-squared is close to one: an error d in b changes the sum only
    by d' X'X d, second order, so the sum is accurate to rounding whenever b is.
    """
    index = torch.tensor(columns, dtype=torch.int64)
    total = 0.0
    for _, z, w in design_blocks(frame, numeric, keep, codes, weights):
        e = z[:, outcome] - means[outcome]
        if columns:
            e = e - (z[:, index] - means[index]) @ slopes
        total += float(e.square().sum() if w is None else (w * e.square()).sum())
    return total


def _information(sse: float, n: float, parameters: int) -> tuple[float | None, float | None]:
    """Stata's estat ic after regress: AIC and BIC from the Gaussian log likelihood."""
    if not sse > 0:
        return None, None
    log_likelihood = -0.5 * n * (math.log(2.0 * math.pi) + math.log(sse / n) + 1.0)
    return -2.0 * log_likelihood + 2.0 * parameters, \
        -2.0 * log_likelihood + math.log(n) * parameters


def stepwise(data: Any, y: str, x: list[str], *, method: str = "stepwise",
             criterion: str = "pvalue", p_enter: float = 0.05, p_remove: float = 0.10,
             forced: list[str] | None = None, categorical: list[str] | None = None,
             weights: str | None = None, weight_type: str | None = None,
             tolerance: float = 1e-4, missing: str = "drop", alpha: float = 0.05) -> TableSet:
    """Stepwise, forward or backward selection of the regressors of a linear model.

    The model is ``y = b0 + sum_j b_j x_j + e`` fitted by (weighted) least
    squares with a constant. Candidate terms enter or leave one at a time
    according to the F test of their contribution or to an information
    criterion.

    Method
    ------
    The data are read once to form the mean-centred cross-product matrix of all
    candidate columns and the outcome; it is scaled to a correlation matrix and
    the search runs on it with the sweep operator (O(p^2) per step, no refit
    from the rows). For a term of q columns and a current model of k columns
    (n observations, residual sum of squares SSE):

    - F-to-enter  = ((SSE - SSE_new) / q) / (SSE_new / (n - 1 - k - q)),
    - F-to-remove = ((SSE_smaller - SSE) / q) / (SSE / (n - 1 - k)),

    both referred to the F distribution (for q = 1 they are the squared t
    statistic of the term). ``statistic`` in the ``steps`` table is this F; it
    equals the "F Change" of SPSS's Model Summary.

    - ``method="forward"``: start from the constant (and ``forced`` terms);
      enter the term with the smallest p-value while p < ``p_enter``.
    - ``method="backward"``: start from all terms; remove the term with the
      largest p-value while p > ``p_remove``.
    - ``method="stepwise"`` (default): forward selection in which, before each
      entry, the term with the largest p-value is removed if p > ``p_remove``
      (SPSS STEPWISE; Stata's forward stepwise). Requires p_enter <= p_remove.
    - ``criterion="aic"`` or ``"bic"``: at each step take the single entry
      (forward), removal (backward) or either (stepwise) that lowers
      ``n ln(SSE/n) + c k`` the most (c = 2 or ln n), until no move lowers it.
      ``p_enter`` and ``p_remove`` are then not used.

    A term may enter only if its tolerance (1 - R-squared on the regressors in
    the model) and the tolerance every regressor in the model would have
    afterwards are at least ``tolerance`` (SPSS's tolerance and minimum
    tolerance tests, default 0.0001).

    The final model's coefficients are solved afresh from the selected block
    of the correlation matrix, and its residual sum of squares is recomputed
    from the data in a second O(n k) pass, so its standard errors, ANOVA table
    and fit statistics stay accurate as R-squared approaches one. If a step
    leaves 1 - R-squared below 1e-13 (the rounding level of the swept matrix),
    the search stops there; that step's F test and the excluded terms' entry
    statistics are left missing with a note. An outcome that is an exact
    linear combination of the selected terms raises ``perfect_fit``.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : outcome column (numeric).
    x : candidate predictors.
    method : ``"stepwise"``, ``"forward"`` or ``"backward"``.
    criterion : ``"pvalue"``, ``"aic"`` or ``"bic"``.
    p_enter, p_remove : significance levels for entry (SPSS PIN, Stata pe())
        and removal (SPSS POUT, Stata pr()).
    forced : predictors that are always in the model and never tested for
        removal (SPSS ``/METHOD=ENTER`` block, Stata ``lockterm1``). They need
        not be listed in ``x``.
    categorical : predictors (from ``x`` or ``forced``) to expand into
        indicator columns, first level as reference. The indicators of one
        predictor enter and leave together and are tested jointly (Stata's
        parenthesized terms).
    weights, weight_type : weight column; ``"aweight"`` (default: analytic /
        WLS weights, SPSS REGWGT, n = number of rows) or ``"fweight"``
        (frequency weights, n = sum of weights). Zero weights are excluded.
    tolerance : minimum tolerance for entry.
    missing : ``"drop"`` (default) excludes rows with a missing value in the
        outcome, any candidate, forced term or weight (listwise, as SPSS and
        Stata do); ``"raise"`` rejects them.
    alpha : 1 - confidence level of the coefficient intervals.

    Returns
    -------
    TableSet with tables

    - ``steps``: one row per step (and a ``start`` row when the starting model
      has regressors): ``action`` (entered / removed), ``term``, ``statistic``
      (F-to-enter or F-to-remove = F change), ``df1``, ``df2``, ``p_value``,
      then the model after the step: ``r_squared``, ``adjusted_r_squared``,
      ``r_squared_change``, ``rmse``, ``aic``, ``bic``.
    - ``coefficients``: final model: ``b``, ``std_error``, ``beta``
      (standardized), ``t``, ``p_value``, ``ci_low``, ``ci_high``,
      ``tolerance``, ``vif``. Classical standard errors with n - k - 1 degrees
      of freedom (Student t).
    - ``excluded``: every term not in the final model: ``beta_in``, ``t``,
      ``statistic`` (F-to-enter), ``df1``, ``df2``, ``p_value``,
      ``partial_correlation``, ``tolerance``, ``min_tolerance`` (SPSS Excluded
      Variables). ``beta_in`` and ``t`` are defined for one-column terms; for a
      categorical term ``partial_correlation`` is the nonnegative root of the
      partial R-squared.
    - ``anova``: regression, residual and total sums of squares of the final
      model with its F test.

    ``attrs``: ``selected`` (terms of the final model in order of entry),
    ``n``, ``n_missing``, ``r_squared``, ``adjusted_r_squared``, ``rmse``,
    ``statistic``, ``df1``, ``df2``, ``p_value``, ``aic``, ``bic``, the
    settings and ``notes``.

    Equivalent commands: SPSS ``REGRESSION /CRITERIA=PIN(.05) POUT(.10)
    /DEPENDENT y /METHOD=STEPWISE x1 x2 x3``; Stata ``stepwise, pe(.05)
    pr(.10) forward: regress y x1 x2 x3``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"y": [1.0, 3.1, 4.9, 7.2, 8.8, 11.1, 13.0, 15.2],
    ...         "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ...         "x2": [1.0, -1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 1.0]}
    >>> result = oe.stepwise(data, "y", ["x1", "x2"])
    >>> result.attrs["selected"]
    ['x1']
    """
    c.check_name(y, "y")
    candidates = c.name_list(x, "x", minimum=1)
    locked = c.name_list(forced, "forced")
    factors = c.name_list(categorical, "categorical")
    c.check_choice(method, "method", ("stepwise", "forward", "backward"))
    c.check_choice(criterion, "criterion", ("pvalue", "aic", "bic"))
    p_enter = c.check_probability(p_enter, "p_enter")
    p_remove = c.check_probability(p_remove, "p_remove")
    tolerance = c.check_probability(tolerance, "tolerance", " (SPSS uses 0.0001)")
    alpha = c.check_probability(alpha, "alpha", " (0.05 gives 95% confidence intervals)")
    if criterion == "pvalue" and method == "stepwise" and p_enter > p_remove:
        raise AnalysisError("invalid_option", "method='stepwise' needs p_enter <= p_remove; "
                            "otherwise a term could enter and leave forever.")
    weights, weight_type = c.check_weights(weights, weight_type)
    terms = locked + [name for name in candidates if name not in set(locked)]
    unknown = [name for name in factors if name not in set(terms)]
    if unknown:
        raise AnalysisError("invalid_spec", "categorical lists columns that are neither in x "
                            f"nor in forced: {', '.join(unknown)}.")
    numeric_terms = [name for name in terms if name not in set(factors)]
    if len(numeric_terms) >= _MAX_COLUMNS:
        raise AnalysisError("design_too_large", f"Stepwise selection takes fewer than "
                            f"{_MAX_COLUMNS} candidate columns (the search keeps their "
                            "cross-product matrix in memory); reduce the candidate list.")
    used = [y, *terms, *([weights] if weights else [])]
    numbers = [y, *numeric_terms, *([weights] if weights else [])]
    frame = c.source(data, used, numeric=numbers)
    keep, dropped = c.listwise(frame, used, missing, numeric=numbers)
    notes: list[str] = []
    w_full = None
    if weights:
        before = int(keep.sum())
        w_full = c.weight_column(frame, weights, weight_type, keep)
        if int(keep.sum()) < before:
            notes.append(f"Excluded {before - int(keep.sum())} observation(s) with zero weight.")

    # Column layout: numeric terms, the outcome, then the dummy blocks.
    numeric = [*numeric_terms, y]
    blocks: dict[str, list[int]] = {name: [i] for i, name in enumerate(numeric_terms)}
    labels: dict[int, str] = dict(enumerate(numeric_terms))
    codes: list[tuple[Tensor, int]] = []
    width = len(numeric)
    for name in factors:
        values, levels = c.group_codes(frame[name], keep, name)
        if len(levels) < 2:
            blocks[name] = []
            continue
        blocks[name] = list(range(width, width + len(levels) - 1))
        labels.update({width + i: f"{name}[{level}]" for i, level in enumerate(levels[1:])})
        codes.append((values.clamp_min(0), len(levels)))
        width += len(levels) - 1
        if width > _MAX_COLUMNS:
            raise AnalysisError("design_too_large", f"The candidate design has more than "
                                f"{_MAX_COLUMNS} columns; reduce the categorical levels or "
                                "the number of candidates.")
    if len(set(labels.values())) != len(labels):
        raise AnalysisError("duplicate_terms", "Predictor names collide with generated "
                            "indicator names. Rename the affected columns.")
    means, cross, total_weight, rows = cross_products(frame, numeric, keep, codes, w_full)
    n = float(rows) if weight_type != "fweight" else total_weight
    cross = cross * (n / total_weight)                  # analytic weights are scaled to sum to n
    # Move the outcome to the last position of the cross-product matrix.
    position = len(numeric_terms)
    order = [*range(position), *range(position + 1, width), position]
    means, cross = means[order], cross[order][:, order]
    remap = {old: new for new, old in enumerate(order)}
    blocks = {name: [remap[i] for i in columns] for name, columns in blocks.items()}
    labels = {remap[i]: label for i, label in labels.items()}
    p = width - 1
    if n - 1.0 < 2.0:
        raise AnalysisError("insufficient_observations", "Stepwise selection needs at least "
                            "three observations.")
    corr, scale, valid = sw.correlation(cross)
    # A column whose variation is at the rounding level of its mean is constant.
    valid &= cross.diagonal() > 1e-24 * n * means.square()
    if not bool(valid[p]):
        raise AnalysisError("zero_variance", f"The outcome '{y}' does not vary in the "
                            "estimation sample.")
    block_list = [blocks[name] for name in terms]
    search, found = kernel_call(
        sw.stepwise_search, corr, block_list, valid[:p], list(range(len(locked))), n=n,
        method=method, criterion=criterion, p_enter=p_enter, p_remove=p_remove,
        tolerance=tolerance)
    if found.exhausted:
        notes.append("The search stopped because the selected terms explain the outcome to "
                     "within the rounding error of the cross-product matrix (1 - R-squared "
                     f"below {sw._PERFECT:g}); the statistic of the last step and the entry "
                     "statistics of the excluded terms are not computable in double precision "
                     "and are left missing. The final model below is recomputed from the data.")
    for term, reason in found.skipped.items():
        if reason == "tolerance":
            role = "forced term" if term < len(locked) else "term"
            notes.append(f"The {role} '{terms[term]}' was not entered: it fails the tolerance "
                         f"test ({tolerance:g}) against the terms entered before it.")
    constant = [terms[t] for t in range(len(terms)) if not bool(search.usable[t])]
    if constant:
        notes.append("Not eligible because they do not vary in the estimation sample: "
                     f"{', '.join(constant)}.")
    if found.stopped:
        notes.append(f"Selection did not settle: {found.stopped}.")
    total_ss = float(cross[p, p])

    def summary(share: float, k: int) -> list[float | None]:
        """[R^2, adjusted R^2, rmse, AIC, BIC] from the residual share 1 - R^2."""
        residual_df = n - 1.0 - k
        sse = total_ss * share
        adjusted = 1.0 - share * (n - 1.0) / residual_df
        aic, bic = _information(sse, n, k + 1)
        return [1.0 - share, adjusted, math.sqrt(sse / residual_df), aic, bic]

    # Final model: fresh Cholesky solve on the selected correlation block for the
    # coefficients; the residual sum of squares is recomputed from the data.
    selected = [terms[t] for t in found.order]
    columns = [column for t in found.order for column in block_list[t]]
    k = len(columns)
    beta, inverse, _ = kernel_call(sw.final_model, corr, columns)
    df_resid = n - 1.0 - k
    index = torch.tensor(columns, dtype=torch.int64)
    b = beta * scale[p] / scale[index]
    original = [order[i] for i in columns]               # positions in the data blocks
    unordered = torch.empty_like(means)
    unordered[torch.tensor(order, dtype=torch.int64)] = means
    sse = residual_sum(frame, numeric, keep, codes, w_full, original, position, b,
                       unordered) * (n / total_weight)
    if k and not sse > 1e-24 * total_ss:
        raise AnalysisError("perfect_fit", f"The outcome '{y}' is an exact linear combination "
                            f"of {', '.join(selected)} and the constant (1 - R-squared is "
                            "zero to rounding), so no standard error or test is defined. "
                            "Remove the term(s) that define the outcome.")
    residual = min(1.0, sse / total_ss)
    mse = sse / df_resid

    # The model after the last step is the final model: its row shows the recomputed fit.
    step_rows: list[list[Any]] = []
    previous = 0.0
    first = 0 if found.steps and found.steps[0].action == "start" else 1
    last = first + len(found.steps) - 1
    for number, move in enumerate(found.steps, start=first):
        if move.action == "start":
            name = "(all terms)" if method == "backward" \
                else ", ".join(terms[t] for t in found.start)
        else:
            name = terms[move.term]
        share = 1.0 - move.r_squared
        if number == last:
            share = residual                   # the final model, recomputed from the data
            move.r_squared = 1.0 - residual
        fit = summary(share, move.columns)
        step_rows.append([number, move.action, name, c.finite(move.statistic), move.df1,
                          move.df2, move.p_value, fit[0], fit[1], move.r_squared - previous,
                          *fit[2:]])
        previous = move.r_squared
    steps = c.result_table(step_rows, columns=[
        "step", "action", "term", "statistic", "df1", "df2", "p_value", "r_squared",
        "adjusted_r_squared", "r_squared_change", "rmse", "aic", "bic"])

    se = (inverse.diagonal() * residual / df_resid).sqrt() * scale[p] / scale[index]
    unit = means[index] / scale[index]
    intercept = float(means[p] - (b * means[index]).sum())
    intercept_se = math.sqrt(mse * (1.0 / n + float(unit @ inverse @ unit)))
    critical = dist.t_isf(0.5 * alpha, df_resid)
    coefficient_rows = []
    estimates = [intercept, *b.tolist()]
    errors = [intercept_se, *se.tolist()]
    betas = [None, *beta.tolist()]
    tolerances = [None, *(1.0 / inverse.diagonal()).tolist()]
    for estimate, error, standardized, tol in zip(estimates, errors, betas, tolerances,
                                                  strict=True):
        t = estimate / error if error > 0 else None
        coefficient_rows.append([
            estimate, error, standardized, t, c.t_two_sided(t, df_resid),
            estimate - critical * error, estimate + critical * error, tol,
            None if tol is None else 1.0 / tol])
    coefficients = c.result_table(
        coefficient_rows, index=["Intercept", *(labels[i] for i in columns)],
        columns=["b", "std_error", "beta", "t", "p_value", "ci_low", "ci_high", "tolerance",
                 "vif"])

    # Excluded terms, from a freshly swept matrix of the final model.
    search.refresh()
    gain, tol_out, low, signed = search.entry()
    excluded_rows, excluded_index = [], []
    for t, name in enumerate(terms):
        if bool(search.term_in[t]):
            continue
        q = len(block_list[t])
        df2 = df_resid - q
        row: list[Any] = [None] * 9
        if q and bool(search.usable[t]) and float(tol_out[t]) > 1e-12 and df2 > 0 \
                and not found.exhausted:
            g, left = float(gain[t]), residual - float(gain[t])
            statistic = (g / q) / (left / df2) if left > 0 else None
            partial = math.sqrt(min(1.0, g / residual)) if residual > 0 else None
            beta_in = t_value = None
            if q == 1:
                beta_in = float(signed[t]) / float(tol_out[t])
                sign = 1.0 if float(signed[t]) >= 0 else -1.0
                t_value = None if statistic is None else sign * math.sqrt(statistic)
                partial = None if partial is None else sign * partial
            row = [beta_in, t_value, statistic, q, df2, c.f_upper(statistic, q, df2), partial,
                   float(tol_out[t]), float(low[t])]
        elif q and found.exhausted and bool(search.usable[t]):
            row[3], row[4], row[7], row[8] = q, df2, float(tol_out[t]), float(low[t])
        elif q:
            row[3], row[7], row[8] = q, 0.0, 0.0
        excluded_rows.append(row)
        excluded_index.append(name)
    excluded = c.result_table(
        excluded_rows, index=excluded_index,
        columns=["beta_in", "t", "statistic", "df1", "df2", "p_value", "partial_correlation",
                 "tolerance", "min_tolerance"])

    r_squared = 1.0 - residual
    regression_ss = total_ss - sse
    f = (regression_ss / k) / mse if k and mse > 0 else None
    p_value = c.f_upper(f, k, df_resid) if k else None
    anova = c.result_table(
        [[regression_ss, k, regression_ss / k if k else None, f, p_value],
         [sse, df_resid, mse, None, None], [total_ss, n - 1.0, None, None, None]],
        index=["regression", "residual", "total"],
        columns=["ss", "df", "ms", "statistic", "p_value"])
    fit = summary(residual, k)
    if not selected:
        notes.append("No term met the criterion to enter or to stay; the final model has only "
                     "the constant.")
    return TableSet(
        {"steps": steps, "coefficients": coefficients, "excluded": excluded, "anova": anova},
        title=f"Stepwise regression of {y} ({method}, {criterion})",
        selected=selected, n=int(n) if n == int(n) else n, n_missing=dropped,
        r_squared=r_squared, adjusted_r_squared=fit[1], rmse=fit[2], statistic=f,
        df1=k, df2=df_resid, p_value=p_value, aic=fit[3], bic=fit[4], method=method,
        criterion=criterion, p_enter=p_enter, p_remove=p_remove, tolerance=tolerance,
        forced=locked, alpha=alpha, missing="listwise",
        weights=None if weights is None else {"column": weights, "type": weight_type},
        notes=notes)
