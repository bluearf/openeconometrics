"""Multivariate analysis of variance and covariance: ``oe.manova`` (SPSS GLM, Stata manova)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call
from openecon.econometrics.stats import common as c
from openecon.econometrics.stats.anova import ANOVA_COLUMNS, anova_rows, prepare
from openecon.engines import linalg


def multivariate_tests(hypothesis: Tensor, error: Tensor, q: int, v: int) -> list[list[Any]]:
    """Pillai, Wilks, Hotelling-Lawley and Roy statistics with their F approximations.

    With l_1 >= ... >= l_s the nonzero eigenvalues of E^{-1}H, p outcomes, q
    hypothesis and v error degrees of freedom, s = min(p, q), m = (|p - q| - 1) / 2
    and n = (v - p - 1) / 2:

        Pillai's trace     V = sum l/(1+l);   F = (2n+s+1)/(2m+s+1) * V/(s-V),
                           df = s(2m+s+1), s(2n+s+1)
        Wilks' lambda      L = prod 1/(1+l);  Rao's F = (1-L^(1/t))/L^(1/t) * df2/df1,
                           t = sqrt((p^2 q^2 - 4)/(p^2 + q^2 - 5)) (1 if undefined),
                           df1 = p q, df2 = (v - (p-q+1)/2) t - (p q - 2)/2
        Hotelling's trace  U = sum l;         F = 2(sn+1) U / (s^2 (2m+s+1)),
                           df = s(2m+s+1), 2(sn+1)
        Roy's largest root l_1;               F = l_1 (v - r + q)/r, r = max(p, q),
                           df = r, v - r + q  (an upper bound on F unless s = 1)

    Rao's F is exact when s <= 2; the Pillai and Hotelling F are exact when s = 1.
    """
    p = error.shape[0]
    # Scaling the outcomes to unit residual norm leaves every statistic unchanged and
    # makes the rank test of E unit free.
    scale = error.diagonal().clamp_min(1e-300).sqrt()
    error = error / torch.outer(scale, scale)
    hypothesis = hypothesis / torch.outer(scale, scale)
    spectrum = torch.linalg.eigvalsh(error)
    if not float(spectrum[0]) > 1e-11 * float(spectrum[-1]):
        raise AnalysisError("singular_error_matrix", "The error SSCP matrix is singular: the "
                            "outcomes are linearly dependent or there are fewer error degrees "
                            "of freedom than outcomes. Remove a redundant outcome.")
    chol = torch.linalg.cholesky(error)
    half = torch.linalg.solve_triangular(chol, hypothesis, upper=False)
    scaled = torch.linalg.solve_triangular(chol, half.T, upper=False)
    roots = torch.linalg.eigvalsh(linalg.symmetrize(scaled)).clamp_min(0.0).flip(0)
    s = min(p, q)
    roots = roots[:s]
    m, n = (abs(p - q) - 1) / 2.0, (v - p - 1) / 2.0
    pillai = float((roots / (1.0 + roots)).sum())
    log_wilks = -float(torch.log1p(roots).sum())
    hotelling = float(roots.sum())
    roy = float(roots[0])
    rows = []
    df1, df2 = s * (2 * m + s + 1), s * (2 * n + s + 1)
    f = c.ratio(df2 * pillai, df1 * (s - pillai)) if df2 > 0 else None
    rows.append(["pillai", pillai, f, df1, df2, c.f_upper(f, df1, df2), pillai / s,
                 "exact" if s == 1 else "approximate"])
    t = math.sqrt((p * p * q * q - 4) / (p * p + q * q - 5)) if p * p + q * q - 5 > 0 else 1.0
    df1, df2 = p * q, (v - (p - q + 1) / 2.0) * t - (p * q - 2) / 2.0
    root = math.exp(log_wilks / t)
    f = c.ratio((1.0 - root) * df2, root * df1) if df2 > 0 else None
    rows.append(["wilks", math.exp(log_wilks), f, df1, df2, c.f_upper(f, df1, df2),
                 1.0 - math.exp(log_wilks / s), "exact" if s <= 2 else "approximate"])
    df1, df2 = s * (2 * m + s + 1), 2 * (s * n + 1)
    f = df2 * hotelling / (s * s * (2 * m + s + 1)) if df2 > 0 else None
    rows.append(["hotelling", hotelling, f, df1, df2, c.f_upper(f, df1, df2),
                 hotelling / (s + hotelling), "exact" if s == 1 else "approximate"])
    r = max(p, q)
    df1, df2 = r, v - r + q
    f = roy * df2 / r if df2 > 0 else None
    rows.append(["roy", roy, f, df1, df2, c.f_upper(f, df1, df2), roy / (1.0 + roy),
                 "exact" if s == 1 else "upper_bound"])
    return rows


def box_m(y: Tensor, cell: Tensor, cells: int) -> dict[str, Any] | None:
    """Box's M test of equal covariance matrices across cells (Box 1949).

    M = (N - g) ln|S| - sum (n_i - 1) ln|S_i| with pooled S; with
    c1 = [sum 1/(n_i-1) - 1/(N-g)] (2p^2+3p-1) / (6 (p+1)(g-1)) and
    c2 = [sum 1/(n_i-1)^2 - 1/(N-g)^2] (p-1)(p+2) / (6 (g-1)):
    chi2 = M (1 - c1) with df1 = p(p+1)(g-1)/2, and the F approximation with
    df2 = (df1 + 2) / |c2 - c1^2|: F = M (1 - c1 - df1/df2) / df1 when c2 > c1^2,
    otherwise F = df2 M / (df1 (b - M)), b = df2 / (1 - c1 + 2/df2).
    Returns None when a cell has no more observations than outcomes or a singular
    covariance matrix.
    """
    p = y.shape[1]
    counts = torch.bincount(cell, minlength=cells)
    if cells < 2 or bool((counts <= p).any()):
        return None
    order = torch.argsort(cell, stable=True)
    ordered = y[order]
    pooled = torch.zeros((p, p), dtype=torch.float64)
    log_dets = []
    start = 0
    for count in counts.tolist():
        block = ordered[start:start + count]
        block = block - block.mean(0)
        sscp = block.T @ block
        pooled += sscp
        sign, log_det = torch.linalg.slogdet(sscp / (count - 1))
        if float(sign) <= 0 or not math.isfinite(float(log_det)):
            return None
        log_dets.append(float(log_det))
        start += count
    total = int(counts.sum())
    df = (counts - 1).to(torch.float64)
    sign, log_pooled = torch.linalg.slogdet(pooled / (total - cells))
    statistic = (total - cells) * float(log_pooled) - float((df * torch.tensor(
        log_dets, dtype=torch.float64)).sum())
    statistic = max(statistic, 0.0)
    c1 = (float((1.0 / df).sum()) - 1.0 / (total - cells)) * (2 * p * p + 3 * p - 1) \
        / (6.0 * (p + 1) * (cells - 1))
    c2 = (float((1.0 / df.square()).sum()) - 1.0 / (total - cells) ** 2) * (p - 1) * (p + 2) \
        / (6.0 * (cells - 1))
    df1 = p * (p + 1) * (cells - 1) / 2.0
    chi2 = statistic * (1.0 - c1)
    f = df2 = None
    gap = c2 - c1 * c1
    if gap > 0:
        df2 = (df1 + 2.0) / gap
        f = statistic * (1.0 - c1 - df1 / df2) / df1
    elif gap < 0:
        df2 = (df1 + 2.0) / -gap
        bound = df2 / (1.0 - c1 + 2.0 / df2)
        f = df2 * statistic / (df1 * (bound - statistic)) if bound > statistic else None
    return {"statistic": statistic, "f": f, "df1": df1, "df2": df2,
            "p_value": c.f_upper(f, df1, df2) if df2 else None, "chi2": chi2,
            "chi2_p_value": c.chi2_upper(chi2, df1)}


def manova(data: Any, y: list[str], factors: list[str] | None = None, *,
           covariates: list[str] | None = None, interactions: Any = "full",
           alpha: float = 0.05, missing: str = "drop") -> TableSet:
    """Multivariate analysis of variance / covariance with Type III hypotheses.

    Model: the general linear model of ``oe.anova`` (sum-to-zero coded factors and
    interactions, linear covariates) fitted to all outcomes at once. For every term
    the hypothesis SSCP matrix H (Type III) and the residual SSCP matrix E give the
    eigenvalues of E^{-1}H and the four multivariate statistics with their F
    approximations (see ``multivariate_tests``): Pillai's trace, Wilks' lambda (Rao's
    F), the Hotelling-Lawley trace and Roy's largest root (the largest eigenvalue,
    as SPSS and Stata report it; its F is an upper bound).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : two or more numeric outcome columns (one is allowed and reproduces ANOVA).
    factors : categorical columns.
    covariates : numeric columns entering linearly (MANCOVA).
    interactions : ``"full"`` (default), ``"none"`` or a list such as ``[["a", "b"]]``.
    alpha : kept in ``attrs`` for reporting.
    missing : ``"drop"`` (listwise over all columns used) or ``"raise"``.

    Returns
    -------
    TableSet with

    * ``multivariate``: effect, test (``pillai``, ``wilks``, ``hotelling``, ``roy``), value,
      statistic (F), df1, df2, p_value, partial_eta_squared and f_type (``exact``,
      ``approximate`` or ``upper_bound``) for the Intercept and every term;
    * ``univariate``: the Type III ANOVA table of each outcome (outcome, source, ss,
      df, ms, statistic, p_value, partial_eta_squared);
    * ``box_m``: Box's M test of equal covariance matrices across the cells of the
      factors (statistic, f, df1, df2, p_value, chi2, chi2_p_value), omitted when a
      cell has no more observations than outcomes.

    ``attrs``: ``n``, ``n_missing``, ``df_resid``, ``outcomes``, ``terms``, ``notes``.

    Multivariate partial eta squared follows SPSS: V / s (Pillai), 1 - L^(1/s)
    (Wilks), (U/s) / (1 + U/s) (Hotelling), l_1 / (1 + l_1) (Roy).

    Equivalent commands: SPSS ``GLM y1 y2 BY a b WITH x /PRINT=HOMOGENEITY``;
    Stata ``manova y1 y2 = a##b c.x``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"y1": [2.1, 3.0, 2.6, 4.2, 4.8, 4.1, 6.0, 5.4, 5.9, 3.3, 4.9, 6.2],
    ...         "y2": [1.0, 1.4, 0.7, 1.9, 2.4, 1.6, 2.2, 2.9, 2.1, 1.2, 2.0, 2.8],
    ...         "g": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "a", "b", "c"]}
    >>> result = oe.manova(data, ["y1", "y2"], ["g"])
    >>> list(result["multivariate"]["test"][:4])
    ['pillai', 'wilks', 'hotelling', 'roy']
    """
    outcomes = c.name_list(y, "y")
    factors = c.name_list(factors, "factors", minimum=0)
    covariates = c.name_list(covariates, "covariates", minimum=0)
    alpha = c.check_alpha(alpha)
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from .streaming_manova import manova as replay_manova
        return replay_manova(data, outcomes, factors, covariates, interactions, alpha, missing)
    model, dropped, values = prepare(data, outcomes, factors, covariates, interactions, False,
                                     missing)
    p = len(outcomes)
    if model.df_error < p:
        raise AnalysisError("insufficient_observations", f"MANOVA needs at least as many error "
                            f"degrees of freedom ({model.df_error}) as outcomes ({p}).")
    centred = values - model.shift
    if bool((model.error.diagonal() <= 1e-24 * centred.square().sum(0)).any()):
        raise AnalysisError("perfect_fit", "The model fits an outcome exactly, so the error "
                            "matrix is singular.")
    multivariate = []
    for term in model.terms:
        rows = kernel_call(multivariate_tests, model.hypothesis(term, 3), model.error,
                           model.df(term), model.df_error)
        multivariate += [[term.name, *row] for row in rows]
    univariate = []
    for column, name in enumerate(outcomes):
        (rows, labels), _ = anova_rows(model, column, 3, values)
        univariate += [[name, label, *row] for label, row in zip(labels, rows, strict=True)]
    tables = {
        "multivariate": c.frame(multivariate, columns=[
            "effect", "test", "value", "statistic", "df1", "df2", "p_value",
            "partial_eta_squared", "f_type"]),
        "univariate": c.frame(univariate, columns=["outcome", "source", *ANOVA_COLUMNS]),
    }
    notes = []
    if factors:
        cell = torch.zeros(model.n, dtype=torch.int64)
        for name in factors:
            cell = cell * len(model.levels[name]) + model.codes[name]
        _, cell = torch.unique(cell, return_inverse=True)
        box = box_m(centred, cell, int(cell.max()) + 1)
        if box is None:
            notes.append("Box's M is not computed: a cell has no more observations than "
                         "outcomes or a singular covariance matrix.")
        else:
            tables["box_m"] = c.frame([list(box.values())], columns=list(box), index=["box_m"])
    return TableSet(tables, title=f"Multivariate analysis of variance of {', '.join(outcomes)}",
                    n=model.n, n_missing=dropped, df_resid=model.df_error, outcomes=outcomes,
                    terms=[term.name for term in model.terms], ss_type=3, alpha=alpha,
                    notes=notes, coding="sum-to-zero (effect) coding", missing="listwise")
