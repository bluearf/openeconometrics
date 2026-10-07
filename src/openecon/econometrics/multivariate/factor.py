"""Factor analysis: ``oe.factor``, ``oe.factortest`` and ``oe.factor_scores``.

Stata ``factor`` / ``rotate`` / ``predict`` / ``estat kmo``; SPSS ``FACTOR`` with
PAF / ML / PC extraction, rotations, KMO and Bartlett's test, saved scores.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import common as c
from openecon.econometrics.multivariate import extraction as ex
from openecon.econometrics.multivariate import rotation as rot
from openecon.econometrics.multivariate.pca import eigen_table, standardize

METHODS = ("pf", "ipf", "ml", "pcf")
ROTATIONS = (*rot.ORTHOGONAL, *rot.OBLIQUE)
# Stata's defaults: factors with an eigenvalue above 5e-6 (pf, ipf, ml), above 1 for pcf.
_MINEIGEN = {"pf": 5e-6, "ipf": 5e-6, "ml": 5e-6, "pcf": 1.0}
# SPSS's defaults for iterated principal axis factoring; BFGS and rotations get 1000.
_IPF_ITERATIONS, _IPF_TOLERANCE, _ITERATIONS = 25, 1e-3, 1000
# Gradient projection converges linearly; its steps cost microseconds.
_ROTATION_ITERATIONS = 10000


def _sphericity(r: Tensor, n: int) -> tuple[float | None, float, float | None]:
    """Bartlett's test that R = I: chi2 = -(n - 1 - (2p + 5)/6) ln|R|, df = p(p - 1)/2."""
    p = r.shape[0]
    sign, log_det = torch.linalg.slogdet(r)
    multiplier = n - 1.0 - (2.0 * p + 5.0) / 6.0
    df = p * (p - 1) / 2.0
    if float(sign) <= 0 or not math.isfinite(float(log_det)) or multiplier <= 0:
        return None, df, None
    statistic = max(-multiplier * float(log_det), 0.0)
    return statistic, df, c.chi2_upper(statistic, df)


def _correlation(data: Any, names: list[str], missing: str, diagnostics: dict | None = None
                 ) -> tuple[Tensor, Tensor, Tensor, int, int]:
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from .replay import moments as replay_moments
        state, extra, dropped = replay_moments(data, names, missing, minimum=3)
        if diagnostics is not None:
            diagnostics.update(extra)
        return c.correlation(state.sscp), state.location, (state.sscp.diagonal() / (state.n - 1)).sqrt(), state.n, dropped
    sample, _, dropped = c.select(data, names, missing=missing)
    x = c.matrix(sample, names)
    n = x.shape[0]
    if n < 3:
        raise AnalysisError("insufficient_observations", "Factor analysis needs at least three "
                            "complete observations.")
    mean, sscp, _ = c.moments(x, names)
    return c.correlation(sscp), mean, (sscp.diagonal() / (n - 1)).sqrt(), n, dropped


def _score_coefficients(method: str, r: Tensor, pattern: Tensor, phi: Tensor,
                        uniqueness: Tensor, names: list[str]) -> Tensor:
    if method == "regression":
        factor = c.positive_definite(
            r, "correlation matrix", "regression factor scores need R^{-1}. Remove the "
            "redundant variable.")
        return torch.cholesky_solve(pattern @ phi, factor)
    bad = [name for name, value in zip(names, uniqueness.tolist(), strict=True)
           if not value > 1e-10]
    if bad:
        raise AnalysisError("heywood_case", "Bartlett factor scores weight each variable by "
                            f"1/uniqueness, but the uniqueness of {', '.join(bad)} is not "
                            "positive (a Heywood case). Use scores='regression'.")
    weighted = pattern / uniqueness[:, None]
    inner = c.positive_definite(
        pattern.T @ weighted, "matrix L'Psi^{-1}L", "the factors are not separately "
        "identified. Extract fewer factors.")
    return torch.cholesky_solve(weighted.T, inner).T


def _variance_table(loadings: Tensor, rotated: rot.Rotation | None, p: int) -> pd.DataFrame:
    """SPSS 'Total Variance Explained': sums of squared loadings as shares of p."""
    ss = loadings.square().sum(0)
    columns = [ss, ss / p, torch.cumsum(ss, 0) / p]
    names = ["variance", "proportion", "cumulative"]
    if rotated is not None:
        if rotated.oblique:
            # Correlated factors: the variance of a factor ignoring the others is the sum
            # of its squared structure loadings; shares do not add up.
            rss = (rotated.pattern @ rotated.phi).square().sum(0)
            columns += [rss, rss / p]
            names += ["rotated_variance", "rotated_proportion"]
        else:
            rss = rotated.pattern.square().sum(0)
            columns += [rss, rss / p, torch.cumsum(rss, 0) / p]
            names += ["rotated_variance", "rotated_proportion", "rotated_cumulative"]
    return c.frame(torch.stack(columns, dim=1), columns=names,
                   index=c.numbered("Factor", loadings.shape[1]))


@c.procedure
def factor(data: Any, columns: list[str], *, method: str = "pf", factors: int | None = None,
           mineigen: float | None = None, rotate: str | None = None, kaiser: bool = True,
           power: float = 4.0, gamma: float = 0.0, scores: str | None = None,
           max_iterations: int | None = None, tolerance: float | None = None,
           missing: str = "drop") -> TableSet:
    """Exploratory factor analysis of a correlation matrix, with rotation and scores.

    Model. The p standardized variables are z = L f + e with m common factors f
    (unit variance), loadings L [p, m] and independent unique parts e with
    variances Psi (the uniquenesses), so that R = L Phi L' + Psi; Phi = I before
    rotation and after an orthogonal rotation.

    Extraction (``method``)

    * ``"pf"`` principal factors (Stata's default): the diagonal of R is replaced by
      the squared multiple correlations (SMC) and L = V_m sqrt(D_m) from the
      eigen-decomposition of that reduced matrix.
    * ``"ipf"`` iterated principal factors (SPSS PAF, Stata ipf): the communalities
      are re-estimated from L and the reduced matrix decomposed again until the
      largest change is below ``tolerance`` (default 0.001 and at most 25
      iterations, SPSS's defaults; non-convergence is an error).
    * ``"pcf"`` principal-component factors: communalities of one (SPSS PC).
    * ``"ml"`` maximum likelihood by Joreskog's method: the concentrated discrepancy
      F(psi) = sum_{j>m} (g_j - ln g_j - 1), g_j the eigenvalues of
      Psi^{-1/2} R Psi^{-1/2}, is minimized by BFGS with its analytic gradient over
      uniquenesses bounded below by 0.005; a uniqueness at the bound is a Heywood
      case, reported in ``attrs["heywood"]`` with a note.

    Number of factors. ``factors`` is the maximum number kept; factors whose
    eigenvalue (of the SMC-reduced matrix for pf / ipf / ml, of R for pcf) does not
    exceed ``mineigen`` are dropped. Defaults follow Stata: 5e-6 (pf, ipf, ml) and
    1 (pcf). SPSS's default (eigenvalues of R above 1) is obtained by passing the
    number of such eigenvalues as ``factors``.

    Rotation (``rotate``). ``"varimax"``, ``"quartimax"``, ``"equamax"`` (orthogonal,
    orthomax family), ``"oblimin"`` (direct oblimin with ``gamma``, SPSS's delta;
    0 = quartimin) and ``"promax"`` (varimax followed by the power-``power``
    Procrustes target; SPSS's default kappa is 4, Stata's ``promax(3)``).
    ``kaiser=True`` applies Kaiser's row normalization during rotation (SPSS's
    default; Stata rotates without it unless ``normalize`` is given). Rotated
    factors are ordered by decreasing sum of squared loadings. Loadings columns
    are signed so that the loading of largest absolute value is positive.

    Scores. ``scores="regression"`` (default for ``oe.factor_scores``; Thomson):
    coefficients B = R^{-1} S with S the structure matrix L Phi; ``"bartlett"``:
    B = Psi^{-1} L (L' Psi^{-1} L)^{-1}. Scores are z'B for standardized z.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    columns : three or more numeric columns (two are allowed for pcf).
    method, factors, mineigen, rotate, kaiser, power, gamma, scores : see above.
    max_iterations : iteration limit of ipf (default 25), of the ML optimizer
        (default 1000) and of the rotation (default 10000).
    tolerance : convergence tolerance of ipf on the communalities (default 0.001).
    missing : ``"drop"`` (listwise deletion) or ``"raise"``.

    Returns
    -------
    TableSet with

    * ``eigenvalues``: for every factor the eigenvalue of the factored matrix with
      difference, proportion and cumulative proportion of its trace (Stata's table;
      for ml the sums of squared loadings of the retained factors), and
      ``initial_eigenvalue``, the eigenvalue of R (SPSS "Initial Eigenvalues");
    * ``variance``: sums of squared loadings of the retained factors as shares of
      the total variance p, before and after rotation (for oblique rotations from
      the structure matrix, without a cumulative share);
    * ``loadings``: unrotated loadings (factor matrix);
    * ``rotated_loadings``: rotated pattern matrix; ``structure`` and
      ``factor_correlations`` for oblique rotations; ``rotation_matrix`` M with
      rotated = unrotated x M (only with ``rotate``);
    * ``communalities``: ``initial`` (SMC; 1 for pcf), ``extraction`` and
      ``uniqueness`` (1 - communality; the estimated Psi for ml);
    * ``uniqueness``: the uniquenesses alone (Stata's "Uniqueness" column);
    * ``fit``: Bartlett-corrected likelihood-ratio tests against the saturated model:
      ``independence`` (R = I; chi2 = -(n - 1 - (2p+5)/6) ln|R|, df = p(p-1)/2) and,
      for ml, ``model`` (chi2 = (n - 1 - (2p+5)/6 - 2m/3) F_min,
      df = ((p-m)^2 - (p+m))/2);
    * ``score_coefficients``: factor score coefficients (regression unless
      ``scores="bartlett"``);
    * ``descriptives``: mean and standard deviation of each variable.

    ``attrs``: ``n``, ``n_missing``, ``method``, ``factors``, ``rotate``, ``kaiser``,
    ``variables``, ``iterations``, ``rotation_iterations``, ``discrepancy`` (ml),
    ``heywood`` (variable names), ``scores``, ``notes``.

    Equivalent commands: Stata ``factor x1-x6, ipf factors(2)`` then
    ``rotate, varimax normalize`` and ``predict f1 f2``; SPSS ``FACTOR /VARIABLES
    x1 TO x6 /CRITERIA FACTORS(2) /EXTRACTION PAF /ROTATION VARIMAX /SAVE REG(ALL)``.

    Example
    -------
    >>> import openecon as oe
    >>> import random
    >>> rng = random.Random(1)
    >>> f = [rng.gauss(0, 1) for _ in range(200)]
    >>> data = {f"x{j}": [0.8 * v + 0.6 * rng.gauss(0, 1) for v in f] for j in range(1, 5)}
    >>> result = oe.factor(data, ["x1", "x2", "x3", "x4"], method="ml", factors=1)
    >>> result.attrs["factors"]
    1
    """
    c.check_choice(method, "method", METHODS)
    names = c.name_list(columns, "columns", minimum=2 if method == "pcf" else 3)
    if factors is not None:
        factors = c.check_count(factors, "factors")
    cutoff = _MINEIGEN[method] if mineigen is None else c.check_number(
        mineigen, "mineigen", minimum=0.0)
    if rotate is not None:
        c.check_choice(rotate, "rotate", ROTATIONS)
    c.check_flag(kaiser, "kaiser")
    power = c.check_number(power, "power", minimum=1.0)
    gamma = c.check_number(gamma, "gamma", maximum=1.0)
    if scores is not None:
        c.check_choice(scores, "scores", ("regression", "bartlett"))
    if max_iterations is not None:
        max_iterations = c.check_count(max_iterations, "max_iterations")
    tol = _IPF_TOLERANCE if tolerance is None else c.check_number(
        tolerance, "tolerance", minimum=0.0, exclusive=True)
    diagnostics: dict[str, Any] = {}
    r, mean, std, n, dropped = _correlation(data, names, missing, diagnostics)
    p = len(names)
    notes: list[str] = []

    # Number of factors from the eigenvalues of the matrix that is factored first.
    initial = c.descending_eigh(r)[0].clamp_min(0.0)
    if method == "pcf":
        smc, screen = torch.ones(p, dtype=c.FLOAT), initial
    else:
        smc = ex.squared_multiple_correlations(r)
        screen = ex.reduced_eigen(r, smc)[0]
    limit = p if factors is None else min(factors, p)
    if method == "ml":
        limit = min(limit, max(m for m in range(p) if (p - m) ** 2 >= p + m))
    m = min(int((screen > cutoff).sum()), limit)
    if m < 1:
        raise AnalysisError("no_factors", f"No factor has an eigenvalue above {cutoff:g}; lower "
                            "mineigen or use method='pcf'.")
    if factors is not None and m < factors:
        notes.append(f"{m} factor(s) retained instead of the {factors} requested: the others "
                     "have eigenvalues below mineigen or are not identified.")

    if method == "pf":
        fit = ex.principal_factors(r, smc, m)
    elif method == "pcf":
        fit = ex.principal_component_factors(r, m)
    elif method == "ipf":
        fit = ex.iterated_principal_factors(r, smc, m, max_iter=max_iterations or _IPF_ITERATIONS,
                                            tol=tol)
    else:
        fit = ex.maximum_likelihood(r, smc, m, max_iter=max_iterations or _ITERATIONS)
    loadings = c.fix_signs(fit.loadings)
    communality = loadings.square().sum(1)
    uniqueness = fit.uniqueness if method == "ml" else 1.0 - communality
    heywood = [names[i] for i in fit.heywood]
    if method != "ml":
        heywood = [name for name, value in zip(names, communality.tolist(), strict=True)
                   if value >= 1.0]
    if heywood:
        where = (f"the uniqueness of {', '.join(heywood)} is at its lower bound "
                 f"{ex.PSI_FLOOR:g}" if method == "ml" else
                 f"the communality of {', '.join(heywood)} reached or exceeded 1")
        notes.append(f"Heywood case: {where}; interpret the solution, the tests and the factor "
                     "scores with caution.")

    labels = c.numbered("Factor", m)
    eigen_rows = eigen_table(fit.factored, "Factor",
                             float(fit.factored.sum()) if method == "ml" else None)
    if method == "ml":
        eigen_rows = eigen_rows.reindex(c.numbered("Factor", p))
    eigen_rows["initial_eigenvalue"] = initial.tolist()
    rotated = None
    if rotate is not None:
        rotated = rot.rotate(loadings, rotate, normalize=kaiser, power=power, gamma=gamma,
                             max_iter=max_iterations or _ROTATION_ITERATIONS)
        if m < 2:
            notes.append("A single factor cannot be rotated; the rotated loadings equal the "
                         "unrotated ones.")
        elif not rotated.converged:
            notes.append("The rotation stopped with the criterion gradient below 1e-5 (the "
                         "SPSS tolerance) but above the 1e-10 aimed at.")
    pattern = loadings if rotated is None else rotated.pattern
    phi = torch.eye(m, dtype=c.FLOAT) if rotated is None else rotated.phi

    tables = {
        "eigenvalues": eigen_rows,
        "variance": _variance_table(loadings, rotated, p),
        "loadings": c.frame(loadings, columns=labels, index=names),
    }
    if rotated is not None:
        tables["rotated_loadings"] = c.frame(pattern, columns=labels, index=names)
        if rotated.oblique:
            tables["structure"] = c.frame(pattern @ phi, columns=labels, index=names)
            tables["factor_correlations"] = c.frame(phi, columns=labels, index=labels)
        tables["rotation_matrix"] = c.frame(rotated.matrix, columns=labels, index=labels)
    tables["communalities"] = c.frame(
        torch.stack([smc, communality, uniqueness], dim=1),
        columns=["initial", "extraction", "uniqueness"], index=names)
    tables["uniqueness"] = c.frame(uniqueness[:, None], columns=["uniqueness"], index=names)

    statistic, df, p_value = _sphericity(r, n)
    fit_rows, fit_index = [[statistic, df, p_value]], ["independence"]
    if method == "ml":
        df_model = ((p - m) ** 2 - (p + m)) / 2.0
        multiplier = n - 1.0 - (2.0 * p + 5.0) / 6.0 - 2.0 * m / 3.0
        chi2 = multiplier * fit.discrepancy if multiplier > 0 and df_model > 0 else None
        fit_rows.insert(0, [chi2, df_model, c.chi2_upper(chi2, df_model)])
        fit_index.insert(0, "model")
        if df_model == 0:
            notes.append("The ML model has zero degrees of freedom (it is saturated), so its "
                         "chi-square test is undefined.")
    tables["fit"] = c.frame(fit_rows, columns=["statistic", "df", "p_value"], index=fit_index)

    score_method = scores or "regression"
    try:
        coefficients = _score_coefficients(score_method, r, pattern, phi, uniqueness, names)
    except AnalysisError:
        if scores is not None:
            raise
        coefficients = None
        notes.append("Regression factor score coefficients are not available: the correlation "
                     "matrix is singular.")
    if coefficients is not None:
        tables["score_coefficients"] = c.frame(coefficients, columns=labels, index=names)
    tables["descriptives"] = c.frame(torch.stack([mean, std], dim=1),
                                     columns=["mean", "std_dev"], index=names)
    title = {"pf": "principal factors", "ipf": "iterated principal factors",
             "pcf": "principal-component factors", "ml": "maximum likelihood"}[method]
    return TableSet(
        tables, title=f"Factor analysis ({title}) of {', '.join(names)}",
        procedure="factor", n=n, n_missing=dropped, method=method, factors=m, rotate=rotate,
        kaiser=kaiser if rotate else None, power=power if rotate == "promax" else None,
        gamma=gamma if rotate == "oblimin" else None, variables=names,
        iterations=fit.iterations, discrepancy=fit.discrepancy,
        rotation_iterations=None if rotated is None else rotated.iterations,
        heywood=heywood, scores=score_method if coefficients is not None else None,
        mineigen=cutoff, notes=notes, missing="listwise",
        sign_convention="largest absolute loading of each factor is positive", **diagnostics)


@c.procedure
def factortest(data: Any, columns: list[str], *, missing: str = "drop") -> pd.DataFrame:
    """Kaiser-Meyer-Olkin sampling adequacy and Bartlett's test of sphericity.

    With R the correlation matrix and a_ij = -r^ij / sqrt(r^ii r^jj) the
    anti-image (partial) correlations from R^{-1},

        KMO   = sum_{i != j} r_ij^2 / (sum_{i != j} r_ij^2 + sum_{i != j} a_ij^2),
        MSA_j = sum_{i != j} r_ij^2 / (sum_{i != j} r_ij^2 + sum_{i != j} a_ij^2)  (row j),

    and Bartlett's test of H0: R = I is chi2 = -(n - 1 - (2p + 5)/6) ln|R| with
    p(p - 1)/2 degrees of freedom.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    columns : two or more numeric columns.
    missing : ``"drop"`` (listwise deletion) or ``"raise"``.

    Returns
    -------
    A table with one row per variable (``kmo`` = its measure of sampling adequacy
    and ``smc`` = its squared multiple correlation) and a final ``Overall`` row with
    the KMO measure. ``attrs``: ``kmo``, ``statistic`` (Bartlett's chi2), ``df``,
    ``p_value``, ``determinant`` (|R|), ``n``, ``n_missing``.

    Equivalent commands: Stata ``estat kmo`` (after ``factor``), SPSS
    ``FACTOR /PRINT KMO``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [1.1, 1.9, 3.2, 3.8, 5.1, 6.3],
    ...         "c": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]}
    >>> oe.factortest(data, ["a", "b", "c"]).attrs["df"]
    3.0
    """
    names = c.name_list(columns, "columns", minimum=2)
    diagnostics: dict[str, Any] = {}
    r, _, _, n, dropped = _correlation(data, names, missing, diagnostics)
    factor_ = c.positive_definite(
        r, "correlation matrix", "a variable is a linear combination of the others, so the "
        "anti-image correlations are undefined. Remove the redundant variable.")
    inverse = torch.cholesky_inverse(factor_)
    scale = inverse.diagonal().sqrt()
    anti = -inverse / torch.outer(scale, scale)
    off = ~torch.eye(len(names), dtype=torch.bool)
    r2 = (r.square() * off).sum(1)
    a2 = (anti.square() * off).sum(1)
    msa = r2 / (r2 + a2)
    overall = float(r2.sum() / (r2.sum() + a2.sum()))
    smc = 1.0 - 1.0 / inverse.diagonal()
    statistic, df, p_value = _sphericity(r, n)
    rows = [[c.finite(a), b] for a, b in zip(msa.tolist(), smc.tolist(), strict=True)]
    rows.append([c.finite(overall), None])
    sign, log_det = torch.linalg.slogdet(r)
    return c.frame(rows, columns=["kmo", "smc"], index=[*names, "Overall"],
                   procedure="factortest", kmo=c.finite(overall), statistic=statistic, df=df,
                   p_value=p_value, determinant=float(sign * torch.exp(log_det)), n=n,
                   n_missing=dropped, missing="listwise", **diagnostics)


@c.procedure
def factor_scores(result: TableSet, data: Any) -> pd.DataFrame:
    """Factor scores of the rows of ``data`` (Stata ``predict`` after factor, SPSS /SAVE).

    The variables are standardized with the means and standard deviations of the
    estimation sample and multiplied by the score coefficients stored in the
    result: regression scores (B = R^{-1} L Phi) unless the factor analysis was run
    with ``scores="bartlett"`` (B = Psi^{-1} L (L'Psi^{-1}L)^{-1}). Rotated
    solutions give scores of the rotated factors.

    Parameters
    ----------
    result : the TableSet returned by ``oe.factor``.
    data : rows to score; must contain the analysed columns. Rows with a missing
        value get missing scores.

    Returns
    -------
    A table indexed like ``data`` with columns ``Factor1``, ``Factor2``, ...

    Example
    -------
    >>> import openecon as oe
    >>> data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0], "b": [1.1, 1.9, 3.2, 3.8, 5.1, 6.3],
    ...         "c": [2.0, 1.0, 4.0, 3.0, 6.0, 5.5]}
    >>> result = oe.factor(data, ["a", "b", "c"], method="pcf", factors=1)
    >>> oe.factor_scores(result, data).shape
    (6, 1)
    """
    c.check_result(result, "factor", "factor_scores")
    if "score_coefficients" not in result:
        raise AnalysisError("scores_unavailable", "This factor analysis has no score "
                            "coefficients (its correlation matrix is singular). Remove the "
                            "redundant variable and run oe.factor again.")
    names = list(result.attrs["variables"])
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from .replay import score_source
        return score_source(data, names, list(result["score_coefficients"].columns),
                            lambda raw: factor_scores(result, raw), procedure="factor_scores")
    z, keep = standardize(result, data, names, scale=True)
    coefficients = result["score_coefficients"]
    weights = torch.as_tensor(coefficients.loc[names].to_numpy(dtype="float64").copy())
    return c.aligned(z @ weights, keep, list(coefficients.columns))
