"""Principal component analysis: ``oe.pca`` and ``oe.pca_scores`` (Stata pca, SPSS FACTOR PC)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import common as c

# Stata's default for pca: components with an eigenvalue above 1e-5 are retained.
_MINEIGEN = 1e-5


def retained(eigenvalues: Tensor, count: int | None, mineigen: float, what: str) -> int:
    """Number of leading components kept: at most ``count``, eigenvalue > ``mineigen``."""
    keep = int((eigenvalues > mineigen).sum())
    if count is not None:
        keep = min(keep, count)
    if keep < 1:
        raise AnalysisError("no_components", f"No {what} has an eigenvalue above {mineigen:g}; "
                            "lower mineigen.")
    return keep


def eigen_table(eigenvalues: Tensor, prefix: str, total: float | None = None) -> pd.DataFrame:
    """Stata's eigenvalue table: eigenvalue, difference, proportion, cumulative."""
    total = float(eigenvalues.sum()) if total is None else total
    values = eigenvalues.tolist()
    rows, running = [], 0.0
    for i, value in enumerate(values):
        running += value
        rows.append([value, value - values[i + 1] if i + 1 < len(values) else None,
                     value / total, running / total])
    return c.frame(rows, columns=["eigenvalue", "difference", "proportion", "cumulative"],
                   index=c.numbered(prefix, len(values)))


@c.procedure
def pca(data: Any, columns: list[str], *, matrix: str = "correlation",
        components: int | None = None, mineigen: float | None = None,
        missing: str = "drop", weights: str | None = None,
        weight_type: str = "fweight") -> TableSet:
    """Principal component analysis of a correlation or covariance matrix.

    Model. With C the correlation matrix R (default) or the covariance matrix S
    (divisor n - 1) of the p variables, the eigen-decomposition

        C = V L V',   L = diag(l_1 >= ... >= l_p),   V'V = I

    gives the principal components: component k is the linear combination with
    weights v_k (column k of V) and variance l_k. The share of the total variance
    trace(C) carried by the first m components is ``rho``.

    Computation. Means and the centred cross-product matrix come from one pass
    over the data (X'X after centring); the p-by-p matrix is decomposed with
    ``torch.linalg.eigh``. The cost is O(n p^2) and no n-by-n matrix is formed.

    Sign convention. Each eigenvector is signed so that its entry of largest
    absolute value is positive (Stata and SPSS leave the sign to the eigen
    routine, so a whole column may differ in sign from their output).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    columns : two or more numeric columns.
    matrix : ``"correlation"`` (default; variables standardized) or ``"covariance"``.
    components : keep at most this many components (Stata ``components(#)``).
    mineigen : keep components whose eigenvalue exceeds this value. Default 1e-5
        as in Stata; SPSS's default rule (eigenvalues above 1) is ``mineigen=1``.
    missing : ``"drop"`` (listwise deletion, reported in ``n_missing``) or ``"raise"``.

    Returns
    -------
    TableSet with

    * ``eigenvalues``: eigenvalue, difference, proportion and cumulative proportion of
      all p components (Stata's first table; SPSS "Total Variance Explained");
    * ``eigenvectors``: the retained unit-length eigenvectors (Stata's "Principal
      components (eigenvectors)") and ``unexplained``, the variance of each variable
      not reproduced by the retained components;
    * ``loadings``: component loadings eigenvector * sqrt(eigenvalue) (SPSS's
      "Component Matrix": with the correlation matrix, the correlations between
      variables and components);
    * ``communalities``: ``initial`` (1, or the variance for a covariance matrix) and
      ``extraction`` (sum of squared loadings of the retained components);
    * ``descriptives``: mean and standard deviation (divisor n - 1) of each variable.

    ``attrs``: ``n``, ``n_missing``, ``matrix``, ``variables``, ``components``,
    ``trace`` (total variance) and ``rho`` (share explained by the retained components).

    Equivalent commands: Stata ``pca x1 x2 x3`` (``, covariance``); SPSS
    ``FACTOR /VARIABLES x1 x2 x3 /EXTRACTION PC`` (``/METHOD=COVARIANCE``).
    Component scores: ``oe.pca_scores(result, data)``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x1": [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
    ...         "x2": [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9],
    ...         "x3": [1.2, 0.3, 1.1, 1.4, 1.2, 1.9, 0.8, 0.9, 0.7, 0.2]}
    >>> result = oe.pca(data, ["x1", "x2", "x3"])
    >>> round(result.attrs["trace"], 6)
    3.0
    """
    names = c.name_list(columns, "columns", minimum=2)
    c.check_choice(matrix, "matrix", ("correlation", "covariance"))
    if components is not None:
        components = c.check_count(components, "components")
    cutoff = _MINEIGEN if mineigen is None else c.check_number(mineigen, "mineigen", minimum=0.0)
    c.check_choice(weight_type, "weight_type", ("fweight",))
    if weights is not None:
        from .weighted import frequency_moments
        mean, sscp, n, dropped, extra = frequency_moments(data, names, weights, missing)
        return _from_moments(mean, sscp, n, names, matrix, components, cutoff, dropped, extra)
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from .replay import moments as replay_moments
        state, extra, dropped = replay_moments(data, names, missing)
        return _from_moments(state.location, state.sscp, state.n, names, matrix,
                             components, cutoff, dropped, extra)
    sample, _, dropped = c.select(data, names, missing=missing)
    x = c.matrix(sample, names)
    n, p = x.shape
    if n < 2:
        raise AnalysisError("insufficient_observations", "Principal components need at least "
                            "two complete observations.")
    mean, sscp, _ = c.moments(x, names)
    return _from_moments(mean, sscp, n, names, matrix, components, cutoff, dropped)


def _from_moments(mean: Tensor, sscp: Tensor, n: int, names: list[str], matrix: str,
                  components: int | None, cutoff: float, dropped: int,
                  extra: dict[str, Any] | None = None) -> TableSet:
    covariance = sscp / (n - 1)
    std = covariance.diagonal().sqrt()
    target = c.correlation(sscp) if matrix == "correlation" else covariance
    eigenvalues, vectors = c.descending_eigh(target)
    eigenvalues = eigenvalues.clamp_min(0.0)
    trace = float(target.diagonal().sum())
    keep = retained(eigenvalues, components, cutoff, "component")
    labels = c.numbered("Comp", keep)
    kept = vectors[:, :keep]
    loadings = kept * eigenvalues[:keep].sqrt()
    extraction = loadings.square().sum(1)
    unexplained = target.diagonal() - extraction
    unexplained = torch.where(unexplained <= 1e-12 * target.diagonal(),
                              torch.zeros_like(unexplained), unexplained)
    tables = {
        "eigenvalues": eigen_table(eigenvalues, "Comp", trace),
        "eigenvectors": c.frame(torch.cat([kept, unexplained[:, None]], dim=1),
                                columns=[*labels, "unexplained"], index=names),
        "loadings": c.frame(loadings, columns=labels, index=names),
        "communalities": c.frame(torch.stack([target.diagonal(), extraction], dim=1),
                                 columns=["initial", "extraction"], index=names),
        "descriptives": c.frame(torch.stack([mean, std], dim=1), columns=["mean", "std_dev"],
                                index=names),
    }
    return TableSet(
        tables, title=f"Principal components ({matrix} matrix) of {', '.join(names)}",
        procedure="pca", n=n, n_missing=dropped, matrix=matrix, variables=names,
        components=keep, trace=trace, rho=min(1.0, float(eigenvalues[:keep].sum()) / trace),
        mineigen=cutoff, missing="listwise",
        sign_convention="largest absolute entry of each eigenvector is positive", **(extra or {}))


def standardize(result: TableSet, data: Any, names: list[str], *, scale: bool,
                centre: bool = True) -> tuple[Tensor, pd.Series]:
    """(z [m, p] for the complete rows of ``data``, mask of those rows).

    The variables are centred at the estimation-sample means and, when ``scale``,
    divided by the estimation-sample standard deviations stored in ``descriptives``.
    """
    frame = c.source(data)
    c.require_numeric(frame, names)
    frame = frame.loc[:, names]
    keep = ~frame.isna().any(axis=1)
    x = c.matrix(frame.loc[keep], names)
    stats = result["descriptives"]
    mean = torch.as_tensor(stats.loc[names, "mean"].to_numpy(dtype="float64").copy())
    std = torch.as_tensor(stats.loc[names, "std_dev"].to_numpy(dtype="float64").copy())
    if centre:
        if result.attrs.get("training_means_supplied") is False:
            raise AnalysisError("missing_training_moments", "Supply the actual training means to score a summary-matrix result.")
        x = x - mean
    if scale and result.attrs.get("training_sds_supplied") is False:
        raise AnalysisError("missing_training_moments", "Supply the actual training standard deviations to score a correlation-matrix result.")
    return (x / std if scale else x), keep


@c.procedure
def pca_scores(result: TableSet, data: Any, *, center: bool = True,
               normalize: bool = False) -> pd.DataFrame:
    """Principal component scores of the rows of ``data`` (Stata ``predict`` after pca).

    For a correlation-matrix PCA the score on component k is z'v_k, where z holds
    the variables standardized with the estimation-sample means and standard
    deviations and v_k is the eigenvector; its variance in the estimation sample
    is the eigenvalue l_k. For a covariance-matrix PCA the variables are centred
    but not scaled.

    Parameters
    ----------
    result : the TableSet returned by ``oe.pca``.
    data : rows to score; must contain the analysed columns. Rows with a missing
        value get missing scores.
    center : covariance-matrix PCA only. ``True`` (default) scores the centred
        variables; ``False`` applies the eigenvectors to the raw variables, which is
        what Stata's ``predict`` does unless its ``center`` option is given.
    normalize : divide each score by sqrt(l_k), giving unit-variance scores (the
        component scores saved by SPSS FACTOR /SAVE REG with PC extraction).

    Returns
    -------
    A table indexed like ``data`` with one column per retained component
    (``Comp1``, ``Comp2``, ...).

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x1": [2.5, 0.5, 2.2, 1.9, 3.1], "x2": [2.4, 0.7, 2.9, 2.2, 3.0]}
    >>> scores = oe.pca_scores(oe.pca(data, ["x1", "x2"]), data)
    >>> list(scores.columns)
    ['Comp1', 'Comp2']
    """
    c.check_result(result, "pca", "pca_scores")
    c.check_flag(center, "center")
    c.check_flag(normalize, "normalize")
    names = list(result.attrs["variables"])
    correlation = result.attrs["matrix"] == "correlation"
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        return _streaming_scores(result, data, center=center, normalize=normalize)
    z, keep = standardize(result, data, names, scale=correlation, centre=center or correlation)
    vectors = result["eigenvectors"].drop(columns="unexplained")
    weights = torch.as_tensor(vectors.loc[names].to_numpy(dtype="float64").copy())
    scores = z @ weights
    if normalize:
        values = result["eigenvalues"]["eigenvalue"].to_numpy(dtype="float64")
        scores = scores / torch.as_tensor(values[:weights.shape[1]].copy()).sqrt()
    return c.aligned(scores, keep, list(vectors.columns))


def _streaming_scores(result: TableSet, data: Any, *, center: bool, normalize: bool):
    from .replay import score_source
    names = list(result.attrs["variables"])
    columns = list(result["eigenvectors"].drop(columns="unexplained").columns)
    return score_source(data, names, columns,
                        lambda raw: pca_scores(result, raw, center=center, normalize=normalize),
                        procedure="pca_scores", options={"center": center, "normalize": normalize})
