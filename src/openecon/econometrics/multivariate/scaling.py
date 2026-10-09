"""Multidimensional scaling and correspondence analysis: ``oe.mds`` and ``oe.ca``.

Stata ``mds`` / ``mdsmat`` and ``ca``; SPSS ``PROXSCAL`` / ``ALSCAL`` and
``CORRESPONDENCE``.
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

_SHOWN_EIGENVALUES = 10          # Stata's neigen(10)


def _distance_input(distances: Any, max_n: int) -> tuple[Tensor, list[Any]]:
    """A validated symmetric dissimilarity matrix and the labels of its objects."""
    labels: list[Any] | None = None
    if isinstance(distances, pd.DataFrame):
        if distances.shape[0] == distances.shape[1]:
            labels = [c.label(item) for item in distances.columns]
        distances = distances.to_numpy(dtype="float64", na_value=float("nan")).copy()
    try:
        d = torch.as_tensor(distances, dtype=c.FLOAT)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_distances", "distances must be a square table of "
                            "numbers.") from exc
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 2:
        raise AnalysisError("invalid_distances", "distances must be a square matrix with at "
                            "least two objects.")
    n = d.shape[0]
    if n > max_n:
        raise AnalysisError("too_many_observations", f"Scaling {n} objects needs {n}-by-{n} "
                            f"matrices, above the max_n = {max_n} guard; raise max_n "
                            "deliberately or scale a subset.")
    if not bool(torch.isfinite(d).all()) or bool((d < 0).any()):
        raise AnalysisError("invalid_distances", "distances must be finite and non-negative.")
    size = float(d.abs().max())
    if float((d - d.T).abs().max()) > 1e-8 * max(size, 1e-300) \
            or float(d.diagonal().abs().max()) > 1e-8 * max(size, 1e-300):
        raise AnalysisError("invalid_distances", "distances must be symmetric with a zero "
                            "diagonal.")
    d = (d + d.T) / 2
    d.diagonal().fill_(0.0)
    return d, labels if labels is not None else list(range(1, n + 1))


def _principal_axes(x: Tensor) -> Tensor:
    """Centre a configuration and rotate it to its principal axes (largest first)."""
    x = x - x.mean(0)
    _, vectors = c.descending_eigh(x.T @ x)
    return c.fix_signs(x @ vectors)


def smacof(delta: Tensor, start: Tensor, *, max_iter: int, tol: float
           ) -> tuple[Tensor, float, int, bool]:
    """Metric MDS by stress majorization (de Leeuw 1977): (X, raw stress, iterations, ok).

    Raw stress sigma(X) = sum_{i<j} (d_ij(X) - delta_ij)^2 never increases under the
    Guttman transform X <- B(X) X / n, b_ij = -delta_ij / d_ij(X) for i != j
    (0 where d_ij = 0) and b_ii = -sum_{j != i} b_ij. Iterations stop when the
    relative decrease of the stress is below ``tol``.
    """
    n = delta.shape[0]
    x = start.clone()

    def stress(config: Tensor) -> tuple[float, Tensor]:
        d = torch.cdist(config, config, compute_mode="donot_use_mm_for_euclid_dist")
        return float((d - delta).square().sum()) / 2.0, d

    value, d = stress(x)
    scale = float(delta.square().sum()) / 2.0
    for iteration in range(1, max_iter + 1):
        ratio = torch.where(d > 0, delta / d.clamp_min(1e-300), torch.zeros_like(d))
        b = -ratio
        b.diagonal().copy_(ratio.sum(1))
        x = (b @ x) / n
        new_value, d = stress(x)
        if value - new_value <= tol * max(value, 1e-300) or new_value <= 1e-28 * scale:
            return x, new_value, iteration, True
        value = new_value
    return x, value, max_iter, False


@c.procedure
def mds(data: Any = None, columns: list[str] | None = None, *, distances: Any = None,
        dimensions: int = 2, method: str = "classical", standardize: bool = False,
        max_iterations: int = 1000, tolerance: float = 1e-9, max_n: int = 5000,
        missing: str = "drop") -> TableSet:
    """Multidimensional scaling: classical (Torgerson) or by stress majorization.

    Input is either ``data`` with ``columns`` (the Euclidean distances between the
    observations are scaled) or a square symmetric dissimilarity matrix
    ``distances`` (Stata ``mdsmat``).

    Classical scaling (``method="classical"``). With D2 the squared dissimilarities
    and J = I - 11'/n, the doubly centred matrix B = -J D2 J / 2 is decomposed,
    B = V L V', and the coordinates in k dimensions are V_k L_k^{1/2} (principal
    coordinates). For Euclidean distances between observations B = X X' for the
    centred data, so the solution is obtained from the p-by-p cross-product
    matrix without forming any n-by-n matrix. Mardia's fit measures are
    sum_{i<=k} l_i / sum |l_i| and sum_{i<=k} l_i^2 / sum l_i^2.

    Modern metric scaling (``method="smacof"``). Starting from the classical
    solution, raw stress sum_{i<j} (d_ij - delta_ij)^2 is minimized by iterated
    Guttman transforms (SMACOF; identity transformation of the dissimilarities).
    Kruskal's stress-1 = sqrt(sum (d_ij - delta_ij)^2 / sum d_ij^2) is reported and
    the configuration is rotated to its principal axes.

    Parameters
    ----------
    data, columns : observations and one or more numeric columns; or
    distances : square symmetric matrix (DataFrame, nested list or tensor) with a
        zero diagonal; a DataFrame's column labels name the objects.
    dimensions : number of dimensions k (default 2).
    method : ``"classical"`` (default) or ``"smacof"``.
    standardize : with ``data``: scale z-scores instead of the raw variables.
    max_iterations, tolerance : SMACOF iteration limit and relative stress decrease
        at which it stops (reaching the limit is reported, not an error: the stress
        never increases).
    max_n : largest number of objects accepted (the result has one row per object
        and SMACOF or a distance matrix need n-by-n storage).
    missing : ``"drop"`` (listwise deletion of observations) or ``"raise"``.

    Returns
    -------
    TableSet with

    * ``coordinates``: one row per object (labelled by the 0-based row position in
      ``data``, or by the labels of ``distances``) and columns ``dim1``, ``dim2``, ...;
    * ``eigenvalues`` (classical): the leading eigenvalues of B with
      ``abs_percent``, ``abs_cumulative``, ``squared_percent``, ``squared_cumulative``
      (Stata's table);
    * ``fit`` (one ``value`` column): ``mardia1`` and ``mardia2`` (classical), or
      ``stress1``, ``raw_stress``, ``iterations`` (smacof).

    ``attrs``: ``n``, ``n_missing``, ``dimensions``, ``method``, ``converged``, and the
    fit measures.

    Sign convention: each dimension is signed so that its coordinate of largest
    absolute value is positive.

    Equivalent commands: Stata ``mds x1 x2 x3, id(id)`` / ``mdsmat D`` (classical),
    ``mds ..., method(modern) loss(stress)``; SPSS ``PROXSCAL`` (ratio
    transformation) / ``ALSCAL``.

    Example
    -------
    >>> import openecon as oe
    >>> d = [[0.0, 3.0, 4.0], [3.0, 0.0, 5.0], [4.0, 5.0, 0.0]]
    >>> result = oe.mds(distances=d, dimensions=2)
    >>> result["coordinates"].shape
    (3, 2)
    """
    c.check_choice(method, "method", ("classical", "smacof"))
    dimensions = c.check_count(dimensions, "dimensions")
    max_n = c.check_count(max_n, "max_n", minimum=2)
    max_iterations = c.check_count(max_iterations, "max_iterations")
    tolerance = c.check_number(tolerance, "tolerance", minimum=0.0)
    c.check_flag(standardize, "standardize")
    if (data is None) == (distances is None):
        raise AnalysisError("invalid_spec", "Give either data with columns, or a distances "
                            "matrix (not both).")
    dropped = 0
    x = delta = None
    if distances is not None:
        delta, labels = _distance_input(distances, max_n)
        n = delta.shape[0]
        # Double centring in O(n^2): b_ij = -(d2_ij - rowmean_i - rowmean_j + mean) / 2.
        b = delta.square()
        row_mean = b.mean(1, keepdim=True)
        b = -0.5 * (b - row_mean - row_mean.T + row_mean.mean())
        values, vectors = c.descending_eigh(b)
    else:
        names = c.name_list(columns, "columns", minimum=1)
        sample, keep, dropped = c.select(data, names, missing=missing)
        n = len(sample)
        if n < 2:
            raise AnalysisError("insufficient_observations", "Scaling needs at least two "
                                "complete observations.")
        if n > max_n:
            raise AnalysisError("too_many_observations", f"The configuration would have {n} "
                                f"rows, above the max_n = {max_n} guard. Scale a subset, use "
                                "oe.pca for a low-dimensional summary, or raise max_n.")
        _, sscp, x = c.moments(c.matrix(sample, names), names, need_variation=standardize)
        if standardize:
            x = x / (sscp.diagonal() / (n - 1)).sqrt()
            sscp = x.T @ x
        labels = torch.nonzero(torch.as_tensor(keep.to_numpy())).flatten().tolist()
        # B = X X' has the non-zero eigenvalues of X'X and eigenvectors X v / sqrt(l).
        values, small = c.descending_eigh(sscp)
        vectors = None
    scale = float(values.abs().max())
    positive = int((values > 1e-12 * max(scale, 1e-300)).sum())
    if dimensions > positive:
        raise AnalysisError("too_many_dimensions", f"Only {positive} dimension(s) have a "
                            f"positive eigenvalue; lower dimensions (requested {dimensions}).")
    if vectors is None:
        classical = c.fix_signs(x @ small[:, :dimensions])
    else:
        classical = c.fix_signs(vectors[:, :dimensions] * values[:dimensions].sqrt())
    dims = [f"dim{i}" for i in range(1, dimensions + 1)]
    tables: dict[str, pd.DataFrame] = {}
    attrs: dict[str, Any] = {}
    if method == "classical":
        config, converged = classical, True
        total_abs, total_sq = float(values.abs().sum()), float(values.square().sum())
        shown = min(values.numel(), max(dimensions, _SHOWN_EIGENVALUES))
        lead = values[:shown]
        tables["eigenvalues"] = c.frame(
            torch.stack([lead, 100.0 * lead.abs() / total_abs,
                         100.0 * torch.cumsum(lead.abs(), 0) / total_abs,
                         100.0 * lead.square() / total_sq,
                         100.0 * torch.cumsum(lead.square(), 0) / total_sq], dim=1),
            columns=["eigenvalue", "abs_percent", "abs_cumulative", "squared_percent",
                     "squared_cumulative"], index=c.numbered("Dim", shown))
        attrs = {"mardia1": float(values[:dimensions].sum()) / total_abs,
                 "mardia2": float(values[:dimensions].square().sum()) / total_sq}
        fit = [["mardia1", attrs["mardia1"]], ["mardia2", attrs["mardia2"]]]
    else:
        if delta is None:
            delta = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")
        config, raw, iterations, converged = smacof(delta, classical, max_iter=max_iterations,
                                                    tol=tolerance)
        config = _principal_axes(config)
        fitted = torch.cdist(config, config, compute_mode="donot_use_mm_for_euclid_dist")
        denominator = float(fitted.square().sum()) / 2.0
        stress1 = math.sqrt(raw / denominator) if denominator > 0 else None
        attrs = {"stress1": stress1, "raw_stress": raw, "iterations": iterations}
        fit = [["stress1", stress1], ["raw_stress", raw], ["iterations", iterations]]
        if not converged:
            attrs["notes"] = [f"SMACOF stopped at max_iterations = {max_iterations} before the "
                              "stress settled; the configuration is the last iterate."]
    tables = {"coordinates": c.frame(config, columns=dims, index=labels), **tables,
              "fit": c.frame([[value] for _, value in fit], columns=["value"],
                             index=[name for name, _ in fit])}
    return TableSet(
        tables, title=f"Multidimensional scaling ({method}, {dimensions} dimensions)",
        procedure="mds", n=n, n_missing=dropped, dimensions=dimensions, method=method,
        converged=converged, missing="listwise" if distances is None else None,
        sign_convention="largest absolute coordinate of each dimension is positive", **attrs)


@c.procedure
def ca(data: Any, row: str, column: str, *, weights: str | None = None, dimensions: int = 2,
       missing: str = "drop") -> TableSet:
    """Simple correspondence analysis of a two-way contingency table.

    The table N (rows x columns, built from the two categorical columns, optionally
    weighted by a count column) is turned into the correspondence matrix
    P = N / n with row masses r and column masses c. The singular value
    decomposition of the standardized residuals

        S = D_r^{-1/2} (P - r c') D_c^{-1/2} = U diag(s_k) V'

    gives the principal inertias s_k^2 (their sum, the total inertia, is the
    Pearson chi-square divided by n), the standard coordinates D_r^{-1/2} U and
    D_c^{-1/2} V, and the principal coordinates (standard coordinates times s_k),
    whose Euclidean distances approximate chi-square distances between profiles.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    row, column : the two categorical columns (any scalar labels).
    weights : optional numeric column of non-negative cell counts or frequency
        weights (for data that are already tabulated).
    dimensions : number of dimensions reported (default 2, at most
        min(rows, columns) - 1; a smaller table is reported with a note).
    missing : ``"drop"`` (rows with a missing value in a used column are excluded) or
        ``"raise"``.

    Returns
    -------
    TableSet with

    * ``inertia``: for every dimension ``singular_value``, ``principal_inertia``,
      ``chi2`` (n times the inertia), ``percent`` and ``cumulative`` percent;
    * ``rows`` and ``columns``: for each category ``mass``, ``quality`` (share of its
      inertia represented in the reported dimensions), ``inertia`` (its share of the
      total inertia) and, per dimension k, the principal coordinate ``dim{k}``, the
      standard coordinate ``dim{k}_standard``, the squared correlation
      ``dim{k}_sqcorr`` and the contribution of the category to the dimension
      ``dim{k}_contribution``;
    * ``table``: the contingency table that was analysed.

    ``attrs``: ``n`` (table total), ``n_missing``, ``chi2``, ``df``, ``p_value``,
    ``total_inertia``, ``dimensions``, ``notes``.

    Normalizations. Stata's default symmetric coordinates are the standard
    coordinates times sqrt(s_k); the principal coordinates here are Stata's
    ``normalize(principal)`` and SPSS's ``/NORMALIZATION=PRINCIPAL``.

    Sign convention: each dimension is signed so that the entry of largest absolute
    value of its row singular vector (sqrt(mass) times the standard coordinate) is
    positive.

    Equivalent commands: Stata ``ca rowvar colvar, dim(2) normalize(principal)``;
    SPSS ``CORRESPONDENCE TABLE=rowvar(1,4) BY colvar(1,3)``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"hair": ["dark", "dark", "fair", "fair", "red", "red"],
    ...         "eye": ["brown", "blue", "brown", "blue", "brown", "blue"],
    ...         "count": [68, 20, 15, 94, 26, 17]}
    >>> result = oe.ca(data, "hair", "eye", weights="count")
    >>> result.attrs["df"]
    2
    """
    row = c.check_name(row, "row")
    column = c.check_name(column, "column")
    dimensions = c.check_count(dimensions, "dimensions")
    used = [row, column] + ([c.check_name(weights, "weights")] if weights is not None else [])
    sample, _, dropped = c.select(data, used, numeric=used[2:], missing=missing)
    row_codes, row_labels = c.group_codes(sample[row], row)
    col_codes, col_labels = c.group_codes(sample[column], column)
    rows_, cols_ = len(row_labels), len(col_labels)
    weight = torch.ones(len(sample), dtype=c.FLOAT)
    if weights is not None:
        weight = c.column(sample, weights)
        if bool((weight < 0).any()):
            raise AnalysisError("invalid_weights", f"Weights in '{weights}' must be "
                                "non-negative counts.")
    counts = torch.zeros(rows_ * cols_, dtype=c.FLOAT).index_add_(
        0, row_codes * cols_ + col_codes, weight).reshape(rows_, cols_)
    notes: list[str] = []
    keep_rows, keep_cols = counts.sum(1) > 0, counts.sum(0) > 0
    if not bool(keep_rows.all()) or not bool(keep_cols.all()):
        notes.append("Categories with a zero total were removed from the table.")
        counts = counts[keep_rows][:, keep_cols]
        row_labels = [item for item, ok in zip(row_labels, keep_rows.tolist(), strict=True) if ok]
        col_labels = [item for item, ok in zip(col_labels, keep_cols.tolist(), strict=True) if ok]
        rows_, cols_ = counts.shape
    if rows_ < 2 or cols_ < 2:
        raise AnalysisError("degenerate_table", "Correspondence analysis needs at least two "
                            f"row and two column categories; the table is {rows_} x {cols_}.")
    total = float(counts.sum())
    p = counts / total
    r, cm = p.sum(1), p.sum(0)
    residual = (p - torch.outer(r, cm)) / torch.outer(r.sqrt(), cm.sqrt())
    u, singular, vh = torch.linalg.svd(residual, full_matrices=False)
    available = min(rows_, cols_) - 1
    u, singular, v = u[:, :available], singular[:available], vh.T[:, :available]
    pivot = u.abs().argmax(0)
    sign = torch.sign(u[pivot, torch.arange(available)])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    u, v = u * sign, v * sign
    inertia = singular.square()
    total_inertia = float(inertia.sum())
    if not total_inertia > 1e-20:
        raise AnalysisError("degenerate_table", "The table has zero inertia: every row has the "
                            "same profile, so there is nothing to scale.")
    k = min(dimensions, available)
    if k < dimensions:
        notes.append(f"The table supports {available} dimension(s); {k} reported instead of "
                     f"the {dimensions} requested.")

    def profile(vectors: Tensor, mass: Tensor, labels_: list[Any]) -> pd.DataFrame:
        standard = vectors / mass.sqrt()[:, None]
        principal = standard * singular
        distance = principal.square().sum(1)                  # chi-square distance to centroid
        safe = distance.clamp_min(1e-300)
        columns_ = [mass, principal[:, :k].square().sum(1) / safe,
                    mass * distance / total_inertia]
        names_ = ["mass", "quality", "inertia"]
        for j in range(k):
            columns_ += [principal[:, j], standard[:, j], principal[:, j].square() / safe,
                         vectors[:, j].square()]
            names_ += [f"dim{j + 1}", f"dim{j + 1}_standard", f"dim{j + 1}_sqcorr",
                       f"dim{j + 1}_contribution"]
        return c.frame(torch.stack(columns_, dim=1), columns=names_, index=labels_)

    chi2 = total * total_inertia
    df = (rows_ - 1) * (cols_ - 1)
    tables = {
        "inertia": c.frame(
            torch.stack([singular, inertia, total * inertia, 100.0 * inertia / total_inertia,
                         100.0 * torch.cumsum(inertia, 0) / total_inertia], dim=1),
            columns=["singular_value", "principal_inertia", "chi2", "percent", "cumulative"],
            index=c.numbered("Dim", available)),
        "rows": profile(u, r, row_labels),
        "columns": profile(v, cm, col_labels),
        "table": c.frame(counts, columns=[str(item) for item in col_labels], index=row_labels),
    }
    return TableSet(
        tables, title=f"Correspondence analysis of {row} by {column}", procedure="ca",
        n=total, n_missing=dropped, chi2=chi2, df=df, p_value=c.chi2_upper(chi2, df),
        total_inertia=total_inertia, dimensions=k, notes=notes, missing="listwise",
        projection_state={"version": 1, "row_labels": row_labels, "column_labels": col_labels,
                          "row_mass": r.tolist(), "column_mass": cm.tolist(),
                          "row_standard": (u[:, :k] / r.sqrt()[:, None]).tolist(),
                          "column_standard": (v[:, :k] / cm.sqrt()[:, None]).tolist(),
                          "singular_values": singular[:k].tolist()},
        sign_convention="largest absolute row singular-vector entry of each dimension is "
                        "positive")
