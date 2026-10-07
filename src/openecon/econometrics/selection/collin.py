"""Collinearity diagnostics: ``oe.collin``.

SPSS ``REGRESSION /STATISTICS=COLLIN TOL`` and Stata ``estat vif`` / ``collin``
/ ``coldiag2``: tolerances and variance inflation factors, and the
Belsley-Kuh-Welsch condition indices with variance-decomposition proportions.

Everything comes from ONE Householder QR of the design ``D = [1, X - s]`` (``s``
is a rough mean, removed only for numerical precision), accumulated over row
blocks: stacking the current triangular factor on the next block of rows and
factorizing again gives exactly the R factor of the whole matrix, with memory
bounded by one block. ``X'X`` is never formed.

With ``D = Q R`` and ``R = [[r00, r01'], [0, R11]]``:

- ``R11`` is the triangular factor of the mean-centred regressors (the constant
  is projected out by the first Householder step). After scaling its columns
  to unit length, ``VIF_k = sum_j (R11^{-1})_kj^2``, the k-th diagonal element
  of the inverse correlation matrix; tolerance = 1 / VIF = 1 - R_k^2.
- ``R T`` with ``T = [[1, s'], [0, I]]`` is the triangular factor of the raw
  design ``[1, X]``. Scaling its columns to unit length and taking the singular
  value decomposition ``U diag(d) V'`` gives the eigenvalues ``d_j^2`` of the
  scaled cross-product matrix, the condition indices ``d_max / d_j`` and the
  variance-decomposition proportions ``pi_jk = (v_kj^2 / d_j^2) / sum_j (v_kj^2 / d_j^2)``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call
from openecon.econometrics.selection import common as c
from openecon.engines.contracts import KernelError

# A column whose squared length after projecting out the earlier columns is below
# this share of its own squared length is an exact linear combination of them
# (the threshold of engines.linalg.collinear_columns).
_DEPENDENT = 1e-13


class BlockQR:
    """Triangular factor of a tall matrix read in row blocks."""

    def __init__(self) -> None:
        self.r: Tensor | None = None

    @torch.no_grad()
    def add(self, block: Tensor) -> None:
        stacked = block if self.r is None else torch.cat([self.r, block], dim=0)
        try:
            self.r = torch.linalg.qr(stacked, mode="r").R
        except RuntimeError as exc:
            raise KernelError("numerical_failure", f"The QR factorization failed: {exc}") from exc


@torch.no_grad()
def _unit_columns(r: Tensor) -> tuple[Tensor, Tensor]:
    """(R with unit-length columns, the column lengths); a zero column keeps length 1."""
    length = torch.linalg.vector_norm(r, dim=0)
    safe = torch.where(length > 0, length, torch.ones_like(length))
    return r / safe, length


@torch.no_grad()
def vif_from_factor(r: Tensor) -> tuple[Tensor, int | None]:
    """(variance inflation factors, index of the first dependent column or None).

    ``r`` is the triangular factor of the regressors (already centred when the
    model has a constant). Its squared diagonal, with unit-length columns, is
    the tolerance of column j given the columns before it.
    """
    scaled, _ = _unit_columns(r)
    dependent = (scaled.diagonal().square() <= _DEPENDENT).nonzero().flatten()
    if dependent.numel():
        return torch.empty(0, dtype=torch.float64), int(dependent[0])
    eye = torch.eye(scaled.shape[0], dtype=torch.float64)
    inverse = torch.linalg.solve_triangular(scaled, eye, upper=True)
    return inverse.square().sum(dim=1), None


@torch.no_grad()
def condition_from_factor(r: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(eigenvalues, condition indices, variance proportions [dimension, coefficient])."""
    scaled, _ = _unit_columns(r)
    try:
        _, singular, vh = torch.linalg.svd(scaled)
    except RuntimeError as exc:
        raise KernelError("numerical_failure", f"The singular value decomposition failed: "
                          f"{exc}") from exc
    if not float(singular[-1]) > 0:
        raise KernelError("perfect_collinearity", "The design is exactly rank deficient.")
    share = (vh / singular[:, None]).square()            # [dimension j, coefficient k]
    return singular.square(), singular[0] / singular, share / share.sum(dim=0)


def collin(data: Any, x: list[str], *, intercept: bool = True, missing: str = "drop") -> TableSet:
    """Collinearity diagnostics of a set of regressors.

    Reports, for the design matrix ``[1, x_1, ..., x_k]`` of a linear model
    (the outcome plays no role):

    - **tolerance** and **VIF**. With ``R_k^2`` the R-squared of the regression
      of ``x_k`` on the other regressors (and the constant),
      ``tolerance_k = 1 - R_k^2`` and ``VIF_k = 1 / tolerance_k``: the factor
      by which collinearity inflates the sampling variance of ``b_k``.
    - **condition indices and variance-decomposition proportions** (Belsley,
      Kuh and Welsch 1980). The columns of the design, including the constant
      and NOT centred, are scaled to unit length; with singular values
      ``d_1 >= ... >= d_p`` of the scaled design, the eigenvalues are ``d_j^2``,
      the condition indices ``d_1 / d_j``, and the share of ``var(b_k)``
      associated with dimension j is ``(v_kj^2 / d_j^2) / sum_j (v_kj^2 / d_j^2)``.
      Each coefficient's proportions sum to one over the dimensions. A large
      condition index (30 is the usual warning level) with two or more
      proportions above 0.5 points to the regressors involved in a near
      dependency.

    Computed from one QR factorization read in row blocks (see the module
    docstring); no cross-product matrix is inverted.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    x : numeric regressors (indicator columns must be created beforehand).
    intercept : True (default) analyses the model with a constant: centred
        VIFs (Stata ``estat vif``, SPSS Tolerance / VIF) and a condition table
        that includes the constant (SPSS Collinearity Diagnostics, Stata
        ``coldiag2``). False analyses the model without a constant: uncentred
        VIFs (Stata ``estat vif, uncentered``) and no constant column.
    missing : ``"drop"`` (default) excludes rows with a missing value in any
        regressor (listwise); ``"raise"`` rejects them.

    Returns
    -------
    TableSet with tables

    - ``vif``: one row per regressor: ``tolerance``, ``vif``, ``r_squared``.
    - ``condition``: one row per dimension (index 1, 2, ...): ``eigenvalue``,
      ``condition_index`` and one column of variance proportions per
      coefficient (``Intercept`` first when the model has a constant).

    ``attrs``: ``n``, ``n_missing``, ``mean_vif``, ``max_vif``,
    ``condition_number`` (the largest condition index), ``intercept``.

    An exact dependency (a duplicated column, a sum of other columns, a column
    without variation next to the constant) has an infinite VIF; it is reported
    as an error that names the offending column.

    Equivalent commands: SPSS ``REGRESSION /STATISTICS=COLLIN TOL``; Stata
    ``regress y x1 x2 x3`` then ``estat vif`` and ``coldiag2`` (or ``collin x1
    x2 x3``).

    Example
    -------
    >>> import openecon as oe
    >>> data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
    ...         "c": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]}
    >>> result = oe.collin(data, ["a", "b", "c"])
    >>> round(float(result["vif"].loc["a", "vif"]), 4)
    3.5547
    """
    names = c.name_list(x, "x", minimum=1)
    c.check_flag(intercept, "intercept")
    c.check_choice(missing, "missing", ("drop", "raise"))
    frame = c.source(data, names, numeric=names)
    k = len(names)
    shift = c.rough_means(frame, names) if intercept else torch.zeros(k, dtype=torch.float64)
    factor = BlockQR()
    n = dropped = 0
    peak = torch.zeros(k, dtype=torch.float64)
    for _, _, block in c.blocks(frame, names, k + 1):
        complete = ~torch.isnan(block).any(dim=1)
        if not bool(complete.all()):
            dropped += int((~complete).sum())
            block = block[complete]
        if not len(block):
            continue
        peak = torch.maximum(peak, c.block_peak(block, "A regressor"))
        if intercept:
            block = torch.cat([torch.ones((len(block), 1), dtype=torch.float64),
                               block - shift], dim=1)
        kernel_call(factor.add, block)
        n += len(block)
    if dropped and missing == "raise":
        raise AnalysisError("missing_values", f"{dropped} observation(s) have missing values in "
                            "the regressors. Choose missing='drop' to exclude them.")
    if not n:
        raise AnalysisError("empty_sample", "No complete observations remain after excluding "
                            "missing values.")
    c.check_scale(peak, names)
    width = k + int(intercept)
    if n <= width:
        raise AnalysisError("insufficient_observations", f"Collinearity diagnostics of {k} "
                            f"regressor(s) need more than {width} complete observations; "
                            f"{n} are available.")
    r = factor.r
    if intercept:
        # Rounding-level variation next to the constant: sum of squares about the mean
        # against n * mean^2.
        centred = r[1:, 1:]
        means = shift + r[0, 1:] / r[0, 0]
        variation = centred.square().sum(dim=0)
        flat = (variation <= 1e-24 * n * means.square()) | (variation == 0)
        if bool(flat.any()):
            raise AnalysisError("zero_variance", "These regressors do not vary and are "
                                "collinear with the constant: "
                                f"{', '.join(names[i] for i in flat.nonzero().flatten().tolist())}"
                                ". Drop them or use intercept=False.")
        transform = torch.eye(width, dtype=torch.float64)
        transform[0, 1:] = shift
        raw = r @ transform                      # triangular factor of [1, X]
    else:
        centred = raw = r
        empty = (r.square().sum(dim=0) == 0).nonzero().flatten().tolist()
        if empty:
            raise AnalysisError("zero_variance", "These regressors are zero in every "
                                f"observation: {', '.join(names[i] for i in empty)}.")
    vif, dependent = kernel_call(vif_from_factor, centred)
    if dependent is not None:
        others = "the regressors before it" + (" and the constant" if intercept else "")
        raise AnalysisError("perfect_collinearity", f"'{names[dependent]}' is an exact linear "
                            f"combination of {others}: its tolerance is zero and its VIF is "
                            "infinite. Drop it (or one of the columns it depends on) and run "
                            "the diagnostics again.")
    tolerance = 1.0 / vif
    vif_table = c.result_table(
        torch.stack([tolerance, vif, 1.0 - tolerance], dim=1).tolist(), index=names,
        columns=["tolerance", "vif", "r_squared"])
    eigenvalues, indices, proportions = kernel_call(condition_from_factor, raw)
    terms = [*(["Intercept"] if intercept else []), *names]
    condition = c.result_table(
        torch.cat([eigenvalues[:, None], indices[:, None], proportions], dim=1).tolist(),
        index=list(range(1, width + 1)), columns=["eigenvalue", "condition_index", *terms])
    condition.index.name = "dimension"
    return TableSet(
        {"vif": vif_table, "condition": condition}, title="Collinearity diagnostics",
        n=n, n_missing=dropped, mean_vif=float(vif.mean()), max_vif=float(vif.max()),
        condition_number=float(indices[-1]), intercept=intercept, missing="listwise")
