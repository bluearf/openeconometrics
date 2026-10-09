"""Canonical correlation analysis: ``oe.canon`` (Stata canon, SPSS CANCORR / MANOVA).

Computation (Bjorck and Golub 1973). The centred data matrices are factored
X = Q_x R_x and Y = Q_y R_y by Householder QR; the singular value decomposition

    Q_x' Q_y = U diag(rho_1 >= ... >= rho_s) V',     s = min(p, q),

gives the canonical correlations rho_k directly, and the raw coefficients are
A = sqrt(n - 1) R_x^{-1} U and B = sqrt(n - 1) R_y^{-1} V (triangular solves), so
that the canonical variates XA and YB have unit variance. No covariance matrix
is inverted, which keeps the full precision of the data when variables are
nearly collinear.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import common as c


# A squared canonical correlation this close to one is a perfect linear relation
# between the sets up to rounding; the F and chi-square statistics are then undefined.
_PERFECT = 1.0 - 1e-12


def rao_f(log_lambda: float, p: int, q: int, w: float) -> tuple[float | None, float, float]:
    """Rao's F approximation for Wilks' lambda of dimension (p, q).

    t = sqrt((p^2 q^2 - 4) / (p^2 + q^2 - 5)) (1 when undefined), df1 = p q,
    df2 = w t - (p q - 2) / 2 and F = (1 - L^(1/t)) / L^(1/t) * df2 / df1.
    """
    t = math.sqrt((p * p * q * q - 4) / (p * p + q * q - 5)) if p * p + q * q - 5 > 0 else 1.0
    df1, df2 = float(p * q), w * t - (p * q - 2) / 2.0
    root = math.exp(log_lambda / t)
    f = (1.0 - root) / root * df2 / df1 if df2 > 0 and root > 0 else None
    return f, df1, df2


def multivariate_tests(rho2: Tensor, p: int, q: int, n: int) -> list[list[Any]]:
    """Wilks, Pillai, Lawley-Hotelling and Roy tests that all canonical correlations are 0.

    With l_k = rho_k^2 / (1 - rho_k^2), s = min(p, q), m = (|p - q| - 1)/2,
    v = n - 1 - q and h = (v - p - 1)/2:

        Wilks      L = prod (1 - rho_k^2); Rao's F with w = n - 1 - (p + q + 1)/2
        Pillai     V = sum rho_k^2;  F = (2h + s + 1)/(2m + s + 1) * V/(s - V),
                   df = s(2m + s + 1), s(2h + s + 1)
        Hotelling  U = sum l_k;      F = 2(s h + 1) U / (s^2 (2m + s + 1)),
                   df = s(2m + s + 1), 2(s h + 1)
        Roy        l_1;              F = l_1 (n - 1 - r)/r, r = max(p, q),
                   df = r, n - 1 - r  (an upper bound unless s = 1)
    """
    s = min(p, q)
    m, h = (abs(p - q) - 1) / 2.0, (n - 1 - q - p - 1) / 2.0
    exact = bool((rho2 < _PERFECT).all())
    roots = rho2 / (1.0 - rho2).clamp_min(1e-300)
    rows = []
    log_wilks = float(torch.log1p(-rho2.clamp_max(1.0 - 1e-300)).sum()) if exact else -math.inf
    f, df1, df2 = rao_f(log_wilks, p, q, n - 1.0 - (p + q + 1) / 2.0) if exact \
        else (None, float(p * q), 0.0)
    rows.append(["wilks", math.exp(log_wilks), f, df1, df2, c.f_upper(f, df1, df2),
                 "exact" if s <= 2 else "approximate"])
    pillai = float(rho2.sum())
    df1, df2 = s * (2 * m + s + 1), s * (2 * h + s + 1)
    f = df2 * pillai / (df1 * (s - pillai)) if df2 > 0 and pillai < s else None
    rows.append(["pillai", pillai, f, df1, df2, c.f_upper(f, df1, df2),
                 "exact" if s == 1 else "approximate"])
    hotelling = float(roots.sum()) if exact else None
    df1, df2 = s * (2 * m + s + 1), 2 * (s * h + 1)
    f = df2 * hotelling / (s * s * (2 * m + s + 1)) if df2 > 0 and exact else None
    rows.append(["hotelling", hotelling, f, df1, df2, c.f_upper(f, df1, df2),
                 "exact" if s == 1 else "approximate"])
    roy = float(roots[0]) if exact else None
    r = max(p, q)
    df1, df2 = float(r), float(n - 1 - r)
    f = roy * df2 / r if df2 > 0 and exact else None
    rows.append(["roy", roy, f, df1, df2, c.f_upper(f, df1, df2),
                 "exact" if s == 1 else "upper_bound"])
    return rows


def _qr(x: Tensor, names: list[str], what: str) -> tuple[Tensor, Tensor]:
    q, r = torch.linalg.qr(x)
    norms = x.square().sum(0).sqrt()
    dependent = [name for name, pivot, norm in zip(names, r.diagonal().abs().tolist(),
                                                   norms.tolist(), strict=True)
                 if not pivot > 1e-10 * norm]
    if dependent:
        raise AnalysisError("singular_matrix", f"The {what} variables are linearly dependent "
                            f"({', '.join(dependent)} is a combination of the variables before "
                            "it). Remove the redundant variable.")
    return q, r


@c.procedure
def canon(data: Any, x: list[str], y: list[str], *, missing: str = "drop",
          weights: str | None = None, weight_type: str = "fweight") -> TableSet:
    """Canonical correlation analysis of two sets of variables.

    The first pair of canonical variates u_1 = x'a_1 and v_1 = y'b_1 are the unit-
    variance linear combinations of the two sets with the largest correlation
    rho_1; each further pair maximizes the correlation subject to being
    uncorrelated with the earlier pairs. There are s = min(p, q) pairs.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    x : numeric columns of the first set (p variables).
    y : numeric columns of the second set (q variables), distinct from ``x``.
    missing : ``"drop"`` (listwise deletion over both sets) or ``"raise"``.
    weights : optional nonnegative integer frequency-count column. The weighted
        route uses joint moments without replicating observations, with n equal
        to the frequency total and covariance divisor n-1. Probability levels
        retain their Gaussian sampling assumptions; counts must represent
        original independent observations, not repeated-measure amplification.
    weight_type : only ``"fweight"`` is supported.

    Returns
    -------
    TableSet with

    * ``correlations``: for each canonical correlation ``correlation``,
      ``squared_correlation``, ``eigenvalue`` rho^2/(1 - rho^2) and the test that it
      and all smaller correlations are zero: ``wilks_lambda`` = prod_{j>=k}(1 - rho_j^2)
      with Rao's ``f`` (``df1``, ``df2``, ``p_value``) and Bartlett's ``chi2`` =
      -(n - 1 - (p + q + 1)/2) ln(lambda) with ``chi2_df`` = (p - k + 1)(q - k + 1)
      and ``chi2_p_value``;
    * ``tests``: Wilks' lambda, Pillai's trace, the Lawley-Hotelling trace and Roy's
      largest root for the hypothesis that all canonical correlations are zero,
      with their F approximations (``f_type``: exact, approximate or upper_bound);
    * ``raw_coefficients``: coefficients a_k and b_k for the variables in their
      original units (the variates have unit variance); ``standardized_coefficients``:
      for standardized variables. The ``set`` column tells the set of each variable;
    * ``loadings``: canonical loadings (``Canon*``: correlations between each variable
      and the variates of its own set) and cross loadings (``Cross*``: with the
      variates of the other set, = loading * rho_k);
    * ``redundancy``: per pair, the share of each set's variance reproduced by its own
      variate (``x_variance``, ``y_variance``: mean squared loading) and by the
      opposite variate (``x_redundancy``, ``y_redundancy``: that share times rho_k^2).

    ``attrs``: ``n``, ``n_missing``, ``x``, ``y``, ``notes``.

    Sign convention: each pair is signed so that the standardized x-coefficient of
    largest absolute value is positive (the y-coefficients follow, keeping rho > 0).

    Equivalent commands: Stata ``canon (x1 x2 x3) (y1 y2)``, ``estat loadings``,
    ``canon, test(1 2)``; SPSS ``MANOVA x1 x2 x3 WITH y1 y2 /DISCRIM ALL`` or the
    CANCORR macro; SAS ``PROC CANCORR``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0],
    ...         "c": [1.5, 1.9, 3.5, 3.6, 5.8, 5.2, 7.9], "d": [0.3, 0.9, 0.1, 0.8, 0.2, 0.7, 0.4]}
    >>> result = oe.canon(data, x=["a", "b"], y=["c", "d"])
    >>> len(result["correlations"])
    2
    """
    x_names = c.name_list(x, "x", minimum=1)
    y_names = c.name_list(y, "y", minimum=1)
    if set(x_names) & set(y_names):
        raise AnalysisError("invalid_spec", "Canonical x and y sets must contain distinct variables.")
    c.check_choice(weight_type, "weight_type", ("fweight",))
    if weights is not None:
        from .canon_options import weighted_canon
        return weighted_canon(data, x_names, y_names, weights=weights, missing=missing)
    sample, _, dropped = c.select(data, [*x_names, *y_names], missing=missing)
    n = len(sample)
    p, q = len(x_names), len(y_names)
    if n < p + q + 2:
        raise AnalysisError("insufficient_observations", f"Canonical correlation of {p} and "
                            f"{q} variables needs at least {p + q + 2} complete observations, "
                            f"but {n} are available.")
    x_mean, x_sscp, xc = c.moments(c.matrix(sample, x_names), x_names)
    y_mean, y_sscp, yc = c.moments(c.matrix(sample, y_names), y_names)
    x_std = (x_sscp.diagonal() / (n - 1)).sqrt()
    y_std = (y_sscp.diagonal() / (n - 1)).sqrt()
    qx, rx = _qr(xc / x_std, x_names, "x")
    qy, ry = _qr(yc / y_std, y_names, "y")
    u, rho, vh = torch.linalg.svd(qx.T @ qy)
    s = min(p, q)
    rho = rho[:s].clamp(0.0, 1.0)
    rho = torch.where(rho.square() >= _PERFECT, torch.ones_like(rho), rho)
    scale = math.sqrt(n - 1.0)
    # Coefficients for the standardized variables (the QR used unit-variance columns).
    a_std = torch.linalg.solve_triangular(rx, u[:, :s], upper=True) * scale
    b_std = torch.linalg.solve_triangular(ry, vh.T[:, :s], upper=True) * scale
    pivot = a_std.abs().argmax(0)
    sign = torch.sign(a_std[pivot, torch.arange(s)])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    a_std, b_std = a_std * sign, b_std * sign
    x_load = (rx.T @ u[:, :s]) * sign / scale
    y_load = (ry.T @ vh.T[:, :s]) * sign / scale

    return _result(rho, a_std, b_std, x_load, y_load, x_std, y_std,
                   torch.cat((x_mean, y_mean)), n, dropped, x_names, y_names,
                   {"input_kind": "resident_rows", "training_means_supplied": True,
                    "training_sds_supplied": True, "algorithm": "raw-data Householder QR/SVD"})


def _result(rho: Tensor, a_std: Tensor, b_std: Tensor, x_load: Tensor, y_load: Tensor,
            x_std: Tensor, y_std: Tensor, mean: Tensor, n: int, dropped: int,
            x_names: list[str], y_names: list[str], extra: dict) -> TableSet:
    """Shared reporting; the original unweighted QR arithmetic is unchanged."""
    p, q, s = len(x_names), len(y_names), len(rho)

    labels = c.numbered("Canon", s)
    rho2 = rho.square()
    w = n - 1.0 - (p + q + 1) / 2.0
    rows = []
    for k in range(s):
        tail = rho2[k:]
        perfect = bool((tail >= _PERFECT).any())
        log_lambda = -math.inf if perfect else float(torch.log1p(-tail).sum())
        f, df1, df2 = (None, float((p - k) * (q - k)), 0.0) if perfect \
            else rao_f(log_lambda, p - k, q - k, w)
        chi2 = -w * log_lambda if not perfect and w > 0 else None
        chi2_df = float((p - k) * (q - k))
        rows.append([
            float(rho[k]), float(rho2[k]),
            float(rho2[k]) / (1.0 - float(rho2[k])) if rho2[k] < _PERFECT else None,
            math.exp(log_lambda), f, df1, df2 if df2 > 0 else None, c.f_upper(f, df1, df2),
            chi2, chi2_df, c.chi2_upper(chi2, chi2_df)])
    sets = ["x"] * p + ["y"] * q
    variables = [*x_names, *y_names]

    def stacked(first: Tensor, second: Tensor) -> list[list[Any]]:
        return [[kind, *line] for kind, line in zip(
            sets, torch.cat([first, second]).tolist(), strict=True)]

    loadings = [[kind, *own, *cross] for kind, own, cross in zip(
        sets, torch.cat([x_load, y_load]).tolist(),
        torch.cat([x_load * rho, y_load * rho]).tolist(), strict=True)]
    x_share, y_share = x_load.square().mean(0), y_load.square().mean(0)
    tables = {
        "correlations": c.frame(rows, columns=[
            "correlation", "squared_correlation", "eigenvalue", "wilks_lambda", "f", "df1",
            "df2", "p_value", "chi2", "chi2_df", "chi2_p_value"], index=labels),
        "tests": c.frame([row[1:] for row in multivariate_tests(rho2, p, q, n)],
                         columns=["value", "statistic", "df1", "df2", "p_value", "f_type"],
                         index=["wilks", "pillai", "hotelling", "roy"]),
        "raw_coefficients": c.frame(
            stacked(a_std / x_std[:, None], b_std / y_std[:, None]),
            columns=["set", *labels], index=variables),
        "standardized_coefficients": c.frame(stacked(a_std, b_std), columns=["set", *labels],
                                             index=variables),
        "loadings": c.frame(loadings, columns=["set", *labels, *c.numbered("Cross", s)],
                            index=variables),
        "redundancy": c.frame(
            torch.stack([x_share, x_share * rho2, y_share, y_share * rho2], dim=1),
            columns=["x_variance", "x_redundancy", "y_variance", "y_redundancy"], index=labels),
        "descriptives": c.frame([[kind, float(location), float(scale)] for kind, location, scale in zip(
            sets, mean, torch.cat((x_std, y_std)), strict=True)],
            columns=["set", "mean", "std_dev"], index=variables),
    }
    notes = []
    if bool((rho2 >= _PERFECT).any()):
        notes.append("A canonical correlation equals 1: a combination of the x variables is "
                     "an exact linear function of the y variables, so the tests that involve "
                     "it are undefined.")
    unidentified = [labels[k] for k in range(s) if float(rho[k]) <= 1e-10
                    or any(j != k and abs(float(rho[k] - rho[j])) <= 1e-10
                           for j in range(s))]
    if unidentified:
        notes.append("Zero or repeated roots do not uniquely identify individual coefficient "
                     f"axes ({', '.join(unidentified)}); the saved solution retains an arbitrary valid basis.")
    notes.append("Canonical-correlation probability levels retain Gaussian sampling assumptions; "
                 "coefficient SE/CI, survey or cluster inference are not provided.")
    extra = {"precision": "float64", "device": "cpu", **extra}
    return TableSet(
        tables, title=f"Canonical correlations of ({', '.join(x_names)}) with "
                      f"({', '.join(y_names)})",
        procedure="canon", n=n, n_missing=dropped, x=x_names, y=y_names, notes=notes,
        missing="listwise",
        variables=variables, canonical_pairs=s, state_schema="openecon.canon.v1",
        canonical_unidentified_axes=unidentified,
        canonical_axis_identification="zero/repeated roots leave arbitrary valid coefficient axes",
        canonical_saved_basis="the actual saved coefficient basis is used without refitting",
        sign_convention="largest absolute standardized x-coefficient of each pair is positive", **extra)


@c.procedure
def canon_matrix(values: Any, *, n: int, x: list[str], y: list[str],
                 matrix: str = "covariance", means: Any = None, sds: Any = None) -> TableSet:
    """CCA of a declared joint covariance/correlation matrix and common sample n.

    Matrix variable order must equal x followed by y. The matrix must describe
    the same complete, centred sample, with covariance divisor n-1; pairwise
    sample sizes, regularized/shrunk and uncentred moment matrices are outside
    this contract. Probability levels retain their Gaussian sample assumptions.
    Within-set blocks must be positive definite and numerically conditioned;
    the full joint matrix may be singular because a canonical root equals one.
    Actual training means are optional for fitting but required by canon_scores.
    Correlation inputs also require actual training sds for raw-row projection.
    Missing moments are never invented. Computation uses Cholesky whitening/SVD,
    on CPU float64, at most 64 joint variables and a declared live-buffer budget.
    """
    from .canon_options import matrix_canon
    return matrix_canon(values, n=n, x=x, y=y, matrix=matrix, means=means, sds=sds)


@c.procedure
def canon_scores(result: TableSet, data: Any):
    """Project raw rows onto the stored canonical coefficient basis, without refitting.

    Uses the actual saved training means and raw coefficients. Both sets are
    scored over jointly complete rows; missing rows retain missing scores and
    original index/order. Columns XCanon1..s and YCanon1..s have unit variance
    in the original fit sample (weighted variance for a frequency fit).
    For repeated/zero roots this projects the saved arbitrary valid basis and
    makes no claim that individual axes are uniquely identified. Dataset score
    sources are bounded and checked for content changes on complete replay.
    """
    from .canon_options import project
    return project(result, data)
