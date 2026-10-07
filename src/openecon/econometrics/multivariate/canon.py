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
def canon(data: Any, x: list[str], y: list[str], *, missing: str = "drop") -> TableSet:
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
    sample, _, dropped = c.select(data, [*x_names, *y_names], missing=missing)
    n = len(sample)
    p, q = len(x_names), len(y_names)
    if n < p + q + 2:
        raise AnalysisError("insufficient_observations", f"Canonical correlation of {p} and "
                            f"{q} variables needs at least {p + q + 2} complete observations, "
                            f"but {n} are available.")
    _, x_sscp, xc = c.moments(c.matrix(sample, x_names), x_names)
    _, y_sscp, yc = c.moments(c.matrix(sample, y_names), y_names)
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
    }
    notes = []
    if bool((rho2 >= _PERFECT).any()):
        notes.append("A canonical correlation equals 1: a combination of the x variables is "
                     "an exact linear function of the y variables, so the tests that involve "
                     "it are undefined.")
    return TableSet(
        tables, title=f"Canonical correlations of ({', '.join(x_names)}) with "
                      f"({', '.join(y_names)})",
        procedure="canon", n=n, n_missing=dropped, x=x_names, y=y_names, notes=notes,
        missing="listwise",
        sign_convention="largest absolute standardized x-coefficient of each pair is positive")
