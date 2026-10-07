"""Johansen's reduced-rank regression and the cointegrating-rank tests (Stata: vecrank).

The VEC form of a VAR(p) in K variables is

    D y_t = alpha (beta' y_(t-1) + mu + rho t) + sum_(i<p) Gamma_i D y_(t-i) + gamma + tau t + e_t.

Following the Methods and formulas of Stata's ``vec`` it is written
``Z0_t = alpha beta~' Z1_t + Psi Z2_t + e_t`` with, by trend specification,

    trend       Z1_t              Z2_t                       restriction
    none        y_(t-1)           D y lags                   mu = rho = gamma = tau = 0
    rconstant   (y_(t-1), 1)      D y lags                   rho = gamma = tau = 0
    constant    y_(t-1)           D y lags, 1                rho = tau = 0        (default)
    rtrend      (y_(t-1), t)      D y lags, 1                tau = 0
    trend       y_(t-1)           D y lags, t, 1             none

R0 and R1 are the residuals of Z0 and Z1 on Z2 and S_ij = R_i'R_j / T. The
eigenvalues 1 > l_1 >= ... >= l_K >= 0 solve |l S11 - S10 S00^-1 S01| = 0. With
the Cholesky factors S00 = L L' and S11 = C C' they are the squared singular
values of W = L^-1 S01 C^-T (the canonical correlations of R0 and R1), and
the eigenvectors V = C^-T (right singular vectors) satisfy V' S11 V = I; no
matrix is inverted explicitly. Then

    ln L(r) = -(T/2) [K {ln(2 pi) + 1} + ln|S00| + sum_(i<=r) ln(1 - l_i)]
    trace(r) = -T sum_(i>r) ln(1 - l_i),     max(r) = -T ln(1 - l_(r+1)).

The trend variable t is the 1-based position of the dependent observation in
the ordered sample.

Whenever the model has a constant (every specification except ``none``), the
lagged levels enter Z1 in deviations from their sample means. This is an exact
reparameterization: with a constant in Z2 the residuals R1 do not change at
all, and with the constant inside the cointegrating equations
(``rconstant``) Z1 is multiplied by a nonsingular matrix, which leaves the
eigenvalues unchanged and only moves the restricted constant
(``c = c_centered - beta' ybar``; ``vec`` maps it back). Series with a large
level therefore keep full accuracy, as they do in ``var``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import kernel_call, table
from openecon.econometrics.var import critical_values
from openecon.econometrics.var.common import (
    build_spec, check_design_size, integer, load_sample, variable_list,
)
from openecon.econometrics.var.kernels import lag_block, logdet, spd_factor
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import least_squares, symmetrize

TRENDS = ("none", "rconstant", "constant", "rtrend", "trend")
_LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class Johansen:
    t: int                       # observations used
    k: int
    p: int                       # lags of the underlying VAR
    trend: str
    z0: Tensor                   # [T, K] first differences
    z1: Tensor                   # [T, m1] lagged levels minus ``shift`` (+ restricted term)
    z2: Tensor                   # [T, m2] lagged differences (+ unrestricted terms)
    z1_labels: list[str]
    z2_labels: list[str]
    s00: Tensor
    s01: Tensor
    s11: Tensor
    eigenvalues: Tensor          # [K], descending
    vectors: Tensor              # [m1, K], V' S11 V = I
    logdet_s00: float
    trend_index: Tensor          # [T] value of t for each used observation
    shift: Tensor                # [K] level subtracted from y_(t-1) in Z1 (zero for trend none)

    @property
    def m1(self) -> int:
        return self.z1.shape[1]

    @property
    def m2(self) -> int:
        return self.z2.shape[1]

    def log_likelihood(self, rank: int) -> float:
        tail = float(torch.log1p(-self.eigenvalues[:rank]).sum())
        return -0.5 * self.t * (self.k * (_LOG_2PI + 1.0) + self.logdet_s00 + tail)

    def parameters(self, rank: int) -> int:
        """Free parameters: K m2 + (K + m1 - r) r (Johansen normalization removes r^2)."""
        return self.k * self.m2 + (self.k + self.m1 - rank) * rank

    def trace(self, rank: int) -> float:
        return float(-self.t * torch.log1p(-self.eigenvalues[rank:]).sum())

    def maximum(self, rank: int) -> float:
        return float(-self.t * torch.log1p(-self.eigenvalues[rank]))


@torch.no_grad()
def johansen(levels: Tensor, names: list[str], p: int, trend: str) -> Johansen:
    """Product moments, eigenvalues and eigenvectors of Johansen's reduced-rank regression."""
    n, k = levels.shape
    t = n - p
    if t < 2:
        raise KernelError("insufficient_observations", f"{p} lag(s) leave {max(t, 0)} "
                          "observation(s); the model needs a longer series.")
    difference = levels[1:] - levels[:-1]
    z0 = difference[p - 1:]
    shift = levels.mean(dim=0) if trend != "none" else levels.new_zeros(k)
    z1 = levels[p - 1:n - 1] - shift
    z2 = lag_block(difference, p - 1)
    index = torch.arange(p + 1, n + 1, dtype=torch.float64)
    ones = torch.ones((t, 1), dtype=torch.float64)
    z1_labels = list(names)
    z2_labels = [("LD." if i == 1 else f"L{i}D.") + name for name in names for i in range(1, p)]
    if trend == "rconstant":
        z1, z1_labels = torch.cat([z1, ones], dim=1), [*z1_labels, "_cons"]
    elif trend == "rtrend":
        z1, z1_labels = torch.cat([z1, index[:, None]], dim=1), [*z1_labels, "_trend"]
    if trend == "trend":
        z2, z2_labels = torch.cat([z2, index[:, None]], dim=1), [*z2_labels, "trend"]
    if trend in {"constant", "rtrend", "trend"}:
        z2, z2_labels = torch.cat([z2, ones], dim=1), [*z2_labels, "Intercept"]
    m1, m2 = z1.shape[1], z2.shape[1]
    if t <= m2 + m1 + k:
        raise KernelError("insufficient_observations", f"The model uses {m2 + m1} regressors "
                          f"per equation but only {t} observations remain after {p} lag(s). "
                          "Reduce the number of lags or variables.")
    both = torch.cat([z0, z1], dim=1)
    if m2:
        dependent = KernelError("collinear_system", "The lagged differences are linearly "
                                "dependent: an endogenous variable is an exact combination of "
                                "the others. Remove it or reduce the number of lags.")
        try:
            fit = least_squares(z2, both, drop_collinear=True)
        except KernelError as exc:
            if exc.code != "singular_design":
                raise
            raise dependent from exc
        if fit.omitted:
            raise dependent
        both = fit.resid
    r0, r1 = both[:, :k], both[:, k:]
    s00, s01, s11 = symmetrize(r0.T @ r0 / t), r0.T @ r1 / t, symmetrize(r1.T @ r1 / t)
    lower = spd_factor(s00, what="covariance matrix of the differences")
    chol = spd_factor(s11, code="collinear_system", what="moment matrix of the lagged levels")
    w = torch.linalg.solve_triangular(lower, s01, upper=False)
    w = torch.linalg.solve_triangular(chol, w.T, upper=False).T
    try:
        _, singular, vh = torch.linalg.svd(w, full_matrices=False)
    except RuntimeError as exc:
        raise KernelError("numerical_failure", f"The eigenvalue problem failed: {exc}") from exc
    eigenvalues = singular[:k].square()
    if not bool(torch.isfinite(eigenvalues).all()) or float(eigenvalues[0]) >= 1.0 - 1e-12:
        raise KernelError("collinear_system", "A canonical correlation between the differences "
                          "and the lagged levels equals one: the system is degenerate. Check for "
                          "variables that are exact functions of the others.")
    vectors = torch.linalg.solve_triangular(chol.T, vh[:k].T, upper=True)
    return Johansen(t, k, p, trend, z0, z1, z2, z1_labels, z2_labels, s00, s01, s11,
                    eigenvalues, vectors, logdet(lower), index, shift)


def rank_rows(result: Johansen, max_rank: int | None = None) -> list[dict[str, Any]]:
    """One record per rank 0..K: parameters, log likelihood, statistics and critical values."""
    k, t = result.k, result.t
    top = k if max_rank is None else min(k, max_rank)
    rows = []
    for rank in range(top + 1):
        ll = result.log_likelihood(rank)
        count = result.parameters(rank)
        row: dict[str, Any] = {
            "rank": rank, "parms": count, "ll": ll,
            "eigenvalue": float(result.eigenvalues[rank - 1]) if rank else None,
            "trace": None, "trace_cv5": None, "trace_cv1": None,
            "max": None, "max_cv5": None, "max_cv1": None,
            "sbic": (-2.0 * ll + math.log(t) * count) / t,
            "hqic": (-2.0 * ll + 2.0 * math.log(math.log(t)) * count) / t,
            "aic": (-2.0 * ll + 2.0 * count) / t,
        }
        if rank < k:
            row["trace"], row["max"] = result.trace(rank), result.maximum(rank)
            for statistic in ("trace", "max"):
                for level in (5, 1):
                    row[f"{statistic}_cv{level}"] = critical_values.lookup(
                        result.trend, k - rank, statistic, level)
        rows.append(row)
    return rows


def select_rank(rows: list[dict[str, Any]], statistic: str, level: int, k: int) -> int | None:
    """Johansen's sequential rule: the first rank whose null hypothesis is not rejected."""
    for row in rows:
        if row[statistic] is None:
            continue
        critical = row[f"{statistic}_cv{level}"]
        if critical is None:
            return None
        if row[statistic] <= critical:
            return row["rank"]
    return k if rows and rows[-1]["rank"] >= k - 1 else None


def vecrank(*, data: Any, y: Any, time: str | None = None, lags: int = 2,
            trend: str = "constant", max_rank: int | None = None,
            missing: str = "raise") -> Any:
    """Johansen tests for the cointegrating rank (Stata ``vecrank``; EViews Johansen test).

    The underlying model is a VAR with ``lags`` lags in the K variables ``y``,
    written as a VEC model with ``lags - 1`` lagged differences. For each rank
    r = 0..K the table reports the number of free parameters, the log
    likelihood, the eigenvalue l_r and

        trace(r) = -T sum_(i>r) ln(1 - l_i)    H0: rank <= r   vs   rank > r
        max(r)   = -T ln(1 - l_(r+1))          H0: rank  = r   vs   rank = r + 1

    with the 5% and 1% critical values of Osterwald-Lenum (1992), and the
    information criteria SBIC, HQIC, AIC of each rank (per observation, as
    Stata's ``ic`` option). ``attrs["selected_rank"]`` is Johansen's estimate:
    the first r whose trace statistic does not exceed its 5% critical value
    (``selected_rank_1pct`` uses the 1% values, ``selected_rank_max`` the
    maximum-eigenvalue statistic at 5%).

    Parameters
    ----------
    data : table; ``y`` : list of the K variables (in levels).
    time : optional time column (rows are sorted by it; no gaps allowed).
    lags : lags of the underlying VAR (default 2).
    trend : deterministic specification, Stata's ``trend()``:
        ``"none"``, ``"rconstant"`` (constant restricted to the cointegrating
        equations), ``"constant"`` (unrestricted constant, the default),
        ``"rtrend"`` (unrestricted constant, trend restricted to the cointegrating
        equations), ``"trend"`` (unrestricted constant and trend).
    max_rank : report ranks 0..max_rank only (default: all).
    missing : ``"raise"`` or ``"drop"`` (dropped rows only at the ends of the series).

    Critical values exist for K - r <= 11 (all five trend specifications).
    ``attrs["critical_values_flag"]`` is ``None`` unless a reported critical
    value could be confirmed against one published transcription only
    (``trend="none"`` with K - r >= 7, ``trend="trend"`` with K - r >= 6); it
    then says so. P-values are not reported: the limiting distributions are
    non-standard and OpenEconometrics does not ship the MacKinnon-Haug-Michelis
    response surfaces. See ``docs/econometrics/var.md`` for the provenance of
    the tables.

    Stata: ``vecrank y1 y2 y3, lags(2) trend(constant) max ic levela``.

    Example
    -------
    >>> oe.vecrank(data=frame, y=["y1", "y2", "y3"], lags=2)
    """
    names = variable_list(y)
    lags = integer(lags)
    if max_rank is not None:
        max_rank = integer(max_rank)
        if not isinstance(max_rank, int) or isinstance(max_rank, bool) or max_rank < 0:
            raise AnalysisError("invalid_option", "max_rank must be a non-negative integer.")
    spec = build_spec("vec", outcome=names[0], time=time, missing=missing,
                      columns={"system": names}, options={"lags": lags, "trend": trend})
    sample = load_sample(spec, data)
    # The validated options: lags=None or trend=None mean the defaults.
    lags, trend = int(sample.frame.option("lags")), sample.frame.option("trend")
    k = len(names)
    check_design_size(sample.frame.n, k * lags + 2 + 2 * k)
    result = kernel_call(johansen, sample.levels, names, lags, trend)
    rows = rank_rows(result, max_rank)
    columns = ["rank", "parms", "ll", "eigenvalue", "trace", "trace_cv5", "trace_cv1", "max",
               "max_cv5", "max_cv1", "sbic", "hqic", "aic"]
    return table(
        [[row[name] for name in columns] for row in rows], columns=columns,
        selected_rank=select_rank(rows, "trace", 5, k),
        selected_rank_1pct=select_rank(rows, "trace", 1, k),
        selected_rank_max=select_rank(rows, "max", 5, k),
        n_obs=result.t, lags=lags, trend=trend, variables=names,
        eigenvalues=result.eigenvalues,
        critical_values=critical_values.SOURCE, p_values=critical_values.P_VALUE_NOTE,
        critical_values_flag=critical_values.provenance_note(trend, k),
        warnings=list(sample.frame.warnings))
