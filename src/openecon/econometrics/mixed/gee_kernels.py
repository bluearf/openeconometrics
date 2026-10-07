"""Working correlations of generalized estimating equations (tensors only).

The panels are the blocks of a block-diagonal working correlation R. A GEE
step needs R^-1 applied to a few columns per panel (the standardized design
and the Pearson residuals); each structure does that without forming R_i in a
Python loop over panels. Rows are sorted by panel and, within a panel, by
period; ``position`` is the 0-based rank of a row within its panel, and as in
Stata's xtgee the working correlation of a panel with n_i observations is the
upper-left n_i-by-n_i block of one max(n_i)-by-max(n_i) matrix.

independent    R = I.
exchangeable   R_i = (1 - a) I + a 11'  =>  R_i^-1 v = [v - c_i 1 (1'v)] / (1 - a),
               c_i = a / (1 + (n_i - 1) a)                        (one index_add_)
ar1            R_i,ts = a^|t - s|  =>  R_i^-1 is tridiagonal:
               (R^-1 v)_t = [(1 + a^2) v_t - a (v_t-1 + v_t+1)] / (1 - a^2) inside a
               panel, [v_t - a v_t+-1] / (1 - a^2) at its first and last period.
stationary(m), nonstationary(m), unstructured
               R_i is the leading n_i-by-n_i block of a T-by-T matrix. Panels are grouped
               by size; for each distinct size the inverse of the leading block is formed
               once and applied to all panels of that size by one batched product on a
               [panels, size, columns] view of their rows (no padding to T).

Moment estimators (Stata's xtgee Methods and formulas, "Correlation
structures"), with Pearson residuals r_it = (y_it - mu_it) / sqrt(V(mu_it)):

exchangeable   a   = [sum_i sum_{t != s} r_it r_is / sum_i n_i (n_i - 1)]
                     / [sum_i sum_t r_it^2 / sum_i n_i]
ar1 / stationary(m)
               a_k = [sum_i (1/n_i) sum_t r_it r_i,t+k] / [sum_i (1/n_i) sum_t r_it^2],
               k = 1..m (Yule-Walker form: lag-k sums divided by n_i, not n_i - k)
nonstationary(m), unstructured
               a_ts = [sum_i r_it r_is / N_ts] / [(1/G) sum_i (1/n_i) sum_t r_it^2],
               N_ts = number of panels observing positions t and s
               (|t - s| <= m for nonstationary, all pairs for unstructured).

None of them depends on the ``nmp`` divisor of the scale parameter.
"""

from __future__ import annotations

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.covariance import group_sums
from openecon.resources import plan_workspace, tensor_bytes

STRUCTURES = ("exchangeable", "independent", "ar1", "unstructured", "stationary",
              "nonstationary")
TIMED = ("ar1", "unstructured", "stationary", "nonstationary")
PATTERNED = ("unstructured", "stationary", "nonstationary")


def gee_workspace_plan(kind, layout, width=1, *, resident_bytes=0, budget_bytes=None):
    """Count the full correlation and both generations of cached inverses."""
    n, groups, periods = len(layout.codes), layout.n_panels, layout.periods
    buffers = {"model_input_and_sample_indices": resident_bytes,
               "design_scoring_and_solve_buffers": tensor_bytes((n, width + 1), itemsize=64),
               "row_and_layout_scratch": tensor_bytes((n,), itemsize=128),
               "group_scores": tensor_bytes((groups, width), itemsize=24),
               "parameter_factors": tensor_bytes((width, width), itemsize=64)}
    if kind in PATTERNED:
        buffers["correlation_moments_masks_cholesky_and_indices"] = tensor_bytes((periods, periods), itemsize=128)
        # Old inverses remain live while a new list is computed at each step.
        buffers["cached_inverse_generations"] = sum(tensor_bytes((size, size), itemsize=16)
                                                     for size, _ in layout.size_groups)
    return plan_workspace("GEE " + kind, buffers, budget_bytes=budget_bytes)


class PanelLayout:
    """Panel codes (rows sorted by panel, then period), sizes and positions within panels."""

    def __init__(self, codes: Tensor, n_panels: int):
        self.codes, self.n_panels = codes, n_panels
        n = len(codes)
        self.sizes = torch.bincount(codes, minlength=n_panels)
        same = torch.zeros(n, dtype=torch.bool)
        if n > 1:
            same[1:] = codes[1:] == codes[:-1]
        if n and int((~same).sum()) != n_panels:
            raise KernelError("invalid_panel_order", "The rows of a panel must be contiguous.")
        self.has_previous = same                                    # row t-1 in the same panel
        self.has_next = torch.zeros(n, dtype=torch.bool)
        self.has_next[:-1] = same[1:]
        start = torch.zeros(n_panels, dtype=torch.int64).scatter_reduce(
            0, codes, torch.arange(n), "amin", include_self=False)
        self.positions = torch.arange(n) - start[codes]               # rank within the panel
        self.periods = int(self.sizes.max()) if n else 0
        self.inverse_size = 1.0 / self.sizes.to(torch.float64)        # 1/n_i per panel
        # Rows of the panels of each distinct size, in panel-then-position order.
        row_size = self.sizes[codes]
        self.size_groups: list[tuple[int, Tensor]] = []
        for size in torch.unique(self.sizes).tolist():
            rows = (row_size == size).nonzero().flatten()
            self.size_groups.append((int(size), rows))

    def lagged(self, values: Tensor, lag: int) -> Tensor:
        """sum over pairs (t, t + lag) of the same panel of v_t v_t+lag, per panel [G]."""
        n = len(values)
        if lag >= n:
            return torch.zeros(self.n_panels, dtype=torch.float64)
        same = self.codes[lag:] == self.codes[:-lag]
        products = values[:-lag] * values[lag:] * same.to(values.dtype)
        return group_sums(products, self.codes[:-lag], self.n_panels)


class WorkingCorrelation:
    """R(alpha) of one structure: estimation of alpha and application of R^-1."""

    def __init__(self, kind: str, layout: PanelLayout, order: int = 1, *, width: int = 1,
                 resident_bytes: int = 0, budget_bytes: int | None = None):
        if kind not in STRUCTURES:
            raise KernelError("invalid_option", f"Unknown working correlation '{kind}'.")
        self.resource_plan = gee_workspace_plan(kind, layout, width, resident_bytes=resident_bytes,
                                                budget_bytes=budget_bytes)
        self.kind, self.layout, self.order = kind, layout, order
        self.alpha: Tensor = torch.zeros(0, dtype=torch.float64)
        self.matrix: Tensor | None = None             # T-by-T for the patterned structures
        self._inverses: list[tuple[Tensor, Tensor, int]] = []

    # ---- estimation ---------------------------------------------------------------

    def estimate(self, resid: Tensor) -> None:
        """Set alpha from the Pearson residuals (see the module notes)."""
        layout = self.layout
        if self.kind == "independent":
            return
        if self.kind == "exchangeable":
            sums = group_sums(resid, layout.codes, layout.n_panels)
            squares = resid.square().sum()
            pairs = (layout.sizes * (layout.sizes - 1)).sum()
            if int(pairs) == 0:
                raise KernelError("insufficient_panel_length", "The exchangeable correlation "
                                  "needs panels with at least two observations.")
            numerator = (sums.square().sum() - squares) / pairs
            self.alpha = (numerator / (squares / len(resid))).reshape(1)
            return
        weights = layout.inverse_size
        variance = (group_sums(resid.square(), layout.codes, layout.n_panels) * weights).sum()
        if not float(variance) > 0:
            raise KernelError("perfect_fit", "The Pearson residuals are all zero; the working "
                              "correlation is not identified.")
        if self.kind in {"ar1", "stationary"}:
            lags = 1 if self.kind == "ar1" else self.order
            values = [(layout.lagged(resid, lag) * weights).sum() / variance
                      for lag in range(1, lags + 1)]
            self.alpha = torch.stack(values)
            if self.kind == "stationary":
                self.set_alpha(self.alpha)
            return
        periods = layout.periods
        cross = torch.zeros((periods, periods), dtype=torch.float64)
        counts = torch.zeros((periods, periods), dtype=torch.float64)
        for size, rows in layout.size_groups:
            block = resid[rows].reshape(-1, size)
            cross[:size, :size] += block.T @ block
            counts[:size, :size] += block.shape[0]
        scale = variance / layout.n_panels
        lags = (torch.arange(periods)[:, None] - torch.arange(periods)[None, :]).abs()
        keep = (lags > 0) & ((lags <= self.order) if self.kind == "nonstationary" else True)
        matrix = torch.where(keep, cross / counts.clamp_min(1) / scale, torch.zeros_like(cross))
        matrix = matrix + torch.eye(periods, dtype=torch.float64)
        rows_, cols = torch.triu_indices(periods, periods, 1)
        self.alpha = matrix[rows_, cols][keep[rows_, cols]]
        self._set_matrix(matrix)

    def set_alpha(self, alpha: Tensor) -> None:
        """Fix alpha (exchangeable / ar1 / stationary: the parameters; others: R itself)."""
        if self.kind in {"exchangeable", "ar1"}:
            self.alpha = alpha.reshape(1)
        elif self.kind == "stationary":
            self.alpha = alpha.reshape(-1)
            periods = self.layout.periods
            lags = (torch.arange(periods)[:, None] - torch.arange(periods)[None, :]).abs()
            matrix = torch.eye(periods, dtype=torch.float64)
            for lag, value in enumerate(self.alpha, start=1):
                matrix = torch.where(lags == lag, value, matrix)
            self._set_matrix(matrix)
        elif self.kind != "independent":
            self._set_matrix(alpha)

    def _set_matrix(self, matrix: Tensor) -> None:
        """Store R and the inverses of its leading blocks, one per distinct panel size."""
        chol, info = torch.linalg.cholesky_ex(matrix)
        if int(info) != 0:
            raise KernelError("working_correlation_not_pd", "The estimated working "
                              "correlation matrix is not positive definite; choose a "
                              "simpler structure (exchangeable or ar1) or a lower order.")
        self.matrix = matrix
        # The leading block of R has the leading block of its Cholesky factor as factor.
        self._inverses = [(rows, torch.cholesky_inverse(chol[:size, :size]), size)
                          for size, rows in self.layout.size_groups]

    # ---- R^-1 v -----------------------------------------------------------------------

    def check(self) -> None:
        """Refuse an alpha outside the region where R is positive definite."""
        if self.kind == "exchangeable":
            alpha = float(self.alpha[0])
            longest = int(self.layout.sizes.max())
            if not -1.0 / max(longest - 1, 1) < alpha < 1.0:
                raise KernelError("working_correlation_not_pd", f"The exchangeable correlation "
                                  f"{alpha:.4g} is outside (-1/(T-1), 1); the working "
                                  "correlation is not positive definite.")
        elif self.kind == "ar1" and not -1.0 < float(self.alpha[0]) < 1.0:
            raise KernelError("working_correlation_not_pd", "The AR(1) correlation is outside "
                              "(-1, 1).")

    def solve(self, values: Tensor) -> Tensor:
        """Block-diagonal R^-1 applied to the columns of values [N, m]."""
        layout = self.layout
        if self.kind == "independent":
            return values
        if self.kind == "exchangeable":
            alpha = self.alpha[0]
            sizes = layout.sizes.to(torch.float64)
            c = alpha / (1 + (sizes - 1) * alpha)
            totals = group_sums(values, layout.codes, layout.n_panels)
            return (values - (c[:, None] * totals)[layout.codes]) / (1 - alpha)
        if self.kind == "ar1":
            alpha = self.alpha[0]
            previous = torch.zeros_like(values)
            following = torch.zeros_like(values)
            previous[1:] = values[:-1]
            following[:-1] = values[1:]
            previous[~layout.has_previous] = 0
            following[~layout.has_next] = 0
            interior = (layout.has_previous & layout.has_next).to(torch.float64)[:, None]
            edge = (layout.has_previous | layout.has_next).to(torch.float64)[:, None]
            diagonal = 1 + alpha.square() * interior
            out = (diagonal * values - alpha * (previous + following)) / (1 - alpha.square())
            # A single-observation panel has R = 1.
            return torch.where(edge > 0, out, values)
        out = torch.empty_like(values)
        width = values.shape[1]
        for rows, inverse, size in self._inverses:
            block = values[rows].reshape(-1, size, width)
            out[rows] = (inverse @ block).reshape(-1, width)
        return out

    def full_matrix(self, size: int) -> Tensor:
        """The working correlation of a panel with ``size`` consecutive observations."""
        if self.matrix is not None:
            return self.matrix[:size, :size]
        eye = torch.eye(size, dtype=torch.float64)
        if self.kind == "independent":
            return eye
        alpha = float(self.alpha[0])
        if self.kind == "exchangeable":
            return (1 - alpha) * eye + alpha * torch.ones_like(eye)
        lags = (torch.arange(size)[:, None] - torch.arange(size)[None, :]).abs()
        return torch.pow(torch.tensor(alpha, dtype=torch.float64), lags.to(torch.float64))
