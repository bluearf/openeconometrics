"""Panel bookkeeping shared by xtgls and xtpcse: layout, AR(1) estimation, Prais-Winsten.

The sample is sorted by panel and time (``ModelFrame.sort_panel``). Panel
``i = 1..m`` has ``T_i`` observations; the layout records the first row of
every panel, a dense code for every distinct period and whether the panels are
balanced (every panel observed in every period) and gap free.

Autocorrelation (Stata's ``rhotype()``; ``e_t`` the residuals of one panel,
sums over the consecutive pairs ``(e_{t-1}, e_t)`` unless noted, ``k`` the
number of regressors)

    regress   sum e_t e_{t-1} / sum e_{t-1}^2       (default: regression of e_t on e_{t-1})
    freg      sum e_t e_{t-1} / sum e_t^2           (regression of e_{t-1} on e_t)
    tscorr    sum e_t e_{t-1} / sum_all e_t^2       (autocorrelation)
    dw        1 - DW/2,  DW = sum (e_t - e_{t-1})^2 / sum_all e_t^2
    theil     tscorr (T_i - k) / T_i
    nagar     (dw T_i^2 + k^2) / (T_i^2 - k^2)

A common AR(1) coefficient is the average of the panel coefficients weighted
by ``T_i - 1`` (``T_i`` with ``np1``), which is the plain mean for balanced
panels.

Prais-Winsten: ``z*_i1 = sqrt(1 - rho_i^2) z_i1`` and ``z*_it = z_it - rho_i z_i,t-1``
for every column of ``[y X]`` (the constant included), within each panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from openecon.resources import plan_workspace, tensor_bytes

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame
from openecon.engines.covariance import group_sums

RHOTYPES = ("regress", "freg", "tscorr", "dw", "theil", "nagar")
# Largest dense [periods, panels, columns] grid built for panel-corrected meats.
# Kept as a compatibility constant; executable allocation checks use the
# configurable whole-operation plan below, including quadratic group buffers.
DENSE_LIMIT = 50_000_000


def panel_covariance_plan(layout, width, *, correlated=True, resident_bytes=0, budget_bytes=None):
    n, groups, periods = len(layout.codes), layout.m, layout.periods
    buffers = {"model_input_and_sample_indices": resident_bytes,
               "design_residual_and_qr_copies": tensor_bytes((n, width + 1), itemsize=64),
               "parameter_factors": tensor_bytes((width, width), itemsize=64),
               "panel_indices_and_row_scratch": tensor_bytes((n,), itemsize=64)}
    if correlated:
        buffers["dense_panel_grids_and_products"] = tensor_bytes((periods, groups, width + 2), itemsize=32)
        buffers["group_covariance_counts_and_factor_scratch"] = tensor_bytes((groups, groups), itemsize=64)
    return plan_workspace("panel covariance", buffers, budget_bytes=budget_bytes)


@dataclass
class PanelLayout:
    codes: Tensor          # panel code per row, [n]
    m: int                 # panels
    tcodes: Tensor         # dense period code per row, [n]
    periods: int           # distinct periods
    sizes: Tensor          # T_i, [m] (float64)
    first: Tensor          # first row of its panel, [n] bool
    balanced: bool
    consecutive: bool      # no gaps inside any panel

    @property
    def pairs(self) -> Tensor:
        """Rows that have a predecessor in the same panel."""
        return (~self.first).nonzero().flatten()


def panel_layout(frame: ModelFrame) -> PanelLayout:
    """Layout of a sample already sorted by panel and time."""
    codes, m = frame.codes(frame.spec.panel)
    period = frame.time_index()
    n = frame.n
    first = torch.ones(n, dtype=torch.bool)
    if n > 1:
        first[1:] = codes[1:] != codes[:-1]
    steps = period[1:] - period[:-1] if n > 1 else torch.zeros(0, dtype=torch.int64)
    consecutive = bool((steps[~first[1:]] == 1).all()) if n > 1 else True
    distinct, tcodes = torch.unique(period, return_inverse=True)
    periods = len(distinct)
    sizes = torch.bincount(codes, minlength=m).to(torch.float64)
    balanced = bool((sizes == periods).all())
    layout = PanelLayout(codes, m, tcodes, periods, sizes, first, balanced, consecutive)
    if frame.spec.estimator == "xtgls" and frame.option("panels") == "correlated":
        plan = panel_covariance_plan(layout, frame.design_width(),
                                     resident_bytes=frame.resource_input_bytes,
                                     budget_bytes=frame.resource_budget_bytes)
        frame.resource_plans.append(plan.record())
    return layout


def panel_rho(resid: Tensor, layout: PanelLayout, rhotype: str, k: int) -> Tensor:
    """Panel-specific AR(1) coefficients of the residuals by ``rhotype``."""
    pairs = layout.pairs
    current, lagged = resid[pairs], resid[pairs - 1]
    codes = layout.codes[pairs]
    m = layout.m

    def total(values: Tensor, index: Tensor = codes) -> Tensor:
        return group_sums(values[:, None], index, m)[:, 0]

    cross = total(current * lagged)
    every = total(resid.square(), layout.codes)
    sizes = layout.sizes
    if rhotype == "regress":
        rho = cross / total(lagged.square())
    elif rhotype == "freg":
        rho = cross / total(current.square())
    elif rhotype in {"tscorr", "theil"}:
        rho = cross / every
        if rhotype == "theil":
            rho = rho * (sizes - k) / sizes
    else:
        dw = 1 - total((current - lagged).square()) / every / 2
        rho = dw if rhotype == "dw" else (dw * sizes.square() + k ** 2) / (sizes.square() - k ** 2)
    return rho


def common_rho(rho: Tensor, layout: PanelLayout, np1: bool = False) -> Tensor:
    """Weighted mean of the panel coefficients; panels without a consecutive pair carry no
    coefficient and are left out (also with ``np1``, which would otherwise give them weight 1).
    """
    weights = layout.sizes if np1 else layout.sizes - 1
    valid = (weights > 0) & (layout.sizes > 1)
    value = (rho[valid] * weights[valid]).sum() / weights[valid].sum()
    return value.expand(layout.m).clone()


def estimate_rho(frame: ModelFrame, resid: Tensor, layout: PanelLayout, corr: str, rhotype: str,
                 k: int, np1: bool = False) -> Tensor:
    """AR(1) coefficient of every panel (common for ``ar1``), validated to lie in (-1, 1)."""
    if corr == "psar1" and bool((layout.sizes < 2).any()):
        raise AnalysisError("insufficient_observations", "Panel-specific AR(1) coefficients need "
                            "at least two observations in every panel.")
    if bool((layout.sizes < 2).all()):
        raise AnalysisError("insufficient_observations", "An AR(1) coefficient needs panels with "
                            "at least two consecutive observations.")
    rho = panel_rho(resid, layout, rhotype, k)
    if corr == "ar1":
        rho = common_rho(rho, layout, np1)
    if not bool(torch.isfinite(rho).all()) or bool((rho.abs() >= 1).any()):
        raise AnalysisError("invalid_rho", f"The estimated AR(1) coefficient is not inside "
                            f"(-1, 1) (rhotype='{rhotype}'); the Prais-Winsten transformation is "
                            "undefined. Use rhotype='tscorr' or 'dw' (bounded estimators), or a "
                            "common coefficient (ar1).")
    return rho


def prais_winsten(values: Tensor, rho: Tensor, layout: PanelLayout) -> Tensor:
    """Prais-Winsten transformation of ``values`` ([n] or [n, c]) within panels."""
    block = values if values.ndim == 2 else values[:, None]
    rows = rho[layout.codes]
    out = block.clone()
    pairs = layout.pairs
    out[pairs] = block[pairs] - rows[pairs, None] * block[pairs - 1]
    first = layout.first
    out[first] = block[first] * (1 - rows[first].square()).sqrt()[:, None]
    return out if values.ndim == 2 else out[:, 0]


def require_consecutive(layout: PanelLayout) -> None:
    if not layout.consecutive:
        raise AnalysisError("time_gaps", "AR(1) disturbances need consecutive periods within "
                            "every panel; the time variable has gaps. Fill or drop the gaps, or "
                            "use corr='independent'.")


def dense(values: Tensor, layout: PanelLayout) -> Tensor:
    """Values on the [periods, panels, ...] grid, zero where a panel is unobserved."""
    shape = (layout.periods, layout.m, *values.shape[1:])
    grid = torch.zeros(shape, dtype=torch.float64)
    grid[layout.tcodes, layout.codes] = values
    return grid


def observed(layout: PanelLayout) -> Tensor:
    grid = torch.zeros((layout.periods, layout.m), dtype=torch.float64)
    grid[layout.tcodes, layout.codes] = 1.0
    return grid


def panel_summary(layout: PanelLayout) -> dict[str, float]:
    return {"n_groups": layout.m, "n_periods": layout.periods,
            "obs_per_group_min": float(layout.sizes.min()),
            "obs_per_group_avg": float(layout.sizes.mean()),
            "obs_per_group_max": float(layout.sizes.max())}
