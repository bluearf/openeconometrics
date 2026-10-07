"""Shared layer of the linear system estimators (sureg, mvreg, reg3).

Equations
---------
A system is a list of equations ``y_i = X_i b_i + e_i`` (``i = 1..M``), given
as ``{"y": outcome, "x": [regressors], "name": label, "constant": bool}``.
The label defaults to the outcome; terms are reported as ``label:term`` (for
example ``price:weight`` and ``price:Intercept``) with the label as the
coefficient's equation.

Compressed data
---------------
Every system estimator needs the data only through cross products of the
outcomes and regressors. All distinct columns are gathered once in
``W = [Z, other columns]`` (the instruments ``Z`` first when there are any;
the constant is always column 0) and reduced by ONE Householder QR,
``sqrt(w) W = Q R``. Then ``W'diag(w)W = R'R`` and every residual
``e = W a`` has ``e'diag(w)e = ||R a||^2``: equation-by-equation OLS, the
residual covariance, the stacked GLS of SUR and the projections of 3SLS all
run on the ``p x p`` factor ``R`` instead of the ``N`` rows, with the accuracy
of a QR least-squares fit on the data (``R`` is the exact factor of a matrix
within rounding of ``W``). Because the leading columns span ``Z``, the
projection on the instruments is the leading block of rows:
``||P_Z W a||^2 = ||R[:L] a||^2``.

Centring
--------
With the constant in column 0, ``R[0, j] / R[0, 0]`` is the weighted mean of
column ``j`` and setting ``R[0, j] = 0`` gives the compressed centred column.
Equations with a constant are fitted on centred regressors, which keeps the
algebra well conditioned when a regressor has a large level (a year, a price
index); the constants are mapped back exactly, ``b_0 = b_0c - m'b``.

A column that is constant in the estimation sample would centre to the QR
rounding residue of ``R[1:, j]`` (about ``n eps`` relative), which the
left-to-right screen cannot tell from a genuine regressor because it judges
every column against its own length. Such columns are therefore detected
exactly on the data (``max == min``) and centre to exact zeros, so the screen
omits them as collinear with the constant (Stata sweeps the constant first).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame
from openecon.engines.distributions import chi2_sf
from openecon.resources import tensor_bytes

# A residual sum of squares at or below this fraction of sum w y^2 is rounding: exact fit.
EXACT_FIT = 1e-28
_EQUATION_KEYS = {"y", "x", "name", "constant"}


@dataclass
class Equation:
    name: str
    outcome: str
    regressors: list[str]
    constant: bool


def _names(value: Any, what: str, position: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, (list, tuple)) \
            or not all(isinstance(item, str) and item.strip() for item in value):
        raise AnalysisError("invalid_spec", f"Equation {position}: {what} must be a list of "
                            f"column names, for example {what}=['x1', 'x2'].")
    if len(set(value)) != len(value):
        raise AnalysisError("invalid_spec", f"Equation {position}: {what} lists a column twice.")
    return list(value)


def parse_equations(value: Any, *, constant: bool = True, minimum: int = 2) -> list[Equation]:
    """Validate the ``equations`` option."""
    if not isinstance(value, (list, tuple)) or len(value) < minimum:
        raise AnalysisError("invalid_spec", f"equations must be a list of at least {minimum} "
                            "equations such as {'y': 'y1', 'x': ['x1', 'x2']}.")
    equations: list[Equation] = []
    for position, entry in enumerate(value, start=1):
        if not isinstance(entry, dict) or "y" not in entry or set(entry) - _EQUATION_KEYS:
            raise AnalysisError("invalid_spec", f"Equation {position} must be a mapping with "
                                "keys 'y', 'x' and optionally 'name' and 'constant'.")
        outcome = entry["y"]
        if not isinstance(outcome, str) or not outcome.strip():
            raise AnalysisError("invalid_spec", f"Equation {position}: 'y' must be a column name.")
        regressors = _names(entry.get("x"), "x", position)
        if outcome in regressors:
            raise AnalysisError("invalid_spec", f"Equation {position}: the outcome '{outcome}' "
                                "is also one of its regressors.")
        name = entry.get("name", outcome)
        if not isinstance(name, str) or not name.strip() or ":" in name:
            raise AnalysisError("invalid_spec", f"Equation {position}: 'name' must be a "
                                "non-empty label without ':'.")
        has_constant = entry.get("constant", constant)
        if not isinstance(has_constant, bool):
            raise AnalysisError("invalid_spec", f"Equation {position}: 'constant' must be "
                                "True or False.")
        if not regressors and not has_constant:
            raise AnalysisError("invalid_spec", f"Equation {position} has neither regressors "
                                "nor a constant.")
        equations.append(Equation(name, outcome, regressors, has_constant))
    labels = [equation.name for equation in equations]
    if len(set(labels)) != len(labels):
        raise AnalysisError("invalid_spec", "Equation labels must be unique; give repeated "
                            "outcomes distinct 'name' entries.")
    return equations


def system_columns(equations: Sequence[dict[str, Any]], *extra: Sequence[str] | None) -> list[str]:
    """Every column of a raw equations list (for the role ``system``), in order."""
    names: list[str] = []
    for entry in equations:
        if isinstance(entry, dict):
            if isinstance(entry.get("y"), str):
                names.append(entry["y"])
            x = entry.get("x")
            if isinstance(x, (list, tuple)):
                names.extend(item for item in x if isinstance(item, str))
    for block in extra:
        names.extend(block or [])
    return list(dict.fromkeys(names))


def check_system_role(frame: ModelFrame, columns: Sequence[str]) -> None:
    listed = set(frame.role("system")) | {frame.spec.outcome}
    absent = [name for name in columns if name not in listed]
    if absent:
        raise AnalysisError("invalid_spec", f"Columns used by the equations must be listed in "
                            f"the column role 'system': {', '.join(absent)}.")


def record_omitted(frame: ModelFrame, names: Sequence[str], reason: str) -> None:
    """Record omitted terms exactly as ``ModelFrame.drop_collinear`` does."""
    if names:
        frame.notes.setdefault("omitted_terms", []).extend(names)
        frame.warn(f"Omitted because of {reason}: {', '.join(names)}.")


def estimation_weights(frame: ModelFrame) -> tuple[Tensor | None, int]:
    """aweights rescaled to sum to the rows (N = rows); fweights replicate rows (N = sum)."""
    weights = frame.weights()
    if weights is None:
        return None, frame.n
    if frame.spec.weight_type == "fweight":
        return weights, int(round(float(weights.sum())))
    return weights * (frame.n / float(weights.sum())), frame.n


# ---- compressed system data ------------------------------------------------------------


@dataclass
class SystemData:
    """The R factor of ``sqrt(w) W`` and the bookkeeping of its columns."""

    r: Tensor                          # [min(n, p), p] upper triangular (trapezoidal)
    labels: list[str]                  # one label per column of W
    blocks: dict[str, list[int]]       # variable -> its columns (categoricals: several)
    terms: dict[str, list[str]]        # variable -> design term names
    constant: bool                     # column 0 is the constant
    instruments: int                   # leading columns spanning Z (0: none)
    nobs: int
    rows: int
    weights: Tensor | None
    raw: Tensor                        # unweighted W, [n, p] (fitted values only)
    categories: dict[str, Any] = field(default_factory=dict)
    flat: set[int] = field(default_factory=set)   # columns constant in the sample

    def columns(self, variables: Sequence[str], constant: bool) -> tuple[list[int], list[str]]:
        index = [0] if constant else []
        terms = ["Intercept"] if constant else []
        for name in variables:
            index.extend(self.blocks[name])
            terms.extend(self.terms[name])
        return index, terms


def _variable_block(frame: ModelFrame, name: str, categorical: set[str]) -> tuple[Tensor, list[str],
                                                                                 dict[str, Any]]:
    if name in categorical:
        design = frame.design([name], intercept=False, categorical=[name])
        return design.x, design.terms, design.categories
    return frame.numeric(name)[:, None], [name], {}


def compress(matrix: Tensor, weights: Tensor | None) -> Tensor:
    """Upper-triangular ``R`` with ``W'diag(w)W = R'R`` from one Householder QR."""
    scaled = matrix if weights is None else matrix * weights.sqrt()[:, None]
    try:
        r = torch.linalg.qr(scaled, mode="r").R
    except RuntimeError as exc:
        raise AnalysisError("numerical_failure", f"The QR factorization failed: {exc}") from exc
    if not bool(torch.isfinite(r).all()):
        raise AnalysisError("non_finite_values", "The data are too large for float64 cross "
                            "products; rescale the variables.")
    return r


def build_system(frame: ModelFrame, *, first: Sequence[str], rest: Sequence[str],
                 constant: bool, outcomes: Sequence[str]) -> SystemData:
    """Gather ``W = [1?, first..., rest...]`` once and compress it.

    ``first`` are the instrument variables of 3SLS (they lead so that the
    leading rows of ``R`` project on them); outcomes must be numeric.
    """
    categorical = set(frame.spec.categorical)
    for name in outcomes:
        if name in categorical:
            raise AnalysisError("invalid_spec", f"The outcome '{name}' cannot be categorical.")
    if frame.n < 2:
        raise AnalysisError("insufficient_observations", f"A system of equations needs more "
                            f"observations than any equation has parameters; the sample has "
                            f"{frame.n} observation(s).")
    distinct = list(dict.fromkeys([*first, *rest]))
    widths = {name: frame.design_width([name], intercept=False, categorical=[name])
              if name in categorical else 1 for name in distinct}
    width = int(constant) + sum(widths.values())
    raw_equations = frame.spec.options.get("equations")
    if raw_equations:
        parameter_count = sum(int(eq.get("constant", frame.spec.intercept))
                              + sum(widths[name] for name in eq.get("x", [])) for eq in raw_equations)
        equations_count = len(raw_equations)
    else:
        equations_count = len(outcomes)
        parameter_count = equations_count * (int(constant) + sum(widths[name] for name in frame.spec.predictors))
    compressed_rows = min(frame.n, width)
    frame.workspace_plan("combined linear system QR and covariance", {
        "combined_columns_weighted_and_qr_buffers": tensor_bytes((frame.n, width), itemsize=40),
        "compressed_factor_scratch": tensor_bytes((width, width), itemsize=64),
        "stacked_compressed_system_and_qr": tensor_bytes((equations_count * compressed_rows, parameter_count), itemsize=24),
        "stacked_parameter_covariance_and_restrictions": tensor_bytes((parameter_count, parameter_count), itemsize=64),
        "equation_residuals_and_row_scratch": tensor_bytes((frame.n, equations_count + 4)),
    })
    pieces: list[Tensor] = []
    labels: list[str] = []
    blocks: dict[str, list[int]] = {}
    terms: dict[str, list[str]] = {}
    categories: dict[str, Any] = {}
    if constant:
        pieces.append(torch.ones((frame.n, 1), dtype=torch.float64))
        labels.append("Intercept")
    for name in dict.fromkeys([*first, *rest]):
        block, names, record = _variable_block(frame, name, categorical)
        blocks[name] = list(range(len(labels), len(labels) + len(names)))
        terms[name] = names
        labels.extend(names)
        pieces.append(block)
        categories.update(record)
    raw = torch.cat(pieces, dim=1)
    weights, nobs = estimation_weights(frame)
    # With fewer rows than distinct columns R is n x p (upper trapezoidal) and still gives
    # every inner product; each equation's own N > k_i is checked when it is prepared.
    instruments = int(constant) + sum(len(blocks[name]) for name in dict.fromkeys(first))
    flat = (raw.amax(dim=0) == raw.amin(dim=0)).nonzero().flatten().tolist()
    return SystemData(compress(raw, weights), labels, blocks, terms, constant, instruments,
                      nobs, frame.n, weights, raw, categories,
                      {j for j in flat if j > 0 or not constant})


def drop_columns(system: SystemData, keep: Sequence[int]) -> SystemData:
    """Remove columns of W by re-triangularizing ``R[:, keep]`` (no pass over the rows)."""
    keep = list(keep)
    position = {old: new for new, old in enumerate(keep)}
    r = torch.linalg.qr(system.r[:, keep], mode="r").R
    blocks = {name: [position[i] for i in index if i in position]
              for name, index in system.blocks.items()}
    terms = {name: [term for i, term in zip(system.blocks[name], system.terms[name], strict=True)
                    if i in position] for name in system.blocks}
    instruments = sum(1 for i in keep if i < system.instruments)
    return SystemData(r, [system.labels[i] for i in keep], blocks, terms, system.constant,
                      instruments, system.nobs, system.rows, system.weights,
                      system.raw[:, keep], system.categories,
                      {position[i] for i in system.flat if i in position})


def compressed_columns(system: SystemData, index: Sequence[int], centred: bool,
                       rows: int | None = None) -> Tensor:
    """``R[:rows, index]``; with ``centred`` the non-constant columns are mean-deviated.

    Centred, ``index[0]`` must be the constant; a column that is constant in the
    sample becomes exactly zero (see "Centring" above).
    """
    index = list(index)
    block = system.r[:, index].clone()
    if centred:
        block[0, 1:] = 0.0
        flat = [p for p, j in enumerate(index) if p > 0 and j in system.flat]
        if flat:
            block[:, flat] = 0.0
    return block if rows is None else block[:rows]


def is_flat(system: SystemData, column: int) -> bool:
    """Whether a column of W is constant in the estimation sample."""
    return column in system.flat


def column_means(system: SystemData, index: Sequence[int]) -> Tensor:
    """Weighted means of the columns (the constant's entry is 0), from row 0 of ``R``."""
    means = system.r[0, list(index)] / system.r[0, 0]
    means[0] = 0.0
    return means


# ---- reporting ------------------------------------------------------------------------


def correlation(sigma: Tensor) -> Tensor:
    scale = sigma.diagonal().sqrt()
    return sigma / scale[:, None] / scale


def breusch_pagan(sigma: Tensor, nobs: int) -> dict[str, Any]:
    """Breusch-Pagan LM test of a diagonal residual covariance: N sum_{i>j} r_ij^2 ~ chi2."""
    corr = correlation(sigma)
    m = corr.shape[0]
    lower = torch.tril(corr, diagonal=-1)
    statistic = float(nobs * lower.square().sum())
    df = m * (m - 1) // 2
    return {"statistic": statistic, "df": df, "p_value": chi2_sf(statistic, df),
            "distribution": "chi2", "label": "Breusch-Pagan test of independent equations"}


def log_likelihood(sigma_ml: Tensor, nobs: int) -> float | None:
    """Gaussian system log likelihood at the residual covariance E'E/N."""
    sign, logdet = torch.linalg.slogdet(sigma_ml)
    if float(sign) <= 0:
        return None
    m = sigma_ml.shape[0]
    return -0.5 * nobs * (m * (1 + math.log(2 * math.pi)) + float(logdet))


def check_equation_fit(rss: float, scale: float, tss: float, name: str) -> None:
    if tss <= 0:
        raise AnalysisError("constant_outcome", f"The outcome of equation '{name}' has no "
                            "variation in the estimation sample.")
    if rss <= EXACT_FIT * scale:
        raise AnalysisError("perfect_fit", f"The regressors of equation '{name}' fit its "
                            "outcome exactly, so its residual variance is zero and the system "
                            "covariance is singular. Remove that equation or the regressor "
                            "that reproduces the outcome.")
