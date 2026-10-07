"""Conventions shared by the treatment-effects estimators.

* **Treatment levels.** The treatment column holds two or more levels (numbers,
  strings or booleans). Levels are sorted; the control is the first level
  unless ``control`` names another one, and it is always moved to position 0,
  so codes are 0 for the control and 1..L-1 for the other levels (Stata's
  ``control()`` default is likewise the lowest level).
* **Term names** follow Stata's ``e(b)``: ``ATE:r1vs0.treat``,
  ``ATET:r1vs0.treat``, ``POmean:0.treat`` and, for ``estimand='pomeans'``,
  ``POmeans:1.treat``; the equation of each term is the part before the colon.
* **Weights.** fweights replicate observations (``N = sum f``) and pweights
  are sampling weights (rescaled to mean one, which leaves every estimate
  and the sandwich covariance unchanged). Both multiply every estimating
  equation.
* **Covariance of stacked estimating equations.** With estimating functions
  ``psi_i(theta)`` and ``sum_i w_i psi_i(theta_hat) = 0``,
  ``V = A^-1 B A^-T`` with ``A = sum_i w_i d psi_i / d theta'`` and
  ``B = sum_i (w_i psi_i)(w_i psi_i)'`` (pweights), ``sum_i f_i psi_i psi_i'``
  (fweights) or ``sum_g t_g t_g'`` with cluster totals ``t_g`` of the weighted
  scores. No small-sample factor is applied, the convention of Stata's
  ``gmm``-based ``teffects`` estimators (recorded as uncertain in the docs).
  The score matrix is never formed for all rows at once: the meat is
  accumulated over row blocks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame
from openecon.econometrics.discrete.common import json_label, label_text

# Rows per block when the score matrix is accumulated into the sandwich meat.
_BLOCK_ROWS = 1 << 16


@dataclass
class Treatment:
    column: str
    labels: list[Any]          # JSON labels, control first
    codes: Tensor              # int64 [n]: 0 = control
    counts: list[int]          # observations per level (rows, not weighted)

    @property
    def levels(self) -> int:
        return len(self.labels)

    def text(self, level: int) -> str:
        return label_text(self.labels[level])

    def effect_term(self, equation: str, level: int) -> str:
        return f"{equation}:r{self.text(level)}vs{self.text(0)}.{self.column}"

    def mean_term(self, equation: str, level: int) -> str:
        return f"{equation}:{self.text(level)}.{self.column}"


def treatment_coding(frame: ModelFrame, column: str, control: Any = None) -> Treatment:
    """Code the treatment column of the current sample (control level = code 0)."""
    series = frame.series(column)
    try:
        values, unique = pd.factorize(series, sort=True)
    except TypeError as exc:
        raise AnalysisError("invalid_treatment", f"Treatment '{column}' mixes label types; use "
                            "one type (numbers or strings) for its levels.") from exc
    labels = [json_label(value) for value in unique]
    if len(labels) < 2:
        raise AnalysisError("invalid_treatment", f"Treatment '{column}' has only one level in the "
                            "estimation sample; treatment effects need treated and control "
                            "observations.")
    order = list(range(len(labels)))
    if control is not None:
        matches = [i for i, label in enumerate(labels)
                   if label == control or label_text(label) == label_text(control)]
        if not matches:
            raise AnalysisError("invalid_treatment", f"control={control!r} is not a level of "
                                f"'{column}' (levels: {', '.join(map(label_text, labels))}).")
        order.remove(matches[0])
        order.insert(0, matches[0])
    position = torch.empty(len(labels), dtype=torch.int64)
    position[torch.tensor(order)] = torch.arange(len(labels))
    codes = position[torch.from_numpy(values.astype("int64"))]
    counts = torch.bincount(codes, minlength=len(labels)).tolist()
    return Treatment(column, [labels[i] for i in order], codes, counts)


@dataclass
class StudyWeights:
    user: Tensor               # w_i of every estimating-equation sum (ones when unweighted)
    frequency: bool            # fweights: the meat weights psi psi' by f_i
    nobs: int                  # N as Stata counts it
    weighted: bool


def study_weights(frame: ModelFrame) -> StudyWeights:
    raw = frame.weights()
    if raw is None:
        return StudyWeights(torch.ones(frame.n, dtype=torch.float64), False, frame.n, False)
    if frame.spec.weight_type == "fweight":
        return StudyWeights(raw, True, int(round(float(raw.sum()))), True)
    return StudyWeights(raw / raw.mean(), False, frame.n, True)


def stacked_covariance(frame: ModelFrame, jacobian: Tensor,
                       scores: Callable[[int, int], Tensor], weights: StudyWeights
                       ) -> tuple[Tensor, dict[str, Any]]:
    """``A^-1 B A^-T`` for stacked estimating equations (see the module notes).

    ``scores(start, stop)`` returns the unweighted estimating-function rows of
    observations ``start:stop``; they are accumulated block by block.
    """
    spec = frame.spec
    p, n = jacobian.shape[0], frame.n
    w = weights.user
    info: dict[str, Any] = {"covariance": spec.covariance}
    if spec.covariance == "cluster":
        dimensions = frame.cluster_dimensions()
        if len(dimensions) != 1:
            raise AnalysisError("cluster_dimensions", "Treatment-effects estimators support one "
                                "cluster column.")
        codes, count = dimensions[0]
        totals = torch.zeros((count, p), dtype=torch.float64)
        for start in range(0, n, _BLOCK_ROWS):
            stop = min(n, start + _BLOCK_ROWS)
            totals.index_add_(0, codes[start:stop], scores(start, stop) * w[start:stop, None])
        meat = totals.T @ totals
        info.update({"correction": "cluster sandwich of the stacked estimating equations, no "
                                   "small-sample factor", "cluster_count": count,
                     "cluster_column": spec.cluster if isinstance(spec.cluster, str)
                     else spec.cluster[0]})
    else:
        meat = torch.zeros((p, p), dtype=torch.float64)
        for start in range(0, n, _BLOCK_ROWS):
            stop = min(n, start + _BLOCK_ROWS)
            rows = scores(start, stop)
            scaled = rows * (w[start:stop].sqrt() if weights.frequency else w[start:stop])[:, None]
            meat += scaled.T @ scaled
        info["correction"] = ("robust sandwich of the stacked estimating equations, no "
                              "small-sample factor")
    try:
        half = torch.linalg.solve(jacobian, meat)
        covariance = torch.linalg.solve(jacobian, half.T)
    except RuntimeError as exc:
        raise AnalysisError("singular_jacobian", "The derivative matrix of the stacked estimating "
                            "equations is singular, so standard errors are undefined. Check for "
                            "covariates that do not vary within a treatment level.") from exc
    covariance = (covariance + covariance.T) / 2
    if not bool(torch.isfinite(covariance).all()):
        raise AnalysisError("non_finite_result", "The stacked covariance is not finite.")
    info.update({"small_sample_correction": 1.0, "df_inference": None,
                 "variance_method": "stacked estimating equations (M-estimation sandwich)"})
    return covariance, info


def require_periods(frame: ModelFrame) -> None:
    """The time column must hold integer periods (or dates), not labels (csdid, eventstudy)."""
    series = frame.series(frame.spec.time)
    if not (is_numeric_dtype(series.dtype) or is_datetime64_any_dtype(series.dtype)) \
            or is_bool_dtype(series.dtype):
        raise AnalysisError("invalid_time", f"Time column '{frame.spec.time}' must hold integer "
                            "periods (or dates) in the units of the treatment-time column.")


def check_overlap(probabilities: Tensor, tolerance: float, treatment: Treatment) -> None:
    """Stata's pstolerance rule: no estimated treatment probability below the tolerance."""
    low = probabilities < tolerance
    if bool(low.any()):
        rows = int(low.any(dim=1).sum())
        levels = [treatment.text(i) for i in range(treatment.levels) if bool(low[:, i].any())]
        raise AnalysisError(
            "overlap_violation",
            f"{rows} observation(s) have an estimated probability below pstolerance={tolerance:g} "
            f"of receiving treatment level(s) {', '.join(levels)}: the overlap assumption is "
            "violated. Restrict the sample to the region of common support, coarsen or drop the "
            "covariates that predict treatment almost perfectly, or (knowingly) lower "
            "pstolerance.")


def _summary(values: Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {}
    sample = values if values.numel() <= 1 << 24 else values[:: values.numel() // (1 << 23) + 1]
    quartiles = torch.quantile(sample, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    return {"n": int(values.numel()), "min": float(values.min()), "p25": float(quartiles[0]),
            "median": float(quartiles[1]), "mean": float(values.mean()),
            "p75": float(quartiles[2]), "max": float(values.max())}


def propensity_summary(probabilities: Tensor, treatment: Treatment) -> dict[str, Any]:
    """Overlap diagnostics: estimated treatment probabilities by observed level.

    For a binary treatment the summary is of P(treated | x) in each group (Stata's
    ``teoverlap`` plots the same densities); for more levels of P(level | x)
    for every level, by observed level.
    """
    summary: dict[str, Any] = {}
    targets = [1] if treatment.levels == 2 else list(range(treatment.levels))
    for target in targets:
        key = f"P({treatment.column}={treatment.text(target)})"
        summary[key] = {treatment.text(level): _summary(probabilities[treatment.codes == level,
                                                                       target])
                        for level in range(treatment.levels)}
    return summary


def coefficient_table(terms: Sequence[str], estimates: Tensor, covariance: Tensor
                      ) -> list[dict[str, Any]]:
    """Auxiliary-equation coefficients with standard errors (for ``extra``)."""
    errors = covariance.diagonal().clamp_min(0).sqrt()
    return [{"term": term, "estimate": float(estimates[i]), "std_error": float(errors[i])}
            for i, term in enumerate(terms)]
