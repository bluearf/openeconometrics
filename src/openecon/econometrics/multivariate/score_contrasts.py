"""Fixed-family simultaneous basic bands for identified query-score contrasts."""
from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary, summary_state
from openecon.resources import plan_workspace

from . import common as c
from . import frequency_bootstrap as f
from . import score_uncertainty as s

MAX_CONTRASTS = 64


def _invalid(message):
    raise AnalysisError("invalid_contrasts", message)


def _shape(value):
    if isinstance(value, (pd.DataFrame, np.ndarray, torch.Tensor)):
        shape = tuple(value.shape)
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= MAX_CONTRASTS or not all(
                isinstance(row, (list, tuple)) for row in value):
            _invalid("Supply one to 64 sized real numeric contrast rows.")
        lengths = [len(row) for row in value]
        if len(set(lengths)) != 1:
            _invalid("The contrast matrix cannot be ragged.")
        shape = (len(value), lengths[0])
    else:
        _invalid("Supply a resident numeric array, tensor, DataFrame or nested row list.")
    if len(shape) != 2 or not 1 <= shape[0] <= MAX_CONTRASTS:
        _invalid("Supply a two-dimensional matrix with one to 64 contrast rows.")
    return shape


def _names(value, labels, columns, h):
    if isinstance(value, pd.DataFrame):
        if labels is not None or value.index.nlevels != 1 or value.columns.nlevels != 1 \
                or list(value.columns) != columns:
            _invalid("Labelled contrasts require exact physical query-score columns and their own row names.")
        labels = list(value.index)
    if labels is None:
        labels = [f"Contrast{i + 1}" for i in range(h)]
    if not isinstance(labels, (list, tuple)) or len(labels) != h or any(
            not isinstance(label, str) or not 1 <= len(label) <= 4096 for label in labels):
        _invalid("Provide one bounded string label per contrast row.")
    labels = c.name_list(labels, "contrast labels")
    if len(labels) != h or any(len(label.encode()) > 4096 for label in labels):
        _invalid("Contrast labels must be unique, bounded strings, one per row.")
    return labels


def _matrix(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad or value.layout != torch.strided \
                or value.dtype == torch.bool or value.is_complex():
            _invalid("Contrast tensors must be real strided CPU values without autograd.")
        values = value
    elif isinstance(value, (pd.DataFrame, np.ndarray)):
        dtypes = value.dtypes if isinstance(value, pd.DataFrame) else [value.dtype]
        if any(getattr(dtype, "kind", None) not in ("i", "u", "f") for dtype in dtypes):
            _invalid("Contrasts exclude boolean, complex and object/string values.")
        values = value.to_numpy() if isinstance(value, pd.DataFrame) else value
    else:
        if any(isinstance(x, bool) or not isinstance(x, Real) for row in value for x in row):
            _invalid("Contrast cells must be real numbers, excluding booleans and complex values.")
        values = value
    try:
        matrix = torch.tensor(values, dtype=c.FLOAT, device="cpu") if not isinstance(values, torch.Tensor) \
            else values.detach().to(dtype=c.FLOAT).clone()
    except (ValueError, TypeError, OverflowError, RuntimeError):
        _invalid("Contrasts must be representable as finite float64 values.")
    if not bool(torch.isfinite(matrix).all()) or bool((matrix.abs().amax(1) == 0).any()):
        _invalid("Each contrast must contain finite values and a nonzero coefficient.")
    return matrix


def _contrasts(result, data, contrasts, *, kind, procedure, labels, confidence, missing):
    c.check_choice(missing, "missing", ("drop", "raise"))
    names, p, physical, n, m, b, cells, replay_work = s._admit(result, kind)
    confidence = result.attrs["confidence"] if confidence is None else confidence
    confidence = c.check_number(confidence, "confidence", minimum=0, maximum=1, exclusive=True)
    if confidence >= 1 or (b + 1) * (1 - confidence) < 1 - 1e-12:
        raise AnalysisError("invalid_option", "The simultaneous upper tail needs at least one expected bootstrap rank.")
    rows = s.cu._rows(data, names)
    k = 2 * m if kind == "canon" else m
    q = rows * k
    if not 1 <= rows <= s.MAX_QUERY_ROWS or not 1 <= q <= s.MAX_PARAMETERS:
        raise AnalysisError("workspace_limit", "At most 128 physical query rows and score coordinates are supported.")
    h, width = _shape(contrasts)
    if width != q:
        _invalid("Contrast columns must include every physical query row, followed by its score axes.")
    axes = [f"{block}:Can{i + 1}" for block in ("X", "Y") for i in range(m)] if kind == "canon" \
        else c.numbered("Comp" if kind == "pca" else "Factor", m)
    columns = [f"query[{i}]:{axis}" for i in range(rows) for axis in axes]
    labels = _names(contrasts, labels, columns, h)
    # Charge the complete nested score state, replay, all transformed draws,
    # deviations and both live copies before tensor conversion or any refit.
    base_work = 2 * replay_work + b * (n * p * p + q * q + rows * p * k + 64 * p**3)
    work = base_work + 8 * b * (h * q + h * h + h) + 128 * h * q
    base_export = 64 * (cells + b * (q + p * k + 3 * p) + q * q + rows * p + 16 * q) + 4 * 1024**2
    export = base_export + 64 * (2 * h * q + h * h + 4 * b * h + 2 * b + 32 * h) + 4 * sum(len(x.encode()) for x in labels)
    if work > s.MAX_WORK:
        raise AnalysisError("work_limit", "Complete score replay and joint contrasts exceed 250 million work units.")
    if export > s.MAX_EXPORT_BYTES:
        raise AnalysisError("export_limit", "Complete score/contrast state exceeds the conservative 32 MiB domain.")
    plan = plan_workspace("simultaneous multivariate score contrasts", {
        "score_replay_and_both_live_outputs": 2 * base_export + 128 * (physical * p + b * p * p),
        "contrast_draws_covariance_and_deviations": 64 * (2 * h * q + h * h + 4 * b * h + b),
        "complete_portable_output": export,
    }).record()
    matrix = _matrix(contrasts)
    # Validate query type/missing alignment before the expensive canonical refit.
    s._measurement_admission(data, names)
    selected = s.cu._selected(data, names)
    if isinstance(selected, pd.DataFrame) and any(s.cu._nonreal_measurement(x) for name in names for x in selected[name]):
        raise AnalysisError("non_numeric_column", "Query measurements exclude boolean and complex values.")
    _, keep, _ = c.select(selected, names, missing=missing)
    mask = torch.tensor(keep.tolist(), dtype=torch.bool).repeat_interleave(k)
    if bool((matrix[:, ~mask] != 0).any()):
        _invalid("A contrast has a nonzero coefficient on a missing physical query coordinate.")
    scores = s._scores(result, data, kind=kind, procedure={
        "pca": "pca_fweight_bootstrap_scores", "canon": "canon_bootstrap_scores",
        "factor": "factor_bootstrap_scores"}[kind], missing=missing)
    reduced = matrix[:, mask]
    score_point = torch.tensor(scores["estimates"]["estimate"].to_numpy(), dtype=c.FLOAT)
    score_draws = torch.tensor(scores["replicates"].to_numpy(), dtype=c.FLOAT)
    point, draws = reduced @ score_point, score_draws @ reduced.T
    covariance, se, bias, lower, upper = f.joint(point, draws, confidence)
    magnitude = torch.maximum((score_draws.abs() @ reduced.abs().T).amax(0), reduced.abs() @ score_point.abs())
    if not bool(torch.isfinite(magnitude).all()) or bool((se <= 128 * torch.finfo(c.FLOAT).eps * magnitude).any()):
        raise AnalysisError("degenerate_contrast", "A declared contrast has zero or numerically cancelled bootstrap variance; rescale or revise the fixed family.")
    deviations = (draws - point) / se
    maxima = deviations.abs().amax(1)
    critical = torch.quantile(maxima, confidence, interpolation="linear")
    half_width = critical * se
    s.cu._finite(point, draws, covariance, deviations, maxima, critical, half_width, point - half_width, point + half_width)
    undefined = torch.full((h,), float("nan"), dtype=c.FLOAT)
    out = TableSet({
        "estimates": c.frame(torch.stack((point, se, point - half_width, point + half_width,
                                           bias, lower, upper, undefined, undefined), 1), index=labels,
                             columns=["estimate", "std_error", "ci_lower", "ci_upper", "bootstrap_bias",
                                      "marginal_percentile_lower", "marginal_percentile_upper", "p_value", "df"]),
        "covariance": c.frame(covariance, columns=labels, index=labels),
        "replicates": c.frame(draws, columns=labels, index=range(1, b + 1)),
        "standardized_deviations": c.frame(deviations, columns=labels, index=range(1, b + 1)),
        "joint_maxima": c.frame(maxima[:, None], columns=["absolute_maximum"], index=range(1, b + 1)),
        "contrasts": c.frame(matrix, columns=columns, index=labels),
        "retained_contrasts": c.frame(reduced, columns=scores.attrs["parameter_labels"], index=labels),
        **{"score__" + name: frame.copy(deep=True) for name, frame in scores.items()},
    }, title="Simultaneous fixed-query score contrasts",
        procedure=procedure, state_schema=f"openecon.{procedure}.v1", score_attrs=scores.attrs.copy(), score_title=scores.title,
        contrast_labels=labels, contrast_dimension=h, physical_parameter_labels=columns, confidence=confidence,
        replications=b, covariance_divisor=b - 1, quantile_interpolation="linear", critical_value=float(critical),
        familywise_intervals=True, uncertainty="first-order centered fixed-scale max-deviation bootstrap basic bands",
        standardization="one fixed bootstrap SE per contrast, shared across every draw",
        inferential_assumptions=scores.attrs["inferential_assumptions"] + "; fixed independent query/contrast family; positive first-order contrast variances; fixed finite dimension",
        query_contract=scores.attrs["query_contract"], missing_policy=missing,
        p_values_available=False, inference_df_available=False, precision="float64", device="cpu",
        estimated_work=work, estimated_complete_export_bytes=export, resource_plan=plan,
        notes=["Family is exactly the prespecified contrast rows; full cross-query and CCA X/Y dependence is retained.",
               "No per-draw studentization, stepdown p-values or exact finite-sample coverage is claimed.",
               "Nonzero weights on missing query coordinates are refused, never silently dropped.",
               "Complete source scoring uses canonical point and every seeded training refit; all original tables persist."])
    saved_summary(out)
    out.attrs["state_content_sha256"] = f._digest(out)
    if len(summary_state(out).encode()) > s.MAX_EXPORT_BYTES:
        raise AnalysisError("export_limit", "The complete portable contrast result exceeds 32 MiB.")
    return out


@c.procedure
@resident_cpu
def pca_fweight_bootstrap_score_contrasts(result: TableSet, data: Any, contrasts: Any, *,
                                         labels: list[str] | None = None, confidence: float | None = None,
                                         missing: str = "drop") -> TableSet:
    """Joint fixed linear frequency-PCA query-score contrasts and simultaneous basic bands.

    C columns follow physical query rows, then axes; nonzero coefficients on
    missing rows refuse. Covariance/correlation PCA uses every draw's moments.
    """
    return _contrasts(result, data, contrasts, kind="pca", procedure="pca_fweight_bootstrap_score_contrasts",
                      labels=labels, confidence=confidence, missing=missing)


@c.procedure
@resident_cpu
def canon_bootstrap_score_contrasts(result: TableSet, data: Any, contrasts: Any, *,
                                   labels: list[str] | None = None, confidence: float | None = None,
                                   missing: str = "drop") -> TableSet:
    """Simultaneous fixed-query paired X/Y CCA contrasts under IID/literal-frequency training.

    Preserve full cross-query and cross-block dependence. Original training
    input requires target='coefficients' and complete canonical numerical replay.
    """
    return _contrasts(result, data, contrasts, kind="canon", procedure="canon_bootstrap_score_contrasts",
                      labels=labels, confidence=confidence, missing=missing)


@c.procedure
@resident_cpu
def factor_bootstrap_score_contrasts(result: TableSet, data: Any, contrasts: Any, *,
                                    labels: list[str] | None = None, confidence: float | None = None,
                                    missing: str = "drop") -> TableSet:
    """Simultaneous fixed-query PF regression-score contrasts, IID/literal-frequency training.

    Fixed unrotated/full orthogonal target, two to four factors; each draw uses
    its own R^-1 L, centering and scales. This is estimator-functional uncertainty.
    """
    return _contrasts(result, data, contrasts, kind="factor", procedure="factor_bootstrap_score_contrasts",
                      labels=labels, confidence=confidence, missing=missing)
