"""Observed-data Gaussian ML and Little's pattern-mean homogeneity diagnostic.

Numerical kernels use resident Torch CPU float64 only. No completed data or
parameter-uncertainty covariance is fabricated by these diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.mi.common import _resident
from openecon.econometrics.multivariate.common import name_list, procedure, source
from openecon.engines.distributions import chi2_sf
from openecon.resources import plan_workspace

_FLOAT = torch.float64
_MAX_COLUMNS = 16
_MAX_PATTERNS = 1024
_MAX_WORK = 1_000_000_000
_SOURCES = (
    "https://doi.org/10.1080/01621459.1988.10478722",
    "https://doi.org/10.1111/j.2517-6161.1977.tb01600.x",
    "https://doi.org/10.1177/1536867X1301300407",
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


class MIPatternDiagnostic(BaseModel):
    """Physical positions and observed moments for one missingness pattern."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    observed_columns: tuple[str, ...]
    positions: tuple[StrictInt, ...]
    observed_means: tuple[float, ...]
    statistic_contribution: float | None = None


class MIDiagnosticSample(BaseModel):
    """Immutable original-row identity and observed-information geometry."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    n_total: StrictInt = Field(ge=1)
    n_informative: StrictInt = Field(ge=2)
    row_labels: tuple[str, ...]
    informative_positions: tuple[StrictInt, ...]
    all_missing_positions: tuple[StrictInt, ...]
    missing_counts: tuple[StrictInt, ...]
    pair_counts: tuple[tuple[StrictInt, ...], ...]


class MIDiagnosticResult(BaseModel):
    """Frozen, complete statistical state; SHA256 detects changes, not authorship.

    ``covariance`` is the full fitted Gaussian population covariance, not the
    sampling covariance of ``estimates``. Little's statistic uses a separately
    recorded n/(n-1) correction of this ML covariance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal["mi-diagnostic-v1"] = "mi-diagnostic-v1"
    method: Literal["mvnorm_em", "little_mcar"]
    columns: tuple[str, ...] = Field(min_length=1, max_length=_MAX_COLUMNS)
    estimates: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    covariance_kind: Literal["Gaussian population covariance; maximum likelihood divisor"]
    observed_loglikelihood: float
    loglikelihood_history: tuple[float, ...] = Field(min_length=2)
    converged: Literal[True]
    iterations: StrictInt = Field(ge=1, le=10000)
    tolerance: float = Field(gt=0, le=1e-2)
    final_parameter_change: float = Field(ge=0)
    max_iterations: StrictInt = Field(ge=1, le=10000)
    sample: MIDiagnosticSample
    patterns: tuple[MIPatternDiagnostic, ...] = Field(min_length=1, max_length=_MAX_PATTERNS)
    resource_plan_json: str
    statistic: float | None = None
    df: StrictInt | None = None
    p_value: float | None = None
    statistic_covariance_scale: float | None = None
    inference: str
    notes: tuple[str, ...]
    source_urls: tuple[str, ...]
    integrity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def _restore_admission(cls, state):
        if isinstance(state, dict):
            names = state.get("columns", ())
            sample = state.get("sample", {})
            sample = sample.model_dump() if isinstance(sample, BaseModel) else sample
            if not isinstance(sample, dict) or type(sample.get("n_total")) is not int:
                raise ValueError("Saved diagnostics need declared integer row geometry.")
            n, p = sample["n_total"], len(names)
            patterns, history = state.get("patterns", ()), state.get("loglikelihood_history", ())
            if (
                not 1 <= p <= _MAX_COLUMNS
                or n < 1
                or len(sample.get("row_labels", ())) != n
                or not 1 <= len(patterns) <= _MAX_PATTERNS
                or not 2 <= len(history) <= 10001
            ):
                raise ValueError("Saved diagnostic dimensions exceed their declared envelope.")
            plan_workspace(
                "Gaussian diagnostic state restoration",
                {
                    "row/index/pattern/JSON copies": n * (1024 + p * 128),
                    "pattern solves and histories": len(patterns) * p * p * 256
                    + len(history) * 128,
                },
            )
            for key in ("estimates", "covariance", "loglikelihood_history"):
                values = state.get(key, ())
                cells = (v for row in values for v in row) if key == "covariance" else iter(values)
                if any(
                    isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                    for v in cells
                ):
                    raise ValueError("Saved diagnostic parameters need finite real numeric values.")
        return state

    @model_validator(mode="after")
    def _state(self):
        p, n = len(self.columns), self.sample.n_total
        if (
            len(set(self.columns)) != p
            or any(not v for v in self.columns)
            or len(self.estimates) != p
            or len(self.covariance) != p
            or any(len(row) != p for row in self.covariance)
        ):
            raise ValueError("Diagnostic parameter dimensions/labels are inconsistent.")
        if (
            len(self.sample.row_labels) != n
            or self.sample.n_informative <= p
            or len(self.sample.informative_positions) != self.sample.n_informative
            or len(self.sample.missing_counts) != p
            or len(self.sample.pair_counts) != p
            or any(len(row) != p for row in self.sample.pair_counts)
        ):
            raise ValueError("Diagnostic sample dimensions are inconsistent.")
        observed = sorted(self.sample.informative_positions)
        omitted = sorted(self.sample.all_missing_positions)
        if (
            observed != list(self.sample.informative_positions)
            or omitted != list(self.sample.all_missing_positions)
            or sorted(observed + omitted) != list(range(n))
        ):
            raise ValueError("Physical sample positions must partition all original rows.")
        pattern_positions, information_positions = [], []
        pair_counts, missing_counts = [[0] * p for _ in range(p)], [0] * p
        seen = set()
        for pattern in self.patterns:
            names = pattern.observed_columns
            if (
                names in seen
                or not pattern.positions
                or list(pattern.positions) != sorted(set(pattern.positions))
                or names != tuple(v for v in self.columns if v in names)
                or len(pattern.observed_means) != len(names)
            ):
                raise ValueError("Saved missing-pattern geometry is inconsistent.")
            seen.add(names)
            pattern_positions.extend(pattern.positions)
            if names:
                information_positions.extend(pattern.positions)
            k = len(pattern.positions)
            for a, name in enumerate(self.columns):
                if name not in names:
                    missing_counts[a] += k
                else:
                    for b, other in enumerate(self.columns):
                        if other in names:
                            pair_counts[a][b] += k
            if not names and tuple(pattern.positions) != self.sample.all_missing_positions:
                raise ValueError("The empty pattern must retain exactly the all-missing rows.")
        if (
            sorted(pattern_positions) != list(range(n))
            or sorted(information_positions) != observed
            or missing_counts != list(self.sample.missing_counts)
            or pair_counts != [list(row) for row in self.sample.pair_counts]
            or any(v < 3 for row in pair_counts for v in row)
        ):
            raise ValueError(
                "Saved pair coverage/missing counts differ from the pattern partition."
            )
        if (
            len(self.loglikelihood_history) != self.iterations + 1
            or self.iterations > self.max_iterations
            or self.observed_loglikelihood != self.loglikelihood_history[-1]
            or self.final_parameter_change > self.tolerance
        ):
            raise ValueError("Saved convergence metadata is inconsistent.")
        for a, b in zip(self.loglikelihood_history, self.loglikelihood_history[1:]):
            if b < a - 1e-10 * max(1.0, abs(a)):
                raise ValueError("Observed log likelihood must be monotone to float64 roundoff.")
        # Admission precedes every numerical restore, including untrusted saved state.
        plan_workspace("MI diagnostic saved covariance", {"covariance restore": p * p * 64})
        cov = torch.tensor(self.covariance, dtype=_FLOAT, device="cpu")
        scale = cov.diagonal().sqrt()
        if (
            not bool(torch.isfinite(cov).all())
            or bool((cov.diagonal() <= 0).any())
            or not torch.allclose(cov, cov.T, rtol=1e-12, atol=0.0)
        ):
            raise ValueError(
                "Saved Gaussian covariance must be finite, symmetric and positive definite."
            )
        correlation = cov / scale[:, None] / scale[None, :]
        _, status = torch.linalg.cholesky_ex(correlation)
        if int(status) or float(torch.linalg.eigvalsh(correlation).min()) <= 1e-10:
            raise ValueError("Saved Gaussian covariance is singular or numerically unidentified.")
        if self.method == "little_mcar":
            df = sum(len(v.observed_columns) for v in self.patterns) - p
            correction = self.sample.n_informative / (self.sample.n_informative - 1)
            if (
                self.df != df
                or df <= 0
                or self.statistic is None
                or self.statistic < 0
                or self.statistic_covariance_scale != correction
                or self.p_value is None
                or not 0 <= self.p_value <= 1
            ):
                raise ValueError("Little MCAR inference geometry is inconsistent.")
            total = 0.0
            for pattern in self.patterns:
                ix = [self.columns.index(v) for v in pattern.observed_columns]
                value = 0.0
                if ix:
                    difference = (
                        torch.tensor(pattern.observed_means, dtype=_FLOAT, device="cpu")
                        - torch.tensor(self.estimates, dtype=_FLOAT, device="cpu")[ix]
                    ) / scale[ix]
                    value = len(pattern.positions) * float(
                        difference
                        @ torch.linalg.solve(correlation[ix][:, ix] * correction, difference)
                    )
                if pattern.statistic_contribution is None or not math.isclose(
                    value, pattern.statistic_contribution, rel_tol=1e-9, abs_tol=1e-10
                ):
                    raise ValueError(
                        "Saved Little pattern contributions differ from the fitted model."
                    )
                total += value
            if not math.isclose(
                total, self.statistic, rel_tol=1e-9, abs_tol=1e-10
            ) or not math.isclose(chi2_sf(total, df), self.p_value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Saved Little statistic/p-value differs from complete inference.")
        elif any(
            v is not None
            for v in (self.statistic, self.df, self.p_value, self.statistic_covariance_scale)
        ):
            raise ValueError("Gaussian EM is estimation, without an invented hypothesis test.")
        elif any(v.statistic_contribution is not None for v in self.patterns):
            raise ValueError("Gaussian EM must not retain MCAR-only pattern statistics.")
        try:
            plan = json.loads(self.resource_plan_json)
            if plan["estimated_workspace_bytes"] > plan["budget_bytes"]:
                raise ValueError("Saved workspace admission was not successful.")
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Saved workspace record is invalid.") from exc
        if (
            _digest(self.model_dump(mode="json", exclude={"integrity_sha256"}))
            != self.integrity_sha256
        ):
            raise ValueError("MI diagnostic state integrity changed.")
        return self

    @property
    def means(self):
        return self.estimates

    def to_json(self, path=None):
        """Save the complete immutable diagnostic state, or return its JSON text."""
        state = type(self).model_validate_json(self.model_dump_json())
        value = state.model_dump_json(indent=2)
        if path is not None:
            Path(path).write_text(value, encoding="utf-8")
        return value

    @classmethod
    def from_json(cls, value):
        """Restore JSON text (or a Path) and verify dimensions, inference and digest."""
        if isinstance(value, Path):
            value = value.read_text(encoding="utf-8")
        return cls.model_validate_json(value)

    def to_frame(self):
        state = type(self).model_validate_json(self.model_dump_json())
        frames = {
            "Gaussian means": table({"column": self.columns, "mean": self.estimates}),
            "Gaussian ML population covariance": table(
                self.covariance, columns=self.columns, index=self.columns
            ),
            "Observed likelihood convergence": table(
                {
                    "iteration": range(len(self.loglikelihood_history)),
                    "observed_loglikelihood": self.loglikelihood_history,
                }
            ),
            "Missing patterns": table(
                [
                    {
                        "observed_columns": ", ".join(v.observed_columns)
                        or "(all missing; no information)",
                        "count": len(v.positions),
                        "positions": list(v.positions),
                        "observed_means": list(v.observed_means),
                        "chi2_contribution": v.statistic_contribution,
                    }
                    for v in self.patterns
                ]
            ),
        }
        if self.method == "little_mcar":
            frames["Little MCAR mean homogeneity"] = table(
                [
                    {
                        "statistic": self.statistic,
                        "df": self.df,
                        "p_value": self.p_value,
                        "covariance_scale": self.statistic_covariance_scale,
                    }
                ]
            )
        return TableSet(
            frames,
            title=self.method,
            procedure=self.method,
            nobs=self.sample.n_informative,
            n_total=self.sample.n_total,
            method=self.method,
            inference=self.inference,
            notes=self.notes,
            diagnostic_state=state.model_dump(mode="json"),
        )

    def to_latex(self, **options):
        return self.to_frame().to_latex(**options)

    def __str__(self):
        return str(self.to_frame())

    def model_copy(self, *, update=None, deep=False):
        """Revalidate copied diagnostics, retaining their scientific integrity."""
        state = self.model_dump(mode="json")
        state.update(update or {})
        return type(self).model_validate(state)


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError(
            "mi_numerical_failure", f"{label} overflowed; rescale input explicitly."
        )
    return value


def _chol(value):
    factor, status = torch.linalg.cholesky_ex(value)
    if int(status):
        raise AnalysisError(
            "mi_singular_covariance",
            "Gaussian observed covariance is not positive definite; no ridge or pseudoinverse is substituted.",
        )
    return factor


def _admit(data, columns, max_iterations, tolerance):
    if isinstance(data, Dataset):
        raise AnalysisError(
            "unsupported_dataset", "MI diagnostics require resident data; collect deliberately."
        )
    names = name_list(columns, "columns")
    if len(names) > _MAX_COLUMNS:
        raise AnalysisError(
            "mi_dimension_limit", "Gaussian diagnostics support at most 16 columns."
        )
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, Integral)
        or not 1 <= max_iterations <= 10000
    ):
        raise AnalysisError(
            "invalid_option", "max_iterations must be an integer from 1 through 10000."
        )
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not math.isfinite(tolerance)
        or not 0 < tolerance <= 1e-2
    ):
        raise AnalysisError(
            "invalid_option", "tolerance must be finite, positive and at most 1e-2."
        )
    n, selected = _resident(data, names)
    p = len(names)
    plan = plan_workspace(
        "Gaussian observed-data MI diagnostics",
        {
            "input/standardized/conditional/temporary matrices": n * p * 128,
            "masks/positions/row identities": n * (p * 16 + 256),
            "pattern solves/covariance/results": min(n, _MAX_PATTERNS) * p * p * 256,
            "iteration history": (int(max_iterations) + 1) * 64,
        },
    )
    if int(max_iterations) * n * p * p > _MAX_WORK:
        raise AnalysisError(
            "mi_work_limit",
            "Declared EM iterations and resident dimensions exceed the one-billion nominal multiply work limit.",
        )
    frame = source(selected)
    for name in names:
        if name not in frame.columns:
            raise AnalysisError("missing_columns", f"Required column '{name}' is absent.")
        if (
            not is_numeric_dtype(frame[name].dtype)
            or is_bool_dtype(frame[name].dtype)
            or is_complex_dtype(frame[name].dtype)
        ):
            raise AnalysisError(
                "non_numeric_column", f"Column '{name}' must contain continuous real numeric data."
            )
    raw = []
    for row in frame[names].itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append(math.nan)
            else:
                value = float(value)
                if not math.isfinite(value):
                    raise AnalysisError(
                        "nonfinite_data",
                        "Observed MI diagnostic data must be finite; infinity is not a missing value.",
                    )
                values.append(value)
        raw.append(values)
    matrix = torch.tensor(raw, dtype=_FLOAT, device="cpu")
    mask = ~torch.isnan(matrix)
    informative = mask.any(1)
    positions = torch.where(informative)[0].tolist()
    all_missing = torch.where(~informative)[0].tolist()
    if len(positions) <= p:
        raise AnalysisError(
            "mi_nonidentification", "Gaussian diagnostics need more informative rows than columns."
        )
    pair_counts = mask.to(_FLOAT).T @ mask.to(_FLOAT)
    if bool((pair_counts < 3).any()):
        raise AnalysisError(
            "mi_nonidentification",
            "Each column and column pair needs at least three jointly observed rows; all-missing columns and unobserved covariances are not identified.",
        )
    offsets, scales = [], []
    for a in range(p):
        observed = matrix[mask[:, a], a]
        offset = _finite(observed.mean(), "Column mean")
        scale = _finite(((observed - offset).square().mean()).sqrt(), "Column scale")
        if float(scale) <= 0:
            raise AnalysisError(
                "mi_singular_covariance", "A diagnostic column has no observed variation."
            )
        offsets.append(offset)
        scales.append(scale)
    offsets, scales = torch.stack(offsets), torch.stack(scales)
    standardized = _finite(
        torch.where(mask, (matrix - offsets) / scales, 0.0), "Standardized observations"
    )
    for a in range(p):
        for b in range(a):
            values = standardized[mask[:, a] & mask[:, b]][:, [a, b]]
            centered = values - values.mean(0)
            pair_cov = centered.T @ centered / len(values)
            diagonal = pair_cov.diagonal().sqrt()
            if (
                bool((diagonal <= 0).any())
                or float(
                    torch.linalg.eigvalsh(pair_cov / diagonal[:, None] / diagonal[None, :]).min()
                )
                <= 1e-10
            ):
                raise AnalysisError(
                    "mi_singular_pair_coverage",
                    "A jointly observed column pair is singular or numerically unidentified; increase pair coverage or remove collinearity.",
                )
    grouped = {}
    for pos, row in enumerate(mask.tolist()):
        grouped.setdefault(tuple(i for i, yes in enumerate(row) if yes), []).append(pos)
    if len(grouped) > _MAX_PATTERNS:
        raise AnalysisError(
            "mi_pattern_limit", "Gaussian diagnostics support at most 1024 missingness patterns."
        )
    if int(max_iterations) * (n * p * p + len(grouped) * p**3) > _MAX_WORK:
        raise AnalysisError(
            "mi_work_limit", "Declared EM pattern solves exceed the one-billion nominal work limit."
        )
    patterns = []
    for ix, rows in sorted(grouped.items()):
        if ix:
            y = standardized[rows][:, list(ix)]
            patterns.append((ix, rows, y))
        else:
            patterns.append((ix, rows, None))
    sample = MIDiagnosticSample(
        n_total=n,
        n_informative=len(positions),
        row_labels=tuple(
            f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"
            for value in frame.index
        ),
        informative_positions=tuple(positions),
        all_missing_positions=tuple(all_missing),
        missing_counts=tuple(int(v) for v in (~mask).sum(0).tolist()),
        pair_counts=tuple(tuple(int(v) for v in row) for row in pair_counts.tolist()),
    )
    return names, patterns, offsets, scales, sample, plan


def _likelihood(patterns, means, covariance, scales):
    total = 0.0
    for observed, rows, y in patterns:
        if not observed:
            continue
        ix = list(observed)
        factor = _chol(covariance[ix][:, ix])
        centered = y - means[ix]
        quadratic = _finite(
            (centered * torch.cholesky_solve(centered.T, factor).T).sum(),
            "Observed likelihood quadratic",
        )
        normalizer = (
            len(ix) * math.log(2 * math.pi)
            + 2 * float(factor.diagonal().log().sum())
            + 2 * float(scales[ix].log().sum())
        )
        total -= 0.5 * (len(rows) * normalizer + float(quadratic))
    if not math.isfinite(total):
        raise AnalysisError("mi_numerical_failure", "Observed Gaussian likelihood is not finite.")
    return total


def _fit(data, columns, max_iterations, tolerance, method):
    names, patterns, offsets, scales, sample, plan = _admit(
        data, columns, max_iterations, tolerance
    )
    p, n = len(names), sample.n_informative
    if method == "little_mcar" and sum(len(v[0]) for v in patterns) - p <= 0:
        raise AnalysisError(
            "mi_degenerate_mcar",
            "Little mean-homogeneity test has zero degrees of freedom for these patterns.",
        )
    means, covariance = (
        torch.zeros(p, dtype=_FLOAT, device="cpu"),
        torch.eye(p, dtype=_FLOAT, device="cpu"),
    )
    history = [_likelihood(patterns, means, covariance, scales)]
    for iteration in range(1, int(max_iterations) + 1):
        first = torch.zeros(p, dtype=_FLOAT, device="cpu")
        second = torch.zeros((p, p), dtype=_FLOAT, device="cpu")
        for observed, rows, y in patterns:
            if not observed:
                continue
            oi = list(observed)
            mi = [i for i in range(p) if i not in observed]
            completed = means.expand(len(rows), p).clone()
            completed[:, oi] = y
            conditional = None
            if mi:
                factor = _chol(covariance[oi][:, oi])
                regression = torch.cholesky_solve(covariance[oi][:, mi], factor).T
                completed[:, mi] = means[mi] + (y - means[oi]) @ regression.T
                conditional = covariance[mi][:, mi] - regression @ covariance[oi][:, mi]
            first += completed.sum(0)
            second += completed.T @ completed
            if mi:
                ix = torch.tensor(mi, dtype=torch.int64, device="cpu")
                second[ix[:, None], ix[None, :]] += len(rows) * conditional
        next_means = _finite(first / n, "EM means")
        next_covariance = _finite(
            second / n - next_means[:, None] * next_means[None, :], "EM covariance"
        )
        next_covariance = (next_covariance + next_covariance.T) / 2
        _chol(next_covariance)
        value = _likelihood(patterns, next_means, next_covariance, scales)
        if value < history[-1] - 1e-10 * max(1.0, abs(history[-1])):
            raise AnalysisError(
                "mi_likelihood_decrease",
                "Observed likelihood decreased beyond float64 roundoff; no successful fit is returned.",
            )
        change = max(
            float((next_means - means).abs().max()),
            float((next_covariance - covariance).abs().max()),
        )
        likelihood_change = abs(value - history[-1]) / max(1.0, abs(history[-1]))
        history.append(value)
        means, covariance = next_means, next_covariance
        if change <= tolerance and likelihood_change <= tolerance:
            break
    else:
        error = AnalysisError(
            "mi_nonconvergence",
            f"Gaussian EM did not converge within {max_iterations} iterations; increase the explicit limit or inspect coverage.",
        )
        error.iterations = int(max_iterations)
        error.loglikelihood_history = tuple(history)
        error.final_parameter_change = change
        raise error
    if (
        float(
            torch.linalg.eigvalsh(
                covariance
                / covariance.diagonal().sqrt()[:, None]
                / covariance.diagonal().sqrt()[None, :]
            ).min()
        )
        <= 1e-10
    ):
        raise AnalysisError(
            "mi_singular_covariance",
            "Gaussian EM converged to a singular or numerically unidentified covariance.",
        )
    original_means = _finite(offsets + scales * means, "Original Gaussian means")
    original_covariance = _finite(
        scales[:, None] * covariance * scales[None, :], "Original Gaussian covariance"
    )
    correction = n / (n - 1) if method == "little_mcar" else None
    contributions, diagnostic_patterns = [], []
    for observed, rows, y in patterns:
        ix = list(observed)
        pattern_mean = (
            () if y is None else tuple(float(v) for v in offsets[ix] + scales[ix] * y.mean(0))
        )
        contribution = None
        if method == "little_mcar":
            contribution = 0.0
            if ix:
                difference = y.mean(0) - means[ix]
                contribution = max(
                    0.0,
                    len(rows)
                    * float(
                        difference
                        @ torch.linalg.solve(covariance[ix][:, ix] * correction, difference)
                    ),
                )
            contributions.append(contribution)
        diagnostic_patterns.append(
            MIPatternDiagnostic(
                observed_columns=tuple(names[i] for i in observed),
                positions=tuple(rows),
                observed_means=pattern_mean,
                statistic_contribution=contribution,
            )
        )
    df = sum(len(v[0]) for v in patterns) - p if method == "little_mcar" else None
    statistic = sum(contributions) if contributions else None
    payload = dict(
        schema_version="mi-diagnostic-v1",
        method=method,
        columns=tuple(names),
        estimates=tuple(float(v) for v in original_means),
        covariance=tuple(tuple(float(v) for v in row) for row in original_covariance),
        covariance_kind="Gaussian population covariance; maximum likelihood divisor",
        observed_loglikelihood=history[-1],
        loglikelihood_history=tuple(history),
        converged=True,
        iterations=iteration,
        tolerance=float(tolerance),
        final_parameter_change=change,
        max_iterations=int(max_iterations),
        sample=sample.model_dump(mode="json"),
        patterns=tuple(v.model_dump(mode="json") for v in diagnostic_patterns),
        resource_plan_json=json.dumps(plan.record(), sort_keys=True, allow_nan=False),
        statistic=statistic,
        df=df,
        p_value=chi2_sf(statistic, df) if df else None,
        statistic_covariance_scale=correction,
        inference="Asymptotic chi-square pattern mean homogeneity under iid quantitative data and common covariance"
        if df
        else "Gaussian observed-data maximum likelihood; no parameter-uncertainty inference",
        notes=(
            "All-missing rows retain original identity but contribute no observed likelihood and are omitted from informative n.",
            "Covariance is the full fitted Gaussian population covariance, not uncertainty of the estimated means.",
            "No categorical, weighted, streaming, GPU, CDM or unequal-pattern-covariance variant is supported.",
            "Little uses informative n/(n-1) times the fitted ML covariance; failure to reject does not establish MCAR."
            if df
            else "EM supplies model parameters, not completed data or multiple imputations.",
            "Conservative admission requires at least three nonsingular jointly observed observations for each column pair.",
        ),
        source_urls=_SOURCES,
    )
    payload["integrity_sha256"] = _digest(payload)
    return MIDiagnosticResult.model_validate(payload)


@procedure
def mvnorm_em(data, columns, *, max_iterations=500, tolerance=1e-8):
    """Fit identified resident Gaussian observed-data ML by conditional-moment EM.

    Unknown keyword options (including weights/device) are rejected by Python.
    Nonconvergence, absent pair coverage and singular fits raise AnalysisError.
    """
    return _fit(data, columns, max_iterations, tolerance, "mvnorm_em")


@procedure
def little_mcar(data, columns, *, max_iterations=500, tolerance=1e-8):
    """Little pattern-mean homogeneity chi-square with EM and n/(n-1) covariance.

    Only the common-covariance quantitative-data test is implemented. A
    non-significant result does not prove missing completely at random.
    """
    return _fit(data, columns, max_iterations, tolerance, "little_mcar")
