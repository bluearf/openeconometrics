"""Versioned, JSON-safe contracts shared by every OpenEconometrics client."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelSpec(BaseModel):
    """An explicit statistical specification for any registered estimator.

    ``outcome``, ``predictors``, ``categorical``, ``intercept``, ``covariance``,
    ``cluster``, ``missing`` and ``alpha`` are shared by all estimators. Weights
    and the panel/time structure are first-class fields. Everything specific to
    one estimator lives in ``columns`` (additional named column roles such as
    ``endogenous``, ``instruments`` or ``absorb``) and ``options`` (settings
    such as ``model="fe"`` or ``lags=4``). The estimator registry declares which
    roles, options, weights and covariance estimators each estimator accepts;
    anything else is rejected here, before data are read.

    ``covariance=None`` selects the estimator's documented default (cluster
    when cluster columns are given). The validated spec always carries the
    resolved name.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    estimator: str = "ols"
    outcome: str = Field(min_length=1)
    predictors: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    intercept: bool = True
    covariance: str | None = None
    cluster: str | list[str] | None = None
    missing: Literal["raise", "drop"] = "raise"
    alpha: float = Field(default=0.05, gt=0, lt=1)
    weights: str | None = None
    weight_type: Literal["aweight", "fweight", "pweight", "iweight"] | None = None
    panel: str | None = None
    time: str | None = None
    columns: dict[str, str | list[str]] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("outcome", "weights", "panel", "time")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Column names must not be blank.")
        return value

    @field_validator("cluster")
    @classmethod
    def validate_cluster(cls, value: str | list[str] | None) -> str | list[str] | None:
        names = [value] if isinstance(value, str) else value or []
        if any(not name.strip() for name in names):
            raise ValueError("Column names must not be blank.")
        if isinstance(value, list):
            # One cluster column is always stored as text; an empty list is no clustering.
            return value[0] if len(value) == 1 else value or None
        return value

    @field_validator("predictors", "categorical")
    @classmethod
    def validate_names(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Column names must not be blank.")
        if len(values) != len(set(values)):
            raise ValueError("Column lists must not contain duplicate names.")
        return values

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, value: dict[str, str | list[str]]) -> dict[str, str | list[str]]:
        for role, names in value.items():
            if any(not name.strip() for name in ([names] if isinstance(names, str) else names)):
                raise ValueError(f"Column role '{role}' contains a blank column name.")
        return value

    @model_validator(mode="after")
    def validate_relationships(self) -> Self:
        if self.estimator == "ols":
            from openecon.linear_ols.spec import validate_spec
            self.covariance = validate_spec(self)
            return self
        from openecon.econometrics.registry import validate_spec

        if self.outcome in self.predictors:
            raise ValueError("The outcome must not also be a predictor.")
        if self.estimator in {"logit", "probit"}:
            # The core binary estimators keep their explicit covariance contract
            # here, where the editor's parameter catalogue reads it.
            if self.covariance is None:
                self.covariance = "cluster" if self.cluster else "nonrobust"
            if self.covariance not in {"nonrobust", "cluster"}:
                raise ValueError("Binary models support nonrobust or cluster covariance.")
        self.covariance = validate_spec(self)
        allowed = set(self.predictors)
        for names in self.columns.values():
            allowed.update([names] if isinstance(names, str) else names)
        if not set(self.categorical).issubset(allowed):
            raise ValueError("Every categorical variable must be a predictor.")
        return self


class Coefficient(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    term: str
    estimate: float
    std_error: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    # Multi-equation models (selection, inflation, outcome categories, ancillary
    # parameters) group their terms; ``term`` itself stays unique in a result.
    equation: str | None = None


class ResultBundle(BaseModel):
    """Persistable results; predictions are a chart sample, not a new fit."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str
    created_at: str
    spec: ModelSpec
    nobs: int
    nobs_original: int
    dropped_rows: int
    coefficients: list[Coefficient]
    covariance_matrix: list[list[float]]
    metrics: dict[str, float | None]
    warnings: list[str]
    predictions: list[dict[str, int | float]]
    sample_positions: list[int]
    provenance: dict[str, Any]
    inference: dict[str, Any]
    # Display title of the fitted model; specification tests (each a mapping
    # with statistic, df, p_value, distribution); bounded model-specific output.
    title: str | None = None
    tests: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_latex(self, buf=None, **options):
        """Export publication coefficient/SE rows; style='diagnostic' includes detailed inference."""
        from openecon.latex import to_latex
        return to_latex(self, buf=buf, **options)

    @property
    def latex(self):
        """A publication table with coefficients, SEs, fit statistics and actual inference notes."""
        return self.to_latex()

    def summary(self, format: Literal["text", "latex"] = "text") -> str:
        """Return a readable coefficient table without exposing compute settings."""
        if format == "latex":
            return self.to_latex()
        if format != "text":
            raise ValueError("summary format must be 'text' or 'latex'.")
        level = 100 * (1 - self.spec.alpha)
        title = f"{self.title or 'OpenEconometrics ' + self.spec.estimator.upper()} — {self.spec.outcome}"
        if self.inference.get("available") is False:
            from openecon.publication import model_notes
            lines = [title, f"Observations: {self.nobs}  |  Target: {self.extra.get('target') or 'fixed-parameter evaluation'}"]
            if self.extra.get("target")=="prediction":
                from openecon.latex import _predictive_rows
                headers, values, _ = _predictive_rows(self)
                rows = [headers[0], *[[f"{v:.6g}" if isinstance(v,(float,int)) else str(v) for v in row] for row in values]]
                widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
                lines.extend("  ".join(value.ljust(widths[i]) for i,value in enumerate(row)) for row in rows)
            lines.extend(f"{name}: {value:.6g}" for name,value in self.metrics.items() if value is not None)
            lines.extend(model_notes(self))
            return "\n".join(lines)
        # Registry results record the covariance actually used (bootstrap, a
        # robust alias resolved to HC1, an unit-root fallback) in inference.
        covariance = (self.inference.get("covariance") or self.spec.covariance
                      if self.title is not None else self.spec.covariance)
        context = (
            f"Observations: {self.nobs}  |  Covariance: {covariance}"
            f"  |  Confidence: {level:g}%"
        )
        headers = ["Term", "Estimate", "Std. error", "t" if self.inference["use_t"] else "z", "P>|stat|", "CI lower", "CI upper"]
        rows = []
        equation = None
        for c in self.coefficients:
            if c.equation != equation:
                equation = c.equation
                if equation is not None:
                    rows.append([f"[{equation}]", "", "", "", "", "", ""])
            rows.append([c.term, f"{c.estimate:.6g}", f"{c.std_error:.6g}", f"{c.statistic:.6g}",
                         f"{c.p_value:.6g}", f"{c.ci_low:.6g}", f"{c.ci_high:.6g}"])
        widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]

        def render(row: list[str]) -> str:
            return "  ".join(value.ljust(widths[i]) if i == 0 else value.rjust(widths[i])
                             for i, value in enumerate(row))

        lines = [title, context, "", render(headers), "  ".join("-" * w for w in widths),
                 *(render(row) for row in rows)]
        if self.dropped_rows:
            reason = "missing " if self.title is None else ""
            lines.append(f"\nExcluded {reason}observations: {self.dropped_rows}")
        if self.title is None:
            metrics = [(name, value) for name, value in self.metrics.items()
                       if name in {"r_squared", "adjusted_r_squared", "pseudo_r_squared", "aic", "bic"}
                       and value is not None]
            if metrics:
                lines.append("\n" + "  |  ".join(f"{name}: {value:.6g}" for name, value in metrics))
        else:
            # Registry estimators order their own fit statistics; print them all.
            metrics = [f"{name}: {value:.6g}" for name, value in self.metrics.items() if value is not None]
            lines.extend(("\n" if start == 0 else "") + "  |  ".join(metrics[start:start + 4])
                         for start in range(0, len(metrics), 4))
        for name, test in self.tests.items():
            if not isinstance(test, dict) or test.get("statistic") is None:
                continue
            degrees = ", ".join(f"{test[key]:g}" for key in ("df", "df2") if test.get(key) is not None)
            label = test.get("label", name)
            reference = f"{test.get('distribution', 'statistic')}({degrees})" if degrees else test.get("distribution", "statistic")
            p_value = test.get("p_value")
            lines.append(f"{label}: {reference} = {test['statistic']:.6g}"
                         + (f", p = {p_value:.4g}" if p_value is not None else ""))
        lines.extend(f"Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)
