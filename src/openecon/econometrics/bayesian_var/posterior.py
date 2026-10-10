"""Immutable complete MNIW state, with canonical source validation before algebra."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictFloat, ValidationError, model_validator

from openecon.analysis_contracts import AnalysisError

from .admission import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_WORK,
    admit,
    digest,
    integer,
    load_mapping,
    metadata_admit,
    real,
)
from .design import matrices, validate_source

Matrix = tuple[tuple[StrictFloat, ...], ...]
Vector = tuple[StrictFloat, ...]


def raw(value):
    metadata_admit(value)
    return BaseModel.model_dump(value, mode="python") if isinstance(value, BaseModel) else value


def matrix(value, name, rows, columns):
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise AnalysisError("invalid_state", f"{name} has the wrong row dimension.")
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise AnalysisError("invalid_state", f"{name} has the wrong column dimension.")
        for cell in row:
            value = real(cell, name)
            if isinstance(cell, int) and int(value) != cell:
                raise AnalysisError(
                    "unsupported_precision",
                    f"{name} integers must be exactly representable in float64.",
                )


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def bounded(cls, value):
        metadata_admit(value)
        return raw(value)

    @classmethod
    def model_validate(cls, obj, **kwargs):
        try:
            return super().model_validate(obj, **kwargs)
        except ValidationError as exc:
            for error in exc.errors(include_url=False):
                cause = error.get("ctx", {}).get("error")
                if isinstance(cause, AnalysisError):
                    raise cause from exc
            raise AnalysisError(
                "invalid_state", "BVAR typed fields, shapes or schema are invalid."
            ) from exc

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        return cls.model_validate(load_mapping(json_data), **kwargs)

    def checked(self):
        return type(self).model_validate(raw(self))

    def model_dump(self, **kwargs):
        self.checked()
        return BaseModel.model_dump(self, **kwargs)

    def model_dump_json(self, **kwargs):
        self.checked()
        return BaseModel.model_dump_json(self, **kwargs)

    def model_copy(self, *, update=None, deep=False):
        # Copies are validated instead of Pydantic's unvalidated update shortcut.
        body = raw(self)
        if update:
            metadata_admit(update)
            body.update(update)
        return type(self).model_validate(body)


class MatrixNormalInverseWishartPrior(FrozenModel):
    """Sigma~IW_m(S,nu); B|Sigma~MN_{k,m}(M,V,Sigma)."""

    mean: Matrix
    row_scale: Matrix
    innovation_scale: Matrix
    degrees_of_freedom: StrictFloat

    @model_validator(mode="before")
    @classmethod
    def proper_dimensions(cls, value):
        value = raw(value)
        k, m = len(value["mean"]), len(value["innovation_scale"])
        if not 2 <= m <= 4 or not 2 <= k <= 17:
            raise AnalysisError(
                "invalid_prior", "MNIW prior dimensions require 2..4 series and 2..17 regressors."
            )
        matrix(value["mean"], "prior mean", k, m)
        matrix(value["row_scale"], "prior row scale", k, k)
        matrix(value["innovation_scale"], "prior innovation scale", m, m)
        if real(value["degrees_of_freedom"], "prior degrees_of_freedom") <= m - 1:
            raise AnalysisError(
                "invalid_prior", "A proper inverse Wishart prior requires nu > m-1."
            )
        return value


class RangeIndexState(FrozenModel):
    kind: Literal["range"]
    name: str | None
    start: int
    stop: int
    step: int


class PlainIndexState(FrozenModel):
    kind: Literal["index"]
    name: str | None
    dtype: str
    values: tuple[str | int, ...]


class DatetimeIndexState(FrozenModel):
    kind: Literal["datetime"]
    name: str | None
    unit: Literal["s", "ms", "us", "ns"]
    freq: str | None
    values: tuple[int, ...]


class BayesianVARSource(FrozenModel):
    series: tuple[str, ...]
    time: str
    period_unit: Literal["unit", "daily", "monthly", "quarterly", "annual"]
    source_columns: tuple[str, ...]
    source_dtypes: tuple[str, ...]
    source_values: tuple[tuple[int | float, ...], ...]
    source_index: RangeIndexState | PlainIndexState | DatetimeIndexState
    permutation: tuple[int, ...]
    source_sha256: str

    @model_validator(mode="before")
    @classmethod
    def canonical_source(cls, value):
        value = raw(value)
        if (
            not isinstance(value, dict)
            or "source_values" not in value
            or not value["source_values"]
        ):
            raise AnalysisError("invalid_state", "Saved BVAR requires complete source values.")
        n, m = len(value["source_values"][0]), len(value.get("series", ()))
        admit(n, m, 1, False)
        try:
            validate_source(value)
        except (KeyError, TypeError, OverflowError, ValueError) as exc:
            if isinstance(exc, AnalysisError):
                raise
            raise AnalysisError(
                "invalid_state", "Saved BVAR source descriptor cannot be reconstructed exactly."
            ) from exc
        return value


class PosteriorAlgebra(FrozenModel):
    location: Matrix
    row_scale: Matrix
    precision: Matrix
    innovation_scale: Matrix
    degrees_of_freedom: StrictFloat
    scalar_degrees_of_freedom: StrictFloat
    coefficient_mean: Matrix | None
    coefficient_covariance: Matrix | None
    innovation_mean: Matrix | None
    innovation_covariance: Matrix | None
    joint_mean: Vector | None
    joint_covariance: Matrix | None
    coefficient_intervals: tuple[tuple[tuple[StrictFloat, StrictFloat], ...], ...]
    log_marginal_likelihood: StrictFloat


class BayesianVARPosterior(FrozenModel):
    schema_version: Literal["openecon.bayesian_var_posterior.v1"] = (
        "openecon.bayesian_var_posterior.v1"
    )
    method: Literal["proper_mniw_conditional"] = "proper_mniw_conditional"
    source: BayesianVARSource
    prior: MatrixNormalInverseWishartPrior
    lags: int
    intercept: bool
    terms: tuple[str, ...]
    alpha: float
    nobs_original: int
    nobs: int
    algebra: PosteriorAlgebra
    max_work: int = DEFAULT_MAX_WORK
    max_bytes: int = DEFAULT_MAX_BYTES
    integrity_sha256: str

    @model_validator(mode="before")
    @classmethod
    def admit_before_copies(cls, value):
        value = raw(value)
        if not isinstance(value, dict) or "source" not in value:
            raise AnalysisError(
                "invalid_state", "Saved BVAR requires its original complete source."
            )
        source = raw(value["source"])
        metadata_admit(value, max_bytes=value.get("max_bytes", DEFAULT_MAX_BYTES))
        integer(value["nobs_original"], "original rows")
        integer(value["nobs"], "response rows", low=1)
        if not 0 < real(value["alpha"], "alpha") < 1:
            raise AnalysisError("invalid_option", "alpha must lie strictly between zero and one.")
        admit(
            len(source["source_values"][0]),
            len(source["series"]),
            value["lags"],
            value["intercept"],
            max_work=value.get("max_work", DEFAULT_MAX_WORK),
            max_bytes=value.get("max_bytes", DEFAULT_MAX_BYTES),
        )
        # This boundary precedes even prior/posterior tensor construction.
        validate_source(source, max_bytes=value.get("max_bytes", DEFAULT_MAX_BYTES))
        return value

    @model_validator(mode="after")
    def replay(self):
        from .kernels import close, posterior_algebra

        integer(self.lags, "lags", low=1, high=4)
        if type(self.intercept) is not bool or not 0 < real(self.alpha, "alpha") < 1:
            raise AnalysisError("invalid_state", "Saved BVAR options are invalid.")
        expected_terms = (("Intercept",) if self.intercept else ()) + tuple(
            f"L{lag}.{name}" for lag in range(1, self.lags + 1) for name in self.source.series
        )
        n = len(self.source.permutation)
        if (
            self.terms != expected_terms
            or self.nobs_original != n
            or self.nobs != n - self.lags
            or self.nobs < 1
        ):
            raise AnalysisError(
                "invalid_state", "Saved lag design/order/sample disagrees with the source."
            )
        y, x, _ = matrices(raw(self.source), self.lags, self.intercept)
        expected = posterior_algebra(y, x, self.prior, self.alpha)
        actual = raw(self.algebra)
        if set(actual) != set(expected) or not all(
            close(actual[k], v) for k, v in expected.items()
        ):
            raise AnalysisError(
                "invalid_state",
                "Saved complete MNIW algebra disagrees with optimizer-free source replay.",
            )
        body = raw(self)
        body.pop("integrity_sha256")
        if digest(body) != self.integrity_sha256:
            raise AnalysisError("invalid_state", "Saved posterior digest disagrees.")
        return self

    def summary(self):
        checked = self.checked()
        return {
            "method": checked.method,
            "series": checked.source.series,
            "nobs": checked.nobs,
            "coefficient_distribution": "matrix-t; scalar/rank-one contrasts Student t",
            "scalar_df": checked.algebra.scalar_degrees_of_freedom,
            "location": checked.algebra.location,
            "mean": checked.algebra.coefficient_mean,
            "full_covariance": checked.algebra.coefficient_covariance,
            "joint_mean": checked.algebra.joint_mean,
            "joint_covariance": checked.algebra.joint_covariance,
            "log_marginal_likelihood": checked.algebra.log_marginal_likelihood,
        }


def restore(value):
    try:
        return BayesianVARPosterior.model_validate(
            raw(value) if isinstance(value, BaseModel) else load_mapping(value)
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError(
            "invalid_state", "Saved BVAR posterior fails typed complete-state validation."
        ) from exc


def admit_parent(value, **outputs):
    """Admit the requested operation before numerical validation of its parent."""
    body = raw(value) if isinstance(value, BaseModel) else load_mapping(value)
    try:
        source = raw(body["source"])
        admit(
            len(source["source_values"][0]),
            len(source["series"]),
            body["lags"],
            body["intercept"],
            max_work=body["max_work"],
            max_bytes=body["max_bytes"],
            **outputs,
        )
        expected = (("Intercept",) if body["intercept"] else ()) + tuple(
            f"L{lag}.{name}" for lag in range(1, body["lags"] + 1) for name in source["series"]
        )
        if tuple(body["terms"]) != expected:
            raise AnalysisError("invalid_state", "Saved complete lag-design order is invalid.")
        validate_source(source, max_bytes=body["max_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError(
            "invalid_state", "Saved parent declarations are invalid before output admission."
        ) from exc
    return body
