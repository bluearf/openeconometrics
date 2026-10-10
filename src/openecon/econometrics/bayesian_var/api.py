"""Bounded conditional proper-conjugate VAR and exact fixed-design queries."""

from __future__ import annotations

from pydantic import StrictFloat, model_validator

from openecon.analysis_contracts import AnalysisError

from .admission import DEFAULT_MAX_BYTES, DEFAULT_MAX_WORK, admit, digest, metadata_admit, real
from .design import capture, matrices
from .posterior import (
    BayesianVARPosterior,
    FrozenModel,
    Matrix,
    MatrixNormalInverseWishartPrior,
    Vector,
    admit_parent,
    matrix,
    raw,
    restore,
)


def bayes_var(
    data,
    series,
    *,
    time,
    prior,
    lags=1,
    intercept=True,
    period_unit="unit",
    alpha=0.05,
    max_work=DEFAULT_MAX_WORK,
    max_bytes=DEFAULT_MAX_BYTES,
    device="cpu",
):
    """Fit explicit MNIW VAR conditional on its first p observations; no prior defaults."""
    if device != "cpu":
        raise AnalysisError(
            "unsupported_device", "Proper conjugate VAR supports native CPU float64 only."
        )
    alpha = real(alpha, "alpha")
    if not 0 < alpha < 1:
        raise AnalysisError("invalid_option", "alpha must lie strictly between zero and one.")
    source = capture(
        data,
        series,
        time,
        period_unit,
        p=lags,
        intercept=intercept,
        max_work=max_work,
        max_bytes=max_bytes,
    )
    metadata_admit(prior, max_bytes=max_bytes)
    try:
        prior = MatrixNormalInverseWishartPrior.model_validate(raw(prior))
    except (ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError(
            "invalid_prior",
            "Declare complete finite proper MNIW prior matrices and degrees of freedom.",
        ) from exc
    from .kernels import posterior_algebra

    y, x, _ = matrices(source, lags, intercept)
    algebra = posterior_algebra(y, x, prior, alpha)
    body = {
        "schema_version": "openecon.bayesian_var_posterior.v1",
        "method": "proper_mniw_conditional",
        "source": source,
        "prior": raw(prior),
        "lags": lags,
        "intercept": intercept,
        "terms": (("Intercept",) if intercept else ())
        + tuple(f"L{lag}.{name}" for lag in range(1, lags + 1) for name in source["series"]),
        "alpha": alpha,
        "nobs_original": len(data),
        "nobs": len(data) - lags,
        "algebra": algebra,
        "max_work": max_work,
        "max_bytes": max_bytes,
    }
    body["integrity_sha256"] = digest(body)
    return BayesianVARPosterior.model_validate(body)


def prediction_algebra(parent, design):
    import torch

    from .kernels import tensor

    x = tensor(design)
    a = parent.algebra
    m = len(parent.source.series)
    nu, df = a.degrees_of_freedom, a.scalar_degrees_of_freedom
    location = x @ tensor(a.location)
    row = x @ tensor(a.row_scale) @ x.T
    # Future innovations are independent conditional on one shared Sigma.
    row += torch.eye(len(x), dtype=x.dtype, device="cpu")
    scale = tensor(a.innovation_scale)
    covariance = torch.kron(scale / (nu - m - 1), row).tolist() if nu > m + 1 else None
    return {
        "location": location.tolist(),
        "row_scale": row.tolist(),
        "innovation_scale": a.innovation_scale,
        "degrees_of_freedom": nu,
        "scalar_degrees_of_freedom": df,
        "row_predictive_scale": [(float(row[i, i]) / df * scale).tolist() for i in range(len(x))],
        "mean": location.tolist() if nu > m else None,
        "full_covariance": covariance,
    }


class BayesianVARPrediction(FrozenModel):
    schema_version: str = "openecon.bayesian_var_fixed_prediction.v1"
    parent: BayesianVARPosterior
    design: Matrix
    terms: tuple[str, ...]
    location: Matrix
    row_scale: Matrix
    innovation_scale: Matrix
    degrees_of_freedom: StrictFloat
    scalar_degrees_of_freedom: StrictFloat
    row_predictive_scale: tuple[Matrix, ...]
    mean: Matrix | None
    full_covariance: Matrix | None
    integrity_sha256: str

    @model_validator(mode="before")
    @classmethod
    def preflight(cls, value):
        value = raw(value)
        metadata_admit(value)
        parent = raw(value["parent"])
        source = raw(parent["source"])
        q = len(value["design"])
        if q < 1:
            raise AnalysisError(
                "empty_sample", "At least one declared fixed-design row is required."
            )
        admit(
            len(source["permutation"]),
            len(source["series"]),
            parent["lags"],
            parent["intercept"],
            queries=q,
            max_work=parent["max_work"],
            max_bytes=parent["max_bytes"],
        )
        matrix(value["design"], "prediction design", q, len(parent["terms"]))
        return value

    @model_validator(mode="after")
    def replay(self):
        from .kernels import close

        if (
            self.schema_version != "openecon.bayesian_var_fixed_prediction.v1"
            or self.terms != self.parent.terms
        ):
            raise AnalysisError("invalid_state", "Fixed prediction design terms/schema disagree.")
        body = raw(self)
        expected = prediction_algebra(self.parent, self.design)
        if not all(close(body[key], value) for key, value in expected.items()):
            raise AnalysisError(
                "invalid_state", "Fixed-design matrix-t predictive state disagrees."
            )
        body.pop("integrity_sha256")
        if digest(body) != self.integrity_sha256:
            raise AnalysisError("invalid_state", "Predictive state digest disagrees.")
        return self


def bayes_var_predict(result, design, *, terms):
    """Declared fixed rows in saved lag-design order; rows share posterior parameters."""
    metadata_admit(design)
    metadata_admit(terms)
    body = admit_parent(result, queries=len(design))
    if not isinstance(terms, (tuple, list)):
        raise AnalysisError("invalid_spec", "Declare bounded ordered query terms as a list/tuple.")
    matrix(design, "prediction design", len(design), len(body["terms"]))
    parent = restore(body)
    if tuple(terms) != parent.terms:
        raise AnalysisError(
            "invalid_spec", "Explicit query terms must exactly equal the saved lag-design order."
        )
    matrix(design, "prediction design", len(design), len(parent.terms))
    admit(
        parent.nobs_original,
        len(parent.source.series),
        parent.lags,
        parent.intercept,
        queries=len(design),
        max_work=parent.max_work,
        max_bytes=parent.max_bytes,
    )
    if not design:
        raise AnalysisError("empty_sample", "At least one fixed-design query row is required.")
    design = tuple(tuple(real(cell, "query design") for cell in row) for row in design)
    body = {
        "schema_version": "openecon.bayesian_var_fixed_prediction.v1",
        "parent": raw(parent),
        "design": design,
        "terms": tuple(terms),
        **prediction_algebra(parent, design),
    }
    body["integrity_sha256"] = digest(body)
    return BayesianVARPrediction.model_validate(body)


def contrast_algebra(parent, left, right, threshold, alpha):
    import math

    from openecon.engines.distributions import t_cdf, t_isf

    from .kernels import tensor

    a = parent.algebra
    left_tensor, right_tensor = tensor(left), tensor(right)
    location = float(left_tensor @ tensor(a.location) @ right_tensor)
    squared_scale = (
        float(left_tensor @ tensor(a.row_scale) @ left_tensor)
        * float(right_tensor @ tensor(a.innovation_scale) @ right_tensor)
        / a.scalar_degrees_of_freedom
    )
    scale = math.sqrt(squared_scale)
    critical = t_isf(alpha / 2, a.scalar_degrees_of_freedom)
    cdf = (
        t_cdf((threshold - location) / scale, a.scalar_degrees_of_freedom)
        if scale
        else float(threshold >= location)
    )
    return {
        "location": location,
        "scale": scale,
        "degrees_of_freedom": a.scalar_degrees_of_freedom,
        "interval": (location - critical * scale, location + critical * scale),
        "probability_le_threshold": cdf,
        "mean": location if a.scalar_degrees_of_freedom > 1 or scale == 0 else None,
        "variance": squared_scale * a.scalar_degrees_of_freedom / (a.scalar_degrees_of_freedom - 2)
        if a.scalar_degrees_of_freedom > 2
        else (0.0 if scale == 0 else None),
    }


class BayesianVARContrast(FrozenModel):
    schema_version: str = "openecon.bayesian_var_rank_one_contrast.v1"
    parent: BayesianVARPosterior
    left: Vector
    right: Vector
    threshold: float
    alpha: float
    location: StrictFloat
    scale: StrictFloat
    degrees_of_freedom: StrictFloat
    interval: tuple[StrictFloat, StrictFloat]
    probability_le_threshold: StrictFloat
    mean: StrictFloat | None
    variance: StrictFloat | None
    integrity_sha256: str

    @model_validator(mode="before")
    @classmethod
    def preflight(cls, value):
        value = raw(value)
        metadata_admit(value)
        parent = raw(value["parent"])
        source = raw(parent["source"])
        matrix((value["left"],), "left contrast", 1, len(parent["terms"]))
        matrix((value["right"],), "right contrast", 1, len(source["series"]))
        real(value["threshold"], "threshold")
        if not 0 < real(value["alpha"], "alpha") < 1:
            raise AnalysisError("invalid_option", "alpha must lie strictly between zero and one.")
        return value

    @model_validator(mode="after")
    def replay(self):
        from .kernels import close

        expected = contrast_algebra(self.parent, self.left, self.right, self.threshold, self.alpha)
        body = raw(self)
        if self.schema_version != "openecon.bayesian_var_rank_one_contrast.v1" or not all(
            close(body[k], v) for k, v in expected.items()
        ):
            raise AnalysisError(
                "invalid_state", "Exact rank-one Student t contrast state disagrees."
            )
        body.pop("integrity_sha256")
        if digest(body) != self.integrity_sha256:
            raise AnalysisError("invalid_state", "Contrast digest disagrees.")
        return self


def bayes_var_contrast(result, left, right, *, threshold=0.0, alpha=0.05):
    metadata_admit((left, right))
    body = admit_parent(result)
    matrix((left,), "left contrast", 1, len(body["terms"]))
    matrix((right,), "right contrast", 1, len(body["source"]["series"]))
    parent = restore(body)
    matrix((left,), "left contrast", 1, len(parent.terms))
    matrix((right,), "right contrast", 1, len(parent.source.series))
    threshold, alpha = real(threshold, "threshold"), real(alpha, "alpha")
    if not 0 < alpha < 1:
        raise AnalysisError("invalid_option", "alpha must lie strictly between zero and one.")
    left, right = (
        tuple(real(v, "left contrast") for v in left),
        tuple(real(v, "right contrast") for v in right),
    )
    body = {
        "schema_version": "openecon.bayesian_var_rank_one_contrast.v1",
        "parent": raw(parent),
        "left": left,
        "right": right,
        "threshold": threshold,
        "alpha": alpha,
        **contrast_algebra(parent, left, right, threshold, alpha),
    }
    body["integrity_sha256"] = digest(body)
    return BayesianVARContrast.model_validate(body)
