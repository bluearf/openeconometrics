"""Strict, versioned analytic posterior state, independent of frequentist results."""
from __future__ import annotations

import math
from functools import wraps
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, model_serializer, model_validator

from openecon.econometrics.mi.common import _decode_index, _encode_index, _index_envelope
from openecon.econometrics.state_lifecycle import _encoded_json_admission, _json_export_admission, _state_copy_admission
from openecon.resources import plan_workspace, workspace_budget_bytes

from .core import DEFAULT_MAX_BYTES, DEFAULT_MAX_WORK, MAX_JSON_BYTES, MAX_PREDICTORS, MAX_ROWS, admit, close, digest, matrices, metadata_bytes, posterior_algebra


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        _encoded_json_admission(json_data, limit=MAX_JSON_BYTES, operation="Bayesian typed JSON decoding")
        return super().model_validate_json(json_data, **kwargs)

    def __deepcopy__(self, memo=None):
        _state_copy_admission(self, operation="Bayesian typed state deep copy")
        # Rebuild through the declared validators before copying an object
        # created through model_construct; shallow model_copy stays unchanged.
        type(self).model_validate(BaseModel.model_dump(self, mode="json"))
        return super().__deepcopy__(memo)

    @wraps(BaseModel.model_dump)
    def model_dump(self, **kwargs):
        _state_copy_admission(self, operation="Bayesian typed portable state copy")
        return super().model_dump(**kwargs)

    @wraps(BaseModel.model_dump_json)
    def model_dump_json(self, **kwargs):
        _json_export_admission(self, indent=kwargs.get("indent"), limit=MAX_JSON_BYTES,
                               operation="Bayesian complete JSON serialization")
        return super().model_dump_json(**kwargs)

    def _checked_summary(self):
        # model_copy/model_construct deliberately bypass Pydantic validation.
        # Rendering uncertainty must enforce the same semantic boundary as a
        # saved prediction, with bounded admission before serialization copies.
        target = self if isinstance(self, PosteriorBundle) else getattr(self, "posterior", None)
        if not isinstance(target, PosteriorBundle):
            raise ValueError("A summary requires a complete typed posterior target.")
        plan_workspace(
            "bayesian_typed_summary_copy",
            {"typed_summary_validation": 3 * metadata_bytes(self)},
            budget_bytes=min(target.max_bytes, workspace_budget_bytes()),
        )
        return type(self).model_validate(self.model_dump(mode="json"))


class IndexLabel(FrozenModel):
    type: Literal["scalar", "tuple", "timestamp", "timedelta", "date", "nan", "missing", "nat"]
    value: str | bool | StrictInt | StrictFloat | None | tuple["IndexLabel", ...]

    @model_validator(mode="after")
    def typed_value(self):
        if self.type == "tuple":
            if not isinstance(self.value, tuple) or len(self.value) > 16:
                raise ValueError("Tuple index labels must have at most 16 typed components.")
        elif self.type in {"nan", "missing", "nat"}:
            if self.value is not None:
                raise ValueError("Missing index labels carry no value.")
        elif self.type in {"timestamp", "date"}:
            if not isinstance(self.value, str):
                raise ValueError("Date labels require their ISO string.")
        elif self.type == "timedelta":
            if type(self.value) is not int:
                raise ValueError("Timedelta index labels require integer nanoseconds.")
        elif isinstance(self.value, tuple):
            raise ValueError("Scalar labels cannot carry tuple descriptors.")
        if isinstance(self.value, str) and len(self.value) > 10_000:
            raise ValueError("Individual index labels are limited to 10,000 characters.")
        return self


class IndexState(FrozenModel):
    kind: Literal["multi", "range", "categorical", "datetime", "timedelta", "index"]
    name: IndexLabel | None = None
    names: tuple[IndexLabel, ...] | None = None
    levels: tuple["IndexState", ...] | None = None
    codes: tuple[StrictInt, ...] | tuple[tuple[StrictInt, ...], ...] | None = None
    start: StrictInt | None = None
    stop: StrictInt | None = None
    step: StrictInt | None = None
    categories: "IndexState | None" = None
    ordered: StrictBool | None = None
    dtype: str | None = None
    values: tuple[IndexLabel, ...] | None = None
    freq: str | None = None
    sortorder: StrictInt | None = None

    @model_serializer(mode="wrap")
    def sortorder_wire(self, handler):
        result = handler(self)
        # Omit only this new absent field. Historical None fields stay unchanged.
        if self.sortorder is None:
            result.pop("sortorder", None)
        return result

    def descriptor(self):
        fields = {
            "multi": ("kind", "names", "levels", "codes"),
            "range": ("kind", "name", "start", "stop", "step"),
            "categorical": ("kind", "name", "categories", "ordered", "codes"),
            "datetime": ("kind", "name", "dtype", "values", "freq"),
            "timedelta": ("kind", "name", "dtype", "values", "freq"),
            "index": ("kind", "name", "dtype", "values"),
        }[self.kind]
        value = self.model_dump(mode="json")
        result = {name: value[name] for name in fields}
        if self.kind == "multi":
            result["levels"] = [level.descriptor() for level in self.levels or ()]
            if self.sortorder is not None:
                result["sortorder"] = self.sortorder
        elif self.kind == "categorical" and self.categories is not None:
            result["categories"] = self.categories.descriptor()
        return result

    @model_validator(mode="after")
    def declared_shape(self):
        if self.sortorder is not None and (
                self.kind != "multi" or not 0 <= self.sortorder <= len(self.levels or ())):
            raise ValueError("MultiIndex sortorder must be in [0, nlevels].")
        relevant = set(self.descriptor())
        for name, value in self.__dict__.items():
            if name not in relevant and value is not None:
                raise ValueError("Index fields disagree with the declared index kind.")
        return self


class NormalInverseGammaPrior(FrozenModel):
    """Proper beta|sigma² ~ N(mean, sigma² scale_matrix), sigma² ~ IG(shape, scale).

    The IG density uses exp(-scale/sigma²), not a rate on sigma² itself.
    Coefficient order includes the intercept when present.
    """
    schema_version: Literal["openecon.normal_inverse_gamma_prior.v1"] = "openecon.normal_inverse_gamma_prior.v1"
    mean: tuple[float, ...]
    scale_matrix: tuple[tuple[float, ...], ...]
    shape: float = Field(gt=0, le=1e12)
    scale: float = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def envelope(cls, value):
        if isinstance(value, dict):
            means, covariance = value.get("mean", ()), value.get("scale_matrix", ())
            if not 1 <= len(means) <= 33 or len(covariance) != len(means) or any(len(row) != len(means) for row in covariance):
                raise ValueError("Prior dimensions must be square with 1 to 33 coefficients.")
            scalars = [value.get("shape"), value.get("scale"), *means,
                       *(v for row in covariance for v in row)]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in scalars):
                raise ValueError("Prior parameters must be real numbers, excluding booleans.")
        return value


class PosteriorState(FrozenModel):
    schema_version: Literal["openecon.bayesian_conjugate_state.v1"] = "openecon.bayesian_conjugate_state.v1"
    outcome: str
    predictors: tuple[str, ...]
    intercept: StrictBool
    missing: Literal["raise", "drop"]
    source_columns: tuple[str, ...]
    source_dtypes: tuple[str, ...]
    source_values: tuple[tuple[StrictInt | StrictFloat | None, ...], ...]
    source_index: IndexState
    sample_positions: tuple[StrictInt, ...]
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def declared_source_names(self):
        """Apply the source-name domain to both original fitting and restoration."""
        columns = (self.outcome, *self.predictors)
        if (len(self.predictors) > MAX_PREDICTORS
                or len(set(columns)) != len(columns)
                or "Intercept" in self.predictors
                or any(not name.strip() or len(name) > 1000 for name in columns)):
            raise ValueError("Source names must be nonempty, distinct and at most 1,000 characters; Intercept is reserved.")
        if self.source_columns != columns:
            raise ValueError("Saved source names disagree with the declared design.")
        return self


class PosteriorBundle(FrozenModel):
    """Analytic posterior; credible intervals and Bayesian probabilities have their own fields.

    model_validate/model_validate_json perform semantic source/posterior replay.
    Ordinary p-values, frequentist standard errors and MCMC diagnostics are absent.
    """
    schema_version: Literal["openecon.posterior_bundle.v1"] = "openecon.posterior_bundle.v1"
    method: Literal["proper_conjugate_gaussian"] = "proper_conjugate_gaussian"
    terms: tuple[str, ...]
    prior: NormalInverseGammaPrior
    state: PosteriorState
    mean: tuple[float, ...]
    conditional_scale_matrix: tuple[tuple[float, ...], ...]
    shape: float
    scale: float
    degrees_of_freedom: float
    coefficient_scale_matrix: tuple[tuple[float, ...], ...]
    coefficient_covariance: tuple[tuple[float, ...], ...] | None
    variance_mean: float | None
    log_marginal_likelihood: float
    alpha: float = Field(gt=0, lt=1)
    credible_intervals: tuple[tuple[float, float], ...]
    max_work: StrictInt = DEFAULT_MAX_WORK
    max_bytes: StrictInt = DEFAULT_MAX_BYTES
    source_rank: StrictInt
    integrity_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="before")
    @classmethod
    def bounded_envelope(cls, value):
        if not isinstance(value, dict):
            return value
        if set(value) - cls.model_fields.keys():
            raise ValueError("Extra inputs are not permitted in a saved posterior.")
        state = value.get("state", {})
        if not isinstance(state, dict):
            raise ValueError("Saved posterior state must be a typed mapping.")
        columns = state.get("source_values", ())
        if not 1 <= len(columns) <= 33:
            raise ValueError("Saved posterior source column count is unsupported.")
        n = len(columns[0])
        if n > MAX_ROWS or any(len(column) != n for column in columns):
            raise ValueError("Saved posterior source dimensions disagree.")
        k = len(value.get("terms", ()))
        admit(n, k, max_work=value.get("max_work", DEFAULT_MAX_WORK),
              max_bytes=value.get("max_bytes", DEFAULT_MAX_BYTES))
        _index_envelope(state.get("source_index"), n)
        size = metadata_bytes(value)
        plan_workspace("bayesian_saved_metadata", {"typed_metadata_and_validation_copy": 3 * size},
                       budget_bytes=min(value.get("max_bytes", DEFAULT_MAX_BYTES), workspace_budget_bytes()))
        for name in ("mean", "credible_intervals", "conditional_scale_matrix",
                     "coefficient_scale_matrix", "coefficient_covariance"):
            array = value.get(name)
            if array is not None and len(array) != k:
                raise ValueError(f"Saved {name} dimensions disagree with terms.")
            if array is not None and name != "mean":
                width = 2 if name == "credible_intervals" else k
                if any(len(row) != width for row in array):
                    raise ValueError(f"Saved {name} row dimensions disagree with terms.")
        return value

    @model_validator(mode="after")
    def replay(self):
        import torch
        from openecon.engines.distributions import t_isf

        with torch.device("cpu"):
            state = self.state
            k, n = len(self.terms), len(state.source_values[0])
            admit(n, k, max_work=self.max_work, max_bytes=self.max_bytes)
            if len(state.source_values) != len(state.source_columns) or any(len(column) != n for column in state.source_values):
                raise ValueError("Saved source dimensions disagree.")
            plan_workspace("bayesian_semantic_replay_metadata", {"typed_state_and_validation": 3*metadata_bytes(self)},
                           budget_bytes=min(self.max_bytes, workspace_budget_bytes()))
            columns = (state.outcome, *state.predictors)
            state.declared_source_names()
            if state.source_columns != columns or len(set(columns)) != len(columns):
                raise ValueError("Saved source names disagree with the declared design.")
            if any(not name.strip() for name in columns) or len(state.source_dtypes) != len(columns):
                raise ValueError("Saved source names or dtypes are invalid.")
            terms = (("Intercept",) if state.intercept else ()) + state.predictors
            if self.terms != terms or len(set(self.terms)) != k or k != len(self.prior.mean):
                raise ValueError("Saved coefficient order disagrees with the prior/design.")
            descriptor = state.source_index.descriptor()
            _index_envelope(descriptor, n)
            index = _decode_index(descriptor)
            if len(index) != n or _encode_index(index) != descriptor:
                raise ValueError("Saved typed index does not round trip exactly.")
            for values, dtype in zip(state.source_values, state.source_dtypes, strict=True):
                restored = pd.Series(values, dtype=dtype)
                from .core import source_values
                recaptured, _ = source_values(pd.DataFrame({"column": restored}), ["column"])
                if recaptured[0] != values:
                    raise ValueError("Saved typed source values do not round trip exactly.")
            positions = tuple(i for i in range(n) if all(column[i] is not None for column in state.source_values))
            if not positions or state.sample_positions != positions or (state.missing == "raise" and len(positions) != n):
                raise ValueError("Saved complete-case sample disagrees with the source/missing policy.")
            source = {"columns": state.source_columns, "dtypes": state.source_dtypes,
                      "values": state.source_values, "index": descriptor}
            if digest(source) != state.source_sha256:
                raise ValueError("Saved source integrity digest disagrees.")
            y, x = matrices(state.source_values, positions, intercept=state.intercept)
            computed = posterior_algebra(y, x, self.prior)
            for name, expected in computed.items():
                close(getattr(self, name), expected, name)
            rank = int(torch.linalg.matrix_rank(x))
            if self.source_rank != rank:
                raise ValueError("Saved source rank disagrees with the declared design.")
            critical = t_isf(self.alpha / 2, self.degrees_of_freedom)
            expected = [[mean - critical * math.sqrt(row[i]), mean + critical * math.sqrt(row[i])]
                        for i, (mean, row) in enumerate(zip(self.mean, self.coefficient_scale_matrix, strict=True))]
            close(self.credible_intervals, expected, "credible_intervals")
            body = self.model_dump(mode="json", exclude={"integrity_sha256"})
            if digest(body) != self.integrity_sha256:
                raise ValueError("Saved posterior integrity digest disagrees.")
        return self

    @property
    def nobs(self):
        return len(self.state.sample_positions)

    @property
    def nobs_original(self):
        return len(self.state.source_values[0])

    def summary(self):
        """Return a native table with separately labelled posterior uncertainty."""
        self = self._checked_summary()
        standard_deviations = ([math.sqrt(row[i]) for i, row in enumerate(self.coefficient_covariance)]
                               if self.coefficient_covariance is not None else [None] * len(self.terms))
        frame = pd.DataFrame({"Term": self.terms, "Posterior mean": self.mean,
                              "Posterior std. dev.": standard_deviations,
                              "Student-t scale": [math.sqrt(row[i]) for i, row in enumerate(self.coefficient_scale_matrix)],
                              "Credible lower": [row[0] for row in self.credible_intervals],
                              "Credible upper": [row[1] for row in self.credible_intervals]})
        frame.attrs["posterior_method"] = self.method
        frame.attrs["credible_probability"] = 1 - self.alpha
        frame.attrs["degrees_of_freedom"] = self.degrees_of_freedom
        return frame

    def to_latex(self, buf=None, **options):
        """Export posterior means/scales/credible limits without frequentist columns."""
        return self.summary().to_latex(buf=buf, index=False, **options)


def _contrast_algebra(posterior, weights, threshold, alpha):
    """Exact marginal linear target, shared by construction and semantic replay."""
    import torch
    from openecon.analysis_contracts import AnalysisError
    from openecon.engines.distributions import t_cdf, t_isf

    with torch.device("cpu"):
        w = torch.tensor(weights, dtype=torch.float64, device="cpu")
        mean = float(w @ torch.tensor(posterior.mean, dtype=torch.float64, device="cpu"))
        factor = torch.linalg.cholesky(torch.tensor(
            posterior.coefficient_scale_matrix, dtype=torch.float64, device="cpu"))
        scale_squared = float((w @ factor).square().sum())
        scale = math.sqrt(scale_squared)
        variance = None
        if posterior.coefficient_covariance is not None:
            factor = torch.linalg.cholesky(torch.tensor(
                posterior.coefficient_covariance, dtype=torch.float64, device="cpu"))
            variance = float((w @ factor).square().sum())
        critical = t_isf(alpha / 2, posterior.degrees_of_freedom)
        probability = (float(mean > threshold) if scale == 0
                       else t_cdf((mean - threshold) / scale, posterior.degrees_of_freedom))
        result = {"mean": mean, "student_t_scale": scale, "posterior_variance": variance,
                  "degrees_of_freedom": posterior.degrees_of_freedom,
                  "credible_lower": mean - critical * scale,
                  "credible_upper": mean + critical * scale,
                  "probability_above_threshold": probability}
        if any(v is not None and not math.isfinite(v) for v in result.values()):
            raise AnalysisError("numerical_failure", "Contrast output is not representable in float64.")
        return result


class PosteriorContrast(FrozenModel):
    """Saved analytic contrast with a complete, semantically validated target."""
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, revalidate_instances="always")
    schema_version: Literal["openecon.posterior_contrast.v1"] = "openecon.posterior_contrast.v1"
    posterior: PosteriorBundle
    posterior_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    terms: tuple[str, ...]
    weights: tuple[StrictFloat | StrictInt, ...]
    mean: StrictFloat
    student_t_scale: StrictFloat = Field(ge=0)
    posterior_variance: StrictFloat | None = Field(ge=0)
    degrees_of_freedom: StrictFloat = Field(gt=0)
    alpha: StrictFloat = Field(gt=0, lt=1)
    credible_lower: StrictFloat
    credible_upper: StrictFloat
    threshold: StrictFloat
    probability_above_threshold: StrictFloat = Field(ge=0, le=1)
    integrity_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="before")
    @classmethod
    def bounded_target(cls, value):
        if not isinstance(value, dict):
            return value
        if set(value) - cls.model_fields.keys():
            raise ValueError("Extra inputs are not permitted in a saved posterior contrast.")
        posterior = value.get("posterior")
        if isinstance(posterior, PosteriorBundle):
            n, k = posterior.nobs_original, len(posterior.terms)
            max_work, max_bytes = posterior.max_work, posterior.max_bytes
        elif isinstance(posterior, dict):
            source = posterior.get("state", {}).get("source_values", ())
            n, k = len(source[0]) if source else 0, len(posterior.get("terms", ()))
            max_work = posterior.get("max_work", DEFAULT_MAX_WORK)
            max_bytes = posterior.get("max_bytes", DEFAULT_MAX_BYTES)
        else:
            raise ValueError("A saved contrast requires its complete PosteriorBundle target.")
        admit(n, k, operation="bayesian_contrast_target", max_work=max_work, max_bytes=max_bytes)
        if len(value.get("weights", ())) != k or len(value.get("terms", ())) != k:
            raise ValueError("Saved contrast weights and terms must match the complete coefficient order.")
        plan_workspace("bayesian_saved_contrast_metadata", {"target_and_contrast_copy": 3 * metadata_bytes(value)},
                       budget_bytes=min(max_bytes, workspace_budget_bytes()))
        if isinstance(posterior, PosteriorBundle):
            # Typed model_copy/model_construct targets must pass the full source
            # and posterior validator before their numbers define the contrast.
            value = value | {"posterior": posterior.model_dump(mode="json")}
        return value

    @model_validator(mode="after")
    def replay(self):
        from .core import real

        posterior = self.posterior
        admit(posterior.nobs_original, len(posterior.terms), operation="bayesian_contrast_replay",
              max_work=posterior.max_work, max_bytes=posterior.max_bytes)
        plan_workspace("bayesian_contrast_semantic_metadata", {"target_and_validation_copy": 3 * metadata_bytes(self)},
                       budget_bytes=min(posterior.max_bytes, workspace_budget_bytes()))
        # Revalidate live constructed/copied contrasts too, which skip the raw
        # mapping envelope. No fit, chain or ambient random generator is used.
        posterior = PosteriorBundle.model_validate(posterior.model_dump(mode="json"))
        if self.posterior_sha256 != posterior.integrity_sha256:
            raise ValueError("Saved contrast target digest disagrees with the complete posterior.")
        if self.terms != posterior.terms or len(self.weights) != len(posterior.terms):
            raise ValueError("Saved contrast weights disagree with the named coefficient order.")
        weights = tuple(real(v, "saved contrast weight") for v in self.weights)
        alpha = real(self.alpha, "saved contrast alpha", low=0, high=1)
        threshold = real(self.threshold, "saved contrast threshold")
        for name, expected in _contrast_algebra(posterior, weights, threshold, alpha).items():
            if getattr(self, name) is not None:
                real(getattr(self, name), f"saved contrast {name}")
            close(getattr(self, name), expected, f"contrast.{name}")
        if digest(self.model_dump(mode="json", exclude={"integrity_sha256"})) != self.integrity_sha256:
            raise ValueError("Saved contrast integrity digest disagrees.")
        return self

    def summary(self):
        """Return the saved posterior contrast mean, t scale, credible limits and tail probability."""
        self = self._checked_summary()
        return pd.DataFrame([{"Posterior mean": self.mean, "Student-t scale": self.student_t_scale,
                              "Credible lower": self.credible_lower, "Credible upper": self.credible_upper,
                              "Probability above threshold": self.probability_above_threshold}])


class PosteriorDraws(FrozenModel):
    """Bounded independent posterior draws with a saved, reproducible target and seed."""
    schema_version: Literal["openecon.posterior_draws.v1"] = "openecon.posterior_draws.v1"
    posterior: PosteriorBundle
    seed: StrictInt
    beta: tuple[tuple[float, ...], ...]
    sigma_squared: tuple[float, ...]
    query_design: tuple[tuple[float, ...], ...] | None = None
    query_index: IndexState | None = None
    mean_draws: tuple[tuple[float, ...], ...] | None = None
    outcome_draws: tuple[tuple[float, ...], ...] | None = None
    integrity_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="before")
    @classmethod
    def admission(cls, value):
        if isinstance(value, dict):
            if set(value) - cls.model_fields.keys():
                raise ValueError("Extra inputs are not permitted in saved posterior draws.")
            posterior = value.get("posterior", {})
            if isinstance(posterior, PosteriorBundle):
                admit(posterior.nobs_original, len(posterior.terms),
                      max_work=posterior.max_work, max_bytes=posterior.max_bytes)
                plan_workspace("bayesian_draw_target_copy", {"target_serialization": 3*metadata_bytes(posterior)},
                               budget_bytes=min(posterior.max_bytes, workspace_budget_bytes()))
            p = posterior.model_dump(mode="json") if isinstance(posterior, PosteriorBundle) else posterior
            k, draws = len(p.get("terms", ())), len(value.get("beta", ()))
            queries = len(value.get("query_design") or ())
            admit(len(p.get("state", {}).get("source_values", [()])[0]), k,
                  operation="bayesian_draw_state", draws=draws, queries=queries,
                  max_work=p.get("max_work", DEFAULT_MAX_WORK), max_bytes=p.get("max_bytes", DEFAULT_MAX_BYTES))
            if len(value.get("sigma_squared", ())) != draws or not draws:
                raise ValueError("Saved draw dimensions disagree.")
            if any(len(row) != k for row in value.get("beta", ())):
                raise ValueError("Saved draw coefficient dimensions disagree.")
            if any(len(row) != k for row in value.get("query_design") or ()):
                raise ValueError("Saved query design dimensions disagree.")
            for name in ("mean_draws", "outcome_draws"):
                array = value.get(name)
                if (array is None) != (value.get("query_design") is None):
                    raise ValueError("Saved predictive arrays disagree with query availability.")
                if array is not None and (len(array) != draws or any(len(row) != queries for row in array)):
                    raise ValueError("Saved predictive draw dimensions disagree.")
            if value.get("query_index") is not None:
                index = value["query_index"]
                _index_envelope(index.descriptor() if isinstance(index, IndexState) else index, queries)
            size = metadata_bytes(value)
            plan_workspace("bayesian_saved_draw_metadata", {"typed_metadata_and_validation_copy": 3*size},
                           budget_bytes=min(p.get("max_bytes", DEFAULT_MAX_BYTES), workspace_budget_bytes()))
        return value

    @model_validator(mode="after")
    def replay(self):
        from .commands import _draw_arrays
        from .core import integer

        integer(self.seed, "seed", low=0)
        queries = len(self.query_design or ())
        admit(self.posterior.nobs_original, len(self.posterior.terms), draws=len(self.beta), queries=queries,
              max_work=self.posterior.max_work, max_bytes=self.posterior.max_bytes)
        plan_workspace("bayesian_draw_semantic_metadata", {"typed_draw_validation": 3*metadata_bytes(self)},
                       budget_bytes=min(self.posterior.max_bytes, workspace_budget_bytes()))
        if any(len(row) != len(self.posterior.terms) for row in self.beta):
            raise ValueError("Saved draw coefficient dimensions disagree.")
        if (self.query_design is None) != (self.query_index is None):
            raise ValueError("Saved predictive draws require both query design and index.")
        if self.query_design is not None:
            if any(len(row) != len(self.posterior.terms) for row in self.query_design):
                raise ValueError("Saved query design dimensions disagree.")
            if len(_decode_index(self.query_index.descriptor())) != len(self.query_design):
                raise ValueError("Saved query index length disagrees.")
        expected = _draw_arrays(self.posterior, len(self.beta), self.seed, self.query_design)
        for name in ("beta", "sigma_squared", "mean_draws", "outcome_draws"):
            close(getattr(self, name), expected[name], name)
        if digest(self.model_dump(mode="json", exclude={"integrity_sha256"})) != self.integrity_sha256:
            raise ValueError("Saved draw integrity digest disagrees.")
        return self

    def summary(self):
        """Return the exact joint posterior coefficient and variance draws in saved seed order."""
        self = self._checked_summary()
        frame = pd.DataFrame(self.beta, columns=self.posterior.terms)
        frame["sigma_squared"] = self.sigma_squared
        frame.attrs.update({"draw_kind": "independent analytic posterior draws", "seed": self.seed,
                            "posterior_sha256": self.posterior.integrity_sha256})
        return frame
