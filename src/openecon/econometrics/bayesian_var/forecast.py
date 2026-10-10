"""Finite-horizon joint paths using one retained parameter draw per entire path."""

from __future__ import annotations

from pydantic import StrictBool, StrictInt, model_validator

from openecon.analysis_contracts import AnalysisError

from .admission import admit, digest, integer, load_mapping, metadata_admit, real
from .draws import BayesianVARDraws, restore_draws
from .moments import summarize
from .posterior import FrozenModel, Matrix, Vector, admit_parent, matrix, raw


class MomentAvailability(FrozenModel):
    mean: StrictBool
    second: StrictBool
    fourth: StrictBool
    mean_threshold: StrictInt | None
    second_threshold: StrictInt | None
    fourth_threshold: StrictInt | None


class MonteCarloSummary(FrozenModel):
    moment_availability: MomentAvailability
    sample_mean: Vector | None
    pointwise_quantiles: Matrix
    covariance_mc_estimate: Matrix | None
    mean_mc_standard_error: Vector | None
    covariance_mc_standard_error: Matrix | None
    covariance_label: str
    mc_error_label: str
    interval_label: str


def innovation_stream(count, horizon, m, seed):
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((count, horizon, m), dtype=torch.float64, generator=generator).tolist()


def paths_algebra(draws, normals, horizon):
    import torch

    from .design import matrices
    from .kernels import spd, tensor

    parent = draws.parent
    _, _, initial = matrices(raw(parent.source), parent.lags, parent.intercept)
    mean_paths, outcome_paths = [], []
    for bv, sv, zv in zip(draws.coefficients, draws.innovations, normals, strict=True):
        b, c, z = tensor(bv), spd(tensor(sv), "draw innovation covariance"), tensor(zv)
        deterministic, stochastic = list(initial.unbind()), list(initial.unbind())
        this_mean, this_outcome = [], []
        for h in range(horizon):
            regressors = []
            for history in (deterministic, stochastic):
                pieces = ([torch.ones(1, dtype=torch.float64)] if parent.intercept else []) + [
                    history[-lag] for lag in range(1, parent.lags + 1)
                ]
                regressors.append(torch.cat(pieces))
            ym, ys = regressors[0] @ b, regressors[1] @ b + c @ z[h]
            if not bool(torch.isfinite(ym).all()) or not bool(torch.isfinite(ys).all()):
                raise AnalysisError(
                    "numeric_failure",
                    "A finite-horizon retained path overflowed; no clipping/draw deletion.",
                )
            deterministic.append(ym)
            stochastic.append(ys)
            this_mean.append(ym.tolist())
            this_outcome.append(ys.tolist())
        mean_paths.append(this_mean)
        outcome_paths.append(this_outcome)
    return mean_paths, outcome_paths


class BayesianVARForecast(FrozenModel):
    schema_version: str = "openecon.bayesian_var_forecast.v1"
    joint_draws: BayesianVARDraws
    horizon: int
    alpha: float
    innovation_seed: int
    innovation_torch_version: str
    innovation_algorithm: str = "native_cpu_float64_standard_normal_v1"
    future_periods: tuple[int, ...]
    standard_innovations: tuple[Matrix, ...]
    conditional_mean_paths: tuple[Matrix, ...]
    outcome_paths: tuple[Matrix, ...]
    conditional_mean_summary: MonteCarloSummary
    outcome_summary: MonteCarloSummary
    integrity_sha256: str

    @model_validator(mode="before")
    @classmethod
    def preflight(cls, value):
        value = raw(value)
        metadata_admit(value)
        draws = raw(value["joint_draws"])
        parent = raw(draws["parent"])
        source = raw(parent["source"])
        m, h, count = (
            len(source["series"]),
            integer(value["horizon"], "horizon", low=1, high=24),
            draws["draws"],
        )
        integer(value["innovation_seed"], "innovation seed")
        if not 0 < real(value["alpha"], "alpha") < 1:
            raise AnalysisError("invalid_option", "alpha must lie strictly between zero and one.")
        admit(
            len(source["permutation"]),
            m,
            parent["lags"],
            parent["intercept"],
            draws=count,
            horizon=h,
            max_work=parent["max_work"],
            max_bytes=parent["max_bytes"],
        )
        for field in ("standard_innovations", "conditional_mean_paths", "outcome_paths"):
            if len(value[field]) != count:
                raise AnalysisError("invalid_state", "Forecast must retain every joint draw path.")
            for item in value[field]:
                matrix(item, field, h, m)
        return value

    @model_validator(mode="after")
    def replay(self):
        from .kernels import close

        parent = self.joint_draws.parent
        last = max(parent.source.source_values[0])
        if (
            self.schema_version != "openecon.bayesian_var_forecast.v1"
            or self.innovation_algorithm != "native_cpu_float64_standard_normal_v1"
            or self.future_periods != tuple(last + h for h in range(1, self.horizon + 1))
        ):
            raise AnalysisError("invalid_state", "Forecast schema/future calendar disagrees.")
        mean, outcome = paths_algebra(self.joint_draws, self.standard_innovations, self.horizon)
        if not close(self.conditional_mean_paths, mean) or not close(self.outcome_paths, outcome):
            raise AnalysisError(
                "invalid_state", "Forecast paths disagree with complete primitive replay."
            )
        for values, stored in (
            (mean, self.conditional_mean_summary),
            (outcome, self.outcome_summary),
        ):
            expected = summarize(
                values,
                parent.algebra.degrees_of_freedom,
                len(parent.source.series),
                self.horizon,
                self.alpha,
            )
            actual = raw(stored)
            if digest(actual) != digest(expected):
                raise AnalysisError(
                    "invalid_state", "Forecast moment availability/MC summaries disagree."
                )
        body = raw(self)
        body.pop("integrity_sha256")
        if digest(body) != self.integrity_sha256:
            raise AnalysisError("invalid_state", "Forecast digest disagrees.")
        return self


def bayes_var_forecast(joint_draws, *, horizon, innovation_seed, alpha=0.05):
    import torch

    h, seed, alpha = (
        integer(horizon, "horizon", low=1, high=24),
        integer(innovation_seed, "innovation seed"),
        real(alpha, "alpha"),
    )
    if not 0 < alpha < 1:
        raise AnalysisError("invalid_option", "alpha must lie strictly between zero and one.")
    saved = raw(joint_draws) if isinstance(joint_draws, FrozenModel) else load_mapping(joint_draws)
    admit_parent(saved["parent"], draws=saved["draws"], horizon=h)
    draws = restore_draws(saved)
    parent = draws.parent
    m = len(parent.source.series)
    admit(
        parent.nobs_original,
        m,
        parent.lags,
        parent.intercept,
        draws=draws.draws,
        horizon=h,
        max_work=parent.max_work,
        max_bytes=parent.max_bytes,
    )
    last = max(parent.source.source_values[0])
    future = tuple(
        integer(last + step, "future period", low=-(2**63), high=2**63 - 1)
        for step in range(1, h + 1)
    )
    z = innovation_stream(draws.draws, h, m, seed)
    mean, outcome = paths_algebra(draws, z, h)
    body = {
        "schema_version": "openecon.bayesian_var_forecast.v1",
        "joint_draws": raw(draws),
        "horizon": h,
        "alpha": alpha,
        "innovation_seed": seed,
        "innovation_torch_version": str(torch.__version__),
        "innovation_algorithm": "native_cpu_float64_standard_normal_v1",
        "future_periods": future,
        "standard_innovations": z,
        "conditional_mean_paths": mean,
        "outcome_paths": outcome,
        "conditional_mean_summary": summarize(mean, parent.algebra.degrees_of_freedom, m, h, alpha),
        "outcome_summary": summarize(outcome, parent.algebra.degrees_of_freedom, m, h, alpha),
    }
    body["integrity_sha256"] = digest(body)
    return BayesianVARForecast.model_validate(body)
