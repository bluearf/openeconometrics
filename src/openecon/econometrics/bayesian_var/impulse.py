"""Joint finite-lag simple/ordered-Cholesky IRFs, retaining unstable draws."""

from __future__ import annotations

from pydantic import model_validator

from openecon.analysis_contracts import AnalysisError

from .admission import admit, digest, integer, load_mapping, metadata_admit, real
from .draws import BayesianVARDraws, restore_draws
from .forecast import MonteCarloSummary
from .moments import summarize
from .posterior import FrozenModel, Matrix, admit_parent, matrix, raw


def impulse_algebra(draws, horizon, orthogonalized):
    import torch

    from .kernels import spd, tensor

    parent = draws.parent
    m, p = len(parent.source.series), parent.lags
    paths = []
    for bv, sv in zip(draws.coefficients, draws.innovations, strict=True):
        b = tensor(bv)[int(parent.intercept) :]
        coefficients = [b[j * m : (j + 1) * m].T for j in range(p)]
        response = [torch.eye(m, dtype=torch.float64)]
        impact = spd(tensor(sv), "draw innovation covariance") if orthogonalized else response[0]
        for h in range(1, horizon + 1):
            phi = torch.zeros((m, m), dtype=torch.float64)
            for lag in range(1, min(h, p) + 1):
                phi += coefficients[lag - 1] @ response[h - lag]
            response.append(phi)
        transformed = [phi @ impact for phi in response]
        if any(not bool(torch.isfinite(phi).all()) for phi in transformed):
            raise AnalysisError(
                "numeric_failure",
                "A retained impulse-response draw overflowed; no clipping/deletion.",
            )
        paths.append([phi.tolist() for phi in transformed])
    return paths


class BayesianVARImpulse(FrozenModel):
    schema_version: str = "openecon.bayesian_var_impulse.v1"
    joint_draws: BayesianVARDraws
    horizon: int
    orthogonalized: bool
    alpha: float
    series_order: tuple[str, ...]
    responses: tuple[tuple[Matrix, ...], ...]
    summary: MonteCarloSummary
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
            integer(value["horizon"], "horizon", high=24),
            draws["draws"],
        )
        if type(value["orthogonalized"]) is not bool or not 0 < real(value["alpha"], "alpha") < 1:
            raise AnalysisError(
                "invalid_option",
                "Declare boolean orthogonalized and alpha strictly between zero and one.",
            )
        admit(
            len(source["permutation"]),
            m,
            parent["lags"],
            parent["intercept"],
            draws=count,
            horizon=h,
            irf=True,
            max_work=parent["max_work"],
            max_bytes=parent["max_bytes"],
        )
        if len(value["responses"]) != count:
            raise AnalysisError("invalid_state", "IRFs must retain every joint draw.")
        for response in value["responses"]:
            if len(response) != h + 1:
                raise AnalysisError(
                    "invalid_state", "Every IRF retains lag zero through the declared horizon."
                )
            for item in response:
                matrix(item, "IRF response", m, m)
        return value

    @model_validator(mode="after")
    def replay(self):
        from .kernels import close

        parent = self.joint_draws.parent
        expected = impulse_algebra(self.joint_draws, self.horizon, self.orthogonalized)
        if (
            self.schema_version != "openecon.bayesian_var_impulse.v1"
            or self.series_order != parent.source.series
            or not close(self.responses, expected)
        ):
            raise AnalysisError(
                "invalid_state", "IRF schema/order/complete response arrays disagree."
            )
        summary = summarize(
            expected,
            parent.algebra.degrees_of_freedom,
            len(parent.source.series),
            self.horizon,
            self.alpha,
            orthogonalized=self.orthogonalized,
            impulse=True,
        )
        if digest(raw(self.summary)) != digest(summary):
            raise AnalysisError("invalid_state", "IRF moment availability/MC summaries disagree.")
        body = raw(self)
        body.pop("integrity_sha256")
        if digest(body) != self.integrity_sha256:
            raise AnalysisError("invalid_state", "IRF digest disagrees.")
        return self


def bayes_var_irf(joint_draws, *, horizon, orthogonalized=False, alpha=0.05):
    h, alpha = integer(horizon, "horizon", high=24), real(alpha, "alpha")
    if type(orthogonalized) is not bool or not 0 < alpha < 1:
        raise AnalysisError(
            "invalid_option",
            "Declare boolean orthogonalized and alpha strictly between zero and one.",
        )
    saved = raw(joint_draws) if isinstance(joint_draws, FrozenModel) else load_mapping(joint_draws)
    admit_parent(saved["parent"], draws=saved["draws"], horizon=h, irf=True)
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
        irf=True,
        max_work=parent.max_work,
        max_bytes=parent.max_bytes,
    )
    responses = impulse_algebra(draws, h, orthogonalized)
    body = {
        "schema_version": "openecon.bayesian_var_impulse.v1",
        "joint_draws": raw(draws),
        "horizon": h,
        "orthogonalized": orthogonalized,
        "alpha": alpha,
        "series_order": parent.source.series,
        "responses": responses,
        "summary": summarize(
            responses,
            parent.algebra.degrees_of_freedom,
            m,
            h,
            alpha,
            orthogonalized=orthogonalized,
            impulse=True,
        ),
    }
    body["integrity_sha256"] = digest(body)
    return BayesianVARImpulse.model_validate(body)
