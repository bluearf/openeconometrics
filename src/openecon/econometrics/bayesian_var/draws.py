"""Joint IID MNIW draws via native Bartlett factors; every draw is retained."""

from __future__ import annotations

from pydantic import model_validator

from openecon.analysis_contracts import AnalysisError

from .admission import admit, digest, integer, metadata_admit
from .posterior import (
    BayesianVARPosterior,
    FrozenModel,
    Matrix,
    Vector,
    admit_parent,
    matrix,
    raw,
    restore,
)


def primitive_stream(parent, draws, seed):
    import torch

    m, k = len(parent.source.series), len(parent.terms)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    factors, normals = [], []
    shapes = torch.tensor(
        [(parent.algebra.degrees_of_freedom - i) / 2 for i in range(m)], dtype=torch.float64
    )
    for _ in range(draws):
        a = torch.zeros((m, m), dtype=torch.float64)
        diagonal = (2 * torch._standard_gamma(shapes, generator=generator)).sqrt()
        a.diagonal().copy_(diagonal)
        for i in range(1, m):
            a[i, :i] = torch.randn(i, dtype=torch.float64, generator=generator)
        z = torch.randn((k, m), dtype=torch.float64, generator=generator)
        if not bool(torch.isfinite(a).all()) or not bool((diagonal > 0).all()):
            raise AnalysisError(
                "numeric_failure",
                "A Bartlett primitive overflowed/underflowed; no resampling is applied.",
            )
        factors.append(a.tolist())
        normals.append(z.tolist())
    return factors, normals


def companion(coefficient, m, p, intercept):
    import torch

    out = torch.zeros((m * p, m * p), dtype=torch.float64)
    lag = coefficient[int(intercept) :]
    out[:m] = torch.cat([lag[j * m : (j + 1) * m].T for j in range(p)], dim=1)
    if p > 1:
        out[m:, :-m] = torch.eye(m * (p - 1), dtype=torch.float64)
    return out


def transform(parent, factors, normals):
    import torch

    from .kernels import gram, spd, tensor

    m, p = len(parent.source.series), parent.lags
    lv = spd(tensor(parent.algebra.row_scale), "posterior row scale")
    ls = spd(tensor(parent.algebra.innovation_scale), "posterior innovation scale")
    mean = tensor(parent.algebra.location)
    coefficients, innovations, radii, stable = [], [], [], []
    for a_values, z_values in zip(factors, normals, strict=True):
        a, z = tensor(a_values), tensor(z_values)
        if not torch.equal(a, torch.tril(a)) or not bool((a.diagonal() > 0).all()):
            raise AnalysisError(
                "invalid_state",
                "Saved Bartlett factors must be lower triangular with positive diagonals.",
            )
        c = torch.linalg.solve_triangular(a, ls.T, upper=False).T
        sigma = gram(c.T)
        b = mean + lv @ z @ c.T
        if not bool(torch.isfinite(sigma).all()) or not bool(torch.isfinite(b).all()):
            raise AnalysisError(
                "numeric_failure",
                "A retained joint MNIW transform overflowed; no clipping/resampling is applied.",
            )
        try:
            radius = float(torch.linalg.eigvals(companion(b, m, p, parent.intercept)).abs().max())
        except RuntimeError as exc:
            raise AnalysisError(
                "numeric_failure", "Stability classification failed; no draw is dropped."
            ) from exc
        coefficients.append(b.tolist())
        innovations.append(sigma.tolist())
        radii.append(radius)
        stable.append(radius < 1)
    return {
        "coefficients": coefficients,
        "innovations": innovations,
        "spectral_radii": radii,
        "stable": stable,
    }


class BayesianVARDraws(FrozenModel):
    schema_version: str = "openecon.bayesian_var_joint_draws.v1"
    parent: BayesianVARPosterior
    seed: int
    draws: int
    algorithm: str = "native_cpu_float64_gamma_bartlett_mniw_v1"
    torch_version: str
    bartlett: tuple[Matrix, ...]
    standard_normals: tuple[Matrix, ...]
    coefficients: tuple[Matrix, ...]
    innovations: tuple[Matrix, ...]
    spectral_radii: Vector
    stable: tuple[bool, ...]
    integrity_sha256: str

    @model_validator(mode="before")
    @classmethod
    def preflight(cls, value):
        value = raw(value)
        metadata_admit(value)
        parent = raw(value["parent"])
        source = raw(parent["source"])
        m, k, count = (
            len(source["series"]),
            len(parent["terms"]),
            integer(value["draws"], "draws", low=1, high=10000),
        )
        integer(value["seed"], "seed")
        admit(
            len(source["permutation"]),
            m,
            parent["lags"],
            parent["intercept"],
            draws=count,
            max_work=parent["max_work"],
            max_bytes=parent["max_bytes"],
        )
        for field, rows, columns in (
            ("bartlett", m, m),
            ("standard_normals", k, m),
            ("coefficients", k, m),
            ("innovations", m, m),
        ):
            if len(value[field]) != count:
                raise AnalysisError("invalid_state", "Every declared joint draw must be retained.")
            for item in value[field]:
                matrix(item, field, rows, columns)
        if (
            len(value["stable"]) != count
            or any(type(v) is not bool for v in value["stable"])
            or len(value["spectral_radii"]) != count
        ):
            raise AnalysisError(
                "invalid_state", "Every draw requires its explicit stability classification."
            )
        return value

    @model_validator(mode="after")
    def replay(self):
        from .kernels import close

        if (
            self.schema_version != "openecon.bayesian_var_joint_draws.v1"
            or self.algorithm != "native_cpu_float64_gamma_bartlett_mniw_v1"
        ):
            raise AnalysisError("invalid_state", "Unknown joint-draw schema/algorithm.")
        expected = transform(self.parent, self.bartlett, self.standard_normals)
        body = raw(self)
        if (
            any(not close(body[key], values) for key, values in expected.items() if key != "stable")
            or tuple(expected["stable"]) != self.stable
        ):
            raise AnalysisError(
                "invalid_state",
                "Joint draw arrays/stability disagree with recorded primitive replay.",
            )
        # Cached primitive replay is portable; the recorded RNG runtime is descriptive.
        body.pop("integrity_sha256")
        if digest(body) != self.integrity_sha256:
            raise AnalysisError("invalid_state", "Joint draw digest disagrees.")
        return self


def bayes_var_draws(result, *, draws, seed):
    import torch

    count, seed = integer(draws, "draws", low=1, high=10000), integer(seed, "seed")
    parent = restore(admit_parent(result, draws=count))
    admit(
        parent.nobs_original,
        len(parent.source.series),
        parent.lags,
        parent.intercept,
        draws=count,
        max_work=parent.max_work,
        max_bytes=parent.max_bytes,
    )
    factors, normals = primitive_stream(parent, count, seed)
    body = {
        "schema_version": "openecon.bayesian_var_joint_draws.v1",
        "parent": raw(parent),
        "seed": seed,
        "draws": count,
        "algorithm": "native_cpu_float64_gamma_bartlett_mniw_v1",
        "torch_version": str(torch.__version__),
        "bartlett": factors,
        "standard_normals": normals,
        **transform(parent, factors, normals),
    }
    body["integrity_sha256"] = digest(body)
    return BayesianVARDraws.model_validate(body)


def restore_draws(value):
    from .admission import load_mapping

    try:
        return BayesianVARDraws.model_validate(
            raw(value) if isinstance(value, FrozenModel) else load_mapping(value)
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError(
            "invalid_state", "Saved BVAR draws fail complete typed primitive replay."
        ) from exc
