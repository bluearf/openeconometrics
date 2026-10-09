"""Saved Rubin pooling and D1 linear joint tests, native float64 Torch.

D1 uses the proportional-between/within covariance approximation of Li,
Raghunathan & Rubin (1991). Finite-complete-df inference uses Reiter (2007)
eqs 1-2 only in their positive-moment domain. It is not a pooled likelihood
ratio or an arbitrary small-sample exact test.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.mi.pooling import _from_bundles, _pool_tables
from openecon.econometrics.postest.multiple import _labels, _level
from openecon.engines.distributions import f_sf
from openecon.models import ResultBundle
from openecon.resources import plan_workspace, workspace_budget_bytes

FLOAT = torch.float64
MAX_M = 100
MAX_P = 32


def digest(state):
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def _shape(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.is_complex() or value.dtype == torch.bool:
            raise AnalysisError("invalid_imputations", "Pooling accepts real CPU tensors only.")
        return tuple(value.shape)
    shape = getattr(value, "shape", None)
    if shape is not None:
        kind = getattr(getattr(value, "dtype", None), "kind", None)
        if kind not in ("i", "u", "f"):
            raise AnalysisError(
                "invalid_imputations",
                "Pooling arrays must be real numeric, not boolean/object/complex.",
            )
        return tuple(shape)
    if not isinstance(value, (list, tuple)):
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise AnalysisError("invalid_imputations", "Supply real numeric rectangular arrays.")
        return ()
    if not value or len(value) > MAX_M:
        raise AnalysisError(
            "work_budget", "Pooling array dimensions must be nonempty and at most 100."
        )
    shapes = [_shape(v) for v in value]
    if any(s != shapes[0] for s in shapes):
        raise AnalysisError("invalid_imputations", "Pooling arrays must be rectangular.")
    return (len(value), *shapes[0])


def tensors(estimates, covariances):
    qs, us = _shape(estimates), _shape(covariances)
    if len(qs) != 2 or not 2 <= qs[0] <= MAX_M or not 1 <= qs[1] <= MAX_P or us != (*qs, qs[1]):
        raise AnalysisError(
            "invalid_imputations",
            "Need 2..100 imputations, 1..32 terms and matching full covariance matrices.",
        )
    m, p = qs
    plan = plan_workspace(
        "multiple-imputation pooling",
        {
            "input/result/serialization tensors": 96 * m * p * (p + 1),
            "pooling and joint matrix workspace": 256 * p * p,
        },
        budget_bytes=workspace_budget_bytes(),
    )
    q = torch.as_tensor(estimates, dtype=FLOAT, device="cpu")
    u = torch.as_tensor(covariances, dtype=FLOAT, device="cpu")
    if not torch.isfinite(q).all() or not torch.isfinite(u).all():
        raise AnalysisError("invalid_imputations", "All estimates and covariances must be finite.")
    scales = u.abs().amax((1, 2))
    if (scales <= 0).any() or not torch.allclose(
        u, u.transpose(1, 2), rtol=1e-12, atol=float(scales.min()) * 1e-12
    ):
        raise AnalysisError("invalid_covariance", "Every covariance must be symmetric and nonzero.")
    if (u.diagonal(dim1=1, dim2=2) <= 0).any() or (
        torch.linalg.eigvalsh(u / scales[:, None, None]) < -1e-12
    ).any():
        raise AnalysisError(
            "invalid_covariance",
            "Every covariance needs positive marginals and must be positive semidefinite.",
        )
    mean, within = q.mean(0), u.mean(0)
    centered = q - mean
    between = centered.T @ centered / (m - 1)
    total = within + (1 + 1 / m) * between
    if not all(torch.isfinite(x).all() for x in (mean, within, between, total)):
        raise AnalysisError(
            "numerical_failure", "Pooling overflowed; rescale the common estimand explicitly."
        )
    return q, u, mean, within, between, total, plan.record()


class MIPoolResult(BaseModel):
    """Immutable inputs and full-covariance reproducible Rubin inference state."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["mi-pool-v1"] = "mi-pool-v1"
    estimates: tuple[tuple[float, ...], ...] = Field(min_length=2, max_length=MAX_M)
    covariances: tuple[tuple[tuple[float, ...], ...], ...] = Field(min_length=2, max_length=MAX_M)
    terms: tuple[str, ...] = Field(min_length=1, max_length=MAX_P)
    imputation_ids: tuple[str, ...]
    complete_df: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    alpha: float = Field(default=0.05, gt=0, lt=1, allow_inf_nan=False)
    imputation_description: str = Field(min_length=1, max_length=8192)
    metadata_json: str = Field(max_length=131072)
    integrity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def raw_admission(cls, state):
        if isinstance(state, dict):
            tensors(state.get("estimates"), state.get("covariances"))
            _level(state.get("alpha", 0.05))
            df = state.get("complete_df")
            if df is not None and (
                isinstance(df, bool)
                or not isinstance(df, (int, float))
                or not math.isfinite(df)
                or df <= 0
            ):
                raise ValueError("Saved complete_df must be finite positive numeric, not boolean.")
            for key in ("terms", "imputation_ids"):
                values = state.get(key)
                if not isinstance(values, (list, tuple)) or any(
                    not isinstance(v, str) or not v.strip() for v in values
                ):
                    raise ValueError("Saved terms/IDs need nonempty string labels.")
        return state

    @model_validator(mode="after")
    def valid(self):
        q, _, *_ = tensors(self.estimates, self.covariances)
        _labels(list(self.terms), q.shape[1])
        _labels(list(self.imputation_ids), q.shape[0])
        if not self.imputation_description.strip():
            raise ValueError("The imputation design must be nonempty.")
        metadata = json.loads(self.metadata_json)
        if (
            not isinstance(metadata, dict)
            or metadata.get("precision") != "float64"
            or metadata.get("stata_parity_validated") is not False
        ):
            raise ValueError("Pooling metadata needs its explicit precision and evidence domain.")
        if (
            digest(self.model_dump(mode="json", exclude={"integrity_sha256"}))
            != self.integrity_sha256
        ):
            raise ValueError("MI pooling state integrity mismatch.")
        return self

    @property
    def metadata(self):
        return json.loads(self.metadata_json)

    @property
    def tables(self):
        return _pool_tables(
            self.estimates,
            self.covariances,
            terms=list(self.terms),
            imputation_ids=list(self.imputation_ids),
            complete_df=self.complete_df,
            alpha=self.alpha,
            imputation_description=self.imputation_description,
        )

    @property
    def table(self):
        return self.tables["coefficients"]

    @property
    def covariance(self):
        return tensors(self.estimates, self.covariances)[5].tolist()

    def to_latex(self, **options):
        return self.tables.to_latex(**options)

    def __str__(self):
        return str(self.tables)

    def model_copy(self, *, update=None, deep=False):
        """Copies revalidate full scientific state and integrity, including updates."""
        state = self.model_dump(mode="json")
        state.update(update or {})
        return type(self).model_validate(state)


def mi_pool(
    estimates,
    covariances=None,
    *,
    terms=None,
    imputation_ids=None,
    complete_df=None,
    alpha=0.05,
    imputation_description=None,
) -> MIPoolResult:
    """Pool compatible coefficients/full covariance with Rubin/Barnard-Rubin rules.

    Resident numeric arrays or compatible restored ResultBundles only, CPU
    float64, m=2..100/p=1..32. Describe imputation and the common estimand.
    A finite complete_df selects Barnard-Rubin marginal t inference; None uses
    large-complete-sample inference. The saved result retains all per-imputation
    estimates, full covariance, IDs and source sample fingerprints for mi_test.
    This procedure never pools p-values or silently changes model/sample rows.
    """
    _level(alpha)
    if (
        not isinstance(imputation_description, str)
        or not imputation_description.strip()
        or len(imputation_description) > 8192
    ):
        raise AnalysisError(
            "missing_imputation_design",
            "Describe the scientifically justified imputation procedure and common estimand (1..8192 characters).",
        )
    if complete_df is not None and (
        isinstance(complete_df, bool)
        or not isinstance(complete_df, (int, float))
        or not math.isfinite(complete_df)
        or complete_df <= 0
    ):
        raise AnalysisError("invalid_df", "complete_df must be finite and positive or None.")
    sources = []
    if (
        isinstance(estimates, (list, tuple))
        and estimates
        and isinstance(estimates[0], ResultBundle)
    ):
        if len(estimates) > MAX_M:
            raise AnalysisError("work_budget", "Pooling admits at most 100 fitted results.")
        if covariances is not None or terms is not None:
            raise AnalysisError(
                "invalid_imputations",
                "Fitted-result pooling reads terms and covariance from results.",
            )
        estimates, covariances, terms, complete_df, sources = _from_bundles(estimates, complete_df)
        if imputation_ids is None:
            imputation_ids = [s["id"] for s in sources]
    q, u, *_, resource = tensors(estimates, covariances)
    names = _labels(
        [f"term-{i + 1}" for i in range(q.shape[1])] if terms is None else terms, q.shape[1]
    )
    ids = _labels(
        [f"imputation-{i + 1}" for i in range(q.shape[0])]
        if imputation_ids is None
        else imputation_ids,
        q.shape[0],
    )
    if any(not isinstance(v, str) or not v.strip() for v in (*names, *ids)):
        raise AnalysisError(
            "invalid_labels", "Terms and imputation IDs need nonempty unique strings."
        )
    # Validate derived marginal inference before admitting the persisted result.
    _pool_tables(
        q,
        u,
        terms=names,
        imputation_ids=ids,
        complete_df=complete_df,
        alpha=alpha,
        imputation_description=imputation_description,
    )
    state = dict(
        schema_version="mi-pool-v1",
        estimates=q.tolist(),
        covariances=u.tolist(),
        terms=names,
        imputation_ids=ids,
        complete_df=None if complete_df is None else float(complete_df),
        alpha=float(alpha),
        imputation_description=imputation_description.strip(),
        metadata_json=json.dumps(
            {
                "precision": "float64",
                "device": "cpu",
                "workspace": resource,
                "source_results": sources,
                "stata_parity_validated": False,
                "sample_contract": "Compatible fitted-result samples or declared supplied common estimands; no p-value pooling",
                "candidate_source": "audited and extended PR91@830b9c387fe980ebae159a0babad45b8c0370948",
            },
            sort_keys=True,
            allow_nan=False,
        ),
    )
    return MIPoolResult(**state, integrity_sha256=digest(state))


def joint_values(pool, restrictions, values, df_method):
    _, _, mean, within, between, _, _ = tensors(pool.estimates, pool.covariances)
    p = len(pool.terms)
    shape = _shape(restrictions)
    if len(shape) != 2 or not 1 <= shape[0] <= p or shape[1] != p or _shape(values) != (shape[0],):
        raise AnalysisError(
            "invalid_restrictions",
            "Restrictions need full-row-rank k-by-p and k finite null values, 1<=k<=p.",
        )
    rmat, null = (
        torch.as_tensor(restrictions, dtype=FLOAT, device="cpu"),
        torch.as_tensor(values, dtype=FLOAT, device="cpu"),
    )
    if (
        not torch.isfinite(rmat).all()
        or not torch.isfinite(null).all()
        or int(torch.linalg.matrix_rank(rmat)) != shape[0]
    ):
        raise AnalysisError(
            "invalid_restrictions", "Restrictions must be finite and have independent rows."
        )
    k, m = shape[0], len(pool.estimates)
    w = rmat @ within @ rmat.T
    b = rmat @ between @ rmat.T
    delta = rmat @ mean - null
    if not all(torch.isfinite(v).all() for v in (w, b, delta)):
        raise AnalysisError(
            "numerical_failure", "Joint targets overflowed; rescale restrictions and estimands."
        )
    scale = float(w.abs().max())
    try:
        if scale <= 0 or float(torch.linalg.eigvalsh(w / scale).min()) <= 1e-12:
            raise ValueError("singular within covariance")
        chol = torch.linalg.cholesky(w)
        riv = (1 + 1 / m) * float(torch.cholesky_solve(b, chol).trace()) / k
        statistic = float(delta @ torch.cholesky_solve(delta[:, None], chol)[:, 0]) / (
            k * (1 + riv)
        )
    except (ValueError, RuntimeError) as exc:
        raise AnalysisError(
            "singular_covariance",
            "The selected within-imputation covariance must be positive definite and numerically resolved.",
        ) from exc
    if riv < -1e-12 or not math.isfinite(riv) or not math.isfinite(statistic):
        raise AnalysisError(
            "numerical_failure", "Joint inference exceeds finite float64 precision."
        )
    riv = max(0.0, riv)  # only nonnegative roundoff of a PSD trace, recorded below
    t = k * (m - 1)
    if df_method == "li1991":
        df = (
            math.inf
            if riv == 0
            else (
                4 + (t - 4) * (1 + (1 - 2 / t) / riv) ** 2
                if t > 4
                else t * (1 + 1 / k) * (1 + 1 / riv) ** 2 / 2
            )
        )
    elif df_method == "reiter2007":
        if pool.complete_df is None or t <= 4:
            raise AnalysisError(
                "unsupported_df",
                "Reiter needs finite complete_df and k*(m-1)>4; increase imputations or explicitly choose li1991.",
            )
        a = riv * t / (t - 2)
        vstar = pool.complete_df * (pool.complete_df + 1) / (pool.complete_df + 3)
        c1, c2 = vstar - 2 * (1 + a), vstar - 4 * (1 + a)
        if c2 <= 0:
            raise AnalysisError(
                "unsupported_df",
                "Reiter's positive-moment domain needs vstar>4*(1+a); no fallback df is substituted.",
            )
        a2, c0 = a * a, 1 / (t - 4)
        z = (
            1 / c2
            + c0 * a2 * c1 / ((1 + a) ** 2 * c2)
            + c0 * (8 * a2 * c1 / ((1 + a) * c2 * c2) + 4 * a2 / ((1 + a) * c2))
            + c0 * (4 * a2 / (c2 * c1) + 16 * a2 * c1 / (c2**3))
            + c0 * 8 * a2 / (c2 * c2)
        )
        df = 4 + 1 / z
    else:
        raise AnalysisError("invalid_df", "df_method must be li1991 or reiter2007.")
    if not math.isinf(df) and (not math.isfinite(df) or df <= 0):
        raise AnalysisError("numerical_failure", "D1 denominator df is not positive and finite.")
    return statistic, k, df, f_sf(statistic, k, df), riv, w * (1 + riv), delta


class MIJointResult(BaseModel):
    """Integrity-checked D1 state retaining its complete pooling parent."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["mi-joint-v1"] = "mi-joint-v1"
    pool: MIPoolResult
    restrictions: tuple[tuple[float, ...], ...] = Field(min_length=1, max_length=MAX_P)
    values: tuple[float, ...] = Field(min_length=1, max_length=MAX_P)
    df_method: Literal["li1991", "reiter2007"]
    integrity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def raw_admission(cls, state):
        if isinstance(state, dict):
            _shape(state.get("restrictions"))
            _shape(state.get("values"))
        return state

    @model_validator(mode="after")
    def valid(self):
        joint_values(self.pool, self.restrictions, self.values, self.df_method)
        if (
            digest(self.model_dump(mode="json", exclude={"integrity_sha256"}))
            != self.integrity_sha256
        ):
            raise ValueError("MI joint state integrity mismatch.")
        return self

    @property
    def table(self):
        stat, k, df, pv, riv, _, _ = joint_values(
            self.pool, self.restrictions, self.values, self.df_method
        )
        return table(
            {
                "method": ["D1"],
                "statistic": [stat],
                "df1": [k],
                "df2": [None if math.isinf(df) else df],
                "p_value": [pv],
                "relative_increase_variance": [riv],
                "df_method": [self.df_method],
            },
            infinite_df_representation="null denotes chi-square/k limit",
            assumption="Approximately normal common estimand; proportional between/within covariance approximation",
            stata_parity_validated=False,
        )

    @property
    def covariance(self):
        return joint_values(self.pool, self.restrictions, self.values, self.df_method)[5].tolist()

    def to_latex(self, **options):
        return self.table.to_latex(**options)

    def __str__(self):
        return str(self.table)

    def model_copy(self, *, update=None, deep=False):
        """Copies revalidate the joint and parent pooling state."""
        state = self.model_dump(mode="json")
        state.update(update or {})
        return type(self).model_validate(state)


def mi_test(pool, restrictions=None, values=None, *, df_method=None) -> MIJointResult:
    """Test R*Q=values by saved full-covariance MI D1 Wald inference.

    Default R=identity, values=zero. Finite complete_df chooses Reiter2007 only
    with k*(m-1)>4 and positive moment denominators; None chooses Li1991.
    li1991 may be explicitly requested to declare the large-complete-sample
    assumption. Singular restrictions/covariance and unsupported finite-df
    cases fail loudly. Marginal Barnard-Rubin and D1 use distinct calibrations.
    """
    if not isinstance(pool, MIPoolResult):
        raise AnalysisError("invalid_imputations", "mi_test requires a restored MIPoolResult.")
    pool = MIPoolResult.model_validate_json(pool.model_dump_json())
    p = len(pool.terms)
    rmat = torch.eye(p, dtype=FLOAT, device="cpu") if restrictions is None else restrictions
    shape = _shape(rmat)
    null = [0.0] * shape[0] if values is None and len(shape) == 2 else values
    method = (
        ("li1991" if pool.complete_df is None else "reiter2007") if df_method is None else df_method
    )
    joint_values(pool, rmat, null, method)
    state = dict(
        schema_version="mi-joint-v1",
        pool=pool.model_dump(mode="json"),
        restrictions=torch.as_tensor(rmat, dtype=FLOAT, device="cpu").tolist(),
        values=torch.as_tensor(null, dtype=FLOAT, device="cpu").tolist(),
        df_method=method,
    )
    return MIJointResult(**state, integrity_sha256=digest(state))
