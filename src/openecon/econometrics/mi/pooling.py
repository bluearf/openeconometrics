"""Explicit missingness and Rubin/Barnard-Rubin coefficient pooling.

Pooling combines estimates and full within-imputation covariance matrices,
never p-values. The caller must describe the imputation and common estimand.
"""

from __future__ import annotations

import math
import torch

from openecon.analysis import _coerce_frame, _hash_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.postest.multiple import _labels, _level
from openecon.engines.distributions import normal_isf, normal_sf, t_isf, t_sf
from openecon.models import ResultBundle


def missing_patterns(data, columns=None) -> TableSet:
    """Summarize observed missingness patterns; this is not an MCAR test.

    The selected columns are grouped by their missing-value indicators. Original
    values are neither changed nor imputed. Counts, selected source fingerprint
    and explicit binary pattern convention are returned as exportable tables.
    """
    frame = _coerce_frame(data)
    names = list(frame.columns) if columns is None else columns
    if isinstance(names, (str, bytes)) or not isinstance(names, (list, tuple)) or not names:
        raise AnalysisError("invalid_columns", "Select a nonempty list of column names.")
    if (
        frame.columns.has_duplicates
        or len(set(names)) != len(names)
        or any(x not in frame for x in names)
    ):
        raise AnalysisError("invalid_columns", "Selected columns must exist and be unique.")
    if not len(frame) or len(frame) * len(names) > 8000000:
        raise AnalysisError(
            "work_budget", "Select a nonempty table with at most eight million selected cells."
        )
    missing = frame.loc[:, names].isna()
    patterns = missing.value_counts(sort=True).reset_index(name="count")
    patterns["fraction"] = patterns["count"] / len(frame)
    summary = table(
        {
            "variable": names,
            "missing": missing.sum().tolist(),
            "observed": (~missing).sum().tolist(),
            "fraction_missing": missing.mean().tolist(),
        }
    )
    return TableSet(
        {"patterns": table(patterns), "variables": summary},
        title="Missingness patterns",
        nobs=len(frame),
        columns=names,
        convention="True denotes missing; no MCAR/MAR conclusion is inferred",
        data_hash=_hash_frame(frame.loc[:, names]),
        stata_parity_validated=False,
    )


def _from_bundles(results, complete_df):
    if not all(isinstance(x, ResultBundle) for x in results):
        raise AnalysisError(
            "invalid_imputations", "All fitted inputs must be ResultBundle objects."
        )
    first = results[0]
    _labels([result.id for result in results], len(results))
    terms = [c.term for c in first.coefficients]
    spec = first.spec.model_dump(mode="json")
    source = []
    for result in results:
        if (
            result.spec.model_dump(mode="json") != spec
            or [c.term for c in result.coefficients] != terms
            or [c.equation for c in result.coefficients] != [c.equation for c in first.coefficients]
            or result.sample_positions != first.sample_positions
            or result.nobs_original != first.nobs_original
            or result.provenance.get("sample_positions_omitted")
            != first.provenance.get("sample_positions_omitted")
            or (
                result.provenance.get("sample_positions_omitted")
                and (
                    not result.provenance.get("sample_positions_hash")
                    or result.provenance.get("sample_positions_hash")
                    != first.provenance.get("sample_positions_hash")
                )
            )
            or result.nobs != first.nobs
            or result.provenance.get("converged") is False
            or result.extra.get("converged") is False
            or result.inference.get("available") is False
            or any(c.std_error is None for c in result.coefficients)
        ):
            raise AnalysisError(
                "incompatible_imputations",
                "Imputations need identical specifications, terms, sample rows and converged inferential fits.",
            )
        source.append({"id": result.id, "data_hash": result.provenance.get("data_hash")})
    if complete_df is None:
        dfs = [x.inference.get("df_inference") for x in results]
        if all(isinstance(df, (int, float)) and df > 0 for df in dfs):
            if max(dfs) - min(dfs) > 1e-8:
                raise AnalysisError(
                    "incompatible_df", "Complete-data degrees of freedom differ across imputations."
                )
            complete_df = dfs[0]
    labels = [f"{c.equation}:{c.term}" if c.equation else c.term for c in first.coefficients]
    return (
        [[c.estimate for c in x.coefficients] for x in results],
        [x.covariance_matrix for x in results],
        labels,
        complete_df,
        source,
    )


@torch.no_grad()
def _pool_tables(
    estimates,
    covariances=None,
    *,
    terms=None,
    imputation_ids=None,
    complete_df=None,
    alpha=0.05,
    imputation_description=None,
) -> TableSet:
    """Pool a common coefficient vector using Rubin and Barnard-Rubin rules.

    Pass either at least two compatible restored ResultBundles, or an m-by-p
    estimate matrix and m-by-p-by-p full covariance tensor. ``complete_df=None``
    uses Rubin's large-complete-sample df (inferred from bundles when recorded).
    Finite positive complete_df selects Barnard-Rubin. The imputation description
    must identify the common estimand and scientifically justified imputation.
    Returns coefficient, within/between/total covariance and df/FMI tables. Joint
    D1/D2/D3 tests, imputation generation and arbitrary pooled prediction are not
    provided by this procedure.
    """
    _level(alpha)
    if not isinstance(imputation_description, str) or not imputation_description.strip():
        raise AnalysisError(
            "missing_imputation_design", "Describe the imputation procedure and common estimand."
        )
    sources = []
    if (
        isinstance(estimates, (list, tuple))
        and estimates
        and isinstance(estimates[0], ResultBundle)
    ):
        if covariances is not None or terms is not None:
            raise AnalysisError(
                "invalid_imputations",
                "Bundle pooling reads terms and covariance from the fitted results.",
            )
        estimates, covariances, terms, complete_df, sources = _from_bundles(estimates, complete_df)
    try:
        if (
            torch.as_tensor(estimates, device="cpu").is_complex()
            or torch.as_tensor(covariances, device="cpu").is_complex()
        ):
            raise ValueError("complex pooling inputs")
        q = torch.as_tensor(estimates, dtype=torch.float64, device="cpu")
        u = torch.as_tensor(covariances, dtype=torch.float64, device="cpu")
    except (ValueError, TypeError, RuntimeError) as exc:
        raise AnalysisError(
            "invalid_imputations", "Supply numeric coefficient and full covariance arrays."
        ) from exc
    if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] < 1 or u.shape != (*q.shape, q.shape[1]):
        raise AnalysisError(
            "invalid_imputations",
            "Need m>=2 coefficient vectors and m matching full covariance matrices.",
        )
    m, p = q.shape
    if m > 1000 or p > 128 or u.numel() > 1000000:
        raise AnalysisError(
            "work_budget",
            "Pooling exceeds 1000 imputations, 128 parameters or one million covariance cells.",
        )
    if not torch.isfinite(q).all() or not torch.isfinite(u).all():
        raise AnalysisError("invalid_imputations", "Estimates and covariances must all be finite.")
    if not torch.allclose(u, u.transpose(1, 2), rtol=1e-10, atol=1e-12):
        raise AnalysisError(
            "invalid_covariance", "Every within-imputation covariance must be symmetric."
        )
    if (u.diagonal(dim1=1, dim2=2) <= 0).any() or (
        torch.linalg.eigvalsh(u) < -1e-10 * u.abs().amax(dim=(1, 2))[:, None]
    ).any():
        raise AnalysisError(
            "invalid_covariance",
            "Within-imputation covariances need positive marginal variances and must be positive semidefinite.",
        )
    if complete_df is not None and (
        isinstance(complete_df, bool)
        or not isinstance(complete_df, (int, float))
        or not math.isfinite(complete_df)
        or complete_df <= 0
    ):
        raise AnalysisError(
            "invalid_df",
            "complete_df must be finite and positive, or None for the large-sample rule.",
        )
    names = _labels(terms, p)
    ids = _labels(imputation_ids, m)
    mean, within = q.mean(0), u.mean(0)
    centered = q - mean
    between = centered.T @ centered / (m - 1)
    added = (1 + 1 / m) * between
    total = within + added
    if not all(torch.isfinite(x).all() for x in (mean, within, between, total)):
        raise AnalysisError(
            "numerical_failure",
            "Pooling exceeded the finite float64 range; rescale the common estimand.",
        )
    lam = added.diag() / total.diag()
    r = added.diag() / within.diag()
    old_df = torch.where(lam > 0, (m - 1) / lam.square(), torch.inf)
    if complete_df is None:
        df = old_df
    else:
        observed_df = complete_df * (complete_df + 1) / (complete_df + 3) * (1 - lam)
        df = 1 / (1 / old_df + 1 / observed_df)
    se = total.diag().sqrt()
    statistic = mean / se
    critical = torch.tensor(
        [
            normal_isf(alpha / 2) if math.isinf(v) else kernel_call(t_isf, alpha / 2, v)
            for v in df.tolist()
        ],
        dtype=torch.float64,
        device="cpu",
    )
    pvalues = [
        2 * (normal_sf(abs(z)) if math.isinf(v) else kernel_call(t_sf, abs(z), v))
        for z, v in zip(statistic.tolist(), df.tolist(), strict=True)
    ]
    fmi = (r + 2 / (df + 3)) / (1 + r)
    if not all(
        torch.isfinite(x).all()
        for x in (se, statistic, critical, fmi, mean - critical * se, mean + critical * se)
    ):
        raise AnalysisError(
            "numerical_failure", "The pooled inferential target exceeds the finite float64 range."
        )
    pooled = table(
        {
            "term": names,
            "estimate": mean.tolist(),
            "std_error": se.tolist(),
            "statistic": statistic.tolist(),
            "df": [None if math.isinf(v) else v for v in df.tolist()],
            "p_value": pvalues,
            "ci_low": (mean - critical * se).tolist(),
            "ci_high": (mean + critical * se).tolist(),
            "lambda": lam.tolist(),
            "fraction_missing_information": fmi.tolist(),
            "relative_increase_variance": r.tolist(),
        }
    )
    tables = {"coefficients": pooled}
    for label, matrix in (("within", within), ("between", between), ("total", total)):
        tables[f"{label}_covariance"] = table(
            matrix.tolist(), columns=[str(x) for x in names], index=names
        )
    return TableSet(
        tables,
        title="Multiple-imputation pooled inference",
        imputations=m,
        terms=names,
        imputation_ids=ids,
        alpha=alpha,
        complete_df=complete_df,
        df_method="Rubin" if complete_df is None else "Barnard-Rubin",
        imputation_description=imputation_description.strip(),
        source_results=sources,
        infinite_df_representation="null denotes the normal complete-data limit",
        precision="float64",
        stata_parity_validated=False,
    )
