"""Resident sample, resource, subject jackknife and complete JSON state contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, _json_safe
from openecon.econometrics.multivariate import common as c
from openecon.engines import distributions as dist
from openecon.resources import plan_workspace

PROCEDURES = (
    "polychoric",
    "polyserial",
    "omega_total",
    "icc",
    "cohen_kappa",
    "fleiss_kappa",
    "krippendorff_alpha",
    "gwet_ac",
)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def options(inference, level, max_fits, max_work, device, weights):
    c.check_choice(inference, "inference", ("none", "jackknife"))
    c.check_number(level, "level", minimum=0, maximum=1, exclusive=True)
    if level == 1:
        raise AnalysisError("invalid_option", "level must be strictly below one.")
    c.check_count(max_fits, "max_fits", maximum=10001)
    c.check_count(max_work, "max_work")
    if device != "cpu":
        raise AnalysisError(
            "unsupported_device", "Measurement procedures support resident CPU float64 only."
        )
    if weights is not None:
        raise AnalysisError(
            "unsupported_weights", "Measurement procedures do not support observation weights."
        )


def sample(
    data,
    columns,
    *,
    numeric=(),
    missing="drop",
    available=False,
    q=2,
    inference="none",
    max_fits=1024,
    max_work=100_000_000,
    fit_cost=1,
):
    if isinstance(data, Dataset):
        raise AnalysisError(
            "unsupported_dataset",
            "Measurement procedures require a resident table; collect deliberately.",
        )
    names = c.name_list(columns, "columns", minimum=2)
    c.check_choice(missing, "missing", ("available", "raise") if available else ("drop", "raise"))
    source = c.source(data)
    n, p = len(source), len(names)
    if p > 64 or n > 100000:
        raise AnalysisError(
            "resource_limit", "Measurement input is bounded to 100000 units and 64 columns."
        )
    absent = [name for name in names if name not in source]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {absent}.")
    c.require_numeric(source, numeric)
    # Includes selected/coded tables, leave-one copy, per-unit contribution buffers,
    # jackknife estimates, category matrices and numeric factor/ANOVA workspace.
    plan = plan_workspace(
        "resident measurement and subject jackknife",
        {
            "selected_and_coded_units": 64 * n * (p + q),
            "numeric_and_category_matrices": 128 * (p * p + q * q),
            "unit_and_jackknife_buffers": 128 * n,
        },
    )
    selected = source.loc[:, names]
    keep = selected.notna().sum(axis=1).ge(2) if available else selected.notna().all(axis=1)
    if missing == "raise" and selected.isna().any().any():
        raise AnalysisError("missing_values", "Selected columns contain missing values.")
    positions = [i for i, value in enumerate(keep) if value]
    selected = selected.loc[keep].reset_index(drop=True)
    used = len(selected)
    if used < 3:
        raise AnalysisError(
            "insufficient_observations", "At least three usable independent units are required."
        )
    for name in numeric:
        c.column(selected, name)
    fits = 1 + (used if inference == "jackknife" else 0)
    if fits > max_fits or fits * used * max(fit_cost, p * p, q * q) > max_work:
        raise AnalysisError(
            "work_budget_exceeded",
            "Complete measurement fits exceed max_fits or max_work; no fits were run.",
        )
    metadata = dict(
        n=used,
        n_input=n,
        n_missing=n - used,
        n_missing_cells=int(source[names].isna().sum().sum()),
        positions=positions,
        unit_labels=[
            None
            if isinstance(source.index[i], float) and not math.isfinite(source.index[i])
            else c.label(source.index[i])
            for i in positions
        ],
        columns=names,
        missing=missing,
        resource_plan=plan.record(),
        planned_fits=fits,
        max_work=max_work,
        max_fits=max_fits,
    )
    return selected, metadata


def levels(categories):
    if not isinstance(categories, (list, tuple)) or not 2 <= len(categories) <= 64:
        raise AnalysisError(
            "invalid_categories",
            "Declare an ordered category list of 2..64 distinct scalar labels.",
        )
    result = []
    for value in categories:
        if hasattr(value, "item"):
            value = value.item()
        if (
            not isinstance(value, (str, int, float, bool))
            or isinstance(value, float)
            and not math.isfinite(value)
        ):
            raise AnalysisError(
                "invalid_categories", "Category labels must be finite JSON scalars."
            )
        if value in result:
            raise AnalysisError("invalid_categories", "Category labels must be distinct.")
        result.append(value)
    return result


def codes(frame, categories):
    mapping = {value: i for i, value in enumerate(categories)}
    rows = []
    for values in frame.itertuples(index=False, name=None):
        row = []
        for value in values:
            if not pd.api.types.is_scalar(value):
                raise AnalysisError(
                    "invalid_categories", "Observed category values must be scalar labels."
                )
            if pd.isna(value):
                row.append(-1)
            else:
                try:
                    row.append(mapping[value])
                except (KeyError, TypeError) as exc:
                    raise AnalysisError(
                        "unknown_category",
                        f"Observed value {value!r} is outside the declared category universe.",
                    ) from exc
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int64)


def agreement_weights(categories, weights):
    q = len(categories)
    if isinstance(weights, str):
        c.check_choice(weights, "agreement_weights", ("unweighted", "linear", "quadratic"))
        d = torch.arange(q, dtype=c.FLOAT)
        delta = (d[:, None] - d[None, :]).abs() / (q - 1)
        w = (
            torch.eye(q, dtype=c.FLOAT)
            if weights == "unweighted"
            else 1 - delta ** (1 if weights == "linear" else 2)
        )
    else:
        if isinstance(weights, torch.Tensor) and weights.device.type != "cpu":
            raise AnalysisError("unsupported_device", "Agreement matrices must be on CPU.")
        if (
            hasattr(weights, "shape")
            and tuple(weights.shape) != (q, q)
            or isinstance(weights, (list, tuple))
            and (
                len(weights) != q
                or any(not isinstance(row, (list, tuple)) or len(row) != q for row in weights)
            )
        ):
            raise AnalysisError(
                "invalid_weights", "Agreement matrix dimensions must match categories."
            )
        try:
            w = torch.as_tensor(weights, dtype=c.FLOAT)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise AnalysisError(
                "invalid_weights", "Supply a finite symmetric q-by-q agreement matrix."
            ) from exc
        if (
            w.shape != (q, q)
            or not bool(torch.isfinite(w).all())
            or bool(((w < 0) | (w > 1)).any())
            or not torch.allclose(w, w.T, atol=1e-12, rtol=0)
            or not torch.allclose(w.diagonal(), torch.ones(q, dtype=c.FLOAT), atol=1e-12, rtol=0)
        ):
            raise AnalysisError(
                "invalid_weights",
                "Agreement weights must be symmetric in [0,1] with unit diagonal and category dimensions.",
            )
    return w


def ratio(numerator, denominator):
    if not math.isfinite(float(denominator)) or abs(float(denominator)) < 1e-12:
        raise AnalysisError(
            "undefined_coefficient", "The reliability coefficient has zero or unstable denominator."
        )
    estimate = float(numerator / denominator)
    if not math.isfinite(estimate):
        raise AnalysisError("non_finite_result", "Reliability coefficient is not finite.")
    return estimate


def result(procedure, selected, metadata, fit, *, inference, level, settings):
    fit = c.procedure(fit)
    estimate, tables, details = fit(selected)
    se = statistic = pvalue = lower = upper = variance = df = None
    replicas = []
    if inference == "jackknife":
        n = len(selected)
        for i in range(n):
            try:
                replicas.append(fit(selected.drop(index=i).reset_index(drop=True))[0])
            except AnalysisError as exc:
                raise AnalysisError(
                    "replicate_failure",
                    f"Delete-one unit at original position {metadata['positions'][i]} failed ({exc.code}); all {n} replicates are required.",
                ) from exc
        draws = torch.tensor(replicas, dtype=c.FLOAT)
        variance = float((n - 1) / n * ((draws - draws.mean()) ** 2).sum())
        if variance <= 1e-28:
            raise AnalysisError(
                "degenerate_inference",
                "Subject jackknife variance is zero; no t inference can be formed. Request inference='none' for the point estimate.",
            )
        se, df = math.sqrt(variance), n - 1
        statistic = estimate / se
        pvalue = min(1.0, 2 * dist.t_sf(abs(statistic), df))
        critical = dist.t_ppf((1 + level) / 2, df)
        lower, upper = estimate - critical * se, estimate + critical * se
        tables["covariance"] = c.frame([[variance]], columns=[procedure], index=[procedure])
        tables["jackknife"] = c.frame(
            [[x] for x in replicas], columns=["estimate"], index=metadata["positions"]
        )
    tables = {
        "estimate": c.frame(
            [[estimate, se, df, statistic, pvalue, lower, upper]],
            columns=["estimate", "std_error", "df", "statistic", "p_value", "ci_lower", "ci_upper"],
            index=[procedure],
        ),
        **tables,
    }
    state = dict(
        schema="openecon.measurement.v1",
        procedure=procedure,
        settings=settings,
        sample=metadata,
        diagnostics=details,
        estimate=estimate,
        covariance=variance,
        inference=inference,
        level=level,
        replicates=replicas,
        assumptions="independent sampled subject/unit rows; raters/items remain together",
        reference="delete-one subject jackknife t approximation; no exact or vendor interval claim"
        if inference == "jackknife"
        else "point estimate only",
    )
    state = _json_safe(state)
    return TableSet(
        tables,
        title=f"Measurement reliability: {procedure}",
        procedure=procedure,
        **metadata,
        estimate=estimate,
        inference=inference,
        level=level,
        completed_fits=1 + len(replicas),
        failed_fits=0,
        device="cpu",
        dtype="float64",
        state=state,
        state_sha256=digest(state),
        tables_sha256=digest(_table_payload(tables)),
        notes=[state["reference"]],
    )


def _table_payload(tables):
    return {
        name: {
            "index": _json_safe(list(table.index)),
            "columns": _json_safe(list(table.columns)),
            "data": _json_safe(table.values.tolist()),
        }
        for name, table in tables.items()
    }


def reliability_save(result: TableSet, path=None) -> dict:
    """Save complete measurement tables/state as a checksummed JSON artifact; optional path."""
    if (
        not isinstance(result, TableSet)
        or result.attrs.get("procedure") not in PROCEDURES
        or digest(result.attrs.get("state")) != result.attrs.get("state_sha256")
        or digest(_table_payload(result)) != result.attrs.get("tables_sha256")
    ):
        raise AnalysisError("invalid_result", "Supply an intact measurement reliability result.")
    tables = _table_payload(result)
    payload = json.loads(
        canonical(
            dict(
                schema="openecon.measurement.artifact.v1",
                title=result.title,
                attrs=result.attrs,
                tables=tables,
            )
        )
    )
    artifact = {"payload": payload, "sha256": digest(payload)}
    if path is not None:
        Path(path).write_text(canonical(artifact) + "\n", encoding="utf-8")
    return artifact


def reliability_load(artifact) -> TableSet:
    """Restore and verify complete JSON tables, sample labels, options, diagnostics and inference."""
    try:
        if isinstance(artifact, (str, Path)):
            artifact = json.loads(Path(artifact).read_text(encoding="utf-8"))
        payload = artifact["payload"]
        if (
            payload["schema"] != "openecon.measurement.artifact.v1"
            or digest(payload) != artifact["sha256"]
        ):
            raise ValueError("artifact checksum mismatch")
        attrs = payload["attrs"]
        if (
            attrs["procedure"] not in PROCEDURES
            or digest(attrs["state"]) != attrs["state_sha256"]
            or digest(payload["tables"]) != attrs["tables_sha256"]
        ):
            raise ValueError("state checksum mismatch")
        tables = {
            name: c.frame(value["data"], columns=value["columns"], index=value["index"])
            for name, value in payload["tables"].items()
        }
        return TableSet(tables, title=payload["title"], **attrs)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise AnalysisError(
            "invalid_saved_result", f"Measurement artifact could not be restored: {exc}."
        ) from exc
