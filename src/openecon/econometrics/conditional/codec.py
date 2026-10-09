"""Complete ordered conditional distributions and checked postestimation artifacts."""

from __future__ import annotations
import json
from pathlib import Path
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.finite.common import canonical, digest, MAX_ARTIFACT

PROCEDURES = tuple(
    f"exact_{family}_{stage}"
    for family in ("logit", "poisson")
    for stage in ("fit", "ci", "test", "moments")
)


def _tables(output):
    return {
        n: {"columns": list(f.columns), "index": list(f.index), "data": f.to_numpy().tolist()}
        for n, f in output.items()
    }


def seal(procedure, tables, state, **attrs):
    output = TableSet(
        tables,
        title=procedure.replace("_", " ").title(),
        procedure=procedure,
        state=state,
        precision="float64",
        device="cpu",
        **attrs,
    )
    output.attrs["integrity_sha256"] = digest(
        {"attrs": output.attrs, "tables": _tables(output), "table_order": list(output)}
    )
    return output


def intact(output, procedure=None):
    if (
        not isinstance(output, TableSet)
        or output.attrs.get("procedure") not in PROCEDURES
        or (procedure is not None and output.attrs["procedure"] != procedure)
    ):
        raise AnalysisError(
            "invalid_result",
            "Supply a complete conditional regression result of the required procedure.",
        )
    attrs = {k: v for k, v in output.attrs.items() if k != "integrity_sha256"}
    if digest(
        {"attrs": attrs, "tables": _tables(output), "table_order": list(output)}
    ) != output.attrs.get("integrity_sha256"):
        raise AnalysisError(
            "invalid_result", "Conditional regression tables/state were modified after creation."
        )
    return output.attrs["state"]


def conditional_save(result, path=None):
    """Save all ordered conditional-regression tables, covariance, samples and numerical state."""
    intact(result)
    payload = {
        "schema": "openecon.conditional.v1",
        "title": result.title,
        "attrs": result.attrs,
        "tables": _tables(result),
        "table_order": list(result),
    }
    artifact = json.loads(canonical({"payload": payload, "sha256": digest(payload)}))
    encoded = canonical(artifact)
    if len(encoded.encode()) > MAX_ARTIFACT:
        raise AnalysisError(
            "resource_limit", "Complete conditional-regression artifact exceeds 32 MiB."
        )
    if path is not None:
        Path(path).write_text(encoded + "\n", encoding="utf-8")
    return artifact


def conditional_load(artifact):
    """Restore a complete checksummed conditional-regression result without numerical refitting."""
    try:
        if isinstance(artifact, (str, Path)):
            path = Path(artifact)
            if path.stat().st_size > MAX_ARTIFACT:
                raise ValueError("Artifact exceeds 32 MiB")
            artifact = json.loads(
                path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v))
            )
        if len(canonical(artifact).encode()) > MAX_ARTIFACT:
            raise ValueError("Artifact exceeds 32 MiB")
        payload = artifact["payload"]
        if payload["schema"] != "openecon.conditional.v1" or digest(payload) != artifact["sha256"]:
            raise ValueError("Checksum/schema mismatch")
        order = payload["table_order"]
        if (
            not isinstance(order, list)
            or not order
            or any(not isinstance(n, str) for n in order)
            or (len(set(order)) != len(order) or set(order) != set(payload["tables"]))
        ):
            raise ValueError("Malformed table order")
        tables = {}
        for n in order:
            f = payload["tables"][n]
            if len(f["data"]) != len(f["index"]) or any(
                len(row) != len(f["columns"]) for row in f["data"]
            ):
                raise ValueError("Malformed table")
            tables[n] = table(f["data"], columns=f["columns"], index=f["index"])
        output = TableSet(tables, title=payload["title"], **payload["attrs"])
        intact(output)
        return output
    except (KeyError, TypeError, ValueError, OSError, AnalysisError, RecursionError) as exc:
        raise AnalysisError(
            "invalid_saved_result", "Conditional artifact schema/checksum/tables/state are invalid."
        ) from exc
