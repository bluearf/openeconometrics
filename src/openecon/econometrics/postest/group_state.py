"""Immutable typed group state: bounded inline records or owned persistent SQLite.

The numerical state is captured at estimation, never reconstructed from a chart
preview. Large maps remain on disk. Exporting the state file alongside a result
is explicit; checksum failure and absent historical state fail openly.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest.index_codec import decode, encode
from openecon.resources import plan_workspace, workspace_budget_bytes
from openecon.streaming_design import encode_cluster_labels

_INLINE = 2 * 1024**2
_DISK = 256 * 1024**2


def keys(frame, columns):
    if len(columns) == 1:
        return encode_cluster_labels(frame[columns[0]])
    return encode_cluster_labels(pd.Series(list(frame[columns].itertuples(index=False, name=None))))


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Builder:
    def __init__(self, columns, kind):
        self.columns, self.kind = list(columns), kind
        self.scratch = tempfile.TemporaryDirectory(
            prefix="openecon-group-state-", dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None
        )
        self.path = Path(self.scratch.name) / "state.sqlite3"
        self.db = None
        try:
            self.db = sqlite3.connect(self.path)
            self.db.execute("PRAGMA journal_mode=OFF")
            self.db.execute("PRAGMA synchronous=OFF")
            self.db.execute("PRAGMA cache_size=-2048")
            self.db.execute("PRAGMA temp_store=FILE")
            self.db.execute("PRAGMA mmap_size=0")
            self.db.execute("CREATE TABLE effects(key BLOB PRIMARY KEY, record TEXT) WITHOUT ROWID")
            self.path.chmod(0o600)
        except BaseException:
            if self.db is not None:
                self.db.close()
            self.scratch.cleanup()
            raise
        self.finished = False

    def add(self, key, record):
        payload = json.dumps(record, separators=(",", ":"), allow_nan=False)
        existing = self.db.execute("SELECT record FROM effects WHERE key=?", (key,)).fetchone()
        if existing is not None:
            prior = json.loads(existing[0])
            if self.kind == "complete_group" and "row_sum" in record:
                prior["n_rows"] += record["n_rows"]
                prior["row_sum"] = hex(
                    (int(prior["row_sum"], 16) + int(record["row_sum"], 16)) % (1 << 256)
                )
                prior["success_count"] = (
                    None
                    if record["success_count"] is None
                    else prior["success_count"] + record["success_count"]
                )
                self.db.execute(
                    "UPDATE effects SET record=? WHERE key=?",
                    (json.dumps(prior, separators=(",", ":")), key),
                )
                return
            if self.kind == "fixed_effect_sum" and not math.isclose(
                prior["effect"], record["effect"], rel_tol=2e-7, abs_tol=2e-7
            ):
                raise AnalysisError(
                    "invalid_group_state",
                    "Fitted fixed-effect sums are inconsistent within a declared joint group key.",
                )
            return
        self.db.execute("INSERT INTO effects VALUES(?,?)", (key, payload))

    def flush(self):
        self.db.commit()
        if self.path.stat().st_size > _DISK:
            raise AnalysisError(
                "group_state_disk_budget", "Saved group state exceeds its 256 MiB disk budget."
            )

    def finish(self, **extra):
        if self.kind == "complete_group":
            for key, payload in self.db.execute("SELECT key,record FROM effects"):
                record = json.loads(payload)
                if "row_sum" in record:
                    total = int(record.pop("row_sum"), 16)
                    record["row_digest"] = hashlib.sha256(
                        total.to_bytes(32, "big") + record["n_rows"].to_bytes(8, "big")
                    ).hexdigest()
                    self.db.execute(
                        "UPDATE effects SET record=? WHERE key=?",
                        (json.dumps(record, separators=(",", ":")), key),
                    )
        self.flush()
        count = self.db.execute("SELECT COUNT(*) FROM effects").fetchone()[0]
        size = self.db.execute(
            "SELECT COALESCE(SUM(LENGTH(record)+2*LENGTH(key)),0) FROM effects"
        ).fetchone()[0]
        state = {
            "schema_version": 1,
            "kind": self.kind,
            "group_columns": self.columns,
            "n_groups": count,
            **extra,
        }
        if size <= _INLINE:
            state["records"] = {
                key.hex(): json.loads(record)
                for key, record in self.db.execute("SELECT key,record FROM effects ORDER BY key")
            }
        else:
            directory = Path(
                os.environ.get("OPENECON_GROUP_STATE_DIRECTORY")
                or Path.home() / ".openecon" / "group-state"
            )
            directory.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(directory).free < self.path.stat().st_size + 16 * 1024**2:
                raise AnalysisError(
                    "group_state_disk_budget",
                    "Insufficient disk space for persistent fitted group state.",
                )
            digest = checksum(self.path)
            target = directory / (digest + ".sqlite3")
            if target.exists() and checksum(target) != digest:
                raise AnalysisError(
                    "group_state_changed", "Existing content-addressed group state has changed."
                )
            if not target.exists():
                temporary = directory / (digest + "." + os.urandom(8).hex() + ".tmp")
                try:
                    shutil.copyfile(self.path, temporary)
                    temporary.chmod(0o600)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            state.update(
                path=str(target),
                sha256=digest,
                bytes=target.stat().st_size,
                storage="immutable owned SQLite; export file with saved result when moving hosts",
            )
        self.finished = True
        return state

    def close(self):
        self.db.close()
        self.scratch.cleanup()


@contextmanager
def lookup(state):
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise AnalysisError(
            "missing_group_state", "This historical result has no validated fitted group state."
        )
    if "records" in state:
        record = state["records"]
        if not isinstance(record, dict) or len(record) != state.get("n_groups"):
            raise AnalysisError("invalid_group_state", "Saved inline group records are malformed.")
        yield lambda key: record.get(key.hex())
        return
    path = Path(state.get("path", ""))
    if (
        not path.is_file()
        or path.stat().st_size != state.get("bytes")
        or path.stat().st_size > _DISK
        or checksum(path) != state.get("sha256")
    ):
        raise AnalysisError(
            "group_state_changed",
            "Saved group-state file is missing or its checksum changed; attach the original state file.",
        )
    db = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    db.execute("PRAGMA cache_size=-2048")
    try:

        def get(key):
            row = db.execute("SELECT record FROM effects WHERE key=?", (key,)).fetchone()
            return None if row is None else json.loads(row[0])

        yield get
    finally:
        db.close()


def _contract_columns(result):
    spec = result.spec
    extra = []
    for role in ("random", "offset", "exposure"):
        value = spec.columns.get(role, [])
        extra += [value] if isinstance(value, str) else list(value)
    return list(
        dict.fromkeys(
            [spec.outcome, *spec.predictors, *extra, *([spec.weights] if spec.weights else [])]
        )
    )


def row_sum(result, frame):
    columns = _contract_columns(result)
    total = 0
    numeric = set(columns) - set(result.spec.categorical)
    for row in frame[columns].itertuples(index=False, name=None):
        values = [
            float(value) if name in numeric else value
            for name, value in zip(columns, row, strict=True)
        ]
        total = (
            total + int.from_bytes(hashlib.sha256(encode(tuple(values)).encode()).digest(), "big")
        ) % (1 << 256)
    return total


def row_digest(result, frame):
    return hashlib.sha256(
        row_sum(result, frame).to_bytes(32, "big") + len(frame).to_bytes(8, "big")
    ).hexdigest()


def capture_contract(result, frame):
    spec = result.spec
    raw = spec.columns.get("group") or spec.panel
    columns = [raw] if isinstance(raw, str) else list(raw)
    # A nested mixed BLUP uses the full top-level group, including all children.
    top = columns[:1]
    builder = Builder(top, "complete_group")
    try:
        for _, block in frame.groupby(top[0], sort=False, observed=True):
            key = keys(block.iloc[:1], top)[0]
            builder.add(
                key,
                {
                    "n_rows": len(block),
                    "success_count": int(block[spec.outcome].sum())
                    if spec.estimator in {"clogit", "xtlogit"}
                    else None,
                    "row_digest": row_digest(result, block),
                    "label": encode(block[top[0]].iloc[0]),
                },
            )
        return builder.finish(
            contract_columns=_contract_columns(result),
            complete_set="all retained fitted rows of each requested top group; row multiset, count and success total must match",
        )
    finally:
        builder.close()


def capture_fixed(result, frame, fitted, x, terms):
    spec = result.spec
    if spec.estimator in {"xtreg", "xtivreg"}:
        columns = [spec.panel]
    else:
        raw = spec.columns["absorb"]
        columns = [raw] if isinstance(raw, str) else list(raw)
    positions = {term: i for i, term in enumerate(terms)}
    selected = [c.term for c in result.coefficients if c.term in positions]
    x = x[:, [positions[name] for name in selected]]
    beta = torch.tensor(
        [next(c.estimate for c in result.coefficients if c.term == name) for name in selected],
        dtype=torch.float64,
    )
    index = x @ beta
    nonlinear = spec.estimator == "ppmlhdfe"
    offset = torch.zeros(len(frame), dtype=torch.float64)
    if nonlinear:
        for role in ("offset", "exposure"):
            name = spec.columns.get(role)
            if name:
                values = torch.tensor(frame[name].to_numpy(dtype=float), dtype=torch.float64)
                offset += values.log() if role == "exposure" else values
    weights = (
        torch.ones(len(frame), dtype=torch.float64)
        if not spec.weights
        else torch.tensor(frame[spec.weights].to_numpy(dtype=float), dtype=torch.float64)
    )
    if spec.estimator in {"xtreg", "xtivreg"}:
        residual = (
            torch.tensor(frame[spec.outcome].to_numpy(dtype=float), dtype=torch.float64) - index
        )
        codes, _ = pd.factorize(frame[columns[0]], sort=False)
        codes = torch.tensor(codes, dtype=torch.int64)
        count = int(codes.max()) + 1
        mass = torch.zeros(count, dtype=torch.float64).index_add(0, codes, weights)
        effect = (
            torch.zeros(count, dtype=torch.float64).index_add(0, codes, residual * weights) / mass
        )[codes]
    else:
        fitted = torch.as_tensor(fitted, dtype=torch.float64)
        effect = (fitted.log() - offset if nonlinear else fitted) - index
    # Orthogonal nuisance coordinates: response gradient X_new-P_FE X, and
    # independent FE projection noise. Supports full inference when its dense
    # LEVEL geometry fits the declared workspace, independently of row count.
    levels, maps = [], []
    width = 0
    for name in columns:
        codes, labels = pd.factorize(frame[name], sort=False)
        maps.append(torch.tensor(codes, dtype=torch.int64) + width)
        levels.append(len(labels))
        width += len(labels)
    joint = None
    planned = 64 * len(frame) * (width + x.shape[1]) + 96 * width**2
    if (
        spec.covariance == "nonrobust"
        and width <= 1024
        and planned <= min(128 * 1024**2, workspace_budget_bytes())
    ):
        plan_workspace("saved full fixed-effect inference", {"nuisance_geometry": planned})
        f = torch.zeros((len(frame), width), dtype=torch.float64)
        for codes in maps:
            f[torch.arange(len(frame)), codes] = 1
        w = weights * torch.as_tensor(fitted, dtype=torch.float64) if nonlinear else weights
        info = f.T @ (w[:, None] * f)
        eigenvalues, eigenvectors = torch.linalg.eigh(info)
        cutoff = float(eigenvalues.max()) * width * torch.finfo(torch.float64).eps * 16
        reciprocal = torch.where(
            eigenvalues > cutoff, eigenvalues.clamp_min(cutoff).reciprocal(), 0.0
        )
        bread = (eigenvectors * reciprocal) @ eigenvectors.T
        projection = f @ (bread @ (f.T @ (w[:, None] * x)))
        scale = 1.0 if nonlinear else result.metrics.get("rmse", result.metrics.get("sigma_e")) ** 2
        joint = {
            "covariance": (bread * scale).tolist(),
            "nuisance_width": width,
            "mean_terms": selected,
            "basis": "orthogonal nuisance projection; slope/nuisance cross uncertainty included through X-P_FE X",
        }
    builder = Builder(columns, "fixed_effect_sum")
    try:
        labels = list(frame[columns].itertuples(index=False, name=None))
        for i, key in enumerate(keys(frame, columns)):
            record = {"effect": float(effect[i]), "label": encode(labels[i])}
            if joint:
                record.update(
                    projection=projection[i].tolist(), nuisance=[int(codes[i]) for codes in maps]
                )
            builder.add(key, record)
        return builder.finish(
            full_inference=joint,
            response_scale="log" if nonlinear else "identity",
            inference_domain="full nonrobust nuisance information when recorded; otherwise point means only",
        )
    finally:
        builder.close()


def capture_resident(result, frame, fitted=None):
    if not hasattr(frame, "design"):
        return result  # A reporting preview is never a complete estimation sample.
    spec = result.spec
    if (
        spec.estimator in {"clogit", "mixed", "melogit", "meprobit", "mepoisson"}
        or spec.estimator in {"xtlogit", "xtprobit", "xtpoisson"}
        and spec.options.get("model", "re") == "re"
    ):
        result.extra["group_state"] = capture_contract(result, frame.sample)
    elif (
        spec.estimator in {"areg", "reghdfe", "ivreghdfe", "ppmlhdfe"}
        or spec.estimator in {"xtreg", "xtivreg"}
        and spec.options.get("model", "fe") == "fe"
    ):
        if fitted is not None:
            if spec.estimator in {"ivreghdfe", "xtivreg"}:
                endogenous = spec.columns.get("endogenous", [])
                predictors = [
                    *spec.predictors,
                    *([endogenous] if isinstance(endogenous, str) else endogenous),
                ]
            else:
                predictors = spec.predictors
            design = frame.design(predictors=predictors)
            result.extra["group_state"] = capture_fixed(
                result, frame.sample, fitted, design.x, design.terms
            )
    return result


def capture_replayed_contract(result, sample, keep=None):
    raw = result.spec.columns.get("group") or result.spec.panel
    columns = ([raw] if isinstance(raw, str) else list(raw))[:1]
    builder = Builder(columns, "complete_group")
    try:
        for batch in sample.batches():
            frame = batch.frame if keep is None else batch.frame.loc[keep(batch.frame)]
            for _, block in frame.groupby(columns[0], sort=False, observed=True):
                builder.add(
                    keys(block.iloc[:1], columns)[0],
                    {
                        "n_rows": len(block),
                        "row_sum": hex(row_sum(result, block)),
                        "success_count": int(block[result.spec.outcome].sum())
                        if result.spec.estimator in {"clogit", "xtlogit"}
                        and result.spec.options.get("model", "fe") == "fe"
                        else None,
                        "label": encode(block[columns[0]].iloc[0]),
                    },
                )
            builder.flush()
        return builder.finish(
            contract_columns=_contract_columns(result),
            complete_set="all retained fitted rows; SHA256 row-hash multiset sum plus exact count",
            capture="bounded projected complete replay; no chart preview or collection",
        )
    finally:
        builder.close()


def _mean_design(result, frame):
    from .linear_prediction import _coding

    predictors = list(result.spec.predictors)
    if result.spec.estimator in {"ivreghdfe", "xtivreg"}:
        raw = result.spec.columns.get("endogenous", [])
        predictors.extend([raw] if isinstance(raw, str) else raw)
    features, _ = _coding(result, predictors)
    columns = []
    for coefficient in result.coefficients:
        kind, value = features[coefficient.term]
        if kind == "constant":
            columns.append(torch.ones(len(frame), dtype=torch.float64))
        elif kind == "numeric":
            columns.append(torch.tensor(frame[value].to_numpy(dtype=float), dtype=torch.float64))
        else:
            columns.append(
                torch.tensor((frame[value[0]] == value[1]).tolist(), dtype=torch.float64)
            )
    return torch.stack(columns, 1)


def capture_fixed_replay(result, factory):
    """Capture actual fitted means and level-space full inference in replay blocks.

    factory yields (retained raw frame, actual full fitted response). No row
    collection occurs. Only bounded nuisance-level moments become resident.
    """
    spec = result.spec
    raw = spec.panel if spec.estimator in {"xtreg", "xtivreg"} else spec.columns["absorb"]
    columns = [raw] if isinstance(raw, str) else list(raw)
    builder = Builder(columns, "fixed_effect_sum")
    nonlinear = spec.estimator == "ppmlhdfe"
    beta = torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)
    try:
        for frame, fitted in factory():
            x = _mean_design(result, frame)
            effect = torch.as_tensor(fitted, dtype=torch.float64)
            if nonlinear:
                effect = effect.log()
                for role in ("offset", "exposure"):
                    name = spec.columns.get(role)
                    if name:
                        val = torch.tensor(frame[name].to_numpy(dtype=float), dtype=torch.float64)
                        effect = effect - (val.log() if role == "exposure" else val)
            effect = effect - x @ beta
            labels = list(frame[columns].itertuples(index=False, name=None))
            for i, key in enumerate(keys(frame, columns)):
                builder.add(key, {"effect": float(effect[i]), "label": encode(labels[i])})
            builder.flush()
        # Discover at most 1024 nuisance levels, without retaining joint keys.
        levels = [{} for _ in columns]
        width = 0
        interior = spec.covariance == "nonrobust"
        for _, payload in builder.db.execute("SELECT key,record FROM effects"):
            labels = decode(json.loads(payload)["label"])
            for j, label in enumerate(labels):
                key = encode_cluster_labels(pd.Series([label]))[0]
                if key not in levels[j]:
                    levels[j][key] = width
                    width += 1
            if width > 1024:
                interior = False
                break
        joint = None
        if interior:
            k = len(beta)
            budget = min(128 * 1024**2, workspace_budget_bytes())
            fixed = 96 * width**2 + 64 * width * k
            if fixed < budget // 2:
                rows = max(1, min(4096, (budget - fixed) // (64 * (width + k + 4))))
                plan_workspace(
                    "replayed saved full FE inference",
                    {"nuisance_moments": fixed, "row_buffers": rows * 64 * (width + k + 4)},
                    budget_bytes=budget,
                )
                gram = torch.zeros((width, width), dtype=torch.float64)
                cross = torch.zeros((width, k), dtype=torch.float64)
                for frame, fitted in factory():
                    for start in range(0, len(frame), rows):
                        block = frame.iloc[start : start + rows]
                        x = _mean_design(result, block)
                        w = (
                            torch.ones(len(block), dtype=torch.float64)
                            if not spec.weights
                            else torch.tensor(
                                block[spec.weights].to_numpy(dtype=float), dtype=torch.float64
                            )
                        )
                        if nonlinear:
                            w = (
                                w
                                * torch.as_tensor(fitted, dtype=torch.float64)[start : start + rows]
                            )
                        f = torch.zeros((len(block), width), dtype=torch.float64)
                        for j, name in enumerate(columns):
                            ids = [levels[j][key] for key in encode_cluster_labels(block[name])]
                            f[torch.arange(len(block)), ids] = 1.0
                        gram += f.T @ (w[:, None] * f)
                        cross += f.T @ (w[:, None] * x)
                eig, vec = torch.linalg.eigh(gram)
                cutoff = float(eig.max()) * width * torch.finfo(torch.float64).eps * 16
                inverse = torch.where(eig > cutoff, eig.clamp_min(cutoff).reciprocal(), 0.0)
                bread = (vec * inverse) @ vec.T
                projected = bread @ cross
                scale = 1.0 if nonlinear else result.metrics["rmse"] ** 2
                joint = {
                    "covariance": (bread * scale).tolist(),
                    "nuisance_width": width,
                    "mean_terms": [c.term for c in result.coefficients],
                    "basis": "orthogonal nuisance projection; full slope/nuisance information via X-P_FE X",
                }
                for key, payload in builder.db.execute("SELECT key,record FROM effects"):
                    record = json.loads(payload)
                    ids = [
                        levels[j][encode_cluster_labels(pd.Series([label]))[0]]
                        for j, label in enumerate(decode(record["label"]))
                    ]
                    record.update(projection=projected[ids].sum(0).tolist(), nuisance=ids)
                    builder.db.execute(
                        "UPDATE effects SET record=? WHERE key=?",
                        (json.dumps(record, separators=(",", ":")), key),
                    )
        return builder.finish(
            full_inference=joint,
            response_scale="log" if nonlinear else "identity",
            inference_domain="full nonrobust level-space information when its bounded geometry fits; otherwise saved point means only",
            capture="actual fitted means on complete verified projected replays; no row collection",
        )
    finally:
        builder.close()


def export_group_state(result, path):
    """Copy persistent disk group state with a saved result; return an updated copy."""
    state = result.extra.get("group_state")
    with lookup(state):
        if "records" in state:
            return result.model_copy(deep=True)
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise AnalysisError("output_exists", "Group-state export path already exists.")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(state["path"], target)
        target.chmod(0o600)
        if checksum(target) != state["sha256"]:
            raise AnalysisError("group_state_changed", "Group-state export checksum failed.")
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    output = result.model_copy(deep=True)
    output.extra["group_state"]["path"] = str(target)
    return output
