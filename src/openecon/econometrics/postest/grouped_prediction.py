"""Complete-group evaluation on bounded SQLite spools and indexed row output."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import pandas as pd
import torch
import torch.nn.functional as functional

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.postest.group_state import _contract_columns, keys, lookup, row_digest
from openecon.econometrics.postest.index_codec import decode, encode
from openecon.engines.inference import critical_value
from openecon.frame import as_frame
from openecon.resources import plan_workspace


def _error(code, message):
    raise AnalysisError(code, message)


def conditional_probability(eta, successes):
    """Exact conditional inclusion probabilities; no independent row logit."""
    if not 0 < successes < len(eta):
        _error("invalid_success_count", "Conditional groups require both outcomes.")
    reflect = 2 * successes > len(eta)
    signed = -eta if reflect else eta
    m = len(eta) - successes if reflect else successes
    if m > 128:
        _error(
            "conditional_recursion_budget",
            "Conditional recursion is bounded to 128 minority outcomes per group.",
        )
    if m == 1:
        probability = torch.softmax(signed, 0)
    else:
        # Only reachable states are evaluated: logaddexp(-inf,-inf) has no
        # defined derivative and must never enter the differentiation tape.
        state = [signed.sum() * 0]
        for value in signed:
            updated = [state[0]]
            for j in range(1, min(len(state), m) + 1):
                updated.append(
                    value + state[j - 1]
                    if j == len(state)
                    else torch.logaddexp(state[j], value + state[j - 1])
                )
            state = updated
        probability = torch.autograd.grad(state[m], signed, create_graph=True)[0]
    return 1 - probability if reflect else probability


def _clogit(result, frame, interval, alpha):
    from .inference import _parameters
    from .linear_prediction import _coding

    state = _parameters(result)
    features, _ = _coding(result, result.spec.predictors)
    x = torch.stack(
        [
            torch.tensor((frame[value[0]] == value[1]).tolist(), dtype=torch.float64)
            if kind == "category"
            else torch.tensor(frame[value].to_numpy(dtype=float), dtype=torch.float64)
            for term in state.terms
            for kind, value in [features[term]]
        ],
        1,
    )
    offset = torch.zeros(len(frame), dtype=torch.float64)
    if result.spec.columns.get("offset"):
        offset = torch.tensor(
            frame[result.spec.columns["offset"]].to_numpy(dtype=float), dtype=torch.float64
        )
    successes = int(frame[result.spec.outcome].sum())
    minority = min(successes, len(frame) - successes)
    plan_workspace(
        "exact conditional recurrence and derivative tape",
        {"recursion": 64 * len(frame) * minority * (len(state.terms) + 2)},
    )
    work = len(frame) * minority * max(1, len(state.terms)) * (len(frame) if interval else 1)
    if work > 100_000_000:
        _error(
            "group_prediction_budget",
            "Exact conditional derivative work exceeds 100 million operations for one group.",
        )

    def values(beta):
        eta = x @ beta + offset
        # The recurrence differentiates log f with respect to each index.
        if not eta.requires_grad:
            eta = eta.requires_grad_()
        return conditional_probability(eta, successes)

    with torch.enable_grad():
        beta = state.beta.detach().requires_grad_()
        mean = values(beta).detach()
        jac = (
            torch.autograd.functional.jacobian(values, state.beta) if interval is not None else None
        )
    return _interval(
        mean, jac, state.covariance, state.df, state.alpha if alpha is None else alpha, interval
    )


def _interval(mean, jac, covariance, df, alpha, interval):
    output = {"response": mean.detach().tolist()}
    if interval is not None:
        if interval != "mean":
            _error(
                "unsupported_prediction_interval",
                "Only full-parameter delta mean intervals are supported.",
            )
        variance = ((jac @ covariance) * jac).sum(1)
        if bool((variance < -1e-10).any()) or not bool(torch.isfinite(variance).all()):
            _error("invalid_inference", "Group response uncertainty is not finite nonnegative.")
        se = variance.clamp_min(0).sqrt()
        critical = critical_value(alpha, df)
        output.update(
            std_error=se.tolist(),
            ci_low=(mean - critical * se).tolist(),
            ci_high=(mean + critical * se).tolist(),
        )
    return output


def _posterior(result, frame, interval, alpha):
    from .prediction import _model, _encode
    from openecon.econometrics.mixed.glmm_kernels import RandomEffectsGLMM

    model = _model(result)
    design = _encode(model, frame)
    adapter = model.response_adapter
    if result.spec.estimator not in {"melogit", "meprobit", "mepoisson"}:
        _error(
            "unsupported_prediction_target",
            "Complete posterior integration currently supports the saved normal logit/probit/Poisson GLMMs.",
        )
    family = {"melogit": "logit", "meprobit": "probit", "mepoisson": "poisson"}[
        result.spec.estimator
    ]
    fixed = [i for i, term in enumerate(model.state.terms) if term in model.mean_terms]
    x = design.x[:, fixed]
    z = design.random_z
    y = torch.tensor(frame[result.spec.outcome].to_numpy(dtype=float), dtype=torch.float64)
    theta = torch.cat((model.state.beta[fixed], 0.5 * model.state.beta[adapter.variances].log()))
    previous = None
    checks = []
    for order in (8, 16, 32, 64):
        plan_workspace(
            "complete group posterior integration",
            {
                "integration_tape": 32
                * len(frame)
                * order ** len(adapter.names)
                * (len(model.state.terms) + 2)
            },
        )
        objective = RandomEffectsGLMM(
            x,
            y,
            z,
            torch.zeros(len(y), dtype=torch.int64),
            1,
            family,
            order,
            offset=design.deterministic,
            method="mvaghermite",
        )
        with torch.no_grad():
            objective.adapt(theta)
            v, base = objective._nodes()

        def values(beta):
            sigma = beta[adapter.variances].sqrt()
            eta = (design.x @ beta + design.deterministic)[:, None] + z @ (v[0] * sigma).T
            if family == "logit":
                logdensity = -functional.softplus(torch.where(y[:, None] == 1, -eta, eta))
                response = torch.sigmoid(eta)
            elif family == "probit":
                logdensity = torch.special.log_ndtr(torch.where(y[:, None] == 1, eta, -eta))
                response = torch.special.ndtr(eta)
            else:
                logdensity = y[:, None] * eta - eta.exp() - torch.lgamma(y[:, None] + 1)
                response = eta.exp()
            weights = torch.softmax(base[0] + logdensity.sum(0), 0)
            return (response * weights).sum(1)

        with torch.enable_grad():
            mean = values(model.state.beta).detach()
            jac = torch.autograd.functional.jacobian(values, model.state.beta).detach()
        if previous is not None:
            value_error = float((mean - previous[0]).abs().max())
            gradient_error = float((jac - previous[1]).abs().max())
            checks.append(
                {
                    "nodes_per_dimension": order,
                    "maximum_mean_change": value_error,
                    "maximum_gradient_change": gradient_error,
                }
            )
            if value_error < 1e-8 and gradient_error < 1e-7:
                return _interval(
                    mean,
                    jac,
                    model.state.covariance,
                    model.state.df,
                    model.state.alpha if alpha is None else alpha,
                    interval,
                ), checks
        previous = mean, jac
    _error(
        "integration_not_converged",
        "Complete-group posterior values and full variance-parameter gradients did not settle at 64 adaptive nodes per dimension.",
    )


def group_predict(
    result,
    data,
    *,
    kind="response",
    target="conditional",
    interval=None,
    alpha=None,
    batch_rows=None,
    max_group_rows=100000,
    max_disk_bytes=1073741824,
):
    """Evaluate retained fitted groups without collecting the entire Dataset.

    Complete counts and row-multiset digests protect alternatives and posterior
    conditioning. Group output and row output use separate owned Parquet files.
    """
    from .streaming_prediction import materialize_predictions

    if data is None:
        _error("prediction_data_required", "Pass explicit complete evaluation groups.")
    estimator = result.spec.estimator
    mixed = estimator == "mixed"
    conditional = (
        estimator == "clogit"
        or estimator == "xtlogit"
        and result.spec.options.get("model", "re") == "fe"
    )
    if not (mixed or conditional or target == "posterior"):
        _error(
            "unsupported_prediction_target",
            "No complete-group adapter is registered for this target.",
        )
    if not mixed and kind not in {"response", "mean", "fitted"}:
        _error(
            "unsupported_prediction_kind",
            "Complete-group likelihood evaluation returns conditional/posterior response probabilities or means.",
        )
    if mixed and kind not in {"fitted", "reffects"}:
        _error(
            "unsupported_prediction_kind", "Complete-group mixed output uses fitted or reffects."
        )
    if mixed and interval is not None:
        _error(
            "unsupported_prediction_interval",
            "mixed BLUP output has no full hyperparameter uncertainty adapter.",
        )
    if conditional and target != "conditional":
        _error(
            "unsupported_prediction_target",
            "Conditional logit conditions on the recorded group success count.",
        )
    for name, value, limit in [
        ("max_group_rows", max_group_rows, 1000000),
        ("batch_rows", batch_rows or 4096, 65536),
        ("max_disk_bytes", max_disk_bytes, 16 * 1024**3),
    ]:
        if type(value) is not int or not 1 <= value <= limit:
            _error(
                "invalid_prediction_budget",
                f"{name} must be a positive integer no greater than {limit}.",
            )
    state = result.extra.get("group_state")
    with lookup(state):
        if state.get("kind") != "complete_group":
            _error(
                "missing_group_state", "Complete-group evaluation needs the fitted row contract."
            )
    source = data if isinstance(data, Dataset) else Dataset.from_frame(data)
    rows_per_batch = batch_rows or 4096
    columns = list(dict.fromkeys([*state["group_columns"], *_contract_columns(result)]))
    if mixed:
        from openecon.econometrics.registry import spec_columns

        columns = list(dict.fromkeys([*columns, *spec_columns(result.spec)]))
    if set(columns) - set(source.columns):
        _error(
            "missing_columns",
            "Complete-group evaluation lacks required outcome, group or model columns.",
        )
    plan_workspace(
        "group spool projected replay",
        {
            "SQLite_and_writer_caches": 8 * 1024**2,
            "projected_rows": rows_per_batch * 256 * (len(columns) + 2),
        },
    )
    scratch = tempfile.TemporaryDirectory(
        prefix="openecon-group-eval-", dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None
    )
    path = Path(scratch.name) / "groups.sqlite3"
    db = None
    try:
        db = sqlite3.connect(path)
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA cache_size=-2048")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA mmap_size=0")
        db.execute("CREATE TABLE rows(position INTEGER PRIMARY KEY,key BLOB,record TEXT,idx TEXT)")
        db.execute("CREATE INDEX group_rows ON rows(key,position)")
        db.execute("CREATE TABLE output(position INTEGER PRIMARY KEY,value TEXT)")
        db.execute("CREATE TABLE effects(position INTEGER PRIMARY KEY,value TEXT)")
        path.chmod(0o600)
    except BaseException:
        if db is not None:
            db.close()
        scratch.cleanup()
        raise
    maximum = 0
    checks = []
    digest = hashlib.sha256()
    index_template = None
    try:
        position = 0
        for batch in source.iter_batches(columns, batch_rows=rows_per_batch):
            if index_template is None:
                index_template = batch.index[:0]
            digest.update(
                pd.util.hash_pandas_object(batch, index=True, categorize=False)
                .to_numpy(dtype="<u8")
                .tobytes()
            )
            complete = ~batch[columns].isna().any(axis=1)
            if result.spec.weights:
                weight = batch[result.spec.weights]
                if bool((weight < 0).any()):
                    _error("invalid_weights", "Complete-group source weights must be nonnegative.")
                complete &= weight > 0
            if not bool(complete.all()) and result.spec.missing == "raise":
                _error("missing_values", "Complete-group evaluation inputs contain missing values.")
            retained = batch.loc[complete]
            group_keys = iter(keys(retained, state["group_columns"]))
            records = []
            for local, (idx, row) in enumerate(
                zip(batch.index, batch[columns].itertuples(index=False, name=None), strict=True)
            ):
                valid = bool(complete.iloc[local])
                key = next(group_keys) if valid else None
                records.append(
                    (position + local, key, encode(tuple(row)) if valid else None, encode(idx))
                )
                if not valid:
                    empty = {
                        name: None
                        for name in (["fitted"] if mixed else ["response"])
                        + (["std_error", "ci_low", "ci_high"] if interval else [])
                    }
                    db.execute(
                        "INSERT INTO output VALUES(?,?)", (position + local, json.dumps(empty))
                    )
            db.executemany("INSERT INTO rows VALUES(?,?,?,?)", records)
            position += len(batch)
            db.commit()
            if path.stat().st_size > max_disk_bytes:
                _error(
                    "prediction_disk_budget",
                    "Complete group spool exceeds max_disk_bytes; no rows were truncated.",
                )
        with lookup(state) as get:
            effect_position = 0
            cursor = db.execute(
                "SELECT key,COUNT(*) FROM rows WHERE key IS NOT NULL GROUP BY key ORDER BY key"
            )
            for key, count in cursor:
                if count > max_group_rows:
                    _error(
                        "group_prediction_budget",
                        "One complete top group exceeds max_group_rows; increase the explicit bound or reduce the model grouping domain.",
                    )
                plan_workspace(
                    "one complete prediction group",
                    {
                        "projected_and_derivative_rows": count
                        * 256
                        * (len(columns) + len(result.coefficients) + 4)
                    },
                )
                rows = db.execute(
                    "SELECT position,record,idx FROM rows WHERE key=? ORDER BY position", (key,)
                ).fetchall()
                frame = pd.DataFrame([decode(row[1]) for row in rows], columns=columns)
                contract = get(key)
                if contract is None:
                    _error(
                        "unknown_group",
                        "Evaluation top group is absent from the fitted complete-group state.",
                    )
                if (
                    contract["n_rows"] != count
                    or contract["row_digest"] != row_digest(result, frame)
                    or conditional
                    and contract["success_count"] != int(frame[result.spec.outcome].sum())
                ):
                    _error(
                        "incomplete_group",
                        "Evaluation must contain the exact retained fitted alternative/conditioning row multiset and success count for each requested group.",
                    )
                maximum = max(maximum, count)
                if mixed:
                    from openecon.econometrics.mixed.predict import mixed_predict

                    for name, coding in result.provenance.get("categorical_encoding", {}).items():
                        if name in frame:
                            frame[name] = pd.Categorical(frame[name], categories=coding["levels"])
                    values = mixed_predict(result, frame, kind=kind)
                    if kind == "reffects":
                        for record in values.to_dict("records"):
                            record["group_identity"] = encode(record.pop("group"))
                            record["top_group_identity"] = contract["label"]
                            db.execute(
                                "INSERT INTO effects VALUES(?,?)",
                                (effect_position, json.dumps(record)),
                            )
                            effect_position += 1
                        continue
                    output = {"fitted": values.fitted.tolist()}
                elif conditional:
                    output = _clogit(result, frame, interval, alpha)
                else:
                    output, local_checks = _posterior(result, frame, interval, alpha)
                    if len(checks) < 100:
                        checks.append({"group": contract["label"], "checks": local_checks})
                db.executemany(
                    "INSERT INTO output VALUES(?,?)",
                    (
                        (
                            row[0],
                            json.dumps(
                                {name: values[i] for name, values in output.items()},
                                allow_nan=False,
                            ),
                        )
                        for i, row in enumerate(rows)
                    ),
                )
                db.commit()
                if path.stat().st_size > max_disk_bytes:
                    _error(
                        "prediction_disk_budget", "Group input/output spool exceeds max_disk_bytes."
                    )
        # Independent complete replay catches mutable factories and files before
        # returning a successful owned output. No replay is collected.
        verification = hashlib.sha256()
        for batch in source.iter_batches(columns, batch_rows=rows_per_batch):
            verification.update(
                pd.util.hash_pandas_object(batch, index=True, categorize=False)
                .to_numpy(dtype="<u8")
                .tobytes()
            )
        if verification.digest() != digest.digest():
            _error(
                "source_changed",
                "Complete-group evaluation source changed between verified passes.",
            )
        definition = (
            "bounded posterior BLUP at saved hyperparameters"
            if mixed
            else "exact conditional probability given the fitted complete alternative set and success count"
            if conditional
            else "complete-group posterior response integrated over saved normal random effects"
        )

        def frames():
            cursor = (
                db.execute("SELECT value FROM effects ORDER BY position")
                if kind == "reffects"
                else db.execute(
                    "SELECT output.value,rows.idx FROM output JOIN rows USING(position) ORDER BY position"
                )
            )
            while True:
                part = cursor.fetchmany(rows_per_batch)
                if not part:
                    break
                frame = pd.DataFrame([json.loads(row[0]) for row in part])
                if kind != "reffects":
                    frame = frame.astype(float)
                    labels = [decode(row[1]) for row in part]
                    frame.index = (
                        pd.MultiIndex.from_tuples(labels, names=index_template.names)
                        if index_template.nlevels > 1
                        else pd.Index(
                            labels,
                            dtype=index_template.dtype,
                            name=index_template.name,
                            tupleize_cols=False,
                        )
                    )
                frame.attrs.update(
                    response_definition=definition,
                    missing_row_positions=frame.index[frame.isna().all(axis=1)].tolist(),
                )
                yield frame

        output = materialize_predictions(
            frames(),
            {
                "estimator": estimator,
                "kind": kind,
                "target": target,
                "complete_groups": True,
                "maximum_group_rows": maximum,
                "max_group_rows": max_group_rows,
                "source_digest": digest.hexdigest(),
                "source_passes": 2,
                "quadrature_checks_sample": checks,
                "spool_disk_bytes": path.stat().st_size,
                "max_disk_bytes": max_disk_bytes,
            },
        )
        if not isinstance(data, Dataset):
            plan_workspace(
                "resident group prediction output",
                {"rows": 128 * max(position, effect_position) * max(1, len(output.columns))},
            )
            return as_frame(pd.concat(list(output.iter_batches(batch_rows=rows_per_batch))))
        return output
    finally:
        db.close()
        scratch.cleanup()
