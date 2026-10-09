"""Exact transformed Gaussian regressions on bounded Dataset replays.

Training quantiles use disk B-tree order statistics with linear interpolation.
All numerical estimation is native float64 Torch: one global augmented TSQR,
small SVD solves and a full-source residual replay. MFP uses a single master
basis, preserving every candidate's complete-sample objective. Its adaptive
closed F tests remain approximate and final inference remains conditional.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

import torch

from openecon.engines.distributions import f_sf
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.streaming_design import numeric_values
from .replay_sample import ReplaySample
from .smoothing.common import DT, fail, finite, seal
from .smoothing.fractional import POWERS, fp_basis
from .smoothing.replay import design
from .smoothing.splines import knot_options, numbers
from .streaming_linear import _Notes, _result

SUPPORTED = frozenset({"bspline_regress", "rcs_regress", "fp_regress", "mfp_regress"})


def _quantiles(sample, probabilities):
    """Exact type-7 order statistics, never a sampled/sketched quantile."""
    scratch = os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None
    parent = Path(scratch) if scratch else Path(tempfile.gettempdir())
    required = 96 * sample.nrows * len(sample.spec.predictors) + 1024**2
    if shutil.disk_usage(parent).free < required:
        fail("scratch_limit", f"Exact training order statistics reserve {required} scratch bytes.")
    with tempfile.TemporaryDirectory(prefix="openecon-spline-knots-", dir=scratch) as directory:
        path = Path(directory) / "values.sqlite"
        connection = sqlite3.connect(path)
        path.chmod(0o600)
        try:
            connection.execute("PRAGMA cache_size=-4096")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            for j in range(len(sample.spec.predictors)):
                connection.execute(f"CREATE TABLE q{j}(value REAL NOT NULL)")
            for batch in sample.batches():
                for j, name in enumerate(sample.spec.predictors):
                    values = batch.numeric(name)
                    connection.executemany(
                        f"INSERT INTO q{j} VALUES (?)", ((v,) for v in values.tolist())
                    )
            result = {}
            for j, name in enumerate(sample.spec.predictors):
                connection.execute(f"CREATE INDEX order{j} ON q{j}(value)")
                ordered = []
                for probability in probabilities:
                    location = (sample.nrows - 1) * probability
                    lower, upper = math.floor(location), math.ceil(location)
                    a = connection.execute(
                        f"SELECT value FROM q{j} ORDER BY value LIMIT 1 OFFSET ?", (lower,)
                    ).fetchone()[0]
                    b = connection.execute(
                        f"SELECT value FROM q{j} ORDER BY value LIMIT 1 OFFSET ?", (upper,)
                    ).fetchone()[0]
                    # torch.quantile's default linear interpolation.
                    ordered.append(a + (b - a) * (location - lower))
                result[name] = ordered
            connection.commit()
            disk_bytes = path.stat().st_size
        finally:
            connection.close()
    return result, {
        "algorithm": "exact disk B-tree order statistics; linear type-7 interpolation",
        "disk_bytes": disk_bytes,
        "reserved_disk_bytes": required,
        "temporary_state_removed": True,
    }


def _state(sample, notes):
    spec = sample.spec
    if spec.estimator in {"fp_regress", "mfp_regress"}:
        powers = notes.option("powers") if spec.estimator == "fp_regress" else POWERS
        if spec.estimator == "fp_regress" and (
            len(powers) not in (1, 2) or any(p not in POWERS for p in powers)
        ):
            fail("invalid_powers", "FP supports one/two powers from [-2,-1,-.5,0,.5,1,2,3].")
        powers = sorted(powers)
        adjustments = spec.predictors[1:]
        return {
            "terms": [
                "_cons",
                *[f"{spec.predictors[0]}:fp{j + 1}({p:g})" for j, p in enumerate(powers)],
                *adjustments,
            ],
            "transforms": [
                {
                    "kind": "fp",
                    "column": spec.predictors[0],
                    "powers": powers,
                    "scale": notes.option("scale"),
                },
                *[{"kind": "linear", "column": col} for col in adjustments],
            ],
        }, None
    raw = knot_options(spec)
    natural = spec.estimator == "rcs_regress"
    boundary = spec.options.get("boundary")
    if boundary is not None and (
        not isinstance(boundary, dict) or set(boundary) != set(spec.predictors)
    ):
        fail("invalid_knots", "boundary must map exactly every predictor to [lower, upper].")
    count = notes.option("n_knots")
    probabilities = (
        torch.linspace(0.05, 0.95, count, dtype=DT).tolist()
        if natural
        else torch.linspace(0, 1, count + 2, dtype=DT).tolist()
    )
    discovery, disk = (None, None)
    if raw is None or not natural and boundary is None:
        discovery, disk = _quantiles(sample, probabilities if raw is None else [0.0, 1.0])
    transforms, terms = [], ["_cons"]
    for name in spec.predictors:
        if natural:
            knots = numbers(raw[name] if raw is not None else discovery[name], "knots")
            if len(knots) < 3:
                fail(
                    "invalid_knots",
                    "Restricted cubic splines need at least three distinct knots including boundaries.",
                )
            record, width = {"kind": "rcs", "column": name, "knots": knots}, len(knots) - 1
        else:
            bounds = numbers(
                boundary[name]
                if boundary is not None
                else [discovery[name][0], discovery[name][-1]],
                "boundary",
                2,
            )
            if len(bounds) != 2:
                fail("invalid_knots", "Boundary requires two distinct ordered values.")
            knots = numbers(raw[name] if raw is not None else discovery[name][1:-1], "knots")
            if any(v <= bounds[0] or v >= bounds[1] for v in knots):
                fail("invalid_knots", "Interior knots must lie strictly inside the boundary.")
            degree = notes.option("degree")
            record, width = (
                {
                    "kind": "bs",
                    "column": name,
                    "knots": knots,
                    "boundary": bounds,
                    "degree": degree,
                },
                len(knots) + degree,
            )
        transforms.append(record)
        terms.extend(f"{name}:basis{j + 1}" for j in range(width))
    return {"terms": terms, "transforms": transforms}, disk


def _master(frame, state):
    """MFP contains all eight simple and all eight repeated-power terms."""
    first = state["transforms"][0]
    x = numeric_values(frame[first["column"]], first["column"])
    basic = torch.cat([fp_basis(x, [p], first["scale"]) for p in POWERS], 1)
    repeated = torch.cat([fp_basis(x, [p, p], first["scale"])[:, 1:] for p in POWERS], 1)
    adjustment = [
        numeric_values(frame[t["column"]], t["column"])[:, None] for t in state["transforms"][1:]
    ]
    return finite(
        torch.cat([torch.ones((len(frame), 1), dtype=DT), basic, repeated, *adjustment], 1)
    )


def _solve(factor, n):
    x, y = factor[:, :-1], factor[:, -1]
    u, s, vt = torch.linalg.svd(x, full_matrices=False)
    if (
        n <= x.shape[1]
        or len(s) < x.shape[1]
        or float(s[-1]) <= float(s[0]) * max(n, x.shape[1]) * torch.finfo(DT).eps * 10
    ):
        fail(
            "rank_deficient",
            "Every declared transformed column must be identified with positive residual df.",
        )
    beta = finite(vt.T @ ((u.T @ y) / s))
    inverse = finite((vt.T / s.square()) @ vt)
    rss = float(finite((y - x @ beta).square()).sum())
    return beta, rss, inverse


def _select(factor, sample, state, notes):
    candidates, indexes = [], []
    adjustment = list(range(17, factor.shape[1] - 1))
    for powers in [
        [],
        *[[p] for p in POWERS],
        *[list(p) for p in combinations_with_replacement(POWERS, 2)],
    ]:
        selected = [0]
        for j, p in enumerate(powers):
            selected.append(
                9 + POWERS.index(p) if j and p == powers[j - 1] else 1 + POWERS.index(p)
            )
        selected += adjustment
        _, rss, _ = _solve(factor[:, [*selected, factor.shape[1] - 1]], sample.nrows)
        candidates.append({"powers": powers, "rss": rss, "n_parameters": len(selected)})
        indexes.append(selected)
    best1 = min(range(1, 9), key=lambda j: (candidates[j]["rss"], j))
    best2 = min(range(9, 45), key=lambda j: (candidates[j]["rss"], j))
    full = candidates[best2]
    df = sample.nrows - full["n_parameters"]
    if full["rss"] <= 0:
        fail(
            "invalid_dispersion",
            "Closed F comparisons require positive best-FP2 residual variance.",
        )
    linear = next(j for j, c in enumerate(candidates) if c["powers"] == [1.0])
    tests = []
    for reduced, dnum, name in [
        (0, 4, "inclusion"),
        (linear, 3, "nonlinearity"),
        (best1, 2, "simplification"),
    ]:
        f = max(0.0, (candidates[reduced]["rss"] - full["rss"]) / dnum) / (full["rss"] / df)
        tests.append(
            {
                "test": name,
                "statistic": f,
                "df_num": dnum,
                "df_den": df,
                "p_value": f_sf(f, dnum, df),
                "reduced_candidate": reduced,
                "full_candidate": best2,
                "reference": "approximate Gaussian closed-test F; adaptive power search is not exact nested-model testing",
            }
        )
    chosen = (
        0
        if tests[0]["p_value"] >= notes.option("select_alpha")
        else linear
        if tests[1]["p_value"] >= notes.option("form_alpha")
        else best1
        if tests[2]["p_value"] >= notes.option("form_alpha")
        else best2
    )
    powers = candidates[chosen]["powers"]
    state["transforms"][0]["powers"] = powers
    state["terms"] = [
        "_cons",
        *[f"{sample.spec.predictors[0]}:fp{j + 1}({p:g})" for j, p in enumerate(powers)],
        *sample.spec.predictors[1:],
    ]
    state.update(
        candidates=candidates,
        selected=chosen,
        closed_tests=tests,
        select_alpha=notes.option("select_alpha"),
        form_alpha=notes.option("form_alpha"),
        selection_scope="one nonlinear variable; other predictors always linear and retained",
    )
    return factor[:, [*indexes[chosen], factor.shape[1] - 1]]


def fit_streaming(spec, source, *, batch_rows=None):
    if spec.estimator not in SUPPORTED:
        fail(
            "streaming_unsupported",
            "This transformed Dataset route supports B/RCS splines and FP/MFP only.",
        )
    if spec.weights is not None or spec.categorical or spec.covariance != "nonrobust":
        fail(
            "unsupported_streaming_option",
            "Transformed Gaussian Dataset fits are unweighted numeric iid OLS only.",
        )
    if not 1 <= len(spec.predictors) <= 20:
        fail("unsupported_dimensions", "Smoothing requires 1..20 predictors.")
    notes = _Notes(spec)
    raw = knot_options(spec) if spec.estimator in {"bspline_regress", "rcs_regress"} else None
    maxknots = max(map(len, raw.values())) if raw else notes.option("n_knots") or 0
    width = (
        16 + len(spec.predictors)
        if spec.estimator == "mfp_regress"
        else len(spec.predictors) + 2
        if spec.estimator == "fp_regress"
        else 1 + len(spec.predictors) * (maxknots + 3)
    )
    sample = ReplaySample(spec, source, batch_rows=batch_rows).prepare()
    if sample.nrows < 8:
        fail("unsupported_dimensions", "Smoothing requires at least8 complete observations.")
    work = (
        8 * sample.nrows * (width + 1) ** 2 + 45 * (width + 1) ** 3
        if spec.estimator == "mfp_regress"
        else 8 * sample.nrows * (width + 1) ** 2
    )
    if work > notes.option("max_work"):
        fail(
            "work_limit",
            f"Smoothing replay plans {work} scalar work units; increase max_work explicitly.",
        )
    resource = sample.plan_rows(
        "transformed Gaussian TSQR, disk knots and full inference",
        {
            "transformed_factors_SVD_covariance": 1024 * (width + 1) ** 2,
            "disk_order_statistics_cache": 8 * 1024**2,
        },
        192 * (width + len(spec.predictors) + 2),
    ).record()
    state, disk = _state(sample, notes)
    tree = _TSQRTree()
    for batch in sample.batches():
        matrix = (
            _master(batch.frame, state)
            if spec.estimator == "mfp_regress"
            else design(batch.frame, state)
        )
        block = finite(torch.cat([matrix, batch.numeric(spec.outcome)[:, None]], 1))
        _, factor = torch.linalg.qr(block, mode="r")
        tree.add(factor)
    factor = tree.finish()
    if spec.estimator == "mfp_regress":
        factor = _select(factor, sample, state, notes)
    beta, _, inverse = _solve(factor, sample.nrows)
    residual = _CompensatedSum(())
    predictions = []
    for batch in sample.batches():
        y = batch.numeric(spec.outcome)
        fitted = finite(design(batch.frame, state) @ beta)
        errors = finite(y - fitted)
        residual.add(finite(errors.square().sum()))
        take = min(400 - len(predictions), len(y))
        predictions.extend(
            {"row": row, "observed": o, "fitted": f, "residual": e}
            for row, o, f, e in zip(
                batch.positions[:take].tolist(),
                y[:take].tolist(),
                fitted[:take].tolist(),
                errors[:take].tolist(),
                strict=True,
            )
        )
    rss, df = float(residual.value), sample.nrows - len(beta)
    covariance = finite(rss / df * inverse)
    if not math.isfinite(rss):
        fail("numerical_failure", "Residual sum of squares exceeds finite float64 arithmetic.")
    note = "Conditional on fixed transformation/model; selection uncertainty is excluded."
    notes.warn(note)
    state = seal(
        {
            **state,
            "columns": spec.predictors,
            "coefficients": beta.tolist(),
            "covariance": covariance.tolist(),
            "df_resid": df,
            "sample_positions": sample.sample_positions,
            "sample_positions_scope": "first400 retained positions only; complete sample hash in ResultBundle provenance",
            "sample_position_count": sample.nrows,
            "method": spec.estimator,
        }
    )
    return _result(
        sample,
        terms=state["terms"],
        beta=beta,
        covariance=covariance,
        info={
            "available": True,
            "distribution": "t",
            "df_inference": df,
            "df_resid": df,
            "conditioning": note,
            "correction": "iid Gaussian OLS",
            "covariance": "nonrobust",
        },
        metrics={"rss": rss, "training_mse": rss / sample.nrows},
        notes=notes,
        predictions=predictions,
        tests={
            "closed_tests": state["closed_tests"],
            "selected_powers": state["transforms"][0]["powers"],
        }
        if spec.estimator == "mfp_regress"
        else {},
        solver="global float64 augmented TSQR and SVD; complete-source residual replay",
        diagnostics={
            "tsqr_reduction_depth": tree.depth,
            "tsqr_peak_factors": tree.peak_factors,
            "planned_scalar_work": work,
            "order_statistics": disk,
            "full_source_collected": False,
        },
        extra={"smoothing_state": state, "notes": [note]},
        resource=resource,
    )
