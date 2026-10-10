#!/usr/bin/env python3
"""Independent development-only NumPy/SciPy oracle for eight saved native outputs.

No OpenEconometrics code or pytest is imported. Original physical rows and the
four parent population counts determine every sampling fraction. Stage-three
centers sampled TSU totals; stage-four centers physical FSU rows inside each
sampled TSU, retaining the actual f1*f2*f3 conditional prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

DESCRIPTIVE = ("mean", "total", "ratio", "proportion")
FAMILIES = ("regress", "logit", "probit", "poisson")
CASES = DESCRIPTIVE + FAMILIES
TABLE_COLUMNS = ["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"]
OLD_ROLES = {
    "psu": "p",
    "ssu": "s",
    "tsu": "j",
    "fsu": "f",
    "strata": "h",
    "population_psu": "N",
    "population_ssu": "M",
    "population_tsu": "L",
    "population_fsu": "K",
}


def identity(value):
    """Independent typed key, with bool distinguished before integral equality."""
    if isinstance(value, (bool, np.bool_)):
        return "bool", bool(value)
    if isinstance(value, Integral):
        return "int", int(value)
    if isinstance(value, Real):
        assert math.isfinite(value)
        return "float", float(value).hex()
    assert isinstance(value, str) and value
    return "str", value


def ordered_groups(values):
    groups = {}
    for position, value in enumerate(values):
        groups.setdefault(identity(value), []).append(position)
    return [np.array(positions, dtype=int) for positions in groups.values()]


def geometry(frame, roles):
    """Build a nested tree of conditional stratum selections from caller frames."""
    names = ("strata", "psu", "ssu_strata", "ssu", "tsu_strata", "tsu", "fsu_strata", "fsu")
    paths = [
        tuple(identity(row[roles[name]]) if roles.get(name) else None for name in names)
        for row in frame.to_dict("records")
    ]
    maps = []
    for stage, frame_name in enumerate(("ssu_frame", "tsu_frame", "fsu_frame"), 1):
        fields = ("stratum", "psu", "ssu_stratum", "ssu", "tsu_stratum", "tsu", "fsu_stratum")[
            : 2 * stage + 1
        ]
        pop = ("population_ssu", "population_tsu", "population_fsu")[stage - 1]
        maps.append(
            {
                tuple(identity(record[k]) if record[k] is not None else None for k in fields): int(
                    record[pop]
                )
                for record in roles[frame_name]
            }
        )
    weights = np.zeros(len(frame))

    def descend(positions, stage, prefix_probability):
        strata = {}
        for i in positions:
            strata.setdefault(paths[i][: 2 * stage + 1], {}).setdefault(
                paths[i][: 2 * stage + 2], []
            ).append(i)
        result = []
        for path, children in strata.items():
            N = (
                int(frame[roles["population_psu"]].iloc[next(iter(children.values()))[0]])
                if stage == 0
                else maps[stage - 1][path]
            )
            fraction = len(children) / N
            units = []
            for rows in children.values():
                if stage == 3:
                    assert len(rows) == 1
                    weights[rows] = 1 / (prefix_probability * fraction)
                    units.append(np.array(rows))
                else:
                    units.append(descend(rows, stage + 1, prefix_probability * fraction))
            result.append((units, fraction))
        return result

    tree = descend(list(range(len(frame))), 0, 1.0)
    return weights, tree


def canonical_sha(value):
    """Hash the documented UTF-8 JSON codec without importing production helpers."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def original_sample_sha(frame, kind, expected, design_sha, spec):
    """Bind selected raw responses/regressors or target primitives to original positions."""
    if kind in DESCRIPTIVE:
        values = expected["primitives"]["primitive_values"]
        denominators = expected["primitives"]["primitive_denominators"]
    else:
        outcome, regressors = spec["args"]
        values = np.zeros((len(frame), 1 + len(regressors)), dtype=float)
        values[expected["selected"]] = frame[[outcome, *regressors]].to_numpy(dtype=float)[
            expected["selected"]
        ]
        denominators = np.ones_like(values)
    return canonical_sha(
        [design_sha, expected["selected"].tolist(), values.tolist(), denominators.tolist()]
    )


def covariance_components(weighted_rows, strata):
    weighted_rows = np.asarray(weighted_rows, dtype=float)
    parts = [np.zeros((weighted_rows.shape[1],) * 2) for _ in range(4)]

    def visit(selections, stage, prefix):
        totals = []
        for units, fraction in selections:
            values = np.array(
                [
                    weighted_rows[u].sum(0)
                    if stage == 3
                    else visit(u, stage + 1, prefix * fraction)
                    for u in units
                ]
            )
            if fraction < 1:
                centered = values - values.mean(0)
                parts[stage] += (
                    prefix
                    * (1 - fraction)
                    * len(values)
                    / (len(values) - 1)
                    * (centered.T @ centered)
                )
            totals.append(values.sum(0))
        return np.array(totals).sum(0)

    visit(strata, 0, 1.0)
    return tuple(parts)


def specification(kind, intercept=True):
    if kind == "ratio":
        return {"args": [["y", "a"], ["den", "den2"]]}
    if kind == "proportion":
        return {"args": ["category"], "categories": ["a", "b", "absent"]}
    if kind in DESCRIPTIVE:
        return {"args": [["y", "a"]]}
    outcome = {"regress": "y", "logit": "binary", "probit": "binary", "poisson": "count"}[kind]
    return {"args": [outcome, ["x", "z"]], "intercept": intercept}


def expected_case(frame, kind, *, roles=None, spec=None, options=None):
    spec = specification(kind) if spec is None else spec
    options = {**(options or {}), **{key: value for key, value in spec.items() if key != "args"}}
    args = spec["args"]
    members = (
        frame[options["domain"]].to_numpy(dtype=bool)
        if options.get("domain")
        else np.ones(len(frame), bool)
    )
    weights, strata = geometry(frame, roles)
    if kind in DESCRIPTIVE:
        names = [args[0]] if isinstance(args[0], str) else list(args[0])
        denominators = list(args[1]) if kind == "ratio" else []
        selected = members & frame[names + denominators].notna().all(1).to_numpy()
        if kind == "proportion":
            categories = options["categories"]
            values = np.array(
                [
                    [identity(value) == identity(category) for category in categories]
                    if selected[position]
                    else [False] * len(categories)
                    for position, value in enumerate(frame[names[0]])
                ],
                dtype=float,
            )
            labels = [
                f"{names[0]}[{i}]={identity(c)[0]}:{identity(c)[1]}"
                for i, c in enumerate(categories)
            ]
        else:
            values = frame[names].fillna(0).to_numpy(dtype=float)
            labels = [f"{y}/{x}" for y, x in zip(names, denominators)] if kind == "ratio" else names
        values[~selected] = 0
        effective = weights * selected
        denominator = (
            frame[denominators].fillna(0).to_numpy(dtype=float)
            if denominators
            else np.ones_like(values)
        )
        if kind == "ratio":
            denominator[~selected] = 0
        numerator = effective @ values
        if kind == "total":
            estimate, influence = numerator, values
        else:
            normalizer = effective @ denominator
            estimate = numerator / normalizer
            influence = selected[:, None] * (values - denominator * estimate) / normalizer
        weighted_rows = weights[:, None] * influence
        primitive = {
            "primitive_values": values,
            "primitive_denominators": denominator,
            "row_influences": weighted_rows,
        }
    else:
        outcome, regressors = args
        selected = members & frame[[outcome, *regressors]].notna().all(1).to_numpy()
        X = frame[regressors].fillna(0).to_numpy(dtype=float)
        intercept = options.get("intercept", True)
        labels = (["_cons"] if intercept else []) + list(regressors)
        if intercept:
            X = np.column_stack((np.ones(len(frame)), X))
        x, y, w = X[selected], frame[outcome].to_numpy(dtype=float)[selected], weights[selected]
        if kind == "regress":
            estimate = np.linalg.solve(x.T @ (w[:, None] * x), x.T @ (w * y))
        else:

            def objective(beta):
                value, scores, _ = likelihood(kind, beta, x, y, w)
                return value / w.sum(), -scores.sum(0) / w.sum()

            fit = optimize.minimize(
                objective,
                np.zeros(x.shape[1]),
                jac=True,
                method="BFGS",
                options={"gtol": 1e-12, "maxiter": 1000},
            )
            root = optimize.root(
                lambda beta: likelihood(kind, beta, x, y, w)[1].sum(0),
                fit.x,
                jac=lambda beta: -likelihood(kind, beta, x, y, w)[2],
                tol=1e-12,
            )
            estimate = root.x
            assert np.max(np.abs(objective(estimate)[1])) < 1e-10
        _, scores, bread = likelihood(kind, estimate, x, y, w)
        full_scores = np.zeros((len(frame), len(estimate)))
        full_scores[selected] = scores
        weighted_rows = np.linalg.solve(bread, full_scores.T).T
        normalization = selected.sum() / w.sum()
        primitive = {
            "primitive_X": x,
            "primitive_y": y,
            "bread": bread * normalization,
            "row_scores": full_scores * normalization,
        }
    parts = covariance_components(weighted_rows, strata)
    return {
        "estimates": estimate,
        "covariance": sum(parts),
        "stage1_covariance": parts[0],
        "stage2_covariance": parts[1],
        "stage3_covariance": parts[2],
        "stage4_covariance": parts[3],
        "selected": selected,
        "members": members,
        "weights": weights,
        "labels": labels,
        "primitives": primitive,
        "df": sum(len(psus) for psus, _ in strata) - len(strata),
        "alpha": options.get("alpha", 0.1),
        "null": np.broadcast_to(options.get("null", 0.0), len(estimate)),
    }


def likelihood(family, beta, x, y, w):
    eta = x @ beta
    if family == "regress":
        factor, curvature = y - eta, np.ones(len(y))
        objective = w @ (factor**2 / 2)
    elif family == "logit":
        p = special.expit(eta)
        factor, curvature = y - p, p * (1 - p)
        objective = w @ (np.logaddexp(0, eta) - y * eta)
    elif family == "probit":
        sign = 2 * y - 1
        logp = special.log_ndtr(sign * eta)
        mills = np.exp(stats.norm.logpdf(eta) - logp)
        factor, curvature = sign * mills, mills * (mills + sign * eta)
        objective = -w @ logp
    else:
        mean = np.exp(eta)
        factor, curvature = y - mean, mean
        objective = w @ (mean - y * eta)
    return objective, (w * factor)[:, None] * x, x.T @ ((w * curvature)[:, None] * x)


def numerical(actual, expected, name):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{name}: unexpected dimensions."
    assert np.isfinite(actual).all(), f"{name}: nonfinite numerical output."
    np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=2e-10, err_msg=name)
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def expected_table(expected):
    estimates, covariance = expected["estimates"], expected["covariance"]
    df, alpha, null = expected["df"], expected["alpha"], expected["null"]
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    result = []
    for i, value in enumerate(estimates):
        statistic = (value - null[i]) / se[i] if df and se[i] else None
        p = 2 * stats.t.sf(abs(statistic), df) if statistic is not None else None
        width = stats.t.ppf(1 - alpha / 2, df) * se[i] if df else (0.0 if se[i] == 0 else None)
        result.append(
            [
                value,
                se[i],
                statistic,
                p,
                None if width is None else value - width,
                None if width is None else value + width,
            ]
        )
    return result


def verify_payload(payload):
    states, tables, inputs = (payload[k] for k in ("states", "poststates", "oracle_inputs"))
    assert set(states) == set(tables) == set(CASES) == set(inputs["cases"])
    frame = pd.DataFrame(inputs["frame"])
    roles = inputs["design_roles"]
    checks = {}
    for kind in CASES:
        wanted = expected_case(
            frame, kind, roles=roles, spec=inputs["cases"][kind], options=inputs["options"]
        )
        state, table = states[kind], tables[kind]
        assert state["method"] == "fully-stratified-four-stage-taylor"
        assert state["design"]["stages"] == 4
        assert state["design"]["validation"]["nobs"] == len(frame)
        saved = {k: v for k, v in state.items() if k != "integrity_sha256"}
        assert canonical_sha(saved) == state["integrity_sha256"]
        assert state["df"] == wanted["df"]
        estimate = state["estimates"] if kind in DESCRIPTIVE else state["coefficients"]
        errors = {
            "estimate": numerical(estimate, wanted["estimates"], kind + " estimate"),
            "covariance": numerical(
                state["covariance"], wanted["covariance"], kind + " covariance"
            ),
        }
        metadata = state["metadata"]
        for field in (
            "stage1_covariance",
            "stage2_covariance",
            "stage3_covariance",
            "stage4_covariance",
        ):
            errors[field] = numerical(metadata[field], wanted[field], field)
        for field, value in wanted["primitives"].items():
            errors[field] = numerical(metadata[field], value, field)
        assert metadata["sample_positions"] == np.flatnonzero(wanted["selected"]).tolist()
        assert metadata["out_of_domain_positions"] == np.flatnonzero(~wanted["members"]).tolist()
        assert (
            metadata["outcome_exclusions"]
            == np.flatnonzero(wanted["members"] & ~wanted["selected"]).tolist()
        )
        assert table["index"] == wanted["labels"] and table["columns"] == TABLE_COLUMNS
        numerical(table["attrs"]["covariance_matrix"], wanted["covariance"], "display covariance")
        for actual, expected in zip(table["data"], expected_table(wanted), strict=True):
            for a, b in zip(actual, expected, strict=True):
                if b is None:
                    assert a is None
                else:
                    numerical(a, b, "full displayed inference")
        checks[kind] = errors
    return dict(
        ok=True,
        cases=8,
        rows=len(frame),
        checks=checks,
        independent="NumPy/SciPy nested conditional strata",
        vendor_parity=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_payload(json.loads(args.payload.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))
