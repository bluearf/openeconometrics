#!/usr/bin/env python3
"""Independent development-only NumPy/SciPy oracle for eight saved native outputs.

No OpenEconometrics code or pytest is imported. Original physical rows and the
explicit positive SSU- and TSU-cell frames determine every sampling fraction.
Stage-2 centers SSU totals within its cell; stage-3 centers rows separately in
each terminal cell and retains its actual parent cell's second-stage fraction
in the f1*f2 prefix.
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
ROLES = {
    "psu": "p",
    "ssu_strata": "g",
    "ssu": "s",
    "tsu_strata": "q",
    "tsu": "j",
    "strata": "h",
    "population_psu": "N",
    "population_ssu": "M",
    "population_tsu": "L",
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


def ssu_frame(frame, roles=ROLES):
    """Explicit positive-cell declaration, independently assembled from this fixture."""
    names = ([roles["strata"]] if roles.get("strata") else []) + [roles["psu"], roles["ssu_strata"]]
    cells = {}
    for row in frame.to_dict("records"):
        key = tuple(identity(row[name]) for name in names)
        record = {name: row[name] for name in names}
        record[roles["population_ssu"]] = row[roles["population_ssu"]]
        assert key not in cells or cells[key] == record
        cells.setdefault(key, record)
    return list(cells.values())


def tsu_frame(frame, roles=ROLES):
    """Explicit positive terminal cells inside every observed SSU."""
    names = ([roles["strata"]] if roles.get("strata") else []) + [
        roles["psu"],
        roles["ssu_strata"],
        roles["ssu"],
        roles["tsu_strata"],
    ]
    cells = {}
    for row in frame.to_dict("records"):
        key = tuple(identity(row[name]) for name in names)
        record = {name: row[name] for name in names}
        record[roles["population_tsu"]] = row[roles["population_tsu"]]
        assert key not in cells or cells[key] == record
        cells.setdefault(key, record)
    return list(cells.values())


def geometry(frame, roles=ROLES, declared_frame=None, terminal_frame=None):
    """Independently group original rows; each lower cell has its own fraction."""
    declared_frame = ssu_frame(frame, roles) if declared_frame is None else declared_frame
    terminal_frame = tsu_frame(frame, roles) if terminal_frame is None else terminal_frame
    parents = ([roles["strata"]] if roles.get("strata") else []) + [
        roles["psu"],
        roles["ssu_strata"],
    ]
    leaves = parents + [roles["ssu"], roles["tsu_strata"]]
    declared = {
        tuple(identity(r[n]) for n in parents): int(r[roles["population_ssu"]])
        for r in declared_frame
    }
    terminal = {
        tuple(identity(r[n]) for n in leaves): int(r[roles["population_tsu"]])
        for r in terminal_frame
    }
    h_values = (
        frame[roles["strata"]].tolist() if roles.get("strata") else ["unstratified"] * len(frame)
    )
    weights, strata = np.zeros(len(frame), dtype=float), []
    for hrows in ordered_groups(h_values):
        p_groups = ordered_groups(frame[roles["psu"]].iloc[hrows].tolist())
        f1 = len(p_groups) / int(frame[roles["population_psu"]].iloc[hrows[0]])
        psus = []
        for relative_p in p_groups:
            prows, cells = hrows[relative_p], []
            for relative_g in ordered_groups(frame[roles["ssu_strata"]].iloc[prows].tolist()):
                crows = prows[relative_g]
                raw = frame.iloc[crows[0]]
                s_groups = ordered_groups(frame[roles["ssu"]].iloc[crows].tolist())
                f2 = len(s_groups) / declared[tuple(identity(raw[n]) for n in parents)]
                ssus = []
                for relative_s in s_groups:
                    srows, tcells = crows[relative_s], []
                    for relative_q in ordered_groups(
                        frame[roles["tsu_strata"]].iloc[srows].tolist()
                    ):
                        positions = srows[relative_q]
                        raw = frame.iloc[positions[0]]
                        f3 = len(positions) / terminal[tuple(identity(raw[n]) for n in leaves)]
                        weights[positions] = 1 / (f1 * f2 * f3)
                        tcells.append((positions, f3))
                    ssus.append(tcells)
                cells.append((ssus, f2))
            psus.append(cells)
        strata.append((psus, f1))
    return weights, strata


def covariance_components(weighted_rows, strata):
    """SSU totals retain all their terminal cells; V3 centers each cell separately."""
    weighted_rows = np.asarray(weighted_rows, dtype=float)
    parts = [np.zeros((weighted_rows.shape[1],) * 2) for _ in range(3)]
    for psus, f1 in strata:
        psu_totals = []
        for cells in psus:
            total = np.zeros(weighted_rows.shape[1])
            for ssus, f2 in cells:
                ssu_totals = np.array(
                    [
                        sum(
                            (weighted_rows[pos].sum(0) for pos, _ in tcells),
                            np.zeros(weighted_rows.shape[1]),
                        )
                        for tcells in ssus
                    ]
                )
                total += ssu_totals.sum(0)
                if f2 < 1:
                    centered = ssu_totals - ssu_totals.mean(0)
                    parts[1] += (
                        f1 * (1 - f2) * len(ssus) / (len(ssus) - 1) * (centered.T @ centered)
                    )
                for tcells in ssus:
                    for positions, f3 in tcells:
                        if f3 < 1:
                            centered = weighted_rows[positions] - weighted_rows[positions].mean(0)
                            parts[2] += (
                                f1
                                * f2
                                * (1 - f3)
                                * len(positions)
                                / (len(positions) - 1)
                                * (centered.T @ centered)
                            )
            psu_totals.append(total)
        if f1 < 1:
            centered = np.array(psu_totals) - np.mean(psu_totals, axis=0)
            parts[0] += (1 - f1) * len(psus) / (len(psus) - 1) * (centered.T @ centered)
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


def expected_case(
    frame, kind, *, roles=ROLES, declared_frame=None, terminal_frame=None, spec=None, options=None
):
    spec = specification(kind) if spec is None else spec
    options = {**(options or {}), **{key: value for key, value in spec.items() if key != "args"}}
    args = spec["args"]
    members = (
        frame[options["domain"]].to_numpy(dtype=bool)
        if options.get("domain")
        else np.ones(len(frame), bool)
    )
    weights, strata = geometry(frame, roles, declared_frame, terminal_frame)
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


def saved_geometry(frame, roles, declared_frame, terminal_frame):
    """Complete original typed partitions and both supplied frame fingerprints."""

    def canonical(raw, terminal):
        names = ["stratum", "psu", "ssu_stratum"] + (["ssu", "tsu_stratum"] if terminal else [])
        role_names = ["strata", "psu", "ssu_strata"] + (["ssu", "tsu_strata"] if terminal else [])
        population = "population_tsu" if terminal else "population_ssu"
        result = []
        for row in raw:
            record = {
                name: (list(identity(row[roles[role]])) if roles.get(role) else None)
                for name, role in zip(names, role_names)
            }
            record[population] = int(row[roles[population]])
            result.append(record)
        result.sort(
            key=lambda r: json.dumps(
                [r[name] for name in names],
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        lookup = {
            json.dumps([r[name] for name in names], ensure_ascii=False, separators=(",", ":")): i
            for i, r in enumerate(result)
        }
        return result, lookup

    frame2, index2 = canonical(declared_frame, False)
    frame3, index3 = canonical(terminal_frame, True)
    hnames = [roles["strata"]] if roles.get("strata") else []
    levels = [
        hnames,
        hnames + [roles["psu"]],
        hnames + [roles["psu"], roles["ssu_strata"]],
        hnames + [roles["psu"], roles["ssu_strata"], roles["ssu"]],
        hnames + [roles["psu"], roles["ssu_strata"], roles["ssu"], roles["tsu_strata"]],
    ]
    partitions = []
    for names in levels:
        grouped = {}
        for i, row in enumerate(frame.to_dict("records")):
            grouped.setdefault(tuple(identity(row[name]) for name in names), []).append(i)
        partitions.append(list(grouped.values()))
    hs, ps, cs, ss, ts = partitions

    def parent(rows, groups):
        return next(i for i, other in enumerate(groups) if rows[0] in other)

    def children(rows, groups):
        return [i for i, other in enumerate(groups) if other[0] in rows]

    result = {"strata": [], "psus": [], "cells": [], "ssus": [], "tsu_cells": []}
    for positions in hs:
        result["strata"].append(
            dict(
                n_psu=len(children(positions, ps)),
                population_psu=int(frame[roles["population_psu"]].iloc[positions[0]]),
                psu_indices=children(positions, ps),
            )
        )
    for positions in ps:
        result["psus"].append(
            dict(
                stratum_index=parent(positions, hs),
                n_ssu=len(children(positions, ss)),
                cell_indices=children(positions, cs),
                ssu_indices=children(positions, ss),
                row_positions=positions,
            )
        )

    def typed_row(positions, terminal):
        row = frame.iloc[positions[0]]
        values = [list(identity(row[roles["strata"]])) if roles.get("strata") else None]
        values += [
            list(identity(row[roles[role]]))
            for role in ("psu", "ssu_strata") + (("ssu", "tsu_strata") if terminal else ())
        ]
        return json.dumps(values, ensure_ascii=False, separators=(",", ":"))

    for positions in cs:
        result["cells"].append(
            dict(
                psu_index=parent(positions, ps),
                frame_index=index2[typed_row(positions, False)],
                n_ssu=len(children(positions, ss)),
                population_ssu=int(frame[roles["population_ssu"]].iloc[positions[0]]),
                ssu_indices=children(positions, ss),
                row_positions=positions,
            )
        )
    for positions in ss:
        result["ssus"].append(
            dict(
                psu_index=parent(positions, ps),
                cell_index=parent(positions, cs),
                n_tsu=len(positions),
                tsu_cell_indices=children(positions, ts),
                row_positions=positions,
            )
        )
    for positions in ts:
        result["tsu_cells"].append(
            dict(
                ssu_index=parent(positions, ss),
                frame_index=index3[typed_row(positions, True)],
                n_tsu=len(positions),
                population_tsu=int(frame[roles["population_tsu"]].iloc[positions[0]]),
                row_positions=positions,
            )
        )
    checksum = hashlib.sha256()
    role_columns = [
        roles[key]
        for key in (
            "psu",
            "ssu_strata",
            "ssu",
            "tsu_strata",
            "tsu",
            "population_psu",
            "population_ssu",
            "population_tsu",
        )
    ] + hnames
    checksum.update(
        json.dumps(
            [(name, str(frame[name].dtype)) for name in role_columns], ensure_ascii=False
        ).encode()
    )
    for name in role_columns:
        dtype = frame[name].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            checksum.update(
                json.dumps(
                    [name, dtype.ordered, [identity(c) for c in dtype.categories]],
                    ensure_ascii=False,
                ).encode()
            )
    for records in (frame2, frame3):
        data = json.dumps(
            records, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
        checksum.update(hashlib.sha256(data).hexdigest().encode())
    for row in frame.to_dict("records"):
        values = [identity(row[roles["strata"]]) if roles.get("strata") else None]
        values += [
            identity(row[roles[key]]) for key in ("psu", "ssu_strata", "ssu", "tsu_strata", "tsu")
        ]
        values += [
            int(row[roles[key]]) for key in ("population_psu", "population_ssu", "population_tsu")
        ]
        checksum.update(
            json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        )
    return frame2, frame3, result, checksum.hexdigest()


def verify_payload(payload):
    assert set(payload) >= {"states", "poststates", "oracle_inputs"}
    states, tables, inputs = (payload[name] for name in ("states", "poststates", "oracle_inputs"))
    assert set(states) == set(CASES) and set(tables) == set(CASES)
    assert set(inputs["cases"]) == set(CASES)
    frame = pd.DataFrame(inputs["frame"])
    roles, shared = inputs["design_roles"], inputs["options"]
    assert (
        len(frame) == 96
        and roles == ROLES
        and len(inputs["ssu_frame"]) == 12
        and len(inputs["tsu_frame"]) == 48
    )
    assert shared["domain"] == "domain" and shared["missing"] == "drop"
    checks = {}
    for kind in CASES:
        expected = expected_case(
            frame,
            kind,
            roles=roles,
            declared_frame=inputs["ssu_frame"],
            terminal_frame=inputs["tsu_frame"],
            spec=inputs["cases"][kind],
            options=shared,
        )
        state, table = states[kind], tables[kind]
        assert state["method"] == "fully-stratified-three-stage-taylor"
        saved_payload = {key: value for key, value in state.items() if key != "integrity_sha256"}
        canonical = json.dumps(
            saved_payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == state["integrity_sha256"]
        if kind in CASES[:4]:
            assert state["schema_version"] == "survey-fully-stratified-three-stage-result-v1"
            assert state["target"] == kind
        else:
            assert (
                state["schema_version"]
                == "survey-fully-stratified-three-stage-regression-result-v1"
            )
            assert state["family"] == ("linear" if kind == "regress" else kind)
            assert state["outcome"] == inputs["cases"][kind]["args"][0]
            assert state["regressors"] == inputs["cases"][kind]["args"][1]
        assert state["design"]["schema_version"] == "survey-fully-stratified-three-stage-design-v1"
        assert state["design"]["stages"] == 3 and state["design"]["sampling"] == "sequential-srswor"
        assert all(state["design"][name] == value for name, value in roles.items())
        estimates = state["estimates" if kind in CASES[:4] else "coefficients"]
        assert state["labels"] == expected["labels"] and table["index"] == expected["labels"]
        assert state["df"] == expected["df"] and state["alpha"] == expected["alpha"]
        numerical(state["null"], expected["null"], kind + " null")
        estimate_error = numerical(estimates, expected["estimates"], kind + " estimates")
        covariance_error = numerical(
            state["covariance"], expected["covariance"], kind + " full covariance"
        )
        validation, metadata = state["design"]["validation"], state["metadata"]
        canonical_frame, canonical_terminal_frame, expected_geometry, input_hash = saved_geometry(
            frame, roles, inputs["ssu_frame"], inputs["tsu_frame"]
        )
        assert state["design"]["ssu_frame"] == canonical_frame
        assert state["design"]["tsu_frame"] == canonical_terminal_frame
        for key in ("strata", "psus", "cells", "ssus", "tsu_cells"):
            assert validation[key] == expected_geometry[key], kind + " saved " + key
        assert (
            validation["n_cells"] == 12
            and validation["n_ssu"] == 24
            and validation["n_tsu_cells"] == 48
        )
        assert validation["design_input_sha256"] == input_hash
        canonical_frame_bytes = json.dumps(
            canonical_frame,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        assert validation["frame_input_sha256"] == hashlib.sha256(canonical_frame_bytes).hexdigest()
        terminal_bytes = json.dumps(
            canonical_terminal_frame,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        assert validation["tsu_frame_input_sha256"] == hashlib.sha256(terminal_bytes).hexdigest()
        assert metadata["tsu_frame_sha256"] == validation["tsu_frame_input_sha256"]
        assert metadata["n_terminal_strata"] == 48
        assert (
            metadata["lower_stage_stratification"]
            == "stage2 within sampled PSU and stage3 within sampled SSU"
        )
        assert validation["nobs"] == len(frame) and validation["design_df"] == expected["df"]
        numerical(
            validation["sum_weights"], expected["weights"].sum(), kind + " derived weight sum"
        )
        assert metadata["sample_positions"] == np.flatnonzero(expected["selected"]).tolist()
        assert metadata["out_of_domain_positions"] == np.flatnonzero(~expected["members"]).tolist()
        assert (
            metadata["outcome_exclusions"]
            == np.flatnonzero(expected["members"] & ~expected["selected"]).tolist()
        )
        assert metadata["n_used"] == int(expected["selected"].sum())
        assert metadata["precision"] == "float64" and metadata["device"] == "cpu"
        assert metadata["multistage_support"] is True
        for key in ("stage1_covariance", "stage2_covariance", "stage3_covariance"):
            numerical(metadata[key], expected[key], kind + " " + key)
        if kind in CASES[:4]:
            for key, values in expected["primitives"].items():
                numerical(metadata[key], values, kind + " " + key)
        else:
            for key in ("primitive_X", "primitive_y"):
                numerical(metadata[key], expected["primitives"][key], kind + " " + key)
            numerical(metadata["bread"], expected["primitives"]["bread"], kind + " observed bread")
            numerical(
                metadata["row_scores"], expected["primitives"]["row_scores"], kind + " row scores"
            )
        assert table["columns"] == TABLE_COLUMNS
        reference_table = expected_table(expected)
        assert len(table["data"]) == len(reference_table)
        display_error = 0.0
        for i, (actual_row, expected_row) in enumerate(zip(table["data"], reference_table)):
            assert len(actual_row) == len(TABLE_COLUMNS)
            for j, (actual_value, expected_value) in enumerate(zip(actual_row, expected_row)):
                name = f"{kind} physical row {i} {TABLE_COLUMNS[j]}"
                if expected_value is None:
                    assert actual_value is None, name + ": undefined statistic must remain null."
                else:
                    display_error = max(
                        display_error, numerical(actual_value, expected_value, name)
                    )
        attrs = table["attrs"]
        assert attrs["survey_state"] == state and attrs["df"] == expected["df"]
        numerical(
            attrs["covariance_matrix"], expected["covariance"], kind + " displayed covariance"
        )
        assert attrs["alpha"] == expected["alpha"]
        checks[kind] = {
            "labels": expected["labels"],
            "n_used": int(expected["selected"].sum()),
            "df": expected["df"],
            "estimate_max_abs_error": estimate_error,
            "covariance_max_abs_error": covariance_error,
            "display_max_abs_error": display_error,
            "all_three_stage_components_verified": True,
            "full_joint_covariance_verified": True,
            "physical_display_rows_verified": True,
        }
    return {
        "status": "passed",
        "independent_numpy_scipy": True,
        "production_numerical_imports": False,
        "case_count": len(checks),
        "display_table_count": len(checks),
        "physical_rows": len(frame),
        "sampling": "three sequential SRSWOR stages with independently stratified SSUs and terminal TSUs",
        "variance": "recursive three-stage Taylor with cell-specific centering and prefixes",
        "stage2_stratification": True,
        "stage3_stratification": True,
        "explicit_positive_cell_frame_verified": True,
        "df": "complete first-stage PSUs minus strata, reference convention only",
        "exact_nonlinear_variance_or_coverage_claim": False,
        "checksums_authenticate_source_data": False,
        "cases": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    raw = args.payload.read_bytes()
    receipt = verify_payload(json.loads(raw))
    receipt["payload_sha256"] = hashlib.sha256(raw).hexdigest()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "cases": receipt["case_count"],
                "tables": receipt["display_table_count"],
                "receipt": str(args.receipt),
            }
        )
    )


if __name__ == "__main__":
    main()
