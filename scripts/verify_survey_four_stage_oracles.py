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
ROLES = {"psu": "p", "ssu": "s", "tsu": "j", "fsu": "f", "strata": "h",
         "population_psu": "N", "population_ssu": "M", "population_tsu": "L", "population_fsu": "K"}


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


def geometry(frame, roles=ROLES):
    """Group original physical rows independently at four typed parent levels."""
    weights = np.zeros(len(frame), dtype=float)
    hvalues = frame[roles["strata"]].tolist() if roles.get("strata") else ["all"]*len(frame)
    strata = []
    for hrows in ordered_groups(hvalues):
        psus = []
        pgroups = ordered_groups(frame[roles["psu"]].iloc[hrows].tolist())
        f1 = len(pgroups)/int(frame[roles["population_psu"]].iloc[hrows[0]])
        for rp in pgroups:
            prows = hrows[rp]
            ssus = []
            sgroups = ordered_groups(frame[roles["ssu"]].iloc[prows].tolist())
            f2 = len(sgroups)/int(frame[roles["population_ssu"]].iloc[prows[0]])
            for rs in sgroups:
                srows = prows[rs]
                tsus = []
                tgroups = ordered_groups(frame[roles["tsu"]].iloc[srows].tolist())
                f3 = len(tgroups)/int(frame[roles["population_tsu"]].iloc[srows[0]])
                for rt in tgroups:
                    positions = srows[rt]
                    f4 = len(positions)/int(frame[roles["population_fsu"]].iloc[positions[0]])
                    weights[positions] = 1/(f1*f2*f3*f4)
                    tsus.append((positions, f4))
                ssus.append((tsus, f3))
            psus.append((ssus, f2))
        strata.append((psus, f1))
    return weights, strata


def canonical_sha(value):
    """Hash the documented UTF-8 JSON codec without importing production helpers."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode()).hexdigest()


def retained_geometry(frame, roles=ROLES):
    """Rebuild every saved parent/row partition and the original typed input digest.

    Each level is indexed by first occurrence in the full physical row order,
    independently of the nested numerical grouping used by ``geometry``.
    The native fixture supplies its original role columns as JSON, whose dtype
    reconstruction is part of this bounded synthetic acceptance contract.
    """
    columns = [roles[name] for name in (
        "psu", "ssu", "tsu", "fsu", "population_psu", "population_ssu",
        "population_tsu", "population_fsu",
    )] + ([roles["strata"]] if roles.get("strata") else [])
    checksum = hashlib.sha256()
    checksum.update(json.dumps(
        [(name, str(frame[name].dtype)) for name in columns], ensure_ascii=False,
    ).encode())
    for name in columns:
        dtype = frame[name].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            checksum.update(json.dumps(
                [name, dtype.ordered, [identity(c) for c in dtype.categories]],
                ensure_ascii=False,
            ).encode())
    level_keys = [dict() for _ in range(4)]
    populations = [dict() for _ in range(4)]
    terminal_ids = set()
    for position, row in enumerate(frame[columns].itertuples(index=False, name=None)):
        raw = dict(zip(columns, row))
        ids = [identity(raw[roles[name]]) for name in ("psu", "ssu", "tsu", "fsu")]
        h = identity(raw[roles["strata"]]) if roles.get("strata") else ("unstratified",)
        counts = []
        for name in ("population_psu", "population_ssu", "population_tsu", "population_fsu"):
            count = raw[roles[name]]
            assert not isinstance(count, (bool, np.bool_)) and 0 < count <= 2**53 and int(count) == count
            counts.append(int(count))
        assert (h, *ids) not in terminal_ids
        terminal_ids.add((h, *ids))
        for level in range(4):
            key = (h, *ids[:level])
            level_keys[level].setdefault(key, []).append(position)
            populations[level].setdefault(key, counts[level])
            assert populations[level][key] == counts[level]
        checksum.update(json.dumps(
            [h, *ids, *counts], ensure_ascii=False, separators=(",", ":"),
        ).encode())
        checksum.update(b"\n")
    ordered = [list(keys) for keys in level_keys]
    indices = [{key: i for i, key in enumerate(keys)} for keys in ordered]
    records = [[], [], [], []]
    for level, keys in enumerate(ordered):
        for key in keys:
            rows = level_keys[level][key]
            if level < 3:
                child_indices = [indices[level + 1][child] for child in ordered[level + 1] if child[:-1] == key]
                count = len(child_indices)
            else:
                child_indices, count = None, len(rows)
            population = populations[level][key]
            assert count <= population and (count > 1 or population == 1)
            if level == 0:
                record = dict(n_psu=count, population_psu=population, psu_indices=child_indices)
            elif level == 1:
                record = dict(stratum_index=indices[0][key[:-1]], n_ssu=count,
                    population_ssu=population, ssu_indices=child_indices, row_positions=rows)
            elif level == 2:
                record = dict(psu_index=indices[1][key[:-1]], n_tsu=count,
                    population_tsu=population, tsu_indices=child_indices, row_positions=rows)
            else:
                record = dict(ssu_index=indices[2][key[:-1]], n_fsu=count,
                    population_fsu=population, row_positions=rows)
            records[level].append(record)
    return dict(nobs=len(frame), n_strata=len(records[0]), n_psu=len(records[1]),
        n_ssu=len(records[2]), n_tsu=len(records[3]),
        design_df=len(records[1])-len(records[0]), design_input_sha256=checksum.hexdigest(),
        **dict(zip(("strata", "psus", "ssus", "tsus"), records)))


def original_sample_sha(frame, kind, expected, design_sha, spec):
    """Bind selected raw responses/regressors or target primitives to original positions."""
    if kind in DESCRIPTIVE:
        values = expected["primitives"]["primitive_values"]
        denominators = expected["primitives"]["primitive_denominators"]
    else:
        outcome, regressors = spec["args"]
        values = np.zeros((len(frame), 1 + len(regressors)), dtype=float)
        values[expected["selected"]] = frame[[outcome, *regressors]].to_numpy(dtype=float)[expected["selected"]]
        denominators = np.ones_like(values)
    return canonical_sha([design_sha, expected["selected"].tolist(), values.tolist(), denominators.tolist()])


def covariance_components(weighted_rows, strata):
    """Independent nested totals, centering and four conditional sampling prefixes."""
    weighted_rows = np.asarray(weighted_rows, dtype=float)
    parts = [np.zeros((weighted_rows.shape[1],)*2) for _ in range(4)]
    def add(stage, values, prefix, fraction):
        if fraction < 1:
            values = np.array(values)
            centered = values-values.mean(0)
            parts[stage] += prefix*(1-fraction)*len(values)/(len(values)-1)*(centered.T@centered)
    for psus, f1 in strata:
        ptotals = []
        for ssus, f2 in psus:
            stotals = []
            for tsus, f3 in ssus:
                ttotals = []
                for positions, f4 in tsus:
                    ttotals.append(weighted_rows[positions].sum(0))
                    add(3, weighted_rows[positions], f1*f2*f3, f4)
                stotals.append(np.array(ttotals).sum(0))
                add(2, ttotals, f1*f2, f3)
            ptotals.append(np.array(stotals).sum(0))
            add(1, stotals, f1, f2)
        add(0, ptotals, 1, f1)
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
    frame, kind, *, roles=ROLES, spec=None, options=None
):
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
    assert set(payload) >= {"states", "poststates", "oracle_inputs"}
    states, tables, inputs = (payload[name] for name in ("states", "poststates", "oracle_inputs"))
    assert set(states) == set(CASES) and set(tables) == set(CASES)
    assert set(inputs["cases"]) == set(CASES)
    frame = pd.DataFrame(inputs["frame"])
    roles, shared = inputs["design_roles"], inputs["options"]
    assert len(frame) == 96 and roles == ROLES
    assert shared["domain"] == "domain" and shared["missing"] == "drop"
    original_geometry = retained_geometry(frame, roles)
    checks = {}
    for kind in CASES:
        expected = expected_case(
            frame,
            kind,
            roles=roles,
            spec=inputs["cases"][kind],
            options=shared,
        )
        state, table = states[kind], tables[kind]
        assert state["method"] == "four-stage-taylor"
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
            assert state["schema_version"] == "survey-four-stage-result-v1"
            assert state["target"] == kind
        else:
            assert (
                state["schema_version"]
                == "survey-four-stage-regression-result-v1"
            )
            assert state["family"] == ("linear" if kind == "regress" else kind)
            assert state["outcome"] == inputs["cases"][kind]["args"][0]
            assert state["regressors"] == inputs["cases"][kind]["args"][1]
        assert state["design"]["schema_version"] == "survey-four-stage-design-v1"
        assert state["design"]["stages"] == 4 and state["design"]["sampling"] == "sequential-srswor"
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
        for name, original in original_geometry.items():
            assert validation[name] == original, kind + " original ordered geometry " + name
        assert metadata["sample_input_sha256"] == original_sample_sha(
            frame, kind, expected, original_geometry["design_input_sha256"], inputs["cases"][kind]
        ), kind + " original numeric sample fingerprint"
        assert validation["n_psu"] == 6 and validation["n_ssu"] == 12 and validation["n_tsu"] == 24
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
        for key in ("stage1_covariance", "stage2_covariance", "stage3_covariance", "stage4_covariance"):
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
            "all_four_stage_components_verified": True,
            "full_joint_covariance_verified": True,
            "physical_display_rows_verified": True,
            "complete_ordered_geometry_verified": True,
            "original_design_input_sha256_verified": True,
            "original_numeric_sample_sha256_verified": True,
        }
    return {
        "status": "passed",
        "independent_numpy_scipy": True,
        "production_numerical_imports": False,
        "case_count": len(checks),
        "display_table_count": len(checks),
        "physical_rows": len(frame),
        "sampling": "four sequential SRSWOR selections with first-stage strata and unstratified lower stages",
        "variance": "four recursive Taylor contributions with prefixes 1, f1, f1*f2, f1*f2*f3",
        "lower_stage_stratification": False,
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
