"""Replay complete single-stage survey and bounded missing-data results.

``run(destination)`` writes complete JSON and LaTeX, then restores every state.
The synthetic finite imputation schedules do not establish chain convergence.
"""

from pathlib import Path
import hashlib
import json
import math


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _table(value):
    return {"index": value.index.tolist(), "columns": value.columns.tolist(),
            "values": [list(row) for row in value.itertuples(index=False, name=None)],
            "attrs": value.attrs}


def run(destination):
    import openecon as oe
    import pandas as pd

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    survey = pd.DataFrame({
        "w": [1., 2., 3., 1., 2., 4., 4., 2., 1., 3., 2., 1.],
        "psu": [i for i in range(6) for _ in range(2)],
        "h": [0] * 6 + [1] * 6, "N": [12] * 6 + [6] * 6,
        "x": [0., 1.] * 6,
        "y": [2., 5., 3., 2., 1., 4., 7., 6., 3., 1., 4., 2.],
        "binary": [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        "count": [0, 2, 1, 0, 3, 1, 2, 4, 0, 1, 2, 0],
    }, index=pd.Index([f"unit-{i // 2}" for i in range(12)], name="unit"))
    design = oe.survey_design(survey, weights="w", psu="psu", strata="h", fpc="N")
    fits = {
        "linear": oe.survey_regress(survey, design, "y", ["x"]),
        "logit": oe.survey_logit(survey, design, "binary", ["x"]),
        "probit": oe.survey_probit(survey, design, "binary", ["x"]),
        "poisson": oe.survey_poisson(survey, design, "count", ["x"]),
    }
    survey_states = {name: value.model_dump(mode="json") for name, value in fits.items()}
    restored_survey = {
        name: oe.SurveyRegressionResult.model_validate_json(_json(state))
        for name, state in survey_states.items()
    }
    evaluation = pd.DataFrame({"x": [-.5, 0., 1.5]}, index=["low", "middle", "high"])
    saved = {
        "predict": oe.survey_predict(restored_survey["probit"], evaluation),
        "margins": oe.survey_margins(restored_survey["logit"], survey,
                                     variables=["x"], at={"x": .25}, weights="w"),
        "lincom": oe.survey_lincom(restored_survey["linear"], [1., 2.], null=3.),
        "test": oe.survey_test(restored_survey["poisson"], [[1., 0.], [0., 1.]]),
    }
    survey_state = {"design": design.model_dump(mode="json"), "fits": survey_states,
                    "evaluation": _table(evaluation), "source": _table(survey),
                    "postestimation": {name: _table(value) for name, value in saved.items()}}

    base = pd.DataFrame({
        "x": [math.sin(i * .63) + i / 80 for i in range(72)],
        "y": [1.4 + .5 * (math.sin(i * .63) + i / 80) + math.cos(i * 1.47)
              for i in range(72)],
    }, index=pd.Index([f"person-{i // 2}" for i in range(72)], name="person"))
    arbitrary = base.copy()
    arbitrary.iloc[[4, 7, 12, 23, 35, 45], 1] = float("nan")
    arbitrary.iloc[25:30, 0] = float("nan")
    monotone = base.copy()
    monotone.iloc[-16:, 1] = float("nan")
    m = 6
    mono = oe.mi_monotone(monotone, ["x", "y"], m=m, seed=30255)
    binary = pd.DataFrame({"x": base.x,
                           "z": [float((i * 7) % 11 >= 5) for i in range(72)]},
                          index=base.index)
    binary.iloc[8:19, 1] = float("nan")
    mi = {
        "em": oe.mvnorm_em(arbitrary, ["x", "y"]),
        "mcar": oe.little_mcar(arbitrary, ["x", "y"]),
        "mvn": oe.mi_mvn(arbitrary, ["x", "y"], m=m, seed=30254, burn=30, thin=5,
                          prior={"mean": [0., 1.], "kappa": .1, "df": 4.,
                                 "scale": [[1., 0.], [0., 1.]]}),
        "monotone": mono,
        "normal": oe.mi_chained(arbitrary, ["x", "y"],
                                 methods={"x": "normal", "y": "normal"},
                                 m=m, seed=30256, burn=3, iterations=2),
        "pmm": oe.mi_chained(arbitrary, ["x", "y"],
                              methods={"x": "pmm", "y": "pmm"},
                              m=m, seed=30257, burn=3, iterations=2, donors=3),
        "logit": oe.mi_chained(binary, ["x", "z"], methods={"z": "logit"},
                                m=m, seed=30258, burn=2, iterations=1,
                                logit_burn=50, logit_steps=80),
    }
    analysis = [oe.ols(data=mono.dataset(i), y="y", x=["x"], device="cpu")
                for i in range(1, m + 1)]
    for i, fit in enumerate(analysis, 1):
        path = destination / f"mi-ols-imputation-{i:02d}.json"
        path.write_text(fit.model_dump_json(), encoding="utf-8")
        restored = oe.ResultBundle.model_validate_json(path.read_text())
        assert restored.model_dump(mode="json") == fit.model_dump(mode="json")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    pool = oe.mi_pool(analysis, imputation_description=
                      "Monotone Gaussian posterior MI; same 72 rows and fixed x, iid OLS")
    pool_path = destination / "mi-pool-result.json"
    pool_path.write_text(pool.model_dump_json(), encoding="utf-8")
    restored_pool = oe.MIPoolResult.model_validate_json(pool_path.read_text())
    assert restored_pool.model_dump(mode="json") == pool.model_dump(mode="json")
    hashes[pool_path.name] = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    pool_tables_path = destination / "mi-pool-complete-tables.json"
    pool_tables = {name: _table(value) for name, value in restored_pool.tables.items()}
    pool_tables_path.write_text(_json(pool_tables), encoding="utf-8")
    hashes[pool_tables_path.name] = hashlib.sha256(pool_tables_path.read_bytes()).hexdigest()
    mi["pool"] = pool
    mi["d1"] = oe.mi_test(pool, restrictions=[[0., 1.]], values=[0.])
    mi_states = {name: value.model_dump(mode="json") for name, value in mi.items()}
    constructors = {"em": oe.MIDiagnosticResult, "mcar": oe.MIDiagnosticResult,
                    "pool": oe.MIPoolResult, "d1": oe.MIJointResult}
    restored_mi = {
        name: constructors.get(name, oe.MIResult).model_validate_json(_json(state))
        for name, state in mi_states.items()
    }
    for name, value in restored_mi.items():
        assert value.model_dump(mode="json") == mi_states[name]
        if isinstance(value, oe.MIResult):
            for i in range(1, m + 1):
                pd.testing.assert_frame_equal(value.dataset(i), mi[name].dataset(i))
                assert value.dataset(i).index.equals(base.index)
    for name, value in restored_survey.items():
        assert value.model_dump(mode="json") == survey_states[name]
        pd.testing.assert_frame_equal(value.to_frame(), fits[name].to_frame())

    output_states = {"survey": survey_state, "missing_data": mi_states}
    for name, state in output_states.items():
        path = destination / f"{name}-full-state.json"
        path.write_text(_json(state), encoding="utf-8")
        assert json.loads(path.read_text()) == state
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    rendered = {**{f"survey-{name}": value.to_frame() for name, value in restored_survey.items()},
                **{f"survey-{name}": value for name, value in saved.items()}}
    for name, value in restored_mi.items():
        if name != "pool":
            rendered[f"mi-{name}"] = value
    for name, value in rendered.items():
        path = destination / f"{name}.tex"
        path.write_text(value.to_latex(), encoding="utf-8")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {"stages": len(rendered), "survey_stages": 8, "mi_stages": 8,
               "survey_design_df": design.validation.design_df, "mi_rows": len(base),
               "imputations": m, "complete_state_replayed": True,
               "conditional_margins": True, "finite_chain_convergence_assessed": False,
               "runtime": "resident CPU float64", "stata_parity_validated": False,
               "files": hashes}
    (destination / "receipt.json").write_text(_json(receipt), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(_json(run(parser.parse_args().destination)))
