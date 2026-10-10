"""Full survey DEFF acceptance artifacts for synthetic resident CPU fixtures.

The unequal-weight reference is our explicitly declared full-row weighted
population law with fixed eligibility. Domain totals are not a comparison to
Stata's default domain DEFF. No licensed vendor or platform parity is claimed.
Run directly, from an isolated wheel, or through the native application's Run.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import openecon as oe

EXPECTED = json.loads(r'''{"unequal_full_mean":{"estimates":[6.310344827586207,3.4827586206896552],"design_covariance":[[0.9445637589586034,0.8039576914974388],[0.8039576914974388,0.8416824430459746]],"srs_reference_covariance":[[1.3606251061661288,0.40716833701375915],[0.40716833701375915,0.4199082724647528]],"design_effect":[0.6942130897614605,2.0044435850370763]},"unequal_full_total":{"estimates":[183,101],"design_covariance":[[1769,1294],[1294,1093]],"srs_reference_covariance":[[1144.2857142857142,342.42857142857144],[342.42857142857144,353.14285714285717]],"design_effect":[1.5459425717852684,3.095064724919094]},"unequal_full_ratio":{"estimates":[1.811881188118812,0.5519125683060109],"design_covariance":[[0.06549121408824861,-0.01994911388558106],[-0.01994911388558106,0.006076649369847185]],"srs_reference_covariance":[[0.1041803129593662,-0.03173410291434484],[-0.03173410291434484,0.009666445215719542]],"design_effect":[0.6286333015124689,0.6286333015124689]},"unequal_full_proportion":{"estimates":[0.3448275862068966,0.6551724137931034,0],"design_covariance":[[0.029490400562152807,-0.029490400562152807,0],[-0.029490400562152807,0.029490400562152807,0],[0,0,0]],"srs_reference_covariance":[[0.03227450314251741,-0.03227450314251741,0],[-0.03227450314251741,0.03227450314251741,0],[0,0,0]],"design_effect":[0.9137367795231242,0.9137367795231242,null]},"unequal_fixed_mean":{"estimates":[3.6666666666666665,2.888888888888889],"design_covariance":[[0.19753086419753085,-0.3292181069958848],[-0.3292181069958848,0.5486968449931413]],"srs_reference_covariance":[[0.7160493827160493,-0.9888300999412111],[-0.9888300999412111,1.3752694493435234]],"design_effect":[0.27586206896551724,0.39897406668566543]},"unequal_fixed_total":{"estimates":[33,26],"design_covariance":[[225,30],[30,4]],"srs_reference_covariance":[[403.7142857142857,192.28571428571428],[192.28571428571428,326]],"design_effect":[0.5573248407643312,0.012269938650306749]},"unequal_fixed_ratio":{"estimates":[1.2692307692307692,0.7878787878787878],"design_covariance":[[0.22971884737929343,-0.14259866008117755],[-0.14259866008117755,0.08851854381531316]],"srs_reference_covariance":[[0.6520311163574705,-0.404750261393618],[-0.404750261393618,0.2512499326924571]],"design_effect":[0.3523127065815553,0.3523127065815553]},"unequal_fixed_proportion":{"estimates":[0.5555555555555556,0.4444444444444444,0],"design_covariance":[[0.0877914951989026,-0.0877914951989026,0],[-0.0877914951989026,0.0877914951989026,0],[0,0,0]],"srs_reference_covariance":[[0.11365863217715069,-0.11365863217715069,0],[-0.11365863217715069,0.11365863217715069,0],[0,0,0]],"design_effect":[0.7724137931034483,0.7724137931034483,null]},"equal_legacy_mean":{"estimates":[5.375,3.375],"design_covariance":[[0.390625,0.3125],[0.3125,0.390625]],"srs_reference_covariance":[[1.390625,0.24776785714285715],[0.24776785714285715,0.3549107142857143]],"design_effect":[0.2808988764044944,1.10062893081761]},"equal_legacy_total":{"estimates":[86,54],"design_covariance":[[100,80],[80,100]],"srs_reference_covariance":[[356,63.42857142857143],[63.42857142857143,90.85714285714286]],"design_effect":[0.2808988764044944,1.10062893081761]},"equal_legacy_ratio":{"estimates":[1.5925925925925926,0.627906976744186],"design_covariance":[[0.033888992381092164,-0.01336131716918128],[-0.01336131716918128,0.005267928727059575]],"srs_reference_covariance":[[0.1318291749628436,-0.05197591592639967],[-0.05197591592639967,0.020492397355513985]],"design_effect":[0.2570674692505954,0.2570674692505954]},"equal_legacy_proportion":{"estimates":[0.5,0.5,0],"design_covariance":[[0.03125,-0.03125,0],[-0.03125,0.03125,0],[0,0,0]],"srs_reference_covariance":[[0.03571428571428571,-0.03571428571428571,0],[-0.03571428571428571,0.03571428571428571,0],[0,0,0]],"design_effect":[0.875,0.875,null]}}''')
RESULT_NAMES = tuple(EXPECTED)
results = {}
artifacts = {}


def matrix_close(actual, expected):
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            matrix_close(a, b)
    elif expected is None:
        assert actual is None
    else:
        assert math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-13), (actual, expected)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return plain(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(f"Unsupported artifact value: {type(value).__name__}")


def frame_record(table):
    return {
        "index": plain(table.index.tolist()),
        "columns": plain(table.columns.tolist()),
        "dtypes": [str(dtype) for dtype in table.dtypes],
        "data": plain([list(row) for row in table.itertuples(index=False, name=None)]),
        "attrs": plain(table.attrs),
    }


for variant in ("unequal_full", "unequal_fixed", "equal_legacy"):
    frame = pd.DataFrame({
        "w": [2.] * 8 if variant == "equal_legacy" else [2., 3., 2., 4., 5., 3., 4., 6.],
        "p": [1, 1, 2, 2, 1, 1, 2, 2],
        "h": ["a"] * 4 + ["b"] * 4,
        "y": [1., 3., 2., 5., 6., 8., 7., 11.],
        "x": [2., 4., 5., 1., 2., 3., 4., 6.],
        "domain": [1] * 4 + [0] * 4,
        "category": ["a", "a", "a", "b", "b", "a", "b", "b"],
    })
    if variant == "unequal_fixed":
        frame.loc[0, ["y", "x"]] = float("nan")
        frame.loc[0, "category"] = None
        # Out-of-domain unavailable outcomes must not remove any design PSU.
        frame.loc[4:, ["y", "x"]] = float("nan")
        frame.loc[4:, "category"] = None
    design = oe.survey_design(frame, weights="w", psu="p", strata="h")
    options = {"deff": True, "alpha": 0.1}
    if variant == "unequal_fixed":
        options.update(domain="domain", missing="drop")
    for kind in ("mean", "total", "ratio", "proportion"):
        name = f"{variant}_{kind}"
        if kind == "ratio":
            fitted = oe.survey_ratio(frame, design, ["y", "x"], ["x", "y"], **options)
        elif kind == "proportion":
            fitted = oe.survey_proportion(frame, design, "category", categories=["a", "b", "absent"], **options)
        else:
            fitted = getattr(oe, "survey_" + kind)(frame, design, ["y", "x"], **options)
        expected = EXPECTED[name]
        matrix_close(fitted.estimates, expected["estimates"])
        matrix_close(fitted.covariance, expected["design_covariance"])
        matrix_close(fitted.metadata["srs_covariance"], expected["srs_reference_covariance"])
        matrix_close(fitted.metadata["design_effect"], expected["design_effect"])
        assert fitted.metadata["n_design"] == 8 and fitted.df == 2
        if variant == "unequal_fixed":
            assert fitted.metadata["n_used"] == 3
            assert fitted.metadata["sample_positions"] == [1, 2, 3]
            assert fitted.metadata["outcome_exclusions"] == [0]
            assert fitted.metadata["psu_influence_sums"][2:] == [[0.] * len(fitted.labels)] * 2
        if variant == "equal_legacy":
            assert "srs_reference_state" not in fitted.metadata
            assert fitted.metadata["srs_reference"].startswith("equal-weight independent row PSUs")
        else:
            assert fitted.metadata["srs_reference_state"]["n_design"] == 8
            assert fitted.metadata["srs_reference_state"]["eligibility"] == "fixed-domain-and-joint-complete-case-indicator"
        assert fitted.metadata["stata_parity_validated"] is False
        state = fitted.model_dump(mode="json")
        restored = oe.SurveyResult.model_validate_json(json.dumps(state, allow_nan=False))
        assert restored.model_dump(mode="json") == state
        coefficients = [1.] + [-1.] * (len(fitted.labels) - 1)
        table, contrast = restored.to_frame(), restored.contrast(coefficients)
        pd.testing.assert_frame_equal(table, fitted.to_frame(), check_exact=True)
        pd.testing.assert_frame_equal(contrast, fitted.contrast(coefficients), check_exact=True)
        latex, contrast_latex = oe.to_latex(table), oe.to_latex(contrast)
        assert latex and contrast_latex
        results[name] = restored
        artifacts[name] = {
            "name": name, "state": state, "table": frame_record(table),
            "covariance": plain(fitted.covariance),
            "srs_reference_covariance": plain(fitted.metadata["srs_covariance"]),
            "reference_state": plain(fitted.metadata.get("srs_reference_state")),
            "design_effect": plain(fitted.metadata["design_effect"]),
            "contrast_coefficients": coefficients, "contrast": frame_record(contrast),
            "table_latex": latex, "contrast_latex": contrast_latex,
        }
        if "display" in globals():
            globals()["display"](table)

assert tuple(results) == RESULT_NAMES
states = {name: item.model_dump(mode="json") for name, item in results.items()}
SURVEY_DEFF_ARTIFACTS = {
    "schema": "survey-deff-complete-artifacts-v1", "status": "passed",
    "result_names": list(RESULT_NAMES), "output_count": len(results),
    "table_count": len(results) * 2, "artifacts": artifacts,
    "stata_parity_validated": False,
}
_bytes = json.dumps(SURVEY_DEFF_ARTIFACTS, sort_keys=True, allow_nan=False, separators=(",", ":"), ensure_ascii=False).encode()
_directory = os.environ.get("OPENECON_ACCEPTANCE_DIR")
if _directory:
    _output = Path(_directory)
    _output.mkdir(parents=True, exist_ok=True)
    (_output / "survey-deff-artifacts.json").write_bytes(_bytes)
    for _name, _artifact in artifacts.items():
        (_output / (_name + ".json")).write_text(json.dumps(_artifact, sort_keys=True, allow_nan=False, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
print("SURVEY_DEFF_RECEIPT:" + json.dumps({
    "schema": SURVEY_DEFF_ARTIFACTS["schema"], "status": "passed",
    "result_names": list(RESULT_NAMES), "output_count": len(results),
    "table_count": len(results) * 2,
    "artifact_filename": "survey-deff-artifacts.json",
    "artifact_sha256": hashlib.sha256(_bytes).hexdigest(),
    "artifact_bytes": len(_bytes), "full_covariance_saved": True,
    "restored_equal": True, "stata_parity_validated": False,
}, sort_keys=True))
