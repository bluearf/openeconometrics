"""Complete meta-extension results for source/frozen backend acceptance.

Run with META_FIXTURE_DIRECTORY and META_RESULT_DIRECTORY set by the caller.
Fixtures and their independent-reference provenance are already committed in
tests/fixtures/meta_extensions. Console output is only a preview; files retain
every result cell and its presentation contract.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.meta import multilevel as ml
from openecon.econometrics.meta import multilevel_kernels as mk


def pack(result):
    return {"summary": oe.summary_state(result), "latex": result.to_latex(),
            "table_order": list(result),
            "table_dtypes": {name: [str(dtype) for dtype in frame.dtypes]
                             for name, frame in result.items()}}


def unpack(payload):
    result = oe.restore_summary(payload["summary"])
    for name, types in payload["table_dtypes"].items():
        result[name] = result[name].astype(dict(zip(result[name].columns, types, strict=True)))
    result = TableSet({name: result[name] for name in payload["table_order"]},
                      title=result.title, **result.attrs)
    assert pack(result) == payload
    return result


def prediction(fit, data):
    return TableSet({"prediction": oe.meta_multilevel_predict(fit, data=data)},
                    title="Saved three-level new-study prediction")


def reject(function, expected):
    try:
        function()
    except AnalysisError as exc:
        assert exc.code == expected, (exc.code, expected)
        return exc.code
    raise AssertionError("Unsupported or invalid input was accepted: " + expected)


def load_inputs(directory):
    def csv(name):
        return pd.read_csv(directory / (name + ".csv"))

    konst = csv("konstantopoulos2011")
    konst["yc"] = konst.year - 1985
    konst["z"] = csv("konst_z").z
    covariance = csv("che_covariance")
    covariance.index = covariance.columns
    dependent_covariance = csv("dependent_covariance")
    dependent_covariance.index = dependent_covariance.columns
    return konst, csv("che"), covariance, csv("dependent"), dependent_covariance, csv("hackshaw1998"), csv("asymmetric")


def run(directory):
    torch.set_num_threads(1)
    konst, che, covariance, dep, dep_cov, hack, asym = load_inputs(directory)
    results = {}
    results["ml"] = oe.meta_multilevel(data=konst, study="district", yi="yi", vi="vi", method="ML")
    results["reml"] = oe.meta_multilevel(data=che, study="study", effect="effect", yi="yi",
                                         covariance=covariance, moderators=["x1", "x2"], method="REML", inference="t")
    results["published_cr2_fit"] = oe.meta_multilevel(data=konst, study="district", yi="yi", vi="vi",
                                                      moderators=["yc", "z"], profile=False)
    results["prediction_ml"] = prediction(unpack(pack(results["ml"])), None)
    results["prediction_reml"] = prediction(unpack(pack(results["reml"])),
                                             pd.DataFrame({"x1": [0., 1.], "x2": [0., -.5]}))
    results["cr2"] = oe.meta_cr2(results["published_cr2_fit"], contrasts=pd.DataFrame(
        {"Intercept": [0., 1.], "yc": [1., 0.], "z": [1., 0.]}, index=["a", "b"]),
        null=pd.Series([.01, .1], index=["a", "b"]))
    for model, method in (("study", "ML"), ("study", "REML"), ("effect", "ML"),
                          ("effect", "REML"), ("common", "common")):
        key = "dependent_" + model + ("" if model == "common" else "_" + method)
        results[key] = oe.meta_dependent(data=dep, covariance=dep_cov, study="study", effect="effect",
                                         yi="yi", moderators=["x"], model=model,
                                         method="REML" if method == "common" else method)
        results[key + "_cr2"] = oe.meta_cr2(unpack(pack(results[key])))
        results[key + "_cr1"] = oe.meta_dependent_robust(results[key], correction="CR1")
        results[key + "_cr1_cr2"] = oe.meta_cr2(unpack(pack(results[key + "_cr1"])))
        pd.testing.assert_frame_equal(results[key + "_cr2"]["covariance"],
                                      results[key + "_cr1_cr2"]["covariance"])
    for label, data, options in (("hack_L0", hack, {}), ("hack_R0", hack, {"estimator": "R0"}),
                                 ("hack_right", hack, {"side": "right"}),
                                 ("asym_L0", asym, {}), ("asym_R0", asym, {"estimator": "R0"})):
        results["trimfill_" + label] = oe.meta_trimfill(data=data, study="study", yi="yi", vi="vi", **options)
    results["pet_hack"] = oe.meta_pet_peese(data=hack, yi="yi", vi="vi", study="study")
    results["pet_asym"] = oe.meta_pet_peese(data=asym, yi="yi", vi="vi", study="study")
    for alternative in ("greater", "less"):
        results["pet_" + alternative] = oe.meta_pet_peese(data=hack, yi="yi", vi="vi",
                                                         alternative=alternative, alpha=.2)
    for label, options in (("estimated", {"steps": [.025, .10, .50, 1.]}),
                            ("fixed", {"steps": [.025, .10, .50, 1.], "weights": [1, .8, .6, .4]}),
                            ("zero", {"steps": [.025, .5], "heterogeneity": False})):
        results["selection_" + label] = oe.meta_selection(data=hack, yi="yi", vi="vi", **options)
    results["selection_moderator"] = oe.meta_selection(data=pd.read_csv(directory / "selection_moderator.csv"),
                                                       yi="yi", vi="vi", study="study", moderators=["x"],
                                                       steps=[.05, .5, 1.])
    for name, result in results.items():
        assert pack(unpack(pack(result))) == pack(result), name

    corrupted = copy.deepcopy(results["reml"].attrs["prediction_state"])
    corrupted["sigma2"][0] *= 1.01
    failures = {
        "resealed_incoherent_state": reject(lambda: ml.load_state(ml._seal(corrupted)), "invalid_result_state"),
        "missing_response": reject(lambda: oe.meta_multilevel(data=che.assign(yi=float("nan")), study="study",
                                                                yi="yi", vi="vi"), "missing_values"),
        "weights": reject(lambda: oe.meta_multilevel(data=che, study="study", yi="yi", vi="vi", weights="vi"), "unsupported_weights"),
        "device": reject(lambda: oe.meta_multilevel(data=che, study="study", yi="yi", vi="vi", device="mps"), "unsupported_device"),
        "dependent_trimfill": reject(lambda: oe.meta_trimfill(data=hack, yi="yi", vi="vi", dependence="multilevel"), "unsupported_dependence"),
        "pet_constant_variance": reject(lambda: oe.meta_pet_peese(data=hack.assign(vi=.02), yi="yi", vi="vi"), "singular_design"),
        "selection_empty_interval": reject(lambda: oe.meta_selection(data=hack, yi="yi", vi="vi", steps=[1e-12, .5]), "empty_interval"),
    }
    return results, failures


def replay(results):
    """Semantic prediction/CR2 replay with variance optimization forbidden."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Variance optimization during saved-result replay")

    maximize = mk.Likelihood.maximize
    mk.Likelihood.maximize = forbidden
    try:
        for name in ("ml", "reml", "published_cr2_fit"):
            restored = ml.make_result(results[name].attrs["prediction_state"])
            # Prediction state reconstructs canonical scientific tables. The
            # outer fit's workspace receipt belongs to the complete summary,
            # whose exact attributes/order/dtypes/LaTeX unpack() checks above.
            assert restored.title == results[name].title
            assert restored.attrs["prediction_state"] == results[name].attrs["prediction_state"]
            assert list(restored) == list(results[name])
            for table in restored:
                pd.testing.assert_frame_equal(restored[table], results[name][table])
        assert pack(prediction(results["ml"], None)) == pack(results["prediction_ml"])
        assert pack(prediction(results["reml"], pd.DataFrame({"x1": [0., 1.], "x2": [0., -.5]}))) == pack(results["prediction_reml"])
        robust = oe.meta_cr2(results["published_cr2_fit"], contrasts=pd.DataFrame(
            {"Intercept": [0., 1.], "yc": [1., 0.], "z": [1., 0.]}, index=["a", "b"]),
            null=pd.Series([.01, .1], index=["a", "b"]))
        assert pack(robust) == pack(results["cr2"])
        for name in results:
            if name.startswith("dependent_") and not name.endswith(("_cr2", "_cr1")):
                assert pack(oe.meta_cr2(results[name])) == pack(results[name + "_cr2"])
            if name.startswith("dependent_") and name.endswith("_cr1"):
                assert pack(oe.meta_cr2(results[name])) == pack(results[name + "_cr2"])
    finally:
        mk.Likelihood.maximize = maximize


if "META_RESULT_DIRECTORY" in globals():
    fixture_directory = Path(globals()["META_FIXTURE_DIRECTORY"])
    result_directory = Path(globals()["META_RESULT_DIRECTORY"])
    if globals().get("META_REPLAY", False):
        stored = json.loads((result_directory / "results.json").read_text())
        results = {name: unpack(payload) for name, payload in stored.items()}
        replay(results)
        print("META_EIGHT_REPLAY_OK")
    else:
        results, failures = run(fixture_directory)
        replay(results)
        result_directory.mkdir(parents=True, exist_ok=False)
        payload = json.dumps({name: pack(result) for name, result in results.items()}, allow_nan=False, sort_keys=True)
        (result_directory / "results.json").write_text(payload)
        (result_directory / "failures.json").write_text(json.dumps(failures, sort_keys=True))
        if "display" in globals():
            globals()["display"](pd.DataFrame({"case": list(results), "tables": [len(x) for x in results.values()]}))
        print("META_EIGHT_OK " + json.dumps({"cases": list(results), "negative_cases": failures,
              "sha256": hashlib.sha256(payload.encode()).hexdigest(),
              "full_roundtrip": True, "frozen": bool(getattr(sys, "frozen", False)),
              "tables": sum(len(x) for x in results.values())}))
