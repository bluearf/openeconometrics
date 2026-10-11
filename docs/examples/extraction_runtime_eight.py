"""Complete extraction/rotation/bootstrap results for source/frozen acceptance.

Set EXTRACTION_RESULT_DIRECTORY to a fresh absolute directory. All synthetic
inputs, parameter/index draws, covariance, diagnostics and portable presentation
state persist. EXTRACTION_REPLAY restores the files without fitting anything.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback

import pandas as pd
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet

BOOTSTRAP_METHODS = ("ml", "minres", "alpha", "image_covariance", "generalized_image")
CASE_PROGRESS = []


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


def inputs():
    """Preregistered heterogeneous one-factor and two-factor raw samples."""
    generator = torch.Generator(device="cpu").manual_seed(824)
    latent = torch.randn((242, 1), generator=generator, dtype=torch.float64)
    loading = torch.tensor([[.75, .72, .80, .68, .77]], dtype=torch.float64)
    raw = latent @ loading + .65 * torch.randn((242, 5), generator=generator, dtype=torch.float64)
    one = pd.DataFrame(raw.numpy(), columns=[f"x{i+1}" for i in range(5)],
                       index=[f"source:{i//2}" for i in range(242)])
    one.iloc[2, 0] = float("nan")
    one.iloc[27, 2] = float("nan")

    generator = torch.Generator(device="cpu").manual_seed(825)
    loading = torch.tensor([[.8, .12], [.7, .21], [.72, .08],
                            [.11, .82], [.18, .76], [.08, .7]], dtype=torch.float64)
    raw = torch.randn((240, 2), generator=generator, dtype=torch.float64) @ loading.T
    raw += .55 * torch.randn((240, 6), generator=generator, dtype=torch.float64)
    two = pd.DataFrame(raw.numpy(), columns=[f"v{i+1}" for i in range(6)],
                       index=[f"source:{i//2}" for i in range(240)])
    two.iloc[4, 1] = float("nan")
    return one, two


def query(data):
    result = data.iloc[[7, 2, 7, 19]].copy()
    result.iloc[1, 0] = float("nan")
    return result


def scores(fit, data):
    return TableSet({"scores": oe.factor_scores(fit, query(data))},
                    title="Saved extraction score query with duplicate labels and missing row")


def reject(function, expected):
    try:
        function()
    except AnalysisError as exc:
        assert exc.code == expected, (exc.code, expected)
        return exc.code
    raise AssertionError("Unsupported or invalid input was accepted: " + expected)


def run():
    torch.set_num_threads(1)
    one, two = inputs()
    names = list(two)
    results = {}

    def save(name, result):
        results[name] = result
        CASE_PROGRESS.append(name)

    for method in ("gls", "generalized_image"):
        for score_method in ("regression", "bartlett"):
            name = method + "_" + score_method
            fit = oe.factor(two, names, method=method, factors=2, scores=score_method)
            save(name, fit)
            save(name + "_scores", scores(unpack(pack(fit)), two))

    base = oe.factor(two, names, factors=2)
    transform = torch.tensor([[1., -.17], [.21, 1.]], dtype=torch.float64)
    transform /= transform.square().sum(0).sqrt()
    target = base["loadings"].to_numpy() @ torch.linalg.inv(transform).T.numpy()
    target_fit = oe.factor(two, names, factors=2, rotate="target", target=target,
                           target_oblique=True, rotation_tolerance=1e-9)
    save("complete_oblique_target", target_fit)
    save("complete_oblique_target_scores", scores(unpack(pack(target_fit)), two))
    for method in BOOTSTRAP_METHODS:
        fit = oe.factor_bootstrap(one, list(one), method=method, replications=199,
                                   seed=831, anchor="x3")
        assert fit.attrs["successful_replications"] == fit.attrs["replications"] == 199
        assert fit.attrs["failed_replications"] == []
        save("bootstrap_" + method, fit)
    for name, result in results.items():
        assert pack(unpack(pack(result))) == pack(result), name
    failures = {
        "gls_iteration_failure": reject(lambda: oe.factor(two, names, method="gls", factors=2,
                                                            max_iterations=1), "no_convergence"),
        "generalized_image_fixed_count": reject(lambda: oe.factor(two, names, method="generalized_image"), "invalid_option"),
        "complete_target_missing_cell": reject(lambda: oe.factor(two, names, factors=2, rotate="target",
                                                                     target=target*float("nan"), target_oblique=True), "invalid_spec"),
        "complete_target_rank": reject(lambda: oe.factor(two, names, factors=2, rotate="target",
                                                             target=target[:, [0, 0]], target_oblique=True), "singular_matrix"),
        "bootstrap_missing_raise": reject(lambda: oe.factor_bootstrap(one, list(one), method="minres",
                                                                      replications=199, missing="raise"), "missing_values"),
        "bootstrap_unsupported_method": reject(lambda: oe.factor_bootstrap(one, list(one), method="gls",
                                                                           replications=199), "invalid_option"),
        "bootstrap_percentile_tail": reject(lambda: oe.factor_bootstrap(one, list(one), method="ml",
                                                                        replications=19), "insufficient_replications"),
        "bootstrap_anchor": reject(lambda: oe.factor_bootstrap(one, list(one), method="alpha",
                                                               replications=199, anchor="unknown"), "invalid_spec"),
    }
    return results, failures


def replay(results):
    """Restore every full result and score without extraction or optimization."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Fitting or optimization during saved-result replay")

    targets = {"openecon": ("factor", "factor_bootstrap"),
               "openecon.engines.optimize": ("maximize_bfgs",),
               "openecon.econometrics.multivariate.extraction": ("maximum_likelihood", "minimum_residual", "principal_factors"),
               "openecon.econometrics.multivariate.extraction_additional": ("generalized_least_squares", "generalized_image_factors"),
               "openecon.econometrics.multivariate.rotation_extensions": ("_fit",),
               "openecon.econometrics.multivariate.extraction_uncertainty": ("factor",)}
    originals = []
    try:
        for module_name, attributes in targets.items():
            module = importlib.import_module(module_name)
            for name in attributes:
                originals.append((module, name, getattr(module, name)))
                setattr(module, name, forbidden)
        _, two = inputs()
        for name, result in results.items():
            restored = unpack(pack(result))
            assert pack(restored) == pack(result), name
            if name.endswith("_scores") or name.startswith("bootstrap_"):
                continue
            assert pack(scores(restored, two)) == pack(results[name + "_scores"]), name
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)


if "EXTRACTION_RESULT_DIRECTORY" in globals():
    result_directory = Path(globals()["EXTRACTION_RESULT_DIRECTORY"]).resolve()
    if globals().get("EXTRACTION_REPLAY", False):
        stored = json.loads((result_directory / "results.json").read_text())
        results = {name: unpack(payload) for name, payload in stored.items()}
        replay(results)
        print("EXTRACTION_EIGHT_REPLAY_OK")
    else:
        result_directory.mkdir(parents=True, exist_ok=False)
        try:
            one, two = inputs()
            input_payload = {"onefactor": json.loads(one.to_json(orient="split", double_precision=15)),
                             "twofactor": json.loads(two.to_json(orient="split", double_precision=15))}
            (result_directory / "inputs.json").write_text(json.dumps(input_payload, allow_nan=False, sort_keys=True))
            results, failures = run()
            replay(results)
            payload = json.dumps({name: pack(result) for name, result in results.items()}, allow_nan=False, sort_keys=True)
            (result_directory / "results.json").write_text(payload)
            (result_directory / "failures.json").write_text(json.dumps(failures, sort_keys=True))
            if "display" in globals():
                globals()["display"](pd.DataFrame({"case": list(results), "tables": [len(x) for x in results.values()]}))
            print("EXTRACTION_EIGHT_OK " + json.dumps({"cases": list(results), "negative_cases": failures,
                  "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                  "full_roundtrip": True, "frozen": bool(getattr(sys, "frozen", False)),
                  "bootstrap_replications": {method: results["bootstrap_"+method].attrs["replications"] for method in BOOTSTRAP_METHODS},
                  "tables": sum(len(x) for x in results.values())}))
        except BaseException as exc:
            receipt = {"type": type(exc).__name__, "message": str(exc), "completed_cases": CASE_PROGRESS,
                       "traceback": traceback.format_exc()}
            for attribute in ("code", "failures", "replications_attempted", "successful_replications",
                              "replicate_indices", "successful_positions", "successful_vectors"):
                if hasattr(exc, attribute):
                    receipt[attribute] = getattr(exc, attribute)
            (result_directory / "run-failure.json").write_text(json.dumps(receipt, allow_nan=False, sort_keys=True))
            raise
