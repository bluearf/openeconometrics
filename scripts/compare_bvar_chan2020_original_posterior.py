"""Once-only full deterministic original-author posterior comparison; not a sampler."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time


def pin(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write(path, value):
    data = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    if len(data) > 32 * 1024**2:
        raise ValueError("Complete comparison output exceeds fixed 32MiB bound")
    with path.open("xb") as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, native, output = args.root.resolve(), args.native.resolve(), args.output.resolve()
    fixture = json.loads(args.fixture.read_bytes())
    if fixture["series_counts"] != [2, 3, 4] or fixture["lags"] != 4 or fixture["intercept"] is not True:
        raise ValueError("Prospective fixed comparison geometry changed")
    if fixture["tolerance"] != {"rtol": 2e-12, "atol": 2e-12}:
        raise ValueError("Prospective full-array statistical tolerance changed")

    import numpy as np
    import pandas as pd
    from scipy.io import loadmat
    import torch
    from openecon.econometrics.bayesian_var.api import bayes_var
    from openecon.econometrics.bayesian_var.design import matrices
    from openecon.econometrics.bayesian_var.posterior import BayesianVARPosterior

    torch.set_num_threads(1)
    levels = np.asarray(fixture["levels"], dtype=np.float64)
    if levels.shape != (68, 4) or not np.isfinite(levels).all():
        raise ValueError("Full fixed fixture arrays changed shape or finiteness")
    runtime = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "torch", "pandas", "pydantic")},
        "torch_threads": torch.get_num_threads(), "default_dtype": str(torch.get_default_dtype()),
        "default_device": str(torch.get_default_device()),
        "native_thread_environment": {key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    # Pin every actually imported project module before the first fit, then all
    # additionally imported project modules at the end. This is a wheel/source
    # byte identity check, not an installed native application claim.
    def project_origins():
        result = []
        for name, module in sorted(sys.modules.items()):
            if name != "openecon" and not name.startswith("openecon."):
                continue
            origin = getattr(module, "__file__", None)
            if not origin or not str(origin).endswith(".py"):
                raise ValueError("Project module lacks an explicit Python source origin: " + name)
            path = Path(origin).resolve()
            relative = Path(*name.split("."))
            expected = root / "src" / relative.with_suffix(".py")
            if path.name == "__init__.py":
                expected = root / "src" / relative / "__init__.py"
            if path.read_bytes() != expected.read_bytes():
                raise ValueError("Imported package source differs from committed source: " + name)
            result.append({"module": name, "actual": pin(path), "committed": pin(expected)})
        return result

    runtime["project_origins_before"] = project_origins()
    write(output / "actual-scientific-runtime-before.json", runtime)

    def rng_state():
        numpy = np.random.get_state()
        return {"python": random.getstate(), "torch": torch.get_rng_state().tolist(),
                "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])]}

    before = rng_state()
    write(output / "complete-ambient-RNG-before.json", before)
    results = []
    try:
        for m in fixture["series_counts"]:
            started = time.monotonic()
            case = output / ("series-" + str(m))
            case.mkdir()
            original_path = native / f"original-author-series-{m}.mat"
            original = loadmat(original_path, struct_as_record=False, squeeze_me=False)
            def arr(name, shape):
                value = original[name]
                if hasattr(value, "toarray"):
                    value = value.toarray()
                value = np.asarray(value, dtype=np.float64)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("Native full array geometry changed: " + name)
                return value
            k, nobs = 1 + 4*m, 64
            for name, expected in (("n", m), ("p", 4), ("k", k), ("Tt", nobs),
                                   ("nsims", 0), ("burnin", 0), ("nu0", m+3), ("nuhat", m+67)):
                if arr(name, (1, 1))[0, 0] != expected:
                    raise ValueError("Native declared scalar changed: " + name)
            for name in ("c1", "c2"):
                if arr(name, (1, 1))[0, 0] != fixture[name]:
                    raise ValueError("Explicit native hyperparameter differs: " + name)
            if not np.array_equal(arr("Y0", (4, m)), levels[:4, :m]):
                raise ValueError("Full native initial conditioning observations differ")
            native_struct = original["native_runtime"][0, 0]
            native_runtime = {name: str(getattr(native_struct, name).item()) for name in
                ("version", "computer", "original_script", "original_prior")}
            if native_runtime["version"] != "8.4.0":
                raise ValueError("Native MAT runtime differs from preregistered Octave8.4.0")
            source_root = native.parent / "unchanged-original-source"
            for key, filename in (("original_script", "forecast_BVAR_NCP.m"), ("original_prior", "prior_NC.m")):
                if Path(native_runtime[key]).resolve() != source_root / filename:
                    raise ValueError("Native original source origin differs")
            if native_struct.sampling_iterations.item() != 0 or native_struct.startup_RNG_purity_claimed.item() != 0:
                raise ValueError("Native sampling/startup qualification differs")
            # Complete original before/after native-generator states are stored,
            # without a claim about interpreter startup or cross-runtime seeds.
            native_rng = []
            for index, name in enumerate(("rand", "randn", "rande", "randg", "randp")):
                a = np.asarray(original["before_states"][0, index])
                b = np.asarray(original["after_states"][0, index])
                if a.shape != (625, 1) or not np.array_equal(a, b):
                    raise ValueError("Native original posterior changed generator state")
                native_rng.append({"generator": name, "before": a.tolist(), "after": b.tolist()})
            prior = {"mean": arr("A0", (k, m)).tolist(),
                     "row_scale": np.diag(arr("VA0", (k, 1))[:, 0]).tolist(),
                     "innovation_scale": arr("S0", (m, m)).tolist(),
                     "degrees_of_freedom": float(m+3)}
            names = ["y" + str(i+1) for i in range(m)]
            data = pd.DataFrame({"time": np.arange(1000, 1068, dtype=np.int64),
                                 **{name: levels[:, i] for i, name in enumerate(names)}})
            result = bayes_var(data, names, time="time", prior=prior, lags=4, intercept=True)
            y, x, _ = matrices(result.source.model_dump(mode="python"), 4, True)
            expected_arrays = {
                "design": arr("X", (64, k)), "response": arr("shortYt", (64, m)),
                "design_crossproduct": arr("XX", (k, k)),
                "precision": arr("KA", (k, k)), "location": arr("Ahat", (k, m)),
                "innovation_scale": arr("Shat", (m, m)),
                "row_scale": arr("posterior_row_scale", (k, k)),
            }
            actual_arrays = {"design": x.tolist(), "response": y.tolist(), "design_crossproduct": (x.T @ x).tolist(),
                **{name: getattr(result.algebra, name) for name in
                   ("precision", "location", "innovation_scale", "row_scale")}}
            checks = []
            for name, expected in expected_arrays.items():
                actual = np.asarray(actual_arrays[name], dtype=np.float64)
                if actual.shape != expected.shape:
                    raise ValueError("Own/native complete array shapes differ: " + name)
                np.testing.assert_allclose(actual, expected, **fixture["tolerance"])
                checks.append({"name": name, "shape": list(expected.shape),
                    "full_native_values": expected.tolist(), "full_own_values": actual.tolist(),
                    "max_absolute_error": float(np.max(np.abs(actual-expected)))})
            if result.nobs != 64 or result.nobs_original != 68 or result.algebra.degrees_of_freedom != m+67:
                raise ValueError("Own/native full observation or IW geometry differs")
            state = result.model_dump(mode="json")
            write(case / "complete-own-posterior-state.json", state)
            persisted = json.loads((case / "complete-own-posterior-state.json").read_bytes())
            if BayesianVARPosterior.model_validate(persisted).model_dump(mode="json") != persisted:
                raise ValueError("Complete own posterior JSON readback changed")
            body = {"status": "FULL_ORIGINAL_AUTHOR_POSTERIOR_ARRAYS_MATCH", "series": m,
                "native_file": pin(original_path), "native_generator_states": native_rng,
                "complete_native_runtime": native_runtime,
                "full_prior": prior, "all_matching_arrays": checks,
                "own_complete_state": pin(case / "complete-own-posterior-state.json"),
                "native_sampling_iterations": 0, "fit_calls": 1,
                "terms": list(result.terms), "source": result.source.model_dump(mode="json"),
                "original_prior_nu": m+3, "posterior_nu": m+67,
                "unmatched_own_moments_remain_in_full_state_without_author_endpoint_claim": True,
                "seconds": time.monotonic()-started}
            write(case / "complete-full-array-comparison.json", body)
            results.append(body)
    finally:
        after = rng_state()
        write(output / "complete-ambient-RNG-after.json", after)
        if before != after:
            raise ValueError("Deterministic original-author comparison advanced ambient RNG")
    runtime["project_origins_after"] = project_origins()
    if runtime["default_dtype"] != str(torch.get_default_dtype()) or runtime["default_device"] != str(torch.get_default_device()) or torch.get_num_threads() != 1:
        raise ValueError("Own deterministic comparison changed default dtype/device/thread contract")
    write(output / "actual-scientific-runtime-after.json", runtime)
    write(output / "complete-original-author-comparison.json", {
        "schema": "openecon.bvar.original-author-Octave.deterministic-posterior-comparison.v1",
        "status": "THREE_FIXED_ORIGINAL_AUTHOR_POSTERIOR_COMPARISONS_PASS",
        "cases": results, "tolerance": fixture["tolerance"],
        "actual_fit_calls": 3, "native_sampling_iterations": 0,
        "ambient_RNG_unchanged_after_imports": True,
        "qualification": "Original-author posterior source under Octave; no MATLAB/vendor, stochastic forecast, full empirical paper reproduction or installed application claim."})


if __name__ == "__main__":
    main()
