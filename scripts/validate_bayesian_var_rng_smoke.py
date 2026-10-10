"""One preregistered tiny native RNG/transport smoke; not distribution calibration.

Register exact sources and inputs first; execute that immutable registration once.
Four original + four reproducibility draws = eight total native primitive draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform

import pandas as pd
import torch

from openecon.econometrics.bayesian_var import draws as draw_module
from openecon.econometrics.bayesian_var.api import bayes_var
from openecon.econometrics.bayesian_var.draws import BayesianVARDraws, bayes_var_draws
from openecon.econometrics.bayesian_var.posterior import raw

SEED = 20261009
COUNT = 4
DATA = {
    "time": list(range(12)),
    "y0": [1.0, 1.1, 0.9, 1.2, 1.3, 1.1, 1.4, 1.6, 1.5, 1.7, 1.8, 1.6],
    "y1": [0.2, 0.3, 0.4, 0.3, 0.35, 0.5, 0.4, 0.45, 0.6, 0.55, 0.65, 0.7],
}
PRIOR = {
    "mean": [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
    "row_scale": [[1.0, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]],
    "innovation_scale": [[1.0, 0.2], [0.2, 0.7]],
    "degrees_of_freedom": 6.0,
}


def source_pin():
    root = pathlib.Path(__file__).resolve().parents[1]
    paths = sorted(root.glob("src/openecon/econometrics/bayesian_var/*.py")) + [
        pathlib.Path(__file__).resolve()
    ]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def declaration():
    return {
        "purpose": "API/local-RNG/saved-state transport smoke only; no SBC/coverage/distribution acceptance",
        "seed": SEED,
        "original_draws": COUNT,
        "reproducibility_draws": COUNT,
        "total_native_primitive_draws": 2 * COUNT,
        "m": 2,
        "p": 1,
        "k": 3,
        "original_n": 12,
        "data": DATA,
        "prior": PRIOR,
        "sources": source_pin(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "pandas": pd.__version__,
        "native_threads": 1,
        "all_draws_retained": True,
        "fixed_seed_no_selection": True,
    }


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def forbidden(*args, **kwargs):
    raise AssertionError("Saved-state replay must not draw again.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("register", "execute"))
    parser.add_argument("registration", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    expected = declaration()
    if args.mode == "register":
        write_new(args.registration, expected)
        print(args.registration)
        return
    if args.output is None or args.output.exists():
        raise ValueError("Execute requires a new output path; this smoke is never overwritten.")
    if json.loads(args.registration.read_text()) != expected:
        raise ValueError("Smoke source/runtime/input registration changed before the first draw.")
    posterior = bayes_var(pd.DataFrame(DATA), ["y0", "y1"], time="time", prior=PRIOR, lags=1)
    before = (
        torch.get_rng_state().clone(),
        torch.get_default_dtype(),
        str(torch.get_default_device()),
    )
    receipt = {
        "registration_sha256": hashlib.sha256(args.registration.read_bytes()).hexdigest(),
        "declaration": expected,
        "posterior": raw(posterior),
    }
    try:
        joint = bayes_var_draws(posterior, draws=COUNT, seed=SEED)
        receipt["original_joint_packet"] = raw(joint)
        factors, normals = draw_module.primitive_stream(posterior, COUNT, SEED)
        receipt["reproducibility_primitives"] = {"bartlett": factors, "standard_normals": normals}
        assert factors == [list(map(list, a)) for a in joint.bartlett]
        assert normals == [list(map(list, a)) for a in joint.standard_normals]
        assert factors[0] != factors[1] and normals[0] != normals[1]
        assert len(joint.coefficients) == COUNT and len(joint.innovations) == COUNT
        assert torch.equal(before[0], torch.get_rng_state())
        assert before[1] == torch.get_default_dtype() and before[2] == str(
            torch.get_default_device()
        )
        encoded = joint.model_dump_json()
        draw_module.primitive_stream = forbidden
        torch.randn = forbidden
        torch._standard_gamma = forbidden
        reloaded = BayesianVARDraws.model_validate_json(encoded)
        copied = reloaded.model_copy(deep=True)
        assert raw(reloaded) == raw(joint) == raw(copied)
        receipt.update(
            {
                "passed": True,
                "same_seed_exact_reproducibility": True,
                "within_stream_advancement": True,
                "global_rng_dtype_device_unchanged": True,
                "full_arrays_and_primitive_readback_without_rng": True,
                "stable_count": sum(joint.stable),
                "unstable_count": COUNT - sum(joint.stable),
            }
        )
    except BaseException as exc:
        receipt.update({"passed": False, "failure_type": type(exc).__name__, "failure": str(exc)})
        write_new(args.output, receipt)
        raise
    write_new(args.output, receipt)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": receipt["passed"],
                "original_draws": COUNT,
                "total_native_draws": 2 * COUNT,
                "stable": receipt["stable_count"],
                "unstable": receipt["unstable_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
