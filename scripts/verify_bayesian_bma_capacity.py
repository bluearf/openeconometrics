"""Measured pre-allocation lifecycle refusals on actual saved finite BMA states."""

from __future__ import annotations

import argparse
import hashlib
import json
import tracemalloc
from pathlib import Path

import pandas as pd

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.bayesian.bma import BMAResult, bayes_bma, bayes_bma_restore
from openecon.econometrics.bayesian.bma_draws import (
    BMADraws,
    bayes_bma_draws,
    bayes_bma_draws_restore,
)
from openecon.econometrics.bayesian.bma_query import (
    BMAPrediction,
    bayes_bma_predict,
    bayes_bma_prediction_restore,
)
from openecon.resources import use_workspace_budget

TYPES = {
    "openecon.bayesian_bma.v1": (BMAResult, bayes_bma_restore),
    "openecon.bayesian_bma_prediction.v1": (BMAPrediction, bayes_bma_prediction_restore),
    "openecon.bayesian_bma_draws.v1": (BMADraws, bayes_bma_draws_restore),
}


def measured(name, fn):
    tracemalloc.start()
    try:
        try:
            fn()
        except AnalysisError as exc:
            error = {"code": exc.code, "message": str(exc)}
        else:
            raise AssertionError(name + " did not refuse")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    if peak >= 32768:
        raise AssertionError(name + f" pre-admission allocation peak {peak}")
    return {"operation": name, "refused": True, "peak_python_bytes": peak, **error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = {"passed": True, "states": []}
    for path in args.state:
        raw = path.read_bytes()
        saved = json.loads(raw)
        cls, restore = TYPES[saved["schema_version"]]
        state = restore(raw)
        dense = [{} for _ in range(50000)]
        text = json.dumps({"schema_version": state.schema_version, "payload": dense})
        forged = cls.model_construct(schema_version=state.schema_version, payload=dense)
        with use_workspace_budget(1):
            checks = [
                measured("typed JSON decoder", lambda: cls.model_validate_json(text)),
                measured("public restore JSON decoder", lambda: restore(text)),
                measured("deep-copy payload", lambda: forged.model_copy(deep=True)),
                measured(
                    "deep-copy dense update",
                    lambda: state.model_copy(deep=True, update={"payload": dense}),
                ),
            ]
        checks.append(
            measured("formatted huge indentation", lambda: state.model_dump_json(indent=10**9))
        )
        receipt["states"].append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "valid_semantic_roundtrip": True,
                "decoded_bytes": len(raw),
                "dense_json_input_bytes": len(text),
                "dense_nodes": 50000,
                "budget_mib": 1,
                "checks": checks,
            }
        )
    parent_path = next(
        path
        for path in args.state
        if json.loads(path.read_bytes())["schema_version"] == "openecon.bayesian_bma.v1"
    )
    parent = bayes_bma_restore(parent_path.read_bytes())
    names = [
        parent.payload["spec"]["y"],
        *parent.payload["spec"]["forced"],
        *parent.payload["spec"]["optional"],
    ]
    frame = pd.DataFrame({name: [0.1 * j for j in range(8)] for name in names})
    name = "name"
    for _ in range(14):
        name = (name, name)
    labels = pd.Index([(j, j + 1, j + 2, j + 3) for j in range(4096)], tupleize_cols=False)
    indices = {
        "expanded tuple name": pd.RangeIndex(8, name=name),
        "unused tuple categories": pd.CategoricalIndex(
            pd.Categorical.from_codes(range(8), categories=labels)
        ),
        "unused tuple levels": pd.MultiIndex(
            levels=[labels, ["a"]], codes=[list(range(8)), [0] * 8]
        ),
    }
    raw_checks = []
    for name, index in indices.items():
        frame.index = index
        with use_workspace_budget(1):
            raw_checks.extend(
                [
                    measured(
                        name + " fit",
                        lambda: bayes_bma(
                            data=frame,
                            y=parent.payload["spec"]["y"],
                            forced=parent.payload["spec"]["forced"],
                            optional=parent.payload["spec"]["optional"],
                            priors=parent.payload["priors"],
                            model_prior_odds=parent.payload["prior_model_odds"],
                        ),
                    ),
                    measured(name + " prediction", lambda: bayes_bma_predict(parent, data=frame)),
                    measured(
                        name + " draws",
                        lambda: bayes_bma_draws(parent, data=frame, draws=4, seed=690),
                    ),
                ]
            )
    receipt["raw_index_checks"] = raw_checks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "checks": sum(len(v["checks"]) for v in receipt["states"]) + len(raw_checks),
                "max_peak_python_bytes": max(
                    [c["peak_python_bytes"] for v in receipt["states"] for c in v["checks"]]
                    + [c["peak_python_bytes"] for c in raw_checks]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
