"""Public dense-route discovery and honest streaming boundaries for the new wave."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ModelSpec

NAMES = (
    "fmols",
    "dols",
    "ccr",
    "panel_fmols",
    "panel_dols",
    "pmg",
    "mg",
    "dfe",
    "svar",
    "lp",
    "lpiv",
    "panel_lp",
    "mediation",
    "oaxaca",
)


def test_fourteen_routes_persist_and_run_without_reference_engines():
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, str(root / "scripts/prove_advanced_runtime.py")],
        cwd=root,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        text=True,
        capture_output=True,
        timeout=45,
        check=True,
    )
    receipt = json.loads(process.stdout)
    assert set(receipt["models"]) == set(NAMES)
    assert receipt["forbidden_engines_loaded"] == []
    assert all(m["json_latex_roundtrip"] for m in receipt["models"].values())
    assert receipt["helpers"]["irf_rows"] == 12


@pytest.mark.parametrize("name", NAMES)
def test_dense_only_routes_do_not_collect_an_explicit_dataset(name, monkeypatch):
    columns = {}
    if name == "svar":
        columns = {"system": ["y", "x"]}
    if name == "lpiv":
        columns = {"instruments": ["z"]}
    if name == "mediation":
        columns = {"mediators": ["m"]}
    if name == "oaxaca":
        columns = {"group": "g"}
    is_panel = name in ("panel_fmols", "panel_dols", "pmg", "mg", "dfe", "panel_lp")
    spec = ModelSpec(
        estimator=name,
        outcome="y",
        predictors=[] if name == "svar" else ["x"],
        panel="unit" if is_panel else None,
        time=None if name in ("mediation", "oaxaca") else "t",
        intercept=name
        not in ("fmols", "dols", "ccr", "panel_fmols", "panel_dols", "pmg", "mg", "dfe"),
        cluster="unit" if name == "panel_lp" else None,
        columns=columns,
    )
    frame = pd.DataFrame(
        {
            "y": [1.0, 2.0],
            "x": [2.0, 3.0],
            "unit": [1, 1],
            "t": [0, 1],
            "z": [1.0, 2.0],
            "m": [3.0, 4.0],
            "g": ["A", "B"],
        }
    )
    source = oe.Dataset.from_frame(frame)

    def forbidden(*args, **kwargs):
        raise AssertionError("Dense-only models must not collect an explicit Dataset.")

    monkeypatch.setattr(oe.Dataset, "iter_batches", forbidden)
    with pytest.raises(AnalysisError) as error:
        oe.fit(spec, data=source)
    assert error.value.code == "streaming_unsupported"
