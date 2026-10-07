"""Public statistical API, dependency boundaries and reproducible execution."""

import json
from importlib.metadata import requires
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import openecon as oe
from openecon.cli import app
from openecon.server import create_app


def test_estimators_do_not_import_statsmodels():
    script = """
import importlib.abc
import random
import sys
class RejectEstimators(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "statsmodels":
            raise AssertionError("Unexpected estimator import: " + fullname)
sys.meta_path.insert(0, RejectEstimators())
import openecon as oe
rng = random.Random(721)
x = [rng.gauss(0, 1) for _ in range(160)]
data = {"x": x, "wage": [1 + 2*v + rng.gauss(0,1) for v in x],
        "binary": [rng.randrange(2) for _ in x]}
for estimator in ["ols", "logit", "probit"]:
    result = getattr(oe, estimator)(data=data, y="wage" if estimator == "ols" else "binary", x=["x"])
    assert result.provenance["backend"] == "openecon.torch"
    assert "statsmodels" not in result.provenance["versions"]
assert "statsmodels" not in sys.modules
print("independent")
"""
    completed = subprocess.run([sys.executable, "-c", script], check=True, text=True, capture_output=True, timeout=60)
    assert completed.stdout.strip() == "independent"


def test_estimator_framework_and_numpy_are_not_direct_runtime_dependencies():
    dependencies = requires("openecon")
    assert not any(req.lower().startswith(("statsmodels", "numpy")) for req in dependencies)
    assert any(req.lower().startswith("torch") and "extra ==" not in req for req in dependencies)


@pytest.mark.parametrize("removed", [{"backend": "numpy"}, {"backend": "torch"}, {"device": "cpu"}, {"device": "cuda"}])
def test_public_spec_rejects_removed_execution_choices(removed):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        oe.ModelSpec(outcome="y", predictors=["x"], **removed)


@pytest.mark.parametrize("shift", [0.0, 1e10, 1e12])
@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_separation_diagnostic_respects_predictor_location(shift, estimator):
    data = pd.DataFrame({"x": np.array([-3, -2, -1, 1, 2, 3]) + shift, "y": [0, 0, 0, 1, 1, 1]})
    with pytest.raises(oe.AnalysisError) as error:
        getattr(oe, estimator)(data=data, y="y", x=["x"])
    assert error.value.code == "separation_detected"


@pytest.mark.parametrize("input_kind", ["frame", "columns", "records"])
@pytest.mark.parametrize("estimator", ["ols", "logit", "probit"])
def test_simple_public_api_accepts_tabular_inputs_and_summarizes(input_kind, estimator):
    frame = oe.example()
    data = frame if input_kind == "frame" else frame.to_dict(orient="list" if input_kind == "columns" else "records")
    outcome = "wage" if estimator == "ols" else "employed"
    result = getattr(oe, estimator)(data=data, y=outcome, x=["education", "experience"])
    assert result.nobs == len(frame)
    assert result.spec.covariance == "nonrobust"
    assert "backend" not in result.spec.model_dump()
    assert "device" not in result.spec.model_dump()
    text = result.summary()
    assert estimator.upper() in text and "education" in text and "Std. error" in text
    assert "torch" not in text and "numpy" not in text
    assert result.model_dump_json()


def test_public_api_cluster_default_and_invalid_column_list():
    result = oe.ols(data=oe.example(), y="wage", x=["education"], cluster="region")
    assert result.spec.covariance == "cluster"
    with pytest.raises(oe.AnalysisError, match="list of predictor"):
        oe.ols(data=oe.example(), y="wage", x="education")


def test_cli_has_no_backend_or_device_options(tmp_path: Path):
    path = tmp_path / "data.csv"
    oe.example().to_csv(path, index=False)
    runner = CliRunner()
    result = runner.invoke(app, ["fit", str(path), "-y", "wage", "-x", "education"])
    assert result.exit_code == 0, result.output
    actual = json.loads(result.stdout)
    assert "backend" not in actual["spec"] and "device" not in actual["spec"]
    rejected = runner.invoke(app, ["fit", str(path), "-y", "wage", "-x", "education", "--backend", "torch"])
    assert rejected.exit_code != 0


def test_http_persists_statistical_spec_and_replay(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    client.headers["X-OpenEcon-Token"] = client.get("/api/session").json()["token"]
    data = client.post("/api/datasets/example").json()
    response = client.post("/api/analyses", json={"dataset_id": data["id"], "spec": {"outcome": "wage", "predictors": ["education", "experience"]}})
    assert response.status_code == 200, response.text
    result = response.json()
    saved = client.get("/api/results/" + result["id"]).json()
    assert saved["provenance"]["backend"] == "openecon.torch"
    assert saved["provenance"]["versions"]["torch"]
    script = client.get("/api/results/" + result["id"] + "/python").text
    assert '"backend"' not in script and '"device"' not in script
    assert oe.capabilities()["engine"] == "openecon"
    assert "backends" not in oe.capabilities()
