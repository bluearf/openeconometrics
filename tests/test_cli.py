from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from openecon.analysis import fit
from openecon.cli import app
from openecon.models import ModelSpec


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(17)
    education = rng.normal(12, 2, 80)
    experience = rng.uniform(0, 20, 80)
    frame = pd.DataFrame(
        {
            "wage": 2 + 0.7 * education + 0.2 * experience + rng.normal(0, 1, 80),
            "education": education,
            "experience": experience,
            "group": np.repeat(np.arange(8), 10),
        }
    )
    path = tmp_path / "wages.csv"
    frame.to_csv(path, index=False)
    return path


def test_inspect_and_import_share_dataset_metadata(data_file: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    inspected = runner.invoke(app, ["inspect", str(data_file)])
    assert inspected.exit_code == 0, inspected.output
    assert json.loads(inspected.stdout)["row_count"] == 80
    imported = runner.invoke(
        app, ["import", str(data_file), "--workspace", str(tmp_path / "workspace")]
    )
    assert imported.exit_code == 0, imported.output
    metadata = json.loads(imported.stdout)
    assert metadata["row_count"] == 80
    assert metadata["id"]


def test_fit_stdout_and_saved_json_match_engine(data_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "results" / "model.json"
    result = CliRunner().invoke(
        app,
        [
            "fit",
            str(data_file),
            "--outcome",
            "wage",
            "--predictor",
            "education",
            "--predictor",
            "experience",
            "--covariance",
            "HC3",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload == json.loads(output.read_text(encoding="utf-8"))
    expected = fit(
        ModelSpec(outcome="wage", predictors=["education", "experience"], covariance="HC3"),
        data=pd.read_csv(data_file),
    ).model_dump(mode="json")
    assert payload["coefficients"] == expected["coefficients"]
    assert payload["nobs"] == expected["nobs"] == 80
    assert "backend" not in payload["spec"]
    assert "device" not in payload["spec"]
    assert payload["provenance"]["backend"] == "openecon.torch"


def test_cli_exposes_statistical_options_without_execution_selectors(data_file: Path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["fit", "--help"])
    assert help_result.exit_code == 0
    assert "--backend" not in help_result.output and "--device" not in help_result.output
    for option, value in [("--backend", "numpy"), ("--device", "cuda")]:
        rejected = runner.invoke(app, ["fit", str(data_file), "-y", "wage", "-x", "education", option, value])
        assert rejected.exit_code != 0


def test_cluster_selection_is_explicit_and_conflicts_fail(data_file: Path) -> None:
    runner = CliRunner()
    arguments = ["fit", str(data_file), "--outcome", "wage", "--predictor", "education"]
    clustered = runner.invoke(app, arguments + ["--cluster", "group"])
    assert clustered.exit_code == 0, clustered.output
    assert json.loads(clustered.stdout)["spec"]["covariance"] == "cluster"
    conflict = runner.invoke(app, arguments + ["--cluster", "group", "--covariance", "HC3"])
    assert conflict.exit_code == 1
    assert conflict.stdout == ""
    assert "requires --covariance cluster" in json.loads(conflict.stderr)["error"]


def test_missing_data_error_is_json_and_does_not_write_output(
    data_file: Path, tmp_path: Path
) -> None:
    frame = pd.read_csv(data_file)
    frame.loc[0, "education"] = np.nan
    frame.to_csv(data_file, index=False)
    output = tmp_path / "invalid.json"
    result = CliRunner().invoke(
        app,
        [
            "fit",
            str(data_file),
            "--outcome",
            "wage",
            "--predictor",
            "education",
            "--missing",
            "raise",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "error" in json.loads(result.stderr)
    assert not output.exists()


def test_bad_input_file_is_reported_without_traceback(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["inspect", str(tmp_path / "not-there.csv")])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "error" in json.loads(result.stderr)
    assert "Traceback" not in result.output


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_binary_default_and_explicit_unsupported_covariance(
    data_file: Path, estimator: str
) -> None:
    frame = pd.read_csv(data_file)
    frame["employed"] = np.random.default_rng(53).binomial(1, 0.45, len(frame))
    frame.to_csv(data_file, index=False)
    arguments = [
        "fit",
        str(data_file),
        "--outcome",
        "employed",
        "--predictor",
        "education",
        "--estimator",
        estimator,
    ]
    runner = CliRunner()
    default = runner.invoke(app, arguments)
    assert default.exit_code == 0, default.output
    assert json.loads(default.stdout)["spec"]["covariance"] == "nonrobust"
    unsupported = runner.invoke(app, arguments + ["--covariance", "HC3"])
    assert unsupported.exit_code == 1
    assert unsupported.stdout == ""
    assert "error" in json.loads(unsupported.stderr)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
def test_serve_rejects_non_loopback_before_starting_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    def must_not_run(*args: object, **kwargs: object) -> None:
        pytest.fail("The server must not start on a non-loopback address.")

    monkeypatch.setattr("uvicorn.run", must_not_run)
    workspace = tmp_path / "uncreated-workspace"
    result = CliRunner().invoke(app, ["serve", "--host", host, "--workspace", str(workspace)])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "only supports loopback hosts" in json.loads(result.stderr)["error"]
    assert not workspace.exists()
