import io
import json
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from openecon.data import HASH_VERSION, DataError, example_frame, frame_hash, read
from openecon.models import ModelSpec
from openecon.workspace import Workspace


def test_dataset_and_result_survive_reopening(tmp_path):
    workspace = Workspace(tmp_path / "project")
    dataset = workspace.create_example()
    result = workspace.run_analysis(dataset["id"], ModelSpec(outcome="wage", predictors=["education", "experience"]))
    reopened = Workspace(tmp_path / "project")
    assert reopened.get_result(result["id"]) == result
    assert reopened.get_dataset(dataset["id"])["data_hash"] == dataset["data_hash"]
    assert reopened.create_example()["id"] == dataset["id"]
    assert result["provenance"]["synthetic_data"] is True
    assert result["nobs"] == 480


def test_source_preserved_and_mutation_detected(tmp_path):
    source = tmp_path / "observations.csv"
    example_frame().to_csv(source, index=False)
    workspace = Workspace(tmp_path / "project")
    dataset = workspace.import_file(source)
    saved_source = workspace.data_path / f"{dataset['id']}.source.csv"
    assert saved_source.read_bytes() == source.read_bytes()
    path = workspace.data_path / f"{dataset['id']}.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "wage"] += 100
    changed.to_parquet(path, index=False)
    with pytest.raises(DataError, match="changed"):
        workspace.load_frame(dataset["id"])


def test_portable_bundle_executes_after_moving(tmp_path):
    workspace = Workspace(tmp_path / "project")
    dataset = workspace.create_example()
    result = workspace.run_analysis(dataset["id"], ModelSpec(outcome="wage", predictors=["education", "experience"]))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    with zipfile.ZipFile(io.BytesIO(workspace.export_bundle(result["id"]))) as archive:
        assert {"dataset.parquet", "analysis.py", "result.json", "dataset.json"} <= set(archive.namelist())
        archive.extractall(destination)
    process = subprocess.run([sys.executable, str(destination / "analysis.py")], cwd=destination,
                             capture_output=True, text=True, check=True)
    reproduced = json.loads(process.stdout)
    assert reproduced["nobs"] == result["nobs"]
    np.testing.assert_allclose([c["estimate"] for c in reproduced["coefficients"]],
                               [c["estimate"] for c in result["coefficients"]], rtol=1e-12)


def test_invalid_identifier_cannot_read_outside_workspace(tmp_path):
    with pytest.raises(DataError):
        Workspace(tmp_path).get_dataset("../../private")


def test_csv_duplicate_header_rejected(tmp_path):
    source = tmp_path / "duplicate.csv"
    source.write_text("x,x,y\n1,2,3\n")
    with pytest.raises(DataError, match="unique"):
        read(source)


def test_csv_missing_large_integer_values_survive_snapshot(tmp_path):
    source = tmp_path / "identifiers.csv"
    source.write_text("id,y\n9007199254740993,1\n,2\n9007199254740995,3\n")
    imported = read(source)
    assert str(imported.id.dtype) == "Int64"
    assert imported.id.iloc[0] == 9007199254740993
    assert imported.id.iloc[2] == 9007199254740995
    assert pd.isna(imported.id.iloc[1])
    workspace = Workspace(tmp_path / "project")
    dataset = workspace.import_file(source)
    snapshot = workspace.load_frame(dataset["id"])
    pd.testing.assert_frame_equal(snapshot, imported)
    assert dataset["preview"][0]["id"] == "9007199254740993"
    assert dataset["preview"][2]["id"] == "9007199254740995"
    assert dataset["preview"][1]["id"] is None
    assert dataset["preview"][0]["y"] == 1
    assert dataset["hash_version"] == HASH_VERSION
    assert dataset["hash_pandas_version"] == pd.__version__


@pytest.mark.parametrize("names", [["x", "x"], ["x", ""], ["x", " "]])
def test_xlsx_duplicate_or_blank_headers_rejected(tmp_path, names):
    source = tmp_path / "ambiguous.xlsx"
    pd.DataFrame([[1, 2], [3, 4]], columns=names).to_excel(source, index=False)
    with pytest.raises(DataError, match="nonempty and unique"):
        read(source)


def test_hash_binds_categorical_reference_and_ordered_flag():
    values = ["b", "a", "b"]
    original = pd.DataFrame({"group": pd.Categorical(values, categories=["a", "b"])})
    reversed_order = pd.DataFrame({"group": pd.Categorical(values, categories=["b", "a"])})
    ordered = pd.DataFrame({"group": pd.Categorical(values, categories=["a", "b"], ordered=True)})
    assert len({frame_hash(original), frame_hash(reversed_order), frame_hash(ordered)}) == 3


def test_category_metadata_roundtrip_and_tampering_are_detected(tmp_path):
    workspace = Workspace(tmp_path / "project")
    frame = pd.DataFrame({"sector": pd.Categorical(["b", "a", "b"], categories=["b", "a"], ordered=True)})
    frame.attrs["metadata"] = {"value_labels": {"code": {1.0: "Yes", 0.0: "No"}}, "column_labels": {"sector": "Sector"}}
    dataset = workspace._save_frame(frame, "Categories", "upload")
    loaded = workspace.load_frame(dataset["id"])
    pd.testing.assert_frame_equal(loaded, frame)
    assert loaded.attrs["metadata"]["value_labels"]["code"]["1.0"] == "Yes"
    loaded["sector"] = loaded.sector.cat.reorder_categories(["a", "b"])
    loaded.to_parquet(workspace.data_path / f"{dataset['id']}.parquet", index=False)
    with pytest.raises(DataError, match="changed"):
        workspace.load_frame(dataset["id"])


@pytest.mark.parametrize("location", ["profile", "parquet"])
def test_metadata_only_changes_fail_integrity_check(tmp_path, location):
    workspace = Workspace(tmp_path / "project")
    dataset = workspace.create_example()
    if location == "profile":
        target = workspace.data_path / f"{dataset['id']}.json"
        edited = json.loads(target.read_text())
        edited["metadata"]["seed"] += 1
        target.write_text(json.dumps(edited))
    else:
        target = workspace.data_path / f"{dataset['id']}.parquet"
        edited = pd.read_parquet(target)
        edited.attrs["metadata"]["seed"] += 1
        edited.to_parquet(target, index=False)
    with pytest.raises(DataError) as caught:
        workspace.load_frame(dataset["id"])
    assert caught.value.code == "DATA_INTEGRITY"


def test_replay_rejects_modified_data_before_fitting(tmp_path):
    workspace = Workspace(tmp_path / "project")
    dataset = workspace.create_example()
    result = workspace.run_analysis(dataset["id"], ModelSpec(outcome="wage", predictors=["education"]))
    script = workspace.python_script(result["id"])
    target = workspace.data_path / f"{dataset['id']}.parquet"
    edited = pd.read_parquet(target)
    edited.loc[0, "wage"] += 1
    edited.to_parquet(target, index=False)
    with pytest.raises(RuntimeError, match="Dataset integrity check failed"):
        exec(compile(script, "analysis.py", "exec"), {})


def test_replay_warns_about_changed_numerical_environment(tmp_path, capsys):
    workspace = Workspace(tmp_path / "project")
    dataset = workspace.create_example()
    result = workspace.run_analysis(dataset["id"], ModelSpec(outcome="wage", predictors=["education"]))
    result["provenance"]["versions"]["numpy"] = "0.0.0"
    target = workspace.result_path / f"{result['id']}.json"
    target.write_text(json.dumps(result))
    script = workspace.python_script(result["id"])
    with pytest.warns(RuntimeWarning, match="Replay environment differs: numpy"):
        exec(compile(script, "analysis.py", "exec"), {})
    reproduced = json.loads(capsys.readouterr().out)
    assert reproduced["nobs"] == result["nobs"]


def test_replay_requires_recorded_pandas_hash_version(tmp_path):
    workspace = Workspace(tmp_path / "project")
    dataset = workspace.create_example()
    result = workspace.run_analysis(dataset["id"], ModelSpec(outcome="wage", predictors=["education"]))
    result["provenance"]["dataset_hash_pandas_version"] = "0.0.0"
    target = workspace.result_path / f"{result['id']}.json"
    target.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="Pandas version differs"):
        exec(compile(workspace.python_script(result["id"]), "analysis.py", "exec"), {})


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".xlsx", ".dta"])
def test_supported_file_roundtrip(tmp_path, suffix):
    frame = example_frame().drop(columns="region")
    path = tmp_path / f"data{suffix}"
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix == ".xlsx":
        frame.to_excel(path, index=False)
    elif suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_stata(path, write_index=False, variable_labels={"wage": "Hourly wage"})
    loaded = read(path)
    pd.testing.assert_frame_equal(loaded, frame, check_dtype=False)
    if suffix == ".dta":
        assert loaded.attrs["metadata"]["column_labels"]["wage"] == "Hourly wage"
