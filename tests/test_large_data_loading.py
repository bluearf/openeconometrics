"""Automatic lazy input, persistent snapshots and bounded large-file profiles."""
import hashlib
import json
import math
import zipfile

import pandas as pd
import pytest

import openecon as oe
import openecon.data as data
from openecon.console import load_dataset
from openecon.data import DataError, profile_dataset
from openecon.dataset import Dataset
from openecon.workspace import Workspace


@pytest.fixture
def lower_dense_thresholds(monkeypatch):
    monkeypatch.setattr(data, "MAX_ROWS", 12)
    monkeypatch.setattr(data, "MAX_FILE_BYTES", 2048)
    monkeypatch.setattr(data, "MAX_COLUMNS", 4)


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_large_read_returns_dataset_and_projects_batches(tmp_path, lower_dense_thresholds, suffix):
    frame = pd.DataFrame({"x": range(31), "y": [2. + 3. * i for i in range(31)], "unused": "payload"})
    path = tmp_path / f"data.{suffix}"
    getattr(frame, "to_" + suffix)(path, index=False)
    source = oe.read(path)
    assert isinstance(source, Dataset)
    blocks = list(source.iter_batches(columns=["x"], batch_rows=7))
    assert [len(block) for block in blocks] == [7, 7, 7, 7, 3]
    assert all(list(block.columns) == ["x"] for block in blocks)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source.file_hash()


def test_small_read_remains_dataframe_and_large_parquet_directory_is_lazy(tmp_path, lower_dense_thresholds):
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(path, index=False)
    assert isinstance(oe.read(path), pd.DataFrame)
    directory = tmp_path / "parts"
    directory.mkdir()
    for index in range(2):
        pd.DataFrame({"x": [index]}).to_parquet(directory / f"{index}.parquet", index=False)
    assert isinstance(oe.read(directory), Dataset)
    assert oe.read(directory).row_count == 2


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_large_workspace_snapshot_reopens_console_and_replays_without_collection(tmp_path, lower_dense_thresholds, suffix, capsys):
    frame = pd.DataFrame({"x": range(41), "y": [2. + .7 * i + math.sin(i) for i in range(41)]})
    path = tmp_path / f"input.{suffix}"
    getattr(frame, "to_" + suffix)(path, index=False)
    store = Workspace(tmp_path / "project")
    profile = store.import_file(path, name="Large sample")
    assert profile["name"] == "Large sample" and profile["row_count"] == 41
    assert profile["storage"]["kind"] == "dataset"
    assert not (store.data_path / f"{profile['id']}.parquet").exists()
    restored = load_dataset(profile["id"], workspace=store.path)
    assert isinstance(restored, Dataset) and restored.row_count == 41
    pd.testing.assert_frame_equal(restored.head(4), oe.read(path).head(4), check_dtype=False)
    result = store.run_analysis(profile["id"], oe.ModelSpec(outcome="y", predictors=["x"], covariance="HC3"))
    assert result["nobs"] == 41 and result["provenance"]["streaming"]["row_limit"] is None
    exec(compile(store.python_script(result["id"]), "replay.py", "exec"), {})
    assert json.loads(capsys.readouterr().out)["nobs"] == 41
    destination = tmp_path / "export.zip"
    assert store.export_bundle(result["id"], destination=destination) == destination
    with zipfile.ZipFile(destination) as archive:
        assert f"dataset.{suffix}" in archive.namelist()
    snapshot = store.data_path / profile["storage"]["file"]
    with snapshot.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(DataError) as caught:
        store.load_frame(profile["id"])
    assert caught.value.code in {"DATA_INTEGRITY", "INVALID_SOURCE"}


def test_bounded_profile_counts_and_moments_do_not_claim_capped_cardinality(tmp_path):
    frame = pd.DataFrame({"x": [float(i) for i in range(999)] + [None],
                          "category": ["A", "B"] * 500})
    path = tmp_path / "values.parquet"
    frame.to_parquet(path, index=False)
    profile = profile_dataset(oe.scan(path), name="Data")
    columns = {item["name"]: item for item in profile["columns"]}
    assert profile["row_count"] == 1000 and len(profile["preview"]) <= 50
    assert columns["x"]["missing"] == 1
    assert columns["x"]["mean"] == pytest.approx(frame.x.mean())
    assert columns["x"]["std"] == pytest.approx(frame.x.std())
    assert columns["x"]["unique"] is None and not columns["x"]["unique_exact"]
    assert columns["x"]["unique_lower_bound"] == 257
    assert columns["category"]["unique"] == 2 and columns["category"]["unique_exact"]


def test_large_parquet_attributes_survive_snapshot_and_metadata_tampering_is_detected(tmp_path, lower_dense_thresholds):
    frame = pd.DataFrame({"x": range(21)})
    frame.attrs["metadata"] = {"column_labels": {"x": "Original label"}, "synthetic": True}
    path = tmp_path / "data.parquet"
    frame.to_parquet(path, index=False)
    store = Workspace(tmp_path / "project")
    profile = store.import_file(path)
    assert profile["columns"][0]["label"] == "Original label"
    assert store.load_frame(profile["id"]).head().attrs["metadata"] == frame.attrs["metadata"]
    profile["metadata"]["synthetic"] = False
    (store.data_path / f"{profile['id']}.json").write_text(json.dumps(profile))
    with pytest.raises(DataError) as caught:
        store.load_frame(profile["id"])
    assert caught.value.code == "DATA_INTEGRITY"


@pytest.mark.parametrize("storage", [{"kind": "dataset", "format": "csv", "file": "../escape.csv"}, [], {"kind": []}])
def test_malformed_dataset_descriptor_is_rejected(tmp_path, lower_dense_thresholds, storage):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": range(20)}).to_csv(path, index=False)
    store = Workspace(tmp_path / "project")
    profile = store.import_file(path)
    record = store.data_path / f"{profile['id']}.json"
    profile["storage"] = storage
    record.write_text(json.dumps(profile))
    with pytest.raises(DataError) as caught:
        store.load_frame(profile["id"])
    assert caught.value.code == "CORRUPT_RECORD"


def test_wide_csv_routes_without_dense_parse_and_duplicate_header_still_fails(tmp_path, lower_dense_thresholds, monkeypatch):
    path = tmp_path / "wide.csv"
    pd.DataFrame({f"x{i}": [i] for i in range(6)}).to_csv(path, index=False)
    def fail(*args, **kwargs):
        raise AssertionError("Automatic wide routing must precede dense CSV parsing")
    monkeypatch.setattr(pd, "read_csv", fail)
    assert isinstance(oe.read(path), Dataset)
    path.write_text("x,x\n1,2\n")
    with pytest.raises(DataError, match="unique"):
        oe.read(path)


def test_local_upload_has_no_csv_parquet_transfer_cap_and_streams_bundle_file(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import openecon.server as server
    monkeypatch.setattr(data, "MAX_FILE_BYTES", 64)
    contents = pd.DataFrame({"x": range(21), "y": [3. + i + math.sin(i) for i in range(21)]}).to_csv(index=False).encode()
    assert len(contents) > 64
    with TestClient(server.create_app(tmp_path / "project")) as client:
        session = client.get("/api/session").json()
        client.headers["X-OpenEcon-Token"] = session["token"]
        assert session["upload_limit_bytes"] is None
        response = client.post("/api/datasets/upload", files={"file": ("large.csv", contents)})
        assert response.status_code == 200, response.text
        profile = response.json()
        assert profile["storage"]["kind"] == "dataset" and profile["row_count"] == 21
        result = client.post("/api/analyses", json={"dataset_id": profile["id"],
            "spec": {"outcome": "y", "predictors": ["x"]}}).json()
        bundle = client.get(f"/api/results/{result['id']}/bundle")
        assert bundle.status_code == 200
        assert bundle.headers["content-type"] == "application/zip"
        refused = client.post("/api/datasets/upload", files={"file": ("large.dta", b"x" * 100)})
        assert refused.status_code == 422 and refused.json()["detail"]["code"] == "INVALID_SOURCE"
