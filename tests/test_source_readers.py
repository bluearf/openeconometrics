"""Physical parser admission, lossless conversion and independent workspace readback."""

from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import tempfile
import os

import pandas as pd
import pyreadstat
import pytest

import openecon as oe
import openecon.data as data
from openecon.data import DataError
from openecon.dataset import Dataset
from openecon.source_readers import Parser, preflight_csv, preflight_parquet
from openecon.workspace import Workspace


@pytest.fixture
def dense_limit(monkeypatch):
    monkeypatch.setattr(data, "MAX_ROWS", 12)


@pytest.fixture(autouse=True)
def owned_scratch(tmp_path, monkeypatch):
    root = tmp_path / "owned-reader-scratch"
    root.mkdir()
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(root))


def scratch():
    return set(
        Path(os.environ.get("OPENECON_SCRATCH_DIRECTORY", tempfile.gettempdir())).glob(
            ".openecon-reader-*"
        )
    )


def test_csv_quote_boundaries_bom_multiline_and_fat_cell_are_admitted_before_parser(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text('\ufeff"label","value"\r\n"a\n""b",1\r\n"c",2', encoding="utf8")
    assert preflight_csv(path) == 2
    result = oe.read(path)
    assert result["label"].tolist() == ['a\n"b', "c"]
    path.write_bytes(b"x\n" + b"a" * (1024 * 1024 + 1))
    before = scratch()
    with pytest.raises(DataError, match="allocation") as caught:
        oe.read(path)
    assert caught.value.code == "READER_LIMIT"
    assert scratch() == before
    assert path.stat().st_size > 1024 * 1024


def test_bad_footer_refused_before_arrow_and_compressed_group_refused_before_values(
    tmp_path, monkeypatch
):
    fake = tmp_path / "bad.parquet"
    fake.write_bytes(b"PAR1" + b"x" * 24 + (2**31).to_bytes(4, "little") + b"PAR1")
    with pytest.raises(DataError):
        preflight_parquet(fake)
    frame = pd.DataFrame({"x": [0] * 10000})
    path = tmp_path / "compressed.parquet"
    frame.to_parquet(path, compression="zstd", row_group_size=10000, index=False)
    import openecon.source_readers as readers

    monkeypatch.setattr(readers, "MAX_ROW_GROUP_BYTES", 1)
    with pytest.raises(DataError, match="row group"):
        oe.scan(path)


@pytest.mark.parametrize("suffix", ["xlsx", "dta"])
def test_bounded_conversion_dates_missing_labels_counts_index_and_cleanup(
    tmp_path, dense_limit, suffix
):
    frame = pd.DataFrame(
        {
            "x": [float(i) for i in range(31)],
            "label": ["Türkiye"] * 31,
            "when": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(31)],
        }
    )
    path = tmp_path / ("input." + suffix)
    if suffix == "xlsx":
        frame.to_excel(path, index=False)
    else:
        frame.loc[2, "x"] = None
        frame["tagged"] = [float(i) for i in range(31)]
        frame["tagged"] = frame["tagged"].astype("object")
        frame.loc[5, "tagged"] = "a"
        frame.loc[22, "tagged"] = "z"
        pyreadstat.write_dta(
            frame,
            str(path),
            column_labels={"x": "Original x"},
            variable_value_labels={"tagged": {"a": "Not available", "z": "Suppressed", 1: "One"}},
            missing_user_values={"tagged": ["a", "z"]},
        )
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    before = scratch()
    source = oe.read(path)
    assert isinstance(source, Dataset) and source.row_count == 31
    directory = source._path.parent
    receipt = source.provenance["conversion"]
    assert receipt["source_sha256"] == fingerprint and receipt["rows"] == 31
    assert receipt["converted_footer_verified"] and receipt["source_unchanged"]
    parts = list(source.iter_batches(batch_rows=7))
    actual = pd.concat(parts)
    assert actual.index.tolist() == list(range(31))
    assert actual["label"].tolist() == frame["label"].tolist()
    assert pd.to_datetime(actual["when"]).tolist() == frame["when"].tolist()
    assert source.reader_receipt["parser_stopped"] and source.reader_receipt["scratch_removed"]
    if suffix == "dta":
        assert source.metadata["column_labels"]["x"] == "Original x"
        assert source.metadata["value_labels"]["tagged"]["a"] == "Not available"
        assert pd.isna(actual.loc[5, "tagged"]) and pd.isna(actual.loc[22, "tagged"])
        codes = list(source.iter_missing_codes("tagged", batch_rows=7))
        merged = {}
        for block in codes:
            for code, positions in block.items():
                merged.setdefault(code, []).extend(positions)
        assert merged == {"a": [5], "z": [22]}
        assert not any(name.startswith("__openecon_missing") for name in source.columns)
    source.close()
    assert not directory.exists() and scratch() == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fingerprint


@pytest.mark.parametrize("suffix", ["xlsx", "dta"])
def test_converted_workspace_snapshot_is_persistent_and_original_retained(
    tmp_path, dense_limit, suffix
):
    frame = pd.DataFrame({"x": range(31), "y": [1 + 0.2 * i for i in range(31)]})
    path = tmp_path / ("input." + suffix)
    if suffix == "xlsx":
        frame.to_excel(path, index=False)
    else:
        pyreadstat.write_dta(frame, str(path), column_labels={"y": "Outcome"})
    before = scratch()
    workspace = Workspace(tmp_path / "project")
    profile = workspace.import_file(path)
    assert profile["storage"]["format"] == "parquet" and profile["row_count"] == 31
    assert (
        workspace.data_path / f"{profile['id']}.original.{suffix}"
    ).read_bytes() == path.read_bytes()
    restored = Workspace(workspace.path).load_frame(profile["id"])
    assert restored.row_count == 31
    assert pd.concat(list(restored.iter_batches()))["x"].tolist() == list(range(31))
    assert scratch() == before


def test_parser_timeout_and_early_close_clean_owned_scratch(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("x\n1\n2\n")
    before = scratch()
    parser = Parser(path, columns=["x"], timeout=0.001)
    try:
        with pytest.raises(DataError, match="time"):
            parser.start()
    finally:
        parser.close()
    assert not parser.process.is_alive() and scratch() == before
    source = oe.scan(path)
    iterator = source.iter_batches(batch_rows=1)
    assert next(iterator)["x"].tolist() == [1]
    iterator.close()
    assert source.reader_receipt["parser_stopped"] and source.reader_receipt["scratch_removed"]
    assert scratch() == before
