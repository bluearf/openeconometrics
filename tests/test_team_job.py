"""Bounded job transport and real fresh-process execution tests."""
import base64
import copy
import io
import json
from unittest.mock import Mock
from uuid import uuid4

import pytest

from openecon import team_job as job

INPUT_URL = "https://storage.googleapis.com/openecon-runs/input.json?X-Goog-Signature=test&X-Goog-Expires=900"
FILE_URL = INPUT_URL + "&generation=1234"
UPLOAD = {"url": "https://storage.googleapis.com/openecon-runs", "fields": {
    "key": "runs/random/result.json", "policy": "signed-policy", "x-goog-signature": "signed",
    "Content-Type": "application/json"}}


def manifest(**changes):
    return {"execution_id": str(uuid4()), "code": "1 + 2", "timeout_seconds": 10,
            "files": [], "output_upload": copy.deepcopy(UPLOAD), **changes}


@pytest.mark.parametrize("name", ["../secret", "/etc/passwd", "foo/bar", "foo\\bar", ".hidden",
                                  ".", "..", "bad\nname", "x:y", " a.csv", "", None])
def test_rejects_unsafe_filenames(name):
    with pytest.raises(job.JobInputError):
        job.safe_filename(name)


def test_supports_simple_international_names():
    assert job.safe_filename("Ücret ve eğitim.csv") == "Ücret ve eğitim.csv"


@pytest.mark.parametrize("url", [
    "http://storage.googleapis.com/bucket/key?X-Goog-Signature=x&X-Goog-Expires=900",
    "https://evil.example/key?X-Goog-Signature=x&X-Goog-Expires=900",
    "https://storage.googleapis.com.evil.example/bucket/key?X-Goog-Signature=x&X-Goog-Expires=900",
    "https://storage.googleapis.com@evil.example/bucket/key?X-Goog-Signature=x&X-Goog-Expires=900",
    "https://storage.googleapis.com/bucket/key", INPUT_URL + "#fragment",
    INPUT_URL.replace("900", "9999"), INPUT_URL.replace("900", "bad"),
    INPUT_URL + "&X-Goog-Signature=other", None,
])
def test_storage_capabilities_reject_other_hosts_and_unbounded_expiry(url):
    with pytest.raises(job.JobInputError):
        job.storage_url(url)


def test_manifest_rejects_unpinned_duplicate_or_excessive_files():
    for files in [
        [{"name": "x.csv", "download_url": INPUT_URL}],
        [{"name": "x.csv", "download_url": FILE_URL}] * 2,
        [{"name": f"x{i}.csv", "download_url": FILE_URL} for i in range(21)],
    ]:
        with pytest.raises(job.JobInputError):
            job.validate_manifest(manifest(files=files))
    assert job.validate_manifest(manifest(files=[{"name": "x.csv", "download_url": FILE_URL}]))


@pytest.mark.parametrize("changes", [{"execution_id": "../x"}, {"code": ""}, {"code": "x" * 64001},
    {"timeout_seconds": True}, {"timeout_seconds": float("nan")}, {"timeout_seconds": 121},
    {"output_upload": {"url": "https://evil.example", "fields": UPLOAD["fields"]}}])
def test_invalid_manifest_is_rejected(changes):
    with pytest.raises(job.JobInputError):
        job.validate_manifest(manifest(**changes))


class Response(io.BytesIO):
    def __init__(self, body, headers=None, status=200):
        super().__init__(body)
        self.headers = headers or {}
        self.status = status


def test_download_enforces_actual_bytes_and_declared_size(monkeypatch):
    for headers in ({}, {"Content-Length": "100"}):
        opener = Mock()
        opener.open.return_value = Response(b"x" * 11, headers)
        monkeypatch.setattr(job, "build_opener", lambda *args: opener)
        with pytest.raises(job.JobInputError):
            job.download(INPUT_URL, 10)
        assert opener.open.call_args.kwargs["timeout"] == 15


def test_redirects_are_never_followed():
    with pytest.raises(job.JobInputError):
        job._NoRedirect().redirect_request(None, None, 302, "", {}, "http://metadata.google.internal/")


def test_collect_generated_excludes_inputs_and_bounds_total(tmp_path, monkeypatch):
    (tmp_path / "input.csv").write_bytes(b"input")
    (tmp_path / "out.txt").write_bytes(b"result")
    result = job.collect_generated(tmp_path, {"input.csv"})
    assert result == [{"name": "out.txt", "size": 6, "content_base64": base64.b64encode(b"result").decode()}]
    monkeypatch.setattr(job, "MAX_GENERATED_TOTAL", 5)
    with pytest.raises(job.JobInputError, match="16 MiB"):
        job.collect_generated(tmp_path, {"input.csv"})


def test_symlinks_and_special_files_cannot_be_exported(tmp_path):
    import os
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("outside")
    (tmp_path / "leak.txt").symlink_to(secret)
    with pytest.raises(job.JobInputError, match="symbolic"):
        job.collect_generated(tmp_path, set())
    (tmp_path / "leak.txt").unlink()
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(job.JobInputError, match="regular"):
        job.collect_generated(tmp_path, set())


def test_generated_file_count_is_limited(tmp_path):
    for index in range(21):
        (tmp_path / f"f{index}.txt").write_text("x")
    with pytest.raises(job.JobInputError, match="20 generated"):
        job.collect_generated(tmp_path, set())


def test_real_run_has_fresh_namespace_and_persistable_file(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "download", lambda url, limit: b"x,y\n1,2\n")
    request = manifest(code='from pathlib import Path\nPath("result.txt").write_text("saved")\nanswer = 42\nanswer',
                       files=[{"name": "input.csv", "download_url": FILE_URL}])
    result = job.run_manifest(request, tmp_path / "first")
    assert result["execution_id"] == request["execution_id"]
    assert result["record"]["status"] == "ok"
    item = result["record"]["outputs"][-1]
    assert item["type"] == "text" and item["data"] == "42"
    assert "42" in item["latex"]
    assert result["record"]["variables"] == []
    assert result["record"]["state_reset"] is True
    assert result["generated_files"][0]["name"] == "result.txt"
    assert base64.b64decode(result["generated_files"][0]["content_base64"]) == b"saved"
    second = job.run_manifest(manifest(code="answer"), tmp_path / "second")
    assert second["record"]["status"] == "error"
    assert second["record"]["error"]["type"] == "NameError"


def test_multipart_upload_uses_fixed_policy_fields_and_bounded_json(monkeypatch):
    opener = Mock()
    opener.open.return_value = Response(b"", status=204)
    monkeypatch.setattr(job, "build_opener", lambda *args: opener)
    result = {"record": {"outputs": []}, "generated_files": []}
    job.upload_result(UPLOAD, result)
    request = opener.open.call_args.args[0]
    assert request.full_url == UPLOAD["url"]
    assert request.method == "POST"
    assert b'name="policy"' in request.data
    assert b'name="file"; filename="result.json"' in request.data
    assert json.dumps(result).encode() in request.data
    monkeypatch.setattr(job, "MAX_RESULT_BYTES", 1)
    with pytest.raises(job.JobInputError, match="output limit"):
        job.upload_result(UPLOAD, result)


def test_job_main_does_not_leak_capabilities_on_failure(monkeypatch, capsys):
    monkeypatch.setenv("OPENECON_RUN_INPUT_URL", INPUT_URL)
    monkeypatch.setattr(job, "download", Mock(side_effect=RuntimeError(INPUT_URL)))
    assert job.main() == 1
    assert INPUT_URL not in capsys.readouterr().out
