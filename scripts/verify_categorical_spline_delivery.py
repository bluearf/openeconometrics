"""Prepare/capture owned native scripts; frozen smoke is distinct from UI Run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/categorical-spline-eight"
spec = importlib.util.spec_from_file_location(
    "categorical_acceptance", ROOT / "benchmarks/categorical_spline_eight.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Client:
    def __init__(self, origin):
        self.origin = origin

    def call(self, path, body=None, token=None, timeout=60):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-OpenEcon-Token"] = token
        request = Request(
            self.origin + path,
            headers=headers,
            data=None if body is None else json.dumps(body).encode(),
        )
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)


def verify_portable(input_dir, output_dir):
    """Reopen complete results in a fresh worker and refuse estimator fitting."""
    from pathlib import Path
    import json
    import openecon as oe
    import pandas as pd
    from openecon.econometrics.categorical import frequency_regression as regression
    from openecon.econometrics.categorical import frequency_rotation
    from openecon.econometrics.categorical import spline_regression
    from openecon.econometrics.categorical import spline_pca

    directory = Path(output_dir)
    data = pd.read_csv(Path(input_dir) / "people.csv")
    queries = data.iloc[[0, 9, 18, 27, 36, 54, 72, 99]].reset_index(drop=True)
    restored = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "receipt.json":
            continue
        text = path.read_text(encoding="utf-8")
        value = oe.restore_summary(text)
        assert oe.summary_state(value) == text, path.name
        assert value.to_latex() == path.with_suffix(".tex").read_text(encoding="utf-8"), path.name
        restored[path.stem] = value
    assert len(restored) == 16

    def forbidden(*args, **kwargs):
        raise AssertionError("Portable replay must not fit learned maps or optimize rotations")

    guards = [(regression, "_fit"), (spline_regression, "_fit_coefficients"),
              (frequency_rotation, "_varimax")]
    if hasattr(spline_pca, "_fit"):
        guards.append((spline_pca, "_fit"))
    originals = [(module, name, getattr(module, name)) for module, name in guards]
    try:
        for module, name, _ in originals:
            setattr(module, name, forbidden)
        for stem in ("weighted-varimax", "weighted-promax"):
            projected = oe.catpca_rotated_predict_fweight(restored[stem], queries)
            assert oe.summary_state(projected) == (directory/(stem+"-projection.json")).read_text(encoding="utf-8")
        for stem in ("frequency-nominal-bootstrap", "frequency-ordinal-bootstrap"):
            verified = oe.catreg_fweight_bootstrap_restore(restored[stem])
            assert oe.summary_state(verified) == (directory/(stem+".json")).read_text(encoding="utf-8")
        for stem in ("spline-nominal-catreg", "spline-monotone-catreg"):
            predicted = oe.catreg_spline_predict(restored[stem], queries)
            assert oe.summary_state(predicted) == (directory/(stem+"-prediction.json")).read_text(encoding="utf-8")
        for stem in ("spline-nominal-catpca", "spline-monotone-catpca"):
            projected = oe.catpca_spline_predict(restored[stem], queries)
            assert oe.summary_state(projected) == (directory/(stem+"-projection.json")).read_text(encoding="utf-8")
    finally:
        for module, name, original in originals:
            setattr(module, name, original)
    print("CATEGORICAL_SPLINE_PORTABLE_REPLAY " + json.dumps({
        "full_results": len(restored), "saved_query_replays": 6,
        "complete_bootstrap_semantic_replays": 2,
        "full_json_and_latex_equal": True, "refit_for_query_replay": False,
        "rotation_reoptimized": False,
        "saved_counts_reconstructed_only": True}))


def prepare(profile, client, mode):
    master = client.call("/api/desktop/session")["token"]
    assert client.call("/api/desktop/local-projects", token=master)["projects"] == []
    project = client.call(
        "/api/desktop/local-projects",
        {
            "name": "Categorical spline and weighted options eight-stage acceptance",
            "description": "Owned synthetic fixed-knot spline, weighted rotation and full-refit frequency bootstrap stages",
        },
        master,
    )
    project_path = profile / "projects" / project["id"]
    input_dir, output_dir = (
        project_path / "datasets/categorical-proof",
        project_path / "results/categorical-proof",
    )
    shutil.copytree(ROOT / "tests/fixtures/categorical_spline", input_dir)
    prefix = f"/api/desktop/projects/{project['id']}/workspace"
    token = client.call(prefix + "/session")["token"]
    scripts = []
    for i, issue in enumerate(bench.ISSUES):
        code = bench.native_script(i, input_dir, output_dir)
        script = client.call(
            prefix + "/console/scripts",
            {"name": issue + " categorical acceptance.py", "code": code},
            token,
        )
        assert client.call(prefix + "/console/scripts/" + script["id"], token=token)["code"] == code
        scripts.append({"id": script["id"], "name": script["name"], "code": code, "issue": issue})
    session = dict(
        project=project,
        prefix=prefix,
        scripts=scripts,
        output_dir=str(output_dir),
        profile=str(profile),
    )
    (BASE / (mode + "-session.json")).write_text(json.dumps(session, indent=2) + "\n")
    return session


def compare_outputs(directory):
    source = BASE / "source"
    source_receipt = json.loads((source/"receipt.json").read_text())
    assert source_receipt["passed"] == 8 and source_receipt["stages"] == list(bench.ISSUES)
    assert source_receipt["input_sha256"] == hashlib.sha256(
        (ROOT/"tests/fixtures/categorical_spline/people.csv").read_bytes()).hexdigest()
    expected = sorted(p.name for p in source.iterdir() if p.name != "receipt.json")
    assert sorted(p.name for p in directory.iterdir()) == expected
    for name in expected:
        assert (directory / name).read_bytes() == (source / name).read_bytes(), (
            f"Full source/runtime payload differs: {name}"
        )
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in expected}


def capture(client, mode, session, *, restart=False):
    prefix = session["prefix"]
    token = client.call(prefix + "/session")["token"]
    state = client.call(prefix + "/console", token=token)
    assert not state.get("running")
    runs = [
        run
        for run in state["history"]
        if "CATEGORICAL_SPLINE_NATIVE_ACCEPTED MARKET-" in run.get("stdout", "")
    ]
    assert len(runs) == 8
    by_issue = {}
    for script in session["scripts"]:
        matched = [
            r
            for r in runs
            if "CATEGORICAL_SPLINE_NATIVE_ACCEPTED " + script["issue"] in r.get("stdout", "")
        ]
        assert len(matched) == 1
        run = matched[0]
        assert run["status"] == "ok" and run["code"] == script["code"]
        assert (
            client.call(prefix + "/console/scripts/" + script["id"], token=token)["code"]
            == script["code"]
        )
        by_issue[script["issue"]] = dict(
            run_id=run["id"],
            status=run["status"],
            duration_ms=run["duration_ms"],
            outputs=len(run["outputs"]),
            code_sha256=hashlib.sha256(script["code"].encode()).hexdigest(),
        )
    frozen_files = compare_outputs(Path(session["output_dir"]))
    saved = BASE / (mode + "-history.json")
    if restart:
        assert runs == json.loads(saved.read_text()), "History/output/events changed after relaunch"
    else:
        saved.write_text(json.dumps(runs, indent=2) + "\n")
    receipt = dict(
        passed=8,
        stages=by_issue,
        complete_files_equal_source=frozen_files,
        named_scripts_read_back=True,
        frozen_assertions_passed=True,
        independent_fixture_oracles=True,
        actual_ui_run=(mode == "native"),
        restart_history_output_events_equal=restart,
        human_profile_accessed=False,
        primary_app_modified=False,
        licensed_vendor_run=False,
        public_release=False,
    )
    (BASE / (mode + ("-restart" if restart else "-capture") + ".json")).write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"mode": mode, "passed": 8, "files_exact": len(frozen_files), "restart": restart}
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("frozen", "prepare-native", "capture-native", "restart-native")
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path.home() / "Library/Application Support/org.openecon.qa.categorical-spline-20261009",
    )
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if args.mode == "frozen":
        profile = BASE / "frozen-profile"
        assert not profile.exists(), "Do not overwrite earlier frozen acceptance"
        executable = ROOT / "desktop/runtime/openecon-runtime/openecon-runtime"
        process = subprocess.Popen(
            [str(executable), "--data-root", str(profile)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=(BASE / "frozen-stderr.log").open("w"),
            text=True,
        )
        try:
            ready = json.loads(process.stdout.readline())
            assert ready["type"] == "ready"
            client = Client(ready["url"])
            session = prepare(profile, client, "frozen")
            token = client.call(session["prefix"] + "/session")["token"]
            for script in session["scripts"]:
                result = client.call(
                    session["prefix"] + "/console/execute",
                    {"script_id": script["id"], "code": script["code"], "timeout_seconds": 120},
                    token,
                    timeout=150,
                )
                if result.get("status") != "ok":
                    (BASE / "frozen-failed-response.json").write_text(json.dumps(result, indent=2))
                    raise AssertionError("Frozen stage failed; retained response")
            capture(client, "frozen", session)
        finally:
            process.stdin.write('{"type":"shutdown"}\n')
            process.stdin.flush()
            process.wait(timeout=30)
        assert process.returncode == 0
        process = subprocess.Popen(
            [str(executable), "--data-root", str(profile)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=(BASE/"frozen-restart-stderr.log").open("w"), text=True,
        )
        try:
            ready = json.loads(process.stdout.readline())
            assert ready["type"] == "ready"
            client = Client(ready["url"])
            capture(client, "frozen", session, restart=True)
            token = client.call(session["prefix"]+"/session")["token"]
            input_dir = profile/"projects"/session["project"]["id"]/"datasets/categorical-proof"
            code = inspect.getsource(verify_portable)+f"\nverify_portable({str(input_dir)!r}, {session['output_dir']!r})\n"
            replay = client.call(session["prefix"]+"/console/execute",
                                 {"code": code, "timeout_seconds": 120}, token, timeout=150)
            (BASE/"frozen-portable-replay.json").write_text(json.dumps(replay, indent=2)+"\n")
            assert replay.get("status") == "ok", "Portable replay failed; retained complete response"
            compare_outputs(Path(session["output_dir"]))
            print(json.dumps({"mode": "frozen-portable-replay", "status": replay["status"]}))
        finally:
            process.stdin.write('{"type":"shutdown"}\n')
            process.stdin.flush()
            process.wait(timeout=30)
        assert process.returncode == 0
    else:
        profile = args.profile
        ready = json.loads((profile / ".runtime-port.json").read_text())
        client = Client(f"http://127.0.0.1:{ready['port']}")
        if args.mode == "prepare-native":
            prepare(profile, client, "native")
            print(json.dumps({"mode": args.mode, "named_scripts_saved": 8}))
        else:
            session = json.loads((BASE / "native-session.json").read_text())
            assert session["profile"] == str(profile)
            capture(client, "native", session, restart=args.mode == "restart-native")


if __name__ == "__main__":
    main()
