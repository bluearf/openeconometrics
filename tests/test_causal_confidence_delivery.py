"""Lazy public delivery and persisted contracts for the confidence, multiarm and individual-effect extensions."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

NAMES = ('randomization_confidence_set', 'paired_randomization_confidence_set', 'cluster_randomization_confidence_set', 'bernoulli_neyman_ate', 'multiarm_neyman_ate', 'stratified_multiarm_neyman_ate', 'treatment_effect_cdf_bounds', 'treatment_effect_quantile_bounds')


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("positions", [False, True, 2, 3, 4, 5]),
        ("unit_labels", [1.0, 2, 3, 4, 5, 6]),
    ],
)
def test_metadata_numeric_aliases_refused_on_save_and_rehashed_load(field, replacement):
    import pandas as pd
    import openecon as oe
    from openecon.analysis_contracts import AnalysisError
    from openecon.econometrics.causal_design.common import digest

    data = pd.DataFrame(
        {"y": [1.0, 3.0, 2.0, 6.0, 4.0, 9.0], "arm": [0, 0, 1, 1, 2, 2]},
        index=pd.Index([1, 2, 3, 4, 5, 6], dtype=object),
    )
    result = oe.multiarm_neyman_ate(data, "y", "arm", arms=[0, 1, 2], design="complete_randomized")
    artifact = oe.causal_design_save(result)
    assert oe.causal_design_save(oe.causal_design_load(artifact)) == artifact
    original = deepcopy(result.attrs["state"]["sample"][field])
    # Equal mathematical numbers cannot replace the original JSON types.
    assert replacement == original
    result.attrs[field] = replacement
    assert result.attrs["state"]["sample"][field] == original
    with pytest.raises(AnalysisError, match="intact causal design result"):
        oe.causal_design_save(result)
    changed = deepcopy(artifact)
    changed["payload"]["attrs"][field] = replacement
    changed["sha256"] = digest(changed["payload"])
    with pytest.raises(AnalysisError, match="artifact/state/table checksum or schema mismatch"):
        oe.causal_design_load(changed)


def test_bounded_integer_identities_do_not_pass_through_float_finiteness():
    import pandas as pd
    import openecon as oe

    base = 2**1100
    labels = [base, base + 1, base + 2]
    data = pd.DataFrame(
        {"y": [0., 1., 2., 4., 3., 6.], "arm": pd.Series([label for label in labels for _ in range(2)], dtype=object)},
    )
    data.index = pd.Index([base + i for i in range(len(data))], dtype=object)
    result = oe.multiarm_neyman_ate(data, "y", "arm", arms=labels, design="complete_randomized")
    assert result["arms"]["identifier"].tolist() == labels
    assert result.attrs["unit_labels"] == data.index.tolist()
    assert result.attrs["sample_hash_encoding"] == "typed-canonical-rows-v1"
    artifact = oe.causal_design_save(result)
    restored = oe.causal_design_load(artifact)
    assert oe.causal_design_save(restored) == artifact
    assert restored["arms"]["identifier"].tolist() == labels
    changed = data.copy()
    changed.loc[changed["arm"] == labels[0], "arm"] = str(labels[0])
    typed = oe.multiarm_neyman_ate(
        changed, "y", "arm", arms=[str(labels[0]), *labels[1:]], design="complete_randomized",
    )
    assert typed.attrs["sample_sha256"] != result.attrs["sample_sha256"]
    assert typed["arms"]["identifier"].iloc[0] == str(labels[0])


@pytest.mark.parametrize("label", ["x" * 257, 2**851])
def test_effect_bound_label_admission_precedes_escaped_copy(label, monkeypatch):
    from openecon.analysis_contracts import AnalysisError
    from openecon.econometrics.causal_design import effect_distribution as module

    monkeypatch.setattr(module.m, "_label", lambda *_: pytest.fail("Escaping preceded label admission"))
    with pytest.raises(AnalysisError, match="escaped-label domain"):
        module._label_size(label)


def test_causal_targets_manifests_are_torch_free():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys, openecon\n"
        "from openecon.econometrics.causal_design import EXPORTS\n"
        f"assert all(name in EXPORTS for name in {NAMES!r})\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'scipy' not in sys.modules\n"
        "assert 'statsmodels' not in sys.modules\n"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=root, text=True,
                         capture_output=True, env=dict(os.environ, PYTHONPATH=str(root / "src")))
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("name", NAMES)
def test_public_causal_target_resolves_to_its_registered_source(name):
    import openecon as oe
    from openecon.econometrics.causal_design import EXPORTS

    function = getattr(oe, name)
    assert callable(function)
    assert function.__name__ == name
    assert function.__module__ == EXPORTS[name].split(":")[0]


def test_runtime_verifier_scopes_new_batch_without_changing_previous_batch():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import verify_causal_targets_runtime as old\n"
        "before = old.EXAMPLE, old.MODULES, old.NAMES, old.MARKER\n"
        "import verify_causal_confidence_runtime as new\n"
        "assert (old.EXAMPLE, old.MODULES, old.NAMES, old.MARKER) == before\n"
        f"assert new.NAMES == {NAMES!r}\n"
        "text = new.header('/synthetic-artifact-path')\n"
        "assert 'confidence_sets' in text and 'effect_distribution' in text\n"
        "try:\n"
        "    with new.configured_engine():\n"
        "        assert old.NAMES == new.NAMES\n"
        "        raise ValueError('scope unwind')\n"
        "except ValueError:\n"
        "    pass\n"
        "assert (old.EXAMPLE, old.MODULES, old.NAMES, old.MARKER) == before\n"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=root, text=True,
                         capture_output=True, env=dict(os.environ, PYTHONPATH=str(root / "scripts")))
    assert run.returncode == 0, run.stderr


def test_full_frozen_receipt_rejects_previous_batch_or_incomplete_tables():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import verify_causal_confidence_runtime as v\n"
        "cases = [dict(status='ok', stdout='CAUSAL_TARGETS_EIGHT_OK {}', outputs=[]),\n"
        "         dict(status='ok', stdout=v.MARKER + '{}', outputs=[])]\n"
        "for case in cases:\n"
        "    try:\n"
        "        v.proof_from_execution(case)\n"
        "    except RuntimeError:\n"
        "        continue\n"
        "    raise AssertionError('incomplete or stale proof accepted')\n"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=root, text=True,
                         capture_output=True, env=dict(os.environ, PYTHONPATH=str(root / "scripts")))
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("mode", ["--seed", "--ui-observed"])
@pytest.mark.parametrize("listener_kind", ["primary_app", "owned_path_prefix"])
def test_installed_stale_port_refuses_foreign_listener_before_any_api_request(
    tmp_path, monkeypatch, mode, listener_kind,
):
    import importlib.util
    import json
    import plistlib

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "causal_confidence_installed_guard", root / "scripts/verify_causal_confidence_installed.py"
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    app, data = tmp_path / "Owned QA.app", tmp_path / "owned-profile"
    (app / "Contents").mkdir(parents=True)
    data.mkdir()
    (app / "Contents/Info.plist").write_bytes(plistlib.dumps({
        "CFBundleIdentifier": "org.openecon.qa.causalconfidence",
    }))
    (data / ".runtime-port.json").write_text(json.dumps({"port": 43123}))
    runtime = app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    foreign = (
        "/Applications/OpenEconometrics.app/Contents/Resources/runtime/openecon-runtime/openecon-runtime"
        if listener_kind == "primary_app" else str(runtime) + "-other"
    )
    calls, requests = [], []

    def fake_process(command, **options):
        calls.append(command)
        if command == ["lsof", "-t", "-nP", "-iTCP:43123", "-sTCP:LISTEN"]:
            return subprocess.CompletedProcess(command, 0, stdout="12345\n")
        assert command == ["ps", "-p", "12345", "-o", "command="]
        return subprocess.CompletedProcess(command, 0, stdout=foreign + " --port 43123\n")

    def forbidden_request(request, **options):
        requests.append(request)
        pytest.fail("A foreign runtime received an API read or mutation before its identity was checked")

    monkeypatch.setattr(verifier, "APP", app)
    monkeypatch.setattr(verifier, "DATA", data)
    monkeypatch.setattr(verifier.subprocess, "run", fake_process)
    monkeypatch.setattr(verifier, "urlopen", forbidden_request)
    monkeypatch.setattr(verifier, "source_identity", lambda *a, **k: pytest.fail("Foreign listener reached module verification"))
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(sys, "argv", ["verify_causal_confidence_installed.py", mode, "--receipt", str(receipt)])
    with pytest.raises(RuntimeError, match="does not belong to its installed runtime"):
        verifier.main()
    assert len(calls) == 2 and requests == []
    assert not receipt.exists()
    assert {path.name for path in data.iterdir()} == {".runtime-port.json"}
