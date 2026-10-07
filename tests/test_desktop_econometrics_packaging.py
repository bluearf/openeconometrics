"""Installer collection covers lazy registry entries without requiring a build."""
import importlib.util
import base64
import csv
import hashlib
import json
from pathlib import Path
import pkgutil
import sys
from types import ModuleType

import pytest

import openecon.econometrics
from openecon.econometrics import registry


@pytest.fixture
def packager(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "desktop/scripts/package_runtime.py"
    spec = importlib.util.spec_from_file_location("openecon_package_runtime_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    modules = ["openecon.econometrics", *[entry.name for entry in
               pkgutil.walk_packages(openecon.econometrics.__path__, "openecon.econometrics.")]]
    calls = []
    hooks = ModuleType("PyInstaller.utils.hooks")

    def collect_submodules(name, *, on_error):
        calls.append((name, on_error))
        return modules

    hooks.collect_submodules = collect_submodules
    for name in ("PyInstaller", "PyInstaller.utils"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    return module, modules, calls, hooks


def test_installer_inventory_covers_every_current_entry_export_forecast_and_family(packager):
    module, available, calls, _ = packager
    available += [available[-1], available[0]]  # Hook duplicates must not repeat flags.
    collected = module.econometrics_hidden_imports()
    required = {entry.entry.split(":")[0] for entry in registry.all_estimators() if not entry.legacy}
    required |= {target[0] for target in registry.public_exports().values()}
    required |= {target[0] for target in registry.forecasters().values()}
    required |= {f"openecon.econometrics.{family}" for family in registry.FAMILIES}
    required |= {"openecon.econometrics.count.censored", "openecon.econometrics.postest.prediction",
                 "openecon.econometrics.postest.inference", "openecon.econometrics.tsmodels.nardl",
                 "openecon.econometrics.postest.limited_prediction",
                 "openecon.econometrics.postest.ordinal_prediction",
                 "openecon.econometrics.unitroot.cips",
                 "openecon.econometrics.unitroot.cips_tables",
                 "openecon.econometrics.panel.homogeneity",
                 "openecon.econometrics.tsmodels.nardl_bootstrap"}
    assert required <= set(collected)
    assert collected == sorted(set(available))
    assert calls == [("openecon.econometrics", "raise")]


@pytest.mark.parametrize("missing", ["openecon.econometrics.count.censored",
                                     "openecon.econometrics.postest.prediction",
                                     "openecon.econometrics.postest.limited_prediction",
                                     "openecon.econometrics.postest.ordinal_prediction",
                                     "openecon.econometrics.unitroot.cips",
                                     "openecon.econometrics.unitroot.cips_tables",
                                     "openecon.econometrics.panel.homogeneity",
                                     "openecon.econometrics.tsmodels.nardl_bootstrap",
                                     "openecon.econometrics.arima.estimators"])
def test_installer_rejects_missing_lazy_estimator_or_public_postestimation_module(packager, missing):
    module, available, _, _ = packager
    assert missing in available
    available.remove(missing)
    with pytest.raises(RuntimeError, match=missing):
        module.econometrics_hidden_imports()


def test_uninspectable_family_import_is_not_silently_skipped(packager, monkeypatch):
    module, _, _, hooks = packager

    def broken(name, *, on_error):
        assert on_error == "raise"
        raise ImportError("family dependency cannot be inspected")

    monkeypatch.setattr(hooks, "collect_submodules", broken)
    with pytest.raises(ImportError, match="family dependency"):
        module.econometrics_hidden_imports()


def test_frozen_notices_survive_stale_editable_metadata_without_local_provenance(packager, tmp_path):
    module, _, _, _ = packager
    internal = tmp_path / "_internal"
    for name in ["openecon-0.3.18a1", "openecon_charts-0.3.0a1", "other-1.0"]:
        dist = internal / (name + ".dist-info")
        (dist / "licenses").mkdir(parents=True)
        (dist / "METADATA").write_text("Metadata-Version: 2.4\nName: " + name + "\nLicense-File: LICENSE\n\nOriginal description\n")
        (dist / "licenses/LICENSE").write_text("stale installed license")
        (dist / "direct_url.json").write_text('{"url":"file:///private/build-checkout"}')
        (dist / "uv_cache.json").write_text("private build provenance")
        with (dist / "RECORD").open("w") as handle:
            csv.writer(handle).writerows([[dist.name + "/METADATA", "", ""], [dist.name + "/direct_url.json", "", ""], [dist.name + "/uv_cache.json", "", ""], [dist.name + "/RECORD", "", ""]])
    module.prepare_distribution_notices(tmp_path)
    for dist in internal.iterdir():
        assert not (dist / "direct_url.json").exists()
        assert not (dist / "uv_cache.json").exists()
        rows = list(csv.reader((dist / "RECORD").read_text().splitlines()))
        assert all("direct_url.json" not in row[0] and "uv_cache.json" not in row[0] for row in rows)
        if dist.name.startswith("other-"):
            continue
        project = module.ROOT / "packages/openecon-charts" if dist.name.startswith("openecon_charts-") else module.ROOT
        assert (dist / "licenses/NOTICE").read_bytes() == (project / "NOTICE").read_bytes()
        row = next(row for row in rows if row[0] == dist.name + "/licenses/NOTICE")
        expected = base64.urlsafe_b64encode(hashlib.sha256((project / "NOTICE").read_bytes()).digest()).decode().rstrip("=")
        assert row[1] == "sha256=" + expected
        assert "License-File: NOTICE" in (dist / "METADATA").read_text()


def test_pyinstaller_command_includes_inventory_and_preserves_native_bootstrap(packager, monkeypatch, tmp_path):
    module, _, _, _ = packager
    inventory = module.econometrics_hidden_imports()
    monkeypatch.setattr(sys, "argv", ["package_runtime.py", "--skip-web-build"])
    monkeypatch.setattr(module, "uv_bundle_inputs", lambda: (tmp_path / "uv", [], {}))
    monkeypatch.setattr(module.importlib.metadata, "distribution", lambda name: object())
    commands = []

    class BuildIntercepted(Exception):
        pass

    def intercept(command, **kwargs):
        commands.append(command)
        raise BuildIntercepted

    monkeypatch.setattr(module.subprocess, "run", intercept)
    with pytest.raises(BuildIntercepted):
        module.main()
    assert len(commands) == 1
    command = commands[0]
    hidden = [command[index + 1] for index, value in enumerate(command) if value == "--hidden-import"]
    assert set(inventory) <= set(hidden)
    assert hidden[:3] == ["openecon.desktop_entry", "openecon.desktop_runtime", "openecon.server"]
    assert "openecon._network_signed" in hidden
    assert "openecon.console_worker" in hidden and "openecon_charts.annotations" in hidden
    assert {"openecon._network_store", "openecon._network_disk_algorithms"} <= set(hidden)
    assert "openecon._network_multi" in hidden
    assert {"openecon._network_dynamic", "openecon._network_dynamic_io"} <= set(hidden)
    assert {"openecon.mcp_launcher", "openecon.mcp_server", "mcp.server.fastmcp", "mcp.server.stdio"} <= set(hidden)
    assert "--onedir" in command and "--console" in command
    excluded = [command[index + 1] for index, value in enumerate(command) if value == "--exclude-module"]
    assert "statsmodels" in excluded and "openecon.team_cloud" in excluded


@pytest.fixture
def mac_packager(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "desktop/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("mac_package_brand_contract", scripts / "package_macos.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("product", ["OpenEcon", "OpenEconometrics"])
def test_dmg_stages_only_selected_app_preserves_aliases_and_retained_siblings(mac_packager, monkeypatch, tmp_path, product):
    module = mac_packager
    source = tmp_path / "macos"
    app = source / f"{product}.app"
    payload = app / "Contents/Resources/runtime"
    payload.mkdir(parents=True)
    (payload / "library.dylib").write_bytes(b"verified library fixture")
    (payload / "alias.dylib").symlink_to("library.dylib")
    sibling = source / ("OpenEcon.app" if product == "OpenEconometrics" else "OpenEconometrics.app")
    sibling.mkdir()
    (sibling / "retain.txt").write_bytes(b"old app remains")
    # An unrelated source-side shortcut is never removed by image staging.
    (source / "Applications").write_bytes(b"unrelated source-side file")
    monkeypatch.setattr(module, "finalize_app", lambda selected: {"selected": selected.name})
    observed = {}

    def hdiutil(command, **kwargs):
        assert command[:2] == ["hdiutil", "create"]
        staging = Path(command[command.index("-srcfolder") + 1])
        assert staging != source
        assert {p.name for p in staging.iterdir()} == {app.name, "Applications"}
        assert (staging / "Applications").readlink() == Path("/Applications")
        copied = staging / app.name / "Contents/Resources/runtime"
        assert (copied / "library.dylib").read_bytes() == (payload / "library.dylib").read_bytes()
        assert (copied / "alias.dylib").is_symlink()
        assert (copied / "alias.dylib").readlink() == Path("library.dylib")
        assert command[command.index("-volname") + 1] == product
        observed["staging"] = staging

    monkeypatch.setattr(module.subprocess, "run", hdiutil)
    output = tmp_path / "images" / f"{product}_0.3.35_aarch64.dmg"
    record = module.package(app, output)
    assert record["app"] == str(app) and record["dmg"] == str(output)
    assert not observed["staging"].exists()
    assert (sibling / "retain.txt").read_bytes() == b"old app remains"
    assert (source / "Applications").read_bytes() == b"unrelated source-side file"
    assert (payload / "alias.dylib").is_symlink()


def test_unknown_app_bundle_is_refused_before_finalization(mac_packager, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(mac_packager, "finalize_app", lambda app: calls.append(app))
    with pytest.raises(RuntimeError, match="recognized application"):
        mac_packager.package(tmp_path / "Other.app", tmp_path / "Other.dmg")
    assert calls == []


def test_default_mac_bundle_names_follow_product_config(mac_packager, monkeypatch, tmp_path):
    desktop = tmp_path / "desktop"
    (desktop / "src-tauri").mkdir(parents=True)
    (desktop / "src-tauri/tauri.conf.json").write_text(json.dumps({"productName": "OpenEconometrics", "version": "0.3.35"}))
    monkeypatch.setattr(mac_packager, "DESKTOP", desktop)
    monkeypatch.setattr(sys, "argv", ["package_macos.py"])
    captured = {}

    def package(app, output):
        captured.update(app=app, output=output)
        return {"intercepted": True}

    monkeypatch.setattr(mac_packager, "package", package)
    mac_packager.main()
    assert captured["app"] == desktop / "src-tauri/target/release/bundle/macos/OpenEconometrics.app"
    assert captured["output"] == desktop / "src-tauri/target/release/bundle/dmg/OpenEconometrics_0.3.35_aarch64.dmg"
