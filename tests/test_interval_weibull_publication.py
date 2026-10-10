"""Public entry points and offline help retain the bounded typed contract."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_public_weibull_routes_keep_package_import_lightweight():
    code = f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r})\n" + """
import sys
import openecon as oe
from openecon.econometrics.interval_weibull_covariate_public import EXPORTS
assert 'torch' not in sys.modules and 'pandas' not in sys.modules
contract = oe.capabilities()['interval_weibull_covariate']
assert contract['devices'] == ['cpu'] and not contract['dataset_support']
assert not contract['stata_parity_validated']
assert contract['budgets']['covariates'] == 8
assert 'global optimum certification' in contract['unsupported']
assert set(EXPORTS) <= set(dir(oe)) & set(oe.__all__)
assert 'torch' not in sys.modules and 'pandas' not in sys.modules
from openecon.econometrics.survival_ext import weibull_regression as native
for name in (*EXPORTS, 'WeibullIntervalFit', 'WeibullIntervalPrediction'):
    assert getattr(oe, name) is getattr(native, name)
"""
    subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, check=True, timeout=30)


def test_offline_weibull_help_exposes_typed_restore_and_complete_exports():
    spec = importlib.util.spec_from_file_location(
        "weibull_public_catalog", ROOT / "scripts/generate_editor_api.py"
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    rows = generator.decode_catalog(json.loads((ROOT / "web/src/editor-api.json").read_bytes()))
    catalog = {row["name"]: row for row in rows}
    for function in ("stinterval_weibull_regression", "restore_interval_weibull_regression"):
        assert catalog["openecon." + function]["returns"] == "openecon.WeibullIntervalFit"
    for function in ("interval_weibull_regression_predict", "restore_interval_weibull_prediction"):
        assert catalog["openecon." + function]["returns"] == "openecon.WeibullIntervalPrediction"
    for cls in ("WeibullIntervalFit", "WeibullIntervalPrediction"):
        for method in ("to_tables", "table", "latex", "to_latex", "model_validate",
                       "model_validate_json", "model_dump", "model_dump_json", "model_copy"):
            entry = catalog[f"openecon.{cls}.{method}"]
            assert entry["kind"] == "method" and entry["description"]
        assert catalog[f"openecon.{cls}.to_tables"]["returns"] == "openecon.TableSet"
        assert catalog[f"openecon.{cls}.model_validate_json"]["returns"] == "openecon." + cls
    assert catalog["openecon.WeibullIntervalFit.dataset"]["returns"] == "pandas.DataFrame"
    assert "openecon.WeibullIntervalPrediction.dataset" not in catalog
    fit = {p["name"]: p for p in catalog["openecon.stinterval_weibull_regression"]["parameters"]}
    assert fit["max_work"]["default"] == "500000000"
    assert fit["device"]["default"] == "'cpu'" and fit["missing"]["default"] == "'raise'"


def test_complete_current_main_inventory_and_editor_objects_survive_weibull_addition():
    import ast
    import hashlib
    plan = json.loads((ROOT / "docs/econometrics/merge-gate-sdk-groups-plan-Weibull159-2026-10-10.json").read_text())
    def selectors(source):
        return ast.literal_eval(next(node.value for node in ast.parse(source).body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "TESTS" for target in node.targets)))
    original = selectors(subprocess.check_output(["git", "show", plan["actual_main_base"] + ":scripts/verify_merge_candidate.py"], cwd=ROOT))
    inventory = selectors((ROOT / "scripts/verify_merge_candidate.py").read_text())
    # The Weibull159 receipt is historical. Admit only the later conjoint
    # bootstrap selector under its own receipt and keep every Weibull159 check exact.
    bootstrap = json.loads((ROOT / "docs/econometrics/merge-gate-conjoint-bootstrap160-2026-10-10.json").read_text())
    assert bootstrap["appended_selectors"] == ["tests/test_econ_conjoint_bootstrap.py"]
    assert inventory == bootstrap["complete_selectors"] == bootstrap["previous_selectors"] + bootstrap["appended_selectors"]
    assert len(inventory) == len(set(inventory)) == 160
    current = inventory[:159]
    assert current == bootstrap["previous_selectors"]
    assert current[:156] == original == plan["complete_previous156_selectors"]
    assert current[156:] == plan["appended_selectors"]
    assert current == plan["selected_tests"] and len(current) == len(set(current)) == 159
    assert hashlib.sha256(json.dumps(current, separators=(",", ":")).encode()).hexdigest() == plan["selected_test_scope_sha256"]
    historical = plan["original_complete_Weibull156_plan"]
    historical_raw = (ROOT / historical["path"]).read_bytes()
    assert len(historical_raw) == historical["bytes"] and hashlib.sha256(historical_raw).hexdigest() == historical["sha256"]
    old_plan = json.loads(historical_raw)
    assert current[:153] + current[156:] == old_plan["selected_tests"]
    assert plan["source_pins"] == old_plan["source_pins"]
    assert plan["timeout_seconds"] == 900
    for name, expected in plan["source_pins"].items():
        raw = (ROOT / name).read_bytes()
        assert len(raw) == expected["bytes"] and hashlib.sha256(raw).hexdigest() == expected["sha256"]
    spec = importlib.util.spec_from_file_location("weibull_preserved_catalog", ROOT / "scripts/generate_editor_api.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    bindings = json.loads((ROOT / "tests/fixtures/editor-catalog-main369d84-Weibull1420-bindings.json").read_text())
    old_raw = (ROOT / bindings["original_baseline"]).read_bytes()
    assert old_raw == subprocess.check_output(["git", "show", bindings["actual_main_before_Weibull"] + ":web/src/editor-api.json"], cwd=ROOT)
    assert hashlib.sha256(old_raw).hexdigest() == bindings["original_baseline_sha256"]
    old = generator.decode_catalog(generator.read_catalog_wire(old_raw))
    raw = (ROOT / "web/src/editor-api.json").read_bytes()
    complete = generator.decode_catalog(generator.read_catalog_wire(raw))
    # Likewise admit only the three named conjoint bootstrap entries before the
    # exact Weibull1420 comparison.
    bootstrap_names = ["openecon.conjoint_bootstrap_fit", "openecon.conjoint_bootstrap_importance",
                       "openecon.conjoint_bootstrap_shares"]
    assert [row["name"] for row in complete if row["name"] in bootstrap_names] == bootstrap_names
    actual = [row for row in complete if row["name"] not in bootstrap_names]
    names = {row["name"] for row in old}
    assert len(old) == len(names) == 1395 and len(actual) == 1420 and len(complete) == 1423
    assert [row for row in actual if row["name"] in names] == old
    assert [row["name"] for row in actual if row["name"] not in names] == bindings["complete_added_Weibull_names_in_order"]
    assert len(raw) <= 500000
