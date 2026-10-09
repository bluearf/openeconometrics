"""Independent fail-closed checks for preserved PR91 metadata utilities."""

from pathlib import Path
import ast
import importlib
import json
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
editor = runpy.run_path(str(ROOT / "scripts/generate_editor_api.py"))


class Reader:
    def __init__(self, manifest):
        self.manifest = manifest

    def committed(self, path):
        if path.endswith("registry.py"):
            return ast.parse("""
class EstimatorInfo:
    name: str
    title: str
    family: str
    entry: str
    function: str
    description: str
FAMILIES = ('example',)
""")
        return ast.parse(self.manifest)


def test_safe_literal_bindings_preserve_unique_old_manifest_forms():
    exports = editor["registry_exports"](
        Reader("""
ESTIMATORS = tuple(EstimatorInfo(name=name, title=title, family='example',
    function='model_' + name, entry=f'openecon.econometrics.example:fit_{name}', description=title)
    for name, title in (('a', 'First estimator'), ('b', 'Second estimator')))
EXPORTS = {name: f'openecon.econometrics.{module}:{name}'
           for name, module in (('helper', 'example'),)}
""")
    )
    assert exports == {
        "forecast": ("openecon.econometrics.core", "forecast", None),
        "model_a": ("openecon.econometrics.example", "model_a", "First estimator"),
        "model_b": ("openecon.econometrics.example", "model_b", "Second estimator"),
        "helper": ("openecon.econometrics.example", "helper", None),
    }


def test_positional_constructor_fields_are_read_from_declared_dataclass_order():
    exports = editor["registry_exports"](
        Reader("""
ESTIMATORS = (EstimatorInfo('a', 'A', 'example', 'openecon.econometrics.example:fit', 'model_a', 'A'),)
EXPORTS = {}
""")
    )
    assert exports["model_a"] == ("openecon.econometrics.example", "model_a", "A")


@pytest.mark.parametrize(
    "iterator", ["danger()", "range(3)", "obj.values", "[x for x in danger()]", "('a',) * 101"]
)
def test_executable_or_unbounded_estimator_iterators_are_refused(iterator, tmp_path):
    marker = tmp_path / "must-not-exist"
    # No manifest is imported or executed, including its top-level statement.
    manifest = f"""\nopen({str(marker)!r}, 'w').write('executed')
ESTIMATORS = tuple(EstimatorInfo(function=name, entry='openecon.econometrics.example:fit') for name in {iterator})
EXPORTS = {{}}
"""
    with pytest.raises(ValueError):
        editor["registry_exports"](Reader(manifest))
    assert not marker.exists()


@pytest.mark.parametrize(
    "manifest",
    [
        "ESTIMATORS = tuple(EstimatorInfo(function=name, entry='openecon.econometrics.example:fit') for name in "
        + repr(tuple("m" + str(i) for i in range(101)))
        + ")",
        "ESTIMATORS = tuple(EstimatorInfo(function=name, entry='openecon.econometrics.example:fit') for name in ('a',) if danger())",
        "ESTIMATORS = tuple(EstimatorInfo(function=name, entry='openecon.econometrics.example:fit') for name in ('a',) for other in ('b',))",
        "ESTIMATORS = tuple(EstimatorInfo(function=name, entry='openecon.econometrics.example:fit') for name, title in (('a',),))",
        "ESTIMATORS = (EstimatorInfo(function='a', entry='openecon.econometrics.example:fit', unknown='field'),)",
        "ESTIMATORS = (EstimatorInfo('a', name='b', function='a', entry='openecon.econometrics.example:fit'),)",
        "ESTIMATORS = (EstimatorInfo(*danger(), function='a', entry='openecon.econometrics.example:fit'),)",
        "ESTIMATORS = (EstimatorInfo(**danger()),)",
    ],
)
def test_manifest_geometry_and_constructor_bindings_fail_closed(manifest):
    with pytest.raises(ValueError):
        editor["registry_exports"](Reader(manifest + "\nEXPORTS = {}"))


def test_auxiliary_metadata_matches_every_current_public_helper_without_tensor_imports():
    code = """
import sys,json
from openecon.analysis_contracts import capabilities
from openecon.econometrics import registry
record=capabilities()['auxiliary_exports']
expected={}
from importlib import import_module
for family in registry.FAMILIES:
    for name,target in getattr(import_module('openecon.econometrics.'+family),'EXPORTS',{}).items():
        assert name not in expected
        expected[name]={'family':family,'entry':target}
assert record==expected and len(record)>100
assert all(name in registry.public_exports() for name in record)
assert not any(name in sys.modules for name in ('torch','pandas','scipy'))
print(json.dumps({'helper_count':len(record)}))
"""
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, check=True
    )
    assert json.loads(process.stdout)["helper_count"] > 100


def test_duplicate_auxiliary_export_refuses_ambiguous_metadata(monkeypatch):
    registry = importlib.import_module("openecon.econometrics.registry")
    monkeypatch.setattr(registry, "FAMILIES", ("a", "b"))
    monkeypatch.setattr(
        registry,
        "import_module",
        lambda _: SimpleNamespace(EXPORTS={"helper": "openecon.econometrics.example:helper"}),
    )
    with pytest.raises(RuntimeError, match="exported twice"):
        registry.auxiliary_exports()
