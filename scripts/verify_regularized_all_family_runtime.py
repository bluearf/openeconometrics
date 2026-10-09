"""Owned frozen execution, complete saved results and actual server restart.

This proves frozen backend/UI-output state, not native-window or release acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from types import CodeType
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "econometrics.regularized.extended",
    "econometrics.regularized.prediction",
    "econometrics.regularized.pls",
    "econometrics.regularized.kernels",
    "econometrics.regularized.glm_design",
    "econometrics.regularized.__init__",
    "econometrics.core",
    "econometrics.registry",
    "analysis",
    "analysis_contracts",
    "models",
    "resources",
]
NAMES = ("ridge", "lasso", "elasticnet", "pls")
MARKER = "REGULARIZED_ALL_FAMILY_FROZEN_OK "
ARTIFACTS = ("regularized-all-family-models.json", "regularized-all-family-data.csv")
SETUP = """import sys,json,hashlib,importlib
from pathlib import Path
assert getattr(sys,'frozen',False)
"""
SAVE = """
models={name:result.model_dump(mode='json') for name,result in results.items()}
predictions={name:oe.regularized_predict(result,data).tolist() for name,result in results.items()}
tables={name:{'columns':list(oe.regularized_table(result).columns),'rows':oe.regularized_table(result).values.tolist()} for name,result in results.items()}
Path('regularized-all-family-models.json').write_text(json.dumps({'models':models,'predictions':predictions,'tables':tables},sort_keys=True,allow_nan=False),encoding='utf-8')
data.to_csv('regularized-all-family-data.csv',index=False)
for name,result in results.items():
    Path('regularized-all-family-'+name+'.json').write_text(result.model_dump_json(),encoding='utf-8')
    Path('regularized-all-family-'+name+'.tex').write_text(str(oe.regularized_table(result).to_latex(index=False)),encoding='utf-8')
files={path.name:{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size} for path in Path('.').glob('regularized-all-family-*') if path.is_file()}
print('REGULARIZED_ALL_FAMILY_FROZEN_OK '+json.dumps({'files':files,'frozen':getattr(sys,'frozen',False)},sort_keys=True,allow_nan=False))
"""
REPLAY = (
    SETUP
    + """import pandas as pd
import openecon as oe
import openecon.analysis as analysis
import openecon.econometrics.regularized.extended as extended
import openecon.econometrics.regularized.kernels as kernels
import openecon.econometrics.regularized.pls as pls_module
import openecon.econometrics.regularized.prediction as prediction_module
from openecon.models import ResultBundle
def no_refit(*args,**kwargs):
    raise AssertionError('Saved replay attempted a new fit')
analysis.fit=no_refit
extended.fit_extended=no_refit
kernels.fit_penalized=no_refit
prediction_module.fit_penalized=no_refit
pls_module.fit_pls=no_refit
pls_module._fit=no_refit
for name in ('ridge','lasso','elasticnet','pls'):
    setattr(oe,name,no_refit)
saved=json.loads(Path('regularized-all-family-models.json').read_text(encoding='utf-8'))
data=pd.read_csv('regularized-all-family-data.csv')
for name,state in saved['models'].items():
    result=ResultBundle.model_validate(state)
    assert result.model_dump(mode='json')==state
    prediction=oe.regularized_predict(result,data).tolist()
    assert all(abs(a-b)<1e-11 for a,b in zip(prediction,saved['predictions'][name],strict=True))
    table=oe.regularized_table(result)
    assert {'columns':list(table.columns),'rows':table.values.tolist()}==saved['tables'][name]
    assert str(table.to_latex(index=False))==Path('regularized-all-family-'+name+'.tex').read_text(encoding='utf-8')
    assert ResultBundle.model_validate_json(Path('regularized-all-family-'+name+'.json').read_text(encoding='utf-8')).model_dump(mode='json')==state
    display(table)
files={path.name:{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size} for path in Path('.').glob('regularized-all-family-*') if path.is_file()}
print('REGULARIZED_ALL_FAMILY_FROZEN_OK '+json.dumps({'files':files,'frozen':getattr(sys,'frozen',False)},sort_keys=True,allow_nan=False))
"""
)


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def normalized(code):
    return code.replace(co_filename='<bundled-source>', co_consts=tuple(
        normalized(item) if isinstance(item, CodeType) else item for item in code.co_consts))


def load_controller(path):
    """File loading also works under -I, without adding repository import paths."""
    spec = importlib.util.spec_from_file_location('_regularized_owned_controller', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('The owned runtime controller cannot be loaded.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WindowsRuntime:
    """Adapt the existing sanitized installed controller, without executing app code."""
    def __init__(self, runtime, root, project, sdk_version):
        controller = load_controller(ROOT / 'desktop/scripts/verify_windows_runtime.py')
        self.runtime = controller.InstalledRuntime(runtime, root, sdk_version)
        self.project = project
        self.workspace_token = None
        try:
            self.runtime.start()
        except BaseException:
            self.close()
            raise

    def request(self, path, body=None, method=None, *, desktop=False):
        token = self.runtime.token if desktop else self.workspace_token
        if method is None:
            return self.runtime.call(path, body, token=token)
        request = Request(self.runtime.descriptor['url'] + path,
                          data=json.dumps(body).encode('utf-8') if body is not None else None,
                          headers={'Content-Type': 'application/json', 'X-OpenEcon-Token': token},
                          method=method)
        with self.runtime.http.open(request, timeout=300) as response:
            payload = response.read(32 * 1024 * 1024 + 1)
        if len(payload) > 32 * 1024 * 1024:
            raise RuntimeError('Runtime response exceeded the verification bound.')
        return json.loads(payload)

    def call(self, path, body=None, method=None):
        return self.request(f'/api/desktop/projects/{self.project}/workspace' + path,
                            body, method)

    def open_project(self):
        self.request(f'/api/desktop/local-projects/{self.project}/open', {}, desktop=True)
        _, self.workspace_token = self.runtime.connect(self.project)

    def close(self):
        try:
            self.runtime.stop()
        finally:
            self.runtime.stderr.close()


def owned_runtime(runtime, root, project, sdk_version):
    if os.name == 'nt':
        return WindowsRuntime(runtime, root, project, sdk_version)
    controller = load_controller(ROOT / 'scripts/verify_bai_perron_runtime.py')
    return controller.OwnedRuntime(runtime, root, project)


def verify_run(run, project_root, *, require_frozen=True, order=NAMES):
    assert run["status"] == "ok", run.get("error")
    payloads = [
        json.loads(line[len(MARKER) :])
        for line in run["stdout"].splitlines()
        if line.startswith(MARKER)
    ]
    assert len(payloads) == 1
    marker = payloads[0]
    assert marker['frozen'] is require_frozen
    required = set(ARTIFACTS) | {f'regularized-all-family-{name}.{suffix}'
                               for name in NAMES for suffix in ('json', 'tex')}
    assert set(marker['files']) == required
    for name, expected in marker['files'].items():
        path = project_root / name
        assert not path.is_symlink() and path.is_file()
        assert 0 < path.stat().st_size == expected['bytes'] <= 2 * 1024 * 1024
        assert digest(path) == expected['sha256']
    payload = json.loads((project_root / ARTIFACTS[0]).read_text(encoding='utf-8'))
    assert set(payload) == {'models', 'predictions', 'tables'}
    assert all(set(payload[key]) == set(NAMES) for key in payload)
    assert len(run["outputs"]) == 4
    assert set(order) == set(NAMES) and len(order) == 4
    for name, output in zip(order, run['outputs'], strict=True):
        assert output["type"] == "table"
        assert len(output["data"]["rows"]) == output["data"]["total_rows"] == 4
        expected = payload['tables'][name]
        assert output['data']['columns'] == expected['columns']
        assert output['data']['rows'] == expected['rows']
        assert output['data']['total_columns'] == len(expected['columns']) == 2
        assert "\\begin{tabular}" in output.get("latex", "") or "\\begin{longtable}" in output.get(
            "latex", ""
        )
    for name, model in payload["models"].items():
        assert model["spec"]["estimator"] == name
        assert model["nobs"] == 60 and model["sample_positions"] == list(range(60))
        assert model["coefficients"] == [] and model["covariance_matrix"] == []
        assert model["inference"]["available"] is False
        state = model["extra"]["regularized_extended_state"]
        assert state["weight_type"] == "fweight" and state["frequency_total"] == 120
        assert len(state["cv"]) == 3 and state["design"]["categorical"] == ["sector"]
        assert len(payload["predictions"][name]) == 60
        assert all(math.isfinite(value) for value in payload['predictions'][name])
        assert json.loads((project_root / f'regularized-all-family-{name}.json').read_text(
            encoding='utf-8')) == model
    payload['artifact_manifest'] = marker['files']
    return payload


def verify(runtime, sdk_version, source_sha):
    from PyInstaller.archive.readers import CArchiveReader

    runtime = runtime.resolve(strict=True)
    if os.name == 'nt' and (os.environ.get('GITHUB_ACTIONS') != 'true'
                           or os.environ.get('RUNNER_ENVIRONMENT') != 'github-hosted'):
        raise RuntimeError('Windows acceptance requires disposable GitHub-hosted Windows.')
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    modules = {}
    for name in MODULES:
        source = ROOT / "src/openecon" / (name.replace(".", "/") + ".py")
        module = ("openecon." + name).removesuffix(".__init__")
        assert normalized(archive.extract(module)) == normalized(
            compile(source.read_text(encoding='utf-8'), str(source), "exec", dont_inherit=True)
        ), module
        modules[module] = digest(source)
    module_guard = ('import openecon as oe\nassert oe.__version__ == ' + repr(sdk_version)
                    + '\nfor name in ' + repr(tuple(modules))
                    + ':\n    module=importlib.import_module(name)\n'
                    + '    assert Path(module.__file__).is_relative_to(Path(sys._MEIPASS))\n')
    example = ROOT / 'docs/examples/regularized_all_family.py'
    code = SETUP + module_guard + example.read_text(encoding='utf-8') + SAVE
    with tempfile.TemporaryDirectory(prefix="openecon-regularized-all-family-") as temporary:
        root = Path(temporary)
        owned = owned_runtime(runtime, root, 'unused', sdk_version)
        try:
            owned.project = owned.request(
                "/api/desktop/local-projects",
                {"name": "Regularized all-family synthetic QA"},
                desktop=True,
            )["id"]
            owned.open_project()
            project = owned.project
            project_root = root / 'projects' / project
            owned.call("/console/script", {"code": code, "name": "analysis.py"}, method="PUT")
            run = owned.call("/console/execute", {"code": code, "timeout_seconds": 120})
            proof = verify_run(run, project_root)
            script = owned.call("/console/script")
            saved = next(
                item for item in owned.call("/console")["history"] if item["id"] == run["id"]
            )
        finally:
            owned.close()
        owned = owned_runtime(runtime, root, project, sdk_version)
        try:
            owned.open_project()
            assert owned.request("/api/desktop/status", desktop=True)["console"]["pid"] is None
            reopened = next(
                item for item in owned.call("/console")["history"] if item["id"] == run["id"]
            )
            assert all(
                reopened[key] == saved[key] for key in ("code", "stdout", "outputs", "events")
            )
            assert owned.call("/console/script") == script
            replay = owned.call("/console/execute", {"code": REPLAY, "timeout_seconds": 120})
            restored = verify_run(replay, project_root, order=sorted(NAMES))
            assert restored == proof
        finally:
            owned.close()
    return {
        "status": "passed",
        'source_sha': source_sha,
        'sdk_version': sdk_version,
        'platform': os.name,
        'verifier_sha256': digest(__file__),
        'example_sha256': digest(example),
        "runtime_sha256": digest(runtime),
        "compiled_modules_equal_source": modules,
        "four_complete_models_equal_after_restart": True,
        "saved_replay_with_fit_disabled": True,
        "code_stdout_outputs_events_equal_after_restart": True,
        "editor_script_equal_after_restart": True,
        "tables": 8,
        "rows_per_table": 4,
        "no_worker_needed_for_history": True,
        "owned_runtime_stopped": True,
        "temporary_profile_removed": not root.exists(),
        "source_path_injected": False,
        "native_window_verified": False,
        "public_release_delivered": False,
        "complete_result_sha256": {
            name: hashlib.sha256(
                json.dumps(state, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            for name, state in proof["models"].items()
        },
        'artifact_manifest': proof['artifact_manifest'],
        'ordered_outputs_sha256': digest_json(saved['outputs']),
        'saved_code_stdout_outputs_events_sha256': digest_json(
            {key: saved[key] for key in ('code', 'stdout', 'outputs', 'events')}),
    }


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--sdk-version', required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    if len(args.source_sha) != 40 or any(char not in '0123456789abcdef' for char in args.source_sha):
        parser.error('--source-sha must be the exact 40-character source commit.')
    try:
        receipt = verify(args.runtime, args.sdk_version, args.source_sha)
    except BaseException as error:
        receipt = {'status': 'error', 'source_sha': args.source_sha, 'sdk_version': args.sdk_version,
                   'error': f'{type(error).__name__}: {error}',
                   'native_window_verified': False, 'public_release_delivered': False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding='utf-8')
    print(json.dumps({"status": receipt["status"], "tables": receipt["tables"]}))
