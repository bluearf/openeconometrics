"""Verify the installed SDK patch in an isolated interpreter (never this checkout)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

PROGRAM = r'''
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import socket
import sys

assert metadata.version('openecon') == '0.3.18a3'
assert metadata.version('openecon-charts') == '0.3.0a2'
import openecon as oe
from openecon.econometrics import registry
names = registry.names()
exports = {**oe._EXPORTS, **registry.public_exports()}
assert {'ols', 'poisson', 'nbreg'} <= set(names)
assert len(names) == len(set(names))
root = Path(oe.__file__).resolve().parent
assert root.is_relative_to(Path(sys.prefix).resolve())
entries = [entry.entry.split(':')[0] for entry in registry.all_estimators()]
entries += [module for module, _ in exports.values()]
for module in set(entries):
    assert module.startswith('openecon.'), module
    candidate = root.joinpath(*module.split('.')[1:])
    assert candidate.with_suffix('.py').is_file() or (candidate / '__init__.py').is_file(), module
assert 'torch' not in sys.modules and 'pandas' not in sys.modules

def denied(*args, **kwargs):
    raise AssertionError('Installed numerical smoke check attempted network access')
class OfflineSocket(socket.socket):
    connect = denied
    connect_ex = denied
socket.socket = OfflineSocket
socket.create_connection = denied

import numpy as np
import pandas as pd
frame = oe.example()
ols = oe.ols(data=frame, y='wage', x=['education', 'experience'], covariance='HC3', device='cpu')
assert ols.nobs == 480
design = np.column_stack([np.ones(len(frame)), frame[['education', 'experience']].to_numpy()])
expected = np.linalg.lstsq(design, frame.wage.to_numpy(), rcond=None)[0]
np.testing.assert_allclose([c.estimate for c in ols.coefficients], expected, rtol=1e-10, atol=1e-10)
assert np.isfinite(ols.covariance_matrix).all()
assert r'\begin{tabular}' in ols.to_latex()
stored_ols = oe.ResultBundle.model_validate_json(ols.model_dump_json())
assert stored_ols.model_dump(mode='json') == ols.model_dump(mode='json')
new = frame.iloc[[5, 1, 4]].copy()
new.index = ['repeated', 'repeated', 'other']
before = oe.predict(ols, new, kind='xb')
after = oe.predict(stored_ols, new, kind='xb')
assert after.index.tolist() == new.index.tolist()
np.testing.assert_allclose(before.to_numpy(), after.to_numpy(), rtol=1e-12, atol=1e-12)

rng = np.random.default_rng(7742)
count = pd.DataFrame({'x': rng.normal(size=240), 'z': rng.normal(size=240)})
count['y'] = rng.poisson(np.exp(.5 + .2 * count.x - .15 * count.z))
poisson = oe.poisson(data=count, y='y', x=['x', 'z'])
assert poisson.nobs == 240
beta = np.array([c.estimate for c in poisson.coefficients])
design = np.column_stack([np.ones(len(count)), count[['x', 'z']].to_numpy()])
mu = np.exp(design @ beta)
assert np.max(np.abs(design.T @ (count.y.to_numpy() - mu))) < 1e-5
assert np.isfinite(poisson.covariance_matrix).all()
assert all(c.std_error > 0 and c.ci_low <= c.estimate <= c.ci_high for c in poisson.coefficients)
stored_poisson = oe.ResultBundle.model_validate_json(poisson.model_dump_json())
assert stored_poisson.model_dump(mode='json') == poisson.model_dump(mode='json')
new_count = count.iloc[[4, 0, 3]].copy()
new_count.index = ['repeated', 'repeated', 'other']
before = oe.predict(poisson, new_count, kind='response')
after = oe.predict(stored_poisson, new_count, kind='response')
assert after.index.tolist() == new_count.index.tolist()
np.testing.assert_allclose(before.to_numpy(), after.to_numpy(), rtol=1e-12, atol=1e-12)
assert r'\begin{tabular}' in stored_poisson.to_latex()
import openecon_charts as charts
assert charts.coefficients(stored_poisson).model_dump()
for package in ['fastapi', 'uvicorn', 'typer', 'mcp', 'openpyxl', 'pyreadstat', 'scipy', 'statsmodels', 'linearmodels']:
    assert importlib.util.find_spec(package) is None, package

# Exercise the guarded Parquet parser after registry and Torch are resident.
import tempfile
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'guarded.parquet'
    frame.to_parquet(path, index=False)
    loaded = oe.read(path)
    assert len(loaded) == 480
    replay = oe.ols(data=loaded, y='wage', x=['education', 'experience'], covariance='HC3', device='cpu')
    np.testing.assert_allclose([c.estimate for c in replay.coefficients], [c.estimate for c in ols.coefficients], rtol=1e-12, atol=1e-12)

# Exact fits must not publish inferential variance made only of QR roundoff.
import torch
from openecon.engines import streaming_ols
from openecon.engines.contracts import KernelError
x = torch.tensor([[1., 0.], [1., 1.], [1., 2.], [1., 3.]], dtype=torch.float64)
for scale in (1e-20, 1., 1e20):
    y = scale * (2 + 3 * x[:, 1])
    for covariance in ('nonrobust', 'HC1', 'HC3'):
        try:
            streaming_ols.solve_ols(lambda: iter([(x[:3], y[:3]), (x[3:], y[3:])]), covariance)
        except KernelError as error:
            assert error.code == 'numerical_failure', error
        else:
            raise AssertionError('Numerical-zero exact fit was accepted')
y = 2 + 3 * x[:, 1] + torch.tensor([1., -1., 1., -1.], dtype=torch.float64) * 1e-10
noisy = streaming_ols.solve_ols(lambda: iter([(x[:3], y[:3]), (x[3:], y[3:])]), 'HC3')
assert noisy.residual_ss > 0

print(json.dumps({'status': 'passed', 'sdk_version': metadata.version('openecon'),
                  'charts_version': metadata.version('openecon-charts'),
                  'registry_estimators': len(names), 'registry_exports': len(exports),
                  'referenced_modules': len(set(entries)), 'lazy_registry_no_heavy_imports': True,
                  'ols_independent_algebra': True, 'guarded_parquet_after_torch': True, 'exact_fit_guard_and_real_noise': True, 'poisson_score_check': True,
                  'json_restore_predictions_duplicate_index': True, 'latex_and_chart_protocol': True,
                  'source': str(root), 'scope': 'Package smoke checks; no vendor parity or CUDA acceptance.'}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Preserve the venv executable path: resolving its symlink selects the host interpreter.
    check = subprocess.run([str(args.python.absolute()), '-I', '-c', PROGRAM], capture_output=True, text=True, check=False)
    if check.returncode:
        raise SystemExit(check.stderr)
    record = json.loads(check.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))

if __name__ == '__main__':
    main()
