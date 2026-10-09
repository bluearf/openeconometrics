"""The actual analysis runtime must work when any SciPy import is denied."""
import ast
import os
from pathlib import Path
import subprocess
import sys


def test_no_runtime_source_imports_scipy():
    root = Path(__file__).resolve().parents[1]
    for folder in (root/"src"/"openecon", root/"packages"/"openecon-charts"/"src"):
        for path in folder.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                assert not any(name == "scipy" or name.startswith("scipy.") for name in names), path


def test_sdk_models_run_with_imports_blocked():
    script = r'''
import builtins, json, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "scipy" or name.startswith("scipy.") or name == "statsmodels" or name.startswith("statsmodels."):
        raise ImportError("External reference-estimator runtime imports are denied in this test")
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import pandas as pd
import torch
import openecon as oe
from openecon.dataset import Dataset
torch.set_num_threads(1)
generator = torch.Generator().manual_seed(712)
x, z, u = torch.randn((3, 320), generator=generator, dtype=torch.float64)
frame = pd.DataFrame({"x": x.numpy(), "z": z.numpy(), "y": (1+2*x+.3*z+u).numpy(),
                      "binary": (u+x > 0).long().numpy(), "group": torch.arange(320).remainder(16).numpy()})
data = Dataset.from_frame(frame)
results = [oe.ols(data=data, y="y", x=["x", "z"], covariance="HC3"),
           oe.logit(data=data, y="binary", x=["x"]),
           oe.probit(data=data, y="binary", x=["x"]),
           oe.cnsreg(data=data, y="y", x=["x", "z"], constraints=[{"terms":{"z":1},"value":0}]),
           oe.areg(data=data, y="y", x=["x", "z"], absorb="group"),
           oe.mvreg(data=data, y=["y", "z"], x=["x"]),
           oe.gmm(data=data, moments=["y - {a} - {b}*x"], instruments=["x", "z"])]
assert all(result.nobs == 320 for result in results)
for procedure in (oe.ngperron, oe.kss):
    tested = procedure(frame, "y", lags=1)
    assert tested.attrs["p_value"] is None
    assert tested.attrs["nobs"] == 318
classical = oe.ols(data=frame, y="y", x=["x", "z"], covariance="nonrobust")
joint = oe.ols_stepdown(classical, error_model="iid_gaussian", draws=1000)
intervals = oe.simultaneous_t_ci([0., 0.], [[1., .2], [.2, 1.]], df=5, draws=1000,
    family_description="Prespecified Gaussian coefficient family",
    pivot_description="Known correlation and one independent common chi-square scale")
region = oe.hotelling_region(frame, ["x", "z"], sampling_model="iid_multivariate_normal")
assert len(joint) == 3 and len(intervals) == len(region) == 2
from pathlib import Path
exec(compile(Path('docs/examples/multivariate_options.py').read_text(), '<multivariate-options>', 'exec'), {'__name__': '__main__'})
exec(compile(Path('docs/examples/multivariate_extension_acceptance.py').read_text(), '<multivariate-extensions>', 'exec'), {'__name__': '__main__'})
exec(compile(Path('docs/examples/multivariate_weight_matrix.py').read_text(), '<multivariate-weight-matrix>', 'exec'), {'__name__': '__main__'})
assert not any(name == "scipy" or name.startswith("scipy.") for name in sys.modules)
assert not any(name == "statsmodels" or name.startswith("statsmodels.") for name in sys.modules)
print(json.dumps({"models": [result.spec.estimator for result in results], "scipy_loaded": False}))
'''
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True,
                            cwd=Path(__file__).resolve().parents[1], env={**os.environ, "PYTHONNOUSERSITE": "1"}, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '"scipy_loaded": false' in result.stdout
