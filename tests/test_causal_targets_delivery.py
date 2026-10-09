"""Lazy public delivery and persisted contracts for the third causal batch."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

NAMES = (
    "ovb_benchmark", "ovb_robustness", "treatment_cdf_ipw", "treatment_quantile_ipw",
    "treatment_cdf_aipw", "treatment_survival_ipcw", "treatment_rmst_ipcw", "treatment_rmst_aipw",
)


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
