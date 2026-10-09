"""Public registry and cold import boundaries for sensitivity method delivery."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

NAMES = ("ovb_sensitivity", "evalue", "manski_ate", "lee_bounds", "randomization_test",
         "stratified_randomization", "rosenbaum_rank_bounds", "bias_sensitivity")


def test_package_and_method_manifests_remain_torch_free():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys, openecon\n"
        "from openecon.econometrics.causal_design import EXPORTS\n"
        f"assert all(name in EXPORTS for name in {NAMES!r})\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'scipy' not in sys.modules\n"
        "assert 'statsmodels' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=root, text=True,
                            capture_output=True, env=dict(os.environ, PYTHONPATH=str(root / "src")))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", NAMES)
def test_lazy_public_method_resolves_to_declared_source(name):
    import openecon as oe
    from openecon.econometrics.causal_design import EXPORTS

    function = getattr(oe, name)
    assert callable(function)
    assert function.__name__ == name
    assert function.__module__ == EXPORTS[name].split(":")[0]
