"""Public names stay lazy and preserve the existing estimator families."""

from pathlib import Path
import subprocess
import sys

import openecon as oe


HELPERS = {
    "bayesian.hypothesis": (
        "bayes_hypothesis", "bayes_hypothesis_restore", "bayes_hypothesis_predict",
        "bayes_hypothesis_draws", "bayes_hypothesis_draws_restore",
    ),
    "latent.sem": (
        "latent_sem", "latent_sem_covariance", "latent_sem_restore", "sem_effects", "latent_scores",
    ),
    "tsworkflows.dfm": ("dfactor", "dfactor_restore", "dfactor_nowcast", "dfactor_forecast"),
    "weakiv.api": ("iv_clr_confidence_set", "iv_clr_restore", "iv_clr_test", "iv_clr_test_restore"),
    "mixtures.gaussian": ("finite_mixture", "finite_mixture_restore", "finite_mixture_predict"),
    "supervised.split": ("prediction_split", "split_restore"),
    "supervised.cart": ("cart", "cart_restore", "cart_predict", "cart_prediction_restore"),
}


def test_all_public_model_helpers_resolve_to_their_owned_modules():
    for suffix, names in HELPERS.items():
        for name in names:
            value = getattr(oe, name)
            assert name in oe.__all__
            assert callable(value)
            assert value.__module__ == "openecon.econometrics." + suffix
    for name in ("PosteriorHypothesisComparison", "PosteriorHypothesisDraws", "SEMState",
                 "CLRConfidenceSet", "CLRTestState", "SplitState", "CartState", "CartQueryState"):
        assert name in oe.__all__
        assert callable(getattr(oe, name).model_validate_json)


def test_new_family_manifests_do_not_load_the_numeric_runtime():
    source = str(Path(__file__).resolve().parents[1] / "src")
    code = f"""import sys, importlib
sys.path.insert(0, {source!r})
import openecon
for name in ('bayesian', 'latent', 'tsworkflows', 'weakiv', 'mixtures', 'supervised'):
    module = importlib.import_module('openecon.econometrics.' + name)
    assert isinstance(module.EXPORTS, dict)
    assert isinstance(module.ESTIMATORS, tuple)
assert 'torch' not in sys.modules
assert 'scipy' not in sys.modules
assert 'statsmodels' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
