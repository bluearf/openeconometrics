"""Public replay conditions agree with actual native option validators."""

import json
import subprocess
import sys

import pytest

from openecon.analysis_contracts import AnalysisError, capabilities
from openecon.econometrics import registry
from openecon.econometrics.streaming_registry import conditions, supports_spec
from openecon.models import ModelSpec


def test_mixed_covariance_is_conditioned_on_ml_or_reml():
    supported = conditions()["mixed"]
    assert supported["method"] == ["ml", "reml"]
    assert supported["covariance_by_method"] == {
        "ml": list(registry.get("mixed").covariances),
        "reml": ["nonrobust"],
    }
    # The public capability response exposes the condition, so users do not
    # infer that every VCE is available for both estimation methods.
    assert capabilities()["streaming"]["conditions"]["mixed"] == supported


def test_streg_subject_ids_and_native_cluster_limit():
    supported = conditions()["streg"]
    assert supported["cluster_dimensions"] == registry.get("streg").cluster_dimensions == 1
    assert supported["id"] is True
    assert supported["independent_records"] is True


def test_native_option_catalogue_exposes_completed_replay_methods():
    supported = conditions()
    covariances = capabilities()['streaming']['covariances_by_estimator']
    assert 'hac' in covariances['ivregress']
    assert 'driscoll_kraay' in covariances['xtreg']
    assert set(supported['xtreg']['model']) == set(registry.get('xtreg').option('model').choices)
    for name in ['ivregress', 'ivreghdfe', 'heckman', 'ivprobit', 'ivtobit']:
        assert set(supported[name]['method']) == set(registry.get(name).option('method').choices)
    assert set(supported['teffects']['method']) == set(registry.get('teffects').option('method').choices)
    assert set(supported['xtgee']['corr']) == set(registry.get('xtgee').option('corr').choices)
    assert supported['mixed']['levels'] == 2
    assert supported['mixed']['random_slopes'] is True
    assert supported['gmm']['quadratic_spectral_hac'] is True


@pytest.mark.parametrize(
    "name,family", [("xtlogit", "logit"), ("xtprobit", "probit"), ("xtpoisson", "poisson")]
)
@pytest.mark.parametrize("model", ["re", "fe", "pa"])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_panel_covariance_conditions_match_native_validator(name, family, model, covariance):
    from openecon.econometrics.mixed.xt import _validate
    from openecon.econometrics.streaming_linear import _Notes

    spec = ModelSpec(
        estimator=name,
        outcome="y",
        predictors=["x"],
        panel="panel",
        covariance=covariance,
        cluster="enclosing" if covariance == "cluster" else None,
        options={"model": model},
    )
    supported = conditions()[name]
    allowed = covariance in supported["covariance_by_model"].get(model, [])
    assert (model in supported["model"]) == (model in supported["covariance_by_model"])
    if allowed:
        _validate(_Notes(spec), family, model, name)
    else:
        with pytest.raises(AnalysisError) as error:
            _validate(_Notes(spec), family, model, name)
        assert error.value.code == (
            "invalid_spec" if name == "xtprobit" and model == "fe" else "unsupported_covariance"
        )


@pytest.mark.parametrize("name", ["xtlogit", "xtprobit", "xtpoisson"])
@pytest.mark.parametrize(
    "corr", ["independent", "exchangeable", "ar1", "stationary", "nonstationary", "unstructured"]
)
def test_panel_correlation_conditions_match_actual_dispatch_scope(name, corr):
    supported = conditions()[name]
    assert supported["corr_options_apply_to"] == "pa"
    assert supported["quadrature_options_apply_to"] == "re"
    spec = ModelSpec(
        estimator=name,
        outcome="y",
        predictors=["x"],
        panel="panel",
        covariance="nonrobust",
        options={"model": "pa", "corr": corr},
    )
    assert supports_spec(spec) == (corr in supported["corr_by_model"]["pa"])


def test_capability_conditions_remain_torch_free_in_fresh_process():
    code = (
        "import json,sys; from openecon.analysis_contracts import capabilities; "
        "c=capabilities(); print(json.dumps({'torch': 'torch' in sys.modules, "
        "'models': len(c['estimators']), 'stream': len(c['streaming']['estimators']), "
        "'conditions': c['streaming']['conditions'], 'eager': c['eager_only_estimators']}))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code], check=True, text=True, capture_output=True
    )
    result = json.loads(process.stdout)
    assert result["torch"] is False
    assert result["stream"] == 80
    assert result["models"] == 80 + len(result["eager"])
    assert result["conditions"]["xtprobit"]["fixed_effects"] is False
