"""The lightweight catalogue agrees with native saved-prediction dispatch."""
import json
import subprocess
import sys

from openecon.analysis_contracts import capabilities
from openecon.prediction_capabilities import (
    ADVANCED_ESTIMATORS, COUNT_ESTIMATORS, DIRECT_ESTIMATORS, GROUP_ESTIMATORS, LINEAR_ESTIMATORS, SAVED_ESTIMATORS,
)


def test_prediction_inventory_matches_native_dispatch_and_not_fitting_count():
    from openecon.econometrics.postest import advanced_prediction, group_response, linear_prediction, mixture_prediction, prediction

    contract = capabilities()
    saved = contract["postestimation"]["saved_prediction"]
    assert LINEAR_ESTIMATORS == linear_prediction.ESTIMATORS
    assert COUNT_ESTIMATORS == mixture_prediction.ESTIMATORS
    assert GROUP_ESTIMATORS == group_response.ESTIMATORS
    assert ADVANCED_ESTIMATORS == advanced_prediction.ESTIMATORS
    assert len(DIRECT_ESTIMATORS) == 21
    assert SAVED_ESTIMATORS == prediction.SUPPORTED == set(saved["estimators"])
    assert saved["estimator_count"] == len(SAVED_ESTIMATORS)
    assert saved["remaining_estimator_count"] == len(contract["estimators"])-len(SAVED_ESTIMATORS)
    assert set(saved["remaining_estimators"]) == set(contract["estimators"]) - prediction.SUPPORTED
    assert set(contract["estimators"]) - set(contract["streaming"]["estimators"]) == set(contract["eager_only_estimators"])


def test_prediction_catalogue_is_torch_free_in_a_fresh_process():
    # Export discovery imports empty package namespaces, never their kernels.
    code = (
        "import json,sys; import openecon as oe; c=oe.capabilities(); "
        "print(json.dumps({'loaded': [name for name in sys.modules if "
        "name == 'torch' or name.startswith('openecon.econometrics.postest.') "
        "or name.startswith('openecon.econometrics.stats.') "
        "or name.startswith('openecon.econometrics.multivariate.')], "
        "'postestimation': c['postestimation']}))"
    )
    process = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
    result = json.loads(process.stdout)
    assert result["loaded"] == []
    assert result["postestimation"]["saved_prediction"]["estimator_count"] == len(SAVED_ESTIMATORS)


def test_saved_domains_and_preview_limit_remain_explicit():
    contract = capabilities()
    saved = contract["postestimation"]["saved_prediction"]
    conditions = saved["conditions"]
    for name in ("areg", "reghdfe", "ivreghdfe", "ppmlhdfe"):
        assert 'saved' in conditions[name]["response"]
        assert conditions[name]["predict_kinds"] == ["xb", "stdp", "response", "derivative"]
        assert conditions[name]["margins_kinds"] == ["xb", "response"]
    for name in ("xtreg", "xtivreg"):
        assert conditions[name]["fd"]["supported"] is False
        assert conditions[name]["be"]["requires_supplied_panel_mean_covariates"] is True
    for name in ("xtlogit", "xtprobit", "xtpoisson"):
        assert conditions[name]["model"] == ["pa", "re"]
    assert "required" in conditions["sqreg"]["outcome"]
    assert "integer" in conditions["tnbreg"]["mem"]
    assert saved["full_saved_parameter_covariance"] is True
    assert saved["dataset"]["predict_output"] == "owned indexed temporary Parquet Dataset"
    assert saved["dataset"]["row_limit"] is None
    assert saved["dataset"]["margins_methods"] == ["ame", "mem"]
    assert contract["streaming"]["max_prediction_sample"] == contract["max_chart_predictions"] == 400
    assert "stored fit/chart preview only" in contract["streaming"]["max_prediction_sample_scope"]


def test_helper_coverage_and_fresh_contract_are_honest():
    contract = capabilities()
    helpers = contract["postestimation"]["dataset_helpers"]
    assert set(helpers["procedures"]) == {
        "ttest", "sdtest", "oneway", "anova", "correlate", "pcorr", "pca", "factor",
        "factortest", "pca_scores", "factor_scores",
        "describe", "rm_anova", "manova",
    }
    assert "Pearson" in helpers["procedures"]["correlate"]
    assert "lazy Dataset" in helpers["procedures"]["pca_scores"]
    assert "Spearman/Kendall" in helpers["procedures"]["correlate"]
    assert "Box M" in helpers["procedures"]["manova"]
    helpers["procedures"].clear()
    contract["postestimation"]["saved_prediction"]["conditions"]["areg"]["predict_kinds"].clear()
    fresh = capabilities()["postestimation"]
    assert len(fresh["dataset_helpers"]["procedures"]) == 14
    assert fresh["saved_prediction"]["conditions"]["areg"]["predict_kinds"] == ["xb", "stdp", "response", "derivative"]
