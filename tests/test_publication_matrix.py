"""Real family fits, independent display assertions, and offline saved-result exports."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

import openecon as oe
from openecon.models import Coefficient, ResultBundle
from openecon.output_latex import PUBLICATION_STYLE, add_output_latex, enrich_record


@pytest.fixture(scope="module")
def matrix():
    path = Path(__file__).parents[1] / "scripts" / "publication_matrix.py"
    spec = importlib.util.spec_from_file_location("publication_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.fitted_models())


CASES = ("ols", "logit", "areg", "xtreg_robust", "xtreg_dk", "ivregress", "glm", "mlogit",
         "ologit", "tobit", "zinb", "qreg", "arima", "arch", "var", "nardl", "streg", "stcox",
         "ahreg", "didregress", "mixed", "sureg")


@pytest.mark.parametrize("name", CASES)
def test_saved_family_results_preserve_numbers_groups_notes_and_do_not_refit(matrix, name, monkeypatch, tmp_path):
    fitted = matrix[name]
    path = tmp_path / "result.json"
    path.write_text(fitted.model_dump_json())
    saved = ResultBundle.model_validate_json(path.read_text())
    snapshot = saved.model_dump_json()
    def forbidden(*args, **kwargs):
        pytest.fail("Formatting attempted estimation")
    monkeypatch.setattr("openecon.analysis.fit", forbidden)
    monkeypatch.setattr("openecon.econometrics.registry.load_entry", forbidden)
    source = saved.to_latex()
    item = add_output_latex({"type": "model", "data": json.loads(snapshot)})
    assert item["latex"] == source
    assert item["latex_math"] == source.math
    assert item["latex_notes"] == source.notes
    assert item["latex_style"] == PUBLICATION_STYLE
    for coefficient in saved.coefficients:
        # Independently formatted expected reporting units; no formatter helper.
        for number in (coefficient.estimate, coefficient.std_error):
            if abs(number) >= .0001:
                assert f"{number:.4f}" in source and f"{number:.4f}" in source.math
        if coefficient.equation:
            label = coefficient.equation.replace("_", r"\_")
            assert f"[{label}]" in source and f"[{label}]" in source.math
    assert saved.model_dump_json() == snapshot == path.read_text()
    assert item["data"] == json.loads(snapshot)
    # Old publication-v1 histories upgrade on read without rewriting storage.
    old = {"outputs": [{"type": "model", "data": json.loads(snapshot),
                        "latex": "old", "latex_style": "publication-v1"}]}
    baseline = deepcopy(old)
    enriched = enrich_record(old)
    assert old == baseline
    assert enriched["outputs"][0] == item
    assert enrich_record(enriched) == enriched


def test_every_registered_result_family_is_represented(matrix):
    from openecon.econometrics.registry import all_estimators, get
    assert {get(m.spec.estimator).family for m in matrix.values()} == {i.family for i in all_estimators()}


@pytest.mark.parametrize("name,notes", [
    ("xtreg_robust", ["id (36 clusters)", "Student t inference (df = 35)", "vce(robust) = vce(cluster panel)", "absorbed effects: 36"]),
    ("xtreg_dk", ["Driscoll-Kraay", "kernel: bartlett", "lags: 1", "periods: 10", "df = 9"]),
    ("ivregress", ["id (36 clusters), region (6 clusters)", "df = 5", "CR1:"]),
    ("zinb", ["fweight weights: fw", "physical estimation sample 360 of 360", "id (36 clusters), region (6 clusters)", "Vuong test"]),
    ("arima", ["outer product of the per-observation scores", "tested one-sided against zero"]),
    ("stcox", ["Lin-Wei sandwich", "partial likelihood: Breslow", "ties: breslow"]),
    ("streg", ["parameter metric: ph", "outcome distribution: weibull", "log-likelihood scale: log time"]),
    ("ahreg", ["id (36 clusters)", "effective covariance: panel", "excluded lag-window rows: 72"]),
    ("mixed", ["variance parameters: observed information", "likelihood: full (ML)"]),
])
def test_actual_advanced_inference_disclosures(matrix, name, notes):
    source = str(matrix[name].to_latex())
    # TeX escapes literal hyphens/underscores only where necessary.
    for note in notes:
        assert note in source


def test_equation_identity_is_not_overwritten_in_model_comparisons(matrix):
    a = matrix["streg"].model_copy(deep=True)
    b = matrix["streg"].model_copy(deep=True)
    b.coefficients[0].equation = "different outcome"
    b.coefficients[0].estimate = 12.3456
    source = oe.regression_table([a, b])
    first = str(source).split("[duration]", 1)[1].split("[ln", 1)[0]
    assert "12.3456" not in first
    other = str(source).split("[different outcome]", 1)[1]
    assert "Intercept &  & $12.3456" in other


def test_stars_tiny_numbers_warnings_and_empty_fit_metrics_use_saved_values(matrix):
    model = matrix["arima"].model_copy(deep=True)
    model.metrics = {}
    model.warnings = [r"Ücret & ücret_%: \input{file}"]
    model.coefficients = [Coefficient(term=f"p{i}", equation="Türkçe & scale", estimate=1.2e-12,
        std_error=4.8e-14, statistic=25, p_value=p, ci_low=1e-12, ci_high=1.4e-12)
        for i, p in enumerate([.009999, .01, .05, .10])]
    source = model.to_latex()
    assert r"{1.2000 \times 10^{-12}}^{***}" in source
    assert r"{1.2000 \times 10^{-12}}^{**}" in source
    assert r"{1.2000 \times 10^{-12}}^{*}" in source
    assert r"p3 & $1.2000 \times 10^{-12}$" in source
    assert r"4.8000 \times 10^{-14}" in source.math
    assert "R^{2}" not in source and "Log likelihood" not in source
    assert "two-sided tests" not in source
    assert r"Ücret \& ücret\_\%: \textbackslash{}input\{file\}" in source
    assert r"\input{file}" not in source


def test_long_preview_never_separates_group_coefficient_and_standard_error(matrix):
    model = matrix["sureg"].model_copy(deep=True)
    template = model.coefficients[0]
    model.coefficients = [template.model_copy(update={"term": f"term_{i}", "equation": "eq"}) for i in range(90)]
    source = model.to_latex()
    assert "Preview: 49 of" in source.math
    assert r"term\_23" in source.math and r"term\_24" not in source.math
    assert r"term\_89" in source and r"\begin{longtable}" in source
    assert r"{[eq]} &  \\*" in source
    assert r"term\_89 &" in source and "\\\\*" in source
    assert "Other parameters" not in source


def test_custom_numeric_format_agrees_between_preview_and_export(matrix):
    model = matrix["sureg"]
    source = model.to_latex(float_format="%.2f")
    assert f"{model.metrics['y:r_squared']:.2f}" in source.math
    assert f"{model.metrics['y:r_squared']:.2f}" in source


def test_model_titles_never_invent_fit_effects_or_inference(matrix):
    model = matrix["xtreg_robust"].model_copy(deep=True)
    model.title = "Fixed effects with robust clustered uncertainty"
    model.metrics, model.extra, model.inference = {}, {}, {}
    model.spec.covariance, model.spec.cluster = "nonrobust", None
    source = model.to_latex()
    assert "conventional standard errors" in source
    assert "inference distribution unavailable" in source
    assert "clustered by" not in source and "absorbed effects" not in source
    assert "R^{2}" not in source and "Log likelihood" not in source


def test_explicit_comparison_keeps_the_same_notes_in_console_preview(matrix):
    from openecon.console_worker import _execute
    source = oe.regression_table([matrix["streg"], matrix["mixed"]], stars=False)
    record = _execute("display(source)", {"source": source}, "publication-notes")
    assert record["status"] == "ok"
    output, = record["outputs"]
    assert output["latex_notes"] == source.notes
    assert output["latex"] == source and output["latex_math"] == source.math
    assert not any("p < 0.10" in note for note in source.notes)
