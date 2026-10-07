"""Publication exports retain fitted statistical meaning and complete samples."""
from copy import deepcopy
import io

import pytest

import openecon as oe
from openecon.latex import display_latex, table_latex
from openecon.models import Coefficient, ModelSpec, ResultBundle


def coefficient(term, estimate=1.23456, std_error=0.23456, p_value=0.04999):
    return Coefficient(term=term, estimate=estimate, std_error=std_error,
                       statistic=5.0, p_value=p_value, ci_low=estimate - 0.5,
                       ci_high=estimate + 0.5)


def fitted(terms, *, estimator="ols", covariance="HC3", nobs=80, original=82,
           alpha=0.05, metrics=None, cluster=None, intercept=True):
    return ResultBundle(
        id="fitted", created_at="2026-10-02", spec=ModelSpec(
            estimator=estimator, outcome="wage_%", predictors=["education"],
            covariance=covariance, cluster=cluster, alpha=alpha, intercept=intercept),
        nobs=nobs, nobs_original=original, dropped_rows=original - nobs,
        coefficients=terms, covariance_matrix=[],
        metrics=metrics or {"r_squared": 0.72, "adjusted_r_squared": 0.70,
                           "condition_number_scaled": 123.0, "aic": 120.0,
                           "bic": 130.0, "rmse": 4.0},
        warnings=["Keep the warning_1 & its meaning."], predictions=[],
        sample_positions=list(range(nobs)), provenance={"solver": "internal-debug-only"},
        inference={"covariance": covariance, "use_t": estimator == "ols",
                   "df_inference": 7 if cluster else nobs - len(terms),
                   "cluster_column": cluster, "cluster_count": 8 if cluster else None,
                   "correction": "CR1: G/(G-1) * (N-1)/(N-K)" if cluster else covariance,
                   "alpha": alpha, "confidence_level": 1 - alpha},
    )


def test_default_model_export_is_two_line_publication_layout_and_preserves_model_json():
    model = fitted([coefficient("Intercept"), coefficient("education", std_error=0.015)])
    before = model.model_dump_json()
    source = model.to_latex()
    assert source == model.latex == model.summary(format="latex")
    assert r"\caption{Regression results}" in source
    assert r"\toprule" in source and r"\midrule" in source and r"\bottomrule" in source
    assert "Dependent variable: & wage\\_\\%" in source and " & (1)" in source
    assert "education & $1.2346^{**}$" in source
    assert r" & $(0.0150)$ \\" in source
    assert source.index("education &") < source.index("Constant &")
    assert "Observations & 80" in source and "Observations & 80.0000" not in source
    assert r"$R^{2}$ & 0.7200" in source
    assert r"$\text{Adjusted }R^{2}$ & 0.7000" in source
    for hidden in ["condition_number", "internal-debug-only", "aic", "bic", "rmse", "P>|stat|"]:
        assert hidden not in source
    for note in ["HC3 heteroskedasticity-robust", "Student t inference", "95\\% confidence",
                 "alpha = 0.05", "estimation sample 80 of 82 observations",
                 "2 observations excluded from estimation", "Warning:",
                 "* p < 0.10; ** p < 0.05; *** p < 0.01"]:
        assert note in source
    assert model.model_dump_json() == before
    _, math = display_latex(model)
    assert "1.2346^{**}" in math and "(0.0150)" in math
    assert "P>|stat|" not in math


@pytest.mark.parametrize("p,expected", [(0.1, ""), (0.05, "*"), (0.01, "**"),
                                        (0.009999, "***"), (0.049999, "**"),
                                        (0.099999, "*")])
def test_significance_uses_original_strict_thresholds_not_rounded_pvalues(p, expected):
    model = fitted([coefficient("education", estimate=1.2e-12, p_value=p)])
    source = oe.regression_table([model])
    estimate = r"1.2000 \times 10^{-12}"
    assert "$" + ("{" + estimate + "}^{" + expected + "}" if expected else estimate) + "$" in source
    assert ("^{" + expected + "}" in source.math) if expected else ("^{*}" not in source.math)


def test_multiple_models_union_absent_terms_integer_n_and_actual_cluster_notes():
    first = fitted([coefficient("Intercept", p_value=0.8), coefficient("education")])
    second = fitted([coefficient("Intercept"), coefficient("education", estimate=-2.1e-10),
                     coefficient("experience", estimate=0.0, p_value=0.5)],
                    covariance="cluster", cluster="firm_id", nobs=65, original=82,
                    alpha=0.10, metrics={"r_squared": 0.8, "adjusted_r_squared": 0.79})
    before = [model.model_dump_json() for model in [first, second]]
    source = oe.regression_table({"Baseline": first, "Controls & sector": second},
                                label="tab:regression", term_labels={"education": "Education (%)"})
    assert " & (1) & (2)" in source
    assert r"Baseline & Controls \& sector" in source
    assert r"Education (\%) &" in source
    assert "experience &  & $0.0000$" in source
    assert r" &  & $(0.2346)$" in source
    assert "Observations & 80 & 65" in source
    assert r"-2.1000 \times 10^{-10}" in source
    assert "clustered by firm\\_id (8 clusters)" in source
    assert "90\\% confidence level (alpha = 0.1)" in source
    assert "estimation sample 65 of 82 observations" in source
    assert "17 observations excluded from estimation" in source
    assert "(1) OLS:" in source and "(2) OLS:" in source
    assert "Fixed effects" not in source
    assert [model.model_dump_json() for model in [first, second]] == before
    assert "experience}" in source.math and "(0.2346)" in source.math


def test_binary_model_only_uses_appropriate_actual_fit_statistics():
    model = fitted([coefficient("education")], estimator="logit", covariance="nonrobust",
                   metrics={"pseudo_r_squared": 0.19, "log_likelihood": -32.12345,
                            "r_squared": None, "adjusted_r_squared": None,
                            "aic": 1.2, "bic": 2.3})
    source = model.to_latex()
    assert r"$\text{Pseudo }R^{2}$ & 0.1900" in source
    assert "Log likelihood & -32.1234" in source
    assert r"$\text{Adjusted }R^{2}$" not in source
    assert "aic" not in source and "bic" not in source
    assert "LOGIT: conventional standard errors" in source
    assert "normal z inference" in source


def test_stars_false_renamed_outcome_and_user_notes_are_safe_and_do_not_drop_warnings():
    model = fitted([coefficient("education"), coefficient("Intercept")])
    source = oe.regression_table([model], stars=False,
        outcome_labels={"wage_%": r"Öğrenci ücreti & \input{secret}"},
        term_labels={"Intercept": "Sabit"}, notes=["Örnek açıklama_1 & değer."],
        caption="Eğitim & ücret", font_size="footnotesize", max_width=r"0.9\textwidth")
    assert "^{*" not in source and "p <" not in source
    assert "Sabit &" in source
    assert r"\caption{Eğitim \& ücret}" in source
    assert r"Öğrenci ücreti \& \textbackslash{}input\{secret\}" in source
    assert r"\input{secret}" not in source
    assert r"Örnek açıklama\_1 \& değer." in source
    assert "Warning:" in source
    assert r"\footnotesize" in source and "max width=0.9\\textwidth" in source


def test_diagnostic_option_keeps_detailed_inference_available_and_file_exports_exact(tmp_path):
    model = fitted([coefficient("education")])
    diagnostic = model.to_latex(style="diagnostic")
    assert "Estimate & Std. error & t & P>|stat| & CI lower & CI upper" in diagnostic
    assert r"condition\_number\_scaled" in diagnostic
    assert "AIC" in model.summary() or "aic:" in model.summary()
    file = tmp_path / "table.tex"
    assert oe.regression_table([model], file) is None
    assert file.read_text() == oe.regression_table([model])
    output = io.StringIO()
    assert oe.regression_table([model], output, precision=3) is None
    assert "1.235" in output.getvalue()


def test_dataframe_publication_defaults_preserve_numeric_zero_bool_index_and_notes():
    frame = oe.DataFrame({"All zero": [0.0, 0.0], "Binary": [0, 1], "Boolean": [False, True]},
                         index=oe.DataFrame({"label": ["Ç_1", "Ü_2"]})["label"])
    before = deepcopy(frame.attrs)
    source = frame.to_latex(caption="Sample & summary", label="tab:data", notes=["Values remain numeric."])
    assert r"\begin{tabular}{@{}lrrl@{}}" in source
    assert r"Ç\_1 & 0.0000 & 0 & False" in source
    assert r"Ü\_2 & 0.0000 & 1 & True" in source
    assert r"\begin{minipage}{\linewidth}" in source
    assert "Values remain numeric." in source
    assert frame.attrs == before
    record_source = table_latex(["Zero", "Binary", "Boolean"], [[0, 0, False], [0, 1, True]])
    assert r"\begin{tabular}{@{}rrl@{}}" in record_source


def test_long_dataframe_auto_pages_repeats_header_and_never_boxes_longtable():
    frame = oe.DataFrame({"x": range(120), "y": range(120)}, index=[f"firm_{i}" for i in range(120)])
    source = frame.to_latex(caption="Long sample", label="tab:long", notes="Complete rows.")
    assert r"\begin{longtable}" in source
    assert r"\endfirsthead" in source and r"\endhead" in source
    assert r"\endfoot" in source and r"\endlastfoot" in source
    assert r"\begin{adjustbox}" not in source
    assert r"firm\_119 & 119 & 119" in source
    assert source.count(" & x & y") == 2
    assert "Complete rows." in source
    short_requested = frame.to_latex(longtable=False)
    assert r"\begin{longtable}" not in short_requested
    assert r"\begin{adjustbox}" in short_requested


def test_no_intercept_r_squared_is_labelled_as_uncentered_in_actual_notes():
    model = fitted([coefficient("education")], intercept=False)
    source = model.to_latex()
    assert "no intercept; R-squared is uncentered" in source
    assert "Constant &" not in source


@pytest.mark.parametrize("options", [
    {"font_size": r"small}\input{file}"}, {"max_width": r"\linewidth}\input{file}"},
    {"stars": "yes"}, {"term_labels": {"x": 5}}, {"outcome_labels": []},
    {"longtable": "auto"}, {"precision": -1},
])
def test_invalid_publication_options_cannot_emit_untrusted_commands(options):
    with pytest.raises((ValueError, TypeError)):
        oe.regression_table([fitted([coefficient("education")])], **options)


def test_comparison_requires_fitted_models_and_rejects_duplicate_terms():
    for value in [[], {}, [1], "not a model"]:
        with pytest.raises((ValueError, TypeError)):
            oe.regression_table(value)
    model = fitted([coefficient("education"), coefficient("education")])
    with pytest.raises(ValueError, match="duplicate"):
        oe.regression_table([model])


def test_long_regression_pairs_prevent_page_break_between_coefficient_and_standard_error():
    terms = [coefficient(f"x_{position}") for position in range(25)]
    source = oe.regression_table([fitted(terms)])
    assert r"\begin{longtable}" in source
    assert source.count(r"\\*") == 25
    assert source.count(r"$(0.2346)$ \\[0.35em]") == 25
    assert r"\begin{adjustbox}" not in source


def test_real_predictors_named_constant_are_not_relabelled_as_the_intercept():
    model = fitted([coefficient("Intercept"), coefficient("constant"),
                    coefficient("const"), coefficient("_cons")])
    source = model.to_latex()
    assert source.count("Constant &") == 1
    assert "constant &" in source and "const &" in source and r"\_cons &" in source
    assert source.index("constant &") < source.index("Constant &")
    no_intercept = fitted([coefficient("Intercept")], intercept=False)
    source = no_intercept.to_latex()
    assert "Intercept &" in source and "Constant &" not in source


def test_wide_longtable_has_bounded_paragraph_columns_without_altering_numeric_cells():
    columns = {"Description": ["A substantial explanation with complete words."] * 75,
               "Grouping": ["a_very_long_unbroken_identifier_12345"] * 75,
               **{f"Numeric {position}": [float(row) for row in range(75)] for position in range(6)}}
    frame = oe.DataFrame(columns)
    source = frame.to_latex(index=False, caption="Wide appendix", notes="All original rows included.")
    assert r"\begin{longtable}" in source
    assert r"\dimexpr\linewidth/8-2\tabcolsep\relax" in source
    assert r"\raggedright\arraybackslash" in source
    assert r"\raggedleft\arraybackslash" in source
    assert r"\allowbreak{}" in source
    assert "74.0000" in source and "0.0000" in source
    assert r"\begin{adjustbox}" not in source
    assert source.count("74.0000") == 6
    assert r"\ifdim\tabcolsep>\dimexpr\linewidth/32\relax" in source
    explicit = frame.to_latex(index=False, column_format="llrrrrrr")
    assert r"\begin{longtable}{llrrrrrr}" in explicit
    assert r"\arraybackslash" not in explicit


def test_extremely_wide_longtable_guards_positive_budget_and_zero_width_is_rejected():
    frame = oe.DataFrame({f"x{position}": [1] for position in range(250)})
    source = frame.to_latex(index=False, longtable=True, max_width="1cm")
    assert r"\ifdim\tabcolsep>\dimexpr1cm/1000\relax" in source
    assert r"\setlength{\tabcolsep}{\dimexpr1cm/1000\relax}" in source
    assert r"\dimexpr1cm/250-2\tabcolsep\relax" in source
    with pytest.raises(ValueError, match="positive"):
        frame.to_latex(max_width=r"0\linewidth")
