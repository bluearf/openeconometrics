"""LaTeX export checks for observation, interval, bin, and category semantics."""
from io import StringIO
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

import openecon_charts as charts


def _plots():
    numeric = {"x": [1, 3, 10], "y": [-2, 4, 6]}
    categorical = {"year": ["2021", "2024"], "a": [2, 3], "b": [4, 5]}
    model = {"coefficients": [
        {"term": "education", "estimate": 1.2, "ci_low": .8, "ci_high": 1.7},
        {"term": "experience", "estimate": -.2, "ci_low": -.5, "ci_high": .1},
    ]}
    return [
        charts.scatter(data=numeric, x="x", y="y"),
        charts.line(data=numeric, x="x", y="y"),
        charts.hist(data=numeric, x="x", bins=2),
        charts.coefficients(model),
        charts.bar(data=categorical, x="year", y=["a", "b"]),
        charts.barh(data=categorical, x="year", y=["a", "b"]),
        charts.area(data=categorical, x="year", y=["a", "b"]),
        charts.stacked_bar(data=categorical, x="year", y=["a", "b"]),
        charts.stacked_bar(data=categorical, x="year", y=["a", "b"], horizontal=True),
        charts.donut(data=categorical, labels="year", values="a"),
    ]


@pytest.mark.parametrize("plot", _plots(), ids=lambda p: p.kind + (
    ":" + p.config["type"] if p.config and "type" in p.config else ""))
def test_every_chart_exports_a_complete_figure_and_standalone_document(plot):
    fragment = plot.to_latex()
    assert fragment.count(r"\begin{figure}") == fragment.count(r"\end{figure}") == 1
    assert fragment.count(r"\begin{tikzpicture}") == fragment.count(r"\end{tikzpicture}") == 1
    assert r"\definecolor{oecolor0}{HTML}{162D4A}" in fragment
    assert plot.latex == fragment
    document = plot.to_latex(standalone=True)
    assert document.count(r"\begin{document}") == document.count(r"\end{document}") == 1
    for package in ("tikz", "pgfplots", "fontspec"):
        assert rf"\usepackage{{{package}}}" in document
    metadata = json.loads(fragment.splitlines()[0].removeprefix("% OpenEcon chart: "))
    assert metadata == {name: getattr(plot, name) for name in ("kind", "sample_n", "total_n", "dropped_n")}
    assert "nan" not in fragment.lower()


def test_scatter_exports_only_the_displayed_sample_with_round_trip_precision():
    plot = charts.scatter(data={"x": list(range(5000)) + [None],
                               "y": [i / 7 for i in range(5000)] + [1]}, x="x", y="y")
    source = plot.to_latex()
    coordinate_source = re.search(r"coordinates \{(.*?)\};", source, re.S).group(1)
    coordinates = [(float(x), float(y)) for x, y in re.findall(r"\(([^,]+),([^\)]+)\)", coordinate_source)]
    assert coordinates == [(row["x"], row["y"]) for row in plot.data]
    assert len(coordinates) == 2000
    assert '"sample_n": 2000, "total_n": 5000, "dropped_n": 1' in source


def test_line_missing_x_and_y_observations_break_paths_without_reindexing():
    plot = charts.line(data={"x": [1, 5, 10, None, 20, 25],
                            "y": [2, None, 3, 100, 6, 8]}, x="x", y="y")
    source = plot.to_latex()
    paths = re.findall(r"coordinates \{(.*?)\};", source, re.S)
    assert paths == ["(1.0,2.0)", "(10.0,3.0)", "(20.0,6.0) (25.0,8.0)"]
    assert "(5.0,0)" not in source
    assert "(None," not in source
    assert "xmin=1.0" in source and "xmax=25.0" in source


def test_histogram_uses_the_original_bin_edges_and_count_exactly():
    plot = charts.hist(data={"x": [-2, -1, 0, 1, 2, None]}, x="x", bins=4)
    source = plot.to_latex()
    rectangles = re.findall(r"\(axis cs:([^,]+),0\) rectangle \(axis cs:([^,]+),([^\)]+)\)", source)
    assert [(float(x0), float(x1), float(count)) for x0, x1, count in rectangles] == [
        (row["x0"], row["x1"], row["count"]) for row in plot.data]
    assert "ymin=0" in source and "ymax=2" in source


def test_coefficients_preserve_confidence_endpoints_even_outside_the_estimate():
    plot = charts.coefficients({"coefficients": [
        {"term": "a_b", "estimate": 1.25, "ci_low": -.35, "ci_high": 1.75},
        {"term": "outside", "estimate": -3, "ci_low": 1, "ci_high": 2},
    ]})
    source = plot.to_latex()
    assert "(axis cs:-0.35,0) -- (axis cs:1.75,0)" in source
    assert "(axis cs:1.0,1) -- (axis cs:2.0,1)" in source
    assert "coordinates {(1.25,0) (-3.0,1)}" in source
    assert "yticklabels={{a\\_b},{outside}}" in source
    assert "y dir=reverse" in source
    assert "yshift=-2pt" in source and "yshift=2pt" in source
    assert "xmin=-3.0" in source and "xmax=2.0" in source


@pytest.mark.parametrize("helper,horizontal", [(charts.bar, False), (charts.barh, True)])
def test_signed_bars_and_missing_values_keep_zero_baselines_and_category_labels(helper, horizontal):
    plot = helper(data={"year": [2021, 2024, 2028], "a": [-3, None, 4], "b": [2, 5, 1]},
                  x="year", y=["a", "b"], palette=["#abc", "#123456"])
    source = plot.to_latex()
    assert source.count(" rectangle ") == 5
    assert "ticklabels={{2021},{2024},{2028}}" in source
    assert r"\definecolor{oecolor0}{HTML}{aabbcc}" in source
    assert r"\definecolor{oecolor1}{HTML}{123456}" in source
    assert source.count(r"\addlegendentry") == 2
    if horizontal:
        assert "xmin=-3.0" in source and "xmax=5.0" in source
        assert "(axis cs:0," in source
        assert "(axis cs:-3.0," in source
        assert "xlabel={}" in source and "ylabel={year}" in source
    else:
        assert "ymin=-3.0" in source and "ymax=5.0" in source
        assert ",-3.0);" in source


def test_area_missing_values_split_both_fill_and_line_paths():
    plot = charts.area(data={"year": ["A", "B", "C", "D", "E"], "value": [2, 4, None, -3, -1]},
                       x="year", y="value")
    source = plot.to_latex()
    paths = re.findall(r"coordinates \{(.*?)\}", source, re.S)
    assert paths == ["(0,2.0) (1,4.0) (1,0) (0,0)", "(0,2.0) (1,4.0)",
                     "(3,-3.0) (4,-1.0) (4,0) (3,0)", "(3,-3.0) (4,-1.0)"]
    assert not any("(2," in path for path in paths)
    assert "ymin=-3.0" in source and "ymax=4.0" in source
    assert "Area closure coordinates are the zero baseline, not observations" in source


@pytest.mark.parametrize("horizontal", [False, True])
def test_stacked_bars_use_supplied_values_and_complete_component_totals(horizontal):
    plot = charts.stacked_bar(data={"cat": ["A", "B"], "a": [2, 3], "b": [4, 5]},
                              x="cat", y=["a", "b"], horizontal=horizontal)
    source = plot.to_latex()
    assert source.count(" rectangle ") == 4
    if horizontal:
        assert "xmin=0" in source and "xmax=8.0" in source
        assert "(axis cs:2.0," in source and "(axis cs:6.0," in source
        assert "(axis cs:3.0," in source and "(axis cs:8.0," in source
    else:
        assert "ymin=0" in source and "ymax=8.0" in source
        assert ",2.0) rectangle" in source and ",6.0);" in source
        assert ",3.0) rectangle" in source and ",8.0);" in source


def test_stacked_last_endpoint_matches_correctly_rounded_total():
    plot = charts.stacked_bar(data={"cat": ["A"], "a": [1e16], "b": [1.0], "c": [1.0]},
                              x="cat", y=["a", "b", "c"])
    source = plot.to_latex()
    assert "ymax=1.0000000000000002e+16" in source
    assert ",1.0000000000000002e+16);" in source


def test_donut_wedges_have_exact_proportions_values_palette_and_zero_components():
    plot = charts.donut(data={"cat": ["A", "B", "C"], "v": [1, 3, 0]}, labels="cat", values="v",
                        unit="kg", palette=["#abc", "#def", "#123"])
    source = plot.to_latex()
    assert source.count(r"\path[fill=") == 2
    assert "start angle=90.0,end angle=0.0,radius=2" in source
    assert "start angle=0.0,end angle=-270.0,radius=2" in source
    assert r"A: \texttt{1.0}" in source and r"B: \texttt{3.0}" in source
    assert r"C: \texttt{0.0}" in source
    assert r"Total\\\texttt{4.0}\\kg" in source
    assert r"\definecolor{oecolor2}{HTML}{112233}" in source
    assert "pgf-pie" not in source


def test_tex_labels_caption_and_title_escape_special_characters_and_preserve_unicode():
    text = "Türkçe Öğrenci \\input{secret} 50% & $x # _ ^ ~\ncomma,A"
    plot = charts.bar(data={"category": [text], "value": [1]}, x="category", y="value", title=text)
    source = plot.to_latex(caption=text, label="fig:turkish_1")
    escaped = (r"Türkçe Öğrenci \textbackslash{}input\{secret\} 50\% \& \$x \# \_ "
               r"\textasciicircum{} \textasciitilde{} comma,A")
    assert f"title={{{escaped}}}" in source
    assert f"xticklabels={{{{{escaped}}}}}" in source
    assert f"\\caption{{{escaped}}}" in source
    assert r"\label{fig:turkish_1}" in source
    assert r"\input{secret}" not in source


def test_label_without_caption_uses_title_to_make_a_numbered_reference():
    plot = charts.hist(data={"x": [1, 2]}, x="x", title="50%", bins=1)
    assert r"\caption{50\%}" in plot.to_latex(label="fig:hist")
    for invalid in (r"foo}\input{secret}", "label with spaces", "", 1):
        with pytest.raises(ValueError, match="label"):
            plot.to_latex(label=invalid)


def test_to_latex_writes_utf8_paths_or_streams_and_always_returns_source(tmp_path):
    plot = charts.line(data={"x": [1, 2], "y": [3, 4]}, x="x", y="y", title="Öğrenci")
    buffer = StringIO()
    source = plot.to_latex(buffer)
    assert buffer.getvalue() == source
    destination = tmp_path / "figür.tex"
    saved = plot.to_latex(destination, standalone=True)
    assert destination.read_text(encoding="utf-8") == saved


def test_mutated_nonfinite_and_invalid_types_fail_before_output_is_written():
    plot = charts.scatter(data={"x": [1], "y": [2]}, x="x", y="y")
    output = StringIO()
    plot.data[0]["x"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        plot.to_latex(output)
    assert not output.getvalue()
    fresh = charts.scatter(data={"x": [1], "y": [2]}, x="x", y="y")
    with pytest.raises(TypeError, match="standalone"):
        fresh.to_latex(standalone=1)
    with pytest.raises(ValueError, match="control characters"):
        fresh.to_latex(caption="nul\x00byte")


def test_mutated_stack_and_donut_do_not_turn_missing_or_negative_components_into_zero():
    for helper in ("stacked", "donut"):
        data = {"cat": ["A", "B"], "v": [1, 2]}
        plot = (charts.stacked_bar(data=data, x="cat", y="v") if helper == "stacked"
                else charts.donut(data=data, labels="cat", values="v"))
        for value in (None, -1):
            plot.config["series"][0]["values"][1] = value
            with pytest.raises(ValueError, match="complete, nonnegative"):
                plot.to_latex()


def test_latex_export_runs_without_site_packages_or_the_openecon_framework():
    package_path = Path(__file__).parents[1] / "src"
    source = (
        "import sys; "
        f"sys.path.insert(0, {str(package_path)!r}); "
        "import openecon_charts as charts; "
        "plot = charts.scatter(data={'x': [1], 'y': [2]}, x='x', y='y'); "
        "assert 'tikzpicture' in plot.to_latex(); "
        "assert not set(('openecon', 'torch', 'numpy', 'pandas', 'scipy')) & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-S", "-c", source], check=True, timeout=10,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
