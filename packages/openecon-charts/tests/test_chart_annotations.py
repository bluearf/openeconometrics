"""Annotations are validated presentation layers, never chart observations."""
from io import StringIO
import json
import re

import pytest

import openecon_charts as charts


def _scatter(**options):
    return charts.scatter(data={"x": [1, 10, 100], "y": [2, 20, 200]}, x="x", y="y", **options)


def _plots(**options):
    numeric = {"x": [1, 10, 100], "y": [2, None, 8]}
    category = {"category": ["A", "B"], "a": [2, 3], "b": [4, 5]}
    return [
        charts.scatter(data=numeric, x="x", y="y", **options),
        charts.line(data=numeric, x="x", y="y", **options),
        charts.hist(data=numeric, x="x", bins=3, **options),
        charts.coefficients({"coefficients": [{"term": "Education", "estimate": 1,
                                               "ci_low": .5, "ci_high": 1.5}]}, **options),
        charts.bar(data=category, x="category", y=["a", "b"], **options),
        charts.barh(data=category, x="category", y=["a", "b"], **options),
        charts.area(data=category, x="category", y=["a", "b"], **options),
        charts.stacked_bar(data=category, x="category", y=["a", "b"], **options),
        charts.donut(data=category, labels="category", values="a", **options),
    ]


def _notes(plot):
    return plot.model_dump()["config"]["options"]["annotations"]


def test_all_nine_helpers_accept_relative_notes_without_changing_data_or_counts():
    note = {"type": "text", "text": "Source: survey", "x": .05, "y": .9, "coords": "axes"}
    for original, annotated in zip(_plots(), _plots(annotations=[note])):
        old, new = original.model_dump(), annotated.model_dump()
        assert len(new["config"]["options"]["annotations"]) == 1
        del new["config"]["options"]
        if old["config"] is None:
            new["config"] = None
        assert new == old
        assert r"\node[anchor=west,text=oeann0" in annotated.to_latex()
    assert note == {"type": "text", "text": "Source: survey", "x": .05, "y": .9, "coords": "axes"}


def test_six_fluent_types_have_canonical_styles_and_append_in_user_order():
    plot = (_scatter().annotate("Plain", x=10, y=20).annotate("Arrow", x=10, y=20, arrow=True)
            .vline(10).hline(20).vspan(5, 50).hspan(5, 50))
    notes = _notes(plot)
    assert [note["type"] for note in notes] == ["text", "arrow", "vline", "hline", "vspan", "hspan"]
    assert all(note["coords"] == "data" and note["color"] == "#162d4a" for note in notes)
    assert (notes[0]["dx"], notes[0]["dy"], notes[0]["font_size"], notes[0]["align"]) == (0, 0, 12, "left")
    assert (notes[1]["dx"], notes[1]["dy"], notes[1]["dash"], notes[1]["line_width"]) == (12, -12, "solid", 1.5)
    assert notes[2]["dash"] == notes[3]["dash"] == "dashed"
    assert [note["opacity"] for note in notes] == [1, 1, 1, 1, .12, .12]
    assert "font_size" not in notes[2] and "line_width" not in notes[4]


def test_deep_independent_copies_and_replacement_clear_annotations():
    source = _scatter(color="#123")
    first = source.annotate("Original", x=10, y=20)
    second = first.vline(10, text="Threshold")
    replacement = [{"type": "text", "text": "Replacement", "x": .5, "y": .5, "coords": "axes"}]
    third = second.with_options(annotations=replacement)
    replacement[0]["text"] = "Changed outside"
    third.config["options"]["annotations"][0]["text"] = "Changed copy"
    third.data[0]["x"] = 999
    assert source.config["options"] == {"color": "#112233"}
    assert len(_notes(first)) == 1 and len(_notes(second)) == 2
    assert _notes(second)[0]["text"] == "Original" and second.data[0]["x"] == 1
    cleared = second.with_options(annotations=[])
    assert _notes(cleared) == [] and cleared.to_latex() == source.to_latex()


@pytest.mark.parametrize("helper", [charts.scatter, charts.line])
def test_numeric_log_annotations_share_data_domains_and_reject_nonpositive_coordinates(helper):
    plot = helper(data={"x": [1, 10, 100], "y": [2, 20, 200]}, x="x", y="y",
                  x_scale="log", y_scale="log")
    styled = plot.annotate("Middle", x=10, y=20, arrow=True).vline(10).hspan(5, 50)
    assert styled.data == plot.data
    assert "(axis description cs:0.5,0.5)" in styled.to_latex()
    for action in [lambda: plot.annotate("Bad", x=0, y=20), lambda: plot.vline(-1),
                   lambda: plot.hline(0), lambda: plot.vspan(0, 10), lambda: plot.hspan(-1, 10)]:
        with pytest.raises(ValueError, match="strictly positive"):
            action()
    assert plot.annotate("Relative", x=0, y=0, coords="axes")


def test_finite_float_large_coordinates_match_chart_data_and_raw_unsafe_integers_fail():
    plot = charts.scatter(data={"x": [1e19, 1e20], "y": [1., 10.]}, x="x", y="y", x_scale="log")
    assert _notes(plot.annotate("Large float", x=1e20, y=10.))[0]["x"] == 1e20
    assert "axis description cs:1.0,1.0)" in plot.annotate("Large float", x=1e20, y=10.).to_latex()
    for value in [2**53, 10**1000, float("inf"), float("nan"), True, "10"]:
        with pytest.raises(ValueError):
            plot.annotate("Invalid", x=value, y=10)
    assert _notes(_scatter().vline(2**53 - 1))[0]["x"] == float(2**53 - 1)


def test_log_anchors_do_not_collapse_for_nearby_large_floats():
    # Subtracting separately rounded logarithms would divide by zero here.
    lower, upper = 1e20, 1e20 + 16384
    plot = charts.scatter(data={"x": [lower, upper], "y": [1, 2]}, x="x", y="y", x_scale="log")
    assert "axis description cs:1.0,1.0)" in plot.annotate("Close floats", x=upper, y=2).to_latex()


def test_physical_category_and_coefficient_coordinates_map_exact_terms_and_reversed_y():
    _, _, _, coeff, bar, barh, area, stacked, donut = _plots()
    for plot in [bar, area, stacked]:
        source = plot.annotate("B note", x="B", y=3).hline(3).to_latex()
        assert "axis description cs:0.75," in source
        with pytest.raises(ValueError, match="numeric x"):
            plot.vline(1)
        with pytest.raises(ValueError, match="exactly match"):
            plot.annotate("Missing", x="b", y=3)
    source = barh.annotate("B note", x=3, y="B").vline(3).to_latex()
    assert ",0.25)" in source
    with pytest.raises(ValueError, match="numeric y"):
        barh.hline(3)
    source = coeff.annotate("Effect", x=1, y="Education", arrow=True).vspan(.5, 1.5).to_latex()
    assert ",0.5)" in source
    with pytest.raises(ValueError, match="exactly match"):
        coeff.annotate("Missing", x=1, y="education")
    for action in [lambda: donut.annotate("Bad", x=0, y=0), lambda: donut.vline(1)]:
        with pytest.raises(ValueError, match="axes coordinates"):
            action()


def test_annotation_defaults_are_normalized_for_raw_specs_and_revalidated_after_mutation():
    plain = _scatter()
    config = {"options": {"annotations": [{"type": "text", "text": "Raw", "x": 10, "y": 20}]}}
    raw = charts.PlotSpec(plain.kind, plain.title, plain.x_label, plain.y_label,
                          plain.data, plain.sample_n, plain.total_n, config=config)
    assert _notes(raw)[0]["font_size"] == 12
    assert r"\definecolor{oeann0}{HTML}{162d4a}" in raw.to_latex()
    raw.config["options"]["annotations"][0]["text"] = "bad\x00"
    stream = StringIO()
    for action in [raw.model_dump, raw.to_html, lambda: raw.to_latex(stream), raw.with_options]:
        with pytest.raises(ValueError, match="control"):
            action()
    assert stream.getvalue() == ""


@pytest.mark.parametrize("entry", [
    {}, {"type": "unknown"}, {"type": []}, {"type": "text", "x": 1, "y": 2},
    {"type": "text", "text": "Missing", "x": 1},
    {"type": "text", "text": "Bad", "x": 1, "y": 2, "url": "javascript:x"},
    {"type": "text", "text": "Bad", "x": 1, "y": 2, "coords": "figure"},
    {"type": "text", "text": "Bad", "x": 1.1, "y": .5, "coords": "axes"},
    {"type": "arrow", "text": "Bad", "x": .5, "y": -.1, "coords": "axes"},
    {"type": "vline", "x": 1, "coords": "axes"}, {"type": "hspan", "y0": 1, "y1": 2, "coords": "axes"},
    {"type": "vspan", "x0": 2, "x1": 1}, {"type": "hspan", "y0": 1, "y1": 1},
    {"type": "vspan", "x0": 1, "x1": 2, "line_width": 2},
    {"type": "hspan", "y0": 1, "y1": 2, "dash": "dashed"},
    {"type": "vline", "x": 1, "font_size": 12},
    {"type": "vline", "x": 1, "color": "red"}, {"type": "vline", "x": 1, "opacity": None},
    {"type": "vline", "x": 1, "opacity": 1.1}, {"type": "vline", "x": 1, "line_width": .1},
    {"type": "vline", "x": 1, "dash": "url()"},
    {"type": "text", "text": "Bad", "x": 1, "y": 2, "font_size": 37},
    {"type": "text", "text": "Bad", "x": 1, "y": 2, "align": "justify"},
    {"type": "text", "text": "Bad", "x": 1, "y": 2, "dx": -501},
    {"type": "arrow", "text": "Bad", "x": 1, "y": 2, "dy": 501},
    {"type": "text", "text": "Bad", "x": 1, "y": 2, "line_width": 2},
])
def test_strict_fields_coordinates_and_styles_reject_malformed_annotations(entry):
    with pytest.raises((TypeError, ValueError)):
        _scatter(annotations=[entry])


def test_type_cannot_be_overridden_through_fluent_style():
    plot = _scatter()
    for action in [lambda: plot.annotate("Bad", x=1, y=2, type="arrow"),
                   lambda: plot.vline(1, type="hline"), lambda: plot.hline(1, type="vline"),
                   lambda: plot.vspan(1, 2, type="hspan"), lambda: plot.hspan(1, 2, type="vspan")]:
        with pytest.raises(ValueError, match="type"):
            action()
    with pytest.raises(TypeError, match="boolean"):
        plot.annotate("Bad", x=1, y=2, arrow=1)


@pytest.mark.parametrize("text", ["A" * 501, "\n", "\t", "\x00", "\x7f", "\u0085", "\ud800", None, 5])
def test_text_bound_and_unicode_scalar_validation(text):
    with pytest.raises(ValueError):
        _scatter().annotate(text, x=1, y=2)


def test_emoji_scalar_length_empty_text_and_all_annotation_bounds():
    assert _notes(_scatter().annotate("😀" * 500, x=1, y=2))[0]["text"] == "😀" * 500
    assert _notes(_scatter().annotate("", x=1, y=2))[0]["text"] == ""
    valid = _scatter().annotate("Boundary", x=0, y=1, coords="axes", font_size=36, dx=-500, dy=500,
                               color="#abc", opacity=0, align="right")
    assert _notes(valid)[0]["color"] == "#aabbcc"
    for annotations in (None, {}, [None], [{"type": "vline", "x": 1}] * 101):
        with pytest.raises((TypeError, ValueError)):
            _scatter(annotations=annotations)
    assert len(_notes(_scatter(annotations=[{"type": "vline", "x": 1}] * 100))) == 100


def test_categories_with_control_characters_cannot_be_annotation_anchors():
    plot = charts.bar(data={"category": ["A\x00"], "value": [2]}, x="category", y="value")
    with pytest.raises(ValueError, match="control"):
        plot.annotate("Bad category", x="A\x00", y=1)


def test_latex_text_styles_offsets_arrow_direction_and_data_are_preserved():
    text = r"Türkçe & 50% _ {x} $ \input{secret} <script>"
    plain = _scatter(width=600, height=360, xlim=[1, 100], ylim=[2, 200])
    plot = plain.annotate(text, x=10, y=20, arrow=True, dx=24, dy=-32,
                          color="#abc", font_size=16, align="right", line_width=2, dash="dotted", opacity=.6)
    source = plot.to_latex()
    assert r"\definecolor{oeann0}{HTML}{aabbcc}" in source
    assert "xshift=18.0675pt,yshift=24.09pt" in source
    assert "line width=1.505625pt,dotted" in source
    assert "font=\\fontsize{12.045pt}" in source
    assert "anchor=east,text=oeann0,opacity=0.6" in source
    assert "-- (oeannanchor0);" in source
    active = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
    assert r"Türkçe \& 50\% \_ \{x\} \$ \textbackslash{}input\{secret\}" in active
    assert r"\input{secret}" not in active
    assert "(1.0,2.0) (10.0,20.0) (100.0,200.0)" in source
    assert plot.data == plain.data and plot.model_dump()["config"]["options"]["xlim"] == [1., 100.]
    html = plot.to_html()
    payload = re.search(r'<script type="application/json" id="chart-data">(.*?)</script>', html).group(1)
    assert "\\u003cscript\\u003e" in payload and '<script>' not in payload
    assert json.loads(payload)["config"]["options"]["annotations"][0]["text"] == text


def test_latex_bands_are_behind_marks_labels_opaque_and_limits_stay_fixed():
    plain = _scatter(width=600, height=360, xlim=[1, 100], ylim=[2, 200])
    source = plain.vspan(-1e300, 50, text="Visible band", opacity=.2).hspan(20, 1e300).to_latex()
    assert "(axis description cs:0.0,0) rectangle" in source
    assert "fill opacity=0.2" in source
    assert "text=oeann0,opacity=1.0" in source
    assert source.index(r"\path[fill=oeann0") < source.index(r"\addplot[only marks")
    assert "execute at end axis={" in source and "\\begin{scope}[overlay]" in source
    assert source.count(r"\clip (axis description cs:0,0) rectangle (axis description cs:1,1);") == 3
    for limit in ["xmin=1.0", "xmax=100.0", "ymin=2.0", "ymax=200.0"]:
        assert limit in source
    assert "(axis description cs:1e" not in source


def test_donut_relative_notes_use_full_picture_inset_and_keep_uniform_circle_scaling():
    plain = _plots()[-1].with_options(width=600, height=400)
    source = plain.annotate("Relative center", x=.5, y=.5, coords="axes", align="center").to_latex()
    assert "overlay,reset cm,transform shape=false" in source
    assert "xshift=15.05625pt,yshift=15.05625pt]current bounding box.south west" in source
    assert "$(oeannbl)!0.5!(oeannbr)$" in source and "$(oeannbl)!0.5!(oeanntl)$" in source
    assert r"\clip (oeannbl) rectangle (oeanntr);" in source
    assert re.search(r"\\begin\{tikzpicture\}\[scale=[0-9.]+,transform shape\]", source)
    assert re.findall(r"arc\[.*?\]", source) == re.findall(r"arc\[.*?\]", plain.to_latex())


def test_empty_annotation_list_preserves_every_legacy_latex_export():
    for plot in _plots():
        assert plot.to_latex() == plot.with_options(annotations=[]).to_latex()
        assert "oeann" not in plot.to_latex()


def test_increasing_spans_crossing_extreme_finite_values_clip_without_overflow():
    plot = _scatter().vspan(-1e308, 1e308).hspan(-1e308, 1e308)
    source = plot.to_latex()
    assert "nan" not in source.lower() and "inf" not in source.lower()
    assert "(axis description cs:0.0,0) rectangle (axis description cs:1.0,1)" in source
    assert "(axis description cs:0,0.0) rectangle (axis description cs:1,1.0)" in source


def test_log_option_change_revalidates_existing_annotation_coords():
    plot = _scatter().annotate("Zero anchor", x=0, y=20)
    with pytest.raises(ValueError, match="strictly positive"):
        plot.with_options(x_scale="log")
    assert plot.data == _scatter().data


@pytest.mark.parametrize("arrow", [False, True])
def test_outside_anchor_is_suppressed_but_inbounds_offset_is_clipped(arrow):
    plot = charts.scatter(data={"x": [0, 1], "y": [0, 1]}, x="x", y="y", xlim=[0, 1], ylim=[0, 1])
    outside = plot.annotate("Outside anchor", x=1.01, y=.5, dx=-20, arrow=arrow).to_latex()
    assert "Outside anchor" not in outside and "oeannanchor0" not in outside
    inside = plot.annotate("Inside anchor", x=.01, y=.5, dx=-20, arrow=arrow).to_latex()
    assert "Inside anchor" in inside and "xshift=-15.05625pt" in inside
    assert r"\clip (axis description cs:0,0) rectangle (axis description cs:1,1);" in inside
    assert "xmin=0.0" in outside and "xmax=1.0" in outside


@pytest.mark.parametrize("kind", ["vline", "hline"])
def test_reference_outside_viewport_cannot_return_its_label_with_an_offset(kind):
    plot = charts.scatter(data={"x": [0, 1], "y": [0, 1]}, x="x", y="y", xlim=[0, 1], ylim=[0, 1])
    outside = getattr(plot, kind)(1.01, text="Outside reference", dx=-20, dy=20).to_latex()
    assert "Outside reference" not in outside and "oeannanchor0" not in outside
    inside = getattr(plot, kind)(.99, text="Inside reference", dx=-20, dy=20).to_latex()
    assert "Inside reference" in inside and "xshift=-15.05625pt" in inside
