"""Complete, safe exports without altering dataframe or estimator semantics."""
import io
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

import openecon as oe
from openecon.data import frame_hash
from openecon.frame import as_frame
from openecon.latex import display_latex, escape_latex, table_latex, table_math
from openecon.models import Coefficient, ModelSpec, ResultBundle
from openecon.workspace import Workspace


def model(covariance=None):
    return ResultBundle(
        id="test", created_at="2026-10-02", spec=ModelSpec(outcome="wage_%", predictors=["x"], covariance=covariance),
        nobs=120, nobs_original=122, dropped_rows=2,
        coefficients=[Coefficient(term="x_1 & y", estimate=1.234567,
                                  std_error=0.01, statistic=123.4567, p_value=1.2e-12,
                                  ci_low=1.2, ci_high=1.3)],
        covariance_matrix=[[0.0001]], metrics={"r_squared": 0.75, "omitted": None},
        warnings=[r"Use x_1, not \input{secret}."], predictions=[], sample_positions=[],
        provenance={}, inference={"use_t": True},
    )


def test_text_escaping_is_literal_unicode_and_commands_do_not_survive():
    value = r"Türkçe: Çğışöü & % $ # _ { } ~ ^ \input{/etc/passwd}"
    escaped = escape_latex(value)
    for token in [r"\&", r"\%", r"\$", r"\#", r"\_", r"\{", r"\}",
                  r"\textasciitilde{}", r"\textasciicircum{}", r"\textbackslash{}"]:
        assert token in escaped
    assert "Türkçe: Çğışöü" in escaped
    assert r"\input{" not in escaped
    frame = oe.DataFrame({"unsafe_column_%": [value, "first\nsecond"]})
    source = frame.to_latex(caption="Örnek & test", label="tab:example_1")
    assert r"\caption{Örnek \& test}" in source
    assert r"\label{tab:example_1}" in source
    assert r"\begin{tabular}[t]{@{}l@{}}first \\ second\end{tabular}" in source
    assert r"unsafe\_column\_\%" in source


def test_full_dataframe_export_preserves_every_duplicate_index_row_and_last_column():
    frame = oe.DataFrame({f"column_{col}": [row * 100 + col for row in range(73)]
                          for col in range(34)}, index=["duplicate"] * 73)
    source = frame.to_latex()
    unwrapped = source.replace(r"\allowbreak{}", "")
    assert unwrapped.count("duplicate &") == 73
    assert "7233" in source
    assert r"column\_33" in unwrapped
    assert "Preview:" not in source
    assert "Preview:" in source.math
    assert "50 of 73 rows, 30 of 35 columns" in source.math
    assert r"column\_33" not in source.math


def test_frame_large_integer_missing_and_tiny_numbers_remain_accurate():
    frame = oe.DataFrame({"identifier": pd.array([9007199254740993, None], dtype="Int64"),
                          "small": [1.2e-12, float("nan")],
                          "infinite": [float("inf"), -float("inf")]})
    source = frame.to_latex(na_rep="N/A", index=False)
    assert "9007199254740993" in source
    assert r"$1.2000 \times 10^{-12}$" in source
    assert source.count("N/A") == 2
    assert r"$\infty$" in source and r"$-\infty$" in source
    assert "9.007" not in source
    explicit = frame.to_latex(index=False, float_format="%.2f")
    assert r"\times" not in explicit
    assert "0.00" in explicit
    assert "9007199254740993" in explicit


def test_multiindex_levels_labels_names_and_column_order_are_preserved():
    index = pd.MultiIndex.from_tuples([("a_1", 2), ("a_1", 2), ("ç", 3)],
                                     names=["firm_id", "year"])
    columns = pd.MultiIndex.from_tuples([("Outcome", "Mean"), ("Outcome", "SD"),
                                         ("Other", "N")], names=["Group", "Statistic"])
    frame = oe.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=columns)
    source = frame.to_latex()
    assert "Group &  & Outcome & Outcome & Other" in source
    assert "Statistic &  & Mean & SD & N" in source
    assert r"firm\_id & year &  &  & " in source
    assert source.count(r"a\_1 & 2 &") == 2
    assert "ç & 3 & 7 & 8 & 9" in source
    no_index = frame.to_latex(index=False)
    assert r"a\_1" not in no_index
    assert "Outcome & Outcome & Other" in no_index


def test_native_frame_operations_keep_api_metadata_and_hash():
    original = pd.DataFrame({"x": [1, 2, 3], "category": pd.Categorical(["b", "a", "b"],
                                                                       categories=["b", "a"])})
    original.attrs["metadata"] = {"labels": {"x": "Ç & x"}}
    wrapped = as_frame(original)
    assert isinstance(wrapped, oe.DataFrame)
    assert frame_hash(wrapped) == frame_hash(original)
    assert wrapped.attrs == original.attrs
    for view in [wrapped.head(2), wrapped.iloc[:2], wrapped[["x"]], wrapped.copy()]:
        assert isinstance(view, oe.DataFrame)
        assert isinstance(view.latex, oe.Latex)
        assert view.attrs == original.attrs
    wrapped.attrs["metadata"]["labels"]["x"] = "changed"
    assert original.attrs["metadata"]["labels"]["x"] == "Ç & x"
    assert isinstance(wrapped.describe(), oe.DataFrame)
    assert isinstance(wrapped.describe().latex, oe.Latex)


def test_read_example_and_saved_snapshot_return_openecon_frames_without_hash_change(tmp_path):
    example = oe.example()
    plain = pd.DataFrame(example)
    plain.attrs = example.attrs.copy()
    assert isinstance(example, oe.DataFrame)
    assert frame_hash(example) == frame_hash(plain)
    source = tmp_path / "dataset.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(source, index=False)
    imported = oe.read(source)
    assert isinstance(imported, oe.DataFrame)
    workspace = Workspace(tmp_path / "workspace")
    profile = workspace.import_file(source)
    restored = oe.load_dataset(profile["id"], workspace=workspace.path)
    assert isinstance(restored, oe.DataFrame)
    assert frame_hash(restored) == profile["data_hash"]
    pd.testing.assert_frame_equal(restored, imported)


def test_plain_pandas_frame_is_supported_without_global_monkeypatch():
    plain = pd.DataFrame({"x": [1, 2]})
    method = pd.DataFrame.to_latex
    source = oe.to_latex(plain)
    assert isinstance(source, oe.Latex)
    assert pd.DataFrame.to_latex is method
    assert not hasattr(plain, "latex")


@pytest.mark.parametrize("covariance,description", [
    (None, "conventional standard errors"), ("HC3", "HC3 heteroskedasticity-robust"),
])
def test_model_latex_contains_full_inference_context_and_leaves_json_contract_intact(tmp_path, covariance, description):
    result = model(covariance=covariance)
    snapshot = result.model_dump_json()
    text = result.summary()
    assert isinstance(text, str) and "Observations: 120" in text
    source = result.summary(format="latex")
    assert isinstance(source, oe.Latex)
    assert source == result.to_latex() == result.latex
    for token in [r"wage\_\%", r"x\_1 \& y", "Observations & 120", description,
                  r"95\% confidence level", "2 observations excluded from estimation",
                  r"$R^{2}$ & 0.7500", r"1.2346^{***}", r"$(0.0100)$"]:
        assert token in source
    assert r"\input{" not in source
    assert result.model_dump_json() == snapshot
    with pytest.raises(ValueError, match="format"):
        result.summary(format="html")
    file = tmp_path / "model.tex"
    assert result.to_latex(file) is None
    assert file.read_text() == source


def test_files_streams_formats_and_aliases():
    frame = oe.DataFrame({"x": [1.2345], "y": [2.3456]})
    stream = io.StringIO()
    assert frame.to_latex(stream, columns=["y"], header=["Y_%"], index=False,
                          float_format=lambda value: f"{value:.2f}") is None
    assert "2.35" in stream.getvalue()
    assert "1.23" not in stream.getvalue()
    assert r"Y\_\%" in stream.getvalue()
    assert str(oe.latex(frame)) == str(oe.to_latex(frame))
    assert isinstance(oe.latex(frame).math, str)
    raw = oe.Latex(r"\alpha+\beta", math=r"\alpha+\beta")
    assert oe.to_latex(raw) is raw
    assert display_latex(raw) == (raw, raw.math)
    assert raw.source == raw._repr_latex_() == r"\alpha+\beta"


@pytest.mark.parametrize("options", [{"precision": -1}, {"precision": True},
                                     {"precision": 19}, {"label": r"x}\input{file}"},
                                     {"column_format": r"l}\input{file}"},
                                     {"column_format": "ll"}, {"header": ["x", "y"]}])
def test_invalid_export_options_raise_instead_of_inserting_untrusted_commands(options):
    with pytest.raises((ValueError, TypeError)):
        oe.DataFrame({"x": [1]}).to_latex(index=False, **options)


def test_display_preview_only_is_bounded_and_series_labels_are_kept():
    frame = oe.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=["one", "two", "last"])
    source, math = display_latex(frame, max_rows=1, max_columns=2)
    assert "last & 3 & 6" in source
    assert r"\begin{array}{lr}" in math
    assert r"\text{one}" in math
    assert r"\text{last}" not in math
    assert "1 of 3 rows, 2 of 3 columns" in math
    series_source, series_math = display_latex(pd.Series([1, None], index=["a_1", "b"], name="score_%"))
    assert r"score\_\%" in series_source and r"a\_1" in series_math


def test_server_record_helpers_do_not_require_pandas_or_torch():
    root = Path(__file__).resolve().parents[1]
    code = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'pandas', 'torch'} or fullname.startswith(('pandas.', 'torch.')):
            raise AssertionError('Heavy import ' + fullname)
sys.meta_path.insert(0, Block())
import openecon as oe
from openecon.latex import display_latex, table_latex, table_math
source = table_latex(['x_%'], [[9007199254740993], [None]], index=[['firm', 2026], ['firm', 2027]], index_names=['firm', 'year'])
assert '9007199254740993' in source
assert callable(oe.latex)
assert 'firm & 2026' in source
assert '\\\\begin{array}' in table_math(['x'], [[1]])
assert display_latex('Ç & _')[1]
assert not {'pandas', 'torch'} & set(sys.modules)
'''
    run = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                         text=True, timeout=15)
    assert run.returncode == 0, run.stderr


def test_record_helpers_validate_row_and_index_lengths_and_escape_data():
    source = table_latex(["x_%"], [["Ç & x"]], index=["a_1"], index_names=["firm"])
    assert r"a\_1 & Ç \& x" in source
    assert r"x\_\%" in source
    assert "array" in table_math(["x"], [[1]])
    with pytest.raises(ValueError, match="row"):
        table_latex(["x"], [[1, 2]])
    with pytest.raises(ValueError, match="index"):
        table_latex(["x"], [[1]], index=[])
    with pytest.raises(ValueError, match="index_names"):
        table_latex(["x"], [[1]], index_names=["firm"])


def test_default_source_uses_publication_rules_and_explicit_legacy_rules_remain_available():
    source = oe.DataFrame({"x": [1]}).to_latex()
    assert r"\toprule" in source and r"\midrule" in source and r"\bottomrule" in source
    assert r"\begin{adjustbox}{max width=\linewidth}" in source
    assert r"\hline" not in source and r"\begin{longtable}" not in source
    legacy = oe.DataFrame({"x": [1]}).to_latex(booktabs=False, max_width=None)
    assert r"\hline" in legacy and r"\toprule" not in legacy
    optional = oe.DataFrame({"x": [1]}).to_latex(booktabs=True, longtable=True,
                                               caption=("Full ] & caption", "Short ] caption"),
                                               label="tab:long")
    assert r"\toprule" in optional and r"\begin{longtable}" in optional
    assert r"\caption[{Short ] caption}]{Full ] \& caption}" in optional
    json.loads(model().model_dump_json())


def test_zero_column_frame_still_exports_all_index_rows():
    frame = oe.DataFrame(index=["first", "duplicate", "duplicate", "last"])
    source = frame.to_latex()
    assert source.count("duplicate") == 2
    assert "first" in source and "last" in source
    assert source.count(r"\\") == 5
    no_index = frame.to_latex(index=False)
    assert no_index.count(r"\\") == 5
    assert "4 of 4 rows" not in source.math


def test_record_helpers_accept_single_level_serialized_index():
    source = table_latex(["x"], [[1], [2]], index=[["one"], ["two"]], index_names=[None])
    assert "one & 1" in source and "two & 2" in source
    assert "['one']" not in source
