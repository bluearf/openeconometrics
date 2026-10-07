"""oe.collin against statsmodels' VIF and an explicit NumPy singular value decomposition."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet

NAMES = ["a", "b", "c", "d", "e"]


def _data(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5)) + np.array([10.0, 0.0, 50.0, -3.0, 0.5])
    x[:, 1] += 0.9 * x[:, 0]
    x[:, 4] = x[:, 2] - 2 * x[:, 3] + 0.05 * rng.normal(size=n)
    return pd.DataFrame(x, columns=NAMES)


def _bkw(design: np.ndarray):
    """Belsley-Kuh-Welsch eigenvalues, condition indices and variance proportions."""
    scaled = design / np.linalg.norm(design, axis=0)
    _, singular, vt = np.linalg.svd(scaled, full_matrices=False)
    phi = (vt.T ** 2) / singular ** 2                      # [coefficient, dimension]
    return singular ** 2, singular[0] / singular, (phi / phi.sum(axis=1, keepdims=True)).T


def test_collin_is_exported_and_renders():
    from openecon.econometrics import registry

    assert registry.public_exports()["collin"][0] == "openecon.econometrics.selection.collin"
    assert "estat vif" in oe.collin.__doc__
    result = oe.collin(_data(), NAMES)
    assert isinstance(result, TableSet) and list(result) == ["vif", "condition"]
    text = str(result)
    assert "Collinearity diagnostics" in text and "[condition]" in text
    assert result.to_latex().count(r"\begin{tabular}") == 2
    assert json.loads(json.dumps(result.attrs)) == result.attrs
    for table in result.values():
        assert set(json.loads(table.to_json())) == set(table.columns)


def test_vif_matches_statsmodels_and_auxiliary_regressions():
    frame = _data()
    result = oe.collin(frame, NAMES)
    design = sm.add_constant(frame[NAMES]).to_numpy()
    expected = np.array([variance_inflation_factor(design, i) for i in range(1, 6)])
    table = result["vif"]
    assert list(table.index) == NAMES
    np.testing.assert_allclose(table["vif"], expected, rtol=1e-8)
    np.testing.assert_allclose(table["tolerance"], 1 / expected, rtol=1e-8)
    for name in NAMES:
        others = [other for other in NAMES if other != name]
        auxiliary = sm.OLS(frame[name], sm.add_constant(frame[others])).fit()
        assert table.loc[name, "r_squared"] == pytest.approx(auxiliary.rsquared, rel=1e-9)
    assert result.attrs["mean_vif"] == pytest.approx(expected.mean(), rel=1e-8)
    assert result.attrs["max_vif"] == pytest.approx(expected.max(), rel=1e-8)
    assert result.attrs["n"] == 400 and result.attrs["n_missing"] == 0
    # The inverse correlation matrix has the VIFs on its diagonal.
    inverse = np.linalg.inv(np.corrcoef(frame[NAMES].to_numpy(), rowvar=False))
    np.testing.assert_allclose(table["vif"], np.diag(inverse), rtol=1e-8)


def test_condition_indices_and_variance_proportions_match_numpy_svd():
    frame = _data()
    result = oe.collin(frame, NAMES)
    eigenvalues, indices, proportions = _bkw(sm.add_constant(frame[NAMES]).to_numpy())
    table = result["condition"]
    assert list(table.columns) == ["eigenvalue", "condition_index", "Intercept", *NAMES]
    assert list(table.index) == [1, 2, 3, 4, 5, 6] and table.index.name == "dimension"
    np.testing.assert_allclose(table["eigenvalue"], eigenvalues, rtol=1e-8)
    np.testing.assert_allclose(table["condition_index"], indices, rtol=1e-8)
    np.testing.assert_allclose(table[["Intercept", *NAMES]].to_numpy(), proportions,
                               rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(table[["Intercept", *NAMES]].sum(axis=0), 1.0, rtol=1e-12)
    assert table["eigenvalue"].sum() == pytest.approx(6.0, rel=1e-12)
    assert result.attrs["condition_number"] == pytest.approx(indices[-1], rel=1e-8)
    # The near dependency e = c - 2 d shows up in the last dimension.
    last = table.iloc[-1]
    assert last["c"] > 0.5 and last["e"] > 0.5


def test_without_intercept_uses_uncentred_quantities():
    frame = _data()
    result = oe.collin(frame, NAMES, intercept=False)
    design = frame[NAMES].to_numpy()
    expected = np.array([variance_inflation_factor(design, i) for i in range(5)])
    np.testing.assert_allclose(result["vif"]["vif"], expected, rtol=1e-8)
    eigenvalues, indices, proportions = _bkw(design)
    table = result["condition"]
    assert list(table.columns) == ["eigenvalue", "condition_index", *NAMES]
    np.testing.assert_allclose(table["eigenvalue"], eigenvalues, rtol=1e-8)
    np.testing.assert_allclose(table["condition_index"], indices, rtol=1e-8)
    np.testing.assert_allclose(table[NAMES].to_numpy(), proportions, rtol=1e-6, atol=1e-10)
    assert result.attrs["intercept"] is False


def test_single_regressor_and_orthogonal_columns():
    frame = _data()
    one = oe.collin(frame, ["a"])
    assert one["vif"].loc["a", "vif"] == pytest.approx(1.0, abs=1e-12)
    assert len(one["condition"]) == 2
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.normal(size=(50, 4)) - 0.0)
    q = q - q.mean(axis=0)
    q, _ = np.linalg.qr(q)
    orthogonal = pd.DataFrame(q[:, :3], columns=["p", "q", "r"])
    result = oe.collin(orthogonal, ["p", "q", "r"])
    np.testing.assert_allclose(result["vif"]["vif"], 1.0, rtol=1e-10)
    np.testing.assert_allclose(result["condition"]["condition_index"], 1.0, rtol=1e-7)


def test_large_offsets_keep_full_precision():
    frame = _data()
    moved = frame.copy()
    moved["a"] = frame["a"] * 1e-4 + 1e6
    moved["c"] = frame["c"] * 1e5
    base, result = oe.collin(frame, NAMES), oe.collin(moved, NAMES)
    # VIFs are invariant to the location and scale of every regressor.
    np.testing.assert_allclose(result["vif"]["vif"], base["vif"]["vif"], rtol=1e-6)
    eigenvalues, indices, _ = _bkw(sm.add_constant(moved[NAMES]).to_numpy())
    np.testing.assert_allclose(result["condition"]["condition_index"], indices, rtol=1e-5)


def test_block_accumulation_and_missing_rows(monkeypatch):
    from openecon.econometrics.selection import common as module

    frame = _data(n=5000)
    frame.loc[[3, 1200, 4999], "b"] = np.nan
    frame.loc[77, "e"] = np.nan
    whole = oe.collin(frame, NAMES)
    assert whole.attrs["n"] == 4996 and whole.attrs["n_missing"] == 4
    reference = oe.collin(frame.dropna(), NAMES)
    monkeypatch.setattr(module, "BLOCK_ELEMENTS", 64)        # five row blocks
    blocked = oe.collin(frame, NAMES)
    for name in whole:
        np.testing.assert_allclose(blocked[name].to_numpy(), whole[name].to_numpy(),
                                   rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(reference[name].to_numpy(), whole[name].to_numpy(),
                                   rtol=1e-9, atol=1e-12)
    with pytest.raises(AnalysisError) as error:
        oe.collin(frame, NAMES, missing="raise")
    assert error.value.code == "missing_values"


def test_exact_dependencies_are_reported_by_name():
    frame = _data()
    frame["copy"] = frame["a"]
    frame["total"] = frame["a"] + 2 * frame["d"]
    frame["flat"] = 7.0
    frame["zero"] = 0.0
    for names, culprit in ((["a", "b", "copy"], "copy"), (["a", "d", "c", "total"], "total")):
        with pytest.raises(AnalysisError) as error:
            oe.collin(frame, names)
        assert error.value.code == "perfect_collinearity" and f"'{culprit}'" in str(error.value)
        with pytest.raises(AnalysisError) as error:
            oe.collin(frame, names, intercept=False)
        assert error.value.code == "perfect_collinearity"
    with pytest.raises(AnalysisError) as error:
        oe.collin(frame, ["a", "flat"])
    assert error.value.code == "zero_variance" and "flat" in str(error.value)
    # Without a constant in the model a constant column is a legitimate regressor.
    assert oe.collin(frame, ["a", "flat"], intercept=False)["vif"].shape == (2, 3)
    with pytest.raises(AnalysisError) as error:
        oe.collin(frame, ["a", "zero"], intercept=False)
    assert error.value.code == "zero_variance"


@pytest.mark.parametrize("kwargs, code", [
    ({"x": "a"}, "invalid_spec"),
    ({"x": []}, "invalid_spec"),
    ({"x": ["a", "a"]}, "invalid_spec"),
    ({"x": ["a", "nope"]}, "missing_columns"),
    ({"x": ["a", "label"]}, "non_numeric_column"),
    ({"x": ["a"], "intercept": 1}, "invalid_option"),
    ({"x": ["a"], "missing": "pairwise"}, "invalid_option"),
])
def test_invalid_arguments_raise_analysis_errors(kwargs, code):
    frame = _data(n=30).assign(label="u")
    with pytest.raises(AnalysisError) as error:
        oe.collin(frame, **kwargs)
    assert error.value.code == code


def test_degenerate_samples_raise_analysis_errors():
    frame = _data(n=30)
    with pytest.raises(AnalysisError) as error:
        oe.collin(frame.iloc[:5], NAMES)
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        oe.collin(frame.assign(a=np.nan), NAMES)
    assert error.value.code == "empty_sample"
    with pytest.raises(AnalysisError) as error:
        oe.collin(frame.assign(a=np.inf), NAMES)
    assert error.value.code == "non_finite_values"
    with pytest.raises(AnalysisError) as error:
        oe.collin(frame.iloc[:0], NAMES)
    assert error.value.code == "empty_data"
    records = frame.to_dict("records")
    assert oe.collin(records, NAMES).attrs["n"] == 30
