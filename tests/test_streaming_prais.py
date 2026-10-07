"""AR(1) global sufficient geometry and actual-period HC/DW contracts."""
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.econometrics.streaming_prais import fit_streaming_prais
from test_streaming_ardl import source
from test_streaming_var import numeric_tree, numerical_blocks


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(333)
    n = 499
    x = torch.randn(n, generator=g, dtype=torch.float64)
    u = torch.randn(n, generator=g, dtype=torch.float64)
    for i in range(1, n):
        u[i] += .6*u[i-1]
    return pd.DataFrame({"x": x.numpy(), "y": (1+.7*x+u).numpy(), "t": range(n),
                         "cat": ["a", "b", "c"]*166+["a"]})


@pytest.mark.parametrize("method", ["prais", "corc"])
@pytest.mark.parametrize("rhotype", ["regress", "freg", "tscorr", "dw", "theil", "nagar"])
def test_all_rho_formulas_and_methods(data, method, rhotype, monkeypatch):
    dense = oe.prais(data=data, y="y", x=["x"], time="t", method=method, rhotype=rhotype)
    numerical_blocks(monkeypatch, 3)
    result = fit_streaming_prais(dense.spec, source(data, 7))
    assert result.nobs == dense.nobs and result.dropped_rows == dense.dropped_rows
    numeric_tree([c.estimate for c in dense.coefficients], [c.estimate for c in result.coefficients], 3e-8)
    numeric_tree(dense.covariance_matrix, result.covariance_matrix, 3e-8)
    numeric_tree(dense.metrics, result.metrics, 3e-8)
    numeric_tree(dense.tests, result.tests, 3e-8)
    numeric_tree(dense.extra, result.extra, 3e-8)
    assert result.provenance["solver_diagnostics"]["rho_iterations_replay_source"] is False


@pytest.mark.parametrize("covariance", ["nonrobust", "HC1", "HC2", "HC3"])
@pytest.mark.parametrize("intercept", [True, False])
def test_final_covariance_and_categories(data, covariance, intercept, monkeypatch):
    dense = oe.prais(data=data, y="y", x=["x", "cat"], categorical=["cat"], time="t",
                     twostep=True, covariance=covariance, intercept=intercept)
    numerical_blocks(monkeypatch, 71)
    result = fit_streaming_prais(dense.spec, source(data, 3))
    numeric_tree([c.estimate for c in dense.coefficients], [c.estimate for c in result.coefficients], 3e-8)
    numeric_tree(dense.covariance_matrix, result.covariance_matrix, 3e-8)
    numeric_tree(dense.metrics, result.metrics, 3e-8)
    numeric_tree(dense.tests, result.tests, 3e-8)
    assert dense.provenance["categorical_encoding"] == result.provenance["categorical_encoding"]


def test_sorted_file_missing_endpoints(data, tmp_path):
    frame = data.copy()
    frame.loc[[0, len(frame)-1], "y"] = float("nan")
    frame.t = pd.date_range("2020-01-01", periods=len(frame), freq="h", tz="Europe/Istanbul")
    frame = frame.sample(frac=1, random_state=871).reset_index(drop=True)
    path = tmp_path/"prais.parquet"
    frame.to_parquet(path, index=False, row_group_size=7)
    dense = oe.prais(data=frame, y="y", x=["x"], time="t", method="corc", missing="drop")
    result = fit_streaming_prais(dense.spec, oe.scan(path))
    numeric_tree(dense.metrics, result.metrics, 3e-8)
    expected = frame.dropna().sort_values("t").index.tolist()[1:401]
    assert [p["row"] for p in result.predictions] == expected
