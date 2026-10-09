"""Frequency reliability/adequacy: independent matrices and full replication."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet
from openecon.resources import use_workspace_budget


def fixture(domain="assessment", n=173):
    rng = np.random.default_rng(693 if domain == "assessment" else 901)
    latent = rng.normal(size=n) if domain == "assessment" else rng.standard_t(9, size=n)
    x = latent[:, None]*np.array([.75, -.65, .8, .7, .6]) + rng.normal(scale=.55, size=(n, 5))
    return pd.DataFrame(x*np.arange(1, 6)+np.arange(5), columns=list("abcde")).assign(w=np.arange(n)%4)


def stream(frame):
    def batches():
        for start in range(0, len(frame), 13):
            yield frame.iloc[start:start+13]
    return Dataset.from_batches(batches, columns=list(frame), row_count=len(frame))


def restored(result):
    payload = json.loads(json.dumps({"attrs": result.attrs, "tables": {
        k: v.astype(object).where(v.notna(), None).to_dict(orient="split") for k, v in result.items()}}, allow_nan=False))
    return TableSet({k: pd.DataFrame(**v) for k, v in payload["tables"].items()}, **payload["attrs"])


@pytest.mark.parametrize("domain", ["assessment", "facility"])
@pytest.mark.parametrize("model", ["alpha", "split", "guttman"])
@pytest.mark.parametrize("standardized", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_all_weighted_reliability_tables_and_json_match_expanded_independent_moments(domain, model, standardized, streaming):
    frame = fixture(domain)
    frame.loc[5, "a"] = np.nan
    frame.loc[9, "w"] = np.nan
    clean = frame.dropna()
    expanded = clean.loc[clean.index.repeat(clean.w.astype(int)), list("abcde")]
    result = oe.alpha(stream(frame) if streaming else frame, list("abcde"), weights="w", model=model, standardized=standardized, reverse=["b"])
    reference = oe.alpha(expanded, list("abcde"), model=model, standardized=standardized, reverse=["b"])
    assert result.attrs["n"] == int(clean.w.sum())
    assert result.attrs["physical_rows"] == int((clean.w > 0).sum())
    assert result.attrs["n_zero_weight"] == int((clean.w == 0).sum())
    assert result.attrs["n_missing"] == 2
    for key in reference:
        np.testing.assert_allclose(result[key], reference[key], atol=2e-12, rtol=2e-12, equal_nan=True)
        np.testing.assert_allclose(restored(result)[key], result[key], atol=0, rtol=0, equal_nan=True)
    values = expanded.to_numpy() * np.array([1, -1, 1, 1, 1])
    covariance = np.cov(values, rowvar=False, ddof=1)
    correlation = np.corrcoef(values, rowvar=False)
    C = correlation if standardized else covariance
    mean = np.zeros(5) if standardized else values.mean(0)
    total = C.sum()
    np.testing.assert_allclose(result.attrs["alpha"], 5/4*(1-np.trace(C)/total), atol=2e-12)
    expected_items = []
    for j in range(5):
        other = np.arange(5) != j
        rest = C[np.ix_(other, other)]
        rvar = rest.sum()
        expected_items.append([mean[j], np.sqrt(C[j,j]), C[j].sum()/np.sqrt(C[j,j]*total),
            C[j,other].sum()/np.sqrt(C[j,j]*rvar), mean[other].sum(), rvar,
            1-1/np.linalg.inv(correlation)[j,j], 4/3*(1-np.trace(rest)/rvar)])
    np.testing.assert_allclose(result["items"], expected_items, atol=3e-12, rtol=3e-12)
    assert "sampling SE and CI unavailable" in result.attrs["inference"]


@pytest.mark.parametrize("domain", ["assessment", "facility"])
@pytest.mark.parametrize("streaming", [False, True])
def test_weighted_factor_adequacy_full_kmo_msa_bartlett_inference(domain, streaming):
    frame = fixture(domain)
    frame.loc[8, "a"] = np.nan
    frame.loc[12, "w"] = np.nan
    clean = frame.dropna()
    expanded = clean.loc[clean.index.repeat(clean.w.astype(int)), list("abcde")]
    result = oe.factortest(stream(frame) if streaming else frame, list("abcde"), weights="w")
    expected = oe.factortest(expanded, list("abcde"))
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12, equal_nan=True)
    R = np.corrcoef(expanded.to_numpy(), rowvar=False)
    inv = np.linalg.inv(R)
    partial = -inv/np.sqrt(np.outer(inv.diagonal(), inv.diagonal()))
    np.fill_diagonal(partial, 0)
    np.fill_diagonal(R, 0)
    r2, a2 = (R**2).sum(1), (partial**2).sum(1)
    np.testing.assert_allclose(result.kmo.iloc[:-1], r2/(r2+a2), atol=1e-12)
    np.testing.assert_allclose(result.attrs["kmo"], r2.sum()/(r2+a2).sum(), atol=1e-12)
    r = np.corrcoef(expanded.to_numpy(), rowvar=False)
    stat = -(len(expanded)-1-15/6)*np.linalg.slogdet(r)[1]
    np.testing.assert_allclose(result.attrs["statistic"], stat, atol=3e-12)
    np.testing.assert_allclose(result.attrs["p_value"], chi2.sf(stat, 10), rtol=1e-10)
    assert result.attrs["df"] == 10 and result.attrs["n_missing"] == 2
    assert result.attrs["covariance_divisor"] == len(expanded)-1
    encoded = json.dumps({"attrs": result.attrs, "table": result.astype(object).where(result.notna(), None).to_dict(orient="split")}, allow_nan=False)
    payload = json.loads(encoded)
    np.testing.assert_allclose(pd.DataFrame(**payload["table"]), result, equal_nan=True)


@pytest.mark.parametrize("procedure", [oe.alpha, oe.factortest])
@pytest.mark.parametrize("bad", [-1, .25, 2**53+1, np.inf])
def test_weight_precision_and_nonfinite_refusals(procedure, bad):
    frame = fixture()
    frame["w"] = pd.Series([1]*len(frame), dtype="uint64" if bad > 2**53 and np.isfinite(bad) else "float64")
    frame.loc[0, "w"] = bad
    with pytest.raises(AnalysisError):
        procedure(frame, list("abcde"), weights="w")


@pytest.mark.parametrize("procedure", [oe.alpha, oe.factortest])
def test_roles_type_total_zero_missing_and_matrix_budget_refusals(procedure):
    frame = fixture()
    for kwargs in ({"weights": "a"}, {"weights": "w", "weight_type": "aweight"}):
        with pytest.raises(AnalysisError):
            procedure(frame, list("abcde"), **kwargs)
    for weights in ([0]*len(frame), [1]+[0]*(len(frame)-1), [2**53]+[1]*(len(frame)-1)):
        with pytest.raises(AnalysisError):
            procedure(frame.assign(w=weights), list("abcde"), weights="w")
    with pytest.raises(AnalysisError, match="missing"):
        procedure(frame.assign(a=np.nan), list("abcde"), weights="w", missing="raise")
    large = pd.DataFrame({f"x{i}": [0., 1., 2.] for i in range(257)}).assign(w=1)
    with pytest.raises(AnalysisError, match="256"):
        procedure(large, list(large.columns[:-1]), weights="w")
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError):
            procedure(pd.concat([frame]*40, ignore_index=True), list("abcde"), weights="w")


def test_weighted_reliability_singular_smc_not_fabricated_and_rank_cannot_be_created():
    frame = fixture()
    frame["e"] = frame["a"]+frame["c"]
    result = oe.alpha(frame, list("abcde"), weights="w", model="guttman")
    assert result["items"].squared_multiple_correlation.isna().all()
    assert pd.isna(result["guttman"].loc["lambda6", "value"])
    with pytest.raises(AnalysisError, match="correlation matrix"):
        oe.factortest(frame, list("abcde"), weights="w")


@pytest.mark.parametrize("procedure", [oe.alpha, oe.factortest])
def test_stable_large_offset_frequency_moments(procedure):
    frame = fixture(n=301)
    frame[list("abcde")] += 1e10
    expanded = frame.loc[frame.index.repeat(frame.w), list("abcde")]
    one = procedure(frame, list("abcde"), weights="w")
    two = procedure(expanded, list("abcde"))
    if procedure is oe.alpha:
        for key in one:
            np.testing.assert_allclose(one[key], two[key], atol=2e-5, rtol=2e-7)
    else:
        np.testing.assert_allclose(one, two, atol=2e-7, rtol=2e-7, equal_nan=True)
