"""Two-domain CCA frequency/summary/score contracts with independent oracles."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import linalg, stats

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate.canon import canon, canon_matrix, canon_scores
from openecon.econometrics.multivariate import canon_options as options
from openecon.resources import use_workspace_budget

XS, YS = ["x1", "x2", "x3"], ["y1", "y2"]
NAMES = [*XS, *YS]


def domain(name="process_quality", *, n=180, seed=830):
    rng = np.random.default_rng(seed)
    if name == "process_quality":
        z = rng.normal(size=(n, 3))
        x = z @ np.array([[.8, -.1, .25], [.1, .75, .15], [.05, .15, .72]])
        y = z @ np.array([[.7, .12], [-.15, .55], [.1, .2]]) + rng.normal(size=(n, 2)) * .75
    else:
        z = rng.lognormal(sigma=.4, size=(n, 3))
        x = z @ np.array([[.7, .2, -.1], [-.15, .8, .2], [.2, .05, .6]])
        y = z @ np.array([[.5, .15], [.1, .75], [.2, -.1]]) + rng.standard_t(7, size=(n, 2)) * .2
    return pd.DataFrame(np.c_[x, y] * [2, 3, .5, 4, 1.5] + [10, -3, 42, 70, -.5], columns=NAMES)


def stream(frame, *, size=17, count=True):
    def batches():
        for start in range(0, len(frame), size):
            yield frame.iloc[start:start + size]
    return Dataset.from_batches(batches, columns=list(frame), row_count=len(frame) if count else None)


def restored(result):
    payload = json.loads(json.dumps({"attrs": result.attrs, "tables": {
        key: json.loads(table.to_json(orient="split", double_precision=15))
        for key, table in result.items()}}, allow_nan=False))
    return TableSet({key: pd.DataFrame(**table) for key, table in payload["tables"].items()}, **payload["attrs"])


def equal_tables(one, two, *, exclude=(), tol=2e-10):
    keys = set(one) - set(exclude)
    assert keys <= set(two)
    for key in keys:
        assert list(one[key].index) == list(two[key].index)
        assert list(one[key].columns) == list(two[key].columns)
        for column in one[key]:
            if column in ("set", "f_type"):
                assert list(one[key][column]) == list(two[key][column])
            else:
                np.testing.assert_allclose(one[key][column].to_numpy(dtype="float64"),
                                           two[key][column].to_numpy(dtype="float64"),
                                           atol=tol, rtol=tol, equal_nan=True)


def independent(values):
    # Generalized covariance eigenproblem, independently of production QR/SVD.
    covariance = np.cov(values, rowvar=False, ddof=1)
    p, s = len(XS), min(len(XS), len(YS))
    xx, xy, yy = covariance[:p, :p], covariance[:p, p:], covariance[p:, p:]
    square, a = linalg.eigh(xy @ linalg.solve(yy, xy.T, assume_a="pos"), xx)
    square, a = square[::-1][:s], a[:, ::-1][:, :s]
    rho = np.sqrt(square)
    b = linalg.solve(yy, xy.T @ a, assume_a="pos") / rho
    sd = np.sqrt(np.diag(covariance))
    standardized_x = a * sd[:p, None]
    sign = np.sign(standardized_x[np.argmax(abs(standardized_x), axis=0), np.arange(s)])
    a, b = a * sign, b * sign
    return covariance, rho, a, b, sd


def verify_oracle(result, values):
    covariance, rho, a, b, sd = independent(values)
    coefficients = np.r_[a, b]
    np.testing.assert_allclose(result["raw_coefficients"].iloc[:, 1:], coefficients, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(result["standardized_coefficients"].iloc[:, 1:], coefficients * sd[:, None], atol=2e-10)
    np.testing.assert_allclose(result["correlations"].correlation, rho, atol=2e-11)
    p, q = len(XS), len(YS)
    x_load, y_load = covariance[:p, :p] @ a / sd[:p, None], covariance[p:, p:] @ b / sd[p:, None]
    load = np.r_[x_load, y_load]
    np.testing.assert_allclose(result["loadings"].iloc[:, 1:3], load, atol=2e-10)
    np.testing.assert_allclose(result["loadings"].iloc[:, 3:], load * rho, atol=2e-10)
    redundancy = np.c_[np.mean(x_load**2, axis=0), np.mean(x_load**2, axis=0)*rho**2,
                       np.mean(y_load**2, axis=0), np.mean(y_load**2, axis=0)*rho**2]
    np.testing.assert_allclose(result["redundancy"], redundancy, atol=2e-10)
    n = len(values)
    w = n - 1 - (p + q + 1) / 2
    for k in range(len(rho)):
        lam = np.prod(1 - rho[k:]**2)
        pk, qk = p - k, q - k
        numerator, denominator = pk*pk*qk*qk - 4, pk*pk + qk*qk - 5
        exponent = math.sqrt(numerator / denominator) if denominator > 0 else 1.
        df1, df2 = pk*qk, w*exponent - (pk*qk - 2)/2
        f = (1 - lam**(1/exponent)) / lam**(1/exponent) * df2/df1
        row = result["correlations"].iloc[k]
        np.testing.assert_allclose(row[["wilks_lambda", "f", "df1", "df2", "p_value", "chi2", "chi2_df", "chi2_p_value"]],
            [lam, f, df1, df2, stats.f.sf(f, df1, df2), -w*np.log(lam), df1, stats.chi2.sf(-w*np.log(lam), df1)],
            atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("name", ["process_quality", "household_resources"])
@pytest.mark.parametrize("streaming", [False, True])
def test_frequency_full_expansion_independent_oracle_and_scores(name, streaming):
    frame = domain(name).assign(w=np.arange(180) % 5)
    frame.loc[11, "x2"] = np.nan
    frame.loc[23, "y1"] = np.nan
    frame.loc[31, "w"] = np.nan
    before = frame.copy(deep=True)
    clean = frame.dropna()
    expanded = clean.loc[clean.index.repeat(clean.w.astype(int)), NAMES]
    result = canon(stream(frame) if streaming else frame, XS, YS, weights="w")
    reference = canon(expanded, XS, YS)
    equal_tables(result, reference, exclude=("joint_covariance",))
    verify_oracle(result, expanded.to_numpy())
    assert result.attrs["n"] == int(clean.w.sum())
    assert result.attrs["physical_rows"] == int((clean.w > 0).sum())
    assert result.attrs["physical_rows_total"] == len(frame)
    assert result.attrs["n_missing"] == 3
    assert result.attrs["n_zero_weight"] == int((clean.w == 0).sum())
    assert result.attrs["covariance_divisor"] == len(expanded)-1
    assert "original independent observation counts" in result.attrs["inference"]
    actual = canon_scores(restored(result), frame)
    expected = canon_scores(reference, frame)
    np.testing.assert_allclose(actual, expected, atol=2e-10, equal_nan=True)
    # Scores need analysed variables, not the fit-weight column: a missing weight
    # excluded from fitting does not make a new projection row incomplete.
    assert actual.loc[31].notna().all()
    assert actual.loc[[11, 23]].isna().all().all()
    pd.testing.assert_frame_equal(frame, before)
    scored = canon_scores(result, expanded)
    np.testing.assert_allclose(scored.mean(0), 0., atol=2e-12)
    joint_score_cov = np.cov(scored.to_numpy(), rowvar=False)
    np.testing.assert_allclose(joint_score_cov[:2, :2], np.eye(2), atol=2e-11)
    np.testing.assert_allclose(joint_score_cov[2:, 2:], np.eye(2), atol=2e-11)
    np.testing.assert_allclose(joint_score_cov[:2, 2:], np.diag(result["correlations"].correlation), atol=2e-11)
    if streaming:
        assert result.attrs["source_content_sha256"] and result.attrs["source_passes"] == 1
        lazy = canon_scores(restored(result), stream(frame, size=13))
        np.testing.assert_allclose(pd.concat(list(lazy.iter_batches())), actual, atol=2e-10, equal_nan=True)


@pytest.mark.parametrize("name", ["process_quality", "household_resources"])
@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_summary_full_tables_generalized_oracle_and_persisted_scores(name, matrix):
    frame = domain(name)
    values = frame.cov() if matrix == "covariance" else frame.corr()
    result = canon_matrix(values, n=len(frame), x=XS, y=YS, matrix=matrix, means=frame.mean(), sds=frame.std())
    reference = canon(frame, XS, YS)
    equal_tables(result, reference, exclude=("joint_covariance",))
    verify_oracle(result, frame.to_numpy())
    saved = restored(result)
    equal_tables(result, saved)
    assert saved.attrs == result.attrs
    holes = frame.copy()
    holes.index = ["duplicate", "duplicate", *[f"row-{i}" for i in range(len(frame)-2)]]
    holes.iloc[13, 4] = np.nan
    actual = canon_scores(saved, holes)
    assert list(actual.index) == list(holes.index) and actual.iloc[13].isna().all()
    np.testing.assert_allclose(actual, canon_scores(reference, holes), atol=2e-10, equal_nan=True)
    assert saved.attrs["input_kind"] == "summary_matrix"
    assert saved.attrs["training_means_supplied"] and saved.attrs["training_sds_supplied"]


def test_summary_missing_moments_are_not_fabricated():
    frame = domain()
    covariance = canon_matrix(frame.cov(), n=180, x=XS, y=YS)
    assert covariance["descriptives"]["mean"].isna().all()
    assert covariance["raw_coefficients"].iloc[:, 1:].notna().all().all()
    with pytest.raises(AnalysisError, match="training means"):
        canon_scores(restored(covariance), frame)
    correlation = canon_matrix(frame.corr(), n=180, x=XS, y=YS, matrix="correlation", means=frame.mean())
    assert correlation["raw_coefficients"].iloc[:, 1:].isna().all().all()
    assert correlation["descriptives"].std_dev.isna().all()
    assert correlation["standardized_coefficients"].iloc[:, 1:].notna().all().all()
    with pytest.raises(AnalysisError, match="training standard deviations"):
        canon_scores(restored(correlation), frame)


def test_raw_qr_path_retains_independent_near_collinear_precision():
    frame = domain(n=100)
    frame.x2 = frame.x1 + np.random.default_rng(501).normal(size=100)*1e-8
    result = canon(frame, XS, YS)
    assert result.attrs["algorithm"] == "raw-data Householder QR/SVD"
    assert result["correlations"].correlation.between(0, 1).all()
    with pytest.raises(AnalysisError, match="ill-conditioned"):
        canon_matrix(frame.cov(), n=100, x=XS, y=YS, means=frame.mean())


def test_frequency_huge_location_single_pass_partition_and_unknown_count():
    frame = domain(n=90).assign(w=np.arange(90) % 4)
    frame[NAMES] += 1e10
    one = canon(frame, XS, YS, weights="w")
    two = canon(stream(frame, size=11, count=False), XS, YS, weights="w")
    equal_tables(one, two, tol=2e-7)
    assert two.attrs["streaming"] and two.attrs["source_passes"] == 1


@pytest.mark.parametrize("roots", [[.6, .6], [.6, 0.], [0., 0.], [1., .3]])
def test_zero_repeated_and_perfect_roots_preserve_saved_valid_basis(roots):
    xx = np.eye(2)
    cross = np.diag(roots)
    covariance = np.block([[xx, cross], [cross.T, xx]])
    result = canon_matrix(covariance, n=80, x=["a", "b"], y=["c", "d"], means=[0, 0, 0, 0])
    rho = result["correlations"].correlation.to_numpy()
    np.testing.assert_allclose(rho, sorted(roots, reverse=True), atol=2e-15)
    a = result["raw_coefficients"].iloc[:2, 1:].to_numpy(dtype=float)
    b = result["raw_coefficients"].iloc[2:, 1:].to_numpy(dtype=float)
    np.testing.assert_allclose(a.T @ xx @ a, np.eye(2), atol=2e-15)
    np.testing.assert_allclose(b.T @ xx @ b, np.eye(2), atol=2e-15)
    np.testing.assert_allclose(a.T @ cross @ b, np.diag(rho), atol=2e-15)
    expected_unidentified = ["Canon1", "Canon2"] if roots[0] == roots[1] else ["Canon2"] if roots[1] == 0 else []
    assert result.attrs["canonical_unidentified_axes"] == expected_unidentified
    rows = pd.DataFrame(np.arange(32).reshape(8, 4), columns=list("abcd"))
    np.testing.assert_array_equal(canon_scores(result, rows), canon_scores(restored(result), rows))
    if roots[0] == 1:
        assert result["correlations"].iloc[0][["eigenvalue", "f", "p_value", "chi2", "chi2_p_value"]].isna().all()
        assert result["tests"].loc["wilks", "value"] == 0


@pytest.mark.parametrize("bad", [-1, .2, np.inf, 2**53+1])
def test_bad_frequency_weights_and_total(bad):
    frame = domain(n=20).assign(w=1)
    if not isinstance(bad, int):
        frame.w = frame.w.astype(float)
    frame.loc[0, "w"] = bad
    with pytest.raises(AnalysisError):
        canon(frame, XS, YS, weights="w")
    frame = domain(n=20).assign(w=2**52)
    with pytest.raises(AnalysisError, match="total"):
        canon(frame, XS, YS, weights="w")


def test_frequency_missing_weight_roles_and_unsupported_options():
    frame = domain(n=40).assign(w=1)
    with pytest.raises(AnalysisError):
        canon(frame, XS, YS, weights="absent")
    with pytest.raises(AnalysisError, match="distinct"):
        canon(frame, XS, YS, weights="x1")
    with pytest.raises(AnalysisError):
        canon(frame, XS, YS, weights="w", weight_type="aweight")
    with pytest.raises(AnalysisError):
        canon(frame.assign(w=0), XS, YS, weights="w")
    frame.loc[0, "w"] = np.nan
    with pytest.raises(AnalysisError, match="missing"):
        canon(frame, XS, YS, weights="w", missing="raise")
    with pytest.raises(AnalysisError, match="distinct"):
        canon(frame, XS, ["x1"])


@pytest.mark.parametrize("mutation", ["asymmetric", "indefinite", "nonfinite", "zero_diag", "rank_x", "wrong_labels", "complex"])
def test_summary_matrix_refusals(mutation):
    frame = domain()
    values = frame.cov()
    if mutation == "asymmetric":
        values.iloc[0, 1] += .1
    elif mutation == "indefinite":
        values.iloc[0, 3] = values.iloc[3, 0] = 100
    elif mutation == "nonfinite":
        values.iloc[0, 0] = np.inf
    elif mutation == "zero_diag":
        values.iloc[0, 0] = 0
    elif mutation == "rank_x":
        values.loc[:, "x2"] = values.loc[:, "x1"].to_numpy()
        values.loc["x2", :] = values.loc["x1", :].to_numpy()
    elif mutation == "wrong_labels":
        values.index = list(reversed(NAMES))
    else:
        values = values.astype(complex)
    with pytest.raises(AnalysisError):
        canon_matrix(values, n=180, x=XS, y=YS)


@pytest.mark.parametrize("kw", [{"n": 6}, {"n": True}, {"n": 2**53+1}, {"means": [0, 0]},
                              {"sds": [-1]*5}, {"matrix": "correlation"}])
def test_summary_n_and_training_moment_gates(kw):
    frame = domain()
    args = {"n": 180, "x": XS, "y": YS, **kw}
    with pytest.raises(AnalysisError):
        canon_matrix(frame.cov(), **args)
    with pytest.raises(AnalysisError):
        canon_matrix(frame.cov(), n=180, x=XS, y=YS, means=pd.Series([0]*5, index=list("abcde")))


def test_work_and_workspace_admission_before_conversion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source conversion preceded resource admission")
    monkeypatch.setattr(options, "_selected", forbidden)
    with pytest.raises(AnalysisError, match="1000000"):
        canon({name: range(1_000_001) for name in [*NAMES, "w"]}, XS, YS, weights="w")
    large_names = [f"v{i}" for i in range(64)]
    with pytest.raises(AnalysisError, match="operation budget"):
        canon({name: range(150_000) for name in [*large_names, "w"]}, large_names[:32], large_names[32:], weights="w")
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        canon({name: range(100_000) for name in [*NAMES, "w"]}, XS, YS, weights="w")
    # Matrix-variable gate is checked before conversion of an arbitrary object.
    with pytest.raises(AnalysisError, match="64 joint"):
        canon_matrix(object(), n=100, x=[f"x{i}" for i in range(40)], y=[f"y{i}" for i in range(25)])


def test_cpu_admission_and_unknown_count_source_budget(monkeypatch):
    with pytest.raises(AnalysisError, match="CPU"):
        canon({name: torch.empty(40, device="meta") for name in [*NAMES, "w"]}, XS, YS, weights="w")
    with pytest.raises(AnalysisError, match="CPU"):
        canon_matrix(torch.eye(5, device="meta"), n=100, x=XS, y=YS)
    frame = domain(n=40).assign(w=1)
    monkeypatch.setattr(options, "MAX_PHYSICAL_ROWS", 35)
    with pytest.raises(AnalysisError, match="source exceeds"):
        canon(stream(frame, size=13, count=False), XS, YS, weights="w")


def test_saved_state_refuses_missing_or_corrupt_labels_and_nonfinite_parameters():
    frame = domain()
    fitted = canon(frame, XS, YS)
    for kind in ("descriptives", "roles", "labels", "means", "coefficients"):
        result = restored(fitted)
        if kind == "descriptives":
            result.pop("descriptives")
        elif kind == "roles":
            result["raw_coefficients"].iloc[0, 0] = "y"
        elif kind == "labels":
            result["raw_coefficients"].index = list(reversed(NAMES))
        elif kind == "means":
            result["descriptives"].iloc[0, 1] = np.nan
        else:
            result["raw_coefficients"].iloc[0, 1] = np.inf
        with pytest.raises(AnalysisError, match="state"):
            canon_scores(result, frame)


def test_streaming_saved_score_source_integrity_and_all_missing_rows():
    frame = domain(n=50)
    result = canon(frame, XS, YS)
    backing = frame.copy()
    lazy = canon_scores(restored(result), stream(backing))
    backing.loc[0, "x1"] += 1
    with pytest.raises(AnalysisError, match="source changed"):
        list(lazy.iter_batches())
    holes = frame.assign(x1=np.nan)
    scored = canon_scores(result, holes)
    assert len(scored) == len(frame) and scored.isna().all().all()


def test_saved_projection_admits_width_before_converting_training_state(monkeypatch):
    names = [f"x{i}" for i in range(64)] + ["y"]
    roles = ["x"]*64 + ["y"]
    result = TableSet({
        "descriptives": pd.DataFrame({"set": roles, "mean": 0., "std_dev": 1.}, index=names),
        "raw_coefficients": pd.DataFrame({"set": roles, "Canon1": 1.}, index=names),
    }, procedure="canon", x=names[:-1], y=["y"])

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved-state conversion happened before width admission")

    monkeypatch.setattr(pd.DataFrame, "to_numpy", forbidden)
    with pytest.raises(AnalysisError, match="64 joint variables"):
        canon_scores(result, None)
