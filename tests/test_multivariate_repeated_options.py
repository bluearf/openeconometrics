"""Independent sums-of-squares, paired-t and Hotelling oracles for saved RM.

The ANOVA oracle is direct double-centering; epsilon uses the projection matrix
I-J/k, independently of the production contrast basis. Statsmodels supplies an
additional balanced uncorrected-F oracle. Exact Hotelling calibration is NIST's
one-sample law, not a generic Wald F(q,n-1) approximation.
"""
from __future__ import annotations

import copy
import json
from uuid import UUID

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import repeated as rm
from openecon.econometrics.stats.rm_anova import rm_anova


def domain(which):
    rng = np.random.default_rng(991 if which == "clinical" else 7103)
    n, k = (27, 4) if which == "clinical" else (41, 5)
    mixing = np.eye(k) + np.tril(np.ones((k, k)), -1) * .23
    if which == "clinical":
        values = 70 + rng.normal(size=(n, 1)) * 3 + np.arange(k) * 1.2
        values = values + rng.normal(size=(n, k)) @ mixing.T
        names = ["baseline", "week2", "week4", "week8"]
    else:
        values = 15 + rng.normal(size=(n, 1)) * .7 + np.array([0, -.5, 1, .25, 1.4])
        values = values + rng.normal(size=(n, k)) @ mixing.T * np.arange(1, k + 1)
        names = ["normal", "hot", "cold", "humid", "dry"]
    frame = pd.DataFrame(values, columns=names, index=[f"source-{n-i}" for i in range(n)])
    frame["participant"] = [f"id-{n-i}" for i in range(n)]
    return frame, names


def payload(result):
    return {"attrs": copy.deepcopy(result.attrs), "tables": {
        key: table.astype(object).where(table.notna(), None).to_dict(orient="split")
        for key, table in result.items()}}


def encode(result):
    return json.dumps(payload(result), allow_nan=False)


def long_data(x):
    n, k = x.shape
    return pd.DataFrame({"id": np.repeat(np.arange(n), k), "time": np.tile(np.arange(k), n),
                         "y": x.ravel()})


@pytest.mark.parametrize("which", ["clinical", "process"])
def test_wide_independent_ss_epsilon_and_long_existing_core(which):
    data, names = domain(which)
    # Nonalphabetical input order must survive both the adapter and saved state.
    names = names[::-1]
    x = data[names].to_numpy()
    n, k = x.shape
    result = rm.rm_anova_wide(data, names, subject="participant", alpha=.1)
    means, grand = x.mean(0), x.mean()
    residual = x - x.mean(1)[:, None] - means + grand
    ss_effect = n * np.square(means - grand).sum()
    ss_error = np.square(residual).sum()
    ss_subject = k * np.square(x.mean(1) - grand).sum()
    df = k - 1
    within = result["within"].set_index(["source", "correction"])
    row = within.loc[("measurement", "sphericity_assumed")]
    assert row.ss == pytest.approx(ss_effect, rel=5e-12)
    assert row.statistic == pytest.approx((ss_effect / df) / (ss_error / (df * (n - 1))), rel=5e-12)
    assert within.loc[("error(measurement)", "sphericity_assumed"), "ss"] == pytest.approx(ss_error)
    assert result["between"].loc["error", "ss"] == pytest.approx(ss_subject)
    s = np.cov(x, rowvar=False, ddof=1)
    h = np.eye(k) - np.ones((k, k)) / k
    projected = h @ s @ h
    epsilon = np.trace(projected)**2 / (df * np.trace(projected @ projected))
    hf = min(1, (n * df * epsilon - 2) / (df * (n - 1 - df * epsilon)))
    roots = np.linalg.eigvalsh(projected)[1:]
    log_w = np.log(roots).sum() - df * np.log(roots.sum() / df)
    chi = -(n - 1 - (2 * df**2 + df + 2) / (6 * df)) * log_w
    sph = result["sphericity"].iloc[0]
    assert sph.epsilon_gg == pytest.approx(epsilon, rel=3e-12)
    assert sph.epsilon_hf == pytest.approx(hf, rel=3e-12)
    assert sph.mauchly_w == pytest.approx(np.exp(log_w), rel=3e-12)
    assert sph.chi2 == pytest.approx(chi, rel=3e-12)
    assert sph.p_value == pytest.approx(stats.chi2.sf(chi, df * (df + 1) / 2 - 1), rel=3e-11)
    for correction, eps in [("greenhouse_geisser", epsilon), ("huynh_feldt", hf), ("lower_bound", 1 / df)]:
        corrected = within.loc[("measurement", correction)]
        assert corrected.statistic == pytest.approx(row.statistic)
        assert corrected.df == pytest.approx(df * eps)
        assert corrected.p_value == pytest.approx(stats.f.sf(row.statistic, df * eps, df * (n - 1) * eps), rel=3e-10)
    existing = rm_anova(long_data(x), "y", "id", ["time"], alpha=.1, missing="raise")
    np.testing.assert_allclose(result["within"].select_dtypes("number"), existing["within"].select_dtypes("number"), equal_nan=True)
    np.testing.assert_allclose(result["sphericity"], existing["sphericity"], equal_nan=True)
    np.testing.assert_allclose(result["covariance"], s, atol=2e-12)
    np.testing.assert_allclose(result["mean_covariance"], s / n, atol=2e-12)
    np.testing.assert_allclose(result["means"]["mean"], means)
    np.testing.assert_array_equal(result["sample"], x)
    assert result.attrs["measurement_columns"] == names
    assert result.attrs["n_subjects"] == n and result.attrs["n"] == n * k
    assert result.attrs["anova_alpha"] == .1
    assert "first-order" in result.attrs["sphericity_convention"]
    assert "long numeric" in " ".join(str(result.attrs["resource_plan"]).split())


@pytest.mark.parametrize("which", ["clinical", "process"])
def test_statsmodels_balanced_uncorrected_f(which):
    anova = pytest.importorskip("statsmodels.stats.anova")
    data, names = domain(which)
    x = data[names].to_numpy()
    result = rm.rm_anova_wide(data, names)
    reference = anova.AnovaRM(long_data(x), "y", "id", within=["time"]).fit().anova_table
    actual = result["within"].iloc[0]
    assert actual.statistic == pytest.approx(reference.loc["time", "F Value"], rel=2e-11)
    assert actual.p_value == pytest.approx(reference.loc["time", "Pr > F"], rel=3e-10)


@pytest.mark.parametrize("which", ["clinical", "process"])
def test_joint_contrast_numpy_covariance_t_hotelling_and_json(which):
    data, names = domain(which)
    x = data[names].to_numpy()
    n, k = x.shape
    c = np.zeros((2, k))
    c[0, :2] = [-1, 1]
    c[1, :3] = [1, -2, 1]
    null = np.array([.2, -.15])
    original = rm.rm_anova_wide(data, names, subject="participant", alpha=.1)
    source = rm.rm_restore(encode(original))
    fit = rm.rm_contrasts(source, c, null=null, alpha=.025, contrast_names=["early_change", "curvature"])
    d = x @ c.T
    estimates, cov = d.mean(0), np.cov(d, rowvar=False, ddof=1) / n
    se = np.sqrt(np.diag(cov))
    marginal = (estimates - null) / se
    t_squared = (estimates - null) @ np.linalg.solve(cov, estimates - null)
    f = (n - 2) / (2 * (n - 1)) * t_squared
    np.testing.assert_allclose(fit["subject_contrasts"], d, atol=5e-14)
    np.testing.assert_allclose(fit["joint_covariance"], cov, atol=2e-14)
    np.testing.assert_allclose(fit["joint_covariance"], c @ np.cov(x, rowvar=False) @ c.T / n, atol=2e-14)
    np.testing.assert_allclose(fit["estimates"]["estimate"], estimates, atol=3e-14)
    np.testing.assert_allclose(fit["estimates"]["std_error"], se, atol=2e-14)
    np.testing.assert_allclose(fit["estimates"]["t"], marginal, atol=3e-13)
    np.testing.assert_allclose(fit["estimates"]["p_value"], 2 * stats.t.sf(abs(marginal), n - 1), atol=1e-12)
    np.testing.assert_allclose(fit["estimates"]["ci_lower"], estimates - stats.t.isf(.025 / 2, n - 1) * se, atol=5e-11)
    np.testing.assert_allclose(fit["estimates"]["ci_upper"], estimates + stats.t.isf(.025 / 2, n - 1) * se, atol=5e-11)
    joint = fit["joint"].iloc[0]
    assert joint.t_squared == pytest.approx(t_squared, rel=4e-12)
    assert joint.statistic == pytest.approx(f, rel=4e-12)
    assert joint.df_numerator == 2 and joint.df_denominator == n - 2
    assert joint.p_value == pytest.approx(stats.f.sf(f, 2, n - 2), rel=3e-11)
    assert "Marginal" in fit.attrs["intervals"] and "not verified" in fit.attrs["contrast_assumptions"]
    assert fit.attrs["anova_alpha"] == .1 and fit.attrs["alpha"] == .025
    restored = rm.rm_restore(encode(fit))
    for key in fit:
        pd.testing.assert_frame_equal(restored[key], fit[key])
    assert restored.attrs["contrast_content_sha256"] == fit.attrs["contrast_content_sha256"]
    assert len(restored.attrs["subject_labels_encoded"]) == n


def test_two_measurement_paired_t_and_no_mauchly_claim():
    data, names = domain("clinical")
    names = names[:2]
    x = data[names].to_numpy()
    result = rm.rm_anova_wide(data, names)
    fit = rm.rm_contrasts(result, [[-1, 1]])
    reference = stats.ttest_rel(x[:, 1], x[:, 0])
    assert fit["estimates"].iloc[0].t == pytest.approx(reference.statistic, rel=3e-12)
    assert fit["estimates"].iloc[0].p_value == pytest.approx(reference.pvalue, rel=3e-11)
    assert fit["joint"].iloc[0].statistic == pytest.approx(reference.statistic**2, rel=3e-12)
    assert fit["joint"].iloc[0].df_denominator == len(x) - 1
    assert pd.isna(result["sphericity"].iloc[0].p_value)
    assert result["sphericity"].iloc[0].epsilon_gg == 1


def test_drop_subject_preserves_physical_positions_typed_ids_and_index():
    data, names = domain("clinical")
    data.loc[data.index[3], names[1]] = np.nan
    data.loc[data.index[5], names[0]] = np.nan
    ids = [UUID(int=i + 1) for i in range(len(data))]
    data["participant"] = ids
    data.index = pd.Index([1, "1", ("batch", 3)] + list(range(3, len(data))), dtype=object)
    result = rm.rm_anova_wide(data, names, subject="participant", missing="drop_subject")
    assert result.attrs["excluded_subjects"] == 2
    assert result.attrs["physical_subjects"] == 27
    assert result.attrs["sample_positions"] == [i for i in range(27) if i not in (3, 5)]
    assert result.attrs["n_missing_subjects"] == 2 and result.attrs["n_missing"] == 0
    restored = rm.rm_restore(encode(result))
    assert restored.attrs["subject_labels_encoded"] == result.attrs["subject_labels_encoded"]
    assert restored.attrs["row_labels_encoded"] == result.attrs["row_labels_encoded"]
    pd.testing.assert_frame_equal(restored["sample"], result["sample"])
    np.testing.assert_allclose(result["means"]["mean"], data.dropna(subset=names)[names].mean())


def test_mapping_records_and_unused_columns_agree():
    data, names = domain("process")
    reference = rm.rm_anova_wide(data, names)
    for source in (data.to_dict("list"), data.to_dict("records")):
        result = rm.rm_anova_wide(source, names)
        for key in reference:
            pd.testing.assert_frame_equal(reference[key], result[key])


@pytest.mark.parametrize("kind", ["duplicate_id", "missing_id", "incomplete", "inf", "complex", "string", "bool", "constant"])
def test_raw_refusals(kind):
    data, names = domain("clinical")
    if kind == "duplicate_id":
        data.loc[data.index[1], "participant"] = data.iloc[0].participant
    elif kind == "missing_id":
        data.loc[data.index[1], "participant"] = None
    elif kind == "incomplete":
        data.loc[data.index[1], names[1]] = np.nan
    elif kind == "inf":
        data.loc[data.index[1], names[1]] = np.inf
    elif kind == "complex":
        data[names[0]] = data[names[0]].astype(complex)
    elif kind == "string":
        data[names[0]] = "bad"
    elif kind == "bool":
        data[names[0]] = True
    elif kind == "constant":
        data[names] = 4.
    with pytest.raises(AnalysisError):
        rm.rm_anova_wide(data, names, subject="participant")


@pytest.mark.parametrize("option", [
    {"columns": ["baseline", "baseline"]}, {"columns": "baseline"},
    {"columns": ["baseline"]}, {"columns": ["baseline", "absent"]},
    {"subject": "baseline"}, {"alpha": 1}, {"alpha": True},
    {"missing": "drop"}, {"sampling_model": "dependent"},
])
def test_option_refusals(option):
    data, names = domain("clinical")
    kwargs = {"columns": names, **option}
    with pytest.raises(AnalysisError):
        rm.rm_anova_wide(data, **kwargs)


def test_nonresident_inputs_and_device_refused():
    data, names = domain("clinical")
    for source in (data[names].to_numpy(), torch.ones(4, 3), object()):
        with pytest.raises(AnalysisError, match="resident"):
            rm.rm_anova_wide(source, names)
    source = {name: torch.empty(5, device="meta") for name in names}
    with pytest.raises(AnalysisError, match="CPU"):
        rm.rm_anova_wide(source, names)


def test_admission_precedes_wide_and_long_materialization(monkeypatch):
    class LazyColumn:
        def __len__(self):
            return 10001

        def __iter__(self):
            raise AssertionError("Out-of-budget data were materialized")

    with pytest.raises(AnalysisError, match="10000"):
        rm.rm_anova_wide({"a": LazyColumn(), "b": LazyColumn()}, ["a", "b"])
    with pytest.raises(AnalysisError, match="32"):
        rm.rm_anova_wide(object(), [f"m{i}" for i in range(33)])
    data, names = domain("clinical")
    monkeypatch.setattr(rm, "_MAX_WORK", 1)
    with pytest.raises(AnalysisError, match="work"):
        rm.rm_anova_wide(data, names)


@pytest.mark.parametrize("field", [
    "source_content_sha256", "sample_positions", "subject_labels_encoded", "row_labels_encoded",
    "measurement_columns", "n_subjects", "n", "physical_subjects", "excluded_subjects",
    "missing", "anova_alpha", "alpha", "precision", "device", "covariance_divisor",
    "sampling_model", "state_schema", "n_missing_subjects",
])
def test_saved_metadata_refuses_tampering(field):
    data, names = domain("clinical")
    record = payload(rm.rm_anova_wide(data, names, subject="participant"))
    previous = record["attrs"][field]
    if isinstance(previous, list):
        record["attrs"][field] = previous[::-1]
    elif isinstance(previous, str):
        record["attrs"][field] = "invalid"
    elif isinstance(previous, float):
        record["attrs"][field] = .23
    else:
        record["attrs"][field] = previous + 1
    with pytest.raises(AnalysisError):
        rm.rm_restore(record)


@pytest.mark.parametrize("table", list(rm._CORE + rm._STATE))
def test_saved_raw_moments_and_anova_tables_revalidated(table):
    data, names = domain("clinical")
    record = payload(rm.rm_anova_wide(data, names))
    values = record["tables"][table]["data"]
    if table in ("within", "descriptives"):
        values[0][2] += 1
    else:
        values[0][0] += 1
    with pytest.raises(AnalysisError):
        rm.rm_restore(record)


@pytest.mark.parametrize("matrix", [
    [[0, 0, 0, 0]], [[1, 0, 0, 0]], [[-1, 1, 0]],
    [[-1, 1, 0, 0], [-2, 2, 0, 0]],
    [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [-1, 0, 0, 1]],
    [[np.nan, 1, 0, 0]], [[np.inf, -np.inf, 0, 0]], [],
])
def test_contrast_matrix_refusals(matrix):
    data, names = domain("clinical")
    result = rm.rm_anova_wide(data, names)
    with pytest.raises(AnalysisError):
        rm.rm_contrasts(result, matrix)


@pytest.mark.parametrize("kwargs", [
    {"null": [0, 0]}, {"null": [np.inf]}, {"null": True},
    {"contrast_names": ["a", "a"]}, {"contrast_names": "a"},
    {"alpha": 0}, {"sampling_model": "clustered"},
])
def test_contrast_option_refusals(kwargs):
    data, names = domain("clinical")
    result = rm.rm_anova_wide(data, names)
    with pytest.raises(AnalysisError):
        rm.rm_contrasts(result, [[-1, 1, 0, 0]], **kwargs)


def test_labeled_contrast_order_is_explicit_and_retained():
    data, names = domain("clinical")
    result = rm.rm_anova_wide(data, names)
    c = pd.DataFrame([[-1, 1, 0, 0]], index=["early_change"], columns=names)
    fitted = rm.rm_contrasts(result, c)
    assert fitted["estimates"].index.tolist() == ["early_change"]
    with pytest.raises(AnalysisError, match="order"):
        rm.rm_contrasts(result, c[names[::-1]])


def test_n_must_exceed_q_and_covariance_must_be_identified():
    data, names = domain("clinical")
    tiny = rm.rm_anova_wide(data.iloc[:2], names)
    with pytest.raises(AnalysisError, match="more complete"):
        rm.rm_contrasts(tiny, [[-1, 1, 0, 0], [0, 0, -1, 1]])
    rng = np.random.default_rng(813)
    x = rng.normal(size=(20, 2))
    flat = pd.DataFrame({"a": x[:, 0], "b": x[:, 0] + 1, "c": x[:, 1]})
    result = rm.rm_anova_wide(flat, ["a", "b", "c"])
    with pytest.raises(AnalysisError, match="variation"):
        rm.rm_contrasts(result, [[-1, 1, 0]])
    # Contrast rows have full algebraic rank, but their observed covariance does not.
    # c=2*b-a gives b-a and c-b equal, despite independent contrast coefficients.
    paired = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": 2 * x[:, 1] - x[:, 0]})
    fit = rm.rm_anova_wide(paired, ["a", "b", "c"])
    with pytest.raises(AnalysisError, match="singular"):
        rm.rm_contrasts(fit, [[-1, 1, 0], [0, -1, 1]])


@pytest.mark.parametrize("table", list(rm._CONTRAST))
def test_saved_contrast_joint_covariance_null_and_vectors_revalidated(table):
    data, names = domain("clinical")
    fit = rm.rm_contrasts(rm.rm_anova_wide(data, names), [[-1, 1, 0, 0]])
    record = payload(fit)
    record["tables"][table]["data"][0][0] += 1
    with pytest.raises(AnalysisError):
        rm.rm_restore(record)


def test_contrast_scaling_and_common_subject_level_invariance():
    data, names = domain("clinical")
    x = data[names].to_numpy()
    c = np.array([[-1, 1, 0, 0], [1, -2, 1, 0.]])
    base = rm.rm_anova_wide(data, names)
    normal = rm.rm_contrasts(base, c)
    scale = np.array([-2., 1e9])
    scaled = rm.rm_contrasts(base, c * scale[:, None])
    np.testing.assert_allclose(scaled["joint_covariance"], np.asarray(normal["joint_covariance"]) * scale[:, None] * scale[None, :], rtol=2e-12)
    assert scaled["joint"].iloc[0].statistic == pytest.approx(normal["joint"].iloc[0].statistic, rel=2e-12)
    np.testing.assert_allclose(scaled["estimates"]["t"], normal["estimates"]["t"] * np.sign(scale), rtol=2e-12)
    # The common level cancels before contrast multiplication; spacing sets the tolerance.
    shifted = data.copy()
    shifted[names] = x + 1e9 + np.arange(len(x))[:, None] * 1e4
    moved = rm.rm_contrasts(rm.rm_anova_wide(shifted, names), c)
    np.testing.assert_allclose(moved["subject_contrasts"], normal["subject_contrasts"], atol=3e-7)
    np.testing.assert_allclose(moved["joint_covariance"], normal["joint_covariance"], atol=4e-8)


def test_portable_payload_geometry_admitted_before_constructing_tables():
    data, names = domain("clinical")
    record = payload(rm.rm_anova_wide(data, names))
    record["tables"]["sample"]["data"] = [[1, 2, 3, 4]] * 10001
    record["tables"]["sample"]["index"] = list(range(10001))
    with pytest.raises(AnalysisError, match="bounded"):
        rm.rm_restore(record)
    for source in ({}, "{bad", {"attrs": {}, "tables": {}}, data):
        with pytest.raises(AnalysisError):
            rm.rm_restore(source)


def test_saved_within_covariance_avoids_large_subject_level_cancellation():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(37, 1)) * 1e10 + rng.normal(size=(37, 3)) + [0, .5, 1.]
    data = pd.DataFrame(x, columns=["a", "b", "c"])
    result = rm.rm_anova_wide(data, ["a", "b", "c"])
    b = result["within_basis"].to_numpy()
    within = (x - x[:, :1]) @ b
    reference = np.cov(within, rowvar=False, ddof=1)
    np.testing.assert_allclose(result["contrast_covariance"], reference, atol=4e-14)
    assert np.linalg.eigvalsh(result["contrast_covariance"]).min() > .5
    pd.testing.assert_frame_equal(rm.rm_restore(encode(result))["contrast_covariance"],
                                  result["contrast_covariance"])


def test_adapter_refuses_nonzero_within_error_truncated_by_existing_core():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(37, 1)) * 1e12 + rng.normal(size=(37, 3)) + [0, .5, 1.]
    assert np.square((x - x[:, :1]) - (x - x[:, :1]).mean(0)).sum() > 10
    with pytest.raises(AnalysisError, match="cannot resolve"):
        rm.rm_anova_wide(pd.DataFrame(x, columns=["a", "b", "c"]), ["a", "b", "c"])


def test_near_zero_row_sum_cannot_change_large_common_level_estimand():
    rng = np.random.default_rng(813)
    x = 1e12 + rng.normal(size=(29, 3)) + [0, .4, 1.]
    base = rm.rm_anova_wide(pd.DataFrame(x, columns=["a", "b", "c"]), ["a", "b", "c"])
    with pytest.raises(AnalysisError, match="exactly"):
        rm.rm_contrasts(base, [[-1, 1 + 5e-13, 0]])
    exact = rm.rm_contrasts(base, [[-1, 1, 0]])
    assert exact["estimates"].iloc[0].estimate == pytest.approx(np.mean(x[:, 1] - x[:, 0]), abs=1e-14)


@pytest.mark.parametrize("contrasts", [np.array([[-1 + 1j, 1, 0, 0]]),
                                      [[-1 + 1j, 1, 0, 0]],
                                      torch.tensor([[-1, 1, 0, 0]], device="meta")])
def test_complex_and_non_cpu_contrasts_refused_without_coercion(contrasts):
    data, names = domain("clinical")
    with pytest.raises(AnalysisError):
        rm.rm_contrasts(rm.rm_anova_wide(data, names), contrasts)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_unrepresentable_saved_joint_covariance_is_refused(scale):
    data, names = domain("clinical")
    with pytest.raises(AnalysisError, match="float64"):
        rm.rm_contrasts(rm.rm_anova_wide(data, names), [[-scale, scale, 0, 0]])


def test_extreme_variance_failure_is_analysis_error_from_unchanged_core():
    data, names = domain("clinical")
    data[names] = data[names] * 1e100
    with pytest.raises(AnalysisError, match="float64"):
        rm.rm_anova_wide(data, names)


def test_workspace_refuses_before_selected_or_long_copies(monkeypatch):
    from openecon.resources import use_workspace_budget

    data, names = domain("clinical")
    def forbid(*args, **kwargs):
        raise AssertionError("Long data allocated before workspace admission")

    monkeypatch.setattr(rm, "_core", forbid)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        rm.rm_anova_wide(data, names)


def test_integer_saved_counts_cannot_be_boolean_or_fractional():
    data, names = domain("clinical")
    result = rm.rm_anova_wide(data.iloc[:2], names)
    for value in (True, 1.0):
        record = payload(result)
        record["attrs"]["covariance_divisor"] = value
        with pytest.raises(AnalysisError, match="counts"):
            rm.rm_restore(record)
