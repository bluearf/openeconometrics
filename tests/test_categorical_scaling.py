"""Independent descriptive geometry, constrained objectives and full saved state."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import isotonic_regression

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical.scaling import (
    _disparities,
    _pava,
    mca,
    mca_project,
    mds_nonmetric,
    overals,
)
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


@pytest.fixture
def frame():
    rng = np.random.default_rng(120)
    group = rng.integers(0, 3, 90)
    return pd.DataFrame(
        {
            "a": group,
            "b": rng.integers(0, 2, 90),
            "c": np.where(rng.random(90) < 0.7, group, rng.integers(0, 3, 90)),
            "d": rng.integers(0, 3, 90),
        },
        index=[f"person-{i}" for i in range(90)],
    )


def indicators(frame, variables, levels=None):
    if levels is None:
        levels = [list(dict.fromkeys(frame[v])) for v in variables]
    return np.column_stack(
        [np.array(frame[v]) == value for v, order in zip(variables, levels) for value in order]
    ).astype(float)


def test_mca_full_numpy_geometry_and_raw_inertia(frame):
    result = mca(frame, list(frame), n_components=3)
    calibration = result.attrs["calibration"]
    z = indicators(frame, list(frame), calibration["levels"])
    mass = z.mean(0) / 4
    s = (z / 4 - mass) / np.sqrt(mass) / np.sqrt(len(frame))
    u, singular, vt = np.linalg.svd(s, full_matrices=False)
    np.testing.assert_allclose(result["inertia"].raw_inertia, singular[:7] ** 2, atol=1e-13)
    reference = u[:, :3] * singular[:3] * np.sqrt(len(frame))
    actual = result["row_coordinates"].to_numpy()
    np.testing.assert_allclose(actual @ actual.T, reference @ reference.T, atol=1e-11)
    category = vt[:3].T / np.sqrt(mass)[:, None] * singular[:3]
    actual = result["category_coordinates"].iloc[:, 3:].to_numpy()
    np.testing.assert_allclose(actual @ actual.T, category @ category.T, atol=1e-11)
    assert result.attrs["raw_total_inertia"] == pytest.approx((z.shape[1] - 4) / 4)
    np.testing.assert_allclose(result["row_contributions"].sum(0), 1.0)
    np.testing.assert_allclose(result["category_contributions"].sum(0), 1.0)


def test_mca_exact_supplementary_transition_after_restore(frame):
    result = mca(frame, list(frame))
    restored = restore_summary(summary_state(result))
    projection = mca_project(restored, frame.iloc[[0, 5, 8]])
    np.testing.assert_allclose(
        projection["row_coordinates"], result["row_coordinates"].iloc[[0, 5, 8]], atol=1e-14
    )
    calibration = restored.attrs["calibration"]
    z = indicators(frame.iloc[[0, 5, 8]], list(frame), calibration["levels"])
    expected = (z / 4 - np.asarray(calibration["category_mass"])) @ np.asarray(
        calibration["category_standard"]
    )
    np.testing.assert_allclose(projection["row_coordinates"], expected, atol=1e-14)
    assert list(projection["row_coordinates"].index) == list(frame.index[[0, 5, 8]])


def test_mca_listwise_source_positions_and_projection_missing(frame):
    damaged = frame.copy()
    damaged.loc[damaged.index[2], "a"] = np.nan
    damaged.loc[damaged.index[6], "b"] = np.nan
    result = mca(damaged, list(frame))
    assert result.attrs["sample_positions"] == [i for i in range(90) if i not in (2, 6)]
    clean = mca(damaged.dropna(), list(frame))
    np.testing.assert_allclose(result["row_coordinates"], clean["row_coordinates"], atol=1e-14)
    with pytest.raises(AnalysisError):
        mca(damaged, list(frame), missing="raise")
    projection = mca_project(result, damaged.iloc[:7], missing="drop")
    assert projection.attrs["sample_positions"] == [0, 1, 3, 4, 5]


def test_mca_typed_categories_unknown_and_tamper():
    frame = pd.DataFrame(
        {"a": pd.Series([True, 1, "1", True, 1, "1"], dtype=object), "b": ["x", "y"] * 3}
    )
    result = mca(frame, ["a", "b"])
    assert len(result.attrs["calibration"]["levels"][0]) == 3
    np.testing.assert_allclose(
        mca_project(result, frame)["row_coordinates"], result["row_coordinates"], atol=1e-14
    )
    unknown = frame.iloc[[0]].copy()
    unknown.loc[0, "a"] = "never-fitted"
    with pytest.raises(AnalysisError, match="unfitted"):
        mca_project(result, unknown)
    damaged = restore_summary(summary_state(result))
    damaged.attrs["calibration"]["category_standard"][0][0] += 0.01
    with pytest.raises(AnalysisError, match="checksum"):
        mca_project(damaged, frame)
    damaged = restore_summary(summary_state(result))
    damaged["category_standard"].iloc[0, 0] += 0.01
    with pytest.raises(AnalysisError, match="checksum"):
        mca_project(damaged, frame)


def reference_projection_loss(frame, sets, d, *, numeric=False):
    n = len(frame)
    mean = np.zeros((n, n))
    for group in sets:
        z = frame[group].to_numpy(float) if numeric else indicators(frame, group)
        z = z - z.mean(0)
        mean += z @ np.linalg.pinv(z)
    mean /= len(sets)
    eigen, vectors = np.linalg.eigh(mean)
    x = vectors[:, -d:] * np.sqrt(n)
    return d - eigen[-d:].sum(), x @ x.T


def test_overals_multiple_nominal_is_genuine_multiset_projection_objective(frame):
    sets = [["a", "b"], ["c", "d"]]
    result = overals(
        frame,
        sets,
        scales={name: "multiple_nominal" for name in frame},
        n_starts=3,
        max_iter=1000,
        tol=1e-10,
    )
    expected_loss, gram = reference_projection_loss(frame, sets, 2)
    assert result.attrs["objective"] == pytest.approx(expected_loss, abs=3e-7)
    actual = result["object_scores"].to_numpy()
    np.testing.assert_allclose(actual @ actual.T, gram, atol=0.006)
    np.testing.assert_allclose(actual.mean(0), 0.0, atol=1e-14)
    np.testing.assert_allclose(actual.T @ actual / len(frame), np.eye(2), atol=1e-12)
    fits = result["set_scores"].iloc[:, 3:].to_numpy().reshape(2, len(frame), 2)
    assert np.mean(np.sum((actual[None] - fits) ** 2, axis=2)) == pytest.approx(
        result.attrs["objective"], abs=1e-12
    )
    assert len(result["starts"]) == 3
    for _, trace in result["iterations"].groupby("start"):
        assert (np.diff(trace.loss) <= 1e-10).all()


def test_homogeneity_single_variable_sets_reduce_to_mca_axes(frame):
    variables = ["a", "b", "c"]
    result = overals(
        frame,
        [[v] for v in variables],
        scales={v: "multiple_nominal" for v in variables},
        max_iter=1000,
        n_starts=2,
        tol=1e-10,
    )
    ordinary = mca(frame, variables)
    eigen = ordinary["inertia"].raw_inertia.to_numpy()[:2]
    assert result.attrs["objective"] == pytest.approx(2 - eigen.sum(), abs=1e-7)
    x = ordinary["row_coordinates"].to_numpy() / np.sqrt(eigen)
    actual = result["object_scores"].to_numpy()
    np.testing.assert_allclose(actual @ actual.T, x @ x.T, atol=0.002)


def test_overals_numeric_linear_set_objective_has_independent_reference():
    rng = np.random.default_rng(513)
    x = rng.normal(size=(65, 4))
    x[:, 2:] += x[:, :2] * 0.8
    frame = pd.DataFrame(x, columns=list("abcd"))
    sets = [["a", "b"], ["c", "d"]]
    result = overals(
        frame, sets, scales={v: "numeric" for v in frame}, n_starts=2, max_iter=1000, tol=1e-10
    )
    expected, gram = reference_projection_loss(frame, sets, 2, numeric=True)
    assert result.attrs["objective"] == pytest.approx(expected, abs=2e-7)
    actual = result["object_scores"].to_numpy()
    np.testing.assert_allclose(actual @ actual.T, gram, atol=0.003)


def test_overals_ordinal_nominal_rank_and_normalization(frame):
    scales = {"a": "ordinal", "b": "nominal", "c": "multiple_nominal", "d": "numeric"}
    result = overals(
        frame,
        [["a", "b"], ["c", "d"]],
        scales=scales,
        orders={"a": [2, 0, 1]},
        n_starts=2,
        max_iter=1000,
        tol=1e-8,
    )
    assert result.attrs["levels"][0] == [2, 0, 1]
    codes = result.attrs["codes"]
    for i, name in enumerate(result.attrs["variables"]):
        quant = np.asarray(result.attrs["quantifications"][i])
        loading = np.asarray(result.attrs["loadings"][i])
        assert quant.shape[1] == (2 if scales[name] == "multiple_nominal" else 1)
        if scales[name] != "multiple_nominal":
            transformed = quant[np.asarray(codes[i])]
            np.testing.assert_allclose(transformed.mean(0), 0.0, atol=1e-12)
            np.testing.assert_allclose((transformed**2).mean(0), 1.0, atol=1e-12)
        if scales[name] == "ordinal":
            assert (np.diff(quant[:, 0]) >= -1e-12).all()
        assert loading.shape[1] == 2
    assert result.attrs["global_optimum"] is False and result.attrs["inference"] is False


def test_overals_saved_maps_reconstruct_each_complete_set_fit(frame):
    result = overals(
        frame, [["a", "b"], ["c", "d"]], scales={v: "nominal" for v in frame}, n_starts=2
    )
    restored = restore_summary(summary_state(result))
    for set_index, group in enumerate(restored.attrs["sets"]):
        predicted = np.zeros((len(frame), 2))
        for variable in group:
            i = restored.attrs["variables"].index(variable)
            quant = np.asarray(restored.attrs["quantifications"][i])
            loading = np.asarray(restored.attrs["loadings"][i])
            predicted += quant[np.asarray(restored.attrs["codes"][i])] @ loading
        actual = restored["set_scores"].query("set == @set_index").iloc[:, 3:].to_numpy()
        np.testing.assert_allclose(actual, predicted, atol=1e-12)


def test_weighted_pava_is_independent_scipy_cone_projection():
    y = np.array([5.0, 3.0, 4.0, -1.0, 6.0, 5.0, 10.0])
    weights = np.array([2.0, 1.0, 4.0, 7.0, 1.0, 3.0, 2.0])
    expected = isotonic_regression(y, weights=weights).x
    actual = _pava(torch.tensor(y), torch.tensor(weights)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("ties", ["primary", "secondary"])
def test_disparity_projection_matches_scipy_with_tie_semantics(ties):
    observed = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 1.0])
    distances = np.array([0.4, 2.0, 1.0, 0.7, 0.5, 2.0, 0.2])
    if ties == "primary":
        order = np.lexsort((distances, observed))
        fit = isotonic_regression(distances[order]).x
        expected = np.empty(len(fit))
        expected[order] = fit
    else:
        _, inverse, counts = np.unique(observed, return_inverse=True, return_counts=True)
        means = np.bincount(inverse, weights=distances) / counts
        expected = isotonic_regression(means, weights=counts).x[inverse]
    expected *= np.sqrt(len(expected) / np.sum(expected**2))
    actual = _disparities(torch.tensor(observed), torch.tensor(distances), ties).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    for value in np.unique(observed):
        if ties == "secondary":
            assert np.ptp(actual[observed == value]) == 0
    assert actual @ actual == pytest.approx(len(actual))


def reference_nmds(delta, init, tol, max_iter, zero="include"):
    n = len(delta)
    ii, jj = np.triu_indices(n, 1)
    if zero == "exclude":
        keep = delta[ii, jj] > 0
        ii, jj = ii[keep], jj[keep]
    obs = delta[ii, jj]
    _, inverse, counts = np.unique(obs, return_inverse=True, return_counts=True)
    w = np.zeros((n, n))
    w[ii, jj] = 1
    w[jj, ii] = 1
    vinv = np.linalg.pinv(np.diag(w.sum(1)) - w)
    x = init - init.mean(0)
    x *= np.sqrt(len(obs) / np.sum((x[ii] - x[jj]) ** 2))
    prior = np.inf
    for iteration in range(max_iter + 1):
        distance = np.linalg.norm(x[ii] - x[jj], axis=1)
        means = np.bincount(inverse, weights=distance) / counts
        hat = isotonic_regression(means, weights=counts).x[inverse]
        hat *= np.sqrt(len(obs) / np.sum(hat**2))
        stress = np.mean((distance - hat) ** 2)
        if iteration and prior - stress <= tol * max(prior, 0.001):
            break
        b = np.zeros((n, n))
        ratio = np.divide(hat, distance, out=np.zeros_like(hat), where=distance > 1e-12)
        b[ii, jj] = -ratio
        b[jj, ii] = -ratio
        np.fill_diagonal(b, -b.sum(1))
        x = vinv @ b @ x
        x -= x.mean(0)
        prior = stress
    return distance, hat, stress


@pytest.fixture
def dissimilarities():
    rng = np.random.default_rng(411)
    x = rng.normal(size=(10, 3))
    delta = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    delta = np.round(delta * 2) ** 1.7
    np.fill_diagonal(delta, 0)
    return delta


def test_nmds_full_independent_majorization_reference(dissimilarities):
    initial = np.random.default_rng(418).normal(size=(10, 2))
    distance, disparity, stress = reference_nmds(dissimilarities, initial, 1e-9, 1000)
    result = mds_nonmetric(dissimilarities, n_starts=1, init=initial, max_iter=1000, tol=1e-9)
    np.testing.assert_allclose(result["pairs"].distance, distance, atol=2e-10)
    np.testing.assert_allclose(result["pairs"].disparity, disparity, atol=2e-10)
    assert result.attrs["normalized_stress"] == pytest.approx(stress, abs=1e-12)
    assert result.attrs["stress_1"] == pytest.approx(
        np.sqrt(stress * len(distance) / sum(distance**2))
    )
    x = result["coordinates"].to_numpy()
    np.testing.assert_allclose(x.mean(0), 0.0, atol=1e-14)
    np.testing.assert_allclose((x.T @ x)[0, 1], 0.0, atol=1e-12)


def test_nmds_nonlinear_monotone_input_is_not_metric_mds():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    distance = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    result = mds_nonmetric(distance**3, init=points, n_starts=3, tol=1e-9)
    assert result.attrs["stress_1"] < 1e-10
    assert len(result["starts"]) == 3
    fit = result["pairs"]
    # The learned disparities are neither the input values nor one fixed ratio.
    assert np.ptp(fit.disparity / fit.dissimilarity) > 0.1
    normalized = distance[np.triu_indices(6, 1)]
    normalized *= np.sqrt(len(normalized) / sum(normalized**2))
    np.testing.assert_allclose(fit.distance, normalized, atol=1e-12)


def test_nmds_ties_zeros_and_monotone_stress_trace(dissimilarities):
    dissimilarities[0, 1] = dissimilarities[1, 0] = 0.0
    included = mds_nonmetric(dissimilarities, n_starts=3, max_iter=1000)
    assert len(included["pairs"]) == 45
    assert (included["pairs"].dissimilarity == 0).sum() >= 1
    excluded = mds_nonmetric(dissimilarities, zero="exclude", n_starts=2, max_iter=1000)
    assert len(excluded["pairs"]) < 45
    for _, block in included["pairs"].groupby("dissimilarity"):
        assert block.disparity.max() == block.disparity.min()
    grouped = included["pairs"].groupby("dissimilarity").disparity.mean()
    assert (np.diff(grouped) >= -1e-12).all()
    for _, trace in included["iterations"].groupby("start"):
        assert (np.diff(trace.normalized_stress) <= 1e-10).all()
    assert sum(included["pairs"].disparity ** 2) == pytest.approx(45.0)


def test_nmds_excluded_zero_graph_matches_independent_reference(dissimilarities):
    dissimilarities[0, 1] = dissimilarities[1, 0] = 0.0
    initial = np.random.default_rng(419).normal(size=(10, 2))
    d, h, s = reference_nmds(dissimilarities, initial, 1e-9, 1000, "exclude")
    actual = mds_nonmetric(
        dissimilarities, init=initial, zero="exclude", n_starts=1, max_iter=1000, tol=1e-9
    )
    np.testing.assert_allclose(actual["pairs"].distance, d, atol=2e-10)
    np.testing.assert_allclose(actual["pairs"].disparity, h, atol=2e-10)
    assert actual.attrs["normalized_stress"] == pytest.approx(s, abs=1e-12)


@pytest.mark.parametrize("kind", ["mca", "project", "overals", "nmds"])
def test_complete_json_and_latex_restore(kind, frame, dissimilarities):
    if kind == "mca":
        result = mca(frame, list(frame))
    elif kind == "project":
        result = mca_project(mca(frame, list(frame)), frame.iloc[:3])
    elif kind == "overals":
        result = overals(
            frame, [["a", "b"], ["c", "d"]], scales={v: "nominal" for v in frame}, n_starts=2
        )
    else:
        result = mds_nonmetric(dissimilarities, n_starts=2)
    state = summary_state(result)
    restored = restore_summary(state)
    assert summary_state(restored) == state
    assert result.to_latex() == restored.to_latex()
    assert result.attrs == restored.attrs
    for name in result:
        assert result[name].equals(restored[name])


@pytest.mark.parametrize("method", ["mca", "overals", "nmds"])
def test_budget_admission_precedes_linalg(method, frame, dissimilarities, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Linear algebra was reached before the budget guard")

    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    monkeypatch.setattr(torch.linalg, "eigh", forbidden)

    def call(**kwargs):
        if method == "overals":
            return overals(
                frame, [["a", "b"], ["c", "d"]], scales={v: "nominal" for v in frame}, **kwargs
            )
        if method == "nmds":
            return mds_nonmetric(dissimilarities, **kwargs)
        return mca(frame, list(frame), **kwargs)

    for options in ({"max_work": 1}, {"max_bytes": 1}):
        with pytest.raises(AnalysisError):
            call(**options)
    large = pd.concat([frame] * 12, ignore_index=True)
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        if method == "mca":
            mca(large, list(large))
        elif method == "overals":
            overals(large, [["a", "b"], ["c", "d"]], scales={v: "nominal" for v in large})
        else:
            mds_nonmetric(np.zeros((80, 80)))


@pytest.mark.parametrize("method", ["mca", "overals", "nmds"])
def test_default_non_cpu_device_does_not_change_resident_domain(method, frame, dissimilarities):
    with torch.device("meta"):
        if method == "mca":
            result = mca(frame, list(frame))
        elif method == "overals":
            result = overals(
                frame, [["a", "b"], ["c", "d"]], scales={v: "nominal" for v in frame}, n_starts=2
            )
        else:
            result = mds_nonmetric(dissimilarities, n_starts=2)
    assert result.attrs["device"] == "cpu"
    assert result.attrs["dtype"] == "float64"


def test_nonconvergence_is_refused(frame, dissimilarities):
    with pytest.raises(AnalysisError, match="converged"):
        overals(
            frame,
            [["a", "b"], ["c", "d"]],
            scales={v: "nominal" for v in frame},
            n_starts=2,
            max_iter=1,
        )
    with pytest.raises(AnalysisError, match="converged"):
        mds_nonmetric(dissimilarities, n_starts=2, max_iter=1)


@pytest.mark.parametrize(
    "options", [{"n_components": True}, {"n_components": 13}, {"missing": "impute"}]
)
def test_mca_options_refused(frame, options):
    with pytest.raises(AnalysisError):
        mca(frame, list(frame), **options)


def test_mca_category_sample_column_guards(frame):
    with pytest.raises(AnalysisError):
        mca(frame.assign(a=1), list(frame))
    with pytest.raises(AnalysisError):
        mca(frame.assign(a=np.arange(len(frame))), list(frame))
    with pytest.raises(AnalysisError):
        mca(frame.iloc[:3], list(frame))
    with pytest.raises(AnalysisError):
        mca(pd.concat([frame, frame[["a"]]], axis=1), list(frame))
    with pytest.raises(AnalysisError):
        mca(frame, ["a", "a"])


@pytest.mark.parametrize(
    "change",
    [
        "missing_scales",
        "unknown_scale",
        "missing_order",
        "wrong_order",
        "overlap",
        "one_set",
        "rank",
        "starts",
        "tol",
    ],
)
def test_overals_explicit_scale_order_set_guards(frame, change):
    sets = [["a", "b"], ["c", "d"]]
    options = {"scales": {v: "nominal" for v in frame}}
    if change == "missing_scales":
        options.pop("scales")
    if change == "unknown_scale":
        options["scales"]["a"] = "smooth"
    if change in ("missing_order", "wrong_order"):
        options["scales"]["a"] = "ordinal"
    if change == "wrong_order":
        options["orders"] = {"a": [0, 1, 9]}
    if change == "overlap":
        sets[1][0] = "a"
    if change == "one_set":
        sets = [list(frame)]
    if change == "rank":
        options["n_components"] = 3
    if change == "starts":
        options["n_starts"] = 13
    if change == "tol":
        options["tol"] = float("nan")
    with pytest.raises(AnalysisError):
        overals(frame, sets, **options)


@pytest.mark.parametrize(
    "change",
    [
        "asymmetric",
        "negative",
        "missing",
        "diagonal",
        "disconnected",
        "all_equal",
        "ties",
        "zero",
        "labels",
        "too_large",
    ],
)
def test_nmds_data_semantics_guards(dissimilarities, change):
    d = dissimilarities.copy()
    options = {}
    if change == "asymmetric":
        d[0, 1] += 1
    if change == "negative":
        d[0, 1] = d[1, 0] = -1.0
    if change == "missing":
        d[0, 1] = d[1, 0] = np.nan
    if change == "diagonal":
        d[0, 0] = 1
    if change == "disconnected":
        d[:5, 5:] = d[5:, :5] = 0
        options["zero"] = "exclude"
    if change == "all_equal":
        d = np.ones((10, 10)) - np.eye(10)
    if change == "ties":
        options["ties"] = "tertiary"
    if change == "zero":
        options["zero"] = "missing"
    if change == "labels":
        d = pd.DataFrame(d, index=list("abcdefghij"))
    if change == "too_large":
        d = np.zeros((151, 151))
    with pytest.raises(AnalysisError):
        mds_nonmetric(d, **options)


def test_labeled_nmds_preserves_finite_json_labels(dissimilarities):
    names = [f"object-{i}" for i in range(10)]
    result = mds_nonmetric(pd.DataFrame(dissimilarities, index=names, columns=names), n_starts=2)
    assert list(result["coordinates"].index) == names
    assert result.attrs["row_labels"] == names
    assert result["pairs"].label_i.iloc[0] == names[0]


def test_projection_metadata_does_not_modify_original_calibration(frame):
    original = mca(frame, list(frame))
    before = copy.deepcopy(original.attrs)
    mca_project(original, frame.iloc[:2])
    assert original.attrs == before
