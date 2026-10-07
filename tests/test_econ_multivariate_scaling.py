"""Multidimensional scaling and correspondence analysis against NumPy / SciPy algebra."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import optimize as sopt
from scipy import stats
from scipy.spatial.distance import pdist, squareform

import openecon as oe
from openecon.analysis_contracts import AnalysisError

NAMES = ["a", "b", "c", "d"]


def _points(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 4)) * [3, 2, 1, 0.5] + [10, 0, -5, 2],
                        columns=NAMES)


def _torgerson(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(d)
    j = np.eye(n) - 1 / n
    values, vectors = np.linalg.eigh(-0.5 * j @ (d ** 2) @ j)
    return values[::-1], vectors[:, ::-1]


def _aligned(ours: np.ndarray, theirs: np.ndarray) -> float:
    """Largest coordinate difference after matching the sign of each dimension."""
    signs = np.sign((ours * theirs).sum(0))
    return np.abs(ours - theirs * signs).max()


# ---- classical scaling ------------------------------------------------------------


def test_classical_mds_from_data_and_from_distances():
    frame = _points()
    frame.loc[3, "b"] = np.nan
    x = frame.dropna().to_numpy()
    d = squareform(pdist(x))
    values, vectors = _torgerson(d)
    result = oe.mds(frame, NAMES, dimensions=2)
    assert result.attrs["n"] == 39 and result.attrs["n_missing"] == 1
    coordinates = result["coordinates"]
    assert 3 not in coordinates.index and list(coordinates.columns) == ["dim1", "dim2"]
    expected = vectors[:, :2] * np.sqrt(values[:2])
    assert _aligned(coordinates.to_numpy(), expected) < 1e-9
    ours = coordinates.to_numpy()
    assert (ours[np.abs(ours).argmax(0), np.arange(2)] > 0).all()
    table = result["eigenvalues"]
    np.testing.assert_allclose(table["eigenvalue"], values[:4], rtol=1e-9)
    positive = values[:4]
    np.testing.assert_allclose(table["abs_percent"], 100 * positive / positive.sum(), rtol=1e-9)
    np.testing.assert_allclose(table["squared_cumulative"],
                               100 * np.cumsum(positive ** 2) / (positive ** 2).sum(), rtol=1e-9)
    assert result.attrs["mardia1"] == pytest.approx(positive[:2].sum() / positive.sum())
    assert result.attrs["mardia2"] == pytest.approx(
        (positive[:2] ** 2).sum() / (positive ** 2).sum())
    assert result["fit"].loc["mardia1", "value"] == pytest.approx(result.attrs["mardia1"])

    labelled = pd.DataFrame(d, index=[f"c{i}" for i in range(39)],
                            columns=[f"c{i}" for i in range(39)])
    from_matrix = oe.mds(distances=labelled, dimensions=2)
    assert list(from_matrix["coordinates"].index) == list(labelled.columns)
    np.testing.assert_allclose(from_matrix["coordinates"].to_numpy(), ours, atol=1e-8)
    assert from_matrix.attrs["mardia1"] == pytest.approx(result.attrs["mardia1"], rel=1e-9)
    # In full dimension the configuration reproduces the distances exactly.
    full = oe.mds(frame, NAMES, dimensions=4)["coordinates"].to_numpy()
    np.testing.assert_allclose(squareform(pdist(full)), d, atol=1e-9)
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    standardized = oe.mds(frame, NAMES, dimensions=2, standardize=True)
    z_values, z_vectors = _torgerson(squareform(pdist(z)))
    assert _aligned(standardized["coordinates"].to_numpy(),
                    z_vectors[:, :2] * np.sqrt(z_values[:2])) < 1e-9


def test_classical_mds_of_non_euclidean_dissimilarities():
    rng = np.random.default_rng(1)
    d = squareform(pdist(rng.normal(size=(12, 3)), metric="cityblock")) ** 1.7
    values, vectors = _torgerson(d)
    assert values.min() < -1e-6                                 # not Euclidean
    result = oe.mds(distances=d.tolist(), dimensions=3)
    assert _aligned(result["coordinates"].to_numpy(), vectors[:, :3] * np.sqrt(values[:3])) < 1e-9
    assert list(result["coordinates"].index) == list(range(1, 13))
    assert result.attrs["mardia1"] == pytest.approx(values[:3].sum() / np.abs(values).sum())
    assert result.attrs["mardia2"] == pytest.approx(
        (values[:3] ** 2).sum() / (values ** 2).sum())
    assert len(result["eigenvalues"]) == 10
    np.testing.assert_allclose(result["eigenvalues"]["eigenvalue"], values[:10], atol=1e-9)


# ---- stress majorization ----------------------------------------------------------


def _smacof(delta: np.ndarray, x: np.ndarray, iterations: int) -> np.ndarray:
    n = len(delta)
    for _ in range(iterations):
        d = squareform(pdist(x))
        ratio = np.divide(delta, d, out=np.zeros_like(d), where=d > 0)
        b = -ratio
        b[np.diag_indices(n)] = ratio.sum(1)
        x = b @ x / n
    return x


def _raw_stress(x: np.ndarray, delta: np.ndarray) -> float:
    return ((squareform(pdist(x)) - delta) ** 2).sum() / 2


def test_smacof_minimizes_stress():
    frame = _points(n=25, seed=2)
    x = frame.to_numpy()
    delta = squareform(pdist(x))
    classical = oe.mds(frame, NAMES, dimensions=2)["coordinates"].to_numpy()
    result = oe.mds(frame, NAMES, dimensions=2, method="smacof", tolerance=1e-13,
                    max_iterations=5000)
    config = result["coordinates"].to_numpy()
    raw = _raw_stress(config, delta)
    assert result.attrs["converged"] and result.attrs["raw_stress"] == pytest.approx(raw)
    assert raw < _raw_stress(classical, delta)
    fitted = squareform(pdist(config))
    assert result.attrs["stress1"] == pytest.approx(
        np.sqrt(((fitted - delta) ** 2).sum() / (fitted ** 2).sum()))
    # Independent Guttman iterations from the same start reach the same stress.
    reference = _smacof(delta, classical, 3000)
    assert raw == pytest.approx(_raw_stress(reference, delta), rel=1e-8)
    # A general-purpose optimizer started at our solution cannot improve it.
    polished = sopt.minimize(lambda t: _raw_stress(t.reshape(-1, 2), delta), config.ravel(),
                             method="BFGS", options={"gtol": 1e-9})
    assert polished.fun == pytest.approx(raw, rel=1e-8)
    # Principal-axes orientation: centred, uncorrelated, ordered dimensions.
    np.testing.assert_allclose(config.mean(0), 0, atol=1e-9)
    cross = config.T @ config
    assert abs(cross[0, 1]) < 1e-8 * cross[0, 0] and cross[0, 0] >= cross[1, 1]

    # A configuration that fits in two dimensions is recovered with zero stress.
    flat = squareform(pdist(x[:, :2]))
    exact = oe.mds(distances=flat, dimensions=2, method="smacof")
    assert exact.attrs["stress1"] < 1e-6
    np.testing.assert_allclose(squareform(pdist(exact["coordinates"].to_numpy())), flat,
                               atol=1e-5)
    stopped = oe.mds(frame, NAMES, dimensions=2, method="smacof", max_iterations=2,
                     tolerance=0.0)
    assert not stopped.attrs["converged"] and "max_iterations" in stopped.attrs["notes"][0]
    assert "stress1" in str(stopped)


def test_mds_error_codes():
    frame = _points(n=10)
    d = squareform(pdist(frame.to_numpy()))

    def code(*args, **kwargs) -> str:
        with pytest.raises(AnalysisError) as error:
            oe.mds(*args, **kwargs)
        return error.value.code

    assert code() == "invalid_spec"
    assert code(frame, NAMES, distances=d) == "invalid_spec"
    assert code(frame, NAMES, method="isomap") == "invalid_option"
    assert code(frame, NAMES, dimensions=0) == "invalid_option"
    assert code(frame, NAMES, dimensions=5) == "too_many_dimensions"
    assert code(frame, NAMES, max_n=5) == "too_many_observations"
    assert code(distances=d, max_n=5) == "too_many_observations"
    assert code(distances=d[:, :3]) == "invalid_distances"
    assert code(distances=-d) == "invalid_distances"
    assert code(distances=d + np.triu(np.ones_like(d), 1)) == "invalid_distances"
    assert code(distances=[["a", "b"], ["c", "d"]]) == "invalid_distances"
    assert code(frame.assign(s="t"), ["a", "s"]) == "non_numeric_column"


# ---- correspondence analysis ------------------------------------------------------


def _table() -> pd.DataFrame:
    counts = np.array([[68, 20, 15, 5], [94, 26, 17, 10], [10, 40, 22, 8], [30, 5, 9, 14],
                       [12, 9, 33, 6]])
    rows = ["dark", "fair", "red", "black", "grey"]
    columns = ["brown", "blue", "green", "hazel"]
    return pd.DataFrame([{"hair": r, "eye": col, "count": int(counts[i, j])}
                         for i, r in enumerate(rows) for j, col in enumerate(columns)])


def test_ca_matches_numpy_svd():
    data = _table()
    result = oe.ca(data, "hair", "eye", weights="count", dimensions=2)
    table = data.pivot(index="hair", columns="eye", values="count")       # sorted labels
    counts = table.to_numpy(dtype=float)
    n = counts.sum()
    p = counts / n
    r, cm = p.sum(1), p.sum(0)
    s = (p - np.outer(r, cm)) / np.sqrt(np.outer(r, cm))
    u, sv, vt = np.linalg.svd(s, full_matrices=False)
    u, sv, v = u[:, :3], sv[:3], vt.T[:, :3]
    inertia = result["inertia"]
    np.testing.assert_allclose(inertia["singular_value"], sv, rtol=1e-10)
    np.testing.assert_allclose(inertia["principal_inertia"], sv ** 2, rtol=1e-10)
    np.testing.assert_allclose(inertia["percent"], 100 * sv ** 2 / (sv ** 2).sum(), rtol=1e-10)
    chi2, p_value, df, _ = stats.chi2_contingency(counts, correction=False)
    assert result.attrs["chi2"] == pytest.approx(chi2) and result.attrs["df"] == df
    assert result.attrs["p_value"] == pytest.approx(p_value, rel=1e-7)
    assert result.attrs["total_inertia"] == pytest.approx(chi2 / n)
    assert inertia["chi2"].sum() == pytest.approx(chi2)
    np.testing.assert_allclose(result["table"].to_numpy(), counts)
    assert list(result["table"].index) == list(table.index)

    for name, vectors, mass, labels in (("rows", u, r, table.index),
                                        ("columns", v, cm, table.columns)):
        ours = result[name]
        assert list(ours.index) == list(labels)
        standard = vectors / np.sqrt(mass)[:, None]
        principal = standard * sv
        signs = np.sign((ours[["dim1_standard", "dim2_standard"]].to_numpy()
                         * standard[:, :2]).sum(0))
        np.testing.assert_allclose(ours["mass"], mass, rtol=1e-12)
        np.testing.assert_allclose(ours[["dim1", "dim2"]].to_numpy(), principal[:, :2] * signs,
                                   atol=1e-10)
        np.testing.assert_allclose(ours[["dim1_standard", "dim2_standard"]].to_numpy(),
                                   standard[:, :2] * signs, atol=1e-10)
        distance = (principal ** 2).sum(1)
        np.testing.assert_allclose(ours["quality"], (principal[:, :2] ** 2).sum(1) / distance,
                                   rtol=1e-9)
        np.testing.assert_allclose(ours["inertia"], mass * distance / (sv ** 2).sum(), rtol=1e-9)
        np.testing.assert_allclose(ours["dim2_sqcorr"], principal[:, 1] ** 2 / distance,
                                   rtol=1e-9)
        np.testing.assert_allclose(ours["dim1_contribution"],
                                   mass * principal[:, 0] ** 2 / sv[0] ** 2, rtol=1e-9)
        assert ours["dim1_contribution"].sum() == pytest.approx(1.0)
        assert ours["inertia"].sum() == pytest.approx(1.0)
        # Weighted standard coordinates are centred and have unit variance.
        assert (mass * ours["dim1_standard"]).sum() == pytest.approx(0.0, abs=1e-10)
        assert (mass * ours["dim1_standard"] ** 2).sum() == pytest.approx(1.0)
    # Chi-square distances between row profiles are Euclidean in full principal space.
    full = oe.ca(data, "hair", "eye", weights="count", dimensions=3)["rows"]
    coords = full[["dim1", "dim2", "dim3"]].to_numpy()
    profiles = p / r[:, None]
    chi_distance = np.sqrt((((profiles[0] - profiles[1]) ** 2) / cm).sum())
    assert np.linalg.norm(coords[0] - coords[1]) == pytest.approx(chi_distance, rel=1e-9)
    np.testing.assert_allclose(full["quality"], 1.0)


def test_ca_from_individual_records_notes_and_errors():
    data = _table()
    records = data.loc[data.index.repeat(data["count"]), ["hair", "eye"]].reset_index(drop=True)
    records.loc[0, "eye"] = None
    weighted = oe.ca(data, "hair", "eye", weights="count")
    unweighted = oe.ca(records, "hair", "eye")
    assert unweighted.attrs["n"] == weighted.attrs["n"] - 1 and unweighted.attrs["n_missing"] == 1
    assert unweighted["inertia"]["singular_value"].iloc[0] == pytest.approx(
        weighted["inertia"]["singular_value"].iloc[0], rel=0.05)

    two = data[data["hair"].isin(["dark", "red"])]
    capped = oe.ca(two, "hair", "eye", weights="count", dimensions=2)
    assert capped.attrs["dimensions"] == 1 and "1 reported" in capped.attrs["notes"][0]
    assert "dim2" not in capped["rows"].columns
    zero = data.copy()
    zero.loc[zero["eye"] == "hazel", "count"] = 0
    dropped = oe.ca(zero, "hair", "eye", weights="count")
    assert "hazel" not in dropped["columns"].index and "removed" in dropped.attrs["notes"][0]
    text = str(weighted)
    assert "[inertia]" in text and weighted.to_latex().count(r"\begin{tabular}") == 4

    def code(*args, **kwargs) -> str:
        with pytest.raises(AnalysisError) as error:
            oe.ca(*args, **kwargs)
        return error.value.code

    assert code(data, "hair", "hair") == "invalid_spec"
    assert code(data, "hair", "nope") == "missing_columns"
    assert code(data, "hair", "eye", weights="hair") == "invalid_spec"
    assert code(data.assign(w="x"), "hair", "eye", weights="w") == "non_numeric_column"
    assert code(data.assign(count=-1), "hair", "eye", weights="count") == "invalid_weights"
    assert code(data[data["hair"] == "dark"], "hair", "eye", weights="count") == "degenerate_table"
    assert code(data.assign(count=1), "hair", "eye", weights="count") == "degenerate_table"
    assert code(data, "hair", "eye", dimensions=0) == "invalid_option"
    assert code(records, "hair", "eye", missing="raise") == "missing_values"
