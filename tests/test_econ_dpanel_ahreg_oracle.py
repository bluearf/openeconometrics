"""Independent dense NumPy AH oracle; no OpenEcon assembly or covariance helpers.

The oracle assembles windows by (panel, integer period) dictionaries and uses
literal matrix IV and dense differencing-error H formulas. Production instead
uses contiguous row masks, shared Torch QR and sparse H cross-products.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import openecon as oe


def simulate(seed=13, groups=65, periods=9, gaps=False):
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(groups):
        effect, prev = rng.normal(), rng.normal()
        for period in range(-10, periods):
            x, w, error = rng.normal() + 0.2 * effect, rng.normal(), rng.normal()
            y = 0.52 * prev + 0.7 * x - 0.3 * w + effect + error
            if period >= (group % 2 if gaps else 0):
                rows.append({"id": group, "time": 2000 + period, "y": y, "x": x, "w": w})
            prev = y
    frame = pd.DataFrame(rows)
    if gaps:
        frame = frame.loc[~((frame.id % 4 == 0) & (frame.time == 2004))]
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


def oracle(frame, instrument="levels", predictors=("x",), covariance="cluster", missing=False):
    data = frame.copy()
    data["position"] = np.arange(len(data))
    if missing:
        data = data.dropna(subset=["id", "time", "y", *predictors])
    rows, target, regressors, instruments, labels, times = [], [], [], [], [], []
    length = 3 if instrument == "levels" else 4
    for group, block in data.groupby("id", sort=True):
        mapping = {int(row.time): row for row in block.itertuples(index=False)}
        for period, current in sorted(mapping.items()):
            if not all(period - lag in mapping for lag in range(length)):
                continue
            previous, lag2 = mapping[period - 1], mapping[period - 2]
            dx = [getattr(current, name) - getattr(previous, name) for name in predictors]
            excluded = lag2.y if instrument == "levels" else lag2.y - mapping[period - 3].y
            rows.append(current.position)
            target.append(current.y - previous.y)
            regressors.append([previous.y - lag2.y, *dx])
            instruments.append([*dx, excluded])
            labels.append(group)
            times.append(period)
    y, x, z = np.array(target), np.array(regressors), np.array(instruments)
    n, k = x.shape
    projected = z @ np.linalg.solve(z.T @ z, z.T @ x)
    bread = np.linalg.inv(projected.T @ x)
    beta = bread @ projected.T @ y
    residual = y - x @ beta
    groups = np.unique(labels)
    scores = projected * residual[:, None]
    if covariance == "nonrobust":
        h = np.eye(n) * 2
        for j in range(n):
            for i in range(j):
                if labels[i] == labels[j] and abs(times[i] - times[j]) == 1:
                    h[i, j] = h[j, i] = -1
        meat = residual @ residual / (2 * (n - k)) * projected.T @ h @ projected
        df = n - k
    else:
        grouped = np.array([scores[np.array(labels) == group].sum(axis=0) for group in groups])
        meat = grouped.T @ grouped * len(groups) / (len(groups) - 1) * (n - 1) / (n - k)
        df = len(groups) - 1
    covariance_matrix = bread @ meat @ bread.T
    fs_bread = np.linalg.inv(z.T @ z)
    fs_beta = fs_bread @ z.T @ x[:, 0]
    fs_residual = x[:, 0] - z @ fs_beta
    fs_scores = z * fs_residual[:, None]
    fs_groups = np.array([fs_scores[np.array(labels) == g].sum(axis=0) for g in groups])
    fs_cov = fs_bread @ (fs_groups.T @ fs_groups) @ fs_bread.T
    fs_cov *= len(groups) / (len(groups) - 1) * (n - 1) / (n - k)
    if predictors:
        exogenous = z[:, :-1]
        restricted = x[:, 0] - exogenous @ np.linalg.solve(exogenous.T @ exogenous,
                                                         exogenous.T @ x[:, 0])
    else:
        restricted = x[:, 0]
    return {
        "beta": beta, "covariance": covariance_matrix, "positions": rows,
        "observed": y, "residual": residual, "df": df, "groups": len(groups),
        "r_squared": 1 - (residual @ residual) / (y @ y),
        "fs_partial_r2": 1 - (fs_residual @ fs_residual) / (restricted @ restricted),
        "fs_coefficient": fs_beta[-1], "fs_f": fs_beta[-1] ** 2 / fs_cov[-1, -1],
    }


@pytest.mark.parametrize("instrument", ["levels", "differences"])
@pytest.mark.parametrize("covariance", ["nonrobust", "cluster", "robust"])
@pytest.mark.parametrize("predictors,gaps", [((), False), (("x", "w"), False), (("x",), True)])
def test_ah_matches_independent_dense_matrix_oracle(instrument, covariance, predictors, gaps):
    frame = simulate(gaps=gaps)
    expected = oracle(frame, instrument, predictors, covariance)
    result = oe.ahreg(data=frame, y="y", x=list(predictors), panel="id", time="time",
                      instrument=instrument, covariance=covariance)
    assert [c.term for c in result.coefficients] == ["L1.y", *predictors]
    assert_allclose([c.estimate for c in result.coefficients], expected["beta"], rtol=1e-10)
    assert_allclose(result.covariance_matrix, expected["covariance"], rtol=2e-9, atol=1e-12)
    assert result.sample_positions == expected["positions"]
    assert result.nobs == len(expected["positions"])
    assert result.metrics["n_groups"] == expected["groups"]
    assert result.metrics["r_squared"] == pytest.approx(expected["r_squared"])
    assert result.inference["df_inference"] == expected["df"]
    errors = np.sqrt(np.diag(expected["covariance"]))
    statistics = expected["beta"] / errors
    assert_allclose([c.p_value for c in result.coefficients],
                    2 * stats.t.sf(np.abs(statistics), expected["df"]), rtol=1e-9, atol=1e-13)
    critical = stats.t.isf(0.025, expected["df"])
    assert_allclose([c.ci_low for c in result.coefficients],
                    expected["beta"] - critical * errors, rtol=1e-9, atol=1e-12)
    joint = expected["beta"] @ np.linalg.solve(expected["covariance"], expected["beta"])
    assert result.tests["model"]["statistic"] == pytest.approx(joint / len(expected["beta"]),
                                                                rel=1e-9)
    assert result.tests["model"]["distribution"] == "F"
    assert result.tests["model"]["df2"] == expected["df"]
    first = result.extra["first_stage"][0]
    assert first["partial_r_squared"] == pytest.approx(expected["fs_partial_r2"])
    assert first["excluded_coefficient"] == pytest.approx(expected["fs_coefficient"])
    assert first["f_statistic"] == pytest.approx(expected["fs_f"], rel=1e-9)
    assert first["df2"] == expected["groups"] - 1
    assert first["p_value"] == pytest.approx(stats.f.sf(expected["fs_f"], 1,
                                                       expected["groups"] - 1), rel=1e-8)
    # Predictions are transformed outcomes aligned with original input positions.
    position = {row: i for i, row in enumerate(expected["positions"])}
    for prediction in result.predictions:
        i = position[prediction["row"]]
        assert prediction["observed"] == pytest.approx(expected["observed"][i])
        assert prediction["residual"] == pytest.approx(expected["residual"][i])


@pytest.mark.parametrize("instrument", ["levels", "differences"])
def test_missing_drop_keeps_time_gaps_and_matches_oracle(instrument):
    frame = simulate(groups=32, gaps=True)
    frame.loc[(frame.id % 3 == 0) & (frame.time == 2002), "x"] = np.nan
    expected = oracle(frame, instrument, missing=True)
    result = oe.ahreg(data=frame, y="y", x=["x"], panel="id", time="time",
                      instrument=instrument, missing="drop")
    assert result.sample_positions == expected["positions"]
    assert_allclose([c.estimate for c in result.coefficients], expected["beta"], rtol=1e-10)
    assert_allclose(result.covariance_matrix, expected["covariance"], rtol=1e-9, atol=1e-12)
    assert result.nobs_original == len(frame)
    assert result.dropped_rows == len(frame) - result.nobs
    assert result.extra["missing_rows"] == int(frame.x.isna().sum())


def test_ma1_classical_covariance_is_not_iid_row_2sls():
    frame = simulate(seed=21)
    result = oe.ahreg(data=frame, y="y", x=["x"], panel="id", time="time",
                      covariance="nonrobust")
    data = frame.sort_values(["id", "time"]).copy()
    data["dy"] = data.groupby("id").y.diff()
    data["dx"] = data.groupby("id").x.diff()
    data["dlag"] = data.groupby("id").dy.shift()
    data["z"] = data.groupby("id").y.shift(2)
    iid = oe.ivregress(data=data, y="dy", x=["dx"], endog=["dlag"], instruments=["z"],
                       intercept=False, small=True, missing="drop")
    assert_allclose([c.estimate for c in result.coefficients],
                    [iid.coefficients[1].estimate, iid.coefficients[0].estimate], rtol=1e-10)
    assert not np.allclose(result.covariance_matrix,
                           np.array(iid.covariance_matrix)[[1, 0]][:, [1, 0]], rtol=1e-3)
    assert result.inference["effective_covariance"] == "homoskedastic_level_ma1"
