"""Independent predictive references, dense HT projection and actual-sample CRE inference."""

import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def panel_data(unbalanced=False):
    rng = np.random.default_rng(148)
    g, t = 32, 8
    ids = np.repeat(np.arange(g), t)
    effects, z1, instrument = rng.normal(size=(3, g))
    x1 = instrument[ids] + rng.normal(size=g * t)
    x2 = 0.8 * effects[ids] + rng.normal(size=g * t)
    z2 = 0.9 * instrument + 0.6 * effects + rng.normal(size=g)
    y = (
        1
        + 0.7 * x1
        - 0.4 * x2
        + 0.3 * z1[ids]
        + 0.6 * z2[ids]
        + effects[ids]
        + rng.normal(size=g * t)
    )
    d = pd.DataFrame(
        dict(y=y, x1=x1, x2=x2, z1=z1[ids], z2=z2[ids], id=ids, t=np.tile(np.arange(t), g))
    )
    if unbalanced:
        d = d.loc[~((d.id % 3 == 0) & (d.t % 4 == 3))].reset_index(drop=True)
    return d


@pytest.mark.parametrize("components", [1, 2, 3, 4])
@pytest.mark.parametrize("standardize", [False, True])
def test_pls_independent_sklearn_heldout(components, standardize):
    pls = pytest.importorskip("sklearn.cross_decomposition").PLSRegression
    rng = np.random.default_rng(147)
    x = rng.normal(size=(100, 4)) * [1, 10, 0.1, 3] + [20, -4, 100, 0]
    y = x @ [0.5, 0.8, -2, 0.6] + rng.normal(size=100)
    data = pd.DataFrame(x, columns=list("abcd"))
    data["y"] = y
    native = oe.pls(
        data=data.iloc[:80],
        y="y",
        x=list("abcd"),
        selection="fixed",
        components=components,
        standardize=standardize,
    )
    oracle = pls(n_components=components, scale=standardize).fit(x[:80], y[:80])
    assert_allclose(
        oe.regularized_predict(native, data.iloc[80:]),
        oracle.predict(x[80:]).ravel(),
        atol=1e-9,
        rtol=1e-10,
    )
    assert native.coefficients == [] and native.covariance_matrix == []
    assert (
        native.inference["available"] is False
        and native.provenance["stata_parity_validated"] is False
    )
    restored = ResultBundle.model_validate_json(native.model_dump_json())
    assert_allclose(
        oe.regularized_predict(restored, data.iloc[80:]), oracle.predict(x[80:]).ravel(), atol=1e-9
    )
    assert oe.regularized_table(restored).attrs["components"] == components


def test_pls_full_component_ols_limit_and_independent_first_direction():
    rng = np.random.default_rng(7147)
    x = rng.normal(size=(50, 3))
    y = x @ [0.5, -0.8, 1.3] + rng.normal(size=50)
    d = pd.DataFrame(x, columns=list("abc"))
    d["y"] = y
    full = oe.pls(data=d, y="y", x=list("abc"), selection="fixed", components=3, standardize=False)
    b = np.linalg.lstsq(np.column_stack([np.ones(50), x]), y, rcond=None)[0]
    assert_allclose(
        oe.regularized_predict(full, d),
        np.column_stack([np.ones(50), x]) @ b,
        rtol=1e-10,
        atol=1e-10,
    )
    one = oe.pls(data=d, y="y", x=list("abc"), selection="fixed", components=1, standardize=False)
    xc, yc = x - x.mean(0), y - y.mean()
    w = xc.T @ yc
    score = xc @ w
    expected = y.mean() + score * (score @ yc) / (score @ score)
    assert_allclose(oe.regularized_predict(one, d), expected, atol=1e-11)


def test_pls_cv_independent_training_fold_preprocessing_and_missing_alignment():
    pls = pytest.importorskip("sklearn.cross_decomposition").PLSRegression
    rng = np.random.default_rng(9147)
    x = rng.normal(size=(71, 3)) * [1, 100, 0.001]
    y = x @ [0.5, 0.04, 200] + rng.normal(size=71)
    d = pd.DataFrame(x, columns=list("abc"))
    d["y"] = y
    d.index = np.zeros(71)
    d.iloc[3, 0] = np.nan
    original = d.copy(deep=True)
    result = oe.pls(data=d, y="y", x=list("abc"), missing="drop", component_path=[1, 2, 3], folds=5)
    clean = d.iloc[result.sample_positions]
    fold = np.asarray(result.extra["pls_state"]["fold_assignments"])
    errors = []
    for count in [1, 2, 3]:
        pred = np.zeros(len(clean))
        for f in range(5):
            train, test = fold != f, fold == f
            oracle = pls(n_components=count).fit(
                clean[list("abc")].to_numpy()[train], clean.y.to_numpy()[train]
            )
            pred[test] = oracle.predict(clean[list("abc")].to_numpy()[test]).ravel()
        errors.append(float(np.mean((clean.y - pred) ** 2)))
    assert_allclose(
        [row["cv_mse"] for row in result.extra["pls_state"]["component_path"]], errors, rtol=1e-10
    )
    assert result.extra["pls_state"]["components"] == int(np.argmin(errors)) + 1
    assert 3 not in result.sample_positions
    pd.testing.assert_frame_equal(d, original)


@pytest.mark.parametrize("unbalanced", [False, True])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_cre_full_covariance_and_mundlak_independent_re_reference(unbalanced, covariance):
    from test_econ_panel_oracle import cr1, check_inference

    d = panel_data(unbalanced)
    d.loc[3, "x1"] = np.nan
    d["cl"] = d.id // 2
    result = oe.xtreg(
        data=d,
        y="y",
        x=["x1", "x2", "z1"],
        panel="id",
        time="t",
        model="cre",
        covariance=covariance,
        cluster="cl" if covariance == "cluster" else None,
        missing="drop",
    )
    clean = d.dropna().rename(columns={"t": "year"}).reset_index(drop=True)
    for name in ["x1", "x2"]:
        clean["mean(" + name + ")"] = clean.groupby("id")[name].transform("mean")
    terms = ["Intercept", "x1", "x2", "z1", "mean(x1)", "mean(x2)"]
    # The between pilot contains one copy of each panel mean; the augmented
    # original/mean columns coincide between panels, so its rank is four, not six.
    a = pd.get_dummies(clean.id, dtype=float).to_numpy()
    sizes = a.sum(0)
    avg = np.diag(1 / sizes) @ a.T
    x = clean[terms[1:]].to_numpy()
    y = clean.y.to_numpy()
    xb = avg @ x
    yb = avg @ y
    wx = x - a @ xb
    wy = y - a @ yb
    bw = np.linalg.lstsq(wx[:, :2], wy, rcond=None)[0]
    se = float(np.linalg.norm(wy - wx[:, :2] @ bw) ** 2 / (len(clean) - len(sizes) - 2))
    between = np.column_stack([np.ones(len(sizes)), xb[:, :3]])
    bb = np.linalg.lstsq(between, yb, rcond=None)[0]
    su = max(
        0.0,
        float(np.linalg.norm(yb - between @ bb) ** 2 / (len(sizes) - 4) - se * np.mean(1 / sizes)),
    )
    theta = 1 - np.sqrt(se / (se + sizes * su))
    xs = np.column_stack([1 - a @ theta, x - (a @ theta)[:, None] * (a @ xb)])
    ys = y - (a @ theta) * (a @ yb)
    beta = np.linalg.lstsq(xs, ys, rcond=None)[0]
    resid = ys - xs @ beta
    ref = {
        "xs": xs,
        "resid": resid,
        "beta": beta,
        "v": np.linalg.inv(xs.T @ xs) * (resid @ resid) / (len(clean) - len(terms)),
    }
    v = (
        ref["v"]
        if covariance == "nonrobust"
        else cr1(
            ref["xs"],
            ref["resid"],
            None,
            pd.factorize(clean["cl" if covariance == "cluster" else "id"])[0],
            len(clean),
            len(terms),
        )[0]
    )
    check_inference(result, terms, ref["beta"], v, rtol=1e-7)
    b = ref["beta"][-2:]
    q = float(b @ np.linalg.solve(v[-2:, -2:], b))
    assert_allclose(result.tests["mundlak"]["statistic"], q, rtol=1e-8)
    assert_allclose(result.tests["mundlak"]["p_value"], stats.chi2.sf(q, 2), rtol=1e-8)
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    predicted = oe.cre_predict(restored, clean.rename(columns={"year": "t"}))
    assert_allclose(
        predicted, np.column_stack([np.ones(len(clean)), clean[terms[1:]]]) @ ref["beta"], rtol=1e-8
    )
    assert oe.mundlak_test(restored).iloc[0]["df"] == 2
    with pytest.raises(AnalysisError, match="cre_predict"):
        oe.predict(restored, clean.rename(columns={"year": "t"}))


def ht_reference(d):
    # Dense incidence/projection matrices intentionally differ from native compact QR.
    a = pd.get_dummies(d.id, dtype=float).to_numpy()
    g = a.shape[1]
    n = len(d)
    sizes = a.sum(0)
    average = np.diag(1 / sizes) @ a.T
    q = np.eye(n) - a @ average
    varying = d[["x1", "x2"]].to_numpy()
    invariant = np.column_stack([np.ones(n), d[["z1", "z2"]]])
    y = d.y.to_numpy()
    bw = np.linalg.lstsq(q @ varying, q @ y, rcond=None)[0]
    se = float(np.linalg.norm(q @ (y - varying @ bw)) ** 2 / (n - g))
    zb = average @ invariant
    h = np.column_stack([np.ones(g), average @ d.x1.to_numpy(), average @ d.z1.to_numpy()])
    ph = h @ np.linalg.inv(h.T @ h) @ h.T
    db = average @ (y - varying @ bw)
    bz = np.linalg.solve(zb.T @ ph @ zb, zb.T @ ph @ db)
    su = max(0.0, float(np.mean((db - zb @ bz) ** 2) - se * np.mean(1 / sizes)))
    theta = 1 - np.sqrt(se / (se + sizes * su))
    transform = np.eye(n) - a @ np.diag(theta) @ average
    raw = np.column_stack([np.ones(n), varying, d[["z1", "z2"]]])
    xt, yt = transform @ raw, transform @ y
    instruments = np.column_stack(
        [np.ones(n), q @ varying, a @ average @ d.x1.to_numpy(), d.z1.to_numpy()]
    )
    projection = instruments @ np.linalg.inv(instruments.T @ instruments) @ instruments.T
    bread = np.linalg.inv(xt.T @ projection @ xt)
    b = bread @ xt.T @ projection @ yt
    return b, bread, projection @ xt, yt - xt @ b, se, su


@pytest.mark.parametrize("unbalanced", [False, True])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("small", [False, True])
def test_ht_dense_iv_full_covariance_inference_and_saved_predictions(unbalanced, covariance, small):
    d = panel_data(unbalanced)
    d["cl"] = d.id // 2
    r = oe.htaylor_moment(
        data=d,
        y="y",
        panel="id",
        time="t",
        varying_exogenous=["x1"],
        varying_endogenous=["x2"],
        invariant_exogenous=["z1"],
        invariant_endogenous=["z2"],
        covariance=covariance,
        cluster="cl" if covariance == "cluster" else None,
        small=small,
    )
    b, bread, score, res, se, su = ht_reference(d)
    n, k = score.shape
    if covariance == "nonrobust":
        v = bread * (res @ res) / (n - k if small else n)
    else:
        groups = d["cl" if covariance == "cluster" else "id"].to_numpy()
        g = len(set(groups))
        sums = np.vstack(
            [(score[groups == i] * res[groups == i, None]).sum(0) for i in sorted(set(groups))]
        )
        factor = g / (g - 1) * (n - 1) / (n - k) if small else g / (g - 1)
        v = bread @ sums.T @ sums @ bread * factor
    from test_econ_panel_oracle import check_inference

    df = (
        min(n - k, len(set(d["cl" if covariance == "cluster" else "id"])) - 1)
        if covariance != "nonrobust"
        else n - k
    )
    check_inference(
        r, ["Intercept", "x1", "x2", "z1", "z2"], b, v, df=df if small else None, rtol=1e-7
    )
    assert_allclose([r.metrics["sigma_e"] ** 2, r.metrics["sigma_u"] ** 2], [se, su], rtol=1e-9)
    saved = ResultBundle.model_validate_json(r.model_dump_json())
    assert_allclose(
        oe.htaylor_predict(saved, d),
        np.column_stack([np.ones(n), d[["x1", "x2", "z1", "z2"]]]) @ b,
        rtol=1e-9,
    )


def test_panel_extension_error_domains():
    d = panel_data()
    with pytest.raises(AnalysisError) as e:
        oe.htaylor_moment(
            data=d, y="y", panel="id", varying_endogenous=["x2"], invariant_endogenous=["z2"]
        )
    assert e.value.code == "underidentified"
    with pytest.raises(AnalysisError) as e:
        oe.htaylor_moment(data=d, y="y", panel="id", varying_exogenous=["z1"])
    assert e.value.code == "invalid_roles"
    with pytest.raises(AnalysisError) as e:
        oe.htaylor_moment(
            data=d, y="y", panel="id", varying_exogenous=["x1"], invariant_exogenous=["x1"]
        )
    assert e.value.code == "invalid_roles"
    d["cl"] = d.t
    with pytest.raises(AnalysisError) as e:
        oe.htaylor_moment(
            data=d, y="y", panel="id", varying_exogenous=["x1"], covariance="cluster", cluster="cl"
        )
    assert e.value.code == "cluster_not_nested"
    cre = oe.xtreg(data=d, y="y", x=["x1", "x2"], panel="id", model="cre")
    query = d.iloc[:2].copy()
    query["id"] = 999
    with pytest.raises(AnalysisError) as e:
        oe.cre_predict(cre, query)
    assert e.value.code == "unknown_group"


@pytest.mark.parametrize("kind", ["pls", "cre", "ht", "reverse", "rolling_predict"])
def test_dataset_refusal_and_budget_before_work(kind, monkeypatch):
    d = panel_data()
    ds = Dataset.from_frame(d)
    monkeypatch.setattr(
        ds, "iter_batches", lambda *a, **k: pytest.fail("must not collect or read Dataset")
    )
    spec = ModelSpec(
        estimator="pls",
        outcome="y",
        predictors=["x1", "x2"],
        options={"selection": "fixed", "components": 1},
    )

    def run(data):
        if kind == "pls":
            return oe.pls(data=data, y="y", x=["x1", "x2"], selection="fixed", components=1)
        if kind == "cre":
            return oe.xtreg(data=data, y="y", x=["x1", "x2"], panel="id", model="cre")
        if kind == "ht":
            return oe.htaylor_moment(data=data, y="y", panel="id", varying_exogenous=["x1"])
        if kind == "reverse":
            return oe.rolling(
                ModelSpec(estimator="ols", outcome="y", predictors=["x1"]),
                data=data,
                window=20,
                reverse=True,
                max_fits=1000,
            )
        return oe.rolling_predict(
            spec, data=data, time="t", panel="id", calendar=True, window=4, max_fits=1000
        )

    with pytest.raises(AnalysisError) as e:
        run(ds)
    assert e.value.code == "streaming_unsupported"
    with use_workspace_budget(1), pytest.raises(AnalysisError) as e:
        run(pd.concat([d] * 500, ignore_index=True))
    assert e.value.code == "workspace_limit"


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("expanding", [False, True])
def test_calendar_panel_windows_match_independent_direct_fits(reverse, expanding):
    d = panel_data(True).sample(frac=1, random_state=184)
    d.index = np.zeros(len(d))
    spec = ModelSpec(
        estimator="xtreg",
        outcome="y",
        predictors=["x1", "x2"],
        panel="id",
        time="t",
        options={"model": "fe"},
    )
    if reverse and expanding:
        with pytest.raises(AnalysisError):
            oe.rolling(spec, data=d, window=4, reverse=True, expanding=True)
        return
    r = oe.rolling(spec, data=d, window=4, step=2, reverse=reverse, expanding=expanding)
    for origin in r.attrs["origins"]:
        chosen = d.iloc[origin["input_positions"]]
        native = oe.fit(spec, data=chosen)
        reference = oe.xtreg(
            data=d.loc[(d.t >= origin["start"]) & (d.t < origin["stop_exclusive"])],
            y="y",
            x=["x1", "x2"],
            panel="id",
            time="t",
        )
        assert_allclose(
            [c["estimate"] for c in origin["coefficients"]],
            [c.estimate for c in reference.coefficients],
            rtol=1e-10,
        )
        assert_allclose(origin["covariance_matrix"], reference.covariance_matrix, rtol=1e-10)
        assert (
            native.sample_positions
            == ResultBundle.model_validate(origin["result"]).sample_positions
        )
    json.dumps(r.attrs, allow_nan=False)
    assert r.attrs["window_units"] == "integer calendar periods"


def test_reverse_physical_missing_gaps_failed_windows_and_budget():
    rng = np.random.default_rng(184)
    d = pd.DataFrame({"y": rng.normal(size=18), "x": rng.normal(size=18), "t": np.arange(18)})
    spec = ModelSpec(estimator="ols", outcome="y", predictors=["x"], time="t")
    r = oe.rolling(spec, data=d, window=6, step=3, reverse_recursive=True)
    assert [o["start"] for o in r.attrs["origins"]] == [0, 3, 6, 9, 12]
    for o in r.attrs["origins"]:
        sm = pytest.importorskip("statsmodels.api")
        expected = sm.OLS(d.y.iloc[o["start"] :], sm.add_constant(d.x.iloc[o["start"] :])).fit()
        assert_allclose([c["estimate"] for c in o["coefficients"]], expected.params, rtol=1e-10)
    with pytest.raises(AnalysisError) as e:
        oe.rolling(spec, data=d, window=6, reverse_recursive=True, max_fits=1)
    assert e.value.code == "search_budget"
    with pytest.raises(AnalysisError) as e:
        oe.rolling(spec, data=d, window=6, reverse_recursive=True, forecast_steps=1)
    assert e.value.code == "unsupported_forecast"
    gaps = d.iloc[[0, 1, 4, 5, 8, 9, 13, 17]]
    failed = oe.rolling(spec, data=gaps, window=4, calendar=True, on_error="record")
    assert len(failed.attrs["origins"]) == 15
    assert any(o["status"] == "failed" for o in failed.attrs["origins"])


@pytest.mark.parametrize("time", [[False, True], [1 + 2j, 3 + 4j]])
def test_physical_time_rejects_non_chronological_numeric_types(time):
    data = pd.DataFrame({"y": [1.0, 2.0], "x": [3.0, 4.0], "t": time})
    spec = ModelSpec(estimator="ols", outcome="y", predictors=["x"], time="t")
    with pytest.raises(AnalysisError) as error:
        oe.rolling(spec, data=data, window=2, reverse_recursive=True)
    assert error.value.code == "invalid_time"


def test_rolling_prediction_tuning_isolated_from_future_and_persisted():
    rng = np.random.default_rng(1184)
    x = rng.normal(size=(45, 3))
    y = x @ [1, -2, 0.7] + rng.normal(size=45)
    d = pd.DataFrame(x, columns=list("abc"))
    d["y"] = y
    d["t"] = np.arange(45)
    spec = ModelSpec(
        estimator="pls",
        outcome="y",
        predictors=list("abc"),
        options={"component_path": [1, 2, 3], "folds": 4},
    )
    r = oe.rolling_predict(spec, data=d, time="t", window=20, step=10, horizon=5, max_fits=3)
    altered = d.copy()
    altered.loc[20:, "y"] += 1000
    altered.loc[20:, "a"] *= 100
    changed = oe.rolling_predict(
        spec, data=altered, time="t", window=20, step=10, horizon=5, max_fits=3
    )
    first = r.attrs["origins"][0]
    future = changed.attrs["origins"][0]
    assert first["result"]["extra"]["pls_state"] == future["result"]["extra"]["pls_state"]
    assert first["training_sample_positions"] == list(range(20))
    assert first["evaluation_sample_positions"] == list(range(20, 25))
    saved = ResultBundle.model_validate(first["result"])
    assert_allclose(
        r["predictions"].query("origin==19").predicted, oe.regularized_predict(saved, d.iloc[20:25])
    )
    assert_allclose(
        first["mse"], np.mean((d.y.iloc[20:25] - oe.regularized_predict(saved, d.iloc[20:25])) ** 2)
    )
    json.dumps(r.attrs, allow_nan=False)


def test_rolling_prediction_calendar_panel_missing_and_failed_evaluation():
    d = panel_data(True)
    spec = ModelSpec(
        estimator="ridge",
        outcome="y",
        predictors=["x1", "x2"],
        missing="drop",
        options={"selection": "fixed", "penalty": 1},
    )
    d.loc[(d.t == 4), "y"] = np.nan
    r = oe.rolling_predict(
        spec, data=d, time="t", panel="id", calendar=True, window=4, horizon=1, on_error="record"
    )
    assert r.attrs["origins"][0]["status"] == "failed"
    assert r.attrs["origins"][0]["error_code"] == "empty_sample"
    assert r.attrs["origins"][1]["status"] == "ok"
    assert all(d.iloc[pos].t <= 4 for pos in r.attrs["origins"][1]["training_sample_positions"])


def test_resident_group_keys_batch_codec_without_truncation(monkeypatch):
    from openecon.econometrics.postest import group_state
    from openecon.streaming_design import encode_cluster_labels

    d = pd.DataFrame({"id": np.arange(400000)})
    calls = []

    def bounded(series, *args, **kwargs):
        calls.append(len(series))
        assert len(series) <= 8192
        return encode_cluster_labels(series, *args, **kwargs)

    monkeypatch.setattr(group_state, "encode_cluster_labels", bounded)
    encoded = group_state.keys(d, ["id"])
    assert len(encoded) == len(d) and max(calls) == 8192
    assert encoded[0] == encode_cluster_labels(d.id.iloc[:1])[0]
    assert encoded[-1] == encode_cluster_labels(d.id.iloc[-1:])[0]
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        group_state.keys(d, ["id"])
    assert error.value.code == "workspace_limit"
