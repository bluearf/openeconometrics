"""Independent SciPy bases and dense NumPy Gaussian/local/kernel references.

Development oracles only; production estimators use resident float64 Torch.
"""

import itertools
import json

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline, CubicSpline
from scipy.stats import f, t, norm

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget

NAMES = [
    "bspline_regress",
    "rcs_regress",
    "gam_gaussian",
    "loess",
    "fp_regress",
    "mfp_regress",
    "mars",
    "npreg_mixed",
]


def data(n=90):
    rng = np.random.default_rng(993)
    x = np.linspace(0.25, 4, n)
    z = rng.uniform(-1, 1, n)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "u": ["a", "b", "c"] * (n // 3) + ["a"] * (n % 3),
            "o": ["low", "middle", "high"] * (n // 3) + ["low"] * (n % 3),
            "y": 1 + np.sin(x) + 0.4 * z + rng.normal(0, 0.12, n),
        }
    )


def opts(name):
    return (
        {
            "variable_types": {"x": "c", "u": "u", "o": "o"},
            "categories": {"u": ["a", "b", "c"], "o": ["low", "middle", "high"]},
            "bandwidth": [0.5, 0.3, 0.4],
        }
        if name == "npreg_mixed"
        else {"max_terms": 7}
        if name == "mars"
        else {}
    )


def model(name, d=None, **options):
    return getattr(oe, name)(
        data=data() if d is None else d,
        y="y",
        x=["x", "u", "o"] if name == "npreg_mixed" else ["x"],
        **{**opts(name), **options},
    )


def state(model):
    return model.extra["smoothing_state"]


def ols_oracle(a, y):
    b = np.linalg.lstsq(a, y, rcond=None)[0]
    residual = y - a @ b
    df = len(y) - a.shape[1]
    cov = (residual @ residual) / df * np.linalg.inv(a.T @ a)
    return b, cov, df


def full_inference(m, a, y):
    b, cov, df = ols_oracle(a, y)
    np.testing.assert_allclose([c.estimate for c in m.coefficients], b, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(m.covariance_matrix, cov, rtol=5e-7, atol=1e-9)
    for j, c in enumerate(m.coefficients):
        se = np.sqrt(cov[j, j])
        assert c.std_error == pytest.approx(se, rel=3e-7)
        assert c.p_value == pytest.approx(2 * t.sf(abs(b[j] / se), df), rel=2e-6, abs=1e-12)
        assert c.ci_low == pytest.approx(b[j] - t.ppf(0.975, df) * se, rel=2e-6, abs=1e-9)
    return b, cov


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("knots", [[], [1.0], [1.0, 2.0, 3.0]])
def test_bspline_scipy_basis_and_complete_inference(degree, knots):
    d = data()
    m = model("bspline_regress", d, degree=degree, knots={"x": knots}, boundary={"x": [0.25, 4.0]})
    tx = [0.25] * (degree + 1) + knots + [4.0] * (degree + 1)
    basis = BSpline.design_matrix(d.x, tx, degree).toarray()
    a = np.column_stack([np.ones(len(d)), basis[:, 1:]])
    b, cov = full_inference(m, a, d.y.to_numpy())
    q = pd.DataFrame({"x": [0.25, 0.8, 1.0, 2.2, 3.9, 4.0]})
    pred = oe.smoothing_predict(m, data=q, interval=True)
    aq = np.column_stack([np.ones(len(q)), BSpline.design_matrix(q.x, tx, degree).toarray()[:, 1:]])
    np.testing.assert_allclose(pred["mean"], aq @ b, atol=1e-10)
    np.testing.assert_allclose(
        pred.std_error, np.sqrt(np.einsum("ij,jk,ik->i", aq, cov, aq)), atol=1e-9
    )
    np.testing.assert_allclose(basis.sum(1), 1, atol=1e-14)


@pytest.mark.parametrize("knots", [[0.25, 2.0, 4.0], [0.5, 1.0, 2.0, 3.0, 3.8]])
def test_rcs_natural_spline_space_and_linear_tails(knots):
    d = data()
    m = model("rcs_regress", d, knots={"x": knots})
    # A different interpolating natural-spline basis spans the same space.
    cs = CubicSpline(knots, np.eye(len(knots)), bc_type="natural", axis=0)

    def natural(q):
        a = cs(q)
        for i, v in enumerate(q):
            if v < knots[0]:
                a[i] = cs(knots[0]) + (v - knots[0]) * cs(knots[0], 1)
            if v > knots[-1]:
                a[i] = cs(knots[-1]) + (v - knots[-1]) * cs(knots[-1], 1)
        return a

    b, cov, _ = ols_oracle(natural(d.x.to_numpy()), d.y.to_numpy())
    q = np.array([-3.0, -2.0, -1.0, 0.5, 2.0, 4.0, 5.0, 6.0, 7.0])
    prediction = oe.smoothing_predict(m, data=pd.DataFrame({"x": q}), interval=True)
    aq = natural(q)
    np.testing.assert_allclose(prediction["mean"], aq @ b, atol=2e-10)
    np.testing.assert_allclose(
        prediction.std_error, np.sqrt(np.einsum("ij,jk,ik->i", aq, cov, aq)), atol=2e-9
    )
    assert np.diff(prediction["mean"].to_numpy()[:3], n=2)[0] == pytest.approx(0, abs=1e-10)
    assert np.diff(prediction["mean"].to_numpy()[-3:], n=2)[0] == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("penalty", [0.0, 0.01, 3.0, 50.0])
def test_gam_full_frequentist_covariance_and_centering(penalty):
    d = data()
    m = oe.gam_gaussian(data=d, y="y", x=["x", "z"], penalty=penalty)
    s = state(m)
    blocks = [np.ones((len(d), 1))]
    for transform in s["transforms"]:
        degree = transform["degree"]
        lo, hi = transform["boundary"]
        tx = [lo] * (degree + 1) + transform["knots"] + [hi] * (degree + 1)
        block = BSpline.design_matrix(d[transform["column"]], tx, degree).toarray()[:, 1:]
        np.testing.assert_allclose(transform["center"], block.mean(0), atol=1e-15)
        blocks.append(block - block.mean(0))
    a = np.column_stack(blocks)
    p = np.zeros((a.shape[1], a.shape[1]))
    offset = 1
    for block in blocks[1:]:
        count = block.shape[1]
        delta = np.diff(np.eye(count + 1), n=2, axis=0)[:, 1:]
        p[offset : offset + count, offset : offset + count] = delta.T @ delta
        offset += count
    gram = a.T @ a
    inv = np.linalg.inv(gram + penalty * p)
    b = inv @ a.T @ d.y
    edf = np.trace(inv @ gram)
    rdf = len(d) - 2 * edf + np.trace(inv @ gram @ inv @ gram)
    rss = np.sum((d.y - a @ b) ** 2)
    cov = rss / rdf * inv @ gram @ inv.T
    np.testing.assert_allclose(s["coefficients"], b, atol=2e-10)
    np.testing.assert_allclose(s["covariance"], cov, atol=2e-10)
    assert m.metrics["edf"] == pytest.approx(edf, abs=1e-9)
    assert m.metrics["residual_trace_df"] == pytest.approx(rdf, abs=1e-9)
    assert not m.coefficients and m.inference["distribution"] == "none"
    pred = oe.smoothing_predict(m, data=d, interval=True)
    np.testing.assert_allclose(
        pred.std_error, np.sqrt(np.einsum("ij,jk,ik->i", a, cov, a)), atol=2e-10
    )
    np.testing.assert_allclose(
        pred.ci_high, pred["mean"] + norm.ppf(0.975) * pred.std_error, atol=1e-9
    )


def test_gam_gcv_complete_training_search_and_replay_no_refit():
    d = data()
    m = model("gam_gaussian", d, selection="gcv", penalty_path=[0.0, 0.1, 1.0, 10.0])
    rows = state(m)["candidates"]
    assert len(rows) == 4
    assert state(m)["selected"] == min(range(4), key=lambda i: rows[i]["gcv"])
    for row in rows:
        assert row["gcv"] == pytest.approx(len(d) * row["rss"] / (len(d) - row["edf"]) ** 2)
    before = m.model_dump_json()
    q = d.iloc[[20, 40, 60]].copy()
    q.y = 1e30
    oe.smoothing_predict(m, data=q)
    assert m.model_dump_json() == before


POWER_LIST = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "powers",
    [[p] for p in POWER_LIST]
    + [list(p) for p in itertools.combinations_with_replacement(POWER_LIST, 2)],
)
def test_fp_all_44_power_models_and_full_covariance(powers):
    d = data()
    m = oe.fp_regress(data=d, y="y", x=["x", "z"], powers=powers, scale=2.0)
    z = d.x.to_numpy() / 2
    transform = [np.log(z) if p == 0 else z**p for p in powers]
    if len(powers) == 2 and powers[0] == powers[1]:
        transform[1] *= np.log(z)
    a = np.column_stack([np.ones(len(d)), *transform, d.z])
    full_inference(m, a, d.y.to_numpy())
    np.testing.assert_allclose(
        oe.smoothing_predict(m, data=d)["mean"], a @ np.asarray(state(m)["coefficients"]), atol=1e-9
    )


@pytest.mark.parametrize("signal", ["null", "linear", "log", "curved"])
def test_mfp_45_dense_refits_and_closed_f_statistics(signal):
    d = data(180)
    rng = np.random.default_rng(42)
    z = d.x.to_numpy()
    d.y = {
        "null": np.zeros(len(d)),
        "linear": 2 * z,
        "log": 2 * np.log(z),
        "curved": z**-1 * np.log(z) * 2,
    }[signal] + rng.normal(0, 0.3, len(d))
    m = model("mfp_regress", d)
    s = state(m)
    assert len(s["candidates"]) == 45
    for c in s["candidates"]:
        terms = [np.log(z) if p == 0 else z**p for p in c["powers"]]
        if len(terms) == 2 and c["powers"][0] == c["powers"][1]:
            terms[1] *= np.log(z)
        a = np.column_stack([np.ones(len(d)), *terms])
        b = np.linalg.lstsq(a, d.y, rcond=None)[0]
        assert c["rss"] == pytest.approx(np.sum((d.y - a @ b) ** 2), rel=1e-9, abs=1e-10)
    for test in s["closed_tests"]:
        full = s["candidates"][test["full_candidate"]]
        reduced = s["candidates"][test["reduced_candidate"]]
        stat = max(0.0, (reduced["rss"] - full["rss"]) / test["df_num"]) / (
            full["rss"] / test["df_den"]
        )
        assert test["statistic"] == pytest.approx(stat)
        assert test["p_value"] == pytest.approx(
            f.sf(stat, test["df_num"], test["df_den"]), rel=1e-8, abs=1e-13
        )
    assert (
        s["candidates"][s["selected"]]["powers"]
        == {"null": [], "linear": [1.0], "log": [0.0], "curved": [-1.0, -1.0]}[signal]
    )


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("span", [0.35, 0.75, 1.0])
def test_loess_independent_tricube_local_lstsq(degree, span):
    d = data()
    m = model("loess", d, degree=degree, span=span)
    q = np.array([0.25, 0.6, 1.7, 2.0, 3.8, 4.0])
    expected = []
    for v in q:
        dist = np.abs(d.x.to_numpy() - v)
        radius = np.partition(dist, int(np.ceil(span * len(d))) - 1)[
            int(np.ceil(span * len(d))) - 1
        ]
        w = (1 - np.minimum(dist / radius, 1) ** 3) ** 3
        a = np.column_stack([((d.x.to_numpy() - v) / radius) ** j for j in range(degree + 1)])
        expected.append(
            np.linalg.lstsq(a * np.sqrt(w[:, None]), d.y.to_numpy() * np.sqrt(w), rcond=None)[0][0]
        )
    np.testing.assert_allclose(
        oe.smoothing_predict(m, data=pd.DataFrame({"x": q}))["mean"], expected, atol=1e-10
    )


def test_loess_linear_reproduction_and_complete_loo_records():
    d = data()
    d.y = 1 + 2 * d.x
    m = model("loess", d, selection="loo", span_path=[0.01, 0.3, 0.7])
    rows = state(m)["candidates"]
    assert rows[0]["loo_mse"] is None and rows[0]["failures"][0]["code"] == "local_rank_deficient"
    assert all(r["loo_mse"] < 1e-20 for r in rows[1:])
    np.testing.assert_allclose(oe.smoothing_predict(m, data=d)["mean"], d.y, atol=1e-10)


def independent_hinges(d, terms):
    a = []
    for factors in terms:
        value = np.ones(len(d))
        for factor in factors:
            value *= np.maximum(
                factor["sign"] * (d.iloc[:, factor["column_index"]].to_numpy() - factor["knot"]), 0
            )
        a.append(value)
    return np.column_stack(a)


@pytest.mark.parametrize("degree", [1, 2])
def test_mars_dense_selected_and_backward_fits_heldout(degree):
    d = data(150)
    rng = np.random.default_rng(883)
    d.y = (
        2
        + 3 * np.maximum(d.x - 2, 0)
        + (0.8 * np.maximum(d.x - 1, 0) * np.maximum(d.z, 0) if degree == 2 else 0)
        + rng.normal(0, 0.02, len(d))
    )
    m = oe.mars(data=d, y="y", x=["x", "z"], max_terms=9, max_degree=degree, max_candidates=12)
    s = state(m)
    for row in s["backward"]:
        a = independent_hinges(d[["x", "z"]], [s["forward_terms"][j] for j in row["retained"]])
        b = np.linalg.lstsq(a, d.y, rcond=None)[0]
        assert row["rss"] == pytest.approx(np.sum((d.y - a @ b) ** 2), abs=1e-9)
    a = independent_hinges(d[["x", "z"]], s["hinges"])
    np.testing.assert_allclose(
        oe.smoothing_predict(m, data=d)["mean"],
        a @ np.linalg.lstsq(a, d.y, rcond=None)[0],
        atol=1e-9,
    )
    assert np.sqrt(np.mean((oe.smoothing_predict(m, data=d)["mean"] - d.y) ** 2)) < 0.1
    assert not m.coefficients and s["tie_policy"]


@pytest.mark.parametrize("bw", [[0.3, 0.0, 0.0], [0.5, 0.3, 0.4], [1.0, 2 / 3, 0.95]])
def test_mixed_independent_gaussian_aa_wr_product_weights(bw):
    d = data()
    m = model("npreg_mixed", d, bandwidth=bw, min_effective=1.0)
    q = d.iloc[[10, 40, 70]].copy()
    expected = []
    for _, row in q.iterrows():
        w = np.exp(-0.5 * ((d.x.to_numpy() - row.x) / bw[0]) ** 2)
        w *= np.where(d.u.to_numpy() == row.u, 1 - bw[1], bw[1] / 2)
        levels = {v: j for j, v in enumerate(["low", "middle", "high"])}
        distance = np.abs(np.array([levels[v] for v in d.o]) - levels[row.o])
        w *= np.where(distance == 0, 1 - bw[2], 0.5 * (1 - bw[2]) * bw[2] ** distance)
        expected.append(w @ d.y / w.sum())
    np.testing.assert_allclose(oe.smoothing_predict(m, data=q)["mean"], expected, atol=1e-12)


def test_mixed_loo_records_and_declared_order_rejection():
    m = model("npreg_mixed", selection="loo", bandwidth_path=[[0.3, 0.2, 0.3], [0.7, 0.6, 0.5]])
    assert len(state(m)["candidates"]) == 2
    assert state(m)["selected"] == min(range(2), key=lambda j: state(m)["candidates"][j]["loo_mse"])
    q = data().iloc[[20]].copy()
    q.o = "unknown"
    with pytest.raises(AnalysisError, match="universe"):
        oe.smoothing_predict(m, data=q)


@pytest.mark.parametrize("name", NAMES)
def test_roundtrip_full_state_replay_latex_missing_alignment(name):
    d = data()
    d.loc[5, "x"] = np.nan
    d.loc[13, "y"] = np.nan
    m = model(name, d, missing="drop")
    assert m.dropped_rows == 2 and 5 not in m.sample_positions and 13 not in m.sample_positions
    saved = oe.ResultBundle.model_validate_json(m.model_dump_json())
    assert saved.extra == m.extra and saved.covariance_matrix == m.covariance_matrix
    assert "\\begin" in m.to_latex()
    q = data().iloc[[20, 30, 40]].copy()
    q.iloc[1, q.columns.get_loc("x")] = np.nan
    a = oe.smoothing_predict(m, data=q, missing="drop")
    b = oe.smoothing_predict(saved, data=q, missing="drop")
    assert a.to_dict() == b.to_dict() and a.row.tolist() == [0, 2]
    before = m.model_dump_json()
    q.y = 1e50
    oe.smoothing_predict(m, data=q, missing="drop")
    assert m.model_dump_json() == before
    saved.extra["smoothing_state"]["method"] = "altered"
    with pytest.raises(AnalysisError, match="digest/spec"):
        oe.smoothing_predict(saved, data=data())


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "option",
    [
        {"max_work": 1},
        {"device": "cuda"},
        {"weights": "w"},
        {"covariance": "HC3"},
        {"unexpected": True},
    ],
)
def test_strict_unsupported_contract_and_work_guards(name, option):
    with pytest.raises(AnalysisError):
        model(name, **option)


@pytest.mark.parametrize("name", NAMES)
def test_dataset_fit_scope_and_replay_budget(name, tmp_path):
    d = data()
    path = tmp_path / "rows.csv"
    d.to_csv(path, index=False)
    if name in {"bspline_regress", "rcs_regress", "fp_regress", "mfp_regress"}:
        streamed = model(name, oe.scan(path))
        assert streamed.nobs == len(d)
        assert streamed.provenance["streaming"]["dense_observation_matrix"] is False
    else:
        with pytest.raises(AnalysisError, match="Dataset"):
            model(name, oe.scan(path))
    m = model(name)
    with pytest.raises(AnalysisError, match="work"):
        oe.smoothing_predict(m, data=d, max_work=1)
    big = pd.concat([d] * 1500, ignore_index=True)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        oe.smoothing_predict(m, data=big, max_work=10**15)


@pytest.mark.parametrize("name", ["bspline_regress", "gam_gaussian", "loess", "npreg_mixed"])
def test_explicit_outside_support(name):
    m = model(name)
    d = data().iloc[[0]].copy()
    d.x = -100.0
    with pytest.raises(AnalysisError, match="boundar|range"):
        oe.smoothing_predict(m, data=d)


@pytest.mark.parametrize("name", ["loess", "npreg_mixed", "mars"])
def test_no_fabricated_adaptive_inference(name):
    m = model(name)
    assert not m.coefficients and not m.covariance_matrix
    with pytest.raises(AnalysisError, match="inference"):
        oe.smoothing_predict(m, data=data(), interval=True)


@pytest.mark.parametrize(
    "name,options",
    [
        ("bspline_regress", {"knots": {"x": [2, 1]}}),
        ("rcs_regress", {"knots": {"x": [1, 2]}}),
        ("fp_regress", {"powers": [4.0]}),
        ("mars", {"max_terms": 6}),
        ("loess", {"span_path": [0.5]}),
        ("gam_gaussian", {"penalty_path": [1.0]}),
    ],
)
def test_malformed_or_ignored_options_raise(name, options):
    with pytest.raises(AnalysisError):
        model(name, **options)


def test_fp_nonpositive_is_never_automatically_shifted():
    d = data()
    d.loc[0, "x"] = 0
    for name in ["fp_regress", "mfp_regress"]:
        with pytest.raises(AnalysisError, match="positive"):
            model(name, d)


def test_category_code_dtype_and_strict_typed_universes():
    d = data()
    d.u = [True, 1, "a"] * 30
    m = model("npreg_mixed", d, categories={"u": [True, 1, "a"], "o": ["low", "middle", "high"]})
    assert state(m)["train_x"][0][1] != state(m)["train_x"][1][1]
    assert json.loads(m.model_dump_json())["extra"]["smoothing_state"]["categories"]["u"] == [
        True,
        1,
        "a",
    ]
