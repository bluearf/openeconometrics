"""Independent numerical balance oracles and complete sample/state contracts."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def fixture():
    rng = np.random.default_rng(512)
    x = rng.normal(size=(90, 2))
    t = np.r_[np.zeros(60), np.ones(30)]
    x[t == 1] += [0.2, -0.1]
    return pd.DataFrame(
        dict(t=t, x=x[:, 0], z=x[:, 1], base=rng.uniform(0.2, 2, 90)),
        index=[f"unit-{i}" for i in range(90)],
    )


def test_entropy_independent_dual_and_exact_moments():
    data = fixture()
    result = oe.ebalance(data, "t", ["x", "z"], base_weights="base", tolerance=1e-10)
    x, t, base = data[["x", "z"]].values, data.t.values.astype(bool), data.base.values
    target = np.average(x[t], weights=base[t], axis=0)
    z = x[~t] - target
    q = base[~t] / base[~t].sum()

    def objective(lam):
        logits = np.log(q) + z @ lam
        w = np.exp(logits - logsumexp(logits))
        return logsumexp(logits), w @ z

    fit = minimize(objective, np.zeros(2), jac=True, method="BFGS", options={"gtol": 1e-10})
    expected = np.exp(np.log(q) + z @ fit.x - logsumexp(np.log(q) + z @ fit.x))
    actual = np.array(result.attrs["state"]["control_probability_weights"])
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(actual @ x[~t], target, atol=1e-10)
    assert np.all(actual > 0)
    assert result.attrs["state"]["covariance"] is None
    assert result.attrs["unit_labels"] == list(data.index)
    assert result.attrs["state"]["convergence"]["converged"]


def test_entropy_base_scale_and_row_permutation_equivariance():
    data = fixture()
    a = oe.ebalance(data, "t", ["x", "z"], base_weights="base")
    b = oe.ebalance(data.assign(base=data.base * 1e80), "t", ["x", "z"], base_weights="base")
    np.testing.assert_allclose(a["weights"].weight, b["weights"].weight, rtol=1e-12)
    perm = np.random.default_rng(5).permutation(len(data))
    r = oe.ebalance(data.iloc[perm], "t", ["x", "z"], base_weights="base")
    np.testing.assert_allclose(
        np.array(r["weights"].weight)[np.argsort(perm)], a["weights"].weight, atol=1e-10
    )


@pytest.mark.parametrize(
    "kind", ["range", "redundant", "convex_hull", "zero_weight", "nonconvergence"]
)
def test_entropy_rejects_invalid_or_infeasible_targets(kind):
    data = fixture()
    opts = {}
    if kind == "range":
        data.loc[data.t == 1, "x"] = 100
    elif kind == "redundant":
        data.z = data.x * 2
    elif kind == "convex_hull":
        # Target inside each coordinate range but outside the joint hull.
        data = pd.DataFrame(
            dict(t=[0, 0, 0, 0, 1, 1], x=[0, 1, 0, 0.2, 0.8, 0.9], z=[0, 0, 1, 0.2, 0.8, 0.9])
        )
    elif kind == "zero_weight":
        data.loc[data.index[0], "base"] = 0
        opts["base_weights"] = "base"
    elif kind == "nonconvergence":
        opts["max_iterations"] = 1
    with pytest.raises(AnalysisError):
        oe.ebalance(data, "t", ["x", "z"], **opts)


def test_cem_author_stratum_weights_and_cut_boundary():
    data = pd.DataFrame(
        dict(t=[0, 0, 1, 1, 0, 1, 1, 0], x=[0, 0.2, 0.4, 0.8, 1, 1.2, 1.5, 2.5], cat=["a"] * 8)
    )
    r = oe.cem(data, "t", ["x"], cutpoints={"x": [1.0, 2.0]}, categorical=["cat"])
    np.testing.assert_allclose(r["weights"].weight, [1, 1, 1, 1, 2, 1, 1, 0])
    assert list(r["weights"].stratum) == [0, 0, 0, 0, 1, 1, 1, 2]
    assert r.attrs["state"]["retained_positions"] == list(range(7))
    assert r.attrs["state"]["excluded_positions"] == [7]
    assert r.attrs["state"]["retained_treated"] == 4


def test_cem_target_loss_categorical_only_and_no_support():
    data = pd.DataFrame(dict(t=[0, 1, 1, 0], cat=["a", "a", "b", "c"]))
    r = oe.cem(data, "t", [], cutpoints={}, categorical=["cat"])
    assert r.attrs["state"]["retained_treated"] == 1
    assert r.attrs["state"]["original_complete_treated"] == 2
    with pytest.raises(AnalysisError, match="No prespecified"):
        oe.cem(data.assign(cat=["a", "b", "b", "a"]), "t", [], cutpoints={}, categorical=["cat"])


@pytest.mark.parametrize("cuts", [{}, {"x": [1, 1]}, {"x": [2, 1]}, {"x": [np.inf]}, {"x": [True]}])
def test_cem_invalid_cuts(cuts):
    with pytest.raises(AnalysisError):
        oe.cem(fixture(), "t", ["x"], cutpoints=cuts)


def test_weighted_balance_independent_covariance_smd_and_ecdf():
    data = fixture().rename(columns={"base": "w"})
    data.loc[data.index[[0, 70]], "w"] = 0
    r = oe.balance(data, "t", ["x", "z"], balance_weights="w")
    x = data[["x", "z"]].values
    covariance, means, variance = np.zeros((2, 2)), [], []
    for group in (0, 1):
        mask = data.t.values == group
        values, weights = x[mask], data.w.values[mask]
        p = weights / weights.sum()
        mean = p @ values
        centered = values - mean
        nn = (weights > 0).sum()
        covariance += (centered.T * p**2) @ centered * nn / (nn - 1)
        means.append(mean)
        variance.append(values.var(axis=0, ddof=1))
    np.testing.assert_allclose(r["covariance"], covariance, atol=1e-14)
    pool = np.sqrt((variance[0] + variance[1]) / 2)
    np.testing.assert_allclose(r["balance"].smd_after, (means[1] - means[0]) / pool)
    for j, name in enumerate(["x", "z"]):
        points = np.unique(x[:, j])
        fs = []
        for g in (0, 1):
            mask = data.t.values == g
            fs.append(
                np.array(
                    [
                        data.w.values[mask][x[mask, j] <= v].sum() / data.w.values[mask].sum()
                        for v in points
                    ]
                )
            )
        assert r["balance"].ecdf_distance_after.iloc[j] == pytest.approx(
            np.max(np.abs(fs[1] - fs[0]))
        )
    assert r.attrs["state"]["zero_weight_positions"] == [0, 70]


def test_balance_ties_degenerate_fixed_scale():
    data = pd.DataFrame(dict(t=[0, 0, 1, 1], x=[2.0, 2.0, 2.0, 2.0]))
    r = oe.balance(data, "t", ["x"])
    assert pd.isna(r["balance"].smd_after.iloc[0])
    assert r["balance"].ecdf_distance_after.iloc[0] == 0
    assert r["balance"].std_error.iloc[0] == 0
    assert pd.isna(r["balance"].p_value.iloc[0])


@pytest.mark.parametrize(
    "name,kwargs", [("ebalance", {}), ("cem", {"cutpoints": {"x": [0]}}), ("balance", {})]
)
def test_full_artifact_roundtrip_tamper_and_input_immutability(name, kwargs, tmp_path):
    data = fixture()
    original = data.copy(deep=True)
    r = getattr(oe, name)(data, "t", ["x"], **kwargs)
    artifact = oe.causal_design_save(r, tmp_path / "full.json")
    restored = oe.causal_design_load(tmp_path / "full.json")
    assert oe.causal_design_save(restored) == artifact
    assert (
        oe.causal_design_save(oe.causal_design_load(json.loads(json.dumps(artifact)))) == artifact
    )
    pd.testing.assert_frame_equal(data, original)
    changed = copy.deepcopy(artifact)
    changed["payload"]["attrs"]["state"]["settings"]["treatment"] = "different"
    with pytest.raises(AnalysisError):
        oe.causal_design_load(changed)
    r[next(iter(r))].iloc[0, 0] = 99999
    with pytest.raises(AnalysisError):
        oe.causal_design_save(r)


@pytest.mark.parametrize(
    "name,kwargs", [("ebalance", {}), ("cem", {"cutpoints": {"x": [0]}}), ("balance", {})]
)
@pytest.mark.parametrize(
    "options", [{"device": "mps"}, {"weights": "w"}, {"max_work": 1}, {"missing": "bad"}]
)
def test_explicit_domain_budgets(name, kwargs, options):
    with pytest.raises(AnalysisError):
        getattr(oe, name)(fixture(), "t", ["x"], **kwargs, **options)


def test_missing_alignment_and_duplicate_roles():
    data = fixture()
    data.iloc[5, data.columns.get_loc("x")] = np.nan
    with pytest.raises(AnalysisError):
        oe.ebalance(data, "t", ["x"])
    r = oe.ebalance(data, "t", ["x"], missing="drop")
    assert r.attrs["positions"] == [i for i in range(len(data)) if i != 5]
    for names in (["x", "x"], ["t"]):
        with pytest.raises(AnalysisError):
            oe.ebalance(data, "t", names)
