"""Independent nuisance/score/covariance oracles and causal admission contracts."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm
from pydantic import ValidationError

import openecon as oe


def sample(n=600, seed=701, hetero=False, iv=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-0.3 * x[:, 0] + 0.2 * x[:, 1]))
    if iv:
        z = rng.binomial(1, p)
        u = rng.uniform(size=n)
        d = (u < (0.15 + 0.65 * z)).astype(int)
    else:
        d = rng.binomial(1, p)
    effect = 2 + 1.5 * x[:, 0] if hetero else np.repeat(2.0, n)
    y = 1 + effect * d + 0.5 * x[:, 0] + rng.normal(size=n)
    df = pd.DataFrame(x, columns=["x1", "x2", "x3"])
    df["d"], df["y"] = d, y
    df["group"] = np.where(x[:, 0] > 0, "high", "low")
    df["policy"], df["w"] = (x[:, 0] > 0).astype(float), 1 + 0.15 * abs(x[:, 1])
    df["cluster"] = np.arange(n) // 4
    if iv:
        df["z"] = z
    return df


def run(df, name="dmlirm", **kw):
    return getattr(oe, name)(data=df, y="y", treatment="d", x=["x1", "x2", "x3"], **kw)


def coefficients(result):
    return np.array([row.estimate for row in result.coefficients])


def covariance(contributions, kind="HC0", codes=None):
    if codes is not None:
        totals = np.stack([contributions[codes == code].sum(0) for code in np.unique(codes)])
        g = len(totals)
        return totals.T @ totals * g / (g - 1)
    v = contributions.T @ contributions
    return v * len(contributions) / (len(contributions) - 1) if kind == "HC1" else v


def independent_predictions(df, result, instrument=None):
    assignment = df.d.to_numpy() if instrument is None else df[instrument].to_numpy()
    values = {
        key: np.empty(len(df))
        for key in ["propensity", "outcome0", "outcome1"]
        + ([] if instrument is None else ["treatment0", "treatment1"])
    }
    raw = np.column_stack(
        [
            df[item["column"]].to_numpy(dtype=float)
            if item["encoding"] == "numeric"
            else (df[item["column"]].to_numpy() == item["level"]).astype(float)
            for item in result.extra["nuisance_design"]
        ]
    )
    for rec in result.extra["fold_records"]:
        train, test = np.array(rec["train_positions"]), np.array(rec["test_positions"])
        design_train = sm.add_constant(raw[train], has_constant="add")
        design_test = sm.add_constant(raw[test], has_constant="add")
        model = sm.GLM(assignment[train], design_train, family=sm.families.Binomial()).fit(
            tol=1e-12
        )
        values["propensity"][test] = model.predict(design_test)
        for level in (0, 1):
            rows = train[assignment[train] == level]
            design = sm.add_constant(raw[rows], has_constant="add")
            values[f"outcome{level}"][test] = (
                sm.OLS(df.y.to_numpy()[rows], design).fit().predict(design_test)
            )
            if instrument is not None:
                values[f"treatment{level}"][test] = (
                    sm.GLM(df.d.to_numpy()[rows], design, family=sm.families.Binomial())
                    .fit(tol=1e-12)
                    .predict(design_test)
                )
    for name, predicted in values.items():
        np.testing.assert_allclose(
            result.extra["nuisance_predictions"][name], predicted, atol=5e-7, rtol=2e-7
        )
    return values


def potential(df, pred, treatment="d"):
    d, y, e = df[treatment].to_numpy(), df.y.to_numpy(), pred["propensity"]
    return (
        pred["outcome0"] + (1 - d) * (y - pred["outcome0"]) / (1 - e),
        pred["outcome1"] + d * (y - pred["outcome1"]) / e,
    )


@pytest.mark.parametrize("estimand", ["ate", "atet"])
@pytest.mark.parametrize("kind", ["HC0", "HC1", "cluster"])
@pytest.mark.parametrize("weighted", [False, True])
def test_irm_against_independent_glm_and_ratio_score(estimand, kind, weighted):
    df = sample()
    result = run(
        df,
        estimand=estimand,
        covariance=kind,
        **({"cluster": "cluster"} if kind == "cluster" else {}),
        **({"weights": "w"} if weighted else {}),
    )
    pred = independent_predictions(df, result)
    phi0, phi1 = potential(df, pred)
    d, y, e = df.d.to_numpy(), df.y.to_numpy(), pred["propensity"]
    b = (
        phi1 - phi0
        if estimand == "ate"
        else d * (y - pred["outcome0"]) - (1 - d) * e * (y - pred["outcome0"]) / (1 - e)
    )
    a = np.ones(len(df)) if estimand == "ate" else d
    w = df.w.to_numpy() if weighted else np.ones(len(df))
    mass = w / w.sum()
    estimate = (mass @ b) / (mass @ a)
    contribution = (mass * (b - estimate * a) / (mass @ a))[:, None]
    reference = covariance(contribution, kind, df.cluster.to_numpy() if kind == "cluster" else None)
    np.testing.assert_allclose(coefficients(result), [estimate], atol=5e-7)
    np.testing.assert_allclose(result.covariance_matrix, reference, atol=3e-9, rtol=3e-6)
    row = result.coefficients[0]
    distribution = stats.t(df.cluster.nunique() - 1) if kind == "cluster" else stats.norm
    assert row.p_value == pytest.approx(2 * distribution.sf(abs(row.statistic)), abs=2e-10)
    assert row.ci_low == pytest.approx(
        row.estimate - distribution.ppf(0.975) * row.std_error, abs=2e-9
    )
    assert result.provenance["stata_parity_validated"] is False
    assert len(result.extra["diagnostics"]["balance"]) == 3


def test_iivm_independent_ratio_covariance_and_truth():
    df = sample(n=1200, iv=True)
    result = run(df, "dmliivm", instrument="z")
    p = independent_predictions(df, result, instrument="z")
    y0, y1 = potential(df, p, treatment="z")
    d, z, e = df.d.to_numpy(), df.z.to_numpy(), p["propensity"]
    a = (
        p["treatment1"]
        + z * (d - p["treatment1"]) / e
        - p["treatment0"]
        - (1 - z) * (d - p["treatment0"]) / (1 - e)
    )
    theta = np.mean(y1 - y0) / np.mean(a)
    contribution = ((y1 - y0 - theta * a) / np.mean(a) / len(df))[:, None]
    np.testing.assert_allclose(coefficients(result), [theta], atol=5e-7)
    np.testing.assert_allclose(result.covariance_matrix, covariance(contribution), atol=3e-9)
    assert abs(theta - 2) < 0.3
    assert result.extra["first_stage"] > 0.5


def test_basis_projection_full_covariance_wald_and_saved_prediction():
    df = sample(hetero=True)
    result = run(df, "dmlcate", basis=["x1", "x2"], covariance="HC1")
    p = independent_predictions(df, result)
    y0, y1 = potential(df, p)
    design = sm.add_constant(df[["x1", "x2"]], has_constant="add").to_numpy()
    reference = sm.OLS(y1 - y0, design).fit(cov_type="HC0")
    expected_v = np.asarray(reference.cov_params()) * len(df) / (len(df) - 1)
    np.testing.assert_allclose(coefficients(result), reference.params, atol=6e-7)
    np.testing.assert_allclose(result.covariance_matrix, expected_v, atol=3e-9)
    b, v = coefficients(result)[1:], np.asarray(result.covariance_matrix)[1:, 1:]
    expected_stat = b @ np.linalg.solve(v, b)
    assert result.tests["heterogeneity"]["statistic"] == pytest.approx(expected_stat)
    assert result.tests["heterogeneity"]["p_value"] == pytest.approx(
        stats.chi2.sf(expected_stat, 2)
    )
    saved = oe.ResultBundle.model_validate_json(result.model_dump_json())
    query = df.iloc[:4].set_axis([17, 17, 3, 1])
    predictions = oe.causal_predict(saved, query)
    np.testing.assert_allclose(predictions.cate, design[:4] @ reference.params, atol=6e-7)
    np.testing.assert_allclose(
        predictions.attrs["covariance_matrix"], design[:4] @ expected_v @ design[:4].T, atol=3e-9
    )
    assert list(predictions.index) == [17, 17, 3, 1]


def test_declared_group_and_policy_full_joint_covariance():
    df = sample(hetero=True)
    gate = run(df, "dmlgate", group="group", cluster="cluster")
    pred = independent_predictions(df, gate)
    y0, y1 = potential(df, pred)
    labels = gate.extra["group_labels"]
    expected, contributions = [], []
    for label in labels:
        selected = df.group.to_numpy() == label
        theta = np.mean((y1 - y0)[selected])
        expected.append(theta)
        contributions.append(selected * (y1 - y0 - theta) / selected.sum())
    np.testing.assert_allclose(coefficients(gate), expected, atol=6e-7)
    np.testing.assert_allclose(
        gate.covariance_matrix,
        covariance(np.array(contributions).T, codes=df.cluster.to_numpy()),
        atol=3e-9,
    )
    policy = run(df, "policyvalue", policy="policy", cost=0.4, cluster="cluster")
    pred = independent_predictions(df, policy)
    y0, y1 = potential(df, pred)
    q = df.policy.to_numpy()
    value = q * (y1 - 0.4) + (1 - q) * y0
    scores = np.column_stack([value, y1 - 0.4, y0, value - y0])
    np.testing.assert_allclose(coefficients(policy), scores.mean(0), atol=6e-7)
    np.testing.assert_allclose(
        policy.covariance_matrix,
        covariance((scores - scores.mean(0)) / len(df), codes=df.cluster.to_numpy()),
        atol=3e-9,
    )
    assert abs(policy.covariance_matrix[0][3]) > 0


def test_whole_cluster_folds_and_direct_crossfit_leakage():
    df = sample(n=400)
    first = run(df, cluster="cluster")
    for rec in first.extra["fold_records"]:
        train, test = rec["train_positions"], rec["test_positions"]
        assert not set(df.cluster.iloc[train]) & set(df.cluster.iloc[test])
    rec = first.extra["fold_records"][0]
    changed = df.copy()
    changed.loc[rec["test_positions"], "y"] += 1000
    second = run(changed, cluster="cluster")
    assert first.extra["fold_records"][0]["models"] == second.extra["fold_records"][0]["models"]
    for name in first.extra["nuisance_predictions"]:
        np.testing.assert_array_equal(
            np.array(first.extra["nuisance_predictions"][name])[rec["test_positions"]],
            np.array(second.extra["nuisance_predictions"][name])[rec["test_positions"]],
        )


def manual_leaf(tree, row):
    node = 0
    while "feature" in tree[node]:
        item = tree[node]
        node = item["left"] if row[item["feature"]] <= item["threshold"] else item["right"]
    return node


@pytest.mark.parametrize(
    "weighted,clustered", [(False, False), (True, False), (False, True), (True, True)]
)
def test_honest_forest_manual_weights_variance_state_and_indirect_leakage(weighted, clustered):
    df = sample(n=600, hetero=True)
    options = dict(trees=8, max_depth=3, min_leaf=15, query=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    if weighted:
        options["weights"] = "w"
    if clustered:
        options["cluster"] = "cluster"
    result = run(df, "causalforest", **options)
    state = result.extra["cate_state"]
    x = np.array(state["estimation_x"])
    scores = np.array(state["scores"])
    weights = np.zeros((2, len(scores)))
    admitted_weights = np.array(state["importance_weights"])
    query = np.array(options["query"])
    for tree in state["trees"]:
        membership = np.array([manual_leaf(tree, row) for row in x])
        for j, row in enumerate(query):
            chosen = membership == manual_leaf(tree, row)
            weights[j, chosen] += (
                admitted_weights[chosen] / admitted_weights[chosen].sum() / len(state["trees"])
            )
    values = weights @ scores
    contributions = (weights * (scores[None, :] - values[:, None])).T
    np.testing.assert_allclose([row["cate"] for row in result.extra["cate"]], values, atol=1e-12)
    np.testing.assert_allclose(
        result.extra["cate_covariance"],
        covariance(
            contributions, codes=None if not clustered else np.array(state["cluster_codes"])
        ),
        atol=1e-12,
    )
    assert values[1] > values[0] + (1.5 if not clustered and not weighted else 0.5)
    n, s, e = [
        set(result.extra[key])
        for key in ("nuisance_positions", "structure_positions", "estimation_positions")
    ]
    assert not n & s and not n & e and not s & e and n | s | e == set(range(len(df)))
    if clustered:
        assert not set(df.cluster.iloc[list(n)]) & set(df.cluster.iloc[list(e)])
        assert not set(df.cluster.iloc[list(s)]) & set(df.cluster.iloc[list(e)])
    changed = df.copy()
    changed.loc[list(e), "y"] += 1000
    again = run(changed, "causalforest", **options)
    assert result.extra["nuisance_models"] == again.extra["nuisance_models"]
    assert state["trees"] == again.extra["cate_state"]["trees"]
    saved = oe.ResultBundle.model_validate_json(result.model_dump_json())
    np.testing.assert_allclose(
        oe.causal_predict(saved, pd.DataFrame(query, columns=["x1", "x2", "x3"])).cate, values
    )
    saved.extra["cate_state"]["scores"][0] += 1
    with pytest.raises(oe.AnalysisError) as caught:
        oe.causal_predict(saved, df.iloc[:2])
    assert caught.value.code == "causal_state_integrity"


@pytest.mark.parametrize(
    "name,kw", [("dmlgate", {"groups": 2}), ("policyvalue", {"learned": True})]
)
def test_learned_rankings_and_policies_never_use_evaluation_outcomes(name, kw):
    df = sample(hetero=True)
    result = run(df, name, trees=8, max_depth=3, min_leaf=12, **kw)
    changed = df.copy()
    changed.loc[result.extra["estimation_positions"], "y"] += 100
    again = run(changed, name, trees=8, max_depth=3, min_leaf=12, **kw)
    key = "group_assignment" if name == "dmlgate" else "policy_probabilities"
    assert result.extra[key] == again.extra[key]
    assert result.extra["nuisance_models"] == again.extra["nuisance_models"]


@pytest.mark.parametrize("learner", ["lasso", "ridge", "forest"])
def test_native_nuisance_learners_are_complete_and_deterministic(learner):
    df = sample()
    kw = dict(learner=learner)
    if learner == "forest":
        kw.update(trees=8, max_depth=3, min_leaf=12)
    else:
        kw.update(penalty=0.04, max_iterations=5000, tolerance=1e-7)
    first = run(df, **kw)
    second = run(df, **kw)
    np.testing.assert_array_equal(
        first.extra["nuisance_predictions"]["propensity"],
        second.extra["nuisance_predictions"]["propensity"],
    )
    assert abs(first.coefficients[0].estimate - 2) < 0.5
    assert len(first.extra["fold_records"]) == 5


@pytest.mark.parametrize("nuisance", ["ridge", "forest"])
def test_plr_cluster_folds_and_independent_covariance(nuisance):
    df = sample()
    kw = dict(nuisance=nuisance, cluster="cluster")
    if nuisance == "forest":
        kw["forest_config"] = {"trees": 8, "max_depth": 3, "min_leaf": 10}
    else:
        kw.update(selection="fixed", penalty=0.01)
    result = oe.dmlplr(data=df, y="y", treatment="d", x=["x1", "x2", "x3"], **kw)
    contribution = np.array(result.extra["influence"])[:, None] / len(df)
    np.testing.assert_allclose(
        result.covariance_matrix, covariance(contribution, codes=df.cluster.to_numpy()), atol=2e-12
    )
    for rec in result.extra["fold_records"]:
        assert not set(df.cluster.iloc[rec["train_positions"]]) & set(
            df.cluster.iloc[rec["test_positions"]]
        )
    assert result.inference["df_inference"] == df.cluster.nunique() - 1


@pytest.mark.parametrize(
    "name,kw,change,code",
    [
        ("dmlirm", {}, lambda d: d.assign(d=2), "invalid_binary_role"),
        ("dmlirm", {"max_work": 1}, lambda d: d, "work_limit"),
        ("dmlirm", {"weights": "w"}, lambda d: d.assign(w=-1), "invalid_importance_weights"),
        (
            "dmlirm",
            {"weights": "w"},
            lambda d: d.assign(w=np.where(d.index == 0, 0, d.w)),
            "invalid_importance_weights",
        ),
        ("dmlirm", {"overlap": 0.49}, lambda d: d, "overlap_violation"),
        ("dmlirm", {"trees": 3}, lambda d: d, "unused_options"),
        ("dmlirm", {}, lambda d: d.assign(y=np.nan), "missing_values"),
        ("dmlcate", {"basis": ["x1", "x2"]}, lambda d: d.assign(x2=d.x1), "singular_basis"),
        ("dmlcate", {"basis": ["d"]}, lambda d: d, "invalid_basis"),
        ("dmlgate", {"group": "d"}, lambda d: d, "invalid_groups"),
        ("dmlgate", {"group": "group", "groups": 3}, lambda d: d, "unused_options"),
        (
            "causalforest",
            {"query": [[999.0, 0.0, 0.0]], "trees": 2},
            lambda d: d,
            "query_outside_support",
        ),
        ("causalforest", {"query": [0.0], "trees": 2}, lambda d: d, "invalid_query"),
        ("policyvalue", {"policy": "policy"}, lambda d: d.assign(policy=2.0), "invalid_policy"),
        ("policyvalue", {}, lambda d: d, "invalid_policy"),
        ("policyvalue", {"policy": "policy", "learned": True}, lambda d: d, "invalid_policy"),
    ],
)
def test_admission_guards(name, kw, change, code):
    with pytest.raises(oe.AnalysisError) as caught:
        run(change(sample()), name, **kw)
    assert caught.value.code == code


def test_weak_iv_missing_alignment_modelspec_and_torch_free_catalog():
    df = sample(iv=True)
    with pytest.raises(oe.AnalysisError) as caught:
        run(df, "dmliivm", instrument="z", min_first_stage=1.0)
    assert caught.value.code == "weak_first_stage"
    data = sample()
    data.loc[11, "x1"] = np.nan
    result = run(data, missing="drop")
    assert 11 not in result.sample_positions and result.dropped_rows == 1
    assert len(result.extra["fold_assignment"]) == len(data) - 1
    for rec in result.extra["fold_records"]:
        assert 11 not in rec["train_positions"] + rec["test_positions"]
    with pytest.raises(ValidationError):
        oe.ModelSpec(
            estimator="dmlirm",
            outcome="y",
            predictors=["x1"],
            columns={"treatment": "d"},
            weights="w",
            weight_type="pweight",
        )
    code = "import sys,openecon as oe; assert 'dmlirm' in oe.capabilities()['estimators']; assert 'torch' not in sys.modules; assert 'sklearn' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
    saved = oe.ResultBundle.model_validate_json(result.model_dump_json())
    assert saved.extra == result.extra
    assert json.loads(saved.model_dump_json())["provenance"]["precision"] == "float64"
