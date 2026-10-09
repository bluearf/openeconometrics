"""Independent factor-weighted optima, honest folds and saved prediction."""

import itertools

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


def data():
    rng = np.random.default_rng(20261007)
    x = rng.normal(size=(88, 4)) * [0.2, 3, 10, 0.7] + [20, -2, 30, 5]
    x[:, 1] += x[:, 0] * 5
    y = 2 + x @ [3, 0.6, -0.3, 0.1] + rng.normal(size=len(x))
    return pd.DataFrame(dict(zip(list("abcd"), x.T, strict=True)) | {"y": y})


def transform(x, y, intercept, standardize):
    center = x.mean(0) if intercept else np.zeros(x.shape[1])
    ycenter = y.mean() if intercept else 0
    z = x - center
    scale = np.sqrt(np.mean(z**2, axis=0)) if standardize else np.ones(x.shape[1])
    scale[scale == 0] = 1
    return z / scale, y - ycenter, center, scale, ycenter


def independent_optimum(z, y, penalty, ratio, factors):
    """Enumerate penalized active sets and jointly solve the forced block."""
    q = np.asarray(factors)
    gram, score = z.T @ z / len(y), z.T @ y / len(y)
    if ratio == 0:
        return np.linalg.solve(gram + penalty * np.diag(q), score)
    penalized, forced = np.flatnonzero(q > 0), np.flatnonzero(q == 0)
    candidates = []
    for choices in itertools.product([-1, 0, 1], repeat=len(penalized)):
        sign = np.zeros(len(q))
        sign[penalized] = choices
        active = np.r_[forced, np.flatnonzero(sign)]
        beta = np.zeros(len(q))
        if len(active):
            beta[active] = np.linalg.solve(
                gram[np.ix_(active, active)] + penalty * (1 - ratio) * np.diag(q[active]),
                score[active] - penalty * ratio * q[active] * sign[active],
            )
        chosen = np.flatnonzero(sign)
        if np.any(beta[chosen] * sign[chosen] <= 0):
            continue
        gradient = gram @ beta - score + penalty * (1 - ratio) * q * beta
        if np.max(np.abs(gradient[forced]), initial=0) > 1e-8:
            continue
        inactive = np.setdiff1d(penalized, chosen)
        if np.any(abs(gradient[inactive]) > penalty * ratio * q[inactive] + 1e-8):
            continue
        objective = np.mean((y - z @ beta) ** 2) / 2 + penalty * (
            ratio * np.sum(q * abs(beta)) + (1 - ratio) * np.sum(q * beta**2) / 2
        )
        candidates.append((objective, beta))
    assert candidates
    return min(candidates, key=lambda candidate: candidate[0])[1]


@pytest.mark.parametrize("name,ratio", [("ridge", 0), ("lasso", 1), ("elasticnet", 0.4)])
@pytest.mark.parametrize("intercept,standardize", [(True, True), (True, False), (False, True)])
def test_full_factor_paths_joint_forced_kkt_and_restored_prediction(name, ratio, intercept, standardize, tmp_path, monkeypatch):
    d = data()
    factors = np.array([0, 0.25, 2, 4.0])
    options = dict(selection="fixed", penalty=0.3, lambda_path=[2.0, 0.3, 0.0],
                   penalty_factors=[3, 0.25, 2, 4], forced_controls=["a"],
                   intercept=intercept, standardize=standardize,
                   tolerance=1e-11, max_iterations=30000)
    if name == "elasticnet":
        options["l1_ratio"] = ratio
    result = getattr(oe, name)(data=d, y="y", x=list("abcd"), **options)
    state = result.extra["penalized_state"]
    assert state["penalty_factors"] == list(factors)
    assert state["forced_controls"] == ["a"]
    z, yc, center, scale, ym = transform(d[list("abcd")].to_numpy(), d.y.to_numpy(), intercept, standardize)
    for lam, beta, diag in zip(state["lambda_path"], state["coefficient_path"], state["path_diagnostics"], strict=True):
        reference = independent_optimum(z, yc, lam, ratio, factors)
        assert_allclose(beta, reference / scale, atol=2e-8, rtol=1e-8)
        objective = np.mean((yc - z @ reference) ** 2) / 2 + lam * (
            ratio * np.sum(factors * abs(reference))
            + (1 - ratio) * np.sum(factors * reference**2) / 2
        )
        assert diag["objective"] == pytest.approx(objective, abs=1e-9)
        gradient = z.T @ (z @ reference - yc) / len(yc) + lam * (1-ratio) * factors * reference
        assert abs(gradient[0]) < 1e-8
        assert diag["kkt_max"] < 2e-8
    path = tmp_path / "complete-model.json"
    path.write_text(result.model_dump_json())
    restored = ResultBundle.model_validate_json(path.read_text())
    monkeypatch.setattr("openecon.econometrics.regularized.prediction.fit_penalized", lambda *a, **k: pytest.fail("restoration must not fit"))
    query = d.iloc[[7, 2, 80]].copy()
    query["y"] = [1e50, -1e50, 0]
    expected = query[list("abcd")].to_numpy() @ np.asarray(state["coefficients"]) + state["constant"]
    predicted = oe.regularized_predict(restored, query)
    assert list(predicted.index) == [7, 2, 80]
    assert_allclose(predicted, expected, atol=1e-10)
    assert result.coefficients == result.covariance_matrix == []
    assert result.inference["available"] is False


def test_factor_cv_training_projection_scaling_and_label_isolation():
    d = data()
    factors = np.array([0, 0.25, 2, 4])
    options = dict(selection="cv", penalty_factors=list(factors), n_lambdas=4,
                   lambda_ratio=0.01, folds=4, seed=927, tolerance=1e-11)
    result = oe.ridge(data=d, y="y", x=list("abcd"), **options)
    state = result.extra["penalized_state"]
    assignments = np.asarray(state["fold_assignments"])
    pooled = np.zeros(4)
    for fold in range(4):
        train, test = assignments != fold, assignments == fold
        z, yc, center, scale, ym = transform(d.loc[train, list("abcd")].to_numpy(), d.loc[train, "y"].to_numpy(), True, True)
        residual = yc - z[:, [0]] @ np.linalg.lstsq(z[:, [0]], yc, rcond=None)[0]
        top = max(np.max(abs(z[:, 1:].T @ residual / sum(train)) / factors[1:]) / 0.01, 1e-8)
        path = top * np.geomspace(1, 0.01, 4)
        assert_allclose(state["fold_lambda_paths"][fold], path, rtol=1e-12)
        for j, lam in enumerate(path):
            beta = independent_optimum(z, yc, lam, 0, factors)
            predicted = (d.loc[test, list("abcd")].to_numpy() - center) / scale @ beta + ym
            errors = (d.loc[test, "y"].to_numpy() - predicted)**2
            pooled[j] += errors.sum()
            assert state["fold_mse"][fold][j] == pytest.approx(errors.mean(), rel=1e-10)
            assert state["fold_path_diagnostics"][fold][j]["kkt_max"] < 1e-8
    assert_allclose(state["cv_mse"], pooled/len(d), rtol=1e-10)
    assert state["selected_index"] == int(pooled.argmin())
    changed = d.copy()
    changed.loc[assignments == 0, "y"] += 10000
    other = oe.ridge(data=changed, y="y", x=list("abcd"), **options).extra["penalized_state"]
    assert other["fold_lambda_paths"][0] == state["fold_lambda_paths"][0]
    assert other["fold_path_diagnostics"][0] == state["fold_path_diagnostics"][0]
    assert other["fold_assignments"] == state["fold_assignments"]


def test_weighted_lasso_lambda_max_retains_forced_projection_and_zero_slopes():
    d = data()
    result = oe.lasso(data=d, y="y", x=list("abcd"), forced_controls=["a"],
                      penalty_factors=[1, 0.25, 2, 4], selection="cv", n_lambdas=3,
                      lambda_ratio=0.1, folds=4, tolerance=1e-11)
    state = result.extra["penalized_state"]
    z, yc, _, scale, _ = transform(d[list("abcd")].to_numpy(), d.y.to_numpy(), True, True)
    beta = np.asarray(state["coefficient_path"][0])*scale
    assert_allclose(beta[1:], 0, atol=2e-9)
    expected = np.linalg.lstsq(z[:, [0]], yc, rcond=None)[0][0]
    assert beta[0] == pytest.approx(expected, abs=2e-9)


def test_literal_factor_and_predictor_unit_conventions_are_invariant_when_reconciled():
    d = data()
    options = dict(selection="fixed", penalty=.2, penalty_factors=[0, .25, 2, 4],
                   standardize=True, tolerance=1e-11, l1_ratio=.6)
    original = oe.elasticnet(data=d, y="y", x=list("abcd"), **options)
    units = d.copy()
    units[list("abcd")] = units[list("abcd")].to_numpy()*[100, .001, 2, 10]
    rescaled = oe.elasticnet(data=units, y="y", x=list("abcd"), **options)
    normalized = oe.elasticnet(data=d, y="y", x=list("abcd"),
                               **(options | {"penalty": .02, "penalty_factors": [0, 2.5, 20, 40]}))
    assert_allclose(oe.regularized_predict(original, d), oe.regularized_predict(rescaled, units), atol=1e-9)
    assert_allclose(oe.regularized_predict(original, d), oe.regularized_predict(normalized, d), atol=1e-10)


def test_all_forced_fixed_fit_has_exact_ols_limit():
    d = data()
    result = oe.ridge(data=d, y="y", x=list("abcd"), selection="fixed", penalty=123,
                      forced_controls=list("abcd"))
    design = np.column_stack([np.ones(len(d)), d[list("abcd")].to_numpy()])
    beta = np.linalg.lstsq(design, d.y.to_numpy(), rcond=None)[0]
    assert_allclose(oe.regularized_predict(result, d), design@beta, atol=1e-10)
    with pytest.raises(AnalysisError) as caught:
        oe.ridge(data=d, y="y", x=list("abcd"), forced_controls=list("abcd"))
    assert caught.value.code == "invalid_penalty"


@pytest.mark.parametrize("options,code", [
    ({"penalty_factors": [0, 1]}, "invalid_penalty_factors"),
    ({"penalty_factors": [0, 1, -1, 1]}, "invalid_penalty_factors"),
    ({"penalty_factors": [0, 1, True, 1]}, "invalid_penalty_factors"),
    ({"penalty_factors": [0, 1, float("inf"), 1]}, "invalid_penalty_factors"),
    ({"forced_controls": ["unknown"]}, "invalid_forced_controls"),
    ({"forced_controls": ["a", "a"]}, "invalid_forced_controls"),
    ({"forced_controls": ["a"], "max_work": 1}, "work_limit"),
    ({"penalty_factors": [1e308]*4}, "numerical_failure"),
])
def test_factor_and_forced_negative_domains(options, code):
    with pytest.raises(AnalysisError) as caught:
        oe.ridge(data=data(), y="y", x=list("abcd"), selection="fixed", penalty=10, **options)
    assert caught.value.code == code


def test_forced_rank_refusal_full_sample_and_training_fold():
    d = data()
    d["b"] = d.a
    with pytest.raises(AnalysisError) as caught:
        oe.ridge(data=d, y="y", x=list("abcd"), forced_controls=["a", "b"],
                 selection="fixed", penalty=0.1)
    assert caught.value.code == "unidentified_forced_controls"
    d = data()
    base = oe.ridge(data=d, y="y", x=list("abcd"), selection="cv", lambda_path=[0.1], folds=4, seed=381)
    assignment = np.asarray(base.extra["penalized_state"]["fold_assignments"])
    d["a"] = (assignment == 0).astype(float)
    with pytest.raises(AnalysisError) as caught:
        oe.ridge(data=d, y="y", x=list("abcd"), forced_controls=["a"],
                 selection="cv", lambda_path=[0.1], folds=4, seed=381)
    assert caught.value.code == "unidentified_forced_controls"


def test_factors_refuse_unsupported_plugin_and_workspace_before_fit():
    with pytest.raises(AnalysisError) as caught:
        oe.lasso(data=data(), y="y", x=list("abcd"), selection="plugin", forced_controls=["a"])
    assert caught.value.code == "unsupported_selection"
    large = pd.concat([data()] * 64, ignore_index=True)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as caught:
        oe.ridge(data=large, y="y", x=list("abcd"), selection="fixed", penalty=0.1,
                 forced_controls=["a"])
    assert caught.value.code == "workspace_limit"


@pytest.mark.parametrize("name", ["ridge", "lasso", "elasticnet"])
def test_declared_cpu_factor_fit_and_saved_prediction_override_foreign_default_device(name):
    import torch

    d = data()
    options = dict(selection="fixed", penalty=.1, forced_controls=["a"],
                   penalty_factors=[1, .25, 2, 4])
    if name == "elasticnet":
        options["l1_ratio"] = .4
    baseline = getattr(oe, name)(data=d, y="y", x=list("abcd"), **options)
    with torch.device("meta"):
        result = getattr(oe, name)(data=d, y="y", x=list("abcd"), **options)
        saved = ResultBundle.model_validate_json(result.model_dump_json())
        predicted = oe.regularized_predict(saved, d)
    assert_allclose(predicted, oe.regularized_predict(baseline, d), atol=0)
