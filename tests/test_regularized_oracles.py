"""Independent NumPy algebra/active-set oracles; never imported at runtime."""

import itertools
import json

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.models import ModelSpec, ResultBundle


def sample(n=120, p=5):
    random = np.random.default_rng(391)
    x = random.normal(size=(n, p)) * np.arange(1, p + 1) + np.arange(p) * 3
    d = 0.7 * x[:, 0] - 0.4 * x[:, 1] + random.normal(size=n)
    y = 1.6 * d + 0.5 * x[:, 1] + 0.2 * x[:, 2] + random.normal(size=n) * (1 + 0.3 * abs(x[:, 0]))
    df = pd.DataFrame(x, columns=[f"x{j}" for j in range(p)])
    df["d"], df["y"] = d, y
    return df, list(df.columns[:p])


def transformed(x, y, intercept, standardize):
    mean = x.mean(0) if intercept else np.zeros(x.shape[1])
    ym = y.mean() if intercept else 0.0
    z = x - mean
    scale = np.sqrt(np.mean(z**2, axis=0)) if standardize else np.ones(x.shape[1])
    scale[scale == 0] = 1
    return z / scale, y - ym, mean, scale, ym


def active_oracle(z, y, penalty, ratio, load=None):
    """Enumerate all sign/support patterns, solve active linear systems, check KKT.

    Independent of the production coordinate-descent update and its iterates.
    This deliberately small oracle tests convex optima including exact zeros.
    """
    n, p = z.shape
    load = np.ones(p) if load is None else np.asarray(load)
    gram, score = z.T @ z / n, z.T @ y / n
    if ratio == 0:
        return np.linalg.solve(gram + penalty * np.eye(p), score)
    candidates = []
    for sign_tuple in itertools.product([-1, 0, 1], repeat=p):
        sign = np.array(sign_tuple)
        active = np.flatnonzero(sign)
        beta = np.zeros(p)
        if len(active):
            try:
                beta[active] = np.linalg.solve(
                    gram[np.ix_(active, active)] + penalty * (1 - ratio) * np.eye(len(active)),
                    score[active] - penalty * ratio * load[active] * sign[active],
                )
            except np.linalg.LinAlgError:
                continue
            if np.any(beta[active] * sign[active] <= 0):
                continue
        gradient = gram @ beta - score + penalty * (1 - ratio) * beta
        inactive = sign == 0
        if np.any(abs(gradient[inactive]) > penalty * ratio * load[inactive] + 1e-9):
            continue
        obj = np.mean((y - z @ beta) ** 2) / 2 + penalty * (
            ratio * np.sum(load * abs(beta)) + (1 - ratio) * np.sum(beta**2) / 2
        )
        candidates.append((obj, beta))
    assert candidates, "independent active-set oracle failed"
    return min(candidates, key=lambda pair: pair[0])[1]


@pytest.mark.parametrize("name,ratio", [("ridge", 0), ("lasso", 1), ("elasticnet", 0.37)])
@pytest.mark.parametrize(
    "intercept,standardize", [(True, True), (True, False), (False, True), (False, False)]
)
def test_penalty_objective_scaling_intercept_and_persisted_prediction(
    name, ratio, intercept, standardize
):
    df, names = sample(p=4)
    options = dict(
        selection="fixed",
        penalty=0.34,
        intercept=intercept,
        standardize=standardize,
        tolerance=1e-11,
        max_iterations=10000,
    )
    if name == "elasticnet":
        options["l1_ratio"] = ratio
    result = getattr(oe, name)(data=df, y="y", x=names, **options)
    state = result.extra["penalized_state"]
    z, yc, mean, scale, ym = transformed(
        df[names].to_numpy(), df.y.to_numpy(), intercept, standardize
    )
    reference = active_oracle(z, yc, 0.34, ratio)
    np.testing.assert_allclose(state["standardized_coefficients"], reference, atol=3e-9)
    np.testing.assert_allclose(state["coefficients"], reference / scale, atol=3e-9)
    assert state["constant"] == pytest.approx(ym - mean @ (reference / scale), abs=3e-9)
    grad = z.T @ (z @ reference - yc) / len(yc) + 0.34 * (1 - ratio) * reference
    kkt = np.where(
        reference != 0,
        abs(grad + 0.34 * ratio * np.sign(reference)),
        np.maximum(abs(grad) - 0.34 * ratio, 0),
    ).max()
    assert kkt < 1e-9
    obj = np.mean((yc - z @ reference) ** 2) / 2 + 0.34 * (
        ratio * abs(reference).sum() + (1 - ratio) * (reference**2).sum() / 2
    )
    assert state["path_diagnostics"][0]["objective"] == pytest.approx(obj, abs=1e-8)
    assert result.coefficients == [] and result.covariance_matrix == []
    assert result.inference["available"] is False and result.extra["target"] == "prediction"
    saved = ResultBundle.model_validate_json(result.model_dump_json())
    query = df.iloc[[5, 1, 16]].copy()
    np.testing.assert_allclose(
        oe.regularized_predict(saved, query),
        query[names].to_numpy() @ (reference / scale) + state["constant"],
        atol=3e-8,
    )
    assert list(oe.regularized_predict(saved, query).index) == [5, 1, 16]
    latex = str(oe.regularized_table(saved).to_latex())
    assert "Estimate" in latex and names[0] in latex
    assert "Std." not in latex and "p-value" not in latex


def test_ridge_cv_independent_fold_training_scaling_scores_and_seed():
    df, names = sample(n=80, p=4)
    result = oe.ridge(
        data=df, y="y", x=names, selection="cv", lambda_path=[2.0, 0.5, 0.1], folds=4, seed=882
    )
    state = result.extra["penalized_state"]
    assignment = np.array(state["fold_assignments"])
    x, y = df[names].to_numpy(), df.y.to_numpy()
    pooled = np.zeros(3)
    for fold in range(4):
        train, test = assignment != fold, assignment == fold
        z, yc, mean, scale, ym = transformed(x[train], y[train], True, True)
        for j, penalty in enumerate(state["lambda_path"]):
            beta = active_oracle(z, yc, penalty, 0)
            predicted = (x[test] - mean) / scale @ beta + ym
            error = (y[test] - predicted) ** 2
            pooled[j] += error.sum()
            assert state["fold_mse"][fold][j] == pytest.approx(error.mean(), rel=1e-11)
    np.testing.assert_allclose(state["cv_mse"], pooled / len(y), rtol=1e-11)
    assert state["selected_index"] == np.argmin(pooled)
    again = oe.ridge(
        data=df, y="y", x=names, selection="cv", lambda_path=[2.0, 0.5, 0.1], folds=4, seed=882
    )
    assert again.extra["penalized_state"] == state
    alternate = oe.ridge(
        data=df, y="y", x=names, selection="cv", lambda_path=[2.0, 0.5, 0.1], folds=4, seed=883
    )
    assert alternate.extra["penalized_state"]["fold_assignments"] != state["fold_assignments"]


def test_automatic_cv_selects_fractions_with_honest_train_only_paths_and_numpy_oracle():
    df, names = sample(n=80, p=4)
    result = oe.ridge(
        data=df, y="y", x=names, selection="cv", n_lambdas=4, lambda_ratio=0.01, folds=4, seed=918
    )
    state = result.extra["penalized_state"]
    assignment = np.array(state["fold_assignments"])
    fractions = np.geomspace(1.0, 0.01, 4)
    np.testing.assert_allclose(state["cv_lambda_fractions"], fractions, atol=1e-14)
    scores = np.zeros(4)
    for fold in range(4):
        train, test = assignment != fold, assignment == fold
        z, yc, mean, scale, ym = transformed(
            df.loc[train, names].to_numpy(), df.loc[train, "y"].to_numpy(), True, True
        )
        top = max(np.max(abs(z.T @ yc / train.sum())) / 0.01, 1e-8)
        path = top * fractions
        np.testing.assert_allclose(state["fold_lambda_paths"][fold], path, rtol=1e-12)
        for j, lam in enumerate(path):
            beta = active_oracle(z, yc, lam, 0)
            predicted = (df.loc[test, names].to_numpy() - mean) / scale @ beta + ym
            error = (df.loc[test, "y"].to_numpy() - predicted) ** 2
            scores[j] += error.sum()
            assert state["fold_mse"][fold][j] == pytest.approx(error.mean(), abs=1e-10)
    np.testing.assert_allclose(state["cv_mse"], scores / len(df), atol=1e-10)
    assert state["selected_index"] == int(scores.argmin())
    assert state["selected_lambda_fraction"] == pytest.approx(fractions[scores.argmin()])
    assert state["cv_selector_units"] == "fraction of training-fold lambda_max"
    z, yc, _, _, _ = transformed(df[names].to_numpy(), df.y.to_numpy(), True, True)
    final_top = max(np.max(abs(z.T @ yc / len(df))) / 0.01, 1e-8)
    np.testing.assert_allclose(state["lambda_path"], final_top * fractions, rtol=1e-12)
    changed = df.copy()
    changed.loc[assignment == 0, "y"] += 10000.0
    second = oe.ridge(
        data=changed,
        y="y",
        x=names,
        selection="cv",
        n_lambdas=4,
        lambda_ratio=0.01,
        folds=4,
        seed=918,
    )
    assert second.extra["penalized_state"]["fold_lambda_paths"][0] == state["fold_lambda_paths"][0]


def test_lambda_path_fixed_and_plugin_loadings_independent_active_set():
    df, names = sample(p=4)
    path = oe.lasso(
        data=df,
        y="y",
        x=names,
        selection="fixed",
        penalty=0.2,
        lambda_path=[1.0, 0.2, 0.01],
        tolerance=1e-10,
    )
    state = path.extra["penalized_state"]
    z, yc, _, scale, _ = transformed(df[names].to_numpy(), df.y.to_numpy(), True, True)
    for lam, beta in zip(state["lambda_path"], state["coefficient_path"], strict=True):
        np.testing.assert_allclose(beta, active_oracle(z, yc, lam, 1) / scale, atol=2e-8)
    plugin = oe.lasso(
        data=df, y="y", x=names, selection="plugin", tolerance=1e-10, plugin_iterations=50
    )
    state = plugin.extra["penalized_state"]
    reference = active_oracle(z, yc, state["selected_penalty"], 1, state["loadings"])
    np.testing.assert_allclose(state["standardized_coefficients"], reference, atol=2e-8)
    last = np.sqrt(np.mean(z**2 * (yc - z @ reference)[:, None] ** 2, axis=0))
    np.testing.assert_allclose(state["loadings"], last, rtol=1.1e-4)
    assert state["loadings_converged"] is True


@pytest.mark.parametrize("covariance", ["HC0", "HC1"])
def test_dml_independent_nested_ridge_cv_and_estimated_nuisance_influence_covariance(covariance):
    df, names = sample(n=120, p=4)
    result = oe.dmlplr(
        data=df,
        y="y",
        treatment="d",
        x=names,
        nuisance="ridge",
        selection="cv",
        lambda_path=[0.8, 0.2, 0.02],
        folds=3,
        seed=103,
        covariance=covariance,
    )
    saved = ResultBundle.model_validate_json(result.model_dump_json())
    x, y, d = df[names].to_numpy(), df.y.to_numpy(), df.d.to_numpy()
    lhat, mhat = np.empty(len(y)), np.empty(len(d))
    coverage = np.zeros(len(y), dtype=int)
    for record in saved.extra["fold_records"]:
        train, test = np.array(record["train_positions"]), np.array(record["test_positions"])
        assert not set(train) & set(test)
        assert set(train) | set(test) == set(range(len(y)))
        coverage[test] += 1
        for response, target, key in [(y, lhat, "outcome_model"), (d, mhat, "treatment_model")]:
            state = record[key]
            assignment = np.array(state["fold_assignments"])
            cv_scores = np.zeros(3)
            for fold in range(3):
                inner_train, inner_test = train[assignment != fold], train[assignment == fold]
                z, yc, mean, scale, ym = transformed(
                    x[inner_train], response[inner_train], True, True
                )
                for j, penalty in enumerate(state["lambda_path"]):
                    beta = active_oracle(z, yc, penalty, 0)
                    predicted = (x[inner_test] - mean) / scale @ beta + ym
                    cv_scores[j] += ((response[inner_test] - predicted) ** 2).sum()
            np.testing.assert_allclose(state["cv_mse"], cv_scores / len(train), atol=1e-10)
            assert state["selected_index"] == int(cv_scores.argmin())
            z, yc, mean, scale, ym = transformed(x[train], response[train], True, True)
            b = active_oracle(z, yc, state["selected_penalty"], 0)
            target[test] = (x[test] - mean) / scale @ b + ym
    assert (coverage == 1).all()
    np.testing.assert_allclose(saved.extra["nuisance_outcome_predictions"], lhat, atol=1e-10)
    np.testing.assert_allclose(saved.extra["nuisance_treatment_predictions"], mhat, atol=1e-10)
    v, ry = d - mhat, y - lhat
    theta = v @ ry / (v @ v)
    score = v * (ry - theta * v)
    influence = score / np.mean(v**2)
    correction = len(y) / (len(y) - 1) if covariance == "HC1" else 1
    variance = np.mean(influence**2) / len(y) * correction
    np.testing.assert_allclose(saved.extra["score"], score, atol=1e-10)
    np.testing.assert_allclose(saved.extra["influence"], influence, atol=1e-10)
    assert saved.coefficients[0].estimate == pytest.approx(theta, abs=1e-11)
    assert saved.covariance_matrix[0][0] == pytest.approx(variance, rel=1e-11)
    assert abs(theta - 1.6) < 0.35 and len(saved.coefficients) == 1
    assert saved.extra["cross_fitted"] is True
    assert "Only the treatment effect" in str(saved.to_latex())


def test_cross_fit_outcome_perturbation_does_not_retrain_its_own_test_fold():
    df, names = sample(n=80, p=4)
    options = dict(
        data=df,
        y="y",
        treatment="d",
        x=names,
        nuisance="ridge",
        selection="fixed",
        penalty=0.2,
        folds=4,
    )
    result = oe.dmlplr(**options)
    test = result.extra["fold_records"][0]["test_positions"]
    changed = df.copy()
    changed.loc[test, "y"] += 1000
    other = oe.dmlplr(**{**options, "data": changed})
    assert (
        result.extra["fold_records"][0]["outcome_model"]
        == other.extra["fold_records"][0]["outcome_model"]
    )
    np.testing.assert_allclose(
        np.array(result.extra["nuisance_outcome_predictions"])[test],
        np.array(other.extra["nuisance_outcome_predictions"])[test],
    )


@pytest.mark.parametrize("name", ["postdouble", "partiallingout"])
def test_in_sample_orthogonal_inference_against_numpy_and_union(name):
    df, names = sample(n=160, p=4)
    result = getattr(oe, name)(
        data=df,
        y="y",
        treatment="d",
        x=names,
        selection="plugin",
        plugin_iterations=50,
        covariance="HC1",
    )
    y, d = df.y.to_numpy(), df.d.to_numpy()
    if name == "postdouble":
        record = result.extra["fold_records"][0]
        assert set(record["selected_union"]) == set(record["selected_outcome"]) | set(
            record["selected_treatment"]
        )
        c = np.column_stack((np.ones(len(y)), df[result.extra["selected_controls"]].to_numpy()))
        projected = c @ np.linalg.lstsq(c, np.column_stack((y, d)), rcond=None)[0]
        ry, v = y - projected[:, 0], d - projected[:, 1]
        correction = len(y) / (len(y) - np.linalg.matrix_rank(c) - 1)
    else:
        ry = y - np.array(result.extra["nuisance_outcome_predictions"])
        v = d - np.array(result.extra["nuisance_treatment_predictions"])
        correction = len(y) / (len(y) - 1)
    theta = v @ ry / (v @ v)
    influence = v * (ry - theta * v) / np.mean(v**2)
    assert result.coefficients[0].estimate == pytest.approx(theta, abs=1e-10)
    assert result.covariance_matrix[0][0] == pytest.approx(
        np.mean(influence**2) / len(y) * correction, rel=1e-10
    )
    assert result.extra["cross_fitted"] is False


@pytest.mark.parametrize("name,degree", [("kernelreg", 0), ("localreg", 1)])
@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov", "uniform", "triangular"])
def test_general_multivariate_kernel_estimates_independent_weighted_oracle(name, degree, kernel):
    rng = np.random.default_rng(812)
    x = rng.uniform(-1, 1, size=(60, 2))
    y = 2 + 0.8 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * x[:, 0] ** 2
    df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})
    query = np.array([[0, 0], [x[:, 0].min(), x[:, 1].min()], [0.7, 0.8]])
    bandwidth = np.array([1.8, 1.5])
    result = getattr(oe, name)(
        data=df,
        y="y",
        x=["a", "b"],
        selection="fixed",
        bandwidth=bandwidth.tolist(),
        query=query.tolist(),
        kernel=kernel,
    )
    reference = []
    for q in query:
        u = (x - q) / bandwidth
        if kernel == "gaussian":
            w = np.exp(-0.5 * np.sum(u**2, axis=1))
        elif kernel == "epanechnikov":
            w = np.maximum(1 - u**2, 0).prod(1)
        elif kernel == "triangular":
            w = np.maximum(1 - abs(u), 0).prod(1)
        else:
            w = (abs(u) <= 1).prod(1)
        if degree == 0:
            estimate = w @ y / w.sum()
        else:
            c = np.column_stack((np.ones(len(x)), u))
            estimate = np.linalg.lstsq(c * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0][0]
        reference.append(estimate)
    np.testing.assert_allclose(result.extra["query_estimates"], reference, atol=2e-12)
    assert result.extra["query_diagnostics"][1]["boundary"] is True
    saved = ResultBundle.model_validate_json(result.model_dump_json())
    np.testing.assert_allclose(
        oe.regularized_predict(saved, pd.DataFrame(query, columns=["a", "b"])),
        reference,
        atol=2e-12,
    )
    assert "Conditional mean" in str(oe.regularized_table(saved).to_latex())


def test_local_linear_exact_boundary_reproduction_and_loo_bandwidth_cv():
    x = np.linspace(0, 1, 40)
    df = pd.DataFrame({"x": x, "y": 2 + 3 * x})
    query = [[0.0], [1.0]]
    local = oe.localreg(
        data=df,
        y="y",
        x=["x"],
        selection="fixed",
        bandwidth=0.15,
        query=query,
        kernel="epanechnikov",
    )
    kernel = oe.kernelreg(
        data=df,
        y="y",
        x=["x"],
        selection="fixed",
        bandwidth=0.15,
        query=query,
        kernel="epanechnikov",
    )
    np.testing.assert_allclose(local.extra["query_estimates"], [2, 5], atol=1e-12)
    assert abs(kernel.extra["query_estimates"][0] - 2) > 0.05
    result = oe.kernelreg(
        data=df, y="y", x=["x"], selection="cv", bandwidth_path=[0.1, 0.3, 0.7], query=query
    )
    expected = []
    for h in [0.1, 0.3, 0.7]:
        w = np.exp(-0.5 * ((x[:, None] - x[None, :]) / h) ** 2)
        np.fill_diagonal(w, 0)
        expected.append(np.mean((df.y - w @ df.y.to_numpy() / w.sum(1)) ** 2))
    np.testing.assert_allclose(
        result.extra["smoother_state"]["selection"]["cv_mse"], expected, atol=1e-12
    )
    assert result.extra["smoother_state"]["selection"]["selected_index"] == int(np.argmin(expected))


def test_direct_modelspec_all_new_methods_have_result_contracts():
    df, names = sample(n=80, p=4)
    for name in [
        "ridge",
        "lasso",
        "elasticnet",
        "kernelreg",
        "localreg",
        "postdouble",
        "partiallingout",
        "dmlplr",
    ]:
        inference = name in {"postdouble", "partiallingout", "dmlplr"}
        local = name in {"kernelreg", "localreg"}
        options = (
            {"selection": "fixed", "bandwidth": 8}
            if local
            else {"selection": "fixed", "penalty": 0.1, "folds": 2}
        )
        spec = ModelSpec(
            estimator=name,
            outcome="y",
            predictors=names,
            columns={"treatment": "d"} if inference else {},
            options=options,
        )
        result = oe.fit(spec, data=df)
        assert isinstance(result, ResultBundle) and result.nobs == len(df)
        assert result.provenance["precision"] == "float64"
        assert result.provenance["stata_parity_validated"] is False
        json.loads(result.model_dump_json())
