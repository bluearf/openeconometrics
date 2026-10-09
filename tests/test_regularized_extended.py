"""Independent weighted algebra, literal replication and saved-state contracts."""

import copy
from decimal import Decimal, localcontext
import itertools

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.regularized.extended import _digest
from openecon.models import ResultBundle


def fixture(n=48):
    rng = np.random.default_rng(173)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    category = np.array(["red", "blue", "green"] * (n // 3), dtype=object)
    y = 1.7 + 1.2 * x - 0.8 * z + (category == "blue") * 0.6 + rng.normal(0, 0.12, n)
    return pd.DataFrame({"x": x, "z": z, "g": category, "y": y, "w": 1 + np.arange(n) % 3})


def oracle_design(df, intercept, standardize):
    x = np.column_stack([df.x, df.z, df.g.eq("blue"), df.g.eq("green")])
    if not intercept:
        x = np.column_stack([df.x, df.z, df.g.eq("red"), df.g.eq("blue"), df.g.eq("green")])
    weights = df.w.to_numpy(dtype=float) / df.w.sum()
    center = weights @ x if intercept else np.zeros(x.shape[1])
    scale = np.sqrt(weights @ (x - center) ** 2) if standardize else np.ones(x.shape[1])
    scale[scale == 0] = 1
    yc = float(weights @ df.y) if intercept else 0
    return (x - center) / scale, df.y.to_numpy() - yc, weights, center, scale, yc


def gaussian_oracle(z, y, weights, lam, ratio, factors):
    """Enumerate all active signs and solve their normal equations, not CD."""
    gram, score = z.T @ (weights[:, None] * z), z.T @ (weights * y)
    best = None
    domains = [[-1, 1] if factor == 0 else [-1, 0, 1] for factor in factors]
    for signs in itertools.product(*domains):
        active = np.flatnonzero(signs)
        beta = np.zeros(z.shape[1])
        block = gram[np.ix_(active, active)] + lam * (1 - ratio) * np.diag(factors[active])
        try:
            beta[active] = np.linalg.solve(
                block, score[active] - lam * ratio * factors[active] * np.array(signs)[active]
            )
        except np.linalg.LinAlgError:
            continue
        if any(factors[j] > 0 and beta[j] * signs[j] <= 0 for j in active):
            continue
        gradient = gram @ beta - score + lam * (1 - ratio) * factors * beta
        inactive = np.flatnonzero(np.array(signs) == 0)
        if np.any(np.abs(gradient[inactive]) > lam * ratio * factors[inactive] + 1e-9):
            continue
        objective = 0.5 * weights @ (y - z @ beta) ** 2 + lam * (
            ratio * (factors * abs(beta)).sum() + 0.5 * (1 - ratio) * (factors * beta**2).sum()
        )
        if best is None or objective < best[0]:
            best = objective, beta
    assert best is not None
    return best


@pytest.mark.parametrize("name,ratio", [("ridge", 0.0), ("lasso", 1.0), ("elasticnet", 0.35)])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("standardize", [True, False])
def test_weighted_gaussian_independent_sign_oracle(name, ratio, intercept, standardize):
    df = fixture()
    result = getattr(oe, name)(
        data=df,
        y="y",
        x=["x", "z", "g"],
        categorical=["g"],
        weights="w",
        selection="fixed",
        penalty=0.17,
        penalty_factors={"x": 0.4, "g": 1.8},
        forced_controls=["z"],
        intercept=intercept,
        standardize=standardize,
        **({"l1_ratio": ratio} if name == "elasticnet" else {}),
    )
    z, y, weights, center, scale, yc = oracle_design(df, intercept, standardize)
    factors = np.array([0.4, 0] + [1.8] * (z.shape[1] - 2))
    objective, beta = gaussian_oracle(z, y, weights, 0.17, ratio, factors)
    point = result.extra["regularized_extended_state"]["paths"][0]
    np.testing.assert_allclose(point["standardized_coefficients"], beta, rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(point["objective"], objective, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        oe.regularized_predict(result, df), z @ beta + yc, rtol=1e-7, atol=1e-7
    )
    assert result.coefficients == [] and result.covariance_matrix == []
    assert result.inference["available"] is False


def pls_oracle(z, y, weights, components, forced):
    """Explicit weighted covariance/deflation, avoiding runtime whitening."""
    controls = z[:, forced]
    candidate = z[:, ~forced]
    control_y = (
        np.linalg.solve(controls.T @ (weights[:, None] * controls), controls.T @ (weights * y))
        if controls.shape[1]
        else np.empty(0)
    )
    control_x = (
        np.linalg.solve(
            controls.T @ (weights[:, None] * controls), controls.T @ (weights[:, None] * candidate)
        )
        if controls.shape[1]
        else np.empty((0, candidate.shape[1]))
    )
    e, f = candidate - controls @ control_x, y - controls @ control_y
    ws, ps, qs = [], [], []
    for _ in range(components):
        direction = e.T @ (weights * f)
        w = direction / np.linalg.norm(direction)
        score = e @ w
        ss = weights @ score**2
        p, q = e.T @ (weights * score) / ss, (weights * f) @ score / ss
        ws.append(w)
        ps.append(p)
        qs.append(q)
        e -= score[:, None] * p
        f -= score * q
    ws, ps, qs = np.column_stack(ws), np.column_stack(ps), np.array(qs)
    latent = ws @ np.linalg.solve(ps.T @ ws, qs)
    beta = np.zeros(z.shape[1])
    beta[~forced] = latent
    beta[forced] = control_y - control_x @ latent
    return beta


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("components", [1, 2, 3])
@pytest.mark.parametrize("forced", [[], ["z"]])
def test_weighted_pls_covariance_oracle(intercept, components, forced):
    df = fixture()
    result = oe.pls(
        data=df,
        y="y",
        x=["x", "z", "g"],
        categorical=["g"],
        weights="w",
        selection="fixed",
        components=components,
        forced_controls=forced,
        intercept=intercept,
    )
    z, y, weights, *_ = oracle_design(df, intercept, True)
    mask = np.zeros(z.shape[1], dtype=bool)
    mask[1] = bool(forced)
    beta = pls_oracle(z.copy(), y.copy(), weights, components, mask)
    np.testing.assert_allclose(
        result.extra["penalized_state"]["standardized_coefficients"], beta, rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("name", ["ridge", "lasso", "elasticnet", "pls"])
def test_frequency_literal_replication_and_scale_invariant_empirical_weights(name):
    df = fixture()
    df["blue"], df["green"] = df.g.eq("blue").astype(float), df.g.eq("green").astype(float)
    kwargs = {"components": 2} if name == "pls" else {"penalty": 0.13}
    fn = getattr(oe, name)
    weighted = fn(
        data=df,
        y="y",
        x=["x", "z", "blue", "green"],
        weights="w",
        weight_type="fweight",
        selection="fixed",
        **kwargs,
    )
    expanded = df.loc[df.index.repeat(df.w)].reset_index(drop=True)
    replicated = fn(
        data=expanded, y="y", x=["x", "z", "blue", "green"], selection="fixed", **kwargs
    )
    np.testing.assert_allclose(
        oe.regularized_predict(weighted, df),
        oe.regularized_predict(replicated, df),
        rtol=5e-8,
        atol=5e-8,
    )
    scaled = df.assign(w=df.w * 1e100)
    for kind in ["aweight", "pweight"]:
        result = fn(
            data=scaled,
            y="y",
            x=["x", "z", "blue", "green"],
            weights="w",
            weight_type=kind,
            selection="fixed",
            **kwargs,
        )
        np.testing.assert_allclose(
            oe.regularized_predict(result, df),
            oe.regularized_predict(weighted, df),
            rtol=1e-10,
            atol=1e-10,
        )


@pytest.mark.parametrize("name", ["ridge", "lasso", "elasticnet", "pls"])
def test_cv_train_only_preprocessing_paths_rng_and_complete_restore(name):
    df = fixture(60)
    kwargs = {"component_path": [1, 2]} if name == "pls" else {"lambda_path": [0.4, 0.12, 0.03]}
    fn = getattr(oe, name)
    before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    result = fn(
        data=df,
        y="y",
        x=["x", "z", "g"],
        categorical=["g"],
        weights="w",
        selection="cv",
        folds=3,
        seed=117,
        **kwargs,
    )
    state = result.extra["regularized_extended_state"]
    changed = df.copy()
    holdout = np.array(state["fold_assignments"]) == 0
    changed.loc[holdout, "y"] += 20
    changed.loc[holdout, "w"] *= 4
    second = fn(
        data=changed,
        y="y",
        x=["x", "z", "g"],
        categorical=["g"],
        weights="w",
        selection="cv",
        folds=3,
        seed=117,
        **kwargs,
    )
    assert state["cv"][0]["design"] == second.extra["regularized_extended_state"]["cv"][0]["design"]
    assert state["cv"][0]["paths"] == second.extra["regularized_extended_state"]["cv"][0]["paths"]
    assert all(np.array_equal(a, b) for a, b in zip(before, np.random.get_state(), strict=True))
    assert torch.equal(torch_before, torch.random.get_rng_state())
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored.extra == result.extra
    query = df.iloc[[5, 1, 5]].copy()
    query.index = [9, 9, 2]
    np.testing.assert_array_equal(
        oe.regularized_predict(restored, query), oe.regularized_predict(result, query)
    )
    assert oe.regularized_predict(restored, query).index.tolist() == [9, 9, 2]
    assert oe.regularized_table(restored).equals(oe.regularized_table(result))


@pytest.mark.parametrize("name", ["ridge", "lasso", "elasticnet", "pls"])
def test_missing_zero_weights_unseen_categories_and_resource_admission(name):
    df = fixture()
    df.loc[2, "y"] = np.nan
    df.loc[4, "w"] = 0
    kwargs = {"components": 2} if name == "pls" else {"penalty": 0.1}
    fn = getattr(oe, name)
    with pytest.raises(AnalysisError, match="missing"):
        fn(
            data=df,
            y="y",
            x=["x", "z", "g"],
            categorical=["g"],
            weights="w",
            selection="fixed",
            **kwargs,
        )
    result = fn(
        data=df,
        y="y",
        x=["x", "z", "g"],
        categorical=["g"],
        weights="w",
        selection="fixed",
        missing="drop",
        **kwargs,
    )
    assert result.sample_positions == [i for i in range(len(df)) if i not in {2, 4}]
    with pytest.raises(AnalysisError) as caught:
        oe.regularized_predict(result, df.assign(g="unknown"))
    assert caught.value.code == "unseen_category"
    with pytest.raises(AnalysisError) as caught:
        fn(
            data=df.dropna(),
            y="y",
            x=["x", "z", "g"],
            categorical=["g"],
            weights="w",
            selection="fixed",
            max_work=1,
            **kwargs,
        )
    assert caught.value.code == "work_limit"
    with pytest.raises(AnalysisError) as caught:
        oe.regularized_predict(result, df.dropna(), max_work=1)
    assert caught.value.code == "work_limit"


@pytest.mark.parametrize(
    "change", ["digest", "spec", "factor", "weights", "point", "alias", "positions"]
)
def test_saved_state_tampering_even_rehashed(change):
    result = oe.elasticnet(
        data=fixture(),
        y="y",
        x=["x", "z", "g"],
        categorical=["g"],
        weights="w",
        selection="fixed",
        penalty=0.1,
    )
    result = copy.deepcopy(result)
    state = result.extra["regularized_extended_state"]
    if change == "digest":
        state["digest"] = "bad"
    elif change == "spec":
        state["spec"]["intercept"] = False
    elif change == "factor":
        state["design"]["effective_factors"][0] = 9
    elif change == "weights":
        state["normalized_weights"][0] *= 2
    elif change == "point":
        state["paths"][0]["coefficients"][0] += 1
    elif change == "alias":
        result.extra["penalized_state"]["constant"] += 1
    elif change == "positions":
        state["physical_positions"] = list(reversed(state["physical_positions"]))
    if change != "digest":
        state["digest"] = _digest(state)
    for call in [
        lambda: oe.regularized_predict(result, fixture()),
        lambda: oe.regularized_table(result),
    ]:
        with pytest.raises(AnalysisError) as caught:
            call()
        assert caught.value.code == "invalid_prediction_state"


@pytest.mark.parametrize("name", ["ridge", "pls"])
@pytest.mark.parametrize("partition", ["full", "fold"])
def test_no_intercept_rehashed_response_shift_is_refused(name, partition):
    options = {"lambda_path": [0.2, 0.1]} if name == "ridge" else {"component_path": [1, 2]}
    result = getattr(oe, name)(
        data=fixture(), y="y", x=["x", "z"], weights="w", intercept=False,
        selection="cv", folds=3, **options,
    )
    state = result.extra["regularized_extended_state"]
    points = state["paths"] if partition == "full" else state["cv"][0]["paths"]
    for point in points:
        if point is not None:
            point["outcome_center"] += 1
            point["constant"] += 1
    if partition == "full":
        result.extra["constant"] += 1
        result.extra["penalized_state"]["outcome_center"] += 1
        result.extra["penalized_state"]["constant"] += 1
    state["digest"] = _digest(state)
    for call in (
        lambda: oe.regularized_predict(result, fixture()),
        lambda: oe.regularized_table(result),
    ):
        with pytest.raises(AnalysisError) as caught:
            call()
        assert caught.value.code == "invalid_prediction_state"


@pytest.mark.parametrize("name", ["ridge", "pls"])
def test_table_validation_uses_cpu_and_preserves_callers_default_device(name):
    options = {"penalty": 0.1} if name == "ridge" else {"components": 2}
    result = getattr(oe, name)(
        data=fixture(), y="y", x=["x", "z"], weights="w", selection="fixed", **options,
    )
    expected = oe.regularized_table(result)
    previous = torch.empty(0).device
    with torch.device("meta"):
        actual = oe.regularized_table(result)
        assert torch.empty(0).device.type == "meta"
    assert torch.empty(0).device == previous
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.attrs == expected.attrs


@pytest.mark.parametrize("name", ["ridge", "lasso", "elasticnet"])
def test_low_weight_finite_outlier_preserves_representable_training_loss(name):
    df = fixture()
    df["w"] = 1.0
    df.loc[0, "w"] = 1e-40
    df.loc[0, "y"] = 1e160
    result = getattr(oe, name)(
        data=df, y="y", x=["x", "z"], weights="w", selection="fixed", penalty=0.1,
    )
    predictions = oe.regularized_predict(result, df)
    with localcontext() as context:
        context.prec = 80
        weights = [Decimal.from_float(value) for value in df.w]
        total = sum(weights)
        expected = sum(
            weight * (Decimal.from_float(observed) - Decimal.from_float(predicted)) ** 2 / total
            for weight, observed, predicted in zip(weights, df.y, predictions, strict=True)
        )
    assert np.isfinite(result.metrics["training_mse"])
    np.testing.assert_allclose(result.metrics["training_mse"], float(expected), rtol=2e-13)


def test_large_response_offset_training_loss_preserves_finite_residual_cancellation():
    df = fixture()
    base = 1e160
    df["y"] = base + np.spacing(base) * np.arange(len(df))
    result = oe.ridge(data=df, y="y", x=["x", "z"], weights="w", selection="fixed", penalty=0.1)
    predictions = oe.regularized_predict(result, df)
    with localcontext() as context:
        context.prec = 80
        weights = [Decimal.from_float(float(value)) for value in df.w]
        total = sum(weights)
        expected = sum(
            weight * (Decimal.from_float(observed) - Decimal.from_float(predicted)) ** 2 / total
            for weight, observed, predicted in zip(weights, df.y, predictions, strict=True)
        )
    np.testing.assert_allclose(result.metrics["training_mse"], float(expected), rtol=2e-13)


@pytest.mark.parametrize("name", ["ridge", "lasso", "elasticnet", "pls"])
def test_nonreplayable_finite_predictor_centering_is_explicitly_refused(name):
    base = 1e160
    step = np.spacing(base)
    x = base + step * np.arange(48)
    df = pd.DataFrame({"x": x, "y": (x - base) / step, "w": np.ones(48)})
    options = {"components": 1} if name == "pls" else {"penalty": 0.0}
    with pytest.raises(AnalysisError) as caught:
        getattr(oe, name)(data=df, y="y", x=["x"], weights="w", selection="fixed", **options)
    assert caught.value.code == "numerical_failure"
    assert "rescale predictors" in str(caught.value)
    scaled = df.assign(x=(df.x - base) / step)
    result = getattr(oe, name)(data=scaled, y="y", x=["x"], weights="w", selection="fixed", **options)
    np.testing.assert_allclose(oe.regularized_predict(result, scaled), scaled.y, atol=1e-10)


@pytest.mark.parametrize("name", ["ridge", "pls"])
def test_rehashed_frequency_booleans_cannot_replace_original_counts(name):
    df = fixture().assign(w=1)
    options = {"penalty": 0.1} if name == "ridge" else {"components": 2}
    result = getattr(oe, name)(
        data=df, y="y", x=["x", "z"], weights="w", weight_type="fweight",
        selection="fixed", **options,
    )
    state = result.extra["regularized_extended_state"]
    state["raw_weights"] = [True] * len(df)
    state["digest"] = _digest(state)
    for call in (lambda: oe.regularized_predict(result, df), lambda: oe.regularized_table(result)):
        with pytest.raises(AnalysisError) as caught:
            call()
        assert caught.value.code == "invalid_prediction_state"


def test_failed_pls_candidate_remains_ineligible_without_aborting_later_folds():
    from openecon.econometrics.regularized.kernels import folds

    rng = np.random.default_rng(173)
    x = rng.normal(size=48)
    z = x.copy()
    groups = folds(48, 3, 117).numpy()
    z[groups == 0] = rng.normal(size=int((groups == 0).sum()))
    y = 1.7 + 1.2 * x - 0.8 * z + rng.normal(0, 0.12, 48)
    df = pd.DataFrame({"x": x, "z": z, "y": y, "w": np.ones(48)})
    result = oe.pls(
        data=df, y="y", x=["x", "z"], weights="w", selection="cv",
        component_path=[1, 2], folds=3, seed=117,
    )
    state = result.extra["regularized_extended_state"]
    assert state["grid"][state["selected_index"]] == 1
    assert state["cv"][0]["paths"][1] is None
    assert state["cv"][1]["paths"][1] is not None
    assert state["cv_scores"][1] is None
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    np.testing.assert_allclose(oe.regularized_predict(restored, df), oe.regularized_predict(result, df))


def test_unseen_training_fold_and_forced_rank_fail_without_omitting_rows():
    df = fixture()
    df.loc[0, "g"] = "unique"
    with pytest.raises(AnalysisError) as caught:
        oe.ridge(
            data=df,
            y="y",
            x=["x", "g"],
            categorical=["g"],
            weights="w",
            selection="cv",
            lambda_path=[0.1],
            folds=3,
        )
    assert caught.value.code == "unseen_category"
    df = fixture().assign(z=lambda d: d.x)
    with pytest.raises(AnalysisError) as caught:
        oe.pls(
            data=df,
            y="y",
            x=["x", "z", "g"],
            categorical=["g"],
            weights="w",
            forced_controls=["x", "z"],
            selection="fixed",
            components=1,
        )
    assert caught.value.code == "unidentified_forced_controls"


def test_weighted_plugin_and_pls_penalties_are_explicitly_unsupported():
    with pytest.raises(AnalysisError) as caught:
        oe.lasso(data=fixture(), y="y", x=["x"], weights="w", selection="plugin")
    assert caught.value.code == "unsupported_selection"
    with pytest.raises(AnalysisError):
        oe.pls(
            data=fixture(),
            y="y",
            x=["x"],
            weights="w",
            selection="fixed",
            components=1,
            penalty_factors=[1.0],
        )


@pytest.mark.parametrize("name", ["ridge", "lasso", "elasticnet"])
def test_automatic_weighted_grid_does_not_observe_validation_labels(name):
    df = fixture(60)
    fn = getattr(oe, name)
    kwargs = dict(
        y="y",
        x=["x", "z", "g"],
        categorical=["g"],
        weights="w",
        selection="cv",
        n_lambdas=4,
        folds=3,
        seed=219,
        forced_controls=["z"],
    )
    original = fn(data=df, **kwargs)
    state = original.extra["regularized_extended_state"]
    changed = df.copy()
    changed.loc[np.array(state["fold_assignments"]) == 1, "y"] += 1000
    later = fn(data=changed, **kwargs).extra["regularized_extended_state"]
    assert state["cv"][1]["grid"] == later["cv"][1]["grid"]
    assert state["cv"][1]["paths"] == later["cv"][1]["paths"]
    assert state["fractions"] == later["fractions"]
    assert oe.regularized_predict(original, df).notna().all()


@pytest.mark.parametrize("change", ["loss", "score", "chosen", "fold", "grid"])
def test_rehashed_cv_semantic_tampering_is_refused(change):
    result = oe.elasticnet(
        data=fixture(),
        y="y",
        x=["x", "z", "g"],
        categorical=["g"],
        weights="w",
        selection="cv",
        lambda_path=[0.3, 0.1, 0.02],
        folds=3,
    )
    state = result.extra["regularized_extended_state"]
    if change == "loss":
        state["cv"][0]["validation_mse"][0] += 5
    elif change == "score":
        state["cv_scores"][0] += 5
    elif change == "chosen":
        state["selected_index"] = (state["selected_index"] + 1) % 3
    elif change == "fold":
        state["cv"][0]["training_positions"].reverse()
    else:
        state["cv"][0]["grid"][0] += 1
    state["digest"] = _digest(state)
    with pytest.raises(AnalysisError) as caught:
        oe.regularized_predict(result, fixture())
    assert caught.value.code == "invalid_prediction_state"


def test_typed_category_identity_and_numeric_name_factor_list():
    df = fixture()
    df["g"] = pd.Series([True, 1, "1"] * 16, dtype=object)
    result = oe.ridge(
        data=df,
        y="y",
        x=["x", "g"],
        categorical=["g"],
        weights="w",
        selection="fixed",
        penalty=0.1,
        penalty_factors=[0.2, 0.8],
    )
    design = result.extra["regularized_extended_state"]["design"]
    assert [level["type"] for level in design["levels"]["g"]] == ["bool", "int", "str"]
    assert design["effective_factors"] == [0.2, 0.8, 0.8]
    assert oe.regularized_predict(result, df).notna().all()


@pytest.mark.parametrize(
    "kind,weights",
    [
        ("fweight", [2**53 + 1] * 48),
        ("fweight", [1.5] * 48),
        ("aweight", [-1] * 48),
        ("pweight", [0] * 48),
    ],
)
def test_invalid_weight_domains_refuse(kind, weights):
    df = fixture().assign(w=weights)
    with pytest.raises(AnalysisError):
        oe.ridge(
            data=df, y="y", x=["x"], weights="w", weight_type=kind, selection="fixed", penalty=0.1
        )


def test_weighted_pls_constant_response_has_no_components():
    df = fixture().assign(y=5.0)
    with pytest.raises(AnalysisError) as caught:
        oe.pls(data=df, y="y", x=["x", "z"], weights="w", selection="fixed", components=1)
    assert caught.value.code == "rank_deficient"


@pytest.mark.parametrize("part", ["full", "fold"])
def test_rehashed_pls_loadings_must_reconstruct_saved_coefficients(part):
    result = oe.pls(
        data=fixture(),
        y="y",
        x=["x", "z", "g"],
        categorical=["g"],
        weights="w",
        selection="cv",
        component_path=[1, 2],
        folds=3,
        forced_controls=["z"],
    )
    state = result.extra["regularized_extended_state"]
    point = state["paths"][0] if part == "full" else state["cv"][0]["paths"][0]
    point["x_loadings"][0][0] += 1
    state["digest"] = _digest(state)
    with pytest.raises(AnalysisError) as caught:
        oe.regularized_predict(result, fixture())
    assert caught.value.code == "invalid_prediction_state"


@pytest.mark.parametrize("kind", ["explicit_path", "default_maximum"])
def test_coherent_pls_candidate_truncation_cannot_change_requested_grid(kind):
    tuning = {"component_path": [1, 2]} if kind == "explicit_path" else {"max_components": 2}
    result = oe.pls(
        data=fixture(), y="y", x=["x", "z"], weights="w", selection="cv", folds=3, **tuning
    )
    state = result.extra["regularized_extended_state"]
    for key in ("grid", "paths", "cv_scores"):
        state[key] = state[key][:1]
    for record in state["cv"]:
        for key in ("grid", "paths", "validation_mse", "failures"):
            record[key] = record[key][:1]
    state["selected_index"] = 0
    point = state["paths"][0]
    result.extra["penalized_state"] = {**point, "selected_penalty": None}
    result.extra["constant"] = point["constant"]
    result.extra["predictive_coefficients"] = point["coefficients"]
    state["digest"] = _digest(state)
    with pytest.raises(AnalysisError) as caught:
        oe.regularized_predict(result, fixture())
    assert caught.value.code == "invalid_prediction_state"
    assert "PLS requested component grid binding" in str(caught.value.__cause__)


def test_auto_fold_grid_cannot_break_its_saved_fraction_reference():
    result = oe.ridge(
        data=fixture(), y="y", x=["x", "z"], weights="w", selection="cv", n_lambdas=4, folds=3
    )
    state = result.extra["regularized_extended_state"]
    record = state["cv"][0]
    record["grid"][1] *= 100
    record["paths"][1]["penalty"] = record["grid"][1]
    state["digest"] = _digest(state)
    with pytest.raises(AnalysisError) as caught:
        oe.regularized_predict(result, fixture())
    assert caught.value.code == "invalid_prediction_state"
    assert "automatic fold grid fraction binding" in str(caught.value.__cause__)


def test_auto_grid_reference_factorization_runs_once_per_training_partition(monkeypatch):
    from openecon.econometrics.regularized import extended

    calls = []
    reference = extended._reference

    def tracked(z, *args):
        calls.append(len(z))
        return reference(z, *args)

    monkeypatch.setattr(extended, "_reference", tracked)
    result = oe.ridge(
        data=fixture(),
        y="y",
        x=["x", "z"],
        weights="w",
        selection="cv",
        n_lambdas=8,
        folds=3,
        forced_controls=["z"],
    )
    assert calls == [48, 32, 32, 32]
    assert len(oe.regularized_predict(result, fixture())) == 48


@pytest.mark.parametrize("maximum", [1, 1000])
def test_work_refusal_precedes_auto_reference_and_control_factorization(monkeypatch, maximum):
    from openecon.econometrics.regularized import extended

    def unexpected(*args, **kwargs):
        raise AssertionError("Reference/factorization ran before work admission")

    monkeypatch.setattr(extended, "_reference", unexpected)
    monkeypatch.setattr(torch.linalg, "svdvals", unexpected)
    monkeypatch.setattr(torch.linalg, "lstsq", unexpected)
    with pytest.raises(AnalysisError) as caught:
        oe.ridge(
            data=fixture(),
            y="y",
            x=["x", "z"],
            weights="w",
            selection="cv",
            n_lambdas=4,
            folds=3,
            forced_controls=["z"],
            max_work=maximum,
        )
    assert caught.value.code == "work_limit"


@pytest.mark.parametrize(
    "tamper", ["unknown_root", "unknown_point", "ragged_matrix", "known_byte_padding"]
)
def test_state_shape_and_actual_bytes_are_checked_before_digest_or_tensors(monkeypatch, tamper):
    from openecon.econometrics.regularized import extended
    from openecon.resources import use_workspace_budget

    result = oe.pls(
        data=fixture(),
        y="y",
        x=["x", "z"],
        weights="w",
        selection="fixed",
        components=1,
        forced_controls=["z"],
    )
    state = result.extra["regularized_extended_state"]
    if tamper == "unknown_root":
        state["ignored_padding"] = "x" * 2_000_000
    elif tamper == "unknown_point":
        state["paths"][0]["ignored_padding"] = "x" * 2_000_000
    elif tamper == "ragged_matrix":
        state["paths"][0]["control_x"][0].append(1.0)
    else:
        for key in ("weight_normalization", "cv_rule", "preprocessing_scope"):
            state[key] = "x" * 45_000
    state["digest"] = _digest(state)

    def unexpected(*args, **kwargs):
        raise AssertionError("Digest/tensor allocation preceded bounded preflight")

    monkeypatch.setattr(extended, "_digest", unexpected)
    monkeypatch.setattr(torch, "tensor", unexpected)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as caught:
        oe.regularized_predict(result, fixture())
    assert caught.value.code == (
        "workspace_limit" if tamper == "known_byte_padding" else "invalid_prediction_state"
    )


def test_canonical_byte_preflight_counts_utf8_and_json_escapes_exactly():
    import json
    from openecon.econometrics.regularized.extended import _json_preflight

    value = {
        "unicode": "測量🌐",
        "escaped": '\x00\n\\"',
        "scalars": [True, False, None, -10, 1.234e-10],
    }
    expected = len(
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
    )
    assert _json_preflight(value, max_work=10000) == expected
