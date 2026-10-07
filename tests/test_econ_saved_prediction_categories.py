"""Independent saved categorical probabilities/effects and full covariance oracles."""

from decimal import Decimal, localcontext
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle


@pytest.fixture(scope="module", params=["ologit", "oprobit", "mlogit"])
def fitted(request):
    rng = np.random.default_rng(41759)
    n = 720
    frame = pd.DataFrame(
        dict(
            x=rng.normal(size=n),
            z=rng.normal(size=n),
            region=rng.choice(["North", "South", "West"], n),
            w=rng.integers(1, 4, n),
            offset=rng.normal(scale=0.15, size=n),
        )
    )
    frame.region = pd.Categorical(frame.region, categories=["North", "South", "West"])
    eta = 0.6 * frame.x - 0.25 * frame.z + 0.4 * (frame.region == "South") + frame.offset
    labels = ["low", "middle", "high", "top"]
    if request.param == "mlogit":
        indexes = np.column_stack(
            [
                0.3 + 0.5 * frame.x,
                np.zeros(n),
                -0.4 * frame.x + 0.2 * frame.z,
                0.2 * frame.x - 0.1 * frame.z,
            ]
        )
        probability = special.softmax(indexes, axis=1)
        codes = (rng.random(n)[:, None] > probability.cumsum(axis=1)).sum(1)
    else:
        error = rng.logistic(size=n) if request.param == "ologit" else rng.normal(size=n)
        codes = np.digitize(eta + error, [-0.6, 0.3, 1.2])
    frame["y"] = pd.Categorical(
        np.array(labels)[codes], categories=labels, ordered=request.param != "mlogit"
    )
    kwargs = dict(
        data=frame,
        y="y",
        x=["x", "z", "region"],
        categorical=["region"],
        weights="w",
        weight_type="fweight",
        covariance="robust",
        missing="drop",
    )
    kwargs.update(base="middle") if request.param == "mlogit" else kwargs.update(offset="offset")
    model = getattr(oe, request.param)(**kwargs)
    return model, frame


def encoded(model, data):
    terms = [item.term for item in model.coefficients]
    matrix = np.zeros((len(data), len(terms)))
    for i, coefficient in enumerate(model.coefficients):
        local = coefficient.term
        if model.spec.estimator == "mlogit":
            local = local[len(coefficient.equation) + 1 :]
        if local.startswith("/cut"):
            continue
        if local == "Intercept":
            matrix[:, i] = 1
        elif local in model.spec.predictors:
            matrix[:, i] = data[local].to_numpy(dtype=float)
        else:
            name, level = local[:-1].split("[", 1)
            matrix[:, i] = (data[name] == level).to_numpy(dtype=float)
    offset = np.zeros(len(data)) if model.spec.estimator == "mlogit" else data.offset.to_numpy()
    return matrix, offset


def oracle(model, beta, matrix, offset, *, kind="response", variable="x"):
    labels = model.extra["categories"]
    if model.spec.estimator == "mlogit":
        eta, slopes = np.zeros((len(matrix), len(labels))), np.zeros(len(labels))
        for j, label in enumerate(labels):
            for i, coefficient in enumerate(model.coefficients):
                if coefficient.equation == label:
                    eta[:, j] += matrix[:, i] * beta[i]
                    if coefficient.term == f"{label}:{variable}":
                        slopes[j] = beta[i]
        if kind == "xb":
            return eta
        probability = special.softmax(eta, axis=1)
        return (
            probability
            if kind == "response"
            else probability * (slopes - (probability @ slopes)[:, None])
        )
    eta = matrix @ beta + offset
    if kind == "xb":
        return eta[:, None]
    cuts = np.array(
        [
            beta[[c.term for c in model.coefficients].index(f"/cut{i}")]
            for i in range(1, len(labels))
        ]
    )
    boundaries = np.column_stack(
        [np.full(len(matrix), -np.inf), cuts[None, :] - eta[:, None], np.full(len(matrix), np.inf)]
    )
    cdf = (
        special.expit(boundaries) if model.spec.estimator == "ologit" else special.ndtr(boundaries)
    )
    if kind == "response":
        return np.diff(cdf, axis=1)
    density = cdf * (1 - cdf) if model.spec.estimator == "ologit" else stats.norm.pdf(boundaries)
    slope = beta[[c.term for c in model.coefficients].index(variable)]
    return (density[:, :-1] - density[:, 1:]) * slope


def jacobian(function, beta):
    columns = []
    for i in range(len(beta)):
        step = 1e-5 * max(1.0, abs(beta[i]))
        hi, lo = beta.copy(), beta.copy()
        hi[i] += step
        lo[i] -= step
        columns.append((function(hi) - function(lo)) / (2 * step))
    return np.stack(columns, axis=-1)


def beta_of(model):
    return np.array([item.estimate for item in model.coefficients])


def arrays(output, prefix):
    return output[[c for c in output if c.startswith(prefix + "[")]].to_numpy()


@pytest.mark.parametrize("kind", ["response", "xb", "derivative", "stdp"])
def test_predictions_full_delta_oracle_without_outcomes(fitted, kind):
    model, frame = fitted
    new = frame.iloc[:8].drop(columns="y")
    matrix, offset = encoded(model, new)
    beta = beta_of(model)

    def evaluate(b):
        return oracle(model, b, matrix, offset, kind="xb" if kind == "stdp" else kind)

    values = evaluate(beta)
    jac = jacobian(evaluate, beta)
    se = np.sqrt(np.einsum("njk,kl,njl->nj", jac, model.covariance_matrix, jac))
    output = oe.predict(
        model,
        new,
        kind=kind,
        interval=None if kind == "stdp" else "mean",
        term="x" if kind == "derivative" else None,
    )
    prefix = "dydx[x]" if kind == "derivative" else kind
    common = model.spec.estimator != "mlogit" and kind in {"xb", "stdp"}
    actual = output[[prefix]].to_numpy() if common else arrays(output, prefix)
    np.testing.assert_allclose(actual, se if kind == "stdp" else values, rtol=2e-8, atol=2e-10)
    if kind != "stdp":
        actual_se = output[["std_error"]].to_numpy() if common else arrays(output, "std_error")
        np.testing.assert_allclose(actual_se, se, rtol=2e-8, atol=2e-10)
        low = output[["ci_low"]].to_numpy() if common else arrays(output, "ci_low")
        high = output[["ci_high"]].to_numpy() if common else arrays(output, "ci_high")
        np.testing.assert_allclose(low, values - stats.norm.ppf(0.975) * se, atol=1e-9)
        np.testing.assert_allclose(high, values + stats.norm.ppf(0.975) * se, atol=1e-9)
    assert output.attrs["outcome_labels"] == model.extra["categories"]


@pytest.mark.parametrize("method", ["ame", "mem"])
@pytest.mark.parametrize("variable", ["x", "region"])
def test_effects_full_parameter_gradient_and_covariance(fitted, method, variable):
    model, frame = fitted
    new = frame.iloc[:41].drop(columns="y").copy()
    beta = beta_of(model)
    weight = new.w.to_numpy(dtype=float)
    weight /= weight.sum()

    def evaluate(b):
        if variable == "x":
            matrix, offset = encoded(model, new)
            if method == "mem":
                matrix, offset = (weight @ matrix)[None], np.array([weight @ offset])
            values = oracle(model, b, matrix, offset, kind="derivative")
        else:
            low, high = new.copy(), new.copy()
            low.region, high.region = "North", "South"
            lm, lo = encoded(model, low)
            hm, ho = encoded(model, high)
            if method == "mem":
                lm, lo = (weight @ lm)[None], np.array([weight @ lo])
                hm, ho = (weight @ hm)[None], np.array([weight @ ho])
            values = oracle(model, b, hm, ho) - oracle(model, b, lm, lo)
        return values[0] if method == "mem" else weight @ values

    expected, gradients = evaluate(beta), jacobian(evaluate, beta)
    output = oe.margins(model, variable, data=new, method=method)
    selected = output[output.variable == ("x" if variable == "x" else "region[South]")]
    np.testing.assert_allclose(selected.estimate, expected, atol=2e-12)
    recorded = np.array(output.attrs["delta_gradients"])[selected.index]
    np.testing.assert_allclose(recorded, gradients, rtol=2e-8, atol=2e-10)
    se = np.sqrt(np.einsum("jk,kl,jl->j", gradients, model.covariance_matrix, gradients))
    np.testing.assert_allclose(selected.std_error, se, rtol=2e-8, atol=2e-10)
    assert list(selected.outcome) == model.extra["categories"]
    assert abs(selected.estimate.sum()) < 2e-15
    np.testing.assert_allclose(recorded.sum(axis=0), 0, atol=2e-14)


def test_json_and_permuted_equations_are_identical(fitted):
    model, frame = fitted
    saved = ResultBundle.model_validate_json(model.model_dump_json())
    order = np.random.default_rng(411).permutation(len(saved.coefficients))
    saved.coefficients = [saved.coefficients[i] for i in order]
    covariance = np.asarray(saved.covariance_matrix)
    saved.covariance_matrix = covariance[np.ix_(order, order)].tolist()
    for kind in ["response", "xb", "derivative"]:
        kwargs = (
            dict(kind=kind, interval="mean", term="x")
            if kind == "derivative"
            else dict(kind=kind, interval="mean")
        )
        pd.testing.assert_frame_equal(
            oe.predict(saved, frame.head(), **kwargs), oe.predict(model, frame.head(), **kwargs)
        )
    a, b = oe.margins(saved, "x", data=frame.head()), oe.margins(model, "x", data=frame.head())
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("outcome", ["low", "middle", "top"])
def test_real_outcome_selection(fitted, outcome):
    model, frame = fitted
    all_rows = oe.predict(model, frame.head(), interval="mean")
    one = oe.predict(model, frame.head(), outcome=outcome, interval="mean")
    for field in ["response", "std_error", "ci_low", "ci_high"]:
        np.testing.assert_allclose(
            one[field], all_rows[f"{field}[{json.dumps(outcome)}]"], rtol=5e-15, atol=0
        )
    margin = oe.margins(model, "x", data=frame.head(), outcome=outcome)
    assert list(margin.outcome) == [outcome]


def test_missing_duplicate_positions_empty_and_inference_mode(fitted):
    model, frame = fitted
    new = frame.head().drop(columns="y").copy()
    new.index = [7, 7, 2, 2, 7]
    new.iloc[1, new.columns.get_loc("x")] = np.nan
    with torch.inference_mode():
        output = oe.predict(model, new, interval="mean")
    assert output.attrs["missing_row_positions"] == [1]
    assert output.iloc[1].isna().all() and list(output.index) == [7, 7, 2, 2, 7]
    assert len(oe.margins(model, "x", data=new)) == 4
    new.x = np.nan
    output = oe.predict(model, new, interval="mean")
    assert output.isna().all().all()
    assert oe.predict(model, new.iloc[:0], interval="mean").shape[0] == 0
    with pytest.raises(AnalysisError, match="at least one"):
        oe.margins(model, "x", data=new)


def test_weight_scale_zero_rows_categories_and_grids(fitted):
    model, frame = fitted
    new = frame.head(10).copy()
    expected = oe.margins(model, ["x", "region"], data=new, at={"x": [-1.0, 1.0]})
    new.w = new.w.astype(float) * 1e300
    # Original fweights still have exact integer values at this finite scale.
    scaled = oe.margins(model, ["x", "region"], data=new, at={"x": [-1.0, 1.0]})
    np.testing.assert_allclose(expected.estimate, scaled.estimate, atol=1e-14)
    new.region = new.region.astype(object)
    new.loc[new.index[-1], ["region", "x", "w"]] = ["unseen", 1e300, 0.0]
    assert len(oe.margins(model, "x", data=new)) == 4
    with pytest.raises(AnalysisError, match="category"):
        oe.predict(model, new)


@pytest.mark.parametrize(
    "corruption",
    [
        "labels_duplicate",
        "labels_missing",
        "categories_count",
        "counts",
        "terms",
        "equation",
        "covariance",
        "threshold_or_base",
    ],
)
def test_corrupted_saved_state_rejected(fitted, corruption):
    original, frame = fitted
    model = original.model_copy(deep=True)
    if corruption == "labels_duplicate":
        model.extra["categories"][1] = model.extra["categories"][0]
    elif corruption == "labels_missing":
        del model.extra["categories"]
    elif corruption == "categories_count":
        model.metrics["n_categories"] = 3
    elif corruption == "counts":
        model.extra["category_counts"][0] = -1
    elif corruption == "terms":
        model.coefficients[0].term = "unknown"
    elif corruption == "equation":
        model.coefficients[0].equation = "unknown"
    elif corruption == "covariance":
        model.covariance_matrix[0][0] = -1.0
    elif model.spec.estimator == "mlogit":
        model.extra["base"] = "unknown"
    else:
        model.extra["cutpoints"][0] += 0.1
    with pytest.raises(AnalysisError):
        oe.predict(model, frame.head())


@pytest.mark.parametrize("kind", ["latent", "conditional", "residual", "class"])
def test_unsupported_tasks_and_invalid_outcome_fail_explicitly(fitted, kind):
    model, frame = fitted
    with pytest.raises(AnalysisError):
        oe.predict(model, frame.head(), kind=kind)
    with pytest.raises(AnalysisError):
        oe.predict(model, frame.head(), outcome="absent")
    with pytest.raises(AnalysisError):
        oe.predict(model, frame.head(), interval="prediction")


def test_native_no_refit_no_external_estimator_runtime(fitted, monkeypatch):
    model, frame = fitted

    def refused(*args, **kwargs):
        raise AssertionError("prediction must not refit")

    monkeypatch.setattr(oe, "fit", refused)
    import openecon.analysis as analysis

    monkeypatch.setattr(analysis, "fit", refused)
    assert oe.predict(model, frame.head(), interval="mean").notna().all().all()
    assert len(oe.margins(model, "x", data=frame.head())) == 4


def test_multinomial_base_index_exact_zero(fitted):
    model, frame = fitted
    if model.spec.estimator != "mlogit":
        with pytest.raises(AnalysisError):
            oe.predict(model, frame.head(), kind="xb", outcome="low")
        return
    for kind in ["xb", "stdp"]:
        result = oe.predict(model, frame.head(), kind=kind, outcome="middle")
        np.testing.assert_array_equal(result[kind], 0.0)
    result = oe.margins(model, "x", data=frame.head(), kind="xb", outcome="middle")
    assert result.estimate.iloc[0] == 0 and result.std_error.iloc[0] == 0
    assert result.statistic.iloc[0] is None and result.p_value.iloc[0] is None


@pytest.mark.parametrize("estimator", ["ologit", "oprobit"])
def test_narrow_symmetric_probabilities_and_effects_high_precision(estimator):
    rng = np.random.default_rng(1283)
    n = 600
    frame = pd.DataFrame({"x": rng.normal(size=n)})
    error = rng.logistic(size=n) if estimator == "ologit" else rng.normal(size=n)
    frame["y"] = np.digitize(0.4 * frame.x + error, [-0.5, 0.5, 1.5])
    model = getattr(oe, estimator)(data=frame, y="y", x=["x"])
    for c in model.coefficients:
        if c.term == "x":
            c.estimate = 1.0
        if c.term.startswith("/cut"):
            c.estimate = {"/cut1": 0.0, "/cut2": 1e-15, "/cut3": 2.0}[c.term]
    model.extra["cutpoints"] = [0.0, 1e-15, 2.0]
    new = pd.DataFrame({"x": [0.0]})
    value = oe.predict(model, new, outcome=1, interval="mean")
    effect = oe.predict(model, new, kind="derivative", term="x", outcome=1, interval="mean")
    with localcontext() as ctx:
        ctx.prec = 100
        width = Decimal.from_float(1e-15)
        if estimator == "ologit":

            def F(z):
                return 1 / (1 + (-z).exp())

            probability = F(width) - Decimal(".5")
            derivative = Decimal(".25") - F(width) * (1 - F(width))
        else:
            pi = Decimal(
                "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
            )
            phi0 = 1 / (2 * pi).sqrt()
            probability = phi0 * sum(
                ((-Decimal(".5")) ** k)
                * width ** (2 * k + 1)
                / (Decimal(math.factorial(k)) * (2 * k + 1))
                for k in range(15)
            )
            derivative = phi0 * (1 - (-width * width / 2).exp())
    np.testing.assert_allclose(value.response, float(probability), rtol=3e-15, atol=0)
    np.testing.assert_allclose(effect["dydx[x]"], float(derivative), rtol=3e-15, atol=0)
    margin = oe.margins(model, "x", data=new, outcome=1)
    np.testing.assert_allclose(margin.estimate, float(derivative), rtol=3e-15, atol=0)
    assert np.isfinite(margin.attrs["delta_gradients"]).all()


@pytest.mark.parametrize(
    "labels,base,invalid",
    [([1, 2, 3], 2, True), (["1", "2", "3"], "2", 1), ([False, True], True, 1)],
)
def test_label_identity_and_nonfirst_base(labels, base, invalid):
    rng = np.random.default_rng(8205)
    n = 500
    frame = pd.DataFrame({"x": rng.normal(size=n), "y": rng.choice(labels, size=n)})
    model = oe.mlogit(data=frame, y="y", x=["x"], base=base)
    result = oe.predict(model, frame.head(), outcome=base)
    assert result.attrs["base_outcome"] == base
    with pytest.raises(AnalysisError):
        oe.predict(model, frame.head(), outcome=invalid)
    with pytest.raises(AnalysisError):
        oe.margins(model, "x", data=frame.head(), outcome=invalid)
    malformed = model.model_copy(deep=True)
    malformed.extra["base"] = invalid
    with pytest.raises(AnalysisError):
        oe.predict(malformed, frame.head())


@pytest.mark.parametrize("estimator", ["ologit", "oprobit"])
def test_ordered_binary_and_omitted_constant_only_design(estimator):
    rng = np.random.default_rng(21)
    n = 350
    frame = pd.DataFrame(
        {"constant": np.ones(n), "x": rng.normal(size=n), "y": rng.integers(0, 2, n)}
    )
    model = getattr(oe, estimator)(data=frame, y="y", x=["constant"])
    assert "constant" in model.provenance["omitted_terms"]
    result = oe.predict(model, frame.head(), interval="mean")
    np.testing.assert_allclose(arrays(result, "response").sum(1), 1, atol=1e-15)
    output = oe.margins(model, "constant", data=frame.head())
    np.testing.assert_array_equal(output.estimate, 0.0)
    np.testing.assert_array_equal(output.std_error, 0.0)


@pytest.mark.parametrize("estimator", ["ologit", "oprobit", "mlogit"])
def test_missing_raise_probability_sum_and_probability_gradient_sum(estimator):
    rng = np.random.default_rng(558)
    n = 500
    frame = pd.DataFrame({"x": rng.normal(size=n), "y": rng.integers(0, 3, n)})
    model = getattr(oe, estimator)(data=frame, y="y", x=["x"])
    new = pd.DataFrame({"x": [-12.0, -1.0, 0.0, 2.0, 12.0]})
    result = oe.predict(model, new, interval="mean")
    np.testing.assert_allclose(arrays(result, "response").sum(1), 1, atol=2e-15)
    effects = oe.predict(model, new, kind="derivative", term="x", interval="mean")
    np.testing.assert_allclose(arrays(effects, "dydx[x]").sum(1), 0, atol=2e-15)
    new.iloc[2, 0] = np.nan
    with pytest.raises(AnalysisError):
        oe.predict(model, new)
    with pytest.raises(AnalysisError):
        oe.margins(model, "x", data=new)


def test_full_cross_equation_covariance_matters(fitted):
    model, frame = fitted
    output = oe.margins(model, "x", data=frame.head())
    gradients = np.array(output.attrs["delta_gradients"])
    covariance = np.asarray(model.covariance_matrix)
    diagonal = np.diag(np.diag(covariance))
    actual = np.einsum("jk,kl,jl->j", gradients, covariance, gradients)
    incorrect = np.einsum("jk,kl,jl->j", gradients, diagonal, gradients)
    assert np.max(np.abs(actual - incorrect)) > 1e-7
    if model.spec.estimator != "mlogit":
        cut_indices = [i for i, c in enumerate(model.coefficients) if c.term.startswith("/cut")]
        assert np.any(np.abs(gradients[:, cut_indices]) > 1e-4)


@pytest.mark.parametrize("estimator", ["ologit", "oprobit"])
def test_narrow_extreme_tail_probabilities_against_adaptive_quadrature(estimator):
    from scipy.integrate import quad
    from openecon.econometrics.postest.ordinal_prediction import CategoryResponse
    from openecon.econometrics.postest.prediction import _Design

    choice = CategoryResponse(estimator, (0, 1, 2), ((0,),), ("x",), (1, 2), None)
    for edge in [-35.0, -8.0, 8.0, 35.0]:
        upper = edge + 0.01
        beta = torch.tensor([0.0, edge, upper], dtype=torch.float64)
        design = _Design(
            torch.zeros((1, 3), dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
        )
        density = (
            (lambda z: special.expit(z) * special.expit(-z))
            if estimator == "ologit"
            else stats.norm.pdf
        )
        expected = quad(density, edge, upper, epsabs=1e-300, epsrel=2e-13)[0]
        actual = float(choice.probabilities(design, beta)[0, 1])
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0)


def test_selected_prediction_guards_full_jacobian_allocation(fitted, monkeypatch):
    import openecon.econometrics.postest.ordinal_prediction as category

    model, frame = fitted
    size = 8 * len(frame.head()) * len(model.coefficients)
    monkeypatch.setattr(category, "_MAX_DESIGN_BYTES", size * 2)

    def unexpected(*args, **kwargs):
        raise AssertionError("The full gradient must not be allocated after its guard fails.")

    monkeypatch.setattr(category.CategoryResponse, "jacobian", unexpected)
    with pytest.raises(AnalysisError) as caught:
        oe.predict(model, frame.head(), outcome="middle", interval="mean")
    assert caught.value.code == "prediction_memory_limit"


def test_margins_guards_full_jacobian_allocation(fitted, monkeypatch):
    import openecon.econometrics.postest.prediction as prediction
    from openecon.econometrics.postest.ordinal_prediction import CategoryResponse

    model, frame = fitted
    size = 8 * len(frame.head()) * len(model.coefficients)
    monkeypatch.setattr(prediction, "_MAX_DESIGN_BYTES", size * 2)

    def unexpected(*args, **kwargs):
        raise AssertionError("The full gradient must not be allocated after its guard fails.")

    monkeypatch.setattr(CategoryResponse, "jacobian", unexpected)
    with pytest.raises(AnalysisError) as caught:
        oe.margins(model, "x", data=frame.head(), outcome="middle")
    assert caught.value.code == "prediction_memory_limit"


@pytest.mark.parametrize("method", ["ame", "mem"])
def test_latent_index_margin_has_slope_gradient_not_design_gradient(fitted, method):
    model, frame = fitted
    result = oe.margins(model, "x", data=frame.head(7), kind="xb", method=method)
    terms = [c.term for c in model.coefficients]
    gradients = np.zeros((len(result), len(terms)))
    for j, row in result.iterrows():
        if model.spec.estimator == "mlogit":
            if row.outcome == model.extra["base"]:
                continue
            term = f"{row.outcome}:x"
        else:
            term = "x"
        position = terms.index(term)
        gradients[j, position] = 1
        np.testing.assert_allclose(row.estimate, model.coefficients[position].estimate)
        np.testing.assert_allclose(
            row.std_error, math.sqrt(model.covariance_matrix[position][position])
        )
    np.testing.assert_array_equal(result.attrs["delta_gradients"], gradients)


@pytest.mark.parametrize("estimator", ["ologit", "oprobit", "mlogit"])
def test_dominant_probability_dummy_change_retains_small_complement(estimator):
    rng = np.random.default_rng(72891)
    frame = pd.DataFrame(
        {
            "g": pd.Categorical(rng.choice(["A", "B"], 400), categories=["A", "B"]),
            "y": rng.integers(0, 3, 400),
        }
    )
    model = getattr(oe, estimator)(
        data=frame,
        y="y",
        x=["g"],
        categorical=["g"],
        **({"base": 1} if estimator == "mlogit" else {}),
    )
    for coefficient in model.coefficients:
        if estimator == "mlogit":
            coefficient.estimate = (
                50.0
                if coefficient.term == "0:Intercept"
                else 1.0
                if coefficient.term == "0:g[B]"
                else 0.0
            )
        else:
            coefficient.estimate = (
                1.0
                if coefficient.term == "g[B]"
                else 35.0
                if coefficient.term == "/cut1" and estimator == "oprobit"
                else 50.0
                if coefficient.term == "/cut1"
                else 37.0
                if estimator == "oprobit"
                else 52.0
            )
    if estimator != "mlogit":
        model.extra["cutpoints"] = [
            c.estimate for c in model.coefficients if c.term.startswith("/cut")
        ]
    model.covariance_matrix = np.eye(len(model.coefficients)).tolist()
    result = oe.margins(model, "g", data=frame.head(), outcome=0)
    if estimator == "mlogit":
        q = math.exp(-50)
        expected = 2 * q * -math.expm1(-1) / ((1 + 2 * q) * (1 + 2 * q / math.e))
    elif estimator == "ologit":
        q = math.exp(-50)
        expected = q * -math.expm1(1) / ((1 + q) * (1 + q * math.e))
    else:
        expected = special.ndtr(-35) - special.ndtr(-34)
    np.testing.assert_allclose(result.estimate, expected, rtol=3e-13, atol=0)
    assert result.std_error.iloc[0] > 0
