"""Independent two-equation algebra for saved heterogeneous normal probabilities."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(58147)
    n = 1000
    frame = pd.DataFrame({name: rng.normal(size=n) for name in ["x", "u", "z"]})
    frame["g"] = pd.Categorical(rng.choice(["A", "B", "C"], n), categories=["A", "B", "C"])
    frame["one"] = 1.0
    frame["w"] = rng.integers(1, 4, n)
    frame["zcopy"] = frame.z
    mean = 0.2 + 0.7 * frame.x - 0.2 * frame.u + 0.2 * (frame.g == "B")
    log_scale = 0.15 * frame.z + 0.1 * frame.x + 0.12 * (frame.g == "B")
    frame["y"] = (mean + np.exp(log_scale) * rng.normal(size=n) > 0).astype(int)
    model = oe.hetprobit(
        data=frame,
        y="y",
        x=["x", "u", "g", "one"],
        het=["z", "x", "g", "one", "zcopy"],
        categorical=["g"],
        weights="w",
        weight_type="fweight",
        missing="drop",
        covariance="robust",
    )
    return ResultBundle.model_validate_json(model.model_dump_json()), frame


def beta_of(model):
    return np.array([item.estimate for item in model.coefficients])


def encoded(model, data):
    """Build the two reported-coordinate designs without implementation helpers."""
    x = np.zeros((len(data), len(model.coefficients)))
    z = np.zeros_like(x)
    for i, item in enumerate(model.coefficients):
        variance = item.equation == "lnsigma"
        term = item.term[len("lnsigma:") :] if variance else item.term
        if term == "Intercept":
            value = np.ones(len(data))
        elif "[" in term:
            name, label = term[:-1].split("[", 1)
            value = (data[name] == label).to_numpy(dtype=float)
        else:
            value = data[term].to_numpy(dtype=float)
        (z if variance else x)[:, i] = value
    return x, z


def oracle(model, beta, x, z, kind, variable=None):
    mean, scale = x @ beta, np.exp(z @ beta)
    a = mean / scale
    density = np.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    if kind in {"response", "xb", "sigma"}:
        value = {"response": special.ndtr(a), "xb": mean, "sigma": scale}[kind]
        gradient = {
            "response": density[:, None] * (x / scale[:, None] - a[:, None] * z),
            "xb": x,
            "sigma": scale[:, None] * z,
        }[kind]
        return value, gradient
    terms = [item.term for item in model.coefficients]
    eb, eg = np.zeros(len(beta)), np.zeros(len(beta))
    if variable in terms:
        eb[terms.index(variable)] = 1
    if f"lnsigma:{variable}" in terms:
        eg[terms.index(f"lnsigma:{variable}")] = 1
    b, g = eb @ beta, eg @ beta
    if kind == "derivative_xb":
        return np.full(len(x), b), np.broadcast_to(eb, x.shape)
    if kind == "derivative_sigma":
        return scale * g, scale[:, None] * (eg + g * z)
    h = b - mean * g
    value = density * h / scale
    gradient = (
        density[:, None]
        / scale[:, None]
        * (
            eb
            - g * x
            - (a * h / scale)[:, None] * x
            + ((a * a - 1) * h)[:, None] * z
            - mean[:, None] * eg
        )
    )
    return value, gradient


def finite_gradient(function, beta):
    values = []
    for i in range(len(beta)):
        step = 1e-5 * max(1, abs(beta[i]))
        hi, lo = beta.copy(), beta.copy()
        hi[i] += step
        lo[i] -= step
        values.append((function(hi) - function(lo)) / (2 * step))
    return np.stack(values, axis=-1)


@pytest.mark.parametrize("kind", ["response", "xb", "sigma", "stdp", "derivative"])
@pytest.mark.parametrize("variable", ["x", "u", "z"])
def test_predictions_full_covariance_without_outcomes(fitted, kind, variable):
    model, frame = fitted
    new = frame.iloc[:13].drop(columns="y")
    x, z = encoded(model, new)
    requested = "xb" if kind == "stdp" else kind
    value, gradient = oracle(model, beta_of(model), x, z, requested, variable)
    se = np.sqrt(np.einsum("nk,kl,nl->n", gradient, model.covariance_matrix, gradient))
    output = oe.predict(
        model,
        new,
        kind=kind,
        term=variable if kind == "derivative" else None,
        interval=None if kind == "stdp" else "mean",
    )
    label = f"dydx[{variable}]" if kind == "derivative" else kind
    np.testing.assert_allclose(
        output[label], se if kind == "stdp" else value, rtol=2e-13, atol=1e-14
    )
    if kind != "stdp":
        np.testing.assert_allclose(output.std_error, se, rtol=2e-13, atol=1e-14)
        np.testing.assert_allclose(output.ci_low, value - stats.norm.ppf(0.975) * se, atol=1e-14)
        np.testing.assert_allclose(output.ci_high, value + stats.norm.ppf(0.975) * se, atol=1e-14)
    assert output.attrs["covariance_source"] == "full saved mean/variance parameter covariance"


@pytest.mark.parametrize("kind", ["response", "xb", "sigma"])
@pytest.mark.parametrize("method", ["ame", "mem"])
@pytest.mark.parametrize("variable", ["x", "u", "z", "g", "one", "zcopy"])
def test_weighted_effects_complete_analytic_and_finite_difference_oracles(
    fitted, kind, method, variable
):
    model, frame = fitted
    new = frame.iloc[:47].drop(columns="y").copy()
    weight = new.w.to_numpy(dtype=float)
    weight /= weight.sum()
    beta = beta_of(model)

    def evaluation(b):
        if variable == "g":
            low, high = new.copy(), new.copy()
            low.g, high.g = "A", "B"
            lx, lz = encoded(model, low)
            hx, hz = encoded(model, high)
            if method == "mem":
                lx, lz, hx, hz = [(weight @ a)[None] for a in [lx, lz, hx, hz]]
            low_value, low_gradient = oracle(model, b, lx, lz, kind)
            high_value, high_gradient = oracle(model, b, hx, hz, kind)
            value, gradient = high_value - low_value, high_gradient - low_gradient
        else:
            x, z = encoded(model, new)
            if method == "mem":
                x, z = (weight @ x)[None], (weight @ z)[None]
            value, gradient = oracle(
                model,
                b,
                x,
                z,
                "derivative" if kind == "response" else f"derivative_{kind}",
                variable,
            )
        return (value[0], gradient[0]) if method == "mem" else (weight @ value, weight @ gradient)

    expected, gradient = evaluation(beta)
    numeric = finite_gradient(lambda b: evaluation(b)[0], beta)
    np.testing.assert_allclose(gradient, numeric, rtol=3e-8, atol=2e-11)
    output = oe.margins(model, variable, data=new, kind=kind, method=method)
    index = 0
    assert output.iloc[index].variable == ("g[B]" if variable == "g" else variable)
    np.testing.assert_allclose(output.iloc[index].estimate, expected, rtol=1e-12, atol=2e-15)
    np.testing.assert_allclose(
        output.attrs["delta_gradients"][index], gradient, rtol=2e-12, atol=2e-15
    )
    se = math.sqrt(gradient @ np.asarray(model.covariance_matrix) @ gradient)
    np.testing.assert_allclose(output.iloc[index].std_error, se, rtol=2e-12, atol=2e-15)
    if variable in {"one", "zcopy"} or (kind == "xb" and variable == "z"):
        assert expected == se == 0
        assert pd.isna(output.iloc[index].statistic)


def test_json_permutation_and_centres_are_not_applied_twice(fitted):
    model, frame = fitted
    new = frame.iloc[:6].drop(columns="y")
    expected = oe.predict(model, new, interval="mean")
    altered = ResultBundle.model_validate_json(model.model_dump_json())
    altered.extra["variance_regressor_means"] = [100.0] * len(altered.extra["variance_terms"])
    order = np.random.default_rng(718).permutation(len(altered.coefficients))
    altered.coefficients = [altered.coefficients[i] for i in order]
    altered.covariance_matrix = np.asarray(altered.covariance_matrix)[np.ix_(order, order)].tolist()
    pd.testing.assert_frame_equal(oe.predict(altered, new, interval="mean"), expected)
    a, b = (
        oe.margins(altered, ["x", "z", "g"], data=new),
        oe.margins(model, ["x", "z", "g"], data=new),
    )
    pd.testing.assert_frame_equal(a, b)


def test_missing_alignment_all_missing_and_inference_mode_cache(fitted):
    model, frame = fitted
    new = frame.iloc[:5].drop(columns="y").copy()
    new.index = [7, 7, 9, 4, 4]
    new.loc[new.index == 9, "z"] = np.nan
    with torch.inference_mode():
        output = oe.predict(model, new, kind="derivative", term="z", interval="mean")
    assert list(output.index) == list(new.index)
    assert output.iloc[2].isna().all()
    assert output.attrs["missing_row_positions"] == [2]
    result = oe.margins(model, "x", data=new)
    assert result.attrs["evaluation_rows"] == 4
    new.x = np.nan
    assert oe.predict(model, new, interval="mean").isna().all().all()
    with pytest.raises(AnalysisError, match="complete evaluation"):
        oe.margins(model, "x", data=new)
    assert oe.predict(model, new.iloc[:0], kind="stdp").empty


def test_all_role_columns_required_but_outcome_and_cluster_are_not(fitted):
    model, frame = fitted
    new = frame.iloc[:3].drop(columns="y")
    expected = oe.predict(model, new)
    for name in ["z", "zcopy", "one"]:
        with pytest.raises(AnalysisError) as exc:
            oe.predict(model, new.drop(columns=name))
        assert exc.value.code == "missing_columns"
    new["y"] = [np.nan] * len(new)
    pd.testing.assert_frame_equal(oe.predict(model, new), expected)


def configured(model, values, covariance=None):
    result = model.model_copy(deep=True)
    for item in result.coefficients:
        item.estimate = values.get(item.term, 0.0)
    result.covariance_matrix = (
        np.eye(len(result.coefficients)) if covariance is None else covariance
    ).tolist()
    return result


def test_zero_variance_equation_is_probit_points_but_not_identical_uncertainty(fitted):
    model, frame = fitted
    new = frame.iloc[:5].drop(columns="y")
    values = {c.term: c.estimate for c in model.coefficients if c.equation == "y"}
    ordinary = configured(model, values)
    x, _ = encoded(ordinary, new)
    a = x @ beta_of(ordinary)
    density = stats.norm.pdf(a)
    np.testing.assert_allclose(oe.predict(ordinary, new).response, special.ndtr(a), atol=2e-16)
    np.testing.assert_allclose(
        oe.predict(ordinary, new, kind="derivative", term="x")["dydx[x]"],
        density * values["x"],
        atol=2e-16,
    )
    assert np.all(oe.predict(ordinary, new, kind="sigma").sigma == 1)
    fixed = ordinary.model_copy(deep=True)
    v = np.eye(len(model.coefficients))
    for i, c in enumerate(model.coefficients):
        if c.equation == "lnsigma":
            v[i, i] = 0
    fixed.covariance_matrix = v.tolist()
    full = oe.predict(ordinary, new, interval="mean")
    simple = oe.predict(fixed, new, interval="mean")
    np.testing.assert_allclose(simple.std_error, density * np.linalg.norm(x, axis=1), atol=1e-15)
    assert np.all(full.std_error > simple.std_error)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.extra.update(variance_function="variance = exp(z'g)"),
        lambda m: m.extra.update(variance_terms=[]),
        lambda m: m.extra.update(variance_regressor_means=[0]),
        lambda m: m.extra.update(variance_regressor_means=[True] * 4),
        lambda m: m.extra.update(variance_regressor_means=[0.0, False, 0.0, 0.0]),
        lambda m: m.extra.update(zero_outcomes=0),
        lambda m: m.extra.update(nonzero_outcomes=False),
        lambda m: m.extra.update(constrained_terms=["x"]),
        lambda m: m.spec.columns.update(offset="z"),
        lambda m: m.spec.columns.update(het="z"),
        lambda m: m.spec.columns.update(het=["z", "z"]),
        lambda m: m.spec.options.update(formula="y~x"),
        lambda m: setattr(m.coefficients[-1], "equation", "y"),
        lambda m: m.provenance.update(omitted_terms=["absent"]),
        lambda m: m.provenance["categorical_encoding"]["g"].update(reference="B"),
        lambda m: m.provenance["categorical_encoding"]["g"].update(coding="effects"),
        lambda m: setattr(m.spec, "categorical", ["g", "absent"]),
        lambda m: m.inference.update(use_t=True, distribution="t", df_inference=30),
    ],
)
def test_malformed_saved_equations_are_structured_errors(fitted, change):
    model, frame = fitted
    invalid = model.model_copy(deep=True)
    change(invalid)
    with pytest.raises(AnalysisError) as exc:
        oe.predict(invalid, frame.iloc[:1].drop(columns="y"))
    assert exc.value.code == "invalid_result"


@pytest.mark.parametrize("kind", ["residual", "scores", "class", "latent", "conditional"])
def test_unsupported_outputs_are_not_silent_identity(fitted, kind):
    model, frame = fitted
    with pytest.raises(AnalysisError) as exc:
        oe.predict(model, frame.iloc[:1], kind=kind)
    assert exc.value.code == "unsupported_prediction_kind"


def test_aliases_outcome_and_interval_guards(fitted):
    model, frame = fitted
    new = frame.iloc[:2]
    expected = oe.predict(model, new)
    for alias in ["pr", "probability", "mean"]:
        pd.testing.assert_frame_equal(oe.predict(model, new, kind=alias), expected)
    with pytest.raises(AnalysisError) as exc:
        oe.predict(model, new, outcome=1)
    assert exc.value.code == "unsupported_prediction_outcome"
    with pytest.raises(AnalysisError) as exc:
        oe.predict(model, new, interval="prediction")
    assert exc.value.code == "unsupported_prediction_interval"


@pytest.mark.parametrize("method", ["predict", "ame", "mem"])
def test_gradient_budget_guard_before_allocation(fitted, monkeypatch, method):
    from openecon.econometrics.postest import heteroskedastic_prediction as native

    model, frame = fitted
    monkeypatch.setattr(native, "_MAX_DESIGN_BYTES", 1)

    def forbidden(*args, **kwargs):
        pytest.fail("A parameter Jacobian was allocated after its budget failed")

    monkeypatch.setattr(native.HeteroskedasticProbit, "jacobian", forbidden)
    with pytest.raises(AnalysisError) as exc:
        if method == "predict":
            oe.predict(model, frame.iloc[:2], interval="mean")
        else:
            oe.margins(model, "x", data=frame.iloc[:2], method=method)
    assert exc.value.code == "prediction_memory_limit"


@pytest.mark.parametrize("log_scale", [-800.0, 800.0])
def test_sigma_unrepresentable_is_refused(fitted, log_scale):
    model, frame = fitted
    altered = configured(model, {"lnsigma:z": log_scale})
    new = frame.iloc[:1].copy()
    new.z = 1.0
    with pytest.raises(AnalysisError) as exc:
        oe.predict(altered, new, kind="sigma")
    assert exc.value.code == "prediction_precision"


def test_nonzero_standardized_underflow_and_overflow_are_refused(fitted):
    model, frame = fitted
    new = frame.iloc[:1].copy()
    new.z = 1.0
    for log_scale in [-800.0, 800.0]:
        altered = configured(model, {"Intercept": 1.0, "lnsigma:z": log_scale})
        with pytest.raises(AnalysisError) as exc:
            oe.predict(altered, new, interval="mean")
        assert exc.value.code == "prediction_precision"


def test_no_refit_or_distribution_delegate(fitted, monkeypatch):
    from openecon.econometrics.discrete import hetprobit as fitter

    model, frame = fitted

    def forbidden(*args, **kwargs):
        pytest.fail("Stored prediction delegated to a fitter or external distribution")

    monkeypatch.setattr(fitter, "fit_hetprobit", forbidden)
    monkeypatch.setattr(special, "ndtr", forbidden)
    monkeypatch.setattr(stats.norm, "cdf", forbidden)
    oe.predict(model, frame.iloc[:5], interval="mean")
    oe.margins(model, ["x", "z", "g"], data=frame.iloc[:5])


def test_no_intercept_and_variance_only_categories_real_fit():
    rng = np.random.default_rng(17074)
    n = 650
    data = pd.DataFrame(dict(x=rng.normal(size=n), z=rng.normal(size=n)))
    data["g"] = pd.Categorical(rng.choice(["A", "B", "C"], n), categories=["A", "B", "C"])
    data["y"] = (
        0.7 * data.x + np.exp(0.15 * data.z + 0.3 * (data.g == "B")) * rng.normal(size=n) > 0
    ).astype(int)
    model = oe.hetprobit(
        data=data, y="y", x=["x"], het=["z", "g"], categorical=["g"], intercept=False
    )
    saved = ResultBundle.model_validate_json(model.model_dump_json())
    assert "Intercept" not in [c.term for c in saved.coefficients]
    new = data.iloc[:11].drop(columns="y")
    x, z = encoded(saved, new)
    value, jac = oracle(saved, beta_of(saved), x, z, "response")
    result = oe.predict(saved, new, interval="mean")
    np.testing.assert_allclose(result.response, value, atol=2e-15)
    np.testing.assert_allclose(
        result.std_error,
        np.sqrt(np.einsum("nk,kl,nl->n", jac, saved.covariance_matrix, jac)),
        atol=2e-15,
    )
    high, low = new.copy(), new.copy()
    high.g, low.g = "B", "A"
    hx, hz = encoded(saved, high)
    lx, lz = encoded(saved, low)
    expected = (
        oracle(saved, beta_of(saved), hx, hz, "response")[0]
        - oracle(saved, beta_of(saved), lx, lz, "response")[0]
    ).mean()
    output = oe.margins(saved, "g", data=new)
    np.testing.assert_allclose(output.estimate.iloc[0], expected, atol=2e-15)


def test_grid_and_zero_weight_rows_apply_both_equations(fitted):
    model, frame = fitted
    new = frame.iloc[:11].drop(columns="y").copy()
    new.g = new.g.astype(object)
    new.iloc[0, new.columns.get_loc("w")] = 0
    new.iloc[0, new.columns.get_loc("g")] = "unfitted-but-zero-weight"
    output = oe.margins(model, "x", data=new, at={"x": [-0.5, 0.75]}, method="mem")
    retained = new.iloc[1:]
    weight = retained.w.to_numpy(dtype=float)
    weight /= weight.sum()
    for row in output.itertuples():
        setting = retained.copy()
        setting.x = output.loc[row.Index, "at[x]"]
        x, z = encoded(model, setting)
        expected = oracle(
            model, beta_of(model), (weight @ x)[None], (weight @ z)[None], "derivative", "x"
        )[0][0]
        np.testing.assert_allclose(row.estimate, expected, atol=2e-15)
    assert output.attrs["zero_weight_rows_excluded"] == 1
    new.w = new.w.astype(float) * 1e290
    scaled = oe.margins(model, "x", data=new, at={"x": [-0.5, 0.75]}, method="mem")
    pd.testing.assert_frame_equal(scaled, output)
