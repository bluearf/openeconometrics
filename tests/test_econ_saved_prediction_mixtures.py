"""Saved mixture means against independent probability and numerical oracles.

SciPy is used only here, never by prediction or fitting. Every fixture is actually
fitted, JSON-restored, and evaluated without outcomes or saved training rows.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.econometrics.postest.inference import _parameters
from openecon.econometrics.postest.mixture_prediction import CountMixture, configure
from openecon.econometrics.postest.prediction import _data, _encode, _model
from openecon.models import ResultBundle

CASES = ["zip_logit", "zip_probit", "zinb", "hurdle_poisson", "hurdle_nb",
         "tpoisson", "tpoisson_varying", "tnbreg_mean", "tnbreg_constant"]


@pytest.fixture(scope="module", params=CASES)
def fitted(request):
    rng = np.random.default_rng(202610060)
    n = 2200
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n),
                          "expo": rng.uniform(.8, 1.3, n), "w": rng.integers(1, 4, n)})
    frame["g"] = pd.Categorical(rng.choice(["A", "B", "C"], n), categories=["A", "B", "C"])
    frame["xcopy"] = frame.x
    frame["ll"] = rng.integers(1, 4, n)
    mean = np.exp(.8 + .3 * frame.x + .15 * (frame.g == "B")) * frame.expo
    count = rng.poisson(mean)
    nb = rng.negative_binomial(2, 2 / (2 + mean))
    active = rng.random(n) < special.expit(.2 + .25 * frame.x - .4 * frame.z + .1 * (frame.g == "C"))
    case = request.param
    common = dict(y="y", x=["x", "g", "xcopy"], categorical=["g"],
                  exposure="expo", weights="w", weight_type="fweight",
                  covariance="robust", missing="drop")
    if case.startswith("zip") or case == "zinb":
        frame["y"] = np.where(active, 0, nb if case == "zinb" else count)
        result = getattr(oe, "zinb" if case == "zinb" else "zip")(
            data=frame, inflate=["x", "z", "g"],
            inflate_link="probit" if case.endswith("probit") or case == "zinb" else "logit", **common)
    elif case.startswith("hurdle"):
        frame["y"] = np.where(active, np.maximum(nb if case == "hurdle_nb" else count, 1), 0)
        result = oe.hurdle(data=frame, select_x=["x", "z", "g"],
                          dist="nbinomial" if case == "hurdle_nb" else "poisson",
                          zero_link="cloglog" if case == "hurdle_nb" else "logit", **common)
    else:
        frame["y"] = nb if case.startswith("tnbreg") else count
        cutoff = frame.ll if case.endswith("varying") else 2
        frame = frame.loc[frame.y > cutoff].copy()
        kwargs = dict(ll="ll" if case.endswith("varying") else 2)
        if case.startswith("tnbreg"):
            kwargs["dispersion"] = case.rsplit("_", 1)[1]
        result = getattr(oe, "tnbreg" if case.startswith("tnbreg") else "tpoisson")(
            data=frame, **kwargs, **common)
    return ResultBundle.model_validate_json(result.model_dump_json()), frame, case


def matrices(model, frame):
    """Independent reported coefficient coding, not the adapter's encoder."""
    count = np.zeros((len(frame), len(model.coefficients)))
    second = np.zeros_like(count)
    for i, coefficient in enumerate(model.coefficients):
        term = coefficient.term
        if term.startswith("/"):
            continue
        auxiliary = term.startswith(("inflate:", "select:"))
        local = term.split(":", 1)[1] if auxiliary else term
        if local == "Intercept":
            value = np.ones(len(frame))
        elif "[" in local:
            name, level = local[:-1].split("[", 1)
            value = (frame[name] == level).to_numpy(dtype=float)
        else:
            value = frame[local].to_numpy(dtype=float)
        (second if auxiliary else count)[:, i] = value
    return count, second


def oracle(model, frame, beta, kind):
    x, z = matrices(model, frame)
    eta = x @ beta + np.log(frame.expo.to_numpy())
    if kind == "xb":
        return eta
    mu = np.exp(eta)
    command = model.spec.estimator
    link = model.extra.get("inflate_link", model.extra.get("zero_link"))
    a = z @ beta
    probability = (special.expit(a) if link == "logit" else special.ndtr(a)
                   if link == "probit" else -np.expm1(-np.exp(a)))
    if command in {"zip", "zinb"} and kind == "response":
        return (1 - probability) * mu
    limit = (frame.ll.to_numpy() if "truncation" in model.spec.columns
             else np.full(len(frame), model.spec.options.get("ll", 0)))
    nb = command in {"zinb", "tnbreg"} or command == "hurdle" and model.extra["dist"] == "nbinomial"
    if nb:
        dispersion = np.exp(beta[-1])
        shape = mu / dispersion if model.spec.options.get("dispersion") == "constant" else np.full_like(mu, 1 / dispersion)
        p = shape / (shape + mu)
        positive = mu * stats.nbinom.sf(limit - 1, shape + 1, p) / stats.nbinom.sf(limit, shape, p)
    else:
        positive = mu * stats.poisson.sf(limit - 1, mu) / stats.poisson.sf(limit, mu)
    return positive * probability if command == "hurdle" and kind == "response" else positive


def gradient(function, beta, step=2e-5):
    columns = []
    for index in range(len(beta)):
        delta = step * max(1, abs(beta[index]))
        high, low = beta.copy(), beta.copy()
        high[index] += delta
        low[index] -= delta
        columns.append((function(high) - function(low)) / (2 * delta))
    return np.stack(columns, axis=-1)


def effect(model, frame, beta, variable, kind):
    if variable == "g":
        low, high = frame.copy(), frame.copy()
        low.g, high.g = "A", "B"
        return oracle(model, high, beta, kind) - oracle(model, low, beta, kind)
    step = 3e-4
    low, high = frame.copy(), frame.copy()
    low[variable] -= step
    high[variable] += step
    return (oracle(model, high, beta, kind) - oracle(model, low, beta, kind)) / (2 * step)


@pytest.mark.parametrize("kind", ["response", "xb", "conditional", "stdp", "derivative"])
def test_saved_predict_full_covariance_and_probability_oracle(fitted, kind):
    model, frame, _ = fitted
    new = frame.iloc[:13].drop(columns="y")
    beta = np.array([c.estimate for c in model.coefficients])
    selected = "xb" if kind == "stdp" else kind
    function = ((lambda b: effect(model, new, b, "x", "response")) if kind == "derivative"
                else lambda b: oracle(model, new, b, selected))
    expected, jac = function(beta), gradient(function, beta)
    se = np.sqrt(np.einsum("nk,kl,nl->n", jac, model.covariance_matrix, jac))
    output = oe.predict(model, new, kind=kind, term="x" if kind == "derivative" else None,
                        interval=None if kind == "stdp" else "mean")
    label = "dydx[x]" if kind == "derivative" else kind
    np.testing.assert_allclose(output[label], se if kind == "stdp" else expected,
                               rtol=3e-6, atol=3e-9)
    if kind != "stdp":
        np.testing.assert_allclose(output.std_error, se, rtol=3e-5, atol=3e-9)
    assert output.attrs["precision"] == "float64"
    assert model.spec.outcome not in new


@pytest.mark.parametrize("variable", ["x", "z", "g", "xcopy"])
@pytest.mark.parametrize("kind", ["response", "xb", "conditional"])
def test_saved_ame_overlap_omissions_discrete_contrasts_and_weights(fitted, variable, kind):
    model, frame, _ = fitted
    if variable not in model.spec.predictors and variable not in {"z", "g"}:
        pytest.fail("Fixture mismatch")
    if variable == "z" and model.spec.estimator in {"tpoisson", "tnbreg"}:
        with pytest.raises(AnalysisError, match="original mean predictor"):
            oe.margins(model, [variable], data=frame.iloc[:19], kind=kind)
        return
    new = frame.iloc[:19].drop(columns="y")
    beta = np.array([c.estimate for c in model.coefficients])
    weight = new.w.to_numpy(dtype=float)
    weight /= weight.sum()
    def fn(b):
        return weight @ effect(model, new, b, variable, kind)
    expected, jac = fn(beta), gradient(fn, beta, step=1e-4)
    se = np.sqrt(jac @ np.array(model.covariance_matrix) @ jac)
    output = oe.margins(model, [variable], data=new, kind=kind)
    row = output.iloc[0]
    np.testing.assert_allclose(row.estimate, expected, rtol=3e-6, atol=3e-9)
    np.testing.assert_allclose(row.std_error, se, rtol=1e-4, atol=3e-8)
    np.testing.assert_allclose(output.attrs["delta_gradients"][0], jac, rtol=1e-4, atol=3e-8)


@pytest.mark.parametrize("kind", ["response", "xb", "conditional"])
def test_mem_uses_one_global_encoded_mean(fitted, kind):
    model, frame, case = fitted
    if case.endswith("varying"):
        new = frame.iloc[:19].drop(columns="y").copy()
        new.ll = np.tile([1, 2], 10)[:len(new)]
        with pytest.raises(AnalysisError, match="encoded mean"):
            oe.margins(model, ["x"], data=new, kind="conditional", method="mem")
        return
    new = frame.iloc[:19].drop(columns="y")
    configured = _model(model)
    _, retained, _ = _data(configured, new, weights=True)
    design = _encode(configured, retained)
    weight = torch.tensor(new.w.tolist(), dtype=torch.float64)
    weight /= weight.sum()
    design = type(design)((weight @ design.x)[None], (weight @ design.deterministic)[None],
                          (weight @ design.scale)[None])
    beta = _parameters(model).beta
    adapter = configured.response_adapter
    expected = adapter.effects(design, beta, "x", kind)[0]
    gradient_expected = adapter.jacobian(design, beta, kind, "x")[0]
    se = (gradient_expected @ _parameters(model).covariance @ gradient_expected).sqrt()
    out = oe.margins(model, ["x"], data=new, kind=kind, method="mem")
    np.testing.assert_allclose(out.estimate, float(expected.detach()), rtol=1e-13)
    np.testing.assert_allclose(out.std_error, float(se), rtol=1e-13)


def test_missing_rows_unknown_categories_and_table_export(fitted):
    model, frame, _ = fitted
    new = frame.iloc[:9].drop(columns="y").copy()
    new.iloc[2, new.columns.get_loc("x")] = np.nan
    output = oe.predict(model, new)
    assert output.index.tolist() == new.index.tolist()
    assert np.isnan(output.response.iloc[2])
    assert output.attrs["missing_row_positions"] == [2]
    assert "\\begin{tabular}" in output.to_latex()
    bad = new.copy()
    bad.g = bad.g.astype(object)
    bad.iloc[0, bad.columns.get_loc("g")] = "unfitted"
    with pytest.raises(AnalysisError, match="category absent"):
        oe.predict(model, bad)


@pytest.mark.parametrize("corruption", ["equation", "omissions", "role", "options", "coding", "extra"])
def test_saved_metadata_cannot_silently_change_model(fitted, corruption):
    model, frame, _ = fitted
    bad = model.model_copy(deep=True)
    if corruption == "equation":
        bad.coefficients[0].equation = "foreign"
    elif corruption == "omissions":
        bad.provenance["omitted_terms"].append("foreign")
    elif corruption == "role":
        bad.spec.columns["foreign"] = "z"
    elif corruption == "options":
        bad.spec.options["foreign"] = True
    elif corruption == "coding":
        bad.provenance["categorical_encoding"]["g"]["reference"] = "B"
    else:
        key = "inflate_link" if model.spec.estimator in {"zip", "zinb"} else "dist" if model.spec.estimator == "hurdle" else "truncation_column" if "truncation" in model.spec.columns else "truncation_point"
        bad.extra[key] = "foreign"
    with pytest.raises(AnalysisError) as exc:
        configure(bad, _parameters(bad))
    assert exc.value.code == "invalid_result"


@pytest.mark.parametrize("kind", ["response", "xb", "conditional", "stdp", "derivative"])
def test_actual_indexed_parquet_prediction_batches_match_full_saved_inference(fitted, kind, tmp_path, monkeypatch):
    model, frame, _ = fitted
    new = frame.iloc[:33].drop(columns="y").copy()
    new.index = pd.Index(np.resize(["same", "other", "same"], len(new)), name="observation")
    new.iloc[11, new.columns.get_loc("x")] = np.nan
    path = tmp_path / "new-counts.parquet"
    new.to_parquet(path, index=True)
    source = scan(path)
    options = dict(kind=kind, term="x" if kind == "derivative" else None,
                   interval=None if kind == "stdp" else "mean")
    expected = oe.predict(model, new, **options)

    encode = CountMixture.encode

    def bounded_encode(self, frame, design):
        assert len(frame) <= 7, "Prediction must not encode the whole evaluation Dataset"
        return encode(self, frame, design)

    monkeypatch.setattr(CountMixture, "encode", bounded_encode)
    output = oe.predict(model, source, batch_rows=7, **options)
    assert isinstance(output, Dataset)
    actual = pd.concat(list(output.iter_batches()))
    assert actual.index.equals(expected.index)
    assert actual.columns.tolist() == expected.columns.tolist()
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-13, equal_nan=True)
    record = output.metadata["analysis"]["streaming"]
    assert record["maximum_batch_rows"] <= 7
    assert record["full_source_collected"] is False
    assert output.metadata["analysis"]["missing_prediction_rows"] == 1


@pytest.mark.parametrize("kind", ["response", "conditional"])
@pytest.mark.parametrize("method", ["ame", "mem"])
def test_actual_parquet_margins_global_weights_gradients_and_mem(fitted, kind, method, tmp_path, monkeypatch):
    model, frame, case = fitted
    new = frame.iloc[:41].drop(columns="y").copy()
    new.index = pd.Index(np.resize(["repeat", "other"], len(new)), name="row")
    new.iloc[11, new.columns.get_loc("x")] = np.nan
    new.iloc[2, new.columns.get_loc("w")] = 0
    if case.endswith("varying") and method == "mem":
        new.ll = 2
    path = tmp_path / "count-margins.parquet"
    new.to_parquet(path, index=True)
    expected = oe.margins(model, ["x", "g"], data=new, kind=kind, method=method)

    encode = CountMixture.encode

    def bounded_encode(self, frame, design):
        assert len(frame) <= 7, "Margins must not encode the whole evaluation Dataset"
        return encode(self, frame, design)

    monkeypatch.setattr(CountMixture, "encode", bounded_encode)
    output = oe.margins(model, ["x", "g"], data=scan(path), kind=kind, method=method, batch_rows=7)
    assert output.columns.tolist() == expected.columns.tolist()
    np.testing.assert_allclose(output.select_dtypes("number"), expected.select_dtypes("number"), rtol=3e-11, atol=1e-13)
    np.testing.assert_allclose(output.attrs["delta_gradients"], expected.attrs["delta_gradients"], rtol=3e-11, atol=1e-13)
    assert output.attrs["streaming"]["maximum_batch_rows"] <= 7
    for name in ["evaluation_rows", "complete_evaluation_rows", "zero_weight_rows_excluded"]:
        assert output.attrs[name] == expected.attrs[name]
