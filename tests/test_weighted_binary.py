"""Independent direct weighted binary API, inference and saved-state acceptance."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import root
from scipy.special import expit, log_ndtr
from scipy.stats import norm

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ModelSpec, ResultBundle

MODELS = ("logit", "probit")
WEIGHTS = ("fweight", "aweight", "iweight", "pweight")
CELLS = [
    (w, v)
    for w in WEIGHTS
    for v in ("nonrobust", "opg", "robust", "cluster")
    if w != "pweight" or v in ("robust", "cluster")
]


def data(seed=752):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=(2, 240))
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "y": rng.binomial(1, expit(0.2 + 0.5 * x - 0.3 * z)),
            "w": rng.integers(1, 5, len(x)).astype(float),
            "g": np.arange(len(x)) % 24,
        }
    )


def reference(frame, model, weight, kind, intercept=True):
    """Solve independent weighted scores and assemble observed-information inference."""
    x = frame[["x", "z"]].to_numpy()
    if intercept:
        x = np.column_stack([np.ones(len(frame)), x])
    y = frame.y.to_numpy()
    w = frame.w.to_numpy().copy()
    n = int(w.sum()) if weight == "fweight" else len(y)
    if weight == "aweight":
        w *= len(w) / w.sum()

    def pieces(beta):
        eta = x @ beta
        if model == "logit":
            p = expit(eta)
            unit = y - p
            curvature = p * (1 - p)
            ll = y * -np.logaddexp(0, -eta) + (1 - y) * -np.logaddexp(0, eta)
        else:
            q = 2 * y - 1
            t = q * eta
            mills = np.exp(norm.logpdf(t) - log_ndtr(t))
            unit = q * mills
            curvature = mills * (mills + t)
            ll = log_ndtr(t)
        return x * unit[:, None], x.T @ ((w * curvature)[:, None] * x), w @ ll

    run = root(
        lambda b: (w[:, None] * pieces(b)[0]).sum(axis=0),
        np.zeros(x.shape[1]),
        jac=lambda b: -pieces(b)[1],
        tol=1e-10,
    )
    assert np.max(np.abs((w[:, None] * pieces(run.x)[0]).sum(axis=0))) < 1e-7
    scores, information, ll = pieces(run.x)
    bread = np.linalg.inv(information)
    if kind == "nonrobust":
        covariance = bread
    elif kind == "opg":
        covariance = np.linalg.inv(scores.T @ (w[:, None] * scores))
    elif kind == "robust":
        meat = scores.T @ ((w if weight == "fweight" else w * w)[:, None] * scores)
        covariance = bread @ meat @ bread * n / (n - 1)
    else:
        sums = np.array(
            [(w[frame.g == g, None] * scores[frame.g == g]).sum(axis=0) for g in pd.unique(frame.g)]
        )
        covariance = bread @ (sums.T @ sums) @ bread * len(sums) / (len(sums) - 1)
    return run.x, covariance, ll, n, scores, information, w


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("weight,kind", CELLS)
@pytest.mark.parametrize("intercept", [True, False])
def test_full_independent_matrix(model, weight, kind, intercept):
    frame = data()
    result = getattr(oe, model)(
        data=frame,
        y="y",
        x=["x", "z"],
        weights="w",
        weight_type=weight,
        covariance=kind,
        intercept=intercept,
        cluster="g" if kind == "cluster" else None,
    )
    beta, covariance, ll, n, scores, information, w = reference(
        frame, model, weight, kind, intercept
    )
    np.testing.assert_allclose(
        [c.estimate for c in result.coefficients], beta, rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(result.covariance_matrix, covariance, rtol=3e-8, atol=2e-10)
    assert result.nobs == n and result.nobs_original == len(frame)
    assert result.sample_positions == list(range(len(frame)))
    assert result.metrics["log_likelihood"] == pytest.approx(ll, rel=1e-10)
    null_mean = float(w @ frame.y.to_numpy() / w.sum()) if intercept else 0.5
    null_ll = float(
        w
        @ (frame.y.to_numpy() * np.log(null_mean) + (1 - frame.y.to_numpy()) * np.log1p(-null_mean))
    )
    assert result.extra["null_log_likelihood"] == pytest.approx(null_ll, rel=1e-10)
    assert result.metrics["pseudo_r_squared"] == pytest.approx(1 - ll / null_ll)
    assert result.metrics["aic"] == pytest.approx(-2 * ll + 2 * len(beta))
    assert result.metrics["bic"] == pytest.approx(-2 * ll + np.log(n) * len(beta))
    se = np.sqrt(covariance.diagonal())
    np.testing.assert_allclose([c.std_error for c in result.coefficients], se, rtol=3e-8)
    np.testing.assert_allclose([c.statistic for c in result.coefficients], beta / se, rtol=3e-8)
    np.testing.assert_allclose(
        [c.p_value for c in result.coefficients], 2 * norm.sf(abs(beta / se)), rtol=5e-8
    )
    np.testing.assert_allclose(
        [c.ci_low for c in result.coefficients], beta - norm.ppf(0.975) * se, atol=2e-8
    )
    np.testing.assert_allclose(
        [c.ci_high for c in result.coefficients], beta + norm.ppf(0.975) * se, atol=2e-8
    )
    assert result.inference["distribution"] == "normal" and result.inference["df_inference"] is None
    assert result.provenance["optimizer"]["converged"]
    assert not result.provenance["stata_parity_validated"]
    from openecon.econometrics.postest.scores import logit_scores, probit_scores

    scored = {"logit": logit_scores, "probit": probit_scores}[model](
        result, frame, result.sample_positions
    )
    np.testing.assert_allclose(scored.scores, scores * w[:, None], atol=2e-8)
    np.testing.assert_allclose(scored.hessian, -information, atol=2e-7)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("kind", ["nonrobust", "opg", "robust", "cluster"])
def test_frequency_exact_row_replication(model, kind):
    frame = data()
    expanded = (
        frame.loc[frame.index.repeat(frame.w.astype(int))].reset_index(drop=True).assign(w=1.0)
    )
    args = dict(
        y="y",
        x=["x", "z"],
        weights="w",
        weight_type="fweight",
        covariance=kind,
        cluster="g" if kind == "cluster" else None,
    )
    actual = getattr(oe, model)(data=frame, **args)
    repeated = getattr(oe, model)(data=expanded, **args)
    np.testing.assert_allclose(
        actual.covariance_matrix, repeated.covariance_matrix, rtol=2e-9, atol=2e-11
    )
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in repeated.coefficients],
        atol=2e-10,
    )
    assert actual.nobs == repeated.nobs


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("weight,kind", [(w, v) for w, v in CELLS if w != "fweight"])
def test_weight_scale_laws(model, weight, kind):
    frame = data()
    args = dict(
        y="y",
        x=["x", "z"],
        weights="w",
        weight_type=weight,
        covariance=kind,
        cluster="g" if kind == "cluster" else None,
    )
    a, b = [getattr(oe, model)(data=d, **args) for d in (frame, frame.assign(w=frame.w * 17))]
    scale = 17 if weight == "iweight" and kind in {"nonrobust", "opg"} else 1
    np.testing.assert_allclose(
        a.covariance_matrix, np.asarray(b.covariance_matrix) * scale, rtol=3e-8
    )
    np.testing.assert_allclose(
        [c.estimate for c in a.coefficients], [c.estimate for c in b.coefficients], atol=2e-9
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("weight", WEIGHTS)
def test_saved_state_sample_categories_and_queries(model, weight):
    frame = data().assign(
        category=pd.Categorical(np.arange(240) % 3, categories=[2, 0, 1], ordered=True)
    )
    frame.index = ["duplicate"] * len(frame)
    frame.iloc[0, frame.columns.get_loc("w")] = 0
    frame.iloc[3, frame.columns.get_loc("x")] = np.nan
    result = getattr(oe, model)(
        data=frame,
        y="y",
        x=["x", "category"],
        categorical=["category"],
        weights="w",
        weight_type=weight,
        missing="drop",
    )
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored == result
    assert result.sample_positions == [i for i in range(240) if i not in (0, 3)]
    assert result.dropped_rows == 2
    assert result.provenance["categorical_encoding"]["category"]["reference"] == 2
    for operation, kwargs in [
        (oe.predict, {"data": frame}),
        (oe.margins, {"data": frame, "variables": ["x"]}),
        (oe.lincom, {"mapping": {"x": 1.0}}),
        (oe.test, {"restrictions": {"x": 1.0}}),
    ]:
        a, b = operation(result, **kwargs), operation(restored, **kwargs)
        assert json.dumps(
            a.model_dump() if hasattr(a, "model_dump") else a, sort_keys=True, default=str
        ) == json.dumps(
            b.model_dump() if hasattr(b, "model_dump") else b, sort_keys=True, default=str
        )


@pytest.mark.parametrize("model", MODELS)
def test_defaults_admission_and_legacy_preservation(model, monkeypatch):
    frame = data()
    assert (
        getattr(oe, model)(
            data=frame, y="y", x=["x"], weights="w", weight_type="pweight"
        ).spec.covariance
        == "robust"
    )
    for kind in ["opg", "nonrobust"]:
        with pytest.raises(AnalysisError, match="pweights require"):
            getattr(oe, model)(
                data=frame, y="y", x=["x"], weights="w", weight_type="pweight", covariance=kind
            )
    for kind in ["robust", "opg"]:
        with pytest.raises(ValueError):
            ModelSpec(estimator=model, outcome="y", predictors=["x"], covariance=kind)
    legacy = getattr(oe, model)(data=frame, y="y", x=["x"], cluster="g")
    assert legacy.inference["correction"].startswith("CR1:")
    from openecon.dataset import Dataset

    with pytest.raises(AnalysisError, match="resident"):
        getattr(oe, model)(
            data=Dataset.from_frame(frame), y="y", x=["x"], weights="w", weight_type="fweight"
        )
    for changed, code in [
        (frame.assign(w=-1.0), "negative_weights"),
        (frame.assign(w=1.5), "noninteger_frequency_weights"),
        (frame.assign(y=0), "constant_outcome"),
        (frame.assign(y=(frame.x > 0).astype(int)), "separation_detected"),
        (frame.assign(z=frame.x), "rank_deficient"),
    ]:
        with pytest.raises(AnalysisError) as exc:
            getattr(oe, model)(
                data=changed, y="y", x=["x", "z"], weights="w", weight_type="fweight"
            )
        assert exc.value.code == code
    from openecon.resources import use_workspace_budget

    with use_workspace_budget(1), pytest.raises(AnalysisError):
        getattr(oe, model)(
            data=pd.concat([frame] * 30), y="y", x=["x", "z"], weights="w", weight_type="fweight"
        )
    import openecon.econometrics.weighted_binary as module

    monkeypatch.setattr(module, "MAX_ROWS", 1)
    with pytest.raises(AnalysisError) as exc:
        getattr(oe, model)(data=frame, y="y", x=["x"], weights="w", weight_type="fweight")
    assert exc.value.code == "weighted_binary_row_limit"


def test_metadata_does_not_invent_weighted_streaming():
    capabilities = oe.capabilities()
    for model in MODELS:
        assert capabilities["streaming"]["covariances_by_estimator"][model] == [
            "nonrobust",
            "cluster",
        ]
        metadata = capabilities["estimators"][model]
        assert set(metadata["weights"]) == set(WEIGHTS)
        assert metadata["weighted_binary"]["covariances_by_weight"]["pweight"] == [
            "robust",
            "cluster",
        ]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "case,code",
    [
        ("missing", "missing_values"),
        ("nonbinary", "invalid_binary_outcome"),
        ("single_cluster", "insufficient_clusters"),
        ("infinite", "non_finite_values"),
        ("frequency_precision", "weight_precision_unsupported"),
        ("overflow", "weight_precision_unsupported"),
        ("width", "weighted_binary_work_limit"),
        ("work", "weighted_binary_work_limit"),
    ],
)
def test_admission_before_unsupported_numerical_work(model, case, code, monkeypatch):
    frame = data()
    import openecon.econometrics.weighted_binary as module

    weight = "fweight"
    x = ["x", "z"]
    if case == "missing":
        frame.loc[0, "x"] = np.nan
    elif case == "nonbinary":
        frame.loc[0, "y"] = 2
    elif case == "single_cluster":
        frame.g = 1
    elif case == "infinite":
        frame.loc[0, "w"] = np.inf
    elif case == "frequency_precision":
        frame.w = float(2**53)
    elif case == "overflow":
        frame.w = 1e308
        weight = "iweight"
    elif case == "width":
        frame["cat"] = pd.Categorical(np.arange(240), categories=range(240))
        x = ["cat"]
    else:
        monkeypatch.setattr(module, "MAX_WORK", 1)
    with pytest.raises(AnalysisError) as exc:
        getattr(oe, model)(
            data=frame,
            y="y",
            x=x,
            weights="w",
            weight_type=weight,
            categorical=["cat"] if case == "width" else None,
            cluster="g" if case == "single_cluster" else None,
        )
    assert exc.value.code == code
