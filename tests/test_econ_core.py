"""Contract tests for the shared registry, sample and result layer."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import stats

from openecon.analysis import AnalysisError, _hash_frame, _prepare_data, fit
from openecon.econometrics import registry
from openecon.econometrics.core import (
    ModelFrame, TableSet, build_result, column_list, information_criteria, lr_test, make_spec,
    ml_covariance, table, wald_test,
)
from openecon.econometrics.registry import EstimatorInfo, Option, Role
from openecon.models import ModelSpec


def _toy_entry(spec, data):
    frame = ModelFrame(spec, data)
    design = frame.drop_collinear(frame.design())
    y = frame.numeric(spec.outcome)
    beta = torch.linalg.lstsq(design.x, y[:, None]).solution[:, 0]
    resid = y - design.x @ beta
    n, k = design.x.shape
    covariance = torch.linalg.inv(design.x.T @ design.x) * (resid @ resid) / (n - k)
    return build_result(frame, terms=design.terms, params=beta, covariance=covariance,
                        df_inference=n - k, df_resid=n - k, fitted=design.x @ beta,
                        metrics={"r_squared": 0.5, "bad": float("nan")}, categories=design.categories,
                        tests={"model": wald_test(beta, covariance, range(1, k), df_resid=n - k, label="F test")},
                        extra={"tensor": torch.tensor([1.0, float("inf")])})


TOY = EstimatorInfo(
    name="toy", title="Toy linear model", family="test", entry="tests.test_econ_core:_toy_entry",
    covariances=("nonrobust", "robust", "opg", "cluster"), default_covariance="nonrobust",
    function=None, stata=("regress",), weights=("aweight", "fweight", "pweight"),
    panel="optional", time="optional", cluster_dimensions=2, inference="t",
    roles=(Role("absorb", many=True, kind="label"), Role("offset")),
    options=(Option("model", "str", "fe", choices=("fe", "re")), Option("lags", "int", 1, minimum=0)),
)


@pytest.fixture(autouse=True)
def toy_estimator(monkeypatch):
    catalogue = dict(registry._load())
    catalogue["toy"] = TOY
    monkeypatch.setattr(registry, "_catalogue", catalogue)
    monkeypatch.setattr(registry, "load_entry", lambda info: _toy_entry if info.name == "toy" else None)


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    n = 60
    frame = pd.DataFrame({
        "y": rng.normal(size=n), "x": rng.normal(size=n), "z": rng.normal(size=n),
        "firm": np.repeat(np.arange(12), 5), "year": np.tile(np.arange(2000, 2005), 12),
        "sector": pd.Categorical(np.resize(["b", "c", "a"], n), categories=["c", "a", "b"]),
        "w": rng.integers(1, 4, size=n).astype(float), "state": np.repeat(["tx", "ca", "ny", "wa"], 15),
    })
    return frame


def test_spec_contract_is_enforced_by_the_registry():
    spec = ModelSpec(estimator="toy", outcome="y", predictors=["x"])
    assert spec.covariance == "nonrobust"
    assert ModelSpec(estimator="toy", outcome="y", predictors=["x"], cluster="firm").covariance == "cluster"
    assert ModelSpec(estimator="toy", outcome="y", predictors=["x"], cluster=["firm"]).cluster == "firm"
    two_way = ModelSpec(estimator="toy", outcome="y", predictors=["x"], cluster=["firm", "year"])
    assert two_way.cluster == ["firm", "year"] and two_way.covariance == "cluster"
    assert ModelSpec.model_validate(json.loads(two_way.model_dump_json())) == two_way
    for bad in [
        {"covariance": "HC3"}, {"covariance": "cluster"}, {"cluster": ["firm", "year", "state"]},
        {"cluster": ["firm", "firm"]}, {"cluster": "firm", "covariance": "robust"},
        {"weights": "w"}, {"weight_type": "aweight"}, {"weights": "w", "weight_type": "iweight"},
        {"columns": {"nope": "x"}}, {"columns": {"offset": ["x", "z"]}}, {"columns": {"absorb": ["firm", "firm"]}},
        {"options": {"model": "be"}}, {"options": {"lags": -1}}, {"options": {"lags": 1.5}},
        {"options": {"unknown": 1}}, {"categorical": ["z"]}, {"panel": " "},
    ]:
        with pytest.raises(ValidationError):
            ModelSpec(**{"estimator": "toy", "outcome": "y", "predictors": ["x"], **bad})
    assert ModelSpec(estimator="toy", outcome="y", predictors=["x"], columns={"absorb": ["firm"]},
                     categorical=["firm"]).categorical == ["firm"]


def test_make_spec_drops_unset_extras_and_column_list_rejects_text():
    spec = make_spec("toy", outcome="y", predictors=["x"], weights=None, weight_type="aweight",
                     columns={"absorb": [], "offset": None}, options={"model": None, "lags": 2})
    assert spec.columns == {} and spec.options == {"lags": 2} and spec.weight_type is None
    with pytest.raises(AnalysisError) as caught:
        column_list("x", "x")
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError, match="must be one of") as caught:
        make_spec("toy", outcome="y", predictors=["x"], options={"model": "be"})
    assert caught.value.code == "invalid_spec"
    assert column_list(None, "x") == [] and column_list(("a", "b"), "x") == ["a", "b"]


def test_frame_missing_policy_positions_and_allowed_missing(data):
    data.loc[[3, 10], "x"] = np.nan
    data.loc[5, "y"] = np.nan
    spec = ModelSpec(estimator="toy", outcome="y", predictors=["x"])
    with pytest.raises(AnalysisError) as caught:
        ModelFrame(spec, data)
    assert caught.value.code == "missing_values"
    spec = spec.model_copy(update={"missing": "drop"})
    frame = ModelFrame(spec, data)
    assert frame.n == 57 and frame.positions == [i for i in range(60) if i not in {3, 5, 10}]
    assert frame.dropped_missing == 3 and "Excluded 3 observation(s)" in frame.warnings[0]
    kept = ModelFrame(spec.model_copy(update={"missing": "raise"}), data.drop(index=[3, 10]), allow_missing=["y"])
    assert kept.n == 58 and kept.series("y").isna().sum() == 1
    partial = kept.numeric("y", allow_missing=True)
    assert int(torch.isnan(partial).sum()) == 1 and partial.dtype == torch.float64
    with pytest.raises(AnalysisError):
        kept.numeric("y")
    with pytest.raises(AnalysisError) as caught:
        ModelFrame(spec, data.assign(x=np.nan))
    assert caught.value.code == "empty_sample"
    with pytest.raises(AnalysisError) as caught:
        ModelFrame(spec, data[["y"]])
    assert caught.value.code == "missing_columns"
    with pytest.raises(AnalysisError) as caught:
        ModelFrame(spec, data.assign(x=np.where(np.arange(60) == 0, np.inf, 1.0))).numeric("x")
    assert caught.value.code == "non_finite_values"


def test_frame_weights_are_screened_like_stata(data):
    spec = ModelSpec(estimator="toy", outcome="y", predictors=["x"], weights="w", weight_type="fweight")
    data.loc[[0, 7], "w"] = 0
    frame = ModelFrame(spec, data)
    assert frame.n == 58 and 0 not in frame.positions and "zero weight" in frame.warnings[0]
    assert_allclose(frame.weights().numpy(), data.w[data.w > 0])
    with pytest.raises(AnalysisError) as caught:
        ModelFrame(spec, data.assign(w=data.w + 0.5))
    assert caught.value.code == "noninteger_frequency_weights"
    with pytest.raises(AnalysisError) as caught:
        ModelFrame(spec.model_copy(update={"weight_type": "aweight"}), data.assign(w=-data.w))
    assert caught.value.code == "negative_weights"


def test_codes_restrict_reorder_and_panel_sort(data):
    shuffled = data.sample(frac=1, random_state=3).reset_index(drop=True)
    spec = ModelSpec(estimator="toy", outcome="y", predictors=["x"], panel="firm", time="year",
                     cluster=["firm", "state"])
    frame = ModelFrame(spec, shuffled)
    codes, count = frame.codes("state")
    assert count == 4 and codes.dtype == torch.int64
    assert frame.levels("state") == list(pd.unique(shuffled.state))
    both, combos = frame.codes(["firm", "year"])
    assert combos == 60 and len(torch.unique(both)) == 60
    frame.sort_panel()
    assert frame.sample[["firm", "year"]].equals(data[["firm", "year"]])
    assert shuffled.loc[frame.positions, "y"].tolist() == data.y.tolist()
    assert frame.time_index().tolist() == data.year.tolist()
    frame.restrict(torch.as_tensor((data.year != 2002).to_numpy()), "Dropped 2002.")
    assert frame.n == 48 and "Dropped 2002." in frame.warnings
    gaps = frame.time_index()
    assert set((gaps[1:] - gaps[:-1]).tolist()) == {1, 2, -4}
    dimensions = frame.cluster_dimensions()
    assert [count for _, count in dimensions] == [12, 4]
    assert any("Only 12 clusters" in warning for warning in frame.warnings)
    with pytest.raises(AnalysisError) as caught:
        ModelFrame(spec, pd.concat([shuffled, shuffled.iloc[:1]])).sort_panel()
    assert caught.value.code == "repeated_time_values"
    with pytest.raises(AnalysisError) as caught:
        frame.restrict(torch.ones(3, dtype=torch.bool))
    assert caught.value.code == "invalid_sample_filter"


def test_time_index_handles_delta_dates_and_fractions(data):
    spec = ModelSpec(estimator="toy", outcome="y", predictors=["x"], time="t")
    frame = ModelFrame(spec, data.assign(t=np.arange(60) * 5 + 3))
    assert frame.time_index(delta=5).tolist() == list(range(60))
    huge = ModelFrame(spec, data.assign(t=np.arange(60, dtype="int64") + 2**60))
    assert (huge.time_index() - 2**60).tolist() == list(range(60))
    dated = ModelFrame(spec, data.assign(t=pd.date_range("2020-01-31", periods=60, freq="ME")))
    assert dated.time_index().tolist() == list(range(60)) and "Datetime column" in dated.warnings[0]
    with pytest.raises(AnalysisError) as caught:
        ModelFrame(spec, data.assign(t=np.arange(60) + 0.5)).time_index()
    assert caught.value.code == "invalid_time"


def test_design_matches_core_treatment_coding(data):
    data.loc[4, "x"] = np.nan
    legacy = ModelSpec(outcome="y", predictors=["x", "sector", "z"], categorical=["sector"], missing="drop")
    _, positions, _, expected, terms, categories = _prepare_data(legacy, data)
    spec = ModelSpec(estimator="toy", outcome="y", predictors=["x", "sector", "z"], categorical=["sector"],
                     missing="drop")
    frame = ModelFrame(spec, data)
    design = frame.design()
    assert design.terms == terms == ["Intercept", "x", "sector[a]", "sector[b]", "z"]
    assert torch.equal(design.x, expected) and design.categories == categories and design.intercept
    assert frame.positions == positions
    auxiliary = frame.design(["z"], intercept=True, prefix="select:")
    assert auxiliary.terms == ["select:Intercept", "select:z"]
    assert frame.design([], intercept=False).x.shape == (59, 0)


def test_collinear_terms_are_omitted_and_recorded(data):
    data["twice"] = 2 * data.x
    data["one"] = 1.0
    spec = ModelSpec(estimator="toy", outcome="y", predictors=["x", "one", "z", "twice"])
    result = fit(spec, data=data)
    assert [c.term for c in result.coefficients] == ["Intercept", "x", "z"]
    assert result.provenance["omitted_terms"] == ["one", "twice"]
    assert any("collinearity" in warning for warning in result.warnings)


def test_centered_screen_keeps_year_polynomials_and_absorbed_screen_drops_within_constants(data):
    data["year2"] = data.year ** 2
    data["year3"] = data.year ** 3
    frame = ModelFrame(ModelSpec(estimator="toy", outcome="y", predictors=["year", "year2", "year3", "x"]), data)
    design = frame.drop_collinear(frame.design())
    # Five distinct years identify the cubic only at the 1e-14 level: it is omitted, the square is kept.
    assert design.terms == ["Intercept", "year", "year2", "x"]
    assert frame.notes["omitted_terms"] == ["year3"]
    weighted = ModelFrame(ModelSpec(estimator="toy", outcome="y", predictors=["year", "year2", "x"], weights="w",
                                    weight_type="aweight"), data)
    assert weighted.drop_collinear(weighted.design(), weighted.weights()).terms == ["Intercept", "year", "year2", "x"]
    # Within-panel demeaning removes a panel-constant regressor exactly up to rounding.
    data["constant_in_firm"] = data.firm * 3.0
    # A large offset with small but real within variation must survive the absorbed screen.
    data["offset"] = 1e4 * data.firm + 0.05 * np.random.default_rng(5).normal(size=60)
    frame = ModelFrame(ModelSpec(estimator="toy", outcome="y", predictors=["x", "constant_in_firm", "z", "offset"]), data)
    design = frame.design(intercept=False)
    codes = torch.as_tensor(data.firm.to_numpy(), dtype=torch.int64)
    means = torch.zeros((12, 4), dtype=torch.float64).index_add_(0, codes, design.x) / 5
    demeaned = design.x - means[codes] + 1e-13 * torch.randn(60, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    reduced, within = frame.drop_absorbed(design, demeaned)
    assert reduced.terms == ["x", "z", "offset"] and within.shape == (60, 3)
    assert frame.notes["omitted_terms"] == ["constant_in_firm"]
    assert "absorbed fixed effects" in frame.warnings[-1]


def test_build_result_inference_provenance_and_json_safety(data):
    spec = ModelSpec(estimator="toy", outcome="y", predictors=["x", "z"], alpha=0.1)
    result = fit(spec, data=data)
    design = np.column_stack([np.ones(60), data.x, data.z])
    beta, *_ = np.linalg.lstsq(design, data.y.to_numpy(), rcond=None)
    residual = data.y.to_numpy() - design @ beta
    covariance = np.linalg.inv(design.T @ design) * (residual @ residual) / 57
    errors = np.sqrt(np.diag(covariance))
    assert_allclose([c.estimate for c in result.coefficients], beta, rtol=1e-10)
    assert_allclose([c.p_value for c in result.coefficients], 2 * stats.t.sf(np.abs(beta / errors), 57), rtol=1e-9)
    critical = stats.t.ppf(0.95, 57)
    assert_allclose([[c.ci_low, c.ci_high] for c in result.coefficients],
                    np.column_stack([beta - critical * errors, beta + critical * errors]), rtol=1e-9)
    # Hashes follow the core convention: model-input columns in positional row order.
    assert result.provenance["data_hash"] == _hash_frame(data[["y", "x", "z"]])
    assert result.provenance["sample_hash"] == _hash_frame(data[["y", "x", "z"]], list(range(60)))
    assert result.title == "Toy linear model" and result.provenance["stata_equivalent"] == ["regress"]
    assert result.metrics == {"r_squared": 0.5, "bad": None}
    assert result.extra == {"tensor": [1.0, None]}
    assert len(result.predictions) == 60 and result.predictions[3]["row"] == 3
    assert_allclose([p["observed"] - p["fitted"] for p in result.predictions],
                    [p["residual"] for p in result.predictions])
    test = result.tests["model"]
    beta = np.array([c.estimate for c in result.coefficients])[1:]
    block = np.asarray(result.covariance_matrix)[1:, 1:]
    statistic = beta @ np.linalg.solve(block, beta) / 2
    assert test["statistic"] == pytest.approx(statistic, rel=1e-10)
    assert test["p_value"] == pytest.approx(stats.f.sf(statistic, 2, 57), rel=1e-9)
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result
    text = result.summary()
    assert "Toy linear model — y" in text and "F test: F(2, 57)" in text
    assert r"R^{2}" in str(result.to_latex())


def test_normal_inference_and_equations(data):
    frame = ModelFrame(ModelSpec(estimator="toy", outcome="y", predictors=["x"]), data)
    params = torch.tensor([0.4, -1.2, 0.1], dtype=torch.float64)
    covariance = torch.diag(torch.tensor([0.04, 0.09, 0.01], dtype=torch.float64))
    result = build_result(frame, terms=["x", "select:x", "/lnsigma"], params=params, covariance=covariance,
                          use_t=False, equations=["y", "select", None])
    assert_allclose([c.p_value for c in result.coefficients], 2 * stats.norm.sf(np.abs([2.0, -4.0, 1.0])), rtol=1e-12)
    assert result.inference["distribution"] == "normal" and result.inference["df_inference"] is None
    assert [c.equation for c in result.coefficients] == ["y", "select", None]
    assert "[select]" in result.summary() and result.predictions == []
    for bad in [{"terms": ["a", "a", "b"]}, {"covariance": torch.diag(torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64))},
                {"params": torch.tensor([1.0, float("nan"), 0.0], dtype=torch.float64)}, {"use_t": True}]:
        arguments = {"terms": ["a", "b", "c"], "params": params, "covariance": covariance, "use_t": False, **bad}
        with pytest.raises(AnalysisError):
            build_result(frame, **arguments)


@pytest.mark.parametrize("kind", ["nonrobust", "opg", "robust", "cluster", "twoway", "frequency"])
def test_ml_covariance_follows_stata_ml_conventions(data, kind):
    rng = np.random.default_rng(11)
    n, k = 60, 3
    scores = rng.normal(size=(n, k))
    root = rng.normal(size=(k, k))
    hessian = -(root @ root.T + 3 * np.eye(k))
    bread = np.linalg.inv(-hessian)
    update = {}
    if kind == "cluster":
        update = {"cluster": "firm"}
    elif kind == "twoway":
        update = {"cluster": ["firm", "state"]}
    elif kind == "frequency":
        update = {"covariance": "robust", "weights": "w", "weight_type": "fweight"}
    else:
        update = {"covariance": kind}
    frame = ModelFrame(ModelSpec(estimator="toy", outcome="y", predictors=["x"], **update), data)
    frequency = frame.weights() if kind == "frequency" else None
    nobs = int(data.w.sum()) if kind == "frequency" else None
    actual, info = ml_covariance(frame, hessian=torch.as_tensor(hessian), scores=torch.as_tensor(scores),
                                 nobs=nobs, frequency=frequency)

    def clustered(labels):
        sums = pd.DataFrame(scores).groupby(np.asarray(labels), sort=False).sum().to_numpy()
        return sums.T @ sums

    if kind == "nonrobust":
        expected = bread
    elif kind == "opg":
        expected = np.linalg.inv(scores.T @ scores)
    elif kind == "robust":
        expected = n / (n - 1) * bread @ scores.T @ scores @ bread
    elif kind == "frequency":
        total = data.w.sum()
        expected = total / (total - 1) * bread @ (scores * data.w.to_numpy()[:, None]).T @ scores @ bread
    elif kind == "cluster":
        expected = 12 / 11 * bread @ clustered(data.firm) @ bread
        assert info["cluster_count"] == 12 and info["cluster_df"] == 11
    else:
        both = data.firm.astype(str) + "/" + data.state
        meat = clustered(data.firm) + clustered(data.state) - clustered(both)
        expected = 4 / 3 * bread @ meat @ bread
        assert info["cluster_counts"][:2] == [12, 4] and info["cluster_columns"] == ["firm", "state"]
    assert_allclose(actual.numpy(), expected, rtol=1e-10, atol=1e-12)


def test_test_helpers_match_reference_distributions():
    params = torch.tensor([0.5, -0.2, 0.3], dtype=torch.float64)
    covariance = torch.tensor([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.01]], dtype=torch.float64)
    wald = wald_test(params, covariance, [0, 1])
    b, v = params.numpy()[:2], covariance.numpy()[:2, :2]
    statistic = b @ np.linalg.solve(v, b)
    assert wald["statistic"] == pytest.approx(statistic) and wald["df"] == 2
    assert wald["p_value"] == pytest.approx(stats.chi2.sf(statistic, 2), rel=1e-10)
    singular = wald_test(torch.tensor([1.0, 1.0], dtype=torch.float64), torch.ones((2, 2), dtype=torch.float64), [0, 1])
    assert singular["df"] == 1 and singular["statistic"] == pytest.approx(1.0)
    assert wald_test(params, covariance, [])["statistic"] is None
    ratio = lr_test(-100.0, -104.5, 3, label="LR")
    assert ratio["statistic"] == pytest.approx(9.0) and ratio["p_value"] == pytest.approx(stats.chi2.sf(9, 3))
    criteria = information_criteria(-100.0, 4, 50)
    assert criteria["aic"] == 208 and criteria["bic"] == pytest.approx(200 + 4 * np.log(50))


def test_tables_carry_scalar_results_and_render():
    anova = table({"source": ["between", "within"], "ss": [10.0, 5.0], "df": [2, 27]}, statistic=27.0,
                  p_value=float("nan"))
    assert type(anova).__name__ == "DataFrame" and anova.attrs == {"statistic": 27.0, "p_value": None}
    assert "between" in str(anova.to_latex())
    result = TableSet({"anova": anova, "means": table([[1, 2.5]], columns=["group", "mean"])},
                      title="One-way ANOVA", f=27.0)
    text = str(result)
    assert text.startswith("One-way ANOVA") and "[anova]" in text and "[means]" in text and "f: 27.0" in text
    assert result["means"].loc[0, "mean"] == 2.5 and "between" in result.to_latex()
