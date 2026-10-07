"""Dynamic panel GMM: failure contract, missing data, warnings, serialization and speed."""

import json
import time

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pydantic import ValidationError
from test_econ_dpanel_oracle import coefs, simulate

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ModelSpec, ResultBundle


@pytest.fixture(scope="module")
def panel():
    return simulate(seed=8)


def _code(**arguments):
    with pytest.raises(AnalysisError) as caught:
        oe.xtdpd(**arguments)
    return caught.value.code


def test_failure_contract(panel):
    base = dict(data=panel, y="y", x=["x"], panel="id", time="year")
    cases = [
        ({"gmm": [{"columns": ["y"], "lags": [3, 2]}]}, "invalid_spec"),
        ({"gmm": [{"columns": ["y"], "lag": [2, None]}]}, "invalid_spec"),
        ({"gmm": [{"columns": ["y"], "equation": "level"}]}, "invalid_spec"),
        ({"gmm": [{"columns": ["y"], "lags": [0, 0]}], "system": True}, "invalid_spec"),
        ({"gmm": [{"columns": ["y"], "lags": [-1, 2]}]}, "invalid_spec"),
        ({"iv": [{"columns": ["x"], "passthru": True}], "system": True}, "invalid_spec"),
        ({"gmm": [{"columns": ["y"], "lags": [2.5, None]}]}, "invalid_spec"),
        ({"gmm": [{"columns": ["y"], "collapse": "yes"}]}, "invalid_spec"),
        ({"iv": [{"columns": []}]}, "invalid_spec"),
        ({"iv": [{"columns": ["x"], "passthru": 1}]}, "invalid_spec"),
        ({"iv": [{"columns": ["x"], "equation": "levels"}]}, "invalid_spec"),
        ({"gmm": "y"}, "invalid_spec"),
        ({"gmm": [{"columns": ["id"]}]}, "invalid_spec"),
        ({"robust": True, "covariance": "nonrobust"}, "invalid_spec"),
        ({"covariance": "HC1"}, "invalid_spec"),
        ({"h": 4}, "invalid_spec"),
        ({"x": "x"}, "invalid_spec"),
        ({"x": None, "lags": 0}, "invalid_spec"),
        ({"gmm": [{"columns": ["nope"]}]}, "missing_columns"),
        ({"gmm": [], "iv": []}, "underidentified"),
        ({"gmm": [{"columns": ["y"], "lags": [2, 2], "collapse": True}], "iv": []},
         "underidentified"),
    ]
    for update, code in cases:
        assert _code(**{**base, **update}) == code, update
    for update in ({"lags": 0}, {"maxldep": 0}, {"predetermined": ["x"]}):
        with pytest.raises(AnalysisError) as caught:
            oe.xtabond(**{**base, **update})
        assert caught.value.code == "invalid_spec", update
    with pytest.raises((ValidationError, AnalysisError)):
        ModelSpec(estimator="xtdpd", outcome="y", predictors=["x"], panel="id")   # time required
    with pytest.raises((ValidationError, AnalysisError)):
        ModelSpec(estimator="xtdpd", outcome="y", predictors=["x"], panel="id", time="year",
                  options={"twostep": "yes"})


def test_data_problems_raise_named_errors(panel):
    base = dict(y="y", x=["x"], panel="id", time="year")
    duplicated = pd.concat([panel, panel.iloc[:1]], ignore_index=True)
    assert _code(data=duplicated, **base) == "repeated_time_values"
    assert _code(data=panel.assign(year=panel.year + 0.5 * (panel.id == 100)),
                 **base) == "invalid_time"
    short = panel[panel.year <= panel.year.min() + 1]
    assert _code(data=short, **base) == "insufficient_observations"
    sparse = pd.DataFrame({"id": [1, 1, 1, 2, 2, 2], "year": [1, 2, 3, 1, 2, 10 ** 8],
                           "y": [1.0, 2.0, 1.5, 0.3, 0.1, 0.2], "x": [0.1, 0.4, 0.2, 1.0, 0.6, 0.1]})
    assert _code(data=sparse, **base) == "time_span_too_large"
    rng = np.random.default_rng(0)
    long = pd.DataFrame({"id": np.repeat(np.arange(4), 70), "year": np.tile(np.arange(70), 4),
                         "y": rng.normal(size=280), "x": rng.normal(size=280)})
    assert _code(data=long, **base) == "instrument_count_too_large"
    static = dict(lags=0, gmm=[], iv=[{"columns": ["x"]}])
    assert _code(data=panel.assign(y=2 * panel.x), **base, **static) == "perfect_fit"
    assert _code(data=panel.assign(y=panel.id * 1.0), **base, **static) == "perfect_fit"


def test_datetime_periods_are_ranked_with_a_warning(panel):
    dated = panel.assign(year=pd.to_datetime(panel.year.astype(str) + "-06-30"))
    result = oe.xtdpd(data=dated, y="y", x=["x"], panel="id", time="year")
    assert any("treated as consecutive periods" in warning for warning in result.warnings)
    assert result.nobs > 0


def test_missing_policy(panel):
    holes = panel.copy()
    holes.loc[[3, 17, 40], "x"] = np.nan
    base = dict(y="y", x=["x"], panel="id", time="year", twostep=True, robust=True)
    assert _code(data=holes, **base) == "missing_values"
    dropped = oe.xtdpd(data=holes, missing="drop", **base)
    manual = oe.xtdpd(data=panel.drop(index=[3, 17, 40]), **base)
    assert_allclose(coefs(dropped), coefs(manual), rtol=1e-12)
    assert any("missing model inputs" in warning for warning in dropped.warnings)
    assert not {3, 17, 40} & set(dropped.sample_positions)


def test_sample_positions_and_counts(panel):
    diff = oe.xtdpd(data=panel, y="y", x=["x"], panel="id", time="year")
    system = oe.xtdpd(data=panel, y="y", x=["x"], panel="id", time="year", system=True)
    assert diff.nobs == len(diff.sample_positions) == diff.extra["n_obs_transformed"]
    assert system.nobs == len(system.sample_positions) == system.extra["n_obs_level"]
    assert system.nobs > diff.nobs
    sorted_panel = panel.loc[system.sample_positions]
    counts = sorted_panel.groupby("id").size()
    assert system.metrics["n_groups"] == len(counts)
    assert system.metrics["obs_per_group_min"] == counts.min()
    assert system.metrics["obs_per_group_max"] == counts.max()
    assert_allclose(system.metrics["obs_per_group_avg"], counts.mean())
    assert diff.extra["constant"] == "differenced out"
    assert system.extra["constant"] == "level equation"


def test_string_panel_labels_and_row_order_do_not_matter(panel):
    labelled = panel.assign(id=panel.id.map(lambda v: f"firm{v}"))
    a = oe.xtdpd(data=labelled, y="y", x=["x"], panel="id", time="year", twostep=True)
    b = oe.xtdpd(data=panel.sort_values(["year", "id"]), y="y", x=["x"], panel="id",
                 time="year", twostep=True)
    assert_allclose(coefs(a), coefs(b), rtol=1e-12)


def test_many_instruments_warn_and_use_a_generalized_inverse():
    small = simulate(seed=4, n=12, periods=8, gaps=False)
    result = oe.xtdpd(data=small, y="y", x=["x"], panel="id", time="year", twostep=True,
                      robust=True)
    assert result.metrics["n_instruments"] > result.metrics["n_groups"]
    assert any("exceeds the number of panels" in warning for warning in result.warnings)
    assert any("two-step weight matrix is singular" in warning for warning in result.warnings)
    assert result.extra["weight_matrix_rank"]["two_step"] <= result.metrics["n_groups"]
    assert np.all(np.isfinite(coefs(result)))


def test_serialization_rendering_and_exports(panel):
    configurations = [{}, {"robust": True}, {"twostep": True, "robust": True, "system": True},
                      {"orthogonal": True, "collapse": True, "small": True, "robust": True},
                      {"time_dummies": True, "system": True}]
    for options in configurations:
        result = oe.xtdpd(data=panel, y="y", x=["x"], panel="id", time="year", **options)
        assert type(result).model_validate_json(result.model_dump_json()) == result
        json.loads(result.model_dump_json())
        assert result.provenance["stata_parity_validated"] is False
        assert result.provenance["family"] == "dpanel"
    for wrapper in (oe.xtabond, oe.xtdpdsys):
        result = wrapper(data=panel, y="y", x=["x"], panel="id", time="year", twostep=True)
        assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    result = oe.xtdpd(data=panel, y="y", x=["x"], panel="id", time="year", system=True,
                      twostep=True, robust=True, time_dummies=True)
    text = result.summary()
    assert "Two-step system GMM (dynamic panel data) — y" in text
    assert "Arellano-Bond test for AR(2) in first differences: normal" in text
    assert "Hansen test of overidentifying restrictions (robust): chi2(" in text
    assert "Difference-in-Hansen (null: exogenous): GMM instruments for levels" in text
    latex = str(result.to_latex())
    assert "begin{tabular}" in latex and "L1.y" in latex
    assert callable(oe.xtdpd) and callable(oe.xtabond) and callable(oe.xtdpdsys)
    assert {"xtdpd", "xtabond", "xtdpdsys"} <= set(dir(oe))
    capability = oe.capabilities()["estimators"]["xtdpd"]
    assert capability["panel"] == "required" and capability["time"] == "required"
    assert capability["covariances"] == ["nonrobust", "robust"]
    assert capability["options"]["h"]["default"] == 3
    spec = ModelSpec(estimator="xtdpd", outcome="y", predictors=["x"], panel="id", time="year",
                     options={"system": True, "twostep": True})
    fitted = oe.fit(spec, data=panel)
    assert fitted.title == "Two-step system GMM (dynamic panel data)"


def test_fifty_thousand_panels_run_in_seconds():
    rng = np.random.default_rng(1)
    n, periods = 50_000, 8
    u = rng.normal(size=n)
    y = 3 * u + rng.normal(size=n)
    frames = []
    for t in range(periods):
        x = rng.normal(size=n) + 0.5 * u
        y = 0.5 * y + x + u + rng.normal(size=n)
        frames.append(pd.DataFrame({"id": np.arange(n), "t": t, "y": y, "x": x}))
    data = pd.concat(frames, ignore_index=True)
    start = time.perf_counter()
    result = oe.xtdpd(data=data, y="y", x=["x"], panel="id", time="t", system=True,
                      twostep=True, robust=True, collapse=True)
    elapsed = time.perf_counter() - start
    assert elapsed < 20, elapsed
    assert result.metrics["n_groups"] == n and result.nobs == n * (periods - 1)
    table = {c.term: c.estimate for c in result.coefficients}
    assert abs(table["L1.y"] - 0.5) < 0.02 and abs(table["x"] - 1.0) < 0.02
