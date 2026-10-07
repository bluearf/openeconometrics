"""Explicit advanced targets, independent values/delta oracles and disk restore."""

import gc
import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle
from openecon.econometrics.postest.prediction import _model


FAMILIES = [
    "nl",
    "sureg",
    "mvreg",
    "reg3",
    "threshold",
    "biprobit",
    "heckman",
    "heckprobit",
    "churdle",
    "ivprobit",
    "ivtobit",
    "frontier",
]


def make_fit(name):
    rng = np.random.default_rng(102807)
    d = pd.DataFrame({"x": rng.normal(size=250), "z": rng.normal(size=250)})
    d["y"] = 1 + 0.5 * d.x + 0.2 * d.z + rng.normal(size=len(d))
    d["y2"] = -0.3 + 0.4 * d.x + rng.normal(size=len(d))
    if name == "nl":
        d.y = 1 + 0.5 * d.x**2 + 0.2 * d.z + rng.normal(size=len(d))
        r = oe.nl(data=d, y="y", formula="{c}+{b}*x^2+{z}*z")
        target = "function"
    elif name in {"sureg", "reg3", "mvreg"}:
        if name == "mvreg":
            r = oe.mvreg(data=d, y=["y", "y2"], x=["x", "z"])
        else:
            options = {"method": "ols"} if name == "reg3" else {}
            r = getattr(oe, name)(
                data=d, equations=[{"y": "y", "x": ["x", "z"]}, {"y": "y2", "x": ["x"]}], **options
            )
        target = "y2"
    elif name == "threshold":
        d.y = np.where(d.z <= 0, 1 + 0.5 * d.x, -1 - 0.7 * d.x) + rng.normal(size=len(d)) * 0.2
        r = oe.threshold(data=d, y="y", x=["x"], threshold_var="z")
        target = "regime_mean"
    elif name == "biprobit":
        e = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], len(d))
        d.y = (0.2 + 0.5 * d.x + e[:, 0] > 0).astype(float)
        d.y2 = (-0.1 + 0.3 * d.z + e[:, 1] > 0).astype(float)
        r = oe.biprobit(data=d, y1="y", y2="y2", x=["x"], x2=["z"])
        target = "joint11"
    elif name in {"heckman", "heckprobit"}:
        from test_econ_limited_selection import data, heckman, heckprobit

        d = data.__wrapped__()
        r = heckman(d) if name == "heckman" else heckprobit(d)
        target = "conditional" if name == "heckman" else "conditional1"
    elif name == "churdle":
        from test_econ_count_hurdle import data, X, Z

        d = data.__wrapped__()
        r = oe.churdle(data=d, y="amount", x=X, select_x=Z)
        target = "mean"
    elif name in {"ivprobit", "ivtobit"}:
        from test_econ_limited_iv import data, ivprobit, ivtobit

        d = data.__wrapped__()
        r = ivprobit(d) if name == "ivprobit" else ivtobit(d)
        target = "structural"
    else:
        from test_econ_systems_frontier import make_frontier

        d = make_frontier()
        r = oe.frontier(data=d, y="yh", x=["x1", "x2"])
        target = "te"
    return r, d, target


@pytest.fixture(scope="module", params=FAMILIES)
def fitted(request):
    return make_fit(request.param)


def oracle(r, d, target, parameters=None):
    p = {c.term: c.estimate for c in r.coefficients} if parameters is None else parameters

    def index(prefix="", names=None):
        names = r.spec.predictors if names is None else names
        return p.get(prefix + "Intercept", 0) + sum(
            p.get(prefix + x, 0) * d[x].to_numpy() for x in names
        )

    name = r.spec.estimator
    if name == "nl":
        return p["c"] + p["b"] * d.x.to_numpy() ** 2 + p["z"] * d.z.to_numpy()
    if name in {"sureg", "mvreg", "reg3"}:
        definition = (
            {"x": r.spec.predictors}
            if name == "mvreg"
            else next(e for e in r.spec.options["equations"] if e["y"] == target)
        )
        return index(target + ":", definition["x"])
    if name == "threshold":
        region = np.searchsorted(r.extra["thresholds"], d.z.to_numpy(), side="left")
        return np.array(
            [
                p[f"region{i + 1}:Intercept"] + p[f"region{i + 1}:x"] * x
                for i, x in zip(region, d.x, strict=True)
            ]
        )
    if name in {"biprobit", "heckprobit"}:
        from test_econ_limited_selection import phi2

        a = index(r.spec.outcome + ":") if name == "biprobit" else index()
        b = (
            index(r.spec.columns["outcome2"] + ":", r.spec.columns["predictors2"])
            if name == "biprobit"
            else index("select:", r.spec.columns["select_x"])
        )
        joint = phi2(a, b, np.tanh(p["/athrho"]))
        return joint / special.ndtr(b) if target == "conditional1" else joint
    if name == "heckman":
        a, b = index(), index("select:", r.spec.columns["select_x"])
        return a + np.tanh(p["/athrho"]) * np.exp(p["/lnsigma"]) * stats.norm.pdf(b) / special.ndtr(
            b
        )
    if name == "churdle":
        a, b = index(), index("select:", r.spec.columns["select_x"])
        return special.ndtr(b) * np.exp(a + 0.5 * np.exp(2 * p["/lnsigma"]))
    if name in {"ivprobit", "ivtobit"}:
        a = index(names=[*r.spec.predictors, *r.spec.columns["endogenous"]])
        if name == "ivprobit":
            return special.ndtr(a)
        sigma = np.exp(p["/lnsigma1"])
        low, high = r.extra["limits"]["lower"], r.extra["limits"]["upper"]
        left, right = stats.norm.cdf((low - a) / sigma), stats.norm.sf((high - a) / sigma)
        return (
            a * (1 - left - right)
            + sigma * (stats.norm.pdf((low - a) / sigma) - stats.norm.pdf((high - a) / sigma))
            + low * left
            + high * right
        )
    a = index()
    distribution = r.extra["distribution"]
    mu = 0.0
    if distribution == "tnormal":
        total = np.exp(p["/lnsigma2"])
        gamma = special.expit(p["/ilgtgamma"])
        u2, v2, mu = gamma * total, (1 - gamma) * total, p["/mu"]
    else:
        u2, v2 = np.exp(p["/lnsig2u"]), np.exp(p["/lnsig2v"])
    sign = -1 if r.extra["cost"] else 1
    if target == "mean":
        expected = (
            np.sqrt(u2)
            if distribution == "exponential"
            else mu
            + np.sqrt(u2) * stats.norm.pdf(mu / np.sqrt(u2)) / stats.norm.cdf(mu / np.sqrt(u2))
        )
        return a - sign * expected
    residual = d[r.spec.outcome].to_numpy() - a
    center = (
        -sign * residual - v2 / np.sqrt(u2)
        if distribution == "exponential"
        else (mu * v2 - sign * residual * u2) / (u2 + v2)
    )
    spread = np.sqrt(v2) if distribution == "exponential" else np.sqrt(u2 * v2 / (u2 + v2))
    z = center / spread
    if target == "u":
        return center + spread * np.exp(stats.norm.logpdf(z) - stats.norm.logcdf(z))
    return np.exp(
        -sign * center
        + 0.5 * spread**2
        + stats.norm.logcdf(z - sign * spread)
        - stats.norm.logcdf(z)
    )


def jacobian(r, d, target):
    p = {c.term: c.estimate for c in r.coefficients}
    columns = []
    for term, value in p.items():
        step = 1e-5 * (1 + abs(value))
        columns.append(
            (
                oracle(r, d, target, {**p, term: value + step})
                - oracle(r, d, target, {**p, term: value - step})
            )
            / (2 * step)
        )
    return np.column_stack(columns)


def test_all_families_saved_values_full_delta_permutation_and_indexed_disk(
    fitted, tmp_path, monkeypatch
):
    r, d, target = fitted
    d = d.iloc[:17].copy()
    d.index = pd.Index([f"unit-{i}" for i in range(len(d))], name="identity")
    r = ResultBundle.model_validate_json(r.model_dump_json())
    expected = oracle(r, d, target)
    j = jacobian(r, d, target)
    se = np.sqrt(np.einsum("ni,ij,nj->n", j, np.array(r.covariance_matrix), j))
    result = oe.predict(r, d, outcome=target, interval="mean")
    assert result.response.to_numpy() == pytest.approx(expected, rel=2e-9, abs=2e-10)
    assert result.std_error.to_numpy() == pytest.approx(se, rel=2e-5, abs=2e-8)
    order = list(reversed(range(len(r.coefficients))))
    r.coefficients = [r.coefficients[i] for i in order]
    r.covariance_matrix = np.array(r.covariance_matrix)[np.ix_(order, order)].tolist()
    pd.testing.assert_frame_equal(
        oe.predict(r, d, outcome=target, interval="mean"), result, rtol=1e-10, atol=1e-10
    )
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    source = oe.Dataset.from_batches(
        lambda: (d.iloc[i : i + 3] for i in range(0, len(d), 3)), list(d)
    )
    output = oe.predict(r, source, outcome=target, interval="mean", batch_rows=3)
    actual = pd.concat(list(output.iter_batches()))
    pd.testing.assert_frame_equal(
        pd.DataFrame(actual), pd.DataFrame(result), check_dtype=False, rtol=1e-10, atol=1e-10
    )
    del output
    gc.collect()
    assert not list(tmp_path.iterdir())


def test_all_families_ame_full_gradient_and_mem_global_raw_means(fitted):
    r, d, target = fitted
    d = d.iloc[:19].copy()
    model = _model(r, outcome=target)
    variable = next(x for x in model.predictors if x not in model.categories)
    actual = oe.margins(r, variable, data=d, outcome=target)
    step = 1e-4
    plus, minus = d.copy(), d.copy()
    plus[variable] += step
    minus[variable] -= step
    expected = np.mean((oracle(r, plus, target) - oracle(r, minus, target)) / (2 * step))
    assert actual.estimate.iloc[0] == pytest.approx(expected, rel=2e-6, abs=2e-7)
    parameters = {c.term: c.estimate for c in r.coefficients}

    def marginal(p):
        return np.mean((oracle(r, plus, target, p) - oracle(r, minus, target, p)) / (2 * step))

    gradient = []
    for term, value in parameters.items():
        delta = 1e-4 * (1 + abs(value))
        gradient.append(
            (
                marginal({**parameters, term: value + delta})
                - marginal({**parameters, term: value - delta})
            )
            / (2 * delta)
        )
    assert actual.attrs["delta_gradients"][0] == pytest.approx(gradient, rel=2e-4, abs=2e-7)
    assert actual.std_error.iloc[0] == pytest.approx(
        np.sqrt(np.array(gradient) @ np.array(r.covariance_matrix) @ gradient), rel=2e-4, abs=2e-8
    )
    source = oe.Dataset.from_batches(
        lambda: (d.iloc[i : i + 4] for i in range(0, len(d), 4)), list(d)
    )
    for method in ["ame", "mem"]:
        left = oe.margins(r, variable, data=d, outcome=target, method=method)
        right = oe.margins(r, variable, data=source, outcome=target, method=method, batch_rows=4)
        pd.testing.assert_frame_equal(left, right, rtol=1e-9, atol=1e-9)
    if r.spec.estimator == "nl":
        assert left.estimate.iloc[0] == pytest.approx(
            2 * next(c.estimate for c in r.coefficients if c.term == "b") * d.x.mean()
        )


def test_unknown_target_and_incomplete_state_are_structured(fitted):
    r, d, target = fitted
    with pytest.raises(AnalysisError) as error:
        oe.predict(r, d.iloc[:2], outcome="unit_mean_invented")
    assert error.value.code == "unknown_prediction_outcome"
    if r.spec.estimator in {
        "sureg",
        "mvreg",
        "reg3",
        "threshold",
        "frontier",
        "churdle",
        "ivtobit",
    }:
        bad = r.model_copy(deep=True)
        bad.extra = {}
        with pytest.raises(AnalysisError):
            oe.predict(bad, d.iloc[:2], outcome=target)


def test_explicit_bivariate_cells_conditionals_and_tail_mills():
    r, d, target = make_fit("biprobit")
    d = d.iloc[:7]
    cells = [
        oe.predict(r, d, outcome=f"joint{a}{b}").response.to_numpy()
        for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]
    ]
    assert np.sum(cells, axis=0) == pytest.approx(np.ones(len(d)), abs=1e-14)
    assert oe.predict(r, d, outcome="marginal1").response.to_numpy() == pytest.approx(
        cells[2] + cells[3], abs=1e-14
    )
    assert oe.predict(r, d, outcome="marginal2").response.to_numpy() == pytest.approx(
        cells[1] + cells[3], abs=1e-14
    )
    assert oe.predict(r, d, outcome="conditional2").response.to_numpy() == pytest.approx(
        cells[3] / (cells[2] + cells[3]), abs=1e-13
    )
    from openecon.econometrics.postest.advanced_prediction import mills
    import torch

    values = np.array([-50.0, -20.0, -12.0, -8.0, 0.0, 10.0])
    assert mills(torch.tensor(values)).numpy() == pytest.approx(
        np.exp(stats.norm.logpdf(values) - stats.norm.logcdf(values)), rel=5e-13
    )


def test_saved_constraints_and_categories_and_unknown_levels():
    rng = np.random.default_rng(102034)
    d = pd.DataFrame({"x": rng.normal(size=120), "g": pd.Categorical(np.tile(["a", "b", "c"], 40))})
    d["y"] = 0.5 * d.x + (d.g == "b") + rng.normal(size=len(d))
    d["z"] = 0.3 * d.x + rng.normal(size=len(d))
    r = oe.sureg(
        data=d,
        equations=[
            {"name": "first", "y": "y", "x": ["x", "g"]},
            {"name": "second", "y": "z", "x": ["x"]},
        ],
        categorical=["g"],
        constraints=[{"terms": {"first:Intercept": 1}, "value": 2}],
    )
    expected = 2 + next(c.estimate for c in r.coefficients if c.term == "first:x") * d.x
    for level in ["b", "c"]:
        expected += next(c.estimate for c in r.coefficients if c.term == f"first:g[{level}]") * (
            d.g == level
        )
    r = ResultBundle.model_validate_json(r.model_dump_json())
    assert oe.predict(r, d, outcome="first").response.to_numpy() == pytest.approx(
        expected.to_numpy(), abs=1e-12
    )
    for method in ["ame", "mem"]:
        left = oe.margins(r, ["x", "g"], data=d, outcome="first", method=method)
        source = oe.Dataset.from_batches(
            lambda: (d.iloc[i : i + 9] for i in range(0, len(d), 9)), list(d)
        )
        right = oe.margins(r, ["x", "g"], data=source, outcome="first", method=method, batch_rows=9)
        pd.testing.assert_frame_equal(left, right, rtol=1e-10, atol=1e-10)
    with pytest.raises(AnalysisError, match="category"):
        oe.predict(r, d.assign(g="unfitted"), outcome="first")


@pytest.mark.parametrize(
    "distribution,cost",
    [("hnormal", False), ("hnormal", True), ("tnormal", False), ("exponential", False)],
)
def test_frontier_distributions_cost_and_posterior_full_ancillary_gradient(distribution, cost):
    from test_econ_systems_frontier import make_frontier

    d = make_frontier()
    outcome = (
        "yc" if cost else {"hnormal": "yh", "tnormal": "yt", "exponential": "ye"}[distribution]
    )
    r = oe.frontier(data=d, y=outcome, x=["x1", "x2"], distribution=distribution, cost=cost)
    d = d.iloc[:5]
    for target in ["mean", "u", "te"]:
        actual = oe.predict(r, d, outcome=target, interval="mean")
        expected = oracle(r, d, target)
        gradient = jacobian(r, d, target)
        se = np.sqrt(np.einsum("ni,ij,nj->n", gradient, np.array(r.covariance_matrix), gradient))
        assert actual.response.to_numpy() == pytest.approx(expected, rel=2e-9, abs=2e-10)
        assert actual.std_error.to_numpy() == pytest.approx(se, rel=2e-5, abs=2e-8)


def test_censored_iv_extreme_tail_is_not_rounded_cdf_subtraction():
    from scipy.integrate import quad

    r, d, _ = make_fit("ivtobit")
    for c in r.coefficients:
        if c.term == "Intercept":
            c.estimate = -15.0
        elif not c.term.startswith("/"):
            c.estimate = 0.0
        elif c.term == "/lnsigma1":
            c.estimate = 0.0
    d = d.iloc[:1].copy()
    actual = oe.predict(r, d, interval="mean")
    expected = quad(lambda t: stats.norm.sf(t + 15), 0, 3, epsabs=1e-100, epsrel=1e-12)[0]
    assert actual.response.iloc[0] == pytest.approx(expected, rel=5e-11, abs=0)
    assert actual.std_error.iloc[0] > 0


def test_threshold_boundary_missing_alignment_and_owned_failure(tmp_path, monkeypatch):
    r, d, target = make_fit("threshold")
    d = d.iloc[:7].copy()
    d.z = r.extra["thresholds"][0]
    # The fitted inequality places ties in the lower region; there is no
    # derivative of the fitted discontinuity at that threshold.
    assert oe.predict(r, d, outcome=target).response.to_numpy() == pytest.approx(
        oracle(r, d, target), abs=1e-12
    )
    model = _model(r, outcome=target)
    from openecon.econometrics.postest.prediction import _encode

    with pytest.raises(AnalysisError) as error:
        model.response_adapter.effects(_encode(model, d), model.state.beta, "z", "response")
    assert error.value.code == "undefined_threshold_effect"
    with pytest.raises(AnalysisError) as error:
        oe.predict(r, d, outcome=target, kind="derivative", term="z")
    assert error.value.code == "undefined_threshold_effect"
    r.spec.missing = "drop"
    d.loc[3, "x"] = np.nan
    d.index = pd.Index(["duplicate"] * len(d), name="unit")
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    source = oe.Dataset.from_batches(
        lambda: (d.iloc[i : i + 2] for i in range(0, len(d), 2)), list(d)
    )
    output = oe.predict(r, source, outcome=target, batch_rows=2)
    result = pd.concat(list(output.iter_batches()))
    assert result.index.equals(d.index)
    assert result.response.isna().to_numpy().tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    del output
    gc.collect()
    assert list(tmp_path.iterdir()) == []
