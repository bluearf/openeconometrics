"""Independent finite enumeration, binomial/hypergeometric reductions and full-state QA."""

import copy
import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist, binom, hypergeom
import torch
import openecon as oe
from openecon.analysis_contracts import AnalysisError


def oracle(frame, family, beta, nuisance):
    """Bernoulli product space or stars-and-bars, independent of production recursion."""
    y = frame.y.to_numpy(int)
    x = frame.x.to_numpy(int)
    n = len(y)
    m = int(y.sum())
    z = frame[nuisance].to_numpy(int)
    constraint = y @ z
    if family == "logit":
        candidates = (v for v in itertools.product([0, 1], repeat=n) if sum(v) == m)
    else:

        def stars():
            for bars in itertools.combinations(range(m + n - 1), n - 1):
                edges = (-1, *bars, m + n - 1)
                yield tuple(edges[i + 1] - edges[i] - 1 for i in range(n))

        candidates = stars()
    a = np.array(
        [v for v in candidates if np.array_equal(np.array(v) @ z, constraint)], dtype=float
    )
    e = frame.e.to_numpy(float) if "e" in frame else np.ones(n)
    lb = (
        np.zeros(len(a))
        if family == "logit"
        else np.array(
            [sum(v[i] * math.log(e[i]) - math.lgamma(v[i] + 1) for i in range(n)) for v in a]
        )
    )
    s = a @ x
    logits = lb + beta * s
    p = np.exp(logits - logsumexp(logits))
    mean = p @ a
    centered = a - mean
    cov = centered.T @ (p[:, None] * centered)
    support = sorted(set(s))
    grouped = np.array([p[s == v].sum() for v in support])
    return a, s, lb, p, mean, cov, np.array(support), grouped


def fixture(family):
    return pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 0, 1] if family == "logit" else [1, 0, 2, 1],
            "x": [-1, 0, 1, -1, 0, 1] if family == "logit" else [-1, 0, 1, 2],
            "z": [0, 0, 0, 1, 1, 1] if family == "logit" else [0, 0, 1, 1],
            **({"e": [1, 2, 0.5, 3]} if family == "poisson" else {}),
        },
        index=[f"row-{i}" for i in range(6 if family == "logit" else 4)],
    )


def fit(frame, family, **kw):
    return getattr(oe, f"exact_{family}_fit")(
        frame, "y", "x", **({"exposure": "e"} if "e" in frame else {}), **kw
    )


@pytest.mark.parametrize("family", ["logit", "poisson"])
@pytest.mark.parametrize("beta", [-2.0, 0.0, 1.3])
def test_complete_distribution_moments_and_test(family, beta):
    frame = fixture(family)
    r = fit(frame, family, nuisance=["z"])
    state = r.attrs["state"]
    a, s, lb, p, mean, cov, support, grouped = oracle(frame, family, beta, ["z"])
    production = sorted(
        zip(
            map(tuple, state["allocations"]),
            state["allocation_statistics"],
            state["allocation_log_base"],
        )
    )
    reference = sorted(zip(map(tuple, a), s, lb))
    assert [v[:2] for v in production] == [v[:2] for v in reference]
    np.testing.assert_allclose([v[2] for v in production], [v[2] for v in reference], atol=2e-14)
    moment = getattr(oe, f"exact_{family}_moments")(r, coefficient=beta)
    np.testing.assert_allclose(moment["response_moments"].conditional_mean, mean, atol=2e-13)
    np.testing.assert_allclose(moment["response_covariance"], cov, atol=2e-13)
    np.testing.assert_allclose(cov @ np.ones(len(frame)), 0, atol=2e-13)
    np.testing.assert_allclose(cov @ frame.z, 0, atol=2e-13)
    assert np.linalg.eigvalsh(cov).min() > -1e-12
    test = getattr(oe, f"exact_{family}_test")(r, null=beta)
    np.testing.assert_allclose(test["null_distribution"].probability, grouped, atol=2e-13)
    observed = float(frame.y @ frame.x)
    prob = grouped[list(support).index(observed)]
    expected = grouped[np.log(grouped) <= math.log(prob) + 1e-12].sum()
    assert test["test"].p_value[0] == pytest.approx(expected, abs=2e-13)
    plugin = getattr(oe, f"exact_{family}_moments")(r)
    cmle = state["coefficient"]
    assert state["boundary"] == "interior"
    _, ss, _, _, mm, cc, _, gg = oracle(frame, family, cmle, ["z"])
    np.testing.assert_allclose(plugin["response_moments"].conditional_mean, mm, atol=2e-13)
    np.testing.assert_allclose(plugin["response_covariance"], cc, atol=2e-13)
    assert (mm @ frame.x) == pytest.approx(observed, abs=2e-12)
    independent = brentq(lambda b: oracle(frame, family, b, ["z"])[4] @ frame.x - observed, -10, 10)
    assert cmle == pytest.approx(independent, abs=1e-12)
    assert state["information"] == pytest.approx(
        gg @ ((np.array(sorted(set(ss))) - observed) ** 2), abs=2e-12
    )


@pytest.mark.parametrize("successes", [0, 1, 3, 5])
def test_poisson_binomial_reduction_fit_ci_test_moments(successes):
    m = 5
    e0, e1 = 2.0, 3.0
    frame = pd.DataFrame({"y": [m - successes, successes], "x": [0, 1], "e": [e0, e1]})
    r = fit(frame, "poisson")
    st = r.attrs["state"]
    expected = None if successes in (0, m) else math.log(successes / (m - successes) * e0 / e1)
    assert (
        st["coefficient"] == pytest.approx(expected)
        if expected is not None
        else st["coefficient"] is None
    )
    ci = oe.exact_poisson_ci(r, level=0.9)["intervals"].iloc[0]
    low = 0 if successes == 0 else beta_dist.ppf(0.05, successes, m - successes + 1)
    high = 1 if successes == m else beta_dist.ppf(0.95, successes + 1, m - successes)
    if low == 0:
        assert ci.lower_boundary == "negative_infinity" and pd.isna(ci.lower)
    else:
        assert ci.lower == pytest.approx(math.log(low / (1 - low) * e0 / e1), abs=1e-11)
    if high == 1:
        assert ci.upper_boundary == "positive_infinity" and pd.isna(ci.upper)
    else:
        assert ci.upper == pytest.approx(math.log(high / (1 - high) * e0 / e1), abs=1e-11)
    null = 0.7
    p = e1 * math.exp(null) / (e0 + e1 * math.exp(null))
    probs = binom.pmf(np.arange(m + 1), m, p)
    expected_p = probs[probs <= probs[successes] * (1 + 1e-12)].sum()
    assert oe.exact_poisson_test(r, null=null)["test"].p_value[0] == pytest.approx(
        expected_p, abs=1e-12
    )
    moments = oe.exact_poisson_moments(r, coefficient=null)
    np.testing.assert_allclose(
        moments["response_moments"].conditional_mean, [m * (1 - p), m * p], atol=1e-12
    )
    np.testing.assert_allclose(
        moments["response_covariance"], m * p * (1 - p) * np.array([[1, -1], [-1, 1]]), atol=1e-12
    )


@pytest.mark.parametrize("exposed_successes", [0, 1, 2, 3])
def test_logistic_hypergeometric_reduction(exposed_successes):
    k = exposed_successes
    m = 3
    frame = pd.DataFrame(
        {"y": [1] * k + [0] * (4 - k) + [1] * (m - k) + [0] * (4 - m + k), "x": [1] * 4 + [0] * 4}
    )
    r = fit(frame, "logit")
    test = oe.exact_logit_test(r)
    probs = hypergeom.pmf(np.arange(4), 8, 4, 3)
    np.testing.assert_allclose(test["null_distribution"].probability, probs, atol=1e-13)
    assert test["test"].p_value[0] == pytest.approx(probs[probs <= probs[k] * (1 + 1e-12)].sum())
    existing = oe.exact_logistic([[[k, 4 - k], [m - k, 4 - m + k]]])
    old = existing.attrs["state"]
    new = r.attrs["state"]
    # Compare scalar CMLE to established fixed-margin reduction without conflating p conventions.
    if new["coefficient"] is not None:
        assert new["coefficient"] == pytest.approx(old["log_odds_cmle"], abs=1e-12)
    else:
        assert old["log_odds_cmle"] is None
    ci = oe.exact_logit_ci(r)["intervals"].iloc[0]
    if k > 0:
        lo = brentq(
            lambda b: (
                sum(probs[j] * math.exp(b * j) for j in range(k, 4))
                / sum(probs[j] * math.exp(b * j) for j in range(4))
                - 0.025
            ),
            -30,
            30,
        )
        assert ci.lower == pytest.approx(lo, abs=1e-11)
    if k < 3:
        hi = brentq(
            lambda b: (
                sum(probs[j] * math.exp(b * j) for j in range(k + 1))
                / sum(probs[j] * math.exp(b * j) for j in range(4))
                - 0.025
            ),
            -30,
            30,
        )
        assert ci.upper == pytest.approx(hi, abs=1e-11)


@pytest.mark.parametrize("family", ["logit", "poisson"])
@pytest.mark.parametrize("beta", [-1.0, 0.0, 1.0])
def test_exact_conditional_size_and_coverage_all_outcomes(family, beta):
    frame = fixture(family)
    a, s, lb, p, _, _, support, grouped = oracle(frame, family, beta, ["z"])
    coverage = 0.0
    size = 0.0
    for stat in support:
        response = a[np.flatnonzero(s == stat)[0]].astype(int)
        candidate = frame.copy()
        candidate["y"] = response
        r = fit(candidate, family, nuisance=["z"])
        ci = getattr(oe, f"exact_{family}_ci")(r, level=0.8)["intervals"].iloc[0]
        covered = (pd.isna(ci.lower) or beta >= ci.lower - 1e-12) and (
            pd.isna(ci.upper) or beta <= ci.upper + 1e-12
        )
        probability = grouped[list(support).index(stat)]
        coverage += probability * covered
        size += probability * (
            getattr(oe, f"exact_{family}_test")(r, null=beta)["test"].p_value[0] <= 0.2 + 1e-13
        )
    assert coverage >= 0.8 - 1e-12
    assert size <= 0.2 + 1e-12


def test_poisson_boundary_face_has_nonuniform_allocation_weights():
    frame = pd.DataFrame({"y": [0, 0, 3], "x": [0, 0, 1], "e": [1.0, 3.0, 2.0]})
    r = fit(frame, "poisson")
    assert r.attrs["state"]["boundary"] == "upper"
    candidate = frame.copy()
    candidate["y"] = [3, 0, 0]
    lower = fit(candidate, "poisson")
    mom = oe.exact_poisson_moments(lower)
    np.testing.assert_allclose(
        mom["response_moments"].conditional_mean, [0.75, 2.25, 0], atol=1e-13
    )
    np.testing.assert_allclose(
        mom["response_covariance"],
        [[0.5625, -0.5625, 0], [-0.5625, 0.5625, 0], [0, 0, 0]],
        atol=1e-13,
    )
    assert np.count_nonzero(mom.attrs["state"]["allocation_probabilities"]) == 4


@pytest.mark.parametrize("family", ["logit", "poisson"])
def test_missing_alignment_complete_ordered_persistence_and_no_refit(family, tmp_path, monkeypatch):
    frame = fixture(family)
    frame.loc["row-1", "x"] = np.nan
    with pytest.raises(AnalysisError):
        fit(frame, family, nuisance=["z"])
    r = fit(frame, family, nuisance=["z"], missing="drop")
    results = [
        r,
        getattr(oe, f"exact_{family}_ci")(r),
        getattr(oe, f"exact_{family}_test")(r),
        getattr(oe, f"exact_{family}_moments")(r, coefficient=0.3),
    ]
    from openecon.econometrics.conditional import engine

    monkeypatch.setattr(
        engine, "fit", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("refit"))
    )
    for output in results:
        path = tmp_path / (output.attrs["procedure"] + ".json")
        saved = oe.conditional_save(output, path)
        restored = oe.conditional_load(path)
        assert list(restored) == list(output)
        assert oe.conditional_save(restored) == saved
        assert "\\begin{tabular}" in restored.to_latex()
        bad = copy.deepcopy(saved)
        bad["payload"]["attrs"]["state"]["corruption"] = 1
        with pytest.raises(AnalysisError):
            oe.conditional_load(bad)
    moments = results[-1]["response_moments"]
    assert len(moments) == len(frame)
    assert moments.row_label.tolist() == frame.index.tolist()
    assert not moments.in_sample[1] and pd.isna(moments.conditional_mean[1])
    modified = oe.conditional_load(oe.conditional_save(r))
    modified["support"].iloc[0, 0] += 1
    with pytest.raises(AnalysisError):
        getattr(oe, f"exact_{family}_test")(modified)
    other = oe.finite_load(oe.finite_save(oe.exact_poisson_rate(2, 1)))
    with pytest.raises(AnalysisError):
        getattr(oe, f"exact_{family}_test")(other)


@pytest.mark.parametrize("family", ["logit", "poisson"])
@pytest.mark.parametrize(
    "options",
    [
        {"device": "cuda"},
        {"weights": "w"},
        {"max_states": True},
        {"max_states": 20001},
        {"max_work": 1},
        {"max_states": 1},
        {"nuisance": ["z", "z"]},
        {"nuisance": ["y"]},
        {"missing": "automatic"},
    ],
)
def test_explicit_option_refusals(family, options):
    with pytest.raises(AnalysisError):
        fit(fixture(family), family, **options)


@pytest.mark.parametrize(
    "column,value", [("x", 0.5), ("x", 9), ("x", math.inf), ("y", -1), ("z", 0.5)]
)
@pytest.mark.parametrize("family", ["logit", "poisson"])
def test_invalid_input_refused(family, column, value):
    frame = fixture(family)
    frame[column] = frame[column].astype(float)
    frame.loc[frame.index[0], column] = value
    with pytest.raises(AnalysisError):
        fit(frame, family, nuisance=["z"])


@pytest.mark.parametrize("family", ["logit", "poisson"])
def test_bool_limits_default_device_and_conditioning_identification(family):
    frame = fixture(family)
    bad = frame.copy()
    bad["x"] = bad.x.astype(bool)
    with pytest.raises(AnalysisError):
        fit(bad, family)
    duplicated = pd.concat([frame] * 8)
    with pytest.raises(AnalysisError):
        fit(duplicated, family)
    with pytest.raises(AnalysisError):
        fit(frame.assign(x=1), family)
    with pytest.raises(AnalysisError):
        fit(frame.assign(z=frame.x), family, nuisance=["z"])
    r = fit(frame, family, nuisance=["z"])
    with torch.device("meta"):
        rr = fit(frame, family, nuisance=["z"])
        mm = getattr(oe, f"exact_{family}_moments")(rr, coefficient=0.2)
    assert oe.conditional_save(rr) == oe.conditional_save(r)
    assert mm.attrs["device"] == "cpu"
    for value in [True, float("nan"), 41, -41]:
        with pytest.raises(AnalysisError):
            getattr(oe, f"exact_{family}_test")(r, null=value)
    for value in [0, 1, True]:
        with pytest.raises(AnalysisError):
            getattr(oe, f"exact_{family}_ci")(r, level=value)


def test_poisson_pre_filter_gate_before_enumeration(monkeypatch):
    from openecon.econometrics.conditional import engine

    monkeypatch.setattr(
        engine, "compositions", lambda *args: (_ for _ in ()).throw(AssertionError("enumerated"))
    )
    frame = pd.DataFrame(
        {"y": [24, 0, 0, 0, 0, 0, 0, 0], "x": list(range(8)), "z": [0, 1, 0, 1, 0, 1, 0, 1]}
    )
    with pytest.raises(AnalysisError, match="structural allocations"):
        fit(frame, "poisson", nuisance=["z"])


@pytest.mark.parametrize("exposure", [0, -1, 1e-13, 1e13, float("inf")])
def test_invalid_exposure(exposure):
    frame = fixture("poisson")
    frame.loc["row-0", "e"] = exposure
    with pytest.raises(AnalysisError):
        fit(frame, "poisson")


def test_unbracketed_root_is_not_clipped():
    frame = pd.DataFrame({"y": [1, 0], "x": [0, 1], "e": [1e-12, 1e12]})
    r = fit(frame, "poisson")
    with pytest.raises(AnalysisError, match="outside"):
        oe.exact_poisson_ci(r)


@pytest.mark.parametrize("family", ["logit", "poisson"])
def test_three_distinct_nuisance_constraints(family):
    frame = pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 1, 0, 1, 0],
            "x": [-2, 1, 0, 2, 2, 0, -1, 1],
            "z1": [0, 0, 0, 0, 1, 1, 1, 1],
            "z2": [0, 1, 0, 1, 0, 1, 0, 1],
            "z3": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    names = ["z1", "z2", "z3"]
    r = fit(frame, family, nuisance=names)
    mean, cov = oracle(frame, family, 0.4, names)[4:6]
    moments = getattr(oe, f"exact_{family}_moments")(r, coefficient=0.4)
    np.testing.assert_allclose(moments["response_moments"].conditional_mean, mean, atol=1e-13)
    np.testing.assert_allclose(moments["response_covariance"], cov, atol=1e-13)
    np.testing.assert_allclose(cov @ frame[names], 0, atol=1e-13)


def test_workspace_limit_precedes_enumeration(monkeypatch):
    from openecon.econometrics.conditional import engine

    monkeypatch.setenv("OPENECON_WORKSPACE_MB", "1")
    monkeypatch.setattr(
        engine, "compositions", lambda *args: (_ for _ in ()).throw(AssertionError("enumerated"))
    )
    frame = pd.DataFrame({"y": [3, 2, 3, 2], "x": [-1, 0, 1, 2]})
    with pytest.raises(AnalysisError, match="workspace"):
        fit(frame, "poisson")


def test_no_information_and_dataset_refused():
    frame = pd.DataFrame({"y": [0, 0, 0], "x": [0, 1, 2]})
    for family in ["logit", "poisson"]:
        with pytest.raises(AnalysisError):
            fit(frame, family)
    from openecon.dataset import Dataset

    dataset = Dataset.from_frame(frame)
    with pytest.raises(AnalysisError):
        oe.exact_logit_fit(dataset, "y", "x")


def test_public_static_family_and_capability_contract():
    from openecon.econometrics.registry import FAMILIES
    from openecon.econometrics.conditional import EXPORTS, ESTIMATORS

    assert "conditional" in FAMILIES and ESTIMATORS == () and len(EXPORTS) == 10
    cap = oe.capabilities()["conditional_regression"]
    assert cap["stata_parity_validated"] is False
    assert len(cap["procedures"]) == 8 and cap["budgets"]["structural_allocations"] == 20000
    assert cap["weights"] == [] and cap["devices"] == ["cpu"]
    assert all(callable(getattr(oe, name)) for name in EXPORTS)
