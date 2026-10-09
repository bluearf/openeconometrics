"""Real saved trajectory families and independent linear-map/MVN references."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest.trajectories import trajectory_bands


ASSUMPTIONS = (
    "Prespecified targets; valid identification and exogeneity/parallel trends; "
    "many independent units or weakly dependent time observations; compatible "
    "joint normal CLT and consistent saved full covariance. IV instruments are strong."
)


def series(seed=501, n=110):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    shock = z + .3 * rng.normal(size=n)
    y = np.zeros(n)
    for j in range(1, n):
        y[j] = .4 * y[j - 1] + .7 * shock[j] + .2 * shock[j - 1] + rng.normal()
    return pd.DataFrame({"t": np.arange(n), "shock": shock, "z": z, "y": y})


def staggered():
    rng = np.random.default_rng(514)
    n, periods = 48, 9
    first = np.repeat([3, 5, periods], 16)
    ids, time = np.repeat(np.arange(n), periods), np.tile(np.arange(periods), n)
    treated = time >= first[ids]
    y = (rng.normal(size=n)[ids] + .2 * time + treated * (1 + .3 * (time - first[ids]))
         + rng.normal(size=n * periods))
    return pd.DataFrame({"id": ids, "t": time, "first": np.where(first[ids] == periods, np.nan, first[ids]),
                         "d": treated.astype(float), "y": y})


def fitted(family):
    if family in ("lp", "lpiv"):
        kwargs = {"instruments": ["z"]} if family == "lpiv" else {}
        return getattr(oe, family)(data=series(), y="y", x=["shock"], time="t", horizons=3, **kwargs)
    if family == "panel_lp":
        frame = pd.concat([series(701 + i, n=24).assign(id=i) for i in range(24)], ignore_index=True)
        return oe.panel_lp(data=frame, y="y", x=["shock"], panel="id", time="t", horizons=3)
    frame = staggered()
    if family in ("bjs", "sunab"):
        return oe.heterodid(data=frame, y="y", treatment="d", panel="id", time="t", model=family, event_max=3)
    if family == "eventstudy":
        return oe.eventstudy(data=frame, y="y", group="id", time="t", treatment_time="first", leads=2, lags=3)
    return oe.csdid(data=frame, y="y", group="id", time="t", treatment_time="first", control="never")


@pytest.fixture(scope="module", params=["lp", "lpiv", "panel_lp", "bjs", "sunab", "eventstudy", "csdid"])
def model(request):
    return fitted(request.param)


def reporting_map(n):
    a = np.zeros((3, n))
    a[0, 0] = 1
    a[1, -1] = 1
    a[2, n // 2] = 1
    a[2, 0] = -.4
    a[2, -1] = .25
    return a


def numpy_mvn_critical(covariance, labels, alpha, draws, seed):
    se = np.sqrt(np.diag(covariance))
    correlation = covariance / se[:, None] / se[None, :]
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    correlation = correlation[np.ix_(order, order)]
    eigenvalues, vectors = np.linalg.eigh(correlation)
    # Eigenvector signs are arbitrary. Align their sign only to the engine's
    # decomposition so the same explicitly seeded normal stream can be compared.
    # The eigenvalues, eigenspaces, root and quantile are computed independently.
    reference_vectors = torch.linalg.eigh(torch.tensor(correlation, dtype=torch.float64)).eigenvectors.numpy()
    for j in range(len(labels)):
        if np.dot(vectors[:, j], reference_vectors[:, j]) < 0:
            vectors[:, j] *= -1
    rng = torch.Generator(device="cpu").manual_seed(seed)
    normal = torch.randn(draws, len(labels), dtype=torch.float64, generator=rng).numpy()
    draws_z = normal @ (vectors * np.sqrt(np.maximum(eigenvalues, 0))).T
    return np.quantile(abs(draws_z).max(axis=1), 1 - alpha, method="higher")


def test_real_saved_families_linear_reporting_full_covariance_and_independent_maxz(model, monkeypatch):
    beta = np.array([c.estimate for c in model.coefficients])
    covariance = np.array(model.covariance_matrix)
    assert len(beta) >= 3
    assert np.max(abs(covariance - np.diag(np.diag(covariance)))) > 1e-7
    a = reporting_map(len(beta))
    labels = ["z-first target", "a-last target", "m-prespecified contrast"]
    expected_beta, expected_v = a @ beta, a @ covariance @ a.T
    saved = oe.ResultBundle.model_validate_json(model.model_dump_json())
    before = saved.model_dump_json()
    import openecon.analysis
    def forbid_fit(*args, **kwargs):
        raise AssertionError("Saved trajectory reporting must not refit.")
    monkeypatch.setattr(openecon.analysis, "fit", forbid_fit)
    with torch.device("meta"):
        actual = trajectory_bands(saved, contrasts=a.tolist(), labels=labels,
                                  assumptions=ASSUMPTIONS, alpha=.1, draws=7000, seed=75109)
    np.testing.assert_allclose(actual["bands"].estimate, expected_beta, atol=2e-12)
    np.testing.assert_allclose(actual["joint_covariance"], expected_v, rtol=1e-12, atol=2e-12)
    np.testing.assert_allclose(actual["reporting_map"], a, atol=0)
    np.testing.assert_allclose(actual["bands"].std_error, np.sqrt(np.diag(expected_v)), rtol=1e-12)
    critical = numpy_mvn_critical(expected_v, labels, .1, 7000, 75109)
    assert actual.attrs["critical_value"] == pytest.approx(critical, rel=5e-10, abs=3e-10)
    np.testing.assert_allclose(actual["bands"].ci_low, expected_beta - critical * np.sqrt(np.diag(expected_v)), atol=3e-10)
    np.testing.assert_allclose(actual["bands"].ci_high, expected_beta + critical * np.sqrt(np.diag(expected_v)), atol=3e-10)
    assert actual.attrs["simulation_order_positions"] == [1, 2, 0]
    assert actual.attrs["source_inference"] == saved.inference
    assert actual.attrs["source_spec"] == saved.spec.model_dump(mode="json")
    assert actual.attrs["refitted"] is False
    assert actual.attrs["device"] == "cpu"
    assert actual.attrs["cdf_monte_carlo_std_error"] == pytest.approx(np.sqrt(.1 * .9 / 7000))
    assert "few-cluster" in " ".join(actual.attrs["excludes"])
    assert "weak-IV" in " ".join(actual.attrs["excludes"])
    assert saved.model_dump_json() == before
    restored = oe.restore_summary(oe.summary_state(actual))
    np.testing.assert_allclose(restored["bands"].iloc[:, 1:], actual["bands"].iloc[:, 1:], atol=0)
    np.testing.assert_allclose(restored["joint_covariance"], expected_v, atol=2e-12)
    assert restored.attrs == actual.attrs


def test_subset_and_permutation_have_saved_cross_target_covariance_and_seed_invariance(model):
    names = [c.term for c in model.coefficients]
    terms = [names[-1], names[0]]
    one = trajectory_bands(model, terms=terms, assumptions=ASSUMPTIONS, draws=4000, seed=159)
    two = trajectory_bands(model, terms=list(reversed(terms)), assumptions=ASSUMPTIONS, draws=4000, seed=159)
    assert one.attrs["critical_value"] == two.attrs["critical_value"]
    np.testing.assert_allclose(one["bands"].iloc[:, 1:], two["bands"].iloc[::-1, 1:], atol=0)
    v = np.array(model.covariance_matrix)
    np.testing.assert_allclose(one["joint_covariance"], v[np.ix_([len(names) - 1, 0], [len(names) - 1, 0])], atol=0)
    assert one.attrs["family_size"] == 2
    assert one["bands"].hypothesis.tolist() == terms
    assert one.attrs["seed_order_policy"] == "Canonical unique explicit labels"


@pytest.fixture(scope="module")
def lp_model():
    return fitted("lp")


@pytest.mark.parametrize("assumptions", [None, "", "   ", "a" * 4001, ["CLT"]])
def test_assumptions_are_required_and_bounded(lp_model, assumptions):
    with pytest.raises(AnalysisError) as caught:
        trajectory_bands(lp_model, assumptions=assumptions, draws=1000)
    assert caught.value.code == "missing_assumptions"


@pytest.mark.parametrize("case", ["none_se", "malformed_v", "nan_beta", "nan_v", "mismatch_se", "asymmetric_discarded", "indefinite_discarded", "zero_se"])
def test_entire_saved_state_is_validated_before_selecting_a_subset(lp_model, case):
    corrupted = lp_model.model_copy(deep=True)
    n = len(corrupted.coefficients)
    if case == "none_se":
        corrupted.coefficients[-1].std_error = None
    elif case == "malformed_v":
        corrupted.covariance_matrix[-1].append(0.)
    elif case == "nan_beta":
        corrupted.coefficients[-1].estimate = float("nan")
    elif case == "nan_v":
        corrupted.covariance_matrix[-1][0] = float("nan")
    elif case == "mismatch_se":
        corrupted.coefficients[-1].std_error *= 2
    elif case == "zero_se":
        corrupted.coefficients[-1].std_error = 0.
        corrupted.covariance_matrix[-1][-1] = 0.
    elif case == "asymmetric_discarded":
        corrupted.covariance_matrix[-1][-2] *= 2
    else:
        value = 1.2 * np.sqrt(corrupted.covariance_matrix[-1][-1] * corrupted.covariance_matrix[-2][-2])
        corrupted.covariance_matrix[-1][-2] = value
        corrupted.covariance_matrix[-2][-1] = value
    with pytest.raises(AnalysisError) as caught:
        trajectory_bands(corrupted, terms=[corrupted.coefficients[0].term], assumptions=ASSUMPTIONS, draws=1000)
    assert caught.value.code == "invalid_result_state"
    assert n == 4


@pytest.mark.parametrize("kind", ["duplicate_terms", "absent_terms", "empty_terms", "label_without_map", "both_map_and_terms", "wrong_width", "duplicate_labels", "boolean_map", "nonfinite_map", "zero_map"])
def test_invalid_prespecified_family_and_reporting_map(lp_model, kind):
    names = [c.term for c in lp_model.coefficients]
    n = len(names)
    kwargs = {}
    if kind == "duplicate_terms":
        kwargs["terms"] = [names[0], names[0]]
    elif kind == "absent_terms":
        kwargs["terms"] = ["missing"]
    elif kind == "empty_terms":
        kwargs["terms"] = []
    elif kind == "label_without_map":
        kwargs["labels"] = ["target"]
    else:
        kwargs = {"contrasts": [[1.] + [0.] * (n - 1)], "labels": ["target"]}
        if kind == "both_map_and_terms":
            kwargs["terms"] = [names[0]]
        elif kind == "wrong_width":
            kwargs["contrasts"][0].append(0.)
        elif kind == "duplicate_labels":
            kwargs["contrasts"] *= 2
            kwargs["labels"] *= 2
        elif kind == "boolean_map":
            kwargs["contrasts"][0][0] = True
        elif kind == "nonfinite_map":
            kwargs["contrasts"][0][0] = float("inf")
        elif kind == "zero_map":
            kwargs["contrasts"][0] = [0.] * n
    with pytest.raises(AnalysisError) as caught:
        trajectory_bands(lp_model, assumptions=ASSUMPTIONS, draws=1000, **kwargs)
    assert caught.value.code == ("invalid_covariance" if kind == "zero_map" else "invalid_family")


@pytest.mark.parametrize("option,value,code", [
    ("draws", 999, "invalid_draws"), ("draws", True, "invalid_draws"),
    ("draws", 200001, "invalid_draws"), ("seed", -1, "invalid_seed"),
    ("seed", True, "invalid_seed"), ("alpha", 0, "invalid_alpha"),
])
def test_simulation_option_refusals(lp_model, option, value, code):
    kwargs = {"assumptions": ASSUMPTIONS, "draws": 1000, option: value}
    with pytest.raises(AnalysisError) as caught:
        trajectory_bands(lp_model, **kwargs)
    assert caught.value.code == code


def test_work_and_workspace_gates_precede_simulation(lp_model, monkeypatch):
    n = len(lp_model.coefficients)
    mapping = [[1.] + [0.] * (n - 1)] * 41
    with pytest.raises(AnalysisError) as caught:
        trajectory_bands(lp_model, contrasts=mapping, labels=[str(i) for i in range(41)],
                         assumptions=ASSUMPTIONS, draws=200000)
    assert caught.value.code == "work_budget"
    monkeypatch.setenv("OPENECON_WORKSPACE_MB", "1")
    with pytest.raises(AnalysisError) as caught:
        trajectory_bands(lp_model, assumptions=ASSUMPTIONS, draws=50000)
    assert caught.value.code == "workspace_limit"


def test_unsupported_source_model_refused(lp_model):
    unsupported = lp_model.model_copy(deep=True)
    unsupported.spec.estimator = "ols"
    with pytest.raises(AnalysisError) as caught:
        trajectory_bands(unsupported, assumptions=ASSUMPTIONS, draws=1000)
    assert caught.value.code == "unsupported_model"


@pytest.mark.parametrize("duplicate", [False, True])
def test_saved_full_family_cardinality_and_unique_term_gate(lp_model, duplicate):
    changed = lp_model.model_copy(deep=True)
    if duplicate:
        changed.coefficients[1].term = changed.coefficients[0].term
    else:
        changed.coefficients = [changed.coefficients[0].model_copy(update={"term": f"target_{i}"}) for i in range(385)]
    with pytest.raises(AnalysisError) as caught:
        trajectory_bands(changed, assumptions=ASSUMPTIONS, draws=1000)
    assert caught.value.code == "work_budget"


def test_simulation_rng_is_local_and_exactly_repeatable(lp_model):
    before = torch.get_rng_state().clone()
    one = trajectory_bands(lp_model, assumptions=ASSUMPTIONS, draws=1000, seed=44)
    two = trajectory_bands(lp_model, assumptions=ASSUMPTIONS, draws=1000, seed=44)
    assert torch.equal(torch.get_rng_state(), before)
    assert one.attrs["critical_value"] == two.attrs["critical_value"]
    np.testing.assert_allclose(one["bands"].iloc[:, 1:], two["bands"].iloc[:, 1:], atol=0)
