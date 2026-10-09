"""Independent known-DGP identities, score covariance and censor-integral oracles."""

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.causal_design.common import causal_design_load, causal_design_save
from openecon.econometrics.causal_design.observational_survival import (
    treatment_rmst_aipw,
    treatment_rmst_ipcw,
    treatment_survival_ipcw,
)
from openecon.resources import use_workspace_budget


GRID = [0.0, 0.5, 1.5, 3.0]


def sample():
    data = pd.DataFrame(
        {
            "u": [0.0, 0.5, 2.0, 4.0, 0.25, 1.0, 2.5, 5.0],
            "event": [1, 0, 1, 0, 1, 1, 0, 1],
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "p": [0.2, 0.4, 0.6, 0.3, 0.7, 0.6, 0.4, 0.8],
            "rate": [0.1, 0.3, 0.2, 0.0, 0.4, 0.2, 0.25, 0.1],
            "m0": [0.1, 0.2, 0.8, 1.2, 0.4, 1.0, 0.6, 1.5],
            "m1": [0.5, 0.7, 1.3, 1.0, 0.3, 1.2, 0.7, 2.0],
        },
        index=pd.Index(["same", "same", True, 2**60 + 1, "a", "b", "c", "d"], dtype=object),
    )
    for j, threshold in enumerate(GRID):
        data[f"g{j}"] = np.exp(-data.rate * threshold)
    return data


def fit(name, data=None, **options):
    data = sample() if data is None else data
    settings = {"design": "unconfounded", "nuisance": "known", **options}
    if name == "survival":
        settings.setdefault("thresholds", GRID)
        return treatment_survival_ipcw(
            data, "u", "event", "a", "p", ["g0", "g1", "g2", "g3"], **settings
        )
    settings.setdefault("tau", 3.0)
    if name == "rmst":
        return treatment_rmst_ipcw(data, "u", "event", "a", "p", "rate", **settings)
    # The default AIPW fixture makes every capped time fully observed.
    if isinstance(data, pd.DataFrame) and "complete" not in data:
        data = data.assign(event=1)
    return treatment_rmst_aipw(data, "u", "event", "a", "p", "m0", "m1", **settings)


def oracle_covariance(scores):
    n, dimension = scores.shape
    means = np.array([math.fsum(scores[:, j]) / n for j in range(dimension)])
    covariance = np.array(
        [
            [
                math.fsum((scores[:, j] - means[j]) * (scores[:, k] - means[k])) / (n * (n - 1))
                for k in range(dimension)
            ]
            for j in range(dimension)
        ]
    )
    return means, covariance


def assert_inference(result, scores, level=0.95):
    means, covariance = oracle_covariance(scores)
    np.testing.assert_allclose(result["scores"], scores, rtol=5e-14, atol=1e-14)
    np.testing.assert_allclose(result["effects"].estimate, means, rtol=5e-14, atol=1e-14)
    np.testing.assert_allclose(result["covariance"], covariance, rtol=2e-13, atol=2e-14)
    errors = np.sqrt(covariance.diagonal())
    critical = norm.isf((1 - level) / 2)
    np.testing.assert_allclose(result["effects"].std_error, errors, rtol=2e-13)
    np.testing.assert_allclose(
        result["effects"].ci_low, means - critical * errors, rtol=2e-13, atol=1e-14
    )
    np.testing.assert_allclose(
        result["effects"].ci_high, means + critical * errors, rtol=2e-13, atol=1e-14
    )
    for j, row in enumerate(result["effects"].itertuples()):
        if errors[j]:
            assert row.p_value == pytest.approx(2 * norm.sf(abs(means[j] / errors[j])), rel=2e-12)
        assert pd.isna(row.df)
    return means, covariance


@pytest.mark.parametrize("seed", range(5))
def test_fixed_grid_all_scores_full_cross_time_and_cross_arm_covariance(seed):
    rng = np.random.default_rng(seed)
    data = sample()
    data["p"] = rng.uniform(0.15, 0.85, len(data))
    data["rate"] = rng.uniform(0.0, 0.5, len(data))
    for j, t in enumerate(GRID):
        data[f"g{j}"] = np.exp(-data.rate * t)
    untouched = data.copy(deep=True)
    result = fit("survival", data, level=0.9)
    p = np.where(data.a == 1, data.p, 1 - data.p)
    score = (
        (data.u.to_numpy()[:, None] > np.array(GRID)[None, :])
        / p[:, None]
        / data[["g0", "g1", "g2", "g3"]].to_numpy()
    )
    s0, s1 = (
        np.where(data.a.to_numpy()[:, None] == 0, score, 0),
        np.where(data.a.to_numpy()[:, None] == 1, score, 0),
    )
    scores = np.stack((s0, s1, s1 - s0), 2).reshape(len(data), -1)
    means, covariance = assert_inference(result, scores, level=0.9)
    assert covariance[0, 1] < 0  # HT means do not condition on arm counts.
    assert covariance[0, 4] < 0
    for j in range(len(GRID)):
        assert covariance[3 * j + 2, 3 * j + 2] == pytest.approx(
            covariance[3 * j, 3 * j]
            + covariance[3 * j + 1, 3 * j + 1]
            - 2 * covariance[3 * j, 3 * j + 1]
        )
    np.testing.assert_allclose(result["curve"].iloc[:, 1:], means.reshape(-1, 3))
    pd.testing.assert_frame_equal(data, untouched)
    assert result.attrs["positions"] == list(range(len(data)))
    assert result.attrs["unit_labels"][0] == result.attrs["unit_labels"][1]
    assert result.attrs["unit_labels"][3] == 2**60 + 1


def test_strict_survival_indicator_uses_u_not_event_and_never_repairs_ht_curve():
    data = sample().assign(u=10.0, p=0.2)
    result = fit("survival", data)
    assert result["curve"].survival_treated.max() > 1
    assert result.attrs["state"]["sample_curve_increases"] == [True, True]
    assert result.attrs["state"]["sample_curve_outside_probability_range"] is True
    pd.testing.assert_frame_equal(
        result["scores"], fit("survival", data.assign(event=1 - data.event))["scores"]
    )
    tied = fit("survival", sample())
    for j, t in enumerate(GRID):
        expected = (sample().u > t).astype(float).tolist()
        assert [row[j] for row in tied.attrs["state"]["observed_survival_indicators"]] == expected


def test_survival_ht_identity_for_exhaustive_known_discrete_dgp_and_censor_ties():
    rows, laws = [], []
    grid = [0.0, 0.25, 1.0, 2.0]
    target0 = np.zeros(4)
    target1 = np.zeros(4)
    for x, b, a, k in itertools.product(range(2), range(3), range(2), range(3)):
        p = [0.3, 0.7][x]
        t0, t1 = [0.0, 0.5, 2.5][b] + 0.1 * x, [0.25, 1.0, 3.0][b] + 0.1 * x
        censor = [0.25, 1.0, 4.0][k]
        cb = [0.2 + 0.05 * x, 0.3 + 0.05 * a, 0.5 - 0.05 * x - 0.05 * a]
        probability = 0.5 / 3 * (p if a else 1 - p) * cb[k]
        time = t1 if a else t0
        g = [sum(q for value, q in zip([0.25, 1.0, 4.0], cb) if value > t) for t in grid]
        rows.append([min(time, censor), int(time <= censor), a, p, *g])
        laws.append(probability)
        if a == 0 and k == 0:
            target0 += 0.5 / 3 * (np.array([t0] * 4) > grid)
            target1 += 0.5 / 3 * (np.array([t1] * 4) > grid)
    data = pd.DataFrame(rows, columns=["u", "event", "a", "p", "g0", "g1", "g2", "g3"])
    result = fit("survival", data, thresholds=grid)
    mean_under_dgp = np.asarray(laws) @ result["scores"].to_numpy()
    np.testing.assert_allclose(
        mean_under_dgp.reshape(-1, 3),
        np.stack((target0, target1, target1 - target0), 1),
        atol=2e-15,
    )
    assert sum(laws) == pytest.approx(1)


@pytest.mark.parametrize("tau", [1.0, 3.0, 6.0])
def test_exponential_rmst_native_closed_form_matches_dense_quadrature_and_full_covariance(tau):
    data = sample()
    # Zero rate is a no-censoring-through-tau submodel.
    data.loc[data.rate == 0, "event"] = 1
    result = fit("rmst", data, tau=tau)
    integrals = np.array(
        [
            quad(lambda t: math.exp(rate * t), 0, min(time, tau), epsabs=1e-12)[0]
            for time, rate in zip(data.u, data.rate)
        ]
    )
    probability = np.where(data.a == 1, data.p, 1 - data.p)
    adjusted = integrals / probability
    s0, s1 = np.where(data.a == 0, adjusted, 0), np.where(data.a == 1, adjusted, 0)
    assert_inference(result, np.stack((s0, s1, s1 - s0), 1))
    np.testing.assert_allclose(result["subjects"].censor_adjusted_integral, integrals, rtol=2e-14)
    np.testing.assert_allclose(
        result["subjects"].censor_survival_tau_left, np.exp(-data.rate * tau)
    )


@pytest.mark.parametrize("rate", [0.0, 1e-14, 0.1, 2.0])
@pytest.mark.parametrize("time", [0.0, 0.25, 2.0, 5.0])
def test_exponential_ipcw_integral_expected_over_censoring_is_true_capped_time(rate, time):
    tau = 3.0
    data = sample().assign(u=time, event=1, rate=rate)
    result = fit("rmst", data, tau=tau, positivity=1e-9)
    h = float(result["subjects"].iloc[0].censor_adjusted_integral)
    u = min(time, tau)
    expected_h = u if rate == 0 else math.expm1(rate * u) / rate
    assert h == pytest.approx(expected_h, rel=3e-14, abs=1e-15)
    # Independently integrate the exact continuous censoring law, including
    # its mass beyond the true capped event time; treatment expectation cancels p.
    if rate:
        population_mean = quad(
            lambda c: math.expm1(rate * c) / rate * rate * math.exp(-rate * c), 0, u
        )[0] + h * math.exp(-rate * u)
    else:
        population_mean = h
    assert population_mean == pytest.approx(u, rel=3e-14, abs=1e-15)


def test_rmst_ht_can_exceed_tau_without_clipping_and_zero_rate_is_exact_limit():
    data = sample().assign(u=4.0, event=1, rate=0.0, p=0.1)
    result = fit("rmst", data)
    assert result["effects"].iloc[1].estimate == 15.0
    assert result.attrs["state"]["subject_integrals"] == [3.0] * len(data)
    np.testing.assert_allclose(result["subjects"].censor_survival_tau_left, 1.0)


@pytest.mark.parametrize("rotation", range(6))
def test_nested_cancellation_preserves_nonzero_contrast_with_native_expansion(rotation):
    data = pd.DataFrame(
        {
            "u": [5e99, 5e-101, 5e15, 5e99, 5e15, 0.0],
            "a": [1, 1, 1, 0, 0, 0],
            "p": [0.5] * 6,
            "event": [1] * 6,
            "rate": [0.0] * 6,
        }
    )
    data = data.iloc[list(range(rotation, 6)) + list(range(rotation))]
    result = fit("rmst", data, tau=1e101)
    contrast_scores = np.where(data.a == 1, 2 * data.u, -2 * data.u)
    expected = math.fsum(contrast_scores) / 6
    assert expected != 0
    assert result["effects"].iloc[2].estimate == expected
    assert result.attrs["state"]["score_sum_capacity"] == 64


def test_expansion_capacity_gate_is_explicit(monkeypatch):
    import openecon.econometrics.causal_design.observational_survival as module

    # Capacity is a declared structural bound, never truncation of residuals.
    monkeypatch.setattr(module, "_SUM_CAPACITY", 1)
    data = sample().assign(u=[1e50, 1.0, 1e30, 1e50, 1e30, 1.0, 0.0, 0.0], event=1, rate=0.0, p=0.5)
    with pytest.raises(AnalysisError, match="capacity"):
        fit("rmst", data, tau=1e51)


def test_zero_rate_administrative_censoring_at_tau_matches_capped_time_identity():
    result = fit("rmst", sample().assign(u=3.0, event=0, rate=0.0))
    assert result.attrs["state"]["subject_integrals"] == [3.0] * 8
    assert result.attrs["state"]["censor_survival_tau_left"] == [1.0] * 8
    assert "administrative censoring at tau allowed" in result.attrs["state"]["censor_model"]


def test_complete_horizon_aipw_all_score_algebra_covariance_and_imperfect_augmentation():
    data = sample().assign(event=1)
    result = fit("aipw", data)
    z, a, p = np.minimum(data.u, 3.0), data.a, data.p
    s0 = data.m0 + (a == 0) * (z - data.m0) / (1 - p)
    s1 = data.m1 + (a == 1) * (z - data.m1) / p
    assert_inference(result, np.stack((s0, s1, s1 - s0), 1))
    assert result.attrs["state"]["estimated_nuisance_supported"] is False
    assert result.attrs["state"]["estimated_nuisance_double_robust_inference"] is False
    assert not np.allclose(data.m0, z)


def test_aipw_expected_score_for_exhaustive_known_treatment_dgp_with_incorrect_external_m():
    rows, law, target0, target1 = [], [], 0.0, 0.0
    for x, b, a in itertools.product(range(2), range(3), range(2)):
        p = [0.25, 0.75][x]
        t0, t1 = [0.25, 1.0, 4.0][b] + x * 0.2, [0.5, 2.0, 5.0][b] + x * 0.3
        t = t1 if a else t0
        rows.append([min(t, 3.0), int(t <= 3.0), a, p, 0.2 + x * 0.1, 0.4 + x * 0.2, 1])
        law.append(0.5 / 3 * (p if a else 1 - p))
        if a == 0:
            target0 += 0.5 / 3 * min(t0, 3.0)
            target1 += 0.5 / 3 * min(t1, 3.0)
    data = pd.DataFrame(rows, columns=["u", "event", "a", "p", "m0", "m1", "complete"])
    result = fit("aipw", data)
    np.testing.assert_allclose(
        np.asarray(law) @ result["scores"].to_numpy(),
        [target0, target1, target1 - target0],
        atol=1e-15,
    )


def test_aipw_zero_variance_is_explicit_and_complete_at_censored_horizon_is_admitted():
    data = sample().assign(u=3.0, event=0, m0=3.0, m1=3.0, complete=1)
    result = fit("aipw", data)
    assert result["effects"].degenerate_variance.tolist() == [True] * 3
    assert result["effects"].std_error.tolist() == [0.0] * 3
    assert result["effects"].p_value.isna().all()
    assert result["effects"].ci_low.tolist() == [3.0, 3.0, 0.0]
    assert result["effects"].ci_high.tolist() == [3.0, 3.0, 0.0]


@pytest.mark.parametrize("name", ["survival", "rmst", "aipw"])
def test_full_json_roundtrip_preserves_table_order_dtype_state_and_tamper_failure(name, tmp_path):
    result = fit(name)
    artifact = json.loads(json.dumps(causal_design_save(result), allow_nan=False))
    restored = causal_design_load(artifact)
    assert list(restored) == list(result)
    for key in result:
        pd.testing.assert_frame_equal(result[key], restored[key], check_exact=True)
    assert restored.attrs["state"] == result.attrs["state"]
    path = tmp_path / f"{name}.json"
    causal_design_save(result, path)
    assert causal_design_save(causal_design_load(path)) == artifact
    artifact["payload"]["attrs"]["state"]["subject_scores"][0][0] += 1
    with pytest.raises(AnalysisError, match="checksum"):
        causal_design_load(artifact)


@pytest.mark.parametrize("name", ["survival", "rmst", "aipw"])
def test_global_meta_default_device_does_not_change_cpu_artifact_or_rng(name):
    expected = causal_design_save(fit(name))
    rng = torch.random.get_rng_state().clone()
    with torch.device("meta"):
        actual = causal_design_save(fit(name))
        assert torch.ones(1).device.type == "meta"
    assert actual == expected
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("name", ["survival", "rmst", "aipw"])
@pytest.mark.parametrize(
    "options",
    [
        {"design": None},
        {"design": "randomized"},
        {"nuisance": "estimated"},
        {"missing": "drop"},
        {"weights": "w"},
        {"device": "cuda"},
        {"level": 0},
        {"level": 1},
        {"positivity": 0},
        {"positivity": 0.5},
        {"max_work": 1},
    ],
)
def test_unsupported_routes_declarations_and_budgets(name, options):
    with pytest.raises(AnalysisError):
        fit(name, **options)


@pytest.mark.parametrize("name", ["survival", "rmst", "aipw"])
@pytest.mark.parametrize(
    "column,value",
    [("u", -1.0), ("event", 0.5), ("a", 2.0), ("p", 0.0), ("p", 1.0), ("p", 1e-9), ("p", np.nan)],
)
def test_original_numeric_sample_domain_and_positivity(name, column, value):
    data = sample()
    if name == "aipw":
        data["complete"] = 1
        data["event"] = 1
    data[column] = data[column].astype(float)
    data.loc[data.index == "same", column] = value
    with pytest.raises(AnalysisError):
        fit(name, data)


@pytest.mark.parametrize("name", ["survival", "rmst", "aipw"])
def test_boolean_treatment_dataset_and_missing_values_are_refused(name):
    data = sample()
    with pytest.raises(AnalysisError):
        fit(name, data.assign(a=data.a.astype(bool)))
    with pytest.raises(AnalysisError):
        fit(name, data.assign(u=np.nan))
    with pytest.raises(AnalysisError) as error:
        fit(name, Dataset.from_frame(data))
    assert error.value.code == "unsupported_dataset"


@pytest.mark.parametrize(
    "options",
    [
        {"thresholds": []},
        {"thresholds": [0.0, 0.0, 1.0, 2.0]},
        {"thresholds": [0.0, -1.0, 1.0, 2.0]},
        {"thresholds": [0.0, True, 1.0, 2.0]},
        {"thresholds": [0.0, 1.0]},
        {"thresholds": list(range(33))},
    ],
)
def test_survival_grid_domain(options):
    with pytest.raises(AnalysisError):
        fit("survival", **options)


@pytest.mark.parametrize("changes", [{"g1": 1.1}, {"g3": 0.0}, {"g0": 0.5, "g1": 0.9}])
def test_survival_censor_probability_and_monotonicity_domain(changes):
    with pytest.raises(AnalysisError):
        fit("survival", sample().assign(**changes))


@pytest.mark.parametrize("changes", [{"rate": -1.0}, {"rate": 100.0}, {"rate": np.inf}])
def test_exponential_rate_domain_and_tau_censor_positivity(changes):
    with pytest.raises(AnalysisError):
        fit("rmst", sample().assign(**changes))


def test_zero_rate_cannot_hide_an_early_censor_and_aipw_cannot_use_early_censor():
    with pytest.raises(AnalysisError, match="zero censor"):
        fit("rmst", sample().assign(rate=0.0))
    with pytest.raises(AnalysisError) as error:
        fit("aipw", sample().assign(complete=1))
    assert error.value.code == "incomplete_horizon"
    with pytest.raises(AnalysisError, match="predictions"):
        fit("aipw", sample().assign(m1=3.1))


def test_numerical_underflow_and_unrepresentable_variance_are_refused():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    with pytest.raises(AnalysisError, match="underflow"):
        fit("rmst", sample().assign(u=1e-200, event=1, rate=tiny), tau=1.0)
    with pytest.raises(AnalysisError, match="covariance"):
        fit("rmst", sample().assign(u=1e-200, event=1, rate=0.0), tau=1.0)
    # Very large capped times and valid tiny p would produce infinite variance.
    with pytest.raises(AnalysisError):
        fit(
            "rmst",
            sample().assign(u=1e150, event=1, rate=0.0, p=1e-150),
            tau=1e150,
            positivity=1e-151,
        )


def test_workspace_and_work_budget_preflight_run_before_score_kernel(monkeypatch):
    import openecon.econometrics.causal_design.observational_survival as module

    def refused(*args, **kwargs):
        raise AssertionError("No full kernel after failed complete preflight")

    monkeypatch.setattr(module, "_infer", refused)
    with pytest.raises(AnalysisError) as error:
        fit("survival", max_work=100)
    assert error.value.code == "work_budget_exceeded"
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        fit("survival", pd.concat([sample()] * 30, ignore_index=True))
    assert error.value.code == "workspace_limit"


def test_complete_workspace_and_covariance_work_preflight_precedes_sample_copy(monkeypatch):
    import openecon.econometrics.causal_design.observational_survival as module

    def refused(*args, **kwargs):
        raise AssertionError("Selected subject data cannot allocate before complete preflight")

    monkeypatch.setattr(module.m, "sample", refused)
    # Initial input work would fit, but the full joint inference would not.
    with pytest.raises(AnalysisError) as error:
        fit("survival", max_work=8 * (64 + 16 * 64 * 12 + 8 * 12 * 12))
    assert error.value.code == "work_budget_exceeded"
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        fit("survival", pd.concat([sample()] * 30, ignore_index=True))
    assert error.value.code == "workspace_limit"


def test_roles_are_distinct_and_known_probabilities_are_not_arbitrary_weights():
    with pytest.raises(AnalysisError):
        treatment_survival_ipcw(
            sample(),
            "u",
            "event",
            "a",
            "p",
            ["p"] * 4,
            design="unconfounded",
            nuisance="known",
            thresholds=GRID,
        )
    with pytest.raises(AnalysisError):
        treatment_rmst_aipw(
            sample(),
            "u",
            "event",
            "a",
            "p",
            "m0",
            "m0",
            design="unconfounded",
            nuisance="known",
            tau=3.0,
        )
