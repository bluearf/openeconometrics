"""Exhaustive NumPy response-pattern oracles for finite IRT prediction.

The oracle enumerates joint category outcomes instead of reproducing the
production moment identities or conditional convolution recurrence.
"""

import copy
from itertools import product

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.irt import calibrated, calibrated_predictive
from openecon.resources import use_workspace_budget


def _sigmoid(value):
    value = np.asarray(value, dtype=float)
    return np.exp(-np.logaddexp(0.0, -value))


def _probabilities(specifications, theta):
    """Separate fixed-bank probability formulas, using no production helpers."""
    theta = np.asarray(theta, dtype=float)
    curves = []
    for item in specifications:
        family = item["family"]
        if family == "binary":
            success = item["c"] + (item["d"] - item["c"]) * _sigmoid(
                item["a"] * (theta - item["b"])
            )
            probability = np.column_stack((1.0 - success, success))
        elif family == "grm":
            boundaries = _sigmoid(
                item["a"] * (theta[:, None] - np.asarray(item["thresholds"]))
            )
            cumulative = np.column_stack((np.ones(len(theta)), boundaries, np.zeros(len(theta))))
            probability = -np.diff(cumulative, axis=1)
        else:
            if family == "gpcm":
                steps = item["a"] * (theta[:, None] - np.asarray(item["thresholds"]))
                logits = np.column_stack((np.zeros(len(theta)), np.cumsum(steps, axis=1)))
            else:
                logits = theta[:, None] * np.asarray(item["slopes"]) + np.asarray(item["intercepts"])
            logits -= np.max(logits, axis=1, keepdims=True)
            probability = np.exp(logits)
            probability /= probability.sum(axis=1, keepdims=True)
        curves.append(probability)
    return curves


def _bank(specifications):
    names = [item["name"] for item in specifications]
    if all(item["family"] == "binary" for item in specifications):
        return calibrated.irt_bank_binary(
            names,
            [item["a"] for item in specifications],
            [item["b"] for item in specifications],
            guessing=[item["c"] for item in specifications],
            upper=[item["d"] for item in specifications],
        )
    return calibrated.irt_bank_polytomous(
        names,
        family=[item["family"] for item in specifications],
        thresholds=[item.get("thresholds") for item in specifications],
        discrimination=[item.get("a") for item in specifications],
        slopes=[item.get("slopes") for item in specifications],
        intercepts=[item.get("intercepts") for item in specifications],
        scores=[item["scores"] for item in specifications],
    )


def _posterior_masses(specifications, support, masses, responses):
    curves = _probabilities(specifications, support)
    prior = np.asarray(masses, dtype=float)
    prior /= prior.sum()
    with np.errstate(divide="ignore"):
        log_joint = np.tile(np.log(prior), (len(responses), 1))
    for position, row in enumerate(responses.to_numpy()):
        for item, category in enumerate(row):
            if pd.notna(category):
                log_joint[position] += np.log(curves[item][:, int(category)])
    log_joint -= np.max(log_joint, axis=1, keepdims=True)
    posterior = np.exp(log_joint)
    return posterior / posterior.sum(axis=1, keepdims=True)


def _enumerate(specifications, support, posterior, selected):
    """Enumerate every replicate response and then average posterior masses."""
    names = [item["name"] for item in specifications]
    indices = [names.index(name) for name in selected]
    chosen = [specifications[index] for index in indices]
    curves = [_probabilities(specifications, support)[index] for index in indices]
    patterns = np.asarray(list(product(*(range(curve.shape[1]) for curve in curves))), dtype=int)
    conditional = np.ones((len(support), len(patterns)))
    for item, curve in enumerate(curves):
        conditional *= curve[:, patterns[:, item]]
    joint = posterior @ conditional
    score_vectors = np.column_stack(
        [np.asarray(item["scores"])[patterns[:, column]] for column, item in enumerate(chosen)]
    )
    means = joint @ score_vectors
    covariance = np.empty((len(posterior), len(selected), len(selected)))
    for position, weights in enumerate(joint):
        centered = score_vectors - means[position]
        covariance[position] = centered.T @ (weights[:, None] * centered)
    totals = score_vectors.sum(axis=1)
    width = 1 + sum(max(item["scores"]) for item in chosen)
    pmf = np.zeros((len(posterior), width))
    for pattern, score in enumerate(totals):
        pmf[:, score] += joint[:, pattern]
    marginals = []
    for item, curve in enumerate(curves):
        marginals.append(
            np.column_stack([joint[:, patterns[:, item] == category].sum(axis=1)
                             for category in range(curve.shape[1])])
        )
    return dict(marginals=marginals, means=means, covariance=covariance, pmf=pmf)


@pytest.fixture(scope="module", params=["binary", "mixed_polytomous"])
def example(request):
    if request.param == "binary":
        specifications = [
            dict(name="two", family="binary", a=1.3, b=-.4, c=0., d=1., scores=[0, 1]),
            dict(name="three", family="binary", a=.7, b=.2, c=.15, d=1., scores=[0, 1]),
            dict(name="four", family="binary", a=1.1, b=-.1, c=.05, d=.91, scores=[0, 1]),
        ]
        values = [[1, 0, 1], [0, np.nan, 1], [np.nan] * 3]
    else:
        specifications = [
            dict(name="graded", family="grm", a=.9, thresholds=[-.8, .7], scores=[0, 2, 4]),
            dict(name="credit", family="gpcm", a=1.2, thresholds=[.6, -.5], scores=[0, 1, 3]),
            dict(name="nominal", family="nrm", slopes=[0., -.4, .8],
                 intercepts=[0., .25, -.5], scores=[1, 1, 5]),
        ]
        values = [[2, 0, 1], [0, np.nan, 2], [np.nan] * 3]
    names = [item["name"] for item in specifications]
    data = pd.DataFrame(values, columns=names, index=["same", "same", "empty"])
    support, masses = [-2.25, -.5, .6, 2.4], [.1, .2, .4, .3]
    bank = _bank(specifications)
    posterior = calibrated.irt_posterior(bank, data=data, support=support, masses=masses)
    weights = _posterior_masses(specifications, support, masses, data)
    return specifications, support, data, posterior, weights


@pytest.fixture(scope="module")
def large_posterior():
    """An admitted finite model whose declared buffers exceed one MiB."""
    names = ["a", "b", "c"]
    bank = calibrated.irt_bank_binary(names, [1., 1., 1.], [0., 0., 0.])
    data = pd.DataFrame(np.full((128, 3), np.nan), columns=names)
    return calibrated.irt_posterior(bank, data=data, support=np.linspace(-8., 8., 101).tolist(),
                                    masses=[1. / 101] * 101)


def _assert_sample(result, data):
    sample = result["sample"]
    assert sample.position.tolist() == list(range(len(data)))
    assert sample.source_index.tolist() == list(data.index)
    assert sample.observed_items.tolist() == data.notna().sum(axis=1).tolist()
    assert sample.missing_items.tolist() == data.isna().sum(axis=1).tolist()


def _assert_prediction(result, oracle, specifications, data, selected):
    by_name = {item["name"]: item for item in specifications}
    _assert_sample(result, data)
    for position in range(len(data)):
        rows = result["means"].query("position == @position")
        assert rows.item.tolist() == selected
        assert rows.source_index.tolist() == [data.index[position]] * len(selected)
        np.testing.assert_allclose(rows.expected_score, oracle["means"][position], atol=3e-13)
        np.testing.assert_allclose(rows.variance, np.diag(oracle["covariance"][position]), atol=3e-12)
        covariance = result["covariance"].query("position == @position")
        assert list(zip(covariance.item1, covariance.item2)) == list(product(selected, repeat=2))
        actual = covariance.covariance.to_numpy().reshape(len(selected), len(selected))
        np.testing.assert_allclose(actual, oracle["covariance"][position], atol=3e-12)
        np.testing.assert_allclose(actual, actual.T, atol=1e-14)
        assert np.linalg.eigvalsh(actual).min() >= -1e-11
        for item, name in enumerate(selected):
            probabilities = result["probabilities"].query("position == @position and item == @name")
            assert probabilities.category.tolist() == list(range(len(by_name[name]["scores"])))
            assert probabilities.score.tolist() == by_name[name]["scores"]
            np.testing.assert_allclose(probabilities.probability, oracle["marginals"][item][position], atol=3e-13)
            assert probabilities.probability.sum() == pytest.approx(1., abs=5e-14)


def _assert_distribution(result, oracle, data, level):
    _assert_sample(result, data)
    width = oracle["pmf"].shape[1]
    scores = np.arange(width)
    for position in range(len(data)):
        rows = result["distribution"].query("position == @position")
        assert rows.score.tolist() == scores.tolist()
        assert rows.source_index.tolist() == [data.index[position]] * width
        expected = oracle["pmf"][position]
        np.testing.assert_allclose(rows.probability, expected, atol=3e-14)
        np.testing.assert_allclose(rows.cdf, expected.cumsum(), atol=5e-14)
        np.testing.assert_allclose(rows.tail_ge, expected[::-1].cumsum()[::-1], atol=5e-14)
        summary = result["summary"].iloc[position]
        mean = expected @ scores
        variance = expected @ ((scores - mean) ** 2)
        assert summary.expected_score == pytest.approx(mean, abs=3e-12)
        assert summary.variance == pytest.approx(variance, abs=3e-12)
        assert summary.sd == pytest.approx(np.sqrt(variance), abs=3e-12)
        assert summary.quantile_low == np.searchsorted(expected.cumsum(), (1 - level) / 2)
        assert summary.quantile_high == np.searchsorted(expected.cumsum(), (1 + level) / 2)
        assert summary.level == level
        assert variance == pytest.approx(oracle["covariance"][position].sum(), abs=3e-12)


def test_full_mixed_bank_prediction_matches_exhaustive_joint_response_oracle(example):
    specifications, support, data, posterior, weights = example
    selected = [item["name"] for item in specifications]
    result = calibrated_predictive.irt_predictive(posterior)
    oracle = _enumerate(specifications, support, weights, selected)
    _assert_prediction(result, oracle, specifications, data, selected)
    assert result.attrs["settings"]["items"] == selected
    assert result.attrs["device"] == "cpu" and result.attrs["dtype"] == "float64"
    assert result.attrs["stata_parity_validated"] is False


@pytest.mark.parametrize("level", [.5, .95, .99])
def test_full_score_pmf_cdf_tails_and_discrete_intervals_against_enumeration(example, level):
    specifications, support, data, posterior, weights = example
    selected = [item["name"] for item in specifications]
    result = calibrated_predictive.irt_test_score_distribution(posterior, level=level)
    oracle = _enumerate(specifications, support, weights, selected)
    _assert_distribution(result, oracle, data, level)


def test_item_subset_order_retains_posterior_conditioning_on_unselected_observed_items(example):
    specifications, support, data, posterior, weights = example
    selected = [specifications[2]["name"], specifications[0]["name"]]
    oracle = _enumerate(specifications, support, weights, selected)
    _assert_prediction(calibrated_predictive.irt_predictive(posterior, items=selected),
                       oracle, specifications, data, selected)
    _assert_distribution(calibrated_predictive.irt_test_score_distribution(posterior, items=selected),
                         oracle, data, .95)


def test_latent_dependence_counterexample_requires_conditional_convolution_before_mixing():
    bank = calibrated.irt_bank_binary(["a", "b", "unused"], [1., 1., 1.], [0., 0., 0.])
    data = pd.DataFrame([[np.nan] * 3], columns=["a", "b", "unused"], index=["empty"])
    posterior = calibrated.irt_posterior(bank, data=data, support=[-float(np.log(9)), float(np.log(9))], masses=[.5, .5])
    prediction = calibrated_predictive.irt_predictive(posterior, items=["a", "b"])
    covariance = prediction["covariance"].covariance.to_numpy().reshape(2, 2)
    np.testing.assert_allclose(covariance, [[.25, .16], [.16, .25]], atol=1e-14)
    distribution = calibrated_predictive.irt_test_score_distribution(posterior, items=["a", "b"])
    np.testing.assert_allclose(distribution["distribution"].probability, [.41, .18, .41], atol=1e-14)
    assert distribution["summary"].variance.iloc[0] == pytest.approx(.82, abs=1e-14)
    assert not np.allclose(distribution["distribution"].probability, [.25, .5, .25])


def test_one_point_prior_produces_conditional_independence_and_all_supported_draws(example):
    specifications, _, data, _, _ = example
    bank = _bank(specifications)
    posterior = calibrated.irt_posterior(bank, data=data, support=[.6], masses=[1.])
    selected = [item["name"] for item in specifications]
    oracle = _enumerate(specifications, [.6], np.ones((len(data), 1)), selected)
    prediction = calibrated_predictive.irt_predictive(posterior)
    _assert_prediction(prediction, oracle, specifications, data, selected)
    for covariance in oracle["covariance"]:
        np.testing.assert_allclose(covariance - np.diag(np.diag(covariance)), 0, atol=1e-13)
    _assert_distribution(calibrated_predictive.irt_test_score_distribution(posterior), oracle, data, .95)
    draws = calibrated_predictive.irt_plausible_values(posterior, draws=99, seed=9)
    assert draws["draws"].support_index.tolist() == [0] * (len(data) * 99)
    assert draws["draws"].theta.tolist() == [.6] * (len(data) * 99)


def test_nominal_duplicate_gap_and_constant_scores_preserve_zero_cells_and_singular_covariance():
    specifications = [
        dict(name="gaps", family="nrm", slopes=[0., .8, -.4], intercepts=[0., -.2, .3], scores=[1, 1, 5]),
        dict(name="duplicate", family="nrm", slopes=[0., -.6, .4], intercepts=[0., .1, -.3], scores=[0, 4, 4]),
        dict(name="constant", family="nrm", slopes=[0., .7, -.5], intercepts=[0., .2, -.1], scores=[7, 7, 7]),
    ]
    support, masses = [-1.5, .2, 1.8], [.2, .5, .3]
    data = pd.DataFrame([[np.nan] * 3], columns=[item["name"] for item in specifications])
    posterior = calibrated.irt_posterior(_bank(specifications), data=data, support=support, masses=masses)
    selected = [item["name"] for item in specifications]
    oracle = _enumerate(specifications, support, np.asarray([masses]) / sum(masses), selected)
    result = calibrated_predictive.irt_predictive(posterior)
    _assert_prediction(result, oracle, specifications, data, selected)
    covariance = result["covariance"].covariance.to_numpy().reshape(3, 3)
    np.testing.assert_allclose(covariance[-1], 0, atol=1e-13)
    single = calibrated_predictive.irt_test_score_distribution(posterior, items=["gaps"])
    assert single["distribution"].score.tolist() == list(range(6))
    assert single["distribution"].probability.iloc[[0, 2, 3, 4]].tolist() == [0.] * 4
    constant = calibrated_predictive.irt_test_score_distribution(posterior, items=["constant"])
    assert constant["distribution"].probability.tolist() == [0.] * 7 + [1.]
    assert constant["summary"].quantile_low.iloc[0] == constant["summary"].quantile_high.iloc[0] == 7
    assert constant["summary"].variance.iloc[0] == pytest.approx(0., abs=1e-13)


def test_private_seeded_draws_reconstruct_support_indices_and_preserve_global_rng(example):
    _, support, data, posterior, weights = example
    before = torch.random.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(83)
    uniforms = torch.rand((len(data), 19), generator=generator, dtype=torch.float64, device="cpu").numpy()
    expected = np.stack([np.searchsorted(cdf, u, side="right") for cdf, u in zip(weights.cumsum(1), uniforms)])
    first = calibrated_predictive.irt_plausible_values(posterior, draws=19, seed=83)
    second = calibrated_predictive.irt_plausible_values(posterior, draws=19, seed=83)
    assert torch.equal(before, torch.random.get_rng_state())
    pd.testing.assert_frame_equal(first["draws"], second["draws"], check_exact=True)
    rows = first["draws"]
    assert rows.position.tolist() == np.repeat(np.arange(len(data)), 19).tolist()
    assert rows.source_index.tolist() == np.repeat(data.index.to_numpy(), 19).tolist()
    assert rows.draw.tolist() == list(range(1, 20)) * len(data)
    np.testing.assert_array_equal(rows.support_index.to_numpy().reshape(len(data), 19), expected)
    np.testing.assert_array_equal(rows.theta.to_numpy().reshape(len(data), 19), np.asarray(support)[expected])
    _assert_sample(first, data)


def test_zero_uniform_cannot_draw_a_zero_mass_support_point(monkeypatch):
    bank = calibrated.irt_bank_binary(["a", "b", "c"], [5., 5., 5.], [8., 8., 8.])
    data = pd.DataFrame([[1, 1, 1]], columns=["a", "b", "c"])
    # Every declared prior mass is positive. The first posterior mass is
    # legitimately zero in float64 after the very small likelihood factor.
    posterior = calibrated.irt_posterior(bank, data=data, support=[-8., 0., 8.], masses=[1e-300, .5, .5])
    assert posterior.attrs["posterior_state"]["posterior"][0][0] == 0.

    def zeros(shape, **kwargs):
        return torch.zeros(shape, dtype=kwargs["dtype"], device=kwargs["device"])

    monkeypatch.setattr(calibrated_predictive.torch, "rand", zeros)
    result = calibrated_predictive.irt_plausible_values(posterior, draws=3)
    assert result["draws"].support_index.tolist() == [1, 1, 1]
    assert result["draws"].theta.tolist() == [0., 0., 0.]


@pytest.mark.parametrize("method", ["irt_plausible_values", "irt_predictive", "irt_test_score_distribution"])
def test_complete_summary_roundtrip_and_reusable_restored_posterior(example, method):
    _, _, _, posterior, _ = example
    restored_posterior = oe.restore_summary(oe.summary_state(posterior))
    function = getattr(calibrated_predictive, method)
    original = function(posterior)
    actual = function(restored_posterior)
    encoded = oe.summary_state(original)
    restored = oe.restore_summary(encoded)
    assert oe.summary_state(restored) == encoded
    assert oe.summary_state(actual) == encoded
    assert restored.attrs == original.attrs
    assert restored.title == original.title
    assert restored.to_latex() == original.to_latex()
    for name in original:
        pd.testing.assert_frame_equal(restored[name], original[name], check_exact=True)
        pd.testing.assert_frame_equal(actual[name], original[name], check_exact=True)
    assert "posterior_state" in original.attrs
    assert len(original.attrs["posterior_state"]["responses"]) == 3


@pytest.mark.parametrize("method", ["irt_plausible_values", "irt_predictive", "irt_test_score_distribution"])
def test_numeric_source_indices_have_complete_json_and_latex_roundtrip(method):
    bank = calibrated.irt_bank_binary(["a"], [1.2], [-.3])
    data = pd.DataFrame({"a": [0., 1., np.nan]}, index=[11, 12, 11])
    posterior = calibrated.irt_posterior(bank, data=data, support=[-1., 0., 2.], masses=[.2, .3, .5])
    result = getattr(calibrated_predictive, method)(posterior)
    encoded = oe.summary_state(result)
    restored = oe.restore_summary(encoded)
    assert oe.summary_state(restored) == encoded
    assert restored.attrs == result.attrs
    assert restored.to_latex() == result.to_latex()
    assert result.attrs["posterior_state"]["indices"] == [11, 12, 11]
    for name in result:
        pd.testing.assert_frame_equal(restored[name], result[name], check_exact=True)


def test_unicode_draw_output_limit_precedes_bayes_reconstruction(monkeypatch):
    bank = calibrated.irt_bank_binary(["a"], [1.], [0.])
    data = pd.DataFrame({"a": [None] * 160}, index=["\U0001f30d" * 256] * 160)
    posterior = calibrated.irt_posterior(bank, data=data, support=[0.], masses=[1.])
    # This is an actually admitted and reusable core posterior, with a full
    # escaped JSON below the core's separate eight-MiB input domain.
    encoded = oe.summary_state(posterior)
    assert len(encoded.encode()) < 8 * 1024**2
    calibrated.irt_posterior_restore(encoded)

    def forbidden(*args, **kwargs):
        raise AssertionError("Bayes reconstruction preceded full escaped draw-output admission")

    monkeypatch.setattr(calibrated, "_bayes", forbidden)
    with pytest.raises(AnalysisError, match="32 MiB"):
        calibrated_predictive.irt_plausible_values(posterior, draws=99)


def test_unicode_item_names_full_covariance_limit_precedes_bayes_reconstruction(monkeypatch):
    names = ["\U0001f30d" * 255 + chr(0x100 + number) for number in range(16)]
    bank = calibrated.irt_bank_binary(names, [1.] * 16, [0.] * 16)
    posterior = calibrated.irt_posterior(bank, data={name: [None] * 128 for name in names},
                                        support=[0.], masses=[1.])
    assert len(oe.summary_state(posterior).encode()) < 8 * 1024**2
    calibrated._posterior(posterior)

    def forbidden(*args, **kwargs):
        raise AssertionError("Bayes reconstruction preceded full escaped covariance-output admission")

    monkeypatch.setattr(calibrated, "_bayes", forbidden)
    with pytest.raises(AnalysisError, match="32 MiB"):
        calibrated_predictive.irt_predictive(posterior)


@pytest.mark.parametrize("method", ["irt_plausible_values", "irt_predictive", "irt_test_score_distribution"])
@pytest.mark.parametrize("option", [{"max_work": 1}, {"max_bytes": 1}, {"device": "mps"}, {"device": "cuda"}])
def test_work_bytes_and_explicit_device_rejections(example, method, option):
    with pytest.raises(AnalysisError):
        getattr(calibrated_predictive, method)(example[3], **option)


@pytest.mark.parametrize("method", ["irt_plausible_values", "irt_predictive", "irt_test_score_distribution"])
def test_workspace_budget_and_non_cpu_caller_default_are_respected(example, large_posterior, method):
    function = getattr(calibrated_predictive, method)
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError):
            function(large_posterior)
    with torch.device("meta"):
        actual = function(example[3])
    assert actual.attrs["device"] == "cpu"


@pytest.mark.parametrize("items", [[], "graded", ["absent"], ["graded", "graded"], [1], [True]])
@pytest.mark.parametrize("method", ["irt_predictive", "irt_test_score_distribution"])
def test_unknown_empty_duplicate_or_untyped_item_selection_is_rejected(example, method, items):
    with pytest.raises(AnalysisError):
        getattr(calibrated_predictive, method)(example[3], items=items)


@pytest.mark.parametrize("options", [{"draws": 0}, {"draws": 100}, {"draws": True}, {"draws": 2.5},
                                     {"seed": -1}, {"seed": 2**63}, {"seed": True}, {"seed": 1.5}])
def test_draw_controls_are_bounded_and_not_implicitly_coerced(example, options):
    with pytest.raises(AnalysisError):
        calibrated_predictive.irt_plausible_values(example[3], **options)


@pytest.mark.parametrize("options", [{"level": 0}, {"level": 1}, {"level": True}, {"level": float("nan")},
                                     {"level": 10**1000},
                                     {"level": 1e-300}, {"level": float(np.nextafter(1., 0.))},
                                     {"max_support": 0}, {"max_support": 257}, {"max_support": True},
                                     {"max_support": 3.5}])
def test_pmf_controls_are_bounded_and_not_implicitly_coerced(example, options):
    with pytest.raises(AnalysisError):
        calibrated_predictive.irt_test_score_distribution(example[3], **options)


def test_support_and_people_limits_reject_before_numerical_posterior_reconstruction(example, monkeypatch):
    _, _, _, posterior, _ = example

    def forbidden(*args, **kwargs):
        raise AssertionError("Numerical posterior reconstruction preceded structural admission")

    monkeypatch.setattr(calibrated, "_posterior", forbidden)
    with pytest.raises(AnalysisError):
        calibrated_predictive.irt_test_score_distribution(posterior, max_support=1)
    # Core validates this primitive input; changing only the verified primitive
    # decoder lets this test isolate the downstream 128-person guard.
    primitive = calibrated._posterior_input(posterior, max_bytes=128 * 1024**2)
    oversized = copy.deepcopy(primitive)
    oversized["responses"] = [primitive["responses"][0]] * 129
    oversized["indices"] = list(range(129))
    monkeypatch.setattr(calibrated, "_posterior_input", lambda *args, **kwargs: oversized)
    for function in (calibrated_predictive.irt_predictive, calibrated_predictive.irt_test_score_distribution):
        with pytest.raises(AnalysisError):
            function(posterior)


@pytest.mark.parametrize("method", ["irt_plausible_values", "irt_predictive", "irt_test_score_distribution"])
def test_downstream_work_and_buffers_are_combined_before_bayes_tensors(example, monkeypatch, method):
    posterior = example[3]
    core_work = posterior.attrs["declared_work"]
    core_bytes = posterior.attrs["resources"]["estimated_workspace_bytes"]
    # The unextended core model really fits each stated bound. Any subsequent
    # rejection therefore comes from the additional requested result, not an
    # arbitrary budget that already excludes posterior validation.
    calibrated._posterior(posterior, max_work=core_work, max_bytes=core_bytes)

    def forbidden(*args, **kwargs):
        raise AssertionError("Bayes tensor reconstruction preceded combined downstream admission")

    monkeypatch.setattr(calibrated, "_bayes", forbidden)
    function = getattr(calibrated_predictive, method)
    with pytest.raises(AnalysisError):
        function(posterior, max_work=core_work)
    with pytest.raises(AnalysisError):
        function(posterior, max_bytes=core_bytes)


@pytest.mark.parametrize("method", ["irt_plausible_values", "irt_predictive", "irt_test_score_distribution"])
def test_changed_posterior_state_is_refused_by_all_consumers(example, method):
    state = copy.deepcopy(example[3])
    state.attrs["posterior_state"]["posterior"][0][0] += .1
    with pytest.raises(AnalysisError):
        getattr(calibrated_predictive, method)(state)
