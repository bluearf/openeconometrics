"""Independent PMF, posterior, predictive and saved-state discrete MI oracles."""
from __future__ import annotations

import ast
import copy
from decimal import Decimal, localcontext
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad, simpson
from scipy.special import expit, logsumexp
from scipy.stats import poisson
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi import discrete
from openecon.econometrics.mi.common import _digest
from openecon.resources import use_workspace_budget


def _tensor(values):
    return torch.tensor(values, dtype=torch.float64, device="cpu")


def _frame():
    return pd.DataFrame({
        "count": [0., 1., 2., None, 1., None, 3., 0., 1., 2., None, 2.],
        "ordered": [10., 30., 20., 30., None, 10., 20., 30., 10., None, 30., 20.],
        "nominal": [7., 8., 9., None, 8., 7., 9., 7., None, 8., 7., 9.],
        "x": np.linspace(-1, 1, 12),
    }, index=pd.Index([i // 2 for i in range(12)], name="row"))


def _run(frame=None, **kwargs):
    return discrete.mi_discrete(
        _frame() if frame is None else frame, ["count", "ordered", "nominal", "x"],
        methods={"count": "poisson", "ordered": "ordinal", "nominal": "multinomial"},
        categories={"ordered": [10, 30, 20], "nominal": [7, 8, 9]},
        m=kwargs.pop("m", 3), burn=kwargs.pop("burn", 1), iterations=kwargs.pop("iterations", 2),
        mh_burn=kwargs.pop("mh_burn", 15), mh_steps=kwargs.pop("mh_steps", 20), **kwargs,
    )


def test_poisson_logposterior_and_gradient_match_original_scipy_pmf():
    x = np.array([[1, -1.2], [1, .5], [1, 2.2], [1, .1]])
    y = np.array([0., 1., 4., 2.])
    beta = np.array([.2, .3])
    scale = 1.7
    expected = poisson.logpmf(y, np.exp(x @ beta)).sum() - np.square(beta).sum() / (2 * scale ** 2)
    expected += sum(math.lgamma(value + 1) for value in y)
    parameter = _tensor(beta).requires_grad_()
    actual = discrete._poisson_log_posterior(_tensor(x), _tensor(y), parameter, scale)
    assert float(actual.detach()) == pytest.approx(expected, abs=1e-13)
    actual.backward()
    np.testing.assert_allclose(parameter.grad, x.T @ (y - np.exp(x @ beta)) - beta / scale ** 2, atol=1e-13)


def test_ordinal_pmf_and_prior_use_declared_coordinates_without_jacobian():
    x = np.array([[-1.], [.2], [1.8], [3.]])
    theta = np.array([.4, -.7, np.log(.9), np.log(1.4)])
    cuts = np.array([-.7, .2, 1.6])
    cdf = expit(cuts[None, :] - x @ theta[:1, None])
    probabilities = np.diff(np.column_stack([np.zeros(4), cdf, np.ones(4)]), axis=1)
    actual = discrete._ordinal_log_probabilities(_tensor(x), _tensor(theta), 4)
    np.testing.assert_allclose(actual.exp(), probabilities, atol=2e-16)
    y = np.array([0, 1, 2, 3])
    scale = 1.4
    expected = np.log(probabilities[np.arange(4), y]).sum() - np.square(theta).sum() / (2 * scale ** 2)
    actual_target = discrete._ordinal_log_posterior(_tensor(x), torch.tensor(y), _tensor(theta), scale, 4)
    assert float(actual_target) == pytest.approx(expected, abs=1e-13)
    # A change-of-variables Jacobian would add log(.9)+log(1.4), which is absent.
    assert abs(float(actual_target) - (expected + theta[2:].sum())) > .1


def test_ordinal_small_positive_gap_remains_finite_when_cdf_subtraction_cancels():
    # Direct sigmoid subtraction gives zero at these logits. Decimal computes
    # the original PMF independently at 90 decimal digits.
    with localcontext() as context:
        context.prec = 90
        a, gap = Decimal("40"), Decimal("0.00000001")
        def sigmoid(value):
            return Decimal(1) / (Decimal(1) + (-value).exp())
        expected = float((sigmoid(a + gap) - sigmoid(a)).ln())
    actual = discrete._ordinal_log_probabilities(_tensor([[]]), _tensor([40., np.log(1e-8)]), 3)
    assert np.isfinite(actual.numpy()).all()
    assert float(actual[0, 1]) == pytest.approx(expected, abs=1e-12)
    assert float(actual.exp().sum()) == pytest.approx(1, abs=1e-15)


def test_multinomial_pmf_and_gradient_match_fixed_baseline_softmax():
    x = np.array([[1., -1.2], [1., .4], [1., 1.8]])
    theta = np.array([.2, -.3, .6, .4])
    beta = theta.reshape(2, 2)
    logits = np.column_stack([np.zeros(3), x @ beta])
    logs = logits - logsumexp(logits, axis=1, keepdims=True)
    y = np.array([0, 2, 1])
    scale = 2.1
    parameter = _tensor(theta).requires_grad_()
    actual = discrete._multinomial_log_posterior(_tensor(x), torch.tensor(y), parameter, scale, 3)
    expected = logs[np.arange(3), y].sum() - np.square(theta).sum() / (2 * scale ** 2)
    assert float(actual.detach()) == pytest.approx(expected, abs=1e-13)
    actual.backward()
    indicators = np.eye(3)[y]
    gradient = (x.T @ (indicators[:, 1:] - np.exp(logs[:, 1:])) - beta / scale ** 2).ravel()
    np.testing.assert_allclose(parameter.grad, gradient, atol=1e-13)
    np.testing.assert_allclose(discrete._multinomial_log_probabilities(_tensor(x), _tensor(theta), 3), logs, atol=1e-13)


def test_poisson_posterior_and_predictive_moments_match_scalar_quadrature():
    x, y, scale = _tensor([[1.], [1.], [1.]]), _tensor([0., 1., 2.]), 1.3
    def density(beta):
        return np.exp(3 * beta - 3 * np.exp(beta) - beta ** 2 / (2 * scale ** 2))
    normalizer = quad(density, -12, 12, epsabs=1e-12)[0]
    expected_beta = quad(lambda beta: beta * density(beta), -12, 12, epsabs=1e-12)[0] / normalizer
    expected_rate = quad(lambda beta: np.exp(beta) * density(beta), -12, 12, epsabs=1e-12)[0] / normalizer
    expected_rate2 = quad(lambda beta: np.exp(2 * beta) * density(beta), -12, 12, epsabs=1e-12)[0] / normalizer
    generator = torch.Generator(device="cpu").manual_seed(89111)
    beta, _ = discrete._poisson_draw(x, y, generator, prior_scale=scale, proposal_scale=.7, burn=2000, steps=1)
    coefficients, rates, predictions = [], [], []
    for _ in range(5000):
        beta, diagnostic = discrete._poisson_draw(x, y, generator, prior_scale=scale,
                                                 proposal_scale=.7, burn=0, steps=4, initial=beta)
        coefficients.append(float(beta[0]))
        rate = beta.exp()
        rates.append(float(rate[0]))
        predictions.append(float(torch.poisson(rate, generator=generator)[0]))
        assert diagnostic["sampling_proposals"] == 4
    assert np.mean(coefficients) == pytest.approx(expected_beta, abs=.04)
    assert np.mean(rates) == pytest.approx(expected_rate, abs=.04)
    assert np.mean(predictions) == pytest.approx(expected_rate, abs=.06)
    expected_variance = expected_rate + expected_rate2 - expected_rate ** 2
    assert np.var(predictions) == pytest.approx(expected_variance, rel=.09)
    assert np.var(rates) > .15  # Posterior parameter uncertainty is present.


@pytest.mark.parametrize("method", ["ordinal", "multinomial"])
def test_categorical_posterior_moments_match_independent_two_dimensional_quadrature(method):
    scale = 1.2
    axis = np.linspace(-7, 7, 181)
    a, b = np.meshgrid(axis, axis, indexing="ij")
    if method == "ordinal":
        probabilities = np.stack([expit(a), expit(a + np.exp(b)) - expit(a), expit(-a - np.exp(b))], axis=-1)
        counts = np.array([2, 2, 2])
        x = torch.empty((6, 0), dtype=torch.float64, device="cpu")
        y = torch.tensor([0, 0, 1, 1, 2, 2], device="cpu")
        def log_target(theta):
            return discrete._ordinal_log_posterior(x, y, theta, scale, 3)
    else:
        logits = np.stack([np.zeros_like(a), a, b], axis=-1)
        probabilities = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
        counts = np.array([2, 3, 1])
        x = torch.ones((6, 1), dtype=torch.float64, device="cpu")
        y = torch.tensor([0, 0, 1, 1, 1, 2], device="cpu")
        def log_target(theta):
            return discrete._multinomial_log_posterior(x, y, theta, scale, 3)
    with np.errstate(divide="ignore"):
        log_density = (np.log(probabilities) * counts).sum(-1) - (a ** 2 + b ** 2) / (2 * scale ** 2)
    density = np.exp(log_density - np.nanmax(log_density))
    def integrate(values):
        return simpson(simpson(values, x=axis, axis=1), x=axis, axis=0)
    normalizer = integrate(density)
    expected = np.array([integrate(a * density), integrate(b * density)]) / normalizer
    generator = torch.Generator(device="cpu").manual_seed(12904)
    state, _ = discrete._mh_draw(log_target, 2, generator, prior_scale=scale, proposal_scale=.7, burn=2500, steps=1)
    draws = []
    for _ in range(5000):
        state, _ = discrete._mh_draw(log_target, 2, generator, prior_scale=scale,
                                     proposal_scale=.7, burn=0, steps=4, initial=state)
        draws.append(state.numpy().copy())
    draws = np.array(draws)
    np.testing.assert_allclose(draws.mean(0), expected, atol=.055)
    assert np.linalg.eigvalsh(np.cov(draws.T)).min() > .1


def test_discrete_fcs_preserves_observed_cells_duplicate_index_support_and_rng():
    frame = _frame()
    before = frame.copy(deep=True)
    rng = torch.random.get_rng_state().clone()
    result = _run(frame, seed=21)
    assert isinstance(result, discrete.MIDiscreteResult)
    assert torch.equal(rng, torch.random.get_rng_state())
    pd.testing.assert_frame_equal(frame, before)
    for number in range(1, 4):
        completed = result.dataset(imputation=number)
        assert completed.index.equals(frame.index)
        np.testing.assert_array_equal(completed.to_numpy()[frame.notna().to_numpy()], frame.to_numpy()[frame.notna().to_numpy()])
        assert np.all(completed["count"] >= 0)
        assert np.all(completed["count"] == np.floor(completed["count"]))
        assert set(completed["ordered"]).issubset({10., 30., 20.})
        assert set(completed["nominal"]).issubset({7., 8., 9.})
    assert result.model_dump_json() == _run(frame, seed=21).model_dump_json()
    assert result.model_dump_json() != _run(frame, seed=22).model_dump_json()
    assert result.metadata["visit_order"] == ("count", "ordered", "nominal")
    assert result.metadata["convergence"] == {"assessed": False, "converged": False}
    restored = discrete.MIDiscreteResult.model_validate_json(result.model_dump_json())
    assert restored == result
    pd.testing.assert_frame_equal(restored.dataset(imputation=2), result.dataset(imputation=2))
    assert result.model_copy() == result


def test_last_conditional_predictions_match_independent_original_pmfs():
    result = _run(seed=99)
    for chain in result.metadata["chain_diagnostics"]:
        for target, model in chain["last_models"].items():
            x = np.array(model["missing_design"])
            if target == "count":
                np.testing.assert_allclose(model["missing_means"], np.exp(x @ np.array(model["coefficients"])), atol=1e-12)
            elif target == "ordered":
                cdf = expit(np.array(model["cutpoints"])[None, :] - x @ np.array(model["coefficients"])[:, None])
                reference = np.diff(np.column_stack([np.zeros(len(x)), cdf, np.ones(len(x))]), axis=1)
                np.testing.assert_allclose(model["missing_probabilities"], reference, atol=2e-15)
            else:
                logits = np.column_stack([np.zeros(len(x)), x @ np.array(model["coefficients"])])
                np.testing.assert_allclose(model["missing_probabilities"], np.exp(logits - logsumexp(logits, axis=1, keepdims=True)), atol=2e-15)


def test_poisson_overflow_proposals_are_rejected_without_clipping_or_rng_changes():
    x, y = _tensor([[1.], [1.]]), _tensor([0., 1.])
    assert float(discrete._poisson_log_posterior(x, y, _tensor([1000.]), 2.5)) == -np.inf
    beta, diagnostic = discrete._poisson_draw(x, y, torch.Generator().manual_seed(33), prior_scale=2.5,
                                             proposal_scale=1e4, burn=0, steps=25)
    assert diagnostic["rejected_nonfinite"] > 0
    assert np.isfinite(beta.numpy()).all()
    with pytest.raises(AnalysisError, match="rates"):
        discrete._rates(_tensor([np.log(1e6) + .01]), "count")


@pytest.mark.parametrize("name,value", [("m", True), ("seed", True), ("burn", True),
    ("iterations", 0), ("mh_burn", -1), ("mh_steps", 0), ("prior_scale", False),
    ("prior_scale", 0), ("proposal_scale", np.inf), ("proposal_scale", 1e-200), ("max_work", True)])
def test_invalid_scalar_options_fail_closed(name, value):
    with pytest.raises(AnalysisError):
        _run(**{name: value})


@pytest.mark.parametrize("observed", [[0., -.1, 2.], [0., 1.5, 2.], [0., 1., 1_000_001.]])
def test_invalid_observed_poisson_counts_are_refused(observed):
    frame = pd.DataFrame({"y": observed + [None]})
    with pytest.raises(AnalysisError, match="counts"):
        discrete.mi_discrete(frame, ["y"], methods={"y": "poisson"}, m=1, burn=0, iterations=1)


@pytest.mark.parametrize("categories", [None, {"y": [1, 2]}, {"y": [1, 2, 2]},
    {"y": [1, 2, 4]}, {"y": [True, 2, 3]}, {"y": [1, 2, np.inf]}, {"y": [1, 2, 3], "x": [1, 2, 3]}])
def test_undeclared_absent_duplicate_or_invalid_categories_are_refused(categories):
    frame = pd.DataFrame({"y": [1., 2., 3., None]})
    with pytest.raises(AnalysisError):
        discrete.mi_discrete(frame, ["y"], methods={"y": "ordinal"}, categories=categories, m=1, burn=0, iterations=1)


def test_intercept_only_ordinal_and_multinomial_work_with_explicit_categories():
    for method in ("ordinal", "multinomial"):
        result = discrete.mi_discrete(pd.DataFrame({"y": [4., 9., 7., None]}), ["y"], methods={"y": method},
                                      categories={"y": [4, 9, 7]}, m=1, burn=0, iterations=1, mh_burn=5, mh_steps=10)
        model = result.metadata["chain_diagnostics"][0]["last_models"]["y"]
        assert model["intercept"] is (method != "ordinal")
        assert model["posterior_dimension"] == 2


def test_work_and_workspace_guards_precede_sampler_and_result_buffers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("The sampler must not be reached after a resource refusal.")
    monkeypatch.setattr(discrete, "_mh_draw", forbidden)
    with pytest.raises(AnalysisError):
        _run(max_work=5000, mh_steps=10000)
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError):
            _run(m=30)


@pytest.mark.parametrize("change", ["support", "categories", "mean", "probability", "baseline", "cutpoint", "trace", "work", "prior", "convergence",
                                   "prior_structure", "chain_structure", "trace_structure", "model_structure", "boolean_cycle"])
def test_restored_copy_revalidates_full_operation_state_even_with_fresh_checksum(change):
    result = _run(m=1, seed=17)
    state = copy.deepcopy(result.model_dump(mode="json"))
    meta = state["metadata"]
    models = meta["chain_diagnostics"][0]["last_models"]
    if change == "support":
        state["completed_matrices"][0][3][2] = 99.
    elif change == "categories":
        meta["categories"]["ordered"] = [10, 20, 40]
    elif change == "mean":
        models["count"]["missing_means"][0] += .5
    elif change == "probability":
        models["nominal"]["missing_probabilities"][0] = [.2, .2, .6]
    elif change == "baseline":
        models["nominal"]["baseline_category"] = 8.
    elif change == "cutpoint":
        models["ordered"]["cutpoints"][0] += .1
    elif change == "trace":
        meta["chain_diagnostics"][0]["trace"][-1]["variables"]["count"]["accepted_sampling"] = 1000
    elif change == "work":
        meta["projected_work"] = 1
    elif change == "prior":
        meta["prior"]["ordinal"] = "normal prior on ordered cutpoints with Jacobian"
    elif change == "convergence":
        meta["sampler"]["stationarity_claim"] = True
    elif change == "prior_structure":
        meta["prior"] = False
    elif change == "chain_structure":
        meta["chain_diagnostics"][0] = None
    elif change == "trace_structure":
        meta["chain_diagnostics"][0]["trace"][-1] = False
    elif change == "model_structure":
        models["count"] = False
    else:
        meta["chain_diagnostics"][0]["trace"][0]["cycle"] = True
    state["integrity_sha256"] = _digest({key: value for key, value in state.items() if key != "integrity_sha256"})
    with pytest.raises(ValueError):
        result.model_copy(update={"completed_matrices": state["completed_matrices"], "metadata": meta,
                                  "integrity_sha256": state["integrity_sha256"]})


def test_native_module_has_no_numpy_scipy_or_statsmodels_runtime_imports():
    tree = ast.parse(Path(discrete.__file__).read_text())
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert all(not module.startswith(("numpy", "scipy", "statsmodels")) for module in modules)


def test_ambient_float32_and_meta_defaults_do_not_change_cpu_float64_kernel():
    reference = _run(m=1, seed=54)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            actual = _run(m=1, seed=54)
    finally:
        torch.set_default_dtype(old_dtype)
    assert actual.model_dump_json() == reference.model_dump_json()
