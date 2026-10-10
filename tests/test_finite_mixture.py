"""Independent Gaussian densities/derivatives, joint inference and saved semantics."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import logsumexp

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mixtures import gaussian as module


def data():
    i = np.arange(80)
    x = (i - 39.5) / 20
    y = np.where(i % 2 == 0, 2 + 0.6 * x, -2 - 0.3 * x) + 0.22 * np.sin(1.7 * i)
    return pd.DataFrame(
        {"y": y, "x": x}, index=pd.Index([f"case-{i}" for i in range(80)], name="case")
    )


START = {"coefficients": [[2.0, 0.6], [-2.0, -0.3]], "scales": [0.5, 0.5], "weights": [0.45, 0.55]}


@pytest.fixture(scope="module")
def fitted():
    return module.finite_mixture(data(), "y", ["x"], sigma_min=0.01, starts=[START], tolerance=1e-9)


def independent_ll(theta, x, y, k):
    d = x.shape[1]
    beta = theta[: k * d].reshape(k, d)
    sigma = np.exp(theta[k * d : k * d + k])
    logits = np.r_[theta[k * d + k :], 0.0]
    log_pi = logits - logsumexp(logits)
    log_joint = (
        log_pi
        - np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
        - 0.5 * ((y[:, None] - x @ beta.T) / sigma) ** 2
    )
    return logsumexp(log_joint, axis=1).sum()


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_full_louis_against_independent_numerical_and_autograd(k):
    rng = np.random.default_rng(2897 + k)
    x = np.c_[np.ones(17), rng.normal(size=(17, 2))]
    y = rng.normal(size=17)
    beta = rng.normal(size=(k, 3))
    sigma = np.exp(rng.normal(size=k) * 0.2)
    pi = rng.uniform(0.2, 1, size=k)
    pi /= pi.sum()
    model = module.GaussianMixture(3, k, 0.001)
    theta = model.pack(torch.tensor(beta), torch.tensor(sigma), torch.tensor(pi))
    xx, yy = torch.tensor(x), torch.tensor(y)
    native = model.evaluate(xx, yy, theta, derivatives=True)

    def objective(t):
        return model.evaluate(xx, yy, t)["log_likelihood"]

    z = theta.clone().requires_grad_()
    gradient = torch.autograd.grad(objective(z), z)[0]
    hessian = torch.autograd.functional.hessian(objective, z)
    torch.testing.assert_close(native["score"], gradient, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(native["information"], -hessian, atol=3e-13, rtol=3e-13)
    point = theta.numpy()
    step = 2e-4
    basis = np.eye(len(point)) * step
    score = np.array(
        [
            (independent_ll(point + v, x, y, k) - independent_ll(point - v, x, y, k)) / (2 * step)
            for v in basis
        ]
    )
    information = np.array(
        [
            [
                -(
                    independent_ll(point + a + b, x, y, k)
                    - independent_ll(point + a - b, x, y, k)
                    - independent_ll(point - a + b, x, y, k)
                    + independent_ll(point - a - b, x, y, k)
                )
                / (4 * step**2)
                for b in basis
            ]
            for a in basis
        ]
    )
    np.testing.assert_allclose(native["score"], score, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(native["information"], information, atol=7e-6, rtol=7e-6)


def test_em_weighted_regression_and_normalized_responsibilities():
    f = data().iloc[:24]
    x = torch.tensor(np.c_[np.ones(24), f.x])
    y = torch.tensor(f.y.to_numpy())
    model = module.GaussianMixture(2, 2, 0.01)
    theta = module._physical_start(model, START)
    next_theta, out = model.em_update(x, y, theta)
    b, s, pi = model.unpack(next_theta)
    density = np.exp(
        -0.5
        * (
            (y.numpy()[:, None] - x.numpy() @ np.array(START["coefficients"]).T)
            / np.array(START["scales"])
        )
        ** 2
    )
    density *= np.array(START["weights"]) / np.array(START["scales"])
    z = density / density.sum(1)[:, None]
    np.testing.assert_allclose(out["responsibility"], z, atol=1e-14)
    expected_beta = np.stack(
        [
            np.linalg.lstsq(
                x.numpy() * np.sqrt(z[:, j, None]), y.numpy() * np.sqrt(z[:, j]), rcond=None
            )[0]
            for j in range(2)
        ]
    )
    expected_scale = np.sqrt(
        (z * (y.numpy()[:, None] - x.numpy() @ expected_beta.T) ** 2).sum(0) / z.sum(0)
    )
    np.testing.assert_allclose(b, expected_beta, atol=3e-14)
    np.testing.assert_allclose(s, expected_scale, atol=3e-14)
    np.testing.assert_allclose(pi, z.mean(0), atol=3e-14)
    assert float(model.evaluate(x, y, next_theta)["log_likelihood"]) >= float(out["log_likelihood"])


def test_k1_gaussian_ml_full_covariance_and_df():
    f = data().iloc[:30]
    result = module.finite_mixture(f, "y", ["x"], components=1, sigma_min=0.01, restarts=1)
    state = result.attrs["state"]
    x = np.c_[np.ones(len(f)), f.x]
    y = f.y.to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    variance = np.mean((y - x @ beta) ** 2)
    covariance = np.zeros((4, 4))
    covariance[:2, :2] = variance * np.linalg.inv(x.T @ x)
    covariance[2, 2] = variance / (2 * len(f))
    np.testing.assert_allclose(state["parameters"], [*beta, np.sqrt(variance), 1.0], atol=2e-13)
    np.testing.assert_allclose(state["inference"]["covariance"], covariance, atol=2e-13)
    assert state["inference"]["df"] is None
    assert result["parameters"].loc["weight[1]", "pvalue"] is None or pd.isna(
        result["parameters"].loc["weight[1]", "pvalue"]
    )


def test_boundary_winner_retained_and_inference_unavailable():
    x = np.linspace(-1, 1, 30)
    y = 1 + 2 * x + 0.02 * np.sin(np.arange(30))
    result = module.finite_mixture(
        {"y": y, "x": x}, "y", ["x"], components=1, sigma_min=0.8, restarts=2
    )
    state = result.attrs["state"]
    assert state["parameters"][2] == pytest.approx(0.8, abs=1e-15)
    assert not state["inference"]["available"]
    assert "active_sigma_bound" in state["inference"]["reasons"]
    assert state["chosen_start"] == max(
        range(2), key=lambda i: state["starts"][i]["log_likelihood"]
    )
    assert module.finite_mixture_restore(result).attrs["digest"] == result.attrs["digest"]
    with pytest.raises(AnalysisError, match="regular joint parameter covariance"):
        module.finite_mixture_predict(result, {"x": [0.0]})
    assert module.finite_mixture_predict(result, {"x": [0.0]}, parameter_uncertainty=False)[
        "prediction"
    ].estimate.iloc[0] == pytest.approx(np.linalg.lstsq(np.c_[np.ones(30), x], y, rcond=None)[0][0])


def test_failed_start_denominator_and_explicit_input(fitted):
    bad = {**START, "weights": [0.8, 0.8]}
    result = module.finite_mixture(data(), "y", ["x"], sigma_min=0.01, starts=[bad, START])
    state = result.attrs["state"]
    assert (state["attempted_starts"], state["converged_starts"], state["failed_starts"]) == (
        2,
        1,
        1,
    )
    assert state["starts"][0]["failure"]["code"] == "invalid_start"
    assert state["chosen_start"] == 1
    assert module.finite_mixture_restore(result).attrs["digest"] == result.attrs["digest"]


def test_label_permutation_preserves_full_joint_law(fitted):
    swapped = {
        "coefficients": START["coefficients"][::-1],
        "scales": START["scales"][::-1],
        "weights": START["weights"][::-1],
    }
    other = module.finite_mixture(
        data(), "y", ["x"], sigma_min=0.01, starts=[swapped], tolerance=1e-9
    )
    np.testing.assert_allclose(
        other.attrs["state"]["parameters"], fitted.attrs["state"]["parameters"], atol=2e-12
    )
    np.testing.assert_allclose(
        other.attrs["state"]["inference"]["covariance"],
        fitted.attrs["state"]["inference"]["covariance"],
        atol=2e-12,
    )
    np.testing.assert_allclose(
        other.attrs["state"]["output"]["log_density"],
        fitted.attrs["state"]["output"]["log_density"],
        atol=2e-12,
    )


def test_restore_never_calls_optimizer_and_preserves_typed_index(fitted, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer invoked")

    for name in ("finite_mixture", "fit_gaussian_mixture", "_polish"):
        monkeypatch.setattr(module, name, forbidden)
    restored = module.finite_mixture_restore(fitted.to_json())
    assert restored.attrs["digest"] == fitted.attrs["digest"]
    pd.testing.assert_index_equal(restored["membership"].index, data().index)


@pytest.mark.parametrize(
    "target",
    [
        "mean",
        "variance",
        "component_mean",
        "prior",
        "posterior",
        "log_density",
        "density",
        "cdf",
        "quantile",
    ],
)
def test_saved_prediction_full_cross_covariance_and_independent_delta(fitted, target):
    query = data().iloc[[11, 2, 17]]
    kwargs = (
        {"y": "y"}
        if target in {"posterior", "log_density", "density", "cdf"}
        else {"probability": 0.73}
        if target == "quantile"
        else {}
    )
    result = module.finite_mixture_predict(fitted, query, target=target, **kwargs)
    state = fitted.attrs["state"]
    theta = np.array(state["theta"])
    k, d = 2, 2
    xx = np.c_[np.ones(3), query.x]
    obs = query.y.to_numpy()

    def independent(point):
        beta = point[: k * d].reshape(k, d)
        sigma = np.exp(point[k * d : k * d + k])
        logits = np.r_[point[k * d + k :], 0.0]
        pi = np.exp(logits - logsumexp(logits))
        mu = xx @ beta.T
        mean = mu @ pi
        if target == "mean":
            return mean
        if target == "variance":
            return (sigma**2 + (mu - mean[:, None]) ** 2) @ pi
        if target == "component_mean":
            return mu.ravel()
        if target == "prior":
            return np.tile(pi, 3)
        from scipy.special import ndtr
        from scipy.optimize import brentq

        if target == "quantile":
            return np.array(
                [brentq(lambda v: ndtr((v - mu[j]) / sigma) @ pi - 0.73, -20, 20) for j in range(3)]
            )
        if target == "cdf":
            return ndtr((obs[:, None] - mu) / sigma) @ pi
        joint = (
            np.log(pi)
            - np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * ((obs[:, None] - mu) / sigma) ** 2
        )
        logdensity = logsumexp(joint, axis=1)
        if target == "log_density":
            return logdensity
        if target == "density":
            return np.exp(logdensity)
        return np.exp(joint - logdensity[:, None]).ravel()

    jac = np.column_stack(
        [
            (independent(theta + v) - independent(theta - v)) / (2e-5)
            for v in np.eye(len(theta)) * 1e-5
        ]
    )
    covariance = jac @ np.array(state["inference"]["chart_covariance"]) @ jac.T
    np.testing.assert_allclose(result.attrs["values"], independent(theta), atol=2e-12)
    np.testing.assert_allclose(result.attrs["jacobian"], jac, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(result.attrs["covariance"], covariance, atol=2e-9, rtol=2e-8)
    assert result.attrs["source"]["positions"] == [0, 1, 2]
    pd.testing.assert_index_equal(
        module._restore_index(result.attrs["source"]["index"], 3), query.index
    )


def rehash(record):
    for name in ("source", "state"):
        record[name]["digest"] = module._digest(
            {k: v for k, v in record[name].items() if k != "digest"}
        )
    record["digest"] = module._digest({k: v for k, v in record.items() if k != "digest"})
    return record


@pytest.mark.parametrize(
    "kind",
    [
        "covariance",
        "score",
        "posterior",
        "parameter",
        "sample",
        "dtype",
        "em",
        "newton",
        "count",
        "index",
        "seed",
    ],
)
def test_rehashed_forgery_refused(fitted, kind):
    r = copy.deepcopy(fitted.attrs)
    s = r["state"]
    if kind == "covariance":
        s["inference"]["covariance"][0][1] += 0.01
    elif kind == "score":
        s["inference"]["score"][0] = 1e99
    elif kind == "posterior":
        s["output"]["responsibility"][0][0] = True
    elif kind == "parameter":
        s["parameters"][0] += 0.01
    elif kind == "sample":
        r["source"]["sample"][0] = 1
    elif kind == "dtype":
        r["source"]["dtypes"][0] = "string"
    elif kind == "em":
        s["starts"][0]["em_path"][1]["theta"][0] += 0.01
    elif kind == "newton":
        s["starts"][0]["polish"]["scaled_score"] = 1e99
    elif kind == "count":
        s["attempted_starts"] = True
    elif kind == "index":
        r["source"]["index"]["labels"][0] = "invented"
    else:
        s["starts"][0]["seed"] += 1
    with pytest.raises(AnalysisError):
        module.finite_mixture_restore(rehash(r))


def test_low_level_budget_refuses_before_any_numeric_scan(monkeypatch):
    x = torch.empty((20001, 2), dtype=torch.float64)
    y = torch.empty(20001, dtype=torch.float64)
    monkeypatch.setattr(
        torch,
        "isfinite",
        lambda *args, **kwargs: pytest.fail("numeric tensor scan preceded shape/work admission"),
    )
    with pytest.raises(AnalysisError, match="rows must be an integer"):
        module.fit_gaussian_mixture(x, y, sigma_min=0.1)


def test_public_work_budget_precedes_frame_copy_and_cell_conversion(monkeypatch):
    from openecon import analysis

    monkeypatch.setattr(
        analysis, "_coerce_frame", lambda *args: pytest.fail("copied before admission")
    )
    with pytest.raises(AnalysisError, match="exceeds max_work"):
        module.finite_mixture(
            {"x": np.zeros(100), "y": np.zeros(100)}, "y", ["x"], sigma_min=0.1, max_work=1
        )


@pytest.mark.parametrize(
    "case", ["generator", "missing", "huge_integer", "bool", "weight", "gpu", "rank"]
)
def test_unsupported_or_invalid_domain_refuses(case):
    f = data()
    kwargs = {}
    if case == "generator":
        f = {"x": iter(range(80)), "y": range(80)}
    elif case == "missing":
        f.loc[f.index[0], "x"] = np.nan
    elif case == "huge_integer":
        f = {"x": [2**54 + 1] * 80, "y": list(range(80))}
    elif case == "bool":
        f["x"] = True
    elif case == "weight":
        kwargs["weights"] = "x"
    elif case == "gpu":
        kwargs["device"] = "cuda"
    elif case == "rank":
        f["x"] = 1.0
    with pytest.raises(AnalysisError):
        module.finite_mixture(f, "y", ["x"], sigma_min=0.01, **kwargs)


def test_rng_default_dtype_and_device_preserved():
    before = torch.random.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    device = torch.get_default_device()
    result = module.finite_mixture(data(), "y", ["x"], sigma_min=0.01, restarts=2)
    assert result.attrs["state"]["attempted_starts"] == 2
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.get_default_dtype() == dtype and torch.get_default_device() == device


def test_all_starts_fail_full_diagnostics():
    with pytest.raises(AnalysisError) as failure:
        module.finite_mixture(
            data(),
            "y",
            ["x"],
            sigma_min=0.01,
            starts=[{**START, "weights": [0.9, 0.9]}, {**START, "scales": [-1.0, 1.0]}],
        )
    assert failure.value.code == "all_starts_failed"
    assert len(failure.value.start_diagnostics) == 2


def test_numeric_extremes_and_stable_variance():
    m = module.GaussianMixture(1, 1, 0.01)
    theta = m.pack(
        torch.tensor([[1e15]], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    )
    out = m.evaluate(
        torch.ones((3, 1), dtype=torch.float64), torch.full((3,), 1e15, dtype=torch.float64), theta
    )
    torch.testing.assert_close(out["variance"], torch.full((3,), 4.0, dtype=torch.float64))


def test_no_future_outcome_for_unconditional_target(fitted):
    with pytest.raises(AnalysisError, match="future outcome"):
        module.finite_mixture_predict(fitted, data().iloc[:2], y="y")
    with pytest.raises(AnalysisError, match="explicit distinct"):
        module.finite_mixture_predict(fitted, data().iloc[:2], target="posterior")


def test_prediction_budget_and_generator_before_restore(fitted, monkeypatch):
    monkeypatch.setattr(
        module,
        "finite_mixture_restore",
        lambda *args: pytest.fail("replayed before query admission"),
    )
    with pytest.raises(AnalysisError):
        module.finite_mixture_predict(fitted, {"x": np.zeros(513)})
    with pytest.raises(AnalysisError):
        module.finite_mixture_predict(fitted, {"x": iter(range(3))})
    with pytest.raises(AnalysisError):
        module.finite_mixture_predict(fitted, {"x": [0.0]}, max_work=1)


def test_json_bound_before_parser(monkeypatch):
    monkeypatch.setattr(json, "loads", lambda *args: pytest.fail("parsed oversized string"))
    with pytest.raises(AnalysisError):
        module.finite_mixture_restore(" " * (module.MAX_JSON + 1))


@pytest.mark.parametrize(
    "index",
    [
        pd.MultiIndex.from_arrays(
            [np.repeat(["a", "b"], 40), np.tile(np.arange(40), 2)], names=["group", "case"]
        ),
        pd.date_range("2020-01-01", periods=80, name="date"),
        pd.CategoricalIndex([f"row-{i}" for i in range(80)], ordered=True, name="ordered-case"),
    ],
)
def test_original_typed_index_preserved(index):
    frame = data()
    frame.index = index
    result = module.finite_mixture(frame, "y", ["x"], sigma_min=0.01, starts=[START])
    pd.testing.assert_index_equal(result["membership"].index, index)
    pd.testing.assert_index_equal(
        module.finite_mixture_restore(result.to_json())["membership"].index, index
    )


def test_information_coordinate_units_and_tiny_covariance_tamper():
    frame = data()
    frame["x"] *= 1e-80
    start = {**START, "coefficients": [[2.0, 0.6e80], [-2.0, -0.3e80]]}
    result = module.finite_mixture(frame, "y", ["x"], sigma_min=0.01, starts=[start])
    assert result.attrs["inference_available"]
    assert result.attrs["state"]["inference"]["covariance"][1][1] > 1e150
    forged = copy.deepcopy(result.attrs)
    forged["state"]["inference"]["covariance"][1][1] *= 1.01
    with pytest.raises(AnalysisError):
        module.finite_mixture_restore(rehash(forged))


@pytest.mark.parametrize(
    "option,value",
    [
        ("components", True),
        ("restarts", False),
        ("seed", True),
        ("max_em_iterations", True),
        ("max_polish_iterations", False),
        ("sigma_min", True),
        ("alpha", True),
        ("tolerance", float("nan")),
        ("sigma_min", 10**1000),
    ],
)
def test_typed_option_refusal(option, value):
    kwargs = {"sigma_min": 0.01, option: value}
    with pytest.raises(AnalysisError):
        module.finite_mixture(data(), "y", ["x"], **kwargs)


def test_frozen_original_author_and_independent_reference_packet(fitted):
    packet = json.loads(
        (Path(__file__).parent / "fixtures/finite-mixture-native-reference.json").read_text()
    )
    state = fitted.attrs["state"]
    assert packet["author_reference"]["package_version"] == "0.4.3"
    assert packet["author_reference"]["license"] == "GPL>=2 external developer oracle"
    for key, expected in packet["native_fixed"].items():
        np.testing.assert_allclose(state["output"][key], expected, atol=5e-12, rtol=5e-12)
    for key, reference in [
        ("information", "independent_information"),
        ("jacobian", "independent_jacobian"),
        ("chart_covariance", "independent_full_chart_covariance"),
        ("covariance", "independent_full_physical_covariance"),
    ]:
        np.testing.assert_allclose(
            state["inference"][key], packet[reference], atol=5e-12, rtol=5e-12
        )
    assert packet["original_paper_summary"]["n"] == 28
    assert packet["original_paper_summary"]["stale_last_E_max"] > 1e-5


def test_structural_zero_fixed_membership_covariance_rejects_subnormal_forgery():
    fitted = module.finite_mixture(data(), "y", ["x"], components=1, sigma_min=0.01, restarts=1)
    covariance = fitted.attrs["state"]["inference"]["covariance"]
    assert covariance[-1][-1] == 0.0
    forged = copy.deepcopy(fitted.attrs)
    forged["state"]["inference"]["covariance"][-1][-1] = 1e-320
    with pytest.raises(AnalysisError, match="optimizer-free numerical replay"):
        module.finite_mixture_restore(rehash(forged))
    assert (
        module.finite_mixture_restore(fitted).attrs["state"]["inference"]["covariance"][-1][-1]
        == 0.0
    )


def test_result_json_serializer_replays_rehashed_full_joint_covariance(fitted):
    forged = copy.deepcopy(fitted)
    forged.attrs["state"]["inference"]["covariance"][0][0] *= 4
    rehash(forged.attrs)
    with pytest.raises(AnalysisError, match="optimizer-free numerical replay"):
        forged.to_json()


def test_result_json_serializer_preserves_ambient_torch_state(fitted, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("serialization started a training optimizer")

    monkeypatch.setattr(module, "fit_gaussian_mixture", forbidden)
    monkeypatch.setattr(module, "_polish", forbidden)
    before = torch.random.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            serialized = fitted.to_json()
            restored = module.finite_mixture_restore(serialized)
    finally:
        torch.set_default_dtype(dtype)
    assert restored.attrs["digest"] == fitted.attrs["digest"]
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "change",
    [
        "physical_covariance",
        "chart_covariance",
        "information",
        "score",
        "jacobian",
        "output_prior",
        "output_log_likelihood",
        "output_mean",
        "theta",
        "parameters",
        "initial",
        "em_chart",
        "terminal",
        "newton_values",
        "active_indices",
    ],
)
def test_numerical_cache_shapes_and_types_admitted_before_likelihood(fitted, monkeypatch, change):
    forged = copy.deepcopy(fitted.attrs)
    state = forged["state"]
    if change == "physical_covariance":
        state["inference"]["covariance"] = []
    elif change == "chart_covariance":
        state["inference"]["chart_covariance"] = [[]]
    elif change == "information":
        state["inference"]["information"][0].pop()
    elif change == "score":
        state["inference"]["score"][0] = True
    elif change == "jacobian":
        state["inference"]["jacobian"][0][0] = "1.0"
    elif change == "output_prior":
        state["output"]["prior"].pop()
    elif change == "output_log_likelihood":
        state["output"]["log_likelihood"] = False
    elif change == "output_mean":
        state["output"]["mean"][0] = 1
    elif change in ("theta", "parameters"):
        state[change] = []
    elif change in ("initial", "terminal"):
        state["starts"][0][change] = []
    elif change == "em_chart":
        state["starts"][0]["em_path"][0]["theta"] = []
    elif change == "newton_values":
        state["starts"][0]["polish"]["values"][0] = True
    else:
        state["starts"][0]["polish"]["active_scale_indices"] = [True]
    rehash(forged)

    def forbidden(*args, **kwargs):
        pytest.fail("malformed numerical cache reached tensor allocation or likelihood replay")

    monkeypatch.setattr(module, "_numeric_tensor", forbidden)
    monkeypatch.setattr(module, "_qr", forbidden)
    monkeypatch.setattr(module.GaussianMixture, "evaluate", forbidden)
    with pytest.raises(AnalysisError):
        module.finite_mixture_restore(forged)
