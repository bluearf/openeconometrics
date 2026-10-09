"""Independent NumPy/SciPy MML oracle; source formulas and portable contracts."""

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, roots_hermitenorm
from scipy.stats import norm

import openecon as oe
from openecon.resources import use_workspace_budget


def geometry(raw, family, categories):
    """Independent slope/step mapping, matching declared identification only."""
    if family == "rsm":
        j = len(categories)
        steps = np.r_[raw[j:], -sum(raw[j:])]
        return np.ones(j), [raw[i] + steps for i in range(j)], np.r_[raw[:j], steps]
    slopes, steps, natural, at = [], [], [], 0
    for k in categories:
        a = np.exp(raw[at]) if family in ("2pl", "3pl", "grm") else 1.0
        if family in ("2pl", "3pl", "grm"):
            at += 1
            natural.append(a)
        b = np.array(raw[at : at + k - 1])
        if family == "grm":
            b[1:] = b[0] + np.cumsum(np.exp(b[1:]))
        at += k - 1
        slopes.append(a)
        steps.append(b)
        natural.extend(b)
    return np.array(slopes), steps, np.array(natural)


def probabilities(raw, family, categories, guessing, theta):
    slopes, steps, _ = geometry(raw, family, categories)
    result = []
    for a, b, k, c in zip(slopes, steps, categories, guessing):
        if family in ("rasch", "2pl", "3pl"):
            p = c + (1 - c) * expit(a * (theta - b[0]))
            result.append(np.column_stack((1 - p, p)))
        elif family == "grm":
            cumulative = np.column_stack(
                (np.ones(len(theta)), expit(a * (theta[:, None] - b)), np.zeros(len(theta)))
            )
            result.append(-np.diff(cumulative, axis=1))
        else:
            logits = np.column_stack((np.zeros(len(theta)), np.cumsum(theta[:, None] - b, axis=1)))
            result.append(np.exp(logits - logsumexp(logits, axis=1)[:, None]))
    return result


def loss(raw, family, categories, guessing, responses, q=83):
    theta, weights = roots_hermitenorm(q)
    weights /= np.sqrt(2 * np.pi)
    likelihood = np.zeros((len(responses), q))
    for j, p in enumerate(probabilities(raw, family, categories, guessing, theta)):
        likelihood += np.log(np.maximum(p[:, responses[:, j]].T, 1e-300))
    return -logsumexp(likelihood + np.log(weights), axis=1).sum()


def hessian(function, x, h=2e-4):
    answer = np.empty((len(x), len(x)))
    for i in range(len(x)):
        ei = np.eye(len(x))[i] * h
        answer[i, i] = (function(x + ei) - 2 * function(x) + function(x - ei)) / h**2
        for j in range(i):
            ej = np.eye(len(x))[j] * h
            answer[i, j] = answer[j, i] = (
                function(x + ei + ej)
                - function(x + ei - ej)
                - function(x - ei + ej)
                + function(x - ei - ej)
            ) / (4 * h * h)
    return answer


@pytest.fixture(scope="module")
def fitted():
    fits = {}
    for ix, family in enumerate(("rasch", "2pl", "3pl", "grm", "pcm", "rsm")):
        rng = np.random.default_rng(887 + ix)
        categories = [3, 3, 3, 3] if family in ("grm", "pcm", "rsm") else [2] * 4
        guessing = [0.12, 0.08, 0.1, 0.05] if family == "3pl" else [0.0] * 4
        if family == "rsm":
            raw = np.array([-0.8, -0.2, 0.3, 0.9, -0.5])
        elif family == "grm":
            raw = np.array(
                [
                    v
                    for b, a in zip([-0.8, -0.2, 0.3, 0.9], [1.0, 1.1, 0.9, 1.2])
                    for v in [np.log(a), b - 0.5, 0.0]
                ]
            )
        elif family == "pcm":
            raw = np.array([v for b in [-0.8, -0.2, 0.3, 0.9] for v in [b - 0.5, b + 0.5]])
        elif family == "rasch":
            raw = np.array([-0.8, -0.2, 0.3, 0.9])
        else:
            raw = np.array(
                [
                    v
                    for b, a in zip([-0.8, -0.2, 0.3, 0.9], [1.0, 1.1, 0.9, 1.2])
                    for v in [np.log(a), b]
                ]
            )
        n = 600
        theta = rng.normal(size=n)
        p = probabilities(raw, family, categories, guessing, theta)
        responses = np.column_stack(
            [(rng.random(n)[:, None] > np.cumsum(item, axis=1)).sum(axis=1) for item in p]
        )
        frame = pd.DataFrame(responses, columns=["a", "b", "c", "d"])
        kwargs = {"guessing": guessing} if family == "3pl" else {}
        result = getattr(oe, "irt_" + family)(data=frame, items=list(frame), points=61, **kwargs)
        fits[family] = (result, frame, raw, categories, guessing)
    return fits


@pytest.mark.parametrize("family", ["rasch", "2pl", "3pl", "grm", "pcm", "rsm"])
def test_all_six_mml_fits_against_independent_quadrature_optimization_and_full_hessian(
    fitted, family
):
    result, data, _, categories, guessing = fitted[family]
    state = result.attrs["state"]
    x = np.array(state["raw_parameters"])
    def function(v):
        return loss(v, family, categories, guessing, data.values)
    independent = minimize(
        function, np.zeros_like(x), method="BFGS", options={"gtol": 1e-6, "maxiter": 500}
    )
    # Finite-difference BFGS may report precision loss at tiny gradients; the
    # accepted objective and separately computed Hessian are what we compare.
    np.testing.assert_allclose(independent.x, x, atol=2e-5, rtol=2e-5)
    assert abs(function(x) + state["log_likelihood"]) < 1e-4
    info = hessian(function, x)
    np.testing.assert_allclose(info, state["observed_information"], atol=1e-4, rtol=1e-4)
    raw_cov = np.linalg.inv(info)
    jac = np.column_stack(
        [
            (
                geometry(x + np.eye(len(x))[i] * 1e-5, family, categories)[2]
                - geometry(x - np.eye(len(x))[i] * 1e-5, family, categories)[2]
            )
            / 2e-5
            for i in range(len(x))
        ]
    )
    covariance = jac @ raw_cov @ jac.T
    np.testing.assert_allclose(covariance, result["covariance"].values, atol=2e-6, rtol=2e-4)
    coef = result["parameters"]
    np.testing.assert_allclose(coef.std_error, np.sqrt(np.diag(covariance)), atol=2e-6, rtol=2e-4)
    np.testing.assert_allclose(coef.p_value, 2 * norm.sf(abs(coef.z)), atol=2e-12)
    np.testing.assert_allclose(coef.ci_low, coef.estimate - norm.ppf(0.975) * coef.std_error)
    assert result.attrs["stata_parity_validated"] is False


@pytest.mark.parametrize("family", ["rasch", "2pl", "3pl", "grm", "pcm", "rsm"])
def test_parameter_recovery_and_exact_complete_state_latex(fitted, family):
    result, _, generating, categories, _ = fitted[family]
    expected = geometry(generating, family, categories)[2]
    # Bounded seeded recovery, not universal recovery or a coverage claim.
    np.testing.assert_allclose(result["parameters"].estimate, expected, atol=0.65, rtol=0.3)
    restored = oe.irt_restore(result.to_json())
    assert restored.to_json() == result.to_json()
    for key in result:
        pd.testing.assert_frame_equal(result[key], restored[key], check_exact=True)
    assert result.to_latex() == restored.to_latex()
    assert "covariance" in result.to_latex() and "posterior" in result.to_latex()
    assert list(result["sample"].position) == list(range(600))
    assert (
        result.attrs["state"]["latent_mean"] == 0 and result.attrs["state"]["latent_variance"] == 1
    )


@pytest.mark.parametrize("family", ["rasch", "2pl", "3pl", "grm", "pcm", "rsm"])
def test_item_test_information_source_formula_and_independent_derivatives(fitted, family):
    result, _, _, categories, guessing = fitted[family]
    raw = np.array(result.attrs["state"]["raw_parameters"])
    grid = np.array([-3.0, 0.0, 2.0])
    actual = oe.irt_information(result, theta=grid.tolist())
    expected = probabilities(raw, family, categories, guessing, grid)
    derivatives = [
        (high - low) / 2e-5
        for high, low in zip(
            probabilities(raw, family, categories, guessing, grid + 1e-5),
            probabilities(raw, family, categories, guessing, grid - 1e-5),
        )
    ]
    total = np.zeros(len(grid))
    for i, name in enumerate(result.attrs["state"]["items"]):
        rows = actual["category_curves"].query("item == @name")
        np.testing.assert_allclose(
            rows.probability.values.reshape(len(grid), categories[i]), expected[i], atol=1e-12
        )
        np.testing.assert_allclose(
            rows.derivative.values.reshape(len(grid), categories[i]), derivatives[i], atol=1e-10
        )
        info = (derivatives[i] ** 2 / expected[i]).sum(axis=1)
        total += info
        np.testing.assert_allclose(
            actual["item_information"].query("item == @name").information, info, atol=1e-10
        )
    np.testing.assert_allclose(actual["test_information"].information, total, atol=2e-10)


@pytest.mark.parametrize("family", ["rasch", "2pl", "3pl", "grm", "pcm", "rsm"])
def test_eap_extreme_partial_and_empty_against_independent_integration(fitted, family):
    result, _, _, categories, guessing = fitted[family]
    state = result.attrs["state"]
    data = pd.DataFrame(
        [[0] * 4, [k - 1 for k in categories], [0, np.nan, 1, np.nan], [np.nan] * 4],
        columns=state["items"],
        index=["low", "high", "partial", "empty"],
    )
    actual = oe.irt_score(result, data=data)["scores"]
    nodes, w = roots_hermitenorm(121)
    w /= np.sqrt(2 * np.pi)
    p = probabilities(np.array(state["raw_parameters"]), family, categories, guessing, nodes)
    for i in range(3):
        likelihood = np.ones(len(nodes))
        for j, value in enumerate(data.iloc[i]):
            if np.isfinite(value):
                likelihood *= p[j][:, int(value)]
        posterior = w * likelihood / (w * likelihood).sum()
        mean = posterior @ nodes
        sd = np.sqrt(posterior @ ((nodes - mean) ** 2))
        assert actual.eap[i] == pytest.approx(mean, abs=1e-6)
        assert actual.posterior_sd[i] == pytest.approx(sd, abs=1e-6)
    assert actual.eap[3] == 0 and actual.posterior_sd[3] == 1 and actual.observed_items[3] == 0
    assert list(actual.source_index) == list(data.index)
    training = oe.irt_score(result)["scores"]
    np.testing.assert_array_equal(training.eap, result["people"].eap)
    assert (
        "conditional" in " ".join(oe.irt_score(result).attrs["notes"]).lower()
        or oe.irt_score(result).attrs["conditional_item_parameters"]
    )


def test_binary_reductions_and_rating_scale_constraint(fitted):
    data = fitted["rasch"][1]
    r, p = oe.irt_rasch(data=data, items=list(data)), oe.irt_pcm(data=data, items=list(data))
    np.testing.assert_allclose(r["parameters"].estimate, p["parameters"].estimate, atol=1e-6)
    g, two = oe.irt_grm(data=data, items=list(data)), oe.irt_2pl(data=data, items=list(data))
    np.testing.assert_allclose(g["parameters"].estimate, two["parameters"].estimate, atol=1e-10)
    rsm = fitted["rsm"][0]
    assert rsm["parameters"].estimate.iloc[-2:].sum() == pytest.approx(0, abs=1e-15)
    np.testing.assert_allclose(rsm["covariance"].values[-2:, :].sum(axis=0), 0, atol=1e-15)


def test_missing_sample_identity_and_no_implicit_coercion(fitted):
    frame = fitted["rasch"][1].astype(float)
    frame.index = [f"person{i}" for i in range(len(frame))]
    frame.iloc[0, 0], frame.iloc[5, 2] = np.nan, np.nan
    r = oe.irt_rasch(data=frame, items=list(frame))
    assert r.attrs["nobs"] == 598 and r.attrs["dropped_rows"] == 2
    assert r.attrs["state"]["sample_positions"] == [i for i in range(600) if i not in (0, 5)]
    assert list(r["people"].source_index) == [
        v for i, v in enumerate(frame.index) if i not in (0, 5)
    ]
    assert oe.irt_restore(r.to_json()).to_json() == r.to_json()


@pytest.mark.parametrize(
    "change", ["checksum", "covariance", "responses", "sample", "guessing", "nodes", "latent"]
)
def test_tampered_and_rechecksummed_inconsistent_state_rejected(fitted, change):
    envelope = json.loads(fitted["2pl"][0].to_json())
    s = envelope["state"]
    if change == "checksum":
        envelope["checksum"] = "wrong"
    elif change == "covariance":
        s["covariance"][0][0] *= 1.2
    elif change == "responses":
        s["responses"][0][0] = 9
    elif change == "sample":
        s["sample_positions"][0] = 1
    elif change == "guessing":
        s["guessing"][0] = 0.1
    elif change == "nodes":
        s["nodes"][2] += 0.01
    else:
        s["latent_variance"] = 2.0
    if change != "checksum":
        envelope["checksum"] = hashlib.sha256(
            json.dumps(s, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    with pytest.raises(oe.AnalysisError, match="state"):
        oe.irt_restore(envelope)


@pytest.mark.parametrize(
    "bad", ["constant", "gapped", "negative", "bool", "text", "fraction", "inf", "duplicate"]
)
def test_invalid_categories_and_columns_rejected(fitted, bad):
    data = copy.deepcopy(fitted["rasch"][1])
    if bad == "constant":
        data["a"] = 0
    elif bad == "gapped":
        data["a"] *= 2
    elif bad == "negative":
        data.loc[0, "a"] = -1
    elif bad == "bool":
        data["a"] = data["a"].astype(bool)
    elif bad == "text":
        data["a"] = data["a"].astype(str)
    elif bad == "fraction":
        data["a"] = data["a"].astype(float)
        data.loc[0, "a"] = 0.5
    elif bad == "inf":
        data["a"] = data["a"].astype(float)
        data.loc[0, "a"] = np.inf
    else:
        data = pd.concat([data, data[["a"]]], axis=1)
    with pytest.raises(oe.AnalysisError):
        oe.irt_rasch(data=data, items=["a", "b", "c", "d"])


def test_workspace_work_convergence_and_default_device_guards(fitted):
    data = fitted["2pl"][1]
    with use_workspace_budget(1), pytest.raises(oe.AnalysisError, match="workspace"):
        oe.irt_2pl(data=data, items=list(data))
    with pytest.raises(oe.AnalysisError, match="converged|budget"):
        oe.irt_2pl(data=data, items=list(data), max_iter=1, max_eval=2)
    with pytest.raises(oe.AnalysisError):
        oe.irt_rsm(data=data, items=list(data))
    with pytest.raises(oe.AnalysisError):
        oe.irt_3pl(data=data, items=list(data), guessing=[0.5] * 4)
    with pytest.raises(TypeError):
        oe.irt_rasch(data=data, items=list(data), weights="w")
    with torch.device("meta"):
        r = oe.irt_rasch(data=data, items=list(data))
        assert r.attrs["device"] == "cpu"
        assert oe.irt_information(r, theta=[0.0])["test_information"].information[0] > 0


def test_public_capability_and_registry_contract():
    from openecon.econometrics.registry import public_exports

    names = oe.capabilities()["irt"]["procedures"]
    expected = {"irt_rasch", "irt_2pl", "irt_3pl", "irt_grm", "irt_pcm", "irt_rsm",
                "irt_information", "irt_score", "irt_restore", "irt_fit_diagnostics",
                "irt_mh_dif", "irt_diagnostics_restore", "irt_bank_binary", "irt_bank_polytomous",
                "irt_bank_restore", "irt_score_mle", "irt_score_map", "irt_posterior",
                "irt_posterior_restore", "irt_plausible_values", "irt_predictive",
                "irt_test_score_distribution"}
    assert set(names) == expected and expected.issubset(public_exports())
    for name in names:
        assert callable(getattr(oe, name))
    assert oe.capabilities()["irt"]["dataset_support"] is False
