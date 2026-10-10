"""Independent natural-parameter ML/OIM and adversarial saved-state acceptance."""

import copy
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from scipy.optimize import minimize
from scipy.stats import norm

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.latent.cfa import CFAState, cfa, cfa_covariance, cfa_restore
from openecon.resources import use_workspace_budget

FACTORS = {"z_factor": ["x1", "x2", "x3"], "a_factor": ["x4", "x5", "x6"]}
NAMES = ["x1", "x2", "x3", "x4", "x5", "x6"]
LOAD = np.array([[1.0, 0.0], [0.8, 0.0], [0.7, 0.0], [0.0, 1.0], [0.0, 0.9], [0.0, 0.6]])
PHI = np.array([[1.0, 0.3], [0.3, 1.2]])
THETA = np.array([0.4, 0.5, 0.6, 0.3, 0.4, 0.5])
SIGMA = LOAD @ PHI @ LOAD.T + np.diag(THETA)
INITIAL = np.r_[0.8, 0.7, 0.9, 0.6, 1.0, 0.3, 1.2, THETA]


def natural_moments(v):
    lam = np.array([[1.0, 0.0], [v[0], 0.0], [v[1], 0.0], [0.0, 1.0], [0.0, v[2]], [0.0, v[3]]])
    phi = np.array([[v[4], v[5]], [v[5], v[6]]])
    return lam @ phi @ lam.T + np.diag(v[7:]), lam, phi


def natural_objective(v, s, n):
    sigma, lam, phi = natural_moments(v)
    if np.linalg.eigvalsh(phi).min() <= 0 or v[7:].min() <= 0:
        return 1e50, np.zeros(13)
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        return 1e50, np.zeros(13)
    inv = np.linalg.inv(sigma)
    weight = inv - inv @ s @ inv
    derivative = []
    for j, f in [(1, 0), (2, 0), (4, 1), (5, 1)]:
        dl = np.zeros((6, 2))
        dl[j, f] = 1
        derivative.append(dl @ phi @ lam.T + lam @ phi @ dl.T)
    for i, j in [(0, 0), (1, 0), (1, 1)]:
        dp = np.zeros((2, 2))
        dp[i, j] = 1
        dp[j, i] = 1
        derivative.append(lam @ dp @ lam.T)
    for j in range(6):
        dt = np.zeros((6, 6))
        dt[j, j] = 1
        derivative.append(dt)
    value = n / 2 * (6 * math.log(2 * math.pi) + logdet + np.trace(inv @ s))
    gradient = n / 2 * np.array([np.sum(weight * d) for d in derivative])
    return value, gradient


def independent_oracle(s, n):
    fit = minimize(
        lambda v: natural_objective(v, s, n),
        INITIAL,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-7, "maxiter": 1000},
    )
    assert np.max(np.abs(natural_objective(fit.x, s, n)[1])) < 2e-5
    # Different parameterization and a finite-difference observed score Hessian;
    # no runtime solver parameter, Torch derivative or information reuse.
    h = np.empty((13, 13))
    for j in range(13):
        step = 2e-5 * max(1.0, abs(fit.x[j]))
        d = np.zeros(13)
        d[j] = step
        h[:, j] = (
            natural_objective(fit.x + d, s, n)[1] - natural_objective(fit.x - d, s, n)[1]
        ) / (2 * step)
    return fit.x, -fit.fun, np.linalg.inv((h + h.T) / 2)


@pytest.fixture(scope="module")
def exact():
    return cfa_covariance(SIGMA.tolist(), n=301, divisor="n", factors=FACTORS)


@pytest.fixture(scope="module")
def sample():
    y = np.random.default_rng(626).multivariate_normal(np.arange(6) * 0.4, SIGMA, size=603)
    frame = pd.DataFrame(
        y,
        columns=NAMES,
        index=pd.date_range("2020-01-01", periods=len(y), tz="Europe/Istanbul", name="day"),
    )
    result = cfa(frame, factors=FACTORS)
    return frame, result


def test_exact_likelihood_and_full_natural_information(exact):
    expected = independent_oracle(SIGMA, 301)
    np.testing.assert_allclose(exact["parameters"]["estimate"], INITIAL, atol=2e-7)
    np.testing.assert_allclose(exact["covariance"], expected[2], atol=3e-9, rtol=2e-6)
    assert exact.attrs["log_likelihood"] == pytest.approx(expected[1], abs=2e-8)
    assert exact.attrs["df_covariance"] == 8
    assert not np.allclose(np.asarray(exact["covariance"]), np.diag(np.diag(exact["covariance"])))
    co = exact["parameters"]
    np.testing.assert_allclose(co["std_error"], np.sqrt(np.diag(expected[2])), rtol=2e-6)
    np.testing.assert_allclose(
        co["p_value"], 2 * norm.sf(np.abs(co["estimate"] / co["std_error"])), atol=2e-14
    )


def test_sample_full_oim_vs_separate_natural_ml(sample):
    frame, result = sample
    centered = frame.to_numpy() - frame.to_numpy().mean(0)
    s = centered.T @ centered / len(frame)
    beta, ll, cov = independent_oracle(s, len(frame))
    np.testing.assert_allclose(result["parameters"]["estimate"], beta, rtol=3e-6, atol=5e-7)
    np.testing.assert_allclose(result["covariance"], cov, rtol=5e-5, atol=1e-8)
    assert result.attrs["log_likelihood"] == pytest.approx(ll, abs=2e-8)


def test_raw_summary_and_divisor_equivalence(sample):
    frame, result = sample
    summary = cfa_covariance(
        frame.cov(), n=len(frame), divisor="n-1", means=frame.mean().tolist(), factors=FACTORS
    )
    np.testing.assert_allclose(summary["parameters"], result["parameters"], rtol=2e-6, atol=3e-7)
    np.testing.assert_allclose(summary["covariance"], result["covariance"], rtol=2e-6, atol=2e-9)
    assert summary.attrs["log_likelihood"] == pytest.approx(
        result.attrs["log_likelihood"], abs=2e-8
    )


def test_marker_raw_units_and_rescaling(sample):
    frame, result = sample
    units = np.array([0.01, 100.0, 3.0, 20.0, 0.2, 4.0])
    rescaled = cfa(frame * units, factors=FACTORS)
    load = result["loadings"].to_numpy() * units[:, None] / units[[0, 3]][None, :]
    np.testing.assert_allclose(rescaled["loadings"], load, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(
        rescaled["factor_covariance"],
        result["factor_covariance"].to_numpy() * units[[0, 3]][:, None] * units[[0, 3]][None, :],
        rtol=2e-6,
    )
    raw_derivative = np.r_[
        units[1] / units[0],
        units[2] / units[0],
        units[4] / units[3],
        units[5] / units[3],
        units[0] ** 2,
        units[0] * units[3],
        units[3] ** 2,
        units**2,
    ]
    np.testing.assert_allclose(
        rescaled["covariance"],
        result["covariance"].to_numpy() * raw_derivative[:, None] * raw_derivative[None, :],
        rtol=3e-6,
        atol=1e-6,
    )
    expected_ll = result.attrs["log_likelihood"] - len(frame) * np.log(units).sum()
    assert rescaled.attrs["log_likelihood"] == pytest.approx(expected_ll, abs=2e-8)


def test_marker_boundary_gate_is_invariant_to_factor_units(exact):
    units = np.array([1e-8, 2.0, 0.5, 1e4, 3.0, 0.25])
    transformed = cfa_covariance(
        (SIGMA * units[:, None] * units[None, :]).tolist(),
        n=301,
        divisor="n",
        factors=FACTORS,
    )
    derivatives = np.r_[
        units[1] / units[0],
        units[2] / units[0],
        units[4] / units[3],
        units[5] / units[3],
        units[0] ** 2,
        units[0] * units[3],
        units[3] ** 2,
        units**2,
    ]
    np.testing.assert_allclose(
        transformed["covariance"],
        exact["covariance"].to_numpy() * derivatives[:, None] * derivatives[None, :],
        rtol=3e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        cfa_restore(transformed)["factor_covariance"],
        transformed["factor_covariance"],
        rtol=2e-8,
        atol=0.0,
    )


def test_unit_variance_reparameterization(exact):
    fixed = cfa_covariance(
        SIGMA.tolist(), n=301, divisor="n", factors=FACTORS, identification="unit_variance"
    )
    expected_load = LOAD * np.sqrt(np.diag(PHI))[None, :]
    expected_phi = PHI / np.sqrt(np.diag(PHI))[:, None] / np.sqrt(np.diag(PHI))[None, :]
    np.testing.assert_allclose(fixed["loadings"], expected_load, atol=3e-7)
    np.testing.assert_allclose(fixed["factor_covariance"], expected_phi, atol=3e-7)
    assert fixed.attrs["log_likelihood"] == pytest.approx(exact.attrs["log_likelihood"], abs=3e-8)
    assert fixed.attrs["df_covariance"] == 8
    np.testing.assert_allclose(np.diag(fixed["factor_covariance"]), 1.0, atol=2e-14)

    def transform(v):
        a, b = np.sqrt(v[4]), np.sqrt(v[6])
        return np.r_[a, v[0] * a, v[1] * a, b, v[2] * b, v[3] * b, v[5] / (a * b), v[7:]]

    derivative = np.empty((13, 13))
    for j in range(13):
        d = np.zeros(13)
        d[j] = 1e-5
        derivative[:, j] = (transform(INITIAL + d) - transform(INITIAL - d)) / 2e-5
    expected_cov = derivative @ exact["covariance"].to_numpy() @ derivative.T
    np.testing.assert_allclose(fixed["covariance"], expected_cov, rtol=5e-6, atol=3e-9)


def test_replay_is_optimizer_free_and_typed_immutable(sample, monkeypatch):
    frame, result = sample
    import openecon.econometrics.latent.cfa as impl

    monkeypatch.setattr(impl, "_solve", lambda *a, **k: pytest.fail("Replay ran the optimizer"))
    bundle = CFAState(payload=result.attrs["cfa_state"])
    with pytest.raises(TypeError):
        bundle.payload["results"]["n"] = 3
    restored = cfa_restore(json.dumps(bundle.model_dump(), sort_keys=True))
    np.testing.assert_array_equal(restored["covariance"], result["covariance"])
    source = restored.attrs["cfa_state"]["source"]
    from openecon.econometrics.mi.common import _decode_index

    pd.testing.assert_index_equal(_decode_index(source["index"]), frame.index)


def resign(state):
    state["digest"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in state.items() if k != "digest"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    "field",
    [
        "parameters",
        "covariance",
        "score_solver",
        "observed_information_solver",
        "natural_jacobian",
        "implied_covariance",
        "loadings",
        "factor_covariance",
        "sample_covariance_ml",
        "scales",
        "information_eigenvalues",
    ],
)
def test_recomputed_digest_cannot_forge_cached_numeric_state(exact, field):
    state = copy.deepcopy(exact.attrs["cfa_state"])
    target = state["results"][field]
    if isinstance(target[0], list):
        target[0][0] += 0.01
    else:
        target[0] += 0.01
    resign(state)
    with pytest.raises((AnalysisError, ValidationError), match="replay"):
        cfa_restore(state)


def test_source_sample_tampering(sample):
    _, result = sample
    state = copy.deepcopy(result.attrs["cfa_state"])
    state["source"]["rows"][0][0] += 10
    resign(state)
    with pytest.raises((AnalysisError, ValidationError)):
        cfa_restore(state)


def test_missing_positions_nullable_and_typed_index(sample):
    frame, _ = sample
    frame = frame.iloc[:80].astype("Float64")
    frame.iloc[[0, 3], 1] = pd.NA
    with pytest.raises(AnalysisError, match="missing"):
        cfa(frame, factors=FACTORS)
    result = cfa(frame, factors=FACTORS, missing="drop")
    assert result.attrs["n"] == 78
    state = result.attrs["cfa_state"]
    assert state["source"]["positions"] == [i for i in range(80) if i not in (0, 3)]
    assert state["source"]["dtypes"] == ["Float64"] * 6
    cfa_restore(result)
    state = copy.deepcopy(state)
    state["source"]["positions"][0] = 0
    resign(state)
    with pytest.raises((AnalysisError, ValidationError), match="source.positions"):
        cfa_restore(state)


def test_local_identification_is_checked_beyond_parameter_count():
    # Two-indicator factor disconnected from a three-indicator factor has q<=m
    # but its three factor/error variances are not locally identified.
    s = np.zeros((5, 5))
    s[:2, :2] = [[1.4, 0.8], [0.8, 1.14]]
    s[2:, 2:] = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.48], [0.6, 0.48, 1.0]])
    with pytest.raises(AnalysisError, match="rank deficient|information"):
        cfa_covariance(
            s.tolist(), n=500, divisor="n", factors={"a": ["a1", "a2"], "b": ["b1", "b2", "b3"]}
        )


def test_boundary_fit_not_repaired():
    # An exact one-factor population with a zero residual variance is a
    # Heywood boundary, even though the observed covariance is nonsingular.
    loading = np.array([1.0, 0.8, 0.6, 0.5])
    sigma = np.outer(loading, loading) + np.diag([0.0, 0.4, 0.5, 0.6])
    with pytest.raises(AnalysisError, match="boundary|information|converg"):
        cfa_covariance(
            sigma.tolist(), n=400, divisor="n", factors={"f": ["a", "b", "c", "d"]}, tolerance=1e-9
        )


def test_preallocation_budget_and_original_rows(monkeypatch):
    frame = pd.DataFrame(np.tile(np.arange(6), (10001, 1)), columns=NAMES)
    monkeypatch.setattr(
        pd.DataFrame, "copy", lambda *a, **k: pytest.fail("Copied before row admission")
    )
    with pytest.raises(AnalysisError, match="n must"):
        cfa(frame, factors=FACTORS)
    with pytest.raises(AnalysisError, match="max_work"):
        cfa_covariance(SIGMA.tolist(), n=301, divisor="n", factors=FACTORS, max_work=1)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        cfa_covariance(
            np.eye(16).tolist(),
            n=301,
            divisor="n",
            factors={f"f{i}": [f"c{j}" for j in range(4 * i, 4 * i + 4)] for i in range(4)},
        )


def test_ambient_defaults_and_rng_preserved(exact):
    torch_state = torch.random.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        restored = cfa_restore(exact)
        assert torch.get_default_dtype() == torch.float32
        np.testing.assert_array_equal(restored["covariance"], exact["covariance"])
        assert torch.equal(torch_state, torch.random.get_rng_state())
    finally:
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": True},
        {"divisor": "automatic"},
        {"identification": "free"},
        {"max_iterations": False},
        {"tolerance": float("nan")},
        {"max_work": False},
    ],
)
def test_invalid_declarations(kwargs):
    options = {"n": 301, "divisor": "n", "factors": FACTORS} | kwargs
    with pytest.raises(AnalysisError):
        cfa_covariance(SIGMA.tolist(), **options)


def test_nonconvergence_refused():
    with pytest.raises(AnalysisError, match="Neither declared"):
        cfa_covariance(
            SIGMA.tolist(), n=301, divisor="n", factors=FACTORS, max_iterations=1, tolerance=1e-10
        )


def test_tiny_units_do_not_mask_asymmetry():
    s = SIGMA * 1e-14
    s[0, 1] *= 0.9
    with pytest.raises(AnalysisError, match="symmetric"):
        cfa_covariance(s.tolist(), n=301, divisor="n", factors=FACTORS)


def test_tiny_units_do_not_mask_forged_covariance():
    result = cfa_covariance((SIGMA * 1e-12).tolist(), n=301, divisor="n", factors=FACTORS)
    state = copy.deepcopy(result.attrs["cfa_state"])
    assert state["results"]["covariance"][4][4] < 1e-20
    state["results"]["covariance"][4][4] = 1e-10
    resign(state)
    with pytest.raises((AnalysisError, ValidationError), match="results.covariance"):
        cfa_restore(state)


def test_meta_default_device_on_fresh_fit():
    old_device, old_dtype = torch.get_default_device(), torch.get_default_dtype()
    rng = torch.random.get_rng_state().clone()
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.float32)
        result = cfa_covariance(SIGMA.tolist(), n=301, divisor="n", factors=FACTORS)
        assert result.attrs["n"] == 301
        assert torch.get_default_device().type == "meta"
        assert torch.get_default_dtype() == torch.float32
        assert torch.equal(rng, torch.random.get_rng_state())
    finally:
        torch.set_default_device(old_device)
        torch.set_default_dtype(old_dtype)


def test_integer_source_precision_refused(sample):
    frame, _ = sample
    frame = frame.iloc[:40].copy()
    frame["x1"] = pd.Series([9007199254740993] + list(range(39)), index=frame.index, dtype="Int64")
    with pytest.raises(AnalysisError, match="exactly representable"):
        cfa(frame, factors=FACTORS)


def test_oversized_index_refused_before_source_copy(sample, monkeypatch):
    frame, _ = sample
    frame = frame.iloc[:40].copy()
    frame.index = pd.Index(["a" * 2048] * len(frame))
    monkeypatch.setattr(
        pd.DataFrame, "copy", lambda *a, **k: pytest.fail("Copied before index admission")
    )
    with pytest.raises(AnalysisError, match="index strings"):
        cfa(frame, factors=FACTORS)


@pytest.mark.parametrize("field", ["covariance", "means"])
def test_summary_state_refuses_coerced_boolean_primitives(exact, field):
    state = copy.deepcopy(exact.attrs["cfa_state"])
    if field == "means":
        state["source"][field] = [True] * 6
    else:
        state["source"][field][0][0] = True
    resign(state)
    with pytest.raises((AnalysisError, ValidationError), match="real numeric"):
        cfa_restore(state)


def test_mapping_alignment_refused_before_outer_join(monkeypatch):
    source = {
        name: pd.Series(np.arange(20), index=np.arange(i * 20, (i + 1) * 20))
        for i, name in enumerate(NAMES)
    }
    monkeypatch.setattr(
        pd.DataFrame,
        "__init__",
        lambda *a, **k: pytest.fail("Materialized before alignment admission"),
    )
    with pytest.raises(AnalysisError, match="identical typed row indexes"):
        cfa(source, factors=FACTORS)


@pytest.mark.parametrize("scale", [1e160, 1e-170])
def test_unrepresentable_original_unit_uncertainty_refused(scale):
    with pytest.raises(AnalysisError, match="uncertainty is not representable"):
        cfa_covariance((SIGMA * scale).tolist(), n=301, divisor="n", factors=FACTORS)


@pytest.mark.parametrize('constructor', ['copy', 'construct'])
def test_direct_typed_table_refuses_rehashed_covariance_forgery(exact, constructor):
    good = CFAState(payload=exact.attrs['cfa_state'])
    payload = good.model_dump()['payload']
    payload['results']['covariance'] = [[4*v for v in row] for row in payload['results']['covariance']]
    resign(payload)
    forged = good.model_copy(update={'payload': payload}) if constructor == 'copy' else CFAState.model_construct(payload=payload)
    with pytest.raises(AnalysisError, match='results.covariance'):
        forged.to_tables()


@pytest.mark.parametrize('field', ['covariance', 'observed_information_solver', 'natural_jacobian', 'loadings', 'parameters'])
def test_cached_shape_refused_before_derivative_replay(exact, monkeypatch, field):
    import openecon.econometrics.latent.cfa as implementation
    payload = copy.deepcopy(exact.attrs['cfa_state'])
    payload['results'][field] = []
    resign(payload)
    monkeypatch.setattr(implementation, '_evaluate', lambda *a, **k: pytest.fail('Derivative replay before cache shape admission'))
    with pytest.raises((AnalysisError, ValidationError), match='results.' + field + ' shape'):
        cfa_restore(payload)


@pytest.mark.parametrize('target', ['covariance', 'means', 'tolerance', 'cached_covariance'])
def test_unrepresentable_integer_is_typed_refusal_before_numerics(exact, monkeypatch, target):
    import openecon.econometrics.latent.cfa as implementation
    huge = 2**1024
    monkeypatch.setattr(implementation, '_evaluate', lambda *a, **k: pytest.fail('Numerical replay before integer refusal'))
    if target == 'cached_covariance':
        saved = copy.deepcopy(exact.attrs['cfa_state'])
        saved['results']['covariance'][0][0] = huge
        resign(saved)
        with pytest.raises((AnalysisError, ValidationError), match='finite'):
            cfa_restore(saved)
        return
    kwargs = dict(n=301, divisor='n', factors=FACTORS)
    covariance = SIGMA.tolist()
    if target == 'covariance':
        covariance[0][0] = huge
    elif target == 'means':
        kwargs['means'] = [huge, 0., 0., 0., 0., 0.]
    else:
        kwargs['tolerance'] = huge
    with pytest.raises(AnalysisError, match='finite'):
        cfa_covariance(covariance, **kwargs)
