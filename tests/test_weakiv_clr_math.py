"""Independent whole-line, Gaussian and invariance acceptance for MARKET-681."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from openecon.econometrics.weakiv import iv_clr_confidence_set, iv_clr_test
from openecon.econometrics.weakiv.kernels import probability
from openecon.econometrics.weakiv.common import controls
from openecon.econometrics.iv.weak_inference import iv_weak_test

HELPER = Path(__file__).resolve().parents[1] / "scripts" / "verify_weakiv_clr_oracles.py"
module = importlib.util.spec_from_file_location("scalar_clr_independent", HELPER)
oracle = importlib.util.module_from_spec(module)
module.loader.exec_module(oracle)


def data(seed=149, strength=0.2, n=80, k=3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k))
    v = rng.normal(size=n)
    d = strength * z[:, 0] + v
    y = 0.7 * d + 0.8 * v + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, d=d, **{f"z{i}": z[:, i] for i in range(k)}))


def fit(frame, **kwargs):
    return iv_clr_confidence_set(
        frame, "y", "d", instruments=[v for v in frame if v.startswith("z")], **kwargs
    )


@pytest.mark.parametrize("k", [1, 2, 3, 7, 12])
@pytest.mark.parametrize(
    "lr,qt",
    [(0, 3), (3.2, 0), (4.4, 1), (5, 50), (3.8414588207, 1e8), (2.0793918229e-7, 1.5726035934)],
)
def test_conditional_tail_independent_adaptive_integral(k, lr, qt):
    native = probability(lr, qt, k, controls())
    expected = oracle.tail(lr, qt, k)
    assert native["value"] == pytest.approx(expected, rel=2e-10, abs=1e-13)
    assert native["error_estimate"] >= 0


@pytest.mark.parametrize(
    "seed,strength,topology",
    [(0, 0, "all_real"), (20, 0, "two_unbounded_rays"), (4, 1, "bounded_interval")],
)
@pytest.mark.parametrize("known", [False, True])
def test_global_geometry_matches_generalized_eigen_brent_reference(seed, strength, topology, known):
    out = fit(data(seed, strength), omega=[[3.25, 1.5], [1.5, 1.0]] if known else None)
    state = out.attrs["state"]
    ref = oracle.reference(state)
    # Known Ω changes the fixed distribution; the topology assertion belongs to
    # the declared estimated-Ω fixture while both branches share the oracle gate.
    if not known:
        assert out.attrs["topology"] == topology
    assert ref["topology"] == out.attrs["topology"]
    audit = oracle.audit(dict(states=[state], tests=[]))
    assert audit["passed"], audit
    for null in [-1e12, -3, 0, 0.7, 5, 1e12]:
        test = iv_clr_test(out, null=float(null)).attrs["state"]["test"]
        expected = oracle.null_reference(ref, null)
        assert test["conditional_p_value"] == pytest.approx(
            expected["conditional_p_value"], rel=3e-10, abs=2e-12
        )
        member = any(
            (a is None or null >= a) and (b is None or null <= b)
            for a, b in state["solution"]["intervals"]
        )
        assert test["accepted"] == member
    assert out.attrs["global_real_line"] and not out.attrs["robust_subvector_supported"]
    assert out.attrs["known_covariance"] == known


@pytest.mark.parametrize("seed,strength", [(0, 0), (20, 0), (4, 1)])
def test_reporting_units_and_nonsingular_instrument_basis_preserve_inference(seed, strength):
    f = data(seed, strength)
    first = fit(f)
    scaled = f.copy()
    scaled["y"] *= 1e-6
    scaled["d"] *= 1e3
    basis = np.array([[2, 0.2, 0.5], [0, 0.4, -0.3], [0.4, 0.1, 1]])
    scaled[["z0", "z1", "z2"]] = scaled[["z0", "z1", "z2"]].to_numpy() @ basis
    second = fit(scaled)
    assert second.attrs["topology"] == first.attrs["topology"]
    ratio = 1e-9
    for row1, row2 in zip(
        first.attrs["state"]["solution"]["intervals"],
        second.attrs["state"]["solution"]["intervals"],
    ):
        for a, b in zip(row1, row2):
            if a is None:
                assert b is None
            else:
                assert b / ratio == pytest.approx(a, rel=3e-9)
    assert iv_clr_test(first, null=0.7)["clr_test"].iloc[0].conditional_p_value == pytest.approx(
        iv_clr_test(second, null=0.7 * ratio)["clr_test"].iloc[0].conditional_p_value, rel=1e-10
    )


def test_included_control_partialling_and_original_units_known_omega():
    f = data(14, 0.4, n=160)
    f["control"] = np.random.default_rng(4).normal(size=len(f))
    f["y"] += 8 * f.control
    f["d"] -= 3 * f.control
    first = iv_clr_confidence_set(
        f, "y", "d", ["control"], instruments=["z0", "z1", "z2"], omega=[[3.25, 1.5], [1.5, 1.0]]
    )
    f["control"] *= 1e-7
    second = iv_clr_confidence_set(
        f, "y", "d", ["control"], instruments=["z0", "z1", "z2"], omega=[[3.25, 1.5], [1.5, 1.0]]
    )
    assert oracle.audit(dict(states=[first.attrs["state"], second.attrs["state"]]))["passed"]
    assert np.asarray(first.attrs["state"]["solution"]["intervals"]) == pytest.approx(
        np.asarray(second.attrs["state"]["solution"]["intervals"]), rel=1e-10
    )


def test_one_instrument_clr_is_the_exact_chi_square_ar_geometry():
    out = fit(data(19, 0.6, n=100, k=1))
    state = out.attrs["state"]
    assert state["root"]["branch"] == "analytic_one_instrument"
    assert state["root"]["lr"] == pytest.approx(chi2.isf(0.05, 1), rel=2e-12)
    for null in [-1, 0, 0.7, 2, 100]:
        tst = iv_clr_test(out, null=float(null)).attrs["state"]["test"]
        assert tst["conditional_p_value"] == pytest.approx(chi2.sf(tst["statistic"], 1), rel=2e-12)


def test_exact_known_covariance_gaussian_null_sampling_check():
    # This is a deterministic sampling sanity check, not proof of exactness.
    # Reference theorem supplies the size claim; all trials retain fixed k=2.
    rejected = 0
    for seed in range(96):
        rng = np.random.default_rng(8200 + seed)
        z = rng.normal(size=(64, 2))
        v, e = rng.normal(size=(2, 64))
        f = pd.DataFrame(dict(y=0.8 * v + 0.6 * e, d=0.1 * z[:, 0] + v, z0=z[:, 0], z1=z[:, 1]))
        out = fit(f, omega=[[1.0, 0.8], [0.8, 1.0]])
        rejected += not iv_clr_test(out, null=0.0)["clr_test"].iloc[0].accepted
    # Exact binomial 99% interval for 96 draws at .05 is [0,12].
    assert 0 <= rejected <= 12


def test_finite_global_nulls_near_float64_limit_are_stable():
    out = fit(data(20, 0))
    for null in [-1e300, 1e300]:
        test = iv_clr_test(out, null=null)["clr_test"].iloc[0]
        assert np.isfinite(test.statistic) and 0 <= test.conditional_p_value <= 1
        assert test.accepted


def test_small_likelihood_ratio_endpoint_layer_has_independent_tail_accuracy():
    lr, qt = 2.079391822894872e-7, 1.572603593406871
    out = probability(lr, qt, 3, controls())
    assert out["value"] == pytest.approx(oracle.tail(lr, qt, 3), rel=3e-12)
    assert out["order"] <= 512


@pytest.mark.parametrize("seed,strength", [(0, 0), (20, 0), (4, 1)])
def test_global_set_contains_likelihood_minimizer_and_is_nonempty(seed, strength):
    out = fit(data(seed, strength))
    state = out.attrs["state"]
    g = state["geometry"]
    direction = g["eigen_directions"][0]
    beta = -direction[1] / direction[0] * g["coefficient_unit"]
    test = iv_clr_test(out, null=float(beta))["clr_test"].iloc[0]
    assert test.statistic <= 1e-25
    assert test.conditional_p_value == pytest.approx(1.0, abs=2e-14)
    assert test.accepted
    assert any(
        (a is None or beta >= a) and (b is None or beta <= b)
        for a, b in state["solution"]["intervals"]
    )


def test_regular_endpoints_have_the_declared_conditional_probability():
    out = fit(data(4, 1))
    for endpoint in out.attrs["state"]["solution"]["intervals"][0]:
        query = iv_clr_test(out, null=endpoint)["clr_test"].iloc[0]
        assert query.conditional_p_value == pytest.approx(0.05, abs=1e-9)


def test_feasible_query_preserves_existing_scalar_clr_point_convention():
    frame = data(4, 1)
    result = fit(frame)
    for null in [0.0, 0.7, 3.0]:
        existing = iv_weak_test(
            data=frame, y="y", endog=["d"], instruments=["z0", "z1", "z2"], null=null, method="clr"
        )
        query = iv_clr_test(result, null=null)["clr_test"].iloc[0]
        assert query.statistic == pytest.approx(
            float(existing["test"].iloc[0].statistic), rel=1e-11
        )
        assert query.conditional_p_value == pytest.approx(
            float(existing["test"].iloc[0].p_value), abs=2e-9
        )
