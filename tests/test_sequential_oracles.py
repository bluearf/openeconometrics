"""Independent development-only Gaussian sequential oracles.

R fixtures come from actual gsDesign.  The separate Python integrator propagates
Brownian scores on uniform Simpson grids rather than the SDK's Z-coordinate
Gauss-Legendre panels.  Exact one-look formulas and adaptive two-look integrals
provide additional checks with no production-kernel imports.
"""
from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

import numpy as np
import openecon as oe
import pytest
from numpy.testing import assert_allclose

scipy = pytest.importorskip("scipy")
from scipy.integrate import quad  # noqa: E402
from scipy.special import ndtr  # noqa: E402
from scipy.stats import norm  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures/group_sequential/gsdesign.json"
SQRT_2PI = math.sqrt(2 * math.pi)


def reference_cases():
    return json.loads(FIXTURE.read_text())["cases"]


STANDARD_INDICES = [i for i in range(24) if i % 3 == 0]


def sdk_design(case):
    return oe.sequential_design(
        fractions=case["fractions"], spending=case["law"], sides=case["sides"],
        alpha=case["alpha"],
        param=case["parameter"] if case["law"] in ("HSD", "power") else None,
        effect=case["effect"], information=case["information"],
    )


def table_probability(result, table_name, summary_prefix=""):
    stage = result[table_name]
    summary = result["summary"].iloc[0]
    return {
        "upper": stage["upper_first_crossing"].to_numpy(dtype=float),
        "lower": stage["lower_first_crossing"].to_numpy(dtype=float),
        "continuation": stage["continuation_probability"].to_numpy(dtype=float),
        "reach": stage["reach_probability"].to_numpy(dtype=float),
        "power": float(summary[summary_prefix + "rejection_probability"]),
        "expected_fraction": float(summary[summary_prefix + "expected_information"])
            / float(summary["maximum_information"]),
    }


def assert_public_geometry(result, case, information, effect):
    t = np.asarray(case["fractions"])
    assert_allclose(result["canonical_covariance"], gaussian_covariance(t), atol=3e-16)
    assert_allclose(result["canonical_means"]["null_mean"], 0, atol=0)
    assert_allclose(result["canonical_means"]["alternative_mean"],
                    effect * np.sqrt(information * t), atol=2e-15)
    assert_allclose(result["boundaries"]["information"], information * t, atol=1e-13)
    assert_allclose(result["boundaries"]["fraction"], t, atol=0)
    assert_allclose(result["boundaries"]["cumulative_total_alpha"],
                    case["spending"], atol=4e-16)
    assert_allclose(result["boundaries"]["incremental_total_alpha"],
                    np.diff(np.r_[0, case["spending"]]), atol=4e-16)
    assert list(result["boundaries"].index) == [f"look_{i+1}" for i in range(len(t))]
    if case["sides"] == 2:
        assert_allclose(result["boundaries"]["lower"], -result["boundaries"]["upper"], atol=0)
        assert result["boundaries"]["lower_present"].all()
    else:
        assert result["boundaries"]["lower"].isna().all()
        assert not result["boundaries"]["lower_present"].any()


def spending_formula(fractions, alpha, sides, law, parameter):
    t = np.asarray(fractions, dtype=float)
    a = alpha / sides
    if law == "LDOF":
        out = 2 * norm.sf(norm.isf(a / 2) / np.sqrt(t))
    elif law == "LDPocock":
        out = a * np.log1p(math.expm1(1) * t)
    elif law == "HSD":
        out = a * t if parameter == 0 else a * np.expm1(-parameter * t) / math.expm1(-parameter)
    elif law == "power":
        out = a * t ** parameter
    else:
        raise AssertionError(law)
    return sides * out


def gaussian_covariance(fractions):
    t = np.asarray(fractions, dtype=float)
    return np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))


def two_look_probability(fractions, bounds, drift, sides):
    """Adaptive integral over the sole previous Z; no grid/SDK dependency."""
    t = np.asarray(fractions, dtype=float)
    b = np.asarray(bounds, dtype=float)
    assert len(t) == 2
    mu = drift * np.sqrt(t)
    ratio = math.sqrt(t[0] / t[1])
    sd = math.sqrt(1 - ratio * ratio)
    increment = drift * (t[1] - t[0]) / math.sqrt(t[1])
    lo = -math.inf if sides == 1 else -b[0]
    upper = [norm.sf(b[0] - mu[0])]
    lower = [0.0 if sides == 1 else norm.cdf(-b[0] - mu[0])]
    upper.append(quad(lambda z: norm.pdf(z - mu[0]) * norm.sf((b[1] - ratio*z - increment) / sd),
                      lo, b[0], epsabs=2e-13, epsrel=2e-12, limit=200)[0])
    lower.append(0.0 if sides == 1 else quad(
        lambda z: norm.pdf(z - mu[0]) * norm.cdf((-b[1] - ratio*z - increment) / sd),
        lo, b[0], epsabs=2e-13, epsrel=2e-12, limit=200)[0])
    upper, lower = np.asarray(upper), np.asarray(lower)
    continuation = 1 - np.cumsum(upper + lower)
    reach = np.r_[1, continuation[:-1]]
    return {"upper": upper, "lower": lower, "continuation": continuation,
            "reach": reach, "power": float(np.sum(upper + lower)),
            "expected_fraction": float(np.diff(np.r_[0,t]) @ reach)}


def _simpson_grid(lo, hi, points):
    assert points >= 3 and points % 2 == 1 and hi > lo
    nodes = np.linspace(lo, hi, points)
    weights = np.ones(points)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    return nodes, weights * ((hi-lo) / (3*(points-1)))


def brownian_probability(fractions, bounds, drift, sides, *, points=1025):
    """Uniform Simpson propagation of S(t)=drift*t+W(t), in normalized time.

    Unbounded one-sided lower ranges truncate at marginal mean minus 12 SD;
    the explicit union tail bound is K*Phi(-12), not hidden grid mass clipping.
    """
    t = np.asarray(fractions, dtype=float)
    b = np.asarray(bounds, dtype=float)
    upper, lower, continuation = [], [], []
    previous = density = weights = None
    for i, time in enumerate(t):
        mean, sd = drift*time, math.sqrt(time)
        ub = b[i]*sd
        lb = -math.inf if sides == 1 else -ub
        if i == 0:
            up = float(norm.sf((ub-mean)/sd))
            low = 0.0 if sides == 1 else float(norm.cdf((lb-mean)/sd))
        else:
            dt = time-t[i-1]
            ds = math.sqrt(dt)
            shift = previous+drift*dt
            weighted = density*weights
            up = float(weighted @ ndtr((shift-ub)/ds))
            low = 0.0 if sides == 1 else float(weighted @ ndtr((lb-shift)/ds))
        upper.append(up)
        lower.append(low)
        continuation.append(1-sum(upper)-sum(lower))
        if i == len(t)-1:
            break
        start = mean-12*sd if sides == 1 else lb
        nodes, new_weights = _simpson_grid(start, ub, points)
        if i == 0:
            new_density = np.exp(-.5*((nodes-mean)/sd)**2) / (sd*SQRT_2PI)
        else:
            new_density = np.empty(points)
            for row in range(0,points,128):
                z = (nodes[row:row+128,None]-shift[None,:])/ds
                new_density[row:row+128] = (np.exp(-.5*z*z) @ weighted)/(ds*SQRT_2PI)
        previous, weights, density = nodes, new_weights, new_density
    reach = np.r_[1,np.asarray(continuation[:-1])]
    return {"upper":np.asarray(upper), "lower":np.asarray(lower),
            "continuation":np.asarray(continuation), "reach":reach,
            "power":float(sum(upper)+sum(lower)),
            "expected_fraction":float(np.diff(np.r_[0,t]) @ reach),
            "tail_bound":len(t)*float(norm.cdf(-12))}


def assert_probability(actual, expected, *, atol=3e-8):
    for key in ("upper","lower","continuation","reach","power","expected_fraction"):
        assert_allclose(actual[key], expected[key], atol=atol, rtol=2e-8, err_msg=key)


def test_r_fixture_provenance_and_complete_cells():
    fixture=json.loads(FIXTURE.read_text())
    assert fixture["provenance"]["gsDesign"] == "3.11.0"
    assert fixture["provenance"]["integration_r"] == 80
    assert fixture["provenance"]["boundary_tolerance"] == 1e-14
    protocol = json.loads(FIXTURE.with_name("protocol.json").read_text())
    assert protocol["fixture_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert protocol["producer_sha256"] == hashlib.sha256(
        FIXTURE.with_name("generate_gsdesign.R").read_bytes()).hexdigest()
    diagnostic = protocol["diagnostic_reference"]
    assert diagnostic["fixture_sha256"] == hashlib.sha256(
        FIXTURE.with_name(diagnostic["fixture"]).read_bytes()).hexdigest()
    assert len(fixture["cases"]) == 36
    assert {(case["law"],case["sides"]) for case in fixture["cases"] if case["id"].endswith("standard")} == {
        (law,sides) for law in ("LDOF","LDPocock","HSD","power") for sides in (1,2)}


@pytest.mark.parametrize("index",range(36))
def test_independent_r_fixture_probability_and_spending_identity(index):
    case=reference_cases()[index]
    expected=spending_formula(case["fractions"],case["alpha"],case["sides"],case["law"],case["parameter"])
    assert_allclose(case["spending"],expected,atol=4e-16,rtol=3e-7)
    assert_allclose(np.cumsum(np.asarray(case["null"]["upper"])+case["null"]["lower"]),
                    case["spending"],atol=2e-10,rtol=2e-7)
    assert_allclose(case["canonical_covariance"],gaussian_covariance(case["fractions"]),atol=3e-16)
    for name in ("null","alternative","solution_probabilities"):
        probability=case[name]
        up,lo=np.asarray(probability["upper"]),np.asarray(probability["lower"])
        assert np.all(up>=0) and np.all(lo>=0)
        assert_allclose(np.cumsum(up+lo)+probability["continuation"],1,atol=4e-16)
        reach=np.r_[1,np.asarray(probability["continuation"][:-1])]
        assert_allclose(probability["reach"],reach,atol=4e-16)
        assert_allclose(probability["expected_fraction"],np.diff(np.r_[0,case["fractions"]])@reach,atol=4e-16)
    assert abs(case["solution_probabilities"]["power"]-case["target_power"])<2e-10


@pytest.mark.parametrize("index",[i for i in range(24) if i%3==1])
def test_adaptive_two_look_oracle_against_actual_r(index):
    case=reference_cases()[index]
    for name,drift in (("null",0),("alternative",case["effect"]*math.sqrt(case["information"]))):
        result=two_look_probability(case["fractions"],case["boundaries"],drift,case["sides"])
        assert_probability(result,case[name],atol=8e-10)


@pytest.mark.parametrize("index",[i for i in range(24) if i%3==0])
def test_independent_brownian_simpson_oracle_against_actual_r(index):
    case=reference_cases()[index]
    for name,drift in (("null",0),("alternative",case["effect"]*math.sqrt(case["information"]))):
        coarse=brownian_probability(case["fractions"],case["boundaries"],drift,case["sides"],points=1025)
        fine=brownian_probability(case["fractions"],case["boundaries"],drift,case["sides"],points=2049)
        assert_probability(coarse,fine,atol=2e-8)
        assert_probability(fine,case[name],atol=3e-9)


@pytest.mark.parametrize("index", range(36))
def test_public_design_against_full_actual_r_geometry(index):
    case = reference_cases()[index]
    result = sdk_design(case)
    assert_public_geometry(result, case, case["information"], case["effect"])
    # The strongest supported R quadrature (r=80) still moves the final
    # front-loaded HSD bound by 2.75e-7 with an 8.25e-6 final alpha increment.
    # We check that bound explicitly and the full crossing probabilities at
    # their tighter scale; independent Simpson stress checks below distinguish
    # root conditioning from a spending/probability error.
    assert_allclose(result["boundaries"]["upper"], case["boundaries"], atol=3e-7, rtol=0)
    assert_probability(table_probability(result, "null_stages", "null_"), case["null"], atol=1e-9)
    assert_probability(table_probability(result, "alternative_stages"), case["alternative"], atol=8e-9)
    # Every directional crossing is a first exit. Full conservation and the
    # expected-information stopping identity are checked independently of R.
    for table_name, prefix in (("null_stages", "null_"), ("alternative_stages", "")):
        actual = table_probability(result, table_name, prefix)
        assert_allclose(np.cumsum(actual["upper"] + actual["lower"]) + actual["continuation"],
                        1, atol=2e-12)
        assert_allclose(actual["reach"], np.r_[1, actual["continuation"][:-1]], atol=2e-12)
        assert_allclose(actual["expected_fraction"], np.diff(np.r_[0, case["fractions"]]) @ actual["reach"],
                        atol=2e-12)
    summary = result["summary"].iloc[0]
    assert summary["total_alpha"] == case["alpha"]
    assert summary["per_direction_alpha"] == case["alpha"] / case["sides"]
    assert abs(summary["null_rejection_probability"] - case["alpha"]) < 2e-12


@pytest.mark.parametrize("index", STANDARD_INDICES)
def test_public_information_inverse_against_actual_r(index):
    case = reference_cases()[index]
    design = sdk_design(case)
    result = oe.sequential_information(design, case["effect"], power=case["target_power"])
    info = float(result["summary"].iloc[0]["maximum_information"])
    assert_allclose(info, case["information_solution"], atol=2e-5, rtol=2e-8)
    assert_public_geometry(result, case, info, case["effect"])
    assert_allclose(result["boundaries"]["upper"], design["boundaries"]["upper"], atol=0)
    assert_probability(table_probability(result, "alternative_stages"), case["solution_probabilities"], atol=1e-8)
    receipt = result["information_inversion"].iloc[0]
    assert receipt["power_low"] < case["target_power"] <= receipt["power_high"]
    assert receipt["drift_high"] - receipt["drift_low"] <= 2.1e-6
    assert receipt["power_high"] - case["target_power"] <= 2e-6
    independent = brownian_probability(case["fractions"], result["boundaries"]["upper"].to_numpy(),
                                      case["effect"] * math.sqrt(info), case["sides"], points=2049)
    assert_probability(table_probability(result, "alternative_stages"), independent, atol=3e-9)
    assert abs(independent["power"] - case["target_power"]) < 1e-8


@pytest.mark.parametrize("index", STANDARD_INDICES)
def test_fixed_boundaries_signed_effect_and_information_units(index):
    case = reference_cases()[index]
    design = sdk_design(case)
    negative = oe.sequential_power(design, -case["effect"])
    positive = table_probability(design, "alternative_stages")
    unfavorable = table_probability(negative, "alternative_stages")
    for name in ("upper", "lower", "fraction", "cumulative_total_alpha"):
        assert_allclose(negative["boundaries"][name].to_numpy(dtype=float),
                        design["boundaries"][name].to_numpy(dtype=float), atol=0, equal_nan=True)
    assert_public_geometry(negative, case, case["information"], -case["effect"])
    if case["sides"] == 2:
        assert_allclose(unfavorable["upper"], positive["lower"], atol=3e-13)
        assert_allclose(unfavorable["lower"], positive["upper"], atol=3e-13)
        for key in ("power", "expected_fraction", "continuation", "reach"):
            assert_allclose(unfavorable[key], positive[key], atol=3e-13)
    else:
        independent = brownian_probability(case["fractions"], case["boundaries"],
                                          -case["effect"]*math.sqrt(case["information"]), 1, points=2049)
        assert_probability(unfavorable, independent, atol=3e-9)
        assert unfavorable["power"] < case["alpha"] < positive["power"]
        assert_allclose(unfavorable["lower"], 0, atol=0)
    # A reciprocal change of effect and information units preserves the
    # complete Gaussian law while dimensional expected information scales.
    scale = 2.5
    rescaled = oe.sequential_power(design, case["effect"] / scale, case["information"] * scale**2)
    assert_probability(table_probability(rescaled, "alternative_stages"), positive, atol=4e-13)
    assert_allclose(rescaled["canonical_means"], design["canonical_means"], atol=2e-15)
    assert_allclose(rescaled["summary"]["expected_information"],
                    design["summary"]["expected_information"] * scale**2, atol=2e-11)


@pytest.mark.parametrize("index", STANDARD_INDICES)
def test_continuous_information_inverse_units(index):
    case = reference_cases()[index]
    design = sdk_design(case)
    first = oe.sequential_information(design, case["effect"], power=case["target_power"])
    second = oe.sequential_information(design, case["effect"] / 3, power=case["target_power"])
    assert_allclose(second["summary"]["maximum_information"],
                    first["summary"]["maximum_information"] * 9, atol=2e-12, rtol=2e-14)
    assert_probability(table_probability(second, "alternative_stages"),
                       table_probability(first, "alternative_stages"), atol=5e-13)
    if case["sides"] == 2:
        mirrored = oe.sequential_information(design, -case["effect"], power=case["target_power"])
        assert_allclose(mirrored["summary"]["maximum_information"],
                        first["summary"]["maximum_information"], atol=2e-12)


@pytest.mark.parametrize("case_id", ("HSD_2_front_loaded", "power_2_early_power", "LDOF_2_small_alpha"))
def test_independent_simpson_stress_calibration_and_alternative(case_id):
    case = next(item for item in reference_cases() if item["id"] == case_id)
    result = sdk_design(case)
    bounds = result["boundaries"]["upper"].to_numpy(dtype=float)
    # These small increments/closely spaced looks are precisely the cells
    # where R boundary position is more sensitive than crossing probability.
    # Integrate the SDK's complete fixed boundaries with another method.
    for name, drift, table_name, prefix in (
        ("null", 0., "null_stages", "null_"),
        ("alternative", case["effect"]*math.sqrt(case["information"]), "alternative_stages", ""),
    ):
        coarse = brownian_probability(case["fractions"], bounds, drift, case["sides"], points=2049)
        fine = brownian_probability(case["fractions"], bounds, drift, case["sides"], points=4097)
        assert_probability(coarse, fine, atol=2e-9)
        assert_probability(table_probability(result, table_name, prefix), fine, atol=8e-10)
        if name == "null":
            assert_allclose(np.cumsum(fine["upper"]+fine["lower"]), case["spending"], atol=8e-10, rtol=0)
