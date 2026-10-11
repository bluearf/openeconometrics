"""PPS without-replacement estimation: R survey/sampling references, exhaustive enumeration, NumPy oracles."""

import json
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError

# R 4.6.1, survey 4.5 (svydesign(fpc=~pi, pps=ppsmat(P, tolerance=0) / "overton" / HR(psum) / HR() /
# "brewer", variance="HT"/"YG"; probs=~pi without fpc) and sampling 2.11 (inclusionprobabilities,
# UPsampfordpi2, varHT): tests/reference/survey_calib_pps.R. Covariances are row-major; "_ratio" holds
# (ratio, variance).
R_PPS = {
    "piA": [
        0.17142857142857143, 0.2857142857142857, 0.45714285714285713, 0.11428571428571428, 0.51428571428571423,
        0.22857142857142856, 0.34285714285714286, 0.40000000000000002, 0.5714285714285714, 0.17142857142857143,
        0.2857142857142857, 0.45714285714285713
    ],
    "piB": [
        0.20000000000000001, 0.45000000000000001, 0.10000000000000001, 0.29999999999999999, 0.55000000000000004,
        0.25, 0.14999999999999999, 0.40000000000000002, 0.34999999999999998, 0.25
    ],
    "PA_sample": [
        0.2857142857142857, 0.12345702013789044, 0.077198944626741359, 0.14027212737322231, 0.12345702013789044,
        0.51428571428571423, 0.15023156423526426, 0.26587172286224642, 0.077198944626741359,
        0.15023156423526426, 0.34285714285714286, 0.17041585238702442, 0.14027212737322231, 0.26587172286224642,
        0.17041585238702442, 0.5714285714285714
    ],
    "PB_sample": [
        0.20000000000000001, 0.086638633613900951, 0.057704989130013866, 0.086638633613900951,
        0.55000000000000004, 0.18421867756799967, 0.057704989130013866, 0.18421867756799967, 0.40000000000000002
    ],
    "psum": [1.5738775510204079, 1.075],
    "exact_HT_total": [318.18434343434342, 72.224747474747474],
    "exact_HT_total_vcov": [697.01738100235229, 235.6292175751461, 235.62921757514619, 363.46552484200402],
    "exact_HT_mean": [16.376527164023916, 3.7173121913179106],
    "exact_HT_mean_vcov": [6.2310226535339375, -0.37292319973722982, -0.37292319973722987, 0.28424526322867327],
    "exact_HT_mean_deff": [1.2654906375629016, 0.54152490586436419],
    "exact_HT_ratio": [2.4475718725718725, 0.0038321839431933848],
    "overton_HT_total": [318.18434343434342, 72.224747474747474],
    "overton_HT_total_vcov": [-49.332599989797799, 108.69554127129858, 108.69554127129864, 332.46830042852764],
    "overton_HT_mean": [16.376527164023916, 3.7173121913179106],
    "overton_HT_mean_vcov": [6.4076107716970583, -0.3112115130625302, -0.31121151306253036, 0.27183205694633455],
    "overton_HT_ratio": [2.4475718725718725, 0.0042747040131386523],
    "hr_HT_total": [318.18434343434342, 72.224747474747474],
    "hr_HT_total_vcov": [425.12068283848674, 155.45476405468844, 155.45476405468844, 337.32891210080601],
    "hr_HT_mean": [16.376527164023916, 3.7173121913179106],
    "hr_HT_mean_vcov": [6.4215312315962194, -0.35384181509927731, -0.35384181509927731, 0.27316253330659679],
    "hr_HT_ratio": [2.4475718725718725, 0.003950975460940062],
    "hrsample_HT_total": [318.18434343434342, 72.224747474747474],
    "hrsample_HT_total_vcov": [-12.40288618508232, 60.37378201203984, 60.373782012039797, 317.06846877869606],
    "hrsample_HT_mean": [16.376527164023916, 3.7173121913179106],
    "hrsample_HT_mean_vcov": [6.5017358536096514, -0.351260380527743, -0.35126038052774294, 0.27115891309465034],
    "hrsample_HT_ratio": [2.4475718725718725, 0.0040150286647619403],
    "exact_YG_total": [318.18434343434342, 72.224747474747474],
    "exact_YG_total_vcov": [62.070376336927666, 4.5741544900095334, 4.5741544900096187, 257.71523815797326],
    "exact_YG_mean": [16.376527164023916, 3.7173121913179106],
    "exact_YG_mean_vcov": [5.6755418203222368, -0.27682876272169321, -0.27682876272169327, 0.2675603363219472],
    "exact_YG_ratio": [2.4475718725718725, 0.0036728033335460147],
    "overton_YG_total": [318.18434343434342, 72.224747474747474],
    "overton_YG_total_vcov": [76.821312876236945, 7.6701101928373561, 7.6701101928374129, 262.54431308029797],
    "overton_YG_mean": [16.376527164023916, 3.7173121913179106],
    "overton_YG_mean_vcov": [6.1616619887658199, -0.1828813858151874, -0.18288138581518756, 0.29624595729071868],
    "overton_YG_ratio": [2.4475718725718725, 0.0045456398151619568],
    "hr_YG_total": [318.18434343434342, 72.224747474747474],
    "hr_YG_total_vcov": [66.534525303540306, 2.5620525966738796, 2.5620525966738796, 261.84238878685835],
    "hr_YG_mean": [16.376527164023916, 3.7173121913179106],
    "hr_YG_mean_vcov": [6.0627439221211707, -0.25239723018298721, -0.25239723018298721, 0.27266335114552026],
    "hr_YG_ratio": [2.4475718725718725, 0.0039369541599728044],
    "hrsample_YG_total": [318.18434343434342, 72.224747474747474],
    "hrsample_YG_total_vcov": [70.470041322313747, 3.1436077951229322, 3.1436077951228896, 273.41415416794194],
    "hrsample_YG_mean": [16.376527164023916, 3.7173121913179106],
    "hrsample_YG_mean_vcov": [6.3301317724179338, -0.25970169403076426, -0.25970169403076421, 0.28606188929107917],
    "hrsample_YG_ratio": [2.4475718725718725, 0.0041698249303144365],
    "brewer_total": [318.18434343434342, 72.224747474747474],
    "brewer_total_vcov": [70.470041322313961, 3.14360779512293, 3.14360779512293, 273.41415416794212],
    "brewer_mean": [16.376527164023916, 3.7173121913179106],
    "brewer_mean_vcov": [6.3301317724179356, -0.25970169403076437, -0.25970169403076437, 0.28606188929107923],
    "brewer_ratio": [2.4475718725718725, 0.0041698249303144383],
    "brewer_total_deff": [0.0379131736793758, 1.3798510257027632],
    "wr_total": [318.18434343434342, 72.224747474747474],
    "wr_total_vcov": [138.33869630649917, 20.313596826854351, 20.313596826854351, 416.53311524334242],
    "wr_mean": [16.376527164023916, 3.7173121913179106],
    "wr_mean_vcov": [9.8687796941472499, -0.1904578417958771, -0.1904578417958771, 0.50291022756390813],
    "wr_ratio": [2.4475718725718725, 0.0081857216749407821],
    "varHT": [697.01738100235298, 62.070376336927623],
    "exact_YG_domain_total": [234.29545454545453, 2537.8573270387951],
    "cal_brewer_linear_g": [
        1.3848271815299218, 0.78748349676705787, 1.2354912603392059, 0.63814757557634172, 1.5341631027206377,
        0.48881165438562585, 0.9368194179577739
    ],
    "cal_brewer_linear_total": [313.2724302914724, 82.639637098139247],
    "cal_brewer_linear_vcov": [41.735429003007297, 35.340649099583793, 35.340649099583793, 97.814110883969846],
    "cal_brewer_raking_g": [
        1.3665810242951748, 0.76083551012494621, 1.1804544481845027, 0.65721069318003777, 1.5820548593263091,
        0.56769944289174123, 0.88079923147037098
    ],
    "cal_brewer_raking_total": [313.86791303329477, 83.99558529894442],
    "cal_brewer_raking_vcov": [39.86787156969158, 34.049804523085605, 34.049804523085605, 98.643041699086154],
    "cal_yg_linear_g": [
        1.3848271815299218, 0.78748349676705787, 1.2354912603392059, 0.63814757557634172, 1.5341631027206377,
        0.48881165438562585, 0.9368194179577739
    ],
    "cal_yg_linear_total": [313.2724302914724, 82.639637098139247],
    "cal_yg_linear_vcov": [38.468681057437564, 33.619363093615284, 33.619363093615284, 93.824945580532045],
    "cal_yg_raking_g": [
        1.3665810242951748, 0.76083551012494621, 1.1804544481845027, 0.65721069318003777, 1.5820548593263091,
        0.56769944289174123, 0.88079923147037098
    ],
    "cal_yg_raking_total": [313.86791303329477, 83.99558529894442],
    "cal_yg_raking_vcov": [36.684431962037678, 32.331793734612305, 32.331793734612312, 94.629144609106703],
}

SIZE_B = np.array([4, 9, 2, 6, 11, 5, 3, 8, 7, 5], dtype=float)
CASES = {
    "exact_HT": dict(variance="ht"), "exact_YG": dict(variance="syg"),
    "overton_HT": dict(variance="ht", joint="overton"), "overton_YG": dict(variance="syg", joint="overton"),
    "hr_HT": dict(variance="ht", joint="hartley_rao"), "hr_YG": dict(variance="syg", joint="hartley_rao"),
    "hrsample_HT": dict(variance="ht", joint="hartley_rao"), "hrsample_YG": dict(variance="syg", joint="hartley_rao"),
    "brewer": dict(variance="brewer"), "wr": dict(variance="hansen_hurwitz"),
}


def fixture():
    pi = np.r_[np.array(R_PPS["piA"])[[1, 4, 6, 8]], np.array(R_PPS["piB"])[[0, 4, 7]]]
    frame = pd.DataFrame({"strat": ["A"] * 4 + ["B"] * 3, "pi": pi, "x": [5.0, 9, 6, 10, 4, 11, 8],
                          "y": [12.5, 20, 15, 27, 9, 30, 18], "y2": [3.0, 1, 4, 1, 5, 9, 2],
                          "dom": [1, 0, 1, 1, 0, 1, 1]})
    joint = np.outer(pi, pi)
    joint[:4, :4] = np.reshape(R_PPS["PA_sample"], (4, 4))
    joint[4:, 4:] = np.reshape(R_PPS["PB_sample"], (3, 3))
    return frame, joint


def options(name, joint):
    kw = dict(CASES[name])
    if name.startswith("exact"):
        kw["joint"] = joint
    if name.startswith("hr_"):
        kw["pi_squared_sum"] = {"A": R_PPS["psum"][0], "B": R_PPS["psum"][1]}
    return kw


@pytest.mark.parametrize("name", list(CASES))
def test_r_survey_pps_designs(name):
    frame, joint = fixture()
    kw = options(name, joint)
    total = oe.survey_pps_total(frame, ["y", "y2"], prob="pi", strata="strat", **kw)
    mean = oe.survey_pps_mean(frame, ["y", "y2"], prob="pi", strata="strat", deff=True, **kw)
    ratio = oe.survey_pps_ratio(frame, "y", "x", prob="pi", strata="strat", **kw)
    for result, key in [(total, "total"), (mean, "mean")]:
        reference = np.array(R_PPS[f"{name}_{key}_vcov"])
        np.testing.assert_allclose(result.estimate, R_PPS[f"{name}_{key}"], rtol=1e-13)
        np.testing.assert_allclose(np.ravel(result.attrs["covariance"]), reference, rtol=1e-9,
                                   atol=1e-12 * np.abs(reference).max())
    np.testing.assert_allclose([ratio.estimate.iloc[0], ratio.attrs["covariance"][0][0]], R_PPS[f"{name}_ratio"],
                               rtol=1e-11)
    if f"{name}_mean_deff" in R_PPS:
        np.testing.assert_allclose(mean.deff, R_PPS[f"{name}_mean_deff"], rtol=1e-11)
    assert total.attrs["df"] == 5 and total.attrs["variance"] == kw["variance"]
    json.dumps(total.attrs)


def test_sampling_varht_brewer_deff_domain_and_calibration():
    frame, joint = fixture()
    ht = oe.survey_pps_total(frame, "y", prob="pi", strata="strat", variance="ht", joint=joint)
    syg = oe.survey_pps_total(frame, "y", prob="pi", strata="strat", variance="syg", joint=joint)
    np.testing.assert_allclose([ht.attrs["covariance"][0][0], syg.attrs["covariance"][0][0]], R_PPS["varHT"],
                               rtol=1e-12)
    brewer = oe.survey_pps_total(frame, ["y", "y2"], prob="pi", strata="strat", deff=True)
    np.testing.assert_allclose(brewer.deff, R_PPS["brewer_total_deff"], rtol=1e-11)
    domain = oe.survey_pps_total(frame, "y", prob="pi", strata="strat", variance="syg", joint=joint, domain="dom")
    np.testing.assert_allclose([domain.estimate.iloc[0], domain.attrs["covariance"][0][0]],
                               R_PPS["exact_YG_domain_total"], rtol=1e-12)
    for name, kw in [("brewer", {}), ("yg", dict(variance="syg", joint=joint))]:
        for method in ("linear", "raking"):
            calibration = dict(method=method, population=22, totals={"x": 130.0}, tol=1e-13)
            result = oe.survey_pps_total(frame, ["y", "y2"], prob="pi", strata="strat", calibration=calibration, **kw)
            np.testing.assert_allclose(result.estimate, R_PPS[f"cal_{name}_{method}_total"], rtol=1e-12)
            np.testing.assert_allclose(np.ravel(result.attrs["covariance"]), R_PPS[f"cal_{name}_{method}_vcov"],
                                       rtol=1e-10)
            weights = oe.svycal(frame.assign(d=1 / frame.pi), "d", **calibration)
            np.testing.assert_allclose(weights["weights"].g_weight, R_PPS[f"cal_{name}_{method}_g"], rtol=1e-12)
            assert result.attrs["calibration"]["method"] == method


def sampford(pi):
    """Every fixed-size sample with p(s) proportional to (n - sum_s pi) prod_s pi/(1 - pi) (Sampford 1967)."""
    n = int(round(pi.sum()))
    samples = list(combinations(range(len(pi)), n))
    p = np.array([(n - pi[list(s)].sum()) * np.prod(pi[list(s)] / (1 - pi[list(s)])) for s in samples])
    return samples, p / p.sum()


def test_sampford_enumeration_unbiasedness_of_ht_and_syg():
    pi = 3 * SIZE_B / SIZE_B.sum()
    samples, p = sampford(pi)
    joint = np.zeros((10, 10))
    for s, ps in zip(samples, p):
        joint[np.ix_(s, s)] += ps
    np.testing.assert_allclose(np.diag(joint), pi, rtol=1e-12)              # Sampford attains pi exactly
    np.testing.assert_allclose(joint[np.ix_([0, 4, 7], [0, 4, 7])].ravel(), R_PPS["PB_sample"], rtol=1e-12)
    y = 2.0 * SIZE_B + np.array([3, -1, 4, 0, -2, 5, 1, -3, 2, 6])
    estimates, ht, syg = [], [], []
    for s in samples:
        rows = list(s)
        frame = pd.DataFrame({"y": y[rows], "pi": pi[rows]})
        block = joint[np.ix_(rows, rows)]
        a = oe.survey_pps_total(frame, "y", prob="pi", variance="ht", joint=block)
        b = oe.survey_pps_total(frame, "y", prob="pi", variance="syg", joint=block)
        estimates.append(a.estimate.iloc[0])
        ht.append(a.attrs["covariance"][0][0])          # may be negative for single samples
        syg.append(b.attrs["covariance"][0][0])
    estimates = np.array(estimates)
    assert p @ estimates == pytest.approx(y.sum(), rel=1e-13)
    true = p @ (estimates - y.sum()) ** 2
    assert p @ np.array(ht) == pytest.approx(true, rel=1e-11)
    assert p @ np.array(syg) == pytest.approx(true, rel=1e-11)
    negative = int(np.argmin(ht))
    assert ht[negative] < 0
    rows = list(samples[negative])
    table = oe.survey_pps_total(pd.DataFrame({"y": y[rows], "pi": pi[rows]}), "y", prob="pi", variance="ht",
                                joint=joint[np.ix_(rows, rows)])
    assert np.isnan(table.std_error.iloc[0]) and "negative" in table.attrs["notes"][0]


def test_approximations_are_unbiased_under_srswor_enumeration():
    population, n = np.array([3.0, 7, 1, 9, 4, 6, 12]), 3
    big, f = len(population), n / len(population)
    true = big ** 2 * (1 - f) * population.var(ddof=1) / n
    pairs = n * (n - 1) / (big * (big - 1))
    expected = {"exact": true, "brewer": true, "hajek": true, "overton_ht": true, "overton_syg": true,
                "hr_ht": true, "hr_syg": true, "hrpop_syg": true, "hansen_hurwitz": true / (1 - f)}
    found = {name: 0.0 for name in expected}
    samples = list(combinations(range(big), n))
    for s in samples:
        frame = pd.DataFrame({"y": population[list(s)], "pi": f})
        block = np.full((n, n), pairs) + np.diag(np.full(n, f - pairs))
        runs = {"exact": dict(variance="syg", joint=block), "brewer": {}, "hajek": dict(variance="hajek"),
                "overton_ht": dict(variance="ht", joint="overton"), "overton_syg": dict(variance="syg", joint="overton"),
                "hr_ht": dict(variance="ht", joint="hartley_rao"), "hr_syg": dict(variance="syg", joint="hartley_rao"),
                "hrpop_syg": dict(variance="syg", joint="hartley_rao", pi_squared_sum=big * f ** 2),
                "hansen_hurwitz": dict(variance="hansen_hurwitz")}
        for name, kw in runs.items():
            found[name] += oe.survey_pps_total(frame, "y", prob="pi", **kw).attrs["covariance"][0][0] / len(samples)
    for name in expected:
        assert found[name] == pytest.approx(expected[name], rel=1e-12), name


def delta_matrix(pi, strata, kind, c=None):
    """Explicit n-by-n Delta-check of the Overton or Hartley–Rao pi_kl approximation."""
    n = np.array([np.sum(strata == h) for h in strata], dtype=float)
    a, b = pi[:, None], pi[None, :]
    if kind == "overton":
        pkl = 2 * (n[:, None] - 1) * a * b / (2 * n[:, None] - a - b)
    else:
        pkl = (n[:, None] - 1) * a * b / (n[:, None] - a - b + c[:, None])
    delta = (pkl - a * b) / pkl
    delta[strata[:, None] != strata[None, :]] = 0.0
    np.fill_diagonal(delta, 1 - pi)
    return delta


def test_closed_forms_equal_explicit_delta_matrices_and_first_order_formulas():
    rng = np.random.default_rng(11)
    strata = np.repeat(np.array(["a", "b", "c"]), [5, 6, 4])
    pi = rng.uniform(0.1, 0.8, len(strata))
    values = rng.normal(size=(len(strata), 2)) * [3.0, 1.0] + [10.0, 2.0]
    frame = pd.DataFrame({"h": strata, "pi": pi, "y": values[:, 0], "v": values[:, 1]})
    z = values / pi[:, None]
    population_sq = {h: 1.5 * np.sum(pi[strata == h] ** 2) for h in "abc"}  # within (sum_s pi^2, n_h)
    counts = np.array([np.sum(strata == h) for h in strata], dtype=float)
    c = np.array([population_sq[h] for h in strata]) / counts
    for kind, extra in [("overton", {}), ("hartley_rao", {"pi_squared_sum": population_sq})]:
        delta = delta_matrix(pi, strata, kind, c)
        ht = z.T @ delta @ z
        yg = ht - (z * delta.sum(1)[:, None]).T @ z
        for form, expected in [("ht", ht), ("syg", yg)]:
            result = oe.survey_pps_total(frame, ["y", "v"], prob="pi", strata="h", variance=form, joint=kind, **extra)
            np.testing.assert_allclose(result.attrs["covariance"], expected, rtol=1e-12, atol=1e-12)
    for name in ("brewer", "hajek", "hansen_hurwitz"):
        expected = np.zeros((2, 2))
        for h in "abc":
            rows = strata == h
            n, zh, ph = rows.sum(), z[rows], pi[rows]
            a = np.ones(n) if name == "hansen_hurwitz" else 1 - ph
            centre = (a @ zh) / a.sum() if name == "hajek" else zh.mean(0)
            expected += n / (n - 1) * ((zh - centre) * a[:, None]).T @ (zh - centre)
        result = oe.survey_pps_total(frame, ["y", "v"], prob="pi", strata="h", variance=name)
        np.testing.assert_allclose(result.attrs["covariance"], expected, rtol=1e-12)
    # Hartley-Rao SYG with the sample estimate of sum_U pi^2 is algebraically Brewer's estimator.
    brewer = oe.survey_pps_total(frame, ["y", "v"], prob="pi", strata="h")
    hartley_rao = oe.survey_pps_total(frame, ["y", "v"], prob="pi", strata="h", variance="syg", joint="hartley_rao")
    np.testing.assert_allclose(hartley_rao.attrs["covariance"], brewer.attrs["covariance"], rtol=1e-12)


def test_psu_aggregation_joint_formats_and_certainty_units():
    frame, joint = fixture()
    split = frame.loc[frame.index.repeat(2)].reset_index(drop=True)
    split["unit"] = np.repeat(np.arange(7), 2)
    split[["y", "y2"]] = split[["y", "y2"]] * np.tile([0.25, 0.75], 7)[:, None]
    for kw in (dict(variance="brewer"), dict(variance="hajek"), dict(variance="syg", joint=joint),
               dict(variance="ht", joint="hartley_rao")):
        a = oe.survey_pps_total(frame, ["y", "y2"], prob="pi", strata="strat", **kw)
        b = oe.survey_pps_total(split, ["y", "y2"], prob="pi", strata="strat", psu="unit", **kw)
        np.testing.assert_allclose(b.estimate, a.estimate, rtol=1e-14)
        np.testing.assert_allclose(b.attrs["covariance"], a.attrs["covariance"], rtol=1e-12)
        assert b.attrs["design"]["n_psu"] == 7 and b.attrs["n_design"] == 14
    blocks = {"A": pd.DataFrame(joint[:4, :4]), "B": joint[4:, 4:].tolist()}
    a = oe.survey_pps_total(frame, "y", prob="pi", strata="strat", variance="syg", joint=blocks)
    b = oe.survey_pps_total(frame, "y", prob="pi", strata="strat", variance="syg", joint=pd.DataFrame(joint))
    np.testing.assert_allclose(a.attrs["covariance"], b.attrs["covariance"], rtol=1e-14)
    certain = pd.concat([frame, pd.DataFrame({"strat": ["C"], "pi": [1.0], "x": [3.0], "y": [40.0], "y2": [1.0],
                                              "dom": [1]})], ignore_index=True)
    with_certain = oe.survey_pps_total(certain, "y", prob="pi", strata="strat")
    plain = oe.survey_pps_total(frame, "y", prob="pi", strata="strat")
    assert with_certain.estimate.iloc[0] == pytest.approx(plain.estimate.iloc[0] + 40.0, rel=1e-14)
    np.testing.assert_allclose(with_certain.attrs["covariance"], plain.attrs["covariance"], rtol=1e-13)
    assert with_certain.attrs["df"] == 5 and with_certain.attrs["design"]["n_certainty_psus"] == 1


def test_failure_contract():
    frame, joint = fixture()
    bad_diag = joint.copy()
    bad_diag[0, 0] = 0.5
    asymmetric = joint.copy()
    asymmetric[0, 1] += 1e-3
    cross = joint.copy()
    cross[0, 5] = cross[5, 0] = 0.0
    too_big = joint.copy()
    too_big[0, 1] = too_big[1, 0] = 0.9
    cases = [
        (dict(variance="ht"), "invalid_pps_option"),
        (dict(variance="brewer", joint=joint), "invalid_pps_option"),
        (dict(variance="syg", joint="sampford"), "invalid_pps_option"),
        (dict(variance="ys"), "invalid_pps_option"),
        (dict(pi_squared_sum=1.0), "invalid_pps_option"),
        (dict(variance="syg", joint="hartley_rao", pi_squared_sum=1.0), "invalid_pps_option"),
        (dict(variance="syg", joint="hartley_rao", pi_squared_sum={"A": 0.1, "B": 1.0}), "invalid_pps_option"),
        (dict(variance="syg", joint="hartley_rao", pi_squared_sum={"A": 9.0, "B": 1.0}), "invalid_pps_option"),
        (dict(variance="ht", joint=bad_diag), "pps_joint_probability"),
        (dict(variance="ht", joint=asymmetric), "pps_joint_probability"),
        (dict(variance="ht", joint=cross), "pps_joint_probability"),
        (dict(variance="ht", joint=too_big), "pps_joint_probability"),
        (dict(variance="ht", joint=joint[:6, :6]), "pps_joint_probability"),
        (dict(variance="ht", joint={"A": joint[:4, :4]}), "pps_joint_probability"),
        (dict(variance="ht", joint=[["a"] * 7] * 7), "pps_joint_probability"),
        (dict(prob="x"), "invalid_pps_probability"),
        (dict(prob="absent"), "missing_columns"),
        (dict(strata="y"), "singleton_pps_stratum"),
        (dict(psu="dom"), "invalid_pps_probability"),
        (dict(alpha=0.0), "invalid_survey_option"),
    ]
    for change, code in cases:
        kw = {"prob": "pi", "strata": "strat", **change}
        with pytest.raises(AnalysisError) as error:
            oe.survey_pps_total(frame, "y", **kw)
        assert error.value.code == code, (change, error.value.code)
    wide = pd.DataFrame({"y": np.arange(2001.0), "pi": 0.5})
    with pytest.raises(AnalysisError) as error:
        oe.survey_pps_total(wide, "y", prob="pi", variance="ht", joint=[[1.0]])
    assert error.value.code == "survey_budget"
    with pytest.raises(AnalysisError) as error:
        oe.survey_pps_mean(frame.assign(y=np.nan), "y", prob="pi", strata="strat")
    assert error.value.code == "survey_missing"
