"""Independent review of survey_calib: new R survey cross-checks, solver robustness and failure contract.

Configurations not in the family's own tests: unequal PSU counts and sizes with a census
stratum and a singleton certainty stratum, domains combined with missing-value exclusion
for calibrated totals/means/ratios, logit/truncated bounds 1e-4 from infeasibility, by-group
calibration across PSUs, and a clustered PPS design with a certainty stratum under every
variance option, with and without calibration.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

import openecon as oe
from openecon.analysis_contracts import AnalysisError

# R 4.6.1, survey 4.5, sampling 2.11: tests/reference/survey_calib_review.R. Covariances are
# row-major after the estimates; "_ratio" holds (ratio, variance[, deff]).
R_REVIEW = {
    "raking_g": [
        0.96653842994607808, 1.1102150745104991, 1.0214476200038414, 1.0338432970766589, 0.96065890359943551,
        1.123289166335478, 1.0520464515545114, 1.0460180211539543, 0.97197179513415022, 1.1365172210105139,
        1.0644355392435174, 1.0396550144447632, 0.98341790930844775, 1.1705631165797901, 1.0769705230510256,
        1.0518981783480668, 0.99499881497602394, 0.97060746911272511, 1.0704192299325621, 1.0642855199452022,
        1.0248053371474233, 0.98203751673911421, 1.0830246787547479, 1.0768187370985485, 1.0368736285877114,
        0.97606370827566635, 1.0957785716029429, 1.1090762846240494, 1.0490840383923445, 0.9875580095529678,
        1.1086826565806196, 1.1221369658550933, 1.0427023808684794, 0.99918766978351492, 0.93581588244288483,
        1.1353514520106349, 1.0549814311053076, 1.0109542829786171, 0.94683621810507756
    ],
    "raking_drop_total": [
        8481.2692491713005, 13976.102045864336, 85834.532285783818, 147902.95421095254, 147902.95421095254,
        333544.1503136867
    ],
    "raking_drop_mean": [
        21.928372851576022, 36.135296223891615, 0.097493821534168384, -0.062000835366283188, -0.062000835366283188,
        0.060193990382135708
    ],
    "raking_domain_total": [5300.0022852674319, 254244.273818596, 2.8254105958258529],
    "raking_domain_mean": [23.084529150895587, 0.65165584933970788, 0.38173216830655315],
    "raking_domain_ratio": [3.2618184001228259, 0.01144405541540738],
    "raking_ratio": [2.2155809426019815, 0.00016507592147005216, 0.010026088163684568],
    "raking_df": [9],
    "edge_logit_g": [
        0.9000000000230326, 1.0880999999999998, 1.0131909048023766, 1.0860333952189429, 0.90000000000319536,
        1.0880999999999998, 1.0880987122083519, 1.0880819575472327, 0.90000000037002792, 1.0880999999999996,
        1.0880999888793423, 1.0879700237144896, 0.90000004285014634, 1.0881000000000001, 1.0880999999039689,
        1.088098876834769, 0.90000496202588964, 0.90611014953894853, 1.0880999993077873, 1.0880999903009736,
        1.0351145003006874, 1.0496177559966866, 1.0880999999940224, 1.0880999999162453, 1.0874651697866442,
        0.96590802993070635, 1.0880999999999486, 1.0880999999999992, 1.0880944995905775, 1.0851360071416145,
        1.0880999999999996, 1.0881000000000003, 1.0880603591218068, 1.0880739985874979, 0.90000000000000013,
        1.0881000000000001, 1.0880996576140505, 1.0880997754369059, 0.90000000000000735
    ],
    "edge_logit_total": [15293.345227674541, 6900, 7111.7258096542464, 0, 0, 0],
    "edge_logit_mean": [35.984341712175393, 0.03937287645483338],
    "edge_truncated_g": [
        0.90000000000000002, 1.0881000000000001, 1.0131888888888889, 1.0881000000000001, 0.90000000000000002,
        1.0881000000000001, 1.0881000000000001, 1.0881000000000001, 0.90000000000000013, 1.0880999999999998,
        1.0881000000000001, 1.0881000000000001, 0.90000000000000002, 1.0881000000000001, 1.0881000000000001,
        1.0881000000000001, 0.90000000000000002, 0.91037750000001061, 1.0881000000000001, 1.0880999999999998,
        1.0324357142857141, 1.0426319444444276, 1.0881000000000001, 1.0880999999999998, 1.0881000000000001,
        0.96372258064516136, 1.0881000000000001, 1.0881000000000001, 1.0881000000000001, 1.0881000000000001,
        1.0881000000000001, 1.0881000000000003, 1.0881000000000001, 1.0881000000000001, 0.90000000000000002,
        1.0881000000000001, 1.0881000000000001, 1.0881000000000001, 0.89999999999999991
    ],
    "edge_truncated_total": [15293.502149999998, 6899.9999999999991, 7110.8393591303493, 0, 0, 0],
    "edge_truncated_mean": [35.984710941176473, 0.039367968770271926],
    "bysex_g": [
        0.83317956803652193, 1.0228457253937111, 1.0393010447810116, 0.99760101983962146, 1.07659976872556,
        1.0184576402237642, 1.13342743654653, 1.0261960549358131, 1.072211683555613, 1.0140695550538172,
        1.1620224716427212, 1.0469801938284189, 1.0678235983856663, 1.2978488883496293, 1.1906175067389124,
        1.0425921086584717, 1.0634355132157194, 0.8403283268105699, 1.0217487041012243, 1.038204023488525,
        0.97615474351747822, 0.86892336190676112, 1.0173606189312774, 1.0338159383185781, 1.0047497786136694,
        1.0711146622631265, 1.0129725337613307, 1.169171230416769, 1.0333448137098606, 1.0667265770931793,
        1.0085844485913837, 1.1977662655129604, 1.0458831725359319, 1.0623384919232326, 0.84747708558461765,
        1.2263613006091514, 1.0414950873659852, 1.0579504067532859, 0.87607212068080886
    ],
    "bysex_total": [15284.588224080499, 7981.4871155243436],
    "bysex_ratio": [2.2151577136348535, 0.00016764308161151765],
    "byregion_g": [
        1.0362694300518136, 1.01437908496732, 1.0823529411764707, 1.0362694300518136, 1.0492462311557791,
        1.0143790849673202, 1.0166666666666664, 1.0362694300518136, 1.0492462311557793, 1.0143790849673202,
        1.0166666666666664, 1.0492462311557791, 1.0492462311557791, 1.1076923076923075, 1.0166666666666664,
        1.0492462311557791, 1.0492462311557791, 1.1076923076923075, 1.0823529411764707, 1.0492462311557791,
        1.0362694300518136, 1.1076923076923075, 1.0823529411764707, 1.0492462311557791, 1.0362694300518136,
        1.0143790849673204, 1.082352941176471, 1.0362694300518136, 1.0362694300518138, 1.0143790849673202,
        1.082352941176471, 1.0362694300518138, 1.0492462311557791, 1.0143790849673202, 1.0166666666666664,
        1.0362694300518136, 1.0492462311557791, 1.0143790849673202, 1.0166666666666664
    ],
    "byregion_total": [15339.305632103156, 7404.717945698626],
    "byregion_ratio": [2.258675742717418, 0.0061607669863239468],
    "PA": [
        0.3888888888888889, 0.15577243926507564, 0.17809630534195883, 0.15577243926507564, 0.5, 0.23836119371018047,
        0.17809630534195883, 0.23836119371018047, 0.55555555555555558
    ],
    "PB": [
        0.27272727272727271, 0.11380895939265524, 0.031652836320280098, 0.068526522827097827, 0.11380895939265524,
        0.54545454545454541, 0.073909051008456114, 0.15633660217021059, 0.031652836320280098, 0.073909051008456114,
        0.18181818181818182, 0.044161600580325082, 0.068526522827097827, 0.15633660217021059, 0.044161600580325082,
        0.36363636363636365
    ],
    "psum": [1.1851851851851851, 2.3471074380165291],
    "pps_brewer_total": [
        391.84285714285716, 312.53571428571433, 1111.7422278911563, 794.81025604686317, 794.81025604686317,
        900.38745984504897
    ],
    "pps_brewer_mean": [11.918742757821553, 0.083610059415303015],
    "pps_brewer_ratio": [1.2537538567020909, 0.0054675992174280036],
    "pps_brewer_raking_g": [
        1.0680812985863777, 1.0367341495100593, 1.0429291003960417, 1.0522910038090927, 1.0214072845531428,
        1.033650486904883, 1.0680812985863775, 1.0712676777362597, 1.039827011525658, 1.0649043970043321,
        1.0712676777362595, 1.0398270115256583, 0.97386756141360309, 0.98554092403642679
    ],
    "pps_brewer_raking_total": [
        414.41817510898716, 325, 65.762567571862562, -1.6309520146045619e-14, -1.6309520146045619e-14,
        8.1101887405785928e-29
    ],
    "pps_brewer_raking_mean": [12.012121017651795, 0.055251054460712157],
    "pps_brewer_raking_ratio": [1.2751328464891913, 0.00062260419002946907],
    "pps_brewer_raking_drop_total": [
        999.62450887428929, 397.67303722112808, 1447.7040053259052, 298.74180583031296, 298.74180583031296,
        130.34532870309366
    ],
    "pps_brewer_raking_domain_total": [698.15200893577162, 18717.528446558041],
    "pps_brewer_raking_domain_mean": [30.182491810314676, 0.97695357576359643],
    "pps_exact_HT_total": [
        391.84285714285716, 312.53571428571433, -7683.2506707827115, -3685.8138832030741, -3685.8138832030736,
        -1296.6505599510092
    ],
    "pps_exact_HT_mean": [11.918742757821553, -0.13491290572219825],
    "pps_exact_HT_ratio": [1.2537538567020909, -0.0049063708426076618],
    "pps_exact_HT_domain_total": [667.99285714285725, 2512.5748440376242],
    "pps_exact_HT_linear_total": [
        414.41920157426728, 325, 75.910928008236397, -1.9975516662262669e-15, -1.9975516662262665e-15,
        1.7643262554303896e-29
    ],
    "pps_exact_HT_linear_mean": [12.012150770268613, 0.063777297213389025],
    "pps_exact_HT_linear_ratio": [1.2751360048438993, 0.00071868334208981049],
    "pps_exact_YG_total": [
        391.84285714285716, 312.53571428571428, 1439.0337910931448, 954.9500759667967, 954.95007596679716,
        1000.7081612628606
    ],
    "pps_exact_YG_mean": [11.918742757821555, 0.097924052389106758],
    "pps_exact_YG_ratio": [1.2537538567020912, 0.0063217651894707126],
    "pps_exact_YG_domain_total": [667.99285714285725, 25510.760391844688],
    "pps_overton_HT_total": [
        391.84285714285716, 312.53571428571433, 1589.7670219198774, 929.07897770219142, 929.0789777021912,
        864.84717498110331
    ],
    "pps_overton_HT_mean": [11.918742757821553, 0.098876166061142759],
    "pps_overton_HT_ratio": [1.2537538567020909, 0.0063427062924769819],
    "pps_hr_HT_total": [
        391.84285714285716, 312.53571428571433, -2209.7459410430852, -870.78815192743866, -870.78815192743832,
        122.23143424036235
    ],
    "pps_hr_HT_mean": [11.918742757821553, 0.0028108159051062264],
    "pps_hr_HT_ratio": [1.2537538567020909, 0.0016984022053394027],
    "pps_hrsample_HT_total": [
        391.84285714285716, 312.53571428571433, 1355.855910808772, 854.70328325775006, 854.70328325775017,
        857.21523053666078
    ],
    "pps_hrsample_HT_mean": [11.918742757821553, 0.086879538123223535],
    "pps_hrsample_HT_ratio": [1.2537538567020909, 0.0057344856176530668],
    "pps_hrsample_HT_raking_total": [
        414.41817510898716, 325, 67.558286435018559, -1.6703514713864113e-14, -1.6703514713864113e-14,
        8.0224539607358752e-29
    ],
    "pps_hrsample_HT_raking_mean": [12.012121017651795, 0.056759744956957493],
    "pps_hrsample_HT_raking_ratio": [1.2751328464891913, 0.00063960507867473319],
    "pps_overton_YG_total": [
        391.84285714285716, 312.53571428571433, 1129.5999697656823, 809.52415910808713, 809.52415910808691,
        925.24180366591042
    ],
    "pps_overton_YG_mean": [11.918742757821553, 0.089899585616466049],
    "pps_overton_YG_ratio": [1.2537538567020909, 0.0056726700683435334],
    "pps_hr_YG_total": [
        391.84285714285716, 312.53571428571433, 1230.9572234819848, 819.04276266061913, 819.04276266061947,
        879.43268140589555
    ],
    "pps_hr_YG_mean": [11.918742757821553, 0.089702581060504938],
    "pps_hr_YG_ratio": [1.2537538567020909, 0.0057287927213105138],
    "pps_hrsample_YG_total": [
        391.84285714285716, 312.53571428571433, 1111.7422278911556, 794.81025604686238, 794.81025604686249,
        900.38745984504908
    ],
    "pps_hrsample_YG_mean": [11.918742757821553, 0.083610059415302945],
    "pps_hrsample_YG_ratio": [1.2537538567020909, 0.0054675992174280019],
    "pps_hrsample_YG_raking_total": [
        414.41817510898716, 325, 65.762567571862547, -1.6309520146045612e-14, -1.6309520146045612e-14,
        8.1101887405785894e-29
    ],
    "pps_hrsample_YG_raking_mean": [12.012121017651795, 0.055251054460712157],
    "pps_hrsample_YG_raking_ratio": [1.2751328464891913, 0.00062260419002946907],
    "pps_exact_HT_linear_domain_total_R": [698.1421547907845, 12268.016527265809],
    "pps_hajek_y2_total_var": [1099.5731090985928],
}

POP = dict(population=425, categorical={"sex": {"F": 197, "M": 228},
                                        "region": {"east": 113.6, "north": 204.4, "south": 107}},
           totals={"x": 6900.0}, tol=1e-13)


def design_fixture():
    """Strata with 3, 5, 2 (census), 1 (certainty) and 3 PSUs of 1..5 rows; R's `dat`."""
    sizes = {1: [2, 4, 3], 2: [1, 3, 2, 5, 2], 3: [4, 3], 4: [3], 5: [2, 2, 3]}
    rows = [(h, p + 1) for h, s in sizes.items() for p, m in enumerate(s) for _ in range(m)]
    i = np.arange(len(rows))
    f = pd.DataFrame(rows, columns=["h", "psu"])
    f["Nh"] = f.h.map({1: 7, 2: 15, 3: 2, 4: 1, 5: 9})
    f["d"] = 5 + (i * 7) % 11 + 0.5 * (i % 3)
    f["sex"] = np.where((i * 5) % 7 < 3, "F", "M")
    f["region"] = np.array(["north", "south", "east"])[(i * 2 + i // 4) % 3]
    f["x"] = 8 + (i * 13) % 17 + 0.25 * (i % 4)
    f["z"] = 4.0 + (i * 5) % 7
    f["y"] = 2 + 1.1 * f.x + (i * 11) % 9 - 3 * (f.sex == "F")
    f["y2"] = 40 - 0.5 * f.x + ((i * 3) % 5) * 2
    f["dom"] = ((i * 7) % 5 < 3).astype(int)
    f.loc[[3, 10, 22, 30], "y"] = np.nan
    f.loc[17, "z"] = np.nan
    return f, oe.survey_design(f, weights="d", psu="psu", strata="h", fpc="Nh", singleton="certainty")


def flat(result):
    return np.r_[result.estimate.to_numpy(), np.ravel(result.attrs["covariance"])]


def test_raking_with_unequal_psus_certainty_domain_and_missing_values():
    frame, design = design_fixture()
    cal = oe.svycal(frame, design, method="raking", **POP)
    np.testing.assert_allclose(cal["weights"].g_weight, R_REVIEW["raking_g"], rtol=1e-12)
    for kind, function in [("total", oe.survey_calibrated_total), ("mean", oe.survey_calibrated_mean)]:
        dropped = function(frame, design, ["y", "y2"], calibration=cal, missing="drop")
        np.testing.assert_allclose(flat(dropped), R_REVIEW[f"raking_drop_{kind}"], rtol=1e-10)
        assert dropped.attrs["n_missing_excluded"] == 4 and dropped.attrs["df"] == R_REVIEW["raking_df"][0]
        domain = function(frame, design, "y", calibration=cal, domain="dom", missing="drop", deff=True)
        np.testing.assert_allclose(np.r_[flat(domain), domain.deff], R_REVIEW[f"raking_domain_{kind}"], rtol=1e-10)
    ratio = oe.survey_calibrated_ratio(frame, design, "y", "z", calibration=cal, domain="dom", missing="drop")
    np.testing.assert_allclose(flat(ratio), R_REVIEW["raking_domain_ratio"], rtol=1e-10)
    assert ratio.attrs["n_used"] == int(((frame.dom == 1) & frame.y.notna() & frame.z.notna()).sum())
    full = oe.survey_calibrated_ratio(frame, design, "y2", "x", calibration=cal, deff=True)
    np.testing.assert_allclose(np.r_[flat(full), full.deff], R_REVIEW["raking_ratio"], rtol=1e-10)
    # The census and certainty strata contribute nothing: their PSU totals can change freely.
    shifted = frame.assign(y2=frame.y2 + np.where(frame.h.isin([3, 4]), 7.0 * frame.psu, 0.0))
    moved = oe.survey_calibrated_total(shifted, design, "y2", calibration=cal)
    base = oe.survey_calibrated_total(frame, design, "y2", calibration=cal)
    assert moved.estimate.iloc[0] != pytest.approx(base.estimate.iloc[0])
    # Only the calibration residualization couples them to other strata, through B.
    assert np.isfinite(moved.std_error.iloc[0])


def lp_feasible(frame, low, high):
    d = frame.d.to_numpy()
    x = np.column_stack([np.ones(len(frame)), frame.sex == "M", frame.region == "north", frame.region == "south",
                         frame.x]).astype(float)
    lp = optimize.linprog(np.zeros(len(frame)), A_eq=(x * d[:, None]).T, b_eq=[425, 228, 204.4, 107, 6900.0],
                          bounds=[(low, high)] * len(frame), method="highs")
    return lp.status == 0


@pytest.mark.parametrize("method", ["logit", "truncated"])
def test_bounded_calibration_at_the_feasibility_edge(method):
    """Upper bound 1.0881, 1e-4 above the smallest feasible one (1.08799 for lower bound 0.9)."""
    frame, design = design_fixture()
    assert lp_feasible(frame, 0.9, 1.0881) and not lp_feasible(frame, 0.9, 1.0879)
    cal = oe.svycal(frame, design, method=method, bounds=(0.9, 1.0881), **POP)
    g = cal["weights"].g_weight.to_numpy()
    np.testing.assert_allclose(g, R_REVIEW[f"edge_{method}_g"], rtol=1e-9)
    assert 0.9 - 1e-15 <= g.min() and g.max() <= 1.0881 + 1e-15
    total = oe.survey_calibrated_total(frame, design, ["y2", "x"], calibration=cal)
    reference = np.array(R_REVIEW[f"edge_{method}_total"])
    np.testing.assert_allclose(total.estimate, reference[:2], rtol=1e-11)
    np.testing.assert_allclose(total.attrs["covariance"][0][0], reference[2], rtol=1e-7)
    assert np.abs(total.attrs["covariance"][1][1]) < 1e-12 * 6900.0 ** 2
    mean = oe.survey_calibrated_mean(frame, design, "y2", calibration=cal)
    np.testing.assert_allclose(flat(mean), R_REVIEW[f"edge_{method}_mean"], rtol=1e-7)
    # Just beyond the edge no weights exist; the per-constraint screen passes, Newton must not.
    with pytest.raises(AnalysisError) as error:
        oe.svycal(frame, design, method=method, bounds=(0.9, 1.0879), **POP)
    assert error.value.code == "calibration_not_converged"


def test_truncated_newton_crosses_kinks_on_the_dual_objective():
    """Regression: the misfit-only line search stalled at a truncation kink (feasible bounds)."""
    frame, design = design_fixture()
    for upper in (1.0881, 1.08805, 1.0885, 1.0905):
        g = oe.svycal(frame, design, method="truncated", bounds=(0.9, upper), **POP)["weights"].g_weight.to_numpy()
        x = np.column_stack([np.ones(len(frame)), frame.sex == "M", frame.region == "north",
                             frame.region == "south", frame.x]).astype(float)
        np.testing.assert_allclose(x.T @ (frame.d.to_numpy() * g), [425, 228, 204.4, 107, 6900.0], rtol=1e-12)
        assert g.min() >= 0.9 and g.max() <= upper


def test_by_group_calibration_across_psus():
    frame, design = design_fixture()
    bysex = oe.svycal(frame, design, by="sex", population={"F": 197, "M": 228},
                      totals={"F": {"x": 3150.0}, "M": {"x": 3750.0}}, tol=1e-13)
    np.testing.assert_allclose(bysex["weights"].g_weight, R_REVIEW["bysex_g"], rtol=1e-12)
    np.testing.assert_allclose(flat(oe.survey_calibrated_total(frame, design, "y2", calibration=bysex)),
                               R_REVIEW["bysex_total"], rtol=1e-10)
    np.testing.assert_allclose(flat(oe.survey_calibrated_ratio(frame, design, "y2", "x", calibration=bysex)),
                               R_REVIEW["bysex_ratio"], rtol=1e-10)
    byregion = dict(method="logit", bounds=(0.5, 2.0), by="region", tol=1e-13,
                    population={"east": 113.6, "north": 204.4, "south": 107},
                    categorical={"east": {"sex": {"F": 36.0, "M": 77.6}}, "north": {"sex": {"F": 100.0, "M": 104.4}},
                                 "south": {"sex": {"F": 61.0, "M": 46.0}}})
    g = oe.svycal(frame, design, **byregion)["weights"].g_weight
    np.testing.assert_allclose(g, R_REVIEW["byregion_g"], rtol=1e-12)
    np.testing.assert_allclose(flat(oe.survey_calibrated_total(frame, design, "y2", calibration=byregion)),
                               R_REVIEW["byregion_total"], rtol=1e-10)
    np.testing.assert_allclose(flat(oe.survey_calibrated_ratio(frame, design, "y2", "x", calibration=byregion)),
                               R_REVIEW["byregion_ratio"], rtol=1e-10)


PI_A = np.array([4, 7, 2, 9, 5, 3, 8, 6, 10]) * 3 / 54
PI_B = np.array([3, 6, 2, 8, 5, 4, 7, 9]) * 4 / 44
UNITS = ["a2", "a4", "a9", "b1", "b2", "b3", "b6", "c1"]
CALIBRATION = dict(method="raking", population=34.5, totals={"x": 325.0}, tol=1e-13)


def pps_fixture():
    """Clustered PPS sample: 3 of 9 and 4 of 8 PSUs (Sampford) plus a certainty PSU; R's `s`."""
    upi = dict(zip(UNITS, np.r_[PI_A[[1, 3, 8]], PI_B[[0, 1, 2, 5]], 1.0]))
    size = dict(zip(UNITS, [7, 9, 10, 3, 6, 2, 4, 30]))
    unit = ["a2", "a2", "a4", "a9", "a9", "a9", "b1", "b2", "b2", "b3", "b6", "b6", "c1", "c1"]
    j = np.arange(14)
    s = pd.DataFrame({"strat": [u[0] for u in unit], "unit": unit, "pi": [upi[u] for u in unit]})
    s["x"] = 3 + (j * 5) % 7 + np.array([size[u] for u in unit]) / 2
    s["y"] = 10 + 2 * s.x + (j * 3) % 5
    s["y2"] = 20 - s.x + (j % 4)
    s["dom"] = ((j * 3) % 4 < 3).astype(int)
    s.loc[5, "y"] = np.nan
    joint = np.outer(list(upi.values()), list(upi.values()))
    joint[:3, :3] = np.reshape(R_REVIEW["PA"], (3, 3))
    joint[3:7, 3:7] = np.reshape(R_REVIEW["PB"], (4, 4))
    joint[7, 7] = 1.0
    return s, joint


PPS_CASES = {
    "brewer": dict(variance="brewer"),
    "exact_HT": dict(variance="ht"), "exact_YG": dict(variance="syg"),
    "overton_HT": dict(variance="ht", joint="overton"), "overton_YG": dict(variance="syg", joint="overton"),
    "hr_HT": dict(variance="ht", joint="hartley_rao"), "hr_YG": dict(variance="syg", joint="hartley_rao"),
    "hrsample_HT": dict(variance="ht", joint="hartley_rao"), "hrsample_YG": dict(variance="syg", joint="hartley_rao"),
}


def pps_options(name, joint):
    kw = dict(PPS_CASES[name], prob="pi", strata="strat", psu="unit")
    if name.startswith("exact"):
        kw["joint"] = {"a": joint[:3, :3], "b": pd.DataFrame(joint[3:7, 3:7]), "c": [[1.0]]}
    if name.startswith("hr_"):
        kw["pi_squared_sum"] = {"a": R_REVIEW["psum"][0], "b": R_REVIEW["psum"][1], "c": 1.0}
    return kw


@pytest.mark.parametrize("name", list(PPS_CASES))
def test_clustered_pps_with_certainty_stratum(name):
    frame, joint = pps_fixture()
    kw = pps_options(name, joint)
    for kind, call in [("total", lambda **k: oe.survey_pps_total(frame, ["y2", "x"], **k)),
                       ("mean", lambda **k: oe.survey_pps_mean(frame, "y2", **k)),
                       ("ratio", lambda **k: oe.survey_pps_ratio(frame, "y2", "x", **k))]:
        result = call(**kw)
        reference = np.array(R_REVIEW[f"pps_{name}_{kind}"])
        np.testing.assert_allclose(flat(result), reference, rtol=1e-10, atol=1e-12 * np.abs(reference).max())
        assert result.attrs["df"] == 5 and result.attrs["design"]["n_certainty_psus"] == 1
    if name.startswith("exact"):
        domain = oe.survey_pps_total(frame, "y", domain="dom", missing="drop", **kw)
        np.testing.assert_allclose(flat(domain), R_REVIEW[f"pps_{name}_domain_total"], rtol=1e-11)
    if name == "exact_HT":
        # A single sample's HT variance can be negative: no SE, a note, the covariance kept.
        total = oe.survey_pps_total(frame, ["y2", "x"], **kw)
        assert np.isnan(total.std_error).all() and len(total.attrs["notes"]) == 2


def psu_delta(frame, joint):
    """Delta-check of the sampled PSUs (UNITS order) and each row's PSU position."""
    pi = np.r_[PI_A[[1, 3, 8]], PI_B[[0, 1, 2, 5]], 1.0]
    delta = (joint - np.outer(pi, pi)) / joint
    np.fill_diagonal(delta, 1 - pi)
    return delta, frame.unit.map({u: k for k, u in enumerate(UNITS)}).to_numpy()


def calibrated_z(frame, u, method="linear"):
    """Row values w_k e_k of a linear calibration on (1, x): explicit GREG residuals."""
    d = 1 / frame.pi.to_numpy()
    x = np.column_stack([np.ones(len(frame)), frame.x])
    g = 1 + x @ np.linalg.solve(x.T @ (x * d[:, None]), np.array([34.5, 325.0]) - x.T @ d)
    root = np.sqrt(d).reshape((-1,) + (1,) * (np.ndim(u) - 1))
    e = u - x @ np.linalg.lstsq(x * np.sqrt(d)[:, None], u * root, rcond=None)[0]
    return (d * g).reshape(root.shape) * e, d * g


@pytest.mark.parametrize("name", ["brewer", "exact_HT", "exact_YG", "hrsample_HT", "hrsample_YG"])
def test_calibrated_clustered_pps(name):
    frame, joint = pps_fixture()
    kw = pps_options(name, joint)
    method = "linear" if name.startswith("exact") else "raking"
    calibration = dict(CALIBRATION, method=method)
    if name == "brewer":
        g = oe.svycal(frame.assign(d=1 / frame.pi), "d", **calibration)["weights"].g_weight
        np.testing.assert_allclose(g, R_REVIEW["pps_brewer_raking_g"], rtol=1e-12)
    if name == "exact_YG":
        # R's ppsmat YG differences elements, not PSU totals (see the reference script): the
        # PSU-level SYG of the summed GREG residuals is written out explicitly instead.
        delta, position = psu_delta(frame, joint)
        z, w = calibrated_z(frame, frame[["y2", "x"]].to_numpy())
        sums = np.zeros((8, 2))
        np.add.at(sums, position, z)
        expected = sums.T @ delta @ sums - (sums * delta.sum(1)[:, None]).T @ sums
        result = oe.survey_pps_total(frame, ["y2", "x"], calibration=calibration, **kw)
        np.testing.assert_allclose(result.estimate, w @ frame[["y2", "x"]].to_numpy(), rtol=1e-12)
        np.testing.assert_allclose(result.attrs["covariance"], expected, rtol=1e-9, atol=1e-9 * expected[0, 0])
        assert result.attrs["covariance"][0][0] != pytest.approx(R_REVIEW["pps_exact_HT_linear_total"][2], rel=1e-3)
        return
    key = f"pps_{name}_{method}"
    for kind, call in [("total", lambda **k: oe.survey_pps_total(frame, ["y2", "x"], **k)),
                       ("mean", lambda **k: oe.survey_pps_mean(frame, "y2", **k)),
                       ("ratio", lambda **k: oe.survey_pps_ratio(frame, "y2", "x", **k))]:
        result = call(calibration=calibration, **kw)
        reference = np.array(R_REVIEW[f"{key}_{kind}"])
        # The calibration variable x is reproduced: its variance is zero up to rounding.
        np.testing.assert_allclose(flat(result), reference, rtol=1e-9, atol=1e-11 * np.abs(reference).max())
    if name == "brewer":
        dropped = oe.survey_pps_total(frame, ["y", "y2"], calibration=calibration, missing="drop", **kw)
        np.testing.assert_allclose(flat(dropped), R_REVIEW["pps_brewer_raking_drop_total"], rtol=1e-10)
        for kind, function in [("total", oe.survey_pps_total), ("mean", oe.survey_pps_mean)]:
            domain = function(frame, "y", calibration=calibration, domain="dom", missing="drop", **kw)
            np.testing.assert_allclose(flat(domain), R_REVIEW[f"pps_brewer_raking_domain_{kind}"], rtol=1e-10)


def test_calibrated_pps_domain_keeps_out_of_domain_residuals():
    """R's `[.pps` zeroes the Delta-check block of excluded rows, dropping their calibration residuals.

    The domain total's linearized value w_k e_k is nonzero outside the domain after
    calibration (e_k = 0 - x_k'B); the full HT form keeps it. An explicit NumPy oracle
    gives the implemented value and reproduces R's number when R's zeroing is imitated.
    """
    frame, joint = pps_fixture()
    kw = pps_options("exact_HT", joint)
    result = oe.survey_pps_total(frame, "y", calibration=dict(CALIBRATION, method="linear"), domain="dom",
                                 missing="drop", **kw)
    keep = ((frame.dom == 1) & frame.y.notna()).to_numpy()
    z, _ = calibrated_z(frame, np.where(keep, frame.y.fillna(0.0), 0.0))
    delta, position = psu_delta(frame, joint)
    rows = delta[np.ix_(position, position)]           # rows of one PSU share D_kk = 1 - pi_k
    assert result.estimate.iloc[0] == pytest.approx(R_REVIEW["pps_exact_HT_linear_domain_total_R"][0], rel=1e-12)
    assert result.attrs["covariance"][0][0] == pytest.approx(z @ rows @ z, rel=1e-10)
    imitated = np.where(~keep[:, None] & ~keep[None, :], 0.0, rows)
    assert z @ imitated @ z == pytest.approx(R_REVIEW["pps_exact_HT_linear_domain_total_R"][1], rel=1e-10)
    assert abs(z @ rows @ z - z @ imitated @ z) > 0.05 * abs(z @ rows @ z)


def test_hajek_centre_matches_sampling_varest():
    frame, _ = pps_fixture()
    result = oe.survey_pps_total(frame, "y2", prob="pi", strata="strat", psu="unit", variance="hajek")
    assert result.attrs["covariance"][0][0] == pytest.approx(R_REVIEW["pps_hajek_y2_total_var"][0], rel=1e-12)


def test_review_failure_contract():
    frame, joint = pps_fixture()
    base = dict(prob="pi", strata="strat", psu="unit")
    cases = [
        (dict(variance="ht", joint=pd.DataFrame([["a"] * 8] * 8)), "pps_joint_probability"),
        (dict(variance="ht", joint={"a": pd.DataFrame([["a"] * 3] * 3), "b": joint[3:7, 3:7], "c": [[1.0]]}),
         "pps_joint_probability"),
        (dict(variance="syg", joint="hartley_rao", pi_squared_sum={"a": "1.2", "b": 2.3, "c": 1.0}), "invalid_pps_option"),
        (dict(variance="syg", joint="hartley_rao", pi_squared_sum={"a": True, "b": 2.3, "c": 1.0}), "invalid_pps_option"),
        (dict(variance="syg", joint="hartley_rao", pi_squared_sum={"a": float("nan"), "b": 2.3, "c": 1.0}),
         "invalid_pps_option"),
    ]
    for change, code in cases:
        with pytest.raises(AnalysisError) as error:
            oe.survey_pps_total(frame, "y2", **base, **change)
        assert error.value.code == code, (change, error.value.code)
    design_frame, design = design_fixture()
    with pytest.raises(AnalysisError) as error:   # calibration variables must be complete
        oe.svycal(design_frame, design, population=425, totals={"z": 2300.0})
    assert error.value.code == "calibration_missing"


def test_multistage_declarations_are_refused():
    i = np.arange(24)
    frame = pd.DataFrame({"h": i // 12, "psu": i // 4, "ssu": i, "N1": 6, "N2": 4, "y": 2.0 + i % 7})
    two_stage = oe.survey_two_stage_design(frame, psu="psu", ssu="ssu", population_psu="N1", population_ssu="N2",
                                           strata="h")
    for call in (lambda: oe.svycal(frame, two_stage, population=150.0),
                 lambda: oe.survey_calibrated_total(frame, two_stage, "y", calibration=dict(population=150.0))):
        with pytest.raises(AnalysisError, match="single-stage") as error:
            call()
        assert error.value.code == "invalid_survey_design"


@pytest.mark.parametrize("method,bounds", [("linear", (-np.inf, np.inf)), ("raking", (0.0, np.inf)),
                                           ("logit", (0.6, 1.5)), ("logit", (0.05, 40.0)), ("truncated", (0.8, 1.25)),
                                           ("truncated", (-np.inf, 1.25)), ("truncated", (0.8, np.inf))])
def test_dual_objective_primitive_differentiates_to_the_distance(method, bounds):
    import torch

    from openecon.econometrics.survey_calib.calibration import distance, primitive
    u = torch.linspace(-2.0, 2.0, 401, dtype=torch.float64)
    u = u[((1 + u - bounds[0]).abs() > 1e-3) & ((1 + u - bounds[1]).abs() > 1e-3)]  # away from truncation kinks
    psi, link = primitive(method, *bounds), distance(method, *bounds)
    numeric = (psi(u + 1e-6) - psi(u - 1e-6)) / 2e-6
    np.testing.assert_allclose(numeric.numpy(), link(u)[0].numpy(), rtol=1e-7, atol=1e-8)
    assert float(psi(torch.zeros(1, dtype=torch.float64))[0]) == 0.0
    assert bool(torch.isfinite(psi(u)).all())
