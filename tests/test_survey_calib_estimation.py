"""Calibrated survey totals, means and ratios: R survey references, NumPy GREG oracle, failure contract."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError

# R 4.6.1, survey 4.5: svytotal/svymean/svyratio(deff=TRUE) on calibrate(epsilon=1e-14) designs with
# svydesign(ids=~psu, strata=~h, fpc=~Nh, weights=~d); tests/reference/survey_calib_calibrate.R.
# Covariances are row-major; domain entries are (estimate, variance).
R_CAL = {
    "linear_total": [10035.580017196186, 20892.681860497149],
    "linear_total_vcov": [11980.871773917559, -4198.121553327047, -4198.121553327047, 9404.4671133218053],
    "linear_total_deff": [0.083737232596770905, 0.19767208593898475],
    "linear_mean": [21.816478298252573, 45.418873609776419],
    "linear_mean_vcov": [0.05662037700339121, -0.01983989391931474, -0.01983989391931474, 0.044444551575245135],
    "linear_mean_deff": [0.08373723259677103, 0.19767208593898622],
    "linear_ratio": [2.8964831374031976, 6.0300750529979696],
    "linear_ratio_vcov": [0.0050045677490292047, 0.0080868277035233659, 0.0080868277035233659, 0.018545927169126463],
    "linear_ratio_deff": [0.23852309157452839, 0.344471100164985],
    "linear_domain_total": [4644.3604818206259, 212621.91603604029],
    "linear_domain_mean": [20.36607437195018, 0.15015045782445741],
    "raking_total": [10036.354894109452, 20893.738252572599],
    "raking_total_vcov": [11883.739100885388, -4203.3837994399564, -4203.3837994399564, 9393.8334345108178],
    "raking_total_deff": [0.083173250316640662, 0.1977400872006764],
    "raking_mean": [21.818162813281425, 45.421170114288266],
    "raking_mean_vcov": [0.056161337905885719, -0.019864762757277447, -0.019864762757277447, 0.044394297894663946],
    "raking_mean_deff": [0.083173250316640843, 0.19774008720067782],
    "raking_ratio": [2.8957566388489049, 6.0284019341293931],
    "raking_ratio_vcov": [0.0049819622821584038, 0.008072531712370757, 0.008072531712370757, 0.018545952193682585],
    "raking_ratio_deff": [0.23775995290387386, 0.34421011057919348],
    "raking_domain_total": [4648.1256449686562, 213555.06751534832],
    "raking_domain_mean": [20.370870639868588, 0.14991277782208201],
    "logit_total": [10035.689249309622, 20891.817751940223],
    "logit_total_vcov": [11967.208429781698, -4190.1351758169531, -4190.1351758169531, 9382.1292029046399],
    "logit_total_deff": [0.083736403630772838, 0.19715204722820825],
    "logit_mean": [21.816715759368744, 45.416995112913511],
    "logit_mean_vcov": [0.056555805433750904, -0.019802151114446748, -0.019802151114446748, 0.04433898489085393],
    "logit_mean_deff": [0.083736403630772796, 0.19715204722820934],
    "logit_ratio": [2.8967893529735314, 6.030397486873869],
    "logit_ratio_vcov": [0.0050020867861734625, 0.0080803574381564641, 0.0080803574381564641, 0.018521399178939989],
    "logit_ratio_deff": [0.23822892331355802, 0.34394713669923632],
    "logit_domain_total": [4646.876450210927, 211811.47333863308],
    "logit_domain_mean": [20.365387451662222, 0.14907451949066261],
    "truncated_total": [10035.560131448265, 20891.501200897037],
    "truncated_total_vcov": [11994.842828662524, -4203.578422511252, -4203.578422511252, 9366.9121054377902],
    "truncated_total_deff": [0.083907353273608903, 0.19678029143607095],
    "truncated_mean": [21.816435068365788, 45.416306958471829],
    "truncated_mean_vcov": [0.056686402781959064, -0.01986568252604556, -0.01986568252604556, 0.044267070441577822],
    "truncated_mean_deff": [0.083907353273609014, 0.19678029143607251],
    "truncated_ratio": [2.8972137343642661, 6.0312671557867086],
    "truncated_ratio_vcov": [0.0050064331131219305, 0.0080880105335822596, 0.0080880105335822596, 0.018549894102008833],
    "truncated_ratio_deff": [0.23832194754425831, 0.34444127298050081],
    "truncated_domain_total": [4647.0790716625652, 211993.46510509538],
    "truncated_domain_mean": [20.368293923569528, 0.1495019883068793],
    "by_raking_total": [9801.4640706802911, 20567.978395902996],
    "by_raking_total_vcov": [12640.156839025054, -4029.4063242071702, -4029.4063242071702, 6236.8012931964095],
}

N = 460.0
CATEGORICAL = {"sex": {"F": 0.47 * N, "M": 0.53 * N}, "age": {"a": 0.33 * N, "b": 0.32 * N, "c": 0.35 * N}}
BOUNDS = {"linear": None, "raking": None, "logit": (0.6, 1.5), "truncated": (0.8, 1.25)}
BY = dict(method="raking", by="h", tol=1e-13,
          categorical={1: {"sex": {"F": 66, "M": 90}}, 2: {"sex": {"F": 63, "M": 85}}, 3: {"sex": {"F": 68, "M": 80}}},
          totals={1: {"x": 2900}, 2: {"x": 2700}, 3: {"x": 2750}})


def spec(method):
    return dict(method=method, bounds=BOUNDS[method], population=N, categorical=CATEGORICAL,
                totals={"x": 8600.0}, tol=1e-13)


def fixture():
    i = np.arange(36)
    frame = pd.DataFrame({
        "h": i // 12 + 1, "psu": (i // 3) % 4 + 1, "d": 8 + (i * 7) % 9 + 0.25 * (i % 4),
        "sex": np.where((i * 5) % 7 < 3, "F", "M"), "age": np.array(["a", "b", "c"])[(i + i // 3) % 3],
        "x": 10 + (i * 13) % 17 + 0.5 * (i % 3), "z": 5.0 + (i * 5) % 6, "dom": ((i * 7) % 4 < 2).astype(int),
    })
    frame["Nh"] = np.array([10, 12, 8])[frame.h - 1]
    frame["y"] = 3 + 0.9 * frame.x + (i * 11) % 7 - 2 * (frame.sex == "F")
    frame["y2"] = 50 - 0.4 * frame.x + ((i * 3) % 5) * 1.5
    return frame, oe.survey_design(frame, weights="d", psu="psu", strata="h", fpc="Nh")


@pytest.mark.parametrize("method", list(BOUNDS))
def test_r_survey_calibrated_totals_means_ratios(method):
    frame, design = fixture()
    cal = oe.svycal(frame, design, **spec(method))
    for kind, function, args in [("total", oe.survey_calibrated_total, (["y", "y2"],)),
                                 ("mean", oe.survey_calibrated_mean, (["y", "y2"],)),
                                 ("ratio", oe.survey_calibrated_ratio, (["y", "y2"], ["z", "z"]))]:
        result = function(frame, design, *args, calibration=cal, deff=True)
        np.testing.assert_allclose(result.estimate, R_CAL[f"{method}_{kind}"], rtol=1e-12)
        np.testing.assert_allclose(np.ravel(result.attrs["covariance"]), R_CAL[f"{method}_{kind}_vcov"],
                                   rtol=1e-10, atol=1e-12 * np.abs(R_CAL[f"{method}_{kind}_vcov"]).max())
        np.testing.assert_allclose(result.deff, R_CAL[f"{method}_{kind}_deff"], rtol=1e-10)
        assert result.attrs["df"] == 9 and result.attrs["calibration"]["method"] == method
    for kind, function in [("total", oe.survey_calibrated_total), ("mean", oe.survey_calibrated_mean)]:
        result = function(frame, design, "y", calibration=cal, domain="dom")
        np.testing.assert_allclose([result.estimate.iloc[0], result.attrs["covariance"][0][0]],
                                   R_CAL[f"{method}_domain_{kind}"], rtol=1e-10)
        assert result.attrs["n_domain"] == int(frame.dom.sum())


def test_r_survey_stratum_wise_raking():
    frame, design = fixture()
    result = oe.survey_calibrated_total(frame, design, ["y", "y2"], calibration=BY)
    np.testing.assert_allclose(result.estimate, R_CAL["by_raking_total"], rtol=1e-12)
    np.testing.assert_allclose(np.ravel(result.attrs["covariance"]), R_CAL["by_raking_total_vcov"], rtol=1e-10)


def greg_oracle(frame, g, x_blocks, kind, y, x=None, member=None):
    """Explicit Deville–Särndal residual linearization with stratified PSU totals and FPC."""
    d = frame.d.to_numpy()
    w = d * g
    member = np.ones(len(frame)) if member is None else member
    values = np.nan_to_num(frame[y].to_numpy(float)) * member[:, None]
    if kind == "total":
        theta, u = w @ values, values
    else:
        den = np.ones_like(values) if kind == "mean" else np.nan_to_num(frame[[x]].to_numpy(float))
        total = (w * member) @ den
        theta = w @ values / total
        u = member[:, None] * (values - den * theta) / total
    e = np.empty_like(u)
    for rows, block in x_blocks:
        root = np.sqrt(d[rows])[:, None]
        beta = np.linalg.lstsq(block * root, u[rows] * root, rcond=None)[0]
        e[rows] = u[rows] - block @ beta
    z = w[:, None] * e
    covariance = np.zeros((len(y), len(y)))
    for h in (1, 2, 3):
        mask = frame.h.to_numpy() == h
        sums = np.array([z[mask & (frame.psu.to_numpy() == p)].sum(0) for p in (1, 2, 3, 4)])
        m, population = 4, frame.Nh[mask].iloc[0]
        centred = sums - sums.mean(0)
        covariance += (1 - m / population) * m / (m - 1) * centred.T @ centred
    return theta, covariance


def test_numpy_oracle_stratum_wise_domain_and_missing_policy():
    frame, design = fixture()
    frame.loc[[4, 8, 17], "y"] = np.nan  # rows 4 and 8 are in the domain
    cal = oe.svycal(frame, design, **BY)
    g = cal["weights"].g_weight.to_numpy()
    blocks = [(np.flatnonzero(frame.h == h), np.column_stack([frame.sex == "F", frame.sex == "M", frame.x])
               [frame.h.to_numpy() == h].astype(float)) for h in (1, 2, 3)]
    with pytest.raises(AnalysisError) as error:
        oe.survey_calibrated_mean(frame, design, ["y"], calibration=cal)
    assert error.value.code == "survey_missing"
    member = (frame.dom.to_numpy() == 1) & frame.y.notna().to_numpy()
    for kind, function, args in [("total", oe.survey_calibrated_total, (["y", "y2"],)),
                                 ("mean", oe.survey_calibrated_mean, (["y", "y2"],)),
                                 ("ratio", oe.survey_calibrated_ratio, (["y", "y2"], ["z", "x"]))]:
        result = function(frame, design, *args, calibration=cal, domain="dom", missing="drop", alpha=0.1)
        y = ["y", "y2"]
        if kind == "ratio":
            theta = [greg_oracle(frame, g, blocks, kind, [a], b, member.astype(float))[0][0]
                     for a, b in zip(y, ["z", "x"])]
            # Joint covariance of two ratios with different denominators, assembled from paired influences.
            w = frame.d.to_numpy() * g
            influences = []
            for a, b, t in zip(y, ["z", "x"], theta):
                total = (w * member) @ frame[b].to_numpy()
                influences.append(member * (np.nan_to_num(frame[a].to_numpy()) - t * frame[b].to_numpy()) / total)
            u = np.column_stack(influences)
            frame_u = frame.assign(u1=u[:, 0], u2=u[:, 1])
            _, covariance = greg_oracle(frame_u, g, blocks, "total", ["u1", "u2"])
        else:
            theta, covariance = greg_oracle(frame, g, blocks, kind, y, member=member.astype(float))
        np.testing.assert_allclose(result.estimate, theta, rtol=1e-12)
        np.testing.assert_allclose(result.attrs["covariance"], covariance, rtol=1e-10,
                                   atol=1e-12 * np.abs(covariance).max())
        assert result.attrs["n_used"] == member.sum() and result.attrs["n_missing_excluded"] == 2
        se = np.sqrt(np.diag(covariance))
        np.testing.assert_allclose(result.ci_high, np.asarray(theta) + stats.t.ppf(0.95, 9) * se, rtol=1e-11)
        np.testing.assert_allclose(result.p_value, 2 * stats.t.sf(np.abs(np.asarray(theta) / se), 9), rtol=1e-9)


@pytest.mark.parametrize("method", ["linear", "raking"])
def test_calibration_variables_are_reproduced_without_variance(method):
    frame, design = fixture()
    frame["female"] = (frame.sex == "F").astype(float)
    result = oe.survey_calibrated_total(frame, design, ["x", "female"], calibration=spec(method))
    np.testing.assert_allclose(result.estimate, [8600.0, 0.47 * N], rtol=1e-12)
    assert np.abs(result.attrs["covariance"]).max() < 1e-12 * 8600.0 ** 2
    mean = oe.survey_calibrated_mean(frame, design, "x", calibration=spec(method))
    assert mean.estimate.iloc[0] == pytest.approx(8600.0 / N, rel=1e-12)
    assert mean.std_error.iloc[0] < 1e-9


def test_saved_specification_and_svycal_result_checks(tmp_path):
    frame, design = fixture()
    cal = oe.svycal(frame, design, **spec("logit"))
    restored = json.loads(json.dumps(cal.attrs["calibration"]))
    a = oe.survey_calibrated_ratio(frame, design, "y", "z", calibration=cal)
    b = oe.survey_calibrated_ratio(frame, design, "y", "z", calibration=restored)
    c = oe.survey_calibrated_ratio(frame, design, "y", "z", calibration=spec("logit"))
    path = tmp_path / "calibration.json"
    path.write_text(oe.summary_state(cal))
    recovered = oe.restore_summary(path.read_text())
    d = oe.survey_calibrated_ratio(frame, design, "y", "z", calibration=recovered)
    assert recovered.attrs == cal.attrs
    for other in (b, c, d):
        np.testing.assert_allclose(other.estimate, a.estimate, rtol=1e-15)
        np.testing.assert_allclose(other.attrs["covariance"], a.attrs["covariance"], rtol=1e-12)
    text = json.dumps(a.attrs)
    assert json.loads(text)["calibration"]["spec"] == restored
    assert "y/z" in str(a) and "estimate" in a.to_latex()
    other = frame.assign(x=frame.x[::-1].to_numpy())
    with pytest.raises(AnalysisError) as error:
        oe.survey_calibrated_total(other, design, "y", calibration=cal)
    assert error.value.code == "calibration_changed"
    with pytest.raises(AnalysisError) as error:
        oe.survey_calibrated_total(frame, design, "y", calibration=oe.svycal(frame.assign(d=frame.d * 2), "d",
                                                                               **spec("linear")))
    assert error.value.code == "calibration_changed"


def test_failure_contract():
    frame, design = fixture()
    calibration = spec("linear")
    cases = [
        (lambda: oe.survey_calibrated_total(frame, "d", "y", calibration=calibration), "invalid_survey_design"),
        (lambda: oe.survey_calibrated_total(frame, design, "y", calibration=None), "invalid_calibration_spec"),
        (lambda: oe.survey_calibrated_total(frame, design, "y", calibration={"method": "raking", "foo": 1}),
         "invalid_calibration_spec"),
        (lambda: oe.survey_calibrated_total(frame, design, "sex", calibration=calibration), "non_numeric_column"),
        (lambda: oe.survey_calibrated_total(frame, design, ["y", "y"], calibration=calibration),
         "invalid_survey_target"),
        (lambda: oe.survey_calibrated_ratio(frame, design, ["y"], ["z", "x"], calibration=calibration),
         "invalid_survey_target"),
        (lambda: oe.survey_calibrated_ratio(frame.assign(z=0.0), design, "y", "z", calibration=calibration),
         "survey_denominator"),
        (lambda: oe.survey_calibrated_mean(frame.assign(dom=0), design, "y", calibration=calibration,
                                           domain="dom"), "empty_survey_domain"),
        (lambda: oe.survey_calibrated_mean(frame.assign(dom=2), design, "y", calibration=calibration,
                                           domain="dom"), "invalid_survey_domain"),
        (lambda: oe.survey_calibrated_mean(frame, design, "y", calibration=calibration, alpha=1.5),
         "invalid_survey_option"),
        (lambda: oe.survey_calibrated_mean(frame, design, "y", calibration=calibration, deff=1),
         "invalid_survey_option"),
        (lambda: oe.survey_calibrated_mean(frame, design, "y", calibration=calibration, missing="ignore"),
         "invalid_survey_option"),
        (lambda: oe.survey_calibrated_mean(frame, design, "absent", calibration=calibration), "missing_columns"),
        (lambda: oe.survey_calibrated_mean(frame.assign(d=frame.d + 1), design, "y", calibration=calibration),
         "survey_design_changed"),
    ]
    for call, code in cases:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code, (code, error.value.code)


def test_census_strata_and_single_stratum_design():
    frame, _ = fixture()
    census = frame.assign(Nh=np.where(frame.h == 1, 4, frame.Nh))
    design = oe.survey_design(census, weights="d", psu="psu", strata="h", fpc="Nh")
    cal = spec("raking")
    full = oe.survey_calibrated_total(census, design, "y", calibration=cal)
    g = oe.svycal(census, design, **cal)["weights"].g_weight.to_numpy()
    x = np.column_stack([np.ones(36), census.sex == "M", census.age == "b", census.age == "c", census.x]).astype(float)
    _, covariance = greg_oracle(census, g, [(np.arange(36), x)], "total", ["y"])
    np.testing.assert_allclose(full.attrs["covariance"], covariance, rtol=1e-10)
    plain = oe.survey_design(frame.assign(one=1), weights="d", psu="psu", strata="h")
    srs = oe.survey_calibrated_total(frame.assign(one=1), plain, "y", calibration=cal)
    assert srs.attrs["covariance"][0][0] > full.attrs["covariance"][0][0]
