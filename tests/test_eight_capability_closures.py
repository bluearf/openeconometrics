"""Independent small-matrix, distribution, bootstrap and persistence oracles."""

import itertools
import json

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.stats.power_distributions import f_power
from openecon.engines.distributions import f_isf
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


def linear_data(seed=27, groups=6):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(groups), np.arange(groups) + 7)
    x, z = rng.normal(size=(2, len(ids)))
    noise = rng.normal(size=len(ids)) + 0.6 * rng.normal(size=groups)[ids]
    return pd.DataFrame(
        dict(id=ids, t=np.arange(len(ids)), x=x, z=z, y=1 + 0.7 * x - 0.3 * z + noise)
    )


def restricted_oracle(data, null, multipliers, fe=False):
    dummy = (
        pd.get_dummies(data.id, drop_first=True, dtype=float).to_numpy()
        if fe
        else np.empty((len(data), 0))
    )
    x = np.c_[np.ones(len(data)), data.x, data.z, dummy]
    y = data.y.to_numpy()
    codes = pd.factorize(data.id)[0]
    n, k = x.shape
    groups = codes.max() + 1
    bread = np.linalg.inv(x.T @ x)
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    r = np.eye(k)[[1, 2]][: len(null)]
    values = np.array(list(null.values()))

    def variance(u):
        scores = np.array([(x * u[:, None])[codes == g].sum(0) for g in range(groups)])
        return bread @ (scores.T @ scores) @ bread * groups / (groups - 1) * (n - 1) / (n - k)

    def statistic(bb, u):
        d = r @ bb - values
        return d @ np.linalg.solve(r @ variance(u) @ r.T, d)

    v = variance(y - x @ b)
    observed = statistic(b, y - x @ b)
    # Independent KKT constrained fit, not the production projection formula.
    a = np.block([[x.T @ x, r.T], [r, np.zeros((len(r), len(r)))]])
    restricted = np.linalg.solve(a, np.r_[x.T @ y, values])[:k]
    draws = []
    for weight in multipliers:
        response = x @ restricted + (y - x @ restricted) * np.array(weight)[codes]
        bb = np.linalg.lstsq(x, response, rcond=None)[0]
        draws.append(statistic(bb, response - x @ bb))
    return observed, v, restricted, np.array(draws)


@pytest.mark.parametrize("fe", [False, True])
@pytest.mark.parametrize("joint", [False, True])
def test_restricted_wild_complete_draws_full_covariance_and_joint_null(fe, joint):
    data = linear_data()
    result = (
        oe.xtreg(data=data, y="y", x=["x", "z"], panel="id", model="fe")
        if fe
        else oe.ols(data=data, y="y", x=["x", "z"])
    )
    null = {"x": 0.7, "z": -0.3} if joint else {"x": 0.7}
    output = oe.wild_cluster_test(result, data=data, cluster="id", null=null, enumerate_all=True)
    expected = restricted_oracle(data, null, output.attrs["multipliers"], fe)
    assert output.attrs["draws"] == 64
    assert set(map(tuple, output.attrs["multipliers"])) == set(
        itertools.product([-1.0, 1.0], repeat=6)
    )
    assert_allclose(output.attrs["statistic"], expected[0], rtol=1e-11)
    assert_allclose(output.attrs["covariance_matrix"], expected[1], rtol=1e-10, atol=1e-12)
    assert_allclose(output.attrs["restricted_parameters"], expected[2], atol=1e-12)
    assert_allclose(output.attrs["bootstrap_statistics"], expected[3], rtol=1e-9, atol=1e-10)
    count = np.sum(expected[3] >= expected[0] * (1 - 1e-12))
    assert output.attrs["p_value"] == count / 64
    json.dumps(output.attrs, allow_nan=False)


@pytest.mark.parametrize("wild", ["rademacher", "mammen", "webb"])
def test_wild_seed_dtype_mc_convention_and_inverted_grid(wild):
    data = linear_data()
    result = oe.ols(data=data, y="y", x=["x", "z"])
    torch.manual_seed(461)
    state = torch.random.get_rng_state().clone()
    output = oe.wild_cluster_test(
        result, data=data, cluster="id", null={"x": 0.7}, reps=49, seed=3, wild=wild
    )
    assert torch.equal(state, torch.random.get_rng_state())
    assert output.attrs["p_value"] == (output.attrs["exceedances"] + 1) / 50
    assert (
        oe.wild_cluster_test(
            result, data=data, cluster="id", null={"x": 0.7}, reps=49, seed=3, wild=wild
        ).attrs
        == output.attrs
    )
    grid = [-0.5, 0.0, 0.5, 1.0, 1.5]
    ci = oe.wild_cluster_confidence_set(
        result, data=data, cluster="id", term="x", grid=grid, reps=49, seed=3, wild=wild
    )
    for row in ci["grid"].to_dict("records"):
        test = oe.wild_cluster_test(
            result, data=data, cluster="id", null={"x": row["null"]}, reps=49, seed=3, wild=wild
        )
        assert row["p_value"] == test.attrs["p_value"]
        assert row["accepted"] == (row["p_value"] > 0.05)
    assert ci.attrs["outside_grid"].startswith("unknown")
    assert "tabular" in ci.to_latex()


def test_wild_missing_matching_and_negative_domains():
    data = linear_data()
    data.loc[2, "y"] = np.nan
    result = oe.ols(data=data, y="y", x=["x", "z"], missing="drop")
    test = oe.wild_cluster_test(result, data=data, cluster="id", null={"x": 0}, reps=49)
    assert 2 not in test.attrs["sample_positions"]
    for options, code in [
        (dict(max_work=1), "work_budget_exceeded"),
        (dict(null={"absent": 0}), "invalid_restrictions"),
        (dict(enumerate_all=True, wild="webb"), "enumeration_limit"),
    ]:
        kw = dict(data=data, cluster="id", null={"x": 0}, reps=49)
        kw.update(options)
        with pytest.raises(AnalysisError) as error:
            oe.wild_cluster_test(result, **kw)
        assert error.value.code == code
    changed = data.copy()
    changed.loc[0, "x"] += 1
    with pytest.raises(AnalysisError):
        oe.wild_cluster_test(result, data=changed, cluster="id", null={"x": 0}, reps=49)


def iv_data():
    data = linear_data(groups=20)
    rng = np.random.default_rng(921)
    data["d"] = 0.6 * data.z + 0.1 * data.x + rng.normal(size=len(data))
    data["y"] += 1.2 * data.d
    return data


@pytest.mark.parametrize("kind", ["nonrobust", "robust", "cluster", "hac"])
def test_saved_iv_structural_tests_have_original_sample_and_covariance(kind):
    data = iv_data()
    data.loc[4, "y"] = np.nan
    extra = (
        {"cluster": "id"}
        if kind == "cluster"
        else {"time": "t", "lags": 2}
        if kind == "hac"
        else {}
    )
    fitted = oe.ivregress(
        data=data,
        y="y",
        x=["x"],
        endog=["d"],
        instruments=["z"],
        covariance=kind,
        missing="drop",
        **extra,
    )
    restored = ResultBundle.model_validate_json(fitted.model_dump_json())
    out = oe.iv_saved_weak_test(restored, data=data, null=1.2)
    direct = oe.iv_weak_test(
        data=data.iloc[fitted.sample_positions],
        y="y",
        x=["x"],
        endog=["d"],
        instruments=["z"],
        null=1.2,
        covariance=kind,
        **extra,
    )
    pd.testing.assert_frame_equal(out["test"], direct["test"])
    assert out.attrs["original_sample_positions"] == fitted.sample_positions
    assert out.attrs["iv_refitted"] is False
    if kind == "nonrobust":
        saved = oe.iv_saved_ar_confidence_set(restored, data=data)
        expected = oe.iv_ar_confidence_set(
            data=data, y="y", x=["x"], endog="d", instruments=["z"], missing="drop"
        )
        pd.testing.assert_frame_equal(saved["confidence_set"], expected["confidence_set"])
        clr = oe.iv_saved_weak_test(restored, data=data, null=1.2, method="clr")
        assert np.isfinite(clr["test"].p_value.iloc[0])
    else:
        with pytest.raises(AnalysisError):
            oe.iv_saved_ar_confidence_set(restored, data=data)
    changed = data.copy()
    changed.loc[0, "y"] += 0.1
    with pytest.raises(AnalysisError):
        oe.iv_saved_weak_test(restored, data=changed, null=1.2)


def test_fisher_combination_independent_reference_and_zero_not_clipped():
    p = [0.001, 0.032, 0.91, 1.0]
    data = pd.DataFrame(dict(unit=list("abcd"), p=p))
    out = oe.fisher_johansen(
        data=data,
        unit="unit",
        p_value="p",
        rank=0,
        calibration="Fixture p-values: trace; constant; calibrated null",
        independent=True,
    )
    statistic = -2 * np.log(p).sum()
    assert_allclose(
        out["test"].iloc[0][["statistic", "p_value"]].to_numpy(float),
        [statistic, stats.chi2.sf(statistic, 8)],
        atol=2e-13,
    )
    for change in [
        dict(independent=False),
        dict(calibration=""),
        dict(data=data.assign(p=[0.0, 0.1, 0.2, 0.3])),
    ]:
        args = dict(
            data=data, unit="unit", p_value="p", rank=0, calibration="declared", independent=True
        )
        args.update(change)
        with pytest.raises(AnalysisError):
            oe.fisher_johansen(**args)


def spatial_data():
    rng = np.random.default_rng(713)
    n = 75
    keys = [f"u{i}" for i in range(n)]
    weights = oe.spatial_weights(
        keys, [(keys[i], keys[j], 1.0) for i in range(n) for j in [(i - 1) % n, (i + 1) % n]]
    )
    w = weights.dense().numpy()
    x = rng.normal(size=n)
    instruments = w @ x
    y = np.linalg.solve(np.eye(n) - 0.3 * w, 1 + 0.8 * x + rng.normal(size=n) * 0.3)
    return pd.DataFrame(dict(key=keys, x=x, z=instruments, z2=w @ instruments, y=y)), weights, w


@pytest.mark.parametrize("kind", ["nonrobust", "HC0"])
def test_sar_iv_projection_full_v_delta_impacts_saved_prediction(kind):
    data, weights, w = spatial_data()
    fitted = oe.sar_iv(
        data,
        "y",
        ["x"],
        key="key",
        instruments=["z", "z2"],
        spatial_weights=weights,
        covariance=kind,
    )
    x = np.c_[np.ones(len(data)), data.x, w @ data.y]
    z = np.c_[np.ones(len(data)), data.x, data.z, data.z2]
    projection = z @ np.linalg.solve(z.T @ z, z.T)
    bread = np.linalg.inv(x.T @ projection @ x)
    b = bread @ x.T @ projection @ data.y
    residual = data.y.to_numpy() - x @ b
    v = (
        bread * (residual @ residual / (len(data) - 3))
        if kind == "nonrobust"
        else bread
        @ ((projection @ x * residual[:, None]).T @ (projection @ x * residual[:, None]))
        @ bread
    )
    assert_allclose([c.estimate for c in fitted.coefficients], b, atol=2e-12)
    assert_allclose(fitted.covariance_matrix, v, rtol=1e-10, atol=1e-12)
    restored = ResultBundle.model_validate_json(fitted.model_dump_json())
    prediction = oe.sar_iv_predict(restored, data=data)["network mean"]

    def means(value):
        return np.linalg.solve(
            np.eye(len(data)) - value[-1] * w, np.c_[np.ones(len(data)), data.x] @ value[:-1]
        )

    mean = means(b)
    gradient = np.column_stack(
        [(means(b + np.eye(3)[j] * 1e-5) - means(b - np.eye(3)[j] * 1e-5)) / 2e-5 for j in range(3)]
    )
    assert_allclose(prediction["mean"], mean, atol=1e-12)
    assert_allclose(
        prediction["std_error"], np.sqrt(np.sum(gradient @ v * gradient, axis=1)), rtol=1e-8
    )
    shuffled = data.sample(frac=1, random_state=3)
    shuffled_pred = oe.sar_iv_predict(restored, data=shuffled)["network mean"].set_index("key")
    assert_allclose(shuffled_pred.loc[data.key, "mean"], mean, atol=1e-12)

    def impacts(value):
        multiplier = np.linalg.inv(np.eye(len(data)) - value[-1] * w) * value[1]
        return np.array(
            [
                np.trace(multiplier) / len(data),
                (multiplier.sum() - np.trace(multiplier)) / len(data),
                multiplier.sum() / len(data),
            ]
        )

    imp = oe.spatial_impacts(restored)["Spatial multiplier impacts (delta method)"]
    jac = np.column_stack(
        [
            (impacts(b + np.eye(3)[j] * 1e-5) - impacts(b - np.eye(3)[j] * 1e-5)) / 2e-5
            for j in range(3)
        ]
    )
    assert_allclose(imp.estimate, impacts(b), atol=2e-12)
    assert_allclose(imp.std_error, np.sqrt(np.diag(jac @ v @ jac.T)), rtol=1e-8)
    assert "tabular" in fitted.to_latex()
    replay = oe.fit(restored.spec, data=data)
    assert_allclose([c.estimate for c in replay.coefficients], b, atol=2e-12)


def panel_data(seed=26, groups=5, periods=24):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(groups), periods)
    x = rng.uniform(-0.7, 0.7, len(ids))
    y = 0.3 * ids + 1.2 * x + (1 + 0.04 * ids + 0.07 * x) * rng.normal(size=len(ids))
    return pd.DataFrame(dict(id=ids, time=np.tile(np.arange(periods), groups), x=x, y=y))


def quantile_oracle(data):
    data = data.sort_values(["id", "time"])
    codes = pd.factorize(data.id)[0]
    design = np.c_[data.x, np.eye(codes.max() + 1)[codes]]
    loc = np.linalg.lstsq(design, data.y, rcond=None)[0]
    residual = data.y - design @ loc
    transformed = 2 * residual * ((residual >= 0) - np.mean(residual >= 0))
    scale = np.linalg.lstsq(design, transformed, rcond=None)[0]
    quantiles = np.quantile(
        residual / (design @ scale), [0.25, 0.5, 0.75], method="averaged_inverted_cdf"
    )
    return loc[0] + quantiles * scale[0]


@pytest.mark.parametrize("split", [False, True])
def test_panel_whole_individual_draws_corrected_estimate_joint_v_persistence(split):
    data = panel_data()
    fitted = oe.panel_mmqr(data=data, y="y", x=["x"], panel="id", time="time")
    state = torch.random.get_rng_state().clone()
    output = oe.panel_mmqr_bootstrap(fitted, data=data, reps=49, seed=81, split_panel=split)
    assert torch.equal(state, torch.random.get_rng_state())

    def oracle(sample):
        full = quantile_oracle(sample)
        if not split:
            return full
        return (
            2 * full
            - (
                quantile_oracle(sample.loc[sample.time < 12])
                + quantile_oracle(sample.loc[sample.time >= 12])
            )
            / 2
        )

    assert_allclose(output["coefficients"].estimate, oracle(data), atol=2e-12)
    values = []
    for choices in output.attrs["panel_draw_indices"]:
        sample = pd.concat(
            [data.loc[data.id == choice].assign(id=i) for i, choice in enumerate(choices)],
            ignore_index=True,
        )
        values.append(oracle(sample))
    assert_allclose(output.attrs["bootstrap_draws"], values, atol=3e-12)
    assert_allclose(
        output.attrs["covariance_matrix"], np.cov(values, rowvar=False), rtol=2e-11, atol=1e-12
    )
    assert abs(output.attrs["covariance_matrix"][0][2]) > 1e-6
    restored = oe.restore_summary(oe.summary_state(output))
    assert restored.attrs == output.attrs
    assert_allclose(restored["covariance"], output["covariance"])
    predicted = oe.panel_mmqr_resampled_predict(restored, data=data)["conditional means"]
    components = []
    for source in output.attrs["fitted_states"]:
        r = ResultBundle.model_validate(source)
        effects = {e["panel"]: e for e in r.extra["individual_effects"]}
        b = np.array(r.extra["location_coefficients"])
        g = np.array(r.extra["scale_coefficients"])
        a = np.array([effects[i]["location"] for i in data.id])
        d = np.array([effects[i]["scale"] for i in data.id])
        components.append(
            (a + data[["x"]].to_numpy() @ b)[:, None]
            + (d + data[["x"]].to_numpy() @ g)[:, None]
            * np.array(r.extra["error_quantiles"])[None, :]
        )
    expected = sum(
        weight * component
        for weight, component in zip(output.attrs["combination_weights"], components, strict=True)
    )
    assert_allclose(predicted[["q0.25", "q0.5", "q0.75"]], expected, atol=1e-12)
    assert "tabular" in output.to_latex()


def test_panel_split_time_short_T_weights_and_failed_draw_domains():
    data = panel_data()
    unbalanced = data.drop(index=0)
    fitted = oe.panel_mmqr(data=unbalanced, y="y", x=["x"], panel="id", time="time")
    with pytest.raises(AnalysisError, match="same complete calendar"):
        oe.panel_mmqr_bootstrap(fitted, data=unbalanced, reps=49, split_panel=True)
    fitted = oe.panel_mmqr(data=data, y="y", x=["x"], panel="id", time="time")
    with pytest.raises(AnalysisError) as error:
        oe.panel_mmqr_bootstrap(fitted, data=data, reps=49, max_work=1)
    assert error.value.code == "work_budget_exceeded"
    with pytest.raises(AnalysisError):
        oe.panel_mmqr_resampled_predict(
            oe.panel_mmqr_bootstrap(fitted, data=data, reps=49), data=data.assign(id=100)
        )


@pytest.mark.parametrize(
    "df1,df2,ncp", [(1, 5, 0), (3, 27, 0.001), (6, 90, 12), (20, 100, 150), (3, 900, 2000)]
)
def test_native_noncentral_F_against_scipy(df1, df2, ncp):
    expected = stats.ncf.sf(stats.f.isf(0.05, df1, df2), df1, df2, ncp) if ncp else 0.05
    assert f_power(f_isf(0.05, df1, df2), df1, df2, ncp) == pytest.approx(expected, abs=3e-11)


@pytest.mark.parametrize(
    "method,options",
    [
        ("power_anova", dict(effect=0.25, groups=4)),
        ("power_regression", dict(effect=0.125, predictors=5, tested=2)),
    ],
)
def test_F_planning_minimum_n_MDE_roundtrip_and_published_regression_example(method, options):
    route = getattr(oe, method)
    out = route(**options, power=0.8)
    n = int(out["plan"].iloc[0].n)
    assert route(**options, n=n)["plan"].iloc[0].power >= 0.8
    assert route(**options, n=n - 1)["plan"].iloc[0].power < 0.8
    effect_key = "effect"
    mde_options = {k: v for k, v in options.items() if k != effect_key}
    mde = route(**mde_options, n=n, power=0.8)
    assert route(**mde_options, **{effect_key: float(mde["plan"].iloc[0].effect)}, n=n)[
        "plan"
    ].iloc[0].power == pytest.approx(0.8, abs=1e-10)
    if method == "power_regression":
        assert n == 81  # Stata public power rsquared .1 .2, tested2 controls3
    json.dumps(out.attrs, allow_nan=False)


def test_paired_cluster_survival_assumptions_oracles_rounding_and_scenarios():
    paired = oe.power_paired_mean(0.5, sd_before=1.2, sd_after=0.9, correlation=0.4, n=90)["plan"].iloc[0]
    sd = np.sqrt(1.2**2 + 0.9**2 - 2 * 0.4 * 1.2 * 0.9)
    z = stats.norm.isf(0.025)
    shift = 0.5 * np.sqrt(90) / sd
    assert paired.power == pytest.approx(
        stats.norm.sf(z - shift) + stats.norm.cdf(-z - shift), abs=1e-13
    )
    clustered = oe.power_cluster_mean(0.3, sd=1, cluster_size=25, icc=0.1, n=12, ratio=1.3)[
        "plan"
    ].iloc[0]
    se = np.sqrt((1 + 24 * 0.1) / 25 * (1 / 12 + 1 / 16))
    assert clustered.design_standard_error == pytest.approx(se)
    assert clustered.total_n == (12 + 16) * 25
    for alternative, hr in [("two-sided", 0.7), ("upper", 1.5), ("lower", 0.65)]:
        out = oe.power_logrank(
            hr, power=0.8, alternative=alternative, information_fraction=0.4, event_fraction=0.3
        )
        count = int(out["plan"].iloc[0].events)
        shift = np.log(hr) * np.sqrt(count * 0.4 * 0.6)
        crit = stats.norm.isf(0.025 if alternative == "two-sided" else 0.05)
        expected = (
            (stats.norm.sf(crit - shift) + stats.norm.cdf(-crit - shift))
            if alternative == "two-sided"
            else stats.norm.sf(crit - (shift if alternative == "upper" else -shift))
        )
        assert out["plan"].iloc[0].power == pytest.approx(expected, abs=1e-13)
        assert out["plan"].iloc[0].expected_enrollment == np.ceil(count / 0.3)
        assert (
            oe.power_logrank(hr, events=count - 1, alternative=alternative, information_fraction=0.4)["plan"]
            .iloc[0]
            .power
            < 0.8
        )
        mde = oe.power_logrank(
            events=count,
            power=0.8,
            alternative=alternative,
            direction="lower" if hr < 1 else "upper",
            information_fraction=0.4,
        )
        assert oe.power_logrank(
            float(mde["plan"].iloc[0].hazard_ratio),
            events=count,
            alternative=alternative,
            information_fraction=0.4,
        )["plan"].iloc[0].power == pytest.approx(0.8, abs=1e-11)
    scenarios = oe.planning_scenarios(
        "power_regression",
        [dict(effect=0.125, predictors=5, tested=2, power=p) for p in [0.8, 0.9]],
    )
    assert scenarios["scenarios"].n.tolist()[0] == 81
    assert scenarios["scenarios"].n.tolist()[1] > 81
    plot = oe.planning_plot(scenarios)
    assert plot.kind == "line" and plot.total_n == 2
    with pytest.raises(AnalysisError):
        oe.power_anova(0.3, groups=4, n=1)
    with pytest.raises(AnalysisError):
        oe.power_regression(1000, predictors=2, tested=2, n=20)
    with pytest.raises(AnalysisError):
        oe.power_logrank(1.0, power=0.8)
    with pytest.raises(AnalysisError):
        oe.power_cluster_mean(0.2, sd=1, cluster_size=5, icc=-0.1, n=12)


def test_new_routes_have_early_resource_rejection():
    data, weights, _ = spatial_data()
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError) as error:
            oe.sar_iv(data, "y", ["x"], key="key", instruments=["z", "z2"], spatial_weights=weights)
        assert error.value.code == "workspace_limit"


def test_complete_state_and_CPU_domain_under_unrelated_default_device():
    data = linear_data()
    result = oe.ols(data=data, y="y", x=["x", "z"])
    with torch.device("meta"):
        output = oe.wild_cluster_test(result, data=data, cluster="id", null={"x": 0.0}, reps=49,seed=np.int64(3))
    restored = oe.restore_summary(oe.summary_state(output))
    assert restored.attrs == output.attrs
    assert all(len(value) <= 400 for value in output["settings"].json)
    with pytest.raises(AnalysisError):
        oe.restore_summary(
            '{"schema":"openecon.summary.v1","title":null,"attrs":{"x":1e999},"tables":{}}'
        )
    with pytest.raises(AnalysisError):
        oe.restore_summary('{"schema":"other","attrs":{},"tables":{}}')
