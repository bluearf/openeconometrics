"""Independent raw-basis, reference-estimator, jackknife and saved-target oracles."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget


@pytest.fixture
def study():
    attributes = {"brand": ["a", "b"], "price": [10, 20, 30, 40], "quality": ["basic", "premium"]}
    plan = oe.conjoint_plan(attributes)
    frame = plan["profiles"]
    # Independently construct a raw polynomial basis, without implementation coding.
    x = np.column_stack([np.ones(len(frame)), np.where(frame.brand == "a", 1., -1.),
                         frame.price.to_numpy(float), frame.price.to_numpy(float)**2,
                         np.where(frame.quality == "basic", 1., -1.)])
    rng = np.random.default_rng(785)
    responses = []
    for subject in range(12):
        beta = [12+rng.normal(), .7+rng.normal(scale=.2), -.03+rng.normal(scale=.01),
                -.002+rng.normal(scale=.0003), .3+rng.normal(scale=.1)]
        errors = rng.normal(size=len(frame))*(.15+.01*frame.price.to_numpy(float))
        errors += np.repeat(rng.normal(scale=.3, size=8), 2)
        for i, (card, y) in enumerate(zip(frame.profile_id, x@beta+errors)):
            responses.append([subject, card, y, i//2])
    data = pd.DataFrame(responses, columns=["subject", "profile_id", "score", "session"],
                        index=[f"original-{i}" for i in range(len(responses))])
    return plan, data, x


def fit(study, method="hc0", data=None):
    plan, responses, _ = study
    return oe.conjoint_fit(plan, responses if data is None else data, factors={"price": "ideal"},
                           covariance=method, **({"cluster": "session"} if method.startswith("cr") else {}))


def raw_cov(result, subject):
    return result["covariance"].query("subject == @subject").iloc[:, 2:].to_numpy(float)


@pytest.mark.parametrize("method", ["hc0", "hc1", "hc2", "hc3", "cr0", "cr1"])
def test_complete_covariance_inference_utilities_and_predictions_match_independent_references(study, method):
    plan, responses, x = study
    output = fit(study, method)
    predicted = oe.conjoint_predict(output, plan, level=.9)["predictions"]
    bread = np.linalg.inv(x.T@x)
    h = np.einsum("ij,jk,ik->i", x, bread, x)
    for subject in responses.subject.unique():
        sample = responses[responses.subject == subject]
        y = sample.score.to_numpy(float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        e = y-x@beta
        if method.startswith("cr"):
            scores = np.stack([(x*e[:, None])[sample.session.to_numpy() == g].sum(0)
                               for g in sorted(sample.session.unique())])
            g = len(scores)
            multiplier = g/(g-1)*(len(x)-1)/(len(x)-x.shape[1]) if method == "cr1" else 1
            df = g-1
            ref = sm.OLS(y, x).fit().get_robustcov_results(cov_type="cluster", groups=sample.session,
                                                         use_correction=method == "cr1", use_t=True)
        else:
            scale = np.ones(len(x))
            if method == "hc1":
                scale *= len(x)/(len(x)-x.shape[1])
            elif method == "hc2":
                scale /= 1-h
            elif method == "hc3":
                scale /= (1-h)**2
            scores, multiplier, df = x*(e*np.sqrt(scale))[:, None], 1, len(x)-x.shape[1]
            ref = sm.OLS(y, x).fit().get_robustcov_results(cov_type=method.upper(), use_t=True)
        expected = multiplier*bread@scores.T@scores@bread
        actual = raw_cov(output, subject)
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(actual, ref.cov_params(), rtol=2e-8, atol=1e-10)
        coefficients = output["coefficients"].query("subject == @subject")
        se = np.sqrt(np.diag(expected))
        np.testing.assert_allclose(coefficients.estimate, beta, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(coefficients.std_error, se, rtol=2e-8, atol=1e-10)
        assert coefficients.df.tolist() == [df]*len(beta)
        np.testing.assert_allclose(coefficients.p_value, 2*stats.t.sf(np.abs(beta/se), df), rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(coefficients.ci_high, beta+stats.t.ppf(.975, df)*se, rtol=2e-8, atol=1e-9)
        for value in plan.attrs["state"]["attributes"]["price"]:
            vector = np.array([0, 0, value, value**2, 0])
            row = output["utilities"].query("subject == @subject and attribute == 'price' and level == @value").iloc[0]
            assert row.estimate == pytest.approx(vector@beta, rel=2e-8)
            assert row.std_error == pytest.approx(np.sqrt(vector@expected@vector), rel=2e-8)
        predictions = predicted.query("subject == @subject")
        prediction_se = np.sqrt(np.einsum("ij,jk,ik->i", x, expected, x))
        np.testing.assert_allclose(predictions.std_error, prediction_se, rtol=2e-8)
        assert predictions.df.tolist() == [df]*len(x)
        np.testing.assert_allclose(predictions.ci_low, x@beta-stats.t.ppf(.95, df)*prediction_se, rtol=2e-8)
        if method.startswith("cr"):
            record = output.attrs["state"]["subjects"][int(subject)]
            assert record["cluster_ids"] == sample.session.tolist()
            assert record["n_clusters"] == g
            assert record["covariance_multiplier"] == pytest.approx(multiplier)


def test_hc3_matches_all_delete_one_profile_coefficient_differences(study):
    _, data, x = study
    result = fit(study, "hc3")
    y = data[data.subject == 0].score.to_numpy(float)
    full = np.linalg.lstsq(x, y, rcond=None)[0]
    changes = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        changes.append(np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]-full)
    changes = np.array(changes)
    np.testing.assert_allclose(raw_cov(result, 0), changes.T@changes, rtol=2e-8, atol=1e-10)


@pytest.mark.parametrize("method", ["hc0", "hc1", "hc2", "hc3", "cr0", "cr1"])
def test_keyed_clusters_and_complete_saved_replay_are_order_invariant(study, method):
    plan, responses, _ = study
    shuffled = responses.sample(frac=1, random_state=792)
    result = fit(study, method, shuffled)
    baseline = fit(study, method)
    for name in ("coefficients", "covariance", "utilities"):
        pd.testing.assert_frame_equal(result[name].sort_values(list(result[name].columns[:2])).reset_index(drop=True),
                                      baseline[name].sort_values(list(baseline[name].columns[:2])).reset_index(drop=True),
                                      check_exact=True)
    restored = oe.conjoint_load(oe.conjoint_save(result))
    assert oe.conjoint_save(restored) == oe.conjoint_save(result)
    assert restored.to_latex() == result.to_latex()
    pd.testing.assert_frame_equal(restored["diagnostics"], result["diagnostics"], check_exact=True)
    assert oe.conjoint_save(oe.conjoint_predict(restored, plan)) == oe.conjoint_save(oe.conjoint_predict(result, plan))
    for record in result.attrs["state"]["subjects"]:
        assert shuffled.iloc[record["sample_positions"]].index.tolist() == record["sample_labels"]
        if method.startswith("cr"):
            assert shuffled.iloc[record["sample_positions"]].session.tolist() == record["cluster_ids"]


@pytest.mark.parametrize("method", ["hc2", "hc3"])
def test_unit_leverage_is_refused_without_deleting_profiles(method):
    attributes = {"a": [0, 1], "b": [0, 1]}
    plan = pd.DataFrame([["u",0,0],["v",0,0],["w",1,0],["z",0,1]], columns=["profile_id", "a", "b"])
    responses = pd.DataFrame({"subject": [1]*4, "profile_id": plan.profile_id, "score": [0., 1., 2., 3.]})
    with pytest.raises(AnalysisError) as error:
        oe.conjoint_fit(plan, responses, attributes, covariance=method)
    assert error.value.code == "unit_leverage"


@pytest.mark.parametrize("case", ["no_column", "same_column", "one_cluster", "missing", "boolean", "hc_cluster"])
def test_invalid_cluster_domains_fail_explicitly(study, case):
    plan, data, _ = study
    data = data.copy()
    options = {"covariance": "cr1", "cluster": "session"}
    if case == "no_column":
        options["cluster"] = "absent"
    elif case == "same_column":
        options["cluster"] = "subject"
    elif case == "one_cluster":
        data.session = 0
    elif case == "missing":
        data.loc[data.index[0], "session"] = np.nan
    elif case == "boolean":
        data.session = True
    else:
        options["covariance"] = "hc0"
    with pytest.raises(AnalysisError):
        oe.conjoint_fit(plan, data, factors={"price": "ideal"}, **options)


def test_cluster_counts_df_typed_labels_and_whole_subject_exclusion(study):
    plan, data, _ = study
    data = data.copy()
    data.session = data.session.astype(object)
    # String and numeric cluster labels remain distinct; 4 clusters for subject 0.
    labels = [1, "1", 2, "2"]
    data.loc[data.subject == 0, "session"] = [labels[i % 4] for i in range(len(plan["profiles"]))]
    data.loc[data.subject == 1, "session"] = 0  # unusable after explicit whole-subject drop
    data.loc[data[data.subject == 1].index[0], "score"] = np.nan
    result = oe.conjoint_fit(plan, data, factors={"price": "ideal"}, covariance="cr0", cluster="session",
                            missing="drop_subjects")
    assert result.attrs["state"]["dropped_subjects"][0]["subject"] == 1
    record = result.attrs["state"]["subjects"][0]
    assert record["n_clusters"] == 4 and record["df"] == 3
    assert result["coefficients"].query("subject == 0").df.unique().tolist() == [3]


@pytest.mark.parametrize("method", ["nonrobust", "hc3", "cr1"])
def test_group_mean_full_covariance_utilities_and_no_noise_double_counting(study, method):
    fitted = fit(study, method)
    output = oe.conjoint_group_mean(fitted, level=.9)
    params = np.stack([g.estimate.to_numpy(float) for _, g in fitted["coefficients"].groupby("subject", sort=False)])
    mean, m = params.mean(0), len(params)
    covariance = np.cov(params, rowvar=False, ddof=1)/m
    np.testing.assert_allclose(output["coefficients"].estimate, mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(output["covariance"].iloc[:, 1:].to_numpy(float), covariance, rtol=1e-12, atol=1e-12)
    assert output["coefficients"].df.tolist() == [m-1]*params.shape[1]
    se = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(output["coefficients"].ci_low, mean-stats.t.ppf(.95, m-1)*se)
    np.testing.assert_allclose(output["coefficients"].p_value, 2*stats.t.sf(np.abs(mean/se), m-1))
    level_map = []
    for row in output["utilities"].itertuples(index=False):
        vector = np.zeros(params.shape[1])
        if row.attribute == "brand":
            vector[1] = 1 if row.level == "a" else -1
        elif row.attribute == "quality":
            vector[4] = 1 if row.level == "basic" else -1
        else:
            vector[2:4] = [row.level, row.level**2]
        level_map.append(vector)
    level_map = np.array(level_map)
    np.testing.assert_allclose(output["utilities"].estimate, level_map@mean, atol=1e-12)
    np.testing.assert_allclose(output["utility_covariance"].iloc[:, 1:].to_numpy(float),
                               level_map@covariance@level_map.T, rtol=1e-12, atol=1e-12)
    assert oe.conjoint_save(oe.conjoint_load(oe.conjoint_save(output))) == oe.conjoint_save(output)


@pytest.mark.parametrize("method", ["nonrobust", "hc0", "hc1", "hc2", "hc3", "cr0", "cr1"])
def test_saved_contrasts_full_target_covariance_nonzero_null_and_joint_distribution(study, method):
    fitted = fit(study, method)
    terms = fitted.attrs["state"]["terms"]
    contrasts = {"price shift": {terms[2]: 10, terms[3]: 300}, "brand and quality": {terms[1]: 2, terms[4]: -.5}}
    null = {"price shift": -.1, "brand and quality": .7}
    result = oe.conjoint_contrast(fitted, contrasts, null=null, level=.9)
    mapping = np.array([[0, 0, 10, 300, 0], [0, 2, 0, 0, -.5]])
    for subject, rows in fitted["coefficients"].groupby("subject", sort=False):
        beta, covariance = rows.estimate.to_numpy(float), raw_cov(fitted, subject)
        target, expected_cov = mapping@beta, mapping@covariance@mapping.T
        actual = result["contrasts"].query("subject == @subject")
        np.testing.assert_allclose(actual.estimate, target, rtol=2e-9)
        np.testing.assert_allclose(result["covariance"].query("subject == @subject").iloc[:,2:].to_numpy(float),
                                   expected_cov, rtol=2e-9, atol=1e-11)
        se, df = np.sqrt(np.diag(expected_cov)), int(rows.df.iloc[0])
        delta = target-np.array([-.1, .7])
        np.testing.assert_allclose(actual.p_value, 2*stats.t.sf(np.abs(delta/se), df), rtol=2e-9)
        np.testing.assert_allclose(actual.ci_high, target+stats.t.ppf(.95, df)*se, rtol=2e-9)
        wald = delta@np.linalg.solve(expected_cov, delta)
        joint = result["joint"].query("subject == @subject").iloc[0]
        if method == "nonrobust":
            assert joint.distribution == "F" and joint.df2 == df
            assert joint.statistic == pytest.approx(wald/2, rel=2e-9)
            assert joint.p_value == pytest.approx(stats.f.sf(wald/2, 2, df), rel=2e-9)
        else:
            assert joint.distribution == "chi2" and pd.isna(joint.df2)
            assert joint.statistic == pytest.approx(wald, rel=2e-9)
            assert joint.p_value == pytest.approx(stats.chi2.sf(wald, 2), rel=2e-9)
    assert oe.conjoint_save(oe.conjoint_load(oe.conjoint_save(result))) == oe.conjoint_save(result)


@pytest.mark.parametrize("contrasts,null", [({}, None), ({"zero": {"_cons":0}}, None),
    ({"a":{"_cons":1}, "b":{"_cons":2}}, None), ({"a":{"unknown":1}}, None),
    ({"a":{"_cons":float("inf")}}, None), ({"a":{"_cons":True}}, None),
    ({"a":{"_cons":1}}, {"other":0}), ({"a":{"_cons":1}}, {"a":float("nan")}),
    ({"subject":{"_cons":1}}, None), ({"contrast":{"_cons":1}}, None)])
def test_invalid_or_unidentified_contrast_is_not_repaired(study, contrasts, null):
    with pytest.raises(AnalysisError):
        oe.conjoint_contrast(fit(study), contrasts, null=null)


def test_zero_variance_contrast_refuses_joint_test_instead_of_pseudoinverse(study):
    plan, data, _ = study
    data = data.copy()
    data.score = 0.0
    fitted = fit(study, "hc0", data)
    with pytest.raises(AnalysisError) as error:
        oe.conjoint_contrast(fitted, {"intercept":{"_cons":1}})
    assert error.value.code == "singular_target_covariance"
    grouped = oe.conjoint_group_mean(fitted)
    assert grouped["coefficients"].std_error.eq(0).all()
    assert grouped["coefficients"].p_value.isna().all()


def test_new_domains_have_real_resource_integrity_and_subject_guards(study):
    fitted = fit(study)
    plan, data, _ = study
    many = pd.concat([data[data.subject == 0].assign(subject=i) for i in range(128)], ignore_index=True)
    many_fitted = oe.conjoint_fit(plan, many, factors={"price":"ideal"}, covariance="hc0")
    for call in [lambda **kw: oe.conjoint_group_mean(fitted, **kw),
                 lambda **kw: oe.conjoint_contrast(fitted, {"intercept":{"_cons":1}}, **kw)]:
        for options, code in [({"max_work":1}, "work_limit"), ({"device":"mps"}, "unsupported_device"),
                              ({"weights":[1]}, "unsupported_weights")]:
            with pytest.raises(AnalysisError) as error:
                call(**options)
            assert error.value.code == code
    for call in [lambda: oe.conjoint_group_mean(many_fitted),
                 lambda: oe.conjoint_contrast(many_fitted, {"intercept":{"_cons":1}})]:
        with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == "workspace_limit"
    single = oe.conjoint_fit(plan, data[data.subject == 0], factors={"price":"ideal"})
    with pytest.raises(AnalysisError) as error:
        oe.conjoint_group_mean(single)
    assert error.value.code == "insufficient_subjects"
    corrupted = copy.deepcopy(fitted)
    corrupted.attrs["state"]["subjects"][0]["covariance_scaled"][0][0] += 1
    with pytest.raises(AnalysisError) as error:
        oe.conjoint_contrast(corrupted, {"intercept":{"_cons":1}})
    assert error.value.code == "invalid_result"


def test_replayed_new_targets_do_not_refit_or_recompute_fit_covariance(study, monkeypatch):
    import openecon.econometrics.conjoint.model as model
    fitted = fit(study, "cr1")
    restored = oe.conjoint_load(oe.conjoint_save(fitted))
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved postestimation must not fit the original equations.")
    monkeypatch.setattr(model, "individual_covariance", forbidden)
    monkeypatch.setattr(model.torch.linalg, "qr", forbidden)
    assert len(oe.conjoint_group_mean(restored)["coefficients"]) == 5
    assert len(oe.conjoint_contrast(restored, {"intercept":{"_cons":1}})["joint"]) == 12
    assert len(oe.conjoint_predict(restored, study[0])["predictions"]) == 192


def test_actual_historical_v1_native_artifact_remains_readable_without_rewriting_its_checksum():
    # Exact conjoint_fit.json member of the committed 2026-10-07 native archive.
    path = Path(__file__).parent/"fixtures/conjoint-legacy-v1.json"
    import json
    original = json.loads(path.read_text())
    restored = oe.conjoint_load(path)
    assert oe.conjoint_save(restored) == original
    assert restored.attrs["state"]["covariance"] == "nonrobust"
