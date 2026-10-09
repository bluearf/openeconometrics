"""Independent design, complete covariance, holdout leakage and state oracles."""

import copy
import itertools
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget

ATTRIBUTES = {"brand": ["a", "b"], "price": [10, 20, 30, 40], "speed": ["low", "high"],
              "quality": ["basic", "premium"]}


@pytest.fixture
def study():
    torch.set_num_threads(2)
    full = oe.conjoint_plan(ATTRIBUTES)
    plan = full["profiles"]
    training = plan.iloc[[i for i in range(len(plan)) if i % 4 != 0]].copy()
    held = plan.iloc[::4].copy()
    training_ids = set(training["profile_id"])
    rng = np.random.default_rng(288)
    responses, holdout = [], []
    for subject in ("person-z", "person-a", "person-m"):
        for card, brand, price, speed, quality in plan.itertuples(index=False, name=None):
            utility = 20 + 1.1*(brand == "a") - 0.02*(price-25)**2 + .5*(speed == "high") + .2*(quality == "premium")
            row = [subject, card, utility+rng.normal(scale=.2)]
            (responses if card in training_ids else holdout).append(row)
    responses = pd.DataFrame(responses, columns=["subject", "profile_id", "score"],
                             index=[f"row-{i}" for i in range(len(responses))])
    holdout = pd.DataFrame(holdout, columns=responses.columns)
    fit = oe.conjoint_fit(training, responses, ATTRIBUTES, factors={"price": "ideal"})
    return full, training, held, responses, holdout, fit


def oracle_design(plan):
    # Independent raw-basis effects coding, with last levels constrained by sums.
    return np.column_stack([np.ones(len(plan)), np.where(plan.brand == "a", 1., -1.),
                            plan.price.to_numpy(float), plan.price.to_numpy(float)**2,
                            np.where(plan.speed == "low", 1., -1.), np.where(plan.quality == "basic", 1., -1.)])


def test_full_cartesian_plan_retains_declared_order_labels_and_persistence():
    result = oe.conjoint_plan({"z": ["second", "first"], "a": [1, 2.5]})
    assert result["profiles"].iloc[:, 1:].values.tolist() == [list(v) for v in itertools.product(["second", "first"], [1, 2.5])]
    restored = oe.conjoint_load(oe.conjoint_save(result))
    assert restored.attrs["state"]["attribute_order"] == ["z", "a"]
    assert oe.conjoint_diagnostics(restored)["summary"].iloc[0]["rank"] == 3


def test_nist_published_resolution_iv_design_and_complete_two_way_aliases():
    # https://www.itl.nist.gov/div898/handbook/pri/section3/eqns/2to4m1.txt
    result = oe.conjoint_orthogonal({a: [-1, 1] for a in "ABCD"}, list("ABC"), {"D": list("ABC")})
    expected = [[-1,-1,-1,-1],[1,-1,-1,1],[-1,1,-1,1],[1,1,-1,-1],
                [-1,-1,1,1],[1,-1,1,-1],[-1,1,1,-1],[1,1,1,1]]
    assert result["profiles"].iloc[:, 1:].values.tolist() == expected
    audit = oe.conjoint_diagnostics(result)
    summary = audit["summary"].iloc[0]
    assert summary["rank"] == 5 and summary["between_factor_orthogonal"]
    assert audit["associations"].cramer_v.tolist() == [0.0]*6
    assert set(map(tuple, audit["aliases"].values.tolist())) == {("A*B", "C*D", 1), ("A*C", "B*D", 1), ("A*D", "B*C", 1)}


def test_signed_resolution_iii_alias_is_not_hidden_as_interaction_free():
    result = oe.conjoint_orthogonal({a: [-1, 1] for a in "ABC"}, ["A", "B"], {"C": ["A", "B"]}, signs={"C": -1})
    audit = oe.conjoint_diagnostics(result)
    assert ("C", "A*B", 1) in set(map(tuple, audit["aliases"].values.tolist()))
    assert audit.attrs["higher_order_aliases_not_excluded"]


@pytest.mark.parametrize("generators,signs", [({"C": []}, None), ({"C": ["A"]}, None),
    ({"C": ["A", "A"]}, None), ({"C": ["D"]}, None), ({}, None),
    ({"C": ["A", "B"]}, {"C": 0}), ({"C": ["A", "B"]}, {"A": 1})])
def test_invalid_fraction_generators_refuse(generators, signs):
    with pytest.raises(AnalysisError):
        oe.conjoint_orthogonal({a: [-1, 1] for a in "ABC"}, ["A", "B"], generators, signs=signs)


def test_full_individual_raw_covariance_se_df_p_ci_and_utilities_match_ols(study):
    _, training, _, responses, _, fit = study
    x = oracle_design(training)
    for subject in responses.subject.unique():
        y = responses[responses.subject == subject].set_index("profile_id").loc[training.profile_id, "score"].to_numpy(float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y-x@beta
        df = len(x)-x.shape[1]
        cov = np.linalg.inv(x.T@x)*(residual@residual/df)
        coefficients = fit["coefficients"].query("subject == @subject")
        saved_cov = fit["covariance"].query("subject == @subject").iloc[:, 2:].to_numpy(float)
        np.testing.assert_allclose(coefficients.estimate, beta, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(saved_cov, cov, rtol=1e-9, atol=1e-10)
        se = np.sqrt(np.diag(cov))
        np.testing.assert_allclose(coefficients.std_error, se, rtol=1e-9)
        assert coefficients.df.tolist() == [df]*len(beta)
        np.testing.assert_allclose(coefficients.t, beta/se, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(coefficients.p_value, 2*stats.t.sf(np.abs(beta/se), df), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(coefficients.ci_low, beta-stats.t.ppf(.975, df)*se, atol=1e-9)
        np.testing.assert_allclose(coefficients.ci_high, beta+stats.t.ppf(.975, df)*se, atol=1e-9)
        utility = fit["utilities"].query("subject == @subject")
        assert utility.query("attribute == 'brand'").estimate.sum() == pytest.approx(0, abs=1e-12)
        for price in ATTRIBUTES["price"]:
            vector = np.array([0, 0, price, price**2, 0, 0])
            row = utility[(utility.attribute == "price") & (utility.level == price)].iloc[0]
            assert row.estimate == pytest.approx(vector@beta, abs=1e-9)
            assert row.std_error == pytest.approx(np.sqrt(vector@cov@vector), rel=1e-9)
    assert fit.attrs["state"]["converged"]


def test_profile_key_join_and_original_row_identity_survive_reordering(study):
    _, training, _, responses, _, fit = study
    shuffled = responses.sample(frac=1, random_state=48)
    again = oe.conjoint_fit(training, shuffled, ATTRIBUTES, factors={"price": "ideal"})
    for output in ("coefficients", "covariance", "utilities"):
        a = fit[output].sort_values(list(fit[output].columns[:2])).reset_index(drop=True)
        b = again[output].sort_values(list(again[output].columns[:2])).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    for record in again.attrs["state"]["subjects"]:
        assert shuffled.iloc[record["sample_positions"]].index.tolist() == record["sample_labels"]
        assert shuffled.iloc[record["sample_positions"]].profile_id.tolist() == training.profile_id.tolist()


def test_predictions_mean_uncertainty_and_restore_use_full_covariance(study):
    _, training, held, responses, _, fit = study
    saved = oe.conjoint_load(oe.conjoint_save(fit))
    predicted = oe.conjoint_predict(saved, held, level=.9)["predictions"]
    x, new = oracle_design(training), oracle_design(held)
    for subject in responses.subject.unique():
        y = responses[responses.subject == subject].set_index("profile_id").loc[training.profile_id, "score"].to_numpy(float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        df = len(x)-x.shape[1]
        cov = np.linalg.inv(x.T@x)*np.sum((y-x@beta)**2)/df
        se = np.sqrt(np.einsum("ij,jk,ik->i", new, cov, new))
        rows = predicted.query("subject == @subject")
        np.testing.assert_allclose(rows.predicted, new@beta, atol=1e-9)
        np.testing.assert_allclose(rows.std_error, se, rtol=1e-9)
        np.testing.assert_allclose(rows.ci_low, new@beta-stats.t.ppf(.95, df)*se, atol=1e-9)


def test_holdout_correlations_and_sample_labels_match_independent_oracles(study):
    _, _, held, _, actual, fit = study
    output = oe.conjoint_holdout(fit, held, actual.sample(frac=1, random_state=24))
    detail = output["holdout_predictions"]
    for row in output["validation"].itertuples(index=False):
        frame = detail[detail.subject == row.subject]
        assert row.pearson == pytest.approx(stats.pearsonr(frame.observed, frame.predicted).statistic, abs=1e-12)
        assert row.kendall_tau_b == pytest.approx(stats.kendalltau(frame.observed, frame.predicted).statistic, abs=1e-12)
        assert row.rmse == pytest.approx(np.sqrt(np.mean((frame.observed-frame.predicted)**2)))
    assert fit.attrs["state"]["training_profile_ids"] == fit["training_plan"].profile_id.tolist()


def test_holdout_training_id_or_renamed_training_geometry_is_refused(study):
    _, training, _, responses, _, fit = study
    with pytest.raises(AnalysisError, match="Holdouts"):
        oe.conjoint_holdout(fit, training, responses)
    renamed = training.copy()
    renamed.profile_id = ["new-"+v for v in renamed.profile_id]
    with pytest.raises(AnalysisError) as exc:
        oe.conjoint_holdout(fit, renamed, responses)
    assert exc.value.code == "holdout_leakage"


def test_missing_drops_complete_subjects_only_and_records_every_exclusion(study):
    _, training, _, responses, _, _ = study
    incomplete = responses.copy()
    incomplete.loc[incomplete.index[0], "score"] = np.nan
    with pytest.raises(AnalysisError) as exc:
        oe.conjoint_fit(training, incomplete, ATTRIBUTES)
    assert exc.value.code == "incomplete_subject"
    result = oe.conjoint_fit(training, incomplete, ATTRIBUTES, missing="drop_subjects")
    assert result.attrs["n_subjects"] == 2
    assert result.attrs["state"]["dropped_subjects"] == [{"subject": "person-z", "reason": "incomplete/missing score"}]


@pytest.mark.parametrize("case", ["duplicate", "unknown_profile", "missing_subject", "infinite_score"])
def test_invalid_response_geometry_is_never_silently_repaired(study, case):
    _, training, _, responses, _, _ = study
    altered = responses.copy()
    if case == "duplicate":
        altered = pd.concat([altered, altered.iloc[:1]])
    elif case == "unknown_profile":
        altered.loc[altered.index[0], "profile_id"] = "absent"
    elif case == "missing_subject":
        altered.loc[altered.index[0], "subject"] = None
    else:
        altered.loc[altered.index[0], "score"] = np.inf
    with pytest.raises(AnalysisError):
        oe.conjoint_fit(training, altered, ATTRIBUTES, missing="drop_subjects")


def test_importance_is_average_individual_normalization_not_normalized_mean():
    plan = oe.conjoint_plan({"A": [-1, 1], "B": [-1, 1], "C": [-1, 1]})
    rows = []
    for subject, slopes in (("one", [10, 1, 1]), ("two", [-1, 4, 1])):
        for card, a, b, c in plan["profiles"].itertuples(index=False, name=None):
            rows.append([subject, card, 50+np.dot([a,b,c], slopes)])
    result = oe.conjoint_fit(plan, pd.DataFrame(rows, columns=["subject", "profile_id", "score"]))
    output = oe.conjoint_importance(result)
    expected = np.mean([[100*10/12,100/12,100/12],[100/6,100*4/6,100/6]], 0)
    np.testing.assert_allclose(output["group"].mean_individual_importance_percent, expected, atol=1e-12)
    assert not np.allclose(output["group"].mean_individual_importance_percent, output["group"].importance_of_mean_utilities_percent)
    assert output["individual"].groupby("subject").importance_percent.sum().tolist() == pytest.approx([100,100])


@pytest.mark.parametrize("method", ["first_choice", "btl", "logit"])
def test_preference_shares_match_declared_probability_algebra(study, method):
    _, _, held, _, _, fit = study
    options = {"temperature": .7} if method == "logit" else {}
    output = oe.conjoint_simulate(fit, held, method=method, **options)
    for subject in output["individual"].subject.unique():
        rows = output["individual"].query("subject == @subject")
        scores = rows.predicted_score.to_numpy()
        if method == "first_choice":
            expected = (scores == max(scores)).astype(float)
            expected /= expected.sum()
        elif method == "btl":
            expected = scores/scores.sum()
        else:
            expected = np.exp((scores-max(scores))/.7)
            expected /= expected.sum()
        np.testing.assert_allclose(rows.conditional_share, expected, atol=1e-14)
    np.testing.assert_allclose(output["group"].equal_subject_share,
                               output["individual"].groupby("profile_id", sort=False).conditional_share.mean(), atol=1e-14)


def test_constant_scores_ties_and_undefined_importance_are_explicit():
    plan = oe.conjoint_plan({"A": [-1,1], "B": [-1,1]})
    response = pd.DataFrame({"subject": [1]*4, "profile_id": plan["profiles"].profile_id, "score": [0.0]*4})
    fit = oe.conjoint_fit(plan, response)
    first = oe.conjoint_simulate(fit, plan)
    assert first["group"].equal_subject_share.tolist() == [.25]*4
    assert oe.conjoint_simulate(fit, plan, method="logit", temperature=1e-300)["group"].equal_subject_share.tolist() == [.25]*4
    with pytest.raises(AnalysisError) as exc:
        oe.conjoint_simulate(fit, plan, method="btl")
    assert exc.value.code == "nonpositive_btl_score"
    importance = oe.conjoint_importance(fit)
    assert not importance.attrs["complete_group_average_defined"]
    assert importance.attrs["state"]["undefined_subjects"] == [1]
    assert importance["group"].mean_individual_importance_percent.isna().all()


@pytest.mark.parametrize("kind", ["coefficient", "coding", "sample", "covariance"])
def test_integrity_blocks_mutated_fit_and_corrupt_artifacts(study, kind, tmp_path):
    _, _, held, _, _, fit = study
    saved = oe.conjoint_save(fit)
    corrupt = copy.deepcopy(saved)
    if kind == "coefficient":
        corrupt["payload"]["tables"]["coefficients"]["data"][0][2] += 1
    elif kind == "coding":
        corrupt["payload"]["attrs"]["state"]["anchors"]["price"]["center"] += 1
    elif kind == "sample":
        corrupt["payload"]["attrs"]["state"]["subjects"][0]["sample_positions"][0] += 1
    else:
        corrupt["payload"]["attrs"]["state"]["subjects"][0]["covariance_scaled"][0][0] += 1
    with pytest.raises(AnalysisError):
        oe.conjoint_load(corrupt)
    file = tmp_path/"complete.json"
    oe.conjoint_save(fit, file)
    assert json.loads(file.read_text()) == saved
    mutated = oe.conjoint_load(saved)
    mutated["coefficients"].iloc[0,2] += 1
    with pytest.raises(AnalysisError):
        oe.conjoint_predict(mutated, held)


def test_rank_df_and_declared_level_errors_are_explicit(study):
    _, training, held, responses, _, fit = study
    aliased = training.copy()
    aliased["quality"] = aliased["speed"].map({"low":"basic", "high":"premium"})
    audit = oe.conjoint_diagnostics(aliased, ATTRIBUTES)
    assert not audit.attrs["main_effect_identified"]
    with pytest.raises(AnalysisError) as exc:
        oe.conjoint_fit(aliased, responses, ATTRIBUTES)
    assert exc.value.code == "unidentified_design"
    small = oe.conjoint_orthogonal({a:[0,1] for a in "ABC"}, ["A","B"], {"C":["A","B"]})
    data = pd.DataFrame({"subject":[1]*4,"profile_id":small["profiles"].profile_id,"score":[1.,2.,4.,5.]})
    with pytest.raises(AnalysisError) as exc:
        oe.conjoint_fit(small, data)
    assert exc.value.code == "insufficient_df"
    held = held.copy()
    held.iloc[0, held.columns.get_loc("brand")] = "unknown"
    with pytest.raises(AnalysisError) as exc:
        oe.conjoint_predict(fit, held)
    assert exc.value.code == "unknown_level"


def test_cpu_dtype_and_workspace_work_limits_are_real(study):
    _, training, held, responses, _, fit = study
    with pytest.raises(AnalysisError) as exc:
        oe.conjoint_fit(training, responses, ATTRIBUTES, max_work=1)
    assert exc.value.code == "work_limit"
    with use_workspace_budget(1), pytest.raises(AnalysisError) as exc:
        oe.conjoint_orthogonal({f"A{i}":[0,1] for i in range(12)}, [f"A{i}" for i in range(12)], {})
    assert exc.value.code == "workspace_limit"
    with pytest.raises(AnalysisError) as exc:
        oe.conjoint_plan({f"A{i}":[0,1] for i in range(16)})
    assert exc.value.code == "resource_limit"
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        predicted = oe.conjoint_predict(fit, held)
        assert predicted["predictions"].predicted.dtype == np.float64
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("options", [{"device":"mps"}, {"device":"cuda"}, {"weights":"w"}])
def test_unsupported_domains_do_not_fall_back(study, options):
    _, _, held, _, _, fit = study
    with pytest.raises(AnalysisError):
        oe.conjoint_predict(fit, held, **options)


def test_manifest_is_lightweight_and_all_eight_are_public():
    subprocess.run([sys.executable,"-c", "import sys; from openecon.econometrics.conjoint import EXPORTS, ESTIMATORS; assert ESTIMATORS == (); assert len(EXPORTS)==10; assert 'torch' not in sys.modules"], check=True)
    assert all(callable(getattr(oe, name)) for name in (
        "conjoint_plan", "conjoint_orthogonal", "conjoint_diagnostics", "conjoint_fit", "conjoint_predict",
        "conjoint_holdout", "conjoint_importance", "conjoint_simulate", "conjoint_save", "conjoint_load"))


@pytest.mark.parametrize("procedure", ["plan", "orthogonal", "diagnostics", "fit", "predict", "holdout", "importance", "simulate"])
def test_every_complete_artifact_restores_all_tables_and_latex(study, procedure):
    full, _, held, _, holdout, fit = study
    binary = oe.conjoint_orthogonal({a:[-1,1] for a in "ABCD"}, list("ABC"), {"D":list("ABC")})
    output = {"plan":lambda:full, "orthogonal":lambda:binary, "diagnostics":lambda:oe.conjoint_diagnostics(binary),
              "fit":lambda:fit, "predict":lambda:oe.conjoint_predict(fit, held),
              "holdout":lambda:oe.conjoint_holdout(fit, held, holdout),
              "importance":lambda:oe.conjoint_importance(fit), "simulate":lambda:oe.conjoint_simulate(fit, held)}[procedure]()
    saved = oe.conjoint_save(output)
    restored = oe.conjoint_load(saved)
    assert oe.conjoint_save(restored) == saved
    assert list(restored) == list(output)
    for name in output:
        pd.testing.assert_frame_equal(restored[name], output[name], check_exact=True)
    assert "\\begin{tabular}" in restored.to_latex()


@pytest.mark.parametrize("mode", ["linear", "ideal", "discrete"])
def test_numeric_model_specs_and_origin_invariance_against_raw_ols(study, mode):
    _, training, _, responses, _, _ = study
    result = oe.conjoint_fit(training, responses, ATTRIBUTES, factors={"price":mode})
    subject = responses.subject.iloc[0]
    x = oracle_design(training)
    if mode == "linear":
        x = np.delete(x, 3, axis=1)
    elif mode == "discrete":
        level = training.price.to_numpy()
        dummies = np.column_stack([np.where(level == v, 1., np.where(level == 40, -1., 0.)) for v in (10,20,30)])
        x = np.column_stack([x[:,:2], dummies, x[:,4:]])
    y = responses[responses.subject == subject].score.to_numpy(float)
    np.testing.assert_allclose(result["coefficients"].query("subject == @subject").estimate,
                               np.linalg.lstsq(x, y, rcond=None)[0], atol=1e-9)
    shifted = responses.copy()
    shifted.score += 1e9
    large = oe.conjoint_fit(training, shifted, ATTRIBUTES, factors={"price":mode})
    np.testing.assert_allclose(result["coefficients"].query("subject == @subject").estimate[1:],
                               large["coefficients"].query("subject == @subject").estimate[1:], atol=1e-6)
    np.testing.assert_allclose(result["covariance"].query("subject == @subject").iloc[:,2:].to_numpy(float),
                               large["covariance"].query("subject == @subject").iloc[:,2:].to_numpy(float), rtol=2e-5, atol=1e-7)


def test_holdout_undefined_correlations_and_absent_subjects_are_explicit(study):
    _, _, held, _, responses, fit = study
    constant = responses.copy()
    constant.score = 1.
    output = oe.conjoint_holdout(fit, held, constant)
    assert output["validation"].pearson_undefined.all() and output["validation"].tau_undefined.all()
    reduced = constant[constant.subject != "person-z"]
    with pytest.raises(AnalysisError):
        oe.conjoint_holdout(fit, held, reduced)
    dropped = oe.conjoint_holdout(fit, held, reduced, missing="drop_subjects")
    assert dropped.attrs["state"]["dropped_subjects"] == [{"subject":"person-z","reason":"no holdout responses"}]


def test_numeric_profile_ids_keep_column_types_when_scores_are_floats():
    plan = oe.conjoint_plan({"A":[0,1], "B":[0,1]})["profiles"].copy()
    plan.profile_id = [1,2,3,4]
    responses = pd.DataFrame({"subject":[1,1,1,1],"profile_id":[4,2,1,3],"score":[4.,1.,2.,3.]})
    output = oe.conjoint_fit(plan, responses, {"A":[0,1],"B":[0,1]})
    assert output.attrs["state"]["subjects"][0]["sample_positions"] == [2,1,3,0]
