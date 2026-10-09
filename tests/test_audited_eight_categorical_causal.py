"""Independent geometry/inference and full-file replay for three native cores."""

from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, chi2, norm, t
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


EXAMPLE = Path(__file__).resolve().parents[1] / "docs/examples/audited_eight_categorical_causal.py"


@pytest.fixture(scope="module")
def audited():
    return runpy.run_path(str(EXAMPLE))["analyze"]()


def _raw_design(frame):
    return np.column_stack([
        np.ones(len(frame)), np.where(frame.brand == "a", 1., -1.),
        frame.price.astype(float), frame.price.astype(float)**2,
        np.where(frame.quality == "basic", 1., -1.),
    ])


@pytest.mark.parametrize("family", ["categorical", "conjoint", "causal_design"])
def test_every_complete_typed_artifact_and_latex_reopens(tmp_path, audited, family):
    results, _ = audited
    for name, output in results.items():
        if not name.startswith(family+"/"):
            continue
        path = tmp_path / (name.split("/")[1]+".json")
        if family == "categorical":
            path.write_text(oe.summary_state(output))
            restored = oe.restore_summary(path.read_text())
            assert oe.summary_state(restored) == path.read_text()
        else:
            save, load = (oe.conjoint_save, oe.conjoint_load) if family == "conjoint" else (oe.causal_design_save, oe.causal_design_load)
            saved = save(output, path)
            restored = load(path)
            assert save(restored) == saved == json.loads(path.read_text())
        assert restored.attrs == output.attrs
        assert list(restored) == list(output)
        for key in output:
            pd.testing.assert_frame_equal(restored[key], output[key], check_exact=True)
        tex = tmp_path / (path.stem+".tex")
        tex.write_text(output.to_latex())
        assert restored.to_latex() == tex.read_text()


def test_nominal_dummy_ols_original_positions_and_outcome_free_projection(audited):
    results, inputs = audited
    output = results["categorical/nominal"]
    frame = inputs["people"].iloc[output.attrs["sample_positions"]]
    x = np.column_stack([
        np.ones(len(frame)), frame.a.eq("green"), frame.a.eq("blue"),
        frame.b.eq(1), frame.b.eq(2), frame.x,
    ]).astype(float)
    expected = x @ np.linalg.lstsq(x, frame.y, rcond=None)[0]
    # The native ALS stopping criterion concerns loss change, not exact OLS
    # coefficient equality; use the independently established solve tolerance.
    np.testing.assert_allclose(output["fitted"].fitted, expected, atol=3e-7, rtol=1e-7)
    assert 5 not in output.attrs["sample_positions"]
    query = inputs["people"].copy()
    query["y"] = np.arange(len(query))*999.
    restored = oe.restore_summary(oe.summary_state(output))
    projection = oe.catreg_predict(restored, query, missing="drop")
    pd.testing.assert_frame_equal(projection["predictions"], results["categorical/nominal_projection"]["predictions"], check_exact=True)
    assert output.attrs["inference"].startswith("descriptive")
    assert "std_error" not in output["coefficients"]


def test_ordinal_maps_are_monotone_and_frequency_normalized(audited):
    results, _ = audited
    output = results["categorical/ordinal"]
    mapping = output["quantifications"].query("variable == 'b'")
    q, counts = mapping.quantification.to_numpy(), mapping["count"].to_numpy()
    assert np.all(np.diff(q) >= -1e-12)
    assert abs(np.average(q, weights=counts)) < 1e-12
    assert abs(np.average(q*q, weights=counts)-1) < 1e-12
    trace = output["iterations"]
    for _, rows in trace.groupby("start", sort=False):
        assert np.all(np.diff(rows.objective) <= 1e-10)
    assert output.attrs["converged"] is True


def test_catpca_transformed_svd_and_saved_projection_geometry(audited):
    results, _ = audited
    output = results["categorical/catpca"]
    q = output["transformed"].drop(columns="source_position").to_numpy()
    _, singular, _ = np.linalg.svd(q, full_matrices=False)
    expected_values = singular**2/len(q)
    np.testing.assert_allclose(output["eigenvalues"].eigenvalue, expected_values, atol=1e-10)
    score = output["scores"].drop(columns="source_position").to_numpy()
    np.testing.assert_allclose(score.T@score/len(q), np.eye(2), atol=1e-10)
    np.testing.assert_allclose(score.mean(0), 0., atol=1e-12)
    reconstruction = score@output["loadings"].to_numpy().T
    np.testing.assert_allclose(np.square(q-reconstruction).sum()/len(q), output["fit"].reconstruction_loss.iloc[0], atol=1e-10)
    np.testing.assert_allclose(results["categorical/catpca_projection"]["scores"].drop(columns="source_position"), score, atol=1e-10)


def test_overals_complete_maps_reconstruct_each_set_and_actual_loss(audited):
    results, _ = audited
    output = results["categorical/overals"]
    metadata = output.attrs
    contributions = {}
    for name, codes, quantification, loading in zip(
        metadata["variables"], metadata["codes"], metadata["quantifications"], metadata["loadings"], strict=True,
    ):
        contributions[name] = np.asarray(quantification)[codes]@np.asarray(loading)
    scores = output["object_scores"].to_numpy()
    np.testing.assert_allclose(scores.T@scores/len(scores), np.eye(2), atol=1e-10)
    losses = []
    for number, variables in enumerate(metadata["sets"]):
        expected = sum(contributions[name] for name in variables)
        saved = output["set_scores"].query("set == @number")[["dimension_1", "dimension_2"]].to_numpy()
        np.testing.assert_allclose(saved, expected, atol=1e-12)
        losses.append(np.square(scores-expected).sum()/len(scores))
    assert np.isclose(np.mean(losses), metadata["objective"], atol=1e-12)
    assert metadata["global_optimum"] is False


def test_nonmetric_saved_pairs_certify_monotone_disparities_and_stress(audited):
    results, _ = audited
    output = results["categorical/nonmetric_mds"]
    pairs = output["pairs"]
    ranked = pairs.sort_values("dissimilarity")
    assert np.all(np.diff(ranked.disparity) >= -1e-12)
    assert all(rows.disparity.nunique() == 1 for _, rows in pairs.groupby("dissimilarity"))
    residual = pairs.distance.to_numpy()-pairs.disparity.to_numpy()
    np.testing.assert_allclose(pairs.squared_residual, residual**2, atol=1e-12)
    assert np.isclose(np.square(pairs.disparity).sum(), len(pairs), atol=1e-10)
    assert np.isclose(np.square(residual).sum()/len(pairs), output.attrs["normalized_stress"], atol=1e-12)
    assert np.isclose(np.sqrt(np.square(residual).sum()/np.square(pairs.distance).sum()), output.attrs["stress_1"], atol=1e-12)


def test_structural_zero_ipf_information_and_nested_likelihood(audited):
    results, _ = audited
    small, full, compared = [results["categorical/"+name] for name in ["ipf", "general_ml", "nested_lr"]]
    state = small.attrs["state"]
    x, mean = np.asarray(state["design"]), np.asarray(state["fitted_active"])
    information = x.T@(mean[:, None]*x)
    np.testing.assert_allclose(state["information"], information, atol=1e-10)
    np.testing.assert_allclose(state["covariance"], np.linalg.inv(information), atol=1e-10)
    assert state["df"] == 11-np.linalg.matrix_rank(x) == 2
    cells = small["cells"]
    impossible = cells.loc[cells.structural_zero]
    assert len(impossible) == 1 and impossible.fitted.iloc[0] == 0
    assert impossible[["fitted_se", "fitted_ci_lower", "fitted_ci_upper"]].isna().all().all()
    for margin in [["a", "b"], ["b", "c"]]:
        totals = cells.groupby(margin, sort=False)[["observed", "fitted"]].sum()
        np.testing.assert_allclose(totals.fitted, totals.observed, atol=1e-7)
    statistic = 2*(full.attrs["log_likelihood"]-small.attrs["log_likelihood"])
    assert compared.attrs["df"] == 2
    assert np.isclose(compared.attrs["statistic"], statistic, atol=1e-10)
    assert np.isclose(compared.attrs["p_value"], chi2.sf(statistic, 2), atol=1e-12)
    assert full.attrs["df_resid"] == 0
    assert full["goodness_of_fit"].p_value.isna().all()


def test_conjoint_raw_ols_full_covariance_and_heldout_mean_ci(audited):
    results, inputs = audited
    fit = results["conjoint/fit"]
    plan, query = inputs["conjoint_training"], inputs["conjoint_holdout"]
    x, a = _raw_design(plan), _raw_design(query)
    df = len(x)-x.shape[1]
    coefficients, predictions, covariance = fit["coefficients"], results["conjoint/predictions"]["predictions"], fit["covariance"]
    for subject in fit.attrs["state"]["subjects"]:
        name = subject["subject"]
        response = inputs["conjoint_responses"]
        score = response.loc[response.subject.eq(name)].set_index("profile_id").loc[plan.profile_id, "score"].to_numpy()
        beta = np.linalg.lstsq(x, score, rcond=None)[0]
        sigma2 = np.square(score-x@beta).sum()/df
        v = sigma2*np.linalg.inv(x.T@x)
        np.testing.assert_allclose(coefficients.query("subject == @name").estimate, beta, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(covariance.query("subject == @name").drop(columns=["subject", "term"]), v, atol=1e-10)
        predicted = predictions.query("subject == @name").set_index("profile_id").loc[query.profile_id]
        mean, se = a@beta, np.sqrt(np.einsum("ij,jk,ik->i", a, v, a))
        np.testing.assert_allclose(predicted.predicted, mean, atol=1e-10)
        np.testing.assert_allclose(predicted.std_error, se, atol=1e-10)
        np.testing.assert_allclose(predicted.ci_low, mean-t.ppf(.975, df)*se, atol=1e-10)
        assert np.count_nonzero(v-np.diag(v.diagonal())) > 0


def test_conjoint_holdout_does_not_refit_and_renamed_training_is_refused(audited):
    results, inputs = audited
    saved = oe.conjoint_save(results["conjoint/fit"])
    fit = oe.conjoint_load(saved)
    altered = inputs["conjoint_validation"].copy()
    altered["score"] += 1000
    validation = oe.conjoint_holdout(fit, inputs["conjoint_holdout"], altered)
    assert validation["validation"].mae.min() > 900
    pd.testing.assert_frame_equal(validation["holdout_predictions"].drop(columns="observed"),
                                  results["conjoint/holdout"]["holdout_predictions"].drop(columns="observed"), check_exact=True)
    assert oe.conjoint_save(fit) == saved
    leaked = inputs["conjoint_training"].iloc[:1].copy()
    leaked["profile_id"] = "renamed-training-profile"
    with pytest.raises(AnalysisError):
        oe.conjoint_holdout(fit, leaked, pd.DataFrame({"subject": ["person-z"], "profile_id": ["renamed-training-profile"], "score": [5.]}))


def test_individual_normalized_importance_and_conditional_logit_shares(audited):
    results, _ = audited
    fit = results["conjoint/fit"]
    importance = results["conjoint/importance"]
    ranges = fit["utilities"].groupby(["subject", "attribute"])["estimate"].agg(lambda v: v.max()-v.min()).unstack()
    expected = ranges.div(ranges.sum(1), axis=0)*100
    actual = importance["group"].set_index("attribute").mean_individual_importance_percent.loc[expected.columns]
    np.testing.assert_allclose(actual, expected.mean(0), atol=1e-12)
    shares = results["conjoint/logit_shares"]["individual"]
    assert set(shares.columns) >= {"subject", "profile_id", "conditional_share"}
    predicted = results["conjoint/predictions"]["predictions"]
    for subject, block in predicted.groupby("subject", sort=False):
        score = block.predicted.to_numpy()/1.25
        p = np.exp(score-score.max())
        p /= p.sum()
        observed = shares.query("subject == @subject").set_index("profile_id").loc[block.profile_id, "conditional_share"]
        np.testing.assert_allclose(observed, p, atol=1e-12)


def test_randomized_cdf_complete_joint_indicator_covariance(audited):
    results, inputs = audited
    result = results["causal_design/treatment_cdf"]
    frame, grid = inputs["experiment"], np.asarray(result.attrs["state"]["thresholds"])
    indicators = [(frame.loc[frame.treated == arm, "outcome"].to_numpy()[:, None] <= grid).astype(float) for arm in [0, 1]]
    covariance = sum(np.cov(a, rowvar=False, ddof=1)/len(a) for a in indicators)
    effect = indicators[1].mean(0)-indicators[0].mean(0)
    np.testing.assert_allclose(result.attrs["state"]["covariance"], covariance, atol=1e-15)
    np.testing.assert_allclose(result["effects"].estimate, effect, atol=1e-15)
    np.testing.assert_allclose(result["effects"].ci_low, effect-norm.ppf(.975)*np.sqrt(covariance.diagonal()), atol=1e-12)
    assert covariance[1, 2] > 0
    assert "pointwise" in result.attrs["state"]["inference"]


def test_quantile_full_bootstrap_joint_state_inverse_ecdf_and_percentiles(audited):
    results, inputs = audited
    result = results["causal_design/treatment_quantile"]
    state = result.attrs["state"]
    probability = np.asarray(state["quantiles"])
    frame = inputs["experiment"]
    for arm in [0, 1]:
        values = np.sort(frame.loc[frame.treated == arm, "outcome"].to_numpy())
        expected = values[np.ceil(len(values)*probability).astype(int)-1]
        np.testing.assert_array_equal(state["q"+str(arm)], expected)
    b0, b1, effect = [np.asarray(state[key]) for key in ["bootstrap_q0", "bootstrap_q1", "bootstrap_effects"]]
    assert b0.shape == b1.shape == effect.shape == (199, 3)
    np.testing.assert_array_equal(effect, b1-b0)
    joint = np.column_stack([b0, b1, effect])
    np.testing.assert_allclose(state["joint_covariance"], np.cov(joint, rowvar=False, ddof=1), atol=1e-14)
    np.testing.assert_allclose(state["ci_low"], np.quantile(effect, .025, axis=0), atol=1e-12)
    np.testing.assert_allclose(state["ci_high"], np.quantile(effect, .975, axis=0), atol=1e-12)
    assert result["effects"].p_value.isna().all()


def test_uncensored_rmst_greenwood_reduces_to_clipped_empirical_mean():
    frame = pd.DataFrame({"t": [1., 2., 3., 5., .5, .9, 3.8, 6.], "event": [1]*8, "d": [0]*4+[1]*4})
    result = oe.treatment_rmst(frame, "t", "event", "d", design="randomized", tau=4.5)
    arm = [np.minimum(frame.loc[frame.d == d, "t"], 4.5).to_numpy() for d in [0, 1]]
    estimates = np.array([a.mean() for a in arm])
    variances = np.array([a.var(ddof=0)/len(a) for a in arm])
    np.testing.assert_allclose(result["arms"].rmst, estimates, atol=1e-12)
    np.testing.assert_allclose(result["arms"].variance, variances, atol=1e-12)
    mapping = np.array([[1., 0.], [0., 1.], [-1., 1.]])
    np.testing.assert_allclose(result.attrs["state"]["covariance_matrix"], mapping@np.diag(variances)@mapping.T, atol=1e-12)


def test_paired_assignment_enumeration_and_sign_only_sensitivity(audited):
    results, _ = audited
    randomization = results["causal_design/paired_randomization"]
    state = randomization.attrs["state"]
    difference = np.asarray(state["differences"])
    enumeration = np.array(list(product([-1., 1.], repeat=len(difference))))@difference/len(difference)
    np.testing.assert_allclose(np.sort(state["randomization_statistics"]), np.sort(enumeration), atol=1e-15)
    assert state["n_assignments"] == 64
    assert len(state["assignment_bits"]) == 64
    observed = difference.mean()
    expected = np.mean(np.abs(enumeration) >= abs(observed)-1e-12)
    assert state["p_value"] == expected
    bounds = results["causal_design/rosenbaum_bounds"]["bounds"]
    n, k = np.count_nonzero(difference), np.count_nonzero(difference > 0)
    for row in bounds.itertuples(index=False):
        assert np.isclose(row.p_lower, binom.sf(k-1, n, 1/(1+row.gamma)), atol=1e-12)
        assert np.isclose(row.p_upper, binom.sf(k-1, n, row.gamma/(1+row.gamma)), atol=1e-12)


@pytest.mark.parametrize("family", ["categorical", "conjoint", "causal_design"])
def test_changed_complete_scientific_state_refuses_post_or_load(audited, family):
    results, inputs = audited
    if family == "categorical":
        payload = json.loads(oe.summary_state(results["categorical/nominal"]))
        payload["attrs"]["optimal_state"]["beta"][0] += 1
        restored = oe.restore_summary(json.dumps(payload))
        with pytest.raises(AnalysisError):
            oe.catreg_predict(restored, inputs["people"], missing="drop")
    else:
        save, load, name = (oe.conjoint_save, oe.conjoint_load, "conjoint/fit") if family == "conjoint" else (oe.causal_design_save, oe.causal_design_load, "causal_design/treatment_cdf")
        payload = deepcopy(save(results[name]))
        payload["payload"]["attrs"]["state"]["changed_scientific_state"] = 1
        with pytest.raises(AnalysisError):
            load(payload)


@pytest.mark.parametrize("family", ["categorical", "conjoint", "causal_design"])
def test_weight_and_work_contract_refusals(audited, family):
    results, inputs = audited
    call = (lambda **kw: oe.catreg_nominal(inputs["people"], "y", ["a", "b"], **kw)) if family == "categorical" else (lambda **kw: oe.conjoint_predict(results["conjoint/fit"], inputs["conjoint_holdout"], **kw)) if family == "conjoint" else (lambda **kw: oe.treatment_cdf(inputs["experiment"], "outcome", "treated", design="randomized", thresholds=[0., 1.], **kw))
    with pytest.raises(AnalysisError):
        call(weights="invented_sampling_weight")
    with pytest.raises(AnalysisError):
        call(max_work=1)


def test_source_example_preserves_private_rng_and_all_target_inputs():
    namespace = runpy.run_path(str(EXAMPLE))
    before = torch.random.get_rng_state().clone()
    namespace["fixtures"]()
    assert torch.equal(torch.random.get_rng_state(), before)
