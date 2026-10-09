"""Independent survey geometry, retained native replay and declared MI coverage.

NumPy/SciPy are development oracles; no reference estimator supplies runtime
answers. The coverage design and Monte Carlo seeds are fixed before execution.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, norm, t
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/audited-eight-capability-closures-2026-10-07"
MI_NATIVE = ROOT / "docs/evidence/missing-data-eight-2026-10-07/persisted-states.json"
SURVEY_NATIVE = ROOT / "docs/evidence/survey-regression-2026-10-07/persisted-states.json"


@pytest.fixture(scope="module", autouse=True)
def one_factor_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _survey_fixture():
    row = np.arange(24)
    frame = pd.DataFrame({
        "w": 1. + row % 4, "p": row // 3, "h": row // 12,
        "N": np.where(row < 12, 8, 10),
        "x": np.sin(row * .71) + row / 18,
        "y": 1.3 + .8 * np.sin(row * .71) + np.cos(row * 1.29),
        "d": row >= 3,
    }, index=pd.Index([f"duplicate-{i // 2}" for i in row], name="unit"))
    frame.iloc[7, frame.columns.get_loc("y")] = np.nan
    return frame


def test_survey_domain_and_missing_retains_zero_score_psus_full_fpc_covariance():
    frame = _survey_fixture()
    design = oe.survey_design(frame, weights="w", psu="p", strata="h", fpc="N")
    result = oe.survey_regress(frame, design, "y", ["x"], domain="d", missing="drop")
    selected = frame.d.to_numpy() & frame.y.notna().to_numpy()
    x = np.column_stack([np.ones(len(frame)), frame.x])
    weights, y = frame.w.to_numpy(), frame.y.fillna(0).to_numpy()
    a = x[selected].T @ (weights[selected, None] * x[selected])
    beta = np.linalg.solve(a, x[selected].T @ (weights[selected] * y[selected]))
    score = np.zeros_like(x)
    score[selected] = weights[selected, None] * x[selected] * (y[selected] - x[selected] @ beta)[:, None]
    meat = np.zeros((2, 2))
    for stratum in [0, 1]:
        groups = sorted(frame.loc[frame.h == stratum, "p"].unique())
        totals = np.array([score[frame.p.to_numpy() == group].sum(0) for group in groups])
        centered = totals - totals.mean(0)
        count, population = len(groups), int(frame.loc[frame.h == stratum, "N"].iloc[0])
        meat += (1 - count / population) * count / (count - 1) * centered.T @ centered
    inv = np.linalg.inv(a)
    covariance = inv @ meat @ inv.T
    assert result.df == 6
    assert result.metadata["sample_positions"] == np.flatnonzero(selected).tolist()
    np.testing.assert_allclose(result.coefficients, beta, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(result.covariance, covariance, rtol=2e-11, atol=2e-12)
    assert abs(covariance[0, 1]) > 1e-4
    actual = result.to_frame()
    se = np.sqrt(covariance.diagonal())
    np.testing.assert_allclose(actual.p_value, 2 * t.sf(abs(beta / se), 6), rtol=2e-11)


def test_survey_fixed_distribution_margins_and_joint_prediction_covariance():
    frame = _survey_fixture()
    design = oe.survey_design(frame, weights="w", psu="p", strata="h", fpc="N")
    fit = oe.survey_regress(frame, design, "y", ["x"], missing="drop")
    evaluation = pd.DataFrame({"x": [-1., np.nan, 2., .5], "weight": [1., 8., 3., 2.]},
                              index=["same", "missing", "same", "last"])
    prediction = oe.survey_predict(fit, evaluation, missing="drop")
    x = np.array([[1., -1.], [1., 2.], [1., .5]])
    beta, cov = np.array(fit.coefficients), np.array(fit.covariance)
    np.testing.assert_allclose(prediction.estimate, x @ beta)
    np.testing.assert_allclose(prediction.attrs["covariance_matrix"], x @ cov @ x.T)
    assert prediction.attrs["physical_positions"] == [0, 2, 3]
    assert prediction.index.tolist() == ["same", "same", "last"]
    margins = oe.survey_margins(fit, evaluation, weights="weight", missing="drop")
    gradient = np.array([1., (np.array([-1., 2., .5]) @ np.array([1., 3., 2.])) / 6])
    np.testing.assert_allclose(margins.estimate, [gradient @ beta])
    np.testing.assert_allclose(margins.attrs["covariance_matrix"], [[gradient @ cov @ gradient]])
    assert margins.attrs["conditional_fixed_covariates"] is True
    assert margins.attrs["population_distribution_uncertainty"] is False


@pytest.mark.parametrize("family", ["linear", "logit", "probit", "poisson"])
def test_retained_native_survey_fit_replays_full_scores_covariance_without_refitting(family, monkeypatch):
    from openecon.econometrics.survey import regression

    def no_fit(*args, **kwargs):
        raise AssertionError("Native survey state replay must not refit")

    monkeypatch.setattr(regression, "_fit", no_fit)
    state = json.loads(SURVEY_NATIVE.read_text())[family]
    restored = oe.SurveyRegressionResult.model_validate_json(json.dumps(state, allow_nan=False))
    assert restored.model_dump(mode="json") == state
    assert restored.df == 4 and len(restored.covariance) == 2
    assert "tabular" in restored.to_frame().to_latex()
    se = np.sqrt(np.diag(restored.covariance))
    np.testing.assert_allclose(restored.to_frame().std_error, se, rtol=2e-13)
    contrast = np.array([.7, -.4])
    linear = oe.survey_lincom(restored, contrast.tolist(), null=.2)
    expected_se = np.sqrt(contrast @ np.array(restored.covariance) @ contrast)
    expected = contrast @ np.array(restored.coefficients)
    np.testing.assert_allclose(linear.std_error, [expected_se], rtol=2e-12)
    np.testing.assert_allclose(linear.p_value, [2 * t.sf(abs((expected - .2) / expected_se), 4)], rtol=2e-12)


@pytest.mark.parametrize("name", ["em", "mcar", "mvn", "monotone", "normal", "pmm", "logit", "d1"])
def test_all_retained_native_mi_states_replay_without_estimation(name, monkeypatch):
    from openecon.econometrics.mi import chained, diagnostics, generation

    def no_fit(*args, **kwargs):
        raise AssertionError("Saved state replay must not estimate or draw imputations")

    for module, function in [(diagnostics, "_fit"), (chained, "mi_chained"),
                             (generation, "mi_mvn"), (generation, "mi_monotone")]:
        monkeypatch.setattr(module, function, no_fit)
    state = json.loads(MI_NATIVE.read_text())[name]
    cls = (oe.MIDiagnosticResult if name in {"em", "mcar"}
           else oe.MIJointResult if name == "d1" else oe.MIResult)
    restored = cls.model_validate_json(json.dumps(state, allow_nan=False))
    assert restored.model_dump(mode="json") == state
    assert "tabular" in str(restored.to_latex())
    if isinstance(restored, oe.MIResult):
        for matrix in restored.completed_matrices:
            for original, complete in zip(restored.original, matrix):
                assert all(math.isfinite(v) for v in complete)
                assert all(a is None or a == b for a, b in zip(original, complete))
        assert restored.metadata["sample"]["positions"] == tuple(range(72))
        assert restored.dataset(1).index.has_duplicates
        assert restored.metadata["converged"] is False


def test_saved_mi_checksum_does_not_override_observed_cell_geometry():
    from openecon.econometrics.mi.common import _digest

    state = json.loads(MI_NATIVE.read_text())["monotone"]
    state["completed_matrices"][0][0][0] += .2
    state["integrity_sha256"] = _digest({k: v for k, v in state.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError, match="observed cell"):
        oe.MIResult.model_validate(state)


@pytest.mark.parametrize("method", ["ordinal", "count", "passive"])
def test_remaining_original_mi_domains_are_explicitly_refused(method):
    with pytest.raises(AnalysisError, match="Supported chained methods"):
        oe.mi_chained(pd.DataFrame({"x": range(8), "y": [0., 1., 2., 3., 4., 5., None, None]}),
                      ["x", "y"], methods={"y": method})


def test_full_d1_is_invariant_to_joint_restriction_basis_and_parameter_units():
    rng = np.random.default_rng(302359)
    estimates = rng.normal(size=(12, 3)) + [1.2, -.4, .7]
    within = np.array([np.array([[2., .4, -.3], [.4, 1.4, .1], [-.3, .1, 1.]])
                       * (1 + i / 20) for i in range(12)])
    r = np.array([[1., .5, -.2], [0., 1., .3]])
    null = np.array([.5, -.1])
    pool = oe.mi_pool(estimates, within, complete_df=150,
                      imputation_description="Declared shared normal estimand, full covariance")
    actual = oe.mi_test(pool, restrictions=r, values=null)
    w = r @ within.mean(0) @ r.T
    centered = estimates - estimates.mean(0)
    b = r @ (centered.T @ centered / 11) @ r.T
    riv = (1 + 1 / 12) * np.trace(np.linalg.solve(w, b)) / 2
    difference = r @ estimates.mean(0) - null
    expected = difference @ np.linalg.solve(w, difference) / (2 * (1 + riv))
    np.testing.assert_allclose(actual.table.statistic, [expected], rtol=2e-12)
    np.testing.assert_allclose(actual.covariance, (1 + riv) * w, rtol=2e-12)
    units = np.array([[2., .1, 0.], [0., .5, .1], [.2, 0., 1.3]])
    basis = np.array([[1., .7], [-.3, 2.]])
    transformed = oe.mi_pool(estimates @ units.T,
                             np.array([units @ u @ units.T for u in within]), complete_df=150,
                             imputation_description="Same estimand in transformed units")
    other = oe.mi_test(transformed, restrictions=basis @ r @ np.linalg.inv(units),
                       values=basis @ null)
    np.testing.assert_allclose(other.table[["statistic", "df2", "p_value"]],
                               actual.table[["statistic", "df2", "p_value"]], rtol=2e-11)


def _wilson(successes, total, level):
    z = norm.isf((1 - level) / 2)
    phat = successes / total
    denominator = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


@pytest.mark.parametrize("mechanism,seed", [("MCAR", 302310), ("MAR_given_x", 302311)])
def test_predeclared_monotone_gaussian_rubin_coverage(mechanism, seed):
    """Predeclared iid linear-Gaussian, missing outcomes only, fixed x analysis.

    160 repeated samples of n=120 and m=20, 95% Barnard–Rubin slope intervals.
    The two-sided 1% binomial test detects gross calibration failure, with exact
    and Wilson Monte Carlo intervals retained. It is not an omnibus MI proof.
    """
    design = {"mechanism": mechanism, "seed": seed, "rng": "NumPy default_rng PCG64",
              "replications": 160, "n": 120, "m": 20, "beta": [.7, 1.2], "sigma": 1.3,
              "missing_probability": ".30" if mechanism == "MCAR" else "logistic(-1+.7*x)",
              "method": "mi_monotone: p(beta,sigma2) proportional to 1/sigma2",
              "analysis": "independent NumPy full iid OLS covariance; native Rubin/Barnard–Rubin",
              "complete_df": 118, "interval_level": .95,
              "assessment": "two-sided binomial nominal .95 at predeclared alpha=.01"}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE / f"survey-mi-coverage-{mechanism}.json"
    path.write_text(json.dumps({"predeclared_design": design}, indent=2) + "\n")
    rng = np.random.default_rng(seed)
    successes, missing_counts, records = 0, [], []
    for replication in range(design["replications"]):
        x = rng.normal(size=120)
        y = .7 + 1.2 * x + rng.normal(scale=1.3, size=120)
        probability = np.full(120, .3) if mechanism == "MCAR" else 1 / (1 + np.exp(1 - .7 * x))
        missing = rng.random(120) < probability
        data = pd.DataFrame({"x": x, "y": np.where(missing, np.nan, y)})
        generated = oe.mi_monotone(data, ["x", "y"], m=20, seed=seed * 1000 + replication * 25)
        xmatrix = np.column_stack([np.ones(120), x])
        inv = np.linalg.inv(xmatrix.T @ xmatrix)
        estimates, covariance = [], []
        for matrix in generated.completed_matrices:
            completed_y = np.array(matrix)[:, 1]
            beta = inv @ xmatrix.T @ completed_y
            residual = completed_y - xmatrix @ beta
            estimates.append(beta)
            covariance.append((residual @ residual / 118) * inv)
        pool = oe.mi_pool(np.array(estimates), np.array(covariance), complete_df=118,
                          terms=["intercept", "slope"],
                          imputation_description="Verified monotone Gaussian outcome MI; iid OLS common slope")
        result = pool.table.iloc[1]
        covered = bool(result.ci_low <= 1.2 <= result.ci_high)
        successes += covered
        missing_counts.append(int(missing.sum()))
        records.append({"replication": replication, "imputation_seed": generated.seed,
                        "missing": int(missing.sum()), "estimate": float(result.estimate),
                        "df": float(result.df), "ci": [float(result.ci_low), float(result.ci_high)],
                        "covered": covered})
    check = binomtest(successes, 160, p=.95)
    interval = check.proportion_ci(confidence_level=.99, method="exact")
    receipt = {"predeclared_design": design, "successes": successes, "coverage": successes / 160,
               "wilson_95_mc_interval": _wilson(successes, 160, .95),
               "exact_99_mc_interval": [float(interval.low), float(interval.high)],
               "two_sided_binomial_p": float(check.pvalue),
               "mean_missing_fraction": float(np.mean(missing_counts) / 120),
               "complete_replicate_records": records,
               "scope": "One congenial Gaussian observed-x/outcome-MCAR-or-MAR geometry; no FCS/PMM/binary/MNAR coverage claim"}
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    assert check.pvalue >= .01, receipt


def test_runnable_example_retains_complete_states_outputs_and_typed_replay(tmp_path):
    path = ROOT / "docs/examples/audited_eight_survey_mi.py"
    spec = importlib.util.spec_from_file_location("audited_survey_mi_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    receipt = example.run(tmp_path)
    assert receipt["stages"] == 16 and receipt["survey_design_df"] == 4
    assert receipt["finite_chain_convergence_assessed"] is False
    assert len(receipt["files"]) == 26
    for name, digest in receipt["files"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    state = json.loads((tmp_path / "missing_data-full-state.json").read_text())
    assert len(state) == 9 and len(state["d1"]["pool"]["estimates"]) == 6
    assert state["pool"] == state["d1"]["pool"]
    for i, source in enumerate(oe.MIPoolResult.model_validate(state["pool"]).metadata["source_results"], 1):
        fit = oe.ResultBundle.model_validate_json((tmp_path / f"mi-ols-imputation-{i:02d}.json").read_text())
        assert fit.id == source["id"]
    survey = json.loads((tmp_path / "survey-full-state.json").read_text())
    assert len(survey["fits"]) == len(survey["postestimation"]) == 4
    assert survey["postestimation"]["margins"]["attrs"]["population_distribution_uncertainty"] is False
