"""Independent dense NumPy QR / SciPy oracles for dependent summary effects.

Reference calculations never call a production kernel, a Torch linear solve,
or the production variance optimizer. SciPy is a development oracle only.
"""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import linalg, optimize, stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.resources import use_workspace_budget


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(
    seed=729636, *, groups=8, sizes=None, correlation=0.45, spread=1.0, tau=0.6, weak=False
):
    """Unequal study blocks with a declared complete known covariance."""
    rng = np.random.default_rng(seed)
    sizes = list(sizes) if sizes is not None else [2 + i % 3 for i in range(groups)]
    ids = np.repeat([f"study:{i}" for i in range(len(sizes))], sizes)
    n = len(ids)
    x = rng.normal(size=n)
    if weak:
        x = 8.0 + x * 1e-4
    variance = np.exp(np.linspace(-spread, spread, n)) * 0.3
    sampling = np.zeros((n, n))
    start = 0
    for size in sizes:
        rows = slice(start, start + size)
        scales = np.sqrt(variance[rows])
        block = (1 - correlation) * np.eye(size) + correlation * np.ones((size, size))
        sampling[rows, rows] = scales[:, None] * block * scales[None, :]
        start += size
    # The fixture declares a literal symmetric known matrix. Products with
    # different multiplication orders can otherwise differ by one ULP.
    sampling = (sampling + sampling.T) * 0.5
    b = rng.normal(scale=math.sqrt(tau), size=len(sizes)) if tau else np.zeros(len(sizes))
    y = 0.4 - 0.3 * x + np.repeat(b, sizes) + np.linalg.cholesky(sampling) @ rng.normal(size=n)
    data = pd.DataFrame(
        {"effect": [f"effect:{i}" for i in range(n)], "study": ids, "yi": y, "x": x},
        index=[f"row:{i // 2}" for i in range(n)],
    )
    covariance = pd.DataFrame(sampling, index=data.effect, columns=data.effect)
    return data, covariance


def dense_oracle(
    data,
    covariance,
    *,
    model="common",
    method="REML",
    moderators=("x",),
    intercept=True,
    fixed_tau=None,
):
    """Raw-design Cholesky whitening, independent QR, full dense likelihood."""
    labels = data.effect.tolist()
    s = covariance.loc[labels, labels].to_numpy(dtype=float)
    y = data.yi.to_numpy(dtype=float)
    x = np.column_stack(
        [
            *([np.ones(len(y))] if intercept else []),
            *[data[c].to_numpy(dtype=float) for c in moderators],
        ]
    )
    n, p = x.shape
    studies = data.study.to_numpy()
    k = np.eye(n) if model == "effect" else (studies[:, None] == studies[None, :]).astype(float)
    if model == "common":
        k = np.zeros((n, n))
    _, raw_r = np.linalg.qr(x, mode="reduced")
    log_x = 2 * np.log(np.abs(np.diag(raw_r))).sum()

    def at(tau):
        m = s + tau * k
        lower = linalg.cholesky(m, lower=True)
        wx = linalg.solve_triangular(lower, x, lower=True)
        wy = linalg.solve_triangular(lower, y, lower=True)
        q, r = np.linalg.qr(wx, mode="reduced")
        beta = linalg.solve_triangular(r, q.T @ wy)
        ri = linalg.solve_triangular(r, np.eye(p))
        bread = ri @ ri.T
        residual = y - x @ beta
        white_resid = linalg.solve_triangular(lower, residual, lower=True)
        rss = float(white_resid @ white_resid)
        log_m = 2 * np.log(np.diag(lower)).sum()
        log_gram = 2 * np.log(np.abs(np.diag(r))).sum()
        ml = -0.5 * (log_m + rss + n * math.log(2 * math.pi))
        reml = -0.5 * (log_m + rss + log_gram - log_x + (n - p) * math.log(2 * math.pi))
        objective = -2 * (ml if method == "ML" else reml)
        inverse = linalg.cho_solve((lower, True), np.eye(n))
        return dict(
            objective=objective,
            beta=beta,
            covariance=bread,
            residual=residual,
            inverse=inverse,
            marginal=m,
            rss=rss,
            loglik_ml=float(ml),
            loglik_reml=float(reml),
        )

    if model == "common":
        tau = 0.0
        candidates = [(0.0, at(0.0)["objective"])]
    elif fixed_tau is not None:
        tau = float(fixed_tau)
        candidates = [(tau, at(tau)["objective"])]
    else:
        # Independently minimize EVERY local grid valley and compare the zero
        # boundary. This does not assume a globally unimodal scalar profile.
        scale = float(np.median(np.diag(s)))
        upper = max(scale, float(np.var(y)), 1.0) * 1e4
        grid = np.r_[0.0, np.geomspace(scale * 1e-11, upper, 241)]
        values = np.asarray([at(v)["objective"] for v in grid])
        assert values[-1] > values.min() + 1.0, "Independent upper profile unresolved"
        candidates = [(0.0, values[0])]
        for i in range(1, len(grid) - 1):
            if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
                minimum = optimize.minimize_scalar(
                    lambda v: at(v)["objective"],
                    bounds=(grid[i - 1], grid[i + 1]),
                    method="bounded",
                    options={"xatol": 1e-13, "maxiter": 200},
                )
                assert minimum.success
                candidates.append((float(minimum.x), float(minimum.fun)))
        tau = min(candidates, key=lambda pair: pair[1])[0]
    result = at(tau)
    result.update(
        tau2=tau, x=x, y=y, sampling=s, kernel=k, studies=studies, candidates=candidates, n=n, p=p
    )
    return result


def run_fit(
    data, covariance, *, model="common", method="REML", moderators=("x",), intercept=True, **kwargs
):
    return oe.meta_dependent(
        data=data,
        covariance=covariance,
        study="study",
        moderators=list(moderators),
        intercept=intercept,
        model=model,
        method=method,
        **kwargs,
    )


def fit_state(result):
    return result.attrs["prediction_state"]


def close(actual, expected, *, rtol=3e-7, atol=2e-9):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("model", ["common", "effect", "study"])
@pytest.mark.parametrize("method", ["ML", "REML"])
@pytest.mark.parametrize("moderators", [(), ("x",)])
def test_complete_dense_qr_likelihood_and_original_design(model, method, moderators):
    data, s = fixture()
    before, s_before = data.copy(deep=True), s.copy(deep=True)
    actual = run_fit(data, s, model=model, method=method, moderators=moderators)
    reference = dense_oracle(data, s, model=model, method=method, moderators=moderators)
    state = fit_state(actual)
    close(actual["coefficients"].estimate, reference["beta"])
    close(actual["covariance"], reference["covariance"])
    close(actual["coefficients"].std_error, np.sqrt(np.diag(reference["covariance"])))
    close(
        actual["coefficients"].p_value,
        2 * stats.norm.sf(abs(reference["beta"]) / np.sqrt(np.diag(reference["covariance"]))),
        rtol=2e-6,
        atol=1e-11,
    )
    assert state["tau2"] == pytest.approx(reference["tau2"], rel=3e-6, abs=3e-8)
    close(state["x"], reference["x"])
    close(state["s"], reference["sampling"])
    close(state["fit_covariance"], reference["covariance"])
    close(actual["fit"].iloc[0].loglik_ml, reference["loglik_ml"], rtol=2e-9)
    close(actual["fit"].iloc[0].loglik_reml, reference["loglik_reml"], rtol=2e-9)
    assert state["effect_ids"] == data.effect.tolist()
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(s, s_before)


@pytest.mark.parametrize("model", ["effect", "study"])
@pytest.mark.parametrize("method", ["ML", "REML"])
def test_zero_boundary_is_compared_not_approximate_positive_variance(model, method):
    data, s = fixture(tau=0)
    data.yi = 0.2 + 0.7 * data.x + np.linspace(-0.0001, 0.0001, len(data))
    actual = run_fit(data, s, model=model, method=method)
    reference = dense_oracle(data, s, model=model, method=method)
    assert fit_state(actual)["tau2"] == reference["tau2"] == 0.0
    close(actual["covariance"], reference["covariance"])


@pytest.mark.parametrize("model", ["common", "effect", "study"])
def test_row_and_independently_permuted_covariance_labels(model):
    data, s = fixture(seed=729635)
    fit = run_fit(data, s, model=model)
    rng = np.random.default_rng(636)
    order = rng.permutation(len(data))
    data2 = data.iloc[order].copy()
    s2 = s.iloc[rng.permutation(len(data)), rng.permutation(len(data))]
    changed = run_fit(data2, s2, model=model)
    close(changed["coefficients"].estimate, fit["coefficients"].estimate)
    close(changed["covariance"], fit["covariance"])
    assert fit_state(changed)["effect_ids"] == data2.effect.tolist()
    close(fit_state(changed)["s"], s.loc[data2.effect, data2.effect])


@pytest.mark.parametrize("model", ["effect", "study"])
@pytest.mark.parametrize("method", ["ML", "REML"])
def test_singleton_studies_reduce_existing_independent_likelihood(model, method):
    data, s = fixture(groups=9, sizes=[1] * 9)
    independent = data.copy()
    independent["vi"] = np.diag(s)
    existing = oe.meta_regress(
        data=independent, moderators=["x"], study="study", method=method, inference="z"
    )
    dependent = run_fit(data, s, model=model, method=method)
    close(dependent["coefficients"].estimate, existing["coefficients"].estimate)
    close(dependent["covariance"], existing["covariance"])
    assert fit_state(dependent)["tau2"] == pytest.approx(
        existing["heterogeneity"].iloc[0].tau2, rel=2e-6, abs=3e-9
    )


@pytest.mark.parametrize("model", ["common", "effect", "study"])
@pytest.mark.parametrize("correction", ["CR0", "CR1"])
def test_full_independent_study_scores_sandwich_and_finite_df(model, correction):
    data, s = fixture(seed=632)
    fitted = run_fit(data, s, model=model)
    actual = oe.meta_dependent_robust(fitted, correction=correction)
    reference = dense_oracle(data, s, model=model)
    scores = []
    study_order = list(dict.fromkeys(data.study))
    for study in study_order:
        rows = np.flatnonzero(data.study.to_numpy() == study)
        scores.append(
            reference["x"][rows].T
            @ reference["inverse"][np.ix_(rows, rows)]
            @ reference["residual"][rows]
        )
    scores = np.asarray(scores)
    g, p = scores.shape
    factor = 1.0 if correction == "CR0" else g / (g - p)
    expected = factor * reference["covariance"] @ scores.T @ scores @ reference["covariance"]
    close(actual["covariance"], expected, rtol=4e-7)
    close(fit_state(actual)["covariance"], expected, rtol=4e-7)
    assert actual.attrs["residual_df"] == g - p
    se = np.sqrt(np.diag(expected))
    close(actual["coefficients"].std_error, se)
    close(
        actual["coefficients"].p_value,
        2 * stats.t.sf(abs(reference["beta"] / se), g - p),
        rtol=3e-6,
    )
    cut = stats.t.isf(0.025, g - p)
    close(actual["coefficients"].ci_low, reference["beta"] - cut * se)
    close(actual["coefficients"].ci_high, reference["beta"] + cut * se)
    # Score rows and EVERY score coordinate must survive the result, including
    # studies whose score nearly vanishes.
    terms = fit_state(actual)["terms"]
    score_table = actual["cluster_scores"]
    close(score_table.loc[:, terms], scores)
    close(scores.sum(axis=0), np.zeros(p), atol=2e-7)


@pytest.mark.parametrize("correction", ["CR0", "CR1"])
@pytest.mark.parametrize(
    "moderators",
    [
        ["x"],
        ["study", "n_effects"],
        ["study", "n_effects", "__study__", "__n_effects__"],
    ],
)
def test_study_score_metadata_survives_coefficient_name_collisions(moderators, correction):
    data, sampling = fixture(seed=629632, tau=0)
    data = data.rename(columns={"study": "sampling_group"})
    rng = np.random.default_rng(629633)
    reference_data = data.loc[:, ["effect", "yi"]].copy()
    reference_data["study"] = data["sampling_group"]
    reference_names = []
    for i, name in enumerate(moderators):
        values = rng.normal(size=len(data))
        data[name] = values
        safe_name = f"moderator_{i}"
        reference_data[safe_name] = values
        reference_names.append(safe_name)
    fitted = oe.meta_dependent(
        data=data,
        covariance=sampling,
        study="sampling_group",
        moderators=moderators,
        model="common",
    )
    actual = oe.meta_dependent_robust(fitted, correction=correction)
    reference = dense_oracle(reference_data, sampling, moderators=reference_names)
    study_order = list(dict.fromkeys(reference_data.study))
    scores, counts = [], []
    for study in study_order:
        rows = np.flatnonzero(reference_data.study.to_numpy() == study)
        counts.append(len(rows))
        scores.append(
            reference["x"][rows].T
            @ reference["inverse"][np.ix_(rows, rows)]
            @ reference["residual"][rows]
        )
    scores = np.asarray(scores)
    g, p = scores.shape
    multiplier = 1.0 if correction == "CR0" else g / (g - p)
    expected_covariance = (
        multiplier * reference["covariance"] @ scores.T @ scores @ reference["covariance"]
    )
    close(actual["coefficients"].estimate, reference["beta"])
    close(actual["covariance"], expected_covariance)
    close(fit_state(actual)["covariance"], expected_covariance)
    assert actual.attrs["residual_df"] == g - p
    metadata = actual.attrs["cluster_score_metadata_columns"]
    assert set(metadata) == {"study", "n_effects"}
    assert len(set(metadata.values())) == 2
    terms = fit_state(actual)["terms"]
    assert not set(metadata.values()) & set(terms)
    score_table = actual["cluster_scores"]
    assert score_table[metadata["study"]].tolist() == study_order
    assert score_table[metadata["n_effects"]].tolist() == counts
    assert len(score_table) == g
    close(score_table.loc[:, terms], scores)
    close(scores.sum(axis=0), np.zeros(p), atol=2e-7)


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("joint", [False, True])
def test_nonzero_null_complete_joint_contrast_covariance(robust, joint):
    data, s = fixture(seed=633)
    fitted = run_fit(data, s, model="study")
    if robust:
        fitted = oe.meta_dependent_robust(fitted)
    state = fit_state(fitted)
    terms = state["terms"]
    r = np.asarray([[1.0, 0.4], [-0.3, 1.2]])[: 2 if joint else 1]
    labels = ["target-a", "target-b"][: len(r)]
    contrast = pd.DataFrame(r, index=labels, columns=terms)
    null = [0.15, -0.2][: len(r)]
    actual = oe.meta_dependent_contrast(fitted, contrasts=contrast.iloc[:, ::-1], null=null)
    beta, covariance = np.asarray(state["beta"]), np.asarray(state["covariance"])
    estimate = r @ beta
    cv = r @ covariance @ r.T
    se = np.sqrt(np.diag(cv))
    difference = estimate - np.asarray(null)
    wald = float(difference @ np.linalg.solve(cv, difference))
    df = state["n_studies"] - len(terms)
    cut = stats.t.isf(0.025, df) if robust else stats.norm.isf(0.025)
    close(actual["contrasts"].estimate, estimate)
    close(actual["covariance"], cv)
    close(actual["contrasts"].std_error, se)
    close(actual["contrasts"].ci_low, estimate - cut * se)
    close(actual["contrasts"].ci_high, estimate + cut * se)
    joint_row = actual["joint_test"].iloc[0]
    close(joint_row.statistic, wald / len(r) if robust else wald)
    close(
        joint_row.p_value,
        stats.f.sf(wald / len(r), len(r), df) if robust else stats.chi2.sf(wald, len(r)),
        rtol=3e-6,
    )


@pytest.mark.parametrize("model", ["common", "effect", "study"])
@pytest.mark.parametrize("robust", [False, True])
def test_saved_full_joint_new_mean_and_model_specific_latent_covariance(model, robust):
    data, s = fixture(seed=635)
    fitted = run_fit(data, s, model=model)
    if robust:
        fitted = oe.meta_dependent_robust(fitted)
    restored = oe.restore_summary(oe.summary_state(fitted))
    state = fit_state(restored)
    before = copy.deepcopy(state)
    new = pd.DataFrame(
        {"x": [-1.3, 0.2, 0.8], "future": ["new:A", "new:A", "new:B"]},
        index=["profile:z", "profile:a", "profile:m"],
    )
    mean = oe.meta_dependent_predict(restored, data=new)
    latent = oe.meta_dependent_predict_effect(state, data=new, study="future")
    x = np.column_stack([np.ones(3), new.x])
    expected_mean = x @ np.asarray(state["beta"])
    expected_cv = x @ np.asarray(state["covariance"]) @ x.T
    kernel = (
        np.zeros((3, 3))
        if model == "common"
        else np.eye(3)
        if model == "effect"
        else (new.future.to_numpy()[:, None] == new.future.to_numpy()[None, :]).astype(float)
    )
    random_cv = state["tau2"] * kernel
    close(mean["prediction"].estimate, expected_mean)
    close(mean["mean_covariance"], expected_cv)
    close(latent["mean_covariance"], expected_cv)
    close(latent["latent_covariance"], random_cv)
    close(latent["predictive_covariance"], expected_cv + random_cv)
    assert mean["mean_covariance"].index.tolist() == [0, 1, 2]
    assert mean["mean_covariance"].columns.tolist() == [0, 1, 2]
    assert mean["prediction"].original_label.tolist() == new.index.tolist()
    assert fit_state(restored) == before


@pytest.mark.parametrize("model", ["common", "effect", "study"])
def test_complete_whole_study_deletions_against_independent_refits(model):
    data, s = fixture(seed=636, groups=5)
    fitted = run_fit(data, s, model=model)
    actual = oe.meta_dependent_diagnostics(fitted)
    deletions = actual["leave_one_study_out"]
    coefficients = actual["deletion_coefficients"]
    assert len(deletions) == data.study.nunique()
    terms = fit_state(fitted)["terms"]
    for study in dict.fromkeys(data.study):
        subset = data.loc[data.study != study]
        keep = subset.effect.tolist()
        expected = dense_oracle(subset, s.loc[keep, keep], model=model)
        row = deletions.loc[deletions.study == study].iloc[0]
        assert row.n_effects == len(subset)
        assert row.n_studies == subset.study.nunique()
        assert row.tau2 == pytest.approx(expected["tau2"], rel=4e-6, abs=4e-8)
        rows = coefficients.loc[coefficients.study == study].set_index("term").loc[terms]
        close(rows.estimate, expected["beta"], rtol=4e-7)
        close(rows.std_error, np.sqrt(np.diag(expected["covariance"])), rtol=4e-7)


def test_deletion_singular_design_is_retained_with_failed_denominator():
    data, s = fixture(groups=4, sizes=[3, 2, 2, 2], tau=0)
    data.x = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fitted = run_fit(data, s)
    actual = oe.meta_dependent_diagnostics(fitted)
    deleted = actual["leave_one_study_out"]
    assert len(deleted) == 4
    assert len(actual["deletion_coefficients"]) == 8
    failed_coefficients = actual["deletion_coefficients"].loc[
        actual["deletion_coefficients"].study == "study:0"
    ]
    assert len(failed_coefficients) == 2
    assert failed_coefficients.estimate.isna().all()
    failed_covariance = actual["deletion_covariance"].loc[
        actual["deletion_covariance"].study == "study:0"
    ]
    assert len(failed_covariance) == 4
    assert failed_covariance.covariance.isna().all()
    row = deleted.loc[deleted.study == "study:0"].iloc[0]
    assert row.status == "failed"
    assert isinstance(row.error_code, str) and row.error_code
    summary = actual["summary"].iloc[0]
    assert summary.attempted == 4 and summary.succeeded == 3 and summary.failed == 1
    failed = next(item for item in actual.attrs["refit_states"] if item["study"] == "study:0")
    assert failed["prediction_state"] is None
    assert failed["error_code"] == row.error_code
    assert failed["retained_positions"] == list(range(3, 9))


@pytest.mark.parametrize("model", ["common", "effect", "study"])
@pytest.mark.parametrize("case", [0, 1, 2])
def test_unequal_variance_correlated_and_weak_moderator_grid(model, case):
    settings = [(9, 0.0, 0.2, False), (11, -0.2, 3.0, False), (12, 0.8, 5.0, True)]
    groups, rho, spread, weak = settings[case]
    # Fixed block size3 admits rho > -1/2, including negative correlation.
    data, s = fixture(
        seed=900 + case,
        groups=groups,
        sizes=[3] * groups,
        correlation=rho,
        spread=spread,
        weak=weak,
    )
    reference = dense_oracle(data, s, model=model)
    actual = run_fit(data, s, model=model)
    close(actual["coefficients"].estimate, reference["beta"], rtol=2e-6, atol=2e-7)
    close(actual["covariance"], reference["covariance"], rtol=2e-6, atol=2e-8)
    assert fit_state(actual)["tau2"] == pytest.approx(reference["tau2"], rel=5e-6, abs=5e-8)


@pytest.mark.parametrize(
    "name,value",
    [
        ("model", "nested"),
        ("method", "DL"),
        ("device", "cuda"),
        ("device", "meta"),
        ("device", "mps"),
        ("weights", "w"),
        ("intercept", 1),
        ("level", True),
        ("level", 0.5),
        ("level", 1.0),
    ],
)
def test_explicit_unsupported_options(name, value):
    data, s = fixture()
    args = dict(data=data, covariance=s, study="study", moderators=["x"])
    args[name] = value
    with pytest.raises(AnalysisError):
        oe.meta_dependent(**args)


@pytest.mark.parametrize(
    "kind",
    [
        "missing-y",
        "missing-study",
        "missing-x",
        "duplicate-effect",
        "bool-y",
        "nonfinite-y",
        "duplicate-column",
        "singular-x",
        "raw-covariance",
        "missing-label",
        "duplicate-label",
        "cross-study",
        "asymmetric",
        "not-spd",
        "nonfinite-covariance",
    ],
)
def test_complete_input_and_covariance_admission_refusals(kind):
    data, s = fixture()
    if kind == "missing-y":
        data.iloc[0, data.columns.get_loc("yi")] = np.nan
    elif kind == "missing-study":
        data.iloc[0, data.columns.get_loc("study")] = None
    elif kind == "missing-x":
        data.iloc[0, data.columns.get_loc("x")] = np.nan
    elif kind == "duplicate-effect":
        data.iloc[1, 0] = data.iloc[0, 0]
    elif kind == "bool-y":
        data.yi = True
    elif kind == "nonfinite-y":
        data.iloc[0, 2] = np.inf
    elif kind == "duplicate-column":
        data = pd.concat([data, data[["x"]]], axis=1)
    elif kind == "singular-x":
        data.x = 2.0
    elif kind == "raw-covariance":
        s = s.to_numpy()
    elif kind == "missing-label":
        s = s.iloc[:-1, :-1]
    elif kind == "duplicate-label":
        s.index = [s.index[0]] * len(s)
    elif kind == "cross-study":
        s.iloc[0, -1] = s.iloc[-1, 0] = 0.001
    elif kind == "asymmetric":
        s.iloc[0, 1] += 0.1
    elif kind == "not-spd":
        s.iloc[0, 0] = -1.0
    elif kind == "nonfinite-covariance":
        s.iloc[0, 0] = np.inf
    with pytest.raises(AnalysisError):
        run_fit(data, s)


def test_dataset_is_not_silently_collected():
    data, s = fixture()
    dataset = Dataset.from_frame(data)
    with pytest.raises(AnalysisError):
        run_fit(dataset, s)


def test_effect_study_term_and_early_workspace_limits():
    for groups, sizes in [(64, [4] * 63 + [5]), (65, [1] * 65)]:
        data, s = fixture(groups=groups, sizes=sizes)
        with pytest.raises(AnalysisError):
            run_fit(data, s)
    data, s = fixture(groups=10, sizes=[3] * 10)
    for i in range(8):
        data[f"x{i}"] = np.sin(np.arange(len(data)) * (i + 0.2))
    with pytest.raises(AnalysisError):
        run_fit(data, s, moderators=[f"x{i}" for i in range(8)])
    data, s = fixture(groups=40, sizes=[5] * 40)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        run_fit(data, s)
    assert error.value.code in {"workspace_limit", "resource_limit"}


def test_cpu_float64_independent_of_caller_dtype_device_and_rng():
    data, s = fixture()
    baseline = run_fit(data, s, model="study")
    rng = torch.get_rng_state().clone()
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            other = run_fit(data, s, model="study")
    finally:
        torch.set_default_dtype(previous)
    close(other["covariance"], baseline["covariance"], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)


def state_rehash(state):
    body = copy.deepcopy(state)
    body.pop("checksum", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    "kind",
    [
        "beta",
        "fit-covariance",
        "inference-covariance",
        "sampling",
        "effect-ids",
        "study-ids",
        "tau",
        "terms",
        "dimension",
        "model",
        "method",
        "inference",
        "extra",
    ],
)
def test_recomputed_checksum_does_not_admit_semantically_forged_state(kind):
    data, s = fixture()
    fitted = run_fit(data, s, model="study")
    state = copy.deepcopy(fit_state(fitted))
    if kind == "beta":
        state["beta"][0] += 0.3
    elif kind == "fit-covariance":
        state["fit_covariance"][0][0] *= 1.2
    elif kind == "inference-covariance":
        state["covariance"][0][0] *= 1.2
    elif kind == "sampling":
        state["s"][0][1] += 0.2
    elif kind == "effect-ids":
        state["effect_ids"][1] = state["effect_ids"][0]
    elif kind == "study-ids":
        state["study_ids"][0] = "not-a-declared-study"
    elif kind == "tau":
        state["tau2"] = -0.1
    elif kind == "terms":
        state["terms"] = ["fake", "x"]
    elif kind == "dimension":
        state["n_effects"] -= 1
    elif kind == "model":
        state["model"] = "unknown"
    elif kind == "method":
        state["method"] = "DL"
    elif kind == "inference":
        state["inference"] = "CR2"
    elif kind == "extra":
        state["unvalidated_future_field"] = True
    state["checksum"] = state_rehash(state)
    with pytest.raises(AnalysisError):
        oe.meta_dependent_predict(state, data={"x": [0.0, 1.0]})


def test_plain_checksum_tamper_rejected_and_no_runtime_oracle_imports(monkeypatch):
    data, s = fixture()
    fitted = run_fit(data, s, model="study")
    state = copy.deepcopy(fit_state(fitted))
    state["beta"][0] += 1.0
    with pytest.raises(AnalysisError):
        oe.meta_dependent_predict(state, data={"x": [0.0]})
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"scipy", "statsmodels"}:
            raise ImportError("Development oracle forbidden at runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    fitted = run_fit(data, s, model="study")
    robust = oe.meta_dependent_robust(fitted)
    terms = fit_state(fitted)["terms"]
    oe.meta_dependent_contrast(robust, contrasts=pd.DataFrame([[1.0, 0.2]], columns=terms))
    new = pd.DataFrame({"x": [0.0, 1.0], "future": ["new:A", "new:B"]})
    oe.meta_dependent_predict(fitted, data=new)
    oe.meta_dependent_predict_effect(fitted, data=new, study="future")
    oe.meta_dependent_diagnostics(fitted)


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "bool", "rank", "null-bool", "null-labels"]
)
def test_post_contrast_semantic_refusals(kind):
    data, s = fixture()
    fitted = run_fit(data, s)
    terms = fit_state(fitted)["terms"]
    contrast = pd.DataFrame([[1.0, 0.2], [0.3, 1.0]], columns=terms, index=["a", "b"])
    null = [0.0, 0.0]
    if kind == "missing":
        contrast = contrast.iloc[:, :1]
    elif kind == "duplicate":
        contrast.columns = [terms[0], terms[0]]
    elif kind == "bool":
        contrast[terms[0]] = True
    elif kind == "rank":
        contrast.iloc[1] = contrast.iloc[0]
    elif kind == "null-bool":
        null = [False, 0.0]
    elif kind == "null-labels":
        null = pd.Series([0.0, 0.0], index=["b", "a"])
    with pytest.raises(AnalysisError):
        oe.meta_dependent_contrast(fitted, contrasts=contrast, null=null)


@pytest.mark.parametrize(
    "kind", ["missing", "bool", "duplicate", "nonfinite", "dataset", "too-many"]
)
def test_post_prediction_data_semantic_refusals(kind):
    data, s = fixture()
    fitted = run_fit(data, s)
    new = pd.DataFrame({"x": [0.0, 1.0]})
    if kind == "missing":
        new.iloc[0, 0] = np.nan
    elif kind == "bool":
        new.x = True
    elif kind == "duplicate":
        new = pd.concat([new, new], axis=1)
    elif kind == "nonfinite":
        new.iloc[0, 0] = np.inf
    elif kind == "dataset":
        new = Dataset.from_frame(new)
    elif kind == "too-many":
        new = pd.DataFrame({"x": np.arange(257)})
    with pytest.raises(AnalysisError):
        oe.meta_dependent_predict(fitted, data=new)


@pytest.mark.parametrize(
    "kind", ["existing-study", "missing-study", "bool-study", "ambiguous-study"]
)
def test_post_new_study_label_semantics(kind):
    data, s = fixture()
    fitted = run_fit(data, s, model="study")
    labels = ["new:A", "new:B"]
    if kind == "existing-study":
        labels[0] = data.study.iloc[0]
    elif kind == "missing-study":
        labels[0] = None
    elif kind == "bool-study":
        labels[0] = True
    elif kind == "ambiguous-study":
        labels = [1, "1"]
    new = pd.DataFrame({"x": [0.0, 1.0], "future": labels})
    with pytest.raises(AnalysisError):
        oe.meta_dependent_predict_effect(fitted, data=new, study="future")


@pytest.mark.parametrize(
    "training_label,future_label,expected_code",
    [
        (1, 1.0, "ambiguous_identifier"),
        (1, "1", "ambiguous_identifier"),
        (0.0, -0.0, "ambiguous_identifier"),
        (1, 1, "existing_study"),
    ],
    ids=["numeric-alias", "display-alias", "signed-zero-alias", "existing-exact"],
)
def test_future_study_identity_is_checked_against_training_domain(
    training_label, future_label, expected_code
):
    data, sampling = fixture(seed=635629)
    labels = data.study.to_numpy(dtype=object, copy=True)
    labels[labels == labels[0]] = training_label
    data["study"] = pd.Series(labels, index=data.index, dtype=object)
    fitted = run_fit(data, sampling, model="study")
    assert fit_state(fitted)["studies"][0] == training_label
    future = pd.DataFrame({"x": [0.25], "future": [future_label]})
    with pytest.raises(AnalysisError) as error:
        oe.meta_dependent_predict_effect(fitted, data=future, study="future")
    assert error.value.code == expected_code


def test_prediction_duplicate_original_labels_remain_unambiguous_by_positions():
    data, s = fixture()
    fitted = run_fit(data, s)
    new = pd.DataFrame({"x": [0.0, 1.0]}, index=["same", "same"])
    result = oe.meta_dependent_predict(fitted, data=new)
    assert result["prediction"].row.tolist() == [0, 1]
    assert result["prediction"].original_label.tolist() == ["same", "same"]
    assert result["mean_covariance"].index.tolist() == [0, 1]


def test_full_latent_prediction_workspace_is_admitted_before_allocation():
    data, s = fixture(groups=3, sizes=[2, 2, 2])
    fitted = run_fit(data, s, model="study")
    new = pd.DataFrame({"x": np.arange(256) / 256, "future": ["new:A"] * 256})
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        oe.meta_dependent_predict_effect(fitted, data=new, study="future")
    assert error.value.code in {"workspace_limit", "resource_limit"}


def test_optimizer_free_saved_postestimation(monkeypatch):
    from openecon.econometrics.meta import dependent_kernels

    data, s = fixture()
    fitted = run_fit(data, s, model="study")

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Saved prediction/contrast/robust inference must not optimize tau again"
        )

    monkeypatch.setattr(dependent_kernels, "fit", forbidden)
    robust = oe.meta_dependent_robust(fitted)
    terms = fit_state(fitted)["terms"]
    oe.meta_dependent_contrast(robust, contrasts=pd.DataFrame([[1.0, 0.2]], columns=terms))
    new = pd.DataFrame({"x": [0.0, 1.0], "future": ["new:A", "new:B"]})
    oe.meta_dependent_predict(fitted, data=new)
    oe.meta_dependent_predict_effect(fitted, data=new, study="future")
