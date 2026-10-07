import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle
from test_econ_teffects_csdid import make_panel, pairs, wide, m_oracle


@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
def test_unbalanced_pair_complete_covariate_comparisons_against_independent_m_estimation(method):
    df = make_panel(n=180)
    df = df.loc[np.random.default_rng(51).uniform(size=len(df)) > 0.12].reset_index(drop=True)
    result = oe.csdid(
        data=df,
        y="y",
        group="id",
        time="t",
        treatment_time="first",
        x=["x"],
        sample="unbalanced",
        method=method,
    )
    for coef, decision in zip(result.coefficients, result.extra["sample_decisions"], strict=True):
        g, t, b = decision["cohort"], decision["period"], decision["base"]
        two = df.loc[df.t.isin([t, b])].pivot(index="id", columns="t", values="y").dropna()
        units = df.groupby("id").first().loc[two.index]
        treated = units["first"].to_numpy() == g
        controls = units["first"].isna().to_numpy()
        use = treated | controls
        estimate, se = m_oracle(
            (two[t] - two[b]).to_numpy()[use],
            units.x.to_numpy()[use, None],
            treated[use].astype(float),
            method,
        )
        assert_allclose([coef.estimate, coef.std_error], [estimate, se], rtol=3e-6, atol=1e-9)
        assert decision["treated_units"] == treated.sum()


def test_repeated_cross_sections_att_full_covariance_pretrend_and_serialization():
    df = make_panel(n=150).drop(columns="id")
    result = oe.csdid(
        data=df, y="y", time="t", treatment_time="first", sample="repeated_cross_section"
    )
    influence = np.zeros((len(df), len(result.coefficients)))
    estimates = []
    for j, decision in enumerate(result.extra["sample_decisions"]):
        g, t, b = decision["cohort"], decision["period"], decision["base"]
        value = 0
        for treated, period, sign in [(True, t, 1), (True, b, -1), (False, t, -1), (False, b, 1)]:
            group = (df["first"] == g) if treated else df["first"].isna()
            used = (group & (df.t == period)).to_numpy()
            mean = df.y[used].mean()
            value += sign * mean
            influence[used, j] = sign * (df.y[used] - mean) / used.sum()
        estimates.append(value)
    assert_allclose([c.estimate for c in result.coefficients], estimates, rtol=1e-11)
    cov = influence.T @ influence
    assert_allclose(result.covariance_matrix, cov, rtol=1e-11, atol=1e-14)
    pre = [j for j, d in enumerate(result.extra["sample_decisions"]) if d["period"] < d["cohort"]]
    b = np.array(estimates)[pre]
    assert_allclose(
        result.tests["pretrends"]["statistic"],
        b @ np.linalg.solve(cov[np.ix_(pre, pre)], b),
        rtol=1e-9,
    )
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored.extra["sample_contract"] == "repeated_cross_section"


def test_shared_multiplier_and_simultaneous_families_include_estimated_cohort_weight_if():
    df = make_panel(n=140)
    torch.manual_seed(983)
    state = torch.get_rng_state().clone()
    result = oe.csdid(
        data=df,
        y="y",
        group="id",
        time="t",
        treatment_time="first",
        bootstrap_reps=399,
        seed=11,
        uniform=True,
    )
    assert torch.equal(state, torch.get_rng_state())
    y, first, _ = wide(df)
    info = pairs(first, sorted(df.t.unique()), "never", "varying")
    att = np.array([c.estimate for c in result.coefficients])
    phi = np.zeros((len(first), len(info)))
    for j, (g, t, ti, b, treated, control) in enumerate(info):
        dy = y[:, ti] - y[:, b]
        phi[treated, j] = (dy[treated] - dy[treated].mean()) / treated.sum()
        phi[control, j] = -(dy[control] - dy[control].mean()) / control.sum()
    draw = torch.randn(
        (399, len(first)), generator=torch.Generator().manual_seed(11), dtype=torch.float64
    ).numpy()
    se = np.sqrt((phi**2).sum(0))
    critical = np.quantile(np.max(np.abs(draw @ phi / se), axis=1), 0.95)
    assert_allclose(result.inference["families"]["att_gt"]["critical_value"], critical, rtol=1e-11)
    for j, row in enumerate(result.extra["att_gt_bands"]):
        assert_allclose(
            [row["uniform_ci_low"], row["uniform_ci_high"]],
            [att[j] - critical * se[j], att[j] + critical * se[j]],
            rtol=1e-11,
        )
    for family in ["dynamic", "calendar", "group"]:
        contributions = []
        for row in result.extra[family]:
            chosen = np.array(
                [
                    t - g == row["label"]
                    if family == "dynamic"
                    else t == row["label"] and t >= g
                    if family == "calendar"
                    else g == row["label"] and t >= g
                    for g, t, *_ in info
                ]
            )
            if family == "group":
                contributions.append(phi[:, chosen].mean(1))
                continue
            shares = np.array(
                [
                    np.mean(first == g) if choose else 0
                    for (g, *_), choose in zip(info, chosen, strict=True)
                ]
            )
            share_if = np.column_stack(
                [
                    ((first == g) - np.mean(first == g)) / len(first)
                    if choose
                    else np.zeros(len(first))
                    for (g, *_), choose in zip(info, chosen, strict=True)
                ]
            )
            total = shares.sum()
            weights_if = (share_if * total - shares * share_if.sum(1, keepdims=True)) / total**2
            contributions.append(phi @ (shares / total) + weights_if @ att)
        influence = np.column_stack(contributions)
        se = np.sqrt((influence**2).sum(0))
        expected = np.quantile(np.max(np.abs(draw @ influence / se), axis=1), 0.95)
        assert_allclose(result.inference["families"][family]["critical_value"], expected, rtol=1e-9)
    # Internal tensors are removed before a persisted result is built.
    assert "_influence" not in result.model_dump_json()


def test_anticipation_baselines_controls_and_excluded_pretrend_window():
    df = make_panel(n=150)
    result = oe.csdid(
        data=df,
        y="y",
        group="id",
        time="t",
        treatment_time="first",
        anticipation=1,
        control="notyet",
    )
    y, first, _ = wide(df)
    for coef, d in zip(result.coefficients, result.extra["sample_decisions"], strict=True):
        g, t, b = d["cohort"], d["period"], d["base"]
        assert b == (g - 2 if t >= g - 1 else t - 1)
        treated = first == g
        controls = (np.isnan(first) | (first > max(t, b) + 1)) & ~treated
        dy = y[:, int(t - 1)] - y[:, int(b - 1)]
        expected = dy[treated].mean() - dy[controls].mean()
        assert_allclose(coef.estimate, expected, rtol=1e-11)


def test_sample_and_multiplier_domain_guards():
    df = make_panel(n=80)
    common = dict(data=df, y="y", time="t", treatment_time="first")
    for options in [
        dict(uniform=True),
        dict(bootstrap_reps=1),
        dict(sample="repeated_cross_section", x=["x"]),
    ]:
        with pytest.raises(AnalysisError):
            oe.csdid(**common, **options)
    fit = oe.csdid(**common, group="id", bootstrap_reps=30, uniform=True)
    assert isinstance(fit.inference["seed"], int)
