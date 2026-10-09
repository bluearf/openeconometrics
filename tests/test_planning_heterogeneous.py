"""Independent laws, full geometry, persistence and invalid-domain checks."""

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.stats.planning_heterogeneous import (
    power_unbalanced_anova,
    power_unequal_cluster_mean,
    power_welch,
)
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


def nct_probability(delta, df, alpha, alternative):
    cut = stats.t.isf(alpha / (2 if alternative == "two-sided" else 1), df)
    if alternative == "two-sided":
        return stats.nct.sf(cut, df, delta) + stats.nct.sf(cut, df, -delta)
    if alternative == "upper":
        return stats.nct.sf(cut, df, delta)
    # Exact nct sign symmetry avoids SciPy's NaN negative-CDF corner at huge df.
    return stats.nct.sf(cut, df, -delta)


@pytest.mark.parametrize(
    "n1,n2,sd1,sd2",
    [(2, 2, 1.0, 2.0), (5, 13, 3.0, 0.7), (40, 90, 0.8, 2.0), (19_000, 20_000, 1.0, 1.4)],
)
@pytest.mark.parametrize("effect", [0.0, 0.2, -0.5])
@pytest.mark.parametrize("alternative", ["two-sided", "upper", "lower"])
def test_welch_complete_satterthwaite_nct_oracle(n1, n2, sd1, sd2, effect, alternative):
    alpha = 0.013
    out = power_welch(effect, sd1=sd1, sd2=sd2, n1=n1, n2=n2, alpha=alpha, alternative=alternative)
    variance1, variance2 = sd1**2 / n1, sd2**2 / n2
    se = math.sqrt(variance1 + variance2)
    df = (variance1 + variance2) ** 2 / (variance1**2 / (n1 - 1) + variance2**2 / (n2 - 1))
    allocation = out["allocation"]
    assert_allclose(allocation.design_mean_variance, [variance1, variance2], rtol=2e-15)
    assert list(allocation.n) == [n1, n2]
    assert list(allocation.sd) == [sd1, sd2]
    for _, row in out["scenarios"].iterrows():
        assert_allclose(
            [row.design_standard_error, row.df, row.noncentrality],
            [se, df, row.effect / se],
            rtol=2e-15,
        )
        assert row.total_n == n1 + n2
        assert row.critical_value == pytest.approx(
            stats.t.isf(alpha / (2 if alternative == "two-sided" else 1), df), rel=2e-11
        )
        assert row.power == pytest.approx(
            nct_probability(row.effect / se, df, alpha, alternative), abs=2e-11
        )
    assert out["plan"].iloc[0].power == out["scenarios"].iloc[-1].power
    assert out.attrs["approximate"] is True
    assert out.attrs["covariance_known_at_analysis"] is False
    assert "not exact" in out.attrs["inference"]
    assert out.attrs["stochastic_draws"] == 0


def test_welch_published_unequal_sd_design_and_equal_variance_limit():
    # Stata13 power twomeans example2: .3 difference, SD .8/.7,100/arm gives >=80%.
    expected = power_welch(-0.3, sd1=0.8, sd2=0.7, n1=100, n2=100)["plan"].iloc[0].power
    previous = power_welch(-0.3, sd1=0.8, sd2=0.7, n1=99, n2=99)["plan"].iloc[0].power
    assert previous < 0.8 <= expected
    out = power_welch(0.4, sd1=1.3, sd2=1.3, n1=20, n2=20)["plan"].iloc[0]
    assert out.df == pytest.approx(38)
    assert out.power == pytest.approx(
        nct_probability(0.4 / (1.3 * math.sqrt(2 / 20)), 38, 0.05, "two-sided"), abs=2e-11
    )


@pytest.mark.parametrize(
    "means,sizes,sd",
    [
        ([0.0, 0.4, -0.2], [15, 20, 30], 1.0),
        ([5.0, 4.0], [1, 17], 2.0),
        ([1.0, 1.0, 1.0], [2, 3, 4], 0.7),
        ([-2.0, 0.0, 1.0, 0.2], [2, 3, 6, 9], 1.7),
    ],
)
def test_unbalanced_anova_complete_f_law_and_group_geometry(means, sizes, sd):
    original = copy.deepcopy((means, sizes))
    out = power_unbalanced_anova(means, sizes=sizes, sd=sd, alpha=0.017)
    n = sum(sizes)
    grand = np.average(means, weights=sizes)
    contributions = np.asarray(sizes) * (np.asarray(means) - grand) ** 2 / sd**2
    nc = contributions.sum()
    df1, df2 = len(sizes) - 1, n - len(sizes)
    cut = stats.f.isf(0.017, df1, df2)
    assert_allclose(out["allocation"]["mean"], means)
    assert_allclose(out["allocation"].n, sizes)
    assert_allclose(out["allocation"].grand_mean, grand, atol=2e-15)
    assert_allclose(out["allocation"].centered_mean, np.array(means) - grand, atol=2e-15)
    assert_allclose(out["allocation"].allocation_fraction, np.array(sizes) / n)
    assert_allclose(out["allocation"].noncentrality_contribution, contributions, atol=2e-15)
    for _, row in out["scenarios"].iterrows():
        assert row.df1 == df1 and row.df2 == df2 and row.total_n == n
        assert row.noncentrality == pytest.approx(nc * row.effect_multiplier**2)
        assert row.critical_value == pytest.approx(cut, rel=2e-11)
        oracle = (
            stats.f.sf(cut, df1, df2)
            if row.noncentrality == 0
            else stats.ncf.sf(cut, df1, df2, row.noncentrality)
        )
        assert row.power == pytest.approx(oracle, abs=2e-11)
    assert (means, sizes) == original
    assert out.attrs["approximate"] is False
    assert out.attrs["covariance_known_at_analysis"] is False


def test_anova_translation_scaling_permutation_and_balanced_limit():
    args = dict(means=[0.0, 0.3, -0.4], sizes=[12, 15, 18], sd=0.8)
    original = power_unbalanced_anova(**args)["plan"].iloc[0].power
    altered = power_unbalanced_anova(
        [7 + 4 * x for x in args["means"]], sizes=args["sizes"], sd=3.2
    )
    permuted = power_unbalanced_anova([-0.4, 0.0, 0.3], sizes=[18, 12, 15], sd=0.8)
    assert altered["plan"].iloc[0].power == pytest.approx(original, abs=2e-15)
    assert permuted["plan"].iloc[0].power == pytest.approx(original, abs=2e-15)
    means = [-0.4, 0.0, 0.4]
    out = power_unbalanced_anova(means, sizes=[20] * 3, sd=1.0)["plan"].iloc[0]
    assert out.noncentrality == pytest.approx(60 * np.mean(np.square(means)))
    assert out.df2 == 57


@pytest.mark.parametrize("weighting", ["participant", "cluster"])
@pytest.mark.parametrize("alternative", ["two-sided", "upper", "lower"])
@pytest.mark.parametrize("iccs", [(0.0, 0.0), (0.15, 0.4), (1.0, 1.0)])
def test_unequal_cluster_full_block_covariance_oracle(weighting, alternative, iccs):
    vectors, sds = ([2, 5, 9], [3, 8]), (1.2, 0.8)
    out = power_unequal_cluster_mean(
        -0.4,
        sd1=sds[0],
        sd2=sds[1],
        sizes1=vectors[0],
        sizes2=vectors[1],
        icc1=iccs[0],
        icc2=iccs[1],
        weighting=weighting,
        alternative=alternative,
    )
    variances = []
    for arm, (sizes, sd, rho) in enumerate(zip(vectors, sds, iccs), 1):
        participants = sum(sizes)
        blocks, weights = [], []
        expected = []
        for cluster, size in enumerate(sizes, 1):
            block = sd**2 * ((1 - rho) * np.eye(size) + rho * np.ones((size, size)))
            weight = size / participants if weighting == "participant" else 1 / len(sizes)
            per_person = np.repeat(weight / size, size)
            contribution = per_person @ block @ per_person
            blocks.append(block)
            weights.extend(per_person)
            expected.append([weight, block.sum() / size**2, contribution, weight**2 * sd**2 / size])
        full = np.zeros((participants, participants))
        start = 0
        for block in blocks:
            length = len(block)
            full[start : start + length, start : start + length] = block
            start += length
        variance = np.array(weights) @ full @ weights
        variances.append(variance)
        rows = out["allocation"].query("arm == @arm")
        assert list(rows["size"]) == sizes
        assert_allclose(
            rows[
                [
                    "weight",
                    "cluster_mean_variance",
                    "variance_contribution",
                    "independent_variance_contribution",
                ]
            ],
            expected,
            rtol=3e-15,
        )
    assert_allclose(out.attrs["known_mean_covariance"], np.diag(variances), rtol=3e-15)
    se = math.sqrt(sum(variances))
    for _, row in out["scenarios"].iterrows():
        shift = row.effect / se
        cut = stats.norm.isf(0.05 / (2 if alternative == "two-sided" else 1))
        expected = (
            stats.norm.sf(cut - shift) + stats.norm.cdf(-cut - shift)
            if alternative == "two-sided"
            else stats.norm.sf(cut - (shift if alternative == "upper" else -shift))
        )
        assert row.design_standard_error == pytest.approx(se)
        assert row.normal_shift == pytest.approx(shift)
        assert row.power == pytest.approx(expected, abs=3e-15)
        assert row.critical_value == pytest.approx(cut, abs=3e-15)
    assert out.attrs["covariance_known_at_analysis"] is True
    assert out.attrs["approximate"] is False


def test_equal_cluster_and_iid_limits_weighting_target_difference():
    base = dict(effect=0.3, sd1=1.0, sd2=1.0, sizes1=[20] * 4, sizes2=[20] * 6, icc1=0.1, icc2=0.1)
    part = power_unequal_cluster_mean(**base)["plan"].iloc[0]
    cluster = power_unequal_cluster_mean(**base, weighting="cluster")["plan"].iloc[0]
    assert part.power == cluster.power
    assert part.design_effect1 == pytest.approx(1 + (20 - 1) * 0.1)
    iid = power_unequal_cluster_mean(**(base | dict(icc1=0.0, icc2=0.0)))["plan"].iloc[0]
    assert iid.variance1 == pytest.approx(1 / 80)
    unequal = base | dict(sizes1=[2, 5, 53], sizes2=[4, 21, 75])
    first = power_unequal_cluster_mean(**unequal)["plan"].iloc[0]
    second = power_unequal_cluster_mean(**unequal, weighting="cluster")["plan"].iloc[0]
    assert abs(first.variance1 - second.variance1) > 0.001
    # Participant-weighting exact design effect from actual size moments, not mean size alone.
    sizes = np.array(unequal["sizes1"])
    assert first.design_effect1 == pytest.approx(1 + 0.1 * ((sizes**2).sum() / sizes.sum() - 1))


@pytest.mark.parametrize(
    "function,args",
    [
        (power_welch, dict(effect=0.4, sd1=1.0, sd2=2.0, n1=12, n2=25)),
        (power_unbalanced_anova, dict(means=[0.0, 0.2, 0.5], sizes=[10, 20, 15], sd=1.0)),
        (
            power_unequal_cluster_mean,
            dict(effect=0.4, sd1=1.0, sd2=2.0, sizes1=[2, 4], sizes2=[5, 9], icc1=0.2, icc2=0.1),
        ),
    ],
)
def test_full_settings_summary_json_owned_file_readback_and_cpu(function, args, tmp_path):
    import torch

    rng = torch.random.get_rng_state().clone()
    with torch.device("meta"):
        out = function(**args)
    assert torch.equal(rng, torch.random.get_rng_state())
    path = tmp_path / "prospective.json"
    path.write_text(summary_state(out))
    restored = restore_summary(path.read_text())
    assert restored.attrs == out.attrs
    settings = dict(zip(out["settings"].setting, out["settings"].json))
    assert json.loads(settings["solver"]) == out.attrs["solver"]
    assert json.loads(settings["inference"]) == out.attrs["inference"]
    assert json.loads(settings["scenario_definition"]) == out.attrs["scenario_definition"]
    for name, frame in out.items():
        pd.testing.assert_frame_equal(restored[name], frame, check_dtype=False)
    assert restored.to_latex() == out.to_latex()
    assert json.loads(path.read_text())["attrs"]["device"] == "cpu"
    assert "prospective" in out.to_latex()


@pytest.mark.parametrize(
    "updates",
    [
        dict(n1=1),
        dict(n2=2.5),
        dict(n1=True),
        dict(n1=20001),
        dict(sd1=0),
        dict(sd2=float("inf")),
        dict(effect=True),
        dict(effect=1e100, sd1=1e-100, sd2=1e-100),
        dict(alpha=0),
        dict(alternative="both"),
    ],
)
def test_welch_invalid_and_probability_domain(updates):
    with pytest.raises(AnalysisError):
        power_welch(**(dict(effect=0.3, sd1=1.0, sd2=2.0, n1=20, n2=25) | updates))


@pytest.mark.parametrize(
    "updates",
    [
        dict(means=[0]),
        dict(sizes=[1, 2]),
        dict(sizes=[2, True, 3]),
        dict(sizes=[1, 1, 1]),
        dict(sizes=[10000] * 3),
        dict(means=[0, float("nan"), 1]),
        dict(sd=0),
        dict(means=[0, 1e100, -1e100], sd=1e-100),
        dict(means=[0] * 31, sizes=[2] * 31),
    ],
)
def test_anova_invalid_rank_size_and_noncentrality_domain(updates):
    with pytest.raises(AnalysisError):
        power_unbalanced_anova(
            **(dict(means=[0.0, 0.3, -0.1], sizes=[10, 20, 30], sd=1.0) | updates)
        )


@pytest.mark.parametrize(
    "updates",
    [
        dict(sizes1=[]),
        dict(sizes1=[0, 2]),
        dict(sizes1=[True]),
        dict(sizes2=[2.5]),
        dict(sizes1=[10_000_000, 1]),
        dict(sizes1=[1] * 10001),
        dict(icc1=-0.01),
        dict(icc2=1.01),
        dict(icc1=float("nan")),
        dict(weighting="optimal"),
        dict(sd2=0),
    ],
)
def test_cluster_invalid_prespecified_geometry(updates):
    with pytest.raises(AnalysisError):
        power_unequal_cluster_mean(
            **(
                dict(effect=0.3, sd1=1.0, sd2=2.0, sizes1=[2, 4], sizes2=[3, 8], icc1=0.2, icc2=0.1)
                | updates
            )
        )


def test_cluster_workspace_admitted_before_geometry(monkeypatch):
    monkeypatch.setattr(
        "openecon.econometrics.stats.planning_heterogeneous._arm",
        lambda *a, **k: pytest.fail("geometry must not run before admission"),
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        power_unequal_cluster_mean(
            0.3, sd1=1.0, sd2=1.0, sizes1=[1] * 2000, sizes2=[1] * 2000, icc1=0.1, icc2=0.2
        )
    assert error.value.code == "workspace_limit"
