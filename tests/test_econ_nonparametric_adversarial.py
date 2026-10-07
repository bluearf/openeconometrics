"""Nonparametric family: adversarial inputs (degenerate samples, extreme scales, odd types)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.nonparametric import exact as exact_kernels


def finite_attrs(result) -> dict:
    attrs = json.loads(json.dumps(result.attrs))
    assert attrs == result.attrs

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            assert not (isinstance(value, float) and not math.isfinite(value))

    walk(attrs)
    return attrs


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(61)
    n = 60
    g = rng.integers(0, 2, size=n)
    return pd.DataFrame({"x": rng.normal(size=n) + 0.5 * g, "z": rng.normal(size=n),
                         "g": g, "k": rng.integers(0, 3, size=n)})


def test_values_beyond_float_range_are_refused_with_advice():
    huge = {"v": [1e300, -1e300, 5e299, 2.0, 1.0, 3.0, 4.0, 8.0], "g": [0, 1] * 4}
    for call in (lambda: oe.ranksum(huge, "v", "g"), lambda: oe.ksmirnov(huge, "v"),
                 lambda: oe.swilk(huge, "v"), lambda: oe.sktest(huge, "v"),
                 lambda: oe.signrank(huge, "v"), lambda: oe.roc(huge, "g", "v"),
                 lambda: oe.friedman({"a": huge["v"], "b": huge["g"]}, ["a", "b"])):
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == "numerical_failure" and "Rescale" in str(error.value)


@pytest.mark.parametrize("scale, shift", [(1e-120, 0.0), (1e6, 1e9), (1e120, 0.0), (-3.0, 7.0)])
def test_location_and_scale_invariance(frame, scale, shift):
    moved = frame.assign(x=frame.x * scale + shift, z=frame.z * scale + shift)
    flip = scale < 0
    for name, call in {
        "ranksum": lambda d: oe.ranksum(d, "x", "g"),
        "kwallis": lambda d: oe.kwallis(d, "x", "k"),
        "signrank": lambda d: oe.signrank(d, "x", "z"),
        "friedman": lambda d: oe.friedman(d, ["x", "z"]),
        "ksmirnov": lambda d: oe.ksmirnov(d, "x"),
        "ks2": lambda d: oe.ksmirnov(d, "x", by="g"),
        "swilk": lambda d: oe.swilk(d, "x"),
        "sfrancia": lambda d: oe.sfrancia(d, "x"),
        "sktest": lambda d: oe.sktest(d, "x"),
        "roccomp": lambda d: oe.roccomp(d, "g", ["x", "z"]),
    }.items():
        base, other = finite_attrs(call(frame)), finite_attrs(call(moved))
        tolerance = 1e-9 if abs(shift) < 1 else 2e-5       # 1e9 + small values lose digits
        if name in ("roccomp",) and flip:
            assert other["statistic"] == pytest.approx(base["statistic"], rel=1e-6)
            continue
        assert other["p_value"] == pytest.approx(base["p_value"], rel=tolerance, abs=1e-12), name
        if "statistic" in base and base["statistic"] is not None:
            assert other["statistic"] == pytest.approx(base["statistic"], rel=tolerance), name


def test_row_order_does_not_matter_for_order_free_tests(frame):
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    calls = [lambda d: oe.ranksum(d, "x", "g"), lambda d: oe.jonckheere(d, "x", "k"),
             lambda d: oe.median_test(d, "x", "k"), lambda d: oe.signrank(d, "x", "z"),
             lambda d: oe.crosstab(d, "g", "k"), lambda d: oe.roc(d, "g", "x"),
             lambda d: oe.ksmirnov(d, "x", by="g"), lambda d: oe.sktest(d, "x")]
    for call in calls:
        first, second = call(frame).attrs, call(shuffled).attrs
        assert second["p_value"] == pytest.approx(first["p_value"], rel=1e-10)


def test_swapping_the_groups_mirrors_directional_statistics(frame):
    swapped = frame.assign(g=1 - frame.g)
    a, b = oe.ranksum(frame, "x", "g").attrs, oe.ranksum(swapped, "x", "g").attrs
    assert b["z"] == pytest.approx(-a["z"]) and b["u1"] == pytest.approx(a["u2"])
    assert b["hl_estimate"] == pytest.approx(-a["hl_estimate"])
    assert (b["hl_ci_low"], b["hl_ci_high"]) == pytest.approx((-a["hl_ci_high"], -a["hl_ci_low"]))
    ks_a, ks_b = oe.ksmirnov(frame, "x", by="g").attrs, oe.ksmirnov(swapped, "x", by="g").attrs
    assert ks_b["d_plus"] == pytest.approx(-ks_a["d_minus"])
    assert oe.roc(swapped, "g", "x").attrs["auc"] == pytest.approx(
        1 - oe.roc(frame, "g", "x").attrs["auc"])
    paired_a = oe.signrank(frame, "x", "z").attrs
    paired_b = oe.signrank(frame, "z", "x").attrs
    assert paired_b["z"] == pytest.approx(-paired_a["z"])
    assert paired_b["t_plus"] == pytest.approx(paired_a["t_minus"])


def test_tiny_samples_never_crash():
    results = [
        oe.ranksum({"v": [1.0, 2.0], "g": [0, 1]}, "v", "g"),
        oe.ranksum({"v": [1.0, 2, 3, 4], "g": [0, 1, 1, 1]}, "v", "g"),
        oe.kwallis({"v": [1.0, 2, 3], "g": [0, 1, 2]}, "v", "g", pairwise=True),
        oe.median_test({"v": [1.0, 2.0], "g": [0, 1]}, "v", "g"),
        oe.jonckheere({"v": [1.0, 2.0, 3.0], "g": [0, 1, 2]}, "v", "g"),
        oe.signrank({"a": [1.0]}, "a"), oe.signtest({"a": [1.0]}, "a"),
        oe.friedman({"a": [1.0], "b": [2.0], "c": [3.0]}, ["a", "b", "c"]),
        oe.ksmirnov({"x": [0.3]}, "x", params=[0, 1]),
        oe.ksmirnov({"x": [0.3, 0.5], "g": [0, 1]}, "x", by="g"),
        oe.swilk({"x": [1.0, 2.0, 3.0]}, "x"), oe.swilk({"x": [1.0, 1.0, 3.0]}, "x"),
        oe.bitest({"y": [0, 0, 0]}, "y"), oe.prtest({"y": [1, 1, 1]}, "y"),
        oe.crosstab({"r": [1], "c": [2]}, "r", "c"),
        oe.roc({"y": [0, 1], "m": [1.0, 2.0]}, "y", "m"),
        oe.tabulate({"x": [True]}, "x"),
    ]
    for result in results:
        attrs = finite_attrs(result)
        if attrs.get("p_value") is not None:
            assert 0.0 <= attrs["p_value"] <= 1.0
    assert oe.ranksum({"v": [1.0, 2.0], "g": [0, 1]}, "v", "g").attrs["p_exact"] == 1.0
    assert oe.swilk({"x": [1.0, 2.0, 3.0]}, "x").attrs["p_value"] == pytest.approx(1.0)
    assert oe.swilk({"x": [1.0, 1.0, 3.0]}, "x").attrs["statistic"] == pytest.approx(0.75)
    assert oe.ksmirnov({"x": [0.3]}, "x", params=[0, 1]).attrs["p_exact"] == pytest.approx(
        stats.kstwo.sf(stats.norm.cdf(0.3), 1))


def test_category_types_booleans_dates_and_mixed_values():
    booleans = oe.crosstab({"r": [True, False, True, False, True],
                            "c": [True, True, False, False, True]}, "r", "c")
    assert booleans.attrs["row_levels"] == [False, True]
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"])
    by_date = oe.crosstab({"r": dates, "c": [1, 2, 2, 1]}, "r", "c")
    assert by_date.attrs["row_levels"] == ["2020-01-01T00:00:00", "2020-01-02T00:00:00"]
    mixed = oe.crosstab({"r": [0, "a", 1, "a"], "c": [0, 0, 1, 1]}, "r", "c")
    assert mixed["counts"].shape == (4, 3)
    for result in (booleans, by_date, mixed):
        finite_attrs(result)
    by_bool = oe.ranksum({"v": [1.0, 2, 3, 4, 6], "g": [True, False, True, False, True]}, "v",
                         "g")
    assert by_bool.attrs["groups"] == [False, True]
    nullable = pd.DataFrame({"v": pd.array([1, 2, None, 4, 5, 7], dtype="Int64"),
                             "g": pd.array(["a", "b", "b", None, "a", "b"], dtype="string")})
    result = oe.ranksum(nullable, "v", "g")
    assert result.attrs["n"] == 4 and result.attrs["n_dropped"] == 2
    assert oe.roc({"y": [True, False, True, False, True], "m": [3.0, 1, 2, 2, 5]}, "y",
                  "m").attrs["positive"] is True


def test_perfect_and_degenerate_tables():
    perfect = oe.crosstab({"r": [0, 0, 1, 1, 2, 2], "c": [0, 0, 1, 1, 2, 2]}, "r", "c",
                          exact=True)
    measures = perfect["measures"]
    assert measures.loc["gamma", "value"] == 1.0 and measures.loc["gamma", "ase"] == 0.0
    assert math.isnan(measures.loc["gamma", "t"])                 # no null variance estimate
    assert math.isnan(measures.loc["uncertainty_row", "t"])
    assert measures.loc["kappa", "value"] == pytest.approx(1.0)
    assert perfect.attrs["p_exact"] == pytest.approx(6 / 90)        # 3! of 6!/(2!2!2!) tables
    assert perfect.attrs["cells_below_5"] == 9
    finite_attrs(perfect)
    zero_cell = oe.crosstab({"r": [0, 0, 1, 1, 1], "c": [0, 0, 1, 1, 0]}, "r", "c")
    assert "odds_ratio" not in zero_cell["risk"].index
    assert "risk_ratio_column_1" in zero_cell["risk"].index
    assert zero_cell["measures"].loc["gamma", "value"] == 1.0
    independent = oe.crosstab({"r": [0, 0, 1, 1], "c": [0, 1, 0, 1]}, "r", "c")
    assert independent.attrs["statistic"] == 0.0 and independent.attrs["p_exact"] == 1.0
    assert independent["measures"].loc["phi", "value"] == 0.0
    finite_attrs(independent)
    weighted_out = oe.crosstab({"r": [0, 0, 1, 1], "c": [0, 1, 0, 1], "w": [2.0, 0, 0, 3.0]},
                               "r", "c", weights="w")
    assert weighted_out.attrs["n"] == 5 and weighted_out["measures"].loc["phi", "value"] == 1.0


def test_heavily_tied_and_constant_within_group_data():
    data = {"v": [1.0] * 6 + [2.0] * 6, "g": [0] * 6 + [1] * 6}
    separated = oe.ranksum(data, "v", "g")
    assert separated.attrs["u1"] == 0.0 and separated.attrs["ties"] is True
    reference = stats.mannwhitneyu([1.0] * 6, [2.0] * 6, method="asymptotic",
                                   use_continuity=False)
    assert separated.attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
    assert separated.attrs["hl_estimate"] == -1.0
    forced = oe.ranksum(data, "v", "g", exact=True).attrs
    assert forced["p_exact"] == pytest.approx(2 / math.comb(12, 6))
    kw = oe.kwallis(data, "v", "g")
    assert kw.attrs["statistic"] == pytest.approx(stats.kruskal([1.0] * 6, [2.0] * 6).statistic)
    binary = oe.ksmirnov(data, "v", by="g").attrs
    assert binary["statistic"] == 1.0 and binary["exact"] is False


def test_exact_and_asymptotic_p_values_converge():
    rng = np.random.default_rng(62)
    x, y = rng.normal(size=45), rng.normal(size=45) + 0.4
    data = {"v": np.r_[x, y], "g": [0] * 45 + [1] * 45}
    attrs = oe.ranksum(data, "v", "g", exact=True).attrs
    assert attrs["p_exact"] == pytest.approx(attrs["p_value_continuity"], abs=2e-3)
    assert attrs["p_exact"] == pytest.approx(stats.mannwhitneyu(x, y, method="exact").pvalue,
                                             rel=1e-9)
    paired = oe.signrank({"a": x, "b": y}, "a", "b", exact=True).attrs
    assert paired["p_exact"] == pytest.approx(paired["p_value"], abs=5e-3)
    ks = oe.ksmirnov({"x": x}, "x", params=[0, 1]).attrs
    assert ks["p_exact"] == pytest.approx(ks["p_asymptotic"], abs=0.05)
    runs = oe.runtest({"y": x}, "y", exact=True, continuity=True).attrs
    assert runs["p_exact"] == pytest.approx(runs["p_value"], abs=0.03)


def test_exact_distributions_are_proper_and_monotone():
    grid = np.linspace(0.01, 0.99, 50)
    for n in (1, 2, 7, 30, 150):
        values = np.array([exact_kernels.kolmogorov_sf_exact(n, d) for d in grid])
        assert np.all((values >= 0) & (values <= 1)) and np.all(np.diff(values) <= 1e-12)
    for n1, n2 in ((3, 5), (10, 10), (17, 4)):
        previous = 1.0
        for h in range(0, n1 * n2 + 2):
            p = exact_kernels.smirnov_exact(n1, n2, h)
            assert 0.0 <= p <= previous + 1e-12
            previous = p
        assert exact_kernels.smirnov_exact(n1, n2, n1 * n2 + 1) == 0.0
    support, pmf = exact_kernels.runs_distribution(1, 9)
    assert support.tolist() == [2.0, 3.0] and pmf.tolist() == pytest.approx([0.2, 0.8])


def test_wide_sparse_group_structures():
    rng = np.random.default_rng(63)
    n = 3000
    data = pd.DataFrame({"v": np.round(rng.normal(size=n), 2), "g": rng.integers(0, 150, size=n)})
    kw = oe.kwallis(data, "v", "g")
    reference = stats.kruskal(*[group.v.to_numpy() for _, group in data.groupby("g")])
    assert kw.attrs["statistic"] == pytest.approx(reference.statistic, rel=1e-10)
    jt = oe.jonckheere(data, "v", "g")
    kendall = stats.kendalltau(data.g, data.v, method="asymptotic")
    assert jt.attrs["p_value"] == pytest.approx(kendall.pvalue, rel=1e-7)
    with pytest.raises(AnalysisError) as error:
        oe.kwallis(data.assign(g=np.arange(n) // 10), "v", "g", pairwise=True)
    assert error.value.code == "too_many_groups"


# ---- verification pass: regressions and further adversarial inputs --------------------------


def test_fisher_exact_on_very_large_weighted_tables_is_bounded_in_memory():
    """Regression: a 2x2 table with totals of 1e9+ used to tabulate the whole support."""
    from openecon.econometrics.nonparametric import tables as table_kernels
    from openecon.econometrics.nonparametric.exact import MAX_SUPPORT, hypergeometric

    cells = {"r": [0, 0, 1, 1], "c": [0, 1, 0, 1]}
    # Total 1.2e8: the support (4e7 points) exceeds MAX_SUPPORT, so only a window is tabulated.
    counts = np.array([[4e7 + 3000, 2e7], [4e7, 2e7]])
    assert min(counts[0].sum(), counts[:, 0].sum()) > MAX_SUPPORT
    result = oe.crosstab({**cells, "w": counts.reshape(-1)}, "r", "c", weights="w")
    reference = stats.fisher_exact(counts)
    assert result.attrs["p_exact"] == pytest.approx(reference.pvalue, rel=1e-6)
    assert result.attrs["p_exact_less"] == pytest.approx(
        stats.fisher_exact(counts, alternative="less").pvalue, rel=1e-6)
    assert result.attrs["p_exact_greater"] == pytest.approx(
        stats.fisher_exact(counts, alternative="greater").pvalue, rel=1e-6)
    # An observed cell far outside the window: the p-value underflows to 0, tails are 0 / 1.
    extreme = torch.tensor([[3e9 + 5e4, 2e9], [2.5e9, 1.7e9]], dtype=torch.float64)
    assert table_kernels.fisher_2x2(extreme) == (0.0, 1.0, 0.0)
    assert table_kernels.fisher_2x2(extreme.flip(1)) == (0.0, 0.0, 1.0)
    # Totals of 1e13: not even the window fits. Default: skipped with a note; forced: refused.
    giant = {**cells, "w": [3e12, 2e12, 2.5e12, 1.7e12]}
    skipped = oe.crosstab(giant, "r", "c", weights="w")
    assert "p_exact" not in skipped.attrs and "fisher_exact" not in skipped["tests"].index
    assert any("Fisher" in note for note in skipped.attrs["notes"])
    assert skipped.attrs["statistic"] > 0 and "odds_ratio" in skipped["risk"].index
    finite_attrs(skipped)
    with pytest.raises(AnalysisError) as error:
        oe.crosstab(giant, "r", "c", weights="w", exact=True)
    assert error.value.code == "exact_unavailable"
    # The ratio recursion agrees with SciPy's pmf on ordinary tables, including the edges
    # of the support.
    for row1, col1, total in ((5, 7, 20), (10, 3, 12), (0, 4, 9), (9, 9, 9), (300, 500, 1000),
                              (1, 1, 2), (40, 70, 75)):
        low, pmf = hypergeometric(row1, col1, total)
        assert low == max(0, row1 + col1 - total)
        support = np.arange(low, low + pmf.numel())
        np.testing.assert_allclose(pmf.numpy(), stats.hypergeom.pmf(support, total, col1, row1),
                                   rtol=1e-11, atol=1e-300)


def test_crosstab_refuses_tables_too_large_to_print():
    rng = np.random.default_rng(64)
    n = 4000
    frame = pd.DataFrame({"a": rng.integers(0, 80, n), "b": rng.integers(0, 80, n),
                          "l": rng.integers(0, 400, n)})
    with pytest.raises(AnalysisError) as error:
        oe.crosstab(frame, "a", "b", layer="l")                 # 80 x 80 x 400 = 2.56e6 cells
    assert error.value.code == "too_many_categories"
    assert oe.crosstab(frame.assign(l=frame.l % 50), "a", "b", layer="l").attrs["n"] == n


def test_non_numeric_outcome_message_names_the_remedy():
    text = {"y": list("abcdef"), "g": [0, 0, 0, 1, 1, 1]}
    dates = {"y": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), "g": [0, 1, 1]}
    for data in (text, dates):
        for call in (lambda d: oe.ranksum(d, "y", "g"), lambda d: oe.swilk(d, "y"),
                     lambda d: oe.signrank(d, "y"), lambda d: oe.roc(d, "g", "y"),
                     lambda d: oe.friedman(d, ["y", "g"]), lambda d: oe.runtest(d, "y")):
            with pytest.raises(AnalysisError) as error:
                call(data)
            assert error.value.code == "non_numeric_column"
            assert "crosstab" in str(error.value) and "predictor" not in str(error.value)


PROCEDURES = {
    "ranksum": lambda d: oe.ranksum(d, "y", "g"),
    "kwallis": lambda d: oe.kwallis(d, "y", "g", pairwise=True),
    "median_test": lambda d: oe.median_test(d, "y", "g"),
    "jonckheere": lambda d: oe.jonckheere(d, "y", "g"),
    "signrank": lambda d: oe.signrank(d, "y", "x"),
    "signtest": lambda d: oe.signtest(d, "y", "x"),
    "mcnemar": lambda d: oe.mcnemar(d, "g", "b"),
    "symmetry": lambda d: oe.symmetry(d, "g", "b"),
    "friedman": lambda d: oe.friedman(d, ["y", "x"]),
    "cochran_q": lambda d: oe.cochran_q(d, ["g", "b"]),
    "ksmirnov": lambda d: oe.ksmirnov(d, "y"),
    "ksmirnov_by": lambda d: oe.ksmirnov(d, "y", by="g"),
    "swilk": lambda d: oe.swilk(d, "y"),
    "sfrancia": lambda d: oe.sfrancia(d, "y"),
    "sktest": lambda d: oe.sktest(d, "y"),
    "runtest": lambda d: oe.runtest(d, "y"),
    "bitest": lambda d: oe.bitest(d, "g"),
    "prtest": lambda d: oe.prtest(d, "g", by="b"),
    "chi2gof": lambda d: oe.chi2gof(d, "g"),
    "crosstab": lambda d: oe.crosstab(d, "g", "b", exact=True),
    "tabulate": lambda d: oe.tabulate(d, "g"),
    "roc": lambda d: oe.roc(d, "g", "y"),
    "roccomp": lambda d: oe.roccomp(d, "g", ["y", "x"]),
}


@pytest.mark.parametrize("name", sorted(PROCEDURES))
def test_empty_all_missing_and_absent_columns_raise_analysis_errors(name):
    call = PROCEDURES[name]
    columns = ["y", "x", "g", "b"]
    with pytest.raises(AnalysisError) as error:
        call({column: [] for column in columns})
    assert error.value.code == "empty_data"
    with pytest.raises(AnalysisError) as error:
        call({column: [None, None, None] for column in columns})
    assert error.value.code == "empty_sample"
    with pytest.raises(AnalysisError) as error:
        call({"other": [1.0, 2.0, 3.0]})
    assert error.value.code == "missing_columns"
    with pytest.raises(AnalysisError) as error:
        call("not a table")
    assert error.value.code == "invalid_data"
    one = {"y": [1.5], "x": [0.5], "g": [1], "b": [0]}
    try:
        result = call(one)                    # a single row either works or is refused cleanly
    except AnalysisError as error:
        assert error.code in {"invalid_groups", "too_few_observations", "sample_size",
                              "no_variation", "not_binary"}
    else:
        finite_attrs(result)


@pytest.mark.parametrize("name", sorted(PROCEDURES))
def test_constant_columns_work_or_raise_no_variation(name):
    constant = {"y": [2.0] * 12, "x": [2.0] * 12, "g": [1] * 12, "b": [1] * 12}
    try:
        result = PROCEDURES[name](constant)
    except AnalysisError as error:
        assert error.code in {"no_variation", "invalid_groups", "not_binary"}
        assert len(str(error)) > 20
    else:
        finite_attrs(result)


def test_exact_kolmogorov_beyond_scipy_accuracy():
    # 50-digit references from the Durbin matrix evaluated with mpmath. SciPy's kstwo
    # switches to the Pelz-Good approximation for n > 140 and is off by about 1e-6 there.
    references = [(200, 0.06422649843704882, 0.36584437619457568),
                  (500, 0.06718874808223785, 0.020888510963016694),
                  (1000, 0.03162277660168379, 0.2644092676966476),
                  (100, 0.1, 0.25269275700639007)]
    for n, d, value in references:
        assert exact_kernels.kolmogorov_sf_exact(n, d) == pytest.approx(value, rel=1e-10)
    assert stats.kstwo.sf(0.06422649843704882, 200) != pytest.approx(0.36584437619457568,
                                                                    rel=1e-7)


def test_options_at_their_bounds():
    rng = np.random.default_rng(65)
    frame = pd.DataFrame({"y": rng.normal(size=30), "g": rng.integers(0, 2, size=30),
                          "b": rng.integers(0, 2, size=30)})
    for alpha in (1e-12, 0.999999):
        attrs = finite_attrs(oe.ranksum(frame, "y", "g", alpha=alpha))
        assert attrs["hl_ci_low"] <= attrs["hl_estimate"] <= attrs["hl_ci_high"]
        finite_attrs(oe.roc(frame, "g", "y", alpha=alpha))
        finite_attrs(oe.bitest(frame, "g", alpha=alpha))
        finite_attrs(oe.crosstab(frame, "g", "b", alpha=alpha))
    wide = oe.ranksum(frame, "y", "g", alpha=1e-12).attrs
    narrow = oe.ranksum(frame, "y", "g", alpha=0.999999).attrs
    assert wide["hl_ci_low"] <= narrow["hl_ci_low"] <= narrow["hl_ci_high"] <= wide["hl_ci_high"]
    for value in (0, 1, -0.1, "0.05", None, True, float("nan")):
        with pytest.raises(AnalysisError) as error:
            oe.roc(frame, "g", "y", alpha=value)
        assert error.value.code == "invalid_option"
    assert finite_attrs(oe.bitest(frame, "g", p=1e-300))["p_value"] == 0.0
    assert finite_attrs(oe.prtest(frame, "g", p=1 - 1e-12))["p_value"] == 0.0
    for p in (0, 1, 1.5, "half"):
        with pytest.raises(AnalysisError):
            oe.bitest(frame, "g", p=p)


def test_bad_weights_are_refused():
    base = {"r": [0, 0, 1, 1], "c": [0, 1, 0, 1]}
    cases = [([1.0, -1.0, 2.0, 1.0], "invalid_weights"), ([0.0, 0.0, 0.0, 0.0], "invalid_weights"),
             (["a", "b", "c", "d"], "non_numeric_column"),
             ([1.0, float("inf"), 1.0, 1.0], "non_finite_values")]
    for weights, code in cases:
        for call in (lambda d: oe.crosstab(d, "r", "c", weights="w"),
                     lambda d: oe.tabulate(d, "r", weights="w"),
                     lambda d: oe.chi2gof(d, "r", weights="w")):
            with pytest.raises(AnalysisError) as error:
                call({**base, "w": weights})
            assert error.value.code == code
    # A missing weight drops the row (and is counted).
    result = oe.crosstab({**base, "w": [2.0, None, 1.0, 3.0]}, "r", "c", weights="w")
    assert result.attrs["n"] == 6 and result.attrs["n_dropped"] == 1


def test_jonckheere_pair_counts_do_not_scale_with_the_number_of_groups():
    """Regression: 100,000 groups used to take minutes (O(groups x distinct values))."""
    import time

    from openecon.econometrics.nonparametric import independent

    rng = np.random.default_rng(66)
    for _ in range(150):                                   # the kernel against brute force
        n = int(rng.integers(0, 40))
        keys = rng.integers(0, int(rng.integers(1, 70)), size=n)
        reference = sum(1 for i in range(n) for j in range(i + 1, n) if keys[i] < keys[j])
        assert independent.ascending_pairs(torch.as_tensor(keys, dtype=torch.int64)) == reference
    for _ in range(100):                                   # both paths against brute force
        n, k = int(rng.integers(2, 60)), int(rng.integers(1, 12))
        g = rng.integers(0, k, size=n)
        v = np.round(rng.normal(size=n), int(rng.integers(0, 3)))
        lower = g[:, None] < g[None, :]
        expected = (float((lower & (v[:, None] < v[None, :])).sum()),
                    float((lower & (v[:, None] > v[None, :])).sum()))
        codes = torch.unique(torch.as_tensor(v), sorted=True, return_inverse=True)[1]
        m = int(codes.max()) + 1
        assert independent._pair_counts_by_table(torch.as_tensor(g), codes, k, m) == expected
        assert independent._pair_counts_by_sorting(torch.as_tensor(g), codes, k, m) == expected
    n = 120_000
    data = {"y": np.round(rng.normal(size=n), 3), "g": rng.integers(0, 40_000, size=n)}
    started = time.perf_counter()
    result = oe.jonckheere(data, "y", "g")
    assert time.perf_counter() - started < 5.0
    kendall = stats.kendalltau(data["g"], data["y"], method="asymptotic")
    assert result.attrs["p_value"] == pytest.approx(kendall.pvalue, rel=1e-6)
    assert result.attrs["statistic"] - result.attrs["mean"] == pytest.approx(
        kendall.statistic * math.sqrt(
            (n * (n - 1) / 2 - _tied_pairs(data["g"])) * (n * (n - 1) / 2 - _tied_pairs(data["y"])))
        / 2, rel=1e-9)


def _tied_pairs(values) -> float:
    counts = np.unique(values, return_counts=True)[1].astype(float)
    return float((counts * (counts - 1) / 2).sum())


def test_wide_paired_structures_are_bounded():
    rng = np.random.default_rng(67)
    continuous = {"a": rng.normal(size=3000), "b": rng.normal(size=3000)}
    for call in (oe.symmetry, oe.mcnemar):                 # 6000 "categories": 3.6e7 cells
        with pytest.raises(AnalysisError) as error:
            call(continuous, "a", "b")
        assert error.value.code == "too_many_categories"
    wide = {f"c{j}": rng.normal(size=6) for j in range(201)}
    names = list(wide)
    assert oe.friedman(wide, names).attrs["k"] == 201
    with pytest.raises(AnalysisError) as error:
        oe.friedman(wide, names, pairwise=True)
    assert error.value.code == "too_many_groups"
    assert len(oe.friedman(wide, names[:200], pairwise=True)["pairwise"]) == 200 * 199 // 2
