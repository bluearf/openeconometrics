"""Independent NumPy per-pair oracles for native residual dependence."""
from itertools import combinations
from io import StringIO
import json

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.stats import chi2, norm
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.panel.dependence import xtcd


def panel_data(seed=102, groups=9, periods=37, *, unbalanced=False):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(groups, periods))
    noise += .6 * rng.normal(size=periods)[None, :]
    noise *= np.linspace(.4, 2.5, groups)[:, None]
    noise += np.linspace(-10, 20, groups)[:, None]
    frame = pd.DataFrame({"unit": np.repeat(np.arange(groups), periods),
                          "date": np.tile(np.arange(periods), groups), "e": noise.ravel()})
    if unbalanced:
        frame = frame.loc[(frame.date + frame.unit) % 5 != 0]
    return frame


def oracle(frame, missing=False):
    if missing:
        frame = frame.dropna(subset=["e"])
    series = {unit: group.set_index("date").e for unit, group in frame.groupby("unit")}
    correlations, weighted, squared, overlaps = [], [], [], []
    for left, right in combinations(series, 2):
        pair = pd.concat([series[left], series[right]], axis=1, join="inner")
        a, b = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
        a, b = a - a.mean(), b - b.mean()
        rho = (a @ b) / np.sqrt((a @ a) * (b @ b))
        count = len(pair)
        correlations.append(rho)
        weighted.append(np.sqrt(count) * rho)
        squared.append(count * rho**2)
        overlaps.append(count)
    pairs = len(overlaps)
    cd = sum(weighted) / np.sqrt(pairs)
    lm = sum(squared)
    scaled = (lm - pairs) / np.sqrt(2 * pairs)
    return {"cd": (cd, norm.sf(abs(cd)) * 2), "lm": (lm, chi2.sf(lm, pairs)),
            "scaled_lm": (scaled, norm.sf(abs(scaled)) * 2),
            "mean_correlation": np.mean(correlations), "overlap_min": min(overlaps), "overlap_max": max(overlaps)}


def run(frame, **options):
    return xtcd(data=frame, residual="e", panel="unit", time="date", **options)


@pytest.mark.parametrize("method", ["cd", "lm", "scaled_lm", "all"])
@pytest.mark.parametrize("block", [1, 4, 128])
def test_balanced_matches_independent_numpy_pair_oracle(method, block):
    frame = panel_data()
    saved = frame.copy(deep=True)
    actual = run(frame, method=method, block_size=block)
    expected = oracle(frame)
    for name in actual.index:
        assert_allclose(actual.loc[name, ["statistic", "p_value"]].astype(float), expected[name], rtol=2e-12, atol=1e-13)
    assert_allclose(actual.attrs["mean_correlation"], expected["mean_correlation"], atol=1e-14)
    assert actual.attrs["n_pairs"] == 36
    assert actual.attrs["overlap_min"] == actual.attrs["overlap_max"] == 37
    assert actual.attrs["provenance"]["stata_parity_validated"] is False
    assert actual.attrs["workspace"]["estimated_tensor_workspace_bytes"] <= actual.attrs["workspace"]["budget_bytes"]
    pd.testing.assert_frame_equal(saved, frame)


@pytest.mark.parametrize("block,memory", [(1, 1), (3, 1), (128, 64), (512, 1)])
def test_unbalanced_overlap_centering_matches_oracle(block, memory):
    frame = panel_data(unbalanced=True)
    actual = run(frame, block_size=block, memory_mb=memory)
    expected = oracle(frame)
    assert_allclose(actual.loc["cd", ["statistic", "p_value"]].astype(float), expected["cd"], rtol=3e-12, atol=1e-13)
    assert actual.attrs["overlap_min"] == expected["overlap_min"]
    assert actual.attrs["overlap_max"] == expected["overlap_max"]
    assert actual.attrs["balanced"] is False


@pytest.mark.parametrize("unbalanced", [False, True])
def test_row_unit_label_date_and_positive_unit_scale_invariance(unbalanced):
    frame = panel_data(unbalanced=unbalanced)
    original = run(frame)
    altered = frame.sample(frac=1, random_state=18).copy()
    altered.e *= altered.unit.map(dict(enumerate(np.geomspace(1e-180, 1e180, 9))))
    altered.unit = altered.unit.map({index: f"label-{8-index}" for index in range(9)})
    # Exact int64 dates above the binary64 consecutive-integer region remain distinct.
    altered.date = altered.date.astype("int64") + 2**60
    actual = run(altered, block_size=2, memory_mb=1)
    assert_allclose(actual.statistic, original.statistic, rtol=2e-12, atol=2e-13)
    assert actual.attrs["n_periods"] == original.attrs["n_periods"]


def test_datetime_dates_with_gaps_are_matched_by_identity_not_unit_row_ranks():
    frame = panel_data(unbalanced=True)
    expected = run(frame)
    frame.date = pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(frame.date * 3, unit="D")
    actual = run(frame)
    assert_allclose(actual.statistic, expected.statistic, atol=1e-13)


def test_unsigned_integer_dates_preserve_full_integer_identity():
    frame = panel_data(unbalanced=True)
    expected = run(frame)
    frame.date = frame.date.astype("uint64") + np.uint64(2**64 - 100)
    assert_allclose(run(frame).statistic, expected.statistic, atol=1e-13)


def test_missing_drop_only_drops_residual_rows_and_preserves_overlap_definition():
    frame = panel_data()
    frame.loc[[4, 40, 78, 141], "e"] = np.nan
    actual = run(frame, missing="drop")
    assert_allclose(actual.statistic.iloc[0], oracle(frame, missing=True)["cd"][0], atol=1e-12)
    assert actual.attrs["n_dropped"] == 4
    assert actual.attrs["nobs"] == len(frame) - 4
    with pytest.raises(AnalysisError, match="residual.*missing") as exc:
        run(frame)
    assert exc.value.code == "missing_values"


def test_result_exports_latex_and_json_safe_diagnostics():
    actual = run(panel_data(), method="all")
    restored = pd.read_json(StringIO(actual.to_json(orient="split")), orient="split")
    assert_allclose(restored.statistic, actual.statistic, rtol=1e-9)
    assert json.loads(json.dumps(actual.attrs, allow_nan=False)) == actual.attrs
    latex = str(actual.to_latex())
    assert "\\begin{tabular}" in latex and "\\toprule" in latex
    assert "scaled" in latex


def test_independently_fitted_unit_ols_residuals_follow_published_statistic():
    rng = np.random.default_rng(93)
    frames = []
    for unit in range(14):
        x = rng.normal(size=(80, 2))
        y = 4 + x @ [.5, -.2] + rng.normal(size=80)
        design = np.column_stack([np.ones(80), x])
        e = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        frames.append(pd.DataFrame({"unit": unit, "date": np.arange(80), "e": e}))
    frame = pd.concat(frames, ignore_index=True)
    actual = run(frame, method="all")
    reference = oracle(frame)
    for name in actual.index:
        assert_allclose(actual.loc[name, ["statistic", "p_value"]].astype(float), reference[name], atol=2e-13)


def test_seeded_independent_innovation_size_and_positive_common_factor_power():
    rng = np.random.default_rng(729)
    units, dates = np.repeat(np.arange(20), 160), np.tile(np.arange(160), 20)
    statistics, rejection = [], []
    for _ in range(180):
        errors = rng.normal(size=(20, 160))
        frame = pd.DataFrame({"unit": units, "date": dates, "e": errors.ravel()})
        result = run(frame)
        statistics.append(result.statistic.iloc[0])
        rejection.append(result.p_value.iloc[0] < .05)
    # Fixed-seed broad Monte Carlo checks, not a finite-sample inferential guarantee.
    assert abs(np.mean(statistics)) < .25
    assert .75 < np.std(statistics, ddof=1) < 1.2
    assert .015 < np.mean(rejection) < .10
    errors += rng.normal(size=160)[None, :]
    correlated = run(pd.DataFrame({"unit": units, "date": dates, "e": errors.ravel()}))
    assert correlated.p_value.iloc[0] < 1e-12


def test_small_memory_uses_bounded_pair_and_time_tiles(monkeypatch):
    frame = panel_data(groups=45, periods=550, unbalanced=True)
    allocations = []
    original = torch.zeros

    def record(shape, *args, **kwargs):
        if isinstance(shape, tuple):
            allocations.append(shape)
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(torch, "zeros", record)
    actual = run(frame, block_size=512, memory_mb=1)
    metadata = actual.attrs["workspace"]
    assert metadata["panel_block_size"] < 45
    assert metadata["time_block_size"] < 550
    assert all(len(shape) < 2 or (shape[0] <= metadata["panel_block_size"]
                                and shape[1] <= max(metadata["panel_block_size"], metadata["time_block_size"]))
               for shape in allocations)
    assert (len(frame), len(frame)) not in allocations
    assert (45, 45) not in allocations
    assert (45, 550) not in allocations
    assert_allclose(actual.statistic.iloc[0], oracle(frame)["cd"][0], rtol=1e-11)


def test_balanced_cd_uses_linear_identity_not_pair_kernel(monkeypatch):
    import openecon.econometrics.panel.dependence as dependence

    def forbidden(*args, **kwargs):
        raise AssertionError("balanced CD should not examine every unit pair")

    monkeypatch.setattr(dependence, "_pair_statistics", forbidden)
    frame = panel_data(groups=70, periods=500)
    result = run(frame, memory_mb=1)
    assert result.attrs["workspace"]["algorithm"] == "balanced_linear_cd"
    assert_allclose(result.statistic.iloc[0], oracle(frame)["cd"][0], rtol=1e-12)


@pytest.mark.parametrize("options", [{"method": "bad"}, {"missing": "bad"}, {"min_overlap": 3},
                                     {"min_overlap": True}, {"block_size": 0}, {"block_size": 513},
                                     {"block_size": 2.5}, {"memory_mb": 0}, {"memory_mb": 1025}])
def test_invalid_options(options):
    with pytest.raises(AnalysisError) as exc:
        run(panel_data(), **options)
    assert exc.value.code == "invalid_option"


@pytest.mark.parametrize("method", ["lm", "scaled_lm", "all"])
def test_unbalanced_lm_methods_fail_explicitly(method):
    with pytest.raises(AnalysisError) as exc:
        run(panel_data(unbalanced=True), method=method)
    assert exc.value.code == "unbalanced_lm_unsupported"


@pytest.mark.parametrize("case,code", [("empty", "empty_data"), ("one", "insufficient_panels"),
                                      ("few", "insufficient_panel_observations"),
                                      ("duplicate", "duplicate_panel_time"),
                                      ("missing_key", "missing_panel_time"),
                                      ("infinite", "non_finite_values"),
                                      ("text", "non_numeric_column"),
                                      ("floatdate", "invalid_time"),
                                      ("largedatefloat", "invalid_time"),
                                      ("booldate", "invalid_time"),
                                      ("constant", "degenerate_residuals"),
                                      ("allmissingunit", "insufficient_panel_observations")])
def test_input_failures(case, code):
    frame = panel_data()
    if case == "empty":
        frame = frame.iloc[:0]
    elif case == "one":
        frame = frame[frame.unit == 0]
    elif case == "few":
        frame = frame[frame.date < 3]
    elif case == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
        frame.iloc[-1, frame.columns.get_loc("e")] = np.nan
    elif case == "missing_key":
        frame.loc[0, "unit"] = np.nan
    elif case == "infinite":
        frame.loc[0, "e"] = np.inf
    elif case == "text":
        frame.e = frame.e.astype(str)
    elif case == "floatdate":
        frame.date = frame.date.astype(float) + .2
    elif case == "largedatefloat":
        frame.date = frame.date.astype(float) + 2**60
    elif case == "booldate":
        frame.date = frame.date.astype(bool)
    elif case == "constant":
        frame.loc[frame.unit == 0, "e"] = 7
    else:
        frame.loc[frame.unit == 0, "e"] = np.nan
    with pytest.raises(AnalysisError) as exc:
        run(frame, missing="drop")
    assert exc.value.code == code


def test_nonoverlapping_unit_pairs_are_never_silently_omitted():
    frame = panel_data()
    frame.loc[frame.unit == 0, "date"] += 1000
    with pytest.raises(AnalysisError) as exc:
        run(frame)
    assert exc.value.code == "insufficient_pair_overlap"


def test_overlap_constant_variance_is_not_masked_by_variation_outside_overlap():
    frame = pd.DataFrame({"unit": [0] * 6 + [1] * 6, "date": list(range(6)) + list(range(4, 10)),
                          "e": [1., 3., 2., 4., 7., 7., 5., 6., 2., 3., 1., 4.]})
    # Expand the overlap to four dates while keeping unit 0 constant there.
    frame.loc[frame.unit == 1, "date"] -= 2
    frame.loc[(frame.unit == 0) & (frame.date >= 2), "e"] = 7
    with pytest.raises(AnalysisError) as exc:
        run(frame)
    assert exc.value.code == "degenerate_pair_variance"


def test_duplicate_columns_and_colliding_roles():
    frame = panel_data()
    with pytest.raises(AnalysisError) as exc:
        xtcd(data=frame, residual="e", panel="unit", time="unit")
    assert exc.value.code == "invalid_spec"
    duplicate = pd.concat([frame, frame[["e"]]], axis=1)
    with pytest.raises(AnalysisError) as exc:
        run(duplicate)
    assert exc.value.code == "duplicate_columns"
