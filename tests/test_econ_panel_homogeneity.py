"""Independent full-projection/least-squares oracles; no estimator dependency."""
from io import StringIO
import json

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.stats import beta as beta_distribution, norm
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.panel.homogeneity import xthst
from openecon.frame import DataFrame


def panel_data(seed=522, groups=11, periods=41, *, unbalanced=False, heterogeneous=True, k=2):
    rng = np.random.default_rng(seed)
    frames = []
    for unit in range(groups):
        # Deliberately many different T_i, including disjoint calendar ranges.
        count = periods - (unit * 3 % 17) if unbalanced else periods
        x = rng.normal(size=(count, k))
        slopes = np.linspace(.4, 1.1, k)
        if heterogeneous:
            slopes = slopes + unit * np.linspace(.13, -.07, k)
        sigma = .4 + unit * .12
        y = 3 + unit * 1.3 + x @ slopes + rng.normal(size=count) * sigma
        date = np.arange(count) * 2 + (unit * 100 if unbalanced else 0)
        frames.append(pd.DataFrame({"unit": unit, "date": date, "y": y,
                                    **{f"x{j}": x[:, j] + unit * (j + 1) for j in range(k)}}))
    return pd.concat(frames, ignore_index=True)


def run(frame, k=2, **options):
    return xthst(data=frame, y="y", x=[f"x{j}" for j in range(k)], panel="unit", time="date", **options)


def oracle(frame, k=2, *, unrestricted_variance=False):
    """Full M_i projection and stacked WLS; separate from tensor TSQR code."""
    xs, ys, unit_beta, sizes = [], [], [], []
    for _, group in frame.dropna(subset=["y", *[f"x{j}" for j in range(k)]]).groupby("unit", sort=True):
        raw_x = group[[f"x{j}" for j in range(k)]].to_numpy()
        raw_y = group.y.to_numpy()
        size = len(group)
        projection = np.eye(size) - np.ones((size, size)) / size
        x, y = projection @ raw_x, projection @ raw_y
        xs.append(x)
        ys.append(y)
        sizes.append(size)
        unit_beta.append(np.linalg.lstsq(x, y, rcond=None)[0])
    beta_fe = np.linalg.lstsq(np.vstack(xs), np.concatenate(ys), rcond=None)[0]
    if unrestricted_variance:
        variance = np.array([(y - x @ b) @ (y - x @ b) / (size - k - 1)
                             for x, y, b, size in zip(xs, ys, unit_beta, sizes, strict=True)])
    else:
        variance = np.array([(y - x @ beta_fe) @ (y - x @ beta_fe) / (size - 1)
                             for x, y, size in zip(xs, ys, sizes, strict=True)])
    beta_wfe = np.linalg.lstsq(np.vstack([x / np.sqrt(s) for x, s in zip(xs, variance, strict=True)]),
                              np.concatenate([y / np.sqrt(s) for y, s in zip(ys, variance, strict=True)]), rcond=None)[0]
    # Use independently solved unit coefficients and explicit cross products.
    d = np.array([(b - beta_wfe) @ (x.T @ x) @ (b - beta_wfe) / s
                  for b, x, s in zip(unit_beta, xs, variance, strict=True)])
    groups = len(xs)
    v = 2 * k * (np.array(sizes) - k - 1) / (np.array(sizes) + 1)
    delta = (d.sum() - groups * k) / np.sqrt(2 * groups * k)
    adjusted = ((d - k) / np.sqrt(v)).sum() / np.sqrt(groups)
    return {"delta": (delta, norm.sf(delta)), "adjusted": (adjusted, norm.sf(adjusted)),
            "dispersion": d.sum(), "unit_dispersion": d, "unit_variance": variance,
            "sizes": np.array(sizes), "beta_fe": beta_fe, "beta_wfe": beta_wfe}


@pytest.mark.parametrize("unbalanced", [False, True])
@pytest.mark.parametrize("method", ["delta", "adjusted", "all"])
@pytest.mark.parametrize("block", [1, 4, 128])
def test_full_projection_and_stacked_wls_oracle(unbalanced, method, block):
    frame = panel_data(unbalanced=unbalanced)
    before = frame.copy(deep=True)
    actual = run(frame, method=method, block_size=block)
    expected = oracle(frame)
    assert isinstance(actual, DataFrame)
    for name in actual.index:
        assert_allclose(actual.loc[name, ["statistic", "p_value"]].astype(float), expected[name], rtol=2e-11, atol=3e-13)
    assert_allclose(actual.attrs["dispersion"], expected["dispersion"], rtol=2e-12, atol=1e-12)
    assert actual.attrs["balanced"] is not unbalanced
    assert actual.attrs["provenance"]["dtype"] == "float64"
    assert actual.attrs["provenance"]["device"] == "cpu"
    assert actual.attrs["provenance"]["stata_parity_validated"] is False
    assert actual.attrs["provenance"]["finite_sample_p_values"] is False
    assert actual.attrs["p_value_tail"] == "upper"
    pd.testing.assert_frame_equal(before, frame)


def test_variance_is_from_null_pooled_fe_not_unrestricted_unit_ols():
    frame = panel_data(heterogeneous=True)
    expected = oracle(frame)
    wrong = oracle(frame, unrestricted_variance=True)
    actual = run(frame)
    assert_allclose(actual.attrs["dispersion"], expected["dispersion"], atol=1e-12)
    assert abs(wrong["dispersion"] - expected["dispersion"]) > 20
    assert "pooled-FE" in actual.attrs["variance_estimator"]


def test_unbalanced_adjustment_standardizes_each_unit_not_average_t():
    frame = panel_data(groups=18, periods=29, unbalanced=True)
    reference = oracle(frame)
    actual = run(frame)
    mean_t = reference["sizes"].mean()
    incorrectly_pooled = (reference["dispersion"] - 18 * 2) / np.sqrt(18 * 4 * (mean_t - 3) / (mean_t + 1))
    assert_allclose(actual.loc["adjusted", "statistic"], reference["adjusted"][0], atol=1e-12)
    assert abs(actual.loc["adjusted", "statistic"] - incorrectly_pooled) > .02


@pytest.mark.parametrize("unbalanced", [False, True])
def test_permutation_unit_labels_unit_offsets_common_scale_and_basis_invariance(unbalanced):
    frame = panel_data(unbalanced=unbalanced)
    expected = run(frame)
    changed = frame.sample(frac=1, random_state=823).copy()
    columns = changed[["x0", "x1"]].to_numpy() @ np.array([[1.3, -.4], [.2, .9]])
    columns += changed.unit.to_numpy()[:, None] * [170, -381]
    changed[["x0", "x1"]] = columns
    changed.y = changed.y * -2.7 + changed.unit * 700
    changed.unit = changed.unit.map({j: f"unit-{19-j}" for j in range(11)})
    changed.date = changed.date.astype("int64") + 2**60
    actual = run(changed, memory_mb=1, block_size=3)
    assert_allclose(actual.statistic, expected.statistic, rtol=2e-11, atol=1e-11)


@pytest.mark.parametrize("outcome_scale,first_scale,second_scale", [(1e-180, 1e160, 1e-170),
                                                                  (1e170, 1e-160, 1e180)])
def test_large_and_tiny_common_units_of_measure(outcome_scale, first_scale, second_scale):
    frame = panel_data()
    expected = run(frame)
    frame.y *= outcome_scale
    frame.x0 *= first_scale
    frame.x1 *= second_scale
    actual = run(frame, block_size=2)
    assert_allclose(actual.statistic, expected.statistic, rtol=2e-11, atol=2e-12)


def test_large_unit_offsets_retain_within_variation_before_normalization():
    frame = panel_data()
    expected = oracle(frame)
    changed = frame.copy()
    changed.y += changed.unit * 1e10
    changed.x0 += changed.unit * 1e10
    changed.x1 -= changed.unit * 1e10
    actual = run(changed)
    # The supplied floating-point values have lost ~1e-5 precision after offsets.
    assert_allclose(actual.statistic, [expected[name][0] for name in ["delta", "adjusted"]], rtol=8e-6, atol=2e-5)


def test_datetime_unsigned_integer_and_irregular_date_identity():
    frame = panel_data(unbalanced=True)
    expected = run(frame)
    frame.date = frame.date.astype("uint64") + np.uint64(2**64 - 10000)
    assert_allclose(run(frame).statistic, expected.statistic, atol=2e-12)
    original_dates = panel_data(unbalanced=True).date
    frame.date = pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(original_dates, unit="D")
    assert_allclose(run(frame).statistic, expected.statistic, atol=2e-12)
    assert run(frame).attrs["balanced"] is False


def test_equal_length_units_on_different_dates_are_not_mislabelled_balanced():
    frame = panel_data()
    expected = run(frame)
    frame.date += frame.unit * 1000
    actual = run(frame)
    assert actual.attrs["balanced"] is False
    assert actual.attrs["equal_unit_lengths"] is True
    assert_allclose(actual.statistic, expected.statistic, atol=1e-12)


def test_missing_drop_uses_complete_y_x_rows_and_never_drops_keys_or_units():
    frame = panel_data()
    frame.loc[[4, 41, 122], "y"] = np.nan
    frame.loc[[45, 53, 167], "x1"] = np.nan
    actual = run(frame, missing="drop")
    reference = oracle(frame)
    assert_allclose(actual.statistic, [reference[name][0] for name in ["delta", "adjusted"]], atol=2e-12)
    assert actual.attrs["n_dropped"] == 6
    assert actual.attrs["nobs"] == len(frame) - 6
    assert actual.attrs["balanced"] is False
    with pytest.raises(AnalysisError) as exc:
        run(frame)
    assert exc.value.code == "missing_values"
    frame.loc[0, "unit"] = np.nan
    with pytest.raises(AnalysisError) as exc:
        run(frame, missing="drop")
    assert exc.value.code == "missing_panel_time"


def test_json_and_publication_latex_have_explicit_inference_scope():
    actual = run(panel_data(unbalanced=True))
    restored = pd.read_json(StringIO(actual.to_json(orient="split")), orient="split")
    assert_allclose(actual.statistic, restored.statistic, rtol=1e-9)
    assert json.loads(json.dumps(actual.attrs, allow_nan=False)) == actual.attrs
    latex = str(actual.to_latex(caption="Slope homogeneity", notes=actual.attrs["notes"]))
    assert "\\toprule" in latex and "\\bottomrule" in latex
    assert "adjusted" in latex and "asymptotic" in latex and "Gaussian" in latex
    assert "exact finite-sample" in latex


def test_gaussian_finite_t_moments_from_independent_beta_ratio_distribution():
    # Under the null and the TRUE common slope, z_i is (T_i-1) times a
    # Beta(k/2,(T_i-1-k)/2) variable. This does NOT say feasible d_i has an
    # exact finite-N distribution after estimating pooled slopes and variances.
    for size, k in [(6, 1), (8, 3), (31, 2), (63, 7)]:
        mean, variance = beta_distribution.stats(k / 2, (size - 1 - k) / 2, moments="mv")
        assert_allclose(mean * (size - 1), k, atol=1e-14)
        assert_allclose(variance * (size - 1)**2, 2 * k * (size - k - 1) / (size + 1), rtol=2e-15)


def test_seeded_null_size_and_slope_dispersion_power_are_not_finite_sample_guarantees():
    rng = np.random.default_rng(783)
    groups, periods, k, replications = 55, 85, 2, 90
    x = rng.normal(size=(groups, periods, k))
    alpha = rng.normal(size=groups)
    sigma = rng.uniform(.7, 1.8, size=groups)
    units, dates = np.repeat(np.arange(groups), periods), np.tile(np.arange(periods), groups)
    base = {"unit": units, "date": dates, "x0": x[..., 0].ravel(), "x1": x[..., 1].ravel()}
    stats = []
    for _ in range(replications):
        y = alpha[:, None] + x @ [.3, -.5] + rng.normal(size=(groups, periods)) * sigma[:, None]
        stats.append(run(pd.DataFrame({**base, "y": y.ravel()}), method="adjusted").statistic.iloc[0])
    assert -.5 < np.mean(stats) < .25
    assert .65 < np.std(stats, ddof=1) < 1.35
    assert .005 < np.mean(np.array(stats) > norm.isf(.05)) < .13
    slopes = rng.normal(scale=1.4, size=(groups, k))
    y = alpha[:, None] + np.einsum("gtk,gk->gt", x, slopes) + rng.normal(size=(groups, periods)) * sigma[:, None]
    power = run(pd.DataFrame({**base, "y": y.ravel()}), method="adjusted")
    assert power.p_value.iloc[0] < 1e-15


def test_long_units_use_bounded_row_qr_tiles_without_unit_factor_caches(monkeypatch):
    frame = panel_data(groups=23, periods=1500, k=4)
    seen = []
    original_qr = torch.linalg.qr

    def qr(values, *args, **kwargs):
        seen.append(tuple(values.shape))
        return original_qr(values, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "qr", qr)
    actual = run(frame, k=4, memory_mb=1, block_size=128)
    workspace = actual.attrs["workspace"]
    assert workspace["row_block_size"] < 1500
    assert workspace["estimated_tensor_workspace_bytes"] <= workspace["budget_bytes"]
    assert workspace["passes"] == 3
    unit_qrs = [shape for shape in seen if len(shape) == 3]
    assert unit_qrs and all(shape[0] <= workspace["panel_block_size"] for shape in unit_qrs)
    assert all(shape[1] <= workspace["row_block_size"] + 5 for shape in unit_qrs)
    assert all(shape[-1] == 5 for shape in seen)
    assert all(len(shape) != 2 or shape[0] <= workspace["panel_block_size"] * 5 + 5 for shape in seen)
    # Independent complete-row oracle (small blocks chosen in a different way).
    assert_allclose(actual.statistic, run(frame, k=4, memory_mb=64, block_size=1).statistic, atol=3e-12, rtol=2e-11)


def test_negative_delta_uses_upper_tail_and_does_not_reject_unusually_close_slopes():
    # Orthogonal errors give every unit exactly the SAME fitted slopes, while
    # residual variances remain positive. Lower-tail dispersion is not H_A.
    rng = np.random.default_rng(421)
    pieces = []
    for unit in range(20):
        x = rng.normal(size=(35, 2))
        design = np.column_stack([np.ones(35), x])
        noise = rng.normal(size=35)
        noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
        y = unit + x @ [.7, .2] + noise
        pieces.append(pd.DataFrame({"unit": unit, "date": np.arange(35), "y": y, "x0": x[:, 0], "x1": x[:, 1]}))
    actual = run(pd.concat(pieces, ignore_index=True))
    assert actual.statistic.max() < -4
    assert actual.p_value.min() > .9999


@pytest.mark.parametrize("options", [{"method": "bad"}, {"method": None}, {"missing": "bad"},
                                     {"block_size": 0}, {"block_size": 513}, {"block_size": True},
                                     {"memory_mb": 0}, {"memory_mb": 1.3}, {"memory_mb": True}])
def test_invalid_options(options):
    with pytest.raises(AnalysisError) as exc:
        run(panel_data(), **options)
    assert exc.value.code == "invalid_option"


@pytest.mark.parametrize("kind,code", [("empty", "empty_data"), ("one_unit", "insufficient_panels"),
                                       ("short_unit", "insufficient_panel_observations"),
                                       ("duplicate_key", "duplicate_panel_time"),
                                       ("missing_key", "missing_panel_time"),
                                       ("text_time", "invalid_time"), ("bool_time", "invalid_time"),
                                       ("fractional_time", "invalid_time"), ("huge_float_date", "invalid_time"),
                                       ("constant_slope", "rank_deficient_panel"), ("rank_one_unit", "rank_deficient_panel"),
                                       ("collinear", "rank_deficient_panel"), ("near_collinear", "rank_deficient_panel"),
                                       ("constant_y", "degenerate_panel_variance"), ("pooled_perfect_fit", "degenerate_panel_variance"),
                                       ("infinite_y", "non_finite_values"), ("complex_x", "complex_values"),
                                       ("text_x", "non_numeric_column"), ("all_missing", "empty_sample"),
                                       ("missing_unit", "insufficient_panel_observations")])
def test_failure_guards(kind, code):
    frame = panel_data()
    missing = "raise"
    if kind == "empty":
        frame = frame.iloc[:0]
    elif kind == "one_unit":
        frame = frame.loc[frame.unit == 0]
    elif kind == "short_unit":
        frame = frame.loc[(frame.unit != 0) | (frame.date < 10)]
    elif kind == "duplicate_key":
        extra = frame.iloc[:1].copy()
        extra.y = np.nan
        frame = pd.concat([frame, extra], ignore_index=True)
        missing = "drop"
    elif kind == "missing_key":
        frame.loc[0, "date"] = np.nan
        missing = "drop"
    elif kind == "text_time":
        frame.date = frame.date.astype(str)
    elif kind == "bool_time":
        frame.date = False
    elif kind == "fractional_time":
        frame.date = frame.date + .2
    elif kind == "huge_float_date":
        frame.date = frame.date.astype(float) + 2**60
    elif kind == "constant_slope":
        frame.x0 = frame.unit * .5
    elif kind == "rank_one_unit":
        frame.loc[frame.unit == 0, "x1"] = frame.loc[frame.unit == 0, "x0"] * 2
    elif kind == "collinear":
        frame.x1 = 2 * frame.x0 + frame.unit
    elif kind == "near_collinear":
        frame.x1 = frame.x0 + np.random.default_rng(28).normal(scale=1e-9, size=len(frame))
    elif kind == "constant_y":
        frame.y = frame.unit * .3
    elif kind == "pooled_perfect_fit":
        frame.y = frame.unit + frame.x0 * .3 + frame.x1 * -.2
    elif kind == "infinite_y":
        frame.loc[0, "y"] = np.inf
        missing = "drop"
    elif kind == "complex_x":
        frame.x0 = frame.x0.astype(complex)
    elif kind == "text_x":
        frame.x0 = frame.x0.astype(str)
    elif kind == "all_missing":
        frame.y = np.nan
        missing = "drop"
    elif kind == "missing_unit":
        frame.loc[frame.unit == 3, "x0"] = np.nan
        missing = "drop"
    with pytest.raises(AnalysisError) as exc:
        run(frame, missing=missing)
    assert exc.value.code == code


@pytest.mark.parametrize("slopes", [[], "x0", ["x0", "x0"], ["y"], ["unit"], [None]])
def test_invalid_or_duplicate_model_columns(slopes):
    with pytest.raises(AnalysisError) as exc:
        xthst(data=panel_data(), y="y", x=slopes, panel="unit", time="date")
    assert exc.value.code == "invalid_spec"


def test_absent_and_duplicate_input_columns():
    frame = panel_data()
    with pytest.raises(AnalysisError) as exc:
        run(frame.drop(columns=["x1"]))
    assert exc.value.code == "missing_columns"
    frame.columns = ["unit", "date", "y", "x0", "x0"]
    with pytest.raises(AnalysisError) as exc:
        run(frame)
    assert exc.value.code == "duplicate_columns"


def test_slope_count_controls_minimum_degrees_and_workspace_guard():
    frame = panel_data(groups=3, periods=9, k=8)
    with pytest.raises(AnalysisError) as exc:
        run(frame, k=8)
    assert exc.value.code == "insufficient_panel_observations"
    frame = panel_data(groups=2, periods=50, k=45)
    with pytest.raises(AnalysisError) as exc:
        run(frame, k=45, memory_mb=1, block_size=1)
    assert exc.value.code == "workspace_too_small"


def test_mapping_input_and_nondefault_torch_precision():
    frame = panel_data()
    expected = run(frame)
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        actual = run(frame.to_dict(orient="list"))
    finally:
        torch.set_default_dtype(dtype)
    assert_allclose(actual.statistic, expected.statistic, atol=2e-12)


def test_cpu_context_overrides_and_restores_callers_default_device():
    frame = panel_data()
    expected = run(frame)
    with torch.device("meta"):
        assert torch.empty(0).device.type == "meta"
        actual = run(frame)
        assert torch.empty(0).device.type == "meta"
    assert_allclose(actual.statistic, expected.statistic, atol=2e-12)
    assert actual.attrs["provenance"]["device"] == "cpu"


def test_opposite_sign_origin_overflow_falls_back_to_common_column_scaling():
    frame = panel_data()
    rng = np.random.default_rng(418)
    frame.y = rng.uniform(-1, 1, size=len(frame))
    frame.loc[frame.groupby("unit").head(1).index, "y"] = -1.0
    frame.loc[frame.groupby("unit").head(2).groupby("unit").tail(1).index, "y"] = 1.0
    expected = run(frame)
    # Every input is finite, but +1e308 - (-1e308) cannot be represented.
    frame.y *= 1e308
    actual = run(frame, block_size=3, memory_mb=1)
    assert_allclose(actual.statistic, expected.statistic, rtol=2e-12, atol=1e-12)
