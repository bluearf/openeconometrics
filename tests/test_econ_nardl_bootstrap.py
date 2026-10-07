"""Recursive NARDL bootstrap checked against an independent NumPy/SVD oracle.

The same integer innovations are used for numerical comparisons; outcome
generation, design rebuilding, constrained refits, step responses and quantiles
are implemented independently, without OpenEcon design/constraint kernels.
"""

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from test_econ_tsmodels_nardl import data

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle


def numerical_oracle(result, frame, method, *, steps, replications, seed):
    frame = frame.sort_values(result.spec.time) if result.spec.time else frame
    required = result.provenance["input_columns"]
    frame = frame.dropna(subset=required)
    original_y = frame.y.to_numpy()
    nall = len(frame)
    hold = nall - result.nobs
    orders = result.extra["lags"]
    p = orders["y"]
    series = {}
    for name in result.spec.predictors:
        original = frame[name].to_numpy()
        if name in result.extra["asymmetric_predictors"]:
            delta = np.r_[0, np.diff(original)]
            series[f"{name}_positive"] = np.maximum(delta, 0).cumsum()
            series[f"{name}_negative"] = np.minimum(delta, 0).cumsum()
        else:
            series[name] = original
    terms = list(result.extra["levels_coefficients"])
    coefficients = np.array(list(result.extra["levels_coefficients"].values()))
    n = nall - hold
    columns = {"Intercept": np.ones(n)}
    if result.extra["trend"] == "trend":
        columns["trend"] = np.arange(1, nall + 1)[hold:]
    for i in range(1, p + 1):
        columns["L.y" if i == 1 else f"L{i}.y"] = original_y[hold - i:nall - i]
    for name, values in series.items():
        for i in range(orders[name] + 1):
            label = name if i == 0 else f"L.{name}" if i == 1 else f"L{i}.{name}"
            columns[label] = values[hold - i:nall - i]
    for name in result.spec.columns.get("exog", []):
        columns[name] = frame[name].to_numpy()[hold:]
    matrix = np.column_stack([columns[name] for name in terms])
    restrictions = []
    for name in result.extra["asymmetric_predictors"]:
        positive, negative = f"{name}_positive", f"{name}_negative"
        if name in result.spec.options.get("long_run_symmetric", []):
            row = np.zeros(len(terms))
            for side, sign in ((positive, 1), (negative, -1)):
                for i in range(orders[side] + 1):
                    label = side if i == 0 else f"L.{side}" if i == 1 else f"L{i}.{side}"
                    row[terms.index(label)] = sign
            restrictions.append(row)
        if name in result.spec.options.get("short_run_symmetric", []):
            for h in range(max(orders[positive], orders[negative])):
                row = np.zeros(len(terms))
                for side, sign in ((positive, 1), (negative, -1)):
                    if h == 0:
                        row[terms.index(side)] = sign
                    else:
                        for i in range(h + 1, orders[side] + 1):
                            label = f"L.{side}" if i == 1 else f"L{i}.{side}"
                            row[terms.index(label)] = -sign
                restrictions.append(row)
    if restrictions:
        R = np.vstack(restrictions)
        _, singular, vt = np.linalg.svd(R, full_matrices=True)
        rank = np.count_nonzero(singular > np.finfo(float).eps * max(R.shape) * singular[0])
        basis = vt[rank:].T
    else:
        basis = np.eye(len(terms))
    resid = original_y[hold:] - matrix @ coefficients
    residuals = (resid - resid.mean()) * math.sqrt(n / (n - basis.shape[1]))
    generator = torch.Generator().manual_seed(seed)
    integers = torch.randint(n if method == "residual" else 2,
                             (replications, n), generator=generator).numpy()
    innovations = residuals[integers] if method == "residual" else residuals * (integers * 2 - 1)
    responses = np.empty((replications, steps + 1, len(result.extra["asymmetric_predictors"]), 3))
    autoregressive = [terms.index("L.y" if i == 1 else f"L{i}.y") for i in range(1, p + 1)]
    for b in range(replications):
        outcome = original_y.copy()
        for t in range(hold, nall):
            row = matrix[t - hold].copy()
            row[autoregressive] = [outcome[t - i] for i in range(1, p + 1)]
            outcome[t] = row @ coefficients + innovations[b, t - hold]
        design = matrix.copy()
        for i, index in enumerate(autoregressive, start=1):
            design[:, index] = outcome[hold - i:nall - i]
        # SVD least squares in an independently formed SVD restriction null space.
        fitted = basis @ np.linalg.lstsq(design @ basis, outcome[hold:], rcond=None)[0]
        for j, name in enumerate(result.extra["asymmetric_predictors"]):
            for s, side in enumerate((f"{name}_positive", f"{name}_negative")):
                for h in range(steps + 1):
                    permanent_step = sum(fitted[terms.index(side if i == 0 else f"L.{side}"
                                         if i == 1 else f"L{i}.{side}")]
                                         for i in range(orders[side] + 1) if h >= i)
                    previous = sum(fitted[index] * responses[b, h - i, j, s]
                                   for i, index in enumerate(autoregressive, start=1) if h >= i)
                    responses[b, h, j, s] = permanent_step + previous
            responses[b, :, j, 2] = responses[b, :, j, 0] - responses[b, :, j, 1]
    return responses


def zero_gradient_data(seed=522, n=90):
    """A fitted stable AR2 with a genuinely stationary step response at h=1.

    The true cumulative multiplier is b0*(1+phi1). At b0=0, phi1=-1 both
    derivatives vanish, although free bootstrap b0/phi1 vary. An unrelated z
    symmetry gives a real constrained fit; exog w and an independently projected
    nonzero residual identify the desired coefficients without zero noise.
    """
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    x, z = rng.normal(size=n).cumsum(), rng.normal(size=n).cumsum()
    dx = np.r_[0, np.diff(x)]
    xp, xn = np.maximum(dx, 0).cumsum(), np.minimum(dx, 0).cumsum()
    base = np.column_stack([np.ones(n - 2), y[1:-1], y[:-2], xp[2:], xn[2:], (z - z[0])[2:]])
    q = np.linalg.qr(base, mode="reduced")[0]
    error = y[2:] + y[1:-1] + .2 * y[:-2]
    perpendicular = error - q @ (q.T @ error)
    other = rng.normal(size=n - 2)
    other -= q @ (q.T @ other)
    other -= perpendicular * (perpendicular @ other) / (perpendicular @ perpendicular)
    other *= np.linalg.norm(perpendicular) / np.linalg.norm(other)
    residual = .5 * (perpendicular + other)
    return pd.DataFrame(dict(y=y, x=x, z=z, w=np.r_[0., 0., error - residual]))


@pytest.mark.parametrize("method", ["residual", "wild"])
def test_stationary_nonlinear_step_response_keeps_empirical_uncertainty(method):
    frame = zero_gradient_data()
    fit = oe.nardl(data=frame, y="y", x=["x", "z"], lags=[2, 0, 0], exog=["w"],
                   long_run_symmetric=["z"], ec=False)
    assert_allclose([fit.extra["levels_coefficients"][name] for name in ("L.y", "L2.y")],
                    [-1., -.2], rtol=0, atol=2e-14)
    delta = oe.nardl_multipliers(fit, steps=2)
    stationary = delta[(delta.variable == "x") & (delta.horizon == 1)].iloc[0]
    assert max(stationary[f"{name}_std_error"] for name in ("positive", "negative", "difference")) < 1e-12
    assert delta.attrs["autoregressive_stable"]
    bands = oe.nardl_multipliers(fit, data=frame, steps=2, method=method,
                                replications=29, seed=981, batch_size=7)
    expected = numerical_oracle(fit, frame, method, steps=2, replications=29, seed=981)
    quantiles = np.quantile(expected, [.025, .975], axis=0)
    stationary = bands[(bands.variable == "x") & (bands.horizon == 1)].iloc[0]
    for i, name in enumerate(("positive", "negative", "difference")):
        assert stationary[f"{name}_std_error"] > 1e-4
        assert_allclose(stationary[f"{name}_std_error"], expected[:, 1, 0, i].std(ddof=1), rtol=1e-9)
        assert_allclose([stationary[f"{name}_ci_low"], stationary[f"{name}_ci_high"]],
                        quantiles[:, 1, 0, i], rtol=1e-9, atol=1e-11)
    # z's q0 LR identity proves only its positive-minus-negative contrast zero;
    # its positive/negative nonlinear step responses have nonzero uncertainty.
    z = bands[bands.variable == "z"]
    assert (z.difference == 0).all()
    assert (z.difference_std_error == 0).all()
    assert (z.difference_ci_low == 0).all() and (z.difference_ci_high == 0).all()
    assert z.loc[z.horizon == 1, "positive_std_error"].iloc[0] > 1e-4


def test_exact_zero_delta_variance_never_erases_recursive_percentiles(monkeypatch):
    from openecon.econometrics.tsmodels import nardl as module

    frame = zero_gradient_data()
    fit = oe.nardl(data=frame, y="y", x=["x", "z"], lags=[2, 0, 0], exog=["w"],
                   long_run_symmetric=["z"], ec=False)
    original = module.nardl_multipliers

    def exact_stationary_delta(*args, **kwargs):
        answer = original(*args, **kwargs)
        row = (answer.variable == "x") & (answer.horizon == 1)
        for side in ("positive", "negative", "difference"):
            answer.loc[row, f"{side}_std_error"] = 0.0
        return answer

    monkeypatch.setattr(module, "nardl_multipliers", exact_stationary_delta)
    bands = original(fit, data=frame, steps=2, method="wild", replications=29, seed=981, batch_size=7)
    stationary = bands[(bands.variable == "x") & (bands.horizon == 1)].iloc[0]
    assert stationary.positive_std_error > 1e-4
    assert stationary.negative_std_error > 1e-4
    assert stationary.difference_std_error > 1e-4
    assert stationary.positive_ci_high - stationary.positive_ci_low > .001


@pytest.mark.parametrize("method", ["residual", "wild"])
@pytest.mark.parametrize("trend,restricted", [("constant", False), ("constant", True),
                                             ("trend", False), ("trend", True)])
def test_actual_recursive_refits_against_independent_numpy(method, trend, restricted):
    frame, _, _ = data(n=100)
    fit = oe.nardl(data=frame, y="y", x=["x", "z"], asymmetric=["x"],
                   lags=[2, 2, 1, 0], exog=["w"], trend=trend,
                   restricted=restricted, covariance="HC3")
    bands = oe.nardl_multipliers(fit, data=frame, method=method, steps=8,
                                replications=29, seed=882, batch_size=7)
    expected = numerical_oracle(fit, frame, method, steps=8, replications=29, seed=882)
    quantiles = np.quantile(expected, [0.025, 0.975], axis=0)
    for i, side in enumerate(("positive", "negative", "difference")):
        assert_allclose(bands[f"{side}_ci_low"], quantiles[0, :, 0, i], rtol=1e-9, atol=1e-10)
        assert_allclose(bands[f"{side}_ci_high"], quantiles[1, :, 0, i], rtol=1e-9, atol=1e-10)
        assert_allclose(bands[f"{side}_std_error"], expected[:, :, 0, i].std(axis=0, ddof=1),
                        rtol=1e-9, atol=1e-10)
    assert bands.attrs["bootstrap"]["case"] == fit.extra["case"]
    assert bands.attrs["bootstrap"]["completed"] == 29


@pytest.mark.parametrize("method", ["residual", "wild"])
@pytest.mark.parametrize("restriction", ["long_run", "short_run", "both"])
def test_constrained_bootstrap_refits_against_independent_numpy(method, restriction):
    frame, _, _ = data(n=110)
    options = {}
    if restriction in {"long_run", "both"}:
        options["long_run_symmetric"] = ["x"]
    if restriction in {"short_run", "both"}:
        options["short_run_symmetric"] = ["x"]
    fit = oe.nardl(data=frame, y="y", x=["x"], lags=[2, 2, 1], **options)
    bands = oe.nardl_multipliers(fit, data=frame, method=method, steps=9,
                                replications=31, seed=771, batch_size=8)
    expected = numerical_oracle(fit, frame, method, steps=9, replications=31, seed=771)
    quantiles = np.quantile(expected, [0.025, 0.975], axis=0)
    for i, side in enumerate(("positive", "negative", "difference")):
        assert_allclose(bands[f"{side}_ci_low"], quantiles[0, :, 0, i], rtol=1e-8, atol=1e-9)
        assert_allclose(bands[f"{side}_ci_high"], quantiles[1, :, 0, i], rtol=1e-8, atol=1e-9)
    if restriction == "both":
        assert (bands.difference == 0).all()
        assert (bands.difference_ci_low == 0).all()
        assert (bands.difference_ci_high == 0).all()
        assert (bands.difference_std_error == 0).all()


@pytest.mark.parametrize("method", ["residual", "wild"])
def test_selected_common_holdback_sorted_missing_sample_json_and_rng(method):
    frame, _, _ = data(n=110)
    frame.loc[0, "x"] = np.nan
    shuffled = frame.sample(frac=1, random_state=441).reset_index(drop=True)
    fit = oe.nardl(data=shuffled, y="y", x=["x"], time="t", maxlags=[3, 1],
                   ic="bic", missing="drop", long_run_symmetric=["x"])
    saved = ResultBundle.model_validate_json(fit.model_dump_json())
    rng_before = torch.random.get_rng_state().clone()
    original_before = shuffled.copy(deep=True)
    a = oe.nardl_multipliers(saved, data=shuffled, method=method, steps=6,
                            replications=27, seed=67, batch_size=1)
    b = oe.nardl_multipliers(saved, data=shuffled, method=method, steps=6,
                            replications=27, seed=67, batch_size=9)
    assert_allclose(a.iloc[:, 2:], b.iloc[:, 2:], rtol=1e-12, atol=1e-12)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    pd.testing.assert_frame_equal(shuffled, original_before)
    assert a.attrs["bootstrap"]["hold_back"] == 3
    assert a.attrs["bootstrap"]["selection_repeated"] is False
    expected = numerical_oracle(fit, shuffled, method, steps=6, replications=27, seed=67)
    assert_allclose(a.positive_ci_low, np.quantile(expected[:, :, 0, 0], 0.025, axis=0),
                    rtol=1e-8, atol=1e-9)
    assert "\\end{tabular}" in a.to_latex(longtable=False)
    assert "bootstrap" in a.to_latex(longtable=False, notes=a.attrs["notes"])
    json.loads(a.to_json(orient="records"))
    json.dumps(a.attrs, allow_nan=False)


def test_multiple_asymmetric_predictors_keep_variable_horizon_order():
    frame, _, _ = data(n=140)
    frame["other"] = np.random.default_rng(892).normal(size=len(frame)).cumsum()
    frame.y += frame.other * .3
    fit = oe.nardl(data=frame, y="y", x=["x", "other"], lags=[2, 1, 0],
                   short_run_symmetric=["other"])
    bands = oe.nardl_multipliers(fit, data=frame, method="wild", steps=4,
                                replications=23, seed=12)
    assert bands.variable.tolist() == ["x"] * 5 + ["other"] * 5
    expected = numerical_oracle(fit, frame, "wild", steps=4, replications=23, seed=12)
    for j, variable in enumerate(("x", "other")):
        selected = bands[bands.variable == variable]
        assert_allclose(selected.negative_ci_high,
                        np.quantile(expected[:, :, j, 1], .975, axis=0), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("p,phi", [(1, [.999999]), (2, [.45, .1]), (3, [.4, .15, .05])])
@pytest.mark.parametrize("n", [1, 2, 37, 257])
def test_batched_recursion_matches_independent_sequential_equation(p, phi, n):
    from openecon.econometrics.tsmodels.nardl_bootstrap import _generate

    rng = np.random.default_rng(412)
    hold = p + 2
    y = rng.normal(size=hold + n)
    x = np.column_stack([np.ones(n), *[y[hold - i:hold + n - i]
                                     for i in range(1, p + 1)], rng.normal(size=n)])
    terms = ["Intercept", *["L.y" if i == 1 else f"L{i}.y" for i in range(1, p + 1)], "x"]
    coefficients = np.array([.3, *phi, .2])
    errors = rng.normal(size=(4, n))
    design, generated = _generate(torch.tensor(y), torch.tensor(x), torch.tensor(coefficients),
                                  terms, "y", p, hold, torch.tensor(errors))
    expected = np.broadcast_to(y, (4, len(y))).copy()
    for b in range(4):
        for t in range(hold, len(y)):
            expected[b, t] = .3 + .2 * x[t - hold, -1] + errors[b, t - hold]
            expected[b, t] += sum(phi[i - 1] * expected[b, t - i] for i in range(1, p + 1))
    assert_allclose(generated, expected[:, hold:], rtol=1e-12, atol=1e-12)
    for i in range(1, p + 1):
        assert_allclose(design[:, :, i], expected[:, hold - i:len(y) - i], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("p", [3, 4, 5, 6])
@pytest.mark.parametrize("radius", [.9, .99])
def test_native_recursion_repeated_stable_roots_matches_ordered_sequential_oracle(p, radius):
    # Repeated roots produce highly nonnormal companion matrices: squaring
    # those matrices gave errors many orders larger than direct AR filtering.
    # Keep the independent float64 oracle's addition order explicit; for p=6
    # and roots=.99, reassociating the sum itself amplifies ordinary roundoff.
    phi = -np.poly(np.repeat(radius, p))[1:].real
    _check_difficult_recursion(phi, 1025)


@pytest.mark.parametrize("roots,n", [
    ([-.999999], 8193),
    ([.98 * np.exp(.8j), .98 * np.exp(-.8j)], 1025),
    ([.999 * np.exp(.8j), .999 * np.exp(-.8j)], 8193),
    (.95 * np.exp(2j * np.pi * np.arange(5) / 5), 1025),
])
def test_oscillating_near_unit_and_high_order_recursion(roots, n):
    phi = -np.poly(roots)[1:].real
    _check_difficult_recursion(phi, n)


def _check_difficult_recursion(phi, n):
    from openecon.econometrics.tsmodels.nardl_bootstrap import _generate

    rng = np.random.default_rng(8153)
    p, hold = len(phi), len(phi) + 2
    y = rng.normal(size=hold + n)
    other = rng.normal(size=n)
    errors = rng.normal(size=(2, n))
    x = np.column_stack([np.ones(n), *[y[hold - i:hold + n - i]
                                     for i in range(1, p + 1)], other])
    beta = np.r_[.3, phi, .2]
    terms = ["Intercept", *["L.y" if i == 1 else f"L{i}.y" for i in range(1, p + 1)], "x"]
    design, actual = _generate(torch.tensor(y), torch.tensor(x), torch.tensor(beta),
                               terms, "y", p, hold, torch.tensor(errors))
    expected = np.broadcast_to(y, (2, len(y))).copy()
    forcing = .3 + .2 * other + errors
    for b in range(2):
        for t in range(hold, len(y)):
            expected[b, t] = forcing[b, t - hold]
            for i in range(1, p + 1):
                expected[b, t] += phi[i - 1] * expected[b, t - i]
    assert_allclose(actual, expected[:, hold:], rtol=1e-12, atol=1e-12)
    values = np.column_stack([np.broadcast_to(y[:hold], (2, hold)), actual])
    products = np.stack([phi[i - 1] * values[:, hold - i:len(y) - i]
                         for i in range(1, p + 1)])
    # Independently reassociated equation check measures local backward error,
    # rather than pretending stable roots guarantee a well-conditioned path.
    residual = actual.numpy() - forcing - products.sum(axis=0)
    scale = np.abs(actual.numpy()) + np.abs(forcing) + np.abs(products).sum(axis=0)
    assert np.all(np.abs(residual) <= 128 * np.finfo(float).eps * np.maximum(scale, np.finfo(float).tiny))
    for i in range(1, p + 1):
        assert_allclose(design[:, :, i], expected[:, hold - i:len(y) - i], rtol=1e-12, atol=1e-12)


def test_scalar_prefix_certification_uses_native_fallback_and_counts_batch(monkeypatch):
    from openecon.econometrics.tsmodels import nardl_bootstrap as module

    # A deterministic cancellation example triggers the residual certificate,
    # then exercises cold compilation from the fixed trusted source string.
    rng = np.random.default_rng(2)
    n, hold, phi = 1025, 3, -.999999
    y = rng.normal(size=n + hold)
    other = rng.normal(size=n)
    errors = rng.normal(size=(2, n))
    x = np.column_stack([np.ones(n), y[hold - 1:-1], other])
    diagnostics = {"scalar_prefix_fallback_batches": 0}
    monkeypatch.setattr(module, "_NATIVE_RECURSION", None)
    _, actual = module._generate(torch.tensor(y), torch.tensor(x),
                                 torch.tensor([.3, phi, .2], dtype=torch.float64),
                                 ["Intercept", "L.y", "x"], "y", 1, hold,
                                 torch.tensor(errors), diagnostics)
    assert diagnostics["scalar_prefix_fallback_batches"] == 1
    assert module._NATIVE_RECURSION is not None
    expected = np.broadcast_to(y, (2, len(y))).copy()
    for b in range(2):
        for t in range(hold, len(y)):
            expected[b, t] = .3 + .2 * other[t - hold] + errors[b, t - hold]
            expected[b, t] += phi * expected[b, t - 1]
    assert_allclose(actual, expected[:, hold:], rtol=0, atol=0)


@pytest.mark.parametrize("p", [1, 3])
def test_bootstrap_reports_actual_recursion_algorithm_and_cost(p):
    frame, _, _ = data(n=100)
    fit = oe.nardl(data=frame, y="y", x=["x"], lags=[p, 1])
    bands = oe.nardl_multipliers(fit, data=frame, method="wild", steps=4, replications=23)
    generation = bands.attrs["bootstrap"]["outcome_generation"]
    assert generation["python_time_row_loop"] is False
    assert "no external compiler" in generation["native_implementation"]
    if p > 1:
        assert generation["algorithm"] == "native_direct_ar"
        assert generation["time_complexity"] == "O(N p), batched replicates"
        assert generation["scalar_prefix_fallback_batches"] == 0
    else:
        assert generation["algorithm"] == "certified_scalar_affine_prefix_with_native_fallback"
        assert "native fallback" in generation["time_complexity"]


@pytest.mark.parametrize("ec", [True, False])
def test_zero_lag_vacuous_sr_and_omitted_static_controls(ec):
    frame, _, _ = data(n=100)
    frame["constant"] = 1.
    frame["duplicate"] = frame.w
    fit = oe.nardl(data=frame, y="y", x=["x"], lags=[3, 0], ec=ec,
                   short_run_symmetric=["x"], exog=["constant", "w", "duplicate"])
    bands = oe.nardl_multipliers(fit, data=frame, method="residual", steps=4,
                                replications=23, seed=899)
    assert fit.extra["imposed_symmetry"]["vacuous_short_run"] == ["x"]
    assert fit.provenance["omitted_terms"] == ["constant", "duplicate"]
    expected = numerical_oracle(fit, frame, "residual", steps=4, replications=23, seed=899)
    assert_allclose(bands.positive_ci_low, np.quantile(expected[:, :, 0, 0], .025, axis=0),
                    rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("mutation", ["coefficient", "public_covariance", "levels_optimum"])
def test_disjoint_public_and_levels_state_fails_saved_result_validation(mutation):
    frame, _, _ = data(n=100)
    fit = oe.nardl(data=frame, y="y", x=["x"], lags=[2, 1])
    if mutation == "coefficient":
        fit.coefficients[0].estimate += .1
    elif mutation == "public_covariance":
        fit.covariance_matrix[0][0] *= 2
    else:
        fit.extra["levels_coefficients"]["Intercept"] += .1
    with pytest.raises(AnalysisError) as error:
        oe.nardl_multipliers(fit, data=frame, method="residual", replications=20)
    assert error.value.code == "invalid_result"


@pytest.mark.parametrize("options", [
    {"method": "gaussian"}, {"method": []}, {"replications": True}, {"replications": 19},
    {"replications": 20.0}, {"seed": -1}, {"seed": 2**63}, {"seed": True},
    {"batch_size": 0}, {"batch_size": 1.0}, {"steps": True}, {"alpha": float("nan")},
])
def test_invalid_options_raise_analysis_error(options):
    frame, _, _ = data(n=80)
    fit = oe.nardl(data=frame, y="y", x=["x"], lags=[1, 0])
    kwargs = dict(method="residual", replications=20, seed=0, batch_size=4, steps=3)
    kwargs.update(options)
    with pytest.raises(AnalysisError) as error:
        oe.nardl_multipliers(fit, data=frame, **kwargs)
    assert error.value.code == "invalid_option"


@pytest.mark.parametrize("mutation", ["values", "dtype", "order", "missing"])
def test_bootstrap_requires_exact_original_data(mutation):
    frame, _, _ = data(n=80)
    fit = oe.nardl(data=frame, y="y", x=["x"], lags=[1, 0])
    changed = frame.copy()
    if mutation == "values":
        changed.loc[3, "x"] += 1
    elif mutation == "dtype":
        changed.x = changed.x.astype("float32")
    elif mutation == "order":
        changed = changed.iloc[::-1]
    else:
        changed = None
    with pytest.raises(AnalysisError) as error:
        oe.nardl_multipliers(fit, data=changed, method="residual", replications=20)
    assert error.value.code == ("missing_data" if mutation == "missing" else "data_mismatch")


@pytest.mark.parametrize("mutation", ["orders", "sample", "case", "constraint", "hash", "ssr"])
def test_corrupt_saved_design_or_restrictions_cannot_fabricate_bands(mutation):
    frame, _, _ = data(n=100)
    fit = oe.nardl(data=frame, y="y", x=["x"], lags=[2, 1], long_run_symmetric=["x"])
    changed = copy.deepcopy(fit)
    if mutation == "orders":
        changed.extra["lags"]["y"] = 1
    elif mutation == "sample":
        changed.sample_positions[0] = 1
    elif mutation == "case":
        changed.extra["case"] = 2
    elif mutation == "constraint":
        changed.extra["imposed_symmetry"]["long_run"] = []
    elif mutation == "hash":
        changed.provenance["sample_hash"] = "0" * 64
    else:
        changed.metrics["ssr"] *= 2
    with pytest.raises(AnalysisError) as error:
        oe.nardl_multipliers(changed, data=frame, method="wild", replications=20)
    assert error.value.code == "invalid_result"


def test_workspace_budget_bounds_batches_and_guards_path_storage(monkeypatch):
    import openecon.econometrics.tsmodels.nardl_bootstrap as bootstrap

    frame, _, _ = data(n=80)
    fit = oe.nardl(data=frame, y="y", x=["x"], lags=[1, 0])
    monkeypatch.setattr(bootstrap, "_MAX_DESIGN_BYTES", 32 * 1024)
    bands = oe.nardl_multipliers(fit, data=frame, method="residual", replications=20,
                                steps=1, batch_size=500)
    assert bands.attrs["bootstrap"]["batch_size_used"] < 500
    with pytest.raises(AnalysisError) as error:
        oe.nardl_multipliers(fit, data=frame, method="residual", replications=10000, steps=10)
    assert error.value.code == "bootstrap_capacity"


def test_one_failed_refit_never_returns_conditioned_or_replacement_bands(monkeypatch):
    import openecon.econometrics.tsmodels.nardl_bootstrap as bootstrap

    frame, _, _ = data(n=80)
    fit = oe.nardl(data=frame, y="y", x=["x"], lags=[1, 0])
    real = bootstrap._refit
    calls = 0

    def fail_second_batch(*args):
        nonlocal calls
        calls += 1
        if calls == 3:  # First call checks the original saved fit; then bootstrap batches.
            raise AnalysisError("bootstrap_failure", "Synthetic singular design")
        return real(*args)

    monkeypatch.setattr(bootstrap, "_refit", fail_second_batch)
    with pytest.raises(AnalysisError, match="Replicates 5..8") as error:
        oe.nardl_multipliers(fit, data=frame, method="residual", replications=20, batch_size=4)
    assert error.value.code == "bootstrap_failure"
    assert calls == 3


def test_unstable_original_fit_is_not_an_unvalidated_unit_root_bootstrap():
    frame, _, _ = data(n=80)
    fit = oe.nardl(data=frame, y="y", x=["x"], lags=[1, 0])
    fit.extra["levels_coefficients"]["L.y"] = 1.01
    with pytest.raises(AnalysisError) as error:
        oe.nardl_multipliers(fit, data=frame, method="wild", replications=20)
    assert error.value.code == "unstable_bootstrap"


def test_unstable_bootstrap_draws_are_retained_and_reported(monkeypatch):
    import openecon.econometrics.tsmodels.nardl_bootstrap as bootstrap

    frame, _, _ = data(n=80)
    fit = oe.nardl(data=frame, y="y", x=["x"], lags=[1, 0])
    real = bootstrap._refit
    index = list(fit.extra["levels_coefficients"]).index("L.y")
    calls = 0

    def force_unstable(*args):
        nonlocal calls
        calls += 1
        beta = real(*args)
        if calls > 1:
            beta[:, index] = 1.001
        return beta

    monkeypatch.setattr(bootstrap, "_refit", force_unstable)
    bands = oe.nardl_multipliers(fit, data=frame, method="wild", replications=20, steps=3)
    assert bands.attrs["bootstrap"]["completed"] == 20
    assert bands.attrs["bootstrap"]["unstable_refits_retained"] == 20
    assert any("unstable" in note for note in bands.attrs["warnings"])
