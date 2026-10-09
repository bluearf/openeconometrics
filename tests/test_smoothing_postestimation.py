"""Independent basis, local-polynomial and finite-difference postestimation oracles."""

import itertools
import copy
import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline, CubicSpline
from scipy.stats import t, norm
import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget
from test_smoothing_eight import data, model, state, ols_oracle

HELPERS = {
    "bspline_regress": "spline_derivative",
    "rcs_regress": "spline_derivative",
    "fp_regress": "fp_derivative",
    "mfp_regress": "fp_derivative",
    "gam_gaussian": "gam_derivative",
    "mars": "mars_derivative",
    "loess": "loess_derivative",
    "npreg_mixed": "kernel_derivative",
}


def check_ci(out, estimate, basis, cov, df, alpha=0.05, gam=False):
    np.testing.assert_allclose(out.estimate, estimate, rtol=2e-7, atol=2e-9)
    se = np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", basis, cov, basis)))
    np.testing.assert_allclose(out.std_error, se, rtol=4e-7, atol=2e-9)
    critical = norm.ppf(1 - alpha / 2) if gam else t.ppf(1 - alpha / 2, df)
    np.testing.assert_allclose(out.ci_low, estimate - critical * se, rtol=4e-7, atol=2e-9)
    np.testing.assert_allclose(out.ci_high, estimate + critical * se, rtol=4e-7, atol=2e-9)
    np.testing.assert_allclose(out.attrs["linear_functionals"], basis, atol=2e-8)
    np.testing.assert_allclose(out.attrs["coefficient_covariance"], cov, atol=2e-8)
    if gam:
        assert "p_value" not in out and "smoothing bias" in out.attrs["inference"]
    else:
        np.testing.assert_allclose(
            out.p_value, 2 * t.sf(np.abs(estimate / se), df), rtol=2e-6, atol=1e-10
        )
        assert list(out.df) == [df] * len(out)


@pytest.mark.parametrize("degree,order", [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2)])
@pytest.mark.parametrize("knots", [[], [1.0, 2.0, 3.0]])
def test_bspline_derivative_independent_basis_endpoints(degree, order, knots):
    d = data()
    m = model("bspline_regress", d, degree=degree, knots={"x": knots}, boundary={"x": [0.25, 4.0]})
    q = np.array([0.25, 0.71, 1.38, 2.58, 4.0])
    tx = [0.25] * (degree + 1) + knots + [4.0] * (degree + 1)
    k = len(tx) - degree - 1
    spline = BSpline(tx, np.eye(k), degree)
    a = np.column_stack([np.ones(len(d)), spline(d.x)[:, 1:]])
    b, cov, df = ols_oracle(a, d.y.to_numpy())
    derivative = np.column_stack([np.zeros(len(q)), spline.derivative(order)(q)[:, 1:]])
    out = oe.spline_derivative(m, data=pd.DataFrame({"x": q}), variable="x", order=order)
    check_ci(out, derivative @ b, derivative, cov, df)
    assert "one-sided" in out.attrs["boundary"]


@pytest.mark.parametrize("order", [1, 2])
def test_rcs_derivative_independent_natural_space_and_tails(order):
    d = data()
    knots = [0.5, 1.0, 2.0, 3.0, 3.8]
    m = model("rcs_regress", d, knots={"x": knots})
    cs = CubicSpline(knots, np.eye(len(knots)), bc_type="natural", axis=0)

    def basis(q, nu):
        a = cs(q, nu)
        for i, v in enumerate(q):
            if v < knots[0] or v > knots[-1]:
                boundary = knots[0] if v < knots[0] else knots[-1]
                a[i] = (
                    cs(boundary) + (v - boundary) * cs(boundary, 1)
                    if nu == 0
                    else cs(boundary, 1)
                    if nu == 1
                    else np.zeros(len(knots))
                )
        return a

    a = basis(d.x.to_numpy(), 0)
    b, cov, df = ols_oracle(a, d.y.to_numpy())
    q = np.array([-4.0, 0.5, 0.8, 2.0, 3.8, 6.0])
    aq = basis(q, order)
    out = oe.spline_derivative(m, data=pd.DataFrame({"x": q}), variable="x", order=order)
    np.testing.assert_allclose(out.estimate, aq @ b, atol=2e-9)
    np.testing.assert_allclose(
        out.std_error, np.sqrt(np.einsum("ij,jk,ik->i", aq, cov, aq)), atol=2e-9
    )
    if order == 2:
        np.testing.assert_allclose(out.estimate.iloc[[0, -1]], 0, atol=1e-12)


@pytest.mark.parametrize(
    "powers",
    list(itertools.combinations_with_replacement([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], 2)),
)
@pytest.mark.parametrize("order", [1, 2])
def test_all_fp2_derivatives_scale_chain_and_full_inference(powers, order):
    d = data()
    m = oe.fp_regress(data=d, y="y", x=["x", "z"], powers=list(powers), scale=2.3)
    q = pd.DataFrame({"x": [0.6, 1.4, 2.9], "z": [-0.3, 0.2, 0.7]})

    # Complex-step first derivative and symmetric finite differences second;
    # evaluation only uses NumPy powers/logs and saved coefficients.
    def basis(v):
        z = v / 2.3
        parts = []
        for j, p in enumerate(powers):
            value = np.log(z) if p == 0 else z**p
            if j and p == powers[j - 1]:
                value = value * np.log(z)
            parts.append(value)
        return np.column_stack(parts)

    x = q.x.to_numpy()
    h = 1e-4
    derivative = (
        np.imag(basis(x + 1e-24j)) / 1e-24
        if order == 1
        else (basis(x + h) - 2 * basis(x) + basis(x - h)) / h**2
    )
    expected = np.column_stack([np.zeros(len(q)), derivative, np.zeros(len(q))])
    out = oe.fp_derivative(m, data=q, variable="x", order=order)
    b = np.asarray(state(m)["coefficients"])
    cov = np.asarray(state(m)["covariance"])
    np.testing.assert_allclose(
        out.estimate,
        expected @ b,
        rtol=4e-6 if order == 2 else 2e-8,
        atol=3e-5 if order == 2 else 2e-9,
    )
    analytic = np.asarray(out.attrs["linear_functionals"])
    np.testing.assert_allclose(
        analytic, expected, rtol=2e-5 if order == 2 else 2e-8, atol=3e-7 if order == 2 else 2e-9
    )
    check_ci(out, analytic @ b, analytic, cov, state(m)["df_resid"])
    linear = oe.fp_derivative(m, data=q, variable="z", order=order, interval=False)
    np.testing.assert_allclose(linear.estimate, b[-1] if order == 1 else 0, atol=1e-12)


@pytest.mark.parametrize("order", [1, 2])
def test_selected_mfp_and_excluded_variable(order):
    d = data()
    m = model("mfp_regress", d)
    q = pd.DataFrame({"x": [0.7, 1.4, 3.1]})
    out = oe.fp_derivative(m, data=q, variable="x", order=order)
    assert "selection uncertainty excluded" in out.attrs["inference"]
    d["y"] = np.random.default_rng(451).normal(size=len(d))
    null = model("mfp_regress", d, select_alpha=1e-8)
    assert state(null)["transforms"][0]["powers"] == []
    zero = oe.fp_derivative(null, data=q, variable="x", order=order)
    np.testing.assert_array_equal(zero.estimate, 0)
    np.testing.assert_array_equal(zero.std_error, 0)
    assert zero.p_value.isna().all()


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("penalty", [0.0, 0.5, 4.0])
def test_gam_derivative_center_and_frequentist_covariance(order, penalty):
    d = data()
    m = oe.gam_gaussian(data=d, y="y", x=["x", "z"], penalty=penalty, n_knots=2)
    s = state(m)
    q = pd.DataFrame({"x": [0.4, 1.8, 3.4], "z": [-0.2, 0.1, 0.8]})
    a = [np.zeros((len(q), 1))]
    for rec in s["transforms"]:
        degree = rec["degree"]
        tx = (
            [rec["boundary"][0]] * (degree + 1) + rec["knots"] + [rec["boundary"][1]] * (degree + 1)
        )
        k = len(tx) - degree - 1
        part = BSpline(tx, np.eye(k), degree).derivative(order)(q[rec["column"]])[:, 1:]
        a.append(part if rec["column"] == "x" else np.zeros_like(part))
    aq = np.column_stack(a)
    out = oe.gam_derivative(m, data=q, variable="x", order=order)
    check_ci(
        out, aq @ np.asarray(s["coefficients"]), aq, np.asarray(s["covariance"]), None, gam=True
    )
    # Query centering cannot be learned or shift derivatives.
    one = oe.gam_derivative(m, data=q.iloc[[1]], variable="x", order=order)
    assert one.estimate.iloc[0] == pytest.approx(out.estimate.iloc[1], abs=1e-12)


@pytest.mark.parametrize("degree,order", [(1, 1), (2, 1), (2, 2)])
def test_local_taylor_polynomial_reproduction_and_direct_numpy(degree, order):
    d = data()
    d["y"] = 2 + 3 * d.x + (1.2 * d.x**2 if degree == 2 else 0)
    m = model("loess", d, degree=degree, span=0.63)
    q = pd.DataFrame({"x": [0.25, 0.83, 2.16, 4.0]})
    out = oe.loess_derivative(m, data=q, variable="x", order=order)
    exact = (
        3 + 2.4 * q.x if degree == 2 and order == 1 else np.full(len(q), 2.4 if order == 2 else 3.0)
    )
    np.testing.assert_allclose(out.estimate, exact, atol=1e-11)
    # Noisy data: independent unscaled Taylor polynomial weighted least squares.
    noisy = data()
    m = model("loess", noisy, degree=degree, span=0.63)
    expected = []
    for v in q.x:
        distance = np.abs(noisy.x - v)
        radius = np.partition(distance, int(np.ceil(0.63 * len(noisy))) - 1)[
            int(np.ceil(0.63 * len(noisy))) - 1
        ]
        w = (1 - np.minimum(distance / radius, 1) ** 3) ** 3
        a = np.column_stack([(noisy.x - v) ** j for j in range(degree + 1)])
        b = np.linalg.lstsq(a * np.sqrt(w.to_numpy())[:, None], noisy.y * np.sqrt(w), rcond=None)[0]
        expected.append(b[order] * (2 if order == 2 else 1))
    np.testing.assert_allclose(
        oe.loess_derivative(m, data=q, variable="x", order=order).estimate, expected, atol=2e-11
    )
    assert "local Taylor" in out.attrs["target"] and "std_error" not in out


@pytest.mark.parametrize("degree", [1, 2])
def test_mars_interaction_derivative_and_active_knot_refusal(degree):
    d = data()
    d["y"] = (d.x - 1.1).clip(lower=0) * (d.z + 0.4).clip(lower=0) + 0.01 * np.sin(d.x)
    m = oe.mars(
        data=d,
        y="y",
        x=["x", "z"],
        max_degree=degree,
        max_terms=7,
        gcv_penalty=0,
        min_span=2,
        max_candidates=8,
    )
    q = pd.DataFrame({"x": [0.73, 1.57, 2.63], "z": [-0.2, 0.31, 0.67]})
    h = 1e-6
    for variable in ["x", "z"]:
        plus = q.copy()
        minus = q.copy()
        plus[variable] += h
        minus[variable] -= h
        expected = (
            oe.smoothing_predict(m, data=plus)["mean"] - oe.smoothing_predict(m, data=minus)["mean"]
        ) / (2 * h)
        out = oe.mars_derivative(m, data=q, variable=variable)
        np.testing.assert_allclose(out.estimate, expected, rtol=3e-7, atol=1e-9)
    hinge = next(a for term in state(m)["hinges"] for a in term)
    variable = state(m)["columns"][hinge["column_index"]]
    knot = q.iloc[[0]].copy()
    knot[variable] = hinge["knot"]
    # Choose all non-var factors active so the tie is identifiable.
    for term in state(m)["hinges"]:
        if hinge in term:
            for a in term:
                if a["column_index"] != hinge["column_index"]:
                    knot[state(m)["columns"][a["column_index"]]] = a["knot"] + a["sign"]
    with pytest.raises(AnalysisError, match="hinge"):
        oe.mars_derivative(m, data=knot, variable=variable)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("bw", [[0.3, 0.0, 0.0], [0.6, 0.3, 0.4], [1.1, 2 / 3, 0.7]])
def test_kernel_quotient_derivatives_finite_difference_and_categories(order, bw):
    m = model("npreg_mixed", bandwidth=bw, min_effective=1)
    q = pd.DataFrame(
        {"x": [0.71, 1.43, 2.97], "u": ["a", "b", "c"], "o": ["low", "middle", "high"]}
    )
    h = 1e-4 if order == 2 else 1e-5
    plus = q.copy()
    minus = q.copy()
    plus.x += h
    minus.x -= h
    p = oe.smoothing_predict(m, data=plus)["mean"].to_numpy()
    z = oe.smoothing_predict(m, data=q)["mean"].to_numpy()
    a = oe.smoothing_predict(m, data=minus)["mean"].to_numpy()
    expected = (p - a) / (2 * h) if order == 1 else (p - 2 * z + a) / h**2
    out = oe.kernel_derivative(m, data=q, variable="x", order=order)
    np.testing.assert_allclose(out.estimate, expected, rtol=1e-5, atol=1e-7)
    assert "std_error" not in out
    with pytest.raises(AnalysisError):
        oe.kernel_derivative(m, data=q, variable="u", order=order)
    bad = q.copy()
    bad.loc[0, "u"] = "unknown"
    with pytest.raises(AnalysisError):
        oe.kernel_derivative(m, data=bad, variable="x")


@pytest.mark.parametrize("name", list(HELPERS))
def test_json_replay_missing_rows_budgets_and_strict_options(name):
    m = model(name)
    restored = oe.ResultBundle.model_validate_json(m.model_dump_json())
    q = pd.DataFrame(
        {"x": [0.69, 1.23, 2.72], "u": ["a", "b", "c"], "o": ["low", "middle", "high"]},
        index=[99, 99, -3],
    )
    fn = getattr(oe, HELPERS[name])
    args = {"data": q, "variable": "x", "interval": False}
    expected = fn(m, **args)
    actual = fn(restored, **args)
    pd.testing.assert_frame_equal(actual, expected)
    q.loc[q.index == -3, "x"] = np.nan
    drop = fn(m, data=q, variable="x", interval=False, missing="drop")
    assert list(drop.row) == [0, 1] and drop.attrs["dropped_rows"] == 1
    with pytest.raises(AnalysisError):
        fn(m, data=q, variable="x")
    with pytest.raises(AnalysisError):
        fn(m, **args, max_work=1)
    with use_workspace_budget(128):
        with pytest.raises(AnalysisError):
            fn(m, **args)
    for bad in [0, 3, True, 1.5]:
        with pytest.raises(AnalysisError):
            fn(m, **args, order=bad)
    with pytest.raises(TypeError):
        fn(m, **args, device="cuda")
    corrupt = copy.deepcopy(m)
    corrupt.extra["smoothing_state"]["method"] = "wrong"
    with pytest.raises(AnalysisError):
        fn(corrupt, **args)
    if name in ["mars", "loess", "npreg_mixed"]:
        with pytest.raises(AnalysisError):
            fn(m, data=args["data"], variable="x", interval=True)


@pytest.mark.parametrize("name", list(HELPERS))
@pytest.mark.parametrize("variable", [None, "x"])
def test_average_margins_weighted_full_covariance_or_descriptive(name, variable):
    m = model(name)
    q = pd.DataFrame(
        {"x": [0.71, 1.47, 2.94], "u": ["a", "b", "c"], "o": ["low", "middle", "high"]}
    )
    w = np.array([1.0, 3.0, 2.0])
    w /= w.sum()
    infer = name not in ["loess", "mars", "npreg_mixed"]
    out = oe.smoothing_margins(
        m, data=q, variable=variable, averaging_weights=[1.0, 3.0, 2.0], interval=infer
    )
    row = (
        getattr(oe, HELPERS[name])(m, data=q, variable="x", interval=infer)
        if variable
        else oe.smoothing_predict(m, data=q, interval=infer)
    )
    expected = w @ (row.estimate if variable else row["mean"])
    assert out.estimate.iloc[0] == pytest.approx(expected, abs=1e-10)
    assert out.attrs["retained_positions"] == [0, 1, 2]
    if infer:
        basis = np.asarray(out.attrs["linear_functionals"])
        cov = np.asarray(state(m)["covariance"])
        check_ci(
            out,
            np.array([expected]),
            basis,
            cov,
            None if name == "gam_gaussian" else state(m)["df_resid"],
            gam=name == "gam_gaussian",
        )
        if np.ptp(row.std_error.to_numpy()) > 1e-5:
            assert out.std_error.iloc[0] != pytest.approx(float(w @ row.std_error), abs=1e-6)
    else:
        assert "std_error" not in out
    with pytest.raises(AnalysisError):
        oe.smoothing_margins(m, data=q, averaging_weights=[0, 0, 0])
    with pytest.raises(AnalysisError):
        oe.smoothing_margins(m, data=q, averaging_weights=[1, True, 2])


@pytest.mark.parametrize("name", list(HELPERS))
@pytest.mark.parametrize("average", [False, True])
def test_paired_contrasts_cross_profile_covariance_and_joint_missing(name, average):
    m = model(name)
    q = pd.DataFrame(
        {"x": [0.7, 1.4, 2.9], "u": ["a", "b", "c"], "o": ["low", "middle", "high"]},
        index=[88, 88, 1],
    )
    ref = q.copy()
    ref.x -= 0.1
    if name == "npreg_mixed":
        ref.u = ["b", "c", "a"]
    infer = name not in ["loess", "mars", "npreg_mixed"]
    result = oe.smoothing_contrast(m, data=q, reference=ref, average=average, interval=infer)
    expected = (
        oe.smoothing_predict(m, data=q)["mean"].to_numpy()
        - oe.smoothing_predict(m, data=ref)["mean"].to_numpy()
    )
    np.testing.assert_allclose(
        result.estimate, [expected.mean()] if average else expected, atol=1e-10
    )
    if infer:
        basis = np.asarray(result.attrs["linear_functionals"])
        cov = np.asarray(state(m)["covariance"])
        check_ci(
            result,
            np.array([expected.mean()]) if average else expected,
            basis,
            cov,
            None if name == "gam_gaussian" else state(m)["df_resid"],
            gam=name == "gam_gaussian",
        )
        # Same profile has identically zero contrast, including covariance.
        same = oe.smoothing_contrast(m, data=q, reference=q, interval=True)
        np.testing.assert_array_equal(same.estimate, 0)
        np.testing.assert_array_equal(same.std_error, 0)
    q.iloc[0, q.columns.get_loc("x")] = np.nan
    ref.iloc[1, ref.columns.get_loc("x")] = np.nan
    drop = oe.smoothing_contrast(m, data=q, reference=ref, average=average, missing="drop")
    assert drop.attrs["retained_positions"] == [2] and drop.attrs["dropped_rows"] == 2
    if not average:
        assert list(drop.row) == [2]
    with pytest.raises(AnalysisError):
        oe.smoothing_contrast(m, data=q, reference=ref.iloc[:2])


def test_low_degree_spline_nondifferentiable_knots():
    for degree, order in [(1, 1), (2, 2)]:
        m = model("bspline_regress", degree=degree, knots={"x": [1.0]})
        with pytest.raises(AnalysisError, match="discontinuous"):
            oe.spline_derivative(m, data={"x": [1.0]}, variable="x", order=order)
        if degree == 1:
            with pytest.raises(AnalysisError):
                oe.spline_derivative(m, data={"x": [0.8]}, variable="x", order=2)


@pytest.mark.parametrize("bad", ["negative", "asymmetric", "dimension", "df"])
def test_invalid_resealed_inference_state_is_rejected(bad):
    from openecon.econometrics.smoothing.common import seal

    m = model("bspline_regress")
    saved = copy.deepcopy(state(m))
    if bad == "negative":
        size = len(saved["coefficients"])
        saved["covariance"] = (-np.eye(size) * 1e-15).tolist()
    elif bad == "asymmetric":
        saved["covariance"][0][1] += 1
    elif bad == "dimension":
        saved["covariance"].pop()
    else:
        saved["df_resid"] = -1
    m.extra["smoothing_state"] = seal({k: v for k, v in saved.items() if k != "digest"})
    with pytest.raises(AnalysisError):
        oe.spline_derivative(m, data={"x": [0.7]}, variable="x")


def test_averaging_weights_use_original_positions_after_joint_drop():
    m = model("rcs_regress")
    q = pd.DataFrame({"x": [0.7, np.nan, 1.5, 2.7]})
    r = pd.DataFrame({"x": [0.6, 1.2, np.nan, 2.4]})
    out = oe.smoothing_contrast(
        m,
        data=q,
        reference=r,
        average=True,
        averaging_weights=[1.0, 999.0, 888.0, 3.0],
        missing="drop",
        interval=True,
    )
    assert out.attrs["retained_positions"] == [0, 3]
    assert out.attrs["normalized_averaging_weights"] == [0.25, 0.75]
    a = oe.smoothing_contrast(m, data=q.iloc[[0, 3]], reference=r.iloc[[0, 3]], interval=True)
    assert out.estimate.iloc[0] == pytest.approx(
        0.25 * a.estimate.iloc[0] + 0.75 * a.estimate.iloc[1]
    )
    assert out.attrs["dropped_rows"] == 2


@pytest.mark.parametrize("name", list(HELPERS))
def test_functionals_preserve_source_result_and_query(name):
    m = model(name)
    before = m.model_dump_json()
    q = pd.DataFrame(
        {"x": [0.71, 1.47, 2.94], "u": ["a", "b", "c"], "o": ["low", "middle", "high"]}
    )
    original = q.copy(deep=True)
    oe.smoothing_margins(m, data=q, variable="x")
    oe.smoothing_contrast(m, data=q, reference=q.copy())
    pd.testing.assert_frame_equal(q, original)
    assert m.model_dump_json() == before
