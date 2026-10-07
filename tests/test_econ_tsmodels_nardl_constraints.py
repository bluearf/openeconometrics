"""Independent constrained NumPy OLS, EC and lag-selection NARDL oracles.

Production uses a QR null-space reparameterization. This oracle instead uses
the equality-constrained OLS/Lagrange formula and its observation influence
matrix for HC covariances; no OpenEcon restriction/design helper is reused.
"""

import itertools
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import stats
from test_econ_tsmodels_nardl import data

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ModelSpec, ResultBundle


def matrices(frame, orders, *, names=("x",), trend=False, exog=(), holdback=None):
    n = len(frame)
    holdback = max(orders) if holdback is None else holdback
    terms = ["Intercept"]
    columns = [np.ones(n - holdback)]
    if trend:
        terms.append("trend")
        columns.append(np.arange(1, n + 1)[holdback:])
    for lag in range(1, orders[0] + 1):
        terms.append("L.y" if lag == 1 else f"L{lag}.y")
        columns.append(frame.y.to_numpy()[holdback - lag:n - lag])
    pairs = {}
    for j, name in enumerate(names):
        differences = np.r_[0.0, np.diff(frame[name].to_numpy())]
        pair = [f"{name}_positive", f"{name}_negative"]
        pairs[name] = pair
        for sign, expanded in enumerate(pair):
            series = (np.maximum(differences, 0) if sign == 0 else np.minimum(differences, 0)).cumsum()
            q = orders[1 + 2*j + sign]
            for lag in range(q + 1):
                term = expanded if lag == 0 else f"L.{expanded}" if lag == 1 else f"L{lag}.{expanded}"
                terms.append(term)
                columns.append(series[holdback - lag:n - lag])
    for name in exog:
        terms.append(name)
        columns.append(frame[name].to_numpy()[holdback:])
    return frame.y.to_numpy()[holdback:], np.column_stack(columns), terms, pairs


def restrictions(terms, orders, *, names=("x",), long=(), short=()):
    rows = []
    for j, name in enumerate(names):
        qp, qn = orders[1 + 2*j:3 + 2*j]
        if name in long:
            row = np.zeros(len(terms))
            for sign, kind, q in ((1, "positive", qp), (-1, "negative", qn)):
                for lag in range(q + 1):
                    term = f"{name}_{kind}" if lag == 0 else (
                        f"L.{name}_{kind}" if lag == 1 else f"L{lag}.{name}_{kind}")
                    row[terms.index(term)] = sign
            rows.append(row)
        if name in short and max(qp, qn):
            for gamma in range(max(qp, qn)):
                row = np.zeros(len(terms))
                for sign, kind, q in ((1, "positive", qp), (-1, "negative", qn)):
                    if gamma == 0:
                        row[terms.index(f"{name}_{kind}")] = sign
                    else:
                        for lag in range(gamma + 1, q + 1):
                            term = f"L.{name}_{kind}" if lag == 1 else f"L{lag}.{name}_{kind}"
                            row[terms.index(term)] = -sign
                rows.append(row)
    return np.array(rows).reshape(-1, len(terms))


def oracle(y, x, matrix, kind="nonrobust"):
    c = np.linalg.inv(x.T @ x)
    unconstrained = c @ x.T @ y
    rank = np.linalg.matrix_rank(matrix) if len(matrix) else 0
    middle = np.linalg.pinv(matrix @ c @ matrix.T) if rank else np.empty((0, 0))
    projection = c @ matrix.T @ middle @ matrix
    beta = unconstrained - projection @ unconstrained
    bread = c - projection @ c
    bread = (bread + bread.T) / 2
    residual = y - x @ beta
    n, k = x.shape
    free, df = k - rank, n - k + rank
    rss = residual @ residual
    if kind == "nonrobust":
        covariance = bread * rss / df
    else:
        leverage = np.einsum("ij,jk,ik->i", x, bread, x)
        adjusted = residual / np.sqrt(1 - leverage) if kind == "HC2" else (
            residual / (1 - leverage) if kind == "HC3" else residual)
        influence = bread @ x.T
        covariance = (influence * adjusted) @ (influence * adjusted).T
        if kind in {"HC1", "robust"}:
            covariance *= n / df
    ll = -0.5 * n * (1 + np.log(2*np.pi) + np.log(rss / n))
    classical_unrestricted = c * ((y - x @ unconstrained) @ (y - x @ unconstrained)) / (n - k)
    pre_covariance = classical_unrestricted
    if kind != "nonrobust":
        u = y - x @ unconstrained
        h = np.einsum("ij,jk,ik->i", x, c, x)
        adj = u / np.sqrt(1-h) if kind == "HC2" else u / (1-h) if kind == "HC3" else u
        influence = c @ x.T
        pre_covariance = (influence * adj) @ (influence * adj).T
        if kind in {"robust", "HC1"}:
            pre_covariance *= n / (n-k)
    contrast = matrix @ unconstrained
    pre_wald = contrast @ np.linalg.pinv(matrix @ pre_covariance @ matrix.T) @ contrast / rank if rank else None
    return {"beta": beta, "covariance": covariance, "free": free, "df": df, "rank": rank,
            "ssr": rss, "aic": -2*ll + 2*free, "bic": -2*ll + free*np.log(n),
            "residual": residual, "pre_wald": pre_wald}


@pytest.mark.parametrize("orders", [[2, 2, 1], [2, 2, 0]])
@pytest.mark.parametrize("kind", ["nonrobust", "HC1", "HC2", "HC3", "robust"])
@pytest.mark.parametrize("long,short", [(["x"], []), ([], ["x"]), (["x"], ["x"])])
def test_constrained_levels_full_covariance_and_df_match_numpy(orders, kind, long, short):
    frame, _, _ = data()
    y, x, terms, _ = matrices(frame, orders, trend=True, exog=["w"])
    matrix = restrictions(terms, orders, long=long, short=short)
    expected = oracle(y, x, matrix, kind)
    result = oe.nardl(data=frame, y="y", x=["x"], lags=orders, ec=False, trend="trend",
                      exog=["w"], covariance=kind, long_run_symmetric=long,
                      short_run_symmetric=short)
    assert_allclose(list(result.extra["levels_coefficients"].values()), expected["beta"],
                    rtol=1e-8, atol=2e-9)
    assert_allclose(result.extra["levels_covariance"], expected["covariance"], rtol=1e-7, atol=2e-10)
    assert_allclose(matrix @ np.array(list(result.extra["levels_coefficients"].values())), 0,
                    atol=2e-13)
    assert result.metrics["df_resid"] == result.inference["df_inference"] == expected["df"]
    assert result.metrics["aic"] == pytest.approx(expected["aic"], rel=1e-9)
    assert result.metrics["bic"] == pytest.approx(expected["bic"], rel=1e-9)
    assert result.metrics["df_constraints"] == expected["rank"]
    assert result.inference["n_parameters"] == expected["free"]
    for coefficient in result.coefficients:
        i = terms.index(coefficient.term)
        se = np.sqrt(expected["covariance"][i, i])
        assert coefficient.std_error == pytest.approx(se, rel=1e-7)
        assert coefficient.p_value == pytest.approx(2 * stats.t.sf(abs(expected["beta"][i] / se),
                                                                 expected["df"]), rel=1e-7, abs=1e-10)
    assert result.tests["imposed_symmetry"]["statistic"] == pytest.approx(expected["pre_wald"], rel=1e-7)
    assert result.tests["imposed_symmetry"]["df2"] == len(y) - x.shape[1]
    assert result.tests["imposed_symmetry"]["reference"].startswith("unrestricted")
    if long:
        assert result.tests["symmetry_long_run:x"]["imposed"] is True
        assert result.tests["symmetry_long_run:x"]["statistic"] is None
    if short:
        assert result.tests["symmetry_short_run_lagwise:x"]["imposed"] is True
        assert result.tests["symmetry_short_run:x"]["p_value"] is None
    if long and short and orders[-1] == 0:
        assert {"L.x_positive", "L2.x_positive"} <= set(result.extra["constrained_terms"])
        assert all(abs(value) < 1e-13 for value in result.extra["constrained_terms"].values())


def test_separate_predictor_subsets_only_restrict_the_requested_horizon():
    frame, _, _ = data(seed=3)
    orders = [2, 2, 1, 1, 2]
    y, x, terms, _ = matrices(frame, orders, names=("x", "z"))
    matrix = restrictions(terms, orders, names=("x", "z"), long=["x"], short=["z"])
    expected = oracle(y, x, matrix, "HC3")
    result = oe.nardl(data=frame, y="y", x=["x", "z"], lags=orders, ec=False,
                      long_run_symmetric=["x"], short_run_symmetric=["z"], covariance="HC3")
    assert_allclose(list(result.extra["levels_coefficients"].values()), expected["beta"],
                    rtol=1e-8, atol=1e-9)
    assert_allclose(result.extra["levels_covariance"], expected["covariance"], rtol=1e-7, atol=1e-10)
    assert result.tests["symmetry_long_run:x"]["imposed"] is True
    assert result.tests["symmetry_short_run_lagwise:z"]["imposed"] is True
    assert result.tests["symmetry_long_run:z"]["statistic"] is not None
    assert result.tests["symmetry_short_run_lagwise:x"]["statistic"] is not None
    unconstrained = oe.nardl(data=frame, y="y", x=["x", "z"], lags=orders, ec=False)
    assert not np.allclose(list(unconstrained.extra["levels_coefficients"].values()), expected["beta"])


@pytest.mark.parametrize("orders", [[2, 2, 1], [2, 2, 0]])
@pytest.mark.parametrize("long,short", [(["x"], []), ([], ["x"]), (["x"], ["x"])])
def test_constrained_ec_delta_covariance_matches_independent_numerical_jacobian(orders, long, short):
    frame, _, _ = data(seed=82)
    y, x, terms, _ = matrices(frame, orders)
    expected = oracle(y, x, restrictions(terms, orders, long=long, short=short), "HC3")
    result = oe.nardl(data=frame, y="y", x=["x"], lags=orders, covariance="HC3",
                      long_run_symmetric=long, short_run_symmetric=short)
    p, qp, qn = orders
    b = expected["beta"]

    def ec(parameters):
        a = 1 - parameters[1:1+p].sum()
        mapping = {"ADJ:L.y": -a, "SR:Intercept": parameters[0]}
        for lag in range(1, p):
            name = "LD.y" if lag == 1 else f"L{lag}D.y"
            mapping[f"SR:{name}"] = -parameters[1+lag:1+p].sum()
        for name, q in (("x_positive", qp), ("x_negative", qn)):
            indexes = [terms.index(name if lag == 0 else f"L.{name}" if lag == 1 else f"L{lag}.{name}")
                       for lag in range(q+1)]
            mapping[f"LR:{name}"] = parameters[indexes].sum() / a
            for lag in range(q):
                term = f"D.{name}" if lag == 0 else f"LD.{name}" if lag == 1 else f"L{lag}D.{name}"
                mapping[f"SR:{term}"] = -parameters[indexes[lag+1:]].sum()
        return np.array([mapping[c.term] for c in result.coefficients])

    jacobian = np.zeros((len(result.coefficients), len(b)))
    for j in range(len(b)):
        up, down = b.copy(), b.copy()
        step = 1e-6 * max(1, abs(b[j]))
        up[j], down[j] = up[j] + step, down[j] - step
        jacobian[:, j] = (ec(up) - ec(down)) / (2*step)
    assert_allclose([c.estimate for c in result.coefficients], ec(b), rtol=1e-8, atol=1e-9)
    assert_allclose(result.covariance_matrix, jacobian @ expected["covariance"] @ jacobian.T,
                    rtol=1e-7, atol=1e-9)
    assert result.inference["df_inference"] == expected["df"]


@pytest.mark.parametrize("ic", ["aic", "bic"])
@pytest.mark.parametrize("long,short", [(["x"], []), ([], ["x"]), (["x"], ["x"])])
def test_constrained_lag_search_matches_exhaustive_numpy_on_common_sample(ic, long, short):
    frame, _, _ = data(seed=19, n=110)
    records = []
    for orders in itertools.product(range(1, 3), range(3), range(3)):
        y, x, terms, _ = matrices(frame, orders, holdback=2)
        fit = oracle(y, x, restrictions(terms, orders, long=long, short=short))
        records.append((fit[ic], list(orders), fit))
    best = min(records, key=lambda record: record[0])
    result = oe.nardl(data=frame, y="y", x=["x"], maxlags=[2, 2], ic=ic,
                      long_run_symmetric=long, short_run_symmetric=short, ec=False)
    assert result.extra["lag_selection"]["constraints_in_selection"] is True
    # Equivalent zero-padded restricted models can tie; accept only actual minima.
    actual = result.extra["lag_selection"]["selected"]
    chosen = next(record for record in records if record[1] == actual)
    assert chosen[0] == pytest.approx(best[0], abs=1e-8)
    assert result.metrics[ic] == pytest.approx(best[0], abs=1e-8)
    assert result.metrics["df_resid"] == chosen[2]["df"]
    assert result.nobs == len(frame) - 2
    assert_allclose(list(result.extra["levels_coefficients"].values()), chosen[2]["beta"],
                    rtol=1e-8, atol=1e-9)


def test_both_horizon_symmetries_reduce_to_classical_ardl_and_zero_contrast_intervals():
    frame, _, _ = data(seed=36)
    symmetric = oe.nardl(data=frame, y="y", x=["x"], lags=[2, 2], long_run_symmetric=["x"],
                         short_run_symmetric=["x"], covariance="HC3", ec=False)
    classical = oe.ardl(data=frame, y="y", x=["x"], lags=[2, 2], covariance="HC3", ec=False)
    assert_allclose(symmetric.metrics["ssr"], classical.metrics["ssr"], rtol=1e-10)
    assert symmetric.metrics["df_resid"] == classical.metrics["df_resid"]
    assert symmetric.metrics["aic"] == pytest.approx(classical.metrics["aic"], rel=1e-10)
    assert_allclose([p["fitted"] for p in symmetric.predictions],
                    [p["fitted"] for p in classical.predictions], rtol=1e-9, atol=1e-10)
    multipliers = oe.nardl_multipliers(symmetric, steps=70)
    assert_allclose(multipliers.positive, multipliers.negative, rtol=1e-10, atol=1e-12)
    assert (multipliers.difference == 0).all()
    assert (multipliers.difference_std_error == 0).all()
    assert (multipliers.difference_ci_low == multipliers.difference_ci_high).all()


def test_unequal_both_symmetries_collapse_to_zero_order_classical_and_retain_fixed_lags():
    frame, _, _ = data()
    result = oe.nardl(data=frame, y="y", x=["x"], lags=[2, 2, 0], ec=False,
                      long_run_symmetric=["x"], short_run_symmetric=["x"])
    classical = oe.ardl(data=frame, y="y", x=["x"], lags=[2, 0])
    assert result.extra["constrained_terms"] == {"L.x_positive": 0.0, "L2.x_positive": 0.0}
    assert "L.x_positive" in result.extra["levels_coefficients"]
    assert "L.x_positive" not in [c.term for c in result.coefficients]
    assert_allclose(result.metrics["ssr"], classical.metrics["ssr"], rtol=1e-10)
    assert result.inference["df_inference"] == classical.inference["df_inference"]
    assert (oe.nardl_multipliers(result, steps=40).difference_std_error == 0).all()


def test_zero_order_short_run_request_is_vacuous_and_not_reported_as_a_test():
    frame, _, _ = data()
    ordinary = oe.nardl(data=frame, y="y", x=["x"], lags=[2, 0], ec=False)
    requested = oe.nardl(data=frame, y="y", x=["x"], lags=[2, 0], ec=False, short_run_symmetric=["x"])
    assert requested.extra["imposed_symmetry"]["vacuous_short_run"] == ["x"]
    assert requested.extra["constraints"]["rank"] == 0
    assert "symmetry_short_run" not in requested.tests
    assert "imposed_symmetry" not in requested.tests
    assert_allclose(requested.covariance_matrix, ordinary.covariance_matrix, rtol=1e-12)
    assert requested.metrics["df_resid"] == ordinary.metrics["df_resid"]


@pytest.mark.parametrize("kind", ["nonrobust", "HC3"])
@pytest.mark.parametrize("ec", [False, True])
def test_symmetry_identifies_rank_deficient_unrestricted_lags(kind, ec):
    frame, _, _ = data(seed=73, n=120)
    frame["x"] = np.arange(len(frame)) % 2
    noise = np.random.default_rng(381).normal(size=len(frame))
    observed = np.zeros(len(frame))
    for t in range(1, len(frame)):
        observed[t] = .5 + .35 * observed[t - 1] + 1.1 * frame.x.iloc[t] + .02*t + noise[t]
    frame["y"] = observed
    y, x, terms, _ = matrices(frame, [1, 0, 0], trend=True)
    assert np.linalg.matrix_rank(x) == x.shape[1] - 1
    # Independent explicit equality substitution: b_positive=b_negative.
    transform = np.zeros((5, 4))
    transform[:3, :3] = np.eye(3)
    transform[3:, 3] = 1
    reduced = x @ transform
    assert np.linalg.matrix_rank(reduced) == 4
    free_beta = np.linalg.lstsq(reduced, y, rcond=None)[0]
    beta = transform @ free_beta
    residual = y - reduced @ free_beta
    bread_free = np.linalg.inv(reduced.T @ reduced)
    bread = transform @ bread_free @ transform.T
    df = len(y) - reduced.shape[1]
    if kind == "nonrobust":
        covariance = bread * (residual @ residual) / df
    else:
        leverage = np.einsum("ij,jk,ik->i", reduced, bread_free, reduced)
        influence = bread @ x.T
        adjusted = residual / (1 - leverage)
        covariance = (influence * adjusted) @ (influence * adjusted).T
    result = oe.nardl(data=frame, y="y", x=["x"], lags=[1, 0], trend="trend",
                      long_run_symmetric=["x"], covariance=kind, ec=ec)
    assert_allclose([result.extra["levels_coefficients"][term] for term in terms], beta,
                    rtol=1e-8, atol=2e-9)
    assert_allclose(result.extra["levels_covariance"], covariance, rtol=1e-7, atol=2e-10)
    assert result.metrics["df_resid"] == result.inference["df_inference"] == df
    assert result.tests["imposed_symmetry"]["statistic"] is None
    assert result.tests["imposed_symmetry"]["reference"] == "unavailable"
    assert "rank-deficient" in result.tests["imposed_symmetry"]["note"]
    ordinary = oe.ardl(data=frame, y="y", x=["x"], lags=[1, 0], trend="trend",
                       covariance=kind, ec=ec)
    assert_allclose([point["fitted"] for point in result.predictions],
                    [point["fitted"] for point in ordinary.predictions], atol=1e-10)
    assert result.metrics["aic"] == pytest.approx(ordinary.metrics["aic"], abs=1e-9)
    selected = oe.nardl(data=frame, y="y", x=["x"], maxlags=[1, 0], trend="trend",
                        long_run_symmetric=["x"], covariance=kind, ec=ec)
    assert selected.extra["lag_selection"]["selected"] == [1, 0, 0]
    assert_allclose(list(selected.extra["levels_coefficients"].values()), beta, atol=2e-9)
    # A vacuous zero-order SR request cannot identify the same deficient design.
    with pytest.raises(AnalysisError) as error:
        oe.nardl(data=frame, y="y", x=["x"], lags=[1, 0], trend="trend",
                  short_run_symmetric=["x"], covariance=kind, ec=ec)
    assert error.value.code == "singular_design"


@pytest.mark.parametrize("options", [
    {"long_run_symmetric": ["absent"]}, {"short_run_symmetric": ["z"]},
    {"long_run_symmetric": ["x", "x"]}, {"short_run_symmetric": ["x", "x"]},
    {"long_run_symmetric": "x"}, {"short_run_symmetric": True},
])
def test_bad_subset_requests_fail_without_silent_relaxation(options):
    frame, _, _ = data()
    with pytest.raises(AnalysisError):
        oe.nardl(data=frame, y="y", x=["x", "z"], asymmetric=["x"], lags=[2, 1, 0], **options)


def test_generic_fit_json_and_original_row_hash_sample_survive_constraints():
    frame, _, _ = data()
    frame.loc[0, "x"] = np.nan
    shuffled = frame.sample(frac=1, random_state=88).reset_index(drop=True)
    spec = ModelSpec(estimator="nardl", outcome="y", predictors=["x"], time="t", missing="drop",
                     options={"lags": [2, 2, 0], "long_run_symmetric": ["x"],
                              "short_run_symmetric": ["x"], "ec": False})
    result = oe.fit(data=shuffled, spec=spec)
    saved = ResultBundle.model_validate_json(result.model_dump_json())
    assert saved.spec == spec and saved.sample_positions == result.sample_positions
    assert result.provenance["input_columns"] == ["y", "x", "t"]
    assert result.extra["partial_sum_origin"]["row"] == int(shuffled.index[shuffled.t == 1][0])
    assert result.dropped_rows == 3
    expected_positions = shuffled.loc[shuffled.t >= 3].sort_values("t").index.tolist()
    assert result.sample_positions == expected_positions
    assert_allclose(oe.nardl_multipliers(saved).positive, oe.nardl_multipliers(result).positive)
    unrestricted = oe.nardl(data=shuffled, y="y", x=["x"], time="t", lags=[2, 2, 0],
                           missing="drop", ec=False)
    assert result.provenance["data_hash"] == unrestricted.provenance["data_hash"]
    assert result.provenance["sample_hash"] == unrestricted.provenance["sample_hash"]
    json.dumps(json.loads(saved.model_dump_json()), allow_nan=False)
    assert "begin{tabular}" in str(saved.to_latex())
    with pytest.raises(ValidationError):
        ModelSpec(estimator="nardl", outcome="y", predictors=["x"],
                  options={"long_run_symmetric": "x"})


def test_corrupt_saved_constraint_covariance_is_rejected_by_multipliers():
    frame, _, _ = data()
    result = oe.nardl(data=frame, y="y", x=["x"], lags=[2, 2, 0],
                      long_run_symmetric=["x"], short_run_symmetric=["x"])
    result.extra["constraints"]["free_covariance"][0][0] *= 10
    with pytest.raises(AnalysisError) as caught:
        oe.nardl_multipliers(result)
    assert caught.value.code == "invalid_result"


def test_rescaled_saved_basis_cannot_silently_zero_all_multiplier_intervals():
    frame, _, _ = data()
    result = oe.nardl(data=frame, y="y", x=["x"], lags=[2, 2], long_run_symmetric=["x"])
    record = result.extra["constraints"]
    record["parameter_basis"] = (np.array(record["parameter_basis"]) * 1e-14).tolist()
    record["free_covariance"] = (np.array(record["free_covariance"]) * 1e28).tolist()
    with pytest.raises(AnalysisError) as caught:
        oe.nardl_multipliers(result)
    assert caught.value.code == "invalid_result"
