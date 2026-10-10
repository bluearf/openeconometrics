"""Independent Bernoulli QML derivatives and complete saved joint systems."""

import importlib

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import optimize, special

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest.scores import model_scores
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget

KINDS = ["cloglog", "logit", "probit"]


def data(n=240):
    rng = np.random.default_rng(9714)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    offset = rng.uniform(-0.25, 0.25, n)
    eta = -0.3 + 0.45 * x - 0.2 * z
    y = rng.binomial(1, -np.expm1(-np.exp(eta + offset)))
    frac = np.clip(special.expit(eta) + rng.normal(0, 0.22, n), 0, 1)
    frac[:4] = [0, 1, 0.17, 0.81]
    return pd.DataFrame(
        dict(
            x=x,
            z=z,
            y=y,
            frac=frac,
            off=offset,
            g=np.arange(n) % 16,
            f=rng.integers(1, 4, n),
            w=rng.uniform(0.4, 2.1, n),
        )
    )


def fitted(df, kind, weight_type=None, **kwargs):
    weights = None if weight_type is None else "f" if weight_type == "fweight" else "w"
    args = dict(data=df, x=["x", "z"], weights=weights, weight_type=weight_type, **kwargs)
    return (
        oe.cloglog(y="y", offset="off", **args)
        if kind == "cloglog"
        else oe.fracreg(y="frac", link=kind, **args)
    )


def raw_design(fit, df):
    sub = df.iloc[fit.sample_positions]
    columns = []
    for c in fit.coefficients:
        if c.term == "Intercept":
            columns.append(np.ones(len(sub)))
        elif "[" in c.term:
            name, level = c.term[:-1].split("[")
            columns.append((sub[name].astype(str) == level).to_numpy(dtype=float))
        else:
            columns.append(sub[c.term].to_numpy(dtype=float))
    return np.column_stack(columns)


def oracle(fit, df, theta=None):
    """Direct likelihood calculus; does not call any OpenEcon kernel/link/objective."""
    sub = df.iloc[fit.sample_positions]
    x = raw_design(fit, df)
    theta = np.array([c.estimate for c in fit.coefficients]) if theta is None else theta
    eta = x @ theta
    if fit.spec.estimator == "cloglog":
        eta = eta + sub.off.to_numpy()
    y = sub[fit.spec.outcome].to_numpy()
    w = (
        np.ones(len(sub))
        if fit.spec.weights is None
        else sub[fit.spec.weights].to_numpy(dtype=float)
    )
    if fit.spec.weight_type == "aweight":
        w = w * len(w) / w.sum()
    kind = fit.extra["link"]
    if kind == "logit":
        p = special.expit(eta)
        ll = y * eta - np.logaddexp(0, eta)
        s, h = y - p, -p * (1 - p)
    elif kind == "probit":
        logphi = -0.5 * eta**2 - 0.5 * np.log(2 * np.pi)
        a = np.exp(logphi - special.log_ndtr(eta))
        b = np.exp(logphi - special.log_ndtr(-eta))
        ll = y * special.log_ndtr(eta) + (1 - y) * special.log_ndtr(-eta)
        s = y * a - (1 - y) * b
        h = -y * a * (eta + a) - (1 - y) * b * (b - eta)
    else:
        t = np.exp(eta)
        a = t / np.expm1(t)
        ll = y * np.log(-np.expm1(-t)) - (1 - y) * t
        s = y * a - (1 - y) * t
        h = y * a * (1 - t - a) - (1 - y) * t
    return ll, x * (w * s)[:, None], x.T @ ((w * h)[:, None] * x), w


def jac(fn, theta, step=2e-5):
    identity = np.eye(len(theta))
    return np.column_stack(
        [(fn(theta + step * v) - fn(theta - step * v)) / (2 * step) for v in identity]
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("weight_type", [None, "fweight", "pweight", "aweight"])
def test_scores_observed_hessian_and_full_coefficients_against_independent_qml(kind, weight_type):
    if kind == "cloglog" and weight_type == "aweight":
        # This weight is outside the existing command's fitting contract.
        with pytest.raises(AnalysisError):
            fitted(data(), kind, weight_type)
        return
    df = data()
    fit = fitted(df, kind, weight_type)
    piece = model_scores(fit, df, fit.sample_positions)
    theta = np.array([c.estimate for c in fit.coefficients])
    _, score, hessian, w = oracle(fit, df)
    assert_allclose(piece.params, theta, rtol=0, atol=0)
    assert_allclose(piece.scores, score, rtol=2e-11, atol=2e-12)
    assert_allclose(piece.hessian, hessian, rtol=2e-11, atol=2e-11)
    numeric_score = jac(lambda t: oracle(fit, df, t)[0], theta) * w[:, None]
    numeric_hessian = jac(lambda t: oracle(fit, df, t)[1].sum(axis=0), theta)
    assert_allclose(score, numeric_score, rtol=1e-7, atol=2e-9)
    assert_allclose(hessian, numeric_hessian, rtol=1e-7, atol=2e-8)
    # Solve the independently coded complete score system from a different start.
    solved = optimize.root(
        lambda t: oracle(fit, df, t)[1].sum(axis=0),
        np.zeros(len(theta)),
        jac=lambda t: oracle(fit, df, t)[2],
        tol=1e-10,
    )
    assert solved.success
    assert_allclose(theta, solved.x, rtol=2e-6, atol=1e-7)
    assert np.linalg.norm(score.sum(axis=0)) < 2e-5
    assert piece.decrement() < 1e-6
    if kind == "probit":
        y = df.frac.to_numpy()
        x = raw_design(fit, df)
        eta = x @ theta
        binary_sign = 2 * y - 1
        binary_shortcut = binary_sign * np.exp(
            -0.5 * (binary_sign * eta) ** 2
            - 0.5 * np.log(2 * np.pi)
            - special.log_ndtr(binary_sign * eta)
        )
        assert np.max(np.abs(binary_shortcut - score[:, 0] / w)) > 0.01


def joint_oracle(fits, df, cluster):
    union = sorted(set().union(*(set(f.sample_positions) for f in fits)))
    p = sum(len(f.coefficients) for f in fits)
    score = np.zeros((len(union), p))
    bread = np.zeros((p, p))
    index = {row: i for i, row in enumerate(union)}
    col = 0
    for fit in fits:
        _, s, h, _ = oracle(fit, df)
        width = len(fit.coefficients)
        score[[index[row] for row in fit.sample_positions], col : col + width] = s
        bread[col : col + width, col : col + width] = np.linalg.inv(-h)
        col += width
    if cluster is None:
        freq = (
            df.f.iloc[union].to_numpy()
            if fits[0].spec.weight_type == "fweight"
            else np.ones(len(union))
        )
        effective_n = freq.sum()
        score = score / np.sqrt(freq)[:, None]
        meat = score.T @ score * effective_n / (effective_n - 1)
    else:
        codes, levels = pd.factorize(df[cluster].iloc[union], sort=False)
        summed = np.zeros((len(levels), p))
        np.add.at(summed, codes, score)
        meat = summed.T @ summed * len(levels) / (len(levels) - 1)
    return bread @ meat @ bread


@pytest.mark.parametrize("cluster", [None, "g"])
@pytest.mark.parametrize("weight_type", [None, "fweight", "pweight"])
def test_complete_three_model_crossblocks_partial_overlap_saved_state(
    cluster, weight_type, monkeypatch
):
    df = data()
    df.loc[::9, "y"] = np.nan
    df.loc[::11, "frac"] = np.nan
    fits = [fitted(df, kind, weight_type, missing="drop") for kind in KINDS]
    saved = [ResultBundle.model_validate_json(f.model_dump_json()) for f in fits]

    # Reconstruction evaluates derivatives at saved coefficients; it never refits.
    def refit_forbidden(*args, **kwargs):
        raise AssertionError("Saved suest must not fit or optimize again")

    monkeypatch.setattr(
        importlib.import_module("openecon.econometrics.glm.glm"), "estimate", refit_forbidden
    )
    monkeypatch.setattr(oe, "fit", refit_forbidden)
    joint = oe.suest(*saved, data=df, names=["c", "l", "p"], cluster=cluster)
    assert_allclose(
        joint.covariance_matrix, joint_oracle(fits, df, cluster), rtol=2e-10, atol=1e-12
    )
    assert_allclose(
        [c.estimate for c in joint.coefficients],
        [c.estimate for f in fits for c in f.coefficients],
        rtol=0,
        atol=0,
    )
    assert joint.sample_positions == sorted(set().union(*(set(f.sample_positions) for f in fits)))
    assert [c.term for c in joint.coefficients] == [
        f"{name}:{c.term}" for name, fit in zip(["c", "l", "p"], fits) for c in fit.coefficients
    ]
    assert joint.provenance["device"] == "cpu"
    assert joint.provenance["stata_parity_validated"] is False
    assert joint.provenance["resource_plan"]["operation"] == "joint binomial suest"
    again = ResultBundle.model_validate_json(joint.model_dump_json())
    assert again.covariance_matrix == joint.covariance_matrix
    test = oe.test(again, {"c:x": 1, "l:x": -1})
    beta = np.array([c.estimate for c in again.coefficients])
    contrast = np.zeros(len(beta))
    contrast[1], contrast[4] = 1, -1
    expected = (contrast @ beta) ** 2 / (contrast @ np.array(again.covariance_matrix) @ contrast)
    assert_allclose(test["statistic"], expected, rtol=1e-12)
    assert "tabular" in again.to_latex()
    assert all(term in again.summary() for term in ["c:x", "l:x", "p:x"])


@pytest.mark.parametrize("cluster", [None, "g"])
@pytest.mark.parametrize("kind", KINDS)
def test_frequency_scores_equal_literal_replication(kind, cluster):
    df = data()
    expanded = df.loc[df.index.repeat(df.f)].reset_index(drop=True)
    weighted = [fitted(df, kind, "fweight"), fitted(df, kind, "fweight", intercept=False)]
    plain = [fitted(expanded, kind), fitted(expanded, kind, intercept=False)]
    a = oe.suest(*weighted, data=df, cluster=cluster)
    b = oe.suest(*plain, data=expanded, cluster=cluster)
    assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=1e-8, atol=1e-10)
    assert_allclose(
        [c.estimate for c in a.coefficients],
        [c.estimate for c in b.coefficients],
        rtol=1e-7,
        atol=1e-8,
    )
    assert a.nobs == b.nobs == df.f.sum()


@pytest.mark.parametrize(
    "kind,weight_type",
    [
        ("cloglog", "pweight"),
        ("logit", "pweight"),
        ("probit", "pweight"),
        ("logit", "aweight"),
        ("probit", "aweight"),
    ],
)
@pytest.mark.parametrize("cluster", [None, "g"])
def test_weight_scale_invariance_with_unequal_samples(kind, weight_type, cluster):
    df = data()
    df.loc[::8, "z"] = np.nan

    def compute(frame):
        a = fitted(frame, kind, weight_type, missing="drop")
        # Same outcome, different selected design/missing sample.
        args = dict(data=frame, y=a.spec.outcome, x=["x"], weights="w", weight_type=weight_type)
        b = oe.cloglog(offset="off", **args) if kind == "cloglog" else oe.fracreg(link=kind, **args)
        joint = oe.suest(a, b, data=frame, cluster=cluster)
        assert_allclose(
            joint.covariance_matrix, joint_oracle([a, b], frame, cluster), rtol=1e-10, atol=1e-12
        )
        return joint

    a, b = compute(df), compute(df.assign(w=df.w * 29))
    assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=1e-7, atol=1e-10)


def test_disjoint_samples_zero_crossblocks_and_physical_positions():
    df = data()
    df.index = ["repeat"] * len(df)
    df.iloc[:120, df.columns.get_loc("y")] = np.nan
    df.iloc[120:, df.columns.get_loc("frac")] = np.nan
    a, b = fitted(df, "cloglog", missing="drop"), fitted(df, "probit", missing="drop")
    joint = oe.suest(a, b, data=df)
    assert joint.sample_positions == list(range(len(df)))
    assert_allclose(np.array(joint.covariance_matrix)[:3, 3:], 0, atol=0)
    assert_allclose(joint.covariance_matrix, joint_oracle([a, b], df, None), rtol=1e-10, atol=1e-12)


def test_categorical_unused_levels_omissions_no_intercept_and_permutation():
    df = data()
    df["cat"] = pd.Categorical(np.where(df.x > 0, "B", "A"), categories=["A", "B", "unused"])
    df["duplicate"] = df.x * 2
    fit = oe.fracreg(
        data=df,
        y="frac",
        x=["x", "duplicate", "cat"],
        categorical=["cat"],
        link="probit",
        intercept=False,
    )
    assert fit.provenance["omitted_terms"] == ["duplicate", "cat[unused]"]
    piece = model_scores(fit, df, fit.sample_positions)
    assert_allclose(piece.scores, oracle(fit, df)[1], rtol=1e-11, atol=1e-12)
    order = list(reversed(range(len(fit.coefficients))))
    permuted = fit.model_copy(deep=True)
    permuted.coefficients = [permuted.coefficients[i] for i in order]
    permuted.provenance["design_terms"] = [c.term for c in permuted.coefficients]
    permuted.covariance_matrix = np.array(permuted.covariance_matrix)[np.ix_(order, order)].tolist()
    reordered = model_scores(permuted, df, permuted.sample_positions)
    assert_allclose(reordered.scores, piece.scores[:, order], rtol=0, atol=0)
    assert_allclose(reordered.hessian, piece.hessian[np.ix_(order, order)], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("kind", KINDS)
def test_cpu_and_global_rng_state_preserved(kind):
    df = data()
    fit = fitted(df, kind)
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    from scripts.torch_test_state import preserve_torch_default_device
    with preserve_torch_default_device():
        torch.set_default_device("meta")
        piece = model_scores(fit, df, fit.sample_positions)
        joint = oe.suest(fit, fit, data=df)
        assert piece.scores.device.type == piece.hessian.device.type == "cpu"
        assert joint.provenance["device"] == "cpu"
        assert torch.get_default_device().type == "meta"
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
    assert np.array_equal(numpy_state[1], after[1])
    assert torch.equal(torch_state, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "mutation",
    ["link", "convergence", "categories", "omissions", "duplicate", "sample", "coefficient"],
)
def test_saved_contract_and_nonstationarity_refusals(mutation):
    df = data()
    fit = fitted(df, "probit").model_copy(deep=True)
    if mutation == "link":
        fit.extra["link"] = "logit"
    elif mutation == "convergence":
        fit.provenance["solver_diagnostics"]["converged"] = False
    elif mutation == "categories":
        fit.provenance["categorical_encoding"] = {"x": {"reference": 0}}
    elif mutation == "omissions":
        fit.provenance["omitted_terms"] = ["x"]
    elif mutation == "duplicate":
        fit.coefficients[1].term = fit.coefficients[0].term
    elif mutation == "sample":
        fit.sample_positions = [*fit.sample_positions[:-1], fit.sample_positions[0]]
    else:
        fit.coefficients[1].estimate += 0.2
    with pytest.raises(
        AnalysisError, match="saved|Saved|score|design|metadata|estimation"
    ) as caught:
        model_scores(fit, df, fit.sample_positions)
    assert caught.value.code == "suest_mismatch"


@pytest.mark.parametrize("kind", KINDS)
def test_stationary_subset_cannot_replace_saved_native_sample(kind):
    df = pd.DataFrame({
        "x": np.tile([-1., -1., 1., 1.], 20),
        "y": np.tile([0., 1., 0., 1.], 20),
        "frac": np.tile([.2, .8, .2, .8], 20),
        "off": np.zeros(80),
    })
    args = dict(data=df, x=["x"])
    fit = (oe.cloglog(y="y", offset="off", **args) if kind == "cloglog"
           else oe.fracreg(y="frac", link=kind, **args))
    bad = ResultBundle.model_validate_json(fit.model_dump_json())
    bad.sample_positions = bad.sample_positions[4:]
    # The removed balanced block has zero total score; the old stationarity
    # check accepted this smaller sample and silently changed its covariance.
    assert_allclose(oracle(fit, df)[1][4:].sum(axis=0), 0, atol=1e-12)
    for operation in (lambda: model_scores(bad, df, bad.sample_positions),
                      lambda: oe.suest(bad, bad, data=df)):
        with pytest.raises(AnalysisError) as caught:
            operation()
        assert caught.value.code == "suest_mismatch"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mutation", ["nobs", "nobs_original", "dropped_rows", "sample_hash", "missing_hash"])
def test_recorded_binomial_sample_counts_and_hash_required(kind, mutation):
    df = data()
    fit = fitted(df, kind).model_copy(deep=True)
    if mutation == "sample_hash":
        fit.provenance["sample_hash"] = "0" * 64
    elif mutation == "missing_hash":
        del fit.provenance["sample_hash"]
    else:
        setattr(fit, mutation, getattr(fit, mutation) + 1)
    with pytest.raises(AnalysisError) as caught:
        model_scores(fit, df, fit.sample_positions)
    assert caught.value.code == "suest_mismatch"


@pytest.mark.parametrize("kind,weight_type", [
    (kind, weight_type) for kind in KINDS
    for weight_type in ("fweight", "pweight", "aweight")
    if not (kind == "cloglog" and weight_type == "aweight")
])
def test_native_missing_zero_weight_categorical_samples_still_reconstruct(kind, weight_type):
    df = data()
    df["cat"] = pd.Categorical(np.where(df.x > 0, "B", "A"), categories=["A", "B", "unused"])
    df.loc[::11, "z"] = np.nan
    column = "f" if weight_type == "fweight" else "w"
    df.loc[::13, column] = 0
    args = dict(data=df, weights=column, weight_type=weight_type, missing="drop")
    fits = []
    for predictors in (["x", "z", "cat"], ["x"]):
        extra = dict(x=predictors, categorical=["cat"] if "cat" in predictors else [])
        fit = (oe.cloglog(y="y", offset="off", **extra, **args) if kind == "cloglog"
               else oe.fracreg(y="frac", link=kind, **extra, **args))
        fits.append(ResultBundle.model_validate_json(fit.model_dump_json()))
    assert fits[0].sample_positions != fits[1].sample_positions
    assert all(df[column].iloc[fit.sample_positions].gt(0).all() for fit in fits)
    joint = oe.suest(*fits, data=df)
    assert_allclose(joint.covariance_matrix, joint_oracle(fits, df, None), rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_importance_weights_and_changed_source_refused(kind):
    df = data()
    fit = fitted(df, kind)
    changed = df.copy()
    changed.iloc[0, changed.columns.get_loc("x")] += 0.1
    with pytest.raises(AnalysisError) as caught:
        oe.suest(fit, fit, data=changed)
    assert caught.value.code == "data_mismatch"
    bad = fit.model_copy(deep=True)
    bad.spec.weights = "w"
    bad.spec.weight_type = "iweight"
    with pytest.raises(AnalysisError) as caught:
        model_scores(bad, df, bad.sample_positions)
    assert caught.value.code == "suest_unsupported"


def test_individual_and_joint_guards_precede_score_reconstruction(monkeypatch):
    df = data()
    fit = fitted(df, "probit")
    module = importlib.import_module("openecon.econometrics.postest.scores")

    def forbidden(*args, **kwargs):
        raise AssertionError("reconstruction must not start")

    monkeypatch.setattr(module, "_frame", forbidden)
    monkeypatch.setattr(module, "MAX_BINOMIAL_SCORE_WORK", 1)
    with pytest.raises(AnalysisError) as caught:
        model_scores(fit, df, fit.sample_positions)
    assert caught.value.code == "suest_work_limit"
    monkeypatch.setattr(module, "MAX_BINOMIAL_SCORE_WORK", 100_000_000)
    # 20-model joint system exceeds byte budget before any provider is called.
    with use_workspace_budget(1), pytest.raises(AnalysisError) as caught:
        oe.suest(*([fit] * 20), data=df)
    assert caught.value.code == "workspace_limit"
    assert (
        not hasattr(caught.value, "resource_plan")
        or caught.value.resource_plan["operation"] == "joint binomial suest"
    )
    joint_module = importlib.import_module("openecon.econometrics.postest.suest")
    monkeypatch.setattr(joint_module, "MAX_BINOMIAL_JOINT_WORK", 1)
    with pytest.raises(AnalysisError) as caught:
        oe.suest(fit, fit, data=df)
    assert caught.value.code == "suest_work_limit"


@pytest.mark.parametrize("kind", KINDS)
def test_large_offset_information_refusal_has_rescaling_remedy(kind):
    df = data()
    df.x += 1e9
    fit = fitted(df, kind)
    assert fit.provenance["solver_diagnostics"]["converged"]
    with pytest.raises(AnalysisError, match="Center/rescale") as caught:
        model_scores(fit, df, fit.sample_positions)
    assert caught.value.code == "singular_information"


@pytest.mark.parametrize("kind", KINDS)
def test_large_offset_refused_even_when_rounded_information_looks_positive(kind, monkeypatch):
    from openecon.engines.optimize import information_inverse

    df = pd.DataFrame({
        "x": 1e9 + np.tile([-1., -1., 1., 1.], 20),
        "z": np.zeros(80), "off": np.zeros(80),
        "y": np.tile([0., 1., 0., 1.], 20),
        "frac": np.tile([.2, .8, .2, .8], 20),
    })
    fit = fitted(df, kind)
    _, scores, hessian, _ = oracle(fit, df)
    assert len(fit.coefficients) == 2
    # An O(eps) positive error in the normalized cross-product is larger than
    # the real O(1e-18) information eigenvalue. A Cholesky-only admission can
    # accept this rounded SPD matrix and its stationary scores on some BLASes.
    norms = np.sqrt(np.diag(-hessian))
    delta = 16 * np.finfo(float).eps
    rounded = norms[:, None] * np.array([[1., 1. - delta], [1. - delta, 1.]]) * norms
    bread = information_inverse(torch.tensor(rounded, dtype=torch.float64))
    gradient = torch.tensor(scores.sum(axis=0), dtype=torch.float64)
    assert float(gradient @ bread @ gradient) < 1e-6
    module = importlib.import_module("openecon.econometrics.postest.scores")
    monkeypatch.setattr(module, "information_inverse", lambda unused: bread)
    with pytest.raises(AnalysisError, match="Center/rescale") as caught:
        model_scores(fit, df, fit.sample_positions)
    assert caught.value.code == "singular_information"


@pytest.mark.parametrize("kind", KINDS)
def test_information_refusal_precedes_rounded_gram_reconstruction(kind, monkeypatch):
    from openecon.econometrics.glm.kernels import GlmObjective

    df = data()
    df.x += 1e9
    fit = fitted(df, kind)

    def forbidden(*args):
        raise AssertionError("Unsafe reported coordinates must be refused before forming X'WX")

    monkeypatch.setattr(GlmObjective, "__call__", forbidden)
    with pytest.raises(AnalysisError) as caught:
        model_scores(fit, df, fit.sample_positions)
    assert caught.value.code == "singular_information"


@pytest.mark.parametrize("kind", KINDS)
def test_centering_restores_valid_full_saved_information(kind):
    df = data()
    df.x += 1e9
    df.x -= df.x.mean()
    fit = fitted(df, kind)
    piece = model_scores(fit, df, fit.sample_positions)
    assert_allclose(piece.scores, oracle(fit, df)[1], rtol=2e-11, atol=2e-12)
    joint = oe.suest(fit, fit, data=df)
    assert_allclose(joint.covariance_matrix, joint_oracle([fit, fit], df, None), rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("weight_type", [None, "pweight"])
def test_information_admission_preserves_predictor_units_and_probability_weight_scale(kind, weight_type):
    df = data()
    scaled = df.assign(x=df.x * 1e6, z=df.z * 1e-6, w=df.w * 29)
    fit, changed = fitted(df, kind, weight_type), fitted(scaled, kind, weight_type)
    joint = oe.suest(fit, fit, data=df)
    other = oe.suest(changed, changed, data=scaled)
    transform = np.tile([1., 1e-6, 1e6], 2)
    expected = np.array(joint.covariance_matrix) * transform[:, None] * transform
    assert_allclose(other.covariance_matrix, expected, rtol=2e-7, atol=1e-12)
    assert_allclose([c.estimate for c in other.coefficients],
                    np.array([c.estimate for c in joint.coefficients]) * transform,
                    rtol=2e-6, atol=1e-10)


def test_individual_byte_guard_before_frame(monkeypatch):
    df = data(5000)
    fit = fitted(df, "probit")
    module = importlib.import_module("openecon.econometrics.postest.scores")

    def forbidden(*args, **kwargs):
        raise AssertionError("reconstruction must not start")

    monkeypatch.setattr(module, "_frame", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as caught:
        model_scores(fit, df, fit.sample_positions)
    assert caught.value.code == "workspace_limit"
    assert caught.value.resource_plan["operation"] == "saved binomial suest scores"


@pytest.mark.parametrize("kind", KINDS)
def test_invalid_nonfinite_coefficients_and_outcome_domain_refused(kind):
    df = data()
    fit = fitted(df, kind)
    bad = fit.model_copy(deep=True)
    bad.coefficients[1].estimate = float("nan")
    with pytest.raises(AnalysisError):
        model_scores(bad, df, bad.sample_positions)
    changed = df.copy()
    changed[fit.spec.outcome] = changed[fit.spec.outcome].astype(float)
    changed.loc[0, fit.spec.outcome] = 0.5 if kind == "cloglog" else 1.2
    with pytest.raises(AnalysisError) as caught:
        model_scores(fit, changed, fit.sample_positions)
    assert caught.value.code == "suest_mismatch"


def test_mixed_original_ols_and_new_providers_keep_full_parameter_system():
    df = data()
    df["continuous"] = 0.5 + 0.3 * df.x - 0.1 * df.z + np.sin(np.arange(len(df)))
    old = oe.ols(data=df, y="continuous", x=["x", "z"])
    new = fitted(df, "probit")
    joint = oe.suest(old, new, data=df, names=["o", "p"], cluster="g")
    x = raw_design(old, df)
    beta = np.array([c.estimate for c in old.coefficients])
    residual = df.continuous.to_numpy() - x @ beta
    sigma2 = residual @ residual / len(df)
    old_score = np.column_stack(
        [x * (residual / sigma2)[:, None], 0.5 * (residual**2 / sigma2 - 1)]
    )
    old_h = np.zeros((4, 4))
    old_h[:3, :3] = -(x.T @ x) / sigma2
    old_h[:3, 3] = old_h[3, :3] = -(x.T @ residual) / sigma2
    old_h[3, 3] = -0.5 * np.sum(residual**2 / sigma2)
    _, new_score, new_h, _ = oracle(new, df)
    score = np.column_stack([old_score, new_score])
    summed = np.zeros((16, 7))
    np.add.at(summed, df.g.to_numpy(), score)
    bread = np.zeros((7, 7))
    bread[:4, :4] = np.linalg.inv(-old_h)
    bread[4:, 4:] = np.linalg.inv(-new_h)
    expected = bread @ (summed.T @ summed) * 16 / 15 @ bread
    assert_allclose(joint.covariance_matrix, expected, rtol=1e-10, atol=1e-12)
    assert [c.term for c in joint.coefficients] == [
        "o:Intercept",
        "o:x",
        "o:z",
        "o:/lnvar",
        "p:Intercept",
        "p:x",
        "p:z",
    ]
    assert [c.equation for c in joint.coefficients][:4] == ["o_mean", "o_mean", "o_mean", "o_lnvar"]


def test_dataset_cluster_and_weight_contract_refusals():
    from openecon.dataset import Dataset

    df = data()
    fit = fitted(df, "probit")
    streamed = Dataset.from_batches(lambda: iter([df]), list(df.columns), row_count=len(df))
    with pytest.raises(AnalysisError) as caught:
        oe.suest(fit, fit, data=streamed)
    assert caught.value.code == "streaming_unsupported"
    for column, code in [
        ("missing", "missing_columns"),
        ("one", "insufficient_clusters"),
        ("empty", "missing_values"),
    ]:
        changed = df.assign(one=1, empty=np.nan)
        with pytest.raises(AnalysisError) as caught:
            oe.suest(fit, fit, data=changed, cluster=column)
        assert caught.value.code == code
    other = fitted(df, "probit", "pweight")
    with pytest.raises(AnalysisError) as caught:
        oe.suest(fit, other, data=df)
    assert caught.value.code == "suest_unsupported"


def test_new_provider_native_weight_and_role_domains_preserved():
    df = data()
    for kind, role in [("cloglog", "exposure"), ("probit", "offset")]:
        fit = fitted(df, kind)
        fit.spec.columns[role] = "off"
        with pytest.raises(AnalysisError) as caught:
            model_scores(fit, df, fit.sample_positions)
        assert caught.value.code == "suest_unsupported"
    fit = fitted(df, "cloglog")
    fit.spec.weights, fit.spec.weight_type = "w", "aweight"
    with pytest.raises(AnalysisError) as caught:
        model_scores(fit, df, fit.sample_positions)
    assert caught.value.code == "suest_unsupported"


@pytest.mark.parametrize("wide_kind", ["collinear", "categorical", "mutated_spec"])
def test_prescreen_work_guard_precedes_design_allocation_and_rank_screen(wide_kind, monkeypatch):
    from openecon.econometrics.core import ModelFrame
    from openecon.econometrics.postest.scores import MAX_BINOMIAL_SCORE_WORK

    df = data()
    if wide_kind == "collinear":
        # A valid native fit with hundreds of genuinely omitted columns and a
        # tiny reported vector. Fitting remains within the native byte budget.
        names = [f"duplicate{i}" for i in range(450)]
        df = pd.concat(
            [df, pd.DataFrame({name: df.x * (i + 1) for i, name in enumerate(names)})], axis=1
        )
        fit = oe.fracreg(data=df, y="frac", x=["x", *names], link="probit")
        assert len(fit.coefficients) == 2
        assert len(fit.provenance["omitted_terms"]) == len(names)
    elif wide_kind == "categorical":
        # Explicit unused levels belong to the saved treatment coding and are
        # part of the full pre-screen design even though they are all omitted.
        df["category"] = pd.Categorical(
            np.where(df.x > 0, "B", "A"), categories=["A", "B", *[f"unused{i}" for i in range(450)]]
        )
        fit = oe.fracreg(
            data=df, y="frac", x=["x", "category"], categorical=["category"], link="probit"
        )
        assert len(fit.coefficients) == 3
        assert len(fit.provenance["omitted_terms"]) == 450
    else:
        fit = fitted(df, "probit").model_copy(deep=True)
        fit.spec.predictors = ["x"] * 451
    frame = ModelFrame(fit.spec, df)
    raw_width = frame.design_width()
    assert len(fit.coefficients) <= 3
    assert frame.n * raw_width**2 + raw_width**3 > MAX_BINOMIAL_SCORE_WORK
    calls = []

    def forbidden_design(*args, **kwargs):
        calls.append("design")
        raise AssertionError("raw design allocation must not start")

    def forbidden_rank(*args, **kwargs):
        calls.append("rank")
        raise AssertionError("rank screening must not start")

    monkeypatch.setattr(ModelFrame, "design", forbidden_design)
    monkeypatch.setattr(ModelFrame, "drop_collinear", forbidden_rank)
    with pytest.raises(AnalysisError) as caught:
        model_scores(fit, df, fit.sample_positions)
    assert caught.value.code == "suest_work_limit"
    assert calls == []


@pytest.mark.parametrize(
    "kind",
    [
        "ols",
        "logit",
        "probit",
        "poisson",
        "ologit",
        "oprobit",
        "mlogit",
        "glm",
        "nbreg",
        "tobit",
        "intreg",
        "truncreg",
        "cloglog",
        "fracreg",
    ],
)
def test_joint_admission_width_includes_every_reported_nuisance_parameter(kind):
    from openecon.econometrics.postest.suest import _binomial_joint_plan

    rng = np.random.default_rng(9381)
    df = data()
    latent = 0.4 + 0.3 * df.x - 0.1 * df.z + rng.normal(size=len(df))
    df["continuous"] = latent
    df["ordered"] = np.digitize(latent, [-0.8, 0.2, 1.1])
    df["choice"] = rng.choice([1, 2, 3], size=len(df))
    mu = np.exp(0.2 + 0.2 * df.x - 0.1 * df.z)
    df["count"] = rng.poisson(mu)
    df["negative_binomial"] = rng.negative_binomial(2, 2 / (2 + mu))
    df["censored"] = np.maximum(latent, 0)
    df["lo"], df["hi"] = latent - 0.25, latent + 0.25
    args = dict(data=df, x=["x", "z"])
    if kind in ["cloglog", "fracreg"]:
        fit = fitted(df, "cloglog" if kind == "cloglog" else "probit")
    elif kind == "ols":
        fit = oe.ols(y="continuous", **args)
    elif kind in ["logit", "probit"]:
        fit = getattr(oe, kind)(y="y", **args)
    elif kind == "poisson":
        fit = oe.poisson(y="count", **args)
    elif kind in ["ologit", "oprobit"]:
        fit = getattr(oe, kind)(y="ordered", **args)
    elif kind == "mlogit":
        fit = oe.mlogit(y="choice", **args)
    elif kind == "glm":
        fit = oe.glm(y="continuous", family="gaussian", link="identity", **args)
    elif kind == "nbreg":
        fit = oe.nbreg(y="negative_binomial", **args)
    elif kind == "tobit":
        fit = oe.tobit(y="censored", ll=0, **args)
    elif kind == "intreg":
        fit = oe.intreg(y_low="lo", y_high="hi", **args)
    else:
        fit = oe.truncreg(y="continuous", ll=0, **args)
    piece = model_scores(fit, df, fit.sample_positions)
    width = len(fit.coefficients) + (kind == "ols")
    assert len(piece.params) == len(piece.terms) == len(piece.equations) == width
    assert piece.scores.shape == (len(fit.sample_positions), width)
    assert piece.hessian.shape == (width, width)
    if kind in ["ologit", "oprobit"]:
        assert sum(term.startswith("/cut") for term in piece.terms) == 3
    elif kind == "nbreg":
        assert any(term.startswith("/ln") for term in piece.terms)
    elif kind in ["tobit", "intreg", "truncreg"]:
        assert any("sigma" in term for term in piece.terms)
    elif kind == "mlogit":
        assert len(set(piece.equations)) == 2
    other = fitted(df, "probit")
    other_piece = model_scores(other, df, other.sample_positions)
    plan = _binomial_joint_plan(
        [fit, other], [(df, fit.sample_positions), (df, other.sample_positions)]
    )
    full_width = width + len(other_piece.terms)
    assert plan.record()["buffers"]["joint_covariance_copies"] == full_width**2 * 8 * 10
    assert (
        plan.record()["buffers"]["retained_model_scores"]
        == (piece.scores.numel() + other_piece.scores.numel()) * 8 * 4
    )
