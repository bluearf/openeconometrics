"""Independent fixed-time pivots, full-path crossing and portable CS contracts."""

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta, chi2, gamma, norm, t
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget

METHODS = [
    ("bernoulli", "bernoulli_confidence_sequence", "iid_bernoulli"),
    ("poisson", "poisson_confidence_sequence", "iid_poisson"),
    ("normal_mean", "normal_mean_confidence_sequence", "iid_normal"),
    ("student_mean", "student_mean_confidence_sequence", "iid_normal"),
    ("normal_variance", "normal_variance_confidence_sequence", "iid_normal"),
    ("exponential_mean", "exponential_mean_confidence_sequence", "iid_exponential"),
    ("uniform_endpoint", "uniform_endpoint_confidence_sequence", "iid_uniform_zero"),
    ("hoeffding", "hoeffding_confidence_sequence", "iid_bounded"),
]


def sample(kind, n=12):
    rng = np.random.default_rng(817)
    a = rng.uniform(.1, .9, (n, 2))
    if kind == "bernoulli":
        a = (a > .4).astype(int)
    elif kind == "poisson":
        a = rng.poisson(2., (n, 2))
    elif kind in ("normal_mean", "student_mean", "normal_variance"):
        a = rng.normal(1., 1.4, (n, 2))
    elif kind == "exponential_mean":
        a = rng.exponential(2., (n, 2))
    return pd.DataFrame(a, columns=["second", "first"], index=[f"unit-{i//2}" for i in range(n)])


def invoke(kind, api, model, data=None, **kwargs):
    data = sample(kind) if data is None else data
    options = {"sampling_model": model, **kwargs}
    if kind == "normal_mean":
        options.setdefault("sd", {"second": 1.4, "first": 2.})
    if kind == "hoeffding":
        options.setdefault("bounds", {"second": (0., 1.), "first": (0., 1.)})
    return getattr(oe, api)(data, ["second", "first"], **options)


@pytest.mark.parametrize("kind,api,model", METHODS)
@pytest.mark.parametrize("alpha", [.001, .05, .4])
def test_every_prefix_against_independent_scipy_pivots(kind, api, model, alpha):
    data = sample(kind)
    result = invoke(kind, api, model, data, alpha=alpha)
    assert len(result["intervals"]) == 2*len(data)
    for row in result["intervals"].itertuples():
        n, j = row.n, list(data.columns).index(row.variable)
        x = data.iloc[:n, j].to_numpy()
        tail = alpha/(2*2*n*(n+1))
        estimate = x.mean()
        if kind == "bernoulli":
            k = int(x.sum())
            low = 0 if k == 0 else beta.ppf(tail, k, n-k+1)
            high = 1 if k == n else beta.isf(tail, k+1, n-k)
        elif kind == "poisson":
            k = int(x.sum())
            low = 0 if k == 0 else chi2.ppf(tail, 2*k)/(2*n)
            high = chi2.isf(tail, 2*(k+1))/(2*n)
        elif kind == "normal_mean":
            radius = norm.isf(tail)*[1.4, 2.][j]/np.sqrt(n)
            low, high = estimate-radius, estimate+radius
        elif kind in ("student_mean", "normal_variance"):
            if n == 1:
                assert row.upper_unbounded
                assert row.lower_unbounded == (kind == "student_mean")
                assert row.status == "insufficient_prefix"
                continue
            variance = x.var(ddof=1)
            if kind == "student_mean":
                radius = t.isf(tail, n-1)*np.sqrt(variance/n)
                low, high = estimate-radius, estimate+radius
            else:
                estimate = variance
                low = (n-1)*variance/chi2.isf(tail, n-1)
                high = (n-1)*variance/chi2.ppf(tail, n-1)
        elif kind == "exponential_mean":
            low = x.sum()/gamma.isf(tail, n)
            high = x.sum()/gamma.ppf(tail, n)
        elif kind == "uniform_endpoint":
            estimate = x.max()
            low = estimate/(1-tail)**(1/n)
            high = estimate/tail**(1/n)
        else:
            radius = np.sqrt(np.log(1/tail)/(2*n))
            low, high = max(0., estimate-radius), min(1., estimate+radius)
        assert row.estimate == pytest.approx(estimate, rel=2e-13, abs=1e-14)
        assert row.ci_low == pytest.approx(low, rel=2e-10, abs=2e-12)
        assert row.ci_high == pytest.approx(high, rel=2e-10, abs=2e-12)
        assert row.local_alpha == alpha/(2*n*(n+1))
        assert not row.lower_unbounded and not row.upper_unbounded
    assert result.attrs["infinite_horizon_alpha_budget"] == alpha
    assert result.attrs["remaining_alpha_budget"] == alpha/(len(data)+1)


@pytest.mark.parametrize("kind,api,model", METHODS)
def test_prefix_extension_does_not_recalibrate_earlier_looks(kind, api, model):
    data = sample(kind)
    short = invoke(kind, api, model, data.iloc[:5])["intervals"]
    full = invoke(kind, api, model, data)["intervals"].iloc[:10]
    pd.testing.assert_frame_equal(pd.DataFrame(short), pd.DataFrame(full), check_exact=True)


@pytest.mark.parametrize("p", [.01, .1, .3, .5, .9, .99])
def test_exhaustive_first_crossing_probability_not_pointwise_coverage(p):
    nmax, alpha = 9, .4
    limits = {}
    for n in range(1, nmax+1):
        for k in range(n+1):
            data = pd.DataFrame({"x": [1]*k+[0]*(n-k)})
            row = oe.bernoulli_confidence_sequence(data, ["x"], sampling_model="iid_bernoulli", alpha=alpha)["intervals"].iloc[-1]
            limits[n, k] = (row.ci_low, row.ci_high)
    crossing = 0.
    for path in itertools.product((0, 1), repeat=nmax):
        counts = np.cumsum(path)
        if any(not limits[n, int(counts[n-1])][0] <= p <= limits[n, int(counts[n-1])][1] for n in range(1, nmax+1)):
            k = sum(path)
            crossing += p**k*(1-p)**(nmax-k)
    assert crossing <= alpha*nmax/(nmax+1)+1e-12


@pytest.mark.parametrize("kind,api,model", METHODS)
def test_full_state_replay_and_typed_duplicate_sample_order(kind, api, model):
    data = sample(kind)
    data.index = pd.Index([1, "1", None, ("a", 2), 1, "x", 3, 4, 5, 6, 7, 8], tupleize_cols=False)
    result = invoke(kind, api, model, data)
    state = oe.summary_state(result)
    restored = oe.restore_summary(state)
    assert oe.summary_state(restored) == state
    assert restored["sample"].position.tolist() == list(range(12))
    identities = restored["sample"].index_identity.tolist()
    assert identities[0] != identities[1] and identities[0] == identities[4]
    assert restored.attrs["time_uniform"] and restored.attrs["optional_stopping_valid"]
    assert restored.attrs["fixed_family_required"] and not restored.attrs["nested_intervals"]
    assert "tabular" in restored["intervals"].to_latex()
    json.dumps(json.loads(state), allow_nan=False)


@pytest.mark.parametrize("api", ["student_mean_confidence_sequence", "normal_variance_confidence_sequence"])
def test_constant_gaussian_prefix_never_reports_false_certainty(api):
    result = getattr(oe, api)(pd.DataFrame({"x": [3.]*4}), ["x"], sampling_model="iid_normal")
    frame = result["intervals"]
    assert frame.upper_unbounded.all()
    assert frame.status.tolist() == ["insufficient_prefix"]+["zero_observed_variation"]*3
    assert frame.lower_unbounded.all() == api.startswith("student")


@pytest.mark.parametrize("kind,api,model", METHODS)
def test_admission_and_no_silent_fallback(kind, api, model):
    data = sample(kind)
    for alpha in [True, .0001, .5, math.nan, "0.05"]:
        with pytest.raises(AnalysisError):
            invoke(kind, api, model, data, alpha=alpha)
    for bad in [data.iloc[:0], pd.concat([data]*43), data.assign(second=math.nan),
                data.assign(second=math.inf), data.assign(second=True), data.assign(second="1"),
                data.assign(second=1e101), data.assign(second=2**53+1)]:
        with pytest.raises(AnalysisError):
            invoke(kind, api, model, bad)
    with pytest.raises(AnalysisError):
        invoke(kind, api, "undeclared_model", data)
    with pytest.raises(TypeError):
        invoke(kind, api, model, data, weights=[1]*len(data))
    with pytest.raises(TypeError):
        invoke(kind, api, model, data, missing="drop")
    large = pd.concat([sample(kind, 512)]*4, axis=1)
    large.columns = [f"x{i}" for i in range(8)]
    options = {"sampling_model": model}
    if kind == "normal_mean":
        options["sd"] = [1.]*8
    if kind == "hoeffding":
        options["bounds"] = [(0., 1.)]*8
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        getattr(oe, api)(large, list(large.columns), **options)


def test_support_count_and_scale_options_are_declared_not_inferred():
    with pytest.raises(AnalysisError):
        oe.bernoulli_confidence_sequence(pd.DataFrame({"x": [.5]}), ["x"], sampling_model="iid_bernoulli")
    for values in [[-.1], [.5], [1_000_001], [500_001, 500_001]]:
        with pytest.raises(AnalysisError):
            oe.poisson_confidence_sequence(pd.DataFrame({"x": values}), ["x"], sampling_model="iid_poisson")
    for api, model in [("exponential_mean_confidence_sequence", "iid_exponential"),
                       ("uniform_endpoint_confidence_sequence", "iid_uniform_zero")]:
        for values in [[0.], [-1.], [1e-101]]:
            with pytest.raises(AnalysisError):
                getattr(oe, api)(pd.DataFrame({"x": values}), ["x"], sampling_model=model)
    for sd in [[0.], [-1.], [True], [math.inf], [1e101], [1e-101], {"other": 1.}]:
        with pytest.raises(AnalysisError):
            oe.normal_mean_confidence_sequence(pd.DataFrame({"x": [1.]}), ["x"], sd=sd, sampling_model="iid_normal")
    with pytest.raises(AnalysisError):
        oe.hoeffding_confidence_sequence(pd.DataFrame({"x": [1.1]}), ["x"], bounds=[(0., 1.)], sampling_model="iid_bounded")


@pytest.mark.parametrize("kind,api,model", METHODS)
def test_declared_resource_boundary_and_extreme_tail_reference(kind, api, model):
    data = sample(kind, 512)
    result = invoke(kind, api, model, data, alpha=.001)
    assert len(result["intervals"]) == 1024
    assert result["intervals"].n.max() == 512
    assert result["intervals"].local_alpha.min() == .001/(2*512*513)
    assert len(result["sample"]) == 512
    assert result.attrs["device"] == "cpu"


def test_resident_cpu_is_not_redirected_by_default_device():
    with torch.device("meta"):
        result = oe.poisson_confidence_sequence(pd.DataFrame({"x": [0, 1, 0]}), ["x"], sampling_model="iid_poisson")
    assert result.attrs["device"] == "cpu"
    assert result["intervals"].ci_high.gt(0).all()
