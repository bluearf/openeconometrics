"""Independent exact-tail, order-statistic and simplex geometry references."""
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import beta, binom

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest.distribution_free_joint import (
    multinomial_region, simultaneous_dkw_band, simultaneous_proportion_ci,
    simultaneous_quantile_ci,
)
from openecon.resources import use_workspace_budget


def test_dkw_complete_knots_ties_and_entire_line():
    df = pd.DataFrame({"b": [3., 1., 1., 2.], "a": [7., 7., 2., 7.]}, index=[8, 8, 2, 6])
    result = simultaneous_dkw_band(df, ["b", "a"], sampling_model="iid_marginals", alpha=.1)
    epsilon = math.sqrt(math.log(40)/8)
    assert result.attrs["epsilon"] == pytest.approx(epsilon)
    assert result["bands"].knot.tolist() == [1., 2., 3., 2., 7.]
    assert result["bands"].empirical_left.tolist() == [0., .5, .75, 0., .25]
    assert result["bands"].empirical_right.tolist() == [.5, .75, 1., .25, 1.]
    for row in result["bands"].itertuples():
        assert row.right_ci_low == pytest.approx(max(0, row.empirical_right-epsilon))
        assert row.right_ci_high == pytest.approx(min(1, row.empirical_right+epsilon))
    assert result.attrs["below_first"]["ci_high"] == pytest.approx(epsilon)
    assert result.attrs["above_last"]["ci_low"] == pytest.approx(1-epsilon)
    assert result["sample"].original_label.tolist() == [8, 8, 2, 6]


@pytest.mark.parametrize("n,p,alpha", list(itertools.product([1, 5, 20, 101], [.01, .25, .5, .9], [.01, .05, .1])))
def test_quantile_ranks_independent_binomial_tails(n, p, alpha):
    df = pd.DataFrame({"a": np.arange(n, dtype=float)})
    result = simultaneous_quantile_ci(df, ["a"], [p], sampling_model="iid_marginals", alpha=alpha)
    row = result["intervals"].iloc[0]
    guard = result.attrs["rank_cdf_roundoff_guard"]
    lower = max(r for r in range(n+1) if r == 0 or binom.cdf(r-1, n, p)+guard <= alpha/2)
    upper = min(s for s in range(1, n+2) if s == n+1 or binom.sf(s-1, n, p)+guard <= alpha/2)
    assert row.lower_rank == lower and row.upper_rank == upper
    assert row.lower_tail_probability == pytest.approx(binom.cdf(lower-1, n, p), abs=2e-12)
    assert row.upper_tail_probability == pytest.approx(binom.sf(upper-1, n, p), abs=2e-12)
    assert row.binomial_marginal_coverage >= 1-alpha-2e-12
    assert row.lower_unbounded == (lower == 0)
    assert row.upper_unbounded == (upper == n+1)
    assert row.estimate == math.ceil(n*p)-1
    if lower:
        assert row.ci_low == lower-1
    if upper <= n:
        assert row.ci_high == upper-1


def test_quantile_multiple_targets_discrete_ties_and_persistence():
    df = pd.DataFrame({"a": [0., 0., 0., 1., 1.], "b": [2., 3., 3., 3., 3.]})
    result = simultaneous_quantile_ci(df, ["b", "a"], [.1, .8], sampling_model="iid_marginals")
    assert result.attrs["family_size"] == 4
    assert result.attrs["per_tail_alpha"] == .05/8
    state = oe.summary_state(result)
    assert "Infinity" not in state and "NaN" not in state
    restored = oe.restore_summary(state)
    assert oe.summary_state(restored) == state
    assert restored["intervals"].lower_unbounded.any()
    assert "tabular" in restored["intervals"].to_latex()


@pytest.mark.parametrize("n,alpha", list(itertools.product([1, 5, 20, 100, 10000], [.01, .05, .1])))
def test_cp_family_against_independent_beta_quantiles(n, alpha):
    counts = sorted(set([0, 1, n//2, n-1, n]))
    result = simultaneous_proportion_ci(counts, [n]*len(counts), sampling_model="binomial_marginals", alpha=alpha)
    tail = alpha/(2*len(counts))
    for row in result["intervals"].itertuples():
        k = row.successes
        lo = 0 if k == 0 else beta.ppf(tail, k, n-k+1)
        hi = 1 if k == n else beta.isf(tail, k+1, n-k)
        assert row.ci_low == pytest.approx(lo, abs=2e-10)
        assert row.ci_high == pytest.approx(hi, abs=2e-10)


@pytest.mark.parametrize("p", [.001, .1, .4, .9, .999])
def test_cp_exhaustive_finite_sample_coverage(p):
    n = 12
    intervals = [simultaneous_proportion_ci([k], [n], sampling_model="binomial_marginals")["intervals"].iloc[0] for k in range(n+1)]
    coverage = sum(math.comb(n,k)*p**k*(1-p)**(n-k) for k, r in enumerate(intervals) if r.ci_low <= p <= r.ci_high)
    assert coverage >= .95-1e-12


@pytest.mark.parametrize("counts", [[3], [0, 8], [2, 3, 0, 7], [0, 0, 12], [3, 3, 3, 3]])
def test_multinomial_exact_coordinate_projection(counts):
    result = multinomial_region(counts, sampling_model="iid_multinomial", labels=[f"c{i}" for i in range(len(counts))])
    frame = result["region"]
    n, m = sum(counts), len(counts)
    tail = .05/(2*m)
    lo = np.array([0 if k == 0 else beta.ppf(tail,k,n-k+1) for k in counts])
    hi = np.array([1 if k == n else beta.isf(tail,k+1,n-k) for k in counts])
    for j, row in enumerate(frame.itertuples()):
        assert row.ci_low == pytest.approx(max(lo[j],1-sum(hi[i] for i in range(m) if i != j)), abs=2e-10)
        assert row.ci_high == pytest.approx(min(hi[j],1-sum(lo[i] for i in range(m) if i != j)), abs=2e-10)
        assert row.ci_low <= row.estimate <= row.ci_high
    assert "sum(" in result.attrs["constraint"]
    assert oe.summary_state(oe.restore_summary(oe.summary_state(result))) == oe.summary_state(result)


def test_multinomial_exhaustive_dependent_count_coverage():
    n, probs = 8, [0.2, .3, .5]
    coverage = 0
    for i in range(n+1):
        for j in range(n-i+1):
            counts = [i,j,n-i-j]
            frame = multinomial_region(counts, sampling_model="iid_multinomial")["region"]
            if all(r.ci_low <= p <= r.ci_high for r,p in zip(frame.itertuples(),probs)):
                mass = math.factorial(n)/math.prod(math.factorial(k) for k in counts)
                coverage += mass*math.prod(p**k for p,k in zip(probs,counts))
    assert coverage >= .95-1e-12


@pytest.mark.parametrize("method", [simultaneous_dkw_band, simultaneous_quantile_ci])
def test_complete_case_alignment_and_input_guards(method):
    kwargs = {"sampling_model":"iid_marginals"}
    if method is simultaneous_quantile_ci:
        kwargs["probabilities"] = [.5]
    data = pd.DataFrame({"a":[1.,math.nan,3.],"b":[2.,4.,math.nan]},index=[7,8,9])
    with pytest.raises(AnalysisError):
        method(data,["a","b"],**kwargs)
    result = method(data,["a","b"],missing="drop",**kwargs)
    assert result.attrs["included_positions"] == [0]
    assert result.attrs["excluded_positions"] == [1,2]
    for invalid in [pd.DataFrame({"a":[True]}),pd.DataFrame({"a":["1"]}),pd.DataFrame({"a":[math.inf]})]:
        with pytest.raises(AnalysisError):
            method(invalid,["a"],**kwargs)
    with pytest.raises(AnalysisError):
        method(pd.DataFrame([[1,2]],columns=["a","a"]),["a"],**kwargs)
    with pytest.raises(AnalysisError):
        method(pd.DataFrame({"a":range(10001)}),["a"],**kwargs)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        method(pd.DataFrame({"a":range(10000)}),["a"],**kwargs)


@pytest.mark.parametrize("counts", [[True], [1.], ["1"], [-1], [1000001], []])
def test_strict_integer_counts(counts):
    with pytest.raises(AnalysisError):
        multinomial_region(counts,sampling_model="iid_multinomial")
    with pytest.raises(AnalysisError):
        simultaneous_proportion_ci(counts,[5],sampling_model="binomial_marginals")


def test_unsupported_and_incompatible_contracts():
    with pytest.raises(AnalysisError):
        simultaneous_proportion_ci([6],[5],sampling_model="binomial_marginals")
    with pytest.raises(AnalysisError):
        multinomial_region([0,0],sampling_model="iid_multinomial")
    with pytest.raises(AnalysisError):
        simultaneous_dkw_band(pd.DataFrame({"a":[1.]}),["a"],sampling_model="normal")
    with pytest.raises(TypeError):
        multinomial_region([1,2],sampling_model="iid_multinomial",weights=[1,1])
    with pytest.raises(AnalysisError):
        simultaneous_quantile_ci(pd.DataFrame({"a":[1.]}),["a"],[0],sampling_model="iid_marginals")
    json.dumps(multinomial_region([0,4],sampling_model="iid_multinomial").attrs,allow_nan=False)


def test_default_device_context_cannot_redirect_cpu_procedures():
    data = pd.DataFrame({"a":[1.,2.,3.]})
    with torch.device("meta"):
        assert simultaneous_dkw_band(data,["a"],sampling_model="iid_marginals").attrs["device"] == "cpu"
        assert simultaneous_quantile_ci(data,["a"],[.5],sampling_model="iid_marginals").attrs["device"] == "cpu"
        assert simultaneous_proportion_ci([1],[3],sampling_model="binomial_marginals").attrs["device"] == "cpu"
        assert multinomial_region([1,2],sampling_model="iid_multinomial").attrs["device"] == "cpu"


def test_extreme_cp_family_tails_avoid_lower_quantile_cancellation():
    n, m, alpha = 1000000, 128, 1e-8
    counts = [0,1,n//2,n-1,n]*25+[0,1,n//2]
    result = simultaneous_proportion_ci(counts,[n]*m,sampling_model="binomial_marginals",alpha=alpha)
    tail = alpha/(2*m)
    for row in result["intervals"].itertuples():
        k = row.successes
        lo = 0 if k == 0 else beta.ppf(tail,k,n-k+1)
        hi = 1 if k == n else beta.isf(tail,k+1,n-k)
        assert row.ci_low == pytest.approx(lo,abs=2e-13)
        assert row.ci_high == pytest.approx(hi,abs=2e-13)


@pytest.mark.parametrize("method",[simultaneous_dkw_band,simultaneous_quantile_ci])
def test_preserve_sample_integer_targets_and_provenance_headers(method):
    kwargs={"sampling_model":"iid_marginals"}
    if method is simultaneous_quantile_ci:
        kwargs["probabilities"]=[.5]
    for value in [2**53+1,-2**53-1]:
        with pytest.raises(AnalysisError,match="safe range"):
            method(pd.DataFrame({"a":[value]*100}),["a"],**kwargs)
    with pytest.raises(AnalysisError,match="provenance"):
        method(pd.DataFrame({"original_position":[1.,2.]}),["original_position"],**kwargs)
    assert method(pd.DataFrame({"a":[2**53]*100}),["a"],**kwargs).attrs["n"] == 100


@pytest.mark.parametrize("label",[True,complex(1,2),"", "x"*257, math.inf, pd.Timestamp("2026-01-01")])
def test_count_family_bounded_identity_contract(label):
    with pytest.raises(AnalysisError):
        simultaneous_proportion_ci([1],[2],sampling_model="binomial_marginals",labels=[label])
    with pytest.raises(AnalysisError):
        multinomial_region([1],sampling_model="iid_multinomial",labels=[label])


def test_extended_float_object_sample_cannot_silently_merge_targets():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("This platform has no floating type wider than float64")
    value = np.longdouble(2**53)+np.longdouble(1)
    data = pd.DataFrame({"a":pd.Series([value]*100,dtype=object)})
    for method in (simultaneous_dkw_band,simultaneous_quantile_ci):
        kwargs={"sampling_model":"iid_marginals"}
        if method is simultaneous_quantile_ci:
            kwargs["probabilities"]=[.5]
        with pytest.raises(AnalysisError,match="scalar"):
            method(data,["a"],**kwargs)


def test_unsupported_identity_probability_and_array_options_raise_analysis_errors():
    data = pd.DataFrame({"a":[1.,2.]},index=pd.Index([10**1000,2],dtype=object))
    with pytest.raises(AnalysisError,match="Sample labels"):
        simultaneous_dkw_band(data,["a"],sampling_model="iid_marginals")
    data = pd.DataFrame({"a":[1.,2.]})
    with pytest.raises(AnalysisError):
        simultaneous_quantile_ci(data,["a"],[10**1000],sampling_model="iid_marginals")
    with pytest.raises(AnalysisError):
        simultaneous_dkw_band(data,["a"],sampling_model=np.array(["iid_marginals","invalid"]))
    with pytest.raises(AnalysisError):
        simultaneous_dkw_band(data,["a"],sampling_model="iid_marginals",missing=np.array(["raise","drop"]))
