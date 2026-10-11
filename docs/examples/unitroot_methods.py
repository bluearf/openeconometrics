"""Run Ng--Perron and KSS on an owned, synthetic series in the code panel."""

import json
import random
import sys

import openecon as oe
from openecon.analysis_contracts import AnalysisError

rng = random.Random(136137)
levels = [0.0]
for _ in range(249):
    levels.append(levels[-1] + rng.gauss(0, 1))
frame = oe.DataFrame({"period": range(len(levels)), "y": levels})
ng_constant = oe.ngperron(frame, "y", time="period", maxlag=8, inference="published")
ng_plain = oe.ngperron(frame, "y", time="period", maxlag=8)
assert ng_plain.attrs["critical_values"] is None
assert ng_plain.attrs["reject"] is None
assert ng_plain.attrs["distribution"] is None
assert ng_plain.attrs["statistics"] == ng_constant.attrs["statistics"]
try:
    oe.ngperron(frame, "y", time="period", trend="trend", lags=2, inference="published")
except AnalysisError as error:
    assert error.code == "critical_values_unverified"
else:
    raise AssertionError("Unverified trend inference was accepted")
ng_trend = oe.ngperron(frame, "y", time="period", trend="trend", lags=2)
kss_raw = oe.kss(frame, "y", time="period", trend="none", lags=1)
kss_constant = oe.kss(frame, "y", time="period")
kss_trend = oe.kss(frame, "y", time="period", trend="trend", lags=2)
tables = [ng_constant, ng_trend, kss_raw, kss_constant, kss_trend]
assert [len(t) for t in tables] == [4, 4, 1, 1, 1]
for result in tables:
    assert result.attrs["p_value"] is None
    assert result.attrs["n_levels"] == 250
    decided = result.attrs["critical_values"] is not None
    assert (
        "asymptotic" in result.attrs["distribution"]
        if decided
        else result.attrs["distribution"] is None
    )
    assert "\\begin{tabular}" in result.to_latex()
    for row in result.to_dict("records"):
        for percent in (1, 5, 10) if decided else ():
            assert row[f"reject_{percent}pct"] == (row["statistic"] < row[f"critical_{percent}pct"])
    render = globals().get("display")
    if render is not None:
        render(result)
    else:
        print(result.to_string(index=False))
print(
    "UNITROOT_METHODS_RECEIPT:"
    + json.dumps(
        {
            "issues": ["MARKET-136", "MARKET-137"],
            "sdk_version": oe.__version__,
            "frozen": bool(getattr(sys, "frozen", False)),
            "row_counts": [len(result) for result in tables],
            "inputs": frame.to_dict("list"),
            "default_statistics_unchanged": True,
            "trend_published_refused": True,
            "results": [
                {
                    "test": result.attrs["test"],
                    "trend": result.attrs["trend"],
                    "lags": result.attrs["lags"],
                    "nobs": result.attrs["nobs"],
                    "rows": result.to_dict("records"),
                    "attrs": result.attrs,
                }
                for result in tables
            ],
            "calibration": "KSS and Ng-Perron constant: published asymptotic quantiles. Ng-Perron trend: statistics only.",
        },
        allow_nan=False,
    )
)
