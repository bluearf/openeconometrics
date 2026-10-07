"""Summarize synthetic QA timing/resource receipts without inspecting UI or user data."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        return [json.loads(row) for row in file if row.strip()]


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}

    def quantile(q):
        index = (len(values) - 1) * q
        low = math.floor(index)
        high = math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)

    return {
        "n": len(values),
        "p50": quantile(.5),
        "p95": quantile(.95),
        "min": values[0],
        "max": values[-1],
    }


def summarize(directory):
    events = read(directory / "events.jsonl.gz")
    resources = read(directory / "resources.jsonl.gz")
    assert all(r.get("cpu_timebase", {}).get("denom", 0) > 0 for r in resources), "CPU clocks must be normalized."
    controls = read(directory / "controls.jsonl")
    executions = json.loads((directory / "executions.json").read_text())
    first = executions[0]
    runs = [e for e in events if e["phase"] == "run"]
    # Match one completed UI paint to the preceding durable execution record.
    # This excludes the separate fourteen-model proof from the OLS workload.
    matched = []
    for event in runs:
        preceding = [r for r in executions if 0 <= event["epoch_ms"] - r["completed_epoch_ms"] < 2000]
        if preceding:
            record = min(preceding, key=lambda r: event["epoch_ms"] - r["completed_epoch_ms"])
            matched.append((event, record))
    assert len(matched) == len(runs), "Every run paint must have a completed execution."
    assert len({r["id"] for _, r in matched}) == len(matched)
    cold = [e["duration_ms"] for e, r in matched if r["id"] == first["id"]]
    warm = [(e, r) for e, r in matched if r["id"] != first["id"] and r["kind"] == "starter_ols_480_hc3_scatter"]
    shell = [e for e in events if e["phase"] == "shell"]
    startups = []
    for event in shell:
        candidates = [r for r in resources if r["native_start_epoch_ms"] <= event["epoch_ms"]]
        latest = max(candidates, key=lambda r: r["native_start_epoch_ms"])
        startups.append(event["epoch_ms"] - latest["native_start_epoch_ms"])
    groups = {}
    scenarios = ["local_off", "local_on", "latency_250", "latency_1000"]
    for scenario in scenarios:
        cohort = [e for e in events if e["scenario"] == scenario]
        phases = {p: distribution([e["duration_ms"] for e in cohort if e["phase"] == p])
                  for p in ("project", "typing", "file", "suggestion")}
        phases["warm_run"] = distribution([e["duration_ms"] for e, _ in warm if e["scenario"] == scenario])
        phases["worker_compute"] = distribution([r["duration_ms"] for e, r in warm if e["scenario"] == scenario])
        intervals = [(c["epoch_ms"], controls[i + 1]["epoch_ms"] if i + 1 < len(controls) else float("inf"))
                     for i, c in enumerate(controls) if c["scenario"] == scenario]
        samples = [r for r in resources if any(a <= r["epoch_ms"] < b for a, b in intervals)]
        active = [r for r in samples if any(e["epoch_ms"] - e["duration_ms"] - 250 <= r["epoch_ms"] <= e["epoch_ms"] + 250 for e in cohort)]
        phases["resources"] = {
            "samples": len(samples),
            "rss_mib": distribution([r["rss_bytes"] / 2**20 for r in samples]),
            "cpu_all_cores_percent": distribution([r["cpu_percent"] for r in samples]),
            "active_samples": len(active),
            "active_cpu_all_cores_percent": distribution([r["cpu_percent"] for r in active]),
            "peak_rss_by_role_mib": {role: max(p["rss_bytes"] / 2**20 for r in samples for p in r["processes"] if p["role"] == role)
                                     for role in sorted({p["role"] for r in samples for p in r["processes"]})},
        }
        groups[scenario] = phases
    return {
        "quantiles": "Empirical method 7; linear interpolation; small sample counts shown.",
        "first_profile_native_start_to_catalog_ms": startups[0],
        "repeat_native_start_to_catalog_ms": distribution(startups[1:6]),
        "restart_validation_native_start_to_catalog_ms": startups[6],
        "web_navigation_to_catalog_ms": distribution([e["duration_ms"] for e in shell[:6]]),
        "first_python_worker_ms": first["duration_ms"],
        "first_python_to_result_paint_ms": cold[0],
        "scenarios": groups,
        "resource_samples": len(resources),
        "observed_total_rss_peak_mib": max(r["rss_bytes"] / 2**20 for r in resources),
        "limits": [
            "Fresh isolated profile first launch; OS file caches were not flushed.",
            "Two animation frames after React state update approximate paint; physical display scanout is not measured.",
            "RSS sums every owned process including responsibility-attributed WebKit XPC helpers; shared pages can be counted multiple times.",
            "CPU sums all process cores; 100 percent means one full core and values may exceed 100.",
            "250/1000 ms are per-request loopback middleware delays, an emulated WAN proxy, not measured cloud networking.",
            "Suggestions use a preseeded vendor-hash-verified 531 MB model; download time is excluded.",
            "Observed synthetic baseline; no universal memory or latency service-level budget is asserted.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    receipt = summarize(args.directory)
    (args.directory / "performance.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"resource_samples": receipt["resource_samples"], "peak_rss_mib": receipt["observed_total_rss_peak_mib"]}))
