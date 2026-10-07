"""Paste into the code panel: bounded charts from a million-row resident table."""
import json
import math

import numpy as np
import openecon as oe

n = 1_000_003
frame = oe.DataFrame({"x": np.arange(n, dtype="float64"), "y": np.arange(n, dtype="float64") % 97})
frame.loc[::997, "x"] = np.nan
frame.loc[::1237, "y"] = np.inf
histogram = oe.plot.hist(data=frame, x="x", bins=40, title="All finite values · resident histogram")
scatter = oe.plot.scatter(data=frame, x="x", y="y", title="Full counts · 2,000 displayed pairs")
missing_x = (n - 1) // 997 + 1
missing_y = (n - 1) // 1237 + 1
missing_both = (n - 1) // math.lcm(997, 1237) + 1
assert histogram.total_n == n - missing_x
assert sum(point["count"] for point in histogram.data) == histogram.total_n
assert scatter.total_n == n - missing_x - missing_y + missing_both
assert scatter.sample_n == len(scatter.data) == 2000
assert histogram.config["processing"]["source_rows"] == scatter.config["processing"]["source_rows"] == n
try:
    oe.plot.line(data=frame, x="x", y="y")
except ValueError as error:
    assert "10,000" in str(error)
else:
    raise AssertionError("Oversized line must be refused before materialization")
gaps = oe.plot.line(data={"x": [10, 1, 5, None, 25, 20], "y": [3, 1, None, 4, 8, 6]},
                    x="x", y="y", title="Irregular distances and preserved gaps")
render = globals().get("display")
if render is not None:
    render(oe.DataFrame([{"chart": "Histogram", "finite_rows": histogram.total_n, "dropped_rows": histogram.dropped_n},
                         {"chart": "Scatter", "finite_rows": scatter.total_n, "dropped_rows": scatter.dropped_n}]))
    render(histogram)
    render(scatter)
    render(gaps)
print("RESIDENT_CHARTS_RECEIPT:" + json.dumps({"source_rows": n,
    "histogram_total_n": histogram.total_n, "histogram_count_sum": sum(p["count"] for p in histogram.data),
    "scatter_total_n": scatter.total_n, "scatter_sample_n": scatter.sample_n,
    "scatter_full_extents": scatter.config["processing"]["extents"],
    "large_line_refused": True, "line_gap_rows": len(gaps.data), "sdk_version": oe.__version__,
}, allow_nan=False))
