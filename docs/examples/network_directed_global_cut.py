"""Paste into the code panel and Run: directed global cut with exact certificate.

The least outgoing cut is {'1'} -> {1, 'a'}, with capacity 2. The source side
intentionally does not contain integer 1, the smallest typed label.
"""
from fractions import Fraction
import json

import openecon as oe


graph = oe.network([
    {"source": 1, "target": "1", "capacity": 9},
    {"source": "1", "target": 1, "capacity": 1},
    {"source": "1", "target": 1, "capacity": 1},
    {"source": 1, "target": "a", "capacity": 10},
    {"source": "a", "target": 1, "capacity": 10},
    {"source": "1", "target": "1", "capacity": 100},
], nodes=[1, "1", "a"], weight="capacity", directed=True)
cut = graph.global_min_cut(max_work=100_000)
exact = Fraction(cut["metadata"]["exact_capacity_numerator"],
                 cut["metadata"]["exact_capacity_denominator"])
assert cut["value"] == 2 and exact == 2
assert cut["source_partition"] == ("1",)
assert sum((Fraction(row.capacity) for row in cut["cut_edges"].itertuples(index=False)), Fraction()) == exact
print("Outgoing capacity:", cut["value"])
print("Source side:", cut["source_partition"])
print("Target side:", cut["target_partition"])
render = globals().get("display")
if render is not None:
    render(cut)
    render(cut["cut_edges"])
    render(oe.plot.network(graph, title="Directed global cut · integer 1 and string 1", layout="circular"))
else:
    print(cut.summary().to_string(index=False))
    print(cut["cut_edges"].to_string(index=False))
print("DIRECTED_GLOBAL_CUT_RECEIPT:" + json.dumps({
    "value": cut["value"], "exact_numerator": exact.numerator, "exact_denominator": exact.denominator,
    "source_partition": cut["source_partition"], "target_partition": cut["target_partition"],
    "flow_problems": cut["metadata"]["flow_problems"], "certified": cut["metadata"]["certified"],
    "work_used": cut["metadata"]["work_used"], "max_work": cut["metadata"]["max_work"],
    "sdk_version": oe.__version__,
}, allow_nan=False))
