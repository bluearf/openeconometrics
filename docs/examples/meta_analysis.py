"""Run in the OpenEconometrics code panel; published independent BCG studies.

Counts from metadat/dat.bcg (GPL >=2); factual study records, not R code.
https://wviechtb.github.io/metadat/reference/dat.bcg.html
"""
import hashlib
import json
from pathlib import Path

import openecon as oe

data = oe.DataFrame({
    "trial": list(range(1, 14)),
    "tpos": [4, 6, 3, 62, 33, 180, 8, 505, 29, 17, 186, 5, 27],
    "tneg": [119, 300, 228, 13536, 5036, 1361, 2537, 87886, 7470, 1699, 50448, 2493, 16886],
    "cpos": [11, 29, 11, 248, 47, 372, 10, 499, 45, 65, 141, 3, 29],
    "cneg": [128, 274, 209, 12619, 5761, 1079, 619, 87892, 7232, 1600, 27197, 2338, 17825],
    "ablat": [44, 55, 42, 52, 13, 44, 19, 13, 27, 42, 18, 33, 33],
    "year": [1948, 1949, 1960, 1977, 1973, 1953, 1973, 1980, 1968, 1961, 1974, 1969, 1976],
    "alloc": ["random", "random", "random", "random", "alternate", "alternate", "random", "random", "random", "systematic", "systematic", "systematic", "systematic"],
})
effects = oe.meta_effectsize(data=data, measure="RR", columns=dict(a="tpos", b="tneg", c="cpos", d="cneg"), study="trial")
for column in ("ablat", "year", "alloc"):
    effects[column] = data[column]
effects["sequence"] = data.trial
pool = oe.meta_pool(data=effects, study="study", method="REML")
regression = oe.meta_regress(data=effects, study="study", moderators=["ablat", "year"], method="REML", inference="hksj")
state = json.loads(json.dumps(regression.attrs["prediction_state"], allow_nan=False))
prediction = oe.meta_predict(state, data=dict(ablat=[30., 50.], year=[1970., 2000.]))
diagnostics = oe.meta_diagnostics(data=effects, study="study", group="alloc", order="sequence")
plots = {kind: oe.meta_plot(diagnostics, kind=kind) for kind in ("forest", "funnel")}
assert abs(pool["coefficients"].iloc[0].estimate+.714532343) < 1e-7
assert diagnostics.attrs["counts"]["leave_one_out"] == dict(planned=13, completed=13, failed=0)
assert diagnostics.attrs["counts"]["cumulative"] == dict(planned=13, completed=12, failed=1)
assert len(regression["covariance"]) == 3 and len(prediction) == 2


def saved_frame(frame):
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="split")


full_results = {"effects": {"attrs": effects.attrs, "tables": {"effects": saved_frame(effects)}, "latex": str(effects.to_latex())},
                "prediction": {"attrs": prediction.attrs, "tables": {"prediction": saved_frame(prediction)}, "latex": str(prediction.to_latex())}}
for name, result in (("pool", pool), ("regression", regression), ("diagnostics", diagnostics)):
    full_results[name] = {"attrs": result.attrs, "tables": {key: saved_frame(frame) for key, frame in result.items()}, "latex": result.to_latex()}
full_results["plots"] = {"plots": {kind: plot.model_dump() for kind, plot in plots.items()},
                         "latex": {kind: str(plot.to_latex()) for kind, plot in plots.items()}}
encoded = {name: json.dumps(payload, sort_keys=True, allow_nan=False) for name, payload in full_results.items()}
if "META_RESULT_DIRECTORY" in globals():
    root = Path(globals()["META_RESULT_DIRECTORY"])
    root.mkdir(parents=True, exist_ok=True)
    for name, content in encoded.items():
        (root/(name+".json")).write_text(content)
if "display" in globals():
    globals()["display"](effects)
    for result in (pool, regression, diagnostics):
        for name, frame in result.items():
            # The console permits 20 displays; the duplicated pooled summary
            # remains in complete JSON/LaTeX rather than consuming a display.
            if result is diagnostics and name == "summary":
                continue
            globals()["display"](frame)
    globals()["display"](prediction)
    for plot in plots.values():
        globals()["display"](plot)
print("META_ACCEPTANCE_OK "+json.dumps({"methods": ["meta_effectsize", "meta_pool", "meta_regress", "meta_predict", "meta_diagnostics", "meta_plot"],
    "study_count": 13, "table_outputs": 18, "saved_tables": 19, "plot_outputs": 2, "complete_covariance": [3, 3],
    "pooled_effect": float(pool["coefficients"].iloc[0].estimate), "tau2": float(pool["heterogeneity"].iloc[0].tau2),
    "counts": diagnostics.attrs["counts"], "publication_latex": True, "metadata_json_roundtrip": True,
    "full_results_sha256": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in encoded.items()}}))
