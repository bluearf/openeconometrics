"""Synthetic source/frozen runtime proof for the fourteen new dense model routes.

Run as a script or copy into the isolated desktop QA editor. Refuses development
estimator imports. The receipt contains aggregate estimates/covariance/settings
and synthetic sample positions; no human datasets or credentials are read.
"""

from __future__ import annotations
import builtins
import json
import sys
from pathlib import Path

_original_import = builtins.__import__
_FORBIDDEN = {"scipy", "statsmodels", "arch", "linearmodels", "sklearn"}


def _native_only(name, *args, **kwargs):
    if name.split(".", 1)[0] in _FORBIDDEN:
        raise ImportError("Development estimation engines are denied in the runtime proof.")
    return _original_import(name, *args, **kwargs)


builtins.__import__ = _native_only
import torch  # noqa: E402 - the import denial must be installed first
import pandas as pd  # noqa: E402
import openecon as oe  # noqa: E402
from openecon.models import ResultBundle  # noqa: E402

torch.set_num_threads(1)
generator = torch.Generator().manual_seed(40219)


def normal(*shape):
    return torch.randn(shape, generator=generator, dtype=torch.float64)


pieces = []
for unit in range(5):
    x = normal(110).cumsum(0)
    error = normal(110)
    for t in range(1, 110):
        error[t] += 0.4 * error[t - 1]
    pieces.append(
        pd.DataFrame(
            {"unit": unit, "t": range(110), "x": x.numpy(), "y": (2 + 1.2 * x + error).numpy()}
        )
    )
panel = pd.concat(pieces, ignore_index=True)
series = pieces[0].copy()
z = normal(180)
shock = z + 0.6 * normal(180)
y = normal(180)
for t in range(1, 180):
    y[t] += 0.4 * y[t - 1] + 0.5 * shock[t]
projection = pd.DataFrame({"t": range(180), "y": y.numpy(), "shock": shock.numpy(), "z": z.numpy()})
projection_panel = pd.concat(
    [projection.assign(unit=i, y=lambda d: d.y + i * 0.4 + normal(180).numpy()) for i in range(5)],
    ignore_index=True,
)
a = normal(180)
m = 0.5 * a + normal(180)
outcome = 0.2 * a + 0.7 * m + normal(180)
decomposition = pd.DataFrame(
    {"a": a.numpy(), "m": m.numpy(), "y": outcome.numpy(), "group": ["A"] * 90 + ["B"] * 90}
)
models = []
for name in ("fmols", "dols", "ccr"):
    models.append(getattr(oe, name)(data=series, y="y", x=["x"], time="t"))
for name in ("panel_fmols", "panel_dols", "pmg", "mg", "dfe"):
    models.append(getattr(oe, name)(data=panel, y="y", x=["x"], panel="unit", time="t"))
structural = oe.svar(
    data=projection, y=["y", "shock"], time="t", short_run=[[None, 0], [None, None]]
)
models.append(structural)
models.append(oe.lp(data=projection, y="y", x=["shock"], time="t", horizons=3))
models.append(oe.lpiv(data=projection, y="y", x=["shock"], instruments=["z"], time="t", horizons=3))
models.append(
    oe.panel_lp(data=projection_panel, y="y", x=["shock"], panel="unit", time="t", horizons=3)
)
models.append(oe.mediation(data=decomposition, y="y", x=["a"], mediators=["m"]))
models.append(
    oe.oaxaca(
        data=decomposition, y="y", x=["a", "m"], group="group", groups=["A", "B"], reps=50, seed=22
    )
)
receipt = {
    "synthetic_seed": 40219,
    "engine": "openecon.torch",
    "precision": "float64",
    "models": {},
    "helpers": {},
}
for result in models:
    saved = ResultBundle.model_validate_json(result.model_dump_json())
    assert saved.covariance_matrix == result.covariance_matrix
    assert saved.sample_positions == result.sample_positions
    assert "\\toprule" in saved.to_latex()
    assert saved.provenance["engine"] == "openecon" and saved.provenance["precision"] == "float64"
    receipt["models"][saved.spec.estimator] = {
        "nobs": saved.nobs,
        "sample_positions": saved.sample_positions,
        "terms": [c.term for c in saved.coefficients],
        "estimates": [c.estimate for c in saved.coefficients],
        "covariance": saved.covariance_matrix,
        "inference": saved.inference,
        "json_latex_roundtrip": True,
    }
    if callable(globals().get("display")):
        globals()["display"](saved)
receipt["helpers"] = {
    "hausman": oe.panel_ardl_hausman(models[6], models[5]),
    "irf_rows": len(oe.svar_irf(structural, steps=2, draws=20, seed=5)),
    "dy_total": oe.connectedness(structural).attrs["total_percent"],
    "oaxaca_rows": len(oe.oaxaca_details(models[-1])),
}
receipt["frozen_execution"] = bool(getattr(sys, "frozen", False))
receipt["sdk_from_bundle"] = bool(
    receipt["frozen_execution"]
    and __import__("pathlib")
    .Path(oe.__file__)
    .is_relative_to(Path(sys._MEIPASS))
)
receipt["forbidden_engines_loaded"] = sorted(
    _FORBIDDEN & {name.split(".", 1)[0] for name in sys.modules}
)
assert receipt["forbidden_engines_loaded"] == []
print(json.dumps(receipt, allow_nan=False, sort_keys=True))
builtins.__import__ = _original_import
