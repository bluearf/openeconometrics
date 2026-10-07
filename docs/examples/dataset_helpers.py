"""MARKET-100 native panel acceptance on an owned synthetic Parquet source."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd
import torch
import openecon as oe
from openecon.resources import use_workspace_budget

display = globals().get("display", print)


def midrank(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            ranks[index] = (start + 1 + end) / 2
        start = end
    return ranks


generator = torch.Generator().manual_seed(100807)
row = torch.arange(1200)
x = (row * 17 % 29).to(torch.float64)
z = (row * 7 % 31).to(torch.float64)
frame = pd.DataFrame({"x": x.tolist(), "z": z.tolist(),
    "y": (.2*x + .4*(row % 3) + .5*((row//3) % 2)
          + torch.randn(len(row), generator=generator, dtype=torch.float64)).tolist(),
    "subject": (row//3).tolist(), "cell": (row % 3).tolist(), "g": ((row//3) % 2).tolist()})
previous_scratch = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
try:
    with tempfile.TemporaryDirectory(prefix="openecon-helper-panel-") as temporary:
        owned = Path(temporary)
        scratch = owned / "scratch"
        scratch.mkdir()
        os.environ["OPENECON_SCRATCH_DIRECTORY"] = str(scratch)
        path = owned / "study.parquet"
        frame.to_parquet(path, row_group_size=23, index=False)
        source = oe.scan(path)
        with use_workspace_budget(64):
            spear = oe.correlate(source, ["x", "z"], method="spearman")
            kendall = oe.correlate(source, ["x", "z"], method="kendall")
            described = oe.describe(source, ["x", "y"], by="g",
                stats=["n", "mean", "p2.5", "p50", "p97.5", "sd"])
            repeated = oe.rm_anova(source, "y", "subject", ["cell"], between=["g"])
            multivariate = oe.manova(source, ["y", "z"], ["g"], covariates=["x"])
        # Independent rank identities, with ties, on the small synthetic fixture.
        a, b = midrank(x.tolist()), midrank(z.tolist())
        mean_a, mean_b = sum(a)/len(a), sum(b)/len(b)
        cov = sum((u-mean_a)*(v-mean_b) for u, v in zip(a, b, strict=True))
        oracle = cov / math.sqrt(sum((u-mean_a)**2 for u in a)*sum((v-mean_b)**2 for v in b))
        assert abs(spear["coefficients"].loc["x", "z"] - oracle) < 1e-12
        xv, zv = x.tolist(), z.tolist()
        score = ties_x = ties_z = 0
        for i in range(len(xv)):
            for j in range(i):
                dx, dz = xv[i]-xv[j], zv[i]-zv[j]
                ties_x += dx == 0
                ties_z += dz == 0
                score += (dx*dz > 0) - (dx*dz < 0)
        pairs = len(xv)*(len(xv)-1)/2
        assert abs(kendall["coefficients"].loc["x", "z"]
                   - score/math.sqrt((pairs-ties_x)*(pairs-ties_z))) < 1e-12
        assert described.n.sum() == 2400
        assert repeated.attrs["n_subjects"] == 400
        assert repeated.attrs["subject_matrix_collected"] is False
        assert len(multivariate["box_m"]) == 1
        assert all(result.attrs["full_source_collected"] is False
                   for result in [spear, kendall, described, repeated, multivariate])
        assert not list(scratch.iterdir())
        for table in [spear["coefficients"], kendall["coefficients"], described,
                      repeated["within"], repeated["sphericity"],
                      multivariate["multivariate"], multivariate["box_m"]]:
            display(table)
        passes = {name: result.attrs["source_passes"] for name, result in
                  [("spearman", spear), ("kendall", kendall), ("describe", described),
                   ("rm_anova", repeated), ("manova", multivariate)]}
finally:
    if previous_scratch is None:
        os.environ.pop("OPENECON_SCRATCH_DIRECTORY", None)
    else:
        os.environ["OPENECON_SCRATCH_DIRECTORY"] = previous_scratch

print("DATASET_HELPERS:" + json.dumps({"physical_parquet_rows": 1200,
    "exact_rank_oracles": True, "full_source_collected": False,
    "subjects": 400, "source_passes": passes, "scratch_removed": not owned.exists(),
    "frozen": bool(getattr(sys, "frozen", False)),
    "sdk_from_bundle": bool(getattr(sys, "frozen", False))
        and Path(oe.__file__).is_relative_to(Path(sys._MEIPASS)),
    "scipy_loaded": any(name == "scipy" or name.startswith("scipy.") for name in sys.modules),
    "device": "cpu", "dtype": "float64"}, sort_keys=True))
