"""Published open data, for instructor preparation only.

Applies the Table IV sample rules of the depositors' ``AngristLavy_Table4.do`` to
the vendored fifth-grade file in ``instructors/sources/angrist-lavy-1999``. Each
file's SHA-256 is checked against ``source.json`` before it is read; nothing is
downloaded.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = "angrist-lavy-1999"


def source_file(name):
    here = Path(__file__).resolve().parent
    for folder in (here.parent / "sources" / DATASET, here / "sources" / DATASET):
        if (folder / "source.json").is_file():
            break
    else:
        raise FileNotFoundError(f"Vendored source {DATASET} is missing")
    recorded = {f["name"]: f["sha256"] for f in json.loads((folder / "source.json").read_text())["files"]}
    if hashlib.sha256((folder / name).read_bytes()).hexdigest() != recorded[name]:
        raise ValueError(f"{name} differs from its recorded SHA-256")
    return folder / name


def decimal(values):
    """Shortest decimal that reproduces each stored float32 value (74.11, not 74.1100006)."""
    return [
        float(np.format_float_positional(np.float32(v), unique=True)) if pd.notna(v) else np.nan
        for v in values
    ]


def make_data():
    d = pd.read_stata(source_file("final5.dta"), convert_categoricals=False)
    # The do-file's cleaning and sample rules, in its order. Stata treats a missing
    # value as larger than every number, so "c_size > 5" also keeps a missing c_size.
    for score in ("avgverb", "avgmath"):
        d.loc[d[score] > 100, score] -= 100
    d["func1"] = d.c_size / ((d.c_size - 1) // 40 + 1)
    d.loc[d.verbsize == 0, "avgverb"] = np.nan
    d.loc[d.mathsize == 0, "avgmath"] = np.nan
    d = d[(d.classize > 1) & (d.classize < 45) & ((d.c_size > 5) | d.c_size.isna())]
    d = d[(d.c_leom == 1) & (d.c_pik < 3)]
    d = d[d.avgverb.notna()]
    return pd.DataFrame(
        {
            "schlcode": d.schlcode.astype(int).to_numpy(),
            "classid": d.classid.astype(int).to_numpy(),
            "c_size": d.c_size.astype(int).to_numpy(),
            "tipuach": d.tipuach.astype(int).to_numpy(),
            "classize": d.classize.astype(int).to_numpy(),
            "func1": d.func1.astype(float).to_numpy(),
            "avgverb": decimal(d.avgverb),
            "avgmath": decimal(d.avgmath),
        }
    )
