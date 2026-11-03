"""Published open data, for instructor preparation only.

Rebuilds the workbook rows from the vendored UCI Online Shoppers extract in
``instructors/sources/uci-online-shoppers``. Each file's SHA-256 is checked
against ``source.json`` before it is read; nothing is downloaded.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

DATASET = "uci-online-shoppers"


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


def make_data():
    raw = pd.read_csv(source_file("online_shoppers_intention.csv"))
    return pd.DataFrame(
        {
            "session": range(1, len(raw) + 1),
            "VisitorType": raw.VisitorType,
            "Month": raw.Month,
            "Revenue": raw.Revenue.astype(int),
        }
    )
