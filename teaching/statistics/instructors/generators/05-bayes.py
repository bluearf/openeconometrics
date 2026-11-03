"""Published open data, for instructor preparation only.

Expands the eight published Table 2 counts of Pray et al. (2021), vendored in
``instructors/sources/mmwr-antigen-2021``, into one row per paired nasal swab.
The file's SHA-256 is checked against ``source.json`` before it is read;
nothing is downloaded.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

DATASET = "mmwr-antigen-2021"


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
    counts = pd.read_csv(source_file("table2_counts.csv"))
    rows = []
    for cell in counts.itertuples(index=False):
        rows += [
            (
                int(cell.symptom_status == "symptomatic"),
                int(cell.antigen_result == "positive"),
                int(cell.rt_pcr_result == "positive"),
            )
        ] * int(cell.paired_swabs)
    data = pd.DataFrame(rows, columns=["symptomatic", "antigen_positive", "pcr_positive"])
    data.insert(0, "specimen_pair_id", range(1, len(data) + 1))
    return data
