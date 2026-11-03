"""Published open data, for instructor preparation only.

Stacks the red and white files of the vendored UCI Wine Quality extract in
``instructors/sources/uci-wine-quality``. Each file's SHA-256 is checked
against ``source.json`` before it is read; nothing is downloaded.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

DATASET = "uci-wine-quality"


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
    parts = [
        pd.read_csv(source_file(f"winequality-{colour}.csv"), sep=";").assign(colour=colour)
        for colour in ("red", "white")
    ]
    wines = pd.concat(parts, ignore_index=True)
    return pd.DataFrame(
        {"wine": range(1, len(wines) + 1), "colour": wines.colour, "quality": wines.quality}
    )
