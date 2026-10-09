"""Eight substantive categorical methods; complete source/frozen/native payloads."""

from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages/openecon-charts/src")]
ISSUES = tuple(f"MARKET-{i}" for i in range(296, 304))


def fixtures(directory):
    import pandas as pd
    import torch
    from itertools import product

    directory.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device="cpu").manual_seed(76312)
    n = 120
    a = torch.randint(0, 3, (n,), generator=g)
    b = torch.randint(0, 4, (n,), generator=g)
    c = torch.randint(0, 2, (n,), generator=g)
    d = (a + torch.randint(0, 2, (n,), generator=g)) % 3
    x = torch.randn(n, dtype=torch.float64, generator=g)
    noise = 0.3 * torch.randn(n, dtype=torch.float64, generator=g)
    y = 3 + 2 * (a == 0) - 1 * (a == 2) + 0.7 * b + 0.4 * x + noise
    frame = pd.DataFrame(
        {
            "a": [["red", "green", "blue"][int(v)] for v in a],
            "b": b.tolist(),
            "c": [["left", "right"][int(v)] for v in c],
            "d": d.tolist(),
            "x": x.tolist(),
            "y": y.tolist(),
        },
        index=[f"subject-{i}" for i in range(n)],
    )
    frame.loc["subject-5", "a"] = None
    frame.to_csv(directory / "people.csv", index_label="subject")
    cells = pd.DataFrame(list(product(range(2), range(3), range(2))), columns=["a", "b", "c"])
    cells["count"] = [21, 13, 35, 20, 18, 31, 27, 30, 17, 16, 25, 19]
    cells.to_csv(directory / "counts.csv", index=False)
    coords = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.4], [0.3, 2.0]],
        dtype=torch.float64,
    )
    distances = torch.cdist(coords, coords).square()
    pd.DataFrame(
        distances.tolist(),
        index=[f"object-{i}" for i in range(6)],
        columns=[f"object-{i}" for i in range(6)],
    ).to_csv(directory / "dissimilarities.csv", index_label="object")


def save(result, stem, output_dir):
    import openecon as oe

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = oe.summary_state(result)
    restored = oe.restore_summary(state)
    assert oe.summary_state(restored) == state
    assert restored.to_latex() == result.to_latex(), stem + " complete LaTeX restore differs"
    for key, frame in result.items():
        assert restored[key].equals(frame), stem + " table restore differs: " + key
    (output / (stem + ".json")).write_text(state)
    (output / (stem + ".tex")).write_text(result.to_latex())
    return restored


def run_stage(stage, input_dir, output_dir):
    import openecon as oe
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    data = pd.read_csv(Path(input_dir) / "people.csv", index_col="subject")
    result = None
    if stage == 0:
        result = oe.catreg_nominal(data, "y", ["a", "b", "c"], seed=83, n_starts=3)
        restored = save(result, "nominal", output_dir)
        pred = oe.catreg_predict(restored, data, missing="drop")
        save(pred, "nominal-prediction", output_dir)
    elif stage == 1:
        result = oe.catreg_ordinal(
            data,
            "y",
            ["b", "x"],
            scales={"b": "ordinal", "x": "numeric"},
            orders={"b": [0, 1, 2, 3]},
            seed=83,
            n_starts=3,
        )
        restored = save(result, "ordinal", output_dir)
        save(oe.catreg_predict(restored, data, missing="drop"), "ordinal-prediction", output_dir)
    elif stage == 2:
        result = oe.catpca(
            data,
            ["a", "b", "c", "x"],
            components=2,
            scales={"a": "nominal", "b": "ordinal", "c": "nominal", "x": "numeric"},
            orders={"b": [0, 1, 2, 3]},
            seed=83,
            n_starts=3,
        )
        restored = save(result, "catpca", output_dir)
        save(oe.catpca_predict(restored, data, missing="drop"), "catpca-projection", output_dir)
    elif stage == 3:
        result = oe.mca(data, ["a", "b", "c", "d"], n_components=2)
        restored = save(result, "mca", output_dir)
        save(oe.mca_project(restored, data, missing="drop"), "mca-projection", output_dir)
    elif stage == 4:
        result = oe.overals(
            data,
            [["a", "b"], ["c", "d"]],
            n_components=2,
            scales={v: "multiple_nominal" for v in ["a", "b", "c", "d"]},
            n_starts=3,
            seed=83,
        )
        save(result, "overals", output_dir)
    elif stage == 5:
        matrix = pd.read_csv(Path(input_dir) / "dissimilarities.csv", index_col="object")
        result = oe.mds_nonmetric(
            matrix,
            n_components=2,
            n_starts=3,
            seed=83,
            max_iter=1000,
            tol=1e-6,
            ties="secondary",
            zero="include",
        )
        save(result, "nmds", output_dir)
    elif stage == 6:
        cells = pd.read_csv(Path(input_dir) / "counts.csv")
        result = oe.loglinear_ipf(
            cells,
            ["a", "b", "c"],
            "count",
            levels={"a": [0, 1], "b": [0, 1, 2], "c": [0, 1]},
            margins=[["a", "b"], ["b", "c"]],
        )
        save(result, "ipf", output_dir)
    elif stage == 7:
        cells = pd.read_csv(Path(input_dir) / "counts.csv")
        levels = {"a": [0, 1], "b": [0, 1, 2], "c": [0, 1]}
        small = oe.loglinear_ipf(
            cells, ["a", "b", "c"], "count", levels=levels, margins=[["a", "b"], ["b", "c"]]
        )
        result = oe.loglinear_ml(
            cells,
            ["a", "b", "c"],
            "count",
            levels=levels,
            design=torch.eye(len(cells), dtype=torch.float64).tolist(),
            terms=[f"cell-{i}" for i in range(len(cells))],
        )
        restored = save(result, "ml", output_dir)
        save(
            oe.loglinear_compare(oe.restore_summary(oe.summary_state(small)), restored),
            "nested-comparison",
            output_dir,
        )
    else:
        raise ValueError("Unknown stage")
    return result


def native_script(stage, input_dir, output_dir):
    common = "import sys\nfrom pathlib import Path\nimport openecon as oe\nassert getattr(sys, 'frozen', False)\nassert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))\nassert oe.capabilities()['categorical_scaling_loglinear']['stata_parity_validated'] is False\n"
    common += inspect.getsource(save) + "\n" + inspect.getsource(run_stage) + "\n"
    common += f"result=run_stage({stage}, {str(input_dir)!r}, {str(output_dir)!r})\n"
    # Public table names are determined from the actual source result, not assumed.
    common += (
        'for name,frame in result.items():\n    if name!="settings":\n        display(frame)\n'
    )
    common += f"print('CATEGORICAL_NATIVE_ACCEPTED {ISSUES[stage]}')\n"
    return common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "tests/fixtures/categorical")
    parser.add_argument("--prepare-fixtures", action="store_true")
    args = parser.parse_args()
    if args.prepare_fixtures:
        fixtures(args.input)
    import openecon as oe

    assert Path(oe.__file__).is_relative_to(ROOT / "src")
    for stage, issue in enumerate(ISSUES):
        result = run_stage(stage, args.input, args.output)
        print(issue, result.attrs.get("method"), result.attrs.get("nobs"))
    receipt = {
        "source_sdk": str(Path(oe.__file__).resolve()),
        "stages": list(ISSUES),
        "passed": 8,
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.output.glob("*"))
            if p.is_file() and p.name != "receipt.json"
        },
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
