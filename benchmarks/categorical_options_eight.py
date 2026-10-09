"""Eight residual categorical options: complete source/frozen/native payloads."""

from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages/openecon-charts/src")]
ISSUES = tuple(f"MARKET-{i}" for i in range(417, 425))


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
    if stage == 0:
        result = oe.catreg_nominal_response(
            data,
            "a",
            ["b", "x"],
            scales={"b": "nominal", "x": "numeric"},
            seed=83,
            n_starts=4,
            maxiter=200,
        )
        restored = save(result, "nominal-response", output_dir)
        save(
            oe.catreg_outcome_predict(restored, data, missing="drop"),
            "nominal-response-prediction",
            output_dir,
        )
    elif stage == 1:
        result = oe.catreg_ordinal_response(
            data,
            "b",
            ["a", "y"],
            scales={"a": "nominal", "y": "numeric"},
            outcome_order=[0, 1, 2, 3],
            seed=83,
            n_starts=4,
            maxiter=200,
        )
        restored = save(result, "ordinal-response", output_dir)
        save(
            oe.catreg_outcome_predict(restored, data, missing="drop"),
            "ordinal-response-prediction",
            output_dir,
        )
    elif stage in (2, 3):
        fit = oe.catpca(
            data,
            ["a", "b", "c", "x"],
            components=2,
            scales={"a": "nominal", "b": "ordinal", "c": "nominal", "x": "numeric"},
            orders={"b": [0, 1, 2, 3]},
            seed=83,
            n_starts=3,
        )
        method = oe.catpca_varimax if stage == 2 else oe.catpca_promax
        result = method(oe.restore_summary(oe.summary_state(fit)))
        stem = "varimax" if stage == 2 else "promax"
        restored = save(result, stem, output_dir)
        save(
            oe.catpca_rotated_predict(restored, data, missing="drop"),
            stem + "-projection",
            output_dir,
        )
    elif stage == 4:
        queries = data.dropna().iloc[:3][["a", "b", "x"]]
        result = oe.catreg_bootstrap(
            data,
            "y",
            ["a", "b", "x"],
            queries=queries,
            scales={"a": "nominal", "b": "nominal", "x": "numeric"},
            seed=83,
            reps=19,
            n_starts=2,
            maxiter=50,
        )
        save(result, "bootstrap", output_dir)
    elif stage in (5, 6):
        cells = pd.read_csv(Path(input_dir) / "counts.csv")
        levels = {"a": [0, 1], "b": [0, 1, 2], "c": [0, 1]}
        x = torch.tensor(
            [
                [float(a == 1), float(b == 1), float(b == 2), float(c == 1)]
                for a, b, c in cells[["a", "b", "c"]].itertuples(index=False, name=None)
            ],
            dtype=torch.float64,
        )
        if stage == 5:
            small = oe.loglinear_multinomial(
                cells, ["a", "b", "c"], "count", levels=levels, design=x.tolist()
            )
            result = oe.loglinear_multinomial(
                cells,
                ["a", "b", "c"],
                "count",
                levels=levels,
                design=torch.eye(len(cells), dtype=torch.float64)[:, 1:].tolist(),
            )
        else:
            small = oe.loglinear_product_multinomial(
                cells,
                ["a", "b", "c"],
                "count",
                levels=levels,
                conditioning=["a"],
                design=x[:, 1:].tolist(),
            )
            # Independent response log-odds for each fixed-a stratum.
            design = []
            for a, b, c in cells[["a", "b", "c"]].itertuples(index=False, name=None):
                category = 2 * b + c
                design.append(
                    [float(a == g and category == k) for g in [0, 1] for k in range(1, 6)]
                )
            result = oe.loglinear_product_multinomial(
                cells, ["a", "b", "c"], "count", levels=levels, conditioning=["a"], design=design
            )
        stem = "multinomial" if stage == 5 else "product-multinomial"
        restored = save(result, stem, output_dir)
        save(
            oe.loglinear_sampling_compare(oe.restore_summary(oe.summary_state(small)), restored),
            stem + "-lr",
            output_dir,
        )
    elif stage == 7:
        cells = pd.read_csv(Path(input_dir) / "counts.csv")
        result = oe.loglinear_select(
            cells,
            ["a", "b", "c"],
            "count",
            levels={"a": [0, 1], "b": [0, 1, 2], "c": [0, 1]},
            margins=[["a", "b", "c"]],
            criterion="bic",
            max_iter=100,
            max_work=300_000_000,
        )
        restored = save(result, "selection", output_dir)
        save(
            oe.restore_summary(restored.attrs["state"]["selected_summary"]),
            "selected-model",
            output_dir,
        )
    else:
        raise ValueError("Unknown stage")
    return result


def native_script(stage, input_dir, output_dir):
    common = "import sys\nfrom pathlib import Path\nimport openecon as oe\nassert getattr(sys,'frozen',False)\nassert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))\nassert oe.capabilities()['categorical_scaling_loglinear']['stata_parity_validated'] is False\n"
    common += inspect.getsource(save) + "\n" + inspect.getsource(run_stage) + "\n"
    common += f"result=run_stage({stage},{str(input_dir)!r},{str(output_dir)!r})\n"
    common += (
        'for name,frame in result.items():\n    if name!="settings":\n        display(frame)\n'
    )
    common += f"print('CATEGORICAL_OPTIONS_NATIVE_ACCEPTED {ISSUES[stage]}')\n"
    return common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "tests/fixtures/categorical")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import openecon as oe

    assert Path(oe.__file__).is_relative_to(ROOT / "src")
    for stage, issue in enumerate(ISSUES):
        result = run_stage(stage, args.input, args.output)
        print(issue, result.attrs.get("method"), result.attrs.get("nobs", result.attrs.get("n")))
    receipt = dict(
        source_sdk=str(Path(oe.__file__).resolve()),
        stages=list(ISSUES),
        passed=8,
        files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.output.iterdir())
            if p.is_file() and p.name != "receipt.json"
        },
    )
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
