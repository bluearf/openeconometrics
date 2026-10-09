"""Complete eight-stage source/frozen/native frequency-categorical acceptance."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages/openecon-charts/src")]
ISSUES = tuple(f"MARKET-{i}" for i in range(664, 672))


def save(result, stem, output_dir):
    import openecon as oe

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = oe.summary_state(result)
    restored = oe.restore_summary(state)
    assert oe.summary_state(restored) == state
    assert restored.to_latex() == result.to_latex(), stem + " complete LaTeX differs"
    for key, frame in result.items():
        assert restored[key].equals(frame), stem + " complete table differs: " + key
    (output / (stem + ".json")).write_text(state)
    (output / (stem + ".tex")).write_text(result.to_latex())
    return restored


def run_stage(stage, input_dir, output_dir):
    import openecon as oe
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    data = pd.read_csv(Path(input_dir) / "people.csv")
    options = dict(frequency="w", n_starts=2, maxiter=300, tol=1e-9, seed=0)
    if stage in (0, 1):
        method = oe.catreg_nominal_fweight if stage == 0 else oe.catreg_ordinal_fweight
        scales = {"a": "nominal", "b": "nominal" if stage == 0 else "ordinal", "z": "numeric"}
        extra = {} if stage == 0 else {"orders": {"b": ["low", "mid", "high"]}}
        result = method(data, "y", ["a", "b", "z"], scales=scales, **options, **extra)
        stem = "numeric-nominal" if stage == 0 else "numeric-ordinal"
        restored = save(result, stem, output_dir)
        save(oe.catreg_fweight_predict(restored, data, missing="drop"), stem+"-prediction", output_dir)
    elif stage in (2, 3):
        if stage == 2:
            result = oe.catreg_nominal_response_fweight(data, "a", ["b", "z", "q"],
                       scales={"b": "nominal", "z": "numeric", "q": "numeric"}, **options)
        else:
            result = oe.catreg_ordinal_response_fweight(data, "b", ["a", "z", "q"],
                       outcome_order=["low", "mid", "high"],
                       scales={"a": "nominal", "z": "numeric", "q": "numeric"}, **options)
        stem = "nominal-response" if stage == 2 else "ordinal-response"
        restored = save(result, stem, output_dir)
        save(oe.catreg_fweight_predict(restored, data, missing="drop"), stem+"-prediction", output_dir)
    elif stage in (4, 7):
        fit = oe.catpca_fweight(data, ["a", "b", "z", "q"], components=2,
                 scales={"a": "nominal", "b": "ordinal", "z": "numeric", "q": "numeric"},
                 orders={"b": ["low", "mid", "high"]}, **options)
        restored = save(fit, "catpca" if stage == 4 else "centroid-source", output_dir)
        if stage == 4:
            result = fit
            save(oe.catpca_fweight_predict(restored, data, missing="drop"), "catpca-projection", output_dir)
        else:
            result = oe.catpca_category_centroids(restored, transform=[[1., .35], [-.15, 1.2]])
            save(result, "category-centroids", output_dir)
            assert result.attrs["membership"].startswith("retained original")
    elif stage == 5:
        result = oe.mca_fweight(data, ["a", "b"], n_components=2, frequency="w")
        restored = save(result, "mca", output_dir)
        save(oe.mca_fweight_project(restored, data, missing="drop"), "mca-projection", output_dir)
    elif stage == 6:
        from openecon.econometrics.categorical.frequency_scaling import _checked

        result = oe.overals_fweight(data, [["a", "z"], ["b", "q"]], n_components=2,
                 frequency="w", scales={"a": "nominal", "b": "ordinal", "z": "numeric", "q": "numeric"},
                 orders={"b": ["low", "mid", "high"]}, n_starts=2, max_iter=400, tol=1e-8, seed=0)
        restored = save(result, "overals", output_dir)
        _checked(restored, "overals_fweight")
    else:
        raise ValueError("Unknown stage")
    return result


def native_script(stage, input_dir, output_dir):
    common = "import sys\nfrom pathlib import Path\nimport openecon as oe\nimport importlib.util\n"
    common += "assert getattr(sys, 'frozen', False)\nassert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))\n"
    common += "assert importlib.util.find_spec('scipy') is None\nassert importlib.util.find_spec('statsmodels') is None\n"
    common += "assert oe.capabilities()['categorical_frequency']['stata_parity_validated'] is False\n"
    common += inspect.getsource(save) + "\n" + inspect.getsource(run_stage) + "\n"
    common += f"result = run_stage({stage}, {str(input_dir)!r}, {str(output_dir)!r})\n"
    common += "for name, frame in result.items():\n    display(frame)\n"
    common += f"print('CATEGORICAL_FREQUENCY_NATIVE_ACCEPTED {ISSUES[stage]}')\n"
    return common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "tests/fixtures/categorical_frequency")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import openecon as oe

    assert Path(oe.__file__).is_relative_to(ROOT / "src")
    for stage, issue in enumerate(ISSUES):
        result = run_stage(stage, args.input, args.output)
        print(issue, result.attrs.get("method"), flush=True)
    receipt = dict(source_sdk=str(Path(oe.__file__).resolve()), stages=list(ISSUES), passed=8,
                   files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())
                          if p.is_file() and p.name != "receipt.json"})
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
