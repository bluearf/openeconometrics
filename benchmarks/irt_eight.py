"""Owned synthetic six-fit/eight-stage source/frozen/native acceptance payload."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages/openecon-charts/src")]

FAMILIES = ("rasch", "2pl", "3pl", "grm", "pcm", "rsm")
ISSUES = ("MARKET-280", "MARKET-281", "MARKET-282", "MARKET-283", "MARKET-284", "MARKET-285", "MARKET-286", "MARKET-287")


def table_payload(result):
    return {name: {"columns": list(frame.columns), "index": list(frame.index),
                   "data": frame.astype(object).where(frame.notna(), None).values.tolist(),
                   "attrs": frame.attrs} for name, frame in result.items()}


def fixtures(directory):
    """Fixed seeded draw from written item probabilities, not a fitted model."""
    import pandas as pd
    import torch
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(8412)
    n = 450
    theta = torch.randn(n, dtype=torch.float64, generator=generator, device="cpu")
    locations = torch.tensor([-.8, -.2, .3, .9], dtype=torch.float64, device="cpu")
    slopes = torch.tensor([1., 1.1, .9, 1.2], dtype=torch.float64, device="cpu")
    guessing = torch.tensor([.1, .12, .08, .05], dtype=torch.float64, device="cpu")
    probability = guessing+(1-guessing)*torch.sigmoid(slopes*(theta[:, None]-locations))
    binary = (torch.rand((n, 4), dtype=torch.float64, generator=generator, device="cpu") < probability).double()
    steps = torch.tensor([-.5, .5], dtype=torch.float64, device="cpu")
    logits = torch.cat((torch.zeros((n, 4, 1), dtype=torch.float64, device="cpu"),
                        (theta[:, None, None]-locations[None, :, None]-steps).cumsum(2)), 2)
    cumulative = torch.softmax(logits, 2).cumsum(2)
    ordinal = (torch.rand((n, 4, 1), dtype=torch.float64, generator=generator, device="cpu") > cumulative).sum(2).double()
    binary[0, 0], ordinal[5, 2] = float("nan"), float("nan")
    for name, values in (("binary", binary), ("ordinal", ordinal)):
        pd.DataFrame(values.tolist(), columns=["a", "b", "c", "d"],
                     index=[f"person-{i}" for i in range(n)]).to_csv(directory/(name+".csv"), index_label="person")


def run_family(family, input_dir, output_dir):
    import pandas as pd
    import openecon as oe
    data = pd.read_csv(Path(input_dir)/("binary.csv" if family in ("rasch", "2pl", "3pl") else "ordinal.csv"), index_col="person")
    kwargs = {"guessing": [.1, .12, .08, .05]} if family == "3pl" else {}
    result = getattr(oe, "irt_"+family)(data=data, items=["a", "b", "c", "d"], points=61, **kwargs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = result.to_json()
    (output/(family+".json")).write_text(state)
    (output/(family+".tex")).write_text(result.to_latex())
    restored = oe.irt_restore(state)
    assert restored.to_json() == state and restored.to_latex() == result.to_latex()
    for name, frame in result.items():
        assert restored[name].equals(frame)
    (output/(family+"-tables.json")).write_text(json.dumps(table_payload(result), sort_keys=True, allow_nan=False))
    return result


def run_postestimation(input_dir, output_dir):
    import pandas as pd
    import openecon as oe
    output = Path(output_dir)
    model = oe.irt_restore((output/"2pl.json").read_text())
    curves = oe.irt_information(model, theta=[-3., -1., 0., 1., 3.])
    scores = oe.irt_score(model, data=pd.DataFrame([[0]*4, [1]*4, [0, None, 1, None], [None]*4], columns=["a", "b", "c", "d"], index=["low", "high", "partial", "empty"]))
    assert list(scores["scores"].observed_items) == [4,4,2,0]
    assert scores["scores"].eap.iloc[-1] == 0 and scores["scores"].posterior_sd.iloc[-1] == 1
    for name, result in (("information", curves), ("scores", scores)):
        (output/(name+"-tables.json")).write_text(json.dumps(table_payload(result), sort_keys=True, allow_nan=False))
        (output/(name+".tex")).write_text(result.to_latex())
    return curves, scores


def native_script(stage, input_dir, output_dir):
    """Self-contained named native script; never imports host source at runtime."""
    import inspect
    common = "import sys, json\nfrom pathlib import Path\nimport openecon as oe\nassert getattr(sys, 'frozen', False)\nassert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))\nassert oe.capabilities()['irt']['stata_parity_validated'] is False\n"
    common += inspect.getsource(table_payload)+"\n"+inspect.getsource(run_family)+"\n"+inspect.getsource(run_postestimation)+"\n"
    if stage < 6:
        family = FAMILIES[stage]
        common += f"result = run_family({family!r}, {str(input_dir)!r}, {str(output_dir)!r})\ndisplay(result['parameters'])\ndisplay(result['covariance'])\ndisplay(result['diagnostics'])\n"
    else:
        common += f"curves, scores = run_postestimation({str(input_dir)!r}, {str(output_dir)!r})\n"
        common += "display(curves['item_information'])\ndisplay(curves['test_information'])\n" if stage == 6 else "display(scores['scores'])\n"
    common += f"print('IRT_NATIVE_ACCEPTED {ISSUES[stage]}')\n"
    return common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT/"tests/fixtures/irt")
    parser.add_argument("--prepare-fixtures", action="store_true")
    args = parser.parse_args()
    if args.prepare_fixtures:
        fixtures(args.input)
    import openecon as oe
    assert Path(oe.__file__).is_relative_to(ROOT/"src"), "Source acceptance must use this owned source"
    for family in FAMILIES:
        result = run_family(family, args.input, args.output)
        print(family, result.attrs["nobs"], result["diagnostics"].max_gradient[0])
    run_postestimation(args.input, args.output)
    receipt = {"source_sdk": str(Path(oe.__file__).resolve()), "stages": list(ISSUES), "passed": 8,
               "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob("*")) if p.is_file()}}
    (args.output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
