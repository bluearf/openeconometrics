"""Save, reread and export the real publication matrix with the UI document builder.

Run: PYTHONPATH=src python scripts/verify_publication.py --output output/publication-market-110
Use --saved-results to verify/export existing JSON with no fitting or source data.
PDF compilation and visual inspection are separate, explicit evidence steps.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import openecon as oe
from openecon.econometrics.registry import all_estimators, get
from openecon.models import ResultBundle
from openecon.output_latex import add_output_latex, enrich_record

from publication_matrix import fitted_models


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saved-results", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    models = ([(p.stem, ResultBundle.model_validate_json(p.read_text()))
               for p in sorted(args.saved_results.glob("*.json"))] if args.saved_results
              else list(fitted_models()))
    outputs, receipts, saved = [], [], {}
    for name, result in models:
        path = out / f"{name}.json"
        path.write_text(result.model_dump_json())
        before = path.read_bytes()
        model = ResultBundle.model_validate_json(path.read_text())
        item = add_output_latex({"type": "model", "data": model.model_dump(mode="json")})
        assert item["latex"] == model.to_latex()
        assert item["latex_math"] == model.to_latex().math
        original = {"outputs": [{"type": "model", "data": item["data"],
                                 "latex": "old", "latex_style": "publication-v1"}]}
        snapshot = json.dumps(original, sort_keys=True)
        assert enrich_record(original)["outputs"][0] == item
        assert json.dumps(original, sort_keys=True) == snapshot
        assert path.read_bytes() == before
        (out / f"{name}.tex").write_text(item["latex"])
        outputs.append(item)
        saved[name] = model
        receipts.append({"case": name, "family": get(model.spec.estimator).family,
            "estimator": model.spec.estimator, "nobs": model.nobs, "nobs_original": model.nobs_original,
            "dropped_rows": model.dropped_rows, "coefficients": [c.model_dump() for c in model.coefficients],
            "metrics": model.metrics, "inference": model.inference, "warnings": model.warnings,
            "notes": item["latex_notes"], "equations": list(dict.fromkeys(c.equation for c in model.coefficients)),
            "full_result_sha256": digest(path), "tex_sha256": digest(out / f"{name}.tex"),
            "readback_without_refit": True, "stored_json_unchanged": True})
    families = {get(m.spec.estimator).family for m in saved.values()}
    assert families == {i.family for i in all_estimators()}
    (out / "family-outputs.json").write_text(json.dumps(outputs, ensure_ascii=False))
    fixtures = []
    for (name, _), item in zip(models, outputs, strict=True):
        display = {**item, "data": {key: value for key, value in item["data"].items()
                   if key not in {"sample_positions", "covariance_matrix", "predictions"}}}
        display["data"].update(predictions=[], display_omitted=["sample_positions", "covariance_matrix"])
        fixtures.append({"case": name, "output": display})
    (out / "browser-fixtures.json").write_text(json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n")
    # Layout-only adversaries are labelled separately from the fitted matrix.
    long_model = saved["sureg"].model_copy(deep=True)
    template = long_model.coefficients[0]
    long_model.coefficients = [template.model_copy(update={"term": f"term_{i:03}",
                               "equation": f"Equation {i // 18 + 1}"}) for i in range(90)]
    long_model.metrics = {}
    long_model.warnings = ["Layout-only fixture: repeated saved coefficient values; not an estimated 90-parameter model."]
    long_source = long_model.to_latex(caption="Long grouped layout fixture")
    wide = oe.regression_table({f"Model {i + 1}": saved["ols"] for i in range(6)}, longtable=True,
        caption="Wide comparison: six saved models", term_labels={"x": "Ücret & eğitim_%"})
    frame = oe.DataFrame({"Ücret_%": [1.2e-12, 0.0, -.00001234],
        "Special characters": [r"A&B # $ % _ { } \ ~ ^", "Türkçe: ı İ ş Ş ğ Ğ ü Ü ö Ö ç Ç", "line one\nline two"]})
    unicode_source = frame.to_latex(index=False, caption="Unicode and escaped special characters")
    # A genuinely complete data export is distinct from the bounded console view.
    data = oe.DataFrame({"row": range(90), "value": ["row_% & complete"] * 90})
    complete_data = data.to_latex(index=False, caption="Complete 90-row data export")
    def explicit(source):
        item = {"type": "latex", "data": str(source), "latex": str(source), "latex_math": source.math}
        if getattr(source, "notes", None):
            item["latex_notes"] = source.notes
        return item
    stress = [explicit(s) for s in (wide, long_source, unicode_source, complete_data)]
    (out / "stress-outputs.json").write_text(json.dumps(stress, ensure_ascii=False))
    root = Path(__file__).resolve().parents[1]
    # This is the exact function used by Copy LaTeX and .tex download in App.tsx.
    builder = root / "web" / "src" / "latex-export.ts"
    for name in ("family", "stress"):
        js = f"import {{readFileSync,writeFileSync}} from 'node:fs'; import {{latexDocument}} from {json.dumps(builder.as_uri())}; writeFileSync(process.argv[2],latexDocument(JSON.parse(readFileSync(process.argv[1],'utf8'))));"
        subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", js,
                        str(out / f"{name}-outputs.json"), str(out / f"{name}.tex")], check=True)
    receipt = {"issue": "MARKET-110", "registry_estimators": len(all_estimators()),
        "families": sorted(families), "fitted_cases": len(models), "cases": receipts,
        "layout_fixtures": ["wide-six-model", "long-five-equation-90-parameter", "unicode-special", "full-90-row-data"],
        "ui_document_builder": "web/src/latex-export.ts:latexDocument",
        "documents": {name: {"sha256": digest(out / f"{name}.tex")} for name in ("family", "stress")},
        "scope": "Publication serialization/formatting; representative family fits, not all 80 specifications or estimator parity."}
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"fitted_cases": len(models), "families": len(families), "output": str(out)}))


if __name__ == "__main__":
    main()
