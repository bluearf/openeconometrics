"""Synthetic single-stage declaration and JSON readback; no survey estimation."""

import json
from pathlib import Path

import openecon as oe


def main():
    frame = oe.DataFrame({
        "w": [2., 3., 4., 2., 2., 3.],
        "psu": [1, 1, 2, 1, 2, 2],
        "stratum": ["a", "a", "a", "b", "b", "b"],
        "population_psu": [8, 8, 8, 4, 4, 4],
    })
    design = oe.survey_design(frame, weights="w", psu="psu", strata="stratum",
                              fpc="population_psu")
    # Deliberately use a new synthetic output directory. Existing files are not overwritten.
    folder = Path("survey-design-example")
    folder.mkdir(exist_ok=False)
    path = folder / "design.json"
    path.write_text(design.model_dump_json(indent=2), encoding="utf-8")
    restored = oe.SurveyDesign.model_validate_json(path.read_text(encoding="utf-8"))
    restored.revalidate(frame)
    print(json.dumps({"declaration": str(path), "geometry": restored.validation.model_dump(),
                      "estimation": False, "inference": False}, indent=2))


if __name__ == "__main__":
    main()
