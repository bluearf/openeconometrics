"""Verify a installed wheel's survey declaration without checkout imports."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PROGRAM = r'''
import hashlib, json, sys, tempfile
from pathlib import Path
import openecon as oe
assert oe.SurveyDesign and oe.survey_design
assert 'torch' not in sys.modules and 'openecon.analysis' not in sys.modules
frame = oe.DataFrame({'w':[2.,3.,4.,2.,2.,3.], 'p':[1,1,2,1,2,2],
                      'h':['a','a','a','b','b','b'], 'N':[8,8,8,4,4,4]})
design = oe.survey_design(frame, weights='w', psu='p', strata='h', fpc='N')
assert (design.validation.nobs, design.validation.n_strata,
        design.validation.n_psu, design.validation.design_df) == (6,2,4,2)
assert design.validation.sum_weights == 16.
with tempfile.TemporaryDirectory() as root:
    path = Path(root)/'design.json'
    data = Path(root)/'survey.csv'
    frame.to_csv(data,index=False)
    path.write_text(design.model_dump_json())
    restored = oe.SurveyDesign.model_validate_json(path.read_text())
    loaded = oe.read(data)
    try:
        restored.revalidate(loaded)
    except oe.AnalysisError as error:
        assert error.code == 'survey_design_changed'
    else:
        raise AssertionError('CSV parser dtype drift was silently accepted')
    loaded = loaded.astype(frame.dtypes.to_dict())
    restored.revalidate(loaded)
    changed = frame.copy()
    changed.loc[0,'w'] = 3.
    try:
        restored.revalidate(changed)
    except oe.AnalysisError as error:
        assert error.code == 'survey_design_changed'
    else:
        raise AssertionError('Changed sample was accepted')
assert 'torch' not in sys.modules and 'openecon.analysis' not in sys.modules
import openecon.survey as module
assert 'site-packages' in str(Path(module.__file__).resolve())
print(json.dumps({'status':'passed', 'geometry':design.validation.model_dump(),
    'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    'installed_module':module.__file__, 'json_csv_roundtrip':True,
    'csv_dtype_drift_rejected':True, 'explicit_csv_dtype_restoration':True,
    'changed_sample_rejected':True,'torch_started':False,
    'estimation':False,'inference':False,'native_desktop':False}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    result = subprocess.run([str(args.python.absolute()), "-I", "-c", PROGRAM],
                            capture_output=True, text=True, check=True)
    record = json.loads(result.stdout)
    root = Path(__file__).resolve().parents[1]
    expected = hashlib.sha256((root / "src/openecon/survey.py").read_bytes()).hexdigest()
    assert record["module_sha256"] == expected, "Installed wheel differs from checked source"
    record["source_ref"] = args.source_ref
    record["scope"] = "Fresh library-only macOS ARM64 wheel; declaration only. No estimator/vendor/frozen/native proof."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "module_sha256": record["module_sha256"]}))


if __name__ == "__main__":
    main()
