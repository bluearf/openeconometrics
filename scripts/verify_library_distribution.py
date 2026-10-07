"""Measure an installed library-only wheel using its own isolated interpreter."""
import argparse
from pathlib import Path
import json
import subprocess
import time

PROGRAM = r'''
import importlib.metadata as meta
import importlib.util
from pathlib import Path
import tempfile
import zipfile
import openecon as oe
from openecon.data import DataError
import openecon_charts as charts
assert '.venv' in oe.__file__ or 'site-packages' in oe.__file__
excluded = ['fastapi','uvicorn','typer','mcp','openpyxl','pyreadstat','scipy','statsmodels','linearmodels']
assert all(importlib.util.find_spec(name) is None for name in excluded)
frame = oe.example()
model = oe.ols(data=frame,y='wage',x=['education','experience'],covariance='HC3')
assert model.nobs == 480
restored = oe.ResultBundle.model_validate_json(model.model_dump_json())
assert restored.model_dump(mode='json') == model.model_dump(mode='json')
assert '\\begin{tabular}' in model.to_latex()
with tempfile.TemporaryDirectory() as root:
    path=Path(root)/'chart.html'
    charts.scatter(data=frame,x='education',y='wage').save_html(path)
    assert path.stat().st_size > 100
    for extension in ['xlsx','dta']:
        missing_file = Path(root) / f'missing-extra.{extension}'
        if extension == 'xlsx':
            # Reader admission now validates the ZIP before checking optional
            # parsers. Supply a real minimal numeric workbook, not invalid bytes.
            documents = {
                '[Content_Types].xml': '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>',
                '_rels/.rels': '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
                'xl/workbook.xml': '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>',
                'xl/_rels/workbook.xml.rels': '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
                'xl/worksheets/sheet1.xml': '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData></worksheet>',
            }
            with zipfile.ZipFile(missing_file, 'w') as archive:
                for name, text in documents.items():
                    archive.writestr(name, text)
        else:
            __import__('pandas').DataFrame({'value':[1.,2.]}).to_stata(missing_file,version=118,write_index=False)
        try:
            oe.read(missing_file)
        except (ImportError, DataError) as error:
            assert 'openecon[files]' in str(error)
        else:
            raise AssertionError('Missing file extra was not reported')
records=[]
for dist in meta.distributions():
    files=[Path(dist.locate_file(file)) for file in dist.files or []]
    records.append({'name':dist.metadata['Name'],'version':dist.version,
                    'installed_bytes':sum(file.stat().st_size for file in files if file.is_file())})
print(__import__('json').dumps({'nobs':model.nobs,'fit_json_latex_html_passed':True,
    'excluded_distributions':excluded,'distributions':sorted(records,key=lambda r:r['name'].lower()),
    'module_from_installed_wheel':str(Path(oe.__file__).resolve()),
    'numpy_transitive_present':importlib.util.find_spec('numpy') is not None,
    'networkx_transitive_present':importlib.util.find_spec('networkx') is not None}))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-ref')
    args=parser.parse_args()
    samples=[]
    for _ in range(3):
        result=subprocess.run([str(args.python.absolute()),'-I','-c',
            'import time; t=time.perf_counter(); import openecon; print(time.perf_counter()-t)'],check=True,capture_output=True,text=True)
        samples.append(float(result.stdout))
    start=time.perf_counter()
    result=subprocess.run([str(args.python.absolute()),'-I','-c',PROGRAM],check=True,capture_output=True,text=True)
    record=json.loads(result.stdout)
    if args.source_ref:
        record['source_ref']=args.source_ref
    record.update(import_seconds=samples,verification_seconds=time.perf_counter()-start,
                  installed_bytes=sum(row['installed_bytes'] for row in record['distributions']),
                  scope='Fresh library-only installed wheels on macOS ARM64; process startup is outside import timings.')
    missing={}
    for label,command in [('cli',['-m','openecon.entrypoints']),('server',['-c','import openecon.server']),('agent',['-c','import openecon.mcp_server']),('desktop',['-c',"from openecon.desktop_entry import main; main(['--data-root','unused-core-check'])"])]:
        # CLI is exercised via its installed console script, not an inert module.
        argv=[str(args.python.absolute().parent/'openecon'),'--help'] if label=='cli' else [str(args.python.absolute()),'-I',*command]
        check=subprocess.run(argv,capture_output=True,text=True)
        assert check.returncode != 0 and f'openecon[{label}]' in check.stderr,(label,check.stderr)
        missing[label]='actionable missing-extra error'
    record['missing_extras']=missing
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'passed','distributions':len(record['distributions']),'installed_bytes':record['installed_bytes']}))


if __name__=='__main__':
    main()
