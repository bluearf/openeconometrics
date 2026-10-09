"""Fresh frozen frequency uncertainty, full-vector reconstruction and restart proof.

Actual native Run/Quit/reopen is collected by the separate read-only capture
helper. This verifier owns only a fresh temporary runtime and fresh receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen
from uuid import uuid4

from verify_multivariate_weight_matrix_runtime import digest, finite_json, normalized

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/frequency_uncertainty_eight.py"
MARKER = "FREQUENCY_UNCERTAINTY_EIGHT_OK "
REOPEN_MARKER = "FREQUENCY_UNCERTAINTY_FULL_RESULTS_RECONSTRUCTED"
CASES = ("covariance_axes", "correlation_axes", "covariance_subspace", "correlation_subspace",
         "canonical_correlations", "canonical_coefficients", "unrotated_factor", "target_factor")
MODULES = tuple("openecon.econometrics.multivariate" + ("."+name if name else "")
    for name in ("", "pca_subspace", "pca_uncertainty", "pca_score_uncertainty",
                 "canon_uncertainty", "ca_uncertainty", "pca", "canon", "common",
                 "uncertainty", "factor", "summary", "extraction", "extraction_extensions",
                 "rotation", "rotation_extensions", "weighted", "replay",
                 "frequency_bootstrap", "pca_frequency_uncertainty", "canon_frequency_uncertainty",
                 "factor_frequency_uncertainty", "factor_uncertainty")) + (
    "openecon.econometrics.stats", "openecon.econometrics.stats.replay",
    "openecon.econometrics.stats.common", "openecon.econometrics.core",
    "openecon.econometrics.postest.index_codec", "openecon.econometrics.summary_state",
    "openecon.econometrics.resident_cpu", "openecon.engines.inference",
    "openecon.engines.linalg", "openecon.engines.distributions",
    "openecon.resources", "openecon.dataset", "openecon.analysis",
    "openecon.analysis_contracts")


def source_identity(runtime, source_ref):
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    sources = {}
    for name in MODULES:
        path = ROOT / "src" / Path(*name.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        source = subprocess.check_output(["git", "show", f"{source_ref}:{path.relative_to(ROOT)}"], cwd=ROOT)
        assert normalized(archive.extract(name)) == normalized(compile(source, str(path), "exec", dont_inherit=True)), name
        sources[name] = hashlib.sha256(source).hexdigest()
    return sources


def verify_outputs(run, *, require_frozen=True):
    assert run["status"] == "ok", run.get("error")
    lines = [line for line in run["stdout"].splitlines() if line.startswith(MARKER)]
    assert len(lines) == 1, "Exactly one fresh frequency-uncertainty marker is required"
    proof = json.loads(lines[0][len(MARKER):])
    assert proof["cases"] == list(CASES)
    if require_frozen:
        assert proof["frozen"]
    assert proof["bootstrap_replications"] == 199
    assert proof["fixture_rows"] == proof["canonical_rows"] == 603
    assert proof["full_saved_roundtrip"]
    assert set(proof["hashes"]) == set(CASES)
    assert set(proof["fit_table_counts"]) == set(proof["n_units"]) == set(CASES)
    assert len(run["outputs"]) == 8
    assert all(item["type"] == "table" and any(environment in item["latex"]
               for environment in ("\\begin{tabular}", "\\begin{longtable}"))
               for item in run["outputs"])
    assert "Display limit reached" not in run["stdout"]
    return proof


def validate_result_files(directory, proof):
    hashes = {name: digest(directory/(name+".json")) for name in CASES}
    assert hashes == proof["hashes"]
    for name in CASES:
        payload = json.loads((directory/(name+".json")).read_text())
        assert set(payload) == {"fit", "post"} and payload["post"] is None and finite_json(payload)
        state = payload["fit"]
        assert set(state) == {"summary", "latex", "table_order", "table_dtypes"}
        assert any(token in state["latex"] for token in ("\\begin{tabular}", "\\begin{longtable}"))
        saved = json.loads(state["summary"])
        assert saved["schema"] == "openecon.summary.v1" and finite_json(saved)
        tables, attrs = saved["tables"], saved["attrs"]
        assert attrs["replications"] == 199 and 0 < attrs["parameter_dimension"] <= 260
        assert len(tables["replicates"]["data"]) == 199
        assert len(tables["replicates"]["columns"]) == attrs["parameter_dimension"]
        assert len(tables["covariance"]["data"]) == len(tables["covariance"]["columns"]) == attrs["parameter_dimension"]
        assert attrs["n_units"] == proof["n_units"][name]
        assert attrs["successful_replications"] == 199 and attrs["failed_replications"] == []
        counts = tables["replicate_counts"]
        source = [row[0] for row in tables["source_frequencies"]["data"]]
        assert len(counts["data"]) == 199 and len(counts["columns"]) == len(source) == attrs["n_complete"]
        assert all(type(value) is int and 0 <= value <= attrs["n_units"] for value in source)
        assert sum(source) == attrs["n_units"]
        for row in counts["data"]:
            assert len(row) == len(source)
            assert all(type(value) is int and 0 <= value <= attrs["n_units"] for value in row)
            assert sum(row) == attrs["n_units"]
            assert all(value == 0 for value, original in zip(row, source, strict=True) if original == 0)
        assert len(tables) == proof["fit_table_counts"][name]
        assert set(state["table_dtypes"]) == set(state["table_order"]) == set(tables)
        assert len(state["table_order"]) == len(tables)
        for key, frame in tables.items():
            assert len(frame["index"]) == len(frame["data"])
            assert all(len(row) == len(frame["columns"]) for row in frame["data"])
            assert len(state["table_dtypes"][key]) == len(frame["columns"])
    fixtures = json.loads((directory/"fixtures.json").read_text())
    assert set(fixtures) == {"spectral", "canonical"} and finite_json(fixtures)
    assert len(fixtures["spectral"]["data"]) == len(fixtures["canonical"]["data"]) == 603
    return hashes


def validate_displayed_tables(run, directory):
    """Every primary estimates table fits the preview; compare every cell."""
    for name, item in zip(CASES, run["outputs"], strict=True):
        payload = json.loads((directory/(name+".json")).read_text())
        frame = json.loads(payload["fit"]["summary"])["tables"]["estimates"]
        shown = item["data"]
        assert len(frame["data"]) <= 50 and len(frame["columns"]) <= 30
        assert shown["total_rows"] == len(frame["data"])
        assert shown["total_columns"] == len(frame["columns"])
        assert shown["columns"] == [str(value) for value in frame["columns"]]
        assert shown["rows"] == frame["data"], name
        assert shown["index"] == [[value] for value in frame["index"]]
        assert shown["index_names"] == frame["index_names"]


def frozen_header():
    return ("import sys, importlib, importlib.util\nfrom pathlib import Path\n"
            "assert getattr(sys, 'frozen', False)\n"
            "assert importlib.util.find_spec('scipy') is None\n"
            "assert importlib.util.find_spec('statsmodels') is None\n"
            f"for module_name in {MODULES!r}:\n"
            "    assert Path(importlib.import_module(module_name).__file__).is_relative_to(Path(sys._MEIPASS))\n")


_RECONSTRUCTION = r'''
import json
from pathlib import Path
import pandas as pd
import torch
import openecon as oe
from openecon.econometrics.multivariate.frequency_bootstrap import validate_saved
from openecon.econometrics.postest.index_codec import encode

torch.set_num_threads(2)
fixtures = json.loads((root/'fixtures.json').read_text())
VARIABLES = [f'x{i+1}' for i in range(6)]
CANONICAL = ['x1','x2','x3','y1','y2','y3']
TARGET = [[.8,0],[.75,.1],[.7,-.05],[0,.65],[.1,.6],[-.05,.55]]

def array(frame):
    return torch.tensor(frame.to_numpy(dtype='float64'), dtype=torch.float64)

def close(actual, expected, name):
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual, dtype=torch.float64)
    assert actual.shape == expected.shape, name
    assert torch.allclose(actual, expected, rtol=3e-10, atol=3e-11), name

def restore_pack(state):
    result = oe.restore_summary(state['summary'])
    assert oe.summary_state(result) == state['summary']
    for key,dtypes in state['table_dtypes'].items():
        result[key] = result[key].astype(dict(zip(result[key].columns,dtypes,strict=True)))
    result = type(result)({key:result[key] for key in state['table_order']},title=result.title,**result.attrs)
    assert oe.summary_state(result) == state['summary']
    assert result.to_latex() == state['latex']
    validate_saved(result)  # Integrity only; numerical refits below are independent.
    return result

def moments(values, origin=None):
    # Literal-expanded verification only. Production never makes this N-by-p array.
    origin = values[0] if origin is None else origin
    differences = values-origin
    average = differences.mean(0)
    correction = (differences-average).mean(0)
    offset = average+correction
    centered = differences-offset
    covariance = centered.T@centered/(len(values)-1)
    covariance = (covariance+covariance.T)/2
    return {'mean':origin+offset, 'sd':covariance.diagonal().sqrt(),
            'covariance':covariance, 'centered':centered, 'origin':origin, 'offset':offset}

def correlation(moment):
    target = moment['covariance']/torch.outer(moment['sd'],moment['sd'])
    target = (target+target.T)/2
    target.diagonal().fill_(1)
    return target

def pca_fit(values, matrix, subspace):
    fit = moments(values)
    target = fit['covariance'] if matrix=='covariance' else correlation(fit)
    roots,vectors = torch.linalg.eigh(target)
    roots,vectors = roots.flip(0),vectors.flip(1)[:,:2]
    fit.update(roots=roots,target=target)
    if subspace:
        projector = vectors@vectors.T
        triangular = torch.triu_indices(6,6)
        captured,residual = roots[:2].sum(),roots[2:].sum()
        fit.update(projector=projector,vector=torch.cat((projector[triangular[0],triangular[1]],torch.stack((captured,residual,captured/(captured+residual))))))
    else:
        signs = torch.tensor([1. if vectors[i,j]>0 else -1. for j,i in enumerate([0,3])],dtype=torch.float64)
        vectors = vectors*signs
        loadings = vectors*roots[:2].sqrt()
        fit.update(vectors=vectors,loadings=loadings,
                   vector=torch.cat([torch.cat((roots[j:j+1],vectors[:,j],loadings[:,j])) for j in range(2)]))
    return fit

def canonical_fit(values, coefficients):
    fit = moments(values)
    r = correlation(fit)
    rxx,ryy,rxy = r[:3,:3],r[3:,3:],r[:3,3:]
    # Independent symmetric whitening, rather than the product's weighted row QR.
    def whitening(block):
        roots,vectors = torch.linalg.eigh(block)
        return (vectors/roots.sqrt())@vectors.T
    wx,wy = whitening(rxx),whitening(ryy)
    left,roots,right_t = torch.linalg.svd(wx@rxy@wy,full_matrices=False)
    fit.update(roots=roots,target=r,vector=roots[:2],matrices={})
    if coefficients:
        a,b = wx@left[:,:2],wy@right_t.T[:,:2]
        signs = torch.tensor([1. if a[i,j]>0 else -1. for j,i in enumerate([0,1])],dtype=torch.float64)
        a,b = a*signs,b*signs
        matrices = {'x_coefficients':a/fit['sd'][:3,None],
                    'y_coefficients':b/fit['sd'][3:,None],
                    'x_standardized_coefficients':a,'y_standardized_coefficients':b,
                    'x_loadings':rxx@a,'y_loadings':ryy@b,
                    'x_cross_loadings':rxy@b,'y_cross_loadings':rxy.T@a}
        fit.update(matrices=matrices,
                   vector=torch.cat([torch.cat([roots[j:j+1],*[matrix[:,j] for matrix in matrices.values()]]) for j in range(2)]))
    return fit

def factor_fit(values, targeted):
    fit = moments(values)
    r = correlation(fit)
    inverse = torch.linalg.solve(r,torch.eye(6,dtype=torch.float64))
    smc = (1-1/inverse.diagonal()).clamp(0,1)
    reduced = r.clone()
    reduced.diagonal().copy_(smc)
    roots,vectors = torch.linalg.eigh(reduced)
    roots,vectors = roots.flip(0),vectors.flip(1)
    loadings = vectors[:,:2]*roots[:2].sqrt()
    if targeted:
        target = torch.tensor(TARGET,dtype=torch.float64)
        left,_,right_t = torch.linalg.svd(loadings.T@target,full_matrices=False)
        rotation = left@right_t
        loadings = loadings@rotation
    else:
        signs = torch.tensor([1. if loadings[i,j]>0 else -1. for j,i in enumerate([0,3])],dtype=torch.float64)
        rotation = torch.diag(signs)
        loadings = loadings*signs
    uniqueness = 1-loadings.square().sum(1)
    fit.update(roots=roots,target=r,smc=smc,loadings=loadings,uniqueness=uniqueness,
               rotation=rotation,vector=torch.cat((loadings.flatten(),uniqueness)))
    return fit

def statistical_tables(result,refits,point):
    vectors = torch.stack([fit['vector'] for fit in refits])
    close(array(result['replicates']),vectors,'every full replicate vector')
    close(torch.tensor(result['estimates']['estimate'].to_numpy(),dtype=torch.float64),point['vector'],'point parameter vector')
    differences = vectors-vectors[0]
    average = differences.mean(0)
    offset = average+(differences-average).mean(0)
    centered = differences-offset
    covariance = centered.T@centered/(len(vectors)-1)
    covariance = (covariance+covariance.T)/2
    close(array(result['covariance']),covariance,'full joint covariance')
    close(torch.tensor(result['estimates']['std_error'].to_numpy(),dtype=torch.float64),covariance.diagonal().sqrt(),'every standard error')
    tails = torch.tensor([(1-result.attrs['confidence'])/2,(1+result.attrs['confidence'])/2],dtype=torch.float64)
    intervals = torch.quantile(vectors,tails,dim=0,interpolation='linear')
    for key,value in [('ci_lower',intervals[0]),('ci_upper',intervals[1]),('bootstrap_bias',(vectors[0]-point['vector'])+offset)]:
        close(torch.tensor(result['estimates'][key].to_numpy(),dtype=torch.float64),value,key)
    assert result['estimates'][['p_value','df']].isna().all().all()
    close(array(result['replicate_means']),torch.stack([fit['mean'] for fit in refits]),'all replicate means')
    close(array(result['replicate_standard_deviations']),torch.stack([fit['sd'] for fit in refits]),'all replicate SDs')
    close(array(result['point_covariance']),point['covariance'],'full point covariance')
    close(array(result['descriptives']),torch.stack((point['mean'],point['sd']),dim=1),'point means and SDs')
    if 'replicate_covariances' in result:
        close(array(result['replicate_covariances']),torch.stack([fit['covariance'].flatten() for fit in refits]),'every full replicate covariance')


def frequency_reconstruction(result,name):
    a = result.attrs
    canonical = name.startswith('canonical_')
    factor = name.endswith('_factor')
    subspace = name.endswith('_subspace')
    fixture = fixtures['canonical' if canonical else 'spectral']
    names = CANONICAL if canonical else VARIABLES
    frame = pd.DataFrame(fixture['data'],columns=fixture['columns'])
    complete = frame[names+['freq']].notna().all(axis=1).tolist()
    positions = [i for i,flag in enumerate(complete) if flag]
    source_codes = [fixture['index_codes'][i] for i in positions]
    values = array(frame.iloc[positions][names])
    counts = torch.tensor(frame.iloc[positions]['freq'].to_numpy(),dtype=torch.int64)
    n = int(counts.sum())
    assert a['variables']==names and a['weights']=='freq' and a['weight_type']=='fweight'
    assert a['n_units']==a['nobs']==n and a['physical_rows']==a['n_physical']==603
    assert a['n_complete']==len(positions) and a['n_missing']==a['n_dropped']==603-len(positions)
    assert a['n_zero_weight']==int((counts==0).sum()) and a['n_positive_frequency']==int((counts>0).sum())
    assert a['index_names_json']==fixture['index_names']
    assert a['source_columns_json']==[encode(label) for label in names+['freq']]
    assert a['column_axis_names_json']==fixture['column_axis_names']
    assert a['covariance_divisor']==198 and a['moment_covariance_divisor']==n-1
    assert a['replications']==a['successful_replications']==199 and a['failed_replications']==[]
    assert a['physical_row_resampling'] is False and a['expanded_measurements_materialized'] is False
    assert a['zero_frequency_source_rows_preserved'] is True and a['confidence']==.95 and a['missing']=='drop'
    close(array(result['sample']),values,'all complete original source measurements, including zero-frequency rows')
    assert list(result['sample'].columns)==names
    assert list(result['source_index']['source_position'])==positions
    assert list(result['source_index']['index_code'])==source_codes
    assert torch.equal(torch.tensor(result['source_frequencies']['frequency'].to_numpy(),dtype=torch.int64),counts)
    accounting = result['source_accounting']
    assert list(accounting['source_position'])==list(range(603))
    assert list(accounting['index_code'])==fixture['index_codes']
    assert list(accounting['complete'])==complete
    expected_frequency = ['null' if pd.isna(value) else str(int(value)) for value in frame['freq']]
    assert list(accounting['frequency_json'])==expected_frequency
    assert list(accounting['zero_complete_frequency'])==[bool(flag and frame['freq'].iloc[i]==0) for i,flag in enumerate(complete)]
    assert list(result['replicate_counts'].columns)==positions
    assert list(result['replicate_counts'].index)==list(range(1,200))
    # Oracle-only expansion preserves the order of literal units implied by CDF bins.
    expanded_source = torch.repeat_interleave(torch.arange(len(values)),counts)
    expanded_values = values[expanded_source]
    generator = torch.Generator(device='cpu').manual_seed(a['seed'])
    if canonical:
        assert a['seed']==19 and a['components']==2 and a['x']==names[:3] and a['y']==names[3:]
        assert a['target']==('coefficients' if name=='canonical_coefficients' else 'correlations')
        assert a['sign_anchors']==(['x1','x2'] if name=='canonical_coefficients' else None)
        fit = lambda data:canonical_fit(data,name=='canonical_coefficients')
    elif factor:
        assert a['seed']==53 and a['factors']==2 and a['method']=='pf'
        assert a['target']==(TARGET if name=='target_factor' else None)
        assert a['sign_anchors']==(None if name=='target_factor' else ['x1','x4'])
        if name=='target_factor':
            close(array(result['target']),torch.tensor(TARGET,dtype=torch.float64),'independently fixed target')
        fit = lambda data:factor_fit(data,name=='target_factor')
    else:
        matrix = 'covariance' if name.startswith('covariance_') else 'correlation'
        assert a['seed']==71 and a['components']==2 and a['matrix']==matrix
        if not subspace:
            assert a['sign_anchors']==['x1','x4']
        fit = lambda data:pca_fit(data,matrix,subspace)
    point = fit(expanded_values)
    refits,all_counts,origins,offsets = [],[],[],[]
    for b in range(199):
        selected = torch.randint(n,(n,),generator=generator,device='cpu')
        current_counts = torch.bincount(expanded_source[selected],minlength=len(values))
        assert int(current_counts.sum())==n and bool((current_counts[counts==0]==0).all())
        all_counts.append(current_counts)
        data = expanded_values[selected]
        refits.append(fit(data))
        # The product anchors to the FIRST positive physical source position,
        # independently of sampled unit order. Compare exact origins separately.
        origin = values[int((current_counts>0).nonzero()[0])]
        origins.append(origin)
        offsets.append(moments(data,origin)['offset'])
    assert torch.equal(torch.tensor(result['replicate_counts'].to_numpy(),dtype=torch.int64),torch.stack(all_counts))
    assert torch.equal(array(result['replicate_origins']),torch.stack(origins))
    close(array(result['replicate_mean_offsets']),torch.stack(offsets),'all unrounded positive-source centering offsets')
    point_origin = values[int((counts>0).nonzero()[0])]
    assert torch.equal(array(result['point_origin']).reshape(-1),point_origin)
    close(array(result['point_mean_offset']).reshape(-1),moments(expanded_values,point_origin)['offset'],'point unrounded centering offset')
    statistical_tables(result,refits,point)
    if canonical:
        close(array(result['point_correlation']),point['target'],'point CCA correlation')
        close(array(result['point_correlations']).reshape(-1),point['roots'],'all point canonical roots')
        close(array(result['replicate_correlations']),torch.stack([fit['roots'] for fit in refits]),'all199 full canonical root vectors')
        for key,matrix in point['matrices'].items():
            close(array(result['point_'+key]),matrix,'point '+key)
    elif factor:
        close(array(result['point_correlation']),point['target'],'point factor correlation')
        for table,key in [('point_eigenvalues','roots'),('point_smc','smc'),('point_uniqueness','uniqueness')]:
            close(array(result[table]).reshape(-1),point[key],table)
        close(array(result['point_loadings']),point['loadings'],'point factor loadings')
        # Raw reduced eigenvectors have arbitrary signs. Validate the saved
        # nuisance change of basis through its mathematical identities instead.
        rotation = array(result['point_rotation_matrix'])
        identity = torch.eye(2,dtype=torch.float64)
        close(rotation.T@rotation,identity,'orthogonal point nuisance rotation')
        raw_loadings = array(result['point_loadings'])@rotation.T
        reduced = point['target'].clone()
        reduced.diagonal().copy_(point['smc'])
        close(reduced@raw_loadings,raw_loadings*point['roots'][:2],'saved nuisance basis is a reduced eigensystem')
        close(raw_loadings.T@raw_loadings,torch.diag(point['roots'][:2]),'saved nuisance basis root normalization')
        if name=='target_factor':
            left,_,right_t = torch.linalg.svd(raw_loadings.T@torch.tensor(TARGET,dtype=torch.float64),full_matrices=False)
            close(rotation,left@right_t,'saved full-target polar mapping')
        else:
            close(rotation,torch.diag(rotation.diagonal()),'unrotated nuisance mapping consists only of signs')
            close(rotation.diagonal().abs(),torch.ones(2,dtype=torch.float64),'unrotated nuisance signs')
        close(array(result['replicate_eigenvalues']),torch.stack([fit['roots'] for fit in refits]),'all199 reduced factor root vectors')
        close(array(result['replicate_smc']),torch.stack([fit['smc'] for fit in refits]),'all199 full SMC vectors')
    else:
        close(array(result['point_matrix']),point['target'],'point PCA matrix')
        close(array(result['point_eigenvalues']).reshape(-1),point['roots'],'full point PCA roots')
        if subspace:
            close(array(result['point_projector']),point['projector'],'point invariant projector')
        else:
            close(array(result['point_eigenvectors']),point['vectors'],'point signed eigenvectors')
            close(array(result['point_loadings']),point['loadings'],'point signed loadings')

for name in CASES:
    payload = json.loads((root/(name+'.json')).read_text())
    assert payload['post'] is None
    frequency_reconstruction(restore_pack(payload['fit']),name)
print(REOPEN_MARKER)
'''


def reconstruction_code(directory):
    return (f"from pathlib import Path\nroot=Path({str(directory)!r})\n"
            f"CASES={CASES!r}\nREOPEN_MARKER={REOPEN_MARKER!r}\n"+_RECONSTRUCTION)


def verify(runtime, result_directory, source_ref):
    runtime = runtime.resolve(strict=True)
    fingerprint, sources = digest(runtime), source_identity(runtime, source_ref)
    result_directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    with tempfile.TemporaryDirectory(prefix="openecon-frequency-uncertainty-frozen-") as temporary:
        data = Path(temporary)
        prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"
        process = descriptor = token = None
        errors = (data/"errors.log").open("ab")

        def call(path, body=None):
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-OpenEcon-Token"] = token
            request = Request(descriptor["url"]+prefix+path, headers=headers,
                              data=json.dumps(body).encode() if body is not None else None)
            with urlopen(request, timeout=180) as response:
                return json.load(response)

        def start():
            nonlocal process, descriptor, token
            process = subprocess.Popen([str(runtime), "--port", "0", "--data-root", str(data)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                start_new_session=True, env=environment)
            descriptor = token = None
            deadline = time.monotonic()+60
            while time.monotonic() < deadline:
                if select.select([process.stdout], [], [], .25)[0]:
                    descriptor = json.loads(process.stdout.readline(8193))
                    break
                assert process.poll() is None, "Owned runtime exited before readiness"
            assert descriptor and descriptor["type"] == "ready"
            token = call("/session")["token"]

        def stop():
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)

        try:
            start()
            code = frozen_header()+f"SPECTRAL_GEOMETRY_RESULT_DIRECTORY={str(result_directory)!r}\n"+EXAMPLE.read_text()
            run = call("/console/execute", {"code": code, "timeout_seconds": 180})
            proof = verify_outputs(run)
            hashes = validate_result_files(result_directory, proof)
            validate_displayed_tables(run, result_directory)
            call("/console/reset", {})
            stored = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert all(stored[key] == run[key] for key in ("code", "stdout", "outputs", "events"))
            stop()
            start()
            stored = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert all(stored[key] == run[key] for key in ("code", "stdout", "outputs", "events"))
            reopened = call("/console/execute", {"code": frozen_header()+reconstruction_code(result_directory), "timeout_seconds": 180})
            assert reopened["status"] == "ok", reopened.get("error")
            assert REOPEN_MARKER in reopened["stdout"]
            assert hashes == validate_result_files(result_directory, proof)
            assert digest(runtime) == fingerprint
            receipt = {"status": "passed", "frozen": True, "compiled_source_ref": source_ref,
                "source_identity": sources, "runtime_sha256": fingerprint, "proof": proof,
                "duration_ms": run["duration_ms"], "complete_payload_hashes": hashes,
                "fixture_sha256": digest(result_directory/"fixtures.json"),
                "example_sha256": digest(EXAMPLE), "verifier_sha256": digest(__file__),
                "worker_reset_readback": True, "runtime_restart_readback": True,
                "full_results_reopened": True, "full_seeded_draw_vectors_reconstructed": 8,
                "full_bootstrap_covariance_SE_bias_CI_rederived": 8,
                "full_seeded_frequency_count_vectors_reconstructed": 8,
                "full_original_frequency_sources_and_accounting_reconstructed": 8,
                "full_point_and_replicate_origins_offsets_reconstructed": 8,
                "literal_expansion_used_only_in_independent_verification": True,
                "primary_display_tables_full_cells_equal": 8,
                "external_oracle_packages_absent": True, "native_ui_verified": False,
                "public_release_delivered": False}
        finally:
            stop()
            errors.close()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path")
    result = verify(args.runtime, args.results, args.source_ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "frozen", "full_results_reopened", "duration_ms")}))
