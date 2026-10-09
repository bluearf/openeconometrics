"""Fresh frozen spectral geometry, full-vector reconstruction and restart proof.

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
EXAMPLE = ROOT / "docs/examples/spectral_geometry_uncertainty.py"
MARKER = "SPECTRAL_GEOMETRY_EIGHT_OK "
REOPEN_MARKER = "SPECTRAL_GEOMETRY_FULL_RESULTS_RECONSTRUCTED"
CASES = ("covariance_subspace", "correlation_subspace", "canonical_correlations",
         "canonical_coefficients", "multinomial_ca", "row_multinomial_ca",
         "covariance_scores", "correlation_scores")
MODULES = tuple("openecon.econometrics.multivariate" + ("."+name if name else "")
    for name in ("", "pca_subspace", "pca_uncertainty", "pca_score_uncertainty",
                 "canon_uncertainty", "ca_uncertainty", "pca", "canon", "common",
                 "uncertainty", "factor", "summary", "extraction", "extraction_extensions",
                 "rotation", "rotation_extensions", "weighted", "replay")) + (
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
    assert len(lines) == 1, "Exactly one fresh spectral-geometry marker is required"
    proof = json.loads(lines[0][len(MARKER):])
    assert proof["cases"] == list(CASES)
    if require_frozen:
        assert proof["frozen"]
    assert proof["bootstrap_replications"] == 199
    assert proof["fixture_rows"] == proof["canonical_rows"] == 603
    assert proof["full_saved_roundtrip"]
    assert set(proof["hashes"]) == set(CASES)
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
        assert attrs["replications"] == 199 and attrs["parameter_dimension"] <= 128
        assert len(tables["replicates"]["data"]) == 199
        assert len(tables["covariance"]["data"]) == len(tables["covariance"]["columns"]) == attrs["parameter_dimension"]
        assert len(tables) == proof["fit_table_counts"][name]
        assert set(state["table_dtypes"]) == set(state["table_order"]) == set(tables)
        assert len(state["table_order"]) == len(tables)
        for key, frame in tables.items():
            assert len(frame["index"]) == len(frame["data"])
            assert all(len(row) == len(frame["columns"]) for row in frame["data"])
            assert len(state["table_dtypes"][key]) == len(frame["columns"])
    fixtures = json.loads((directory/"fixtures.json").read_text())
    assert set(fixtures) == {"spectral", "canonical", "counts", "query"} and finite_json(fixtures)
    assert len(fixtures["spectral"]["data"]) == len(fixtures["canonical"]["data"]) == 603
    assert sum(sum(row) for row in fixtures["counts"]["data"]) == 4280
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
import math
from pathlib import Path
import pandas as pd
import torch
import openecon as oe
from openecon.econometrics.postest.index_codec import encode
torch.set_num_threads(2)
fixtures = json.loads((root/'fixtures.json').read_text())
def array(frame):
    return torch.tensor(frame.to_numpy(dtype='float64'), dtype=torch.float64)
def close(actual, expected, name):
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual, dtype=torch.float64)
    assert actual.shape == expected.shape, name
    assert torch.allclose(actual, expected, rtol=3e-10, atol=3e-11), name
def restore_pack(state):
    result = oe.restore_summary(state['summary'])
    assert json.loads(oe.summary_state(result)) == json.loads(state['summary'])
    for key,dtypes in state['table_dtypes'].items():
        result[key] = result[key].astype(dict(zip(result[key].columns,dtypes)))
    result = type(result)({key:result[key] for key in state['table_order']},title=result.title,**result.attrs)
    assert oe.summary_state(result) == state['summary']
    assert result.to_latex() == state['latex']
    return result
def moments(values):
    origin = values[0]
    differences = values-origin
    average = differences.mean(0)
    centered = differences-average
    correction = centered.mean(0)
    centered = centered-correction
    mean = origin+(average+correction)
    covariance = centered.T@centered/(len(values)-1)
    covariance = (covariance+covariance.T)/2
    sd = covariance.diagonal().sqrt()
    return mean,sd,covariance,centered
def pca_fit(values, matrix, components, anchors=None, subspace=False):
    mean,sd,covariance,_ = moments(values)
    target = covariance.clone()
    if matrix == 'correlation':
        target = covariance/torch.outer(sd,sd)
        target = (target+target.T)/2
        target.diagonal().fill_(1)
    roots,vectors = torch.linalg.eigh(target)
    roots,vectors = roots.flip(0),vectors.flip(1)[:,:components]
    if subspace:
        projector = vectors@vectors.T
        triangular = torch.triu_indices(len(mean),len(mean))
        captured,residual = roots[:components].sum(),roots[components:].sum()
        parameters = torch.cat((projector[triangular[0],triangular[1]],torch.stack((captured,residual,captured/(captured+residual)))))
        return parameters,mean,sd,projector,roots,covariance,target
    signs = torch.tensor([1. if vectors[i,j]>0 else -1. for j,i in enumerate(anchors)],dtype=torch.float64)
    vectors = vectors*signs
    parameters = torch.cat([torch.cat((roots[j:j+1],vectors[:,j],vectors[:,j]*roots[j].sqrt())) for j in range(components)])
    return parameters,mean,sd,vectors,roots,covariance,target
def canonical_fit(values, a):
    mean,sd,covariance,centered = moments(values)
    standardized = centered/sd
    correlation = standardized.T@standardized/(len(values)-1)
    correlation = (correlation+correlation.T)/2
    correlation.diagonal().fill_(1)
    p,m = len(a['x']),a['components']
    rxx,ryy,rxy = correlation[:p,:p],correlation[p:,p:],correlation[:p,p:]
    # Symmetric whitening/SVD is independent of the product's row QR route.
    def whitening(block):
        roots,vectors = torch.linalg.eigh(block)
        return (vectors/roots.sqrt())@vectors.T
    wx,wy = whitening(rxx),whitening(ryy)
    left,roots,right_t = torch.linalg.svd(wx@rxy@wy,full_matrices=False)
    if a['target']=='correlations':
        return roots[:m],mean,sd,roots,covariance,correlation
    xcoeff,ycoeff = wx@left[:,:m],wy@right_t.T[:,:m]
    anchor_positions = [a['x'].index(label) for label in a['sign_anchors']]
    signs = torch.tensor([1. if xcoeff[i,j]>0 else -1. for j,i in enumerate(anchor_positions)],dtype=torch.float64)
    xcoeff,ycoeff = xcoeff*signs,ycoeff*signs
    matrices = [xcoeff/sd[:p,None],ycoeff/sd[p:,None],xcoeff,ycoeff,
                rxx@xcoeff,ryy@ycoeff,rxy@ycoeff,rxy.T@xcoeff]
    parameters = torch.cat([torch.cat([roots[j:j+1],*[matrix[:,j] for matrix in matrices]]) for j in range(m)])
    return parameters,mean,sd,roots,covariance,correlation
def correspondence_fit(counts,a):
    values = counts.to(torch.float64)
    total = counts.sum()
    p = values/total
    r,c = counts.sum(1).to(torch.float64)/total,counts.sum(0).to(torch.float64)/total
    residual = (p-torch.outer(r,c))/torch.sqrt(torch.outer(r,c))
    left,roots,right_t = torch.linalg.svd(residual,full_matrices=False)
    m = a['dimensions']
    left,right = left[:,:m],right_t.T[:,:m]
    signs = torch.tensor([1. if left[i,j]>0 else -1. for j,i in enumerate(a['sign_anchor_positions'])],dtype=torch.float64)
    left,right = left*signs,right*signs
    row_std,col_std = left/r.sqrt()[:,None],right/c.sqrt()[:,None]
    row_principal,col_principal = row_std*roots[:m],col_std*roots[:m]
    # Total inertia is independently expressed as Pearson chi-square/N.
    expected = torch.outer(counts.sum(1),counts.sum(0)).to(torch.float64)/total
    inertia = ((values-expected).square()/expected).sum()/total
    parameters = torch.cat((inertia.reshape(1),r,c,*[torch.cat((roots[j:j+1],roots[j:j+1].square(),row_std[:,j],row_principal[:,j],col_std[:,j],col_principal[:,j])) for j in range(m)]))
    return parameters,roots
def statistical_tables(result,vectors,point):
    close(array(result['replicates']),vectors,'every full replicate vector')
    close(torch.tensor(result['estimates']['estimate'].to_numpy(),dtype=torch.float64),point,'point parameter vector')
    # Anchored centering leaves structurally constant targets exactly zero.
    differences = vectors-vectors[0]
    mean_difference = differences.mean(0)
    centered = differences-mean_difference
    covariance = centered.T@centered/(len(vectors)-1)
    close(array(result['covariance']),covariance,'full covariance')
    close(torch.tensor(result['estimates']['std_error'].to_numpy(),dtype=torch.float64),covariance.diagonal().sqrt(),'every standard error')
    tails = torch.tensor([(1-result.attrs['confidence'])/2,(1+result.attrs['confidence'])/2],dtype=torch.float64)
    intervals = torch.quantile(vectors,tails,dim=0,interpolation='linear')
    for key,value in [('ci_lower',intervals[0]),('ci_upper',intervals[1]),('bootstrap_bias',(vectors[0]-point)+mean_difference)]:
        close(torch.tensor(result['estimates'][key].to_numpy(),dtype=torch.float64),value,key)
    assert result['estimates'][['p_value','df']].isna().all().all()
def raw_fixture(name):
    f = fixtures[name]
    return pd.DataFrame(f['data'],columns=f['columns'])
def row_reconstruction(result,name,subspace=False,score=False,canonical=False):
    a = result.attrs['fit_attrs'] if score else result.attrs
    prefix = 'fit__' if score else ''
    values = array(result[prefix+'sample'])
    fixture_name = 'canonical' if canonical else 'spectral'
    expected_sample = raw_fixture(fixture_name).iloc[a['sample_positions']][a['variables']]
    close(values,array(expected_sample),'original complete sample')
    assert a['sample_index_codes']==[fixtures[fixture_name]['index_codes'][i] for i in a['sample_positions']]
    generator = torch.Generator(device='cpu').manual_seed(a['seed'])
    indices = torch.stack([torch.randint(len(values),(len(values),),generator=generator,device='cpu') for _ in range(a['replications'])])
    assert torch.equal(torch.tensor(result[prefix+'resample_indices'].to_numpy(),dtype=torch.int64),indices)
    anchors = None if subspace or canonical else [a['variables'].index(label) for label in a['sign_anchors']]
    fit = (lambda x:canonical_fit(x,a)) if canonical else (lambda x:pca_fit(x,a['matrix'],a['components'],anchors,subspace))
    point_fit = fit(values)
    refits = [fit(values[index]) for index in indices]
    vectors = torch.stack([fit[0] for fit in refits])
    close(array(result[prefix+'replicate_means']),torch.stack([fit[1] for fit in refits]),'all replicate means')
    close(array(result[prefix+'replicate_standard_deviations']),torch.stack([fit[2] for fit in refits]),'all replicate SDs')
    if score:
        fit_result = type(result)({key[len(prefix):]:frame for key,frame in result.items() if key.startswith(prefix)},title=result.attrs['fit_title'],**a)
        statistical_tables(fit_result,vectors,point_fit[0])
        query = array(result['query'])[result.attrs['query_positions']]
        expected_query = raw_fixture('query').iloc[result.attrs['query_positions']][a['variables']]
        close(query,array(expected_query),'original fixed query')
        score_vectors = []
        origins,offsets = [],[]
        for ordinal,fitted in enumerate(refits):
            training = values[indices[ordinal]]
            origin = training[0]
            shifted = training-origin
            average = shifted.mean(0)
            correction = (shifted-average).mean(0)
            offset = average+correction
            origins.append(origin)
            offsets.append(offset)
            centered = (query-origin)-offset
            if a['matrix']=='correlation':
                centered /= fitted[2]
            score_vectors.append((centered@fitted[3]).reshape(-1))
        assert torch.equal(array(result['replicate_centering_origins']),torch.stack(origins))
        close(array(result['replicate_centering_offsets']),torch.stack(offsets),'all unrounded centering offsets')
        shifted = values-values[0]
        average = shifted.mean(0)
        point_offset = average+(shifted-average).mean(0)
        assert torch.equal(torch.tensor(result['point_centering']['origin'].to_numpy(),dtype=torch.float64),values[0])
        close(torch.tensor(result['point_centering']['offset'].to_numpy(),dtype=torch.float64),point_offset,'point unrounded centering offset')
        centered = (query-values[0])-point_offset
        if a['matrix']=='correlation':
            centered /= point_fit[2]
        point = (centered@point_fit[3]).reshape(-1)
        statistical_tables(result,torch.stack(score_vectors),point)
    else:
        statistical_tables(result,vectors,point_fit[0])
        if subspace:
            close(array(result['point_projector']),point_fit[3],'point invariant projector')
        if canonical:
            close(array(result['replicate_correlations']),torch.stack([fit[3] for fit in refits]),'all canonical roots')
def count_reconstruction(result):
    a = result.attrs
    counts = torch.tensor(result['point_counts'].to_numpy(),dtype=torch.int64)
    assert torch.equal(counts,torch.tensor(fixtures['counts']['data'],dtype=torch.int64))
    assert a['row_label_codes']==[encode(label) for label in fixtures['counts']['index']]
    assert a['column_label_codes']==[encode(label) for label in fixtures['counts']['columns']]
    generator = torch.Generator(device='cpu').manual_seed(a['seed'])
    rows,columns = counts.shape
    total,row_totals = int(counts.sum()),counts.sum(1)
    all_counts = []
    for b in range(a['replications']):
        if a['sampling']=='multinomial':
            probabilities = counts.flatten().to(torch.float64)/total
            selected = torch.multinomial(probabilities,total,replacement=True,generator=generator)
            draw = torch.bincount(selected,minlength=rows*columns).reshape(rows,columns)
        else:
            draw = []
            for i in range(rows):
                probabilities = counts[i].to(torch.float64)/row_totals[i]
                selected = torch.multinomial(probabilities,int(row_totals[i]),replacement=True,generator=generator)
                draw.append(torch.bincount(selected,minlength=columns))
            draw = torch.stack(draw)
            assert torch.equal(draw.sum(1),row_totals)
        assert int(draw.sum())==total
        all_counts.append(draw)
    draws = torch.stack(all_counts)
    assert torch.equal(torch.tensor(result['resample_counts'].to_numpy(),dtype=torch.int64),draws.reshape(len(draws),-1))
    fits = [correspondence_fit(draw,a) for draw in draws]
    statistical_tables(result,torch.stack([fit[0] for fit in fits]),correspondence_fit(counts,a)[0])
    close(array(result['replicate_spectrum']),torch.stack([fit[1] for fit in fits]),'all CA singular spectra')
for name in CASES:
    payload = json.loads((root/(name+'.json')).read_text())
    assert payload['post'] is None
    result = restore_pack(payload['fit'])
    if name.endswith('_ca'):
        count_reconstruction(result)
    else:
        row_reconstruction(result,name,subspace=name.endswith('_subspace'),
                           score=name.endswith('_scores'),canonical=name.startswith('canonical_'))
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
    with tempfile.TemporaryDirectory(prefix="openecon-spectral-geometry-frozen-") as temporary:
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
                "retained_original_PCA_fit_draw_vectors_reconstructed": 2,
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
