"""Check isolated wheel/sdist source identity and all complete dependent-meta artifacts."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import zipfile

from verify_dependent_meta_runtime import MODULES, RESULT_NAMES, ROOT, receipt


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NoRuntimeOracles(importlib.abc.MetaPathFinder):
    """Make development estimator packages unavailable to this isolated run."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("scipy","statsmodels"):
            raise ModuleNotFoundError(f"Development oracle unavailable: {fullname}")
        return None


def verify(site, wheel, charts_wheel, sdist, source_directory, output_directory, source_revision):
    site = site.resolve(strict=True)
    inherited_sources = [value for value in sys.path if Path(value).name=="src" and
                         ((Path(value)/"openecon").is_dir() or (Path(value)/"openecon_charts").is_dir())]
    sys.path[:] = [value for value in sys.path if value not in inherited_sources]
    if any(name.split(".")[0] in ("openecon","openecon_charts","scipy","statsmodels") for name in sys.modules):
        raise RuntimeError("Expected a fresh isolated package process")
    sys.path.insert(0,str(site))
    sys.meta_path.insert(0,NoRuntimeOracles())
    wheel_records = {}
    with zipfile.ZipFile(wheel) as core_archive,zipfile.ZipFile(charts_wheel) as charts_archive,tarfile.open(sdist,"r:gz") as source_archive:
        members = source_archive.getnames()
        prefix = members[0].split("/",1)[0]
        for name in MODULES:
            is_charts = name.startswith("openecon_charts")
            directory = "packages/openecon-charts/src" if is_charts else "src"
            relative = Path(*name.split("."))
            source = ROOT/directory/relative
            relative = relative/"__init__.py" if source.is_dir() else relative.with_suffix(".py")
            source = ROOT/directory/relative
            imported = importlib.import_module(name)
            installed = Path(imported.__file__).resolve(strict=True)
            if not installed.is_relative_to(site):
                raise RuntimeError(f"Module escaped the owned wheel installation: {name}")
            expected = source.read_bytes()
            committed = subprocess.check_output(["git","show",f"{source_revision}:{source.relative_to(ROOT).as_posix()}"],cwd=ROOT)
            if committed!=expected:
                raise RuntimeError(f"Module differs from the declared scientific source revision: {name}")
            archive = charts_archive if is_charts else core_archive
            source_member = f"{prefix}/{directory}/{relative.as_posix()}"
            if installed.read_bytes()!=expected or archive.read(relative.as_posix())!=expected or source_archive.extractfile(source_member).read()!=expected:
                raise RuntimeError(f"Full module bytes differ across source/wheel/sdist/install: {name}")
            wheel_records[name] = hashlib.sha256(expected).hexdigest()
        family = [name for name in core_archive.namelist() if name.startswith("openecon/econometrics/meta/") and name.endswith(".py")]
        source_family = sorted((ROOT/"src/openecon/econometrics/meta").glob("*.py"))
        if {Path(name).name for name in family}!={path.name for path in source_family}:
            raise RuntimeError("The wheel omits a meta-family source module")
        for source in source_family:
            relative = f"openecon/econometrics/meta/{source.name}"
            if core_archive.read(relative)!=source.read_bytes():
                raise RuntimeError("The complete meta family changed in the wheel")
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        runpy.run_path(str(ROOT/"docs/examples/dependent_meta_eight.py"),init_globals={
            "DEPENDENT_META_RESULT_DIRECTORY":str(output_directory),"display":lambda value:None})
    proof = receipt(stdout.getvalue())
    expected = {name:digest(source_directory/(name+".json")) for name in RESULT_NAMES}
    actual = {name:digest(output_directory/(name+".json")) for name in RESULT_NAMES}
    if expected!=actual or actual!=proof["artifact_sha256"] or not proof["complete_artifacts_equal"] or not proof["all_latex"]:
        raise RuntimeError("The installed wheel changed a complete eight-result artifact or its restoration")
    if any(name.split(".")[0] in ("scipy","statsmodels") for name in sys.modules):
        raise RuntimeError("A runtime estimator oracle was imported")
    return {"status":"passed","source_revision":source_revision,
            "verification_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "verification_worktree_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True).splitlines(),
            "wheel_sha256":digest(wheel),"charts_wheel_sha256":digest(charts_wheel),"sdist_sha256":digest(sdist),
            "source_wheel_sdist_installed_bytes_equal":wheel_records,"complete_meta_family_modules":len(source_family),
            "complete_source_installed_artifacts_equal":actual,"proof":proof,
            "source_path_injected":False,"scipy_statsmodels_imported":False,
            "inherited_editable_source_paths_removed":inherited_sources,
            "development_estimators_unavailable":True,"primary_environment_changed":False,
            "native_ui_acceptance_claimed":False,"public_release_delivered":False,
            "installed_site":str(site),"runtime_verifier_sha256":digest(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("site","wheel","charts-wheel","sdist","source-artifacts","installed-artifacts","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--source-revision",required=True)
    args = parser.parse_args()
    args.source_artifacts.mkdir(parents=True,exist_ok=True)
    args.installed_artifacts.mkdir(parents=True,exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT/"src")+os.pathsep+str(ROOT/"packages/openecon-charts/src")
    code = "import runpy\nrunpy.run_path("+repr(str(ROOT/"docs/examples/dependent_meta_eight.py"))+",init_globals="+repr({
        "DEPENDENT_META_RESULT_DIRECTORY":str(args.source_artifacts.resolve())})+" | {'display': lambda value: None})\n"
    source = subprocess.run([sys.executable,"-c",code],cwd=ROOT,env=environment,capture_output=True,text=True,timeout=180,check=True)
    source_proof = receipt(source.stdout)
    if source_proof["frozen"]:
        raise RuntimeError("Expected independent source execution")
    record = verify(args.site,args.wheel,args.charts_wheel,args.sdist,args.source_artifacts,args.installed_artifacts,args.source_revision)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":record["status"],"full_module_count":len(record["source_wheel_sdist_installed_bytes_equal"]),
                      "complete_artifact_count":len(record["complete_source_installed_artifacts_equal"])}))


if __name__ == "__main__":
    main()
