"""Adversarial release fixtures: immutable runtime, fresh receipts and safe archives."""

from __future__ import annotations

import base64
import copy
import csv
import io
import json
from pathlib import Path
import runpy
import stat
import tarfile
import tempfile
import unittest
import warnings
import zipfile


CHECKS = Path(__file__).resolve().parent
M = runpy.run_path(str(CHECKS / "verify_metadata_refresh_release.py"))
PAIR = runpy.run_path(str(CHECKS.parents[1] / "scripts/verify_public_package_pair.py"))


def encoded(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def write_zip(path, files):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)


def write_tar(path, prefix, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, value in files.items():
            row = tarfile.TarInfo(prefix + "/" + name)
            row.size = len(value)
            archive.addfile(row, io.BytesIO(value))


def xml(count):
    return ('<testsuites><testsuite tests="' + str(count)
            + '" failures="0" errors="0" skipped="0">'
            + ''.join(f'<testcase classname="acceptance" name="case-{n}"/>'
                      for n in range(count)) + '</testsuite></testsuites>').encode()


def project(name, version):
    dependency = ('dependencies = ["openecon-charts=='
                  + ("0.3.2a1" if version.endswith("a1") else "0.3.2a2") + '"]\n'
                  if name == "openecon" else 'dependencies = []\n')
    return (f'[project]\nname = "{name}"\nversion = "{version}"\n'
            'readme = "PYPI-README.md"\nrequires-python = ">=3.11,<3.15"\n'
            'license = "Apache-2.0"\nlicense-files = ["LICENSE", "NOTICE", "THIRD_PARTY.md"]\n'
            + dependency + '\n[build-system]\nrequires = ["hatchling>=1.26"]\n'
            'build-backend = "hatchling.build"\n').encode()


def wheel(name, version, production, readme):
    info = name.replace("-", "_") + "-" + version + ".dist-info/"
    files = dict(production)
    metadata = (f'Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n'
                'License-Expression: Apache-2.0\nRequires-Python: >=3.11,<3.15\n'
                'Description-Content-Type: text/markdown\n')
    if name == "openecon":
        metadata += ('Requires-Dist: openecon-charts=='
                     + ("0.3.2a1" if version.endswith("a1") else "0.3.2a2") + '\n')
    files[info + "METADATA"] = metadata.encode() + b"\n" + readme
    files[info + "WHEEL"] = (b"Wheel-Version: 1.0\nGenerator: independent-fixture\n"
                              b"Root-Is-Purelib: true\nTag: py3-none-any\n")
    for notice in ("LICENSE", "NOTICE", "THIRD_PARTY.md"):
        files[info + "licenses/" + notice] = (notice + " preserved\n").encode()
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for path, value in sorted(files.items()):
        digest = base64.urlsafe_b64encode(bytes.fromhex(M["sha"](value))).decode().rstrip("=")
        writer.writerow((path, "sha256=" + digest, len(value)))
    writer.writerow((info + "RECORD", "", ""))
    files[info + "RECORD"] = record.getvalue().encode()
    return files


def ci(validation):
    run = {
        "id": validation["run_id"], "run_attempt": validation["run_attempt"],
        "repository": {"full_name": M["REPOSITORY"]},
        "head_repository": {"full_name": M["REPOSITORY"]},
        "head_sha": validation["api_head_sha"], "event": validation["event"],
        "path": M["WORKFLOW"], "status": "completed", "conclusion": "success",
    }
    build_steps = ["Bind metadata refresh to the published runtime source",
                   "Build source-pinned frontend and exactly four distributions",
                   "Record immutable build execution and retain its exact bytes"]
    package_steps = ["Independently verify the same metadata-only runtime identity",
                     "Verify the same four distributions against source and build execution",
                     "Select provisioned Chrome with its sandbox enabled",
                     "Install and verify dependency-free charts wheel and sdist",
                     "Install and test fresh SDK wheel and sdist environments",
                     "Run bounded metadata, parser and OLS guard regressions",
                     "Verify standalone wheel and sdist sandboxed Chrome exports"]
    jobs = {"total_count": 3, "jobs": [{
        "name": name, "run_id": run["id"], "head_sha": run["head_sha"],
        "status": "completed", "conclusion": "success",
        "steps": [{"name": step, "status": "completed", "conclusion": "success"}
                  for step in (build_steps if name == "Build one fixed SDK and charts pair"
                               else package_steps)],
    } for name in ("Build one fixed SDK and charts pair", "Verify SDK patch package (3.11)",
                   "Verify SDK patch package (3.13)")]}
    return run, jobs


def identity(validation, old_wheels):
    result = {
        "schema_version": 1, "status": "passed_metadata_only_runtime_identity",
        "baseline_repository": M["REPOSITORY"], "baseline_public_commit": M["PUBLIC_BASELINE"],
        "public_checkout_commit": validation["tested_checkout_commit"],
        "event": validation["event"], "run_id": str(validation["run_id"]),
        "run_attempt": str(validation["run_attempt"]),
        "scopes": ["src/openecon", "packages/openecon-charts/src/openecon_charts", "web"],
        "excluded_generated_directories": ["src/openecon/static", "web/node_modules", "web/dist"],
        "generated_frontend": "separately bound by verify_public_package_pair.py",
        "new_full_scientific_gate_claimed": False, "runtime_source_files": 1040,
        "unchanged_runtime_source_files": 1038,
        "baseline_inventory_sha256": M["RUNTIME_INVENTORY"],
        "normalized_current_inventory_sha256": M["RUNTIME_INVENTORY"], "version_only_files": [],
    }
    for source, path in sorted(M["SOURCE_VERSION_FILES"].items()):
        before, after = M["VERSION_FILES"][path]
        value = old_wheels[path]
        result["version_only_files"].append({
            "path": source, "before": before, "after": after, "before_sha256": M["sha"](value),
            "after_sha256": M["sha"](value.replace(before.encode(), after.encode(), 1)),
        })
    return result


def linux(build, execution, source_identity, provenance, descriptions, minor):
    files = {
        "pair-build/build-execution.json": encoded(execution),
        "pair-build/package-pair-provenance.json": provenance,
        "pair-build/source-identity.json": source_identity,
        "_temp/source-identity.json": source_identity,
        "_temp/pair-execution.json": encoded({
            "status": "passed", **execution, "artifacts": build["artifacts"],
            "generated_frontend": build["generated_frontend"], "same_four_distributions": True,
        }),
    }
    for name, count in M["SUITES"].items():
        files["_temp/" + name] = xml(count)
    for package, projects in (("sdk", set(M["VERSIONS"])), ("charts", {"openecon-charts"})):
        for mode in ("wheel", "sdist"):
            prefix = f"_temp/{package}-{mode}-"
            files[prefix + "installed.json"] = encoded({
                "status": "passed", "source_commit": M["SOURCE_COMMIT"],
                "source_files_sha256": build["source_files_sha256"],
                "provenance_sha256": M["sha"](provenance), "python": minor + ".14 fixture",
                "isolated": True, "workspace_paths_absent": True,
                "editable_installations_absent": True,
                "packages": {name: {
                    "version": M["VERSIONS"][name],
                    "production_files": build["production_files"][name],
                    "imported_modules": {name.replace("-", "_"):
                                         name.replace("-", "_") + "/__init__.py"},
                } for name in projects},
            })
            if package == "sdk":
                smoke = {"status": "passed", "sdk_version": "0.3.20a2",
                         "charts_version": "0.3.2a2", "registry_estimators": 171,
                         "registry_exports": 995, "referenced_modules": 440}
                for key in ("lazy_registry_no_heavy_imports", "ols_independent_algebra",
                            "hc3_independent_full_covariance", "guarded_parquet_after_torch",
                            "exact_fit_guard_and_real_noise", "poisson_score_check",
                            "json_restore_predictions_duplicate_index", "latex_and_chart_protocol"):
                    smoke[key] = True
                name = "openecon"
            else:
                smoke = {"status": "passed", "version": "0.3.2a2",
                         "standalone_no_heavy_dependencies": True, "offline_html_latex": True,
                         "embedded_woff2_and_ttf_fonts": True}
                name = "openecon-charts"
            files[prefix + "smoke.json"] = encoded(smoke)
            blocks = M["re"].findall(r"```python\r?\n(.*?)\r?\n```", descriptions[name].decode(),
                                    flags=M["re"].DOTALL)
            files[prefix + "readme.json"] = encoded({
                "status": "passed", "package": name, "python_blocks": len(blocks),
                "description_sha256": M["sha"](descriptions[name]),
                "code_sha256": [M["sha"](block.encode()) for block in blocks],
                "isolated_installed_examples": True, "offline": True,
            })
    return {"openeconometrics/openeconometrics/" + path if path.startswith("pair-build/") else path: value
            for path, value in files.items()}


class RefreshFixture:
    """Small genuine wheel RECORDs and sdist payloads, with independently authored receipts."""

    def __init__(self, root):
        self.directory, self.baseline = root / "refresh", root / "baseline"
        self.directory.mkdir()
        self.baseline.mkdir()
        self.validation = {"run_id": 12345, "run_attempt": 2, "api_head_sha": "a" * 40,
                           "tested_checkout_commit": "b" * 40, "event": "pull_request",
                           "workflow_path": M["WORKFLOW"]}
        self.run, self.jobs = ci(self.validation)
        inventories, metadata, descriptions, old_wheels = {}, {}, {}, {}
        sdist_trees = {"old": {}, "new": {}}
        wheel_trees = {"old": {}, "new": {}}
        for kind, versions in (("old", M["OLD_VERSIONS"]), ("new", M["VERSIONS"])):
            for name, version in versions.items():
                basename = name.replace("-", "_")
                production = {basename + "/__init__.py":
                              f'__version__ = "{version}"\nVALUE = 42\n'.encode()}
                if name == "openecon":
                    production.update({"openecon/static/index.html": b"<html></html>\n",
                                       "openecon/static/assets/main.js": b"console.log(1);\n",
                                       "openecon/static/assets/main.css": b"body{}\n"})
                description = f"# {name} {kind}\n\n```python\nprint(42)\n```\n".encode()
                files = wheel(name, version, production, description)
                wheel_trees[kind][name] = files
                sdist = {"src/" + path: value for path, value in production.items()}
                sdist.update({"pyproject.toml": project(name, version), "PYPI-README.md": description,
                              "PKG-INFO": files[f"{basename}-{version}.dist-info/METADATA"]})
                sdist.update({notice: (notice + " preserved\n").encode()
                              for notice in ("LICENSE", "NOTICE", "THIRD_PARTY.md")})
                sdist_trees[kind][name] = sdist
                if kind == "old":
                    old_wheels.update(files)
                else:
                    inventories[name] = {path: M["sha"](value)
                                         for path, value in sorted(production.items())}
                    descriptions[name] = description
                    definition = M["tomllib"].loads(sdist["pyproject.toml"].decode())["project"]
                    definition.update({"readme_bytes": description,
                                       "license_bytes": {key: sdist[key]
                                                         for key in definition["license-files"]},
                                       "charts_version": "0.3.2a2"})
                    metadata[name] = {
                        "wheel": PAIR["check_metadata"](files, definition, wheel=True),
                        "sdist": PAIR["check_metadata"](sdist, definition, wheel=False),
                    }
        for kind in ("old", "new"):
            sdk = sdist_trees[kind]["openecon"]
            sdk.update({"packages/openecon-charts/" + path: value
                        for path, value in sdist_trees[kind]["openecon-charts"].items()
                        if path != "PKG-INFO"})
            sdk["uv.lock"] = (b'version = "0.3.20a1"\ncharts = "0.3.2a1"\n'
                              if kind == "old" else b'version = "0.3.20a2"\ncharts = "0.3.2a2"\n')
            sdk["scripts/verify_public_package_pair.py"] = ("# " + kind + " package guard\n").encode()
            sdk["tests/test_public_package_pair_metadata.py"] = ("# " + kind + " guard tests\n").encode()
            source_ref = "c" * 40 if kind == "old" else M["CAPABILITY_METADATA_REF"]
            sdk_version = "0.3.20a1" if kind == "old" else "0.3.20a2"
            charts_version = "0.3.2a1" if kind == "old" else "0.3.2a2"
            sdk["docs/econometrics/capabilities.generated.json"] = encoded({
                "source_ref": source_ref, "versions": {"sdk": sdk_version, "charts": charts_version,
                                                        "native": "0.3.46"},
                "dependencies": {"runtime": ["torch>=2.6,<3", "openecon-charts==" + charts_version]},
                "source_fingerprints": {path: M["sha"](sdk[path]) for path in (
                    "pyproject.toml", "packages/openecon-charts/pyproject.toml", "src/openecon/__init__.py")},
                "estimators": {"ols": {"domain": "unchanged"}}, "stata_parity_validated": False,
            })
            sdk["docs/capabilities.md"] = (
                f"Source reference: `{source_ref}`.\nSDK `{sdk_version}` · charts `{charts_version}`.\n"
                f"Dependency: `openecon-charts=={charts_version}`.\nMethods/domains unchanged.\n").encode()
        old_rows = [{"path": path, "bytes": len(value), "sha256": M["sha"](value),
                     "original_sha256": M["sha"](value), "transformed": False}
                    for path, value in sorted(sdist_trees["old"]["openecon"].items())
                    if path != "PKG-INFO"]
        self.manifests = {}
        for kind in ("old", "new"):
            rows = copy.deepcopy(old_rows)
            for row in rows:
                value = sdist_trees[kind]["openecon"][row["path"]]
                row.update({"bytes": len(value), "sha256": M["sha"](value)})
                row["transformed"] = row["sha256"] != row["original_sha256"]
            manifest = {"source_commit": M["SOURCE_COMMIT"], "files": rows,
                        "source_files_sha256": M["sha"](json.dumps(rows, sort_keys=True).encode())}
            if kind == "new":
                previous, current = sdist_trees["old"]["openecon"], sdist_trees["new"]["openecon"]
                changes = [{"path": path, "baseline_sha256": M["sha"](previous[path]),
                            "refreshed_sha256": M["sha"](value),
                            "baseline_bytes": len(previous[path]), "refreshed_bytes": len(value)}
                           for path, value in sorted(current.items())
                           if path not in {"PKG-INFO", "SOURCE-MANIFEST.json"} and value != previous[path]]
                manifest["metadata_refresh"] = {
                    "schema_version": 1, "baseline_public_commit": M["PUBLIC_METADATA_BASELINE"],
                    "baseline_runtime_public_commit": M["PUBLIC_BASELINE"],
                    "inherited_private_source_commit": M["SOURCE_COMMIT"],
                    "previous_versions": {"sdk": "0.3.20a1", "charts": "0.3.2a1"},
                    "versions": {"sdk": "0.3.20a2", "charts": "0.3.2a2"},
                    "scope": "Metadata and exact version literals only.",
                    "runtime_identity_verifier": ".github/checks/verify_metadata_refresh_source.py",
                    "runtime_identity_receipt": "source-identity.json",
                    "baseline_runtime_inventory_sha256": M["RUNTIME_INVENTORY"],
                    "new_full_scientific_gate_claimed": False,
                    "registry_publication": "Separate protected publication required.",
                    "changed_files": changes, "generated_documents": "No recursive self-hash.",
                    "packaging_verification_transforms": [
                        {"path": "scripts/verify_public_package_pair.py", "reason": "Exact version guard."},
                        {"path": "tests/test_public_package_pair_metadata.py", "reason": "Adversarial cases."},
                    ],
                    "generated_capability_transforms": [
                        {"path": "docs/capabilities.md", "reason": "Refresh version labels only."},
                        {"path": "docs/econometrics/capabilities.generated.json",
                         "reason": "Refresh exact source/version/fingerprint metadata only."},
                    ],
                }
            self.manifests[kind] = manifest
            sdist_trees[kind]["openecon"]["SOURCE-MANIFEST.json"] = encoded(manifest)
            destination = self.baseline if kind == "old" else self.directory
            versions = M["OLD_VERSIONS"] if kind == "old" else M["VERSIONS"]
            for name, version in versions.items():
                prefix = name.replace("-", "_") + "-" + version
                write_tar(destination / (prefix + ".tar.gz"), prefix, sdist_trees[kind][name])
                write_zip(destination / (prefix + "-py3-none-any.whl"), wheel_trees[kind][name])
        generated = {"src/" + path: value for path, value in inventories["openecon"].items()
                     if path.startswith("openecon/static/")}
        self.build = {
            "status": "passed",
            "scope": "Frozen source and distribution byte verification; no publication or method parity claim.",
            "source_commit": M["SOURCE_COMMIT"],
            "source_manifest_sha256": M["sha"](encoded(self.manifests["new"])),
            "source_files_sha256": self.manifests["new"]["source_files_sha256"],
            "versions": M["VERSIONS"], "artifacts": self.hashes(self.directory),
            "production_files": inventories, "metadata": metadata, "generated_frontend": generated,
            "external_fixture_test_modules": [],
        }
        self.provenance = encoded(self.build)
        (self.directory / "PACKAGE-PAIR-PROVENANCE.json").write_bytes(self.provenance)
        (self.baseline / "PACKAGE-PAIR-PROVENANCE.json").write_bytes(encoded({
            "generated_frontend": generated, "external_fixture_test_modules": [],
        }))
        self.execution = {
            "public_checkout_commit": self.validation["tested_checkout_commit"],
            "private_source_commit": M["SOURCE_COMMIT"],
            "source_files_sha256": self.build["source_files_sha256"], "event": "pull_request",
            "run_id": "12345", "run_attempt": "2", "sdk_version": "0.3.20a2",
            "charts_version": "0.3.2a2",
        }
        self.identity = encoded(identity(self.validation, old_wheels))
        (self.directory / "BUILD-EXECUTION.json").write_bytes(encoded(self.execution))
        (self.directory / "SOURCE-IDENTITY.json").write_bytes(self.identity)
        self.descriptions = descriptions
        self.linux = {}
        for minor in ("3.11", "3.13"):
            files = linux(self.build, self.execution, self.identity, self.provenance, descriptions, minor)
            self.linux[minor] = files
            write_zip(self.directory / f"linux-package-{minor}-evidence.zip", files)
        self.historical = encoded({"schema_version": 4, "release": M["BASELINE_RELEASE"],
                                   "repository": M["REPOSITORY"],
                                   "assets": self.hashes(self.baseline)})
        self.profile = {"schema_version": 1, "repository": M["REPOSITORY"], "release": M["RELEASE"],
                        "versions": M["VERSIONS"], "validation": self.validation,
                        "baseline": {"release": M["BASELINE_RELEASE"],
                                     "profile_sha256": M["sha"](self.historical)},
                        "assets": self.hashes(self.directory)}

    @staticmethod
    def hashes(directory):
        return {path.name: {"bytes": path.stat().st_size, "sha256": M["sha"](path.read_bytes())}
                for path in directory.iterdir() if path.is_file()}

    def api_reader(self, url):
        return self.jobs if "/jobs?" in url else self.run

    def verify(self):
        return M["verify"](self.directory, self.baseline, self.profile, self.historical,
                           api_reader=self.api_reader, checker=PAIR)

    def update_linux(self, minor):
        path = self.directory / f"linux-package-{minor}-evidence.zip"
        write_zip(path, self.linux[minor])
        self.profile["assets"][path.name] = self.hashes(self.directory)[path.name]


class MetadataRefreshReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def fixture(self):
        return RefreshFixture(self.root)

    def test_full_valid_refresh_preserves_pr_head_and_merge_checkout_distinction(self):
        fixture = self.fixture()
        result = fixture.verify()
        self.assertEqual(result["status"], "passed_metadata_only_refresh")
        self.assertFalse(result["new_full_scientific_gate_claimed"])
        for name in M["VERSIONS"]:
            outputs = list((fixture.directory / name / "dist").iterdir())
            self.assertEqual(len(outputs), 2)
            for path in outputs:
                self.assertEqual(path.read_bytes(), (fixture.directory / path.name).read_bytes())

    def test_changed_baseline_is_rejected_before_outputs(self):
        fixture = self.fixture()
        path = fixture.baseline / "openecon-0.3.20a1-py3-none-any.whl"
        path.write_bytes(path.read_bytes() + b"foreign")
        with self.assertRaisesRegex(ValueError, "asset bytes differ"):
            fixture.verify()
        self.assertFalse((fixture.directory / "openecon").exists())

    def test_changed_download_is_rejected(self):
        fixture = self.fixture()
        (fixture.directory / "BUILD-EXECUTION.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "asset bytes differ"):
            fixture.verify()

    def test_rehashed_wheel_with_unreviewed_runtime_still_rejected(self):
        fixture = self.fixture()
        path = fixture.directory / "openecon-0.3.20a2-py3-none-any.whl"
        files = M["zip_files"](path)
        files["openecon/__init__.py"] += b"UNREVIEWED_RUNTIME = True\n"
        write_zip(path, files)
        fixture.profile["assets"][path.name] = fixture.hashes(fixture.directory)[path.name]
        with self.assertRaisesRegex(ValueError, "Runtime changed beyond exact version"):
            fixture.verify()
        self.assertFalse((fixture.directory / "openecon").exists())

    def test_rehashed_sdist_with_unreviewed_source_still_rejected(self):
        fixture = self.fixture()
        path = fixture.directory / "openecon-0.3.20a2.tar.gz"
        files = M["tar_files"](path, "openecon-0.3.20a2")
        files["src/openecon/foreign.py"] = b"UNREVIEWED_RUNTIME = True\n"
        write_tar(path, "openecon-0.3.20a2", files)
        fixture.profile["assets"][path.name] = fixture.hashes(fixture.directory)[path.name]
        with self.assertRaisesRegex(ValueError, "sdist source file inventory changed"):
            fixture.verify()

    def test_foreign_asset_refused(self):
        fixture = self.fixture()
        (fixture.directory / "FOREIGN").write_bytes(b"unknown")
        with self.assertRaisesRegex(ValueError, "foreign downloaded"):
            fixture.verify()

    def test_profile_foreign_release_and_unbounded_assets_refused(self):
        fixture = self.fixture()
        mutations = [lambda p: p.update(release="latest"),
                     lambda p: p.update(repository="foreign/repo"),
                     lambda p: p.update(extra=True),
                     lambda p: p["assets"].update({"../../foreign": {"bytes": 1, "sha256": "a"*64}}),
                     lambda p: p["assets"]["BUILD-EXECUTION.json"].update(bytes=True),
                     lambda p: p["assets"]["BUILD-EXECUTION.json"].update(bytes=M["MAX_ASSET"]),
                     lambda p: p["assets"]["BUILD-EXECUTION.json"].update(sha256="A"*64),
                     lambda p: p["baseline"].update(profile_sha256="b"*64)]
        for mutate in mutations:
            profile = copy.deepcopy(fixture.profile)
            mutate(profile)
            with self.subTest(profile=profile["release"]):
                with self.assertRaises(ValueError):
                    M["check_profile"](profile, fixture.historical)

    def test_push_cannot_use_unrelated_checkout(self):
        fixture = self.fixture()
        fixture.profile["validation"]["event"] = "push"
        with self.assertRaisesRegex(ValueError, "Push checkout"):
            M["check_profile"](fixture.profile, fixture.historical)

    def test_failed_stale_foreign_or_incomplete_ci_refused(self):
        fixture = self.fixture()
        for key, value in (("conclusion", "failure"), ("status", "in_progress"),
                           ("run_attempt", 1), ("head_sha", "c"*40),
                           ("path", ".github/workflows/foreign.yml")):
            run = copy.deepcopy(fixture.run)
            run[key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "run identity/conclusion"):
                    M["check_ci"](fixture.validation, lambda url: run)

    def test_missing_or_failed_minor_job_refused(self):
        fixture = self.fixture()
        for operation in (lambda jobs: jobs["jobs"].pop(),
                          lambda jobs: jobs["jobs"][2].update(name="Verify SDK patch package (3.11)"),
                          lambda jobs: jobs["jobs"][2].update(conclusion="skipped"),
                          lambda jobs: jobs["jobs"][2]["steps"][0].update(conclusion="failure"),
                          lambda jobs: jobs["jobs"][2]["steps"][0].update(conclusion="skipped"),
                          lambda jobs: jobs["jobs"][2]["steps"].pop()):
            jobs = copy.deepcopy(fixture.jobs)
            operation(jobs)
            with self.assertRaises(ValueError):
                M["check_ci"](fixture.validation,
                              lambda url: jobs if "/jobs?" in url else fixture.run)

    def test_zip_duplicate_traversal_symlink_and_size_refused(self):
        path = self.root / "archive.zip"
        mutations = ["duplicate", "../escape", "/absolute", "a\\b", "symlink", "too-large"]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with zipfile.ZipFile(path, "w") as archive:
                        if mutation == "duplicate":
                            archive.writestr("a", b"first")
                            archive.writestr("a", b"second")
                        elif mutation == "symlink":
                            row = zipfile.ZipInfo("link")
                            row.external_attr = (stat.S_IFLNK | 0o777) << 16
                            archive.writestr(row, "target")
                        elif mutation == "too-large":
                            archive.writestr("a", b"123456")
                        else:
                            archive.writestr(mutation, b"bad")
                with self.assertRaises(ValueError):
                    M["zip_files"](path, limit=5 if mutation == "too-large" else M["MAX_ARCHIVE"])

    def test_tar_symlink_hardlink_foreign_root_duplicate_and_traversal_refused(self):
        path = self.root / "archive.tar.gz"
        for mutation in ("symlink", "hardlink", "foreign", "duplicate", "traversal"):
            with self.subTest(mutation=mutation):
                with tarfile.open(path, "w:gz") as archive:
                    row = tarfile.TarInfo("package/file")
                    if mutation == "symlink":
                        row.type, row.linkname = tarfile.SYMTYPE, "target"
                    elif mutation == "hardlink":
                        row.type, row.linkname = tarfile.LNKTYPE, "target"
                    elif mutation == "foreign":
                        row.name = "other/file"
                    elif mutation == "traversal":
                        row.name = "package/../file"
                    archive.addfile(row)
                    if mutation == "duplicate":
                        archive.addfile(row)
                with self.assertRaises(ValueError):
                    M["tar_files"](path, "package")

    def test_linked_download_or_linked_parent_refused(self):
        target = self.root / "real"
        target.mkdir()
        (target / "file").write_bytes(b"bytes")
        linked = self.root / "linked"
        linked.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            M["regular"](linked / "file")

    def test_two_exact_version_literals_are_allowed_but_runtime_changes_are_not(self):
        for path, (before, after) in M["VERSION_FILES"].items():
            old = {path: f'__version__ = "{before}"\nVALUE = 1\n'.encode()}
            good = {path: old[path].replace(before.encode(), after.encode(), 1)}
            M["compare_runtime"](old, good, path.split("/")[0] + "/")
            for bad in ({path: good[path] + b"VALUE = 2\n"},
                        {path: good[path], path.rsplit("/", 1)[0]+"/new.py": b"injection"},
                        {path: old[path]}):
                with self.assertRaises(ValueError):
                    M["compare_runtime"](old, bad, path.split("/")[0] + "/")

    def test_static_payload_is_not_metadata(self):
        old = {"openecon/static/assets/app.js": b"reviewed"}
        with self.assertRaisesRegex(ValueError, "Runtime changed"):
            M["compare_runtime"](old, {next(iter(old)): b"changed"}, "openecon/")

    def test_dependency_build_backend_and_extra_changes_refused(self):
        before = project("openecon", "0.3.20a1")
        good = project("openecon", "0.3.20a2")
        M["compare_project"](before, good, "openecon")
        for bad in (good.replace(b"openecon-charts==0.3.2a2", b"openecon-charts>=0.3.2a2"),
                    good.replace(b"hatchling>=1.26", b"foreign-build==1"),
                    good + b'\n[project.optional-dependencies]\nforeign = ["foreign"]\n'):
            with self.assertRaisesRegex(ValueError, "configuration"):
                M["compare_project"](before, bad, "openecon")

    def test_public_metadata_headers_only_allow_version_and_exact_charts_pin(self):
        old = (b"Name: openecon\nVersion: 0.3.20a1\nRequires-Dist: openecon-charts==0.3.2a1\n"
               b"Project-URL: Homepage, https://openeconometrics.com\nLicense-Expression: Apache-2.0\n\nOld\n")
        good = old.replace(b"0.3.20a1", b"0.3.20a2").replace(b"0.3.2a1", b"0.3.2a2")
        M["compare_metadata_headers"](old, good, "openecon")
        for bad in (good.replace(b"openeconometrics.com", b"foreign.invalid"),
                    good.replace(b"Apache-2.0", b"MIT"),
                    b"Summary: Unreviewed change\n" + good):
            with self.assertRaisesRegex(ValueError, "Metadata headers changed"):
                M["compare_metadata_headers"](old, bad, "openecon")

    def test_wheel_generator_change_requires_explicit_review(self):
        fixture = self.fixture()
        path = fixture.directory / "openecon-0.3.20a2-py3-none-any.whl"
        files = M["zip_files"](path)
        control = "openecon-0.3.20a2.dist-info/WHEEL"
        files[control] = files[control].replace(b"independent-fixture", b"different-generator")
        write_zip(path, files)
        fixture.profile["assets"][path.name] = fixture.hashes(fixture.directory)[path.name]
        with self.assertRaisesRegex(ValueError, "exact Generator/header difference"):
            fixture.verify()

    def test_sdist_undeclared_source_and_lock_dependency_changes_refused(self):
        old = {"PYPI-README.md": b"old", "other.py": b"constant", "uv.lock": b'v = "0.3.20a1"\n'}
        good = {**old, "PYPI-README.md": b"new", "uv.lock": b'v = "0.3.20a2"\n'}
        M["compare_sdists"](old, good, charts=False)
        for bad in ({**good, "other.py": b"changed"}, {**good, "new.py": b"new"},
                    {**good, "uv.lock": good["uv.lock"] + b'foreign = "1"\n'}):
            with self.assertRaises(ValueError):
                M["compare_sdists"](old, bad, charts=False)

    def test_xml_skipped_failed_missing_duplicate_or_false_summary_refused(self):
        good = xml(2)
        M["check_xml"](good, 2)
        for bad in (good.replace(b"<testcase", b"<skipped/><testcase", 1),
                    good.replace(b"<testcase", b"<failure/><testcase", 1), xml(1),
                    good.replace(b"case-1", b"case-0"),
                    good.replace(b'failures="0"', b'failures="1"'),
                    b'<!DOCTYPE a [<!ENTITY x "value">]>' + good):
            with self.assertRaises(ValueError):
                M["check_xml"](bad, 2)

    def test_wrong_python_minor_cannot_relabel_one_job_as_both(self):
        fixture = self.fixture()
        row = json.loads(fixture.linux["3.13"]["_temp/sdk-wheel-installed.json"])
        row["python"] = "3.11.14"
        fixture.linux["3.13"]["_temp/sdk-wheel-installed.json"] = encoded(row)
        fixture.update_linux("3.13")
        with self.assertRaisesRegex(ValueError, "identity/isolation"):
            fixture.verify()
        self.assertFalse((fixture.directory / "openecon").exists())

    def test_readme_receipt_must_bind_exact_description_and_offline_code(self):
        fixture = self.fixture()
        for key, bad in (("description_sha256", "f" * 64), ("offline", False),
                         ("offline", 1), ("code_sha256", []), ("isolated_installed_examples", False)):
            files = dict(fixture.linux["3.11"])
            receipt = json.loads(files["_temp/charts-sdist-readme.json"])
            receipt[key] = bad
            files["_temp/charts-sdist-readme.json"] = encoded(receipt)
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "description/offline"):
                    M["check_linux"](files, "3.11", fixture.build, fixture.execution,
                                     fixture.identity, fixture.provenance, fixture.descriptions)

    def test_execution_inventory_and_source_identity_cannot_be_substituted(self):
        fixture = self.fixture()
        for filename in ("_temp/pair-execution.json", "_temp/source-identity.json",
                         "openeconometrics/openeconometrics/pair-build/package-pair-provenance.json"):
            files = dict(fixture.linux["3.11"])
            receipt = json.loads(files[filename])
            receipt["foreign"] = True
            files[filename] = encoded(receipt)
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    M["check_linux"](files, "3.11", fixture.build, fixture.execution,
                                     fixture.identity, fixture.provenance, fixture.descriptions)

    def test_linux_original_zip_cannot_be_flattened_or_given_foreign_workspace(self):
        fixture = self.fixture()
        original = fixture.linux["3.11"]
        for prefix in ("", "foreign/repository/"):
            files = {name.replace("openeconometrics/openeconometrics/", prefix, 1): value
                     for name, value in original.items()}
            with self.assertRaisesRegex(ValueError, "Original Linux artifact workspace"):
                M["check_linux"](files, "3.11", fixture.build, fixture.execution,
                                 fixture.identity, fixture.provenance, fixture.descriptions)

    def test_installed_runtime_or_smoke_claim_cannot_be_changed(self):
        fixture = self.fixture()
        for filename, change in (
            ("_temp/sdk-wheel-installed.json", lambda r: r.update(provenance_sha256="f"*64)),
            ("_temp/sdk-wheel-installed.json", lambda r: r.update(isolated=False)),
            ("_temp/sdk-wheel-installed.json", lambda r:
             r["packages"]["openecon"]["production_files"].update({"foreign.py": "f"*64})),
            ("_temp/sdk-wheel-smoke.json", lambda r: r.update(sdk_version="0.3.21")),
            ("_temp/sdk-wheel-smoke.json", lambda r: r.update(registry_estimators=208)),
        ):
            files = dict(fixture.linux["3.11"])
            receipt = json.loads(files[filename])
            change(receipt)
            files[filename] = encoded(receipt)
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    M["check_linux"](files, "3.11", fixture.build, fixture.execution,
                                     fixture.identity, fixture.provenance, fixture.descriptions)

    def test_identity_cannot_claim_new_full_scientific_gate(self):
        fixture = self.fixture()
        old = {path: f'__version__ = "{before}"\nVALUE = 42\n'.encode()
               for path, (before, _) in M["VERSION_FILES"].items()}
        for key, bad in (("new_full_scientific_gate_claimed", True),
                         ("new_full_scientific_gate_claimed", 0), ("schema_version", True)):
            value = json.loads(fixture.identity)
            value[key] = bad
            with self.subTest(key=key, bad=bad):
                with self.assertRaisesRegex(ValueError, "source identity"):
                    M["check_identity"](value, fixture.validation, old)

    def test_manifest_cannot_rewrite_scientific_claim_or_changed_file_audit(self):
        fixture = self.fixture()
        old = M["tar_files"](fixture.baseline / "openecon-0.3.20a1.tar.gz", "openecon-0.3.20a1")
        source = M["tar_files"](fixture.directory / "openecon-0.3.20a2.tar.gz", "openecon-0.3.20a2")
        for change in (lambda m: m.update(source_commit="c" * 40),
                       lambda m: m.update(scientific_gate="new complete pass"),
                       lambda m: m["metadata_refresh"].update(new_full_scientific_gate_claimed=True),
                       lambda m: m["metadata_refresh"]["changed_files"].pop()):
            current = copy.deepcopy(fixture.manifests["new"])
            change(current)
            with self.assertRaises(ValueError):
                M["check_manifest"](old["SOURCE-MANIFEST.json"], encoded(current), source, old)

    def test_capability_docs_cannot_broaden_methods_domains_or_validation_claims(self):
        fixture = self.fixture()
        old = M["tar_files"](fixture.baseline / "openecon-0.3.20a1.tar.gz", "openecon-0.3.20a1")
        source = M["tar_files"](fixture.directory / "openecon-0.3.20a2.tar.gz", "openecon-0.3.20a2")
        M["compare_capability_docs"](old, source)
        path = "docs/econometrics/capabilities.generated.json"
        for change in (lambda value: value["estimators"]["ols"].update(domain="all options"),
                       lambda value: value.update(stata_parity_validated=True),
                       lambda value: value["source_fingerprints"].update({"foreign.py": "a" * 64}),
                       lambda value: value.update(source_ref="f" * 40)):
            updated = dict(source)
            value = json.loads(source[path])
            change(value)
            updated[path] = encoded(value)
            with self.assertRaisesRegex(ValueError, "Capability methods/domains"):
                M["compare_capability_docs"](old, updated)
        with self.assertRaisesRegex(ValueError, "Capability Markdown"):
            M["compare_capability_docs"](old, {**source, "docs/capabilities.md":
                                              source["docs/capabilities.md"] + b"Full vendor parity.\n"})

    def test_duplicate_json_fields_refused(self):
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            M["json_bytes"](b'{"status":"passed","status":"failed"}')


if __name__ == "__main__":
    unittest.main()
