"""Adversarial guards for a metadata refresh that must not change runtime code."""

import copy
from pathlib import Path
import runpy
import tempfile
import unittest


CHECK = runpy.run_path(str(Path(__file__).with_name("verify_metadata_refresh_source.py")))
COMPARE = CHECK["compare_sources"]


class MetadataRefreshSourceTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "src/openecon/__init__.py": ("100644", b'__version__ = "0.3.20a1"\n'),
            "src/openecon/engine.py": ("100644", b"def fit(): return 42\n"),
            "packages/openecon-charts/src/openecon_charts/__init__.py":
                ("100644", b'__version__ = "0.3.2a1"\n'),
            "packages/openecon-charts/src/openecon_charts/chart.py":
                ("100644", b"def render(): return 'offline'\n"),
            "web/src/app.ts": ("100644", b"export const alpha = true;\n"),
        }
        self.current = copy.deepcopy(self.baseline)
        for path, (before, after) in CHECK["VERSION_FILES"].items():
            mode, data = self.current[path]
            self.current[path] = (mode, data.replace(before.encode(), after.encode()))

    def test_only_literal_versions_pass_with_identical_normalized_inventory(self):
        result = COMPARE(self.baseline, self.current)
        self.assertEqual(result["runtime_source_files"], 5)
        self.assertEqual(result["unchanged_runtime_source_files"], 3)
        self.assertEqual(result["baseline_inventory_sha256"],
                         result["normalized_current_inventory_sha256"])
        self.assertEqual(len(result["version_only_files"]), 2)

    def test_scientific_or_chart_algorithm_or_frontend_change_fails(self):
        for path in ("src/openecon/engine.py",
                     "packages/openecon-charts/src/openecon_charts/chart.py", "web/src/app.ts"):
            with self.subTest(path=path):
                current = copy.deepcopy(self.current)
                mode, data = current[path]
                current[path] = (mode, data + b"# unexpected change\n")
                with self.assertRaisesRegex(ValueError, "beyond version literal"):
                    COMPARE(self.baseline, current)

    def test_added_runtime_file_fails(self):
        self.current["src/openecon/new_method.py"] = ("100644", b"pass\n")
        with self.assertRaisesRegex(ValueError, "added=.*new_method"):
            COMPARE(self.baseline, self.current)

    def test_removed_runtime_file_fails(self):
        del self.current["web/src/app.ts"]
        with self.assertRaisesRegex(ValueError, "removed=.*app.ts"):
            COMPARE(self.baseline, self.current)

    def test_executable_mode_change_fails(self):
        path = "src/openecon/engine.py"
        self.current[path] = ("100755", self.current[path][1])
        with self.assertRaisesRegex(ValueError, "file mode differs"):
            COMPARE(self.baseline, self.current)

    def test_injection_next_to_allowed_version_fails(self):
        path = "src/openecon/__init__.py"
        self.current[path] = ("100644", self.current[path][1] + b"import injected\n")
        with self.assertRaisesRegex(ValueError, "beyond version literal"):
            COMPARE(self.baseline, self.current)

    def test_unchanged_or_different_target_version_fails(self):
        path = "src/openecon/__init__.py"
        for version in (b"0.3.20a1", b"0.3.21a1"):
            with self.subTest(version=version):
                current = copy.deepcopy(self.current)
                current[path] = ("100644", b'__version__ = "' + version + b'"\n')
                with self.assertRaisesRegex(ValueError, "beyond version literal"):
                    COMPARE(self.baseline, current)

    def test_unexpected_baseline_literal_fails(self):
        path = "src/openecon/__init__.py"
        self.baseline[path] = ("100644", b'__version__ = "0.3.19a1"\n')
        with self.assertRaisesRegex(ValueError, "Published version literal differs"):
            COMPARE(self.baseline, self.current)

    def test_missing_baseline_version_file_fails(self):
        del self.baseline["src/openecon/__init__.py"]
        with self.assertRaisesRegex(ValueError, "version files are missing"):
            COMPARE(self.baseline, self.current)

    def test_another_baseline_commit_is_rejected_before_git_execution(self):
        with self.assertRaisesRegex(ValueError, "baseline pin differs"):
            CHECK["_read_baseline"](Path("/nonexistent"), "0" * 40)

    def test_source_reader_rejects_file_and_directory_symlinks(self):
        for directory_link in (False, True):
            with self.subTest(directory_link=directory_link), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                for scope in CHECK["SCOPES"]:
                    (root / scope).mkdir(parents=True)
                target = root / "outside"
                if directory_link:
                    target.mkdir()
                else:
                    target.write_text("unexpected runtime\n")
                (root / "src/openecon/linked").symlink_to(target, target_is_directory=directory_link)
                with self.assertRaisesRegex(ValueError, "runtime (directory|source)"):
                    CHECK["_read_current"](root)

    def test_generated_frontend_is_separate_but_new_source_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for scope in CHECK["SCOPES"]:
                (root / scope).mkdir(parents=True)
            generated = root / "src/openecon/static/built.js"
            generated.parent.mkdir()
            generated.write_text("build output\n")
            added = root / "src/openecon/hidden_new_method.py"
            added.write_text("pass\n")
            actual = CHECK["_read_current"](root)
            self.assertNotIn("src/openecon/static/built.js", actual)
            self.assertIn("src/openecon/hidden_new_method.py", actual)


if __name__ == "__main__":
    unittest.main()
