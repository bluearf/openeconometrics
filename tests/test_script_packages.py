"""Code-driven package syntax and active-project context, without installation."""
from __future__ import annotations

import ast
import threading

import pytest

from openecon import script_packages as scripts
from openecon.project_packages import PackageError


@pytest.mark.parametrize("container", [tuple, list])
def test_plain_names_and_exact_pins_are_canonicalized(container):
    requested = scripts.parse_requirements(container([
        "Humanize", "Demo___Pkg.Name==1!2.3RC1.post2.dev3+CPU", "polars==1.44.2",
    ]))
    assert requested == [
        {"name": "humanize", "version": None},
        {"name": "demo-pkg-name", "version": "1!2.3rc1.post2.dev3+cpu"},
        {"name": "polars", "version": "1.44.2"},
    ]


@pytest.mark.parametrize("requirement", [
    "", "demo @ https://example.test/a.whl",
    "https://example.test/a.whl", "git+https://example.test/repo", "../demo", "./demo",
    "/tmp/demo", "--index-url", "--target=/tmp", "demo;echo", "demo|echo", "$(echo)",
    "`echo`", "demo\nother", " demo", "demo ",
    "demo==", "demo==>=1", "demo==1==2", "demo==https://example.test/a", "demo==1 ",
    None, True, 12, {"name": "demo"},
])
def test_unsafe_or_nonexact_requirements_are_rejected_before_callback(requirement):
    called = []
    with scripts._use_installer(called.append), pytest.raises(PackageError) as error:
        scripts.install(requirement)
    assert error.value.code in {"INVALID_PACKAGE", "INVALID_VERSION", "UNSUPPORTED_PACKAGE_SOURCE"}
    assert called == []


@pytest.mark.parametrize("requirements", [
    (), [], None, True, 12, "demo", {"demo": "1.0"}, {"demo"}, iter(["demo"]),
    ["demo"] * 2, ["Demo___Pkg", "demo-pkg==1.0"],
    [f"demo{index}" for index in range(101)],
])
def test_requirement_container_duplicates_and_count_are_bounded(requirements):
    with pytest.raises(PackageError) as error:
        scripts.parse_requirements(requirements)
    assert error.value.code == "INVALID_PACKAGE"


def test_one_hundred_unique_requirements_are_supported():
    result = scripts.parse_requirements([f"demo{index}==1.2.3" for index in range(100)])
    assert len(result) == 100
    assert len({row["name"] for row in result}) == 100


@pytest.mark.parametrize("code", [
    'message = "%pip install demo"\n',
    "# %pip install demo\nvalue = 1\n",
    'message = """first\n%pip install demo\nlast"""\n',
    "message = '''first\n\t%pip install demo\nlast'''\n",
    'message = r"""first\n%pip install demo\nlast"""\n',
    'message = b"""first\n%pip install demo\nlast"""\n',
    'value = 2\nmessage = f"""first {value}\n%pip install demo\nlast"""\n',
    'value = 2\nmessage = f"""first\n%pip install demo\n{f"nested {value}"}"""\n',
    "value = 7\nvalue %= 3\n",
])
def test_python_strings_f_strings_comments_and_modulo_remain_unchanged(code):
    transformed = scripts.rewrite_install_commands(code)
    assert transformed == code
    ast.parse(transformed)


@pytest.mark.parametrize("prefix", ["%pip ", "%pip\t"])
def test_workspace_command_calls_the_same_safe_api_synchronously(prefix, capsys):
    calls = []
    code = ("before = 7\n" + prefix + "install Humanize polars==1.44.2 # a comment\n"
            "after = before + 1\n")

    def installer(requested):
        calls.append(requested)
        assert namespace["before"] == 7 and "after" not in namespace
        return {"installed": [{"name": "humanize", "version": "4.16.0"},
                              {"name": "polars", "version": "1.44.2"}]}

    namespace = {}
    transformed = scripts.rewrite_install_commands(code)
    with scripts._use_installer(installer):
        exec(compile(transformed, "<test-workspace>", "exec"), namespace)
    assert calls == [[{"name": "humanize", "version": None},
                      {"name": "polars", "version": "1.44.2"}]]
    assert namespace["after"] == 8
    assert "humanize==4.16.0" in capsys.readouterr().out


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("indent", ["    ", "\t"])
def test_conditional_install_preserves_indentation_and_execution(enabled, indent):
    calls = []
    code = f"if {enabled}:\n{indent}%pip install demo==1.2.3\nvalue = 8\n"

    def installer(requested):
        calls.append(requested)
        return {"installed": requested}

    transformed = scripts.rewrite_install_commands(code)
    assert transformed.splitlines()[1].startswith(indent)
    with scripts._use_installer(installer):
        namespace = {}
        exec(compile(transformed, "<test-condition>", "exec"), namespace)
    assert len(calls) == int(enabled)
    assert namespace["value"] == 8


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_rewrite_preserves_line_numbers_and_newline_convention(newline):
    code = newline.join(["value = 7", "%pip install demo", "raise RuntimeError('line three')", ""])
    transformed = scripts.rewrite_install_commands(code)
    assert transformed.count(newline) == code.count(newline)
    assert transformed.endswith(newline)
    tree = ast.parse(transformed)
    assert [node.lineno for node in tree.body] == [1, 2, 3]


@pytest.mark.parametrize("command", [
    "%pip install", "%pip uninstall demo", "%pip list",
    "%pip install --index-url https://user:secret@example.test demo", "%pip install ./demo",
    "%pip install demo; echo bad",
    "%pip install demo && echo bad", "%pip install demo | echo bad", "%pip install $(echo bad)",
    "%pip install 'unterminated", "%pip install Demo_Pkg demo-pkg", "%pip install demo==1==2",
    "%pip install demo \\\nother",
])
def test_workspace_rejects_shell_options_sources_duplicates_and_unsupported_actions(command):
    calls = []
    with scripts._use_installer(calls.append), pytest.raises(PackageError):
        scripts.rewrite_install_commands(command)
    assert calls == []


def test_command_rewrite_has_no_installer_side_effects():
    calls = []
    with scripts._use_installer(calls.append):
        transformed = scripts.rewrite_install_commands("%pip install demo\n")
    assert calls == []
    ast.parse(transformed)


@pytest.mark.parametrize("invalid_tail", ["if :\n", "def incomplete(\n", 'text = "unterminated\n'])
def test_entire_source_is_parsed_before_any_installation(invalid_tail):
    calls = []
    code = "%pip install demo\n" + invalid_tail
    with scripts._use_installer(calls.append), pytest.raises(SyntaxError):
        exec(compile(scripts.rewrite_install_commands(code), "<bad-workspace>", "exec"), {})
    assert calls == []


def test_install_requires_an_active_desktop_project():
    with pytest.raises(PackageError) as error:
        scripts.install("humanize")
    assert error.value.code == "PROJECT_INSTALL_UNAVAILABLE"


def test_install_returns_after_callback_and_preserves_existing_namespace():
    namespace = {"saved": object(), "events": []}
    saved = namespace["saved"]

    def installer(requested):
        namespace["events"].append("installed")
        return {"installed": [{"name": "demo", "version": "1.2.3"}]}

    with scripts._use_installer(installer):
        namespace["oe_install"] = scripts.install
        exec("returned = oe_install('demo')\nevents.append('continued')", namespace)
    assert namespace["returned"] is None
    assert namespace["saved"] is saved
    assert namespace["events"] == ["installed", "continued"]


def test_nested_context_restores_outer_installer_and_clears_after_exit():
    calls = []

    def outer(requested):
        calls.append("outer")
        return {"installed": [{"name": "demo", "version": "1"}]}

    def inner(requested):
        calls.append("inner")
        return {"installed": [{"name": "demo", "version": "2"}]}

    with scripts._use_installer(outer):
        scripts.install("demo")
        with scripts._use_installer(inner):
            scripts.install("demo")
        scripts.install("demo")
    assert calls == ["outer", "inner", "outer"]
    with pytest.raises(PackageError) as error:
        scripts.install("demo")
    assert error.value.code == "PROJECT_INSTALL_UNAVAILABLE"


def test_installer_error_is_preserved_and_context_is_reset():
    def failed(requested):
        raise PackageError("NO_WHEEL", "No wheel is available for this platform.")

    with pytest.raises(PackageError) as error, scripts._use_installer(failed):
        scripts.install("demo")
    assert error.value.code == "NO_WHEEL"
    with pytest.raises(PackageError) as unavailable:
        scripts.install("demo")
    assert unavailable.value.code == "PROJECT_INSTALL_UNAVAILABLE"


def test_installer_context_does_not_leak_to_other_threads():
    errors = []
    calls = []

    def other_thread():
        try:
            scripts.install("demo")
        except PackageError as error:
            errors.append(error.code)

    with scripts._use_installer(calls.append):
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == ["PROJECT_INSTALL_UNAVAILABLE"]
    assert calls == []
