"""The count family's documentation page runs as written and reports what it shows."""

import math
import pathlib
import re
import warnings

import openecon as oe

_TOKEN = re.compile(r"[^\s,;:|(){}\[\]=']+")


def _same_output(printed, documented):
    """Token-wise equality; numbers agree to 4 significant digits (platform rounding)."""
    first, second = _TOKEN.findall(printed), _TOKEN.findall(documented)
    if len(first) != len(second):
        return False
    for a, b in zip(first, second, strict=True):
        try:
            if not math.isclose(float(a), float(b), rel_tol=1e-4, abs_tol=1e-12):
                return False
        except ValueError:
            if a != b:
                return False
    return True


def test_documentation_examples_run_and_report_what_the_page_says(capsys):
    page = pathlib.Path(__file__).resolve().parents[1] / "docs" / "econometrics" / "count.md"
    text = page.read_text(encoding="utf-8")
    fences = re.findall(r"```(python|text)\n(.*?)```", text, flags=re.DOTALL)
    assert sum(kind == "python" for kind, _ in fences) >= 11
    namespace: dict = {}
    compared = 0
    capsys.readouterr()
    for position, (kind, block) in enumerate(fences):
        if kind != "python":
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(block, str(page), "exec"), namespace)     # noqa: S102
        printed = capsys.readouterr().out
        following = fences[position + 1] if position + 1 < len(fences) else ("", "")
        if following[0] == "text":                 # the output shown on the page is the output
            assert _same_output(printed, following[1]), (printed, following[1])
            compared += 1
        else:
            assert printed == ""
    assert compared >= 10
    # Every estimator of the family is documented under its own heading.
    for name in ("zip", "zinb", "tpoisson", "tnbreg", "churdle", "hurdle", "gnbreg"):
        assert f"## `oe.{name}`" in text, name
        assert callable(getattr(oe, name))
    for word in ("Uncertain conventions", "N/(N-1)", "G/(G-1)", "separation_detected",
                 "boundary_solution", "Deliberate differences from Stata", "Limitations",
                 "Vuong", "chibar2(01)"):
        assert word in text, word


def test_convenience_function_docstrings_describe_the_command():
    for name in ("zip", "zinb", "tpoisson", "tnbreg", "churdle", "hurdle", "gnbreg"):
        doc = getattr(oe, name).__doc__
        for section in ("Model", "Parameters", "Result", "Stata", "Example", "covariance",
                        "weights"):
            assert section in doc, (name, section)
