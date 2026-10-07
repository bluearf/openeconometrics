"""Dependency-free LaTeX exports and structured mathematical previews.

Exports preserve every row, its index and the original column order. Text is
escaped by default; constructing ``Latex`` explicitly is the only raw-source
escape hatch. Unicode source is intended for XeLaTeX or LuaLaTeX.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, TextIO


class Latex(str):
    """An explicit LaTeX source value, also usable as an ordinary string."""

    def __new__(cls, source: str, math: str | None = None):
        instance = super().__new__(cls, source)
        instance.math = math
        return instance

    @property
    def source(self) -> str:
        return str(self)

    def _repr_latex_(self) -> str:
        return str(self)


@dataclass(frozen=True)
class _MathValue:
    """Trusted formatter-generated mathematics, never arbitrary cell text."""
    source: str


_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def escape_latex(value: Any) -> str:
    """Escape ordinary text without escaping the commands just introduced."""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return "".join(_ESCAPES.get(char, char) for char in text)


def _precision(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 18:
        raise ValueError("precision must be an integer between 0 and 18.")
    return value


def _scalar(value: Any) -> Any:
    # pandas exposes NumPy scalar cells; use their public Python conversion
    # without importing NumPy or altering integer precision.
    if type(value).__module__.split(".", 1)[0] == "numpy" and hasattr(value, "item"):
        return value.item()
    return value


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"} and type(value).__module__.startswith("pandas."):
        return True
    if isinstance(value, Decimal):
        return value.is_nan()
    return isinstance(value, Real) and not isinstance(value, Integral) and math.isnan(value)


def _value_text(value: Any, *, precision: int, float_format: str | Callable | None,
                na_rep: str) -> str:
    value = _scalar(value)
    if _missing(value):
        return na_rep
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(value)
    if isinstance(value, (Real, Decimal)):
        if float_format is not None:
            if callable(float_format):
                return str(float_format(value))
            if isinstance(float_format, str):
                return float_format % value
            raise TypeError("float_format must be a callable or a percent-format string.")
        if isinstance(value, Decimal):
            if value.is_finite() and value != 0 and abs(value) < Decimal(10) ** -precision:
                return format(value, f".{precision}e")
            return format(value, f".{precision}f")
        if math.isfinite(value) and value != 0 and abs(value) < 10 ** -precision:
            return format(value, f".{precision}e")
        return format(value, f".{precision}f")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _cell(value: Any, *, precision: int, float_format: str | Callable | None,
          na_rep: str, wrap_at: int | None = None) -> str:
    if isinstance(value, _MathValue):
        return "$" + value.source + "$"
    text = _value_text(value, precision=precision, float_format=float_format, na_rep=na_rep)
    numeric_math = _numeric_math(value, text, float_format=float_format)
    if numeric_math is not None:
        return "$" + numeric_math + "$"
    if wrap_at is not None:
        lines = []
        for line in text.split("\n"):
            parts = []
            for token in re.split(r"(\s+)", line):
                if not token.isspace() and len(token) > wrap_at and not isinstance(_scalar(value), (Real, Decimal)):
                    parts.append(r"\allowbreak{}".join(escape_latex(token[position:position + wrap_at])
                                                     for position in range(0, len(token), wrap_at)))
                else:
                    parts.append(escape_latex(token))
            lines.append("".join(parts))
        rendered = r"\newline{}".join(lines)
        return "{" + rendered + "}" if rendered.startswith("[") else rendered
    text = escape_latex(text)
    # Newlines must not insert an extra table row or TeX control sequence.
    if "\n" in text:
        return r"\begin{tabular}[t]{@{}l@{}}" + r" \\ ".join(text.split("\n")) + r"\end{tabular}"
    # A leading [ after the preceding row's \\ is otherwise parsed as the
    # optional row-spacing dimension (e.g. [outcome] is not a dimension).
    return "{" + text + "}" if text.startswith("[") else text


def _math_cell(value: Any, *, precision: int, float_format: str | Callable | None,
               na_rep: str) -> str:
    if isinstance(value, _MathValue):
        return value.source
    text = _value_text(value, precision=precision, float_format=float_format, na_rep=na_rep)
    numeric_math = _numeric_math(value, text, float_format=float_format)
    if numeric_math is not None:
        return numeric_math
    text = escape_latex(text)
    lines = text.split("\n")
    rendered = [r"\text{" + line + "}" for line in lines]
    return rendered[0] if len(rendered) == 1 else r"\begin{gathered}" + r" \\ ".join(rendered) + r"\end{gathered}"


def _numeric_math(value: Any, text: str, *, float_format: str | Callable | None) -> str | None:
    value = _scalar(value)
    if float_format is not None or isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        return None
    if _missing(value):
        return None
    scientific = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)", text)
    if scientific:
        mantissa, exponent = scientific.groups()
        return mantissa + r" \times 10^{" + str(int(exponent)) + "}"
    if text.lower() in {"inf", "infinity", "+inf", "+infinity"}:
        return r"\infty"
    if text.lower() in {"-inf", "-infinity"}:
        return r"-\infty"
    return None


def _is_dataframe(value: Any) -> bool:
    return any(cls.__name__ == "DataFrame" and cls.__module__.startswith("pandas.")
               for cls in type(value).__mro__)


def _is_series(value: Any) -> bool:
    return any(cls.__name__ == "Series" and cls.__module__.startswith("pandas.")
               for cls in type(value).__mro__)


def _is_result(value: Any) -> bool:
    return any(cls.__name__ == "ResultBundle" and cls.__module__ == "openecon.models"
               for cls in type(value).__mro__)


def _table_rows(frame: Any, *, index: bool, header: bool | Sequence[str],
                columns: Sequence | None = None) -> tuple[list[list[Any]], list[list[Any]], list[str]]:
    if columns is not None:
        frame = frame.loc[:, list(columns)]
    index_levels = frame.index.nlevels if index else 0
    names = list(frame.index.names) if index else []
    headers: list[list[Any]] = []
    if header is not False:
        if header is not True:
            if isinstance(header, (str, bytes)) or len(header) != len(frame.columns):
                raise ValueError("Header aliases must match the number of exported columns.")
            headers = [[*names, *header]]
        elif frame.columns.nlevels > 1:
            for level in range(frame.columns.nlevels):
                left = [None] * index_levels
                if left and frame.columns.names[level] is not None:
                    left[0] = frame.columns.names[level]
                headers.append([*left, *(label[level] for label in frame.columns)])
            if index and any(name is not None for name in names):
                headers.append([*names, *([None] * len(frame.columns))])
        else:
            headers = [[*names, *list(frame.columns)]]
    rows = []
    # pandas.itertuples(index=False) yields no tuples for a zero-column frame,
    # even when index rows exist. Preserve those rows explicitly.
    values_iterator = frame.itertuples(index=False, name=None) if len(frame.columns) else (() for _ in range(len(frame)))
    for position, values in enumerate(values_iterator):
        left = []
        if index:
            row_index = frame.index[position]
            left = list(row_index) if index_levels > 1 else [row_index]
        rows.append([*left, *values])
    alignment = ["l"] * index_levels
    # Data columns are right-aligned only when their actual dtype is numeric.
    for dtype in frame.dtypes:
        alignment.append("r" if getattr(dtype, "kind", None) in {"i", "u", "f", "c"} else "l")
    return headers, rows, alignment


def _column_format(value: str | None, alignment: list[str]) -> str:
    if value is None:
        return "".join(alignment) or "l"
    if not isinstance(value, str) or not re.fullmatch(r"[lcr| ]+", value):
        raise ValueError("column_format supports only l, c, r, spaces and vertical rules.")
    if sum(char in "lcr" for char in value) != len(alignment):
        raise ValueError("column_format must contain one alignment for each exported column.")
    return value


def _label(value: str | None) -> str | None:
    if value is None:
        return None
    # Labels are TeX identifiers rather than visible text. Escaping an arbitrary
    # command here changes the identifier and is not reliably compilable.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9:._/-]+", value):
        raise ValueError("label must contain only letters, numbers, :, ., _, / or -.")
    return value


def _caption(value: str | tuple[str, str] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        if len(value) != 2 or not all(isinstance(part, str) for part in value):
            raise TypeError("caption must be text or a (full, short) text pair.")
        full, short = value
        # A literal closing bracket cannot terminate the optional argument.
        return r"\caption[{" + escape_latex(short) + "}]{" + escape_latex(full) + "}"
    if not isinstance(value, str):
        raise TypeError("caption must be text or a (full, short) text pair.")
    return r"\caption{" + escape_latex(value) + "}"


def _layout(font_size: str, max_width: str | None) -> None:
    if font_size not in {"normalsize", "small", "footnotesize"}:
        raise ValueError("font_size must be normalsize, small or footnotesize.")
    if max_width is not None and (not isinstance(max_width, str) or not re.fullmatch(
            r"(?:(?:0(?:\.\d+)?|1(?:\.0+)?))?\\(?:line|text|column)width"
            r"|(?:[1-9][0-9]*(?:\.[0-9]+)?)(?:cm|mm|in|pt|em)", max_width)):
        raise ValueError("max_width must be a safe TeX width, such as \\linewidth or 0.9\\textwidth.")
    if max_width is not None and "\\" in max_width:
        fraction = max_width.split("\\", 1)[0]
        if fraction and float(fraction) <= 0:
            raise ValueError("max_width must be positive.")


def _notes(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and all(isinstance(note, str) for note in value):
        return list(value)
    raise TypeError("notes must be text or a sequence of text strings.")


def _tabular(headers: list[list[Any]], rows: list[list[Any]], alignment: list[str], *,
             precision: int, float_format: str | Callable | None, na_rep: str,
             caption: str | tuple[str, str] | None, label: str | None,
             column_format: str | None, longtable: bool, booktabs: bool,
             bold_rows: bool = False, index_levels: int = 0,
             notes: str | Sequence[str] | None = None, font_size: str = "small",
             max_width: str | None = r"\linewidth", rules_before: set[int] | None = None,
             spaces_after: set[int] | None = None,
             keep_with_next: set[int] | None = None) -> Latex:
    _layout(font_size, max_width)
    note_lines = _notes(notes)
    spec = _column_format(column_format, alignment)
    paragraph_columns = longtable and column_format is None and (len(alignment) >= 6 or any(
        len(_value_text(value, precision=precision, float_format=float_format, na_rep=na_rep)) > 40
        for row in [*headers, *rows] for value in row if not isinstance(value, _MathValue)))
    if paragraph_columns:
        # Fixed paragraph budgets preserve longtable page breaks while keeping
        # every column inside the available width. A boxed resize cannot do so.
        width = r"\dimexpr" + (max_width or r"\linewidth") + "/" + str(len(alignment)) + r"-2\tabcolsep\relax"
        commands = {"l": r"\raggedright", "c": r"\centering", "r": r"\raggedleft"}
        spec = "@{}" + "".join(
            ">{" + commands[align] + r"\arraybackslash}p{" + width + "}"
            for align in alignment) + "@{}"
    elif column_format is None:
        spec = "@{}" + spec + "@{}"
    caption_source = _caption(caption)
    label_value = _label(label)
    env = "longtable" if longtable else "tabular"
    lines = []
    wrapper = not longtable and (caption_source is not None or label_value is not None)
    if wrapper:
        lines.extend([r"\begin{table}[htbp]", r"\centering"])
        if caption_source is not None:
            lines.append(caption_source)
        elif label_value is not None:
            # A label needs a table counter, including when the caller chooses
            # not to show a caption.
            lines.append(r"\refstepcounter{table}")
        if label_value is not None:
            lines.append(r"\label{" + label_value + "}")
    lines.extend([r"\begingroup", "\\" + font_size,
                  r"\setlength{\tabcolsep}{" + (("4pt" if len(alignment) <= 12 else "1pt")
                                              if paragraph_columns else "6pt") + "}",
                  r"\renewcommand{\arraystretch}{1.12}"])
    if paragraph_columns:
        # Clamp padding to at most one quarter of each column's width even in
        # a narrow document column or an unusually wide user-selected table.
        # The remaining paragraph budget is therefore strictly positive.
        padding_budget = r"\dimexpr" + (max_width or r"\linewidth") + "/" + str(4 * len(alignment)) + r"\relax"
        lines.extend([r"\ifdim\tabcolsep>" + padding_budget,
                      r"\setlength{\tabcolsep}{" + padding_budget + "}", r"\fi"])
    if longtable:
        lines.extend([r"\setlength{\LTleft}{\fill}", r"\setlength{\LTright}{\fill}",
                      r"\setlength{\LTcapwidth}{\linewidth}"])
    elif max_width is not None:
        lines.append(r"\begin{adjustbox}{max width=" + max_width + "}")
    lines.append(r"\begin{" + env + "}{" + spec + "}")
    if longtable and (caption_source is not None or label_value is not None):
        if caption_source is not None:
            lines.append(caption_source)
        if label_value is not None:
            lines.append(r"\label{" + label_value + "}")
        lines.append(r"\\")
    top, middle, bottom = (r"\toprule", r"\midrule", r"\bottomrule") if booktabs else (r"\hline",) * 3
    lines.append(top)

    def render(row: list[Any], *, bold: bool = False) -> str:
        values = [_cell(value, precision=precision, float_format=float_format, na_rep=na_rep,
                        wrap_at=max(4, 48 // len(alignment)) if paragraph_columns else None)
                  for value in row]
        if bold:
            values[:index_levels] = [r"\textbf{" + value + "}" for value in values[:index_levels]]
        return " & ".join(values) + r" \\"

    lines.extend(render(row) for row in headers)
    if headers:
        lines.append(middle)
    if longtable:
        lines.append(r"\endfirsthead")
        lines.extend([top, *(render(row) for row in headers)])
        if headers:
            lines.append(middle)
        lines.extend([r"\endhead", bottom, r"\endfoot", bottom, r"\endlastfoot"])
    for position, row in enumerate(rows):
        if rules_before and position in rules_before:
            lines.append(middle)
        line = render(row, bold=bold_rows)
        if longtable and keep_with_next and position in keep_with_next:
            line += "*"
        if spaces_after and position in spaces_after:
            line += "[0.35em]"
        lines.append(line)
    if not longtable:
        lines.append(bottom)
    lines.append(r"\end{" + env + "}")
    if not longtable and max_width is not None:
        lines.append(r"\end{adjustbox}")
    if note_lines:
        lines.extend([r"\par\smallskip", r"\begin{minipage}{\linewidth}",
                      r"\footnotesize", r"\raggedright"])
        for position, note in enumerate(note_lines):
            prefix = r"\textit{Notes:} " if position == 0 else ""
            lines.append(prefix + escape_latex(note) + r"\par")
        lines.append(r"\end{minipage}")
    lines.append(r"\endgroup")
    if wrapper:
        lines.append(r"\end{table}")
    source = Latex("\n".join(lines))
    source.notes = note_lines
    return source


def _write(source: Latex, buf: str | Path | TextIO | None, *, encoding: str) -> Latex | None:
    if buf is None:
        return source
    if hasattr(buf, "write"):
        buf.write(str(source))
    else:
        Path(buf).write_text(str(source), encoding=encoding)
    return None


def _result_rows(value: Any) -> tuple[list[list[Any]], list[list[Any]], list[str]]:
    if value.inference.get("available") is False and value.extra.get("target")=="prediction":
        return _predictive_rows(value)
    use_t = value.inference.get("use_t", False)
    headers = [["Term", "Estimate", "Std. error", "t" if use_t else "z", "P>|stat|", "CI lower", "CI upper"]]
    rows = []
    grouped = any(c.equation is not None for c in value.coefficients)
    previous = object()
    for c in value.coefficients:
        if grouped and c.equation != previous:
            rows.append([f"[{c.equation}]" if c.equation is not None else "[Other parameters]", *([None] * 6)])
        previous = c.equation
        rows.append([c.term, c.estimate, c.std_error, c.statistic, c.p_value, c.ci_low, c.ci_high])
    return headers, rows, ["l", *("r" for _ in range(6))]


def _predictive_rows(value: Any):
    """Saved prediction state without importing a numerical runtime or inventing CI."""
    extra = value.extra
    if "penalized_state" in extra:
        state = extra["penalized_state"]
        rows = [["Constant",state["constant"]]] if value.spec.intercept else []
        rows.extend([[term,estimate] for term,estimate in zip(extra["terms"],state["coefficients"],strict=True)])
        return [["Predictive term","Estimate"]],rows,["l","r"]
    if "smoother_state" in extra:
        rows = [[*query,estimate] for query,estimate in zip(extra["query"],extra["query_estimates"],strict=True)]
        return [[*extra["terms"],"Conditional mean"]],rows,["r"]*(len(extra["terms"])+1)
    return [["Target"]],[["Prediction estimates unavailable"]],["l"]


def _number_atom(value: float, *, precision: int, float_format: str | Callable | None) -> str:
    text = _value_text(value, precision=precision, float_format=float_format, na_rep="")
    scientific = _numeric_math(value, text, float_format=float_format)
    if scientific is not None:
        return scientific
    if re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", text):
        return text
    return r"\text{" + escape_latex(text) + "}"


def _star(p_value: float) -> str:
    if not math.isfinite(p_value) or not 0 <= p_value <= 1:
        raise ValueError("Regression coefficient p-values must be between zero and one.")
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def _label_mapping(value: Mapping[str, str] | None, name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) and isinstance(label, str)
                                                 for key, label in value.items()):
        raise TypeError(f"{name} must map original text labels to display text labels.")
    return dict(value)


def _regression_models(models: Any) -> tuple[list[Any], list[str] | None]:
    if _is_result(models):
        values, labels = [models], None
    elif isinstance(models, Mapping):
        values, labels = list(models.values()), [str(name) for name in models]
    elif isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
        values, labels = list(models), None
    else:
        raise TypeError("regression_table requires fitted models, a model list or a named model mapping.")
    if not values or not all(_is_result(model) for model in values):
        raise ValueError("At least one fitted OpenEconometrics model is required.")
    return values, labels


def _regression_rows(models: Any, *, precision: int, stars: bool,
                     float_format: str | Callable | None,
                     term_labels: Mapping[str, str] | None,
                     outcome_labels: Mapping[str, str] | None
                     ) -> tuple[list[list[Any]], list[list[Any]], list[str], list[str], set[int], set[int], set[int]]:
    values, model_labels = _regression_models(models)
    terms_map = _label_mapping(term_labels, "term_labels")
    outcomes_map = _label_mapping(outcome_labels, "outcome_labels")
    if not isinstance(stars, bool):
        raise TypeError("stars must be True or False.")
    if len(values)>1 and any(model.inference.get("available") is False and model.extra.get("target")=="prediction" for model in values):
        raise ValueError("Prediction-only results require separate predictive tables; a multi-model inference table cannot silently omit predictive estimates.")
    if len(values)==1 and values[0].inference.get("available") is False and values[0].extra.get("target")=="prediction":
        headers,rows,alignment = _predictive_rows(values[0])
        from openecon.publication import model_notes
        return headers,rows,alignment,model_notes(values[0]),set(),set(),set()
    coefficients = []
    groups = {}
    for model in values:
        lookup = {(c.equation, c.term): c for c in model.coefficients}
        if len({c.term for c in model.coefficients}) != len(model.coefficients):
            raise ValueError("A regression table cannot silently overwrite duplicate coefficient terms.")
        coefficients.append(lookup)
        for key in lookup:
            if key not in groups.setdefault(key[0], []):
                groups[key[0]].append(key)
    grouped = any(equation is not None for equation in groups)
    constant_terms = set()
    if not grouped:
        constant_terms = {key for key in groups.get(None, []) if key[1] == "Intercept" and all(
            model.spec.intercept for model, lookup in zip(values, coefficients, strict=True) if key in lookup)}
        groups[None] = [key for key in groups.get(None, []) if key not in constant_terms] + list(constant_terms)
    terms = [key for group in groups.values() for key in group]
    headers = [["Dependent variable:", *(outcomes_map.get(model.spec.outcome, model.spec.outcome) for model in values)],
               ["", *(f"({number})" for number in range(1, len(values) + 1))]]
    if model_labels is not None:
        headers.append(["", *model_labels])
    rows = []
    spaces_after = set()
    group_headers = set()
    previous_equation = object()
    for key in terms:
        equation, term = key
        if grouped and equation != previous_equation:
            rows.append([f"[{equation}]" if equation is not None else "[Other parameters]", *([None] * len(values))])
            group_headers.add(len(rows) - 1)
        previous_equation = equation
        default_label = "Constant" if key in constant_terms else term
        estimates, errors = [terms_map.get(term, default_label)], [""]
        for lookup in coefficients:
            coefficient = lookup.get(key)
            if coefficient is None:
                estimates.append(None)
                errors.append(None)
                continue
            if coefficient.std_error < 0:
                raise ValueError("Regression standard errors must be nonnegative.")
            estimate = _number_atom(coefficient.estimate, precision=precision, float_format=float_format)
            significance = _star(coefficient.p_value) if stars else ""
            if significance:
                if "^" in estimate:
                    # Scientific notation already has an exponent: stars
                    # belong to the whole coefficient, not that exponent.
                    estimate = "{" + estimate + "}"
                estimate += "^{" + significance + "}"
            estimates.append(_MathValue(estimate))
            errors.append(_MathValue("(" + _number_atom(coefficient.std_error, precision=precision,
                                                        float_format=float_format) + ")"))
        rows.extend([estimates, errors])
        spaces_after.add(len(rows) - 1)
    rules_before = {len(rows)} if rows else set()
    rows.append(["Observations", *(model.nobs for model in values)])
    statistic_definitions = [
        ("r_squared", _MathValue(r"R^{2}"), {"ols"}),
        ("adjusted_r_squared", _MathValue(r"\text{Adjusted }R^{2}"), {"ols"}),
        ("r_squared_within", _MathValue(r"\text{Within }R^{2}"), set()),
        ("r_squared_between", _MathValue(r"\text{Between }R^{2}"), set()),
        ("r_squared_overall", _MathValue(r"\text{Overall }R^{2}"), set()),
        ("pseudo_r_squared", _MathValue(r"\text{Pseudo }R^{2}"), {"logit", "probit"}),
        ("log_likelihood", "Log likelihood", {"logit", "probit"}),
    ]

    def fit_statistic(model: Any, key: str, estimators: set[str]) -> Any:
        if model.spec.estimator in {"ols", "logit", "probit"}:
            return model.metrics.get(key) if model.spec.estimator in estimators else None
        # Registry fit statistics are available only when explicitly saved.
        return model.metrics.get(key)

    for key, name, estimators in statistic_definitions:
        metrics = [fit_statistic(model, key, estimators) for model in values]
        if any(metric is not None for metric in metrics):
            rows.append([name, *metrics])
    # Equation-specific fit statistics retain their recorded equation labels.
    for equation in groups:
        if equation is None:
            continue
        for statistic, label in (("r_squared", "R-squared"), ("rmse", "RMSE")):
            metrics = [model.metrics.get(f"{equation}:{statistic}") for model in values]
            if any(metric is not None for metric in metrics):
                rows.append([f"{equation}: {label}", *metrics])
    has_inference = any(model.coefficients and model.inference.get("available") is not False for model in values)
    notes = ["Standard errors are in parentheses."] if has_inference else []
    if stars and has_inference:
        notes.append("* p < 0.10; ** p < 0.05; *** p < 0.01 (saved p-values; test conventions below).")
    from openecon.publication import model_notes
    for number, model in enumerate(values, start=1):
        notes.extend(model_notes(model, number=number))
    keep = {position - 1 for position in spaces_after} | group_headers
    return headers, rows, ["l", *("c" for _ in values)], notes, rules_before, spaces_after, keep


def regression_table(models: Any, buf: str | Path | TextIO | None = None, *,
                     precision: int = 4, stars: bool = True,
                     caption: str | tuple[str, str] | None = "Regression results",
                     label: str | None = None, term_labels: Mapping[str, str] | None = None,
                     outcome_labels: Mapping[str, str] | None = None,
                     notes: str | Sequence[str] | None = None,
                     float_format: str | Callable | None = None,
                     font_size: str = "small", max_width: str | None = r"\linewidth",
                     booktabs: bool = True, longtable: bool | None = None,
                     encoding: str = "utf-8") -> Latex | None:
    """Create a publication-style coefficient/SE table from fitted models.

    Model columns keep their input order; rows take the union of fitted terms,
    with constants last and absent terms left empty. Significance uses the
    original two-sided p-values, independent of rounding and model alpha.
    The defaults require booktabs and adjustbox; longtable is required when
    more than 40 body rows trigger automatic page breaks.
    """
    precision = _precision(precision)
    if float_format is not None and not isinstance(float_format, str) and not callable(float_format):
        raise TypeError("float_format must be a callable or a percent-format string.")
    if longtable is not None and not isinstance(longtable, bool):
        raise TypeError("longtable must be True, False or None.")
    headers, rows, alignment, model_notes, rules, spaces, keep = _regression_rows(
        models, precision=precision, stars=stars, float_format=float_format,
        term_labels=term_labels, outcome_labels=outcome_labels)
    source = _tabular(headers, rows, alignment, precision=precision,
        float_format=float_format, na_rep="", caption=caption, label=label,
        column_format=None, longtable=len(rows) > 40 if longtable is None else longtable,
        booktabs=booktabs, notes=[*model_notes, *_notes(notes)], font_size=font_size,
        max_width=max_width, rules_before=rules, spaces_after=spaces,
        keep_with_next=keep)
    source.notes = [*model_notes, *_notes(notes)]
    source.math = _array(headers, rows, alignment, precision=precision,
                         max_rows=50, max_columns=30, rules_before=rules,
                         keep_with_next=keep, float_format=float_format)
    return _write(source, buf, encoding=encoding)


def to_latex(value: Any, buf: str | Path | TextIO | None = None, *, index: bool = True,
             caption: str | tuple[str, str] | None = None, label: str | None = None,
             float_format: str | Callable | None = None, precision: int = 4,
             na_rep: str = "", header: bool | Sequence[str] = True,
             columns: Sequence | None = None, column_format: str | None = None,
             longtable: bool | None = None, booktabs: bool = True, bold_rows: bool = False,
             encoding: str = "utf-8", style: str = "publication", stars: bool = True,
             term_labels: Mapping[str, str] | None = None, outcome_labels: Mapping[str, str] | None = None,
             notes: str | Sequence[str] | None = None, font_size: str = "small",
             max_width: str | None = r"\linewidth") -> Latex | None:
    """Export a frame, series, model or ordinary value as safe LaTeX.

    Publication defaults require ``booktabs`` and ``adjustbox``. Tables with
    more than 40 rows use ``longtable`` automatically unless explicitly
    overridden. Longtables repeat their header and are never boxed, preserving
    page breaks; their font/column layout controls their width. Precision only
    changes presentation. There is no
    implicit row/column limit, and integer cells are never converted to floats.
    """
    precision = _precision(precision)
    if float_format is not None and not isinstance(float_format, str) and not callable(float_format):
        raise TypeError("float_format must be a callable or a percent-format string.")
    if not isinstance(na_rep, str):
        raise TypeError("na_rep must be text.")
    if style not in {"publication", "diagnostic"}:
        raise ValueError("style must be publication or diagnostic.")
    if longtable is not None and not isinstance(longtable, bool):
        raise TypeError("longtable must be True, False or None for automatic page breaks.")
    if isinstance(value, Latex):
        return _write(value, buf, encoding=encoding)
    if _is_series(value):
        value = value.to_frame()
    if _is_result(value) and style == "publication":
        return regression_table(value, buf=buf, caption="Regression results" if caption is None else caption,
            label=label, precision=precision, stars=stars, term_labels=term_labels,
            outcome_labels=outcome_labels, notes=notes, float_format=float_format,
            font_size=font_size, max_width=max_width, booktabs=booktabs,
            longtable=longtable, encoding=encoding)
    if _is_dataframe(value):
        headers, rows, alignment = _table_rows(value, index=index, header=header, columns=columns)
        table_notes = [*_notes(value.attrs.get("publication_notes")), *_notes(notes)]
        source = _tabular(headers, rows, alignment, precision=precision,
                          float_format=float_format, na_rep=na_rep, caption=caption,
                          label=label, column_format=column_format,
                          longtable=len(rows) > 40 if longtable is None else longtable,
                          booktabs=booktabs, bold_rows=bold_rows,
                          index_levels=value.index.nlevels if index else 0,
                          notes=table_notes, font_size=font_size, max_width=max_width)
    elif _is_result(value):
        headers, rows, alignment = _result_rows(value)
        title = f"OpenEconometrics {value.spec.estimator.upper()} — {value.spec.outcome}"
        coefficient_source = _tabular(headers, rows, alignment, precision=precision,
                                      float_format=float_format, na_rep=na_rep,
                                      caption=title if caption is None else caption, label=label,
                                      column_format=column_format,
                                      longtable=len(rows) > 40 if longtable is None else longtable,
                                      booktabs=booktabs, font_size=font_size, max_width=max_width)
        context = [["Observations", value.nobs]]
        if value.inference.get("available") is False:
            context.append(["Target", value.extra.get("target") or "fixed-parameter evaluation"])
        else:
            context.extend([["Covariance", value.inference.get("covariance") or value.spec.covariance],
                            ["Confidence", f"{100 * value.inference.get('confidence_level', 1 - value.spec.alpha):g}%"]])
        if value.dropped_rows:
            excluded_label = "Excluded missing observations" if value.spec.estimator in {"logit", "probit"} else "Excluded observations"
            context.append([excluded_label, value.dropped_rows])
        context.extend([[name, metric] for name, metric in value.metrics.items() if metric is not None])
        context.extend([["Warning", warning] for warning in value.warnings])
        context_notes = _notes(notes)
        from openecon.publication import model_notes
        context_notes.extend(model_notes(value))
        context_source = _tabular([], context, ["l", "r"], precision=precision,
                                  float_format=float_format, na_rep=na_rep,
                                  caption=None, label=None, column_format=None,
                                  longtable=False, booktabs=booktabs, notes=context_notes,
                                  font_size=font_size, max_width=max_width)
        source = Latex(str(coefficient_source) + "\n\n" + str(context_source))
    elif isinstance(value, Mapping):
        headers, rows, alignment = [["Key", "Value"]], [[key, item] for key, item in value.items()], ["l", "l"]
        source = _tabular([["Key", "Value"]], [[key, item] for key, item in value.items()],
                          ["l", "l"], precision=precision, float_format=float_format,
                          na_rep=na_rep, caption=caption, label=label,
                          column_format=column_format,
                          longtable=len(rows) > 40 if longtable is None else longtable,
                          booktabs=booktabs, notes=notes, font_size=font_size, max_width=max_width)
    else:
        headers, rows, alignment = [], [[value]], ["l"]
        source = _tabular([], [[value]], ["l"], precision=precision,
                          float_format=float_format, na_rep=na_rep, caption=caption,
                          label=label, column_format=column_format, longtable=bool(longtable),
                          booktabs=booktabs, notes=notes, font_size=font_size, max_width=max_width)
    source.math = _array(headers, rows, alignment, precision=precision,
                         max_rows=50, max_columns=30, float_format=float_format)
    return _write(source, buf, encoding=encoding)


def latex(value: Any, **options: Any) -> Latex:
    """Return a LaTeX value; use ``Latex(source)`` for deliberate raw source."""
    source = to_latex(value, **options)
    if source is None:
        raise ValueError("latex() returns source; use to_latex(..., buf=...) to write a file.")
    return source


def _array(headers: list[list[Any]], rows: list[list[Any]], alignment: list[str], *,
           precision: int, max_rows: int | None, max_columns: int | None,
           rules_before: set[int] | None = None, keep_with_next: set[int] | None = None,
           float_format: str | Callable | None = None) -> str:
    total_rows, total_columns = len(rows), len(alignment)
    if max_rows is not None:
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 0:
            raise ValueError("max_rows must be a nonnegative integer or None.")
        limit = min(max_rows, total_rows)
        if limit < total_rows:
            while limit and keep_with_next and limit - 1 in keep_with_next:
                limit -= 1
        rows = rows[:limit]
    if max_columns is not None:
        if isinstance(max_columns, bool) or not isinstance(max_columns, int) or max_columns < 1:
            raise ValueError("max_columns must be a positive integer or None.")
        alignment = alignment[:max_columns]
        headers = [row[:max_columns] for row in headers]
        rows = [row[:max_columns] for row in rows]
    lines = [r"\begin{array}{" + ("".join(alignment) or "l") + "}", r"\hline"]
    for row in headers:
        lines.append(" & ".join(_math_cell(value, precision=precision, float_format=float_format, na_rep="")
                                for value in row) + r" \\")
    if headers:
        lines.append(r"\hline")
    for position, row in enumerate(rows):
        if rules_before and position in rules_before:
            lines.append(r"\hline")
        lines.append(" & ".join(_math_cell(value, precision=precision, float_format=float_format, na_rep="")
                                for value in row) + r" \\")
    lines.extend([r"\hline", r"\end{array}"])
    omitted_rows, omitted_columns = total_rows - len(rows), total_columns - len(alignment)
    if omitted_rows or omitted_columns:
        note = f"Preview: {len(rows)} of {total_rows} rows, {len(alignment)} of {total_columns} columns"
        lines = [r"\begin{gathered}", *lines, r"\\", r"\text{" + escape_latex(note) + "}", r"\end{gathered}"]
    return "\n".join(lines)


def display_latex(value: Any, *, precision: int = 4, max_rows: int | None = None,
                  max_columns: int | None = None) -> tuple[Latex, str | None]:
    """Return full export source and a KaTeX-compatible structured preview.

    Only the mathematical preview accepts a visible row/column bound. The
    export source always remains complete. Explicit raw ``Latex`` values have
    no guessed preview; a renderer can show/copy/download their source.
    """
    precision = _precision(precision)
    source = to_latex(value, precision=precision)
    assert source is not None
    if isinstance(value, Latex):
        return source, value.math
    if _is_series(value):
        value = value.to_frame()
    if _is_dataframe(value):
        headers, rows, alignment = _table_rows(value, index=True, header=True)
    elif _is_result(value):
        headers, rows, alignment, _, rules, _, keep = _regression_rows(
            value, precision=precision, stars=True, float_format=None,
            term_labels=None, outcome_labels=None)
        return source, _array(headers, rows, alignment, precision=precision,
                              max_rows=max_rows, max_columns=max_columns, rules_before=rules,
                              keep_with_next=keep)
    elif isinstance(value, Mapping):
        headers, rows, alignment = [["Key", "Value"]], [[key, item] for key, item in value.items()], ["l", "l"]
    else:
        headers, rows, alignment = [], [[value]], ["l"]
    return source, _array(headers, rows, alignment, precision=precision,
                          max_rows=max_rows, max_columns=max_columns)


def _record_table(columns: Sequence[Any], rows: Sequence[Sequence[Any]], *,
                  index: Sequence[Any] | None, index_names: Sequence[Any] | None
                  ) -> tuple[list[list[Any]], list[list[Any]], list[str]]:
    """Build a table from JSON-safe records without initializing pandas."""
    columns = list(columns)
    rows = [list(row) for row in rows]
    if any(len(row) != len(columns) for row in rows):
        raise ValueError("Each table row must match the number of columns.")
    data_alignment = []
    for position in range(len(columns)):
        cells = [_scalar(row[position]) for row in rows if not _missing(_scalar(row[position]))]
        data_alignment.append("r" if cells and all(isinstance(cell, (Real, Decimal))
                              and not isinstance(cell, bool) for cell in cells) else "l")
    left_count = 0
    names = []
    if index is not None:
        if len(index) != len(rows):
            raise ValueError("The index must contain one entry per table row.")
        left_count = len(index_names) if index_names is not None else 1
        if not left_count:
            raise ValueError("An exported index must contain at least one level.")
        names = list(index_names) if index_names is not None else [None]
        prefixed = []
        for row, row_index in zip(rows, index, strict=True):
            # Stored console tables use one list of levels per row even when
            # the index has only one level; direct callers can use scalars.
            left = list(row_index) if isinstance(row_index, (list, tuple)) else [row_index]
            if len(left) != left_count:
                raise ValueError("Index values must match the number of index names.")
            prefixed.append([*left, *row])
        rows = prefixed
    elif index_names is not None:
        raise ValueError("index_names requires index values.")
    return [[*names, *columns]], rows, [*(["l"] * left_count), *data_alignment]


def table_latex(columns: Sequence[Any], rows: Sequence[Sequence[Any]], *,
                index: Sequence[Any] | None = None, index_names: Sequence[Any] | None = None,
                precision: int = 4, caption: str | tuple[str, str] | None = None,
                label: str | None = None, na_rep: str = "", booktabs: bool = True,
                notes: str | Sequence[str] | None = None, font_size: str = "small",
                max_width: str | None = r"\linewidth", longtable: bool | None = None) -> Latex:
    """Export record rows in a server without importing dataframe libraries."""
    headers, values, alignment = _record_table(columns, rows, index=index, index_names=index_names)
    source = _tabular(headers, values, alignment, precision=_precision(precision),
                      float_format=None, na_rep=na_rep, caption=caption, label=label,
                      column_format=None,
                      longtable=len(values) > 40 if longtable is None else longtable,
                      booktabs=booktabs, notes=notes, font_size=font_size, max_width=max_width)
    source.math = _array(headers, values, alignment, precision=precision,
                         max_rows=50, max_columns=30)
    return source


def table_math(columns: Sequence[Any], rows: Sequence[Sequence[Any]], *,
               index: Sequence[Any] | None = None, index_names: Sequence[Any] | None = None,
               precision: int = 4, max_rows: int | None = 50,
               max_columns: int | None = 30) -> str:
    """Render JSON-safe record rows as a bounded KaTeX array."""
    headers, values, alignment = _record_table(columns, rows, index=index, index_names=index_names)
    return _array(headers, values, alignment, precision=_precision(precision),
                  max_rows=max_rows, max_columns=max_columns)
