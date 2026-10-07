"""A small, safe Python formula compiler with reusable treatment coding.

No eval, Patsy or statistical package is involved. Expressions are parsed to an
allowlisted AST; the fitted category levels are frozen for new-data predictions.
"""
from __future__ import annotations

import ast
from itertools import combinations, product
import math
import re

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.streaming_design import MAX_CATEGORY_BYTES, MAX_PARAMETERS, numeric_values


def _split(text: str, separator: str) -> list[str]:
    depth, start, result = 0, 0, []
    for i, char in enumerate(text):
        depth += (char == "(") - (char == ")")
        if depth < 0:
            raise AnalysisError("invalid_formula", "Unbalanced formula parentheses.")
        if depth == 0 and char == separator:
            result.append(text[start:i].strip())
            start = i + 1
    if depth:
        raise AnalysisError("invalid_formula", "Unbalanced formula parentheses.")
    result.append(text[start:].strip())
    return result


def expand_terms(expressions: list[str], *, literal_names=()) -> list[str]:
    if not isinstance(expressions, list) or len(expressions) > MAX_PARAMETERS:
        raise AnalysisError("model_too_wide", "Formula terms exceed 384 parameters.")
    expanded = []
    for expression in expressions:
        if not expression.strip():
            raise AnalysisError("invalid_formula", "Empty formula term.")
        # ** inside I(...) is arithmetic, whereas top-level * expands all
        # main effects and interactions, as in Python statistical formulae.
        factors = [expression] if expression in literal_names else _split(expression, "*")
        if len(factors) > 8 or (len(factors) > 1 and 2 ** len(factors) - 1 > MAX_PARAMETERS):
            raise AnalysisError("model_too_wide", "Factorial expansion exceeds 384 parameters.")
        terms = [expression] if len(factors) == 1 else [
            ":".join(combo) for count in range(1, len(factors) + 1)
            for combo in combinations(factors, count)]
        for term in terms:
            if term not in expanded:
                expanded.append(term)
                if len(expanded) > MAX_PARAMETERS:
                    raise AnalysisError("model_too_wide", "Formula terms exceed 384 parameters.")
    return expanded


def parse_formula(formula: str) -> tuple[str, list[str], bool]:
    if not isinstance(formula, str) or formula.count("~") != 1:
        raise AnalysisError("invalid_formula", "Use a formula such as 'wage ~ education + experience'.")
    outcome, right = (part.strip() for part in formula.split("~"))
    if not outcome.isidentifier():
        raise AnalysisError("invalid_formula", "The formula outcome must be a column identifier.")
    intercept = True
    terms = []
    # Only intercept subtraction is accepted; general numeric subtraction
    # belongs inside I(...), preventing ambiguous design interpretation.
    right = re.sub(r"\s*-\s*1\s*$", "+ 0", right)
    for term in _split(right, "+"):
        if term in {"0", "-1"}:
            intercept = False
        elif term == "1":
            intercept = True
        elif term:
            terms.append(term)
    return outcome, expand_terms(terms), intercept


def _tree(expression: str):
    if len(expression) > 4096:
        raise AnalysisError("invalid_formula", "A formula term exceeds the bounded expression size.")
    try:
        node = ast.parse(expression, mode="eval").body
    except (SyntaxError, RecursionError) as exc:
        raise AnalysisError("invalid_formula", f"Invalid formula term: {expression}.") from exc
    allowed = (ast.Name, ast.Load, ast.Constant, ast.BinOp, ast.UnaryOp,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Call)
    calls = {"I", "log", "exp", "sqrt", "C", "L", "D"}
    for count, item in enumerate(ast.walk(node)):
        if count > 256:
            raise AnalysisError("invalid_formula", "A formula term has too many operations.")
        if not isinstance(item, allowed):
            raise AnalysisError("invalid_formula", "Formula expressions contain an unsupported operation.")
        if isinstance(item, ast.Call) and (not isinstance(item.func, ast.Name)
                                          or item.func.id not in calls or item.keywords):
            raise AnalysisError("invalid_formula", "Allowed formula functions: I, C, log, exp, sqrt, L, D.")
        if isinstance(item, ast.Constant) and (isinstance(item.value, bool)
                                              or not isinstance(item.value, (int, float))):
            raise AnalysisError("invalid_formula", "Formula constants must be numeric.")
    return node


def expression_columns(expression: str) -> list[str]:
    node = _tree(expression)
    functions = {id(item.func) for item in ast.walk(node) if isinstance(item, ast.Call)}
    return list(dict.fromkeys(item.id for item in ast.walk(node)
                             if isinstance(item, ast.Name) and id(item) not in functions))


def _current_columns(node):
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, ast.Call):
        if node.func.id != "L":
            for argument in node.args:
                yield from _current_columns(argument)
    else:
        for child in ast.iter_child_nodes(node):
            yield from _current_columns(child)


class OLSDesign:
    def __init__(self, expressions: list[str], categorical: list[str], *, intercept: bool,
                 time: str | None = None, literal_names=()):
        self.literal_names = frozenset(literal_names)
        self.expressions = expand_terms(expressions, literal_names=self.literal_names)
        self.intercept, self.time = intercept, time
        if intercept and "Intercept" in self.expressions:
            raise AnalysisError("duplicate_terms", "The predictor name Intercept conflicts with the generated intercept.")
        self.categorical = list(categorical)
        self.required = []
        current_names = set()
        self.factor_specs = []
        for expression in self.expressions:
            factors = []
            for factor in ([expression] if expression in self.literal_names else _split(expression, ":")):
                literal = factor in self.literal_names
                if literal and len(factor) > 4096:
                    raise AnalysisError("invalid_spec", "A predictor column name exceeds the bounded name size.")
                tree = ast.Name(id=factor, ctx=ast.Load()) if literal else _tree(factor)
                current_names.update(_current_columns(tree))
                if isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name) and tree.func.id == "C":
                    if len(tree.args) != 1 or not isinstance(tree.args[0], ast.Name):
                        raise AnalysisError("invalid_formula", "C() needs exactly one column name.")
                    name = tree.args[0].id
                    if name not in self.categorical:
                        self.categorical.append(name)
                    factors.append((name, "categorical", tree))
                elif factor in self.categorical:
                    factors.append((factor, "categorical", tree))
                else:
                    factors.append((factor, "numeric", tree))
                for column in ([factor] if literal else expression_columns(factor)):
                    if column not in self.required:
                        self.required.append(column)
            self.factor_specs.append(factors)
        self.current_required = [name for name in self.required
                                 if name in current_names or name in self.categorical]
        self.categories = {}
        self.terms = []
        self.blocks = []
        self.lag_specs = {}
        for factors in self.factor_specs:
            for _, _, tree in factors:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and node.func.id in {"L", "D"}:
                        if self.time is None or not node.args or not isinstance(node.args[0], ast.Name):
                            raise AnalysisError("invalid_formula", "L() and D() need a column name and time=.")
                        lag = 1 if len(node.args) == 1 else getattr(node.args[1], "value", None)
                        if len(node.args) > 2 or type(lag) is not int or lag < 1:
                            raise AnalysisError("invalid_formula", "The lag length must be a positive integer.")
                        key = (node.args[0].id, lag)
                        if key not in self.lag_specs:
                            self.lag_specs[key] = f"__oe_lag_{len(self.lag_specs)}__"
        self.lag_columns = list(self.lag_specs.values())

    def with_streaming_history(self, source, *, batch_rows=65536, columns=None):
        from .lags import lag_source
        return lag_source(source, self, batch_rows=batch_rows, columns=columns)

    def prepare(self, frame: pd.DataFrame):
        categories_bytes = 0
        for column in self.categorical:
            values = frame[column]
            if isinstance(values.dtype, pd.CategoricalDtype):
                if len(values.cat.categories) > MAX_PARAMETERS + 1:
                    raise AnalysisError("model_too_wide", "Categorical expansion exceeds the model-width budget.")
                levels = list(values.cat.categories)
            else:
                observed = set()
                for start in range(0, len(values), 4096):
                    for level in values.iloc[start:start + 4096].dropna().unique():
                        observed.add(level.item() if hasattr(level, "item") else level)
                    if len(observed) > MAX_PARAMETERS + 1:
                        raise AnalysisError("model_too_wide", "Categorical expansion exceeds the model-width budget.")
                try:
                    levels = sorted(observed)
                except TypeError as exc:
                    raise AnalysisError("unsupported_category", "Category levels must have one sortable scalar type.") from exc
            if not levels or len(levels) > MAX_PARAMETERS + 1:
                raise AnalysisError("model_too_wide", "Categorical expansion exceeds the model-width budget.")
            for level in levels:
                if not isinstance(level, (str, int, float, bool)):
                    raise AnalysisError("unsupported_category", "Category levels must be strings, numbers or booleans.")
                categories_bytes += len(str(level).encode("utf-8")) + 64
            if categories_bytes > MAX_CATEGORY_BYTES:
                raise AnalysisError("category_budget", "Category labels exceed the bounded metadata budget.")
            self.categories[column] = levels
        self._build_blocks()
        return self

    def _build_blocks(self):
        self.terms = ["Intercept"] if self.intercept else []
        self.blocks = [{"name": "Intercept", "factors": []}] if self.intercept else []
        for factors in self.factor_specs:
            width = 1
            for name, kind, _ in factors:
                width *= len(self.categories[name]) - int(self.intercept) if kind == "categorical" else 1
                if width > MAX_PARAMETERS:
                    raise AnalysisError("model_too_wide", "The expanded model exceeds 384 parameters.")
            choices = []
            for name, kind, tree in factors:
                if kind == "categorical":
                    # A category alone without an intercept keeps every level;
                    # interactions use reference coding with their main effects.
                    levels = self.categories[name][int(self.intercept):]
                    choices.append([{"name": name, "kind": kind, "level": level,
                                     "tree": tree} for level in levels])
                else:
                    choices.append([{"name": name, "kind": kind, "tree": tree}])
            for combo in product(*choices):
                label = ":".join(f"{item['name']}[{item['level']}]" if item["kind"] == "categorical"
                                 else item["name"] for item in combo)
                if label not in self.terms:
                    if len(self.terms) >= MAX_PARAMETERS:
                        raise AnalysisError("model_too_wide", "The expanded model exceeds 384 parameters.")
                    self.terms.append(label)
                    self.blocks.append({"name": label, "factors": list(combo)})
        if len(self.terms) > MAX_PARAMETERS:
            raise AnalysisError("model_too_wide", "The expanded model exceeds 384 parameters.")

    def _value(self, node, frame: pd.DataFrame, *, allow_missing=False):
        if isinstance(node, ast.Name):
            if allow_missing and frame[node.id].isna().any():
                mask = ~frame[node.id].isna()
                result = torch.full((len(frame),), math.nan, dtype=torch.float64)
                result[torch.tensor(mask.tolist(), dtype=torch.bool)] = numeric_values(frame.loc[mask, node.id], node.id)
                return result
            return numeric_values(frame[node.id], node.id)
        if isinstance(node, ast.Constant):
            return torch.full((len(frame),), float(node.value), dtype=torch.float64)
        if isinstance(node, ast.UnaryOp):
            value = self._value(node.operand, frame, allow_missing=allow_missing)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = self._value(node.left, frame, allow_missing=allow_missing), self._value(node.right, frame, allow_missing=allow_missing)
            operation = {ast.Add: torch.add, ast.Sub: torch.sub, ast.Mult: torch.mul,
                         ast.Div: torch.div, ast.Pow: torch.pow}[type(node.op)]
            return operation(left, right)
        name = node.func.id
        if name in {"L", "D"}:
            if self.time is None or not node.args or not isinstance(node.args[0], ast.Name):
                raise AnalysisError("invalid_formula", "L() and D() need a column name and time=.")
            lag = 1 if len(node.args) == 1 else getattr(node.args[1], "value", None)
            if len(node.args) > 2 or isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
                raise AnalysisError("invalid_formula", "The lag length must be a positive integer.")
            column = self.lag_specs[(node.args[0].id, lag)]
            if column in frame:
                result = torch.tensor(frame[column].to_numpy(dtype="float64", na_value=math.nan), dtype=torch.float64)
            else:
                from .lags import lag_values
                result = lag_values(frame, self.time, node.args[0].id, lag, allow_missing=allow_missing)
            return self._value(node.args[0], frame, allow_missing=allow_missing) - result if name == "D" else result
        if len(node.args) != 1:
            raise AnalysisError("invalid_formula", f"{name}() needs exactly one argument.")
        value = self._value(node.args[0], frame, allow_missing=allow_missing)
        return value if name == "I" else {"log": torch.log, "exp": torch.exp, "sqrt": torch.sqrt}[name](value)

    def encode(self, frame: pd.DataFrame, *, allow_missing=False) -> torch.Tensor:
        absent = set(self.required) - set(frame.columns)
        if absent:
            raise AnalysisError("missing_columns", f"Missing predictor columns: {', '.join(sorted(absent))}.")
        for column, levels in self.categories.items():
            if not (frame[column].isin(levels) | (frame[column].isna() & allow_missing)).all():
                raise AnalysisError("unknown_category", f"Column '{column}' contains a missing or unfitted category level.")
        columns = []
        cache = {}
        for block in self.blocks:
            value = torch.ones(len(frame), dtype=torch.float64)
            for factor in block["factors"]:
                key = factor["name"], factor.get("level")
                if key not in cache:
                    if factor["kind"] == "categorical":
                        cache[key] = torch.tensor((frame[factor["name"]] == factor["level"]).fillna(False).tolist(), dtype=torch.float64)
                        if allow_missing:
                            cache[key][torch.tensor(frame[factor["name"]].isna().tolist(), dtype=torch.bool)] = math.nan
                    else:
                        cache[key] = self._value(factor["tree"], frame, allow_missing=allow_missing)
                value = value * cache[key]
            columns.append(value)
        return torch.stack(columns, dim=1) if columns else torch.empty((len(frame), 0), dtype=torch.float64)
