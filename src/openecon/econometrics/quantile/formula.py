"""Safe parser and analytic differentiator for the regression functions of ``nl``.

Grammar (Stata's substitutable expressions, restricted):

    parameters   {b0}  or  {b0=1.5}  (a name in braces, optionally with a starting value)
    columns      identifiers (data column names that are valid identifiers)
    numbers      1, 2.5, 1e-3
    operators    + - * / ^ (power; ** is accepted too), unary + and -, parentheses
    functions    exp, ln, log (natural logarithm, as in Stata), sqrt, abs, sin, cos, tan,
                 expit / invlogit (1 / (1 + exp(-u))), normal (Phi), normalden (phi)

Precedence is Stata's: ^ binds tighter than unary minus (-x^2 = -(x^2)), then * and /,
then + and -.  Stata evaluates a^b^c left to right while Python and most languages
read it right to left; to avoid a silent difference an unparenthesized chain of
powers is rejected.

Safety.  The text is never evaluated as code.  Parameters are replaced by
placeholder names, ``ast.parse`` (eval mode) builds a syntax tree, and the tree is
walked once: any node outside the grammar (attribute access, subscripts, comparisons,
lambdas, keyword arguments, strings, calls of unknown names ...) raises
``AnalysisError('invalid_formula')``.  The accepted tree is compiled into a flat
"tape" of elementary operations (with common subexpressions shared), which is all that
is executed.

Derivatives.  The Jacobian with respect to the parameters is analytic: the tape is
evaluated in forward mode, every operation propagating the exact partial derivatives
of its result by the chain rule

    (a b)' = a' b + a b'        (a / b)' = (a' - (a/b) b') / b
    (a^c)' = c a^(c-1) a'       (a^b)' = a^b (b' ln a + b a' / a)
    exp' = exp,  ln' = 1/a,  sqrt' = 1/(2 sqrt a),  |a|' = sign(a),  sin' = cos,
    cos' = -sin,  tan' = 1 + tan^2,  expit' = expit (1 - expit),  Phi' = phi,
    phi' = -a phi.

Partials are kept per parameter only where they are nonzero, so a column of the
Jacobian costs as much as the part of the formula that involves its parameter. No
autograd graph is built.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError

FUNCTIONS = ("exp", "ln", "log", "sqrt", "abs", "sin", "cos", "tan", "expit", "invlogit",
             "normal", "normalden")
_MAX_LENGTH = 4000
_PLACEHOLDER = "__oe_parameter_"
_BRACES = re.compile(r"\{([^{}]*)\}")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_BINARY = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow"}
_SQRT_2PI = math.sqrt(2 * math.pi)


def _invalid(message: str) -> AnalysisError:
    return AnalysisError("invalid_formula", message)


@dataclass(frozen=True)
class Formula:
    """A parsed regression function: a tape of operations over columns and parameters."""

    source: str
    parameters: tuple[str, ...]          # in order of first appearance
    columns: tuple[str, ...]             # data columns the formula reads
    start: dict[str, float]              # {b=value} starting values written in the formula
    tape: tuple[tuple, ...]              # (op, *operands); operands index earlier entries


def _parameters(text: str) -> tuple[str, list[str], dict[str, float]]:
    names: list[str] = []
    start: dict[str, float] = {}

    def replace(match: re.Match) -> str:
        name, _, value = match.group(1).partition("=")
        name, value = name.strip(), value.strip()
        if not _NAME.match(name):
            raise _invalid(f"'{{{match.group(1)}}}' is not a parameter: write a name in braces "
                           "such as {b0} or {b0=1.5}. Linear-combination shorthands like "
                           "{xb: x1 x2} are not supported; write {b1}*x1 + {b2}*x2.")
        if "=" in match.group(1):
            try:
                number = float(value)
            except ValueError:
                raise _invalid(f"The starting value of parameter '{name}' must be a number, "
                               f"not '{value}'.") from None
            if not math.isfinite(number):
                raise _invalid(f"The starting value of parameter '{name}' must be finite.")
            if name in start and start[name] != number:
                raise _invalid(f"Parameter '{name}' is given two different starting values.")
            start[name] = number
        if name not in names:
            names.append(name)
        return f" {_PLACEHOLDER}{name} "

    replaced = _BRACES.sub(replace, text)
    if "{" in replaced or "}" in replaced:
        raise _invalid("Unbalanced braces: parameters are written as {name} or {name=value}.")
    return replaced, names, start


class _Compiler:
    """Walk the syntax tree once, rejecting foreign nodes and emitting the tape."""

    def __init__(self, text: str, parameters: list[str]):
        self.text = text
        self.parameters = parameters
        self.columns: list[str] = []
        self.tape: list[tuple] = []
        self.index: dict[tuple, int] = {}

    def emit(self, *entry) -> int:
        if entry not in self.index:
            self.index[entry] = len(self.tape)
            self.tape.append(entry)
        return self.index[entry]

    def _parenthesized(self, node: ast.AST) -> bool:
        before = self.text[:node.col_offset].rstrip()
        return before.endswith("(")

    def visit(self, node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                raise _invalid(f"Only finite numbers are allowed as constants, not {value!r}.")
            return self.emit("const", float(value))
        if isinstance(node, ast.Name):
            if node.id.startswith(_PLACEHOLDER):
                return self.emit("par", self.parameters.index(node.id[len(_PLACEHOLDER):]))
            if node.id not in self.columns:
                self.columns.append(node.id)
            return self.emit("col", node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = self.visit(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else self.emit("neg", operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            if isinstance(node.op, ast.Pow):
                for child in (node.left, node.right):
                    inner = child
                    while isinstance(inner, ast.UnaryOp):
                        inner = inner.operand
                    if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Pow) \
                            and not (self._parenthesized(child) or self._parenthesized(inner)):
                        raise _invalid("A chain of powers such as a^b^c is ambiguous (Stata "
                                       "reads it left to right); add parentheses.")
            return self.emit(_BINARY[type(node.op)], self.visit(node.left), self.visit(node.right))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                called = node.func.id if isinstance(node.func, ast.Name) else "this expression"
                raise _invalid(f"'{called}' is not an available function. Functions: "
                               f"{', '.join(FUNCTIONS)}.")
            if node.keywords or len(node.args) != 1 or isinstance(node.args[0], ast.Starred):
                raise _invalid(f"{node.func.id}() takes exactly one argument.")
            return self.emit("fn", node.func.id, self.visit(node.args[0]))
        raise _invalid("The formula may contain only parameters in braces, column names, "
                       "numbers, + - * / ^, parentheses and the functions "
                       f"{', '.join(FUNCTIONS)}.")


def parse(text: object) -> Formula:
    """Parse an nl regression function; raises ``AnalysisError('invalid_formula')``."""
    if not isinstance(text, str) or not text.strip():
        raise _invalid("formula must be a non-empty string such as '{b0} + {b1}*exp(-{b2}*x)'.")
    if len(text) > _MAX_LENGTH:
        raise _invalid(f"The formula is longer than {_MAX_LENGTH} characters.")
    if "=" in _BRACES.sub("", text):
        raise _invalid("Give the regression function only (the right-hand side): the outcome "
                       "is passed separately as y.")
    if _PLACEHOLDER in text:
        raise _invalid(f"Names starting with '{_PLACEHOLDER}' are reserved.")
    replaced, parameters, start = _parameters(" ".join(text.split()))
    replaced = replaced.replace("^", "**").strip()
    try:
        tree = ast.parse(replaced, mode="eval")
        compiler = _Compiler(replaced, parameters)
        compiler.visit(tree)
    except AnalysisError:
        raise
    except SyntaxError as exc:
        raise _invalid(f"The formula could not be parsed ({exc.msg}). Check parentheses and "
                       "operators.") from None
    except (RecursionError, MemoryError, ValueError):
        raise _invalid("The formula is nested too deeply to be parsed.") from None
    if not parameters:
        raise _invalid("The formula contains no parameter; write parameters in braces, for "
                       "example {b0} + {b1}*x.")
    return Formula(source=text, parameters=tuple(parameters), columns=tuple(compiler.columns),
                   start=start, tape=tuple(compiler.tape))


# ---- forward-mode evaluation ---------------------------------------------------------------

Partials = dict[int, Tensor]


def _combine(left: Partials, right: Partials, a: Tensor | float, b: Tensor | float) -> Partials:
    """a * left + b * right over the union of their parameters."""
    out = {j: a * d for j, d in left.items()}
    for j, d in right.items():
        out[j] = out[j] + b * d if j in out else b * d
    return out


def _scale(partials: Partials, factor: Tensor | float) -> Partials:
    return {j: factor * d for j, d in partials.items()}


def _function(name: str, a: Tensor, da: Partials, jacobian: bool) -> tuple[Tensor, Partials]:
    """Value of one elementary function and the chain-rule partials of its result."""
    if name == "exp":
        value = torch.exp(a)
    elif name in ("ln", "log"):
        value = torch.log(a)
    elif name == "sqrt":
        value = torch.sqrt(a)
    elif name == "abs":
        value = a.abs()
    elif name == "sin":
        value = torch.sin(a)
    elif name == "cos":
        value = torch.cos(a)
    elif name == "tan":
        value = torch.tan(a)
    elif name in ("expit", "invlogit"):
        value = torch.sigmoid(a)
    elif name == "normal":
        value = 0.5 * torch.special.erfc(-a / math.sqrt(2.0))
    else:                                    # normalden
        value = torch.exp(-0.5 * a.square()) / _SQRT_2PI
    if not (jacobian and da):
        return value, {}
    if name in ("exp",):
        slope = value
    elif name in ("ln", "log"):
        slope = 1 / a
    elif name == "sqrt":
        slope = 0.5 / value
    elif name == "abs":
        slope = torch.sign(a)
    elif name == "sin":
        slope = torch.cos(a)
    elif name == "cos":
        slope = -torch.sin(a)
    elif name == "tan":
        slope = 1 + value.square()
    elif name in ("expit", "invlogit"):
        slope = value * (1 - value)
    elif name == "normal":
        slope = torch.exp(-0.5 * a.square()) / _SQRT_2PI
    else:
        slope = -a * value
    return value, _scale(da, slope)


def _power(a: Tensor, b: Tensor, da: Partials, db: Partials, constant: float | None,
           jacobian: bool) -> tuple[Tensor, Partials]:
    value = torch.pow(a, b)
    if not jacobian or not (da or db):
        return value, {}
    partials: Partials = {}
    if da:
        if constant is not None:
            # d a^c = c a^(c-1) da, finite at a = 0 for c >= 1 (and for c = 0: zero).
            slope = torch.zeros_like(a) if constant == 0 else constant * torch.pow(a, constant - 1)
        else:
            slope = b * torch.pow(a, b - 1)
        partials = _scale(da, slope)
    if db:
        # d a^b / db = a^b ln a, which is 0 where a^b = 0 (a = 0 with a positive exponent).
        growth = torch.where(value == 0, torch.zeros_like(value), value * torch.log(a))
        partials = _combine(partials, db, 1.0, growth)
    return value, partials


@torch.no_grad()
def evaluate(formula: Formula, columns: dict[str, Tensor], theta: Tensor, n: int, *,
             jacobian: bool = True) -> tuple[Tensor, Tensor | None]:
    """f(x; theta) [n] and, when requested, its analytic Jacobian [n, p].

    ``columns`` maps every name in ``formula.columns`` to a float64 [n] tensor
    and ``theta`` holds the parameters in the order of ``formula.parameters``.
    Non-finite values are returned as they arise; the caller decides.
    """
    one = torch.ones((), dtype=torch.float64)
    values: list[Tensor] = []
    partials: list[Partials] = []
    for entry in formula.tape:
        op = entry[0]
        if op == "const":
            value, grad = torch.tensor(entry[1], dtype=torch.float64), {}
        elif op == "col":
            value, grad = columns[entry[1]], {}
        elif op == "par":
            value, grad = theta[entry[1]], ({entry[1]: one} if jacobian else {})
        elif op == "neg":
            value = -values[entry[1]]
            grad = _scale(partials[entry[1]], -1.0)
        elif op == "fn":
            value, grad = _function(entry[1], values[entry[2]], partials[entry[2]], jacobian)
        else:
            a, b = values[entry[1]], values[entry[2]]
            da, db = partials[entry[1]], partials[entry[2]]
            if op == "add":
                value, grad = a + b, _combine(da, db, 1.0, 1.0)
            elif op == "sub":
                value, grad = a - b, _combine(da, db, 1.0, -1.0)
            elif op == "mul":
                value, grad = a * b, _combine(da, db, b, a)
            elif op == "div":
                value = a / b
                grad = _scale(_combine(da, db, 1.0, -value), 1 / b) if (da or db) else {}
            else:
                exponent = formula.tape[entry[2]]
                constant = exponent[1] if exponent[0] == "const" else None
                value, grad = _power(a, b, da, db, constant, jacobian)
        values.append(value)
        partials.append(grad)
    result = values[-1]
    result = result.expand(n).clone() if result.ndim == 0 else result
    if not jacobian:
        return result, None
    matrix = torch.zeros((n, len(formula.parameters)), dtype=torch.float64)
    for j, column in partials[-1].items():
        matrix[:, j] = column
    return result, matrix
