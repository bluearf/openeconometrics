#!/usr/bin/env python3
"""Build editor help from committed Python syntax, without importing user code.

Run with the Python used to package OpenEconometrics so inherited pandas methods and
builtins match that environment. openecon and the charts package are read from
Git, including when the working tree contains experimental model changes.
"""

from __future__ import annotations

import argparse
import ast
import builtins
from copy import deepcopy
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "web" / "src" / "editor-api.json"
FRAME_METHODS = (
    "head", "tail", "describe", "copy", "sample", "sort_values", "drop",
    "dropna", "fillna", "rename", "assign", "query", "merge", "groupby",
    "drop_duplicates", "value_counts", "reset_index", "set_index", "isna",
    "notna", "corr", "mean", "sum", "std", "to_csv", "to_parquet", "to_excel",
)
BUILTINS = (
    "abs", "all", "any", "enumerate", "isinstance", "issubclass", "len",
    "print", "repr", "round", "sorted", "sum",
)

# Short descriptions explain the existing public behavior. Signatures and
# defaults always come from syntax or the trusted standard-library builtin.
DESCRIPTIONS = {
    "openecon.ngperron": "Ng-Perron GLS MZa/MZt/MSB/MPT statistics only, p_value=None. Fixed lags or MAIC including lag zero. No critical values or rejection decisions until published-table calibration is resolved (MARKET-136). Complete consecutive integer periods required.",
    "openecon.kss": "KSS cubic-regression unit-root test against globally stationary ESTAR dynamics. Raw, demeaned or detrended levels with fixed augmentation; published asymptotic 1/5/10% lower-tail critical values, p_value=None.",
    "openecon.ols": "Fit an OLS/WLS regression. Use y/x or formula; explicitly choose standard errors, weights and missing-value handling.",
    "openecon.logit": "Fit a logistic regression for a binary (0/1) outcome.",
    "openecon.probit": "Fit a probit regression for a binary (0/1) outcome.",
    "openecon.fit": "Fit a model from a validated ModelSpec and data.",
    "openecon.read": "Read a local data file. Large CSV/Parquet sources automatically return a bounded, replayable Dataset; small files return a DataFrame.",
    "openecon.scan": "Open a local CSV or Parquet source for batch reading without loading all data into memory.",
    "openecon.example": "Return synthetic wage, education and experience data with 480 observations.",
    "openecon.load_dataset": "Open an imported dataset by its ID in the active local workspace.",
    "openecon.install": "Install packages in the project's Python environment and continue running the same script.",
    "openecon.capabilities": "List the implemented statistical features and current limits.",
    "openecon.predict": "Predict from fitted OLS or supported saved models. DataFrame inputs return a table; Dataset inputs return complete disk-backed predictions in bounded blocks. Native OLS defaults to xb; saved adapters default to response. Targets depend on the fitted model.",
    "openecon.margins": "Compute supported marginal effects from fitted OLS or saved parameters. Dataset inputs use global weighted effects (AME) or global encoded means (MEM), with saved full covariance. Options depend on the fitted model.",
    "openecon.DataFrame": "OpenEconometrics' pandas-compatible data table with direct LaTeX export support.",
    "openecon.ModelSpec": "A model specification that validates the estimator, outcome, predictors and inference options.",
    "openecon.ResultBundle": "A validated result containing coefficients, covariance, fit statistics and model information.",
    "openecon.Dataset": "A batch data source. Use scan for a file or Dataset.from_frame for an existing table.",
    "openecon.network": "Build a directed or undirected weighted sparse network from an edge table. Dataset inputs are read in batches; the retained graph must fit the explicit memory budget.",
    "openecon.hypergraph": "Build typed directed or undirected hyperedges with sparse incidence and explicit bounded projections.",
    "openecon.NetworkMatchingResult.summary": "Return objective, cardinality, total reward, unmatched count and exact certificate status.",
    "openecon.NetworkMatchingResult.to_latex": "Export the certified matching summary as an escaped LaTeX table.",
    "openecon.network_snapshots": "Capture ordered temporal or general network snapshots. All sparse snapshots stay resident; use transitions, persistence, aggregation or explicitly ordered temporal paths.",
    "openecon.ergm": "Exact or MPLE ERGM on bounded binary graphs.",
    "openecon.simulate_ergm": "Seeded ERGM Gibbs chain with burn-in and thinning.",
    "openecon.saom": "Exact CTMC SAOM on small fully observed panels.",
    "openecon.simulate_saom": "Simulate SAOM opportunities at explicit times.",
    "openecon.gaussian_block_model": "Gaussian blocks on explicit real dyads.",
    "openecon.mixed_membership_block_model": "Conditional Bernoulli simplexes on train dyads.",
    "openecon.network_embedding": "Mini-batch link factors on explicit binary dyads.",
    "openecon.network_gnn": "Sampled mean-GraphSAGE with node and edge splits.",
    "openecon.Latex": "Store explicitly supplied raw LaTeX source as a displayable value.",
    "openecon.DataFrame.describe": "Return descriptive statistics for numeric or selected columns as an OpenEconometrics table.",
    "openecon.DataFrame.groupby": "Group rows by the specified columns or keys; call aggregation separately.",
    "openecon.DataFrame.merge": "Merge two tables using shared columns or indexes.",
    "openecon.DataFrame.mean": "Calculate means along the selected axis; numeric_only selects numeric columns.",
    "openecon.DataFrame.sum": "Calculate sums along the selected axis.",
    "openecon.DataFrame.std": "Calculate standard deviations along the selected axis; ddof=1 gives the sample standard deviation.",
    "openecon.DataFrame.to_latex": "Convert the table to escaped, publication-ready LaTeX source.",
    "openecon.ResultBundle.summary": "Return estimated coefficients as readable text or a LaTeX table.",
    "openecon.ResultBundle.to_latex": "Export coefficients, standard errors, fit statistics and inference notes as a LaTeX table.",
    "openecon.OLSResult.vif": "Calculate variance inflation factors for OLS predictors.",
    "openecon.OLSResult.white_test": "Calculate White's heteroskedasticity test from OLS residuals.",
    "console.display": "Display a table, model, chart or text in the results panel.",
}

PARAMETER_DESCRIPTIONS = {
    "openecon.Network.qap_regression": {
        "predictors": "Mapping from unique predictor names to aligned Network snapshots; all dyads include absent edges as zero.",
        "values": "'weight' uses aggregate strengths; 'binary' uses edge presence.",
        "include_loops": "Include diagonal dyads; False excludes them.",
        "permutations": "Uniform node permutations per coefficient-specific reduced-model residual test; plus-one p-values.",
        "seed": "Private CPU Torch random seed; predictor names and node labels use canonical order.",
        "alternative": "'two-sided' compares absolute partial correlations; 'greater' and 'less' use the signed statistic.",
        "intercept": "Fit a constant; the constant receives no fabricated permutation test.",
        "max_work": "Explicit complete-test structural work budget; no partial permutation p-values.",
        "method": "'freedman_lane' permutes reduced response residuals; 'dsp' permutes focal predictor residuals and projects nuisance effects again.",
        "joint": "Named coefficient subsets tested together, conditional on remaining predictors; upper-tail partial R-squared.",
        "adjustment": "'none', 'bonferroni', 'holm' or 'bh' over all reported slopes and joint hypotheses; the intercept is excluded.",
    },
    "openecon.Network.block_model": {
        "groups": "Fixed number of nonempty Bernoulli blocks; no automatic group selection.",
        "values": "'binary' explicitly reduces positive aggregate weights to edge presence; self-loops are excluded.",
        "initial": "Optional complete initial node-to-block membership assignment.",
        "seed": "Private CPU Torch seed for reproducible starting partitions.",
        "starts": "Number of starting partitions; the highest profile likelihood is retained.",
        "max_iter": "Maximum coordinate-ascent sweeps; convergence is reported explicitly.",
        "tol": "Minimum accepted likelihood improvement; no global optimum is claimed.",
        "max_work": "Explicit structural work budget; exhaustion fails without returning a partial fit.",
    },
    "openecon.Network.poisson_block_model": {
        "groups": "Fixed number of nonempty Poisson blocks; no automatic group selection.",
        "initial": "Optional complete initial node-to-block assignment.",
        "seed": "Private CPU Torch seed for reproducible starting partitions.",
        "starts": "Number of starts; retain the highest complete Poisson likelihood.",
        "max_iter": "Maximum sparse coordinate-ascent sweeps; report convergence explicitly.",
        "tol": "Minimum accepted local profile-likelihood gain, with a numerical rounding guard.",
        "max_work": "Explicit complete-fit work budget; no partial fit on exhaustion.",
    },
    "openecon.Network.degree_corrected_block_model": {
        "groups": "Fixed number of nonempty degree-corrected Poisson groups.",
        "initial": "Optional complete initial node-to-block assignment.",
        "seed": "Private CPU Torch seed for reproducible starting partitions.",
        "starts": "Number of starts; retain the highest complete Poisson likelihood.",
        "max_iter": "Maximum sparse coordinate-ascent sweeps; report convergence explicitly.",
        "tol": "Minimum accepted local profile-likelihood gain, with a numerical rounding guard.",
        "max_work": "Explicit complete-fit work budget; no partial fit on exhaustion.",
    },
    "openecon.NetworkBlockResult.expected_edges": {
        "pairs": "Explicit (source, target) pairs or a source/target table; preserve exact typed IDs, order and duplicates. Only degree-corrected Poisson supports loops.",
        "max_pairs": "Maximum complete output size; oversized requests fail without silent truncation.",
        "max_memory_mb": "Budget for resident fit tables and planned indexing/output; excludes caller input and process RSS.",
    },
    "openecon.network_snapshots": {
        "layers": "Insertion-ordered mapping from exact integer/string layer labels to Network snapshots.",
        "ordered": "True explicitly treats layer order as time; required for temporal_path.",
        "max_work": "Explicit structural work budget for snapshot admission.",
    },
    "openecon.Network.qap_correlation": {
        "other": "A network with the same exact node-label set, isolates and directedness.",
        "values": "'weight' uses aggregate strengths; 'binary' uses edge presence. Absent dyads are zero.",
        "include_loops": "Include self-loop dyads; False excludes them from the correlation.",
        "permutations": "Number of uniform node-label Monte Carlo permutations; p-values use a plus-one correction.",
        "seed": "Private CPU Torch random seed with canonical typed-node order.",
        "alternative": "'two-sided' compares absolute correlations; 'greater' or 'less' tests one tail.",
        "max_work": "Explicit structural work budget; an oversized complete test is refused.",
    },
    "openecon.Network.global_min_cut": {
        "max_work": "Explicit traversal and integer word-operation budget for exact sparse contractions or directed rooted flows.",
    },
    "openecon.Network.triad_census": {
        "max_work": "Explicit structural budget for setup and forward triangle intersections.",
    },
    "openecon.ols": {
        "data": "A DataFrame, column dictionary, row records or batch Dataset source.",
        "y": "Outcome column name; cannot be combined with formula.",
        "x": "Predictor column names; cannot be combined with formula.",
        "formula": "Example: 'wage ~ education + experience + C(region)'.",
        "covariance": "Standard error method, such as 'nonrobust', 'HC3', 'cluster' or 'hac'.",
        "categorical": "Predictor names to encode as categorical variables.",
        "intercept": "Include an intercept.",
        "cluster": "Column name or names for clustered standard errors.",
        "weights": "Column name containing the weights.",
        "weight_type": "'aweight', 'fweight', 'pweight' or 'iweight'.",
        "missing": "'drop' removes observations with missing values; 'raise' reports an error.",
        "alpha": "Inference significance level; 0.05 gives a 95% confidence interval.",
        "time": "Time column used for HAC or lagged formulas.",
        "lags": "HAC lag count or a supported automatic selection name.",
        "kernel": "HAC weighting kernel.",
        "reps": "Replication option for bootstrap or jackknife.",
        "seed": "Random seed for resampling.",
        "device": "'auto', 'cpu', 'cuda', 'cuda:<index>' or 'mps'. CUDA uses float64; Metal preconditions bounded factors with checked CPU float64 refinement.",
    },
    "openecon.install": {
        "requirements": "Package names or exact versions; for example, 'pandas==2.3.3'.",
        "installer": "'pip' or 'uv'.",
        "requirements_file": "Path to the requirements.txt file within the project.",
        "upgrade": "Upgrade installed packages to the requested versions.",
    },
    "openecon.DataFrame.to_latex": {
        "buf": "File path or writable buffer; None returns the LaTeX source.",
        "index": "Show the row index in the table.",
        "caption": "Table caption.",
        "label": "LaTeX cross-reference label.",
        "precision": "Number of decimal places for numeric values.",
        "notes": "Notes to show below the table.",
    },
    "openecon.ResultBundle.summary": {"format": "'text' or 'latex'."},
}


class SourceReader:
    def __init__(self, ref: str, pandas_root: Path | None = None):
        self.ref = ref
        self.cache: dict[str, ast.Module] = {}
        self.pandas_root = pandas_root

    def committed(self, path: str) -> ast.Module:
        if path not in self.cache:
            raw = subprocess.check_output(
                ["git", "show", f"{self.ref}:{path}"], cwd=ROOT, text=True,
            )
            self.cache[path] = ast.parse(raw, filename=path)
        return self.cache[path]

    def pandas(self, path: str) -> ast.Module:
        if self.pandas_root is not None:
            source = self.pandas_root / path.removeprefix("pandas/")
        else:
            source = Path(importlib.metadata.distribution("pandas").locate_file(path))
        return ast.parse(source.read_text(encoding="utf-8"), filename=path)

    def installed(self, package: str, path: str) -> ast.Module:
        source = Path(importlib.metadata.distribution(package).locate_file(path))
        return ast.parse(source.read_text(encoding="utf-8"), filename=path)


def class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def export_literal(node: ast.expr, names: dict[str, Any] | None = None) -> Any:
    """Read literal export mappings, including bounded name comprehensions.

    Calls, attributes and arbitrary expressions are deliberately unsupported.
    This is syntax inspection, never Python evaluation.
    """
    names = names or {}
    if isinstance(node, ast.Name) and node.id in names:
        value = names[node.id]
        return export_literal(value, names) if isinstance(value, ast.expr) else value
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [export_literal(item, names) for item in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(part.value)
            elif (isinstance(part, ast.FormattedValue) and part.conversion == -1
                  and part.format_spec is None):
                value = export_literal(part.value, names)
                if not isinstance(value, str):
                    raise ValueError("Export interpolation requires a literal string")
                parts.append(value)
            else:
                raise ValueError("Only literal export interpolation is supported")
        return "".join(parts)
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                result.update(export_literal(value, names))
            else:
                result[export_literal(key, names)] = export_literal(value, names)
        return result
    if isinstance(node, ast.DictComp) and len(node.generators) == 1:
        generator = node.generators[0]
        if not isinstance(generator.target, ast.Name) or generator.ifs or generator.is_async:
            raise ValueError("Only literal public-name export comprehensions are supported")
        values = export_literal(generator.iter, names)
        if not isinstance(values, (list, tuple)) or len(values) > 100:
            raise ValueError("Public-name export lists must be short literals")
        result = {}
        for value in values:
            bound = {**names, generator.target.id: value}
            result[export_literal(node.key, bound)] = export_literal(node.value, bound)
        return result
    raise ValueError(f"Unsupported export syntax: {type(node).__name__}")


def registry_exports(reader: SourceReader) -> dict[str, tuple[str, str, str | None]]:
    """Inspect committed manifests, including their bounded metadata factories.

    Only the literal function/entry/description fields of EstimatorInfo are
    read. A manifest factory must consist of one Return of EstimatorInfo;
    none of its option expressions or Python code is evaluated or imported.
    """
    registry = reader.committed("src/openecon/econometrics/registry.py")
    families = next(ast.literal_eval(node.value) for node in registry.body
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                    and ((isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "FAMILIES"
                        for target in node.targets))
                         or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                             and node.target.id == "FAMILIES")))
    exports = {"forecast": ("openecon.econometrics.core", "forecast", None)}

    def publish(name, target, description=None):
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise ValueError("Registry exports require public identifiers")
        if (not isinstance(target, str) or target.count(":") != 1
                or not target.startswith("openecon.econometrics.")):
            raise ValueError(f"Invalid registry target for {name}")
        module, function = target.split(":")
        if not all(part.isidentifier() for part in module.split(".")) or not function.isidentifier():
            raise ValueError(f"Invalid registry target for {name}")
        if name in exports:
            raise ValueError(f"Duplicate registry export: {name}")
        exports[name] = (module, function, description)

    for family in families:
        tree = reader.committed(f"src/openecon/econometrics/{family}/__init__.py")
        names = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                 for target in node.targets if isinstance(target, ast.Name)}
        factories = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

        def estimator(call, bindings):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                raise ValueError("Estimator metadata must be a literal constructor")
            if call.func.id != "EstimatorInfo":
                factory = factories.get(call.func.id)
                if (factory is None or len(factory.body) != 1
                        or not isinstance(factory.body[0], ast.Return)
                        or not isinstance(factory.body[0].value, ast.Call)
                        or not isinstance(factory.body[0].value.func, ast.Name)
                        or factory.body[0].value.func.id != "EstimatorInfo"
                        or factory.args.vararg or factory.args.kwarg):
                    raise ValueError("Only single-return EstimatorInfo factories are supported")
                arguments = [*factory.args.posonlyargs, *factory.args.args]
                if any(isinstance(value, ast.Starred) for value in call.args) or any(
                    keyword.arg is None for keyword in call.keywords
                ):
                    raise ValueError("Manifest factory calls cannot unpack arguments")
                if len(call.args) > len(arguments):
                    raise ValueError("Manifest factory has too many positional arguments")
                bound = {arg.arg: value for arg, value in zip(arguments, call.args)}
                positional_only = {arg.arg for arg in factory.args.posonlyargs}
                accepted = {arg.arg for arg in [*factory.args.args, *factory.args.kwonlyargs]}
                for keyword in call.keywords:
                    if keyword.arg not in accepted or keyword.arg in positional_only:
                        raise ValueError("Manifest factory has an unknown or positional-only keyword")
                    if keyword.arg in bound:
                        raise ValueError("Manifest factory has duplicate argument bindings")
                    bound[keyword.arg] = keyword.value
                defaults = dict(zip(
                    [arg.arg for arg in arguments[len(arguments) - len(factory.args.defaults):]],
                    factory.args.defaults,
                ))
                defaults.update({arg.arg: value for arg, value in zip(
                    factory.args.kwonlyargs, factory.args.kw_defaults,
                ) if value is not None})
                for argument in [*arguments, *factory.args.kwonlyargs]:
                    if argument.arg not in bound:
                        if argument.arg not in defaults:
                            raise ValueError("Manifest factory is missing a required argument")
                        bound[argument.arg] = defaults[argument.arg]
                local = {**bindings, **bound}
                return estimator(factory.body[0].value, local)
            fields = {kw.arg: kw.value for kw in call.keywords}
            function = export_literal(fields.get("function", ast.Constant(None)), bindings)
            if function is not None:
                target = export_literal(fields["entry"], bindings).split(":")[0] + ":" + function
                description = export_literal(fields.get("description", ast.Constant(None)), bindings)
                publish(function, target, summary(description) if description else None)

        def entries(value):
            if isinstance(value, (ast.List, ast.Tuple)):
                for item in value.elts:
                    estimator(item, names)
            elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                # A repeated ESTIMATORS assignment appends an explicit tuple.
                if not isinstance(value.left, ast.Name) or value.left.id != "ESTIMATORS":
                    raise ValueError("Only literal estimator tuple appends are supported")
                entries(value.right)
            else:
                raise ValueError("Estimator inventory must be literal tuples")

        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
                node, (ast.AnnAssign, ast.AugAssign)) else []
            identifiers = {target.id for target in targets if isinstance(target, ast.Name)}
            if "ESTIMATORS" in identifiers:
                if isinstance(node, ast.AugAssign) and not isinstance(node.op, ast.Add):
                    raise ValueError("Only estimator tuple addition is supported")
                entries(node.value)
            if "EXPORTS" in identifiers:
                for name, target in export_literal(node.value, names).items():
                    publish(name, target)
    return exports


def function_node(nodes: list[ast.stmt], name: str) -> ast.FunctionDef:
    # pandas has @overload stubs followed by the actual implementation.
    return next(node for node in reversed(nodes) if isinstance(node, ast.FunctionDef) and node.name == name)


def methods(node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        child.name: child
        for child in node.body
        if isinstance(child, ast.FunctionDef)
        and not child.name.startswith("_")
        and not any(isinstance(item, ast.Name) and item.id == "property" for item in child.decorator_list)
    }


def summary(doc: str) -> str:
    paragraph = doc.strip().split("\n\n", 1)[0]
    paragraph = re.sub(r"\s+", " ", paragraph)
    return paragraph[:500]


def numpy_parameter_docs(doc: str) -> dict[str, str]:
    """Read the first description paragraph from a NumPy-style docstring."""
    lines = inspect.cleandoc(doc).splitlines()
    result: dict[str, str] = {}
    active: list[str] = []
    words: list[str] = []
    in_parameters = False

    def flush() -> None:
        if active and words:
            for name in active:
                result[name.lstrip("*")] = " ".join(words)[:300]

    for line in lines:
        if line == "Parameters":
            in_parameters = True
            continue
        if not in_parameters or line and set(line) == {"-"}:
            continue
        if line and not line[0].isspace():
            flush()
            words = []
            if ":" not in line:
                break
            active = [part.strip() for part in line.split(":", 1)[0].split(",")]
        elif active and line.strip():
            # A paragraph break ends the concise text, leaving examples out.
            words.append(line.strip())
        elif words:
            flush()
            active = []
            words = []
    flush()
    return result


def arguments(node: ast.FunctionDef, *, bound: bool = False) -> ast.arguments:
    args = deepcopy(node.args)
    if bound:
        if args.posonlyargs and args.posonlyargs[0].arg in {"self", "cls"}:
            args.posonlyargs.pop(0)
        elif args.args and args.args[0].arg in {"self", "cls"}:
            args.args.pop(0)
    return args


def parameter_list(args: ast.arguments, docs: dict[str, str]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    positions = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positions) - len(args.defaults)) + list(args.defaults)

    def add(arg: ast.arg, kind: str, default: ast.expr | None = None) -> None:
        item = {"name": arg.arg, "kind": kind}
        if default is not None:
            item["default"] = ast.unparse(default)
        if arg.arg in docs:
            item["description"] = display_description(docs[arg.arg])
        output.append(item)

    for index, (arg, default) in enumerate(zip(positions, defaults)):
        add(arg, "positional-only" if index < len(args.posonlyargs) else "positional-or-keyword", default)
    if args.vararg:
        add(args.vararg, "var-positional")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        add(arg, "keyword-only", default)
    if args.kwarg:
        add(args.kwarg, "var-keyword")
    return output


def display_description(text: str) -> str:
    """Use the application brand in prose while preserving Python API names."""
    return re.sub(r"\bOpenEcon\b", "OpenEconometrics", text)


def entry(
    canonical: str, node: ast.FunctionDef, *, bound: bool = False,
    kind: str = "function", returns: str | None = None, description: str | None = None,
) -> dict[str, Any]:
    args = arguments(node, bound=bound)
    doc = ast.get_docstring(node) or ""
    parameter_docs = numpy_parameter_docs(doc)
    parameter_docs.update(PARAMETER_DESCRIPTIONS.get(canonical, {}))
    text = description or DESCRIPTIONS.get(canonical) or summary(doc)
    if not text:
        raise ValueError(f"A verified description is required for {canonical}")
    result: dict[str, Any] = {
        "name": canonical,
        "signature": f"{canonical.rsplit('.', 1)[-1]}({ast.unparse(args)})",
        "description": display_description(text),
        "parameters": parameter_list(args, parameter_docs),
        "kind": kind,
    }
    if kind == "method":
        result["owner"] = canonical.rsplit(".", 1)[0]
    inferred = returns or (ast.unparse(node.returns) if node.returns is not None else None)
    if inferred:
        result["returns"] = inferred
    return result


def parameter_choices(reader: SourceReader) -> dict[str, dict[str, list[str]]]:
    """Read accepted literals from published syntax, without loading estimators."""
    spec = reader.committed("src/openecon/linear_ols/spec.py")
    constants = {
        target.id: ast.literal_eval(node.value)
        for node in spec.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"COVARIANCES", "WEIGHTS", "KERNELS"}
    }
    model = class_node(reader.committed("src/openecon/models.py"), "ModelSpec")
    missing = next(
        node.annotation.slice
        for node in model.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name) and node.target.id == "missing"
        and isinstance(node.annotation, ast.Subscript)
        and isinstance(node.annotation.value, ast.Name) and node.annotation.value.id == "Literal"
    )
    binary_covariance = next(
        node.comparators[0]
        for node in ast.walk(function_node(model.body, "validate_relationships"))
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute) and node.left.attr == "covariance"
        and isinstance(node.left.value, ast.Name) and node.left.value.id == "self"
        and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotIn)
        and isinstance(node.comparators[0], ast.Set)
    )

    def literals(values: Any) -> list[str]:
        if not isinstance(values, (list, tuple, set)) or not 0 < len(values) <= 32:
            raise ValueError("Parameter choices must be bounded literal collections")
        if not all(isinstance(value, str) and len(value) <= 100 for value in values):
            raise ValueError("Parameter choices must be short strings")
        return [repr(value) for value in (sorted(values) if isinstance(values, set) else values)]

    missing_choices = literals(ast.literal_eval(missing))
    binary_choices = {
        "covariance": literals(ast.literal_eval(binary_covariance)), "missing": missing_choices,
    }
    return {
        "openecon.ols": {
            "covariance": literals(constants["COVARIANCES"]),
            "weight_type": literals(constants["WEIGHTS"]),
            "kernel": literals(constants["KERNELS"]),
            "missing": missing_choices,
        },
        "openecon.logit": binary_choices,
        "openecon.probit": binary_choices,
    }


def generate(ref: str = "HEAD", pandas_root: Path | None = None) -> list[dict[str, Any]]:
    reader = SourceReader(ref, pandas_root)
    catalog: dict[str, dict[str, Any]] = {}

    def add(item: dict[str, Any]) -> None:
        catalog[item["name"]] = item

    init = reader.committed("src/openecon/__init__.py")
    exports = next(
        export_literal(node.value)
        for node in init.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_EXPORTS" for target in node.targets)
    )
    public_functions = {
        "fit", "ols", "logit", "probit", "read", "scan", "example", "load_dataset",
        "latex", "to_latex", "regression_table", "install", "capabilities",
        "test", "testparm", "lincom", "nlcom", "predict", "margins", "network", "read_network", "network_snapshots", "signed_network",
        "multigraph", "read_multigraph", "dynamic_network", "read_dynamic_network",
        "ergm", "simulate_ergm", "saom", "simulate_saom", "gaussian_block_model",
        "mixed_membership_block_model", "network_embedding", "network_gnn",
        "multilayer_network", "read_multilayer_network",
        "hypergraph", "read_hypergraph",
    }
    return_types = {
        "bai_perron": "openecon.TableSet",
        "fit": "openecon.ResultBundle", "ols": "openecon.OLSResult",
        "logit": "openecon.ResultBundle", "probit": "openecon.ResultBundle",
        "read": "openecon.DataFrame", "example": "openecon.DataFrame",
        "load_dataset": "openecon.DataFrame", "scan": "openecon.Dataset",
        "predict": "openecon.DataFrame", "margins": "openecon.DataFrame",
        "xtcd": "openecon.DataFrame", "nardl_multipliers": "openecon.DataFrame",
        "ngperron": "openecon.DataFrame", "kss": "openecon.DataFrame",
        "network": "openecon.Network",
        "signed_network": "openecon.SignedNetwork",
        "multigraph": "openecon.MultiNetwork",
        "read_multigraph": "openecon.MultiNetwork",
        "dynamic_network": "openecon.DynamicNetwork",
        "read_dynamic_network": "openecon.DynamicNetwork",
        "multilayer_network": "openecon.MultilayerNetwork",
        "read_multilayer_network": "openecon.MultilayerNetwork",
        "hypergraph": "openecon.Hypergraph",
        "read_hypergraph": "openecon.Hypergraph",
        "read_network": "openecon.Network",
        "network_snapshots": "openecon.NetworkSnapshots",
    }
    post = class_node(reader.committed("src/openecon/linear_ols/postestimation.py"), "OLSPostestimation")
    for name in sorted(public_functions):
        module, original = exports[name]
        path = "src/" + module.replace(".", "/") + ".py"
        if module == "openecon.linear_ols":
            # These wrappers simply forward to the model method. Read the
            # delegated parameter contract rather than inventing **kwargs.
            method = deepcopy(function_node(post.body, original))
            method.args.args[0].arg = "result"
            add(entry(f"openecon.{name}", method))
        else:
            tree = reader.committed(path)
            item = entry(f"openecon.{name}", function_node(tree.body, original), returns=return_types.get(name))
            if module == "openecon.analysis" and name in {"predict", "margins"}:
                # Keep the dispatcher signature exact. Its **kwargs accepts
                # these source-derived saved-adapter options; do not hide them
                # from keyword completion, or invent a shared kind default.
                delegated = function_node(reader.committed(
                    "src/openecon/econometrics/postest/prediction.py"
                ).body, original)
                options = parameter_list(delegated.args, {})[1:]
                native = {option["name"]: option for option in parameter_list(
                    function_node(post.body, original).args, {}
                )}
                for option in options:
                    option["kind"] = "keyword-only"
                    if option["name"] == "kind":
                        option.pop("default", None)
                        option["description"] = "Native OLS defaults to xb; saved adapters default to response. Hetprobit also supports sigma."
                    elif option["name"] == "outcome":
                        option["description"] = "Saved categorical models: actual fitted outcome label; None selects all. Simultaneous quantile regression: required fitted quantile equation."
                    elif option["name"] == "batch_rows":
                        option["description"] = "Dataset evaluation only: 1 to 65,536 rows per block; None chooses a bounded workspace plan."
                    elif option["name"] == "data":
                        option["description"] = "Explicit DataFrame or replayable Dataset evaluation rows; fitted chart previews are not reused."
                    elif (option["name"] in native
                          and option.get("default") != native[option["name"]].get("default")):
                        option.pop("default", None)
                    option.setdefault("description", "Forwarded option; support depends on the fitted model.")
                item["parameters"] = [item["parameters"][0], *options, *item["parameters"][1:]]
            add(item)

    # Lazy registry functions are shipped public APIs too. Their signatures
    # are read from the same committed revision without importing the tensor
    # runtime, so model help remains available offline and before execution.
    lazy_exports = registry_exports(reader)
    collisions = sorted(set(lazy_exports) & set(exports))
    if collisions:
        raise ValueError("Registry exports collide with the core API: " + ", ".join(collisions))
    for name, (module, original, description) in sorted(lazy_exports.items()):
        tree = reader.committed("src/" + module.replace(".", "/") + ".py")
        node = function_node(tree.body, original)
        inferred = return_types.get(name) or ("openecon.ResultBundle" if description is not None else None)
        add(entry(f"openecon.{name}", node, returns=inferred, description=description))

    charts = reader.committed("packages/openecon-charts/src/openecon_charts/charts.py")
    plotting = reader.committed("src/openecon/plotting.py")
    chart_exports = next(ast.literal_eval(node.value) for node in plotting.body if isinstance(node, ast.Assign))
    for name in chart_exports:
        if name == "PlotSpec":
            continue
        chart_source = (reader.committed("packages/openecon-charts/src/openecon_charts/network.py")
                        if name == "network" else charts)
        add(entry(f"openecon.plot.{name}", function_node(chart_source.body, name), returns="openecon.plot.PlotSpec"))
    for name, node in methods(class_node(charts, "PlotSpec")).items():
        description = summary(ast.get_docstring(node) or "")
        if description:
            add(entry(f"openecon.plot.PlotSpec.{name}", node, bound=True, kind="method"))

    network_tree = reader.committed("src/openecon/networks.py")
    for name, node in methods(class_node(network_tree, "Network")).items():
        if name.startswith("_") or any(isinstance(item, ast.Name) and item.id == "property"
                                        for item in node.decorator_list):
            continue
        description = summary(ast.get_docstring(node) or "")
        if description:
            returns = ("openecon.DataFrame" if name in {"degree", "pagerank", "components", "shortest_paths", "summary",
                                                      "communities", "betweenness", "closeness", "harmonic", "eigenvector",
                                                      "triangles", "clustering", "core_numbers", "topology_summary",
                                                      "hits", "katz", "shortest_path", "distances", "eccentricity",
                                                      "distance_summary", "bridges", "articulation_points", "bipartite",
                                                      "maximum_matching", "link_prediction", "triad_census",
                                                      "qap_correlation", "qap_regression", "nodes", "edges",
                                                      "k_shortest_paths", "strong_bridges", "strong_articulation_points"}
                       else "openecon.Network" if name in {"subgraph", "with_attributes", "minimum_spanning_forest",
                                                          "bipartite_projection", "edit_nodes", "edit_edges", "filter",
                                                          "update_attributes", "rename_attributes", "drop_attributes",
                                                          "with_positions", "from_plot_data"}
                       else "openecon.NetworkMatchingResult" if name in {"weighted_assignment", "general_matching"}
                       else "openecon.NetworkFlowResult" if name in {"max_flow", "min_cut"}
                       else "openecon.NetworkCutResult" if name == "global_min_cut"
                       else "openecon.NetworkBlockResult" if name in {"block_model", "poisson_block_model",
                                                                     "degree_corrected_block_model"} else None)
            add(entry(f"openecon.Network.{name}", node, bound=True, kind="method", returns=returns))

    multi_tree = reader.committed("src/openecon/_network_multi.py")
    for name, node in methods(class_node(multi_tree, "MultiNetwork")).items():
        if name.startswith("_") or any(isinstance(item, ast.Name) and item.id == "property"
                                        for item in node.decorator_list):
            continue
        returns = ("openecon.DataFrame" if name in {"nodes", "edges", "degree", "summary"}
                   else "openecon.MultiNetwork" if name in {"edit_nodes", "edit_edges", "filter", "with_attributes"}
                   else "openecon.NetworkMatchingResult" if name in {"weighted_assignment", "general_matching"}
                   else "openecon.Network" if name == "to_network" else None)
        add(entry(f"openecon.MultiNetwork.{name}", node, bound=True, kind="method", returns=returns))

    multilayer_tree = reader.committed("src/openecon/_network_multilayer.py")
    for name, node in methods(class_node(multilayer_tree, "MultilayerNetwork")).items():
        if name.startswith("_") or any(isinstance(item, ast.Name) and item.id == "property"
                                        for item in node.decorator_list):
            continue
        returns = ("openecon.DataFrame" if name in {"nodes", "edges", "layers", "matvec", "pagerank"}
                   else "openecon.MultiNetwork" if name == "layer"
                   else "openecon.MultilayerNetwork" if name == "edit_edges"
                   else "openecon.Network" if name in {"project", "supra_network"} else "pathlib.Path")
        add(entry(f"openecon.MultilayerNetwork.{name}", node, bound=True, kind="method", returns=returns))

    dynamic_tree = reader.committed("src/openecon/_network_dynamic.py")
    for name, node in methods(class_node(dynamic_tree, "DynamicNetwork")).items():
        if name.startswith("_") or any(isinstance(item, ast.Name) and item.id == "property"
                                        for item in node.decorator_list):
            continue
        returns = ("openecon.DataFrame" if name in {"nodes", "edges", "summary"}
                   else "openecon.MultiNetwork" if name in {"at", "window"} else "pathlib.Path")
        add(entry(f"openecon.DynamicNetwork.{name}", node, bound=True, kind="method", returns=returns))

    hyper_tree = reader.committed("src/openecon/_network_hypergraph.py")
    for name, node in methods(class_node(hyper_tree, "Hypergraph")).items():
        if name.startswith("_") or any(isinstance(item, ast.Name) and item.id == "property"
                                        for item in node.decorator_list):
            continue
        returns = ("openecon.DataFrame" if name in {"degree", "strength", "memberships", "summary"}
                   else "openecon.Network" if name in {"clique_projection", "star_projection"}
                   else "torch.Tensor" if name in {"incidence", "incidence_matvec"} else None)
        add(entry(f"openecon.Hypergraph.{name}", node, bound=True, kind="method", returns=returns))

    signed_tree = reader.committed("src/openecon/_network_signed.py")
    for name, node in methods(class_node(signed_tree, "SignedNetwork")).items():
        if name in {"weighted_assignment", "general_matching"}:
            add(entry(f"openecon.SignedNetwork.{name}", node, bound=True, kind="method",
                      returns="openecon.NetworkMatchingResult"))
        if name.startswith("signed_"):
            add(entry(f"openecon.SignedNetwork.{name}", node, bound=True, kind="method",
                      returns="float" if name == "signed_modularity" else "openecon.DataFrame"))

    matching_tree = reader.committed("src/openecon/_network_matching.py")
    for name, node in methods(class_node(matching_tree, "NetworkMatchingResult")).items():
        if name in {"summary", "to_latex"}:
            add(entry(f"openecon.NetworkMatchingResult.{name}", node, bound=True, kind="method",
                      returns="openecon.DataFrame" if name == "summary" else None))

    flow_tree = reader.committed("src/openecon/_network_flow.py")
    for name, node in methods(class_node(flow_tree, "NetworkFlowResult")).items():
        if not name.startswith("_"):
            add(entry(f"openecon.NetworkFlowResult.{name}", node, bound=True, kind="method",
                      returns="openecon.DataFrame" if name == "summary" else None))

    cut_tree = reader.committed("src/openecon/_network_cut.py")
    for name, node in methods(class_node(cut_tree, "NetworkCutResult")).items():
        if not name.startswith("_"):
            add(entry(f"openecon.NetworkCutResult.{name}", node, bound=True, kind="method",
                      returns="openecon.DataFrame" if name == "summary" else None))

    block_tree = reader.committed("src/openecon/_network_sbm.py")
    for name, node in methods(class_node(block_tree, "NetworkBlockResult")).items():
        if not name.startswith("_"):
            add(entry(f"openecon.NetworkBlockResult.{name}", node, bound=True, kind="method",
                      returns="openecon.DataFrame" if name in {"summary", "expected_edges"} else None))

    snapshot_tree = reader.committed("src/openecon/_network_temporal.py")
    for name, node in methods(class_node(snapshot_tree, "NetworkSnapshots")).items():
        if name.startswith("_") or any(isinstance(item, ast.Name) and item.id == "property"
                                        for item in node.decorator_list):
            continue
        description = summary(ast.get_docstring(node) or "")
        if description:
            returns = ("openecon.Network" if name == "aggregate"
                       else "openecon.DataFrame" if name in {"summary", "snapshot_summary", "transitions",
                                                            "edge_persistence", "temporal_path"} else None)
            add(entry(f"openecon.NetworkSnapshots.{name}", node, bound=True, kind="method", returns=returns))

    pandas_frame = class_node(reader.pandas("pandas/core/frame.py"), "DataFrame")
    generic = class_node(reader.pandas("pandas/core/generic.py"), "NDFrame")
    own_frame = class_node(reader.committed("src/openecon/frame.py"), "DataFrame")
    add(entry("openecon.DataFrame", function_node(pandas_frame.body, "__init__"), bound=True, kind="class", returns="openecon.DataFrame"))
    frame_methods = {**methods(generic), **methods(pandas_frame), **methods(own_frame)}
    guaranteed_frames = {"head", "tail", "describe", "copy", "sample", "assign", "merge", "isna", "notna", "corr"}
    for name in FRAME_METHODS:
        description = None
        if not ast.get_docstring(frame_methods[name]) and f"openecon.DataFrame.{name}" not in DESCRIPTIONS:
            description = summary(ast.get_docstring(methods(generic).get(name, frame_methods[name])) or "") or None
        add(entry(
            f"openecon.DataFrame.{name}", frame_methods[name], bound=True, kind="method",
            returns="openecon.DataFrame" if name in guaranteed_frames else None,
            description=description,
        ))
    add(entry("openecon.DataFrame.to_latex", frame_methods["to_latex"], bound=True, kind="method"))

    model_tree = reader.committed("src/openecon/models.py")
    pydantic_init = function_node(class_node(reader.installed("pydantic", "pydantic/main.py"), "BaseModel").body, "__init__")
    for name in ("ModelSpec", "ResultBundle"):
        # Pydantic accepts model fields through its actual **data constructor.
        # Do not pretend that AST field annotations are ordinary Python args.
        add(entry(f"openecon.{name}", pydantic_init, bound=True, kind="class", returns=f"openecon.{name}"))
    result = class_node(model_tree, "ResultBundle")
    for name in ("summary", "to_latex"):
        item = entry(f"openecon.ResultBundle.{name}", function_node(result.body, name), bound=True, kind="method")
        add(item)
        alias = deepcopy(item)
        alias["name"] = f"openecon.OLSResult.{name}"
        alias["owner"] = "openecon.OLSResult"
        add(alias)
    for name, node in methods(post).items():
        add(entry(f"openecon.OLSResult.{name}", node, bound=True, kind="method"))
    linear = reader.committed("src/openecon/linear_ols/__init__.py")
    ols_class = next(node for node in ast.walk(linear) if isinstance(node, ast.ClassDef) and node.name == "OLSResult")
    for name in ("iter_predict", "iter_influence"):
        node = function_node(ols_class.body, name)
        add(entry(
            f"openecon.OLSResult.{name}", node, bound=True, kind="method",
            description="Return OLS influence statistics in bounded batches." if name == "iter_influence" else None,
        ))

    dataset = class_node(reader.committed("src/openecon/dataset.py"), "Dataset")
    add(entry("openecon.Dataset", function_node(dataset.body, "__init__"), bound=True, kind="class", returns="openecon.Dataset"))
    transforms = {"project", "filter", "map", "join", "reshape_long", "reshape_wide"}
    for name in ("from_frame", "from_batches", "head", "iter_batches", "close", "iter_missing_codes", *sorted(transforms)):
        add(entry(
            f"openecon.Dataset.{name}", function_node(dataset.body, name), bound=True, kind="method",
            returns="openecon.DataFrame" if name == "head" else "openecon.Dataset" if name.startswith("from_") or name in transforms else None,
        ))

    for name in BUILTINS:
        function = getattr(builtins, name)
        signature = inspect.signature(function)
        params = []
        for parameter in signature.parameters.values():
            param = {"name": parameter.name, "kind": parameter.kind.name.lower().replace("_", "-")}
            if parameter.default is not inspect.Parameter.empty:
                param["default"] = repr(parameter.default)
            params.append(param)
        add({
            "name": f"builtins.{name}", "signature": f"{name}{signature}",
            "description": summary(inspect.getdoc(function) or ""),
            "parameters": params, "kind": "function",
        })
    worker = reader.committed("src/openecon/console_worker.py")
    display = next(node for node in ast.walk(worker) if isinstance(node, ast.FunctionDef) and node.name == "display")
    add(entry("console.display", display, returns="None"))
    latex_class = class_node(reader.committed("src/openecon/latex.py"), "Latex")
    add(entry("openecon.Latex", function_node(latex_class.body, "__new__"), bound=True, kind="class", returns="openecon.Latex"))
    for name, choices in parameter_choices(reader).items():
        for parameter in catalog[name]["parameters"]:
            if parameter["name"] in choices:
                parameter["choices"] = choices[parameter["name"]]
    return [catalog[name] for name in sorted(catalog)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", "--revision", dest="ref", default="HEAD", help="Committed Git ref for the published openecon API")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--pandas-root", type=Path, help="Optional installed pandas package directory")
    parser.add_argument("--check", action="store_true", help="Check the saved catalog without writing")
    args = parser.parse_args()
    catalog = generate(args.ref, args.pandas_root)
    if any(name in sys.modules for name in ("openecon", "openecon_charts", "pandas", "torch")):
        raise RuntimeError("Catalog generation must never import statistical or project code")
    if args.check:
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        if saved != catalog:
            raise SystemExit("Editor API catalog is stale; regenerate it with the packaging Python.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized = "[\n" + ",\n".join(
            "  " + json.dumps(item, ensure_ascii=False, separators=(",", ": "))
            for item in catalog
        ) + "\n]\n"
        args.output.write_text(serialized, encoding="utf-8")
    print(f"Verified {len(catalog)} editor API entries from committed source.")


if __name__ == "__main__":
    main()
