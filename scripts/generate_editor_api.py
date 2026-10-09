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
STORED_PARAMETER_KINDS = {
    "keyword-only": "k", "positional-only": "p",
    "var-positional": "*", "var-keyword": "**",
}
STORED_ENTRY_KINDS = {"method": "m", "class": "c"}
STORED_DEFAULT_LITERALS = ("'raise'", '0.05', "'cpu'", '100000000', '50000000', "'drop'", '0.95', '0', 'DEFAULT_BYTES', 'DEFAULT_WORK', "'two-sided'", "'nonrobust'", "'constant'", '1000000', '2000000000', '100000')
STORED_RETURN_TYPES = {"openecon.TableSet": 0, "openecon.ResultBundle": 1,
                       "openecon.DataFrame": 2}
# Fixed exact suffixes are reversible wire tokens, not synthesized signatures.
# Append only: a different installed Pydantic signature falls back to literal S.
STORED_SIGNATURE_SUFFIXES = (
    '(json_data: str | bytes | bytearray, *, strict: bool | None=None, extra: ExtraValues | None=None, context: Any | None=None, by_alias: bool | None=None, by_name: bool | None=None)',
    "(*, indent: int | None=None, ensure_ascii: bool=False, include: IncEx | None=None, exclude: IncEx | None=None, context: Any | None=None, by_alias: bool | None=None, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, exclude_computed_fields: bool=False, round_trip: bool=False, warnings: bool | Literal['none', 'warn', 'error']=True, fallback: Callable[[Any], Any] | None=None, serialize_as_any: bool=False, polymorphic_serialization: bool | None=None)",
    '(obj: Any, *, strict: bool | None=None, extra: ExtraValues | None=None, from_attributes: bool | None=None, context: Any | None=None, by_alias: bool | None=None, by_name: bool | None=None)',
    "(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None=None, covariance: str='robust', cluster: str | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05, max_iterations: int=100, tolerance: float=1e-09, max_work: int=10000000000, device: str='cpu')",
    "(data, design, outcome, regressors, *, intercept=True, domain=None, missing='raise', alpha=0.05, null=0.0, max_iter=100, tolerance=1e-09)",
    "(data, design, outcome, regressors, *, method, intercept=True, domain=None, missing='raise', alpha=0.05, null=0.0, max_iter=100, tolerance=1e-09, replicates=None, replicate_weights=None, centering='original', rho=None, justification=None, scale=None, rscales=None, df=None)",
    "(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None=None, covariance: str='robust', cluster: str | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05, max_iterations: int=100, tolerance: float=1e-09, max_work: int=10000000000, device: str='cpu', batch_rows: int | None=None)",
    "(*, data: Any, y: str, x: Sequence[str], panel: str, time: str | None=None, model: str='re', covariance: str | None=None, cluster: str | None=None, intpoints: int | None=None, intmethod: str | None=None, corr: str | None=None, corr_order: int | None=None, force: bool=False, offset: str | None=None, categorical: Sequence[str] | None=None, missing: str='raise', alpha: float=0.05)",
    "(*, data: Any, y: str, x: Sequence[str], inflate: Sequence[str], inflate_link: str='logit', covariance: str | None=None, cluster: str | Sequence[str] | None=None, weights: str | None=None, weight_type: str | None=None, offset: str | None=None, exposure: str | None=None, categorical: Sequence[str] | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05)",
    "(low, indicator, *, low_periods, high_periods, low_frequency='Y', high_frequency='Q', aggregation='sum', as_of=None, low_releases=None, high_releases=None, device='cpu', weights=None)",
    "(*, data, y, x, categorical=None, weights=None, weight_type='aweight', intercept=True, missing='raise', l1_ratio=0.5, selection='cv', penalty=None, lambda_path=None, n_lambdas=20, lambda_ratio=0.001, folds=5, seed=1729, standardize=True, penalty_factors=None, forced_controls=None, max_iterations=200, tolerance=1e-08, max_work=2000000000, device='cpu')",
    "(data, design, outcomes, *, domain=None, missing='raise', alpha=0.05, null=0.0)",
    '(*, data, items: list[str], points: int=41, max_iter: int=200, max_eval: int=400, tolerance: float=1e-06, level: float=0.95)',
    "(*, data: Any, y: str, x: Sequence[str] | None=None, group: str, random: Sequence[str] | None=None, intpoints: int=7, intmethod: str='mvaghermite', covariance: str | None=None, cluster: str | None=None, categorical: Sequence[str] | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05)",
    "(*, data: Any, y: str, x: Sequence[str], covariance: str | None=None, cluster: str | Sequence[str] | None=None, weights: str | None=None, weight_type: str | None=None, offset: str | None=None, categorical: Sequence[str] | None=None, missing: str='raise', alpha: float=0.05)",
    "(result: TableSet, data: Any, *, missing: str='raise', max_work: int=DEFAULT_WORK, max_bytes: int=DEFAULT_BYTES, device: str='cpu')",
    "(data: Any, raters: list[str], *, categories: list, agreement_weights: Any='unweighted', inference: str='none', level: float=0.95, missing: str='drop', device: str='cpu', weights: Any=None, max_fits: int=1024, max_work: int=100000000)",
    "(low, indicator, *, rho, low_periods, high_periods, low_frequency='Y', high_frequency='Q', aggregation='sum', as_of=None, low_releases=None, high_releases=None, device='cpu', weights=None, intercept=True, alpha=0.05)",
    "(*, data, x: str, y: str | Sequence[str], title: str | None=None, unit: str='', palette=None, aggregate: str | None=None, **options)",
    "(*, data: Any, y: str, x: Sequence[str], covariance: str | None=None, categorical: Sequence[str] | None=None, intercept: bool=True, cluster: str | None=None, missing: str='raise', alpha: float=0.05)",
    "(data, design, numerators, denominators, *, domain=None, missing='raise', alpha=0.05, null=0.0)",
    "(data, design, outcome, *, categories=None, domain=None, missing='raise', alpha=0.05, null=0.0)",
    "(result: ResultBundle, *, data, variable: str, order=1, interval=False, alpha=0.05, missing='raise', max_work=2000000000)",
    "(result: ResultBundle, *, data, variable: str, order=1, interval=True, alpha=0.05, missing='raise', max_work=2000000000)",
    '(**data: Any)',
    '(data, dimensions, count, *, levels, design, terms=None, structural=None, offset=None, max_iter=200, tol=1e-09, level=0.95, max_work=300000000, max_bytes=128 * 1024 ** 2)',
    "(lower, upper, *, times=None, level=0.95, device='cpu', weights=None, maxiter=500)",
    '(*, max_work=50000000)',
    "(*, data, y, x, panel, time, p=1, q=1, trend='c', max_iterations=500, missing='raise', alpha=0.05)",
    "(*, data, y, x, missing='raise', alpha=0.05, **options)",
    '(result: ResultBundle, steps: int, *, data: Any=None, exog: Any=None, alpha: float=0.05)',
    "(*, objective='weight', max_component_nodes=24, max_states=1000000, max_work=50000000)",
    "(partition=None, *, objective='weight', max_matrix_entries=1000000, max_work=50000000)",
    "(*, data, y, x, intercept=True, missing='raise', **options)",
    "(*, data, y, x, time=None, trend='c', kernel='bartlett', bandwidth=4, df_adjust=False, missing='raise', alpha=0.05)",
    "(data, *, kind='response', missing='raise', alpha=None)",
    "(*, data, y, x, missing='raise', **options)",
    '(data, outcome, predictors, *, key, spatial_weights, **options)',
    "(data, design, outcomes, *, domain=None, missing='raise', alpha=0.05, null=0.0, deff=False)",
    '(groups, *, initial=None, seed=0, starts=4, max_iter=100, tol=1e-09, max_work=50000000)',
    "(result, *, device='cpu', weights=None, max_work=DEFAULT_WORK, max_bytes=DEFAULT_BYTES)",
    "(time, event, *, causes=None, times=None, level=0.95, device='cpu', weights=None)",
    '(coefficients, *, null=0.0, alpha=None)',
    '(result, *args, **kwargs)',
    '(path, *, format=None, max_memory_mb=256, batch_rows=65536, max_file_mb=512)',
    '(*, nodes=None, edges=None, graph_attributes=None)',
    '(*, data, y, treatment, x, **options)',
    '(result: TableSet, data: Any)',
    "(data, columns, *, bounds, sampling_model, alpha=0.05, missing='raise')",
    '(coefficients, *, null=0.0)',
    '(path, *, overwrite=False)',
    "(data: Any, y: str, *, missing: str='drop')",
    "(*, direction='out', disconnected='infinite', max_work=50000000)",
    "(data, columns, *, sampling_model, alpha=0.05, missing='raise')",
    '(*, attributes=True)',
    '(*, data, x: str, y: str, title: str | None=None, **options)',
    '()',
    "(result, *, data, alignment=None, alpha=None, kind='mean')",
    '(source, target, *, max_work=50000000)',
    '(restrictions, *, null=None)',
    '(data, columns, *, max_iterations=500, tolerance=1e-08)',
    '(**options)',
    '(*, add=(), remove=(), weights=None, attributes=None)',
    '(max_nodes=1000, max_edges=5000, seed=0, groups=None)',
    '(buf=None, **kwargs)',
    "(data: Any, a: str, b: str, *, missing: str='drop')",
    '(result: ResultBundle, *, data, max_work=200000000)',
    "(data: pd.DataFrame, variables: list[str], *, knots: dict, components: int=2, n_starts: int=3, seed: int=0, maxiter: int=500, tol: float=1e-08, missing: str='drop', max_work: int=WORK, max_bytes: int=BYTES, device: str='cpu')",
)

# Fixed complete parameter lists; append only and match every field exactly.
STORED_PARAMETER_LISTS = ([{'name': 'json_data', 'kind': 'positional-or-keyword'}, {'name': 'strict', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'extra', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'context', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'by_alias', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'by_name', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'indent', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'ensure_ascii', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'include', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'exclude', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'context', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'by_alias', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'exclude_unset', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'exclude_defaults', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'exclude_none', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'exclude_computed_fields', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'round_trip', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'warnings', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'fallback', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'serialize_as_any', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'polymorphic_serialization', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'obj', 'kind': 'positional-or-keyword'}, {'name': 'strict', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'extra', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'from_attributes', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'context', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'by_alias', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'by_name', 'kind': 'keyword-only', 'default': 'None'}], [{'kind': 'keyword-only', 'name': 'data'}, {'kind': 'keyword-only', 'name': 'y'}, {'kind': 'keyword-only', 'name': 'endogenous'}, {'kind': 'keyword-only', 'name': 'instruments'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'x'}, {'default': "'robust'", 'kind': 'keyword-only', 'name': 'covariance'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'cluster'}, {'default': 'True', 'kind': 'keyword-only', 'name': 'intercept'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'default': '100', 'kind': 'keyword-only', 'name': 'max_iterations'}, {'default': '1e-09', 'kind': 'keyword-only', 'name': 'tolerance'}, {'default': '10000000000', 'kind': 'keyword-only', 'name': 'max_work'}, {'default': "'cpu'", 'kind': 'keyword-only', 'name': 'device'}], [{'kind': 'positional-or-keyword', 'name': 'data'}, {'kind': 'positional-or-keyword', 'name': 'design'}, {'kind': 'positional-or-keyword', 'name': 'outcome'}, {'kind': 'positional-or-keyword', 'name': 'regressors'}, {'default': 'True', 'kind': 'keyword-only', 'name': 'intercept'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'domain'}, {'choices': ["'drop'", "'raise'"], 'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'default': '0.0', 'kind': 'keyword-only', 'name': 'null'}, {'default': '100', 'kind': 'keyword-only', 'name': 'max_iter'}, {'default': '1e-09', 'kind': 'keyword-only', 'name': 'tolerance'}], [{'kind': 'positional-or-keyword', 'name': 'data'}, {'kind': 'positional-or-keyword', 'name': 'design'}, {'kind': 'positional-or-keyword', 'name': 'outcome'}, {'kind': 'positional-or-keyword', 'name': 'regressors'}, {'kind': 'keyword-only', 'name': 'method'}, {'default': 'True', 'kind': 'keyword-only', 'name': 'intercept'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'domain'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'default': '0.0', 'kind': 'keyword-only', 'name': 'null'}, {'default': '100', 'kind': 'keyword-only', 'name': 'max_iter'}, {'default': '1e-09', 'kind': 'keyword-only', 'name': 'tolerance'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'replicates'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'replicate_weights'}, {'default': "'original'", 'kind': 'keyword-only', 'name': 'centering'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'rho'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'justification'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'scale'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'rscales'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'df'}], [{'kind': 'positional-or-keyword', 'name': 'low'}, {'kind': 'positional-or-keyword', 'name': 'indicator'}, {'kind': 'keyword-only', 'name': 'low_periods'}, {'kind': 'keyword-only', 'name': 'high_periods'}, {'default': "'Y'", 'kind': 'keyword-only', 'name': 'low_frequency'}, {'default': "'Q'", 'kind': 'keyword-only', 'name': 'high_frequency'}, {'default': "'sum'", 'kind': 'keyword-only', 'name': 'aggregation'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'as_of'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'low_releases'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'high_releases'}, {'default': "'cpu'", 'kind': 'keyword-only', 'name': 'device'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'weights'}], [{'kind': 'positional-or-keyword', 'name': 'data'}, {'kind': 'positional-or-keyword', 'name': 'design'}, {'kind': 'positional-or-keyword', 'name': 'outcome'}, {'kind': 'positional-or-keyword', 'name': 'regressors'}, {'default': 'True', 'kind': 'keyword-only', 'name': 'intercept'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'domain'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'default': '0.0', 'kind': 'keyword-only', 'name': 'null'}, {'default': '100', 'kind': 'keyword-only', 'name': 'max_iter'}, {'default': '1e-09', 'kind': 'keyword-only', 'name': 'tolerance'}], [{'kind': 'keyword-only', 'name': 'data'}, {'kind': 'keyword-only', 'name': 'items'}, {'default': '41', 'kind': 'keyword-only', 'name': 'points'}, {'default': '200', 'kind': 'keyword-only', 'name': 'max_iter'}, {'default': '400', 'kind': 'keyword-only', 'name': 'max_eval'}, {'default': '1e-06', 'kind': 'keyword-only', 'name': 'tolerance'}, {'default': '0.95', 'kind': 'keyword-only', 'name': 'level'}], [{'kind': 'positional-or-keyword', 'name': 'data'}, {'kind': 'positional-or-keyword', 'name': 'design'}, {'kind': 'positional-or-keyword', 'name': 'outcomes'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'domain'}, {'choices': ["'drop'", "'raise'"], 'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'default': '0.0', 'kind': 'keyword-only', 'name': 'null'}], [{'kind': 'keyword-only', 'name': 'data'}, {'kind': 'keyword-only', 'name': 'y'}, {'kind': 'keyword-only', 'name': 'x'}, {'kind': 'keyword-only', 'name': 'panel'}, {'kind': 'keyword-only', 'name': 'time'}, {'default': '1', 'kind': 'keyword-only', 'name': 'p'}, {'default': '1', 'kind': 'keyword-only', 'name': 'q'}, {'default': "'c'", 'kind': 'keyword-only', 'name': 'trend'}, {'default': '500', 'kind': 'keyword-only', 'name': 'max_iterations'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}], [{'kind': 'positional-or-keyword', 'name': 'lower'}, {'kind': 'positional-or-keyword', 'name': 'upper'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'times'}, {'default': '0.95', 'kind': 'keyword-only', 'name': 'level'}, {'default': "'cpu'", 'kind': 'keyword-only', 'name': 'device'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'weights'}, {'default': '500', 'kind': 'keyword-only', 'name': 'maxiter'}], [{'kind': 'positional-or-keyword', 'name': 'result'}, {'kind': 'positional-or-keyword', 'name': 'data'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': 'DEFAULT_WORK', 'kind': 'keyword-only', 'name': 'max_work'}, {'default': 'DEFAULT_BYTES', 'kind': 'keyword-only', 'name': 'max_bytes'}, {'default': "'cpu'", 'kind': 'keyword-only', 'name': 'device'}], [{'kind': 'keyword-only', 'name': 'data'}, {'kind': 'keyword-only', 'name': 'y'}, {'kind': 'keyword-only', 'name': 'x'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'kind': 'var-keyword', 'name': 'options'}], [{'kind': 'positional-or-keyword', 'name': 'result'}, {'kind': 'keyword-only', 'name': 'data'}, {'kind': 'keyword-only', 'name': 'variable'}, {'default': '1', 'kind': 'keyword-only', 'name': 'order'}, {'default': 'False', 'kind': 'keyword-only', 'name': 'interval'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '2000000000', 'kind': 'keyword-only', 'name': 'max_work'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weight_type', 'kind': 'keyword-only', 'default': "'aweight'"}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'l1_ratio', 'default': '0.5', 'kind': 'keyword-only'}, {'name': 'selection', 'kind': 'keyword-only', 'default': "'cv'"}, {'name': 'penalty', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'lambda_path', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'n_lambdas', 'default': '20', 'kind': 'keyword-only'}, {'name': 'lambda_ratio', 'default': '0.001', 'kind': 'keyword-only'}, {'name': 'folds', 'default': '5', 'kind': 'keyword-only'}, {'name': 'seed', 'default': '1729', 'kind': 'keyword-only'}, {'name': 'standardize', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'penalty_factors', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'forced_controls', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'max_iterations', 'default': '200', 'kind': 'keyword-only'}, {'name': 'tolerance', 'default': '1e-08', 'kind': 'keyword-only'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '2000000000'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'panel', 'kind': 'keyword-only'}, {'name': 'time', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'model', 'kind': 'keyword-only', 'default': "'re'"}, {'name': 'covariance', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intpoints', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intmethod', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'corr', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'corr_order', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'force', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'offset', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'low', 'kind': 'positional-or-keyword'}, {'name': 'indicator', 'kind': 'positional-or-keyword'}, {'name': 'rho', 'kind': 'keyword-only'}, {'name': 'low_periods', 'kind': 'keyword-only'}, {'name': 'high_periods', 'kind': 'keyword-only'}, {'name': 'low_frequency', 'kind': 'keyword-only', 'default': "'Y'"}, {'name': 'high_frequency', 'kind': 'keyword-only', 'default': "'Q'"}, {'name': 'aggregation', 'kind': 'keyword-only', 'default': "'sum'"}, {'name': 'as_of', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'low_releases', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'high_releases', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'design', 'kind': 'positional-or-keyword'}, {'name': 'outcome', 'kind': 'positional-or-keyword'}, {'name': 'categories', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'domain', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'missing', 'choices': ["'drop'", "'raise'"], 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'null', 'default': '0.0', 'kind': 'keyword-only'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'design', 'kind': 'positional-or-keyword'}, {'name': 'numerators', 'kind': 'positional-or-keyword'}, {'name': 'denominators', 'kind': 'positional-or-keyword'}, {'name': 'domain', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'missing', 'choices': ["'drop'", "'raise'"], 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'null', 'default': '0.0', 'kind': 'keyword-only'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'inflate', 'kind': 'keyword-only'}, {'name': 'inflate_link', 'kind': 'keyword-only', 'default': "'logit'"}, {'name': 'covariance', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weight_type', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'offset', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'exposure', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'dimensions', 'kind': 'positional-or-keyword'}, {'name': 'count', 'kind': 'positional-or-keyword'}, {'name': 'levels', 'kind': 'keyword-only'}, {'name': 'design', 'kind': 'keyword-only'}, {'name': 'terms', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'structural', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'offset', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'max_iter', 'default': '200', 'kind': 'keyword-only'}, {'name': 'tol', 'default': '1e-09', 'kind': 'keyword-only'}, {'name': 'level', 'kind': 'keyword-only', 'default': '0.95'}, {'name': 'max_work', 'default': '300000000', 'kind': 'keyword-only'}, {'name': 'max_bytes', 'default': '128 * 1024 ** 2', 'kind': 'keyword-only'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'group', 'kind': 'keyword-only'}, {'name': 'random', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intpoints', 'default': '7', 'kind': 'keyword-only'}, {'name': 'intmethod', 'kind': 'keyword-only', 'default': "'mvaghermite'"}, {'name': 'covariance', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'll', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'ul', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'covariance', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weight_type', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'offset', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'title', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'unit', 'kind': 'keyword-only', 'default': "''"}, {'name': 'palette', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'aggregate', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'raters', 'kind': 'positional-or-keyword'}, {'name': 'categories', 'kind': 'keyword-only'}, {'name': 'agreement_weights', 'kind': 'keyword-only', 'default': "'unweighted'"}, {'name': 'inference', 'kind': 'keyword-only', 'default': "'none'"}, {'name': 'level', 'kind': 'keyword-only', 'default': '0.95'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'drop'"}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'max_fits', 'default': '1024', 'kind': 'keyword-only'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '100000000'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'data', 'kind': 'keyword-only'}, {'name': 'variable', 'kind': 'keyword-only'}, {'name': 'order', 'default': '1', 'kind': 'keyword-only'}, {'name': 'interval', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '2000000000'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'time', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'trend', 'kind': 'keyword-only', 'default': "'c'"}, {'name': 'kernel', 'kind': 'keyword-only', 'default': "'bartlett'"}, {'name': 'bandwidth', 'default': '4', 'kind': 'keyword-only'}, {'name': 'df_adjust', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'covariance', 'choices': ["'cluster'", "'nonrobust'"], 'kind': 'keyword-only', 'default': 'None'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'missing', 'choices': ["'raise'", "'drop'"], 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'covariance', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weight_type', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'offset', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'kind', 'choices': ["'linear'", "'response'"], 'kind': 'keyword-only', 'default': "'response'"}, {'name': 'missing', 'choices': ["'drop'", "'raise'"], 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'objective', 'kind': 'keyword-only', 'default': "'weight'"}, {'name': 'max_component_nodes', 'default': '24', 'kind': 'keyword-only'}, {'name': 'max_states', 'kind': 'keyword-only', 'default': '1000000'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '50000000'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'treatment', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'data', 'kind': 'var-keyword'}], [{'name': 'partition', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'objective', 'kind': 'keyword-only', 'default': "'weight'"}, {'name': 'max_matrix_entries', 'kind': 'keyword-only', 'default': '1000000'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '50000000'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'design', 'kind': 'positional-or-keyword'}, {'name': 'outcomes', 'kind': 'positional-or-keyword'}, {'name': 'domain', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'null', 'default': '0.0', 'kind': 'keyword-only'}, {'name': 'deff', 'kind': 'keyword-only', 'default': 'False'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'outcome', 'kind': 'positional-or-keyword'}, {'name': 'predictors', 'kind': 'positional-or-keyword'}, {'name': 'key', 'kind': 'keyword-only'}, {'name': 'spatial_weights', 'kind': 'keyword-only'}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'path', 'kind': 'positional-or-keyword'}, {'name': 'format', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'max_memory_mb', 'default': '256', 'kind': 'keyword-only'}, {'name': 'batch_rows', 'default': '65536', 'kind': 'keyword-only'}, {'name': 'max_file_mb', 'default': '512', 'kind': 'keyword-only'}], [{'description': 'A DataFrame, column dictionary, row records or batch Dataset source.', 'name': 'data', 'kind': 'keyword-only'}, {'description': 'Outcome column name; cannot be combined with formula.', 'name': 'y', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Predictor column names; cannot be combined with formula.', 'name': 'x', 'kind': 'keyword-only', 'default': 'None'}, {'description': "Example: 'wage ~ education + experience + C(region)'.", 'name': 'formula', 'kind': 'keyword-only', 'default': 'None'}, {'description': "Standard error method, such as 'nonrobust', 'HC3', 'cluster' or 'hac'.", 'name': 'covariance', 'choices': ["'nonrobust'", "'HC0'", "'HC1'", "'HC2'", "'HC3'", "'cluster'", "'cluster_hc2'", "'cluster_hc3'", "'hac'", "'bootstrap'", "'jackknife'"], 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Predictor names to encode as categorical variables.', 'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Include an intercept.', 'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'description': 'Column name or names for clustered standard errors.', 'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Column name containing the weights.', 'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'description': "'aweight', 'fweight', 'pweight' or 'iweight'.", 'name': 'weight_type', 'choices': ["'aweight'", "'fweight'", "'pweight'", "'iweight'"], 'kind': 'keyword-only', 'default': 'None'}, {'description': "'drop' removes observations with missing values; 'raise' reports an error.", 'name': 'missing', 'choices': ["'raise'", "'drop'"], 'kind': 'keyword-only', 'default': "'drop'"}, {'description': 'Inference significance level; 0.05 gives a 95% confidence interval.', 'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'description': 'Time column used for HAC or lagged formulas.', 'name': 'time', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'HAC lag count or a supported automatic selection name.', 'name': 'lags', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'HAC weighting kernel.', 'name': 'kernel', 'choices': ["'bartlett'", "'parzen'", "'quadratic_spectral'", "'truncated'"], 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Replication option for bootstrap or jackknife.', 'name': 'reps', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Random seed for resampling.', 'name': 'seed', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'dfadjust', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'hansen', 'kind': 'keyword-only', 'default': 'False'}, {'description': "'auto', 'cpu', 'cuda', 'cuda:<index>' or 'mps'. CUDA uses float64; Metal preconditions bounded factors with checked CPU float64 refinement.", 'name': 'device', 'kind': 'keyword-only', 'default': "'auto'"}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'endogenous', 'kind': 'keyword-only'}, {'name': 'instruments', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'covariance', 'kind': 'keyword-only', 'default': "'robust'"}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'max_iterations', 'default': '100', 'kind': 'keyword-only'}, {'name': 'tolerance', 'default': '1e-09', 'kind': 'keyword-only'}, {'name': 'max_work', 'default': '10000000000', 'kind': 'keyword-only'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}, {'name': 'batch_rows', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'coefficients', 'kind': 'positional-or-keyword'}, {'name': 'null', 'default': '0.0', 'kind': 'keyword-only'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'time', 'kind': 'positional-or-keyword'}, {'name': 'event', 'kind': 'positional-or-keyword'}, {'name': 'causes', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'times', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'level', 'kind': 'keyword-only', 'default': '0.95'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'nodes', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'edges', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'graph_attributes', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'coefficients', 'kind': 'positional-or-keyword'}, {'name': 'null', 'default': '0.0', 'kind': 'keyword-only'}], [{'name': 'add', 'default': '()', 'kind': 'keyword-only'}, {'name': 'remove', 'default': '()', 'kind': 'keyword-only'}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'attributes', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'max_work', 'kind': 'keyword-only', 'default': '50000000'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'columns', 'kind': 'positional-or-keyword'}, {'name': 'bounds', 'kind': 'keyword-only'}, {'name': 'sampling_model', 'kind': 'keyword-only'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'y', 'kind': 'positional-or-keyword'}, {'description': '``n_dropped``) or "raise".', 'name': 'missing', 'kind': 'keyword-only', 'default': "'drop'"}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'data', 'kind': 'keyword-only'}, {'name': 'alignment', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'kind', 'kind': 'keyword-only', 'default': "'mean'"}], [{'name': 'path', 'kind': 'positional-or-keyword'}, {'name': 'overwrite', 'kind': 'keyword-only', 'default': 'False'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'args', 'kind': 'var-positional'}, {'name': 'kwargs', 'kind': 'var-keyword'}], [{'name': 'buf', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'kwargs', 'kind': 'var-keyword'}], [{'name': 'direction', 'kind': 'keyword-only', 'default': "'out'"}, {'name': 'disconnected', 'kind': 'keyword-only', 'default': "'infinite'"}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '50000000'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': 'DEFAULT_WORK'}, {'name': 'max_bytes', 'kind': 'keyword-only', 'default': 'DEFAULT_BYTES'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'title', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'attributes', 'kind': 'keyword-only', 'default': 'True'}], [{'name': 'restrictions', 'kind': 'positional-or-keyword'}, {'name': 'null', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'columns', 'kind': 'positional-or-keyword'}, {'name': 'sampling_model', 'kind': 'keyword-only'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}], [{'name': 'add', 'default': '()', 'kind': 'keyword-only'}, {'name': 'remove', 'default': '()', 'kind': 'keyword-only'}, {'name': 'rename', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'max_nodes', 'default': '1000', 'kind': 'positional-or-keyword'}, {'name': 'max_edges', 'default': '5000', 'kind': 'positional-or-keyword'}, {'name': 'seed', 'default': '0', 'kind': 'positional-or-keyword'}, {'name': 'groups', 'default': 'None', 'kind': 'positional-or-keyword'}], [{'name': 'options', 'kind': 'var-keyword'}], [{'name': 'source', 'kind': 'positional-or-keyword'}, {'name': 'target', 'kind': 'positional-or-keyword'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '50000000'}], [{'name': 'path', 'kind': 'positional-or-keyword'}, {'name': 'format', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'overwrite', 'kind': 'keyword-only', 'default': 'False'}], [{'name': 'restricted', 'kind': 'positional-or-keyword'}, {'name': 'full', 'kind': 'positional-or-keyword'}, {'name': 'max_bytes', 'default': '128 * 1024 ** 2', 'kind': 'keyword-only'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'columns', 'kind': 'positional-or-keyword'}, {'name': 'max_iterations', 'default': '500', 'kind': 'keyword-only'}, {'name': 'tolerance', 'default': '1e-08', 'kind': 'keyword-only'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'path', 'default': 'None', 'kind': 'positional-or-keyword'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'coefficient', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '100000000'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'level', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'null', 'default': '0.0', 'kind': 'keyword-only'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '100000000'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'series', 'kind': 'positional-or-keyword'}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'variables', 'kind': 'positional-or-keyword'}, {'name': 'knots', 'kind': 'keyword-only'}, {'name': 'components', 'default': '2', 'kind': 'keyword-only'}, {'name': 'n_starts', 'default': '3', 'kind': 'keyword-only'}, {'name': 'seed', 'kind': 'keyword-only', 'default': '0'}, {'name': 'maxiter', 'default': '500', 'kind': 'keyword-only'}, {'name': 'tol', 'default': '1e-08', 'kind': 'keyword-only'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'drop'"}, {'name': 'max_work', 'default': 'WORK', 'kind': 'keyword-only'}, {'name': 'max_bytes', 'default': 'BYTES', 'kind': 'keyword-only'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'outcome', 'kind': 'positional-or-keyword'}, {'name': 'predictors', 'kind': 'positional-or-keyword'}, {'name': 'knots', 'kind': 'keyword-only'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'drop'"}, {'name': 'max_bytes', 'default': 'c.LIMIT_BYTES', 'kind': 'keyword-only'}, {'name': 'max_work', 'default': 'c.WORK', 'kind': 'keyword-only'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'max_work', 'default': 'WORK', 'kind': 'keyword-only'}, {'name': 'max_bytes', 'default': 'BYTES', 'kind': 'keyword-only'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}], [{'description': 'Any registry fit except time-series models and fits that already use a resampling covariance.', 'name': 'result', 'kind': 'positional-or-keyword'}, {'description': 'Exactly the dataset ``result`` was fitted on (verified by hash).', 'name': 'data', 'kind': 'positional-or-keyword'}, {'description': 'Number of bootstrap replications R (at least 2; 200 by default, use 1000 or more for percentile/bc intervals).', 'name': 'reps', 'default': '200', 'kind': 'keyword-only'}, {'description': 'Seed (0 to 2**64 - 1) of the ``torch.Generator`` that draws every replicate; the same seed reproduces the result exactly. A random seed is drawn and recorded when omitted.', 'name': 'seed', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Columns defining resampling clusters and strata.', 'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Columns defining resampling clusters and strata.', 'name': 'strata', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Units drawn per stratum.', 'name': 'size', 'kind': 'keyword-only', 'default': 'None'}, {'description': "Level of the intervals (default: the fit's alpha).", 'name': 'alpha', 'kind': 'keyword-only', 'default': 'None'}, {'description': "Interval type stored in ``extra['bootstrap_ci']``: normal based, percentile (Stata's ``_pctile`` definition) or bias-corrected percentile (z0 = Phi^-1(#{b_r <= b}/R)), BCa delete-unit acceleration or bootstrap-t using each refit's standard error. Advanced intervals currently require unweighted, unst", 'name': 'ci', 'kind': 'keyword-only', 'default': "'normal'"}, {'description': 'Dependent schemes validate explicit-column OLS. Blocks require a regular time column and block_length; wild_cluster requires cluster=.', 'name': 'scheme', 'kind': 'keyword-only', 'default': "'iid'"}, {'name': 'block_length', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'time', 'kind': 'keyword-only', 'default': 'None'}, {'description': "Coefficient restrictions for restricted wild cluster-t tests; requires the original covariance clustered on the same column and ci='normal'. Confidence-set inversion is not supplied.", 'name': 'null', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Wild cluster multiplier distribution.', 'name': 'wild', 'kind': 'keyword-only', 'default': "'rademacher'"}, {'description': 'Advanced draw plus BCa acceleration budget (default 10,000).', 'name': 'max_refits', 'default': '10000', 'kind': 'keyword-only'}], [{'description': "String, path object (implementing os.PathLike[str]), or file-like object implementing a write() function. If None, the result is returned as a string. If a non-binary file object is passed, it should be opened with `newline=''`, disabling universal newlines. If a binary file object is passed, `mode`", 'name': 'path_or_buf', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'String of length 1. Field delimiter for the output file.', 'name': 'sep', 'default': "','", 'kind': 'positional-or-keyword'}, {'description': 'Missing data representation.', 'name': 'na_rep', 'default': "''", 'kind': 'positional-or-keyword'}, {'description': 'Format string for floating point numbers. If a Callable is given, it takes precedence over other numeric formatting parameters, like decimal.', 'name': 'float_format', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Columns to write.', 'name': 'columns', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Write out the column names. If a list of strings is given it is assumed to be aliases for the column names.', 'name': 'header', 'default': 'True', 'kind': 'positional-or-keyword'}, {'description': 'Write row names (index).', 'name': 'index', 'default': 'True', 'kind': 'positional-or-keyword'}, {'description': 'Column label for index column(s) if desired. If None is given, and `header` and `index` are True, then the index names are used. A sequence should be given if the object uses MultiIndex. If False do not print fields for index names. Use index_label=False for easier importing in R.', 'name': 'index_label', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Forwarded to either `open(mode=)` or `fsspec.open(mode=)` to control the file opening. Typical values include:', 'name': 'mode', 'default': "'w'", 'kind': 'positional-or-keyword'}, {'description': "A string representing the encoding to use in the output file, defaults to 'utf-8'. `encoding` is not supported if `path_or_buf` is a non-binary file object.", 'name': 'encoding', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'compression', 'default': "'infer'", 'kind': 'positional-or-keyword'}, {'name': 'quoting', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'quotechar', 'default': '\'"\'', 'kind': 'positional-or-keyword'}, {'name': 'lineterminator', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'chunksize', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'date_format', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'doublequote', 'default': 'True', 'kind': 'positional-or-keyword'}, {'name': 'escapechar', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'decimal', 'default': "'.'", 'kind': 'positional-or-keyword'}, {'name': 'errors', 'default': "'strict'", 'kind': 'positional-or-keyword'}, {'name': 'storage_options', 'default': 'None', 'kind': 'positional-or-keyword'}], [{'description': 'File path or existing ExcelWriter.', 'name': 'excel_writer', 'kind': 'positional-or-keyword'}, {'description': 'Name of sheet which will contain DataFrame.', 'name': 'sheet_name', 'default': "'Sheet1'", 'kind': 'positional-or-keyword'}, {'description': 'Missing data representation.', 'name': 'na_rep', 'default': "''", 'kind': 'positional-or-keyword'}, {'description': 'Format string for floating point numbers. For example ``float_format="%.2f"`` will format 0.1234 to 0.12.', 'name': 'float_format', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Columns to write.', 'name': 'columns', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Write out the column names. If a list of string is given it is assumed to be aliases for the column names.', 'name': 'header', 'default': 'True', 'kind': 'positional-or-keyword'}, {'description': 'Write row names (index).', 'name': 'index', 'default': 'True', 'kind': 'positional-or-keyword'}, {'description': 'Column label for index column(s) if desired. If not specified, and `header` and `index` are True, then the index names are used. A sequence should be given if the DataFrame uses MultiIndex.', 'name': 'index_label', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Upper left cell row to dump data frame.', 'name': 'startrow', 'default': '0', 'kind': 'positional-or-keyword'}, {'description': 'Upper left cell column to dump data frame.', 'name': 'startcol', 'default': '0', 'kind': 'positional-or-keyword'}, {'description': "Write engine to use, 'openpyxl' or 'xlsxwriter'. You can also set this via the options ``io.excel.xlsx.writer`` or ``io.excel.xlsm.writer``.", 'name': 'engine', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Write MultiIndex and Hierarchical Rows as merged cells.', 'name': 'merge_cells', 'default': 'True', 'kind': 'positional-or-keyword'}, {'description': 'Representation for infinity (there is no native representation for infinity in Excel).', 'name': 'inf_rep', 'default': "'inf'", 'kind': 'positional-or-keyword'}, {'description': 'Specifies the one-based bottommost row and rightmost column that is to be frozen.', 'name': 'freeze_panes', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'storage_options', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'engine_kwargs', 'default': 'None', 'kind': 'positional-or-keyword'}], [{'description': 'Mapping from unique predictor names to aligned Network snapshots; all dyads include absent edges as zero.', 'name': 'predictors', 'kind': 'positional-or-keyword'}, {'description': "'weight' uses aggregate strengths; 'binary' uses edge presence.", 'name': 'values', 'kind': 'keyword-only', 'default': "'weight'"}, {'description': 'Include diagonal dyads; False excludes them.', 'name': 'include_loops', 'kind': 'keyword-only', 'default': 'False'}, {'description': 'Uniform node permutations per coefficient-specific reduced-model residual test; plus-one p-values.', 'name': 'permutations', 'default': '999', 'kind': 'keyword-only'}, {'description': 'Private CPU Torch random seed; predictor names and node labels use canonical order.', 'name': 'seed', 'kind': 'keyword-only', 'default': '0'}, {'description': "'two-sided' compares absolute partial correlations; 'greater' and 'less' use the signed statistic.", 'name': 'alternative', 'kind': 'keyword-only', 'default': "'two-sided'"}, {'description': 'Fit a constant; the constant receives no fabricated permutation test.', 'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'description': "'freedman_lane' permutes reduced response residuals; 'dsp' permutes focal predictor residuals and projects nuisance effects again.", 'name': 'method', 'kind': 'keyword-only', 'default': "'freedman_lane'"}, {'description': 'Named coefficient subsets tested together, conditional on remaining predictors; upper-tail partial R-squared.', 'name': 'joint', 'kind': 'keyword-only', 'default': 'None'}, {'description': "'none', 'bonferroni', 'holm' or 'bh' over all reported slopes and joint hypotheses; the intercept is excluded.", 'name': 'adjustment', 'kind': 'keyword-only', 'default': "'none'"}, {'description': 'Explicit complete-test structural work budget; no partial permutation p-values.', 'name': 'max_work', 'kind': 'keyword-only', 'default': '50000000'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'y', 'kind': 'positional-or-keyword'}, {'name': 'factors', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'covariates', 'kind': 'keyword-only', 'default': 'None'}, {'description': '``"none"`` (main effects only) or a list such as ``[["a", "b"]]`` or ``["a#b"]``; a covariate may appear in an interaction. An interaction needs the effects it contains in the model (``a#b#c`` needs ``a#b``, ``a#c`` and ``b#c``; ``a#b#x`` needs ``a#x`` and ``b#x``), otherwise ``invalid_spec``.', 'name': 'interactions', 'kind': 'keyword-only', 'default': "'full'"}, {'name': 'ss_type', 'default': '3', 'kind': 'keyword-only'}, {'description': '``oe.rm_anova`` with ``factors`` as between-subject factors (Type III, all between-subject interactions; covariates, ``emmeans`` and other values of ``ss_type`` or ``interactions`` are rejected).', 'name': 'within', 'kind': 'keyword-only', 'default': 'None'}, {'description': '``oe.rm_anova`` with ``factors`` as between-subject factors (Type III, all between-subject interactions; covariates, ``emmeans`` and other values of ``ss_type`` or ``interactions`` are rejected).', 'name': 'subject', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'reported: predictions averaged with equal weights over the levels of the other factors, with covariates at their sample means. Single factors also get pairwise comparisons.', 'name': 'emmeans', 'kind': 'keyword-only', 'default': 'None'}, {'description': "SPSS's default) for the pairwise comparisons and their intervals.", 'name': 'adjust', 'kind': 'keyword-only', 'default': "'bonferroni'"}, {'description': 'tests the ANCOVA assumption of equal slopes.', 'name': 'homogeneity_of_slopes', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'description': '``"raise"``.', 'name': 'missing', 'kind': 'keyword-only', 'default': "'drop'"}])

STORED_PARAMETER_KEYS = {"description": "d", "name": "n", "annotation": "a",
                         "default": "v", "choices": "c", "kind": "k"}
STORED_ENTRY_KEYS = {"signature": "s", "description": "d",
                     "parameters": "p", "returns": "r", "name": "n", "kind": "k"}
MAX_SAFE_INTEGER = 2**53 - 1
# Protocol order is fixed; changing it would reinterpret saved catalogs.
STORED_PARAMETER_NAMES = (
    "max_work", "max_iterations", "tolerance", "covariance", "max_bytes",
    "intercept", "missing", "instruments", "endogenous", "data",
    "alpha", "weights", "result", "device", "time", "cluster", "categorical",
    "seed", "weight_type", "confidence", "components", "replications", "max_iter", "options", "columns", "level", "method", "batch_rows", "treatment", "alternative", "max_memory_mb",
)
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
    "openecon.repeated_gls": "Gaussian ML/REML with CS, signed integer-gap AR1, occasion diagonal or full residual covariance. Complete CPU float64; asymptotic uncertainty.",
    "openecon.restore_repeated_gls": "Validate complete repeated GLS sample, likelihood and covariance; replay without optimization.",
    "openecon.repeated_gls_predict": "Saved joint population means, full covariance and asymptotic marginal CI. No future residual variance.",
    "openecon.repeated_gls_contrast": "Saved fixed-effect scalar/joint contrasts with full covariance, nonzero null and normal/chi-square inference.",
    'openecon.survey_four_stage_design': 'Declare exactly four sequential SRSWOR selections with nested PSU/SSU/TSU/FSU IDs. Optional first-stage strata, unstratified lower stages and four exact parent population counts derive N/n times M/m times L/l times K/k weights. Each physical row is one unique FSU.',
    'openecon.survey_four_stage_mean': 'Joint weighted means with four recursive SRSWOR covariance components and complete domain/missing geometry.',
    'openecon.survey_four_stage_total': 'Joint Horvitz-Thompson totals with four parent-specific FPC contributions and primitive saved replay.',
    'openecon.survey_four_stage_ratio': 'Paired total ratios with full four-stage Taylor covariance. Zero denominators fail; full physical geometry persists.',
    'openecon.survey_four_stage_proportion': 'Ordered typed category shares, absent levels and full four-stage SRSWOR covariance.',
    'openecon.survey_four_stage_regress': 'Weighted linear regression with four-stage score covariance, full bread and primitive saved replay. CPU float64; rank loss fails.',
    'openecon.survey_four_stage_logit': 'Binary logit with observed bread and four-stage score covariance. Finite identified fits only; all four FPC contributions persist.',
    'openecon.survey_four_stage_probit': 'Binary probit with observed Hessian bread and full four-stage score covariance. Complete original geometry persists.',
    'openecon.survey_four_stage_poisson': 'Poisson counts with observed bread and four-stage score covariance. Exact nonnegative counts and finite identified fits only.',
    'openecon.survey_fully_stratified_three_stage_design': 'Declare three SRSWOR selections with independently stratified SSUs within each PSU and TSUs within each SSU. Two explicit positive-cell frames and exact populations derive N/n times M_cell/m_cell times L_cell/l_cell weights. One distinct terminal TSU per physical row; external frame completeness cannot be authenticated.',
    'openecon.survey_fully_stratified_three_stage_mean': 'Joint weighted means with all three SRSWOR covariance contributions, centering SSUs and TSUs within their own declared sampling cells. Complete domain/missing geometry and saved primitive replay.',
    'openecon.survey_fully_stratified_three_stage_total': 'Joint Horvitz-Thompson totals with cell-specific weights and all three SRSWOR variance terms. Full positive-cell frame, sampling geometry and ordered primitive targets persist.',
    'openecon.survey_fully_stratified_three_stage_ratio': 'Joint weighted ratios with full three-stage Taylor covariance and independently stratified second- and third-stage units. Zero denominators fail; all three FPC terms and excluded-row geometry persist.',
    'openecon.survey_fully_stratified_three_stage_proportion': 'Ordered typed category shares with full three-stage SRSWOR covariance and independently stratified second- and third-stage units. Complete domain/missing geometry persists.',
    'openecon.survey_fully_stratified_three_stage_regress': 'Weighted linear regression with three-stage score covariance, centering SSUs and TSUs within their declared cells. CPU float64; rank loss fails. Saved scores replay cell weights and every FPC term.',
    'openecon.survey_fully_stratified_three_stage_logit': 'Binary logit with observed bread and full three-stage score covariance for stratified SSUs within each PSU and TSUs within each SSU. Finite identified fits only; cell-specific weights and all FPC terms persist.',
    'openecon.survey_fully_stratified_three_stage_probit': 'Binary probit with observed bread and three-stage score covariance for stratified SSUs within each PSU and TSUs within each SSU. Finite identified fits only; complete cells and sampling geometry persist.',
    'openecon.survey_fully_stratified_three_stage_poisson': 'Poisson counts with observed bread and three-stage score covariance for stratified SSUs within each PSU and TSUs within each SSU. Exact nonnegative counts, finite identified fits and complete FPC replay.',
    "openecon.meta_dependent": "Known study-block covariance GLS or one effect/study variance ML/REML. CPU float64; complete labeled effects; plug-in z uncertainty.",
    "openecon.meta_dependent_robust": "Saved study-score CR0/CR1 G/(G-p) sandwich with approximate t/F(G-p). Full covariance; no refit or CR2.",
    "openecon.meta_dependent_contrast": "Saved named scalar/joint linear contrasts, full covariance and z/chi-square or study t/F inference. No refit.",
    "openecon.meta_dependent_predict": "Saved joint new-profile population means and full covariance; normal or study t marginal CI. No refit.",
    "openecon.meta_dependent_predict_effect": "Joint future latent-effect covariance and plug-in PI; effect or shared-study heterogeneity. No sampling error or tau uncertainty.",
    "openecon.meta_dependent_diagnostics": "Whole-study covariance-block deletion refits; complete influence/state and explicit failed fits. Bounded cumulative work.",
    "openecon.nlsur": "Shared-parameter nonlinear Gaussian SUR ML; full physical mean/Sigma OIM, whole-row HC0 or cluster CR0. CPU float64.",
    "openecon.nlsur_restore": "Reparse safe formulas and replay complete joint nonlinear SUR state without fitting.",
    "openecon.nlsur_predict": "Saved/new Gaussian means and conditional means/Schur residual covariance; full joint delta. No refit.",
    "openecon.nlsur_margins": "Mixed-derivative effects, positive-domain elasticities and fixed weighted averages; full joint covariance.",
    "openecon.nlsur_contrast": "Safe physical-parameter nonlinear contrasts and regular full-rank joint Wald inference.",
    "openecon.nlogit": "Two-level disjoint-nest joint ML; interior dissimilarities, full OIM/case HC0/respondent CR0. CPU float64.",
    "openecon.mprobit": "Normalized two/three-alternative Gaussian choice ML; full physical coefficient/covariance OIM, case HC0 or respondent CR0. CPU float64.",
    "openecon.mprobit_restore": "Replay complete normalized Gaussian choice state and joint physical information without fitting.",
    "openecon.mprobit_predict": "Saved/new availability-aware Gaussian choice probabilities/logs with full joint delta. No refit.",
    "openecon.mprobit_margins": "Own/cross raw-attribute effects, positive elasticities and fixed weighted averages with full joint delta.",
    "openecon.nlogit_restore": "Replay complete nested-choice state and physical stationarity without refitting.",
    "openecon.nlogit_predict": "Saved/new alternative, within-nest and nest probabilities/logs; full joint delta. No refit.",
    "openecon.nlogit_margins": "Analytic own/cross effects and positive-attribute elasticities; fixed weighted means with pair support and joint delta.",
    'openecon.survey_stratified_three_stage_design': 'Declare three SRSWOR selections with independently stratified SSUs within each PSU. Explicit positive-cell frame and exact populations derive N/n times M_cell/m_cell times L/l weights. One distinct terminal TSU per physical row; external frame completeness cannot be authenticated.',
    'openecon.survey_stratified_three_stage_mean': 'Joint weighted means with all three SRSWOR covariance contributions, centering second-stage SSUs within each declared cell. Complete domain/missing geometry and saved primitive replay.',
    'openecon.survey_stratified_three_stage_total': 'Joint Horvitz-Thompson totals with cell-specific weights and all three SRSWOR variance terms. Full positive-cell frame, sampling geometry and ordered primitive targets persist.',
    'openecon.survey_stratified_three_stage_ratio': 'Joint weighted ratios with full three-stage Taylor covariance and independently stratified second-stage SSUs. Zero denominators fail; all three FPC terms and excluded-row geometry persist.',
    'openecon.survey_stratified_three_stage_proportion': 'Ordered typed category shares with full three-stage SRSWOR covariance and independently stratified second-stage SSUs. Complete domain/missing geometry persists.',
    'openecon.survey_stratified_three_stage_regress': 'Weighted linear regression with three-stage score covariance, centering SSUs within declared cells. CPU float64; rank loss fails. Saved scores replay cell weights and every FPC term.',
    'openecon.survey_stratified_three_stage_logit': 'Binary logit with observed bread and full three-stage score covariance for stratified SSUs within each PSU. Finite identified fits only; cell-specific weights and all FPC terms persist.',
    'openecon.survey_stratified_three_stage_probit': 'Binary probit with observed bread and three-stage score covariance for stratified SSUs within each PSU. Finite identified fits only; complete cells and sampling geometry persist.',
    'openecon.survey_stratified_three_stage_poisson': 'Poisson counts with observed bread and three-stage score covariance for stratified SSUs within each PSU. Exact nonnegative counts, finite identified fits and complete FPC replay.',

    "openecon.rologit": "Strict rank-prefix logit; complete case/cluster covariance. CPU float64 N<=4096; no ties/intercept/weights.",
    "openecon.rologit_restore": "Replay full state and stationarity without refitting.",
    "openecon.rologit_predict": "Saved/new first/stage/prefix probabilities; full risksets and joint delta. No refit.",
    "openecon.rologit_margins": "Own/cross effects or positive-attribute elasticities; fixed weighted means, eligible pair support and joint delta.",
    "openecon.smoothing_margins": "Average response or first partial over explicit fixed covariates; aggregate full basis before covariance, never average row SEs.",
    "openecon.survey_margins_replicate": "Joint partial-profile predictive means or continuous AMEs for logit/probit/Poisson. Every BRR/Fay/PSU-jackknife/supplied-bootstrap replica refits and reweights the empirical distribution; complete covariance and design-t inference. Saved primitives replay targets without refitting; associational, bounded resident CPU only.",


    "openecon.discrim_stepwise": "Forward, backward or removal-first stepwise Wilks LDA screening on one fixed complete candidate sample. Exact integer frequency counts, deterministic ties and protected terms; complete path and saved posterior state. Screening reference p-values are not post-selection inference; no selection-aware LOO claim.",
    "openecon.discrim_stepwise_predict": "Validate saved selected-LDA state and project every complete prediction row, preserving index and missing alignment. An empty selected model predicts its saved priors. No refit or selected-model sampling uncertainty.",
    "openecon.manova_oneway": "One-way MANOVA with nonnegative integer frequency counts and full group/pooled moments, H/E and saved contrast geometry. Counts represent original independent Gaussian observations for inference; no row expansion or survey/multifactor weighted support.",
    "openecon.manova_summary": "One-way MANOVA from declared ordered group means, unbiased sample covariances and exact counts. Full H/E and saved coefficient/error geometry; PSD/count-rank admission. No fabricated observations or raw-sample diagnostics.",
    "openecon.manova_contrast": "Predeclared full-rank L B M = C hypotheses from validated saved MANOVA geometry. Full target covariance, marginal Student t intervals and four multivariate tests retain exact, approximate and Roy upper-bound conventions. Fixed Gaussian common-covariance design; no post-selection or familywise interval claim.",
    "openecon.rm_mtest": "Joint between-design and original-cell L B M = C hypotheses from saved repeated-measures geometry. Full covariance and marginal t intervals; four multivariate tests preserve exact/approximate/upper-bound labels. Independent Gaussian subjects, common unrestricted covariance; sphericity is not required.",
    "openecon.mediation_binary": "Binary logit/probit mediator with Gaussian/logit/probit/Poisson outcome and optional exposure interaction. Exact response-mean contrasts over fixed retained controls with full joint OIM/HC0 delta covariance; complete saved state. Associational by default; causal labeling requires declared assumptions.",
    "openecon.mediation_binary_restore": "Validate and restore all binary-mediation parameters, scores, covariance, counterfactual means and effects from complete saved state without refitting. Retains the fixed empirical control target; no superpopulation, weight or cluster inference.",
    "openecon.mi_discrete": "Proper-prior Poisson, ordered-logit and baseline multinomial FCS on declared numeric codes. Retains finite-MH parameter/predictive state; convergence and conditional compatibility are unassessed. Resident CPU data only.",
    "openecon.mi_delta": "One incomplete target with complete predictors and a caller-declared fixed mean, log-odds or log-mean offset. Saves posterior/predictive state; delta is unidentified from observed data. No general multivariable MNAR claim.",
    "openecon.mi_passive": "Derive declared affine/product/integer-power columns from every saved MI completion. Preserves source type, rows and null geometry; no FCS feedback, new stochastic draws or substantive-model compatibility claim.",
    "openecon.mi_lincom": "Project every imputation's full coefficient/covariance state before pooling linear contrasts. Preserves contrast-specific Rubin/Barnard-Rubin uncertainty and nonzero-null tests; marginal intervals, no joint test or multiplicity adjustment.",


    "openecon.exact_logistic": "Complete fixed-margin common odds over declared independent 2x2 strata. Conditional CMLE retains zero/infinite boundaries; inclusive equal-tail CI and doubled smaller-tail p. No multivariable exact regression or MUE substitution.",
    "openecon.exact_poisson_rate": "Garwood exact central Poisson rate CI and inclusive tail test for known fixed exposure. Integer counts; zero-event bound explicit, variance is descriptive MLE plugin. No covariate Poisson regression or estimated exposure.",

    "openecon.spatial_predict": "Unconditional ML mean on every effective saved SAR/SEM/SAC/SDM graph unit. Full parameter delta covariance and keyed state; resident N<=512/P<=128. No graph replacement, observation interval, IV or streaming support.",
    "openecon.spatial_diagnostics": "Eight saved numeric unweighted OLS spatial diagnostics with exact original-data replay and a declared keyed graph: projected residual Moran moments/Gaussian simulation, lag/error and locally adjusted/joint Rao scores, and fixed-W WX finite Gaussian F. Full state; CPU float64 N<=512/P<=32; explicit rank/resource/refusal gates.",
    "openecon.stinterval_exponential": "Intercept-only exponential survival ML with exact density, left, interval and right censoring. Saves full observed-information covariance and survival curves; iid CPU vectors only, no covariate regression.",
    "openecon.stinterval_weibull": "Intercept-only interval-censored Weibull shape/scale ML with complete observed-information and curve covariance. Finite interior fits only; no covariate regression or weighted inference.",
    "openecon.stinterval_lognormal": "Intercept-only interval-censored lognormal meanlog/sdlog ML with stable tail likelihood and complete covariance. Saves actual-time likelihood and pointwise survival uncertainty.",
    "openecon.stinterval_loglogistic": "Intercept-only interval-censored loglogistic shape/scale ML with full observed-information covariance and pointwise survival uncertainty. Independent noninformative censoring is assumed.",
    "openecon.interval_survival_predict": "Restore validated complete interval-survival ML state and covariance for prespecified survival times. No refit; accepts fitted TableSet or its saved attrs, CPU only.",
    "openecon.turnbull": "Bounded interval-censored NPMLE on maximal-intersection support with likelihood/KKT and unique-mass gates. Reports sharp support-location bounds; no invented event midpoint, cure atom or sampling covariance/CI.",
    "openecon.cumulative_incidence": "Aalen-Johansen cumulative incidence for iid mutually exclusive event causes with full cross-time/cause IJ covariance. Exact tied risk sets; pointwise normal inference, no Fine-Gray model.",
    "openecon.cause_specific_hazard": "Cause-specific cumulative Nelson-Aalen hazard with complete time/cause IJ covariance. Estimates cumulative hazard, not an instantaneous hazard or CIF; iid right-censored single events only.",
    "openecon.cif_compare": "Independent two-group fixed-time CIF contrasts, group_2 minus group_1, with full covariance and normal tests. Joint Wald requires full rank; no Gray global test or Fine-Gray regression.",
    "openecon.pca": "Descriptive correlation/covariance PCA; optional nonnegative integer frequency weights on resident or replayable inputs. Save training moments and eigenvectors for scores. No loading standard errors or sampling-weight support.",
    "openecon.pca_matrix": "PCA of a declared symmetric PSD summary covariance/correlation matrix with explicit n. No PSD repair or invented observations. Supply actual means and scales for centred/standardized scores.",
    "openecon.factor_matrix": "Factor analysis from declared summary covariance/correlation input with explicit n. Reuses extraction/rotation; actual training means and scales are required for saved scores. No fabricated loading uncertainty.",
    "openecon.factor": "EFA with integer frequency weights, PF/IPF/PCF/ML/minres/alpha/SAS image-covariance extraction and persisted scores. CF and partial-target rotations retain transformation and factor covariance; partial targets require an identified binary mask. Fixed counts required for minres/alpha/image. No vendor parity.",
    "openecon.factor_bootstrap": "Marginal IID percentile intervals and full joint covariance for fixed unrotated one-factor principal factoring. Resident rows, fixed sign anchor, regular sampling, all replicate refits required. Saves full sample/vectors; no p/df, rotated or selected-factor inference.",
    "openecon.pca_bootstrap": "Fixed covariance or restandardized correlation PCA IID bootstrap. Saves full eigenpair/loading covariance, marginal percentile intervals and every draw. Simple roots and fixed signs; no selected count, p/df or exact coverage.",
    "openecon.pca_subspace_bootstrap": "IID bootstrap of a fixed leading PCA projector and captured/residual trace. Internal ties allowed; retained boundary must separate. Full joint covariance and all draws; marginal percentiles, no simultaneous region.",
    "openecon.canon_bootstrap": "Fixed positive simple interior CCA roots, or signed coefficients and loadings. Every IID draw refits joint moments; full covariance and marginal percentiles. Fixed anchors/chart; no zero-root test or selected count.",
    "openecon.ca_bootstrap": "Fixed-total or fixed-row multinomial bootstrap of identified correspondence axes. Saves all count tables and joint mass/inertia/coordinate covariance. Fixed signs, marginal percentiles; no independence-null test.",
    "openecon.pca_bootstrap_scores": "Validate and replay complete saved PCA bootstrap for fixed query scores. Each draw recentres/restandardizes queries; full cross-query covariance and missing alignment. Training uncertainty; no future noise interval.",
    "openecon.factor_multifactor_bootstrap": "Fixed 2..4 principal-factor IID bootstrap with signed unrotated axes or one fixed full orthogonal target. Full loading/uniqueness covariance and draws; no selected count, local rotation or ML loading inference.",
    "openecon.pca_fweight_bootstrap": "Fixed PCA eigenpair uncertainty for integer counts of independent units. Multinomial count refits, full joint covariance and every draw; no physical-row bootstrap or data expansion.",
    "openecon.pca_subspace_fweight_bootstrap": "Integer-frequency IID bootstrap of a fixed PCA projector/trace. Internal ties allowed; separated boundary required. Full count state and marginal joint inference.",
    "openecon.canon_fweight_bootstrap": "Integer-frequency IID uncertainty for fixed simple CCA roots or identified coefficients/loadings. Refits each count draw; full joint covariance and state, without expanded measurements.",
    "openecon.factor_multifactor_fweight_bootstrap": "Integer-frequency IID uncertainty for fixed multifactor PF or one fixed full orthogonal target. Full loading/uniqueness/count state; no selected rotation or latent ML inference.",
    "openecon.manova_factorial": "Complete crossed categorical Type III MANOVA with exact integer frequencies and sum coding. Full H/E, covariance and saved contrasts; no row expansion, continuous covariates or survey inference.",
    "openecon.manova_factorial_summary": "Type III factorial MANOVA from declared ordered cell means, unbiased covariances and counts. Complete crossed design and saved full geometry; no synthetic rows or incomplete cells.",
    "openecon.rm_anova_fweight": "Repeated-measures ANOVA from exact whole-subject profile frequencies. Existing correction conventions and saved original-cell hypotheses; no row expansion, incomplete subjects or dependent/survey inference.",
    "openecon.rm_anova_summary": "Repeated-measures tests from declared ordered within-cell group means, unbiased subject covariance and counts. Full saved hypotheses; no invented subject vectors or incomplete/dependent designs.",
    "openecon.ca_project": "Project supplementary row/column profiles onto saved active correspondence axes, matching all opposite-axis categories. No refit, passive contribution or sampling CI; null axes have undefined coordinates.",
    "openecon.alpha": "Descriptive reliability alpha/split/Guttman and full item statistics. Optional integer frequency counts on resident/Dataset moments without row expansion; explicit reversal/standardization and listwise missing policy. No sampling SE/CI or analytic/probability weights.",
    "openecon.factortest": "KMO/item MSA and Bartlett Gaussian correlation test. Optional integer frequency moments on resident/Dataset, n=sum(counts) with separate physical/missing/zero counts. Counts must describe original independent observations for Gaussian inference; no survey weights.",
    "openecon.discrim": "Gaussian LDA/QDA and saved posterior classification. Optional resident integer frequency counts, weighted confusion and one-expanded-copy LOO with full-fit priors. Frequency counts do not repair rank; no Dataset/survey weights or weighted-LOO vendor parity.",
    "openecon.discrim_summary": "LDA/QDA from declared ordered group means, unbiased sample group covariances and integer counts. Retains full saved prediction state; no fabricated training accuracy/LOO or synthetic observations.",
    "openecon.canon": "Canonical correlation with existing raw QR or bounded integer-frequency joint moments. Full coefficients/moments persist for scores; Gaussian correlation tests assume original independent frequencies. Repeated/zero roots retain an explicit arbitrary valid basis, not unique axes.",
    "openecon.canon_matrix": "CCA from a declared joint covariance/correlation and explicit n. Joint PSD with full-rank conditioned within-set blocks; no PSD repair or invented rows. Actual training moments are needed for raw saved scores; repeated roots have nonunique axes.",
    "openecon.canon_scores": "Project complete raw rows through persisted canonical coefficients and actual training moments. Preserve missing rows/index and lazy Dataset integrity checks. Reuse saved arbitrary basis for tied roots; never re-estimate or invent training means/scales.",
    "openecon.rm_anova_wide": "Bounded complete-subject wide adapter to the existing repeated-measures ANOVA kernel. Ordered measurement columns, whole-subject missing policy and full raw sample/moments/order/hash persist. Gaussian independent subjects; no mixed/weighted/dependent design or vendor parity.",
    "openecon.rm_restore": "Validate portable wide RM sample/order/labels/hash and recomputed moments without repair. Retain exact training state for predeclared within-subject contrasts.",
    "openecon.rm_contrasts": "Predeclared zero-sum full-rank contrasts on validated saved RM state. Full mean covariance, marginal t/CI(df n-1), exact Gaussian joint Hotelling F(df q,n-q) and all subject contrast vectors persist. No selected/dependent/weighted contrasts or familywise coverage.",
    "openecon.survey_design": "Declare a bounded resident single-stage sampling design. Validate sampling weights, nested PSU/strata IDs, population-count FPC and singleton policy; persist JSON then revalidate before use. No survey estimates or standard errors are produced.",
    "openecon.survey_three_stage_design": "Declare nested sequential SRSWOR PSU/SSU/TSU sampling. Exact stage population counts derive N/n times M/m times L/l weights; one distinct terminal TSU per physical row. Revalidate saved geometry before use.",
    "openecon.survey_three_stage_mean": "Joint weighted means with complete three-stage SRSWOR covariance and all three stage FPCs. Domain/missing exclusions retain the full design; saved primitives replay inference.",
    "openecon.survey_three_stage_total": "Joint Horvitz-Thompson totals with all three stage SRSWOR covariance terms. Full design, ordered primitive targets and finite population counts persist.",
    "openecon.survey_three_stage_ratio": "Joint weighted ratios with full three-stage Taylor covariance; zero weighted denominators fail. Both stage FPCs and complete excluded-row geometry persist.",
    "openecon.survey_three_stage_proportion": "Ordered category proportions with full three-stage SRSWOR covariance and typed categories. Domain/missing exclusions retain complete design geometry.",
    "openecon.survey_three_stage_regress": "Weighted linear regression with complete nested three-stage score covariance. CPU float64; rank loss fails. Saved primitive scores replay all three stage FPC terms.",
    "openecon.survey_three_stage_logit": "Binary logit with complete three-stage SRSWOR score covariance and observed bread. Finite identified fits only; saved primitives retain all three stage FPC terms.",
    "openecon.survey_three_stage_probit": "Binary probit with observed bread and full three-stage SRSWOR score covariance. Finite identified fits only; complete nested sampling geometry persists.",
    "openecon.survey_three_stage_poisson": "Poisson counts with observed bread and full three-stage SRSWOR score covariance. Exact nonnegative counts and finite identified fits; all three stage FPC terms persist.",
    "openecon.survey_two_stage_design": "Declare nested sequential SRSWOR PSU/SSU sampling. Exact stage population counts derive N/n times M/m weights; one distinct SSU per physical row. Revalidate saved geometry before use.",
    "openecon.survey_two_stage_mean": "Joint weighted means with complete two-stage SRSWOR covariance and both stage FPCs. Domain/missing exclusions retain the full design; saved primitives replay inference.",
    "openecon.survey_two_stage_total": "Joint Horvitz-Thompson totals with both stage SRSWOR covariance terms. Full design, ordered primitive targets and finite population counts persist.",
    "openecon.survey_two_stage_ratio": "Joint weighted ratios with full two-stage Taylor covariance; zero weighted denominators fail. Both stage FPCs and complete excluded-row geometry persist.",
    "openecon.survey_two_stage_proportion": "Ordered category proportions with full two-stage SRSWOR covariance and typed categories. Domain/missing exclusions retain complete design geometry.",
    "openecon.survey_two_stage_regress": "Weighted linear regression with complete nested two-stage score covariance. CPU float64; rank loss fails. Saved primitive scores replay both stage FPC terms.",
    "openecon.survey_two_stage_logit": "Binary logit with complete two-stage SRSWOR score covariance and observed bread. Finite identified fits only; saved primitives retain both stage FPC terms.",
    "openecon.survey_two_stage_probit": "Binary probit with observed bread and full two-stage SRSWOR score covariance. Finite identified fits only; complete nested sampling geometry persists.",
    "openecon.survey_two_stage_poisson": "Poisson counts with observed bread and full two-stage SRSWOR score covariance. Exact nonnegative counts and finite identified fits; both stage FPC terms persist.",
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
    "openecon.multipletests": "Adjust one declared hypothesis family with explicit FWER/FDR and dependence assumptions. Preserve order and record any explicitly excluded missing tests.",
    "openecon.stepdown": "Apply Romano-Wolf maxT or Westfall-Young minP to justified supplied joint null draws. Specify the null design and Monte Carlo or complete-enumeration calibration.",
    "openecon.simultaneous_ci": "Construct full-covariance normal-limit simultaneous intervals with a local seed and recorded Monte Carlo precision. Supports CPU summaries only.",
    "openecon.ols_stepdown": "Generate joint-t maxT stepdown from one classical fixed-design OLS or known-precision aweight WLS result. Explicitly declare Gaussian errors; robust/cluster covariance and outcome-selected families are unsupported.",
    "openecon.simultaneous_t_ci": "Construct simultaneous intervals for known joint correlation and one independent common chi-square scale with a single df. Requires an explicit pivot justification; records Monte Carlo precision.",
    "openecon.hotelling_region": "Construct a Hotelling confidence ellipsoid for one iid multivariate-normal mean with unknown full covariance and n greater than dimensions. Coordinate projections are conservative; explicitly choose shared-row missing handling.",
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
    "openecon.survey_four_stage_design": {'tsu': 'Typed third-stage parent identity nested within each sampled SSU; sampled TSUs may contain multiple FSU rows.', 'fsu': 'Typed terminal fourth-stage identity; unique within its sampled TSU, one physical row per FSU.', 'population_tsu': 'Exact TSU population count constant within each sampled SSU.', 'population_fsu': 'Exact FSU population count constant within each sampled TSU.', 'max_memory_mb': 'Explicit resident complete four-stage geometry and inference allocation budget; default64MB, cap512MB.'},
    "openecon.survey_fully_stratified_three_stage_design": {'ssu_strata': 'Typed second-stage stratum identifier nested within each sampled PSU; cells sample SSUs independently.', 'ssu_frame': 'Explicit sequence of role-keyed mappings for every positive second-stage cell in sampled PSUs, with psu, ssu_strata, population_ssu and optional first-stage strata keys. No zero-sample positive cell or unknown PSU; provenance cannot authenticate externally omitted cells.', 'tsu_strata': 'Typed terminal sampling-stratum identity within each observed SSU; cells independently select terminal TSUs.', 'tsu_frame': 'Complete positive terminal-cell frame for every observed SSU; role keys strata when declared, psu, ssu_strata, ssu, tsu_strata and population_tsu. Every positive cell must have sampled rows. Supplied inputs cannot authenticate a cell omitted externally.', 'population_tsu': 'Exact TSU population count within each terminal sampling cell, not a pooled SSU count.', 'population_ssu': 'Exact SSU population count constant within each second-stage cell; not a pooled PSU count.', 'max_memory_mb': 'Explicit complete geometry, cell frame and inference allocation budget.'},
    "openecon.pca_fweight_bootstrap": {
        "weights": "Column of exact nonnegative counts of independent units; total at most 100000.",
        "weight_type": "Only fweight; no analytic/probability weights or physical-row resampling.",
    },
    "openecon.pca_subspace_fweight_bootstrap": {
        "weights": "Exact independent-unit counts; complete zeros persist but contribute no draws.",
        "weight_type": "Only fweight; each multinomial count draw refits its own moments.",
    },
    "openecon.canon_fweight_bootstrap": {
        "weights": "Exact independent-unit counts; total at most 100000, covariance divisor N-1.",
        "weight_type": "Only fweight, excluding correlated copies and survey/analytic weights.",
    },
    "openecon.factor_multifactor_fweight_bootstrap": {
        "weights": "Exact counts of literal independent units; no repeated measurement allocation.",
        "weight_type": "Only fweight; fixed count/target and full planned state retained.",
    },
    "openecon.survey_stratified_three_stage_design": {'ssu_strata': 'Typed second-stage stratum identifier nested within each sampled PSU; cells sample SSUs independently.', 'ssu_frame': 'Explicit sequence of role-keyed mappings for every positive second-stage cell in sampled PSUs, with psu, ssu_strata, population_ssu and optional first-stage strata keys. No zero-sample positive cell or unknown PSU; provenance cannot authenticate externally omitted cells.', 'population_ssu': 'Exact SSU population count constant within each second-stage cell; not a pooled PSU count.', 'max_memory_mb': 'Explicit complete geometry, cell frame and inference allocation budget.'},

    "openecon.pca_subspace_bootstrap": {
        "data": "Resident IID numeric rows; complete-case sample and typed identities persist.",
        "columns": "Ordered distinct variables; at most 15 within the 128-parameter joint projector bound.",
        "components": "Fixed leading rank, strictly below variable count; only its boundary gap must separate.",
        "matrix": "Covariance refits centring; correlation also refits each draw's SDs.",
        "replications": "19..1999 planned IID draws; any failure refuses the whole result.",
        "confidence": "Marginal percentile coverage level; no exact or simultaneous guarantee.",
        "seed": "Private CPU Torch seed; all drawn row indices persist.",
        "missing": "Whole-row complete-case drop or refusal.",
    },
    "openecon.canon_bootstrap": {
        "data": "Resident IID joint rows; full sample, means and scales persist.",
        "x": "Ordered distinct numeric X variables, disjoint from Y.",
        "y": "Ordered distinct numeric Y variables; at most 16 variables overall.",
        "components": "Fixed 1..4 positive simple interior roots; omitted boundary must separate.",
        "target": "Roots only, or roots plus raw/standard coefficients and within/cross loadings.",
        "anchors": "One fixed X sign anchor per coefficient axis; None uses unique point maxima. Root-only rejects anchors.",
        "replications": "19..1999 planned joint IID draws; all must remain admitted.",
        "confidence": "Marginal percentile level, without p-values or familywise inference.",
        "seed": "Private CPU Torch seed; no global RNG mutation.",
        "missing": "Whole joint-row complete-case drop or refusal.",
    },
    "openecon.ca_bootstrap": {
        "counts": "Labelled exact nonnegative integer table; 2..16 rows/columns, total at most 100000.",
        "dimensions": "Fixed 1..4 positive simple axes with a separated omitted boundary.",
        "sampling": "Fixed grand total, or independent fixed row totals; caller declares the experiment.",
        "anchors": "One fixed row sign anchor per axis; None uses unique point maxima.",
        "replications": "19..1999 complete count draws; failed geometry refuses the whole result.",
        "confidence": "Marginal percentile level; no independence-null or simultaneous inference.",
        "seed": "Private CPU Torch multinomial seed; all count tables persist.",
    },
    "openecon.pca_bootstrap_scores": {
        "result": "Complete identified pca_bootstrap TableSet; point and every draw must exactly replay.",
        "data": "Caller-fixed query rows in original variable units; at most 128 physical rows and joint scores.",
        "missing": "Preserve missing-query alignment with drop, or refuse incomplete rows.",
    },
    "openecon.SurveyTwoStageRegressionResult.predict": {
        "data": "At most 256 fixed resident evaluation rows; retain physical index and joint coefficient covariance.",
        "kind": "Conditional response mean or original linear predictor.",
        "alpha": "None retains the saved confidence level; positive-variance df zero has no interval.",
    },
    "openecon.survey_two_stage_design": {
        "psu": "Typed PSU identifier, nested within first-stage strata.",
        "ssu": "Typed SSU identifier, unique within each nested PSU; one sampled SSU per row.",
        "population_psu": "Exact population PSU count, constant within each stratum and at least its sampled PSU count.",
        "population_ssu": "Exact population SSU count, constant within each nested PSU and at least its sampled SSU count.",
        "strata": "Optional typed first-stage stratum identifier.",
        "max_rows": "Complete physical-row admission limit; exclusions do not shrink the declared design.",
        "max_memory_mb": "Explicit resident geometry and inference allocation budget.",
    },
    "openecon.nlogit": {
        "chosen": "Exact binary chosen indicator; one available choice per case.",
        "x": "1..8 explicit numeric attributes; no automatic intercept.",
        "nest": "Typed disjoint nest key; stable assignment for each alternative.",
        "fixed_dissimilarity": "Explicit nest/lambda pairs in [0.001,1]; omitted nonsingleton nests estimated jointly.",
        "vce": "Joint oim/case hc0/respondent cr0; no df multiplier.",
        "cluster": "Respondent key constant per case; required for cr0.",
    },
    "openecon.rologit": {
        "rank": "1 best, 0 unranked; strict contiguous prefix.",
        "x": "1..8 numeric attributes; no categories.",
        "case": "Typed ranking-case key; one summed score.",
        "alternative": "Typed key unique per case.",
        "available": "Boolean/0-1; rank0 when unavailable.",
        "vce": "oim/case hc0/cluster cr0; no df correction.",
        "cluster": "Constant per case; required for cr0.",
    },
    "openecon.survey_margins_replicate": {
        "profiles": "Ordered label-to-regressor-value mappings for mean targets; each fixes a nonempty proper subset and retains empirical remaining covariates.",
        "variables": "Distinct fitted continuous numeric regressors for AMEs; omit profiles.",
        "null": "Finite scalar or one ordered null value per empirical target, independent of coefficient width.",
        "replicate_weights": "Resident supplied weights in exact original physical row order, with distinct replica IDs and whole-PSU factors.",
    },
    "openecon.mediation_binary": {
        "data": "Complete resident numeric rows; at most 4096 observations and ten controls, with no silent missing-row deletion.",
        "mediator_link": "Binary mediator likelihood: logit or probit, exact numeric 0/1 with both levels.",
        "outcome_model": "Gaussian continuous outcome, logit/probit exact binary outcome, or Poisson integer counts with normalized likelihood.",
        "interaction": "Include the treatment-by-mediator term in the conditional outcome equation.",
        "covariance": "Full joint OIM or uncorrected HC0; HC0 retains the empirical cross-equation score blocks.",
        "interpretation": "Adjusted associational contrasts by default; causal labels require explicit identification assumptions.",
        "controls": "Distinct numeric controls in both equations; standardize over these fixed retained rows.",
    },
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


# Identification declarations and finite domains are part of public editor help.
for _name, _design in (
    ("randomization_confidence_set", "complete_randomized"),
    ("paired_randomization_confidence_set", "paired_randomized"),
    ("cluster_randomization_confidence_set", "cluster_randomized"),
):
    PARAMETER_DESCRIPTIONS["openecon." + _name] = {
        "candidates": "Fixed predeclared list of 1..64 distinct sharp constant effects; only this finite grid is inverted.",
        "design": "Explicitly declare '" + _design + "'; column labels cannot establish random assignment.",
        "level": "Coverage level applies only when the true constant effect belongs to the supplied candidate grid.",
    }
for _name, _design in (
    ("bernoulli_neyman_ate", "bernoulli_randomized"),
    ("multiarm_neyman_ate", "complete_randomized"),
    ("stratified_multiarm_neyman_ate", "stratified_randomized"),
):
    PARAMETER_DESCRIPTIONS["openecon." + _name] = {
        "design": "Explicitly declare '" + _design + "'. Conservative covariance and pointwise normal inference are asymptotic.",
        "probability": "Column of externally known true independent assignment probabilities strictly inside (0,1).",
        "arms": "Required ordered list of 3..8 distinct typed arm labels; at least two rows in every arm within every stratum.",
        "contrasts": "Ordered mapping of contrast names to coefficients whose represented sum is exactly zero; None selects all pairs.",
    }
for _name in ("treatment_effect_cdf_bounds", "treatment_effect_quantile_bounds"):
    PARAMETER_DESCRIPTIONS["openecon." + _name] = {
        "design": "Declare 'randomized'. Empirical-law identification bounds have no population sampling confidence interval.",
        "support": "Prespecified common support of at most eight numeric values with exactly representable pairwise differences.",
        "thresholds": "Fixed list of at most 64 individual-effect thresholds; each bound has its own complete optimal coupling.",
        "quantiles": "Fixed list of at most 16 probabilities in (0,1); exact integer left-quantile crossings, each with its own witness.",
    }

_IRT_CALIBRATED_PARAMETERS = {
    "bank": "Supplied fixed-calibration bank or its validated complete saved summary; no item fitting.",
    "posterior": "Validated complete finite-prior IRT posterior summary; every original response row retained.",
    "result": "Complete calibrated bank/posterior TableSet or full summary JSON; semantic and canonical validation required before reuse.",
    "items": "Distinct item names; predictive subsets retain the requested order.",
    "data": "Resident responses coded 0..K-1; missing items skipped and source positions/indices retained. No weights or Dataset route.",
    "discrimination": "Positive supplied item slopes; fixed and known, without calibration uncertainty.",
    "difficulty": "Supplied binary difficulty parameters on the declared latent scale.",
    "guessing": "Known binary lower asymptotes; default zero, not estimated.",
    "upper": "Known binary upper asymptotes; default one, not estimated.",
    "family": "grm, gpcm or nrm, supplied once or per item for a mixed bank.",
    "thresholds": "Per-item GRM ordered thresholds or GPCM unordered adjacent-step difficulties; None for NRM.",
    "slopes": "NRM category slopes with exact baseline zero; None for other families.",
    "intercepts": "NRM category intercepts with exact baseline zero; None for other families.",
    "scores": "Explicit nonnegative integer category-score maps 0..10; duplicates/gaps allowed, NRM requires a declared rule.",
    "support": "Sorted unique finite latent support in [-8,8], at most 101 points; the actual prior distribution.",
    "masses": "Positive prior masses summing to one; no continuous-prior quadrature claim.",
    "prior_mean": "Known normal-prior mean in [-4,4] for conditional MAP.",
    "prior_sd": "Known normal-prior SD in [.25,3]; reported score SD is local Laplace curvature only.",
    "theta_limit": "Interior mode search limit in [.25,12]; boundary or unbracketed scores refuse the call rather than clip.",
    "tolerance": "Absolute score-gradient convergence tolerance.",
    "max_iter": "Maximum safeguarded person-score iterations; complete traces retained.",
    "level": "MLE Wald confidence level or PMF future-score equal-tail probability; requires distinct interior tail probabilities in float64.",
    "draws": "1..99 independent draws per person from the exact declared finite posterior.",
    "seed": "Private CPU generator seed; does not change the caller's global random state.",
    "max_support": "Maximum complete raw score cells, up to 256; oversized support is refused before allocation.",
    "max_work": "Combined numerical validation and downstream work admitted before tensor allocation.",
    "max_bytes": "Named live-buffer budget, also limited by the global workspace budget; not a total process-memory limit.",
    "device": "cpu only; native float64 and explicit local device context.",
}
for _irt_name in (
    "irt_bank_binary", "irt_bank_polytomous", "irt_bank_restore", "irt_score_mle", "irt_score_map",
    "irt_posterior", "irt_posterior_restore", "irt_plausible_values", "irt_predictive", "irt_test_score_distribution",
):
    PARAMETER_DESCRIPTIONS["openecon." + _irt_name] = _IRT_CALIBRATED_PARAMETERS


for _frequency_name in (
    "catreg_nominal_fweight", "catreg_ordinal_fweight", "catreg_nominal_response_fweight",
    "catreg_ordinal_response_fweight", "catpca_fweight", "mca_fweight", "overals_fweight",
):
    PARAMETER_DESCRIPTIONS["openecon." + _frequency_name] = {
        "frequency": "Count column of nonnegative exact integers, supplied total at most 1e9; zero rows excluded. Not survey or precision weights.",
    }
PARAMETER_DESCRIPTIONS["openecon.catpca_category_centroids"] = {
    "transform": "Optional nonsingular component basis T: loadings A@T, scores/centroids X@inv(T).T; no automatic rotation.",
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
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = export_literal(node.left, names), export_literal(node.right, names)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        raise ValueError("Export addition requires literal strings")
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
        if generator.ifs or generator.is_async:
            raise ValueError("Only literal public-name export comprehensions are supported")
        values = export_literal(generator.iter, names)
        if not isinstance(values, (list, tuple)) or len(values) > 100:
            raise ValueError("Public-name export lists must be short literals")
        result = {}
        for value in values:
            bound = literal_bindings(generator.target, value, names)
            result[export_literal(node.key, bound)] = export_literal(node.value, bound)
        return result
    raise ValueError(f"Unsupported export syntax: {type(node).__name__}")


def literal_bindings(target, value, names):
    """Bind one finite literal comprehension item, without executing code."""
    if isinstance(target, ast.Name):
        return {**names, target.id: value}
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)):
        if len(target.elts) == len(value) and all(isinstance(x, ast.Name) for x in target.elts):
            return {**names, **{x.id: v for x, v in zip(target.elts, value, strict=True)}}
    raise ValueError("Comprehension bindings require names or fixed literal tuples")


def registry_exports(reader: SourceReader) -> dict[str, tuple[str, str, str | None]]:
    """Inspect committed manifests, including their bounded metadata factories.

    Only the literal function/entry/description fields of EstimatorInfo are
    read. A manifest factory must consist of one Return of EstimatorInfo;
    none of its option expressions or Python code is evaluated or imported.
    """
    registry = reader.committed("src/openecon/econometrics/registry.py")
    constructor = next((node for node in registry.body
                        if isinstance(node, ast.ClassDef) and node.name == "EstimatorInfo"), None)
    field_names = [] if constructor is None else [node.target.id for node in constructor.body
                   if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]
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
            if len(call.args) > len(field_names) or any(isinstance(x, ast.Starred) for x in call.args):
                raise ValueError("Estimator constructors require known literal field bindings")
            fields = dict(zip(field_names, call.args))
            for keyword in call.keywords:
                if (keyword.arg is None or keyword.arg in fields
                        or (field_names and keyword.arg not in field_names)):
                    raise ValueError("Estimator constructors have unknown or duplicate field bindings")
                fields[keyword.arg] = keyword.value
            function = export_literal(fields.get("function", ast.Constant(None)), bindings)
            if function is not None:
                target = export_literal(fields["entry"], bindings).split(":")[0] + ":" + function
                description = export_literal(fields.get("description", ast.Constant(None)), bindings)
                publish(function, target, summary(description) if description else None)

        def entries(value, bindings=None):
            bindings = names if bindings is None else bindings
            if isinstance(value, (ast.List, ast.Tuple)):
                for item in value.elts:
                    estimator(item, bindings)
            elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                # A repeated ESTIMATORS assignment appends an explicit tuple.
                if not (isinstance(value.left, ast.Name) and value.left.id == "ESTIMATORS"):
                    entries(value.left, bindings)
                entries(value.right, bindings)
            elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                  and value.func.id == "tuple" and len(value.args) == 1 and not value.keywords
                  and isinstance(value.args[0], ast.GeneratorExp)):
                expression = value.args[0]
                if len(expression.generators) != 1:
                    raise ValueError("Estimator comprehensions require one finite literal iterator")
                generator = expression.generators[0]
                if generator.ifs or generator.is_async:
                    raise ValueError("Estimator comprehension conditions are unsupported")
                values = export_literal(generator.iter, bindings)
                if not isinstance(values, (list, tuple)) or len(values) > 100:
                    raise ValueError("Estimator comprehensions require at most 100 literal items")
                for item in values:
                    estimator(expression.elt, literal_bindings(generator.target, item, bindings))
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
    mediation_options = function_node(
        reader.committed("src/openecon/econometrics/decomposition/binary.py").body, "_options")
    mediation_choices = {
        node.left.id: literals(ast.literal_eval(node.comparators[0]))
        for node in ast.walk(mediation_options)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id in {"mediator_link", "outcome_model", "covariance", "interpretation"}
        and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotIn)
        and isinstance(node.comparators[0], ast.Tuple)
    }
    target_options = class_node(reader.committed(
        "src/openecon/econometrics/survey/replicate_margins.py"), "_TargetTrace")
    target_choices = {
        node.left.id: literals(ast.literal_eval(node.comparators[0]))
        for node in ast.walk(function_node(target_options.body, "__init__"))
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id in {"family", "target"}
        and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotIn)
        and isinstance(node.comparators[0], ast.Set)
    }
    result = {
        "openecon.survey_margins_replicate": target_choices,
        "openecon.mediation_binary": mediation_choices,
        "openecon.ols": {
            "covariance": literals(constants["COVARIANCES"]),
            "weight_type": literals(constants["WEIGHTS"]),
            "kernel": literals(constants["KERNELS"]),
            "missing": missing_choices,
        },
        "openecon.logit": binary_choices,
        "openecon.probit": binary_choices,
    }
    two_stage_prepare = function_node(reader.committed(
        "src/openecon/econometrics/survey/two_stage_common.py").body, "prepare_two_stage")
    two_stage_missing = next(
        literals(ast.literal_eval(node.comparators[0]))
        for node in ast.walk(two_stage_prepare)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id == "missing" and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotIn) and isinstance(node.comparators[0], ast.Set)
    )
    for name in ("mean", "total", "ratio", "proportion", "regress", "logit", "probit", "poisson"):
        result["openecon.survey_two_stage_" + name] = {"missing": two_stage_missing}
    two_stage_prediction = function_node(reader.committed(
        "src/openecon/econometrics/survey/two_stage_regression_state.py").body, "predict")
    prediction_kinds = next(
        literals(ast.literal_eval(node.comparators[0]))
        for node in ast.walk(two_stage_prediction)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id == "kind" and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotIn) and isinstance(node.comparators[0], ast.Set)
    )
    result["openecon.SurveyTwoStageRegressionResult.predict"] = {
        "kind": prediction_kinds, "missing": two_stage_missing,
    }
    for module, name in (
        ("pca_subspace", "pca_subspace_bootstrap"),
        ("canon_uncertainty", "canon_bootstrap"),
        ("ca_uncertainty", "ca_bootstrap"),
        ("pca_score_uncertainty", "pca_bootstrap_scores"),
        ("pca_frequency_uncertainty", "pca_fweight_bootstrap"),
        ("pca_frequency_uncertainty", "pca_subspace_fweight_bootstrap"),
        ("canon_frequency_uncertainty", "canon_fweight_bootstrap"),
        ("factor_frequency_uncertainty", "factor_multifactor_fweight_bootstrap"),
    ):
        procedure = function_node(reader.committed(
            f"src/openecon/econometrics/multivariate/{module}.py").body, name)
        procedures = [procedure]
        public_parameters = {argument.arg for argument in
                             (*procedure.args.posonlyargs, *procedure.args.args, *procedure.args.kwonlyargs)}
        if module.endswith("_frequency_uncertainty"):
            procedures.append(function_node(reader.committed(
                "src/openecon/econometrics/multivariate/frequency_bootstrap.py").body, "prepare"))
            if module == "pca_frequency_uncertainty":
                procedures.append(function_node(reader.committed(
                    f"src/openecon/econometrics/multivariate/{module}.py").body, "_bootstrap"))
        result["openecon." + name] = {
            node.args[0].id: literals(ast.literal_eval(node.args[2]))
            for procedure_node in procedures for node in ast.walk(procedure_node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "check_choice" and len(node.args) == 3
            and isinstance(node.args[0], ast.Name) and isinstance(node.args[2], ast.Tuple)
            and node.args[0].id in public_parameters
        }
    three_stage_prepare = function_node(reader.committed(
        "src/openecon/econometrics/survey/three_stage_common.py").body, "prepare_three_stage")
    three_stage_missing = next(
        literals(ast.literal_eval(node.comparators[0]))
        for node in ast.walk(three_stage_prepare)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id == "missing" and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotIn) and isinstance(node.comparators[0], ast.Set)
    )
    for name in ("mean", "total", "ratio", "proportion", "regress", "logit", "probit", "poisson"):
        result["openecon.survey_three_stage_" + name] = {"missing": three_stage_missing}
    three_stage_prediction = function_node(reader.committed(
        "src/openecon/econometrics/survey/three_stage_regression_state.py").body, "predict")
    prediction_kinds = next(
        literals(ast.literal_eval(node.comparators[0]))
        for node in ast.walk(three_stage_prediction)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id == "kind" and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotIn) and isinstance(node.comparators[0], ast.Set)
    )
    result["openecon.SurveyThreeStageRegressionResult.predict"] = {
        "kind": prediction_kinds, "missing": three_stage_missing,
    }
    stratified_three_stage_prepare = function_node(reader.committed(
        "src/openecon/econometrics/survey/stratified_three_stage_common.py").body, "prepare_stratified_three_stage")
    stratified_three_stage_missing = next(
        literals(ast.literal_eval(node.comparators[0]))
        for node in ast.walk(stratified_three_stage_prepare)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id == "missing" and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotIn) and isinstance(node.comparators[0], ast.Set)
    )
    for name in ("mean", "total", "ratio", "proportion", "regress", "logit", "probit", "poisson"):
        result["openecon.survey_stratified_three_stage_" + name] = {"missing": stratified_three_stage_missing}
    stratified_three_stage_prediction = function_node(reader.committed(
        "src/openecon/econometrics/survey/stratified_three_stage_regression_state.py").body, "predict")
    prediction_kinds = next(
        literals(ast.literal_eval(node.comparators[0]))
        for node in ast.walk(stratified_three_stage_prediction)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id == "kind" and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotIn) and isinstance(node.comparators[0], ast.Set)
    )
    result["openecon.SurveyStratifiedThreeStageRegressionResult.predict"] = {
        "kind": prediction_kinds, "missing": stratified_three_stage_missing,
    }
    four_stage_prepare = function_node(reader.committed(
        "src/openecon/econometrics/survey/four_stage_common.py").body, "prepare_four_stage")
    four_stage_missing = next(
        literals(ast.literal_eval(node.comparators[0]))
        for node in ast.walk(four_stage_prepare)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id == "missing" and node.comparators
        and isinstance(node.comparators[0], (ast.Set, ast.Tuple, ast.List))
    )
    for name in ("mean", "total", "ratio", "proportion", "regress", "logit", "probit", "poisson"):
        result["openecon.survey_four_stage_" + name] = {"missing": four_stage_missing}
    four_stage_prediction = function_node(reader.committed(
        "src/openecon/econometrics/survey/four_stage_regression_state.py").body, "predict")
    prediction_kinds = next(
        literals(ast.literal_eval(node.comparators[0]))
        for node in ast.walk(four_stage_prediction)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id == "kind" and node.comparators
        and isinstance(node.comparators[0], (ast.Set, ast.Tuple, ast.List))
    )
    result["openecon.SurveyFourStageRegressionResult.predict"] = {
        "kind": prediction_kinds, "missing": four_stage_missing,
    }
    fully_stratified_three_stage_prepare = function_node(reader.committed(
        "src/openecon/econometrics/survey/fully_stratified_three_stage_common.py").body, "prepare_fully_stratified_three_stage")
    fully_stratified_three_stage_missing = next(
        literals(ast.literal_eval(node.comparators[0]))
        for node in ast.walk(fully_stratified_three_stage_prepare)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id == "missing" and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotIn) and isinstance(node.comparators[0], ast.Set)
    )
    for name in ("mean", "total", "ratio", "proportion", "regress", "logit", "probit", "poisson"):
        result["openecon.survey_fully_stratified_three_stage_" + name] = {"missing": fully_stratified_three_stage_missing}
    fully_stratified_three_stage_prediction = function_node(reader.committed(
        "src/openecon/econometrics/survey/fully_stratified_three_stage_regression_state.py").body, "predict")
    prediction_kinds = next(
        literals(ast.literal_eval(node.comparators[0]))
        for node in ast.walk(fully_stratified_three_stage_prediction)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
        and node.left.id == "kind" and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotIn) and isinstance(node.comparators[0], ast.Set)
    )
    result["openecon.SurveyFullyStratifiedThreeStageRegressionResult.predict"] = {
        "kind": prediction_kinds, "missing": fully_stratified_three_stage_missing,
    }
    calibrated = reader.committed("src/openecon/econometrics/irt/calibrated.py")
    constructor = function_node(calibrated.body, "irt_bank_polytomous")
    families = set()
    for comparison in ast.walk(constructor):
        if (isinstance(comparison, ast.Compare) and isinstance(comparison.left, ast.Name)
                and comparison.left.id == "kind" and len(comparison.comparators) == 1):
            accepted = comparison.comparators[0]
            if isinstance(comparison.ops[0], ast.In):
                families.update(ast.literal_eval(accepted))
            elif isinstance(comparison.ops[0], ast.Eq):
                families.add(ast.literal_eval(accepted))
    if families != {"grm", "gpcm", "nrm"}:
        raise ValueError("Calibrated IRT family choices changed; verify the declared help contract")
    result["openecon.irt_bank_polytomous"] = {"family": literals(sorted(families))}
    for name in (
        "irt_bank_binary", "irt_bank_polytomous", "irt_bank_restore", "irt_score_mle", "irt_score_map",
        "irt_posterior", "irt_posterior_restore", "irt_plausible_values", "irt_predictive", "irt_test_score_distribution",
    ):
        result.setdefault("openecon." + name, {})["device"] = literals(["cpu"])
    return result


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
        "survey_design",
        "survey_two_stage_design",
        "survey_three_stage_design",
        "survey_four_stage_design",
        "survey_fully_stratified_three_stage_design",
        "survey_stratified_three_stage_design",
    }
    return_types = {
        **{name: "openecon.TableSet" for name in ("repeated_gls", "restore_repeated_gls", "repeated_gls_predict", "repeated_gls_contrast")},
        **{name: "openecon.TableSet" for name in ("nlsur", "nlsur_restore", "nlsur_predict", "nlsur_margins", "nlsur_contrast")},
        **{name: "openecon.TableSet" for name in ("mprobit", "mprobit_restore", "mprobit_predict", "mprobit_margins")},
        **{name: "openecon.TableSet" for name in ("nlogit", "nlogit_restore", "nlogit_predict", "nlogit_margins")},
        **{name: "openecon.TableSet" for name in ("rologit", "rologit_restore", "rologit_predict", "rologit_margins")},
        "cf_predict": "openecon.DataFrame",
        "cf_restore": "openecon.ResultBundle",
        "mediation_binary": "openecon.TableSet", "mediation_binary_restore": "openecon.TableSet",
        "mi_discrete": "openecon.MIDiscreteResult", "mi_delta": "openecon.MIDeltaResult",
        "mi_passive": "openecon.MIPassiveResult", "mi_lincom": "openecon.MILincomResult",
        **{name: "openecon.TableSet" for name in (
            "stinterval_exponential", "stinterval_weibull", "stinterval_lognormal", "stinterval_loglogistic",
            "interval_survival_predict", "turnbull", "cumulative_incidence", "cause_specific_hazard", "cif_compare")},
        **{name: "openecon.TableSet" for name in ("mixed_satterthwaite", "trajectory_bands", "ols_cusum", "rm_contrast", "proxy_svar", "irt_fit_diagnostics", "irt_mh_dif", "irt_diagnostics_restore")},
        **{name: "openecon.DataFrame" for name in ("local_derivatives", "local_average_derivatives", "proxy_svar_irf")},
        **{name: "openecon.TableSet" for name in (
            "meta_dependent", "meta_dependent_robust", "meta_dependent_contrast",
            "meta_dependent_predict", "meta_dependent_predict_effect", "meta_dependent_diagnostics")},
        "ebalance": "openecon.TableSet", "cem": "openecon.TableSet", "balance": "openecon.TableSet",
        "rosenbaum_bounds": "openecon.TableSet", "paired_randomization": "openecon.TableSet",
        "treatment_cdf": "openecon.TableSet", "treatment_quantile": "openecon.TableSet", "treatment_rmst": "openecon.TableSet",
        "causal_design_save": "dict", "causal_design_load": "openecon.TableSet",
        **{name: "openecon.TableSet" for name in ('randomization_confidence_set', 'paired_randomization_confidence_set', 'cluster_randomization_confidence_set', 'bernoulli_neyman_ate', 'multiarm_neyman_ate', 'stratified_multiarm_neyman_ate', 'treatment_effect_cdf_bounds', 'treatment_effect_quantile_bounds')},
        **{name: "openecon.TableSet" for name in ('cluster_randomization', 'bernoulli_randomization', 'neyman_ate', 'stratified_neyman_ate', 'cluster_neyman_ate', 'paired_neyman_ate', 'manski_ate_inference', 'stratified_lee_bounds')},
        **{name: "openecon.TableSet" for name in ("ovb_sensitivity", "evalue", "manski_ate", "lee_bounds", "randomization_test", "stratified_randomization", "rosenbaum_rank_bounds", "bias_sensitivity")},
        **{name: "openecon.TableSet" for name in ("ovb_benchmark", "ovb_robustness", "treatment_cdf_ipw", "treatment_quantile_ipw", "treatment_cdf_aipw", "treatment_survival_ipcw", "treatment_rmst_ipcw", "treatment_rmst_aipw")},
        "mvnorm_em": "openecon.MIDiagnosticResult", "little_mcar": "openecon.MIDiagnosticResult",
        "mi_mvn": "openecon.MIResult", "mi_monotone": "openecon.MIResult",
        "mi_chained": "openecon.MIResult", "mi_pool": "openecon.MIPoolResult",
        "mi_test": "openecon.MIJointResult", "missing_patterns": "openecon.TableSet",
        "survey_margins_replicate": "openecon.SurveyReplicateMarginsResult",
        **{name: "openecon.SurveyRegressionResult" for name in ("survey_regress", "survey_logit", "survey_probit", "survey_poisson", "survey_regress_replicate", "survey_logit_replicate", "survey_probit_replicate", "survey_poisson_replicate")},
        **{name: "openecon.DataFrame" for name in ("survey_predict", "survey_margins", "survey_lincom", "survey_test")},
        "mean_sign_stepdown": "openecon.TableSet", "mean_permutation_stepdown": "openecon.TableSet",
        "simultaneous_dkw_band": "openecon.TableSet", "simultaneous_quantile_ci": "openecon.TableSet",
        "simultaneous_proportion_ci": "openecon.TableSet", "multinomial_region": "openecon.TableSet",
        "hoeffding_mean_ci": "openecon.TableSet", "empirical_bernstein_mean_ci": "openecon.TableSet",
        **{name: "openecon.TableSet" for name in
           ("conjoint_plan", "conjoint_orthogonal", "conjoint_diagnostics", "conjoint_fit", "conjoint_predict",
            "conjoint_holdout", "conjoint_importance", "conjoint_simulate", "conjoint_load")},
        "conjoint_save": "dict",
        **{name: "openecon.TableSet" for name in
           ("firth_logit", "firth_predict", "firth_profile", "firth_test", "exact_logistic", "exact_poisson_rate", "lts", "lts_predict", "finite_load")},
        **{name: "openecon.TableSet" for name in
           ("exact_logit_fit", "exact_logit_ci", "exact_logit_test", "exact_logit_moments", "exact_poisson_fit", "exact_poisson_ci", "exact_poisson_test", "exact_poisson_moments", "conditional_load")},
        "conditional_save": "dict",
        "finite_save": "dict",
        **{name: "openecon.SurveyResult" for name in ("survey_mean", "survey_total", "survey_ratio", "survey_proportion", "survey_brr", "survey_fay", "survey_jackknife", "survey_bootstrap")},
        **{name: "openecon.IRTResult" for name in
           ("irt_rasch", "irt_2pl", "irt_3pl", "irt_grm", "irt_pcm", "irt_rsm", "irt_restore")},
        "irt_score": "openecon.TableSet", "irt_information": "openecon.TableSet",
        **{name: "openecon.TableSet" for name in (
            "irt_bank_binary", "irt_bank_polytomous", "irt_bank_restore", "irt_score_mle", "irt_score_map",
            "irt_posterior", "irt_posterior_restore", "irt_plausible_values", "irt_predictive", "irt_test_score_distribution",
        )},
        "survey_design": "openecon.SurveyDesign",
        "survey_two_stage_design": "openecon.SurveyTwoStageDesign",
        **{name: "openecon.SurveyTwoStageResult" for name in (
            "survey_two_stage_mean", "survey_two_stage_total", "survey_two_stage_ratio", "survey_two_stage_proportion")},
        **{name: "openecon.SurveyTwoStageRegressionResult" for name in (
            "survey_two_stage_regress", "survey_two_stage_logit", "survey_two_stage_probit", "survey_two_stage_poisson")},
        "survey_three_stage_design": "openecon.SurveyThreeStageDesign",
        "survey_four_stage_design": "openecon.SurveyFourStageDesign",
        **{name: "openecon.SurveyFourStageResult" for name in (
            "survey_four_stage_mean", "survey_four_stage_total", "survey_four_stage_ratio", "survey_four_stage_proportion")},
        **{name: "openecon.SurveyFourStageRegressionResult" for name in (
            "survey_four_stage_regress", "survey_four_stage_logit", "survey_four_stage_probit", "survey_four_stage_poisson")},
        "survey_fully_stratified_three_stage_design": "openecon.SurveyFullyStratifiedThreeStageDesign",
        **{name: "openecon.SurveyFullyStratifiedThreeStageResult" for name in (
            "survey_fully_stratified_three_stage_mean", "survey_fully_stratified_three_stage_total", "survey_fully_stratified_three_stage_ratio", "survey_fully_stratified_three_stage_proportion")},
        **{name: "openecon.SurveyFullyStratifiedThreeStageRegressionResult" for name in (
            "survey_fully_stratified_three_stage_regress", "survey_fully_stratified_three_stage_logit", "survey_fully_stratified_three_stage_probit", "survey_fully_stratified_three_stage_poisson")},
        "survey_stratified_three_stage_design": "openecon.SurveyStratifiedThreeStageDesign",
        **{name: "openecon.SurveyStratifiedThreeStageResult" for name in (
            "survey_stratified_three_stage_mean", "survey_stratified_three_stage_total", "survey_stratified_three_stage_ratio", "survey_stratified_three_stage_proportion")},
        **{name: "openecon.SurveyStratifiedThreeStageRegressionResult" for name in (
            "survey_stratified_three_stage_regress", "survey_stratified_three_stage_logit", "survey_stratified_three_stage_probit", "survey_stratified_three_stage_poisson")},
        **{name: "openecon.SurveyThreeStageResult" for name in (
            "survey_three_stage_mean", "survey_three_stage_total", "survey_three_stage_ratio", "survey_three_stage_proportion")},
        **{name: "openecon.SurveyThreeStageRegressionResult" for name in (
            "survey_three_stage_regress", "survey_three_stage_logit", "survey_three_stage_probit", "survey_three_stage_poisson")},
        "power_mean": "openecon.TableSet", "power_proportion": "openecon.TableSet",
        "power_correlation": "openecon.TableSet", "precision_mean": "openecon.TableSet",
        **{name: "openecon.TableSet" for name in (
            "power_welch", "power_unbalanced_anova", "power_unequal_cluster_mean",
            "power_mcnemar_unconditional", "precision_twomeans_unknown",
            "precision_binomial", "precision_poisson", "survival_accrual")},
        "power_anova": "openecon.TableSet", "power_regression": "openecon.TableSet",
        "power_paired_mean": "openecon.TableSet", "power_cluster_mean": "openecon.TableSet",
        "power_logrank": "openecon.TableSet", "planning_scenarios": "openecon.TableSet",
        "planning_plot": "openecon_charts.PlotSpec",
        "summary_state": "str", "restore_summary": "openecon.TableSet",
        "iv_saved_weak_test": "openecon.TableSet", "iv_saved_ar_confidence_set": "openecon.TableSet",
        "wild_cluster_test": "openecon.TableSet", "wild_cluster_confidence_set": "openecon.TableSet",
        "fisher_johansen": "openecon.TableSet", "sar_iv_predict": "openecon.TableSet",
        "spatial_predict": "openecon.TableSet",
        "spatial_diagnostics": "openecon.TableSet",
        "panel_mmqr_bootstrap": "openecon.TableSet", "panel_mmqr_resampled_predict": "openecon.TableSet",
        "power_two_proportions": "openecon.TableSet",
        "power_two_correlations": "openecon.TableSet",
        "power_slope": "openecon.TableSet",
        "power_mcnemar": "openecon.TableSet",
        "precision_mean_unknown": "openecon.TableSet",
        "precision_variance": "openecon.TableSet",
        "bai_perron": "openecon.TableSet",
        "asymcausality": "openecon.TableSet", "effective_f": "openecon.TableSet",
        "panel_structural_predict": "openecon.TableSet", "recursive_ols": "openecon.TableSet",
        "rolling_panel": "openecon.TableSet",
        "pca_matrix": "openecon.TableSet", "factor_matrix": "openecon.TableSet",
        "factor_bootstrap": "openecon.TableSet",
        "discrim_summary": "openecon.TableSet", "canon_matrix": "openecon.TableSet",
        "canon_scores": "openecon.DataFrame | openecon.Dataset",
        "rm_anova_wide": "openecon.TableSet", "rm_restore": "openecon.TableSet",
        "rm_contrasts": "openecon.TableSet",
        **{name: "openecon.TableSet" for name in ("discrim_stepwise", "manova_oneway", "manova_summary", "manova_contrast", "rm_mtest")},
        "discrim_stepwise_predict": "openecon.DataFrame",
        "ca_project": "openecon.DataFrame",
        **{name: "openecon.TableSet" for name in ("catreg_nominal", "catreg_ordinal", "catreg_predict", "catpca", "catpca_predict", "mca", "mca_project", "overals", "mds_nonmetric", "loglinear_ipf", "loglinear_ml", "loglinear_compare", "catreg_nominal_response", "catreg_ordinal_response", "catreg_outcome_predict", "catreg_bootstrap", "catpca_varimax", "catpca_promax", "catpca_rotated_predict", "loglinear_multinomial", "loglinear_product_multinomial", "loglinear_sampling_compare", "loglinear_select")},
        **{name: "openecon.TableSet" for name in ("catreg_nominal_fweight", "catreg_ordinal_fweight", "catreg_nominal_response_fweight", "catreg_ordinal_response_fweight", "catreg_fweight_predict", "catpca_fweight", "catpca_fweight_predict", "catpca_category_centroids", "mca_fweight", "mca_fweight_project", "overals_fweight")},
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

    irt_tree = reader.committed("src/openecon/econometrics/irt/models.py")
    for name, node in methods(class_node(irt_tree, "IRTResult")).items():
        if not name.startswith("_"):
            add(entry(f"openecon.IRTResult.{name}", node, bound=True, kind="method"))

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
    add(entry("openecon.SurveyResult", pydantic_init, bound=True, kind="class", returns="openecon.SurveyResult"))
    for class_name, path in (
        ("MIDiscreteResult", "src/openecon/econometrics/mi/discrete.py"),
        ("MIDeltaResult", "src/openecon/econometrics/mi/sensitivity.py"),
        ("MIPassiveResult", "src/openecon/econometrics/mi/passive.py"),
        ("MILincomResult", "src/openecon/econometrics/mi/lincom.py"),
        ("MIResult", "src/openecon/econometrics/mi/common.py"),
        ("MIPoolResult", "src/openecon/econometrics/mi/joint.py"),
        ("MIJointResult", "src/openecon/econometrics/mi/joint.py"),
        ("MIDiagnosticResult", "src/openecon/econometrics/mi/diagnostics.py"),
    ):
        add(entry(f"openecon.{class_name}", pydantic_init, bound=True, kind="class",
                  returns=f"openecon.{class_name}",
                  description="Bounded immutable missing-data result with complete validated JSON state and explicit model/sample assumptions."))
        typed = class_node(reader.committed(path), class_name)
        inherited = class_node(reader.committed("src/openecon/econometrics/mi/common.py"), "MIResult") if class_name in {"MIDiscreteResult", "MIDeltaResult"} else typed
        for method in ("dataset", "to_latex"):
            if method in methods(inherited):
                add(entry(f"openecon.{class_name}.{method}", function_node(inherited.body, method),
                          bound=True, kind="method",
                          description="Return a new complete aligned imputation table." if method == "dataset" else "Export the result's full supported table to LaTeX.",
                          returns="openecon.DataFrame" if method == "dataset" else "str"))
        # Offer the complete JSON save/restore path. As with other result
        # classes, generic inherited BaseModel utilities need not be repeated
        # for every concrete owner in the bounded editor help payload.
        for method in ("model_validate_json", "model_dump_json"):
            base = class_node(reader.installed("pydantic", "pydantic/main.py"), "BaseModel")
            add(entry(f"openecon.{class_name}.{method}", function_node(base.body, method),
                      bound=True, kind="method",
                      description="Validate and restore complete typed result state." if "validate" in method else "Serialize the full typed result state including integrity and sample geometry.",
                      returns=f"openecon.{class_name}" if "validate" in method else "str" if method.endswith("json") else "dict"))
    add(entry("openecon.SurveyRegressionResult", pydantic_init, bound=True, kind="class", returns="openecon.SurveyRegressionResult"))
    regression_result = class_node(reader.committed("src/openecon/econometrics/survey/regression_common.py"), "SurveyRegressionResult")
    add(entry("openecon.SurveyRegressionResult.to_frame", function_node(regression_result.body, "to_frame"),
              bound=True, kind="method", returns="openecon.DataFrame"))
    for name in ("model_validate", "model_validate_json"):
        base = class_node(reader.installed("pydantic", "pydantic/main.py"), "BaseModel")
        add(entry(f"openecon.SurveyRegressionResult.{name}", function_node(base.body, name), bound=True,
                  kind="method", returns="openecon.SurveyRegressionResult"))
    add(entry("openecon.SurveyReplicateMarginsResult", pydantic_init, bound=True, kind="class",
              returns="openecon.SurveyReplicateMarginsResult",
              description="Complete empirical-target replicate state with primitive target and covariance replay; content digests are not authentication."))
    replicated_margins = class_node(reader.committed("src/openecon/econometrics/survey/replicate_margins_state.py"), "SurveyReplicateMarginsResult")
    add(entry("openecon.SurveyReplicateMarginsResult.to_frame", function_node(replicated_margins.body, "to_frame"),
              bound=True, kind="method", returns="openecon.DataFrame"))
    for name in ("model_validate_json", "model_dump_json"):
        base = class_node(reader.installed("pydantic", "pydantic/main.py"), "BaseModel")
        add(entry(f"openecon.SurveyReplicateMarginsResult.{name}", function_node(base.body, name), bound=True,
                  kind="method", returns="openecon.SurveyReplicateMarginsResult" if "validate" in name else "str",
                  description="Restore and replay full empirical-target state." if "validate" in name else "Serialize all empirical-target primitives and inference."))
    for class_name, path, description in (
        ("SurveyFourStageDesign", "src/openecon/survey_four_stage.py",
         "Immutable four-stage SRSWOR geometry with exact parent counts and derived weights."),
        ("SurveyFourStageResult", "src/openecon/econometrics/survey/four_stage_targets.py",
         "Joint target state retaining all four recursive FPC contributions and complete primitive replay."),
        ("SurveyFourStageRegressionResult", "src/openecon/econometrics/survey/four_stage_regression_state.py",
         "Regression state retaining all four score covariance contributions and complete primitive replay."),
        ("SurveyFullyStratifiedThreeStageDesign", "src/openecon/survey_fully_stratified_three_stage.py",
         "Immutable three-stage SRSWOR geometry with complete positive SSU/TSU stratum frames, exact cell counts and derived weights."),
        ("SurveyFullyStratifiedThreeStageResult", "src/openecon/econometrics/survey/fully_stratified_three_stage_targets.py",
         "Joint target state retaining both lower strata, all three FPC contributions and complete primitive covariance replay."),
        ("SurveyFullyStratifiedThreeStageRegressionResult", "src/openecon/econometrics/survey/fully_stratified_three_stage_regression_state.py",
         "Regression state retaining both lower strata, all three score covariance contributions and complete primitive replay."),
        ("SurveyStratifiedThreeStageDesign", "src/openecon/survey_stratified_three_stage.py",
         "Three actual SRSWOR selections with independently stratified stage-two SSUs and an explicit typed positive-cell frame."),
        ("SurveyStratifiedThreeStageResult", "src/openecon/econometrics/survey/stratified_three_stage_targets.py",
         "Joint target state retaining typed lower strata, full hierarchy and all three primitive covariance contributions."),
        ("SurveyStratifiedThreeStageRegressionResult", "src/openecon/econometrics/survey/stratified_three_stage_regression_state.py",
         "Regression state replaying cell-specific weights, score covariance and all three sampling-stage contributions."),
        ("SurveyThreeStageDesign", "src/openecon/survey_three_stage.py",
         "Immutable nested three-stage SRSWOR geometry with exact counts and derived weights."),
        ("SurveyThreeStageResult", "src/openecon/econometrics/survey/three_stage_targets.py",
         "Joint target state retaining all three stages and full primitive covariance replay."),
        ("SurveyThreeStageRegressionResult", "src/openecon/econometrics/survey/three_stage_regression_state.py",
         "Regression state with all three stage covariance terms and full primitive replay."),
        ("SurveyTwoStageDesign", "src/openecon/survey_two_stage.py",
         "Immutable nested two-stage SRSWOR geometry with exact stage counts and derived weights."),
        ("SurveyTwoStageResult", "src/openecon/econometrics/survey/two_stage_targets.py",
         "Validated joint target state retaining both sampling stages and complete primitive covariance replay."),
        ("SurveyTwoStageRegressionResult", "src/openecon/econometrics/survey/two_stage_regression_state.py",
         "Validated regression state with both stage score covariance terms and complete primitive replay."),
    ):
        stage_name = ("four-stage" if class_name.startswith("SurveyFourStage")
                      else "fully-stratified-three-stage" if class_name.startswith("SurveyFullyStratifiedThreeStage")
                      else "stratified-three-stage" if class_name.startswith("SurveyStratifiedThreeStage")
                      else "three-stage" if class_name.startswith("SurveyThreeStage") else "two-stage")
        canonical = "openecon." + class_name
        add(entry(canonical, pydantic_init, bound=True, kind="class", returns=canonical,
                  description=description))
        typed = class_node(reader.committed(path), class_name)
        method_descriptions = {
            "revalidate": "Recheck all stage inputs, physical row order, dtype and nested geometry against the saved declaration.",
            "to_frame": "Replay complete saved state and return estimates with reference first-stage t inference; df zero with positive variance has no test or interval.",
            "contrast": "Project the full joint target covariance onto one declared linear contrast.",
            "predict": "Predict from validated saved coefficients with full coefficient delta uncertainty; preserve missing-row alignment.",
            "lincom": "Project validated full two-stage coefficient covariance onto a declared linear combination.",
            "test": "Test declared joint coefficient restrictions using saved full two-stage covariance and first-stage reference df.",
        }
        for method, method_description in method_descriptions.items():
            if method in methods(typed):
                add(entry(canonical + "." + method, function_node(typed.body, method),
                          bound=True, kind="method", description=method_description.replace("two-stage", stage_name),
                          returns="openecon.DataFrame" if method != "revalidate" else ("FourStageValidation" if stage_name == "four-stage" else "FullyStratifiedThreeStageValidation" if stage_name == "fully-stratified-three-stage" else "StratifiedThreeStageValidation" if stage_name == "stratified-three-stage" else "ThreeStageValidation" if stage_name == "three-stage" else "TwoStageValidation")))
        for method in ("model_validate_json", "model_dump_json"):
            base = class_node(reader.installed("pydantic", "pydantic/main.py"), "BaseModel")
            add(entry(canonical + "." + method, function_node(base.body, method),
                      bound=True, kind="method", returns=canonical if "validate" in method else "str",
                      description=("Restore and validate full " + stage_name + " JSON state.") if "validate" in method else
                                  ("Serialize complete " + stage_name + " state, geometry and primitive inference records.")))
    survey_result = class_node(reader.committed("src/openecon/econometrics/survey/common.py"), "SurveyResult")
    for name in ("to_frame", "contrast"):
        add(entry(f"openecon.SurveyResult.{name}", function_node(survey_result.body, name), bound=True,
                  kind="method", returns="openecon.DataFrame"))
    for name in ("model_validate", "model_validate_json"):
        base = class_node(reader.installed("pydantic", "pydantic/main.py"), "BaseModel")
        add(entry(f"openecon.SurveyResult.{name}", function_node(base.body, name), bound=True,
                  kind="method", returns="openecon.SurveyResult"))
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


def _integer_default(text: str) -> int | None:
    """Compress only decimal spellings that survive Python and JS exactly."""
    if len(text) > 17 or re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text) is None:
        return None
    value = int(text)
    return value if abs(value) <= MAX_SAFE_INTEGER and str(value) == text else None


def _simple_quoted_default(text: str) -> bool:
    return (len(text) >= 2 and text.startswith("'") and text.endswith("'")
            and not any(character in text[1:-1] for character in ("'", "\\", "\n", "\r")))


def encode_catalog(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact only metadata that the editor can restore exactly."""
    stored = deepcopy(catalog)
    for item in stored:
        if "P" in item:
            raise ValueError("Canonical entry metadata cannot contain reserved parameter-list marker P")
        parameters_token = next((i for i, values in enumerate(STORED_PARAMETER_LISTS) if item["parameters"] == values), None)
        if "T" in item:
            raise ValueError("Canonical entry metadata cannot contain the reserved 'T' marker")
        if "I" in item:
            raise ValueError("Canonical entry metadata cannot contain the reserved 'I' marker")
        if item["name"].startswith("~"):
            raise ValueError("Canonical API names cannot start with the reserved '~' marker")
        signature_prefix = item["name"].rsplit(".", 1)[-1]
        if isinstance(item.get("signature"), str) and item["signature"].startswith(signature_prefix + "("):
            item["S"] = item.pop("signature")[len(signature_prefix):]
        elif isinstance(item.get("signature"), str) and item["signature"].startswith("("):
            raise ValueError("Logical signatures cannot start with reserved compact parenthesis")
        if item.get("S") in STORED_SIGNATURE_SUFFIXES:
            item["I"] = STORED_SIGNATURE_SUFFIXES.index(item.pop("S"))
        kind = item["kind"]
        if kind == "function":
            item.pop("kind")
        else:
            item["kind"] = STORED_ENTRY_KINDS.get(kind, kind)
        if kind == "method":
            if item["owner"] != item["name"].rsplit(".", 1)[0]:
                raise ValueError(f"Method owner does not match its canonical name: {item['name']}")
            item.pop("owner")
        # '~' cannot begin a Python API name; restore this shared prefix on load.
        if item["name"].startswith("openecon."):
            item["name"] = "~" + item["name"][len("openecon."):]
        for parameter in item["parameters"]:
            if "V" in parameter:
                raise ValueError("Canonical parameter metadata cannot contain the reserved 'V' marker")
            if "B" in parameter:
                raise ValueError("Canonical parameter metadata cannot contain the reserved 'B' marker")
            if "D" in parameter:
                raise ValueError("Canonical parameter metadata cannot contain the reserved 'D' marker")
            preserve_default = "default" in parameter and not isinstance(parameter["default"], str)
            if parameter["name"] in STORED_PARAMETER_NAMES:
                parameter["name"] = STORED_PARAMETER_NAMES.index(parameter["name"])
            # V:0 denotes the literal Python default string, not a JSON null.
            if parameter.get("default") == "None":
                parameter.pop("default")
                parameter["V"] = 0
            elif isinstance(parameter.get("default"), str) and parameter["default"] in ("False", "True"):
                parameter["B"] = int(parameter.pop("default") == "True")
            elif isinstance(parameter.get("default"), str) and parameter["default"] in STORED_DEFAULT_LITERALS:
                parameter["D"] = STORED_DEFAULT_LITERALS.index(parameter.pop("default"))
            if parameter.get("kind") == "positional-or-keyword":
                parameter.pop("kind")
            else:
                parameter_kind = parameter["kind"]
                parameter["kind"] = STORED_PARAMETER_KINDS.get(parameter_kind, parameter_kind)
            default = parameter.get("default")
            if isinstance(default, str) and default in {"True", "False"}:
                parameter["default"] = {"True": True, "False": False}[default]
            elif isinstance(default, str):
                integer = _integer_default(default)
                if integer is not None:
                    parameter["default"] = integer
                elif _simple_quoted_default(default):
                    parameter["q"] = parameter.pop("default")[1:-1]
            for key, token in STORED_PARAMETER_KEYS.items():
                if key == "default" and preserve_default:
                    continue
                if key in parameter:
                    parameter[token] = parameter.pop(key)
        # Exact common types have stable tokens; older namespace markers remain readable.
        returns = item.get("returns")
        if isinstance(returns, str) and returns in STORED_RETURN_TYPES:
            item["T"] = STORED_RETURN_TYPES[item.pop("returns")]
        elif isinstance(returns, str) and returns.startswith("openecon."):
            item["R"] = item.pop("returns")[len("openecon."):]
        for key, token in STORED_ENTRY_KEYS.items():
            # A literal leading tilde stays explicit to distinguish legacy r~.
            if key == "returns" and isinstance(item.get(key), str) and item[key].startswith("~"):
                continue
            if key in item:
                item[token] = item.pop(key)
        if parameters_token is not None:
            item.pop("p")
            item["P"] = parameters_token
    return stored


def decode_catalog(stored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore complete logical metadata for semantic catalog checks."""
    catalog = deepcopy(stored)
    entry_kinds = {token: kind for kind, token in STORED_ENTRY_KINDS.items()}
    parameter_kinds = {token: kind for kind, token in STORED_PARAMETER_KINDS.items()}
    return_types = {token: returns for returns, token in STORED_RETURN_TYPES.items()}
    for item in catalog:
        compact_signature = "signature" not in item and str(item.get("s", "")).startswith("(")
        if "r" in item:
            legacy = item.pop("r")
            if "returns" not in item:
                item["returns"] = ("openecon." + legacy[1:]
                                   if isinstance(legacy, str) and legacy.startswith("~") else legacy)
        for key, token in STORED_ENTRY_KEYS.items():
            if token in item:
                item.setdefault(key, item.pop(token))
        if "R" in item:
            suffix = item.pop("R")
            item.setdefault("returns", "openecon." + suffix)
        if "T" in item:
            token = item.pop("T")
            if "returns" not in item:
                if type(token) is not int or token not in return_types:
                    raise ValueError("Compact return type needs a recognized integer token")
                item["returns"] = return_types[token]
        if item["name"].startswith("~"):
            item["name"] = "openecon." + item["name"][1:]
        if compact_signature:
            item["signature"] = item["name"].rsplit(".", 1)[-1] + item["signature"]
        if "S" in item:
            suffix = item.pop("S")
            if "signature" not in item:
                if not isinstance(suffix, str) or not suffix.startswith("("):
                    raise ValueError("Compact signature needs a parenthesized suffix")
                item["signature"] = item["name"].rsplit(".", 1)[-1] + suffix
        if "I" in item:
            token = item.pop("I")
            if "signature" not in item:
                if type(token) is not int or not 0 <= token < len(STORED_SIGNATURE_SUFFIXES):
                    raise ValueError("Compact signature needs a recognized integer token")
                item["signature"] = item["name"].rsplit(".", 1)[-1] + STORED_SIGNATURE_SUFFIXES[token]
        kind = item.get("kind", "function")
        item["kind"] = entry_kinds.get(kind, kind)
        if item["kind"] == "method":
            item.setdefault("owner", item["name"].rsplit(".", 1)[0])
        if "P" in item:
            token = item.pop("P")
            if "parameters" not in item:
                if type(token) is not int or not 0 <= token < len(STORED_PARAMETER_LISTS):
                    raise ValueError("Compact parameters need a recognized integer token")
                item["parameters"] = deepcopy(STORED_PARAMETER_LISTS[token])
        for parameter in item["parameters"]:
            for key, token in STORED_PARAMETER_KEYS.items():
                if token in parameter:
                    value = parameter.pop(token)
                    if token == "n" and key not in parameter and not isinstance(value, str):
                        if (isinstance(value, bool) or not isinstance(value, (int, float))
                                or not 0 <= value < len(STORED_PARAMETER_NAMES) or int(value) != value
                                or isinstance(value, float) and value == 0 and str(value).startswith("-")):
                            raise ValueError("Compact parameter names must be strings or integer tokens from 0 to 30")
                        value = STORED_PARAMETER_NAMES[int(value)]
                    if token == "v" and key not in parameter:
                        if value is None or isinstance(value, bool):
                            value = "None" if value is None else "True" if value else "False"
                        elif isinstance(value, (int, float)):
                            if (not abs(value) <= MAX_SAFE_INTEGER or int(value) != value
                                    or isinstance(value, float) and value == 0 and str(value).startswith("-")):
                                raise ValueError("Compact numeric defaults must be safe decimal integers")
                            value = str(int(value))
                    parameter.setdefault(key, value)
            if "V" in parameter:
                literal = parameter.pop("V")
                if "default" not in parameter:
                    if type(literal) is not int or literal != 0:
                        raise ValueError("Compact default needs the integer zero None tag")
                    parameter["default"] = "None"
            if "B" in parameter:
                literal = parameter.pop("B")
                if "default" not in parameter:
                    if type(literal) is not int or literal not in (0, 1):
                        raise ValueError("Compact default needs the integer zero or one Boolean tag")
                    parameter["default"] = "True" if literal == 1 else "False"
            if "D" in parameter:
                literal = parameter.pop("D")
                if "default" not in parameter:
                    defaults = STORED_DEFAULT_LITERALS
                    if type(literal) is not int or not 0 <= literal < len(defaults):
                        raise ValueError("Compact default needs an integer fixed-literal tag from zero through fifteen")
                    parameter["default"] = defaults[literal]
            if "q" in parameter:
                quoted = parameter.pop("q")
                if "default" not in parameter:
                    if not isinstance(quoted, str) or not _simple_quoted_default("'" + quoted + "'"):
                        raise ValueError("Compact quoted defaults must contain an exact simple string interior")
                    parameter["default"] = "'" + quoted + "'"
            if "N" in parameter:
                literal = parameter.pop("N")
                if "default" not in parameter:
                    if type(literal) is not int or literal != 1:
                        raise ValueError("Legacy compact default needs the integer one None tag")
                    parameter["default"] = "None"
            # Both published compact kind keys remain readable; explicit kind
            # and the canonical k field take precedence over the t alias.
            if "t" in parameter:
                parameter.setdefault("kind", parameter.pop("t"))
            kind = parameter.get("kind", "positional-or-keyword")
            parameter["kind"] = parameter_kinds.get(kind, kind)
    return catalog


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
        if decode_catalog(saved) != catalog:
            raise SystemExit("Editor API catalog is stale; regenerate it with the packaging Python.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Encode entry/parameter kinds and derive method owners on editor loading.
        # Retain all signatures, defaults, choices, types and descriptions.
        stored = encode_catalog(catalog)
        # Whitespace-free JSON retains every logical entry under the size guard.
        serialized = json.dumps(stored, ensure_ascii=False, separators=(",", ":")) + "\n"
        args.output.write_text(serialized, encoding="utf-8")
    print(f"Verified {len(catalog)} editor API entries from committed source.")


if __name__ == "__main__":
    main()
