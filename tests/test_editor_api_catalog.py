"""Editor documentation must describe shipped APIs without starting Python jobs."""

import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_editor_api.py"
COMPACT_PARAMETER_NAMES = (
    "max_work", "max_iterations", "tolerance", "covariance", "max_bytes",
    "intercept", "missing", "instruments", "endogenous", "data",
    "alpha", "weights", "result", "device", "time", "cluster", "categorical",
    "seed", "weight_type", "confidence", "components", "replications", "max_iter", "options", "columns", "level", "method", "batch_rows", "treatment", "alternative", "max_memory_mb",
)
COMPACT_SIGNATURE_SUFFIXES = (
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
COMPACT_PARAMETER_LISTS = ([{'name': 'json_data', 'kind': 'positional-or-keyword'}, {'name': 'strict', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'extra', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'context', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'by_alias', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'by_name', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'indent', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'ensure_ascii', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'include', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'exclude', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'context', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'by_alias', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'exclude_unset', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'exclude_defaults', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'exclude_none', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'exclude_computed_fields', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'round_trip', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'warnings', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'fallback', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'serialize_as_any', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'polymorphic_serialization', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'obj', 'kind': 'positional-or-keyword'}, {'name': 'strict', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'extra', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'from_attributes', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'context', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'by_alias', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'by_name', 'kind': 'keyword-only', 'default': 'None'}], [{'kind': 'keyword-only', 'name': 'data'}, {'kind': 'keyword-only', 'name': 'y'}, {'kind': 'keyword-only', 'name': 'endogenous'}, {'kind': 'keyword-only', 'name': 'instruments'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'x'}, {'default': "'robust'", 'kind': 'keyword-only', 'name': 'covariance'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'cluster'}, {'default': 'True', 'kind': 'keyword-only', 'name': 'intercept'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'default': '100', 'kind': 'keyword-only', 'name': 'max_iterations'}, {'default': '1e-09', 'kind': 'keyword-only', 'name': 'tolerance'}, {'default': '10000000000', 'kind': 'keyword-only', 'name': 'max_work'}, {'default': "'cpu'", 'kind': 'keyword-only', 'name': 'device'}], [{'kind': 'positional-or-keyword', 'name': 'data'}, {'kind': 'positional-or-keyword', 'name': 'design'}, {'kind': 'positional-or-keyword', 'name': 'outcome'}, {'kind': 'positional-or-keyword', 'name': 'regressors'}, {'default': 'True', 'kind': 'keyword-only', 'name': 'intercept'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'domain'}, {'choices': ["'drop'", "'raise'"], 'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'default': '0.0', 'kind': 'keyword-only', 'name': 'null'}, {'default': '100', 'kind': 'keyword-only', 'name': 'max_iter'}, {'default': '1e-09', 'kind': 'keyword-only', 'name': 'tolerance'}], [{'kind': 'positional-or-keyword', 'name': 'data'}, {'kind': 'positional-or-keyword', 'name': 'design'}, {'kind': 'positional-or-keyword', 'name': 'outcome'}, {'kind': 'positional-or-keyword', 'name': 'regressors'}, {'kind': 'keyword-only', 'name': 'method'}, {'default': 'True', 'kind': 'keyword-only', 'name': 'intercept'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'domain'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'default': '0.0', 'kind': 'keyword-only', 'name': 'null'}, {'default': '100', 'kind': 'keyword-only', 'name': 'max_iter'}, {'default': '1e-09', 'kind': 'keyword-only', 'name': 'tolerance'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'replicates'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'replicate_weights'}, {'default': "'original'", 'kind': 'keyword-only', 'name': 'centering'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'rho'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'justification'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'scale'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'rscales'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'df'}], [{'kind': 'positional-or-keyword', 'name': 'low'}, {'kind': 'positional-or-keyword', 'name': 'indicator'}, {'kind': 'keyword-only', 'name': 'low_periods'}, {'kind': 'keyword-only', 'name': 'high_periods'}, {'default': "'Y'", 'kind': 'keyword-only', 'name': 'low_frequency'}, {'default': "'Q'", 'kind': 'keyword-only', 'name': 'high_frequency'}, {'default': "'sum'", 'kind': 'keyword-only', 'name': 'aggregation'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'as_of'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'low_releases'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'high_releases'}, {'default': "'cpu'", 'kind': 'keyword-only', 'name': 'device'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'weights'}], [{'kind': 'positional-or-keyword', 'name': 'data'}, {'kind': 'positional-or-keyword', 'name': 'design'}, {'kind': 'positional-or-keyword', 'name': 'outcome'}, {'kind': 'positional-or-keyword', 'name': 'regressors'}, {'default': 'True', 'kind': 'keyword-only', 'name': 'intercept'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'domain'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'default': '0.0', 'kind': 'keyword-only', 'name': 'null'}, {'default': '100', 'kind': 'keyword-only', 'name': 'max_iter'}, {'default': '1e-09', 'kind': 'keyword-only', 'name': 'tolerance'}], [{'kind': 'keyword-only', 'name': 'data'}, {'kind': 'keyword-only', 'name': 'items'}, {'default': '41', 'kind': 'keyword-only', 'name': 'points'}, {'default': '200', 'kind': 'keyword-only', 'name': 'max_iter'}, {'default': '400', 'kind': 'keyword-only', 'name': 'max_eval'}, {'default': '1e-06', 'kind': 'keyword-only', 'name': 'tolerance'}, {'default': '0.95', 'kind': 'keyword-only', 'name': 'level'}], [{'kind': 'positional-or-keyword', 'name': 'data'}, {'kind': 'positional-or-keyword', 'name': 'design'}, {'kind': 'positional-or-keyword', 'name': 'outcomes'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'domain'}, {'choices': ["'drop'", "'raise'"], 'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'default': '0.0', 'kind': 'keyword-only', 'name': 'null'}], [{'kind': 'keyword-only', 'name': 'data'}, {'kind': 'keyword-only', 'name': 'y'}, {'kind': 'keyword-only', 'name': 'x'}, {'kind': 'keyword-only', 'name': 'panel'}, {'kind': 'keyword-only', 'name': 'time'}, {'default': '1', 'kind': 'keyword-only', 'name': 'p'}, {'default': '1', 'kind': 'keyword-only', 'name': 'q'}, {'default': "'c'", 'kind': 'keyword-only', 'name': 'trend'}, {'default': '500', 'kind': 'keyword-only', 'name': 'max_iterations'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}], [{'kind': 'positional-or-keyword', 'name': 'lower'}, {'kind': 'positional-or-keyword', 'name': 'upper'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'times'}, {'default': '0.95', 'kind': 'keyword-only', 'name': 'level'}, {'default': "'cpu'", 'kind': 'keyword-only', 'name': 'device'}, {'default': 'None', 'kind': 'keyword-only', 'name': 'weights'}, {'default': '500', 'kind': 'keyword-only', 'name': 'maxiter'}], [{'kind': 'positional-or-keyword', 'name': 'result'}, {'kind': 'positional-or-keyword', 'name': 'data'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': 'DEFAULT_WORK', 'kind': 'keyword-only', 'name': 'max_work'}, {'default': 'DEFAULT_BYTES', 'kind': 'keyword-only', 'name': 'max_bytes'}, {'default': "'cpu'", 'kind': 'keyword-only', 'name': 'device'}], [{'kind': 'keyword-only', 'name': 'data'}, {'kind': 'keyword-only', 'name': 'y'}, {'kind': 'keyword-only', 'name': 'x'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'kind': 'var-keyword', 'name': 'options'}], [{'kind': 'positional-or-keyword', 'name': 'result'}, {'kind': 'keyword-only', 'name': 'data'}, {'kind': 'keyword-only', 'name': 'variable'}, {'default': '1', 'kind': 'keyword-only', 'name': 'order'}, {'default': 'False', 'kind': 'keyword-only', 'name': 'interval'}, {'default': '0.05', 'kind': 'keyword-only', 'name': 'alpha'}, {'default': "'raise'", 'kind': 'keyword-only', 'name': 'missing'}, {'default': '2000000000', 'kind': 'keyword-only', 'name': 'max_work'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weight_type', 'kind': 'keyword-only', 'default': "'aweight'"}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'l1_ratio', 'default': '0.5', 'kind': 'keyword-only'}, {'name': 'selection', 'kind': 'keyword-only', 'default': "'cv'"}, {'name': 'penalty', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'lambda_path', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'n_lambdas', 'default': '20', 'kind': 'keyword-only'}, {'name': 'lambda_ratio', 'default': '0.001', 'kind': 'keyword-only'}, {'name': 'folds', 'default': '5', 'kind': 'keyword-only'}, {'name': 'seed', 'default': '1729', 'kind': 'keyword-only'}, {'name': 'standardize', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'penalty_factors', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'forced_controls', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'max_iterations', 'default': '200', 'kind': 'keyword-only'}, {'name': 'tolerance', 'default': '1e-08', 'kind': 'keyword-only'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '2000000000'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'panel', 'kind': 'keyword-only'}, {'name': 'time', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'model', 'kind': 'keyword-only', 'default': "'re'"}, {'name': 'covariance', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intpoints', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intmethod', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'corr', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'corr_order', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'force', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'offset', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'low', 'kind': 'positional-or-keyword'}, {'name': 'indicator', 'kind': 'positional-or-keyword'}, {'name': 'rho', 'kind': 'keyword-only'}, {'name': 'low_periods', 'kind': 'keyword-only'}, {'name': 'high_periods', 'kind': 'keyword-only'}, {'name': 'low_frequency', 'kind': 'keyword-only', 'default': "'Y'"}, {'name': 'high_frequency', 'kind': 'keyword-only', 'default': "'Q'"}, {'name': 'aggregation', 'kind': 'keyword-only', 'default': "'sum'"}, {'name': 'as_of', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'low_releases', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'high_releases', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'design', 'kind': 'positional-or-keyword'}, {'name': 'outcome', 'kind': 'positional-or-keyword'}, {'name': 'categories', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'domain', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'missing', 'choices': ["'drop'", "'raise'"], 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'null', 'default': '0.0', 'kind': 'keyword-only'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'design', 'kind': 'positional-or-keyword'}, {'name': 'numerators', 'kind': 'positional-or-keyword'}, {'name': 'denominators', 'kind': 'positional-or-keyword'}, {'name': 'domain', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'missing', 'choices': ["'drop'", "'raise'"], 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'null', 'default': '0.0', 'kind': 'keyword-only'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'inflate', 'kind': 'keyword-only'}, {'name': 'inflate_link', 'kind': 'keyword-only', 'default': "'logit'"}, {'name': 'covariance', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weight_type', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'offset', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'exposure', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'dimensions', 'kind': 'positional-or-keyword'}, {'name': 'count', 'kind': 'positional-or-keyword'}, {'name': 'levels', 'kind': 'keyword-only'}, {'name': 'design', 'kind': 'keyword-only'}, {'name': 'terms', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'structural', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'offset', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'max_iter', 'default': '200', 'kind': 'keyword-only'}, {'name': 'tol', 'default': '1e-09', 'kind': 'keyword-only'}, {'name': 'level', 'kind': 'keyword-only', 'default': '0.95'}, {'name': 'max_work', 'default': '300000000', 'kind': 'keyword-only'}, {'name': 'max_bytes', 'default': '128 * 1024 ** 2', 'kind': 'keyword-only'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'group', 'kind': 'keyword-only'}, {'name': 'random', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intpoints', 'default': '7', 'kind': 'keyword-only'}, {'name': 'intmethod', 'kind': 'keyword-only', 'default': "'mvaghermite'"}, {'name': 'covariance', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'll', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'ul', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'covariance', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weight_type', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'offset', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'title', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'unit', 'kind': 'keyword-only', 'default': "''"}, {'name': 'palette', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'aggregate', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'raters', 'kind': 'positional-or-keyword'}, {'name': 'categories', 'kind': 'keyword-only'}, {'name': 'agreement_weights', 'kind': 'keyword-only', 'default': "'unweighted'"}, {'name': 'inference', 'kind': 'keyword-only', 'default': "'none'"}, {'name': 'level', 'kind': 'keyword-only', 'default': '0.95'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'drop'"}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'max_fits', 'default': '1024', 'kind': 'keyword-only'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '100000000'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'data', 'kind': 'keyword-only'}, {'name': 'variable', 'kind': 'keyword-only'}, {'name': 'order', 'default': '1', 'kind': 'keyword-only'}, {'name': 'interval', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '2000000000'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'time', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'trend', 'kind': 'keyword-only', 'default': "'c'"}, {'name': 'kernel', 'kind': 'keyword-only', 'default': "'bartlett'"}, {'name': 'bandwidth', 'default': '4', 'kind': 'keyword-only'}, {'name': 'df_adjust', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'covariance', 'choices': ["'cluster'", "'nonrobust'"], 'kind': 'keyword-only', 'default': 'None'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'missing', 'choices': ["'raise'", "'drop'"], 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'covariance', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'weight_type', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'offset', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'kind', 'choices': ["'linear'", "'response'"], 'kind': 'keyword-only', 'default': "'response'"}, {'name': 'missing', 'choices': ["'drop'", "'raise'"], 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'objective', 'kind': 'keyword-only', 'default': "'weight'"}, {'name': 'max_component_nodes', 'default': '24', 'kind': 'keyword-only'}, {'name': 'max_states', 'kind': 'keyword-only', 'default': '1000000'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '50000000'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'treatment', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'data', 'kind': 'var-keyword'}], [{'name': 'partition', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'objective', 'kind': 'keyword-only', 'default': "'weight'"}, {'name': 'max_matrix_entries', 'kind': 'keyword-only', 'default': '1000000'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '50000000'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'design', 'kind': 'positional-or-keyword'}, {'name': 'outcomes', 'kind': 'positional-or-keyword'}, {'name': 'domain', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'null', 'default': '0.0', 'kind': 'keyword-only'}, {'name': 'deff', 'kind': 'keyword-only', 'default': 'False'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'outcome', 'kind': 'positional-or-keyword'}, {'name': 'predictors', 'kind': 'positional-or-keyword'}, {'name': 'key', 'kind': 'keyword-only'}, {'name': 'spatial_weights', 'kind': 'keyword-only'}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'path', 'kind': 'positional-or-keyword'}, {'name': 'format', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'max_memory_mb', 'default': '256', 'kind': 'keyword-only'}, {'name': 'batch_rows', 'default': '65536', 'kind': 'keyword-only'}, {'name': 'max_file_mb', 'default': '512', 'kind': 'keyword-only'}], [{'description': 'A DataFrame, column dictionary, row records or batch Dataset source.', 'name': 'data', 'kind': 'keyword-only'}, {'description': 'Outcome column name; cannot be combined with formula.', 'name': 'y', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Predictor column names; cannot be combined with formula.', 'name': 'x', 'kind': 'keyword-only', 'default': 'None'}, {'description': "Example: 'wage ~ education + experience + C(region)'.", 'name': 'formula', 'kind': 'keyword-only', 'default': 'None'}, {'description': "Standard error method, such as 'nonrobust', 'HC3', 'cluster' or 'hac'.", 'name': 'covariance', 'choices': ["'nonrobust'", "'HC0'", "'HC1'", "'HC2'", "'HC3'", "'cluster'", "'cluster_hc2'", "'cluster_hc3'", "'hac'", "'bootstrap'", "'jackknife'"], 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Predictor names to encode as categorical variables.', 'name': 'categorical', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Include an intercept.', 'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'description': 'Column name or names for clustered standard errors.', 'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Column name containing the weights.', 'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'description': "'aweight', 'fweight', 'pweight' or 'iweight'.", 'name': 'weight_type', 'choices': ["'aweight'", "'fweight'", "'pweight'", "'iweight'"], 'kind': 'keyword-only', 'default': 'None'}, {'description': "'drop' removes observations with missing values; 'raise' reports an error.", 'name': 'missing', 'choices': ["'raise'", "'drop'"], 'kind': 'keyword-only', 'default': "'drop'"}, {'description': 'Inference significance level; 0.05 gives a 95% confidence interval.', 'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'description': 'Time column used for HAC or lagged formulas.', 'name': 'time', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'HAC lag count or a supported automatic selection name.', 'name': 'lags', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'HAC weighting kernel.', 'name': 'kernel', 'choices': ["'bartlett'", "'parzen'", "'quadratic_spectral'", "'truncated'"], 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Replication option for bootstrap or jackknife.', 'name': 'reps', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Random seed for resampling.', 'name': 'seed', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'dfadjust', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'hansen', 'kind': 'keyword-only', 'default': 'False'}, {'description': "'auto', 'cpu', 'cuda', 'cuda:<index>' or 'mps'. CUDA uses float64; Metal preconditions bounded factors with checked CPU float64 refinement.", 'name': 'device', 'kind': 'keyword-only', 'default': "'auto'"}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'endogenous', 'kind': 'keyword-only'}, {'name': 'instruments', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'covariance', 'kind': 'keyword-only', 'default': "'robust'"}, {'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'max_iterations', 'default': '100', 'kind': 'keyword-only'}, {'name': 'tolerance', 'default': '1e-09', 'kind': 'keyword-only'}, {'name': 'max_work', 'default': '10000000000', 'kind': 'keyword-only'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}, {'name': 'batch_rows', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'coefficients', 'kind': 'positional-or-keyword'}, {'name': 'null', 'default': '0.0', 'kind': 'keyword-only'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'time', 'kind': 'positional-or-keyword'}, {'name': 'event', 'kind': 'positional-or-keyword'}, {'name': 'causes', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'times', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'level', 'kind': 'keyword-only', 'default': '0.95'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'nodes', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'edges', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'graph_attributes', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'coefficients', 'kind': 'positional-or-keyword'}, {'name': 'null', 'default': '0.0', 'kind': 'keyword-only'}], [{'name': 'add', 'default': '()', 'kind': 'keyword-only'}, {'name': 'remove', 'default': '()', 'kind': 'keyword-only'}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'attributes', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'max_work', 'kind': 'keyword-only', 'default': '50000000'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'columns', 'kind': 'positional-or-keyword'}, {'name': 'bounds', 'kind': 'keyword-only'}, {'name': 'sampling_model', 'kind': 'keyword-only'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'y', 'kind': 'positional-or-keyword'}, {'description': '``n_dropped``) or "raise".', 'name': 'missing', 'kind': 'keyword-only', 'default': "'drop'"}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'data', 'kind': 'keyword-only'}, {'name': 'alignment', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'kind', 'kind': 'keyword-only', 'default': "'mean'"}], [{'name': 'path', 'kind': 'positional-or-keyword'}, {'name': 'overwrite', 'kind': 'keyword-only', 'default': 'False'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'args', 'kind': 'var-positional'}, {'name': 'kwargs', 'kind': 'var-keyword'}], [{'name': 'buf', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'kwargs', 'kind': 'var-keyword'}], [{'name': 'direction', 'kind': 'keyword-only', 'default': "'out'"}, {'name': 'disconnected', 'kind': 'keyword-only', 'default': "'infinite'"}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '50000000'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}, {'name': 'weights', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': 'DEFAULT_WORK'}, {'name': 'max_bytes', 'kind': 'keyword-only', 'default': 'DEFAULT_BYTES'}], [{'name': 'data', 'kind': 'keyword-only'}, {'name': 'x', 'kind': 'keyword-only'}, {'name': 'y', 'kind': 'keyword-only'}, {'name': 'title', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'attributes', 'kind': 'keyword-only', 'default': 'True'}], [{'name': 'restrictions', 'kind': 'positional-or-keyword'}, {'name': 'null', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'columns', 'kind': 'positional-or-keyword'}, {'name': 'sampling_model', 'kind': 'keyword-only'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}], [{'name': 'add', 'default': '()', 'kind': 'keyword-only'}, {'name': 'remove', 'default': '()', 'kind': 'keyword-only'}, {'name': 'rename', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'max_nodes', 'default': '1000', 'kind': 'positional-or-keyword'}, {'name': 'max_edges', 'default': '5000', 'kind': 'positional-or-keyword'}, {'name': 'seed', 'default': '0', 'kind': 'positional-or-keyword'}, {'name': 'groups', 'default': 'None', 'kind': 'positional-or-keyword'}], [{'name': 'options', 'kind': 'var-keyword'}], [{'name': 'source', 'kind': 'positional-or-keyword'}, {'name': 'target', 'kind': 'positional-or-keyword'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '50000000'}], [{'name': 'path', 'kind': 'positional-or-keyword'}, {'name': 'format', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'overwrite', 'kind': 'keyword-only', 'default': 'False'}], [{'name': 'restricted', 'kind': 'positional-or-keyword'}, {'name': 'full', 'kind': 'positional-or-keyword'}, {'name': 'max_bytes', 'default': '128 * 1024 ** 2', 'kind': 'keyword-only'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'columns', 'kind': 'positional-or-keyword'}, {'name': 'max_iterations', 'default': '500', 'kind': 'keyword-only'}, {'name': 'tolerance', 'default': '1e-08', 'kind': 'keyword-only'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'path', 'default': 'None', 'kind': 'positional-or-keyword'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'coefficient', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '100000000'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'level', 'kind': 'keyword-only', 'default': 'None'}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'null', 'default': '0.0', 'kind': 'keyword-only'}, {'name': 'max_work', 'kind': 'keyword-only', 'default': '100000000'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'series', 'kind': 'positional-or-keyword'}, {'name': 'options', 'kind': 'var-keyword'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'variables', 'kind': 'positional-or-keyword'}, {'name': 'knots', 'kind': 'keyword-only'}, {'name': 'components', 'default': '2', 'kind': 'keyword-only'}, {'name': 'n_starts', 'default': '3', 'kind': 'keyword-only'}, {'name': 'seed', 'kind': 'keyword-only', 'default': '0'}, {'name': 'maxiter', 'default': '500', 'kind': 'keyword-only'}, {'name': 'tol', 'default': '1e-08', 'kind': 'keyword-only'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'drop'"}, {'name': 'max_work', 'default': 'WORK', 'kind': 'keyword-only'}, {'name': 'max_bytes', 'default': 'BYTES', 'kind': 'keyword-only'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'outcome', 'kind': 'positional-or-keyword'}, {'name': 'predictors', 'kind': 'positional-or-keyword'}, {'name': 'knots', 'kind': 'keyword-only'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'drop'"}, {'name': 'max_bytes', 'default': 'c.LIMIT_BYTES', 'kind': 'keyword-only'}, {'name': 'max_work', 'default': 'c.WORK', 'kind': 'keyword-only'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}], [{'name': 'result', 'kind': 'positional-or-keyword'}, {'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'missing', 'kind': 'keyword-only', 'default': "'raise'"}, {'name': 'max_work', 'default': 'WORK', 'kind': 'keyword-only'}, {'name': 'max_bytes', 'default': 'BYTES', 'kind': 'keyword-only'}, {'name': 'device', 'kind': 'keyword-only', 'default': "'cpu'"}], [{'description': 'Any registry fit except time-series models and fits that already use a resampling covariance.', 'name': 'result', 'kind': 'positional-or-keyword'}, {'description': 'Exactly the dataset ``result`` was fitted on (verified by hash).', 'name': 'data', 'kind': 'positional-or-keyword'}, {'description': 'Number of bootstrap replications R (at least 2; 200 by default, use 1000 or more for percentile/bc intervals).', 'name': 'reps', 'default': '200', 'kind': 'keyword-only'}, {'description': 'Seed (0 to 2**64 - 1) of the ``torch.Generator`` that draws every replicate; the same seed reproduces the result exactly. A random seed is drawn and recorded when omitted.', 'name': 'seed', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Columns defining resampling clusters and strata.', 'name': 'cluster', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Columns defining resampling clusters and strata.', 'name': 'strata', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Units drawn per stratum.', 'name': 'size', 'kind': 'keyword-only', 'default': 'None'}, {'description': "Level of the intervals (default: the fit's alpha).", 'name': 'alpha', 'kind': 'keyword-only', 'default': 'None'}, {'description': "Interval type stored in ``extra['bootstrap_ci']``: normal based, percentile (Stata's ``_pctile`` definition) or bias-corrected percentile (z0 = Phi^-1(#{b_r <= b}/R)), BCa delete-unit acceleration or bootstrap-t using each refit's standard error. Advanced intervals currently require unweighted, unst", 'name': 'ci', 'kind': 'keyword-only', 'default': "'normal'"}, {'description': 'Dependent schemes validate explicit-column OLS. Blocks require a regular time column and block_length; wild_cluster requires cluster=.', 'name': 'scheme', 'kind': 'keyword-only', 'default': "'iid'"}, {'name': 'block_length', 'kind': 'keyword-only', 'default': 'None'}, {'name': 'time', 'kind': 'keyword-only', 'default': 'None'}, {'description': "Coefficient restrictions for restricted wild cluster-t tests; requires the original covariance clustered on the same column and ci='normal'. Confidence-set inversion is not supplied.", 'name': 'null', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'Wild cluster multiplier distribution.', 'name': 'wild', 'kind': 'keyword-only', 'default': "'rademacher'"}, {'description': 'Advanced draw plus BCa acceleration budget (default 10,000).', 'name': 'max_refits', 'default': '10000', 'kind': 'keyword-only'}], [{'description': "String, path object (implementing os.PathLike[str]), or file-like object implementing a write() function. If None, the result is returned as a string. If a non-binary file object is passed, it should be opened with `newline=''`, disabling universal newlines. If a binary file object is passed, `mode`", 'name': 'path_or_buf', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'String of length 1. Field delimiter for the output file.', 'name': 'sep', 'default': "','", 'kind': 'positional-or-keyword'}, {'description': 'Missing data representation.', 'name': 'na_rep', 'default': "''", 'kind': 'positional-or-keyword'}, {'description': 'Format string for floating point numbers. If a Callable is given, it takes precedence over other numeric formatting parameters, like decimal.', 'name': 'float_format', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Columns to write.', 'name': 'columns', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Write out the column names. If a list of strings is given it is assumed to be aliases for the column names.', 'name': 'header', 'default': 'True', 'kind': 'positional-or-keyword'}, {'description': 'Write row names (index).', 'name': 'index', 'default': 'True', 'kind': 'positional-or-keyword'}, {'description': 'Column label for index column(s) if desired. If None is given, and `header` and `index` are True, then the index names are used. A sequence should be given if the object uses MultiIndex. If False do not print fields for index names. Use index_label=False for easier importing in R.', 'name': 'index_label', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Forwarded to either `open(mode=)` or `fsspec.open(mode=)` to control the file opening. Typical values include:', 'name': 'mode', 'default': "'w'", 'kind': 'positional-or-keyword'}, {'description': "A string representing the encoding to use in the output file, defaults to 'utf-8'. `encoding` is not supported if `path_or_buf` is a non-binary file object.", 'name': 'encoding', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'compression', 'default': "'infer'", 'kind': 'positional-or-keyword'}, {'name': 'quoting', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'quotechar', 'default': '\'"\'', 'kind': 'positional-or-keyword'}, {'name': 'lineterminator', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'chunksize', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'date_format', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'doublequote', 'default': 'True', 'kind': 'positional-or-keyword'}, {'name': 'escapechar', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'decimal', 'default': "'.'", 'kind': 'positional-or-keyword'}, {'name': 'errors', 'default': "'strict'", 'kind': 'positional-or-keyword'}, {'name': 'storage_options', 'default': 'None', 'kind': 'positional-or-keyword'}], [{'description': 'File path or existing ExcelWriter.', 'name': 'excel_writer', 'kind': 'positional-or-keyword'}, {'description': 'Name of sheet which will contain DataFrame.', 'name': 'sheet_name', 'default': "'Sheet1'", 'kind': 'positional-or-keyword'}, {'description': 'Missing data representation.', 'name': 'na_rep', 'default': "''", 'kind': 'positional-or-keyword'}, {'description': 'Format string for floating point numbers. For example ``float_format="%.2f"`` will format 0.1234 to 0.12.', 'name': 'float_format', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Columns to write.', 'name': 'columns', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Write out the column names. If a list of string is given it is assumed to be aliases for the column names.', 'name': 'header', 'default': 'True', 'kind': 'positional-or-keyword'}, {'description': 'Write row names (index).', 'name': 'index', 'default': 'True', 'kind': 'positional-or-keyword'}, {'description': 'Column label for index column(s) if desired. If not specified, and `header` and `index` are True, then the index names are used. A sequence should be given if the DataFrame uses MultiIndex.', 'name': 'index_label', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Upper left cell row to dump data frame.', 'name': 'startrow', 'default': '0', 'kind': 'positional-or-keyword'}, {'description': 'Upper left cell column to dump data frame.', 'name': 'startcol', 'default': '0', 'kind': 'positional-or-keyword'}, {'description': "Write engine to use, 'openpyxl' or 'xlsxwriter'. You can also set this via the options ``io.excel.xlsx.writer`` or ``io.excel.xlsm.writer``.", 'name': 'engine', 'default': 'None', 'kind': 'positional-or-keyword'}, {'description': 'Write MultiIndex and Hierarchical Rows as merged cells.', 'name': 'merge_cells', 'default': 'True', 'kind': 'positional-or-keyword'}, {'description': 'Representation for infinity (there is no native representation for infinity in Excel).', 'name': 'inf_rep', 'default': "'inf'", 'kind': 'positional-or-keyword'}, {'description': 'Specifies the one-based bottommost row and rightmost column that is to be frozen.', 'name': 'freeze_panes', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'storage_options', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'engine_kwargs', 'default': 'None', 'kind': 'positional-or-keyword'}], [{'description': 'Mapping from unique predictor names to aligned Network snapshots; all dyads include absent edges as zero.', 'name': 'predictors', 'kind': 'positional-or-keyword'}, {'description': "'weight' uses aggregate strengths; 'binary' uses edge presence.", 'name': 'values', 'kind': 'keyword-only', 'default': "'weight'"}, {'description': 'Include diagonal dyads; False excludes them.', 'name': 'include_loops', 'kind': 'keyword-only', 'default': 'False'}, {'description': 'Uniform node permutations per coefficient-specific reduced-model residual test; plus-one p-values.', 'name': 'permutations', 'default': '999', 'kind': 'keyword-only'}, {'description': 'Private CPU Torch random seed; predictor names and node labels use canonical order.', 'name': 'seed', 'kind': 'keyword-only', 'default': '0'}, {'description': "'two-sided' compares absolute partial correlations; 'greater' and 'less' use the signed statistic.", 'name': 'alternative', 'kind': 'keyword-only', 'default': "'two-sided'"}, {'description': 'Fit a constant; the constant receives no fabricated permutation test.', 'name': 'intercept', 'kind': 'keyword-only', 'default': 'True'}, {'description': "'freedman_lane' permutes reduced response residuals; 'dsp' permutes focal predictor residuals and projects nuisance effects again.", 'name': 'method', 'kind': 'keyword-only', 'default': "'freedman_lane'"}, {'description': 'Named coefficient subsets tested together, conditional on remaining predictors; upper-tail partial R-squared.', 'name': 'joint', 'kind': 'keyword-only', 'default': 'None'}, {'description': "'none', 'bonferroni', 'holm' or 'bh' over all reported slopes and joint hypotheses; the intercept is excluded.", 'name': 'adjustment', 'kind': 'keyword-only', 'default': "'none'"}, {'description': 'Explicit complete-test structural work budget; no partial permutation p-values.', 'name': 'max_work', 'kind': 'keyword-only', 'default': '50000000'}], [{'name': 'data', 'kind': 'positional-or-keyword'}, {'name': 'y', 'kind': 'positional-or-keyword'}, {'name': 'factors', 'default': 'None', 'kind': 'positional-or-keyword'}, {'name': 'covariates', 'kind': 'keyword-only', 'default': 'None'}, {'description': '``"none"`` (main effects only) or a list such as ``[["a", "b"]]`` or ``["a#b"]``; a covariate may appear in an interaction. An interaction needs the effects it contains in the model (``a#b#c`` needs ``a#b``, ``a#c`` and ``b#c``; ``a#b#x`` needs ``a#x`` and ``b#x``), otherwise ``invalid_spec``.', 'name': 'interactions', 'kind': 'keyword-only', 'default': "'full'"}, {'name': 'ss_type', 'default': '3', 'kind': 'keyword-only'}, {'description': '``oe.rm_anova`` with ``factors`` as between-subject factors (Type III, all between-subject interactions; covariates, ``emmeans`` and other values of ``ss_type`` or ``interactions`` are rejected).', 'name': 'within', 'kind': 'keyword-only', 'default': 'None'}, {'description': '``oe.rm_anova`` with ``factors`` as between-subject factors (Type III, all between-subject interactions; covariates, ``emmeans`` and other values of ``ss_type`` or ``interactions`` are rejected).', 'name': 'subject', 'kind': 'keyword-only', 'default': 'None'}, {'description': 'reported: predictions averaged with equal weights over the levels of the other factors, with covariates at their sample means. Single factors also get pairwise comparisons.', 'name': 'emmeans', 'kind': 'keyword-only', 'default': 'None'}, {'description': "SPSS's default) for the pairwise comparisons and their intervals.", 'name': 'adjust', 'kind': 'keyword-only', 'default': "'bonferroni'"}, {'description': 'tests the ANCOVA assumption of equal slopes.', 'name': 'homogeneity_of_slopes', 'kind': 'keyword-only', 'default': 'False'}, {'name': 'alpha', 'kind': 'keyword-only', 'default': '0.05'}, {'description': '``"raise"``.', 'name': 'missing', 'kind': 'keyword-only', 'default': "'drop'"}])
spec = importlib.util.spec_from_file_location("editor_api_generator", SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def decoded_catalog(entries):
    # Inspect complete logical metadata, as the editor does on loading.
    entries = deepcopy(entries)
    decoded_kinds = {"k": "keyword-only", "p": "positional-only",
                     "*": "var-positional", "**": "var-keyword"}
    decoded_entry_kinds = {"m": "method", "c": "class"}
    for entry in entries:
        compact_signature = "signature" not in entry and str(entry.get("s", "")).startswith("(")
        for key, token in {"signature": "s", "description": "d",
                           "name": "n", "parameters": "p", "kind": "k"}.items():
            if token in entry:
                entry.setdefault(key, entry.pop(token))
        if "r" in entry:
            legacy_returns = entry.pop("r")
            if isinstance(legacy_returns, str) and legacy_returns.startswith("~"):
                legacy_returns = "openecon." + legacy_returns[1:]
            entry.setdefault("returns", legacy_returns)
        if "R" in entry:
            entry.setdefault("returns", "openecon." + entry.pop("R"))
        if "T" in entry:
            token = entry.pop("T")
            if "returns" not in entry:
                types = {0: "openecon.TableSet", 1: "openecon.ResultBundle", 2: "openecon.DataFrame"}
                if type(token) is not int or token not in types:
                    raise ValueError("Compact return type needs a recognized integer token")
                entry["returns"] = types[token]
        if entry["name"].startswith("~"):
            entry["name"] = "openecon." + entry["name"][1:]
        if compact_signature:
            entry["signature"] = entry["name"].rsplit(".", 1)[-1] + entry["signature"]
        if "S" in entry:
            suffix = entry.pop("S")
            if "signature" not in entry:
                if not isinstance(suffix, str) or not suffix.startswith("("):
                    raise ValueError("Compact signature needs a parenthesized suffix")
                entry["signature"] = entry["name"].rsplit(".", 1)[-1] + suffix
        if "I" in entry:
            token = entry.pop("I")
            if "signature" not in entry:
                if type(token) is not int or not 0 <= token < len(COMPACT_SIGNATURE_SUFFIXES):
                    raise ValueError("Compact signature needs a recognized integer token")
                entry["signature"] = entry["name"].rsplit(".", 1)[-1] + COMPACT_SIGNATURE_SUFFIXES[token]
        kind = entry.get("kind", "function")
        entry["kind"] = decoded_entry_kinds.get(kind, kind)
        if entry["kind"] == "method":
            entry.setdefault("owner", entry["name"].rsplit(".", 1)[0])
        if "P" in entry:
            token = entry.pop("P")
            if "parameters" not in entry:
                if type(token) is not int or not 0 <= token < len(COMPACT_PARAMETER_LISTS):
                    raise ValueError("Compact parameters need a recognized integer token")
                entry["parameters"] = deepcopy(COMPACT_PARAMETER_LISTS[token])
        for parameter in entry["parameters"]:
            if "v" in parameter:
                default = parameter.pop("v")
                if "default" not in parameter:
                    if default is None or isinstance(default, bool):
                        default = "None" if default is None else "True" if default else "False"
                    elif isinstance(default, (int, float)):
                        if (abs(default) > 2**53 - 1 or not math.isfinite(default)
                                or default != int(default) or (default == 0 and math.copysign(1, default) < 0)):
                            raise ValueError("Compact numeric default needs a safe integer")
                        default = str(int(default))
                    parameter["default"] = default
            for token, key in {"d": "description", "n": "name", "a": "annotation",
                               "c": "choices", "k": "kind", "t": "kind"}.items():
                if token in parameter:
                    value = parameter.pop(token)
                    if token == "n" and "name" not in parameter and not isinstance(value, str):
                        if (isinstance(value, bool) or not isinstance(value, (int, float))
                                or not 0 <= value < len(COMPACT_PARAMETER_NAMES)
                                or not math.isfinite(value) or value != int(value)
                                or (value == 0 and math.copysign(1, value) < 0)):
                            raise ValueError("Compact parameter name needs a valid integer index")
                        value = COMPACT_PARAMETER_NAMES[int(value)]
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
                    defaults = ("'raise'", '0.05', "'cpu'", '100000000', '50000000', "'drop'", '0.95', '0', 'DEFAULT_BYTES', 'DEFAULT_WORK', "'two-sided'", "'nonrobust'", "'constant'", '1000000', '2000000000', '100000')
                    if type(literal) is not int or not 0 <= literal < len(defaults):
                        raise ValueError("Compact default needs an integer fixed-literal tag from zero through fifteen")
                    parameter["default"] = defaults[literal]
            if "q" in parameter:
                quoted = parameter.pop("q")
                if "default" not in parameter:
                    if not isinstance(quoted, str) or any(char in quoted for char in ("'", "\\", "\n", "\r")):
                        raise ValueError("Compact quoted default needs an exact simple string")
                    parameter["default"] = "'" + quoted + "'"
            if "N" in parameter:
                literal = parameter.pop("N")
                if "default" not in parameter:
                    if type(literal) is not int or literal != 1:
                        raise ValueError("Legacy compact default needs the integer one None tag")
                    parameter["default"] = "None"
            kind = parameter.get("kind", "positional-or-keyword")
            parameter["kind"] = decoded_kinds.get(kind, kind)
    return entries


def saved_catalog():
    entries = json.loads((ROOT / "web/src/editor-api.json").read_text(encoding="utf-8"))
    entries = decoded_catalog(entries)
    return {entry["name"]: entry for entry in entries}


@pytest.mark.parametrize("batch,tokens", [(2, [37, 38, 39]), (3, [40]), (4, [41]), (5, [72, 73, 74])])
def test_integration_complete_parameter_templates_preserve_golden_fields_in_all_readers(batch, tokens):
    fixtures = json.loads((ROOT / f"web/tests/fixtures/integration{batch}-parameter-templates.json").read_text())
    assert [item["token"] for item in fixtures] == tokens
    preservation_spec = importlib.util.spec_from_file_location(
        "integration_editor_preservation", ROOT / "scripts/verify_editor_api_preserved.py"
    )
    preservation = importlib.util.module_from_spec(preservation_spec)
    preservation_spec.loader.exec_module(preservation)
    for fixture in fixtures:
        stored = [{"n": "~integration.example", "S": "()", "d": "Complete original fields.", "P": fixture["token"]}]
        before = deepcopy(stored)
        outputs = [
            decoded_catalog(stored)[0], generator.decode_catalog(stored)[0],
            preservation.logical(json.dumps(stored))["openecon.integration.example"],
        ]
        for decoded in outputs:
            assert decoded["parameters"] == fixture["parameters"]
            decoded["parameters"][0]["description"] = "Locally changed"
            with_choices = next((parameter for parameter in decoded["parameters"] if "choices" in parameter), None)
            if with_choices is not None:
                with_choices["choices"].append("'local'")
        assert stored == before
        assert generator.decode_catalog(stored)[0]["parameters"] == fixture["parameters"]
        assert decoded_catalog(stored)[0]["parameters"] == fixture["parameters"]
        assert preservation.logical(json.dumps(stored))["openecon.integration.example"]["parameters"] == fixture["parameters"]


def test_none_default_compaction_is_exact_lossless_and_does_not_mutate():
    defaults = ["None", "False", "0", "", "'None'", "None ", None, False, 0]
    logical = [{
        "name": "openecon.example", "kind": "function", "signature": "example()",
        "description": "Keep every default literally.", "returns": "openecon.TableSet",
        "parameters": [{"name": f"p{i}", "kind": "keyword-only", "default": value}
                       for i, value in enumerate(defaults)] + [{"name": "required", "kind": "keyword-only"}],
    }]
    before = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    assert stored[0]["p"][0]["V"] == 0 and "v" not in stored[0]["p"][0]
    for parameter, default in zip(stored[0]["p"][1:], defaults[1:]):
        assert "V" not in parameter and "N" not in parameter
        if isinstance(default, str) and default in ("False", "True"):
            assert parameter["B"] == int(default == "True") and "v" not in parameter
        elif default == "0":
            assert parameter["D"] == 7 and "v" not in parameter
        elif default == "'None'":
            assert parameter["q"] == "None" and "v" not in parameter
        elif isinstance(default, str):
            assert parameter["v"] == default and "B" not in parameter
        else:
            assert parameter["default"] == default and type(parameter["default"]) is type(default)
            assert "v" not in parameter and "B" not in parameter
    assert "V" not in stored[0]["p"][-1] and "N" not in stored[0]["p"][-1] and "v" not in stored[0]["p"][-1]
    encoded_before = deepcopy(stored)
    assert generator.decode_catalog(stored) == decoded_catalog(stored) == logical
    assert stored == encoded_before and logical == before


@pytest.mark.parametrize("fields,expected", [
    ({"N": 1}, "None"),
    ({"v": "", "N": 1}, ""),
    ({"v": "0", "N": 1}, "0"),
    ({"default": "False", "v": "0", "N": 1}, "False"),
    ({"default": "", "v": "None", "N": 1}, ""),
])
def test_none_default_alias_respects_full_and_compact_default_precedence(fields, expected):
    stored = [{"n": "~example", "S": "()", "d": "Complete help.",
               "p": [{"n": "value", **fields}], "R": "TableSet"}]
    before = deepcopy(stored)
    decoded = generator.decode_catalog(stored)
    assert decoded == decoded_catalog(stored)
    assert decoded[0]["parameters"][0] == {
        "name": "value", "default": expected, "kind": "positional-or-keyword",
    }
    assert stored == before


def test_mi_extension_help_preserves_typed_results_and_scientific_scope():
    catalog = saved_catalog()
    expected = {"mi_discrete": ("MIDiscreteResult", "unassessed"),
                "mi_delta": ("MIDeltaResult", "unidentified"),
                "mi_passive": ("MIPassiveResult", "no FCS feedback"),
                "mi_lincom": ("MILincomResult", "no joint test")}
    for helper, (result, scope) in expected.items():
        entry = catalog[f"openecon.{helper}"]
        assert entry["returns"] == f"openecon.{result}"
        assert scope in entry["description"]
        assert catalog[f"openecon.{result}.model_validate_json"]["owner"] == f"openecon.{result}"
        assert catalog[f"openecon.{result}.model_dump_json"]["returns"] == "str"


def test_compact_catalog_roundtrip_preserves_every_field_and_legacy_explicit_fields():
    logical = [{"name": "example.function", "kind": "function",
                "signature": "function(x, /, optional='text', *args, flag=None, **kwargs)",
                "description": "Entry description.", "returns": "example.Result",
                "parameters": [
                    {"name": "x", "kind": "positional-only", "description": "Required value."},
                    {"name": "optional", "kind": "positional-or-keyword", "default": "'text'",
                     "description": "Parameter description.", "choices": ["'text'", "'latex'"]},
                    {"name": "args", "kind": "var-positional"},
                    {"name": "flag", "kind": "keyword-only", "default": "None", "description": "Optional flag."},
                    {"name": "kwargs", "kind": "var-keyword"}]}]
    encoded = generator.encode_catalog(logical)
    optional = encoded[0]["p"][1]
    assert optional["q"] == "text" and optional["d"] == "Parameter description."
    assert "v" not in optional
    assert "default" not in optional and "description" not in optional
    assert encoded[0]["S"] == logical[0]["signature"][len("function"):]
    assert encoded[0]["d"] == logical[0]["description"]
    assert "kind" not in encoded[0] and "kind" not in optional
    assert generator.decode_catalog(json.loads(json.dumps(encoded))) == logical
    assert generator.decode_catalog(logical) == logical
    assert logical[0]["parameters"][1]["default"] == "'text'"
    assert logical[0]["parameters"][1]["description"] == "Parameter description."


def test_compact_generated_catalog_preserves_all_fields_without_writing():
    logical = generator.generate("HEAD")
    stored = generator.encode_catalog(logical)
    # Keep the original conservative whitespace allowance as well as roundtrip equality.
    serialized = "[\n" + ",\n".join(
        json.dumps(entry, ensure_ascii=False, separators=(",", ":")) for entry in stored
    ) + "\n]\n"
    assert generator.decode_catalog(json.loads(serialized)) == logical
    assert len(serialized.encode("utf-8")) < 500_000


def test_shared_control_function_template_preserves_current_catalog_and_original_byte_budget():
    fixture = json.loads((ROOT / "web/tests/fixtures/integration4-parameter-templates.json").read_text())[0]
    assert fixture["token"] == 41
    names = {
        "openecon.cfcloglog", "openecon.cffraclogit", "openecon.cfgamma",
        "openecon.cfinvgauss", "openecon.cflogit", "openecon.cfpoisson",
        "openecon.cfprobit", "openecon.cfregress",
    }
    logical = decoded_catalog(json.loads((ROOT / "web/src/editor-api.json").read_text()))
    assert {entry["name"] for entry in logical if entry["parameters"] == fixture["parameters"]} == names
    stored = generator.encode_catalog(logical)
    assert {entry["name"] for entry, wire in zip(logical, stored) if wire.get("P") == 41} == names
    for decode in compact_default_readers()[:3]:
        assert decode(stored) == logical
    # Expand only the shared control-function slot; retain later independent slots.
    literal = deepcopy(stored)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(generator, "STORED_PARAMETER_LISTS", generator.STORED_PARAMETER_LISTS[:41])
        for entry, wire in zip(logical, literal):
            if wire.get("P") == 41:
                wire.pop("P")
                wire["p"] = generator.encode_catalog([entry])[0]["p"]
    compact = "[\n" + ",\n".join(json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                                  for entry in stored) + "\n]\n"
    before = "[\n" + ",\n".join(json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                                 for entry in literal) + "\n]\n"
    assert len(before.encode()) - len(compact.encode()) == 2_704
    assert len(compact.encode()) < 500_000
    assert generator.decode_catalog(literal) == logical


def test_compact_return_prefixes_keep_typed_help_and_legacy_precedence():
    logical = [{"name": f"openecon.example_{i}", "kind": "function",
                "signature": "example()", "description": "Complete help.",
                "parameters": [], "returns": returns}
               for i, returns in enumerate(("openecon.TableSet", "openecon.MIDeltaResult", "str", ""))]
    before = deepcopy(logical)
    encoded = generator.encode_catalog(logical)
    assert encoded[0]["T"] == 0
    assert [entry.get("R") for entry in encoded] == [None, "MIDeltaResult", None, None]
    assert encoded[2]["r"] == "str" and encoded[3]["r"] == ""
    assert generator.decode_catalog(encoded) == logical
    assert generator.decode_catalog(logical) == logical
    assert logical == before
    incoming = deepcopy(encoded)
    for entry in incoming:
        if "R" in entry:
            entry["r"] = "~" + entry.pop("R")
    assert generator.decode_catalog(incoming) == decoded_catalog(incoming) == logical
    mixed = deepcopy(encoded)
    mixed[0]["returns"] = "legacy.Result"
    mixed[1]["returns"] = ""
    mixed_before = deepcopy(mixed)
    assert [entry["returns"] for entry in generator.decode_catalog(mixed)][:2] == ["legacy.Result", ""]
    assert mixed == mixed_before
    literal = deepcopy(logical)
    literal[0]["returns"] = "~Result"
    literal_stored = generator.encode_catalog(literal)
    assert literal_stored[0]["returns"] == "~Result"
    assert "r" not in literal_stored[0] and "R" not in literal_stored[0]
    assert generator.decode_catalog(literal_stored) == decoded_catalog(literal_stored) == literal


def test_catalog_decoder_preserves_explicit_legacy_precedence_over_aliases():
    logical = [{"name": "openecon.Example.method", "kind": "method",
                "owner": "openecon.Example", "signature": "method(value)",
                "description": "Explicit entry description.", "returns": "openecon.Result",
                "parameters": [{"name": "value", "kind": "keyword-only", "default": "None",
                                "description": "Explicit parameter description.",
                                "choices": ["'first'", "'second'"]}]}]
    mixed = deepcopy(logical)
    mixed[0].update(n="ignored.name", s="ignored()", d="Ignored entry description.",
                    p=[], r="~IgnoredResult")
    mixed[0]["parameters"][0].update(n="ignored_parameter", v="'ignored'",
                                    d="Ignored parameter description.", c=["'ignored'"], k="p", t="*")
    original = deepcopy(mixed)
    assert generator.decode_catalog(mixed) == logical
    assert decoded_catalog(mixed) == logical
    assert mixed == original


@pytest.mark.parametrize("signature_key", ["s", "S"])
@pytest.mark.parametrize("kind_key", ["k", "t"])
@pytest.mark.parametrize("return_key", ["r", "R"])
def test_both_compact_codec_forms_preserve_complete_logical_metadata(
    signature_key, kind_key, return_key,
):
    stored = [{
        "n": "~Result.call", "k": "m",
        signature_key: "(frame, /, *, option=None, enabled=True, disabled=False)",
        "d": "Complete Türkçe help.",
        return_key: "~DataFrame" if return_key == "r" else "DataFrame",
        "p": [
            {"n": "frame", kind_key: "p", "a": "DataFrame", "d": "Declared frame."},
            {"n": "option", kind_key: "k", "v": None, "c": [], "d": ""},
            {"n": "enabled", kind_key: "k", "v": True},
            {"n": "disabled", kind_key: "k", "v": False},
        ],
    }]
    expected = [{
        "name": "openecon.Result.call", "kind": "method", "owner": "openecon.Result",
        "signature": "call(frame, /, *, option=None, enabled=True, disabled=False)",
        "description": "Complete Türkçe help.", "returns": "openecon.DataFrame",
        "parameters": [
            {"name": "frame", "kind": "positional-only", "annotation": "DataFrame", "description": "Declared frame."},
            {"name": "option", "kind": "keyword-only", "default": "None", "choices": [], "description": ""},
            {"name": "enabled", "kind": "keyword-only", "default": "True"},
            {"name": "disabled", "kind": "keyword-only", "default": "False"},
        ],
    }]
    before = deepcopy(stored)
    preservation_spec = importlib.util.spec_from_file_location(
        "editor_api_preservation", ROOT / "scripts/verify_editor_api_preserved.py",
    )
    preservation = importlib.util.module_from_spec(preservation_spec)
    preservation_spec.loader.exec_module(preservation)
    assert generator.decode_catalog(stored) == decoded_catalog(stored) == expected
    assert preservation.logical(json.dumps(stored)) == {expected[0]["name"]: expected[0]}
    assert stored == before
    mixed = deepcopy(stored)
    mixed[0].update(signature="", s="ignored()", S="(ignored)", returns="", r="~Ignored", R="Ignored",
                    kind="method", k="c")
    mixed[0]["p"][1].update(default="", v=True, kind="positional-only", k="k", t="*")
    mixed_expected = deepcopy(expected)
    mixed_expected[0].update(signature="", returns="")
    mixed_expected[0]["parameters"][1].update(default="", kind="positional-only")
    mixed_before = deepcopy(mixed)
    assert generator.decode_catalog(mixed) == decoded_catalog(mixed) == mixed_expected
    assert preservation.logical(json.dumps(mixed)) == {mixed_expected[0]["name"]: mixed_expected[0]}
    assert mixed == mixed_before


def test_numeric_and_quoted_default_codec_preserves_exact_complete_metadata():
    integers = ["0", "-1", "9007199254740991", "-9007199254740991"]
    quoted = ["''", "'cpu'", "'Türkçe'", "'two words'", "'None'"]
    unsafe = ["1.0", "-0", "00", "+1", "0x10", "9007199254740992",
              "-9007199254740992", '"cpu"', "'can't'", r"'can\'t'",
              r"'path\file'", "'line\nbreak'", "'carriage\rreturn'"]
    defaults = integers + quoted + ["None", "True", "False"] + unsafe
    logical = [{
        "name": "openecon.call", "kind": "function", "signature": "call(*, option=None)",
        "description": "Complete literal metadata. Türkçe.", "returns": "openecon.DataFrame",
        "parameters": [{"name": f"value{i}", "kind": "keyword-only", "default": value,
                        "annotation": "object", "description": "Exact declared spelling.",
                        "choices": ["'cpu'", "'Türkçe'"]} for i, value in enumerate(defaults)],
    }]
    before = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    parameters = stored[0]["p"]
    assert parameters[0]["D"] == 7 and "v" not in parameters[0]
    assert [p["v"] for p in parameters[1:len(integers)]] == [int(value) for value in integers[1:]]
    assert all(type(p["v"]) is int for p in parameters[1:len(integers)])
    start = len(integers)
    quoted_parameters = parameters[start:start + len(quoted)]
    assert quoted_parameters[1]["D"] == 2 and "q" not in quoted_parameters[1]
    assert [p["q"] for i, p in enumerate(quoted_parameters) if i != 1] == [value[1:-1] for i, value in enumerate(quoted) if i != 1]
    assert all("v" not in p for p in quoted_parameters)
    assert parameters[start + len(quoted)]["V"] == 0 and "v" not in parameters[start + len(quoted)]
    assert [p["B"] for p in parameters[start + len(quoted) + 1:start + len(quoted) + 3]] == [1, 0]
    assert [p["v"] for p in parameters[-len(unsafe):]] == unsafe
    assert all("q" not in p for p in parameters[-len(unsafe):])
    preservation_spec = importlib.util.spec_from_file_location(
        "editor_api_preservation", ROOT / "scripts/verify_editor_api_preserved.py",
    )
    preservation = importlib.util.module_from_spec(preservation_spec)
    preservation_spec.loader.exec_module(preservation)
    assert generator.decode_catalog(stored) == decoded_catalog(stored) == logical
    assert preservation.logical(json.dumps(stored)) == {logical[0]["name"]: logical[0]}
    assert logical == before
    mixed = deepcopy(stored)
    mixed[0]["p"][0].update(default="", v=17, q="ignored")
    mixed[0]["p"][1].update(v=0, q="ignored")
    mixed[0]["p"][2].update(v=False, q="ignored")
    mixed[0]["p"][3].update(v=None, q="ignored")
    mixed[0]["p"][4].update(v="", q="ignored")
    mixed[0]["p"][5].update(v="'legacy'", q="ignored")
    mixed_expected = deepcopy(logical)
    for parameter, value in zip(mixed_expected[0]["parameters"], ["", "0", "False", "None", "", "'legacy'"]):
        parameter["default"] = value
    mixed_before = deepcopy(mixed)
    assert generator.decode_catalog(mixed) == decoded_catalog(mixed) == mixed_expected
    assert preservation.logical(json.dumps(mixed)) == {mixed_expected[0]["name"]: mixed_expected[0]}
    assert mixed == mixed_before


def test_parameter_name_tokens_preserve_complete_metadata_and_explicit_names():
    names = [*COMPACT_PARAMETER_NAMES, "max_work_alias", "Türkçe"]
    logical = [{
        "name": "openecon.call", "kind": "function", "signature": "call(*, max_work=0)",
        "description": "Exact parameter names. Türkçe.", "returns": "openecon.DataFrame",
        "parameters": [{"name": name, "kind": "keyword-only", "default": str(index),
                        "annotation": "int", "description": "Complete parameter help.",
                        "choices": ["0", "1"]} for index, name in enumerate(names)],
    }]
    before = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    assert [parameter["n"] for parameter in stored[0]["p"]] == [*range(31), "max_work_alias", "Türkçe"]
    assert all(type(parameter["n"]) is int for parameter in stored[0]["p"][:31])
    preservation_spec = importlib.util.spec_from_file_location(
        "editor_api_preservation", ROOT / "scripts/verify_editor_api_preserved.py",
    )
    preservation = importlib.util.module_from_spec(preservation_spec)
    preservation_spec.loader.exec_module(preservation)
    stored_before = deepcopy(stored)
    assert generator.decode_catalog(stored) == decoded_catalog(stored) == logical
    assert preservation.logical(json.dumps(stored)) == {logical[0]["name"]: logical[0]}
    assert stored == stored_before and logical == before
    mixed = deepcopy(stored)
    mixed_expected = deepcopy(logical)
    for index, malformed in enumerate((True, None, {"invalid": "unused"}, 100, 1.5)):
        explicit_name = "" if index == 0 else f"legacy_{index}"
        mixed[0]["p"][index].update(name=explicit_name, n=malformed)
        mixed_expected[0]["parameters"][index]["name"] = explicit_name
    mixed_before = deepcopy(mixed)
    assert generator.decode_catalog(mixed) == decoded_catalog(mixed) == mixed_expected
    assert preservation.logical(json.dumps(mixed)) == {mixed_expected[0]["name"]: mixed_expected[0]}
    assert mixed == mixed_before


def test_data_name_token_appends_to_fixed_protocol_without_losing_any_catalog_field():
    # The original nine token meanings stay fixed. The new data token is a
    # general parameter-name abbreviation, not a method-specific exception.
    assert generator.STORED_PARAMETER_NAMES == (
        "max_work", "max_iterations", "tolerance", "covariance", "max_bytes",
        "intercept", "missing", "instruments", "endogenous", "data",
        "alpha", "weights", "result", "device", "time", "cluster", "categorical",
        "seed", "weight_type", "confidence", "components", "replications", "max_iter", "options", "columns", "level", "method", "batch_rows", "treatment", "alternative", "max_memory_mb",
    )
    raw = json.loads((ROOT / "web/src/editor-api.json").read_text(encoding="utf-8"))
    logical = decoded_catalog(raw)
    before = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    assert any(parameter.get("n") == 9 for entry in stored for parameter in entry["p"])
    assert generator.decode_catalog(stored) == decoded_catalog(stored) == logical
    for decode in compact_default_readers():
        # MI readers intentionally select their own family. Compare each
        # reader's complete result with the same reader of the older wire.
        assert decode(stored) == decode(raw)
    assert logical == before
    # Names that merely resemble data remain literal and every field is exact.
    fixture = [{"name": "openecon.mi_chained", "kind": "function", "signature": "mi_chained(data, /)",
                "description": "Complete Türkçe help.", "returns": "dict[str, Any]",
                "parameters": [{"name": name, "kind": "positional-only", "annotation": "Any",
                                "default": "'None'", "description": "Full parameter help.",
                                "choices": ["'None'", "None", "False"]}
                               for name in ("data", "Data", "data_alias", "Türkçe")]}]
    encoded = generator.encode_catalog(fixture)
    assert [parameter["n"] for parameter in encoded[0]["p"]] == [9, "Data", "data_alias", "Türkçe"]
    for decode in compact_default_readers():
        assert decode(encoded) == fixture


@pytest.mark.parametrize("token", [True, False, None, -1, 31, 1.5, float("nan"), float("inf"), -0.0, {}, []])
def test_active_parameter_name_token_refuses_invalid_aliases(token):
    stored = [{"n": "~call", "S": "()", "d": "Complete help.", "p": [{"n": token}]}]
    preservation_spec = importlib.util.spec_from_file_location(
        "editor_api_preservation", ROOT / "scripts/verify_editor_api_preserved.py",
    )
    preservation = importlib.util.module_from_spec(preservation_spec)
    preservation_spec.loader.exec_module(preservation)
    for decode in (generator.decode_catalog, decoded_catalog):
        with pytest.raises(ValueError):
            decode(stored)
    with pytest.raises(ValueError):
        preservation.logical(json.dumps(stored))


def test_integration_help_readers_preserve_kind_key_aliases_and_explicit_precedence(monkeypatch):
    # The integration scripts normally run with their own directory on sys.path.
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    logical = [{
        "name": "openecon.mi_chained", "kind": "function",
        "signature": "mi_chained(k, /, ordinary, *args, option=None, **kwargs)",
        "description": "Complete Türkçe help.", "returns": "openecon.MIResult",
        "parameters": [
            {"name": "k", "kind": "positional-only", "annotation": "DataFrame"},
            {"name": "ordinary", "kind": "positional-or-keyword", "default": ""},
            {"name": "args", "kind": "var-positional"},
            {"name": "option", "kind": "keyword-only", "default": "None", "choices": []},
            {"name": "kwargs", "kind": "var-keyword"},
        ],
    }]
    compact = generator.encode_catalog(logical)
    mixed = deepcopy(logical)
    mixed[0]["r"] = "~IgnoredResult"
    mixed[0]["R"] = "IgnoredResult"
    mixed[0]["S"] = "ignored malformed signature"
    mixed[0]["s"] = "ignored_alias()"
    signature_empty_expected = deepcopy(logical)
    signature_empty_expected[0]["signature"] = ""
    signature_empty = deepcopy(compact)
    signature_empty[0].update(signature="", s="ignored_alias()", S="ignored malformed signature")
    signature_legacy_expected = deepcopy(logical)
    signature_legacy_expected[0]["signature"] = "different_alias(value)"
    signature_legacy = deepcopy(compact)
    signature_legacy[0].update(s="different_alias(value)", S="ignored malformed signature")
    mixed[0]["parameters"][0]["k"] = "k"
    mixed[0]["parameters"][0]["t"] = "**"
    legacy_t = deepcopy(compact)
    for parameter in legacy_t[0]["p"]:
        if "k" in parameter:
            parameter["t"] = parameter.pop("k")
    legacy = deepcopy(compact)
    legacy[0]["r"] = "~" + legacy[0].pop("R")
    literal_expected = deepcopy(logical)
    literal_expected[0]["returns"] = "~MIResult"
    literal = deepcopy(literal_expected)
    literal[0].update(r="~IgnoredResult", R="IgnoredResult")
    empty_expected = deepcopy(logical)
    empty_expected[0]["returns"] = ""
    empty = deepcopy(empty_expected)
    empty[0].update(r="~IgnoredResult", R="IgnoredResult")
    for filename, reader in (
        ("verify_editor_api_preserved.py", "logical"),
        ("verify_missing_data_integrated.py", "help_entries"),
        ("verify_mi_extensions_integrated.py", "help_entries"),
    ):
        module_spec = importlib.util.spec_from_file_location(filename, ROOT / "scripts" / filename)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        for stored, expected in ((logical, logical), (compact, logical), (legacy, logical), (legacy_t, logical),
                                 (mixed, logical), (literal, literal_expected), (empty, empty_expected),
                                 (signature_empty, signature_empty_expected),
                                 (signature_legacy, signature_legacy_expected)):
            assert getattr(module, reader)(json.dumps(stored)) == {"openecon.mi_chained": expected[0]}
        with pytest.raises(ValueError, match="parenthesized suffix"):
            getattr(module, reader)(json.dumps([{**compact[0], "S": "invalid"}]))
    assert generator.decode_catalog(mixed) == logical
    assert mixed[0]["parameters"][0]["k"] == "k"


def test_return_namespace_codec_preserves_literal_legacy_types_and_mixed_precedence():
    common = {"signature": "call(frame, /, *, option='')", "description": "Complete Türkçe help.",
              "parameters": [{"name": "frame", "kind": "positional-only", "annotation": "DataFrame"},
                             {"name": "option", "kind": "keyword-only", "default": "",
                              "description": "", "choices": []}], "kind": "function"}
    returns = ["openecon.SurveyRegressionResult", "external.Result", "~DataFrame", "", "openecon.",
               "openecon.survey.Result | None", "str | openecon.TableSet"]
    logical = [{**deepcopy(common), "name": f"openecon.call{i}", "returns": value}
               for i, value in enumerate(returns)]
    logical.append({**deepcopy(common), "name": "openecon.no_return"})
    before = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    assert [entry.get("R") for entry in stored] == [
        "SurveyRegressionResult", None, None, None, "", "survey.Result | None", None, None,
    ]
    assert stored[1]["r"] == "external.Result" and stored[2]["returns"] == "~DataFrame"
    assert stored[3]["r"] == "" and stored[6]["r"] == "str | openecon.TableSet"
    assert all("r" not in stored[i] for i in (0, 4, 5))
    assert generator.decode_catalog(stored) == decoded_catalog(stored) == logical
    legacy = deepcopy(stored)
    for entry in legacy:
        if "R" in entry:
            entry["r"] = "openecon." + entry.pop("R")
    assert generator.decode_catalog(legacy) == decoded_catalog(legacy) == logical
    incoming = deepcopy(stored)
    for entry in incoming:
        if "R" in entry:
            entry["r"] = "~" + entry.pop("R")
    assert generator.decode_catalog(incoming) == decoded_catalog(incoming) == logical
    mixed = deepcopy(stored)
    mixed[0].update(returns="openecon.SurveyRegressionResult", r="external.Ignored", R="Ignored")
    mixed[1].update(R="Ignored")
    mixed[3].update(R="Ignored")
    mixed[4].update(returns="openecon.", r="external.Ignored")
    mixed[2].update(r="~Ignored", R="Ignored")
    mixed_before = deepcopy(mixed)
    assert generator.decode_catalog(mixed) == decoded_catalog(mixed) == logical
    assert mixed == mixed_before and logical == before
    assert "R" not in generator.decode_catalog(stored)[0]


def test_complete_catalog_return_namespace_compaction_is_lossless_and_saves_payload():
    logical = generator.generate("HEAD")
    stored = generator.encode_catalog(logical)
    # Isolate namespace savings from the newer exact return-type tokens.
    for entry in stored:
        if "T" in entry:
            entry["R"] = {0: "TableSet", 1: "ResultBundle", 2: "DataFrame"}[entry.pop("T")]
    prefix_count = sum("R" in entry for entry in stored)
    legacy = deepcopy(stored)
    for entry in legacy:
        if "R" in entry:
            entry["r"] = "openecon." + entry.pop("R")
    compact_bytes = json.dumps(stored, ensure_ascii=False, separators=(",", ":")).encode()
    legacy_bytes = json.dumps(legacy, ensure_ascii=False, separators=(",", ":")).encode()
    assert len(legacy_bytes)-len(compact_bytes) == 9*prefix_count
    assert len(legacy_bytes)-len(compact_bytes) >= 1_300
    assert generator.decode_catalog(json.loads(compact_bytes)) == logical
    assert decoded_catalog(json.loads(compact_bytes)) == logical
    assert generator.decode_catalog(legacy) == logical


def test_penalized_glm_help_describes_saved_prediction_and_literal_factors():
    catalog = saved_catalog()
    for name in ("elasticnet_logit", "elasticnet_poisson"):
        entry = catalog[f"openecon.{name}"]
        assert entry["returns"] == "openecon.ResultBundle"
        assert "prediction" in entry["description"]
        assert "inference" in entry["description"]
        params = {p["name"]: p for p in entry["parameters"]}
        assert params["device"]["default"] == "'cpu'"
        assert params["max_iterations"]["default"] == "200"
        assert {"penalty_factors", "forced_controls", "categorical", "weights", "weight_type", "lambda_path"} <= params.keys()
    for name in ("regularized_glm_predict", "regularized_glm_path"):
        assert catalog[f"openecon.{name}"]["returns"] == "openecon.DataFrame"


def test_catalog_encoding_preserves_all_logical_metadata_without_mutating_input():
    parameters = [
        {"name": "data", "kind": "positional-or-keyword", "annotation": "DataFrame"},
        {"name": "axis", "kind": "positional-only", "default": "0"},
        {"name": "options", "kind": "keyword-only", "default": "None",
         "description": "Explicit options.", "choices": ["'a'", "'b'"]},
        {"name": "args", "kind": "var-positional"},
        {"name": "kwargs", "kind": "var-keyword"},
    ]
    common = {"signature": "example(data, /, *args, options=None, **kwargs)",
              "description": "Preserve all metadata, including Türkçe.",
              "parameters": parameters, "returns": "openecon.DataFrame"}
    logical = [
        {**deepcopy(common), "name": "openecon.example", "kind": "function"},
        {**deepcopy(common), "name": "openecon.Example", "kind": "class"},
        {**deepcopy(common), "name": "openecon.plot.Example.example", "kind": "method",
         "owner": "openecon.plot.Example"},
    ]
    original = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    assert logical == original
    assert "kind" not in stored[0]
    assert stored[1]["k"] == "c"
    assert stored[2]["k"] == "m" and "owner" not in stored[2]
    assert [item["n"] for item in stored] == [
        "~example", "~Example", "~plot.Example.example",
    ]
    for item in stored:
        assert "signature" not in item and "description" not in item
        assert decoded_catalog([item])[0]["signature"] == common["signature"]
        assert item["d"] == common["description"]
        assert "parameters" not in item and "returns" not in item and "name" not in item
        assert item["p"][2]["d"] == parameters[2]["description"]
        assert "description" not in item["p"][2]
        assert item["p"][0]["n"] == 9 and item["p"][0]["a"] == "DataFrame"
        assert item["p"][2]["V"] == 0 and item["p"][2]["c"] == ["'a'", "'b'"]
        assert "name" not in item["p"][0] and "annotation" not in item["p"][0]
        assert "default" not in item["p"][2] and "choices" not in item["p"][2]
        assert item["p"][2]["n"] == 23
        assert item["p"][2]["V"] == 0
        assert item["p"][2]["c"] == ["'a'", "'b'"]
        assert item["p"][0]["a"] == "DataFrame"
        assert [p["n"] for p in item["p"]] == [9, "axis", 23, "args", "kwargs"]
        assert all("name" not in p for p in item["p"])
        assert item["p"][1]["D"] == 7 and "v" not in item["p"][1]
        assert decoded_catalog([item])[0]["parameters"][1]["default"] == parameters[1]["default"]
        assert decoded_catalog([item])[0]["parameters"][2]["default"] == parameters[2]["default"]
        assert item["p"][2]["c"] == parameters[2]["choices"]
        assert all("default" not in p and "choices" not in p for p in item["p"])
    assert "kind" not in stored[0]["p"][0] and "k" not in stored[0]["p"][0]
    assert all("kind" not in p for p in stored[0]["p"])
    assert [p["k"] for p in stored[0]["p"][1:]] == ["p", "k", "*", "**"]
    serialized = json.loads(json.dumps(stored, ensure_ascii=False))
    assert decoded_catalog(serialized) == original
    assert generator.decode_catalog(serialized) == original
    assert serialized == stored
    assert decoded_catalog(original) == original
    assert generator.decode_catalog(original) == original


def test_catalog_encoding_rejects_an_owner_that_cannot_be_derived_losslessly():
    with pytest.raises(ValueError, match="Method owner does not match"):
        generator.encode_catalog([{
            "name": "openecon.Example.example", "kind": "method",
            "owner": "openecon.Other", "parameters": [],
        }])


def test_parameter_compaction_keeps_legacy_records_and_falsey_metadata_readable():
    full = [{
        "name": "openecon.example", "kind": "function", "signature": "example(a, b='', *, c=None)",
        "description": "Complete help.", "returns": "openecon.DataFrame",
        "parameters": [
            {"name": "a", "kind": "positional-only", "annotation": "DataFrame"},
            {"name": "b", "kind": "positional-or-keyword", "default": "", "description": "", "choices": []},
            {"name": "c", "kind": "keyword-only", "default": "None", "choices": ["'Türkçe'", "'normal'"]},
        ],
    }]
    stored = generator.encode_catalog(full)
    assert stored[0]["p"][1] == {"n": "b", "d": "", "v": "", "c": []}
    assert generator.decode_catalog(stored) == decoded_catalog(stored) == full
    legacy = deepcopy(stored)
    for parameter in legacy[0]["p"]:
        for key, token in {"name": "n", "default": "v", "choices": "c", "kind": "k", "annotation": "a"}.items():
            if token in parameter:
                value = parameter.pop(token)
                parameter[key] = "None" if token == "v" and value is None else value
    assert generator.decode_catalog(legacy) == decoded_catalog(legacy) == full
    upstream = deepcopy(stored)
    for parameter in upstream[0]["p"]:
        if "k" in parameter:
            parameter["t"] = parameter.pop("k")
    assert generator.decode_catalog(upstream) == decoded_catalog(upstream) == full
    mixed = deepcopy(stored)
    mixed[0]["p"][1].update(name="b", default="", description="", choices=[])
    mixed[0]["p"][0].update(kind="positional-only", annotation="DataFrame", t="k", a="ignored")
    assert generator.decode_catalog(mixed) == decoded_catalog(mixed) == full


def test_parameter_kind_key_alias_restores_every_kind_and_legacy_precedence():
    stored = [{
        "n": "~example", "s": "example()", "d": "Complete help.",
        "p": [
            {"n": "positional", "t": "p"},
            {"n": "keyword", "t": "k"},
            {"n": "args", "t": "*"},
            {"n": "kwargs", "t": "**"},
            {"n": "default"},
            {"n": "legacy", "kind": "positional-only", "t": "k"},
        ],
    }]
    before = deepcopy(stored)
    expected = ["positional-only", "keyword-only", "var-positional",
                "var-keyword", "positional-or-keyword", "positional-only"]
    for decode in (decoded_catalog, generator.decode_catalog):
        parameters = decode(stored)[0]["parameters"]
        assert [p["kind"] for p in parameters] == expected
        assert all("t" not in p for p in parameters)
    assert stored == before


def test_complete_committed_catalog_parameter_compaction_stays_within_original_budget():
    logical = generator.generate("HEAD")
    stored = generator.encode_catalog(logical)
    serialized = "[\n" + ",\n".join(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) for entry in stored) + "\n]\n"
    assert len(serialized.encode("utf-8")) < 500_000
    assert generator.decode_catalog(json.loads(serialized)) == logical


def test_catalog_encoding_rejects_reserved_name_prefix():
    with pytest.raises(ValueError, match="reserved '~' marker"):
        generator.encode_catalog([{
            "name": "~example", "kind": "function", "parameters": [],
        }])


def test_legacy_parameter_type_alias_restores_all_kind_tokens_losslessly():
    legacy = [{
        "n": "~example", "s": "example(a, /, *args, b, **kwargs)",
        "d": "Legacy kind-key records remain readable.",
        "p": [{"n": "a", "t": "p"}, {"n": "args", "t": "*"},
              {"n": "b", "t": "k"}, {"n": "kwargs", "t": "**"}],
    }]
    expected = [{
        "name": "openecon.example", "signature": legacy[0]["s"],
        "description": legacy[0]["d"], "kind": "function",
        "parameters": [{"name": "a", "kind": "positional-only"},
                       {"name": "args", "kind": "var-positional"},
                       {"name": "b", "kind": "keyword-only"},
                       {"name": "kwargs", "kind": "var-keyword"}],
    }]
    before = deepcopy(legacy)
    assert generator.decode_catalog(legacy) == decoded_catalog(legacy) == expected
    assert legacy == before


def test_default_function_kind_encoding_preserves_complete_entry_metadata():
    stored = json.loads((ROOT / "web/src/editor-api.json").read_text())
    assert any("kind" not in entry for entry in stored)
    catalog = saved_catalog()
    assert catalog["openecon.catreg_nominal"]["kind"] == "function"
    assert catalog["openecon.Latex"]["kind"] == "class"
    assert catalog["openecon.DataFrame.head"]["kind"] == "method"
    expected = generator.generate("HEAD")
    assert list(catalog.values()) == expected


def test_categorical_option_catalog_retains_all_metadata_and_full_logical_equality():
    stored = json.loads((ROOT / "web/src/editor-api.json").read_text())
    assert any("d" in entry for entry in stored)
    catalog = saved_catalog()
    for name in ("catreg_nominal_response", "catreg_ordinal_response", "catreg_outcome_predict",
                 "catreg_bootstrap", "catpca_varimax", "catpca_promax", "catpca_rotated_predict",
                 "loglinear_multinomial", "loglinear_product_multinomial", "loglinear_sampling_compare", "loglinear_select"):
        entry = catalog[f"openecon.{name}"]
        assert entry["returns"] == "openecon.TableSet"
        assert len(entry["description"]) > 20
        assert "d" not in entry
        assert all("d" not in parameter for parameter in entry["parameters"])
        assert all("f" not in parameter for parameter in entry["parameters"])
    assert list(catalog.values()) == generator.generate("HEAD")


def test_conjoint_help_keeps_scored_alignment_and_conditional_inference():
    catalog = saved_catalog()
    names = ("plan", "orthogonal", "diagnostics", "fit", "predict", "holdout", "importance", "simulate")
    for name in names:
        entry = catalog[f"openecon.conjoint_{name}"]
        assert entry["returns"] == "openecon.TableSet"
        assert "max_work" in {p["name"] for p in entry["parameters"]}
    assert "iid" in catalog["openecon.conjoint_fit"]["description"]
    assert "without fitting" in catalog["openecon.conjoint_holdout"]["description"]
    assert "conditional" in catalog["openecon.conjoint_simulate"]["description"]


def test_multivariate_option_help_preserves_summary_and_scoring_limits():
    catalog = saved_catalog()
    for name in ("pca_matrix", "factor_matrix"):
        entry = catalog[f"openecon.{name}"]
        assert entry["returns"] == "openecon.TableSet"
        assert "actual" in entry["description"]
        parameters = {p["name"]: p for p in entry["parameters"]}
        assert parameters["matrix"]["default"] == "'correlation'"
    assert catalog["openecon.ca_project"]["returns"] == "openecon.DataFrame"
    assert "null axes" in catalog["openecon.ca_project"]["description"]
    pca = {p["name"]: p for p in catalog["openecon.pca"]["parameters"]}
    assert pca["weight_type"]["default"] == "'fweight'"
    factor = {p["name"]: p for p in catalog["openecon.factor"]["parameters"]}
    assert factor["geomin_epsilon"]["default"] == "0.01"


def test_advanced_unitroot_help_preserves_discrete_inference_contracts():
    catalog = saved_catalog()
    for name in ("ngperron", "kss"):
        entry = catalog[f"openecon.{name}"]
        assert entry["returns"] == "openecon.DataFrame"
        assert "p_value=None" in entry["description"]
        params = {p["name"]: p for p in entry["parameters"]}
        assert params["trend"]["default"] == "'constant'"
        assert params["max_work"]["default"] == "100000000"
    assert {p["name"]: p for p in catalog["openecon.ngperron"]["parameters"]}["lags"]["default"] == "None"


def test_multivariate_extension_help_preserves_inference_and_rotation_domains():
    catalog = saved_catalog()
    entry = catalog["openecon.factor_bootstrap"]
    assert entry["returns"] == "openecon.TableSet"
    assert "one-factor" in entry["description"] and "no p/df" in entry["description"]
    defaults = {p["name"]: p.get("default") for p in entry["parameters"]}
    assert defaults["replications"] == "199" and defaults["method"] == "'pf'"
    params = {p["name"]: p.get("default") for p in catalog["openecon.factor"]["parameters"]}
    assert params["weight_type"] == "'fweight'"
    assert params["cf_kappa"] == "0.0" and params["target_oblique"] == "False"
    assert params["rotation_tolerance"] == "1e-10"


def test_multivariate_weight_matrix_help_exposes_saved_state_and_assumptions():
    catalog = saved_catalog()
    for name in ("alpha", "factortest", "discrim", "canon"):
        params = {p["name"]: p for p in catalog[f"openecon.{name}"]["parameters"]}
        assert params["weights"]["default"] == "None"
        assert params["weight_type"]["default"] == "'fweight'"
    for name in ("discrim_summary", "canon_matrix", "rm_anova_wide", "rm_restore", "rm_contrasts"):
        assert catalog[f"openecon.{name}"]["returns"] == "openecon.TableSet"
    assert "Dataset" in catalog["openecon.canon_scores"]["returns"]
    assert "one-expanded-copy" in catalog["openecon.discrim"]["description"]
    assert "no fabricated training accuracy" in catalog["openecon.discrim_summary"]["description"]
    assert "Hotelling" in catalog["openecon.rm_contrasts"]["description"]


def test_selection_and_general_contrast_help_preserves_inference_boundaries():
    catalog = saved_catalog()
    for name in ("discrim_stepwise", "manova_oneway", "manova_summary", "manova_contrast", "rm_mtest"):
        assert catalog[f"openecon.{name}"]["returns"] == "openecon.TableSet"
    assert catalog["openecon.discrim_stepwise_predict"]["returns"] == "openecon.DataFrame"
    selection = catalog["openecon.discrim_stepwise"]
    assert "not post-selection inference" in selection["description"]
    defaults = {p["name"]: p.get("default") for p in selection["parameters"]}
    assert defaults["method"] == "'stepwise'"
    assert defaults["weights"] == "None" and defaults["weight_type"] == "'fweight'"
    for name in ("manova_contrast", "rm_mtest"):
        entry = catalog[f"openecon.{name}"]
        assert "L B M = C" in entry["description"]
        assert "upper-bound" in entry["description"]
        assert {p["name"] for p in entry["parameters"]} >= {"L", "M", "null", "alpha"}


def test_capability_window_and_diagnostic_help_preserves_named_result_types():
    catalog = saved_catalog()
    for name in ("asymcausality", "effective_f", "panel_structural_predict", "recursive_ols", "rolling_panel"):
        assert catalog[f"openecon.{name}"]["returns"] == "openecon.TableSet"
    assert "conditional" in catalog["openecon.asymcausality"]["description"].lower() or "partial-sum" in catalog["openecon.asymcausality"]["description"].lower()
    assert "ONE endogenous" in catalog["openecon.effective_f"]["description"]


def test_eight_closure_helpers_publish_bounded_saved_contracts():
    catalog=saved_catalog()
    for name in ("iv_saved_weak_test","iv_saved_ar_confidence_set","wild_cluster_test",
                 "wild_cluster_confidence_set","fisher_johansen","sar_iv_predict",
                 "panel_mmqr_bootstrap","panel_mmqr_resampled_predict","power_logrank",
                 "power_paired_mean","planning_scenarios","restore_summary"):
        assert catalog[f"openecon.{name}"]["returns"]=="openecon.TableSet"
    assert "finite" in catalog["openecon.wild_cluster_confidence_set"]["description"].lower()
    assert "Schoenfeld" in catalog["openecon.power_logrank"]["description"]
    assert catalog["openecon.summary_state"]["returns"]=="str"
    assert catalog["openecon.sar_iv"]["returns"]=="openecon.ResultBundle"


def test_meta_summary_helpers_have_source_signatures_and_explicit_defaults():
    catalog = saved_catalog()
    for name in ("meta_effectsize", "meta_pool", "meta_regress", "meta_predict", "meta_diagnostics", "meta_plot",
                 "meta_dependent", "meta_dependent_robust", "meta_dependent_contrast",
                 "meta_dependent_predict", "meta_dependent_predict_effect", "meta_dependent_diagnostics"):
        entry = catalog["openecon."+name]
        assert entry["description"]
        assert entry["signature"].startswith(name+"(")
    for name in ("meta_pool", "meta_regress", "meta_diagnostics"):
        options = {p["name"]: p for p in catalog["openecon."+name]["parameters"]}
        assert options["method"]["default"] == "'REML'"
        assert options["dependence"]["default"] == "'independent'"
    dependent = catalog["openecon.meta_dependent"]
    assert dependent["returns"] == "openecon.TableSet"
    required = {p["name"]: p for p in dependent["parameters"]}
    for name in ("data", "covariance", "study"):
        assert "default" not in required[name]
        assert required[name]["kind"] == "keyword-only"
    assert required["model"]["default"] == "'common'"
    assert required["method"]["default"] == "'REML'"
    assert "sampling error" in catalog["openecon.meta_dependent_predict_effect"]["description"]


def test_next_eight_helpers_preserve_distinct_returns_and_required_assumptions():
    catalog = saved_catalog()
    tables = ("mixed_satterthwaite", "trajectory_bands", "ols_cusum", "rm_contrast",
              "proxy_svar", "irt_fit_diagnostics", "irt_mh_dif", "irt_diagnostics_restore")
    frames = ("proxy_svar_irf", "local_derivatives", "local_average_derivatives")
    for name in tables:
        assert catalog[f"openecon.{name}"]["returns"] == "openecon.TableSet"
    for name in frames:
        assert catalog[f"openecon.{name}"]["returns"] == "openecon.DataFrame"
    trajectory = {p["name"]: p for p in catalog["openecon.trajectory_bands"]["parameters"]}
    assert "default" not in trajectory["assumptions"]
    assert trajectory["assumptions"]["kind"] == "keyword-only"
    proxy = {p["name"]: p for p in catalog["openecon.proxy_svar"]["parameters"]}
    assert "default" not in proxy["exogeneity_assumed"]
    assert "Satterthwaite" in catalog["openecon.mixed_satterthwaite"]["description"]
    assert "Gaussian" in catalog["openecon.ols_cusum"]["description"]
    assert "fixed" in catalog["openecon.local_derivatives"]["description"].lower()


def test_multiple_break_help_states_bounded_iid_inference():
    entry = saved_catalog()["openecon.bai_perron"]
    assert entry["returns"] == "openecon.TableSet"
    options = {p["name"]: p for p in entry["parameters"]}
    assert options["errors"]["default"] == "'iid_gaussian'"
    assert "fixed exogenous" in options["x"]["description"].lower()
    assert options["replications"]["default"] == "499"
    assert options["seed"]["default"] == "0"


def test_native_network_extensions_have_explicit_offline_contracts():
    catalog = saved_catalog()
    assert catalog["openecon.hypergraph"]["returns"] == "openecon.Hypergraph"
    assert catalog["openecon.Hypergraph.degree"]["returns"] == "openecon.DataFrame"
    for owner in ("Network", "MultiNetwork", "SignedNetwork"):
        for method in ("weighted_assignment", "general_matching"):
            assert catalog[f"openecon.{owner}.{method}"]["returns"] == "openecon.NetworkMatchingResult"
    for method in ("k_shortest_paths", "strong_bridges", "strong_articulation_points"):
        assert catalog[f"openecon.Network.{method}"]["returns"] == "openecon.DataFrame"
    parameters = {p["name"]: p for p in catalog["openecon.Network.qap_regression"]["parameters"]}
    assert parameters["method"]["default"] == "'freedman_lane'"
    assert parameters["adjustment"]["default"] == "'none'"
    assert "conditional" in parameters["joint"]["description"]


def test_dynamic_help_requires_explicit_selection_before_simple_projection():
    catalog = saved_catalog()
    for name in ('dynamic_network', 'read_dynamic_network'):
        assert catalog[f'openecon.{name}']['returns'] == 'openecon.DynamicNetwork'
    for name in ('at', 'window'):
        assert catalog[f'openecon.DynamicNetwork.{name}']['returns'] == 'openecon.MultiNetwork'
    window = {p['name']: p for p in catalog['openecon.DynamicNetwork.window']['parameters']}
    assert window['selection']['default'] == "'overlap'"
    assert window['attributes']['default'] == window['weight']['default'] == "'raise'"
    assert 'openecon.DynamicNetwork.pagerank' not in catalog


def test_hypergraph_and_bounded_path_help_exposes_only_supported_types():
    catalog = saved_catalog()
    assert catalog['openecon.hypergraph']['returns'] == 'openecon.Hypergraph'
    assert catalog['openecon.read_hypergraph']['returns'] == 'openecon.Hypergraph'
    for name in ['degree', 'strength', 'memberships', 'summary']:
        assert catalog[f'openecon.Hypergraph.{name}']['returns'] == 'openecon.DataFrame'
    for name in ['clique_projection', 'star_projection']:
        assert catalog[f'openecon.Hypergraph.{name}']['returns'] == 'openecon.Network'
    assert 'openecon.Hypergraph.pagerank' not in catalog
    for name in ['k_shortest_paths', 'strong_bridges', 'strong_articulation_points']:
        assert catalog[f'openecon.Network.{name}']['returns'] == 'openecon.DataFrame'
    options = {p['name']: p for p in catalog['openecon.Network.k_shortest_paths']['parameters']}
    assert options['max_path_length']['default'] == 'None'
    assert all(options[name]['kind'] == 'keyword-only' for name in ['k', 'max_path_length', 'max_output_nodes', 'max_frontier', 'max_work'])


def test_multigraph_help_requires_reducer_and_preserves_distinct_result_types():
    catalog = saved_catalog()
    assert catalog['openecon.multigraph']['returns'] == 'openecon.MultiNetwork'
    assert catalog['openecon.read_multigraph']['returns'] == 'openecon.MultiNetwork'
    assert catalog['openecon.MultiNetwork.filter']['returns'] == 'openecon.MultiNetwork'
    assert catalog['openecon.MultiNetwork.degree']['returns'] == 'openecon.DataFrame'
    projection = catalog['openecon.MultiNetwork.to_network']
    assert projection['returns'] == 'openecon.Network'
    reducer = next(p for p in projection['parameters'] if p['name'] == 'reducer')
    assert reducer['kind'] == 'keyword-only' and 'default' not in reducer
    assert 'openecon.MultiNetwork.pagerank' not in catalog


def test_catalog_regenerates_from_committed_source_without_importing_runtime():
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--revision", "HEAD", "--check"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "editor API entries from committed source" in result.stdout


def test_ols_signature_and_defaults_match_the_shipped_public_interface():
    source = subprocess.check_output(
        ["git", "show", "HEAD:src/openecon/analysis.py"], cwd=ROOT, text=True,
    )
    tree = ast.parse(source)
    ols = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "ols")
    entry = saved_catalog()["openecon.ols"]
    assert entry["signature"] == f"ols({ast.unparse(ols.args)})"
    assert [p["name"] for p in entry["parameters"]] == [arg.arg for arg in ols.args.kwonlyargs]
    defaults = {p["name"]: p.get("default") for p in entry["parameters"]}
    assert defaults["missing"] == "'drop'"
    assert defaults["intercept"] == "True"
    assert defaults["alpha"] == "0.05"
    assert entry["returns"] == "openecon.OLSResult"


def test_ols_only_postestimation_is_not_suggested_for_binary_model_results():
    catalog = saved_catalog()
    assert catalog["openecon.logit"]["returns"] == "openecon.ResultBundle"
    assert catalog["openecon.probit"]["returns"] == "openecon.ResultBundle"
    assert "openecon.OLSResult.predict" in catalog
    assert "openecon.OLSResult.vif" in catalog
    assert "openecon.ResultBundle.predict" not in catalog
    assert "openecon.ResultBundle.vif" not in catalog
    assert catalog["openecon.ResultBundle.summary"]["parameters"] == [
        {"name": "format", "kind": "positional-or-keyword", "default": "'text'", "description": "'text' or 'latex'."}
    ]


def test_table_and_chart_help_preserve_useful_actual_defaults():
    catalog = saved_catalog()
    assert catalog["openecon.DataFrame.head"]["parameters"][0]["default"] == "5"
    assert catalog["openecon.DataFrame.describe"]["returns"] == "openecon.DataFrame"
    assert catalog["openecon.DataFrame.to_latex"]["kind"] == "method"
    hist = catalog["openecon.plot.hist"]
    assert hist["parameters"][2] == {"name": "bins", "kind": "keyword-only", "default": "20"}
    assert hist["returns"] == "openecon.plot.PlotSpec"
    assert catalog["console.display"]["signature"] == "display(value)"
    assert catalog["console.display"]["returns"] == "None"


def test_prediction_dispatch_help_lists_source_derived_category_options_without_false_defaults():
    reader = generator.SourceReader("HEAD")
    catalog = saved_catalog()
    wrappers = reader.committed("src/openecon/analysis.py")
    delegated = reader.committed("src/openecon/econometrics/postest/prediction.py")
    for name in ("predict", "margins"):
        wrapper = generator.function_node(wrappers.body, name)
        item = catalog[f"openecon.{name}"]
        assert item["signature"] == f"{name}({ast.unparse(wrapper.args)})"
        options = {p["name"]: p for p in item["parameters"]}
        target = generator.function_node(delegated.body, name)
        assert {p["name"] for p in generator.parameter_list(target.args, {})} <= options.keys()
        assert options["outcome"]["default"] == "None"
        assert "Saved categorical" in options["outcome"]["description"]
        assert "default" not in options["kind"]
        assert "xb" in options["kind"]["description"] and "response" in options["kind"]["description"]


def test_documentation_is_bounded_unique_and_excludes_unpublished_models():
    catalog = saved_catalog()
    assert len(catalog) >= 280
    # Lossless entry/parameter metadata encoding retains every public signature
    # within the original decimal payload budget.
    assert (ROOT / "web/src/editor-api.json").stat().st_size < 500_000
    assert all(entry["description"] and len(entry["description"]) <= 500 for entry in catalog.values())
    assert "openecon.panel_reg" not in catalog
    for entry in catalog.values():
        assert len({p["name"] for p in entry["parameters"]}) == len(entry["parameters"])
        if entry["kind"] in {"class", "method"}:
                assert all(p["name"] not in {"self", "cls"} for p in entry["parameters"])


def test_wave5_test_procedures_and_heteroskedastic_scale_are_documented():
    catalog = saved_catalog()
    assert "PANIC" in catalog["openecon.xtpanic"]["description"]
    assert "Toda" in catalog["openecon.tycausality"]["description"]
    for name, expected in (("xtpanic", {"factors": "1", "inference": "'asymptotic'"}),
                           ("tycausality", {"dmax": "1", "joint": "True"})):
        params = {p["name"]: p for p in catalog[f"openecon.{name}"]["parameters"]}
        for param, default in expected.items():
            assert params[param]["default"] == default
    for name in ("predict", "margins"):
        kind = next(p for p in catalog[f"openecon.{name}"]["parameters"] if p["name"] == "kind")
        assert "sigma" in kind["description"] and "default" not in kind


def test_export_reader_refuses_calls_or_user_code():
    expression = ast.parse("{'function': __import__('os').system('false')}", mode="eval").body
    try:
        generator.export_literal(expression)
    except ValueError as error:
        assert "Unsupported export syntax" in str(error)
    else:
        raise AssertionError("Export metadata cannot execute source expressions")


def test_econometrics_catalog_covers_lazy_public_targets_and_exact_parameters():
    from openecon.econometrics import registry

    reader = generator.SourceReader("HEAD")
    exports = generator.registry_exports(reader)
    assert {name: target[:2] for name, target in exports.items()} == registry.public_exports()
    catalog = saved_catalog()
    for name, (module, original, _) in exports.items():
        node = generator.function_node(
            reader.committed("src/" + module.replace(".", "/") + ".py").body, original,
        )
        item = catalog[f"openecon.{name}"]
        assert item["signature"] == f"{name}({ast.unparse(node.args)})"
        assert [parameter["name"] for parameter in item["parameters"]] == [
            parameter["name"] for parameter in generator.parameter_list(node.args, {})
        ]
    for name in ("nardl", "tobit", "truncreg", "intreg", "xtreg", "meprobit"):
        assert catalog[f"openecon.{name}"]["returns"] == "openecon.ResultBundle"


def test_export_reader_allows_literal_templates_but_never_evaluates_expressions():
    node = ast.parse("f'{package}.test:helper'", mode="eval").body
    assert generator.export_literal(node, {"package": "openecon.econometrics.panel"}) == (
        "openecon.econometrics.panel.test:helper"
    )
    for expression in ("f'{__import__(\"os\").system(\"false\")}'", "f'{package!r}'", "f'{package:10}'"):
        try:
            generator.export_literal(ast.parse(expression, mode="eval").body, {"package": "trusted"})
        except ValueError:
            pass
        else:
            raise AssertionError("Export templates cannot execute or format expressions")


def test_registry_reader_refuses_a_factory_with_executable_statements():
    class UntrustedReader:
        def committed(self, path):
            if path.endswith("registry.py"):
                return ast.parse("FAMILIES = ('example',)")
            return ast.parse('''
def unsafe(name):
    __import__('os').system('false')
    return EstimatorInfo(function=name, entry='openecon.econometrics.example:fit')
ESTIMATORS = (unsafe('model'),)
EXPORTS = {}
''')

    try:
        generator.registry_exports(UntrustedReader())
    except ValueError as error:
        assert "single-return" in str(error)
    else:
        raise AssertionError("Manifest factories cannot execute project code")


@pytest.mark.parametrize("call", [
    "factory('wrong', name='model')", "factory('model', unexpected=12)",
    "factory('model', 'ignored')", "factory('model', **__import__('os').environ)",
    "factory(*['model'])", "factory()", "factory(name='a', name='b')",
])
def test_registry_reader_rejects_invalid_factory_call_shapes(call):
    class Reader:
        def committed(self, path):
            if path.endswith("registry.py"):
                return ast.parse("FAMILIES = ('example',)")
            return ast.parse(f'''
def factory(name):
    return EstimatorInfo(function=name, entry='openecon.econometrics.example:fit')
ESTIMATORS = ({call},)
EXPORTS = {{}}
''')

    with pytest.raises(ValueError, match="factory"):
        generator.registry_exports(Reader())


def test_registry_reader_obeys_positional_only_keyword_only_and_literal_defaults():
    class Reader:
        def committed(self, path):
            if path.endswith("registry.py"):
                return ast.parse("FAMILIES = ('example',)")
            return ast.parse('''
def factory(name, /, unused='default', *, module='openecon.econometrics.example'):
    return EstimatorInfo(function=name, entry=f'{module}:fit')
ESTIMATORS = (factory('model'),)
EXPORTS = {}
''')

    assert generator.registry_exports(Reader())["model"][:2] == (
        "openecon.econometrics.example", "model",
    )


@pytest.mark.parametrize("name", ["ols", "predict", "ModelSpec"])
def test_registry_help_cannot_overwrite_a_core_export(monkeypatch, name):
    monkeypatch.setattr(generator, "registry_exports", lambda reader: {
        name: ("openecon.econometrics.example", "unused", None),
    })
    with pytest.raises(ValueError, match="collide with the core API"):
        generator.generate("HEAD")


def test_parameter_choices_follow_committed_statistical_contracts():
    reader = generator.SourceReader("HEAD")
    choices = generator.parameter_choices(reader)
    catalog = saved_catalog()
    constants = {
        target.id: ast.literal_eval(node.value)
        for node in reader.committed("src/openecon/linear_ols/spec.py").body
        if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
        and target.id in {"COVARIANCES", "WEIGHTS", "KERNELS"}
    }
    assert [ast.literal_eval(value) for value in choices["openecon.ols"]["covariance"]] == list(constants["COVARIANCES"])
    assert [ast.literal_eval(value) for value in choices["openecon.ols"]["kernel"]] == list(constants["KERNELS"])
    assert choices["openecon.logit"]["covariance"] == ["'cluster'", "'nonrobust'"]
    assert choices["openecon.probit"]["covariance"] == choices["openecon.logit"]["covariance"]
    assert choices["openecon.ols"]["missing"] == ["'raise'", "'drop'"]
    for name, parameters in choices.items():
        for parameter in catalog[name]["parameters"]:
            if parameter["name"] in parameters:
                assert parameter["choices"] == parameters[parameter["name"]]
                assert all(isinstance(ast.literal_eval(value), str) for value in parameter["choices"])


def test_network_workflow_completion_preserves_chained_result_types():
    catalog = saved_catalog()
    assert catalog['openecon.Network.max_flow']['returns'] == 'openecon.NetworkFlowResult'
    assert catalog['openecon.Network.min_cut']['returns'] == 'openecon.NetworkFlowResult'
    assert catalog['openecon.NetworkFlowResult.summary']['returns'] == 'openecon.DataFrame'
    assert catalog['openecon.Network.bipartite_projection']['returns'] == 'openecon.Network'
    assert catalog['openecon.Network.minimum_spanning_forest']['returns'] == 'openecon.Network'
    assert catalog['openecon.Network.link_prediction']['returns'] == 'openecon.DataFrame'
    assert catalog['openecon.Network.shortest_path']['returns'] == 'openecon.DataFrame'


def test_network_inference_completion_exposes_result_tables_and_bounded_cut_summary():
    catalog = saved_catalog()
    assert catalog['openecon.Network.global_min_cut']['returns'] == 'openecon.NetworkCutResult'
    assert catalog['openecon.NetworkCutResult.summary']['returns'] == 'openecon.DataFrame'
    assert catalog['openecon.Network.triad_census']['returns'] == 'openecon.DataFrame'
    assert catalog['openecon.Network.qap_correlation']['returns'] == 'openecon.DataFrame'


def test_network_models_have_exact_signatures_and_bounded_chain_returns():
    catalog = saved_catalog()
    assert catalog['openecon.network_snapshots']['returns'] == 'openecon.NetworkSnapshots'
    assert catalog['openecon.Network.qap_regression']['returns'] == 'openecon.DataFrame'
    assert catalog['openecon.Network.block_model']['returns'] == 'openecon.NetworkBlockResult'
    for cls in ('NetworkSnapshots','NetworkBlockResult'):
        assert catalog[f'openecon.{cls}.summary']['returns'] == 'openecon.DataFrame'
        assert catalog[f'openecon.{cls}.to_latex']['kind'] == 'method'
    assert catalog['openecon.NetworkSnapshots.aggregate']['returns'] == 'openecon.Network'
    for method in ('snapshot_summary','transitions','edge_persistence','temporal_path'):
        assert catalog[f'openecon.NetworkSnapshots.{method}']['returns'] == 'openecon.DataFrame'
    params = {p['name']:p for p in catalog['openecon.Network.qap_regression']['parameters']}
    assert params['permutations']['default'] == '999'
    assert params['intercept']['default'] == 'True'
    assert 'Freedman' in catalog['openecon.Network.qap_regression']['description']


def test_count_models_and_selected_pair_predictions_chain_to_tables():
    catalog = saved_catalog()
    for method in ('poisson_block_model', 'degree_corrected_block_model'):
        entry = catalog[f'openecon.Network.{method}']
        assert entry['returns'] == 'openecon.NetworkBlockResult'
        assert {p['name'] for p in entry['parameters']} == {
            'groups', 'initial', 'seed', 'starts', 'max_iter', 'tol', 'max_work'}
        assert 'Poisson' in entry['description']
    predicted = catalog['openecon.NetworkBlockResult.expected_edges']
    assert predicted['returns'] == 'openecon.DataFrame'
    parameters = {p['name']:p for p in predicted['parameters']}
    assert parameters['max_pairs']['default'] == '100000'
    assert parameters['max_memory_mb']['default'] == '256'
    assert 'presence' in predicted['description']


def test_advanced_method_help_has_explicit_offline_parameters():
    catalog = saved_catalog()
    for name in ("fmols", "dols", "ccr", "panel_fmols", "panel_dols", "pmg", "mg", "dfe", "svar", "lp", "lpiv", "panel_lp", "mediation", "oaxaca"):
        item = catalog[f"openecon.{name}"]
        assert item["returns"] == "openecon.ResultBundle"
        params = {p["name"] for p in item["parameters"]}
        assert {"data", "y"} <= params
        assert "kwargs" not in params
    assert "instruments" in {p["name"] for p in catalog["openecon.lpiv"]["parameters"]}
    assert "pooling" in {p["name"] for p in catalog["openecon.panel_dols"]["parameters"]}


def test_finite_help_preserves_asymptotic_exact_and_descriptive_contracts():
    catalog = saved_catalog()
    names = ("firth_logit", "firth_predict", "firth_profile", "firth_test",
             "exact_logistic", "exact_poisson_rate", "lts", "lts_predict")
    for name in names:
        entry = catalog["openecon."+name]
        assert entry["returns"] == "openecon.TableSet"
        parameters = {p["name"]: p for p in entry["parameters"]}
        assert parameters["max_work"]["default"] == "100000000"
        assert parameters["device"]["default"] == "'cpu'"
    assert "Fisher" in catalog["openecon.firth_logit"]["description"]
    assert "inclusive" in catalog["openecon.exact_logistic"]["description"].lower()
    assert "descriptive" in catalog["openecon.lts_predict"]["description"].lower()
    assert catalog["openecon.finite_save"]["returns"] == "dict"
    assert catalog["openecon.finite_load"]["returns"] == "openecon.TableSet"

def test_smoothing_functionals_publish_target_and_inference_boundaries():
    catalog = saved_catalog()
    for name in ('spline_derivative', 'fp_derivative', 'gam_derivative', 'mars_derivative',
                 'loess_derivative', 'kernel_derivative', 'smoothing_margins', 'smoothing_contrast'):
        entry = catalog['openecon.' + name]
        assert entry['returns'] == 'openecon.DataFrame'
        params = {p['name']: p for p in entry['parameters']}
        assert params['missing']['default'] == "'raise'"
        assert params['max_work']['default'] == '2000000000'
    assert 'moving' in catalog['openecon.loess_derivative']['description']
    assert 'bias' in catalog['openecon.gam_derivative']['description']
    assert 'cross-profile' in catalog['openecon.smoothing_contrast']['description']


def test_empirical_survey_target_help_exposes_distinct_mean_and_ame_contracts():
    catalog = saved_catalog()
    entry = catalog["openecon.survey_margins_replicate"]
    assert entry["returns"] == "openecon.SurveyReplicateMarginsResult"
    assert "empirical distribution" in entry["description"]
    parameters = {p["name"]: p for p in entry["parameters"]}
    assert parameters["family"]["choices"] == ["'logit'", "'poisson'", "'probit'"]
    assert parameters["target"]["choices"] == ["'ame'", "'mean'"]
    assert "proper subset" in parameters["profiles"]["description"]
    assert catalog["openecon.SurveyReplicateMarginsResult.to_frame"]["returns"] == "openecon.DataFrame"
    assert catalog["openecon.SurveyReplicateMarginsResult.model_validate_json"]["owner"] == "openecon.SurveyReplicateMarginsResult"


def test_kind_aliases_keep_explicit_legacy_precedence():
    stored = [{"n": "~Example.run", "s": "run(*, flag=False)", "d": "Help.",
               "k": "m", "kind": "class", "p": [{"n": "flag", "k": "k",
               "kind": "positional-only", "v": ""}]}]
    original = deepcopy(stored)
    expected = [{"name": "openecon.Example.run", "signature": "run(*, flag=False)",
                 "description": "Help.", "kind": "class", "parameters": [
                     {"name": "flag", "kind": "positional-only", "default": ""}]}]
    assert generator.decode_catalog(stored) == expected
    assert decoded_catalog(stored) == expected
    assert stored == original


def test_signature_prefix_codec_preserves_aliases_unicode_and_legacy_precedence():
    logical = [
        {"name": "openecon.call", "signature": "call(x, /, *, label='Türkçe')", "kind": "function", "description": "Full help.", "parameters": []},
        {"name": "openecon.Result.method", "signature": "method(*args, **kwargs)", "kind": "method", "owner": "openecon.Result", "description": "Full help.", "parameters": []},
        {"name": "builtins.abs", "signature": "abs(x, /)", "kind": "function", "description": "Full help.", "parameters": []},
        {"name": "openecon.alias", "signature": "different_name(x)", "kind": "function", "description": "Full help.", "parameters": []},
    ]
    before = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    assert [item.get("S") for item in stored] == ["(x, /, *, label='Türkçe')", "(*args, **kwargs)", "(x, /)", None]
    assert stored[-1]["s"] == "different_name(x)"
    assert generator.decode_catalog(stored) == decoded_catalog(stored) == logical
    assert logical == before
    mixed = deepcopy(stored)
    mixed[0].update(signature="", s="ignored()")
    mixed[1]["s"] = "legacy_alias()"
    decoded = generator.decode_catalog(mixed)
    assert decoded[0]["signature"] == "" and decoded[1]["signature"] == "legacy_alias()"
    assert all("S" not in item for item in decoded)
    with pytest.raises(ValueError, match="parenthesized suffix"):
        generator.decode_catalog([{**stored[0], "S": "not a call"}])


def test_none_default_wire_marker_is_lossless_for_every_help_reader(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    logical = [{
        "name": "openecon.mi_chained", "kind": "function", "signature": "mi_chained(**options)",
        "description": "Complete help.", "parameters": [
            {"name": "required", "kind": "keyword-only"},
            {"name": "optional", "kind": "keyword-only", "default": "None"},
            {"name": "quoted", "kind": "keyword-only", "default": "'None'"},
            {"name": "empty", "kind": "keyword-only", "default": ""},
            {"name": "unknown", "kind": "keyword-only", "default": "MAX_N"},
            {"name": "null_literal", "kind": "keyword-only", "default": "null"},
        ],
    }]
    original = deepcopy(logical)
    compact = generator.encode_catalog(logical)
    assert "v" not in compact[0]["p"][0]
    assert compact[0]["p"][1]["V"] == 0
    assert compact[0]["p"][2]["q"] == "None" and "v" not in compact[0]["p"][2]
    assert [p["v"] for p in compact[0]["p"][3:]] == ["", "MAX_N", "null"]
    legacy = deepcopy(compact)
    legacy[0]["p"][1].pop("V")
    legacy[0]["p"][1]["v"] = "None"
    null_marker = deepcopy(legacy)
    null_marker[0]["p"][1]["v"] = None
    assert len(json.dumps(legacy)) - len(json.dumps(compact)) == 5
    mixed = deepcopy(compact)
    mixed[0]["p"][1]["default"] = ""
    explicit = deepcopy(logical)
    explicit[0]["parameters"][1]["default"] = ""
    for stored, expected in ((compact, logical), (legacy, logical), (null_marker, logical), (logical, logical), (mixed, explicit)):
        snapshot = deepcopy(stored)
        for decode in (generator.decode_catalog, decoded_catalog):
            actual = decode(json.loads(json.dumps(stored)))
            assert actual == expected
            assert "default" not in actual[0]["parameters"][0]
            assert all("v" not in p for p in actual[0]["parameters"])
        for filename, reader in (
            ("verify_editor_api_preserved.py", "logical"),
            ("verify_missing_data_integrated.py", "help_entries"),
            ("verify_mi_extensions_integrated.py", "help_entries"),
        ):
            reader_spec = importlib.util.spec_from_file_location(filename, ROOT / "scripts" / filename)
            module = importlib.util.module_from_spec(reader_spec)
            reader_spec.loader.exec_module(module)
            assert getattr(module, reader)(json.dumps(stored)) == {"openecon.mi_chained": expected[0]}
        assert stored == snapshot
    assert logical == original


def test_boolean_default_wire_markers_and_entry_kinds_are_lossless_for_every_help_reader(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    logical = [{
        "name": "openecon.mi_chained", "kind": "method", "owner": "openecon",
        "signature": "mi_chained(**options)", "description": "Complete help.", "parameters": [
            {"name": "required", "kind": "keyword-only"},
            *[{"name": name, "kind": "keyword-only", "default": default}
              for name, default in (("yes", "True"), ("no", "False"),
                                    ("quoted_yes", "'True'"), ("quoted_no", "'False'"),
                                    ("empty", ""), ("unknown", "MAX_N"), ("lowercase", "true"))],
        ],
    }]
    original = deepcopy(logical)
    compact = generator.encode_catalog(logical)
    assert compact[0]["k"] == "m"
    assert "v" not in compact[0]["p"][0]
    assert compact[0]["p"][1]["B"] == 1 and "v" not in compact[0]["p"][1]
    assert compact[0]["p"][2]["B"] == 0 and "v" not in compact[0]["p"][2]
    assert [p["q"] for p in compact[0]["p"][3:5]] == ["True", "False"]
    assert all("v" not in p for p in compact[0]["p"][3:5])
    assert [p["v"] for p in compact[0]["p"][5:]] == ["", "MAX_N", "true"]
    legacy = deepcopy(compact)
    legacy[0]["p"][1].pop("B")
    legacy[0]["p"][2].pop("B")
    legacy[0]["p"][1]["v"] = "True"
    legacy[0]["p"][2]["v"] = "False"
    boolean_marker = deepcopy(legacy)
    boolean_marker[0]["p"][1]["v"] = True
    boolean_marker[0]["p"][2]["v"] = False
    assert len(json.dumps(legacy)) - len(json.dumps(compact)) == 11
    mixed = deepcopy(compact)
    mixed[0]["p"][1]["default"] = ""
    mixed[0]["p"][2]["default"] = "MAX_N"
    explicit = deepcopy(logical)
    explicit[0]["parameters"][1]["default"] = ""
    explicit[0]["parameters"][2]["default"] = "MAX_N"
    mixed_kind = deepcopy(compact)
    mixed_kind[0]["kind"] = "function"
    explicit_kind = deepcopy(logical)
    explicit_kind[0]["kind"] = "function"
    explicit_kind[0].pop("owner")
    numeric = deepcopy(compact)
    numeric[0]["p"][1]["v"], numeric[0]["p"][2]["v"] = 1, 0
    numeric_expected = deepcopy(logical)
    numeric_expected[0]["parameters"][1]["default"] = "1"
    numeric_expected[0]["parameters"][2]["default"] = "0"
    for stored, expected in ((compact, logical), (legacy, logical), (boolean_marker, logical), (logical, logical),
                             (mixed, explicit), (mixed_kind, explicit_kind), (numeric, numeric_expected)):
        snapshot = deepcopy(stored)
        for decode in (generator.decode_catalog, decoded_catalog):
            actual = decode(stored)
            assert actual == expected
            assert "default" not in actual[0]["parameters"][0]
            assert "k" not in actual[0]
            assert all("v" not in p and "k" not in p for p in actual[0]["parameters"])
        for filename, reader in (
            ("verify_editor_api_preserved.py", "logical"),
            ("verify_missing_data_integrated.py", "help_entries"),
            ("verify_mi_extensions_integrated.py", "help_entries"),
        ):
            reader_spec = importlib.util.spec_from_file_location(filename, ROOT / "scripts" / filename)
            module = importlib.util.module_from_spec(reader_spec)
            reader_spec.loader.exec_module(module)
            assert getattr(module, reader)(json.dumps(stored)) == {"openecon.mi_chained": expected[0]}
        assert stored == snapshot
    assert logical == original


def compact_default_readers():
    readers = [generator.decode_catalog, decoded_catalog]
    for filename, reader in (
        ("verify_editor_api_preserved.py", "logical"),
        ("verify_missing_data_integrated.py", "help_entries"),
        ("verify_mi_extensions_integrated.py", "help_entries"),
    ):
        module_spec = importlib.util.spec_from_file_location(filename, ROOT / "scripts" / filename)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        function = getattr(module, reader)
        def decode_help(entries, function=function):
            # Standalone helpers import their shared reader from the scripts directory.
            with pytest.MonkeyPatch.context() as patch:
                patch.syspath_prepend(str(ROOT / "scripts"))
                return list(function(json.dumps(entries)).values())
        readers.append(decode_help)
    return readers


def test_none_default_tag_preserves_literals_required_parameters_and_legacy_precedence(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    defaults = ["None", "True", "False", "'None'", "", False, 0, None]
    logical = [{
        "name": "openecon.mi_chained", "kind": "function", "signature": "mi_chained(required, **options)",
        "description": "Complete Türkçe help.",
        "parameters": [{"name": "required", "kind": "positional-or-keyword"}] + [
            {"name": f"option_{i}", "kind": "keyword-only", "default": value}
            for i, value in enumerate(defaults)
        ],
    }]
    before = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    assert stored[0]["p"][1]["V"] == 0 and "v" not in stored[0]["p"][1]
    assert "V" not in stored[0]["p"][0] and "v" not in stored[0]["p"][0]
    assert all("V" not in p for p in stored[0]["p"][2:])
    for decode in compact_default_readers():
        assert decode(stored) == logical
        legacy = deepcopy(stored)
        legacy[0]["p"][1]["v"] = None
        legacy[0]["p"][1].pop("V")
        assert decode(legacy) == logical
        for fields, expected in (({"v": "", "V": "ignored"}, ""),
                                 ({"v": None, "V": "ignored", "q": "ignored"}, "None"),
                                 ({"v": False, "V": "ignored", "q": "ignored"}, "False"),
                                 ({"v": 0, "V": "ignored", "q": "ignored"}, "0"),
                                 ({"V": 0, "q": "ignored"}, "None"),
                                 ({"default": False, "v": "ignored", "V": "ignored"}, False),
                                 ({"default": "", "v": "ignored", "V": "ignored"}, "")):
            mixed = deepcopy(stored)
            mixed[0]["p"][1].update(fields)
            desired = deepcopy(logical)
            desired[0]["parameters"][1]["default"] = expected
            mixed_before = deepcopy(mixed)
            assert decode(mixed) == desired
            assert mixed == mixed_before
    assert logical == before
    with pytest.raises(ValueError, match="reserved 'V' marker"):
        generator.encode_catalog([{**logical[0], "parameters": [{"name": "x", "kind": "keyword-only", "V": 0}]}])


@pytest.mark.parametrize("tag", [None, True, False, 0.0, 1, "0", []])
def test_none_default_tag_rejects_invalid_required_tags_in_all_python_readers(tag, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    stored = [{"n": "~mi_chained", "S": "(x=None)", "d": "Complete help.",
               "p": [{"n": "x", "V": tag}]}]
    before = deepcopy(stored)
    for decode in compact_default_readers():
        with pytest.raises(ValueError, match="integer zero None tag"):
            decode(stored)
    assert stored == before


@pytest.mark.parametrize("fields,expected", [
    ({"N": 1}, "None"),
    ({"V": 0, "N": False}, "None"),
    ({"v": "", "V": "ignored", "N": "ignored"}, ""),
    ({"default": "False", "v": "0", "V": "ignored", "N": "ignored"}, "False"),
])
def test_legacy_none_marker_and_precedence_in_all_python_readers(fields, expected):
    stored = [{"n": "~mi_chained", "S": "(x=None)", "d": "Complete help.",
               "p": [{"n": "x", **fields}]}]
    before = deepcopy(stored)
    logical = [{"name": "openecon.mi_chained", "signature": "mi_chained(x=None)",
                "description": "Complete help.", "kind": "function",
                "parameters": [{"name": "x", "default": expected,
                                "kind": "positional-or-keyword"}]}]
    for decode in compact_default_readers():
        assert decode(stored) == logical
    assert stored == before


@pytest.mark.parametrize("tag", [None, True, False, 1.0, 0, "1", []])
def test_legacy_none_marker_rejects_invalid_required_tags_in_all_python_readers(tag):
    stored = [{"n": "~mi_chained", "S": "(x=None)", "d": "Complete help.",
               "p": [{"n": "x", "N": tag}]}]
    before = deepcopy(stored)
    for decode in compact_default_readers():
        with pytest.raises(ValueError, match="integer one None tag"):
            decode(stored)
    assert stored == before


def test_boolean_default_marker_compaction_is_exact_lossless_and_nonmutating():
    defaults = ["False", "True", "0", "", "'False'", "'True'", "False ",
                False, True, 0, 1, None, "None"]
    logical = [{"name": "openecon.mi_chained", "kind": "function",
                "signature": "mi_chained(**options)", "description": "Full Türkçe help.",
                "returns": "openecon.MIResult",
                "parameters": [{"name": f"p{i}", "kind": "keyword-only", "default": value}
                               for i, value in enumerate(defaults)]}]
    before = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    assert [parameter["B"] for parameter in stored[0]["p"][:2]] == [0, 1]
    assert all("v" not in parameter for parameter in stored[0]["p"][:2])
    for parameter, expected in zip(stored[0]["p"][2:7], defaults[2:7]):
        if expected == "0":
            assert parameter["D"] == 7 and "v" not in parameter
        elif expected in ("'False'", "'True'"):
            assert parameter["q"] == expected[1:-1] and "v" not in parameter
        else:
            assert parameter["v"] == expected and "B" not in parameter
    for parameter, expected in zip(stored[0]["p"][7:12], defaults[7:12]):
        assert parameter["default"] == expected and type(parameter["default"]) is type(expected)
        assert "v" not in parameter and "B" not in parameter
    assert stored[0]["p"][-1]["V"] == 0
    encoded_before = deepcopy(stored)
    for decode in compact_default_readers():
        assert decode(json.loads(json.dumps(stored))) == logical
    assert logical == before and stored == encoded_before
    with pytest.raises(ValueError, match="reserved 'B' marker"):
        generator.encode_catalog([{**logical[0], "parameters": [
            {"name": "x", "kind": "keyword-only", "B": 0}]}])


@pytest.mark.parametrize("fields,expected", [
    ({"B": 0}, "False"), ({"B": 1, "N": "ignored"}, "True"),
    ({"V": 0, "B": "ignored", "N": "ignored"}, "None"),
    ({"v": "", "V": "ignored", "B": "ignored", "N": "ignored"}, ""),
    ({"default": False, "v": "True", "B": "ignored"}, False),
    ({"v": True, "B": "ignored"}, "True"),
    ({"v": False, "B": "ignored"}, "False"),
    ({"v": None, "B": "ignored"}, "None"),
])
def test_boolean_default_marker_precedence_and_legacy_wire_in_all_python_readers(fields, expected):
    stored = [{"n": "~mi_chained", "S": "(x=False)", "R": "MIResult", "d": "Complete help.",
               "p": [{"n": "x", **fields}]}]
    before = deepcopy(stored)
    logical = [{"name": "openecon.mi_chained", "signature": "mi_chained(x=False)",
                "returns": "openecon.MIResult", "description": "Complete help.", "kind": "function",
                "parameters": [{"name": "x", "default": expected, "kind": "positional-or-keyword"}]}]
    for decode in compact_default_readers():
        assert decode(stored) == logical
    assert stored == before


@pytest.mark.parametrize("tag", [None, True, False, 0.0, 1.0, -1, 2, "0", []])
def test_boolean_default_marker_rejects_invalid_tags_in_all_python_readers(tag):
    stored = [{"n": "~mi_chained", "S": "(x=False)", "d": "Complete help.",
               "p": [{"n": "x", "B": tag, "N": 1}]}]
    before = deepcopy(stored)
    for decode in compact_default_readers():
        with pytest.raises(ValueError, match="integer zero or one Boolean tag"):
            decode(stored)
    assert stored == before


def test_fixed_default_marker_compaction_is_exact_lossless_and_nonmutating():
    literals = ["'raise'", "0.05", "'cpu'", "100000000", "50000000", "'drop'", "0.95", "0",
                "DEFAULT_BYTES", "DEFAULT_WORK", "'two-sided'", "'nonrobust'", "'constant'",
                "1000000", "2000000000", "100000"]
    extras = [0.05, 0, 100000000, False, True, None, "0.050", "'raise' ", "DEFAULT_WORK "]
    logical = [{"name": "openecon.mi_chained", "kind": "function",
                "signature": "mi_chained(**options)", "description": "Full Türkçe help.",
                "returns": "openecon.MIResult",
                "parameters": [{"name": f"p{i}", "kind": "keyword-only", "default": value,
                                "description": "Exact default.", "choices": []}
                               for i, value in enumerate(literals + extras)]}]
    before = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    assert [parameter["D"] for parameter in stored[0]["p"][:16]] == list(range(16))
    assert all("v" not in parameter for parameter in stored[0]["p"][:16])
    for parameter, expected in zip(stored[0]["p"][16:22], extras[:6]):
        assert parameter["default"] == expected and type(parameter["default"]) is type(expected)
        assert "D" not in parameter and "v" not in parameter
    assert [parameter["v"] for parameter in stored[0]["p"][22:]] == extras[6:]
    encoded_before = deepcopy(stored)
    for decode in compact_default_readers():
        assert decode(json.loads(json.dumps(stored))) == logical
    assert logical == before and stored == encoded_before
    with pytest.raises(ValueError, match="reserved 'D' marker"):
        generator.encode_catalog([{**logical[0], "parameters": [
            {"name": "x", "kind": "keyword-only", "D": 0}]}])


@pytest.mark.parametrize("fields,expected", [
    ({"D": 0}, "'raise'"), ({"D": 15, "q": "bad'quote", "N": "ignored"}, "100000"),
    ({"B": 0, "D": "ignored", "q": "bad'quote", "N": "ignored"}, "False"),
    ({"q": "Türkçe", "N": "ignored"}, "'Türkçe'"),
    ({"B": 0, "D": "ignored", "N": "ignored"}, "False"),
    ({"V": 0, "B": "ignored", "D": "ignored", "N": "ignored"}, "None"),
    ({"v": "", "D": "ignored"}, ""),
    ({"default": False, "v": "True", "D": "ignored"}, False),
    ({"v": False, "D": -1}, "False"), ({"v": None, "D": True}, "None"),
])
def test_fixed_default_marker_precedence_in_all_python_readers(fields, expected):
    stored = [{"n": "~mi_chained", "S": "(x=0)", "R": "MIResult", "d": "Complete help.",
               "p": [{"n": "x", **fields}]}]
    before = deepcopy(stored)
    logical = [{"name": "openecon.mi_chained", "signature": "mi_chained(x=0)",
                "returns": "openecon.MIResult", "description": "Complete help.", "kind": "function",
                "parameters": [{"name": "x", "default": expected, "kind": "positional-or-keyword"}]}]
    for decode in compact_default_readers():
        assert decode(stored) == logical
    assert stored == before


@pytest.mark.parametrize("tag", [None, True, False, 0.0, 1.0, -1, 16, "0", []])
def test_fixed_default_marker_rejects_invalid_tags_in_all_python_readers(tag):
    stored = [{"n": "~mi_chained", "S": "(x=0)", "d": "Complete help.",
               "p": [{"n": "x", "D": tag, "N": 1}]}]
    before = deepcopy(stored)
    for decode in compact_default_readers():
        with pytest.raises(ValueError, match="integer fixed-literal tag"):
            decode(stored)
    assert stored == before


@pytest.mark.parametrize("kind,name", [("class", "openecon.MIResult"),
                                      ("method", "openecon.MIResult.model_validate_json")])
def test_integration_readers_restore_entry_kinds_and_none_defaults(kind, name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    logical = [{"name": name, "kind": kind, "signature": name.rsplit(".", 1)[-1] + "(option=None)",
                "description": "Complete help.",
                "parameters": [{"name": "option", "kind": "keyword-only", "default": "None"}]}]
    if kind == "method":
        logical[0]["owner"] = "openecon.MIResult"
    stored = generator.encode_catalog(logical)
    before = deepcopy(stored)
    assert stored[0]["k"] == {"class": "c", "method": "m"}[kind]
    for decode in compact_default_readers():
        assert decode(stored) == logical
        mixed = deepcopy(stored)
        mixed[0]["kind"] = "function"
        desired = deepcopy(logical)
        desired[0]["kind"] = "function"
        desired[0].pop("owner", None)
        assert decode(mixed) == desired
    assert stored == before


@pytest.mark.parametrize("kind,name,signature", [
    ("function", "openecon.mi_chained", "mi_chained(frame, /, *, option=None)"),
    ("class", "openecon.MIResult", "MIResult(frame, /, *, option=None)"),
    ("method", "openecon.MIResult.model_validate_json", "model_validate_json(frame, /, *, option=None)"),
    ("function", "openecon.mi_chained", "different_alias(frame, /, *, option=None)"),
])
def test_exact_return_type_tokens_roundtrip_complete_metadata(kind, name, signature, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    names = ([name, "openecon.mi_pool", "openecon.mi_test"] if kind == "function" else
             [name, name.replace("MIResult", "MIPoolResult"), name.replace("MIResult", "MIJointResult")])
    logical = [{
        "name": current_name, "kind": kind, "signature": signature,
        "description": "Complete Türkçe help.", "returns": returns,
        "parameters": [{"name": "frame", "kind": "positional-only", "annotation": "DataFrame"},
                       {"name": "option", "kind": "keyword-only", "default": "None",
                        "description": "", "choices": ["'Türkçe'", "'None'"]}],
    } for current_name, returns in zip(names, ("openecon.TableSet", "openecon.ResultBundle", "openecon.DataFrame"))]
    if kind == "method":
        for item in logical:
            item["owner"] = item["name"].rsplit(".", 1)[0]
    before = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    assert [item["T"] for item in stored] == [0, 1, 2]
    assert all("returns" not in item and "R" not in item and "r" not in item for item in stored)
    legacy = deepcopy(stored)
    for item in legacy:
        item["R"] = {0: "TableSet", 1: "ResultBundle", 2: "DataFrame"}[item.pop("T")]
    incoming = deepcopy(legacy)
    for item in incoming:
        item["r"] = "~" + item.pop("R")
    wire_before = deepcopy(stored)
    for decode in compact_default_readers():
        for wire in (stored, legacy, incoming, logical):
            assert decode(json.loads(json.dumps(wire, ensure_ascii=False))) == logical
    assert logical == before and stored == wire_before


def test_return_type_tokens_match_only_exact_types_and_retain_explicit_precedence(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    types = ["TableSet", "openecon.tableset", "openecon.TableSet | None", "'openecon.TableSet'",
             "openecon.Özel", "~TableSet", "", "openecon.TableSet"]
    names = ["mi_chained", "mi_pool", "mi_test", "missing_patterns", "mvnorm_em", "little_mcar", "mi_monotone", "mi_mvn"]
    logical = [{"name": "openecon." + name, "kind": "function", "signature": name + "()",
                "description": "Complete help.", "parameters": [], "returns": value}
               for name, value in zip(names, types)]
    stored = generator.encode_catalog(logical)
    assert ["T" in item for item in stored] == [False] * 7 + [True]
    for decode in compact_default_readers():
        assert decode(stored) == logical
        for fields, expected in (({"returns": "", "r": "ignored", "R": "Ignored"}, ""),
                                 ({"returns": None, "r": "ignored", "R": "Ignored"}, None),
                                 ({"r": "", "R": "Ignored"}, ""),
                                 ({"r": "~ResultBundle", "R": "Ignored"}, "openecon.ResultBundle"),
                                 ({"R": "DataFrame"}, "openecon.DataFrame")):
            wire = [{"n": "~mi_chained", "S": "()", "d": "Complete help.", "p": [],
                     "T": "ignored", **fields}]
            before = deepcopy(wire)
            desired = {"name": "openecon.mi_chained", "kind": "function", "signature": "mi_chained()",
                       "description": "Complete help.", "parameters": [], "returns": expected}
            assert decode(wire) == [desired]
            assert wire == before
    with pytest.raises(ValueError, match="reserved 'T' marker"):
        generator.encode_catalog([{**logical[0], "T": 0}])


@pytest.mark.parametrize("token", [None, True, False, 0.0, 0.5, -1, 3, "0", []])
def test_return_type_tokens_reject_malformed_tags_in_all_python_readers(token, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    stored = [{"n": "~mi_chained", "S": "()", "d": "Complete help.", "p": [], "T": token}]
    before = deepcopy(stored)
    for decode in compact_default_readers():
        with pytest.raises(ValueError, match="recognized integer token"):
            decode(stored)
    assert stored == before


def test_return_type_tokens_preserve_full_catalog_and_save_actual_payload():
    logical = generator.generate("HEAD")
    stored = generator.encode_catalog(logical)
    legacy = deepcopy(stored)
    count = 0
    for item in legacy:
        if "T" in item:
            item["R"] = {0: "TableSet", 1: "ResultBundle", 2: "DataFrame"}[item.pop("T")]
            count += 1
    compact_bytes = json.dumps(stored, ensure_ascii=False, separators=(",", ":")).encode()
    legacy_bytes = json.dumps(legacy, ensure_ascii=False, separators=(",", ":")).encode()
    assert count >= 400
    assert len(legacy_bytes) - len(compact_bytes) > 866
    assert len(compact_bytes) + 1 < 500_000
    assert generator.decode_catalog(json.loads(compact_bytes)) == logical
    assert decoded_catalog(json.loads(compact_bytes)) == logical
    preservation = compact_default_readers()[2]
    assert preservation(stored) == logical
    # Every pre-existing saved logical record survives the new wire format.
    baseline = decoded_catalog(json.loads((ROOT / "web/src/editor-api.json").read_text()))
    assert generator.decode_catalog(generator.encode_catalog(baseline)) == baseline
    assert preservation(generator.encode_catalog(baseline)) == baseline


@pytest.mark.parametrize("kind,name", [("function", "openecon.mi_chained"),
                                       ("class", "openecon.MIResult"),
                                       ("method", "openecon.MIResult.model_validate_json")])
@pytest.mark.parametrize("token", range(len(COMPACT_SIGNATURE_SUFFIXES)))
def test_shared_signature_tokens_preserve_complete_metadata_in_all_readers(kind, name, token):
    suffix = COMPACT_SIGNATURE_SUFFIXES[token]
    logical = [{"name": name, "kind": kind, "signature": name.rsplit(".", 1)[-1] + suffix,
                "description": "Complete Türkçe saved-state help.", "returns": "openecon.Result | None",
                "parameters": [{"name": "payload", "kind": "positional-only", "annotation": "Any"},
                               {"name": "option", "kind": "keyword-only", "default": "None",
                                "description": "", "choices": ["'Türkçe'", "'None'"]}]}]
    if kind == "method":
        logical[0]["owner"] = name.rsplit(".", 1)[0]
    original = deepcopy(logical)
    stored = generator.encode_catalog(logical)
    assert stored[0]["I"] == token
    assert all(key not in stored[0] for key in ("S", "s", "signature"))
    snapshot = deepcopy(stored)
    for decode in compact_default_readers():
        assert decode(json.loads(json.dumps(stored, ensure_ascii=False))) == logical
    assert logical == original and stored == snapshot


@pytest.mark.parametrize("fields,expected", [
    ({"signature": "", "s": "ignored", "S": "(ignored)"}, ""),
    ({"s": "old_alias(value)", "S": "(ignored)"}, "old_alias(value)"),
    ({"s": "(value)", "S": "(ignored)"}, "restore(value)"),
    ({"S": "(value)"}, "restore(value)"),
])
def test_shared_signature_tokens_respect_explicit_and_legacy_precedence(fields, expected):
    stored = [{"n": "~MIResult.restore", "k": "m", "I": "ignored", "d": "Complete help.",
               "p": [], **fields}]
    snapshot = deepcopy(stored)
    for decode in compact_default_readers():
        result = decode(stored)[0]
        assert result["signature"] == expected
        assert "I" not in result and "S" not in result and "s" not in result
        assert result["owner"] == "openecon.MIResult"
    assert stored == snapshot


@pytest.mark.parametrize("token", [None, True, False, 0.0, .5, -1, len(COMPACT_SIGNATURE_SUFFIXES), "0", []])
def test_shared_signature_tokens_reject_malformed_tags_in_all_readers(token):
    stored = [{"n": "~Result.restore", "k": "m", "I": token, "d": "Complete help.", "p": []}]
    snapshot = deepcopy(stored)
    for decode in compact_default_readers():
        with pytest.raises(ValueError, match="recognized integer token"):
            decode(stored)
    assert stored == snapshot


def test_shared_signature_codec_requires_exact_suffix_and_preserves_all_saved_catalog_entries():
    suffix = COMPACT_SIGNATURE_SUFFIXES[0]
    logical = [{"name": "openecon.restore", "kind": "function", "signature": signature,
                "description": "Complete help.", "parameters": []}
               for signature in ("different_alias" + suffix, "restore" + suffix.replace("None=None", "None = None"))]
    encoded = generator.encode_catalog(logical)
    assert all("I" not in item for item in encoded)
    assert generator.decode_catalog(encoded) == logical
    with pytest.raises(ValueError, match="reserved 'I' marker"):
        generator.encode_catalog([{**logical[0], "I": 0}])
    baseline = decoded_catalog(json.loads((ROOT / "web/src/editor-api.json").read_text()))
    stored = generator.encode_catalog(baseline)
    literal = deepcopy(stored)
    count = 0
    for item in literal:
        if "I" in item:
            item["S"] = COMPACT_SIGNATURE_SUFFIXES[item.pop("I")]
            count += 1
    compact_bytes = json.dumps(stored, ensure_ascii=False, separators=(",", ":")).encode()
    literal_bytes = json.dumps(literal, ensure_ascii=False, separators=(",", ":")).encode()
    assert count >= 20 and len(literal_bytes) - len(compact_bytes) > 5_000
    for decode in compact_default_readers()[:3]:
        assert decode(stored) == baseline


@pytest.mark.parametrize("index", range(15, len(COMPACT_PARAMETER_LISTS)))
def test_appended_complete_parameter_templates_preserve_near_matches_in_all_readers(index):
    for field, value in (("annotation", "Any"), ("description", "Exact Turkish help: Türkçe."),
                         ("default", "None "), ("choices", ["None"]), ("kind", "positional-only")):
        parameters = deepcopy(COMPACT_PARAMETER_LISTS[index])
        parameters[0][field] = value
        logical = [{"name": "openecon.MIResult.call", "signature": "call(payload)",
                    "kind": "method", "owner": "openecon.MIResult", "description": "Complete help.",
                    "parameters": parameters, "returns": "dict[str, Any]"}]
        original = deepcopy(logical)
        stored = generator.encode_catalog(logical)
        assert "P" not in stored[0], "Near-template metadata must retain its explicit fields."
        for decode in compact_default_readers():
            assert decode(stored) == original
        assert logical == original


def test_preservation_reader_decodes_each_pinned_branch_token_layout(monkeypatch):
    module_spec = importlib.util.spec_from_file_location(
        "preserved_branch_layout", ROOT / "scripts" / "verify_editor_api_preserved.py",
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    pinned_source = b'''STORED_SIGNATURE_SUFFIXES = ("(incoming)",)
STORED_PARAMETER_LISTS = ([{"name": "incoming", "kind": "keyword-only", "default": "None"}],)
raise RuntimeError("Historical source must never execute")
'''
    calls = []
    def historical_source(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return pinned_source
    monkeypatch.setattr(module.subprocess, "check_output", historical_source)
    suffixes = module.signature_suffixes_at("incoming-pin")
    parameters = module.parameter_lists_at("incoming-pin")
    raw = json.dumps([{"n": "~Fixture.call", "I": 0, "P": 0,
                       "d": "Complete incoming help.", "k": "m"}])
    assert module.logical(raw, signature_suffixes=suffixes, parameter_lists=parameters) == {
        "openecon.Fixture.call": {
            "name": "openecon.Fixture.call", "signature": "call(incoming)",
            "parameters": [{"name": "incoming", "kind": "keyword-only", "default": "None"}],
            "description": "Complete incoming help.", "kind": "method", "owner": "openecon.Fixture",
        },
    }
    assert len(calls) == 2
    assert all(arguments == ["git", "show", "incoming-pin:scripts/generate_editor_api.py"]
               for arguments, _ in calls)
    assert module.logical(raw)["openecon.Fixture.call"]["signature"] != "call(incoming)"
