import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { EditorState } from "@codemirror/state";
import { python } from "@codemirror/lang-python";
import { ensureSyntaxTree, syntaxTree } from "@codemirror/language";
import { CompletionContext } from "@codemirror/autocomplete";
import rawEditorWire from "../src/editor-api.wire.ts";
import { decodeEditorWire } from "../src/editor-catalog-shapes.ts";
import { API_CATALOG, decodeApiEntry, type StoredApiEntry, type ApiParameter } from "../src/editor-api.ts";
import {
  getEditorSymbol,
  getActiveCall,
  editorCompletionSource,
} from "../src/editor-help.ts";

const storedCatalog = decodeEditorWire(rawEditorWire);

const COMPLETE_PARAMETER_RECORDS: ApiParameter[] = [
  { name: "alpha", kind: "keyword-only", default: "0.05" },
  { name: "missing", kind: "keyword-only", default: "'raise'" },
  { name: "data", kind: "positional-or-keyword" },
  { name: "data", kind: "keyword-only" },
  { name: "y", kind: "keyword-only" },
  { name: "weights", kind: "keyword-only", default: "None" },
  { name: "result", kind: "positional-or-keyword" },
  { name: "device", kind: "keyword-only", default: "'cpu'" },
  { name: "x", kind: "keyword-only" },
  { name: "intercept", kind: "keyword-only", default: "True" },
  { name: "categorical", kind: "keyword-only", default: "None" },
  { name: "cluster", kind: "keyword-only", default: "None" },
  { name: "max_work", kind: "keyword-only", default: "100000000" },
  { name: "y", kind: "positional-or-keyword" },
  { name: "covariance", kind: "keyword-only", default: "None" },
  { name: "max_work", kind: "keyword-only", default: "50000000" },
  { name: "options", kind: "var-keyword" },
  { name: "time", kind: "keyword-only", default: "None" },
  { name: "level", kind: "keyword-only", default: "0.95" },
  { name: "x", kind: "keyword-only", default: "None" },
];

const COMPACT_PARAMETER_NAMES = [
  "max_work", "max_iterations", "tolerance", "covariance", "max_bytes",
  "intercept", "missing", "instruments", "endogenous", "data",
  "alpha", "weights", "result", "device", "time", "cluster", "categorical",
  "seed", "weight_type", "confidence", "components", "replications", "max_iter", "options", "columns", "level", "method", "batch_rows", "treatment", "alternative", "max_memory_mb",
] as const;
const COMPACT_RETURN_TYPES = ["openecon.TableSet", "openecon.ResultBundle", "openecon.DataFrame"] as const;
const COMPACT_SIGNATURE_SUFFIXES = [
  "(json_data: str | bytes | bytearray, *, strict: bool | None=None, extra: ExtraValues | None=None, context: Any | None=None, by_alias: bool | None=None, by_name: bool | None=None)",
  "(*, indent: int | None=None, ensure_ascii: bool=False, include: IncEx | None=None, exclude: IncEx | None=None, context: Any | None=None, by_alias: bool | None=None, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, exclude_computed_fields: bool=False, round_trip: bool=False, warnings: bool | Literal['none', 'warn', 'error']=True, fallback: Callable[[Any], Any] | None=None, serialize_as_any: bool=False, polymorphic_serialization: bool | None=None)",
  "(obj: Any, *, strict: bool | None=None, extra: ExtraValues | None=None, from_attributes: bool | None=None, context: Any | None=None, by_alias: bool | None=None, by_name: bool | None=None)",
  "(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None=None, covariance: str='robust', cluster: str | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05, max_iterations: int=100, tolerance: float=1e-09, max_work: int=10000000000, device: str='cpu')",
  "(data, design, outcome, regressors, *, intercept=True, domain=None, missing='raise', alpha=0.05, null=0.0, max_iter=100, tolerance=1e-09)",
  "(data, design, outcome, regressors, *, method, intercept=True, domain=None, missing='raise', alpha=0.05, null=0.0, max_iter=100, tolerance=1e-09, replicates=None, replicate_weights=None, centering='original', rho=None, justification=None, scale=None, rscales=None, df=None)",
  "(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None=None, covariance: str='robust', cluster: str | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05, max_iterations: int=100, tolerance: float=1e-09, max_work: int=10000000000, device: str='cpu', batch_rows: int | None=None)",
  "(*, data: Any, y: str, x: Sequence[str], panel: str, time: str | None=None, model: str='re', covariance: str | None=None, cluster: str | None=None, intpoints: int | None=None, intmethod: str | None=None, corr: str | None=None, corr_order: int | None=None, force: bool=False, offset: str | None=None, categorical: Sequence[str] | None=None, missing: str='raise', alpha: float=0.05)",
  "(*, data: Any, y: str, x: Sequence[str], inflate: Sequence[str], inflate_link: str='logit', covariance: str | None=None, cluster: str | Sequence[str] | None=None, weights: str | None=None, weight_type: str | None=None, offset: str | None=None, exposure: str | None=None, categorical: Sequence[str] | None=None, intercept: bool=True, missing: str='raise', alpha: float=0.05)",
  "(low, indicator, *, low_periods, high_periods, low_frequency='Y', high_frequency='Q', aggregation='sum', as_of=None, low_releases=None, high_releases=None, device='cpu', weights=None)",
  "(*, data, y, x, categorical=None, weights=None, weight_type='aweight', intercept=True, missing='raise', l1_ratio=0.5, selection='cv', penalty=None, lambda_path=None, n_lambdas=20, lambda_ratio=0.001, folds=5, seed=1729, standardize=True, penalty_factors=None, forced_controls=None, max_iterations=200, tolerance=1e-08, max_work=2000000000, device='cpu')",
  "(data, design, outcomes, *, domain=None, missing='raise', alpha=0.05, null=0.0)",
  "(*, data, items: list[str], points: int=41, max_iter: int=200, max_eval: int=400, tolerance: float=1e-06, level: float=0.95)",
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
  "(**data: Any)",
  "(data, dimensions, count, *, levels, design, terms=None, structural=None, offset=None, max_iter=200, tol=1e-09, level=0.95, max_work=300000000, max_bytes=128 * 1024 ** 2)",
  "(lower, upper, *, times=None, level=0.95, device='cpu', weights=None, maxiter=500)",
  "(*, max_work=50000000)",
  "(*, data, y, x, panel, time, p=1, q=1, trend='c', max_iterations=500, missing='raise', alpha=0.05)",
  "(*, data, y, x, missing='raise', alpha=0.05, **options)",
  "(result: ResultBundle, steps: int, *, data: Any=None, exog: Any=None, alpha: float=0.05)",
  "(*, objective='weight', max_component_nodes=24, max_states=1000000, max_work=50000000)",
  "(partition=None, *, objective='weight', max_matrix_entries=1000000, max_work=50000000)",
  "(*, data, y, x, intercept=True, missing='raise', **options)",
  "(*, data, y, x, time=None, trend='c', kernel='bartlett', bandwidth=4, df_adjust=False, missing='raise', alpha=0.05)",
  "(data, *, kind='response', missing='raise', alpha=None)",
  "(*, data, y, x, missing='raise', **options)",
  "(data, outcome, predictors, *, key, spatial_weights, **options)",
  "(data, design, outcomes, *, domain=None, missing='raise', alpha=0.05, null=0.0, deff=False)",
  "(groups, *, initial=None, seed=0, starts=4, max_iter=100, tol=1e-09, max_work=50000000)",
  "(result, *, device='cpu', weights=None, max_work=DEFAULT_WORK, max_bytes=DEFAULT_BYTES)",
  "(time, event, *, causes=None, times=None, level=0.95, device='cpu', weights=None)",
  "(coefficients, *, null=0.0, alpha=None)",
  "(result, *args, **kwargs)",
  "(path, *, format=None, max_memory_mb=256, batch_rows=65536, max_file_mb=512)",
  "(*, nodes=None, edges=None, graph_attributes=None)",
  "(*, data, y, treatment, x, **options)",
  "(result: TableSet, data: Any)",
  "(data, columns, *, bounds, sampling_model, alpha=0.05, missing='raise')",
  "(coefficients, *, null=0.0)",
  "(path, *, overwrite=False)",
  "(data: Any, y: str, *, missing: str='drop')",
  "(*, direction='out', disconnected='infinite', max_work=50000000)",
  "(data, columns, *, sampling_model, alpha=0.05, missing='raise')",
  "(*, attributes=True)",
  "(*, data, x: str, y: str, title: str | None=None, **options)",
  "()",
  "(result, *, data, alignment=None, alpha=None, kind='mean')",
  "(source, target, *, max_work=50000000)",
  "(restrictions, *, null=None)",
  "(data, columns, *, max_iterations=500, tolerance=1e-08)",
  "(**options)",
  "(*, add=(), remove=(), weights=None, attributes=None)",
  "(max_nodes=1000, max_edges=5000, seed=0, groups=None)",
  "(buf=None, **kwargs)",
  "(data: Any, a: str, b: str, *, missing: str='drop')",
  "(result: ResultBundle, *, data, max_work=200000000)",
  "(data: pd.DataFrame, variables: list[str], *, knots: dict, components: int=2, n_starts: int=3, seed: int=0, maxiter: int=500, tol: float=1e-08, missing: str='drop', max_work: int=WORK, max_bytes: int=BYTES, device: str='cpu')",
] as const;

const COMPACT_PARAMETER_LISTS: NonNullable<StoredApiEntry["parameters"]>[] = [[{"name":"json_data","kind":"positional-or-keyword"},{"name":"strict","kind":"keyword-only","default":"None"},{"name":"extra","kind":"keyword-only","default":"None"},{"name":"context","kind":"keyword-only","default":"None"},{"name":"by_alias","kind":"keyword-only","default":"None"},{"name":"by_name","kind":"keyword-only","default":"None"}],[{"name":"indent","kind":"keyword-only","default":"None"},{"name":"ensure_ascii","kind":"keyword-only","default":"False"},{"name":"include","kind":"keyword-only","default":"None"},{"name":"exclude","kind":"keyword-only","default":"None"},{"name":"context","kind":"keyword-only","default":"None"},{"name":"by_alias","kind":"keyword-only","default":"None"},{"name":"exclude_unset","kind":"keyword-only","default":"False"},{"name":"exclude_defaults","kind":"keyword-only","default":"False"},{"name":"exclude_none","kind":"keyword-only","default":"False"},{"name":"exclude_computed_fields","kind":"keyword-only","default":"False"},{"name":"round_trip","kind":"keyword-only","default":"False"},{"name":"warnings","kind":"keyword-only","default":"True"},{"name":"fallback","kind":"keyword-only","default":"None"},{"name":"serialize_as_any","kind":"keyword-only","default":"False"},{"name":"polymorphic_serialization","kind":"keyword-only","default":"None"}],[{"name":"obj","kind":"positional-or-keyword"},{"name":"strict","kind":"keyword-only","default":"None"},{"name":"extra","kind":"keyword-only","default":"None"},{"name":"from_attributes","kind":"keyword-only","default":"None"},{"name":"context","kind":"keyword-only","default":"None"},{"name":"by_alias","kind":"keyword-only","default":"None"},{"name":"by_name","kind":"keyword-only","default":"None"}],[{"kind":"keyword-only","name":"data"},{"kind":"keyword-only","name":"y"},{"kind":"keyword-only","name":"endogenous"},{"kind":"keyword-only","name":"instruments"},{"default":"None","kind":"keyword-only","name":"x"},{"default":"'robust'","kind":"keyword-only","name":"covariance"},{"default":"None","kind":"keyword-only","name":"cluster"},{"default":"True","kind":"keyword-only","name":"intercept"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"default":"100","kind":"keyword-only","name":"max_iterations"},{"default":"1e-09","kind":"keyword-only","name":"tolerance"},{"default":"10000000000","kind":"keyword-only","name":"max_work"},{"default":"'cpu'","kind":"keyword-only","name":"device"}],[{"kind":"positional-or-keyword","name":"data"},{"kind":"positional-or-keyword","name":"design"},{"kind":"positional-or-keyword","name":"outcome"},{"kind":"positional-or-keyword","name":"regressors"},{"default":"True","kind":"keyword-only","name":"intercept"},{"default":"None","kind":"keyword-only","name":"domain"},{"choices":["'drop'","'raise'"],"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"default":"0.0","kind":"keyword-only","name":"null"},{"default":"100","kind":"keyword-only","name":"max_iter"},{"default":"1e-09","kind":"keyword-only","name":"tolerance"}],[{"kind":"positional-or-keyword","name":"data"},{"kind":"positional-or-keyword","name":"design"},{"kind":"positional-or-keyword","name":"outcome"},{"kind":"positional-or-keyword","name":"regressors"},{"kind":"keyword-only","name":"method"},{"default":"True","kind":"keyword-only","name":"intercept"},{"default":"None","kind":"keyword-only","name":"domain"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"default":"0.0","kind":"keyword-only","name":"null"},{"default":"100","kind":"keyword-only","name":"max_iter"},{"default":"1e-09","kind":"keyword-only","name":"tolerance"},{"default":"None","kind":"keyword-only","name":"replicates"},{"default":"None","kind":"keyword-only","name":"replicate_weights"},{"default":"'original'","kind":"keyword-only","name":"centering"},{"default":"None","kind":"keyword-only","name":"rho"},{"default":"None","kind":"keyword-only","name":"justification"},{"default":"None","kind":"keyword-only","name":"scale"},{"default":"None","kind":"keyword-only","name":"rscales"},{"default":"None","kind":"keyword-only","name":"df"}],[{"kind":"positional-or-keyword","name":"low"},{"kind":"positional-or-keyword","name":"indicator"},{"kind":"keyword-only","name":"low_periods"},{"kind":"keyword-only","name":"high_periods"},{"default":"'Y'","kind":"keyword-only","name":"low_frequency"},{"default":"'Q'","kind":"keyword-only","name":"high_frequency"},{"default":"'sum'","kind":"keyword-only","name":"aggregation"},{"default":"None","kind":"keyword-only","name":"as_of"},{"default":"None","kind":"keyword-only","name":"low_releases"},{"default":"None","kind":"keyword-only","name":"high_releases"},{"default":"'cpu'","kind":"keyword-only","name":"device"},{"default":"None","kind":"keyword-only","name":"weights"}],[{"kind":"positional-or-keyword","name":"data"},{"kind":"positional-or-keyword","name":"design"},{"kind":"positional-or-keyword","name":"outcome"},{"kind":"positional-or-keyword","name":"regressors"},{"default":"True","kind":"keyword-only","name":"intercept"},{"default":"None","kind":"keyword-only","name":"domain"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"default":"0.0","kind":"keyword-only","name":"null"},{"default":"100","kind":"keyword-only","name":"max_iter"},{"default":"1e-09","kind":"keyword-only","name":"tolerance"}],[{"kind":"keyword-only","name":"data"},{"kind":"keyword-only","name":"items"},{"default":"41","kind":"keyword-only","name":"points"},{"default":"200","kind":"keyword-only","name":"max_iter"},{"default":"400","kind":"keyword-only","name":"max_eval"},{"default":"1e-06","kind":"keyword-only","name":"tolerance"},{"default":"0.95","kind":"keyword-only","name":"level"}],[{"kind":"positional-or-keyword","name":"data"},{"kind":"positional-or-keyword","name":"design"},{"kind":"positional-or-keyword","name":"outcomes"},{"default":"None","kind":"keyword-only","name":"domain"},{"choices":["'drop'","'raise'"],"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"default":"0.0","kind":"keyword-only","name":"null"}],[{"kind":"keyword-only","name":"data"},{"kind":"keyword-only","name":"y"},{"kind":"keyword-only","name":"x"},{"kind":"keyword-only","name":"panel"},{"kind":"keyword-only","name":"time"},{"default":"1","kind":"keyword-only","name":"p"},{"default":"1","kind":"keyword-only","name":"q"},{"default":"'c'","kind":"keyword-only","name":"trend"},{"default":"500","kind":"keyword-only","name":"max_iterations"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"}],[{"kind":"positional-or-keyword","name":"lower"},{"kind":"positional-or-keyword","name":"upper"},{"default":"None","kind":"keyword-only","name":"times"},{"default":"0.95","kind":"keyword-only","name":"level"},{"default":"'cpu'","kind":"keyword-only","name":"device"},{"default":"None","kind":"keyword-only","name":"weights"},{"default":"500","kind":"keyword-only","name":"maxiter"}],[{"kind":"positional-or-keyword","name":"result"},{"kind":"positional-or-keyword","name":"data"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"DEFAULT_WORK","kind":"keyword-only","name":"max_work"},{"default":"DEFAULT_BYTES","kind":"keyword-only","name":"max_bytes"},{"default":"'cpu'","kind":"keyword-only","name":"device"}],[{"kind":"keyword-only","name":"data"},{"kind":"keyword-only","name":"y"},{"kind":"keyword-only","name":"x"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"kind":"var-keyword","name":"options"}],[{"kind":"positional-or-keyword","name":"result"},{"kind":"keyword-only","name":"data"},{"kind":"keyword-only","name":"variable"},{"default":"1","kind":"keyword-only","name":"order"},{"default":"False","kind":"keyword-only","name":"interval"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"2000000000","kind":"keyword-only","name":"max_work"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"'aweight'"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"l1_ratio","default":"0.5","kind":"keyword-only"},{"name":"selection","kind":"keyword-only","default":"'cv'"},{"name":"penalty","kind":"keyword-only","default":"None"},{"name":"lambda_path","kind":"keyword-only","default":"None"},{"name":"n_lambdas","default":"20","kind":"keyword-only"},{"name":"lambda_ratio","default":"0.001","kind":"keyword-only"},{"name":"folds","default":"5","kind":"keyword-only"},{"name":"seed","default":"1729","kind":"keyword-only"},{"name":"standardize","kind":"keyword-only","default":"True"},{"name":"penalty_factors","kind":"keyword-only","default":"None"},{"name":"forced_controls","kind":"keyword-only","default":"None"},{"name":"max_iterations","default":"200","kind":"keyword-only"},{"name":"tolerance","default":"1e-08","kind":"keyword-only"},{"name":"max_work","kind":"keyword-only","default":"2000000000"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"panel","kind":"keyword-only"},{"name":"time","kind":"keyword-only","default":"None"},{"name":"model","kind":"keyword-only","default":"'re'"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"intpoints","kind":"keyword-only","default":"None"},{"name":"intmethod","kind":"keyword-only","default":"None"},{"name":"corr","kind":"keyword-only","default":"None"},{"name":"corr_order","kind":"keyword-only","default":"None"},{"name":"force","kind":"keyword-only","default":"False"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"low","kind":"positional-or-keyword"},{"name":"indicator","kind":"positional-or-keyword"},{"name":"rho","kind":"keyword-only"},{"name":"low_periods","kind":"keyword-only"},{"name":"high_periods","kind":"keyword-only"},{"name":"low_frequency","kind":"keyword-only","default":"'Y'"},{"name":"high_frequency","kind":"keyword-only","default":"'Q'"},{"name":"aggregation","kind":"keyword-only","default":"'sum'"},{"name":"as_of","kind":"keyword-only","default":"None"},{"name":"low_releases","kind":"keyword-only","default":"None"},{"name":"high_releases","kind":"keyword-only","default":"None"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"design","kind":"positional-or-keyword"},{"name":"outcome","kind":"positional-or-keyword"},{"name":"categories","kind":"keyword-only","default":"None"},{"name":"domain","kind":"keyword-only","default":"None"},{"name":"missing","choices":["'drop'","'raise'"],"kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"null","default":"0.0","kind":"keyword-only"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"design","kind":"positional-or-keyword"},{"name":"numerators","kind":"positional-or-keyword"},{"name":"denominators","kind":"positional-or-keyword"},{"name":"domain","kind":"keyword-only","default":"None"},{"name":"missing","choices":["'drop'","'raise'"],"kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"null","default":"0.0","kind":"keyword-only"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"inflate","kind":"keyword-only"},{"name":"inflate_link","kind":"keyword-only","default":"'logit'"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"None"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"exposure","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"dimensions","kind":"positional-or-keyword"},{"name":"count","kind":"positional-or-keyword"},{"name":"levels","kind":"keyword-only"},{"name":"design","kind":"keyword-only"},{"name":"terms","kind":"keyword-only","default":"None"},{"name":"structural","kind":"keyword-only","default":"None"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"max_iter","default":"200","kind":"keyword-only"},{"name":"tol","default":"1e-09","kind":"keyword-only"},{"name":"level","kind":"keyword-only","default":"0.95"},{"name":"max_work","default":"300000000","kind":"keyword-only"},{"name":"max_bytes","default":"128 * 1024 ** 2","kind":"keyword-only"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only","default":"None"},{"name":"group","kind":"keyword-only"},{"name":"random","kind":"keyword-only","default":"None"},{"name":"intpoints","default":"7","kind":"keyword-only"},{"name":"intmethod","kind":"keyword-only","default":"'mvaghermite'"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"ll","kind":"keyword-only","default":"None"},{"name":"ul","kind":"keyword-only","default":"None"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"None"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"title","kind":"keyword-only","default":"None"},{"name":"unit","kind":"keyword-only","default":"''"},{"name":"palette","kind":"keyword-only","default":"None"},{"name":"aggregate","kind":"keyword-only","default":"None"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"raters","kind":"positional-or-keyword"},{"name":"categories","kind":"keyword-only"},{"name":"agreement_weights","kind":"keyword-only","default":"'unweighted'"},{"name":"inference","kind":"keyword-only","default":"'none'"},{"name":"level","kind":"keyword-only","default":"0.95"},{"name":"missing","kind":"keyword-only","default":"'drop'"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"max_fits","default":"1024","kind":"keyword-only"},{"name":"max_work","kind":"keyword-only","default":"100000000"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"data","kind":"keyword-only"},{"name":"variable","kind":"keyword-only"},{"name":"order","default":"1","kind":"keyword-only"},{"name":"interval","kind":"keyword-only","default":"True"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"max_work","kind":"keyword-only","default":"2000000000"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"time","kind":"keyword-only","default":"None"},{"name":"trend","kind":"keyword-only","default":"'c'"},{"name":"kernel","kind":"keyword-only","default":"'bartlett'"},{"name":"bandwidth","default":"4","kind":"keyword-only"},{"name":"df_adjust","kind":"keyword-only","default":"False"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"covariance","choices":["'cluster'","'nonrobust'"],"kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"missing","choices":["'raise'","'drop'"],"kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"None"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"kind","choices":["'linear'","'response'"],"kind":"keyword-only","default":"'response'"},{"name":"missing","choices":["'drop'","'raise'"],"kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"None"}],[{"name":"objective","kind":"keyword-only","default":"'weight'"},{"name":"max_component_nodes","default":"24","kind":"keyword-only"},{"name":"max_states","kind":"keyword-only","default":"1000000"},{"name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"treatment","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"var-keyword"}],[{"name":"partition","default":"None","kind":"positional-or-keyword"},{"name":"objective","kind":"keyword-only","default":"'weight'"},{"name":"max_matrix_entries","kind":"keyword-only","default":"1000000"},{"name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"design","kind":"positional-or-keyword"},{"name":"outcomes","kind":"positional-or-keyword"},{"name":"domain","kind":"keyword-only","default":"None"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"null","default":"0.0","kind":"keyword-only"},{"name":"deff","kind":"keyword-only","default":"False"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"outcome","kind":"positional-or-keyword"},{"name":"predictors","kind":"positional-or-keyword"},{"name":"key","kind":"keyword-only"},{"name":"spatial_weights","kind":"keyword-only"},{"name":"options","kind":"var-keyword"}],[{"name":"path","kind":"positional-or-keyword"},{"name":"format","kind":"keyword-only","default":"None"},{"name":"max_memory_mb","default":"256","kind":"keyword-only"},{"name":"batch_rows","default":"65536","kind":"keyword-only"},{"name":"max_file_mb","default":"512","kind":"keyword-only"}],[{"description":"A DataFrame, column dictionary, row records or batch Dataset source.","name":"data","kind":"keyword-only"},{"description":"Outcome column name; cannot be combined with formula.","name":"y","kind":"keyword-only","default":"None"},{"description":"Predictor column names; cannot be combined with formula.","name":"x","kind":"keyword-only","default":"None"},{"description":"Example: 'wage ~ education + experience + C(region)'.","name":"formula","kind":"keyword-only","default":"None"},{"description":"Standard error method, such as 'nonrobust', 'HC3', 'cluster' or 'hac'.","name":"covariance","choices":["'nonrobust'","'HC0'","'HC1'","'HC2'","'HC3'","'cluster'","'cluster_hc2'","'cluster_hc3'","'hac'","'bootstrap'","'jackknife'"],"kind":"keyword-only","default":"None"},{"description":"Predictor names to encode as categorical variables.","name":"categorical","kind":"keyword-only","default":"None"},{"description":"Include an intercept.","name":"intercept","kind":"keyword-only","default":"True"},{"description":"Column name or names for clustered standard errors.","name":"cluster","kind":"keyword-only","default":"None"},{"description":"Column name containing the weights.","name":"weights","kind":"keyword-only","default":"None"},{"description":"'aweight', 'fweight', 'pweight' or 'iweight'.","name":"weight_type","choices":["'aweight'","'fweight'","'pweight'","'iweight'"],"kind":"keyword-only","default":"None"},{"description":"'drop' removes observations with missing values; 'raise' reports an error.","name":"missing","choices":["'raise'","'drop'"],"kind":"keyword-only","default":"'drop'"},{"description":"Inference significance level; 0.05 gives a 95% confidence interval.","name":"alpha","kind":"keyword-only","default":"0.05"},{"description":"Time column used for HAC or lagged formulas.","name":"time","kind":"keyword-only","default":"None"},{"description":"HAC lag count or a supported automatic selection name.","name":"lags","kind":"keyword-only","default":"None"},{"description":"HAC weighting kernel.","name":"kernel","choices":["'bartlett'","'parzen'","'quadratic_spectral'","'truncated'"],"kind":"keyword-only","default":"None"},{"description":"Replication option for bootstrap or jackknife.","name":"reps","kind":"keyword-only","default":"None"},{"description":"Random seed for resampling.","name":"seed","kind":"keyword-only","default":"None"},{"name":"dfadjust","kind":"keyword-only","default":"False"},{"name":"hansen","kind":"keyword-only","default":"False"},{"description":"'auto', 'cpu', 'cuda', 'cuda:<index>' or 'mps'. CUDA uses float64; Metal preconditions bounded factors with checked CPU float64 refinement.","name":"device","kind":"keyword-only","default":"'auto'"}],[{"name":"coefficients","kind":"positional-or-keyword"},{"name":"null","default":"0.0","kind":"keyword-only"},{"name":"alpha","kind":"keyword-only","default":"None"}],[{"name":"time","kind":"positional-or-keyword"},{"name":"event","kind":"positional-or-keyword"},{"name":"causes","kind":"keyword-only","default":"None"},{"name":"times","kind":"keyword-only","default":"None"},{"name":"level","kind":"keyword-only","default":"0.95"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"weights","kind":"keyword-only","default":"None"}],[{"name":"nodes","kind":"keyword-only","default":"None"},{"name":"edges","kind":"keyword-only","default":"None"},{"name":"graph_attributes","kind":"keyword-only","default":"None"}],[{"name":"coefficients","kind":"positional-or-keyword"},{"name":"null","default":"0.0","kind":"keyword-only"}],[{"name":"add","default":"()","kind":"keyword-only"},{"name":"remove","default":"()","kind":"keyword-only"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"attributes","kind":"keyword-only","default":"None"}],[{"name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"columns","kind":"positional-or-keyword"},{"name":"bounds","kind":"keyword-only"},{"name":"sampling_model","kind":"keyword-only"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"missing","kind":"keyword-only","default":"'raise'"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"y","kind":"positional-or-keyword"},{"description":"``n_dropped``) or \"raise\".","name":"missing","kind":"keyword-only","default":"'drop'"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"data","kind":"keyword-only"},{"name":"alignment","kind":"keyword-only","default":"None"},{"name":"alpha","kind":"keyword-only","default":"None"},{"name":"kind","kind":"keyword-only","default":"'mean'"}],[{"name":"path","kind":"positional-or-keyword"},{"name":"overwrite","kind":"keyword-only","default":"False"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"args","kind":"var-positional"},{"name":"kwargs","kind":"var-keyword"}],[{"name":"buf","default":"None","kind":"positional-or-keyword"},{"name":"kwargs","kind":"var-keyword"}],[{"name":"direction","kind":"keyword-only","default":"'out'"},{"name":"disconnected","kind":"keyword-only","default":"'infinite'"},{"name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"max_work","kind":"keyword-only","default":"DEFAULT_WORK"},{"name":"max_bytes","kind":"keyword-only","default":"DEFAULT_BYTES"}],[{"name":"data","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"title","kind":"keyword-only","default":"None"},{"name":"options","kind":"var-keyword"}],[{"name":"attributes","kind":"keyword-only","default":"True"}],[{"name":"restrictions","kind":"positional-or-keyword"},{"name":"null","kind":"keyword-only","default":"None"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"columns","kind":"positional-or-keyword"},{"name":"sampling_model","kind":"keyword-only"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"missing","kind":"keyword-only","default":"'raise'"}],[{"name":"add","default":"()","kind":"keyword-only"},{"name":"remove","default":"()","kind":"keyword-only"},{"name":"rename","kind":"keyword-only","default":"None"}],[{"name":"max_nodes","default":"1000","kind":"positional-or-keyword"},{"name":"max_edges","default":"5000","kind":"positional-or-keyword"},{"name":"seed","default":"0","kind":"positional-or-keyword"},{"name":"groups","default":"None","kind":"positional-or-keyword"}],[{"name":"options","kind":"var-keyword"}],[{"name":"source","kind":"positional-or-keyword"},{"name":"target","kind":"positional-or-keyword"},{"name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"path","kind":"positional-or-keyword"},{"name":"format","kind":"keyword-only","default":"None"},{"name":"overwrite","kind":"keyword-only","default":"False"}],[{"name":"restricted","kind":"positional-or-keyword"},{"name":"full","kind":"positional-or-keyword"},{"name":"max_bytes","default":"128 * 1024 ** 2","kind":"keyword-only"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"columns","kind":"positional-or-keyword"},{"name":"max_iterations","default":"500","kind":"keyword-only"},{"name":"tolerance","default":"1e-08","kind":"keyword-only"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"path","default":"None","kind":"positional-or-keyword"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"coefficient","kind":"keyword-only","default":"None"},{"name":"max_work","kind":"keyword-only","default":"100000000"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"level","kind":"keyword-only","default":"None"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"null","default":"0.0","kind":"keyword-only"},{"name":"max_work","kind":"keyword-only","default":"100000000"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"series","kind":"positional-or-keyword"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"variables","kind":"positional-or-keyword"},{"name":"knots","kind":"keyword-only"},{"name":"components","default":"2","kind":"keyword-only"},{"name":"n_starts","default":"3","kind":"keyword-only"},{"name":"seed","kind":"keyword-only","default":"0"},{"name":"maxiter","default":"500","kind":"keyword-only"},{"name":"tol","default":"1e-08","kind":"keyword-only"},{"name":"missing","kind":"keyword-only","default":"'drop'"},{"name":"max_work","default":"WORK","kind":"keyword-only"},{"name":"max_bytes","default":"BYTES","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"outcome","kind":"positional-or-keyword"},{"name":"predictors","kind":"positional-or-keyword"},{"name":"knots","kind":"keyword-only"},{"name":"missing","kind":"keyword-only","default":"'drop'"},{"name":"max_bytes","default":"c.LIMIT_BYTES","kind":"keyword-only"},{"name":"max_work","default":"c.WORK","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"data","kind":"positional-or-keyword"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"max_work","default":"WORK","kind":"keyword-only"},{"name":"max_bytes","default":"BYTES","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"endogenous","kind":"keyword-only"},{"name":"instruments","kind":"keyword-only"},{"name":"x","kind":"keyword-only","default":"None"},{"name":"covariance","kind":"keyword-only","default":"'robust'"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"max_iterations","default":"100","kind":"keyword-only"},{"name":"tolerance","default":"1e-09","kind":"keyword-only"},{"name":"max_work","default":"10000000000","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"batch_rows","kind":"keyword-only","default":"None"}],[{"description":"Any registry fit except time-series models and fits that already use a resampling covariance.","name":"result","kind":"positional-or-keyword"},{"description":"Exactly the dataset ``result`` was fitted on (verified by hash).","name":"data","kind":"positional-or-keyword"},{"description":"Number of bootstrap replications R (at least 2; 200 by default, use 1000 or more for percentile/bc intervals).","name":"reps","default":"200","kind":"keyword-only"},{"description":"Seed (0 to 2**64 - 1) of the ``torch.Generator`` that draws every replicate; the same seed reproduces the result exactly. A random seed is drawn and recorded when omitted.","name":"seed","kind":"keyword-only","default":"None"},{"description":"Columns defining resampling clusters and strata.","name":"cluster","kind":"keyword-only","default":"None"},{"description":"Columns defining resampling clusters and strata.","name":"strata","kind":"keyword-only","default":"None"},{"description":"Units drawn per stratum.","name":"size","kind":"keyword-only","default":"None"},{"description":"Level of the intervals (default: the fit's alpha).","name":"alpha","kind":"keyword-only","default":"None"},{"description":"Interval type stored in ``extra['bootstrap_ci']``: normal based, percentile (Stata's ``_pctile`` definition) or bias-corrected percentile (z0 = Phi^-1(#{b_r <= b}/R)), BCa delete-unit acceleration or bootstrap-t using each refit's standard error. Advanced intervals currently require unweighted, unst","name":"ci","kind":"keyword-only","default":"'normal'"},{"description":"Dependent schemes validate explicit-column OLS. Blocks require a regular time column and block_length; wild_cluster requires cluster=.","name":"scheme","kind":"keyword-only","default":"'iid'"},{"name":"block_length","kind":"keyword-only","default":"None"},{"name":"time","kind":"keyword-only","default":"None"},{"description":"Coefficient restrictions for restricted wild cluster-t tests; requires the original covariance clustered on the same column and ci='normal'. Confidence-set inversion is not supplied.","name":"null","kind":"keyword-only","default":"None"},{"description":"Wild cluster multiplier distribution.","name":"wild","kind":"keyword-only","default":"'rademacher'"},{"description":"Advanced draw plus BCa acceleration budget (default 10,000).","name":"max_refits","default":"10000","kind":"keyword-only"}],[{"description":"String, path object (implementing os.PathLike[str]), or file-like object implementing a write() function. If None, the result is returned as a string. If a non-binary file object is passed, it should be opened with `newline=''`, disabling universal newlines. If a binary file object is passed, `mode`","name":"path_or_buf","default":"None","kind":"positional-or-keyword"},{"description":"String of length 1. Field delimiter for the output file.","name":"sep","default":"','","kind":"positional-or-keyword"},{"description":"Missing data representation.","name":"na_rep","default":"''","kind":"positional-or-keyword"},{"description":"Format string for floating point numbers. If a Callable is given, it takes precedence over other numeric formatting parameters, like decimal.","name":"float_format","default":"None","kind":"positional-or-keyword"},{"description":"Columns to write.","name":"columns","default":"None","kind":"positional-or-keyword"},{"description":"Write out the column names. If a list of strings is given it is assumed to be aliases for the column names.","name":"header","default":"True","kind":"positional-or-keyword"},{"description":"Write row names (index).","name":"index","default":"True","kind":"positional-or-keyword"},{"description":"Column label for index column(s) if desired. If None is given, and `header` and `index` are True, then the index names are used. A sequence should be given if the object uses MultiIndex. If False do not print fields for index names. Use index_label=False for easier importing in R.","name":"index_label","default":"None","kind":"positional-or-keyword"},{"description":"Forwarded to either `open(mode=)` or `fsspec.open(mode=)` to control the file opening. Typical values include:","name":"mode","default":"'w'","kind":"positional-or-keyword"},{"description":"A string representing the encoding to use in the output file, defaults to 'utf-8'. `encoding` is not supported if `path_or_buf` is a non-binary file object.","name":"encoding","default":"None","kind":"positional-or-keyword"},{"name":"compression","default":"'infer'","kind":"positional-or-keyword"},{"name":"quoting","default":"None","kind":"positional-or-keyword"},{"name":"quotechar","default":"'\"'","kind":"positional-or-keyword"},{"name":"lineterminator","default":"None","kind":"positional-or-keyword"},{"name":"chunksize","default":"None","kind":"positional-or-keyword"},{"name":"date_format","default":"None","kind":"positional-or-keyword"},{"name":"doublequote","default":"True","kind":"positional-or-keyword"},{"name":"escapechar","default":"None","kind":"positional-or-keyword"},{"name":"decimal","default":"'.'","kind":"positional-or-keyword"},{"name":"errors","default":"'strict'","kind":"positional-or-keyword"},{"name":"storage_options","default":"None","kind":"positional-or-keyword"}],[{"description":"File path or existing ExcelWriter.","name":"excel_writer","kind":"positional-or-keyword"},{"description":"Name of sheet which will contain DataFrame.","name":"sheet_name","default":"'Sheet1'","kind":"positional-or-keyword"},{"description":"Missing data representation.","name":"na_rep","default":"''","kind":"positional-or-keyword"},{"description":"Format string for floating point numbers. For example ``float_format=\"%.2f\"`` will format 0.1234 to 0.12.","name":"float_format","default":"None","kind":"positional-or-keyword"},{"description":"Columns to write.","name":"columns","default":"None","kind":"positional-or-keyword"},{"description":"Write out the column names. If a list of string is given it is assumed to be aliases for the column names.","name":"header","default":"True","kind":"positional-or-keyword"},{"description":"Write row names (index).","name":"index","default":"True","kind":"positional-or-keyword"},{"description":"Column label for index column(s) if desired. If not specified, and `header` and `index` are True, then the index names are used. A sequence should be given if the DataFrame uses MultiIndex.","name":"index_label","default":"None","kind":"positional-or-keyword"},{"description":"Upper left cell row to dump data frame.","name":"startrow","default":"0","kind":"positional-or-keyword"},{"description":"Upper left cell column to dump data frame.","name":"startcol","default":"0","kind":"positional-or-keyword"},{"description":"Write engine to use, 'openpyxl' or 'xlsxwriter'. You can also set this via the options ``io.excel.xlsx.writer`` or ``io.excel.xlsm.writer``.","name":"engine","default":"None","kind":"positional-or-keyword"},{"description":"Write MultiIndex and Hierarchical Rows as merged cells.","name":"merge_cells","default":"True","kind":"positional-or-keyword"},{"description":"Representation for infinity (there is no native representation for infinity in Excel).","name":"inf_rep","default":"'inf'","kind":"positional-or-keyword"},{"description":"Specifies the one-based bottommost row and rightmost column that is to be frozen.","name":"freeze_panes","default":"None","kind":"positional-or-keyword"},{"name":"storage_options","default":"None","kind":"positional-or-keyword"},{"name":"engine_kwargs","default":"None","kind":"positional-or-keyword"}],[{"description":"Mapping from unique predictor names to aligned Network snapshots; all dyads include absent edges as zero.","name":"predictors","kind":"positional-or-keyword"},{"description":"'weight' uses aggregate strengths; 'binary' uses edge presence.","name":"values","kind":"keyword-only","default":"'weight'"},{"description":"Include diagonal dyads; False excludes them.","name":"include_loops","kind":"keyword-only","default":"False"},{"description":"Uniform node permutations per coefficient-specific reduced-model residual test; plus-one p-values.","name":"permutations","default":"999","kind":"keyword-only"},{"description":"Private CPU Torch random seed; predictor names and node labels use canonical order.","name":"seed","kind":"keyword-only","default":"0"},{"description":"'two-sided' compares absolute partial correlations; 'greater' and 'less' use the signed statistic.","name":"alternative","kind":"keyword-only","default":"'two-sided'"},{"description":"Fit a constant; the constant receives no fabricated permutation test.","name":"intercept","kind":"keyword-only","default":"True"},{"description":"'freedman_lane' permutes reduced response residuals; 'dsp' permutes focal predictor residuals and projects nuisance effects again.","name":"method","kind":"keyword-only","default":"'freedman_lane'"},{"description":"Named coefficient subsets tested together, conditional on remaining predictors; upper-tail partial R-squared.","name":"joint","kind":"keyword-only","default":"None"},{"description":"'none', 'bonferroni', 'holm' or 'bh' over all reported slopes and joint hypotheses; the intercept is excluded.","name":"adjustment","kind":"keyword-only","default":"'none'"},{"description":"Explicit complete-test structural work budget; no partial permutation p-values.","name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"y","kind":"positional-or-keyword"},{"name":"factors","default":"None","kind":"positional-or-keyword"},{"name":"covariates","kind":"keyword-only","default":"None"},{"description":"``\"none\"`` (main effects only) or a list such as ``[[\"a\", \"b\"]]`` or ``[\"a#b\"]``; a covariate may appear in an interaction. An interaction needs the effects it contains in the model (``a#b#c`` needs ``a#b``, ``a#c`` and ``b#c``; ``a#b#x`` needs ``a#x`` and ``b#x``), otherwise ``invalid_spec``.","name":"interactions","kind":"keyword-only","default":"'full'"},{"name":"ss_type","default":"3","kind":"keyword-only"},{"description":"``oe.rm_anova`` with ``factors`` as between-subject factors (Type III, all between-subject interactions; covariates, ``emmeans`` and other values of ``ss_type`` or ``interactions`` are rejected).","name":"within","kind":"keyword-only","default":"None"},{"description":"``oe.rm_anova`` with ``factors`` as between-subject factors (Type III, all between-subject interactions; covariates, ``emmeans`` and other values of ``ss_type`` or ``interactions`` are rejected).","name":"subject","kind":"keyword-only","default":"None"},{"description":"reported: predictions averaged with equal weights over the levels of the other factors, with covariates at their sample means. Single factors also get pairwise comparisons.","name":"emmeans","kind":"keyword-only","default":"None"},{"description":"SPSS's default) for the pairwise comparisons and their intervals.","name":"adjust","kind":"keyword-only","default":"'bonferroni'"},{"description":"tests the ANCOVA assumption of equal slopes.","name":"homogeneity_of_slopes","kind":"keyword-only","default":"False"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"description":"``\"raise\"``.","name":"missing","kind":"keyword-only","default":"'drop'"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"psu","kind":"keyword-only"},{"name":"ssu_strata","kind":"keyword-only","description":"Typed second-stage stratum identifier nested within each sampled PSU; cells sample SSUs independently."},{"name":"ssu_frame","kind":"keyword-only","description":"Explicit sequence of role-keyed mappings for every positive second-stage cell in sampled PSUs, with psu, ssu_strata, population_ssu and optional first-stage strata keys. No zero-sample positive cell or unknown PSU; provenance cannot authenticate externally omitted cells."},{"name":"ssu","kind":"keyword-only"},{"name":"tsu_strata","kind":"keyword-only","description":"Typed terminal sampling-stratum identity within each observed SSU; cells independently select terminal TSUs."},{"name":"tsu_frame","kind":"keyword-only","description":"Complete positive terminal-cell frame for every observed SSU; role keys strata when declared, psu, ssu_strata, ssu, tsu_strata and population_tsu. Every positive cell must have sampled rows. Supplied inputs cannot authenticate a cell omitted externally."},{"name":"tsu","kind":"keyword-only"},{"name":"population_psu","kind":"keyword-only"},{"name":"population_ssu","kind":"keyword-only","description":"Exact SSU population count constant within each second-stage cell; not a pooled PSU count."},{"name":"population_tsu","kind":"keyword-only","description":"Exact TSU population count within each terminal sampling cell, not a pooled SSU count."},{"name":"strata","kind":"keyword-only","default":"None"},{"name":"max_rows","kind":"keyword-only","default":"1000000"},{"name":"max_memory_mb","kind":"keyword-only","default":"64","description":"Explicit complete geometry, cell frame and inference allocation budget."}],[{"name":"data","kind":"keyword-only"},{"name":"moments","kind":"keyword-only","description":"``{label: expression}``; labels default to 1, 2, ..."},{"name":"instruments","kind":"keyword-only","default":"None","description":"the constant is added unless ``instrument_constant=False``."},{"name":"instrument_constant","kind":"keyword-only","default":"True"},{"name":"start","kind":"keyword-only","default":"None","description":"``{b=value}`` in the expressions)."},{"name":"wmatrix","kind":"keyword-only","default":"None"},{"name":"winitial","kind":"keyword-only","default":"'identity'"},{"name":"twostep","kind":"keyword-only","default":"True"},{"name":"igmm","kind":"keyword-only","default":"False"},{"name":"center","kind":"keyword-only","default":"False"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"lags","kind":"keyword-only","default":"None","description":"expand as dummies. weights, weight_type : ``'aweight'``, ``'fweight'`` or ``'pweight'`` (pweights need a robust covariance)."},{"name":"kernel","kind":"keyword-only","default":"None","description":"expand as dummies. weights, weight_type : ``'aweight'``, ``'fweight'`` or ``'pweight'`` (pweights need a robust covariance)."},{"name":"panel","kind":"keyword-only","default":"None","description":"expand as dummies. weights, weight_type : ``'aweight'``, ``'fweight'`` or ``'pweight'`` (pweights need a robust covariance)."},{"name":"time","kind":"keyword-only","default":"None","description":"expand as dummies. weights, weight_type : ``'aweight'``, ``'fweight'`` or ``'pweight'`` (pweights need a robust covariance)."},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"None"},{"name":"tolerance","kind":"keyword-only","default":"1e-08"},{"name":"max_iterations","kind":"keyword-only","default":"500"},{"name":"igmm_tolerance","kind":"keyword-only","default":"1e-10"},{"name":"igmm_max_iterations","kind":"keyword-only","default":"1000"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only","description":"Complete resident numeric rows; at most 4096 observations and ten controls, with no silent missing-row deletion."},{"name":"y","kind":"keyword-only"},{"name":"treatment","kind":"keyword-only"},{"name":"mediator","kind":"keyword-only"},{"name":"controls","kind":"keyword-only","default":"None","description":"Distinct numeric controls in both equations; standardize over these fixed retained rows."},{"name":"mediator_link","kind":"keyword-only","default":"'logit'","description":"Binary mediator likelihood: logit or probit, exact numeric 0/1 with both levels.","choices":["'logit'","'probit'"]},{"name":"outcome_model","kind":"keyword-only","default":"'gaussian'","description":"Gaussian continuous outcome, logit/probit exact binary outcome, or Poisson integer counts with normalized likelihood.","choices":["'gaussian'","'logit'","'probit'","'poisson'"]},{"name":"interaction","kind":"keyword-only","default":"True","description":"Include the treatment-by-mediator term in the conditional outcome equation."},{"name":"covariance","kind":"keyword-only","default":"'HC0'","description":"Full joint OIM or uncorrected HC0; HC0 retains the empirical cross-equation score blocks.","choices":["'OIM'","'HC0'"]},{"name":"level","kind":"keyword-only","default":"0.95"},{"name":"interpretation","kind":"keyword-only","default":"'associational'","description":"Adjusted associational contrasts by default; causal labels require explicit identification assumptions.","choices":["'associational'","'causal'"]},{"name":"assumptions","kind":"keyword-only","default":"None"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"max_iterations","kind":"keyword-only","default":"200"},{"name":"tolerance","kind":"keyword-only","default":"1e-09"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"columns","kind":"positional-or-keyword"},{"name":"by","kind":"keyword-only","default":"None","description":"``Total`` row per variable. Rows with a missing group are excluded."},{"name":"stats","kind":"keyword-only","default":"None","description":"max``):"},{"name":"weights","kind":"keyword-only","default":"None","description":"``summarize``: ``\"aweight\"`` (default; analytic weights, rescaled to sum to the number of rows) or ``\"fweight\"`` (frequency weights: a row counts ``w`` times; this is how SPSS's WEIGHT command behaves). Rows with a zero or missing weight are excluded."},{"name":"weight_type","kind":"keyword-only","default":"None","description":"``summarize``: ``\"aweight\"`` (default; analytic weights, rescaled to sum to the number of rows) or ``\"fweight\"`` (frequency weights: a row counts ``w`` times; this is how SPSS's WEIGHT command behaves). Rows with a zero or missing weight are excluded."},{"name":"moments","kind":"keyword-only","default":"'stata'","description":"m2^2 with m_r = sum(w (x - mean)^r) / sum(w), so a normal variable has kurtosis 3; ``\"spss\"``: the bias-corrected G1 = g1 sqrt(n(n-1)) / (n-2) and excess G2 = (n-1) ((n+1) g2 - 3(n-1)) / ((n-2)(n-3))."},{"name":"listwise","kind":"keyword-only","default":"False","description":"values (Stata tabstat, SPSS MEANS); True keeps only rows complete in all ``columns`` (Stata's ``casewise``)."},{"name":"total","kind":"keyword-only","default":"True"},{"name":"anova","kind":"keyword-only","default":"False","description":"each variable on the groups and eta / eta squared (SPSS ``MEANS /STATISTICS ANOVA``)."}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"method","kind":"keyword-only","default":"'hwinters'"},{"name":"alpha","kind":"keyword-only","default":"None","description":"seasonal recursions. Any that the method uses and that is left ``None`` is estimated by minimizing the in-sample sum of squared one-step forecast errors; a parameter that the method does not use must stay ``None``. The criterion can have several local minima, so the search descends from the three be"},{"name":"beta","kind":"keyword-only","default":"None","description":"seasonal recursions. Any that the method uses and that is left ``None`` is estimated by minimizing the in-sample sum of squared one-step forecast errors; a parameter that the method does not use must stay ``None``. The criterion can have several local minima, so the search descends from the three be"},{"name":"gamma","kind":"keyword-only","default":"None","description":"seasonal recursions. Any that the method uses and that is left ``None`` is estimated by minimizing the in-sample sum of squared one-step forecast errors; a parameter that the method does not use must stay ``None``. The criterion can have several local minima, so the search descends from the three be"},{"name":"period","kind":"keyword-only","default":"None"},{"name":"additive","kind":"keyword-only","default":"True","description":"``tssmooth shwinters`` is multiplicative unless its ``additive`` option is given."},{"name":"time","kind":"keyword-only","default":"None"},{"name":"forecast","kind":"keyword-only","default":"0"}]];

const appendedParameterTemplates = JSON.parse(readFileSync(
  new URL("./fixtures/causal-assignment-parameter-templates.json", import.meta.url), "utf8",
)) as { token: number; parameters: NonNullable<StoredApiEntry["parameters"]> }[];
assert.deepEqual(appendedParameterTemplates.map(({ token }) => token), [15, 16, 17, 20, 21, 22, 23]);
for (const { token, parameters } of appendedParameterTemplates) {
  assert.deepEqual(COMPACT_PARAMETER_LISTS[token], parameters);
}

const thirdIntegrationTemplates = JSON.parse(readFileSync(
  new URL("./fixtures/integration3-parameter-templates.json", import.meta.url), "utf8",
)) as { token: number; parameters: NonNullable<StoredApiEntry["parameters"]> }[];
assert.deepEqual(thirdIntegrationTemplates.map(({ token }) => token), [40]);
for (const { token, parameters } of thirdIntegrationTemplates) {
  assert.deepEqual(COMPACT_PARAMETER_LISTS[token], parameters);
}

const fourthIntegrationTemplates = JSON.parse(readFileSync(
  new URL("./fixtures/integration4-parameter-templates.json", import.meta.url), "utf8",
)) as { token: number; parameters: NonNullable<StoredApiEntry["parameters"]> }[];
assert.deepEqual(fourthIntegrationTemplates.map(({ token }) => token), [74]);
for (const { token, parameters } of fourthIntegrationTemplates) {
  assert.deepEqual(COMPACT_PARAMETER_LISTS[token], parameters);
}

const fourStageSignatures = JSON.parse(readFileSync(
  new URL("./fixtures/four-stage-signature-suffixes.json", import.meta.url), "utf8",
)) as { token: number; suffix: string }[];
assert.deepEqual(fourStageSignatures.map(({token}) => token), Array.from({length: 60}, (_, i) => 7 + i));
for (const {token, suffix} of fourStageSignatures) assert.equal(COMPACT_SIGNATURE_SUFFIXES[token], suffix);
const fourStageParameters = JSON.parse(readFileSync(
  new URL("./fixtures/four-stage-parameter-templates.json", import.meta.url), "utf8",
)) as { token: number; parameters: NonNullable<StoredApiEntry["parameters"]> }[];
assert.deepEqual(fourStageParameters.map(({token}) => token), Array.from({length: 30}, (_, i) => 41 + i));
for (const {token, parameters} of fourStageParameters) assert.deepEqual(COMPACT_PARAMETER_LISTS[token], parameters);

const fifthIntegrationTemplates = JSON.parse(readFileSync(
  new URL("./fixtures/integration5-parameter-templates.json", import.meta.url), "utf8",
)) as { token: number; parameters: NonNullable<StoredApiEntry["parameters"]> }[];
assert.deepEqual(fifthIntegrationTemplates.map(({ token }) => token), [71, 72, 73]);
for (const { token, parameters } of fifthIntegrationTemplates) {
  assert.deepEqual(COMPACT_PARAMETER_LISTS[token], parameters);
}

function state(code: string, readOnly = false) {
  const result = EditorState.create({
    doc: code,
    selection: { anchor: code.length },
    extensions: [python(), EditorState.readOnly.of(readOnly)],
  });
  // Symbol-resolution fixtures start parsed. Keep large-document cases partial
  // so they still exercise the editor's bounded incremental parse behavior.
  if (code.length <= 4096) ensureSyntaxTree(result, code.length, 1000);
  return result;
}

test("selection and saved matrix hypotheses expose their actual signatures and inference limits", () => {
  const selection = getActiveCall(state("import openecon as oe\noe.discrim_stepwise(data, 'group', ['x'], method="));
  assert.equal(selection?.entry.name, "openecon.discrim_stepwise");
  assert.equal(selection?.entry.parameters.find((p) => p.name === "method")?.default, "'stepwise'");
  assert.match(selection?.entry.description ?? "", /not post-selection inference/);
  for (const name of ["manova_contrast", "rm_mtest"]) {
    const call = getActiveCall(state(`import openecon as oe\noe.${name}(saved, L, M=`));
    assert.equal(call?.entry.name, `openecon.${name}`);
    assert.equal(call?.entry.returns, "openecon.TableSet");
    assert.match(call?.entry.description ?? "", /L B M = C/);
    assert.match(call?.entry.description ?? "", /upper-bound/);
  }
  assert.equal(API_CATALOG["openecon.discrim_stepwise_predict"].returns, "openecon.DataFrame");
});

test("compact stored defaults preserve every complete editor parameter", () => {
  for (const entry of storedCatalog) {
    const value = entry as StoredApiEntry;
    const name = decodeApiEntry(value).name;
    assert.ok(name);
    if (value.I !== undefined) {
      assert.equal(typeof COMPACT_SIGNATURE_SUFFIXES[value.I], "string",
        `Unknown independent signature token ${value.I} for ${name}`);
    }
    const signature = value.signature ?? (value.s?.startsWith("(")
      ? name.slice(name.lastIndexOf(".") + 1) + value.s : value.s)
      ?? (value.S !== undefined ? name.slice(name.lastIndexOf(".") + 1) + value.S : undefined)
      ?? (value.I !== undefined ? name.slice(name.lastIndexOf(".") + 1) + COMPACT_SIGNATURE_SUFFIXES[value.I] : undefined);
    assert.equal(API_CATALOG[name].signature, signature);
    assert.equal(Object.hasOwn(API_CATALOG[name], "P"), false);
    assert.equal(Object.hasOwn(API_CATALOG[name], "S"), false);
    assert.equal(Object.hasOwn(API_CATALOG[name], "I"), false);
    assert.equal(API_CATALOG[name].description, value.description ?? value.d);
    const legacyReturns = typeof value.r === "string" && value.r.startsWith("~")
      ? `openecon.${value.r.slice(1)}` : value.r;
    const returns = Object.hasOwn(value, "returns") ? value.returns
      : Object.hasOwn(value, "r") ? legacyReturns
      : Object.hasOwn(value, "R") ? `openecon.${value.R}`
      : value.T !== undefined ? COMPACT_RETURN_TYPES[value.T] : undefined;
    assert.equal(API_CATALOG[name].returns, returns);
    assert.equal(Object.hasOwn(API_CATALOG[name], "T"), false);
    assert.equal(Object.hasOwn(API_CATALOG[name], "R"), false);
    assert.equal(Object.hasOwn(API_CATALOG[name], "r"), false);
    for (const storedParameter of value.parameters ?? value.p ?? (value.P !== undefined ? COMPACT_PARAMETER_LISTS[value.P] : [])) {
      const parameter = typeof storedParameter === "number"
        ? COMPLETE_PARAMETER_RECORDS[storedParameter] : storedParameter;
      assert.ok(parameter);
      const parameterName = parameter.name ?? (typeof parameter.n === "number"
        ? COMPACT_PARAMETER_NAMES[parameter.n] : parameter.n);
      const decoded = API_CATALOG[name].parameters.find((p) => p.name === parameterName);
      const literal = parameter.v === null ? "None" : typeof parameter.v === "boolean"
        ? (parameter.v ? "True" : "False") : typeof parameter.v === "number"
          ? String(parameter.v) : parameter.v;
      assert.equal(decoded?.default, parameter.default !== undefined ? parameter.default : literal
        ?? (parameter.V === 0 ? "None" : undefined)
        ?? (parameter.B === 0 ? "False" : parameter.B === 1 ? "True" : undefined)
        ?? (parameter.D !== undefined ? ["'raise'", "0.05", "'cpu'", "100000000", "50000000", "'drop'", "0.95", "0", "DEFAULT_BYTES", "DEFAULT_WORK", "'two-sided'", "'nonrobust'", "'constant'", "1000000", "2000000000", "100000"][parameter.D] : undefined)
        ?? (parameter.q !== undefined ? `'${parameter.q}'` : undefined)
        ?? (parameter.N === 1 ? "None" : undefined));
      assert.equal(decoded?.description, parameter.description ?? parameter.d);
      assert.deepEqual(decoded?.choices, parameter.choices ?? parameter.c);
      assert.equal(Object.hasOwn(decoded!, "d"), false);
      assert.equal(Object.hasOwn(decoded!, "v"), false);
      assert.equal(Object.hasOwn(decoded!, "V"), false);
      assert.equal(Object.hasOwn(decoded!, "B"), false);
      assert.equal(Object.hasOwn(decoded!, "D"), false);
      assert.equal(Object.hasOwn(decoded!, "N"), false);
      assert.equal(Object.hasOwn(decoded!, "q"), false);
    }
  }
  const call = getActiveCall(state("import openecon as oe\noe.rm_contrasts(saved, C, alpha="));
  assert.equal(call?.entry.parameters.find((p) => p.name === "alpha")?.default, "0.05");
  const weight = getActiveCall(state("import openecon as oe\noe.discrim(data, 'group', ['x'], weight_type="));
  assert.equal(weight?.entry.parameters.find((p) => p.name === "weight_type")?.default, "'fweight'");
});

test("explicit legacy metadata wins over compact aliases including v defaults", () => {
  const stored: StoredApiEntry = {
    name: "openecon.Legacy.call", n: "openecon.Other.call",
    signature: "call(legacy=5)", s: "call(legacy=99)",
    description: "Legacy help. Türkçe.", d: "Compact help.",
    returns: "LegacyReturn", r: "~CompactReturn",
    kind: "method", owner: "openecon.DeclaredOwner",
    parameters: [
      { name: "legacy", kind: "k", k: "p", default: "5", v: "99",
        description: "Legacy parameter.", d: "Compact parameter." },
      { name: "compact", v: "'iid_multivariate_normal'", d: "Declared subject model.",
        choices: ["'iid_multivariate_normal'"] },
    ],
    p: [],
  };
  const before = structuredClone(stored);
  assert.deepEqual(decodeApiEntry(stored), {
    name: "openecon.Legacy.call", signature: "call(legacy=5)",
    description: "Legacy help. Türkçe.", returns: "LegacyReturn",
    kind: "method", owner: "openecon.DeclaredOwner",
    parameters: [
      { name: "legacy", kind: "keyword-only", default: "5", description: "Legacy parameter." },
      { name: "compact", kind: "positional-or-keyword", default: "'iid_multivariate_normal'",
        description: "Declared subject model.", choices: ["'iid_multivariate_normal'"] },
    ],
  });
  assert.deepEqual(stored, before);
});

test("both compact codec forms preserve every logical field and explicit precedence", () => {
  for (const signatureKey of ["s", "S"] as const) {
    for (const kindKey of ["k", "t"] as const) {
      for (const returnKey of ["r", "R"] as const) {
        const stored: StoredApiEntry = {
          n: "~Result.call", k: "m",
          [signatureKey]: "(frame, /, *, option=None, enabled=True, disabled=False)",
          d: "Complete Türkçe help.",
          [returnKey]: returnKey === "r" ? "~DataFrame" : "DataFrame",
          p: [
            { n: "frame", [kindKey]: "p", a: "DataFrame", d: "Declared frame." },
            { n: "option", [kindKey]: "k", v: null, c: [], d: "" },
            { n: "enabled", [kindKey]: "k", v: true },
            { n: "disabled", [kindKey]: "k", v: false },
          ],
        };
        const expected = {
          name: "openecon.Result.call", kind: "method", owner: "openecon.Result",
          signature: "call(frame, /, *, option=None, enabled=True, disabled=False)",
          description: "Complete Türkçe help.", returns: "openecon.DataFrame",
          parameters: [
            { name: "frame", kind: "positional-only", annotation: "DataFrame", description: "Declared frame." },
            { name: "option", kind: "keyword-only", default: "None", choices: [], description: "" },
            { name: "enabled", kind: "keyword-only", default: "True" },
            { name: "disabled", kind: "keyword-only", default: "False" },
          ],
        };
        const before = structuredClone(stored);
        assert.deepEqual(decodeApiEntry(stored), expected);
        assert.deepEqual(stored, before);
        const mixed = structuredClone(stored);
        Object.assign(mixed, { signature: "", s: "ignored()", S: "(ignored)",
          returns: "", r: "~Ignored", R: "Ignored", kind: "method", k: "c" });
        Object.assign(mixed.p![1], { default: "", v: true, kind: "positional-only", k: "k", t: "*" });
        const mixedExpected = structuredClone(expected);
        mixedExpected.signature = "";
        mixedExpected.returns = "";
        mixedExpected.parameters[1].default = "";
        mixedExpected.parameters[1].kind = "positional-only";
        const mixedBefore = structuredClone(mixed);
        assert.deepEqual(decodeApiEntry(mixed), mixedExpected);
        assert.deepEqual(mixed, mixedBefore);
      }
    }
  }
});

test("numeric and quoted defaults preserve exact complete metadata and mixed precedence", () => {
  const stored: StoredApiEntry = {
    n: "~call", S: "(*, option=None)", d: "Complete literal metadata. Türkçe.", R: "DataFrame",
    p: [
      { n: "zero", k: "k", v: 0, a: "int", d: "Zero.", c: ["0", "1"] },
      { n: "negative", k: "k", v: -1 },
      { n: "maximum", k: "k", v: Number.MAX_SAFE_INTEGER },
      { n: "minimum", k: "k", v: -Number.MAX_SAFE_INTEGER },
      { n: "empty", k: "k", q: "" },
      { n: "quoted", k: "k", q: "Türkçe" },
      { n: "none", k: "k", v: null },
      { n: "enabled", k: "k", v: true },
      { n: "disabled", k: "k", v: false },
      { n: "floatSpelling", k: "k", v: "1.0" },
      { n: "negativeZero", k: "k", v: "-0" },
      { n: "bigInteger", k: "k", v: "9007199254740992" },
      { n: "escaped", k: "k", v: "'path\\file'" },
      { n: "doubleQuoted", k: "k", v: '"cpu"' },
    ],
  };
  const defaults = ["0", "-1", "9007199254740991", "-9007199254740991", "''", "'Türkçe'",
    "None", "True", "False", "1.0", "-0", "9007199254740992", "'path\\file'", '"cpu"'];
  const expected = {
    name: "openecon.call", kind: "function", signature: "call(*, option=None)",
    description: "Complete literal metadata. Türkçe.", returns: "openecon.DataFrame",
    parameters: stored.p!.map((parameter, index) => ({ name: parameter.n,
      kind: "keyword-only", default: defaults[index],
      ...(index === 0 ? { annotation: "int", description: "Zero.", choices: ["0", "1"] } : {}) })),
  };
  const before = structuredClone(stored);
  assert.deepEqual(decodeApiEntry(stored), expected);
  assert.deepEqual(stored, before);
  const mixed = structuredClone(stored);
  Object.assign(mixed.p![0], { default: "", v: 17, q: "ignored" });
  Object.assign(mixed.p![1], { v: 0, q: "ignored" });
  Object.assign(mixed.p![2], { v: false, q: "ignored" });
  Object.assign(mixed.p![3], { v: null, q: "ignored" });
  Object.assign(mixed.p![4], { v: "", q: "ignored" });
  Object.assign(mixed.p![5], { v: "'legacy'", q: "ignored" });
  const mixedExpected = structuredClone(expected);
  for (const [index, value] of ["", "0", "False", "None", "", "'legacy'"].entries()) {
    mixedExpected.parameters[index].default = value;
  }
  const mixedBefore = structuredClone(mixed);
  assert.deepEqual(decodeApiEntry(mixed), mixedExpected);
  assert.deepEqual(mixed, mixedBefore);
});

test("None default tag retains legacy null and full v V q precedence without mutation", () => {
  const stored: StoredApiEntry = {
    n: "~call", S: "()", d: "Complete help.", R: "DataFrame",
    p: [
      { n: "required", k: "k" },
      { n: "none", k: "k", V: 0, q: "ignored" },
      { n: "legacyNull", k: "k", v: null, V: "ignored", q: "ignored" },
      { n: "legacyFalse", k: "k", v: false, V: "ignored", q: "ignored" },
      { n: "legacyZero", k: "k", v: 0, V: "ignored", q: "ignored" },
      { n: "legacyEmpty", k: "k", v: "", V: "ignored", q: "ignored" },
      { n: "fullEmpty", k: "k", default: "", v: null, V: "ignored", q: "ignored" },
      { n: "quoted", k: "k", q: "None" },
    ],
  };
  const defaults = [undefined, "None", "None", "False", "0", "", "", "'None'"];
  const expected = {
    name: "openecon.call", kind: "function", signature: "call()", description: "Complete help.",
    returns: "openecon.DataFrame", parameters: stored.p!.map((parameter, index) => ({
      name: parameter.n, kind: "keyword-only",
      ...(defaults[index] !== undefined ? { default: defaults[index] } : {}),
    })),
  };
  const before = structuredClone(stored);
  assert.deepEqual(decodeApiEntry(stored), expected);
  assert.deepEqual(stored, before);
  for (const tag of [null, true, false, 1, "0", Number.NaN, Infinity, -0, []]) {
    const invalid = { ...stored, p: [{ n: "x", V: tag }] } as unknown as StoredApiEntry;
    assert.throws(() => decodeApiEntry(invalid), /integer zero None tag/);
  }
});

test("parameter name tokens preserve complete metadata and explicit name precedence", () => {
  const names = [...COMPACT_PARAMETER_NAMES, "max_work_alias", "Türkçe"];
  const stored: StoredApiEntry = {
    n: "~call", S: "(*, max_work=0)", d: "Exact parameter names. Türkçe.", R: "DataFrame",
    p: names.map((name, index) => ({ n: index < COMPACT_PARAMETER_NAMES.length ? index : name,
      k: "k", v: index, a: "int", d: "Complete parameter help.", c: ["0", "1"] })),
  };
  const expected = {
    name: "openecon.call", kind: "function", signature: "call(*, max_work=0)",
    description: "Exact parameter names. Türkçe.", returns: "openecon.DataFrame",
    parameters: names.map((name, index) => ({ name, kind: "keyword-only", default: String(index),
      annotation: "int", description: "Complete parameter help.", choices: ["0", "1"] })),
  };
  const before = structuredClone(stored);
  assert.deepEqual(decodeApiEntry(stored), expected);
  assert.deepEqual(stored, before);
  const mixed = structuredClone(stored);
  const mixedExpected = structuredClone(expected);
  for (const [index, malformed] of [true, null, { invalid: "unused" }, 100, 1.5].entries()) {
    const explicitName = index === 0 ? "" : `legacy_${index}`;
    Object.assign(mixed.p![index], { name: explicitName, n: malformed });
    mixedExpected.parameters[index].name = explicitName;
  }
  const mixedBefore = structuredClone(mixed);
  assert.deepEqual(decodeApiEntry(mixed), mixedExpected);
  assert.deepEqual(mixed, mixedBefore);
  for (const token of [true, false, null, -1, 31, 1.5, Number.NaN, Infinity, -0, {}, []]) {
    const invalid = { ...stored, p: [{ n: token }] } as unknown as StoredApiEntry;
    assert.throws(() => decodeApiEntry(invalid), /name/);
  }
});

test("parameter kind key aliases restore every signature domain without leaking or mutation", () => {
  const stored: StoredApiEntry = {
    n: "~example", s: "example(k, /, ordinary, *args, option=None, **kwargs)",
    d: "Complete help.",
    p: [
      { n: "k", k: "p" },
      { n: "ordinary" },
      { n: "args", k: "*" },
      { n: "option", k: "k", v: "None" },
      { n: "kwargs", k: "**" },
    ],
  };
  const before = structuredClone(stored);
  const decoded = decodeApiEntry(stored);
  assert.equal(decoded.name, "openecon.example");
  assert.deepEqual(decoded.parameters.map(p => p.kind), [
    "positional-only", "positional-or-keyword", "var-positional", "keyword-only", "var-keyword",
  ]);
  assert.equal(decoded.parameters[0].name, "k");
  assert.ok(decoded.parameters.every(p => !Object.hasOwn(p, "k")));
  const legacy = structuredClone(stored);
  for (const parameter of legacy.p!) {
    if (parameter.k !== undefined) {
      parameter.t = parameter.k;
      delete parameter.k;
    }
  }
  const legacyBefore = structuredClone(legacy);
  assert.deepEqual(decodeApiEntry(legacy), decoded);
  assert.deepEqual(legacy, legacyBefore);
  assert.equal(decodeApiEntry({ ...stored, p: [{ n: "k", kind: "positional-only", k: "k", t: "**" }] }).parameters[0].kind,
    "positional-only");
  assert.equal(decodeApiEntry({ ...stored, p: [{ n: "k", k: "p", t: "**" }] }).parameters[0].kind,
    "positional-only");
  assert.deepEqual(stored, before);
});

test("return namespace compression preserves mixed old and new logical metadata", () => {
  const parameters = [
    { n: "frame", annotation: "DataFrame", kind: "p", d: "Declared frame." },
    { n: "option", v: "", c: [], d: "", kind: "k" },
  ];
  const common = { s: "to_frame(frame, /, *, option='')", d: "Türkçe complete help.", p: parameters };
  const stored: StoredApiEntry[] = [
    { ...structuredClone(common), n: "~SurveyRegressionResult.to_frame", kind: "m", R: "DataFrame" },
    { ...structuredClone(common), n: "~survey_probit_replicate", R: "SurveyRegressionResult" },
    { ...structuredClone(common), n: "~survey_poisson_replicate", r: "openecon.SurveyRegressionResult" },
    { ...structuredClone(common), name: "legacy.foreign", r: "external.Result" },
    { ...structuredClone(common), name: "legacy.marker", r: "~DataFrame" },
    { ...structuredClone(common), name: "explicit.marker", returns: "~DataFrame",
      r: "~Ignored", R: "Ignored" },
    { ...structuredClone(common), name: "legacy.empty", r: "", R: "Ignored" },
    { ...structuredClone(common), name: "explicit.empty", returns: "", r: "Ignored", R: "Ignored" },
    { ...structuredClone(common), name: "prefix.empty", R: "" },
    { ...structuredClone(common), name: "prefix.nested", R: "survey.Result | None" },
    { ...structuredClone(common), name: "explicit.precedence", returns: "external.Explicit",
      r: "external.Legacy", R: "Ignored" },
    { ...structuredClone(common), name: "legacy.precedence", r: "external.Legacy", R: "Ignored" },
    { ...structuredClone(common), name: "absent.return" },
  ];
  const before = structuredClone(stored);
  const decoded = stored.map(decodeApiEntry);
  assert.deepEqual(decoded.map(entry => entry.returns), [
    "openecon.DataFrame", "openecon.SurveyRegressionResult", "openecon.SurveyRegressionResult",
    "external.Result", "openecon.DataFrame", "~DataFrame", "", "", "openecon.", "openecon.survey.Result | None",
    "external.Explicit", "external.Legacy", undefined,
  ]);
  assert.equal(decoded[0].name, "openecon.SurveyRegressionResult.to_frame");
  assert.equal(decoded[0].kind, "method");
  assert.equal(decoded[0].owner, "openecon.SurveyRegressionResult");
  assert.equal(decoded[1].kind, "function");
  for (const entry of decoded) {
    assert.equal(entry.signature, common.s);
    assert.equal(entry.description, common.d);
    assert.deepEqual(entry.parameters, [
      { name: "frame", annotation: "DataFrame", kind: "positional-only", description: "Declared frame." },
      { name: "option", default: "", choices: [], description: "", kind: "keyword-only" },
    ]);
    assert.equal(Object.hasOwn(entry, "R"), false);
    assert.equal(Object.hasOwn(entry, "r"), false);
  }
  assert.equal(Object.hasOwn(decoded.at(-1)!, "returns"), false);
  assert.deepEqual(stored, before);
});

test("stored default function kind restores while classes and methods keep their kinds", () => {
  assert.equal(API_CATALOG["openecon.catreg_nominal"].kind, "function");
  assert.equal(API_CATALOG["openecon.Latex"].kind, "class");
  assert.equal(API_CATALOG["openecon.DataFrame.head"].kind, "method");
  assert.equal(API_CATALOG["openecon.catreg_nominal"].returns, "openecon.TableSet");
  assert.ok(API_CATALOG["openecon.catreg_nominal"].parameters.some(p => p.name === "max_bytes"));
});

test("compact description fields preserve categorical option help and keyword signatures", () => {
  const names = ["catreg_nominal_response", "catreg_ordinal_response", "catreg_outcome_predict",
    "catreg_bootstrap", "catpca_varimax", "catpca_promax", "catpca_rotated_predict",
    "loglinear_multinomial", "loglinear_product_multinomial", "loglinear_sampling_compare", "loglinear_select"];
  for (const name of names) {
    const entry = API_CATALOG[`openecon.${name}`];
    assert.equal(entry.returns, "openecon.TableSet");
    assert.ok(entry.description.length > 20);
    assert.equal("d" in entry, false);
    for (const parameter of entry.parameters) {
      assert.equal("d" in parameter, false);
      assert.equal("f" in parameter, false);
    }
  }
  assert.match(API_CATALOG["openecon.loglinear_multinomial"].description, /multinomial/);
  assert.match(API_CATALOG["openecon.catreg_bootstrap"].description, /full-refit/);
  const call = getActiveCall(state("import openecon as oe\noe.catreg_ordinal_response(data, 'y', ['x'], outcome_order="));
  assert.equal(call?.activeParameter, "outcome_order");
  assert.equal(call?.entry.parameters.find(p => p.name === "outcome_order")?.kind, "keyword-only");
  const known = API_CATALOG["openecon.ols"].parameters.find(p => p.name === "weights");
  assert.ok(known?.description && known.description.length > 0);
  assert.equal(API_CATALOG["openecon.catreg_bootstrap"].parameters.find(p => p.name === "confidence")?.default, "0.95");
  assert.equal(API_CATALOG["openecon.pca"].parameters.find(p => p.name === "components")?.default, "None");
});

test("stored API parameter kinds preserve positional and keyword signature help", () => {
  const positional = getActiveCall(state("import openecon as oe\noe.polyserial(data, 'score', "));
  assert.equal(positional?.activeParameter, "ordinal");
  assert.ok(positional?.entry.parameters.slice(0, 3).every(
    (parameter) => parameter.kind === "positional-or-keyword",
  ));
  const keyword = getActiveCall(state("import openecon as oe\noe.icc(data, raters, level="));
  assert.equal(keyword?.activeParameter, "level");
  assert.equal(keyword?.entry.parameters.find((parameter) => parameter.name === "level")?.kind, "keyword-only");
  const builtin = getActiveCall(state("len(data"));
  assert.equal(builtin?.activeParameter, "obj");
  assert.equal(builtin?.entry.parameters[0]?.kind, "positional-only");
  const variadic = getActiveCall(state("print('first', 'second', "));
  assert.equal(variadic?.activeParameter, "args");
  assert.equal(variadic?.entry.parameters[0]?.kind, "var-positional");
  const arbitraryKeyword = getActiveCall(state("import openecon as oe\ndf = oe.DataFrame()\ndf.assign(new_column="));
  assert.equal(arbitraryKeyword?.activeParameter, "kwargs");
  assert.equal(arbitraryKeyword?.entry.parameters[0]?.kind, "var-keyword");
});
test("implicit stored function kind and method owner preserve complete API metadata", () => {
  assert.equal(API_CATALOG["openecon.cif_compare"].kind, "function");
  assert.equal(API_CATALOG["openecon.OLSResult.diagnostics"].kind, "method");
  assert.equal(API_CATALOG["openecon.OLSResult.diagnostics"].owner, "openecon.OLSResult");
  assert.equal(API_CATALOG["openecon.DataFrame"].kind, "class");
});
test("conjoint help separates scored responses from keyword model options", () => {
  const responses = getActiveCall(state("import openecon as oe\noe.conjoint_fit(plan, "));
  assert.equal(responses?.activeParameter, "responses");
  assert.equal(responses?.entry.returns, "openecon.TableSet");
  const inference = getActiveCall(state("import openecon as oe\noe.conjoint_fit(plan, responses, covariance="));
  assert.equal(inference?.activeParameter, "covariance");
  assert.equal(inference?.entry.parameters.find((p) => p.name === "covariance")?.kind, "keyword-only");
});
function symbol(code: string, target: string, occurrence = 0) {
  let pos = -1;
  for (let i = 0; i <= occurrence; i++) pos = code.indexOf(target, pos + 1);
  assert.ok(pos >= 0, target);
  const expression = target.split("(")[0];
  const identifier = expression.match(/[A-Za-z_]\w*$/)!;
  return getEditorSymbol(
    state(code),
    pos + identifier.index! + Math.max(1, Math.floor(identifier[0].length / 2)),
  );
}
function complete(code: string, explicit = true, readOnly = false) {
  return editorCompletionSource(
    new CompletionContext(state(code, readOnly), code.length, explicit),
  );
}
function markedCompletion(code: string, explicit = false) {
  const pos = code.indexOf("|");
  assert.ok(pos >= 0, "Cursor marker exists");
  const source = code.slice(0, pos) + code.slice(pos + 1);
  return {
    source,
    result: editorCompletionSource(
      new CompletionContext(state(source), pos, explicit),
    ),
  };
}
function acceptedValue(code: string, label: string) {
  const { source, result } = markedCompletion(code);
  const option = result?.options.find((item) => item.label === label);
  assert.ok(option, label);
  const insert = option.apply ?? option.label;
  assert.equal(typeof insert, "string");
  return (
    source.slice(0, result!.from) +
    insert +
    source.slice(result!.to ?? code.indexOf("|"))
  );
}

test("hover follows actual module and from-import aliases", () => {
  const code =
    "import openecon as econ\nfrom openecon import ols as regress\necon.ols(data=df)\nregress(data=df)";
  assert.equal(symbol(code, "ols(data")?.entry.name, "openecon.ols");
  assert.equal(symbol(code, "regress(data")?.entry.name, "openecon.ols");
  assert.equal(symbol("oe.ols(data=df)", "ols")?.entry, undefined);
});

test("registry models and helpers have offline hover and completion help", () => {
  for (const name of ["nardl", "nardl_multipliers", "tobit", "truncreg", "intreg", "xtcd"]) {
    const code = `import openecon as oe\noe.${name}(`;
    assert.equal(symbol(code, `${name}(`)?.entry.name, `openecon.${name}`);
    assert.ok(complete(`import openecon as oe\noe.${name.slice(0, -1)}`)?.options
      .some((option) => option.label === name), name);
  }
  const fit = "import openecon as oe\nm=oe.tobit(data=df,y='y',x=['x'])\nm.summary()";
  assert.equal(symbol(fit, "summary")?.entry.name, "openecon.ResultBundle.summary");
  assert.equal(symbol(fit.replace("summary", "vif"), "vif")?.entry, undefined);
  for (const helper of ["xtcd", "nardl_multipliers", "predict", "margins"]) {
    const code = `import openecon as oe\nout=oe.${helper}(data=df)\nout.to_latex()`;
    assert.equal(symbol(code, "to_latex")?.entry.name, "openecon.DataFrame.to_latex");
  }
});

test("unknown objects, assignment shadowing, comments and strings have no invented docs", () => {
  assert.equal(
    symbol("import openecon as oe\nthing.ols(data=df)", "ols"),
    null,
  );
  assert.equal(
    symbol("import openecon as oe\noe = object()\noe.ols(data=df)", "ols"),
    null,
  );
  assert.equal(symbol('import openecon as oe\n"oe.ols(data=df)"', "ols"), null);
  assert.equal(symbol("import openecon as oe\n# oe.ols(data=df)", "ols"), null);
  assert.equal(symbol("print = 4\nprint()", "print()"), null);
});

test("inferred frames and model results expose only their actual type's methods", () => {
  const code =
    "import openecon as oe\ndf=oe.example()\ncopy=df.head()\ncopy.to_latex()\nmodel=oe.ols(data=df,y='wage',x=['education'])\nmodel.summary()\nmodel.predict()";
  assert.equal(
    symbol(code, "to_latex")?.entry.name,
    "openecon.DataFrame.to_latex",
  );
  assert.equal(
    symbol(code, "summary")?.entry.name,
    "openecon.OLSResult.summary",
  );
  assert.equal(
    symbol(code, "predict()")?.entry.name,
    "openecon.OLSResult.predict",
  );
  assert.equal(
    symbol(
      "import openecon as oe\nm=oe.logit(data=df,y='employed',x=['education'])\nm.vif()",
      "vif",
    ),
    null,
  );
});

test("network construction exposes its own methods and graph chart help", () => {
  const code = "import openecon as oe\ngraph=oe.network(data=edges)\ngraph.pagerank()\noe.plot.network(graph)";
  assert.equal(symbol(code, "pagerank()")?.entry.name, "openecon.Network.pagerank");
  assert.equal(symbol(code, "network(graph)")?.entry.name, "openecon.plot.network");
  const options = complete("import openecon as oe\ngraph=oe.network(data=edges)\ngraph.")?.options;
  assert.ok(options?.some(item => item.label === "pagerank"));
  assert.ok(options?.some(item => item.label === "shortest_paths"));
  assert.ok(!options?.some(item => item.label === "vif"));
});

test("network workflow return values expose chained completion without executing code", () => {
  const prelude = "import openecon as oe\ngraph=oe.network(data=edges)\nflow=graph.max_flow('a','b')\n";
  assert.equal(symbol(prelude + "flow.summary()", "summary")?.entry.name,
    "openecon.NetworkFlowResult.summary");
  const methods = complete(prelude + "flow.")?.options;
  assert.ok(methods?.some(item => item.label === "to_latex"));
  assert.ok(!methods?.some(item => item.label === "pagerank"));
  const projected = "import openecon as oe\ngraph=oe.network(data=edges)\nprojection=graph.bipartite_projection()\n";
  assert.equal(symbol(projected + "projection.degree()", "degree")?.entry.name,
    "openecon.Network.degree");
  const graphMethods = complete(projected + "projection.")?.options;
  assert.ok(graphMethods?.some(item => item.label === "shortest_path"));
  assert.ok(graphMethods?.some(item => item.label === "link_prediction"));
});

test("network inference and global cut results retain chained table help", () => {
  const prelude = "import openecon as oe\ngraph=oe.network(data=edges)\ncut=graph.global_min_cut()\n";
  assert.equal(symbol(prelude + "cut.summary()", "summary")?.entry.name,
    "openecon.NetworkCutResult.summary");
  const cutMethods = complete(prelude + "cut.")?.options;
  assert.ok(cutMethods?.some(item => item.label === "to_latex"));
  assert.ok(!cutMethods?.some(item => item.label === "pagerank"));
  const triads = prelude + "counts=graph.triad_census()\n";
  assert.equal(symbol(triads + "counts.head()", "head")?.entry.name,
    "openecon.DataFrame.head");
  const comparison = prelude + "comparison=graph.qap_correlation(other)\n";
  assert.equal(symbol(comparison + "comparison.to_latex()", "to_latex")?.entry.name,
    "openecon.DataFrame.to_latex");
});

test("function parameters shadow global aliases and nested locals stay scoped", () => {
  const code =
    "import openecon as oe\ndef first(oe):\n    oe.ols()\ndef second():\n    import openecon as local\n    local.ols()\nlocal.ols()\n";
  assert.equal(symbol(code, "oe.ols", 0), null);
  assert.equal(symbol(code, "local.ols", 0)?.entry.name, "openecon.ols");
  assert.equal(symbol(code, "local.ols", 1), null);
});

test("user function docs and multiline signatures are derived without executing code", () => {
  const code =
    'def adjust(data,\n precision=4, *, robust=True, **options):\n    """Adjust a supplied frame.\n\nLonger explanation."""\n    return data\nadjust(';
  const entry = symbol(code, "adjust(", 1)?.entry;
  assert.equal(
    entry?.signature,
    "adjust(data, precision=4, *, robust=True, **options)",
  );
  assert.equal(entry?.description, "Adjust a supplied frame.");
  assert.deepEqual(
    entry?.parameters.map((parameter) => parameter.name),
    ["data", "precision", "robust", "options"],
  );
  assert.equal(entry?.parameters[2].kind, "keyword-only");
});

test("signature help counts only outer argument commas", () => {
  const prefix = "def f(first, second, third):\n    return first\n";
  for (const argument of [
    "[1, 2, 3]",
    "{'a': 1, 'b': 2}",
    "dict(a=1, b=2)",
    "'a, b'",
    "(1, 2)",
  ]) {
    const active = getActiveCall(state(`${prefix}f(${argument}, `));
    assert.equal(active?.activeParameter, "second", argument);
    assert.equal(active?.argumentIndex, 1, argument);
  }
});

test("innermost open call wins and closed calls restore their outer call", () => {
  const prefix =
    "def outer(first, second):\n    pass\ndef inner(a, b):\n    pass\n";
  assert.equal(
    getActiveCall(state(`${prefix}outer(inner(1, `))?.entry.name,
    "inner",
  );
  assert.equal(
    getActiveCall(state(`${prefix}outer(inner(1, 2), `))?.entry.name,
    "outer",
  );
  assert.equal(getActiveCall(state(`${prefix}outer(inner(1, 2))`)), null);
});

test("keywords select the named parameter regardless of declaration order", () => {
  const code = "def f(first, second, third):\n    pass\nf(third=3, first=";
  const active = getActiveCall(state(code));
  assert.equal(active?.activeParameter, "first");
  assert.deepEqual(active?.supplied, ["third"]);
});

test("literal calls and comments never trigger signature help", () => {
  assert.equal(getActiveCall(state('"print(hello, "')), null);
  assert.equal(getActiveCall(state("# print(")), null);
  assert.equal(getActiveCall(state("print(1, # hello")), null);
});

test("completion resolves module children, inferred methods, variables and parameters", () => {
  const rootOptions = complete("import openecon as oe\noe.")!.options;
  assert.ok(!rootOptions.some((option) => option.label === "OLSResult"));
  assert.ok(rootOptions.some((option) => option.label === "DataFrame"));
  assert.ok(
    complete("import openecon as econ\necon.")?.options.some(
      (option) => option.label === "ols",
    ),
  );
  assert.ok(
    complete("import openecon as oe\noe.plot.")?.options.some(
      (option) => option.label === "scatter",
    ),
  );
  assert.ok(
    complete("import openecon as oe\ndf=oe.example()\ndf.")?.options.some(
      (option) => option.label === "to_latex",
    ),
  );
  assert.ok(
    complete("def f(first, *, robust=True):\n    pass\nf(")?.options.some(
      (option) => option.label === "robust=",
    ),
  );
  const keywords = complete(
    "def f(first, *, robust=True):\n    pass\nf(robust=True, ",
  )?.options.map((option) => option.label);
  assert.ok(!keywords?.includes("robust="));
  assert.ok(
    complete("def adjust(x):\n    pass\nadj")?.options.some(
      (option) => option.label === "adjust",
    ),
  );
});

test("completion leaves unknown objects and protected text alone", () => {
  assert.equal(complete("unknown."), null);
  assert.equal(complete('import openecon as oe\n"oe.'), null);
  assert.equal(complete("import openecon as oe\n# oe."), null);
  assert.equal(complete("import openecon as oe\noe.", true, true), null);
});

test("updating a document invalidates earlier type and alias inferences", () => {
  const original = state("import openecon as oe\ndf=oe.example()\ndf.head()");
  assert.equal(
    getEditorSymbol(original, original.doc.length - 3)?.entry.name,
    "openecon.DataFrame.head",
  );
  const changed = original.update({
    changes: {
      from: 0,
      to: original.doc.length,
      insert: "df=object()\ndf.head()",
    },
  }).state;
  assert.equal(getEditorSymbol(changed, changed.doc.length - 3), null);
});

test("updated aliases, context bindings and comprehension targets suppress stale help", () => {
  for (const snippet of [
    "oe += 1\noe.ols()",
    "with resource() as oe:\n    oe.ols()",
    "try:\n    pass\nexcept Exception as oe:\n    oe.ols()",
    "[oe.ols() for oe in items]",
    "def inner():\n    oe.ols()\n    oe=object()",
    "make = lambda oe: oe.ols()",
    "(oe := object())\noe.ols()",
    "del oe\noe.ols()",
    "result = [oe.ols() for oe in items]",
  ]) {
    assert.equal(
      symbol(`import openecon as oe\n${snippet}`, "ols()"),
      null,
      snippet,
    );
  }
  assert.equal(
    symbol(
      "import openecon as oe\n[oe.ols() for oe in items]\noe.ols()",
      "ols()",
      1,
    )?.entry.name,
    "openecon.ols",
  );
});

test("unpacking does not assign the original frame type to each element", () => {
  assert.equal(
    symbol("import openecon as oe\na,b=oe.example()\na.head()", "head"),
    null,
  );
  assert.equal(
    symbol("import openecon as oe\nx=df=oe.example()\ndf.head()", "head")?.entry
      .name,
    "openecon.DataFrame.head",
  );
});

test("keyword completion excludes arguments already supplied positionally", () => {
  const options = complete(
    "def f(first, second, *, robust=False):\n    pass\nf(1, ",
  )!.options;
  assert.ok(!options.some((option) => option.label === "first="));
  assert.ok(options.some((option) => option.label === "second="));
});

test("Turkish identifiers receive function docs and parameter completion", () => {
  const code =
    'def ölç(veri, *, ücret=4):\n    """Ücreti ölç."""\n    return veri\nölç(ücret=';
  assert.equal(getActiveCall(state(code))?.activeParameter, "ücret");
  assert.equal(
    getEditorSymbol(state(code), code.lastIndexOf("ölç") + 1)?.entry
      .description,
    "Ücreti ölç.",
  );
  assert.ok(
    complete(
      "def ölç(veri, *, ücret=4):\n    return veri\nölç(ü",
    )?.options.some((option) => option.label === "ücret="),
  );
});

test("assistance can resolve a requested token beyond the initial partial parse", () => {
  const prefix =
    "import openecon as econ\n" + "# deferred analysis notes\n".repeat(400);
  const code = `${prefix}econ.ols(covariance=`;
  const editor = state(code);
  assert.ok(
    syntaxTree(editor).length < code.length,
    "The initial parser stops before the requested call",
  );
  const requested = getEditorSymbol(editor, prefix.length + "econ.o".length);
  // A busy machine can exhaust the query's short parse budget. Assistance
  // stays absent rather than guessing until the view's parser finishes.
  if (requested) assert.equal(requested.entry.name, "openecon.ols");
  assert.ok(ensureSyntaxTree(editor, code.length, 1000));
  assert.equal(
    getEditorSymbol(editor, prefix.length + "econ.o".length)?.entry.name,
    "openecon.ols",
  );
  assert.equal(getActiveCall(editor)?.activeParameter, "covariance");
  assert.equal(editor.doc.toString(), code);
});

test("known argument punctuation opens useful suggestions automatically", () => {
  for (const code of [
    "import openecon as oe\noe.ols(",
    "import openecon as oe\noe.ols(data=df, ",
  ]) {
    const options = complete(code, false)?.options;
    assert.ok(options?.some((option) => option.label === "y="));
  }
  assert.ok(
    complete("import openecon as oe\noe.ols(covariance=", false)?.options.some(
      (option) => option.label === "'HC3'",
    ),
  );
  assert.equal(complete("import openecon as oe\n", false), null);
  assert.equal(complete("unknown(", false), null);
});

test("literal choices remain specific to the published estimator contract", () => {
  const labels = (model: string) =>
    complete(`import openecon as oe\noe.${model}(covariance=`, false)
      ?.options.filter((option) => option.type === "constant")
      .map((option) => option.label);
  assert.ok(labels("ols")?.includes("'HC2'"));
  assert.ok(labels("ols")?.includes("'hac'"));
  assert.ok(labels("logit")?.includes("'cluster'"));
  assert.ok(!labels("logit")?.includes("'HC2'"));
  assert.ok(!labels("probit")?.includes("'hac'"));
  assert.ok(labels("ols")?.includes("None"));
});

test("defaults provide inert scalar values and both boolean choices", () => {
  const code =
    "def configure(rows=5, *, enabled=True, mode='quiet', unsafe=do_not_call()):\n    pass\n";
  const values = (name: string) =>
    complete(`${code}configure(${name}=`, false)
      ?.options.filter((option) => option.type === "constant")
      .map((option) => option.label);
  assert.deepEqual(values("rows"), ["5"]);
  assert.deepEqual(values("enabled"), ["True", "False"]);
  assert.deepEqual(values("mode"), ["'quiet'"]);
  assert.deepEqual(values("unsafe"), []);
  assert.equal(
    complete("import openecon as oe\noe.ols(covariance=", false, true),
    null,
  );
});

test("quoted value completion preserves delimiters and replaces a whole edited value", () => {
  const prefix = "import openecon as oe\n";
  assert.equal(
    acceptedValue(`${prefix}oe.ols(covariance='H|`, "HC3"),
    `${prefix}oe.ols(covariance='HC3'`,
  );
  assert.equal(
    acceptedValue(`${prefix}oe.ols(covariance="H|C1", data=df)`, "HC3"),
    `${prefix}oe.ols(covariance="HC3", data=df)`,
  );
  assert.equal(
    acceptedValue(`${prefix}oe.ols(covariance='H|C1')`, "HC3"),
    `${prefix}oe.ols(covariance='HC3')`,
  );
});

test("protected and unsupported string expressions never receive value suggestions", () => {
  for (const code of [
    'import openecon as oe\n"H|"',
    'import openecon as oe\n# oe.ols(covariance="H|',
    'import openecon as oe\noe.ols(covariance=f"H|C1")',
    'import openecon as oe\noe.ols(covariance="""H|C1""")',
    'import openecon as oe\noe.ols(covariance="H|" "C1")',
    'import openecon as oe\noe.ols(covariance=unknown("H|"))',
    'import openecon as oe\noe.ols(covariance=["H|"])',
  ])
    assert.equal(markedCompletion(code, true).result, null, code);
});

test("unused keywords include later supplied arguments and required parameters rank first", () => {
  const options = markedCompletion(
    "def f(required, optional=1, *, robust=True):\n    pass\nf(|, robust=True)",
  ).result?.options;
  assert.ok(!options?.some((option) => option.label === "robust="));
  assert.ok(
    (options?.find((option) => option.label === "required=")?.boost ?? 0) >
      (options?.find((option) => option.label === "optional=")?.boost ?? 0),
  );
  assert.equal(
    acceptedValue(
      "def f(first, *, robust=True):\n    pass\nf(ro|bust=True)",
      "robust=",
    ),
    "def f(first, *, robust=True):\n    pass\nf(robust=True)",
  );
});

test("future locals suppress outer namespace and builtin completion claims", () => {
  for (const name of ["oe", "len"]) {
    const options = markedCompletion(
      `import openecon as oe\ndef f():\n    ${name}|\n    ${name}=object()`,
      true,
    ).result?.options;
    assert.ok(!options?.some((option) => option.label === name), name);
  }
  assert.equal(
    markedCompletion(
      "import openecon as oe\ndef f():\n    df=oe.example()\n    oe=object()\n    df.h|",
      true,
    ).result,
    null,
  );
});

test("inferred instance aliases are local values and outrank generic builtins", () => {
  const options = complete(
    "import openecon as oe\ndf=oe.example()\ncopy=df\noe.ols(data=",
    false,
  )!.options;
  for (const name of ["df", "copy"]) {
    const option = options.find((item) => item.label === name);
    assert.equal(option?.type, "variable");
    assert.equal(option?.detail, "DataFrame");
    assert.ok(
      (option?.boost ?? 0) >
        (options.find((item) => item.label === "len")?.boost ?? 0),
    );
  }
});

test("unsupported chained receivers receive no unrelated global suggestions", () => {
  assert.equal(
    complete("import openecon as oe\ndf=oe.example()\ndf.head().", true),
    null,
  );
  assert.equal(
    acceptedValue("import openecon as oe\noe.o|ls", "ols"),
    "import openecon as oe\noe.ols",
  );
});

test("numeric and named literal completion replaces the complete edited token", () => {
  assert.equal(
    acceptedValue("import openecon as oe\noe.ols(alpha=0.|01)", "0.05"),
    "import openecon as oe\noe.ols(alpha=0.05)",
  );
  assert.equal(
    acceptedValue("import openecon as oe\noe.ols(covariance=N|one)", "None"),
    "import openecon as oe\noe.ols(covariance=None)",
  );
  assert.equal(
    acceptedValue("def f(x=-1):\n    pass\nf(x=-|)", "-1"),
    "def f(x=-1):\n    pass\nf(x=-1)",
  );
  assert.equal(
    acceptedValue("def f(x=True):\n    pass\nf(x=Tr|ue)", "True"),
    "def f(x=True):\n    pass\nf(x=True)",
  );
  for (const code of [
    'import openecon as oe\noe.ols(covariance=|"HC1")',
    "def f(x=True):\n    pass\nf(x=|True)",
    "def f(x=100):\n    pass\nf(x=|100)",
  ])
    assert.equal(markedCompletion(code).result, null, code);
});

test("MRQAP, SBM and snapshot objects offer their actual chained methods", () => {
  const prelude = "import openecon as oe\ngraph=oe.network(data=edges)\n";
  assert.equal(symbol(prelude + "fit=graph.block_model(2)\nfit.summary()", "summary")?.entry.name,
    "openecon.NetworkBlockResult.summary");
  assert.equal(symbol(prelude + "table=graph.qap_regression({'a':graph})\ntable.to_latex()", "to_latex")?.entry.name,
    "openecon.DataFrame.to_latex");
  const layers = prelude + "layers=oe.network_snapshots({'first':graph},ordered=True)\n";
  const methods = complete(layers + "layers.")?.options;
  assert.ok(methods?.some(item => item.label === "temporal_path"));
  assert.ok(methods?.some(item => item.label === "edge_persistence"));
  assert.ok(!methods?.some(item => item.label === "block_model"));
  assert.equal(symbol(layers + "combined=layers.aggregate()\ncombined.degree()", "degree")?.entry.name,
    "openecon.Network.degree");
  assert.equal(symbol(layers + "rows=layers.transitions()\nrows.to_latex()", "to_latex")?.entry.name,
    "openecon.DataFrame.to_latex");
});

test("signed graph results resolve to signed methods and complete table types", () => {
  const base = "import openecon as oe\ng = oe.signed_network(rows)\n";
  const options = complete(base + "g.signed_")?.options.map((item) => item.label) ?? [];
  for (const method of ["signed_strength", "signed_katz", "signed_modularity", "signed_communities", "signed_shortest_paths"]) {
    assert.ok(options.includes(method));
    assert.equal(symbol(base + `g.${method}()`, method)?.entry.name, `openecon.SignedNetwork.${method}`);
  }
  assert.ok(!complete(base + "g.")?.options.some((item) => item.label === "pagerank"));
  assert.equal(symbol(base + "out = g.signed_strength()\nout.head()", "head")?.entry.name, "openecon.DataFrame.head");
});

test("dynamic help selects intervals before explicit simple projection", () => {
  const base = "import openecon as oe\ng=oe.dynamic_network(nodes,edges)\n";
  const options = complete(base + "g.")?.options.map((item) => item.label) ?? [];
  for (const method of ["nodes", "edges", "summary", "at", "window", "write"])
    assert.ok(options.includes(method));
  assert.ok(!options.includes("pagerank"));
  assert.equal(symbol(base + "point=g.at(1)\npoint.degree()", "degree")?.entry.name,
    "openecon.MultiNetwork.degree");
  assert.equal(symbol(base + "point=g.window(0,1)\nsimple=point.to_network(reducer='count')\nsimple.pagerank()", "pagerank")?.entry.name,
    "openecon.Network.pagerank");
  assert.equal(symbol("import openecon as oe\ng=oe.read_dynamic_network('owned.gexf')\nt=g.edges()\nt.to_latex()", "to_latex")?.entry.name,
    "openecon.DataFrame.to_latex");
});

test("multigraph help retains edge IDs until an explicit projection", () => {
  const base = "import openecon as oe\ng = oe.multigraph(rows)\n";
  const options = complete(base + "g.")?.options.map((item) => item.label) ?? [];
  for (const method of ["edges", "degree", "filter", "edit_edges", "to_network", "write"])
    assert.ok(options.includes(method));
  assert.ok(!options.includes("pagerank"));
  assert.equal(symbol(base + "edited=g.edit_edges(weights={1:9})\nchanged=edited.filter(edge_ids=[1])\nchanged.degree()", "degree")?.entry.name,
    "openecon.MultiNetwork.degree");
  assert.equal(symbol(base + "simple=g.to_network(reducer='count')\nsimple.pagerank()", "pagerank")?.entry.name,
    "openecon.Network.pagerank");
  assert.equal(symbol("import openecon as oe\ng=oe.read_multigraph('owned.graphml')\nt=g.edges()\nt.to_latex()", "to_latex")?.entry.name,
    "openecon.DataFrame.to_latex");
});

test("compact literal defaults preserve exact and descriptive signature help", () => {
  const call = getActiveCall(state("import openecon as oe\noe.firth_logit(data, 'y', level="));
  assert.equal(call?.entry.parameters.find((parameter) => parameter.name === "level")?.default, "0.95");
  assert.equal(call?.entry.parameters.find((parameter) => parameter.name === "device")?.default, "'cpu'");
  const exact = getActiveCall(state("import openecon as oe\noe.exact_poisson_rate(84, 36, null_rate="));
  assert.equal(exact?.entry.parameters.find((parameter) => parameter.name === "null_rate")?.default, "1.0");
  assert.equal(Object.hasOwn(exact?.entry.parameters[0] ?? {}, "default"), false);
  const trimmed = getActiveCall(state("import openecon as oe\noe.lts(data, 'y', h="));
  assert.equal(trimmed?.entry.parameters.find((parameter) => parameter.name === "h")?.default, "None");
});

const confidenceTemplates = JSON.parse(readFileSync(new URL("./fixtures/integration6-parameter-templates.json", import.meta.url), "utf8"));
assert.deepEqual(confidenceTemplates.map(({ token }: { token: number }) => token), [75, 76, 77, 78, 79]);
for (const { token, parameters } of confidenceTemplates) {
  assert.deepEqual(COMPACT_PARAMETER_LISTS[token], parameters);
}
