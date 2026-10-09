import entries from "./editor-api.json" with { type: "json" };

export interface ApiParameter {
  name: string;
  annotation?: string;
  description?: string;
  kind?: string;
  default?: string;
  // Valid Python literals read from the published parameter contracts.
  choices?: string[];
}

export interface ApiEntry {
  name: string;
  signature: string;
  description: string;
  parameters: ApiParameter[];
  returns?: string;
  kind: "function" | "method" | "class";
  owner?: string;
}

type StoredApiParameter = Omit<ApiParameter, "name" | "description" | "annotation" | "default" | "choices" | "kind"> & {
  name?: string;
  description?: string;
  annotation?: string;
  default?: string;
  choices?: string[];
  kind?: string;
  n?: string | number;
  d?: string;
  a?: string;
  // JSON singleton and safe integer markers restore exact Python literals.
  v?: string | number | boolean | null;
  q?: string;
  // Exact literal Python default None; previous null markers remain readable.
  V?: number;
  // Exact literal Python Boolean defaults.
  B?: number;
  // Index into the fixed Python literal dictionary.
  D?: number;
  // Legacy literal None marker remains readable.
  N?: 1;
  c?: string[];
  k?: string;
  t?: string;
};

export type StoredApiEntry = Omit<ApiEntry,
  "kind" | "name" | "signature" | "description" | "parameters" | "returns"
> & {
  kind?: ApiEntry["kind"] | "m" | "c";
  k?: ApiEntry["kind"] | "m" | "c";
  name?: string;
  signature?: string;
  description?: string;
  parameters?: StoredApiParameter[];
  P?: number;
  returns?: string;
  n?: string;
  s?: string;
  // Call signature suffix; restore the canonical final name component.
  S?: string;
  // Exact shared call suffix; explicit and legacy signature fields win.
  I?: number;
  d?: string;
  p?: StoredApiParameter[];
  r?: string;
  // Canonical openecon. return namespace; older r may use the '~' prefix.
  R?: string;
  // Exact common canonical return type; older full and namespace fields win.
  T?: number;
};

// Generated from committed openecon APIs and the packaged Python environment.
// Static data keeps editor assistance independent of code execution and login.
const STORED_DEFAULT_LITERALS = ["'raise'", "0.05", "'cpu'", "100000000", "50000000", "'drop'", "0.95", "0", "DEFAULT_BYTES", "DEFAULT_WORK", "'two-sided'", "'nonrobust'", "'constant'", "1000000", "2000000000", "100000"];
const STORED_PARAMETER_KINDS: Record<string, string> = {
  k: "keyword-only", p: "positional-only",
  "*": "var-positional", "**": "var-keyword",
};
const STORED_RETURN_TYPES = ["openecon.TableSet", "openecon.ResultBundle", "openecon.DataFrame"];
// Append only; literal S remains the fallback for different installed signatures.
const STORED_SIGNATURE_SUFFIXES = [
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
];
// Protocol order is fixed; changing it would reinterpret saved catalogs.
const STORED_PARAMETER_LISTS: StoredApiParameter[][] = [[{"name":"json_data","kind":"positional-or-keyword"},{"name":"strict","kind":"keyword-only","default":"None"},{"name":"extra","kind":"keyword-only","default":"None"},{"name":"context","kind":"keyword-only","default":"None"},{"name":"by_alias","kind":"keyword-only","default":"None"},{"name":"by_name","kind":"keyword-only","default":"None"}],[{"name":"indent","kind":"keyword-only","default":"None"},{"name":"ensure_ascii","kind":"keyword-only","default":"False"},{"name":"include","kind":"keyword-only","default":"None"},{"name":"exclude","kind":"keyword-only","default":"None"},{"name":"context","kind":"keyword-only","default":"None"},{"name":"by_alias","kind":"keyword-only","default":"None"},{"name":"exclude_unset","kind":"keyword-only","default":"False"},{"name":"exclude_defaults","kind":"keyword-only","default":"False"},{"name":"exclude_none","kind":"keyword-only","default":"False"},{"name":"exclude_computed_fields","kind":"keyword-only","default":"False"},{"name":"round_trip","kind":"keyword-only","default":"False"},{"name":"warnings","kind":"keyword-only","default":"True"},{"name":"fallback","kind":"keyword-only","default":"None"},{"name":"serialize_as_any","kind":"keyword-only","default":"False"},{"name":"polymorphic_serialization","kind":"keyword-only","default":"None"}],[{"name":"obj","kind":"positional-or-keyword"},{"name":"strict","kind":"keyword-only","default":"None"},{"name":"extra","kind":"keyword-only","default":"None"},{"name":"from_attributes","kind":"keyword-only","default":"None"},{"name":"context","kind":"keyword-only","default":"None"},{"name":"by_alias","kind":"keyword-only","default":"None"},{"name":"by_name","kind":"keyword-only","default":"None"}],[{"kind":"keyword-only","name":"data"},{"kind":"keyword-only","name":"y"},{"kind":"keyword-only","name":"endogenous"},{"kind":"keyword-only","name":"instruments"},{"default":"None","kind":"keyword-only","name":"x"},{"default":"'robust'","kind":"keyword-only","name":"covariance"},{"default":"None","kind":"keyword-only","name":"cluster"},{"default":"True","kind":"keyword-only","name":"intercept"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"default":"100","kind":"keyword-only","name":"max_iterations"},{"default":"1e-09","kind":"keyword-only","name":"tolerance"},{"default":"10000000000","kind":"keyword-only","name":"max_work"},{"default":"'cpu'","kind":"keyword-only","name":"device"}],[{"kind":"positional-or-keyword","name":"data"},{"kind":"positional-or-keyword","name":"design"},{"kind":"positional-or-keyword","name":"outcome"},{"kind":"positional-or-keyword","name":"regressors"},{"default":"True","kind":"keyword-only","name":"intercept"},{"default":"None","kind":"keyword-only","name":"domain"},{"choices":["'drop'","'raise'"],"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"default":"0.0","kind":"keyword-only","name":"null"},{"default":"100","kind":"keyword-only","name":"max_iter"},{"default":"1e-09","kind":"keyword-only","name":"tolerance"}],[{"kind":"positional-or-keyword","name":"data"},{"kind":"positional-or-keyword","name":"design"},{"kind":"positional-or-keyword","name":"outcome"},{"kind":"positional-or-keyword","name":"regressors"},{"kind":"keyword-only","name":"method"},{"default":"True","kind":"keyword-only","name":"intercept"},{"default":"None","kind":"keyword-only","name":"domain"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"default":"0.0","kind":"keyword-only","name":"null"},{"default":"100","kind":"keyword-only","name":"max_iter"},{"default":"1e-09","kind":"keyword-only","name":"tolerance"},{"default":"None","kind":"keyword-only","name":"replicates"},{"default":"None","kind":"keyword-only","name":"replicate_weights"},{"default":"'original'","kind":"keyword-only","name":"centering"},{"default":"None","kind":"keyword-only","name":"rho"},{"default":"None","kind":"keyword-only","name":"justification"},{"default":"None","kind":"keyword-only","name":"scale"},{"default":"None","kind":"keyword-only","name":"rscales"},{"default":"None","kind":"keyword-only","name":"df"}],[{"kind":"positional-or-keyword","name":"low"},{"kind":"positional-or-keyword","name":"indicator"},{"kind":"keyword-only","name":"low_periods"},{"kind":"keyword-only","name":"high_periods"},{"default":"'Y'","kind":"keyword-only","name":"low_frequency"},{"default":"'Q'","kind":"keyword-only","name":"high_frequency"},{"default":"'sum'","kind":"keyword-only","name":"aggregation"},{"default":"None","kind":"keyword-only","name":"as_of"},{"default":"None","kind":"keyword-only","name":"low_releases"},{"default":"None","kind":"keyword-only","name":"high_releases"},{"default":"'cpu'","kind":"keyword-only","name":"device"},{"default":"None","kind":"keyword-only","name":"weights"}],[{"kind":"positional-or-keyword","name":"data"},{"kind":"positional-or-keyword","name":"design"},{"kind":"positional-or-keyword","name":"outcome"},{"kind":"positional-or-keyword","name":"regressors"},{"default":"True","kind":"keyword-only","name":"intercept"},{"default":"None","kind":"keyword-only","name":"domain"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"default":"0.0","kind":"keyword-only","name":"null"},{"default":"100","kind":"keyword-only","name":"max_iter"},{"default":"1e-09","kind":"keyword-only","name":"tolerance"}],[{"kind":"keyword-only","name":"data"},{"kind":"keyword-only","name":"items"},{"default":"41","kind":"keyword-only","name":"points"},{"default":"200","kind":"keyword-only","name":"max_iter"},{"default":"400","kind":"keyword-only","name":"max_eval"},{"default":"1e-06","kind":"keyword-only","name":"tolerance"},{"default":"0.95","kind":"keyword-only","name":"level"}],[{"kind":"positional-or-keyword","name":"data"},{"kind":"positional-or-keyword","name":"design"},{"kind":"positional-or-keyword","name":"outcomes"},{"default":"None","kind":"keyword-only","name":"domain"},{"choices":["'drop'","'raise'"],"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"default":"0.0","kind":"keyword-only","name":"null"}],[{"kind":"keyword-only","name":"data"},{"kind":"keyword-only","name":"y"},{"kind":"keyword-only","name":"x"},{"kind":"keyword-only","name":"panel"},{"kind":"keyword-only","name":"time"},{"default":"1","kind":"keyword-only","name":"p"},{"default":"1","kind":"keyword-only","name":"q"},{"default":"'c'","kind":"keyword-only","name":"trend"},{"default":"500","kind":"keyword-only","name":"max_iterations"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"}],[{"kind":"positional-or-keyword","name":"lower"},{"kind":"positional-or-keyword","name":"upper"},{"default":"None","kind":"keyword-only","name":"times"},{"default":"0.95","kind":"keyword-only","name":"level"},{"default":"'cpu'","kind":"keyword-only","name":"device"},{"default":"None","kind":"keyword-only","name":"weights"},{"default":"500","kind":"keyword-only","name":"maxiter"}],[{"kind":"positional-or-keyword","name":"result"},{"kind":"positional-or-keyword","name":"data"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"DEFAULT_WORK","kind":"keyword-only","name":"max_work"},{"default":"DEFAULT_BYTES","kind":"keyword-only","name":"max_bytes"},{"default":"'cpu'","kind":"keyword-only","name":"device"}],[{"kind":"keyword-only","name":"data"},{"kind":"keyword-only","name":"y"},{"kind":"keyword-only","name":"x"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"kind":"var-keyword","name":"options"}],[{"kind":"positional-or-keyword","name":"result"},{"kind":"keyword-only","name":"data"},{"kind":"keyword-only","name":"variable"},{"default":"1","kind":"keyword-only","name":"order"},{"default":"False","kind":"keyword-only","name":"interval"},{"default":"0.05","kind":"keyword-only","name":"alpha"},{"default":"'raise'","kind":"keyword-only","name":"missing"},{"default":"2000000000","kind":"keyword-only","name":"max_work"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"'aweight'"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"l1_ratio","default":"0.5","kind":"keyword-only"},{"name":"selection","kind":"keyword-only","default":"'cv'"},{"name":"penalty","kind":"keyword-only","default":"None"},{"name":"lambda_path","kind":"keyword-only","default":"None"},{"name":"n_lambdas","default":"20","kind":"keyword-only"},{"name":"lambda_ratio","default":"0.001","kind":"keyword-only"},{"name":"folds","default":"5","kind":"keyword-only"},{"name":"seed","default":"1729","kind":"keyword-only"},{"name":"standardize","kind":"keyword-only","default":"True"},{"name":"penalty_factors","kind":"keyword-only","default":"None"},{"name":"forced_controls","kind":"keyword-only","default":"None"},{"name":"max_iterations","default":"200","kind":"keyword-only"},{"name":"tolerance","default":"1e-08","kind":"keyword-only"},{"name":"max_work","kind":"keyword-only","default":"2000000000"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"panel","kind":"keyword-only"},{"name":"time","kind":"keyword-only","default":"None"},{"name":"model","kind":"keyword-only","default":"'re'"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"intpoints","kind":"keyword-only","default":"None"},{"name":"intmethod","kind":"keyword-only","default":"None"},{"name":"corr","kind":"keyword-only","default":"None"},{"name":"corr_order","kind":"keyword-only","default":"None"},{"name":"force","kind":"keyword-only","default":"False"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"low","kind":"positional-or-keyword"},{"name":"indicator","kind":"positional-or-keyword"},{"name":"rho","kind":"keyword-only"},{"name":"low_periods","kind":"keyword-only"},{"name":"high_periods","kind":"keyword-only"},{"name":"low_frequency","kind":"keyword-only","default":"'Y'"},{"name":"high_frequency","kind":"keyword-only","default":"'Q'"},{"name":"aggregation","kind":"keyword-only","default":"'sum'"},{"name":"as_of","kind":"keyword-only","default":"None"},{"name":"low_releases","kind":"keyword-only","default":"None"},{"name":"high_releases","kind":"keyword-only","default":"None"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"design","kind":"positional-or-keyword"},{"name":"outcome","kind":"positional-or-keyword"},{"name":"categories","kind":"keyword-only","default":"None"},{"name":"domain","kind":"keyword-only","default":"None"},{"name":"missing","choices":["'drop'","'raise'"],"kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"null","default":"0.0","kind":"keyword-only"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"design","kind":"positional-or-keyword"},{"name":"numerators","kind":"positional-or-keyword"},{"name":"denominators","kind":"positional-or-keyword"},{"name":"domain","kind":"keyword-only","default":"None"},{"name":"missing","choices":["'drop'","'raise'"],"kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"null","default":"0.0","kind":"keyword-only"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"inflate","kind":"keyword-only"},{"name":"inflate_link","kind":"keyword-only","default":"'logit'"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"None"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"exposure","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"dimensions","kind":"positional-or-keyword"},{"name":"count","kind":"positional-or-keyword"},{"name":"levels","kind":"keyword-only"},{"name":"design","kind":"keyword-only"},{"name":"terms","kind":"keyword-only","default":"None"},{"name":"structural","kind":"keyword-only","default":"None"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"max_iter","default":"200","kind":"keyword-only"},{"name":"tol","default":"1e-09","kind":"keyword-only"},{"name":"level","kind":"keyword-only","default":"0.95"},{"name":"max_work","default":"300000000","kind":"keyword-only"},{"name":"max_bytes","default":"128 * 1024 ** 2","kind":"keyword-only"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only","default":"None"},{"name":"group","kind":"keyword-only"},{"name":"random","kind":"keyword-only","default":"None"},{"name":"intpoints","default":"7","kind":"keyword-only"},{"name":"intmethod","kind":"keyword-only","default":"'mvaghermite'"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"ll","kind":"keyword-only","default":"None"},{"name":"ul","kind":"keyword-only","default":"None"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"None"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"title","kind":"keyword-only","default":"None"},{"name":"unit","kind":"keyword-only","default":"''"},{"name":"palette","kind":"keyword-only","default":"None"},{"name":"aggregate","kind":"keyword-only","default":"None"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"raters","kind":"positional-or-keyword"},{"name":"categories","kind":"keyword-only"},{"name":"agreement_weights","kind":"keyword-only","default":"'unweighted'"},{"name":"inference","kind":"keyword-only","default":"'none'"},{"name":"level","kind":"keyword-only","default":"0.95"},{"name":"missing","kind":"keyword-only","default":"'drop'"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"max_fits","default":"1024","kind":"keyword-only"},{"name":"max_work","kind":"keyword-only","default":"100000000"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"data","kind":"keyword-only"},{"name":"variable","kind":"keyword-only"},{"name":"order","default":"1","kind":"keyword-only"},{"name":"interval","kind":"keyword-only","default":"True"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"max_work","kind":"keyword-only","default":"2000000000"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"time","kind":"keyword-only","default":"None"},{"name":"trend","kind":"keyword-only","default":"'c'"},{"name":"kernel","kind":"keyword-only","default":"'bartlett'"},{"name":"bandwidth","default":"4","kind":"keyword-only"},{"name":"df_adjust","kind":"keyword-only","default":"False"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"covariance","choices":["'cluster'","'nonrobust'"],"kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"missing","choices":["'raise'","'drop'"],"kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"covariance","kind":"keyword-only","default":"None"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"weight_type","kind":"keyword-only","default":"None"},{"name":"offset","kind":"keyword-only","default":"None"},{"name":"categorical","kind":"keyword-only","default":"None"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"kind","choices":["'linear'","'response'"],"kind":"keyword-only","default":"'response'"},{"name":"missing","choices":["'drop'","'raise'"],"kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"None"}],[{"name":"objective","kind":"keyword-only","default":"'weight'"},{"name":"max_component_nodes","default":"24","kind":"keyword-only"},{"name":"max_states","kind":"keyword-only","default":"1000000"},{"name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"treatment","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"var-keyword"}],[{"name":"partition","default":"None","kind":"positional-or-keyword"},{"name":"objective","kind":"keyword-only","default":"'weight'"},{"name":"max_matrix_entries","kind":"keyword-only","default":"1000000"},{"name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"design","kind":"positional-or-keyword"},{"name":"outcomes","kind":"positional-or-keyword"},{"name":"domain","kind":"keyword-only","default":"None"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"null","default":"0.0","kind":"keyword-only"},{"name":"deff","kind":"keyword-only","default":"False"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"outcome","kind":"positional-or-keyword"},{"name":"predictors","kind":"positional-or-keyword"},{"name":"key","kind":"keyword-only"},{"name":"spatial_weights","kind":"keyword-only"},{"name":"options","kind":"var-keyword"}],[{"name":"path","kind":"positional-or-keyword"},{"name":"format","kind":"keyword-only","default":"None"},{"name":"max_memory_mb","default":"256","kind":"keyword-only"},{"name":"batch_rows","default":"65536","kind":"keyword-only"},{"name":"max_file_mb","default":"512","kind":"keyword-only"}],[{"description":"A DataFrame, column dictionary, row records or batch Dataset source.","name":"data","kind":"keyword-only"},{"description":"Outcome column name; cannot be combined with formula.","name":"y","kind":"keyword-only","default":"None"},{"description":"Predictor column names; cannot be combined with formula.","name":"x","kind":"keyword-only","default":"None"},{"description":"Example: 'wage ~ education + experience + C(region)'.","name":"formula","kind":"keyword-only","default":"None"},{"description":"Standard error method, such as 'nonrobust', 'HC3', 'cluster' or 'hac'.","name":"covariance","choices":["'nonrobust'","'HC0'","'HC1'","'HC2'","'HC3'","'cluster'","'cluster_hc2'","'cluster_hc3'","'hac'","'bootstrap'","'jackknife'"],"kind":"keyword-only","default":"None"},{"description":"Predictor names to encode as categorical variables.","name":"categorical","kind":"keyword-only","default":"None"},{"description":"Include an intercept.","name":"intercept","kind":"keyword-only","default":"True"},{"description":"Column name or names for clustered standard errors.","name":"cluster","kind":"keyword-only","default":"None"},{"description":"Column name containing the weights.","name":"weights","kind":"keyword-only","default":"None"},{"description":"'aweight', 'fweight', 'pweight' or 'iweight'.","name":"weight_type","choices":["'aweight'","'fweight'","'pweight'","'iweight'"],"kind":"keyword-only","default":"None"},{"description":"'drop' removes observations with missing values; 'raise' reports an error.","name":"missing","choices":["'raise'","'drop'"],"kind":"keyword-only","default":"'drop'"},{"description":"Inference significance level; 0.05 gives a 95% confidence interval.","name":"alpha","kind":"keyword-only","default":"0.05"},{"description":"Time column used for HAC or lagged formulas.","name":"time","kind":"keyword-only","default":"None"},{"description":"HAC lag count or a supported automatic selection name.","name":"lags","kind":"keyword-only","default":"None"},{"description":"HAC weighting kernel.","name":"kernel","choices":["'bartlett'","'parzen'","'quadratic_spectral'","'truncated'"],"kind":"keyword-only","default":"None"},{"description":"Replication option for bootstrap or jackknife.","name":"reps","kind":"keyword-only","default":"None"},{"description":"Random seed for resampling.","name":"seed","kind":"keyword-only","default":"None"},{"name":"dfadjust","kind":"keyword-only","default":"False"},{"name":"hansen","kind":"keyword-only","default":"False"},{"description":"'auto', 'cpu', 'cuda', 'cuda:<index>' or 'mps'. CUDA uses float64; Metal preconditions bounded factors with checked CPU float64 refinement.","name":"device","kind":"keyword-only","default":"'auto'"}],[{"name":"data","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"endogenous","kind":"keyword-only"},{"name":"instruments","kind":"keyword-only"},{"name":"x","kind":"keyword-only","default":"None"},{"name":"covariance","kind":"keyword-only","default":"'robust'"},{"name":"cluster","kind":"keyword-only","default":"None"},{"name":"intercept","kind":"keyword-only","default":"True"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"max_iterations","default":"100","kind":"keyword-only"},{"name":"tolerance","default":"1e-09","kind":"keyword-only"},{"name":"max_work","default":"10000000000","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"batch_rows","kind":"keyword-only","default":"None"}],[{"name":"coefficients","kind":"positional-or-keyword"},{"name":"null","default":"0.0","kind":"keyword-only"},{"name":"alpha","kind":"keyword-only","default":"None"}],[{"name":"time","kind":"positional-or-keyword"},{"name":"event","kind":"positional-or-keyword"},{"name":"causes","kind":"keyword-only","default":"None"},{"name":"times","kind":"keyword-only","default":"None"},{"name":"level","kind":"keyword-only","default":"0.95"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"weights","kind":"keyword-only","default":"None"}],[{"name":"nodes","kind":"keyword-only","default":"None"},{"name":"edges","kind":"keyword-only","default":"None"},{"name":"graph_attributes","kind":"keyword-only","default":"None"}],[{"name":"coefficients","kind":"positional-or-keyword"},{"name":"null","default":"0.0","kind":"keyword-only"}],[{"name":"add","default":"()","kind":"keyword-only"},{"name":"remove","default":"()","kind":"keyword-only"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"attributes","kind":"keyword-only","default":"None"}],[{"name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"columns","kind":"positional-or-keyword"},{"name":"bounds","kind":"keyword-only"},{"name":"sampling_model","kind":"keyword-only"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"missing","kind":"keyword-only","default":"'raise'"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"y","kind":"positional-or-keyword"},{"description":"``n_dropped``) or \"raise\".","name":"missing","kind":"keyword-only","default":"'drop'"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"data","kind":"keyword-only"},{"name":"alignment","kind":"keyword-only","default":"None"},{"name":"alpha","kind":"keyword-only","default":"None"},{"name":"kind","kind":"keyword-only","default":"'mean'"}],[{"name":"path","kind":"positional-or-keyword"},{"name":"overwrite","kind":"keyword-only","default":"False"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"args","kind":"var-positional"},{"name":"kwargs","kind":"var-keyword"}],[{"name":"buf","default":"None","kind":"positional-or-keyword"},{"name":"kwargs","kind":"var-keyword"}],[{"name":"direction","kind":"keyword-only","default":"'out'"},{"name":"disconnected","kind":"keyword-only","default":"'infinite'"},{"name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"device","kind":"keyword-only","default":"'cpu'"},{"name":"weights","kind":"keyword-only","default":"None"},{"name":"max_work","kind":"keyword-only","default":"DEFAULT_WORK"},{"name":"max_bytes","kind":"keyword-only","default":"DEFAULT_BYTES"}],[{"name":"data","kind":"keyword-only"},{"name":"x","kind":"keyword-only"},{"name":"y","kind":"keyword-only"},{"name":"title","kind":"keyword-only","default":"None"},{"name":"options","kind":"var-keyword"}],[{"name":"attributes","kind":"keyword-only","default":"True"}],[{"name":"restrictions","kind":"positional-or-keyword"},{"name":"null","kind":"keyword-only","default":"None"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"columns","kind":"positional-or-keyword"},{"name":"sampling_model","kind":"keyword-only"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"name":"missing","kind":"keyword-only","default":"'raise'"}],[{"name":"add","default":"()","kind":"keyword-only"},{"name":"remove","default":"()","kind":"keyword-only"},{"name":"rename","kind":"keyword-only","default":"None"}],[{"name":"max_nodes","default":"1000","kind":"positional-or-keyword"},{"name":"max_edges","default":"5000","kind":"positional-or-keyword"},{"name":"seed","default":"0","kind":"positional-or-keyword"},{"name":"groups","default":"None","kind":"positional-or-keyword"}],[{"name":"options","kind":"var-keyword"}],[{"name":"source","kind":"positional-or-keyword"},{"name":"target","kind":"positional-or-keyword"},{"name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"path","kind":"positional-or-keyword"},{"name":"format","kind":"keyword-only","default":"None"},{"name":"overwrite","kind":"keyword-only","default":"False"}],[{"name":"restricted","kind":"positional-or-keyword"},{"name":"full","kind":"positional-or-keyword"},{"name":"max_bytes","default":"128 * 1024 ** 2","kind":"keyword-only"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"columns","kind":"positional-or-keyword"},{"name":"max_iterations","default":"500","kind":"keyword-only"},{"name":"tolerance","default":"1e-08","kind":"keyword-only"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"path","default":"None","kind":"positional-or-keyword"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"coefficient","kind":"keyword-only","default":"None"},{"name":"max_work","kind":"keyword-only","default":"100000000"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"level","kind":"keyword-only","default":"None"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"null","default":"0.0","kind":"keyword-only"},{"name":"max_work","kind":"keyword-only","default":"100000000"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"series","kind":"positional-or-keyword"},{"name":"options","kind":"var-keyword"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"variables","kind":"positional-or-keyword"},{"name":"knots","kind":"keyword-only"},{"name":"components","default":"2","kind":"keyword-only"},{"name":"n_starts","default":"3","kind":"keyword-only"},{"name":"seed","kind":"keyword-only","default":"0"},{"name":"maxiter","default":"500","kind":"keyword-only"},{"name":"tol","default":"1e-08","kind":"keyword-only"},{"name":"missing","kind":"keyword-only","default":"'drop'"},{"name":"max_work","default":"WORK","kind":"keyword-only"},{"name":"max_bytes","default":"BYTES","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"outcome","kind":"positional-or-keyword"},{"name":"predictors","kind":"positional-or-keyword"},{"name":"knots","kind":"keyword-only"},{"name":"missing","kind":"keyword-only","default":"'drop'"},{"name":"max_bytes","default":"c.LIMIT_BYTES","kind":"keyword-only"},{"name":"max_work","default":"c.WORK","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"name":"result","kind":"positional-or-keyword"},{"name":"data","kind":"positional-or-keyword"},{"name":"missing","kind":"keyword-only","default":"'raise'"},{"name":"max_work","default":"WORK","kind":"keyword-only"},{"name":"max_bytes","default":"BYTES","kind":"keyword-only"},{"name":"device","kind":"keyword-only","default":"'cpu'"}],[{"description":"Any registry fit except time-series models and fits that already use a resampling covariance.","name":"result","kind":"positional-or-keyword"},{"description":"Exactly the dataset ``result`` was fitted on (verified by hash).","name":"data","kind":"positional-or-keyword"},{"description":"Number of bootstrap replications R (at least 2; 200 by default, use 1000 or more for percentile/bc intervals).","name":"reps","default":"200","kind":"keyword-only"},{"description":"Seed (0 to 2**64 - 1) of the ``torch.Generator`` that draws every replicate; the same seed reproduces the result exactly. A random seed is drawn and recorded when omitted.","name":"seed","kind":"keyword-only","default":"None"},{"description":"Columns defining resampling clusters and strata.","name":"cluster","kind":"keyword-only","default":"None"},{"description":"Columns defining resampling clusters and strata.","name":"strata","kind":"keyword-only","default":"None"},{"description":"Units drawn per stratum.","name":"size","kind":"keyword-only","default":"None"},{"description":"Level of the intervals (default: the fit's alpha).","name":"alpha","kind":"keyword-only","default":"None"},{"description":"Interval type stored in ``extra['bootstrap_ci']``: normal based, percentile (Stata's ``_pctile`` definition) or bias-corrected percentile (z0 = Phi^-1(#{b_r <= b}/R)), BCa delete-unit acceleration or bootstrap-t using each refit's standard error. Advanced intervals currently require unweighted, unst","name":"ci","kind":"keyword-only","default":"'normal'"},{"description":"Dependent schemes validate explicit-column OLS. Blocks require a regular time column and block_length; wild_cluster requires cluster=.","name":"scheme","kind":"keyword-only","default":"'iid'"},{"name":"block_length","kind":"keyword-only","default":"None"},{"name":"time","kind":"keyword-only","default":"None"},{"description":"Coefficient restrictions for restricted wild cluster-t tests; requires the original covariance clustered on the same column and ci='normal'. Confidence-set inversion is not supplied.","name":"null","kind":"keyword-only","default":"None"},{"description":"Wild cluster multiplier distribution.","name":"wild","kind":"keyword-only","default":"'rademacher'"},{"description":"Advanced draw plus BCa acceleration budget (default 10,000).","name":"max_refits","default":"10000","kind":"keyword-only"}],[{"description":"String, path object (implementing os.PathLike[str]), or file-like object implementing a write() function. If None, the result is returned as a string. If a non-binary file object is passed, it should be opened with `newline=''`, disabling universal newlines. If a binary file object is passed, `mode`","name":"path_or_buf","default":"None","kind":"positional-or-keyword"},{"description":"String of length 1. Field delimiter for the output file.","name":"sep","default":"','","kind":"positional-or-keyword"},{"description":"Missing data representation.","name":"na_rep","default":"''","kind":"positional-or-keyword"},{"description":"Format string for floating point numbers. If a Callable is given, it takes precedence over other numeric formatting parameters, like decimal.","name":"float_format","default":"None","kind":"positional-or-keyword"},{"description":"Columns to write.","name":"columns","default":"None","kind":"positional-or-keyword"},{"description":"Write out the column names. If a list of strings is given it is assumed to be aliases for the column names.","name":"header","default":"True","kind":"positional-or-keyword"},{"description":"Write row names (index).","name":"index","default":"True","kind":"positional-or-keyword"},{"description":"Column label for index column(s) if desired. If None is given, and `header` and `index` are True, then the index names are used. A sequence should be given if the object uses MultiIndex. If False do not print fields for index names. Use index_label=False for easier importing in R.","name":"index_label","default":"None","kind":"positional-or-keyword"},{"description":"Forwarded to either `open(mode=)` or `fsspec.open(mode=)` to control the file opening. Typical values include:","name":"mode","default":"'w'","kind":"positional-or-keyword"},{"description":"A string representing the encoding to use in the output file, defaults to 'utf-8'. `encoding` is not supported if `path_or_buf` is a non-binary file object.","name":"encoding","default":"None","kind":"positional-or-keyword"},{"name":"compression","default":"'infer'","kind":"positional-or-keyword"},{"name":"quoting","default":"None","kind":"positional-or-keyword"},{"name":"quotechar","default":"'\"'","kind":"positional-or-keyword"},{"name":"lineterminator","default":"None","kind":"positional-or-keyword"},{"name":"chunksize","default":"None","kind":"positional-or-keyword"},{"name":"date_format","default":"None","kind":"positional-or-keyword"},{"name":"doublequote","default":"True","kind":"positional-or-keyword"},{"name":"escapechar","default":"None","kind":"positional-or-keyword"},{"name":"decimal","default":"'.'","kind":"positional-or-keyword"},{"name":"errors","default":"'strict'","kind":"positional-or-keyword"},{"name":"storage_options","default":"None","kind":"positional-or-keyword"}],[{"description":"File path or existing ExcelWriter.","name":"excel_writer","kind":"positional-or-keyword"},{"description":"Name of sheet which will contain DataFrame.","name":"sheet_name","default":"'Sheet1'","kind":"positional-or-keyword"},{"description":"Missing data representation.","name":"na_rep","default":"''","kind":"positional-or-keyword"},{"description":"Format string for floating point numbers. For example ``float_format=\"%.2f\"`` will format 0.1234 to 0.12.","name":"float_format","default":"None","kind":"positional-or-keyword"},{"description":"Columns to write.","name":"columns","default":"None","kind":"positional-or-keyword"},{"description":"Write out the column names. If a list of string is given it is assumed to be aliases for the column names.","name":"header","default":"True","kind":"positional-or-keyword"},{"description":"Write row names (index).","name":"index","default":"True","kind":"positional-or-keyword"},{"description":"Column label for index column(s) if desired. If not specified, and `header` and `index` are True, then the index names are used. A sequence should be given if the DataFrame uses MultiIndex.","name":"index_label","default":"None","kind":"positional-or-keyword"},{"description":"Upper left cell row to dump data frame.","name":"startrow","default":"0","kind":"positional-or-keyword"},{"description":"Upper left cell column to dump data frame.","name":"startcol","default":"0","kind":"positional-or-keyword"},{"description":"Write engine to use, 'openpyxl' or 'xlsxwriter'. You can also set this via the options ``io.excel.xlsx.writer`` or ``io.excel.xlsm.writer``.","name":"engine","default":"None","kind":"positional-or-keyword"},{"description":"Write MultiIndex and Hierarchical Rows as merged cells.","name":"merge_cells","default":"True","kind":"positional-or-keyword"},{"description":"Representation for infinity (there is no native representation for infinity in Excel).","name":"inf_rep","default":"'inf'","kind":"positional-or-keyword"},{"description":"Specifies the one-based bottommost row and rightmost column that is to be frozen.","name":"freeze_panes","default":"None","kind":"positional-or-keyword"},{"name":"storage_options","default":"None","kind":"positional-or-keyword"},{"name":"engine_kwargs","default":"None","kind":"positional-or-keyword"}],[{"description":"Mapping from unique predictor names to aligned Network snapshots; all dyads include absent edges as zero.","name":"predictors","kind":"positional-or-keyword"},{"description":"'weight' uses aggregate strengths; 'binary' uses edge presence.","name":"values","kind":"keyword-only","default":"'weight'"},{"description":"Include diagonal dyads; False excludes them.","name":"include_loops","kind":"keyword-only","default":"False"},{"description":"Uniform node permutations per coefficient-specific reduced-model residual test; plus-one p-values.","name":"permutations","default":"999","kind":"keyword-only"},{"description":"Private CPU Torch random seed; predictor names and node labels use canonical order.","name":"seed","kind":"keyword-only","default":"0"},{"description":"'two-sided' compares absolute partial correlations; 'greater' and 'less' use the signed statistic.","name":"alternative","kind":"keyword-only","default":"'two-sided'"},{"description":"Fit a constant; the constant receives no fabricated permutation test.","name":"intercept","kind":"keyword-only","default":"True"},{"description":"'freedman_lane' permutes reduced response residuals; 'dsp' permutes focal predictor residuals and projects nuisance effects again.","name":"method","kind":"keyword-only","default":"'freedman_lane'"},{"description":"Named coefficient subsets tested together, conditional on remaining predictors; upper-tail partial R-squared.","name":"joint","kind":"keyword-only","default":"None"},{"description":"'none', 'bonferroni', 'holm' or 'bh' over all reported slopes and joint hypotheses; the intercept is excluded.","name":"adjustment","kind":"keyword-only","default":"'none'"},{"description":"Explicit complete-test structural work budget; no partial permutation p-values.","name":"max_work","kind":"keyword-only","default":"50000000"}],[{"name":"data","kind":"positional-or-keyword"},{"name":"y","kind":"positional-or-keyword"},{"name":"factors","default":"None","kind":"positional-or-keyword"},{"name":"covariates","kind":"keyword-only","default":"None"},{"description":"``\"none\"`` (main effects only) or a list such as ``[[\"a\", \"b\"]]`` or ``[\"a#b\"]``; a covariate may appear in an interaction. An interaction needs the effects it contains in the model (``a#b#c`` needs ``a#b``, ``a#c`` and ``b#c``; ``a#b#x`` needs ``a#x`` and ``b#x``), otherwise ``invalid_spec``.","name":"interactions","kind":"keyword-only","default":"'full'"},{"name":"ss_type","default":"3","kind":"keyword-only"},{"description":"``oe.rm_anova`` with ``factors`` as between-subject factors (Type III, all between-subject interactions; covariates, ``emmeans`` and other values of ``ss_type`` or ``interactions`` are rejected).","name":"within","kind":"keyword-only","default":"None"},{"description":"``oe.rm_anova`` with ``factors`` as between-subject factors (Type III, all between-subject interactions; covariates, ``emmeans`` and other values of ``ss_type`` or ``interactions`` are rejected).","name":"subject","kind":"keyword-only","default":"None"},{"description":"reported: predictions averaged with equal weights over the levels of the other factors, with covariates at their sample means. Single factors also get pairwise comparisons.","name":"emmeans","kind":"keyword-only","default":"None"},{"description":"SPSS's default) for the pairwise comparisons and their intervals.","name":"adjust","kind":"keyword-only","default":"'bonferroni'"},{"description":"tests the ANCOVA assumption of equal slopes.","name":"homogeneity_of_slopes","kind":"keyword-only","default":"False"},{"name":"alpha","kind":"keyword-only","default":"0.05"},{"description":"``\"raise\"``.","name":"missing","kind":"keyword-only","default":"'drop'"}]];
const STORED_PARAMETER_NAMES = [
  "max_work", "max_iterations", "tolerance", "covariance", "max_bytes",
  "intercept", "missing", "instruments", "endogenous", "data",
  "alpha", "weights", "result", "device", "time", "cluster", "categorical",
  "seed", "weight_type", "confidence", "components", "replications", "max_iter", "options", "columns", "level", "method", "batch_rows", "treatment", "alternative", "max_memory_mb",
] as const;
function decodeParameterName(n: StoredApiParameter["n"]): string {
  if (typeof n === "string") return n;
  if (typeof n === "number" && Number.isSafeInteger(n) && !Object.is(n, -0)
    && n >= 0 && n < STORED_PARAMETER_NAMES.length) {
    return STORED_PARAMETER_NAMES[n];
  }
  throw new Error("Editor API parameter needs a name; name for every parameter must be a string or compact integer token from 0 to 30.");
}
function decodeDefault(v: StoredApiParameter["v"], V: number | undefined, B: number | undefined,
  D: number | undefined, q: string | undefined, N: number | undefined): string | undefined {
  if (v !== undefined) {
    if (v === null) return "None";
    if (typeof v === "boolean") return v ? "True" : "False";
    if (typeof v === "number") {
      if (!Number.isSafeInteger(v) || Object.is(v, -0)) {
        throw new Error("Compact numeric defaults must be safe decimal integers.");
      }
      return String(v);
    }
    return v;
  }
  if (V !== undefined) {
    if (typeof V !== "number" || !Number.isInteger(V) || V !== 0 || Object.is(V, -0)) {
      throw new Error("Compact default needs the integer zero None tag");
    }
    return "None";
  }
  if (B !== undefined) {
    if (typeof B !== "number" || (B !== 0 && B !== 1) || Object.is(B, -0)) {
      throw new Error("Compact default needs the integer zero or one Boolean tag");
    }
    return B === 1 ? "True" : "False";
  }
  if (D !== undefined) {
    if (typeof D !== "number" || !Number.isInteger(D) || Object.is(D, -0)
      || D < 0 || D >= STORED_DEFAULT_LITERALS.length) {
      throw new Error("Compact default needs an integer fixed-literal tag from zero through fifteen");
    }
    return STORED_DEFAULT_LITERALS[D];
  }
  if (q !== undefined) {
    if (typeof q !== "string" || /['\\\n\r]/.test(q)) {
      throw new Error("Compact quoted defaults must contain an exact simple string interior.");
    }
    return `'${q}'`;
  }
  if (N !== undefined) {
    if (typeof N !== "number" || N !== 1) {
      throw new Error("Legacy compact default needs the integer one None tag");
    }
    return "None";
  }
  return undefined;
}
export function decodeApiEntry(entry: StoredApiEntry): ApiEntry {
  const { n, s, S, I, d, p, P, r, R, T, k, ...fields } = entry;
  const storedName = fields.name ?? n;
  const name = typeof storedName === "string" && storedName.startsWith("~")
    ? `openecon.${storedName.slice(1)}` : storedName;
  const legacySignature = typeof s === "string" && s.startsWith("(")
    && typeof name === "string" ? name.slice(name.lastIndexOf(".") + 1) + s : s;
  let signature = fields.signature ?? legacySignature
    ?? (typeof S === "string" && S.startsWith("(") && typeof name === "string"
      ? name.slice(name.lastIndexOf(".") + 1) + S : undefined);
  if (signature === undefined && !Object.hasOwn(entry, "S") && Object.hasOwn(entry, "I")) {
    if (typeof I !== "number" || !Number.isInteger(I) || Object.is(I, -0)
      || I < 0 || I >= STORED_SIGNATURE_SUFFIXES.length) {
      throw new Error("Compact signature needs a recognized integer token");
    }
    if (typeof name === "string") {
      signature = name.slice(name.lastIndexOf(".") + 1) + STORED_SIGNATURE_SUFFIXES[I];
    }
  }
  const description = fields.description ?? d;
  let parameters = fields.parameters ?? p;
  if (parameters === undefined && Object.hasOwn(entry, "P")) {
    if (typeof P !== "number" || !Number.isInteger(P) || Object.is(P, -0)
      || P < 0 || P >= STORED_PARAMETER_LISTS.length) {
      throw new Error("Compact parameters need a recognized integer token");
    }
    parameters = structuredClone(STORED_PARAMETER_LISTS[P]);
  }
  const legacyReturns = typeof r === "string" && r.startsWith("~")
    ? `openecon.${r.slice(1)}` : r;
  let returns = Object.hasOwn(fields, "returns") ? fields.returns
    : Object.hasOwn(entry, "r") ? legacyReturns
    : Object.hasOwn(entry, "R") ? `openecon.${R}` : undefined;
  if (!Object.hasOwn(fields, "returns") && !Object.hasOwn(entry, "r")
    && !Object.hasOwn(entry, "R") && Object.hasOwn(entry, "T")) {
    if (typeof T !== "number" || !Number.isInteger(T) || T < 0 || T >= STORED_RETURN_TYPES.length) {
      throw new Error("Compact return type needs a recognized integer token");
    }
    returns = STORED_RETURN_TYPES[T];
  }
  if (typeof name !== "string" || typeof signature !== "string"
    || typeof description !== "string" || !Array.isArray(parameters)) {
    throw new Error(`Editor API entry ${name ?? "<unknown>"} needs a signature and description, a name, and parameter metadata.`);
  }
  const storedKind = fields.kind ?? k;
  const kind = storedKind === "m" ? "method"
    : storedKind === "c" ? "class"
    : storedKind ?? "function";
  return {
    ...fields,
    name,
    signature,
    description,
    ...(returns !== undefined ? { returns } : {}),
    kind,
    ...(kind === "method"
      ? { owner: entry.owner ?? name.slice(0, name.lastIndexOf(".")) }
      : {}),
    parameters: parameters.map(({ n, d, a, v, V, B, D, q, N, c, k, t, ...parameter }) => {
      const name = parameter.name ?? decodeParameterName(n);
      if (typeof name !== "string") {
        throw new Error("Editor API parameter needs a name; name for every parameter is required.");
      }
      const description = parameter.description ?? d;
      const annotation = parameter.annotation ?? a;
      const defaultValue = parameter.default !== undefined
        ? parameter.default : decodeDefault(v, V, B, D, q, N);
      const choices = parameter.choices ?? c;
      const parameterKind = parameter.kind ?? k ?? t;
      return {
        ...parameter, name,
        ...(description !== undefined ? { description } : {}),
        ...(annotation !== undefined ? { annotation } : {}),
        ...(defaultValue !== undefined ? { default: defaultValue } : {}),
        ...(choices !== undefined ? { choices } : {}),
        kind: parameterKind !== undefined
          ? STORED_PARAMETER_KINDS[parameterKind] ?? parameterKind
          : "positional-or-keyword",
      };
    }),
  };
}

export const API_CATALOG: Record<string, ApiEntry> = Object.fromEntries(
  (entries as unknown as StoredApiEntry[]).map((entry) => {
    const decoded = decodeApiEntry(entry);
    return [decoded.name, decoded];
  }),
);
