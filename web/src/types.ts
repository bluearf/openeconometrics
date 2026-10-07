// The estimator registry is open-ended: every registered model is valid here.
export type Estimator = "ols" | "logit" | "probit" | (string & {});
export type Covariance =
  | "nonrobust"
  | "HC1"
  | "HC3"
  | "cluster"
  | (string & {});
export interface ColumnProfile {
  name: string;
  dtype: string;
  numeric: boolean;
  missing: number;
  unique: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
}
export interface DatasetProfile {
  id: string;
  name: string;
  row_count: number | null;
  column_count: number | null;
  python_path?: string;
  columns: ColumnProfile[];
  preview: Record<string, unknown>[];
  source: "example" | "upload";
  data_hash: string;
  created_at: string;
  size_bytes: number;
}
export interface ModelSpec {
  estimator: Estimator;
  outcome: string;
  predictors: string[];
  categorical: string[];
  intercept: boolean;
  covariance: Covariance;
  cluster: string | string[] | null;
  missing: "raise" | "drop";
  alpha: number;
  weights?: string | null;
  weight_type?: string | null;
  panel?: string | null;
  time?: string | null;
  columns?: Record<string, string | string[]>;
  options?: Record<string, unknown>;
}
export interface Coefficient {
  term: string;
  estimate: number;
  std_error: number;
  statistic: number;
  p_value: number;
  ci_low: number;
  ci_high: number;
  equation?: string | null;
}
export interface ResultBundle {
  id: string;
  created_at: string;
  dataset_id: string;
  dataset_name: string;
  spec: ModelSpec;
  nobs: number;
  nobs_original: number;
  dropped_rows: number;
  coefficients: Coefficient[];
  covariance_matrix: number[][];
  metrics: Record<string, number | null>;
  warnings: string[];
  predictions: {
    row: number;
    observed: number;
    fitted: number;
    residual: number;
  }[];
  sample_positions: number[];
  provenance: Record<string, unknown>;
  inference: Record<string, unknown>;
  title?: string | null;
  tests?: Record<string, unknown>;
  extra?: Record<string, unknown>;
  display_omitted?: string[];
}
export interface AppConfig {
  mcp_available?: boolean;
  notice?: string;
  mcp_command: string;
  codex_command: string;
  claude_command: string;
  version: string;
}

export interface ConsoleVariable {
  name: string;
  type: string;
  preview: string;
}
export interface ConsoleOutput {
  type: "model" | "plot" | "table" | "text" | "latex";
  data: unknown;
  latex?: string;
  latex_math?: string | null;
  latex_style?: "publication-v1" | "publication-v2";
  latex_notes?: string[];
}
export interface ExecutionArtifact {
  name: string;
  size_bytes: number;
  url: string;
}
export type ExecutionEvent =
  { type: "stdout"; text: string } | { type: "output"; index: number };
export interface ExecutionRecord {
  source?: "mcp";
  actor_uid?: string;
  local_only?: boolean;
  sharing_scope?: "partial";
  local_plot_count?: number;
  artifacts?: ExecutionArtifact[];
  id: string;
  code: string;
  status: "ok" | "error" | "timeout" | "interrupted";
  stdout: string;
  events?: ExecutionEvent[];
  error: { type: string; message: string; traceback: string } | null;
  outputs: ConsoleOutput[];
  variables: ConsoleVariable[];
  duration_ms: number;
  session_generation: number;
  active_session_generation?: number;
  state_reset?: boolean;
  created_at?: string;
}
export interface ConsoleState {
  history: ExecutionRecord[];
  status: { running: boolean; session_generation: number; pid?: number | null };
  variables: ConsoleVariable[];
}
