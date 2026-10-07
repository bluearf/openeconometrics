import entries from "./editor-api.json" with { type: "json" };

export interface ApiParameter {
  name: string;
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

// Generated from committed openecon APIs and the packaged Python environment.
// Static data keeps editor assistance independent of code execution and login.
export const API_CATALOG: Record<string, ApiEntry> = Object.fromEntries(
  (entries as ApiEntry[]).map((entry) => [entry.name, entry]),
);
