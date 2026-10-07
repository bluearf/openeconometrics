export interface PackageVersion {
  name: string;
  version: string;
}

interface PackageManifestBase {
  python: string;
  core: Record<string, string>;
  requirements: PackageVersion[];
  locked: PackageVersion[];
}

export type PackageManifest = PackageManifestBase &
  (
    | { schema: 1 }
    | { schema: 2; installer: "pip" | "uv"; specifications: string[] }
  );

export interface PortableEnvironment {
  format: "openecon.environment.v1";
  runtime: {
    platform: string;
    machine: string;
    implementation: string;
    abi: string;
  };
  manifest: PackageManifest;
  sha256: string;
}

export interface EnvironmentPreview {
  manifest: PackageManifest;
  matches: boolean;
}

export interface PackageJob {
  id: string;
  state: "running" | "complete" | "error" | "cancelled";
  message: string;
  log: string | string[];
}

export interface ProjectPackages {
  available: boolean;
  base: PackageVersion[];
  requirements: PackageVersion[];
  installed: PackageVersion[];
  manifest: PackageManifest;
  job: PackageJob | null;
}

export type PackageSnapshot = ProjectPackages;

export interface SharedPackages {
  manifest: PackageManifest | null;
  version: number;
  localMatches: boolean;
  offline?: boolean;
}

export function normalizePackageName(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[-_.]+/g, "-");
}

export function boundedPackageLog(
  log: PackageJob["log"],
  maxChars = 8192,
): string {
  const text = Array.isArray(log) ? log.slice(-100).join("\n") : log;
  if (text.length <= maxChars) return text;
  let tail = text.slice(-maxChars);
  // Keep a cut UTF-16 character out of the displayed tail.
  if (tail.charCodeAt(0) >= 0xdc00 && tail.charCodeAt(0) <= 0xdfff) {
    tail = tail.slice(1);
  }
  return tail;
}
