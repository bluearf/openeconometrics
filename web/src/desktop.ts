/** Desktop-only native bridge. Cloud credentials never enter the Python API. */
import { ApiError, decodeResponse } from "./api.ts";
import { nativeCoreInvoke } from "./native-core.ts";
export { isDesktop } from "./native-core.ts";

export function nativeInvoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  return nativeCoreInvoke<T>(command, args).catch((error: unknown) => {
    if (error === "OPENECON_NETWORK") throw new TypeError("The cloud connection was lost.");
    if (typeof error === "string" && /^OPENECON_HTTP:[1-5][0-9]{2}$/.test(error))
      throw new ApiError("Could not complete the cloud request.", Number(error.slice(14)));
    throw error;
  });
}

export function isTransientDesktopFailure(error: unknown): boolean {
  return error instanceof ApiError ? error.status >= 500
    : error instanceof TypeError || Boolean(error && typeof error === "object"
      && "code" in error && error.code === "auth/network-request-failed");
}

export async function desktopCloudFetch(path: string, token: string, options: RequestInit = {}): Promise<Response> {
  if (options.signal?.aborted) throw new DOMException("Request cancelled", "AbortError");
  if (options.body && typeof options.body !== "string") throw new Error("Invalid cloud request.");
  const response = await nativeInvoke<{ status: number; body: unknown }>("cloud_request", {
    method: options.method || "GET", path, token,
    body: options.body ? JSON.parse(options.body) : null,
  });
  if (options.signal?.aborted) throw new DOMException("Request cancelled", "AbortError");
  return new Response(response.status === 204 ? null : JSON.stringify(response.body), {
    status: response.status, headers: { "Content-Type": "application/json" },
  });
}

export async function desktopCloudRequest<T>(path: string, token: string, options: RequestInit = {}): Promise<T> {
  return decodeResponse<T>(await desktopCloudFetch(path, token, options));
}

export interface CachedCloudFile {
  cloud_id: string;
  name: string;
  python_path: string;
  sha256: string;
  data_hash: string;
  size_bytes: number;
  reused: boolean;
}
