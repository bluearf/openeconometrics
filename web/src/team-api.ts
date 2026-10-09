import { ApiError, decodeResponse, type TokenGetter } from "./api.ts";
import {
  desktopCloudFetch,
  isDesktop,
  isTransientDesktopFailure,
} from "./desktop.ts";

export type ProjectRole = "owner" | "editor" | "viewer";
export interface Project {
  id: string;
  name: string;
  description?: string;
  role: ProjectRole;
  created_at: string;
  updated_at?: string;
  name_version?: number;
}
export interface Invitation {
  id: string;
  project_id: string;
  project_name: string;
  email: string;
  role: "editor" | "viewer";
  expires_at: string;
  url?: string;
}
export interface Member {
  uid: string;
  email: string;
  name?: string;
  role: ProjectRole;
}
export interface TeamProfile {
  user: { uid: string; email: string; name?: string; email_verified?: boolean };
  enabled: boolean;
  projects: Project[];
  invitations: Invitation[];
}
export interface TeamAuthConfig {
  mode: "teams";
  account_link_available?: boolean;
  firebase: {
    apiKey: string;
    authDomain: string;
    projectId: string;
    appId: string;
  };
}
export interface TeamRoute {
  projectId: string | null;
  invitationId: string | null;
}

export function teamRoute(url: string): TeamRoute {
  const parsed = new URL(url, "http://localhost");
  const invitation =
    parsed.searchParams.get("invite") ||
    parsed.pathname.match(/^\/invite\/([^/]+)\/?$/)?.[1] ||
    null;
  const project =
    parsed.searchParams.get("project") ||
    parsed.pathname.match(/^\/projects\/([^/]+)\/?$/)?.[1] ||
    null;
  return { projectId: project, invitationId: invitation };
}
export function projectUrl(projectId: string | null): string {
  return projectId ? `/?project=${encodeURIComponent(projectId)}` : "/";
}
export const canEditProject = (role: ProjectRole) =>
  role === "owner" || role === "editor";

/** One authenticated account per client; project scoping lives in a separate immutable client. */
export function createTeamApi(getToken: TokenGetter) {
  return async function request<T>(
    path: string,
    options: RequestInit = {},
  ): Promise<T> {
    if (!path.startsWith("/") || path.startsWith("//"))
      throw new Error("Invalid API path.");
    const perform = async (refresh: boolean) => {
      const token = await getToken(refresh);
      if (options.signal?.aborted)
        throw new DOMException("Request cancelled", "AbortError");
      if (!token)
        throw new ApiError("You need to sign in.", 401, "AUTH_REQUIRED");
      const headers = new Headers(options.headers);
      headers.set("Authorization", `Bearer ${token}`);
      if (options.body && !(options.body instanceof FormData))
        headers.set("Content-Type", "application/json");
      return isDesktop()
        ? desktopCloudFetch(`/api${path}`, token, options)
        : fetch(`/api${path}`, { ...options, headers });
    };
    let response = await perform(false);
    if (response.status === 401 && !options.signal?.aborted)
      response = await perform(true);
    return decodeResponse<T>(response);
  };
}

/** A failed team configuration must never silently enter unauthenticated local mode. */
export async function loadAuthConfig(
  signal?: AbortSignal,
): Promise<TeamAuthConfig | "local"> {
  let response: Response;
  try {
    response = isDesktop()
      ? await desktopCloudFetch("/api/auth/config", "", { signal })
      : await fetch("/api/auth/config", { signal });
  } catch (error) {
    if (!isDesktop() || signal?.aborted || !isTransientDesktopFailure(error))
      throw error;
    const cached = localStorage.getItem("openecon-desktop-auth-config");
    if (!cached) throw error;
    response = new Response(cached, {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
  if (isDesktop() && response.status >= 500) {
    try {
      const cached = localStorage.getItem("openecon-desktop-auth-config");
      if (cached)
        response = new Response(cached, {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
    } catch {
      /* The original service response stays authoritative. */
    }
  }
  if (response.status === 404) {
    if (isDesktop())
      throw new ApiError("Could not configure the cloud connection.", 404);
    return "local";
  }
  const value = await decodeResponse<TeamAuthConfig | { mode: "local" }>(
    response,
  );
  if (value.mode === "local") {
    if (isDesktop())
      throw new Error("The desktop cloud configuration is invalid.");
    return "local";
  }
  if (
    value.mode !== "teams" ||
    !value.firebase?.apiKey ||
    !value.firebase.projectId ||
    !value.firebase.authDomain ||
    !value.firebase.appId
  )
    throw new Error(
      "Sign-in configuration is missing. Contact the administrator.",
    );
  if (isDesktop()) {
    try {
      localStorage.setItem(
        "openecon-desktop-auth-config",
        JSON.stringify(value),
      );
    } catch {
      /* No auth secrets. */
    }
  }
  return value;
}
