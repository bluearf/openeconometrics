/** Local catalog identity never comes from a cloud profile or URL alone. */
import {
  createWorkspaceClient,
  decodeResponse,
  type WorkspaceClient,
} from "./api.ts";

export interface LocalProject {
  id: string;
  name: string;
  description: string;
}
export function createLocalProjects() {
  let token = "";
  async function request<T>(path = "", options: RequestInit = {}): Promise<T> {
    if (!token)
      token = (
        await decodeResponse<{ token: string }>(
          await fetch("/api/desktop/session"),
        )
      ).token;
    return decodeResponse<T>(
      await fetch(`/api/desktop/local-projects${path}`, {
        ...options,
        headers: {
          "X-OpenEcon-Token": token,
          "Content-Type": "application/json",
        },
      }),
    );
  }
  return {
    list: () => request<{ projects: LocalProject[] }>(),
    create: (name: string, description: string) =>
      request<LocalProject>("", {
        method: "POST",
        body: JSON.stringify({ name, description }),
      }),
    async open(
      id: string,
    ): Promise<{ project: LocalProject; client: WorkspaceClient }> {
      if (!/^[0-9a-f]{32}$/.test(id)) throw new Error("Invalid local project.");
      const project = await request<LocalProject>(`/${id}/open`, {
        method: "POST",
      });
      if (project.id !== id)
        throw new Error("Could not verify the local project.");
      const transport = createWorkspaceClient(
        null,
        undefined,
        `/api/desktop/projects/${id}/workspace`,
      );
      return { project, client: { ...transport, projectId: id, teams: false, localDesktop: true } };
    },
  };
}
