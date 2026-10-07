import { ApiError } from "./api.ts";
import type { Project, TeamProfile } from "./team-api.ts";

export function projectName(value: string): string {
  const name = value.trim();
  if (
    !name ||
    Array.from(name).length > 100 ||
    /[\p{Cc}\p{Cs}\u2028\u2029]/u.test(value)
  )
    throw new Error(
      "The project name must be 1–100 characters and contain no control characters.",
    );
  return name;
}

const version = (project: Project) => project.name_version ?? 0;
export interface ProjectNameChange {
  readonly account: string;
  readonly generation: number;
  readonly projectId: string;
  readonly name: string;
  readonly originalName: string;
  readonly name_version: number;
}

/** Names are optimistic views; cached profiles contain acknowledged metadata only. */
export class ProjectNames {
  private account: string | null = null;
  private generation = 0;
  private pending = new Map<string, ProjectNameChange>();
  private accepted = new Map<string, Project>();

  select(account: string | null): void {
    if (account === this.account) return;
    this.account = account;
    this.generation++;
    this.pending.clear();
    this.accepted.clear();
  }

  busy(projectId: string): boolean {
    return this.pending.has(projectId);
  }

  begin(project: Project, value: string): ProjectNameChange {
    if (!this.account || project.role !== "owner")
      throw new ApiError(
        "Only the project owner can rename the project.",
        403,
        "ROLE_REQUIRED",
      );
    if (this.pending.has(project.id))
      throw new Error("Saving project name.");
    if (!Number.isSafeInteger(version(project)) || version(project) < 0)
      throw new Error("The project name version is invalid. Refresh the projects.");
    const change = {
      account: this.account,
      generation: this.generation,
      projectId: project.id,
      name: projectName(value),
      originalName: project.name,
      name_version: version(project),
    };
    this.pending.set(project.id, change);
    return change;
  }

  current(change: ProjectNameChange): boolean {
    return (
      this.account === change.account &&
      this.generation === change.generation &&
      this.pending.get(change.projectId) === change
    );
  }

  cancel(change: ProjectNameChange): void {
    if (this.current(change)) this.pending.delete(change.projectId);
  }

  accept(change: ProjectNameChange, saved: Project): boolean {
    if (!this.current(change)) return false;
    if (
      saved.id !== change.projectId ||
      saved.name !== change.name ||
      !Number.isSafeInteger(saved.name_version) ||
      !(
        saved.name_version === change.name_version + 1 ||
        (change.name === change.originalName &&
          saved.name_version === change.name_version)
      )
    )
      throw new Error("Could not verify the saved project name. Refresh the projects.");
    const previous = this.accepted.get(saved.id);
    if (!previous || version(saved) >= version(previous))
      this.accepted.set(saved.id, saved);
    this.pending.delete(change.projectId);
    return true;
  }

  reconcile(profile: TeamProfile): TeamProfile {
    if (profile.user.uid !== this.account) return profile;
    const projects = profile.projects.map((project) => {
      const saved = this.accepted.get(project.id);
      return saved && version(saved) > version(project)
        ? {
            ...project,
            name: saved.name,
            name_version: saved.name_version,
            updated_at: saved.updated_at,
          }
        : project;
    });
    const names = new Map(
      projects.map((project) => [project.id, project.name]),
    );
    return {
      ...profile,
      projects,
      invitations: profile.invitations.map((invite) =>
        names.has(invite.project_id)
          ? { ...invite, project_name: names.get(invite.project_id)! }
          : invite,
      ),
    };
  }

  view(profile: TeamProfile): TeamProfile {
    const canonical = this.reconcile(profile);
    if (canonical.user.uid !== this.account) return canonical;
    return {
      ...canonical,
      projects: canonical.projects.map((project) => {
        const pending = this.pending.get(project.id);
        return pending && project.role === "owner"
          ? { ...project, name: pending.name }
          : project;
      }),
    };
  }
}
