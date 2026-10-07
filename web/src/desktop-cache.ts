/** Public configuration and account-scoped project metadata, never auth tokens. */
import type { TeamProfile } from "./team-api.ts";

export function forgetDesktopProfile(uid: string): void {
  try { localStorage.removeItem(`openecon-desktop-profile:${uid}`); } catch { /* No cached metadata to remove. */ }
}

export function cacheDesktopProfile(profile: TeamProfile): void {
  try { localStorage.setItem(`openecon-desktop-profile:${profile.user.uid}`, JSON.stringify(profile)); }
  catch { /* Disk metadata remains available; quota does not affect calculation. */ }
}

export function cachedDesktopProfile(uid: string): TeamProfile | null {
  try {
    const value = JSON.parse(localStorage.getItem(`openecon-desktop-profile:${uid}`) || "null");
    if (!value || value.user?.uid !== uid || !Array.isArray(value.projects) || !Array.isArray(value.invitations)
      || !value.projects.every((project: {id?:string;role?:string}) => typeof project.id === "string"
        && /^[0-9a-f]{32}$/.test(project.id) && ["owner", "editor", "viewer"].includes(project.role || ""))) return null;
    return value as TeamProfile;
  } catch { return null; }
}
