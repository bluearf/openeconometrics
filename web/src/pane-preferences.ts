export type PaneKind = "sidebar" | "editor" | "terminal";

export interface PaneBounds {
  minimum: number;
  maximum: number;
}

export const PANE_DEFAULTS: Record<PaneKind, number> = {
  sidebar: 220,
  editor: 0.5,
  terminal: 226,
};

const SAFE_BOUNDS: Record<PaneKind, PaneBounds> = {
  sidebar: { minimum: 180, maximum: 560 },
  editor: { minimum: 0.05, maximum: 0.95 },
  terminal: { minimum: 80, maximum: 1600 },
};

export function clampPane(value: number, bounds: PaneBounds): number {
  return Math.min(bounds.maximum, Math.max(bounds.minimum, value));
}

export function paneStorageKey(
  projectId: string | null | undefined,
  kind: PaneKind,
): string {
  return `openecon:panes:v1:${encodeURIComponent(projectId ?? "local")}:${kind}`;
}

export function readPanePreference(key: string, kind: PaneKind): number {
  try {
    const raw = window.localStorage.getItem(key);
    if (raw !== null) {
      const value: unknown = JSON.parse(raw);
      if (typeof value === "number" && Number.isFinite(value)) {
        return clampPane(value, SAFE_BOUNDS[kind]);
      }
    }
  } catch {
    // A private or unavailable storage area never blocks the workspace.
  }
  return PANE_DEFAULTS[kind];
}

export function savePanePreference(key: string, value: number): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Resizing remains available even when local preferences cannot be saved.
  }
}

export function sidebarBounds(width: number): PaneBounds {
  return {
    minimum: 180,
    maximum: Math.max(180, Math.min(560, width * 0.42, width - 572)),
  };
}

export function editorBounds(width: number): PaneBounds {
  const available = Math.max(1, width - 6);
  return {
    minimum: Math.min(300 / available, 0.52),
    maximum: 1 - Math.min(260 / available, 0.48),
  };
}

export function terminalBounds(height: number): PaneBounds {
  const maximum = Math.max(80, Math.min(1600, height - 180));
  return { minimum: Math.min(106, maximum), maximum };
}
