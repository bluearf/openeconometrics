/** Low-level native IPC, independent of workspace API initialization. */
declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
    __TAURI__?: {
      core: {
        invoke<T>(command: string, args?: Record<string, unknown>): Promise<T>;
      };
    };
  }
}

export const isDesktop = () =>
  typeof window !== "undefined" && Boolean(window.__TAURI_INTERNALS__);

export function nativeCoreInvoke<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T> {
  const invoke = window.__TAURI__?.core.invoke;
  if (!isDesktop() || !invoke)
    throw new Error("Could not connect to the desktop app.");
  return invoke<T>(command, args);
}
