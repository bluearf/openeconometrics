import { Children, useRef, type CSSProperties, type ReactNode } from "react";
import {
  ResizeSeparator,
  usePaneExtent,
  usePanePreference,
} from "./PaneResize";
import { editorBounds, sidebarBounds } from "./pane-preferences";

interface WorkspaceLayoutProps {
  projectId?: string | null;
  collapsed: boolean;
  sidebar: ReactNode;
  children: ReactNode;
}

export function WorkspaceLayout({
  projectId,
  collapsed,
  sidebar,
  children,
}: WorkspaceLayoutProps) {
  const layout = useRef<HTMLDivElement>(null);
  const extent = usePaneExtent(layout, "width");
  const bounds = sidebarBounds(extent || 1024);
  const size = usePanePreference(projectId, "sidebar", bounds);
  return (
    <div
      ref={layout}
      className={`workspace-layout workspace-layout--resizable${collapsed ? "" : " workspace-layout--files-open"}`}
      style={{ "--files-pane-width": `${size.value}px` } as CSSProperties}
    >
      {sidebar}
      {!collapsed && (
        <ResizeSeparator
          label="Files panel width"
          orientation="vertical"
          {...bounds}
          {...size}
          onChange={size.change}
          onCommit={size.commit}
          onCancel={size.cancel}
        />
      )}
      {children}
    </div>
  );
}

export function WorkbenchPanels({
  projectId,
  mobilePane,
  children,
}: {
  projectId?: string | null;
  mobilePane: "editor" | "output";
  children: ReactNode;
}) {
  const layout = useRef<HTMLDivElement>(null);
  const width = usePaneExtent(layout, "width");
  const bounds = editorBounds(width || 800);
  const size = usePanePreference(projectId, "editor", bounds);
  const panes = Children.toArray(children);
  return (
    <div
      ref={layout}
      className={`workbench workbench--resizable mobile-${mobilePane}`}
      style={{ "--editor-pane-ratio": size.value } as CSSProperties}
    >
      {panes[0]}
      <ResizeSeparator
        label="Code and results panel width"
        orientation="vertical"
        {...bounds}
        {...size}
        step={0.01}
        unitsPerPixel={1 / Math.max(1, (width || 800) - 6)}
        valueScale={100}
        onChange={size.change}
        onCommit={size.commit}
        onCancel={size.cancel}
      />
      {panes[1]}
    </div>
  );
}
