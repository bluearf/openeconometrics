import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import {
  createFileFolder,
  fileEntryKey,
  fileFolderPath,
  mergeFileLayout,
  moveFileEntry,
  placeFileAfter,
  placeFileBefore,
  renameFileEntry,
  reorderFileEntry,
  type FileLayoutEntry,
  documentExtension,
  normalizeDocumentName,
  fileNameFold,
} from "./file-layout";
import type { DatasetProfile } from "./types";
import "./files-sidebar.css";

export interface SidebarScript {
  id: string;
  name: string;
}

export interface FilesSidebarProps {
  files: readonly DatasetProfile[];
  collapsed: boolean;
  onCollapsedChange: (collapsed: boolean) => void;
  onOpen: (file: DatasetProfile) => void;
  scripts?: readonly SidebarScript[];
  activeScriptId?: string;
  onOpenScript?: (script: SidebarScript) => void;
  onCreateScript?: (
    parent: string | null,
    name: string,
  ) => void | Promise<void>;
  onUpload?: (parent?: string | null) => void;
  onImportScript?: (parent?: string | null) => void;
  onDownload?: (file: DatasetProfile) => void;
  activeFileId?: string;
  disabled?: boolean;
  navigationPending?: boolean;
  readOnly?: boolean;
  uploading?: boolean;
  pendingSync?: boolean;
  layout?: readonly FileLayoutEntry[];
  onLayoutChange?: (entries: FileLayoutEntry[]) => Promise<void>;
  footer?: ReactNode;
}

type FileDropPosition = "before" | "after" | "into";
type FileDropTarget = { key: string; position: FileDropPosition };

function fileDropPosition(
  event: React.DragEvent<HTMLElement>,
  isFolder: boolean,
): FileDropPosition {
  const bounds = event.currentTarget.getBoundingClientRect();
  if (bounds.height <= 0) return isFolder ? "into" : "before";
  const position = (event.clientY - bounds.top) / bounds.height;
  if (isFolder && position > 0.25 && position < 0.75) return "into";
  return position > 0.5 ? "after" : "before";
}

function FileIcon({
  kind,
}: {
  kind:
    | "new-folder"
    | "chevron"
    | "folder"
    | "file"
    | "code"
    | "new-code"
    | "import-code"
    | "add-data"
    | "collapse";
}) {
  return (
    <svg
      width="17"
      height="17"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {kind === "chevron" && <path d="m9 5 7 7-7 7" />}
      {kind === "new-folder" && (
        <>
          <path d="M3 6h7l2 3h9v5M3 6V4h7l2 2h9v3M3 6v14h10" />
          <path d="M18 15v6m-3-3h6" />
        </>
      )}
      {kind === "folder" && <path d="M3 6h7l2 3h9v11H3ZM3 6V4h7l2 2h9v3" />}
      {kind === "file" && (
        <>
          <path d="M6 3h9l4 4v14H6Z" />
          <path d="M14 3v5h5M9 12h7m-7 4h7" />
        </>
      )}
      {kind === "code" && <path d="m8 7-5 5 5 5m8-10 5 5-5 5m-3-13-2 16" />}
      {kind === "new-code" && (
        <>
          <path d="M5 3h9l4 4v7M5 3v18h8M14 3v5h4" />
          <path d="M18 15v6m-3-3h6" />
        </>
      )}
      {kind === "import-code" && (
        <>
          <path d="M5 3h9l4 4v14H5ZM14 3v5h4" />
          <path d="M11.5 10v7m-3-3 3 3 3-3" />
        </>
      )}
      {kind === "add-data" && (
        <>
          <path d="M3 4h17v8M3 4v16h9M3 9h17M9 4v16M3 14h9" />
          <path d="M18 15v6m-3-3h6" />
        </>
      )}
      {kind === "collapse" && <path d="m14 6-6 6 6 6" />}
    </svg>
  );
}

function SidebarAction({
  label,
  kind,
  disabled,
  onClick,
  expanded,
}: {
  label: string;
  kind: "new-code" | "import-code" | "add-data" | "new-folder";
  disabled: boolean;
  onClick: () => void;
  expanded?: boolean;
}) {
  const tooltipId = useId();
  const [visible, setVisible] = useState(false);
  const hovered = useRef(false);
  const focused = useRef(false);
  const dismissed = useRef(false);
  const showing = visible && !disabled;
  function enter(source: "hover" | "focus") {
    if (!hovered.current && !focused.current) dismissed.current = false;
    if (source === "hover") hovered.current = true;
    else focused.current = true;
    if (!disabled && !dismissed.current) setVisible(true);
  }
  function leave(source: "hover" | "focus") {
    if (source === "hover") hovered.current = false;
    else focused.current = false;
    dismissed.current = hovered.current || focused.current;
    setVisible(false);
  }
  function dismiss() {
    dismissed.current = true;
    setVisible(false);
  }
  useEffect(() => {
    if (disabled) {
      focused.current = false;
      dismissed.current = true;
      setVisible(false);
      return;
    }
    if (!visible) return;
    const timer = window.setTimeout(() => {
      dismissed.current = true;
      setVisible(false);
    }, 4000);
    return () => window.clearTimeout(timer);
  }, [visible, disabled]);
  return (
    <span
      className="files-sidebar__action"
      onMouseEnter={() => enter("hover")}
      onMouseLeave={() => leave("hover")}
    >
      <button
        type="button"
        className="files-sidebar__icon"
        aria-label={label}
        aria-expanded={expanded}
        aria-haspopup={expanded === undefined ? undefined : "menu"}
        aria-describedby={showing ? tooltipId : undefined}
        disabled={disabled}
        onClick={() => {
          dismiss();
          onClick();
        }}
        onFocus={() => enter("focus")}
        onBlur={() => leave("focus")}
        onKeyDown={(event) => {
          if (event.key === "Escape") dismiss();
        }}
      >
        <FileIcon kind={kind} />
      </button>
      <span
        id={tooltipId}
        className="files-sidebar__tooltip"
        role="tooltip"
        hidden={!showing}
      >
        {label}
      </span>
    </span>
  );
}

export default function FilesSidebar({
  files,
  collapsed,
  onCollapsedChange,
  onOpen,
  scripts,
  activeScriptId,
  onOpenScript,
  onCreateScript,
  onUpload,
  onImportScript,
  onDownload,
  activeFileId,
  disabled = false,
  navigationPending = false,
  readOnly = false,
  uploading = false,
  pendingSync = false,
  layout,
  onLayoutChange,
  footer,
}: FilesSidebarProps) {
  const contentId = useId();
  const toggleLabel = collapsed ? "Show files" : "Hide files";
  const visibleScripts =
    scripts ??
    (onOpenScript ? [{ id: "analysis.py", name: "analysis.py" }] : []);
  const selectedScriptId =
    activeScriptId ??
    (scripts === undefined && activeFileId === "analysis.py"
      ? "analysis.py"
      : undefined);
  const entries = mergeFileLayout(layout, visibleScripts, files);
  const [closedFolders, setClosedFolders] = useState<Set<string>>(new Set());
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);
  const [menu, setMenu] = useState<string | null>(null);
  const [menuPosition, setMenuPosition] = useState({ left: 0, top: 0 });
  const [queuedRename, setQueuedRename] = useState<string | null>(null);
  const [editing, setEditing] = useState<{
    key: string | null;
    name: string;
    parent: string | null;
    type?: "file";
  } | null>(null);
  const [moving, setMoving] = useState<{
    key: string;
    parent: string | null;
  } | null>(null);
  const [saving, setSaving] = useState(false);
  const savingRef = useRef(false);
  const [error, setError] = useState("");
  const [dropTarget, setDropTarget] = useState<FileDropTarget | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuOriginRef = useRef<HTMLButtonElement | null>(null);
  const dragged = useRef<string | null>(null);
  const suppressFileClickUntil = useRef(0);
  const mutable = !readOnly && !!onLayoutChange;
  const mutableRef = useRef(mutable);
  const canCreateRef = useRef(!readOnly && !!onCreateScript);
  const entriesRef = useRef(entries);
  const mountedRef = useRef(true);
  const dialogGeneration = useRef(0);
  mutableRef.current = mutable;
  canCreateRef.current = !readOnly && !!onCreateScript;
  entriesRef.current = entries;
  const blocked = disabled || saving || uploading;
  const draggable = mutable && !blocked && !editing && !moving;
  const parent = entries.some(
    (entry) => entry.kind === "folder" && entry.id === selectedFolder,
  )
    ? selectedFolder
    : null;
  const selectedEntry = moving
    ? entries.find((entry) => fileEntryKey(entry) === moving.key)
    : undefined;
  const folders = entries.filter((entry) => entry.kind === "folder");
  const destinations = folders.filter((folder) => {
    if (!selectedEntry) return true;
    try {
      moveFileEntry(entries, fileEntryKey(selectedEntry), folder.id);
      return true;
    } catch {
      return false;
    }
  });

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      dialogGeneration.current += 1;
    };
  }, []);
  useEffect(() => {
    const rejectOutsideDrop = (event: DragEvent) => {
      if (
        !dragged.current ||
        (event.target instanceof Node &&
          contentRef.current?.contains(event.target))
      )
        return;
      event.preventDefault();
      event.stopPropagation();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "none";
      if (event.type === "drop") finishDrag();
      else setDropTarget(null);
    };
    document.addEventListener("dragenter", rejectOutsideDrop, true);
    document.addEventListener("dragover", rejectOutsideDrop, true);
    document.addEventListener("drop", rejectOutsideDrop, true);
    return () => {
      document.removeEventListener("dragenter", rejectOutsideDrop, true);
      document.removeEventListener("dragover", rejectOutsideDrop, true);
      document.removeEventListener("drop", rejectOutsideDrop, true);
    };
  }, []);
  useEffect(() => {
    if (readOnly) {
      dialogGeneration.current += 1;
      setMenu(null);
      setQueuedRename(null);
      setEditing(null);
      setMoving(null);
      setError("");
    }
  }, [readOnly]);
  useEffect(() => {
    if (editing && error && !blocked) nameInputRef.current?.focus();
  }, [editing, error, blocked]);
  useEffect(() => {
    if (!menu) return;
    const close = (event: MouseEvent) => {
      if (
        !(event.target instanceof Element) ||
        !menuRef.current?.contains(event.target)
      )
        closeMenu();
    };
    const dismiss = () => closeMenu();
    document.addEventListener("mousedown", close);
    window.addEventListener("resize", dismiss);
    contentRef.current?.addEventListener("scroll", dismiss);
    return () => {
      document.removeEventListener("mousedown", close);
      window.removeEventListener("resize", dismiss);
      contentRef.current?.removeEventListener("scroll", dismiss);
    };
  }, [menu]);
  useLayoutEffect(() => {
    if (!menu || !menuRef.current) return;
    const bounds = menuRef.current.getBoundingClientRect();
    const sidebar = contentRef.current?.getBoundingClientRect();
    const width = bounds.width || 155;
    const height = bounds.height || 150;
    const leftEdge = Math.min(
      Math.max(8, sidebar?.left ?? 8),
      Math.max(8, window.innerWidth - width - 8),
    );
    const rightEdge = Math.min(
      window.innerWidth - 8,
      sidebar?.right || window.innerWidth - 8,
    );
    setMenuPosition((current) => ({
      left: Math.max(
        leftEdge,
        Math.min(current.left, Math.max(leftEdge, rightEdge - width)),
      ),
      top: Math.max(8, Math.min(current.top, window.innerHeight - height - 8)),
    }));
    menuRef.current
      .querySelector<HTMLButtonElement>("button:not(:disabled)")
      ?.focus();
  }, [menu, menuPosition.left, menuPosition.top]);
  useEffect(() => {
    if (
      menu &&
      (collapsed || !entries.some((entry) => fileEntryKey(entry) === menu))
    )
      closeMenu();
  }, [menu, collapsed, entries]);
  useEffect(() => {
    if (!queuedRename) return;
    const entry = entries.find((item) => fileEntryKey(item) === queuedRename);
    if (!mutable || !entry || (blocked && !navigationPending)) {
      setQueuedRename(null);
      return;
    }
    if (!blocked) {
      setQueuedRename(null);
      startRename(entry);
    }
  }, [queuedRename, blocked, navigationPending, mutable, entries]);
  useEffect(() => {
    if (!queuedRename) return;
    const cancel = (event: KeyboardEvent) => {
      if (event.key === "Escape") setQueuedRename(null);
    };
    document.addEventListener("keydown", cancel, true);
    return () => document.removeEventListener("keydown", cancel, true);
  }, [queuedRename]);

  function closeMenu(restoreFocus = false) {
    setMenu(null);
    if (restoreFocus && menuOriginRef.current?.isConnected)
      menuOriginRef.current.focus();
  }

  function openMenu(
    entry: FileLayoutEntry,
    row: HTMLElement,
    left: number,
    top: number,
  ) {
    if (blocked || savingRef.current || editing || moving || dragged.current)
      return;
    if (!mutable && !(entry.kind === "dataset" && onDownload)) return;
    menuOriginRef.current = row.querySelector<HTMLButtonElement>(
      ".files-sidebar__file",
    );
    setMenuPosition({ left, top });
    setMenu(fileEntryKey(entry));
    setError("");
  }

  function startRename(entry: FileLayoutEntry) {
    if (
      !mutable ||
      savingRef.current ||
      editing ||
      moving ||
      Date.now() < suppressFileClickUntil.current
    )
      return;
    if (blocked) {
      if (navigationPending) {
        setMenu(null);
        setQueuedRename(fileEntryKey(entry));
      }
      return;
    }
    setQueuedRename(null);
    setMenu(null);
    setMoving(null);
    setEditing({
      key: fileEntryKey(entry),
      name: entry.name,
      parent: entry.parent,
    });
    setError("");
  }

  async function save(update: () => FileLayoutEntry[], done?: () => void) {
    if (
      disabled ||
      uploading ||
      readOnly ||
      savingRef.current ||
      !onLayoutChange
    )
      return;
    setError("");
    const submitted = {
      editing,
      moving,
      menu,
      generation: dialogGeneration.current,
    };
    try {
      const next = update();
      savingRef.current = true;
      setSaving(true);
      const accepted = onLayoutChange(next);
      done?.();
      setMenu(null);
      await accepted;
    } catch (reason) {
      if (
        !mountedRef.current ||
        !mutableRef.current ||
        submitted.generation !== dialogGeneration.current
      )
        return;
      const hasEntry = (key: string) =>
        entriesRef.current.some((entry) => fileEntryKey(entry) === key);
      const hasFolder = (id: string | null) =>
        id !== null &&
        entriesRef.current.some(
          (entry) => entry.kind === "folder" && entry.id === id,
        );
      if (
        submitted.editing &&
        (submitted.editing.key === null || hasEntry(submitted.editing.key))
      )
        setEditing({
          ...submitted.editing,
          parent: hasFolder(submitted.editing.parent)
            ? submitted.editing.parent
            : null,
        });
      if (submitted.moving && hasEntry(submitted.moving.key))
        setMoving({
          ...submitted.moving,
          parent: hasFolder(submitted.moving.parent)
            ? submitted.moving.parent
            : null,
        });
      if (submitted.menu && hasEntry(submitted.menu)) setMenu(submitted.menu);
      setError(
        reason instanceof Error
          ? reason.message
          : "Could not save the file layout.",
      );
    } finally {
      savingRef.current = false;
      if (mountedRef.current) setSaving(false);
    }
  }

  function toggleFolder(id: string) {
    setSelectedFolder(id);
    setClosedFolders((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function finishDrag() {
    dragged.current = null;
    suppressFileClickUntil.current = Date.now() + 250;
    setDropTarget(null);
  }

  function acceptRowDrop(
    event: React.DragEvent<HTMLDivElement>,
    key: string,
    isFolder: boolean,
  ) {
    if (!dragged.current) return;
    event.stopPropagation();
    if (!draggable || dragged.current === key) {
      event.dataTransfer.dropEffect = "none";
      setDropTarget(null);
      return;
    }
    event.preventDefault();
    const position = fileDropPosition(event, isFolder);
    setDropTarget((current) =>
      current?.key === key && current.position === position
        ? current
        : { key, position },
    );
    event.dataTransfer.dropEffect = "move";
  }

  function acceptRootDrop(event: React.DragEvent<HTMLDivElement>) {
    if (!draggable || !dragged.current) return;
    event.preventDefault();
    setDropTarget((current) =>
      current?.key === "root" ? current : { key: "root", position: "into" },
    );
    event.dataTransfer.dropEffect = "move";
  }

  function nameEditor() {
    if (!editing) return null;
    return (
      <form
        className="files-sidebar__edit"
        onSubmit={(event) => {
          event.preventDefault();
          const current = editing;
          if (current.type === "file") {
            void createDocument(current);
            return;
          }
          void save(
            () =>
              current.key
                ? renameFileEntry(entries, current.key, current.name)
                : createFileFolder(entries, current.name, current.parent),
            () => {
              setEditing(null);
              if (current.parent)
                setClosedFolders((closed) => {
                  const next = new Set(closed);
                  next.delete(current.parent!);
                  return next;
                });
            },
          );
        }}
      >
        <input
          ref={nameInputRef}
          autoFocus
          aria-label={
            editing.key
              ? "New file name"
              : editing.type === "file"
                ? "File name"
                : "Folder name"
          }
          value={editing.name}
          disabled={blocked}
          maxLength={180}
          onFocus={(event) => {
            if (!editing.key) return;
            const entry = entries.find(
              (item) => fileEntryKey(item) === editing.key,
            );
            const dot = editing.name.lastIndexOf(".");
            event.currentTarget.setSelectionRange(
              0,
              entry?.kind !== "folder" && dot > 0 ? dot : editing.name.length,
            );
          }}
          onBlur={() => {
            if (!savingRef.current && !error) setEditing(null);
          }}
          onChange={(event) =>
            setEditing({ ...editing, name: event.target.value })
          }
          onKeyDown={(event) => {
            if (event.key === "Enter" && !blocked) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
            if (event.key === "Escape" && !saving) {
              event.preventDefault();
              setEditing(null);
              setError("");
            }
          }}
        />
      </form>
    );
  }

  async function createDocument(current: NonNullable<typeof editing>) {
    if (
      blocked ||
      savingRef.current ||
      !canCreateRef.current ||
      !onCreateScript
    )
      return;
    const generation = dialogGeneration.current;
    try {
      const name = normalizeDocumentName(current.name);
      if (
        entriesRef.current.some(
          (entry) =>
            entry.parent === current.parent &&
            fileNameFold(entry.name) === fileNameFold(name),
        )
      )
        throw new Error("This folder already contains an item with that name.");
      setError("");
      savingRef.current = true;
      setSaving(true);
      await onCreateScript(current.parent, name);
      if (mountedRef.current && generation === dialogGeneration.current)
        setEditing(null);
    } catch (reason) {
      if (
        !mountedRef.current ||
        !canCreateRef.current ||
        generation !== dialogGeneration.current
      )
        return;
      setEditing({
        ...current,
        parent: entriesRef.current.some(
          (entry) => entry.kind === "folder" && entry.id === current.parent,
        )
          ? current.parent
          : null,
      });
      setError(
        reason instanceof Error ? reason.message : "Could not create the file.",
      );
    } finally {
      savingRef.current = false;
      if (mountedRef.current) setSaving(false);
    }
  }

  function startName(type?: "file") {
    setMenu(null);
    setMoving(null);
    setEditing({ key: null, name: "", parent, type });
    setError("");
    if (parent)
      setClosedFolders((closed) => {
        const next = new Set(closed);
        next.delete(parent);
        return next;
      });
  }

  function rows(folder: string | null, depth = 0): React.ReactNode {
    const draft =
      editing?.key === null && editing.parent === folder ? (
        <li key="new-entry" className="files-sidebar__item">
          <div
            className="files-sidebar__row"
            style={{ paddingInlineStart: depth * 13 }}
          >
            <FileIcon kind={editing.type === "file" ? "file" : "folder"} />
            {nameEditor()}
          </div>
        </li>
      ) : null;
    const children = entries
      .filter((entry) => entry.parent === folder)
      .map((entry, siblingIndex, siblings) => {
        const key = fileEntryKey(entry);
        const isFolder = entry.kind === "folder";
        const open = isFolder && !closedFolders.has(entry.id);
        const script =
          entry.kind === "script"
            ? visibleScripts.find((item) => item.id === entry.id)
            : undefined;
        const file =
          entry.kind === "dataset"
            ? files.find((item) => item.id === entry.id)
            : undefined;
        return (
          <li
            key={key}
            className={`files-sidebar__item${dropTarget?.key === key && dropTarget.position === "after" ? " files-sidebar__item--drop-after" : ""}`}
          >
            <div
              className={`files-sidebar__row${dropTarget?.key === key ? ` files-sidebar__row--drop-${dropTarget.position}` : ""}`}
              style={{ paddingInlineStart: depth * 13 }}
              draggable={draggable}
              onContextMenu={(event) => {
                if (
                  event.target instanceof Element &&
                  event.target.closest(
                    ".files-sidebar__edit, .files-sidebar__move",
                  )
                )
                  return;
                event.preventDefault();
                event.stopPropagation();
                openMenu(
                  entry,
                  event.currentTarget,
                  event.clientX,
                  event.clientY,
                );
              }}
              onDragStart={(event) => {
                if (
                  !draggable ||
                  (event.target instanceof Element &&
                    event.target.closest(
                      "[data-file-actions], .files-sidebar__edit",
                    ))
                ) {
                  event.preventDefault();
                  return;
                }
                dragged.current = key;
                suppressFileClickUntil.current = Infinity;
                setMenu(null);
                event.dataTransfer.setData("application/x-openecon-file", key);
                event.dataTransfer.setData("text/plain", entry.name);
                event.dataTransfer.effectAllowed = "move";
              }}
              onDragEnd={finishDrag}
              onDragEnter={(event) => acceptRowDrop(event, key, isFolder)}
              onDragOver={(event) => acceptRowDrop(event, key, isFolder)}
              onDragLeave={(event) => {
                if (
                  dropTarget?.key === key &&
                  !(
                    event.relatedTarget instanceof Node &&
                    event.currentTarget.contains(event.relatedTarget)
                  )
                )
                  setDropTarget(null);
              }}
              onDrop={(event) => {
                const source = dragged.current;
                if (!source) return;
                finishDrag();
                event.stopPropagation();
                if (!draggable || source === key) return;
                event.preventDefault();
                const position = fileDropPosition(event, isFolder);
                void save(
                  () =>
                    position === "into"
                      ? moveFileEntry(entries, source, entry.id)
                      : position === "after"
                        ? placeFileAfter(entries, source, key)
                        : placeFileBefore(entries, source, key),
                  () => {
                    if (position === "into")
                      setClosedFolders((closed) => {
                        const next = new Set(closed);
                        next.delete(entry.id);
                        return next;
                      });
                  },
                );
              }}
            >
              {editing?.key === key ? (
                nameEditor()
              ) : (
                <button
                  type="button"
                  className="files-sidebar__file"
                  draggable={draggable}
                  disabled={
                    (blocked && !navigationPending) ||
                    (entry.kind === "script" && !onOpenScript) ||
                    (entry.kind === "dataset" && readOnly && !onDownload)
                  }
                  aria-disabled={blocked || undefined}
                  aria-current={
                    isFolder && parent === entry.id
                      ? "location"
                      : (entry.kind === "script" &&
                            selectedScriptId === entry.id) ||
                          (entry.kind === "dataset" &&
                            activeFileId === entry.id)
                        ? "page"
                        : undefined
                  }
                  aria-expanded={isFolder ? open : undefined}
                  aria-haspopup={
                    mutable || (file && onDownload) ? "menu" : undefined
                  }
                  onClick={(event) => {
                    if (blocked) return;
                    if (event.detail > 1) return;
                    if (Date.now() < suppressFileClickUntil.current) return;
                    if (isFolder) toggleFolder(entry.id);
                    else if (script)
                      onOpenScript?.({ ...script, name: entry.name });
                    else if (file && !readOnly) onOpen(file);
                  }}
                  onDoubleClick={() => startRename(entry)}
                  onKeyDown={(event) => {
                    if (event.key === "F2") {
                      event.preventDefault();
                      startRename(entry);
                    }
                    if (
                      event.key === "ContextMenu" ||
                      (event.key === "F10" && event.shiftKey)
                    ) {
                      event.preventDefault();
                      const bounds =
                        event.currentTarget.getBoundingClientRect();
                      openMenu(
                        entry,
                        event.currentTarget.closest(
                          ".files-sidebar__row",
                        ) as HTMLElement,
                        bounds.left + 12,
                        bounds.bottom,
                      );
                    }
                  }}
                  title={entry.name}
                >
                  {isFolder && (
                    <span
                      className={`files-sidebar__chevron${open ? " files-sidebar__chevron--open" : ""}`}
                    >
                      <FileIcon kind="chevron" />
                    </span>
                  )}
                  <FileIcon
                    kind={
                      isFolder
                        ? "folder"
                        : entry.kind === "script" &&
                            documentExtension(entry.name) === ".py"
                          ? "code"
                          : "file"
                    }
                  />
                  <span>{entry.name}</span>
                </button>
              )}
              {menu === key &&
                editing?.key !== key &&
                (mutable || (file && onDownload)) && (
                  <div
                    ref={menuRef}
                    className="files-sidebar__menu"
                    data-file-actions
                    role="menu"
                    aria-label={`Actions for ${entry.name}`}
                    style={menuPosition}
                    onKeyDown={(event) => {
                      if (event.key === "Escape") {
                        event.preventDefault();
                        closeMenu(true);
                        event.stopPropagation();
                      }
                      if (event.key === "Tab") closeMenu(true);
                      if (
                        ["ArrowDown", "ArrowUp", "Home", "End"].includes(
                          event.key,
                        )
                      ) {
                        event.preventDefault();
                        const items = [
                          ...event.currentTarget.querySelectorAll<HTMLButtonElement>(
                            "button:not(:disabled)",
                          ),
                        ];
                        const index = items.indexOf(
                          document.activeElement as HTMLButtonElement,
                        );
                        const next =
                          event.key === "Home"
                            ? 0
                            : event.key === "End"
                              ? items.length - 1
                              : (index +
                                  (event.key === "ArrowDown" ? 1 : -1) +
                                  items.length) %
                                items.length;
                        items[next]?.focus();
                      }
                    }}
                  >
                    {mutable && (
                      <>
                        <button
                          type="button"
                          role="menuitem"
                          tabIndex={-1}
                          disabled={blocked}
                          onClick={() => startRename(entry)}
                        >
                          Rename
                        </button>
                        <button
                          type="button"
                          role="menuitem"
                          tabIndex={-1}
                          disabled={blocked}
                          onClick={() => {
                            setMoving({ key, parent: entry.parent });
                            setMenu(null);
                          }}
                        >
                          Move
                        </button>
                        <button
                          type="button"
                          role="menuitem"
                          tabIndex={-1}
                          disabled={blocked || siblingIndex === 0}
                          onClick={() =>
                            void save(() => reorderFileEntry(entries, key, -1))
                          }
                        >
                          Move up
                        </button>
                        <button
                          type="button"
                          role="menuitem"
                          tabIndex={-1}
                          disabled={
                            blocked || siblingIndex === siblings.length - 1
                          }
                          onClick={() =>
                            void save(() => reorderFileEntry(entries, key, 1))
                          }
                        >
                          Move down
                        </button>
                      </>
                    )}
                    {file && onDownload && (
                      <button
                        type="button"
                        role="menuitem"
                        tabIndex={-1}
                        disabled={blocked}
                        onClick={() => {
                          closeMenu(true);
                          onDownload({ ...file, name: entry.name });
                        }}
                      >
                        Download file
                      </button>
                    )}
                  </div>
                )}
            </div>
            {isFolder && open && depth < 16 && (
              <ul className="files-sidebar__list">
                {rows(entry.id, depth + 1)}
              </ul>
            )}
          </li>
        );
      });
    return [draft, ...children];
  }

  return (
    <aside
      className={`files-sidebar${collapsed ? " files-sidebar--collapsed" : ""}`}
      aria-label="Files"
    >
      <div className="files-sidebar__header">
        <button
          type="button"
          className="files-sidebar__icon"
          aria-label={toggleLabel}
          title={toggleLabel}
          aria-expanded={!collapsed}
          aria-controls={contentId}
          onClick={() => onCollapsedChange(!collapsed)}
        >
          <FileIcon kind={collapsed ? "folder" : "collapse"} />
        </button>
        {!collapsed && (
          <>
            <h2>
              Files
              {pendingSync && (
                <span
                  className="files-sidebar__pending"
                  title="The file layout has not been saved to the cloud yet."
                >
                  Local
                </span>
              )}
            </h2>
            {mutable && (
              <SidebarAction
                label="New folder"
                kind="new-folder"
                disabled={blocked}
                onClick={() => startName()}
              />
            )}
            {!readOnly && onCreateScript && (
              <SidebarAction
                label="New file"
                kind="new-code"
                disabled={blocked}
                onClick={() => startName("file")}
              />
            )}
            {!readOnly && onImportScript && (
              <SidebarAction
                label="Import file"
                kind="import-code"
                disabled={blocked}
                onClick={() => onImportScript(parent)}
              />
            )}
            {!readOnly && onUpload && (
              <SidebarAction
                label={
                  uploading ? "Uploading data file" : "Add data file"
                }
                kind="add-data"
                disabled={blocked || uploading}
                onClick={() => onUpload(parent)}
              />
            )}
          </>
        )}
      </div>
      <div
        ref={contentRef}
        id={contentId}
        className={`files-sidebar__content${dropTarget?.key === "root" ? " files-sidebar__content--drop" : ""}`}
        hidden={collapsed}
        onDragEnter={acceptRootDrop}
        onDragOver={acceptRootDrop}
        onDrop={(event) => {
          const source = dragged.current;
          if (!source) return;
          finishDrag();
          if (!draggable) return;
          event.preventDefault();
          void save(() => moveFileEntry(entries, source, null));
        }}
      >
        {moving && (
          <form
            className="files-sidebar__move"
            onSubmit={(event) => {
              event.preventDefault();
              const current = moving;
              void save(
                () => moveFileEntry(entries, current.key, current.parent),
                () => {
                  setMoving(null);
                  if (current.parent)
                    setClosedFolders((closed) => {
                      const next = new Set(closed);
                      next.delete(current.parent!);
                      return next;
                    });
                },
              );
            }}
          >
            <label>
              Folder
              <select
                autoFocus
                aria-label="Destination folder"
                value={moving.parent ?? ""}
                disabled={blocked}
                onChange={(event) =>
                  setMoving({ ...moving, parent: event.target.value || null })
                }
              >
                <option value="">Files</option>
                {destinations.map((folder) => (
                  <option key={folder.id} value={folder.id}>
                    {fileFolderPath(entries, folder.id)}
                  </option>
                ))}
              </select>
            </label>
            <div>
              <button type="submit" disabled={blocked}>
                Move
              </button>
              <button
                type="button"
                disabled={saving}
                onClick={() => {
                  setMoving(null);
                  setError("");
                }}
              >
                Cancel
              </button>
            </div>
          </form>
        )}
        {error && (
          <p className="files-sidebar__error" role="alert">
            {error}
          </p>
        )}
        {entries.length || editing?.key === null ? (
          <ul className="files-sidebar__list">{rows(null)}</ul>
        ) : (
          <p className="files-sidebar__empty">No files.</p>
        )}
        {selectedFolder && (
          <button
            type="button"
            className="files-sidebar__root"
            disabled={blocked}
            onClick={() => setSelectedFolder(null)}
            title="Create new files in the root folder"
          >
            Root folder
          </button>
        )}
        {footer}
      </div>
      <span className="files-sidebar__status" role="status">
        {saving
          ? "Saving file layout."
          : uploading
            ? "Uploading file."
            : ""}
      </span>
    </aside>
  );
}
