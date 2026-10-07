import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import {
  clampPane,
  paneStorageKey,
  PANE_DEFAULTS,
  readPanePreference,
  savePanePreference,
  type PaneBounds,
  type PaneKind,
} from "./pane-preferences";
import "./pane-resize.css";

export function usePaneExtent(
  ref: RefObject<HTMLElement | null>,
  dimension: "width" | "height",
  parent = false,
): number {
  const [extent, setExtent] = useState(0);
  useLayoutEffect(() => {
    const element = parent ? ref.current?.parentElement : ref.current;
    if (!element) return;
    const measure = () => setExtent(element.getBoundingClientRect()[dimension]);
    measure();
    const observer =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(measure)
        : null;
    observer?.observe(element);
    window.addEventListener("resize", measure);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [ref, dimension, parent]);
  return extent;
}

export function usePanePreference(
  projectId: string | null | undefined,
  kind: PaneKind,
  bounds: PaneBounds,
) {
  const key = paneStorageKey(projectId, kind);
  const [preference, setPreference] = useState(() => ({
    key,
    value: readPanePreference(key, kind),
  }));
  const current =
    preference.key === key ? preference.value : readPanePreference(key, kind);
  const value = clampPane(current, bounds);
  const change = useCallback(
    (next: number) => setPreference({ key, value: next }),
    [key],
  );
  const commit = useCallback(
    (next: number) => {
      setPreference({ key, value: next });
      savePanePreference(key, next);
    },
    [key],
  );
  return {
    value,
    change,
    commit,
    cancel: () => setPreference({ key, value: current }),
    defaultValue: PANE_DEFAULTS[kind],
  };
}

export interface ResizeSeparatorProps extends PaneBounds {
  label: string;
  orientation: "vertical" | "horizontal";
  value: number;
  defaultValue: number;
  step?: number;
  unitsPerPixel?: number;
  direction?: 1 | -1;
  valueScale?: number;
  onChange: (value: number) => void;
  onCommit: (value: number) => void;
  onCancel?: () => void;
}

export function ResizeSeparator(props: ResizeSeparatorProps) {
  const latest = useRef(props);
  latest.current = props;
  const [dragging, setDragging] = useState(false);
  const cleanup = useRef<(() => void) | null>(null);
  useEffect(() => () => cleanup.current?.(), []);

  function bounded(next: number): number {
    return clampPane(next, latest.current);
  }

  return (
    <div
      className={`pane-separator pane-separator--${props.orientation}${dragging ? " pane-separator--dragging" : ""}`}
      role="separator"
      tabIndex={0}
      aria-label={props.label}
      aria-orientation={props.orientation}
      aria-valuemin={Math.round(props.minimum * (props.valueScale ?? 1))}
      aria-valuemax={Math.round(props.maximum * (props.valueScale ?? 1))}
      aria-valuenow={Math.round(props.value * (props.valueScale ?? 1))}
      title={props.label}
      onDoubleClick={() => {
        cleanup.current?.();
        const value = bounded(latest.current.defaultValue);
        latest.current.onChange(value);
        latest.current.onCommit(value);
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          cleanup.current?.();
          return;
        }
        const horizontal = latest.current.orientation === "horizontal";
        const previous = horizontal ? "ArrowUp" : "ArrowLeft";
        const next = horizontal ? "ArrowDown" : "ArrowRight";
        let value: number;
        if (event.key === "Home") value = latest.current.minimum;
        else if (event.key === "End") value = latest.current.maximum;
        else if (event.key === previous || event.key === next) {
          value =
            latest.current.value +
            (event.key === next ? 1 : -1) *
              (latest.current.direction ?? 1) *
              (latest.current.step ?? 10) *
              (event.shiftKey ? 5 : 1);
        } else return;
        event.preventDefault();
        cleanup.current?.();
        value = bounded(value);
        latest.current.onChange(value);
        latest.current.onCommit(value);
      }}
      onPointerDown={(event) => {
        if (event.button !== 0 || event.isPrimary === false || cleanup.current)
          return;
        event.preventDefault();
        const target = event.currentTarget;
        target.focus({ preventScroll: true });
        const initial = latest.current.value;
        const restorePreference = latest.current.onCancel;
        const origin =
          latest.current.orientation === "vertical"
            ? event.clientX
            : event.clientY;
        const pointerId = event.pointerId;
        const previousCursor = document.body.style.cursor;
        const previousSelect = document.body.style.userSelect;
        let pending: number | null = null;
        let frame: number | null = null;
        let last = initial;
        let finished = false;
        const apply = () => {
          frame = null;
          if (pending === null || finished) return;
          last = bounded(pending);
          pending = null;
          latest.current.onChange(last);
        };
        const move = (movement: PointerEvent) => {
          if (movement.pointerId !== pointerId) return;
          movement.preventDefault();
          const coordinate =
            latest.current.orientation === "vertical"
              ? movement.clientX
              : movement.clientY;
          pending =
            initial +
            (coordinate - origin) *
              (latest.current.unitsPerPixel ?? 1) *
              (latest.current.direction ?? 1);
          if (frame === null) frame = requestAnimationFrame(apply);
        };
        const finish = (cancel: boolean) => {
          if (finished) return;
          if (frame !== null) {
            cancelAnimationFrame(frame);
            frame = null;
          }
          if (!cancel) apply();
          finished = true;
          document.removeEventListener("pointermove", move, true);
          document.removeEventListener("pointerup", up, true);
          document.removeEventListener("pointercancel", cancelled, true);
          document.removeEventListener("keydown", escape);
          window.removeEventListener("blur", cancelled);
          target.removeEventListener("lostpointercapture", cancelled);
          try {
            if (target.hasPointerCapture?.(pointerId))
              target.releasePointerCapture(pointerId);
          } catch {
            /* The browser may already have released a cancelled pointer. */
          }
          document.body.style.cursor = previousCursor;
          document.body.style.userSelect = previousSelect;
          cleanup.current = null;
          setDragging(false);
          if (cancel && restorePreference) restorePreference();
          else latest.current.onChange(cancel ? bounded(initial) : last);
          if (!cancel) latest.current.onCommit(last);
        };
        const up = (release: PointerEvent) => {
          if (release.pointerId !== pointerId) return;
          move(release);
          finish(false);
        };
        const cancelled = (cancellation?: Event) => {
          if (
            cancellation &&
            "pointerId" in cancellation &&
            (cancellation as PointerEvent).pointerId !== pointerId
          )
            return;
          finish(true);
        };
        const escape = (key: KeyboardEvent) => {
          if (key.key === "Escape") {
            key.preventDefault();
            finish(true);
          }
        };
        cleanup.current = cancelled;
        document.body.style.cursor =
          latest.current.orientation === "vertical"
            ? "col-resize"
            : "row-resize";
        document.body.style.userSelect = "none";
        setDragging(true);
        document.addEventListener("pointermove", move, {
          passive: false,
          capture: true,
        });
        document.addEventListener("pointerup", up, true);
        document.addEventListener("pointercancel", cancelled, true);
        document.addEventListener("keydown", escape);
        window.addEventListener("blur", cancelled);
        target.addEventListener("lostpointercapture", cancelled);
        try {
          target.setPointerCapture?.(pointerId);
        } catch {
          /* Document listeners retain the drag in older webviews. */
        }
      }}
    />
  );
}
