import { useEffect, useRef, useState, type ReactNode } from "react";

let renderer: Promise<typeof import("./latex-render")> | null = null;
function loadRenderer() {
  renderer ??= import("./latex-render").catch((error: unknown) => {
    renderer = null;
    throw error;
  });
  return renderer;
}

export default function LatexView({
  source,
  fallback,
}: {
  source: string;
  fallback?: ReactNode;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  useEffect(() => {
    let active = true;
    setState("loading");
    const element = host.current;
    element?.replaceChildren();
    void loadRenderer()
      .then(({ renderLatexMath }) => {
        if (!active || !element) return;
        renderLatexMath(element, source);
        setState("ready");
      })
      .catch(() => {
        if (!active) return;
        element?.replaceChildren();
        setState("error");
      });
    return () => {
      active = false;
      element?.replaceChildren();
    };
  }, [source]);
  return (
    <>
      <div
        ref={host}
        className="latex-preview"
        hidden={state !== "ready"}
        aria-label="LaTeX result"
      />
      {state !== "ready" &&
        (fallback ?? <pre className="latex-source">{source}</pre>)}
    </>
  );
}
