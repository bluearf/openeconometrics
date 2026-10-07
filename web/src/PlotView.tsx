import { useEffect, useRef, useState } from "react";
import type { WorkspaceClient } from "./api.ts";
import { resolvePlot } from "./plot-artifacts.ts";

export interface PlotSpec {
  kind: "scatter" | "line" | "hist" | "histogram" | "coefficients" | "d3" | "network";
  title: string;
  x_label: string;
  y_label: string;
  data: Record<string, number | string | null>[];
  sample_n: number;
  total_n: number;
  dropped_n: number;
  config?: Record<string, unknown> | null;
  artifact?: { version: 1; id: string; bytes: number };
}
interface ChartRenderer {
  mount(host: HTMLElement, spec: PlotSpec): Promise<unknown>;
  unmount(host: HTMLElement): void;
}
declare global {
  interface Window {
    OpenEconCharts?: ChartRenderer;
  }
}
let loading: Promise<ChartRenderer> | null = null;
function asset(tag: "script" | "link", filename: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const node = document.createElement(tag);
    const url = `/chart-assets/${filename}?v=0.3.0a1-network-workbench`;
    if (node instanceof HTMLScriptElement) node.src = url;
    else {
      node.rel = "stylesheet";
      node.href = url;
    }
    const timer = window.setTimeout(() => fail(), 20000);
    function fail() {
      window.clearTimeout(timer);
      node.remove();
      reject(new Error("Could not load the chart component. Try again."));
    }
    node.onload = () => {
      window.clearTimeout(timer);
      resolve();
    };
    node.onerror = fail;
    document.head.append(node);
  });
}
function renderer(): Promise<ChartRenderer> {
  if (!loading)
    loading = (async () => {
      await Promise.all([
        asset("link", "charts.css"),
        asset("script", "d3.min.js"),
        asset("link", "network.css"),
      ]);
      await asset("script", "network-webgl.js");
      await asset("script", "network-font.js");
      await asset("script", "network-pdf.js");
      await asset("script", "network-renderer.js");
      await asset("script", "renderer.js");
      if (!window.OpenEconCharts)
        throw new Error("Could not initialize the chart component.");
      return window.OpenEconCharts;
    })().catch((error) => {
      loading = null;
      throw error;
    });
  return loading;
}
export default function PlotView({ plot, client }: { plot: PlotSpec; client?: WorkspaceClient }) {
  const host = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const element = host.current;
    if (!element) return;
    let active = true;
    const abort = new AbortController();
    setError(null);
    setReady(false);
    Promise.all([renderer(), resolvePlot(plot, client, abort.signal)])
      .then(async ([charts, actual]) => {
        if (!active) return;
        await charts.mount(element, actual);
        if (active) setReady(true);
      })
      .catch((reason: unknown) => {
        if (active)
          setError(
            reason instanceof Error ? reason.message : "Could not create the chart.",
          );
      });
    return () => {
      active = false;
      abort.abort();
      window.OpenEconCharts?.unmount(element);
    };
  }, [plot, client, attempt]);
  return (
    <div className="plot-view">
      {!ready && !error && (
        <p className="chart-loading" role="status">
          Preparing chart…
        </p>
      )}
      {error && (
        <div className="chart-error" role="alert">
          <p>{error}</p>
          <button
            className="subtle-button"
            onClick={() => setAttempt((n) => n + 1)}
          >
            Try again
          </button>
        </div>
      )}
      <div ref={host} className="d3-chart-host" />
    </div>
  );
}
