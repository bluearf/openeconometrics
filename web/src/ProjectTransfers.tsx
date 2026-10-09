import { useEffect, useRef, useState } from "react";
import type { DatasetProfile } from "./types";
import type { ProjectTransferActions } from "./project-transfers";
import "./project-transfers.css";
export default function ProjectTransfers({
  actions,
  readOnly,
  disabled,
  onReady,
}: {
  actions: ProjectTransferActions;
  readOnly: boolean;
  disabled: boolean;
  onReady: (file: DatasetProfile) => void | Promise<void>;
}) {
  const [snapshot, setSnapshot] = useState(() => actions.getSnapshot());
  const mounted = useRef(true);
  const activeActions = useRef(actions);
  activeActions.current = actions;
  useEffect(() => {
    mounted.current = true;
    const unsubscribe = actions.subscribe(setSnapshot);
    const refresh = () => {
      void actions.refresh().catch(() => {});
    };
    refresh();
    window.addEventListener("online", refresh);
    return () => {
      mounted.current = false;
      unsubscribe();
      window.removeEventListener("online", refresh);
    };
  }, [actions]);
  useEffect(() => {
    const id = snapshot.busyRequestId;
    if (!id || !snapshot.transfers.some((item) => item.request_id === id))
      return;
    let reading = false;
    const timer = window.setInterval(() => {
      if (reading) return;
      reading = true;
      void actions
        .status(id)
        .catch(() => {})
        .finally(() => {
          reading = false;
        });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [actions, snapshot.busyRequestId, snapshot.transfers]);
  async function resume(id: string) {
    try {
      const file = await actions.resume(id);
      if (file && mounted.current && activeActions.current === actions)
        await onReady(file);
    } catch {
      /* Safe errors are already in the controller snapshot. */
    }
  }
  if (!snapshot.transfers.length) return null;
  return (
    <section className="project-transfers" aria-label="Pending data sharing">
      <h3>Bekleyen paylaşımlar</h3>
      {snapshot.error && <p role="alert">{snapshot.error}</p>}
      {snapshot.transfers.map((item) => (
        <div className="project-transfer" key={item.request_id}>
          <strong title={item.name}>{item.name}</strong>
          <progress
            aria-label={`Sharing ${item.name}`}
            max={item.size_bytes}
            value={item.acknowledged_bytes}
          />
          <span>
            {Math.floor((100 * item.acknowledged_bytes) / item.size_bytes)}% ·{" "}
            {item.state === "cancel_pending"
              ? "İptal bekliyor"
              : snapshot.busyRequestId === item.request_id
                ? "Paylaşılıyor…"
                : item.state === "ready"
                  ? "Paylaşıldı; açılmaya hazır"
                  : "Bekliyor"}
          </span>
          {!readOnly && (
            <div>
              <button
                disabled={
                  disabled ||
                  !!snapshot.busyRequestId ||
                  item.state === "cancel_pending"
                }
                onClick={() => void resume(item.request_id)}
              >
                Devam et
              </button>
              <button
                onClick={() =>
                  void actions.cancel(item.request_id).catch(() => {})
                }
              >
                {item.state === "cancel_pending" ? "İptali tamamla" : "İptal"}
              </button>
            </div>
          )}
        </div>
      ))}
    </section>
  );
}
