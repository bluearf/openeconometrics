import { useEffect, useRef, useState } from "react";
import { ApiError } from "./api";
import "./project-transfers.css";
export default function ProjectFileDownload({
  file,
  cancel,
}: {
  file: { id: string; name: string };
  cancel: (id: string) => Promise<void>;
}) {
  const [busy, setBusy] = useState(false),
    [notice, setNotice] = useState("");
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  async function stop() {
    if (busy) return;
    setBusy(true);
    setNotice("");
    try {
      await cancel(file.id);
      if (mounted.current)
        setNotice(
          "Cancellation requested. Existing local files are preserved.",
        );
    } catch (error) {
      if (mounted.current)
        setNotice(
          error instanceof ApiError && [401, 403].includes(error.status)
            ? "Could not verify access. Reopen the project."
            : "Could not cancel the download. Check your connection and try again.",
        );
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  return (
    <section className="project-transfers" aria-label="Active file download">
      <h3>İndiriliyor</h3>
      <p title={file.name}>{file.name}</p>
      <button
        className="project-download-cancel"
        disabled={busy}
        onClick={() => void stop()}
      >
        {busy ? "İptal ediliyor…" : "İptal"}
      </button>
      {notice && <p role="status">{notice}</p>}
    </section>
  );
}
