/** Native text exports use a Save panel; browser exports retain Blob downloads. */
import { isDesktop, nativeCoreInvoke } from "./native-core.ts";

export const MAX_TEXT_EXPORT_BYTES = 32 * 1024 * 1024;
export type TextExportReceipt = {
  status: "saved" | "cancelled" | "browser" | "failed";
  path?: string | null;
  bytes?: number;
  sha256?: string | null;
};

export async function saveTextExport(
  content: string,
  filename: string,
  type = "application/json",
): Promise<TextExportReceipt> {
  if (
    !filename ||
    new TextEncoder().encode(filename).length > 240 ||
    filename.trim() !== filename ||
    [".", ".."].includes(filename) ||
    filename.endsWith(".") ||
    /[\x00-\x1f\x7f/\\:]/.test(filename)
  )
    throw new Error("Choose a valid export filename without a directory path.");
  if (isDesktop()) {
    const bytes = new TextEncoder().encode(content);
    if (bytes.length > MAX_TEXT_EXPORT_BYTES)
      throw new Error(
        "This text export exceeds the 32 MiB desktop export limit.",
      );
    const receipt = await nativeCoreInvoke<TextExportReceipt>(
      "save_text_export",
      { content, filename },
    );
    if (
      receipt?.status === "cancelled" &&
      !receipt.path &&
      !receipt.sha256 &&
      receipt.bytes === 0
    )
      return receipt;
    if (
      receipt?.status !== "saved" ||
      typeof receipt.path !== "string" ||
      !receipt.path ||
      receipt.bytes !== bytes.length ||
      !/^[a-f0-9]{64}$/.test(receipt.sha256 ?? "")
    )
      throw new Error("The desktop app could not verify the saved export.");
    const digest = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
      (b) => b.toString(16).padStart(2, "0"),
    ).join("");
    if (digest !== receipt.sha256)
      throw new Error("The saved export does not match its source.");
    return receipt;
  }
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.append(anchor);
  try {
    anchor.click();
  } finally {
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return { status: "browser" };
}
