/** Retain full notices for the locked production dependency graph in every web build. */
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const web = join(root, "web");
const lockBytes = readFileSync(join(web, "package-lock.json"));
const lock = JSON.parse(lockBytes);
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const packages = [];
const sections = [
  "OpenEconometrics web interface — third-party notices",
  "Generated from the locked production dependency graph; this conservative inventory also includes dependencies that bundling may omit.",
  "Each upstream license remains in force independently of OpenEconometrics' Apache-2.0 license.",
];
for (const [path, entry] of Object.entries(lock.packages).sort(([a], [b]) => a.localeCompare(b))) {
  if (!path || entry.dev) continue;
  const directory = join(web, path);
  const installed = JSON.parse(readFileSync(join(directory, "package.json")));
  if (installed.version !== entry.version) throw new Error(`Locked version differs for ${path}`);
  let files = readdirSync(directory, { withFileTypes: true })
    .filter((item) => item.isFile() && /^(license|copying|notice)(\.|$|-)/i.test(item.name))
    .map((item) => ({ file: item.name, bytes: readFileSync(join(directory, item.name)) }));
  // Firebase npm modules omit their repository-root license. Retain the exact
  // upstream text from the pinned Firebase release, rather than invent a grant.
  if (!files.length && (installed.name === "firebase" || installed.name.startsWith("@firebase/")) && entry.license === "Apache-2.0") {
    files = [{ file: "upstream:FIREBASE-LICENSE.txt", bytes: readFileSync(join(web, "public/licenses/FIREBASE-LICENSE.txt")) }];
  }
  if (!files.length) throw new Error(`Missing upstream notice for ${installed.name}`);
  packages.push({ name: installed.name, version: entry.version, license: entry.license ?? installed.license,
    lock_integrity: entry.integrity ?? null, notices: files.map(({ file, bytes }) => ({ file, sha256: hash(bytes) })) });
  sections.push(`\n${"=".repeat(72)}\n${installed.name} ${entry.version}\nLicense: ${entry.license ?? installed.license}`);
  for (const { file, bytes } of files) sections.push(`\n--- ${file} ---\n${bytes.toString("utf8").replace(/\r\n?/g, "\n").trimEnd()}`);
}
// The font files carry their own OFL grant. The font-build repository's root
// MIT license does not replace that embedded notice or reserved font names.
const fontDirectory = join(web, "node_modules/katex/dist/fonts");
const fontNotices = new Set();
const fonts = [];
for (const file of readdirSync(fontDirectory).filter((name) => name.endsWith(".ttf")).sort()) {
  const bytes = readFileSync(join(fontDirectory, file));
  const count = bytes.readUInt16BE(4);
  let offset;
  for (let i = 0; i < count; i++) {
    const start = 12 + 16 * i;
    if (bytes.toString("ascii", start, start + 4) === "name") offset = bytes.readUInt32BE(start + 8);
  }
  if (offset === undefined) throw new Error(`Missing font name table: ${file}`);
  const records = bytes.readUInt16BE(offset + 2), storage = bytes.readUInt16BE(offset + 4);
  let notice;
  for (let i = 0; i < records; i++) {
    const start = offset + 6 + i * 12;
    const platform = bytes.readUInt16BE(start), name = bytes.readUInt16BE(start + 6);
    if (name !== 13) continue;
    const size = bytes.readUInt16BE(start + 8), position = bytes.readUInt16BE(start + 10);
    const raw = bytes.subarray(offset + storage + position, offset + storage + position + size);
    notice = platform === 0 || platform === 3 ? Buffer.from(raw).swap16().toString("utf16le") : raw.toString("latin1");
  }
  if (!notice?.includes("SIL Open Font License, Version 1.1")) throw new Error(`Unreviewed font license: ${file}`);
  fontNotices.add(notice);
  fonts.push({ file, sha256: hash(bytes), embedded_notice_sha256: hash(notice), license: "OFL-1.1" });
}
const ofl = readFileSync(join(web, "public/assets/BARLOW-OFL.txt"), "utf8");
const fontText = "KaTeX font notices extracted verbatim from the installed locked TTF name tables.\nThe unmodified WOFF/WOFF2 variants come from the same locked KaTeX distribution.\n\n"
  + [...fontNotices].sort().join("\n\n") + "\n\n" + ofl.slice(ofl.indexOf("SIL OPEN FONT LICENSE Version"));
const manifest = { schema_version: 1, package_lock_sha256: hash(lockBytes), scope: "all locked non-dev npm dependencies; conservative superset of bundled modules", packages, fonts };
const outputs = [
  ["THIRD-PARTY-NOTICES.txt", sections.join("\n") + "\n"],
  ["third-party-manifest.json", JSON.stringify(manifest, null, 2) + "\n"],
  ["licenses/KATEX-FONTS-LICENSE.txt", fontText],
];
for (const [name, value] of outputs) {
  const path = join(web, "public", name);
  if (process.argv.includes("--check")) {
    if (readFileSync(path, "utf8") !== value) throw new Error(`Stale ${name}; run the web build`);
  } else writeFileSync(path, value);
}
console.log(`Retained full third-party notices for ${packages.length} locked production packages.`);
