import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

function writer() {
  const root: any = { atob: (value: string) => Buffer.from(value, 'base64').toString('binary') };
  const context = vm.createContext({ window: root, TextEncoder, Uint8Array, DataView });
  for (const name of ['network-font.js', 'network-pdf.js']) {
    vm.runInContext(readFileSync(new URL('../../packages/openecon-charts/src/openecon_charts/assets/' + name, import.meta.url), 'utf8'), context);
  }
  return root.OpenEconNetworkPDF;
}
function scene(overrides: any = {}) {
  return { width: 400, height: 300, title: 'Araştırma ağı', coverage: '2 nodes · 1 edge · Directed', directed: true,
    transform: { k: 1, x: 20, y: 20 },
    nodes: [{ id: 0, x: 50, y: 100, radius: 6, color: '#123', highlighted: true }, { id: 1, x: 250, y: 100, radius: 8, color: '#315f91' }],
    edges: [{ source: 0, target: 1, x: 50, y: 100, tx: 250, ty: 100, color: '#456', width: 2 }],
    labels: [{ text: 'Öğrenci', x: 60, y: 100 }], annotations: [{ text: 'İlişki', x: 100, y: 170 }],
    legend: [{ label: 'Eğitim', color: '#123' }], ...overrides };
}
test('vector PDF embeds Unicode font, complete xref, legend, selected ring and directed edges', () => {
  const output = writer().bytes(scene()), text = Buffer.from(output).toString('latin1');
  assert.ok(text.startsWith('%PDF-1.7'));
  assert.match(text, /\/FontFile2 \d+ 0 R/); assert.match(text, /\/ToUnicode \d+ 0 R/);
  assert.match(text, /<\w{4}> <0131>/); // dotless i, no ASCII substitution
  assert.match(text, /<\w{4}> <011F>/); // g breve
  assert.ok(text.includes('18 372 8 8 re f')); // legend swatch after coverage
  assert.match(text, /2 w .* S Q/); // selected ring
  assert.match(text, /l h f/); // directed arrow polygon
  assert.doesNotMatch(text, /\/Subtype \/Image|\/JavaScript|\/OpenAction|\/URI/);
  const xref = Number(/startxref\n(\d+)/.exec(text)?.[1]);
  assert.equal(text.slice(xref, xref + 4), 'xref');
  const entries = text.slice(xref).split('\n').slice(3);
  for (let i = 0; /^\d{10} 00000 n /.test(entries[i] || ''); i++) {
    assert.ok(text.slice(Number(entries[i].slice(0, 10))).startsWith(`${i + 1} 0 obj\n`));
  }
});
test('PDF wraps long title and legend to actual font width and expands page', () => {
  const pdf = writer(), short = Buffer.from(pdf.bytes(scene())).toString('latin1');
  const long = Buffer.from(pdf.bytes(scene({ title: 'Araştırma '.repeat(50), legend: [{ label: 'Uzun açıklama '.repeat(50), color: '#123' }] }))).toString('latin1');
  const pageHeight = (source: string) => Number(/\/MediaBox \[0 0 400 (\d+(?:\.\d+)?)\]/.exec(source)?.[1]);
  assert.ok(pageHeight(long) > pageHeight(short) + 200);
  assert.ok((long.match(/18 \d+ Tm/g) || []).length > 10);
});
test('PDF rejects unsupported glyphs, nonfinite viewport and oversized vector scene explicitly', () => {
  const pdf = writer();
  assert.throws(() => pdf.bytes(scene({ title: '漢' })), /export SVG\/PNG/);
  assert.throws(() => pdf.bytes(scene({ transform: { k: NaN, x: 0, y: 0 } })), /viewport/);
  assert.throws(() => pdf.bytes(scene({ nodes: Array(200001).fill({}) })), /200,000/);
});
