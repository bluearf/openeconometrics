import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const asset = (name: string) => fileURLToPath(new URL(`../../packages/openecon-charts/src/openecon_charts/assets/${name}`, import.meta.url));
const d3Code = readFileSync(asset('d3.min.js'), 'utf8');
const rendererCode = readFileSync(asset('renderer.js'), 'utf8');

function setup() {
  const dom = new JSDOM('<!doctype html><html><body><main><div id="chart"></div></main></body></html>', {
    runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://localhost:8765/',
  });
  const { window } = dom;
  let observed = 0;
  const observers: { active: boolean; callback: () => void }[] = [];
  window.ResizeObserver = class {
    active = false;
    callback: () => void;
    constructor(callback: () => void) { this.callback = callback; observers.push(this); }
    observe() { if (!this.active) { observed++; this.active = true; } }
    disconnect() { if (this.active) { observed--; this.active = false; } }
  };
  window.matchMedia = () => ({ matches: true });
  window.HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
  window.HTMLDialogElement.prototype.close = function () {
    this.removeAttribute('open'); this.dispatchEvent(new window.Event('close'));
  };
  window.eval(d3Code);
  window.eval(rendererCode);
  const host = window.document.querySelector('#chart');
  Object.defineProperty(host, 'clientWidth', { value: 640, configurable: true });
  const resize = async (width: number) => {
    Object.defineProperty(host, 'clientWidth', { value: width, configurable: true });
    host.getBoundingClientRect = () => ({ width, height: 360, left: 0, top: 0, right: width, bottom: 360 });
    observers.filter(observer => observer.active).forEach(observer => observer.callback());
    await new Promise(resolve => window.setTimeout(resolve, 40));
  };
  return { dom, window, host, api: window.OpenEconCharts, observed: () => observed, resize };
}

function numeric(kind: string, data: unknown[], extra = {}) {
  return { kind, title: 'Örnek grafik', x_label: 'Eğitim', y_label: 'Ücret', data,
    sample_n: data.length, total_n: data.length, dropped_n: 0, ...extra };
}
function categorical(type = 'area', extra = {}) {
  return { kind: 'd3', title: 'Dönemler', x_label: 'Yıl', y_label: 'Değer', data: [],
    sample_n: 4, total_n: 4, dropped_n: 0,
    config: { type, categories: ['2020', '2022', '2023', '2024'],
      series: [{ id: 'sales', name: 'Satış', values: [1, null, -2, 0] }], ...extra } };
}

test('vendored D3 draws real numeric negative coordinates without matrix thresholds', async () => {
  const env = setup();
  try {
    assert.equal(env.window.d3.version, '7.9.0');
    await env.api.mount(env.host, numeric('scatter', [{ x: -100, y: -12 }, { x: 0, y: 0 }, { x: 50, y: 9 }]));
    const points = [...env.host.querySelectorAll('.numeric-point')];
    assert.equal(points.length, 3);
    const xs = points.map(p => Number(p.getAttribute('cx')));
    const ys = points.map(p => Number(p.getAttribute('cy')));
    assert.ok(xs[0] < xs[1] && xs[1] < xs[2]);
    assert.ok(ys[0] > ys[1] && ys[1] > ys[2]);
    assert.ok(xs.every(x => x >= 0 && x <= 640));
    assert.equal(env.host.querySelectorAll('.matrix-point').length, 0);
    assert.equal(env.host.querySelectorAll('[stroke-dasharray]').length, 0);
    assert.equal(env.host.querySelectorAll('.bluearf-chart-tools > button:not([hidden]) svg, .bluearf-chart-tools summary svg').length, 3);
    assert.deepEqual([...env.host.querySelectorAll('[data-export]')].map(b => b.dataset.export), ['png', 'jpeg', 'svg', 'csv']);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('numeric line preserves missing-y and missing-x breaks without joining across them', async () => {
  const env = setup();
  try {
    const points = [{ x: -4, y: 1 }, { x: -3, y: 2 }, { x: -2, y: null },
      { x: 0, y: 3 }, { x: 1, y: 4 }, { x: null, y: null }, { x: 7, y: -1 }, { x: 8, y: -2 }];
    await env.api.mount(env.host, numeric('line', points, { sample_n: 6, total_n: 6, dropped_n: 2 }));
    const path = env.host.querySelector('.numeric-line').getAttribute('d');
    assert.equal((path.match(/M/g) || []).length, 3);
    assert.equal(env.host.querySelectorAll('.numeric-point').length, 6);
    assert.match(env.host.querySelector('.bluearf-chart-footnote').textContent, /6 \/ 6 observations · 2 values excluded/);
    assert.equal(env.api.get(env.host).tableRows()[3][1], null);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('categorical normalization keeps original years, zero, negatives and explicit gaps', async () => {
  const env = setup();
  try {
    const plot = categorical();
    await env.api.mount(env.host, plot);
    const chart = env.api.get(env.host);
    assert.deepEqual(Array.from(chart.config.categories), ['2020', '2022', '2023', '2024']);
    assert.deepEqual(Array.from(chart.config.series[0].values), [1, null, -2, 0]);
    const line = [...env.host.querySelectorAll('path')].find(p => p.getAttribute('stroke') === '#163d68');
    assert.equal((line.getAttribute('d').match(/M/g) || []).length, 2);
    assert.equal(env.host.querySelectorAll('.point-0').length, 3);
    assert.match(env.host.querySelector('.bluearf-chart-footnote').textContent, /values/);
    assert.ok(env.host.querySelector('linearGradient stop[stop-opacity="0.55"]'));
    assert.equal(chart.tableRows()[2][1], null);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('stacked composition retains percentages, toggled legends and exact export metadata', async () => {
  const env = setup();
  try {
    const plot = categorical('stackedBar', { categories: ['A', 'B'], compositional: true,
      series: [{ id: 'one', name: 'Birinci', values: [10, 0] }, { id: 'two', name: 'İkinci', values: [20, 30] }] });
    plot.title = 'Tam görünür başlık';
    await env.api.mount(env.host, plot);
    const chart = env.api.get(env.host);
    assert.equal(chart.type, 'stacked');
    env.host.querySelector('[data-action="percent"]').click();
    assert.equal(chart.percent, true);
    assert.ok(env.host.textContent.includes('100%'));
    env.host.querySelector('.bluearf-chart-legend button').click();
    assert.equal(chart.hidden.has('one'), true);
    assert.match(env.host.querySelector('.bluearf-chart-footnote').textContent, /2 \/ 4 values/);
    const svg = env.api.snapshot(env.host);
    assert.equal(svg.tagName, 'svg');
    assert.ok(svg.textContent.includes('Tam görünür başlık'));
    assert.ok(svg.textContent.includes(env.host.querySelector('.bluearf-chart-footnote').textContent));
    const strike = svg.querySelector('text[text-decoration="line-through"]');
    assert.equal(strike.textContent, 'Birinci');
    assert.ok(svg.textContent.includes('İkinci'));
    assert.ok(svg.querySelector('style').textContent.includes('data:font/woff2;base64,'));
    assert.deepEqual(Array.from(chart.tableRows()[0]), ['Yıl', 'İkinci']);
    env.api.unmount(env.host);
    assert.equal(env.observed(), 0);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('donut palette maps to category slices and hidden slice is removed from CSV rows', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, categorical('donut', { categories: ['A', 'B'], compositional: true,
      palette: ['#123456', '#abcdef'], series: [{ id: 'v', name: 'Tutar', values: [1, 3] }] }));
    assert.deepEqual([...env.host.querySelectorAll('.donut-segment')].map(p => p.getAttribute('fill')), ['#123456', '#abcdef']);
    env.host.querySelector('.bluearf-chart-legend button').click();
    assert.equal(env.host.querySelectorAll('.donut-segment').length, 1);
    assert.match(env.host.querySelector('.bluearf-chart-footnote').textContent, /1 \/ 4 values/);
    assert.equal(env.api.get(env.host).tableRows().length, 2);
    assert.equal(env.api.get(env.host).tableRows()[1][0], 'B');
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('single-series stacked horizontal keeps its requested layout and percent control', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, categorical('stacked-horizontal', { categories: ['A', 'B'], compositional: true,
      series: [{ id: 'v', name: 'Tutar', values: [1, 3] }] }));
    const chart = env.api.get(env.host);
    assert.equal(chart.type, 'stacked-horizontal');
    assert.equal(env.host.dataset.chartType, 'stacked-horizontal');
    const percent = env.host.querySelector('[data-action="percent"]');
    assert.equal(percent.hidden, false);
    percent.click();
    assert.equal(chart.percent, true);
    assert.equal(chart.type, 'stacked-horizontal');
    assert.ok(env.host.textContent.includes('100%'));
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('histogram counts and coefficient intervals render their actual quantities', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, numeric('hist', [{ x0: -4, x1: -2, count: 2 }, { x0: -2, x1: 0, count: 0 }, { x0: 0, x1: 2, count: 5 }], { sample_n: 7, total_n: 7 }));
    const bars = [...env.host.querySelectorAll('.hist-bin')];
    assert.equal(Number(bars[1].getAttribute('height')), 0);
    assert.ok(Number(bars[2].getAttribute('height')) > Number(bars[0].getAttribute('height')));
    await env.api.mount(env.host, numeric('coefficients', [{ term: 'x', estimate: -2, ci_low: -3, ci_high: -1 }, { term: 'z', estimate: 0, ci_low: -1, ci_high: 1 }]));
    assert.equal(env.host.querySelectorAll('.coefficient-interval').length, 2);
    const interval = env.host.querySelector('.coefficient-interval');
    const point = env.host.querySelector('.coefficient-point');
    assert.ok(Number(interval.getAttribute('x1')) < Number(point.getAttribute('cx')));
    assert.ok(Number(point.getAttribute('cx')) < Number(interval.getAttribute('x2')));
    assert.match(env.host.querySelector('.bluearf-chart-footnote').textContent, /coefficients/);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('unmount invalidates asynchronous mount, newer mount wins and sweep disconnects observers', async () => {
  const env = setup();
  try {
    const plot = numeric('scatter', [{ x: 1, y: 2 }]);
    const pending = env.api.mount(env.host, plot);
    env.api.unmount(env.host);
    assert.equal(await pending, null);
    assert.equal(env.host.children.length, 0);
    const older = env.api.mount(env.host, plot);
    const newer = env.api.mount(env.host, { ...plot, title: 'Yeni' });
    assert.equal(await older, null);
    await newer;
    assert.equal(env.api.get(env.host).config.title, 'Yeni');
    assert.equal(env.observed(), 1);
    env.host.remove();
    env.api.sweep();
    assert.equal(env.observed(), 0);
    assert.equal(env.api.get(env.host), undefined);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('expansion never moves the React-owned host and unmount cleans the dialog', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, categorical());
    const parent = env.host.parentElement;
    env.host.querySelector('[data-action="expand"]').click();
    assert.equal(env.host.parentElement, parent);
    assert.ok(env.window.document.querySelector('dialog[open]'));
    assert.equal(env.observed(), 2);
    env.api.unmount(env.host);
    assert.equal(env.window.document.querySelector('dialog'), null);
    assert.equal(env.observed(), 0);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('malformed composition and extreme numeric ranges reject without leaked observers', async () => {
  const env = setup();
  try {
    await assert.rejects(env.api.mount(env.host, categorical('stackedBar', { compositional: true })), /nonnegative/);
    await assert.rejects(env.api.mount(env.host, numeric('scatter', [{ x: -1e308, y: 1 }, { x: 1e308, y: 2 }])), /rescale the data/);
    await assert.rejects(env.api.mount(env.host, numeric('scatter', [{ x: 0, y: 1 }, { x: 1.6e308, y: 2 }])), /rescale the data/);
    await assert.rejects(env.api.mount(env.host, categorical('donut', { categories: ['A', 'B'], compositional: true, series: [{ id: 'v', values: [1e308, 1e308] }] })), /totals exceed/);
    assert.equal(env.observed(), 0);
    assert.equal(env.api.get(env.host), undefined);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('tiny nonzero coordinates stay distinct on axes and in tooltips', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, numeric('scatter', [{ x: 1e-6, y: 1e-12 }, { x: 2e-6, y: 2e-12 }]));
    const chart = env.api.get(env.host);
    assert.notEqual(chart.format(1e-6, true), chart.format(2e-6, true));
    assert.notEqual(chart.format(1e-12), '0');
    const ticks = [...env.host.querySelectorAll('.bluearf-chart-x-axis .tick text')].map(t => t.textContent);
    assert.ok(ticks.length > 1);
    assert.equal(new Set(ticks).size, ticks.length);
    assert.ok(ticks.some(t => t.includes('0.00000')));
    env.host.querySelector('.numeric-point').dispatchEvent(new env.window.Event('focus'));
    assert.ok(env.host.querySelector('.bluearf-chart-tooltip').textContent.includes(chart.format(1e-12)));
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('adaptive axis precision distinguishes small and narrowly shifted numeric ranges', async () => {
  const env = setup();
  try {
    for (const [lo, hi] of [[.01, .02], [1.01, 1.02], [1000001, 1000002]]) {
      await env.api.mount(env.host, numeric('scatter', [{ x: lo, y: lo }, { x: hi, y: hi }]));
      for (const selector of ['.bluearf-chart-x-axis', '.bluearf-chart-y-axis']) {
        const ticks = [...env.host.querySelectorAll(`${selector} .tick text`)].map(t => t.textContent);
        assert.ok(ticks.length > 1);
        assert.equal(new Set(ticks).size, ticks.length, `${selector}: ${ticks.join(', ')}`);
      }
    }
    await env.api.mount(env.host, categorical('bar', { categories: ['A', 'B'], series: [{ id: 'v', values: [.01, .02] }] }));
    const ticks = [...env.host.querySelectorAll('.bluearf-chart-value-axis .tick text')].map(t => t.textContent);
    assert.ok(ticks.length > 1);
    assert.equal(new Set(ticks).size, ticks.length);
    assert.ok(ticks.some(t => t.includes('0.01')));
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('SVG and CSV download paths use the selected data and visible metadata', async () => {
  const env = setup();
  try {
    const blobs: Blob[] = [], downloads: string[] = [];
    env.window.URL.createObjectURL = (blob: Blob) => { blobs.push(blob); return `blob:test-${blobs.length}`; };
    env.window.URL.revokeObjectURL = () => {};
    env.window.HTMLAnchorElement.prototype.click = function () { downloads.push(this.download); };
    const textOf = (blob: Blob) => new Promise<string>((resolve, reject) => {
      const reader = new env.window.FileReader();
      reader.onload = () => resolve(String(reader.result)); reader.onerror = reject; reader.readAsText(blob);
    });
    await env.api.mount(env.host, categorical('bar', { categories: ['A', 'B'], series: [
      { id: 'first', name: 'Birinci', values: [1, 2] }, { id: 'second', name: 'İkinci', values: [3, 4] },
    ] }));
    env.host.querySelector('.bluearf-chart-legend button').click();
    const chart = env.api.get(env.host);
    await chart.export('svg'); await chart.export('csv');
    assert.deepEqual(downloads, ['Dönemler.svg', 'Dönemler.csv']);
    const svg = await textOf(blobs[0]), csv = await textOf(blobs[1]);
    assert.ok(svg.includes(env.host.querySelector('.bluearf-chart-footnote').textContent));
    assert.ok(svg.includes('Birinci'));
    assert.ok(csv.includes('İkinci'));
    assert.ok(!csv.includes('Birinci'));
    assert.ok(csv.includes('"A";3'));
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('titles and series names remain text, and CSV guards formula strings', async () => {
  const env = setup();
  try {
    const plot = categorical('bar', { series: [{ id: 'a', name: '<img src=x onerror=alert(1)>', values: [0, 1, 2, 3] }] });
    plot.title = '<script>alert(1)</script>';
    await env.api.mount(env.host, plot);
    assert.equal(env.host.querySelectorAll('script,img').length, 0);
    const xml = new env.window.XMLSerializer().serializeToString(env.api.snapshot(env.host));
    assert.ok(xml.includes('&lt;script&gt;'));
    assert.equal(env.api.csvCell('=1+2'), '"\'=1+2"');
    assert.equal(env.api.csvCell(-3), '-3');
    assert.equal(env.api.csvCell(null), '""');
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('numeric publication options apply physical labels, exact viewports and styling without changing stored observations', async () => {
  const env = setup();
  try {
    const data = [{ x: -1, y: 0 }, { x: 2, y: 0 }, { x: 20, y: 2 }];
    const options = { width: 800, height: 400, color: '#abc', opacity: .35, point_size: 7,
      grid: false, x_label: '<Physical x>', y_label: 'Outcome', xlim: [0, 10], ylim: [-10, 10] };
    await env.api.mount(env.host, numeric('scatter', data, { config: { options } }));
    const chart = env.api.get(env.host), svg = env.host.querySelector('svg[role="img"]');
    assert.equal(svg.getAttribute('viewBox'), '0 0 800 400');
    assert.equal(env.host.querySelector('.bluearf-chart-x-label').textContent, '<Physical x>');
    assert.equal(env.host.querySelector('.bluearf-chart-y-label').textContent, 'Outcome');
    const points = [...env.host.querySelectorAll('.numeric-point')];
    assert.equal(points.length, 3);
    assert.equal(points[0].getAttribute('fill'), '#aabbcc');
    assert.equal(points[0].getAttribute('r'), '7');
    assert.equal(points[0].getAttribute('fill-opacity'), '0.35');
    assert.equal(Number(points[1].getAttribute('cx')), 141.8);
    assert.equal(Number(points[1].getAttribute('cy')), 159);
    assert.ok(Number(points[0].getAttribute('cx')) < 0);
    assert.ok(Number(points[2].getAttribute('cx')) > 709);
    assert.ok(points[2].parentElement.hasAttribute('clip-path'));
    assert.deepEqual(Array.from(chart.tableRows()[3]), [20, 2]);
    assert.equal(env.host.querySelectorAll('.tick line[y2]:not([y2="0"]), .tick line[x2]:not([x2="0"])').length, 0);
    assert.match(env.host.querySelector('.bluearf-chart-footnote').textContent, /3 \/ 3 observations.*Axis view limited/);
    const snapshot = env.api.snapshot(env.host);
    assert.equal(snapshot.getAttribute('width'), '800');
    assert.equal(snapshot.querySelector('clipPath rect').getAttribute('width'), '709');
    assert.equal(snapshot.querySelector('.numeric-point').getAttribute('r'), '7');
    assert.ok(snapshot.textContent.includes('<Physical x>'));
    assert.equal(JSON.stringify(data), JSON.stringify([{ x: -1, y: 0 }, { x: 2, y: 0 }, { x: 20, y: 2 }]));
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('logarithmic numeric axes retain distances, null gaps and tiny constant positive values', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, numeric('line', [{ x: 1, y: 1 }, { x: 10, y: 10 }, { x: null, y: null }, { x: 100, y: 100 }],
      { sample_n: 3, total_n: 3, config: { options: { x_scale: 'log', y_scale: 'log', xlim: [1, 100], ylim: [1, 100], palette: ['#123456'], point_size: 6, line_width: 4 } } }));
    const points = [...env.host.querySelectorAll('.numeric-point')];
    const xs = points.map(point => Number(point.getAttribute('cx')));
    assert.ok(Math.abs((xs[1] - xs[0]) - (xs[2] - xs[1])) < 1e-9);
    assert.equal((env.host.querySelector('.numeric-line').getAttribute('d').match(/M/g) || []).length, 2);
    assert.equal(env.host.querySelector('.numeric-line').getAttribute('stroke'), '#123456');
    assert.equal(env.host.querySelector('.numeric-line').getAttribute('stroke-width'), '4');
    assert.equal(points[0].getAttribute('r'), '6');
    await env.api.mount(env.host, numeric('scatter', [{ x: 1e-12, y: 1e-20 }], { config: { options: { x_scale: 'log', y_scale: 'log' } } }));
    assert.ok(Number.isFinite(Number(env.host.querySelector('.numeric-point').getAttribute('cx'))));
    assert.ok(Number.isFinite(Number(env.host.querySelector('.numeric-point').getAttribute('cy'))));
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('categorical axes default to physical orientation and explicit labels survive compatible type switching', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, categorical('horizontal'));
    assert.equal(env.host.querySelector('.bluearf-chart-x-label').textContent, 'Değer');
    assert.equal(env.host.querySelector('.bluearf-chart-y-label').textContent, 'Yıl');
    assert.equal(env.api.get(env.host).tableRows()[0][0], 'Yıl');
    await env.api.mount(env.host, categorical('bar', { options: { x_label: 'Physical horizontal', y_label: 'Physical vertical' } }));
    const select = env.host.querySelector('select'); select.value = 'horizontal';
    select.dispatchEvent(new env.window.Event('change', { bubbles: true }));
    assert.equal(env.host.dataset.chartType, 'horizontal');
    assert.equal(env.host.querySelector('.bluearf-chart-x-label').textContent, 'Physical horizontal');
    assert.equal(env.host.querySelector('.bluearf-chart-y-label').textContent, 'Physical vertical');
    assert.equal(env.api.get(env.host).tableRows()[0][0], 'Yıl');
    await env.api.mount(env.host, categorical('bar', { options: { ylim: [-5, 10], y_format: 'integer', grid: false } }));
    const choices = [...env.host.querySelectorAll('select option')].map(option => option.value);
    assert.ok(choices.includes('bar'));
    assert.ok(!choices.includes('horizontal'));
    assert.ok(!choices.includes('donut'));
    assert.equal(env.host.querySelector('.bluearf-chart-grid'), null);
    assert.match(env.host.querySelector('.bluearf-chart-footnote').textContent, /Axis view limited/);
    assert.equal(env.api.snapshot(env.host).querySelector('clipPath rect').getAttribute('width'), '547');
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('common palette and uniform color overrides, opacity and top or hidden legends reach SVG export', async () => {
  const env = setup();
  try {
    const config = { categories: ['A', 'B'], series: [{ id: 'a', name: 'Alpha', color: '#ffffff', values: [1, 2] }, { id: 'b', name: 'Beta', values: [3, 4] }],
      options: { palette: ['#123456', '#abcdef'], opacity: .4, legend_position: 'top' } };
    await env.api.mount(env.host, categorical('bar', config));
    assert.equal(env.host.querySelector('.bar-0').getAttribute('fill'), '#123456');
    assert.equal(env.host.querySelector('.bar-1').getAttribute('fill'), '#abcdef');
    assert.equal(env.host.querySelector('.bar-1').getAttribute('fill-opacity'), '0.4');
    assert.ok(env.host.querySelector('.bluearf-chart-legend').compareDocumentPosition(env.host.querySelector('.bluearf-chart-stage')) & env.window.Node.DOCUMENT_POSITION_FOLLOWING);
    const snapshot = env.api.snapshot(env.host);
    assert.equal(snapshot.querySelector('svg').getAttribute('y'), '98');
    assert.equal([...snapshot.querySelectorAll('text')].find(text => text.textContent === 'Alpha').getAttribute('y'), '70');
    await env.api.mount(env.host, categorical('bar', { ...config, options: { ...config.options, color: '#333', legend: false } }));
    assert.equal(env.host.querySelector('.bar-0').getAttribute('fill'), '#333333');
    assert.equal(env.host.querySelector('.bar-1').getAttribute('fill'), '#333333');
    assert.equal(env.host.querySelector('.bluearf-chart-legend').hidden, true);
    assert.equal(env.api.snapshot(env.host).querySelector('svg').getAttribute('y'), '54');
    assert.ok(!env.api.snapshot(env.host).textContent.includes('Alpha'));
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('percent tick formatting uses percentage-view units and declared ranges suppress percent conversion', async () => {
  const env = setup();
  try {
    const config = { categories: ['A', 'B'], compositional: true,
      series: [{ id: 'a', values: [.1, .2] }, { id: 'b', values: [.2, .3] }], options: { y_format: 'percent' } };
    await env.api.mount(env.host, categorical('stackedBar', config));
    env.host.querySelector('[data-action="percent"]').click();
    const ticks = [...env.host.querySelectorAll('.bluearf-chart-value-axis .tick text')].map(text => text.textContent);
    assert.ok(ticks.some(text => /100/.test(text)));
    assert.ok(!ticks.some(text => /10[.,]?000/.test(text)));
    assert.ok(ticks.every(text => !/%%|%.*%/.test(text)));
    await env.api.mount(env.host, categorical('stackedBar', { ...config, options: { ylim: [0, 1] }, view: { percent: true } }));
    assert.equal(env.api.get(env.host).percent, false);
    assert.equal(env.host.querySelector('[data-action="percent"]').hidden, true);
    env.host.querySelector('[data-action="percent"]').click();
    assert.equal(env.api.get(env.host).percent, false);
    assert.ok(![...env.host.querySelectorAll('select option')].some(option => option.value === 'stacked-horizontal'));
    assert.equal(env.api.get(env.host).tableRows()[1][1], .1);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('histogram and coefficient styling retain complete bins and confidence intervals under viewport clipping', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, numeric('hist', [{ x0: 0, x1: 1, count: 2 }, { x0: 1, x1: 2, count: 9 }],
      { sample_n: 11, total_n: 11, config: { options: { color: '#123456', opacity: .5, xlim: [0, 1.5], ylim: [0, 5], y_format: 'integer' } } }));
    assert.equal(env.host.querySelectorAll('.hist-bin').length, 2);
    assert.equal(env.host.querySelector('.hist-bin').getAttribute('fill'), '#123456');
    assert.equal(env.api.get(env.host).tableRows()[2][2], 9);
    await env.api.mount(env.host, numeric('coefficients', [{ term: 'x', estimate: 2, ci_low: -1, ci_high: 4 }],
      { config: { options: { palette: ['#123456', '#abcdef'], xlim: [-2, 3], point_size: 8, line_width: 3, x_label: 'Coefficient', y_label: 'Regressor' } } }));
    assert.equal(env.host.querySelector('.coefficient-interval').getAttribute('stroke'), '#abcdef');
    assert.equal(env.host.querySelector('.coefficient-interval').getAttribute('stroke-width'), '3');
    assert.equal(env.host.querySelector('.coefficient-point').getAttribute('fill'), '#123456');
    assert.equal(env.host.querySelector('.coefficient-point').getAttribute('r'), '8');
    assert.equal(env.host.querySelector('.bluearf-chart-y-label').textContent, 'Regressor');
    assert.equal(env.api.get(env.host).tableRows()[1][3], 4);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('safe numeric format enums change only tick text, not coordinates or stored data', async () => {
  const env = setup();
  try {
    for (const mode of ['auto', 'number', 'integer', 'percent', 'scientific']) {
      await env.api.mount(env.host, numeric('scatter', [{ x: .1, y: .2 }, { x: .3, y: .4 }],
        { config: { options: { xlim: [0, 1], ylim: [0, 1], x_format: mode, y_format: mode } } }));
      const chart = env.api.get(env.host), ticks = [...env.host.querySelectorAll('.bluearf-chart-x-axis .tick text')].map(text => text.textContent);
      if (mode === 'percent') assert.ok(ticks.some(text => text.includes('%')));
      if (mode === 'scientific') assert.ok(ticks.some(text => /E/i.test(text)));
      if (mode === 'integer') assert.ok(ticks.every(text => !text.includes(',')));
      assert.equal(chart.tableRows()[1][0], .1);
      assert.ok(Math.abs(Number(env.host.querySelector('.numeric-point').getAttribute('cx')) - 54.9) < 1e-9);
    }
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('invalid styles, log domains and geometry options fail before creating observers or unsafe DOM', async () => {
  const env = setup();
  try {
    const invalid = [{ width: 319 }, { height: 1601 }, { color: 'url(javascript:1)' }, { palette: [] }, { opacity: 2 },
      { point_size: 0 }, { line_width: 1 }, { grid: 'false' }, { legend: true }, { legend_position: 'left' },
      { xlim: [1, 1] }, { x_scale: 'symlog' }, { y_format: '<script>' }, { unknown: 1 }];
    for (const options of invalid) await assert.rejects(env.api.mount(env.host, numeric('scatter', [{ x: 1, y: 2 }], { config: { options } })));
    await assert.rejects(env.api.mount(env.host, numeric('line', [{ x: 0, y: 2 }], { config: { options: { x_scale: 'log' } } })), /positive/);
    await assert.rejects(env.api.mount(env.host, numeric('hist', [{ x0: 1, x1: 2, count: 1 }], { config: { options: { ylim: [1, 2] } } })), /baseline/);
    await assert.rejects(env.api.mount(env.host, numeric('coefficients', [{ term: 'a', estimate: 2, ci_low: 1, ci_high: 3 }], { config: { options: { xlim: [1, 4] } } })), /baseline/);
    await assert.rejects(env.api.mount(env.host, categorical('donut', { compositional: true, series: [{ id: 'v', values: [1, 2, 3, 4] }], options: { grid: true } })), /Cartesian/);
    assert.equal(env.observed(), 0);
    assert.equal(env.api.get(env.host), undefined);
    assert.equal(env.host.querySelectorAll('svg,img,script').length, 0);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('area styling keeps explicit gaps and constrains conversions that cannot retain markers or physical axes', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, categorical('area', { options: { color: '#123456', point_size: 9, line_width: 5, opacity: .6, ylim: [-3, 3] } }));
    const path = [...env.host.querySelectorAll('path')].find(path => path.getAttribute('stroke') === '#123456');
    assert.equal(path.getAttribute('stroke-width'), '5');
    assert.equal(path.getAttribute('stroke-opacity'), '0.6');
    assert.equal((path.getAttribute('d').match(/M/g) || []).length, 2);
    assert.equal(env.host.querySelector('.point-0').getAttribute('r'), '9');
    assert.deepEqual([...env.host.querySelectorAll('select option')].map(option => option.value), ['area', 'line']);
    assert.equal(env.api.get(env.host).tableRows()[2][1], null);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('expanded charts retain configured dimensions, styles and viewports while dialog teardown preserves the source host', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, numeric('scatter', [{ x: 1, y: 2 }, { x: 100, y: 3 }],
      { config: { options: { width: 1200, height: 600, color: '#123456', point_size: 10, xlim: [0, 10], ylim: [0, 5] } } }));
    const parent = env.host.parentElement;
    env.host.querySelector('[data-action="expand"]').click();
    const expanded = env.window.document.querySelector('dialog[open] .bluearf-chart');
    assert.equal(expanded.querySelector('svg[role="img"]').getAttribute('viewBox'), '0 0 1200 600');
    assert.equal(expanded.querySelector('.numeric-point').getAttribute('fill'), '#123456');
    assert.equal(expanded.querySelector('.numeric-point').getAttribute('r'), '10');
    assert.equal(expanded.querySelectorAll('.numeric-point').length, 2);
    assert.equal(env.host.parentElement, parent);
    assert.equal(env.observed(), 2);
    env.window.document.querySelector('[aria-label="Close chart"]').click();
    assert.equal(env.window.document.querySelector('dialog'), null);
    assert.equal(env.observed(), 1);
    assert.equal(env.host.querySelector('.numeric-point').getAttribute('r'), '10');
    assert.equal(env.api.get(env.host).tableRows()[2][0], 100);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('SVG and CSV downloads preserve publication options and unclipped source data', async () => {
  const env = setup();
  try {
    const blobs: Blob[] = [], names: string[] = [];
    env.window.URL.createObjectURL = (blob: Blob) => { blobs.push(blob); return `blob:options-${blobs.length}`; };
    env.window.URL.revokeObjectURL = () => {};
    env.window.HTMLAnchorElement.prototype.click = function () { names.push(this.download); };
    const textOf = (blob: Blob) => new Promise<string>((resolve, reject) => {
      const reader = new env.window.FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = reject; reader.readAsText(blob);
    });
    await env.api.mount(env.host, numeric('scatter', [{ x: 1, y: 2 }, { x: 100, y: 20 }],
      { config: { options: { width: 800, height: 400, color: '#123456', point_size: 8, x_label: 'Custom x', y_label: 'Custom y', xlim: [0, 5], ylim: [0, 5] } } }));
    const chart = env.api.get(env.host); await chart.export('svg'); await chart.export('csv');
    assert.deepEqual(names, ['Örnek-grafik.svg', 'Örnek-grafik.csv']);
    const svg = await textOf(blobs[0]), csv = await textOf(blobs[1]);
    assert.ok(svg.includes('width="800"'));
    assert.ok(svg.includes('fill="#123456"'));
    assert.ok(svg.includes('r="8"'));
    assert.ok(svg.includes('clip-path="url(#'));
    assert.ok(svg.includes('Custom x'));
    assert.ok(svg.includes('Axis view limited'));
    assert.ok(csv.includes('"Custom x";"Custom y"'));
    assert.ok(csv.includes('100;20'));
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('all six annotations use physical coordinates, styles and clipped layers without changing observations', async () => {
  const env = setup();
  try {
    const annotations = [
      { type: 'text', text: '<script>not markup</script> 😀', x: 5, y: 5, color: '#123', font_size: 20, align: 'center', dx: 2, dy: -3, opacity: .6 },
      { type: 'arrow', text: 'Peak', x: 5, y: 5, dx: 20, dy: -30, color: '#abcdef', line_width: 3, dash: 'dotted' },
      { type: 'vline', x: 5, text: 'Cutoff', font_size: 14 },
      { type: 'hline', y: 5, text: 'Mean', dash: 'solid', opacity: .7 },
      { type: 'vspan', x0: 2, x1: 4, text: 'Window', align: 'center' },
      { type: 'hspan', y0: 2, y1: 4, text: 'Band', color: '#aabbcc' },
    ];
    const plot = numeric('scatter', [{ x: 5, y: 5 }, { x: 100, y: 100 }], { config: { options: { xlim: [0, 10], ylim: [0, 10], annotations } } });
    const original = JSON.stringify(plot);
    await env.api.mount(env.host, plot);
    const point = env.host.querySelector('.numeric-point'), px = Number(point.getAttribute('cx')), py = Number(point.getAttribute('cy'));
    const text = env.host.querySelector('.bluearf-chart-annotation--text text');
    assert.equal(text.textContent, '<script>not markup</script> 😀');
    assert.equal(text.getAttribute('x'), String(px + 2)); assert.equal(text.getAttribute('y'), String(py - 3));
    assert.equal(text.getAttribute('fill'), '#112233'); assert.equal(text.getAttribute('fill-opacity'), '0.6');
    assert.equal(text.getAttribute('font-size'), '20'); assert.equal(text.getAttribute('text-anchor'), 'middle');
    assert.equal(env.host.querySelectorAll('script').length, 0);
    const arrow = env.host.querySelector('.bluearf-chart-annotation--arrow line');
    assert.equal(arrow.getAttribute('x2'), String(px)); assert.equal(arrow.getAttribute('y2'), String(py));
    assert.equal(arrow.getAttribute('x1'), String(px + 20)); assert.equal(arrow.getAttribute('y1'), String(py - 30));
    assert.equal(arrow.getAttribute('stroke-width'), '3'); assert.equal(arrow.getAttribute('stroke-dasharray'), '2 3');
    assert.equal(env.host.querySelector('.bluearf-chart-annotation--vline line').getAttribute('x1'), String(px));
    assert.equal(env.host.querySelector('.bluearf-chart-annotation--vline text').getAttribute('y'), '14');
    assert.equal(env.host.querySelector('.bluearf-chart-annotation--hline text').getAttribute('x'), '12');
    const bands = env.host.querySelector('.bluearf-chart-annotations--bands');
    assert.equal(bands.querySelectorAll('rect').length, 2);
    assert.equal(bands.querySelector('rect').getAttribute('fill-opacity'), '0.12');
    assert.equal(env.host.querySelector('.bluearf-chart-annotation--vspan text').getAttribute('fill-opacity'), '1');
    assert.ok(bands.compareDocumentPosition(point) & env.window.Node.DOCUMENT_POSITION_FOLLOWING);
    assert.ok(point.compareDocumentPosition(env.host.querySelector('.bluearf-chart-annotations--foreground')) & env.window.Node.DOCUMENT_POSITION_FOLLOWING);
    const clip = env.host.querySelector('clipPath[id$="-annotations"] rect');
    assert.equal(clip.getAttribute('width'), '549'); assert.equal(clip.getAttribute('height'), '278');
    assert.equal(env.host.querySelectorAll('.numeric-point').length, 2);
    assert.equal(env.api.get(env.host).tableRows()[2][0], 100);
    assert.equal(JSON.stringify(plot), original);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('axes annotations work across all nine helper charts and radial gauge without altering chart geometry', async () => {
  const env = setup();
  try {
    const annotations = [{ type: 'text', text: 'Relative', x: .25, y: .75, coords: 'axes' }, { type: 'arrow', text: 'Here', x: .5, y: .5, coords: 'axes' }];
    const common = { config: { options: { annotations } } };
    const composition = { categories: ['A', 'B'], compositional: true, series: [{ id: 's', values: [2, 3] }], options: { annotations } };
    const plots = [numeric('scatter', [{ x: 1, y: 2 }], common), numeric('line', [{ x: 1, y: 2 }, { x: 2, y: 3 }], common),
      numeric('hist', [{ x0: 0, x1: 1, count: 2 }], common), numeric('coefficients', [{ term: 'x', estimate: 1, ci_low: 0, ci_high: 2 }], common),
      categorical('area', { options: { annotations } }), categorical('bar', { options: { annotations } }), categorical('horizontal', { options: { annotations } }),
      categorical('stacked', composition), categorical('donut', composition), categorical('gauge', { categories: ['Total'], series: [{ id: 'g', values: [60] }], options: { annotations } })];
    for (const plot of plots) {
      await env.api.mount(env.host, plot);
      const clip = env.host.querySelector('clipPath[id$="-annotations"] rect');
      const w = Number(clip.getAttribute('width')), h = Number(clip.getAttribute('height'));
      const label = env.host.querySelector('.bluearf-chart-annotation--text text');
      assert.equal(Number(label.getAttribute('x')), .25 * w); assert.equal(Number(label.getAttribute('y')), .25 * h);
      assert.equal(env.host.querySelectorAll('.bluearf-chart-annotation--arrow marker').length, 0);
      assert.equal(env.host.querySelectorAll('marker').length, 1);
      if (['donut', 'gauge'].includes(env.api.get(env.host).type)) {
        assert.equal(w, 600); assert.equal(h, 280);
        assert.equal(label.closest('.bluearf-chart-annotations').parentElement.getAttribute('transform'), 'translate(20,20)');
      }
    }
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('category, horizontal bar and coefficient annotations bind to exact labels and keep physical axis meaning', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, categorical('area', { options: { annotations: [{ type: 'arrow', text: 'Year', x: '2023', y: -2 }] } }));
    let dot = [...env.host.querySelectorAll('.point-0')][1], arrow = env.host.querySelector('.bluearf-chart-annotation--arrow line');
    assert.equal(arrow.getAttribute('x2'), dot.getAttribute('cx')); assert.equal(arrow.getAttribute('y2'), dot.getAttribute('cy'));
    assert.deepEqual([...env.host.querySelectorAll('select option')].map(option => option.value), ['area', 'line', 'bar']);
    await env.api.mount(env.host, categorical('horizontal', { categories: ['A', 'B'], series: [{ id: 's', values: [3, 5] }],
      options: { xlim: [0, 10], annotations: [{ type: 'text', text: 'A value', x: 3, y: 'A' }, { type: 'vline', x: 3 }] } }));
    const rect = env.host.querySelector('.bar-0'), text = env.host.querySelector('.bluearf-chart-annotation--text text');
    assert.equal(Number(text.getAttribute('y')), Number(rect.getAttribute('y')) + Number(rect.getAttribute('height')) / 2);
    assert.equal(Number(text.getAttribute('x')), Number(rect.getAttribute('x')) + Number(rect.getAttribute('width')));
    assert.equal(env.host.querySelector('select'), null);
    await env.api.mount(env.host, numeric('coefficients', [{ term: 'education', estimate: 2, ci_low: 1, ci_high: 3 }, { term: 'experience', estimate: 1, ci_low: .5, ci_high: 1.5 }],
      { config: { options: { annotations: [{ type: 'text', text: 'Effect', x: 2, y: 'education' }, { type: 'vline', x: 0 }] } } }));
    dot = env.host.querySelector('.coefficient-point');
    const label = env.host.querySelector('.bluearf-chart-annotation--text text');
    assert.equal(label.getAttribute('x'), dot.getAttribute('cx')); assert.equal(label.getAttribute('y'), dot.getAttribute('cy'));
    assert.equal(env.host.querySelectorAll('.coefficient-interval').length, 2);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('log annotations follow geometric scale distances and viewport clipping hides outside anchors without expanding axes', async () => {
  const env = setup();
  try {
    const annotations = [{ type: 'text', text: 'ten', x: 10, y: 10 }, { type: 'text', text: 'outside', x: 1000, y: 10 },
      { type: 'arrow', text: 'outside arrow', x: 10, y: .01, dx: 0, dy: -500 }, { type: 'vline', x: 1000 },
      { type: 'vspan', x0: .1, x1: 10, text: 'Visible interval' }, { type: 'hspan', y0: 1000, y1: 2000 }];
    await env.api.mount(env.host, numeric('line', [{ x: 1, y: 1 }, { x: 10, y: 10 }, { x: 100, y: 100 }],
      { config: { options: { x_scale: 'log', y_scale: 'log', xlim: [1, 100], ylim: [1, 100], annotations } } }));
    const points = [...env.host.querySelectorAll('.numeric-point')], label = env.host.querySelector('.bluearf-chart-annotation--text text');
    assert.equal(label.getAttribute('x'), points[1].getAttribute('cx')); assert.equal(label.getAttribute('y'), points[1].getAttribute('cy'));
    assert.equal(env.host.querySelectorAll('.bluearf-chart-annotation--text text').length, 1);
    assert.equal(env.host.querySelectorAll('.bluearf-chart-annotation--arrow,.bluearf-chart-annotation--vline').length, 0);
    const band = env.host.querySelector('.bluearf-chart-annotation--vspan rect');
    assert.equal(band.getAttribute('x'), '0'); assert.equal(band.getAttribute('width'), points[1].getAttribute('cx'));
    assert.equal(Number(env.host.querySelector('.bluearf-chart-annotation--vspan text').getAttribute('x')), Number(band.getAttribute('width')) / 2);
    assert.equal(env.host.querySelectorAll('.bluearf-chart-annotation--hspan').length, 0);
    assert.equal(env.api.get(env.host).config.data.length, 3);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('data annotations suppress percent composition and incompatible conversions while relative annotations remain flexible', async () => {
  const env = setup();
  try {
    const composition = { categories: ['A', 'B'], compositional: true, series: [{ id: 's', values: [2, 3] }] };
    await env.api.mount(env.host, categorical('stacked', { ...composition, view: { percent: true, type: 'donut' }, options: { annotations: [{ type: 'text', text: 'Value', x: 'A', y: 2 }] } }));
    let chart = env.api.get(env.host);
    assert.equal(chart.type, 'stacked'); assert.equal(chart.percent, false);
    assert.equal(env.host.querySelector('[data-action="percent"]').hidden, true);
    env.host.querySelector('[data-action="percent"]').click(); assert.equal(chart.percent, false);
    const types = [...env.host.querySelectorAll('select option')].map(option => option.value);
    assert.ok(types.includes('stacked-area')); assert.ok(!types.includes('horizontal')); assert.ok(!types.includes('stacked-horizontal')); assert.ok(!types.includes('donut'));
    await env.api.mount(env.host, categorical('stacked', { ...composition, options: { annotations: [{ type: 'text', text: 'Caption', x: .5, y: .8, coords: 'axes' }] } }));
    chart = env.api.get(env.host); env.host.querySelector('[data-action="percent"]').click();
    assert.equal(chart.percent, true); assert.equal(env.host.querySelector('[data-action="percent"]').hidden, false);
    const select = env.host.querySelector('select'); select.value = 'donut'; select.dispatchEvent(new env.window.Event('change', { bubbles: true }));
    assert.equal(chart.type, 'donut'); assert.equal(env.host.querySelector('.bluearf-chart-annotation--text text').textContent, 'Caption');
    assert.equal(env.host.querySelector('.bluearf-chart-annotation--text text').getAttribute('x'), '300');
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('annotation positions recompute through the actual resize observer and expansion keeps isolated arrow markers', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, numeric('scatter', [{ x: 5, y: 5 }], { config: { options: { xlim: [0, 10], ylim: [0, 10], annotations: [{ type: 'arrow', text: 'Center', x: 5, y: 5 }] } } }));
    const initial = Number(env.host.querySelector('.bluearf-chart-annotation--arrow line').getAttribute('x2'));
    await env.resize(1000);
    const after = env.host.querySelector('.bluearf-chart-annotation--arrow line');
    assert.ok(Number(after.getAttribute('x2')) > initial); assert.equal(after.getAttribute('x2'), env.host.querySelector('.numeric-point').getAttribute('cx'));
    const originalMarker = env.host.querySelector('marker').id;
    env.host.querySelector('[data-action="expand"]').click();
    const expanded = env.window.document.querySelector('dialog[open] .bluearf-chart'), expandedMarker = expanded.querySelector('marker').id;
    assert.notEqual(expandedMarker, originalMarker);
    assert.equal(expanded.querySelector('.bluearf-chart-annotation--arrow line').getAttribute('marker-end'), `url(#${expandedMarker})`);
    assert.equal(env.host.querySelector('.bluearf-chart-annotation--arrow line').getAttribute('marker-end'), `url(#${originalMarker})`);
    const snapshot = env.api.snapshot(env.host), exportedMarker = snapshot.querySelector('marker').id;
    assert.notEqual(exportedMarker, originalMarker);
    assert.equal(snapshot.querySelector('.bluearf-chart-annotation--arrow line').getAttribute('marker-end'), `url(#${exportedMarker})`);
    assert.notEqual(env.api.snapshot(env.host).querySelector('marker').id, exportedMarker);
    env.window.document.querySelector('[aria-label="Close chart"]').click();
    assert.equal(env.observed(), 1);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('annotation SVG download and JSON reopen preserve safe text, markers, clipping and complete CSV data', async () => {
  const env = setup();
  try {
    const blobs: Blob[] = [];
    env.window.URL.createObjectURL = (blob: Blob) => { blobs.push(blob); return `blob:annotation-${blobs.length}`; };
    env.window.URL.revokeObjectURL = () => {}; env.window.HTMLAnchorElement.prototype.click = function () {};
    const textOf = (blob: Blob) => new Promise<string>(resolve => { const reader = new env.window.FileReader(); reader.onload = () => resolve(String(reader.result)); reader.readAsText(blob); });
    const plot = numeric('scatter', [{ x: 2, y: 3 }, { x: 100, y: 200 }], { config: { options: { xlim: [0, 5], ylim: [0, 5], annotations: [{ type: 'arrow', text: '<b>Value & estimate</b>', x: 2, y: 3, color: '#987654' }] } } });
    await env.api.mount(env.host, plot); await env.api.get(env.host).export('svg'); await env.api.get(env.host).export('csv');
    const svg = await textOf(blobs[0]), csv = await textOf(blobs[1]);
    const document = new env.window.DOMParser().parseFromString(svg, 'image/svg+xml');
    assert.equal(document.querySelector('parsererror'), null);
    assert.equal(document.querySelector('.bluearf-chart-annotation--arrow text').textContent, '<b>Value & estimate</b>');
    assert.equal(document.querySelector('b'), null);
    const id = document.querySelector('marker').id;
    assert.equal(document.querySelector('.bluearf-chart-annotation--arrow line').getAttribute('marker-end'), `url(#${id})`);
    assert.equal(document.querySelector('clipPath[id*="-annotations"] rect').getAttribute('width'), '549');
    assert.ok(csv.includes('100;200')); assert.ok(!csv.includes('estimate'));
    await env.api.mount(env.host, JSON.parse(JSON.stringify(plot)));
    assert.equal(env.host.querySelector('.bluearf-chart-annotation--arrow text').textContent, '<b>Value & estimate</b>');
    assert.equal(env.host.querySelector('.bluearf-chart-annotation--arrow line').getAttribute('stroke'), '#987654');
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('two SVG exports after resize isolate all gradients, clips and arrow references from each other and the live chart', async () => {
  const env = setup();
  try {
    await env.api.mount(env.host, categorical('area', { options: { ylim: [-3, 3], annotations: [{ type: 'arrow', text: 'Effect', x: '2023', y: -2 }] } }));
    const liveSvg = env.host.querySelector('.bluearf-chart-stage > svg'), liveIds = [...liveSvg.querySelectorAll('[id]')].map(node => node.id);
    const first = env.api.snapshot(env.host);
    await env.resize(1000);
    const liveAfterResize = liveSvg.outerHTML, second = env.api.snapshot(env.host);
    const exports = env.window.document.createElement('section'); exports.append(first, second); env.window.document.body.append(exports);
    const firstIds = new Set([...first.querySelectorAll('[id]')].map(node => node.id));
    const secondIds = new Set([...second.querySelectorAll('[id]')].map(node => node.id));
    assert.ok(firstIds.size >= 4); assert.equal(secondIds.size, firstIds.size);
    assert.ok([...firstIds].every(id => !secondIds.has(id) && !liveIds.includes(id)));
    assert.ok([...secondIds].every(id => !liveIds.includes(id)));
    assert.notEqual(first.querySelector('clipPath[id*="-annotations"] rect').getAttribute('width'), second.querySelector('clipPath[id*="-annotations"] rect').getAttribute('width'));
    for (const output of [first, second]) {
      let references = 0;
      for (const node of output.querySelectorAll('*')) {
        for (const attr of [...node.attributes]) {
          const matches = [...attr.value.matchAll(/url\(#([^)]*)\)/g)].map(match => match[1]);
          if (attr.localName === 'href' && attr.value.startsWith('#')) matches.push(attr.value.slice(1));
          for (const id of matches) {
            references++;
            const target = env.window.document.getElementById(id);
            assert.ok(target && output.contains(target), `Reference ${id} must resolve within its own export`);
          }
        }
      }
      assert.ok(references >= 4);
    }
    assert.equal(liveSvg.outerHTML, liveAfterResize);
    assert.deepEqual([...liveSvg.querySelectorAll('[id]')].map(node => node.id), liveIds);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('malformed annotation payloads and inapplicable styles reject before creating DOM or observers', async () => {
  const env = setup();
  try {
    const good = { type: 'text', text: 'text', x: 1, y: 2 };
    const invalid = [null, {}, new Array(1), Array.from({ length: 101 }, () => good), [null], [{ ...good, type: 'circle' }],
      [{ ...good, text: 'a'.repeat(501) }], [{ ...good, text: '\n' }], [{ ...good, text: '\uD800' }], [{ ...good, x: Infinity }],
      [{ ...good, y: true }], [{ ...good, opacity: null }], [{ ...good, coords: null }], [{ ...good, coords: 'figure' }],
      [{ ...good, coords: 'axes', x: 1.01 }], [{ ...good, color: 'red' }], [{ ...good, font_size: 37 }], [{ ...good, dx: 501 }],
      [{ ...good, align: 'justify' }], [{ ...good, line_width: 2 }], [{ ...good, html: true }], [{ type: 'arrow', x: 1, y: 2 }],
      [{ type: 'vline', x: 1, coords: 'axes' }], [{ type: 'vline', x: 1, dx: 2 }], [{ type: 'vline', x: 1, dash: 'dashdot' }],
      [{ type: 'vspan', x0: 1, x1: 1 }], [{ type: 'hspan', y0: 1, y1: 2, line_width: 2 }]];
    for (const annotations of invalid) await assert.rejects(env.api.mount(env.host, numeric('scatter', [{ x: 1, y: 2 }], { config: { options: { annotations } } })));
    await assert.rejects(env.api.mount(env.host, numeric('scatter', [{ x: 1, y: 2 }], { config: { options: { x_scale: 'log', annotations: [{ ...good, x: 0 }] } } })), /positive/);
    await assert.rejects(env.api.mount(env.host, categorical('bar', { options: { annotations: [{ ...good, x: 'missing' }] } })), /category/);
    await assert.rejects(env.api.mount(env.host, categorical('bar', { options: { annotations: [{ type: 'vline', x: 1 }] } })), /numeric x/);
    await assert.rejects(env.api.mount(env.host, categorical('horizontal', { options: { annotations: [{ type: 'hline', y: 1 }] } })), /numeric y/);
    await assert.rejects(env.api.mount(env.host, numeric('coefficients', [{ term: 'x', estimate: 1, ci_low: 0, ci_high: 2 }], { config: { options: { annotations: [{ ...good, y: 'not x' }] } } })), /category/);
    await assert.rejects(env.api.mount(env.host, categorical('donut', { compositional: true, series: [{ id: 's', values: [1, 2, 3, 4] }], options: { annotations: [good] } })), /axes-relative/);
    assert.equal(env.observed(), 0); assert.equal(env.api.get(env.host), undefined); assert.equal(env.host.querySelectorAll('svg').length, 0);
    await env.api.mount(env.host, numeric('scatter', [{ x: 1e20, y: 2 }], { config: { options: { annotations: [{ ...good, x: 1e20, text: '😀'.repeat(500) }] } } }));
    assert.equal([...env.host.querySelector('.bluearf-chart-annotation--text text').textContent].length, 500);
    assert.ok(env.api.get(env.host).config.options.annotations[0].x === 1e20);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});

test('absent and empty annotations retain existing chart defaults and produce no annotation layers', async () => {
  const env = setup();
  try {
    const base = numeric('scatter', [{ x: 1, y: 2 }]);
    await env.api.mount(env.host, base);
    const before = env.host.querySelector('.numeric-point').outerHTML;
    assert.deepEqual(Object.keys(env.api.get(env.host).config.options), []);
    await env.api.mount(env.host, { ...base, config: { options: { annotations: [] } } });
    assert.equal(env.host.querySelector('.numeric-point').outerHTML, before);
    assert.equal(env.host.querySelectorAll('.bluearf-chart-annotations,marker').length, 0);
  } finally { env.api.unmount(env.host); env.dom.window.close(); }
});
