const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const script = fs.readFileSync(__dirname + '/site.js', 'utf8');
const repository = 'bluearf/openeconometrics';
const asset = { name: 'OpenEconometrics_0.3.43_aarch64.dmg', state: 'uploaded', size: 324208247,
  browser_download_url: `https://github.com/${repository}/releases/download/v0.3.43-alpha.1/OpenEconometrics_0.3.43_aarch64.dmg` };

async function run(status, releases) {
  const elements = new Map();
  const document = { querySelector(selector) {
    if (!elements.has(selector)) elements.set(selector, {
      textContent: 'Public Mac download pending.', attributes: new Set(['aria-disabled', 'tabindex']),
      removeAttribute(name) { this.attributes.delete(name); },
      classList: { add() {}, remove() {} }
    });
    return elements.get(selector);
  } };
  const fetch = async (url, options) => {
    assert.equal(url, `https://api.github.com/repos/${repository}/releases?per_page=10`);
    assert.equal(options.credentials, 'omit');
    return { status, ok: status === 200, json: async () => releases };
  };
  const context = vm.createContext({ document, fetch, URL, AbortSignal });
  vm.runInContext(script, context);
  await vm.runInContext('findDownload()', context);
  return { button: elements.get('#download-button'), status: elements.get('#release-status-text'),
    note: elements.get('#release-note') };
}

test('uploaded public alpha activates the exact matching Mac asset and discloses signing', async () => {
  const result = await run(200, [{ tag_name: 'v0.3.43-alpha.1', draft: false, assets: [asset] }]);
  assert.equal(result.button.href, asset.browser_download_url);
  assert.equal(result.button.attributes.has('aria-disabled'), false);
  assert.match(result.note.textContent, /ad-hoc signed/);
  assert.match(result.note.textContent, /notarization are pending/);
});

test('drafts, empty or unuploaded assets, and mismatched URLs remain disabled', async () => {
  for (const release of [
    { draft: true }, { assets: [{ ...asset, state: 'new' }] }, { assets: [{ ...asset, size: 0 }] },
    { assets: [{ ...asset, browser_download_url: asset.browser_download_url.replace(repository, 'other/repo') }] },
    { assets: [{ ...asset, browser_download_url: asset.browser_download_url.replace('v0.3.43-alpha.1', 'wrong-tag') }] },
    { assets: [{ ...asset, browser_download_url: asset.browser_download_url + '?redirect=elsewhere' }] }
  ]) {
    const result = await run(200, [{ tag_name: 'v0.3.43-alpha.1', draft: false, assets: [asset], ...release }]);
    assert.equal(result.button.href, undefined);
    assert.equal(result.button.attributes.has('aria-disabled'), true);
  }
});

test('404, rate limits, and malformed responses report unavailable without an active link', async () => {
  for (const [status, response] of [[404, null], [403, null], [429, null], [200, {}]]) {
    const result = await run(status, response);
    assert.equal(result.button.href, undefined);
    assert.match(result.status.textContent, /currently unavailable/);
  }
});
