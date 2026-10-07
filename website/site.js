'use strict';

const repository = 'bluearf/openeconometrics';
const downloadButton = document.querySelector('#download-button');
const statusText = document.querySelector('#release-status-text');
const releaseNote = document.querySelector('#release-note');
const otherPlatforms = document.querySelector('#other-platforms');

function showUnavailable() {
  statusText.textContent = 'The download link is currently unavailable.';
  releaseNote.textContent = 'Release information could not be retrieved. Check GitHub releases for current packages and installation notes.';
}

// Activate the download only for an uploaded asset owned by this repository.
async function findDownload() {
  try {
    const response = await fetch(`https://api.github.com/repos/${repository}/releases?per_page=10`, {
      headers: { Accept: 'application/vnd.github+json' }, signal: AbortSignal.timeout(8000), credentials: 'omit'
    });
    if (response.status === 404) {
      showUnavailable();
      return;
    }
    if (!response.ok) throw new Error('Release lookup unavailable');
    const releases = await response.json();
    if (!Array.isArray(releases)) throw new Error('Invalid release response');
    for (const release of releases) {
      if (release.draft || !Array.isArray(release.assets)) continue;
      const asset = release.assets.find(asset => {
        if (asset.state !== 'uploaded' || asset.size <= 0 || !/^(?:OpenEconometrics|OpenEcon)_[0-9][\w.-]*_aarch64\.dmg$/.test(asset.name)) return false;
        try {
          const url = new URL(asset.browser_download_url);
          const expectedPath = `/${repository}/releases/download/${encodeURIComponent(release.tag_name)}/${encodeURIComponent(asset.name)}`;
          return url.origin === 'https://github.com' && url.pathname === expectedPath && !url.username && !url.password && !url.search && !url.hash;
        } catch { return false; }
      });
      if (!asset) continue;
      downloadButton.href = asset.browser_download_url;
      downloadButton.removeAttribute('aria-disabled');
      downloadButton.removeAttribute('tabindex');
      downloadButton.classList.remove('unavailable');
      document.querySelector('#release-status').classList.add('available');
      statusText.textContent = `${release.tag_name} · Download available`;
      releaseNote.textContent = 'Mac alpha for Apple Silicon and macOS 15 or later. This package is ad-hoc signed; Apple Developer ID signing and notarization are pending. Read the release notes before installation.';
      // A matching ARM Mac asset establishes no Intel/Windows/Linux availability.
      return;
    }
    if (releases.length) {
      statusText.textContent = 'A Mac download has not been published yet.';
      releaseNote.textContent = 'Check GitHub releases for current packages and installation notes.';
    }
  } catch {
    showUnavailable();
  }
}

findDownload();
