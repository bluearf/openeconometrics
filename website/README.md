# OpenEconometrics download site

A single-page, responsive English landing page on a pure white background with the bundled Barlow font. Static HTML, CSS and JavaScript; no build step, account system or tracking. The font and brand asset are served locally.

Serve `website/` with any static host. From the repository root, preview with:

```sh
python3 -m http.server 4178 --bind 127.0.0.1 --directory website
```

The download starts disabled. On load, the site checks the unauthenticated GitHub releases API for `bluearf/openeconometrics`. It activates the Mac button only when a published release has a nonempty, uploaded `OpenEconometrics_*_aarch64.dmg` asset (or a legacy `OpenEcon_*_aarch64.dmg` asset) and an exact download URL matching this repository, release tag and asset filename. A missing release keeps the button disabled; 404, rate limits, malformed responses and network errors report unavailable. The first ten releases, including previews, are checked because OpenEconometrics is currently an alpha.

The first public alpha is 0.3.43: Apple Silicon and macOS 15+, verified against the packaged app (arm64, LSMinimumSystemVersion 15.0, ad-hoc signature with no Apple Developer team). Developer ID signing and notarization are pending and disclosed in the page and release notes. There are no download claims for Windows, Intel Mac or Linux.

Public source is a reviewed snapshot in `bluearf/openeconometrics`. Its manifest records the source commit and included file hashes; private development history, internal evidence and external research fixture bytes are excluded. Anonymous repository, release and downloaded DMG checks remain separate publication receipts.

Run `node --test website/site.test.cjs` from the repository root to verify asset selection, signing copy and failure states.

The canonical URL is https://openeconometrics.com. The apex and www hostnames serve the same source, both declaring the apex canonical URL. No redirect or DNS change is claimed. Live source, browser views, deployment, and unauthenticated GitHub access are checked separately in the MARKET-81 evidence.

`LICENSE.txt` is the OpenEconometrics Apache-2.0 license; `assets/BARLOW-OFL.txt` covers the bundled font.
