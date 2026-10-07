
### Direct offline network files

`plot.to_pdf(path)`, `plot.to_svg(path)` and `plot.to_png(path)` run the saved
network chart in a disposable headless session of an installed Chrome, Chromium
or Edge. No panel, account, browser download, user profile, CDN or remote font is
needed. Saving HTML alone does not execute a browser layout. `browser_executable` selects an explicit executable. Layout, seed,
filters, annotations, labels, selected timeline frame and saved camera come from
the PlotSpec; export does not mutate it. A saved view bypasses layout; otherwise
worker completion and font readiness are required. Layout error, timeout or work
limit causes a controlled error before publication.

Common keywords: `width=960` (320–2400 CSS pixels), `height=600` (240–1600 workspace
pixels), `timeout=60` (1–300 seconds), `overwrite=False`. The output includes its
title and coverage/legend footer, so image height is larger than workspace height.
PNG alone accepts `scale=1` (0.5–2); it refuses allocations above 16 million pixels.
PDF and SVG refuse over 200,000 displayed node/edge primitives and never silently
rasterize. Files and offline source documents are limited to 128 MiB. Publication
is atomic; existing files require explicit overwrite. The bundled Barlow font is
embedded in vector output. PDF refuses missing glyphs; SVG/PNG may use the
installed browser's system glyph fallback for scripts outside Barlow coverage.
Reproducibility is within a given browser/font/platform version; PDF/SVG preserve
vector geometry while PNG captures that browser's rasterization.

### Timeline transport

HTML, saved views and local history artifacts use `timeline-pool-v1` when smaller
than the expanded representation. Identical complete node/edge records (including
attributes and positions) are interned once; ordered reference arrays preserve
parallel edges, exact typed identities and all full/shown counts. Records with
changed degree, attributes, weights or positions stay distinct. Public
`model_dump()` and browser Graph JSON remain expanded. `load_view` accepts both.

The decoder admits total reference counts before allocating expanded arrays, and
keeps the existing 60-frame, 100,000-node, 1,000,000-edge aggregate display bounds
and 128 MiB workspace/payload guards. Normalization caches one record per unique
pooled record, bounded by those same limits. Frames are synchronous and wholly
local; this design avoids external frame files, asynchronous cache races and
missing-file dependencies in offline exports. Closing or switching a panel still
aborts its original artifact fetch and terminates its layout worker.

The checked local benchmark in
`docs/evidence/market-72-timeline-transport-2026-10-07.json` measures identical
600-node, 6,000-edge, eight-frame production fixtures in fresh browsers three
times per representation. Its timing and heap observations describe that fixture
and machine, not a universal performance guarantee.
