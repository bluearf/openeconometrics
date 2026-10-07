# Chart source provenance and port boundaries

Inspected on 30 September–1 October 2026. The user asked to reuse their existing D3 charts
in a separate Python package and integrate that package with OpenEcon. The source
repositories were inspected read-only; customer data, authentication modules and
live service integrations are outside this port.

## Source identification

Two relevant local repositories exist:

- `source-checkouts/dart-d3` contains the original standalone Dart
  HTML generators and their generated chart examples. `bin/main.dart` is only a
  Hello World entry point; it is not a chart implementation.
- `source-checkouts/bluearf-html-repo` contains expanded dashboard
  generators and `dashboard/reporting/bluearf_charts.js`, `.css` and `.md`, a
  reusable, instance-scoped adaptation of those dashboard charts.

The port uses the existing shared renderer as its implementation base and
preserves the user's Dart/D3 chart design family. The relationship supported by
source inspection is:

| Layer | Observed source and relationship |
| --- | --- |
| Standalone Dart/D3 | `dart-d3/bin/stacked_bar.dart`, `vertical_bar_chart_one_by_one.dart`, `horizontal_bar_chart_one_by_one.dart`: Barlow typography, blue/navy palette, animated bars, tooltip/legend behavior and export menus |
| Dashboard family | `bluearf-html-repo/dashboard/dashboard-grafikleri/createVerticalBarChartD3.dart`, `createHorizontalStackedBarChartD3.dart`, `createVerticalAreaChartD3.dart`: expanded multi-series/localized generators with the same visual vocabulary |
| Shared renderer | `dashboard/reporting/bluearf_charts.js` explicitly attributes area gradients/reveals, bars/stacking and donut/gauge behavior to the dashboard generators and named functions in `analytics_dashboard.html`; the accompanying Markdown describes the adaptation |
| Python package | A port of the shared renderer, retaining source attribution and adding Python data validation, offline assets and econometric chart contracts; integration does not embed the authenticated dashboard |

The shared renderer's dashboard derivation is documented in its source header.
Dart/D3 and dashboard files were inspected for matching interfaces and visual
patterns. This inspection does not claim a verified Git commit ancestry or that
the shared renderer is a byte-for-byte copy of the standalone Dart files.

The Bluearf repository's root `AGENTS.md` was read. Its deployment and Nova
backend rules do not require changes for this read-only chart extraction. No
`AGENTS.md` was found in the standalone `dart-d3` repository.

## Original contracts and behavior

The standalone Dart interfaces are HTML generators, not independently installable
JavaScript or Python chart packages:

- `plotCreateStackedBarChart(values, year, language, showRightButton, seriesNames,
  seriesColors, axisLabels, unitLabels)` expects a series-by-month value matrix
  and generates the selected year's 12 monthly categories.
- `vertical_bar_chart_one_by_one(...)` accepts localized category arrays,
  one value/color per category, axis/unit labels and height. The old generator
  substitutes zero when a value array is shorter than its category array.
- `horizontal_bar_chart_one_by_one(...)` accepts localized category names,
  values, colors, labels and height. Its original scale starts at zero and its
  resize handler clears legend selection.

The chosen shared JavaScript renderer exposes `BluearfCharts.mount(host, config)`
as an asynchronous factory, plus `get`, `sweep`, `normalize`, `allowedTypes`,
`rows`, `csvCell`, `COLORS` and a dependency loader. Returned chart instances
support `destroy()` and `view()`; DOM IDs, listeners, observers and chart state
are scoped to the instance.

Its data contract contains:

- `type`, `title`, `unit`, `locale`, `height`, `legend` and optional compact mode;
- `categories` and `series`, with each series carrying `id`, `name`, `color` and
  a value array aligned to categories;
- `compositional` for valid complete/nonnegative component data;
- `points` with `x`, `y`, optional `id`, `label` and color for the original matrix;
- saved `view` state for chart type, hidden series/slices and percentage mode.

Original chart types are area, line, vertical bar, horizontal bar, stacked
vertical/horizontal/area, donut, gauge and a materiality-style scatter matrix.
Stacking and percentages are allowed only for explicitly compositional,
complete, nonnegative data. Missing observations stay null; signed values are
not silently turned into zero. Ordered year categories are automatically
expanded to include intervening missing years in the original renderer.

The original scatter matrix has a 0-to-5 default domain and threshold guides at
3. It is not already a general econometric scatter renderer. Numeric-x line,
general scatter, histogram and coefficient/interval handling in the Python port
must be identified as extensions, not pre-existing dashboard capabilities.
Missingness, observation counts, numeric spacing and interval limits must come
from the supplied data; decorative dashboard semantics must not change them.

## Offline and integration requirements

The legacy Dart HTML loads D3 `@7`, html2canvas 1.4.1, jsPDF 2.5.1, SheetJS 0.18.5
and Barlow from external CDNs/Google Fonts. It uses global DOM IDs, global state
and page-wide resize listeners. Directly inserting several such pages into one
workbench would cause collisions and would not be reliably offline.

The shared renderer already removes dashboard authentication and business-data
queries. Inspection found no Firebase calls, HTTP data fetches, local/session
storage integration, `eval` or dynamic-function construction in that module.
Its remaining integration assumptions need explicit handling in the port:

1. Its lazy loader has a hard-coded `/dashboard/reporting/assets/` prefix. A
   Python package must provide its own local/inline asset resolution rather than
   retain that route or silently fall back to a CDN.
2. Toolbar icons originally use an optional global `lucide`. Without it,
   icon-only controls lose their visible symbols. The standalone package needs
   bundled/inline icons or readable text controls.
3. Barlow is named in CSS, but font files are not embedded by the shared module.
   Local font packaging must retain its OFL notice. Some export heading text
   deliberately uses Arial; a standalone SVG cannot assume the workbench's
   installed fonts or CSS are available.
4. `ResizeObserver`, `AbortController`, SVG, canvas, Blob URLs and modal dialogs
   are browser facilities. Python creates chart descriptions/HTML; it does not
   turn browser raster/PDF export into a headless server renderer automatically.
5. Standalone HTML must encode untrusted labels safely, including `</script>`
   sequences. The source renderer's DOM escaping and CSV formula-text protection
   must survive adaptation. No customer/authentication code or data is needed.
6. SVG exports contain title, units and visible-series legend. CSV/XLSX exports
   use the selected series. Missing data and selection semantics should remain
   consistent across the displayed chart, data table and downloads.

The shared implementation does not need html2canvas: it serializes SVG, then
uses browser canvas for PNG/JPEG and raster-backed PDF. Its image/PDF export is
not a claim of editable vector PDF output.

## Current package, distinct from the original catalogue

`packages/openecon-charts` is an independently installable Python distribution
with no required Python dependencies. `openecon.plot` re-exports this package;
the model framework is not a dependency of the chart library. Mapping, record
and DataFrame-style inputs use the same validated chart descriptions. Model
coefficient charts accept records through a small attribute/mapping protocol.

The current public helpers are `bar`, `barh`, `area`, `stacked_bar` (including
horizontal layout), `donut`, `scatter`, numeric-x `line`, `hist` and
`coefficients`. The first group adapts the existing visual components. The
numeric-x line, general scatter, histogram and coefficient/interval charts are
OpenEcon extensions. The old materiality matrix, gauge and complete dashboard
catalogue are not promised as Python helpers.

The adapted JavaScript namespace is `OpenEconCharts`. The port removes the
dashboard asset loader and optional Lucide dependency, supplies inline control
icons, and bundles D3 7.9.0 and the Barlow regular WOFF2 font. Saved HTML embeds
these assets and blocks external network requests. Notebook display uses an
isolated iframe. SVG export embeds the font and includes chart title, unit,
legend selection and count metadata.

Current downloads are **SVG, PNG, JPEG and CSV**. PDF and XLSX are intentionally
outside this port; their inspected source dependencies are not shipped.
Browser raster export still uses canvas and Blob URLs; it is not a Python
server-side image export API. Notebook download support also depends on the
surrounding notebook application's policy.

The port preserves categorical input order and does not insert intervening
years or substitute zero for missing values. Numeric lines retain irregular x
distances and explicit gaps. Scatter sampling is declared in count metadata;
numeric lines are not downsampled. Signed bar/area values remain signed, while
compositions require complete nonnegative inputs. Categorical counts refer to
value cells across the supplied series, not independent observations. These
are deliberate statistical semantics rather than inherited dashboard defaults.

## Licenses and attribution

No repository-root LICENSE/COPYING file was found in either inspected source
repository, and the custom reusable renderer has no standalone license grant in
its header. The user explicitly requested this port into the Apache-2.0
OpenEcon project. That authorization is the basis for porting the user-provided
custom code; this document does not invent a pre-existing third-party license
for it. Keep its source attribution and modification history.

Third-party assets have their own accompanying notices and are not relicensed
by OpenEcon's Apache-2.0 license:

| Asset inspected | Version | Accompanying license |
| --- | --- | --- |
| D3 bundled JavaScript | 7.9.0 | ISC; `dashboard/reporting/assets/d3.LICENSE` |
| jsPDF | 2.5.1 | MIT; `dashboard/reporting/assets/jspdf.LICENSE` |
| SheetJS Community Edition | 0.18.5 | Apache-2.0; `dashboard/reporting/assets/xlsx.LICENSE` |
| Barlow regular font | Source local TTF | SIL Open Font License 1.1; `assets/fonts/barlow/OFL.txt` |

The **shipped** Barlow asset is the unchanged regular WOFF2 file from
`source-checkouts/bluearf-website/assets/fonts/barlow/Barlow-Regular.woff2`,
with its adjacent `OFL.txt`. It is packaged as `assets/Barlow.woff2` and
`assets/BARLOW-OFL.txt`. Its source and packaged SHA-256 values match. The
Bluearf HTML repository's TTF above was inspected as an available source asset
but is not shipped. D3's ISC notice is packaged as `assets/D3-LICENSE.txt`.

Other Barlow weights exist in the source directories. Copy only assets
actually shipped by the package, with their notices. Existing Bluearf marks,
logos, provider icons, standards images and dashboard data are not necessary
chart dependencies and are outside this extraction.

## Inspected source SHA-256

These identify source inputs, not the modified Python package outputs. Paths in
the table are relative to the named source repository.

| Source | SHA-256 |
| --- | --- |
| `dart-d3/bin/stacked_bar.dart` | `e4969d023d7699fadfa5c61c1d7edad6debfe301c540d231c90bfffb0d272270` |
| `dart-d3/bin/vertical_bar_chart_one_by_one.dart` | `4b5f72a783e9efd8a7df8987fd38e18f21245a9eb014d376bb6de4bc567dc716` |
| `dart-d3/bin/horizontal_bar_chart_one_by_one.dart` | `97cba3a6a546cc4b58ccc14f5bf1cc817e32e4df62c3db2fe89bf561dd5e46a9` |
| `dart-d3/chart.html` | `4c0106eeb2d852023b7d68e71fdadcdb363e6e0f8984b573fc9c2a2f20c98f81` |
| `dart-d3/yearly_chart.html` | `03b5d9c89e901e79e28dcd99415b0ddb3322427a89cb03cb27eebfeb18653455` |
| `bluearf-html-repo/dashboard/dashboard-grafikleri/createVerticalBarChartD3.dart` | `fb6e169d632e6e89284f8807c22bb2b409e3793c75177202eb6dbff609ee06b4` |
| `bluearf-html-repo/dashboard/dashboard-grafikleri/createHorizontalStackedBarChartD3.dart` | `17c340a71099ed9a77af2344d7af7f045fc133604ada61d319e9818936d5348d` |
| `bluearf-html-repo/dashboard/dashboard-grafikleri/createVerticalAreaChartD3.dart` | `3e285bd33074cb46c90224188f8a2c66a11701949505b8c422341250027be6ca` |
| `bluearf-html-repo/dashboard/reporting/bluearf_charts.js` | `6fb1630665d26689f6028a84599e92f5313880f2aef90276386c2cdea704256f` |
| `bluearf-html-repo/dashboard/reporting/bluearf_charts.css` | `c9f5e272d8e1bdfbd71169840dbfcd1dab0eb29087d96f9183fe0c75526db1d0` |
| `bluearf-html-repo/dashboard/reporting/bluearf_charts.md` | `b7f362bd888c1cd17216552c44d49ebfc0bf8258a0b13b725be803fbb7f26c72` |
| `bluearf-html-repo/dashboard/reporting/assets/d3-7.9.0.min.js` | `f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539` |
| `bluearf-html-repo/dashboard/reporting/assets/jspdf-2.5.1.min.js` | `98ccf17aa10c20bb1301762618fcc9b6ab3a4e7f26b6071d64d0b41154df3875` |
| `bluearf-html-repo/dashboard/reporting/assets/xlsx-0.18.5.min.js` | `c9506197caf809a075b6dee1da0d36fb19da7158ffe8a88e7b0c96c5d8623c99` |
| `bluearf-html-repo/assets/fonts/barlow/Barlow-Regular.ttf` | `95aa02c7c43096e0dd44d787ba6216864a67157e402adab59b35572e0c1577ea` |
| `bluearf-website/assets/fonts/barlow/Barlow-Regular.woff2` (shipped) | `2904d763039d78176366e8e32c2c8cebecf2da19e249a7c077cd8c8a736c5cd4` |
| `bluearf-website/assets/fonts/barlow/OFL.txt` (shipped) | `186d750eb496a4c17a76385f82be6aea2ac1cf2de074a811d63786cf374ea73f` |
| `bluearf-html-repo/dashboard/reporting/assets/d3.LICENSE` (shipped) | `3e6849627f74ff73c257a3ae1efb574015d94fc1035c05ec3c15805165efcbc4` |

This source inventory does not itself certify the new package's browser behavior,
offline loading, export correctness or package independence. Those require
separate tests of the actual packaged implementation.
