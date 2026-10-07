# Chart assets and source attribution

The custom renderer and stylesheet adapt the user's existing Bluearf chart
components, with permission to port them into the Apache-2.0 OpenEconometrics project.
The inspected custom source repositories did not contain an explicit
repository-root license grant; this is not a claim that they were previously
distributed under Apache-2.0.

The implementation base is `bluearf-html-repo/dashboard/reporting/bluearf_charts.js`
and `bluearf_charts.css`. Its header attributes area gradients/reveal behavior
to `analytics_dashboard.html`, bar/stacking behavior to the dashboard Dart chart
generators, and donut/gauge behavior to the dashboard functions. The earlier
`dart-d3` generators share the Barlow, palette and interaction design family;
direct Git ancestry from those files has not been established.

OpenEconometrics adds validated Python table/model adapters, numeric-x line and scatter
charts, histograms and coefficient intervals, offline assets, safe standalone
HTML, instance lifecycle handling and statistical missing-value semantics.
Dashboard authentication, customer data, remote asset loading, PDF and XLSX
dependencies are not included. The repository document
`docs/charts-provenance.md` records the inspected source inventory and port
boundaries in more detail.

| Packaged asset | Original asset | License notice |
| --- | --- | --- |
| `d3.min.js` | D3 7.9.0 from `bluearf-html-repo/dashboard/reporting/assets/d3-7.9.0.min.js` | ISC; see `D3-LICENSE.txt` |
| `Barlow.woff2` | `bluearf-website/assets/fonts/barlow/Barlow-Regular.woff2` | SIL Open Font License 1.1; see `BARLOW-OFL.txt` |
| `Barlow.ttf`, embedded by `network-font.js` | The same bundled Barlow converted from CFF to TrueType quadratic outlines with maximum one font-unit error; cmap and advance metrics unchanged | SIL Open Font License 1.1; rebuild with `scripts/build_network_pdf_font.py` |

Third-party asset licenses remain in force independently of the Python
package's Apache-2.0 license. The regular Barlow font is unchanged; the renderer
also embeds the same bytes in exported SVG files.
The PDF uses the derived TrueType asset with embedded Unicode mapping, without
a font conversion library at runtime. Network layout, WebGL and PDF writers are
authored in this project. The ForceAtlas2 equations follow Jacomy et al. (2014),
[doi:10.1371/journal.pone.0098679](https://doi.org/10.1371/journal.pone.0098679);
this is an independent implementation, not a copied Gephi solver.

Source SHA-256 values:

- Shared JavaScript: `6fb1630665d26689f6028a84599e92f5313880f2aef90276386c2cdea704256f`
- Shared CSS: `c9f5e272d8e1bdfbd71169840dbfcd1dab0eb29087d96f9183fe0c75526db1d0`
- D3 JavaScript: `f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539`
- D3 license: `3e6849627f74ff73c257a3ae1efb574015d94fc1035c05ec3c15805165efcbc4`
- Barlow WOFF2: `2904d763039d78176366e8e32c2c8cebecf2da19e249a7c077cd8c8a736c5cd4`
- Barlow license: `186d750eb496a4c17a76385f82be6aea2ac1cf2de074a811d63786cf374ea73f`
