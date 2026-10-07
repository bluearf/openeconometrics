# Third-party distribution notices

OpenEconometrics code is Apache-2.0 licensed. This does not replace upstream
licenses or grant redistribution rights for external research datasets.

| Distributed component | License and preserved notice |
| --- | --- |
| Custom chart adaptation | Project-owner permission; source attribution in the chart package's `assets/THIRD_PARTY.md` and `docs/charts-provenance.md` |
| D3 7.9.0 | ISC, `openecon_charts/assets/D3-LICENSE.txt` |
| Barlow font and derived PDF font | SIL OFL 1.1, `openecon_charts/assets/BARLOW-OFL.txt`; web font notice in `assets/BARLOW-OFL.txt` |
| React, CodeMirror, Lezer, KaTeX and other locked web dependencies | Full upstream license texts in the built interface's `THIRD-PARTY-NOTICES.txt`; package versions/integrities and text hashes in `third-party-manifest.json` |
| Firebase JavaScript modules | Apache-2.0 text retained from the pinned upstream Firebase release; original source copyright headers remain in the distributed code |
| KaTeX fonts | Upstream SIL OFL notice in the built interface's `licenses/KATEX-FONTS-LICENSE.txt` |
| Python/native runtime dependencies | Their original distribution metadata and license files are retained by the frozen packaging process; they are separate from the project license |
| Bundled uv executable | Upstream distribution license files in frozen `tools/licenses/uv` |
| Local suggestion engine/model | Separate notices in `desktop/licenses`; model resources and source hashes in the suggestion manifest |

`npm --prefix web run build` verifies the installed production dependencies
against `web/package-lock.json`, collects their full upstream notices and fails
on missing notices. The inventory includes the complete non-dev dependency
graph, a conservative superset of modules used in the browser bundle. Vite
copies the notices into the Python package's static resources. Wheel and frozen
runtime builds retain those resources; a source-only archive retains the public
notice inputs and generator. Regenerate notices after changing the lockfile.

The web font notice inputs include `web/public/licenses/upstream.json`, which
records their upstream Git blob IDs and SHA-256 values. Attribution is not a
claim that scientific reference data or software manuals are Apache-2.0 licensed.
External fixture redistribution must be reviewed separately before public
source publication. Public availability of a download alone is insufficient to
establish its redistribution license.
