# Vendored source: Wine Quality

These files are the published data the teaching lab workbooks are rebuilt from. They are not
covered by the repository's Apache-2.0 licence; they keep their own terms.

- **Publisher:** UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/186/wine+quality>
- **Licence:** Creative Commons Attribution 4.0 International (`CC-BY-4.0`), <https://creativecommons.org/licenses/by/4.0/>
- **Licence evidence:** The dataset landing page names the licence "Creative Commons Attribution 4.0 International" and links its legal code; read on 2026-10-18. The bundled winequality.names asks users to cite Cortez et al. (2009); that is a citation request, not an extra licence condition.
- **Cite the data as:** Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T
- **Article:** Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553. https://doi.org/10.1016/j.dss.2009.05.016
- **Retrieved:** 2026-10-18 from <https://archive.ics.uci.edu/static/public/186/wine+quality.zip>. UCI record last updated 2023-11-15
- **Used by:** `18-rank-comparisons`

| File | Bytes | SHA-256 | Origin |
|---|---:|---|---|
| `winequality-red.csv` | 84,199 | `4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e` | Member of the downloaded archive, byte for byte (semicolon-separated). |
| `winequality-white.csv` | 264,426 | `76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836` | Member of the downloaded archive, byte for byte (semicolon-separated). |
| `winequality.names` | 3,305 | `15d215e73b39105952380fd487e4ce1bf5a6b83a425abb285c2ec55193d3ca92` | Member of the downloaded archive, byte for byte (the publishers' description and citation request). |

None to the vendored files. The lab workbook stacks the red file above the white file, numbers the wines in that order and keeps colour and quality, as stated in its dataset.json filters. The generators in `../../generators/` check every SHA-256 above before reading
a file, and verification never downloads anything.
