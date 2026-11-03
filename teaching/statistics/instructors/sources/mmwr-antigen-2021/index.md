# Vendored source: Table 2 of Pray et al. (2021): Sofia SARS Antigen FIA against real-time RT-PCR by symptom status

These files are the published data the teaching lab workbooks are rebuilt from. They are not
covered by the repository's Apache-2.0 licence; they keep their own terms.

- **Publisher:** Centers for Disease Control and Prevention, Morbidity and Mortality Weekly Report, <https://www.cdc.gov/mmwr/volumes/69/wr/mm695152a3.htm>
- **Licence:** Public domain (MMWR series statement) (`US-Government-Public-Domain`), <https://www.cdc.gov/mmwr/about.html>
- **Licence evidence:** CDC's About MMWR page states that all material in the MMWR series is in the public domain and may be used and reprinted without special permission, and that citation as to source is appreciated; read on 2026-10-18. The PMC copy's licence block says the same.
- **Cite the data as:** Pray, I. W., Ford, L., Cole, D., et al. (2021). Performance of an antigen-based test for asymptomatic and symptomatic SARS-CoV-2 testing at two university campuses - Wisconsin, September-October 2020. MMWR Morbidity and Mortality Weekly Report, 69(51-52), 1642-1647. https://doi.org/10.15585/mmwr.mm695152a3
- **Article:** Same report. PMC copy: https://pmc.ncbi.nlm.nih.gov/articles/PMC9191905/
- **Retrieved:** 2026-10-18 from <https://pmc.ncbi.nlm.nih.gov/articles/PMC9191905/>. MMWR 69(51-52), published 2021-01-01. The CDC page carries the notice "This report has been corrected"; no erratum describing the correction was found. Table 2 is identical on the current CDC page and on PMC, and its cells add up to the Table 1 totals (39 true positives, 18 false negatives, 16 false positives, 1,025 true negatives).
- **Used by:** `05-bayes`

| File | Bytes | SHA-256 | Origin |
|---|---:|---|---|
| `table2_counts.csv` | 324 | `7b02a6ce1ab9f685118c279e3617e30f44da58b839889516178116b590651ee5` | The eight cell counts of Table 2, transcribed by OpenEconometrics; one row per symptom group, antigen result and RT-PCR result. |

The report publishes counts, not records. The lab workbook expands each count into that many rows (one per paired nasal swab) in the file's row order, as stated in its dataset.json filters. The generators in `../../generators/` check every SHA-256 above before reading
a file, and verification never downloads anything.
