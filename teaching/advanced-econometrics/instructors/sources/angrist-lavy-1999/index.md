# Vendored source: Replication data for: Using Maimonides' Rule to Estimate the Effect of Class Size on Student Achievement

These files are the published data the teaching lab workbooks are rebuilt from. They are not
covered by the repository's Apache-2.0 licence; they keep their own terms.

- **Publisher:** Harvard Dataverse, <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XRSUJU>
- **Licence:** CC0 1.0 Universal (`CC0-1.0`), <https://creativecommons.org/publicdomain/zero/1.0/>
- **Licence evidence:** The Dataverse API record for version 1.0 gives license name "CC0 1.0" (SPDX CC0-1.0), empty terms of use, no guestbook and no file access request; read on 2026-10-18.
- **Cite the data as:** Angrist, J. D., & Lavy, V. (2009). Replication data for: Using Maimonides' Rule to Estimate the Effect of Class Size on Student Achievement (Version 1.0) [Dataset]. Harvard Dataverse. https://doi.org/10.7910/DVN/XRSUJU
- **Article:** Angrist, J. D., & Lavy, V. (1999). Using Maimonides' rule to estimate the effect of class size on scholastic achievement. Quarterly Journal of Economics, 114(2), 533-575. https://doi.org/10.1162/003355399556061
- **Retrieved:** 2026-10-18 from <https://dataverse.harvard.edu/api/access/datafile/1232180?format=original>. Dataverse version 1.0, released 2009-01-21. Files fetched through the access API: 1232180 (final5, original Stata format), 1232176 (AngristLavy_Table4.do) and 1232178 (readme.txt); each file's MD5 equals the checksum in the Dataverse record.
- **Used by:** `15-iv-moment-condition`

| File | Bytes | SHA-256 | Origin |
|---|---:|---|---|
| `final5.dta` | 241,713 | `bf443edee9033926b4e74dc2c17a98b04405b899735bf4c372e0e6d468431a54` | The depositors' original Stata file for fifth-grade classes, byte for byte (Dataverse MD5 ffca19b408107d5ff7bd50efe8278ddc). |
| `AngristLavy_Table4.do` | 2,119 | `ebdd331196a4aad50d5cb65a51949151daee2c155f17768f498baf72827baae8` | The depositors' Stata program for Table IV, byte for byte; it defines the sample filters the lab applies. |
| `readme.txt` | 971 | `c87254399a91d2e00e9cc0fd986d296011e1981f0b3814b16a7cccf0d9ca71b1` | The depositors' readme, byte for byte. |

None to the vendored files. The lab workbook applies the Table IV sample rules from AngristLavy_Table4.do, keeps the columns the lab needs and adds func1, as stated in its dataset.json filters. The generators in `../../generators/` check every SHA-256 above before reading
a file, and verification never downloads anything.
