# Real published data for the teaching labs: a proposal

*Proposal for review, 18 October 2026. This document changes no lab code, workbook, handout or reference. It covers the 60 labs in `statistics/`, `microeconomics/` and `advanced-econometrics/`.*

Every one of these labs currently ships an original synthetic workbook. The question here is which labs could instead use a **real, published dataset that may be redistributed in this public repository** and that answers the same question with the same method, and what each switch would cost.

## Summary

- **38 of the 60 labs have a usable candidate**: 18 in Statistics, 3 in Microeconomics and 17 in Advanced Econometrics. They are ranked in three tiers below. The 15 Tier A switches keep the lab's lesson intact and need only routine changes. The 17 Tier B switches work but change something the handout must address, such as independence, weighting or a data-use agreement. The 6 Tier C switches are possible but weaken or alter the lesson; they need an explicit decision.
- **22 labs should stay synthetic.** Seventeen are microeconomics theory labs, whose workbooks list points on a stated model function rather than observations. The other five need something real data cannot supply:
  - Advanced 03: the true regressor beside a mismeasured one.
  - Advanced 06: known error variances.
  - Advanced 16: one structural equation held fixed while only instrument strength changes.
  - Statistics 06: a known success probability.
  - Statistics 11: a natural null at a moderate sample size.

  If their Tier C options are declined, Microeconomics 02 and Advanced 19 also stay synthetic.
- **A small set of sources covers most of the candidates.** These are three UCI datasets (CC BY 4.0), World Bank WDI (CC BY 4.0), BEA/BLS series (US public domain), FARS and FHWA highway statistics (US public domain), the MMWR (public domain), and CC0 deposits on Harvard and Yale Dataverse: Tennessee's Project STAR, Galton's family heights, Gerber, Green & Larimer (2008), Angrist & Lavy (1999), Zame et al. (2026) and several further field experiments.
- **Switching any lab needs three cross-cutting changes first**, in addition to the per-lab work below:
  - The public-source allowlist in `scripts/prepare_public_source.py` would silently drop every real-data workbook.
  - The workbook verifier rebuilds each workbook from a seeded generator.
  - The packaging text states that all data are synthetic.

  See [Changes every switch needs](#changes-every-switch-needs).
- **Suggested first step:** a pilot covering Statistics 04, 05, 16 and 18 and Advanced 15. It exercises every cross-cutting change on small files with explicit licences before the larger switches.

## How candidates were chosen and checked

**Licence rule.** A dataset was accepted only when the publisher's own page states one of the following:
- public domain, or a US federal government work statement;
- CC0;
- CC BY (any version);
- a government open-data licence.

Rejected for this proposal:
- non-commercial or share-alike conditions;
- research-only terms;
- guestbook, registration or data-use-agreement downloads where redistribution is not stated;
- pages with no licence at all;
- textbook companion datasets, and R or Python package copies whose only source is a textbook.

Where a repository (Dataverse, ZBW, UCI) grants a licence for data that an original author created, both are cited. Datasets that were considered and rejected are listed in [Datasets considered and rejected](#datasets-considered-and-rejected).

**Same question, same method.** A candidate must support the lab's procedure without changing it: the same test, estimator, covariance and summary quantities. Its observational story may differ: a shopping basket can become an online retail invoice, and a quality alert can become an antigen test. Where the method must change, for example a clustered rather than independent covariance, the candidate is placed in Tier B or C and the change is named.

**Verification.** On 18 October 2026:
- Every URL in this proposal was fetched; URL patterns containing `{YYYY}`, `{M}` or `{ID}` were fetched for at least one value. Every licence statement was read on the publisher's page, and its wording is quoted below.
- Every article DOI was resolved against the Crossref REST API (`https://api.crossref.org/works/<doi>`). The volume, issue, pages and year given are Crossref's.
- Each candidate was downloaded where its licence allows, and the numbers marked *exploratory* below were computed from those files with throwaway scripts. They show whether the concept still comes through. They are not reference values: a lab rebuild must recompute them from its pinned extract.
- Nothing downloaded is committed.

**Privacy rule.** Requests for this work must never carry the project owner's email address or any other personal data, in a header, URL or payload. Some sites refuse anonymous script requests: `bls.gov`, `cdc.gov`, `bts.gov` and `nhtsa.gov` pages return HTTP 403, and `download.bls.gov` asks for a contact User-Agent. Those pages were read in a browser. Any file that can only be fetched that way is marked **manual download**, and a future rebuild must not work around this by adding personal data to requests.

## Ranking at a glance

| Rank | Lab | Real dataset | Licence | Rows the lab would use |
|---:|---|---|---|---:|
| **A1** | Statistics 05 · Quality alerts (Bayes) | Antigen test vs RT-PCR, MMWR 2021 | Public domain | 1,098 |
| **A2** | Statistics 04 · Conditional probability | UCI Online Shoppers Purchasing Intention | CC BY 4.0 | 12,330 |
| **A3** | Statistics 16 · Categorical independence | UCI Online Shoppers (visitor type × purchase) | CC BY 4.0 | 12,330 |
| **A4** | Statistics 18 · Rank comparisons | UCI Wine Quality (red vs white) | CC BY 4.0 | 6,497 |
| **A5** | Advanced 15 · IV moment condition | Angrist & Lavy (1999) Maimonides' rule | CC0 | 2,019 |
| **A6** | Advanced 08 · HAC inference | Okun's law from BEA GDP and BLS unemployment | Public domain | 287 |
| **A7** | Advanced 12 · Polynomial centering | WDI life expectancy and income (Preston curve) | CC BY 4.0 | 199 |
| **A8** | Advanced 11 · Common sample | WDI with education spending as the control | CC BY 4.0 | 199 → 164 |
| **A9** | Advanced 01 · Partialling out | WDI life expectancy, income, health spending | CC BY 4.0 | 186 |
| **A10** | Advanced 02 · Omitted-variable bias | Same WDI cross-section | CC BY 4.0 | 186 |
| **A11** | Statistics 02 · Which average | UCI Online Retail, one calendar week of invoices | CC BY 4.0 | ≈ 290 |
| **A12** | Statistics 19 · Bootstrap | UCI Online Retail, one trading day of invoices | CC BY 4.0 | ≈ 60–80 |
| **A13** | Statistics 14 · Two proportions | Gerber, Green & Larimer (2008) mailings: civic duty vs neighbours | CC0 | 76,419 |
| **A14** | Advanced 07 · Clustered inference | Project STAR kindergarten classes | CC0 | 5,853 in 325 classes |
| **A15** | Statistics 15 · Correlation | Galton's family heights (fathers and sons) | CC0 | 465 sons (173 families) |
| **B1** | Statistics 13 · Paired means | State road-death rates 2019 vs 2021 (FARS, FHWA) | Public domain | 51 pairs |
| **B2** | Advanced 10 · Delta method | Okun's law: growth that keeps unemployment constant | Public domain | 287 |
| **B3** | Statistics 07 · Normal model | NHANES 2021–2023 adult heights | Public domain + NCHS data-use agreement | 2,690 |
| **B4** | Statistics 12 · Independent means | Project STAR: small vs regular classes | CC0 | 3,794 students or 226 classes |
| **B5** | Statistics 17 · One-way ANOVA | Project STAR: three class types | CC0 | 5,871 students or 325 classes |
| **B6** | Statistics 20 · Statistical report | Karlan & List (2007) matching-grant letters | CC0 (manual download) | 50,083 |
| **B7** | Advanced 20 · Specification report | Alan, Boneva & Ertac (2019) grit intervention | CC0 | 2,575 or 1,499 |
| **B8** | Microeconomics 08 · Revealed preference | Zame et al. (2026) budget experiment | CC0 | 50 per subject |
| **B9** | Advanced 05 · Leverage and influence | WDI 2019 cross-section | CC BY 4.0 | 186 |
| **B10** | Advanced 17 · Logit marginal effects | UCI Online Shoppers (purchase on page value) | CC BY 4.0 | 12,330 |
| **B11** | Advanced 18 · Poisson with exposure | FARS deaths with FHWA vehicle-miles offset | Public domain | 663 |
| **B12** | Advanced 13 · Fixed effects | WDI country panel | CC BY 4.0 | balanced panel |
| **B13** | Advanced 14 · First differences | Same WDI country panel | CC BY 4.0 | balanced panel |
| **B14** | Statistics 03 · Spread | BTS on-time data: taxi-out minutes, one airport-day | Public domain | ≈ 350 |
| **B15** | Statistics 09 · Confidence intervals | BTS taxi-out with cancelled flights as blanks | Public domain | ≈ 70–340 |
| **B16** | Statistics 01 · What one row represents | BLS Consumer Expenditure Interview households | Public domain (manual download) | ≈ 4,700 |
| **B17** | Statistics 08 · Sampling means | UCI Online Retail invoices as a finite population | CC BY 4.0 | ≈ 19,900 |
| **C1** | Statistics 10 · Mean test | Michelson's 1879 speed-of-light trials | Public domain in the USA | 100 |
| **C2** | Microeconomics 20 · Estimated demand | US gasoline quantity, price and income (BEA) | Public domain | 67 years |
| **C3** | Advanced 19 · Local projections | Ramey–Zubairy military news shock | CC BY 4.0 (depositor) | 504 quarters |
| **C4** | Advanced 04 · Collinearity | WDI income and health spending (r = 0.91) | CC BY 4.0 | 186 |
| **C5** | Advanced 09 · Joint restrictions | WDI cross-section | CC BY 4.0 | 186 |
| **C6** | Microeconomics 02 · Comparative advantage | FAOSTAT crop yields | CC BY 4.0 + FAO promotion clause | 2 × 2 |

## Changes every switch needs

These are one-time changes, made in the pilot, that every later switch reuses.

1. **Public-source allowlist.** `teaching_workbook()` in `scripts/prepare_public_source.py` (lines 68–90) publishes a teaching workbook only when its `dataset.json` `source` begins with "Original synthetic OpenEconometrics teaching data;" or "Original OpenEconometrics teaching simulation.". A real-data workbook would therefore be silently left out of the public source snapshot. `tests/test_public_distribution.py` asserts that 79 workbooks are included and that a non-original source ("Customer data") is excluded. Proposed fix:
   - Add a second explicit admission rule: a `source` prefix such as "Published open data:", plus a `licence` field from a closed list (`CC0-1.0`, `CC-BY-4.0`, `US-Government-Public-Domain`, `Public-Domain-Mark`), a `licence_url` and a `citation`.
   - Extend the test with an admitted real-data case and a rejected unlisted licence.
2. **Workbooks from pinned extracts, not seeds.** `workbook_check()` in `scripts/verify_course_series.py` runs `instructors/generators/<lab>.py` `make_data()` and requires the workbook to equal its output to 1e-13. For a real-data lab:
   - `make_data()` should read a small **vendored raw extract** (for example `instructors/sources/<lab>/`, holding the publisher's rows before any lab filter, a licence or attribution note, and a SHA-256), then apply the lab's documented filters deterministically.
   - Verification then stays offline and reproducible even when publishers revise data, as WDI, BEA and FHWA do.
   - Keep verification offline. `scripts/verify_teaching_labs.py`, whose helpers the course verifier imports, documents that it needs "No empirical data, network access"; keep the no-network part and reword the rest once real data arrive.
3. **`dataset.json` fields.** For each real-data lab:
   - Replace the "original synthetic" `source` and drop `generation_seed`.
   - Add `publisher`, `dataset_url`, `dataset_doi` where one exists, `licence`, `licence_url`, `citation` (dataset and article), `retrieved` (date and the publisher's version or release), `raw_sha256` and `filters` (the row rules in words).
4. **`lab.py`.**
   - Set `metadata["synthetic"]` to `False`.
   - Replace the LaTeX caption suffix "(original synthetic data)".
   - Rename variables to the real ones and state the sample rule.
   - Regenerate `reference.json`, `table.tex` and `figure.svg`.
5. **Independent references.**
   - `independent_statistics()` in `scripts/verify_course_series.py` hard-codes column names and constants per lab number: Lab 06 `binom.sf(4, 10, 0.3)`, Lab 07 `norm.sf(1)`, null means 10 and 500 for Labs 09 and 10, group labels "A"/"B", and so on.
   - `advanced_econometrics()` in `scripts/teaching_references.py` hard-codes `x`, `z`, `y`, `control`, `group`, `unit`, `time`, `exposure` and `weight`. It also hard-codes design facts: Lab 05 deletes `data.iloc[1:]` because the flagged row is first; Lab 16 rebuilds `y` with the true coefficient 1.2.
   - Proposed fix: give each switched lab a `roles` map in `dataset.json` (for example `{"y": "avgverb", "x": "classize", "z": "func1", "controls": ["tipuach", "c_size"], "cluster": "schlcode"}`) that both `lab.py` and the reference read. Do not rename real columns to `x`, `y`, `z` in the student workbook.
6. **Handout and instructor companion.**
   - Replace the paragraph that says the file holds "original synthetic observations … not an empirical estimate".
   - Add a short **Data source and licence** section with the dataset citation, the article, the licence and the filters.
   - Rewrite the numbers and the worked interpretation, and update `instructors/<lab>.md`.
   - The journal case and further reading can often stay; several labs already cite the article whose data are proposed.
7. **Packaging text.** `scripts/build_teaching_handouts.py` prints "Original synthetic data" in the navigation (line 149) and footers (lines 470 and 565). It also prints "All observations are original synthetic teaching data." in `START-HERE.md` (line 616) and "Original synthetic teaching data." in the LaTeX book (line 686). These need a per-lab data statement.
8. **Attribution.**
   - CC BY requires attribution in the distributed copy: add a teaching-data table to `THIRD_PARTY.md` and keep the attribution in each workbook's **Dictionary** sheet.
   - CC0 and public-domain sources need no attribution, but cite them anyway. MMWR, BLS, BEA and EIA explicitly ask for a source citation.
   - `NOTICE` already says that external research fixtures are not relicensed under Apache-2.0. Keep that boundary.
9. **Vintages.** Record the publisher's release:
   - WDI `lastupdated` (2026-10-08 at retrieval);
   - BEA `NipaDataQ.txt` release (30 September 2026);
   - the FHWA VM-2 edition;
   - the FARS file dates (re-issued in 2026).

   Never re-fetch at verification time.

## Tier A: recommended

### A1 · Statistics 05 · What Does a Quality Alert Tell Us? → antigen test versus RT-PCR

- **Dataset.** The 2×2 counts of paired nasal swabs (Sofia SARS Antigen Fluorescent Immunoassay vs RT-PCR) by symptom status, Table 2 of Pray IW et al., "Performance of an Antigen-Based Test for Asymptomatic and Symptomatic SARS-CoV-2 Testing at Two University Campuses — Wisconsin, September–October 2020", *MMWR* 69(51–52), CDC.
  - Pages: <https://www.cdc.gov/mmwr/volumes/69/wr/mm695152a3.htm> and PMC9191905.
  - Size: eight counts, expanded to 1,098 rows (one row per paired specimen).
- **Licence.** "All material in the MMWR Series is in the public domain and may be used and reprinted without special permission; citation as to source, however, is appreciated." (<https://www.cdc.gov/mmwr/about.html>)
- **Article.** The same report, *MMWR* 69(5152): 1642–1647 (2021), doi:[10.15585/mmwr.mm695152a3](https://doi.org/10.15585/mmwr.mm695152a3) (Crossref ✓).
- **Counts** (read on cdc.gov and PMC, which agree):

  | Group | Antigen+ PCR+ | Antigen+ PCR− | Antigen− PCR+ | Antigen− PCR− | Total |
  |---|---:|---:|---:|---:|---:|
  | Asymptomatic | 7 | 14 | 10 | 840 | 871 |
  | Symptomatic | 32 | 2 | 8 | 185 | 227 |

- **Columns.**
  - `component_id` → `specimen_pair_id`. This is a label created when expanding the table; the source has no row order.
  - `defect` → `pcr_positive`.
  - `alert` → `antigen_positive`.
  - New column `symptomatic` (0/1).
- **What the lab shows.** The same test has a positive predictive value of 7/21 = 33.3% in asymptomatic people (prevalence 1.95%) and 32/34 = 94.1% in symptomatic people (prevalence 17.6%). Applying the asymptomatic sensitivity (0.412) and specificity (0.984) at the symptomatic prevalence already gives 84.3% (*exploratory*). The Bayes calculation therefore carries the lesson, which is stronger than the current synthetic one.
- **Changes.**
  - `lab.py` computes the table and the predictive values overall and by group. `ppv_at_one_percent_prevalence` can stay.
  - The handout changes from manufactured components to screening. The cited Gigerenzer & Hoffrage (1995) and Begg & Greenes (1983) remain apt.
  - The Lab 05 check in `independent_statistics()` changes its column names. Add a check that the expanded rows reproduce the published eight counts.
- **Risks.**
  - RT-PCR is the reference standard, not certain truth.
  - The unit is a swab pair: 57 people contributed more than one pair, so rows are not fully independent.
  - Sensitivity differs between the groups (41% vs 80%), so the handout must not attribute the whole PPV difference to prevalence.
  - The report carries a "This report has been corrected" notice whose content was not found. Read the correction before building.

### A2 · Statistics 04 · Which Denominator Belongs in a Probability? → returning visitors and purchases

- **Dataset.** Sakar C & Kastro Y (2018), *Online Shoppers Purchasing Intention Dataset*, UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>, doi:[10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).
  - Size: zip 1,072,219 bytes; 12,330 sessions × 18 columns; no missing cells.
- **Licence.** The landing page states "Creative Commons Attribution 4.0 International (CC BY 4.0)".
- **Article.** Sakar, Polat, Katircioglu & Kastro, "Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks", *Neural Computing and Applications* 31(10): 6893–6908, doi:[10.1007/s00521-018-3523-0](https://doi.org/10.1007/s00521-018-3523-0) (Crossref ✓; online May 2018, print October 2019).
- **Columns.**
  - `visit_id` → session row number (the source has no session ID).
  - `member` → `returning` = (`VisitorType` == "Returning_Visitor").
  - `purchase` → `Revenue`.
- **What the lab shows** (*exploratory*):
  - P(purchase) = 1,908/12,330 = 0.155.
  - P(purchase | returning) = 1,470/10,551 = 0.139.
  - P(returning | purchase) = 1,470/1,908 = 0.770.

  Returning visitors buy *less* often than other visitors, yet they make 77% of purchases. That is a sharper denominator lesson than the synthetic file gives.
- **Changes.**
  - `lab.py` builds the indicator. The handout must decide on the 85 "Other" visitor-type sessions; the simplest rule is to count them as not returning.
  - The Lab 04 check renames `member`/`purchase`.
- **Risks.**
  - Rows are sessions, not people. The source has no visitor ID, so one person can contribute many sessions.
  - There are 125 exact duplicate rows; keep them unless the source documents them as errors.
  - UCI's variable table mislabels `PageValues` as integer. This is irrelevant here.

### A3 · Statistics 16 · Are Shopping Channels and Membership Independent? → visitor type and purchase

- **Dataset, licence, article.** As A2.
- **Columns.**
  - `transaction_id` → session row.
  - `channel` (Store, Web, App) → `VisitorType` (New_Visitor, Other, Returning_Visitor).
  - `member` → `Revenue`.
- **What the lab shows** (*exploratory*):
  - Counts (no purchase / purchase): New 1,272/422, Other 69/16, Returning 9,081/1,470.
  - Pearson χ² = 135.3 on 2 df, p ≈ 4e−30.
  - Minimum expected count 13.2; Cramér's V = 0.105.

  The lab's lesson that a tiny p value can sit beside a weak association comes through clearly.
- **Changes.** As A2. The Lab 16 check uses the new column names.
- **Risks.** As A2. The rows are sessions, not people, so the count of independent units is uncertain, and the handout must say so.

### A4 · Statistics 18 · Comparing Ordinal Customer Ratings → red and white wine quality scores

- **Dataset.** Cortez P et al. (2009), *Wine Quality*, UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/186/wine+quality>, doi:[10.24432/C56S3T](https://doi.org/10.24432/C56S3T).
  - Size: zip 91,353 bytes, with `winequality-red.csv` (1,599 rows) and `winequality-white.csv` (4,898 rows), semicolon-separated.
- **Licence.** "Creative Commons Attribution 4.0 International (CC BY 4.0)" on the landing page. The bundled `winequality.names` asks users to cite the article; that is a request, not a licence condition.
- **Article.** Cortez, Cerdeira, Almeida, Matos & Reis, "Modeling wine preferences by data mining from physicochemical properties", *Decision Support Systems* 47(4): 547–553, doi:[10.1016/j.dss.2009.05.016](https://doi.org/10.1016/j.dss.2009.05.016) (Crossref ✓).
- **Columns.**
  - `customer_id` → wine row.
  - `service` (A/B) → `colour` (red/white).
  - `rating` (1–7) → `quality`. Integer on a 0–10 scale; 3–9 observed.
- **What the lab shows** (*exploratory*):
  - Both medians are 6, and 33.0% of red–white pairs are tied.
  - U_red = 3,311,514; the probability that a red outscores a white plus half the ties is 0.423.
  - z = −9.91, p = 3.6e−23.

  This is the point of the lab's own reference, Divine et al. (2018), "The Wilcoxon–Mann–Whitney procedure fails as a test of medians": equal medians and a decisive rank test at once.
- **Changes.**
  - `lab.py` reads two CSVs into one workbook and keeps the midrank, tie-corrected statistic.
  - The handout must stress effect size over the p value at n = 6,497.
  - The Lab 18 check renames groups.
- **Risks.**
  - Each score is the median of at least three expert assessors, not a customer rating.
  - Red and white wines are separate populations, not a randomized comparison.
  - All wines are Portuguese *vinho verde* from one study.

### A5 · Advanced 15 · Instrumental Variables and the Projection Formula → Maimonides' rule

- **Dataset.** Angrist JD & Lavy V, *Replication data for: Using Maimonides' Rule to Estimate the Effect of Class Size on Student Achievement*, Harvard Dataverse, doi:[10.7910/DVN/XRSUJU](https://doi.org/10.7910/DVN/XRSUJU).
  - Files: `final5.tab` (fifth grade, 388,814 bytes) and `final4.tab`, plus the authors' Stata files.
  - After the filters in `AngristLavy_Table4.do` (class size 2–44, enrolment > 5, `c_leom == 1`, `c_pik < 3`, reading score present): 2,019 classes in 1,002 schools.
- **Licence.** The Dataverse API reports `"license": {"name": "CC0 1.0", "uri": "http://creativecommons.org/publicdomain/zero/1.0"}`, with no extra terms of use.
- **Article.** Angrist & Lavy, "Using Maimonides' Rule to Estimate the Effect of Class Size on Scholastic Achievement", *Quarterly Journal of Economics* 114(2): 533–575 (1999), doi:[10.1162/003355399556061](https://doi.org/10.1162/003355399556061) (Crossref ✓). Follow-up: Angrist, Lavy, Leder-Luis & Shany, "Maimonides Rule Redux", *AER: Insights* 1(3): 309–324 (2019), doi:[10.1257/aeri.20180120](https://doi.org/10.1257/aeri.20180120) (Crossref ✓).
- **Columns.**
  - `y` → `avgverb` (class mean reading score).
  - `x` → `classize`.
  - `z` → `func1`, Maimonides' predicted class size, enrolment/(int((enrolment − 1)/40) + 1).
  - `control` → `tipuach` (% disadvantaged) and `c_size` (enrolment).
- **What the lab shows** (*exploratory*):
  - With `tipuach` only: first-stage slope 0.70, β_IV = −0.158.
  - Adding enrolment, as the paper does: first-stage slope 0.54, school-clustered F = 218, β_IV = −0.277.

  The projection formula, the first stage and the matrix check all carry over unchanged.
- **Changes.**
  - The lab's single `control` becomes two controls. Generalize `predictors = ["control", "x"]` and the instrument set in `advanced_econometrics()` through the `roles` map.
  - The reference uses non-robust covariance. Keep it for the exact algebra check, but report school-clustered inference in the handout.
  - Compare the rebuilt estimate with the published Table IV before writing the handout.
- **Risks.**
  - The instrument is valid only conditional on enrolment; dropping `c_size` changes the estimate.
  - Classes are nested in schools.
  - "Maimonides Rule Redux" documents enrolment manipulation around the cutoffs in later Israeli data. The handout should present it as a threat to the design.

### A6 · Advanced 08 · Serial Dependence and HAC Covariance → Okun's law

- **Data.**
  - US real GDP: BEA NIPA Table 1.1.6, series `A191RX`, from the flat file <https://apps.bea.gov/national/Release/TXT/NipaDataQ.txt> (35.4 MB, release of 30 September 2026). `SeriesRegister.txt` alongside it maps codes to tables.
  - Unemployment rate: BLS series `LNS14000000`, through the BLS Public Data API v1, which needs no key or personal data (`https://api.bls.gov/publicAPI/v1/timeseries/data/LNS14000000`; 10 years per query, 25 queries a day), averaged to quarters.
- **Licence.**
  - BEA: "Unless stated otherwise, the information posted on this web site is in the public domain and may be used or reproduced without specific permission." (<https://www.bea.gov/help/faq/147>)
  - BLS: everything it publishes "is in the public domain", except previously copyrighted photographs and illustrations, with a request to cite BLS (<https://www.bls.gov/opub/copyright-information.htm>).
- **Article.** Ball, Leigh & Loungani, "Okun's Law: Fit at 50?", *Journal of Money, Credit and Banking* 49(7): 1413–1441 (2017), doi:[10.1111/jmcb.12420](https://doi.org/10.1111/jmcb.12420) (Crossref ✓).
- **Columns.**
  - `y` → change in the quarterly-average unemployment rate (percentage points).
  - `x` → annualized quarter-on-quarter real GDP growth (%).
  - `z` → the previous quarter's growth.
  - `time` → quarter.
- **What the lab shows** (*exploratory*, without `z`), for 1948Q2–2019Q4, n = 287:
  - Δu = 0.218 − 0.0673 g.
  - Classical SE 0.0043; Newey–West (Bartlett, 4 lags) SE 0.0066.

  A clean 1.5× HAC correction.
- **Changes.**
  - `lab.py` and the reference read real column names.
  - The HAC call (`maxlags=4`) is unchanged.
  - The handout replaces the synthetic time series with Okun's law.
- **Risks.**
  - **End the sample at 2019Q4.** Through 2024Q4 (n = 307) the two 2020 quarters dominate: slope −0.113, classical SE 0.0060, HAC SE 0.031.
  - The BLS flat files at `download.bls.gov` refuse anonymous scripts. Use the API, never a personal User-Agent.
  - GDP is revised, so pin the BEA release.
  - October 2025 unemployment is missing because of the lapse in appropriations. This is outside the proposed sample.

### A7 · Advanced 12 · Centering a Polynomial without Changing Its Fit → the Preston curve

- **Data.** World Bank, *World Development Indicators*, for 2019: `SP.DYN.LE00.IN` (life expectancy at birth) and `NY.GDP.PCAP.PP.KD` (GDP per capita, PPP, constant international dollars). The API needs no key: `https://api.worldbank.org/v2/country/all/indicator/{ID}?date=2019&format=json&per_page=20000`. Aggregates are excluded with the country list's `region` field.
  - 199 economies have both.
- **Licence.**
  - <https://www.worldbank.org/ext/en/legal/terms-conditions/datasets>: "Unless specifically labeled otherwise", datasets are CC BY 4.0, with the attribution form "The World Bank: Dataset name: Data source".
  - The per-indicator metadata (`License_Type`) confirms CC BY 4.0 for all four indicators used in this proposal. Nine of 1,498 WDI series carry other terms (three CC BY 3.0 IGO, six SIPRI); none is used here.
- **Article.** Preston SH, "The Changing Relation between Mortality and Level of Economic Development", *Population Studies* 29(2): 231–248 (1975), doi:[10.1080/00324728.1975.10410201](https://doi.org/10.1080/00324728.1975.10410201) (Crossref ✓). The handout's current case, Murphy & Welch (1990) on age–earnings profiles, can stay as further reading.
- **Columns.**
  - `y` → life expectancy (years).
  - `x` → ln GDP per capita.
  - `z` → ln health spending per capita (`SH.XPD.CHEX.PC.CD`), or drop `z`.
- **What the lab shows** (*exploratory*, without `z`):
  - Raw fit: 9.74 ln Y − 0.202 (ln Y)².
  - Centred at mean ln Y = 9.634: 5.85 (ln Y − c) − 0.202 (ln Y − c)².
  - Maximum fitted difference 6e−12.

  The linear coefficient changes from 9.74 to 5.85 while the fit is identical: exactly the lesson.
- **Changes.**
  - Real column names and the `roles` map.
  - The handout replaces the synthetic quadratic with the concave income–longevity relation.
- **Risks.**
  - WDI revises data, so pin the vintage (`lastupdated` 2026-10-08).
  - The Central African Republic's 2019 life expectancy is 31.5 in a series that reads 52.3, 31.5, 50.6, 40.3 and 18.8 for 2018–2022. Check it in the pinned vintage, and either keep it with a note or exclude it by a stated rule.
  - Countries are not a random sample.

### A8 · Advanced 11 · Specification Changes and Common Samples → WDI with a patchy control

- **Data, licence, article.** As A7. The control is `SE.XPD.TOTL.GD.ZS` (government education spending, % of GDP).
- **Columns.**
  - `y` → life expectancy.
  - `x` → ln GDP per capita.
  - `z` → education spending (missing for 35 of the 199 economies).
- **What the lab shows.** 199 economies have life expectancy and GDP, and 164 remain once the control is added (*exploratory*). The sample and control components of the coefficient change are therefore both substantial; the synthetic file loses only 3 of 240 rows.
- **Changes.**
  - Real column names.
  - The handout must say which countries drop out and why.
- **Risks.**
  - The missing rows are not random (small, poor and conflict-affected economies), which is the lab's point but needs careful wording.
  - Vintage, as in A7.

### A9 · Advanced 01 · The Frisch–Waugh–Lovell Identity → life expectancy, income and health spending

- **Data, licence, article.** As A7. Health spending is `SH.XPD.CHEX.PC.CD`; 186 economies have all three variables.
- **Columns.**
  - `y` → life expectancy.
  - `x` → ln GDP per capita.
  - `z` → ln health spending per capita.
- **What the lab shows** (*exploratory*). The full-regression slope on ln GDP is 4.693 (HC3 SE 0.68), and residualizing both variables on `z` reproduces it exactly. The identity is algebraic, so real data illustrate it as cleanly as synthetic data.
- **Changes.** Real column names and the `roles` map. Nothing else.
- **Risks.** As A7. The handout must not read the coefficient causally.

### A10 · Advanced 02 · An Exact Sample Omitted-Variable Decomposition

- **Data, licence, article.** As A9.
- **What the lab shows** (*exploratory*):
  - Short slope on ln GDP 6.028; controlled slope 4.693.
  - Gap 1.335 = 1.018 (health-spending coefficient) × 1.311 (slope of ln health on ln GDP).
  - corr(ln GDP, ln health) = 0.913.
- **Changes, risks.** As A9.

### A11 · Statistics 02 · Which Average Answers the Question? → one week of online retail invoices

- **Dataset.** Chen D (2015), *Online Retail*, UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/352/online+retail>, doi:[10.24432/C5BW33](https://doi.org/10.24432/C5BW33).
  - Download: <https://archive.ics.uci.edu/static/public/352/online+retail.zip>, 23,715,478 bytes.
  - Contents: 541,909 lines × 8 columns from a UK online retailer, December 2010 to December 2011.
- **Licence.** "Creative Commons Attribution 4.0 International (CC BY 4.0)" on the landing page.
- **Article.** Chen, Sain & Guo, "Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining", *Journal of Database Marketing & Customer Strategy Management* 19(3): 197–208 (2012), doi:[10.1057/dbm.2012.17](https://doi.org/10.1057/dbm.2012.17) (Crossref ✓).
- **Columns.**
  - `basket_id` → `InvoiceNo`.
  - `spending` → Σ `Quantity` × `UnitPrice` over the invoice's lines (pounds sterling).
- **Filters** (all needed):
  - Drop cancellation invoices ("C…", 3,836) and bad-debt adjustments ("A…").
  - Drop invoices with any line where `Quantity` ≤ 0 or `UnitPrice` ≤ 0.
  - Drop invoices that a later cancellation reverses in full. Three of the five largest invoices in the file are such reversed orders.
- **What the lab shows** (*exploratory*, one calendar week, 7–13 March 2011):
  - 294 invoices; mean 455.07, median 277.65, geometric mean 233.98.
  - Largest 5,307.65; mean without it 438.51.

  The mean–median gap and the influence of one order are both visible.
- **Changes.**
  - `lab.py` aggregates lines to invoices by the stated rule.
  - The handout's currency becomes sterling and its story becomes an online retailer's orders. The UCI page notes that "many customers of the company are wholesalers".
- **Risks.**
  - Fix the week by a rule written before looking at the data (for example, the first full week of March 2011) to avoid choosing a convenient sample.
  - UCI's page says "no missing values", but `CustomerID` is blank in 135,080 lines. This is irrelevant to invoice totals; note it in the Dictionary.
  - There are 5,268 exact duplicate lines. Decide whether they are double entries.
  - **Alternative:** *Online Retail II* (doi:[10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D), CC BY 4.0, 45.6 MB) covers 2009–2011 with renamed columns. Its 2010–2011 sheet matches *Online Retail* apart from one line.

### A12 · Statistics 19 · Resampling the Observed Baskets → one trading day of invoices

- **Dataset, licence, article, filters.** As A11.
- **What the lab shows** (*exploratory*). Days have 11–142 invoices after filtering, with a median of 63. On 1 March 2011 there were 62 invoices: mean 425.08, median 262.73, maximum 2,541.15, skewness 2.6. With a skewed sample of this size, the bootstrap and t intervals visibly differ.
- **Changes.**
  - The resample count and seed stay; only the input changes.
  - The `numpy_empirical_distribution_standard_error` check reads the new column.
- **Risks.** As A11. Choose the day by a stated rule. The bootstrap seed remains, but it no longer generates the data.

### A13 · Statistics 14 · Did Two Messages Produce Different Response Rates? → two get-out-the-vote mailings

- **Dataset.** Gerber AS, Green DP & Larimer CW, *Social Pressure and Voter Turnout: Evidence from a Large-Scale Field Experiment*, Yale ISPS Dataverse, doi:[10.60600/YU/CGMWNW](https://doi.org/10.60600/YU/CGMWNW).
  - Size: individual-level file 20.1 MB as `.tab` (41.3 MB as `.dta`), with a household-level file and the authors' Stata and R code.
  - Content: 344,084 registered Michigan voters in 180,002 households, randomized by household before the August 2006 primary to a control group or one of four mailings.
- **Licence.** The Yale Dataverse API reports "CC0 1.0", with no terms of use and no guestbook. The old ISPS archive page now returns 404; the ISPS article page links to this deposit.
- **Article.** Gerber, Green & Larimer, "Social Pressure and Voter Turnout: Evidence from a Large-Scale Field Experiment", *American Political Science Review* 102(1): 33–48 (2008), doi:[10.1017/S000305540808009X](https://doi.org/10.1017/S000305540808009X) (Crossref ✓).
- **Columns.**
  - `recipient_id` → voter row.
  - `message` (A/B) → the "Civic Duty" mailing versus the "Neighbors" mailing.
  - `response` → `voted` (2006 primary).

  In the deposit these arms are `treatment` codes 2 and 3. The codes were matched to arms here by their published turnout; confirm them against the deposit's value labels.
- **What the lab shows** (*exploratory*):
  - Civic Duty 31.45% (n = 38,218); Neighbors 37.79% (n = 38,201).
  - Difference 6.34 points, pooled z = 18.4.

  Every turnout rate by arm matches the authors' Stata log.
- **Changes.**
  - `lab.py` selects two arms and keeps the pooled test and the unpooled interval.
  - The handout story becomes social pressure. Its existing references (Bertrand & Mullainathan 2004; Altman & Bland 1995; Lakens 2017) still fit.
  - The Lab 14 check renames the groups.
- **Risks.**
  - Households, not voters, were randomized. In these two arms the household-clustered standard error is 0.00429, against 0.00343 heteroskedasticity-robust (×1.25, 40,001 households). Either keep one voter per household by a stated rule, which preserves the lab's independent-groups test, or show the clustered value beside it.
  - With n ≈ 76,000 the interval is narrow, so the lesson shifts toward practical size.
  - The file holds individual turnout with sex and birth year from a public voter file. The student workbook should carry only the three columns the lab uses.
- **Alternatives.**
  - Fernandes, Huber & Plaza's Swiss apprenticeship correspondence test.
    - Data: Harvard Dataverse doi:[10.7910/DVN/PIUJW4](https://doi.org/10.7910/DVN/PIUJW4), CC0, `swissexperiment.tab` 219,366 bytes; 2,928 applications to 1,687 firms with randomized applicant gender.
    - Article: *Economics of Education Review* 95: 102399 (2023), doi:[10.1016/j.econedurev.2023.102399](https://doi.org/10.1016/j.econedurev.2023.102399) (Crossref ✓).
    - Results (*exploratory*): interview invitations went to 27.2% of men and 30.6% of women (pooled z = 2.05, p = 0.040). Taking only the first application to each firm gives 26.6% and 28.9%.
    - Caveats: 1,241 firms received two applications, so rows are clustered by firm. The deposit cites the 2019 working paper.
  - Gerber, Huber, Fang & Reardon, *PLOS ONE* 12(8): e0182199 (2017), doi:[10.1371/journal.pone.0182199](https://doi.org/10.1371/journal.pone.0182199), data doi:[10.7910/DVN/ZJINIB](https://doi.org/10.7910/DVN/ZJINIB) (CC0). It gives a clean null: 1.42% vs 1.39% turnout (n = 12,738).

### A14 · Advanced 07 · Clustered Errors and the Sampling Unit → Project STAR kindergarten classes

- **Dataset.** Achilles CM et al., *Tennessee's Student Teacher Achievement Ratio (STAR) project*, Harvard Dataverse, doi:[10.7910/DVN/SIWH9F](https://doi.org/10.7910/DVN/SIWH9F).
  - Size: `STAR_Students` is 13.1 MB as `.tab` (9.0 MB as SPSS), with school files and a user's guide.
  - Content: 11,601 students, of whom 6,325 have a kindergarten class assignment.
- **Licence.** The Dataverse API reports "CC0 1.0", with no terms of use and no guestbook. An archived HEROS web page inside the deposit carries an "All Rights Reserved" site footer; that is the old website's footer, not the data licence.
- **Articles.**
  - Krueger AB, "Experimental Estimates of Education Production Functions", *Quarterly Journal of Economics* 114(2): 497–532 (1999), doi:[10.1162/003355399556052](https://doi.org/10.1162/003355399556052) (Crossref ✓).
  - Finn JD & Achilles CM, "Answers and Questions About Class Size: A Statewide Experiment", *American Educational Research Journal* 27(3): 557–577 (1990), doi:[10.3102/00028312027003557](https://doi.org/10.3102/00028312027003557) (Crossref ✓).
- **Columns.**
  - `y` → `gktmathss` (kindergarten maths scaled score).
  - `x` → small class (`gkclasstype` == 1; regular and regular-with-aide classes form the comparison).
  - `z` → free lunch (`gkfreelunch` == 1).
  - `group` → `gktchid` (one teacher per class).
- **What the lab shows** (*exploratory*), for n = 5,853 students in 325 classes and 79 schools:
  - Small-class coefficient 7.71 points.
  - HC3 SE 1.35; clustered by class 3.05; clustered by school 2.32.

  Clustering more than doubles the standard error, which is exactly the lab's lesson about which observations carry independent information.
- **Changes.**
  - Clusters vary in size (about 18 students) instead of the synthetic six.
  - The reference reads the cluster column through the `roles` map; the cluster degrees of freedom become G − 1 = 324.
  - The handout can tie the case to Statistics 12 and 17 (B4, B5).
- **Risks.**
  - Assignment was random *within* schools, so the full analysis (Krueger 1999) adds school fixed effects. The handout should say the lab's regression omits them.
  - Class versus school clustering is a design choice to discuss.
  - The student file holds full dates of birth (`birthmonth`, `birthday`, `birthyear`). Leave them out of the teaching extract.

### A15 · Statistics 15 · Do Study Hours and Scores Move Together? → Galton's fathers and sons

- **Dataset.** *Galton height data*, Harvard Dataverse, doi:[10.7910/DVN/T0HSJ1](https://doi.org/10.7910/DVN/T0HSJ1), deposited by Jeremy Freese.
  - Size: `galton-stata11.tab`, 31,122 bytes.
  - Content: 898 children in 197 families, with columns `family`, `father`, `mother`, `gender`, `height` and `kids`; heights in inches; no missing cells.
- **Licence.** The Dataverse API reports "CC0 1.0", and the deposit's readme says the data are public domain for teaching or any other use. Galton collected the records in the 1880s.
- **Articles.**
  - Galton F, "Regression Towards Mediocrity in Hereditary Stature", *Journal of the Anthropological Institute of Great Britain and Ireland* 15 (1886), from p. 246, doi:[10.2307/2841583](https://doi.org/10.2307/2841583). Crossref ✓, though it gives only the first page.
  - Hanley JA, "'Transmuting' Women into Men", *The American Statistician* 58(3): 237–243 (2004), doi:[10.1198/000313004X1558](https://doi.org/10.1198/000313004X1558) (Crossref ✓). It documents the transcription of Galton's notebook.
- **Columns.**
  - `student_id` → child row.
  - `hours` → `father` (inches).
  - `score` → `height` of the son.
- **What the lab shows** (*exploratory*):
  - All 465 sons: r = 0.391.
  - The first son listed in each of the 173 families: r = 0.502.

  The scatter is the original regression-to-the-mean picture.
- **Changes.**
  - The handout story becomes inherited stature. Fisher (1915), already cited, stays.
  - The Lab 15 check renames columns.
- **Risks.**
  - Brothers are not independent. Use one son per family by a stated rule (n = 173) for the lab's t test of zero correlation, or say why all sons are kept.
  - The file has no blanks, so the lab's complete-case discussion becomes hypothetical.
  - **Provenance:** the deposit does not credit its transcription. It matches Hanley's (197 of 205 families, including family "136A"), and Hanley's own page states no licence. Cite Hanley for the transcription and rely on the CC0 deposit for redistribution.

## Tier B: workable with a stated caveat

### B1 · Statistics 13 · The Same Workers Before and After → state road-death rates, 2019 and 2021

- **Data.**
  - Deaths: NHTSA Fatality Analysis Reporting System (FARS) national CSV files, `https://static.nhtsa.gov/nhtsa/downloads/FARS/{YYYY}/National/FARS{YYYY}NationalCSV.zip` (14–35 MB a year; only the accident file's `STATE` and `FATALS` are needed).
  - Vehicle-miles: FHWA *Highway Statistics* table VM-2, `https://www.fhwa.dot.gov/policyinformation/statistics/{YYYY}/vm2.cfm` (with `.xls` files).
  - All 50 states and the District of Columbia (51 units) match for 2010–2022.
- **Licence.** NHTSA's and FHWA's site policies: "Information presented on this web site is considered public information and may be distributed or copied." (<https://www.nhtsa.gov/about-nhtsa/terms-use>, <https://www.fhwa.dot.gov/disclaim.cfm>)
- **Article.** Cohen A & Einav L, "The Effects of Mandatory Seat Belt Laws on Driving Behavior and Traffic Fatalities", *Review of Economics and Statistics* 85(4): 828–843 (2003), doi:[10.1162/003465303772815754](https://doi.org/10.1162/003465303772815754) (Crossref ✓). It uses FARS state-year data.
- **Columns.**
  - `worker_id` → state.
  - `before` → deaths per 100 million vehicle-miles in 2019.
  - `after` → the same rate in 2021.
- **What the lab shows** (*exploratory*):
  - n = 51; mean change +0.219 deaths per 100 million miles (SD 0.172); paired t = 9.10.
  - The correlation between the two years is 0.845, so pairing removes most between-state variation, which is the lab's lesson.
- **Changes.**
  - The handout story becomes the pandemic-era rise in road-death rates.
  - The lab has no missing pair. Either drop the "excluded pair" discussion or keep it as a hypothetical.
- **Risks.**
  - FHWA revises VM-2 and NHTSA re-issued the FARS files in 2026, so record both editions.
  - States are a population, not a sample. The handout must say what the t interval means here: variation across states, not sampling error.
  - nhtsa.gov pages return 403 to scripts, but static.nhtsa.gov and fhwa.dot.gov files download normally.

### B2 · Advanced 10 · Delta-Method Inference for a Coefficient Ratio → Okun's "stall speed"

- **Data, licence, article.** As A6.
- **What the lab shows** (*exploratory*). The growth rate that holds unemployment constant is g* = −â/b̂ = 0.2176/0.0673 = 3.23% a year (1948Q2–2019Q4). Its delta-method standard error uses the HAC covariance.
- **Changes.**
  - The lab's ratio changes from b_x/b_z (two slopes) to −b₀/b₁ (intercept over slope).
  - The gradient in `advanced_econometrics()` changes from `[0, 1/b2, −b1/b2²]` to `[−1/b1, b0/b1²]`.
  - Covariance becomes HAC.
- **Risks.**
  - The ratio's form changes, a small method change that the handout must explain.
  - The denominator is well away from zero (t ≈ 10), so the delta method behaves.

### B3 · Statistics 07 · From Parcel Weights to Standard Scores → adult heights

- **Data.** CDC/NCHS, National Health and Nutrition Examination Survey 2021–2023:
  - `BMX_L.xpt` (1,563,200 bytes) and `DEMO_L.xpt` (2,582,160 bytes) at `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/`.
  - Variables `BMXHT` (standing height, cm), `RIAGENDR` and `RIDAGEYR`.
- **Licence and conditions.**
  - CDC: "Most of the information on the CDC and ATSDR websites is not subject to copyright, is in the public domain…", with attribution, a non-endorsement disclaimer and a request not to change substantive content (<https://www.cdc.gov/other/agencymaterials.html>).
  - The **NCHS Data User Agreement** travels with the data (<https://www.cdc.gov/nchs/policy/data-user-agreement.html>): use "for statistical reporting and analysis only", "make no attempt to learn the identity of any person", and do not link the data with individually identifiable data.
- **Articles.**
  - Schilling, Watkins & Watkins, "Is Human Height Bimodal?", *The American Statistician* 56(3): 223–229 (2002), doi:[10.1198/00031300265](https://doi.org/10.1198/00031300265) (Crossref ✓). It explains why one sex at a time is the right population.
  - Komlos & Lauderdale, "Underperformance in Affluence: The Remarkable Relative Decline in U.S. Heights in the Second Half of the 20th Century", *Social Science Quarterly* 88(2): 283–305 (2007), doi:[10.1111/j.1540-6237.2007.00458.x](https://doi.org/10.1111/j.1540-6237.2007.00458.x) (Crossref ✓). It uses NHANES heights.
- **Columns.**
  - `parcel_id` → row number. Drop `SEQN` from the student file to keep linkage out of scope.
  - `grams` → `height_cm`, for men aged 20 or over.
- **What the lab shows** (*exploratory*):
  - Men: n = 2,690, mean 174.47 cm, SD 7.74.
  - Observed share above mean + 1 SD 0.1595 (normal 0.1587); within ±2 SD 0.9524 (normal 0.9545).

  The fit is very close to normal.
- **Changes.**
  - `lab.py` filters sex and age.
  - The Lab 07 check keeps `norm.sf(1)`.
  - The handout must reproduce the NCHS data-use terms.
- **Risks.**
  - The data-use agreement is a condition on users, not a ban on redistribution. Have the project owner confirm it is acceptable in a public repository.
  - The sample is unweighted, so the mean describes the examined sample, not the US population (`WTMEC2YR` exists).
  - 67 examined adults lack a height.

### B4 · Statistics 12 · Comparing Two Delivery Methods → small and regular kindergarten classes

- **Dataset, licence, articles.** As A14 (Project STAR, CC0).
- **Columns.**
  - `delivery_id` → student, or class.
  - `method` (A/B) → small or regular class; regular-with-aide classes are left out.
  - `minutes` → `gktmathss`.
- **What the lab shows** (*exploratory*):
  - Students: small 1,762 (mean 490.9, SD 49.5) and regular 2,032 (mean 483.2, SD 47.6). The Welch difference is 7.73 points, t = 4.88, df 3,672, 95% CI 4.63 to 10.84.
  - Class means (127 small, 99 regular): difference 7.07, Welch t = 1.89, df 213, p = 0.060.
- **Why Tier B.**
  - Students in one class share a teacher and a room, so the student-level Welch test treats dependent rows as independent and overstates precision.
  - Using class means restores the lab's independence assumption, but it changes the unit and the conclusion's strength.
  - Both versions ignore the within-school randomization.
  - This is a strong lesson, but the handout must make the unit choice explicit and link to Advanced 07.

### B5 · Statistics 17 · Comparing Three Training Formats → three class types

- **Dataset, licence, articles.** As A14.
- **Columns.**
  - `trainee_id` → student, or class.
  - `format` (A, B, C) → small, regular, regular with aide.
  - `score` → `gktmathss`.
- **What the lab shows** (*exploratory*):
  - Students: F = 17.2, p = 3.6e−8, η² = 0.006.
  - Class means: F = 3.04, p = 0.049.
- **Why Tier B.** The same unit question as B4. The small η² makes a good discussion of significance versus size.

### B6 · Statistics 20 · From a Question to a Statistical Report → a matching-grant fundraising letter

- **Dataset.** Karlan D & List JA, replication data, Harvard Dataverse, doi:[10.7910/DVN/27853](https://doi.org/10.7910/DVN/27853).
  - Size: `AER merged.tab`, 11,223,292 bytes.
  - Content: the variable metadata, readable without download, lists 50,083 prior donors and 51 variables, including `treatment`, `ratio`, `size`, `ask`, `gave`, `amount`, `hpa` and `freq`. 16,687 were in the control group.
- **Licence.** The Dataverse API reports "CC0 1.0", but every file is behind guestbook 80, which asks for the downloader's details. This is a **manual download** for the project owner; no agent may fill in the form. CC0 allows redistribution of the downloaded file.
- **Article.** Karlan & List, "Does Price Matter in Charitable Giving? Evidence from a Large-Scale Natural Field Experiment", *American Economic Review* 97(5): 1774–1793 (2007), doi:[10.1257/aer.97.5.1774](https://doi.org/10.1257/aer.97.5.1774) (Crossref ✓).
- **Columns.**
  - `customer_id` → donor row.
  - `offer` (A/B) → `treatment`: standard letter versus a matching-grant offer.
  - `spending` → `amount` given.
  - `returned` → `gave`.
- **What the lab shows.** Not computed, because the file was not downloaded.
- **Risks.**
  - `amount` is zero for everyone who did not give, so the spending comparison is dominated by the giving rate. The handout must present both outcomes, as the lab already does.
  - The synthetic file's six blank spending cells have no counterpart.
  - The openICPSR copy (doi:10.3886/E113224V1) could not be checked.

### B7 · Advanced 20 · A Reproducible Treatment-Effect Report → a grit intervention in Turkish schools

- **Dataset.** Alan S, Boneva T & Ertac S, replication data, Harvard Dataverse, doi:[10.7910/DVN/SAVGAL](https://doi.org/10.7910/DVN/SAVGAL).
  - `Sample1_Data.tab` (610,038 bytes): 2,575 students in 36 schools.
  - `Sample2_Data.tab` (416,506 bytes): 1,499 students in 16 schools.
- **Licence.** The Dataverse API reports "CC0 1.0", with no terms of use and no guestbook.
- **Article.** Alan, Boneva & Ertac, "Ever Failed, Try Again, Succeed Better: Results from a Randomized Educational Intervention on Grit", *Quarterly Journal of Economics* 134(3): 1121–1162 (2019), doi:[10.1093/qje/qjz006](https://doi.org/10.1093/qje/qjz006) (Crossref ✓).
- **Columns.**
  - `treat` → school-level treatment.
  - `y` → `mathscore2` (standardized post-test).
  - `x` → `mathscore1`.
  - `z` → `verbalscore1`.
  - `interaction` → treatment × `mathscore1`.
- **What the lab shows** (*exploratory*, Sample 2, point estimates only): treatment 0.176, baseline 0.483, interaction −0.044.
- **Why Tier B.** Treatment was assigned by school, so the lab's HC3 inference for individually randomized units must become school-clustered. With 16 clusters (Sample 2) cluster-robust inference is unreliable. Sample 1's 36 schools are better, but its outcome variables have not been inspected yet.
- **Alternatives** (both CC0 with no guestbook, checked through the Dataverse API):
  - Haushofer & Shapiro, *QJE* 131(4): 1973–2042 (2016), doi:[10.1093/qje/qjw025](https://doi.org/10.1093/qje/qjw025), data doi:[10.7910/DVN/M2GAZN](https://doi.org/10.7910/DVN/M2GAZN). Use the within-village comparison only (n = 940), because pure-control households have no baseline survey.
  - Armantier et al., *Review of Economics and Statistics* 98(3): 503–523 (2016), doi:[10.1162/REST_a_00499](https://doi.org/10.1162/REST_a_00499), data doi:[10.7910/DVN/27102](https://doi.org/10.7910/DVN/27102). Randomization is individual (705 respondents), and the prior inflation forecast is a natural baseline for the interaction. It needs the authors' cleaning script and a review of free-text answers.

### B8 · Microeconomics 08 · Affordability and Revealed Preference → laboratory budget choices

- **Dataset.** Zame, Tungodden, Sørensen, Kariv & Cappelen, Harvard Dataverse, doi:[10.7910/DVN/LUF59R](https://doi.org/10.7910/DVN/LUF59R).
  - File: `mmzame-decisions.tab` (2.64 MB).
  - Content: 310 subjects × 50 linear two-good budgets in each treatment, with columns `id`, `treatment`, `round`, `maxx`, `maxy`, `x`, `y`.
- **Licence.**
  - The Dataverse API reports "CC0 1.0" with no terms of use.
  - The package README says "the data are licensed under a Creative Commons/CC0 license".
- **Article.** "Linking Social and Personal Preferences: Theory and Experiment", *Journal of Political Economy* 134(6): 1890–1930 (2026), doi:[10.1086/739832](https://doi.org/10.1086/739832) (Crossref ✓). The handout already cites Varian (1982).
- **Columns.**
  - `choice_id` → `round`.
  - `px` → 1/`maxx`; `py` → 1/`maxy`; `income` → 1.
  - `x`, `y` → `x`, `y`, for one subject in the "dictator" (self-versus-other) treatment.
- **What the lab shows.** 53.5% of subjects satisfy GARP in the dictator task (*exploratory*), so the lab can show one consistent subject and one with violations.
- **Changes.**
  - Four observations become fifty.
  - The reference's strict-affordability test runs as is, but its 1e−10 tolerance must become a stated budget tolerance: 261 of 61,998 choices sit more than 0.02 off their budget line, mostly on budgets with an intercept near zero.
  - Choose subjects by a written rule.
- **Risks.**
  - Picking a subject is a choice the handout must disclose.
  - "Income = 1" is a normalization.
  - **Alternative:** the aggregate test the handout cites, Varian (1982), using BEA personal consumption by type of product (Tables 2.4.4U/2.4.5U in the same public-domain flat files as A6), at the cost of a representative-consumer assumption.

### B9 · Advanced 05 · Leverage and Leave-One-Out Sensitivity → WDI 2019

- **Data, licence, article.** As A9.
- **What the lab shows** (*exploratory*), for life expectancy on ln GDP and ln health spending (n = 186):
  - The highest-leverage economy is Tuvalu (h = 0.079, against an average of 0.016). Deleting it moves the ln GDP slope from 4.693 to 4.556.
  - The most *influential* row is the Central African Republic, whose 31.5-year life expectancy looks like a data problem (see A7). Deleting it moves the slope to 3.95.
- **Changes.** `advanced_econometrics()` deletes `data.iloc[1:]` because the synthetic flagged row is the first. It must instead drop the flagged row by identifier, and `lab.py` must flag it by a stated rule.
- **Risks.** Using a suspected data error as the influential case is a strong lesson, but it depends on the pinned vintage. A later WDI revision may remove it.

### B10 · Advanced 17 · Logit Probabilities, Odds and Marginal Effects → online purchases

- **Data, licence, article.** As A2.
- **Columns.**
  - `y` → `Revenue`.
  - `x` → ln(1 + `PageValues`). `PageValues` is zero in 77.9% of sessions; their purchase rate is 3.9%.
  - `z` → `ExitRates`.
- **What the lab shows** (*exploratory*). Log-odds coefficient 1.046 on ln(1 + PageValues), and average marginal effect 0.080 per log point. The average predicted probability equals the observed rate of 0.155, as the logit score equations require.
- **Changes.**
  - Real names.
  - The handout explains the log transform.
- **Risks.** `PageValues` is computed from pages viewed before a transaction, so it is partly mechanical with the outcome. The handout must keep the interpretation descriptive (probability versus odds), which is the lab's point anyway.

### B11 · Advanced 18 · Poisson Rates with an Exposure Offset → road deaths per vehicle-mile

- **Data, licence.** As B1, for 2010–2022 (663 state-years).
- **Article.** As B1.
- **Columns.**
  - `y` → deaths.
  - `exposure` → vehicle-miles (millions).
  - `x` → centred year.
  - `z` → an indicator for 2020–2022, or another state-year covariate assembled from the same sources.
- **What the lab shows** (*exploratory*). The rate ratio is 1.017 a year.
- **Changes.** **Inference must change.** The Pearson χ²/df is 36.4, so model-based Poisson standard errors are far too small. The lab and the reference would need robust (sandwich) or quasi-Poisson standard errors, and the handout must discuss over-dispersion.
- **Risks.** The covariance change is a method change, which places this in Tier B. VM-2 and FARS vintages, as in B1.

### B12 · Advanced 13 · Fixed Effects as Dummies and Within Variation → a WDI country panel

- **Data, licence.** As A7, for 2000–2019 at a fixed interval (for example 2000, 2005, 2010, 2015, 2019) and the countries with all values in every period. The panel has not been assembled yet.
- **Article.** Preston (1975), as A7.
- **Columns.**
  - `y` → life expectancy.
  - `x` → ln GDP per capita.
  - `z` → ln health spending.
  - `unit` → country; `time` → year.
- **Expected lesson.** The within-country slope is much smaller than the cross-country slope, and the dummy and demeaned estimates agree. The FE algebra holds exactly for any panel.
- **Changes.**
  - The reference builds unit dummies from `unit`, which is already generic once the `roles` map exists.
  - Cluster by country.
- **Risks.**
  - The balanced-panel rule drops countries non-randomly.
  - Health-spending data start in 2000.
  - These numbers have not been computed yet.

### B13 · Advanced 14 · First Differences and Their Error Structure

- **Data, licence, article, columns.** As B12.
- **Changes.** None beyond B12. The lab's lesson that differencing induces MA(1) dependence when the level errors are white is a statement about the data-generating process; with real data the handout can only show the observed autocorrelation of the differenced residuals.
- **Risks.** As B12. The DGP statement becomes descriptive.

### B14 · Statistics 03 · How Variable Are Delivery Times? → aircraft taxi-out times

- **Data.** Bureau of Transportation Statistics, *Reporting Carrier On-Time Performance (1987–present)*.
  - Monthly zips at `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YYYY}_{M}.zip`; January 2024 is 27,573,265 bytes, holding 547,271 flights.
  - Fields used: `TaxiOut`, `Cancelled`, `Origin`, `Reporting_Airline`.
- **Licence.**
  - The TranStats download page shows no licence text.
  - A BTS publication, the *Transportation Statistics Annual Report 2016*, states in its front matter: "All material contained in this report is in the public domain and may be used and reprinted without special permission. Source citation is required." (<https://www.bts.gov/archive/publications/transportation_statistics_annual_report/2016/front>)
  - data.bts.gov labels its on-time summary dataset "Public Domain U.S. Government".
  - Treat it as a US federal government work, with that caveat recorded.
- **Article.** Mayer C & Sinai T, "Network Effects, Congestion Externalities, and Air Traffic Delays: Or Why Not All Delays Are Evil", *American Economic Review* 93(4): 1194–1215 (2003), doi:[10.1257/000282803769206269](https://doi.org/10.1257/000282803769206269) (Crossref ✓). The handout's travel-time-variability references stay.
- **Columns.**
  - `delivery_id` → flight (carrier and flight number).
  - `minutes` → `TaxiOut`, for San Francisco (SFO) departures on one day.
- **What the lab shows** (*exploratory*). On 4 January 2024 there were 355 departures and no cancellations; taxi-out mean 17.59 minutes, SD 5.43.
- **Changes.** `lab.py` reads one airport-day; the variance, SD, IQR and coefficient of variation are unchanged.
- **Risks.**
  - Only about 355 of 547,271 rows are needed, so vendor the extract, not the month.
  - bts.gov pages return 403 to scripts; transtats.bts.gov downloads work.
  - Choose the airport and day by rule.

### B15 · Statistics 09 · What Does a Confidence Interval Cover? → taxi-out with cancelled flights

- **Data, licence, article.** As B14.
- **Columns.**
  - `encounter_id` → flight.
  - `minutes` → `TaxiOut`, blank when `Cancelled == 1`.
- **What the lab shows** (*exploratory*). SFO on 15 January 2024 had 338 departures, 37 of them cancelled. Taxi-out for the 302 flights that left: mean 17.25, SD 9.35. A smaller airport-day would give the synthetic file's scale of about 70 rows.
- **Changes.** The handout must say the interval describes departures that happened. Cancellations are not missing at random.
- **Risks.**
  - Missing `TaxiOut` and `Cancelled` are not the same thing: 132 cancelled flights nationally still carry a taxi-out value. Define blanks by `Cancelled`.
  - The informative missingness is a fair lesson, but a different one from the synthetic file's two random blanks.

### B16 · Statistics 01 · What Does One Row Represent? → Consumer Expenditure Survey households

- **Data.** BLS, *Consumer Expenditure Surveys Public-Use Microdata*, 2024 Interview survey CSV, `https://www.bls.gov/cex/pumd/data/csv/intrvw24.zip` (53,613,162 bytes). **Manual download**: bls.gov refuses anonymous scripts, and no personal data may be added to requests.
  - The second-quarter household file `FMLI` has 4,673 households.
- **Licence.** BLS: "everything that we publish… is in the public domain, except for previously copyrighted photographs and illustrations… we do ask that you cite the Bureau of Labor Statistics as the source." (<https://www.bls.gov/opub/copyright-information.htm>) No use agreement or registration was found.
- **Articles.**
  - Attanasio OP & Pistaferri L, "Consumption Inequality", *Journal of Economic Perspectives* 30(2): 3–28 (2016), doi:[10.1257/jep.30.2.3](https://doi.org/10.1257/jep.30.2.3) (Crossref ✓).
  - Aguiar M & Hurst E, "Deconstructing Life Cycle Expenditure", *Journal of Political Economy* 121(3): 437–492 (2013), doi:[10.1086/670740](https://doi.org/10.1086/670740) (Crossref ✓).
- **Columns.**
  - `household_id` → `NEWID`.
  - `region` → `REGION` (1–4 Census regions; 93 blank).
  - `household_size` → `FAM_SIZE`.
  - `monthly_spending` → (`TOTEXPPQ` + `TOTEXPCQ`)/3.
- **Changes.**
  - `lab.py` builds the monthly total.
  - The handout explains quarterly reference periods and survey weights (`FINLWT21`).
- **Risks.**
  - The blanks move from spending to region: CE totals are imputed, so the "blank is not zero" lesson must be retold with region.
  - Unweighted means do not describe the US population, which suits the lab's survey-design paragraph.
  - The file is topcoded.

### B17 · Statistics 08 · Why Do Averages Vary Less Than Observations? → all invoices as a finite population

- **Data, licence, article, filters.** As A11. The whole filtered file is the population: 19,903 invoices, mean 516.35, maximum 52,940.94 (*exploratory*).
- **Changes.**
  - `population_sd_denominator_N` and the n = 5 and n = 40 resampling are unchanged.
  - The handout describes the file as one retailer's complete year.
- **Risks.** The population is extremely skewed, so the sampling distribution at n = 40 is still visibly right-skewed and the normal approximation is poorer than in the synthetic file. This is honest, but the lab's message ("averages vary less") remains true while "and look normal" weakens. A one-month population would be less extreme.

## Tier C: possible, needs a decision

### C1 · Statistics 10 · Is the Filling Process Centered on Its Target? → Michelson's 1879 speed of light

- **Data.** A. A. Michelson's 100 trials of June–July 1879, from his report in *Astronomical Papers of the American Ephemeris* vol. 1. The values are transcribed in Project Gutenberg eBook #11753 (<https://www.gutenberg.org/ebooks/11753>), and they match NIST StRD `Michelso.dat` exactly.
- **Licence.**
  - The Gutenberg page marks the work "Public domain in the USA."
  - The Internet Archive scan of the volume is labelled NOT_IN_COPYRIGHT.
  - Use the Gutenberg transcription, not NIST: the NIST StRD pages carry no licence statement, and NIST's licence page says Standard Reference Data may be copyrighted.
- **Article.** Stigler SM, "Do Robust Estimators Work with Real Data?", *Annals of Statistics* 5(6): 1055–1098 (1977), doi:[10.1214/aos/1176343997](https://doi.org/10.1214/aos/1176343997) (Crossref ✓).
- **Columns.**
  - `container_id` → trial.
  - `milliliters` → speed (km/s).
- **What the lab shows** (*exploratory*):
  - Mean 299,852.4 km/s, SD 79.0.
  - t = 7.59 against the defined vacuum value 299,792.458 km/s.
- **Why Tier C.**
  - **Target:** the trials are uncorrected speeds in air. The commonly quoted target of 299,734.5 comes from secondary sources quoting Stigler; the full paper could not be opened to confirm it.
  - **Independence:** the lag-1 autocorrelation is 0.535, so the t test's standard error is too small. Both points make a good lesson about systematic error, but not the clean one-sample test the lab now teaches.

### C2 · Microeconomics 20 · From a Demand Model to an Estimated Elasticity → US gasoline

- **Data.** BEA NIPA Tables 2.4.4U/2.4.5U, line 115, from the public-domain flat files in A6:
  - `DGASRX`: real personal consumption of gasoline and other energy goods.
  - `DGASRG`: its price index.
  - Annual 1959–2025, quarterly to 2026Q2.
  - Real disposable income comes from the same file. Its series code is still to be confirmed.

  EIA product supplied (`MGFUPUS2`, <https://www.eia.gov/dnav/pet/hist_xls/MGFUPUS2m.xls>; "U.S. government publications are in the public domain") is a quantity alternative.
- **Article.** Hughes, Knittel & Sperling, "Evidence of a Shift in the Short-Run Price Elasticity of Gasoline Demand", *The Energy Journal* 29(1): 113–134 (2008), doi:[10.5547/ISSN0195-6574-EJ-Vol29-No1-9](https://doi.org/10.5547/ISSN0195-6574-EJ-Vol29-No1-9) (Crossref ✓). It is already cited in Microeconomics 04.
- **Why Tier C.** The synthetic lab draws prices independently of demand shocks, so its OLS elasticity is causal. Real prices are not exogenous, and annual levels trend together. The handout's own question (identification versus fit) would become the main story, and HC3 should become HAC. Same estimator, different lesson.

### C3 · Advanced 19 · Local Projections over Several Horizons → a narrative military-news shock

- **Data.** Ramey & Zubairy's quarterly series (1875–2015; `news` from 1890), as redistributed in the ZBW JAE Data Archive, doi:[10.15456/jae.2024364.1002804655](https://doi.org/10.15456/jae.2024364.1002804655) (<https://journaldata.zbw.eu/dataset/dynamic-effects-of-persistent-shocks>; `DATA.zip`, 687 KB, `RZDAT.xlsx`). The deposit accompanies Alloza, Gonzalo & Sanz, "Dynamic Effects of Persistent Shocks", *Journal of Applied Econometrics* 40(4): 380–394 (2025), doi:[10.1002/jae.3115](https://doi.org/10.1002/jae.3115) (Crossref ✓).
- **Licence.** DataCite and the page metadata state CC BY 4.0. Ramey's own page states no licence, so the grant is the depositor's; cite both.
- **Article.** Ramey & Zubairy, "Government Spending Multipliers in Good Times and in Bad: Evidence from US Historical Data", *Journal of Political Economy* 126(2): 850–901 (2018), doi:[10.1086/696277](https://doi.org/10.1086/696277) (Crossref ✓). The handout already cites Jordà (2005).
- **Why Tier C.**
  - The lab reports `population_h2_response`, the true impulse response, which no real dataset has. That key and its comparison must go.
  - The shock series is dominated by the world wars.
- **Alternative.** Känzig's oil-supply surprise (CC0; <https://github.com/dkaenzig/replicationOilSupplyNews>; *AER* 111(4): 1092–1125, doi:[10.1257/aer.20190964](https://doi.org/10.1257/aer.20190964), Crossref ✓). It is monthly 1975–2017 but stored in MATLAB files, and it is an external instrument rather than a shock.

### C4 · Advanced 04 · Collinearity, Coefficients and Stable Combinations

- **Data, licence, article.** As A9.
- **What the lab would show.** ln GDP and ln health spending correlate at 0.913 (*exploratory*), and their coefficient sum is the effect of scaling income and health spending together.
- **Why Tier C.** The synthetic design's near-collinearity (≈ 0.98) is what makes individual slopes imprecise while the sum stays precise. At 0.91 the contrast is much weaker, and the economic meaning of the sum is strained.
- **Rejected alternative.** Longley (1967), the classic collinearity dataset. The NIST copy has no licence statement.

### C5 · Advanced 09 · Joint Wald Restrictions with Robust Covariance

- **Data, licence, article.** As A9.
- **Why Tier C.** The lab's lesson is a pair of individually insignificant slopes that are jointly significant. In the real cross-section both slopes are individually significant, so the contrast the lab teaches does not appear. A real example would have to be searched for, and it would then be chosen for its result.

### C6 · Microeconomics 02 · Comparative Advantage and Trading Prices → crop yields

- **Data.** FAOSTAT *Crops and livestock products* (QCL), bulk file `https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip` (33.9 MB). Yields of two crops in two countries become land requirements per tonne.
- **Licence.** CC BY 4.0 "unless specified otherwise" (<https://www.fao.org/contact-us/terms/db-terms-of-use/en/>), but the same terms add that datasets "shall not be used for or in conjunction with the promotion of a commercial enterprise". Decide whether OpenEconometrics marketing could fall under that clause before using it.
- **Article.** Costinot & Donaldson, "Ricardo's Theory of Comparative Advantage: Old Idea, New Evidence", *AER* 102(3): 453–458 (2012), doi:[10.1257/aer.102.3.453](https://doi.org/10.1257/aer.102.3.453) (Crossref ✓). It is already cited in the lab.
- **Why Tier C.**
  - Ricardian unit *labour* requirements are not observed; land yields are a one-factor stand-in.
  - The lab's two-row table becomes a choice of two crops and two countries, which is arbitrary.

## Labs that should stay synthetic

| Lab | Why it should stay synthetic |
|---|---|
| Statistics 06 · Orders in ten offers (binomial) | The lab compares data with a binomial model of *known* success probability and independent trials. Weldon's dice (Pearson 1900) is the classic real case, but no transcription with an explicit licence was found (its public-domain status rests on age alone), and Pearson's own test rejected the fair-dice model for those throws, so the stated model fails. |
| Statistics 11 · Opt-in rate (one proportion) | The lesson contrasts Wald and Wilson intervals at a moderate n against a substantive null. Open real data are either large, where the two intervals coincide, or need an arbitrary subsample, and none supplied a natural null. |
| Microeconomics 01, 03–07, 09–19 | These workbooks are points on a stated model function: a frontier, linear demand and supply, Cobb–Douglas preferences and technology, quadratic costs, Cournot best responses. The independent references re-solve those functions analytically (for example `brentq` on 120 − 2p = −20 + 2p for Lab 03). Real observations lie on no exact function, so the labs would change from exact model analysis to estimation. Nothing in them is simulated noise to replace. Lab 10's journal case, Christensen & Greene (1976), estimates a cost function from utility data; using such data would make the lab an estimation exercise, and no open copy was searched for. |
| Microeconomics 02 | Unless C6 is accepted. Unit labour requirements are not observed. |
| Advanced 03 · Measurement error | The lab needs the true regressor beside its mismeasured version. Validation data with both, such as survey earnings matched to administrative records, are restricted. |
| Advanced 06 · Known-variance WLS | The lab weights by *known* error variances. Real variances are estimated, which makes the method feasible GLS, not the lab's known-variance WLS. |
| Advanced 16 · Weak instruments | The reference rebuilds the outcome as y − 1.2x + 1.2·weak_x, holding the structural equation and its error fixed while only instrument strength changes, and it uses the true coefficient 1.2. Real data cannot hold the structure fixed. A supplementary real illustration is possible with the Angrist & Krueger (1991) quarter-of-birth extract (Dataverse doi:[10.7910/DVN/ENLGZX](https://doi.org/10.7910/DVN/ENLGZX), CC0, 329,509 men): `lwklywge`, `educ`, `yob`, `qob`; a quarter-1 indicator gives first-stage F ≈ 65 (*exploratory*), and the weak-instrument critique is Bound, Jaeger & Baker (1995), doi:[10.1080/01621459.1995.10476536](https://doi.org/10.1080/01621459.1995.10476536). It would supplement, not replace, the controlled comparison. |
| Advanced 19 · Local projections | Unless C3 is accepted. The lab compares estimates with the population impulse response. |

## Datasets considered and rejected

| Dataset | Reason |
|---|---|
| Cushny & Peebles (1905) sleep data, Student's first t test | PMC (PMC1465734) shows no licence, and PMC's notice says to assume standard copyright when none is stated. Public domain by age is likely, but no explicit statement was found. |
| NIST Statistical Reference Datasets (Michelson, Longley) | No copyright or public-domain statement on any StRD page or file. NIST's licence page says Standard Reference Data may be copyrighted under 15 U.S.C. § 290e. Use the primary source where one exists (C1). |
| Current Population Survey ASEC (Census) | Licence wording not found: census.gov's open-data page commits to "sharing its public data as open data" but states no licence for the files. Usable if the project accepts the US-government-work rule (17 U.S.C. § 105) without a dataset statement; not relied on here. |
| FRED copies of any series | FRED's terms say some series are owned by third parties and require permission, and they prohibit scraping. Cite BEA, BLS and EIA directly. |
| OECD Data Explorer | OECD data have their own terms: free use "for any purpose, even for commercial use", with attribution and pass-on conditions. CC BY 4.0 is named only for written content. Not needed here. |
| Bertrand & Mullainathan (2004) résumés; Romer & Romer (2004) shocks; other AEA openICPSR packages | openICPSR pages could not be read: a Cloudflare challenge blocked them and was not bypassed. DataCite lists no rights. Unverified, so not used. |
| Miguel, Satyanath & Sergenti (2004) rainfall IV | Dataverse terms are research-only and forbid sharing. |
| Card (1995) college proximity | No licence on the author's page. |
| DellaVigna, List & Malmendier (2012) | Labelled CC0, but the download is behind a guestbook form. |
| Bauer & Swanson monetary surprises (SF Fed) | Site policy allows reproduction only if "not distributed for the purpose of private gain". |
| SOFR monetary surprises (Acosta et al. 2024) | CC BY-NC 4.0. |
| Cappelen, Kariv, Sørensen & Tungodden (2023) budgets | Dataverse says CC0, but the code README also says "Creative Commons Non-commercial license". Confirm with the author first. |
| Jarociński & Karadi (2020) shocks via ZBW (Piger & Stockwell 2025, doi:[10.1002/jae.70003](https://doi.org/10.1002/jae.70003)) | CC BY 4.0 from the depositor, but the package's `SP500` column is S&P data that FRED says may not be reproduced, and the source repository has no licence. Usable without that column; not needed for C3. |
| R and Python package copies (`sleep`, Galton, Michelson, iris, and others) | Not used as sources. Where a dataset is proposed, the original deposit or primary publication is cited. |

## Suggested order of work

1. **Pilot (one PR).**
   - Make the cross-cutting changes: allowlist rule and test, the `roles` map, generator-from-extract, dataset fields, packaging text and `THIRD_PARTY.md`.
   - Switch A1–A4 (Statistics 04, 05, 16, 18) and A5 (Advanced 15). Every licence is explicit, every file is under 1 MB, and the labs need few computational changes.
   - Run the focused course verifier for those labs only (`scripts/verify_course_series.py --course statistics --labs …` and `--course advanced-econometrics --labs 15-iv-moment-condition`) and the public-distribution tests. Per the current testing policy, do not dispatch the full daily gate for a teaching-content change.
2. **WDI batch.** A7–A10, B9, B12 and B13 share one pinned WDI extract.
3. **Okun batch.** A6 and B2 share one BEA/BLS extract.
4. **Online Retail batch.** A11, A12 and B17 share one filtered invoice table.
5. **Remaining Tier B labs**, each with its caveat written into the handout first.
6. **Tier C** only after an explicit decision per lab.

Each batch should regenerate figures and LaTeX tables, rewrite the handout and instructor companion, and leave the Statistics, Microeconomics and Advanced journal cases intact where they still fit. Lesson videos for these courses remain on hold; this proposal changes nothing about them.

## Verification record

Licence statements, as quoted above, were read on these pages on 18 October 2026:

- **UCI:** dataset pages 186, 352, 468 and 502, and the CC BY 4.0 legal code.
- **Harvard Dataverse API:** doi:10.7910/DVN/XRSUJU, ENLGZX, LUF59R, PIUJW4, ZJINIB, SIWH9F, T0HSJ1, 27853, SAVGAL, M2GAZN and 27102; Yale Dataverse API: doi:10.60600/YU/CGMWNW.
- **ZBW JAE Data Archive:** doi:10.15456/jae.2024364.1002804655.
- **GitHub:** `dkaenzig/replicationOilSupplyNews` (licence API, CC0-1.0).
- **Agency policies:**
  - cdc.gov/mmwr/about.html and cdc.gov/other/agencymaterials.html;
  - cdc.gov/nchs/policy/data-user-agreement.html;
  - bea.gov/help/faq/147;
  - bls.gov/opub/copyright-information.htm;
  - eia.gov/about/copyrights_reuse.php;
  - nhtsa.gov/about-nhtsa/terms-use and fhwa.dot.gov/disclaim.cfm;
  - the BTS *Transportation Statistics Annual Report 2016* front matter and the data.bts.gov dataset labels.
- **World Bank:** worldbank.org/ext/en/legal/terms-conditions/datasets and the WDI API `License_Type` metadata.
- **FAO:** fao.org/contact-us/terms/db-terms-of-use.
- **OECD:** oecd.org/en/about/terms-conditions.
- **FRED:** fred.stlouisfed.org/legal.
- **NIST:** nist.gov/open/license and the StRD pages.
- **Project Gutenberg:** eBook #11753.
- **PMC:** the copyright notice.

Every article DOI in this document resolved on the Crossref REST API on the same date. Dataset DOIs (prefixes 10.7910, 10.60600, 10.24432 and 10.15456) are DataCite DOIs; each resolves through doi.org to the landing page named beside it.
