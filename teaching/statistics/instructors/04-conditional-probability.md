# Instructor companion · Lab 04

## Teaching sequence

Draw the two-by-two table on the board and have students circle a different denominator for each conditional question. Stress early that the unit is a browsing session, so every share is session-weighted, and that the published file has no visitor identifier.

Before running, use the pause question in Section 3: students name each denominator as a count of sessions. Then show that the headline conditionals (0.139 and 0.770) are far apart only because returning sessions are 5.5 times as common as purchase sessions; exercise 2 supplies a reversed pair that is close because its two margins are nearly equal. The data are real and the association runs the "wrong" way for the manager's story, which makes a good discussion of why a large share among buyers says nothing about a group's buying rate. Leave the chi-square output to Lab 16.

## Worked exercise answers

**1.** `New_Visitor` sessions: $422/1{,}694=0.2491$; `Other` sessions: $16/85=0.1882$; returning sessions: $1{,}470/10{,}551=0.1393$. New minus returning is 0.1098 (about 11 percentage points) with ratio $0.2491/0.1393=1.788$; Other minus returning is 0.0489 with ratio 1.351. Counting the 85 `Other` sessions as returning gives $P(P\mid R)=1{,}486/10{,}636=0.1397$ and $P(R\mid P)=1{,}486/1{,}908=0.7788$. The conclusion does not change: the `Other` group is too small (0.7% of sessions) to move either share by more than a point.

**2.** $P(R\mid\text{no purchase})=9{,}081/10{,}422=0.8713$ and $P(\text{no purchase}\mid R)=9{,}081/10{,}551=0.8607$. By the Section 4 identity their ratio is $P(R)/P(\text{no purchase})=10{,}551/10{,}422=1.012$. Returning sessions (10,551) and no-purchase sessions (10,422) are almost equally common, so the shared numerator 9,081 is divided by nearly the same denominator. For the purchase pair the denominators are 10,551 and 1,908.

**3.** Expected counts are row total × column total / 12,330: returning and purchase $10{,}551\times1{,}908/12{,}330=1{,}632.71$; returning and no purchase $10{,}551\times10{,}422/12{,}330=8{,}918.29$; other and purchase $1{,}779\times1{,}908/12{,}330=275.29$; other and no purchase $1{,}779\times10{,}422/12{,}330=1{,}503.71$. The rows sum to 10,551 and 1,779 and the columns to 1,908 and 10,422, as in the observed table. Observed minus expected is $1{,}470-1{,}632.71=-162.71$ sessions in the returning-and-purchase cell, which equals $12{,}330\times(-0.013196)$. In a two-by-two table all four gaps have this size, with signs $-,+,+,-$ ($9{,}081-8{,}918.29$, $438-275.29$ and $1{,}341-1{,}503.71$).

**4.** $P(P\mid\text{weekend})=499/2{,}868=0.1740$ is the purchase share among weekend sessions; $P(\text{weekend}\mid P)=499/1{,}908=0.2615$ is the weekend share among purchase sessions; $P(\text{weekend})=2{,}868/12{,}330=0.2326$ is the weekend share of all sessions. Weekday sessions buy at $1{,}409/9{,}462=0.1489$, so weekend sessions buy a little more often, and weekends supply slightly more of the purchases (26.2%) than of the sessions (23.3%).

**5.** Group the sessions by visitor, compute each returning visitor's own purchase share (or whether they ever bought), and average over visitors so that each counts once. If a few very frequent visitors browse many times without buying, they pull the session-weighted 0.139 down, and a visitor-weighted share could be noticeably higher; the reverse is also possible. This file cannot answer the question because it has no visitor identifier, and the 125 exact duplicate rows cannot be told apart from genuine repeat sessions.

**6.** For example: "Of 1,908 purchase sessions, 1,470 (77.0%) were by returning visitors. Of 10,551 returning visitors' sessions, 1,470 (13.9%) ended in a purchase, compared with 438 of 1,779 other sessions (24.6%)." The conclusion does not follow because $P(R\mid P)=P(P\mid R)P(R)/P(P)$; the 77% is high because returning visitors produce 85.6% of all sessions.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/04-conditional-probability/shopper_sessions.xlsx) for the published analysis. Its rows are published data, not a simulation: the [generator](generators/04-conditional-probability.py) rebuilds them from the publisher's file vendored in [sources/uci-online-shoppers](sources/uci-online-shoppers/index.md) after checking its SHA-256, and nothing is downloaded. The source folder records the citation, the CC BY 4.0 licence and the retrieval. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/04-conditional-probability/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. The [LaTeX table](../labs/04-conditional-probability/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: four cells partition sessions, conditional times marginal equals joint, Bayes reverses the denominator. The course verifier compares fresh execution with this saved state, executes the exact student source through a disposable application worker, imports the workbook through the normal reader and reopens saved history. It also checks that `ROLES` in `lab.py` equals the `roles` in `dataset.json`, that the vendored file matches its recorded SHA-256 and that the workbook's Dictionary carries the citation and licence. Development-only independent references recompute both conditional frequencies from selected rows. These checks concern this teaching example and its published data, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Reading a share among buyers as a buying rate; ignoring the size of the denominator group; treating session shares as visitor shares; reading an association as an effect of being a returning visitor.
