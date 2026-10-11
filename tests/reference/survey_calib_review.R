# Reference numbers for tests/test_survey_calib_review.py (hard-coded there; R is not needed at test time).
#
# R 4.6.1, survey 4.5, sampling 2.11 (private test-only library):
#   R_LIBS=<scratchpad>/Rlib Rscript tests/reference/survey_calib_review.R
#
# Configurations not covered by survey_calib_calibrate.R / survey_calib_pps.R:
#   * a stratified clustered design with unequal PSU counts (3, 5, 2, 1, 3) and PSU sizes
#     (1..5 rows), a census stratum (fpc = 2 sampled PSUs) and a singleton certainty
#     stratum (fpc = 1), missing outcome values (na.rm / subset) and domains;
#   * raking with domain + missing-value exclusion for totals, means and ratios, deff;
#   * logit and bounded-linear (truncated) calibration at bounds 0.0001 above the
#     smallest feasible upper bound (linear programme: 1.087987 for lower bound 0.9);
#   * by-group calibration across PSUs (linear by sex) and logit by region;
#   * a clustered PPS design (rows summed to PSU totals) with a certainty stratum:
#     Brewer, exact Sampford pi_kl (ppsmat), Overton, HR(psum), HR(), with and
#     without calibration, domains and missing values; sampling::varest for the Hajek centre.
#     No two sampled PSUs of a stratum have pi_k + pi_l = 1, where R's HRDcheck sets D_kl = 0.
suppressMessages({library(survey); library(sampling)})
options(survey.lonely.psu = "fail")
emit <- function(name, x) cat(name, " = [", paste(sprintf("%.17g", as.vector(x)), collapse = ", "), "]\n", sep = "")
diagonal <- function(x) if (is.matrix(x)) diag(x) else x

# ---- calibrated SurveyDesign fixture -------------------------------------------------
i <- 0:38
sizes <- list(c(2, 4, 3), c(1, 3, 2, 5, 2), c(4, 3), c(3), c(2, 2, 3)); fpcs <- c(7, 15, 2, 1, 9)
h <- unlist(lapply(1:5, function(k) rep(k, sum(sizes[[k]]))))
psu <- unlist(lapply(1:5, function(k) rep(seq_along(sizes[[k]]), sizes[[k]])))
dat <- data.frame(h = h, psu = psu, Nh = fpcs[h], d = 5 + (i * 7) %% 11 + 0.5 * (i %% 3),
                  sex = ifelse((i * 5) %% 7 < 3, "F", "M"),
                  region = c("north", "south", "east")[(i * 2 + i %/% 4) %% 3 + 1],
                  x = 8 + (i * 13) %% 17 + 0.25 * (i %% 4), z = 4 + (i * 5) %% 7)
dat$y <- 2 + 1.1 * dat$x + (i * 11) %% 9 - 3 * (dat$sex == "F")
dat$y2 <- 40 - 0.5 * dat$x + ((i * 3) %% 5) * 2
dat$dom <- as.integer((i * 7) %% 5 < 3)
dat$y[c(3, 10, 22, 30) + 1] <- NA
dat$z[17 + 1] <- NA
des <- svydesign(ids = ~psu, strata = ~h, fpc = ~Nh, weights = ~d, data = dat, nest = TRUE)
pop <- c(`(Intercept)` = 425, sexM = 228, regionnorth = 204.4, regionsouth = 107, x = 6900)

rk <- calibrate(des, ~sex + region + x, population = pop, calfun = "raking", epsilon = 1e-14, maxit = 200)
emit("raking_g", weights(rk) / dat$d)
t2 <- svytotal(~y + y2, rk, na.rm = TRUE); emit("raking_drop_total", c(coef(t2), vcov(t2)))
m2 <- svymean(~y + y2, rk, na.rm = TRUE); emit("raking_drop_mean", c(coef(m2), vcov(m2)))
sub <- subset(rk, dom == 1 & !is.na(y))
dt <- svytotal(~y, sub, deff = TRUE, na.rm = TRUE); emit("raking_domain_total", c(coef(dt), vcov(dt), diagonal(deff(dt))))
dm <- svymean(~y, sub, deff = TRUE, na.rm = TRUE); emit("raking_domain_mean", c(coef(dm), vcov(dm), diagonal(deff(dm))))
rr <- svyratio(~y, ~z, subset(rk, dom == 1 & !is.na(y) & !is.na(z)), na.rm = TRUE)
emit("raking_domain_ratio", c(coef(rr), rr$var))
fr <- svyratio(~y2, ~x, rk, deff = TRUE); emit("raking_ratio", c(coef(fr), fr$var, attr(fr, "deff")))
emit("raking_df", degf(rk))

for (cf in c("logit", "linear")) {
  cal <- calibrate(des, ~sex + region + x, population = pop, calfun = cf, bounds = c(0.9, 1.0881),
                   epsilon = 1e-14, maxit = 500)
  name <- if (cf == "logit") "edge_logit" else "edge_truncated"
  emit(paste0(name, "_g"), weights(cal) / dat$d)
  tt <- svytotal(~y2 + x, cal); emit(paste0(name, "_total"), c(coef(tt), vcov(tt)))
  mm <- svymean(~y2, cal); emit(paste0(name, "_mean"), c(coef(mm), vcov(mm)))
}

bysex <- calibrate(des, ~0 + sex + sex:x, population = c(sexF = 197, sexM = 228, `sexF:x` = 3150, `sexM:x` = 3750),
                   epsilon = 1e-14)
emit("bysex_g", weights(bysex) / dat$d)
tt <- svytotal(~y2, bysex); emit("bysex_total", c(coef(tt), vcov(tt)))
rr <- svyratio(~y2, ~x, bysex); emit("bysex_ratio", c(coef(rr), rr$var))

byregion <- calibrate(des, ~0 + region + region:sex, calfun = "logit", bounds = c(0.5, 2), epsilon = 1e-14,
                      maxit = 200, population = c(regioneast = 113.6, regionnorth = 204.4, regionsouth = 107,
                                                  `regioneast:sexM` = 77.6, `regionnorth:sexM` = 104.4,
                                                  `regionsouth:sexM` = 46))
emit("byregion_g", weights(byregion) / dat$d)
tt <- svytotal(~y2, byregion); emit("byregion_total", c(coef(tt), vcov(tt)))
rr <- svyratio(~y2, ~x, byregion); emit("byregion_ratio", c(coef(rr), rr$var))

# ---- clustered PPS fixture ----------------------------------------------------------------
sA <- c(4, 7, 2, 9, 5, 3, 8, 6, 10); sB <- c(3, 6, 2, 8, 5, 4, 7, 9)
piA <- inclusionprobabilities(sA, 3); piB <- inclusionprobabilities(sB, 4)
PA <- UPsampfordpi2(piA)[c(2, 4, 9), c(2, 4, 9)]; PB <- UPsampfordpi2(piB)[c(1, 2, 3, 6), c(1, 2, 3, 6)]
emit("PA", PA); emit("PB", PB); emit("psum", c(sum(piA^2), sum(piB^2)))
unit <- c("a2", "a2", "a4", "a9", "a9", "a9", "b1", "b2", "b2", "b3", "b6", "b6", "c1", "c1")
upi <- c(a2 = piA[2], a4 = piA[4], a9 = piA[9], b1 = piB[1], b2 = piB[2], b3 = piB[3], b6 = piB[6], c1 = 1)
usize <- c(a2 = 7, a4 = 9, a9 = 10, b1 = 3, b2 = 6, b3 = 2, b6 = 4, c1 = 30)
j <- 0:13
s <- data.frame(strat = match(substr(unit, 1, 1), c("a", "b", "c")), unit = unit, pi = unname(upi[unit]))
s$x <- 3 + (j * 5) %% 7 + unname(usize[unit]) / 2
s$y <- 10 + 2 * s$x + (j * 3) %% 5
s$y2 <- 20 - s$x + (j %% 4)
s$dom <- as.integer((j * 3) %% 4 < 3)
s$y[5 + 1] <- NA
# Row-level joint probabilities: rows of one PSU share it (pi_kl = pi_k), PSUs within a
# stratum take the Sampford pi_kl, strata are independent.
psus <- c("a2", "a4", "a9", "b1", "b2", "b3", "b6", "c1")
U <- outer(upi[psus], upi[psus]); dimnames(U) <- list(psus, psus)
U[1:3, 1:3] <- PA; U[4:7, 4:7] <- PB; U["c1", "c1"] <- 1
P <- U[unit, unit]
ppop <- c(`(Intercept)` = 34.5, x = 325)

report <- function(name, des) {
  tt <- svytotal(~y2 + x, des); emit(paste0(name, "_total"), c(coef(tt), vcov(tt)))
  mm <- svymean(~y2, des); emit(paste0(name, "_mean"), c(coef(mm), vcov(mm)))
  rr <- svyratio(~y2, ~x, des); emit(paste0(name, "_ratio"), c(coef(rr), rr$var))
}
brewer <- svydesign(ids = ~unit, strata = ~strat, fpc = ~pi, data = s, pps = "brewer")
report("pps_brewer", brewer)
rk <- calibrate(brewer, ~x, population = ppop, calfun = "raking", epsilon = 1e-14, maxit = 200)
emit("pps_brewer_raking_g", weights(rk) * s$pi)
report("pps_brewer_raking", rk)
tt <- svytotal(~y + y2, rk, na.rm = TRUE); emit("pps_brewer_raking_drop_total", c(coef(tt), vcov(tt)))
sub <- subset(rk, dom == 1 & !is.na(y))
dt <- svytotal(~y, sub, na.rm = TRUE); emit("pps_brewer_raking_domain_total", c(coef(dt), vcov(dt)))
dm <- svymean(~y, sub, na.rm = TRUE); emit("pps_brewer_raking_domain_mean", c(coef(dm), vcov(dm)))
# ppsmat takes element-level pi_kl and R's ygvar.matrix differences elements, not PSU totals,
# so the clustered SYG reference uses the PSU-aggregated sample (one row per PSU; the mean is
# the ratio to the row count m). HT is bilinear: the row-level matrix (pi_kl = pi_k within a
# PSU) is exact for it, also after calibration.
exact <- svydesign(ids = ~unit, strata = ~strat, fpc = ~pi, data = s, pps = ppsmat(P, tolerance = 0), variance = "HT")
report("pps_exact_HT", exact)
dt <- svytotal(~y, subset(exact, dom == 1 & !is.na(y)), na.rm = TRUE); emit("pps_exact_HT_domain_total", c(coef(dt), vcov(dt)))
report("pps_exact_HT_linear", calibrate(exact, ~x, population = ppop, epsilon = 1e-14))
agg <- data.frame(strat = c(1, 1, 1, 2, 2, 2, 2, 3), pi = unname(upi[psus]),
                  y2 = rowsum(s$y2, s$unit, reorder = FALSE)[, 1], x = rowsum(s$x, s$unit, reorder = FALSE)[, 1],
                  m = rowsum(rep(1, 14), s$unit, reorder = FALSE)[, 1],
                  yd = rowsum(ifelse(s$dom == 1 & !is.na(s$y), s$y, 0), s$unit, reorder = FALSE)[, 1])
yg <- svydesign(ids = ~1, strata = ~strat, fpc = ~pi, data = agg, pps = ppsmat(unname(U), tolerance = 0), variance = "YG")
tt <- svytotal(~y2 + x, yg); emit("pps_exact_YG_total", c(coef(tt), vcov(tt)))
rr <- svyratio(~y2, ~m, yg); emit("pps_exact_YG_mean", c(coef(rr), rr$var))
rr <- svyratio(~y2, ~x, yg); emit("pps_exact_YG_ratio", c(coef(rr), rr$var))
dt <- svytotal(~yd, yg); emit("pps_exact_YG_domain_total", c(coef(dt), vcov(dt)))
for (v in c("HT", "YG")) {
  report(paste0("pps_overton_", v), svydesign(ids = ~unit, strata = ~strat, fpc = ~pi, data = s, pps = "overton",
                                               variance = v))
  pbar <- c(sum(piA^2) / 3, sum(piB^2) / 4, 1)
  report(paste0("pps_hr_", v), svydesign(ids = ~unit, strata = ~strat, fpc = ~pi, data = s, variance = v,
                                          pps = HR(psum = data.frame(p = unname(pbar[s$strat])),
                                                   strata = data.frame(s = s$strat))))
  hr1 <- svydesign(ids = ~unit, strata = ~strat, fpc = ~pi, data = s, variance = v, pps = HR())
  report(paste0("pps_hrsample_", v), hr1)
  report(paste0("pps_hrsample_", v, "_raking"), calibrate(hr1, ~x, population = ppop, calfun = "raking",
                                                           epsilon = 1e-14, maxit = 200))
}
# R's `[.pps` zeroes the Delta-check block of out-of-domain rows; after calibration their
# g-weighted residuals are not zero, so R drops part of the calibrated domain variance.
dt <- svytotal(~y, subset(calibrate(exact, ~x, population = ppop, epsilon = 1e-14), dom == 1 & !is.na(y)),
               na.rm = TRUE)
emit("pps_exact_HT_linear_domain_total_R", c(coef(dt), vcov(dt)))

# Hajek (1964) variance through sampling::varest (Deville's estimator with the same centre
# A = sum (1 - pi) z / sum (1 - pi)): Hajek = varest * (1 - sum a^2) * n/(n - 1).
hajek <- 0
for (st in 1:2) {
  rows <- s$strat == st
  z <- rowsum(s$y2[rows] / s$pi[rows], s$unit[rows], reorder = FALSE)[, 1]
  p <- upi[names(z)]; a <- (1 - p) / sum(1 - p); n <- length(z)
  hajek <- hajek + varest(z * p, pik = p) * (1 - sum(a^2)) * n / (n - 1)
}
emit("pps_hajek_y2_total_var", hajek)
