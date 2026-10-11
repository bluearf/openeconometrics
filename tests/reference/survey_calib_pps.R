# Reference numbers for tests/test_survey_calib_pps.py (hard-coded there; R is not needed at test time).
#
# R 4.6.1, survey 4.5, sampling 2.11 (private test-only library):
#   R_LIBS=<scratchpad>/Rlib Rscript tests/reference/survey_calib_pps.R
#
# Two strata of a finite population with size measures; inclusion probabilities
# pi = n * size / sum(size) (sampling::inclusionprobabilities) and exact Sampford
# joint probabilities (sampling::UPsampfordpi2). The sample is fixed (rows below);
# the estimators do not depend on how it was drawn.
suppressMessages({library(survey); library(sampling)})
options(survey.lonely.psu = "fail")
emit <- function(name, x) cat(name, " = [", paste(sprintf("%.17g", as.vector(x)), collapse = ", "), "]\n", sep = "")
diagonal <- function(x) if (is.matrix(x)) diag(x) else x

sizeA <- c(3, 5, 8, 2, 9, 4, 6, 7, 10, 3, 5, 8); sA <- c(2, 5, 7, 9)
sizeB <- c(4, 9, 2, 6, 11, 5, 3, 8, 7, 5); sB <- c(1, 5, 8)
piA <- inclusionprobabilities(sizeA, 4); piB <- inclusionprobabilities(sizeB, 3)
PA <- UPsampfordpi2(piA); PB <- UPsampfordpi2(piB)
emit("piA", piA); emit("piB", piB)
emit("PA_sample", PA[sA, sA]); emit("PB_sample", PB[sB, sB])
emit("psum", c(sum(piA^2), sum(piB^2)))

s <- data.frame(strat = c(1, 1, 1, 1, 2, 2, 2), pi = c(piA[sA], piB[sB]), x = c(sizeA[sA], sizeB[sB]),
                y = c(12.5, 20, 15, 27, 9, 30, 18), y2 = c(3, 1, 4, 1, 5, 9, 2), dom = c(1, 0, 1, 1, 0, 1, 1))
P <- outer(s$pi, s$pi); P[1:4, 1:4] <- PA[sA, sA]; P[5:7, 5:7] <- PB[sB, sB]

report <- function(name, des, deff = FALSE) {
  tot <- svytotal(~y + y2, des); mn <- svymean(~y + y2, des, deff = deff); rt <- svyratio(~y, ~x, des)
  emit(paste0(name, "_total"), coef(tot)); emit(paste0(name, "_total_vcov"), vcov(tot))
  emit(paste0(name, "_mean"), coef(mn)); emit(paste0(name, "_mean_vcov"), vcov(mn))
  if (deff) emit(paste0(name, "_mean_deff"), diagonal(deff(mn)))
  emit(paste0(name, "_ratio"), c(coef(rt), rt$var))
}
for (v in c("HT", "YG")) {
  report(paste0("exact_", v), svydesign(ids = ~1, strata = ~strat, fpc = ~pi, data = s,
                                        pps = ppsmat(P, tolerance = 0), variance = v), deff = v == "HT")
  report(paste0("overton_", v), svydesign(ids = ~1, strata = ~strat, fpc = ~pi, data = s, pps = "overton", variance = v))
  report(paste0("hr_", v), svydesign(ids = ~1, strata = ~strat, fpc = ~pi, data = s, variance = v,
                                    pps = HR(psum = data.frame(p = c(sum(piA^2) / 4, sum(piB^2) / 3)[s$strat]),
                                             strata = data.frame(s = s$strat))))
  report(paste0("hrsample_", v), svydesign(ids = ~1, strata = ~strat, fpc = ~pi, data = s, pps = HR(), variance = v))
}
brewer <- svydesign(ids = ~1, strata = ~strat, fpc = ~pi, data = s, pps = "brewer")
report("brewer", brewer)
emit("brewer_total_deff", diagonal(deff(svytotal(~y + y2, brewer, deff = TRUE))))
report("wr", svydesign(ids = ~1, strata = ~strat, probs = ~pi, data = s))

# Exact HT / SYG through sampling::varHT, stratum by stratum.
emit("varHT", c(varHT(s$y[1:4], PA[sA, sA], 1) + varHT(s$y[5:7], PB[sB, sB], 1),
                varHT(s$y[1:4], PA[sA, sA], 2) + varHT(s$y[5:7], PB[sB, sB], 2)))

# Domain (subset) and calibrated PPS designs.
yg <- svydesign(ids = ~1, strata = ~strat, fpc = ~pi, data = s, pps = ppsmat(P, tolerance = 0), variance = "YG")
dt <- svytotal(~y, subset(yg, dom == 1)); emit("exact_YG_domain_total", c(coef(dt), vcov(dt)))
pop <- c(`(Intercept)` = 22, x = sum(sizeA) + sum(sizeB))
for (name in c("brewer", "yg")) {
  des <- if (name == "brewer") brewer else yg
  for (fun in c("linear", "raking")) {
    cal <- calibrate(des, ~x, population = pop, calfun = fun, epsilon = 1e-14, maxit = 200)
    tot <- svytotal(~y + y2, cal)
    emit(paste0("cal_", name, "_", fun, "_g"), weights(cal) * s$pi)
    emit(paste0("cal_", name, "_", fun, "_total"), coef(tot)); emit(paste0("cal_", name, "_", fun, "_vcov"), vcov(tot))
  }
}
