# Reference numbers for tests/test_survey_calib_calibration.py and
# tests/test_survey_calib_estimation.py (hard-coded there; R is not needed at test time).
#
# R 4.6.1, survey 4.5, sampling 2.11 (private test-only library):
#   R_LIBS=<scratchpad>/Rlib Rscript tests/reference/survey_calib_calibrate.R
#
# The fixture is built from the same integer formulas as `fixture()` in the tests:
# 3 strata x 4 PSUs x 3 rows, FPC = population PSU counts (10, 12, 8).
suppressMessages({library(survey); library(sampling)})
options(survey.lonely.psu = "fail")

emit <- function(name, x) cat(name, " = [", paste(sprintf("%.17g", as.vector(x)), collapse = ", "), "]\n", sep = "")
diagonal <- function(x) if (is.matrix(x)) diag(x) else x

i <- 0:35
dat <- data.frame(
  h = i %/% 12 + 1,
  psu = (i %/% 3) %% 4 + 1,
  d = 8 + (i * 7) %% 9 + 0.25 * (i %% 4),
  sex = ifelse((i * 5) %% 7 < 3, "F", "M"),
  age = c("a", "b", "c")[(i + i %/% 3) %% 3 + 1],
  x = 10 + (i * 13) %% 17 + 0.5 * (i %% 3),
  z = 5 + (i * 5) %% 6,
  dom = as.integer((i * 7) %% 4 < 2)
)
dat$Nh <- c(10, 12, 8)[dat$h]
dat$y <- 3 + 0.9 * dat$x + (i * 11) %% 7 - 2 * (dat$sex == "F")
dat$y2 <- 50 - 0.4 * dat$x + ((i * 3) %% 5) * 1.5

des <- svydesign(ids = ~psu, strata = ~h, fpc = ~Nh, weights = ~d, data = dat, nest = TRUE)
N <- 460
pop <- c(`(Intercept)` = N, sexM = 0.53 * N, ageb = 0.32 * N, agec = 0.35 * N, x = 8600)
fit <- function(calfun, bounds = c(-Inf, Inf))
  calibrate(des, ~sex + age + x, population = pop, calfun = calfun, bounds = bounds,
            epsilon = 1e-14, maxit = 200)
designs <- list(linear = fit("linear"), raking = fit("raking"),
                logit = fit("logit", c(0.6, 1.5)), truncated = fit("linear", c(0.8, 1.25)))
for (name in names(designs)) {
  cal <- designs[[name]]
  emit(paste0(name, "_g"), weights(cal) / dat$d)
  tot <- svytotal(~y + y2, cal, deff = TRUE)
  emit(paste0(name, "_total"), coef(tot)); emit(paste0(name, "_total_vcov"), vcov(tot))
  emit(paste0(name, "_total_deff"), diagonal(deff(tot)))
  mn <- svymean(~y + y2, cal, deff = TRUE)
  emit(paste0(name, "_mean"), coef(mn)); emit(paste0(name, "_mean_vcov"), vcov(mn))
  emit(paste0(name, "_mean_deff"), diagonal(deff(mn)))
  rt <- svyratio(~y + y2, ~z, cal, covmat = TRUE, deff = TRUE)
  emit(paste0(name, "_ratio"), coef(rt)); emit(paste0(name, "_ratio_vcov"), vcov(rt))
  emit(paste0(name, "_ratio_deff"), attr(rt, "deff"))
  sub <- subset(cal, dom == 1)
  dt <- svytotal(~y, sub); dm <- svymean(~y, sub)
  emit(paste0(name, "_domain_total"), c(coef(dt), vcov(dt)))
  emit(paste0(name, "_domain_mean"), c(coef(dm), vcov(dm)))
}

# Stratum-wise (by = h) raking on sex and x within each stratum.
dat$hf <- factor(dat$h)
des <- svydesign(ids = ~psu, strata = ~h, fpc = ~Nh, weights = ~d, data = dat, nest = TRUE)
grp <- c(`hf1:sexF` = 66, `hf2:sexF` = 63, `hf3:sexF` = 68, `hf1:sexM` = 90, `hf2:sexM` = 85,
         `hf3:sexM` = 80, `hf1:x` = 2900, `hf2:x` = 2700, `hf3:x` = 2750)
by <- calibrate(des, ~0 + hf:sex + hf:x, population = grp, calfun = "raking", epsilon = 1e-14, maxit = 200)
emit("by_raking_g", weights(by) / dat$d)
tot <- svytotal(~y + y2, by)
emit("by_raking_total", coef(tot)); emit("by_raking_total_vcov", vcov(tot))

# sampling::calib g-weights (unstratified matrix form) for the same constraints.
X <- model.matrix(~sex + age + x, dat)
emit("sampling_linear_g", calib(X, dat$d, pop, method = "linear"))
emit("sampling_raking_g", calib(X, dat$d, pop, method = "raking", max_iter = 500))
