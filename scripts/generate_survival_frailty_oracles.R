#!/usr/bin/env Rscript
# External oracle only: no OpenEconometrics code is loaded by this process.
# Reads tests/fixtures/survival_frailty/data_{cox,recurrent}.json (written by
# scripts/generate_survival_frailty_oracles.py data) and writes r_reference.json.
#
#   Rscript --vanilla scripts/generate_survival_frailty_oracles.R
#
# Run from the repository root. OPENECON_R_LIB optionally prepends a library path
# holding coxme and jsonlite (committed fixtures: survival 3.8.6, coxme 2.2.22,
# R 4.6.1).
#
# References: R survival (coxph with frailty(distribution = "gamma"), cluster(),
# strata(), Surv(start, stop, event)) and coxme (log-normal frailty, Laplace
# integrated partial likelihood). The frailty variance is located by optimize()
# on R's own integrated likelihood at fixed theta (tight tolerance); R's default
# EM/BFGS estimates are recorded as well.
lib <- Sys.getenv("OPENECON_R_LIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({
  library(survival); library(coxme); library(jsonlite)
})
options(digits = 17)
fx <- file.path("tests", "fixtures", "survival_frailty")
cox <- fromJSON(file.path(fx, "data_cox.json"))$datasets
rec <- fromJSON(file.path(fx, "data_recurrent.json"))$datasets
as_frame <- function(x) as.data.frame(lapply(x, unlist))
ctrl <- coxph.control(eps = 1e-11, iter.max = 200, outer.max = 60, toler.inf = 1e-7)
mctrl <- coxme.control(eps = 1e-11, iter.max = 200, sparse = c(100000, 0.02))
out <- list(versions = list(R = R.version.string,
                            survival = as.character(packageVersion("survival")),
                            coxme = as.character(packageVersion("coxme"))),
            generated = format(Sys.Date()))

fixed_names <- function(fit, k) names(fit$coefficients)[seq_len(k)]

gamma_fit <- function(df, rhs, theta, ties) {
  form <- as.formula(paste(rhs, "+ frailty(g, distribution = 'gamma', theta = ", theta,
                           ", sparse = FALSE)"))
  coxph(form, data = df, ties = ties, control = ctrl)
}

gamma_record <- function(fit, k) {
  q <- length(fit$coefficients) - k
  list(theta = fit$history[[1]]$theta,
       coef = unname(fit$coefficients[seq_len(k)]),
       var = unname(fit$var[seq_len(k), seq_len(k)]),
       var2 = unname(fit$var2[seq_len(k), seq_len(k)]),
       frail = unname(fit$coefficients[k + seq_len(q)]),
       partial = fit$loglik[2], ilik = fit$history[[1]]$c.loglik,
       penalty = fit$penalty[2])
}

coxme_record <- function(fit) {
  list(theta = unname(unlist(VarCorr(fit))), coef = unname(fixef(fit)),
       var = unname(as.matrix(vcov(fit))), frail = unname(ranef(fit)[[1]]),
       integrated = unname(fit$loglik["Integrated"]),
       penalized = unname(fit$loglik["Penalized"]))
}

frailty_case <- function(df, lhs, covariates, extra_rhs = "") {
  rhs <- paste(lhs, "~", paste(covariates, collapse = " + "), extra_rhs)
  k <- length(covariates)
  res <- list()
  for (ties in c("breslow", "efron")) {
    cfit <- coxph(as.formula(rhs), data = df, ties = ties, control = ctrl)
    ilik <- function(lt) gamma_fit(df, rhs, exp(lt), ties)$history[[1]]$c.loglik
    opt <- optimize(ilik, c(log(1e-3), log(10)), maximum = TRUE, tol = 1e-10)
    best <- gamma_fit(df, rhs, exp(opt$maximum), ties)
    fixed <- gamma_fit(df, rhs, 0.7, ties)
    em <- coxph(as.formula(paste(rhs, "+ frailty(g, distribution = 'gamma', sparse = FALSE,",
                                 "eps = 1e-10)")), data = df, ties = ties, control = ctrl)
    mform <- as.formula(paste(rhs, "+ (1 | g)"))
    mfit <- function(theta) coxme(mform, data = df, ties = ties, vfixed = list(g = theta),
                                  control = mctrl)
    mlik <- function(lt) unname(mfit(exp(lt))$loglik["Integrated"])
    mopt <- optimize(mlik, c(log(1e-3), log(10)), maximum = TRUE, tol = 1e-10)
    mdefault <- coxme(mform, data = df, ties = ties, control = mctrl)
    res[[ties]] <- list(
      cox = list(coef = unname(cfit$coefficients), loglik = cfit$loglik[2],
                 var = unname(cfit$var)),
      gamma_profile_max = gamma_record(best, k), gamma_fixed_0.7 = gamma_record(fixed, k),
      gamma_em = gamma_record(em, k),
      lognormal_profile_max = coxme_record(mfit(exp(mopt$maximum))),
      lognormal_fixed_0.5 = coxme_record(mfit(0.5)),
      lognormal_default = coxme_record(mdefault))
  }
  res$terms <- covariates
  res
}

out$cox_gamma <- frailty_case(as_frame(cox$cox_gamma), "Surv(t, d)", c("x1", "x2", "w"))
out$cox_entry <- frailty_case(as_frame(cox$cox_entry), "Surv(t0, t, d)", c("x1", "x2"),
                              "+ strata(s)")

# Boundary case: R's integrated likelihood relative to the Cox partial likelihood.
null <- as_frame(cox$cox_null)
cnull <- coxph(Surv(t, d) ~ x1, data = null, ties = "breslow", control = ctrl)
grid <- c(1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 1)
out$cox_null <- list(
  cox = list(coef = unname(cnull$coefficients), loglik = cnull$loglik[2],
             var = unname(cnull$var)),
  grid = grid,
  gamma_gain = sapply(grid, function(th) gamma_fit(null, "Surv(t, d) ~ x1", th,
                                                    "breslow")$history[[1]]$c.loglik)
               - cnull$loglik[2],
  lognormal_gain = sapply(grid, function(th) unname(coxme(Surv(t, d) ~ x1 + (1 | g),
                                                          data = null, ties = "breslow",
                                                          vfixed = list(g = th),
                                                          control = mctrl)$loglik["Integrated"]))
                   - cnull$loglik[2])

# ---- recurrent events -----------------------------------------------------------------
r <- as_frame(rec$recurrent)
r <- r[order(r$id, r$start), ]
r$prior <- ave(r$event, r$id, FUN = function(e) cumsum(e) - e)
r$prev <- 0
for (i in unique(r$id)) {
  rows <- which(r$id == i)
  last <- 0
  for (j in rows) {
    r$prev[j] <- last
    if (r$event[j] == 1) last <- r$stop[j]
  }
}
r$gstart <- r$start - r$prev
r$gstop <- r$stop - r$prev

wlw_data <- function(r, K) {
  parts <- lapply(seq_len(K), function(k) {
    keep <- r[r$prior < k, ]
    keep$status <- as.integer(keep$event == 1 & keep$prior + 1 == k)
    keep$stratum <- k
    keep
  })
  do.call(rbind, parts)
}

rec_record <- function(fit) {
  list(coef = unname(fit$coefficients), robust = unname(fit$var),
       naive = unname(fit$naive.var), loglik = fit$loglik[2],
       n = fit$n, nevent = fit$nevent)
}

recurrent <- list(terms = c("treat", "age", "z"),
                  n_subjects = length(unique(r$id)), max_events = max(r$prior + r$event))
for (ties in c("breslow", "efron")) {
  fits <- list()
  fits$ag <- coxph(Surv(start, stop, event) ~ treat + age + z + cluster(id), data = r,
                   ties = ties, control = ctrl)
  fits$pwp_total <- coxph(Surv(start, stop, event) ~ treat + age + z + strata(prior) +
                            cluster(id), data = r, ties = ties, control = ctrl)
  fits$pwp_gap <- coxph(Surv(gstart, gstop, event) ~ treat + age + z + strata(prior) +
                          cluster(id), data = r, ties = ties, control = ctrl)
  w <- wlw_data(r, max(r$prior + r$event))
  fits$wlw <- coxph(Surv(start, stop, status) ~ treat + age + z + strata(stratum) +
                      cluster(id), data = w, ties = ties, control = ctrl)
  capped <- r
  capped$cap <- pmin(capped$prior, 2)
  fits$pwp_total_cap3 <- coxph(Surv(start, stop, event) ~ treat + age + z + strata(cap) +
                                 cluster(id), data = capped, ties = ties, control = ctrl)
  w3 <- wlw_data(r, 3)
  fits$wlw_cap3 <- coxph(Surv(start, stop, status) ~ treat + age + z + strata(stratum) +
                           cluster(id), data = w3, ties = ties, control = ctrl)
  recurrent[[ties]] <- lapply(fits, rec_record)
}
recurrent$ag_gamma_frailty <- gamma_record(
  coxph(Surv(start, stop, event) ~ treat + age + z +
          frailty(id, distribution = "gamma", theta = 0.5, sparse = FALSE),
        data = r, ties = "breslow", control = ctrl), 3)
out$recurrent <- recurrent

writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = FALSE),
           file.path(fx, "r_reference.json"))
cat("wrote", file.path(fx, "r_reference.json"), "\n")
