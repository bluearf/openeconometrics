#!/usr/bin/env Rscript
# External oracle only: metafor (rma.mv, confint, anova, predict, trimfill,
# selmodel), clubSandwich (vcovCR CR2, coef_test, Wald_test HTZ), weightr
# (weightfunct) and base R lm. No OpenEconometrics code is loaded here.
# Usage: Rscript generate_meta_extension_oracles.R <input-dir> <output-dir>
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: Rscript generate_meta_extension_oracles.R input-dir output-dir")
suppressPackageStartupMessages({
  library(metafor); library(clubSandwich); library(weightr); library(jsonlite); library(metadat)
})
input <- args[1L]; output <- args[2L]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
tight <- list(optimizer = "optim", optmethod = "BFGS", reltol = 1e-15, maxit = 20000)
uni <- list(threshold = 1e-13, maxiter = 10000)
mat <- function(x) unname(as.matrix(x))
save <- function(x, name) write_json(x, file.path(output, paste0(name, ".json")), digits = NA,
                                     auto_unbox = TRUE, null = "null", na = "null", matrix = "rowmajor")

# ---- published datasets -----------------------------------------------------
write.csv(dat.konstantopoulos2011, file.path(output, "konstantopoulos2011.csv"), row.names = FALSE)
write.csv(dat.hackshaw1998[, c("study", "author", "year", "yi", "vi")], file.path(output, "hackshaw1998.csv"), row.names = FALSE)
konst <- dat.konstantopoulos2011
konst$yc <- konst$year - 1985
konst$es <- seq_len(nrow(konst))
hack <- dat.hackshaw1998

i2 <- function(fit, V) {
  W <- solve(V); X <- model.matrix(fit)
  P <- W - W %*% X %*% solve(t(X) %*% W %*% X) %*% t(X) %*% W
  typical <- (fit$k - fit$p) / sum(diag(P))
  total <- sum(fit$sigma2)
  list(typical = typical, total = 100 * total / (total + typical),
       components = 100 * fit$sigma2 / (total + typical))
}

multilevel <- function(fit, V, data, formula, newmods, profile = TRUE) {
  ci <- if (profile) confint(fit, control = list(tol = 1e-13, maxiter = 2000)) else NULL
  reduced <- lapply(1:2, function(a) {
    fixed <- c(NA, NA); fixed[a] <- 0
    update(fit, sigma2 = fixed)
  })
  pred <- if (is.null(newmods)) predict(fit) else predict(fit, newmods = newmods)
  list(sigma2 = fit$sigma2, beta = c(fit$beta), vb = mat(fit$vb), se = fit$se, zval = fit$zval,
       pval = fit$pval, ci_lb = fit$ci.lb, ci_ub = fit$ci.ub, ddf = if (fit$test == "t") fit$ddf else NULL,
       QE = fit$QE, QEp = fit$QEp, QM = fit$QM, QMp = fit$QMp, QMdf = fit$QMdf,
       fitstats = as.list(setNames(c(fitstats(fit)), rownames(fitstats(fit)))),
       vvc = mat(fit$vvc), i2 = i2(fit, V),
       profile = if (profile) lapply(1:2, function(a) unname(ci[[a]]$random[1, ])) else NULL,
       lrt = lapply(reduced, function(r) { a <- anova(fit, r); list(LRT = a$LRT, pval = a$pval, ll0 = c(logLik(r))) }),
       predict = list(pred = c(pred$pred), se = c(pred$se), ci_lb = c(pred$ci.lb), ci_ub = c(pred$ci.ub),
                      pi_lb = c(pred$pi.lb), pi_ub = c(pred$pi.ub)))
}

cr2 <- function(fit, cluster, constraints = NULL, nulls = NULL, moderators = NULL) {
  V <- if (is.null(cluster)) vcovCR(fit, type = "CR2") else vcovCR(fit, type = "CR2", cluster = cluster)
  ct <- coef_test(fit, vcov = V, test = "Satterthwaite")
  out <- list(vcov = mat(V), se = ct$SE, tstat = ct$tstat, df = ct$df_Satt, p = ct$p_Satt)
  if (!is.null(moderators)) {
    w <- Wald_test(fit, constraints = constrain_zero(moderators), vcov = V, test = c("HTZ", "chi-sq"))
    out$moderators <- list(F = w$Fstat[w$test == "HTZ"], df1 = w$df_num[w$test == "HTZ"],
                           df2 = w$df_denom[w$test == "HTZ"], p = w$p_val[w$test == "HTZ"],
                           delta = w$delta[w$test == "HTZ"], Q = w$Fstat[w$test == "chi-sq"] * w$df_num[1])
  }
  if (!is.null(constraints)) {
    w <- Wald_test(fit, constraints = constraints, vcov = V, test = c("HTZ", "chi-sq"), null_constant = nulls)
    out$contrasts <- list(F = w$Fstat[w$test == "HTZ"], df1 = w$df_num[w$test == "HTZ"],
                          df2 = w$df_denom[w$test == "HTZ"], p = w$p_val[w$test == "HTZ"],
                          Q = w$Fstat[w$test == "chi-sq"] * w$df_num[1])
    rows <- lapply(seq_len(nrow(constraints)), function(i) {
      w1 <- Wald_test(fit, constraints = constraints[i, , drop = FALSE], vcov = V, test = "HTZ", null_constant = nulls[i])
      list(F = w1$Fstat, df2 = w1$df_denom, p = w1$p_val)
    })
    out$contrast_rows <- rows
  }
  out
}

results <- list()

# ---- three-level: Konstantopoulos (2011) published data ---------------------
for (m in c("REML", "ML")) {
  for (test in c("z", "t")) {
    f0 <- rma.mv(yi, vi, random = ~ 1 | district/school, data = konst, method = m, test = test,
                 cvvc = TRUE, control = tight)
    results[[paste0("konst_int_", m, "_", test)]] <- multilevel(f0, diag(konst$vi), konst, NULL, NULL)
    f1 <- rma.mv(yi, vi, mods = ~ yc, random = ~ 1 | district/school, data = konst, method = m, test = test,
                 cvvc = TRUE, control = tight)
    results[[paste0("konst_yc_", m, "_", test)]] <- multilevel(f1, diag(konst$vi), konst, ~yc, c(-5, 0, 10),
                                                               profile = test == "z")
  }
}

# ---- three-level: synthetic correlated within-study sampling covariance -----
che <- read.csv(file.path(input, "che.csv"))
che_v <- as.matrix(read.csv(file.path(input, "che_covariance.csv"), check.names = FALSE))
for (m in c("REML", "ML")) {
  f <- rma.mv(yi, che_v, mods = ~ x1 + x2, random = ~ 1 | study/effect, data = che, method = m,
              cvvc = TRUE, control = tight)
  results[[paste0("che_", m)]] <- multilevel(f, che_v, che, ~x1 + x2, rbind(c(0, 0), c(1, -0.5)))
  results[[paste0("che_", m)]]$cr2 <- cr2(f, NULL, rbind(c(0, 1, 1), c(1, 0, 0)), c(0.1, 0.2), 2:3)
}

# ---- three-level boundary case ----------------------------------------------
boundary <- read.csv(file.path(input, "boundary.csv"))
for (m in c("REML", "ML")) {
  f <- rma.mv(yi, vi, random = ~ 1 | study/effect, data = boundary, method = m, control = tight)
  results[[paste0("boundary_", m)]] <- list(sigma2 = f$sigma2, beta = c(f$beta), vb = mat(f$vb),
                                            ll = c(logLik(f)))
}

# ---- CR2 on the published three-level data with a synthetic second moderator -
konst$z <- read.csv(file.path(input, "konst_z.csv"))$z
fz <- rma.mv(yi, vi, mods = ~ yc + z, random = ~ 1 | district/school, data = konst, method = "REML", control = tight)
results$konst_cr2 <- c(list(sigma2 = fz$sigma2, beta = c(fz$beta)),
                       cr2(fz, NULL, rbind(c(0, 1, 1), c(1, 0, 0)), c(0.01, 0.1), 2:3))

# ---- CR2 on dependent (one-component) fits with explicit study clusters ------
dep <- read.csv(file.path(input, "dependent.csv"))
dep_v <- as.matrix(read.csv(file.path(input, "dependent_covariance.csv"), check.names = FALSE))
for (m in c("REML", "ML")) {
  study_fit <- rma.mv(yi, dep_v, mods = ~ x, random = ~ 1 | study, data = dep, method = m, control = tight)
  effect_fit <- rma.mv(yi, dep_v, mods = ~ x, random = ~ 1 | effect, data = dep, method = m, control = tight)
  results[[paste0("dependent_study_", m)]] <- c(list(tau2 = study_fit$sigma2, beta = c(study_fit$beta)),
                                                cr2(study_fit, dep$study, moderators = 2))
  results[[paste0("dependent_effect_", m)]] <- c(list(tau2 = effect_fit$sigma2, beta = c(effect_fit$beta)),
                                                 cr2(effect_fit, dep$study, moderators = 2))
}
common_fit <- rma.mv(yi, dep_v, mods = ~ x, data = dep)
results$dependent_common <- c(list(beta = c(common_fit$beta)), cr2(common_fit, dep$study, moderators = 2))

# ---- trim-and-fill ------------------------------------------------------------
tf <- function(fit, ...) {
  t <- trimfill(fit, ...)
  # trimfill's final fit uses default rma control; refit the returned filled
  # data with the tight threshold for a high-precision adjusted estimate.
  r <- rma(c(t$yi), c(t$vi), method = fit$method, control = uni)
  list(k0 = t$k0, se_k0 = t$se.k0, side = t$side, p_k0 = t$p.k0, beta = c(r$beta), se = r$se,
       tau2 = r$tau2, ci_lb = r$ci.lb, ci_ub = r$ci.ub, QE = r$QE, I2 = r$I2,
       reported = list(beta = c(t$beta), se = t$se, tau2 = t$tau2),
       yi = c(t$yi), vi = c(t$vi), fill = t$fill)
}
asym <- read.csv(file.path(input, "asymmetric.csv"))
methods <- c(REML = "REML", ML = "ML", DL = "DL", common = "EE")
for (name in names(methods)) {
  hfit <- rma(yi, vi, data = hack, method = methods[[name]], control = uni)
  afit <- rma(yi, vi, data = asym, method = methods[[name]], control = uni)
  for (est in c("L0", "R0")) {
    results[[paste0("trimfill_hack_", name, "_", est)]] <- tf(hfit, estimator = est)
    results[[paste0("trimfill_asym_", name, "_", est)]] <- tf(afit, estimator = est)
  }
  results[[paste0("trimfill_hack_", name, "_right")]] <- tf(hfit, side = "right")
}

# ---- PET / PEESE (ordinary weighted least squares) ---------------------------
wls <- function(data, term) {
  f <- if (term == "sei") lm(yi ~ sqrt(vi), weights = 1 / vi, data = data) else lm(yi ~ vi, weights = 1 / vi, data = data)
  s <- summary(f)
  list(beta = unname(coef(f)), vcov = mat(vcov(f)), sigma = s$sigma, df = s$df[2],
       t = unname(s$coefficients[, 3]), p = unname(s$coefficients[, 4]))
}
for (nm in c("hack", "asym")) {
  d <- if (nm == "hack") hack else asym
  results[[paste0("pet_", nm)]] <- wls(d, "sei")
  results[[paste0("peese_", nm)]] <- wls(d, "vi")
}

# ---- step-function selection models ------------------------------------------
sel <- function(fit, steps, delta = NULL) {
  s <- if (is.null(delta)) selmodel(fit, type = "stepfun", steps = steps, control = tight)
       else selmodel(fit, type = "stepfun", steps = steps, delta = delta, control = tight)
  list(beta = c(s$beta), se = s$se, vb = mat(s$vb), tau2 = s$tau2, se_tau2 = s$se.tau2, delta = s$delta,
       se_delta = s$se.delta, zval_delta = s$zval.delta, ll = c(logLik(s)), ll0 = s$ll0, LRT = s$LRT,
       LRTdf = s$LRTdf, LRTp = s$LRTp, LRT_tau2 = s$LRT.tau2, k_intervals = c(table(factor(s$pgrp, levels = seq_along(s$steps)))))
}
wf <- function(effect, v, steps, mods = NULL, weights = NULL, fe = FALSE) {
  w <- weightfunct(effect, v, steps = steps, mods = mods, weights = weights, fe = fe)
  list(adj_est = c(w$adj_est), adj_se = c(w$adj_se), unadj_est = c(w$unadj_est), neg_ll_adj = w$output_adj$value,
       neg_ll_unadj = w$output_unadj$value)
}
hfit_ml <- rma(yi, vi, data = hack, method = "ML", control = uni)
results$selection_hack_4 <- sel(hfit_ml, c(.025, .10, .50, 1))
results$selection_hack_4$weightr <- wf(hack$yi, hack$vi, c(.025, .10, .50, 1))
results$selection_hack_2 <- sel(hfit_ml, c(.025, 1))
results$selection_hack_2$weightr <- wf(hack$yi, hack$vi, c(.025, 1))
results$selection_hack_fixed <- sel(hfit_ml, c(.025, .10, .50, 1), delta = c(1, .8, .6, .4))
results$selection_hack_fixed$weightr <- wf(hack$yi, hack$vi, c(.025, .10, .50, 1), weights = c(1, .8, .6, .4))
results$selection_hack_ee <- sel(rma(yi, vi, data = hack, method = "EE", control = uni), c(.025, .50, 1))
smod <- read.csv(file.path(input, "selection_moderator.csv"))
mfit <- rma(yi, vi, mods = ~ x, data = smod, method = "ML", control = uni)
results$selection_moderator <- sel(mfit, c(.05, .50, 1))
results$selection_moderator$weightr <- wf(smod$yi, smod$vi, c(.05, .50, 1), mods = ~ smod$x)

save(results, "oracles")
writeLines(c(R.version.string,
             paste("metafor", as.character(packageVersion("metafor"))),
             paste("metadat", as.character(packageVersion("metadat"))),
             paste("clubSandwich", as.character(packageVersion("clubSandwich"))),
             paste("weightr", as.character(packageVersion("weightr"))),
             paste("jsonlite", as.character(packageVersion("jsonlite")))),
           file.path(output, "runtime.txt"))
