# Independent R references for the Bai-Perron extensions (QA only).
#
#   Rscript --vanilla scripts/generate_unitroot_breaks_oracles.R <mbreaks-R-dir> <out.json>
#
# <mbreaks-R-dir> holds the R sources of CRAN mbreaks 1.0.1 (Utilities.R,
# Variance.R, Test.R, Estimation.R and sysdata.rda from
# https://github.com/cran/mbreaks/tree/master/R; Nguyen, Yamamoto and Perron).
# strucchange (CRAN) and jsonlite must be installed.
#
# * mbreaks dating/pftest/spflp1 give the global dates, the HAC (robust = 1,
#   hetdat = hetvar = 1, prewhit = 0/1) and homoskedastic (robust = 0,
#   hetvar = 0) supF statistics and the sequential supF(l+1|l) statistics.
#   mbreaks reports q times the F statistic.
# * strucchange's pargmaxV (Bai 1997 CDF, an implementation independent of
#   Perron's funcg) supplies the unrounded 90% / 95% break-date bounds with
#   regime-specific Q and error variances (het.reg = het.err = TRUE), as in
#   confint.breakpointsfull but before ceiling/floor.
# * The HAC-interval inputs follow BRCODE.SRC (phi2 uses Q of regime i+1;
#   mbreaks 1.0.1 uses Q of regime i there) with mbreaks' correct(); their
#   quantiles come from strucchange's pargmaxV.
suppressMessages({library(strucchange); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]; out <- args[2]
for (f in c("Utilities.R", "Variance.R", "Test.R", "Estimation.R")) source(file.path(dir, f))
load(file.path(dir, "sysdata.rda"))

case_data <- function() {
  set.seed(20261010)
  n1 <- 120
  e1 <- as.numeric(arima.sim(list(ar = 0.5), n1))
  y1 <- 1 + 2 * (seq_len(n1) > 40) - 1.5 * (seq_len(n1) > 80) + e1
  n2 <- 100
  x2 <- rnorm(n2)
  u2 <- rnorm(n2) * (1 + 0.8 * (seq_len(n2) > 60))
  y2 <- ifelse(seq_len(n2) > 50, 3 + 1.5 * x2, 1 + 0.5 * x2) + u2
  list(mean = list(y = y1, x = NULL, m = 3, eps = 0.15),
       regression = list(y = y2, x = x2, m = 2, eps = 0.20))
}

strucchange_bounds <- function(y, z, bp, level) {
  # confint.breakpointsfull internals (het.reg = het.err = TRUE, vcov. = NULL), unrounded.
  n <- length(y); a2 <- (1 - level) / 2
  myfun <- function(x, level, xi, phi1, phi2) pargmaxV(x, xi = xi, phi1 = phi1, phi2 = phi2) - level
  b <- c(0, bp, n); res <- list()
  for (i in 2:(length(b) - 1)) {
    i1 <- (b[i - 1] + 1):b[i]; i2 <- (b[i] + 1):b[i + 1]
    f1 <- lm.fit(z[i1, , drop = FALSE], y[i1]); f2 <- lm.fit(z[i2, , drop = FALSE], y[i2])
    delta <- f2$coefficients - f1$coefficients
    Q1 <- crossprod(z[i1, , drop = FALSE]) / length(i1); Q2 <- crossprod(z[i2, , drop = FALSE]) / length(i2)
    Qp1 <- as.numeric(t(delta) %*% Q1 %*% delta); Qp2 <- as.numeric(t(delta) %*% Q2 %*% delta)
    phi1 <- sqrt(sum(f1$residuals^2) / length(i1)); phi2 <- sqrt(sum(f2$residuals^2) / length(i2))
    xi <- Qp2 / Qp1
    ub <- lb <- 0
    while (pargmaxV(ub, xi = xi, phi1 = phi1, phi2 = phi2) < (1 - a2)) ub <- ub + 1000
    while (pargmaxV(lb, xi = xi, phi1 = phi1, phi2 = phi2) > a2) lb <- lb - 1000
    upper <- uniroot(myfun, c(0, ub), level = 1 - a2, xi = xi, phi1 = phi1, phi2 = phi2, tol = 1e-12)$root
    lower <- uniroot(myfun, c(lb, 0), level = a2, xi = xi, phi1 = phi1, phi2 = phi2, tol = 1e-12)$root
    scale <- phi1^2 / Qp1
    res[[i - 1]] <- c(lower = b[i] - upper * scale, upper = b[i] - lower * scale)
  }
  res
}

gauss_hac_interval <- function(y, z, b, prewhit) {
  # BRCODE.SRC interval(), robust = 1, hetq = hetomega = 1, before int().
  n <- nrow(z); q <- ncol(z); m <- length(b); bf <- c(0, b, n)
  zbar <- diag_par(z, m, matrix(b, ncol = 1))
  delta <- OLS(matrix(y, ncol = 1), zbar); res <- matrix(y, ncol = 1) - zbar %*% delta
  out <- list()
  for (i in 1:m) {
    delv <- delta[(i * q + 1):((i + 1) * q), 1] - delta[((i - 1) * q + 1):(i * q), 1]
    i1 <- (bf[i] + 1):bf[i + 1]; i2 <- (bf[i + 1] + 1):bf[i + 2]
    qmat <- crossprod(z[i1, , drop = FALSE]) / length(i1)
    qmat1 <- crossprod(z[i2, , drop = FALSE]) / length(i2)
    omega <- correct(z[i1, , drop = FALSE], res[i1, 1, drop = FALSE], prewhit)
    omega1 <- correct(z[i2, , drop = FALSE], res[i2, 1, drop = FALSE], prewhit)
    dq <- as.numeric(t(delv) %*% qmat %*% delv); dq1 <- as.numeric(t(delv) %*% qmat1 %*% delv)
    phi1s <- as.numeric(t(delv) %*% omega %*% delv) / dq
    phi2s <- as.numeric(t(delv) %*% omega1 %*% delv) / dq1
    # Quantiles of the argmax CDF by strucchange's pargmaxV (tight root finding)
    # rather than mbreaks' cvg, whose bisection stops at |CDF - level| < 1e-6.
    xi <- dq1 / dq
    cvf <- sapply(c(0.025, 0.05, 0.95, 0.975), function(level) uniroot(function(x)
      pargmaxV(x, xi = xi, phi1 = sqrt(phi1s), phi2 = sqrt(phi2s)) - level, c(-2000, 2000), tol = 1e-12)$root)
    a <- dq^2 / as.numeric(t(delv) %*% omega %*% delv)
    out[[i]] <- c(lower_95 = b[i] - cvf[4] / a, upper_95 = b[i] - cvf[1] / a,
                  lower_90 = b[i] - cvf[3] / a, upper_90 = b[i] - cvf[2] / a,
                  eta = dq1 / dq, phi1 = phi1s, phi2 = phi2s)
  }
  out
}

result <- list(source = "mbreaks 1.0.1 R sources; strucchange", strucchange = as.character(packageVersion("strucchange")))
# strucchange's argmax CDF on a grid: funcg(x; eta, phi1^2, phi2^2) must equal pargmaxV(x; xi = eta, phi1, phi2).
grid <- c(-60, -25, -8, -2, -0.5, -0.01, 0.01, 0.5, 2, 8, 25, 60)
result$cdf <- lapply(list(c(1, 1, 1), c(1.7, 0.6, 1.4), c(0.4, 2.0, 0.8), c(2.5, 1.0, 3.0)), function(p)
  list(xi = p[1], phi1 = p[2], phi2 = p[3], x = grid,
       cdf = sapply(grid, function(x) pargmaxV(x, xi = p[1], phi1 = p[2], phi2 = p[3]))))
for (name in names(case_data())) {
  d <- case_data()[[name]]
  y <- matrix(d$y, ncol = 1); n <- length(d$y)
  z <- if (is.null(d$x)) matrix(1, n, 1) else cbind(1, d$x)
  q <- ncol(z); h <- floor(d$eps * n); m <- d$m
  dt <- dating(y, z, h, m, q, n)
  entry <- list(y = d$y, x = d$x, m = m, eps = d$eps, h = h, datevec = dt$datevec, glb = as.numeric(dt$glb))
  for (option in list(c(robust = 1, prewhit = 1, hetvar = 1), c(robust = 1, prewhit = 0, hetvar = 1),
                      c(robust = 0, prewhit = 0, hetvar = 0))) {
    key <- sprintf("robust%d_prewhit%d", option[["robust"]], option[["prewhit"]])
    supf <- sapply(1:m, function(i) pftest(y, z, i, q, n, dt$datevec, option[["prewhit"]], option[["robust"]], 0, 0, 1, option[["hetvar"]]))
    seq <- c(supf[1], sapply(seq_len(m - 1), function(i) spflp1(dt$bigvec, dt$datevec[1:i, i, drop = FALSE], i + 1, y, z, h, q,
                                                   option[["prewhit"]], option[["robust"]], 0, 0, 1, option[["hetvar"]])$maxf))
    entry[[key]] <- list(supf_scaled = as.numeric(supf), sequential_scaled = as.numeric(seq))
  }
  for (k in 1:m) {
    bp <- dt$datevec[1:k, k]
    entry[[sprintf("strucchange_%d", k)]] <- list(breaks = bp, ci95 = strucchange_bounds(d$y, z, bp, 0.95),
                                                    ci90 = strucchange_bounds(d$y, z, bp, 0.90))
    entry[[sprintf("gauss_hac_interval_%d", k)]] <- list(prewhit1 = gauss_hac_interval(d$y, z, bp, 1),
                                                         prewhit0 = gauss_hac_interval(d$y, z, bp, 0))
  }
  result[[name]] <- entry
}
writeLines(toJSON(result, digits = NA, auto_unbox = TRUE), out)
