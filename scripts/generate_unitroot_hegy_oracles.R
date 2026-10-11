# R reference values and published-table comparisons for oe.hegy.
#
# Source: R package uroot 2.1-3 (CRAN, GPL-2, Lopez-de-Lacalle and Boshnakov),
# https://cran.r-project.org/src/contrib/uroot_2.1-3.tar.gz, SHA-256
# 30f623d119299c8020c81c577d83e825811e96ec694ee1e2522fcbe02fed4c2f. The
# package is not installed: its plain R files (hegy.test, hegy.regressors,
# hegy.rs.pvalue, uroot.raw.pvalue) are sourced from the verified tarball and
# its sysdata.rda is loaded, which hold
#   .HEGY.orig.cv  Hylleberg, Engle, Granger and Yoo (1990), Table 1a/1b
#                  (t_pi1, t_pi2, F_pi3pi4; T = 48, 100, 136, 200),
#   .BM.orig.cv    Beaulieu and Miron (1993) monthly tables (T = 240, 480, 10000),
#   .HEGY.CM.tabs  Diaz-Emparanza (2014, CSDA 76) response-surface coefficients.
# Recorded:
#   1. hegy.test statistics, chosen lags and response-surface p-values on seeded
#      quarterly and monthly series (an independent lm() implementation that
#      oe.hegy must reproduce);
#   2. the original tables and the response-surface 1/5/10% quantiles at the
#      degrees of freedom of the OpenEconometrics simulation grid, so the tests
#      can report how far each source is from the seeded simulation;
#   3. a bounded null simulation with uroot's own hegy.test (3,000 seasonal random
#      walks, constant + seasonal dummies, T = 100 with set.seed(42) and T = 48
#      with set.seed(48)): quantiles of the five statistics and the rejection
#      rates of the response-surface p-values.
#
# Run (about two minutes; jsonlite in <lib>, network access to CRAN or a local
# copy of the tarball as the second argument):
#   R_LIBS=<lib> Rscript --vanilla scripts/generate_unitroot_hegy_oracles.R \
#     tests/fixtures/unitroot_hegy/uroot_reference.json [uroot_2.1-3.tar.gz]

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[1] else "uroot_reference.json"
suppressPackageStartupMessages(library(jsonlite))
expected <- "30f623d119299c8020c81c577d83e825811e96ec694ee1e2522fcbe02fed4c2f"
tarball <- if (length(args) > 1) args[2] else {
  path <- file.path(tempdir(), "uroot_2.1-3.tar.gz")
  for (url in c("https://cran.r-project.org/src/contrib/uroot_2.1-3.tar.gz",
                "https://cran.r-project.org/src/contrib/Archive/uroot/uroot_2.1-3.tar.gz")) {
    ok <- tryCatch(utils::download.file(url, path, quiet = TRUE, mode = "wb") == 0,
                   error = function(e) FALSE, warning = function(w) FALSE)
    if (isTRUE(ok)) break
  }
  path
}
if (unname(tools::sha256sum(tarball)) != expected) stop("uroot tarball checksum mismatch")
src <- file.path(tempdir(), "uroot-src")
utils::untar(tarball, exdir = src)
for (f in c("hegy-test.R", "hegy-regressors.R", "hegy-rs-pvalue.R", "tables-critical-values.R")) {
  sys.source(file.path(src, "uroot", "R", f), envir = globalenv())
}
load(file.path(src, "uroot", "R", "sysdata.rda"), envir = globalenv())

seasonal_walk <- function(n, S, e) {
  x <- numeric(n)
  for (t in seq_len(n)) x[t] <- (if (t > S) x[t - S] else 0) + e[t]
  x
}
set.seed(19900101)
series <- list()
for (S in c(4L, 12L)) {
  for (n in if (S == 4L) c(64L, 120L) else c(144L, 240L)) {
    e <- rnorm(n)
    walk <- seasonal_walk(n, S, e + 0.4 * c(0, head(e, -1)))
    stationary <- as.numeric(stats::filter(e, 0.5, method = "recursive")) +
      rep(c(2, -1, 0.5, -1.5, 1, 0, -0.5, 1.5, -2, 0.7, 0.3, -1)[seq_len(S)], length.out = n)
    series[[length(series) + 1]] <- list(name = paste0("walk", S, "_", n), S = S, y = walk + 0.02 * seq_len(n))
    series[[length(series) + 1]] <- list(name = paste0("stationary", S, "_", n), S = S, y = stationary)
  }
}
dets <- list(constant = c(1, 0, 0), seasonal = c(1, 0, 1), trend = c(1, 1, 0),
             seasonal_trend = c(1, 1, 1))
cases <- list()
for (s in series) {
  x <- ts(s$y, frequency = s$S)
  for (det in names(dets)) {
    specs <- list(list(method = "fixed", maxlag = 0L), list(method = "fixed", maxlag = 2L),
                  list(method = "AIC", maxlag = 4L), list(method = "BIC", maxlag = 4L))
    for (spec in specs) {
      r <- hegy.test(x, deterministic = dets[[det]], lag.method = spec$method,
                     maxlag = spec$maxlag, pvalue = "RS")
      cases[[length(cases) + 1]] <- list(series = s$name, seasons = s$S, deterministic = det,
        lag_method = spec$method, maxlag = spec$maxlag, lag_order = r$lag.order,
        statistic_names = names(r$statistics), statistics = unname(r$statistics),
        rs_pvalues = unname(r$pvalues), df_resid = df.residual(r$fitted.model))
    }
  }
  # deterministic = c(0, 0, 0) is not recorded: uroot 2.1-3 fails in lm() with an
  # empty xreg; the tests cover that case with an explicit NumPy regression.
}

rs_quantiles <- function(type, det, S, n) {
  label <- paste(switch(type, zero = "Ct1", pi = "Ct2", pair = "CF", seasall = "CFs", all = "CFt"),
                 switch(det, constant = "c", trend = "ct", seasonal = "cD", seasonal_trend = "cDt"),
                 "fijo", sep = "_")
  C <- .HEGY.CM.tabs[[label]]
  p <- 0
  x <- c(1, 1/n, 1/n^2, 1/n^3, p/n, p/n^2, p/n^3, p^2/n, p^2/n^2, p^2/n^3, p^3/n, p^3/n^2,
         p^3/n^3, S/n, S/n^2, S/n^3)
  q <- C[, seq_len(ncol(C) - 1)] %*% x
  as.numeric(q[c(13, 21, 31), 1])            # rq = 0.01, 0.05, 0.10 (upper tail for F)
}
grid <- list(`4` = c(40, 80, 160, 320, 1280), `12` = c(120, 240, 480, 1200))
response_surface <- list()
for (S in names(grid)) {
  for (det in names(dets)) {
    k <- switch(det, constant = 1, trend = 2, seasonal = as.integer(S), seasonal_trend = as.integer(S) + 1)
    for (N in grid[[S]]) {
      df <- N - k - as.integer(S)
      response_surface[[length(response_surface) + 1]] <- list(seasons = as.integer(S),
        deterministic = det, nobs = N, df_resid = df,
        t_1 = rs_quantiles("zero", det, as.integer(S), df), t_2 = rs_quantiles("pi", det, as.integer(S), df),
        pair = rs_quantiles("pair", det, as.integer(S), df),
        F_seasonal = rs_quantiles("seasall", det, as.integer(S), df),
        F_all = rs_quantiles("all", det, as.integer(S), df))
    }
  }
}
tab <- function(a) {
  lapply(dimnames(a)[[1]], function(ctd) list(code = ctd,
    values = lapply(dimnames(a)[[4]], function(stat) list(statistic = stat,
      rows = lapply(dimnames(a)[[2]], function(nn) list(T = as.numeric(nn),
        levels = as.numeric(dimnames(a)[[3]]), values = as.numeric(a[ctd, nn, , stat])))))))
}

uroot_null <- function(T, reps, seed) {
  set.seed(seed)
  stats <- pv <- matrix(NA, reps, 5)
  for (i in seq_len(reps)) {
    x <- seasonal_walk(T, 4L, rnorm(T))
    r <- hegy.test(ts(x, frequency = 4), deterministic = c(1, 0, 1), lag.method = "fixed",
                   maxlag = 0L, pvalue = "RS")
    stats[i, ] <- r$statistics
    pv[i, ] <- r$pvalues
  }
  lower <- c(0.01, 0.025, 0.05, 0.10)
  list(reps = reps, T = T, seasons = 4L, deterministic = "seasonal", seed = seed,
       statistic_names = names(r$statistics),
       quantiles = lapply(seq_len(5), function(j)
         as.numeric(quantile(stats[, j], if (j <= 2) lower else 1 - lower))),
       rejection_1pct = colMeans(pv < 0.01), rejection_5pct = colMeans(pv < 0.05),
       rejection_10pct = colMeans(pv < 0.10))
}
size <- uroot_null(100L, 3000L, 42L)
size48 <- uroot_null(48L, 3000L, 48L)

meta <- list(script = "scripts/generate_unitroot_hegy_oracles.R", uroot = "2.1-3",
             uroot_sha256 = expected, r = paste(R.version$major, R.version$minor, sep = "."),
             generated = as.character(Sys.Date()))
writeLines(toJSON(list(meta = meta, series = series, cases = cases, response_surface = response_surface,
                       hegy_1990 = tab(.HEGY.orig.cv), beaulieu_miron_1993 = tab(.BM.orig.cv),
                       response_surface_size = size,
                       uroot_null_t48 = size48),
                  auto_unbox = TRUE, digits = NA, null = "null"), out)
