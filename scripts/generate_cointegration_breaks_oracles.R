#!/usr/bin/env Rscript
# External oracles for tests/fixtures/cointegration_breaks/oracles.json.
# No OpenEconometrics code runs in this process. Usage:
#   Rscript generate_cointegration_breaks_oracles.R cases.json hansen_main.R combcoint_sysdata.rda out.json [lib]
# hansen_main.R is main.R of Bruce Hansen's R programs for Gregory and Hansen (1996)
# (https://users.ssc.wisc.edu/~bhansen/progs/joe_96r.zip); only its procedures
# ESTIMATE, ADF and PHILLIPS are used. combcoint_sysdata.rda is R/sysdata.rda of
# CRAN combcoint 0.2.0 (the authors' Bayer-Hanck null distributions). [lib] is an
# optional colon-separated library path list holding makicoint 2.0.0, hatemicoint 1.1.0 and cointests 1.1.0.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L) stop("usage: cases.json hansen_main.R sysdata.rda out.json [lib]")
if (length(args) >= 5L) .libPaths(c(strsplit(args[5L], ":", fixed = TRUE)[[1L]], .libPaths()))
suppressPackageStartupMessages({
    library(jsonlite)
    library(urca)
})
source(args[2L])                      # defines estimate(), adf(), phillips(), main()
cases <- fromJSON(args[1L], simplifyVector = FALSE)
sysdata <- new.env()
load(args[3L], envir = sysdata)
options(digits = 17)

as_matrix <- function(columns) do.call(cbind, lapply(columns, function(v) as.numeric(unlist(v))))
grh_round <- function(v) floor(v + 0.5)
choice_code <- function(rule) switch(rule, fixed = 1, aic = 2, bic = 3, t = 4)

gh_design <- function(x, n, model, breaks) {
    out <- matrix(1, n, 1)
    dummies <- sapply(breaks, function(b) c(rep(0, b), rep(1, n - b)))
    dummies <- matrix(dummies, n)
    out <- cbind(out, dummies)
    tr <- seq_len(n)
    if (model %in% c("trend", "regime_trend")) out <- cbind(out, tr)
    if (model == "regime_trend") out <- cbind(out, dummies * tr)
    out <- cbind(out, x)
    if (model %in% c("regime", "regime_trend"))
        for (j in seq_len(ncol(dummies))) out <- cbind(out, dummies[, j] * x)
    out
}

scan_breaks <- function(y, x, model, rule, kmax, grid) {
    n <- nrow(y)
    stats <- matrix(NA_real_, nrow(grid), 4)
    for (r in seq_len(nrow(grid))) {
        x1 <- gh_design(x, n, model, grid[r, ])
        a <- adf(y, x1, kmax, choice_code(rule))
        p <- phillips(y, x1)
        stats[r, ] <- c(a$stata, a$lag, as.numeric(p$za), as.numeric(p$zt))
    }
    pick <- function(col) { i <- which.min(stats[, col]); list(i = i, value = stats[i, col]) }
    adf_best <- pick(1); za <- pick(3); zt <- pick(4)
    list(adf = adf_best$value, adf_breaks = as.integer(grid[adf_best$i, ]),
         adf_lag = stats[adf_best$i, 2],
         zt = zt$value, zt_breaks = as.integer(grid[zt$i, ]),
         za = za$value, za_breaks = as.integer(grid[za$i, ]))
}

results <- list()
for (case in cases$gregory_hansen) {
    data <- cases$datasets[[case$data]]
    y <- matrix(as.numeric(unlist(data$y)), ncol = 1)
    x <- as_matrix(data$x)
    n <- nrow(y)
    grid <- matrix(seq(grh_round(0.15 * n), grh_round(0.85 * n)), ncol = 1)
    out <- scan_breaks(y, x, case$model, case$rule, case$kmax, grid)
    results$gregory_hansen[[length(results$gregory_hansen) + 1L]] <- c(case, out)
}
for (case in cases$hatemi_j) {
    data <- cases$datasets[[case$data]]
    y <- matrix(as.numeric(unlist(data$y)), ncol = 1)
    x <- as_matrix(data$x)
    n <- nrow(y)
    begin <- grh_round(0.15 * n)
    pairs <- NULL
    for (t1 in begin:grh_round(0.70 * n)) for (t2 in (t1 + begin):grh_round(0.85 * n))
        pairs <- rbind(pairs, c(t1, t2))
    out <- scan_breaks(y, x, "regime", case$rule, case$kmax, pairs)
    if (requireNamespace("hatemicoint", quietly = TRUE) && case$rule == "fixed") {
        hj <- hatemicoint::hatemicoint(as.numeric(y), x, maxlags = case$kmax,
                                       lag_selection = "fixed", kernel = "iid")
        out$hatemicoint_adf <- hj$adf_min
        out$hatemicoint_adf_breaks <- c(hj$tb1_adf, hj$tb2_adf)
    }
    results$hatemi_j[[length(results$hatemi_j) + 1L]] <- c(case, out)
}
for (case in cases$maki) {
    data <- cases$datasets[[case$data]]
    mat <- cbind(as.numeric(unlist(data$y)), as_matrix(data$x))
    method <- if (case$rule == "t") "tsig" else "fixed"
    fit <- makicoint::coint_maki(mat, m = case$breaks,
                                 model = match(case$model, c("level", "trend", "regime", "regime_trend")) - 1L,
                                 trimm = 0.10, lagmethod = method, maxlags = case$kmax, engine = "gauss")
    results$maki[[length(results$maki) + 1L]] <- c(case, list(statistic = fit$statistic,
        break_dates = I(as.integer(fit$breakpoints)), critical = as.numeric(fit$critical_values)))
}

# ---- Bayer-Hanck components (bayerhanck.ado semantics) --------------------------------
ecm_stats <- function(z, lags, trend) {
    n <- nrow(z); k <- ncol(z)
    d <- diff(z)                                    # row s: D at observation s + 1
    rows <- (lags + 2):n                            # observations used
    w <- d[rows - 1, -1, drop = FALSE]              # contemporaneous Dx
    if (lags > 0) for (i in 1:lags) w <- cbind(w, d[rows - 1 - i, , drop = FALSE])
    det <- switch(trend, none = NULL, constant = matrix(1, length(rows)),
                  trend = cbind(1, rows))
    wd <- cbind(w, det)
    res <- function(v) qr.resid(qr(wd), v)
    r_levels <- sapply(seq_len(k), function(j) res(z[rows - 1, j]))
    r_dy <- res(d[rows - 1, 1])
    xf <- cbind(r_levels, det)
    fit <- lm.fit(xf, r_dy)
    s2 <- sum(fit$residuals^2) / (length(rows) - ncol(xf))
    v <- s2 * chol2inv(qr.R(qr(xf)))
    b <- fit$coefficients
    list(banerjee = b[1] / sqrt(v[1, 1]), boswijk = as.numeric(t(b) %*% solve(v, b)))
}
null_dist <- as.matrix(sysdata$null_dist)
write.csv(null_dist, file.path(dirname(args[4L]), "combcoint_null.csv"), row.names = FALSE)
pval <- function(stat, j, nvar, trend) {
    column <- 44 * (match(trend, c("none", "constant", "trend")) - 1) + 4 * (nvar - 2) + j
    draws <- null_dist[, column]
    if (j %in% c(1, 3)) sum(stat > draws) / length(draws) + 1e-12
    else sum(stat < draws) / length(draws) + 1e-12
}
for (case in cases$bayer_hanck) {
    data <- cases$datasets[[case$data]]
    z <- cbind(as.numeric(unlist(data$y)), as_matrix(data$x)); colnames(z) <- paste0("v", seq_len(ncol(z)))
    n <- nrow(z); L <- case$lags
    det <- switch(case$trend, none = NULL, constant = matrix(1, n), trend = cbind(1, seq_len(n)))
    e <- qr.resid(qr(cbind(z[, -1], det)), z[, 1])
    eg <- as.numeric(ur.df(e, type = "none", lags = L)@teststat)
    jo <- NA_real_
    if (case$trend == "constant" && L >= 1) {
        cj <- ca.jo(z, type = "eigen", ecdet = "none", K = L + 1)
        jo <- tail(cj@teststat, 1)
    }
    ecm <- ecm_stats(z, L, case$trend)
    nv <- ncol(z)
    stats <- c(eg, jo, ecm$banerjee, ecm$boswijk)
    p <- sapply(seq_along(stats), function(j) if (is.na(stats[j])) NA_real_ else pval(stats[j], j, nv, case$trend))
    results$bayer_hanck[[length(results$bayer_hanck) + 1L]] <- c(case, list(
        engle_granger = eg, johansen_max = jo, banerjee = unname(ecm$banerjee),
        boswijk = ecm$boswijk, p_values = p))
}

# ---- Fourier tests (CRAN cointests 1.1.0) -----------------------------------------------
for (case in cases$fourier) {
    data <- cases$datasets[[case$data]]
    y <- as.numeric(unlist(data$y)); x <- as_matrix(data$x)
    if (case$test == "fadl") {
        fit <- cointests::fcoint(y, x, test = "fadl", model = case$model,
                                 max_freq = case$max_frequency, max_lag = case$max_lag,
                                 criterion = case$criterion)$results$fadl
        out <- list(statistic = fit$tstat, frequency = fit$frequency, lags = fit$lag,
                    dx_lags = I(as.integer(fit$lag_dx)), nobs = fit$nobs)
    } else {
        fit <- cointests::fcoint(y, x, test = "tsong", model = case$model,
                                 max_freq = case$max_frequency, dols_lags = case$leads_lags,
                                 lrv = case$kernel, bandwidth = case$bandwidth)$results$tsong
        out <- list(statistic = fit$CI_stat, frequency = fit$frequency, fourier_F = fit$F_stat,
                    nobs = fit$nobs, statistic_ols = fit$CI_ols, frequency_ols = fit$frequency_ols,
                    fourier_F_ols = fit$F_ols)
    }
    results$fourier[[length(results$fourier) + 1L]] <- c(case, out)
}
writeLines(toJSON(results, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), args[4L])
