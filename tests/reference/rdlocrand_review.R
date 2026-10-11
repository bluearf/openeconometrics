# Reference numbers for tests/test_rdlocrand_review.py (independent review of rdlocrand).
#
# Generated offline on 2026-10-10 with R 4.6.1 and rdlocrand 3.0 (imports AER and
# sandwich) from a private library:
#   R_LIBS=<private library> Rscript tests/reference/rdlocrand_review.R
# The tests hard-code the printed numbers; neither R nor rdlocrand is needed at
# test time. Data come from the MINSTD integer generator with sequential double
# sums and explicit floor rounding (as in rdlocrand_reference.R), so the test
# regenerates them bit for bit. Configurations not covered by
# rdlocrand_reference.R: discrete and heaped running variables (mass points) in
# the symmetric, asymmetric, wmin and wstep window sequences, five covariates
# with Hotelling, asymmetric windows with polynomial/kernel fits, fuzzy TSLS and
# Anderson-Rubin fits with kernels, window selection inside rdrandinf, sensitivity
# grids with kernels and fuzzy designs, and exact randomization laws by
# brute-force enumeration (Bernoulli with unit probabilities from a logistic
# score, Bernoulli fuzzy Anderson-Rubin, kernel-weighted Anderson-Rubin,
# Hotelling with many covariates, tied outcomes on a discrete running variable,
# every Rosenbaum configuration of the package's search).
suppressMessages(library(rdlocrand))
out <- function(name, value) cat(name, "=", paste(sprintf("%.15g", as.numeric(value)), collapse = ", "), "\n")

minstd <- function(seed, n) {
  s <- seed; u <- numeric(n)
  for (i in seq_len(n)) { s <- (48271 * s) %% 2147483647; u[i] <- s / 2147483647 }
  u
}
normals <- function(seed, n) {
  u <- minstd(seed, 12 * n); z <- numeric(n)
  for (i in seq_len(n)) { acc <- 0; for (j in 1:12) acc <- acc + u[12 * (i - 1) + j]; z[i] <- acc - 6 }
  z
}
rnd <- function(x, k) floor(x * 10^k + 0.5) / 10^k
# Continuous design (as rdlocrand_reference.R) plus four more covariates.
dataset <- function(n = 200) {
  u <- minstd(20261010, n); v <- minstd(1234567891, n)
  z1 <- normals(987654321, n); z2 <- normals(1122334455, n); z3 <- normals(2024681357, n)
  r <- rnd(2 * u - 1, 3)
  t <- as.numeric((r >= 0 & v < 0.85) | (r < 0 & v < 0.15))
  x1 <- rnd(0.5 * r + z1, 3)
  x2 <- rnd(z2 + ifelse(r > 0.5, 0.6, ifelse(r < -0.5, -0.6, 0)), 3)
  y <- rnd(1 + r - 0.4 * r * r + 0.7 * t + z3, 3)
  x3 <- rnd(normals(31415, n) + 0.3 * x1, 3)
  x4 <- rnd(normals(27182, n) - 0.2 * r, 3)
  x5 <- rnd(normals(16180, n) + 0.5 * x2, 3)
  x6 <- rnd(normals(14142, n), 3)
  data.frame(r = r, y = y, t = t, x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5, x6 = x6)
}
# Discrete running variable with 21 mass points and an integer (tied) outcome.
discrete <- function(n = 150) {
  rd <- floor(minstd(424242, n) * 21) - 10
  v <- minstd(515151, n)
  td <- as.numeric((rd >= 0 & v < 0.8) | (rd < 0 & v < 0.2))
  yd <- rnd(0.5 + 0.08 * rd + 0.6 * td + normals(616161, n), 2)
  yc <- floor(minstd(717171, n) * 4) + (rd >= 0)
  xd1 <- rnd(normals(818181, n) + 0.03 * rd, 3)
  xd2 <- rnd(normals(919191, n), 3)
  xd3 <- rnd(normals(101010, n) - 0.02 * rd, 3)
  data.frame(r = rd, y = yd, yc = yc, t = td, x1 = xd1, x2 = xd2, x3 = xd3)
}
# Heaped running variable: one decimal, 21 heaps.
heaped <- function(n = 300) {
  rh <- rnd(2 * minstd(232323, n) - 1, 1)
  data.frame(r = rh, x1 = rnd(normals(343434, n) + 0.4 * rh, 3), x2 = rnd(normals(454545, n), 3))
}
d <- dataset(); dd <- discrete(); dh <- heaped()
X6 <- as.matrix(d[, paste0("x", 1:6)])
out("check_d", c(sum(d$x3), sum(d$x4), sum(d$x5), sum(d$x6)))
out("check_dd", c(sum(dd$r), sum(dd$y), sum(dd$yc), sum(dd$t), sum(dd$x1), sum(dd$x2), sum(dd$x3)))
out("check_dh", c(sum(dh$r), sum(dh$x1), sum(dh$x2)))

# ---- rdwinselect: mass points, heaps, many covariates (approximate tests) ----------------------
wins <- list(
  dd_default = list(R = dd$r, X = as.matrix(dd[, c("x1", "x2", "x3")])),
  dd_wobs = list(R = dd$r, X = as.matrix(dd[, c("x1", "x2", "x3")]), obsmin = 15, wobs = 25),
  dd_asym = list(R = dd$r, X = as.matrix(dd[, c("x1", "x2", "x3")]), obsmin = 12, wobs = 20, wasymmetric = TRUE),
  dd_wmin = list(R = dd$r, X = as.matrix(dd[, c("x1", "x2", "x3")]), wmin = 2.5, wobs = 18),
  dd_pairasym = list(R = dd$r, X = as.matrix(dd[, c("x1", "x2", "x3")]), wmin = c(-3, 1), wobs = 15, wasymmetric = TRUE),
  dd_step = list(R = dd$r, X = as.matrix(dd[, c("x1", "x2", "x3")]), wmin = 1, wstep = 1, nwindows = 8),
  dd_cut = list(R = dd$r, X = as.matrix(dd[, c("x1", "x2", "x3")]), cutoff = 0.5, wasymmetric = TRUE),
  dd_hot = list(R = dd$r, X = as.matrix(dd[, c("x1", "x2", "x3")]), statistic = "hotelling"),
  dh_default = list(R = dh$r, X = as.matrix(dh[, c("x1", "x2")]), obsmin = 15, wobs = 20),
  dh_asym = list(R = dh$r, X = as.matrix(dh[, c("x1", "x2")]), obsmin = 15, wobs = 20, wasymmetric = TRUE),
  d_hot6 = list(R = d$r, X = X6, statistic = "hotelling", wmin = 0.1, wstep = 0.05, nwindows = 8),
  d_rank6 = list(R = d$r, X = X6, statistic = "ranksum", obsmin = 8, wobs = 6, wasymmetric = TRUE),
  d_ks6 = list(R = d$r, X = X6, statistic = "ksmirnov", wmin = c(-0.2, 0.1), wobs = 7),
  d_p2_6 = list(R = d$r, X = X6, p = 2, vce = "HC2", wmin = 0.15, wstep = 0.05, nwindows = 6),
  d_epan6 = list(R = d$r, X = X6, kernel = "epan", wmin = 0.1, wstep = 0.05, nwindows = 6)
)
for (nm in names(wins)) {
  res <- suppressWarnings(do.call(rdwinselect, c(list(approx = TRUE, quietly = TRUE), wins[[nm]])))
  out(paste0("win_", nm, "_left"), res$results[, 6])
  out(paste0("win_", nm, "_right"), res$results[, 7])
  out(paste0("win_", nm, "_n"), c(res$results[, 4], res$results[, 5]))
  out(paste0("win_", nm, "_p"), res$results[, 1])
  out(paste0("win_", nm, "_var"), res$results[, 2])
  out(paste0("win_", nm, "_binom"), res$results[, 3])
  out(paste0("win_", nm, "_rec"), c(res$w_left, res$w_right))
  out(paste0("win_", nm, "_summary"), res$summary)
}

# ---- rdrandinf: deterministic output in new configurations ---------------------------------------
cfg <- list(
  T1 = list(Y = d$y, R = d$r, wl = -0.4, wr = 0.35, fuzzy = list(d$t, "tsls"), kernel = "triangular", ci = 0.05),
  T2 = list(Y = d$y, R = d$r, wl = -0.45, wr = 0.3, fuzzy = list(d$t, "tsls"), kernel = "epan", p = 1, vce = "HC2", nulltau = 0.2),
  T3 = list(Y = d$y, R = d$r, wl = -0.4, wr = 0.4, fuzzy = d$t, kernel = "triangular", p = 1, vce = "HC1", nulltau = 0.3),
  T4 = list(Y = d$y, R = d$r, wl = -0.5, wr = 0.45, fuzzy = d$t, p = 2, evall = -0.05, evalr = 0.1, vce = "HC2"),
  T5 = list(Y = d$y, R = d$r, wl = -0.45, wr = 0.3, p = 2, kernel = "triangular", nulltau = 0.1, ci = 0.1),
  T6 = list(Y = d$y, R = d$r, wl = -0.45, wr = 0.5, p = 1, kernel = "epan", ci = c(0.1, seq(-1, 2.5, by = 0.1))),
  T7 = list(Y = d$y, R = d$r, wl = -0.25, wr = 0.4, statistic = "all", d = 0.4),
  T8 = list(Y = dd$y, R = dd$r, wl = -2, wr = 2, statistic = "all"),
  T9 = list(Y = dd$y, R = dd$r, wl = -3, wr = 3, fuzzy = list(dd$t, "tsls"), p = 1, vce = "HC3"),
  T10 = list(Y = dd$yc, R = dd$r, wl = -4, wr = 3, statistic = "ranksum"),
  T11 = list(Y = dd$y, R = dd$r, cutoff = 0.5, wl = -1.5, wr = 3.5, p = 1, vce = "HC1", ci = 0.05),
  T12 = list(Y = dd$yc, R = dd$r, wl = -4, wr = 3, statistic = "all")
)
for (nm in names(cfg)) {
  res <- suppressWarnings(do.call(rdrandinf, c(list(reps = 10, quietly = TRUE), cfg[[nm]])))
  out(paste0("ri_", nm, "_stat"), res$obs.stat)
  out(paste0("ri_", nm, "_asy"), res$asy.pvalue)
  out(paste0("ri_", nm, "_sum"), res$sumstats)
  if (!is.null(res$se)) out(paste0("ri_", nm, "_se"), res$se)
  if (!is.null(res$ci)) out(paste0("ri_", nm, "_ci"), res$ci)
}
# rdlocrand's 'all' standardizes the rank sum with var(1:n); its 'ranksum' uses the
# tie-corrected var(rank(Y)), which the port uses for both.
res <- rdrandinf(dd$yc, dd$r, wl = -4, wr = 3, statistic = "ranksum", reps = 10, quietly = TRUE)
out("ri_T12_ranksum_stat", c(res$obs.stat, res$asy.pvalue))
# Window selected inside rdrandinf from covariates (approximate balance tests).
res <- suppressWarnings(rdrandinf(d$y, d$r, cutoff = 0.1, covariates = d[, c("x1", "x2")], approx = TRUE,
                                  reps = 10, quietly = TRUE))
out("ri_covsel_window", res$window)
out("ri_covsel_stat", c(res$obs.stat, res$asy.pvalue))

# ---- rdsensitivity --------------------------------------------------------------------------------
sens <- rdsensitivity(d$y, d$r, wlist = c(0.2, 0.3, 0.4), wlist_left = c(-0.3, -0.3, -0.35),
                      tlist = seq(-0.5, 1.5, by = 0.5), p = 1, kernel = "triangular", vce = "HC2",
                      ci = c(-0.3, 0.3), ci_alpha = 0.1, nodraw = TRUE, quietly = TRUE)
out("sens_kernel_results", sens$results)
out("sens_kernel_ci", sens$ci)
sens <- rdsensitivity(d$y, d$r, wlist = c(0.3, 0.4), tlist = c(0, 0.5, 1, 1.5), p = 1, fuzzy = d$t,
                      nodraw = TRUE, quietly = TRUE)
out("sens_fuzzy_results", sens$results)
sens <- rdsensitivity(dd$y, dd$r, reps = 10, nodraw = TRUE, quietly = TRUE)
out("sens_dd_wlist", sens$wlist)
out("sens_dd_tlist", sens$tlist)
sens <- rdsensitivity(d$y, d$r, wlist = c(0.25, 0.35), fuzzy = d$t, reps = 10, nodraw = TRUE, quietly = TRUE)
out("sens_fuzzy_default_tlist", sens$tlist)

# ---- exact randomization laws by enumeration ------------------------------------------------------
stat_fun <- function(name, kw = NULL) function(Y, D) {
  w <- if (is.null(kw)) rep(1, length(D)) else kw
  as.numeric(rdlocrand:::rdrandinf.model(Y, D, name, kweights = w)$statistic)
}
ks_num <- function(Y, D) {
  o <- order(Y); Ys <- Y[o]; Ds <- D[o]; n1 <- sum(D); n0 <- length(D) - n1
  c1 <- cumsum(Ds); c0 <- seq_along(Ds) - c1
  ends <- c(Ys[-1] != Ys[-length(Ys)], TRUE)
  max(abs(c1 * n0 - c0 * n1)[ends]) / (n0 * n1)
}
exact_fixed <- function(Y, D, stat, tol = 1e-9) {
  n <- length(D); obs <- abs(stat(Y, D)); cmb <- combn(n, sum(D))
  v <- apply(cmb, 2, function(idx) { a <- numeric(n); a[idx] <- 1; stat(Y, a) })
  c(mean(abs(v[!is.na(v)]) >= obs - tol), sum(!is.na(v)))
}
all_bits <- function(n) {
  m <- matrix(0L, 2^n, n)
  for (j in seq_len(n)) m[, j] <- (0:(2^n - 1) %/% 2^(j - 1)) %% 2
  m
}
exact_bernoulli <- function(Y, D, prob, stat, tol = 1e-9) {
  n <- length(D); obs <- abs(stat(Y, D)); A <- all_bits(n)
  keep <- rowSums(A) > 0 & rowSums(A) < n; A <- A[keep, , drop = FALSE]
  pr <- apply(A, 1, function(a) prod(ifelse(a == 1, prob, 1 - prob)))
  hit <- apply(A, 1, function(a) abs(stat(Y, a)) >= obs - tol)
  sum(pr * hit) / sum(pr)
}
window <- function(r, wl, wr) round(r, 8) >= round(wl, 8) & round(r, 8) <= round(wr, 8)
# Bernoulli with unit probabilities from a logistic score in the running variable.
ww <- window(d$r, -0.06, 0.06); Y <- d$y[ww]; Rw <- d$r[ww]; D <- as.numeric(Rw >= 0); Tw <- d$t[ww]
prob <- plogis(-0.4 + 9 * Rw)
out("logit_prob", prob)
out("bern_logit_diffmeans", exact_bernoulli(Y, D, prob, stat_fun("diffmeans")))
out("bern_logit_ranksum", exact_bernoulli(Y, D, prob, stat_fun("ranksum")))
out("bern_logit_ksmirnov", exact_bernoulli(Y, D, prob, ks_num, tol = 1e-12))
out("bern_logit_diffmeans_tau1", exact_bernoulli(Y - 1 * D, D, prob, stat_fun("diffmeans")))
# Bernoulli fuzzy Anderson-Rubin (difference in means of Y - tau T by D); rdrandinf itself
# fails for fuzzy designs with bernoulli (it calls the model with an empty statistic).
out("bern_ar_tau05", exact_bernoulli(Y - 0.5 * Tw, D, rep(0.35, length(D)), stat_fun("diffmeans")))
# Kernel-weighted Anderson-Rubin (rdlocrand ignores kernel weights for 'ar'; the weighted
# statistic is the package's weighted difference in means of Y - tau T).
w8 <- window(d$r, -0.08, 0.07); Y8 <- d$y[w8]; R8 <- d$r[w8]; D8 <- as.numeric(R8 >= 0); T8 <- d$t[w8]
kw8 <- ifelse(D8 == 1, (1 - abs(R8 / 0.07)) * (abs(R8 / 0.07) < 1), (1 - abs(R8 / -0.08)) * (abs(R8 / -0.08) < 1))
tri <- function(Y, D) { if (sum(kw8[D == 1]) == 0 || sum(kw8[D == 0]) == 0) return(NA); stat_fun("diffmeans", kw8)(Y, D) }
out("exact_ar_triangular_tau04", exact_fixed(Y8 - 0.4 * T8, D8, tri))
out("exact_ar_uniform_tau04", exact_fixed(Y8 - 0.4 * T8, D8, stat_fun("diffmeans")))
# Hotelling with five and six covariates (fixed margins, every assignment).
hot <- function(cols) function(Y, D) rdlocrand:::hotelT2(X6[wh, cols], D)$statistic
wh <- d$r >= -0.07 & d$r <= 0.07; Dh <- as.numeric(d$r[wh] >= 0)
out("hot_n", c(sum(1 - Dh), sum(Dh)))
out("hot5_exact", exact_fixed(NULL, Dh, hot(1:5)))
out("hot6_exact", exact_fixed(NULL, Dh, hot(1:6)))
out("hot6_F", rdlocrand:::hotelT2(X6[wh, ], Dh)$p.value)
# Discrete running variable with a tied integer outcome: window [-1, 0] holds two mass points.
wd <- window(dd$r, -1, 0); Yd <- dd$yc[wd]; Dd <- as.numeric(dd$r[wd] >= 0)
out("disc_n", c(sum(1 - Dd), sum(Dd)))
out("disc_ranksum", exact_fixed(Yd, Dd, stat_fun("ranksum")))
out("disc_ksmirnov", exact_fixed(Yd, Dd, ks_num, tol = 1e-12))
out("disc_diffmeans", exact_fixed(Yd, Dd, stat_fun("diffmeans")))
out("disc_kstest", suppressWarnings(ks.test(Yd[Dd == 0], Yd[Dd == 1])$p.value))
out("disc_stats", c(stat_fun("ranksum")(Yd, Dd), ks_num(Yd, Dd)))
# Every configuration of the package's Rosenbaum search (the u units with the largest
# outcomes receive Gamma/(1 + Gamma), the others 1/(1 + Gamma)), window 0.06, exact.
ww <- window(d$r, -0.06, 0.06); Y <- d$y[ww]; D <- as.numeric(d$r[ww] >= 0)
for (st in c("diffmeans", "ksmirnov")) {
  f <- if (st == "ksmirnov") ks_num else stat_fun(st)
  base <- exact_bernoulli(Y, D, rep(mean(D), length(D)), f)
  out(paste0("rb_", st, "_base"), base)
  for (G in c(1.5, 3)) {
    lo <- 1 / (1 + G); hi <- G / (1 + G); dec <- order(Y, decreasing = TRUE); pv <- c()
    for (u in seq_along(Y)) { pu <- rep(lo, length(Y)); pu[dec[1:u]] <- hi; pv <- c(pv, exact_bernoulli(Y, D, pu, f)) }
    out(paste0("rb_", st, "_G", G, "_u"), pv)
  }
}

# ---- zero kernel weights at the window limits (discrete running variable, window [-3, 3]) -------
# The package's p = 0 kernel paths call sandwich::vcovHC on lm/AER::ivreg fits with zero weights
# (units at r = -3 and r = 3); the same fits on the positive-weight units are the reference.
res <- rdrandinf(dd$y, dd$r, wl = -3, wr = 3, fuzzy = list(dd$t, "tsls"), kernel = "triangular", quietly = TRUE)
out("zero_tsls_package", c(res$obs.stat, res$asy.pvalue))
res <- suppressWarnings(rdrandinf(dd$y, dd$r, wl = -3, wr = 3, kernel = "triangular", reps = 10, quietly = TRUE))
out("zero_hc2_package", c(res$obs.stat, res$asy.pvalue))
ww <- window(dd$r, -3, 3); Rw <- dd$r[ww]; Dw <- as.numeric(Rw >= 0); Yw <- dd$y[ww]; Tw <- dd$t[ww]
kw <- ifelse(Dw == 1, (1 - abs(Rw / 3)) * (abs(Rw / 3) < 1), (1 - abs(Rw / -3)) * (abs(Rw / -3) < 1))
pos <- kw > 0
out("zero_weights", c(sum(!pos), length(pos)))
iv <- AER::ivreg(Yw[pos] ~ Tw[pos] | Dw[pos], weights = kw[pos])
se <- sqrt(sandwich::vcovHC(iv, type = "HC1")[2, 2])
out("zero_tsls_positive", c(coef(iv)[2], se, 2 * pnorm(-abs(coef(iv)[2] / se))))
fit <- lm(Yw[pos] ~ Dw[pos], weights = kw[pos])
se <- sqrt(sandwich::vcovHC(fit, type = "HC2")[2, 2])
out("zero_hc2_positive", c(coef(fit)[2], se, 2 * pnorm(-abs(coef(fit)[2] / se))))
