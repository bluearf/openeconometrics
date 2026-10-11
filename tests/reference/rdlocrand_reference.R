# Reference numbers for tests/test_rdlocrand_reference.py (RD local randomization).
#
# Generated offline on 2026-10-10 with R 4.6.1 and rdlocrand 3.0 (CRAN 2026-10-04;
# imports AER 1.2-x and sandwich) from a private library:
#   R_LIBS=<private library> Rscript tests/reference/rdlocrand_reference.R
# The tests hard-code the printed numbers; neither R nor rdlocrand is needed at
# test time. The synthetic data come from the MINSTD integer generator
# (s <- 48271 s mod 2^31 - 1), sequential double sums and explicit
# floor(x 10^k + 0.5)/10^k rounding, so the test regenerates them bit for bit.
#
# Three kinds of numbers:
# * deterministic package output (large-sample statistics, HC fits, TSLS,
#   window sequences, approximate balance tests, default grids);
# * exact randomization p-values by brute-force enumeration of every assignment
#   with the package's own statistic function rdlocrand:::rdrandinf.model
#   (ties counted with |T*| >= |T| - 1e-9; the KS law also from ks.test's exact
#   tie-aware Smirnov p-value);
# * Monte Carlo p-values of rdrandinf with reps = 20000 (seed 123), compared
#   statistically because R's and torch's random streams differ.
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
dataset <- function(n = 200) {
  u <- minstd(20261010, n); v <- minstd(1234567891, n)
  z1 <- normals(987654321, n); z2 <- normals(1122334455, n); z3 <- normals(2024681357, n)
  r <- rnd(2 * u - 1, 3)
  t <- as.numeric((r >= 0 & v < 0.85) | (r < 0 & v < 0.15))
  x1 <- rnd(0.5 * r + z1, 3)
  x2 <- rnd(z2 + ifelse(r > 0.5, 0.6, ifelse(r < -0.5, -0.6, 0)), 3)
  y <- rnd(1 + r - 0.4 * r * r + 0.7 * t + z3, 3)
  data.frame(r = r, y = y, t = t, x1 = x1, x2 = x2)
}
d <- dataset()
X <- as.matrix(d[, c("x1", "x2")])
out("check_sums", c(sum(d$r), sum(d$y), sum(d$t), sum(d$x1), sum(d$x2)))

# ---- rdrandinf: deterministic (large-sample) output -------------------------------------------
cfg <- list(
  A = list(wl = -0.3, wr = 0.3, statistic = "all"),
  B = list(wl = -0.3, wr = 0.25, kernel = "triangular"),
  C = list(wl = -0.5, wr = 0.5, p = 1, ci = 0.05),
  D = list(wl = -0.5, wr = 0.4, p = 2, vce = "HC1", nulltau = 0.3, ci = 0.1),
  E = list(wl = -0.5, wr = 0.5, p = 1, vce = "HC2", kernel = "epan"),
  F = list(wl = -0.4, wr = 0.4, fuzzy = list(d$t, "tsls"), ci = 0.05),
  G = list(wl = -0.4, wr = 0.4, fuzzy = list(d$t, "tsls"), p = 1, nulltau = 0.5),
  H = list(wl = -0.4, wr = 0.4, fuzzy = d$t, nulltau = 0.5),
  I = list(wl = -0.4, wr = 0.4, fuzzy = d$t, p = 1, nulltau = 0.2),
  J = list(wl = -0.3, wr = 0.3, statistic = "ranksum", nulltau = 0.4, d = 0.7),
  K = list(wl = -0.2, wr = 0.35, p = 1, evall = -0.1, evalr = 0.1),
  L = list(wl = -0.5, wr = 0.5, p = 1, ci = c(0.05, seq(-1, 2, by = 0.05))),
  M = list(statistic = "ksmirnov")
)
for (nm in names(cfg)) {
  res <- suppressWarnings(do.call(rdrandinf, c(list(Y = d$y, R = d$r, reps = 10, quietly = TRUE), cfg[[nm]])))
  out(paste0("ri_", nm, "_stat"), res$obs.stat)
  out(paste0("ri_", nm, "_asy"), res$asy.pvalue)
  out(paste0("ri_", nm, "_sum"), res$sumstats)
  if (!is.null(res$se)) out(paste0("ri_", nm, "_se"), res$se)
  if (!is.null(res$ci)) out(paste0("ri_", nm, "_ci"), res$ci)
}
# Asymptotic power: the package computes it in rdrandinf.model (printed, not returned).
power_of <- function(wl, wr, statistic, nulltau = 0, delta = NULL) {
  ww <- round(d$r, 8) >= round(wl, 8) & round(d$r, 8) <= round(wr, 8)
  Y <- d$y[ww]; D <- as.numeric(d$r[ww] >= 0)
  if (is.null(delta)) delta <- 0.5 * sd(Y[D == 0])
  rdlocrand:::rdrandinf.model(Y - nulltau * D, D, statistic, pvalue = TRUE,
                              kweights = rep(1, length(D)), delta = delta)$asy.power
}
out("power_A", power_of(-0.3, 0.3, "all"))
out("power_J", power_of(-0.3, 0.3, "ranksum", 0.4, 0.7))

# ---- exact randomization p-values by enumeration ------------------------------------------------
stat_fun <- function(name, kw = NULL) function(Y, D) {
  w <- if (is.null(kw)) rep(1, length(D)) else kw
  as.numeric(rdlocrand:::rdrandinf.model(Y, D, name, kweights = w)$statistic)
}
ks_num <- function(Y, D) {  # max |c1 n0 - c0 n1| over the ends of tied blocks
  o <- order(Y); Ys <- Y[o]; Ds <- D[o]; n1 <- sum(D); n0 <- length(D) - n1
  c1 <- cumsum(Ds); c0 <- seq_along(Ds) - c1
  ends <- c(Ys[-1] != Ys[-length(Ys)], TRUE)
  max(abs(c1 * n0 - c0 * n1)[ends]) / (n0 * n1)
}
exact_fixed <- function(Y, D, stat, tol = 1e-9) {
  n <- length(D); obs <- abs(stat(Y, D)); cmb <- combn(n, sum(D))
  hits <- apply(cmb, 2, function(idx) { a <- numeric(n); a[idx] <- 1; abs(stat(Y, a)) >= obs - tol })
  c(mean(hits), ncol(cmb))
}
exact_bernoulli <- function(Y, D, prob, stat, tol = 1e-9) {
  n <- length(D); obs <- abs(stat(Y, D)); num <- 0; den <- 0
  for (k in 0:(2^n - 1)) {
    a <- as.integer(intToBits(k))[1:n]; s <- sum(a)
    if (s == 0 || s == n) next
    pr <- prod(ifelse(a == 1, prob, 1 - prob)); den <- den + pr
    if (abs(stat(Y, a)) >= obs - tol) num <- num + pr
  }
  num / den
}
window <- function(w) round(d$r, 8) >= round(-w, 8) & round(d$r, 8) <= round(w, 8)
ww <- window(0.1); Y <- d$y[ww]; D <- as.numeric(d$r[ww] >= 0)
out("exact_w010_n", c(sum(1 - D), sum(D)))
out("exact_w010_diffmeans", exact_fixed(Y, D, stat_fun("diffmeans")))
out("exact_w010_diffmeans_tau05", exact_fixed(Y - 0.5 * D, D, stat_fun("diffmeans")))
out("exact_w010_ranksum", exact_fixed(Y, D, stat_fun("ranksum")))
out("exact_w010_ksmirnov", exact_fixed(Y, D, ks_num, tol = 1e-12))
out("exact_w010_ksmirnov_kstest", ks.test(Y[D == 0], Y[D == 1])$p.value)
out("exact_w010_ar_tau04", exact_fixed(Y - 0.4 * d$t[ww], D, stat_fun("diffmeans")))
ww8 <- window(0.08); Y8 <- d$y[ww8]; R8 <- d$r[ww8]; D8 <- as.numeric(R8 >= 0)
kw8 <- ifelse(D8 == 1, (1 - abs(R8 / 0.08)) * (abs(R8 / 0.08) < 1), (1 - abs(R8 / -0.08)) * (abs(R8 / -0.08) < 1))
tri <- function(Y, D) { if (sum(kw8[D == 1]) == 0 || sum(kw8[D == 0]) == 0) return(NA); stat_fun("diffmeans", kw8)(Y, D) }
exact_fixed_na <- function(Y, D, stat, tol = 1e-9) {
  n <- length(D); obs <- abs(stat(Y, D)); cmb <- combn(n, sum(D))
  v <- apply(cmb, 2, function(idx) { a <- numeric(n); a[idx] <- 1; stat(Y, a) })
  c(mean(abs(v[!is.na(v)]) >= obs - tol), sum(!is.na(v)))
}
out("exact_w008_triangular", exact_fixed_na(Y8, D8, tri))
ww6 <- window(0.06); Y6 <- d$y[ww6]; D6 <- as.numeric(d$r[ww6] >= 0)
out("exact_w006_n", c(sum(1 - D6), sum(D6)))
for (nm in c("diffmeans", "ranksum")) {
  out(paste0("bern_w006_p04_", nm), exact_bernoulli(Y6, D6, rep(0.4, length(D6)), stat_fun(nm)))
  out(paste0("bern_w006_share_", nm), exact_bernoulli(Y6, D6, rep(mean(D6), length(D6)), stat_fun(nm)))
}
out("bern_w006_p04_ksmirnov", exact_bernoulli(Y6, D6, rep(0.4, length(D6)), ks_num, tol = 1e-12))
unit_prob <- seq(0.2, 0.7, length.out = length(D6))
out("bern_w006_unit_diffmeans", exact_bernoulli(Y6, D6, unit_prob, stat_fun("diffmeans")))

# Exact confidence set by inversion over a grid (window 0.06, diffmeans).
grid <- seq(-3, 4, by = 0.25)
pv <- sapply(grid, function(tau) exact_fixed(Y6 - tau * D6, D6, stat_fun("diffmeans"))[1])
out("ci_w006_grid_p", pv)
out("ci_w006_set", rdlocrand:::find_CI(pv, 0.05, grid))

# Rosenbaum-bound configurations (window 0.06, ranksum), every u, exact.
for (G in c(1.5, 2)) {
  lo <- 1 / (1 + G); hi <- G / (1 + G); dec <- order(Y6, decreasing = TRUE); up <- c(); low <- c()
  for (u in seq_along(Y6)) {
    pu <- rep(lo, length(Y6)); pu[dec[1:u]] <- hi
    pl <- rep(lo, length(Y6)); pl[rev(dec)[1:u]] <- hi
    up <- c(up, exact_bernoulli(Y6, D6, pu, stat_fun("ranksum")))
    low <- c(low, exact_bernoulli(Y6, D6, pl, stat_fun("ranksum")))
  }
  out(paste0("rbounds_w006_G", G, "_upper_u"), up)
  out(paste0("rbounds_w006_G", G, "_lower_u"), low)
}

# ---- Monte Carlo p-values (reps = 20000) ------------------------------------------------------
for (nm in c("diffmeans", "ranksum")) {
  res <- rdrandinf(d$y, d$r, wl = -0.3, wr = 0.3, statistic = nm, reps = 20000, seed = 123, quietly = TRUE)
  out(paste0("mc_w030_", nm), res$p.value)
}
res <- rdrandinf(d$y, d$r, wl = -0.3, wr = 0.3, bernoulli = rep(0.5, nrow(d)), reps = 20000, seed = 123, quietly = TRUE)
out("mc_w030_bernoulli05_diffmeans", res$p.value)

# ---- rdsensitivity --------------------------------------------------------------------------
sens <- rdsensitivity(d$y, d$r, reps = 10, nodraw = TRUE, quietly = TRUE)
out("sens_default_wlist", sens$wlist)
out("sens_default_wlist_left", sens$wlist_left)
out("sens_default_tlist", sens$tlist)
sens <- rdsensitivity(d$y, d$r, wlist = c(0.2, 0.3, 0.4), tlist = seq(-0.5, 1.5, by = 0.5), p = 1,
                      nodraw = TRUE, quietly = TRUE, ci = c(-0.3, 0.3))
out("sens_p1_results", sens$results)
out("sens_p1_ci", sens$ci)
sens <- rdsensitivity(d$y, d$r, wlist = c(0.2, 0.3, 0.4), wlist_left = c(-0.25, -0.3, -0.45),
                      tlist = c(0, 0.5, 1), p = 2, evalat = "means", vce = "HC2", nodraw = TRUE, quietly = TRUE)
out("sens_p2_means_results", sens$results)

# ---- rdwinselect ---------------------------------------------------------------------------------
wcfg <- list(
  default = list(), asym = list(wasymmetric = TRUE), step = list(wmin = 0.1, wstep = 0.05),
  pair = list(wmin = c(-0.12, 0.06), wobs = 6), small = list(obsmin = 6, wobs = 3, nwindows = 6),
  pairasym = list(wmin = c(-0.12, 0.06), wobs = 6, wasymmetric = TRUE), cut = list(cutoff = 0.1)
)
for (nm in names(wcfg)) {
  res <- suppressWarnings(do.call(rdwinselect, c(list(R = d$r, X = X, approx = TRUE, quietly = TRUE), wcfg[[nm]])))
  out(paste0("win_", nm, "_left"), res$results[, 6])
  out(paste0("win_", nm, "_right"), res$results[, 7])
  out(paste0("win_", nm, "_n"), c(res$results[, 4], res$results[, 5]))
  out(paste0("win_", nm, "_p"), res$results[, 1])
  out(paste0("win_", nm, "_binom"), res$results[, 3])
  out(paste0("win_", nm, "_rec"), c(res$w_left, res$w_right))
  out(paste0("win_", nm, "_summary"), res$summary)
}
scfg <- list(ks = list(statistic = "ksmirnov"), rank = list(statistic = "ranksum"),
             hot = list(statistic = "hotelling"), tri = list(kernel = "triangular"),
             p1 = list(p = 1), p1means = list(p = 1, evalat = "means", vce = "HC1"))
for (nm in names(scfg)) {
  res <- suppressWarnings(do.call(rdwinselect, c(list(R = d$r, X = X, approx = TRUE, quietly = TRUE,
                                                      wmin = 0.1, wstep = 0.05, nwindows = 6), scfg[[nm]])))
  out(paste0("winstat_", nm, "_p"), res$results[, 1])
}
# Exact randomization balance tests in the first three windows (obsmin 4, wobs 2).
res <- rdwinselect(d$r, X, obsmin = 4, wobs = 2, nwindows = 3, approx = TRUE, quietly = TRUE)
pmins <- c()
for (j in 1:3) {
  inside <- d$r >= res$results[j, 6] & d$r <= res$results[j, 7]
  Dj <- as.numeric(d$r[inside] >= 0)
  pj <- sapply(1:2, function(k) exact_fixed(X[inside, k], Dj, stat_fun("diffmeans"))[1])
  pmins <- c(pmins, min(pj))
}
out("win_exact_right", res$results[, 7])
out("win_exact_pmin", pmins)
# A discrete running variable: one window per mass point.
rd <- floor(minstd(55555, 300) * 31) - 15
xd <- cbind(rnd(normals(66666, 300), 3), rnd(normals(77777, 300) + 0.05 * rd, 3))
res <- rdwinselect(rd, xd, wmasspoints = TRUE, approx = TRUE, quietly = TRUE)
out("mass_left", res$results[, 6])
out("mass_right", res$results[, 7])
out("mass_n", c(res$results[, 4], res$results[, 5]))
out("mass_p", res$results[, 1])
out("mass_rec", c(res$w_left, res$w_right))
# Larger sample: Kolmogorov limit branch of ks.test (n0 n1 >= 10000).
big <- dataset(420)
res <- rdrandinf(big$y, big$r, statistic = "ksmirnov", reps = 10, quietly = TRUE)
out("big_ks", c(res$obs.stat, res$asy.pvalue, res$sumstats[2, ]))
