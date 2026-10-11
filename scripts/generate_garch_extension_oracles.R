# Reference values for the GARCH extensions from R (rugarch, fGarch, rmgarch, midasr).
#
# Development-only oracle: none of these packages is a runtime or test dependency of
# OpenEconometrics. The JSON written here is committed under
# tests/fixtures/garch_extensions/ and the tests compare the native float64 Torch
# implementation with it. Every series is simulated here from a fixed seed and stored.
#
# Usage (from the repository root, with the packages on the library path):
#
#     Rscript --vanilla scripts/generate_garch_extension_oracles.R [LIBRARY]
#
# Sections (one JSON file each, all under 200 KB):
#
# rugarch_sged.json
#   * density: fGarch::dsged and rugarch::ddist("sged") log densities on a grid.
#   * likelihood: sGARCH(1,1) with sged errors at fixed parameters by
#     rugarch::ugarchfilter. rugarch starts its recursion with h_1 = mean(e^2)
#     (C function sgarchfilterC: h[0] = hEst), while oe.arch primes the PRESAMPLE values
#     e_0^2 = h_0 = mean(e^2) and computes h_1 by the recursion. To compare the same
#     likelihood, one presample observation y_0 = mu + sqrt(h0) is prepended: the
#     filter's mean of squared residuals over the n + 1 values is then exactly h0, and
#     its h_1, e_0^2 are the presample values of oe.arch; the per-observation log
#     likelihood of the prepended observation is dropped.
#   * mle: the same rugarch filter likelihood (prepended series, first observation
#     dropped) maximized by stats::optim (BFGS, reltol 1e-15) over omega, alpha1, beta1,
#     skew, shape; standard errors from numDeriv::hessian (Richardson extrapolation).
#   * package_fit: rugarch::ugarchfit on the original series (its own presample
#     convention and solver), recorded for information only.
#
# rugarch_csgarch.json (component GARCH, Engle-Lee; rugarch "csGARCH")
#   * likelihood: rugarch::ugarchfilter at fixed parameters, with the per-observation log
#     likelihoods and the h and q paths. rugarch starts at observation 1 with
#     q_1 = omega / (1 - rho) and h_1 = mean(e^2) + q_1 (C function csgarchfilterC),
#     which cannot be matched by prepending observations; the tests therefore run their
#     explicit NumPy two-state recursion from rugarch's state and require rugarch's path
#     and likelihood, and separately compare the kernel with the same NumPy recursion
#     under oe.arch's presample convention.
#   * package_fit: rugarch::ugarchfit (solver "hybrid") on a simulated series: estimates,
#     standard errors and log likelihood under rugarch's own start-up convention.
#
# rmgarch_adcc.json (asymmetric DCC, Cappiello-Engle-Sheppard; rmgarch model "aDCC")
#   * a simulated bivariate aDCC series with sGARCH(1,1) margins, rmgarch::dccfit (two-step:
#     univariate rugarch fits, then the aDCC parameters with Qbar = cov(z) and
#     Nbar = cov(z 1(z<0)), both centred with n - 1, as normal.dccLLH2 computes them), and
#     the per-period joint log likelihoods, conditional correlations and variances. rmgarch
#     starts the Q recursion from a presample Q_0 = Qbar with presample z = n = (1, ..., 1)
#     (rdcc.cpp dccnormC2 after the rbind of ones in normal.dccLLH2); the tests reproduce
#     this with an explicit NumPy recursion and compare the kernel with the same recursion
#     under oe.mgarch_adcc's conventions.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) .libPaths(c(args[1], .libPaths()))
suppressMessages({
  library(jsonlite)
  library(rugarch)
  library(fGarch)
  library(rmgarch)
})
options(xts_check_TZ = FALSE)

out_dir <- file.path("tests", "fixtures", "garch_extensions")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
seed <- 1612026L

provenance <- function() {
  list(
    generator = "scripts/generate_garch_extension_oracles.R",
    date = as.character(Sys.Date()),
    seed = seed,
    R = R.version.string,
    rugarch = as.character(packageVersion("rugarch")),
    fGarch = as.character(packageVersion("fGarch")),
    rmgarch = as.character(packageVersion("rmgarch")),
    numDeriv = as.character(packageVersion("numDeriv")),
    note = "development-only oracle; R packages are not OpenEconometrics dependencies"
  )
}

write_fixture <- function(name, sections) {
  path <- file.path(out_dir, name)
  writeLines(toJSON(c(list(provenance = provenance()), sections), digits = NA,
                    auto_unbox = TRUE), path)
  cat(sprintf("wrote %s (%.0f KiB)\n", path, file.info(path)$size / 1024))
}

simulate_garch <- function(n, omega, alpha, beta, shocks) {
  burn <- 300
  e <- numeric(n + burn); h <- rep(omega / (1 - alpha - beta), n + burn)
  for (t in 2:(n + burn)) {
    h[t] <- omega + alpha * e[t - 1]^2 + beta * h[t - 1]
    e[t] <- sqrt(h[t]) * shocks[t]
  }
  e[(burn + 1):(burn + n)]
}

sged_spec <- function(mean, fixed = NULL) {
  ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
             mean.model = list(armaOrder = c(0, 0), include.mean = mean),
             distribution.model = "sged", fixed.pars = fixed)
}

# Log likelihood of observations 1..n with the oe.arch presample convention.
sged_loglik <- function(y, mu, pars) {
  h0 <- mean((y - mu)^2)
  fixed <- list(omega = pars[1], alpha1 = pars[2], beta1 = pars[3], skew = pars[4],
                shape = pars[5])
  if (mu != 0) fixed <- c(list(mu = mu), fixed)
  filtered <- ugarchfilter(sged_spec(mu != 0, fixed), c(mu + sqrt(h0), y))
  filtered@filter$log.likelihoods[-1]
}

sged_section <- function() {
  set.seed(seed)
  grid <- seq(-6, 6, length.out = 49)
  density <- lapply(list(c(1.3, 1.35), c(2.5, 0.6), c(0.9, 1.8), c(1.8, 1.0)), function(p) {
    list(shape = p[1], skew = p[2], z = grid,
         fgarch_log_density = dsged(grid, mean = 0, sd = 1, nu = p[1], xi = p[2], log = TRUE),
         rugarch_density = ddist("sged", grid, mu = 0, sigma = 1, skew = p[2], shape = p[1]))
  })
  likelihood <- lapply(list(list(mu = 0.0, pars = c(0.05, 0.08, 0.88, 1.25, 1.4)),
                            list(mu = 0.12, pars = c(0.03, 0.12, 0.84, 0.8, 1.9))),
                       function(case) {
    shocks <- rsged(800, mean = 0, sd = 1, nu = 1.5, xi = 1.3)
    y <- case$mu + 0.1 + simulate_garch(500, 0.05, 0.1, 0.85, shocks)
    ll <- sged_loglik(y, case$mu, case$pars)
    list(mu = case$mu, omega = case$pars[1], alpha = case$pars[2], beta = case$pars[3],
         skew = case$pars[4], shape = case$pars[5], y = y, h0 = mean((y - case$mu)^2),
         loglik = ll)
  })
  shocks <- rsged(1300, mean = 0, sd = 1, nu = 1.4, xi = 1.35)
  y <- simulate_garch(1000, 0.05, 0.1, 0.85, shocks)
  objective <- function(p) {
    if (p[1] <= 0 || p[4] <= 0.05 || p[5] <= 0.2) return(1e10)
    value <- tryCatch(-sum(sged_loglik(y, 0, p)), error = function(e) 1e10)
    if (!is.finite(value)) 1e10 else value
  }
  start <- c(0.05, 0.1, 0.8, 1.2, 1.5)
  opt <- optim(start, objective, method = "BFGS",
               control = list(reltol = 1e-15, maxit = 2000, parscale = c(0.05, 0.1, 0.5, 1, 1)))
  opt <- optim(opt$par, objective, method = "Nelder-Mead",
               control = list(reltol = 1e-16, maxit = 5000))
  opt <- optim(opt$par, objective, method = "BFGS",
               control = list(reltol = 1e-16, maxit = 2000, parscale = c(0.05, 0.1, 0.5, 1, 1)))
  hessian <- numDeriv::hessian(objective, opt$par, method = "Richardson",
                               method.args = list(eps = 1e-4, d = 1e-3, r = 6))
  package <- ugarchfit(sged_spec(FALSE), y, solver = "hybrid")
  list(
    density = density,
    likelihood = likelihood,
    mle = list(y = y, names = c("omega", "alpha1", "beta1", "skew", "shape"),
               params = opt$par, std_errors = sqrt(diag(solve(hessian))),
               loglik = -opt$value, convergence = opt$convergence,
               method = "stats::optim BFGS/Nelder-Mead/BFGS on the rugarch filter likelihood; numDeriv::hessian (Richardson)"),
    package_fit = list(names = names(coef(package)), params = unname(coef(package)),
                       std_errors = unname(package@fit$se.coef),
                       loglik = likelihood(package),
                       presample = "rugarch: h_1 = mean(e^2) over the sample (no prepended observation)")
  )
}

write_fixture("rugarch_sged.json", sged_section())

csgarch_spec <- function(mean, fixed = NULL) {
  ugarchspec(variance.model = list(model = "csGARCH", garchOrder = c(1, 1)),
             mean.model = list(armaOrder = c(0, 0), include.mean = mean),
             distribution.model = "norm", fixed.pars = fixed)
}

simulate_csgarch <- function(n, omega, alpha, beta, rho, phi, shocks) {
  burn <- 500
  q <- omega / (1 - rho); h <- q; e <- 0; out <- numeric(n + burn)
  for (t in 1:(n + burn)) {
    q_next <- omega + rho * q + phi * (e^2 - h)
    h <- q_next + alpha * (e^2 - q) + beta * (h - q)
    q <- q_next
    e <- sqrt(h) * shocks[t]
    out[t] <- e
  }
  out[(burn + 1):(burn + n)]
}

csgarch_section <- function() {
  set.seed(seed + 1L)
  likelihood <- lapply(list(list(mu = 0.0, pars = c(0.02, 0.08, 0.70, 0.98, 0.04)),
                            list(mu = 0.05, pars = c(0.01, 0.12, 0.55, 0.995, 0.03))),
                       function(case) {
    y <- case$mu + simulate_csgarch(600, 0.02, 0.1, 0.6, 0.98, 0.05, rnorm(1100))
    fixed <- list(omega = case$pars[1], alpha1 = case$pars[2], beta1 = case$pars[3],
                  eta11 = case$pars[4], eta21 = case$pars[5])
    if (case$mu != 0) fixed <- c(list(mu = case$mu), fixed)
    filtered <- ugarchfilter(csgarch_spec(case$mu != 0, fixed), y)
    list(mu = case$mu, omega = case$pars[1], alpha = case$pars[2], beta = case$pars[3],
         rho = case$pars[4], phi = case$pars[5], y = y,
         loglik = filtered@filter$log.likelihoods,
         h = as.numeric(sigma(filtered))^2, q = as.numeric(filtered@filter$q),
         start = "q_1 = omega / (1 - rho), h_1 = mean((y - mu)^2) + q_1")
  })
  y <- simulate_csgarch(3000, 0.01, 0.12, 0.60, 0.98, 0.04, rnorm(3500))
  package <- ugarchfit(csgarch_spec(FALSE), y, solver = "hybrid")
  list(likelihood = likelihood,
       package_fit = list(y = y, names = names(coef(package)), params = unname(coef(package)),
                          std_errors = unname(package@fit$se.coef),
                          loglik = likelihood(package),
                          start = "rugarch: q_1 = omega / (1 - rho), h_1 = mean(y^2) + q_1"))
}

write_fixture("rugarch_csgarch.json", csgarch_section())

simulate_adcc <- function(n, omega, alpha, beta, a, b, g, target, nbar) {
  burn <- 300
  d <- length(omega)
  h <- omega / (1 - alpha - beta); e <- rep(0, d); q <- target
  intercept <- (1 - a - b) * target - g * nbar
  out <- matrix(0, n + burn, d)
  for (t in 1:(n + burn)) {
    r <- q / sqrt(outer(diag(q), diag(q)))
    z <- as.numeric(t(chol(r)) %*% rnorm(d))
    e <- sqrt(h) * z
    out[t, ] <- e
    neg <- z * (z < 0)
    q <- intercept + a * outer(z, z) + b * q + g * outer(neg, neg)
    h <- omega + alpha * e^2 + beta * h
  }
  out[(burn + 1):(burn + n), ]
}

adcc_section <- function() {
  set.seed(seed + 2L)
  y <- simulate_adcc(500, c(0.05, 0.08), c(0.08, 0.1), c(0.88, 0.85), 0.04, 0.88, 0.08,
                     matrix(c(1, 0.5, 0.5, 1), 2), matrix(c(0.5, 0.33, 0.33, 0.5), 2))
  colnames(y) <- c("x", "y")
  uspec <- ugarchspec(mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                      variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                      distribution.model = "norm")
  spec <- dccspec(uspec = multispec(replicate(2, uspec)), dccOrder = c(1, 1),
                  model = "aDCC", distribution = "mvnorm")
  fit <- dccfit(spec, data = y)
  list(y = unname(as.matrix(y)), names = names(coef(fit)), params = unname(coef(fit)),
       std_errors = unname(fit@mfit$matcoef[, 2]),
       loglik = likelihood(fit), log_likelihoods = fit@mfit$log.likelihoods,
       Qbar = fit@mfit$Qbar, Nbar = fit@mfit$Nbar,
       correlation_xy = as.numeric(rcor(fit)[1, 2, ]),
       variance_x = as.numeric(rcov(fit)[1, 1, ]), variance_y = as.numeric(rcov(fit)[2, 2, ]),
       convergence = fit@mfit$convergence,
       estimation = "rmgarch two-step: univariate rugarch sGARCH fits, then aDCC given z")
}

write_fixture("rmgarch_adcc.json", adcc_section())
