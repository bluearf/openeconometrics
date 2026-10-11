# R reference values for the ERS point-optimal test (oe.ers_pt).
#
# urca::ur.ers(type = "P-test") on seeded synthetic series. urca 1.3-4 (CRAN,
# GPL) is an independent implementation of Elliott, Rothenberg and Stock
# (1996): GLS sums of squares S(abar) and S(1), the autoregressive spectral
# estimator from Delta y on a constant, y_{t-1} and lags, BIC over 1..lag.max
# lags on unequal samples. In the trend case urca computes S(1) from
# (0, Delta y) demeaned over all T entries, which exceeds the GLS residual sum
# of squares by (y_T - y_1)^2 / (T (T - 1)); the tests apply that identity.
#
# Run (urca and jsonlite installed in <lib>; output is deterministic):
#   R_LIBS=<lib> Rscript --vanilla scripts/generate_unitroot_fourier_oracles.R \
#     tests/fixtures/unitroot_fourier/ers_urca.json

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[1] else "ers_urca.json"
suppressPackageStartupMessages({
  library(urca)
  library(jsonlite)
})
set.seed(20261010)
make <- function(n, kind) {
  e <- rnorm(n)
  switch(kind,
    random_walk = cumsum(e + 0.3 * c(0, head(e, -1))),
    stationary = as.numeric(stats::filter(e, 0.6, method = "recursive")) + 5,
    trending = 0.05 * seq_len(n) + cumsum(e) * 0.4 + 2)
}
cases <- list()
for (kind in c("random_walk", "stationary", "trending")) {
  for (n in c(60L, 150L)) {
    y <- make(n, kind)
    for (model in c("constant", "trend")) {
      for (lag.max in c(0L, 1L, 4L)) {
        r <- ur.ers(y, type = "P-test", model = model, lag.max = lag.max)
        cases[[length(cases) + 1]] <- list(
          kind = kind, n = n, model = model, lag_max = lag.max, y = y,
          teststat = unname(r@teststat), lag = r@lag,
          cval = unname(as.numeric(r@cval[1, ])))
      }
    }
  }
}
meta <- list(script = "scripts/generate_unitroot_fourier_oracles.R",
             urca = as.character(packageVersion("urca")),
             r = paste(R.version$major, R.version$minor, sep = "."),
             seed = 20261010L, generated = as.character(Sys.Date()))
writeLines(toJSON(list(meta = meta, cases = cases), auto_unbox = TRUE, digits = NA), out)
