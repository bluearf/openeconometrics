# Independent R references for the native seasonal adjustment / forecasting routines.
#
# Development-only oracle (not a runtime dependency). Regenerate with
#   Rscript --vanilla scripts/generate_seasonal_oracles.R tests/fixtures/seasonal
# using R >= 4.6 with the packages stats (stl) and forecast (forecast.stl, rwf,
# croston_model) and jsonlite. The script records the R and package versions.
# All input series are generated here from fixed seeds and stored in the fixture.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[1] else "tests/fixtures/seasonal"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
suppressPackageStartupMessages({
  library(stats)
  library(forecast)
  library(jsonlite)
})

digits <- I(17)  # jsonlite: 17 significant digits, exact float64 round trip
emit <- function(x, file) {
  writeLines(toJSON(x, digits = digits, auto_unbox = TRUE, null = "null", na = "null",
                    pretty = FALSE), file.path(out_dir, file))
}

# ---- series ---------------------------------------------------------------------
set.seed(20261010)
t <- 1:125
monthly <- 100 + 0.4 * t + 8 * sin(2 * pi * t / 12) + 3 * cos(4 * pi * t / 12) +
  rnorm(125, sd = 1.5)
monthly_outliers <- monthly
monthly_outliers[c(17, 64, 101)] <- monthly_outliers[c(17, 64, 101)] + c(25, -30, 18)
tq <- 1:46
quarterly <- 50 + 0.8 * tq + c(4, -2, -5, 3)[(tq - 1) %% 4 + 1] * (1 + tq / 60) +
  rnorm(46, sd = 0.8)
quarterly[c(9, 30)] <- quarterly[c(9, 30)] + c(9, -7)
td <- 1:59
daily <- 20 + 0.05 * td + c(1, 2, 0, -1, -3, 4, -3)[(td - 1) %% 7 + 1] + rnorm(59, sd = 0.6)

# ---- STL ------------------------------------------------------------------------
stl_case <- function(name, y, frequency, ...) {
  fit <- stl(ts(y, frequency = frequency), ...)
  ts_ <- fit$time.series
  list(name = name, y = as.numeric(y), period = frequency,
       call = deparse1(match.call()),
       seasonal = as.numeric(ts_[, "seasonal"]), trend = as.numeric(ts_[, "trend"]),
       remainder = as.numeric(ts_[, "remainder"]), weights = as.numeric(fit$weights),
       win = as.list(fit$win), deg = as.list(fit$deg), jump = as.list(fit$jump),
       inner = fit$inner, outer = fit$outer)
}

stl_cases <- list(
  stl_case("monthly_s7", monthly, 12, s.window = 7),
  stl_case("monthly_s13_robust", monthly_outliers, 12, s.window = 13, robust = TRUE),
  stl_case("monthly_periodic", monthly, 12, s.window = "periodic"),
  stl_case("monthly_periodic_robust", monthly_outliers, 12, s.window = "periodic", robust = TRUE),
  stl_case("quarterly_custom", quarterly, 4, s.window = 9, s.degree = 1, t.window = 13,
           l.window = 5, s.jump = 2, t.jump = 3, l.jump = 2, robust = TRUE),
  stl_case("quarterly_inner3_outer2", quarterly, 4, s.window = 7, inner = 3, outer = 2),
  stl_case("daily_wide_window", daily, 7, s.window = 11, t.window = 21, robust = TRUE)
)

# ---- STL forecasts (forecast::forecast.stl with random-walk methods) -----------
fit <- stl(ts(monthly, frequency = 12), s.window = 7)
forecasts <- lapply(c("naive", "rwdrift"), function(method) {
  f <- forecast(fit, method = method, h = 15, level = c(80, 95))
  list(method = method, h = 15, mean = as.numeric(f$mean),
       lower80 = as.numeric(f$lower[, 1]), upper80 = as.numeric(f$upper[, 1]),
       lower95 = as.numeric(f$lower[, 2]), upper95 = as.numeric(f$upper[, 2]))
})

emit(list(
  provenance = list(
    generator = "scripts/generate_seasonal_oracles.R",
    r_version = R.version.string,
    forecast_version = as.character(packageVersion("forecast")),
    seed = 20261010,
    note = "R stats::stl (netlib Fortran) and forecast::forecast.stl; development-only references"),
  stl = stl_cases,
  stl_forecast = list(case = "monthly_s7", forecasts = forecasts)
), "r_stl.json")
cat("wrote", file.path(out_dir, "r_stl.json"), "\n")

# ---- intermittent demand (forecast::croston_model, forecast 9.x) --------------------
set.seed(162)
occurs <- rbinom(90, 1, 0.3)
sizes <- rpois(90, 4) + 1
intermittent <- occurs * sizes
intermittent[90] <- 0          # ends with a zero period
ending <- intermittent[1:77]
ending[77] <- 6                # ends with a demand
croston_case <- function(name, y, alpha, type) {
  fit <- croston_model(y, alpha = alpha, type = type)
  f <- forecast(fit, h = 5)
  list(name = name, y = as.numeric(y), alpha = alpha, type = type,
       fit_demand = as.numeric(fit$fit_demand), fit_interval = as.numeric(fit$fit_interval),
       fitted = as.numeric(fit$fitted), forecast = as.numeric(f$mean))
}
croston_cases <- list(
  croston_case("croston_a01", intermittent, 0.1, "croston"),
  croston_case("sba_a01", intermittent, 0.1, "sba"),
  croston_case("sbj_a02", intermittent, 0.2, "sbj"),
  croston_case("croston_a035", intermittent, 0.35, "croston"),
  croston_case("croston_ends_with_demand", ending, 0.15, "croston")
)
emit(list(
  provenance = list(
    generator = "scripts/generate_seasonal_oracles.R",
    r_version = R.version.string,
    forecast_version = as.character(packageVersion("forecast")),
    seed = 162,
    note = "forecast::croston_model and forecast(); development-only references"),
  croston = croston_cases
), "r_croston.json")
cat("wrote", file.path(out_dir, "r_croston.json"), "\n")
