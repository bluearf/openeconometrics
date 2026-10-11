#!/usr/bin/env Rscript
# External oracle only. No OpenEcon code is loaded by this process.
# Called by scripts/generate_iv_extension_oracles.py --r with the Lewbel data set
# (generated instruments p_x1_g, p_x2_g already formed as data, ivreg2h's convention).
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: Rscript generate_iv_extension_oracles.R data.csv output.json")
suppressPackageStartupMessages({
    library(AER)
    library(sandwich)
    library(jsonlite)
})
data <- read.csv(args[1L])
fits <- list(
    generated = ivreg(y ~ x1 + x2 + p | x1 + x2 + p_x1_g + p_x2_g, data = data),
    generated_external = ivreg(y ~ x1 + x2 + p | x1 + x2 + zext + p_x1_g + p_x2_g, data = data)
)
out <- list(versions = list(R = as.character(getRversion()),
                            AER = as.character(packageVersion("AER")),
                            sandwich = as.character(packageVersion("sandwich"))))
for (name in names(fits)) {
    fit <- fits[[name]]
    out[[name]] <- list(
        params = unname(coef(fit)),
        hc0 = unname(vcovHC(fit, type = "HC0")),
        # sandwich's vcovCL with type HC0 and cadjust = TRUE: G/(G-1) only
        cluster_hc0 = unname(vcovCL(fit, cluster = ~g, type = "HC0", cadjust = TRUE)),
        # type HC1: G/(G-1) (N-1)/(N-K)
        cluster_hc1 = unname(vcovCL(fit, cluster = ~g, type = "HC1", cadjust = TRUE))
    )
}
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, matrix = "rowmajor"), args[2L])
