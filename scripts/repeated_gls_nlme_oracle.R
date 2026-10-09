#!/usr/bin/env Rscript
# External oracle only. No OpenEcon code is loaded by this process.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("usage: Rscript repeated_gls_nlme_oracle.R data.csv structure output-directory")
suppressPackageStartupMessages(library(nlme))
data <- read.csv(args[1L], stringsAsFactors = FALSE)
structure <- args[2L]
out <- args[3L]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
data$subject <- factor(data$subject, levels = unique(data$subject))
data$occasion_factor <- factor(data$occasion_code, levels = 1:4)
options(digits = 17)
for (method in c("ML", "REML")) {
    correlation <- switch(structure,
        cs = corCompSymm(value = -0.1, form = ~ 1 | subject),
        ar1 = corAR1(value = -0.3, form = ~ occasion | subject),
        diagonal = NULL,
        unstructured = corSymm(form = ~ occasion_code | subject),
        stop("unknown covariance structure"))
    weights <- if (structure %in% c("diagonal", "unstructured")) varIdent(form = ~ 1 | occasion_factor) else NULL
    fit <- gls(y ~ x + z, data = data, correlation = correlation, weights = weights,
        method = method, control = glsControl(msMaxIter = 2000L, tolerance = 1e-10,
            msTol = 1e-10, apVar = FALSE, opt = "optim", optimMethod = "BFGS"))
    prefix <- file.path(out, tolower(method))
    write.csv(summary(fit)$tTable, paste0(prefix, "-coefficients.csv"), row.names = TRUE)
    write.csv(as.matrix(vcov(fit)), paste0(prefix, "-reported-coefficient-covariance.csv"), row.names = FALSE)
    # Complete fitted residual covariance on the original observation order.
    # corSymm uses global occasion indices: missing occasions select submatrices.
    groups <- split(seq_len(nrow(data)), data$subject)
    correlations <- if (is.null(fit$modelStruct$corStruct)) NULL else corMatrix(fit$modelStruct$corStruct)
    scales <- rep(fit$sigma, nrow(data))
    if (!is.null(fit$modelStruct$varStruct)) scales <- fit$sigma / varWeights(fit$modelStruct$varStruct)
    residual_covariance <- matrix(0, nrow(data), nrow(data))
    for (name in names(groups)) {
        ii <- groups[[name]]
        block <- if (is.null(correlations)) diag(length(ii)) else correlations[[name]]
        residual_covariance[ii, ii] <- block * outer(scales[ii], scales[ii])
    }
    write.csv(residual_covariance, paste0(prefix, "-residual-covariance.csv"), row.names = FALSE)
    first <- groups[[1L]]
    if (length(first) != 4L) stop("oracle's first subject must contain all four occasions")
    write.csv(residual_covariance[first, first], paste0(prefix, "-occasion-covariance.csv"), row.names = FALSE)
    rho <- if (structure %in% c("cs", "ar1")) unname(coef(fit$modelStruct$corStruct, unconstrained = FALSE)) else NA_real_
    write.csv(data.frame(method = method, n = nrow(data), p = length(coef(fit)),
        sigma2 = fit$sigma^2, rho = rho, log_likelihood = as.numeric(logLik(fit)),
        AIC = AIC(fit), BIC = BIC(fit), df_total = attr(logLik(fit), "df")),
        paste0(prefix, "-fit.csv"), row.names = FALSE)
}
writeLines(c(R.version.string, paste("nlme", as.character(packageVersion("nlme")))), file.path(out, "runtime.txt"))
