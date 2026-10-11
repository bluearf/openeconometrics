#!/usr/bin/env Rscript
# External oracle only: no OpenEconometrics code is loaded by this process.
# usage: Rscript generate_mixed_extension_oracles.R <mode> <spec.json> <output.json>
#   mode lmer : lme4/lmerTest/pbkrtest Kenward-Roger and Satterthwaite contrasts
#   mode lme  : nlme::lme random effects with corAR1/corCompSymm/varIdent residuals
#   mode clmm : ordinal::clmm random-intercept cumulative link by adaptive GHQ
#   mode sleepstudy : write lme4's published sleepstudy data (Belenky et al. 2003) as CSV
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 2L && args[1L] == "sleepstudy") {
    write.csv(lme4::sleepstudy, args[2L], row.names = FALSE)
    quit(save = "no")
}
if (length(args) != 3L) stop("usage: Rscript generate_mixed_extension_oracles.R mode spec.json output.json")
mode <- args[1L]
spec <- jsonlite::fromJSON(args[2L], simplifyVector = TRUE, simplifyMatrix = FALSE)
out <- args[3L]
data <- read.csv(spec$data, stringsAsFactors = FALSE)
for (name in spec$factors) data[[name]] <- factor(data[[name]], levels = unique(sort(data[[name]])))
mat <- function(m) unname(lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ])))
result <- list(R = R.version.string)

if (mode == "lmer") {
    suppressPackageStartupMessages({library(lme4); library(lmerTest); library(pbkrtest)})
    result$packages <- c(lme4 = as.character(packageVersion("lme4")),
                         lmerTest = as.character(packageVersion("lmerTest")),
                         pbkrtest = as.character(packageVersion("pbkrtest")))
    control <- lmerControl(optimizer = "bobyqa",
                           optCtrl = list(maxfun = 1e6, rhoend = 1e-12, rhobeg = 0.2),
                           check.conv.grad = "ignore")
    model <- lmer(as.formula(spec$formula), data = data, REML = isTRUE(spec$reml), control = control)
    result$fixef <- as.numeric(fixef(model))
    result$fixef_names <- names(fixef(model))
    result$vcov <- mat(as.matrix(vcov(model)))
    vc <- as.data.frame(VarCorr(model))
    result$varcorr <- list(grp = vc$grp, var1 = ifelse(is.na(vc$var1), "", vc$var1),
                           var2 = ifelse(is.na(vc$var2), "", vc$var2), vcov = vc$vcov)
    result$logLik <- as.numeric(logLik(model))
    if (isTRUE(spec$reml)) result$vcov_adjusted <- mat(as.matrix(vcovAdj(model)))
    sets <- list()
    for (k in seq_along(spec$contrasts$name)) {
        L <- do.call(rbind, spec$contrasts$L[[k]])
        rhs <- as.numeric(spec$contrasts$rhs[[k]])
        rows <- list()
        for (i in seq_len(nrow(L))) {
            entry <- list()
            sat <- contest1D(model, L[i, ], rhs = rhs[i], ddf = "Satterthwaite")
            entry$satterthwaite <- list(estimate = sat$Estimate, se = sat[["Std. Error"]], df = sat$df,
                                        t = sat[["t value"]], p = sat[["Pr(>|t|)"]])
            if (isTRUE(spec$reml)) {
                kr <- contest1D(model, L[i, ], rhs = rhs[i], ddf = "Kenward-Roger")
                entry$kenward_roger <- list(estimate = kr$Estimate, se = kr[["Std. Error"]], df = kr$df,
                                            t = kr[["t value"]], p = kr[["Pr(>|t|)"]])
            }
            rows[[i]] <- entry
        }
        joint <- list()
        satj <- contestMD(model, L, rhs = rhs, ddf = "Satterthwaite")
        joint$satterthwaite <- list(F = satj[["F value"]], ndf = satj$NumDF, ddf = satj$DenDF,
                                    p = satj[["Pr(>F)"]])
        if (isTRUE(spec$reml)) {
            betaH <- if (all(rhs == 0)) 0 else drop(MASS::ginv(L) %*% rhs)
            krj <- KRmodcomp(model, L, betaH = betaH)
            joint$kenward_roger <- list(F = krj$test["Ftest", "stat"], F_unscaled = krj$test["FtestU", "stat"],
                                        ndf = krj$test["Ftest", "ndf"], ddf = krj$test["Ftest", "ddf"],
                                        scale = krj$test["Ftest", "F.scaling"], p = krj$test["Ftest", "p.value"])
        }
        sets[[k]] <- list(name = spec$contrasts$name[[k]], rows = rows, joint = joint)
    }
    result$contrasts <- sets
} else if (mode == "lme") {
    suppressPackageStartupMessages(library(nlme))
    result$packages <- c(nlme = as.character(packageVersion("nlme")))
    # random, correlation and weights are R expressions, e.g. "list(g = pdDiag(~ x))".
    random <- eval(parse(text = spec$random))
    correlation <- eval(parse(text = spec$correlation))
    weights <- eval(parse(text = spec$weights))
    model <- lme(as.formula(spec$fixed), random = random, data = data,
                 correlation = correlation, weights = weights, method = spec$method,
                 control = lmeControl(maxIter = 1000, msMaxIter = 1000, niterEM = 0,
                                      tolerance = 1e-12, msTol = 1e-14, opt = "nlminb",
                                      returnObject = FALSE))
    result$fixef <- as.numeric(fixef(model))
    result$fixef_names <- names(fixef(model))
    result$vcov <- mat(as.matrix(vcov(model)))
    result$logLik <- as.numeric(logLik(model))
    result$sigma <- model$sigma
    re <- as.matrix(getVarCov(model))
    result$random_covariance <- mat(re)
    result$random_names <- colnames(re)
    result$corStruct <- if (is.null(model$modelStruct$corStruct)) NULL else
        as.numeric(coef(model$modelStruct$corStruct, unconstrained = FALSE))
    if (!is.null(model$modelStruct$varStruct)) {
        result$varStruct <- as.list(coef(model$modelStruct$varStruct, unconstrained = FALSE, allCoef = TRUE))
    }
    result$apVar <- if (is.character(model$apVar)) model$apVar else mat(unclass(model$apVar))
    result$apVar_names <- if (is.character(model$apVar)) NULL else rownames(model$apVar)
    result$apVar_meanings <- "nlme apVar: reStruct log sd / atanh-type correlation, corStruct and varStruct unconstrained, then log(sigma)"
} else if (mode == "clmm") {
    suppressPackageStartupMessages(library(ordinal))
    result$packages <- c(ordinal = as.character(packageVersion("ordinal")))
    data[[spec$outcome]] <- factor(data[[spec$outcome]], levels = sort(unique(data[[spec$outcome]])), ordered = TRUE)
    data[[spec$group]] <- factor(data[[spec$group]])
    model <- clmm(as.formula(spec$formula), data = data, link = spec$link, nAGQ = spec$nAGQ,
                  Hess = TRUE, control = clmm.control(method = "ucminf", innerCtrl = "noWarn",
                                                      gradTol = 1e-10, grtol = 1e-10, xtol = 1e-14,
                                                      maxeval = 10000))
    result$coefficients <- as.numeric(coef(model))
    result$coefficient_names <- names(coef(model))
    result$vcov <- mat(as.matrix(vcov(model)))
    result$vcov_names <- rownames(vcov(model))
    result$logLik <- as.numeric(logLik(model))
    result$random_sd <- as.numeric(attr(VarCorr(model)[[1L]], "stddev"))
    result$random_variance <- as.numeric(VarCorr(model)[[1L]])
    result$maxGradient <- max(abs(model$gradient))
} else {
    stop("unknown mode")
}
writeLines(jsonlite::toJSON(result, digits = I(17), auto_unbox = TRUE, null = "null", na = "null"), out)
