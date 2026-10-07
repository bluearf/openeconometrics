/** Development-only published-source reference, executed in actual R/WebAssembly.
 * Setup: npm install --prefix artifacts/mgarch-reference webr@0.6.0
 * Run with --fetch to retrieve the source files at the exact pinned commit.
 * No OpenEconometrics numerical implementation is imported or used here.
 * Supplied stationary test parameters are not an empirical fit from the paper.
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve } from 'node:path';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const work = resolve(root, 'artifacts/mgarch-reference');
const output = resolve(root, 'docs/evidence/market-131-mgarch/published-reference.json');
const pin = '01b20c676192a99b16785ed5a02b1fd55038436c';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
if (process.argv.includes('--fetch')) {
  await mkdir(work,{recursive:true});
  for (const path of ['R/bekk_forecast.R','src/BekkFunctions.cpp']) {
    const url=`https://raw.githubusercontent.com/cran/BEKKs/${pin}/${path}`;
    const response=await fetch(url);
    if (!response.ok) throw new Error(`Source download failed: ${response.status} ${url}`);
    await writeFile(resolve(work,path.split('/').at(-1)),Buffer.from(await response.arrayBuffer()));
  }
}
const source = await readFile(resolve(work, 'bekk_forecast.R'));
const cpp = await readFile(resolve(work, 'BekkFunctions.cpp'));
if (hash(source)!=='8c7583fb61bf28b5384100443c6977af14a5cb7c563aaa4228aaeb0f48a92f50' ||
    hash(cpp)!=='17c8c14774dbfdee7cdc371dd90af66c74a78de1315bfad2e68bbcf9d81b3edb') {
  throw new Error('Source checksum mismatch against the verified pinned BEKKs commit');
}
const code = source.toString('utf8');
const begin = code.indexOf('  x <- object', code.indexOf('predict.bekk <- function'));
const end = code.indexOf('\n  sigma_t <- matrix', begin);
if (begin < 0 || end < begin) throw new Error('Published predict.bekk block not found');
const literalForecastBlock = code.slice(begin, end);
const { WebR } = await import(pathToFileURL(resolve(work, 'node_modules/webr/dist/webr.mjs')));
const r = new WebR();
await r.init();
try {
  const setup = `
returns <- matrix(c(.4,-.6,-.2,.8,.9,.2,-1.1,-.3,.5,.7,-.7,.4,.2,-.8,1.0,.3),ncol=2,byrow=TRUE)
C0 <- matrix(c(.25,0,.06,.22),2,2,byrow=TRUE)
A <- matrix(c(.15,.04,-.03,.12),2,2,byrow=TRUE)
G <- matrix(c(.65,.08,-.04,.60),2,2,byrow=TRUE)
H0 <- t(returns)%*%returns/nrow(returns)
H_path <- list(H0)
# Evaluate the paper's BEKK(1,1,1) recurrence. C++ sigma_bekk uses
# CC=t(C)%*%C with C=t(C0), the package's orientation convention.
for (i in 2:nrow(returns)) {
  shock <- matrix(returns[i-1,],nrow=1)
  H_path[[i]] <- C0%*%t(C0)+t(A)%*%t(shock)%*%shock%*%A+t(G)%*%H_path[[i-1]]%*%G
}
eigen_value_decomposition <- function(H) {
  eig <- eigen(H,symmetric=TRUE)
  eig$vectors%*%diag(sqrt(eig$values))%*%t(eig$vectors)
}
published_forecast <- function(object,n.ahead) {
${literalForecastBlock}
  H_t[-1]
}
H_t <- do.call(rbind,lapply(H_path,as.vector))
obj <- list(data=returns,C0=C0,A=A,G=G,H_t=H_t)
BEKK_forecast <- published_forecast(obj,5)
omega <- c(.03,.04)
alpha <- c(.08,.05)
beta <- c(.84,.88)
target <- matrix(c(1,.5,.5,1),2,2,byrow=TRUE)
h0 <- c(.45,.65)
dcc_a <- .01
dcc_b <- .98
run_correlation_model <- function(dynamic) {
  h <- h0; Q <- target
  path <- list(); qpath <- list(); variances <- list()
  for (i in 1:nrow(returns)) {
    R <- cov2cor(Q)
    path[[i]] <- diag(sqrt(h))%*%R%*%diag(sqrt(h))
    qpath[[i]] <- Q; variances[[i]] <- h
    if (i<nrow(returns)) {
      z <- returns[i,]/sqrt(h)
      if (dynamic) Q <- (1-dcc_a-dcc_b)*target+dcc_a*(z%o%z)+dcc_b*Q
      h <- omega+alpha*returns[i,]^2+beta*h
    }
  }
  origin_h <- h; origin_Q <- Q
  forecasts <- list(); qforecasts <- list()
  for (step in 1:5) {
    if (step==1) {
      z <- returns[nrow(returns),]/sqrt(h)
      if (dynamic) Q <- (1-dcc_a-dcc_b)*target+dcc_a*(z%o%z)+dcc_b*Q
      h <- omega+alpha*returns[nrow(returns),]^2+beta*h
    } else {
      # Engle--Sheppard section 7: explicitly approximate Q-forward.
      if (dynamic) Q <- (1-dcc_a-dcc_b)*target+(dcc_a+dcc_b)*Q
      h <- omega+(alpha+beta)*h
    }
    forecasts[[step]] <- diag(sqrt(h))%*%cov2cor(Q)%*%diag(sqrt(h))
    qforecasts[[step]] <- Q
  }
  list(path=path,Q=qpath,variances=variances,forecasts=forecasts,
       forecast_Q=qforecasts,last_h=origin_h,last_Q=origin_Q)
}
CCC <- run_correlation_model(FALSE)
DCC <- run_correlation_model(TRUE)
gaussian_loglik <- function(path) {
  value <- 0
  for (i in seq_along(path)) {
    e <- matrix(returns[i,],ncol=1)
    value <- value-.5*(2*log(2*pi)+log(det(path[[i]]))+t(e)%*%solve(path[[i]],e))
  }
  as.numeric(value)
}
`;
  await r.evalRVoid(setup);
  const matrix = async expression => {
    const vals = await r.evalRRaw(`as.numeric(t(${expression}))`, 'number[]');
    return [vals.slice(0,2), vals.slice(2,4)];
  };
  const matrices = async expression => {
    const vals = await r.evalRRaw(`unlist(lapply(${expression},function(x)as.numeric(t(x))))`, 'number[]');
    const out = [];
    for (let i=0;i<vals.length;i+=4) out.push([vals.slice(i,i+2),vals.slice(i+2,i+4)]);
    return out;
  };
  const bekk = {
    parameters:{C:await matrix('C0'),A:await matrix('A'),B:await matrix('G')},
    initializer:await matrix('H0'),
    covariance_path:await matrices('H_path'),
    correlation_path:await matrices('lapply(H_path,cov2cor)'),
    covariance_forecast:await matrices('BEKK_forecast'),
    correlation_forecast:await matrices('lapply(BEKK_forecast,cov2cor)'),
    gaussian_log_likelihood:await r.evalRNumber('gaussian_loglik(H_path)'),
    parameter_domain_spectral_norm_bound:await r.evalRNumber('max(svd(A)$d)^2+max(svd(G)$d)^2'),
    forecast_method:'literal published predict.bekk first forecast loop; symmetric eigen square root, exact BEKK expectation',
  };
  const corrModel = async model => ({
    parameters:{omega:[.03,.04],alpha:[.08,.05],beta:[.84,.88],target:[[1,.5],[.5,1]],
      ...(model==='DCC'?{dcc_alpha:.01,dcc_beta:.98}: {})},
    initializer_variances:[.45,.65],
    covariance_path:await matrices(`${model}$path`),
    correlation_path:await matrices(`lapply(${model}$path,cov2cor)`),
    Q_path:await matrices(`${model}$Q`),
    last_variances:await r.evalRRaw(`${model}$last_h`,'number[]'),
    covariance_forecast:await matrices(`${model}$forecasts`),
    correlation_forecast:await matrices(`lapply(${model}$forecasts,cov2cor)`),
    Q_forecast:await matrices(`${model}$forecast_Q`),
    gaussian_log_likelihood:await r.evalRNumber(`gaussian_loglik(${model}$path)`),
    forecast_method:model==='DCC'?'published section-7 Q-forward approximation after exact one-step Q; not exact nonlinear correlation expectation':'exact marginal variance forecasts; full covariance beyond one-step uses fixed-correlation plug-in scaling',
  });
  const report = {
    schema_version:1,
    reference_engine:await r.evalRString('R.version.string'),
    reference_runtime:'WebR 0.6.0, development-only actual R/WebAssembly execution',
    primary_sources:[
      {title:'Fuelle, Lange, Hafner, Herwartz (2024), BEKKs',url:'https://www.jstatsoft.org/article/view/v111i04',formula:'section 2.1 BEKK(1,1,1) and Gaussian likelihood',
       pdf_sha256:'005c3d05286db1e0ca4d220edcf855617884f7609f029abab0e45b9d96263ce7'},
      {title:'BEKKs author implementation',url:`https://github.com/cran/BEKKs/tree/${pin}`,commit:pin,
       files:[{path:'R/bekk_forecast.R',sha256:hash(source),executed_block_sha256:hash(literalForecastBlock)},
              {path:'src/BekkFunctions.cpp',sha256:hash(cpp),reference_blocks:'sigma_bekk and eigen_value_decomposition'}]},
      {title:'Engle and Sheppard (2001), Dynamic Conditional Correlation',url:'https://pages.stern.nyu.edu/~rengle/Dcc-Sheppard.pdf',formula:'DCC(1,1) recursion; section 7 Q-forward forecast approximation',
       pdf_sha256:'4b10041b7244354bf3bdffb590ff4d4010dbad2ef97cafd4dd1bf8d46a85afe0'},
    ],
    literal_bekk_forecast_block:literalForecastBlock,
    evaluated_R_setup_sha256:hash(setup),
    returns:[[.4,-.6],[-.2,.8],[.9,.2],[-1.1,-.3],[.5,.7],[-.7,.4],[.2,-.8],[1,.3]],
    mean:[0,0], horizons:5, bekk, ccc:await corrModel('CCC'), dcc:await corrModel('DCC'),
    limitations:['Supplied deterministic stationary parameters/returns, not estimated empirical coefficients from a published table.',
      'Actual source-extracted published R forecast loop executed; C++ covariance-path recurrence independently evaluated in R, not compiled BEKKs Rcpp execution.',
      'CCC/DCC numeric formulas independently evaluated in R from the primary paper, not execution of an external full package.',
      'Tests recursion, full matrices, correlation normalization, likelihood and multi-step forecast conventions; does not establish external coefficient/Hessian/OPG parity or installed-app/release validation.'],
  };
  await mkdir(dirname(output),{recursive:true});
  await writeFile(output,JSON.stringify(report,null,2)+'\n');
  process.stdout.write(JSON.stringify({path:output,reference_engine:report.reference_engine,
    literal_published_R_block_executed:true,BEKK_first_forecast:bekk.covariance_forecast[0],
    models:['BEKK','CCC','DCC'],horizons:5})+'\n');
} finally {
  await r.close();
}
