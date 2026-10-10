# Eight direct weighted binary API gates — 10 October 2026

Preimplementation source: 2921a8ea53dd7c6441cbd5b749d4acaedac864a9. Branch: codex/eight-next-20261010.
Parent: MARKET-177; no active direct weighted-binary implementation found in the live tracker or open PRs.

The existing direct logit/probit APIs reject weights, although the accepted GLM binary likelihood already implements weighted estimation. Extend those public APIs by reusing that likelihood and reporting engine, preserving the existing unweighted resident and Dataset routes. Eight acceptance domains are logit/probit crossed with fweight, aweight, iweight, pweight. This is direct API/inference support, not a new statistical likelihood or a recount of GLM.

Contracts: strictly binary 0/1; explicit missing raise/drop and original physical row positions; zero-weight exclusion; finite positive retained weights; fweight exact replication with N=sum(f), aweight normalized to retained rows, iweight/pweight used as given with N=physical rows. OIM and OPG for f/a/i; robust N/(N-1) and one-column cluster G/(G-1) for all; pweight defaults robust and rejects OIM/OPG. These explicit ML conventions differ from the preserved legacy unweighted cluster CR1 correction. Analytic weights are a declared OpenEconometrics extension, not a Stata logit/probit feature.

Weighted routes are resident CPU float64 only, original rows <=100000, expanded parameters <=64, cumulative 100-iteration dense likelihood work <=5000000000 and the existing workspace budget. Dataset never silently collects or falls back. These are implementation/resource bounds. Reject rank deficiency, separation, invalid weights, single clusters, unsupported options and work overflow.

Each domain requires independent SciPy/NumPy weighted objective/observed-Hessian/full covariance/SE/z/p/CI checks, frequency replication or scale-law checks, sample/category alignment, full JSON restoration, saved prediction/margins/lincom/test and score reconstruction. Retain original unweighted tests. Update registry/editor/capability docs without dropping existing APIs. Run a source-pinned frozen installed Mac example through native Run and Quit/relaunch, retain receipts, and pass genuine current-head/current-base hosted required checks before normal protected merge and exact eight-child tracker readback. Parent remains open for Tweedie/additional links and generalized ordered/discrete/resampling scopes.

Primary formula references: https://www.stata.com/manuals/rlogit.pdf and https://www.stata.com/manuals/rprobit.pdf. These published documents are references, not licensed executable comparison evidence. No vendor parity or public release claim.

