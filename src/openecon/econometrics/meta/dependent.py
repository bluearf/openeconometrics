"""Explicit sampling-covariance meta models and complete semantic saved state."""

from __future__ import annotations

import copy
import json
import math

import pandas as pd
import torch

from openecon.analysis import _coerce_frame, _json_scalar, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.engines.distributions import chi2_sf
from openecon.resources import plan_workspace

from .common import checksum, critical, level_check, probability
from . import dependent_kernels as kernels

SCHEMA = "openecon.meta.dependent.v1"
SOURCE = "https://wviechtb.github.io/metafor/reference/rma.mv.html"
MAX_STATE_BYTES = 8*1024**2
STATE_KEYS = {"schema", "device", "dtype", "roles", "effect_ids", "study_ids", "studies",
              "sample_positions", "sample_labels", "sample_hash", "terms", "moderators",
              "intercept", "model", "method", "tau2", "x", "y", "s", "beta",
              "fit_covariance", "covariance", "inference", "residual_df", "cluster_adjustment",
              "n_effects", "n_studies", "level", "fit", "solver", "resource_plan", "checksum"}


def _key(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _labels(values, *, unique=False, name="identifier"):
    labels = [_json_scalar(value) for value in values]
    if any(type(value) is bool or (isinstance(value,str) and not value) for value in labels):
        raise AnalysisError("invalid_identifier", f"{name} labels must be nonempty strings or finite numbers, not booleans.")
    tokens = [_key(value) for value in labels]
    shown = [str(value) for value in labels]
    if len(set(tokens)) != len(set(labels)):
        raise AnalysisError("ambiguous_identifier", f"{name} labels contain distinct encodings that compare equal, such as 1/1.0 or signed zero; normalize labels explicitly.")
    if len(set(tokens)) != len(set(shown)):
        raise AnalysisError("ambiguous_identifier", f"{name} labels have ambiguous displayed values.")
    if unique and len(set(tokens)) != len(tokens):
        raise AnalysisError("duplicate_identifier", f"{name} labels must be unique.")
    return labels


def seal_state(state):
    """Return a separate JSON-safe checked state; the checksum is not authentication."""
    result = copy.deepcopy(state)
    result.pop("checksum", None)
    try:
        encoded = json.dumps(result, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode()) > MAX_STATE_BYTES:
            raise AnalysisError("resource_limit", "The complete saved dependent-meta state exceeds 8 MiB.")
        result["checksum"] = checksum(result)
    except (ValueError, TypeError, OverflowError) as exc:
        raise AnalysisError("invalid_result_state", "State must contain complete finite JSON values.") from exc
    return result


def sample_digest(state):
    """Hash the complete canonical numerical sample, labels and declared covariance."""
    return checksum({name:state[name] for name in ("roles","effect_ids","study_ids",
                     "sample_positions","sample_labels","terms","x","y","s")})


def _plan(n, p, *, replay=False):
    work = 12*n**3+12*(n*p*p+p**3) if replay else kernels.work_bound(n,p)
    if work > kernels.MAX_WORK:
        raise AnalysisError("work_budget", "The complete spectral GLS geometry exceeds the 500M structural work bound.")
    resource = plan_workspace("dependent meta saved replay" if replay else "dependent meta spectral GLS", {
        "sampling_covariance_whitening_spectrum_and_inverse": 24*n*n*8,
        "design_response_information_and_scores": 32*(n*p+p*p+n)*8,
        "complete_state_tables_and_profile": 512*(n*n+n*p+1024),
    }).record()
    return resource, work


def _validate_covariance(s, cluster_index):
    if not bool(torch.isfinite(s).all()) or not torch.equal(s,s.T):
        raise AnalysisError("invalid_covariance", "Supply a finite exactly symmetric sampling covariance matrix; no symmetrization is applied.")
    between = cluster_index[:,None] != cluster_index[None,:]
    if bool((s[between] != 0).any()):
        raise AnalysisError("unsupported_dependence", "Cross-study sampling covariance is unsupported; independent study blocks are required.")
    diagonal = s.diagonal()
    if bool(((diagonal<1e-12)|(diagonal>1e12)).any()) or float(diagonal.max()/diagonal.min()) > 1e12:
        raise AnalysisError("invalid_covariance", "Sampling variances must lie in [1e-12,1e12] and span at most 1e12; rescale explicitly.")
    spectrum = torch.linalg.eigvalsh(s/float(diagonal.median()))
    if float(spectrum[0]) <= 0 or float(spectrum[-1]/spectrum[0]) > 1e10:
        raise AnalysisError("invalid_covariance", "The complete sampling covariance must be SPD with condition ratio at most 1e10.")


def cluster_covariance(state, tensors, adjustment="CR1"):
    """Fixed-weight study sandwich with the meta-analysis CR1 G/(G-p) convention."""
    if adjustment not in ("CR0","CR1"):
        raise AnalysisError("invalid_spec", "Use CR0 or CR1 study-cluster covariance.")
    g, p = state["n_studies"], len(state["terms"])
    if g <= p:
        raise AnalysisError("insufficient_studies", "Study-cluster inference needs G>p.")
    x,y,s,index = (tensors[name] for name in ("x","y","s","cluster_index"))
    replay = kernels.at_variance(x,y,s,index,state["model"],state["intercept"],state["tau2"])
    observation_score = x*(replay["inverse_v"]@replay["residual"])[:,None]
    scores = torch.zeros((g,p),dtype=torch.float64,device="cpu")
    scores.index_add_(0,index,observation_score)
    bread = tensors["fit_covariance"]
    covariance = bread@(scores.T@scores)@bread
    if adjustment == "CR1":
        covariance *= g/(g-p)
    covariance = (covariance+covariance.T)/2
    spectrum = torch.linalg.eigvalsh(covariance)
    if not bool(torch.isfinite(covariance).all()) or float(spectrum[0]) <= float(spectrum[-1])*1e-12:
        raise AnalysisError("degenerate_inference", "Study scores do not identify a full-rank cluster covariance; no ridge or pseudoinverse is used.")
    return covariance


def _same(a,b):
    return bool(torch.allclose(a,b,rtol=2e-8,atol=2e-11*max(1e-12,float(b.abs().max()))))


@resident_cpu
@torch.no_grad()
def load_state(result):
    """Validate complete scientific state and replay GLS at saved tau; no optimizer.

    Returns independent (state,tensors), with raw x/y/s/cluster_index/beta and
    model/current covariance. A resealed but semantically incoherent state is
    rejected. This integrity contract is not cryptographic authentication.
    """
    original = result.attrs.get("prediction_state") if isinstance(result,TableSet) else result
    try:
        encoded = json.dumps(original,allow_nan=False,separators=(",",":"))
        if len(encoded.encode()) > MAX_STATE_BYTES:
            raise ValueError("oversized state")
        state = json.loads(encoded)
        if not isinstance(state,dict) or set(state) != STATE_KEYS:
            raise ValueError("schema fields")
        digest = state.pop("checksum")
        if checksum(state) != digest or state["schema"] != SCHEMA or state["device"] != "cpu" or state["dtype"] != "float64":
            raise ValueError("state checksum or domain")
        state["checksum"] = digest
        n,g = state["n_effects"],state["n_studies"]
        names,intercept = state["moderators"],state["intercept"]
        if type(n) is not int or type(g) is not int or not 2<=n<=256 or not 2<=g<=min(64,n):
            raise ValueError("sample dimensions")
        if type(intercept) is not bool or not isinstance(names,list) or any(not isinstance(name,str) or not name for name in names):
            raise ValueError("moderator roles")
        terms = (["Intercept"] if intercept else [])+names
        p = len(terms)
        if not 1<=p<=8 or n<=p or len(set(terms)) != p or state["terms"] != terms:
            raise ValueError("term dimensions")
        roles = state["roles"]
        if not isinstance(roles,dict) or set(roles) != {"study","effect","yi"} or any(not isinstance(v,str) or not v for v in roles.values()) or len(set([*roles.values(),*names])) != 3+len(names):
            raise ValueError("data roles")
        if state["model"] not in ("common","effect","study") or (state["method"] != "common" if state["model"]=="common" else state["method"] not in ("ML","REML")):
            raise ValueError("likelihood domain")
        if type(state["tau2"]) not in (int,float) or not math.isfinite(state["tau2"]) or state["tau2"]<0 or (state["model"]=="common" and state["tau2"] != 0):
            raise ValueError("variance component")
        level_check(state["level"])
        effects = _labels(state["effect_ids"],unique=True,name="effect")
        studies_per_row = _labels(state["study_ids"],name="study")
        studies = list(dict.fromkeys(_key(value) for value in studies_per_row))
        if len(effects)!=n or len(studies_per_row)!=n or len(studies)!=g or [_key(v) for v in state["studies"]] != studies:
            raise ValueError("effect/study alignment")
        positions,labels = state["sample_positions"],state["sample_labels"]
        if positions != list(range(n)) or any(type(v) is not int for v in positions) or len(labels)!=n:
            raise ValueError("physical row alignment")
        _labels(labels,name="original row")
        if not isinstance(state["sample_hash"],str) or len(state["sample_hash"])!=64 or any(v not in "0123456789abcdef" for v in state["sample_hash"]):
            raise ValueError("sample digest")
        if sample_digest(state) != state["sample_hash"]:
            raise ValueError("numerical sample digest")
        _plan(n,p,replay=True)
        tensors = {name:torch.tensor(state[name],dtype=torch.float64,device="cpu") for name in ("x","y","s","beta","fit_covariance","covariance")}
        shapes = {"x":(n,p),"y":(n,),"s":(n,n),"beta":(p,),"fit_covariance":(p,p),"covariance":(p,p)}
        if any(value.shape != shapes[name] or not bool(torch.isfinite(value).all()) for name,value in tensors.items()):
            raise ValueError("scientific array geometry")
        # Boolean numeric inputs are not accepted even after a caller reseals.
        def numeric_array(value):
            return all(numeric_array(v) for v in value) if isinstance(value,list) else type(value) in (int,float)
        if any(not numeric_array(state[name]) for name in tensors):
            raise ValueError("boolean scientific values")
        if float(tensors["x"].abs().max())>1e8 or float(tensors["y"].abs().max())>1e8:
            raise ValueError("numeric domain")
        if intercept and not bool((tensors["x"][:,0]==1).all()):
            raise ValueError("intercept column")
        mapping = {key:j for j,key in enumerate(studies)}
        tensors["cluster_index"] = torch.tensor([mapping[_key(value)] for value in studies_per_row],dtype=torch.int64,device="cpu")
        _validate_covariance(tensors["s"],tensors["cluster_index"])
        replay = kernels.at_variance(tensors["x"],tensors["y"],tensors["s"],tensors["cluster_index"],state["model"],intercept,state["tau2"])
        if not _same(tensors["beta"],replay["beta"]) or not _same(tensors["fit_covariance"],replay["covariance"]):
            raise ValueError("saved GLS coefficients/covariance")
        fit = state["fit"]
        if not isinstance(fit,dict) or set(fit)!={"rss","q0","loglik_ml","loglik_reml","variance_scale","work_bound","max_work"}:
            raise ValueError("fit receipt")
        for name in ("rss","loglik_ml","loglik_reml","variance_scale"):
            if type(fit[name]) not in (int,float) or not math.isclose(fit[name],replay[name],rel_tol=2e-8,abs_tol=2e-9):
                raise ValueError("saved likelihood receipt")
        if type(fit["q0"]) not in (int,float) or not math.isfinite(fit["q0"]) or fit["q0"]<0 or fit["work_bound"] != kernels.work_bound(n,p) or fit["max_work"] != kernels.MAX_WORK:
            raise ValueError("fit work/null receipt")
        zero = kernels.at_variance(tensors["x"],tensors["y"],tensors["s"],tensors["cluster_index"],state["model"],intercept,0.0)
        if not math.isclose(fit["q0"],zero["rss"],rel_tol=2e-8,abs_tol=2e-9):
            raise ValueError("null covariance Q")
        solver = state["solver"]
        if not isinstance(solver,dict) or solver.get("max_evaluations") != kernels.MAX_EVALUATIONS or type(solver.get("evaluations")) is not int or not 1<=solver["evaluations"]<=kernels.MAX_EVALUATIONS:
            raise ValueError("solver receipt")
        if solver.get("boundary") is not (state["tau2"]==0):
            raise ValueError("variance boundary")
        if state["model"] != "common":
            normalized_tau = state["tau2"]/replay["variance_scale"]
            score = replay["score_reml" if state["method"]=="REML" else "score_ml"]
            residual = score*(1+normalized_tau)/n
            if (state["tau2"]==0 and residual>1e-7) or (state["tau2"]>0 and abs(residual)>1e-7):
                raise ValueError("saved variance score stationarity")
            if type(solver.get("score_residual")) not in (int,float) or not math.isclose(solver["score_residual"],residual,rel_tol=1e-6,abs_tol=1e-9):
                raise ValueError("saved variance score receipt")
        if state["inference"] == "z":
            if state["residual_df"] is not None or state["cluster_adjustment"] is not None or not _same(tensors["covariance"],tensors["fit_covariance"]):
                raise ValueError("model covariance inference")
        elif state["inference"] == "cluster":
            if type(state["residual_df"]) is not int or state["residual_df"] != g-p or state["cluster_adjustment"] not in ("CR0","CR1"):
                raise ValueError("study-cluster degrees of freedom")
            expected = cluster_covariance(state,tensors,state["cluster_adjustment"])
            if not _same(tensors["covariance"],expected):
                raise ValueError("saved study score covariance")
        else:
            raise ValueError("inference domain")
        for name in ("fit_covariance","covariance"):
            covariance = tensors[name]
            if not torch.allclose(covariance,covariance.T,rtol=1e-12,atol=0) or int(torch.linalg.cholesky_ex(covariance)[1])!=0:
                raise ValueError("positive-definite coefficient covariance")
        if isinstance(result,TableSet):
            if list(result["coefficients"].term) != terms or not _same(torch.tensor(result["coefficients"].estimate.tolist(),dtype=torch.float64),tensors["beta"]) or not _same(torch.tensor(result["covariance"].to_numpy(),dtype=torch.float64),tensors["covariance"]):
                raise ValueError("result tables disagree with complete state")
    except (ValueError,TypeError,KeyError,AttributeError,RuntimeError,OverflowError,AnalysisError) as exc:
        raise AnalysisError("invalid_result_state", "Use complete, coherent dependent-meta saved state with exact sample/covariance alignment and full inference.") from exc
    return state,tensors


@resident_cpu
@torch.no_grad()
def make_result(state, *, procedure="meta_dependent", title=None):
    """Build canonical tables from an admitted state, without variance refitting."""
    state,tensors = load_state(state)
    beta,covariance = tensors["beta"],tensors["covariance"]
    level,inference,df = state["level"],state["inference"],state["residual_df"]
    se = covariance.diagonal().sqrt()
    statistics = beta/se
    c = critical(level,inference,df)
    terms = state["terms"]
    coefficients = table({"term":terms,"estimate":beta.tolist(),"std_error":se.tolist(),
                          "statistic":statistics.tolist(),"p_value":[probability(float(value),inference,df) for value in statistics],
                          "df":[df]*len(terms),"ci_low":(beta-c*se).tolist(),"ci_high":(beta+c*se).tolist()})
    fitted = tensors["x"]@beta
    effects = table({"effect":state["effect_ids"],"study":state["study_ids"],
                     "position":state["sample_positions"],"yi":state["y"],
                     "sampling_variance":tensors["s"].diagonal().tolist(),
                     "fitted":fitted.tolist(),"residual":(tensors["y"]-fitted).tolist()})
    tests = table([{"test":"Known-sampling-covariance residual Q","statistic":state["fit"]["q0"],
                    "distribution":"chi2","df":state["n_effects"]-len(terms),
                    "p_value":chi2_sf(state["fit"]["q0"],state["n_effects"]-len(terms))}])
    return TableSet({"coefficients":coefficients,
                     "covariance":table(covariance.tolist(),columns=terms,index=terms),
                     "effects":effects,
                     "fit":table([{**{name:state["fit"][name] for name in ("rss","q0","loglik_ml","loglik_reml","variance_scale","work_bound","max_work")},
                                   "tau2":state["tau2"],"model":state["model"],"method":state["method"],"boundary":state["tau2"]==0}]),
                     "tests":tests},
                    title=title or "Dependent-effect meta-analysis",procedure=procedure,
                    prediction_state=state,model=state["model"],method=state["method"],inference=inference,
                    level=level,n_effects=state["n_effects"],n_studies=state["n_studies"],
                    residual_df=df,source=SOURCE,variance_solver=state["solver"],
                    notes=["Known sampling covariance and independent study blocks are supplied assumptions.",
                           "Random-effect GLS uncertainty plugs in the estimated variance; it excludes variance-estimation uncertainty.",
                           "The finite profile grid is not a global optimizer proof for arbitrary pathological data.",
                           "Cluster CR0/CR1 t/F inference is approximate; CR2/Satterthwaite is unavailable."])


@resident_cpu
@torch.no_grad()
def meta_dependent(*, data, covariance, study, effect="effect", yi="yi", moderators=None,
                   intercept=True, model="common", method="REML", level=.95, device="cpu", weights=None):
    """Fit bounded explicit-covariance GLS/common, effect- or study-random meta models.

    covariance is a DataFrame with exactly the effect labels on both axes;
    axis order may differ and is aligned explicitly. Study blocks must be
    independent and exactly symmetric SPD; no covariance symmetrization or
    row deletion. All rows/roles complete, numeric moderators explicit.
    <=256 effects, <=64 studies, <=8 coefficients. Model 'effect' adds tau²I;
    'study' adds tau²ZZ'; both offer ML/REML, zero-variance boundaries and full
    plugged-in GLS covariance. No sampling covariance estimation, arbitrary
    weights, categorical encoding, Dataset or accelerator route is implied.
    """
    if device != "cpu" or type(device) is not str:
        raise AnalysisError("unsupported_device", "Dependent meta-analysis uses explicit CPU float64 only.")
    if weights is not None:
        raise AnalysisError("unsupported_weights", "Supply sampling covariance; observation weights are unsupported.")
    if isinstance(data,Dataset):
        raise AnalysisError("unsupported_dataset", "Supply an explicitly resident complete table.")
    if model not in ("common","effect","study") or method not in ("ML","REML","common") or (model!="common" and method=="common"):
        raise AnalysisError("invalid_spec", "Use common/effect/study models and ML/REML for random models.")
    if type(intercept) is not bool:
        raise AnalysisError("invalid_spec", "intercept must be a boolean.")
    moderators = [] if moderators is None else moderators
    if not isinstance(moderators,list) or any(not isinstance(name,str) or not name for name in moderators):
        raise AnalysisError("invalid_spec", "moderators must be an explicit list of numeric column names.")
    roles = {"study":study,"effect":effect,"yi":yi}
    if any(not isinstance(name,str) or not name for name in roles.values()) or len(set([*roles.values(),*moderators])) != 3+len(moderators):
        raise AnalysisError("invalid_spec", "Study, effect, response and moderator roles must be distinct nonempty names.")
    terms = (["Intercept"] if intercept else [])+moderators
    p = len(terms)
    if not 1<=p<=8 or len(set(terms))!=p:
        raise AnalysisError("resource_limit", "Declare 1..8 distinct numeric coefficients including any intercept.")
    level = level_check(level)
    raw = _coerce_frame(data)
    n = len(raw)
    if not 2<=n<=256 or n<=p or len(raw.columns)>128:
        raise AnalysisError("resource_limit", "This resident route needs 2..256 effects, n>p and at most 128 input columns.")
    selected = [*roles.values(),*moderators]
    if any(list(raw.columns).count(name)!=1 for name in selected):
        raise AnalysisError("invalid_columns", "Each selected role must exist exactly once.")
    if not isinstance(covariance,pd.DataFrame) or covariance.shape != (n,n):
        raise AnalysisError("invalid_covariance", "Supply a labelled n-by-n covariance DataFrame; bare matrices are ambiguous.")
    if covariance.index.has_duplicates or covariance.columns.has_duplicates:
        raise AnalysisError("covariance_alignment", "Both covariance axes must contain unique labels under table label equality.")
    frame = raw.loc[:,selected].copy()
    if bool(frame.isna().any().any()):
        raise AnalysisError("missing_values", "All selected effects/studies must be complete; no rows are dropped.")
    ids = _labels(frame[effect],unique=True,name="effect")
    study_ids = _labels(frame[study],name="study")
    study_tokens = list(dict.fromkeys(_key(value) for value in study_ids))
    g = len(study_tokens)
    if not 2<=g<=64:
        raise AnalysisError("resource_limit", "Declare 2..64 independent study blocks.")
    row_labels = _labels(covariance.index,unique=True,name="covariance row")
    col_labels = _labels(covariance.columns,unique=True,name="covariance column")
    if set(map(_key,row_labels)) != set(map(_key,ids)) or set(map(_key,col_labels)) != set(map(_key,ids)):
        raise AnalysisError("covariance_alignment", "Covariance rows and columns must contain exactly the effect labels.")
    resource,work = _plan(n,p)
    for name in [yi,*moderators]:
        if pd.api.types.is_bool_dtype(frame[name].dtype):
            raise AnalysisError("non_numeric_column", "Boolean responses/moderators are unsupported.")
    values = {name:_numeric(frame[name],name).clone() for name in [yi,*moderators]}
    if any(float(value.abs().max())>1e8 for value in values.values()):
        raise AnalysisError("numeric_domain", "Effects/moderators must lie in +/-1e8; rescale explicitly.")
    parts = [torch.ones(n,dtype=torch.float64,device="cpu")] if intercept else []
    x = torch.stack([*parts,*[values[name] for name in moderators]],1)
    aligned = covariance.loc[ids,ids]
    if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype) for dtype in aligned.dtypes):
        raise AnalysisError("invalid_covariance", "Covariance entries must be finite real quantitative numbers.")
    s = torch.tensor(aligned.to_numpy(dtype="float64"),dtype=torch.float64,device="cpu")
    mapping = {token:j for j,token in enumerate(study_tokens)}
    index = torch.tensor([mapping[_key(value)] for value in study_ids],dtype=torch.int64,device="cpu")
    _validate_covariance(s,index)
    method = "common" if model=="common" else method
    fitted = kernels.fit(x,values[yi],s,index,model,method,intercept)
    fit_receipt = {name:fitted[name] for name in ("rss","q0","loglik_ml","loglik_reml","variance_scale")}
    fit_receipt.update(work_bound=work,max_work=kernels.MAX_WORK)
    state = {"schema":SCHEMA,"device":"cpu","dtype":"float64","roles":roles,
                        "effect_ids":ids,"study_ids":study_ids,"studies":[json.loads(token) for token in study_tokens],
                        "sample_positions":list(range(n)),"sample_labels":_labels(frame.index,name="original row"),
                        "sample_hash":None,"terms":terms,"moderators":moderators.copy(),
                        "intercept":intercept,"model":model,"method":method,"tau2":fitted["tau2"],
                        "x":x.tolist(),"y":values[yi].tolist(),"s":s.tolist(),"beta":fitted["beta"].tolist(),
                        "fit_covariance":fitted["covariance"].tolist(),"covariance":fitted["covariance"].tolist(),
                        "inference":"z","residual_df":None,"cluster_adjustment":None,
                        "n_effects":n,"n_studies":g,"level":level,"fit":fit_receipt,
                        "solver":fitted["solver"],"resource_plan":resource}
    state["sample_hash"] = sample_digest(state)
    return make_result(seal_state(state))
