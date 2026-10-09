"""Independent four-selection population enumeration and NumPy/SciPy oracles."""
from itertools import combinations, product
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    'independent_four_stage', Path(__file__).resolve().parents[1] / 'scripts/verify_survey_four_stage_oracles.py'
)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)
GATES = reference.CASES
ROLES = reference.ROLES


def declare(frame):
    import openecon as oe
    return oe.survey_four_stage_design(frame, **ROLES)


def fixture():
    rng = np.random.default_rng(20261009)
    rows = []
    for h in range(3):
        for p in range(3):
            for s in range(2 + p % 2):
                for j in range(2 + s % 2):
                    for f in range(2 + j % 2):
                        x, z, noise = rng.normal(size=3)
                        rows.append(dict(h=h,p=p,s=s,j=j,f=f,N=5+h,M=4+p,L=4+s,K=4+j,
                            x=x,z=z,y=1.4+.65*x-.4*z+noise,a=2+.3*x+noise/4,
                            den=3.5+abs(z),den2=2.5+abs(x),binary=(f+p+s+j)%2,
                            count=(0,2,1)[f]+int(p==2),category='a' if (f+s+j)%3 else 'b',
                            domain=int(not(h==2 and p==2) and not(h==1 and p==0 and s==1 and j==1))))
    frame = pd.DataFrame(rows).iloc[rng.permutation(len(rows))].reset_index(drop=True)
    frame.index = np.resize([37,-4,37,2],len(frame))
    return frame


def call(frame, design, kind, *, domain=False, missing='raise', intercept=True, alpha=.1, null=0.):
    import openecon as oe
    spec = reference.specification(kind,intercept)
    kwargs = dict(domain='domain' if domain else None,missing=missing,alpha=alpha,null=null)
    kwargs.update({k:v for k,v in spec.items() if k!='args'})
    if kind in reference.FAMILIES:
        kwargs.update(tolerance=1e-11,max_iter=200)
    return getattr(oe,'survey_four_stage_'+kind)(frame,design,*spec['args'],**kwargs)


def assert_expected(state, expected):
    numeric = reference.numerical
    estimates = state.estimates if hasattr(state,'estimates') else state.coefficients
    numeric(estimates,expected['estimates'],'estimates')
    numeric(state.covariance,expected['covariance'],'full covariance')
    assert list(state.labels)==expected['labels'] and state.df==expected['df']
    for key in ('stage1_covariance','stage2_covariance','stage3_covariance','stage4_covariance'):
        numeric(state.metadata[key],expected[key],key)
    for key,value in expected['primitives'].items():
        numeric(state.metadata[key],value,key)
    numeric(state.design.validation.weights,expected['weights'],'four derived fractions')
    assert state.metadata['sample_positions']==np.flatnonzero(expected['selected']).tolist()
    assert state.metadata['out_of_domain_positions']==np.flatnonzero(~expected['members']).tolist()
    assert state.metadata['outcome_exclusions']==np.flatnonzero(expected['members']&~expected['selected']).tolist()
    table=state.to_frame()
    numeric(table.attrs['covariance_matrix'],expected['covariance'],'displayed full covariance')
    actual=table.astype(object).where(pd.notna(table),None).values.tolist()
    for row,wanted in zip(actual,reference.expected_table(expected)):
        for a,b in zip(row,wanted):
            if b is None:
                assert a is None
            else:
                numeric(a,b,'reference inference')
    assert type(state).model_validate_json(state.model_dump_json()).model_dump(mode='json')==state.model_dump(mode='json')


@pytest.mark.parametrize('kind',GATES)
@pytest.mark.parametrize('sample',['full','domain','missing','both'])
@pytest.mark.parametrize('census_stage',[None,1,2,3,4])
def test_all_eight_independent_full_covariance_and_saved_primitive_replay(kind,sample,census_stage):
    frame=fixture()
    if census_stage:
        pop=('N','M','L','K')[census_stage-1]
        parents=['h','p','s','j'][:census_stage]
        child=['p','s','j','f'][census_stage-1]
        frame[pop]=frame.groupby(parents)[child].transform('nunique')
    domain=sample in ('domain','both')
    missing='drop' if sample in ('missing','both') else 'raise'
    if missing=='drop':
        outcome=reference.specification(kind)['args'][0]
        outcome=outcome[0] if isinstance(outcome,list) else outcome
        frame.iloc[5,frame.columns.get_loc(outcome)]=None
    state=call(frame,declare(frame),kind,domain=domain,missing=missing)
    expected=reference.expected_case(frame,kind,options={'domain':'domain' if domain else None,'missing':missing,'alpha':.1})
    assert_expected(state,expected)
    if census_stage:
        assert np.count_nonzero(state.metadata[f'stage{census_stage}_covariance'])==0
    else:
        assert np.linalg.norm(state.metadata['stage4_covariance'])>1e-8


def population(prefix=()):
    if len(prefix)==4:
        p,s,j,f=prefix
        return np.array([2+3*p+5*s*s+7*j+f*f,3-p+2*s-9*j+(-1)**f],float)
    children=3 if all(v==0 for v in prefix) else 2
    return [population(prefix+(i,)) for i in range(children)]


def leaves(node,prefix=()):
    if isinstance(node,np.ndarray):
        return [(prefix,node)]
    return sum((leaves(child,prefix+(i,)) for i,child in enumerate(node)),[])


def worlds(node,census,prefix=(),counts=()):
    """Enumerate the actual successive selection probabilities, not a covariance formula."""
    if isinstance(node,np.ndarray):
        p,s,j,f=prefix
        N,M,L,K=counts
        return [([dict(h=0,p=p,s=s,j=j,f=f,N=N,M=M,L=L,K=K,y=node[0],a=node[1])],1.)]
    n=len(node) if len(prefix)+1 in census else min(2,len(node))
    samples=list(combinations(range(len(node)),n))
    result=[]
    for selection in samples:
        options=[worlds(node[i],census,prefix+(i,),counts+(len(node),)) for i in selection]
        for choice in product(*options):
            result.append((sum((rows for rows,_ in choice),[]),np.prod([prob for _,prob in choice])/len(samples)))
    return result


@pytest.mark.parametrize('mask',range(16))
def test_exact_four_real_selections_unbiased_ht_full_covariance_all_census_masks(mask):
    import openecon as oe
    tree=population()
    census={stage+1 for stage in range(4) if mask&(1<<stage)}
    sampled=worlds(tree,census)
    assert len(leaves(tree))==31 and len(sampled)<=31
    total=sum((value for _,value in leaves(tree)),np.zeros(2))
    expected_estimate=np.zeros(2)
    reported_cov=np.zeros((2,2))
    randomization_cov=np.zeros((2,2))
    for rows,prob in sampled:
        frame=pd.DataFrame(rows)
        state=oe.survey_four_stage_total(frame,declare(frame),['y','a'])
        expected=reference.expected_case(frame,'total',spec={'args':[['y','a']]},options={'alpha':.05})
        assert_expected(state,expected)
        estimate=np.array(state.estimates)
        expected_estimate+=prob*estimate
        reported_cov+=prob*np.array(state.covariance)
        randomization_cov+=prob*np.outer(estimate-total,estimate-total)
        for stage in census:
            assert np.count_nonzero(state.metadata[f'stage{stage}_covariance'])==0
    np.testing.assert_allclose(sum(prob for _,prob in sampled),1.,atol=1e-14)
    np.testing.assert_allclose(expected_estimate,total,atol=1e-11)
    np.testing.assert_allclose(reported_cov,randomization_cov,rtol=1e-12,atol=1e-10)
    if not census:
        assert len(sampled)==31 and randomization_cov[0,0]>0


@pytest.mark.parametrize('kind',GATES)
def test_exact_old_three_stage_reduction_only_terminal_census_singleton(kind):
    import openecon as oe
    frame=fixture()
    frame['j']=frame['j'].astype(str)+'/'+frame['f'].astype(str)
    frame['f']=0
    frame['K']=1
    frame['L']=frame.groupby(['h','p','s'])['j'].transform('nunique')+3
    new=call(frame,declare(frame),kind)
    roles={k:v for k,v in ROLES.items() if k not in ('fsu','population_fsu')}
    design=oe.survey_three_stage_design(frame,**roles)
    spec=reference.specification(kind)
    opts={'alpha':.1,**{k:v for k,v in spec.items() if k!='args'}}
    if kind in reference.FAMILIES:
        opts.update(tolerance=1e-11,max_iter=200)
    old=getattr(oe,'survey_three_stage_'+kind)(frame,design,*spec['args'],**opts)
    reference.numerical(new.covariance,old.covariance,'terminal census singleton reduction')
    reference.numerical(new.estimates if hasattr(new,'estimates') else new.coefficients,
                        old.estimates if hasattr(old,'estimates') else old.coefficients,'same estimand')
    assert np.count_nonzero(new.metadata['stage4_covariance'])==0


def test_positive_fourth_component_has_all_three_actual_prefix_fractions():
    frame=fixture()
    expected=reference.expected_case(frame,'total')
    weights,strata=reference.geometry(frame)
    values=frame[['y','a']].to_numpy()
    naive=np.zeros((2,2))
    for psus,f1 in strata:
        for ssus,f2 in psus:
            for tsus,f3 in ssus:
                for positions,f4 in tsus:
                    block=(weights[:,None]*values)[positions]
                    centered=block-block.mean(0)
                    naive+=(1-f4)*len(positions)/(len(positions)-1)*(centered.T@centered)
    assert np.linalg.norm(naive-expected['stage4_covariance'])>1
    assert_expected(call(frame,declare(frame),'total'),expected)


def test_complete_example_and_standalone_native_payload_oracle():
    import runpy
    namespace=runpy.run_path(str(Path(__file__).resolve().parents[1]/'docs/examples/survey_four_stage_eight.py'),init_globals={'display':lambda _:None})
    payload={key:namespace[key] for key in ('states','poststates','oracle_inputs')}
    assert reference.verify_payload(json.loads(json.dumps(payload,allow_nan=False)))['case_count']==8
