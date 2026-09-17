from __future__ import annotations
import argparse, json, math, pickle, hashlib
from pathlib import Path
import numpy as np

PURE=[
 "PURE_HOME_POSITION_ERROR",
 "PURE_SENSOR_ZERO_OFFSET",
 "PURE_ACTUATOR_GAIN_ERROR",
 "PURE_ACTUATOR_ADDITIVE_BIAS"
]
NONPURE=[
 "COMPOSITE_HOME_PLUS_GAIN",
 "COMPOSITE_SENSOR_PLUS_BIAS",
 "NONPURE_CART_MASS_SHIFT"
]
STATES=PURE+NONPURE
SID={s:i for i,s in enumerate(STATES)}
FAMILY={
 "PURE_HOME_POSITION_ERROR":"STATIC",
 "PURE_SENSOR_ZERO_OFFSET":"STATIC",
 "PURE_ACTUATOR_GAIN_ERROR":"DYNAMIC",
 "PURE_ACTUATOR_ADDITIVE_BIAS":"DYNAMIC",
 "COMPOSITE_HOME_PLUS_GAIN":"NONPURE",
 "COMPOSITE_SENSOR_PLUS_BIAS":"NONPURE",
 "NONPURE_CART_MASS_SHIFT":"NONPURE"
}

def wrap(x):
    return (x+np.pi)%(2*np.pi)-np.pi

class IDP:
    def __init__(self,seed=0,mass_mult=1.0):
        import gymnasium as gym, mujoco
        self.mujoco=mujoco
        self.env=gym.make('InvertedDoublePendulum-v5')
        self.env.reset(seed=seed)
        self.model=self.env.unwrapped.model
        self.data=self.env.unwrapped.data
        slide=int(mujoco.mjtJoint.mjJNT_SLIDE)
        hinge=int(mujoco.mjtJoint.mjJNT_HINGE)
        self.slide=None; hs=[]
        for jid,t in enumerate(self.model.jnt_type):
            if int(t)==slide and self.slide is None: self.slide=jid
            if int(t)==hinge: hs.append(jid)
        assert self.slide is not None and len(hs)>=2
        self.hs=hs[:2]
        self.cart_q=int(self.model.jnt_qposadr[self.slide])
        self.cart_d=int(self.model.jnt_dofadr[self.slide])
        self.hq=[int(self.model.jnt_qposadr[j]) for j in self.hs]
        self.hd=[int(self.model.jnt_dofadr[j]) for j in self.hs]
        self.act=None
        for aid in range(self.model.nu):
            if int(self.model.actuator_trnid[aid,0])==self.slide:
                self.act=aid; break
        assert self.act is not None
        self.ctrlrange=np.asarray(self.model.actuator_ctrlrange[self.act],float)
        self.qpos0=self.model.qpos0.copy()
        if mass_mult!=1.0:
            body=int(self.model.jnt_bodyid[self.slide])
            self.model.body_mass[body]*=float(mass_mult)
            mujoco.mj_setConst(self.model,self.data)

    def set_state(self,cart,hp,cv,hv):
        self.data.qpos[:]=self.qpos0; self.data.qvel[:]=0
        self.data.qpos[self.cart_q]=self.qpos0[self.cart_q]+float(cart)
        for q,v in zip(self.hq,hp): self.data.qpos[q]=self.qpos0[q]+float(v)
        self.data.qvel[self.cart_d]=float(cv)
        for d,v in zip(self.hd,hv): self.data.qvel[d]=float(v)
        self.data.ctrl[:]=0
        self.mujoco.mj_forward(self.model,self.data)

    def obs(self,sensor_bias=0.0):
        return np.array([
            self.data.qpos[self.cart_q]-self.qpos0[self.cart_q]+float(sensor_bias),
            wrap(self.data.qpos[self.hq[0]]-self.qpos0[self.hq[0]]),
            wrap(self.data.qpos[self.hq[1]]-self.qpos0[self.hq[1]]),
            self.data.qvel[self.cart_d],self.data.qvel[self.hd[0]],self.data.qvel[self.hd[1]]
        ],float)

    def step_actual(self,actual):
        self.data.ctrl[:]=0
        self.data.ctrl[self.act]=float(actual)
        self.mujoco.mj_step(self.model,self.data)

    def close(self): self.env.close()

def feature(x,u):
    x=np.asarray(x,float)
    return np.array([
        x[0],math.sin(x[1]),math.cos(x[1]),math.sin(x[2]),math.cos(x[2]),
        x[3],x[4],x[5],float(u)
    ],np.float32)

def target_delta(x,y):
    d=np.asarray(y,float)-np.asarray(x,float)
    d[1]=wrap(d[1]); d[2]=wrap(d[2])
    return d.astype(np.float32)

class WM:
    def __init__(self,indim,outdim):
        import torch, torch.nn as nn
        self.torch=torch
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net=nn.Sequential(
            nn.Linear(indim,128),nn.SiLU(),
            nn.Linear(128,128),nn.SiLU(),
            nn.Linear(128,128),nn.SiLU(),
            nn.Linear(128,outdim)
        ).float().to(self.device)

    def load(self,b):
        self.net.load_state_dict(b['state_dict'])
        self.fmu=np.asarray(b['fmu'],np.float32); self.fsd=np.asarray(b['fsd'],np.float32)
        self.rmu=np.asarray(b['rmu'],np.float32); self.rsd=np.asarray(b['rsd'],np.float32)
        self.net.eval(); return self

    def pred(self,F):
        torch=self.torch
        F=np.asarray(F,np.float32)
        with torch.no_grad():
            X=torch.tensor((F-self.fmu)/self.fsd,dtype=torch.float32,device=self.device)
            p=self.net(X).cpu().numpy()
        return p*self.rsd+self.rmu

def nominal_stats(wm,Fc,Rc):
    e=Rc-wm.pred(Fc)
    return e.mean(0),e.std(0)+1e-6

def make_group_params(cfg,rng):
    r=cfg['scenario_family']['parameter_ranges']
    sign=float(rng.choice([-1.,1.]))
    c=sign*float(rng.uniform(*r['cart_offset_abs']))
    usign=float(rng.choice([-1.,1.]))
    u1=usign*float(rng.uniform(*r['u1_abs']))
    du=usign*float(rng.uniform(*r['delta_u_abs']))
    gain=float(rng.uniform(*r['gain_multiplier']))
    bias=(gain-1.0)*u1
    return {
      'c':c,'u1':u1,'du':du,'u2':u1+du,'gain':gain,'bias':bias,
      'mass_mult':float(rng.uniform(*r['cart_mass_multiplier'])),
      'cart':float(rng.uniform(*r['cart_position'])),
      'cv':float(rng.uniform(*r['cart_velocity'])),
      'hp':rng.uniform(r['hinge_pose'][0],r['hinge_pose'][1],2),
      'hv':rng.uniform(r['hinge_velocity'][0],r['hinge_velocity'][1],2)
    }

def class_spec(state,p):
    if state=="PURE_HOME_POSITION_ERROR":
        return dict(cart=p['c'],sensor=p['c']*0,gain=1.0,bias=0.0,mass=1.0,ref=p['c'])
    if state=="PURE_SENSOR_ZERO_OFFSET":
        return dict(cart=0.0,sensor=p['c'],gain=1.0,bias=0.0,mass=1.0,ref=0.0)
    if state=="PURE_ACTUATOR_GAIN_ERROR":
        return dict(cart=0.0,sensor=0.0,gain=p['gain'],bias=0.0,mass=1.0,ref=0.0)
    if state=="PURE_ACTUATOR_ADDITIVE_BIAS":
        return dict(cart=0.0,sensor=0.0,gain=1.0,bias=p['bias'],mass=1.0,ref=0.0)
    if state=="COMPOSITE_HOME_PLUS_GAIN":
        return dict(cart=p['c'],sensor=0.0,gain=p['gain'],bias=0.0,mass=1.0,ref=p['c'])
    if state=="COMPOSITE_SENSOR_PLUS_BIAS":
        return dict(cart=0.0,sensor=p['c'],gain=1.0,bias=p['bias'],mass=1.0,ref=0.0)
    if state=="NONPURE_CART_MASS_SHIFT":
        return dict(cart=0.0,sensor=0.0,gain=1.0,bias=0.0,mass=p['mass_mult'],ref=0.0)
    raise KeyError(state)

def generate_raw(cfg):
    sc=cfg['scenario_family']; rng=np.random.default_rng(sc['scenario_root'])
    rows=[]; invalid=0
    for gi in range(sc['groups']):
        p=make_group_params(cfg,rng)
        for state in STATES:
            s=class_spec(state,p)
            env=IDP(sc['scenario_root']+gi*100+SID[state],mass_mult=s['mass'])
            scale=max(abs(env.ctrlrange[0]),abs(env.ctrlrange[1]))
            actuals=[]
            FF=[]; RR=[]
            for u in [p['u1'],p['u2']]:
                actual=s['gain']*u+s['bias']
                actuals.append(actual)
                env.set_state(s['cart'],p['hp'],p['cv'],p['hv'])
                x=env.obs(s['sensor'])
                env.step_actual(actual)
                y=env.obs(s['sensor'])
                FF.append(feature(x,u)); RR.append(target_delta(x,y))
            frac=max(abs(x)/scale for x in actuals)
            if frac>sc['parameter_ranges']['max_faulty_control_fraction']+1e-12:
                invalid+=1
            rows.append({
              'group':gi,'state':state,'label':SID[state],
              'F':np.asarray(FF,np.float32),'R':np.asarray(RR,np.float32),
              'du':p['du'],'ref':float(s['ref'])
            })
            env.close()
    return rows,invalid

def residualize(wm,mu,sd,raw):
    out=[]
    for r in raw:
        e=np.asarray(r['R'],float)-wm.pred(r['F'])
        z=(e-mu)/sd
        passive=z[0].copy()
        slope=(z[1]-z[0])/float(r['du'])
        out.append({
          'group':r['group'],'state':r['state'],'label':r['label'],
          'family':FAMILY[r['state']],
          'PASSIVE':passive,
          'STATIC':np.r_[passive,r['ref']],
          'DYNAMIC':np.r_[passive,slope],
          'FULL':np.r_[passive,r['ref'],slope]
        })
    return out

def rf(cfg):
    from sklearn.ensemble import RandomForestClassifier
    p=cfg['policy']['rf']
    return RandomForestClassifier(
      n_estimators=p['n_estimators'],min_samples_leaf=p['min_samples_leaf'],
      max_features=p['max_features'],class_weight=p['class_weight'],
      random_state=p['random_state'],n_jobs=-1
    )

def cp_upper(x,n,alpha=0.05):
    from scipy.stats import beta
    x=int(x); n=int(n)
    if n<=0: return 1.0
    if x>=n: return 1.0
    return float(beta.ppf(1-alpha,x+1,n-x))

def prediction_table(model,rows,key):
    X=np.asarray([r[key] for r in rows])
    proba=model.predict_proba(X)
    classes=model.classes_.astype(int)
    j=np.argmax(proba,axis=1)
    pred=classes[j]
    conf=proba[np.arange(len(rows)),j]
    return pred,conf

def threshold_stats(rows,pred,conf,tau):
    truth=np.asarray([r['label'] for r in rows],int)
    pure_true=np.isin(truth,[SID[s] for s in PURE])
    pure_pred=np.isin(pred,[SID[s] for s in PURE])
    commit=pure_pred&(conf>=tau)

    pure_commit=commit&pure_true
    ncommit=int(np.sum(pure_commit))
    wrong=int(np.sum(pure_commit&(pred!=truth)))
    nonpure=~pure_true
    nonpure_n=int(np.sum(nonpure))
    nonpure_commit=int(np.sum(commit&nonpure))
    pure_n=int(np.sum(pure_true))
    return {
      'wrong':wrong,'ncommit':ncommit,
      'wrong_ucb':cp_upper(wrong,ncommit),
      'nonpure_commit':nonpure_commit,'nonpure_n':nonpure_n,
      'nonpure_ucb':cp_upper(nonpure_commit,nonpure_n),
      'coverage':float(ncommit/max(pure_n,1))
    }

def calibrate_tau(model,rows,key,source_wms,maxrisk=0.05):
    pred,conf=prediction_table(model,rows,key)
    candidates=sorted(set([0.0]+[float(x) for x in conf]+[1.000001]))
    feasible=[]
    for tau in candidates:
        pooled=threshold_stats(rows,pred,conf,tau)
        if pooled['wrong_ucb']>maxrisk or pooled['nonpure_ucb']>maxrisk:
            continue
        per=[]
        ok=True
        for wm in source_wms:
            idx=[i for i,r in enumerate(rows) if r['wm']==wm]
            sub=[rows[i] for i in idx]
            st=threshold_stats(sub,pred[idx],conf[idx],tau)
            per.append(st)
            if st['wrong_ucb']>maxrisk or st['nonpure_ucb']>maxrisk:
                ok=False; break
        if ok:
            mincov=min(x['coverage'] for x in per)
            feasible.append((mincov,pooled['coverage'],-tau,tau,pooled,per))
    if not feasible:
        return {'enabled':False,'tau':1.000001}
    feasible.sort(reverse=True,key=lambda x:(x[0],x[1],x[2]))
    b=feasible[0]
    return {'enabled':b[3]<=1.0,'tau':float(b[3]),'min_source_coverage':b[0],
            'pooled_coverage':b[1],'pooled_calibration':b[4]}

def commit_from(model,row,key,cal):
    if not cal['enabled']: return None
    p=model.predict_proba(np.asarray(row[key])[None,:])[0]
    classes=model.classes_.astype(int)
    j=int(np.argmax(p)); pred=int(classes[j]); conf=float(p[j])
    if pred in [SID[s] for s in PURE] and conf>=cal['tau']:
        return pred
    return None

def evaluate_policy(rows,family_model,models,cals,cfg,random_mode=False,held=0):
    truth=[]; pred=[]; costs=[]
    rng=np.random.default_rng(cfg['random_baseline_seed']+int(held))
    costs_map=cfg['evidence']['costs']
    for r in rows:
        if random_mode:
            first="STATIC" if int(rng.integers(0,2))==0 else "DYNAMIC"
        else:
            fam=family_model.predict(np.asarray(r['PASSIVE'])[None,:])[0]
            first=fam if fam in ("STATIC","DYNAMIC") else "FULL"

        out=None
        if first in ("STATIC","DYNAMIC"):
            out=commit_from(models[first],r,first,cals[first])
            if out is not None:
                cost=costs_map[first]
            else:
                out=commit_from(models['FULL'],r,'FULL',cals['FULL'])
                cost=costs_map['FULL']
        else:
            out=commit_from(models['FULL'],r,'FULL',cals['FULL'])
            cost=costs_map['FULL']

        truth.append(r['label']); pred.append(-1 if out is None else int(out)); costs.append(float(cost))

    truth=np.asarray(truth,int); pred=np.asarray(pred,int); costs=np.asarray(costs,float)
    pure=np.isin(truth,[SID[s] for s in PURE]); nonpure=~pure
    commit=pred>=0
    pure_commit=commit&pure
    wrong=int(np.sum(pure_commit&(pred!=truth)))
    ncommit=int(np.sum(pure_commit))
    coverage=float(ncommit/max(int(np.sum(pure)),1))
    correct=float(np.sum(pure_commit&(pred==truth))/max(int(np.sum(pure)),1))
    risk=float(wrong/max(ncommit,1))
    unsafe=float(np.sum(commit&nonpure)/max(int(np.sum(nonpure)),1))
    return {
      'wrong_intervention_risk':risk,
      'coverage':coverage,
      'correct_intervention_rate':correct,
      'declared_nonpure_pure_commit':unsafe,
      'declared_nonpure_safe':1.0-unsafe,
      'mean_probe_cost':float(np.mean(costs))
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--v02dir',required=True)
    ap.add_argument('--outdir',required=True)
    a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    V=Path(a.v02dir); O=Path(a.outdir); O.mkdir(parents=True,exist_ok=True)

    assert cfg['hidden_unseal_authorized'] is False
    forbidden=set(cfg['hidden_seeds_forbidden']['world_model_seeds'])|set(cfg['hidden_seeds_forbidden']['scenario_roots'])
    used=set(cfg['world_models']['development_seeds'])|{
      cfg['scenario_family']['scenario_root'],cfg['scenario_family']['split_seed'],cfg['random_baseline_seed']
    }
    assert not (forbidden&used)

    import torch
    nz=np.load(V/'nominal_data.npz')
    Fc,Rc=nz['Fc'],nz['Rc']
    v02=json.loads((V/'MEC_CROSSFAMILY_IDP_WM_THRESHOLD_DEV_V0_2_RESULT.json').read_text())
    assert v02['summary']['quality_models_passed']==cfg['world_models']['required_quality_models']
    assert v02['hidden_unsealed'] is False and v02['hidden_seeds_used'] is False

    raw_path=O/'broad_raw.pkl'
    if raw_path.exists():
        raw,invalid=pickle.loads(raw_path.read_bytes())
    else:
        print('generating fresh broad scenarios',flush=True)
        raw,invalid=generate_raw(cfg)
        raw_path.write_bytes(pickle.dumps((raw,invalid)))

    rng=np.random.default_rng(cfg['scenario_family']['split_seed'])
    ids=np.arange(cfg['scenario_family']['groups']); rng.shuffle(ids)
    ntr=cfg['scenario_family']['train_groups']; ncal=cfg['scenario_family']['calibration_groups']
    train=set(map(int,ids[:ntr]))
    cal=set(map(int,ids[ntr:ntr+ncal]))
    test=set(map(int,ids[ntr+ncal:]))

    records={}
    for seed in cfg['world_models']['development_seeds']:
        b=torch.load(V/f'wm_{seed}.pt',map_location='cpu',weights_only=False)
        wm=WM(Fc.shape[1],Rc.shape[1]).load(b)
        mu,sd=nominal_stats(wm,Fc,Rc)
        rr=residualize(wm,mu,sd,raw)
        for r in rr: r['wm']=seed
        records[seed]=rr

    folds=[]
    m=cfg['metrics']
    for held in cfg['world_models']['development_seeds']:
        source=[s for s in cfg['world_models']['development_seeds'] if s!=held]

        trainrows=[r for s in source for r in records[s] if r['group'] in train]
        calrows=[r for s in source for r in records[s] if r['group'] in cal]
        testrows=[r for r in records[held] if r['group'] in test]

        # Family selector.
        fam_model=rf(cfg)
        X=np.asarray([r['PASSIVE'] for r in trainrows])
        y=np.asarray([r['family'] for r in trainrows])
        fam_model.fit(X,y)

        models={}; cals={}
        for key in ['STATIC','DYNAMIC','FULL']:
            model=rf(cfg)
            model.fit(np.asarray([r[key] for r in trainrows]),np.asarray([r['label'] for r in trainrows]))
            models[key]=model
            cals[key]=calibrate_tau(
                model,calrows,key,source,
                maxrisk=cfg['policy']['wrong_pure_commit_ucb_max']
            )

        mec=evaluate_policy(testrows,fam_model,models,cals,cfg,False,held)
        rnd=evaluate_policy(testrows,fam_model,models,cals,cfg,True,held)
        full_cost=cfg['evidence']['costs']['FULL']
        ratio=float(mec['mean_probe_cost']/rnd['mean_probe_cost'])

        primary=(
          cals['FULL']['enabled'] and
          mec['wrong_intervention_risk']<=m['wrong_intervention_risk_max'] and
          mec['coverage']>=m['pure_coverage_min'] and
          mec['declared_nonpure_pure_commit']<=m['declared_nonpure_pure_commit_max'] and
          mec['declared_nonpure_safe']>=m['declared_nonpure_safe_min']
        )
        eff=ratio<=m['mec_to_random_probe_cost_ratio_max']
        folds.append({
          'heldout_wm':held,'calibration':cals,'MEC':mec,'RANDOM':rnd,
          'FULL_EVIDENCE_mean_probe_cost':full_cost,'cost_ratio':ratio,
          'primary_pass':bool(primary),'efficiency_pass':bool(eff)
        })
        print('fold',held,'primary',primary,'eff',eff,'ratio',ratio,flush=True)

    def median_metric(policy,key):
        return float(np.median([f[policy][key] for f in folds]))

    med={
      'wrong_intervention_risk':median_metric('MEC','wrong_intervention_risk'),
      'coverage':median_metric('MEC','coverage'),
      'correct_intervention_rate':median_metric('MEC','correct_intervention_rate'),
      'declared_nonpure_pure_commit':median_metric('MEC','declared_nonpure_pure_commit'),
      'declared_nonpure_safe':median_metric('MEC','declared_nonpure_safe'),
      'mean_probe_cost':median_metric('MEC','mean_probe_cost'),
      'random_mean_probe_cost':median_metric('RANDOM','mean_probe_cost'),
      'cost_ratio':float(np.median([f['cost_ratio'] for f in folds]))
    }
    median_primary=(
      med['wrong_intervention_risk']<=m['wrong_intervention_risk_max'] and
      med['coverage']>=m['pure_coverage_min'] and
      med['declared_nonpure_pure_commit']<=m['declared_nonpure_pure_commit_max'] and
      med['declared_nonpure_safe']>=m['declared_nonpure_safe_min']
    )
    primary_folds=sum(f['primary_pass'] for f in folds)
    eff_folds=sum(f['efficiency_pass'] for f in folds)
    primary_qualified=primary_folds>=cfg['qualification']['primary_fold_passes_required'] and median_primary
    eff_qualified=eff_folds>=cfg['qualification']['efficiency_fold_passes_required'] and med['cost_ratio']<=m['mec_to_random_probe_cost_ratio_max']

    if primary_qualified and eff_qualified:
        decision='BROAD_DEV_PRIMARY_EFFICIENCY_PASS'
    elif primary_qualified:
        decision='BROAD_DEV_PRIMARY_PASS_EFFICIENCY_FAIL'
    else:
        decision='BROAD_DEV_PRIMARY_FAIL'

    result={
      'protocol_id':cfg['protocol_id'],'decision':decision,
      'hidden_unsealed':False,'hidden_seeds_used':False,'world_models_retrained':False,
      'invalid_scenarios':invalid,'folds':folds,
      'summary':{
        'primary_fold_passes':int(primary_folds),
        'efficiency_fold_passes':int(eff_folds),
        'median_primary_pass':bool(median_primary),
        'median':med
      }
    }
    (O/'MEC_IDP_BROAD_SELECTIVE_INTERVENTION_DEV_V0_4C_RESULT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'decision':decision,'summary':result['summary']},indent=2))

if __name__=='__main__':
    main()
