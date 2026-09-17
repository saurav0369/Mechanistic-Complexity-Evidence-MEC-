from __future__ import annotations
import argparse, json, math, hashlib, sys, time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import mec_e2e_v1_runner as core


def save_json(p, x): p.write_text(json.dumps(x, indent=2))
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()




class FastResidualWM:
    """Same frozen 3x128 SiLU architecture; float32 and CUDA when available (engineering-only acceleration)."""
    def __init__(self,input_dim,output_dim,hidden=128,layers=3):
        import torch, torch.nn as nn
        self.torch=torch; self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mods=[nn.Linear(input_dim,hidden),nn.SiLU()]
        for _ in range(layers-1): mods += [nn.Linear(hidden,hidden),nn.SiLU()]
        mods += [nn.Linear(hidden,output_dim)]
        self.net=nn.Sequential(*mods).float().to(self.device)
        self.fmu=self.fsd=self.rmu=self.rsd=None
    def fit(self,F,R,epochs,batch,lr,wd,seed):
        torch=self.torch; core.seed_all(seed)
        self.fmu=F.mean(0); self.fsd=F.std(0)+1e-8; self.rmu=R.mean(0); self.rsd=R.std(0)+1e-8
        X=torch.tensor((F-self.fmu)/self.fsd,dtype=torch.float32)
        Y=torch.tensor((R-self.rmu)/self.rsd,dtype=torch.float32)
        opt=torch.optim.AdamW(self.net.parameters(),lr=lr,weight_decay=wd)
        g=torch.Generator(device='cpu'); g.manual_seed(seed)
        for _ in range(epochs):
            order=torch.randperm(len(X),generator=g)
            for st in range(0,len(X),batch):
                j=order[st:st+batch]
                xb=X[j].to(self.device,non_blocking=True); yb=Y[j].to(self.device,non_blocking=True)
                pred=self.net(xb); loss=((pred-yb)**2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return self
    def predict_residual(self,F):
        torch=self.torch
        with torch.no_grad():
            X=torch.tensor((F-self.fmu)/self.fsd,dtype=torch.float32,device=self.device)
            p=self.net(X).cpu().numpy().astype(float)
        return p*self.rsd+self.rmu,p
    def state_dict_bundle(self):
        return {'state_dict':{k:v.detach().cpu() for k,v in self.net.state_dict().items()},
                'fmu':self.fmu,'fsd':self.fsd,'rmu':self.rmu,'rsd':self.rsd,
                'device_used':str(self.device),'dtype':'float32'}


def residual_summary_matrix(res):
    """Fixed episode summary; no labels or simulator params."""
    res=np.asarray(res,float)
    feats=[res.mean(0), res.std(0), np.mean(np.abs(res),0), np.max(np.abs(res),0),
           np.quantile(res,0.25,axis=0), np.quantile(res,0.50,axis=0), np.quantile(res,0.75,axis=0)]
    return np.concatenate(feats)


def predict_episode_residuals(model, X, U, Y, angle_mask):
    F,R=core.make_training_arrays(np.asarray(X),np.asarray(U),np.asarray(Y),angle_mask)
    pr,_=model.predict_residual(F)
    return R-pr


def set_nonhinge_zero(A):
    A.data.qpos[:] = A.model.qpos0
    A.data.qvel[:] = 0


def common_model_setup(A, mass_mult, damping_mult):
    A.restore_model(); A.apply_morphology(mass_mult,damping_mult)


def matched_static_raw_pair(env_id,cfg,seed,steps):
    """Construct two separately evaluated systems with common h/nuisance/action stream.
    Pairing is only dataset construction; diagnostic features never compare the two worlds.
    """
    rng=np.random.default_rng(seed)
    mm=cfg['morphology_and_nuisance_ranges']; fr=cfg['fault_ranges']; dg=cfg['data_generation']
    mass=float(rng.uniform(*mm['hidden_in_hull_mass_multiplier']))
    damp=float(rng.uniform(*mm['hidden_nominal_nuisance_damping_multiplier']))
    mag=float(rng.uniform(*fr['encoder_offset_rad_abs'])); sign=float(rng.choice([-1.,1.]))
    h=np.array([0.,sign*mag])
    qjit=max(abs(x) for x in mm['sensor_uniform_jitter_q_rad']); vjit=max(abs(x) for x in mm['sensor_uniform_jitter_qdot'])

    action_fracs=[]; segs=[]; left=0; frac=None
    # Generate command stream independent of either environment.
    for t in range(steps):
        if left<=0:
            frac=rng.uniform(*dg['action_fraction_of_control_range_hidden'])
            left=int(rng.integers(1,6))
        action_fracs.append(float(frac)); left-=1

    out=[]
    for label,kind in [(0,'ENCODER_OFFSET'),(1,'MECHANICAL_REFERENCE_SHIFT')]:
        A=core.EnvAdapter(env_id,seed+10000+label)
        common_model_setup(A,mass,damp)
        enc=np.zeros(2)
        if kind=='ENCODER_OFFSET': enc=-h
        else: A.apply_mechanical_reference(h)
        A.reset(seed+20000)
        X=[];U=[];Y=[]
        # Same stochastic sensor nuisance sequence for the pair.
        srng=np.random.default_rng(seed+30000)
        for frac in action_fracs:
            u=np.clip(frac*np.maximum(np.abs(A.action_low),np.abs(A.action_high)),A.action_low,A.action_high)
            x=A.observed_state(enc,qjit,vjit,srng); A.step(u); y=A.observed_state(enc,qjit,vjit,srng)
            X.append(x);U.append(u.copy());Y.append(y)
        ucmd=np.array([0.21,-0.17]); R1=np.array([[1.,0.]]); R2=np.eye(2)
        c1,w1=A.static_reference_probe(ucmd,enc,R1); c2,w2=A.static_reference_probe(ucmd,enc,R2)
        out.append({'label':label,'class':kind,'X':np.asarray(X),'U':np.asarray(U),'Y':np.asarray(Y),
                    'angle_mask':A.jmap.angle_mask.copy(),'E1_c':c1,'E1_w':w1,'E2_c':c2,'E2_w':w2})
        A.close()
    return out


def dynamic_raw(env_id,cfg,seed,kind):
    """Single-system diagnostic residual data at the frozen D2/D3 points."""
    rng=np.random.default_rng(seed); mm=cfg['morphology_and_nuisance_ranges']; fr=cfg['fault_ranges']
    A=core.EnvAdapter(env_id,seed+40000)
    common_model_setup(A,float(rng.uniform(*mm['hidden_in_hull_mass_multiplier'])),
                       float(rng.uniform(*mm['hidden_nominal_nuisance_damping_multiplier'])))
    act_bias=None
    if kind=='DYNAMICS_DAMPING_FAULT': A.apply_damping_fault(float(rng.uniform(*fr['damping_multiplier'])))
    elif kind=='ACTUATOR_BIAS':
        m=float(rng.uniform(*fr['actuator_bias_fraction_of_ctrl_range_abs'])); s=float(rng.choice([-1.,1.]))
        act_bias=np.full(A.action_low.shape,s*m*np.maximum(np.abs(A.action_low),np.abs(A.action_high)))
    else: raise ValueError(kind)

    if env_id.startswith('Reacher'):
        spec=fr['dynamic_threshold_points_reacher']; pose=np.asarray(spec['pose_q']); pts=list(spec['velocity_points_D2'])+[spec['velocity_point_added_D3']]; ctrl=np.asarray(spec['control'])
    else:
        spec=fr['dynamic_threshold_points_inverted_double_pendulum']; pose=np.asarray(spec['hinge_pose_q']); pts=list(spec['hinge_velocity_points_D2'])+[spec['hinge_velocity_point_added_D3']]; ctrl=np.asarray(spec['control'])

    rows=[]
    for v in pts:
        A.set_diagnostic_state(pose,np.asarray(v,float))
        x=A.observed_state(None,0,0,None); F=core.feature_map(x,ctrl,A.jmap.angle_mask)[None,:]
        A.step(ctrl,act_bias); y=A.observed_state(None,0,0,None)
        R=core.residual_target(x,y,A.jmap.angle_mask); pr,_=CURRENT_MODEL.predict_residual(F)
        rows.append(R-pr[0])
    am=A.jmap.angle_mask.copy(); A.close()
    return np.asarray(rows),am


def fit_eval_binary(X,y,groups,seed):
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score,accuracy_score
    X=np.asarray(X,float); y=np.asarray(y,int); groups=np.asarray(groups)
    sp=GroupShuffleSplit(n_splits=1,test_size=.35,random_state=seed)
    tr,te=next(sp.split(X,y,groups))
    out={}
    models={
      'rf':RandomForestClassifier(n_estimators=400,min_samples_leaf=3,class_weight='balanced',random_state=seed),
      'svm':SVC(C=10,kernel='rbf',probability=True,class_weight='balanced',random_state=seed)
    }
    for name,m in models.items():
        m.fit(X[tr],y[tr]); p=m.predict_proba(X[te])[:,1]; pred=(p>=.5).astype(int)
        auc=float(roc_auc_score(y[te],p)); auc=max(auc,1-auc) # best orientation; leakage test
        out[name]={'auc':auc,'accuracy':float(accuracy_score(y[te],pred)),'n_train':len(tr),'n_test':len(te)}
    return out


def calibration_stats(model,X,U,Y,am):
    res=predict_episode_residuals(model,X,U,Y,am)
    energy=np.sum(res*res,axis=1)
    return {'residual_mean':res.mean(0).tolist(),'residual_std':res.std(0).tolist(),
            'energy_q90':float(np.quantile(energy,.90)),'energy_q95':float(np.quantile(energy,.95)),
            'energy_q99':float(np.quantile(energy,.99))}


def train_models_for_env(env_cfg,cfg,out,smoke=False):
    env_id=env_cfg['id']; wm_cfg=cfg['world_models']
    ntr=env_cfg['train_transitions']; ncal=env_cfg['calibration_transitions']; nte=env_cfg['nominal_test_transitions']
    print(f'[{env_id}] collecting nominal train/cal/test',flush=True)
    X,U,Y,am=core.collect_nominal(env_id,cfg,ntr,cfg['split_seeds']['nominal_train'],'train')
    Xc,Uc,Yc,amc=core.collect_nominal(env_id,cfg,ncal,cfg['split_seeds']['calibration'],'cal')
    Xt,Ut,Yt,amt=core.collect_nominal(env_id,cfg,nte,cfg['split_seeds']['calibration']+99,'test')
    if not (np.array_equal(am,amc) and np.array_equal(am,amt)): raise RuntimeError('angle mask changed')
    F,R=core.make_training_arrays(X,U,Y,am)
    models=[]; qual=[]; cal=[]
    for i,seed in enumerate(wm_cfg['seeds']):
        print(f'[{env_id}] train model {i+1}/8 seed={seed}',flush=True)
        m=FastResidualWM(F.shape[1],R.shape[1],128,3).fit(F,R,wm_cfg['epochs'],wm_cfg['batch_size'],wm_cfg['learning_rate'],wm_cfg['weight_decay'],seed)
        q=core.evaluate_model(m,Xt,Ut,Yt,am); q['seed']=seed; qual.append(q); cal.append(calibration_stats(m,Xc,Uc,Yc,am)); models.append(m)
        try:
            import torch; torch.save(m.state_dict_bundle(),out/f"{env_id.replace('-','_')}_wm_seed{seed}.pt")
        except Exception: pass
    return models,qual,cal


def dev_threshold_audits(env_id,cfg,models,seed):
    n=cfg['data_generation']['development_fault_episodes_per_class_per_environment']
    steps=cfg['hidden_test']['trajectory_length_reacher'] if env_id.startswith('Reacher') else cfg['hidden_test']['trajectory_length_inverted_double_pendulum']
    report={'static_by_model':[],'dynamic_by_model':[],'n_pairs':n}
    # Generate static raw pairs once; then each model gets its own residual features.
    print(f'[{env_id}] generating {n} matched static development pairs',flush=True)
    static_pairs=[matched_static_raw_pair(env_id,cfg,seed+1000+i,steps) for i in range(n)]
    for mi,m in enumerate(models):
        X0=[];X1=[];X2=[];y=[];g=[]
        for i,pair in enumerate(static_pairs):
            for row in pair:
                rs=predict_episode_residuals(m,row['X'],row['U'],row['Y'],row['angle_mask']); e0=residual_summary_matrix(rs)
                X0.append(e0); X1.append(np.r_[e0,row['E1_c'],row['E1_w']]); X2.append(np.r_[e0,row['E2_c'],row['E2_w']]); y.append(row['label']); g.append(i)
        r={'model_seed':cfg['world_models']['seeds'][mi],
           'E0':fit_eval_binary(X0,y,g,seed+mi),'E1':fit_eval_binary(X1,y,g,seed+100+mi),'E2':fit_eval_binary(X2,y,g,seed+200+mi)}
        report['static_by_model'].append(r)

    # Dynamic raw generation depends on model because residual = learned-model error.
    print(f'[{env_id}] generating dynamic development audits',flush=True)
    global CURRENT_MODEL
    for mi,m in enumerate(models):
        CURRENT_MODEL=m; X2=[];X3=[];y=[];g=[]
        for i in range(n):
            for lab,kind in [(0,'ACTUATOR_BIAS'),(1,'DYNAMICS_DAMPING_FAULT')]:
                rows,_=dynamic_raw(env_id,cfg,seed+50000+i,kind)
                X2.append(rows[:2].ravel()); X3.append(rows[:3].ravel()); y.append(lab); g.append(i)
        report['dynamic_by_model'].append({'model_seed':cfg['world_models']['seeds'][mi],
                                           'D2':fit_eval_binary(X2,y,g,seed+300+mi),
                                           'D3':fit_eval_binary(X3,y,g,seed+400+mi)})
    return report


def summarize(cfg,allrep):
    qg=cfg['world_models']['quality_gate']; summary={}
    for env_id,r in allrep.items():
        qual=r['model_quality']
        passed=[q['rmse_ratio']<=qg['aggregate_state_rmse_vs_persistence_ratio_max'] and q['normalized_residual_target_mse']<=qg['normalized_residual_target_mse_max'] for q in qual]
        p0=sum(passed)>=qg['models_required']
        static=r['development_threshold_audit']['static_by_model']; dynamic=r['development_threshold_audit']['dynamic_by_model']
        best_e0=[max(x['E0']['rf']['auc'],x['E0']['svm']['auc']) for x in static]
        best_e1=[max(x['E1']['rf']['auc'],x['E1']['svm']['auc']) for x in static]
        best_e2=[max(x['E2']['rf']['auc'],x['E2']['svm']['auc']) for x in static]
        best_d2=[max(x['D2']['rf']['auc'],x['D2']['svm']['auc']) for x in dynamic]
        best_d3=[max(x['D3']['rf']['auc'],x['D3']['svm']['auc']) for x in dynamic]
        summary[env_id]={
          'P0_dev_proxy':p0,'quality_models_passed':int(sum(passed)),
          'static_best_auc_median':{'E0':float(np.median(best_e0)),'E1':float(np.median(best_e1)),'E2':float(np.median(best_e2))},
          'dynamic_best_auc_median':{'D2':float(np.median(best_d2)),'D3':float(np.median(best_d3))},
          'development_warning_static_below_threshold_leaks':bool(np.median(best_e0)>.65 or np.median(best_e1)>.65),
          'development_warning_dynamic_D2_leaks':bool(np.median(best_d2)>.65),
        }
    return summary


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--out',required=True); ap.add_argument('--env',default=None)
    a=ap.parse_args(); cfg=json.loads(Path(a.config).read_text()); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if any(s in cfg['split_seeds']['hidden_test'] for s in [cfg['split_seeds']['nominal_train'],cfg['split_seeds']['calibration'],cfg['data_generation']['development_seed']]):
        raise RuntimeError('development seed collides with hidden seed')
    report={'protocol_id':cfg['protocol_id'],'phase':'FULL_DEVELOPMENT_ONLY','hidden_unsealed':False,
            'config_sha256':sha256(a.config),'dev_runner_sha256':sha256(__file__),'environments':{}}
    envs=[e for e in cfg['environments'] if a.env in (None,e['id'])]
    for ec in envs:
        models,qual,cal=train_models_for_env(ec,cfg,out)
        audit=dev_threshold_audits(ec['id'],cfg,models,cfg['data_generation']['development_seed'])
        report['environments'][ec['id']]={'model_quality':qual,'calibration':cal,'development_threshold_audit':audit}
        save_json(out/'MEC_E2E_V1_DEV_RESULT_PARTIAL.json',report)
    report['summary']=summarize(cfg,report['environments'])
    save_json(out/'MEC_E2E_V1_FULL_DEV_RESULT.json',report)
    print(json.dumps({'phase':report['phase'],'hidden_unsealed':False,'summary':report['summary']},indent=2))

if __name__=='__main__': main()
